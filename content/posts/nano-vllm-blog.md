---
title: "从零写一个 nano-vLLM：用 Qwen2.5-0.5B 做的小推理引擎实践记录"
date: 2025-12-05T20:19:00+08:00
---

<!--more-->
---


> 这篇文章是我最近做的一次“小手工”：  
> 用一个 0.5B 的 Qwen2.5 Instruct 模型，自己搭了一个 **简化版 vLLM 推理循环**，  
> 一边踩坑一边做了一点点性能优化，最后还写了一个简单的 **fused MLP CUDA kernel**。

代码大致分 3 个阶段：

- **Stage A：最朴素的 HuggingFace 推理循环**（`nano_infer_stageA.py`）
- **Stage B：把 decode 单步抽象出来 + SDPA + 预分配张量 + warmup**（`stageB.py`）
- **Stage B（fused）：替换 Qwen 模型里的 MLP 为自定义 fused CUDA kernel**（`stageB_fused.py` + `fused_mlp*`）

下面按顺序把整个过程整理一下，希望对理解 vLLM 类推理引擎里“预填充 + 增量解码 + 内核融合”这套思路有点帮助。

---

## 1. 目标 & 场景设定

- **模型**：`Qwen2.5-0.5B-Instruct`
- **框架**：PyTorch + HuggingFace Transformers
- **硬件**：单卡 GPU（示例代码里假定 `cuda` 可用）
- **任务**：给一个中文问题：  
  > “用要点概述：如何从零实现一个迷你版 vLLM 推理循环？”

  然后自己 **显式实现**：
  - Prefill 阶段：把整段 prompt 喂给模型，得到最后一个 token 的 logits + KV cache
  - Decode 阶段：单步解码，每次只喂一个 token，复用 KV cache
  - 自己计时，算出 prefill 和 decode 的耗时 / 吞吐量

整体结构可以理解为：

```text
[Load Model] -> [Build Prompt] -> [Prefill] -> [Decode Loop] -> [统计性能]
```

---

## 2. Stage A：最朴素版 nano-vLLM 推理循环

文件：`nano_infer_stageA.py`

这一版的目标很简单：**先让“手写 vLLM 推理循环”跑起来**，  
只要能完成下面四件事就够了：

1. 加载 tokenizer 和模型（半精度、指定设备）
2. 把聊天消息转换成模型能理解的 prompt
3. Prefill 一次，拿到 KV cache 和最后一个 token 的 logits
4. 用一个 `for` 循环做增量解码，统计时间

### 2.1 模型 & tokenizer 加载

```python
MODEL_ID = "/home/dyz/nano-vllm-demo/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda"
DTYPE =  torch.float16

def load_model_and_tokenizer(model_id: str):
    t0 = time.time()
    model_dir = os.path.dirname(model_id) if os.path.isfile(model_id) else model_id

    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=DTYPE,
        device_map={"": DEVICE},
    ).eval()

    t1 = time.time()
    load_time = t1 - t0
    return model, tok, load_time
```

要点：

- 强制 `pad_token` 存在，否则部分模型会在 batch 时出问题。
- `dtype=DTYPE` 设为 `float16`，`device_map` 指到 `cuda`。
- 返回 `load_time`，方便后面打印整体性能数据。

### 2.2 简单的 Chat 模板

```python
def chat_template(tokenizer, user_text: str) -> str:
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_text},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # fallback
        return (
            f"<|im_start|>system
You are a helpful assistant.<|im_end|>
"
            f"<|im_start|>user
{user_text}<|im_end|>
<|im_start|>assistant
"
        )
```

很多新版模型都支持 `apply_chat_template`，  先尝试用官方模板，不行再 fallback 到手写格式。

### 2.3 Prefill + Decode 核心循环

核心函数是 `generate_one`：

```python
@torch.inference_mode()
def generate_one(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_k: int = 50,
):
    # === Prefill 阶段 ===
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    t_prefill_start = time.time()

    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)

    t_prefill_end = time.time()

    past_key_values = out.past_key_values
    next_token_logits = out.logits[:, -1, :]

    # === Decode 阶段 ===
    generated = []
    eos_id = tokenizer.eos_token_id
    t_decode_start = time.time()

    for step in range(max_new_tokens):
        # 贪心 / 采样
        if temperature > 0.0:
            logits = next_token_logits / max(temperature, 1e-6)
            if top_k > 0:
                topk = torch.topk(logits, k=min(top_k, logits.size(-1)))
                mask = torch.full_like(logits, float("-inf"))
                mask.scatter_(1, topk.indices, topk.values)
                logits = mask
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        token_id = int(next_token.squeeze().cpu().item())
        generated.append(token_id)
        if token_id == eos_id:
            break

        # 增量 decode：只喂一个 token + 复用 KV cache
        next_input = torch.tensor([[token_id]], device=DEVICE, dtype=torch.long)
        out = model(input_ids=next_input, use_cache=True, past_key_values=past_key_values)
        past_key_values = out.past_key_values
        next_token_logits = out.logits[:, -1, :]

    t_decode_end = time.time()

    output_text = tokenizer.decode(generated, skip_special_tokens=True)
    perf_info = {
        "prefill_time": t_prefill_end - t_prefill_start,
        "decode_time": t_decode_end - t_decode_start,
        "num_generated_tokens": len(generated),
        "tokens_per_sec": len(generated) / (t_decode_end - t_decode_start + 1e-8),
    }
    return output_text, perf_info
```

这里其实就已经是一个“迷你 vLLM 推理循环”了：

- **Prefill**：一次性把所有 prompt 转成 hidden states，同时得到 `past_key_values`。
- **Decode**：每次只喂 1 个 token，模型内部复用 KV cache，避免重复计算前文。
- 用 `tokens_per_sec` 粗略衡量 decode 吞吐量。

主程序里，我用 `torch.amp.autocast` 包了一层：

```python
with torch.amp.autocast("cuda", dtype=DTYPE):
    text, perf_info = generate_one(...)
```

这一版有几个明显的问题：

- 没有做 **warmup**，第一次跑的时间包含了很多 JIT / kernel 初始化。
- decode 循环里每一步都在 **重新分配张量**（`torch.tensor(...)`）。
- 没显式控制 attention 后端（SDPA / flash / math）。
- `return_dict=True` 有一点小开销（虽然不大）。

Stage B 就是围绕这些点开始优化。

---

## 3. Stage B：把 decode 单步抽出来 + SDPA + 预分配张量

文件：`stageB.py`

这一版的目标是：

- 更像“真正的推理引擎”一点
- 把容易影响性能的部分剥离出来，便于后面做 `torch.compile` 或自定义 kernel

### 3.1 启用 SDPA & matmul 精度

```python
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

torch.set_float32_matmul_precision("high")
```

并在加载模型时指定：

```python
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=DTYPE,
    device_map=DEVICE,
    attn_implementation="sdpa",  # 使用 SDPA 自动选择后端
).eval()
```

这样 decoder 的 self-attention 会走 PyTorch 的 SDPA 路径， 在支持的 GPU 上能用到 flash / mem-efficient 实现。

### 3.2 Decode 单步模块：`DecodeStep`

```python
class DecodeStep(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, token, pkv):
        logits, pkv_out = self.model(
            input_ids=token,
            use_cache=True,
            past_key_values=pkv,
            return_dict=False
        )[:2]
        return logits[:, -1, :], pkv_out
```

好处：

- 把 decode 的“单步”封装成一个 `nn.Module`。
- 后续如果要对 decode 部分 `torch.compile`，只需要对这个 `DecodeStep` 下手。
- `return_dict=False`，直接拿 tuple，略微省一点开销。

### 3.3 Prefill：pin_memory + non_blocking

```python
enc = tokenizer(prompt, return_tensors="pt")
input_ids = enc["input_ids"].pin_memory().to(DEVICE, non_blocking=True)
attention_mask = enc["attention_mask"].pin_memory().to(DEVICE, non_blocking=True)
```

这里做了两件事：

1. `pin_memory()` 把 CPU Tensor 锁页，提升 H2D 拷贝效率；
2. `to(..., non_blocking=True)` 让拷贝可以和后续计算更好地重叠。

在单条推理场景里收益有限，但这是走向“引擎化”的典型步骤。

### 3.4 Decode：预分配 GPU 缓冲区 + 原地写入

```python
generated_gpu = torch.empty((1, max_new_tokens), device=DEVICE, dtype=torch.long)
next_input = torch.empty((1, 1), device=DEVICE, dtype=torch.long)
eos_id = tokenizer.eos_token_id
decode_step = DecodeStep(model)

t_decode_start = time.time()
for step in range(max_new_tokens):
    # same: 贪心/采样逻辑...
    next_input.copy_(next_token)
    generated_gpu[:, step:step + 1].copy_(next_token)

    next_token_logits, past_key_values = decode_step(next_input, past_key_values)
```

对比 Stage A：

- 不再在循环里创建新的张量，而是在 GPU 上 **一次性预分配**：
  - `generated_gpu`：保存所有生成的 token id
  - `next_input`：单 token 输入，循环中原地覆盖
- 循环结束后再一次性搬回 CPU 解码：

```python
generated = generated_gpu.squeeze(0).tolist()
output_text = tokenizer.decode(generated, skip_special_tokens=True)
```

末尾做了一次 EOS 截断：

```python
if eos_id is not None:
    eos_mask = (generated_gpu == eos_id)
    if eos_mask.any():
        first_eos_idx = torch.where(eos_mask)[1].min().item()
        generated_gpu = generated_gpu[:, :first_eos_idx + 1]
```

### 3.5 Warmup：稳定 CUDA 内核和 JIT 行为

```python
@torch.inference_mode()
def warmup_model(model, tokenizer, warmup_steps: int = 5):
    model.eval()
    enc = tokenizer("Hello", return_tensors="pt")
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(DEVICE)

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=False,
    )

    logits, past_key_values = out[0], out[1]
    next_token_logits = logits[:, -1, :]
    next_input = torch.empty((1, 1), device=DEVICE, dtype=torch.long)

    t0 = time.time()
    for _ in range(warmup_steps):
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        next_input.copy_(next_token)
        out = model(
            input_ids=next_input,
            use_cache=True,
            past_key_values=past_key_values,
            return_dict=False,
        )
        logits, past_key_values = out[0], out[1]
        next_token_logits = logits[:, -1, :]
    t1 = time.time()
    print(f"[info] Warm-up completed: {warmup_steps} steps in {t1 - t0:.3f}s")
```

主入口里调用：

```python
try:
    warmup_model(model, tokenizer, warmup_steps=5)
except Exception as e:
    print(f"[warn] Warm-up failed (continuing): {e}")
```

这样测量 prefill / decode 的时间时，就不会混进第一次调用时的 JIT / 内核加载成本。

---

## 4. Stage B（fused）：自定义 Fused MLP CUDA kernel 替换 Qwen MLP

文件：
- `stageB_fused.py`
- `fused_mlp_kernel.cu`
- `fused_mlp.cpp`
- `fused_mlp.py`（`FusedLayerNormMLP`）
- `setup.py`
- `test_fused_mlp.py`

前面 Stage A / B 都还停留在“调 PyTorch 现成 kernel + 一些小工程优化”的层面。  
如果想再往前走一步，就得开始碰 **自定义 CUDA kernel / fused kernel** 了。

### 4.1 为什么挑 MLP 来做融合？

Transformer 的 decoder block 大致是：

```text
x -> LayerNorm -> Self-Attention -> Residual
  -> LayerNorm -> MLP (up_proj + 激活 + down_proj) -> Residual
```

在 decode 阶段，每一步都要走一遍 block，MLP 是其中一块 **算力大、内存访问密集** 的部分：

- 通常是 `hidden_size -> intermediate_size（如 4x） -> hidden_size`
- 一般流程：**LayerNorm -> Linear -> 激活（SiLU/GELU） -> Linear**
- 每一小步都是一个 kernel，对显存读写很多，kernel 启动开销也多

所以很自然的一个优化方向就是：

> 把 LayerNorm + up_proj + 激活 + down_proj **融合成一个 CUDA kernel**

这就是 `fused_ln_mlp_forward` 在做的事情。

### 4.2 CUDA kernel：`fused_ln_mlp_forward_kernel`

简化后的结构：

```cpp
template <typename scalar_t>
__global__ void fused_ln_mlp_forward_kernel(
    const scalar_t* __restrict__ x,      // [B, H]
    const scalar_t* __restrict__ gamma,  // [H]
    const scalar_t* __restrict__ beta,   // [H]
    const scalar_t* __restrict__ w1,     // [I, H]
    const scalar_t* __restrict__ w2,     // [H, I]
    scalar_t* __restrict__ y,            // [B, H]
    int hidden, int intermediate)
{
    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const scalar_t* x_ptr = x + bid * hidden;
    scalar_t* y_ptr = y + bid * hidden;

    // Step 1: LayerNorm —— 求 mean / var
    float mean = 0.f;
    float var = 0.f;
    for (int i = tid; i < hidden; i += blockDim.x) {
        float v = static_cast<float>(x_ptr[i]);
        mean += v;
        var += v * v;
    }

    __shared__ float mean_buf;
    __shared__ float var_buf;
    atomicAdd(&mean_buf, mean);
    atomicAdd(&var_buf, var);
    __syncthreads();

    if (tid == 0) {
        mean_buf /= hidden;
        var_buf = rsqrtf(var_buf / hidden - mean_buf * mean_buf + 1e-5f);
    }
    __syncthreads();

    float mean_final = mean_buf;
    float inv_std = var_buf;

    // Step 2: up_proj + SiLU
    for (int i = tid; i < intermediate; i += blockDim.x) {
        float acc = 0.f;
        for (int j = 0; j < hidden; ++j) {
            float xn = (static_cast<float>(x_ptr[j]) - mean_final) * inv_std;
            float normed = xn * static_cast<float>(gamma[j]) + static_cast<float>(beta[j]);
            acc += normed * static_cast<float>(w1[i * hidden + j]);
        }
        sdata[i] = silu(acc);  // 写到 shared memory
    }
    __syncthreads();

    // Step 3: down_proj
    for (int j = tid; j < hidden; j += blockDim.x) {
        float acc = 0.f;
        for (int i = 0; i < intermediate; ++i) {
            acc += sdata[i] * static_cast<float>(w2[j * intermediate + i]);
        }
        y_ptr[j] = static_cast<scalar_t>(acc);
    }
}
```

可以看到：

1. 一次 kernel 中做了：
   - LayerNorm（mean / var）
   - LN affine（gamma / beta）
   - 上投影 + SiLU
   - 下投影
2. 中间结果放到了 `shared memory`（`sdata`），避免多次写回显存。
3. 内部用 `float` 做累加，最后再 cast 回 `scalar_t`（支持 fp16/bf16）。

然后用 `AT_DISPATCH_FLOATING_TYPES_AND_HALF` 包一层：

```cpp
torch::Tensor fused_ln_mlp_forward(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor w1,
    torch::Tensor w2)
{
    auto hidden = x.size(-1);
    auto batch = x.size(0);
    auto intermediate = w1.size(0);
    auto y = torch::empty_like(x);

    int threads = 256;
    int blocks = batch;
    size_t shared_mem = sizeof(float) * intermediate;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "fused_ln_mlp_forward", [&] {
        fused_ln_mlp_forward_kernel<scalar_t><<<blocks, threads, shared_mem>>>(
            x.data_ptr<scalar_t>(),
            gamma.data_ptr<scalar_t>(),
            beta.data_ptr<scalar_t>(),
            w1.data_ptr<scalar_t>(),
            w2.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            hidden,
            intermediate);
    });
    return y;
}
```

> 这里的实现还有不少可以优化的空间：  
> 比如 mean/var 的 reduction 可以不用 `atomicAdd`，改成 warp-level reduce + shared 内规约；  
> matrix multiply 也可以更好地 tile 化。  
> 但作为“练手的第一版 fused kernel”，已经足够用来替换小模型的 MLP 做测试了。

### 4.3 PyTorch Extension & Python 包装：`FusedLayerNormMLP`

`fused_mlp.cpp` 里仅仅是 forward 函数声明 + `PYBIND11_MODULE` 注册（略），  
Python 端通过 extension 加载：

```python
# fused_mlp.py
import os
import sys
import torch
import torch.nn as nn

# --- 自动加载 fused_mlp_cuda.so ---
ext_path = os.path.expanduser("~/.cache/torch_extensions/py310_cu128/fused_mlp_cuda")
if ext_path not in sys.path:
    sys.path.append(ext_path)

import fused_mlp_cuda  # 来自 C++/CUDA 编译好的 so


class FusedLayerNormMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.ln_gamma = nn.Parameter(torch.ones(hidden_size))
        self.ln_beta = nn.Parameter(torch.zeros(hidden_size))
        self.fc1_weight = nn.Parameter(torch.randn(intermediate_size, hidden_size) * 0.02)
        self.fc2_weight = nn.Parameter(torch.randn(hidden_size, intermediate_size) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_mlp_cuda.fused_ln_mlp_forward(
            x,
            self.ln_gamma,
            self.ln_beta,
            self.fc1_weight,
            self.fc2_weight,
        )
```

编译通过 `setup.py`：

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="fused_mlp_cuda",
    ext_modules=[
        CUDAExtension(
            name="fused_mlp_cuda",
            sources=["fused_mlp.cpp", "fused_mlp_kernel.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

可以单独写个小脚本测试一下：

```python
# test_fused_mlp.py
import torch
from fused_mlp import FusedMLP  # 或 FusedLayerNormMLP

if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, I = 1, 896, 4864

    x = torch.randn(B, H, device="cuda", dtype=torch.float16)
    model = FusedMLP(H, I).cuda().half()

    y = model(x)
    print("Output shape:", y.shape)
    print("Output sample:", y[0, :5].detach().cpu())
```

### 4.4 替换 Qwen 模型中的 MLP：`replace_mlp_in_qwen`

在 `stageB_fused.py` 里有这么一句：

```python
from patch_qwen_fused_mlp import replace_mlp_in_qwen

# ...
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE
).eval()

replace_mlp_in_qwen(model, verbose=False)  # 层替换和构建合并为同一阶段
```

`replace_mlp_in_qwen` 的思路大概是：

1. 遍历 `model.model.layers`（Qwen 的每个 decoder block）；
2. 找到其中原本的 MLP / FFN 模块（比如 `layer.mlp`）；
3. 读取其中的：
   - LayerNorm 参数（gamma / beta）
   - up_proj / down_proj 权重
4. 构建一个 `FusedLayerNormMLP`，把权重拷贝进去；
5. 用 `FusedLayerNormMLP` 替换掉原来的 MLP 模块。

这样做完之后，再跑 `generate_one`，  
模型结构就已经实质上在用我们自定义的 fused kernel 了。

### 4.5 stageB_fused 的整体脚本

`stageB_fused.py` 的结构跟 `stageB.py` 类似：

- 加载 tokenizer & 模型
- 调用 `replace_mlp_in_qwen` 完成“构建阶段”（加载 + 层替换）
- 做 warmup
- 调用 `generate_one` 做 prefill + decode，并统计时间

```python
if __name__ == "__main__":
    print("=" * 70)
    print(f"[INFO] Model: {MODEL_ID}")
    print(f"[INFO] Device: {DEVICE} | DType: {DTYPE}")
    print("=" * 70)

    # 构建阶段 
    t_build_start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE
    ).eval()

    replace_mlp_in_qwen(model, verbose=False)

    t_build_end = time.time()

    # Warmup
    warmup_model(model, tokenizer, warmup_steps=5)

    # 推理
    user_prompt = "用要点概述：如何从零实现一个迷你版 vLLM 推理循环？"
    prompt = chat_template(tokenizer, user_prompt)

    text, perf = generate_one(model, tokenizer, prompt, max_new_tokens=300)

    print("
[PERFORMANCE SUMMARY]
")
    print(f"构建阶段耗时      : {t_build_end - t_build_start:8.3f} 秒")
    print(f"Prefill 阶段耗时 : {perf['prefill_time']:8.3f} 秒")
    print(f"Decode 阶段耗时  : {perf['decode_time']:8.3f} 秒")
    print(f"生成 Token 数    : {perf['num_generated']:8d}")
    print(f"生成速度         : {perf['tokens_per_sec']:8.2f} tokens/sec")
```

我把“模型加载 + MLP 层替换”整体算作 **构建阶段**，和后面的推理阶段区分开来，  
这样更贴近大引擎里“build engine / runtime vs. run inference”的思路。

---

## 5. 三个阶段对比：从朴素实现到“nano-vLLM”

简单总结一下三个阶段的差别：

| Stage           | 主要特性                                                                 | 关注点                                                     |
|-----------------|--------------------------------------------------------------------------|------------------------------------------------------------|
| **Stage A**     | 朴素 HF 推理循环；显式 prefill + decode；每步新建张量                  | 先跑通逻辑，理解 KV cache + 增量解码                      |
| **Stage B**     | SDPA；DecodeStep 抽象；pin_memory + non_blocking；GPU 预分配；warmup   | 让 decode 更“喂饱 GPU”，为后续编译 / kernel 优化铺路     |
| **Stage B fused** | 替换 MLP 为自定义 fused CUDA kernel；统计构建阶段和推理阶段的耗时     | 体验一次“自己写 kernel、接到模型里”的完整链路            |

从工程角度看，这已经是一个 **单请求、单模型的小型推理引擎** 了：

- 拥有清晰的 **prefill / decode 分层**
- 对 decode 循环做了基本的工程优化
- 支持替换内部算子为自定义 kernel

离 vLLM 等成熟引擎还有很远，但思路的骨架已经搭起来了。

---

## 6. 实践中的一些细节 & 可以改进的点

随手记几个实践中的小心得：

1. **计时最好配合 `torch.cuda.synchronize()`**  
   本文代码里为了简洁没写，但更严谨的写法是：

   ```python
   torch.cuda.synchronize()
   t0 = time.time()
   # ... kernel 调用
   torch.cuda.synchronize()
   t1 = time.time()
   ```

   这样能避免 CUDA 异步执行带来的误差。

2. **warmup 非常必要**  
   不管是 PyTorch 自身的 graph capture、还是 CUDA 内核的首次加载，  
   都会让第一次调用显著偏慢。如果你只测一次，就很容易被这个误差误导。

3. **fused kernel 要注意数值精度**  
   - 累加时用 `float`；  
   - 最后 cast 回 `half`；  
   - LayerNorm 里要加 `eps`，避免除 0；  
   - 和原始 PyTorch 实现对比时，不要指望 bitwise 一致，只要误差在合理范围即可。

4. **替换 MLP 时要小心对齐维度**  
   - `hidden_size`、`intermediate_size` 都要从原模型里读；  
   - 有的 MLP 是“门控 + up_proj”的结构，权重更复杂，需要一一对齐。

5. **nano-vLLM 的限制**  
   目前这个实践只做了“单请求单批单卡”的路径，  
   和真正的 vLLM 相比还缺很多东西，比如：

   - paged KV cache / chunk kv cache
   - multi-request scheduling / continuous batching
   - tensor parallel / pipeline parallel
   - streaming 输出、服务化、监控等

   但对于理解“LLM 推理引擎内部到底在忙什么”，已经是个不错的入门项目。

---

## 7. 小结 & 下一步打算

这次小实践大概经历了这样一个路径：

1. **先用 HuggingFace 朴素实现一个“手写 vLLM 循环”**  
   - 亲手拆开 prefill 和 decode
   - 看清楚 `past_key_values` 在循环中如何被复用

2. **再做一些常见的推理层面优化**  
   - decode 单步模块化（`DecodeStep`）
   - pin_memory + non_blocking
   - 预分配 GPU 张量，避免循环中的频繁 alloc/free
   - warmup 稳定测量

3. **最后尝试写一个自己的 fused MLP kernel，并替换到模型里**  
   - 体会从 C++/CUDA -> PyTorch Extension -> 模型参数替换的完整链路
   - 用性能统计验证“写 kernel 不是玄学，而是可以量化的”

后续可以进一步做的事情包括：

- 把 decode 路径尝试 `torch.compile`，看看和手写 fused 的差异；
- 实验多请求场景（简单版 batch & KV cache 管理）；
- 改进 fused kernel：用更合理的 reduction、tiling、甚至引入 cutlass 等库。

---

如果你也想自己写个“nano-vLLM”，  
完全可以从 Stage A 那几十行代码开始，一点点往前推。  
哪怕最后发现性能提升有限，这个过程本身也很有趣：  
你会对“LLM 推理”这件事的成本构成有更直观的感受。

> 以上就是这次小实践的全部记录，如果你在复现过程中遇到问题，  
> 可以从以下几个方向排查：  
> - 维度是否对齐（特别是 fused MLP 部分）  
> - dtype / 设备是否一致  
> - KV cache 是否正确传递到 decode 单步  
> - 是否忘了 warmup / cuda synchronize 影响了测量结果
