---
title: "基于 cuBLASLt 的 A8W8 线性层：从静态量化到 Qwen 推理评测实战"
date: 2025-12-05T21:07:00+08:00
---

<!--more-->
---

> 这篇是我在 Qwen2.5-0.5B-Instruct 上做 **A8W8 静态量化 + cuBLASLt INT8 Tensor Core** 的一次完整实验记录：  
> 从离线权重量化、INT8 线性层实现，到 CUDA 扩展、模型替换和 benchmark，对“为什么当前 INT8 没有比 FP16 快”也做了一点拆解和反思。

---

## 0. TL;DR

- 我实现了一个 **Static A8W8** 推理原型：权重量化为 INT8 + 激活固定缩放因子 `a_scale`。
- 用 cuBLASLt 写了一个 **INT8 GEMM 扩展**（单次 GEMM + batched GEMM），支持 Tensor Core。
- 通过递归替换 `torch.nn.Linear`，把 Qwen2.5-0.5B 模型中所有 Linear 换成 `QuantLinearInt8`。
- 写了一个小 benchmark，比较 **FP16 vs INT8** 的速度和显存占用。
- 实验结果：**INT8 目前并没有跑赢 FP16**，但显存占用确实下降了 ~10–15%。  
  核心原因是：目前 pipeline 还不是真正的 A8W8，只是“**W8A16 + 额外反量化开销**”。

后面会从上到下把这条 pipeline 拆开讲一遍，也会聊一下后续可以如何演进到真正高性能的 INT8 推理。

---

## 1. 整体架构：一个“小型 INT8 推理引擎”

这一套代码可以看作一个 **轻量级 INT8 推理实验框架**，围绕 Qwen2.5-0.5B-Instruct 展开，大致分成四块：

1. **离线权重量化**  
   - `quantize/quantize_model_int8.py`  
   - 输入：原始 safetensors 模型  
   - 输出：per-channel INT8 权重、scale、meta 信息

2. **QuantLinearInt8 层实现**  
   - `quant_linear_int8_tc_static.py`  
   - 提供 `QuantLinearInt8`，支持：
     - 输入为 INT8：走 cuBLASLt INT8 GEMM + 反量化
     - 输入为 FP16：fallback 到 FP16 GEMM（W8A16）

3. **cuBLASLt INT8 GEMM CUDA 扩展**  
   - `int8_gemm_tc_ext.cu` + Python 侧 `load(...)` 构建 .so  
   - 负责把 `(A_int8, W_int8)` 映射到 Tensor Core 上做 `int8 → int32` 矩阵乘

4. **Benchmark 脚本**  
   - 加载 FP16 模型做 baseline  
   - 加载 FP16 模型并自动替换 Linear → QuantLinearInt8  
   - 构造随机输入，测 latency / tok/s / 显存占用  
   - 导出结果到 CSV 方便对比

理解这一层架构之后，再看每个模块的代码就会清晰很多。

---

## 2. 离线权重量化：`quantize_model_int8.py`

### 2.1 目标：把所有 Linear 权重变成 INT8 + per-channel scale

我们先针对 **权重** 做静态量化（不在推理时重新量化），脚本入口是：

```python
SAVE_DIR = "quantized_qwen_int8"
MODEL_PATH = "../Qwen2.5-0.5B-Instruct/model.safetensors"
os.makedirs(SAVE_DIR, exist_ok=True)

QMAX = 127  # int8 对称量化的最大绝对值
```

### 2.2 遍历 safetensors，按行做对称量化

核心逻辑：

```python
tensors = load_file(MODEL_PATH)
meta = {}

for name, W in tensors.items():
    if W.ndim == 2:
        # 只量化 Linear 权重
        W = W.to("cuda")

        # 逐行 scale: scale_i = max(|W[i]|) / 127
        scales = W.abs().amax(dim=1) / QMAX
        scales = scales.clamp(min=1e-8)

        # int8 量化
        Wq = torch.round(W / scales[:, None]).clamp(-128, 127).to(torch.int8)

        np_q = Wq.cpu().numpy()
        np_s = scales.float().cpu().numpy()

        np.save(os.path.join(SAVE_DIR, name.replace(".", "__") + ".int8.npy"), np_q)
        np.save(os.path.join(SAVE_DIR, name.replace(".", "__") + ".scale.npy"), np_s)

        meta[name] = {
            "shape": list(W.shape),
            "bits": 8,
            "type": "int8_per_channel",
            "in_features": int(W.shape[1]),
            "out_features": int(W.shape[0])
        }
    else:
        # 非线性层参数：直接以 FP32 numpy 保存
        np.save(os.path.join(SAVE_DIR, name.replace(".", "__") + ".fp.npy"),
                W.float().cpu().numpy())
```

要点：

- **只量化 2D 权重**：即 Linear 层的 `[out_features, in_features]` 矩阵。
- 使用 **对称 per-channel 量化**（按输出通道行）：
  - `W_int8[i, :] = round(W[i, :] / scale_i)`
  - `scale_i = max(|W[i]|) / 127`
- 所有非 Linear 参数（LayerNorm、bias、embedding 等）都以 FP32 numpy 原样保存。

最后会写出一个 `quant_meta.json`：

```python
with open(os.path.join(SAVE_DIR, "quant_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
```

INT8 权重目录长这样：

```text
quantized_qwen_int8/
  - model.layers.0.mlp.gate_proj.weight.int8.npy
  - model.layers.0.mlp.gate_proj.weight.scale.npy
  - ...
  - quant_meta.json
```

---

## 3. 静态 A8W8 线性层：`QuantLinearInt8`

### 3.1 模块设计

`quant_linear_int8_tc_static.py` 里定义了核心类：

```python
class QuantLinearInt8(nn.Module):
    # Static-A8W8 Linear:
    #   - 输入必须是 int8，不做 runtime quant
    #   - y_int32 = A_int8 @ W_int8^T
    #   - y = y_int32 * (a_scale * w_scale)
```

这里强调的是 **Static**：

- 激活量化 scale `a_scale` 是一个 **标量**，由离线校准/脚本给出；
- 权重用的是刚才存好的 INT8 + per-channel `w_scale`；
- 不在 `forward` 里做 runtime quant，而是认为输入已经是 INT8（理想情况）。

定义时，模块持有以下参数/缓冲：

```python
self.register_buffer("a_scale", torch.tensor(float(a_scale), dtype=torch.float16))
self.register_buffer("w_int8", None)   # [N, K]
self.register_buffer("w_scale", None)  # [N], fp16

if bias:
    self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
```

权重通过 `set_weight` 注入：

```python
def set_weight(self, w_int8: torch.Tensor, w_scale: torch.Tensor):
    assert w_int8.dtype == torch.int8
    assert w_scale.ndim == 1
    assert w_scale.shape[0] == w_int8.shape[0]

    self.w_int8 = w_int8.contiguous()
    self.w_scale = w_scale.to(torch.float16).contiguous()
```

### 3.2 forward：INT8 路径 vs FP16 fallback

`forward` 支持两条路径：

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # 展平 3D -> 2D
    if orig_dim == 3:
        B, S, K = x.shape
        x_2d = x.reshape(-1, K)
        B_eff = B * S
    else:
        B_eff, K = x.shape
        S = 1
        B = None

    # 1) 真正 INT8 path：输入 x 是 int8
    if x.dtype == torch.int8:
        y_int32 = int8_gemm_tc_ext.int8_gemm_tc(x_2d, self.w_int8)  # [B_eff, N]

        a_s = self.a_scale.to(y_int32.device).to(torch.float32)     # scalar
        w_s = self.w_scale.to(torch.float32)                        # [N]

        eff = a_s * w_s                                             # [N]
        y_fp32 = y_int32.to(torch.float32) * eff[None, :]
        y_fp16 = y_fp32.to(torch.float16)

    # 2) FP16 fallback：输入 x 是 fp16
    elif x.dtype == torch.float16:
        a_s = self.a_scale.to(x.device).to(torch.float32)
        w_s = self.w_scale.to(torch.float32)
        eff = a_s * w_s

        W_deq = self.w_int8.to(torch.float16) * eff.to(torch.float16)[:, None]
        y_fp16 = x_2d.to(torch.float16) @ W_deq.t()

    else:
        raise AssertionError(...)
```

两条路径含义：

- **INT8 path（理想态）**
  - `A_int8`: 已量化好的激活
  - `W_int8`: 量化权重
  - 先用 cuBLASLt 做 `int8_gemm_tc` 得到 `y_int32`  
  - 再用 `a_scale * w_scale` 做一次性浮点反量化。

- **FP16 fallback（目前真实在跑的）**
  - 输入 `x` 是 FP16 embedding/hidden states；
  - 只反量化权重：`W_fp16 = W_int8 * eff`；
  - 再做标准 FP16 GEMM：`y = x @ W_fp16^T`。

> 也就是说，在当前 pipeline 里，**并没有真正起用 A8W8**：  
> - 激活没有被量化（仍然是 FP16）  
> - 权重每次前向都要 **先反量化** 才能 GEMM  
> 这就解释了为什么实验里 INT8 路径比 FP16 还慢——因为它“多做了一步”。

---

## 4. cuBLASLt INT8 GEMM CUDA 扩展：`int8_gemm_tc_ext.cu`

### 4.1 为什么需要 cuBLASLt？

PyTorch 自带的 `torch.matmul`/`mm` 对 INT8 支持有限，而且对 Tensor Core 的使用和张量布局有要求。  
cuBLASLt 是 NVIDIA 提供的 **高性能矩阵乘** 库，特点：

- 支持 INT8 Tensor Core
- 支持多种矩阵布局（RowMajor, ColMajor 等）
- 支持 Heuristic 筛选最佳算法
- 支持 Strided Batch GEMM——对 attention、batched matmul 很友好

因此这里选择写一个小扩展，把 PyTorch 的 Tensor 传给 cuBLASLt 来做 INT8 GEMM。

### 4.2 单次 GEMM：`int8_gemm_tc`

接口定义：

```cpp
// C[M,N] = A[M,K] @ B[N,K]^T
torch::Tensor int8_gemm_tc(torch::Tensor A, torch::Tensor B);
```

- `A`: `[M, K]` row-major int8
- `B`: `[N, K]` row-major int8（内部转置为 `[K, N]`）
- `C`: `[M, N]` row-major int32

关键步骤：

1. **创建 Matmul 描述符**

```cpp
cublasLtMatmulDesc_t desc;
CHECK_LT(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32I, CUDA_R_32I));

cublasOperation_t opA = CUBLAS_OP_N;
cublasOperation_t opB = CUBLAS_OP_T;

CHECK_LT(cublasLtMatmulDescSetAttribute(
    desc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
CHECK_LT(cublasLtMatmulDescSetAttribute(
    desc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));
```

2. **定义矩阵布局（RowMajor + leading dimension）**

```cpp
cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_8I, M, K, K);
cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_8I, N, K, K);
cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_32I, M, N, N);

cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
cublasLtMatrixLayoutSetAttribute(layoutA, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
...
```

3. **设置 Preference（例如不使用额外 workspace）**

```cpp
cublasLtMatmulPreference_t pref;
cublasLtMatmulPreferenceCreate(&pref);
size_t ws_size = 0;
cublasLtMatmulPreferenceSetAttribute(
    pref,
    CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
    &ws_size, sizeof(ws_size));
```

4. **通过 heuristic 选择算法**

```cpp
cublasLtMatmulHeuristicResult_t heur;
int returned = 0;
cublasLtMatmulAlgoGetHeuristic(
    lt, desc, layoutA, layoutB,
    layoutC, layoutC,
    pref, 1, &heur, &returned);
TORCH_CHECK(returned > 0, "No valid INT8 algo found");
```

5. **执行 GEMM**

```cpp
int32_t alpha = 1, beta = 0;
cublasLtMatmul(
    lt, desc,
    &alpha,
      A_ptr, layoutA,
      B_ptr, layoutB,
    &beta,
      nullptr, layoutC,
      C_ptr, layoutC,
    &heur.algo,
    nullptr, 0,
    stream);
```

最后返回一个 `[M, N]` 的 `int32` Tensor，Python 端再乘上 `(a_scale * w_scale)` 做反量化即可。

### 4.3 Strided Batch GEMM：`int8_bmm_tc`

另一个函数 `int8_bmm_tc` 支持 batched GEMM：

```cpp
// A: [B, M, K]
// B: [B, N, K]
// C: [B, M, N]
torch::Tensor int8_bmm_tc(torch::Tensor A, torch::Tensor B);
```

这里利用了 cuBLASLt 的 batch + stride 特性：

```cpp
int32_t batchCount = static_cast<int32_t>(batch);
long long strideA = static_cast<long long>(M) * K;
long long strideB = static_cast<long long>(N) * K;
long long strideC = static_cast<long long>(M) * N;

cublasLtMatrixLayoutSetAttribute(layoutA,
    CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount));
cublasLtMatrixLayoutSetAttribute(layoutA,
    CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA, sizeof(strideA));
...
```

这个在 MHA、Cross-Attention 等模块里很有用，虽然目前 demo 只在 Linear 上用到了单次 GEMM。

### 4.4 Python 侧构建扩展

Python 端通过 `torch.utils.cpp_extension.load` 动态编译：

```python
from torch.utils.cpp_extension import load

int8_ext = load(
    name="int8_gemm_tc_ext",
    sources=["int8_gemm_tc_ext.cu"],
    verbose=True,
    extra_cuda_cflags=[
        "-O3",
        "-std=c++17",
        "--use_fast_math",
        "-gencode=arch=compute_80,code=sm_80",
        "--expt-relaxed-constexpr"
    ],
)
```

- 指定 `sm_80`，说明主要考虑 A100 级别 GPU；
- `--use_fast_math` 提高吞吐，但也会稍微牺牲一点数值精度（对 GEMM 来说一般还好）。

---

## 5. 模型自动替换：把 Linear 换成 QuantLinearInt8

### 5.1 替换策略

切换到 benchmark 脚本的下半部分：

```python
from quant_linear_int8_tc_static import QuantLinearInt8TC

int8_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map={"": device},
).eval()

with open(os.path.join(QUANT_DIR, "quant_meta.json")) as f:
    quant_meta = json.load(f)

with open(ACT_SCALE_FILE) as f:
    act_scales = json.load(f)
```

然后通过一个递归函数完成替换：

```python
def replace_with_int8(module, prefix=""):
    for name, child in list(module.named_children()):
        full = f"{prefix}.{name}" if prefix else name
        key = full + ".weight"

        if isinstance(child, torch.nn.Linear) and key in quant_meta:

            info = quant_meta[key]
            w_path = os.path.join(QUANT_DIR, key.replace(".", "__") + ".int8.npy")
            s_path = os.path.join(QUANT_DIR, key.replace(".", "__") + ".scale.npy")

            w_int8 = torch.from_numpy(np.load(w_path)).to(torch.int8).cuda()
            w_scale = torch.from_numpy(np.load(s_path)).to(torch.float16).cuda()

            a_scale = float(act_scales.get(full, 0.02))

            ql = QuantLinearInt8TC(
                info["in_features"],
                info["out_features"],
                a_scale=a_scale,
                bias=(child.bias is not None)
            ).cuda()

            ql.set_weight(w_int8, w_scale)
            if child.bias is not None:
                ql.bias.data = child.bias.clone()

            setattr(module, name, ql)
            print("Replaced:", full)

        else:
            replace_with_int8(child, full)
```

关键点：

- 通过 `named_children()` + `prefix` 拼出 `full` 名字；
- 用 `full + ".weight"` 对应到 `quant_meta` 和 `.int8.npy/.scale.npy` 文件；
- 对每个 Linear：
  - 加载量化后的 `w_int8`、`w_scale`；
  - 创建 `QuantLinearInt8TC`（同 `QuantLinearInt8`）；
  - 把原 Linear 的 bias 拷贝过去；
  - 用 `setattr(module, name, ql)` 就地替换模块。

替换完成后 `int8_model` 的结构保持不变，但 Linear 都变成了自家写的 INT8 版本。

---

## 6. Benchmark：FP16 vs INT8

### 6.1 随机 batch 构造

```python
def build_random_batch(batch, seq_len):
    ids = torch.randint(
        100, tokenizer.vocab_size - 100,
        (batch, seq_len),
        dtype=torch.long,
        device=device
    )
    if tokenizer.eos_token_id:
        ids[:, -1] = tokenizer.eos_token_id
    mask = torch.ones_like(ids)
    return {"input_ids": ids, "attention_mask": mask}
```

简单起见，这里用随机 token 序列（中间避免特殊 token），最后一个位置强制设为 `eos_token_id`。

### 6.2 计时函数

```python
@torch.no_grad()
def benchmark_forward(model, batch, seq_len=SEQ_LEN, steps=10, use_cache=True):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    inputs = build_random_batch(batch, seq_len)

    # warmup
    for _ in range(3):
        _ = model(**inputs, use_cache=use_cache)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(steps):
        _ = model(**inputs, use_cache=use_cache)
    torch.cuda.synchronize()
    t1 = time.time()

    latency_ms = (t1 - t0) / steps * 1000
    tok_per_s = batch * seq_len / ((t1 - t0) / steps)
    peak_mem = torch.cuda.max_memory_allocated(device) / 1024 / 1024

    return latency_ms, tok_per_s, peak_mem
```

这里做了几件对 benchmark 来说很重要的事情：

- 每次 benchmark 之前 `empty_cache + reset_peak_memory_stats`，保证显存统计干净；
- **先 warmup 几步** 再开始计时，避免 JIT/内核构建影响结果；
- 调用 `torch.cuda.synchronize()` 确保计时窗口内的 CUDA 任务都执行完。

### 6.3 实验设置

- 模型：Qwen2.5-0.5B-Instruct
- 精度：FP16 / INT8(权重量化)
- 序列长度：1024
- batch size：`[1, 4, 8, 16, 32]`
- 每个点跑 10 次 forward 取平均

结果会被收集到 `bench_fp16_vs_int8tc.csv`，并打印 DataFrame。

---

## 7. 实验结果 & 现象

基于这套脚本的实际实验（这里参考了 PDF 报告中的数据），结果大致是：  

- **batch = 1** 时：
  - FP16 每次前向 ~30ms
  - INT8 每次前向 ~48ms —— **明显更慢**

- **batch 变大（4、8、16、32）**：
  - INT8 的延迟逐渐接近 FP16，但仍然略慢
  - 显存占用降低约 10–15%

简单说：

- 当前版本的 INT8 **没有带来速度优势**；
- 但 **显存的确少了一截**，说明权重量化本身是生效的。

这和很多人直觉上的“INT8 一定比 FP16 快很多”相反，背后有几个关键原因。

---

## 8. 为什么 INT8 没有跑赢 FP16？

结合代码和实验，可以把问题拆成几个层面来看。

### 8.1 激活没有真的量化（pipeline 不是 A8W8）

在目前的 pipeline 中：

- 模型的 embedding、LayerNorm 输出、attention 输出等全部仍然是 **FP16**；
- `QuantLinearInt8.forward` 在实际运行时走的是 **FP16 fallback path**：
  - `W_deq = W_int8 * eff`（每次前向都做）
  - `y = A_fp16 @ W_deq^T`（标准 FP16 GEMM）

也就是说：

> 真正的 `y_int32 = A_int8 @ W_int8^T` + `dequant` 路径根本没被触发。  
> 这就直接导致“INT8 算子没有用起来”。

要想用上 INT8 Tensor Core：

- 要么在 **embedding 之后** 把 activations 转成 INT8（乘上 `1/a_scale` 再 round/clamp）；
- 要么在 model 内部专门插入 quant/dequant 节点，保证 `QuantLinearInt8` 的输入是 INT8。

### 8.2 每次 forward 都在重复反量化权重

在 FP16 fallback path 中：

```python
W_deq = self.w_int8.to(torch.float16) * eff.to(torch.float16)[:, None]
y_fp16 = x_2d.to(torch.float16) @ W_deq.t()
```

注意，这段逻辑是 **每一次 forward、每一个 batch 都重新做一遍**。

但本质上：

- `W_deq` 对同一个 Linear 层而言，是一个**固定不变**的矩阵（只取决于权重和 scale）；
- 完全可以在：
  - 构建模型后 **离线反量化一次**；
  - 或者在首次 forward 的时候缓存起来，后面直接用。

重复反量化相当于在 original FP16 路径之外额外多了一个 GEMM 级别的代价，  
所以在 batch 不大的时候，INT8 路径自然更慢。

### 8.3 没有做权重预布局（weight prepacking）

cuBLASLt 对 INT8 的性能很依赖权重布局，比如：

- `ROW32`, `COL32`, `ROW4`, `COL4_4R2_8C` 等 layout
- 需要配合 Tensor Core 的 tile 形状

目前的实现里：

- 直接用 row-major `[N, K]` 去调用 `cublasLtMatmul`；
- 没有对权重做任何形式的 prepacking/reorder；
- 也没有为单个层缓存多次调用共享的优化结果。

这会导致：

- 内部需要做更多数据重排/拷贝；
- 实际吞吐很难达到 cuBLASLt 的最佳状态。

### 8.4 FP16 Tensor Core 本来就很强

在很多现代 GPU（尤其 A100/H100）上：

- FP16 Tensor Core 的性能已经很高；
- 如果 INT8 pipeline 本身不够“干净”，  
  比如有额外 dequant、reorder、类型转换等开销，  
  **很容易被 FP16 反超**。

---

## 9. 如何从这个原型演进到“真正的 A8W8 加速”？

如果把目前这套代码看成一个可跑的 **原型**，那么下一步可以从下面几个方向完善：

### 9.1 激活真正量化：打通 INT8 前向路径

目标是让 `QuantLinearInt8` 看到的 `x` 真的是 int8：

1. 在 embedding/前几层之后插入 `quantize` 节点：
   - `x_int8 = clamp(round(x_fp16 / a_scale), -128, 127)`
2. 确保后续 Linear 都用 `x_int8` 做 GEMM：
   - `y_int32 = int8_gemm_tc(x_int8, w_int8)`
   - `y_fp16 = y_int32 * (a_scale * w_scale)`
3. 只在必要的地方（比如输出 logits）才做 dequant 回 FP16。

可以先从 **某几层** 局部实验，验证数值误差和性能收益，再逐步扩展到全模型。

### 9.2 权重预处理：预反量化 / weight packing

两种典型思路：

- **预反量化 + FP16 GEMM**
  - 如果只想省显存（而不是追求极致性能），可以在构建后一次性生成所有 `W_fp16`；
  - 推理时只做 FP16 GEMM，相当于“W8A16 + 低显存”。

- **Weight packing + INT8 GEMM**
  - 在构建阶段，把权重转换到 cuBLASLt 友好的 layout；
  - 调用 `cublasLtMatmul` 时直接喂 packed 权重，减少重排开销；
  - 一般需要配合 `cublasLtMatmulAlgoGetHeuristic` 和 `cublasLtMatrixLayout` 的高级用法。

### 9.3 算子融合：INT32→FP16 + bias + activation

目前 INT8 路径的末尾大致是：

```text
y_int32  --(scale)-->  y_fp32  → (cast) → y_fp16
  + bias
  + 后续非线性/LayerNorm
```

可以考虑把以下步骤一并融合进一个 kernel 或一个 GEMM 调用后的小核：

- `int32 -> float`、乘以 scale；
- 加上 bias；
- 可选：再做一个简单的激活函数（如 ReLU）。

减少 kernel 启动次数，节省显存读写，是 INT8 真正跑快的重要一环。

### 9.4 对标开源方案，学习工程细节

想进一步深入，可以参考的一些方向：

- **AWQ / GPTQ**：如何设计权重量化 + 激活缩放策略，使得精度损失最小；
- **TRT-LLM / vLLM INT8**：如何做 paged KV cache + multi-request scheduling 下的 INT8 推理；
- **bitsandbytes / llama.cpp**：不同量化方案（4bit/8bit）的工程实现细节。


---

## 10. 小结：这是一个怎样的“练手项目”？

把这次实践压缩成几句话：

1. **离线权重量化**：把 Qwen2.5 的 Linear 权重做成 INT8 + per-channel scale；
2. **自定义 INT8 Linear**：支持 A8W8 与 W8A16 两条路径；
3. **cuBLASLt Tensor Core**：写了一个最小可用的 INT8 GEMM / BMM 扩展；
4. **模型自动替换 + benchmark**：一键把 Linear 换成 INT8 版本，并对比 FP16 baseline。

虽然目前的 INT8 还没有“跑赢 FP16”，但这套代码已经像一个小号的 INT8 推理引擎骨架：

- 掌握了 **量化格式 → kernel → module → 模型替换 → benchmark** 的完整闭环；
- 后续只要把激活量化、预处理与算子融合一步步补齐，就有机会做出 **真正有加速收益** 的 A8W8 pipeline。

---
