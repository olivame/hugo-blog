
---
title: "VLESS + WebSocket + Cloudflare Tunnel + WARP 完整配置指南"
date: 2025-12-08T16:48:00+08:00
---

<!--more-->
---

> 本文为技术实践总结，所有示例中的域名、UUID、Tunnel ID 等敏感信息均为示例，请务必替换为你自己的实际信息。

---

## 1. 整体思路与架构概览

我们要在一台云服务器上搭建这样一条链路：

```text
客户端 →（VLESS+WS+TLS）→ Cloudflare Anycast → Cloudflare Tunnel
      →（HTTP 回源）→ Xray (127.0.0.1)
      →（出站）→ 目标网站 / WARP → Internet
```

并且希望支持 **三种使用方式**：

1. **方案一：直连出口（freedom）**  
   - 入口：Cloudflare Tunnel  
   - 出口：云服务器自己的公网 IP  

2. **方案二：WARP 出口**  
   - 入口：Cloudflare Tunnel  
   - 出口：Cloudflare WARP IP（隐藏服务器出口）  

3. **方案三：直连 + WARP 双出口共存**  
   - 同时提供两个节点：  
     - 节点 A：直连出口  
     - 节点 B：WARP 出口  

下面的章节会把：

- **通用准备工作（安装软件 / Tunnel 登录 / 域名绑定）** 放在前面  
- 然后再针对不同方案分别给出「只改 Xray 配置即可」的部分  

---

## 2. 环境准备与通用安装步骤

这一节是**所有方案共用**的准备工作，只需要做一次。

### 2.1 基本环境

- 系统：推荐 Debian / Ubuntu 20.04+（本文以 Ubuntu 为例）
- 架构：x86_64
- 已具备：
  - 一个 Cloudflare 账号，并已将你的域名托管到 Cloudflare
  - 一台云服务器（GCP / AWS / Vultr / 其他）

假设：

- 域名：`example.com`
- 子域名：`us-node.example.com`
- 服务器用户名：`ubuntu`

> 以下命令均在服务器上以具有 sudo 权限的用户执行。  
> 所有 `UUID` / `TUNNEL_ID` / `域名` 等请替换为你自己的值。

---

### 2.2 安装 Xray（两步执行）


```bash
# ① 下载安装脚本
curl -Ls https://raw.githubusercontent.com/XTLS/Xray-install/main/install-release.sh -o install-release.sh

# ② 赋予执行权限并运行
sudo bash install-release.sh
```

安装完成后：

- 程序路径：`/usr/local/bin/xray`
- 配置路径：`/usr/local/etc/xray/config.json`
- 服务名：`xray.service`

---

### 2.3 安装 Cloudflare Tunnel（cloudflared）

```bash
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o cloudflared.deb
sudo dpkg -i cloudflared.deb
```

验证版本：

```bash
cloudflared --version
```

---

### 2.4 安装 Cloudflare WARP（Linux）

> 这是为了后面方案二 / 方案三使用的。  
> 如果你只打算用「直连出口」，可以先略过 2.4（但建议装好以备后用）。

1. 添加官方源并安装：

```bash
curl https://pkg.cloudflareclient.com/pubkey.gpg \
  | sudo gpg --yes --dearmor --output /usr/share/keyrings/cloudflare-warp-archive-keyring.gpg

echo "deb [signed-by=/usr/share/keyrings/cloudflare-warp-archive-keyring.gpg] \
  https://pkg.cloudflareclient.com/ $(lsb_release -sc) main" \
  | sudo tee /etc/apt/sources.list.d/cloudflare-client.list

sudo apt update
sudo apt install cloudflare-warp -y
```

2. 初始化并设置为 **代理模式**（不是全局模式）：

```bash
warp-cli registration new
warp-cli mode proxy
warp-cli proxy port 40000
warp-cli connect
```

3. 测试 WARP 出口 IP：

```bash
curl --socks5 127.0.0.1:40000 https://ifconfig.me
# 应返回 Cloudflare IP，如 104.x.x.x / 172.x.x.x 等
```

---

### 2.5 通用：Cloudflare Tunnel 登录 & 创建 Tunnel & 绑定域名

> 这部分是 **Cloudflare Tunnel 的通用前置步骤**，之后所有方案都会用到同一个 Tunnel。

#### 2.5.1 登录 Cloudflare 账号

```bash
cloudflared tunnel login
```

按照提示在浏览器中确认授权，完成后本地会生成：

- `~/.cloudflared/cert.pem` 等认证文件

#### 2.5.2 创建 Tunnel

```bash
cloudflared tunnel create myproxy
```

命令输出中记下：

- `Tunnel ID`：例如 `TUNNEL_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`
- 凭证文件：`~/.cloudflared/TUNNEL_ID.json`

#### 2.5.3 为 Tunnel 绑定域名（DNS 路由）

假设要使用子域名 `us-node.example.com` 作为入口：

```bash
cloudflared tunnel route dns myproxy us-node.example.com
```

Cloudflare 会自动创建一条 CNAME：

```text
us-node.example.com → xxxx.cfargotunnel.com
```

---

### 2.6 通用：Cloudflare Tunnel 基础配置与服务

> 接下来我们写一个「基础」的 cloudflared 配置。  
> 之后各方案只需要改 `ingress` 部分即可。

`~/.cloudflared/config.yml` 示例：

```yaml
tunnel: TUNNEL_ID
credentials-file: /home/ubuntu/.cloudflared/TUNNEL_ID.json

# 这里只写一个基础 ingress，占位用，后面各方案会覆盖这一块
ingress:
  - service: http_status:404
```

> 注意：这里暂时只放一个兜底规则，真正的反代规则会在后面的各个方案中分别给出。

然后创建 systemd 服务（全局通用）：

`/etc/systemd/system/cloudflared.service`：

```ini
[Unit]
Description=Cloudflare Tunnel
After=network.target

[Service]
Type=simple
User=ubuntu
ExecStart=/usr/local/bin/cloudflared tunnel run TUNNEL_ID
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
```

加载并启动：

```bash
sudo systemctl daemon-reload
sudo systemctl enable cloudflared --now
sudo systemctl status cloudflared
```

看到类似：

```text
INF Registered tunnel connection ...
```

说明 Tunnel 已经和 Cloudflare 网络连上了。

> 至此：**入口（Cloudflare 侧）已经准备好，接下来只需要配置 Xray 与 Tunnel 之间如何回源，以及 Xray 如何出站即可。**

---

## 3. 共用的 Xray 基础入站（VLESS + WS）

这一节给出一个「通用模版」：  
**一个 VLESS + WS 的 inbound，在本地 127.0.0.1:18888 监听。**

后续三种方案只是在 **outbounds / routing / 多 inbound** 上做变化。

### 3.1 生成 UUID

```bash
uuidgen
```

假设生成：

```text
xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

---

### 3.2 Xray 基础配置骨架

`/usr/local/etc/xray/config.json`：

```jsonc
{
  "inbounds": [
    {
      "port": 18888,
      "listen": "127.0.0.1",
      "protocol": "vless",
      "settings": {
        "clients": [
          {
            "id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx", // 替换为你的 UUID
            "flow": ""
          }
        ],
        "decryption": "none"
      },
      "streamSettings": {
        "network": "ws",
        "wsSettings": {
          "path": "/vless" // 默认路径，用于单节点或 WARP 节点
        }
      }
    }
  ],
  "outbounds": [
    // 不同方案在这里替换
  ],
  "routing": {
    // 不同方案在这里替换（或留空）
  }
}
```

> **重要：**  
> - 所有方案都会用 `VLESS + WS` 作为入口协议  
> - 路径默认为 `/vless`，在「双节点方案」中我们会再额外加一个 `/direct`

重启 Xray：

```bash
sudo systemctl restart xray
sudo systemctl status xray
```

---

## 4. 方案一：VLESS + WS + Cloudflare Tunnel + 直连出口（freedom）

这是最基础的方案，只隐藏入口 IP，不隐藏出口。

### 4.1 Xray 配置

将 `config.json` 改为：

```jsonc
{
  "inbounds": [
    {
      "port": 18888,
      "listen": "127.0.0.1",
      "protocol": "vless",
      "settings": {
        "clients": [
          {
            "id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
          }
        ],
        "decryption": "none"
      },
      "streamSettings": {
        "network": "ws",
        "wsSettings": {
          "path": "/vless"
        }
      }
    }
  ],
  "outbounds": [
    {
      "protocol": "freedom"  // 直接使用服务器出口访问目标网站
    }
  ],
  "routing": {
    "domainStrategy": "AsIs"
  }
}
```

---

### 4.2 Cloudflare Tunnel ingress

现在我们需要让 `us-node.example.com` 的流量回源到 `127.0.0.1:18888`：

编辑 `~/.cloudflared/config.yml`：

```yaml
tunnel: TUNNEL_ID
credentials-file: /home/ubuntu/.cloudflared/TUNNEL_ID.json

ingress:
  - hostname: us-node.example.com
    service: http://localhost:18888
  - service: http_status:404
```

重启 cloudflared：

```bash
sudo systemctl restart cloudflared
sudo systemctl status cloudflared
```

---

### 4.3 客户端节点配置

客户端使用的 VLESS 节点示例：

```text
vless://UUID@us-node.example.com:443?encryption=none&security=tls&type=ws&host=us-node.example.com&path=/vless#US-Node-Direct
```

- 协议：VLESS  
- 传输：WS  
- TLS：开启（Cloudflare 提供）  
- Host / SNI：`us-node.example.com`  
- Path：`/vless`  

**特点：**

- 入口：Cloudflare  
- 出口：你的云服务器 IP（非 WARP）

---

## 5. 方案二：VLESS + WS + Tunnel + WARP 出口（隐藏出口 IP）

在方案一的基础上，把 Xray 的出站改为 **走本机 WARP SOCKS5 代理**，从而隐藏出口。

> 假设之前已经完成了「2.4 安装 WARP 并设为代理模式」。

### 5.1 Xray 配置（全局走 WARP 出口）

替换 `/usr/local/etc/xray/config.json` 为：

```jsonc
{
  "inbounds": [
    {
      "port": 18888,
      "listen": "127.0.0.1",
      "protocol": "vless",
      "settings": {
        "clients": [
          {
            "id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
          }
        ],
        "decryption": "none"
      },
      "streamSettings": {
        "network": "ws",
        "wsSettings": {
          "path": "/vless"
        }
      }
    }
  ],

  "outbounds": [
    {
      "protocol": "socks",
      "tag": "warp-out",
      "settings": {
        "servers": [
          {
            "address": "127.0.0.1",
            "port": 40000   // WARP 代理端口
          }
        ]
      }
    },
    {
      "protocol": "freedom",
      "tag": "direct"
    }
  ],

  "routing": {
    "domainStrategy": "AsIs",
    "rules": [
      {
        "type": "field",
        "network": "tcp,udp",
        "outboundTag": "warp-out" // 所有流量走 WARP 出口
      }
    ]
  }
}
```

Cloudflare Tunnel 的 `config.yml` 可以保持和方案一一样（回源到 18888），不需要修改。

重启 Xray：

```bash
sudo systemctl restart xray
```

---

### 5.2 客户端节点配置（WARP 出口）

```text
vless://UUID@us-node.example.com:443?encryption=none&security=tls&type=ws&host=us-node.example.com&path=/vless#US-Node-WARP
```

**区别：**

- 从客户端角度看，配置基本一模一样  
- 真正的差异在于 **服务器出站走的是 WARP**，目标网站看到的是 Cloudflare WARP IP

---

## 6. 方案三：直连 + WARP 双出口共存（两个节点）

这一方案实现的是：

- **一个 Tunnel，一个域名**  
- **两个不同 path**：
  - `/vless` → WARP 出口  
  - `/direct` → 直连出口  

最终在客户端中可以看到两个节点：

- `US-WARP`：更隐蔽，出口是 WARP IP  
- `US-DIRECT`：更简单，出口是服务器 IP  

---

### 6.1 Xray 配置（双 inbound + 双 outbound + 路由区分）

`/usr/local/etc/xray/config.json`：

```jsonc
{
  "inbounds": [
    {
      "tag": "warp-in",
      "port": 18888,
      "listen": "127.0.0.1",
      "protocol": "vless",
      "settings": {
        "clients": [
          {
            "id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
          }
        ],
        "decryption": "none"
      },
      "streamSettings": {
        "network": "ws",
        "wsSettings": {
          "path": "/vless"
        }
      }
    },
    {
      "tag": "direct-in",
      "port": 18889,
      "listen": "127.0.0.1",
      "protocol": "vless",
      "settings": {
        "clients": [
          {
            "id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
          }
        ],
        "decryption": "none"
      },
      "streamSettings": {
        "network": "ws",
        "wsSettings": {
          "path": "/direct"
        }
      }
    }
  ],

  "outbounds": [
    {
      "protocol": "socks",
      "tag": "warp-out",
      "settings": {
        "servers": [
          {
            "address": "127.0.0.1",
            "port": 40000
          }
        ]
      }
    },
    {
      "protocol": "freedom",
      "tag": "direct"
    }
  ],

  "routing": {
    "domainStrategy": "AsIs",
    "rules": [
      {
        "type": "field",
        "inboundTag": ["warp-in"],
        "outboundTag": "warp-out"
      },
      {
        "type": "field",
        "inboundTag": ["direct-in"],
        "outboundTag": "direct"
      }
    ]
  }
}
```

> 两个 inbound 使用同一个 UUID 即可。

重启 Xray：

```bash
sudo systemctl restart xray
```

---

### 6.2 Cloudflare Tunnel ingress（按 path 分流）

`~/.cloudflared/config.yml`：

```yaml
tunnel: TUNNEL_ID
credentials-file: /home/ubuntu/.cloudflared/TUNNEL_ID.json

ingress:
  - hostname: us-node.example.com
    path: /vless
    service: http://localhost:18888

  - hostname: us-node.example.com
    path: /direct
    service: http://localhost:18889

  - service: http_status:404
```

重启：

```bash
sudo systemctl restart cloudflared
sudo systemctl status cloudflared
```

---

### 6.3 客户端两个节点配置

#### 6.3.1 WARP 出口节点（隐藏出口 IP）

```text
vless://UUID@us-node.example.com:443?encryption=none&security=tls&type=ws&path=%2Fvless&host=us-node.example.com&sni=us-node.example.com#US-WARP
```

#### 6.3.2 直连出口节点（不隐藏出口 IP）

```text
vless://UUID@us-node.example.com:443?encryption=none&security=tls&type=ws&path=%2Fdirect&host=us-node.example.com&sni=us-node.example.com#US-DIRECT
```

导入后，你会在客户端看到两个节点：

- `US-WARP`：走 WARP 出口  
- `US-DIRECT`：走服务器直连出口  

---

## 7. 开机自启配置

无论使用哪一种方案，建议都开启开机自启。

### 7.1 Xray 开机启动

```bash
sudo systemctl enable xray
```

### 7.2 Cloudflare Tunnel 开机启动

```bash
sudo systemctl enable cloudflared
```

### 7.3 WARP 后台服务

安装 WARP 后会自带 `warp-svc` 服务，一般自动随系统启动，可用：

```bash
sudo systemctl status warp-svc
```

检查运行状态。

---

## 8. 三种代理方式的优缺点对比

### 8.1 方案一：直连出口（只隐藏入口 IP）

**优点：**

- 架构最简单：Xray + Cloudflare Tunnel
- 延迟最低（少了一跳 WARP）
- 故障点少，排错容易
- 所有协议和端口基本不受 WARP 限制

**缺点：**

- 目标网站看到的是你的 **云服务器出口 IP**
- 云服务器 IP 如果被封锁 / 限速，会直接影响所有流量
- 某些服务对云服务器 IP 有天然风控

**适用场景：**

- 个人小流量使用  
- 不频繁被封 IP 的地区  
- 更在意稳定与延迟，而不是 IP 特征

---

### 8.2 方案二：WARP 出口（入口 + 出口双重隐藏）

**优点：**

- 出口 IP 由 WARP 提供，目标网站看不到你的服务器真实 IP
- 出口 IP 来自 WARP 池，相比云服务器单 IP，更不容易被单点封死
- 入口由 Cloudflare Tunnel 隐藏，整体隐蔽性更高

**缺点：**

- 多了一层代理，延迟略高
- WARP 本身也可能有风控 / 限速
- 某些网站对 Cloudflare 段 IP 有特殊处理
- 增加一个新故障点（WARP）

**适用场景：**

- 本地 IP / 云服务器 IP 容易被封
- 特别在意「不要暴露真实服务器出口 IP」
- 可以接受略高的延迟来换取更高隐蔽性

---

### 8.3 方案三：直连 + WARP 共存（最灵活）

**优点：**

- 一台服务器支持两种出口模式：
  - 不敏感、追求速度 → 用直连节点
  - 敏感、需换 IP 特征 → 用 WARP 节点
- 入口配置统一：同一个域名 + 同一个 Tunnel，不需要多域名
- WARP 出现问题时可以快速切回直连；直连被限时也可以临时切到 WARP

**缺点：**

- 配置比前两种方案复杂（多 inbound、多 path、多节点）
- 排错时需要分清是哪一段出问题（Xray / Tunnel / WARP / 客户端）
- 占用略多内存（多一个 inbound）

**适用场景：**

- 想“一台服务器打两份工”，在不同场景下切换出口策略
- 对 IP 特征、稳定性、延迟都有不同需求
- 长期使用，希望有冗余方案，而不是只有一种线路

---

## 9. 总结

通过 **Xray + VLESS + WebSocket + Cloudflare Tunnel**，你可以实现：

- **入口 IP 隐藏**：外界只看到 Cloudflare 边缘节点 IP  
- **可选出口 IP 隐藏**：结合 WARP，将出口 IP 伪装为 Cloudflare 官方 IP  
- 使用标准 `443 + TLS + WS`，流量与正常 HTTPS 十分相似，不易被简单特征识别  

你可以根据自己的需求选择：

- 只要简单稳定 → 用 **方案一（直连出口）**  
- 追求入口 + 出口双隐藏 → 用 **方案二（WARP 出口）**  
- 需要灵活切换 / 备份线路 → 用 **方案三（直连 + WARP 双出口）**  

实际部署时，只需记住三件事：

1. **Xray：** 在本机回环（127.0.0.1）监听 VLESS+WS  
2. **Cloudflare Tunnel：** 把域名流量转发到对应端口（并可用 path 分流多个节点）  
3. **客户端：** 严格匹配服务器的 UUID、域名、端口、TLS 与 WS path  

按照本文的步骤，一台小服务器就能搭建出一套 **稳定、隐蔽、可扩展** 的访问方案。
