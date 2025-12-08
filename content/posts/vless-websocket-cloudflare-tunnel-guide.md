
---
title: "VLESS + WebSocket + Cloudflare Tunnel + WARP 完整配置指南 "
date: 2025-12-08T16:48:00+08:00
---

<!--more-->
---

> 本文为技术实践总结，所有示例中的域名、UUID、Tunnel ID 等敏感信息均为示例，请务必替换为你自己的实际信息。

---

## 0. 环境准备与软件安装

### 0.1 基本环境

- 系统：推荐 Debian / Ubuntu 20.04+（本文以 Ubuntu 为例）
- 架构：x86_64
- 已有：
  - 一个 Cloudflare 账号，已将你的域名托管到 Cloudflare
  - 一台云服务器（GCP/AWS/Vultr 等）

假设：

- 域名：`example.com`
- 子域名：`us-node.example.com`
- 服务器用户名：`ubuntu`

> 以下命令均在服务器上以具有 sudo 权限的用户执行。

---

### 0.2 安装 Xray

```bash
#下载安装脚本
curl -Ls https://raw.githubusercontent.com/XTLS/Xray-install/main/install-release.sh -o install-release.sh
#运行安装脚本
sudo bash install-release.sh
```

安装完成后：

- 程序路径：`/usr/local/bin/xray`
- 配置路径：`/usr/local/etc/xray/config.json`
- 服务：`xray.service`

---

### 0.3 安装 Cloudflare Tunnel（cloudflared）

```bash
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o cloudflared.deb
sudo dpkg -i cloudflared.deb
```

验证：

```bash
cloudflared --version
```

---

### 0.4 安装 Cloudflare WARP（Linux）

> **强烈建议只使用 Proxy 模式，不要用全局模式 warp**，否则可能改写系统路由导致服务器网络异常。

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

2. 初始化并设置为 **代理模式**：

```bash
warp-cli registration new
warp-cli mode proxy
warp-cli proxy port 40000
warp-cli connect
```

3. 测试 WARP 出口 IP：

```bash
curl --socks5 127.0.0.1:40000 https://ifconfig.me
# 应返回 Cloudflare IP，如 104.x.x.x / 172.x.x.x
```

---

## 一、整体架构概览

**目标：** 在云服务器上搭建一套基于 **VLESS + WebSocket (WS) + Cloudflare Tunnel** 的代理体系，并支持：

- **方案一：直连出口**（freedom，出口是服务器 IP）
- **方案二：WARP 出口**（出口是 Cloudflare WARP IP）
- **方案三：直连 + WARP 双出口共存**（两个节点自由切换）

| 组件              | 角色                     | 说明                                               |
| ----------------- | ------------------------ | -------------------------------------------------- |
| VLESS             | 代理协议                 | 轻量、高性能、无状态的加密传输协议                 |
| WebSocket (WS)    | 传输层                   | 伪装为 HTTPS 流量，特征不明显，便于穿透与伪装      |
| Xray              | 核心程序                 | 实现 VLESS 协议和 WebSocket 传输                   |
| Cloudflare Tunnel | 流量入口 / 反向代理      | 隐藏真实服务器 IP，提供免费 TLS 与全球加速         |
| Cloudflare WARP   | 二次代理（可选）         | 隐藏出口 IP，出口表现为 Cloudflare 官方 IP         |

---

## 二、方案一：VLESS + WS + CF Tunnel + 直连出口（freedom）

这是最基础的方案：**只隐藏入口 IP，不隐藏出口 IP**。

### 2.1 Xray 直连出口配置

> 配置文件路径：`/usr/local/etc/xray/config.json`

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
          "path": "/vless" // 必须与客户端一致
        }
      }
    }
  ],
  "outbounds": [
    {
      "protocol": "freedom" // 使用服务器出口直接访问目标网站
    }
  ]
}
```

生成 UUID：

```bash
uuidgen
```

写入 `"id"` 字段后重启 Xray：

```bash
sudo systemctl restart xray
```

---

### 2.2 Cloudflare Tunnel 配置（基础版）

1. 登录授权：

```bash
cloudflared tunnel login
```

2. 创建 Tunnel：

```bash
cloudflared tunnel create myproxy
```

记录生成的：

- `Tunnel ID`：`TUNNEL_ID`
- 凭证文件：`~/.cloudflared/TUNNEL_ID.json`

3. 绑定子域名（例：`us-node.example.com`）：

```bash
cloudflared tunnel route dns myproxy us-node.example.com
```

4. Tunnel 配置（`~/.cloudflared/config.yml`）：

```yaml
tunnel: TUNNEL_ID
credentials-file: /home/ubuntu/.cloudflared/TUNNEL_ID.json

ingress:
  - hostname: us-node.example.com
    service: http://localhost:18888
  - service: http_status:404
```

5. systemd 服务（`/etc/systemd/system/cloudflared.service`）：

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

> 如果 cloudflared 在 `/usr/bin`，记得改路径。

启用 & 启动：

```bash
sudo systemctl daemon-reload
sudo systemctl enable cloudflared --now
sudo systemctl status cloudflared
```

看到多条 `Registered tunnel connection` 即为成功。

---

### 2.3 客户端节点示例（直连出口）

```text
vless://UUID@us-node.example.com:443?encryption=none&security=tls&type=ws&host=us-node.example.com&path=/vless#US-Node-Direct
```

---

## 三、方案二：在服务器上单独使用 WARP 出口（隐藏出口 IP）

在方案一的基础上，把 Xray 出站改为 **走 WARP 的本地 SOCKS5 代理**，从而隐藏出口 IP。

### 3.1 确认 WARP 代理模式已启用

前面 0.4 中已安装并设置：

```bash
warp-cli registration new
warp-cli mode proxy
warp-cli proxy port 40000
warp-cli connect
warp-cli status
```

确认状态为：

```text
Status update: Connected
Mode: proxy
Proxy port: 40000
```

---

### 3.2 Xray 使用 WARP 出口配置

> 仍然只有一个入口 `/vless`，但出站走 WARP。

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
        "network": "tcp,udp",
        "outboundTag": "warp-out" // 全部走 WARP 出口
      }
    ]
  }
}
```

> Tunnel 配置（`~/.cloudflared/config.yml`）：(不做改变)

```yaml
tunnel: TUNNEL_ID
credentials-file: /home/ubuntu/.cloudflared/TUNNEL_ID.json

ingress:
  - hostname: us-node.example.com
    service: http://localhost:18888
  - service: http_status:404
```

重启 Xray：

```bash
sudo systemctl restart xray
```

客户端节点格式与方案一相同（只是出口变成 Cloudflare WARP IP）：

```text
vless://UUID@us-node.example.com:443?encryption=none&security=tls&type=ws&host=us-node.example.com&path=/vless#US-Node-WARP
```

---

## 四、方案三：直连 + WARP 双出口共存（两个节点）

这一部分就是“同一台服务器，同时提供‘隐藏出口’和‘不隐藏出口’的两个节点”，方便按场景切换。

### 4.1 架构说明

- 入口：仍然是 Cloudflare Tunnel → `us-node.example.com:443`
- Xray 有两个 inbound：

  1. `18888` + `/vless` → 走 WARP 出口  
  2. `18889` + `/direct` → 走 freedom 直连出口  

- Cloudflare Tunnel 根据 path 分流：

  - `/vless` → `localhost:18888`
  - `/direct` → `localhost:18889`

---

### 4.2 Xray 配置（双入口 + 双出口）

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

> 注意：两个 inbound 使用同一个 UUID 即可，便于管理。

---

### 4.3 Cloudflare Tunnel 配置（按 Path 分流）

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
sudo systemctl restart xray
```

---

### 4.4 客户端两个节点示例

#### 4.4.1 WARP 出口节点（隐藏出口 IP）

```text
vless://UUID@us-node.example.com:443?encryption=none&security=tls&type=ws&path=%2Fvless&host=us-node.example.com&sni=us-node.example.com#US-WARP
```

#### 4.4.2 直连出口节点（不隐藏出口 IP）

```text
vless://UUID@us-node.example.com:443?encryption=none&security=tls&type=ws&path=%2Fdirect&host=us-node.example.com&sni=us-node.example.com#US-DIRECT
```

在客户端中即可看到两个节点，按需要切换。

---

## 五、开机自启配置

### 5.1 Xray 开机启动

```bash
sudo systemctl enable xray
```

### 5.2 Cloudflare Tunnel 开机启动

```bash
sudo systemctl enable cloudflared
```

### 5.3 WARP 服务

WARP 安装时会创建 `warp-svc`，默认随系统启动即可。如果需要手动控制：

```bash
sudo systemctl status warp-svc
```

---

## 六、几种代理方式的优缺点分析

这里对比三种使用方式：

1. **方案一：直连出口（freedom）**  
2. **方案二：WARP 出口**  
3. **方案三：直连 + WARP 同时存在**

---

### 6.1 方案一：直连出口（只隐藏入口 IP）

**优点：**

- 架构最简单，只有 Xray + Cloudflare Tunnel  
- 延迟相对最低（少一跳 WARP）  
- 故障点更少，排错容易  
- 所有协议和端口基本不受 WARP 限制

**缺点：**

- 目标网站看到的仍是 **云服务器 IP**  
- 如果服务器 IP 被封锁 / 频繁访问敏感服务，风险集中在单一 IP  
- 某些地区对云服务器 IP 有天然偏见或频繁风控

适用场景：

- 自己用的小流量  
- 对 IP 更换需求不高  
- 需要最稳定、最简单的架构

---

### 6.2 方案二：WARP 出口（隐藏出口 IP）

**优点：**

- 目标网站看到的是 Cloudflare WARP IP，而不是你的服务器 IP  
- 出口具有一定“池化”效果：IP 来自 WARP 池，相对不容易被单点封死  
- 对一些「只要不是云服务器 IP 就行」的服务更友好  
- 入口仍然由 Cloudflare Tunnel 保护，整体较隐蔽

**缺点：**

- 多了一层代理 → 延迟略有增加  
- WARP 也可能因流量、地区、内容类型受到风控或限速  
- 某些协议或端口可能在 WARP 上有不可预知的表现（例如非 80/443）  
- 增加了一个故障点（WARP 自身如果连不上，整体就会降级）

适用场景：

- 希望**最大限度隐藏服务器真实出口 IP**  
- IP 容易被封的服务 / 地区  
- 接受稍高延迟，换取更高隐蔽性

---

### 6.3 方案三：直连 + WARP 共存（双节点）

**优点：**

- 灵活性最高：  
  - 需要“干净 IP”时用 WARP 节点  
  - 需要最低延迟、最高稳定性时用直连节点  
- 临时出现 WARP 问题时可以快速切到直连  
- 同一域名、同一 Tunnel 内完成分流，无需多台服务器

**缺点：**

- 配置相对更复杂（多 inbound + 多 path + 多节点）  
- 运维时要区分是哪一路出问题（Xray / Tunnel / WARP / 客户端）  
- 比方案一略多占用少量内存（多一个 inbound）

适用场景：

- 想要“一台服务器，多种出口策略”  
- 有不同地区/场景需要切换 IP 特征  
- 常年使用，方便日常 A/B 比较哪条线路更适合某些网站

---

## 七、总结

通过 **Xray + VLESS + WebSocket + Cloudflare Tunnel** 的组合，你可以轻松实现：

- **隐藏入口 IP**：外界只看到 Cloudflare 节点，而不是你的服务器  
- **可选隐藏出口 IP**：配合 WARP，将出口 IP 伪装为 Cloudflare 官方 IP  
- 使用标准 `443 + TLS + WebSocket`，整体流量与普通 HTTPS 高度相似

可以根据自身需求选择：

- **只直连出口**：简单、稳定、延迟低  
- **只 WARP 出口**：入口 + 出口双重隐藏  
- **直连 + WARP 共存**：按场景切换，灵活性最高  

实际部署时，只需牢牢记住三点：

1. **Xray：**  
   - 本地监听 `127.0.0.1:<port>`，协议 `VLESS + WS`  
2. **Cloudflare Tunnel：**  
   - 将域名流量转发到 `http://localhost:<port>`  
   - 如需多节点，可用 path 分流  
3. **客户端：**  
   - 使用 `VLESS + TLS + WS`，域名、路径、端口与服务端配置严格一致  

这样，你就拥有了一套相对稳健、隐蔽、且可扩展的代理接入方案。
