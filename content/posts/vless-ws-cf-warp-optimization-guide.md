---
title: "VLESS + WebSocket + Cloudflare Tunnel + WARP 完整配置指南"
date: 2025-12-08T23:49:00+08:00
---

<!--more-->
---

# VLESS + WebSocket + Cloudflare Tunnel + WARP 性能优化实践指南

> 本文不是「从零搭建教程」，而是**在你已经搭好 VLESS + WS + Cloudflare Tunnel + WARP 双节点之后**，进一步做性能优化和排错的实战笔记。

- 适用读者：已经会搭 VLESS + WS + Cloudflare Tunnel，了解基本 Linux 运维
- 目标：在**不改变整体架构**的前提下，让链路更稳、更快、更可观测

---

## 0. 前提假设 & 环境说明

### 0.1 已有架构

你已经完成了以下基础架构（名字仅示例）：

- 域名：`example.com`
- 子域名：`us-node.example.com`
- 协议：**VLESS + WebSocket (WS)**  
- 入口：**Cloudflare Tunnel**（而不是直接 DNS 回源）
- 出口：
  - `/vless` → 走 WARP 出口（Cloudflare WARP，本地 SOCKS5：`127.0.0.1:40000`）
  - `/direct` → 走 freedom 直连出口（服务器原生 IP）

也就是说，你现在这台机子已经对外提供两类节点（示例）：

- **US-WARP**：VLESS + TLS + WS + CF Tunnel → Xray → WARP
- **US-DIRECT**：VLESS + TLS + WS + CF Tunnel → Xray → 直连

### 0.2 环境基本信息

- 系统：推荐 Debian / Ubuntu 20.04+
- 用户：示例用 `ubuntu`（root 同理）
- Xray：
  - 程序：`/usr/local/bin/xray`
  - 配置：`/usr/local/etc/xray/config.json`
  - 服务：`xray.service`
- Cloudflare Tunnel：
  - 程序：`/usr/bin/cloudflared` 或 `/usr/local/bin/cloudflared`
  - 配置：`/home/ubuntu/.cloudflared/config.yml`
  - 服务：`cloudflared.service`
- WARP：
  - 客户端：`cloudflare-warp`（`warp-cli` + `warp-svc`）
  - 模式：**proxy 模式**（本地 SOCKS5）

---

## 1. 优化思路：先“稳”，再“快”

这篇文章的核心理念：

1. **先确认「双节点都能正常使用」**
   - `/vless`（WARP 出口）节点可用
   - `/direct`（直连）节点可用
2. **在“可用”的前提下，分层优化：**
   - **系统层**：BBR、队列策略、缓冲区
   - **Tunnel 层**：QUIC、多连接、metrics 观测
   - **WebSocket 层**：Early-Data / 0-RTT 握手优化
   - **WARP 层**：只用 Proxy 模式、避免改乱路由
3. **可观测性**：用 `metrics`、`warp-cli status`、`curl` 等量化优化效果  

---

## 2. 系统网络层：开启 BBR + fq 队列

Cloudflare Tunnel 和 Xray 的传输最终跑在 Linux TCP/UDP 上，先把底层打好基础。

### 2.1 新建单独的 sysctl 配置文件

相比直接改 `/etc/sysctl.conf`，建议新建一个 `/etc/sysctl.d/99-bbr.conf`，方便回滚和管理。

```bash
sudo tee /etc/sysctl.d/99-bbr.conf >/dev/null << 'EOF'
# 启用 fq 队列 + BBR 拥塞控制
net.core.default_qdisc = fq
net.ipv4.tcp_congestion_control = bbr

# 可选：适当增大缓冲区和队列
fs.file-max = 1000000
net.core.rmem_max = 25000000
net.core.wmem_max = 25000000
net.core.netdev_max_backlog = 50000
net.core.somaxconn = 65535

net.ipv4.tcp_rmem = 4096 87380 25000000
net.ipv4.tcp_wmem = 4096 65536 25000000

# 可选：弱网环境友好一些
net.ipv4.tcp_mtu_probing = 1
net.ipv4.tcp_fastopen = 3
EOF

sudo sysctl --system
```

### 2.2 验证 BBR 生效

```bash
sysctl net.ipv4.tcp_congestion_control
# 期望输出：
# net.ipv4.tcp_congestion_control = bbr

lsmod | grep bbr
# 有输出表示 bbr 模块已加载
```

**常见坑：**

- 忘记 `sudo` / 权限不够，导致配置信息没真正写入；
- 内核过老（< 4.9）没有 BBR，需要升级内核；
- 其它地方（如面板）重复写 sysctl，互相覆盖。

---

## 3. Cloudflare Tunnel：保持简洁 + 打开 QUIC / 多连接可观测性

### 3.1 Cloudflare Tunnel 配置模板（config.yml）

配置文件示例路径：`/home/ubuntu/.cloudflared/config.yml`  

> 请把 `TUNNEL_ID`、路径、用户目录根据自己的实际情况替换掉。

```yaml
tunnel: TUNNEL_ID
credentials-file: /home/ubuntu/.cloudflared/TUNNEL_ID.json

# 可选：显式指定传输协议，视 cloudflared 版本而定
# protocol: quic

# 打开本地 metrics，方便观察 QUIC 多连接和 RTT
metrics: 127.0.0.1:20241

ingress:
  - hostname: us-node.example.com
    path: /vless
    service: http://127.0.0.1:18888

  - hostname: us-node.example.com
    path: /direct
    service: http://127.0.0.1:18889

  - service: http_status:404
```

**注意：**

- 这里的 `hostname` 必须与你在 Cloudflare 面板里绑定的子域名一致；
- `service` 指向的是 **本机 Xray 的监听地址**（通常是 127.0.0.1 + 对应端口）；
- `metrics` 是在本机开放一个 Prometheus 风格的指标端点。

### 3.2 systemd 服务配置（cloudflared.service）

确保 **systemd 启动命令和你手动测试成功的命令完全一致**。

```ini
# /etc/systemd/system/cloudflared.service

[Unit]
Description=Cloudflare Tunnel
After=network.target

[Service]
Type=simple
User=ubuntu
ExecStart=/usr/bin/cloudflared --config /home/ubuntu/.cloudflared/config.yml tunnel run
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
```

应用配置：

```bash
sudo systemctl daemon-reload
sudo systemctl enable cloudflared --now
sudo systemctl status cloudflared
```

如果成功，你会在日志中看到类似：

```text
Starting metrics server on 127.0.0.1:20241/metrics
Registered tunnel connection connIndex=0 ...
Registered tunnel connection connIndex=1 ...
Registered tunnel connection connIndex=2 ...
Registered tunnel connection connIndex=3 ...
```

### 3.3 使用 metrics 观察 QUIC 多连接表现

查看 metrics：

```bash
curl 127.0.0.1:20241/metrics | grep quic_client
```

你会看到类似：

```text
quic_client_total_connections 4

quic_client_latest_rtt{conn_index="0"} 10
quic_client_latest_rtt{conn_index="1"} 12
...

quic_client_congestion_window{conn_index="0"} 12345
...
```

可重点关注：

- `quic_client_total_connections`：总连接数，一般为 4 表示多路连接正常；
- `quic_client_latest_rtt` / `quic_client_smoothed_rtt`：延迟是否稳定、是否明显高于机房到 Cloudflare 边缘节点的正常 RTT；
- `quic_client_congestion_window`：拥塞窗口在负载下是否能够合理增长。

---

## 4. WebSocket 层：Early-Data / 0-RTT 握手优化

在 WebSocket 传输上，可以通过 `maxEarlyData` + `earlyDataHeaderName` 实现类似 0-RTT 的首包优化，减少一次往返。

> 不支持 Early-Data 的客户端不会受影响，连接依然正常，只是用不到这项优化。

### 4.1 Xray 双入口配置示例（含 Early-Data）

假设你的 Xray 配置路径为 `/usr/local/etc/xray/config.json`，这里给出**简化版**模板，重点在 `inbounds` 部分：

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
            "id": "YOUR-UUID-HERE"
          }
        ],
        "decryption": "none"
      },
      "streamSettings": {
        "network": "ws",
        "wsSettings": {
          "path": "/vless",
          "acceptProxyProtocol": false,
          "maxEarlyData": 2048,
          "earlyDataHeaderName": "Sec-WebSocket-Protocol"
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
            "id": "YOUR-UUID-HERE"
          }
        ],
        "decryption": "none"
      },
      "streamSettings": {
        "network": "ws",
        "wsSettings": {
          "path": "/direct",
          "acceptProxyProtocol": false,
          "maxEarlyData": 2048,
          "earlyDataHeaderName": "Sec-WebSocket-Protocol"
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

关键说明：

- `maxEarlyData: 2048`  
  - 允许客户端在握手阶段提前携带最多 2048 字节数据，减小首包延迟。
- `earlyDataHeaderName: "Sec-WebSocket-Protocol"`  
  - 使用一个正常的 WS 头字段承载 Early-Data，兼容 Cloudflare / Cloudflare Tunnel，避免早期数据被中间层丢弃。

修改配置后重启 Xray：

```bash
sudo systemctl restart xray
sudo systemctl status xray
```

如果你用支持 Early-Data 的客户端（如部分新版本的 Xray / sing-box），可以明显感觉到首包延迟有改善，尤其在高 RTT 或高丢包网络下。

---

## 5. WARP 层：坚持 Proxy 模式，避免改乱路由

### 5.1 WARP 安装与初始化

（如你已安装可略过此小节）

```bash
# 添加官方源（Ubuntu/Debian 示例）
curl https://pkg.cloudflareclient.com/pubkey.gpg   | sudo gpg --yes --dearmor --output /usr/share/keyrings/cloudflare-warp-archive-keyring.gpg

echo "deb [signed-by=/usr/share/keyrings/cloudflare-warp-archive-keyring.gpg]   https://pkg.cloudflareclient.com/ $(lsb_release -sc) main"   | sudo tee /etc/apt/sources.list.d/cloudflare-client.list

sudo apt update
sudo apt install cloudflare-warp -y
```

初始化并切换到 **Proxy 模式**：

```bash
sudo warp-cli registration new
sudo warp-cli mode proxy
sudo warp-cli proxy port 40000
sudo warp-cli connect
warp-cli status
```

你应该看到类似输出：

```text
Status update: Connected
Mode: proxy
Proxy port: 40000
Network: healthy
```

### 5.2 与 Xray 对接：统一出站走 WARP

在上面的 Xray 配置中，我们已经定义了：

```jsonc
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
]
```

和对应的 `routing`：

- `warp-in` → `warp-out`（WARP 出口）
- `direct-in` → `direct`（直连出口）

保证这几点：

1. `warp-cli mode` 一定是 **proxy**，而不是 **warp**（全局），否则会改写服务器路由，甚至导致 SSH 断连；
2. WARP 监听端口（例如 40000）必须和 Xray 配置一致；
3. 如果 WARP 出现 `Registration Missing` / `IPC timeout`，可以用下面的「急救包」重置：

```bash
sudo systemctl restart warp-svc

# 如果依然异常：
sudo rm -rf /var/lib/cloudflare-warp/*
sudo systemctl restart warp-svc

warp-cli registration new
warp-cli mode proxy
warp-cli proxy port 40000
warp-cli connect
```

---

## 6. 常见坑 & 排查套路

这一节可以直接在博客中单独列为「Troubleshooting」，很多人都会遇到类似问题。

### 6.1 cloudflared：手动运行正常，systemd 一启就退出

**典型症状：**

- 手动运行：

  ```bash
  /usr/bin/cloudflared --config /home/ubuntu/.cloudflared/config.yml tunnel run
  ```

  一切正常；  

- 但 `sudo systemctl start cloudflared` 后：

  - `systemctl status cloudflared` 显示 `active (exited)` 或不断 `activating (auto-restart)`；
  - `journalctl -u cloudflared` 中只出现一大串帮助说明，而非连接日志。

**常见原因：**

1. `ExecStart` 命令和你手动运行的不一样（少写了 `tunnel run`、路径错误等）；  
2. `User` 不对，导致 cloudflared 找不到 `config.yml` 或 `credentials-file`；  
3. 配置文件路径写成 root 的 HOME，而 `User=ubuntu`。

**建议排查步骤：**

```bash
journalctl -u cloudflared -n 50 --no-pager
```

然后：

- 把 `ExecStart=` 修改为 **你手动测试成功的那条命令**；
- 确保 `User=...` 和配置所在目录一致；
- `daemon-reload + restart` 后再看一次日志：

```bash
sudo systemctl daemon-reload
sudo systemctl restart cloudflared
sudo systemctl status cloudflared
```

### 6.2 WARP：Registration Missing / IPC timeout

**典型症状：**

- `warp-cli status` 显示 `Registration Missing due to: Daemon Startup`；
- 连续执行多个 `warp-cli` 命令后出现 `Error communicating with daemon`。

**修复套路：**

```bash
sudo systemctl restart warp-svc

# 如果还不行：
sudo rm -rf /var/lib/cloudflare-warp/*
sudo systemctl restart warp-svc

warp-cli registration new
warp-cli mode proxy
warp-cli proxy port 40000
warp-cli connect
warp-cli status
```

### 6.3 BBR：明明设置了，实际没启

**排查 checklist：**

1. `sysctl` 是否真写进去了？

   ```bash
   sysctl net.ipv4.tcp_congestion_control
   ```

2. 当前加载模块：

   ```bash
   lsmod | grep bbr
   ```

3. 内核版本是否支持：

   ```bash
   uname -r
   # 建议 >= 4.9
   ```

4. 是否有其它 panel 或脚本在覆盖 sysctl 设置？

   - 检查 `/etc/sysctl.conf`
   - 检查 `/etc/sysctl.d/*.conf`

### 6.4 路由被 WARP 或其它脚本改乱

如果你曾经用过 WARP 的全局模式，或者跑过“路由优化脚本”，建议先看一下路由表是否异常：

```bash
ip route
```

一般干净的云服务器上，只应该有：

- 默认路由指向机房网关（例如 `default via 10.x.x.x dev eth0`）；
- 本地网段路由；
- `169.254.0.0/16` 这种基础系统保留。

如果看到大量指向 Cloudflare 边缘 IP 段的静态路由，而你本来只想用 WARP Proxy 模式，那很可能是之前脚本留下的残留，需要手动删除或重启后恢复。

---

## 7. 总结：优化后的整体链路

在完成本文这些优化之后，你的链路大致具备：

1. **系统层：**
   - BBR 拥塞控制 + fq 队列，加快长肥管道下的收敛速度；
   - 合理的 socket buffer 和连接队列配置，提高并发能力。

2. **Tunnel 层：**
   - Cloudflare Tunnel 使用 QUIC 协议，多条到边缘的连接；
   - metrics 端点可以量化 RTT、拥塞窗口、连接质量，而不只是靠感觉。

3. **WebSocket 层：**
   - 通过 `maxEarlyData + earlyDataHeaderName` 实现 Early-Data / 0-RTT 首包优化；
   - 高延迟、高丢包环境下，握手时间明显缩短。

4. **WARP 层：**
   - 坚持使用 Proxy 模式，避免 WARP 改乱服务器路由；
   - 根据需要灵活选择出口：WARP（隐藏出口 IP）或 direct（最低延迟）。

5. **运维层：**
   - 知道从哪几层去排错：systemd 日志、WARP 状态、metrics 指标、路由表；
   - 重要配置都有模板可复用，迁移到新机器时只需替换 UUID / 域名 / Tunnel ID。
