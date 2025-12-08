
---
title: "VLESS + WebSocket + Cloudflare Tunnel 完整配置指南 "
date: 2025-12-05T19:05:00+08:00
---

<!--more-->
---
> 本文为技术实践总结，所有示例中的域名、UUID、Tunnel ID 等敏感信息均已脱敏，仅作为配置参考使用。请务必替换为你自己的实际信息。

---

## 一、整体架构概览

**目标：** 在云服务器上搭建一套基于 **VLESS + WebSocket (WS) + Cloudflare Tunnel** 的代理体系：

| 组件               | 角色                     | 说明                                               |
| ------------------ | ------------------------ | -------------------------------------------------- |
| VLESS              | 代理协议                 | 轻量、高性能、无状态的加密传输协议                |
| WebSocket (WS)     | 传输层                   | 伪装为 HTTPS 流量，特征不明显，便于穿透与伪装     |
| Xray               | 核心程序                 | 实现 VLESS 协议和 WebSocket 传输                  |
| Cloudflare Tunnel  | 流量入口 / 反向代理      | 隐藏真实服务器 IP，提供免费 TLS 与全球加速        |

简要流程：

1. Xray 在服务器 `127.0.0.1:18888` 本地端口监听 VLESS + WS 流量；
2. Cloudflare Tunnel 将来自公网的 `443` 端口流量，通过 Cloudflare 网络回源到服务器本地端口；
3. 客户端通过 VLESS + TLS + WS 的方式接入 Cloudflare Anycast 网络。

---

## 二、服务器端：Xray 监听配置

### 1. 配置目标

在云服务器（如 GCP、AWS 等）上：

- 使用 Xray 在 **本地回环地址** `127.0.0.1` 的 `18888` 端口监听 VLESS + WS 流量；
- 不直接对外开放端口，由 Cloudflare Tunnel 负责公网访问。

### 2. 配置示例

> 配置文件路径示例：`/usr/local/etc/xray/config.json`

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

### 3. 核心操作步骤

1. **生成 UUID**：

   ```bash
   cat /proc/sys/kernel/random/uuid
   ```

2. 将生成的 UUID 填写到 `config.json` 中对应的 `id` 字段；  
3. 重启 Xray 服务：

   ```bash
   sudo systemctl restart xray
   ```

---

## 三、Cloudflare Tunnel 配置

Cloudflare Tunnel（`cloudflared`）负责将 Cloudflare 边缘节点收到的 HTTPS 流量，安全地回源到服务器本地端口，从而 **隐藏真实服务器 IP**。

### 1. 登录授权

安装 `cloudflared` 后，执行：

```bash
cloudflared tunnel login
```

浏览器中完成 Cloudflare 账号授权后，本地会生成相应的认证文件。

### 2. 创建 Tunnel

```bash
cloudflared tunnel create myproxy
```

成功后会得到：

- 一个 **Tunnel ID**：如 `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`
- 一个对应的 **凭证文件**（JSON），路径类似：

  ```text
  ~/.cloudflared/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx.json
  ```

> 实际路径请以命令输出为准。

### 3. 绑定域名（DNS 路由）

假设你在 Cloudflare 上有一个域名 `example.com`，希望使用子域名 `us-node.example.com` 作为入口：

```bash
cloudflared tunnel route dns myproxy us-node.example.com
```

执行后：

- Cloudflare 会为 `us-node.example.com` 创建一个 CNAME 记录，指向该 Tunnel；
- 目标网站只能看到 Cloudflare 边缘节点的 IP，而看不到你的服务器真实 IP。

### 4. Tunnel 配置文件

> 配置文件路径示例：`/etc/cloudflared/config.yml`

```yaml
tunnel: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx  # 替换为实际 Tunnel ID
credentials-file: /home/your-user/.cloudflared/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx.json

ingress:
  - hostname: us-node.example.com
    service: http://localhost:18888
  - service: http_status:404
```

含义说明：

- `hostname`: 外部访问使用的域名（必须与前面 DNS 绑定的一致）；
- `service`: 指明将流量转发到本地的 `http://localhost:18888`，即 Xray 监听端口；
- 最后一条 `http_status:404` 用于兜底未匹配到 hostname 的请求。

### 5. 配置 systemd 服务

> 服务文件路径示例：`/etc/systemd/system/cloudflared.service`

```ini
[Unit]
Description=Cloudflare Tunnel
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/cloudflared tunnel run
Restart=on-failure
User=your-user

[Install]
WantedBy=multi-user.target
```

> 请根据实际情况修改：
> - `ExecStart` 中 `cloudflared` 的绝对路径；
> - `User` 为运行 `cloudflared` 的系统用户；

### 6. 启动与验证

```bash
sudo systemctl daemon-reload
sudo systemctl enable cloudflared --now
sudo systemctl restart cloudflared
```

查看状态：

```bash
sudo systemctl status cloudflared
```

正常情况下可以看到：

- 服务为 `active (running)`；
- 日志中有 `Connection established` 类似字样，表示 Tunnel 已与 Cloudflare 建立连接。

---

## 四、客户端节点配置（最终使用）

当 Xray 与 Cloudflare Tunnel 均正常运行后，就可以在客户端（如 v2rayN、v2rayNG、Shadowrocket 等）中导入 VLESS 节点。

### 1. VLESS 链接示例（已脱敏）

```text
vless://xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx@us-node.example.com:443?encryption=none&security=tls&type=ws&host=us-node.example.com&path=/vless#US-Node
```

### 2. 关键参数说明

| 参数       | 示例值                          | 含义说明                            |
| ---------- | ------------------------------- | ----------------------------------- |
| UUID       | `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` | 客户端身份标识，需与 Xray 配置一致  |
| 地址       | `us-node.example.com`          | Cloudflare 上配置的域名             |
| 端口       | `443`                          | 使用标准 HTTPS 端口                 |
| 底层安全   | `tls`                          | 由 Cloudflare 提供证书与加密        |
| 传输协议   | `ws`                           | WebSocket                           |
| WS 路径    | `/vless`                       | 必须与 Xray 中 `wsSettings.path` 一致 |
| SNI / Host | `us-node.example.com`          | 一般填写为同一个域名                |

> 移动端客户端通常支持通过二维码导入，可以在桌面客户端中生成二维码后扫码导入。

---

## 五、优势与注意事项

### 1. 方案优势

- **隐藏真实服务器 IP**  
  服务器仅与 Cloudflare 建立出站连接，公网只暴露 Cloudflare 节点的 IP。

- **流量伪装优秀**  
  使用 `443 + TLS + WS`，整体特征与普通 HTTPS 网站接近，更难被简单特征识别。

- **免费 TLS 证书**  
  可依赖 Cloudflare 提供的 SSL/TLS 证书，无需自行在服务器上部署证书。

- **全球加速**  
  利用 Cloudflare 的 Anycast 网络，就近接入、回源你的服务器。

### 2. 注意事项

1. **带宽与流量计费**
   - Xray 使用 `freedom` 出站，最终访问目标网站时仍然通过你的云服务器出口；
   - 云厂商（如 GCP、AWS）的 **出站流量** 仍然会被计费，Cloudflare 不能帮你“免流量费”。

2. **出口 IP 问题**
   - 目标网站看到的 IP 是你的 **云服务器 IP**，而不是 Cloudflare IP；
   - 如有 IP 封锁或限速，仍然会体现到你的云服务器 IP 上。

3. **合规与风控**
   - 使用代理访问网络时，请遵守所在地相关法律法规与服务条款；
   - 合理控制流量规模，避免异常流量触发云厂商或 Cloudflare 风控。

4. **配置一致性**
   - 客户端的 UUID、WS 路径、TLS 域名等参数必须与服务器、Cloudflare 配置完全一致；
   - 修改任一环节配置后，记得同步更新客户端节点信息。

---

## 六、总结

通过 **Xray + VLESS + WebSocket + Cloudflare Tunnel** 的组合，可以实现：

- 服务器 IP 对公网隐藏，仅与 Cloudflare 交互；
- 使用标准 HTTPS 端口与传输方式，大幅降低被识别与干扰的风险；
- 借助 Cloudflare 的全球节点，实现更好的网络接入体验。

实际部署时，只需记住三点：

1. **Xray：** 本地监听 `127.0.0.1:18888`，协议 VLESS + WS；  
2. **Cloudflare Tunnel：** 将域名流量转发到 `http://localhost:18888`；  
3. **客户端：** 使用 `VLESS + TLS + WS`，域名与路径与服务端配置严格一致。

完成以上步骤，就拥有了一套相对稳健、隐蔽且易于迁移的代理接入方案。
