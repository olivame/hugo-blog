---
title: "Hugo 自动部署系统(Webhook + Caddy + GitHub)"
date: 2025-12-05T17:01:00+08:00
description: "使用 Webhook、Caddy 和 GitHub 搭建自动化部署 Hugo 博客的完整流程。"
tags: ["Hugo", "部署", "Caddy", "Webhook"]
---


本文记录如何构建一个 **自动从 GitHub 拉取并重新部署 Hugo 博客** 的完整系统。
<!--more-->
---

##  系统结构

```
GitHub → Webhook → 服务器监听端口 9000 → 执行 deploy.sh →
Hugo 重新构建 → Caddy 热重载 → 网站更新
```

---

##  1. 服务器目录结构

```
/home/ubuntu/
 ├── hugo-site/        # Hugo 博客目录（git clone）
 ├── deploy.sh         # 自动部署脚本
 ├── hooks.json        # Webhook 配置
 ├── webhook.log       # 日志（可选）
```

---

##  2. deploy.sh（自动部署脚本）

```bash
#!/bin/bash
echo ">>> Pulling latest GitHub changes..."
git -C /home/ubuntu/hugo-site reset --hard
git -C /home/ubuntu/hugo-site pull origin main

echo ">>> Building Hugo site..."
cd /home/ubuntu/hugo-site
/usr/bin/hugo

echo ">>> Reloading Caddy..."
sudo systemctl reload caddy

echo ">>> Deploy completed!"
```

---

##  3. Webhook 配置文件 hooks.json

```json
[
  {
    "id": "deploy-blog",
    "execute-command": "/home/ubuntu/deploy.sh",
    "command-working-directory": "/home/ubuntu",
    "response-message": "Deploy started"
  }
]
```

---

##  4. Webhook systemd 服务

路径：`/etc/systemd/system/webhook.service`

```ini
[Unit]
Description=Webhook Listener for Hugo Auto Deployment
After=network.target

[Service]
ExecStart=/usr/bin/webhook -hooks /home/ubuntu/hooks.json -port 9000 -verbose
Restart=always
User=ubuntu
WorkingDirectory=/home/ubuntu

[Install]
WantedBy=multi-user.target
```

启动：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now webhook
```

---

##  5. GitHub Webhook 设置

Settings → Webhooks：

- **Payload URL**：`http://你的IP:9000/hooks/deploy-blog`
- Content-Type：`application/json`
- 触发：Just the push event

必须保证 9000 端口外网可访问。

---

##  6. Caddy 配置（Hugo 站点）

路径：`/etc/caddy/Caddyfile`

```caddy
blog.olivame.xyz {
    root * /home/ubuntu/hugo-site/public
    file_server
    try_files {path} {path}/ /index.html
}
```

重载：

```bash
sudo systemctl reload caddy
```

---

##  系统完成

现在只要：

> push 到 GitHub → 网站自动同步

博客文章、主题修改、首页内容更新都会自动部署。

---
