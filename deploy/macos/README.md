# macOS 后台运行与临时公网入口

先按根 README 完成 Python 3.12、`.venv`、依赖和 `.env` 配置。`MEMORY_API_TOKEN` 必须是独立随机值。这里的运行文件全部写入仓库下 `.runtime/macos/`（已被 Git 忽略）。

从 [Cloudflare 官方渠道](https://developers.cloudflare.com/tunnel/downloads/) 安装 cloudflared，然后在仓库根执行：

```bash
mkdir -p .runtime/macos/bin
ln -s "$(command -v cloudflared)" .runtime/macos/bin/cloudflared
.venv/bin/python deploy/macos/manage.py start
.venv/bin/python deploy/macos/manage.py status
```

若希望登录自动启动，显式执行：

```bash
.venv/bin/python deploy/macos/manage.py install-login
```

服务入口是原生 `fusion_memory.app:create_app`，监听 `127.0.0.1:18080`，只启动一个 worker。`launchd` 管理两个进程，服务运行时通过 `caffeinate -i` 防止空闲睡眠。已有其它程序占用端口时，应先处理冲突，不要运行多个实例。

隧道分配地址后会生成 `.runtime/macos/current-endpoints.txt` 和 JSON。Add/Search 地址为 `/add`、`/search`，鉴权 Header 是 `Authorization: Token <MEMORY_API_TOKEN>`，也支持 Bearer。端点文件不包含 Key。域名生成不等于链路已验证，请从另一台机器进行真实 HTTP 检查。`/` 与 `/healthz` 提供不含密钥的公开状态；其它端点要求服务 Key。

```bash
.venv/bin/python deploy/macos/manage.py stop
.venv/bin/python deploy/macos/manage.py restart
.venv/bin/python deploy/macos/manage.py uninstall-login
```

`stop` 停止当前后台服务，但保留登录项；`uninstall-login` 只移除本脚本创建的两个登录项，不停止当前进程。`restart` 重新创建隧道，会更换公网域名。若仅修改模型配置，可只重启接口服务，保留隧道进程：

```bash
launchctl kickstart -k gui/$(id -u)/org.aurel-mem.local
```

Cloudflare Quick Tunnel 无需账号/域名，但没有可用性保证，重建会换地址，机器睡眠/退出登录/关机/断网会中断访问；最多 200 个在途请求，不支持 SSE。它适合临时联调，固定长期地址需另外配置命名隧道或 HTTPS 反向代理。[官方说明](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/)

不要将 `.env`、运行日志、SQLite 或凭证提交到 GitHub。生成的登录项引用此仓库绝对路径；移动仓库后需重新配置，不能直接删除仍在运行的代码目录。
