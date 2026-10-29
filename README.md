# AUREL-Mem

**Anchored Unified Retrieval with Evidence Linking — Multimodal Memory**

AUREL-Mem v1.0 是提供同步 Add/Search API 的图文记忆研究原型。它保留原始图文与来源，用结构化语义、多视角词法检索、1024 维向量和有限上下文扩展检索证据。Search 返回原始证据，不生成最终问题答案。无需训练、显卡或本地模型权重。

本仓库是组合研究思想后的独立实现，不是任一论文仓库的服务封装。参考作者、原论文、固定源码提交、方法差异及许可证边界见 [研究披露](docs/research-disclosure.md) 和 [源码研究记录](docs/source-review.md)。没有继承论文成绩，尚无赛事官方 Smoke/Full 成绩。

## 快速运行

要求 Python 3.12。依赖锁定版本已在 macOS arm64 与 Linux 上验证；首次启动会创建 SQLite 数据库。

```bash
git clone https://github.com/Crosser-XDU/AUREL-Mem.git
cd AUREL-Mem
git checkout v1.0
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
cp .env.example .env
chmod 600 .env
```

编辑 `.env`，填写自己的模型 API Key。抽取和视觉查询统一使用 `gpt-4o-mini`；向量模型标识为 `text-embedding-v4`，请求显式携带 `dimensions=1024` 并检查响应维度。当前示例使用第三方兼容网关 `https://api.gpt.ge/v1`；调用库是 OpenAI 官方 Python SDK，但该网关不是 OpenAI 官方域名，`text-embedding-v4` 也不是 OpenAI 官方 embedding 模型。若网关没有相应模型权限，请配置独立的 `EMBEDDING_API_BASE` 和 `EMBEDDING_API_KEY`。

为 `MEMORY_API_TOKEN` 另外生成随机值，与模型 API Key 分开。可用以下命令生成后写入 `.env`：

```bash
python -c 'import secrets; print(secrets.token_urlsafe(32))'
python scripts/check_provider.py
MEMORY_HOST=127.0.0.1 bash scripts/run.sh
```

`run.sh` 默认监听 `0.0.0.0:18080`，可通过 shell 环境变量 `MEMORY_HOST` / `MEMORY_PORT` 覆盖。`ENV_FILE` 可指定配置文件；默认读取项目根目录 `.env`，默认数据库是 `data/memory.sqlite3`。保持单 worker 运行。

## API 与赛事表单

- Add：`POST https://YOUR-HOST/add`
- Search：`POST https://YOUR-HOST/search`
- 认证方式：`Authorization: Token`
- 记忆系统 Key：自己的 `MEMORY_API_TOKEN` 值，填写时不加 `Token ` 前缀。
- 系统名称：`AUREL-Mem`；版本名称：`v1.0`。
- 同时兼容 `Authorization: Bearer <MEMORY_API_TOKEN>`。未配置服务 token 时关闭鉴权，仅适合隔离的本地测试。

模型 API Key、记忆系统 Key、赛事颁发的 Eval Key 是不同凭证。请勿把任一真实 Key 或带凭证的 URL 放入公开仓库。本仓库不包含部署地址、真实配置、评测数据或私密日志。

示例 Header：

```http
Authorization: Token YOUR_MEMORY_SYSTEM_KEY
Content-Type: application/json
```

Add 请求：

```json
{"request_id":"example:chunk0","user_id":"example:user","session_id":"example:session","messages":[{"role":"user","timestamp":1704067200000,"content":"我的自行车是蓝色的。"}]}
```

Add 成功响应：

```json
{"success":true,"request_id":"example:chunk0","user_id":"example:user","session_id":"example:session"}
```

Search 请求（`top_k` 必填）：

```json
{"user_id":"example:user","query":"自行车的颜色","top_k":100}
```

Search 返回 `{"data":[{"id":"稳定ID","content":"原始证据与来源","score":0.8,"created_at":"2024-01-01T00:00:00Z"}]}`；分数仅为格式示例。无匹配结果时返回 `{"data":[]}`。图文消息及图片查询使用有序 `text` / `image_url` 内容数组，图片必须为 JPEG/PNG/WebP Base64 Data URI；多模态结果采用同种内容数组。

`request_id` 重试保持不变：相同用户、ID、载荷重复提交返回相同结果；同 ID 不同载荷返回 409。只有所有抽取和向量请求完成且事务提交后才返回成功。模型抽取格式不合约定时重试一次，仍失败则返回 503，不留下部分记忆；不会偷偷替换模型。`user_id` 是数据分区，不是独立用户登录身份，服务 Key 持有者拥有整个实例的接口访问权。

完整契约以 [AML 官方 API 指南](https://agentmemoryleaderboard.ai/api-guide) 为准。

## 方法结构

1. **原始层与语义层**：SQLite 保存有序文字、原图、来源、事件与写入时间；相同图片字节可去重，每次出现记录独立保存。
2. **结构化索引**：GPT-4o-mini 抽取事实、实体、视觉描述、OCR、关系、时间与有依据的状态变更。
3. **多路检索**：原文、语义、视觉/OCR 文字的 BM25 与 text-embedding-v4 向量排名经 RRF 融合。视觉向量来自图片的文字化表示，不是原始像素的联合编码。
4. **上下文与历史**：有限会话邻居、实体扩展、时间排序和状态历史；互补覆盖奖励与重复惩罚选择证据。
5. **原模态交付**：按条数与字节预算交付原始图文及来源，不回答最终问题。

## 容量与运行限制

| 项目 | 当前设置 |
|---|---|
| Add 消息数 | 每请求最多 1024 条 |
| HTTP 请求体 | 最多 48 MiB |
| 图片输入 | JPEG/PNG/WebP；单图最多 10 MiB、4000 万像素；不支持动画或远程图片 URL |
| 图片总量 | 单次查询/写入图片总和最多 30 MiB |
| 返回条数 | `top_k` 为 1–100；默认检索上限 32 条 |
| 返回预算 | 默认最多 12 张图、图片总和 30 MiB、100000 个文本字符 |
| 服务进程 | 单 worker；上游模型并发信号量默认为 4 |
| 上游超时 | 60 秒/次，SDK 有界重试；复杂 Add 可能涉及多批调用 |

达到输出预算会明确标记省略，原始数据库证据仍保留。每次搜索读取当前用户记录并在内存中计算排名，尚未完成 Full 规模容量/并发压测，不能承诺特定吞吐。建议平台初次 Smoke 使用 Add 并发 1、Search 并发 1，再根据实测调整。

## 部署

Linux/macOS 可直接运行 `scripts/run.sh`，置于可信 HTTPS 反向代理或隧道后。Linux 用户服务示例见 [systemd 模板](deploy/systemd/aurel-mem.service.example)；Mac 后台管理与临时公网隧道见 [macOS 部署](deploy/macos/README.md)。

临时 Cloudflare Quick Tunnel 不要求域名，但重建进程会换域名、没有可用性保证，不是固定赛事托管承诺。正式长时间评测应保证机器持续在线并维持一致的提交端点。[官方说明](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/)

## 验证

```bash
python -m pytest -q tests
```

真实链路核验会调用上游模型并产生用量。脚本读取环境中的 `MEMORY_API_TOKEN`，可通过下面的方式从 `.env` 加载：

```bash
python - <<'PYTHON'
from dotenv import load_dotenv
import runpy
load_dotenv()
runpy.run_path('scripts/smoke.py', run_name='__main__')
PYTHON
```

`smoke.py` 使用合成图像验证图文 Add、文字/图片 Search、原图回传、幂等及用户隔离；`verify_visual_dense.py` 核验图片独有 OCR、实际 1024 维向量及 dense-only 对照。合成测试不等于准确率 Benchmark 或赛事官方 Smoke。评测数据应按主办方要求保持私密，并在规定保留期限内由运营者清理。

实现与引用披露见 [docs/research-disclosure.md](docs/research-disclosure.md)。
