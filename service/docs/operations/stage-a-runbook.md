# 阶段 A 运行手册（Runbook）

本文面向部署/运维：如何启动本地 Compose 栈、建库迁移、创建初始管理员、模型只读挂载、
隐私清理与备份恢复。所有服务与队列命名以 `stage_a` 隔离，避免污染未来阶段。

## 0. 前置

- Docker Engine + Compose v2（本仓库不提交任何模型权重，模型来自原闭集实验项目）
- 实验侧只读路径（默认仓库根下的 `项目/`）
  - `项目/Files`：`herb_authentication` 推理包（含 `InferenceService`）
  - `项目/Models/best_multitask_model.calibrated.pth`：实验闭集 checkpoint（`experimental_closed_set=true`、`open_set_calibrated=false`）

> 请勿把原项目或权重复制进本仓库。API 容器不挂载模型；只有 Worker 只读挂载。

## 1. 一键启动

```bash
# 环境变量示例
cp .env.example .env            # 若需本地直连开发再修改；Compose 使用自身 environment

# 从仓库根目录运行
docker compose -f infra/docker-compose.yml up -d --build
docker compose -f infra/docker-compose.yml ps
```

- 入口：`http://localhost:8080`（nginx；`/api/` 转发 API，其余转发 Web）
- MinIO 控制台：`http://localhost:9001`（user `drynaria-minio` / pass `minio-stage-a-local-only`）
- PostgreSQL：`localhost:5432`；Redis：`localhost:6379`

健康检查就绪后 API 自动执行 `alembic upgrade head` 建表；随后 Worker 与 Web 启动。

## 2. 数据库迁移（不依赖 Compose 也可手动执行）

```bash
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt   # 首次
# 设 DATABASE_URL 后：
set DATABASE_URL=postgresql+psycopg://drynaria:...@localhost:5432/drynaria_stage_a
alembic -c apps/api/alembic.ini upgrade head
# 生成新迁移（改动 models 后）：
alembic -c apps/api/alembic.ini revision --autogenerate -m "describe change"
```

## 3. 创建账户

公开注册只能创建 `user`。专家/管理员由管理员创建：

```bash
# 初始管理员（容器内执行；镜像未安装 editable 入口点，用 python -m 调用）：
docker compose -f infra/docker-compose.yml exec api python -m drynaria_api.cli create-admin --email admin@example.com --password 'STRONG-pass'

# 或用已装好的本地环境：
.venv\Scripts\python.exe -m drynaria_api.cli create-admin --email admin@example.com
```

管理员登录后用 `POST /api/v1/admin/users`（或前端）创建 `expert`/`admin`。

## 4. 本地（无容器）开发

```bash
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
copy .env.example .env        # 调整 DATABASE_URL 等

# 本地无 PostgreSQL 时可用 SQLite 快速跑：
set DATABASE_URL=sqlite:///./var/dev.db
set STORAGE_BACKEND=local
set CELERY_BROKER_URL=        # 留空则不投递任务（直接调用 worker 函数或改设 redis）
set AUTO_CREATE_TABLES=true
.venv\Scripts\python.exe -m drynaria_api.cli schema-create   # 仅开发

# API
uvicorn drynaria_api.main:app --reload --port 8000   # cwd = apps/api（或 PYTHONPATH=apps/api）
# 前端
npm --prefix apps/web run dev                          # http://localhost:5173（代理 /api 到 8000）
# Worker（需要原项目 venv 具备 torch 或本环境安装 torch）
celery -A drynaria_api.worker:celery_app worker -Q stage_a_analysis -c 1 -l info
```

### 4.1 本地预览：无 broker（var/dev-preview.db + 本地轮询 Worker）

以下为“无 Celery broker 的本地预览”精确命令。本仓库验证环境为
`var\venv-stage-a`（解释器 `var\venv-stage-a\Scripts\python.exe`，未安装 editable
入口点，故用 `python -m` 且设置 `PYTHONPATH=apps\api`）。所有终端都在**仓库根目录**
执行，保证 `DATABASE_URL=sqlite:///./var/dev-preview.db` 指向同一 `var/dev-preview.db`、
`STORAGE_BACKEND=local` 落到同一 `var/local-storage`。

首次建表（已建过可跳过）：

```bat
set PYTHONPATH=apps\api
set DATABASE_URL=sqlite:///./var/dev-preview.db
set STORAGE_BACKEND=local
set CELERY_BROKER_URL=
var\venv-stage-a\Scripts\python.exe -m drynaria_api.cli schema-create
```

终端 1 — API（该进程永不加载模型）：

```bat
set PYTHONPATH=apps\api
set DATABASE_URL=sqlite:///./var/dev-preview.db
set STORAGE_BACKEND=local
set CELERY_BROKER_URL=
var\venv-stage-a\Scripts\python.exe -m uvicorn drynaria_api.main:app --port 8010
```

终端 2 — 本地开发 Worker：轮询数据库中的 pending 分析任务并逐个执行
（无 broker 时替代 Celery 成为消费者，模型只在本进程加载）：

```bat
set PYTHONPATH=apps\api
set DATABASE_URL=sqlite:///./var/dev-preview.db
set STORAGE_BACKEND=local
set CELERY_BROKER_URL=
var\venv-stage-a\Scripts\python.exe -m drynaria_api.cli worker --poll-seconds 2
```

说明：

- Worker 的环境变量必须与 API 完全一致（同一 SQLite、同一本地存储目录）。
- 此前因没有消费者而卡在 queued 的 job，Worker 会在下一轮拾取执行；领取/幂等/
  崩溃恢复语义与 Celery 任务一致（复用 `run_analysis_job` 的原子领取）。
- `--once`：只处理一轮后退出，便于脚本/自检（本轮正常 = 退出码 0）。
- 模式守卫：`environment=prod` 或 `CELERY_BROKER_URL` 非空时命令拒绝启动
  （退出码 2），避免与 Celery 消费者竞争同一批 pending 任务。
- 真实模型推理需要带 torch 的环境，请按第 5 节准备；Ctrl+C 可随时优雅停止
  Worker（中断在提交前的任务回到 pending，下次启动会重新领取执行）。

### 本机已验证的真实模型 Worker（2026-09-09）

当前 `var/venv-stage-a` 没有 torch；实验侧 `项目/.venv/python.exe` 是带 torch 2.6.0 CPU 的 Conda Python。以下命令只读使用该环境，并从本项目验证环境补充 API 依赖。在 `service/` 目录运行，保持终端进程存活：

```powershell
$env:DATABASE_URL='sqlite:///./var/dev-preview.db'
$env:STORAGE_BACKEND='local'
$env:CELERY_BROKER_URL=''
$env:WORKER_DEVICE='cpu'
$env:PYTHONUNBUFFERED='1'
& '..\项目\.venv\python.exe' -B -c "import sys;sys.path.append(r'$PWD\var\venv-stage-a\Lib\site-packages');sys.path.insert(0,'apps/api');from drynaria_api.cli import main;raise SystemExit(main(['worker','--poll-seconds','2']))"
```

本次已验证：原有 queued 案件经真实模型完成；持续 Worker 启动后又自动消费一条任务，返回 succeeded。相关 14 项测试通过。开发预览必须同时运行 API、Web、Worker；仅启动 API/Web 会让案件停在 queued。停止 Worker 或重启机器后，需重新运行上述命令。

## 5. Worker 与模型说明

- Worker 进程内一次性装配 `InferenceService.from_checkpoint(...)`（进程级缓存，只加载一次并复用）。
- Worker 只把 `view_type=whole` 且质量合格的主图传给 `predict_experimental_closed_set()`。
- `detail` / `cross_section` 仅入库供复核与数据积累，绝不参与推理。
- API 进程不加载模型：模型中 import 链（torch）只在 Worker 任务内按需导入。
- 结果始终携带 `experimental_closed_set=true`、`open_set_calibrated=false`；
  `failed` 响应只含工程错误类别，绝不生成真伪结论。

模型目录只读挂载：

```yaml
# infra/docker-compose.yml（worker 服务）
volumes:
  - ${ORIGINAL_PROJECT_ROOT:-../项目}/项目/Files:/opt/inference:ro
  - ${ORIGINAL_PROJECT_ROOT:-../项目}/项目/Models:/models:ro
```

## 6. 运维例行

- 每日 PostgreSQL 备份：

```bash
docker compose -f infra/docker-compose.yml exec -T postgres pg_dump -U drynaria drynaria_stage_a | gzip > backup-$(date +%F).sql.gz
```

- MinIO 桶建议开启版本控制并定期恢复演练（阶段 A 基线：每日备份、周期恢复演练）。
- 隐私清理（删除申请立即生效；未授权研究原图在案件完成满 30 天后删除对象）：

```bash
.venv\Scripts\python.exe -m drynaria_api.cli purge-images --retention-days 30 --dry-run   # 先试跑
.venv\Scripts\python.exe -m drynaria_api.cli purge-images --retention-days 30             # 定时任务（建议每日）
```

- 生产环境：将 `infra/nginx/default.conf` 替换为 443 TLS 配置；`COOKIE_SECURE=true`；
  密钥只经环境变量注入（.env 不入库/不入镜像）。

## 7. 测试

```bash
# 后端（需要 Python 3.11-3.12）
.venv\Scripts\python.exe -m pytest apps/api/tests -v

# Web 单元
npm --prefix apps/web run test:unit
npm --prefix apps/web run build

# 端到端（需先启动完整 Compose 栈）
cd apps/web
npm i -D @playwright/test && npx playwright install chromium
npm run test:e2e
```

## 8. 已知限制（阶段 A）

- 限流仅在文档层预留；多实例部署需要 Redis 限流中间件（阶段 B）。
- 生产 HTTPS/TLS、CSRF 严格模式依赖真实域名与证书，需按 runbook 第 6 节配置。
- `closed_set_min_confidence=0.0` 默认不启用“无法判断”门槛；阈值应由阶段 C 校准数据确定。
- 定时清理建议以系统 cron/计划任务接入，容器内未内置 beat 调度器。
- **范围门控（scope，默认 `SCOPE_ENFORCEMENT=true`）**：只有 `scope=accepted` 的图片才给出
  公开真伪结论；`rejected`/`unavailable` 一律 inconclusive（不伪造真伪预测）。未配置
  `SCOPE_ARTIFACT_PATH` 工件时为 unavailable 门控。当前没有经独立标注且授权的真实
  out-of-scope 语料，故不部署任何真实范围工件——保守拒识不等于学到开放集（OOD）识别。
  范围训练/校准/评估及 JSON 工件绑定见 `Files/scope-pipeline-windows.md`。
