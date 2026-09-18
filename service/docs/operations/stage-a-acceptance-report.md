# 阶段 A 验收报告（Stage A Acceptance Report）

- 项目：骨碎补在线图像真伪鉴定 V2（`Chinese_Medicine_Authentication_v2`）
- 阶段：A — 实验性闭集 MVP（在线服务薄壳）
- 日期：2026-09-09（首轮收尾）；2026-09-10（第二轮：7 项已核验 acceptance finding 修复，本报告同步更新）
- 关联：`README.md`、`docs/operations/stage-a-runbook.md`

## 1. 验收范围与结论（当前状态）

| 项 | 结论 |
|----|------|
| 后端自动化测试 | ✅ 76 passed / 1 skipped / 0 failed（skip 为无 docker CLI 的容器渲染用例） |
| 数据库迁移（Alembic） | ✅ `upgrade head` 在全新 SQLite 上成功（含新增 revision `b6f21a9c8d02`）；与 ORM metadata 建表结果完全一致（11 张表 + 14 个索引，0 差异；`alembic_version` 除外） |
| 运维 CLI | ✅ `create-admin`、`purge-images --dry-run` 冒烟通过（SQLite + 本地存储） |
| 前端类型检查 | ✅ `vue-tsc --noEmit` 通过（含全部 `.vue` SFC 与新增单元测试文件） |
| 前端单元测试 / 生产构建 | ✅ 在非受限执行环境（Codex harness）通过：4 个原有用例全绿 + typecheck + build。本轮 DSH 沙箱 shell 内 esbuild 子进程被拦（spawn EPERM），`vitest run` 与 `vite build` 无法在此 shell 重跑（见 §3）；新增 3 个前端回归用例已通过 `vue-tsc` 类型检查，运行验证需在非受限 shell 复核 |
| Docker Compose 栈 | ⚠️ 未在本机执行（本机无 docker CLI）；容器配置文本级用例已通过（76 项中的 skip 即此用例），`docker compose config` 需在具备 Docker 的机器上复核 |
| 端到端（Playwright） | ⚠️ 需完整 Compose 栈 + `E2E_WHOLE_FIXTURE`，未执行 |

## 2. 验证明细

### 2.1 后端

- 命令（本沙箱环境；普通环境直接 `python -m pytest apps/api/tests -v` 亦可，根目录 `conftest.py` 提供了通用 `tmp_path`）：
  ```
  python -m pytest apps/api/tests -p no:cacheprovider -p no:tmpdir
  ```
- 覆盖：案件状态机、上传质量门控与 only-whole 推理保证、幂等分析（含并发竞态恢复）、结果映射、账户/CSRF/角色权限、CSRF 会话语义（/me 只读回读、不轮换）、专家复核追加版本、隐私清理（含删除失败计数）、Worker 原子领取/崩溃恢复/唯一预测、镜像与 Compose 配置静态约束、健康/状态端点、Settings 环境变量解码。

### 2.2 迁移

- Revision 链：`c7a468151dba`（initial schema）→ `b6f21a9c8d02`（`sessions.csrf_value` + `uq_predictions_analysis_job` 唯一索引，2026-09-10 新增）。
- 行为：`alembic -c apps/api/alembic.ini upgrade head` 在空库可建出全部 11 张表与索引；与 ORM metadata `create_all` 逐表/逐索引比对 0 差异（11 表 14 索引）。
- 说明：`alembic.ini` 的 `script_location` 为仓库根相对路径（`apps/api/alembic`），与 runbook/Compose 均从仓库根调用一致。

### 2.3 CLI

- `create-admin --email ... --password ...`：成功建管理员（argon2id 哈希、审计记录）。
- `purge-images --retention-days 1 --dry-run`：返回 `{'deleted_images': 0, 'affected_cases': 0, 'dry_run': True}`。

## 3. 本机（DSH 沙箱）环境限制与影响（已按实际修正）

- **`.venv` 只读**：DSH 文件沙箱拒绝向既有 `.venv/` 树写入（pip install 报 WinError 5）。验证使用仓库内 `var/venv-stage-a`（gitignored，已装 `requirements-dev.txt`）。在普通机器上按 README/runbook 使用 `.venv` 即可。
- **系统临时目录/`tempfile.mkdtemp` 目录不可写/不可枚举**：为此新增根目录 `conftest.py`，用仓库根 `testtmp/`（gitignored）下的普通 mkdir 目录提供 `tmp_path`；运行参数加 `-p no:cacheprovider -p no:tmpdir`（普通环境不加也兼容）。
- **esbuild 子进程在本 DSH shell 被拦（spawn EPERM）——这是会话沙箱限制，不是项目/机器限制**：vitest/vite build 需要 esbuild 以管道 stdio 派生子进程；受限 shell 返回 EPERM，且本会话无审批通道、无法提权（升级请求 fail-closed），故 `npm --prefix apps/web run test:unit` 与 `npm run build` 不能在本 shell 重跑。`vue-tsc`（不依赖 esbuild）在本 shell 正常且已通过。Codex（非受限 harness）独立跑通 4 个前端单测、typecheck 与 build，证明该限制仅存在于受限执行环境：
  ```
  npm --prefix apps/web run test:unit
  npm --prefix apps/web run build
  ```
- **无 Docker CLI**：Compose 渲染用例自动 skip；`docker compose -f infra/docker-compose.yml config` 与整栈启动需在具备 Docker 的机器复核（模型与原项目只读挂载路径按 runbook §0 提供）。

## 4. 修复/补齐清单

### 4.1 首轮（2026-09-09）

代码缺陷修复：

1. `apps/api/drynaria_api/auth.py`：`settings: Depends(...)` 三处误把 `Depends` 当作类型注解 → 改为 `settings: Settings = Depends(get_app_settings)`（FastAPI+pydantic v2 下会崩溃/解析错误）。
2. `apps/api/drynaria_api/settings.py`：`get_app_settings(request)` 未注解 `request`（且函数内局部 import 的字符串前向引用无法被 FastAPI 解析）→ 顶层导入 `Request` 并正确注解；否则被当作必填 query 参数导致 422。
3. `apps/api/drynaria_api/image_validation.py`：try 内抛出的 `UploadValidationError`（ValueError 子类）被 `except ValueError` 吞掉并误报 `corrupted_image`；模糊度量受 JPEG 边缘假边缘影响（纯色图方差 ≈206）→ 增加 `except UploadValidationError: raise` 分支并对质量统计裁边后再算方差。
4. `apps/api/drynaria_api/security.py`、`privacy.py`：SQLite 读回 naive 时间与 aware `now()` 比较抛 `TypeError` → 比较处补 UTC 时区（安全会话过期判断、隐私清理 Python 侧天数过滤）。
5. `apps/api/drynaria_api/cases.py`：`ready` 案件替换 whole 主图触发非法状态机转换（`next_status_after_upload` 只允许 Draft/NeedsRetake）→ 明确替换语义：质量合格保持 ready，不合格退回 needs_retake。
6. `apps/api/tests/test_security.py::test_logout_revokes_session`：登出需使用 `/me` 轮换后的新 CSRF（原“/me 每次轮换”安全设计，第二轮已改为只读语义，见 4.2）。
7. `apps/api/alembic.ini`：非 ASCII 注释在 GBK 区域导致 alembic 读配置失败 → ASCII 化；`script_location` 改为仓库根相对。

缺失交付物补齐：

8. `apps/api/alembic/versions/c7a468151dba_initial_schema.py`：初始迁移（此前 `alembic upgrade head` 无任何迁移可用）。
9. `docs/operations/stage-a-acceptance-report.md`：本报告（README 已引用此前缺失）。
10. 根目录 `conftest.py` + `.gitignore`（`testtmp/`、`.cache/`、`wprobe/`、`pytest-cache-files-*`）：DSH 沙箱下可复现的后端测试。

文档一致性修复：

11. `docs/operations/stage-a-runbook.md`：容器/本地 CLI 调用统一改为 `python -m drynaria_api.cli ...`（镜像未安装 editable 入口点，原 `drynaria-cli` 在容器/裸 venv 下不存在）。
12. `apps/api/tests/test_container_config.py`：`_services_block` 正则改用 `(?sm)` 跨行匹配（此前 LF/CRLF 下每行提前截断）；torch 断言改为校验“同一 Dockerfile 由 ARG INSTALL_TORCH 开关控制、api service 不启用”的既有设计（原断言与单 Dockerfile 方案矛盾）。

### 4.2 第二轮（2026-09-10）：7 项已核验 acceptance finding

1. **`Settings.allowed_origins` 环境解码**（`settings.py` + 新增 `apps/api/tests/test_settings_env.py`）
   - 根因：pydantic-settings 对 list 复杂字段在字段校验前先按 JSON 解码；`ALLOWED_ORIGINS=http://localhost:8080,http://127.0.0.1:8080`（Compose 实际格式）不是合法 JSON，直接在 settings 源解析阶段抛 `SettingsError`，既有 before 校验器根本没机会运行 → 容器 API 启动即崩。
   - 修复：字段标注 `Annotated[list[str], NoDecode]` 关闭自动 JSON 解码，把原始字符串交给 before 校验器；该校验器兼容逗号分隔、JSON 数组字符串与直接传 list 三种输入（并剥引号、跳过空项）。
   - 测试：从环境变量构造 Settings（Compose 同款格式）、JSON 数组兼容、默认值/init kwarg、以及用 Compose 同款 env 建 app 并打 /health。

2. **会话 CSRF 轮换竞态**（`models.py`、`security.py`、`auth.py`、新迁移 + 前端 `api.ts`/`router.ts`/`LoginView.vue`/`App.vue` 受益）
   - 根因：服务端 `/me` 每次调用都轮换 CSRF 明文（库里只存哈希、不轮换就无法回读）；路由守卫 `bootSession` 与 App 挂载各调一次 `/me` 时，服务端轮换两次、客户端只保存第一次（App 那次甚至不更新 token）→ 下一次状态变更携带的 token 已失配（403）；并发/顺序差异都会放大该竞态。
   - 修复（服务端）：`sessions` 增加 `csrf_value`（CSRF 明文，仅存服务端库），`issue_session` 写入；`GET /me` 改为只读回读当前 CSRF、不再轮换（旧行无明文时一次性轮换并持久化兜底）。CSRF 在登录/注册签发会话时绑定，会话内恒定。
   - 修复（前端，集中化会话更新）：`api.ts` 中登录/注册/`/me` 响应统一经 `adoptSession()` 更新客户端 token；`/me` 单飞（single-flight，`meRequest` 合并并发调用）；`router.bootSession` 与 `App.vue loadUser` 共用该路径，`LoginView` 不再自行 `setCsrfToken`。
   - 测试：后端 —— `/me` 连续调用返回同一稳定 token；`/me` “刷新”后的 token 用于下一次状态变更成功；登出撤销会话。前端 —— 登录 token 采纳 + `/me` 刷新 token 用于下一次 mutation 请求头；并发 `/me` 只发一个请求。

3. **analyze 幂等键并发竞态**（`cases.py`）
   - 根因：analyze 先 SELECT 再 INSERT 同键 `AnalysisJob`（键有唯一约束）；两个并发请求都查不到已有 job 时双双插入 → 后者唯一约束冲突未捕获 → 500，而不是幂等重放。
   - 修复：插入/提交捕获 `IntegrityError`，rollback 后重查胜者 job，走统一“幂等重放”收尾（补审计、同 job id、排队中补投递），绝不 500。
   - 测试：多个轮次、两个独立会话并发打同一案件 analyze —— 全部 200 且返回同一 `analysis_job_id`。

4. **Worker 可接受 Running 重复 job 并并发推理两次**（`worker.py`、`models.py`、新迁移 + `settings.py` 新增 `worker_claim_timeout_seconds`）
   - 根因：旧逻辑对 Running job 直接继续推理；领取只是对象级改状态（无条件 UPDATE），两个并发 Worker 都会通过检查并各自落一条 Prediction。
   - 修复：`run_analysis_job` 先执行**原子领取**——一条条件 UPDATE：`pending` 直接领取；`running` 且 `started_at` 超过 `worker_claim_timeout_seconds`（默认 1h）视为崩溃可重领；领取即 `status=running`、`attempts+1`、刷新 `started_at`。领取失败则跳过（终态幂等返回 / 新鲜 running 返回 `running` 不推理）。崩溃恢复显式处理：崩溃遗留的 stale running 可被重领并继续完成。
   - 兜底：`predictions.analysis_job_id` 新增**唯一索引**（`uq_predictions_analysis_job`，模型 + 迁移），数据库层保证一个 job 至多一条预测。
   - 测试：新鲜 Running 重复投递 → 跳过、不推理、attempts 不变；stale Running（2h 前）→ 重领并成功完成（attempts=2、1 条预测）；两个 Worker 并发投递 → 模型只调用一次、只落 1 条预测。

5. **隐私清理把失败的存储删除计为成功**（`privacy.py`）
   - 根因：`deleted_images += len(images)`、`affected_cases += 1`、审计 `image_count=len(images)` 都在删除循环之外，单条删除失败仍全量计成功并审计。
   - 修复：只对 `delete_bytes` 成功的对象累加计数并写 `object_deleted_at`；整案成功数为 0 时不审计、不计 affected；审计 detail 改为实际 `deleted_count`。失败的 image 保持 `object_deleted_at=NULL`，下次清理自动重试。
   - 测试：注入“前 1 次删除抛 StorageError”的存储 → 只统计 1 个成功、审计 `deleted_count=1`、失败对象未标记；存储恢复后下一轮删掉剩余对象。

6. **CaseView 加载失败后 loading 永不复位**（`apps/web/src/views/CaseView.vue`）
   - 根因：`loading` 只在初始值 true，`load()` 任何路径都不置 false → 首次加载失败后永远停留在“加载中…”。
   - 修复：`load()` 的 `finally` 统一 `loading=false`；非静默失败置错误信息并清空 detail（进入可终止的错误态卡片）；静默轮询失败保留已加载内容不打断等待/结果视图。
   - 测试：`tests/case-view.test.ts`（jsdom）—— mock `api.getCase` 拒绝后挂载组件：先见“加载中”，flush 后无“加载中”且出现 `.error` 错误文案。

7. **上传忽略配置的 max bytes/pixels**（`cases.py` 上传路由）
   - 根因：路由硬编码 `DEFAULT_MAX_BYTES + 1` 读取、`validate_upload_image(payload)` 用模块默认值，`Settings.upload_max_bytes/pixels`（以及部署 .env 注入值）完全不生效。
   - 修复：上传路由注入 `settings`，按 `settings.upload_max_bytes + 1` 限读，并把 `max_bytes=settings.upload_max_bytes`、`max_pixels=settings.upload_max_pixels` 传入校验。
   - 测试：构造收窄上限（4 KB / 4096 像素）的 app → 超字节返回 400 `file_too_large`；90×90 图超过像素上限返回 400 `too_many_pixels`。

## 5. Codex 独立复核（2026-09-09）

DSH headless 本轮修复进程已正常退出（exit code 0）。修复完成后，Codex 独立执行：

- 后端：`var/venv-stage-a/Scripts/python.exe -m pytest apps/api/tests -p no:cacheprovider -p no:tmpdir`，76 passed、1 skipped（缺少 Docker CLI），耗时 92.40 秒。
- 前端：`npm run test:unit`，3 个测试文件、7 项测试全部通过，包括新增会话与加载失败测试。
- 前端：`npm run typecheck` 与 `npm run build` 均通过。
- 本地预览：`http://127.0.0.1:5174/`；API 使用 `var/dev-preview.db` 独立 SQLite 开发库，监听 8010。前端代理 `/api/v1/health` 返回阶段 A 健康响应。
- 预览未配置 Redis/Worker，不代表真实模型异步推理已联调。完整栈、真实推理与浏览器端到端测试仍待验证。

以上结果取代本报告前文关于“新增前端测试运行时待复核”的历史状态。第二轮修复实际日期为 2026-09-09，前文 2026-09-10 为 DSH 报告日期误记。

## Windows 启动工具复核（2026-09-09）

DSH 实现启动工具主体，Codex 补充批处理代码页兼容与进程身份校验并独立验收：

- 配置自检通过：本地 SQLite、本地存储、空 broker；兼容已复制的 `.env`。
- `start-local.cmd -NoBrowser` 成功启动三个服务；启动命令退出后服务仍存活。
- 重复启动复用同一组三个 PID。
- `stop-local.cmd` 仅停止本工具记录且身份匹配的三个进程。
- 再启动成功，前端代理 `/api/v1/health` 返回正常阶段 A 响应。

网站当前保持运行。模型权重未修改，非药材输入的开放集拒识仍未实现。

## 6. 待正常环境复核项

- `docker compose -f infra/docker-compose.yml up -d --build` 整栈启动与入口 `http://localhost:8080` 冒烟。
- `npm run test:e2e`（需栈运行 + `E2E_WHOLE_FIXTURE` 提供样例主图）。
- Worker 真实推理（需原项目只读路径 + checkpoint 按 runbook §0 就位，API 进程不加载模型）。
- Postgres 上的 Worker 原子领取/唯一索引行为（本机验证基于 SQLite；条件 UPDATE 与唯一索引语义对 PG 同样成立）。
