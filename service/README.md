# 骨碎补中药材与饮片智能鉴定（实验 + 在线服务）

> Chinese Medicine Authentication — Drynaria fortunei authenticity: offline closed-set experiment + online service shell.
> 一个仓库装两件事：**`项目/` 离线实验（数据/训练/校准/评估/范围管线）** 和 **`service/` 在线服务
> （FastAPI + Vue 3 PWA + 容器编排）**。服务以只读方式调用实验侧的 `herb_authentication`
> 推理包，不复制权重、不复制训练代码。

> **这不是一个可以拿手机随手拍就能验真伪的成品。** 模型是**闭集二分类**（只认得"这一张图在
> 正品/伪品两类里更靠近哪一类"），**不具备拒识非本药材照片的能力**；默认范围门控下公开结论
> 一律为"无法判断"。详见 [已知限制](#已知限制务必先读)。

- 阶段：**A（实验性闭集 MVP + 在线服务壳，本仓库）** → B（开放集拒识 / 真实手机数据采集）→
  C（校准并晋升为正式版本）。
- 本仓库**不包含**训练图片（`项目/References/raw/`，427 张，本地数据；`.gitignore` 已排除）
  与模型权重（`项目/Models/*.pth`，`.gitignore` 已排除）。

## 目录

```text
项目/                        离线实验：数据清单、训练、校准、评估、晋升、范围门控管线
  Files/                     训练与推理代码（herb_authentication 包 + train/calibrate/evaluate/promote）
  Files/config_closed_set.yaml  闭集实验配置（路径均相对仓库根）
  Data/experiments/          数据清单与切分（metadata.csv / splits.json；分区 CSV 本地生成）
  Models/                    checkpoint（不进 Git：*.pth 全部忽略）
  References/raw/            训练图片（不进 Git：本地数据）
  Results/                   运行产物（不进 Git：本地可重建）
  docs/                      数据规范、实验记录、推理接口契约、设计规格与实施计划
  tests/                     实验侧测试
  requirements.txt pyproject.toml  实验侧依赖（torch 2.6 CPU / sklearn 等）
  .venv/                     实验侧 Python 环境（conda -p，本地；不进 Git）

service/                     在线服务：账户/案件/上传/幂等异步推理/结果/复核/隐私
  apps/api/drynaria_api/     FastAPI 后端（settings/database/models/schemas/security/audit/storage/
                             image_validation/cases/admin_reviews/worker/inference_adapter/privacy/cli）
  apps/api/drynaria_api/scope/  范围门控（manifest 校验/冻结特征/sklearn 训练校准评估/JSON 工件/运行时 gate）
  apps/api/tests/            后端测试 136 项（状态机/上传/幂等/only-whole 推理/权限/复核/隐私/迁移/容器配置）
  apps/api/alembic/          数据库迁移（c7a468151dba → b6f21a9c8d02 → c9d4e2f1a8b0）
  apps/web/                  Vue 3 + Vite + TS PWA（登录/案件/采集上传/等待轮询/结果/专家复核）
  infra/                     docker-compose.yml（PostgreSQL/MinIO/Redis/API/Worker/Web/Nginx）+ nginx + Dockerfile
  docs/operations/           运行手册、阶段 A 验收报告
  Files/                     范围清单模板、范围管线说明、范围门控验收报告
  start-local.cmd / .ps1      Windows 无 Docker 本地一键起停（隐藏后台进程 + 浏览器）
  stop-local.cmd / .ps1       只停止自己记录且身份校验通过的进程
```

## 任务

给定一张声称是**骨碎补**的药材照片，回答：**它在固定闭集实验里更像正品还是伪品**，
并落成一条可审计的业务记录。工程侧要解决的是可信与可追溯，不是模型能力：

1. **证据可信**：上传做真实解码校验（不信扩展名/MIME）、字节与像素上限、质量门控
   （模糊/纯色/过小 → needs_retake）；只有 `view_type=whole` 的主图进模型，
   `detail`/`cross_section` 只用于复核与数据积累。
2. **一次一结**：`analyze` 以（案件 + 主图指纹 + 模型版本 + 策略版本 + 范围指纹）为幂等键，
   并发重放返回同一个 `analysis_job_id`，不产生第二条预测。
3. **不伪造结论**：Worker 原子领取（条件 UPDATE + `WORKER_CLAIM_TIMEOUT_SECONDS` 崩溃重领），
   `predictions.analysis_job_id` 唯一索引保证一个 job 至多一条预测；失败只返回工程错误类别
   （`image_decode_failed` / `model_load_failed` / `prediction_failed`），绝不编造真伪结论。
4. **范围门控**：只有 `scope=accepted` 的图片才给公开真伪结论；`rejected`/`unavailable`
   一律 inconclusive。
5. **可审计**：审计日志、预测行不可变、专家复核追加版本、账户/CSRF/角色权限、隐私清理
   （原图保留期、删除请求、删除失败可重试且不谎报成功）。

## 数据

### 模型侧训练数据（`项目/`，本地）

固定数据集 `controlled_closed_set_v1`：**427 张 1920×1080 RGB JPEG**，正品 335 / 伪品 92，约 99.17 MB。

- 目录：`Drynaria_fortunei_authentic` 335、`Drynaria_fortunei_fake` 92；无 SHA-256 完全重复；
  EXIF 无相机型号；正品与伪品分别集中写入两个短时间窗口。
- 切分 `group-stratified-near-duplicate-components-v2`（求解 `optimal`）：
  train 235 正 / 64 伪 = 299，validation 50 / 14 = 64，locked-test 50 / 14 = 64。
- 防泄漏：8×8 average-hash（Hamming ≤ 4）全两两比较 + 传递闭包组件，按组件原子分配；
  134 条近重复边 / 328 个组件 / 36 个多成员 / 5 个混合标签 / 最大组件 36 张；
  独立 oracle 复核全部 90,951 对图片，跨分区距离 ≤ 4 为 0。
- 指纹（SHA-256）：`metadata.csv` `c93b3917…`、`train.csv` `1738c256…`、`val.csv` `870dac09…`、
  `test.csv` `d4ddb2f8…`、`splits.json` `931c7705…`。

**数据缺口（决定结论边界）**：缺 `batch_id`/实物样本 ID/来源/设备/拍摄会话/鉴定方法；伪品只有
真伪标签、**实际物种未经检验**；统一白底受控拍摄与真实手机照片分布差异明显。

### 范围（out-of-scope）语料

范围门控需要**独立标注授权**的 out-of-scope 语料，目前 **known=427 / unknown=0** —— 没有。
故不部署任何真实范围工件；默认（`SCOPE_ARTIFACT_PATH` 为空）即 `unavailable` 门控，
所有案件公开结果为"无法判断"。这是**保守拒识**，**不等于**学到开放集（OOD）识别能力。

### 服务侧

服务不发布数据集。用户上传图片按案件存储（本地磁盘或 S3/MinIO），保留期由
`IMAGE_RETENTION_DAYS`（默认 30 天）与用户删除/同意操作控制；**不会**用案件图片自动训练。

## 模型

| 项 | 值 |
|---|---|
| 形态 | 多任务视觉模型，服务只消费其闭集真伪头 |
| 代码 | `项目/Files/herb_authentication/`（训练/推理/校准/晋升全在仓库内） |
| checkpoint | `项目/Models/best_multitask_model.calibrated.pth`（**本地，不进 Git**） |
| checkpoint SHA-256 | `e25895c7dce0b1258377561af4a8a5bb73d1e8a6201c627b244d4470a13d1a95` |
| 训练 run | `20260825-drynaria-closed-set-v2`，9 epoch，patience 早停（最佳 epoch 3，约 150.6 s，CPU） |
| 选模公式 | `(validation authenticity Macro F1 + Balanced Accuracy) / 2` = `0.9099249061` |
| 类别平衡 | loss class weights `[0.6361702085, 2.3359375]`（`use_weighted_sampler=false`） |
| 校准 | validation-only 温度标定 T=`0.7948504334`（test 未参与调参） |
| 环境 | Python 3.12.13、PyTorch 2.6.0+cpu、torchvision 0.21.0+cpu、seed 42 |
| 可追溯 | 配置快照 SHA-256 `9e259b43…`、配置语义指纹 `6aeba8c8…`、训练时 Git HEAD `8acf46af…` |
| 特征维度 | 1280（范围管线绑定 checkpoint SHA-256 与 feature_dim） |

**运行时语义（服务侧强制）**：`experimental_closed_set=true`、`open_set_calibrated=false`
不符即判模型装配失败；`CLOSED_SET_MIN_CONFIDENCE=0.0` 默认不启用"无法判断"门槛
（这正是无法拒识非本药材的原因）；API 进程**不加载 torch/模型**，`InferenceService` 只在
Worker 进程按 checkpoint 路径做进程级单例加载。

## 结果

### 模型实验指标（固定数据集上的闭集二分类）

**Validation（64 张）** — 混淆矩阵 `[[44, 6], [0, 14]]`：

| 类别 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| authentic | 1.0000000 | 0.8800000 | 0.9361702 | 50 |
| counterfeit | 0.7000000 | 1.0000000 | 0.8235294 | 14 |

| 指标 | 点估计 | bootstrap 95% 区间 |
|---|---:|---|
| Macro F1 | 0.8798498 | [0.7971766, 0.9564626] |
| Balanced Accuracy | 0.9400000 | [0.8900000, 0.9800000] |
| AUROC | 0.9885714 | [0.9657143, 1.0000000] |
| PR-AUC | 0.9588976 | [0.8882178, 1.0000000] |
| calibrated ECE | 0.0611806 | [0.0386654, 0.1246833] |

**Locked-test（64 张，先写 reservation 再看 test logits，只评估一次、未用于调参）** —
混淆矩阵 `[[46, 4], [2, 12]]`：

| 类别 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| authentic | 0.9583333 | 0.9200000 | 0.9387755 | 50 |
| counterfeit | 0.7500000 | 0.8571429 | 0.8000000 | 14 |

| 指标 | 点估计 | bootstrap 95% 区间 |
|---|---:|---|
| Macro F1 | 0.8693878 | [0.7714286, 0.9564626] |
| Balanced Accuracy | 0.8885714 | [0.7814286, 0.9800000] |
| AUROC | 0.9800000 | [0.9471429, 1.0000000] |
| PR-AUC | 0.9387118 | [0.8543661, 1.0000000] |
| calibrated ECE | 0.0493175 | [0.0206831, 0.1124587] |

置信区间很宽（伪品仅 14 张，counterfeit recall 下界 `0.6428571`）。**单图返回的
`probabilities.authentic/counterfeit` 是该输入在闭集内的归一化输出（和为 1），不是实验准确率、
也不表示诊断确定性**；上表才是固定样本集上的实验指标。

### 服务层验收结果

| 项 | 结果 |
|---|---|
| 后端全量 pytest | **136 passed / 1 skipped**（skip 为无 Docker CLI 的容器渲染用例） |
| Alembic `upgrade head`（空 SQLite） | ✅ 11 张表 + 14 个索引，与 ORM metadata 逐表逐索引比对 **0 差异** |
| 运维 CLI 冒烟 | ✅ `create-admin`（argon2id + 审计）、`purge-images --dry-run` |
| 模型导入隔离 | ✅ API 模块不导入 `torch`/`sklearn`/`herb_authentication` |
| 范围管线自检（合成数据） | 8 passed —— **只证机械正确性，不代表识别准确率** |
| 真实特征提取冒烟 | ✅ 1280 维，与模型匹配 |
| 前端 `vue-tsc --noEmit` / `vitest` / `vite build` | ✅ 通过（11 tests） |
| 本地预览（SQLite + 本地存储 + 轮询 Worker） | ✅ 8 个已有案件公开结论无效、结果为 inconclusive |
| Docker Compose 整栈 / Playwright E2E | ⚠️ **未执行**（本机无 Docker CLI），需在具备 Docker 的机器上复核 |

第二轮修复的 7 项已核验缺陷（环境变量 list 解码致容器 API 启动崩溃、`/me` CSRF 轮换竞态、
`analyze` 幂等键并发竞态、Worker 重复领取并发推理、隐私清理失败计为成功、`CaseView` 加载
失败 loading 不复位、上传忽略配置上限）见
[`service/docs/operations/stage-a-acceptance-report.md`](service/docs/operations/stage-a-acceptance-report.md)。

## 怎么跑

### 0. 前置条件

- **Python `>=3.11,<3.13`**（torch 2.6 CPU 只提供该区间）。实验侧环境是
  `项目/.venv`（conda `-p` 建的 3.12，含 torch/sklearn）；**不要用系统 Python 3.13 建环境**。
- Node 18+（前端）与 Docker（整栈方案）。
- 训练图片 `项目/References/raw/` 与 checkpoint `项目/Models/*.pth` 是**本地资产**
  （不进 Git）；两者都没有时训练与真实推理都无法进行，但服务测试仍可跑（模型路径缺失会以
  `model_load_failed` 失败，不会给假结论）。

### 1. 实验侧：复现数据切分 / 评估

```powershell
cd 项目
\.venv\python.exe Files\prepare_closed_set.py --config Files\config_closed_set.yaml
\.venv\python.exe Files\evaluate.py --config Files\config_closed_set.yaml --device cpu --split locked_test
\.venv\python.exe Files\train.py --config Files\config_closed_set.yaml --device cpu --run-id <run-id>
```

配置里的路径（`项目/Data/...`、`项目/Models/...`）都相对**仓库根**，请在仓库根或
`项目/Files` 下按 `项目/docs/` 的运行手册执行。

### 2. 服务侧：只跑测试（不需要模型和权重）

```powershell
cd service
:: 直接使用仓库内已有的服务虚拟环境（3.12，已装 requirements-dev）
var\venv-stage-a\Scripts\python.exe -m pip install -r requirements-dev.txt
var\venv-stage-a\Scripts\python.exe -m pytest apps/api/tests -v
```

后端测试运行在 `STORAGE_BACKEND=local`、`S3_ENDPOINT_URL=`；仓库根 `conftest.py` 提供不依赖
系统临时目录的 `tmp_path`。前端：

```bash
npm --prefix apps/web ci
npm --prefix apps/web run typecheck
npm --prefix apps/web run test:unit
npm --prefix apps/web run build
npm --prefix apps/web run test:e2e    # 需完整栈 + E2E_WHOLE_FIXTURE 样例主图
```

### 3. 服务侧：容器整栈（PostgreSQL / MinIO / Redis / API / Worker / Web / Nginx）

```bash
cd service
cp .env.example .env      # 按需改密钥；模型/训练代码按 ORIGINAL_PROJECT_ROOT 只读挂载
docker compose -f infra/docker-compose.yml up -d --build
# 入口 http://localhost:8080
```

```bash
docker compose -f infra/docker-compose.yml exec api alembic -c apps/api/alembic.ini upgrade head
python -m drynaria_api.cli create-admin --email admin@example.com --password '...'
python -m drynaria_api.cli purge-images --retention-days 30 --dry-run
```

> `docker compose` 整栈与 Playwright E2E **尚未在具备 Docker 的机器上验收**；API 镜像默认
> **不装 torch**（由 `ARG INSTALL_TORCH` 控制，仅 Worker 需要）。

### 4. 服务侧：Windows 本地预览（无 Docker，双击即用）

```bat
start-local.cmd            :: 起 API(8010) + Web(5174) + Worker，并打开浏览器
start-local.cmd -CheckOnly :: 只校验前置条件与当前状态
start-local.cmd -SkipWorker:: 不启 Worker（案件会一直停在 queued）
stop-local.cmd             :: 只停止本工具记录且身份校验通过的进程
```

- 数据强制本地：SQLite `service/var/dev-preview.db` + 本地存储 `service/var/local-storage`，
  覆盖 `.env` 里的 PostgreSQL/Redis/S3 默认值；空库先 `python -m drynaria_api.cli schema-create`。
- Worker 是**无 Celery broker 的本地轮询消费者**，只读使用 `项目/.venv/python.exe`（带 torch）
  加载真实模型；日志写到 `service/Cache/local-services/`。

## 配置

见 [`service/.env.example`](service/.env.example)（无真实密钥）。与结论正确性直接相关的项：

| 变量 | 默认 | 作用 |
|---|---|---|
| `DATABASE_URL` | PostgreSQL | 本地预览被启动器覆盖为 SQLite |
| `STORAGE_BACKEND` | `auto` | `auto\|local\|s3`；S3 端点为空时落本地磁盘 |
| `INFERENCE_PROJECT_PATH` | `../项目/Files` | 只读引用实验侧 `herb_authentication` 包 |
| `MODEL_CHECKPOINT_PATH` | `../项目/Models/best_multitask_model.calibrated.pth` | checkpoint（不入库、不入镜像） |
| `CLOSED_SET_MIN_CONFIDENCE` | `0.0` | 为 0 时不启用"无法判断"门槛 |
| `SCOPE_ENFORCEMENT` / `SCOPE_ARTIFACT_PATH` | `true` / 空 | 空 → `unavailable` 门控 → 公开结论一律"无法判断" |
| `WORKER_CLAIM_TIMEOUT_SECONDS` | `3600` | Running 超时视为崩溃，可被原子重领 |
| `UPLOAD_MAX_BYTES` / `UPLOAD_MAX_PIXELS` | 20 MB / 4000 万 | 上传硬上限（真实解码后判定） |
| `IMAGE_RETENTION_DAYS` | `30` | 未授权研究原图保留天数 |

## 数据流向（一次分析的完整链路）

```text
浏览器上传 → 内容解码/字节与像素上限/质量门控 → 存储(local|S3) 落库
  → 只有 whole 主图参与 → POST /cases/{id}/analyze（幂等键，可并发重放）
  → AnalysisJob(pending) → Worker 原子领取(条件 UPDATE) → 范围门控
       ├─ accepted   → InferenceService.predict_experimental_closed_set()（进程级单例）
       ├─ rejected   → inconclusive（不推理）
       └─ unavailable→ inconclusive（不推理；默认状态）
  → Prediction 落库（每个 job 至多一条，唯一索引）
  → GET /cases/{id} → 结果展示：有有效范围证据才给真伪，否则"无法判断"
  → 专家复核追加版本（预测行不可变）/ 反馈 / 删除请求 / 保留期清理
```

主要端点：`GET /api/v1/health`、`GET /api/v1/status`；
`POST /api/v1/auth/register|login|logout`、`GET /api/v1/auth/me`（Cookie 会话 + CSRF，`/me` 只读回读不轮换）；
`POST|GET /api/v1/cases`、`GET /api/v1/cases/{id}`；
`POST /api/v1/cases/{id}/images`、`DELETE …/images/{image_id}`、`GET …/images/{image_id}/content`；
`POST /api/v1/cases/{id}/analyze`；
`POST /api/v1/cases/{id}/review-requests|feedback|consent|deletion-request`；
`GET /api/v1/admin/reviews…`、`POST …/decisions`、`GET /api/v1/admin/model-versions`；
`GET /api/v1/docs`、`/api/v1/openapi.json`。

## 范围门控管线

```powershell
cd service
python -m drynaria_api.cli scope-validate-manifest --manifest Files/scope-manifest-template.csv
python -m drynaria_api.cli scope-train --manifest <manifest.csv> ...
python -m drynaria_api.cli scope-verify --artifact <artifact.json>
```

清单校验（列 / `in_scope`|`out_of_scope` / `train`|`calibration`|`test` / `source` / `group_id` /
授权同意，含分组隔离与 SHA-256 重复检测）→ 冻结 encoder 特征 + sklearn → 只在 calibration
选阈值、只在 test 评估一次（95% Wilson 区间、受 max-false-accept/reject 约束）→ JSON 工件
（无 pickle，绑定 checkpoint SHA-256 与 feature_dim）→ 运行时 gate。当前**没有**经独立标注
授权的真实 out-of-scope 语料，因此未部署任何工件。说明见
[`service/Files/scope-pipeline-windows.md`](service/Files/scope-pipeline-windows.md) 与
[`service/Files/scope-rejection-acceptance-report.md`](service/Files/scope-rejection-acceptance-report.md)。

## 已知限制（务必先读）

1. **无法拒识非本药材**：闭集二分类天然总在两类里取 argmax；`closed_set_min_confidence=0.0`
   默认不启用门槛，非骨碎补（甚至非药材）的照片也会被硬分到"真/伪"之一。该能力属于阶段 B/C
   的开放集范围，**本项目不声称**通过 ML 修复了它。
2. **没有真实范围工件**：默认 unavailable 门控是**保守拒识**，**不等于**学到 OOD 识别。
3. **不识别伪品实际物种**：伪品只有真伪标签，实际物种未经检验；结果永不声称物种鉴定。
4. **不替代专业鉴定**：不替代药典/显微/理化/分子鉴定；晋升正式版本必须通过来源、批次、
   设备独立的手机 locked test。
5. **指标区间很宽**：locked-test 仅 64 张（伪品 14 张），counterfeit recall 下界 `0.6428571`；
   单图概率不是准确率。
6. **未验收项**：Docker Compose 整栈、Playwright E2E、Postgres 上的 Worker 原子领取与唯一索引
   行为（本机验证基于 SQLite）仍待在具备 Docker 的机器上复核。

## 仓库来源与数据资产说明

本仓库由 DSH 于 2026-09-09 从个人工作区 `E:\Desktop\WorkSpace`（一个混装多个课程项目的
单一仓库）中**整合**而来，是一个**全新的干净仓库**（单次初始提交，不带旧的多项目历史）：

- 实验代码原为 `Chinese_Medicine_Authentication/`，现为 `项目/`；
- 在线服务原为 `Chinese_Medicine_Authentication_v2/`，现为 `service/`；
- 旧目录在 `E:\Desktop\WorkSpace` 下**保留为只读参考副本**；
- 旧仓库中的 46 个中药实验提交（含数据切分、校准、晋升等决策轨迹）可在旧副本
  `git log` 中查阅，未带入本仓库历史。

**不进 Git 的本地资产**：`项目/References/raw/`（427 张训练图，约 99 MB）、
`项目/Models/*.pth`（权重）、`项目/Results/`（运行产物）、`项目/.venv/` 与
`service/var/`（本地环境与开发库）。
