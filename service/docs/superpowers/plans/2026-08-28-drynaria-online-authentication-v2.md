# Drynaria Online Authentication V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `Chinese_Medicine_Authentication_v2` 中落地骨碎补在线图像鉴伪阶段 A 实验版，提供手机上传、异步闭集推理、结果查询和专家复核基础流程。

**Architecture:** 新项目作为在线服务壳，不复制或改写原闭集实验项目的训练代码；推理 Worker 通过只读路径依赖调用原项目 `herb_authentication.inference.service.InferenceService`。API 只负责账户、任务、上传、状态、审计和队列编排，模型只在 Worker 进程加载一次并复用。

**Tech Stack:** FastAPI, SQLAlchemy 2, Alembic, PostgreSQL, Redis, Celery, MinIO/S3, Vue 3, Vite, TypeScript, Vitest, Playwright, Docker Compose, Python 3.11-3.12.

**Spec:** `E:/Desktop/WorkSpace/Chinese_Medicine_Authentication/docs/superpowers/specs/2026-08-28-drynaria-online-authentication-v2-design.md`

## Global Constraints

- 阶段 A 只能发布实验性在线服务，所有模型结果必须保留 `experimental_closed_set=true` 和 `open_set_calibrated=false`。
- 阶段 A 只用 `view_type=whole` 的主图参与闭集推理；`detail` 和 `cross_section` 只进入审核与候选数据区。
- 阶段 A 不识别伪品实际物种，不证明未知药材拒识、炮制质量或真实手机场景准确率。
- API 进程不加载模型，不在路由中拼接领域结论，不重算概率。
- 推理 Worker 启动时加载并复用一份模型，调用 `InferenceService.predict_experimental_closed_set()`。
- 结果必须区分 `needs_retake`、`inconclusive`、`failed`、模型结论和专家结论；系统错误绝不生成真伪结论。
- 上传压缩大小上限 20 MB，解码像素上限 4000 万；允许 JPEG、PNG、WebP，但必须以实际解码结果为准。
- 对象键使用随机 UUID，不包含用户标识或原文件名；保存前删除非必要 EXIF。
- 本地账户使用 Argon2id 密码哈希；会话 ID 通过 `HttpOnly`、`Secure`、`SameSite=Lax` Cookie 传输，状态变更接口校验 CSRF token。
- 用户只能访问自己的任务和图片；专家、管理员账户由管理员创建。
- 未授权用于研究的原图在任务完成 30 天后删除；研究训练授权独立记录。
- 模型晋升到正式鉴伪版本必须通过来源、批次、设备独立的手机 locked test。

---

## File Structure

- Create: `pyproject.toml`，声明后端依赖、测试配置和格式化配置。
- Create: `requirements.txt`，锁定阶段 A 运行依赖。
- Create: `apps/api/drynaria_api/`，FastAPI 后端包。
- Create: `apps/api/drynaria_api/main.py`，应用入口、CORS、请求 ID、中间件。
- Create: `apps/api/drynaria_api/settings.py`，环境变量配置。
- Create: `apps/api/drynaria_api/database.py`，SQLAlchemy engine/session。
- Create: `apps/api/drynaria_api/models.py`，用户、任务、图片、分析、预测、复核、同意和审计表。
- Create: `apps/api/drynaria_api/schemas.py`，版本化 API schema。
- Create: `apps/api/drynaria_api/security.py`，密码哈希、会话、CSRF、角色校验。
- Create: `apps/api/drynaria_api/storage.py`，MinIO/S3 对象写入、短期签名 URL。
- Create: `apps/api/drynaria_api/image_validation.py`，上传大小、真实解码、像素、EXIF 清理和质量策略。
- Create: `apps/api/drynaria_api/cases.py`，用户任务、图片、分析和结果接口。
- Create: `apps/api/drynaria_api/admin_reviews.py`，专家审核接口。
- Create: `apps/api/drynaria_api/audit.py`，安全和业务事件记录。
- Create: `apps/api/drynaria_api/worker.py`，Celery app 与分析任务。
- Create: `apps/api/drynaria_api/inference_adapter.py`，只读加载原项目 `InferenceService`。
- Create: `apps/api/tests/`，后端单元与集成测试。
- Create: `apps/web/`，Vue 3 PWA。
- Create: `apps/web/src/`，页面、API client、状态管理和组件。
- Create: `infra/docker-compose.yml`，Nginx、Web、API、Worker、PostgreSQL、Redis、MinIO。
- Create: `infra/nginx/default.conf`，HTTPS 终止和反向代理模板。
- Create: `docs/operations/stage-a-runbook.md`，阶段 A 启动、备份、恢复和模型挂载说明。

## Solution Summary

推荐采用“薄 Web 壳 + 可审计异步推理”的方案，而不是把原实验项目直接改成 Web 应用。理由是原项目已经稳定承担数据、训练、校准、晋升和 Python 推理职责；在线系统新增的是身份、上传、队列、审计、对象存储和页面体验。把两者分开，能避免 Web 改动污染 closed-set 实验，也能让阶段 B/C 后续替换模型时只改 Worker 适配层。

阶段 A 最小闭环是：

1. 用户登录后创建 case。
2. 上传至少一张 `whole` 主图，可选上传 `detail` 与 `cross_section`。
3. 服务端真实解码并做基础质量检查，不合格进入 `needs_retake`。
4. `POST /api/v1/cases/{case_id}/analyze` 使用 `case_id + whole_image_sha256 + model_version` 做幂等键，只排队一次。
5. Celery Worker 读取 MinIO 原图，调用 `InferenceService.predict_experimental_closed_set()`。
6. Worker 保存不可变 `predictions`，并更新 case 当前状态。
7. 结果页分栏显示模型结果、实验范围、未参与判断的视角和专家复核入口。
8. 专家审核只追加 `expert_reviews` 版本，不覆盖历史模型预测。

## Task 1: Backend Scaffold And Configuration

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `apps/api/drynaria_api/__init__.py`
- Create: `apps/api/drynaria_api/settings.py`
- Create: `apps/api/drynaria_api/main.py`
- Create: `apps/api/tests/test_health.py`

**Interfaces:**
- Produces: `create_app() -> FastAPI`
- Produces: `Settings` with `database_url`, `redis_url`, `s3_endpoint_url`, `model_checkpoint_path`, `inference_project_path`, `allowed_origins`

- [ ] **Step 1: Write the failing health test**

```python
from fastapi.testclient import TestClient

from drynaria_api.main import create_app


def test_health_returns_stage_a_metadata():
    client = TestClient(create_app())

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "service": "drynaria-online-authentication-v2",
        "stage": "A",
        "experimental_closed_set": True,
        "open_set_calibrated": False,
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest apps/api/tests/test_health.py -v`

Expected: FAIL because `drynaria_api.main` does not exist.

- [ ] **Step 3: Implement minimal app and settings**

Create `apps/api/drynaria_api/settings.py`:

```python
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "drynaria-online-authentication-v2"
    database_url: str = "postgresql+psycopg://drynaria:drynaria@localhost:5432/drynaria"
    redis_url: str = "redis://localhost:6379/0"
    s3_endpoint_url: str = "http://localhost:9000"
    s3_bucket: str = "drynaria-stage-a"
    model_checkpoint_path: str = Field(default="")
    inference_project_path: str = Field(default="../Chinese_Medicine_Authentication/项目/Files")
    allowed_origins: list[str] = ["http://localhost:5173"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

Create `apps/api/drynaria_api/main.py`:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .settings import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "X-CSRF-Token"],
    )

    @app.get("/api/v1/health")
    def health() -> dict[str, str | bool]:
        return {
            "service": settings.app_name,
            "stage": "A",
            "experimental_closed_set": True,
            "open_set_calibrated": False,
        }

    return app


app = create_app()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest apps/api/tests/test_health.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml requirements.txt apps/api
git commit -m "feat: scaffold stage a backend"
```

## Task 2: Data Model And Case State Machine

**Files:**
- Create: `apps/api/drynaria_api/models.py`
- Create: `apps/api/drynaria_api/database.py`
- Create: `apps/api/drynaria_api/schemas.py`
- Create: `apps/api/tests/test_case_state_machine.py`

**Interfaces:**
- Produces: enums `CaseStatus`, `ViewType`, `AnalysisStatus`, `ReviewStatus`
- Produces: ORM tables `User`, `Case`, `CaseImage`, `AnalysisJob`, `Prediction`, `ReviewRequest`, `ExpertReview`, `ConsentRecord`, `AuditLog`
- Produces: `Case.can_transition_to(next_status: CaseStatus) -> bool`

- [ ] **Step 1: Write state transition tests**

```python
import pytest

from drynaria_api.models import CaseStatus, can_transition_case


@pytest.mark.parametrize(
    ("current", "next_status"),
    [
        (CaseStatus.Draft, CaseStatus.Ready),
        (CaseStatus.Ready, CaseStatus.Queued),
        (CaseStatus.Queued, CaseStatus.Analyzing),
        (CaseStatus.Analyzing, CaseStatus.Completed),
        (CaseStatus.Analyzing, CaseStatus.NeedsRetake),
        (CaseStatus.Analyzing, CaseStatus.Inconclusive),
        (CaseStatus.Analyzing, CaseStatus.Failed),
    ],
)
def test_allowed_case_transitions(current, next_status):
    assert can_transition_case(current, next_status)


def test_completed_case_does_not_transition_back_to_analyzing():
    assert not can_transition_case(CaseStatus.Completed, CaseStatus.Analyzing)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest apps/api/tests/test_case_state_machine.py -v`

Expected: FAIL because `models.py` does not exist.

- [ ] **Step 3: Implement enums and transition function**

```python
from enum import StrEnum


class CaseStatus(StrEnum):
    Draft = "draft"
    Ready = "ready"
    Queued = "queued"
    Analyzing = "analyzing"
    Completed = "completed"
    NeedsRetake = "needs_retake"
    Inconclusive = "inconclusive"
    Failed = "failed"


class ViewType(StrEnum):
    Whole = "whole"
    Detail = "detail"
    CrossSection = "cross_section"


class AnalysisStatus(StrEnum):
    Pending = "pending"
    Running = "running"
    Succeeded = "succeeded"
    Failed = "failed"


class ReviewStatus(StrEnum):
    Requested = "requested"
    InReview = "in_review"
    Completed = "completed"


ALLOWED_CASE_TRANSITIONS = {
    CaseStatus.Draft: {CaseStatus.Ready},
    CaseStatus.Ready: {CaseStatus.Queued},
    CaseStatus.Queued: {CaseStatus.Analyzing},
    CaseStatus.Analyzing: {
        CaseStatus.Completed,
        CaseStatus.NeedsRetake,
        CaseStatus.Inconclusive,
        CaseStatus.Failed,
    },
}


def can_transition_case(current: CaseStatus, next_status: CaseStatus) -> bool:
    return next_status in ALLOWED_CASE_TRANSITIONS.get(current, set())
```

- [ ] **Step 4: Add ORM models**

Implement SQLAlchemy 2 declarative models with UUID primary keys, `created_at`, `updated_at`, foreign keys, JSON fields for quality metrics and prediction payloads, and immutable prediction rows. Use table names from the spec: `users`, `cases`, `case_images`, `analysis_jobs`, `predictions`, `review_requests`, `expert_reviews`, `model_versions`, `consent_records`, `audit_logs`.

- [ ] **Step 5: Run tests**

Run: `pytest apps/api/tests/test_case_state_machine.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add apps/api/drynaria_api/models.py apps/api/drynaria_api/database.py apps/api/drynaria_api/schemas.py apps/api/tests/test_case_state_machine.py
git commit -m "feat: add case state model"
```

## Task 3: Upload Validation And Object Storage

**Files:**
- Create: `apps/api/drynaria_api/image_validation.py`
- Create: `apps/api/drynaria_api/storage.py`
- Create: `apps/api/tests/test_image_validation.py`

**Interfaces:**
- Produces: `validate_upload_image(payload: bytes) -> ValidatedImage`
- Produces: `ValidatedImage.sha256`, `width`, `height`, `format`, `sanitized_bytes`, `quality`
- Produces: `ObjectStorage.put_image(validated: ValidatedImage) -> StoredObject`

- [ ] **Step 1: Write failing validation tests**

Tests must cover empty bytes, corrupted bytes, size over 20 MB, a generated valid RGB PNG, and an image larger than 40,000,000 pixels.

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest apps/api/tests/test_image_validation.py -v`

Expected: FAIL because validation module does not exist.

- [ ] **Step 3: Implement validation**

Use Pillow actual decode. Save sanitized image bytes without carrying original EXIF. Return `needs_retake` quality flags for blur/exposure only when measurable; corrupted or oversized inputs raise validation errors and do not create prediction records.

- [ ] **Step 4: Run tests**

Run: `pytest apps/api/tests/test_image_validation.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/api/drynaria_api/image_validation.py apps/api/drynaria_api/storage.py apps/api/tests/test_image_validation.py
git commit -m "feat: validate and sanitize uploads"
```

## Task 4: Case API And Idempotent Analyze Endpoint

**Files:**
- Create: `apps/api/drynaria_api/cases.py`
- Modify: `apps/api/drynaria_api/main.py`
- Create: `apps/api/tests/test_cases_api.py`

**Interfaces:**
- Produces: `POST /api/v1/cases`
- Produces: `POST /api/v1/cases/{case_id}/images`
- Produces: `DELETE /api/v1/cases/{case_id}/images/{image_id}`
- Produces: `POST /api/v1/cases/{case_id}/analyze`
- Produces: `GET /api/v1/cases/{case_id}`

- [ ] **Step 1: Write tests for ownership, required whole image, and idempotency**

Include a test that two repeated analyze calls with the same `case_id`, whole-image SHA-256 and model version return the same `analysis_job_id`.

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest apps/api/tests/test_cases_api.py -v`

Expected: FAIL because routes do not exist.

- [ ] **Step 3: Implement routes**

Routes must not import torch or `InferenceService`. They create and query database rows, enqueue Celery tasks, and return versioned JSON. `failed` responses must contain engineering error categories only, never authenticity conclusions.

- [ ] **Step 4: Run tests**

Run: `pytest apps/api/tests/test_cases_api.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/api/drynaria_api/cases.py apps/api/drynaria_api/main.py apps/api/tests/test_cases_api.py
git commit -m "feat: add case analysis api"
```

## Task 5: Worker Inference Adapter

**Files:**
- Create: `apps/api/drynaria_api/inference_adapter.py`
- Create: `apps/api/drynaria_api/worker.py`
- Create: `apps/api/tests/test_inference_adapter.py`

**Interfaces:**
- Consumes: `InferenceService.from_checkpoint(path, device="auto")`
- Consumes: `InferenceService.predict_experimental_closed_set(image_bytes)`
- Produces: `run_analysis_job(job_id: UUID) -> None`

- [ ] **Step 1: Write tests with a fake inference service**

Fake service returns:

```python
{
    "model_version": "closed-set-v1",
    "authenticity": {"label": "authentic", "confidence": 0.91},
    "probabilities": {"authentic": 0.91, "counterfeit": 0.09},
    "actual_species_verified": False,
    "message": "此药品为正品，有91.0%的可能该药材为正品。",
    "model_fingerprint": "0" * 64,
    "data_fingerprint": "1" * 64,
    "split_fingerprint": "2" * 64,
}
```

Assert the stored prediction also includes `experimental_closed_set=true`, `open_set_calibrated=false`, source image ID, strategy version and analysis time.

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest apps/api/tests/test_inference_adapter.py -v`

Expected: FAIL because worker modules do not exist.

- [ ] **Step 3: Implement adapter and worker**

`inference_adapter.py` must add `Settings.inference_project_path` to `sys.path` only inside Worker startup, never in API routes. It must load the checkpoint once with `lru_cache` or a process-global singleton. `worker.py` must update case state from `queued` to `analyzing`, then to `completed`, `inconclusive`, `needs_retake`, or `failed`.

- [ ] **Step 4: Run tests**

Run: `pytest apps/api/tests/test_inference_adapter.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/api/drynaria_api/inference_adapter.py apps/api/drynaria_api/worker.py apps/api/tests/test_inference_adapter.py
git commit -m "feat: add asynchronous inference worker"
```

## Task 6: Auth, CSRF, Roles, Audit

**Files:**
- Create: `apps/api/drynaria_api/security.py`
- Create: `apps/api/drynaria_api/audit.py`
- Modify: `apps/api/drynaria_api/main.py`
- Modify: `apps/api/drynaria_api/cases.py`
- Create: `apps/api/tests/test_security.py`

**Interfaces:**
- Produces: `POST /api/v1/auth/register`
- Produces: `POST /api/v1/auth/login`
- Produces: `POST /api/v1/auth/logout`
- Produces: `GET /api/v1/auth/me`
- Produces: role dependency for `user`, `expert`, `admin`

- [ ] **Step 1: Write tests**

Cover Argon2id hash verification, user cannot access another user's case, public register cannot create expert/admin, and state-changing routes reject missing CSRF token.

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest apps/api/tests/test_security.py -v`

Expected: FAIL because security module does not exist.

- [ ] **Step 3: Implement security and audit**

Use opaque server-side session IDs, secure cookie options, CSRF token binding and audit rows for login, upload, analyze, review, data consent and deletion request actions.

- [ ] **Step 4: Run tests**

Run: `pytest apps/api/tests/test_security.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/api/drynaria_api/security.py apps/api/drynaria_api/audit.py apps/api/drynaria_api/main.py apps/api/drynaria_api/cases.py apps/api/tests/test_security.py
git commit -m "feat: add auth and audit controls"
```

## Task 7: Expert Review API

**Files:**
- Create: `apps/api/drynaria_api/admin_reviews.py`
- Modify: `apps/api/drynaria_api/main.py`
- Create: `apps/api/tests/test_admin_reviews.py`

**Interfaces:**
- Produces: `GET /api/v1/admin/reviews`
- Produces: `GET /api/v1/admin/reviews/{review_id}`
- Produces: `POST /api/v1/admin/reviews/{review_id}/decisions`
- Produces: `GET /api/v1/admin/model-versions`

- [ ] **Step 1: Write tests**

Cover role gating, review queue filters, appending a new expert decision without mutating prior predictions, and required fields `actual_species`, `authenticity_class`, `processing_status`, `verification_method`.

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest apps/api/tests/test_admin_reviews.py -v`

Expected: FAIL because admin review routes do not exist.

- [ ] **Step 3: Implement admin routes**

The review detail response must include all uploaded views, quality metrics, model output, previous review versions and signed image URLs. Expert decisions append rows with version numbers.

- [ ] **Step 4: Run tests**

Run: `pytest apps/api/tests/test_admin_reviews.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/api/drynaria_api/admin_reviews.py apps/api/drynaria_api/main.py apps/api/tests/test_admin_reviews.py
git commit -m "feat: add expert review workflow"
```

## Task 8: Vue PWA User Flow

**Files:**
- Create: `apps/web/package.json`
- Create: `apps/web/index.html`
- Create: `apps/web/src/main.ts`
- Create: `apps/web/src/api.ts`
- Create: `apps/web/src/App.vue`
- Create: `apps/web/src/views/LoginView.vue`
- Create: `apps/web/src/views/CaseListView.vue`
- Create: `apps/web/src/views/CaseCaptureView.vue`
- Create: `apps/web/src/views/CaseResultView.vue`
- Create: `apps/web/src/views/AdminReviewView.vue`
- Create: `apps/web/src/styles.css`
- Create: `apps/web/tests/case-flow.spec.ts`

**Interfaces:**
- Consumes: backend `/api/v1` routes from Tasks 4, 6 and 7.
- Produces: mobile-first PWA for login, case creation, upload, analysis, result, retake and review request.

- [ ] **Step 1: Write Playwright smoke test**

Cover login, create case, upload whole image fixture, analyze, see experimental scope text, and open review request.

- [ ] **Step 2: Run test to verify failure**

Run: `npm --prefix apps/web run test:e2e`

Expected: FAIL because web app does not exist.

- [ ] **Step 3: Implement PWA**

Use restrained work-focused UI. The first screen after login is the case workflow, not a marketing page. Result page must show: participating whole image, probabilities with closed-set explanation, non-participating detail/cross-section images, model version, strategy version, analysis time, professional identification disclaimer, retake action and expert review action.

- [ ] **Step 4: Run web tests**

Run: `npm --prefix apps/web run test`
Run: `npm --prefix apps/web run test:e2e`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/web
git commit -m "feat: add mobile pwa workflow"
```

## Task 9: Docker Compose And Operations

**Files:**
- Create: `infra/docker-compose.yml`
- Create: `infra/nginx/default.conf`
- Create: `apps/api/Dockerfile`
- Create: `apps/web/Dockerfile`
- Create: `.env.example`
- Create: `docs/operations/stage-a-runbook.md`
- Create: `apps/api/tests/test_container_config.py`

**Interfaces:**
- Produces: one-command local stack with Nginx, Web, API, Worker, PostgreSQL, Redis and MinIO.

- [ ] **Step 1: Write config tests**

Validate required environment variables, model path mount, separate API and Worker commands, and no secrets committed in `.env.example`.

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest apps/api/tests/test_container_config.py -v`

Expected: FAIL because infra files do not exist.

- [ ] **Step 3: Implement deployment files**

`docker-compose.yml` must mount the original model directory read-only into Worker only. API container must not receive the model mount unless a test proves it needs metadata. PostgreSQL, Redis, MinIO and queue namespaces must be named for stage A.

- [ ] **Step 4: Run verification**

Run: `pytest apps/api/tests/test_container_config.py -v`
Run: `docker compose -f infra/docker-compose.yml config`

Expected: PASS and compose config renders successfully.

- [ ] **Step 5: Commit**

```bash
git add infra apps/api/Dockerfile apps/web/Dockerfile .env.example docs/operations/stage-a-runbook.md apps/api/tests/test_container_config.py
git commit -m "chore: add stage a deployment stack"
```

## Task 10: End-To-End Acceptance

**Files:**
- Modify: `apps/api/tests/`
- Modify: `apps/web/tests/`
- Create: `docs/operations/stage-a-acceptance-report.md`

**Interfaces:**
- Consumes: all previous tasks.
- Produces: acceptance report showing stage A gates.

- [ ] **Step 1: Add acceptance tests**

Cover corrupted image, over-limit image, MIME spoofing, unauthorized case access, repeated analyze, worker failure producing `failed` without conclusion, and fixed golden image consistency with original `InferenceService`.

- [ ] **Step 2: Run full verification**

Run: `pytest apps/api/tests -v`
Run: `npm --prefix apps/web run test`
Run: `npm --prefix apps/web run test:e2e`
Run: `docker compose -f infra/docker-compose.yml config`

Expected: all commands pass.

- [ ] **Step 3: Write acceptance report**

The report must state which stage A gates passed, which require a real server run, and which are explicitly out of stage A scope.

- [ ] **Step 4: Commit**

```bash
git add apps/api/tests apps/web/tests docs/operations/stage-a-acceptance-report.md
git commit -m "test: verify stage a acceptance gates"
```

## Handoff To dsh

dsh should start with Task 1 and continue sequentially. Do not implement stage B/C model work in this pass. If a step reveals the plan conflicts with the approved spec, the approved spec wins. If the original project path differs on the target machine, update only `Settings.inference_project_path` and `.env.example`; do not vendor-copy the old training package into this repo.

Recommended first implementation slice for dsh:

1. Complete Tasks 1-5 to establish the backend and Worker inference loop.
2. Run backend tests.
3. Stop and report changed paths, test output, any assumptions, and whether the original checkpoint path exists locally.

## Self-Review

- Placeholder scan: no unresolved placeholder markers or intentionally blank implementation targets remain.
- Scope check: plan covers stage A only; stage B/C are recorded as future gates, not implementation work.
- Spec coverage: architecture, state machine, result semantics, API surface, security baseline, deployment, and testing strategy are mapped to tasks.
- Key ambiguity resolved: current `_v2` workspace is treated as the new online-service repository, while `Chinese_Medicine_Authentication` is treated as a read-only inference dependency.
