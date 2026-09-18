"""Celery Worker：消费 stage_a_analysis 队列并执行一次闭集分析任务。

只把 case 的 whole 主图传给模型；detail/cross_section 从不参与推理。
所有失败都写工程错误类别（error_category），绝不生成真伪结论。
本模块 import 不触发 torch；模型加载发生在任务函数内部（worker 进程）。

并发/崩溃语义：
- 每次执行先用一条条件 UPDATE 原子“领取”job（pending → running），同一 job
  同时只会有一个 Worker 通过领取检查；
- running 且 started_at 距今超过 worker_claim_timeout_seconds 视为上一执行者
  崩溃，可被重领（崩溃恢复）；
- predictions.analysis_job_id 上的唯一索引兜底：任何情况下一个 job 至多
  一条预测。
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable
from uuid import UUID

from sqlalchemy import and_, or_, select, update
from sqlalchemy.orm import Session, sessionmaker

from .database import make_session_factory
from .inference_adapter import InferenceError, predict_experimental_closed_set_bytes
from .models import (
    AnalysisJob,
    AnalysisStatus,
    Case,
    CaseImage,
    CaseStatus,
    ModelVersion,
    Prediction,
    ResultType,
    ViewType,
    apply_case_transition,
)
from .settings import Settings, get_settings
from .storage import StorageError, StorageBackend, build_storage
from .scope.gate import (
    SCOPE_ACCEPTED,
    SCOPE_REJECTED,
    SCOPE_UNAVAILABLE,
    ScopeDecision,
    ScopeGate,
    build_scope_gate,
)

logger = logging.getLogger(__name__)

_celery_app = None

# 供 `celery -A drynaria_api.worker:celery_app worker` 使用
def get_celery_app():
    global _celery_app
    if _celery_app is None:
        from celery import Celery

        settings = get_settings()
        app = Celery(
            "drynaria_stage_a",
            broker=settings.celery_broker_url,
            backend=settings.celery_result_backend,
        )
        app.conf.update(
            task_default_queue="stage_a_analysis",
            task_acks_late=True,
            worker_prefetch_multiplier=1,
            task_track_started=True,
        )
        _celery_app = app
    return _celery_app


def enqueue_analysis_job(job_id: UUID, settings: Settings | None = None) -> bool:
    """API 进程把 job 交给 Celery；无 broker（本地测试）时不排队，任务由测试直接执行。"""
    settings = settings or get_settings()
    if not settings.celery_broker_url:
        return False
    get_celery_app().send_task(
        "drynaria_api.worker.run_analysis_job_task",
        args=[str(job_id)],
        queue="stage_a_analysis",
        task_id=f"analyze-{job_id}",
    )
    return True


@get_celery_app().task(name="drynaria_api.worker.run_analysis_job_task")
def run_analysis_job_task(job_id: str) -> dict[str, Any]:
    return run_analysis_job(job_id)


def _claim_analysis_job(db: Session, job_id: UUID, stale_after: timedelta) -> bool:
    """原子领取：pending 直接领取；running 且超过崩溃窗口（started_at 过旧）可重领。

    条件 UPDATE 在数据库层完成比较与写入，两个并发 Worker 中至多一个
    rowcount=1。领取即把 status 置 running、attempts+1、刷新 started_at。
    """
    now = datetime.now(timezone.utc)
    stale_before = now - stale_after
    result = db.execute(
        update(AnalysisJob)
        .where(
            AnalysisJob.id == job_id,
            or_(
                AnalysisJob.status == AnalysisStatus.Pending,
                and_(
                    AnalysisJob.status == AnalysisStatus.Running,
                    or_(
                        AnalysisJob.started_at.is_(None),
                        AnalysisJob.started_at < stale_before,
                    ),
                ),
            ),
        )
        .values(
            status=AnalysisStatus.Running,
            attempts=(AnalysisJob.attempts or 0) + 1,
            started_at=now,
        )
    )
    return result.rowcount == 1


def compute_case_outcome(
    payload: dict[str, Any], min_confidence: float, scope_decision: ScopeDecision | None = None
) -> tuple[CaseStatus, ResultType]:
    """结果映射（纯函数）：模型标签 + 范围门控 -> 案件状态/结果类别。

    范围门控（本组件默认启用）：只有 scope=accepted 的图片才允许给出公开真伪结论；
    rejected / unavailable 一律落为 inconclusive，**不伪造**真伪预测。关闭门控时
    （scope_decision=None）保持旧的闭集映射行为。

    阶段 A 闭集二分类不识别伪品物种；counterfeit 一律映射为“疑似伪品”
    （completed 态展示，message 保留模型原文并提示人工复核）。
    """
    if scope_decision is not None and scope_decision.scope != SCOPE_ACCEPTED:
        return CaseStatus.Inconclusive, ResultType.Inconclusive
    authenticity = payload.get("authenticity") or {}
    label = authenticity.get("label")
    confidence = float(authenticity.get("confidence") or 0.0)
    if payload.get("experimental_closed_set") is not True:
        raise InferenceError("prediction_failed", "预测缺少 experimental_closed_set=true")
    if payload.get("open_set_calibrated") is not False:
        raise InferenceError("prediction_failed", "预测缺少 open_set_calibrated=false")
    if label not in {"authentic", "counterfeit"}:
        raise InferenceError("prediction_failed", "预测 authenticity label 不合法")
    if min_confidence > 0 and confidence < min_confidence:
        return CaseStatus.Inconclusive, ResultType.Inconclusive
    if label == "counterfeit":
        return CaseStatus.Completed, ResultType.SuspectedCounterfeit
    return CaseStatus.Completed, ResultType.AuthenticLikely


def run_analysis_job(
    job_id: UUID | str,
    *,
    session_factory: sessionmaker[Session] | None = None,
    storage: StorageBackend | None = None,
    predict: Callable[[bytes], dict[str, Any]] | None = None,
    settings: Settings | None = None,
    recorder: Callable[[Session], None] | None = None,
    claim_timeout_seconds: float | None = None,
    scope_gate: ScopeGate | None = None,
    feature_extract: Callable[[bytes], list[float]] | None = None,
) -> dict[str, Any]:
    """执行一个分析 job：原子领取 → 加载原图（仅 whole）→ 闭集预测 → 范围门控 → 不可变存储。

    可注入 session_factory / storage / predict / scope_gate / feature_extract 供测试使用
    （不加载真实模型）。``scope_gate`` 默认为 ``None``，此时按配置构建（可能为 unavailable）；
    ``feature_extract`` 为默认 None 时用冻结 encoder 特征提取（Worker 进程才带 torch）。
    """
    settings = settings or get_settings()
    session_factory = session_factory or make_session_factory(settings.database_url)
    storage = storage or build_storage(settings)
    if predict is None:
        predict = lambda payload: predict_experimental_closed_set_bytes(payload, settings)  # noqa: E731
    if scope_gate is None:
        scope_gate = build_scope_gate(settings)
    if feature_extract is None:
        feature_extract = _build_default_feature_extract(settings)
    stale_after = timedelta(
        seconds=float(
            claim_timeout_seconds
            if claim_timeout_seconds is not None
            else settings.worker_claim_timeout_seconds
        )
    )

    job_id = UUID(str(job_id))
    with session_factory() as db:
        if not _claim_analysis_job(db, job_id, stale_after):
            # 领取失败：另一执行者正在跑（running 且未超时），或 job 已终态/不存在
            job = db.get(AnalysisJob, job_id)
            if job is None:
                logger.warning("analysis job %s not found; skip", job_id)
                return {"job_id": str(job_id), "status": "not_found"}
            if job.status == AnalysisStatus.Succeeded:
                return {"job_id": str(job_id), "status": "succeeded"}
            if job.status == AnalysisStatus.Running:
                # 最近刚被其它 Worker 领取：跳过，绝不并发推理第二次
                return {"job_id": str(job_id), "status": "running"}
            return {"job_id": str(job_id), "status": job.status.value}

        job = db.get(AnalysisJob, job_id)
        if job is None:  # pragma: no cover —— 领取成功后 job 理论上必然存在
            logger.warning("analysis job %s vanished after claim; skip", job_id)
            return {"job_id": str(job_id), "status": "not_found"}
        case = db.get(Case, job.case_id)
        if case is None:
            _mark_job_failed(db, job, "case_state_invalid", "job 对应的 case 不存在", settings, case=None)
            return {"job_id": str(job_id), "status": "failed"}

        whole = db.get(CaseImage, job.whole_image_id)
        if whole is None or whole.view_type != ViewType.Whole or not whole.participates_in_inference:
            _mark_job_failed(db, job, "case_state_invalid", "缺少可参与的 whole 主图", settings, case=case)
            return {"job_id": str(job_id), "status": "failed"}

        # 案件状态推进：首次领取时 queued -> analyzing；崩溃重领时可能已是
        # analyzing（上一执行者在推进后崩溃）；其它状态视为非法。
        if case.status == CaseStatus.Queued:
            apply_case_transition(case, CaseStatus.Analyzing)
        elif case.status != CaseStatus.Analyzing:
            _mark_job_failed(
                db, job, "case_state_invalid", f"案件状态 {case.status.value} 不允许执行分析", settings, case=case
            )
            return {"job_id": str(job_id), "status": "failed"}

        try:
            image_bytes = storage.get_bytes(whole.object_key)
        except StorageError as exc:
            _mark_job_failed(db, job, "storage_read_failed", str(exc), settings, case=case)
            return {"job_id": str(job_id), "status": "failed"}

        try:
            payload = predict(image_bytes)
            scope_decision = _compute_scope_decision(
                scope_gate, feature_extract, image_bytes
            )
            case_status, result_type = compute_case_outcome(
                payload, settings.closed_set_min_confidence, scope_decision
            )
        except InferenceError as exc:
            _mark_job_failed(db, job, exc.category, str(exc), settings, case=case)
            return {"job_id": str(job_id), "status": "failed"}
        except Exception as exc:  # noqa: BLE001 —— 统一工程失败，不生成结论
            _mark_job_failed(db, job, "prediction_failed", f"{type(exc).__name__}: {exc}", settings, case=case)
            return {"job_id": str(job_id), "status": "failed"}

        authenticity = payload["authenticity"]
        scope_warning = _scope_warning_text(scope_decision) or payload.get("scope_warning")
        prediction = Prediction(
            case_id=case.id,
            analysis_job_id=job.id,
            image_id=whole.id,
            view_type=ViewType.Whole,
            model_version=payload.get("model_version") or job.model_version_hint,
            strategy_version=settings.strategy_version,
            experimental_closed_set=bool(payload["experimental_closed_set"]),
            open_set_calibrated=bool(payload["open_set_calibrated"]),
            authenticity_label=authenticity["label"],
            confidence=float(authenticity["confidence"]),
            probabilities=dict(payload["probabilities"]),
            actual_species_verified=bool(payload.get("actual_species_verified")),
            message=payload.get("message") or "",
            scope_warning=scope_warning,
            scope_decision=scope_decision.to_dict() if scope_decision is not None else None,
            model_fingerprint=payload.get("model_fingerprint") or "",
            data_fingerprint=payload.get("data_fingerprint"),
            split_fingerprint=payload.get("split_fingerprint"),
            payload=payload,
        )
        db.add(prediction)
        db.flush()

        existing = db.query(ModelVersion).filter(
            ModelVersion.model_version == prediction.model_version
        ).one_or_none()
        if existing is None:
            db.add(
                ModelVersion(
                    model_version=prediction.model_version,
                    checkpoint_sha256=prediction.model_fingerprint or None,
                    strategy_version=prediction.strategy_version,
                    experimental_closed_set=prediction.experimental_closed_set,
                    open_set_calibrated=prediction.open_set_calibrated,
                    data_fingerprint=prediction.data_fingerprint,
                    split_fingerprint=prediction.split_fingerprint,
                )
            )

        if case.status == CaseStatus.Analyzing:
            apply_case_transition(case, case_status)
        case.current_prediction_id = prediction.id
        case.result_type = result_type
        case.strategy_version = settings.strategy_version
        job.status = AnalysisStatus.Succeeded
        job.finished_at = _now_utc()
        job.error_category = None
        job.error_detail = None
        from .audit import record_audit

        record_audit(
            db,
            action="analysis.succeeded",
            actor_user_id=case.owner_id,
            resource_type="case",
            resource_id=str(case.id),
            detail={"job_id": str(job.id), "prediction_id": str(prediction.id)},
        )
        if recorder is not None:
            recorder(db)
        db.commit()
        logger.info("analysis job %s succeeded -> %s", job_id, case_status.value)
        return {"job_id": str(job_id), "status": "succeeded", "case_status": case_status.value}


def process_pending_analysis_jobs_once(
    *,
    session_factory: sessionmaker[Session] | None = None,
    storage: StorageBackend | None = None,
    predict: Callable[[bytes], dict[str, Any]] | None = None,
    settings: Settings | None = None,
    claim_timeout_seconds: float | None = None,
) -> dict[str, int]:
    """本地无 broker 模式的一轮轮询：处理所有待执行的 AnalysisJob。

    供 `python -m drynaria_api.cli worker`（独立于 API 的独立进程，模型只在
    本进程加载）周期性调用；也供测试注入假 predict 直接调用。挑选 pending
    （以及超过领取超时的 stale running，即上一执行进程崩溃遗留）的 job 交给
    run_analysis_job——领取与幂等语义全在 run_analysis_job 内原子完成：正在被
    其它执行者处理的新鲜 running 任务不会被再次推理。predict=None 时走真实
    inference adapter（默认）。

    单个任务抛出的异常只记录并计入 failed，不会中断本轮其它任务。
    返回 {"attempted": N, "succeeded": N, "failed": N, "skipped": N}。
    """
    settings = settings or get_settings()
    session_factory = session_factory or make_session_factory(settings.database_url)
    stale_before = datetime.now(timezone.utc) - timedelta(
        seconds=float(
            claim_timeout_seconds
            if claim_timeout_seconds is not None
            else settings.worker_claim_timeout_seconds
        )
    )
    with session_factory() as db:
        candidate_ids = list(
            db.scalars(
                select(AnalysisJob.id)
                .where(
                    or_(
                        AnalysisJob.status == AnalysisStatus.Pending,
                        and_(
                            AnalysisJob.status == AnalysisStatus.Running,
                            or_(
                                AnalysisJob.started_at.is_(None),
                                AnalysisJob.started_at < stale_before,
                            ),
                        ),
                    )
                )
                .order_by(AnalysisJob.created_at)
            ).all()
        )

    summary: dict[str, int] = {
        "attempted": len(candidate_ids),
        "succeeded": 0,
        "failed": 0,
        "skipped": 0,
    }
    for job_id in candidate_ids:
        try:
            outcome = run_analysis_job(
                job_id,
                session_factory=session_factory,
                storage=storage,
                predict=predict,
                settings=settings,
                claim_timeout_seconds=claim_timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001 —— 单个任务异常不杀死轮询循环
            logger.exception("local worker job %s crashed: %s", job_id, exc)
            summary["failed"] += 1
            continue
        status = outcome.get("status")
        if status == "succeeded":
            summary["succeeded"] += 1
        elif status == "failed":
            summary["failed"] += 1
        else:
            summary["skipped"] += 1
    return summary


def _mark_job_failed(
    db: Session,
    job: AnalysisJob,
    category: str,
    detail: str,
    settings: Settings,
    *,
    case: Case | None,
) -> None:
    from .audit import record_audit

    if case is not None and case.status == CaseStatus.Analyzing:
        apply_case_transition(case, CaseStatus.Failed)
        case.current_prediction_id = None
        case.result_type = None
    job.status = AnalysisStatus.Failed
    job.finished_at = _now_utc()
    job.error_category = category
    job.error_detail = (detail or "")[:2000]
    record_audit(
        db,
        action="analysis.failed",
        actor_user_id=case.owner_id if case is not None else None,
        resource_type="case",
        resource_id=str(case.id) if case is not None else None,
        detail={"job_id": str(job.id), "error_category": category},
    )
    db.commit()
    logger.error("analysis job %s failed: %s (%s)", job.id, category, detail)


def _compute_scope_decision(
    scope_gate: ScopeGate | None,
    feature_extract: Callable[[bytes], list[float]] | None,
    image_bytes: bytes,
) -> ScopeDecision | None:
    """根据门控状态与特征输出范围决策；特征提取失败时退化为 unavailable。"""
    if scope_gate is None or not scope_gate.enforced:
        return None
    if not scope_gate.available:
        return scope_gate.unavailable_decision()
    try:
        features = feature_extract(image_bytes)
    except Exception as exc:  # noqa: BLE001 —— 特征失败不代表真伪结论
        return ScopeDecision(
            scope=SCOPE_UNAVAILABLE,
            reason=f"范围特征提取失败：{type(exc).__name__}",
            enforced=True,
            artifact_version=scope_gate.version,
            detail={"error": str(exc)},
        )
    return scope_gate.decide(features)


def _scope_warning_text(scope_decision: ScopeDecision | None) -> str | None:
    if scope_decision is None:
        return None
    if scope_decision.scope == SCOPE_ACCEPTED:
        return None  # 在范围内，沿用闭集范围提示
    if scope_decision.scope == SCOPE_REJECTED:
        return "图片被判定为超出当前范围门控，不给出真伪结论。"
    return "当前无法提供可验证的范围证据，不给出真伪结论。"


def _build_default_feature_extract(settings: Settings):
    """默认冻结 encoder 特征提取器（Worker 进程内按需导入 torch）。"""

    def extract(image_bytes: bytes) -> list[float]:
        from .scope.features import build_feature_extractor

        extractor = build_feature_extractor(settings)
        return extractor(image_bytes)

    return extract


def _now_utc():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)


# `celery -A drynaria_api.worker:celery_app` 需要的模块级 app 别名
celery_app = get_celery_app()
