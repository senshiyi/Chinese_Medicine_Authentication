"""案件路由：创建、多视角上传、删除图片、幂等分析、结果查询、复核申请等。"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .audit import record_audit
from .database import DbSession
from .image_validation import UploadValidationError, validate_upload_image
from .models import (
    AnalysisJob,
    AnalysisStatus,
    Case,
    CaseImage,
    CaseStatus,
    ConsentRecord,
    ConsentType,
    ExpertReview,
    Prediction,
    ResultType,
    ReviewRequest,
    ReviewStatus,
    ViewType,
    apply_case_transition,
    can_transition_case,
    next_status_after_upload,
)
from .schemas import (
    RESULT_DISPLAY,
    AnalyzeOk,
    CaseCreateRequest,
    CaseDetailOut,
    CaseImageOut,
    CaseSummaryOut,
    ConsentIn,
    FeedbackIn,
    PredictionOut,
    ReviewRequestIn,
)
from .security import CsrfGuard, CurrentUser
from .settings import Settings, get_app_settings
from .storage import StorageBackend, content_type_for
from .scope.gate import SCOPE_ACCEPTED, SCOPE_UNAVAILABLE, scope_fingerprint as _scope_fingerprint

router = APIRouter(prefix="/api/v1/cases", tags=["cases"])


def _not_found() -> HTTPException:
    return HTTPException(
        status_code=404,
        detail={"code": "case_not_found", "message": "案件不存在。"},
    )


def _scope_enforced(settings: Settings) -> bool:
    return bool(getattr(settings, "scope_enforcement", True))


def _scope_decision_of(prediction: Prediction | None) -> dict | None:
    return prediction.scope_decision if prediction is not None else None


def _effective_scope(prediction: Prediction | None, scope_enforced: bool) -> str | None:
    """返回公开呈现时的有效范围：accepted / rejected / unavailable；未启用门控为 None。"""
    if not scope_enforced:
        return None
    decision = _scope_decision_of(prediction)
    if not decision:
        return SCOPE_UNAVAILABLE
    scope = decision.get("scope")
    if scope not in (SCOPE_ACCEPTED, "rejected"):
        return SCOPE_UNAVAILABLE
    return scope


def _result_verdict_valid(prediction: Prediction | None, scope_enforced: bool) -> bool:
    """是否允许把（可能的历史闭集）预测当作有效真伪结论呈现。"""
    if not scope_enforced:
        return True
    return _effective_scope(prediction, scope_enforced) == SCOPE_ACCEPTED


def _effective_result_type(
    case: Case, prediction: Prediction | None, scope_enforced: bool
) -> ResultType | None:
    """公开展示的结果类别：未启用门控或 accepted 时使用案件实际结果，否则一律 inconclusive。"""
    if not scope_enforced:
        return case.result_type
    if _result_verdict_valid(prediction, scope_enforced):
        return case.result_type
    return ResultType.Inconclusive


def _scope_out(prediction: Prediction | None, scope_enforced: bool) -> dict | None:
    """对外暴露的范围门控信息；未启用门控时为 None。"""
    if not scope_enforced:
        return None
    decision = _scope_decision_of(prediction)
    scope = _effective_scope(prediction, scope_enforced) or SCOPE_UNAVAILABLE
    return {
        "scope": scope,
        "enforced": True,
        "reason": (decision or {}).get("reason") or "未记录可验证的范围证据",
        "version": (decision or {}).get("version"),
        "score": (decision or {}).get("score"),
        "threshold": (decision or {}).get("threshold"),
        "artifact_version": (decision or {}).get("artifact_version"),
        "feature_dim": (decision or {}).get("feature_dim"),
    }


def _predictions_by_case(db: Session, cases: list[Case]) -> dict[UUID, Prediction]:
    ids = [case.current_prediction_id for case in cases if case.current_prediction_id is not None]
    if not ids:
        return {}
    rows = db.scalars(select(Prediction).where(Prediction.id.in_(ids))).all()
    return {row.id: row for row in rows}


def _owned_case(db: Session, user_id: UUID, case_id: UUID) -> Case:
    case = db.get(Case, case_id)
    if case is None or case.owner_id != user_id:
        raise _not_found()
    return case


def _whole_image(db: Session, case_id: UUID) -> CaseImage | None:
    return db.scalar(
        select(CaseImage)
        .where(
            CaseImage.case_id == case_id,
            CaseImage.view_type == ViewType.Whole,
            CaseImage.participates_in_inference.is_(True),
        )
        .order_by(CaseImage.created_at.desc())
        .limit(1)
    )


def _counts(db: Session, case_ids: list[UUID]) -> dict[UUID, dict[str, int]]:
    result: dict[UUID, dict[str, int]] = {}
    if not case_ids:
        return result
    rows = (
        db.execute(
            select(
                CaseImage.case_id,
                CaseImage.view_type,
                func.count(CaseImage.id),
            )
            .where(CaseImage.case_id.in_(case_ids))
            .group_by(CaseImage.case_id, CaseImage.view_type)
        )
        .all()
    )
    for case_id, view_type, count in rows:
        bucket = result.setdefault(case_id, {"whole": 0, "detail": 0, "cross_section": 0})
        bucket[view_type.value if isinstance(view_type, ViewType) else str(view_type)] = int(count)
    return result


def _summary(
    db: Session,
    case: Case,
    counts: dict[str, int] | None = None,
    *,
    prediction: Prediction | None = None,
    scope_enforced: bool = False,
) -> CaseSummaryOut:
    if counts is None:
        counts = _counts(db, [case.id]).get(case.id, {"whole": 0, "detail": 0, "cross_section": 0})
    effective = _effective_result_type(case, prediction, scope_enforced)
    return CaseSummaryOut(
        id=case.id,
        status=case.status.value if isinstance(case.status, CaseStatus) else str(case.status),
        result_type=effective.value if effective else None,
        claimed_herb=case.claimed_herb,
        created_at=case.created_at,
        updated_at=case.updated_at,
        completed_at=case.completed_at,
        whole_image_count=counts.get("whole", 0),
        detail_image_count=counts.get("detail", 0),
        cross_section_image_count=counts.get("cross_section", 0),
        scope_decision=_scope_out(prediction, scope_enforced),
        scope_enforced=scope_enforced,
        result_verdict_valid=_result_verdict_valid(prediction, scope_enforced),
    )


def _image_out(case: Case, image: CaseImage) -> CaseImageOut:
    return CaseImageOut(
        id=image.id,
        case_id=case.id,
        view_type=image.view_type.value
        if isinstance(image.view_type, ViewType)
        else str(image.view_type),
        object_key_hint=image.object_key[-8:],
        sha256=image.sha256,
        width=image.width,
        height=image.height,
        format=image.format,
        size_bytes=image.size_bytes,
        quality=image.quality,
        quality_flags=image.quality_flags or [],
        participates_in_inference=bool(image.participates_in_inference),
        created_at=image.created_at,
        content_url=f"/api/v1/cases/{case.id}/images/{image.id}/content",
    )


def _prediction_out(prediction: Prediction, *, scope_enforced: bool = False) -> PredictionOut:
    return PredictionOut(
        id=prediction.id,
        case_id=prediction.case_id,
        analysis_job_id=prediction.analysis_job_id,
        image_id=prediction.image_id,
        view_type=prediction.view_type.value
        if isinstance(prediction.view_type, ViewType)
        else str(prediction.view_type),
        model_version=prediction.model_version,
        strategy_version=prediction.strategy_version,
        experimental_closed_set=bool(prediction.experimental_closed_set),
        open_set_calibrated=bool(prediction.open_set_calibrated),
        authenticity_label=prediction.authenticity_label,
        confidence=float(prediction.confidence),
        probabilities=dict(prediction.probabilities or {}),
        actual_species_verified=bool(prediction.actual_species_verified),
        message=(prediction.message if _result_verdict_valid(prediction, scope_enforced)
                 else "范围验证未通过，无法鉴别真伪，请人工复核。"),
        scope_warning=prediction.scope_warning,
        scope_decision=prediction.scope_decision,
        scope_enforced=scope_enforced,
        model_fingerprint=prediction.model_fingerprint,
        created_at=prediction.created_at,
    )


def _idempotency_key(
    case_id: UUID, whole_sha256: str, model_version_hint: str, scope_fingerprint: str
) -> str:
    digest = hashlib.sha256()
    digest.update(str(case_id).encode("utf-8"))
    digest.update(b":")
    digest.update(whole_sha256.encode("utf-8"))
    digest.update(b":")
    digest.update(model_version_hint.encode("utf-8"))
    digest.update(b":")
    digest.update(scope_fingerprint.encode("utf-8"))
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# 案件创建 / 列表 / 详情
# ---------------------------------------------------------------------------


@router.post("", status_code=201)
def create_case(
    body: CaseCreateRequest,
    db: DbSession,
    user: CurrentUser,
    _csrf: CsrfGuard,
) -> CaseSummaryOut:
    if not body.analysis_consent:
        raise HTTPException(
            status_code=400,
            detail={"code": "analysis_consent_required", "message": "必须同意图片分析授权。"},
        )
    case = Case(
        owner_id=user.id,
        claimed_herb=body.claimed_herb,
        status=CaseStatus.Draft,
        analysis_consent=True,
        research_consent=bool(body.research_consent),
    )
    db.add(case)
    db.flush()
    db.add(
        ConsentRecord(
            user_id=user.id,
            case_id=case.id,
            consent_type=ConsentType.Analysis,
            granted=True,
            granted_at=datetime.now(timezone.utc),
        )
    )
    if body.research_consent:
        db.add(
            ConsentRecord(
                user_id=user.id,
                case_id=case.id,
                consent_type=ConsentType.ResearchTraining,
                granted=True,
                granted_at=datetime.now(timezone.utc),
            )
        )
    record_audit(
        db,
        action="case.create",
        actor_user_id=user.id,
        resource_type="case",
        resource_id=str(case.id),
    )
    db.commit()
    return _summary(db, case)


@router.get("")
def list_cases(
    db: DbSession,
    user: CurrentUser,
    settings: Settings = Depends(get_app_settings),
) -> list[CaseSummaryOut]:
    cases = db.scalars(
        select(Case).where(Case.owner_id == user.id).order_by(Case.created_at.desc())
    ).all()
    counts = _counts(db, [c.id for c in cases])
    predict_map = _predictions_by_case(db, cases)
    scope_enforced = _scope_enforced(settings)
    return [
        _summary(
            db,
            c,
            counts.get(c.id),
            prediction=predict_map.get(c.id),
            scope_enforced=scope_enforced,
        )
        for c in cases
    ]


@router.get("/{case_id}")
def get_case(
    case_id: UUID,
    db: DbSession,
    user: CurrentUser,
    settings: Settings = Depends(get_app_settings),
) -> CaseDetailOut:
    case = _owned_case(db, user.id, case_id)
    images = db.scalars(
        select(CaseImage).where(CaseImage.case_id == case.id).order_by(CaseImage.created_at)
    ).all()
    counts = _counts(db, [case.id]).get(case.id, {"whole": 0, "detail": 0, "cross_section": 0})

    prediction: Prediction | None = None
    if case.current_prediction_id is not None:
        prediction = db.get(Prediction, case.current_prediction_id)
    job: AnalysisJob | None = db.scalar(
        select(AnalysisJob)
        .where(AnalysisJob.case_id == case.id)
        .order_by(AnalysisJob.created_at.desc())
        .limit(1)
    )

    failure = None
    if job is not None and job.status == AnalysisStatus.Failed:
        failure = {
            "category": job.error_category or "prediction_failed",
            "occurred_at": job.finished_at.isoformat() if job.finished_at else None,
        }

    review_requests = db.scalars(
        select(ReviewRequest)
        .where(ReviewRequest.case_id == case.id)
        .order_by(ReviewRequest.created_at)
    ).all()
    request_out = [
        {
            "id": rr.id,
            "status": rr.status.value if isinstance(rr.status, ReviewStatus) else str(rr.status),
            "reason": rr.reason,
            "created_at": rr.created_at,
        }
        for rr in review_requests
    ]
    expert_rows: list[dict] = []
    for rr in review_requests:
        reviews = db.scalars(
            select(ExpertReview)
            .where(ExpertReview.review_request_id == rr.id)
            .order_by(ExpertReview.version)
        ).all()
        expert_rows.extend(
            {
                "id": r.id,
                "review_request_id": rr.id,
                "version": r.version,
                "actual_species": r.actual_species,
                "authenticity_class": r.authenticity_class,
                "processing_status": r.processing_status,
                "verification_method": r.verification_method,
                "basis": r.basis,
                "notes": r.notes,
                "created_at": r.created_at,
            }
            for r in reviews
        )

    scope_enforced = _scope_enforced(settings)
    effective_result_type = _effective_result_type(case, prediction, scope_enforced)
    result_display = None
    if effective_result_type is not None:
        result_display = RESULT_DISPLAY.get(str(effective_result_type))

    return CaseDetailOut(
        case=_summary(
            db, case, counts, prediction=prediction, scope_enforced=scope_enforced
        ),
        images=[_image_out(case, image) for image in images],
        prediction=(
            _prediction_out(prediction, scope_enforced=scope_enforced) if prediction else None
        ),
        prediction_message=(
            prediction.message if _result_verdict_valid(prediction, scope_enforced)
            else "范围验证未通过，无法鉴别真伪，请人工复核。"
        ) if prediction else None,
        prediction_scope_warning=prediction.scope_warning if prediction else None,
        result_display=result_display,
        result_verdict_valid=_result_verdict_valid(prediction, scope_enforced),
        scope=_scope_out(prediction, scope_enforced),
        scope_enforced=scope_enforced,
        failure=failure,
        analysis_job_id=job.id if job else None,
        analysis_status=job.status.value if job and job.status else None,
        review_requests=request_out,
        expert_reviews=expert_rows,
        consent={
            "analysis": bool(case.analysis_consent),
            "research_training": bool(case.research_consent),
        },
        deletion_requested=case.deletion_requested_at is not None,
    )


# ---------------------------------------------------------------------------
# 图片上传 / 删除 / 内容访问
# ---------------------------------------------------------------------------


def _storage(request: Request) -> StorageBackend:
    return request.app.state.storage


@router.post("/{case_id}/images", status_code=201)
def upload_image(
    case_id: UUID,
    db: DbSession,
    user: CurrentUser,
    _csrf: CsrfGuard,
    request: Request,
    settings: Settings = Depends(get_app_settings),
    view_type: str = Form(...),
    file: UploadFile = File(...),
):
    case = _owned_case(db, user.id, case_id)
    try:
        parsed_view = ViewType(view_type)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_view_type", "message": "view_type 只允许 whole/detail/cross_section。"},
        ) from exc

    # 只读取“上限+1”字节以限制内存占用；超限交给 validate 返回 400。
    # 上限来自配置（upload_max_bytes），不再是模块级硬编码常量。
    payload = file.file.read(settings.upload_max_bytes + 1)
    if parsed_view == ViewType.Whole and case.status not in (
        CaseStatus.Draft,
        CaseStatus.Ready,
        CaseStatus.NeedsRetake,
    ):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "case_not_uploadable",
                "message": "当前案件状态不允许替换主图；请新建案件。",
            },
        )
    if parsed_view in (ViewType.Detail, ViewType.CrossSection) and case.status not in (
        CaseStatus.Draft,
        CaseStatus.NeedsRetake,
        CaseStatus.Ready,
    ):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "case_not_uploadable",
                "message": "当前案件状态不允许再上传视角图。",
            },
        )

    try:
        validated = validate_upload_image(
            payload,
            max_bytes=settings.upload_max_bytes,
            max_pixels=settings.upload_max_pixels,
        )
    except UploadValidationError as exc:
        status_code = 400
        raise HTTPException(
            status_code=status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc

    # 先写对象存储，成功后再改数据库；失败不触碰数据库状态
    storage = _storage(request)
    object_key = f"images/{case.id}/{_random_key()}.img"
    try:
        storage.put_bytes(object_key, validated.sanitized_bytes, content_type_for(validated.format))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503,
            detail={"code": "storage_write_failed", "message": "图片暂存失败，请稍后重试。"},
        ) from exc

    # whole 替换：只保留一张参与推理的主图
    if parsed_view == ViewType.Whole:
        previous = _whole_image(db, case.id)
        if previous is not None and previous.id is not None:
            old_key = previous.object_key
            db.delete(previous)
            db.flush()
            try:
                storage.delete_bytes(old_key)
            except Exception:  # noqa: BLE001 —— 旧对象清理失败不影响新图
                pass

    image = CaseImage(
        case_id=case.id,
        view_type=parsed_view,
        object_key=object_key,
        sha256=validated.sha256,
        width=validated.width,
        height=validated.height,
        format=validated.format,
        size_bytes=validated.size_bytes,
        quality=validated.quality,
        quality_flags=validated.quality_flags,
        participates_in_inference=(parsed_view == ViewType.Whole),
    )
    db.add(image)
    db.flush()

    if parsed_view == ViewType.Whole:
        # 主图状态推进：draft/needs_retake 由质量决定 ready/needs_retake；
        # ready 案件允许替换主图：质量合格保持 ready，质量不合格退回 needs_retake。
        current_status = case.status
        if current_status in (CaseStatus.Draft, CaseStatus.NeedsRetake):
            target_status = next_status_after_upload(
                current_status, parsed_view, validated.quality_passes
            )
        elif current_status == CaseStatus.Ready:
            target_status = (
                CaseStatus.NeedsRetake if not validated.quality_passes else None
            )
        else:
            target_status = None
        if target_status is not None and target_status != current_status:
            apply_case_transition(case, target_status)

    record_audit(
        db,
        action="case_image.upload",
        actor_user_id=user.id,
        resource_type="case_image",
        resource_id=str(image.id),
        detail={
            "case_id": str(case.id),
            "view_type": parsed_view.value,
            "quality_flags": validated.quality_flags,
            "participates": parsed_view == ViewType.Whole,
        },
    )
    db.commit()
    return {
        "image": _image_out(case, image),
        "case": _summary(db, case),
    }


def _random_key() -> str:
    import uuid

    return str(uuid.uuid4())


@router.delete("/{case_id}/images/{image_id}", status_code=200)
def delete_image(
    case_id: UUID,
    image_id: UUID,
    db: DbSession,
    user: CurrentUser,
    _csrf: CsrfGuard,
    request: Request,
) -> dict:
    case = _owned_case(db, user.id, case_id)
    if case.status not in (CaseStatus.Draft, CaseStatus.Ready, CaseStatus.NeedsRetake):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "case_not_deletable",
                "message": "当前案件状态不允许删除图片。",
            },
        )
    image = db.get(CaseImage, image_id)
    if image is None or image.case_id != case.id:
        raise HTTPException(
            status_code=404,
            detail={"code": "image_not_found", "message": "图片不存在。"},
        )
    _storage(request).delete_bytes(image.object_key)
    db.delete(image)
    db.flush()
    if image.view_type == ViewType.Whole and _whole_image(db, case.id) is None:
        if case.status != CaseStatus.Draft and can_transition_case(case.status, CaseStatus.Draft):
            apply_case_transition(case, CaseStatus.Draft)
    record_audit(
        db,
        action="case_image.delete",
        actor_user_id=user.id,
        resource_type="case_image",
        resource_id=str(image_id),
        detail={"case_id": str(case.id)},
    )
    db.commit()
    return {"ok": True, "case_status": case.status.value}


@router.get("/{case_id}/images/{image_id}/content")
def image_content(case_id: UUID, image_id: UUID, db: DbSession, user: CurrentUser, request: Request) -> Response:
    case = _owned_case(db, user.id, case_id)
    image = db.get(CaseImage, image_id)
    if image is None or image.case_id != case.id:
        raise HTTPException(
            status_code=404,
            detail={"code": "image_not_found", "message": "图片不存在。"},
        )
    data = _storage(request).get_bytes(image.object_key)
    return Response(content=data, media_type=content_type_for(image.format))


# ---------------------------------------------------------------------------
# 幂等分析
# ---------------------------------------------------------------------------


@router.post("/{case_id}/analyze")
def analyze_case(
    case_id: UUID,
    db: DbSession,
    user: CurrentUser,
    _csrf: CsrfGuard,
    settings: Settings = Depends(get_app_settings),
) -> AnalyzeOk:
    case = _owned_case(db, user.id, case_id)
    whole = _whole_image(db, case.id)
    if whole is None:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "whole_image_required",
                "message": "请先上传一张质量合格的 whole 主图。",
            },
        )
    key = _idempotency_key(
        case.id, whole.sha256, settings.model_version_hint, _scope_fingerprint(settings)
    )

    existing = db.scalar(select(AnalysisJob).where(AnalysisJob.idempotency_key == key))
    if existing is not None:
        # 幂等重放：返回同一 analysis_job_id；排队中可补投递一次
        return _finish_analyze(db, case, whole, existing, settings)

    if case.status != CaseStatus.Ready:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "case_not_ready",
                "message": "只有状态为 ready 的案件可以排队分析（先上传合格 whole 主图）。",
            },
        )
    job = AnalysisJob(
        case_id=case.id,
        idempotency_key=key,
        whole_image_id=whole.id,
        model_version_hint=settings.model_version_hint,
        status=AnalysisStatus.Pending,
    )
    db.add(job)
    try:
        db.flush()
        apply_case_transition(case, CaseStatus.Queued)
        record_audit(
            db,
            action="analysis.submit",
            actor_user_id=user.id,
            resource_type="analysis_job",
            resource_id=str(job.id),
            detail={"case_id": str(case.id), "replayed": False, "whole_sha256": whole.sha256[:16]},
        )
        db.commit()
    except IntegrityError:
        # 并发竞态：另一请求在同一幂等键上先提交了 job。rollback 后复用胜者的
        # job 做幂等重放，绝不返回 500 / 绝不重复排队。
        db.rollback()
        winner = db.scalar(select(AnalysisJob).where(AnalysisJob.idempotency_key == key))
        if winner is None:  # 理论上不可能：IntegrityError 意味着同键已提交
            raise HTTPException(
                status_code=503,
                detail={"code": "analysis_conflict", "message": "分析任务提交冲突，请稍后重试。"},
            ) from None
        case = db.get(Case, case_id)  # rollback 后对象失效，重新读取
        if case is None:  # pragma: no cover —— 竞态中案件被删除的极端情形
            raise HTTPException(status_code=404, detail={"code": "case_not_found", "message": "案件不存在。"})
        whole = db.get(CaseImage, whole.id)
        return _finish_analyze(db, case, whole, winner, settings)

    # 先落库再入队：若入队失败，案件留在 queued，用户可幂等重试补投递
    _safe_dispatch(job.id, settings, fail_hard=True)
    return AnalyzeOk(
        case_id=case.id,
        analysis_job_id=job.id,
        job_status=job.status.value if job.status else "pending",
        case_status=case.status.value if case.status else str(case.status),
        replayed=False,
    )


def _finish_analyze(
    db: Session,
    case: Case,
    whole: CaseImage | None,
    job: AnalysisJob,
    settings: Settings,
) -> AnalyzeOk:
    """幂等重放收尾：补审计并返回同一 job（排队中可补投递一次）。"""
    record_audit(
        db,
        action="analysis.submit",
        actor_user_id=case.owner_id,
        resource_type="analysis_job",
        resource_id=str(job.id),
        detail={
            "case_id": str(case.id),
            "replayed": True,
            "whole_sha256": (whole.sha256[:16] if whole is not None else ""),
        },
    )
    db.commit()
    if job.status == AnalysisStatus.Pending and case.status in (
        CaseStatus.Queued,
        CaseStatus.Analyzing,
    ):
        _safe_dispatch(job.id, settings, fail_hard=False)
    return AnalyzeOk(
        case_id=case.id,
        analysis_job_id=job.id,
        job_status=job.status.value if job.status else "pending",
        case_status=case.status.value if case.status else str(case.status),
        replayed=True,
    )


def _safe_dispatch(job_id: UUID, settings: Settings, *, fail_hard: bool) -> None:
    from .worker import enqueue_analysis_job  # 延迟导入避免 API 启动链路引入 celery worker 命名空间

    try:
        enqueue_analysis_job(job_id, settings)
    except Exception as exc:  # noqa: BLE001
        if fail_hard:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "enqueue_failed",
                    "message": "案件已排队但任务队列暂不可用；请稍后重试（幂等，不会重复分析）。",
                },
            ) from exc


# ---------------------------------------------------------------------------
# 复核申请 / 反馈 / 同意 / 删除申请
# ---------------------------------------------------------------------------


@router.post("/{case_id}/review-requests", status_code=201)
def request_review(
    case_id: UUID,
    body: ReviewRequestIn,
    db: DbSession,
    user: CurrentUser,
    _csrf: CsrfGuard,
) -> dict:
    case = _owned_case(db, user.id, case_id)
    if case.status in (
        CaseStatus.Draft,
        CaseStatus.Ready,
        CaseStatus.Queued,
        CaseStatus.Analyzing,
    ):
        raise HTTPException(
            status_code=409,
            detail={"code": "case_not_reviewable", "message": "分析未结束前不能申请复核。"},
        )
    active = db.scalar(
        select(ReviewRequest).where(
            ReviewRequest.case_id == case.id,
            ReviewRequest.status.in_([ReviewStatus.Requested, ReviewStatus.InReview]),
        )
    )
    if active is not None:
        raise HTTPException(
            status_code=409,
            detail={"code": "review_already_requested", "message": "该案件已有进行中的复核申请。"},
        )
    review_request = ReviewRequest(
        case_id=case.id, user_id=user.id, status=ReviewStatus.Requested, reason=body.reason
    )
    db.add(review_request)
    record_audit(
        db,
        action="review.request",
        actor_user_id=user.id,
        resource_type="review_request",
        resource_id=str(review_request.id),
        detail={"case_id": str(case.id)},
    )
    db.commit()
    return {"id": review_request.id, "status": review_request.status.value, "case_id": case.id}


@router.post("/{case_id}/feedback", status_code=202)
def submit_feedback(
    case_id: UUID,
    body: FeedbackIn,
    db: DbSession,
    user: CurrentUser,
    _csrf: CsrfGuard,
) -> dict:
    case = _owned_case(db, user.id, case_id)
    record_audit(
        db,
        action="case.feedback",
        actor_user_id=user.id,
        resource_type="case",
        resource_id=str(case.id),
        detail={"category": body.category, "message": body.message},
    )
    db.commit()
    return {"ok": True}


@router.post("/{case_id}/consent", status_code=200)
def update_consent(
    case_id: UUID,
    body: ConsentIn,
    db: DbSession,
    user: CurrentUser,
    _csrf: CsrfGuard,
) -> dict:
    case = _owned_case(db, user.id, case_id)
    now = datetime.now(timezone.utc)
    if body.consent_type == "research_training":
        if body.granted and case.research_consent:
            return {"ok": True, "research_training": True}
        db.add(
            ConsentRecord(
                user_id=user.id,
                case_id=case.id,
                consent_type=ConsentType.ResearchTraining,
                granted=body.granted,
                granted_at=now if body.granted else None,
                revoked_at=None if body.granted else now,
            )
        )
        case.research_consent = body.granted
        case.research_consent_revoked_at = None if body.granted else now
        if not body.granted:
            case.updated_at = now
    record_audit(
        db,
        action="consent.update",
        actor_user_id=user.id,
        resource_type="case",
        resource_id=str(case.id),
        detail={"consent_type": body.consent_type, "granted": body.granted},
    )
    db.commit()
    return {"ok": True, "research_training": case.research_consent}


@router.post("/{case_id}/deletion-request", status_code=202)
def request_deletion(
    case_id: UUID,
    db: DbSession,
    user: CurrentUser,
    _csrf: CsrfGuard,
) -> dict:
    case = _owned_case(db, user.id, case_id)
    if case.deletion_requested_at is None:
        case.deletion_requested_at = datetime.now(timezone.utc)
    record_audit(
        db,
        action="case.deletion_request",
        actor_user_id=user.id,
        resource_type="case",
        resource_id=str(case.id),
    )
    db.commit()
    return {"ok": True, "deletion_requested": True}
