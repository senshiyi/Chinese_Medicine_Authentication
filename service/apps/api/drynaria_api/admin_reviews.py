"""专家复核后台路由（expert/admin）。

复核决定只追加版本，不修改历史模型预测；响应同时呈现多视角图、质量指标、
模型输出与历史审核版本。
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .audit import record_audit
from .cases import _image_out, _summary
from .database import DbSession
from .models import (
    Case,
    CaseImage,
    ExpertReview,
    ModelVersion,
    Prediction,
    ReviewRequest,
    ReviewStatus,
    User,
    UserRole,
)
from .schemas import ReviewDecisionIn
from .security import CsrfGuard, require_roles

router = APIRouter(prefix="/api/v1/admin", tags=["admin-review"])

ReviewerUser = Annotated[User, Depends(require_roles(UserRole.Expert, UserRole.Admin))]


def _review_row(db: Session, request_id: UUID) -> tuple[ReviewRequest, Case]:
    review_request = db.get(ReviewRequest, request_id)
    if review_request is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "review_not_found", "message": "复核申请不存在。"},
        )
    case = db.get(Case, review_request.case_id)
    if case is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "case_not_found", "message": "案件不存在。"},
        )
    return review_request, case


@router.get("/reviews")
def list_reviews(
    db: DbSession,
    user: ReviewerUser,
    status_filter: str | None = None,
    case_status: str | None = None,
) -> list[dict]:
    query = select(ReviewRequest).order_by(ReviewRequest.created_at.desc())
    if status_filter is not None:
        try:
            parsed_status = ReviewStatus(status_filter)
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_status", "message": "status 只允许 requested/in_review/completed。"},
            ) from exc
        query = query.where(ReviewRequest.status == parsed_status)
    requests = db.scalars(query).all()
    items: list[dict] = []
    for review_request in requests:
        case = db.get(Case, review_request.case_id)
        if case is None:
            continue
        if case_status is not None and str(case.status) != case_status:
            continue
        items.append(
            {
                "id": review_request.id,
                "case_id": case.id,
                "status": str(review_request.status),
                "reason": review_request.reason,
                "requested_at": review_request.created_at,
                "case_status": str(case.status),
                "result_type": str(case.result_type) if case.result_type else None,
                "claimed_herb": case.claimed_herb,
            }
        )
    return items


@router.get("/reviews/{review_id}")
def review_detail(review_id: UUID, db: DbSession, user: ReviewerUser) -> dict:
    review_request, case = _review_row(db, review_id)
    images = db.scalars(
        select(CaseImage).where(CaseImage.case_id == case.id).order_by(CaseImage.created_at)
    ).all()
    prediction = None
    if case.current_prediction_id is not None:
        prediction = db.get(Prediction, case.current_prediction_id)
    reviews = db.scalars(
        select(ExpertReview)
        .where(ExpertReview.review_request_id == review_request.id)
        .order_by(ExpertReview.version)
    ).all()
    return {
        "review_request": {
            "id": review_request.id,
            "status": str(review_request.status),
            "reason": review_request.reason,
            "created_at": review_request.created_at,
        },
        "case": _summary(db, case),
        "images": [_image_out(case, image) for image in images],
        "model_prediction": (
            {
                "model_version": prediction.model_version,
                "authenticity_label": prediction.authenticity_label,
                "confidence": prediction.confidence,
                "probabilities": dict(prediction.probabilities or {}),
                "experimental_closed_set": bool(prediction.experimental_closed_set),
                "open_set_calibrated": bool(prediction.open_set_calibrated),
                "actual_species_verified": bool(prediction.actual_species_verified),
                "message": prediction.message,
                "scope_warning": prediction.scope_warning,
                "model_fingerprint": prediction.model_fingerprint,
                "created_at": prediction.created_at,
            }
            if prediction is not None
            else None
        ),
        "expert_reviews": [
            {
                "id": item.id,
                "version": item.version,
                "actual_species": item.actual_species,
                "authenticity_class": item.authenticity_class,
                "processing_status": item.processing_status,
                "verification_method": item.verification_method,
                "basis": item.basis,
                "notes": item.notes,
                "created_at": item.created_at,
            }
            for item in reviews
        ],
    }


@router.post("/reviews/{review_id}/decisions", status_code=201)
def review_decision(
    review_id: UUID,
    body: ReviewDecisionIn,
    db: DbSession,
    user: ReviewerUser,
    _csrf: CsrfGuard,
) -> dict:
    review_request, case = _review_row(db, review_id)
    if not body.verification_method.strip():
        raise HTTPException(
            status_code=422,
            detail={"code": "verification_method_required", "message": "必须填写鉴定依据方法。"},
        )
    if review_request.status != ReviewStatus.Requested and not body.keep_open:
        # 已有结论时允许追加（keep_open=True）；否则视为已结案
        pass
    max_version = db.scalar(
        select(func.max(ExpertReview.version)).where(
            ExpertReview.review_request_id == review_request.id
        )
    )
    next_version = (max_version or 0) + 1
    expert_review = ExpertReview(
        review_request_id=review_request.id,
        reviewer_id=user.id,
        version=next_version,
        actual_species=body.actual_species,
        authenticity_class=body.authenticity_class,
        processing_status=body.processing_status,
        verification_method=body.verification_method.strip(),
        basis=body.basis,
        notes=body.notes,
    )
    db.add(expert_review)
    if review_request.status == ReviewStatus.Requested:
        review_request.status = ReviewStatus.InReview
    if not body.keep_open:
        review_request.status = ReviewStatus.Completed
        from datetime import datetime, timezone  # noqa: PLC0415

        review_request.decided_at = datetime.now(timezone.utc)
    record_audit(
        db,
        action="review.decision",
        actor_user_id=user.id,
        resource_type="review_request",
        resource_id=str(review_request.id),
        detail={
            "case_id": str(case.id),
            "version": next_version,
            "authenticity_class": body.authenticity_class,
            "processing_status": body.processing_status,
        },
    )
    db.commit()
    return {
        "id": expert_review.id,
        "review_request_id": review_request.id,
        "version": expert_review.version,
        "authenticity_class": expert_review.authenticity_class,
        "processing_status": expert_review.processing_status,
        "status": str(review_request.status),
    }


@router.get("/model-versions")
def model_versions(db: DbSession, user: ReviewerUser) -> list[dict]:
    rows = db.scalars(select(ModelVersion).order_by(ModelVersion.registered_at.desc())).all()
    return [
        {
            "model_version": row.model_version,
            "checkpoint_sha256": row.checkpoint_sha256,
            "strategy_version": row.strategy_version,
            "experimental_closed_set": bool(row.experimental_closed_set),
            "open_set_calibrated": bool(row.open_set_calibrated),
            "data_fingerprint": row.data_fingerprint,
            "split_fingerprint": row.split_fingerprint,
            "registered_at": row.registered_at,
        }
        for row in rows
    ]
