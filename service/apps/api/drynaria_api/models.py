"""ORM 模型、枚举与案件状态机（阶段 A）。

表名来自设计规格：users / cases / case_images / analysis_jobs / predictions /
review_requests / expert_reviews / model_versions / consent_records / audit_logs。
预测行只追加、不可变；案件状态转换集中在这里判定。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import StrEnum

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    String,
    Text,
    Uuid,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


def _enum(enum_cls: type[StrEnum], *, default: StrEnum | None = None, nullable: bool = False):
    return Enum(
        enum_cls,
        name=f"{enum_cls.__name__.lower()}_type",
        native_enum=False,
        length=32,
        validate_strings=True,
        values_callable=lambda cls: [member.value for member in cls],
        default=default.value if default is not None else None,
        nullable=nullable,
    )


# ---------------------------------------------------------------------------
# 枚举
# ---------------------------------------------------------------------------


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


class UserRole(StrEnum):
    User = "user"
    Expert = "expert"
    Admin = "admin"


class ResultType(StrEnum):
    AuthenticLikely = "authentic_likely"
    SuspectedCounterfeit = "suspected_counterfeit"
    Inconclusive = "inconclusive"


class ConsentType(StrEnum):
    Analysis = "analysis"
    ResearchTraining = "research_training"


# ---------------------------------------------------------------------------
# 案件状态机
# ---------------------------------------------------------------------------

ALLOWED_CASE_TRANSITIONS: dict[CaseStatus, set[CaseStatus]] = {
    CaseStatus.Draft: {CaseStatus.Ready, CaseStatus.NeedsRetake},
    CaseStatus.Ready: {CaseStatus.Draft, CaseStatus.Queued, CaseStatus.NeedsRetake},
    CaseStatus.NeedsRetake: {CaseStatus.Draft, CaseStatus.Ready},
    CaseStatus.Queued: {CaseStatus.Analyzing},
    CaseStatus.Analyzing: {
        CaseStatus.Completed,
        CaseStatus.NeedsRetake,
        CaseStatus.Inconclusive,
        CaseStatus.Failed,
    },
    # completed / inconclusive / failed 在阶段 A 为吸收态：结果页引导复拍走新案件。
    CaseStatus.Completed: set(),
    CaseStatus.Inconclusive: set(),
    CaseStatus.Failed: set(),
}


class CaseStateError(RuntimeError):
    """非法状态转换；携带工程错误码，不携带真伪结论。"""

    def __init__(self, current: CaseStatus, next_status: CaseStatus) -> None:
        self.current = current
        self.next_status = next_status
        super().__init__(f"case transition {current.value} -> {next_status.value} 不允许")


def can_transition_case(current: CaseStatus, next_status: CaseStatus) -> bool:
    return next_status in ALLOWED_CASE_TRANSITIONS.get(current, set())


def apply_case_transition(case: "Case", next_status: CaseStatus, *, now: datetime | None = None) -> None:
    if not can_transition_case(case.status, next_status):
        raise CaseStateError(case.status, next_status)
    stamp = now or _now()
    case.status = next_status
    case.updated_at = stamp
    if next_status in (CaseStatus.Completed, CaseStatus.Inconclusive, CaseStatus.Failed):
        case.completed_at = stamp


def next_status_after_upload(
    current: CaseStatus, view_type: ViewType, quality_passes: bool
) -> CaseStatus | None:
    """上传后的案件目标状态；None 表示状态不变（detail/cross_section 不参与分析）。"""
    if current not in (CaseStatus.Draft, CaseStatus.NeedsRetake):
        raise CaseStateError(current, current)
    if view_type != ViewType.Whole:
        return None
    if quality_passes:
        return CaseStatus.Ready
    return CaseStatus.NeedsRetake


# ---------------------------------------------------------------------------
# Base / Mixin
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now, nullable=False
    )


# ---------------------------------------------------------------------------
# 用户与会话
# ---------------------------------------------------------------------------


class User(TimestampMixin, Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(
        _enum(UserRole, default=UserRole.User), nullable=False
    )
    display_name: Mapped[str] = mapped_column(String(120), nullable=False, default="")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class Session(TimestampMixin, Base):
    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("users.id"), index=True, nullable=False
    )
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    csrf_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    # CSRF 明文只存服务端 sessions 表，供 GET /me 回读（/me 只读、不轮换，
    # 避免前端多次 /me 的轮换竞态导致客户端 token 与服务端不同步）。
    csrf_value: Mapped[str | None] = mapped_column(String(200), nullable=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


# ---------------------------------------------------------------------------
# 案件 / 图片 / 分析 / 预测
# ---------------------------------------------------------------------------


class Case(TimestampMixin, Base):
    __tablename__ = "cases"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    owner_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("users.id"), index=True, nullable=False
    )
    claimed_herb: Mapped[str] = mapped_column(String(120), nullable=False, default="骨碎补")
    status: Mapped[CaseStatus] = mapped_column(
        _enum(CaseStatus, default=CaseStatus.Draft), nullable=False
    )
    result_type: Mapped[ResultType | None] = mapped_column(_enum(ResultType), nullable=True)
    strategy_version: Mapped[str | None] = mapped_column(String(120), nullable=True)
    # 指向 predictions.id 的展示引用；故意不加外键约束以避免表间环依赖，
    # 由代码保证只指向同案件预测。
    current_prediction_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, index=True, nullable=True)
    analysis_consent: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    research_consent: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    research_consent_revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    deletion_requested_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class CaseImage(TimestampMixin, Base):
    __tablename__ = "case_images"
    __table_args__ = (
        Index("ix_case_images_case_view", "case_id", "view_type"),
        Index("ix_case_images_sha256", "sha256"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    case_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("cases.id"), nullable=False
    )
    view_type: Mapped[ViewType] = mapped_column(_enum(ViewType), nullable=False)
    object_key: Mapped[str] = mapped_column(String(160), unique=True, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    width: Mapped[int] = mapped_column(nullable=False)
    height: Mapped[int] = mapped_column(nullable=False)
    format: Mapped[str] = mapped_column(String(16), nullable=False)
    size_bytes: Mapped[int] = mapped_column(nullable=False)
    quality: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    quality_flags: Mapped[list | None] = mapped_column(JSON, nullable=True)
    # 真伪推理只允许 whole 主图参与：由 API/Worker 显式过滤 view_type。
    participates_in_inference: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    # 隐私留存：原图对象被清理的时间（对象删除后保留元数据行用于审计）
    object_deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class AnalysisJob(TimestampMixin, Base):
    __tablename__ = "analysis_jobs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    case_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("cases.id"), index=True, nullable=False
    )
    # 幂等键 = sha256(case_id + whole.sha256 + model_version_hint)
    idempotency_key: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    whole_image_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("case_images.id"), nullable=False
    )
    model_version_hint: Mapped[str] = mapped_column(String(120), nullable=False)
    status: Mapped[AnalysisStatus] = mapped_column(
        _enum(AnalysisStatus, default=AnalysisStatus.Pending), nullable=False
    )
    attempts: Mapped[int] = mapped_column(nullable=False, default=0)
    error_category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Prediction(TimestampMixin, Base):
    """不可变模型原始输出（含实验范围标志与模型指纹）。只追加，不修改。"""

    __tablename__ = "predictions"
    __table_args__ = (
        Index("ix_predictions_case_created", "case_id", "created_at"),
        # 数据库层保证“一个分析 job 至多一条预测”：即使 Worker 领取/投递出现
        # 竞态，重复推理也不可能落两条预测。
        Index("uq_predictions_analysis_job", "analysis_job_id", unique=True),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    case_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("cases.id"), nullable=False)
    analysis_job_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("analysis_jobs.id"), nullable=False
    )
    image_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("case_images.id"), nullable=False
    )
    view_type: Mapped[ViewType] = mapped_column(_enum(ViewType), nullable=False)
    model_version: Mapped[str] = mapped_column(String(120), nullable=False)
    strategy_version: Mapped[str] = mapped_column(String(120), nullable=False)
    experimental_closed_set: Mapped[bool] = mapped_column(Boolean, nullable=False)
    open_set_calibrated: Mapped[bool] = mapped_column(Boolean, nullable=False)
    authenticity_label: Mapped[str] = mapped_column(String(32), nullable=False)
    confidence: Mapped[float] = mapped_column(nullable=False)
    probabilities: Mapped[dict] = mapped_column(JSON, nullable=False)
    actual_species_verified: Mapped[bool] = mapped_column(Boolean, nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    scope_warning: Mapped[str | None] = mapped_column(Text, nullable=True)
    # 范围门控决策（accepted/rejected/unavailable + reason/version/score/threshold）；
    # None 表示该预测产生于未启用范围门控的阶段，或该次推理未记录范围决策。
    scope_decision: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    model_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    data_fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True)
    split_fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    model_version: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    checkpoint_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    strategy_version: Mapped[str | None] = mapped_column(String(120), nullable=True)
    experimental_closed_set: Mapped[bool] = mapped_column(Boolean, nullable=False)
    open_set_calibrated: Mapped[bool] = mapped_column(Boolean, nullable=False)
    data_fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True)
    split_fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True)
    registered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )


# ---------------------------------------------------------------------------
# 复核 / 同意 / 审计
# ---------------------------------------------------------------------------


class ReviewRequest(TimestampMixin, Base):
    __tablename__ = "review_requests"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    case_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("cases.id"), index=True, nullable=False
    )
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("users.id"), nullable=False)
    status: Mapped[ReviewStatus] = mapped_column(
        _enum(ReviewStatus, default=ReviewStatus.Requested), nullable=False
    )
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ExpertReview(TimestampMixin, Base):
    """版本化专家结论；追加版本，不覆盖模型历史预测。"""

    __tablename__ = "expert_reviews"
    __table_args__ = (Index("ix_expert_reviews_request_version", "review_request_id", "version"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    review_request_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("review_requests.id"), nullable=False
    )
    reviewer_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("users.id"), nullable=False
    )
    version: Mapped[int] = mapped_column(nullable=False)
    actual_species: Mapped[str | None] = mapped_column(String(160), nullable=True)
    authenticity_class: Mapped[str] = mapped_column(String(32), nullable=False)
    processing_status: Mapped[str] = mapped_column(String(32), nullable=False)
    verification_method: Mapped[str] = mapped_column(String(160), nullable=False)
    basis: Mapped[str | None] = mapped_column(Text, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


class ConsentRecord(Base):
    __tablename__ = "consent_records"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("users.id"), nullable=False)
    case_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("cases.id"), nullable=True
    )
    consent_type: Mapped[ConsentType] = mapped_column(_enum(ConsentType), nullable=False)
    granted: Mapped[bool] = mapped_column(Boolean, nullable=False)
    granted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )


class AuditLog(Base):
    __tablename__ = "audit_logs"
    __table_args__ = (
        Index("ix_audit_logs_actor_created", "actor_user_id", "created_at"),
        Index("ix_audit_logs_action", "action"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    actor_user_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("users.id"), nullable=True
    )
    action: Mapped[str] = mapped_column(String(80), nullable=False)
    resource_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    resource_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    detail: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    ip_address: Mapped[str | None] = mapped_column(String(64), nullable=True)
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )
