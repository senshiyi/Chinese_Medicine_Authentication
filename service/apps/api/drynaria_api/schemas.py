"""API 请求/响应 schema（版本化、仅向后兼容扩展）。

响应正文采用 JSON 原生类型；领域结论文案一律来自模型 payload 或本文件中的
预注册固定文案（RESULT_DISPLAY），不在路由里拼接或重算概率。
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

DISCLAIMER = "模型结果不能替代专业鉴定。"
EXPERIMENTAL_NOTICE = "本服务为阶段A实验性闭集服务，仅支持实验室白底拍摄范围；不识别伪品实际物种。"

# label -> 固定展示分类（预注册文案，不随概率改变）
RESULT_DISPLAY: dict[str, str] = {
    "authentic_likely": "正品特征吻合度高",
    "suspected_counterfeit": "疑似伪品（无法确认实际物种，请人工复核）",
    "inconclusive": "无法判断",
}

# 工程失败类别白名单：失败响应只能返回这些类别，绝不生成真伪结论。
FAILURE_CATEGORIES = {
    "storage_read_failed",
    "image_decode_failed",
    "model_load_failed",
    "prediction_failed",
    "database_error",
    "case_state_invalid",
    "job_not_found",
}


# ---------------------------------------------------------------------------
# 通用
# ---------------------------------------------------------------------------


class ErrorBody(BaseModel):
    code: str
    message: str
    detail: dict | None = None


# ---------------------------------------------------------------------------
# 鉴权
# ---------------------------------------------------------------------------


class RegisterRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=8, max_length=256)
    display_name: str = Field(default="", max_length=120)

    @field_validator("email")
    @classmethod
    def _normalize_email(cls, value: str) -> str:
        return value.strip().lower()


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=256)

    @field_validator("email")
    @classmethod
    def _normalize_email(cls, value: str) -> str:
        return value.strip().lower()


class AdminCreateUserRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=8, max_length=256)
    role: Literal["user", "expert", "admin"]
    display_name: str = Field(default="", max_length=120)

    @field_validator("email")
    @classmethod
    def _normalize_email(cls, value: str) -> str:
        return value.strip().lower()


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    email: str
    role: str
    display_name: str


class MeOut(BaseModel):
    user: UserOut
    csrf_token: str


# ---------------------------------------------------------------------------
# 案件
# ---------------------------------------------------------------------------


class CaseCreateRequest(BaseModel):
    claimed_herb: str = Field(default="骨碎补", max_length=120)
    analysis_consent: bool = True
    research_consent: bool = False


class CaseImageOut(BaseModel):
    id: UUID
    case_id: UUID
    view_type: str
    object_key_hint: str | None = None  # 仅调试返回截断键，不返回完整内部键
    sha256: str
    width: int
    height: int
    format: str
    size_bytes: int
    quality: dict | None = None
    quality_flags: list[str] = []
    participates_in_inference: bool
    created_at: datetime
    content_url: str


class PredictionOut(BaseModel):
    id: UUID
    case_id: UUID
    analysis_job_id: UUID
    image_id: UUID
    view_type: str
    model_version: str
    strategy_version: str
    experimental_closed_set: bool
    open_set_calibrated: bool
    authenticity_label: str
    confidence: float
    probabilities: dict[str, float]
    actual_species_verified: bool
    message: str
    scope_warning: str | None = None
    scope_decision: dict | None = None
    scope_enforced: bool = False
    model_fingerprint: str | None = None
    created_at: datetime


class CaseSummaryOut(BaseModel):
    id: UUID
    status: str
    result_type: str | None = None
    claimed_herb: str
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    whole_image_count: int = 0
    detail_image_count: int = 0
    cross_section_image_count: int = 0
    scope_decision: dict | None = None
    scope_enforced: bool = False
    result_verdict_valid: bool = True


class CaseDetailOut(BaseModel):
    case: CaseSummaryOut
    images: list[CaseImageOut]
    prediction: PredictionOut | None = None
    prediction_message: str | None = None
    prediction_scope_warning: str | None = None
    result_display: str | None = None
    result_verdict_valid: bool = True
    scope: dict | None = None
    scope_enforced: bool = False
    experimental_notice: str = EXPERIMENTAL_NOTICE
    disclaimer: str = DISCLAIMER
    failure: dict | None = None
    analysis_job_id: UUID | None = None
    analysis_status: str | None = None
    review_requests: list[dict] = []
    expert_reviews: list[dict] = []
    consent: dict = {"analysis": True, "research_training": False}
    deletion_requested: bool = False


class AnalyzeOk(BaseModel):
    case_id: UUID
    analysis_job_id: UUID
    job_status: str
    case_status: str
    replayed: bool


class ReviewRequestIn(BaseModel):
    reason: str | None = Field(default=None, max_length=2000)


class ConsentIn(BaseModel):
    consent_type: Literal["research_training"]
    granted: bool


class FeedbackIn(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    category: str | None = Field(default=None, max_length=80)


# ---------------------------------------------------------------------------
# 专家审核
# ---------------------------------------------------------------------------


class ReviewDecisionIn(BaseModel):
    actual_species: str | None = Field(default=None, max_length=160)
    authenticity_class: Literal["authentic", "substitute", "adulterant", "unknown"]
    processing_status: Literal[
        "compliant", "noncompliant", "not_applicable", "unknown"
    ]
    verification_method: str = Field(min_length=1, max_length=160)
    basis: str | None = Field(default=None, max_length=4000)
    notes: str | None = Field(default=None, max_length=4000)
    keep_open: bool = False
