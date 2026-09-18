"""多任务训练损失、指标、checkpoint、训练循环与安全晋升。"""

from .promotion import (
    DEFAULT_PROMOTION_THRESHOLDS,
    PromotionError,
    PromotionRecord,
    PromotionRequest,
    PromotionThresholds,
    promote_candidate,
)
from .run_artifacts import (
    RunContext,
    create_run_context,
    sha256_file,
    write_run_metadata,
)

__all__ = [
    "DEFAULT_PROMOTION_THRESHOLDS",
    "PromotionError",
    "PromotionRecord",
    "PromotionRequest",
    "PromotionThresholds",
    "RunContext",
    "create_run_context",
    "promote_candidate",
    "sha256_file",
    "write_run_metadata",
]
