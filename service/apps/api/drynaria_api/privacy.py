"""隐私留存清理：未授权用于研究且完成满 N 天的原图；删除申请立即清理。

对象删除后保留 case_images 元数据（object_deleted_at 标记）用于审计，
但内容不再可读（content 端点会 404）。
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from .audit import record_audit
from .models import Case, CaseImage
from .storage import StorageBackend

logger = logging.getLogger(__name__)


def _as_utc(value: datetime) -> datetime:
    """SQLite 读回的 naive 时间按 UTC 语义补时区，便于与 aware now 比较。"""
    if value is None:
        raise ValueError("datetime 不能为 None")
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def purge_expired_originals(
    *,
    session_factory: sessionmaker[Session],
    storage: StorageBackend,
    retention_days: int,
    now: datetime | None = None,
    dry_run: bool = False,
) -> dict:
    stamp = now or datetime.now(timezone.utc)
    cutoff = stamp - timedelta(days=max(0, retention_days))
    stats = {"deleted_images": 0, "affected_cases": 0, "dry_run": dry_run}

    with session_factory() as db:
        # 1) 未授权研究 + 完成满 N 天的案件（SQL 层只做粗筛，天数比较在 Python 侧
        #    补齐时区后完成，兼容 SQLite naive 时间与 PostgreSQL timestamptz）
        # 2) 用户已提删除申请的案件（立即）
        cases = db.scalars(
            select(Case).where(
                Case.research_consent.is_(False),
                Case.completed_at.is_not(None),
            )
        ).all()
        expired_cases = [
            case for case in cases if _as_utc(case.completed_at) <= cutoff
        ]
        requested_cases = db.scalars(
            select(Case).where(Case.deletion_requested_at.is_not(None))
        ).all()
        seen: set = set()
        for case in [*expired_cases, *requested_cases]:
            if case.id in seen:
                continue
            seen.add(case.id)
            images = db.scalars(
                select(CaseImage)
                .where(CaseImage.case_id == case.id, CaseImage.object_deleted_at.is_(None))
            ).all()
            if not images:
                continue
            if not dry_run:
                # 只统计/审计真正删除成功的对象：删除失败的对象保持
                # object_deleted_at=NULL，下次清理会重试，不计入成功数。
                deleted_in_case = 0
                for image in images:
                    try:
                        storage.delete_bytes(image.object_key)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("privacy purge object %s failed: %s", image.object_key, exc)
                        continue
                    image.object_deleted_at = stamp
                    deleted_in_case += 1
                if deleted_in_case == 0:
                    continue
                record_audit(
                    db,
                    action="privacy.images_deleted",
                    resource_type="case",
                    resource_id=str(case.id),
                    detail={
                        "deleted_count": deleted_in_case,
                        "retention_days": retention_days,
                        "reason": "deletion_requested" if case.deletion_requested_at else "expired",
                    },
                )
                stats["affected_cases"] += 1
                stats["deleted_images"] += deleted_in_case
        if not dry_run:
            db.commit()
    return stats
