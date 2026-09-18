"""隐私留存清理单元测试：过期原图删除、研究授权豁免、失败不计入成功。"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient
from sqlalchemy import select

from drynaria_api.database import make_session_factory
from drynaria_api.models import AuditLog, Case, CaseImage
from drynaria_api.privacy import purge_expired_originals
from drynaria_api.settings import Settings
from drynaria_api.storage import LocalStorageBackend, StorageError
from drynaria_api.worker import run_analysis_job
from helpers import analyze_case, create_case, register, upload_image


class _FlakyDeleteStorage(LocalStorageBackend):
    """前 N 次 delete_bytes 抛 StorageError，之后正常（模拟对象存储故障）。"""

    def __init__(self, base_dir, *, fail_first: int = 1) -> None:
        super().__init__(base_dir)
        self._fail_first = fail_first
        self._calls = 0

    def delete_bytes(self, key: str) -> None:
        self._calls += 1
        if self._calls <= self._fail_first:
            raise StorageError(f"simulated delete failure for {key}")
        return super().delete_bytes(key)


def _complete(client: TestClient, settings: Settings, account: dict, research_consent: bool) -> str:
    case = create_case(client, account, research_consent=research_consent)
    upload_image(client, case["id"], "detail", account)
    upload_image(client, case["id"], "whole", account)
    result = analyze_case(client, case["id"], account)
    run_analysis_job(
        result["analysis_job_id"],
        session_factory=make_session_factory(settings.database_url),
        storage=LocalStorageBackend(settings.storage_local_dir),
        predict=lambda _b: {
            "schema_version": 1,
            "model_version": "closed-set-v1",
            "experimental_closed_set": True,
            "open_set_calibrated": False,
            "message": "此药品为正品。",
            "authenticity": {"label": "authentic", "confidence": 0.91},
            "probabilities": {"authentic": 0.91, "counterfeit": 0.09},
            "actual_species_verified": False,
            "model_fingerprint": "a" * 64,
            "data_fingerprint": "b" * 64,
            "split_fingerprint": "c" * 64,
        },
        settings=settings,
    )
    return case["id"]


def test_purge_deletes_expired_unauthorized_originals_but_keeps_authorized(
    client: TestClient, app_settings: Settings, session_factory
):
    account = register(client)
    unauthorized_case = _complete(client, app_settings, account, research_consent=False)
    authorized_case = _complete(client, app_settings, account, research_consent=True)

    now = datetime.now(timezone.utc)
    with session_factory() as db:
        expired = db.get(Case, uuid.UUID(unauthorized_case))
        expired.completed_at = now - timedelta(days=40)
        db.commit()

    storage = LocalStorageBackend(app_settings.storage_local_dir)
    stats = purge_expired_originals(
        session_factory=session_factory,
        storage=storage,
        retention_days=30,
        now=now,
    )
    assert stats["deleted_images"] == 2
    assert stats["affected_cases"] == 1

    with session_factory() as db:
        for case_id in (unauthorized_case, authorized_case):
            images = db.scalars(
                select(CaseImage).where(CaseImage.case_id == uuid.UUID(case_id))
            ).all()
            for image in images:
                if case_id == unauthorized_case:
                    assert image.object_deleted_at is not None
                else:
                    assert image.object_deleted_at is None


def test_dry_run_does_not_delete(client: TestClient, app_settings: Settings, session_factory):
    account = register(client)
    case_id = _complete(client, app_settings, account, research_consent=False)
    now = datetime.now(timezone.utc)
    with session_factory() as db:
        case = db.get(Case, uuid.UUID(case_id))
        case.completed_at = now - timedelta(days=40)
        db.commit()
    storage = LocalStorageBackend(app_settings.storage_local_dir)
    stats = purge_expired_originals(
        session_factory=session_factory,
        storage=storage,
        retention_days=30,
        now=now,
        dry_run=True,
    )
    assert stats["deleted_images"] == 0
    with session_factory() as db:
        image = db.scalar(select(CaseImage).where(CaseImage.case_id == uuid.UUID(case_id)))
        assert image.object_deleted_at is None
        # 对象仍在存储中
        assert storage.get_bytes(image.object_key) != b""


def test_failed_storage_deletions_are_not_counted_or_audited_as_success(
    client: TestClient, app_settings: Settings, session_factory
):
    account = register(client)
    case_id = _complete(client, app_settings, account, research_consent=False)
    now = datetime.now(timezone.utc)
    with session_factory() as db:
        case = db.get(Case, uuid.UUID(case_id))
        case.completed_at = now - timedelta(days=40)
        db.commit()

    healthy = LocalStorageBackend(app_settings.storage_local_dir)
    flaky = _FlakyDeleteStorage(healthy.base_dir, fail_first=1)  # 第一个对象删除失败

    stats = purge_expired_originals(
        session_factory=session_factory,
        storage=flaky,
        retention_days=30,
        now=now,
    )
    # 只统计实际删除成功的对象：1 个成功、1 个失败不计入
    assert stats["deleted_images"] == 1
    assert stats["affected_cases"] == 1

    with session_factory() as db:
        images = db.scalars(
            select(CaseImage).where(CaseImage.case_id == uuid.UUID(case_id))
        ).all()
        marked = [image for image in images if image.object_deleted_at is not None]
        assert len(marked) == 1  # 失败的 image 保持 object_deleted_at=None
        audits = db.scalars(
            select(AuditLog).where(AuditLog.action == "privacy.images_deleted")
        ).all()
        assert len(audits) == 1
        assert audits[0].detail["deleted_count"] == 1  # 审计只记实际成功数

    # 下一轮清理（存储恢复）删除上次失败的那个对象
    stats2 = purge_expired_originals(
        session_factory=session_factory,
        storage=healthy,
        retention_days=30,
        now=now,
    )
    assert stats2["deleted_images"] == 1
    with session_factory() as db:
        images = db.scalars(
            select(CaseImage).where(CaseImage.case_id == uuid.UUID(case_id))
        ).all()
        assert all(image.object_deleted_at is not None for image in images)
