"""Worker 推理适配集成：仅 whole 参与、状态/结果映射、失败无结论、幂等。"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from uuid import UUID

from fastapi.testclient import TestClient
from sqlalchemy import func, select

from drynaria_api.database import make_session_factory
from drynaria_api.inference_adapter import InferenceError
from drynaria_api.models import (
    AnalysisJob,
    AnalysisStatus,
    Case,
    CaseImage,
    CaseStatus,
    Prediction,
    ResultType,
    ViewType,
)
from drynaria_api.settings import Settings
from drynaria_api.storage import LocalStorageBackend
from drynaria_api.worker import compute_case_outcome, run_analysis_job
from helpers import analyze_case, create_case, make_image_bytes, register, upload_image

EXPECTED_VERSION = "closed-set-v1"


def _payload(label: str = "authentic", confidence: float = 0.91) -> dict:
    probabilities = (
        {"authentic": confidence, "counterfeit": round(1.0 - confidence, 6)}
        if label == "authentic"
        else {"authentic": round(1.0 - confidence, 6), "counterfeit": confidence}
    )
    if label == "authentic":
        message = f"此药品为正品，有{confidence:.1%}的可能该药材为正品。"
    else:
        message = f"实际药材无法确定，有{confidence:.1%}的可能该药材为伪品，请人工复核。"
    return {
        "schema_version": 1,
        "model_version": EXPECTED_VERSION,
        "experimental_closed_set": True,
        "open_set_calibrated": False,
        "scope_warning": "仅适用于本数据集定义的骨碎补正品/伪品范围",
        "message": message,
        "authenticity": {"label": label, "confidence": confidence},
        "probabilities": probabilities,
        "actual_species_verified": False,
        "model_fingerprint": "a" * 64,
        "data_fingerprint": "b" * 64,
        "split_fingerprint": "c" * 64,
    }


def _build_case_with_images(client: TestClient, settings: Settings, account: dict) -> dict:
    case = create_case(client, account)
    upload_image(client, case["id"], "detail", account)
    upload_image(client, case["id"], "cross_section", account)
    upload_image(client, case["id"], "whole", account)
    result = analyze_case(client, case["id"], account)
    return {"case": case, "job_id": result["analysis_job_id"]}


def _run(job_id: str, settings: Settings, predict, captured: list | None = None) -> dict:
    storage = LocalStorageBackend(settings.storage_local_dir)

    def recording_predict(data: bytes) -> dict:
        if captured is not None:
            captured.append(data)
        return predict(data)

    return run_analysis_job(
        job_id,
        session_factory=make_session_factory(settings.database_url),
        storage=storage,
        predict=recording_predict,
        settings=settings,
    )


def test_worker_uses_only_whole_image_and_maps_authentic_result(
    client: TestClient, app_settings: Settings, session_factory
):
    account = register(client)
    built = _build_case_with_images(client, app_settings, account)
    captured: list[bytes] = []

    outcome = _run(built["job_id"], app_settings, lambda _b: _payload("authentic", 0.91), captured)

    assert outcome["status"] == "succeeded"
    assert outcome["case_status"] == "completed"
    # detail/cross_section 绝不参与：只调用一次，且收到的就是 whole 图内容
    assert len(captured) == 1
    with session_factory() as db:
        whole = db.scalar(
            select(CaseImage)
            .where(CaseImage.case_id == UUID(built["case"]["id"]), CaseImage.view_type == ViewType.Whole)
        )
        assert hashlib.sha256(captured[0]).hexdigest() == whole.sha256
        case = db.get(Case, whole.case_id)
        assert case.status == CaseStatus.Completed
        assert case.result_type == ResultType.AuthenticLikely
        job = db.get(AnalysisJob, UUID(built["job_id"]))
        assert job.status == AnalysisStatus.Succeeded
        prediction = db.get(Prediction, case.current_prediction_id)
        assert prediction.image_id == whole.id
        assert prediction.view_type == ViewType.Whole
        assert prediction.experimental_closed_set is True
        assert prediction.open_set_calibrated is False
        assert prediction.model_version == EXPECTED_VERSION
        assert prediction.authenticity_label == "authentic"
        assert prediction.payload["experimental_closed_set"] is True
        assert prediction.payload["open_set_calibrated"] is False
        count = db.scalar(select(func.count()).select_from(Prediction).where(Prediction.case_id == case.id))
        assert count == 1
    detail = client.get(f"/api/v1/cases/{built['case']['id']}").json()
    assert detail["prediction"]["view_type"] == "whole"
    assert detail["prediction"]["experimental_closed_set"] is True
    assert detail["prediction"]["open_set_calibrated"] is False
    assert detail["result_display"] == "正品特征吻合度高"


def test_counterfeit_is_suspected_only_without_species_claim(
    client: TestClient, app_settings: Settings, session_factory
):
    account = register(client)
    built = _build_case_with_images(client, app_settings, account)
    outcome = _run(built["job_id"], app_settings, lambda _b: _payload("counterfeit", 0.88))
    assert outcome["case_status"] == "completed"
    with session_factory() as db:
        case = db.get(Case, UUID(built["case"]["id"]))
        assert case.result_type == ResultType.SuspectedCounterfeit
        prediction = db.get(Prediction, case.current_prediction_id)
        assert prediction.actual_species_verified is False
        # 阶段 A 不声称伪品实际物种：payload 无 actual_species 输出字段
        assert "actual_species" not in prediction.payload
        assert "欧当归" not in prediction.message and "实际应为" not in prediction.message
    detail = client.get(f"/api/v1/cases/{built['case']['id']}").json()
    assert detail["result_display"] == "疑似伪品（无法确认实际物种，请人工复核）"


def test_worker_failure_never_produces_conclusion(
    client: TestClient, app_settings: Settings, session_factory
):
    account = register(client)
    built = _build_case_with_images(client, app_settings, account)

    def broken(_b: bytes) -> dict:
        raise InferenceError("model_load_failed", "模型装配失败")

    outcome = _run(built["job_id"], app_settings, broken)
    assert outcome["status"] == "failed"
    with session_factory() as db:
        case = db.get(Case, UUID(built["case"]["id"]))
        assert case.status == CaseStatus.Failed
        assert case.current_prediction_id is None
        assert case.result_type is None
        job = db.get(AnalysisJob, UUID(built["job_id"]))
        assert job.error_category == "model_load_failed"
        predictions = db.scalars(
            select(Prediction).where(Prediction.case_id == case.id)
        ).all()
        assert predictions == []
    detail = client.get(f"/api/v1/cases/{built['case']['id']}").json()
    assert detail["prediction"] is None
    assert detail["result_display"] is None
    assert detail["failure"]["category"] == "model_load_failed"
    # 失败响应只有工程类别，没有任何真伪/百分比结论
    serialized = str(detail)
    assert "正品" not in serialized and "%" not in serialized


def test_worker_is_idempotent_across_redelivery(
    client: TestClient, app_settings: Settings, session_factory
):
    account = register(client)
    built = _build_case_with_images(client, app_settings, account)
    captured: list[bytes] = []

    first = _run(built["job_id"], app_settings, lambda _b: _payload("authentic"), captured)
    second = _run(built["job_id"], app_settings, lambda _b: _payload("authentic"), captured)

    assert first["status"] == "succeeded"
    assert second["status"] == "succeeded"
    assert len(captured) == 1  # 重投递不会再次推理、不会重复建预测
    with session_factory() as db:
        count = db.scalar(
            select(func.count()).select_from(Prediction).where(
                Prediction.analysis_job_id == UUID(built["job_id"])
            )
        )
        assert count == 1


def test_low_confidence_maps_to_inconclusive(client: TestClient, app_settings: Settings):
    account = register(client)
    built = _build_case_with_images(client, app_settings, account)
    strict = app_settings.model_copy(update={"closed_set_min_confidence": 0.6})
    outcome = _run(built["job_id"], strict, lambda _b: _payload("authentic", 0.55))
    assert outcome["case_status"] == "inconclusive"


def test_compute_case_outcome_rejects_unknown_label():
    payload = _payload("authentic")
    payload["authenticity"]["label"] = "unknown"
    try:
        compute_case_outcome(payload, 0.0)
    except InferenceError as exc:
        assert exc.category == "prediction_failed"
    else:  # pragma: no cover
        raise AssertionError("unknown label should raise")


def test_compute_case_outcome_rejects_missing_flags():
    payload = _payload("authentic")
    payload["experimental_closed_set"] = False
    try:
        compute_case_outcome(payload, 0.0)
    except InferenceError as exc:
        assert exc.category == "prediction_failed"
    else:  # pragma: no cover
        raise AssertionError("missing experimental flags should raise")


def test_running_duplicate_delivery_is_skipped_without_second_inference(
    client: TestClient, app_settings: Settings, session_factory
):
    """重复投递：另一 Worker 正在跑（running 且未超时）时绝不并发推理第二次。"""
    account = register(client)
    built = _build_case_with_images(client, app_settings, account)
    job_id = UUID(built["job_id"])
    with session_factory() as db:
        job = db.get(AnalysisJob, job_id)
        job.status = AnalysisStatus.Running  # 模拟第一个 Worker 已领取并在跑
        job.started_at = datetime.now(timezone.utc)
        job.attempts = 1
        case = db.get(Case, UUID(built["case"]["id"]))
        case.status = CaseStatus.Analyzing
        db.commit()

    captured: list[bytes] = []
    outcome = _run(built["job_id"], app_settings, lambda _b: _payload("authentic"), captured)

    assert outcome["status"] == "running"
    assert len(captured) == 0  # 没有第二次推理
    with session_factory() as db:
        job = db.get(AnalysisJob, job_id)
        assert job.status == AnalysisStatus.Running
        assert job.attempts == 1  # 未再次领取，attempts 不被重复累加
        count = db.scalar(
            select(func.count()).select_from(Prediction).where(Prediction.analysis_job_id == job_id)
        )
        assert count == 0


def test_worker_reclaims_stale_running_job_after_crash(
    client: TestClient, app_settings: Settings, session_factory
):
    """崩溃恢复：running 超过领取超时视为执行者崩溃，重领后继续完成。"""
    account = register(client)
    built = _build_case_with_images(client, app_settings, account)
    job_id = UUID(built["job_id"])
    with session_factory() as db:
        job = db.get(AnalysisJob, job_id)
        job.status = AnalysisStatus.Running
        job.started_at = datetime.now(timezone.utc) - timedelta(hours=2)  # 崩溃遗留
        job.attempts = 1
        case = db.get(Case, UUID(built["case"]["id"]))
        case.status = CaseStatus.Analyzing
        db.commit()

    captured: list[bytes] = []
    outcome = _run(built["job_id"], app_settings, lambda _b: _payload("authentic"), captured)

    assert outcome["status"] == "succeeded"
    assert outcome["case_status"] == "completed"
    assert len(captured) == 1
    with session_factory() as db:
        job = db.get(AnalysisJob, job_id)
        assert job.status == AnalysisStatus.Succeeded
        assert job.attempts == 2  # 崩溃恢复 = 第二次领取
        case = db.get(Case, UUID(built["case"]["id"]))
        assert case.status == CaseStatus.Completed
        count = db.scalar(
            select(func.count()).select_from(Prediction).where(Prediction.analysis_job_id == job_id)
        )
        assert count == 1  # 唯一约束下也只有一个预测


def test_concurrent_worker_deliveries_run_single_inference(
    client: TestClient, app_settings: Settings, session_factory
):
    """两个 Worker 并发投递同一 job：原子领取保证只推理一次、只落一条预测。"""
    account = register(client)
    built = _build_case_with_images(client, app_settings, account)
    captured: list[bytes] = []

    def execute(_: int) -> dict:
        return _run(built["job_id"], app_settings, lambda _b: _payload("authentic"), captured)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(execute, (0, 1)))

    statuses = sorted(result["status"] for result in results)
    assert set(statuses) <= {"succeeded", "running"}
    assert len(captured) == 1  # 模型只被调用一次
    with session_factory() as db:
        count = db.scalar(
            select(func.count()).select_from(Prediction).where(
                Prediction.analysis_job_id == UUID(built["job_id"])
            )
        )
        assert count == 1
