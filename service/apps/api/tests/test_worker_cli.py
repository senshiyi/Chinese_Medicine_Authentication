"""本地无 broker Worker（drynaria-cli worker）轮询行为测试。

场景：API 以 `DATABASE_URL=sqlite… + STORAGE_BACKEND=local + CELERY_BROKER_URL=空`
运行时，analyze 只把 AnalysisJob 落库为 pending，没有任何 Celery 消费者 → 任务
卡在 queued。本地轮询 Worker（process_pending_analysis_jobs_once）必须把 pending
job 执行掉；模型用注入的假预测替代，测试里绝不加载真实 checkpoint。
"""

from __future__ import annotations

import argparse
from uuid import UUID

import drynaria_api.cli as cli
from drynaria_api.inference_adapter import InferenceError
from drynaria_api.models import (
    AnalysisJob,
    AnalysisStatus,
    Case,
    CaseStatus,
    Prediction,
    ResultType,
)
from drynaria_api.settings import Settings
from drynaria_api.worker import process_pending_analysis_jobs_once

from helpers import analyze_case, create_case, register, upload_image


def _payload(label: str = "authentic", confidence: float = 0.91) -> dict:
    probabilities = (
        {"authentic": confidence, "counterfeit": round(1.0 - confidence, 6)}
        if label == "authentic"
        else {"authentic": round(1.0 - confidence, 6), "counterfeit": confidence}
    )
    return {
        "schema_version": 1,
        "model_version": "closed-set-v1",
        "experimental_closed_set": True,
        "open_set_calibrated": False,
        "scope_warning": "仅适用于本数据集定义的骨碎补正品/伪品范围",
        "message": "此药品为正品。",
        "authenticity": {"label": label, "confidence": confidence},
        "probabilities": probabilities,
        "actual_species_verified": False,
        "model_fingerprint": "a" * 64,
        "data_fingerprint": "b" * 64,
        "split_fingerprint": "c" * 64,
    }


def _queue_case(client, account: dict) -> dict:
    """建 case + 传 whole 主图 + analyze → 无 broker 时任务落库为 pending。"""
    case = create_case(client, account)
    upload_image(client, case["id"], "whole", account)
    return analyze_case(client, case["id"], account)


def test_local_worker_processes_queued_pending_job(
    client, app_settings, session_factory
):
    account = register(client)
    queued = _queue_case(client, account)
    job_id = queued["analysis_job_id"]
    case_id = queued["case_id"]

    # 前置：无 broker → job 持久化为 pending、案件 queued，等待本地 Worker 消费
    with session_factory() as db:
        assert db.get(AnalysisJob, UUID(job_id)).status == AnalysisStatus.Pending
        assert db.get(Case, UUID(case_id)).status == CaseStatus.Queued

    captured: list[bytes] = []

    def fake_predict(data: bytes) -> dict:
        captured.append(data)
        return _payload("authentic", 0.91)

    summary = process_pending_analysis_jobs_once(settings=app_settings, predict=fake_predict)

    assert summary == {"attempted": 1, "succeeded": 1, "failed": 0, "skipped": 0}
    assert len(captured) == 1  # 注入的假预测只被调用一次
    with session_factory() as db:
        job = db.get(AnalysisJob, UUID(job_id))
        assert job.status == AnalysisStatus.Succeeded
        assert job.attempts == 1
        case = db.get(Case, UUID(case_id))
        assert case.status == CaseStatus.Completed
        assert case.result_type == ResultType.AuthenticLikely
        prediction = db.get(Prediction, case.current_prediction_id)
        assert prediction.authenticity_label == "authentic"
        assert prediction.model_version == "closed-set-v1"

    # 幂等：已成功任务不再被轮询拾取（防止重复推理/重复建预测）
    second = process_pending_analysis_jobs_once(
        settings=app_settings, predict=lambda _b: _payload()
    )
    assert second["attempted"] == 0


def test_local_worker_job_failure_does_not_block_other_jobs(
    client, app_settings, session_factory
):
    account = register(client)
    job_ids = [_queue_case(client, account)["analysis_job_id"] for _ in range(2)]

    calls: list[bytes] = []

    def flaky_predict(data: bytes) -> dict:
        if not calls:  # 第一个任务模拟模型装配失败
            calls.append(data)
            raise InferenceError("model_load_failed", "注入的模型装配失败")
        calls.append(data)
        return _payload("authentic", 0.91)

    summary = process_pending_analysis_jobs_once(settings=app_settings, predict=flaky_predict)

    assert summary["attempted"] == 2
    assert summary["failed"] == 1
    assert summary["succeeded"] == 1  # 一个失败不杀死本轮，其余任务照常执行
    with session_factory() as db:
        statuses = {db.get(AnalysisJob, UUID(jid)).status for jid in job_ids}
        assert statuses == {AnalysisStatus.Failed, AnalysisStatus.Succeeded}


def _settings(tmp_path, **overrides) -> Settings:
    kwargs = dict(
        environment="dev",
        database_url=f"sqlite:///{(tmp_path / 'guard.db').as_posix()}",
        storage_backend="local",
        storage_local_dir=tmp_path / "storage",
        celery_broker_url="",
        celery_result_backend="",
        auto_create_tables=False,
        cookie_secure=False,
        allowed_origins=["http://testserver"],
    )
    kwargs.update(overrides)
    return Settings(_env_file=None, **kwargs)


def test_mode_guard_blocks_prod_and_nonempty_broker(tmp_path):
    assert cli._local_worker_block_reason(_settings(tmp_path, environment="prod")) is not None
    assert (
        cli._local_worker_block_reason(
            _settings(tmp_path, celery_broker_url="redis://localhost:6379/0")
        )
        is not None
    )
    assert cli._local_worker_block_reason(_settings(tmp_path)) is None


def test_cmd_worker_refuses_blocked_modes(monkeypatch, tmp_path):
    namespace = argparse.Namespace(
        once=True,
        poll_seconds=2.0,
        database_url=None,
        storage_backend=None,
        storage_local_dir=None,
        s3_endpoint_url=None,
        s3_bucket=None,
    )
    monkeypatch.setattr(
        cli, "_settings_from_args", lambda _args: _settings(tmp_path, environment="prod")
    )
    assert cli.cmd_worker(namespace) == 2
    monkeypatch.setattr(
        cli,
        "_settings_from_args",
        lambda _args: _settings(tmp_path, celery_broker_url="redis://localhost:6379/0"),
    )
    assert cli.cmd_worker(namespace) == 2
