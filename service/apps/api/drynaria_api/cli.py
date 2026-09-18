"""运维/开发 CLI：drynaria-cli create-admin / purge-images / schema-create / worker。

用法示例：
    drynaria-cli create-admin --email admin@example.com --password '...'
    drynaria-cli purge-images --retention-days 30 --dry-run
    drynaria-cli worker --poll-seconds 2      # 本地无 broker 预览的轮询 Worker
    drynaria-cli worker --once                # 只处理一轮（脚本/自检）
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import sys
import time
from pathlib import Path

from .database import create_all, make_session_factory
from .models import UserRole
from .privacy import purge_expired_originals
from .security import new_user
from .settings import Settings, get_settings
from .storage import build_storage


def _settings_from_args(args: argparse.Namespace) -> Settings:
    kwargs: dict = {}
    for field in ("database_url", "storage_backend", "storage_local_dir", "s3_endpoint_url", "s3_bucket"):
        value = getattr(args, field, None)
        if value:
            kwargs[field] = value
    if not kwargs:
        return get_settings()
    base = get_settings()
    merged = base.model_copy(update=kwargs)
    return merged


def _add_env_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--database-url", default=None, help="覆盖 DATABASE_URL")
    parser.add_argument("--storage-backend", default=None, help="local | s3 | auto")
    parser.add_argument("--storage-local-dir", default=None, help="本地存储目录")
    parser.add_argument("--s3-endpoint-url", default=None)
    parser.add_argument("--s3-bucket", default=None)


def _add_scope_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", default=None, help="原闭集 checkpoint 路径")
    parser.add_argument("--inference-project", default=None, help="原项目 herb_authentication 目录")
    parser.add_argument("--device", default=None, help="worker_device（auto/cpu/cuda）")


def _scope_settings(args: argparse.Namespace) -> Settings:
    updates: dict = {}
    if getattr(args, "checkpoint", None):
        updates["model_checkpoint_path"] = args.checkpoint
    if getattr(args, "inference_project", None):
        updates["inference_project_path"] = args.inference_project
    if getattr(args, "device", None):
        updates["worker_device"] = args.device
    if not updates:
        return get_settings()
    return get_settings().model_copy(update=updates)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cmd_scope_validate_manifest(args: argparse.Namespace) -> int:
    from .scope.manifest import validate_manifest

    report = validate_manifest(args.manifest, args.image_root)
    print("scope manifest OK")
    print(f"rows={report['rows']} unique_content={report['unique_content']}")
    for split, counts in report["counts"].items():
        print(
            f"  {split}: total={counts['total']} in_scope={counts['in_scope']} "
            f"out_of_scope={counts['out_of_scope']}"
        )
    return 0


def cmd_scope_train(args: argparse.Namespace) -> int:
    settings = _scope_settings(args)
    from .scope.artifact import save_artifact
    from .scope.features import build_feature_extractor
    from .scope.pipeline import run_scope_training

    checkpoint = Path(args.checkpoint or settings.model_checkpoint_path).expanduser()
    if not checkpoint.is_file():
        print(f"checkpoint 不存在：{checkpoint}", file=sys.stderr)
        return 2
    feature_extractor = build_feature_extractor(settings)
    result = run_scope_training(
        args.manifest,
        args.image_root,
        feature_extract=feature_extractor,
        artifact_version=args.artifact_version,
        checkpoint_sha256=_sha256_file(checkpoint),
        feature_dim=int(feature_extractor.feature_dim),
        max_false_accept=args.max_false_accept,
        max_false_reject=args.max_false_reject,
    )
    save_artifact(result["artifact"], args.out)
    report = result["report"]
    print(
        f"scope-train done: version={report['artifact_version']} "
        f"approved={report['approved']} threshold={report['threshold']:.4f} "
        f"feature_dim={report['feature_dim']}"
    )
    test_eval = report["evaluation"]["test"]
    print(
        f"  test: false_accept={test_eval['false_accept_rate']:.4f} "
        f"(CI95 {test_eval['false_accept_ci95']}) "
        f"false_reject={test_eval['false_reject_rate']:.4f} "
        f"(CI95 {test_eval['false_reject_ci95']}) satisfied={test_eval['satisfied']}"
    )
    if not report["approved"]:
        print(
            "  工件被标记为候选（未通过 held-out 上限），不可部署；"
            "不得据此声称通用 OOD 防护。",
            file=sys.stderr,
        )
    return 0


def cmd_scope_verify(args: argparse.Namespace) -> int:
    from .scope.artifact import load_artifact, verify_artifact

    artifact = load_artifact(args.artifact)
    checkpoint = Path(args.checkpoint).expanduser()
    if not checkpoint.is_file():
        print(f"checkpoint 不存在：{checkpoint}", file=sys.stderr)
        return 2
    verify_artifact(artifact, checkpoint_sha256=_sha256_file(checkpoint))
    print(
        f"scope artifact OK: version={artifact.artifact_version} "
        f"approved={artifact.approved} feature_dim={artifact.feature_dim} "
        f"threshold={artifact.threshold:.4f}"
    )
    return 0


def cmd_create_admin(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    create_all(settings.database_url)
    email = (args.email or "").strip().lower()
    password = args.password
    if not password:
        password = getpass.getpass("Admin password: ")
    if not email or len(password) < 8:
        print("需要 email 和至少 8 位密码", file=sys.stderr)
        return 2
    factory = make_session_factory(settings.database_url)
    with factory() as db:
        from .security import lookup_user_by_email

        if lookup_user_by_email(db, email) is not None:
            print(f"用户已存在: {email}", file=sys.stderr)
            return 2
        user = new_user(db, email=email, password=password, role=UserRole.Admin, display_name="管理员")
        db.commit()
        print(f"created admin {user.id} {email}")
    return 0


def cmd_purge(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    stats = purge_expired_originals(
        session_factory=make_session_factory(settings.database_url),
        storage=build_storage(settings),
        retention_days=args.retention_days,
        dry_run=bool(args.dry_run),
    )
    print(stats)
    return 0


def cmd_schema_create(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    create_all(settings.database_url)
    print("tables created")
    return 0


def _local_worker_block_reason(settings: Settings) -> str | None:
    """返回拒绝本地轮询 Worker 启动的原因；None 表示允许。

    本地轮询 Worker 是无 broker 本地预览的专用消费者：生产模式必须走 Celery，
    且 CELERY_BROKER_URL 非空时任务已由 Celery 消费，两者并行会竞争领取同一批
    pending 任务（run_analysis_job 的原子领取虽能兜底，但禁止混用两种模式）。
    """
    if settings.environment == "prod":
        return "生产模式（environment=prod）禁止本地轮询 Worker，请使用 Celery Worker 消费任务"
    if settings.celery_broker_url:
        return (
            "CELERY_BROKER_URL 非空：任务由 Celery 消费，本地轮询 Worker 会与之竞争领取"
            "同一批 pending 任务；请先清空 CELERY_BROKER_URL（无 broker 本地预览模式）再启动"
        )
    return None


def cmd_worker(args: argparse.Namespace) -> int:
    """本地开发 Worker：轮询数据库中的 pending AnalysisJob 并逐个执行。

    本命令在独立于 API 的进程中运行：模型（torch）只在本进程加载，API 进程永不
    加载模型。无 broker（CELERY_BROKER_URL 为空）时替代 Celery 成为消费者；
    生产环境或 broker 非空时拒绝启动，避免与 Celery 竞争消费模式。
    """
    settings = _settings_from_args(args)
    reason = _local_worker_block_reason(settings)
    if reason:
        print(reason, file=sys.stderr)
        return 2
    if args.poll_seconds < 0:
        print("--poll-seconds 不能为负数", file=sys.stderr)
        return 2

    # 延迟导入：只有 Worker 进程才引入 celery 任务命名空间与 inference adapter
    from .worker import process_pending_analysis_jobs_once

    once = bool(args.once)
    cadence = "单轮（--once）" if once else f"每 {args.poll_seconds:g} 秒轮询一次"
    print(
        f"本地开发 Worker 启动：{cadence}；environment={settings.environment}，"
        f"database={settings.database_url}",
        flush=True,
    )
    try:
        while True:
            summary = None
            try:
                summary = process_pending_analysis_jobs_once(settings=settings)
            except Exception as exc:  # noqa: BLE001 —— 单轮失败不退出，记录后下轮再试
                print(
                    f"轮询执行异常（已忽略，下轮重试）：{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
            if summary is not None and summary["attempted"]:
                print(
                    "本轮处理 {attempted} 个任务：succeeded={succeeded} "
                    "failed={failed} skipped={skipped}".format(**summary),
                    flush=True,
                )
            if once:
                return 0 if summary is not None else 1
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("\n本地开发 Worker 已停止（Ctrl+C）", file=sys.stderr)
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="drynaria-cli", description="骨碎补在线鉴伪阶段A 运维命令")
    sub = parser.add_subparsers(dest="command", required=True)

    p_admin = sub.add_parser("create-admin", help="创建管理员账户（角色只能由此创建）")
    p_admin.add_argument("--email", required=True)
    p_admin.add_argument("--password", default=None)
    _add_env_args(p_admin)
    p_admin.set_defaults(func=cmd_create_admin)

    p_purge = sub.add_parser("purge-images", help="按留存策略清理原图对象")
    p_purge.add_argument("--retention-days", type=int, default=None)
    p_purge.add_argument("--dry-run", action="store_true")
    _add_env_args(p_purge)
    p_purge.set_defaults(func=cmd_purge)

    p_schema = sub.add_parser("schema-create", help="（本地/测试用）直接建表，生产请用 alembic")
    _add_env_args(p_schema)
    p_schema.set_defaults(func=cmd_schema_create)

    p_worker = sub.add_parser(
        "worker",
        help="本地开发 Worker：轮询执行 pending 分析任务（无 broker 模式；生产用 Celery）",
        description=(
            "无 Celery broker 的本地/测试预览中，本命令在独立于 API 的进程里轮询数据库"
            "中处于 pending 的 AnalysisJob 并逐个执行 run_analysis_job（模型只在 Worker "
            "进程加载，API 永不加载模型）。environment=prod 或 CELERY_BROKER_URL 非空时"
            "拒绝启动，避免与 Celery 竞争消费同一批任务。"
        ),
    )
    p_worker.add_argument("--once", action="store_true", help="只执行一轮后退出（脚本/自检）")
    p_worker.add_argument(
        "--poll-seconds", type=float, default=2.0, help="轮询间隔秒数（默认 2）"
    )
    _add_env_args(p_worker)
    p_worker.set_defaults(func=cmd_worker)

    p_scope_manifest = sub.add_parser(
        "scope-validate-manifest",
        help="校验范围标注清单（列/取值/授权/分组隔离/重复内容）",
    )
    p_scope_manifest.add_argument("--manifest", required=True, help="清单 CSV 路径")
    p_scope_manifest.add_argument("--image-root", required=True, help="图片根目录")
    p_scope_manifest.set_defaults(func=cmd_scope_validate_manifest)

    p_scope_train = sub.add_parser(
        "scope-train",
        help="训练/校准/评估范围门控并写出 JSON 工件（需 sklearn + torch，原项目 venv）",
    )
    p_scope_train.add_argument("--manifest", required=True, help="清单 CSV 路径")
    p_scope_train.add_argument("--image-root", required=True, help="图片根目录")
    p_scope_train.add_argument("--out", required=True, help="输出工件 JSON 路径")
    p_scope_train.add_argument("--artifact-version", default="scope-logreg-v1")
    p_scope_train.add_argument("--max-false-accept", type=float, default=0.05)
    p_scope_train.add_argument("--max-false-reject", type=float, default=0.05)
    _add_scope_model_args(p_scope_train)
    p_scope_train.set_defaults(func=cmd_scope_train)

    p_scope_verify = sub.add_parser(
        "scope-verify", help="校验范围工件：绑定 checkpoint SHA-256 且已批准方可部署"
    )
    p_scope_verify.add_argument("--artifact", required=True, help="工件 JSON 路径")
    p_scope_verify.add_argument("--checkpoint", required=True, help="原闭集 checkpoint 路径")
    p_scope_verify.set_defaults(func=cmd_scope_verify)

    args = parser.parse_args(argv)
    if getattr(args, "retention_days", None) is None:
        args.retention_days = get_settings().image_retention_days
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
