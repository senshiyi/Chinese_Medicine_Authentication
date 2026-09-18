"""中药三任务模型训练 CLI；重量级依赖仅在参数校验后加载。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections.abc import Mapping
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
PROJECT_ROOT = SCRIPT_DIR.parent.parent
LEGACY_MODEL_PATHS = (
    PROJECT_ROOT / "项目" / "Models" / "best_model.pth",
    PROJECT_ROOT / "项目" / "Models" / "epoch_001.pth",
)


def legacy_main():
    parser = argparse.ArgumentParser(description="中药真伪判别 — 模型训练")
    import torch
    from config import Config
    from dataset import HerbDataset
    from model import create_model
    from trainer import Trainer

    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "config.yaml"),
        help="配置文件路径",
    )
    parser.add_argument("--device", type=str, default=None, help="训练设备 (cuda:0 / cpu)")
    args = parser.parse_args()

    # ── 加载配置 ──────────────────────────────────────
    cfg = Config.from_yaml(args.config)
    cfg.resolve_paths(PROJECT_ROOT)

    # ── 设备 ──────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ── 数据 ──────────────────────────────────────────
    print(f"\n加载数据: {cfg.data.root}")
    dataset = HerbDataset(
        data_root=cfg.data.root,
        image_size=cfg.data.image_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        use_weighted_sampler=cfg.data.use_weighted_sampler,
    )

    # 自动检测类别数
    num_classes = cfg.model.num_classes or dataset.num_classes
    print(f"\n类别数: {num_classes}")

    dataset.split()
    train_loader = dataset.get_train_loader()
    val_loader = dataset.get_val_loader()

    # ── 模型 ──────────────────────────────────────────
    print(f"\n创建模型: backbone={cfg.model.backbone}, classes={num_classes}")
    model = create_model(
        backbone=cfg.model.backbone,
        num_classes=num_classes,
        pretrained=cfg.model.pretrained,
        freeze_backbone=cfg.model.freeze_backbone,
        dropout=cfg.model.dropout,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    # ── 训练 ──────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg.train,
        logging_config=cfg.logging,
        device=device,
    )

    history = trainer.fit(
        unfreeze_epoch=cfg.model.unfreeze_epochs if not cfg.model.freeze_backbone else None
    )

    print(f"\n最终最佳验证准确率: {history['val_acc'][-1]:.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="中药图片三任务模型训练")
    parser.add_argument(
        "--config", type=Path, default=SCRIPT_DIR / "config.yaml", help="配置文件路径"
    )
    parser.add_argument("--output", type=Path, help="新版 checkpoint 输出路径")
    parser.add_argument("--resume", type=Path, help="从可信新版 checkpoint 恢复")
    parser.add_argument("--device", help="训练设备，例如 cpu、cuda 或 cuda:0")
    parser.add_argument("--run-id", help="审计运行 ID；默认由时间戳和配置摘要生成")
    return parser


def _is_legacy_path(path: Path) -> bool:
    resolved = path.expanduser().resolve()
    return any(resolved == legacy.resolve() for legacy in LEGACY_MODEL_PATHS)


def main(argv: list[str] | None = None) -> int:
    from herb_authentication.environment import configure_utf8_stdio, run_bootstrap_preflight

    configure_utf8_stdio()
    if not run_bootstrap_preflight().python_supported:
        print("当前 Python 版本不受支持，请使用 3.11 或 3.12。", file=sys.stderr)
        return 2
    args = build_parser().parse_args(argv)
    if args.output is not None and _is_legacy_path(args.output):
        print(f"不允许覆盖旧模型：{args.output}", file=sys.stderr)
        return 2
    try:
        return _run_training(args)
    except Exception as exc:
        print(f"训练失败：{exc}", file=sys.stderr)
        return 2


def _load_split_assignment(path: Path):
    from herb_authentication.data import split_assignment_from_payload
    from herb_authentication.errors import ManifestValidationError

    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(value, Mapping):
            raise TypeError("split payload 必须是 JSON 对象")
        return split_assignment_from_payload(value)
    except (OSError, UnicodeError, json.JSONDecodeError, ManifestValidationError, TypeError) as exc:
        raise ValueError(f"锁定 split 字段不完整或无法读取：{path}") from exc


def _configured_split_version(config) -> str:
    return str(getattr(config.data, "split_algorithm_version", "group-stratified-v1"))


def _require_configured_split_version(config, assignment, path: Path):
    configured = _configured_split_version(config)
    if assignment.algorithm_version != configured:
        raise ValueError(
            "切分算法版本与配置不一致："
            f"config={configured!r}, splits.json={assignment.algorithm_version!r}, path={path}"
        )
    return assignment


def _preflight_split_assignment(config, path: Path):
    configured = _configured_split_version(config)
    if path.is_file():
        return _require_configured_split_version(config, _load_split_assignment(path), path)
    if configured == "group-stratified-near-duplicate-components-v2":
        raise ValueError(f"V2 配置缺少锁定 split：{path}；请先运行 prepare_closed_set.py。")
    return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_fingerprints(config, artifacts) -> dict[str, str]:
    values = {
        "manifest": _sha256_file(config.data.manifest_path),
        "known_validation": _sha256_file(artifacts.val_csv),
        "locked_split": _sha256_file(artifacts.splits_json),
    }
    unknown = config.data.unknown_manifest_path
    if unknown is not None:
        values["unknown_calibration"] = _sha256_file(unknown)
    return values


def _training_config_fingerprint(config) -> str:
    payload = {
        "data": {
            "image_size": config.data.image_size,
            "batch_size": config.data.batch_size,
            "num_workers": config.data.num_workers,
            "seed": config.data.seed,
            "split_algorithm_version": _configured_split_version(config),
        },
        "model": {
            "backbone": config.model.backbone,
            "pretrained": config.model.pretrained,
            "dropout": config.model.dropout,
            "freeze_backbone": config.model.freeze_backbone,
            "unfreeze_epoch": config.model.unfreeze_epoch,
        },
        "training": {
            name: getattr(config.training, name, None)
            for name in (
                "epochs",
                "learning_rate",
                "weight_decay",
                "claimed_herb_weight",
                "actual_species_weight",
                "authenticity_weight",
                "label_smoothing",
                "gradient_clip",
                "mixed_precision",
                "early_stop_patience",
                "min_species_supervision_coverage",
                "use_authenticity_class_weights",
            )
        },
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


SELECTION_FORMULA = "(authenticity_macro_f1 + authenticity_balanced_accuracy) / 2.0"


def _entity_metrics_payload(metrics) -> dict[str, object] | None:
    if metrics is None:
        return None
    return {
        "claimed_macro_f1": metrics.claimed_macro_f1,
        "authenticity_macro_f1": metrics.authenticity_macro_f1,
        "authenticity_balanced_accuracy": metrics.authenticity_balanced_accuracy,
        "species_macro_f1": metrics.species_macro_f1,
        "species_supervision_coverage": metrics.species_supervision_coverage,
        "selection_score": metrics.selection_score,
        "selection_tasks": list(metrics.selection_tasks),
    }


def _epoch_payload(result) -> dict[str, object]:
    return {
        "losses": {
            "total": result.total_loss,
            "claimed_herb": result.claimed_herb_loss,
            "actual_species": result.actual_species_loss,
            "authenticity": result.authenticity_loss,
        },
        "diagnostic_metrics": _entity_metrics_payload(result.entity_metrics),
        "species_supervised_rows": result.species_supervised_rows,
        "batches": result.batches,
        "samples": result.samples,
        "freeze_stage": result.freeze_stage,
        "learning_rates": list(result.learning_rates),
        "duration_seconds": result.duration_seconds,
    }


def _write_training_history(
    run_root: Path,
    summary,
    *,
    class_counts: dict[str, int],
    class_weights: list[float] | None,
) -> Path:
    validation_by_epoch = {result.epoch: result for result in summary.validation_epochs}
    epochs = []
    for train_result in summary.train_epochs:
        validation_result = validation_by_epoch.get(train_result.epoch)
        if validation_result is None:
            raise ValueError(f"epoch={train_result.epoch} 缺少验证历史")
        epochs.append(
            {
                "epoch": train_result.epoch,
                "training": _epoch_payload(train_result),
                "validation": _epoch_payload(validation_result),
            }
        )
    payload = {
        "selection_formula": SELECTION_FORMULA,
        "best_epoch": summary.best_epoch,
        "best_selection_score": summary.best_selection_score,
        "early_stop_reason": (
            "patience_exhausted" if summary.stopped_early else "epochs_completed"
        ),
        "authenticity_class_counts": class_counts,
        "authenticity_class_weights": class_weights,
        "epochs": epochs,
    }
    _require_finite_numbers(payload)
    destination = run_root / "training_history.json"
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def _require_finite_numbers(value) -> None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise ValueError("training_history 包含非有限数值")
    elif isinstance(value, dict):
        for item in value.values():
            _require_finite_numbers(item)
    elif isinstance(value, list | tuple):
        for item in value:
            _require_finite_numbers(item)


def _resume_provenance(resume_from: Path | None) -> dict[str, object] | None:
    if resume_from is None:
        return None
    import torch

    try:
        checkpoint_path = resume_from.expanduser().resolve(strict=True)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"无法读取父 checkpoint provenance：{resume_from}") from exc
    if not isinstance(state, Mapping):
        raise ValueError("父 checkpoint 必须是映射")
    metadata = state.get("metadata")
    training = state.get("training")
    if not isinstance(metadata, Mapping) or not isinstance(training, Mapping):
        raise ValueError("父 checkpoint 缺少 metadata 或 training")
    parent_run_id = metadata.get("run_id")
    if not isinstance(parent_run_id, str) or not parent_run_id:
        raise ValueError("父 checkpoint 缺少有效 parent run_id")
    parent_epoch = training.get("epoch")
    if isinstance(parent_epoch, bool) or not isinstance(parent_epoch, int) or parent_epoch < 0:
        raise ValueError("父 checkpoint epoch 必须是非负整数")
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "parent_run_id": parent_run_id,
        "start_epoch": parent_epoch + 1,
    }


def _run_training(args: argparse.Namespace) -> int:
    import numpy as np
    import torch
    from herb_authentication.config import load_config
    from herb_authentication.data import (
        HerbImageDataset,
        apply_split,
        build_eval_transform,
        build_label_space,
        build_train_transform,
        create_group_splits,
        create_herb_dataloader,
        validate_manifest_pair,
        write_split_artifacts,
    )
    from herb_authentication.models import MultiTaskHerbModel, build_torchvision_encoder
    from herb_authentication.training.checkpoints import CheckpointContract
    from herb_authentication.training.losses import LossWeights
    from herb_authentication.training.run_artifacts import (
        create_run_context,
        write_run_metadata,
    )
    from herb_authentication.training.trainer import Trainer, compute_authenticity_class_weights

    config = load_config(args.config)
    split_path = config.data.manifest_path.parent / "splits.json"
    assignment = _preflight_split_assignment(config, split_path)
    run_context = create_run_context(config, getattr(args, "run_id", None))
    random.seed(config.data.seed)
    np.random.seed(config.data.seed)
    torch.manual_seed(config.data.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.data.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    output = (args.output or config.training.checkpoint_path).expanduser().resolve()
    if _is_legacy_path(output):
        raise ValueError(f"不允许覆盖旧模型：{output}")
    bundle = validate_manifest_pair(
        config.data.manifest_path,
        unknown_manifest_path=config.data.unknown_manifest_path,
        data_root=config.data.data_root,
        report_path=config.training.log_dir / "validation_report.json",
        encoding=config.data.text_encoding,
        max_image_bytes=config.data.max_image_bytes,
        max_image_pixels=config.data.max_image_pixels,
        perceptual_hash_distance=config.data.perceptual_hash_distance,
    )
    if assignment is None:
        assignment = create_group_splits(
            bundle.known,
            (config.data.train_ratio, config.data.val_ratio, config.data.test_ratio),
            seed=config.data.seed,
            perceptual_hash_distance=config.data.perceptual_hash_distance,
            algorithm_version="group-stratified-v1",
        )
    artifacts = write_split_artifacts(bundle.known, assignment, split_path.parent)
    rows = apply_split(bundle.known, assignment)
    labels = build_label_space(rows)
    train_rows = tuple(row for row in rows if row.partition == "train")
    val_rows = tuple(row for row in rows if row.partition == "val")
    authenticity_targets = tuple(
        labels.authenticity_classes.index(row.row.authenticity) for row in train_rows
    )
    authenticity_class_counts = {
        class_name: authenticity_targets.count(index)
        for index, class_name in enumerate(labels.authenticity_classes)
    }
    if any(count == 0 for count in authenticity_class_counts.values()):
        raise ValueError("训练集必须同时包含两个 authenticity 类别")
    authenticity_class_weight = (
        compute_authenticity_class_weights(authenticity_targets)
        if config.training.use_authenticity_class_weights
        else None
    )
    applied_class_weights = (
        None
        if authenticity_class_weight is None
        else [float(value) for value in authenticity_class_weight.tolist()]
    )
    counterfeit_species_verified = all(
        row.row.species_annotation_status == "verified"
        for row in train_rows
        if row.row.authenticity == "counterfeit"
    )
    train_dataset = HerbImageDataset(
        train_rows, labels, build_train_transform(config.data.image_size)
    )
    val_dataset = HerbImageDataset(val_rows, labels, build_eval_transform(config.data.image_size))
    train_loader = create_herb_dataloader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        seed=config.data.seed,
        num_workers=config.data.num_workers,
    )
    val_loader = create_herb_dataloader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        seed=config.data.seed,
        num_workers=config.data.num_workers,
    )
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    encoder = build_torchvision_encoder(config.model.backbone, pretrained=config.model.pretrained)
    model = MultiTaskHerbModel(
        encoder.encoder,
        encoder.feature_dim,
        len(labels.claimed_classes),
        len(labels.actual_species_classes),
        dropout=config.model.dropout,
    )
    contract = CheckpointContract(
        schema_version=1,
        backbone=config.model.backbone,
        claimed_herb_classes=labels.claimed_classes,
        actual_species_classes=labels.actual_species_classes,
        authenticity_classes=labels.authenticity_classes,
        image_size=config.data.image_size,
        data_fingerprint=assignment.input_fingerprint,
        split_fingerprint=_sha256_file(artifacts.splits_json),
        training_config_fingerprint=_training_config_fingerprint(config),
        preprocessing_version="imagenet-v1",
    )
    resume_provenance = _resume_provenance(args.resume)
    write_run_metadata(
        run_context,
        {
            "run_id": run_context.run_id,
            "checkpoint": str(output),
            "selection_formula": SELECTION_FORMULA,
            "counterfeit_species_verified": counterfeit_species_verified,
            "authenticity_class_counts": authenticity_class_counts,
            "authenticity_class_weights": applied_class_weights,
            "resume_from": resume_provenance,
        },
    )
    trainer = Trainer(
        model,
        epochs=config.training.epochs,
        learning_rate=config.training.learning_rate,
        encoder_learning_rate=config.training.learning_rate * 0.1,
        weight_decay=config.training.weight_decay,
        freeze_backbone=config.model.freeze_backbone,
        unfreeze_epoch=config.model.unfreeze_epoch,
        loss_weights=LossWeights(
            claimed_herb=config.training.claimed_herb_weight,
            actual_species=config.training.actual_species_weight,
            authenticity=config.training.authenticity_weight,
        ),
        label_smoothing=config.training.label_smoothing,
        gradient_clip=config.training.gradient_clip,
        mixed_precision=config.training.mixed_precision,
        device=device,
        checkpoint_path=output,
        checkpoint_contract=contract,
        authenticity_class_weight=authenticity_class_weight,
        checkpoint_metadata={
            "run_id": run_context.run_id,
            "selection_formula": SELECTION_FORMULA,
            "counterfeit_species_verified": counterfeit_species_verified,
            "resume_from": resume_provenance,
            "provenance": config.data.provenance,
            "source_encoding": bundle.known_encoding,
            "split_input_fingerprint": assignment.input_fingerprint,
            "calibration": {"is_calibrated": False, "method": None, "parameters": {}},
            "split_artifact_fingerprint": _sha256_file(artifacts.splits_json),
            "fingerprints": _checkpoint_fingerprints(config, artifacts),
            "reproducibility": {
                "seed": config.data.seed,
                "deterministic_algorithms": True,
                "deterministic_warn_only": True,
                "cudnn_benchmark": False,
            },
        },
        protected_checkpoint_paths=LEGACY_MODEL_PATHS,
        early_stop_patience=getattr(config.training, "early_stop_patience", None),
        min_species_supervision_coverage=getattr(
            config.training, "min_species_supervision_coverage", 0.2
        ),
    )
    summary = trainer.fit(train_loader, val_loader, resume_from=args.resume)
    history_path = _write_training_history(
        run_context.root,
        summary,
        class_counts=authenticity_class_counts,
        class_weights=applied_class_weights,
    )
    print(
        json.dumps(
            {
                "run_id": run_context.run_id,
                "checkpoint": str(output),
                "training_history": str(history_path),
                "epochs_completed": len(summary.train_epochs),
                "device": summary.device,
                "amp_enabled": summary.amp_enabled,
                "best_epoch": getattr(summary, "best_epoch", None),
                "best_selection_score": getattr(summary, "best_selection_score", None),
                "stopped_early": getattr(summary, "stopped_early", False),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
