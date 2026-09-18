"""只读、可审计的真实数据评估入口。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成中药识别评估审计报告")
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config.yaml")
    parser.add_argument("--checkpoint", type=Path, help="覆盖配置中的 checkpoint")
    parser.add_argument(
        "--partition",
        choices=("validation", "locked-test"),
        help="实验闭集评估分区；必须显式选择",
    )
    parser.add_argument("--run-dir", type=Path, help="训练 run 目录")
    parser.add_argument("--json", action="store_true", help="同时将报告输出到 stdout")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    closed_set_logits_collector: Callable[..., tuple[Any, Any]] | None = None,
) -> int:
    from herb_authentication.environment import configure_utf8_stdio, run_bootstrap_preflight

    configure_utf8_stdio()
    if not run_bootstrap_preflight().python_supported:
        print("当前 Python 版本不受支持，请使用 3.11 或 3.12。", file=sys.stderr)
        return 2
    args = build_parser().parse_args(argv)
    from herb_authentication.config import load_config
    from herb_authentication.errors import HerbAuthenticationError
    from herb_authentication.evaluation import build_evaluation_report, write_evaluation_artifacts

    try:
        config = load_config(args.config)
        _preflight_split_assignment(config, config.data.manifest_path.parent / "splits.json")
        if config.calibration.mode == "closed_set_temperature":
            if args.partition is None or args.run_dir is None or args.checkpoint is None:
                raise ValueError("实验闭集评估必须显式提供 --partition、--run-dir 和 --checkpoint")
            checkpoint = args.checkpoint.expanduser().resolve()
            metadata = _checkpoint_metadata(checkpoint)
            report, artifacts = _evaluate_closed_set_cli(
                config=config,
                checkpoint=checkpoint,
                metadata=metadata,
                partition=args.partition,
                run_dir=args.run_dir,
                closed_set_logits_collector=closed_set_logits_collector,
            )
        else:
            if args.partition is not None or args.run_dir is not None:
                raise ValueError("--partition/--run-dir 仅适用于实验闭集评估")
            checkpoint = (args.checkpoint or config.inference.checkpoint_path).resolve()
            metadata = _checkpoint_metadata(checkpoint) if checkpoint.is_file() else {}
            locked_split = config.data.manifest_path.parent / "splits.json"
            locked_fingerprint = _locked_split_fingerprint(locked_split)
            checkpoint_split = _checkpoint_split_fingerprint(metadata)
            calibration = metadata.get("calibration", {})
            known_metrics: Mapping[str, Any] | None = None
            unknown_metrics: Mapping[str, Any] | None = None
            if (
                checkpoint.is_file()
                and config.data.manifest_path.is_file()
                and locked_split.is_file()
            ):
                known_metrics, unknown_metrics = _evaluate_available_data(
                    config, checkpoint, locked_split
                )
            report = build_evaluation_report(
                data_provenance=config.data.provenance,
                checkpoint_provenance=_optional_string(metadata.get("provenance")),
                data_fingerprint=_sha256(config.data.manifest_path)
                if config.data.manifest_path.is_file()
                else None,
                checkpoint_fingerprint=_sha256(checkpoint) if checkpoint.is_file() else None,
                locked_split_fingerprint=locked_fingerprint,
                checkpoint_split_fingerprint=checkpoint_split,
                is_calibrated=bool(
                    isinstance(calibration, Mapping) and calibration.get("is_calibrated") is True
                ),
                known_metrics=known_metrics,
                unknown_metrics=unknown_metrics,
            )
            artifacts = write_evaluation_artifacts(report, config.evaluation.report_dir)
    except (HerbAuthenticationError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"评估失败：{exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"评估报告：{artifacts.json_path}")
        print(f"指标表：{artifacts.csv_path}")
        if "official_metrics" in report:
            print("正式指标：是" if report["official_metrics"] else "正式指标：否")
        else:
            print(f"闭集分区：{report['partition']}")
    return 0


def _require_fixed_promotion_thresholds(
    thresholds: Mapping[str, Any],
) -> dict[str, float]:
    required = {"macro_f1", "balanced_accuracy", "counterfeit_recall"}
    if set(thresholds) != required or any(
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isclose(float(value), 0.80, rel_tol=0.0, abs_tol=1e-12)
        for value in thresholds.values()
    ):
        raise ValueError("实验闭集三项 promotion 门槛必须固定为 0.80")
    return {name: 0.80 for name in required}


def _evaluate_closed_set_cli(
    *,
    config: Any,
    checkpoint: Path,
    metadata: Mapping[str, Any],
    partition: str,
    run_dir: Path,
    closed_set_logits_collector: Callable[..., tuple[Any, Any]] | None = None,
) -> tuple[dict[str, Any], Any]:
    resolved_run = run_dir.expanduser().resolve()
    data_dir = config.data.manifest_path.parent.resolve()
    _preflight_split_assignment(config, data_dir / "splits.json")
    thresholds = _require_fixed_promotion_thresholds(
        {
            "macro_f1": config.promotion.min_macro_f1,
            "balanced_accuracy": config.promotion.min_balanced_accuracy,
            "counterfeit_recall": config.promotion.min_counterfeit_recall,
        }
    )
    partition_manifest = data_dir / ("val.csv" if partition == "validation" else "test.csv")
    split = data_dir / "splits.json"
    source_candidate = _source_candidate_from_run(resolved_run)
    config_fingerprint = _training_config_fingerprint(config)
    fingerprint_bundle, calibration = _validate_closed_set_fingerprints(
        metadata=metadata,
        manifest=config.data.manifest_path,
        partition_manifest=partition_manifest,
        split=split,
        config_fingerprint=config_fingerprint,
        source_candidate=source_candidate,
        calibrated_checkpoint=checkpoint,
        partition=partition,
        data_root=config.data.data_root,
        encoding=config.data.text_encoding,
        max_image_bytes=config.data.max_image_bytes,
        max_image_pixels=config.data.max_image_pixels,
    )
    collector = closed_set_logits_collector or _collect_closed_set_logits
    return _execute_closed_set_partition(
        partition=partition,
        data_dir=data_dir,
        run_dir=resolved_run,
        fingerprint_bundle=fingerprint_bundle,
        temperature=calibration.temperature,
        bootstrap_samples=config.evaluation.bootstrap_samples,
        seed=config.data.seed,
        thresholds=thresholds,
        logits_collector=lambda manifest: collector(
            checkpoint=checkpoint,
            manifest=manifest,
            data_root=config.data.data_root,
            encoding=config.data.text_encoding,
            device=config.inference.device,
            split=split,
            expected_partition="val" if partition == "validation" else "test",
        ),
    )


def _source_candidate_from_run(run_dir: Path) -> Path:
    metadata_path = run_dir / "run_metadata.json"
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"run metadata 无法读取：{metadata_path}") from exc
    checkpoint = payload.get("checkpoint") if isinstance(payload, Mapping) else None
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError("run metadata 缺少源候选 checkpoint")
    return Path(checkpoint).expanduser().resolve()


def _training_config_fingerprint(config: Any) -> str:
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


def _collect_closed_set_logits(
    *,
    checkpoint: Path,
    manifest: Path,
    data_root: Path,
    encoding: str | None,
    device: str,
    split: Path,
    expected_partition: str,
) -> tuple[Any, Any]:
    import csv

    import torch
    from herb_authentication.inference.service import InferenceService

    rows: list[dict[str, str]] | None = None
    last_error: UnicodeError | None = None
    for candidate_encoding in (encoding,) if encoding else ("utf-8-sig", "gbk"):
        try:
            with manifest.open("r", encoding=candidate_encoding, newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    raise ValueError(f"CSV 缺少表头：{manifest}")
                rows = [dict(row) for row in reader]
            break
        except UnicodeError as exc:
            last_error = exc
    if rows is None:
        raise ValueError(f"CSV 无法按指定编码读取：{manifest}") from last_error
    if not rows:
        raise ValueError(f"评估分区 CSV 为空：{manifest}")
    assignment = _load_split_assignment(split)
    labels = {"authentic": 0, "counterfeit": 1}
    service = InferenceService.from_checkpoint(checkpoint, device=device)
    logits: list[torch.Tensor] = []
    targets: list[int] = []
    root = data_root.expanduser().resolve()
    for line, row in enumerate(rows, start=2):
        role = (row.get("partition") or "").strip().lower()
        if role != expected_partition:
            raise ValueError(f"{manifest.name} 第 {line} 行 partition 不属于请求分区")
        group_id = (row.get("group_id") or "").strip()
        if not group_id or assignment.by_group.get(group_id) != expected_partition:
            raise ValueError(f"{manifest.name} 第 {line} 行与 locked split 不一致")
        label = (row.get("authenticity") or "").strip()
        if label not in labels:
            raise ValueError(f"{manifest.name} 第 {line} 行 authenticity 不合法")
        raw_path = (row.get("image_path") or "").strip()
        if not raw_path:
            raise ValueError(f"{manifest.name} 第 {line} 行 image_path 为空")
        candidate = Path(raw_path).expanduser()
        image = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        if not image.is_relative_to(root):
            raise ValueError(f"{manifest.name} 第 {line} 行 image_path 越过 data_root")
        output = service.predict_logits(image)
        value = getattr(output, "authenticity", None)
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != (1, 2)
            or not bool(torch.isfinite(value).all())
        ):
            raise RuntimeError("公共 logits 接口返回的真伪输出不兼容")
        logits.append(value[0].detach().cpu())
        targets.append(labels[label])
    return torch.stack(logits), torch.tensor(targets, dtype=torch.long)


def _validate_partition_manifest_rows(
    manifest: Path,
    partition_manifest: Path,
    split_payload: Mapping[str, Any],
    expected_partition: str,
    canonical_rows: Sequence[Any] | None = None,
) -> None:
    """证明所选 CSV 的标签与 full manifest + locked split 完全一致。"""

    import csv
    from collections import Counter

    def read_rows(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
        last_error: UnicodeError | None = None
        for encoding in ("utf-8-sig", "gbk"):
            try:
                with path.open("r", encoding=encoding, newline="") as handle:
                    reader = csv.DictReader(handle)
                    if reader.fieldnames is None:
                        raise ValueError(f"CSV 缺少表头：{path}")
                    return tuple(reader.fieldnames), [dict(row) for row in reader]
            except UnicodeError as exc:
                last_error = exc
        raise ValueError(f"CSV 无法按兼容编码读取：{path}") from last_error

    by_group = split_payload.get("by_group")
    if not isinstance(by_group, Mapping):
        raise ValueError("锁定 split 缺少 by_group")
    manifest_fields = (
        "image_path",
        "group_id",
        "sample_id",
        "claimed_herb_name",
        "authenticity",
        "actual_species",
        "species_annotation_status",
        "batch_id",
        "origin",
        "capture_device",
        "collected_at",
        "annotator",
        "notes",
    )
    partition_fields = manifest_fields + (
        "sha256",
        "perceptual_hash",
        "manifest_encoding",
        "partition",
    )

    manifest_schema, manifest_rows = read_rows(manifest)
    selected_schema, selected_rows = read_rows(partition_manifest)
    if manifest_schema != manifest_fields or selected_schema != partition_fields:
        raise ValueError("manifest/partition CSV schema 不符合锁定格式")

    def key(row: Mapping[str, str]) -> tuple[str, ...]:
        return tuple((row.get(field) or "").strip() for field in manifest_fields)

    expected = Counter(
        key(row)
        for row in manifest_rows
        if by_group.get((row.get("group_id") or "").strip()) == expected_partition
    )
    if canonical_rows is not None:
        expected = Counter(
            (
                item.row.image_path.as_posix(),
                item.row.group_id,
                item.row.sample_id,
                item.row.claimed_herb_name,
                item.row.authenticity,
                "|".join(sorted(item.row.actual_species)),
                item.row.species_annotation_status,
                item.row.batch_id or "",
                item.row.origin or "",
                item.row.capture_device or "",
                item.row.collected_at or "",
                item.row.annotator or "",
                item.row.notes or "",
                item.sha256,
                item.perceptual_hash,
                item.manifest_encoding,
                item.partition,
            )
            for item in canonical_rows
            if item.partition == expected_partition
        )
    if any(
        (row.get("partition") or "").strip().lower() != expected_partition for row in selected_rows
    ):
        raise ValueError("partition manifest 包含错误分区")
    lowercase_hex = frozenset("0123456789abcdef")
    if any(
        len((row.get("sha256") or "").strip()) != 64
        or not set((row.get("sha256") or "").strip()) <= lowercase_hex
        or len((row.get("perceptual_hash") or "").strip()) != 16
        or not set((row.get("perceptual_hash") or "").strip()) <= lowercase_hex
        or (row.get("manifest_encoding") or "").strip().lower() != "utf-8-sig"
        for row in selected_rows
    ):
        raise ValueError("partition manifest 派生字段不符合锁定 schema")
    actual_fields = partition_fields if canonical_rows is not None else manifest_fields
    actual = Counter(
        tuple((row.get(field) or "").strip() for field in actual_fields) for row in selected_rows
    )
    if not expected or actual != expected:
        raise ValueError("partition manifest 与 full manifest/locked split 不一致")


def _validate_current_closed_set_data(
    *,
    manifest: Path,
    split: Path,
    data_root: Path,
    encoding: str | None,
    max_image_bytes: int,
    max_image_pixels: int,
) -> tuple[Any, ...]:
    """按 Task 1 的同一图片哈希和 input fingerprint 算法重验完整数据集。"""

    from herb_authentication.data import apply_split, validate_manifest_pair

    assignment = _load_split_assignment(split)
    with tempfile.TemporaryDirectory(prefix="herb-evaluation-manifest-") as temporary:
        bundle = validate_manifest_pair(
            manifest,
            unknown_manifest_path=None,
            data_root=data_root,
            report_path=Path(temporary) / "validation.json",
            encoding=encoding,
            max_image_bytes=max_image_bytes,
            max_image_pixels=max_image_pixels,
            perceptual_hash_distance=assignment.perceptual_hash_distance,
        )
    return apply_split(bundle.known, assignment)


def _validate_closed_set_fingerprints(
    *,
    metadata: Mapping[str, Any],
    manifest: Path,
    partition_manifest: Path,
    split: Path,
    config_fingerprint: str,
    source_candidate: Path,
    calibrated_checkpoint: Path,
    partition: str,
    data_root: Path,
    encoding: str | None,
    max_image_bytes: int,
    max_image_pixels: int,
) -> tuple[dict[str, str], Any]:
    """在图像 inference 前验证完整闭集工件指纹链。"""

    from herb_authentication.inference.closed_set import ClosedSetCalibration

    if partition not in {"validation", "locked-test"}:
        raise ValueError("partition 只允许 validation 或 locked-test")
    if not manifest.is_file() or not partition_manifest.is_file() or not split.is_file():
        raise ValueError("评估 manifest 或 split 工件不存在")
    fingerprints = metadata.get("fingerprints")
    if not isinstance(fingerprints, Mapping):
        raise ValueError("checkpoint 缺少 fingerprints")
    manifest_fingerprint = _sha256(manifest)
    if fingerprints.get("manifest") != manifest_fingerprint:
        raise ValueError("checkpoint manifest 指纹与当前工件不一致")
    split_fingerprint = _sha256(split)
    if (
        metadata.get("split_fingerprint") != split_fingerprint
        or metadata.get("split_artifact_fingerprint") != split_fingerprint
        or fingerprints.get("locked_split") != split_fingerprint
    ):
        raise ValueError("checkpoint split 指纹与当前工件不一致")
    try:
        split_payload = json.loads(split.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("锁定 split 无法读取") from exc
    if not isinstance(split_payload, Mapping):
        raise ValueError("锁定 split 必须是 JSON 对象")
    canonical_rows = _validate_current_closed_set_data(
        manifest=manifest,
        split=split,
        data_root=data_root,
        encoding=encoding,
        max_image_bytes=max_image_bytes,
        max_image_pixels=max_image_pixels,
    )
    _validate_partition_manifest_rows(
        manifest,
        partition_manifest,
        split_payload,
        "val" if partition == "validation" else "test",
        canonical_rows,
    )
    input_fingerprint = split_payload.get("input_fingerprint")
    if (
        not isinstance(input_fingerprint, str)
        or metadata.get("data_fingerprint") != input_fingerprint
        or metadata.get("split_input_fingerprint") != input_fingerprint
    ):
        raise ValueError("checkpoint manifest/split input 指纹链不一致")
    if metadata.get("training_config_fingerprint") != config_fingerprint:
        raise ValueError("checkpoint config 指纹与当前配置不一致")
    authenticity_classes = metadata.get("authenticity_classes")
    if tuple(authenticity_classes or ()) != ("authentic", "counterfeit"):
        raise ValueError("checkpoint 真伪类别顺序不兼容")
    if (
        metadata.get("experimental_closed_set") is not True
        or metadata.get("open_set_calibrated") is not False
    ):
        raise ValueError("checkpoint 不是实验闭集校准候选")
    if not source_candidate.is_file() or not calibrated_checkpoint.is_file():
        raise ValueError("源候选或闭集校准 checkpoint 不存在")
    source_fingerprint = _sha256(source_candidate)
    calibration = ClosedSetCalibration.from_dict(
        metadata.get("closed_set_calibration"),
        checkpoint_fingerprint=source_fingerprint,
    )
    recorded_validation = fingerprints.get("known_validation")
    if calibration.validation_fingerprint != recorded_validation:
        raise ValueError("calibration validation 指纹链不一致")
    if partition == "validation" and _sha256(partition_manifest) != recorded_validation:
        raise ValueError("validation manifest 指纹与 calibration 不一致")
    return (
        {
            "manifest": manifest_fingerprint,
            "split": split_fingerprint,
            "config": config_fingerprint,
            "source_candidate": source_fingerprint,
            "calibration": _sha256(calibrated_checkpoint),
        },
        calibration,
    )


def _execute_closed_set_partition(
    *,
    partition: str,
    data_dir: Path,
    run_dir: Path,
    fingerprint_bundle: Mapping[str, str],
    temperature: float,
    bootstrap_samples: int,
    seed: int,
    thresholds: Mapping[str, float],
    logits_collector: Callable[[Path], tuple[Any, Any]],
) -> tuple[dict[str, Any], Any]:
    """执行已经完成指纹验证的单一闭集分区。"""

    import math

    import torch
    from herb_authentication.evaluation import (
        bootstrap_authenticity_metrics,
        build_closed_set_evaluation_report,
        compute_authenticity_metrics,
        reserve_locked_test,
        write_evaluation_artifacts,
    )

    if partition not in {"validation", "locked-test"}:
        raise ValueError("partition 只允许 validation 或 locked-test")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, int | float)
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("closed-set temperature 必须是有限正数")
    manifest = data_dir.expanduser().resolve() / (
        "val.csv" if partition == "validation" else "test.csv"
    )
    if not manifest.is_file():
        raise ValueError(f"评估分区 CSV 不存在：{manifest}")
    resolved_run = run_dir.expanduser().resolve()
    if partition == "locked-test":
        reserve_locked_test(resolved_run, fingerprint_bundle)
    logits, targets = logits_collector(manifest)
    if (
        not isinstance(logits, torch.Tensor)
        or logits.ndim != 2
        or logits.shape[1] != 2
        or not bool(torch.isfinite(logits).all())
    ):
        raise ValueError("真伪 logits 必须是有限 [N, 2]")
    uncalibrated = compute_authenticity_metrics(torch.softmax(logits, dim=1), targets)
    calibrated = compute_authenticity_metrics(
        torch.softmax(logits / float(temperature), dim=1), targets
    )
    intervals = bootstrap_authenticity_metrics(
        torch.softmax(logits / float(temperature), dim=1),
        targets,
        samples=bootstrap_samples,
        seed=seed,
    )
    report = build_closed_set_evaluation_report(
        run_id=resolved_run.name,
        partition=partition,
        uncalibrated=uncalibrated,
        calibrated=calibrated,
        bootstrap=intervals,
        fingerprints=fingerprint_bundle,
        thresholds=thresholds,
    )
    output_dir = resolved_run / ("validation" if partition == "validation" else "locked_test")
    artifacts = write_evaluation_artifacts(report, output_dir)
    return report, artifacts


def _evaluate_available_data(
    config: Any, checkpoint: Path, locked_split: Path
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    from herb_authentication.data import (
        HerbImageDataset,
        apply_split,
        build_label_space,
        create_herb_dataloader,
        validate_manifest_pair,
    )
    from herb_authentication.evaluation import (
        UnknownPrediction,
        collect_known_predictions,
        summarize_known_predictions,
        summarize_unknown_predictions,
    )
    from herb_authentication.inference.service import InferenceService

    bundle = validate_manifest_pair(
        config.data.manifest_path,
        unknown_manifest_path=config.data.unknown_manifest_path,
        data_root=config.data.data_root,
        report_path=config.evaluation.report_dir / "validation_report.json",
        encoding=config.data.text_encoding,
        max_image_bytes=config.data.max_image_bytes,
        max_image_pixels=config.data.max_image_pixels,
        perceptual_hash_distance=config.data.perceptual_hash_distance,
    )
    assignment = _require_configured_split_version(
        config, _load_split_assignment(locked_split), locked_split
    )
    partitioned = apply_split(bundle.known, assignment)
    labels = build_label_space(partitioned)
    test_rows = tuple(item for item in partitioned if item.partition == "test")
    if not test_rows:
        raise ValueError("锁定 known test 分区为空")
    service = InferenceService.from_checkpoint(
        checkpoint,
        device=config.inference.device,
        max_image_bytes=config.inference.max_image_bytes,
        max_pixels=config.inference.max_image_pixels,
    )
    _validate_label_contract(service, labels)
    dataset = HerbImageDataset(test_rows, labels, service.preprocess)
    loader = create_herb_dataloader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        seed=config.data.seed,
        num_workers=config.data.num_workers,
    )
    temperatures = (
        service.calibration.temperatures
        if service.calibration.is_calibrated
        else {"claimed_herb": 1.0, "actual_species": 1.0, "authenticity": 1.0}
    )
    predictions = collect_known_predictions(
        service.model,
        loader,
        device=service.device,
        temperatures=temperatures,
    )
    thresholds = service.calibration.thresholds
    known_metrics = summarize_known_predictions(
        predictions,
        min_species_supervision_coverage=config.training.min_species_supervision_coverage,
        species_threshold=0.5 if thresholds is None else thresholds.actual_species,
        claimed_threshold=0.5 if thresholds is None else thresholds.claimed_herb,
        authenticity_threshold=0.5 if thresholds is None else thresholds.authenticity,
        bootstrap_samples=config.evaluation.bootstrap_samples,
        seed=config.data.seed,
    )
    unknown_rows = tuple(item for item in bundle.unknown if item.row.dataset_role == "test")
    if not unknown_rows:
        return known_metrics, {"status": "not_available", "reason": "没有 unknown test 数据"}
    unknown_predictions = tuple(
        UnknownPrediction(
            unknown_type=item.row.unknown_type,
            sample_id=item.row.sample_id,
            group_id=item.row.group_id,
            rejected=service.predict_image(item.row.image_path).rejected,
        )
        for item in unknown_rows
    )
    return known_metrics, summarize_unknown_predictions(unknown_predictions)


def _validate_label_contract(service: Any, labels: Any) -> None:
    if service.claimed_herb_classes != labels.claimed_classes:
        raise ValueError("checkpoint 声称药材词表与锁定数据不兼容")
    if service.actual_species_classes != labels.actual_species_classes:
        raise ValueError("checkpoint 实际物种词表与锁定数据不兼容")


def _load_split_assignment(path: Path) -> Any:
    from herb_authentication.data import split_assignment_from_payload
    from herb_authentication.errors import ManifestValidationError

    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"锁定 split 文件无法读取：{path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("锁定 split 必须是 schema v1 JSON 对象")
    try:
        return split_assignment_from_payload(value)
    except ManifestValidationError as exc:
        raise ValueError("锁定 split 字段不完整或类型错误") from exc


def _configured_split_version(config: Any) -> str:
    return str(getattr(config.data, "split_algorithm_version", "group-stratified-v1"))


def _require_configured_split_version(config: Any, assignment: Any, path: Path) -> Any:
    configured = _configured_split_version(config)
    if assignment.algorithm_version != configured:
        raise ValueError(
            "切分算法版本与配置不一致："
            f"config={configured!r}, splits.json={assignment.algorithm_version!r}, path={path}"
        )
    return assignment


def _preflight_split_assignment(config: Any, path: Path) -> Any | None:
    configured = _configured_split_version(config)
    if path.is_file():
        return _require_configured_split_version(config, _load_split_assignment(path), path)
    if configured == "group-stratified-near-duplicate-components-v2":
        raise ValueError(f"V2 配置缺少锁定 split：{path}；请先运行 prepare_closed_set.py。")
    return None


def _checkpoint_metadata(path: Path) -> Mapping[str, Any]:
    import torch
    from herb_authentication.errors import CheckpointCompatibilityError

    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except (EOFError, OSError, RuntimeError, TypeError, ValueError, pickle.UnpicklingError) as exc:
        raise CheckpointCompatibilityError(f"无法安全加载 checkpoint：{path}") from exc
    if not isinstance(state, Mapping) or state.get("schema_version") != 1:
        raise CheckpointCompatibilityError("评估只接受 schema v1 checkpoint")
    metadata = state.get("metadata")
    if not isinstance(metadata, Mapping):
        raise CheckpointCompatibilityError("checkpoint 缺少 metadata")
    return metadata


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _locked_split_fingerprint(path: Path) -> str | None:
    if not path.is_file():
        return None
    return _sha256(path)


def _checkpoint_split_fingerprint(metadata: Mapping[str, Any]) -> str | None:
    direct = _optional_string(metadata.get("split_artifact_fingerprint"))
    return direct or _nested_string(metadata, "fingerprints", "locked_split")


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _nested_string(mapping: Mapping[str, Any], parent: str, child: str) -> str | None:
    value = mapping.get(parent)
    if not isinstance(value, Mapping):
        return None
    return _optional_string(value.get(child))


if __name__ == "__main__":
    raise SystemExit(main())
