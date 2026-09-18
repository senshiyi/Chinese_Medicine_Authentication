"""使用 locked validation 与 unknown calibration 生成新校准 checkpoint。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
LEGACY_MODEL_PATHS = (
    PROJECT_ROOT / "项目" / "Models" / "best_model.pth",
    PROJECT_ROOT / "项目" / "Models" / "epoch_001.pth",
)

CalibrationDataLoader = Callable[..., tuple[Any, Any]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="校准中药三任务模型并生成新的 checkpoint")
    parser.add_argument(
        "--mode",
        choices=("open-set", "closed-set-temperature"),
        default="open-set",
        help="默认保持原开放集校准；实验闭集模式只拟合真伪温度",
    )
    parser.add_argument("--checkpoint", required=True, type=Path, help="未校准 checkpoint")
    parser.add_argument("--known-val", required=True, type=Path, help="锁定的 known validation CSV")
    parser.add_argument("--unknown", type=Path, help="open-set 模式必需的 unknown calibration CSV")
    parser.add_argument("--output", type=Path, help="新校准 checkpoint；默认追加 .calibrated")
    parser.add_argument("--data-root", type=Path, help="图片根目录；默认使用各 CSV 所在目录")
    parser.add_argument("--encoding", help="CSV 编码；默认依次尝试 UTF-8-SIG 与 GBK")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--max-unknown-accept-rate", type=float, default=0.05)
    parser.add_argument("--min-known-selective-accuracy", type=float, default=0.95)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    calibration_data_loader: CalibrationDataLoader | None = None,
) -> int:
    _configure_utf8_stdio()
    args = build_parser().parse_args(argv)
    checkpoint = args.checkpoint.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else checkpoint.with_name(f"{checkpoint.stem}.calibrated{checkpoint.suffix}")
    )
    try:
        _validate_probability(args.max_unknown_accept_rate, "--max-unknown-accept-rate")
        _validate_probability(args.min_known_selective_accuracy, "--min-known-selective-accuracy")
        _validate_output_path(checkpoint, output)
        known_path = args.known_val.expanduser().resolve()
        known_rows = _read_csv(known_path, args.encoding)
        _require_locked_known_validation(known_rows)

        if str(SCRIPT_DIR) not in sys.path:
            sys.path.insert(0, str(SCRIPT_DIR))
        if args.mode == "closed-set-temperature":
            if args.unknown is not None:
                raise ValueError("closed-set-temperature 模式禁止使用 --unknown")
            written, calibration = calibrate_closed_set_checkpoint(
                checkpoint=checkpoint,
                known_manifest=known_path,
                known_rows=known_rows,
                output=output,
                data_root=(
                    None if args.data_root is None else args.data_root.expanduser().resolve()
                ),
                device=args.device,
                calibration_data_loader=calibration_data_loader,
            )
            print(
                json.dumps(
                    {
                        "output": str(written),
                        "closed_set_calibration": calibration.to_dict(),
                        "experimental_closed_set": True,
                        "open_set_calibrated": False,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0

        if args.unknown is None:
            raise ValueError("open-set 模式必须提供 --unknown")
        unknown_path = args.unknown.expanduser().resolve()
        unknown_rows = _require_unknown_calibration(_read_csv(unknown_path, args.encoding))
        calibrate_checkpoint(
            checkpoint=checkpoint,
            known_manifest=known_path,
            unknown_manifest=unknown_path,
            known_rows=known_rows,
            unknown_rows=unknown_rows,
            output=output,
            data_root=None if args.data_root is None else args.data_root.expanduser().resolve(),
            device=args.device,
            max_unknown_accept_rate=args.max_unknown_accept_rate,
            min_known_selective_accuracy=args.min_known_selective_accuracy,
            calibration_data_loader=calibration_data_loader,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"校准失败：{exc}", file=sys.stderr)
        return 2
    print(f"校准完成：{output}")
    return 0


def calibrate_closed_set_checkpoint(
    *,
    checkpoint: Path,
    known_manifest: Path,
    known_rows: list[dict[str, str]],
    output: Path,
    data_root: Path | None,
    device: str,
    calibration_data_loader: CalibrationDataLoader | None = None,
) -> tuple[Path, Any]:
    """拟合实验闭集真伪温度并原子写入独立 metadata 块。"""

    import torch
    from herb_authentication.inference.closed_set import fit_closed_set_temperature
    from herb_authentication.training.checkpoints import save_checkpoint_atomic

    state = _load_checkpoint_state(checkpoint, torch)
    validation_fingerprint = _validate_known_validation_fingerprint(state, known_manifest)
    checkpoint_fingerprint = _sha256_file(checkpoint)
    loader = calibration_data_loader or _collect_closed_set_authenticity_data
    logits, targets = loader(
        checkpoint=checkpoint,
        known_rows=known_rows,
        known_manifest=known_manifest,
        data_root=data_root,
        device=device,
    )
    calibration = fit_closed_set_temperature(
        logits,
        targets,
        validation_fingerprint=validation_fingerprint,
        checkpoint_fingerprint=checkpoint_fingerprint,
        algorithm_version="closed-set-temperature-v1",
    )

    updated = dict(state)
    metadata = dict(updated["metadata"])
    metadata["closed_set_calibration"] = calibration.to_dict()
    metadata["experimental_closed_set"] = True
    metadata["open_set_calibrated"] = False
    updated["metadata"] = metadata
    written = save_checkpoint_atomic(
        output,
        updated,
        protected_paths=(checkpoint, *LEGACY_MODEL_PATHS),
    )
    return written, calibration


def calibrate_checkpoint(
    *,
    checkpoint: Path,
    known_manifest: Path,
    unknown_manifest: Path,
    known_rows: list[dict[str, str]],
    unknown_rows: list[dict[str, str]],
    output: Path,
    data_root: Path | None,
    device: str,
    max_unknown_accept_rate: float,
    min_known_selective_accuracy: float,
    calibration_data_loader: CalibrationDataLoader | None = None,
) -> Path:
    """拟合校准参数，并通过现有 checkpoint 原子写接口发布新文件。"""

    import torch
    from herb_authentication.inference.calibration import fit_calibrators
    from herb_authentication.training.checkpoints import save_checkpoint_atomic

    state = _load_checkpoint_state(checkpoint, torch)
    _validate_manifest_fingerprints(state, known_manifest, unknown_manifest)
    loader = calibration_data_loader or _collect_calibration_data
    known_data, unknown_data = loader(
        checkpoint=checkpoint,
        known_rows=known_rows,
        unknown_rows=unknown_rows,
        known_manifest=known_manifest,
        unknown_manifest=unknown_manifest,
        data_root=data_root,
        device=device,
    )
    bundle = fit_calibrators(
        known_data,
        unknown_data,
        max_unknown_accept_rate=max_unknown_accept_rate,
        min_known_selective_accuracy=min_known_selective_accuracy,
    )
    if not bundle.is_calibrated:
        raise ValueError("没有满足安全约束的可行阈值组合，未生成校准 checkpoint")

    updated = dict(state)
    metadata = dict(updated["metadata"])
    calibration = bundle.to_dict()
    calibration["constraints"] = {
        "max_unknown_accept_rate": max_unknown_accept_rate,
        "min_known_selective_accuracy": min_known_selective_accuracy,
    }
    metadata["calibration"] = calibration
    fingerprints = dict(metadata["fingerprints"])
    fingerprints.update(
        {
            "known_validation": _sha256_file(known_manifest),
            "unknown_calibration": _sha256_file(unknown_manifest),
        }
    )
    metadata["fingerprints"] = fingerprints
    updated["metadata"] = metadata
    return save_checkpoint_atomic(
        output,
        updated,
        protected_paths=(checkpoint, *LEGACY_MODEL_PATHS),
    )


def _collect_closed_set_authenticity_data(
    *,
    checkpoint: Path,
    known_rows: list[dict[str, str]],
    known_manifest: Path,
    data_root: Path | None,
    device: str,
) -> tuple[Any, Any]:
    """只采集锁定 validation 的真伪 logits 与 targets。"""

    import torch
    from herb_authentication.inference.service import InferenceService

    service = InferenceService.from_checkpoint(checkpoint, device=device)
    authenticity_index = {"authentic": 0, "counterfeit": 1}
    logits: list[torch.Tensor] = []
    targets: list[int] = []
    for line, row in enumerate(known_rows, start=2):
        label = _required(row, "authenticity", known_manifest, line)
        if label not in authenticity_index:
            raise ValueError(f"known validation 第 {line} 行 authenticity 不合法")
        image = _image_path(row, known_manifest, data_root, line)
        logits.append(_predict_authenticity_logits(service, image, torch))
        targets.append(authenticity_index[label])
    return torch.stack(logits), torch.tensor(targets, dtype=torch.long)


def _collect_calibration_data(
    *,
    checkpoint: Path,
    known_rows: list[dict[str, str]],
    unknown_rows: list[dict[str, str]],
    known_manifest: Path,
    unknown_manifest: Path,
    data_root: Path | None,
    device: str,
) -> tuple[Any, Any]:
    """从 checkpoint 装配模型，对两个锁定 CSV 的图片采集未校准 logits。"""

    import torch
    from herb_authentication.inference.calibration import (
        KnownValidationData,
        UnknownCalibrationData,
    )
    from herb_authentication.inference.service import InferenceService

    service = InferenceService.from_checkpoint(checkpoint, device=device)
    claimed_index = {label: index for index, label in enumerate(service.claimed_herb_classes)}
    species_index = {label: index for index, label in enumerate(service.actual_species_classes)}
    authenticity_index = {"authentic": 0, "counterfeit": 1}
    known_logits: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    claimed_targets: list[int] = []
    species_targets: list[torch.Tensor] = []
    species_mask: list[bool] = []
    authenticity_targets: list[int] = []

    for line, row in enumerate(known_rows, start=2):
        claimed = _required(row, "claimed_herb_name", known_manifest, line)
        authenticity = _required(row, "authenticity", known_manifest, line)
        if claimed not in claimed_index:
            raise ValueError(
                f"known validation 第 {line} 行声称药材不在 checkpoint 词表：{claimed}"
            )
        if authenticity not in authenticity_index:
            raise ValueError(f"known validation 第 {line} 行 authenticity 不合法")
        status = row.get("species_annotation_status", "unverified").strip()
        verified = status == "verified"
        labels = tuple(
            item.strip() for item in row.get("actual_species", "").split("|") if item.strip()
        )
        if verified and not labels:
            raise ValueError(f"known validation 第 {line} 行 verified 实际物种不能为空")
        unknown_labels = sorted(set(labels).difference(species_index))
        if unknown_labels:
            raise ValueError(
                f"known validation 第 {line} 行实际物种不在 checkpoint 词表：{unknown_labels}"
            )
        image = _image_path(row, known_manifest, data_root, line)
        known_logits.append(_predict_logits(service, image, torch))
        target = torch.zeros(len(species_index), dtype=torch.float32)
        for label in labels:
            target[species_index[label]] = 1.0
        claimed_targets.append(claimed_index[claimed])
        species_targets.append(target)
        species_mask.append(verified)
        authenticity_targets.append(authenticity_index[authenticity])

    unknown_logits = [
        _predict_logits(
            service,
            _image_path(row, unknown_manifest, data_root, line),
            torch,
        )
        for line, row in enumerate(unknown_rows, start=2)
    ]
    return (
        KnownValidationData(
            claimed_logits=torch.stack([item[0] for item in known_logits]),
            claimed_targets=torch.tensor(claimed_targets, dtype=torch.long),
            species_logits=torch.stack([item[1] for item in known_logits]),
            species_targets=torch.stack(species_targets),
            species_mask=torch.tensor(species_mask, dtype=torch.bool),
            authenticity_logits=torch.stack([item[2] for item in known_logits]),
            authenticity_targets=torch.tensor(authenticity_targets, dtype=torch.long),
        ),
        UnknownCalibrationData(
            claimed_logits=torch.stack([item[0] for item in unknown_logits]),
            species_logits=torch.stack([item[1] for item in unknown_logits]),
            authenticity_logits=torch.stack([item[2] for item in unknown_logits]),
        ),
    )


def _predict_authenticity_logits(service: Any, image: Path, torch: Any) -> Any:
    output = service.predict_logits(image)
    value = getattr(output, "authenticity", None)
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != (1, 2)
        or not bool(torch.isfinite(value).all())
    ):
        raise RuntimeError("公共 logits 接口返回的真伪输出不兼容")
    return value[0].detach().cpu()


def _predict_logits(service: Any, image: Path, torch: Any) -> tuple[Any, Any, Any]:
    output = service.predict_logits(image)
    values = tuple(
        getattr(output, name, None) for name in ("claimed_herb", "actual_species", "authenticity")
    )
    if any(
        not isinstance(value, torch.Tensor)
        or value.ndim != 2
        or value.shape[0] != 1
        or not bool(torch.isfinite(value).all())
        for value in values
    ):
        raise RuntimeError("公共 logits 接口返回的三头输出不兼容")
    return tuple(value[0].detach().cpu() for value in values)


def _load_checkpoint_state(checkpoint: Path, torch: Any) -> dict[str, Any]:
    if not checkpoint.is_file():
        raise ValueError(f"checkpoint 不存在或不是文件：{checkpoint}")
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except (EOFError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"无法安全加载 checkpoint：{checkpoint}") from exc
    if not isinstance(state, Mapping) or state.get("schema_version") != 1:
        raise ValueError("checkpoint schema_version 不是受支持的 v1")
    if not isinstance(state.get("metadata"), Mapping):
        raise ValueError("checkpoint 缺少 metadata")
    return dict(state)


def _validate_known_validation_fingerprint(state: Mapping[str, Any], known_manifest: Path) -> str:
    metadata = state["metadata"]
    fingerprints = metadata.get("fingerprints", {})
    if not isinstance(fingerprints, Mapping):
        raise ValueError("checkpoint fingerprints 必须是映射")
    recorded = fingerprints.get("known_validation")
    if not isinstance(recorded, str) or not recorded:
        raise ValueError("checkpoint 缺少必需数据指纹：known_validation")
    actual = _sha256_file(known_manifest)
    if recorded != actual:
        raise ValueError(f"known_validation 数据指纹与 checkpoint 不一致：{known_manifest}")
    return actual


def _validate_manifest_fingerprints(
    state: Mapping[str, Any], known_manifest: Path, unknown_manifest: Path
) -> None:
    metadata = state["metadata"]
    fingerprints = metadata.get("fingerprints", {})
    if not isinstance(fingerprints, Mapping):
        raise ValueError("checkpoint fingerprints 必须是映射")
    expected = {
        "known_validation": (known_manifest, _sha256_file(known_manifest)),
        "unknown_calibration": (unknown_manifest, _sha256_file(unknown_manifest)),
    }
    for key, (path, actual) in expected.items():
        recorded = fingerprints.get(key)
        if not isinstance(recorded, str) or not recorded:
            raise ValueError(f"checkpoint 缺少必需数据指纹：{key}")
        if recorded != actual:
            raise ValueError(f"{key} 数据指纹与 checkpoint 不一致：{path}")


def _read_csv(path: Path, encoding: str | None) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"CSV 不存在或不是文件：{path}")
    encodings = (encoding,) if encoding else ("utf-8-sig", "gbk")
    last_error: UnicodeError | None = None
    for candidate in encodings:
        try:
            with path.open("r", encoding=candidate, newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    raise ValueError(f"CSV 缺少表头：{path}")
                rows = [dict(row) for row in reader]
            if not rows:
                raise ValueError(f"CSV 没有数据行：{path}")
            return rows
        except UnicodeError as exc:
            last_error = exc
    raise ValueError(f"CSV 无法按指定编码读取：{path}") from last_error


def _require_locked_known_validation(rows: list[dict[str, str]]) -> None:
    for line, row in enumerate(rows, start=2):
        role = (
            (row.get("partition") or row.get("dataset_role") or row.get("split") or "")
            .strip()
            .lower()
        )
        if role == "test":
            raise ValueError("test 数据禁止用于校准")
        if role not in {"val", "validation", "known_validation"}:
            raise ValueError(f"known validation 第 {line} 行缺少锁定的 validation 角色")


def _require_unknown_calibration(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for row in rows:
        role = row.get("dataset_role", "").strip().lower()
        if role not in {"calibration", "test"}:
            raise ValueError("unknown 数据只允许 dataset_role=calibration")
        if role == "calibration":
            selected.append(row)
    if not selected:
        raise ValueError("test 数据禁止用于校准；unknown calibration 分区为空")
    return selected


def _validate_output_path(checkpoint: Path, output: Path) -> None:
    normalized = os.path.normcase(str(output))
    if normalized == os.path.normcase(str(checkpoint)):
        raise ValueError("不允许覆盖输入 checkpoint")
    if normalized in {os.path.normcase(str(path.resolve())) for path in LEGACY_MODEL_PATHS}:
        raise ValueError("不允许覆盖旧模型")
    if output.exists():
        raise ValueError(f"输出 checkpoint 已存在，请指定新路径：{output}")


def _image_path(row: Mapping[str, str], manifest: Path, data_root: Path | None, line: int) -> Path:
    raw = _required(row, "image_path", manifest, line)
    root = (data_root or manifest.parent).resolve()
    candidate = Path(raw).expanduser()
    resolved = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"第 {line} 行 image_path 越过 data_root：{raw}")
    return resolved


def _required(row: Mapping[str, str], field: str, manifest: Path, line: int) -> str:
    value = row.get(field, "").strip()
    if not value:
        raise ValueError(f"{manifest.name} 第 {line} 行 {field} 不能为空")
    return value


def _validate_probability(value: float, option: str) -> None:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{option} 必须位于 [0, 1]")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="strict")


if __name__ == "__main__":
    sys.exit(main())
