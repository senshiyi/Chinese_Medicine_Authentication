"""生成骨碎补闭集 manifest，并写出确定性的锁定切分。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

from herb_authentication.config import load_config
from herb_authentication.data import (
    V2_ALGORITHM_VERSION,
    ManifestSource,
    apply_split,
    build_closed_set_manifest,
    create_group_splits,
    split_assignment_to_payload,
    validate_manifest_pair,
    write_split_artifacts,
)
from herb_authentication.environment import configure_utf8_stdio
from herb_authentication.errors import HerbAuthenticationError, ManifestValidationError

EXPECTED = {
    "all": {"authentic": 335, "counterfeit": 92, "total": 427},
    "train": {"authentic": 235, "counterfeit": 64, "total": 299},
    "val": {"authentic": 50, "counterfeit": 14, "total": 64},
    "test": {"authentic": 50, "counterfeit": 14, "total": 64},
}


LOCKED_RATIOS = (0.70, 0.15, 0.15)
LOCKED_SEED = 42


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="准备骨碎补正品/伪品闭集数据与锁定切分")
    parser.add_argument("--config", type=Path, required=True, help="实验 YAML 配置路径")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="在临时目录完成全部校验，不修改正式产物",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_utf8_stdio()
    args = build_parser().parse_args(argv)
    try:
        report = _prepare(args.config, check_only=args.check_only)
    except (HerbAuthenticationError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"闭集数据准备失败：{exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False))
    return 0


def _prepare(config_path: Path, *, check_only: bool = False) -> dict[str, object]:
    config = load_config(config_path)
    if config.data.unknown_manifest_path is not None:
        raise ManifestValidationError("闭集数据准备要求 data.unknown_manifest_path=null。")
    _validate_locked_split_config(config.data)

    if check_only:
        with TemporaryDirectory(prefix="herb-closed-set-check-") as directory:
            temporary_manifest = Path(directory) / config.data.manifest_path.name
            report = _prepare_outputs(config, temporary_manifest)
        report["output_paths"] = _output_paths(config.data.manifest_path)
        report["check_only"] = True
        report["artifacts_written"] = False
        return report

    destination_manifest = config.data.manifest_path
    destination_dir = destination_manifest.parent
    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix="herb-closed-set-stage-", dir=destination_dir.parent
    ) as directory:
        staged_manifest = Path(directory) / destination_manifest.name
        report = _prepare_outputs(config, staged_manifest)
        _publish_artifacts(
            Path(directory), destination_dir, manifest_name=destination_manifest.name
        )
    report["output_paths"] = _output_paths(destination_manifest)
    report["check_only"] = False
    report["artifacts_written"] = True
    return report


def _v2_target_counts() -> dict[str, dict[str, int]]:
    return {
        partition: {label: EXPECTED[partition][label] for label in ("authentic", "counterfeit")}
        for partition in ("train", "val", "test")
    }


def _prepare_outputs(config, manifest_path: Path) -> dict[str, object]:
    output_dir = manifest_path.parent
    manifest = build_closed_set_manifest(
        sources=_sources(config.data.data_root),
        data_root=config.data.data_root,
        output_path=manifest_path,
        claimed_herb_name="Drynaria_fortunei",
    )
    validation_report = output_dir / "manifest_validation.json"
    bundle = validate_manifest_pair(
        manifest.path,
        unknown_manifest_path=None,
        data_root=config.data.data_root,
        report_path=validation_report,
        encoding=config.data.text_encoding,
        max_image_bytes=config.data.max_image_bytes,
        max_image_pixels=config.data.max_image_pixels,
        perceptual_hash_distance=config.data.perceptual_hash_distance,
    )
    target_counts = (
        _v2_target_counts() if config.data.split_algorithm_version == V2_ALGORITHM_VERSION else None
    )
    assignment = create_group_splits(
        bundle.known,
        ratios=(config.data.train_ratio, config.data.val_ratio, config.data.test_ratio),
        seed=config.data.seed,
        perceptual_hash_distance=config.data.perceptual_hash_distance,
        algorithm_version=config.data.split_algorithm_version,
        target_counts=target_counts,
    )
    counts = _counts(bundle.known, assignment)
    if counts != EXPECTED:
        raise ManifestValidationError(f"闭集数据计数不符合锁定要求：实际 {counts}，预期 {EXPECTED}")

    artifacts = write_split_artifacts(bundle.known, assignment, output_dir)
    return {
        "schema_version": 1,
        "output_paths": _output_paths(manifest.path),
        "counts": counts,
        "manifest_sha256": manifest.sha256,
        "split_input_fingerprint": assignment.input_fingerprint,
        "split_file_sha256": _sha256_file(artifacts.splits_json),
        "split_audit": split_assignment_to_payload(assignment),
    }


def _publish_artifacts(
    staging_dir: Path,
    destination_dir: Path,
    *,
    manifest_name: str,
) -> None:
    names = (
        manifest_name,
        "manifest_validation.json",
        "train.csv",
        "val.csv",
        "test.csv",
        "splits.json",
    )
    if Path(manifest_name).name != manifest_name or len(set(names)) != len(names):
        raise ValueError("manifest 文件名与闭集产物名称冲突")
    for name in names:
        if not (staging_dir / name).is_file():
            raise OSError(f"暂存闭集产物缺失：{name}")

    destination_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix="herb-closed-set-backup-", dir=destination_dir.parent
    ) as backup_value:
        backup_dir = Path(backup_value)
        installed: list[str] = []
        backed_up: list[str] = []
        try:
            for name in names:
                source = staging_dir / name
                target = destination_dir / name
                backup = backup_dir / name
                if target.exists():
                    os.replace(target, backup)
                    backed_up.append(name)
                os.replace(source, target)
                installed.append(name)
        except BaseException:
            for name in reversed(installed):
                target = destination_dir / name
                if target.exists():
                    target.unlink()
            for name in reversed(backed_up):
                backup = backup_dir / name
                if backup.exists():
                    os.replace(backup, destination_dir / name)
            raise


def _validate_locked_split_config(data) -> None:
    if data.seed != LOCKED_SEED:
        raise ManifestValidationError("闭集锁定切分要求 seed=42。")
    ratios = (data.train_ratio, data.val_ratio, data.test_ratio)
    if ratios != LOCKED_RATIOS:
        raise ManifestValidationError("闭集锁定切分比例必须为 0.70/0.15/0.15。")


def _output_paths(manifest_path: Path) -> dict[str, str]:
    output_dir = manifest_path.parent.resolve()
    return {
        "manifest": str(manifest_path.resolve()),
        "validation_report": str(output_dir / "manifest_validation.json"),
        "train_csv": str(output_dir / "train.csv"),
        "val_csv": str(output_dir / "val.csv"),
        "test_csv": str(output_dir / "test.csv"),
        "splits_json": str(output_dir / "splits.json"),
    }


def _sources(data_root: Path) -> tuple[ManifestSource, ManifestSource]:
    raw = data_root / "raw"
    return (
        ManifestSource(
            raw / "Drynaria_fortunei_authentic",
            re.compile(r"Drynaria-fortunei_AUTHENTIC_(\d{4})\.jpg"),
            "df-auth",
            "authentic",
            "Drynaria_fortunei",
            "verified",
        ),
        ManifestSource(
            raw / "Drynaria_fortunei_fake",
            re.compile(r"Drynaria-fortunei_FAKE_(\d{4})\.jpg"),
            "df-fake",
            "counterfeit",
            None,
            "unverified",
        ),
    )


def _counts(rows, assignment) -> dict[str, dict[str, int]]:
    result = {
        name: {"authentic": 0, "counterfeit": 0, "total": 0}
        for name in ("all", "train", "val", "test")
    }
    for item in apply_split(rows, assignment):
        authenticity = item.row.authenticity
        for partition in ("all", item.partition):
            result[partition][authenticity] += 1
            result[partition]["total"] += 1
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
