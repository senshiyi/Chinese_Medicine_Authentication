"""严格、类型化且路径解析幂等的项目配置。"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml

from .encoding import normalize_encoding, read_text
from .errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class DataConfig:
    manifest_path: Path
    unknown_manifest_path: Path | None
    data_root: Path
    provenance: str = "unknown"
    text_encoding: str | None = None
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 0
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    seed: int = 42
    max_image_bytes: int = 20 * 1024 * 1024
    max_image_pixels: int = 40_000_000
    perceptual_hash_distance: int = 4
    split_algorithm_version: str = "group-stratified-v1"
    use_weighted_sampler: bool = True


@dataclass(frozen=True, slots=True)
class ModelConfig:
    backbone: str = "efficientnet_b3"
    pretrained: bool = True
    freeze_backbone: bool = True
    unfreeze_epoch: int = 5
    dropout: float = 0.3


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int = 50
    learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    claimed_herb_weight: float = 1.0
    actual_species_weight: float = 1.0
    authenticity_weight: float = 1.0
    label_smoothing: float = 0.1
    early_stop_patience: int = 10
    mixed_precision: bool = True
    gradient_clip: float = 1.0
    min_species_supervision_coverage: float = 0.2
    use_authenticity_class_weights: bool = True
    checkpoint_path: Path = Path("项目/Models/best_multitask_model.pth")
    log_dir: Path = Path("项目/Results/logs")


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    mode: Literal["open_set", "closed_set_temperature"] = "open_set"
    algorithm_version: str = "closed-set-temperature-v1"
    max_unknown_accept_rate: float = 0.05
    min_known_selective_accuracy: float = 0.95
    output_checkpoint_path: Path = Path("项目/Models/best_multitask_model.calibrated.pth")


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    report_dir: Path = Path("项目/Results/evaluation")
    bootstrap_samples: int = 1000


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    checkpoint_path: Path = Path("项目/Models/best_multitask_model.pth")
    device: str = "auto"
    allow_uncalibrated: bool = False
    max_image_bytes: int = 20 * 1024 * 1024
    max_image_pixels: int = 40_000_000


@dataclass(frozen=True, slots=True)
class RunConfig:
    root_dir: Path = Path("项目/Results/experiments/drynaria_closed_set/runs")


@dataclass(frozen=True, slots=True)
class PromotionConfig:
    min_macro_f1: float = 0.80
    min_balanced_accuracy: float = 0.80
    min_counterfeit_recall: float = 0.80
    target_checkpoint_path: Path = Path("项目/Models/best_multitask_model.calibrated.pth")


@dataclass(frozen=True, slots=True)
class AppConfig:
    project_root: Path
    source_path: Path
    source_encoding: str
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    calibration: CalibrationConfig
    evaluation: EvaluationConfig
    inference: InferenceConfig
    run: RunConfig
    promotion: PromotionConfig


ConfigSection = TypeVar("ConfigSection")


def load_config(path: str | Path, encoding: str | None = None) -> AppConfig:
    """加载 YAML，拒绝未知键并一次性解析所有路径。"""

    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise ConfigurationError(f"配置文件不存在：{source_path}")

    try:
        decoded = read_text(source_path, encoding=encoding)
        raw = yaml.safe_load(decoded.text)
    except (OSError, UnicodeError, yaml.YAMLError, ValueError) as exc:
        raise ConfigurationError(f"无法读取配置文件 {source_path}：{exc}") from exc

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigurationError("配置根节点必须是映射对象。")

    allowed_root_keys = {
        "project_root",
        "data",
        "model",
        "training",
        "calibration",
        "evaluation",
        "inference",
        "run",
        "promotion",
    }
    _reject_unknown_keys(raw, allowed_root_keys, "")

    project_root = _resolve_project_root(source_path, raw.get("project_root", "."))
    data_raw = _mapping(raw.get("data"), "data")
    model_raw = _mapping(raw.get("model"), "model")
    training_raw = _mapping(raw.get("training"), "training")
    calibration_raw = _mapping(raw.get("calibration"), "calibration")
    evaluation_raw = _mapping(raw.get("evaluation"), "evaluation")
    inference_raw = _mapping(raw.get("inference"), "inference")
    run_raw = _mapping(raw.get("run"), "run")
    promotion_raw = _mapping(raw.get("promotion"), "promotion")

    _reject_dataclass_unknown_keys(data_raw, DataConfig, "data")
    _reject_dataclass_unknown_keys(model_raw, ModelConfig, "model")
    _reject_dataclass_unknown_keys(training_raw, TrainingConfig, "training")
    _reject_dataclass_unknown_keys(calibration_raw, CalibrationConfig, "calibration")
    _reject_dataclass_unknown_keys(evaluation_raw, EvaluationConfig, "evaluation")
    _reject_dataclass_unknown_keys(inference_raw, InferenceConfig, "inference")
    _reject_dataclass_unknown_keys(run_raw, RunConfig, "run")
    _reject_dataclass_unknown_keys(promotion_raw, PromotionConfig, "promotion")

    data = _build_data_config(data_raw, project_root)
    model = _construct(ModelConfig, model_raw, "model")
    training = _build_training_config(training_raw, project_root)
    calibration = _build_calibration_config(calibration_raw, project_root)
    evaluation = _build_evaluation_config(evaluation_raw, project_root)
    inference = _build_inference_config(inference_raw, project_root)
    run = _build_run_config(run_raw, project_root)
    promotion = _build_promotion_config(promotion_raw, project_root)

    _validate(data, model, training, calibration, evaluation, inference, promotion)
    return AppConfig(
        project_root=project_root,
        source_path=source_path,
        source_encoding=decoded.actual_encoding,
        data=data,
        model=model,
        training=training,
        calibration=calibration,
        evaluation=evaluation,
        inference=inference,
        run=run,
        promotion=promotion,
    )


def _build_data_config(raw: dict[str, Any], project_root: Path) -> DataConfig:
    values = dict(raw)
    values["manifest_path"] = _resolve_path(
        project_root,
        values.get("manifest_path", "项目/Data/metadata.csv"),
        "data.manifest_path",
    )
    unknown_path = values.get("unknown_manifest_path", "项目/Data/unknown_metadata.csv")
    values["unknown_manifest_path"] = (
        None
        if unknown_path is None
        else _resolve_path(project_root, unknown_path, "data.unknown_manifest_path")
    )
    values["data_root"] = _resolve_path(
        project_root, values.get("data_root", "项目/References"), "data.data_root"
    )
    if values.get("text_encoding") is not None:
        try:
            values["text_encoding"] = normalize_encoding(str(values["text_encoding"]))
        except ValueError as exc:
            raise ConfigurationError(f"data.text_encoding：{exc}") from exc
    return _construct(DataConfig, values, "data")


def _build_training_config(raw: dict[str, Any], project_root: Path) -> TrainingConfig:
    values = dict(raw)
    values["checkpoint_path"] = _resolve_path(
        project_root,
        values.get("checkpoint_path", "项目/Models/best_multitask_model.pth"),
        "training.checkpoint_path",
    )
    values["log_dir"] = _resolve_path(
        project_root, values.get("log_dir", "项目/Results/logs"), "training.log_dir"
    )
    return _construct(TrainingConfig, values, "training")


def _build_calibration_config(raw: dict[str, Any], project_root: Path) -> CalibrationConfig:
    values = dict(raw)
    values["output_checkpoint_path"] = _resolve_path(
        project_root,
        values.get("output_checkpoint_path", "项目/Models/best_multitask_model.calibrated.pth"),
        "calibration.output_checkpoint_path",
    )
    return _construct(CalibrationConfig, values, "calibration")


def _build_evaluation_config(raw: dict[str, Any], project_root: Path) -> EvaluationConfig:
    values = dict(raw)
    values["report_dir"] = _resolve_path(
        project_root,
        values.get("report_dir", "项目/Results/evaluation"),
        "evaluation.report_dir",
    )
    return _construct(EvaluationConfig, values, "evaluation")


def _build_inference_config(raw: dict[str, Any], project_root: Path) -> InferenceConfig:
    values = dict(raw)
    values["checkpoint_path"] = _resolve_path(
        project_root,
        values.get("checkpoint_path", "项目/Models/best_multitask_model.pth"),
        "inference.checkpoint_path",
    )
    return _construct(InferenceConfig, values, "inference")


def _build_run_config(raw: dict[str, Any], project_root: Path) -> RunConfig:
    values = dict(raw)
    values["root_dir"] = _resolve_path(
        project_root,
        values.get("root_dir", "项目/Results/experiments/drynaria_closed_set/runs"),
        "run.root_dir",
    )
    return _construct(RunConfig, values, "run")


def _build_promotion_config(raw: dict[str, Any], project_root: Path) -> PromotionConfig:
    values = dict(raw)
    values["target_checkpoint_path"] = _resolve_path(
        project_root,
        values.get(
            "target_checkpoint_path",
            "项目/Models/best_multitask_model.calibrated.pth",
        ),
        "promotion.target_checkpoint_path",
    )
    return _construct(PromotionConfig, values, "promotion")


def _resolve_project_root(source_path: Path, value: Any) -> Path:
    if not isinstance(value, str | Path):
        raise ConfigurationError("project_root 必须是路径字符串。")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = source_path.parent / candidate
    return candidate.resolve()


def _resolve_path(project_root: Path, value: Any, config_path: str) -> Path:
    if not isinstance(value, str | Path):
        raise ConfigurationError(f"{config_path} 必须是路径字符串，实际为 {type(value).__name__}。")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigurationError(f"{path} 必须是映射对象。")
    return dict(value)


def _reject_dataclass_unknown_keys(
    raw: dict[str, Any], section_type: type[ConfigSection], path: str
) -> None:
    _reject_unknown_keys(raw, {field.name for field in fields(section_type)}, path)


def _reject_unknown_keys(raw: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(raw).difference(allowed))
    if unknown:
        name = f"{path}.{unknown[0]}" if path else unknown[0]
        raise ConfigurationError(f"未知配置项：{name}")


def _construct(section_type: type[ConfigSection], raw: dict[str, Any], path: str) -> ConfigSection:
    try:
        return section_type(**raw)
    except TypeError as exc:
        raise ConfigurationError(f"{path} 配置类型错误：{exc}") from exc


def _validate(
    data: DataConfig,
    model: ModelConfig,
    training: TrainingConfig,
    calibration: CalibrationConfig,
    evaluation: EvaluationConfig,
    inference: InferenceConfig,
    promotion: PromotionConfig,
) -> None:
    if data.provenance not in {"unknown", "synthetic", "real"}:
        raise ConfigurationError("data.provenance 只允许 unknown、synthetic 或 real。")
    _positive_int(data.image_size, "data.image_size")
    _positive_int(data.batch_size, "data.batch_size")
    _nonnegative_int(data.num_workers, "data.num_workers")
    _positive_int(data.seed, "data.seed", allow_zero=True)
    _positive_int(data.max_image_bytes, "data.max_image_bytes")
    _positive_int(data.max_image_pixels, "data.max_image_pixels")
    _nonnegative_int(data.perceptual_hash_distance, "data.perceptual_hash_distance")
    _boolean(data.use_weighted_sampler, "data.use_weighted_sampler")
    _probability(data.train_ratio, "data.train_ratio", inclusive_zero=False)
    _probability(data.val_ratio, "data.val_ratio", inclusive_zero=False)
    _probability(data.test_ratio, "data.test_ratio", inclusive_zero=False)
    if data.split_algorithm_version not in {
        "group-stratified-v1",
        "group-stratified-near-duplicate-components-v2",
    }:
        raise ConfigurationError(
            "data.split_algorithm_version 只允许 group-stratified-v1 或 "
            "group-stratified-near-duplicate-components-v2。"
        )
    if not math.isclose(
        data.train_ratio + data.val_ratio + data.test_ratio,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ConfigurationError("data 的训练、验证、测试切分比例之和必须为 1。")

    if not isinstance(model.backbone, str) or not model.backbone.strip():
        raise ConfigurationError("model.backbone 必须是非空字符串。")
    _boolean(model.pretrained, "model.pretrained")
    _boolean(model.freeze_backbone, "model.freeze_backbone")
    _nonnegative_int(model.unfreeze_epoch, "model.unfreeze_epoch")
    _probability(model.dropout, "model.dropout", inclusive_one=False)

    _positive_int(training.epochs, "training.epochs")
    _positive_number(training.learning_rate, "training.learning_rate")
    _nonnegative_number(training.weight_decay, "training.weight_decay")
    task_loss_weights = (
        training.claimed_herb_weight,
        training.actual_species_weight,
        training.authenticity_weight,
    )
    _nonnegative_number(task_loss_weights[0], "training.claimed_herb_weight")
    _nonnegative_number(task_loss_weights[1], "training.actual_species_weight")
    _nonnegative_number(task_loss_weights[2], "training.authenticity_weight")
    if not any(task_loss_weights):
        raise ConfigurationError("training 的三个任务 loss 权重不能同时为 0。")
    _probability(training.label_smoothing, "training.label_smoothing", inclusive_one=False)
    _positive_int(training.early_stop_patience, "training.early_stop_patience")
    _boolean(training.mixed_precision, "training.mixed_precision")
    _positive_number(training.gradient_clip, "training.gradient_clip")
    _probability(
        training.min_species_supervision_coverage,
        "training.min_species_supervision_coverage",
    )
    _boolean(
        training.use_authenticity_class_weights,
        "training.use_authenticity_class_weights",
    )

    if calibration.mode not in {"open_set", "closed_set_temperature"}:
        raise ConfigurationError("calibration.mode 只允许 open_set 或 closed_set_temperature。")
    if (
        not isinstance(calibration.algorithm_version, str)
        or not calibration.algorithm_version.strip()
    ):
        raise ConfigurationError("calibration.algorithm_version 必须是非空字符串。")
    if calibration.mode == "closed_set_temperature":
        if data.use_weighted_sampler and training.use_authenticity_class_weights:
            raise ConfigurationError("闭集训练只能启用一种类别平衡机制。")
        if data.unknown_manifest_path is not None:
            raise ConfigurationError(
                "closed_set_temperature 要求 data.unknown_manifest_path 为 null。"
            )

    _probability(
        calibration.max_unknown_accept_rate,
        "calibration.max_unknown_accept_rate",
    )
    _probability(
        calibration.min_known_selective_accuracy,
        "calibration.min_known_selective_accuracy",
    )
    _positive_int(evaluation.bootstrap_samples, "evaluation.bootstrap_samples")
    _probability(promotion.min_macro_f1, "promotion.min_macro_f1")
    _probability(
        promotion.min_balanced_accuracy,
        "promotion.min_balanced_accuracy",
    )
    _probability(
        promotion.min_counterfeit_recall,
        "promotion.min_counterfeit_recall",
    )

    if inference.device not in {"auto", "cpu", "cuda"}:
        raise ConfigurationError("inference.device 只允许 auto、cpu 或 cuda。")
    _boolean(inference.allow_uncalibrated, "inference.allow_uncalibrated")
    _positive_int(inference.max_image_bytes, "inference.max_image_bytes")
    _positive_int(inference.max_image_pixels, "inference.max_image_pixels")


def _boolean(value: Any, path: str) -> None:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{path} 必须是布尔值。")


def _positive_int(value: Any, path: str, allow_zero: bool = False) -> None:
    valid = isinstance(value, int) and not isinstance(value, bool)
    valid = valid and (value >= 0 if allow_zero else value > 0)
    if not valid:
        qualifier = "非负整数" if allow_zero else "正整数"
        raise ConfigurationError(f"{path} 必须是{qualifier}。")


def _nonnegative_int(value: Any, path: str) -> None:
    _positive_int(value, path, allow_zero=True)


def _positive_number(value: Any, path: str) -> None:
    valid = (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(value)
        and value > 0
    )
    if not valid:
        raise ConfigurationError(f"{path} 必须是有限正数。")


def _nonnegative_number(value: Any, path: str) -> None:
    valid = (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(value)
        and value >= 0
    )
    if not valid:
        raise ConfigurationError(f"{path} 必须是有限非负数。")


def _probability(
    value: Any,
    path: str,
    *,
    inclusive_zero: bool = True,
    inclusive_one: bool = True,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigurationError(f"{path} 必须是数值概率。")
    lower_ok = value >= 0 if inclusive_zero else value > 0
    upper_ok = value <= 1 if inclusive_one else value < 1
    if not lower_ok or not upper_ok:
        left = "[" if inclusive_zero else "("
        right = "]" if inclusive_one else ")"
        raise ConfigurationError(f"{path} 必须位于 {left}0, 1{right}。")
