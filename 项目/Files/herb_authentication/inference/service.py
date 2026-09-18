"""与 CLI、HTTP 框架解耦的单图片推理服务。"""

from __future__ import annotations

import math
import pickle
import warnings
from collections.abc import Callable, Mapping, Sequence
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError
from torch import nn

from herb_authentication.data.transforms import build_eval_transform
from herb_authentication.errors import CheckpointCompatibilityError, InputImageError
from herb_authentication.models.multitask import (
    MultiTaskHerbModel,
    MultiTaskLogits,
    build_torchvision_encoder,
)

from .calibration import CalibrationBundle
from .closed_set import ClosedSetCalibration, ClosedSetPredictionResult
from .formatter import ResultFormatter
from .policy import PredictionScores, ScoredLabel, apply_rejection_policy
from .results import PredictionResult

Preprocess = Callable[[Image.Image], torch.Tensor]


class InferenceService:
    """一次装配模型和预处理，多次执行受限的单图片推理。"""

    def __init__(
        self,
        *,
        model: nn.Module,
        preprocess: Preprocess,
        claimed_herb_classes: Sequence[str],
        actual_species_classes: Sequence[str],
        calibration: CalibrationBundle,
        model_version: str,
        device: str | torch.device = "auto",
        closed_set_calibration: ClosedSetCalibration | None = None,
        counterfeit_species_verified: bool | None = None,
        experimental_closed_set: bool | None = None,
        open_set_calibrated: bool | None = None,
        model_fingerprint: str | None = None,
        data_fingerprint: str | None = None,
        split_fingerprint: str | None = None,
        max_image_bytes: int = 10 * 1024 * 1024,
        max_pixels: int = 40_000_000,
        max_decoded_bytes: int = 120_000_000,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model 必须是 torch.nn.Module")
        if not callable(preprocess):
            raise TypeError("preprocess 必须可调用")
        self.claimed_herb_classes = _class_names(claimed_herb_classes, "claimed_herb")
        self.actual_species_classes = _class_names(actual_species_classes, "actual_species")
        if not isinstance(calibration, CalibrationBundle):
            raise TypeError("calibration 必须是 CalibrationBundle")
        if not isinstance(model_version, str) or not model_version.strip():
            raise ValueError("model_version 必须是非空字符串")
        self.max_image_bytes = _positive_limit(max_image_bytes, "max_image_bytes")
        self.max_pixels = _positive_limit(max_pixels, "max_pixels")
        self.max_decoded_bytes = _positive_limit(max_decoded_bytes, "max_decoded_bytes")
        self.device = _resolve_device(device)
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        self.calibration = calibration
        self.model_version = model_version.strip()
        self.closed_set_calibration = closed_set_calibration
        self.counterfeit_species_verified = counterfeit_species_verified
        self.experimental_closed_set = experimental_closed_set
        self.open_set_calibrated = open_set_calibrated
        self.model_fingerprint = model_fingerprint
        self.data_fingerprint = data_fingerprint
        self.split_fingerprint = split_fingerprint
        self.formatter = ResultFormatter()

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: str | torch.device = "auto",
        *,
        model_factory: Callable[[Mapping[str, Any]], nn.Module] | None = None,
        preprocess_factory: Callable[[Mapping[str, Any]], Preprocess] | None = None,
        max_image_bytes: int = 10 * 1024 * 1024,
        max_pixels: int = 40_000_000,
        max_decoded_bytes: int = 120_000_000,
    ) -> InferenceService:
        """严格读取 schema v1 checkpoint；测试可注入轻量装配工厂。"""

        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise CheckpointCompatibilityError(f"checkpoint 不存在：{source}")
        try:
            state = torch.load(source, map_location="cpu", weights_only=True)
        except (
            EOFError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            pickle.UnpicklingError,
        ) as exc:
            raise CheckpointCompatibilityError(f"无法安全加载 checkpoint：{source}") from exc
        if not isinstance(state, Mapping) or state.get("schema_version") != 1:
            raise CheckpointCompatibilityError("checkpoint schema_version 不是受支持的 v1")
        metadata = state.get("metadata")
        model_state = state.get("model_state")
        if not isinstance(metadata, Mapping) or not isinstance(model_state, Mapping):
            raise CheckpointCompatibilityError("checkpoint 缺少 metadata 或 model_state")

        claimed = _metadata_classes(metadata, "claimed_herb_classes")
        species = _metadata_classes(metadata, "actual_species_classes")
        authenticity = _metadata_classes(metadata, "authenticity_classes")
        if authenticity != ("authentic", "counterfeit"):
            raise CheckpointCompatibilityError(
                "checkpoint authenticity_classes 必须依次为 authentic、counterfeit"
            )
        try:
            calibration = CalibrationBundle.from_dict(metadata.get("calibration"))
        except (TypeError, ValueError) as exc:
            raise CheckpointCompatibilityError("checkpoint calibration 不兼容") from exc
        closed_set_calibration = None
        if metadata.get("closed_set_calibration") is not None:
            try:
                closed_set_calibration = ClosedSetCalibration.from_dict(
                    metadata["closed_set_calibration"]
                )
            except (TypeError, ValueError) as exc:
                raise CheckpointCompatibilityError(
                    "checkpoint closed_set_calibration 不兼容"
                ) from exc

        if model_factory is None:
            backbone = metadata.get("backbone")
            if not isinstance(backbone, str):
                raise CheckpointCompatibilityError("checkpoint 缺少 backbone")
            encoder = build_torchvision_encoder(backbone, pretrained=False)
            dropout = metadata.get("dropout", 0.3)
            model = MultiTaskHerbModel(
                encoder.encoder,
                encoder.feature_dim,
                len(claimed),
                len(species),
                dropout=float(dropout),
            )
        else:
            model = model_factory(metadata)
            if not isinstance(model, nn.Module):
                raise CheckpointCompatibilityError("model_factory 必须返回 torch.nn.Module")
        try:
            model.load_state_dict(model_state, strict=True)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise CheckpointCompatibilityError("checkpoint model_state 与模型结构不兼容") from exc

        if preprocess_factory is None:
            image_size = metadata.get("image_size")
            if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size <= 0:
                raise CheckpointCompatibilityError("checkpoint image_size 必须是正整数")
            normalization = metadata.get("normalization", {})
            if normalization is None:
                normalization = {}
            if not isinstance(normalization, Mapping):
                raise CheckpointCompatibilityError("checkpoint normalization 必须是映射")
            try:
                preprocess = build_eval_transform(
                    image_size,
                    normalization.get("mean", (0.485, 0.456, 0.406)),
                    normalization.get("std", (0.229, 0.224, 0.225)),
                )
            except (TypeError, ValueError) as exc:
                raise CheckpointCompatibilityError("checkpoint 预处理配置不兼容") from exc
        else:
            preprocess = preprocess_factory(metadata)
            if not callable(preprocess):
                raise CheckpointCompatibilityError("preprocess_factory 必须返回可调用对象")

        model_version = metadata.get("model_version", "multitask-v1")
        if not isinstance(model_version, str) or not model_version.strip():
            raise CheckpointCompatibilityError("checkpoint model_version 必须是非空字符串")
        return cls(
            model=model,
            preprocess=preprocess,
            claimed_herb_classes=claimed,
            actual_species_classes=species,
            calibration=calibration,
            model_version=model_version,
            device=device,
            closed_set_calibration=closed_set_calibration,
            counterfeit_species_verified=metadata.get("counterfeit_species_verified"),
            experimental_closed_set=metadata.get("experimental_closed_set"),
            open_set_calibrated=metadata.get("open_set_calibrated"),
            model_fingerprint=_sha256_file(source),
            data_fingerprint=metadata.get("data_fingerprint"),
            split_fingerprint=metadata.get("split_fingerprint"),
            max_image_bytes=max_image_bytes,
            max_pixels=max_pixels,
            max_decoded_bytes=max_decoded_bytes,
        )

    def predict_image(self, image: Path | bytes) -> PredictionResult:
        logits = self.predict_logits(image)
        scores = self._scores_from_logits(logits)
        decision = apply_rejection_policy(scores, self.calibration)
        selected_species = self._selected_species(scores.actual_species)
        draft = PredictionResult(
            model_version=self.model_version,
            is_calibrated=self.calibration.is_calibrated,
            claimed_herb=scores.claimed_herb,
            actual_species=selected_species,
            authenticity=scores.authenticity,
            rejected=decision.rejected,
            rejection_reasons=decision.reason_codes,
            message="",
        )
        return draft.with_message(self.formatter.format(draft))

    def predict_experimental_closed_set(self, image: Path | bytes) -> ClosedSetPredictionResult:
        """运行显式实验闭集真伪推理，不使用开放集拒识策略。"""

        calibration = self._require_closed_set_metadata()
        logits = self.predict_logits(image)
        authenticity_logits = _head_logits(logits, "authenticity", 2)
        probabilities_tensor = torch.softmax(authenticity_logits / calibration.temperature, dim=1)[
            0
        ]
        probabilities = {
            "authentic": float(probabilities_tensor[0].item()),
            "counterfeit": float(probabilities_tensor[1].item()),
        }
        authenticity_index = int(probabilities_tensor.argmax().item())
        authenticity_label = ("authentic", "counterfeit")[authenticity_index]
        confidence = probabilities[authenticity_label]
        if authenticity_label == "authentic":
            message = f"此药品为正品，有{confidence:.1%}的可能该药材为正品。"
        elif not self.counterfeit_species_verified:
            message = f"实际药材无法确定，有{confidence:.1%}的可能该药材为伪品，请人工复核。"
        else:
            claimed_index = int(logits.claimed_herb.argmax(dim=1).item())
            species_index = int(logits.actual_species.argmax(dim=1).item())
            message = (
                f"此药品为伪品，有{confidence:.1%}的可能该药材为伪品，"
                f"声称为‘{self.claimed_herb_classes[claimed_index]}’，"
                f"实际应为‘{self.actual_species_classes[species_index]}’。"
            )
        return ClosedSetPredictionResult(
            model_version=self.model_version,
            authenticity=ScoredLabel(authenticity_label, confidence),
            probabilities=probabilities,
            actual_species_verified=self.counterfeit_species_verified,
            message=message,
            model_fingerprint=self.model_fingerprint,
            data_fingerprint=self.data_fingerprint,
            split_fingerprint=self.split_fingerprint,
        )

    def _require_closed_set_metadata(self) -> ClosedSetCalibration:
        calibration = self.closed_set_calibration
        metadata_complete = (
            isinstance(calibration, ClosedSetCalibration)
            and isinstance(self.counterfeit_species_verified, bool)
            and self.experimental_closed_set is True
            and self.open_set_calibrated is False
            and all(
                _is_sha256(value)
                for value in (
                    self.model_fingerprint,
                    self.data_fingerprint,
                    self.split_fingerprint,
                )
            )
        )
        if not metadata_complete:
            raise CheckpointCompatibilityError("checkpoint 缺少完整实验闭集校准、状态标志或指纹")
        return calibration

    def predict_logits(self, image: Path | bytes) -> MultiTaskLogits:
        """经同一安全解码和预处理路径返回单张图片的三头原始 logits。"""

        decoded = self._decode_image(image)
        tensor = self.preprocess(decoded)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("preprocess 必须返回 torch.Tensor")
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4 or tensor.shape[0] != 1:
            raise ValueError("preprocess 输出必须是 [C,H,W] 或 [1,C,H,W]")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("preprocess 输出包含非有限数值")

        with torch.inference_mode():
            output = self.model(tensor.to(self.device))
        return MultiTaskLogits(
            claimed_herb=_head_logits(output, "claimed_herb", len(self.claimed_herb_classes)),
            actual_species=_head_logits(output, "actual_species", len(self.actual_species_classes)),
            authenticity=_head_logits(output, "authenticity", 2),
        )

    def _decode_image(self, image: Path | bytes) -> Image.Image:
        payload = self._read_payload(image)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(BytesIO(payload)) as opened:
                    width, height = opened.size
                    if width <= 0 or height <= 0:
                        raise InputImageError("图片尺寸无效，无法解码。")
                    pixels = width * height
                    if pixels > self.max_pixels:
                        raise InputImageError(f"图片像素数 {pixels} 超过限制 {self.max_pixels}。")
                    decoded_bytes = pixels * 3
                    if decoded_bytes > self.max_decoded_bytes:
                        raise InputImageError(
                            f"图片解码后字节数 {decoded_bytes} 超过限制 {self.max_decoded_bytes}。"
                        )
                    opened.load()
                    return opened.convert("RGB")
        except InputImageError:
            raise
        except (
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
            UnidentifiedImageError,
            OSError,
            SyntaxError,
            ValueError,
        ) as exc:
            raise InputImageError("图片损坏或格式不受支持，无法解码。") from exc

    def _read_payload(self, image: Path | bytes) -> bytes:
        if isinstance(image, bytes):
            payload = image
        elif isinstance(image, Path):
            try:
                if not image.is_file():
                    raise InputImageError(f"图片文件不存在：{image}")
                size = image.stat().st_size
                if size > self.max_image_bytes:
                    raise InputImageError(f"图片字节数 {size} 超过限制 {self.max_image_bytes}。")
                payload = image.read_bytes()
            except InputImageError:
                raise
            except OSError as exc:
                raise InputImageError(f"无法读取图片文件：{image}") from exc
        else:
            raise InputImageError("image 只接受 pathlib.Path 或 bytes。")
        if not payload:
            raise InputImageError("图片字节为空，无法解码。")
        if len(payload) > self.max_image_bytes:
            raise InputImageError(f"图片字节数 {len(payload)} 超过限制 {self.max_image_bytes}。")
        return payload

    def _scores_from_logits(self, logits: Any) -> PredictionScores:
        claimed_logits = _head_logits(logits, "claimed_herb", len(self.claimed_herb_classes))
        species_logits = _head_logits(logits, "actual_species", len(self.actual_species_classes))
        authenticity_logits = _head_logits(logits, "authenticity", 2)
        claimed_probs = torch.softmax(claimed_logits / self._temperature("claimed_herb"), dim=1)[0]
        species_probs = torch.sigmoid(species_logits / self._temperature("actual_species"))[0]
        authenticity_probs = torch.softmax(
            authenticity_logits / self._temperature("authenticity"), dim=1
        )[0]
        claimed_index = int(claimed_probs.argmax().item())
        authenticity_index = int(authenticity_probs.argmax().item())
        return PredictionScores(
            claimed_herb=ScoredLabel(
                self.claimed_herb_classes[claimed_index],
                float(claimed_probs[claimed_index].item()),
            ),
            actual_species=tuple(
                ScoredLabel(label, float(species_probs[index].item()))
                for index, label in enumerate(self.actual_species_classes)
            ),
            authenticity=ScoredLabel(
                ("authentic", "counterfeit")[authenticity_index],
                float(authenticity_probs[authenticity_index].item()),
            ),
        )

    def _temperature(self, head: str) -> float:
        if not self.calibration.is_calibrated:
            return 1.0
        value = self.calibration.temperatures.get(head)
        if value is None or not math.isfinite(value) or value <= 0:
            raise CheckpointCompatibilityError(f"校准参数缺少有效温度：{head}")
        return value

    def _selected_species(self, species: tuple[ScoredLabel, ...]) -> tuple[ScoredLabel, ...]:
        thresholds = self.calibration.thresholds
        if not self.calibration.is_calibrated or thresholds is None:
            return ()
        return tuple(
            sorted(
                (item for item in species if item.confidence >= thresholds.actual_species),
                key=lambda item: (-item.confidence, item.label),
            )
        )


def _head_logits(output: Any, name: str, classes: int) -> torch.Tensor:
    value = getattr(output, name, None)
    if not isinstance(value, torch.Tensor) or value.shape != (1, classes):
        actual = None if not isinstance(value, torch.Tensor) else tuple(value.shape)
        raise RuntimeError(f"模型 {name} 输出应为 [1,{classes}]，实际为 {actual}")
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"模型 {name} 输出包含非有限数值")
    return value


def _class_names(values: Sequence[str], field: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"{field}_classes 必须是字符串序列")
    normalized = tuple(values)
    if not normalized or any(not isinstance(item, str) or not item.strip() for item in normalized):
        raise ValueError(f"{field}_classes 必须包含非空字符串")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field}_classes 不得重复")
    return normalized


def _metadata_classes(metadata: Mapping[str, Any], field: str) -> tuple[str, ...]:
    value = metadata.get(field)
    if not isinstance(value, list | tuple):
        raise CheckpointCompatibilityError(f"checkpoint {field} 必须是列表")
    try:
        return _class_names(value, field.removesuffix("_classes"))
    except (TypeError, ValueError) as exc:
        raise CheckpointCompatibilityError(f"checkpoint {field} 不兼容") from exc


def _positive_limit(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} 必须是正整数")
    return value


def _resolve_device(value: str | torch.device) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(value)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"无效 device：{value!r}") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("指定了 CUDA，但当前环境不可用")
    return device


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_file(path: Path) -> str:
    """计算当前加载 checkpoint 文件的 SHA-256，而非校准块中的源候选指纹。"""

    digest = sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CheckpointCompatibilityError(f"无法读取 checkpoint 指纹：{path}") from exc
    return digest.hexdigest()
