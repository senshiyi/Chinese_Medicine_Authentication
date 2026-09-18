"""闭集模型的本地实验试运行入口；输出概率不作为正式准确率。"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

DEFAULT_CHECKPOINT = (
    SCRIPT_DIR.parent / "Models" / "experiments" / "drynaria_closed_set_preview.candidate.pth"
)
DISCLAIMER = "实验试运行：概率未校准，不能作为药品质量或医疗结论。"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="骨碎补真伪识别实验试运行")
    parser.add_argument("image", type=Path, help="待识别图片路径")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="schema v1 试运行 checkpoint",
    )
    parser.add_argument("--device", default="cpu", help="cpu、cuda 或 auto")
    parser.add_argument("--json", action="store_true", help="输出完整 JSON")
    return parser


def main(
    argv: Sequence[str] | None = None,
    service_factory: Callable[[Path, str], Any] | None = None,
) -> int:
    from herb_authentication.environment import configure_utf8_stdio

    configure_utf8_stdio()
    args = build_parser().parse_args(argv)
    if service_factory is None:
        from herb_authentication.inference.service import InferenceService

        def service_factory(checkpoint: Path, device: str) -> InferenceService:
            return InferenceService.from_checkpoint(checkpoint, device=device)

    try:
        service = service_factory(args.checkpoint, args.device)
        logits = service.predict_logits(args.image)
        result = _format_preview(logits.authenticity, service.model_version)
    except Exception as exc:
        from herb_authentication.errors import HerbAuthenticationError

        if not isinstance(exc, HerbAuthenticationError | OSError | RuntimeError | ValueError):
            raise
        print(f"预测失败：{exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result["message"])
        print(result["disclaimer"])
    return 0


def _format_preview(authenticity_logits: Any, model_version: str) -> dict[str, Any]:
    import torch

    if not isinstance(authenticity_logits, torch.Tensor) or authenticity_logits.shape != (1, 2):
        raise RuntimeError("真伪模型输出应为 [1,2]")
    if not bool(torch.isfinite(authenticity_logits).all()):
        raise RuntimeError("真伪模型输出包含非有限数值")
    probabilities = torch.softmax(authenticity_logits, dim=1)[0].detach().cpu().tolist()
    authentic_probability = float(probabilities[0])
    counterfeit_probability = float(probabilities[1])
    predicted_label = (
        "authentic" if authentic_probability >= counterfeit_probability else "counterfeit"
    )
    predicted_confidence = max(authentic_probability, counterfeit_probability)

    if predicted_label == "authentic":
        message = f"此药品为正品，有{authentic_probability * 100:.1f}%的可能该药材为正品。"
    else:
        message = (
            f"实际药材无法确定，有{counterfeit_probability * 100:.1f}%"
            "的可能该药材为伪品，请人工复核。"
        )

    return {
        "schema_version": 1,
        "official": False,
        "experimental_preview": True,
        "probability_type": "uncalibrated_softmax",
        "message": message,
        "disclaimer": DISCLAIMER,
        "model_version": model_version,
        "authenticity": {
            "predicted_label": predicted_label,
            "predicted_confidence": round(predicted_confidence, 6),
            "authentic_probability": round(authentic_probability, 6),
            "counterfeit_probability": round(counterfeit_probability, 6),
        },
        "actual_species": {
            "status": "unknown",
            "reason": "当前伪品实际物种未经检验，试运行模型不输出实际药材名称。",
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
