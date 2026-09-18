"""候选模型固定门禁晋升 CLI；不训练、不校准、不评估。"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from herb_authentication.training.promotion import (  # noqa: E402
    DEFAULT_PROMOTION_THRESHOLDS,
    PromotionError,
    PromotionRequest,
    PromotionThresholds,
    promote_candidate,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="安全晋升实验闭集候选模型")
    parser.add_argument("--config", type=Path, required=True, help="typed YAML 配置")
    parser.add_argument("--run-dir", type=Path, required=True, help="候选 run 目录")
    parser.add_argument("--candidate", type=Path, required=True, help="校准后候选 checkpoint")
    parser.add_argument("--smoke-image", type=Path, required=True, help="Path/bytes smoke 图片")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from herb_authentication.config import load_config
    from herb_authentication.environment import configure_utf8_stdio

    configure_utf8_stdio()
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        thresholds = PromotionThresholds(
            macro_f1=config.promotion.min_macro_f1,
            balanced_accuracy=config.promotion.min_balanced_accuracy,
            counterfeit_recall=config.promotion.min_counterfeit_recall,
        )
        if thresholds != DEFAULT_PROMOTION_THRESHOLDS:
            raise PromotionError("typed config 的三个晋升门槛必须精确固定为 0.80")
        run_dir = args.run_dir.expanduser().resolve()
        request = PromotionRequest(
            candidate=args.candidate.expanduser().resolve(),
            target=config.promotion.target_checkpoint_path.expanduser().resolve(),
            validation_report=run_dir / "validation" / "evaluation.json",
            locked_test_report=run_dir / "locked_test" / "evaluation.json",
            run_dir=run_dir,
        )
        record = promote_candidate(
            request,
            thresholds,
            args.smoke_image.expanduser().resolve(),
        )
    except (PromotionError, OSError, TypeError, ValueError) as exc:
        print(f"晋升失败：{exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            record.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
