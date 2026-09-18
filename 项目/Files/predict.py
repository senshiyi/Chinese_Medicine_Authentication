"""稳定的单图片预测命令行入口。"""

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="中药图片三任务识别")
    parser.add_argument("image", nargs="?", type=Path, help="待识别图片路径（兼容位置参数）")
    parser.add_argument("--image", dest="image_option", type=Path, help="待识别图片路径")
    parser.add_argument("--checkpoint", type=Path, required=True, help="schema v1 checkpoint")
    parser.add_argument("--device", default="auto", help="auto、cpu 或 cuda")
    parser.add_argument("--json", action="store_true", help="输出完整 schema-v1 JSON")
    parser.add_argument(
        "--experimental-closed-set",
        action="store_true",
        help="显式启用仅限当前数据集范围的实验闭集真伪推理；必须同时指定 --json",
    )
    parser.add_argument(
        "--allow-uncalibrated",
        action="store_true",
        help="仅供开发查看未校准的拒识 JSON；必须同时指定 --json",
    )
    return parser


def main(
    argv: Sequence[str] | None = None, service_factory: Callable[[Path, str], Any] | None = None
) -> int:
    from herb_authentication.environment import configure_utf8_stdio, run_bootstrap_preflight

    configure_utf8_stdio()
    if not run_bootstrap_preflight().python_supported:
        print("当前 Python 版本不受支持，请使用 3.11 或 3.12。", file=sys.stderr)
        return 2
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.image is not None and args.image_option is not None:
        parser.error("图片路径只能通过位置参数或 --image 指定一次")
    image = args.image_option or args.image
    if image is None:
        parser.error("必须提供图片路径位置参数或 --image")
    if args.allow_uncalibrated and not args.json:
        parser.error("--allow-uncalibrated 必须与 --json 同时使用")
    if args.experimental_closed_set and not args.json:
        parser.error("--experimental-closed-set 必须与 --json 同时使用")
    if service_factory is None:
        from herb_authentication.inference.service import InferenceService

        def service_factory(checkpoint: Path, device: str) -> InferenceService:
            return InferenceService.from_checkpoint(checkpoint, device=device)

    try:
        service = service_factory(args.checkpoint, args.device)
        result = (
            service.predict_experimental_closed_set(image)
            if args.experimental_closed_set
            else service.predict_image(image)
        )
    except Exception as exc:
        from herb_authentication.errors import HerbAuthenticationError

        if not isinstance(exc, HerbAuthenticationError):
            raise
        print(f"预测失败：{exc}", file=sys.stderr)
        return 2
    if (
        not args.experimental_closed_set
        and args.allow_uncalibrated
        and (
            not result.rejected
            or "正品" in result.message
            or "伪品" in result.message
            or "%" in result.message
        )
    ):
        print("未校准开发输出违反拒识约束。", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, allow_nan=False))
    else:
        print(result.message)
        print(result.disclaimer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
