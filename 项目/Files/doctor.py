"""项目运行环境诊断入口。"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from herb_authentication.environment import (
    configure_utf8_stdio,
    run_bootstrap_preflight,
    run_dependency_probe,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="检查中药识别项目的 Python、编码与依赖环境")
    parser.add_argument("--json", action="store_true", help="以 UTF-8 JSON 输出诊断结果")
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="重量级依赖子进程探测超时秒数（默认 60）",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_utf8_stdio()
    args = build_parser().parse_args(argv)
    if args.timeout <= 0:
        build_parser().error("--timeout 必须大于 0")

    bootstrap = run_bootstrap_preflight()
    dependencies = run_dependency_probe(timeout_seconds=args.timeout)
    ok = bootstrap.python_supported and dependencies.ok
    report = {
        "schema_version": 1,
        "ok": ok,
        "bootstrap": bootstrap.to_dict(),
        "dependencies": dependencies.to_dict(),
        "recommendations": _recommendations(bootstrap.python_supported, dependencies.ok),
    }

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("环境检查通过。" if ok else "环境检查未通过。")
        for recommendation in report["recommendations"]:
            print(f"- {recommendation}")
    return 0 if ok else 2


def _recommendations(python_supported: bool, dependencies_ok: bool) -> list[str]:
    recommendations: list[str] = []
    if not python_supported:
        recommendations.append("请使用项目专属 Python 3.11 或 3.12 环境。")
    if not dependencies_ok:
        recommendations.append("请在项目 .venv 中重新安装 requirements.txt，勿混用全局依赖。")
    if not recommendations:
        recommendations.append("依赖来自兼容的隔离环境，可以继续运行项目。")
    return recommendations


if __name__ == "__main__":
    sys.exit(main())
