#!/usr/bin/env python3
"""
run_pipeline.py

Future-ready PHOENIX pipeline entrypoint.

Current mode:
- synthetic_v1: delegates to run_pseudodata_to_impact.py

Planned future modes:
- full_engine: end-to-end integration with Agentic steps 03-05 and UI orchestration
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence


def parse_args(argv: Optional[Sequence[str]] = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="PHOENIX Engine pipeline launcher with mode-based routing."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="synthetic_v1",
        choices=["synthetic_v1", "full_engine"],
        help="Pipeline mode. Use synthetic_v1 for current thesis evaluation flow.",
    )
    parser.add_argument(
        "--ui",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reserved for future web-app integrated execution mode.",
    )
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    return parser.parse_known_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args, passthrough = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]

    if args.mode == "full_engine":
        print(
            "[PHOENIX] mode=full_engine is reserved for future integration (Agentic steps 03-05 + UI).",
            flush=True,
        )
        print(
            "[PHOENIX] Use --mode synthetic_v1 for the current research pipeline.",
            flush=True,
        )
        return 2

    if bool(args.ui):
        print(
            "[PHOENIX] --ui is currently a reserved flag. CLI execution continues without UI bootstrap.",
            flush=True,
        )

    target_script = repo_root / "Evaluation/00_pipeline_orchestration/run_pseudodata_to_impact.py"
    cmd = [args.python_exe, str(target_script)] + passthrough
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())

