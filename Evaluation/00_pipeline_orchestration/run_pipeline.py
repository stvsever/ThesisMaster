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
from datetime import datetime
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
    parser.add_argument("--cycles", type=int, default=1, help="Number of iterative pipeline cycles to execute.")
    parser.add_argument(
        "--resume-from-run",
        type=str,
        default="",
        help="Optional prior run_id to use as memory seed when iterative memory is enabled.",
    )
    parser.add_argument(
        "--history-root",
        type=str,
        default="",
        help="Optional persistent history root. Defaults to Evaluation/05_integrated_pipeline_runs/_history.",
    )
    parser.add_argument(
        "--profile-memory-window",
        type=int,
        default=3,
        help="How many prior cycles to include when assembling profile memory.",
    )
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
    cycles = max(1, int(args.cycles))
    base_run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    history_root = (
        Path(args.history_root).expanduser().resolve()
        if str(args.history_root).strip()
        else (repo_root / "Evaluation/05_integrated_pipeline_runs/_history").resolve()
    )

    for cycle_index in range(1, cycles + 1):
        cmd = [
            args.python_exe,
            str(target_script),
            "--run-id",
            base_run_id,
            "--enable-iterative-memory",
            "--cycle-index",
            str(cycle_index),
            "--cycles",
            str(cycles),
            "--memory-policy",
            "v1_weighted_fusion",
            "--history-root",
            str(history_root),
            "--profile-memory-window",
            str(max(1, int(args.profile_memory_window))),
        ]
        if str(args.resume_from_run).strip():
            cmd.extend(["--resume-from-run", str(args.resume_from_run).strip()])
        cmd.extend(passthrough)
        print(f"[PHOENIX] Running cycle {cycle_index}/{cycles}", flush=True)
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            return int(proc.returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
