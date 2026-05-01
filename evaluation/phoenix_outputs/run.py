#!/usr/bin/env python3
"""CLI for preparing PHOENIX outputs for the survey-analysis judge."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import List, Optional

try:
    from .agentic_quality_gate import REFINED_OUTPUTS_JSON, refine_file
    from .canonicalize_outputs import canonicalize_file, write_template_outputs
    from .paths import (
        CASE_CONTEXTS_JSON,
        CASE_INPUTS_JSON,
        SYSTEM_OUTPUTS_JSON,
        SURVEY_ANALYSIS_CONTEXTS_JSON,
        SURVEY_ANALYSIS_SYSTEM_OUTPUTS_JSON,
        ensure_dirs,
    )
    from .qualtrics_inputs import write_case_inputs
    from .rule_based_fixture import write_rule_based_outputs
    from .phoenix_engine_runner import run_phoenix_engine
except ImportError:  # direct execution: python evaluation/phoenix_outputs/run.py
    from agentic_quality_gate import REFINED_OUTPUTS_JSON, refine_file  # type: ignore
    from canonicalize_outputs import canonicalize_file, write_template_outputs  # type: ignore
    from paths import (  # type: ignore
        CASE_CONTEXTS_JSON,
        CASE_INPUTS_JSON,
        SYSTEM_OUTPUTS_JSON,
        SURVEY_ANALYSIS_CONTEXTS_JSON,
        SURVEY_ANALYSIS_SYSTEM_OUTPUTS_JSON,
        ensure_dirs,
    )
    from qualtrics_inputs import write_case_inputs  # type: ignore
    from rule_based_fixture import write_rule_based_outputs  # type: ignore
    from phoenix_engine_runner import run_phoenix_engine  # type: ignore


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evaluation/phoenix_outputs/run.py",
        description="Prepare exact Qualtrics-matched PHOENIX inputs and canonical system outputs.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    extract = sub.add_parser(
        "extract-inputs",
        help="Extract exact non-image case inputs from the Qualtrics source files.",
    )
    extract.add_argument("--inputs-out", type=Path, default=CASE_INPUTS_JSON)
    extract.add_argument("--contexts-out", type=Path, default=CASE_CONTEXTS_JSON)

    template = sub.add_parser(
        "write-template",
        help="Write an empty canonical system-output template for all cases.",
    )
    template.add_argument("--case-inputs", type=Path, default=CASE_INPUTS_JSON)
    template.add_argument("--out", type=Path, default=None)

    canonicalize = sub.add_parser(
        "canonicalize",
        help="Canonicalize real PHOENIX outputs into the judge-ready JSON format.",
    )
    canonicalize.add_argument("--raw", type=Path, required=True)
    canonicalize.add_argument("--case-inputs", type=Path, default=CASE_INPUTS_JSON)
    canonicalize.add_argument("--out", type=Path, default=SYSTEM_OUTPUTS_JSON)

    quality_gate = sub.add_parser(
        "quality-gate",
        help="Apply the PHOENIX agentic quality gate to a canonical output bundle.",
    )
    quality_gate.add_argument("--raw", type=Path, default=SYSTEM_OUTPUTS_JSON)
    quality_gate.add_argument("--case-inputs", type=Path, default=CASE_INPUTS_JSON)
    quality_gate.add_argument("--out", type=Path, default=REFINED_OUTPUTS_JSON)
    quality_gate.add_argument(
        "--sync",
        action="store_true",
        help="Also copy refined outputs to the canonical analysis locations.",
    )

    fixture = sub.add_parser(
        "write-fixture",
        help="Write deterministic rule-based fixture outputs for dry-run testing only.",
    )
    fixture.add_argument("--case-inputs", type=Path, default=CASE_INPUTS_JSON)
    fixture.add_argument("--out", type=Path, default=SYSTEM_OUTPUTS_JSON)

    sync = sub.add_parser(
        "sync-to-analysis",
        help="Copy prepared contexts and system outputs into survey_analysis/data.",
    )
    sync.add_argument("--contexts", type=Path, default=CASE_CONTEXTS_JSON)
    sync.add_argument("--system-outputs", type=Path, default=SYSTEM_OUTPUTS_JSON)
    sync.add_argument("--skip-system-outputs", action="store_true")

    all_cmd = sub.add_parser(
        "prepare-fixture-analysis",
        help="Extract inputs, write fixture outputs, validate, and sync for a dry run.",
    )
    all_cmd.add_argument("--inputs-out", type=Path, default=CASE_INPUTS_JSON)
    all_cmd.add_argument("--contexts-out", type=Path, default=CASE_CONTEXTS_JSON)
    all_cmd.add_argument("--system-out", type=Path, default=SYSTEM_OUTPUTS_JSON)

    engine_cmd = sub.add_parser(
        "run-engine",
        help="Run the PHOENIX LLM engine (Gemini Flash via OpenRouter) for all 10 cases.",
    )
    engine_cmd.add_argument(
        "--model", default="google/gemini-3.1-flash-lite-preview",
        help="OpenRouter model identifier (default: google/gemini-3.1-flash-lite-preview).",
    )
    engine_cmd.add_argument(
        "--workers", type=int, default=10,
        help="Max concurrent LLM calls (default: 10).",
    )
    engine_cmd.add_argument(
        "--case-inputs", type=Path, default=CASE_INPUTS_JSON,
        help="Path to case_inputs.json (from extract-inputs).",
    )
    engine_cmd.add_argument(
        "--case-contexts", type=Path, default=CASE_CONTEXTS_JSON,
        help="Path to case_contexts.json (from extract-inputs).",
    )

    return p


def _copy(src: Path, dst: Path) -> Path:
    if not src.exists():
        raise FileNotFoundError(f"Cannot sync missing file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _cmd_extract_inputs(args: argparse.Namespace) -> int:
    paths = write_case_inputs(args.inputs_out, args.contexts_out)
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
    return 0


def _cmd_write_template(args: argparse.Namespace) -> int:
    out = args.out or (args.case_inputs.parent.parent / "outputs" / "system_outputs_template.json")
    path = write_template_outputs(args.case_inputs, out)
    print(str(path))
    return 0


def _cmd_canonicalize(args: argparse.Namespace) -> int:
    paths = canonicalize_file(args.raw, case_inputs_path=args.case_inputs, output_path=args.out)
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
    return 0


def _cmd_quality_gate(args: argparse.Namespace) -> int:
    paths = refine_file(
        args.raw,
        case_inputs_path=args.case_inputs,
        output_path=args.out,
        sync_to_pipeline=args.sync,
    )
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
    return 0


def _cmd_write_fixture(args: argparse.Namespace) -> int:
    path = write_rule_based_outputs(args.case_inputs, args.out)
    paths = canonicalize_file(path, case_inputs_path=args.case_inputs, output_path=args.out)
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
    return 0


def _cmd_sync(args: argparse.Namespace) -> int:
    copied = {
        "case_contexts": str(_copy(args.contexts, SURVEY_ANALYSIS_CONTEXTS_JSON)),
    }
    if not args.skip_system_outputs:
        copied["system_outputs"] = str(
            _copy(args.system_outputs, SURVEY_ANALYSIS_SYSTEM_OUTPUTS_JSON)
        )
    print(json.dumps(copied, indent=2))
    return 0


def _cmd_prepare_fixture_analysis(args: argparse.Namespace) -> int:
    ensure_dirs()
    write_case_inputs(args.inputs_out, args.contexts_out)
    write_rule_based_outputs(args.inputs_out, args.system_out)
    canonicalize_file(args.system_out, case_inputs_path=args.inputs_out, output_path=args.system_out)
    _copy(args.contexts_out, SURVEY_ANALYSIS_CONTEXTS_JSON)
    _copy(args.system_out, SURVEY_ANALYSIS_SYSTEM_OUTPUTS_JSON)
    print(json.dumps({
        "case_inputs": str(args.inputs_out),
        "case_contexts": str(args.contexts_out),
        "system_outputs": str(args.system_out),
        "analysis_case_contexts": str(SURVEY_ANALYSIS_CONTEXTS_JSON),
        "analysis_system_outputs": str(SURVEY_ANALYSIS_SYSTEM_OUTPUTS_JSON),
    }, indent=2))
    return 0


def _cmd_run_engine(args: argparse.Namespace) -> int:
    """Run the PHOENIX LLM engine and sync outputs to survey_analysis/data."""
    import logging
    import os as _os
    from pathlib import Path as _Path

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")

    # Load .env if present (for OPENROUTER_API_KEY)
    env_file = _Path(__file__).resolve().parents[2] / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                _os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

    run_phoenix_engine(
        model=args.model,
        max_workers=args.workers,
        case_inputs_path=args.case_inputs,
        case_contexts_path=args.case_contexts,
        also_copy_to_pipeline=True,
        also_copy_to_judge_dir=True,
    )
    # Report output locations
    print(json.dumps({
        "primary":          str(SYSTEM_OUTPUTS_JSON),
        "pipeline_system":  str(SURVEY_ANALYSIS_SYSTEM_OUTPUTS_JSON),
    }, indent=2))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    command = args.command
    if command == "extract-inputs":
        return _cmd_extract_inputs(args)
    if command == "write-template":
        return _cmd_write_template(args)
    if command == "canonicalize":
        return _cmd_canonicalize(args)
    if command == "quality-gate":
        return _cmd_quality_gate(args)
    if command == "write-fixture":
        return _cmd_write_fixture(args)
    if command == "sync-to-analysis":
        return _cmd_sync(args)
    if command == "prepare-fixture-analysis":
        return _cmd_prepare_fixture_analysis(args)
    if command == "run-engine":
        return _cmd_run_engine(args)
    raise ValueError(f"Unhandled command {command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
