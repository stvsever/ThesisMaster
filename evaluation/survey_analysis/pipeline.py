"""
End-to-end orchestrator for the PHOENIX evaluation pipeline.

Stages:
    1. Parse the Qualtrics CSV into per-(case, hcp) canonical JSON.
       (Or, in pseudo mode, generate pseudo HCP outputs.)
    2. Load PHOENIX outputs from data/03_system/system_outputs.json.
       (Or, in pseudo mode, generate pseudo PHOENIX outputs.)
    3. Run the signed LLM judge (real OpenRouter call, or pseudo).
    4. Run the per-part signed LMM/TOST analyses.
    5. Run the cross-part synthesis.

Usage:
    python -m evaluation.survey_analysis.pipeline --mode pseudo
    python -m evaluation.survey_analysis.pipeline --mode real \\
        --n-runs 5 --cases C01 C02 --parts part1 part5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure the survey_analysis folder is importable when this script is
# launched directly (e.g. by run_pipeline.sh).
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from analysis import (  # noqa: E402  (import after sys.path tweak)
    part1_operationalization,
    part2_initial_model,
    part3_treatment_targets,
    part4_updated_model,
    part5_intervention,
    synthesis,
)
from analysis.shared.survey_paths import (  # noqa: E402
    JUDGMENTS_DIR,
    PARSED_DIR,
    PSEUDODATA_DIR,
    QUALTRICS_RAW_CSV,
    SYSTEM_DIR,
    ensure_data_dirs,
    judgments_csv,
)
from llm_as_judge.dimensions import DIMENSIONS_BY_PART  # noqa: E402
from llm_as_judge.judge_runner import JudgeRunConfig, run_judge  # noqa: E402
from parsing.qualtrics_parser import parse_qualtrics_csv, save_parsed_outputs  # noqa: E402
from parsing.system_output_loader import load_system_outputs  # noqa: E402
from parsing.case_context_loader import (  # noqa: E402
    DEFAULT_CASE_CONTEXTS_PATH,
    load_case_contexts,
    make_case_context_provider,
)
from pseudodata import (  # noqa: E402
    CASE_IDS,
    generate_case_contexts,
    generate_hcp_outputs,
    generate_phoenix_outputs,
    generate_pseudo_judgments,
)

logger = logging.getLogger("phoenix.pipeline")


PART_RUNNERS = {
    "part1": part1_operationalization.run,
    "part2": part2_initial_model.run,
    "part3": part3_treatment_targets.run,
    "part4": part4_updated_model.run,
    "part5": part5_intervention.run,
}


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evaluation/survey_analysis/pipeline.py",
        description="Run the PHOENIX evaluation pipeline end-to-end.",
    )
    p.add_argument(
        "--mode", choices=["pseudo", "real"], default="pseudo",
        help="Use pseudo HCP/PHOENIX/judge data (default) or call the real "
             "LLM judge (requires OPENROUTER_API_KEY).",
    )
    p.add_argument(
        "--judge", choices=["openrouter", "pseudo"], default=None,
        help="Force a judge backend; defaults to match --mode.",
    )
    p.add_argument(
        "--n-runs", type=int, default=5,
        help="Number of stochastic judge runs per (case, part).",
    )
    p.add_argument(
        "--cases", nargs="+", default=list(CASE_IDS),
        help="Subset of case ids to evaluate (default: C01..C10).",
    )
    p.add_argument(
        "--parts", nargs="+", default=list(DIMENSIONS_BY_PART.keys()),
        help="Subset of parts to evaluate (default: all five).",
    )
    p.add_argument(
        "--qualtrics-csv", type=Path, default=QUALTRICS_RAW_CSV,
        help="Path to the Qualtrics raw CSV (real mode only).",
    )
    p.add_argument(
        "--case-contexts", type=Path, default=DEFAULT_CASE_CONTEXTS_PATH,
        help="Path to per-case context JSON shown to both sources in judge prompts.",
    )
    p.add_argument(
        "--skip-parse", action="store_true",
        help="Skip the Qualtrics parsing stage (assume parsed JSON is fresh).",
    )
    p.add_argument(
        "--skip-judge", action="store_true",
        help="Skip the judging stage (re-use existing judgments_long.csv).",
    )
    p.add_argument(
        "--skip-analysis", action="store_true",
        help="Skip the per-part LMM analyses.",
    )
    p.add_argument(
        "--skip-synthesis", action="store_true",
        help="Skip the cross-part synthesis.",
    )
    p.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"],
    )
    return p


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: parsing / pseudo HCP outputs
# ──────────────────────────────────────────────────────────────────────────────

def _stage_parse(args: argparse.Namespace) -> Dict[str, Any]:
    """
    In real mode: parse Qualtrics CSV.
    In pseudo mode: generate pseudo HCP outputs.

    Always writes the canonical per-case file to ``data/02_parsed/hcp_outputs.json``.
    """
    if args.mode == "pseudo":
        out_path = generate_hcp_outputs()
        # Mirror to the parsed dir so downstream stages have one canonical
        # location regardless of mode.
        target = PARSED_DIR / "hcp_outputs.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(out_path.read_bytes())
        logger.info("Pseudo HCP outputs written to %s", target)
        return {"hcp_outputs": json.loads(target.read_text(encoding="utf-8")),
                "path": target}

    if not args.qualtrics_csv.exists():
        raise FileNotFoundError(
            f"Qualtrics CSV not found at {args.qualtrics_csv}. "
            "Pass --qualtrics-csv or run in --mode pseudo."
        )
    responses = parse_qualtrics_csv(args.qualtrics_csv)
    paths = save_parsed_outputs(responses, PARSED_DIR)
    logger.info("Parsed %d HCP submissions; canonical -> %s",
                len(responses), paths["hcp_outputs"])
    return {
        "hcp_outputs": json.loads(paths["hcp_outputs"].read_text(encoding="utf-8")),
        "path": paths["hcp_outputs"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: system outputs
# ──────────────────────────────────────────────────────────────────────────────

def _stage_system_outputs(args: argparse.Namespace) -> Dict[str, Any]:
    """In pseudo mode generate; in real mode load from data/03_system/."""
    if args.mode == "pseudo":
        ph = generate_phoenix_outputs()
        target = SYSTEM_DIR / "system_outputs.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(ph.read_bytes())
        logger.info("Pseudo PHOENIX outputs written to %s", target)
    else:
        target = SYSTEM_DIR / "system_outputs.json"
        if not target.exists():
            raise FileNotFoundError(
                f"PHOENIX outputs not found at {target}. "
                "Place the production system outputs there or use --mode pseudo."
            )
    bundle = load_system_outputs(target)
    return {"system_outputs": bundle.by_case, "path": target}


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3: judge
# ──────────────────────────────────────────────────────────────────────────────

def _stage_judge(
    args: argparse.Namespace,
    hcp_outputs: Dict[str, Any],
    system_outputs: Dict[str, Any],
) -> Path:
    judge_backend = args.judge or ("openrouter" if args.mode == "real" else "pseudo")

    if judge_backend == "pseudo":
        # Reuse the standalone pseudo generator so the same CSV is built.
        # Persist pseudo HCP / PHOENIX bundles to PSEUDODATA_DIR for inspection.
        (PSEUDODATA_DIR / "hcp_outputs.json").write_text(
            json.dumps(hcp_outputs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (PSEUDODATA_DIR / "phoenix_outputs.json").write_text(
            json.dumps(system_outputs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        generate_case_contexts(PSEUDODATA_DIR / "case_contexts.json")
        return generate_pseudo_judgments(
            n_runs=args.n_runs,
            cases=args.cases,
            parts=args.parts,
        )

    # Real LLM judge.
    contexts = load_case_contexts(args.case_contexts)
    if not contexts:
        logger.warning(
            "No case contexts found at %s; real judge prompts will contain "
            "explicit not-provided placeholders.",
            args.case_contexts,
        )
    config = JudgeRunConfig(
        cases=args.cases,
        parts=args.parts,
        n_runs=args.n_runs,
        mode="real",
        case_context_provider=make_case_context_provider(contexts),
    )
    return run_judge(
        hcp_outputs=hcp_outputs,
        system_outputs=system_outputs,
        config=config,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Stage 4 + 5: analysis
# ──────────────────────────────────────────────────────────────────────────────

def _stage_part_analysis(parts: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for part in parts:
        runner = PART_RUNNERS.get(part)
        if runner is None:
            logger.warning("Unknown part %s; skipping.", part)
            continue
        logger.info("Running per-part analysis for %s", part)
        out[part] = runner()
    return out


def _stage_synthesis() -> Dict[str, Any]:
    logger.info("Running cross-part synthesis")
    return synthesis.run()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    ensure_data_dirs()

    logger.info("Stage 0/5: ensuring data directories")
    logger.info("Stage 1/5: %s", "generating pseudo HCP outputs"
                if args.mode == "pseudo" else "parsing Qualtrics CSV")
    if not args.skip_parse:
        parse_out = _stage_parse(args)
    else:
        path = PARSED_DIR / "hcp_outputs.json"
        parse_out = {"hcp_outputs": json.loads(path.read_text(encoding="utf-8"))}

    logger.info("Stage 2/5: %s", "generating pseudo PHOENIX outputs"
                if args.mode == "pseudo" else "loading system outputs")
    sys_out = _stage_system_outputs(args)

    if not args.skip_judge:
        logger.info("Stage 3/5: judging")
        _stage_judge(args, parse_out["hcp_outputs"], sys_out["system_outputs"])
    else:
        logger.info("Stage 3/5: SKIPPED (re-using %s)", judgments_csv())

    if not args.skip_analysis:
        logger.info("Stage 4/5: per-part analyses")
        _stage_part_analysis(args.parts)
    else:
        logger.info("Stage 4/5: SKIPPED")

    if not args.skip_synthesis:
        logger.info("Stage 5/5: cross-part synthesis")
        _stage_synthesis()
    else:
        logger.info("Stage 5/5: SKIPPED")

    logger.info("Pipeline complete. Results under results/ and data/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
