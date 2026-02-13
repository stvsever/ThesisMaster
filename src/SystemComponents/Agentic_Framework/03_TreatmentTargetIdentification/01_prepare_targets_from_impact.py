#!/usr/bin/env python3
"""
01_prepare_targets_from_impact.py

Prepare treatment-target candidate inputs from momentary-impact outputs.

This script is intentionally a handoff/preparation step for Agentic Framework step 03.
It does not perform treatment planning itself.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def discover_profiles(impact_root: Path, pattern: str, max_profiles: int) -> List[Path]:
    dirs: List[Path] = []
    for child in sorted(impact_root.iterdir()):
        if not child.is_dir():
            continue
        if pattern and pattern not in child.name:
            continue
        if not (child / "predictor_composite.csv").exists():
            continue
        dirs.append(child)
    if max_profiles > 0:
        dirs = dirs[:max_profiles]
    return dirs


def priority_from_impact(score: float) -> str:
    if score >= 0.70:
        return "very_high"
    if score >= 0.50:
        return "high"
    if score >= 0.30:
        return "medium"
    if score >= 0.10:
        return "low"
    return "very_low"


def prepare_one_profile(
    profile_dir: Path,
    output_root: Path,
    top_k: int,
    min_impact: float,
) -> Dict[str, object]:
    profile_id = profile_dir.name
    source_csv = profile_dir / "predictor_composite.csv"
    df = pd.read_csv(source_csv)

    required = {"predictor", "predictor_label", "predictor_impact", "predictor_impact_pct", "predictor_rank"}
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{profile_id}: predictor_composite.csv missing required columns: {missing}")

    df = df.sort_values("predictor_impact", ascending=False).reset_index(drop=True)
    df = df[df["predictor_impact"] >= float(min_impact)].copy()
    if top_k > 0:
        df = df.head(top_k).copy()

    if df.empty:
        log(f"[WARN] {profile_id}: no predictors above min_impact={min_impact}")

    df["priority_level"] = df["predictor_impact"].map(priority_from_impact)
    df["selection_reason"] = "ranked_by_momentary_impact"
    df["source_profile"] = profile_id

    out_profile_dir = ensure_dir(output_root / profile_id)
    out_csv = out_profile_dir / "top_treatment_target_candidates.csv"
    out_json = out_profile_dir / "top_treatment_target_candidates.json"

    keep_cols = [
        "source_profile",
        "predictor_rank",
        "predictor",
        "predictor_label",
        "predictor_impact",
        "predictor_impact_pct",
        "priority_level",
        "selection_reason",
    ]
    df[keep_cols].to_csv(out_csv, index=False)

    payload = {
        "profile_id": profile_id,
        "generated_at_local": ts(),
        "source_predictor_composite_csv": str(source_csv),
        "top_k": int(top_k),
        "min_impact": float(min_impact),
        "n_candidates": int(len(df)),
        "candidates": df[keep_cols].to_dict(orient="records"),
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return {
        "profile_id": profile_id,
        "n_candidates": int(len(df)),
        "csv": str(out_csv),
        "json": str(out_json),
    }


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    default_impact_root = (
        repo_root / "Evaluation/04_initial_observation_analysis/02_momentary_impact_coefficients"
    )
    default_output_root = repo_root / "Evaluation/05_treatment_target_handoff"

    parser = argparse.ArgumentParser(
        description="Prepare treatment-target candidate tables from momentary impact outputs."
    )
    parser.add_argument("--impact-root", type=str, default=str(default_impact_root))
    parser.add_argument("--output-root", type=str, default=str(default_output_root))
    parser.add_argument("--pattern", type=str, default="pseudoprofile_FTC_")
    parser.add_argument("--max-profiles", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--min-impact", type=float, default=0.10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    impact_root = Path(args.impact_root).expanduser().resolve()
    output_root = ensure_dir(Path(args.output_root).expanduser().resolve())

    if not impact_root.exists():
        log(f"[ERROR] impact-root not found: {impact_root}")
        return 2

    profiles = discover_profiles(
        impact_root=impact_root,
        pattern=str(args.pattern or "").strip(),
        max_profiles=int(args.max_profiles),
    )
    if not profiles:
        log(f"[ERROR] no profile folders found in {impact_root} with pattern={args.pattern!r}")
        return 3

    log("========== TREATMENT-TARGET HANDOFF PREP START ==========")
    log(f"impact_root: {impact_root}")
    log(f"output_root: {output_root}")
    log(f"profiles:    {len(profiles)}")
    log(f"top_k:       {int(args.top_k)}")
    log(f"min_impact:  {float(args.min_impact):.4f}")

    rows: List[Dict[str, object]] = []
    failed = 0
    for profile_dir in profiles:
        try:
            row = prepare_one_profile(
                profile_dir=profile_dir,
                output_root=output_root,
                top_k=int(args.top_k),
                min_impact=float(args.min_impact),
            )
            rows.append(row)
            log(f"[OK] {profile_dir.name}: {row['n_candidates']} candidates")
        except Exception as exc:
            failed += 1
            log(f"[ERROR] {profile_dir.name}: {repr(exc)}")

    summary = {
        "generated_at_local": ts(),
        "impact_root": str(impact_root),
        "output_root": str(output_root),
        "pattern": str(args.pattern),
        "top_k": int(args.top_k),
        "min_impact": float(args.min_impact),
        "n_profiles_attempted": len(profiles),
        "n_profiles_success": len(rows),
        "n_profiles_failed": failed,
        "profiles": rows,
    }
    (output_root / "handoff_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if rows:
        merged = pd.DataFrame(rows)
        merged.to_csv(output_root / "handoff_profiles.csv", index=False)

    log("========== TREATMENT-TARGET HANDOFF PREP COMPLETE ==========")
    log(f"success={len(rows)} failed={failed}")
    return 0 if failed == 0 else 4


if __name__ == "__main__":
    raise SystemExit(main())
