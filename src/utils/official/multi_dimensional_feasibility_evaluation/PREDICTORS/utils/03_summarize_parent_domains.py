#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _normalize_path(raw: str) -> str:
    text = str(raw or "").strip().replace("|", "/").replace("\\", "/")
    text = " / ".join([part.strip() for part in text.split("/") if part.strip()])
    return text


def _parent_domain(path_value: str, levels: int = 2) -> str:
    parts = [part.strip() for part in _normalize_path(path_value).split(" / ") if part.strip()]
    if not parts:
        return ""
    if len(parts) <= levels:
        return " / ".join(parts)
    return " / ".join(parts[:levels])


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[6]
    default_input = (
        repo_root
        / "src/utils/official/multi_dimensional_feasibility_evaluation/PREDICTORS/results/summary/predictor_rankings.csv"
    )
    default_output = (
        repo_root
        / "src/utils/official/multi_dimensional_feasibility_evaluation/PREDICTORS/results/summary/predictor_parent_domain_rankings.csv"
    )
    parser = argparse.ArgumentParser(description="Aggregate predictor feasibility scores at parent-domain level.")
    parser.add_argument("--input-csv", type=str, default=str(default_input))
    parser.add_argument("--output-csv", type=str, default=str(default_output))
    parser.add_argument("--levels", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_csv = Path(args.input_csv).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    if not input_csv.exists():
        print(f"[ERROR] input not found: {input_csv}")
        return 2
    frame = pd.read_csv(input_csv)
    if frame.empty:
        print(f"[ERROR] empty input table: {input_csv}")
        return 3
    if "path_str" not in frame.columns:
        print("[ERROR] missing required column: path_str")
        return 4
    if "overall_suitability" not in frame.columns:
        print("[ERROR] missing required column: overall_suitability")
        return 4

    frame["path_norm"] = frame["path_str"].astype(str).apply(_normalize_path)
    frame["parent_domain"] = frame["path_norm"].astype(str).apply(lambda x: _parent_domain(x, levels=max(1, int(args.levels))))
    frame = frame.loc[frame["parent_domain"].astype(str).str.len() > 0].copy()
    frame["overall_suitability"] = pd.to_numeric(frame["overall_suitability"], errors="coerce").fillna(0.0)
    if "risk.scientific_utility" not in frame.columns:
        frame["risk.scientific_utility"] = 0.0
    frame["risk.scientific_utility"] = pd.to_numeric(frame["risk.scientific_utility"], errors="coerce").fillna(0.0)

    grouped = (
        frame.groupby("parent_domain", as_index=False)
        .agg(
            n_predictors=("path_norm", "count"),
            mean_overall_suitability=("overall_suitability", "mean"),
            max_overall_suitability=("overall_suitability", "max"),
            mean_scientific_utility_risk=("risk.scientific_utility", "mean"),
        )
        .sort_values(["mean_overall_suitability", "n_predictors"], ascending=[False, False])
        .reset_index(drop=True)
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output_csv, index=False)
    print(f"[OK] wrote: {output_csv} rows={len(grouped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
