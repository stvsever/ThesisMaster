from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd

from .target_refinement import normalize_path_text, path_segments, path_similarity


def load_predictor_feasibility_table(table_path: Path) -> pd.DataFrame:
    if not table_path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(table_path)
    except Exception:
        return pd.DataFrame()
    if frame.empty:
        return frame
    if "path_str" not in frame.columns:
        if "full_path" in frame.columns:
            frame = frame.rename(columns={"full_path": "path_str"})
    if "overall_suitability" not in frame.columns:
        candidates = [col for col in frame.columns if "overall_suitability" in str(col)]
        if candidates:
            frame = frame.rename(columns={candidates[0]: "overall_suitability"})
    if "path_str" not in frame.columns:
        return pd.DataFrame()
    frame["path_norm"] = frame["path_str"].astype(str).apply(normalize_path_text)
    frame["path_norm"] = frame["path_norm"].fillna("")
    frame = frame.loc[frame["path_norm"].astype(str).str.len() > 0].copy()
    if frame.empty:
        return frame
    if "overall_suitability" in frame.columns:
        frame["overall_suitability"] = pd.to_numeric(frame["overall_suitability"], errors="coerce").fillna(0.0)
    else:
        frame["overall_suitability"] = 0.0
    if "risk.scientific_utility" in frame.columns:
        frame["risk.scientific_utility"] = pd.to_numeric(frame["risk.scientific_utility"], errors="coerce").fillna(0.0)
    else:
        frame["risk.scientific_utility"] = 0.0
    return frame


def path_to_parent_domain(path_value: str, levels: int = 2) -> str:
    segments = path_segments(path_value)
    if not segments:
        return ""
    if len(segments) <= int(levels):
        return " / ".join(segments)
    return " / ".join(segments[: int(levels)])


def build_parent_domain_scores(
    feasibility_frame: pd.DataFrame,
    *,
    parent_levels: int = 2,
) -> pd.DataFrame:
    if feasibility_frame.empty:
        return pd.DataFrame()
    frame = feasibility_frame.copy()
    frame["parent_domain"] = frame["path_norm"].astype(str).apply(lambda x: path_to_parent_domain(x, levels=parent_levels))
    frame = frame.loc[frame["parent_domain"].astype(str).str.len() > 0].copy()
    if frame.empty:
        return frame
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
    return grouped


def match_predictors_to_parent_feasibility(
    predictor_paths: Iterable[str],
    feasibility_frame: pd.DataFrame,
    *,
    top_k_parent_domains: int = 30,
    parent_levels: int = 2,
) -> List[Dict[str, Any]]:
    if feasibility_frame.empty:
        return []
    parent_table = build_parent_domain_scores(feasibility_frame, parent_levels=parent_levels)
    if parent_table.empty:
        return []
    parent_rows = parent_table.to_dict(orient="records")

    rows: List[Dict[str, Any]] = []
    for predictor_path in predictor_paths:
        normalized_predictor = normalize_path_text(str(predictor_path or ""))
        if not normalized_predictor:
            continue
        local: List[Dict[str, Any]] = []
        for parent in parent_rows:
            parent_domain = str(parent.get("parent_domain") or "")
            if not parent_domain:
                continue
            similarity = path_similarity(normalized_predictor, parent_domain)
            if similarity < 0.10:
                continue
            suitability = float(parent.get("mean_overall_suitability", 0.0))
            utility_risk = float(parent.get("mean_scientific_utility_risk", 0.0))
            blended = max(0.0, min(1.0, 0.75 * suitability + 0.25 * (1.0 - utility_risk)))
            local.append(
                {
                    "predictor_path": normalized_predictor,
                    "parent_domain": parent_domain,
                    "path_similarity_0_1": float(similarity),
                    "parent_mean_overall_suitability_0_1": float(suitability),
                    "parent_mean_scientific_utility_risk_0_1": float(utility_risk),
                    "blended_parent_feasibility_0_1": float(max(0.0, min(1.0, blended * similarity))),
                    "n_parent_predictors": int(parent.get("n_predictors", 0)),
                }
            )
        local = sorted(local, key=lambda row: float(row["blended_parent_feasibility_0_1"]), reverse=True)
        rows.extend(local[: max(1, int(top_k_parent_domains))])
    rows = sorted(rows, key=lambda row: float(row["blended_parent_feasibility_0_1"]), reverse=True)
    return rows


def top_parent_domains_for_bundle(
    predictor_paths: Sequence[str],
    feasibility_frame: pd.DataFrame,
    *,
    top_k: int = 30,
    per_predictor_k: int = 30,
    parent_levels: int = 2,
) -> List[Dict[str, Any]]:
    matches = match_predictors_to_parent_feasibility(
        predictor_paths=predictor_paths,
        feasibility_frame=feasibility_frame,
        top_k_parent_domains=per_predictor_k,
        parent_levels=parent_levels,
    )
    if not matches:
        return []
    aggregate: Dict[str, Dict[str, Any]] = {}
    for row in matches:
        parent_domain = str(row.get("parent_domain") or "")
        if not parent_domain:
            continue
        entry = aggregate.setdefault(
            parent_domain,
            {
                "parent_domain": parent_domain,
                "parent_feasibility_score_0_1": 0.0,
                "supporting_predictors": set(),
                "mean_overall_suitability_0_1": 0.0,
                "mean_scientific_utility_risk_0_1": 0.0,
                "support_count": 0,
            },
        )
        entry["parent_feasibility_score_0_1"] = max(
            float(entry["parent_feasibility_score_0_1"]),
            float(row.get("blended_parent_feasibility_0_1", 0.0)),
        )
        entry["mean_overall_suitability_0_1"] = max(
            float(entry["mean_overall_suitability_0_1"]),
            float(row.get("parent_mean_overall_suitability_0_1", 0.0)),
        )
        entry["mean_scientific_utility_risk_0_1"] = max(
            float(entry["mean_scientific_utility_risk_0_1"]),
            float(row.get("parent_mean_scientific_utility_risk_0_1", 0.0)),
        )
        entry["supporting_predictors"].add(str(row.get("predictor_path") or ""))
        entry["support_count"] = int(entry["support_count"]) + 1
    output: List[Dict[str, Any]] = []
    for row in aggregate.values():
        output.append(
            {
                "parent_domain": row["parent_domain"],
                "parent_feasibility_score_0_1": float(row["parent_feasibility_score_0_1"]),
                "mean_overall_suitability_0_1": float(row["mean_overall_suitability_0_1"]),
                "mean_scientific_utility_risk_0_1": float(row["mean_scientific_utility_risk_0_1"]),
                "supporting_predictors": sorted([p for p in row["supporting_predictors"] if p]),
                "support_count": int(row["support_count"]),
            }
        )
    output = sorted(
        output,
        key=lambda row: (
            float(row["parent_feasibility_score_0_1"]),
            float(row["mean_overall_suitability_0_1"]),
            -float(row["mean_scientific_utility_risk_0_1"]),
            int(row["support_count"]),
        ),
        reverse=True,
    )
    return output[: max(1, int(top_k))]

