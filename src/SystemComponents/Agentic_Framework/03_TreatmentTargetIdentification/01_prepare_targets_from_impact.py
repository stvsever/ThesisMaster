#!/usr/bin/env python3
"""
01_prepare_targets_from_impact.py

Agentic Framework Step-03: treatment-target identification.

This module consumes integrated PHOENIX outputs (readiness, network analysis, momentary impact,
free-text profile context, and mapped initial observation model) and produces:
1) ranked candidate treatment targets,
2) a selected subset (typically 2-3, but can be 0/1 if evidence is weak),
3) a Step-04 updated observation-model suggestion with ontology-subtree guidance.

Outputs remain backward-compatible with prior handoff files:
- top_treatment_target_candidates.csv
- top_treatment_target_candidates.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

THIS_DIR = Path(__file__).resolve().parent
def _bootstrap_repo_root(start_dir: Path) -> Path:
    for candidate in [start_dir, *start_dir.parents]:
        marker = candidate / "src" / "utils" / "agentic_core" / "shared" / "__init__.py"
        if marker.exists():
            return candidate
    raise RuntimeError(f"Unable to locate repository root from {start_dir}")


REPO_ROOT = _bootstrap_repo_root(THIS_DIR)
AGENTIC_CORE_ROOT = REPO_ROOT / "src" / "utils" / "agentic_core"
if str(AGENTIC_CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENTIC_CORE_ROOT))

from shared import (
    PromptSection,
    StructuredLLMClient,
    best_path_match,
    build_bfs_candidates,
    decision_from_score,
    discover_latest_hyde_dense_profiles,
    fuse_updated_model_matrix,
    generate_updated_model_visuals,
    load_predictor_feasibility_table,
    load_impact_matrix,
    load_predictor_leaf_paths,
    load_profile_hyde_scores,
    load_profile_mapping_rows,
    load_prompt,
    normalize_score,
    normalize_path_text,
    pack_prompt_sections,
    render_prompt,
    top_parent_domains_for_bundle,
    weighted_composite,
)


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_read_csv(path: Path) -> pd.DataFrame:
    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            if df.shape[1] > 1 or sep == ",":
                return df
        except Exception:
            pass
    return pd.read_csv(path, engine="python")


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


def extract_profile_number(profile_id: str) -> Optional[str]:
    match = re.search(r"ID(\d{3})$", str(profile_id))
    if not match:
        return None
    return match.group(1)


def parse_profile_text_file(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    output: Dict[str, List[str]] = {}
    current_key: Optional[str] = None
    for raw in lines:
        line = raw.rstrip()
        if re.match(r"^pseudoprofile_[A-Za-z_]+_ID\d{3}\s*$", line):
            current_key = line.strip()
            output[current_key] = []
            continue
        if current_key is None:
            continue
        output[current_key].append(line)
    return {k: "\n".join(v).strip() for k, v in output.items()}


def profile_text_bundle(
    profile_id: str,
    complaints: Dict[str, str],
    person_profiles: Dict[str, str],
    context_profiles: Dict[str, str],
) -> Dict[str, str]:
    number = extract_profile_number(profile_id)
    if number is None:
        return {"complaint_text": "", "person_text": "", "context_text": ""}
    return {
        "complaint_text": complaints.get(f"pseudoprofile_FTC_ID{number}", ""),
        "person_text": person_profiles.get(f"pseudoprofile_person_ID{number}", ""),
        "context_text": context_profiles.get(f"pseudoprofile_context_ID{number}", ""),
    }


def find_latest_initial_model(
    runs_root: Path,
    profile_id: str,
    filename: str = "llm_observation_model_mapped.json",
) -> Optional[Path]:
    candidates = sorted(
        runs_root.glob(f"*/profiles/{profile_id}/{filename}"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
        reverse=True,
    )
    return candidates[0] if candidates else None


def token_set(text: str) -> set[str]:
    return {item.lower() for item in re.findall(r"[A-Za-z0-9_]+", text or "") if len(item) > 1}


def jaccard_similarity(a: str, b: str) -> float:
    ta = token_set(a)
    tb = token_set(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta.intersection(tb))
    union = len(ta.union(tb))
    if union <= 0:
        return 0.0
    return float(inter / union)


@dataclass
class PredictorLeaf:
    predictor_id: str
    full_path: str
    root_domain: str
    secondary_node: str
    leaf_label: str


def parse_predictor_catalog(path: Path) -> List[PredictorLeaf]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    root_domain = ""
    branch_stack: Dict[int, str] = {}
    out: List[PredictorLeaf] = []
    for raw in lines:
        if not raw.strip():
            continue
        bracket = re.match(r"^\[([A-Za-z0-9_]+)\]\s*$", raw.strip())
        if bracket:
            root_domain = bracket.group(1)
            branch_stack.clear()
            continue
        if "└─" not in raw:
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        depth = max(1, int(indent / 2) + 1)
        body = raw.split("└─", 1)[1].strip()
        id_match = re.search(r"\(ID:(\d+)\)\s*$", body)
        label = re.sub(r"\s*\(ID:\d+\)\s*$", "", body).strip()
        for level in list(branch_stack.keys()):
            if level >= depth:
                branch_stack.pop(level, None)
        if id_match is None:
            branch_stack[depth] = label
            continue
        lineage = [root_domain]
        for level in sorted(branch_stack.keys()):
            if level < depth:
                lineage.append(branch_stack[level])
        lineage.append(label)
        full_path = " / ".join([item for item in lineage if item])
        out.append(
            PredictorLeaf(
                predictor_id=f"ID{id_match.group(1)}",
                full_path=full_path,
                root_domain=root_domain,
                secondary_node=lineage[1] if len(lineage) > 2 else root_domain,
                leaf_label=label,
            )
        )
    return out


def summarize_readiness(readiness_payload: Dict[str, Any]) -> Dict[str, Any]:
    overall = readiness_payload.get("overall", {}) or {}
    score_breakdown = overall.get("score_breakdown", {}) or {}
    breakdown_components = (score_breakdown.get("components", {}) or {}) if isinstance(score_breakdown, dict) else {}
    tiers = readiness_payload.get("tiers", {}) or {}
    tier3 = tiers.get("tier3", {}) or {}
    tv_variant = tier3.get("variant_time_varying", {}) or {}
    static_variant = tier3.get("variant_static", {}) or {}
    return {
        "label": overall.get("readiness_label"),
        "score_0_100": overall.get("readiness_score_0_100"),
        "recommended_tier": overall.get("recommended_tier"),
        "tier3_variant": overall.get("tier3_variant"),
        "ready_variables": list(overall.get("ready_variables", []) or []),
        "missingness_component_score": breakdown_components.get("missing_score"),
        "time_component_score": breakdown_components.get("time_score"),
        "sample_component_score": breakdown_components.get("sample_score"),
        "quality_component_score": breakdown_components.get("variable_quality_score"),
        "tv_feasible": tv_variant.get("feasible"),
        "tv_required_points_heuristic": tv_variant.get("tv_required_points_heuristic"),
        "tv_effective_n_q25": ((tv_variant.get("n_eff_lagged", {}) or {}).get("q25_per_variable")),
        "static_feasible": static_variant.get("feasible"),
        "static_required_n": static_variant.get("required_n_eff_heuristic"),
        "static_effective_n_q25": ((static_variant.get("n_eff_lagged", {}) or {}).get("q25_per_variable")),
        "why": list(overall.get("why", []) or []),
        "technical_summary": overall.get("technical_summary"),
        "client_friendly_summary": overall.get("client_friendly_summary"),
        "next_steps": list(overall.get("next_steps", []) or []),
        "caveats": list(overall.get("caveats", []) or []),
    }


def summarize_network_metrics(network_profile_root: Path) -> Dict[str, Any]:
    metrics_root = network_profile_root / "network_metrics"
    if not metrics_root.exists():
        return {}
    summary: Dict[str, Any] = {}
    predictor_importance = metrics_root / "predictor_importance_tv.csv"
    if predictor_importance.exists():
        df = safe_read_csv(predictor_importance)
        if "out_strength_criteria_mean" in df.columns:
            df = df.sort_values("out_strength_criteria_mean", ascending=False)
        summary["top_predictor_importance_rows"] = df.head(12).fillna("").to_dict(orient="records")
    criterion_dependence = metrics_root / "criterion_dependence_tv.csv"
    if criterion_dependence.exists():
        df = safe_read_csv(criterion_dependence)
        summary["criterion_dependence_top"] = df.head(12).fillna("").to_dict(orient="records")
    lagged_global = metrics_root / "temporal_lagged_global_metrics.csv"
    if lagged_global.exists():
        df = safe_read_csv(lagged_global)
        summary["temporal_lagged_global_metrics"] = df.head(20).fillna("").to_dict(orient="records")
    contemp_global = metrics_root / "temporal_contemp_global_metrics.csv"
    if contemp_global.exists():
        df = safe_read_csv(contemp_global)
        summary["temporal_contemp_global_metrics"] = df.head(20).fillna("").to_dict(orient="records")
    multicollinearity = metrics_root / "multicollinearity_all.json"
    if multicollinearity.exists():
        summary["multicollinearity_all"] = read_json(multicollinearity)
    return summary


def summarize_initial_model(initial_model_payload: Dict[str, Any]) -> Dict[str, Any]:
    predictors = initial_model_payload.get("predictor_variables", []) or []
    criteria = initial_model_payload.get("criteria_variables", []) or []
    predictor_summary: List[Dict[str, Any]] = []
    for item in predictors:
        predictor_summary.append(
            {
                "var_id": item.get("var_id"),
                "label": item.get("label"),
                "include_priority": item.get("include_priority"),
                "mapped_leaf_full_path": item.get("mapped_leaf_full_path"),
                "mapped_confidence": item.get("mapped_confidence"),
                "bio_psycho_social_domain": item.get("bio_psycho_social_domain"),
                "targets_criteria_var_ids": item.get("targets_criteria_var_ids", []),
            }
        )
    criteria_summary: List[Dict[str, Any]] = []
    for item in criteria:
        criteria_summary.append(
            {
                "var_id": item.get("var_id"),
                "label": item.get("label"),
                "mapped_leaf_full_path": item.get("mapped_leaf_full_path"),
                "mapped_confidence": item.get("mapped_confidence"),
            }
        )
    relevance_edges = initial_model_payload.get("edges", []) or []
    relevance_edges_sorted = sorted(
        relevance_edges,
        key=lambda row: float(row.get("estimated_relevance_0_1", 0.0)),
        reverse=True,
    )
    return {
        "criteria_summary": criteria_summary,
        "predictor_summary": predictor_summary,
        "top_edges_predictor_to_criterion": relevance_edges_sorted[:30],
    }


def build_ontology_candidate_set(
    predictor_catalog: Sequence[PredictorLeaf],
    initial_model_summary: Dict[str, Any],
    impact_df: pd.DataFrame,
    max_candidates: int,
) -> List[Dict[str, Any]]:
    anchor_paths: List[str] = []
    impact_anchor: Dict[str, float] = {}

    for row in initial_model_summary.get("predictor_summary", []):
        mapped = str(row.get("mapped_leaf_full_path") or "").strip()
        if mapped:
            anchor_paths.append(mapped)

    impact_index = impact_df.set_index("predictor") if "predictor" in impact_df.columns else pd.DataFrame()
    for row in initial_model_summary.get("predictor_summary", []):
        pid = str(row.get("var_id") or "").strip()
        path = str(row.get("mapped_leaf_full_path") or "").strip()
        if not pid or not path:
            continue
        if not impact_index.empty and pid in impact_index.index and "predictor_impact" in impact_index.columns:
            impact_anchor[path] = float(impact_index.loc[pid, "predictor_impact"])

    ranked: List[Dict[str, Any]] = []
    for leaf in predictor_catalog:
        similarities = [jaccard_similarity(leaf.full_path, anchor) for anchor in anchor_paths]
        best_similarity = max(similarities) if similarities else 0.0
        secondary_match = 1.0 if any(
            leaf.secondary_node.lower() in str(anchor).lower() for anchor in anchor_paths
        ) else 0.0
        anchor_weight = max(
            [
                impact_anchor.get(anchor, 0.0) * jaccard_similarity(leaf.full_path, anchor)
                for anchor in anchor_paths
            ]
            or [0.0]
        )
        score = 0.55 * best_similarity + 0.25 * secondary_match + 0.20 * anchor_weight
        ranked.append(
            {
                "predictor_path": leaf.full_path,
                "predictor_id": leaf.predictor_id,
                "root_domain": leaf.root_domain,
                "secondary_node": leaf.secondary_node,
                "leaf_label": leaf.leaf_label,
                "subtree_relevance_score_0_1": round(float(min(1.0, max(0.0, score))), 6),
            }
        )
    ranked = sorted(ranked, key=lambda row: row["subtree_relevance_score_0_1"], reverse=True)
    return ranked[: max(1, int(max_candidates))]


class RankedPredictorModel(BaseModel):
    predictor: str
    predictor_label: str
    score_0_1: float = Field(ge=0.0, le=1.0)
    rationale: str
    mapped_leaf_path: str = ""
    evidence_refs: List[str] = Field(default_factory=list)
    confidence_0_1: float = Field(default=0.5, ge=0.0, le=1.0)


class Step03SelectionModel(BaseModel):
    contract_version: str = "1.0.0"
    profile_id: str
    recommended_targets: List[RankedPredictorModel] = Field(default_factory=list)
    ranked_predictors: List[RankedPredictorModel]
    summary: str
    safety_considerations: List[str] = Field(default_factory=list)
    data_limitations: List[str] = Field(default_factory=list)


class UpdatedPredictorCandidateModel(BaseModel):
    predictor_path: str
    score_0_1: float = Field(ge=0.0, le=1.0)
    source: str
    reason: str


class Step04UpdatedObservationModel(BaseModel):
    contract_version: str = "1.0.0"
    profile_id: str
    retained_criteria_ids: List[str]
    refined_predictor_shortlist: List[UpdatedPredictorCandidateModel]
    recommended_next_observation_predictors: List[str]
    dropped_predictors: List[str]
    added_predictors: List[str]
    rationale: str


class StageCriticReviewModel(BaseModel):
    contract_version: str = "1.0.0"
    profile_id: str
    stage: str
    pass_decision: str
    composite_score_0_1: float = Field(ge=0.0, le=1.0)
    weighted_subscores_0_1: Dict[str, float] = Field(default_factory=dict)
    critical_issues: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    feedback_for_revision: List[str] = Field(default_factory=list)
    evidence_gaps: List[str] = Field(default_factory=list)
    confidence_0_1: float = Field(default=0.5, ge=0.0, le=1.0)


def _enforce_step03_hard_ontology(
    step03_output: Step03SelectionModel,
    *,
    allowed_predictor_paths: Sequence[str],
    predictor_var_to_path: Dict[str, str],
) -> Dict[str, Any]:
    allowed = [normalize_path_text(path) for path in allowed_predictor_paths if normalize_path_text(path)]
    if not allowed:
        return {"applied": False, "reason": "no_allowed_paths"}
    changes = 0
    dropped: List[str] = []
    for item in step03_output.ranked_predictors:
        inferred_path = normalize_path_text(
            str(
                item.mapped_leaf_path
                or predictor_var_to_path.get(str(item.predictor), "")
                or item.predictor_label
                or item.predictor
            )
        )
        mapped_path, similarity = best_path_match(inferred_path, allowed)
        if not mapped_path or similarity < 0.35:
            dropped.append(str(item.predictor))
            continue
        if normalize_path_text(item.mapped_leaf_path) != mapped_path:
            item.mapped_leaf_path = mapped_path
            changes += 1
    if dropped:
        step03_output.ranked_predictors = [
            item for item in step03_output.ranked_predictors if str(item.predictor) not in set(dropped)
        ]
        step03_output.recommended_targets = [
            item for item in step03_output.recommended_targets if str(item.predictor) not in set(dropped)
        ]
        changes += len(dropped)
    for item in step03_output.recommended_targets:
        inferred_path = normalize_path_text(
            str(
                item.mapped_leaf_path
                or predictor_var_to_path.get(str(item.predictor), "")
                or item.predictor_label
                or item.predictor
            )
        )
        mapped_path, similarity = best_path_match(inferred_path, allowed)
        if mapped_path and similarity >= 0.35:
            if normalize_path_text(item.mapped_leaf_path) != mapped_path:
                item.mapped_leaf_path = mapped_path
                changes += 1
    step03_output.ranked_predictors = step03_output.ranked_predictors[: max(1, len(step03_output.ranked_predictors))]
    step03_output.recommended_targets = step03_output.recommended_targets[:3]
    return {"applied": True, "changed_count": int(changes), "dropped_predictors": dropped}


def _enforce_step04_hard_ontology(
    step04_output: Step04UpdatedObservationModel,
    *,
    allowed_predictor_paths: Sequence[str],
) -> Dict[str, Any]:
    allowed = [normalize_path_text(path) for path in allowed_predictor_paths if normalize_path_text(path)]
    if not allowed:
        return {"applied": False, "reason": "no_allowed_paths"}
    changes = 0
    unmatched: List[str] = []
    new_shortlist: List[UpdatedPredictorCandidateModel] = []
    for row in step04_output.refined_predictor_shortlist:
        candidate = normalize_path_text(row.predictor_path)
        matched, similarity = best_path_match(candidate, allowed)
        if not matched or similarity < 0.35:
            unmatched.append(candidate)
            continue
        if matched != candidate:
            changes += 1
        new_shortlist.append(
            UpdatedPredictorCandidateModel(
                predictor_path=matched,
                score_0_1=normalize_score(row.score_0_1),
                source=row.source,
                reason=row.reason,
            )
        )
    step04_output.refined_predictor_shortlist = new_shortlist
    remapped_next: List[str] = []
    seen = set()
    for predictor_path in step04_output.recommended_next_observation_predictors:
        candidate = normalize_path_text(str(predictor_path))
        matched, similarity = best_path_match(candidate, allowed)
        if not matched or similarity < 0.35:
            unmatched.append(candidate)
            continue
        if matched != candidate:
            changes += 1
        if matched not in seen:
            seen.add(matched)
            remapped_next.append(matched)
    step04_output.recommended_next_observation_predictors = remapped_next
    step04_output.dropped_predictors = [normalize_path_text(item) for item in step04_output.dropped_predictors if normalize_path_text(item)]
    step04_output.added_predictors = [normalize_path_text(item) for item in step04_output.added_predictors if normalize_path_text(item)]
    return {"applied": True, "changed_count": int(changes), "unmatched_paths": unmatched[:50]}


def heuristic_step03_selection(
    profile_id: str,
    impact_df: pd.DataFrame,
    mapped_predictor_paths: Dict[str, str],
    top_k: int,
    min_impact: float,
) -> Step03SelectionModel:
    local = impact_df.copy()
    if "predictor_impact" not in local.columns:
        local["predictor_impact"] = 0.0
    local = local.sort_values("predictor_impact", ascending=False).reset_index(drop=True)
    if top_k > 0:
        local = local.head(int(top_k)).copy()
    ranked: List[RankedPredictorModel] = []
    for _, row in local.iterrows():
        impact = float(row.get("predictor_impact", 0.0))
        predictor = str(row.get("predictor", ""))
        label = str(row.get("predictor_label", predictor))
        ranked.append(
            RankedPredictorModel(
                predictor=predictor,
                predictor_label=label,
                score_0_1=float(min(1.0, max(0.0, impact))),
                rationale="Heuristic fallback ranking based on momentary impact composite.",
                mapped_leaf_path=mapped_predictor_paths.get(predictor, ""),
                evidence_refs=["momentary_impact.predictor_composite"],
                confidence_0_1=0.55,
            )
        )

    recommended = [row for row in ranked if row.score_0_1 >= float(min_impact)][:3]
    if not recommended and ranked and ranked[0].score_0_1 >= 0.25:
        recommended = ranked[:1]

    return Step03SelectionModel(
        profile_id=profile_id,
        recommended_targets=recommended,
        ranked_predictors=ranked,
        summary="Heuristic fallback used because no valid structured LLM output was available.",
        safety_considerations=["Use outputs as decision support, not as diagnosis."],
        data_limitations=["Limitation recorded: LLM unavailable, so it’s impact-driven only."],
    )


def heuristic_step04_update(
    profile_id: str,
    selected_step03: Step03SelectionModel,
    initial_model_summary: Dict[str, Any],
    ontology_candidates: Sequence[Dict[str, Any]],
    shortlist_limit: int,
) -> Step04UpdatedObservationModel:
    criteria_ids = [
        str(item.get("var_id"))
        for item in initial_model_summary.get("criteria_summary", [])
        if str(item.get("var_id") or "").strip()
    ]
    predictor_rows = list(initial_model_summary.get("predictor_summary", []))
    current_predictors = {
        normalize_path_text(str(item.get("mapped_leaf_full_path") or item.get("var_id") or ""))
        for item in predictor_rows
        if str(item.get("mapped_leaf_full_path") or item.get("var_id") or "").strip()
    }
    var_to_path = {
        str(item.get("var_id") or ""): normalize_path_text(str(item.get("mapped_leaf_full_path") or ""))
        for item in predictor_rows
        if str(item.get("var_id") or "").strip()
    }
    shortlist: List[UpdatedPredictorCandidateModel] = []
    for item in list(ontology_candidates)[: max(1, int(shortlist_limit))]:
        shortlist.append(
            UpdatedPredictorCandidateModel(
                predictor_path=normalize_path_text(str(item.get("predictor_path"))),
                score_0_1=float(item.get("subtree_relevance_score_0_1", 0.0)),
                source=str(item.get("bfs_stage") or "ontology_subtree_similarity"),
                reason=(
                    "Selected via breadth-first ontology expansion, constrained by profile-specific "
                    "mapping evidence and idiographic impact anchors."
                ),
            )
        )
    selected_paths = [
        normalize_path_text(var_to_path.get(item.predictor, item.mapped_leaf_path))
        for item in selected_step03.recommended_targets
    ]
    selected_paths = [item for item in selected_paths if item]
    shortlist_paths = [item.predictor_path for item in shortlist if item.predictor_path]
    merged_next = []
    seen_next = set()
    for candidate in selected_paths + shortlist_paths:
        if not candidate or candidate in seen_next:
            continue
        seen_next.add(candidate)
        merged_next.append(candidate)

    if not merged_next:
        fallback_paths = [
            normalize_path_text(var_to_path.get(item.predictor, item.mapped_leaf_path))
            for item in selected_step03.ranked_predictors[:5]
        ]
        merged_next = [item for item in fallback_paths if item]

    dropped = sorted(list(current_predictors.difference(set(merged_next))))
    added = sorted(list(set(merged_next).difference(current_predictors)))
    return Step04UpdatedObservationModel(
        profile_id=profile_id,
        retained_criteria_ids=criteria_ids,
        refined_predictor_shortlist=shortlist,
        recommended_next_observation_predictors=merged_next[:20],
        dropped_predictors=dropped,
        added_predictors=added,
        rationale=(
            "Heuristic update with breadth-first candidate expansion: retain criteria, enforce domain coverage "
            "before subtree deepening, and prioritize mapped sub-predictors aligned with idiographic evidence."
        ),
    )


def run_llm_step03(
    *,
    client: StructuredLLMClient,
    profile_id: str,
    evidence_bundle: Dict[str, Any],
    prompt_budget_tokens: int,
) -> Tuple[Optional[Step03SelectionModel], Dict[str, Any]]:
    system_template = load_prompt("step03_target_selection_system.md")
    user_template = load_prompt("step03_target_selection_user_template.md")

    sections = [
        PromptSection("meta", json.dumps(evidence_bundle.get("meta", {}), ensure_ascii=False, indent=2), priority=1),
        PromptSection("free_text", json.dumps(evidence_bundle.get("free_text", {}), ensure_ascii=False, indent=2), priority=2),
        PromptSection("readiness", json.dumps(evidence_bundle.get("readiness", {}), ensure_ascii=False, indent=2), priority=3),
        PromptSection("impact", json.dumps(evidence_bundle.get("impact", {}), ensure_ascii=False, indent=2), priority=4),
        PromptSection("network", json.dumps(evidence_bundle.get("network", {}), ensure_ascii=False, indent=2), priority=5),
        PromptSection("initial_model", json.dumps(evidence_bundle.get("initial_model", {}), ensure_ascii=False, indent=2), priority=6),
        PromptSection("mapping_evidence", json.dumps(evidence_bundle.get("mapping_evidence", {}), ensure_ascii=False, indent=2), priority=7),
        PromptSection("hyde_evidence", json.dumps(evidence_bundle.get("hyde_evidence", {}), ensure_ascii=False, indent=2), priority=8),
        PromptSection("bfs_planner", json.dumps(evidence_bundle.get("bfs_planner", {}), ensure_ascii=False, indent=2), priority=9),
        PromptSection("ontology_candidates", json.dumps(evidence_bundle.get("ontology_candidates", []), ensure_ascii=False, indent=2), priority=10),
        PromptSection("critic_feedback", json.dumps(evidence_bundle.get("critic_feedback", {}), ensure_ascii=False, indent=2), priority=11),
    ]
    packed = pack_prompt_sections(
        sections,
        max_tokens=int(prompt_budget_tokens),
        reserve_tokens=3000,
        model=client.model,
    )
    user_prompt = render_prompt(user_template, {"EVIDENCE_BUNDLE_JSON": packed.text})
    llm_result = client.generate_structured(
        system_prompt=system_template,
        user_prompt=user_prompt,
        schema_model=Step03SelectionModel,
    )
    trace = {
        "profile_id": profile_id,
        "packed_prompt_estimated_tokens": packed.estimated_tokens,
        "packed_prompt_max_tokens": packed.max_tokens,
        "included_sections": packed.included_sections,
        "truncated_sections": packed.truncated_sections,
        "section_token_estimates": packed.section_token_estimates,
        "provider": llm_result.provider,
        "model": llm_result.model,
        "success": llm_result.success,
        "used_repair": llm_result.used_repair,
        "validation_error": llm_result.validation_error,
        "failure_reason": llm_result.failure_reason,
        "usage": llm_result.usage,
    }
    if not llm_result.success or llm_result.parsed is None:
        return None, trace
    parsed = Step03SelectionModel.model_validate(llm_result.parsed)
    parsed.profile_id = profile_id
    parsed.recommended_targets = parsed.recommended_targets[:3]
    return parsed, trace


def run_llm_step04(
    *,
    client: StructuredLLMClient,
    profile_id: str,
    evidence_bundle: Dict[str, Any],
    prompt_budget_tokens: int,
) -> Tuple[Optional[Step04UpdatedObservationModel], Dict[str, Any]]:
    system_template = load_prompt("step04_observation_update_system.md")
    user_template = load_prompt("step04_observation_update_user_template.md")
    sections = [
        PromptSection("meta", json.dumps(evidence_bundle.get("meta", {}), ensure_ascii=False, indent=2), priority=1),
        PromptSection("step03_output", json.dumps(evidence_bundle.get("step03_output", {}), ensure_ascii=False, indent=2), priority=2),
        PromptSection("initial_model", json.dumps(evidence_bundle.get("initial_model", {}), ensure_ascii=False, indent=2), priority=3),
        PromptSection("readiness", json.dumps(evidence_bundle.get("readiness", {}), ensure_ascii=False, indent=2), priority=4),
        PromptSection("mapping_evidence", json.dumps(evidence_bundle.get("mapping_evidence", {}), ensure_ascii=False, indent=2), priority=5),
        PromptSection("bfs_planner", json.dumps(evidence_bundle.get("bfs_planner", {}), ensure_ascii=False, indent=2), priority=6),
        PromptSection("fusion_prior", json.dumps(evidence_bundle.get("fusion_prior", {}), ensure_ascii=False, indent=2), priority=7),
        PromptSection("ontology_candidates", json.dumps(evidence_bundle.get("ontology_candidates", []), ensure_ascii=False, indent=2), priority=8),
        PromptSection("critic_feedback", json.dumps(evidence_bundle.get("critic_feedback", {}), ensure_ascii=False, indent=2), priority=9),
    ]
    packed = pack_prompt_sections(
        sections,
        max_tokens=int(prompt_budget_tokens),
        reserve_tokens=3500,
        model=client.model,
    )
    user_prompt = render_prompt(user_template, {"EVIDENCE_BUNDLE_JSON": packed.text})
    llm_result = client.generate_structured(
        system_prompt=system_template,
        user_prompt=user_prompt,
        schema_model=Step04UpdatedObservationModel,
    )
    trace = {
        "profile_id": profile_id,
        "packed_prompt_estimated_tokens": packed.estimated_tokens,
        "packed_prompt_max_tokens": packed.max_tokens,
        "included_sections": packed.included_sections,
        "truncated_sections": packed.truncated_sections,
        "section_token_estimates": packed.section_token_estimates,
        "provider": llm_result.provider,
        "model": llm_result.model,
        "success": llm_result.success,
        "used_repair": llm_result.used_repair,
        "validation_error": llm_result.validation_error,
        "failure_reason": llm_result.failure_reason,
        "usage": llm_result.usage,
    }
    if not llm_result.success or llm_result.parsed is None:
        return None, trace
    parsed = Step04UpdatedObservationModel.model_validate(llm_result.parsed)
    parsed.profile_id = profile_id
    return parsed, trace


def _critic_weights(stage: str) -> Dict[str, float]:
    stage_key = str(stage).strip().lower()
    if stage_key == "step04":
        return {
            "predictor_grounding": 0.28,
            "criterion_continuity": 0.20,
            "bfs_depth_balance": 0.18,
            "fusion_consistency": 0.20,
            "feasibility_alignment": 0.14,
        }
    return {
        "reasoning_quality": 0.22,
        "evidence_grounding": 0.28,
        "readiness_feasibility_alignment": 0.20,
        "bfs_policy_adherence": 0.16,
        "ontology_alignment": 0.14,
    }


def _heuristic_stage_critic(
    *,
    stage: str,
    profile_id: str,
    stage_output: Dict[str, Any],
    evidence_bundle: Dict[str, Any],
    pass_threshold_0_1: float,
) -> StageCriticReviewModel:
    stage_key = str(stage).strip().lower()
    readiness = evidence_bundle.get("readiness", {}) if isinstance(evidence_bundle, dict) else {}
    has_readiness = bool(readiness.get("label") or readiness.get("readiness_label"))

    if stage_key == "step04":
        predictors = [normalize_path_text(p) for p in stage_output.get("recommended_next_observation_predictors", []) if normalize_path_text(p)]
        shortlist = [normalize_path_text((row or {}).get("predictor_path", "")) for row in stage_output.get("refined_predictor_shortlist", [])]
        shortlist = [item for item in shortlist if item]
        criteria = [str(c) for c in stage_output.get("retained_criteria_ids", []) if str(c).strip()]
        subscores = {
            "predictor_grounding": 0.30 + 0.70 * min(1.0, len(shortlist) / max(1.0, len(predictors) or 1.0)),
            "criterion_continuity": 0.70 if len(criteria) >= 3 else 0.45,
            "bfs_depth_balance": 0.75 if "breadth" in str(stage_output.get("rationale", "")).lower() else 0.55,
            "fusion_consistency": 0.70 if "fusion" in str(stage_output.get("rationale", "")).lower() else 0.55,
            "feasibility_alignment": 0.78 if 2 <= len(predictors) <= 8 else 0.52,
        }
    else:
        recommended = stage_output.get("recommended_targets", []) or []
        ranked = stage_output.get("ranked_predictors", []) or []
        mapped_count = 0
        for item in ranked:
            path_value = normalize_path_text(str((item or {}).get("mapped_leaf_path") or ""))
            if path_value:
                mapped_count += 1
        mapped_ratio = mapped_count / max(1.0, float(len(ranked) or 1))
        subscores = {
            "reasoning_quality": 0.72 if str(stage_output.get("summary", "")).strip() else 0.48,
            "evidence_grounding": 0.25 + 0.75 * mapped_ratio,
            "readiness_feasibility_alignment": 0.70 if has_readiness else 0.52,
            "bfs_policy_adherence": 0.75 if len(ranked) >= max(2, len(recommended)) else 0.50,
            "ontology_alignment": 0.25 + 0.75 * mapped_ratio,
        }

    weighted = weighted_composite(subscores=subscores, weights=_critic_weights(stage_key))
    composite = float(weighted.get("composite_score_0_1", 0.0))
    decision = decision_from_score(score_0_1=composite, threshold_0_1=pass_threshold_0_1)
    return StageCriticReviewModel(
        profile_id=profile_id,
        stage=stage_key,
        pass_decision=decision,
        composite_score_0_1=composite,
        weighted_subscores_0_1={k: float(v) for k, v in weighted.get("subscores_0_1", {}).items()},
        critical_issues=[],
        strengths=["Heuristic critic fallback used due unavailable or disabled LLM critic."],
        feedback_for_revision=[] if decision == "PASS" else ["Increase evidence grounding and strengthen ontology/path alignment."],
        evidence_gaps=[] if has_readiness else ["Readiness evidence unavailable or incomplete."],
        confidence_0_1=0.55,
    )


def run_llm_stage_critic(
    *,
    client: StructuredLLMClient,
    profile_id: str,
    stage: str,
    stage_output_payload: Dict[str, Any],
    evidence_bundle: Dict[str, Any],
    pass_threshold_0_1: float,
    prompt_budget_tokens: int,
) -> Tuple[Optional[StageCriticReviewModel], Dict[str, Any]]:
    stage_key = str(stage).strip().lower()
    if stage_key not in {"step03", "step04"}:
        raise ValueError(f"Unsupported critic stage: {stage}")
    system_name = f"{stage_key}_target_selection_critic_system.md" if stage_key == "step03" else "step04_observation_update_critic_system.md"
    user_name = f"{stage_key}_target_selection_critic_user_template.md" if stage_key == "step03" else "step04_observation_update_critic_user_template.md"
    system_template = load_prompt(system_name)
    user_template = load_prompt(user_name)

    sections = [
        PromptSection("meta", json.dumps(evidence_bundle.get("meta", {}), ensure_ascii=False, indent=2), priority=1),
        PromptSection("stage_output", json.dumps(stage_output_payload, ensure_ascii=False, indent=2), priority=2),
        PromptSection("readiness", json.dumps(evidence_bundle.get("readiness", {}), ensure_ascii=False, indent=2), priority=3),
        PromptSection("network", json.dumps(evidence_bundle.get("network", {}), ensure_ascii=False, indent=2), priority=4),
        PromptSection("impact", json.dumps(evidence_bundle.get("impact", {}), ensure_ascii=False, indent=2), priority=5),
        PromptSection("mapping_evidence", json.dumps(evidence_bundle.get("mapping_evidence", {}), ensure_ascii=False, indent=2), priority=6),
        PromptSection("bfs_planner", json.dumps(evidence_bundle.get("bfs_planner", {}), ensure_ascii=False, indent=2), priority=7),
        PromptSection("parent_feasibility", json.dumps(evidence_bundle.get("predictor_parent_feasibility", []), ensure_ascii=False, indent=2), priority=8),
    ]
    packed = pack_prompt_sections(
        sections,
        max_tokens=int(prompt_budget_tokens),
        reserve_tokens=2500,
        model=client.model,
    )
    user_prompt = render_prompt(
        user_template,
        {
            "STAGE": stage_key,
            "PASS_THRESHOLD": f"{float(pass_threshold_0_1):.3f}",
            "EVIDENCE_BUNDLE_JSON": packed.text,
        },
    )
    llm_result = client.generate_structured(
        system_prompt=system_template,
        user_prompt=user_prompt,
        schema_model=StageCriticReviewModel,
    )
    trace = {
        "profile_id": profile_id,
        "stage": stage_key,
        "packed_prompt_estimated_tokens": packed.estimated_tokens,
        "packed_prompt_max_tokens": packed.max_tokens,
        "included_sections": packed.included_sections,
        "truncated_sections": packed.truncated_sections,
        "section_token_estimates": packed.section_token_estimates,
        "provider": llm_result.provider,
        "model": llm_result.model,
        "success": llm_result.success,
        "used_repair": llm_result.used_repair,
        "validation_error": llm_result.validation_error,
        "failure_reason": llm_result.failure_reason,
        "usage": llm_result.usage,
    }
    if not llm_result.success or llm_result.parsed is None:
        return None, trace
    parsed = StageCriticReviewModel.model_validate(llm_result.parsed)
    parsed.profile_id = profile_id
    parsed.stage = stage_key
    weighted = weighted_composite(subscores=parsed.weighted_subscores_0_1, weights=_critic_weights(stage_key))
    parsed.composite_score_0_1 = float(weighted.get("composite_score_0_1", parsed.composite_score_0_1))
    parsed.pass_decision = decision_from_score(
        score_0_1=float(parsed.composite_score_0_1),
        threshold_0_1=float(pass_threshold_0_1),
        critical_issues=list(parsed.critical_issues or []),
    )
    return parsed, trace


def serialize_step03_to_frame(
    profile_id: str,
    step03: Step03SelectionModel,
    predictor_impact_lookup: Dict[str, float],
    predictor_impact_pct_lookup: Dict[str, float],
) -> pd.DataFrame:
    selected_ids = {item.predictor for item in step03.recommended_targets}
    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(step03.ranked_predictors, start=1):
        rows.append(
            {
                "source_profile": profile_id,
                "predictor_rank": idx,
                "predictor": item.predictor,
                "predictor_label": item.predictor_label,
                "predictor_impact": float(predictor_impact_lookup.get(item.predictor, item.score_0_1)),
                "predictor_impact_pct": float(predictor_impact_pct_lookup.get(item.predictor, item.score_0_1 * 100.0)),
                "priority_level": priority_from_impact(float(item.score_0_1)),
                "selection_reason": item.rationale,
                "selected_for_intervention": item.predictor in selected_ids,
                "selection_score_0_1": float(item.score_0_1),
                "mapped_leaf_path": item.mapped_leaf_path,
                "confidence_0_1": float(item.confidence_0_1),
            }
        )
    return pd.DataFrame(rows)


def summarize_mapping_rows(mapping_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not mapping_rows:
        return {"n_rows": 0, "parts": {}, "top_predictor_paths": []}
    frame = pd.DataFrame(mapping_rows)
    part_counts = frame["part"].astype(str).value_counts().to_dict() if "part" in frame.columns else {}
    top_paths: List[Dict[str, Any]] = []
    if "predictor_path" in frame.columns and "relevance_score_0_1" in frame.columns:
        top_frame = (
            frame.groupby("predictor_path", as_index=False)["relevance_score_0_1"]
            .max()
            .sort_values("relevance_score_0_1", ascending=False)
            .head(30)
        )
        top_paths = top_frame.to_dict(orient="records")
    criterion_count = int(frame["criterion_path"].replace("", np.nan).dropna().nunique()) if "criterion_path" in frame.columns else 0
    return {
        "n_rows": int(len(frame)),
        "parts": {str(k): int(v) for k, v in part_counts.items()},
        "n_unique_criterion_paths": criterion_count,
        "top_predictor_paths": top_paths,
    }


def load_previous_cycle_scores(
    *,
    profile_id: str,
    history_snapshots_path: Optional[Path],
    memory_window: int,
) -> Dict[str, float]:
    if history_snapshots_path is None or not history_snapshots_path.exists():
        return {}
    try:
        frame = pd.read_parquet(history_snapshots_path)
    except Exception:
        return {}
    if frame.empty or "profile_id" not in frame.columns:
        return {}
    local = frame.loc[frame["profile_id"].astype(str) == str(profile_id)].copy()
    if local.empty:
        return {}
    if "cycle_index" in local.columns:
        local["cycle_index"] = pd.to_numeric(local["cycle_index"], errors="coerce").fillna(0).astype(int)
        local = local.sort_values("cycle_index")
    if memory_window > 0:
        local = local.tail(max(1, int(memory_window)))
    if "next_predictors" not in local.columns:
        return {}
    score_map: Dict[str, float] = {}
    for _, row in local.iterrows():
        predictors_raw = str(row.get("next_predictors") or "")
        predictors = [normalize_path_text(item) for item in predictors_raw.split("|") if normalize_path_text(item)]
        for rank, predictor_path in enumerate(predictors, start=1):
            weight = max(0.10, 1.0 - (rank - 1) / 20.0)
            key = predictor_path.lower()
            score_map[key] = max(score_map.get(key, 0.0), float(weight))
    return score_map


def apply_fusion_to_step04_output(
    *,
    step04_output: Step04UpdatedObservationModel,
    step03_output: Step03SelectionModel,
    initial_model_summary: Dict[str, Any],
    initial_model_payload: Dict[str, Any],
    impact_matrix: pd.DataFrame,
    mapping_rows: Sequence[Dict[str, Any]],
    readiness_score_0_100: Optional[float],
    previous_cycle_scores: Optional[Dict[str, float]],
    max_candidate_predictors: int,
) -> Dict[str, Any]:
    candidate_prior_scores: Dict[str, float] = {}
    candidate_paths: List[str] = []
    for item in step04_output.refined_predictor_shortlist:
        path = normalize_path_text(item.predictor_path)
        if not path:
            continue
        candidate_paths.append(path)
        candidate_prior_scores[path] = max(candidate_prior_scores.get(path, 0.0), float(item.score_0_1))

    if not candidate_paths:
        candidate_paths = [
            normalize_path_text(item.mapped_leaf_path)
            for item in step03_output.recommended_targets
            if normalize_path_text(item.mapped_leaf_path)
        ]
        for path in candidate_paths:
            candidate_prior_scores[path] = 0.30

    fusion_payload = fuse_updated_model_matrix(
        criteria_summary=initial_model_summary.get("criteria_summary", []),
        predictor_summary=initial_model_summary.get("predictor_summary", []),
        initial_model_payload=initial_model_payload,
        impact_matrix=impact_matrix,
        candidate_paths=candidate_paths,
        candidate_prior_scores=candidate_prior_scores,
        mapping_rows=mapping_rows,
        readiness_score_0_100=readiness_score_0_100,
        previous_cycle_scores=previous_cycle_scores,
        max_predictors=max_candidate_predictors,
    )

    fused_shortlist: List[UpdatedPredictorCandidateModel] = []
    for ranking in fusion_payload.get("predictor_rankings", []):
        predictor_path = normalize_path_text(str(ranking.get("predictor_path") or ""))
        if not predictor_path:
            continue
        fused_shortlist.append(
            UpdatedPredictorCandidateModel(
                predictor_path=predictor_path,
                score_0_1=float(ranking.get("fused_score_0_1", 0.0)),
                source="nomothetic_idiographic_fusion",
                reason=(
                    "Fused from profile-specific mapping priors and observed idiographic impact, "
                    "while preserving breadth-first domain coverage."
                ),
            )
        )
    if fused_shortlist:
        step04_output.refined_predictor_shortlist = fused_shortlist

    next_predictors = [item.predictor_path for item in fused_shortlist[:20]]
    if not next_predictors:
        next_predictors = [item.predictor_path for item in step04_output.refined_predictor_shortlist[:20]]
    step04_output.recommended_next_observation_predictors = next_predictors

    current_predictor_paths = {
        normalize_path_text(str(item.get("mapped_leaf_full_path") or item.get("var_id") or ""))
        for item in initial_model_summary.get("predictor_summary", [])
        if str(item.get("mapped_leaf_full_path") or item.get("var_id") or "").strip()
    }
    next_set = set(next_predictors)
    step04_output.dropped_predictors = sorted(list(current_predictor_paths.difference(next_set)))
    step04_output.added_predictors = sorted(list(next_set.difference(current_predictor_paths)))

    fusion_weights = fusion_payload.get("weights", {}) or {}
    readiness_weight = fusion_weights.get("readiness_0_1", 0.0)
    step04_output.rationale = (
        "Updated model uses breadth-first subtree selection with weighted nomothetic/idiographic fusion. "
        f"Readiness-scaled idiographic weight={fusion_weights.get('idiographic_weight', 0.0)} "
        f"(readiness_0_1={readiness_weight})."
    )

    return fusion_payload


def enforce_step04_range_policy(
    *,
    step04_output: Step04UpdatedObservationModel,
    step03_output: Step03SelectionModel,
    initial_model_summary: Dict[str, Any],
    ontology_candidates: Sequence[Dict[str, Any]],
    predictor_min: int = 2,
    predictor_max: int = 8,
    criterion_min: int = 3,
    criterion_max: int = 6,
    preferred_predictor_count: Optional[int] = None,
    preferred_criterion_count: Optional[int] = None,
) -> Dict[str, Any]:
    reason_codes: List[str] = []

    current = [normalize_path_text(p) for p in step04_output.recommended_next_observation_predictors if normalize_path_text(p)]
    seen = set(current)
    if len(current) < predictor_min:
        for item in step03_output.ranked_predictors:
            candidate = normalize_path_text(item.mapped_leaf_path)
            if not candidate:
                continue
            if candidate in seen:
                continue
            current.append(candidate)
            seen.add(candidate)
            if len(current) >= predictor_min:
                break
    if len(current) < predictor_min:
        for item in ontology_candidates:
            candidate = normalize_path_text(str(item.get("predictor_path") or ""))
            if not candidate or candidate in seen:
                continue
            current.append(candidate)
            seen.add(candidate)
            if len(current) >= predictor_min:
                break
    if len(current) < predictor_min:
        reason_codes.append("predictor_count_below_range")
    if len(current) > predictor_max:
        current = current[:predictor_max]
        reason_codes.append("predictor_count_trimmed_to_policy")
    preferred_predictors = int(preferred_predictor_count) if preferred_predictor_count is not None else 0
    if preferred_predictors > 0 and len(current) > preferred_predictors:
        current = current[:preferred_predictors]
        reason_codes.append("predictor_count_trimmed_to_preferred")

    criteria = [str(c) for c in step04_output.retained_criteria_ids if str(c).strip()]
    if len(criteria) < criterion_min:
        initial = [
            str(item.get("var_id"))
            for item in initial_model_summary.get("criteria_summary", [])
            if str(item.get("var_id") or "").strip()
        ]
        for criterion in initial:
            if criterion in criteria:
                continue
            criteria.append(criterion)
            if len(criteria) >= criterion_min:
                break
    if len(criteria) < criterion_min:
        reason_codes.append("criterion_count_below_range")
    if len(criteria) > criterion_max:
        reason_codes.append("criterion_count_above_range")
    preferred_criteria = int(preferred_criterion_count) if preferred_criterion_count is not None else 0
    if preferred_criteria > 0 and len(criteria) > preferred_criteria:
        criteria = criteria[:preferred_criteria]
        reason_codes.append("criterion_count_trimmed_to_preferred")

    step04_output.recommended_next_observation_predictors = current
    step04_output.retained_criteria_ids = criteria
    return {
        "predictor_target_range": [predictor_min, predictor_max],
        "criterion_target_range": [criterion_min, criterion_max],
        "preferred_predictor_count": preferred_predictors,
        "preferred_criterion_count": preferred_criteria,
        "predictor_count": len(current),
        "criterion_count": len(criteria),
        "reason_codes": reason_codes,
    }


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[4]
    default_impact_root = repo_root / "evaluation/04_initial_observation_analysis/02_momentary_impact_coefficients"
    default_output_root = repo_root / "evaluation/05_treatment_target_handoff"
    default_readiness_root = repo_root / "evaluation/04_initial_observation_analysis/00_readiness_check"
    default_network_root = repo_root / "evaluation/04_initial_observation_analysis/01_time_series_analysis/network"
    default_model_runs = repo_root / "evaluation/03_construction_initial_observation_model/constructed_PC_models/runs"
    default_free_text_root = repo_root / "evaluation/01_pseudoprofile(s)/free_text"
    default_mapping_ranks = (
        repo_root
        / "evaluation/03_construction_initial_observation_model/helpers/00_LLM_based_mapping_based_predictor_ranks/all_pseudoprofiles__predictor_ranks_dense.csv"
    )
    default_hyde_runs_root = (
        repo_root / "evaluation/03_construction_initial_observation_model/helpers/00_HyDe_based_predictor_ranks/runs"
    )
    default_predictor_leaf_paths = (
        repo_root
        / "src/SystemComponents/Agentic_Framework/02_ConstructionInitialObservationModel/utils/helpers/embeddings/PREDICTORS/PREDICTOR_leaf_paths_FULL.json"
    )
    default_predictor_list = (
        repo_root / "src/utils/official/ontology_mappings/CRITERION/predictor_to_criterion/input_lists/predictors_list.txt"
    )
    default_predictor_feasibility = (
        repo_root
        / "src/utils/official/multi_dimensional_feasibility_evaluation/PREDICTORS/results/summary/predictor_rankings.csv"
    )

    parser = argparse.ArgumentParser(
        description="Identify treatment targets (Step-03) and produce updated observation model suggestions (Step-04)."
    )
    parser.add_argument("--impact-root", type=str, default=str(default_impact_root))
    parser.add_argument("--output-root", type=str, default=str(default_output_root))
    parser.add_argument("--readiness-root", type=str, default=str(default_readiness_root))
    parser.add_argument("--network-root", type=str, default=str(default_network_root))
    parser.add_argument("--initial-model-runs-root", type=str, default=str(default_model_runs))
    parser.add_argument("--free-text-root", type=str, default=str(default_free_text_root))
    parser.add_argument("--mapping-ranks-csv", type=str, default=str(default_mapping_ranks))
    parser.add_argument("--hyde-dense-profiles-csv", type=str, default="")
    parser.add_argument("--hyde-runs-root", type=str, default=str(default_hyde_runs_root))
    parser.add_argument("--predictor-leaf-paths-json", type=str, default=str(default_predictor_leaf_paths))
    parser.add_argument("--predictor-list-path", type=str, default=str(default_predictor_list))
    parser.add_argument("--predictor-feasibility-csv", type=str, default=str(default_predictor_feasibility))
    parser.add_argument("--pattern", type=str, default="pseudoprofile_FTC_")
    parser.add_argument("--max-profiles", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--min-impact", type=float, default=0.10)
    parser.add_argument("--max-candidate-predictors", type=int, default=200)
    parser.add_argument("--llm-model", type=str, default="gpt-5-nano")
    parser.add_argument("--llm-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--llm-max-attempts", type=int, default=2)
    parser.add_argument("--llm-repair-attempts", type=int, default=1)
    parser.add_argument("--disable-llm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prompt-budget-tokens", type=int, default=400000)
    parser.add_argument("--critic-max-iterations", type=int, default=2)
    parser.add_argument("--critic-pass-threshold", type=float, default=0.74)
    parser.add_argument("--hard-ontology-constraint", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--preferred-predictor-count", type=int, default=6)
    parser.add_argument("--preferred-criterion-count", type=int, default=4)
    parser.add_argument("--parent-feasibility-top-k", type=int, default=30)
    parser.add_argument("--visualize-updated-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visualization-dpi", type=int, default=300)
    parser.add_argument("--visualization-top-predictors", type=int, default=20)
    parser.add_argument("--visualization-top-edges", type=int, default=90)
    parser.add_argument("--history-snapshots-path", type=str, default="")
    parser.add_argument("--profile-memory-window", type=int, default=3)
    parser.add_argument("--contract-version", type=str, default="1.0.0")
    parser.add_argument("--trace-output", type=str, default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    impact_root = Path(args.impact_root).expanduser().resolve()
    output_root = ensure_dir(Path(args.output_root).expanduser().resolve())
    readiness_root = Path(args.readiness_root).expanduser().resolve()
    network_root = Path(args.network_root).expanduser().resolve()
    model_runs_root = Path(args.initial_model_runs_root).expanduser().resolve()
    free_text_root = Path(args.free_text_root).expanduser().resolve()
    mapping_ranks_csv = Path(args.mapping_ranks_csv).expanduser().resolve()
    hyde_runs_root = Path(args.hyde_runs_root).expanduser().resolve()
    predictor_leaf_paths_json = Path(args.predictor_leaf_paths_json).expanduser().resolve()
    predictor_list_path = Path(args.predictor_list_path).expanduser().resolve()
    predictor_feasibility_csv = Path(args.predictor_feasibility_csv).expanduser().resolve()
    history_snapshots_path = (
        Path(args.history_snapshots_path).expanduser().resolve()
        if str(args.history_snapshots_path).strip()
        else None
    )
    hyde_dense_profiles_csv = (
        Path(args.hyde_dense_profiles_csv).expanduser().resolve()
        if str(args.hyde_dense_profiles_csv).strip()
        else None
    )
    if hyde_dense_profiles_csv is None or not hyde_dense_profiles_csv.exists():
        hyde_dense_profiles_csv = discover_latest_hyde_dense_profiles(hyde_runs_root)

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

    complaints = parse_profile_text_file(free_text_root / "free_text_complaints.txt")
    person_profiles = parse_profile_text_file(free_text_root / "free_text_person.txt")
    context_profiles = parse_profile_text_file(free_text_root / "free_text_context.txt")
    predictor_catalog = parse_predictor_catalog(predictor_list_path)
    predictor_leaf_paths = load_predictor_leaf_paths(predictor_leaf_paths_json)
    predictor_feasibility_df = load_predictor_feasibility_table(predictor_feasibility_csv)
    if predictor_leaf_paths:
        log(f"loaded predictor leaf paths: {len(predictor_leaf_paths)}")
    else:
        log("predictor leaf paths unavailable; falling back to predictor_list shallow catalog.")
    if predictor_feasibility_df.empty:
        log(f"predictor feasibility table unavailable or empty: {predictor_feasibility_csv}")
    else:
        log(f"loaded predictor feasibility rows: {len(predictor_feasibility_df)}")

    llm_client = StructuredLLMClient(
        model=str(args.llm_model),
        timeout_seconds=float(args.llm_timeout_seconds),
        max_attempts=int(args.llm_max_attempts),
        repair_attempts=int(args.llm_repair_attempts),
    )

    log("========== TREATMENT-TARGET IDENTIFICATION START ==========")
    log(f"impact_root: {impact_root}")
    log(f"output_root: {output_root}")
    log(f"profiles: {len(profiles)}")
    log(f"llm_model: {args.llm_model}")
    log(f"disable_llm: {bool(args.disable_llm)}")
    log(f"hard_ontology_constraint: {bool(args.hard_ontology_constraint)}")
    log(f"mapping_ranks_csv: {mapping_ranks_csv}")
    log(f"hyde_dense_profiles_csv: {hyde_dense_profiles_csv if hyde_dense_profiles_csv else 'not_found'}")
    if history_snapshots_path is not None:
        log(f"history_snapshots_path: {history_snapshots_path}")

    profile_rows: List[Dict[str, Any]] = []
    updated_model_rows: List[Dict[str, Any]] = []
    failed = 0

    for profile_dir in profiles:
        profile_id = profile_dir.name
        out_profile_dir = ensure_dir(output_root / profile_id)
        try:
            predictor_composite_path = profile_dir / "predictor_composite.csv"
            if not predictor_composite_path.exists():
                raise FileNotFoundError(f"Missing predictor composite file: {predictor_composite_path}")
            impact_df_all = safe_read_csv(predictor_composite_path)
            if "predictor" not in impact_df_all.columns:
                raise ValueError(f"{profile_id}: predictor_composite.csv missing required column 'predictor'")
            if "predictor_impact" not in impact_df_all.columns:
                impact_df_all["predictor_impact"] = 0.0
            if "predictor_impact_pct" not in impact_df_all.columns:
                impact_df_all["predictor_impact_pct"] = (
                    pd.to_numeric(impact_df_all["predictor_impact"], errors="coerce").fillna(0.0) * 100.0
                )
            if "predictor_label" not in impact_df_all.columns:
                impact_df_all["predictor_label"] = impact_df_all["predictor"].astype(str)
            impact_df_all = impact_df_all.sort_values("predictor_impact", ascending=False).reset_index(drop=True)
            impact_df = impact_df_all.copy()
            if args.top_k > 0:
                impact_df = impact_df.head(int(args.top_k)).copy()

            impact_lookup = {str(row["predictor"]): float(row.get("predictor_impact", 0.0)) for _, row in impact_df.iterrows()}
            impact_pct_lookup = {
                str(row["predictor"]): float(row.get("predictor_impact_pct", float(row.get("predictor_impact", 0.0)) * 100.0))
                for _, row in impact_df.iterrows()
            }

            edge_rows = []
            edge_path = profile_dir / "edge_composite.csv"
            if edge_path.exists():
                edge_df = safe_read_csv(edge_path)
                edge_rows = edge_df.sort_values("edge_impact", ascending=False).head(30).fillna("").to_dict(orient="records")

            readiness_payload = read_json(readiness_root / profile_id / "readiness_report.json") if (readiness_root / profile_id / "readiness_report.json").exists() else {}
            network_summary = summarize_network_metrics(network_root / profile_id)
            initial_model_path = find_latest_initial_model(model_runs_root, profile_id=profile_id)
            initial_model_payload = read_json(initial_model_path) if initial_model_path else {}
            initial_model_summary = summarize_initial_model(initial_model_payload) if initial_model_payload else {"criteria_summary": [], "predictor_summary": [], "top_edges_predictor_to_criterion": []}

            mapped_predictor_paths = {
                str(item.get("var_id")): normalize_path_text(str(item.get("mapped_leaf_full_path") or ""))
                for item in initial_model_summary.get("predictor_summary", [])
            }
            mapped_predictor_path_values = [path for path in mapped_predictor_paths.values() if path]
            mapping_rows = load_profile_mapping_rows(mapping_ranks_csv, profile_id=profile_id)
            hyde_scores = (
                load_profile_hyde_scores(hyde_dense_profiles_csv, profile_id=profile_id, top_k=300)
                if hyde_dense_profiles_csv and hyde_dense_profiles_csv.exists()
                else {}
            )
            impact_lookup_all = {
                str(row["predictor"]): float(row.get("predictor_impact", 0.0)) for _, row in impact_df_all.iterrows()
            }

            if predictor_leaf_paths:
                ontology_candidates = build_bfs_candidates(
                    leaf_paths=predictor_leaf_paths,
                    mapping_rows=mapping_rows,
                    hyde_scores=hyde_scores,
                    mapped_predictor_paths=mapped_predictor_path_values,
                    impact_by_predictor=impact_lookup_all,
                    predictor_var_to_path=mapped_predictor_paths,
                    max_candidates=int(args.max_candidate_predictors),
                )
            else:
                ontology_candidates = []
            if not ontology_candidates:
                ontology_candidates = build_ontology_candidate_set(
                    predictor_catalog=predictor_catalog,
                    initial_model_summary=initial_model_summary,
                    impact_df=impact_df_all,
                    max_candidates=int(args.max_candidate_predictors),
                )
                for idx, row in enumerate(ontology_candidates, start=1):
                    row["bfs_stage"] = "fallback_similarity"
                    row["bfs_rank"] = idx
                    row["bfs_domain_key"] = str(row.get("secondary_node", "")).lower()

            free_text = profile_text_bundle(
                profile_id=profile_id,
                complaints=complaints,
                person_profiles=person_profiles,
                context_profiles=context_profiles,
            )

            parent_feasibility_paths = [
                normalize_path_text(str(item.get("predictor_path") or ""))
                for item in ontology_candidates[: max(1, int(args.parent_feasibility_top_k))]
            ]
            parent_feasibility_paths.extend(mapped_predictor_path_values[:20])
            predictor_parent_feasibility = top_parent_domains_for_bundle(
                predictor_paths=[item for item in parent_feasibility_paths if item],
                feasibility_frame=predictor_feasibility_df,
                top_k=max(1, int(args.parent_feasibility_top_k)),
                per_predictor_k=max(1, int(args.parent_feasibility_top_k)),
                parent_levels=2,
            )

            ontology_candidates_frame = pd.DataFrame(ontology_candidates) if ontology_candidates else pd.DataFrame()
            bfs_stage_counts = (
                ontology_candidates_frame["bfs_stage"].astype(str).value_counts().to_dict()
                if not ontology_candidates_frame.empty and "bfs_stage" in ontology_candidates_frame.columns
                else {}
            )
            bfs_domain_count = (
                int(ontology_candidates_frame["bfs_domain_key"].astype(str).nunique())
                if not ontology_candidates_frame.empty and "bfs_domain_key" in ontology_candidates_frame.columns
                else 0
            )

            evidence_bundle = {
                "meta": {
                    "profile_id": profile_id,
                    "generated_at_local": ts(),
                    "sources": {
                        "impact_root": str(impact_root),
                        "readiness_root": str(readiness_root),
                        "network_root": str(network_root),
                        "initial_model_path": str(initial_model_path) if initial_model_path else "",
                        "mapping_ranks_csv": str(mapping_ranks_csv),
                        "hyde_dense_profiles_csv": str(hyde_dense_profiles_csv) if hyde_dense_profiles_csv else "",
                    },
                },
                "free_text": free_text,
                "readiness": summarize_readiness(readiness_payload) if readiness_payload else {},
                "impact": {
                    "top_predictors": impact_df_all.head(25).fillna("").to_dict(orient="records"),
                    "top_edges_predictor_to_criterion": edge_rows,
                },
                "network": network_summary,
                "initial_model": initial_model_summary,
                "mapping_evidence": summarize_mapping_rows(mapping_rows),
                "hyde_evidence": {
                    "n_hyde_paths": int(len(hyde_scores)),
                    "top_hyde_paths": [
                        {"predictor_path": key, "score_0_1": value}
                        for key, value in sorted(hyde_scores.items(), key=lambda item: item[1], reverse=True)[:40]
                    ],
                },
                "bfs_planner": {
                    "n_candidates": int(len(ontology_candidates)),
                    "n_domains": bfs_domain_count,
                    "stage_counts": {str(k): int(v) for k, v in bfs_stage_counts.items()},
                    "top_candidates": [
                        {
                            "predictor_path": str(item.get("predictor_path")),
                            "score_0_1": float(item.get("subtree_relevance_score_0_1", 0.0)),
                            "bfs_stage": str(item.get("bfs_stage", "")),
                            "bfs_domain_key": str(item.get("bfs_domain_key", "")),
                        }
                        for item in list(ontology_candidates)[:40]
                    ],
                },
                "predictor_parent_feasibility": predictor_parent_feasibility,
                "ontology_candidates": ontology_candidates,
            }

            critic_max_iterations = max(0, int(args.critic_max_iterations))
            step03_trace: Dict[str, Any] = {"provider": "heuristic", "reason": "disabled", "actor_attempts": [], "critic_attempts": []}
            step03_critic_review: Optional[StageCriticReviewModel] = None
            step03_feedback: List[str] = []
            step03_output: Optional[Step03SelectionModel] = None

            for attempt in range(critic_max_iterations + 1):
                step03_bundle_for_attempt = dict(evidence_bundle)
                if step03_feedback:
                    step03_bundle_for_attempt["critic_feedback"] = {
                        "feedback_for_revision": step03_feedback,
                        "attempt_index": attempt,
                    }
                if not bool(args.disable_llm):
                    step03_candidate, step03_actor_trace = run_llm_step03(
                        client=llm_client,
                        profile_id=profile_id,
                        evidence_bundle=step03_bundle_for_attempt,
                        prompt_budget_tokens=int(args.prompt_budget_tokens),
                    )
                else:
                    step03_candidate = None
                    step03_actor_trace = {"provider": "heuristic", "reason": "disabled"}
                step03_actor_trace["attempt_index"] = attempt
                step03_trace["actor_attempts"].append(step03_actor_trace)

                actor_mode = "structured_llm_success"
                if step03_candidate is None:
                    actor_mode = "heuristic_fallback_disabled_llm" if bool(args.disable_llm) else "heuristic_fallback_llm_failure"
                    step03_candidate = heuristic_step03_selection(
                        profile_id=profile_id,
                        impact_df=impact_df,
                        mapped_predictor_paths=mapped_predictor_paths,
                        top_k=int(args.top_k),
                        min_impact=float(args.min_impact),
                    )

                if bool(args.hard_ontology_constraint):
                    step03_constraint = _enforce_step03_hard_ontology(
                        step03_candidate,
                        allowed_predictor_paths=[str(item.get("predictor_path") or "") for item in ontology_candidates],
                        predictor_var_to_path=mapped_predictor_paths,
                    )
                else:
                    step03_constraint = {"applied": False}

                run_critic = (not bool(args.disable_llm)) and (actor_mode == "structured_llm_success")
                if run_critic:
                    critic_payload = {
                        **evidence_bundle,
                        "step_output": step03_candidate.model_dump(mode="json"),
                    }
                    critic_llm, critic_trace = run_llm_stage_critic(
                        client=llm_client,
                        profile_id=profile_id,
                        stage="step03",
                        stage_output_payload=step03_candidate.model_dump(mode="json"),
                        evidence_bundle=critic_payload,
                        pass_threshold_0_1=float(args.critic_pass_threshold),
                        prompt_budget_tokens=int(args.prompt_budget_tokens),
                    )
                    if critic_llm is None:
                        step03_critic_review = _heuristic_stage_critic(
                            stage="step03",
                            profile_id=profile_id,
                            stage_output=step03_candidate.model_dump(mode="json"),
                            evidence_bundle=critic_payload,
                            pass_threshold_0_1=float(args.critic_pass_threshold),
                        )
                        critic_trace["mode"] = "heuristic_fallback"
                    else:
                        step03_critic_review = critic_llm
                        critic_trace["mode"] = "structured_llm_success"
                    critic_trace["attempt_index"] = attempt
                    critic_trace["constraint"] = step03_constraint
                    critic_trace["critic_decision"] = step03_critic_review.pass_decision
                    critic_trace["critic_composite_score_0_1"] = step03_critic_review.composite_score_0_1
                    step03_trace["critic_attempts"].append(critic_trace)

                    if (
                        step03_critic_review.pass_decision == "PASS"
                        or attempt >= critic_max_iterations
                    ):
                        step03_output = step03_candidate
                        step03_trace["reason"] = actor_mode
                        break
                    step03_feedback = list(step03_critic_review.feedback_for_revision or [])
                    if not step03_feedback:
                        step03_feedback = ["Increase ontology alignment and provide stronger evidence grounding."]
                    continue

                step03_critic_review = _heuristic_stage_critic(
                    stage="step03",
                    profile_id=profile_id,
                    stage_output=step03_candidate.model_dump(mode="json"),
                    evidence_bundle=evidence_bundle,
                    pass_threshold_0_1=float(args.critic_pass_threshold),
                )
                step03_trace["critic_attempts"].append(
                    {
                        "attempt_index": attempt,
                        "mode": "heuristic_auto",
                        "constraint": step03_constraint,
                        "critic_decision": step03_critic_review.pass_decision,
                        "critic_composite_score_0_1": step03_critic_review.composite_score_0_1,
                    }
                )
                step03_output = step03_candidate
                step03_trace["reason"] = actor_mode
                step03_trace["constraint"] = step03_constraint
                break

            assert step03_output is not None
            step03_output.contract_version = str(args.contract_version)
            if step03_critic_review is not None:
                step03_trace["critic_final_decision"] = step03_critic_review.pass_decision
                step03_trace["critic_final_score_0_1"] = step03_critic_review.composite_score_0_1

            step04_bundle = {
                "meta": evidence_bundle["meta"],
                "step03_output": step03_output.model_dump(mode="json"),
                "initial_model": initial_model_summary,
                "readiness": evidence_bundle["readiness"],
                "mapping_evidence": evidence_bundle.get("mapping_evidence", {}),
                "bfs_planner": evidence_bundle.get("bfs_planner", {}),
                "predictor_parent_feasibility": predictor_parent_feasibility,
                "fusion_prior": {
                    "candidate_seed_top": [
                        {
                            "predictor_path": str(item.get("predictor_path")),
                            "seed_score_0_1": float(item.get("subtree_relevance_score_0_1", 0.0)),
                        }
                        for item in list(ontology_candidates)[:40]
                    ]
                },
                "ontology_candidates": ontology_candidates,
            }

            step04_trace: Dict[str, Any] = {"provider": "heuristic", "reason": "disabled", "actor_attempts": [], "critic_attempts": []}
            step04_critic_review: Optional[StageCriticReviewModel] = None
            step04_feedback: List[str] = []
            step04_output: Optional[Step04UpdatedObservationModel] = None
            for attempt in range(critic_max_iterations + 1):
                step04_bundle_for_attempt = dict(step04_bundle)
                if step04_feedback:
                    step04_bundle_for_attempt["critic_feedback"] = {
                        "feedback_for_revision": step04_feedback,
                        "attempt_index": attempt,
                    }
                if not bool(args.disable_llm):
                    step04_candidate, step04_actor_trace = run_llm_step04(
                        client=llm_client,
                        profile_id=profile_id,
                        evidence_bundle=step04_bundle_for_attempt,
                        prompt_budget_tokens=int(args.prompt_budget_tokens),
                    )
                else:
                    step04_candidate = None
                    step04_actor_trace = {"provider": "heuristic", "reason": "disabled"}
                step04_actor_trace["attempt_index"] = attempt
                step04_trace["actor_attempts"].append(step04_actor_trace)

                actor_mode = "structured_llm_success"
                if step04_candidate is None:
                    actor_mode = "heuristic_fallback_disabled_llm" if bool(args.disable_llm) else "heuristic_fallback_llm_failure"
                    step04_candidate = heuristic_step04_update(
                        profile_id=profile_id,
                        selected_step03=step03_output,
                        initial_model_summary=initial_model_summary,
                        ontology_candidates=ontology_candidates,
                        shortlist_limit=int(args.max_candidate_predictors),
                    )
                if bool(args.hard_ontology_constraint):
                    step04_constraint = _enforce_step04_hard_ontology(
                        step04_candidate,
                        allowed_predictor_paths=[str(item.get("predictor_path") or "") for item in ontology_candidates],
                    )
                else:
                    step04_constraint = {"applied": False}

                run_critic = (not bool(args.disable_llm)) and (actor_mode == "structured_llm_success")
                if run_critic:
                    critic_payload = {
                        **step04_bundle_for_attempt,
                        "predictor_parent_feasibility": predictor_parent_feasibility,
                    }
                    critic_llm, critic_trace = run_llm_stage_critic(
                        client=llm_client,
                        profile_id=profile_id,
                        stage="step04",
                        stage_output_payload=step04_candidate.model_dump(mode="json"),
                        evidence_bundle=critic_payload,
                        pass_threshold_0_1=float(args.critic_pass_threshold),
                        prompt_budget_tokens=int(args.prompt_budget_tokens),
                    )
                    if critic_llm is None:
                        step04_critic_review = _heuristic_stage_critic(
                            stage="step04",
                            profile_id=profile_id,
                            stage_output=step04_candidate.model_dump(mode="json"),
                            evidence_bundle=critic_payload,
                            pass_threshold_0_1=float(args.critic_pass_threshold),
                        )
                        critic_trace["mode"] = "heuristic_fallback"
                    else:
                        step04_critic_review = critic_llm
                        critic_trace["mode"] = "structured_llm_success"
                    critic_trace["attempt_index"] = attempt
                    critic_trace["constraint"] = step04_constraint
                    critic_trace["critic_decision"] = step04_critic_review.pass_decision
                    critic_trace["critic_composite_score_0_1"] = step04_critic_review.composite_score_0_1
                    step04_trace["critic_attempts"].append(critic_trace)

                    if (
                        step04_critic_review.pass_decision == "PASS"
                        or attempt >= critic_max_iterations
                    ):
                        step04_output = step04_candidate
                        step04_trace["reason"] = actor_mode
                        break
                    step04_feedback = list(step04_critic_review.feedback_for_revision or [])
                    if not step04_feedback:
                        step04_feedback = ["Improve predictor grounding and BFS breadth-vs-depth rationale."]
                    continue

                step04_critic_review = _heuristic_stage_critic(
                    stage="step04",
                    profile_id=profile_id,
                    stage_output=step04_candidate.model_dump(mode="json"),
                    evidence_bundle=step04_bundle_for_attempt,
                    pass_threshold_0_1=float(args.critic_pass_threshold),
                )
                step04_trace["critic_attempts"].append(
                    {
                        "attempt_index": attempt,
                        "mode": "heuristic_auto",
                        "constraint": step04_constraint,
                        "critic_decision": step04_critic_review.pass_decision,
                        "critic_composite_score_0_1": step04_critic_review.composite_score_0_1,
                    }
                )
                step04_output = step04_candidate
                step04_trace["reason"] = actor_mode
                step04_trace["constraint"] = step04_constraint
                break

            assert step04_output is not None
            step04_output.contract_version = str(args.contract_version)
            if step04_critic_review is not None:
                step04_trace["critic_final_decision"] = step04_critic_review.pass_decision
                step04_trace["critic_final_score_0_1"] = step04_critic_review.composite_score_0_1

            impact_matrix = load_impact_matrix(profile_dir / "impact_matrix.csv")
            previous_cycle_scores = load_previous_cycle_scores(
                profile_id=profile_id,
                history_snapshots_path=history_snapshots_path,
                memory_window=int(max(1, int(args.profile_memory_window))),
            )
            step04_trace["memory_prior_loaded"] = bool(previous_cycle_scores)
            step04_trace["memory_prior_count"] = int(len(previous_cycle_scores))
            fusion_payload = apply_fusion_to_step04_output(
                step04_output=step04_output,
                step03_output=step03_output,
                initial_model_summary=initial_model_summary,
                initial_model_payload=initial_model_payload,
                impact_matrix=impact_matrix,
                mapping_rows=mapping_rows,
                readiness_score_0_100=(evidence_bundle.get("readiness", {}) or {}).get("score_0_100"),
                previous_cycle_scores=previous_cycle_scores,
                max_candidate_predictors=int(args.max_candidate_predictors),
            )

            step04_trace["fusion_weights"] = fusion_payload.get("weights", {})
            step04_trace["fusion_predictor_count"] = len(fusion_payload.get("predictor_rankings", []))
            range_policy = enforce_step04_range_policy(
                step04_output=step04_output,
                step03_output=step03_output,
                initial_model_summary=initial_model_summary,
                ontology_candidates=ontology_candidates,
                preferred_predictor_count=int(args.preferred_predictor_count),
                preferred_criterion_count=int(args.preferred_criterion_count),
            )
            step04_trace["target_range_policy"] = range_policy
            if bool(args.hard_ontology_constraint):
                step04_trace["hard_ontology_post_fusion"] = _enforce_step04_hard_ontology(
                    step04_output,
                    allowed_predictor_paths=[str(item.get("predictor_path") or "") for item in ontology_candidates],
                )

            visual_files: List[str] = []
            if bool(args.visualize_updated_model):
                visual_files = generate_updated_model_visuals(
                    fusion_payload=fusion_payload,
                    output_dir=ensure_dir(out_profile_dir / "visuals"),
                    profile_id=profile_id,
                    dpi=int(args.visualization_dpi),
                    top_predictors=int(args.visualization_top_predictors),
                    top_edges=int(args.visualization_top_edges),
                )

            frame = serialize_step03_to_frame(
                profile_id=profile_id,
                step03=step03_output,
                predictor_impact_lookup=impact_lookup,
                predictor_impact_pct_lookup=impact_pct_lookup,
            )
            if frame.empty:
                frame = pd.DataFrame(
                    columns=[
                        "source_profile",
                        "predictor_rank",
                        "predictor",
                        "predictor_label",
                        "predictor_impact",
                        "predictor_impact_pct",
                        "priority_level",
                        "selection_reason",
                        "selected_for_intervention",
                        "selection_score_0_1",
                        "mapped_leaf_path",
                        "confidence_0_1",
                    ]
                )
            frame.to_csv(out_profile_dir / "top_treatment_target_candidates.csv", index=False)
            (out_profile_dir / "top_treatment_target_candidates.json").write_text(
                json.dumps(
                    {
                        "contract_version": str(args.contract_version),
                        "profile_id": profile_id,
                        "generated_at_local": ts(),
                        "step03_selection": step03_output.model_dump(mode="json"),
                        "step03_guardrail_review": (
                            step03_critic_review.model_dump(mode="json")
                            if step03_critic_review is not None
                            else {}
                        ),
                        "step04_updated_observation_model": step04_output.model_dump(mode="json"),
                        "step04_guardrail_review": (
                            step04_critic_review.model_dump(mode="json")
                            if step04_critic_review is not None
                            else {}
                        ),
                        "step04_nomothetic_idiographic_fusion": fusion_payload,
                        "predictor_parent_feasibility_top": predictor_parent_feasibility[:30],
                        "updated_model_visuals": visual_files,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (out_profile_dir / "step03_target_selection.json").write_text(
                json.dumps(step03_output.model_dump(mode="json"), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            (out_profile_dir / "step04_updated_observation_model.json").write_text(
                json.dumps(step04_output.model_dump(mode="json"), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            (out_profile_dir / "step04_nomothetic_idiographic_fusion.json").write_text(
                json.dumps(fusion_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            pd.DataFrame(fusion_payload.get("edge_rows", []) or []).to_csv(
                out_profile_dir / "step04_fusion_edges.csv",
                index=False,
            )
            if fusion_payload.get("matrix"):
                matrix_frame = pd.DataFrame(
                    fusion_payload.get("matrix", []),
                    index=fusion_payload.get("criterion_order", []),
                    columns=fusion_payload.get("predictor_order", []),
                )
                matrix_frame.to_csv(out_profile_dir / "step04_fusion_matrix.csv")
            pd.DataFrame(fusion_payload.get("predictor_rankings", []) or []).to_csv(
                out_profile_dir / "step04_fusion_predictor_rankings.csv",
                index=False,
            )
            (out_profile_dir / "step03_evidence_bundle.json").write_text(
                json.dumps(evidence_bundle, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            (out_profile_dir / "step03_prompt_trace.json").write_text(
                json.dumps(step03_trace, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            (out_profile_dir / "step04_prompt_trace.json").write_text(
                json.dumps(step04_trace, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            if step03_critic_review is not None:
                (out_profile_dir / "step03_guardrail_review.json").write_text(
                    json.dumps(step03_critic_review.model_dump(mode="json"), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            if step04_critic_review is not None:
                (out_profile_dir / "step04_guardrail_review.json").write_text(
                    json.dumps(step04_critic_review.model_dump(mode="json"), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            (out_profile_dir / "predictor_parent_feasibility_top30.json").write_text(
                json.dumps(predictor_parent_feasibility[:30], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            pd.DataFrame(ontology_candidates).to_csv(
                out_profile_dir / "ontology_subtree_candidates_top200.csv",
                index=False,
            )

            profile_rows.append(
                {
                    "profile_id": profile_id,
                    "n_ranked_predictors": int(len(step03_output.ranked_predictors)),
                    "n_recommended_targets": int(len(step03_output.recommended_targets)),
                    "step03_mode": str(step03_trace.get("reason")),
                    "step04_mode": str(step04_trace.get("reason")),
                    "step03_critic_decision": str(step03_trace.get("critic_final_decision", "")),
                    "step03_critic_score_0_1": step03_trace.get("critic_final_score_0_1", ""),
                    "step04_critic_decision": str(step04_trace.get("critic_final_decision", "")),
                    "step04_critic_score_0_1": step04_trace.get("critic_final_score_0_1", ""),
                    "fusion_nomothetic_weight": (fusion_payload.get("weights", {}) or {}).get("nomothetic_weight", ""),
                    "fusion_idiographic_weight": (fusion_payload.get("weights", {}) or {}).get("idiographic_weight", ""),
                    "n_visuals": int(len(visual_files)),
                    "target_policy_reason_codes": "|".join((step04_trace.get("target_range_policy", {}) or {}).get("reason_codes", []) or []),
                    "top_target": (
                        step03_output.recommended_targets[0].predictor
                        if step03_output.recommended_targets
                        else ""
                    ),
                    "top_target_score_0_1": (
                        step03_output.recommended_targets[0].score_0_1
                        if step03_output.recommended_targets
                        else ""
                    ),
                }
            )
            updated_model_rows.append(
                {
                    "profile_id": profile_id,
                    "n_shortlist_predictors": int(len(step04_output.refined_predictor_shortlist)),
                    "n_next_observation_predictors": int(len(step04_output.recommended_next_observation_predictors)),
                    "n_dropped_predictors": int(len(step04_output.dropped_predictors)),
                    "n_added_predictors": int(len(step04_output.added_predictors)),
                    "n_fusion_ranked_predictors": int(len(fusion_payload.get("predictor_rankings", []))),
                    "n_fusion_edges": int(len(fusion_payload.get("edge_rows", []))),
                }
            )

            log(
                f"[OK] {profile_id}: ranked={len(step03_output.ranked_predictors)} "
                f"selected={len(step03_output.recommended_targets)} "
                f"next_model={len(step04_output.recommended_next_observation_predictors)} "
                f"fusion_ranked={len(fusion_payload.get('predictor_rankings', []))} "
                f"visuals={len(visual_files)}"
            )
        except Exception as exc:
            failed += 1
            log(f"[ERROR] {profile_id}: {repr(exc)}")

    if profile_rows:
        pd.DataFrame(profile_rows).to_csv(output_root / "handoff_profiles.csv", index=False)
    if updated_model_rows:
        pd.DataFrame(updated_model_rows).to_csv(output_root / "updated_observation_models.csv", index=False)

    summary = {
        "contract_version": str(args.contract_version),
        "generated_at_local": ts(),
        "impact_root": str(impact_root),
        "output_root": str(output_root),
        "readiness_root": str(readiness_root),
        "network_root": str(network_root),
        "initial_model_runs_root": str(model_runs_root),
        "free_text_root": str(free_text_root),
        "mapping_ranks_csv": str(mapping_ranks_csv),
        "hyde_dense_profiles_csv": str(hyde_dense_profiles_csv) if hyde_dense_profiles_csv else "",
        "predictor_leaf_paths_json": str(predictor_leaf_paths_json),
        "predictor_list_path": str(predictor_list_path),
        "predictor_feasibility_csv": str(predictor_feasibility_csv),
        "pattern": str(args.pattern),
        "top_k": int(args.top_k),
        "min_impact": float(args.min_impact),
        "max_candidate_predictors": int(args.max_candidate_predictors),
        "llm_model": str(args.llm_model),
        "disable_llm": bool(args.disable_llm),
        "hard_ontology_constraint": bool(args.hard_ontology_constraint),
        "critic_max_iterations": int(args.critic_max_iterations),
        "critic_pass_threshold": float(args.critic_pass_threshold),
        "preferred_predictor_count": int(args.preferred_predictor_count),
        "preferred_criterion_count": int(args.preferred_criterion_count),
        "parent_feasibility_top_k": int(args.parent_feasibility_top_k),
        "prompt_budget_tokens": int(args.prompt_budget_tokens),
        "visualize_updated_model": bool(args.visualize_updated_model),
        "visualization_dpi": int(args.visualization_dpi),
        "visualization_top_predictors": int(args.visualization_top_predictors),
        "visualization_top_edges": int(args.visualization_top_edges),
        "n_profiles_attempted": len(profiles),
        "n_profiles_success": len(profile_rows),
        "n_profiles_failed": failed,
        "profiles": profile_rows,
    }
    (output_root / "handoff_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if str(args.trace_output).strip():
        trace_path = Path(args.trace_output).expanduser().resolve()
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_payload = {
            "contract_version": str(args.contract_version),
            "generated_at_local": ts(),
            "stage": "handoff",
            "n_profiles_attempted": len(profiles),
            "n_profiles_success": len(profile_rows),
            "n_profiles_failed": failed,
            "llm_model": str(args.llm_model),
            "disable_llm": bool(args.disable_llm),
            "hard_ontology_constraint": bool(args.hard_ontology_constraint),
            "critic_max_iterations": int(args.critic_max_iterations),
            "critic_pass_threshold": float(args.critic_pass_threshold),
            "prompt_budget_tokens": int(args.prompt_budget_tokens),
        }
        trace_path.write_text(json.dumps(trace_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log("========== TREATMENT-TARGET IDENTIFICATION COMPLETE ==========")
    log(f"success={len(profile_rows)} failed={failed}")
    return 0 if failed == 0 else 4


if __name__ == "__main__":
    raise SystemExit(main())
