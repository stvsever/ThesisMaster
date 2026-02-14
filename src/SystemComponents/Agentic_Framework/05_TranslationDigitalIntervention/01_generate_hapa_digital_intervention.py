#!/usr/bin/env python3
"""
01_generate_hapa_digital_intervention.py

PHOENIX Agentic Framework Step-05:
Generate a personalized HAPA-based digital intervention from integrated pipeline evidence.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[3]
AGENTIC_CORE_ROOT = REPO_ROOT / "src" / "utils" / "agentic_core"
if str(AGENTIC_CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENTIC_CORE_ROOT))

from shared import (  # noqa: E402
    PromptSection,
    StructuredLLMClient,
    best_path_match,
    decision_from_score,
    load_predictor_feasibility_table,
    load_prompt,
    normalize_path_text,
    pack_prompt_sections,
    path_similarity,
    render_prompt,
    top_parent_domains_for_bundle,
    weighted_composite,
)


LIMITATION_LLM_UNAVAILABLE = "Limitation recorded: LLM unavailable, so it’s impact-driven only."


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{ts()}] {message}", flush=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _init_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def _save_figure_multi(fig: plt.Figure, png_path: Path, *, dpi: int, metadata: Dict[str, Any]) -> List[str]:
    ensure_dir(png_path.parent)
    svg_path = png_path.with_suffix(".svg")
    pdf_path = png_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    figure_meta = dict(metadata)
    figure_meta["generated_at_local"] = ts()
    figure_meta["files"] = [str(png_path), str(svg_path), str(pdf_path)]
    png_path.with_suffix(".figure.json").write_text(
        json.dumps(figure_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return [str(png_path), str(svg_path), str(pdf_path), str(png_path.with_suffix(".figure.json"))]


def _plot_ranked_bars(
    *,
    rows: Sequence[Dict[str, Any]],
    label_key: str,
    score_key: str,
    title: str,
    out_png: Path,
    dpi: int,
    top_n: int = 10,
) -> List[str]:
    if not rows:
        return []
    data = sorted(list(rows), key=lambda r: float(r.get(score_key, 0.0)), reverse=True)[: max(1, int(top_n))]
    labels = [str(item.get(label_key) or "")[:80] for item in data]
    scores = [float(item.get(score_key, 0.0)) for item in data]
    fig = plt.figure(figsize=(10, max(4.5, 0.45 * len(labels))))
    ax = fig.add_subplot(111)
    y = np.arange(len(labels))
    ax.barh(y, scores, color="#2a9d8f")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Score (0-1)")
    ax.set_title(title)
    return _save_figure_multi(
        fig,
        out_png,
        dpi=dpi,
        metadata={
            "plot_type": "ranked_horizontal_bars",
            "label_key": label_key,
            "score_key": score_key,
            "n_items": len(data),
        },
    )


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            if df.shape[1] > 1 or sep == ",":
                return df
        except Exception:
            pass
    return pd.read_csv(path, engine="python")


def _score_0_1(raw: Any) -> float:
    try:
        value = float(raw)
    except Exception:
        return 0.0
    if not np.isfinite(value):
        return 0.0
    if value > 1.0:
        value = value / 1000.0
    return float(max(0.0, min(1.0, value)))


def _tokenize(text: str) -> List[str]:
    return [item.lower() for item in re.findall(r"[a-zA-Z0-9_]+", str(text or "")) if len(item) > 1]


def _token_jaccard(a: str, b: str) -> float:
    ta = set(_tokenize(a))
    tb = set(_tokenize(b))
    if not ta or not tb:
        return 0.0
    return float(len(ta.intersection(tb)) / max(1, len(ta.union(tb))))


def _normalize_key_path(text: str) -> str:
    return normalize_path_text(str(text or "")).lower()


def _parse_profile_number(profile_id: str) -> Optional[str]:
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
    evidence_bundle: Dict[str, Any],
) -> Dict[str, str]:
    from_evidence = evidence_bundle.get("free_text", {}) if isinstance(evidence_bundle, dict) else {}
    complaint = str(from_evidence.get("complaint_text") or "").strip()
    person = str(from_evidence.get("person_text") or "").strip()
    context = str(from_evidence.get("context_text") or "").strip()
    if complaint and person and context:
        return {
            "complaint_text": complaint,
            "person_text": person,
            "context_text": context,
        }
    number = _parse_profile_number(profile_id)
    if number is None:
        return {
            "complaint_text": complaint,
            "person_text": person,
            "context_text": context,
        }
    return {
        "complaint_text": complaint or complaints.get(f"pseudoprofile_FTC_ID{number}", ""),
        "person_text": person or person_profiles.get(f"pseudoprofile_person_ID{number}", ""),
        "context_text": context or context_profiles.get(f"pseudoprofile_context_ID{number}", ""),
    }


def discover_profiles(handoff_root: Path, pattern: str, max_profiles: int) -> List[Path]:
    out: List[Path] = []
    if not handoff_root.exists():
        return out
    for child in sorted(handoff_root.iterdir()):
        if not child.is_dir():
            continue
        if pattern and pattern not in child.name:
            continue
        if not (child / "step04_updated_observation_model.json").exists():
            continue
        out.append(child)
    if max_profiles > 0:
        out = out[: max_profiles]
    return out


def summarize_network_metrics(network_profile_root: Path) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    comparison = read_json(network_profile_root / "comparison_summary.json")
    if comparison:
        summary["comparison_summary"] = {
            "n_rows": comparison.get("n_rows"),
            "n_vars": comparison.get("n_vars"),
            "predictors": comparison.get("predictors", []),
            "criteria": comparison.get("criteria", []),
            "execution_plan": comparison.get("execution_plan", {}),
            "method_status": comparison.get("method_status", {}),
        }
    metrics_root = network_profile_root / "network_metrics"
    predictor_importance = safe_read_csv(metrics_root / "predictor_importance_tv.csv")
    if not predictor_importance.empty:
        summary["predictor_importance_tv_top"] = (
            predictor_importance.sort_values(
                by="out_strength_criteria_mean",
                ascending=False,
            )
            .head(25)
            .fillna("")
            .to_dict(orient="records")
        )
    criterion_dependence = safe_read_csv(metrics_root / "criterion_dependence_tv.csv")
    if not criterion_dependence.empty:
        summary["criterion_dependence_tv_top"] = criterion_dependence.head(25).fillna("").to_dict(orient="records")
    change_points = read_json(network_profile_root / "method 1/numerical outputs/tv_network_change_points.json")
    if change_points:
        summary["tv_network_change_points"] = change_points
    return summary


def collect_predictor_candidates(
    profile_id: str,
    step03_payload: Dict[str, Any],
    step04_payload: Dict[str, Any],
    fusion_payload: Dict[str, Any],
    evidence_bundle: Dict[str, Any],
) -> List[Dict[str, Any]]:
    initial_model = evidence_bundle.get("initial_model", {}) if isinstance(evidence_bundle, dict) else {}
    predictor_summary = initial_model.get("predictor_summary", []) if isinstance(initial_model, dict) else []
    predictor_map: Dict[str, Dict[str, str]] = {}
    for row in predictor_summary:
        pid = str(row.get("var_id") or "").strip()
        predictor_map[pid] = {
            "label": str(row.get("label") or pid).strip(),
            "path": normalize_path_text(str(row.get("mapped_leaf_full_path") or "")),
        }

    ranked_impact: Dict[str, float] = {}
    for row in step03_payload.get("ranked_predictors", []) or []:
        ranked_impact[str(row.get("predictor") or "")] = _score_0_1(row.get("score_0_1"))

    candidates: Dict[str, Dict[str, Any]] = {}

    for row in step03_payload.get("recommended_targets", []) or []:
        predictor = str(row.get("predictor") or "").strip()
        mapped_path = normalize_path_text(str(row.get("mapped_leaf_path") or predictor_map.get(predictor, {}).get("path") or ""))
        key = mapped_path.lower() if mapped_path else predictor.lower()
        if not key:
            continue
        candidates[key] = {
            "profile_id": profile_id,
            "predictor": predictor,
            "predictor_label": str(row.get("predictor_label") or predictor_map.get(predictor, {}).get("label") or predictor),
            "predictor_path": mapped_path,
            "source": "step03_recommended_targets",
            "priority_0_1": max(_score_0_1(row.get("score_0_1")), 0.20),
            "linked_criteria_ids": step04_payload.get("retained_criteria_ids", []) or [],
        }

    for rank, predictor_path in enumerate(step04_payload.get("recommended_next_observation_predictors", []) or [], start=1):
        npath = normalize_path_text(str(predictor_path or ""))
        if not npath:
            continue
        key = npath.lower()
        base = candidates.get(key, {})
        candidates[key] = {
            "profile_id": profile_id,
            "predictor": str(base.get("predictor") or ""),
            "predictor_label": str(base.get("predictor_label") or npath.split(" / ")[-1]),
            "predictor_path": npath,
            "source": "step04_updated_model",
            "priority_0_1": max(float(base.get("priority_0_1", 0.0)), max(0.15, 1.0 - (rank - 1) / 12.0)),
            "linked_criteria_ids": step04_payload.get("retained_criteria_ids", []) or [],
        }

    for rank, row in enumerate(fusion_payload.get("predictor_rankings", []) or [], start=1):
        npath = normalize_path_text(str(row.get("predictor_path") or ""))
        if not npath:
            continue
        key = npath.lower()
        fused_score = _score_0_1(row.get("fused_score_0_1"))
        prior = _score_0_1(row.get("prior_score_0_1"))
        score = max(fused_score, 0.60 * fused_score + 0.40 * prior)
        base = candidates.get(key, {})
        candidates[key] = {
            "profile_id": profile_id,
            "predictor": str(base.get("predictor") or ""),
            "predictor_label": str(base.get("predictor_label") or npath.split(" / ")[-1]),
            "predictor_path": npath,
            "source": "step04_fusion_ranking",
            "priority_0_1": max(float(base.get("priority_0_1", 0.0)), score),
            "linked_criteria_ids": step04_payload.get("retained_criteria_ids", []) or [],
        }

    if not candidates:
        for predictor, score in ranked_impact.items():
            meta = predictor_map.get(predictor, {})
            path = normalize_path_text(meta.get("path") or predictor)
            key = path.lower()
            candidates[key] = {
                "profile_id": profile_id,
                "predictor": predictor,
                "predictor_label": meta.get("label") or predictor,
                "predictor_path": path,
                "source": "step03_ranked_fallback",
                "priority_0_1": max(score, 0.10),
                "linked_criteria_ids": step04_payload.get("retained_criteria_ids", []) or [],
            }

    rows = sorted(candidates.values(), key=lambda r: float(r.get("priority_0_1", 0.0)), reverse=True)
    return rows


def _barrier_parent_domain(barrier_path: str) -> str:
    parts = [p.strip() for p in normalize_path_text(barrier_path).split(" / ") if p.strip()]
    if len(parts) >= 2:
        return " / ".join(parts[:2])
    if parts:
        return parts[0]
    return "UNKNOWN"


def barrier_candidates_from_predictors(
    predictor_candidates: Sequence[Dict[str, Any]],
    predictor_to_barrier_df: pd.DataFrame,
    top_n_per_predictor: int,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    aggregate: Dict[str, Dict[str, Any]] = {}
    if predictor_to_barrier_df.empty or not predictor_candidates:
        return pd.DataFrame(), aggregate

    for predictor in predictor_candidates:
        predictor_path = normalize_path_text(str(predictor.get("predictor_path") or ""))
        predictor_label = str(predictor.get("predictor_label") or predictor_path).strip()
        predictor_weight = _score_0_1(predictor.get("priority_0_1"))
        if not predictor_path:
            continue
        local: List[Dict[str, Any]] = []
        for _, map_row in predictor_to_barrier_df.iterrows():
            map_predictor_path = normalize_path_text(str(map_row.get("predictor_full_path") or ""))
            if not map_predictor_path:
                continue
            sim_path = path_similarity(predictor_path, map_predictor_path)
            sim_token = _token_jaccard(predictor_path, map_predictor_path)
            similarity = max(sim_path, sim_token)
            if similarity < 0.12:
                continue
            map_score = _score_0_1(map_row.get("score"))
            barrier_path = normalize_path_text(str(map_row.get("barrier_full_path") or ""))
            barrier_name = str(map_row.get("barrier_name") or "").strip() or barrier_path.split(" / ")[-1]
            contribution = float(max(0.0, predictor_weight * map_score * similarity))
            local.append(
                {
                    "predictor_path": predictor_path,
                    "predictor_label": predictor_label,
                    "predictor_weight_0_1": predictor_weight,
                    "map_predictor_path": map_predictor_path,
                    "path_similarity_0_1": similarity,
                    "mapping_score_0_1": map_score,
                    "barrier_name": barrier_name,
                    "barrier_path": barrier_path,
                    "barrier_parent_domain": _barrier_parent_domain(barrier_path),
                    "predictor_barrier_contribution_0_1": contribution,
                    "source_mapping": "predictor_to_barrier",
                }
            )
        local = sorted(local, key=lambda x: float(x["predictor_barrier_contribution_0_1"]), reverse=True)[: max(1, int(top_n_per_predictor))]
        for rank, item in enumerate(local, start=1):
            item["rank_within_predictor"] = rank
            rows.append(item)
            key = _normalize_key_path(item["barrier_path"])
            agg = aggregate.setdefault(
                key,
                {
                    "barrier_name": item["barrier_name"],
                    "barrier_path": item["barrier_path"],
                    "barrier_parent_domain": item["barrier_parent_domain"],
                    "predictor_score_0_1": 0.0,
                    "supporting_predictors": set(),
                },
            )
            agg["predictor_score_0_1"] = max(float(agg["predictor_score_0_1"]), float(item["predictor_barrier_contribution_0_1"]))
            agg["supporting_predictors"].add(item["predictor_path"])

    for value in aggregate.values():
        value["supporting_predictors"] = sorted(list(value.get("supporting_predictors", set())))
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(
            by=["predictor_path", "predictor_barrier_contribution_0_1"],
            ascending=[True, False],
        ).reset_index(drop=True)
    return frame, aggregate


def _barrier_scores_from_text_mapping(
    text: str,
    mapping_df: pd.DataFrame,
    *,
    text_col_path: str,
    text_col_name: str,
    source: str,
) -> Dict[str, Dict[str, Any]]:
    text_value = str(text or "").strip()
    out: Dict[str, Dict[str, Any]] = {}
    if mapping_df.empty:
        return out
    for _, row in mapping_df.iterrows():
        barrier_path = normalize_path_text(str(row.get("barrier_full_path") or ""))
        if not barrier_path:
            continue
        barrier_name = str(row.get("barrier_name") or "").strip() or barrier_path.split(" / ")[-1]
        map_entity = normalize_path_text(str(row.get(text_col_path) or ""))
        map_name = str(row.get(text_col_name) or "").strip()
        map_score = _score_0_1(row.get("score"))
        sim = max(_token_jaccard(text_value, map_entity), _token_jaccard(text_value, map_name))
        blended = map_score * (0.25 + 0.75 * sim) if text_value else 0.30 * map_score
        key = _normalize_key_path(barrier_path)
        current = out.get(key)
        payload = {
            "barrier_name": barrier_name,
            "barrier_path": barrier_path,
            "barrier_parent_domain": _barrier_parent_domain(barrier_path),
            f"{source}_score_0_1": float(max(0.0, min(1.0, blended))),
            f"{source}_evidence_path": map_entity,
            f"{source}_text_similarity_0_1": sim,
        }
        if current is None or payload[f"{source}_score_0_1"] > float(current.get(f"{source}_score_0_1", 0.0)):
            out[key] = payload
    return out


def combine_barrier_evidence(
    predictor_barriers: Dict[str, Dict[str, Any]],
    profile_barriers: Dict[str, Dict[str, Any]],
    context_barriers: Dict[str, Dict[str, Any]],
    complaint_text: str,
    *,
    top_k: int,
) -> List[Dict[str, Any]]:
    all_keys = set(predictor_barriers.keys()) | set(profile_barriers.keys()) | set(context_barriers.keys())
    rows: List[Dict[str, Any]] = []
    for key in all_keys:
        pred = predictor_barriers.get(key, {})
        prof = profile_barriers.get(key, {})
        ctx = context_barriers.get(key, {})
        barrier_name = str(pred.get("barrier_name") or prof.get("barrier_name") or ctx.get("barrier_name") or "UnknownBarrier")
        barrier_path = str(pred.get("barrier_path") or prof.get("barrier_path") or ctx.get("barrier_path") or "")
        predictor_score = _score_0_1(pred.get("predictor_score_0_1"))
        profile_score = _score_0_1(prof.get("profile_score_0_1"))
        context_score = _score_0_1(ctx.get("context_score_0_1"))
        complaint_match = _token_jaccard(complaint_text, barrier_name + " " + barrier_path)
        total = float(max(0.0, min(1.0, (0.60 * predictor_score) + (0.20 * profile_score) + (0.15 * context_score) + (0.05 * complaint_match))))
        rows.append(
            {
                "barrier_name": barrier_name,
                "barrier_path": barrier_path,
                "barrier_parent_domain": _barrier_parent_domain(barrier_path),
                "total_score_0_1": total,
                "predictor_score_0_1": predictor_score,
                "profile_score_0_1": profile_score,
                "context_score_0_1": context_score,
                "complaint_match_0_1": complaint_match,
                "supporting_predictors": pred.get("supporting_predictors", []),
                "profile_evidence_path": prof.get("profile_evidence_path", ""),
                "context_evidence_path": ctx.get("context_evidence_path", ""),
            }
        )
    rows = sorted(rows, key=lambda x: float(x["total_score_0_1"]), reverse=True)
    return rows[: max(1, int(top_k))]


def select_coping_candidates(
    selected_barriers: Sequence[Dict[str, Any]],
    coping_to_barrier_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    detail_rows: List[Dict[str, Any]] = []
    aggregate: Dict[str, Dict[str, Any]] = {}
    if coping_to_barrier_df.empty or not selected_barriers:
        return pd.DataFrame(), []

    for barrier_rank, barrier in enumerate(selected_barriers, start=1):
        barrier_path = normalize_path_text(str(barrier.get("barrier_path") or ""))
        barrier_name = str(barrier.get("barrier_name") or barrier_path).strip()
        barrier_score = _score_0_1(barrier.get("total_score_0_1"))
        local: List[Dict[str, Any]] = []
        for _, row in coping_to_barrier_df.iterrows():
            map_barrier_path = normalize_path_text(str(row.get("barrier_full_path") or ""))
            map_barrier_name = str(row.get("barrier_name") or "").strip()
            sim = max(path_similarity(barrier_path, map_barrier_path), _token_jaccard(barrier_name, map_barrier_name))
            if sim < 0.20:
                continue
            mapping_score = _score_0_1(row.get("score"))
            coping_path = normalize_path_text(str(row.get("coping_full_path") or ""))
            coping_name = str(row.get("coping_name") or coping_path.split(" / ")[-1]).strip()
            contribution = float(max(0.0, min(1.0, barrier_score * mapping_score * sim)))
            local.append(
                {
                    "barrier_rank": barrier_rank,
                    "barrier_name": barrier_name,
                    "barrier_path": barrier_path,
                    "barrier_score_0_1": barrier_score,
                    "coping_name": coping_name,
                    "coping_path": coping_path,
                    "mapping_score_0_1": mapping_score,
                    "barrier_match_0_1": sim,
                    "coping_contribution_0_1": contribution,
                }
            )
        local = sorted(local, key=lambda x: float(x["coping_contribution_0_1"]), reverse=True)[:12]
        for row in local:
            detail_rows.append(row)
            key = _normalize_key_path(row["coping_path"])
            agg = aggregate.setdefault(
                key,
                {
                    "coping_name": row["coping_name"],
                    "coping_path": row["coping_path"],
                    "score_0_1": 0.0,
                    "linked_barriers": set(),
                },
            )
            agg["score_0_1"] = max(float(agg["score_0_1"]), float(row["coping_contribution_0_1"]))
            agg["linked_barriers"].add(row["barrier_name"])

    top = sorted(aggregate.values(), key=lambda x: float(x["score_0_1"]), reverse=True)
    result: List[Dict[str, Any]] = []
    for item in top:
        result.append(
            {
                "coping_name": item["coping_name"],
                "coping_path": item["coping_path"],
                "score_0_1": float(item["score_0_1"]),
                "linked_barriers": sorted(list(item["linked_barriers"])),
            }
        )
    detail = pd.DataFrame(detail_rows)
    if not detail.empty:
        detail = detail.sort_values(by=["barrier_rank", "coping_contribution_0_1"], ascending=[True, False]).reset_index(drop=True)
    return detail, result


class InterventionTargetModel(BaseModel):
    predictor: str
    predictor_label: str
    predictor_path: str
    linked_criteria_ids: List[str] = Field(default_factory=list)
    rationale: str
    priority_0_1: float = Field(ge=0.0, le=1.0)


class InterventionBarrierModel(BaseModel):
    barrier_name: str
    barrier_path: str
    barrier_parent_domain: str
    score_0_1: float = Field(ge=0.0, le=1.0)
    rationale: str
    evidence_refs: List[str] = Field(default_factory=list)


class InterventionCopingModel(BaseModel):
    coping_name: str
    coping_path: str
    linked_barriers: List[str] = Field(default_factory=list)
    score_0_1: float = Field(ge=0.0, le=1.0)
    rationale: str


class HAPAComponentStepModel(BaseModel):
    component: str
    objective: str
    actions: List[str] = Field(default_factory=list)
    digital_delivery: str
    measurement_signals: List[str] = Field(default_factory=list)


class PhaseStepModel(BaseModel):
    phase: str
    time_window: str
    primary_goal: str
    concrete_actions: List[str] = Field(default_factory=list)
    linked_targets: List[str] = Field(default_factory=list)
    linked_barriers: List[str] = Field(default_factory=list)


class Step05InterventionModel(BaseModel):
    contract_version: str = "1.0.0"
    profile_id: str
    intervention_title: str
    user_friendly_summary: str
    clinical_case_formulation: str
    selected_treatment_targets: List[InterventionTargetModel] = Field(default_factory=list)
    selected_barriers: List[InterventionBarrierModel] = Field(default_factory=list)
    selected_coping_strategies: List[InterventionCopingModel] = Field(default_factory=list)
    hapa_component_plan: List[HAPAComponentStepModel] = Field(default_factory=list)
    phased_delivery_plan: List[PhaseStepModel] = Field(default_factory=list)
    personalized_message: str
    monitoring_plan: List[str] = Field(default_factory=list)
    safety_notes: List[str] = Field(default_factory=list)
    confidence_0_1: float = Field(ge=0.0, le=1.0)
    limitations: List[str] = Field(default_factory=list)


class Step05CriticReviewModel(BaseModel):
    contract_version: str = "1.0.0"
    profile_id: str
    stage: str = "step05"
    pass_decision: str
    composite_score_0_1: float = Field(ge=0.0, le=1.0)
    weighted_subscores_0_1: Dict[str, float] = Field(default_factory=dict)
    critical_issues: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    feedback_for_revision: List[str] = Field(default_factory=list)
    evidence_gaps: List[str] = Field(default_factory=list)
    confidence_0_1: float = Field(default=0.5, ge=0.0, le=1.0)


def _step05_critic_weights() -> Dict[str, float]:
    return {
        "reasoning_quality": 0.17,
        "evidence_grounding": 0.21,
        "hapa_consistency": 0.16,
        "medical_safety": 0.16,
        "personalization_context_fit": 0.12,
        "regulatory_ethical_alignment": 0.08,
        "intervention_feasibility": 0.10,
    }


def _heuristic_step05_critic(
    *,
    profile_id: str,
    intervention: Step05InterventionModel,
    evidence_bundle: Dict[str, Any],
    pass_threshold_0_1: float,
) -> Step05CriticReviewModel:
    targets = intervention.selected_treatment_targets or []
    barriers = intervention.selected_barriers or []
    coping = intervention.selected_coping_strategies or []
    hapa = intervention.hapa_component_plan or []
    phased = intervention.phased_delivery_plan or []
    has_monitoring = len(intervention.monitoring_plan or []) > 0
    has_safety = len(intervention.safety_notes or []) > 0
    free_text = evidence_bundle.get("free_text", {}) if isinstance(evidence_bundle, dict) else {}
    has_personalization = bool(str((free_text or {}).get("person_text", "")).strip() or str((free_text or {}).get("context_text", "")).strip())

    subscores = {
        "reasoning_quality": 0.70 if intervention.clinical_case_formulation.strip() else 0.50,
        "evidence_grounding": 0.35 + 0.65 * min(1.0, len(targets) / 3.0),
        "hapa_consistency": 0.35 + 0.65 * min(1.0, len(hapa) / 4.0),
        "medical_safety": 0.75 if has_safety else 0.45,
        "personalization_context_fit": 0.72 if has_personalization else 0.50,
        "regulatory_ethical_alignment": 0.62 if has_monitoring else 0.48,
        "intervention_feasibility": 0.25 + 0.75 * min(1.0, len(barriers) / 10.0),
    }
    weighted = weighted_composite(subscores=subscores, weights=_step05_critic_weights())
    composite = float(weighted.get("composite_score_0_1", 0.0))
    decision = decision_from_score(score_0_1=composite, threshold_0_1=pass_threshold_0_1)
    critical_issues: List[str] = []
    if len(targets) < 2:
        critical_issues.append("Too few treatment targets selected for robust intervention coverage.")
    if len(barriers) < 3:
        critical_issues.append("Barrier set is too narrow for HAPA coping translation.")
    if len(coping) < 3:
        critical_issues.append("Coping set is too narrow to support barrier diversity.")
    if len(hapa) < 4:
        critical_issues.append("HAPA component coverage is incomplete.")
    if not has_safety:
        critical_issues.append("Safety/escalation notes missing.")
    if critical_issues:
        decision = "REVISE"
    return Step05CriticReviewModel(
        profile_id=profile_id,
        pass_decision=decision,
        composite_score_0_1=composite,
        weighted_subscores_0_1={k: float(v) for k, v in weighted.get("subscores_0_1", {}).items()},
        critical_issues=critical_issues,
        strengths=["Heuristic critic fallback used due unavailable or disabled LLM critic."],
        feedback_for_revision=(
            []
            if decision == "PASS"
            else [
                "Improve HAPA component completeness and strengthen target-barrier-coping traceability.",
                "Increase safety and monitoring specificity using observed profile evidence.",
            ]
        ),
        evidence_gaps=[],
        confidence_0_1=0.56,
    )


def _enforce_step05_hard_ontology(
    intervention: Step05InterventionModel,
    *,
    allowed_predictor_paths: Sequence[str],
    allowed_barrier_paths: Sequence[str],
    allowed_coping_paths: Sequence[str],
) -> Dict[str, Any]:
    changed = 0
    dropped_targets = 0
    dropped_barriers = 0
    dropped_coping = 0

    normalized_predictors = [normalize_path_text(path) for path in allowed_predictor_paths if normalize_path_text(path)]
    normalized_barriers = [normalize_path_text(path) for path in allowed_barrier_paths if normalize_path_text(path)]
    normalized_coping = [normalize_path_text(path) for path in allowed_coping_paths if normalize_path_text(path)]

    if normalized_predictors:
        filtered_targets: List[InterventionTargetModel] = []
        for row in intervention.selected_treatment_targets:
            candidate = normalize_path_text(str(row.predictor_path or row.predictor_label or row.predictor))
            matched, similarity = best_path_match(candidate, normalized_predictors)
            if not matched or similarity < 0.35:
                dropped_targets += 1
                continue
            if matched != normalize_path_text(row.predictor_path):
                row.predictor_path = matched
                changed += 1
            filtered_targets.append(row)
        intervention.selected_treatment_targets = filtered_targets

    if normalized_barriers:
        filtered_barriers: List[InterventionBarrierModel] = []
        for row in intervention.selected_barriers:
            candidate = normalize_path_text(str(row.barrier_path or row.barrier_name))
            matched, similarity = best_path_match(candidate, normalized_barriers)
            if not matched or similarity < 0.35:
                dropped_barriers += 1
                continue
            if matched != normalize_path_text(row.barrier_path):
                row.barrier_path = matched
                row.barrier_parent_domain = _barrier_parent_domain(matched)
                changed += 1
            filtered_barriers.append(row)
        intervention.selected_barriers = filtered_barriers

    if normalized_coping:
        filtered_coping: List[InterventionCopingModel] = []
        for row in intervention.selected_coping_strategies:
            candidate = normalize_path_text(str(row.coping_path or row.coping_name))
            matched, similarity = best_path_match(candidate, normalized_coping)
            if not matched or similarity < 0.35:
                dropped_coping += 1
                continue
            if matched != normalize_path_text(row.coping_path):
                row.coping_path = matched
                changed += 1
            filtered_coping.append(row)
        intervention.selected_coping_strategies = filtered_coping

    return {
        "changed_count": int(changed),
        "dropped_targets": int(dropped_targets),
        "dropped_barriers": int(dropped_barriers),
        "dropped_coping": int(dropped_coping),
    }


def heuristic_step05_intervention(
    profile_id: str,
    free_text: Dict[str, str],
    criteria_ids: Sequence[str],
    predictor_candidates: Sequence[Dict[str, Any]],
    selected_barriers: Sequence[Dict[str, Any]],
    coping_candidates: Sequence[Dict[str, Any]],
) -> Step05InterventionModel:
    ranked_targets = sorted(
        list(predictor_candidates),
        key=lambda row: (
            0 if str(row.get("predictor") or "").strip() else 1,
            -_score_0_1(row.get("priority_0_1")),
        ),
    )
    known_targets = [row for row in ranked_targets if str(row.get("predictor") or "").strip()]
    top_targets = known_targets[:3]
    if len(top_targets) < 2:
        for row in ranked_targets:
            if row in top_targets:
                continue
            top_targets.append(row)
            if len(top_targets) >= 2:
                break
    top_targets = top_targets[:3]
    barriers = list(selected_barriers)[:10]
    copings = list(coping_candidates)[:15]

    target_models: List[InterventionTargetModel] = []
    for row in top_targets:
        target_models.append(
            InterventionTargetModel(
                predictor=str(row.get("predictor") or ""),
                predictor_label=str(row.get("predictor_label") or row.get("predictor_path") or "Predictor"),
                predictor_path=str(row.get("predictor_path") or ""),
                linked_criteria_ids=list(criteria_ids),
                rationale="Selected by fused Step-03/Step-04 ranking and ontology-constrained relevance signals.",
                priority_0_1=_score_0_1(row.get("priority_0_1")),
            )
        )

    barrier_models: List[InterventionBarrierModel] = []
    for row in barriers:
        barrier_models.append(
            InterventionBarrierModel(
                barrier_name=str(row.get("barrier_name") or ""),
                barrier_path=str(row.get("barrier_path") or ""),
                barrier_parent_domain=str(row.get("barrier_parent_domain") or ""),
                score_0_1=_score_0_1(row.get("total_score_0_1")),
                rationale="High combined plausibility from predictor, profile, and context-to-barrier mapping evidence.",
                evidence_refs=[
                    "predictor_to_barrier",
                    "profile_to_barrier",
                    "context_to_barrier",
                ],
            )
        )

    coping_models: List[InterventionCopingModel] = []
    for row in copings:
        coping_models.append(
            InterventionCopingModel(
                coping_name=str(row.get("coping_name") or ""),
                coping_path=str(row.get("coping_path") or ""),
                linked_barriers=list(row.get("linked_barriers", [])),
                score_0_1=_score_0_1(row.get("score_0_1")),
                rationale="Ranked via coping-to-barrier mapping with barrier-weighted relevance.",
            )
        )

    complaint = str(free_text.get("complaint_text") or "").strip()
    person = str(free_text.get("person_text") or "").strip()
    context = str(free_text.get("context_text") or "").strip()

    hapa_components = [
        HAPAComponentStepModel(
            component="Motivation: Risk perception and outcome expectancy",
            objective="Translate symptom burden into concrete and meaningful change targets.",
            actions=[
                "Reflect on current symptom-impact pattern using concise daily feedback.",
                "Link expected gains to personally meaningful outcomes.",
            ],
            digital_delivery="Short daily check-in + reflective prompt card.",
            measurement_signals=["criterion intensity trend", "engagement with reflection prompt"],
        ),
        HAPAComponentStepModel(
            component="Motivation: Task self-efficacy and intention",
            objective="Increase confidence and commitment to start behavior change.",
            actions=[
                "Define one small, feasible behavior target per day.",
                "Use confidence scaling (0-10) and adjust task size when confidence <7.",
            ],
            digital_delivery="Adaptive intent prompt with confidence slider.",
            measurement_signals=["self-efficacy rating", "intention completion"],
        ),
        HAPAComponentStepModel(
            component="Volition: Action planning and coping planning",
            objective="Specify when/where/how actions happen and how barriers are handled.",
            actions=[
                "Create if-then plans for the top barriers.",
                "Attach one coping option to each prioritized barrier.",
            ],
            digital_delivery="If-then planner with barrier-specific coping menu.",
            measurement_signals=["plan completion", "coping plan coverage"],
        ),
        HAPAComponentStepModel(
            component="Volition: Action control and maintenance",
            objective="Sustain execution through monitoring, feedback, and rapid recovery after setbacks.",
            actions=[
                "Run micro self-monitoring loop with daily progress feedback.",
                "Apply restart protocol after missed actions.",
            ],
            digital_delivery="Daily progress dashboard + restart nudges.",
            measurement_signals=["adherence rate", "time-to-restart after lapse"],
        ),
    ]

    phased = [
        PhaseStepModel(
            phase="Week 1",
            time_window="Days 1-7",
            primary_goal="Stabilize commitment and remove the highest-friction barriers.",
            concrete_actions=[
                "Focus on one primary target with one backup action.",
                "Use one coping routine linked to the top barrier each day.",
            ],
            linked_targets=[item.predictor_label for item in target_models[:2]],
            linked_barriers=[item.barrier_name for item in barrier_models[:3]],
        ),
        PhaseStepModel(
            phase="Week 2",
            time_window="Days 8-14",
            primary_goal="Expand to multi-target regulation while preserving adherence.",
            concrete_actions=[
                "Add second target once week-1 adherence is stable.",
                "Use daily review to adapt coping option choice.",
            ],
            linked_targets=[item.predictor_label for item in target_models[:3]],
            linked_barriers=[item.barrier_name for item in barrier_models[:5]],
        ),
    ]

    summary_line = (
        "Intervention prioritizes high-impact predictors, barrier-focused coping, and HAPA-consistent phased support."
    )
    personalized_message = (
        f"Based on your recent pattern, we will focus first on {target_models[0].predictor_label if target_models else 'your top daily lever'}, "
        "using short actions that fit your day. "
        "When barriers appear, follow the linked coping option immediately and restart quickly after misses."
    )
    if person:
        personalized_message += f" Personal context considered: {person[:180]}."
    if context:
        personalized_message += f" Environmental context considered: {context[:180]}."

    return Step05InterventionModel(
        profile_id=profile_id,
        intervention_title="PHOENIX HAPA-Guided Digital Intervention (Heuristic Fallback)",
        user_friendly_summary=summary_line,
        clinical_case_formulation=f"Complaint-focused formulation: {complaint[:600]}",
        selected_treatment_targets=target_models,
        selected_barriers=barrier_models,
        selected_coping_strategies=coping_models,
        hapa_component_plan=hapa_components,
        phased_delivery_plan=phased,
        personalized_message=personalized_message,
        monitoring_plan=[
            "Track criterion changes daily and compare against baseline week.",
            "Track adherence to target actions and coping-plan usage.",
            "Track barrier recurrence and time-to-recovery after lapses.",
        ],
        safety_notes=[
            "Decision-support output only; not diagnostic.",
            "Escalate to qualified care when acute risk or rapid deterioration is detected.",
        ],
        confidence_0_1=0.52,
        limitations=[
            LIMITATION_LLM_UNAVAILABLE,
            "Heuristic mode used ranked ontology evidence without structured LLM synthesis.",
        ],
    )


def run_llm_step05(
    *,
    client: StructuredLLMClient,
    profile_id: str,
    evidence_bundle: Dict[str, Any],
    prompt_budget_tokens: int,
) -> Tuple[Optional[Step05InterventionModel], Dict[str, Any]]:
    system_template = load_prompt("step05_hapa_intervention_system.md")
    user_template = load_prompt("step05_hapa_intervention_user_template.md")

    sections = [
        PromptSection("meta", json.dumps(evidence_bundle.get("meta", {}), ensure_ascii=False, indent=2), priority=1),
        PromptSection("free_text", json.dumps(evidence_bundle.get("free_text", {}), ensure_ascii=False, indent=2), priority=2),
        PromptSection("readiness", json.dumps(evidence_bundle.get("readiness", {}), ensure_ascii=False, indent=2), priority=3),
        PromptSection("network", json.dumps(evidence_bundle.get("network", {}), ensure_ascii=False, indent=2), priority=4),
        PromptSection("impact", json.dumps(evidence_bundle.get("impact", {}), ensure_ascii=False, indent=2), priority=5),
        PromptSection("step03_output", json.dumps(evidence_bundle.get("step03_output", {}), ensure_ascii=False, indent=2), priority=6),
        PromptSection("step04_output", json.dumps(evidence_bundle.get("step04_output", {}), ensure_ascii=False, indent=2), priority=7),
        PromptSection("targets", json.dumps(evidence_bundle.get("target_candidates", []), ensure_ascii=False, indent=2), priority=8),
        PromptSection("barriers", json.dumps(evidence_bundle.get("barrier_candidates", {}), ensure_ascii=False, indent=2), priority=9),
        PromptSection("coping", json.dumps(evidence_bundle.get("coping_candidates", {}), ensure_ascii=False, indent=2), priority=10),
        PromptSection("predictor_parent_feasibility", json.dumps(evidence_bundle.get("predictor_parent_feasibility", []), ensure_ascii=False, indent=2), priority=11),
        PromptSection("critic_feedback", json.dumps(evidence_bundle.get("critic_feedback", {}), ensure_ascii=False, indent=2), priority=12),
    ]
    packed = pack_prompt_sections(
        sections,
        max_tokens=int(prompt_budget_tokens),
        reserve_tokens=4000,
        model=client.model,
    )
    user_prompt = render_prompt(user_template, {"EVIDENCE_BUNDLE_JSON": packed.text})

    llm_result = client.generate_structured(
        system_prompt=system_template,
        user_prompt=user_prompt,
        schema_model=Step05InterventionModel,
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
    parsed = Step05InterventionModel.model_validate(llm_result.parsed)
    parsed.profile_id = profile_id
    return parsed, trace


def run_llm_step05_critic(
    *,
    client: StructuredLLMClient,
    profile_id: str,
    intervention_payload: Dict[str, Any],
    evidence_bundle: Dict[str, Any],
    pass_threshold_0_1: float,
    prompt_budget_tokens: int,
) -> Tuple[Optional[Step05CriticReviewModel], Dict[str, Any]]:
    system_template = load_prompt("step05_hapa_intervention_critic_system.md")
    user_template = load_prompt("step05_hapa_intervention_critic_user_template.md")
    sections = [
        PromptSection("meta", json.dumps(evidence_bundle.get("meta", {}), ensure_ascii=False, indent=2), priority=1),
        PromptSection("intervention_candidate", json.dumps(intervention_payload, ensure_ascii=False, indent=2), priority=2),
        PromptSection("readiness", json.dumps(evidence_bundle.get("readiness", {}), ensure_ascii=False, indent=2), priority=3),
        PromptSection("network", json.dumps(evidence_bundle.get("network", {}), ensure_ascii=False, indent=2), priority=4),
        PromptSection("impact", json.dumps(evidence_bundle.get("impact", {}), ensure_ascii=False, indent=2), priority=5),
        PromptSection("target_candidates", json.dumps(evidence_bundle.get("target_candidates", []), ensure_ascii=False, indent=2), priority=6),
        PromptSection("barrier_candidates", json.dumps(evidence_bundle.get("barrier_candidates", {}), ensure_ascii=False, indent=2), priority=7),
        PromptSection("coping_candidates", json.dumps(evidence_bundle.get("coping_candidates", {}), ensure_ascii=False, indent=2), priority=8),
        PromptSection("parent_feasibility", json.dumps(evidence_bundle.get("predictor_parent_feasibility", []), ensure_ascii=False, indent=2), priority=9),
    ]
    packed = pack_prompt_sections(
        sections,
        max_tokens=int(prompt_budget_tokens),
        reserve_tokens=3000,
        model=client.model,
    )
    user_prompt = render_prompt(
        user_template,
        {
            "PASS_THRESHOLD": f"{float(pass_threshold_0_1):.3f}",
            "EVIDENCE_BUNDLE_JSON": packed.text,
        },
    )
    llm_result = client.generate_structured(
        system_prompt=system_template,
        user_prompt=user_prompt,
        schema_model=Step05CriticReviewModel,
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
    parsed = Step05CriticReviewModel.model_validate(llm_result.parsed)
    parsed.profile_id = profile_id
    weighted = weighted_composite(subscores=parsed.weighted_subscores_0_1, weights=_step05_critic_weights())
    parsed.composite_score_0_1 = float(weighted.get("composite_score_0_1", parsed.composite_score_0_1))
    parsed.pass_decision = decision_from_score(
        score_0_1=float(parsed.composite_score_0_1),
        threshold_0_1=float(pass_threshold_0_1),
        critical_issues=list(parsed.critical_issues or []),
    )
    return parsed, trace


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[4]
    default_handoff_root = repo_root / "Evaluation/05_treatment_target_handoff"
    default_output_root = repo_root / "Evaluation/06_translation_digital_intervention"
    default_readiness_root = repo_root / "Evaluation/04_initial_observation_analysis/00_readiness_check"
    default_network_root = repo_root / "Evaluation/04_initial_observation_analysis/01_time_series_analysis/network"
    default_impact_root = repo_root / "Evaluation/04_initial_observation_analysis/02_momentary_impact_coefficients"
    default_free_text_root = repo_root / "Evaluation/01_pseudoprofile(s)/free_text"
    default_predictor_barrier = (
        repo_root
        / "src/utils/official/ontology_mappings/PREDICTOR/barrier_to_predictor/results/gpt-5-nano/predictor_to_barrier_edges_long.csv"
    )
    default_profile_barrier = (
        repo_root
        / "src/utils/official/ontology_mappings/HAPA/profile_to_barrier/results/gpt-5-nano/profile_to_barrier_edges_long.csv"
    )
    default_context_barrier = (
        repo_root
        / "src/utils/official/ontology_mappings/HAPA/context_to_barrier/results/gpt-5-nano/context_to_barrier_edges_long.csv"
    )
    default_coping_barrier = (
        repo_root
        / "src/utils/official/ontology_mappings/HAPA/coping_to_barrier/results/gpt-5-nano/coping_to_barrier_edges_long.csv"
    )
    default_predictor_feasibility = (
        repo_root
        / "src/utils/official/multi_dimensional_feasibility_evaluation/PREDICTORS/results/summary/predictor_rankings.csv"
    )

    parser = argparse.ArgumentParser(
        description="Generate Step-05 HAPA-based digital interventions from integrated PHOENIX evidence."
    )
    parser.add_argument("--handoff-root", type=str, default=str(default_handoff_root))
    parser.add_argument("--output-root", type=str, default=str(default_output_root))
    parser.add_argument("--readiness-root", type=str, default=str(default_readiness_root))
    parser.add_argument("--network-root", type=str, default=str(default_network_root))
    parser.add_argument("--impact-root", type=str, default=str(default_impact_root))
    parser.add_argument("--free-text-root", type=str, default=str(default_free_text_root))
    parser.add_argument("--predictor-to-barrier-csv", type=str, default=str(default_predictor_barrier))
    parser.add_argument("--profile-to-barrier-csv", type=str, default=str(default_profile_barrier))
    parser.add_argument("--context-to-barrier-csv", type=str, default=str(default_context_barrier))
    parser.add_argument("--coping-to-barrier-csv", type=str, default=str(default_coping_barrier))
    parser.add_argument("--predictor-feasibility-csv", type=str, default=str(default_predictor_feasibility))
    parser.add_argument("--pattern", type=str, default="pseudoprofile_FTC_")
    parser.add_argument("--max-profiles", type=int, default=0)
    parser.add_argument("--top-barriers-per-predictor", type=int, default=30)
    parser.add_argument("--select-top-barriers", type=int, default=10)
    parser.add_argument("--llm-model", type=str, default="gpt-5-nano")
    parser.add_argument("--llm-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--llm-max-attempts", type=int, default=2)
    parser.add_argument("--llm-repair-attempts", type=int, default=1)
    parser.add_argument("--disable-llm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prompt-budget-tokens", type=int, default=400000)
    parser.add_argument("--critic-max-iterations", type=int, default=2)
    parser.add_argument("--critic-pass-threshold", type=float, default=0.74)
    parser.add_argument("--hard-ontology-constraint", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--parent-feasibility-top-k", type=int, default=30)
    parser.add_argument("--contract-version", type=str, default="1.0.0")
    parser.add_argument("--trace-output", type=str, default="")
    parser.add_argument("--visualization-dpi", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _init_plot_style()
    handoff_root = Path(args.handoff_root).expanduser().resolve()
    output_root = ensure_dir(Path(args.output_root).expanduser().resolve())
    readiness_root = Path(args.readiness_root).expanduser().resolve()
    network_root = Path(args.network_root).expanduser().resolve()
    impact_root = Path(args.impact_root).expanduser().resolve()
    free_text_root = Path(args.free_text_root).expanduser().resolve()

    predictor_to_barrier_df = safe_read_csv(Path(args.predictor_to_barrier_csv).expanduser().resolve())
    profile_to_barrier_df = safe_read_csv(Path(args.profile_to_barrier_csv).expanduser().resolve())
    context_to_barrier_df = safe_read_csv(Path(args.context_to_barrier_csv).expanduser().resolve())
    coping_to_barrier_df = safe_read_csv(Path(args.coping_to_barrier_csv).expanduser().resolve())
    predictor_feasibility_csv = Path(args.predictor_feasibility_csv).expanduser().resolve()
    predictor_feasibility_df = load_predictor_feasibility_table(predictor_feasibility_csv)

    profiles = discover_profiles(
        handoff_root=handoff_root,
        pattern=str(args.pattern or "").strip(),
        max_profiles=int(args.max_profiles),
    )
    if not profiles:
        log(f"[ERROR] no profiles found in handoff root: {handoff_root}")
        return 2

    complaints = parse_profile_text_file(free_text_root / "free_text_complaints.txt")
    person_profiles = parse_profile_text_file(free_text_root / "free_text_person.txt")
    context_profiles = parse_profile_text_file(free_text_root / "free_text_context.txt")

    llm_client = StructuredLLMClient(
        model=str(args.llm_model),
        timeout_seconds=float(args.llm_timeout_seconds),
        max_attempts=int(args.llm_max_attempts),
        repair_attempts=int(args.llm_repair_attempts),
    )

    log("========== STEP-05 DIGITAL INTERVENTION START ==========")
    log(f"handoff_root: {handoff_root}")
    log(f"output_root: {output_root}")
    log(f"profiles: {len(profiles)}")
    log(f"llm_model: {args.llm_model}")
    log(f"disable_llm: {bool(args.disable_llm)}")
    log(f"hard_ontology_constraint: {bool(args.hard_ontology_constraint)}")
    if predictor_feasibility_df.empty:
        log(f"predictor feasibility table unavailable/empty: {predictor_feasibility_csv}")
    else:
        log(f"predictor feasibility rows loaded: {len(predictor_feasibility_df)}")

    summary_rows: List[Dict[str, Any]] = []
    failed = 0

    for profile_dir in profiles:
        profile_id = profile_dir.name
        out_profile_dir = ensure_dir(output_root / profile_id)
        try:
            step03_payload = read_json(profile_dir / "step03_target_selection.json")
            step04_payload = read_json(profile_dir / "step04_updated_observation_model.json")
            fusion_payload = read_json(profile_dir / "step04_nomothetic_idiographic_fusion.json")
            evidence_bundle = read_json(profile_dir / "step03_evidence_bundle.json")

            free_text = profile_text_bundle(
                profile_id=profile_id,
                complaints=complaints,
                person_profiles=person_profiles,
                context_profiles=context_profiles,
                evidence_bundle=evidence_bundle,
            )

            readiness_payload = read_json(readiness_root / profile_id / "readiness_report.json")
            impact_predictor_df = safe_read_csv(impact_root / profile_id / "predictor_composite.csv")
            impact_edge_df = safe_read_csv(impact_root / profile_id / "edge_composite.csv")
            network_summary = summarize_network_metrics(network_root / profile_id)

            predictor_candidates = collect_predictor_candidates(
                profile_id=profile_id,
                step03_payload=step03_payload,
                step04_payload=step04_payload,
                fusion_payload=fusion_payload,
                evidence_bundle=evidence_bundle,
            )
            predictor_parent_feasibility = top_parent_domains_for_bundle(
                predictor_paths=[str(row.get("predictor_path") or "") for row in predictor_candidates[:60]],
                feasibility_frame=predictor_feasibility_df,
                top_k=max(1, int(args.parent_feasibility_top_k)),
                per_predictor_k=max(1, int(args.parent_feasibility_top_k)),
                parent_levels=2,
            )

            barrier_detail_df, predictor_barrier_agg = barrier_candidates_from_predictors(
                predictor_candidates=predictor_candidates,
                predictor_to_barrier_df=predictor_to_barrier_df,
                top_n_per_predictor=int(args.top_barriers_per_predictor),
            )

            profile_barriers = _barrier_scores_from_text_mapping(
                free_text.get("person_text", ""),
                profile_to_barrier_df,
                text_col_path="profile_full_path",
                text_col_name="profile_name",
                source="profile",
            )
            context_barriers = _barrier_scores_from_text_mapping(
                free_text.get("context_text", ""),
                context_to_barrier_df,
                text_col_path="context_full_path",
                text_col_name="context_name",
                source="context",
            )
            selected_barriers = combine_barrier_evidence(
                predictor_barriers=predictor_barrier_agg,
                profile_barriers=profile_barriers,
                context_barriers=context_barriers,
                complaint_text=str(free_text.get("complaint_text") or ""),
                top_k=int(args.select_top_barriers),
            )

            coping_detail_df, coping_candidates = select_coping_candidates(
                selected_barriers=selected_barriers,
                coping_to_barrier_df=coping_to_barrier_df,
            )

            criteria_ids = step04_payload.get("retained_criteria_ids", []) or []
            evidence_payload = {
                "meta": {
                    "profile_id": profile_id,
                    "generated_at_local": ts(),
                    "sources": {
                        "handoff_root": str(handoff_root),
                        "readiness_root": str(readiness_root),
                        "network_root": str(network_root),
                        "impact_root": str(impact_root),
                    },
                },
                "free_text": free_text,
                "readiness": (readiness_payload.get("overall", {}) if isinstance(readiness_payload, dict) else {}),
                "network": network_summary,
                "impact": {
                    "top_predictors": impact_predictor_df.head(30).fillna("").to_dict(orient="records") if not impact_predictor_df.empty else [],
                    "top_edges": impact_edge_df.head(40).fillna("").to_dict(orient="records") if not impact_edge_df.empty else [],
                },
                "step03_output": step03_payload,
                "step04_output": step04_payload,
                "target_candidates": predictor_candidates[:60],
                "predictor_parent_feasibility": predictor_parent_feasibility,
                "barrier_candidates": {
                    "selected_top10": selected_barriers,
                    "predictor_barrier_rows_top": barrier_detail_df.head(300).fillna("").to_dict(orient="records") if not barrier_detail_df.empty else [],
                },
                "coping_candidates": {
                    "top_ranked": coping_candidates[:80],
                    "detail_rows_top": coping_detail_df.head(350).fillna("").to_dict(orient="records") if not coping_detail_df.empty else [],
                },
            }

            critic_max_iterations = max(0, int(args.critic_max_iterations))
            step05_trace: Dict[str, Any] = {"provider": "heuristic", "reason": "disabled", "actor_attempts": [], "critic_attempts": []}
            step05_feedback: List[str] = []
            step05_guardrail: Optional[Step05CriticReviewModel] = None
            step05_output: Optional[Step05InterventionModel] = None
            for attempt in range(critic_max_iterations + 1):
                payload_for_attempt = dict(evidence_payload)
                if step05_feedback:
                    payload_for_attempt["critic_feedback"] = {
                        "feedback_for_revision": step05_feedback,
                        "attempt_index": attempt,
                    }
                if not bool(args.disable_llm):
                    step05_candidate, step05_actor_trace = run_llm_step05(
                        client=llm_client,
                        profile_id=profile_id,
                        evidence_bundle=payload_for_attempt,
                        prompt_budget_tokens=int(args.prompt_budget_tokens),
                    )
                else:
                    step05_candidate = None
                    step05_actor_trace = {"provider": "heuristic", "reason": "disabled"}
                step05_actor_trace["attempt_index"] = attempt
                step05_trace["actor_attempts"].append(step05_actor_trace)

                actor_mode = "structured_llm_success"
                if step05_candidate is None:
                    actor_mode = "heuristic_fallback_disabled_llm" if bool(args.disable_llm) else "heuristic_fallback_llm_failure"
                    step05_candidate = heuristic_step05_intervention(
                        profile_id=profile_id,
                        free_text=free_text,
                        criteria_ids=criteria_ids,
                        predictor_candidates=predictor_candidates,
                        selected_barriers=selected_barriers,
                        coping_candidates=coping_candidates,
                    )

                if bool(args.hard_ontology_constraint):
                    hard_constraint_trace = _enforce_step05_hard_ontology(
                        step05_candidate,
                        allowed_predictor_paths=[str(row.get("predictor_path") or "") for row in predictor_candidates],
                        allowed_barrier_paths=[str(row.get("barrier_path") or "") for row in selected_barriers],
                        allowed_coping_paths=[str(row.get("coping_path") or "") for row in coping_candidates],
                    )
                else:
                    hard_constraint_trace = {"changed_count": 0}

                run_critic = (not bool(args.disable_llm)) and (actor_mode == "structured_llm_success")
                if run_critic:
                    critic_llm, critic_trace = run_llm_step05_critic(
                        client=llm_client,
                        profile_id=profile_id,
                        intervention_payload=step05_candidate.model_dump(mode="json"),
                        evidence_bundle=payload_for_attempt,
                        pass_threshold_0_1=float(args.critic_pass_threshold),
                        prompt_budget_tokens=int(args.prompt_budget_tokens),
                    )
                    if critic_llm is None:
                        step05_guardrail = _heuristic_step05_critic(
                            profile_id=profile_id,
                            intervention=step05_candidate,
                            evidence_bundle=payload_for_attempt,
                            pass_threshold_0_1=float(args.critic_pass_threshold),
                        )
                        critic_trace["mode"] = "heuristic_fallback"
                    else:
                        step05_guardrail = critic_llm
                        critic_trace["mode"] = "structured_llm_success"
                    critic_trace["attempt_index"] = attempt
                    critic_trace["hard_constraint"] = hard_constraint_trace
                    critic_trace["critic_decision"] = step05_guardrail.pass_decision
                    critic_trace["critic_composite_score_0_1"] = step05_guardrail.composite_score_0_1
                    step05_trace["critic_attempts"].append(critic_trace)

                    if step05_guardrail.pass_decision == "PASS" or attempt >= critic_max_iterations:
                        step05_output = step05_candidate
                        step05_trace["reason"] = actor_mode
                        break
                    step05_feedback = list(step05_guardrail.feedback_for_revision or [])
                    if not step05_feedback:
                        step05_feedback = [
                            "Increase explicit evidence grounding for target-barrier-coping links.",
                            "Improve HAPA completeness and safety/monitoring specificity.",
                        ]
                    continue

                step05_guardrail = _heuristic_step05_critic(
                    profile_id=profile_id,
                    intervention=step05_candidate,
                    evidence_bundle=payload_for_attempt,
                    pass_threshold_0_1=float(args.critic_pass_threshold),
                )
                step05_trace["critic_attempts"].append(
                    {
                        "attempt_index": attempt,
                        "mode": "heuristic_auto",
                        "hard_constraint": hard_constraint_trace,
                        "critic_decision": step05_guardrail.pass_decision,
                        "critic_composite_score_0_1": step05_guardrail.composite_score_0_1,
                    }
                )
                step05_output = step05_candidate
                step05_trace["reason"] = actor_mode
                step05_trace["hard_constraint"] = hard_constraint_trace
                break

            assert step05_output is not None
            step05_output.contract_version = str(args.contract_version)

            if "heuristic_fallback" in str(step05_trace.get("reason", "")) and LIMITATION_LLM_UNAVAILABLE not in step05_output.limitations:
                step05_output.limitations = [LIMITATION_LLM_UNAVAILABLE, *step05_output.limitations]
            if step05_guardrail is not None:
                step05_trace["critic_final_decision"] = step05_guardrail.pass_decision
                step05_trace["critic_final_score_0_1"] = step05_guardrail.composite_score_0_1

            visuals_dir = ensure_dir(out_profile_dir / "visuals")
            barrier_visual_files = _plot_ranked_bars(
                rows=selected_barriers,
                label_key="barrier_name",
                score_key="total_score_0_1",
                title=f"{profile_id} — Step05 Top Barrier Domains",
                out_png=visuals_dir / "step05_barriers_ranked.png",
                dpi=int(args.visualization_dpi),
                top_n=10,
            )
            coping_visual_files = _plot_ranked_bars(
                rows=coping_candidates,
                label_key="coping_name",
                score_key="score_0_1",
                title=f"{profile_id} — Step05 Top Coping Strategies",
                out_png=visuals_dir / "step05_coping_ranked.png",
                dpi=int(args.visualization_dpi),
                top_n=12,
            )
            step05_trace["visual_files"] = [*barrier_visual_files, *coping_visual_files]

            (out_profile_dir / "step05_hapa_evidence_bundle.json").write_text(
                json.dumps(evidence_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            (out_profile_dir / "step05_hapa_prompt_trace.json").write_text(
                json.dumps(step05_trace, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            (out_profile_dir / "step05_hapa_intervention.json").write_text(
                json.dumps(step05_output.model_dump(mode="json"), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            if step05_guardrail is not None:
                (out_profile_dir / "step05_guardrail_review.json").write_text(
                    json.dumps(step05_guardrail.model_dump(mode="json"), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            (out_profile_dir / "step05_predictor_parent_feasibility_top30.json").write_text(
                json.dumps(predictor_parent_feasibility[:30], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            barrier_detail_df.to_csv(out_profile_dir / "step05_barrier_candidates_top30_per_predictor.csv", index=False)
            pd.DataFrame(selected_barriers).to_csv(out_profile_dir / "step05_selected_barriers_top10.csv", index=False)
            coping_detail_df.to_csv(out_profile_dir / "step05_coping_candidates_detail.csv", index=False)
            pd.DataFrame(coping_candidates).to_csv(out_profile_dir / "step05_coping_candidates_ranked.csv", index=False)

            summary_rows.append(
                {
                    "profile_id": profile_id,
                    "mode": str(step05_trace.get("reason")),
                    "guardrail_decision": str(step05_trace.get("critic_final_decision", "")),
                    "guardrail_score_0_1": step05_trace.get("critic_final_score_0_1", ""),
                    "n_targets": int(len(step05_output.selected_treatment_targets)),
                    "n_barriers": int(len(step05_output.selected_barriers)),
                    "n_coping": int(len(step05_output.selected_coping_strategies)),
                    "n_hapa_components": int(len(step05_output.hapa_component_plan)),
                    "n_phases": int(len(step05_output.phased_delivery_plan)),
                    "n_visual_files": int(len(step05_trace.get("visual_files", []) or [])),
                    "top_target": (
                        step05_output.selected_treatment_targets[0].predictor_label
                        if step05_output.selected_treatment_targets
                        else ""
                    ),
                    "top_barrier": (
                        step05_output.selected_barriers[0].barrier_name
                        if step05_output.selected_barriers
                        else ""
                    ),
                    "confidence_0_1": float(step05_output.confidence_0_1),
                    "criteria_count": int(len(criteria_ids)),
                }
            )

            log(
                f"[OK] {profile_id}: targets={len(step05_output.selected_treatment_targets)} "
                f"barriers={len(step05_output.selected_barriers)} coping={len(step05_output.selected_coping_strategies)} "
                f"mode={step05_trace.get('reason')}"
            )
        except Exception as exc:
            failed += 1
            log(f"[ERROR] {profile_id}: {repr(exc)}")

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(output_root / "step05_interventions_summary.csv", index=False)
    summary = {
        "contract_version": str(args.contract_version),
        "generated_at_local": ts(),
        "handoff_root": str(handoff_root),
        "output_root": str(output_root),
        "readiness_root": str(readiness_root),
        "network_root": str(network_root),
        "impact_root": str(impact_root),
        "predictor_feasibility_csv": str(predictor_feasibility_csv),
        "llm_model": str(args.llm_model),
        "disable_llm": bool(args.disable_llm),
        "hard_ontology_constraint": bool(args.hard_ontology_constraint),
        "critic_max_iterations": int(args.critic_max_iterations),
        "critic_pass_threshold": float(args.critic_pass_threshold),
        "parent_feasibility_top_k": int(args.parent_feasibility_top_k),
        "n_profiles_attempted": len(profiles),
        "n_profiles_success": len(summary_rows),
        "n_profiles_failed": failed,
        "profiles": summary_rows,
    }
    (output_root / "step05_interventions_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if str(args.trace_output).strip():
        trace_path = Path(args.trace_output).expanduser().resolve()
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_payload = {
            "contract_version": str(args.contract_version),
            "generated_at_local": ts(),
            "stage": "intervention",
            "llm_model": str(args.llm_model),
            "disable_llm": bool(args.disable_llm),
            "hard_ontology_constraint": bool(args.hard_ontology_constraint),
            "critic_max_iterations": int(args.critic_max_iterations),
            "critic_pass_threshold": float(args.critic_pass_threshold),
            "n_profiles_attempted": len(profiles),
            "n_profiles_success": len(summary_rows),
            "n_profiles_failed": failed,
        }
        trace_path.write_text(json.dumps(trace_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log("========== STEP-05 DIGITAL INTERVENTION COMPLETE ==========")
    log(f"success={len(summary_rows)} failed={failed}")
    return 0 if failed == 0 else 4


if __name__ == "__main__":
    raise SystemExit(main())
