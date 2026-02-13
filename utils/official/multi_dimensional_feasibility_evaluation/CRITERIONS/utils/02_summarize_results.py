#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_summarize_criterion_cluster_rankings.py  (a.k.a. 02_summarize_results.py)

What this script does (end-to-end)
----------------------------------
1) Reads cluster-level LLM evaluations from cache_evaluations/*.json (one evaluation per cluster).
2) Dynamically imports your scores-only evaluation module with internal weights.
   - IMPORTANT: runs Pydantic v2 model_rebuild() after import (fixes forward refs).
3) Computes:
   - overall suitability + per-dimension suitability/risk (via module.compute_overall_suitability)
   - leaf-feature weighted contributions to overall risk (sanity-checked: sum == overall risk)
   - leaf-feature raw scores (Likert 1..9) for plotting (with raw datapoints)
4) Loads the semantic cluster taxonomy:
   - 04_semantically_clustered_items.json (gives leaf cluster ids and their taxonomy path)
   - Optionally loads CRITERION_ontology.json to infer missing/ambiguous system/domain metadata
5) Adds classification metadata per cluster:
   - system ∈ {DSM-5TR, ICD-10, RDoC, UNKNOWN}
   - criterion_domain (e.g., CLINICAL, ... if present)
   - taxonomy path string
6) Writes structured outputs to /results/summary
7) Writes publish-ready figures (bar plots only) into a clean /results/visuals substructure, including:
   A) System comparison figures (3-row subplots: DSM-5TR vs ICD-10 vs RDoC)
      - overall suitability distributions (bar=mean; points=clusters)
      - per-primary-dimension suitability/risk (bars per dimension, same dimension colors across rows)
      - per-feature-group comparisons (bars per feature; same feature colors across rows; points=clusters)
   B) Ontology/taxonomy node rankings per system (top nodes by aggregated suitability)
      - 3-row figure: each row a system, bars = top taxonomy nodes, points = cluster datapoints

Notes
-----
- Likert9 scores represent PROBLEM likelihood (risk): 1=low risk, 9=high risk.
- Suitability = 1 - normalized_risk, risk normalized to [0,1].
- Visualization uses matplotlib Agg backend for headless reliability on macOS/CI.
- Colors:
  - For dimension comparison: same dimension == same color everywhere
  - For feature comparison: same feature label == same color everywhere (within that plot)

Defaults assume your project paths (override via CLI args if needed).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

import matplotlib
matplotlib.use("Agg")  # must be set before pyplot import
import matplotlib.pyplot as plt


EPS = 1e-9

# --- Evaluation module (preferred + fallback)
DEFAULT_EVAL_MODULE_PATH = Path(
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/"
    "multi_dimensional_feasibility_evaluation/CRITERIONS/utils/"
    "00_hierarchical_criterion_evaluation_modules.py"
)
FALLBACK_EVAL_MODULE_PATH = Path(
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/"
    "multi_dimensional_feasibility_evaluation/CRITERIONS/utils/"
    "00_hierarchical_evaluation_modules.py"
)

# --- Caches (LLM outputs)
DEFAULT_CACHE_DIR = Path(
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/"
    "multi_dimensional_feasibility_evaluation/CRITERIONS/results/responses/cache_evaluations"
)

# --- Cluster taxonomy (leaf cluster ids + taxonomy path)
DEFAULT_CLUSTER_TAXONOMY_JSON = Path(
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/"
    "cluster_criterions/results/04_semantically_clustered_items.json"
)

# --- Ontology (optional fallback inference of paths)
DEFAULT_CRITERION_ONTOLOGY_JSON = Path(
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/SystemComponents/PHOENIX_ontology/separate/"
    "01_raw/CRITERION/steps/01_raw/aggregated/CRITERION_ontology.json"
)

# --- Output roots
DEFAULT_SUMMARY_DIR = Path(
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/"
    "multi_dimensional_feasibility_evaluation/CRITERIONS/results/summary"
)
DEFAULT_VISUALS_DIR = Path(
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/"
    "multi_dimensional_feasibility_evaluation/CRITERIONS/results/visuals"
)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def normalize_likert_to_unit_interval(score_1_to_9: int) -> float:
    s = int(score_1_to_9)
    assert 1 <= s <= 9, f"Likert score out of range 1..9: {s}"
    return (float(s) - 1.0) / 8.0


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(float(v) for v in weights.values()))
    assert total > EPS, f"Cannot normalize weights; sum too small: {total}"
    return {k: float(v) / total for k, v in weights.items()}


def short(s: str, n: int = 120) -> str:
    s = str(s).replace("\n", " ").strip()
    return s if len(s) <= n else (s[: n - 1] + "…")


def _safe_filename(s: str) -> str:
    s = str(s).strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "plot"


def _is_nan_str(x: Any) -> bool:
    return str(x) == "nan" or str(x) == "None"


# -----------------------------------------------------------------------------
# System normalization & path inference
# -----------------------------------------------------------------------------

SYSTEM_CANONICAL = ["DSM-5TR", "ICD-10", "RDoC"]
SYSTEM_SYNONYMS = {
    "DSM5TR": "DSM-5TR",
    "DSM-5TR": "DSM-5TR",
    "DSM-5-TR": "DSM-5TR",
    "DSM5-TR": "DSM-5TR",
    "ICD10": "ICD-10",
    "ICD-10": "ICD-10",
    "ICD_10": "ICD-10",
    "RDOC": "RDoC",
    "RDoC": "RDoC",
    "R-DOC": "RDoC",
}


def normalize_system_token(token: str) -> Optional[str]:
    t = str(token).strip()
    if not t:
        return None
    u = t.upper().replace(" ", "")
    # preserve RDoC capitalization
    if u in SYSTEM_SYNONYMS:
        return SYSTEM_SYNONYMS[u]
    # try a few common shapes
    u2 = u.replace("_", "-")
    if u2 in SYSTEM_SYNONYMS:
        return SYSTEM_SYNONYMS[u2]
    return None


def find_system_in_path(path_elems: List[str]) -> Optional[str]:
    for pe in path_elems:
        sys = normalize_system_token(pe)
        if sys is not None:
            return sys
    # Also handle "DSM-5TR" etc appearing inside strings
    joined = " ".join(path_elems).upper()
    if "DSM" in joined:
        return "DSM-5TR"
    if "ICD" in joined:
        return "ICD-10"
    if "RDOC" in joined:
        return "RDoC"
    return None


def find_domain_in_path(path_elems: List[str]) -> Optional[str]:
    """
    Heuristic: criterion domain is the node after 'CRITERION' if present.
    """
    for i, pe in enumerate(path_elems):
        if str(pe).strip().upper() == "CRITERION":
            if i + 1 < len(path_elems):
                return str(path_elems[i + 1])
    return None


def path_to_str(path_elems: List[str], sep: str = " / ") -> str:
    return sep.join([str(x) for x in path_elems if str(x)])


# -----------------------------------------------------------------------------
# Dynamic import + Pydantic rebuild
# -----------------------------------------------------------------------------

def resolve_eval_module_path(p: Path) -> Path:
    if p.exists():
        return p
    if DEFAULT_EVAL_MODULE_PATH.exists():
        return DEFAULT_EVAL_MODULE_PATH
    if FALLBACK_EVAL_MODULE_PATH.exists():
        return FALLBACK_EVAL_MODULE_PATH
    raise FileNotFoundError(
        f"Evaluation module not found. Tried: {p} | {DEFAULT_EVAL_MODULE_PATH} | {FALLBACK_EVAL_MODULE_PATH}"
    )


def rebuild_pydantic_models_in_module(mod: Any) -> None:
    """
    Required for Pydantic v2 with forward refs when dynamically importing.
    """
    from pydantic import BaseModel as PydBaseModel  # type: ignore

    ns: Dict[str, Any] = {name: getattr(mod, name) for name in dir(mod)}
    rebuilt = 0
    failed: List[str] = []

    for name, obj in ns.items():
        if not isinstance(obj, type):
            continue
        try:
            if issubclass(obj, PydBaseModel) and obj is not PydBaseModel:
                try:
                    obj.model_rebuild(force=True, _types_namespace=ns)  # type: ignore[arg-type]
                except TypeError:
                    obj.model_rebuild(force=True)
                rebuilt += 1
        except Exception as e:
            failed.append(f"{name}: {repr(e)}")

    if hasattr(mod, "CriterionEvaluation"):
        ce = getattr(mod, "CriterionEvaluation")
        if isinstance(ce, type) and issubclass(ce, PydBaseModel):
            try:
                try:
                    ce.model_rebuild(force=True, _types_namespace=ns)  # type: ignore[arg-type]
                except TypeError:
                    ce.model_rebuild(force=True)
            except Exception as e:
                raise AssertionError(f"CriterionEvaluation.model_rebuild() failed: {repr(e)}") from e

    if failed:
        print(f"[{utc_now_iso()}] WARNING: model_rebuild failed for {len(failed)} models; showing first 10:")
        for s in failed[:10]:
            print(f"  - {s}")
    print(f"[{utc_now_iso()}] Pydantic rebuild complete: rebuilt_models={rebuilt} failed={len(failed)}")


def import_eval_module(path: Path) -> Any:
    path = resolve_eval_module_path(path)
    spec = importlib.util.spec_from_file_location("hier_crit_eval", str(path))
    assert spec is not None and spec.loader is not None, f"Could not load module spec: {path}"

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    rebuild_pydantic_models_in_module(mod)

    assert hasattr(mod, "CriterionEvaluation"), "Module missing CriterionEvaluation"
    assert hasattr(mod, "compute_overall_suitability"), "Module missing compute_overall_suitability"
    assert hasattr(mod, "default_dimension_weights"), "Module missing default_dimension_weights"
    assert hasattr(mod, "DataCollectionAggregation"), "Module missing DataCollectionAggregation"
    return mod


# -----------------------------------------------------------------------------
# Cache parsing
# -----------------------------------------------------------------------------

CACHE_FILE_RE = re.compile(
    r"^cluster_(?P<cluster_id>[^_]+)_(?P<cluster_hash>[0-9a-f]{40})\.json$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ClusterCache:
    cache_path: Path
    cluster_id: str
    cluster_hash: str
    cluster_tree_path: str
    cluster_size: int
    n_members: int
    member_ids: List[str]
    member_sample: str
    evaluation: Dict[str, Any]
    cached_at_utc: Optional[str]


def iter_cache_files(cache_dir: Path) -> Iterable[Path]:
    assert cache_dir.exists(), f"cache_dir does not exist: {cache_dir}"
    assert cache_dir.is_dir(), f"cache_dir is not a directory: {cache_dir}"
    return sorted([p for p in cache_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json"])


def load_cluster_cache(path: Path) -> ClusterCache:
    m = CACHE_FILE_RE.match(path.name)
    assert m is not None, f"Unexpected cache filename: {path.name}"

    payload = read_json(path)
    assert isinstance(payload, dict), f"Cache JSON root must be object: {path}"

    cluster_id = str(payload.get("cluster_id", m.group("cluster_id")))
    cluster_hash = str(payload.get("cluster_hash", m.group("cluster_hash")))
    cluster_tree_path = str(payload.get("cluster_tree_path", cluster_id))
    cluster_size = int(payload.get("cluster_size", 0) or 0)
    n_members = int(payload.get("n_members", 0) or 0)
    cached_at_utc = payload.get("cached_at_utc", None)

    members = payload.get("members", [])
    assert isinstance(members, list), f"'members' must be a list in {path}"

    member_ids: List[str] = []
    for it in members:
        if isinstance(it, dict) and "criterion_id" in it:
            member_ids.append(str(it["criterion_id"]))

    sample_texts: List[str] = []
    for it in members[:12]:
        if isinstance(it, dict):
            txt = it.get("raw_item_text", it.get("criterion_id", ""))
            sample_texts.append(short(txt, 90))
    member_sample = " | ".join([t for t in sample_texts if t])

    evaluation = payload.get("evaluation", None)
    assert isinstance(evaluation, dict), f"'evaluation' missing or not an object in {path}"

    return ClusterCache(
        cache_path=path,
        cluster_id=cluster_id,
        cluster_hash=cluster_hash,
        cluster_tree_path=cluster_tree_path,
        cluster_size=cluster_size,
        n_members=n_members if n_members > 0 else len(members),
        member_ids=member_ids,
        member_sample=member_sample,
        evaluation=evaluation,
        cached_at_utc=cached_at_utc if isinstance(cached_at_utc, str) else None,
    )


# -----------------------------------------------------------------------------
# Cluster taxonomy parsing (leaf clusters and their taxonomy path)
# -----------------------------------------------------------------------------

def load_cluster_taxonomy_paths(taxonomy_json_path: Path) -> Dict[str, List[str]]:
    """
    Reads 04_semantically_clustered_items.json and returns:
      cluster_id -> taxonomy_path_elems (including the cluster_id as last element)

    Works for both:
      - nested taxonomy dicts, where leaves look like:
          "c1887": {"size": 5, "items": [...]}
      - flat dict mapping "c1887" -> {"size":..., "items":[...]} (path becomes ["c1887"])
    """
    if not taxonomy_json_path.exists():
        return {}
    root = read_json(taxonomy_json_path)

    cluster_paths: Dict[str, List[str]] = {}

    def walk(node: Any, path: List[str]) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                k_str = str(k)
                if isinstance(v, dict) and ("items" in v) and ("size" in v):
                    # likely a leaf cluster
                    cluster_paths[k_str] = path + [k_str]
                else:
                    walk(v, path + [k_str])
        # lists or other types: ignore (taxonomy is dict-driven)

    if isinstance(root, dict):
        # If it's flat: keys are clusters
        all_leaf_like = True
        for k, v in root.items():
            if not (isinstance(v, dict) and ("items" in v) and ("size" in v)):
                all_leaf_like = False
                break
        if all_leaf_like:
            for k in root.keys():
                cluster_paths[str(k)] = [str(k)]
        else:
            walk(root, [])
    return cluster_paths


# -----------------------------------------------------------------------------
# Ontology parsing (optional fallback inference from criterion_id -> path)
# -----------------------------------------------------------------------------

def load_ontology_leaf_paths(ontology_json_path: Path) -> Dict[str, List[str]]:
    """
    Reads CRITERION_ontology.json and returns:
      leaf_key (criterion id) -> path_elems to that leaf (excluding leaf_key itself)
    where leaves are dicts (often empty):  "leaf_key": {}
    """
    if not ontology_json_path.exists():
        return {}
    root = read_json(ontology_json_path)
    leaf_paths: Dict[str, List[str]] = {}

    def walk(node: Any, path: List[str]) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                k_str = str(k)
                if isinstance(v, dict) and len(v) == 0:
                    leaf_paths[k_str] = path[:]  # leaf
                else:
                    walk(v, path + [k_str])

    walk(root, [])
    return leaf_paths


def infer_cluster_system_from_members(
    member_ids: List[str],
    criterion_leaf_paths: Dict[str, List[str]],
) -> Optional[str]:
    """
    If taxonomy path doesn't contain system labels, infer from member criterion ids via ontology.
    Uses majority vote of system tokens found in ontology paths.
    """
    votes: Dict[str, int] = {}
    for cid in member_ids:
        p = criterion_leaf_paths.get(cid)
        if not p:
            continue
        sys = find_system_in_path(p)
        if sys:
            votes[sys] = votes.get(sys, 0) + 1
    if not votes:
        return None
    return sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


# -----------------------------------------------------------------------------
# Computation: contributions + raw scores
# -----------------------------------------------------------------------------

@dataclass
class ContributionRow:
    cluster_id: str
    cluster_hash: str
    cluster_tree_path: str
    dimension: str
    module: str
    method: str
    feature: str
    likert_score: int
    risk_unit: float
    weight_product: float
    contribution_overall_risk: float


@dataclass
class FeatureScoreRow:
    cluster_id: str
    cluster_hash: str
    cluster_tree_path: str
    dimension: str
    module: str
    method: str
    feature: str
    likert_score: int
    risk_unit: float


def weighted_risk_from_scores(scores: Dict[str, int], weights: Dict[str, float]) -> float:
    assert set(scores.keys()) == set(weights.keys()), f"Scores/weights mismatch: {set(scores) ^ set(weights)}"
    w = normalize_weights(weights)
    r = 0.0
    for k, s in scores.items():
        r += float(w[k]) * normalize_likert_to_unit_interval(int(s))
    return max(0.0, min(1.0, float(r)))


def compute_contributions_and_raw_scores_for_cluster(
    cluster_id: str,
    cc: ClusterCache,
    e: Any,
    mod: Any,
) -> Tuple[Dict[str, Any], List[ContributionRow], List[FeatureScoreRow]]:
    """
    Returns:
      breakdown (from module.compute_overall_suitability)
      contributions: leaf-feature weighted contributions to overall risk (sum matches overall risk)
      raw_scores: leaf-feature raw likert + risk_unit (no weighting) for plotting
    """
    dim_w = normalize_weights({k: float(v) for k, v in mod.default_dimension_weights().items()})
    assert abs(sum(dim_w.values()) - 1.0) < 1e-6, f"Dimension weights must sum to 1: sum={sum(dim_w.values())}"

    breakdown_model = mod.compute_overall_suitability(e)
    breakdown = breakdown_model.model_dump()

    contribs: List[ContributionRow] = []
    raw_rows: List[FeatureScoreRow] = []

    def add_leaf_rows(
        dimension: str,
        module: str,
        method: str,
        scores: Dict[str, int],
        feat_w: Optional[Dict[str, float]] = None,
        extra_weight: float = 1.0,
        compute_contrib: bool = True,
    ) -> None:
        # Raw rows always
        for feat, s in scores.items():
            ru = normalize_likert_to_unit_interval(int(s))
            raw_rows.append(
                FeatureScoreRow(
                    cluster_id=cluster_id,
                    cluster_hash=cc.cluster_hash,
                    cluster_tree_path=cc.cluster_tree_path,
                    dimension=dimension,
                    module=module,
                    method=method,
                    feature=feat,
                    likert_score=int(s),
                    risk_unit=float(ru),
                )
            )

        if not compute_contrib:
            return

        assert feat_w is not None, "feat_w required when compute_contrib=True"
        assert dimension in dim_w, f"Unknown dimension key: {dimension}"
        dw = float(dim_w[dimension])
        feat_w_n = normalize_weights({k: float(v) for k, v in feat_w.items()})
        assert set(scores.keys()) == set(feat_w_n.keys()), (
            f"Feature keys mismatch for {dimension}/{module}/{method}: "
            f"{set(scores.keys()) ^ set(feat_w_n.keys())}"
        )

        for feat, s in scores.items():
            ru = normalize_likert_to_unit_interval(int(s))
            wp = dw * float(extra_weight) * float(feat_w_n[feat])
            contribs.append(
                ContributionRow(
                    cluster_id=cluster_id,
                    cluster_hash=cc.cluster_hash,
                    cluster_tree_path=cc.cluster_tree_path,
                    dimension=dimension,
                    module=module,
                    method=method,
                    feature=feat,
                    likert_score=int(s),
                    risk_unit=float(ru),
                    weight_product=float(wp),
                    contribution_overall_risk=float(wp * ru),
                )
            )

    # --- Mathematical restrictions (flat)
    add_leaf_rows(
        "mathematical_restrictions",
        "mathematical_restrictions",
        "",
        {k: int(v) for k, v in e.mathematical_restrictions.scores.model_dump().items()},
        feat_w=mod.MathematicalRestrictions.default_weights(),
    )

    # --- Data collection feasibility (methods)
    dcf = e.data_collection_feasibility
    method_names = [
        "self_report_ema",
        "third_party_ema",
        "wearable",
        "user_device_data",
        "etl_pipeline",
        "third_party_api",
    ]

    per_method_feat_w = mod.CollectionMethodFeasibility.default_weights()
    method_w_all = normalize_weights({k: float(v) for k, v in mod.DataCollectionFeasibility.default_method_weights().items()})

    method_scores: Dict[str, Dict[str, int]] = {}
    method_risks: Dict[str, float] = {}
    available: List[str] = []
    for mn in method_names:
        mobj = getattr(dcf, mn, None)
        if mobj is None:
            continue
        sc = {k: int(v) for k, v in mobj.scores.model_dump().items()}
        method_scores[mn] = sc
        method_risks[mn] = weighted_risk_from_scores(sc, per_method_feat_w)
        available.append(mn)

    assert available, f"No data-collection methods present for cluster={cluster_id}."

    # Raw rows for ALL methods
    for mn in available:
        add_leaf_rows(
            "data_collection_feasibility",
            "data_collection_feasibility",
            mn,
            method_scores[mn],
            feat_w=None,
            compute_contrib=False,
        )

    # Contributions depend on aggregation
    if dcf.aggregation == mod.DataCollectionAggregation.BEST_AVAILABLE:
        best_m = min(available, key=lambda k: method_risks[k])
        add_leaf_rows(
            "data_collection_feasibility",
            "data_collection_feasibility",
            best_m,
            method_scores[best_m],
            feat_w=per_method_feat_w,
            extra_weight=1.0,
            compute_contrib=True,
        )
    else:
        total_mw = sum(float(method_w_all[mn]) for mn in available)
        assert total_mw > EPS, "Sum of method weights for available methods is 0."
        for mn in available:
            mw = float(method_w_all[mn]) / float(total_mw)
            add_leaf_rows(
                "data_collection_feasibility",
                "data_collection_feasibility",
                mn,
                method_scores[mn],
                feat_w=per_method_feat_w,
                extra_weight=mw,
                compute_contrib=True,
            )

    # --- Validity threats (modules)
    vt = e.validity_threats
    vt_module_w = normalize_weights({k: float(v) for k, v in mod.ValidityThreatWeights().as_dict().items()})
    vt_modules: List[Tuple[str, Any]] = [
        ("response_bias", mod.ResponseBiasRisk),
        ("insight_capacity", mod.InsightReportingCapacityRisk),
        ("measurement_validity", mod.MeasurementValidityRisk),
    ]
    available_mods = [mn for mn, _ in vt_modules if getattr(vt, mn, None) is not None]
    assert available_mods, f"No validity submodules present for cluster={cluster_id}."
    total_vt = sum(float(vt_module_w[mn]) for mn in available_mods)
    assert total_vt > EPS, "Sum of validity module weights is 0."

    # raw for each module
    for mn, _mcls in vt_modules:
        mobj = getattr(vt, mn, None)
        if mobj is None:
            continue
        add_leaf_rows(
            "validity_threats",
            mn,
            "",
            {k: int(v) for k, v in mobj.scores.model_dump().items()},
            feat_w=None,
            compute_contrib=False,
        )
    # weighted contributions
    for mn, mcls in vt_modules:
        mobj = getattr(vt, mn, None)
        if mobj is None:
            continue
        mw = float(vt_module_w[mn]) / float(total_vt)
        add_leaf_rows(
            "validity_threats",
            mn,
            "",
            {k: int(v) for k, v in mobj.scores.model_dump().items()},
            feat_w=mcls.default_weights(),
            extra_weight=mw,
            compute_contrib=True,
        )

    # --- EU regulatory risk (modules)
    rr = e.eu_regulatory_risk
    rr_module_w = normalize_weights({k: float(v) for k, v in mod.RegulatoryWeights().as_dict().items()})
    rr_modules: List[Tuple[str, Any]] = [
        ("gdpr", mod.GDPRComplianceRisk),
        ("eu_ai_act", mod.EUAIActRisk),
        ("medical_device", mod.MedicalDeviceRegRisk),
        ("eprivacy", mod.ePrivacyRisk),
        ("cybersecurity", mod.CybersecurityRisk),
    ]
    available_mods = [mn for mn, _ in rr_modules if getattr(rr, mn, None) is not None]
    assert available_mods, f"No regulatory submodules present for cluster={cluster_id}."
    total_rr = sum(float(rr_module_w[mn]) for mn in available_mods)
    assert total_rr > EPS, "Sum of regulatory module weights is 0."

    # raw
    for mn, _mcls in rr_modules:
        mobj = getattr(rr, mn, None)
        if mobj is None:
            continue
        add_leaf_rows(
            "eu_regulatory_risk",
            mn,
            "",
            {k: int(v) for k, v in mobj.scores.model_dump().items()},
            feat_w=None,
            compute_contrib=False,
        )
    # weighted contributions
    for mn, mcls in rr_modules:
        mobj = getattr(rr, mn, None)
        if mobj is None:
            continue
        mw = float(rr_module_w[mn]) / float(total_rr)
        add_leaf_rows(
            "eu_regulatory_risk",
            mn,
            "",
            {k: int(v) for k, v in mobj.scores.model_dump().items()},
            feat_w=mcls.default_weights(),
            extra_weight=mw,
            compute_contrib=True,
        )

    # --- General importance (flat)
    add_leaf_rows(
        "general_importance",
        "general_importance",
        "",
        {k: int(v) for k, v in e.general_importance.scores.model_dump().items()},
        feat_w=mod.GeneralImportance.default_weights(),
    )

    # --- Scientific utility (flat)
    add_leaf_rows(
        "scientific_utility",
        "scientific_utility",
        "",
        {k: int(v) for k, v in e.scientific_utility.scores.model_dump().items()},
        feat_w=mod.ScientificUtility.default_weights(),
    )

    # Sanity: overall risk equals sum contributions
    overall_risk_from_breakdown = 1.0 - float(breakdown["overall_suitability"])
    overall_risk_from_contrib = float(sum(r.contribution_overall_risk for r in contribs))
    assert abs(overall_risk_from_breakdown - overall_risk_from_contrib) < 1e-6, (
        f"Overall risk mismatch for cluster={cluster_id}: "
        f"breakdown={overall_risk_from_breakdown} contrib_sum={overall_risk_from_contrib}"
    )

    return breakdown, contribs, raw_rows


# -----------------------------------------------------------------------------
# Console: hierarchical printing for clusters (clean)
# -----------------------------------------------------------------------------

def format_tree(lines: List[Tuple[int, str]]) -> str:
    out: List[str] = []
    for indent, text in lines:
        if indent <= 0:
            out.append(text)
        else:
            out.append(("  " * (indent - 1)) + "└─ " + text)
    return "\n".join(out)


def print_cluster_hierarchy(
    cluster_id: str,
    row: Dict[str, Any],
    df_contrib_cluster: pd.DataFrame,
    top_per_dimension: int = 6,
) -> None:
    lines: List[Tuple[int, str]] = []
    sys = row.get("system", "UNKNOWN")
    dom = row.get("criterion_domain", "")
    hdr = f"{cluster_id} | system={sys}"
    if dom:
        hdr += f" | domain={dom}"
    hdr += f" | overall_suitability={float(row['overall_suitability']):.4f} | members={int(row['n_members'])}"
    lines.append((0, hdr))

    if row.get("taxonomy_path_str"):
        lines.append((1, f"taxonomy_path: {row['taxonomy_path_str']}"))
    if row.get("cluster_tree_path"):
        lines.append((1, f"cache_tree_path: {row['cluster_tree_path']}"))
    if row.get("member_sample"):
        lines.append((1, f"sample: {short(row['member_sample'], 220)}"))

    dim_suit_cols = sorted([k for k in row.keys() if isinstance(k, str) and k.startswith("suitability.")])
    if dim_suit_cols:
        lines.append((1, "dimensions:"))
        for sc in dim_suit_cols:
            dk = sc[len("suitability."):]
            rc = f"risk.{dk}"
            suit = float(row.get(sc, float("nan")))
            risk = float(row.get(rc, float("nan"))) if rc in row else float("nan")
            lines.append((2, f"{dk}: suitability={suit:.4f} | risk={risk:.4f}"))

            sub = df_contrib_cluster[df_contrib_cluster["dimension"] == dk].copy()
            if len(sub) == 0:
                continue
            sub = sub.sort_values("contribution_overall_risk", ascending=False).head(top_per_dimension)
            lines.append((3, "top_contributors (weighted to overall risk):"))
            for _, rr in sub.iterrows():
                mod_ = str(rr.get("module", ""))
                meth = str(rr.get("method", ""))
                feat = str(rr.get("feature", ""))
                score = int(rr.get("likert_score", -1))
                contrib = float(rr.get("contribution_overall_risk", 0.0))
                path = mod_
                if meth and not _is_nan_str(meth):
                    path += f":{meth}"
                path += f".{feat}"
                lines.append((4, f"{path} | score={score} | contrib={contrib:.6f}"))

    print(format_tree(lines), flush=True)


# -----------------------------------------------------------------------------
# Plotting: publish-ready barplots (with consistent colors)
# -----------------------------------------------------------------------------

def _get_cmap_colors(n: int, cmap_name: str) -> List[Any]:
    cmap = plt.get_cmap(cmap_name)
    if n <= 1:
        return [cmap(0.5)]
    return [cmap(i / (n - 1)) for i in range(n)]


def plot_system_overall_suitability_3rows(
    df_clusters: pd.DataFrame,
    systems: List[str],
    out_path: Path,
    rng_seed: int = 42,
) -> None:
    """
    3-row comparison: each row is a system, bar=mean overall suitability, points=raw clusters.
    """
    ensure_dir(out_path.parent)
    rng = random.Random(int(rng_seed))

    fig, axes = plt.subplots(nrows=len(systems), ncols=1, figsize=(10, 2.8 * len(systems)), sharex=True)
    if len(systems) == 1:
        axes = [axes]

    for ax, sys in zip(axes, systems):
        sub = df_clusters[df_clusters["system"] == sys].copy()
        vals = sub["overall_suitability"].astype(float).tolist()
        vals = [v for v in vals if not math.isnan(v)]
        if not vals:
            ax.set_title(f"{sys} (no data)")
            ax.set_xlim(0.0, 1.0)
            ax.grid(True, axis="x", linestyle="--", linewidth=0.6, alpha=0.5)
            continue

        mean_v = float(sum(vals) / len(vals))
        ax.barh([0], [mean_v], height=0.55)
        jitter = [(0 + (rng.random() - 0.5) * 0.22) for _ in vals]
        ax.scatter(vals, jitter, s=12, alpha=0.25, color="black", linewidths=0)

        ax.set_yticks([0])
        ax.set_yticklabels([sys])
        ax.set_xlim(0.0, 1.0)
        ax.grid(True, axis="x", linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_xlabel("overall_suitability (higher = better)")

    fig.suptitle("Overall suitability by system (bar=mean; points=clusters)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_system_dimensions_3rows(
    df_clusters: pd.DataFrame,
    systems: List[str],
    dim_keys: List[str],
    prefix: str,  # "suitability." or "risk."
    out_path: Path,
    title: str,
    x_label: str,
    rng_seed: int = 42,
) -> None:
    """
    3-row comparison: each row is a system, bars are per-dimension mean values, points are raw cluster values.
    Bars have consistent colors by dimension across rows.
    """
    ensure_dir(out_path.parent)

    # Colors consistent by dimension
    dim_colors = {dk: c for dk, c in zip(dim_keys, _get_cmap_colors(len(dim_keys), "tab10"))}
    rng = random.Random(int(rng_seed))

    fig, axes = plt.subplots(nrows=len(systems), ncols=1, figsize=(14, 3.2 * len(systems)), sharex=True)
    if len(systems) == 1:
        axes = [axes]

    for ax, sys in zip(axes, systems):
        sub = df_clusters[df_clusters["system"] == sys].copy()
        ax.set_title(sys)

        # compute means per dim
        means: List[float] = []
        raws: Dict[str, List[float]] = {}
        for dk in dim_keys:
            col = f"{prefix}{dk}"
            if col not in sub.columns:
                means.append(float("nan"))
                raws[dk] = []
                continue
            vals = sub[col].astype(float).tolist()
            vals = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
            raws[dk] = vals
            means.append(float(sum(vals) / len(vals)) if vals else float("nan"))

        y = list(range(len(dim_keys)))
        colors = [dim_colors[dk] for dk in dim_keys]
        ax.barh(y, means, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(dim_keys)
        ax.invert_yaxis()
        ax.grid(True, axis="x", linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_xlabel(x_label)

        # raw points overlay
        for i, dk in enumerate(dim_keys):
            vals = raws.get(dk, [])
            if not vals:
                continue
            jitter = [(i + (rng.random() - 0.5) * 0.22) for _ in vals]
            ax.scatter(vals, jitter, s=10, alpha=0.20, color="black", linewidths=0)

    # Legend (dimension colors)
    handles = [plt.Rectangle((0, 0), 1, 1, color=dim_colors[dk]) for dk in dim_keys]
    fig.legend(handles, dim_keys, loc="upper center", ncol=min(6, len(dim_keys)), frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(title, y=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_feature_group_system_comparison_3rows(
    df_scores: pd.DataFrame,
    systems: List[str],
    dimension: str,
    module: str,
    method: str,
    out_path: Path,
    title: str,
    max_features_per_part: int = 40,
    rng_seed: int = 42,
) -> None:
    """
    For one feature group (dimension/module/method), create a 3-row comparison (one row per system).
    Bars are per-feature means (Likert), points are per-cluster raw Likert values.
    Uses consistent colors for the SAME feature label across rows.
    """
    ensure_dir(out_path.parent)

    sub_all = df_scores[
        (df_scores["dimension"] == dimension)
        & (df_scores["module"] == module)
        & (df_scores["method"] == method)
        & (df_scores["system"].isin(systems))
    ].copy()

    if len(sub_all) == 0:
        return

    # Feature ordering based on global mean (across all systems)
    feat_means = (
        sub_all.groupby("feature", dropna=False)["likert_score"]
        .mean()
        .sort_values(ascending=False)
    )
    features = feat_means.index.astype(str).tolist()

    # Split if too many features
    parts: List[List[str]] = []
    if len(features) <= max_features_per_part:
        parts = [features]
    else:
        for i in range(0, len(features), max_features_per_part):
            parts.append(features[i:i + max_features_per_part])

    # Global feature colors (consistent)
    feat_colors = {f: c for f, c in zip(features, _get_cmap_colors(len(features), "tab20"))}

    rng = random.Random(int(rng_seed))

    for part_idx, part_feats in enumerate(parts, start=1):
        fig, axes = plt.subplots(nrows=len(systems), ncols=1, figsize=(16, 3.5 * len(systems)), sharex=True)
        if len(systems) == 1:
            axes = [axes]

        for ax, sys in zip(axes, systems):
            sub = sub_all[sub_all["system"] == sys].copy()
            ax.set_title(sys)

            # means + IQR for bar annotation (optional)
            means = []
            q25s = []
            q75s = []
            rawvals: Dict[str, List[float]] = {}
            for feat in part_feats:
                vals = sub[sub["feature"] == feat]["likert_score"].astype(float).tolist()
                vals = [v for v in vals if not math.isnan(v)]
                rawvals[feat] = vals
                if vals:
                    s = pd.Series(vals)
                    means.append(float(s.mean()))
                    q25s.append(float(s.quantile(0.25)))
                    q75s.append(float(s.quantile(0.75)))
                else:
                    means.append(float("nan"))
                    q25s.append(float("nan"))
                    q75s.append(float("nan"))

            y = list(range(len(part_feats)))
            colors = [feat_colors[f] for f in part_feats]
            ax.barh(y, means, color=colors)
            ax.set_yticks(y)
            ax.set_yticklabels([short(f, 70) for f in part_feats])
            ax.invert_yaxis()
            ax.set_xlim(1.0, 9.0)
            ax.grid(True, axis="x", linestyle="--", linewidth=0.6, alpha=0.5)
            ax.set_xlabel("Likert score (risk; 1=low, 9=high)")

            # IQR whiskers
            for i, (a, b) in enumerate(zip(q25s, q75s)):
                if math.isnan(a) or math.isnan(b):
                    continue
                ax.hlines(i, a, b, linewidth=2.0, alpha=0.85, color="black")

            # raw points
            for i, feat in enumerate(part_feats):
                vals = rawvals.get(feat, [])
                if not vals:
                    continue
                jitter = [(i + (rng.random() - 0.5) * 0.22) for _ in vals]
                ax.scatter(vals, jitter, s=10, alpha=0.22, color="black", linewidths=0)

        # Title / save
        ttl = title if len(parts) == 1 else f"{title} (part {part_idx}/{len(parts)})"
        fig.suptitle(ttl, y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        save_path = out_path if len(parts) == 1 else out_path.with_name(out_path.stem + f"_part{part_idx}" + out_path.suffix)
        fig.savefig(save_path, dpi=240)
        plt.close(fig)


def plot_taxonomy_node_rankings_3rows(
    df_clusters: pd.DataFrame,
    systems: List[str],
    compare_level: int,
    top_n_nodes: int,
    out_path: Path,
    rng_seed: int = 42,
) -> None:
    """
    Uses taxonomy_path_elems to aggregate cluster suitability to an internal node level.
    For each system: bars = top nodes by mean overall suitability, points = cluster datapoints.
    """
    ensure_dir(out_path.parent)
    rng = random.Random(int(rng_seed))

    # Prepare node labels per cluster
    rows: List[Dict[str, Any]] = []
    for _, r in df_clusters.iterrows():
        sys = str(r.get("system", "UNKNOWN"))
        if sys not in systems:
            continue
        path_str = r.get("taxonomy_path_str", "")
        path_elems = r.get("taxonomy_path_elems", None)
        if not isinstance(path_elems, list) or not path_elems:
            continue
        sys_idx = None
        for i, pe in enumerate(path_elems):
            if normalize_system_token(pe) == sys:
                sys_idx = i
                break
        if sys_idx is None:
            # try any system token in path
            continue

        after = path_elems[sys_idx + 1 : ]
        # pick node level within system
        node_elems = after[: max(1, int(compare_level))]
        node_label = path_to_str([sys] + node_elems)
        rows.append({
            "system": sys,
            "node_label": node_label,
            "overall_suitability": float(r["overall_suitability"]),
        })

    df_nodes = pd.DataFrame(rows)
    if len(df_nodes) == 0:
        return

    fig, axes = plt.subplots(nrows=len(systems), ncols=1, figsize=(16, 3.8 * len(systems)), sharex=True)
    if len(systems) == 1:
        axes = [axes]

    # Global color mapping by node_label (consistent across rows if labels repeat)
    all_nodes = df_nodes["node_label"].astype(str).unique().tolist()
    node_colors = {nl: c for nl, c in zip(all_nodes, _get_cmap_colors(len(all_nodes), "tab20"))}

    for ax, sys in zip(axes, systems):
        sub = df_nodes[df_nodes["system"] == sys].copy()
        if len(sub) == 0:
            ax.set_title(f"{sys} (no nodes)")
            continue

        node_mean = (
            sub.groupby("node_label")["overall_suitability"]
            .mean()
            .sort_values(ascending=False)
        )
        top_nodes = node_mean.head(int(top_n_nodes)).index.astype(str).tolist()

        means = [float(node_mean.loc[nl]) for nl in top_nodes]
        y = list(range(len(top_nodes)))
        colors = [node_colors[nl] for nl in top_nodes]

        ax.barh(y, means, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels([short(nl, 85) for nl in top_nodes])
        ax.invert_yaxis()
        ax.set_xlim(0.0, 1.0)
        ax.grid(True, axis="x", linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_xlabel("overall_suitability (higher = better)")
        ax.set_title(f"{sys} — top taxonomy nodes (level={compare_level})")

        # raw cluster points
        for i, nl in enumerate(top_nodes):
            vals = sub[sub["node_label"] == nl]["overall_suitability"].astype(float).tolist()
            jitter = [(i + (rng.random() - 0.5) * 0.22) for _ in vals]
            ax.scatter(vals, jitter, s=10, alpha=0.22, color="black", linewidths=0)

    fig.suptitle("Taxonomy node suitability rankings by system (bar=mean; points=clusters)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--eval-module", type=str, default=str(DEFAULT_EVAL_MODULE_PATH))
    ap.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))

    ap.add_argument("--cluster-taxonomy-json", type=str, default=str(DEFAULT_CLUSTER_TAXONOMY_JSON))
    ap.add_argument("--criterion-ontology-json", type=str, default=str(DEFAULT_CRITERION_ONTOLOGY_JSON))

    ap.add_argument("--summary-dir", type=str, default=str(DEFAULT_SUMMARY_DIR))
    ap.add_argument("--visuals-dir", type=str, default=str(DEFAULT_VISUALS_DIR))

    ap.add_argument("--systems", type=str, default="DSM-5TR,ICD-10,RDoC", help="Comma-separated system order for 3-row comparisons.")
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--top-n-hierarchy", type=int, default=10)
    ap.add_argument("--top-per-dimension", type=int, default=6)

    ap.add_argument("--compare-level", type=int, default=2, help="Taxonomy node depth within system for node ranking plots.")
    ap.add_argument("--compare-top-n-nodes", type=int, default=20)

    ap.add_argument("--max-features-per-plot", type=int, default=40)
    ap.add_argument("--rng-seed", type=int, default=42)

    args = ap.parse_args()

    systems = [normalize_system_token(s.strip()) or s.strip() for s in str(args.systems).split(",")]
    # normalize + keep only known order (but allow any)
    systems = [s for s in systems if s]
    if len(systems) != 3:
        # still works, but the user asked for 3-row comparisons; keep as-is
        pass

    eval_module_path = resolve_eval_module_path(Path(args.eval_module))
    cache_dir = Path(args.cache_dir)
    summary_dir = Path(args.summary_dir)
    visuals_dir = Path(args.visuals_dir)

    cluster_taxonomy_json = Path(args.cluster_taxonomy_json)
    criterion_ontology_json = Path(args.criterion_ontology_json)

    ensure_dir(summary_dir)
    ensure_dir(visuals_dir)

    # --- visuals substructure (clean)
    vis_rankings = visuals_dir / "rankings"
    vis_dimensions = visuals_dir / "dimensions"
    vis_system = visuals_dir / "system_comparison"
    vis_system_summary = vis_system / "00_summary"
    vis_system_features = vis_system / "01_feature_groups"
    vis_system_taxonomy = vis_system / "02_taxonomy_rankings"
    for p in [vis_rankings, vis_dimensions, vis_system_summary, vis_system_features, vis_system_taxonomy]:
        ensure_dir(p)

    print(f"[{utc_now_iso()}] Loading evaluation module: {eval_module_path}")
    mod = import_eval_module(eval_module_path)
    CriterionEvaluation = mod.CriterionEvaluation

    cache_files = list(iter_cache_files(cache_dir))
    assert cache_files, f"No JSON cache files found in: {cache_dir}"
    print(f"[{utc_now_iso()}] Reading caches: n_files={len(cache_files)} dir={cache_dir}")

    caches: List[ClusterCache] = [load_cluster_cache(p) for p in cache_files]
    assert caches, "No valid caches loaded."

    # --- Load taxonomy paths (cluster_id -> path elems)
    taxonomy_paths = load_cluster_taxonomy_paths(cluster_taxonomy_json)
    print(f"[{utc_now_iso()}] Taxonomy paths loaded: n_cluster_paths={len(taxonomy_paths)} from={cluster_taxonomy_json}")

    # --- Load ontology leaf paths for fallback inference
    ontology_leaf_paths = load_ontology_leaf_paths(criterion_ontology_json)
    print(f"[{utc_now_iso()}] Ontology leaf paths loaded: n_leaf_paths={len(ontology_leaf_paths)} from={criterion_ontology_json}")

    cluster_rows: List[Dict[str, Any]] = []
    contrib_rows: List[Dict[str, Any]] = []
    score_rows: List[Dict[str, Any]] = []
    wide_rows: List[Dict[str, Any]] = []

    for cc in caches:
        try:
            e = CriterionEvaluation.model_validate(cc.evaluation)
        except Exception as ex:
            raise AssertionError(f"Evaluation schema validation failed for {cc.cache_path}: {repr(ex)}") from ex

        breakdown, contribs, raw_scores = compute_contributions_and_raw_scores_for_cluster(cc.cluster_id, cc, e, mod)

        # taxonomy metadata
        tax_path_elems = taxonomy_paths.get(cc.cluster_id, [])
        sys_from_tax = find_system_in_path(tax_path_elems) if tax_path_elems else None
        dom_from_tax = find_domain_in_path(tax_path_elems) if tax_path_elems else None

        # fallback: infer system from members using ontology
        sys_from_members = infer_cluster_system_from_members(cc.member_ids, ontology_leaf_paths) if (not sys_from_tax) else None
        system = sys_from_tax or sys_from_members or "UNKNOWN"
        criterion_domain = dom_from_tax or find_domain_in_path(tax_path_elems) or ""

        # Per-cluster row
        row = {
            "cluster_id": cc.cluster_id,
            "cluster_hash": cc.cluster_hash,
            "cluster_tree_path": cc.cluster_tree_path,
            "cluster_size": cc.cluster_size,
            "n_members": cc.n_members,
            "member_sample": cc.member_sample,
            "cached_at_utc": cc.cached_at_utc,
            "overall_suitability": float(breakdown["overall_suitability"]),
            "system": system,
            "criterion_domain": criterion_domain,
            "taxonomy_path_str": path_to_str(tax_path_elems) if tax_path_elems else "",
            "taxonomy_path_elems": tax_path_elems,
        }
        for k, v in breakdown["by_dimension_suitability"].items():
            row[f"suitability.{k}"] = float(v)
        for k, v in breakdown["by_dimension_risk"].items():
            row[f"risk.{k}"] = float(v)
        cluster_rows.append(row)

        # Weighted contributions
        for r in contribs:
            contrib_rows.append(
                {
                    "cluster_id": r.cluster_id,
                    "cluster_hash": r.cluster_hash,
                    "cluster_tree_path": r.cluster_tree_path,
                    "dimension": r.dimension,
                    "module": r.module,
                    "method": r.method,
                    "feature": r.feature,
                    "likert_score": r.likert_score,
                    "risk_unit": r.risk_unit,
                    "weight_product": r.weight_product,
                    "contribution_overall_risk": r.contribution_overall_risk,
                }
            )

        # Raw score rows
        for r in raw_scores:
            score_rows.append(
                {
                    "cluster_id": r.cluster_id,
                    "cluster_hash": r.cluster_hash,
                    "cluster_tree_path": r.cluster_tree_path,
                    "dimension": r.dimension,
                    "module": r.module,
                    "method": r.method,
                    "feature": r.feature,
                    "likert_score": r.likert_score,
                    "risk_unit": r.risk_unit,
                }
            )

        # Wide row (debug/inspection)
        def flatten(d: Any, prefix: str = "") -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            if isinstance(d, dict):
                for kk, vv in d.items():
                    key = f"{prefix}.{kk}" if prefix else str(kk)
                    out.update(flatten(vv, key))
            elif isinstance(d, list):
                out[prefix] = json.dumps(d, ensure_ascii=False)
            else:
                out[prefix] = d
            return out

        wide = {
            "cluster_id": cc.cluster_id,
            "cluster_hash": cc.cluster_hash,
            "cluster_tree_path": cc.cluster_tree_path,
            "cluster_size": cc.cluster_size,
            "n_members": cc.n_members,
            "cached_at_utc": cc.cached_at_utc,
        }
        wide.update(flatten({"evaluation": cc.evaluation}))
        wide.update(flatten({"breakdown": breakdown}))
        wide_rows.append(wide)

    df_clusters = pd.DataFrame(cluster_rows)
    df_contrib = pd.DataFrame(contrib_rows)
    df_scores = pd.DataFrame(score_rows)
    df_wide = pd.DataFrame(wide_rows)

    assert len(df_clusters) == len(caches), "Cluster row count mismatch."
    assert len(df_contrib) > 0, "No contributions computed."
    assert len(df_scores) > 0, "No raw scores extracted."

    # Merge system/domain metadata into df_scores and df_contrib
    meta_cols = ["cluster_id", "system", "criterion_domain", "taxonomy_path_str"]
    df_scores = df_scores.merge(df_clusters[meta_cols], on="cluster_id", how="left")
    df_contrib = df_contrib.merge(df_clusters[meta_cols], on="cluster_id", how="left")

    # --- Write outputs
    out_clusters_csv = summary_dir / "cluster_rankings.csv"
    out_contrib_csv = summary_dir / "cluster_feature_contributions_long.csv"
    out_scores_csv = summary_dir / "cluster_feature_scores_long.csv"
    out_wide_csv = summary_dir / "cluster_evaluations_wide_from_cache.csv"

    df_clusters.to_csv(out_clusters_csv, index=False, encoding="utf-8")
    df_contrib.to_csv(out_contrib_csv, index=False, encoding="utf-8")
    df_scores.to_csv(out_scores_csv, index=False, encoding="utf-8")
    df_wide.to_csv(out_wide_csv, index=False, encoding="utf-8")

    # --- Ranking for console
    df_sorted = df_clusters.sort_values("overall_suitability", ascending=False).reset_index(drop=True)
    top_n = min(int(args.top_n), len(df_sorted))

    print("\n" + "=" * 112)
    print("OVERALL CLUSTER RANKING (best -> worst)")
    print("=" * 112)
    for i, r in df_sorted.head(top_n).iterrows():
        print(
            f"{i+1:>3}. {r['cluster_id']:<8} "
            f"system={r['system']:<7} "
            f"overall_suitability={float(r['overall_suitability']):.4f} "
            f"members={int(r['n_members'])} "
            f"sample={short(r['member_sample'], 90)}"
        )

    print("\n" + "=" * 112)
    print(f"TOP {min(int(args.top_n_hierarchy), len(df_sorted))} CLUSTERS — HIERARCHICAL BREAKDOWN (weighted)")
    print("=" * 112)
    top_for_tree = df_sorted.head(min(int(args.top_n_hierarchy), len(df_sorted))).copy()
    for _, rr in top_for_tree.iterrows():
        cid = str(rr["cluster_id"])
        subc = df_contrib[df_contrib["cluster_id"] == cid].copy()
        row_dict = {k: rr[k] for k in rr.index}
        print_cluster_hierarchy(
            cluster_id=cid,
            row=row_dict,
            df_contrib_cluster=subc,
            top_per_dimension=int(args.top_per_dimension),
        )
        print("-" * 112)

    # --- Global feature importance
    g = (
        df_contrib.groupby(["dimension", "module", "method", "feature"], dropna=False)["contribution_overall_risk"]
        .mean()
        .reset_index()
        .sort_values("contribution_overall_risk", ascending=False)
        .reset_index(drop=True)
    )
    out_global_csv = summary_dir / "global_feature_importance.csv"
    g.to_csv(out_global_csv, index=False, encoding="utf-8")

    # --- Text report
    report_lines: List[str] = []
    report_lines.append("Criterion cluster weighted ranking report")
    report_lines.append(f"generated_at_utc: {utc_now_iso()}")
    report_lines.append(f"cache_dir: {cache_dir}")
    report_lines.append(f"eval_module: {eval_module_path}")
    report_lines.append(f"taxonomy_json: {cluster_taxonomy_json}")
    report_lines.append(f"ontology_json: {criterion_ontology_json}")
    report_lines.append(f"n_clusters: {len(df_clusters)}")
    report_lines.append("")
    report_lines.append("# TODO: Once you stop testing, switch input to the *_wide.csv produced by 01_run_cluster_evaluation.py")
    report_lines.append("")

    report_lines.append("TOP CLUSTERS (overall_suitability)")
    report_lines.append("--------------------------------")
    for i, r in df_sorted.head(top_n).iterrows():
        report_lines.append(
            f"{i+1:>3}. {r['cluster_id']:<8} system={r['system']:<7} suit={float(r['overall_suitability']):.4f} "
            f"members={int(r['n_members'])} taxonomy_path={r.get('taxonomy_path_str','')}"
        )
        report_lines.append(f"     sample: {short(r['member_sample'], 260)}")

    out_report = summary_dir / "criterion_cluster_weighted_ranking_report.txt"
    write_text(out_report, "\n".join(report_lines) + "\n")

    # -----------------------------------------------------------------------------
    # VISUALIZATIONS (clean substructure + publish-ready)
    # -----------------------------------------------------------------------------

    rng_seed = int(args.rng_seed)
    dim_keys = list(mod.default_dimension_weights().keys())
    max_feats = int(args.max_features_per_plot)

    # (1) System comparison: overall suitability
    plot_system_overall_suitability_3rows(
        df_clusters=df_clusters,
        systems=systems,
        out_path=vis_system_summary / "overall_suitability_by_system_3rows.png",
        rng_seed=rng_seed,
    )

    # (2) System comparison: primary dimensions suitability + risk
    plot_system_dimensions_3rows(
        df_clusters=df_clusters,
        systems=systems,
        dim_keys=dim_keys,
        prefix="suitability.",
        out_path=vis_system_summary / "primary_dimensions_suitability_by_system_3rows.png",
        title="Primary dimension suitability by system (bars=means; points=clusters; consistent dimension colors)",
        x_label="suitability (higher = better)",
        rng_seed=rng_seed,
    )
    plot_system_dimensions_3rows(
        df_clusters=df_clusters,
        systems=systems,
        dim_keys=dim_keys,
        prefix="risk.",
        out_path=vis_system_summary / "primary_dimensions_risk_by_system_3rows.png",
        title="Primary dimension risk by system (bars=means; points=clusters; consistent dimension colors)",
        x_label="risk (0..1; higher = worse)",
        rng_seed=rng_seed,
    )

    # (3) Feature-group comparisons across systems (3-row plots)
    #    Groups mirror your evaluation hierarchy.
    #    Output is organized by dimension/module/method.
    feature_groups: List[Tuple[str, str, str]] = []

    # Flat dimensions
    feature_groups += [
        ("mathematical_restrictions", "mathematical_restrictions", ""),
        ("general_importance", "general_importance", ""),
        ("scientific_utility", "scientific_utility", ""),
    ]

    # Validity threats modules
    for mod_name in ["response_bias", "insight_capacity", "measurement_validity"]:
        feature_groups.append(("validity_threats", mod_name, ""))

    # EU regulatory modules
    for mod_name in ["gdpr", "eu_ai_act", "medical_device", "eprivacy", "cybersecurity"]:
        feature_groups.append(("eu_regulatory_risk", mod_name, ""))

    # Data collection feasibility methods (only those present in the data, to avoid empty plots)
    dcf_methods = sorted([m for m in df_scores[df_scores["dimension"] == "data_collection_feasibility"]["method"].dropna().astype(str).unique().tolist() if m and not _is_nan_str(m)])
    for meth in dcf_methods:
        feature_groups.append(("data_collection_feasibility", "data_collection_feasibility", meth))

    for (dimension, module, method) in feature_groups:
        sub = df_scores[
            (df_scores["dimension"] == dimension)
            & (df_scores["module"] == module)
            & (df_scores["method"] == method)
            & (df_scores["system"].isin(systems))
        ].copy()
        if len(sub) == 0:
            continue

        out_dir = vis_system_features / _safe_filename(dimension) / _safe_filename(module)
        ensure_dir(out_dir)

        group_tag = "features" if not method else _safe_filename(method)
        out_path = out_dir / f"{_safe_filename(dimension)}__{_safe_filename(module)}__{group_tag}__systems_3rows.png"

        plot_feature_group_system_comparison_3rows(
            df_scores=df_scores,
            systems=systems,
            dimension=dimension,
            module=module,
            method=method,
            out_path=out_path,
            title=f"{dimension} | {module}{(' | '+method) if method else ''} — feature scores across systems (3-row)",
            max_features_per_part=max_feats,
            rng_seed=rng_seed,
        )

    # (4) Taxonomy node rankings (aggregated) — system comparison (3-row)
    plot_taxonomy_node_rankings_3rows(
        df_clusters=df_clusters,
        systems=systems,
        compare_level=int(args.compare_level),
        top_n_nodes=int(args.compare_top_n_nodes),
        out_path=vis_system_taxonomy / f"taxonomy_node_rankings_level{int(args.compare_level)}_top{int(args.compare_top_n_nodes)}_3rows.png",
        rng_seed=rng_seed,
    )

    # -----------------------------------------------------------------------------
    # Final prints
    # -----------------------------------------------------------------------------

    print("\n" + "=" * 112)
    print("WROTE OUTPUTS")
    print("=" * 112)
    print(f"- {out_report}")
    print(f"- {out_clusters_csv}")
    print(f"- {out_contrib_csv}")
    print(f"- {out_scores_csv}")
    print(f"- {out_global_csv}")
    print(f"- {out_wide_csv}")
    print(f"- visuals_root: {visuals_dir}")
    print(f"  - system_summary: {vis_system_summary}")
    print(f"  - system_feature_groups: {vis_system_features}")
    print(f"  - taxonomy_rankings: {vis_system_taxonomy}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# TODO: improve plotting logic --> with reverse inference based on paths to show the suitability of disorder(/domains)
# TODO: later reverse the scoring direction ; is less intuitive for the LLM — might reduce reasoning performance
# TODO: check current output logic ; like this gets higher suitability score 'family_history_adnfle_in_first_degree_relatives' ; does not make sense...
# TODO: use radar plots as well... ; like multi-domain feasibility radar plots
