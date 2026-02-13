#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_summarize_predictor_rankings.py  (a.k.a. 02_summarize_results_predictors.py)

Compute weighted, hierarchical suitability rankings for ALL evaluated PREDICTORS (leaf nodes),
using the *scores-only* per-predictor caches (preferred) or the consolidated CSVs as fallback.

Primary goal
- Rank predictors by overall_suitability (higher is better).
- Provide correct, auditable weighted computations:
  - overall suitability
  - per-dimension risk/suitability
  - leaf-feature contributions to overall risk (weights applied)
  - sanity checks:
      sum(leaf_contributions) == overall_risk
      per-dimension contributions sum / dim_weight == dimension_risk

Visual goal (publish-ready barplots; no treemaps)
- Compare BIO vs PSYCHO vs SOCIAL (3 subplot rows) across:
  1) primary dimension suitability
  2) per-dimension feature-level Likert score plots (bar=mean, points=raw, whiskers=IQR)
- Bars are the secondary feature scores (Likert 1..9; higher = worse).
- Colors are stable across plots for identical (dimension,module,method,feature) keys.

Inputs (preferred)
- Reads per-predictor JSON caches from:
    .../PREDICTORS/results/responses/cache_evaluations/*.json
  These are written by 01_run_predictor_evaluation.py when --no-json-cache is NOT used.

Inputs (fallback)
- If cache_evaluations has no JSONs, loads the most recent:
    .../PREDICTORS/results/responses/tables/predictor_evaluations*_wide.csv
  or the base CSV if wide isn't available.

Ontology (layer/path inference)
- Reads PREDICTOR ontology JSON to build leaf->(layer, paths) mapping:
    /Users/.../PREDICTOR_ontology.json
- Layer for each predictor is resolved by priority:
    (1) cache field biopsychosocial_layer
    (2) first segment of full_path line (pipe-separated) if present
    (3) ontology-derived mapping by leaf label (best-effort)
  Final layer is one of: BIO / PSYCHO / SOCIAL / UNKNOWN.

Interpretation
- Likert9 scores represent PROBLEM likelihood (risk): 1=low risk, 9=high risk.
- Risk is normalized to [0,1].
- Suitability = 1 - risk.
- Overall suitability is the weighted mean of per-dimension suitability.

Outputs
- summary_dir (default set to PREDICTORS/results/summary):
    - predictor_rankings.csv
    - predictor_feature_contributions_long.csv
    - predictor_feature_scores_long.csv
    - global_feature_importance.csv
    - predictor_weighted_ranking_report.txt
- visuals_dir (default set to PREDICTORS/results/visuals), with clean substructure:
    - overall/
    - dimensions/
    - features/<dimension>/<subgroup>/
    - comparisons/layers/

Plotting is headless-safe (Agg backend).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import re
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

# Headless-safe plotting (works on macOS + CI/headless)
import matplotlib
matplotlib.use("Agg")  # must be set before pyplot import
import matplotlib.pyplot as plt


# -----------------------------
# Defaults (USER paths)
# -----------------------------

DEFAULT_EVAL_MODULE_PATH = Path(
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/"
    "multi_dimensional_feasibility_evaluation/PREDICTORS/utils/"
    "01_hierarchical_predictor_evaluation_modules.py"
)

# Fallbacks commonly used in your project structure
FALLBACK_EVAL_MODULE_PATHS = [
    Path(
        "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/"
        "multi_dimensional_feasibility_evaluation/PREDICTORS/utils/"
        "00_hierarchical_evaluation_modules.py"
    ),
    Path(
        "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/"
        "multi_dimensional_feasibility_evaluation/PREDICTORS/utils/"
        "00_hierarchical_predictor_evaluation_modules.py"
    ),
    Path(
        "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/"
        "multi_dimensional_feasibility_evaluation/PREDICTORS/utils/"
        "01_hierarchical_evaluation_modules.py"
    ),
]

DEFAULT_RESPONSES_DIR = Path(
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/"
    "multi_dimensional_feasibility_evaluation/PREDICTORS/results/responses"
)

DEFAULT_CACHE_DIR = DEFAULT_RESPONSES_DIR / "cache_evaluations"
DEFAULT_TABLES_DIR = DEFAULT_RESPONSES_DIR / "tables"

DEFAULT_SUMMARY_DIR = Path(
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/"
    "multi_dimensional_feasibility_evaluation/PREDICTORS/results/summary"
)

DEFAULT_VISUALS_DIR = Path(
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/"
    "multi_dimensional_feasibility_evaluation/PREDICTORS/results/visuals"
)

DEFAULT_PREDICTOR_ONTOLOGY_JSON = Path(
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/SystemComponents/PHOENIX_ontology/separate/"
    "01_raw/PREDICTOR/steps/01_raw/aggregated/PREDICTOR_ontology.json"
)


# -----------------------------
# Constants / helpers
# -----------------------------

EPS = 1e-9

LAYERS = ("BIO", "PSYCHO", "SOCIAL")
UNKNOWN_LAYER = "UNKNOWN"

# deterministic jitter seed default
DEFAULT_RNG_SEED = 42


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


def short(s: str, n: int = 120) -> str:
    s = str(s).replace("\n", " ").strip()
    return s if len(s) <= n else (s[: n - 1] + "…")


def normalize_likert_to_unit_interval(score_1_to_9: int) -> float:
    s = int(score_1_to_9)
    assert 1 <= s <= 9, f"Likert score out of range 1..9: {s}"
    return (float(s) - 1.0) / 8.0


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(float(v) for v in weights.values()))
    assert total > EPS, f"Cannot normalize weights; sum too small: {total}"
    return {k: float(v) / total for k, v in weights.items()}


def _safe_filename(s: str) -> str:
    s = str(s).strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "plot"


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


# -----------------------------
# Dynamic import + Pydantic rebuild
# -----------------------------

def resolve_eval_module_path(p: Path) -> Path:
    if p.exists():
        return p
    if DEFAULT_EVAL_MODULE_PATH.exists():
        return DEFAULT_EVAL_MODULE_PATH
    for alt in FALLBACK_EVAL_MODULE_PATHS:
        if alt.exists():
            return alt
    raise FileNotFoundError(
        "Predictor evaluation module not found. Tried:\n"
        f"- {p}\n- {DEFAULT_EVAL_MODULE_PATH}\n"
        + "\n".join([f"- {a}" for a in FALLBACK_EVAL_MODULE_PATHS])
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

    if hasattr(mod, "PredictorEvaluation"):
        ce = getattr(mod, "PredictorEvaluation")
        if isinstance(ce, type) and issubclass(ce, PydBaseModel):
            try:
                try:
                    ce.model_rebuild(force=True, _types_namespace=ns)  # type: ignore[arg-type]
                except TypeError:
                    ce.model_rebuild(force=True)
            except Exception as e:
                raise AssertionError(f"PredictorEvaluation.model_rebuild() failed: {repr(e)}") from e

    if failed:
        print(f"[{utc_now_iso()}] WARNING: model_rebuild failed for {len(failed)} models; showing first 10:")
        for s in failed[:10]:
            print(f"  - {s}")
    print(f"[{utc_now_iso()}] Pydantic rebuild complete: rebuilt_models={rebuilt} failed={len(failed)}")


def import_eval_module(path: Path) -> Any:
    path = resolve_eval_module_path(path)
    spec = importlib.util.spec_from_file_location("hier_eval_predictors", str(path))
    assert spec is not None and spec.loader is not None, f"Could not load module spec: {path}"

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    rebuild_pydantic_models_in_module(mod)

    assert hasattr(mod, "PredictorEvaluation"), "Module missing PredictorEvaluation"
    assert hasattr(mod, "compute_overall_suitability"), "Module missing compute_overall_suitability"
    assert hasattr(mod, "default_dimension_weights"), "Module missing default_dimension_weights"
    assert hasattr(mod, "DataCollectionAggregation"), "Module missing DataCollectionAggregation"
    return mod


# -----------------------------
# Ontology parsing (leaf -> paths, layer inference)
# -----------------------------

@dataclass(frozen=True)
class OntologyLeafInfo:
    label: str
    layer: str
    path_segments: List[str]   # including layer
    path_str: str              # "BIO | A | ... | Leaf"


def build_leaf_index_from_ontology(ontology_json: Dict[str, Any]) -> Dict[str, List[OntologyLeafInfo]]:
    """
    Traverse the ontology JSON:
      - root keys include BIO/PSYCHO/SOCIAL
      - leaves are dicts with {} (or empty dict) as value
    Returns mapping: leaf_label -> list[OntologyLeafInfo] (could be >1 if duplicates exist).
    """
    out: Dict[str, List[OntologyLeafInfo]] = {}

    def walk(node: Any, segs: List[str]) -> None:
        if isinstance(node, dict):
            if len(node) == 0:
                # empty dict: treat as leaf at current path (leaf label is last segment)
                if segs:
                    leaf = segs[-1]
                    layer = segs[0] if segs and segs[0] in LAYERS else UNKNOWN_LAYER
                    info = OntologyLeafInfo(
                        label=leaf,
                        layer=layer,
                        path_segments=list(segs),
                        path_str=" | ".join(segs),
                    )
                    out.setdefault(leaf, []).append(info)
                return

            for k, v in node.items():
                walk(v, segs + [str(k)])
        else:
            # unexpected primitive; ignore
            return

    for root_k, root_v in ontology_json.items():
        walk(root_v, [str(root_k)])

    return out


def parse_pipe_path(s: str) -> Tuple[str, List[str]]:
    """
    Given "BIO | A | B | Leaf", return (layer, segments_without_layer).
    """
    parts = [p.strip() for p in str(s).split("|")]
    parts = [p for p in parts if p]
    if not parts:
        return UNKNOWN_LAYER, []
    layer = parts[0] if parts[0] in LAYERS else UNKNOWN_LAYER
    segs = parts[1:] if layer != UNKNOWN_LAYER else parts
    return layer, segs


def resolve_layer(
    explicit_layer: Optional[str],
    full_path: Optional[str],
    leaf_label: Optional[str],
    leaf_index: Optional[Dict[str, List[OntologyLeafInfo]]],
) -> str:
    """
    Resolve BIO/PSYCHO/SOCIAL/UNKNOWN with priority:
      1) explicit_layer if valid
      2) parse from full_path if present
      3) ontology-derived by leaf_label (best-effort)
    """
    if explicit_layer in LAYERS:
        return str(explicit_layer)

    if full_path:
        layer, _ = parse_pipe_path(full_path)
        if layer in LAYERS:
            return layer

    if leaf_label and leaf_index:
        infos = leaf_index.get(leaf_label, [])
        # If unique layer exists, use it; else UNKNOWN
        layers = sorted({i.layer for i in infos if i.layer in LAYERS})
        if len(layers) == 1:
            return layers[0]

    return UNKNOWN_LAYER


def resolve_best_path_str(
    full_path: Optional[str],
    leaf_label: Optional[str],
    leaf_index: Optional[Dict[str, List[OntologyLeafInfo]]],
) -> str:
    """
    For reporting only:
      - prefer full_path if available
      - else use ontology-derived first path
      - else leaf_label
    """
    if full_path and str(full_path).strip():
        return str(full_path).strip()

    if leaf_label and leaf_index:
        infos = leaf_index.get(leaf_label, [])
        if infos:
            return infos[0].path_str

    return str(leaf_label or "").strip()


# -----------------------------
# Input reading: caches preferred, CSV fallback
# -----------------------------

def iter_json_files(dir_path: Path) -> List[Path]:
    if not dir_path.exists() or not dir_path.is_dir():
        return []
    return sorted([p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() == ".json"])


@dataclass(frozen=True)
class PredictorCache:
    cache_path: Path
    predictor_hash: str
    predictor_id: str
    label: str
    biopsychosocial_layer: Optional[str]
    full_path: Optional[str]
    path_segments: Optional[List[str]]
    evaluation: Dict[str, Any]
    cached_at_utc: Optional[str]


def load_predictor_cache(path: Path) -> PredictorCache:
    payload = read_json(path)
    assert isinstance(payload, dict), f"Cache JSON root must be object: {path}"

    predictor_hash = str(payload.get("predictor_hash", "")) or ""
    predictor_id = str(payload.get("predictor_id", "")) or ""
    label = str(payload.get("label", payload.get("predictor_label", ""))) or ""
    biopsychosocial_layer = payload.get("biopsychosocial_layer", None)
    biopsychosocial_layer = str(biopsychosocial_layer) if isinstance(biopsychosocial_layer, str) else None
    full_path = payload.get("full_path", None)
    full_path = str(full_path) if isinstance(full_path, str) else None
    path_segments = payload.get("path_segments", None)
    if isinstance(path_segments, list):
        path_segments = [str(x) for x in path_segments]
    else:
        path_segments = None

    cached_at_utc = payload.get("cached_at_utc", None)
    cached_at_utc = str(cached_at_utc) if isinstance(cached_at_utc, str) else None

    evaluation = payload.get("evaluation", None)
    assert isinstance(evaluation, dict), f"'evaluation' missing or not an object in {path}"

    # Robustness: infer missing label from predictor_id format "Leaf::hash8"
    if not label and predictor_id and "::" in predictor_id:
        label = predictor_id.split("::", 1)[0].strip()

    # Robustness: infer missing hash from filename if possible
    if not predictor_hash:
        m = re.search(r"([0-9a-f]{40})", path.name, flags=re.IGNORECASE)
        if m:
            predictor_hash = m.group(1)

    return PredictorCache(
        cache_path=path,
        predictor_hash=predictor_hash,
        predictor_id=predictor_id,
        label=label,
        biopsychosocial_layer=biopsychosocial_layer,
        full_path=full_path,
        path_segments=path_segments,
        evaluation=evaluation,
        cached_at_utc=cached_at_utc,
    )


def find_latest_csv(pattern: str, tables_dir: Path) -> Optional[Path]:
    if not tables_dir.exists() or not tables_dir.is_dir():
        return None
    candidates = sorted(tables_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def load_predictors_from_csv_or_wide(tables_dir: Path) -> List[PredictorCache]:
    """
    Fallback loader:
      - try latest wide csv first (contains flattened evaluation)
      - else latest base csv with evaluation_json
    Produces PredictorCache-like objects (cache_path points to csv file for traceability).
    """
    wide = find_latest_csv("predictor_evaluations*_wide.csv", tables_dir)
    base = find_latest_csv("predictor_evaluations*.csv", tables_dir)

    use = wide or base
    if not use:
        return []

    df = pd.read_csv(use, dtype="string")

    # Prefer base CSV if it exists because it has evaluation_json directly
    if wide is not None and base is not None:
        # If both exist and base is newer than wide, use base
        if base.stat().st_mtime >= wide.stat().st_mtime:
            use = base
            df = pd.read_csv(use, dtype="string")

    caches: List[PredictorCache] = []
    if "evaluation_json" in df.columns:
        # base csv
        for _, r in df.iterrows():
            ej = r.get("evaluation_json", "")
            try:
                ev = json.loads(ej) if isinstance(ej, str) and ej.strip() else {}
            except Exception:
                ev = {}
            caches.append(
                PredictorCache(
                    cache_path=use,
                    predictor_hash=str(r.get("predictor_hash", "") or ""),
                    predictor_id=str(r.get("predictor_id", "") or ""),
                    label=str(r.get("predictor_label", "") or ""),
                    biopsychosocial_layer=str(r.get("biopsychosocial_layer", "") or "") or None,
                    full_path=str(r.get("full_path", "") or "") or None,
                    path_segments=None,
                    evaluation=ev,
                    cached_at_utc=str(r.get("cached_at_utc", "") or "") or None,
                )
            )
    else:
        # wide csv (no evaluation_json); we reconstruct evaluation dict by unflattening is non-trivial.
        # In practice, your pipeline writes evaluation_json in base csv, so this is a last resort.
        raise RuntimeError(
            f"Wide CSV fallback selected ({use}), but it does not contain evaluation_json. "
            f"Use the base CSV or JSON caches. columns={list(df.columns)[:25]}..."
        )

    return caches


# -----------------------------
# Computation: contributions + raw scores
# -----------------------------

@dataclass
class ContributionRow:
    predictor_id: str
    predictor_hash: str
    label: str
    layer: str
    path_str: str
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
    predictor_id: str
    predictor_hash: str
    label: str
    layer: str
    path_str: str
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


def compute_contributions_and_raw_scores_for_predictor(
    pc: PredictorCache,
    e: Any,
    mod: Any,
    layer: str,
    path_str: str,
) -> Tuple[Dict[str, Any], List[ContributionRow], List[FeatureScoreRow]]:
    """
    Returns:
      breakdown (from module.compute_overall_suitability)
      contributions: leaf-feature weighted contributions to overall risk
      raw_scores: leaf-feature raw likert + risk_unit (no weighting)
    """
    dim_w = normalize_weights({k: float(v) for k, v in mod.default_dimension_weights().items()})
    assert abs(sum(dim_w.values()) - 1.0) < 1e-6, f"Dimension weights must sum to 1: sum={sum(dim_w.values())}"

    breakdown_model = mod.compute_overall_suitability(e)
    breakdown = breakdown_model.model_dump()

    contribs: List[ContributionRow] = []
    raw_rows: List[FeatureScoreRow] = []

    def add_leaf_rows(
        dimension: str,
        module_name: str,
        method_name: str,
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
                    predictor_id=pc.predictor_id,
                    predictor_hash=pc.predictor_hash,
                    label=pc.label,
                    layer=layer,
                    path_str=path_str,
                    dimension=dimension,
                    module=module_name,
                    method=method_name,
                    feature=feat,
                    likert_score=int(s),
                    risk_unit=float(ru),
                )
            )

        # Weighted contributions require feature weights
        if not compute_contrib:
            return
        assert feat_w is not None, "feat_w required when compute_contrib=True"
        assert dimension in dim_w, f"Unknown dimension key: {dimension}"
        dw = float(dim_w[dimension])
        feat_w_n = normalize_weights({k: float(v) for k, v in feat_w.items()})
        assert set(scores.keys()) == set(feat_w_n.keys()), (
            f"Feature keys mismatch for {dimension}/{module_name}/{method_name}: "
            f"{set(scores.keys()) ^ set(feat_w_n.keys())}"
        )

        for feat, s in scores.items():
            ru = normalize_likert_to_unit_interval(int(s))
            wp = dw * float(extra_weight) * float(feat_w_n[feat])
            contribs.append(
                ContributionRow(
                    predictor_id=pc.predictor_id,
                    predictor_hash=pc.predictor_hash,
                    label=pc.label,
                    layer=layer,
                    path_str=path_str,
                    dimension=dimension,
                    module=module_name,
                    method=method_name,
                    feature=feat,
                    likert_score=int(s),
                    risk_unit=float(ru),
                    weight_product=float(wp),
                    contribution_overall_risk=float(wp * ru),
                )
            )

    # --- Dimension 1: Mathematical suitability (flat)
    add_leaf_rows(
        "mathematical_suitability",
        "mathematical_suitability",
        "",
        {k: int(v) for k, v in e.mathematical_suitability.scores.model_dump().items()},
        feat_w=mod.MathematicalSuitability.default_weights(),
        extra_weight=1.0,
        compute_contrib=True,
    )

    # --- Dimension 2: Data collection feasibility (methods)
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
    method_w_all = normalize_weights(
        {k: float(v) for k, v in mod.DataCollectionFeasibility.default_method_weights().items()}
    )

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

    assert available, f"No data-collection methods present for predictor={pc.predictor_id}"

    # Raw rows for ALL available methods
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

    # --- Dimension 3: Validity threats (module-weighted)
    vt = e.validity_threats
    vt_module_w = normalize_weights({k: float(v) for k, v in mod.ValidityThreatWeights().as_dict().items()})
    vt_modules: List[Tuple[str, Any]] = [
        ("response_bias", mod.ResponseBiasRisk),
        ("insight_capacity", mod.InsightReportingCapacityRisk),
        ("measurement_validity", mod.MeasurementValidityRisk),
    ]
    available_mods = [mn for mn, _ in vt_modules if getattr(vt, mn, None) is not None]
    assert available_mods, f"No validity submodules present for predictor={pc.predictor_id}"
    total_vt = sum(float(vt_module_w[mn]) for mn in available_mods)
    assert total_vt > EPS, "Sum of validity module weights is 0."

    # Raw always
    for mn, _cls in vt_modules:
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

    # Contributions: module-weighted
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

    # --- Dimension 4: Treatment translation (flat)
    add_leaf_rows(
        "treatment_translation",
        "treatment_translation",
        "",
        {k: int(v) for k, v in e.treatment_translation.scores.model_dump().items()},
        feat_w=mod.TreatmentTranslationPotential.default_weights(),
        extra_weight=1.0,
        compute_contrib=True,
    )

    # --- Dimension 5: EU regulatory risk (module-weighted)
    rr = e.eu_regulatory_risk
    rr_module_w = normalize_weights({k: float(v) for k, v in mod.RegulatoryModuleWeights().as_dict().items()})
    rr_modules: List[Tuple[str, Any]] = [
        ("gdpr", mod.GDPRComplianceRisk),
        ("eu_ai_act", mod.EUAIActRisk),
        ("medical_device", mod.MedicalDeviceRegRisk),
        ("eprivacy", mod.ePrivacyRisk),
        ("cybersecurity", mod.CybersecurityRisk),
    ]
    available_mods = [mn for mn, _ in rr_modules if getattr(rr, mn, None) is not None]
    assert available_mods, f"No regulatory submodules present for predictor={pc.predictor_id}"
    total_rr = sum(float(rr_module_w[mn]) for mn in available_mods)
    assert total_rr > EPS, "Sum of regulatory module weights is 0."

    # Raw always
    for mn, _cls in rr_modules:
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

    # Contributions: module-weighted
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

    # --- Dimension 6: Scientific utility (flat)
    add_leaf_rows(
        "scientific_utility",
        "scientific_utility",
        "",
        {k: int(v) for k, v in e.scientific_utility.scores.model_dump().items()},
        feat_w=mod.ScientificUtility.default_weights(),
        extra_weight=1.0,
        compute_contrib=True,
    )

    # Sanity 1: overall risk equals sum contributions
    overall_risk_from_breakdown = 1.0 - float(breakdown["overall_suitability"])
    overall_risk_from_contrib = float(sum(r.contribution_overall_risk for r in contribs))
    assert abs(overall_risk_from_breakdown - overall_risk_from_contrib) < 1e-6, (
        f"Overall risk mismatch for predictor={pc.predictor_id}: "
        f"breakdown={overall_risk_from_breakdown} contrib_sum={overall_risk_from_contrib}"
    )

    # Sanity 2: per-dimension risk matches dimension contribution sum / dim_weight
    by_dim_risk = breakdown.get("by_dimension_risk", {})
    for dim, dw in dim_w.items():
        if dim not in by_dim_risk:
            continue
        csum = sum(r.contribution_overall_risk for r in contribs if r.dimension == dim)
        if dw <= EPS:
            continue
        dim_risk_from_contrib = float(csum) / float(dw)
        dim_risk_from_breakdown = float(by_dim_risk[dim])
        assert abs(dim_risk_from_contrib - dim_risk_from_breakdown) < 1e-6, (
            f"Dimension risk mismatch predictor={pc.predictor_id} dim={dim}: "
            f"breakdown={dim_risk_from_breakdown} contrib_based={dim_risk_from_contrib}"
        )

    return breakdown, contribs, raw_rows


# -----------------------------
# Console reporting (hierarchical)
# -----------------------------

def format_tree_lines(lines: List[Tuple[int, str]]) -> str:
    out: List[str] = []
    for indent, text in lines:
        if indent <= 0:
            out.append(text)
        else:
            out.append(("  " * (indent - 1)) + "└─ " + text)
    return "\n".join(out)


def print_predictor_hierarchy(
    row: Dict[str, Any],
    df_contrib_pred: pd.DataFrame,
    top_per_dimension: int = 6,
) -> None:
    pid = str(row.get("predictor_id", ""))
    label = str(row.get("label", ""))
    layer = str(row.get("layer", UNKNOWN_LAYER))
    suit = float(row.get("overall_suitability", float("nan")))
    path_str = str(row.get("path_str", ""))

    lines: List[Tuple[int, str]] = []
    lines.append((0, f"{pid} | layer={layer:<6} | overall_suitability={suit:.4f} | label={short(label, 80)}"))
    if path_str:
        lines.append((1, f"path: {short(path_str, 220)}"))

    dim_suit_cols = sorted([k for k in row.keys() if isinstance(k, str) and k.startswith("suitability.")])
    if dim_suit_cols:
        lines.append((1, "dimensions:"))
        for sc in dim_suit_cols:
            dk = sc[len("suitability."):]
            rc = f"risk.{dk}"
            ds = float(row.get(sc, float("nan")))
            dr = float(row.get(rc, float("nan")))
            lines.append((2, f"{dk}: suitability={ds:.4f} | risk={dr:.4f}"))

            sub = df_contrib_pred[df_contrib_pred["dimension"] == dk].copy()
            if len(sub) == 0:
                continue
            sub = sub.sort_values("contribution_overall_risk", ascending=False).head(top_per_dimension)
            lines.append((3, "top_contributors (weighted overall-risk):"))
            for _, rr in sub.iterrows():
                module = str(rr.get("module", ""))
                method = str(rr.get("method", ""))
                feat = str(rr.get("feature", ""))
                score = int(rr.get("likert_score", -1))
                contrib = float(rr.get("contribution_overall_risk", 0.0))
                path = module
                if method and method != "nan":
                    path += f":{method}"
                path += f".{feat}"
                lines.append((4, f"{path} | score={score} | contrib={contrib:.6f}"))

    print(format_tree_lines(lines), flush=True)


# -----------------------------
# Plotting utilities (publish-ready barplots with raw points)
# -----------------------------

# add this import near your other matplotlib imports
import matplotlib.colors as mcolors

# add this near your helpers / globals
_FEATURE_COLOR_CACHE: Dict[str, Tuple[float, float, float, float]] = {}

def stable_color_for_key(key: str) -> Tuple[float, float, float, float]:
    """
    Deterministic, high-cardinality colors:
    - Same key -> same RGBA every run
    - Avoids tab20 collisions when you have >20 features
    """
    if key in _FEATURE_COLOR_CACHE:
        return _FEATURE_COLOR_CACHE[key]

    h = int(sha1_text(key)[:12], 16)

    # HSV components derived from hash (stable, wide variety)
    hue = (h % 360) / 360.0                         # 0..1
    sat = 0.55 + (((h >> 9)  % 41) / 100.0)         # 0.55..0.96
    val = 0.65 + (((h >> 17) % 31) / 100.0)         # 0.65..0.96

    r, g, b = mcolors.hsv_to_rgb((hue, sat, val))
    rgba = (float(r), float(g), float(b), 1.0)

    _FEATURE_COLOR_CACHE[key] = rgba
    return rgba



def plot_overall_ranking(
    df_sorted: pd.DataFrame,
    out_path: Path,
    top_n: int = 40,
) -> None:
    ensure_dir(out_path.parent)
    n = min(int(top_n), len(df_sorted))
    if n <= 0:
        return
    df_top = df_sorted.head(n).copy()

    fig_h = max(7, 0.28 * n + 3)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    ax.barh(list(reversed(df_top["predictor_id"].tolist())), list(reversed(df_top["overall_suitability"].tolist())))
    ax.set_xlabel("overall_suitability (higher = better)")
    ax.set_ylabel("predictor_id")
    ax.set_title(f"Top {n} predictors by overall suitability")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.6, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_overall_ranking_by_layer(
    df: pd.DataFrame,
    out_dir: Path,
    top_n: int = 30,
) -> None:
    ensure_dir(out_dir)
    for layer in list(LAYERS) + [UNKNOWN_LAYER]:
        sub = df[df["layer"] == layer].copy()
        if len(sub) == 0:
            continue
        sub = sub.sort_values("overall_suitability", ascending=False).reset_index(drop=True)
        out = out_dir / f"top_{top_n}_overall_suitability__{_safe_filename(layer)}.png"
        plot_overall_ranking(sub, out, top_n=top_n)


def plot_primary_dimensions_with_points_by_layer(
    df_clusters: pd.DataFrame,
    dimension_keys: List[str],
    value_prefix: str,  # "suitability." or "risk."
    out_path: Path,
    title: str,
    x_label: str,
    rng_seed: int = DEFAULT_RNG_SEED,
) -> None:
    """
    3-row subplot comparison (BIO/PSYCHO/SOCIAL).
      - Bars = mean across predictors for that layer
      - Points = raw predictor values (jittered)
    """
    ensure_dir(out_path.parent)

    layers = list(LAYERS)
    # Keep dimension order as provided by module
    dims = [d for d in dimension_keys if f"{value_prefix}{d}" in df_clusters.columns]
    if not dims:
        return

    # NEW: stable per-dimension colors (same dim -> same color everywhere)
    dim_colors = [stable_color_for_key(f"PRIMARY_DIM|{d}") for d in dims]

    fig_h = max(8, 0.45 * len(dims) + 6)
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(16, fig_h), sharex=True)

    rng = random.Random(int(rng_seed))
    y = list(range(len(dims)))

    for ax, layer in zip(axes, layers):
        sub = df_clusters[df_clusters["layer"] == layer].copy()
        if len(sub) == 0:
            ax.set_axis_off()
            ax.set_title(f"{layer} (no data)")
            continue

        means: List[float] = []
        raw_vals_per_dim: List[List[float]] = []

        for d in dims:
            col = f"{value_prefix}{d}"
            vals = sub[col].astype(float).tolist()
            vals = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
            raw_vals_per_dim.append(vals)
            means.append(float(sum(vals) / len(vals)) if vals else float("nan"))

        # Plot ONCE, after means is filled
        ax.barh(y, means, color=dim_colors)

        ax.set_yticks(y)
        ax.set_yticklabels(dims)
        ax.invert_yaxis()
        ax.grid(True, axis="x", linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_title(f"{layer} (bar = mean; points = raw predictors)")

        # overlay raw points
        for i, vals in enumerate(raw_vals_per_dim):
            if not vals:
                continue
            jitter = [(i + (rng.random() - 0.5) * 0.22) for _ in vals]
            ax.scatter(vals, jitter, s=10, alpha=0.25, color="black", linewidths=0)

    axes[-1].set_xlabel(x_label)
    fig.suptitle(title, y=0.995, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_feature_scores_by_layer(
    df_scores_subset: pd.DataFrame,
    out_path: Path,
    title: str,
    feature_key_cols: Tuple[str, str, str, str],  # (dimension,module,method,feature) for stable key
    max_features: int = 30,
    rng_seed: int = DEFAULT_RNG_SEED,
) -> None:
    """
    For a fixed subgroup (dimension + module + method), create a 3-row subplot figure:
      - Each row is a layer (BIO/PSYCHO/SOCIAL)
      - X axis: Likert 1..9 (higher = worse)
      - Bars: mean Likert score per feature
      - Whiskers: IQR (q25..q75)
      - Points: raw predictor scores for that layer
    Colors are stable per (dimension,module,method,feature).
    """
    ensure_dir(out_path.parent)
    if len(df_scores_subset) == 0:
        return

    df = df_scores_subset.copy()
    df["likert_score"] = df["likert_score"].astype(int)

    # Build a stable feature label and a stable feature key for coloring
    dim_col, mod_col, meth_col, feat_col = feature_key_cols

    def feature_label(r: pd.Series) -> str:
        module = str(r.get(mod_col, "") or "")
        method = str(r.get(meth_col, "") or "")
        feat = str(r.get(feat_col, "") or "")
        if method and method != "nan":
            return f"{module}:{method}.{feat}"
        return f"{module}.{feat}"

    def feature_key(r: pd.Series) -> str:
        return f"{r.get(dim_col,'')}|{r.get(mod_col,'')}|{r.get(meth_col,'')}|{r.get(feat_col,'')}"

    df["feature_label"] = df.apply(feature_label, axis=1)
    df["feature_key"] = df.apply(feature_key, axis=1)

    # Determine global feature ordering by mean (across all layers)
    agg_all = (
        df.groupby(["feature_label", "feature_key"], dropna=False)["likert_score"]
        .mean()
        .reset_index()
        .sort_values("likert_score", ascending=False)
        .reset_index(drop=True)
    )

    if len(agg_all) > max_features:
        agg_all = agg_all.head(max_features).copy()

    feat_labels = agg_all["feature_label"].tolist()
    feat_keys = agg_all["feature_key"].tolist()

    layers = list(LAYERS)
    fig_h = max(10, 0.42 * len(feat_labels) + 8)
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(18, fig_h), sharex=True)

    rng = random.Random(int(rng_seed))
    y = list(range(len(feat_labels)))

    for ax, layer in zip(axes, layers):
        sub = df[df["layer"] == layer].copy()

        # Stats per feature within this layer
        means: List[float] = []
        q25s: List[float] = []
        q75s: List[float] = []
        raw_vals_per_feat: List[List[float]] = []

        for fl in feat_labels:
            vals = sub[sub["feature_label"] == fl]["likert_score"].astype(float).tolist()
            vals = [v for v in vals if not math.isnan(v)]
            raw_vals_per_feat.append(vals)
            if vals:
                s = pd.Series(vals)
                means.append(float(s.mean()))
                q25s.append(float(s.quantile(0.25)))
                q75s.append(float(s.quantile(0.75)))
            else:
                means.append(float("nan"))
                q25s.append(float("nan"))
                q75s.append(float("nan"))

        colors = [stable_color_for_key(k) for k in feat_keys]
        ax.barh(y, means, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels([short(s, 90) for s in feat_labels])
        ax.invert_yaxis()
        ax.set_xlim(1.0, 9.0)
        ax.grid(True, axis="x", linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_title(f"{layer} (bar = mean; whiskers = IQR; points = raw predictors)")

        # IQR whiskers
        for i, (a, b) in enumerate(zip(q25s, q75s)):
            if not (math.isnan(a) or math.isnan(b)):
                ax.hlines(i, a, b, linewidth=2.0, alpha=0.9, color="black")

        # raw points
        for i, vals in enumerate(raw_vals_per_feat):
            if not vals:
                continue
            jitter = [(i + (rng.random() - 0.5) * 0.22) for _ in vals]
            ax.scatter(vals, jitter, s=10, alpha=0.25, color="black", linewidths=0)

    axes[-1].set_xlabel("Likert score (problem likelihood; 1=low, 9=high)")
    fig.suptitle(title, y=0.995, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--eval-module", type=str, default=str(DEFAULT_EVAL_MODULE_PATH))
    ap.add_argument("--responses-dir", type=str, default=str(DEFAULT_RESPONSES_DIR))
    ap.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    ap.add_argument("--tables-dir", type=str, default=str(DEFAULT_TABLES_DIR))

    ap.add_argument("--summary-dir", type=str, default=str(DEFAULT_SUMMARY_DIR))
    ap.add_argument("--visuals-dir", type=str, default=str(DEFAULT_VISUALS_DIR))

    ap.add_argument("--ontology-json", type=str, default=str(DEFAULT_PREDICTOR_ONTOLOGY_JSON))

    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--top-n-hierarchy", type=int, default=10)
    ap.add_argument("--top-per-dimension", type=int, default=6)

    ap.add_argument("--plot-top-overall", type=int, default=40)
    ap.add_argument("--plot-top-overall-per-layer", type=int, default=30)
    ap.add_argument("--plot-max-features", type=int, default=30)

    ap.add_argument("--rng-seed", type=int, default=DEFAULT_RNG_SEED)

    args = ap.parse_args()

    eval_module_path = resolve_eval_module_path(Path(args.eval_module))
    responses_dir = Path(args.responses_dir)
    cache_dir = Path(args.cache_dir)
    tables_dir = Path(args.tables_dir)

    summary_dir = Path(args.summary_dir)
    visuals_dir = Path(args.visuals_dir)

    ontology_path = Path(args.ontology_json)

    ensure_dir(summary_dir)
    ensure_dir(visuals_dir)

    # Visuals substructure
    out_vis_overall = visuals_dir / "overall"
    out_vis_dims = visuals_dir / "dimensions"
    out_vis_features = visuals_dir / "features"
    out_vis_comp = visuals_dir / "comparisons" / "layers"
    for p in [out_vis_overall, out_vis_dims, out_vis_features, out_vis_comp]:
        ensure_dir(p)

    # Ontology index (optional but recommended)
    leaf_index: Optional[Dict[str, List[OntologyLeafInfo]]] = None
    if ontology_path.exists():
        try:
            onto = read_json(ontology_path)
            if isinstance(onto, dict):
                leaf_index = build_leaf_index_from_ontology(onto)
                print(f"[{utc_now_iso()}] Loaded ontology leaf index: leaves={len(leaf_index)} from {ontology_path}")
        except Exception as e:
            print(f"[{utc_now_iso()}] WARNING: failed loading ontology json: {repr(e)} ({ontology_path})")
            leaf_index = None
    else:
        print(f"[{utc_now_iso()}] WARNING: ontology json not found: {ontology_path}")

    # Load eval module
    print(f"[{utc_now_iso()}] Loading evaluation module: {eval_module_path}")
    mod = import_eval_module(eval_module_path)
    PredictorEvaluation = mod.PredictorEvaluation

    # Load predictor evaluations
    caches: List[PredictorCache] = []
    json_files = iter_json_files(cache_dir)
    if json_files:
        print(f"[{utc_now_iso()}] Reading predictor caches from JSON: n_files={len(json_files)} dir={cache_dir}")
        for p in json_files:
            try:
                caches.append(load_predictor_cache(p))
            except Exception as e:
                raise AssertionError(f"Failed parsing cache JSON {p}: {repr(e)}") from e
    else:
        print(f"[{utc_now_iso()}] No cache JSON files found in {cache_dir}; falling back to tables in {tables_dir}")
        caches = load_predictors_from_csv_or_wide(tables_dir)

    assert caches, f"No predictor evaluations found (cache_dir={cache_dir}, tables_dir={tables_dir})."

    # Compute rankings, contributions, raw feature scores
    cluster_rows: List[Dict[str, Any]] = []
    contrib_rows: List[Dict[str, Any]] = []
    score_rows: List[Dict[str, Any]] = []

    for pc in caches:
        # Validate evaluation schema
        try:
            e = PredictorEvaluation.model_validate(pc.evaluation)
        except Exception as ex:
            raise AssertionError(
                f"Predictor evaluation schema validation failed for source={pc.cache_path} predictor_id={pc.predictor_id}: {repr(ex)}"
            ) from ex

        # Resolve label (robust)
        label = pc.label
        if not label and pc.predictor_id and "::" in pc.predictor_id:
            label = pc.predictor_id.split("::", 1)[0].strip()

        # Resolve layer + best path string
        layer = resolve_layer(
            explicit_layer=pc.biopsychosocial_layer,
            full_path=pc.full_path,
            leaf_label=label,
            leaf_index=leaf_index,
        )
        path_str = resolve_best_path_str(pc.full_path, label, leaf_index)

        breakdown, contribs, raw_scores = compute_contributions_and_raw_scores_for_predictor(
            pc=pc,
            e=e,
            mod=mod,
            layer=layer,
            path_str=path_str,
        )

        # Per-predictor row
        row = {
            "predictor_id": pc.predictor_id,
            "predictor_hash": pc.predictor_hash,
            "label": label,
            "layer": layer,
            "path_str": path_str,
            "cached_at_utc": pc.cached_at_utc,
            "overall_suitability": float(breakdown["overall_suitability"]),
        }
        for k, v in breakdown["by_dimension_suitability"].items():
            row[f"suitability.{k}"] = float(v)
        for k, v in breakdown["by_dimension_risk"].items():
            row[f"risk.{k}"] = float(v)
        cluster_rows.append(row)

        # Contributions
        for r in contribs:
            contrib_rows.append(
                {
                    "predictor_id": r.predictor_id,
                    "predictor_hash": r.predictor_hash,
                    "label": r.label,
                    "layer": r.layer,
                    "path_str": r.path_str,
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

        # Raw scores
        for r in raw_scores:
            score_rows.append(
                {
                    "predictor_id": r.predictor_id,
                    "predictor_hash": r.predictor_hash,
                    "label": r.label,
                    "layer": r.layer,
                    "path_str": r.path_str,
                    "dimension": r.dimension,
                    "module": r.module,
                    "method": r.method,
                    "feature": r.feature,
                    "likert_score": r.likert_score,
                    "risk_unit": r.risk_unit,
                }
            )

    df_pred = pd.DataFrame(cluster_rows)
    df_contrib = pd.DataFrame(contrib_rows)
    df_scores = pd.DataFrame(score_rows)

    assert len(df_pred) == len(cluster_rows) and len(df_pred) > 0, "No predictor rows created."
    assert len(df_contrib) > 0, "No contributions computed."
    assert len(df_scores) > 0, "No feature scores extracted."

    # Write CSVs
    out_rankings_csv = summary_dir / "predictor_rankings.csv"
    out_contrib_csv = summary_dir / "predictor_feature_contributions_long.csv"
    out_scores_csv = summary_dir / "predictor_feature_scores_long.csv"
    out_global_csv = summary_dir / "global_feature_importance.csv"

    df_pred.to_csv(out_rankings_csv, index=False, encoding="utf-8")
    df_contrib.to_csv(out_contrib_csv, index=False, encoding="utf-8")
    df_scores.to_csv(out_scores_csv, index=False, encoding="utf-8")

    # Ranking
    df_sorted = df_pred.sort_values("overall_suitability", ascending=False).reset_index(drop=True)
    top_n = min(int(args.top_n), len(df_sorted))

    # Layer counts
    layer_counts = df_pred["layer"].value_counts(dropna=False).to_dict()

    print("\n" + "=" * 110)
    print("PREDICTOR RANKING (best -> worst)  |  overall_suitability higher = better")
    print("=" * 110)
    print(f"n_predictors={len(df_pred)}  layer_counts={layer_counts}")
    for i, r in df_sorted.head(top_n).iterrows():
        print(
            f"{i+1:>3}. {str(r['predictor_id']):<28} layer={str(r['layer']):<6} "
            f"overall_suitability={float(r['overall_suitability']):.4f}  "
            f"label={short(str(r['label']), 70)}"
        )

    print("\n" + "-" * 110)
    print("PREDICTOR RANKING (worst -> best)")
    print("-" * 110)
    worst = df_sorted.tail(top_n).sort_values("overall_suitability", ascending=True).reset_index(drop=True)
    for i, r in worst.iterrows():
        print(
            f"{i+1:>3}. {str(r['predictor_id']):<28} layer={str(r['layer']):<6} "
            f"overall_suitability={float(r['overall_suitability']):.4f}  "
            f"label={short(str(r['label']), 70)}"
        )

    # Hierarchical console print for top predictors
    print("\n" + "=" * 110)
    print(f"TOP {min(int(args.top_n_hierarchy), len(df_sorted))} PREDICTORS — HIERARCHICAL BREAKDOWN (weighted)")
    print("=" * 110)
    top_for_tree = df_sorted.head(min(int(args.top_n_hierarchy), len(df_sorted))).copy()
    for _, rr in top_for_tree.iterrows():
        pid = str(rr["predictor_id"])
        subc = df_contrib[df_contrib["predictor_id"] == pid].copy()
        row_dict = {k: rr[k] for k in rr.index}
        print_predictor_hierarchy(
            row=row_dict,
            df_contrib_pred=subc,
            top_per_dimension=int(args.top_per_dimension),
        )
        print("-" * 110)

    # Global feature importance (mean weighted contribution to overall risk)
    g = (
        df_contrib.groupby(["dimension", "module", "method", "feature"], dropna=False)["contribution_overall_risk"]
        .mean()
        .reset_index()
        .sort_values("contribution_overall_risk", ascending=False)
        .reset_index(drop=True)
    )
    g.to_csv(out_global_csv, index=False, encoding="utf-8")

    # Text report
    report_lines: List[str] = []
    report_lines.append("Predictor weighted ranking report")
    report_lines.append(f"generated_at_utc: {utc_now_iso()}")
    report_lines.append(f"eval_module: {eval_module_path}")
    report_lines.append(f"responses_dir: {responses_dir}")
    report_lines.append(f"cache_dir: {cache_dir}")
    report_lines.append(f"tables_dir: {tables_dir}")
    report_lines.append(f"ontology_json: {ontology_path}")
    report_lines.append(f"n_predictors: {len(df_pred)}")
    report_lines.append(f"layer_counts: {layer_counts}")
    report_lines.append("")
    report_lines.append("TOP PREDICTORS (overall_suitability)")
    report_lines.append("----------------------------------")
    for i, r in df_sorted.head(top_n).iterrows():
        report_lines.append(
            f"{i+1:>3}. {str(r['predictor_id']):<28} layer={str(r['layer']):<6} "
            f"suitability={float(r['overall_suitability']):.4f} label={short(str(r['label']), 80)}"
        )
        report_lines.append(f"     path: {short(str(r.get('path_str','')), 220)}")
    report_lines.append("")
    report_lines.append("BOTTOM PREDICTORS (overall_suitability)")
    report_lines.append("-------------------------------------")
    for i, r in worst.iterrows():
        report_lines.append(
            f"{i+1:>3}. {str(r['predictor_id']):<28} layer={str(r['layer']):<6} "
            f"suitability={float(r['overall_suitability']):.4f} label={short(str(r['label']), 80)}"
        )
        report_lines.append(f"     path: {short(str(r.get('path_str','')), 220)}")
    report_lines.append("")
    report_lines.append("GLOBAL FEATURE IMPORTANCE (mean contribution to overall risk)")
    report_lines.append("-----------------------------------------------------------")
    for i, r in g.head(140).iterrows():
        meth = str(r["method"]) if str(r["method"]) and str(r["method"]) != "nan" else "-"
        report_lines.append(
            f"{i+1:>3}. {r['dimension']} | {r['module']} | {meth} | {r['feature']} -> "
            f"mean_contrib_risk={float(r['contribution_overall_risk']):.6f}"
        )

    out_report = summary_dir / "predictor_weighted_ranking_report.txt"
    write_text(out_report, "\n".join(report_lines) + "\n")

    # -----------------------------
    # Plots
    # -----------------------------

    # Overall ranking plot (all predictors)
    plot_overall_ranking(
        df_sorted=df_sorted,
        out_path=out_vis_overall / "top_predictors_overall_suitability.png",
        top_n=int(args.plot_top_overall),
    )

    # Overall ranking per layer
    plot_overall_ranking_by_layer(
        df=df_pred,
        out_dir=out_vis_overall / "by_layer",
        top_n=int(args.plot_top_overall_per_layer),
    )

    # Primary dimension suitability comparison (3-row BIO/PSYCHO/SOCIAL)
    dim_keys = list(mod.default_dimension_weights().keys())
    plot_primary_dimensions_with_points_by_layer(
        df_clusters=df_pred,
        dimension_keys=dim_keys,
        value_prefix="suitability.",
        out_path=out_vis_dims / "primary_dimensions_suitability__BIO_PSYCHO_SOCIAL.png",
        title="Primary dimension suitability by biopsychosocial layer (BIO vs PSYCHO vs SOCIAL)",
        x_label="suitability (higher = better)",
        rng_seed=int(args.rng_seed),
    )

    # Primary dimension risk comparison
    plot_primary_dimensions_with_points_by_layer(
        df_clusters=df_pred,
        dimension_keys=dim_keys,
        value_prefix="risk.",
        out_path=out_vis_dims / "primary_dimensions_risk__BIO_PSYCHO_SOCIAL.png",
        title="Primary dimension risk by biopsychosocial layer (BIO vs PSYCHO vs SOCIAL)",
        x_label="risk (0..1; higher = worse)",
        rng_seed=int(args.rng_seed),
    )

    # Feature-level plots by dimension/subgroup (3-row layer comparison)
    max_feat = int(args.plot_max_features)

    # 1) Mathematical suitability (flat)
    dim = "mathematical_suitability"
    sub = df_scores[df_scores["dimension"] == dim].copy()
    if len(sub) > 0:
        outp = out_vis_features / _safe_filename(dim) / "features" / "feature_scores__BIO_PSYCHO_SOCIAL.png"
        plot_feature_scores_by_layer(
            df_scores_subset=sub,
            out_path=outp,
            title=f"{dim} — feature scores by layer (bar=mean Likert; points=raw; whiskers=IQR)",
            feature_key_cols=("dimension", "module", "method", "feature"),
            max_features=max_feat,
            rng_seed=int(args.rng_seed),
        )

    # 2) Data collection feasibility: per method
    dim = "data_collection_feasibility"
    sub = df_scores[df_scores["dimension"] == dim].copy()
    if len(sub) > 0:
        for method in sorted([m for m in sub["method"].dropna().astype(str).unique().tolist() if m and m != "nan"]):
            subm = sub[sub["method"].astype(str) == method].copy()
            if len(subm) == 0:
                continue
            outp = out_vis_features / _safe_filename(dim) / _safe_filename(method) / "feature_scores__BIO_PSYCHO_SOCIAL.png"
            plot_feature_scores_by_layer(
                df_scores_subset=subm,
                out_path=outp,
                title=f"{dim}:{method} — feature scores by layer (bar=mean Likert; points=raw; whiskers=IQR)",
                feature_key_cols=("dimension", "module", "method", "feature"),
                max_features=max_feat,
                rng_seed=int(args.rng_seed),
            )

    # 3) Validity threats: per module
    dim = "validity_threats"
    sub = df_scores[df_scores["dimension"] == dim].copy()
    if len(sub) > 0:
        for module in sorted([m for m in sub["module"].dropna().astype(str).unique().tolist() if m and m != "nan"]):
            subm = sub[sub["module"].astype(str) == module].copy()
            if len(subm) == 0:
                continue
            outp = out_vis_features / _safe_filename(dim) / _safe_filename(module) / "feature_scores__BIO_PSYCHO_SOCIAL.png"
            plot_feature_scores_by_layer(
                df_scores_subset=subm,
                out_path=outp,
                title=f"{dim}.{module} — feature scores by layer (bar=mean Likert; points=raw; whiskers=IQR)",
                feature_key_cols=("dimension", "module", "method", "feature"),
                max_features=max_feat,
                rng_seed=int(args.rng_seed),
            )

    # 4) Treatment translation (flat)
    dim = "treatment_translation"
    sub = df_scores[df_scores["dimension"] == dim].copy()
    if len(sub) > 0:
        outp = out_vis_features / _safe_filename(dim) / "features" / "feature_scores__BIO_PSYCHO_SOCIAL.png"
        plot_feature_scores_by_layer(
            df_scores_subset=sub,
            out_path=outp,
            title=f"{dim} — feature scores by layer (bar=mean Likert; points=raw; whiskers=IQR)",
            feature_key_cols=("dimension", "module", "method", "feature"),
            max_features=max_feat,
            rng_seed=int(args.rng_seed),
        )

    # 5) EU regulatory risk: per module
    dim = "eu_regulatory_risk"
    sub = df_scores[df_scores["dimension"] == dim].copy()
    if len(sub) > 0:
        for module in sorted([m for m in sub["module"].dropna().astype(str).unique().tolist() if m and m != "nan"]):
            subm = sub[sub["module"].astype(str) == module].copy()
            if len(subm) == 0:
                continue
            outp = out_vis_features / _safe_filename(dim) / _safe_filename(module) / "feature_scores__BIO_PSYCHO_SOCIAL.png"
            plot_feature_scores_by_layer(
                df_scores_subset=subm,
                out_path=outp,
                title=f"{dim}.{module} — feature scores by layer (bar=mean Likert; points=raw; whiskers=IQR)",
                feature_key_cols=("dimension", "module", "method", "feature"),
                max_features=max_feat,
                rng_seed=int(args.rng_seed),
            )

    # 6) Scientific utility (flat)
    dim = "scientific_utility"
    sub = df_scores[df_scores["dimension"] == dim].copy()
    if len(sub) > 0:
        outp = out_vis_features / _safe_filename(dim) / "features" / "feature_scores__BIO_PSYCHO_SOCIAL.png"
        plot_feature_scores_by_layer(
            df_scores_subset=sub,
            out_path=outp,
            title=f"{dim} — feature scores by layer (bar=mean Likert; points=raw; whiskers=IQR)",
            feature_key_cols=("dimension", "module", "method", "feature"),
            max_features=max_feat,
            rng_seed=int(args.rng_seed),
        )

    # Final prints
    print("\n" + "=" * 110)
    print("WROTE OUTPUTS")
    print("=" * 110)
    print(f"- {out_report}")
    print(f"- {out_rankings_csv}")
    print(f"- {out_contrib_csv}")
    print(f"- {out_scores_csv}")
    print(f"- {out_global_csv}")
    print(f"- visuals_dir: {visuals_dir}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# TODO:
# - Add "priority" composite metrics that incorporate clinical impact vs feasibility (orthogonal axis).
# - Add per-layer statistical summaries (e.g., bootstrap CI for dimension means) for publication.
# - If needed, enrich path resolution by matching full_path segments against ontology paths (for duplicates).
