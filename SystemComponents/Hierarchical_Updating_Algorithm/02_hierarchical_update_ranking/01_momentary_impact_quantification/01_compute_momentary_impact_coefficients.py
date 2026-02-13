#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_momentary_impact_coefficients.py

===============================================================================
SUMMARY (main logic)
===============================================================================
Goal
- For each profile produced by apply_tv_gVAR_network.py, compute a single overall
  MOMENTARY IMPACT COMPOSITE score at two levels:

  (A) EDGE-level composite impact:
      predictor_j  ->  criterion_i
      => one number per directed pair (j,i)

  (B) PREDICTOR-level composite impact:
      predictor_j  ->  whole set of criteria
      => one number per predictor j

"Momentary" / recency
- All time-varying inputs are aggregated using exponential time-decay over tvKS
  estimation points (estpoints in [0,1]).
- More recent points get higher weight. Half-life is a fraction of the [0,1] span.
  w(t) ∝ exp{-λ(1−t)},  λ = ln(2)/half_life.

Local metrics requirement (NEW)
- You requested that the "local metrics" from temporal_lagged_node_centrality.csv
  (pagerank, betweenness, closeness, eigenvector, katz, core_number,
   participation_coeff, etc.) MUST be used.
- We implement this by allocating a TOTAL SHARE of 0.4 to LOCAL METRICS
  at BOTH composite levels:
    * Edge-level composite:     0.6 base (beta/pcorr/reliability) + 0.4 local metrics
    * Predictor-level composite:0.6 base (edge aggregation + importance) + 0.4 local metrics

Important: data flow assertions
- The script will RAISE AssertionError (and print a clear console error) if:
  * tvKS_full_arrays.npz is missing or missing required arrays
  * temporal_lagged_node_centrality.csv is missing OR lacks required columns for local metrics
    (because you required local metrics usage)
  * profile has no valid predictors/criteria indices
- The assertions include exact file paths expected.

Inputs (from your tv-gVAR pipeline)
- method 1/numerical outputs/tvKS_full_arrays.npz:
    * Bhat(t): lagged coefficients (dst <- src), shape (m, p, p)
    * partial_corr(t): contemporaneous partial correlations, shape (m, p, p)
    * estpoints: shape (m,)
    * ci_low/ci_high (optional): shape (m, p, p)
- network_metrics/temporal_lagged_node_centrality.csv (REQUIRED in this version)
    * local metrics used: pagerank, betweenness, closeness, eigenvector, katz,
      core_number, participation_coeff, in_strength_abs, out_strength_abs, etc.
- network_metrics/predictor_importance_tv.csv (optional but used if present)
- network_metrics/criterion_dependence_tv.csv (optional)

Outputs (per profile)
- <output_root>/<profile_id>/edge_composite.csv
- <output_root>/<profile_id>/impact_matrix.csv
- <output_root>/<profile_id>/impact_matrix_signed.csv
- <output_root>/<profile_id>/predictor_composite.csv
- <output_root>/<profile_id>/overall_predictor_impact.json
- <output_root>/<profile_id>/momentary_impact.json
- <output_root>/<profile_id>/config_used.json

Default paths
- input_root:
    /Users/stijnvanseveren/PythonProjects/MASTERPROEF/Evaluation/04_initial_observation_analysis/01_time_series_analysis/network
- output_root:
    /Users/stijnvanseveren/PythonProjects/MASTERPROEF/Evaluation/04_initial_observation_analysis/02_momentary_impact_coefficients
===============================================================================
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Default roots requested by the user
# -----------------------------------------------------------------------------
DEFAULT_INPUT_ROOT = (
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/Evaluation/"
    "04_initial_observation_analysis/01_time_series_analysis/network"
)
DEFAULT_OUTPUT_ROOT = (
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/Evaluation/"
    "04_initial_observation_analysis/02_momentary_impact_coefficients"
)


# -----------------------------------------------------------------------------
# Output file inventory inferred from apply_tv_gVAR_network.py
# -----------------------------------------------------------------------------
OUTPUT_SPECS: Dict[str, Dict[str, Any]] = {
    "selected_vars": {"relpath": "data/selected_vars.csv", "type": "csv", "fields": ["variable"]},
    "roles": {"relpath": "data/roles_predictor_criterion.json", "type": "json", "fields": ["predictors", "criteria", "notes"]},
    "labels_map": {"relpath": "data/variables_labels.json", "type": "json", "fields": ["<var_code> -> <label>"]},
    "tv_arrays": {
        "relpath": "method 1/numerical outputs/tvKS_full_arrays.npz",
        "type": "npz",
        "fields": ["estpoints", "Bhat", "partial_corr", "ci_low", "ci_high", "mse_point", "ess_point"],
    },
    "pred_importance_tv": {
        "relpath": "network_metrics/predictor_importance_tv.csv",
        "type": "csv",
        "fields": ["predictor", "out_strength_criteria_mean", "delta_mse_criteria", "nonzero_fraction_mean"],
    },
    "criterion_dependence_tv": {
        "relpath": "network_metrics/criterion_dependence_tv.csv",
        "type": "csv",
        "fields": ["criterion", "incoming_from_predictors_mean", "incoming_from_predictors_sd", "incoming_from_predictors_max"],
    },
    "temporal_lagged_node_centrality": {
        "relpath": "network_metrics/temporal_lagged_node_centrality.csv",
        "type": "csv",
        "fields": [
            "time_index", "t", "node",
            "in_strength_abs", "out_strength_abs", "in_strength_signed", "out_strength_signed",
            "pagerank", "betweenness", "closeness", "eigenvector", "katz",
            "core_number", "participation_coeff", "module_id",
        ],
    },
}


# -----------------------------------------------------------------------------
# Composite weight SHARE constraints (your requirement)
# -----------------------------------------------------------------------------
EDGE_BASE_SHARE = 0.60
EDGE_LOCAL_SHARE = 0.40

PRED_BASE_SHARE = 0.60
PRED_LOCAL_SHARE = 0.40

assert abs((EDGE_BASE_SHARE + EDGE_LOCAL_SHARE) - 1.0) < 1e-9
assert abs((PRED_BASE_SHARE + PRED_LOCAL_SHARE) - 1.0) < 1e-9


# -----------------------------------------------------------------------------
# Base (non-local) EDGE features (from Bhat/pcorr/CI)
# Weights are normalized within BASE and then scaled by EDGE_BASE_SHARE.
# -----------------------------------------------------------------------------
EDGE_BASE_COMPONENTS: Dict[str, Dict[str, Any]] = {
    "beta_abs_recent": {"weight": 0.28, "direction": "pos", "source": "Bhat(t)", "desc": "Recency-weighted mean |beta(t)|"},
    "beta_last_abs":   {"weight": 0.12, "direction": "pos", "source": "Bhat(t)", "desc": "Most recent |beta|"},
    "pcorr_abs_recent": {"weight": 0.12, "direction": "pos", "source": "partial_corr(t)", "desc": "Recency-weighted mean |partial_corr(t)|"},
    "pcorr_last_abs":   {"weight": 0.06, "direction": "pos", "source": "partial_corr(t)", "desc": "Most recent |partial_corr|"},
    "focus_recent": {"weight": 0.10, "direction": "pos", "source": "Bhat(t)", "desc": "Recency-weighted outgoing fraction allocated to criterion"},
    "stability_recent": {"weight": 0.12, "direction": "pos", "source": "Bhat(t)", "desc": "1/(1+SD(|beta(t)|))"},
    "precision_recent": {"weight": 0.08, "direction": "pos", "source": "CI widths", "desc": "1/(1+recency-weighted CI width) if available"},
    "change_last": {"weight": 0.12, "direction": "neg", "source": "Bhat(t)", "desc": "|beta_last - beta_prev| (penalty)"},
}

# -----------------------------------------------------------------------------
# Local EDGE features: derived from temporal_lagged_node_centrality.csv
# The local metrics are node-level; we turn them into edge-relevant features by
# combining predictor-local and criterion-local signals:
#
#   edge_local_x = f(pred_node_metric, crit_node_metric, and/or predictor-to-crit allocation)
#
# All local features are normalized within profile and then combined with weights.
# Weights are normalized within LOCAL and then scaled by EDGE_LOCAL_SHARE.
# -----------------------------------------------------------------------------
EDGE_LOCAL_COMPONENTS: Dict[str, Dict[str, Any]] = {
    # Strength-like local support (momentary, from node centrality)
    "local_pred_out_strength_abs": {"weight": 0.18, "direction": "pos", "source": "node_centrality", "desc": "Recency-weighted predictor out_strength_abs"},
    "local_crit_in_strength_abs":  {"weight": 0.18, "direction": "pos", "source": "node_centrality", "desc": "Recency-weighted criterion in_strength_abs"},

    # Global topology metrics (pagerank etc.) — must be used per your request
    # We use the average of predictor and criterion values as an edge-level local factor.
    "local_pr_mean_pagerank":      {"weight": 0.10, "direction": "pos", "source": "node_centrality", "desc": "Mean of (pred pagerank, crit pagerank)"},
    "local_pr_mean_eigenvector":   {"weight": 0.10, "direction": "pos", "source": "node_centrality", "desc": "Mean of (pred eigenvector, crit eigenvector)"},
    "local_pr_mean_katz":          {"weight": 0.08, "direction": "pos", "source": "node_centrality", "desc": "Mean of (pred katz, crit katz)"},

    # Flow/bridging metrics can indicate structural influence but can be noisy:
    "local_pr_mean_betweenness":   {"weight": 0.06, "direction": "pos", "source": "node_centrality", "desc": "Mean of (pred betweenness, crit betweenness)"},
    "local_pr_mean_closeness":     {"weight": 0.05, "direction": "pos", "source": "node_centrality", "desc": "Mean of (pred closeness, crit closeness)"},

    # Community participation: higher participation can indicate cross-module influence
    "local_pr_mean_participation": {"weight": 0.05, "direction": "pos", "source": "node_centrality", "desc": "Mean of (pred participation_coeff, crit participation_coeff)"},

    # Core number: penalize very low core membership (optional interpretability)
    "local_pr_mean_core_number":   {"weight": 0.10, "direction": "pos", "source": "node_centrality", "desc": "Mean of (pred core_number, crit core_number)"},
}


# -----------------------------------------------------------------------------
# Base PREDICTOR features (non-local): edge aggregation + importance file
# Weights normalized within BASE then scaled by PRED_BASE_SHARE.
# -----------------------------------------------------------------------------
PRED_BASE_COMPONENTS: Dict[str, Dict[str, Any]] = {
    "edge_impact_to_criteria": {"weight": 0.55, "direction": "pos", "source": "edge_composite", "desc": "Criterion-weighted mean edge_impact across criteria"},
    "delta_mse_criteria": {"weight": 0.20, "direction": "pos", "source": "predictor_importance_tv.csv", "desc": "LOO delta-MSE (criteria), clipped >=0"},
    "out_strength_criteria_mean": {"weight": 0.15, "direction": "pos", "source": "predictor_importance_tv.csv", "desc": "Mean outgoing |beta| to criteria"},
    "nonzero_fraction_mean": {"weight": 0.10, "direction": "pos", "source": "predictor_importance_tv.csv", "desc": "Mean nonzero fraction of outgoing edges"},
}

# -----------------------------------------------------------------------------
# Local PREDICTOR features: direct local metrics for predictor node
# Weights normalized within LOCAL then scaled by PRED_LOCAL_SHARE.
# -----------------------------------------------------------------------------
PRED_LOCAL_COMPONENTS: Dict[str, Dict[str, Any]] = {
    "local_out_strength_abs":   {"weight": 0.22, "direction": "pos", "source": "node_centrality", "desc": "Recency-weighted predictor out_strength_abs"},
    "local_out_strength_signed": {"weight": 0.08, "direction": "pos", "source": "node_centrality", "desc": "Recency-weighted predictor out_strength_signed (magnitude helps)"},
    "local_pagerank":           {"weight": 0.12, "direction": "pos", "source": "node_centrality", "desc": "Recency-weighted predictor pagerank"},
    "local_eigenvector":        {"weight": 0.10, "direction": "pos", "source": "node_centrality", "desc": "Recency-weighted predictor eigenvector centrality"},
    "local_katz":               {"weight": 0.08, "direction": "pos", "source": "node_centrality", "desc": "Recency-weighted predictor katz"},
    "local_betweenness":        {"weight": 0.08, "direction": "pos", "source": "node_centrality", "desc": "Recency-weighted predictor betweenness"},
    "local_closeness":          {"weight": 0.06, "direction": "pos", "source": "node_centrality", "desc": "Recency-weighted predictor closeness"},
    "local_participation":      {"weight": 0.10, "direction": "pos", "source": "node_centrality", "desc": "Recency-weighted predictor participation_coeff"},
    "local_core_number":        {"weight": 0.06, "direction": "pos", "source": "node_centrality", "desc": "Recency-weighted predictor core_number"},
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    def _jsonify(x: Any) -> Any:
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.ndarray,)):
            return x.tolist()
        if isinstance(x, dict):
            return {str(k): _jsonify(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_jsonify(v) for v in x]
        return x

    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonify(obj), f, indent=2, ensure_ascii=False)


def safe_read_csv(path: Path) -> pd.DataFrame:
    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            if df.shape[1] > 1 or sep == ",":
                return df
        except Exception:
            pass
    return pd.read_csv(path, engine="python")


def exponential_recency_weights(estpoints: np.ndarray, half_life: float) -> np.ndarray:
    est = np.asarray(estpoints, dtype=float).reshape(-1)
    hl = float(max(1e-6, half_life))
    lam = math.log(2.0) / hl
    w = np.exp(-lam * (1.0 - est))
    w = w / (np.sum(w) + 1e-12)
    return w


def robust_minmax_01(x: np.ndarray, lo_q: float = 0.02, hi_q: float = 0.98) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        return a
    lo = float(np.nanquantile(a, lo_q))
    hi = float(np.nanquantile(a, hi_q))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-12:
        return np.full_like(a, 0.5, dtype=float)
    y = (a - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0)


def normalize_weights(components: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    w = {k: float(max(0.0, float(v.get("weight", 0.0)))) for k, v in components.items()}
    s = float(sum(w.values()))
    if s <= 1e-12:
        n = len(w)
        return {k: 1.0 / max(1, n) for k in w}
    return {k: v / s for k, v in w.items()}


def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    return float(np.sum(w * x) / (np.sum(w) + 1e-12))


def weighted_mean_abs(x: np.ndarray, w: np.ndarray) -> float:
    return weighted_mean(np.abs(np.asarray(x, dtype=float)), w)


def get_profile_dirs(input_root: Path, pattern: str) -> List[Path]:
    if not input_root.exists():
        return []
    dirs = [p for p in input_root.iterdir() if p.is_dir()]
    if pattern:
        dirs = [d for d in dirs if pattern in d.name]
    return sorted(dirs)


def apply_composite(
    df: pd.DataFrame,
    components: Dict[str, Dict[str, Any]],
    composite_col: str,
    share: float,
    norm_suffix: str = "_norm",
    contrib_prefix: str = "contrib__",
) -> pd.DataFrame:
    """
    Computes a composite within df:
      - normalize each component to [0,1] robustly
      - invert if direction == 'neg'
      - contrib = (normalized_weight * share) * normalized_value
      - composite_col = sum(contrib)
    """
    out = df.copy()
    w_norm = normalize_weights(components)

    for comp, spec in components.items():
        direction = str(spec.get("direction", "pos")).lower().strip()

        if comp in out.columns:
            raw = pd.to_numeric(out[comp], errors="coerce").to_numpy(dtype=float)
            nn = robust_minmax_01(raw)
        else:
            nn = np.full(len(out), 0.5, dtype=float)

        if direction == "neg":
            nn = 1.0 - nn

        out[comp + norm_suffix] = nn

    composite = np.zeros(len(out), dtype=float)
    for comp in components.keys():
        c = float(w_norm.get(comp, 0.0)) * float(share) * out[comp + norm_suffix].to_numpy(dtype=float)
        out[contrib_prefix + comp] = c
        composite += c

    out[composite_col] = out.get(composite_col, 0.0) + composite
    return out


def apply_two_stage_composite(
    df: pd.DataFrame,
    base_components: Dict[str, Dict[str, Any]],
    local_components: Dict[str, Dict[str, Any]],
    base_share: float,
    local_share: float,
    out_col: str,
) -> pd.DataFrame:
    """
    out_col = base_share*(weighted base norm features) + local_share*(weighted local norm features)
    with full transparency (per-feature normalized values + contributions).
    """
    out = df.copy()
    out[out_col] = 0.0
    out = apply_composite(out, base_components, composite_col=out_col, share=base_share)
    out = apply_composite(out, local_components, composite_col=out_col, share=local_share)
    out[out_col + "_pct"] = 100.0 * out[out_col].to_numpy(dtype=float)
    return out


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------
@dataclass
class ProfileData:
    profile_id: str
    var_codes: List[str]
    var_labels: Dict[str, str]
    predictors: List[str]
    criteria: List[str]
    roles_notes: str

    estpoints: np.ndarray
    Bhat: np.ndarray
    pcorr: np.ndarray
    ci_low: Optional[np.ndarray]
    ci_high: Optional[np.ndarray]

    pred_importance_tv: Optional[pd.DataFrame]
    temporal_node_centrality: pd.DataFrame  # REQUIRED now
    criterion_dependence_tv: Optional[pd.DataFrame]


# -----------------------------------------------------------------------------
# Required-file assertions
# -----------------------------------------------------------------------------
def assert_exists(path: Path, what: str) -> None:
    assert path.exists(), f"[ASSERTION FAILED] Missing {what}: {path}"


def assert_has_columns(df: pd.DataFrame, cols: List[str], what: str, path: Path) -> None:
    missing = [c for c in cols if c not in df.columns]
    assert not missing, f"[ASSERTION FAILED] {what} missing columns {missing} in file: {path}"


def load_profile(profile_dir: Path) -> ProfileData:
    profile_id = profile_dir.name

    sel_path = profile_dir / OUTPUT_SPECS["selected_vars"]["relpath"]
    assert_exists(sel_path, "selected_vars.csv")
    sel_df = safe_read_csv(sel_path)
    assert "variable" in sel_df.columns, f"[ASSERTION FAILED] {sel_path} has no 'variable' column."
    var_codes = [str(v).strip() for v in sel_df["variable"].tolist() if str(v).strip()]
    assert len(var_codes) >= 2, f"[ASSERTION FAILED] {profile_id}: <2 vars in {sel_path}"

    labels_path = profile_dir / OUTPUT_SPECS["labels_map"]["relpath"]
    var_labels = read_json(labels_path) if labels_path.exists() else {}
    var_labels = {str(k): str(v) for k, v in var_labels.items()}

    roles_notes = ""
    roles_path = profile_dir / OUTPUT_SPECS["roles"]["relpath"]
    predictors, criteria = [], []
    if roles_path.exists():
        roles = read_json(roles_path)
        predictors = [str(x).strip() for x in (roles.get("predictors") or [])]
        criteria = [str(x).strip() for x in (roles.get("criteria") or [])]
        roles_notes = str(roles.get("notes") or "")

    if not predictors:
        predictors = var_codes.copy()
    if not criteria:
        criteria = var_codes.copy()
    predictors = [v for v in predictors if v in var_codes]
    criteria = [v for v in criteria if v in var_codes]
    assert len(predictors) >= 1, f"[ASSERTION FAILED] {profile_id}: no predictors after filtering to selected vars."
    assert len(criteria) >= 1, f"[ASSERTION FAILED] {profile_id}: no criteria after filtering to selected vars."

    tv_path = profile_dir / OUTPUT_SPECS["tv_arrays"]["relpath"]
    assert_exists(tv_path, "tvKS_full_arrays.npz")
    npz = np.load(tv_path, allow_pickle=False)

    for k in ["estpoints", "Bhat", "partial_corr"]:
        assert k in npz, f"[ASSERTION FAILED] {profile_id}: NPZ missing '{k}' in {tv_path}"

    estpoints = np.asarray(npz["estpoints"], dtype=float).reshape(-1)
    Bhat = np.asarray(npz["Bhat"], dtype=float)
    pcorr = np.asarray(npz["partial_corr"], dtype=float)

    assert Bhat.ndim == 3, f"[ASSERTION FAILED] {profile_id}: Bhat expected (m,p,p), got shape {Bhat.shape}"
    assert pcorr.ndim == 3, f"[ASSERTION FAILED] {profile_id}: partial_corr expected (m,p,p), got shape {pcorr.shape}"
    assert Bhat.shape == pcorr.shape, f"[ASSERTION FAILED] {profile_id}: Bhat shape {Bhat.shape} != pcorr shape {pcorr.shape}"
    assert Bhat.shape[0] == estpoints.shape[0], f"[ASSERTION FAILED] {profile_id}: m mismatch Bhat.m={Bhat.shape[0]} estpoints={estpoints.shape[0]}"
    assert Bhat.shape[1] == Bhat.shape[2], f"[ASSERTION FAILED] {profile_id}: Bhat not square in p: {Bhat.shape}"

    ci_low = None
    ci_high = None
    if "ci_low" in npz and "ci_high" in npz:
        cl = np.asarray(npz["ci_low"], dtype=float)
        ch = np.asarray(npz["ci_high"], dtype=float)
        if np.isfinite(cl).any() and np.isfinite(ch).any():
            assert cl.shape == Bhat.shape and ch.shape == Bhat.shape, (
                f"[ASSERTION FAILED] {profile_id}: CI shapes must match Bhat. "
                f"ci_low={cl.shape}, ci_high={ch.shape}, Bhat={Bhat.shape}"
            )
            ci_low, ci_high = cl, ch

    # predictor importance is optional
    pi_path = profile_dir / OUTPUT_SPECS["pred_importance_tv"]["relpath"]
    pred_importance_tv = safe_read_csv(pi_path) if pi_path.exists() else None
    if pred_importance_tv is not None and not pred_importance_tv.empty:
        pred_importance_tv["predictor"] = pred_importance_tv["predictor"].astype(str)

    # LOCAL METRICS FILE IS REQUIRED (your requirement)
    nc_path = profile_dir / OUTPUT_SPECS["temporal_lagged_node_centrality"]["relpath"]
    assert_exists(nc_path, "temporal_lagged_node_centrality.csv (REQUIRED for local metrics)")
    temporal_node_centrality = safe_read_csv(nc_path)

    # required columns for the local-metric pipeline
    required_nc = [
        "time_index", "node",
        "in_strength_abs", "out_strength_abs",
        "pagerank", "betweenness", "closeness", "eigenvector", "katz",
        "core_number", "participation_coeff",
    ]
    assert_has_columns(temporal_node_centrality, required_nc, "temporal_lagged_node_centrality.csv", nc_path)

    temporal_node_centrality["node"] = temporal_node_centrality["node"].astype(str)
    temporal_node_centrality["time_index"] = pd.to_numeric(temporal_node_centrality["time_index"], errors="coerce").fillna(-1).astype(int)

    # criterion dependence optional
    cd_path = profile_dir / OUTPUT_SPECS["criterion_dependence_tv"]["relpath"]
    criterion_dependence_tv = safe_read_csv(cd_path) if cd_path.exists() else None

    return ProfileData(
        profile_id=profile_id,
        var_codes=var_codes,
        var_labels=var_labels,
        predictors=predictors,
        criteria=criteria,
        roles_notes=roles_notes,
        estpoints=estpoints,
        Bhat=Bhat,
        pcorr=pcorr,
        ci_low=ci_low,
        ci_high=ci_high,
        pred_importance_tv=pred_importance_tv,
        temporal_node_centrality=temporal_node_centrality,
        criterion_dependence_tv=criterion_dependence_tv,
    )


# -----------------------------------------------------------------------------
# Local metrics extraction (recency-weighted)
# -----------------------------------------------------------------------------
def compute_recency_weighted_node_metrics(data: ProfileData, half_life: float) -> pd.DataFrame:
    """
    Returns node-level table with recency-weighted values per node for all local metrics.
    Required because local metrics are mandatory in this version.
    """
    nc = data.temporal_node_centrality.copy()

    m = len(data.estpoints)
    w = exponential_recency_weights(data.estpoints, half_life=half_life)
    w_map = {int(i): float(w[i]) for i in range(m)}
    nc["w"] = nc["time_index"].map(w_map).fillna(0.0).astype(float)

    # only keep nodes in selected var set
    nc = nc[nc["node"].isin(set(data.var_codes))].copy()
    assert not nc.empty, f"[ASSERTION FAILED] {data.profile_id}: no matching nodes in temporal_lagged_node_centrality after filtering to selected vars."

    metrics = [
        "in_strength_abs", "out_strength_abs", "in_strength_signed", "out_strength_signed",
        "pagerank", "betweenness", "closeness", "eigenvector", "katz",
        "core_number", "participation_coeff",
    ]
    # in_strength_signed/out_strength_signed may not exist in required list, but may be present; handle gracefully
    for c in metrics:
        if c in nc.columns:
            nc[c] = pd.to_numeric(nc[c], errors="coerce")

    rows = []
    for node, g in nc.groupby("node", sort=False):
        ww = g["w"].to_numpy(dtype=float)
        if np.sum(ww) <= 1e-12:
            continue

        row = {"node": node}
        for c in metrics:
            if c not in g.columns:
                row[c] = np.nan
                continue
            xx = np.nan_to_num(g[c].to_numpy(dtype=float), nan=0.0)
            row[c] = float(np.sum(ww * xx) / (np.sum(ww) + 1e-12))
        rows.append(row)

    out = pd.DataFrame(rows)
    assert not out.empty, f"[ASSERTION FAILED] {data.profile_id}: could not compute any recency-weighted node metrics."
    return out


# -----------------------------------------------------------------------------
# EDGE features and composite (base + local)
# -----------------------------------------------------------------------------
def compute_edge_features(data: ProfileData, half_life: float, node_metrics: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    """
    Builds edge feature table including BOTH base metrics and local-metric-derived features.
    """
    vars_ = data.var_codes
    idx = {v: i for i, v in enumerate(vars_)}
    pred = data.predictors
    crit = data.criteria

    m = int(data.Bhat.shape[0])
    assert m == len(data.estpoints), f"[ASSERTION FAILED] {data.profile_id}: Bhat m != len(estpoints)."

    w = exponential_recency_weights(data.estpoints, half_life=half_life)

    # node metric maps for quick lookup
    nm = node_metrics.set_index("node")
    for v in set(pred + crit):
        assert v in nm.index, f"[ASSERTION FAILED] {data.profile_id}: node '{v}' missing from recency-weighted node_metrics (from temporal_lagged_node_centrality.csv)."

    crit_idx = [idx[c] for c in crit]

    rows: List[Dict[str, Any]] = []
    for pcode in pred:
        j = idx[pcode]
        denom_t = np.sum(np.abs(data.Bhat[:, crit_idx, j]), axis=1) + eps

        # predictor local signals
        pred_out_strength_abs = float(nm.loc[pcode, "out_strength_abs"])
        pred_pagerank = float(nm.loc[pcode, "pagerank"])
        pred_eigenvector = float(nm.loc[pcode, "eigenvector"])
        pred_katz = float(nm.loc[pcode, "katz"])
        pred_betweenness = float(nm.loc[pcode, "betweenness"])
        pred_closeness = float(nm.loc[pcode, "closeness"])
        pred_participation = float(nm.loc[pcode, "participation_coeff"])
        pred_core_number = float(nm.loc[pcode, "core_number"])

        for ccode in crit:
            i = idx[ccode]
            if i == j:
                continue

            beta_t = data.Bhat[:, i, j]
            pc_t = data.pcorr[:, i, j]

            # --- base features ---
            beta_abs_recent = weighted_mean_abs(beta_t, w)
            beta_last_abs = float(abs(beta_t[-1]))
            beta_signed_recent = weighted_mean(beta_t, w)

            pcorr_abs_recent = weighted_mean_abs(pc_t, w)
            pcorr_last_abs = float(abs(pc_t[-1]))

            focus_recent = float(np.sum(w * (np.abs(beta_t) / denom_t)))
            stability_recent = float(1.0 / (1.0 + np.std(np.abs(beta_t))))
            change_last = float(abs(beta_t[-1] - beta_t[-2])) if m >= 2 else 0.0

            precision_recent = 0.0
            if data.ci_low is not None and data.ci_high is not None:
                wdt = (data.ci_high[:, i, j] - data.ci_low[:, i, j]).astype(float)
                if np.isfinite(wdt).any():
                    ci_width_recent = float(np.sum(w * wdt))
                    precision_recent = float(1.0 / (1.0 + ci_width_recent))
                else:
                    ci_width_recent = np.nan
                    precision_recent = 0.0
            else:
                ci_width_recent = np.nan
                precision_recent = 0.0

            # --- local features (edge-level) ---
            crit_in_strength_abs = float(nm.loc[ccode, "in_strength_abs"])
            crit_pagerank = float(nm.loc[ccode, "pagerank"])
            crit_eigenvector = float(nm.loc[ccode, "eigenvector"])
            crit_katz = float(nm.loc[ccode, "katz"])
            crit_betweenness = float(nm.loc[ccode, "betweenness"])
            crit_closeness = float(nm.loc[ccode, "closeness"])
            crit_participation = float(nm.loc[ccode, "participation_coeff"])
            crit_core_number = float(nm.loc[ccode, "core_number"])

            # edge-local: combine predictor + criterion topology by mean
            local_pr_mean_pagerank = 0.5 * (pred_pagerank + crit_pagerank)
            local_pr_mean_eigenvector = 0.5 * (pred_eigenvector + crit_eigenvector)
            local_pr_mean_katz = 0.5 * (pred_katz + crit_katz)
            local_pr_mean_betweenness = 0.5 * (pred_betweenness + crit_betweenness)
            local_pr_mean_closeness = 0.5 * (pred_closeness + crit_closeness)
            local_pr_mean_participation = 0.5 * (pred_participation + crit_participation)
            local_pr_mean_core_number = 0.5 * (pred_core_number + crit_core_number)

            rows.append(
                {
                    "predictor": pcode,
                    "criterion": ccode,
                    "predictor_label": data.var_labels.get(pcode, pcode),
                    "criterion_label": data.var_labels.get(ccode, ccode),

                    # base metrics
                    "beta_abs_recent": beta_abs_recent,
                    "beta_last_abs": beta_last_abs,
                    "beta_signed_recent": beta_signed_recent,
                    "pcorr_abs_recent": pcorr_abs_recent,
                    "pcorr_last_abs": pcorr_last_abs,
                    "focus_recent": focus_recent,
                    "stability_recent": stability_recent,
                    "precision_recent": precision_recent,
                    "ci_width_recent": ci_width_recent,
                    "change_last": change_last,

                    # local metrics
                    "local_pred_out_strength_abs": pred_out_strength_abs,
                    "local_crit_in_strength_abs": crit_in_strength_abs,
                    "local_pr_mean_pagerank": local_pr_mean_pagerank,
                    "local_pr_mean_eigenvector": local_pr_mean_eigenvector,
                    "local_pr_mean_katz": local_pr_mean_katz,
                    "local_pr_mean_betweenness": local_pr_mean_betweenness,
                    "local_pr_mean_closeness": local_pr_mean_closeness,
                    "local_pr_mean_participation": local_pr_mean_participation,
                    "local_pr_mean_core_number": local_pr_mean_core_number,
                }
            )

    out = pd.DataFrame(rows)
    assert not out.empty, f"[ASSERTION FAILED] {data.profile_id}: edge feature table is empty."
    return out


def compute_edge_composite(edge_feat: pd.DataFrame) -> pd.DataFrame:
    df = apply_two_stage_composite(
        edge_feat,
        base_components=EDGE_BASE_COMPONENTS,
        local_components=EDGE_LOCAL_COMPONENTS,
        base_share=EDGE_BASE_SHARE,
        local_share=EDGE_LOCAL_SHARE,
        out_col="edge_impact",
    )
    # signed variant (based on sign of beta_signed_recent)
    sign = np.sign(pd.to_numeric(df["beta_signed_recent"], errors="coerce").fillna(0.0).to_numpy(dtype=float))
    df["edge_impact_signed"] = df["edge_impact"].to_numpy(dtype=float) * sign
    df["edge_impact_signed_pct"] = 100.0 * df["edge_impact_signed"].to_numpy(dtype=float)

    df = df.sort_values("edge_impact", ascending=False).reset_index(drop=True)
    df["edge_rank"] = np.arange(1, len(df) + 1, dtype=int)
    return df


# -----------------------------------------------------------------------------
# Criterion weights for aggregating predictor edge impacts
# -----------------------------------------------------------------------------
def criterion_weights_from_node_metrics(data: ProfileData, node_metrics: pd.DataFrame) -> Dict[str, float]:
    """
    Use recency-weighted criterion in_strength_abs as criterion weights (normalized).
    This is consistent with "local metrics are used"; it also makes aggregation more meaningful.
    """
    nm = node_metrics.set_index("node")
    vals = {c: float(nm.loc[c, "in_strength_abs"]) for c in data.criteria if c in nm.index}
    # fallback uniform if degenerate
    s = float(sum(max(0.0, v) for v in vals.values()))
    if s <= 1e-12:
        return {c: 1.0 / max(1, len(data.criteria)) for c in data.criteria}
    return {c: max(0.0, vals.get(c, 0.0)) / s for c in data.criteria}


# -----------------------------------------------------------------------------
# Predictor-level features and composite (base + local)
# -----------------------------------------------------------------------------
def compute_predictor_features(
    data: ProfileData,
    edge_comp: pd.DataFrame,
    node_metrics: pd.DataFrame,
    crit_w: Dict[str, float],
) -> pd.DataFrame:
    nm = node_metrics.set_index("node")

    # predictor importance file is optional; but if missing, those base components become neutral after normalization
    pi = data.pred_importance_tv
    if pi is not None and not pi.empty and "predictor" in pi.columns:
        pi = pi.copy()
        pi["predictor"] = pi["predictor"].astype(str)
    else:
        pi = None

    rows: List[Dict[str, Any]] = []
    for p in data.predictors:
        # base component: criterion-weighted mean edge_impact
        ed = edge_comp[edge_comp["predictor"] == p]
        edge_impact_to_criteria = 0.0
        if not ed.empty:
            for _, r in ed.iterrows():
                c = str(r["criterion"])
                edge_impact_to_criteria += float(crit_w.get(c, 0.0)) * float(r["edge_impact"])

        # base components from predictor_importance_tv.csv
        delta_mse_criteria = np.nan
        out_strength_criteria_mean = np.nan
        nonzero_fraction_mean = np.nan
        if pi is not None:
            hit = pi[pi["predictor"] == p]
            if not hit.empty:
                if "delta_mse_criteria" in hit.columns:
                    val = float(pd.to_numeric(hit["delta_mse_criteria"].iloc[0], errors="coerce"))
                    delta_mse_criteria = max(0.0, val) if np.isfinite(val) else np.nan
                if "out_strength_criteria_mean" in hit.columns:
                    out_strength_criteria_mean = float(pd.to_numeric(hit["out_strength_criteria_mean"].iloc[0], errors="coerce"))
                if "nonzero_fraction_mean" in hit.columns:
                    nonzero_fraction_mean = float(pd.to_numeric(hit["nonzero_fraction_mean"].iloc[0], errors="coerce"))

        # local predictor components directly from node metrics
        assert p in nm.index, f"[ASSERTION FAILED] {data.profile_id}: predictor '{p}' missing from node_metrics."
        local_out_strength_abs = float(nm.loc[p, "out_strength_abs"])
        local_out_strength_signed = float(nm.loc[p, "out_strength_signed"]) if "out_strength_signed" in nm.columns else np.nan
        local_pagerank = float(nm.loc[p, "pagerank"])
        local_eigenvector = float(nm.loc[p, "eigenvector"])
        local_katz = float(nm.loc[p, "katz"])
        local_betweenness = float(nm.loc[p, "betweenness"])
        local_closeness = float(nm.loc[p, "closeness"])
        local_participation = float(nm.loc[p, "participation_coeff"])
        local_core_number = float(nm.loc[p, "core_number"])

        rows.append(
            {
                "predictor": p,
                "predictor_label": data.var_labels.get(p, p),

                # base
                "edge_impact_to_criteria": edge_impact_to_criteria,
                "delta_mse_criteria": delta_mse_criteria,
                "out_strength_criteria_mean": out_strength_criteria_mean,
                "nonzero_fraction_mean": nonzero_fraction_mean,

                # local
                "local_out_strength_abs": local_out_strength_abs,
                "local_out_strength_signed": local_out_strength_signed,
                "local_pagerank": local_pagerank,
                "local_eigenvector": local_eigenvector,
                "local_katz": local_katz,
                "local_betweenness": local_betweenness,
                "local_closeness": local_closeness,
                "local_participation": local_participation,
                "local_core_number": local_core_number,
            }
        )

    out = pd.DataFrame(rows)
    assert not out.empty, f"[ASSERTION FAILED] {data.profile_id}: predictor feature table is empty."
    return out


def compute_predictor_composite(pred_feat: pd.DataFrame) -> pd.DataFrame:
    df = apply_two_stage_composite(
        pred_feat,
        base_components=PRED_BASE_COMPONENTS,
        local_components=PRED_LOCAL_COMPONENTS,
        base_share=PRED_BASE_SHARE,
        local_share=PRED_LOCAL_SHARE,
        out_col="predictor_impact",
    )
    df = df.sort_values("predictor_impact", ascending=False).reset_index(drop=True)
    df["predictor_rank"] = np.arange(1, len(df) + 1, dtype=int)
    return df


# -----------------------------------------------------------------------------
# Saving outputs
# -----------------------------------------------------------------------------
def save_impact_matrices(edge_comp: pd.DataFrame, criteria: List[str], predictors: List[str], out_dir: Path) -> None:
    mat = (
        edge_comp.pivot_table(index="criterion", columns="predictor", values="edge_impact", aggfunc="mean")
        .reindex(index=criteria, columns=predictors)
        .fillna(0.0)
    )
    mat.to_csv(out_dir / "impact_matrix.csv")

    mat_s = (
        edge_comp.pivot_table(index="criterion", columns="predictor", values="edge_impact_signed", aggfunc="mean")
        .reindex(index=criteria, columns=predictors)
        .fillna(0.0)
    )
    mat_s.to_csv(out_dir / "impact_matrix_signed.csv")


def write_config_used(out_dir: Path) -> None:
    cfg = {
        "EDGE_BASE_SHARE": EDGE_BASE_SHARE,
        "EDGE_LOCAL_SHARE": EDGE_LOCAL_SHARE,
        "PRED_BASE_SHARE": PRED_BASE_SHARE,
        "PRED_LOCAL_SHARE": PRED_LOCAL_SHARE,
        "EDGE_BASE_COMPONENTS": EDGE_BASE_COMPONENTS,
        "EDGE_BASE_WEIGHTS_NORMALIZED": normalize_weights(EDGE_BASE_COMPONENTS),
        "EDGE_LOCAL_COMPONENTS": EDGE_LOCAL_COMPONENTS,
        "EDGE_LOCAL_WEIGHTS_NORMALIZED": normalize_weights(EDGE_LOCAL_COMPONENTS),
        "PRED_BASE_COMPONENTS": PRED_BASE_COMPONENTS,
        "PRED_BASE_WEIGHTS_NORMALIZED": normalize_weights(PRED_BASE_COMPONENTS),
        "PRED_LOCAL_COMPONENTS": PRED_LOCAL_COMPONENTS,
        "PRED_LOCAL_WEIGHTS_NORMALIZED": normalize_weights(PRED_LOCAL_COMPONENTS),
    }
    write_json(out_dir / "config_used.json", cfg)


def build_payload(
    data: ProfileData,
    half_life: float,
    crit_w: Dict[str, float],
    out_dir: Path,
    edge_comp: pd.DataFrame,
    pred_comp: pd.DataFrame,
    top_k_edges: int,
) -> Dict[str, Any]:
    overall_pred = {
        r["predictor"]: {
            "label": r.get("predictor_label", r["predictor"]),
            "predictor_impact": float(r["predictor_impact"]),
            "predictor_impact_pct": float(r["predictor_impact_pct"]),
            "rank": int(r["predictor_rank"]),
        }
        for r in pred_comp.to_dict(orient="records")
    }
    top_edges = (
        edge_comp.sort_values("edge_impact", ascending=False)
        .head(int(top_k_edges))
        .to_dict(orient="records")
    )
    return {
        "meta": {
            "profile_id": data.profile_id,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "half_life": float(half_life),
            "n_vars": len(data.var_codes),
            "n_predictors": len(data.predictors),
            "n_criteria": len(data.criteria),
            "roles_notes": data.roles_notes,
            "inputs_used": OUTPUT_SPECS,
            "files_written": {
                "edge_composite_csv": str((out_dir / "edge_composite.csv").resolve()),
                "impact_matrix_csv": str((out_dir / "impact_matrix.csv").resolve()),
                "impact_matrix_signed_csv": str((out_dir / "impact_matrix_signed.csv").resolve()),
                "predictor_composite_csv": str((out_dir / "predictor_composite.csv").resolve()),
                "overall_predictor_impact_json": str((out_dir / "overall_predictor_impact.json").resolve()),
                "config_used_json": str((out_dir / "config_used.json").resolve()),
                "momentary_impact_json": str((out_dir / "momentary_impact.json").resolve()),
            },
        },
        "criterion_weights_used": crit_w,
        "composites": {"predictor_impact_map": overall_pred, "top_edges": top_edges},
    }


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def run_one_profile(profile_dir: Path, out_root: Path, half_life: float, top_k_edges: int) -> None:
    data = load_profile(profile_dir)

    log(f"--- PROFILE {data.profile_id} ---")
    log(f"Vars={len(data.var_codes)}  Predictors={len(data.predictors)}  Criteria={len(data.criteria)}")
    log("Local metrics REQUIRED: temporal_lagged_node_centrality.csv (will assert if missing).")

    # compute recency-weighted node metrics (local metrics)
    node_metrics = compute_recency_weighted_node_metrics(data, half_life=half_life)
    log(f"Computed recency-weighted node metrics for {node_metrics.shape[0]} nodes.")

    # criterion weights derived from local metrics
    crit_w = criterion_weights_from_node_metrics(data, node_metrics)

    # edge-level
    edge_feat = compute_edge_features(data, half_life=half_life, node_metrics=node_metrics)
    edge_comp = compute_edge_composite(edge_feat)

    # predictor-level
    pred_feat = compute_predictor_features(data, edge_comp=edge_comp, node_metrics=node_metrics, crit_w=crit_w)
    pred_comp = compute_predictor_composite(pred_feat)

    out_dir = ensure_dir(out_root / data.profile_id)

    # write outputs
    edge_comp.to_csv(out_dir / "edge_composite.csv", index=False)
    pred_comp.to_csv(out_dir / "predictor_composite.csv", index=False)
    save_impact_matrices(edge_comp, data.criteria, data.predictors, out_dir)

    overall_pred = {
        r["predictor"]: {
            "label": r.get("predictor_label", r["predictor"]),
            "predictor_impact": float(r["predictor_impact"]),
            "predictor_impact_pct": float(r["predictor_impact_pct"]),
            "rank": int(r["predictor_rank"]),
        }
        for r in pred_comp.to_dict(orient="records")
    }
    write_json(out_dir / "overall_predictor_impact.json", overall_pred)

    write_config_used(out_dir)

    payload = build_payload(
        data=data,
        half_life=half_life,
        crit_w=crit_w,
        out_dir=out_dir,
        edge_comp=edge_comp,
        pred_comp=pred_comp,
        top_k_edges=top_k_edges,
    )
    write_json(out_dir / "momentary_impact.json", payload)

    log(f"[DONE] {data.profile_id} -> {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute momentary impact composites (edge + predictor) from tv-gVAR outputs.")
    p.add_argument("--input-root", type=str, default=DEFAULT_INPUT_ROOT)
    p.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--pattern", type=str, default="pseudoprofile_FTC_", help="Substring filter for profile directory names.")
    p.add_argument("--half-life", type=float, default=0.20, help="Recency half-life as fraction of [0,1] estpoint span.")
    p.add_argument("--max-profiles", type=int, default=0, help="0 = all; otherwise process first N profiles.")
    p.add_argument("--top-k-edges", type=int, default=200, help="How many top edges to include in momentary_impact.json.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root).expanduser()
    out_root = Path(args.output_root).expanduser()
    ensure_dir(out_root)

    prof_dirs = get_profile_dirs(input_root, pattern=str(args.pattern or ""))
    if args.max_profiles and int(args.max_profiles) > 0:
        prof_dirs = prof_dirs[: int(args.max_profiles)]

    if not prof_dirs:
        log(f"No profile dirs found in {input_root} with pattern={args.pattern!r}")
        return 2

    log("========== MOMENTARY IMPACT (WITH LOCAL METRICS) START ==========")
    log(f"input_root:  {input_root}")
    log(f"output_root: {out_root}")
    log(f"pattern:     {args.pattern!r}")
    log(f"profiles:    {len(prof_dirs)}")
    log(f"half_life:   {float(args.half_life):.4f}")
    log(f"EDGE shares: base={EDGE_BASE_SHARE:.2f}, local={EDGE_LOCAL_SHARE:.2f}")
    log(f"PRED shares: base={PRED_BASE_SHARE:.2f}, local={PRED_LOCAL_SHARE:.2f}")
    log("")

    n_ok, n_fail = 0, 0
    for d in prof_dirs:
        try:
            run_one_profile(
                profile_dir=d,
                out_root=out_root,
                half_life=float(args.half_life),
                top_k_edges=int(args.top_k_edges),
            )
            n_ok += 1
        except AssertionError as e:
            n_fail += 1
            log(str(e))
        except Exception as e:
            n_fail += 1
            log(f"[ERROR] {d.name}: {repr(e)}")

    log("")
    log("========== MOMENTARY IMPACT (WITH LOCAL METRICS) COMPLETE ==========")
    log(f"Success: {n_ok}  Failed: {n_fail}")
    log(f"Output root: {out_root}")
    return 0 if n_fail == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
