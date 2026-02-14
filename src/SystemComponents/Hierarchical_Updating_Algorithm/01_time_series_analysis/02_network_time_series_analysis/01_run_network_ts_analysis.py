#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_tv_gVAR_network.py

Batch tv-gVAR (kernel-smoothed time-varying VAR(1) with L1), stationary gVAR (L1 VAR(1)),
and correlation/partial-correlation baselines, driven by readiness_report.json to select
ONLY variables suitable for multivariate (tv-)gVAR + correlation analyses.

NO VISUALIZATION. Numerical outputs only.

Key additions vs your baseline apply_tv_gVAR.py
1) Variable selection uses readiness_report.json + strict numeric suitability filtering.
2) Automated temporal network regime/cluster detection (two complementary methods):
   - error-ratio segmentation (stationary MSE / tv MSE)
   - change-point heuristic on network distance (Frobenius deltas of B(t))
3) Individual relationship shift detection (edge-level change ranking, change time localization).
4) Comprehensive network analysis (temporal lagged network from B(t) AND contemporaneous
   network from partial correlations):
   - Local centralities (many): in/out degree, in/out strength, betweenness, closeness,
     harmonic, eigenvector, PageRank, Katz, HITS hubs/authorities, clustering, k-core,
     constraint/effective size (if available), participation coefficient (modules).
   - Global metrics: density, reciprocity (directed), modularity, communities count,
     transitivity, assortativity, global efficiency, path-length summaries, component sizes,
     spectral radius, stability proxies, etc.
5) Predictor -> criterion importance:
   - coefficient-based (out-strength to criteria, nonzero fraction, signed totals)
   - predictive importance via leave-one-predictor-out one-step MSE delta (tv + stationary)
6) Multicollinearity diagnostics:
   - VIF (ridge-stabilized), condition number, eigen spectrum, high-correlation pairs
   - computed for: all vars, predictors-only, criteria-only (if definable)

Outputs (per profile) under:
  /Users/stijnvanseveren/PythonProjects/MASTERPROEF/Evaluation/04_initial_observation_analysis/01_time_series_analysis/network/<profile_id>/

Directory layout (per profile)
  data/
    raw_wide_copy.csv
    selected_vars.json
    selected_vars.csv
    X_obs_selected.csv
    X_imputed_raw.csv
    X_imputed_zscored.csv
    t_norm.csv
    variables_labels.json
    roles_predictor_criterion.json
    kalman_model.json
    kalman_diagnostics.json

  method 1/  (tvKS time-varying gVAR)
    numerical outputs/
      tv_hyperparam_cv.json
      tvKS_full_arrays.npz
      time_index_map.json
      Bhat_time_XX_t_YYY.csv
      partialcorr_time_XX_t_YYY.csv
      (optional) CI_low_time_XX_t_YYY.csv / CI_high_time_XX_t_YYY.csv
      tv_temporal_long(_with_CI).csv
      tv_one_step_Y.csv / tv_one_step_Yhat.csv
      tv_cluster_detection.json
      tv_network_change_points.json
      edge_shifts_top.csv
      metrics.json

  method 2/  (stationary gVAR)
    numerical outputs/
      B_stationary.csv
      residual_cov.csv
      partial_corr.csv
      stationary_full_arrays.npz
      stationary_one_step_Y.csv / stationary_one_step_Yhat.csv
      metrics.json

  method 3/  (correlation baseline)
    numerical outputs/
      correlation_pearson.csv
      correlation_spearman.csv
      partial_corr_ledoitwolf.csv
      correlation_full_arrays.npz
      metrics.json

  network_metrics/
    temporal_lagged_node_centrality.csv
    temporal_contemp_node_centrality.csv
    temporal_lagged_global_metrics.csv
    temporal_contemp_global_metrics.csv
    stationary_lagged_node_centrality.csv
    stationary_contemp_node_centrality.csv
    stationary_lagged_global_metrics.json
    stationary_contemp_global_metrics.json
    predictor_importance_tv.csv
    predictor_importance_stationary.csv
    criterion_dependence_tv.csv
    multicollinearity_all.json
    multicollinearity_predictors.json
    multicollinearity_criteria.json

  comparison_summary.json

Run:
  python apply_tv_gVAR_network.py \
    --input-root "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/Evaluation/01_pseudoprofile(s)/time_series_data/pseudodata" \
    --readiness-root "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/Evaluation/04_initial_observation_analysis/00_readiness_check" \
    --output-root "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/Evaluation/04_initial_observation_analysis/01_time_series_analysis/network" \
    --prefer-tier tier1 \
    --pattern "pseudoprofile_FTC_" \
    --boot 80 --block-len 20 --jobs 1

Notes:
- tvKS uses Gaussian kernel weights over normalized time t_norm.
- Centrality uses ABS weights by default for algorithms requiring nonnegative weights.
- For shortest-path based measures, distance = 1/(abs_weight + eps).
- Participation coefficient computed from communities on UNDIRECTED abs-weight graph.

Author: generated; adapt thresholds/roles parsing to your metadata conventions.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLassoCV, LedoitWolf
from sklearn.linear_model import Lasso
from sklearn.model_selection import TimeSeriesSplit

# -----------------------------
# Optional deps (network metrics, parallel bootstrap)
# -----------------------------
HAVE_NETWORKX = False
try:
    import networkx as nx  # type: ignore
    HAVE_NETWORKX = True
except Exception:
    HAVE_NETWORKX = False

HAVE_JOBLIB = False
try:
    from joblib import Parallel, delayed  # type: ignore
    HAVE_JOBLIB = True
except Exception:
    HAVE_JOBLIB = False


# ============================================================
# Defaults (match your layout)
# ============================================================
def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        has_eval = (candidate / "evaluation").exists() or (candidate / "Evaluation").exists()
        if has_eval and (candidate / "README.md").exists():
            return candidate
    raise RuntimeError("Could not locate repository root from 01_run_network_ts_analysis.py")


REPO_ROOT = _find_repo_root()

DEFAULT_INPUT_ROOT = str(
    REPO_ROOT / "evaluation/01_pseudoprofile(s)/time_series_data/pseudodata"
)
DEFAULT_READINESS_ROOT = str(
    REPO_ROOT / "evaluation/04_initial_observation_analysis/00_readiness_check"
)
DEFAULT_OUTPUT_ROOT = str(
    REPO_ROOT / "evaluation/04_initial_observation_analysis/01_time_series_analysis/network"
)
DEFAULT_DATA_FILENAME = "pseudodata_wide.csv"
DEFAULT_META_FILENAME = "variables_metadata.csv"
DEFAULT_READINESS_FILENAME = "readiness_report.json"


# ============================================================
# Config
# ============================================================
@dataclass
class Config:
    # variable selection
    prefer_tier: str = "overall_ready"  # "tier1" | "overall_ready" | "all_non_hard"
    min_nonmissing_for_network: int = 20
    max_missing_frac_for_network: float = 0.65
    min_unique_for_network: int = 3
    min_std_for_network: float = 1e-8
    allow_binary: bool = True
    allowed_types: Tuple[str, ...] = ("continuous", "count", "proportion", "binary", "unknown")

    # tv resolution
    m: int = 20
    cv_m: int = 12
    cv_splits: int = 5
    standardize_in_cv: bool = True
    ridge_precision: float = 1e-3

    # alpha grid (lasso)
    alpha_min_exp: float = -2.0
    alpha_max_exp: float = 0.2
    alpha_num: int = 18
    alpha_floor: float = 0.0

    # bandwidth grid
    bw_grid: Tuple[float, ...] = (0.03, 0.05, 0.08, 0.11, 0.15, 0.20, 0.28, 0.36)

    # bootstrap
    boot: int = 80
    block_len: int = 20
    jobs: int = 1
    ci_low: float = 0.05
    ci_high: float = 0.95

    # cluster / regimes
    ratio_threshold: float = 3.0
    change_point_z: float = 2.5  # threshold on z-scored ||B(t)-B(t-1)||_F

    # edge shift detection
    top_k_edge_shifts: int = 200

    # kalman
    kalman_refine: int = 3
    kalman_ridge_alpha: float = 1.0
    kalman_meas_noise_scale: float = 1e-2
    kalman_stabilize_rho: float = 0.98
    kalman_ridge: float = 1e-8

    # network construction
    edge_threshold_abs: float = 1e-10  # keep essentially all nonzero edges
    exclude_self_loops: bool = True

    # multicollinearity
    vif_ridge: float = 1e-3
    high_corr_thresholds: Tuple[float, ...] = (0.80, 0.90)


CFG = Config()

TEMPORAL_NODE_CENTRALITY_COLUMNS = [
    "time_index",
    "t",
    "node",
    "in_degree",
    "out_degree",
    "in_strength_abs",
    "out_strength_abs",
    "in_strength_signed",
    "out_strength_signed",
    "betweenness",
    "closeness",
    "harmonic",
    "eigenvector",
    "pagerank",
    "katz",
    "hits_hub",
    "hits_authority",
    "clustering",
    "core_number",
    "constraint",
    "effective_size",
    "participation_coeff",
    "module_id",
]

PREDICTOR_IMPORTANCE_COLUMNS = [
    "predictor",
    "out_strength_criteria_mean",
    "nonzero_fraction_mean",
    "out_strength_all_mean",
    "nonzero_fraction_all",
    "delta_mse_all",
    "delta_mse_criteria",
]

CRITERION_DEPENDENCE_COLUMNS = [
    "criterion",
    "incoming_from_predictors_mean",
    "incoming_from_predictors_sd",
    "incoming_from_predictors_max",
]


# ============================================================
# Logging
# ============================================================
def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


# ============================================================
# General utilities
# ============================================================
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    return obj


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv_robust(path: Path) -> pd.DataFrame:
    last_err = None
    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            if df.shape[1] > 1:
                return df
        except Exception as e:
            last_err = e
            continue
    try:
        return pd.read_csv(path, engine="python")
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV {path}. Last errors: {repr(last_err)} ; {repr(e)}")


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


# ============================================================
# Readiness discovery + variable selection
# ============================================================
def discover_readiness_reports(readiness_root: Path, readiness_filename: str) -> List[Path]:
    return sorted(Path(readiness_root).expanduser().rglob(readiness_filename))


def _infer_var_type_from_readiness(readiness: Dict[str, Any], var: str) -> str:
    try:
        return str(readiness.get("variables", {}).get(var, {}).get("type_inferred", "unknown")).strip().lower()
    except Exception:
        return "unknown"


def _hard_dropped(readiness: Dict[str, Any], var: str) -> bool:
    try:
        reasons = readiness.get("variables", {}).get(var, {}).get("drop_hard_reasons") or []
        return bool(len(reasons) > 0)
    except Exception:
        return False


def _readiness_candidate_vars(readiness: Dict[str, Any], prefer_tier: str) -> List[str]:
    if prefer_tier == "overall_ready":
        return list(readiness.get("overall", {}).get("ready_variables", []) or [])
    if prefer_tier == "tier1":
        return list(readiness.get("tiers", {}).get("tier1", {}).get("ready_variables", []) or [])
    # all_non_hard:
    vars_meta = readiness.get("variables", {}) or {}
    out = []
    for v in vars_meta.keys():
        if not _hard_dropped(readiness, v):
            out.append(v)
    return out


def build_execution_plan_from_readiness(readiness: Dict[str, Any], execution_policy: str) -> Dict[str, Any]:
    overall = readiness.get("overall", {}) or {}
    tier = str(overall.get("recommended_tier", "Tier0_DescriptivesOnly"))
    variant_raw = overall.get("tier3_variant", None)
    variant = str(variant_raw) if variant_raw is not None else None
    tv_full_confidence = bool(overall.get("tv_full_confidence", False))
    why_not_tv = list(overall.get("why_not_time_varying", []) or [])
    plan_from_readiness = overall.get("analysis_execution_plan", {}) or {}

    if execution_policy == "all_methods":
        return {
            "execution_policy": execution_policy,
            "recommended_tier": tier,
            "recommended_variant": variant,
            "analysis_set": "all_methods_for_diagnostics",
            "run_tv_gvar": True,
            "run_stationary_gvar": True,
            "run_correlation_baseline": True,
            "run_descriptives_only": False,
            "can_compute_momentary_impact": True,
            "full_time_varying_ready": bool(variant == "TIME_VARYING_gVAR"),
            "tv_full_confidence": tv_full_confidence,
            "why_not_time_varying": why_not_tv,
            "notes": [
                "Execution policy is all_methods: running all methods regardless of readiness recommendation.",
            ],
        }

    run_tv = False
    run_stationary = False
    run_corr = False
    run_desc = False

    if tier == "Tier3_LaggedDynamicNetwork" and variant == "TIME_VARYING_gVAR":
        run_tv = True
        run_stationary = True
        run_corr = True
        analysis_set = "tier3_full_time_varying"
    elif tier == "Tier3_LaggedDynamicNetwork" and variant == "STATIC_gVAR":
        run_tv = False
        run_stationary = True
        run_corr = True
        analysis_set = "tier3_static_lagged"
    elif tier == "Tier2_ContemporaneousPartialCorrelation":
        run_tv = False
        run_stationary = False
        run_corr = True
        analysis_set = "tier2_contemporaneous"
    elif tier == "Tier1_CorrelationMatrix":
        run_tv = False
        run_stationary = False
        run_corr = True
        analysis_set = "tier1_correlation_only"
    else:
        run_tv = False
        run_stationary = False
        run_corr = False
        run_desc = True
        analysis_set = "tier0_descriptives_only"

    notes: List[str] = [
        "Execution policy is readiness_aligned: methods follow readiness tier/variant recommendation.",
    ]
    if plan_from_readiness:
        notes.append("Readiness report includes analysis_execution_plan metadata.")

    return {
        "execution_policy": execution_policy,
        "recommended_tier": tier,
        "recommended_variant": variant,
        "analysis_set": analysis_set,
        "run_tv_gvar": run_tv,
        "run_stationary_gvar": run_stationary,
        "run_correlation_baseline": run_corr,
        "run_descriptives_only": run_desc,
        "can_compute_momentary_impact": bool(run_tv or run_stationary),
        "full_time_varying_ready": bool(run_tv),
        "tv_full_confidence": tv_full_confidence,
        "why_not_time_varying": why_not_tv,
        "notes": notes,
    }


def _infer_time_cols_from_readiness(readiness: Dict[str, Any], df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    ti = readiness.get("dataset_overview", {}).get("time_info", {}) or {}
    time_col = ti.get("time_col")
    date_col = ti.get("date_col")

    if time_col not in df.columns:
        time_col = None
    if date_col not in df.columns:
        date_col = None

    # fallback heuristics
    if date_col is None:
        for c in df.columns:
            lc = str(c).strip().lower()
            if lc in {"timestamp", "datetime", "date_time", "date", "time"}:
                date_col = c
                break
        if date_col is None:
            for c in df.columns:
                lc = str(c).strip().lower()
                if "date" in lc or "time" in lc:
                    parsed = pd.to_datetime(df[c], errors="coerce")
                    if parsed.notna().mean() >= 0.80:
                        date_col = c
                        break

    if time_col is None:
        for c in df.columns:
            lc = str(c).strip().lower()
            if lc in {"t_index", "t", "time_index", "timeidx", "index"}:
                time_col = c
                break
        if time_col is None:
            for c in df.columns:
                if c == date_col:
                    continue
                s = pd.to_numeric(df[c], errors="coerce")
                if s.notna().mean() >= 0.95:
                    diffs = s.dropna().diff()
                    if diffs.empty or (diffs >= 0).mean() > 0.98:
                        time_col = c
                        break

    return time_col, date_col


def _sort_df_by_time(df: pd.DataFrame, time_col: Optional[str], date_col: Optional[str]) -> pd.DataFrame:
    out = df.copy()
    if date_col and date_col in out.columns:
        dt = pd.to_datetime(out[date_col], errors="coerce")
        if dt.notna().mean() >= 0.80:
            out["_tmp_dt_"] = dt
            out = out.sort_values("_tmp_dt_", kind="mergesort").drop(columns=["_tmp_dt_"])
            return out
    if time_col and time_col in out.columns:
        t = pd.to_numeric(out[time_col], errors="coerce")
        if t.notna().mean() >= 0.80:
            out["_tmp_t_"] = t
            out = out.sort_values("_tmp_t_", kind="mergesort").drop(columns=["_tmp_t_"])
            return out
    return out


def compute_t_norm(df: pd.DataFrame, time_col: Optional[str], date_col: Optional[str]) -> np.ndarray:
    if time_col and (time_col in df.columns):
        t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(t).sum() >= max(3, int(0.7 * len(t))):
            if not np.all(np.isfinite(t)):
                idx = np.arange(len(t), dtype=float)
                good = np.isfinite(t)
                if good.sum() >= 2:
                    t = np.interp(idx, idx[good], t[good])
                else:
                    t = idx
            t0, t1 = float(np.min(t)), float(np.max(t))
            return (t - t0) / (t1 - t0 + 1e-12)

    if date_col and (date_col in df.columns):
        dt = pd.to_datetime(df[date_col], errors="coerce")
        if dt.notna().sum() >= max(3, int(0.7 * len(dt))):
            t = dt.view("int64").to_numpy(dtype=float)
            t0, t1 = float(np.nanmin(t)), float(np.nanmax(t))
            return (t - t0) / (t1 - t0 + 1e-12)

    idx = np.arange(len(df), dtype=float)
    return (idx - idx[0]) / (idx[-1] - idx[0] + 1e-12)


def select_network_variables(
    readiness: Dict[str, Any],
    df: pd.DataFrame,
) -> Tuple[List[str], Dict[str, Any]]:
    vars_meta = readiness.get("variables", {}) or {}
    cand = _readiness_candidate_vars(readiness, prefer_tier=CFG.prefer_tier)
    cand = [v for v in cand if v in df.columns and v in vars_meta]

    info: Dict[str, Any] = {
        "prefer_tier": CFG.prefer_tier,
        "n_candidates_from_readiness": len(cand),
        "filters": {
            "min_nonmissing_for_network": CFG.min_nonmissing_for_network,
            "max_missing_frac_for_network": CFG.max_missing_frac_for_network,
            "min_unique_for_network": CFG.min_unique_for_network,
            "min_std_for_network": CFG.min_std_for_network,
            "allow_binary": CFG.allow_binary,
            "allowed_types": list(CFG.allowed_types),
        },
        "dropped": [],
        "kept": [],
        "notes": [],
    }

    kept: List[str] = []
    for v in cand:
        if _hard_dropped(readiness, v):
            info["dropped"].append({"var": v, "reason": "hard_dropped_in_readiness"})
            continue

        t = _infer_var_type_from_readiness(readiness, v)
        if t not in CFG.allowed_types:
            info["dropped"].append({"var": v, "reason": f"type_not_allowed:{t}"})
            continue
        if (t == "binary") and (not CFG.allow_binary):
            info["dropped"].append({"var": v, "reason": "binary_not_allowed"})
            continue

        s = pd.to_numeric(df[v], errors="coerce").astype(float)
        nn = int(s.notna().sum())
        if nn < CFG.min_nonmissing_for_network:
            info["dropped"].append({"var": v, "reason": f"too_few_nonmissing:{nn}"})
            continue
        miss_frac = float(1.0 - nn / max(1, len(s)))
        if miss_frac > CFG.max_missing_frac_for_network:
            info["dropped"].append({"var": v, "reason": f"missing_too_high:{miss_frac:.3f}"})
            continue

        nunique = int(s.dropna().nunique())
        if nunique < CFG.min_unique_for_network:
            info["dropped"].append({"var": v, "reason": f"too_few_unique:{nunique}"})
            continue

        sd = float(np.nanstd(s.to_numpy(dtype=float)))
        if (not np.isfinite(sd)) or (sd < CFG.min_std_for_network):
            info["dropped"].append({"var": v, "reason": f"too_low_sd:{sd:.3g}"})
            continue

        kept.append(v)

    if len(kept) < 2:
        info["notes"].append("After strict filtering <2 vars remained; relaxing to any numeric non-hard-drop with enough data.")
        kept2 = []
        for v in df.columns:
            if v not in vars_meta:
                continue
            if _hard_dropped(readiness, v):
                continue
            s = pd.to_numeric(df[v], errors="coerce").astype(float)
            nn = int(s.notna().sum())
            if nn >= max(8, CFG.min_nonmissing_for_network // 2) and s.dropna().nunique() >= 3:
                kept2.append(v)
        kept = kept2

    info["kept"] = kept
    return kept, info


# ============================================================
# Variables metadata: labels + predictor/criterion roles
# ============================================================
def load_variable_labels(metadata_csv: Path) -> Dict[str, str]:
    if not metadata_csv.exists():
        return {}
    df = pd.read_csv(metadata_csv)
    if df.shape[1] == 0:
        return {}

    cols_lower = [c.lower() for c in df.columns]
    code_col = None
    label_col = None

    for cand in ["code", "var", "variable", "name", "item", "id"]:
        if cand in cols_lower:
            code_col = df.columns[cols_lower.index(cand)]
            break
    for cand in ["label", "description", "text", "meaning", "title"]:
        if cand in cols_lower:
            label_col = df.columns[cols_lower.index(cand)]
            break

    if code_col is None:
        code_col = df.columns[0]
    if label_col is None:
        label_col = df.columns[1] if df.shape[1] >= 2 else df.columns[0]

    mapping: Dict[str, str] = {}
    for _, row in df.iterrows():
        code = str(row[code_col]).strip()
        lab = str(row[label_col]).strip()
        if code and code.lower() != "nan":
            mapping[code] = lab if lab.lower() != "nan" else code
    return mapping


def infer_roles_predictor_criterion(metadata_csv: Path, vars_kept: List[str]) -> Dict[str, Any]:
    """
    Tries to infer predictor/criterion sets.
    Priority:
      1) variables_metadata.csv columns like role/group/category containing predictor/criterion keywords
      2) fallback: variable name prefix: P* -> predictor, C* -> criterion
    """
    predictors: List[str] = []
    criteria: List[str] = []
    notes: List[str] = []

    if metadata_csv.exists():
        df = pd.read_csv(metadata_csv)
        cols_lower = [c.lower() for c in df.columns]
        code_col = None
        role_col = None

        for cand in ["code", "var", "variable", "name", "item", "id"]:
            if cand in cols_lower:
                code_col = df.columns[cols_lower.index(cand)]
                break
        for cand in ["role", "group", "set", "category", "type", "domain"]:
            if cand in cols_lower:
                role_col = df.columns[cols_lower.index(cand)]
                break

        if code_col is not None and role_col is not None:
            role_map: Dict[str, str] = {}
            for _, r in df.iterrows():
                v = str(r[code_col]).strip()
                role = str(r[role_col]).strip().lower()
                role_map[v] = role

            for v in vars_kept:
                role = role_map.get(v, "")
                if "predict" in role or "x" == role:
                    predictors.append(v)
                elif "criter" in role or "outcome" in role or "target" in role or "y" == role:
                    criteria.append(v)

            if predictors or criteria:
                notes.append("Roles inferred from metadata role/group column.")
        else:
            notes.append("Metadata file present but role/group column not detected; using prefix fallback.")
    else:
        notes.append("No metadata file; using prefix fallback.")

    if not predictors and not criteria:
        for v in vars_kept:
            sv = str(v)
            if sv.startswith("P"):
                predictors.append(v)
            elif sv.startswith("C"):
                criteria.append(v)
        if predictors or criteria:
            notes.append("Roles inferred from name prefixes P*/C*.")
        else:
            notes.append("No roles inferred; treating ALL variables as both predictors and criteria (VAR uses all).")

    # In VAR(1), every node is both predictor and criterion; role sets are used only for importance summaries.
    predictors = [v for v in predictors if v in vars_kept]
    criteria = [v for v in criteria if v in vars_kept]

    return {
        "predictors": predictors,
        "criteria": criteria,
        "notes": notes,
    }


# ============================================================
# Repro + numerics
# ============================================================
def set_seed(seed: int) -> np.random.Generator:
    np.random.seed(seed)
    return np.random.default_rng(seed)


def safe_inv(M: np.ndarray, ridge: float = 1e-10) -> np.ndarray:
    p = int(M.shape[0])
    return np.linalg.inv(M + float(ridge) * np.eye(p))


def spectral_radius(A: np.ndarray) -> float:
    vals = np.linalg.eigvals(A)
    return float(np.max(np.abs(vals)))


def stabilize_by_scaling(A: np.ndarray, target_rho: float = 0.98) -> np.ndarray:
    rho = spectral_radius(A)
    if rho <= target_rho or rho == 0.0:
        return A
    return (target_rho / rho) * A


def zscore_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return mu, sd


def zscore_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd


def zscore(X: np.ndarray) -> np.ndarray:
    mu, sd = zscore_fit(X)
    return zscore_apply(X, mu, sd)


def gaussian_kernel_weights(t: np.ndarray, te: float, bandwidth: float) -> np.ndarray:
    bw = max(float(bandwidth), 1e-9)
    z = (t - float(te)) / bw
    w = np.exp(-0.5 * z * z) / (math.sqrt(2.0 * math.pi) * bw)
    return w


def normalize_weights_sum_to_n(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    n = float(w.size)
    s = float(np.sum(w)) + 1e-12
    return w * (n / s)


def weighted_mean(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    wsum = np.sum(w) + 1e-12
    return (w[:, None] * X).sum(axis=0) / wsum


def weighted_cov(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    mu = weighted_mean(X, w)
    Xc = X - mu[None, :]
    wsum = np.sum(w) + 1e-12
    return (Xc.T * w) @ Xc / wsum


def precision_to_partial_corr(Theta: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(Theta), 1e-12, np.inf))
    pc = -Theta / (d[:, None] * d[None, :])
    np.fill_diagonal(pc, 1.0)
    return pc


def effective_sample_size(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    sw = float(np.sum(w)) + 1e-12
    return float((sw * sw) / (float(np.sum(w * w)) + 1e-12))


# ============================================================
# Multivariate Kalman smoother imputation (VAR(1) state model)
# ============================================================
@dataclass
class KalmanModel:
    A: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    mu: np.ndarray  # (p,)


def initial_interpolate_fill(Y: np.ndarray) -> np.ndarray:
    df = pd.DataFrame(Y.copy())
    df = df.interpolate(method="linear", axis=0, limit_direction="both")
    df = df.ffill().bfill().fillna(0.0)
    return df.to_numpy(dtype=float)


def estimate_var1_ridge_from_filled(
    Y_filled_centered: np.ndarray,
    ridge_alpha: float,
    stabilize_rho: float = 0.98,
) -> Tuple[np.ndarray, np.ndarray]:
    Y = np.asarray(Y_filled_centered, dtype=float)
    n, p = Y.shape
    if n < 3:
        return np.eye(p) * 0.4, np.eye(p) * 1e-2

    X = Y[:-1]   # (n-1, p)
    T = Y[1:]    # (n-1, p)

    XtX = X.T @ X
    lam = float(ridge_alpha)
    B = safe_inv(XtX + lam * np.eye(p), ridge=1e-10) @ (X.T @ T)  # (p,p)
    A = B.T

    E = T - (X @ B)
    Q = np.cov(E, rowvar=False) + 1e-8 * np.eye(p)

    A = stabilize_by_scaling(A, target_rho=float(stabilize_rho))
    return A, Q


def kalman_filter_smoother_var1(
    Y_obs_centered: np.ndarray,
    A: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    m0: np.ndarray,
    P0: np.ndarray,
    ridge: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    x_t = A x_{t-1} + w_t
    y_t = x_t + v_t
    Missing entries treated as unobserved (partial measurement update).
    """
    Y = np.asarray(Y_obs_centered, dtype=float)
    n, p = Y.shape
    I = np.eye(p)

    m_pred = np.zeros((n, p))
    P_pred = np.zeros((n, p, p))
    m_filt = np.zeros((n, p))
    P_filt = np.zeros((n, p, p))

    m_prev = m0.copy()
    P_prev = P0.copy()

    for t in range(n):
        if t == 0:
            mp, Pp = m_prev, P_prev
        else:
            mp = A @ m_prev
            Pp = A @ P_prev @ A.T + Q

        m_pred[t] = mp
        P_pred[t] = Pp

        y = Y[t]
        obs_mask = np.isfinite(y)
        k = int(obs_mask.sum())

        if k == 0:
            mf, Pf = mp, Pp
        else:
            idx = np.where(obs_mask)[0]
            H = I[idx, :]                 # (k,p)
            y_obs = y[idx]                # (k,)
            Rk = R[np.ix_(idx, idx)]      # (k,k)

            S = H @ Pp @ H.T + Rk + float(ridge) * np.eye(k)
            K = Pp @ H.T @ safe_inv(S, ridge=float(ridge))   # (p,k)
            innov = y_obs - (H @ mp)
            mf = mp + K @ innov
            Pf = Pp - K @ S @ K.T
            Pf = 0.5 * (Pf + Pf.T)

        m_filt[t] = mf
        P_filt[t] = Pf
        m_prev, P_prev = mf, Pf

    # RTS smoother
    m_smooth = np.zeros((n, p))
    P_smooth = np.zeros((n, p, p))
    m_smooth[-1] = m_filt[-1]
    P_smooth[-1] = P_filt[-1]

    for t in range(n - 2, -1, -1):
        Pp_next = P_pred[t + 1]
        J = P_filt[t] @ A.T @ safe_inv(Pp_next, ridge=float(ridge))
        m_smooth[t] = m_filt[t] + J @ (m_smooth[t + 1] - m_pred[t + 1])
        P_smooth[t] = P_filt[t] + J @ (P_smooth[t + 1] - Pp_next) @ J.T
        P_smooth[t] = 0.5 * (P_smooth[t] + P_smooth[t].T)

    P_diag = np.stack([np.diag(P_smooth[t]) for t in range(n)], axis=0)
    return m_smooth, P_diag


def kalman_impute_multivariate(
    Y_obs: Optional[np.ndarray] = None,
    *,
    # alias for older call-site compatibility
    X_obs: Optional[np.ndarray] = None,
    seed: int = 42,
    n_refine: int = 3,
    ridge_alpha: float = 1.0,
    meas_noise_scale: float = 1e-2,
    stabilize_rho: float = 0.98,
    kalman_ridge: float = 1e-8,
    verbose: bool = True,
) -> Tuple[np.ndarray, KalmanModel, Dict[str, float]]:
    if Y_obs is None and X_obs is None:
        raise ValueError("kalman_impute_multivariate requires Y_obs or X_obs.")
    if Y_obs is None:
        Y_obs = X_obs
    assert Y_obs is not None

    _ = set_seed(seed)
    Y = np.asarray(Y_obs, dtype=float)
    n, p = Y.shape

    mu = np.nanmean(Y, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    Yc = Y - mu[None, :]

    missing_frac = float(np.mean(~np.isfinite(Y)))

    Yc_fill = initial_interpolate_fill(Yc)

    var = np.nanvar(Yc, axis=0)
    var = np.where((~np.isfinite(var)) | (var < 1e-8), 1.0, var)
    R = np.diag(float(meas_noise_scale) * var)

    m0 = np.zeros(p)
    if n > 0:
        first = Yc[0]
        obs0 = np.isfinite(first)
        if obs0.any():
            m0[obs0] = first[obs0]
    P0 = np.diag(var + 1e-2)

    A = np.eye(p) * 0.3
    Q = np.eye(p) * 1e-2

    iters = max(1, int(n_refine))
    if verbose:
        log(f"Kalman imputation: n={n}, p={p}, missing={missing_frac:.3%}, refine_iters={iters}")

    for it in range(iters):
        if verbose:
            log(f"  [Kalman] refine {it+1}/{iters}: estimate VAR(1) transition (ridge_alpha={ridge_alpha})")
        A, Q = estimate_var1_ridge_from_filled(
            Y_filled_centered=Yc_fill,
            ridge_alpha=float(ridge_alpha),
            stabilize_rho=float(stabilize_rho),
        )
        if verbose:
            log(f"  [Kalman] refine {it+1}/{iters}: run filter+smoother (rho(A)={spectral_radius(A):.4f})")
        x_smooth, _Pdiag = kalman_filter_smoother_var1(
            Y_obs_centered=Yc,
            A=A,
            Q=Q,
            R=R,
            m0=m0,
            P0=P0,
            ridge=float(kalman_ridge),
        )
        missing = ~np.isfinite(Yc)
        Yc_fill = Yc_fill.copy()
        Yc_fill[missing] = x_smooth[missing]

    Y_imp = Yc_fill + mu[None, :]
    model = KalmanModel(A=A, Q=Q, R=R, mu=mu)

    diagnostics = {
        "n": int(n),
        "p": int(p),
        "missing_fraction": float(missing_frac),
        "spectral_radius_A": float(spectral_radius(A)),
        "meas_noise_scale": float(meas_noise_scale),
        "ridge_alpha": float(ridge_alpha),
        "n_refine": int(iters),
        "kalman_ridge": float(kalman_ridge),
        "stabilize_rho": float(stabilize_rho),
    }
    return Y_imp, model, diagnostics


# ============================================================
# VAR(1) with weighted Lasso
# ============================================================
@dataclass
class LassoFitResult:
    coef: np.ndarray
    intercept: float
    alpha: float


def weighted_lasso_fit(Z: np.ndarray, y: np.ndarray, w: np.ndarray, alpha: float, seed: int = 42) -> LassoFitResult:
    model = Lasso(alpha=float(alpha), fit_intercept=True, max_iter=20000, random_state=int(seed))
    model.fit(Z, y, sample_weight=w)
    return LassoFitResult(coef=model.coef_.copy(), intercept=float(model.intercept_), alpha=float(alpha))


# ============================================================
# tvKS estimator (time-varying gVAR)
# ============================================================
@dataclass
class TvKsResult:
    estpoints: np.ndarray
    bandwidth: float
    Bhat: np.ndarray
    intercepts: np.ndarray
    alphas: np.ndarray
    r2_node: np.ndarray
    partial_corr: np.ndarray
    mse_point: np.ndarray
    ess_point: np.ndarray


def estimate_tvKS(
    X: np.ndarray,
    t_norm: np.ndarray,
    estpoints: np.ndarray,
    bandwidth: float,
    alphas_node: np.ndarray,
    seed: int = 42,
    ridge_precision: float = 1e-3,
    verbose: bool = False,
) -> TvKsResult:
    """
    Kernel-smoothed L1 VAR(1):
      X_t = B(t) X_{t-1} + c(t) + e_t
    Each estpoint te: weighted Lasso per node using Gaussian kernel on t_obs (aligned with Y=X[1:]).
    """
    Y = X[1:]
    Z = X[:-1]
    t_obs = t_norm[1:]
    n_obs, p = Y.shape
    m = len(estpoints)

    Bhat = np.zeros((m, p, p))
    intercepts = np.zeros((m, p))
    r2_node = np.zeros((m, p))
    partial_corr = np.zeros((m, p, p))
    mse_point = np.zeros(m)
    ess_point = np.zeros(m)

    alphas = np.tile(alphas_node[None, :], (m, 1))

    for ei, te in enumerate(estpoints):
        if verbose:
            log(f"    [tvKS] estpoint {ei+1}/{m} (t={te:.3f})")
        w_raw = gaussian_kernel_weights(t_obs, float(te), float(bandwidth))
        w = normalize_weights_sum_to_n(w_raw)
        ess_point[ei] = effective_sample_size(w)

        for i in range(p):
            a = float(alphas_node[i])
            fit = weighted_lasso_fit(Z, Y[:, i], w, alpha=a, seed=seed)
            Bhat[ei, i, :] = fit.coef
            intercepts[ei, i] = fit.intercept

            y = Y[:, i]
            y_pred = (Z @ fit.coef) + fit.intercept
            wsum = float(np.sum(w))
            y_bar = float(np.sum(w * y) / (wsum + 1e-12))
            sst = float(np.sum(w * (y - y_bar) ** 2))
            sse = float(np.sum(w * (y - y_pred) ** 2))
            r2_node[ei, i] = 1.0 - (sse / (sst + 1e-12))

        Yhat = intercepts[ei][None, :] + (Z @ Bhat[ei].T)
        mse_point[ei] = float(np.sum(w[:, None] * (Y - Yhat) ** 2) / (np.sum(w) * p + 1e-12))

        # contemporaneous: precision of residual covariance -> partial correlations
        E = Y - Yhat
        S = weighted_cov(E, w)
        Theta = safe_inv(S, ridge=float(ridge_precision))
        partial_corr[ei] = precision_to_partial_corr(Theta)

    return TvKsResult(
        estpoints=estpoints.copy(),
        bandwidth=float(bandwidth),
        Bhat=Bhat,
        intercepts=intercepts,
        alphas=alphas,
        r2_node=r2_node,
        partial_corr=partial_corr,
        mse_point=mse_point,
        ess_point=ess_point,
    )


# ============================================================
# CV for tvKS hyperparameters (TimeSeriesSplit, leakage-safe)
# ============================================================
@dataclass
class TvCvSummary:
    n_splits: int
    standardize_in_cv: bool
    alpha_fixed_for_bw: float
    bw_grid: List[float]
    bw_cv_mse: List[float]
    bw_selected: float
    alpha_grid: List[float]
    alpha_cv_mse: List[float]
    alpha_selected: float


def _tv_cv_mse(
    X_raw: np.ndarray,
    t_norm: np.ndarray,
    estpoints: np.ndarray,
    bandwidth: float,
    alpha: float,
    n_splits: int,
    seed: int,
    standardize_in_cv: bool,
    ridge_precision: float,
) -> float:
    """
    Forward-chaining CV MSE:
      - Split on VAR rows (Z/Y)
      - Fit tvKS on training prefix only
      - Predict validation block using nearest estpoint model
      - Standardization is leakage-safe (fit on training prefix only), if enabled
    """
    X_raw = np.asarray(X_raw, dtype=float)
    t_norm = np.asarray(t_norm, dtype=float)

    Z_full = X_raw[:-1]
    Y_full = X_raw[1:]
    n_obs, p = Y_full.shape

    tscv = TimeSeriesSplit(n_splits=int(n_splits))
    fold_mses: List[float] = []

    for fold_id, (tr, va) in enumerate(tscv.split(Z_full)):
        if len(tr) == 0 or len(va) == 0:
            continue

        train_end = int(tr[-1])
        val_end = int(va[-1])

        X_prefix_raw = X_raw[: val_end + 2]
        t_prefix = t_norm[: val_end + 2]

        if standardize_in_cv:
            mu, sd = zscore_fit(X_prefix_raw[: train_end + 2])
            X_prefix = zscore_apply(X_prefix_raw, mu, sd)
        else:
            X_prefix = X_prefix_raw

        X_train = X_prefix[: train_end + 2]
        t_train = t_prefix[: train_end + 2]

        tv = estimate_tvKS(
            X=X_train,
            t_norm=t_train,
            estpoints=estpoints,
            bandwidth=float(bandwidth),
            alphas_node=np.full(p, float(alpha)),
            seed=int(seed) + 17 * fold_id,
            ridge_precision=float(ridge_precision),
            verbose=False,
        )

        Zp = X_prefix[:-1]
        Yp = X_prefix[1:]
        t_obs_p = t_prefix[1:]

        va_t = t_obs_p[va]
        idx = np.argmin(np.abs(estpoints[None, :] - va_t[:, None]), axis=1).astype(int)

        B_sel = tv.Bhat[idx]
        c_sel = tv.intercepts[idx]
        x_sel = Zp[va]

        yhat = c_sel + np.einsum("nij,nj->ni", B_sel, x_sel)
        mse = float(np.mean((Yp[va] - yhat) ** 2))
        fold_mses.append(mse)

    return float(np.mean(fold_mses)) if fold_mses else float("inf")


def select_bandwidth_tv_cv(
    X_raw: np.ndarray,
    t_norm: np.ndarray,
    estpoints_cv: np.ndarray,
    bw_grid: np.ndarray,
    alpha_fixed: float,
    n_splits: int,
    seed: int,
    standardize_in_cv: bool,
    ridge_precision: float,
    verbose: bool,
) -> Tuple[float, List[float]]:
    scores: List[float] = []
    for b in bw_grid:
        mse = _tv_cv_mse(
            X_raw=X_raw,
            t_norm=t_norm,
            estpoints=estpoints_cv,
            bandwidth=float(b),
            alpha=float(alpha_fixed),
            n_splits=int(n_splits),
            seed=int(seed),
            standardize_in_cv=bool(standardize_in_cv),
            ridge_precision=float(ridge_precision),
        )
        scores.append(float(mse))
        if verbose:
            log(f"  [CV bw] bw={float(b):.4f} mse={float(mse):.6f}")
    best_idx = int(np.argmin(scores))
    return float(bw_grid[best_idx]), scores


def select_alpha_tv_cv(
    X_raw: np.ndarray,
    t_norm: np.ndarray,
    estpoints_cv: np.ndarray,
    bandwidth_fixed: float,
    alpha_grid: np.ndarray,
    n_splits: int,
    seed: int,
    standardize_in_cv: bool,
    ridge_precision: float,
    verbose: bool,
) -> Tuple[float, List[float]]:
    scores: List[float] = []
    for a in alpha_grid:
        mse = _tv_cv_mse(
            X_raw=X_raw,
            t_norm=t_norm,
            estpoints=estpoints_cv,
            bandwidth=float(bandwidth_fixed),
            alpha=float(a),
            n_splits=int(n_splits),
            seed=int(seed),
            standardize_in_cv=bool(standardize_in_cv),
            ridge_precision=float(ridge_precision),
        )
        scores.append(float(mse))
        if verbose:
            log(f"  [CV alpha] alpha={float(a):.6f} mse={float(mse):.6f}")
    best_idx = int(np.argmin(scores))
    return float(alpha_grid[best_idx]), scores


# ============================================================
# Moving-block bootstrap CIs (temporal B only)
# ============================================================
@dataclass
class BootstrapCI:
    B_low: np.ndarray
    B_high: np.ndarray


def moving_block_bootstrap_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    if block_len <= 1:
        return rng.integers(0, n, size=n)
    starts = rng.integers(0, max(1, n - block_len + 1), size=int(math.ceil(n / block_len)))
    idx = np.concatenate([np.arange(s, s + block_len) for s in starts], axis=0)[:n]
    return idx


def bootstrap_tvKS_CI(
    X: np.ndarray,
    t_norm: np.ndarray,
    tv_fit: TvKsResult,
    alphas_node: np.ndarray,
    n_boot: int,
    block_len: int,
    ci: Tuple[float, float],
    seed: int,
    n_jobs: int,
    verbose: bool,
) -> BootstrapCI:
    """
    Bootstrap only the temporal coefficients Bhat(t).
    """
    Y0 = X[1:]
    Z0 = X[:-1]
    t_obs0 = t_norm[1:]
    n_obs, p = Y0.shape
    m = len(tv_fit.estpoints)

    def one_boot(bi: int) -> np.ndarray:
        rng = np.random.default_rng(int(seed) + 1000 + bi)
        idx = moving_block_bootstrap_indices(n_obs, int(block_len), rng)
        Y = Y0[idx]
        Z = Z0[idx]
        t_obs = t_obs0[idx]

        B_b = np.zeros((m, p, p), dtype=float)
        for ei, te in enumerate(tv_fit.estpoints):
            w_raw = gaussian_kernel_weights(t_obs, float(te), float(tv_fit.bandwidth))
            w = normalize_weights_sum_to_n(w_raw)
            for i in range(p):
                fit = weighted_lasso_fit(Z, Y[:, i], w, alpha=float(alphas_node[i]), seed=int(seed) + bi)
                B_b[ei, i, :] = fit.coef
        return B_b

    if n_jobs != 1 and HAVE_JOBLIB:
        if verbose:
            log(f"[Bootstrap] parallel: n_boot={n_boot}, n_jobs={n_jobs}, block_len={block_len}")
        draws = Parallel(n_jobs=int(n_jobs), verbose=0)(delayed(one_boot)(bi) for bi in range(int(n_boot)))
    else:
        if verbose:
            log(f"[Bootstrap] serial: n_boot={n_boot}, block_len={block_len}")
        draws = [one_boot(bi) for bi in range(int(n_boot))]

    B_draws = np.stack(draws, axis=0)
    low = np.quantile(B_draws, float(ci[0]), axis=0)
    high = np.quantile(B_draws, float(ci[1]), axis=0)
    return BootstrapCI(B_low=low, B_high=high)


# ============================================================
# Stationary gVAR
# ============================================================
@dataclass
class StationaryGvarResult:
    B: np.ndarray
    intercept: np.ndarray
    alphas: np.ndarray
    r2_node: np.ndarray
    residual_cov: np.ndarray
    partial_corr: np.ndarray


def fit_stationary_gvar(X: np.ndarray, alpha_grid: np.ndarray, seed: int = 42, verbose: bool = False) -> StationaryGvarResult:
    Y = X[1:]
    Z = X[:-1]
    n_obs, p = Y.shape
    tscv = TimeSeriesSplit(n_splits=5)
    w = np.ones(n_obs)

    B = np.zeros((p, p))
    intercept = np.zeros(p)
    alphas = np.zeros(p)
    r2_node = np.zeros(p)

    for i in range(p):
        best_a = float(alpha_grid[0])
        best_mse = float("inf")
        y = Y[:, i]

        for a in alpha_grid:
            mses = []
            for tr, va in tscv.split(Z):
                fit = weighted_lasso_fit(Z[tr], y[tr], w[tr], alpha=float(a), seed=seed)
                pred = (Z[va] @ fit.coef) + fit.intercept
                mses.append(float(np.mean((y[va] - pred) ** 2)))
            m = float(np.mean(mses))
            if m < best_mse:
                best_mse = m
                best_a = float(a)

        fit = weighted_lasso_fit(Z, y, w, alpha=best_a, seed=seed)
        B[i, :] = fit.coef
        intercept[i] = fit.intercept
        alphas[i] = best_a

        y_pred = (Z @ fit.coef) + fit.intercept
        y_bar = float(np.mean(y))
        sst = float(np.sum((y - y_bar) ** 2))
        sse = float(np.sum((y - y_pred) ** 2))
        r2_node[i] = 1.0 - (sse / (sst + 1e-12))

        if verbose and (i % max(1, p // 10) == 0):
            log(f"  [stationary] node {i+1}/{p}, alpha={best_a:.6f}, r2={r2_node[i]:.3f}")

    E = Y - (Z @ B.T + intercept[None, :])
    residual_cov = np.cov(E, rowvar=False)

    try:
        gl = GraphicalLassoCV()
        gl.fit(E)
        Theta = gl.precision_
    except Exception:
        lw = LedoitWolf().fit(E)
        Theta = safe_inv(lw.covariance_, ridge=1e-3)

    partial_corr = precision_to_partial_corr(Theta)
    return StationaryGvarResult(
        B=B,
        intercept=intercept,
        alphas=alphas,
        r2_node=r2_node,
        residual_cov=residual_cov,
        partial_corr=partial_corr,
    )


# ============================================================
# Correlation baselines
# ============================================================
@dataclass
class CorrResult:
    pearson: np.ndarray
    spearman: np.ndarray
    partial_corr_lw: np.ndarray


def compute_corr_baselines(X: np.ndarray) -> CorrResult:
    X = np.asarray(X, dtype=float)
    pearson = np.corrcoef(X, rowvar=False)

    # spearman via rank transform
    ranks = np.apply_along_axis(lambda a: pd.Series(a).rank(method="average").to_numpy(dtype=float), 0, X)
    spearman = np.corrcoef(ranks, rowvar=False)

    # partial correlation via LedoitWolf precision
    lw = LedoitWolf().fit(X)
    Theta = safe_inv(lw.covariance_, ridge=1e-3)
    pc = precision_to_partial_corr(Theta)
    return CorrResult(pearson=pearson, spearman=spearman, partial_corr_lw=pc)


# ============================================================
# Cluster/regime + edge shift detection
# ============================================================
@dataclass
class TvClusterResult:
    create_clusters: bool
    ratio_stationary_over_tv: np.ndarray
    threshold_ratio: float
    peaks: List[int]
    segments: List[Tuple[int, int]]


def detect_tv_clusters_by_error_ratio(
    X: np.ndarray,
    t_norm: np.ndarray,
    tv: TvKsResult,
    st: StationaryGvarResult,
    ratio_threshold: float = 3.0,
) -> TvClusterResult:
    Y = X[1:]
    Z = X[:-1]
    t_obs = t_norm[1:]
    m = len(tv.estpoints)
    p = Y.shape[1]

    mse_stat = np.zeros(m)
    Yhat_stat_full = st.intercept[None, :] + (Z @ st.B.T)

    for ei, te in enumerate(tv.estpoints):
        w_raw = gaussian_kernel_weights(t_obs, float(te), float(tv.bandwidth))
        w = normalize_weights_sum_to_n(w_raw)
        mse_stat[ei] = float(np.sum(w[:, None] * (Y - Yhat_stat_full) ** 2) / (np.sum(w) * p + 1e-12))

    ratio = mse_stat / (tv.mse_point + 1e-12)
    create = bool(np.max(ratio) >= ratio_threshold)

    peaks: List[int] = []
    segments: List[Tuple[int, int]] = []

    if not create:
        return TvClusterResult(False, ratio, float(ratio_threshold), [], [])

    idx = np.where(ratio >= ratio_threshold)[0]
    runs: List[List[int]] = []
    cur: List[int] = []
    for ii in idx:
        if not cur or ii == cur[-1] + 1:
            cur.append(int(ii))
        else:
            runs.append(cur)
            cur = [int(ii)]
    if cur:
        runs.append(cur)

    for run in runs:
        peak = int(run[int(np.argmax(ratio[run]))])
        peaks.append(peak)

    peaks = sorted(set(peaks))
    boundaries = [pp for pp in peaks if 0 <= pp < m - 1]
    start = 0
    for b in boundaries:
        segments.append((start, b))
        start = b + 1
    segments.append((start, m - 1))

    return TvClusterResult(True, ratio, float(ratio_threshold), peaks, segments)


def detect_change_points_by_network_distance(Bhat: np.ndarray, z_thresh: float = 2.5) -> Dict[str, Any]:
    """
    Heuristic regime detection:
    - compute d_t = ||B(t) - B(t-1)||_F
    - z-score d_t; change points where z > z_thresh
    """
    m = Bhat.shape[0]
    if m < 3:
        return {"enabled": False, "reason": "too_few_timepoints", "d": [], "z": [], "change_points": [], "segments": [(0, m - 1)]}

    d = []
    for t in range(1, m):
        diff = Bhat[t] - Bhat[t - 1]
        d.append(float(np.linalg.norm(diff, ord="fro")))
    d = np.array(d, dtype=float)
    mu = float(np.mean(d))
    sd = float(np.std(d) + 1e-12)
    z = (d - mu) / sd
    cps = [int(i + 1) for i in np.where(z > float(z_thresh))[0].tolist()]  # indices in [1..m-1]
    cps = sorted(set(cps))

    # segments
    segs: List[Tuple[int, int]] = []
    start = 0
    for cp in cps:
        segs.append((start, cp - 1))
        start = cp
    segs.append((start, m - 1))

    return {"enabled": True, "d": d.tolist(), "z": z.tolist(), "z_thresh": float(z_thresh), "change_points": cps, "segments": segs}


def detect_top_edge_shifts(
    Bhat: np.ndarray,
    estpoints: np.ndarray,
    labels: List[str],
    top_k: int = 200,
    exclude_self: bool = True,
) -> pd.DataFrame:
    """
    Edge-level shift ranking for B(t): for each directed edge dst<-src:
      - max_abs_delta_adj: max_t |b(t)-b(t-1)|
      - range: max(b)-min(b)
      - slope: linear slope over estpoints
      - argmax_delta_t: time index where adjacent delta is maximal
    """
    m, p, _ = Bhat.shape
    rows = []
    for dst in range(p):
        for src in range(p):
            if exclude_self and dst == src:
                continue
            b = Bhat[:, dst, src].astype(float)
            db = np.abs(np.diff(b)) if m >= 2 else np.array([0.0])
            max_adj = float(np.max(db)) if db.size else 0.0
            argmax = int(np.argmax(db) + 1) if db.size else 0
            rng = float(np.max(b) - np.min(b)) if m else 0.0
            # slope on estpoints
            if m >= 3:
                slope = float(np.polyfit(estpoints.astype(float), b, deg=1)[0])
            else:
                slope = 0.0
            rows.append(
                {
                    "dst": labels[dst],
                    "src": labels[src],
                    "max_abs_delta_adj": max_adj,
                    "argmax_delta_time_index": argmax,
                    "argmax_delta_t": float(estpoints[argmax]) if (0 <= argmax < m) else None,
                    "range": rng,
                    "slope": slope,
                    "mean": float(np.mean(b)),
                    "sd": float(np.std(b)),
                    "min": float(np.min(b)),
                    "max": float(np.max(b)),
                }
            )
    df = pd.DataFrame(rows)
    df = df.sort_values(["max_abs_delta_adj", "range"], ascending=False).head(int(top_k)).reset_index(drop=True)
    return df


# ============================================================
# Prediction and importance metrics
# ============================================================
def one_step_prediction_tv(X: np.ndarray, t_norm: np.ndarray, tv: TvKsResult) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Y = X[1:]
    Z = X[:-1]
    t_obs = t_norm[1:]
    idx = np.argmin(np.abs(tv.estpoints[None, :] - t_obs[:, None]), axis=1).astype(int)
    B_sel = tv.Bhat[idx]
    c_sel = tv.intercepts[idx]
    Yhat = c_sel + np.einsum("nij,nj->ni", B_sel, Z)
    return Y, Yhat, idx


def one_step_prediction_stationary(X: np.ndarray, st: StationaryGvarResult) -> Tuple[np.ndarray, np.ndarray]:
    Y = X[1:]
    Z = X[:-1]
    Yhat = st.intercept[None, :] + (Z @ st.B.T)
    return Y, Yhat


def prediction_metrics(Y: np.ndarray, Yhat: np.ndarray) -> Dict[str, float]:
    mse = float(np.mean((Y - Yhat) ** 2))
    y_bar = np.mean(Y, axis=0, keepdims=True)
    sst = float(np.sum((Y - y_bar) ** 2))
    sse = float(np.sum((Y - Yhat) ** 2))
    r2 = 1.0 - (sse / (sst + 1e-12))
    return {"mse": mse, "r2_overall": float(r2)}


def predictor_importance_leave_one_out_tv(
    X: np.ndarray,
    t_norm: np.ndarray,
    tv: TvKsResult,
    idx_map: np.ndarray,
    predictors_idx: List[int],
    criteria_idx: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Leave-one-predictor-out importance for tv:
      baseline Yhat already: c_sel + B_sel @ Z
      remove predictor j contribution: subtract B_sel[:,:,j] * Z[:,j]
    Return delta MSE for:
      - all targets
      - criteria-only targets (if provided)
    """
    Y = X[1:]
    Z = X[:-1]
    B_sel = tv.Bhat[idx_map]  # (n_obs,p,p)
    c_sel = tv.intercepts[idx_map]  # (n_obs,p)
    Yhat_base = c_sel + np.einsum("nij,nj->ni", B_sel, Z)

    base_mse_all = float(np.mean((Y - Yhat_base) ** 2))

    if criteria_idx:
        Yc = Y[:, criteria_idx]
        Yhc = Yhat_base[:, criteria_idx]
        base_mse_crit = float(np.mean((Yc - Yhc) ** 2))
    else:
        base_mse_crit = float("nan")

    rows = []
    for j in predictors_idx:
        contrib = B_sel[:, :, j] * Z[:, j][:, None]  # (n_obs,p)
        Yhat_loo = Yhat_base - contrib
        mse_all = float(np.mean((Y - Yhat_loo) ** 2))
        d_all = mse_all - base_mse_all

        if criteria_idx:
            mse_crit = float(np.mean((Y[:, criteria_idx] - Yhat_loo[:, criteria_idx]) ** 2))
            d_crit = mse_crit - base_mse_crit
        else:
            d_crit = float("nan")

        rows.append({"predictor_index": int(j), "delta_mse_all": d_all, "delta_mse_criteria": d_crit})

    return pd.DataFrame(rows)


def predictor_importance_leave_one_out_stationary(
    X: np.ndarray,
    st: StationaryGvarResult,
    predictors_idx: List[int],
    criteria_idx: Optional[List[int]] = None,
) -> pd.DataFrame:
    Y = X[1:]
    Z = X[:-1]
    Yhat_base = st.intercept[None, :] + (Z @ st.B.T)
    base_mse_all = float(np.mean((Y - Yhat_base) ** 2))
    if criteria_idx:
        base_mse_crit = float(np.mean((Y[:, criteria_idx] - Yhat_base[:, criteria_idx]) ** 2))
    else:
        base_mse_crit = float("nan")

    rows = []
    for j in predictors_idx:
        contrib = (st.B[:, j][None, :] * Z[:, j][:, None])  # WRONG orientation, fix below
        # st.B is (dst, src). Contribution to each dst is B[dst,j]*Z[:,j]
        contrib = Z[:, j][:, None] * st.B[:, j][None, :]  # (n_obs, p)
        Yhat_loo = Yhat_base - contrib
        mse_all = float(np.mean((Y - Yhat_loo) ** 2))
        d_all = mse_all - base_mse_all

        if criteria_idx:
            mse_crit = float(np.mean((Y[:, criteria_idx] - Yhat_loo[:, criteria_idx]) ** 2))
            d_crit = mse_crit - base_mse_crit
        else:
            d_crit = float("nan")

        rows.append({"predictor_index": int(j), "delta_mse_all": d_all, "delta_mse_criteria": d_crit})

    return pd.DataFrame(rows)


def coefficient_based_importance(
    B: np.ndarray,
    labels: List[str],
    predictors_idx: List[int],
    criteria_idx: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Coefficient-based importance (stationary B):
    - out_strength_all: sum abs(B[:,j]) over dst
    - out_strength_criteria: sum abs(B[criteria,j]) if criteria provided
    - signed_out_all: sum(B[:,j])
    - nonzero_frac_all: fraction(|B[:,j]|>0)
    """
    p = B.shape[0]
    crit = criteria_idx if criteria_idx else list(range(p))
    rows = []
    for j in predictors_idx:
        col = B[:, j]
        out_all = float(np.sum(np.abs(col)))
        out_crit = float(np.sum(np.abs(B[crit, j])))
        signed = float(np.sum(col))
        nz = float(np.mean(np.abs(col) > 1e-12))
        rows.append(
            {
                "predictor": labels[j],
                "out_strength_all": out_all,
                "out_strength_criteria": out_crit,
                "signed_out_all": signed,
                "nonzero_fraction_all": nz,
            }
        )
    return pd.DataFrame(rows).sort_values("out_strength_criteria", ascending=False).reset_index(drop=True)


def coefficient_based_importance_tv(
    Bhat: np.ndarray,
    labels: List[str],
    predictors_idx: List[int],
    criteria_idx: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Coefficient-based importance for tv:
    Use mean over time of abs coefficients.
    """
    m, p, _ = Bhat.shape
    crit = criteria_idx if criteria_idx else list(range(p))
    rows = []
    absB = np.abs(Bhat)
    for j in predictors_idx:
        out_all_t = absB[:, :, j].sum(axis=1)  # (m,) sum over dst
        out_crit_t = absB[:, crit, j].sum(axis=1)  # (m,)
        signed_t = Bhat[:, :, j].sum(axis=1)

        rows.append(
            {
                "predictor": labels[j],
                "out_strength_all_mean": float(np.mean(out_all_t)),
                "out_strength_all_sd": float(np.std(out_all_t)),
                "out_strength_criteria_mean": float(np.mean(out_crit_t)),
                "out_strength_criteria_sd": float(np.std(out_crit_t)),
                "signed_out_all_mean": float(np.mean(signed_t)),
                "nonzero_fraction_mean": float(np.mean(np.abs(Bhat[:, :, j]) > 1e-12)),
            }
        )
    return pd.DataFrame(rows).sort_values("out_strength_criteria_mean", ascending=False).reset_index(drop=True)


def criterion_dependence_tv(
    Bhat: np.ndarray,
    labels: List[str],
    predictors_idx: List[int],
    criteria_idx: List[int],
) -> pd.DataFrame:
    """
    For each criterion, quantify total incoming strength from predictors (mean over time).
    """
    absB = np.abs(Bhat)
    rows = []
    for i in criteria_idx:
        incoming = absB[:, i, predictors_idx].sum(axis=1)  # (m,)
        rows.append(
            {
                "criterion": labels[i],
                "incoming_from_predictors_mean": float(np.mean(incoming)),
                "incoming_from_predictors_sd": float(np.std(incoming)),
                "incoming_from_predictors_max": float(np.max(incoming)),
            }
        )
    return pd.DataFrame(rows).sort_values("incoming_from_predictors_mean", ascending=False).reset_index(drop=True)


# ============================================================
# Multicollinearity diagnostics
# ============================================================
def ridge_r2(y: np.ndarray, X: np.ndarray, ridge: float = 1e-3) -> float:
    """
    Ridge regression closed form; returns R^2 on the same data (diagnostic only).
    """
    y = y.astype(float)
    X = X.astype(float)
    if X.size == 0 or X.shape[1] == 0:
        return 0.0
    # center
    y0 = y - np.mean(y)
    X0 = X - np.mean(X, axis=0, keepdims=True)
    XtX = X0.T @ X0 + float(ridge) * np.eye(X0.shape[1])
    beta = safe_inv(XtX, ridge=1e-10) @ (X0.T @ y0)
    yhat = X0 @ beta
    sst = float(np.sum((y0) ** 2))
    sse = float(np.sum((y0 - yhat) ** 2))
    return float(1.0 - sse / (sst + 1e-12))


def compute_vif(X: np.ndarray, ridge: float = 1e-3) -> Dict[str, float]:
    """
    VIF per column: 1/(1-R^2_j), with ridge-stabilized R^2.
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    out: Dict[str, float] = {}
    for j in range(p):
        y = X[:, j]
        others = np.delete(X, j, axis=1)
        r2 = ridge_r2(y, others, ridge=float(ridge))
        vif = float(1.0 / (1.0 - r2 + 1e-12))
        out[str(j)] = vif
    return out


def multicollinearity_report(X: np.ndarray, labels: List[str], ridge: float = 1e-3, corr_thresholds: Tuple[float, ...] = (0.8, 0.9)) -> Dict[str, Any]:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.size == 0 or X.shape[1] == 0:
        return {
            "enabled": False,
            "n": int(X.shape[0]) if X.ndim == 2 else 0,
            "p": 0,
            "notes": ["insufficient variables for multicollinearity diagnostics"],
        }
    if X.shape[1] == 1:
        label = labels[0] if labels else "var_0"
        return {
            "enabled": True,
            "n": int(X.shape[0]),
            "p": 1,
            "condition_number_corr": 1.0,
            "eigenvalues_corr_desc": [1.0],
            "avg_abs_corr_offdiag": 0.0,
            "max_abs_corr_offdiag": 0.0,
            "high_corr_pairs": {str(thr): [] for thr in corr_thresholds},
            "vif_ridge": float(ridge),
            "vif": {label: 1.0},
            "notes": ["single-variable diagnostics only; pairwise correlation structure is not defined."],
        }
    # correlation
    C = np.corrcoef(X, rowvar=False)
    # cond number
    try:
        cond = float(np.linalg.cond(C))
    except Exception:
        cond = float("nan")
    # eigen spectrum
    try:
        eig = np.linalg.eigvalsh(C)
        eig = np.sort(eig)[::-1]
        eig_list = eig.tolist()
    except Exception:
        eig_list = []
    # high corr pairs
    pairs: Dict[str, List[Dict[str, Any]]] = {}
    absC = np.abs(C)
    iu = np.triu_indices_from(absC, k=1)
    for thr in corr_thresholds:
        sel = np.where(absC[iu] >= float(thr))[0]
        lst = []
        for k in sel[:20000]:  # guard
            i = int(iu[0][k])
            j = int(iu[1][k])
            lst.append({"i": labels[i], "j": labels[j], "abs_corr": float(absC[i, j]), "corr": float(C[i, j])})
        pairs[str(thr)] = lst

    vifs = compute_vif(X, ridge=float(ridge))
    vif_named = {labels[int(k)]: float(v) for k, v in vifs.items()}

    return {
        "n": int(X.shape[0]),
        "p": int(X.shape[1]),
        "condition_number_corr": cond,
        "eigenvalues_corr_desc": eig_list,
        "avg_abs_corr_offdiag": float(np.mean(absC[~np.eye(absC.shape[0], dtype=bool)])),
        "max_abs_corr_offdiag": float(np.max(absC[~np.eye(absC.shape[0], dtype=bool)])),
        "high_corr_pairs": pairs,
        "vif_ridge": float(ridge),
        "vif": vif_named,
    }


# ============================================================
# Network analysis (centralities & globals)
# ============================================================
def save_matrix_csv(path: Path, M: np.ndarray, labels: List[str]) -> None:
    df = pd.DataFrame(M, index=labels, columns=labels)
    df.to_csv(path, index=True)


def _build_directed_graph_from_B(
    B: np.ndarray,
    labels: List[str],
    abs_threshold: float = 1e-12,
    exclude_self: bool = True,
):
    if not HAVE_NETWORKX:
        return None

    G = nx.DiGraph()
    for node in labels:
        G.add_node(node)

    eps = 1e-12
    p = B.shape[0]
    for dst in range(p):
        for src in range(p):
            if exclude_self and dst == src:
                continue
            w = float(B[dst, src])
            aw = float(abs(w))
            if aw < float(abs_threshold):
                continue
            G.add_edge(
                labels[src],
                labels[dst],
                weight=w,
                abs_weight=aw,
                distance=float(1.0 / (aw + eps)),
                sign=1.0 if w >= 0 else -1.0,
            )
    return G


def _build_undirected_graph_from_W(
    W: np.ndarray,
    labels: List[str],
    abs_threshold: float = 1e-12,
    exclude_self: bool = True,
):
    if not HAVE_NETWORKX:
        return None

    G = nx.Graph()
    for node in labels:
        G.add_node(node)

    eps = 1e-12
    p = W.shape[0]
    for i in range(p):
        for j in range(i + 1, p):
            if exclude_self and i == j:
                continue
            w = float(W[i, j])
            aw = float(abs(w))
            if aw < float(abs_threshold):
                continue
            G.add_edge(
                labels[i],
                labels[j],
                weight=w,
                abs_weight=aw,
                distance=float(1.0 / (aw + eps)),
                sign=1.0 if w >= 0 else -1.0,
            )
    return G


def _communities_undirected(G_und: "nx.Graph") -> Tuple[List[List[str]], Dict[str, int], Dict[str, Any]]:
    """
    Get communities/modules on undirected graph.
    """
    meta: Dict[str, Any] = {"method": None, "ok": False, "notes": []}
    if not HAVE_NETWORKX:
        return [], {}, {"method": None, "ok": False, "notes": ["networkx not available"]}

    if G_und.number_of_edges() == 0 or G_und.number_of_nodes() == 0:
        return [], {}, {"method": None, "ok": False, "notes": ["empty graph"]}

    try:
        # Prefer Louvain if available
        if hasattr(nx.algorithms.community, "louvain_communities"):
            comms = nx.algorithms.community.louvain_communities(G_und, weight="abs_weight", seed=42)  # type: ignore
            meta.update({"method": "louvain_communities", "ok": True})
        else:
            comms = list(nx.algorithms.community.greedy_modularity_communities(G_und, weight="abs_weight"))
            meta.update({"method": "greedy_modularity_communities", "ok": True})
        comms_list = [sorted(list(c)) for c in comms]
        node_to_mod = {}
        for mi, c in enumerate(comms_list):
            for n in c:
                node_to_mod[n] = int(mi)
        return comms_list, node_to_mod, meta
    except Exception as e:
        meta["notes"].append(f"community detection failed: {repr(e)}")
        return [], {}, meta


def participation_coefficient(G_und: "nx.Graph", node_to_mod: Dict[str, int]) -> Dict[str, float]:
    """
    PC_i = 1 - sum_m (k_im/k_i)^2
    using abs_weight strengths on undirected graph.
    """
    out: Dict[str, float] = {}
    # strengths
    for n in G_und.nodes():
        ki = 0.0
        km: Dict[int, float] = {}
        for nb in G_und.neighbors(n):
            w = float(G_und.edges[n, nb].get("abs_weight", 0.0))
            ki += w
            m = int(node_to_mod.get(nb, -1))
            km[m] = km.get(m, 0.0) + w
        if ki <= 1e-12:
            out[n] = 0.0
        else:
            s = 0.0
            for m, v in km.items():
                frac = v / ki
                s += frac * frac
            out[n] = float(1.0 - s)
    return out


def _weighted_global_efficiency(G: "nx.Graph") -> float:
    """
    Weighted global efficiency: average of 1/d(i,j) over all pairs
    using 'distance' attribute.
    """
    n = G.number_of_nodes()
    if n <= 1:
        return 0.0
    nodes = list(G.nodes())
    total = 0.0
    count = 0
    for i in range(n):
        src = nodes[i]
        # single-source shortest paths
        try:
            lengths = nx.single_source_dijkstra_path_length(G, src, weight="distance")
        except Exception:
            lengths = {}
        for j in range(n):
            if i == j:
                continue
            dst = nodes[j]
            d = lengths.get(dst, None)
            if d is None:
                continue
            total += 1.0 / (float(d) + 1e-12)
            count += 1
    return float(total / max(1, count))


def compute_node_centralities(G: Any, directed: bool) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Returns a DataFrame with many centralities.
    Uses abs_weight for nonnegative-weight algorithms, and distance for path-based ones.
    """
    meta: Dict[str, Any] = {"enabled": HAVE_NETWORKX, "notes": []}
    if not HAVE_NETWORKX or G is None:
        meta["enabled"] = False
        meta["notes"].append("networkx not available or graph None.")
        return pd.DataFrame(), meta

    nodes = list(G.nodes())
    if len(nodes) == 0:
        meta["notes"].append("empty graph.")
        return pd.DataFrame(), meta

    df = pd.DataFrame({"node": nodes})

    # degree / strength
    if directed:
        df["in_degree"] = df["node"].map(lambda n: float(G.in_degree(n)))
        df["out_degree"] = df["node"].map(lambda n: float(G.out_degree(n)))
        df["in_strength_abs"] = df["node"].map(lambda n: float(G.in_degree(n, weight="abs_weight")))
        df["out_strength_abs"] = df["node"].map(lambda n: float(G.out_degree(n, weight="abs_weight")))
        # signed strengths
        def _signed_in(n: str) -> float:
            s = 0.0
            for u, v, d in G.in_edges(n, data=True):
                s += float(d.get("weight", 0.0))
            return float(s)

        def _signed_out(n: str) -> float:
            s = 0.0
            for u, v, d in G.out_edges(n, data=True):
                s += float(d.get("weight", 0.0))
            return float(s)

        df["in_strength_signed"] = df["node"].map(_signed_in)
        df["out_strength_signed"] = df["node"].map(_signed_out)
    else:
        df["degree"] = df["node"].map(lambda n: float(G.degree(n)))
        df["strength_abs"] = df["node"].map(lambda n: float(G.degree(n, weight="abs_weight")))
        def _signed_strength(n: str) -> float:
            s = 0.0
            for u, v, d in G.edges(n, data=True):
                s += float(d.get("weight", 0.0))
            return float(s)
        df["strength_signed"] = df["node"].map(_signed_strength)

    # betweenness (distance)
    try:
        btw = nx.betweenness_centrality(G, weight="distance", normalized=True)
        df["betweenness"] = df["node"].map(lambda n: float(btw.get(n, 0.0)))
    except Exception as e:
        meta["notes"].append(f"betweenness failed: {repr(e)}")
        df["betweenness"] = np.nan

    # closeness (distance)
    try:
        clo = nx.closeness_centrality(G, distance="distance")
        df["closeness"] = df["node"].map(lambda n: float(clo.get(n, 0.0)))
    except Exception as e:
        meta["notes"].append(f"closeness failed: {repr(e)}")
        df["closeness"] = np.nan

    # harmonic centrality (distance)
    try:
        # harmonic_centrality uses distance if available in recent nx
        if hasattr(nx, "harmonic_centrality"):
            hc = nx.harmonic_centrality(G, distance="distance")  # type: ignore
            df["harmonic"] = df["node"].map(lambda n: float(hc.get(n, 0.0)))
        else:
            df["harmonic"] = np.nan
    except Exception as e:
        meta["notes"].append(f"harmonic failed: {repr(e)}")
        df["harmonic"] = np.nan

    # eigenvector centrality (abs weights)
    try:
        eig = nx.eigenvector_centrality_numpy(G, weight="abs_weight")
        df["eigenvector"] = df["node"].map(lambda n: float(eig.get(n, 0.0)))
    except Exception as e:
        meta["notes"].append(f"eigenvector failed: {repr(e)}")
        df["eigenvector"] = np.nan

    # PageRank (abs weights)
    try:
        pr = nx.pagerank(G, weight="abs_weight")
        df["pagerank"] = df["node"].map(lambda n: float(pr.get(n, 0.0)))
    except Exception as e:
        meta["notes"].append(f"pagerank failed: {repr(e)}")
        df["pagerank"] = np.nan

    # Katz (abs weights)
    try:
        kz = nx.katz_centrality_numpy(G, weight="abs_weight", alpha=0.1, beta=1.0)
        df["katz"] = df["node"].map(lambda n: float(kz.get(n, 0.0)))
    except Exception as e:
        meta["notes"].append(f"katz failed: {repr(e)}")
        df["katz"] = np.nan

    # HITS hubs/authorities (abs weights)
    try:
        if directed:
            hubs, auth = nx.hits(G, max_iter=1000, normalized=True)
            df["hits_hub"] = df["node"].map(lambda n: float(hubs.get(n, 0.0)))
            df["hits_authority"] = df["node"].map(lambda n: float(auth.get(n, 0.0)))
        else:
            df["hits_hub"] = np.nan
            df["hits_authority"] = np.nan
    except Exception as e:
        meta["notes"].append(f"hits failed: {repr(e)}")
        df["hits_hub"] = np.nan
        df["hits_authority"] = np.nan

    # clustering coefficient (on undirected version)
    try:
        if directed:
            UG = G.to_undirected()
        else:
            UG = G
        clust = nx.clustering(UG, weight="abs_weight")
        df["clustering"] = df["node"].map(lambda n: float(clust.get(n, 0.0)))
    except Exception as e:
        meta["notes"].append(f"clustering failed: {repr(e)}")
        df["clustering"] = np.nan

    # k-core number (on undirected)
    try:
        UG = G.to_undirected() if directed else G
        core = nx.core_number(UG)
        df["core_number"] = df["node"].map(lambda n: float(core.get(n, 0.0)))
    except Exception as e:
        meta["notes"].append(f"core_number failed: {repr(e)}")
        df["core_number"] = np.nan

    # structural holes: constraint / effective size (if present)
    try:
        if hasattr(nx.algorithms, "structuralholes"):
            UG = G.to_undirected() if directed else G
            cons = nx.algorithms.structuralholes.constraint(UG, weight="abs_weight")  # type: ignore
            eff = nx.algorithms.structuralholes.effective_size(UG, weight="abs_weight")  # type: ignore
            df["constraint"] = df["node"].map(lambda n: float(cons.get(n, 0.0)))
            df["effective_size"] = df["node"].map(lambda n: float(eff.get(n, 0.0)))
        else:
            df["constraint"] = np.nan
            df["effective_size"] = np.nan
    except Exception as e:
        meta["notes"].append(f"structural holes metrics failed: {repr(e)}")
        df["constraint"] = np.nan
        df["effective_size"] = np.nan

    # participation coefficient (requires communities on undirected)
    try:
        UG = G.to_undirected() if directed else G
        comms, node_to_mod, cmeta = _communities_undirected(UG)
        if cmeta.get("ok"):
            pc = participation_coefficient(UG, node_to_mod=node_to_mod)
            df["participation_coeff"] = df["node"].map(lambda n: float(pc.get(n, 0.0)))
            df["module_id"] = df["node"].map(lambda n: int(node_to_mod.get(n, -1)))
            meta["communities"] = {"n_modules": len(comms), "method": cmeta.get("method")}
        else:
            df["participation_coeff"] = np.nan
            df["module_id"] = -1
            meta["communities"] = cmeta
    except Exception as e:
        meta["notes"].append(f"participation coefficient failed: {repr(e)}")
        df["participation_coeff"] = np.nan
        df["module_id"] = -1

    return df, meta


def compute_global_metrics(G: Any, directed: bool) -> Dict[str, Any]:
    out: Dict[str, Any] = {"enabled": HAVE_NETWORKX, "directed": bool(directed), "notes": []}
    if not HAVE_NETWORKX or G is None:
        out["enabled"] = False
        out["notes"].append("networkx not available or graph None.")
        return out

    n = int(G.number_of_nodes())
    e = int(G.number_of_edges())
    out.update({"n_nodes": n, "n_edges": e, "density": float(nx.density(G)) if n > 1 else 0.0})

    # reciprocity for directed graphs
    if directed:
        try:
            out["reciprocity"] = float(nx.reciprocity(G)) if e > 0 else 0.0
        except Exception:
            out["reciprocity"] = None

    # transitivity / clustering
    try:
        UG = G.to_undirected() if directed else G
        out["transitivity"] = float(nx.transitivity(UG)) if n > 2 else 0.0
        out["avg_clustering"] = float(nx.average_clustering(UG, weight="abs_weight")) if n > 2 else 0.0
    except Exception as ex:
        out["notes"].append(f"clustering/transitivity failed: {repr(ex)}")

    # assortativity (degree)
    try:
        UG = G.to_undirected() if directed else G
        out["degree_assortativity"] = float(nx.degree_assortativity_coefficient(UG)) if n > 2 else None
    except Exception:
        out["degree_assortativity"] = None

    # global efficiency (weighted)
    try:
        UG = G.to_undirected() if directed else G
        out["global_efficiency_weighted"] = float(_weighted_global_efficiency(UG))
    except Exception as ex:
        out["notes"].append(f"global efficiency failed: {repr(ex)}")

    # average shortest path length on largest connected component
    try:
        if directed:
            H = G.to_undirected()
        else:
            H = G
        if H.number_of_nodes() > 1 and H.number_of_edges() > 0:
            comps = list(nx.connected_components(H))
            largest = max(comps, key=len)
            Hs = H.subgraph(largest).copy()
            out["largest_component_size"] = int(len(largest))
            if Hs.number_of_nodes() > 1:
                out["avg_shortest_path_length_weighted_lcc"] = float(nx.average_shortest_path_length(Hs, weight="distance"))
                # diameter can be expensive; do only if small
                if Hs.number_of_nodes() <= 200:
                    out["diameter_weighted_lcc"] = float(nx.diameter(Hs, e=None)) if hasattr(nx, "diameter") else None
        else:
            out["largest_component_size"] = n
    except Exception as ex:
        out["notes"].append(f"path length metrics failed: {repr(ex)}")

    # communities + modularity on undirected
    try:
        UG = G.to_undirected() if directed else G
        comms, node_to_mod, cmeta = _communities_undirected(UG)
        if cmeta.get("ok") and len(comms) >= 1:
            out["n_modules"] = int(len(comms))
            try:
                mod = nx.algorithms.community.quality.modularity(UG, [set(c) for c in comms], weight="abs_weight")  # type: ignore
                out["modularity"] = float(mod)
            except Exception:
                out["modularity"] = None
        else:
            out["n_modules"] = None
            out["modularity"] = None
    except Exception as ex:
        out["notes"].append(f"communities/modularity failed: {repr(ex)}")

    return out


# ============================================================
# Saving outputs (numerical only)
# ============================================================
def save_tv_outputs(
    base_method_dir: Path,
    tv: TvKsResult,
    ci: Optional[BootstrapCI],
    labels: List[str],
) -> None:
    num = ensure_dir(base_method_dir / "numerical outputs")

    np.savez_compressed(
        num / "tvKS_full_arrays.npz",
        estpoints=tv.estpoints,
        bandwidth=np.array([tv.bandwidth]),
        Bhat=tv.Bhat,
        intercepts=tv.intercepts,
        alphas=tv.alphas,
        r2_node=tv.r2_node,
        partial_corr=tv.partial_corr,
        mse_point=tv.mse_point,
        ess_point=tv.ess_point,
        ci_low=(ci.B_low if ci else np.full_like(tv.Bhat, np.nan)),
        ci_high=(ci.B_high if ci else np.full_like(tv.Bhat, np.nan)),
    )

    meta = []
    m = tv.Bhat.shape[0]
    for ei in range(m):
        te = float(tv.estpoints[ei])
        fnB = f"Bhat_time_{ei:02d}_t_{te:.3f}.csv"
        fnP = f"partialcorr_time_{ei:02d}_t_{te:.3f}.csv"
        save_matrix_csv(num / fnB, tv.Bhat[ei], labels)
        save_matrix_csv(num / fnP, tv.partial_corr[ei], labels)

        entry = {"time_index": ei, "t": te, "Bhat_csv": fnB, "partialcorr_csv": fnP}
        if ci is not None:
            fnL = f"CI_low_time_{ei:02d}_t_{te:.3f}.csv"
            fnH = f"CI_high_time_{ei:02d}_t_{te:.3f}.csv"
            save_matrix_csv(num / fnL, ci.B_low[ei], labels)
            save_matrix_csv(num / fnH, ci.B_high[ei], labels)
            entry["ci_low_csv"] = fnL
            entry["ci_high_csv"] = fnH
        meta.append(entry)

    (num / "time_index_map.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def save_stationary_outputs(base_method_dir: Path, st: StationaryGvarResult, labels: List[str]) -> None:
    num = ensure_dir(base_method_dir / "numerical outputs")
    save_matrix_csv(num / "B_stationary.csv", st.B, labels)
    save_matrix_csv(num / "residual_cov.csv", st.residual_cov, labels)
    save_matrix_csv(num / "partial_corr.csv", st.partial_corr, labels)

    np.savez_compressed(
        num / "stationary_full_arrays.npz",
        B=st.B,
        intercept=st.intercept,
        alphas=st.alphas,
        r2_node=st.r2_node,
        residual_cov=st.residual_cov,
        partial_corr=st.partial_corr,
    )


def save_corr_outputs(base_method_dir: Path, cr: CorrResult, labels: List[str]) -> None:
    num = ensure_dir(base_method_dir / "numerical outputs")
    save_matrix_csv(num / "correlation_pearson.csv", cr.pearson, labels)
    save_matrix_csv(num / "correlation_spearman.csv", cr.spearman, labels)
    save_matrix_csv(num / "partial_corr_ledoitwolf.csv", cr.partial_corr_lw, labels)

    np.savez_compressed(
        num / "correlation_full_arrays.npz",
        pearson=cr.pearson,
        spearman=cr.spearman,
        partial_corr_lw=cr.partial_corr_lw,
    )


def _write_empty_csv(path: Path, columns: List[str]) -> None:
    pd.DataFrame(columns=columns).to_csv(path, index=False)


# ============================================================
# Per-profile runner (readiness-driven)
# ============================================================
def analyze_one_profile_from_readiness(
    readiness_path: Path,
    input_root: Path,
    output_root: Path,
    data_filename: str,
    metadata_filename: str,
    seed: int,
    verbose: bool,
    pattern: Optional[str],
    execution_policy: str,
) -> None:
    readiness = _read_json(readiness_path)
    profile_id = str(readiness.get("meta", {}).get("profile_id", readiness_path.parent.name))
    if pattern and (pattern not in profile_id):
        return

    raw_csv_str = readiness.get("meta", {}).get("input_file")
    raw_csv_path = Path(raw_csv_str) if raw_csv_str else (Path(input_root) / profile_id / data_filename)
    if not raw_csv_path.exists():
        raw_csv_path = Path(input_root) / profile_id / data_filename
    if not raw_csv_path.exists():
        raise FileNotFoundError(f"{profile_id}: raw CSV not found: tried {raw_csv_str} and {raw_csv_path}")

    meta_csv_path = Path(input_root) / profile_id / metadata_filename

    log(f"========== PROFILE: {profile_id} ==========")
    log(f"Readiness: {readiness_path}")
    log(f"Raw CSV:   {raw_csv_path}")

    df_raw = _read_csv_robust(raw_csv_path)
    df_raw.columns = [str(c).strip() for c in df_raw.columns]

    execution_plan = build_execution_plan_from_readiness(readiness, execution_policy=execution_policy)
    log(
        f"Execution plan: set={execution_plan['analysis_set']} "
        f"(tv={execution_plan['run_tv_gvar']}, stationary={execution_plan['run_stationary_gvar']}, "
        f"corr={execution_plan['run_correlation_baseline']}, desc_only={execution_plan['run_descriptives_only']})"
    )

    time_col, date_col = _infer_time_cols_from_readiness(readiness, df_raw)
    df = _sort_df_by_time(df_raw, time_col=time_col, date_col=date_col)

    vars_kept, sel_info = select_network_variables(readiness, df)
    if len(vars_kept) < 2 and not execution_plan["run_descriptives_only"]:
        raise ValueError(f"{profile_id}: <2 variables selected for network modeling after filtering.")

    labels_map = load_variable_labels(meta_csv_path)
    labels_map = {str(k): str(v) for k, v in labels_map.items()}

    roles = infer_roles_predictor_criterion(meta_csv_path, vars_kept)
    predictors = roles.get("predictors", [])
    criteria = roles.get("criteria", [])

    # indices for role-based summaries
    label_list = [str(v) for v in vars_kept]
    pred_idx = [label_list.index(v) for v in predictors if v in label_list]
    crit_idx = [label_list.index(v) for v in criteria if v in label_list]

    # data matrices
    t_norm = compute_t_norm(df, time_col=time_col, date_col=date_col)
    if vars_kept:
        X_obs = df[vars_kept].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    else:
        X_obs = np.empty((len(df), 0), dtype=float)

    n, p = X_obs.shape
    miss = float(np.mean(~np.isfinite(X_obs))) if p > 0 else 1.0
    log(f"Rows={n}, Vars={p}, Missing={miss:.3%}")
    log(f"Selected vars: {label_list[:12]}{' ...' if len(label_list) > 12 else ''}")

    if p >= 2 and not execution_plan["run_descriptives_only"]:
        X_imp_raw, kmodel, kdiag = kalman_impute_multivariate(
            X_obs=X_obs,
            seed=int(seed),
            n_refine=int(CFG.kalman_refine),
            ridge_alpha=float(CFG.kalman_ridge_alpha),
            meas_noise_scale=float(CFG.kalman_meas_noise_scale),
            stabilize_rho=float(CFG.kalman_stabilize_rho),
            kalman_ridge=float(CFG.kalman_ridge),
            verbose=bool(verbose),
        )
        X_imp_z = zscore(X_imp_raw)
    else:
        X_imp_raw = X_obs.copy()
        X_imp_z = X_obs.copy()
        kmodel = KalmanImputeModel(
            A=np.zeros((max(1, p), max(1, p)), dtype=float),
            Q=np.eye(max(1, p), dtype=float),
            R=np.eye(max(1, p), dtype=float),
            mu=np.zeros(max(1, p), dtype=float),
        )
        kdiag = {
            "enabled": False,
            "reason": "Kalman imputation skipped due to descriptive-only plan or <2 selected variables.",
            "n_rows": int(n),
            "n_vars": int(p),
        }

    # ---- output dirs
    prof_out = ensure_dir(Path(output_root) / profile_id)
    data_out = ensure_dir(prof_out / "data")
    m1 = ensure_dir(prof_out / "method 1")
    m2 = ensure_dir(prof_out / "method 2")
    m3 = ensure_dir(prof_out / "method 3")
    net_out = ensure_dir(prof_out / "network_metrics")

    # ---- save data artifacts
    shutil.copy2(raw_csv_path, data_out / "raw_wide_copy.csv")

    # save selected variable info
    (data_out / "selected_vars.json").write_text(json.dumps(_jsonify(sel_info), indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame({"variable": vars_kept}).to_csv(data_out / "selected_vars.csv", index=False)

    pd.DataFrame(X_obs, columns=label_list).to_csv(data_out / "X_obs_selected.csv", index=False)
    pd.DataFrame(X_imp_raw, columns=label_list).to_csv(data_out / "X_imputed_raw.csv", index=False)
    pd.DataFrame(X_imp_z, columns=label_list).to_csv(data_out / "X_imputed_zscored.csv", index=False)
    pd.DataFrame({"t_norm": t_norm}).to_csv(data_out / "t_norm.csv", index=False)

    (data_out / "variables_labels.json").write_text(json.dumps(labels_map, indent=2, ensure_ascii=False), encoding="utf-8")
    (data_out / "roles_predictor_criterion.json").write_text(json.dumps(_jsonify(roles), indent=2, ensure_ascii=False), encoding="utf-8")
    (data_out / "kalman_model.json").write_text(
        json.dumps(
            _jsonify({"A": kmodel.A, "Q": kmodel.Q, "R": kmodel.R, "mu": kmodel.mu, "labels": label_list}),
            indent=2,
        ),
        encoding="utf-8",
    )
    (data_out / "kalman_diagnostics.json").write_text(json.dumps(_jsonify(kdiag), indent=2), encoding="utf-8")

    # ---- method execution switches
    run_tv = bool(execution_plan.get("run_tv_gvar", False)) and p >= 2
    run_stationary = bool(execution_plan.get("run_stationary_gvar", False)) and p >= 2
    run_corr = bool(execution_plan.get("run_correlation_baseline", False)) and p >= 2
    if p < 2:
        execution_plan["notes"] = execution_plan.get("notes", []) + [
            "Method execution disabled because <2 selected variables remained after readiness filtering.",
        ]
    execution_plan["run_tv_gvar"] = run_tv
    execution_plan["run_stationary_gvar"] = run_stationary
    execution_plan["run_correlation_baseline"] = run_corr
    execution_plan["can_compute_momentary_impact"] = bool(run_tv or run_stationary)

    alpha_grid = np.logspace(float(CFG.alpha_min_exp), float(CFG.alpha_max_exp), int(CFG.alpha_num))
    bw_grid = np.array([float(x) for x in CFG.bw_grid], dtype=float)

    num1 = ensure_dir(m1 / "numerical outputs")
    num2 = ensure_dir(m2 / "numerical outputs")
    num3 = ensure_dir(m3 / "numerical outputs")

    method_status = {"tv": "skipped", "stationary": "skipped", "corr": "skipped"}
    tv: Optional[TvResult] = None
    st: Optional[StationaryGvarResult] = None
    cr: Optional[CorrResult] = None
    ci_obj: Optional[BootstrapCI] = None
    cl: Optional[TvClusterResult] = None
    cp: Dict[str, Any] = {"change_points": [], "network_dist": [], "z_scores": [], "threshold_z": float(CFG.change_point_z)}
    cv_summary: Optional[TvCvSummary] = None
    tv_pred: Dict[str, Any] = {"enabled": False, "reason": "tv method not executed."}
    st_pred: Dict[str, Any] = {"enabled": False, "reason": "stationary method not executed."}
    alpha_sel_eff = float("nan")
    boot_eff = 0
    block_len_eff = 0
    idx_map = np.array([], dtype=int)

    if run_tv:
        method_status["tv"] = "executed"
        log("[tvKS] Hyperparameter CV: bandwidth + alpha")
        estpoints_cv = np.linspace(0.0, 1.0, int(CFG.cv_m))
        alpha_fixed_for_bw = float(np.median(alpha_grid))

        bw_sel, bw_scores = select_bandwidth_tv_cv(
            X_raw=X_imp_raw,
            t_norm=t_norm,
            estpoints_cv=estpoints_cv,
            bw_grid=bw_grid,
            alpha_fixed=alpha_fixed_for_bw,
            n_splits=int(CFG.cv_splits),
            seed=int(seed),
            standardize_in_cv=bool(CFG.standardize_in_cv),
            ridge_precision=float(CFG.ridge_precision),
            verbose=bool(verbose),
        )

        alpha_sel, alpha_scores = select_alpha_tv_cv(
            X_raw=X_imp_raw,
            t_norm=t_norm,
            estpoints_cv=estpoints_cv,
            bandwidth_fixed=float(bw_sel),
            alpha_grid=alpha_grid,
            n_splits=int(CFG.cv_splits),
            seed=int(seed),
            standardize_in_cv=bool(CFG.standardize_in_cv),
            ridge_precision=float(CFG.ridge_precision),
            verbose=bool(verbose),
        )
        alpha_sel_eff = float(max(alpha_sel, float(CFG.alpha_floor)))
        cv_summary = TvCvSummary(
            n_splits=int(CFG.cv_splits),
            standardize_in_cv=bool(CFG.standardize_in_cv),
            alpha_fixed_for_bw=float(alpha_fixed_for_bw),
            bw_grid=[float(x) for x in bw_grid.tolist()],
            bw_cv_mse=[float(x) for x in bw_scores],
            bw_selected=float(bw_sel),
            alpha_grid=[float(x) for x in alpha_grid.tolist()],
            alpha_cv_mse=[float(x) for x in alpha_scores],
            alpha_selected=float(alpha_sel_eff),
        )
        (m1 / "tv_hyperparam_cv.json").write_text(json.dumps(_jsonify(cv_summary.__dict__), indent=2), encoding="utf-8")

        log("[tvKS] Final fit on z-scored imputed data")
        estpoints = np.linspace(0.0, 1.0, int(CFG.m))
        alphas_node = np.full(p, float(alpha_sel_eff), dtype=float)
        tv = estimate_tvKS(
            X=X_imp_z,
            t_norm=t_norm,
            estpoints=estpoints,
            bandwidth=float(bw_sel),
            alphas_node=alphas_node,
            seed=int(seed),
            ridge_precision=float(CFG.ridge_precision),
            verbose=False,
        )

        n_obs = X_imp_z.shape[0] - 1
        boot_eff = int(max(0, CFG.boot))
        if n_obs < 10:
            boot_eff = 0
        block_len_eff = int(min(max(2, int(CFG.block_len)), max(2, n_obs)))
        if boot_eff > 0:
            log(f"[tvKS] Bootstrap CI: n_boot={boot_eff}, block_len={block_len_eff}, jobs={CFG.jobs}")
            ci_obj = bootstrap_tvKS_CI(
                X=X_imp_z,
                t_norm=t_norm,
                tv_fit=tv,
                alphas_node=alphas_node,
                n_boot=int(boot_eff),
                block_len=int(block_len_eff),
                ci=(float(CFG.ci_low), float(CFG.ci_high)),
                seed=int(seed),
                n_jobs=max(1, int(CFG.jobs)),
                verbose=bool(verbose),
            )
        else:
            log("[tvKS] Bootstrap CI skipped (boot=0 or too few observations).")

        save_tv_outputs(m1, tv, ci_obj, labels=label_list)
        st_for_clusters = fit_stationary_gvar(X_imp_z, alpha_grid=alpha_grid, seed=int(seed), verbose=False)
        if st is None:
            st = st_for_clusters
        cl = detect_tv_clusters_by_error_ratio(
            X_imp_z,
            t_norm,
            tv,
            st_for_clusters,
            ratio_threshold=float(CFG.ratio_threshold),
        )
        (num1 / "tv_cluster_detection.json").write_text(
            json.dumps(
                _jsonify(
                    {
                        "create_clusters": cl.create_clusters,
                        "threshold_ratio": cl.threshold_ratio,
                        "peaks": cl.peaks,
                        "segments": cl.segments,
                        "ratio_stationary_over_tv": cl.ratio_stationary_over_tv,
                    }
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        cp = detect_change_points_by_network_distance(tv.Bhat, z_thresh=float(CFG.change_point_z))
        (num1 / "tv_network_change_points.json").write_text(json.dumps(_jsonify(cp), indent=2), encoding="utf-8")

        log("[tvKS] Edge shift detection (top changing edges)")
        edge_shifts = detect_top_edge_shifts(
            Bhat=tv.Bhat,
            estpoints=tv.estpoints,
            labels=label_list,
            top_k=int(CFG.top_k_edge_shifts),
            exclude_self=bool(CFG.exclude_self_loops),
        )
        edge_shifts.to_csv(num1 / "edge_shifts_top.csv", index=False)

        Y_tv, Yhat_tv, idx_map = one_step_prediction_tv(X_imp_z, t_norm, tv)
        tv_pred = prediction_metrics(Y_tv, Yhat_tv)
        pd.DataFrame(Y_tv, columns=label_list).to_csv(num1 / "tv_one_step_Y.csv", index=False)
        pd.DataFrame(Yhat_tv, columns=label_list).to_csv(num1 / "tv_one_step_Yhat.csv", index=False)

        log("[tvKS] Save long-format temporal edges")
        rows = []
        if ci_obj is not None:
            for ei in range(len(tv.estpoints)):
                te = float(tv.estpoints[ei])
                for dst in range(p):
                    for src in range(p):
                        rows.append(
                            {
                                "time_index": ei,
                                "t": te,
                                "dst": label_list[dst],
                                "src": label_list[src],
                                "beta": float(tv.Bhat[ei, dst, src]),
                                "ci_low": float(ci_obj.B_low[ei, dst, src]),
                                "ci_high": float(ci_obj.B_high[ei, dst, src]),
                            }
                        )
            pd.DataFrame(rows).to_csv(num1 / "tv_temporal_long_with_CI.csv", index=False)
        else:
            for ei in range(len(tv.estpoints)):
                te = float(tv.estpoints[ei])
                for dst in range(p):
                    for src in range(p):
                        rows.append(
                            {
                                "time_index": ei,
                                "t": te,
                                "dst": label_list[dst],
                                "src": label_list[src],
                                "beta": float(tv.Bhat[ei, dst, src]),
                            }
                        )
            pd.DataFrame(rows).to_csv(num1 / "tv_temporal_long.csv", index=False)
    else:
        (num1 / "tv_hyperparam_cv.json").write_text(
            json.dumps(
                {"skipped": True, "reason": f"tv method not selected by execution plan ({execution_plan['analysis_set']})."},
                indent=2,
            ),
            encoding="utf-8",
        )
        (num1 / "tv_cluster_detection.json").write_text(json.dumps({"skipped": True}, indent=2), encoding="utf-8")
        (num1 / "tv_network_change_points.json").write_text(json.dumps({"skipped": True}, indent=2), encoding="utf-8")
        _write_empty_csv(num1 / "edge_shifts_top.csv", ["time_index", "t", "dst", "src", "abs_change", "peak_time", "direction"])

    if run_stationary:
        method_status["stationary"] = "executed"
        if st is None:
            log("[stationary] Fit stationary gVAR")
            st = fit_stationary_gvar(X_imp_z, alpha_grid=alpha_grid, seed=int(seed), verbose=False)
        else:
            log("[stationary] Reusing stationary gVAR fit from tv cluster diagnostics")
        save_stationary_outputs(m2, st, labels=label_list)
        Y_st, Yhat_st = one_step_prediction_stationary(X_imp_z, st)
        st_pred = prediction_metrics(Y_st, Yhat_st)
        pd.DataFrame(Y_st, columns=label_list).to_csv(num2 / "stationary_one_step_Y.csv", index=False)
        pd.DataFrame(Yhat_st, columns=label_list).to_csv(num2 / "stationary_one_step_Yhat.csv", index=False)
    else:
        (num2 / "metrics.json").write_text(
            json.dumps(
                {"skipped": True, "reason": f"stationary method not selected by execution plan ({execution_plan['analysis_set']})."},
                indent=2,
            ),
            encoding="utf-8",
        )

    if run_corr:
        method_status["corr"] = "executed"
        log("[corr] Compute Pearson/Spearman + partial corr baseline")
        cr = compute_corr_baselines(X_imp_z)
        save_corr_outputs(m3, cr, labels=label_list)
        corr_metrics = {
            "avg_abs_corr_offdiag": float(np.mean(np.abs(cr.pearson[~np.eye(p, dtype=bool)]))),
            "max_abs_corr_offdiag": float(np.max(np.abs(cr.pearson[~np.eye(p, dtype=bool)]))),
        }
        (num3 / "metrics.json").write_text(json.dumps(_jsonify(corr_metrics), indent=2), encoding="utf-8")
    else:
        (num3 / "metrics.json").write_text(
            json.dumps(
                {"skipped": True, "reason": f"correlation baseline not selected by execution plan ({execution_plan['analysis_set']})."},
                indent=2,
            ),
            encoding="utf-8",
        )

    # ---- importance (predictor -> criterion)
    log("[importance] Predictor->criterion importance (respecting execution plan)")
    if not pred_idx:
        pred_idx = list(range(p)) if p > 0 else []
    crit_idx_eff = crit_idx if crit_idx else None

    if run_tv and tv is not None:
        tv_coef_imp = coefficient_based_importance_tv(tv.Bhat, label_list, pred_idx, criteria_idx=crit_idx_eff)
        tv_loo = predictor_importance_leave_one_out_tv(X_imp_z, t_norm, tv, idx_map, pred_idx, criteria_idx=crit_idx_eff)
        tv_loo["predictor"] = tv_loo["predictor_index"].map(lambda j: label_list[int(j)])
        tv_imp = tv_coef_imp.merge(tv_loo[["predictor", "delta_mse_all", "delta_mse_criteria"]], on="predictor", how="left")
        tv_imp.to_csv(net_out / "predictor_importance_tv.csv", index=False)
        if crit_idx:
            cd = criterion_dependence_tv(tv.Bhat, label_list, pred_idx, criteria_idx=crit_idx)
            cd.to_csv(net_out / "criterion_dependence_tv.csv", index=False)
        else:
            _write_empty_csv(net_out / "criterion_dependence_tv.csv", CRITERION_DEPENDENCE_COLUMNS)
    else:
        _write_empty_csv(net_out / "predictor_importance_tv.csv", PREDICTOR_IMPORTANCE_COLUMNS)
        _write_empty_csv(net_out / "criterion_dependence_tv.csv", CRITERION_DEPENDENCE_COLUMNS)

    if run_stationary and st is not None:
        st_coef_imp = coefficient_based_importance(st.B, label_list, pred_idx, criteria_idx=crit_idx_eff)
        st_loo = predictor_importance_leave_one_out_stationary(X_imp_z, st, pred_idx, criteria_idx=crit_idx_eff)
        st_loo["predictor"] = st_loo["predictor_index"].map(lambda j: label_list[int(j)])
        st_imp = st_coef_imp.merge(st_loo[["predictor", "delta_mse_all", "delta_mse_criteria"]], on="predictor", how="left")
        st_imp.to_csv(net_out / "predictor_importance_stationary.csv", index=False)
    else:
        _write_empty_csv(net_out / "predictor_importance_stationary.csv", PREDICTOR_IMPORTANCE_COLUMNS)

    # ---- multicollinearity
    log("[multicollinearity] Computing VIF/condition/correlation flags")
    if p >= 2 and X_imp_z.shape[0] >= 3:
        Z = X_imp_z[:-1]
        mc_all = multicollinearity_report(Z, label_list, ridge=float(CFG.vif_ridge), corr_thresholds=CFG.high_corr_thresholds)
        if pred_idx:
            mc_pred = multicollinearity_report(Z[:, pred_idx], [label_list[i] for i in pred_idx], ridge=float(CFG.vif_ridge), corr_thresholds=CFG.high_corr_thresholds)
        else:
            mc_pred = {"enabled": False, "notes": ["no predictors inferred"]}
        if crit_idx:
            mc_crit = multicollinearity_report(Z[:, crit_idx], [label_list[i] for i in crit_idx], ridge=float(CFG.vif_ridge), corr_thresholds=CFG.high_corr_thresholds)
        else:
            mc_crit = {"enabled": False, "notes": ["no criteria inferred"]}
    else:
        mc_all = {"enabled": False, "notes": ["insufficient rows/variables for multicollinearity diagnostics"]}
        mc_pred = {"enabled": False, "notes": ["insufficient rows/variables for predictor multicollinearity diagnostics"]}
        mc_crit = {"enabled": False, "notes": ["insufficient rows/variables for criterion multicollinearity diagnostics"]}
    (net_out / "multicollinearity_all.json").write_text(json.dumps(_jsonify(mc_all), indent=2), encoding="utf-8")
    (net_out / "multicollinearity_predictors.json").write_text(json.dumps(_jsonify(mc_pred), indent=2), encoding="utf-8")
    (net_out / "multicollinearity_criteria.json").write_text(json.dumps(_jsonify(mc_crit), indent=2), encoding="utf-8")

    # ---- network centrality/global metrics
    log("[network metrics] Centrality + global metrics")
    if not HAVE_NETWORKX:
        log("  [WARN] networkx not available -> writing empty centrality/global outputs.")
        _write_empty_csv(net_out / "temporal_lagged_node_centrality.csv", TEMPORAL_NODE_CENTRALITY_COLUMNS)
        _write_empty_csv(net_out / "temporal_contemp_node_centrality.csv", TEMPORAL_NODE_CENTRALITY_COLUMNS)
        _write_empty_csv(net_out / "temporal_lagged_global_metrics.csv", ["time_index", "t", "network"])
        _write_empty_csv(net_out / "temporal_contemp_global_metrics.csv", ["time_index", "t", "network"])
        _write_empty_csv(net_out / "stationary_lagged_node_centrality.csv", TEMPORAL_NODE_CENTRALITY_COLUMNS[2:])
        _write_empty_csv(net_out / "stationary_contemp_node_centrality.csv", TEMPORAL_NODE_CENTRALITY_COLUMNS[2:])
        (net_out / "stationary_lagged_global_metrics.json").write_text(json.dumps({"enabled": False, "reason": "networkx_unavailable"}, indent=2), encoding="utf-8")
        (net_out / "stationary_contemp_global_metrics.json").write_text(json.dumps({"enabled": False, "reason": "networkx_unavailable"}, indent=2), encoding="utf-8")
    else:
        if run_tv and tv is not None:
            lagged_c_rows = []
            lagged_g_rows = []
            contemp_c_rows = []
            contemp_g_rows = []
            for ei in range(len(tv.estpoints)):
                te = float(tv.estpoints[ei])
                Gd = _build_directed_graph_from_B(
                    tv.Bhat[ei],
                    labels=label_list,
                    abs_threshold=float(CFG.edge_threshold_abs),
                    exclude_self=bool(CFG.exclude_self_loops),
                )
                cdf, _ = compute_node_centralities(Gd, directed=True)
                if not cdf.empty:
                    cdf.insert(0, "time_index", ei)
                    cdf.insert(1, "t", te)
                    lagged_c_rows.append(cdf)
                gmet = compute_global_metrics(Gd, directed=True)
                gmet.update({"time_index": ei, "t": te, "network": "lagged_directed"})
                lagged_g_rows.append(gmet)

                Gu = _build_undirected_graph_from_W(
                    tv.partial_corr[ei],
                    labels=label_list,
                    abs_threshold=float(CFG.edge_threshold_abs),
                    exclude_self=True,
                )
                cdf2, _ = compute_node_centralities(Gu, directed=False)
                if not cdf2.empty:
                    cdf2.insert(0, "time_index", ei)
                    cdf2.insert(1, "t", te)
                    contemp_c_rows.append(cdf2)
                gmet2 = compute_global_metrics(Gu, directed=False)
                gmet2.update({"time_index": ei, "t": te, "network": "contemp_undirected"})
                contemp_g_rows.append(gmet2)

            if lagged_c_rows:
                pd.concat(lagged_c_rows, ignore_index=True).to_csv(net_out / "temporal_lagged_node_centrality.csv", index=False)
            else:
                _write_empty_csv(net_out / "temporal_lagged_node_centrality.csv", TEMPORAL_NODE_CENTRALITY_COLUMNS)
            if contemp_c_rows:
                pd.concat(contemp_c_rows, ignore_index=True).to_csv(net_out / "temporal_contemp_node_centrality.csv", index=False)
            else:
                _write_empty_csv(net_out / "temporal_contemp_node_centrality.csv", TEMPORAL_NODE_CENTRALITY_COLUMNS)
            pd.DataFrame(lagged_g_rows).to_csv(net_out / "temporal_lagged_global_metrics.csv", index=False)
            pd.DataFrame(contemp_g_rows).to_csv(net_out / "temporal_contemp_global_metrics.csv", index=False)
        else:
            _write_empty_csv(net_out / "temporal_lagged_node_centrality.csv", TEMPORAL_NODE_CENTRALITY_COLUMNS)
            _write_empty_csv(net_out / "temporal_contemp_node_centrality.csv", TEMPORAL_NODE_CENTRALITY_COLUMNS)
            _write_empty_csv(net_out / "temporal_lagged_global_metrics.csv", ["time_index", "t", "network"])
            _write_empty_csv(net_out / "temporal_contemp_global_metrics.csv", ["time_index", "t", "network"])

        if run_stationary and st is not None:
            Gd_st = _build_directed_graph_from_B(st.B, label_list, abs_threshold=float(CFG.edge_threshold_abs), exclude_self=bool(CFG.exclude_self_loops))
            cdf_st, _ = compute_node_centralities(Gd_st, directed=True)
            cdf_st.to_csv(net_out / "stationary_lagged_node_centrality.csv", index=False)
            g_st = compute_global_metrics(Gd_st, directed=True)
            (net_out / "stationary_lagged_global_metrics.json").write_text(json.dumps(_jsonify(g_st), indent=2), encoding="utf-8")

            Gu_st = _build_undirected_graph_from_W(st.partial_corr, label_list, abs_threshold=float(CFG.edge_threshold_abs), exclude_self=True)
            cdf_cst, _ = compute_node_centralities(Gu_st, directed=False)
            cdf_cst.to_csv(net_out / "stationary_contemp_node_centrality.csv", index=False)
            g_cst = compute_global_metrics(Gu_st, directed=False)
            (net_out / "stationary_contemp_global_metrics.json").write_text(json.dumps(_jsonify(g_cst), indent=2), encoding="utf-8")
        else:
            _write_empty_csv(net_out / "stationary_lagged_node_centrality.csv", TEMPORAL_NODE_CENTRALITY_COLUMNS[2:])
            _write_empty_csv(net_out / "stationary_contemp_node_centrality.csv", TEMPORAL_NODE_CENTRALITY_COLUMNS[2:])
            (net_out / "stationary_lagged_global_metrics.json").write_text(
                json.dumps({"enabled": False, "reason": "stationary_not_executed"}, indent=2), encoding="utf-8"
            )
            (net_out / "stationary_contemp_global_metrics.json").write_text(
                json.dumps({"enabled": False, "reason": "stationary_not_executed"}, indent=2), encoding="utf-8"
            )

    # ---- write metrics jsons
    if run_tv and tv is not None and cv_summary is not None and cl is not None:
        m1_metrics = {
            "generated_at": _now_iso(),
            "dependencies": {"networkx": HAVE_NETWORKX, "joblib": HAVE_JOBLIB},
            "selection": sel_info,
            "execution_plan": execution_plan,
            "tv_cv": cv_summary.__dict__,
            "bandwidth_selected_final": float(tv.bandwidth),
            "alpha_selected_final": float(alpha_sel_eff),
            "bootstrap": {"n_boot": int(boot_eff), "block_len": int(block_len_eff), "ci": [float(CFG.ci_low), float(CFG.ci_high)]} if ci_obj is not None else {"n_boot": 0},
            "prediction": tv_pred,
            "tv_clusters_error_ratio": {"create_clusters": cl.create_clusters, "threshold_ratio": cl.threshold_ratio, "peaks": cl.peaks, "segments": cl.segments},
            "tv_clusters_change_points": cp,
            "debug": {
                "ess_mean": float(np.mean(tv.ess_point)),
                "spectral_radius_mean": float(np.mean([spectral_radius(tv.Bhat[i]) for i in range(len(tv.estpoints))])),
                "nonzero_fraction": float(np.mean(np.abs(tv.Bhat) > 1e-12)),
            },
        }
    else:
        m1_metrics = {
            "generated_at": _now_iso(),
            "skipped": True,
            "execution_plan": execution_plan,
            "reason": f"tv method not executed under analysis_set={execution_plan['analysis_set']}",
        }
    (num1 / "metrics.json").write_text(json.dumps(_jsonify(m1_metrics), indent=2, ensure_ascii=False), encoding="utf-8")

    if run_stationary and st is not None:
        m2_metrics = {"prediction": st_pred, "r2_node_mean": float(np.mean(st.r2_node)), "alphas": st.alphas.tolist(), "execution_plan": execution_plan}
    else:
        m2_metrics = {"skipped": True, "execution_plan": execution_plan, "reason": f"stationary method not executed under analysis_set={execution_plan['analysis_set']}"}
    (num2 / "metrics.json").write_text(json.dumps(_jsonify(m2_metrics), indent=2), encoding="utf-8")

    # ---- summary
    summary = {
        "contract_version": "1.0.0",
        "profile": profile_id,
        "base_outdir": str(prof_out),
        "n_rows": int(n),
        "n_vars": int(p),
        "variables": label_list,
        "predictors": predictors,
        "criteria": criteria,
        "execution_plan": execution_plan,
        "method_status": method_status,
        "outputs": {
            "data_dir": "data/",
            "tv_dir": "method 1/numerical outputs/",
            "stationary_dir": "method 2/numerical outputs/",
            "corr_dir": "method 3/numerical outputs/",
            "network_metrics_dir": "network_metrics/",
        },
    }
    (prof_out / "comparison_summary.json").write_text(json.dumps(_jsonify(summary), indent=2, ensure_ascii=False), encoding="utf-8")

    log(f"[DONE] {profile_id} -> {prof_out}")
    log("")


# ============================================================
# CLI
# ============================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Readiness-driven tv-gVAR + comprehensive network metrics (no visualization).")

    p.add_argument("--input-root", type=str, default=DEFAULT_INPUT_ROOT)
    p.add_argument("--readiness-root", type=str, default=DEFAULT_READINESS_ROOT)
    p.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)

    p.add_argument("--data-filename", type=str, default=DEFAULT_DATA_FILENAME)
    p.add_argument("--metadata-filename", type=str, default=DEFAULT_META_FILENAME)
    p.add_argument("--readiness-filename", type=str, default=DEFAULT_READINESS_FILENAME)

    p.add_argument("--pattern", type=str, default="", help="Optional substring filter for profile_id.")
    p.add_argument("--max-profiles", type=int, default=0)

    # selection strategy
    p.add_argument("--prefer-tier", type=str, default=CFG.prefer_tier, choices=["tier1", "overall_ready", "all_non_hard"])
    p.add_argument(
        "--execution-policy",
        type=str,
        default="readiness_aligned",
        choices=["readiness_aligned", "all_methods"],
        help="How strictly method execution follows readiness recommendations.",
    )
    p.add_argument("--min-nonmissing", type=int, default=CFG.min_nonmissing_for_network)
    p.add_argument("--max-missing-frac", type=float, default=CFG.max_missing_frac_for_network)
    p.add_argument("--allow-binary", action=argparse.BooleanOptionalAction, default=CFG.allow_binary)

    # tv controls
    p.add_argument("--m", type=int, default=CFG.m)
    p.add_argument("--cv-m", type=int, default=CFG.cv_m)
    p.add_argument("--cv-splits", type=int, default=CFG.cv_splits)
    p.add_argument("--standardize-in-cv", action=argparse.BooleanOptionalAction, default=CFG.standardize_in_cv)
    p.add_argument("--ridge-precision", type=float, default=CFG.ridge_precision)

    p.add_argument("--alpha-min-exp", type=float, default=CFG.alpha_min_exp)
    p.add_argument("--alpha-max-exp", type=float, default=CFG.alpha_max_exp)
    p.add_argument("--alpha-num", type=int, default=CFG.alpha_num)
    p.add_argument("--alpha-floor", type=float, default=CFG.alpha_floor)

    p.add_argument("--bw-grid", type=str, default=",".join([str(x) for x in CFG.bw_grid]))

    # bootstrap
    p.add_argument("--boot", type=int, default=CFG.boot)
    p.add_argument("--block-len", type=int, default=CFG.block_len)
    p.add_argument("--jobs", type=int, default=CFG.jobs)
    p.add_argument("--ci-low", type=float, default=CFG.ci_low)
    p.add_argument("--ci-high", type=float, default=CFG.ci_high)

    # regimes / shifts
    p.add_argument("--ratio-threshold", type=float, default=CFG.ratio_threshold)
    p.add_argument("--change-point-z", type=float, default=CFG.change_point_z)
    p.add_argument("--top-k-edge-shifts", type=int, default=CFG.top_k_edge_shifts)

    # network
    p.add_argument("--edge-threshold-abs", type=float, default=CFG.edge_threshold_abs)
    p.add_argument("--exclude-self-loops", action=argparse.BooleanOptionalAction, default=CFG.exclude_self_loops)

    # multicollinearity
    p.add_argument("--vif-ridge", type=float, default=CFG.vif_ridge)

    # kalman
    p.add_argument("--kalman-refine", type=int, default=CFG.kalman_refine)
    p.add_argument("--kalman-ridge-alpha", type=float, default=CFG.kalman_ridge_alpha)
    p.add_argument("--kalman-meas-noise-scale", type=float, default=CFG.kalman_meas_noise_scale)
    p.add_argument("--kalman-stabilize-rho", type=float, default=CFG.kalman_stabilize_rho)
    p.add_argument("--kalman-ridge", type=float, default=CFG.kalman_ridge)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)

    return p.parse_args()


def main() -> int:
    args = parse_args()

    # update config from CLI
    CFG.prefer_tier = str(args.prefer_tier)
    CFG.min_nonmissing_for_network = int(args.min_nonmissing)
    CFG.max_missing_frac_for_network = float(args.max_missing_frac)
    CFG.allow_binary = bool(args.allow_binary)

    CFG.m = int(args.m)
    CFG.cv_m = int(args.cv_m)
    CFG.cv_splits = int(args.cv_splits)
    CFG.standardize_in_cv = bool(args.standardize_in_cv)
    CFG.ridge_precision = float(args.ridge_precision)

    CFG.alpha_min_exp = float(args.alpha_min_exp)
    CFG.alpha_max_exp = float(args.alpha_max_exp)
    CFG.alpha_num = int(args.alpha_num)
    CFG.alpha_floor = float(args.alpha_floor)

    CFG.bw_grid = tuple(float(x.strip()) for x in str(args.bw_grid).split(",") if x.strip())

    CFG.boot = int(args.boot)
    CFG.block_len = int(args.block_len)
    CFG.jobs = int(args.jobs)
    CFG.ci_low = float(args.ci_low)
    CFG.ci_high = float(args.ci_high)

    CFG.ratio_threshold = float(args.ratio_threshold)
    CFG.change_point_z = float(args.change_point_z)
    CFG.top_k_edge_shifts = int(args.top_k_edge_shifts)

    CFG.edge_threshold_abs = float(args.edge_threshold_abs)
    CFG.exclude_self_loops = bool(args.exclude_self_loops)

    CFG.vif_ridge = float(args.vif_ridge)

    CFG.kalman_refine = int(args.kalman_refine)
    CFG.kalman_ridge_alpha = float(args.kalman_ridge_alpha)
    CFG.kalman_meas_noise_scale = float(args.kalman_meas_noise_scale)
    CFG.kalman_stabilize_rho = float(args.kalman_stabilize_rho)
    CFG.kalman_ridge = float(args.kalman_ridge)

    input_root = Path(args.input_root).expanduser()
    readiness_root = Path(args.readiness_root).expanduser()
    output_root = Path(args.output_root).expanduser()
    ensure_dir(output_root)

    pattern = args.pattern.strip() if args.pattern else None

    if not readiness_root.exists():
        raise FileNotFoundError(f"readiness-root not found: {readiness_root}")
    if not input_root.exists():
        raise FileNotFoundError(f"input-root not found: {input_root}")

    readiness_paths = discover_readiness_reports(readiness_root, args.readiness_filename)
    if args.max_profiles and args.max_profiles > 0:
        readiness_paths = readiness_paths[: int(args.max_profiles)]

    if not readiness_paths:
        raise RuntimeError(f"No readiness reports named '{args.readiness_filename}' found under: {readiness_root}")

    log("========== BATCH START ==========")
    log(f"readiness_root: {readiness_root}")
    log(f"input_root:     {input_root}")
    log(f"output_root:    {output_root}")
    log(f"profiles:       {len(readiness_paths)}")
    log(f"pattern:        {pattern!r}")
    log(f"execution:      {args.execution_policy}")
    log(f"networkx:       {HAVE_NETWORKX}")
    log(f"joblib:         {HAVE_JOBLIB}")
    log("")

    n_ok = 0
    n_fail = 0

    for i, rp in enumerate(readiness_paths):
        try:
            analyze_one_profile_from_readiness(
                readiness_path=rp,
                input_root=input_root,
                output_root=output_root,
                data_filename=args.data_filename,
                metadata_filename=args.metadata_filename,
                seed=int(args.seed) + 11 * i,
                verbose=bool(args.verbose),
                pattern=pattern,
                execution_policy=str(args.execution_policy),
            )
            n_ok += 1
        except Exception as e:
            n_fail += 1
            log(f"[ERROR] Failed on {rp}: {repr(e)}")

    log("========== BATCH COMPLETE ==========")
    log(f"Success: {n_ok}  Failed: {n_fail}")
    log(f"Output root: {output_root}")
    return 0 if n_fail == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
