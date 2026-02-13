#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_readiness_check.py

Comprehensive readiness check for idiographic (single-subject) time-series network analysis
(e.g., EMA, digital phenotyping; mixed datatypes; predictors Pxx + criteria Cxx).

Key upgrade in THIS version (per your request)
- Adds explicit support for time-varying gVAR (tv-gVAR) feasibility.
- Stationarity is treated as:
    * Important for STATIC lagged VAR/gVAR feasibility (Tier 3 static variant)
    * NOT a blocker for TIME-VARYING gVAR (Tier 3 time-varying variant),
      provided the dataset is sufficiently “ready” (enough time points, usable lagged information,
      acceptable missingness, workable time ordering/regularity, etc.).

What this script does (per pseudoprofile CSV)
- Detects time columns (t_index / date / datetime) and enforces time ordering (stable mergesort).
- Assesses sampling regularity + duplicates in time.
- Infers variable types; checks data quality:
    missingness, streaks, low variability, dominance, outliers, zero-inflation hints.
- Checks feasibility / assumptions:
    effective N for lagged models (strict multivariate AND per-variable lagged N; q25 used),
    stationarity signals (ADF/KPSS; warnings suppressed but recorded),
    collinearity risk (robust correlation method selection).
- Chooses highest feasible analysis tier + Tier 3 variant:
    Tier 3: Lagged dynamic network (gVAR feasibility heuristic)
        - Variant A: STATIC_gVAR  (stationarity helpful/expected)
        - Variant B: TIME_VARYING_gVAR (stationarity NOT required; higher N requirements)
    Tier 2: Contemporaneous partial correlation / GGM feasibility heuristic
    Tier 1: Correlation matrix only (robust selection)
    Tier 0: Descriptives only
- Produces a single readiness score (0–100) + auditable breakdown.
- Produces BOTH explanation fields:
    (1) technical_summary (for developers / analysts)
    (2) client_friendly_summary (for end-user UI)

Outputs (per profile)
- <output_root>/<profile_id>/readiness_report.json
- <output_root>/<profile_id>/readiness_summary.txt

LLM finalization (ON by default in THIS version, per your request)
- Sends a condensed payload to GPT-5-nano to:
    - optionally override tier/variant (rare; must cite evidence)
    - adjust score within [-10, +10]
    - improve technical + client summaries + next steps + caveats
- If LLM fails, pipeline continues and stores error details in JSON.
- Uses OPENAI_API_KEY from environment.

Usage
    python apply_readiness_check.py
    python apply_readiness_check.py --llm-finalize False
    python apply_readiness_check.py --max-profiles 3
    python apply_readiness_check.py --prefer-time-varying True

Dependencies
- Required: pandas, numpy
- Optional: scipy, statsmodels
- Optional (LLM step): openai (new client preferred; legacy fallback included)

Author: generated; tune thresholds + weights for your domain.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# optional .env support (safe if missing)
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

# -----------------------------
# Optional dependencies
# -----------------------------
HAVE_SCIPY = False
HAVE_STATSMODELS = False
HAVE_OPENAI_NEW = False
HAVE_OPENAI_LEGACY = False

try:
    import scipy.stats as sps  # type: ignore

    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# statsmodels
try:
    from statsmodels.stats.diagnostic import acorr_ljungbox  # type: ignore
    from statsmodels.tsa.stattools import adfuller, kpss  # type: ignore

    try:
        from statsmodels.tools.sm_exceptions import InterpolationWarning  # type: ignore
    except Exception:
        InterpolationWarning = None  # type: ignore

    HAVE_STATSMODELS = True
except Exception:
    HAVE_STATSMODELS = False
    InterpolationWarning = None  # type: ignore

# openai (new)
try:
    from openai import OpenAI  # type: ignore

    HAVE_OPENAI_NEW = True
except Exception:
    HAVE_OPENAI_NEW = False

# openai (legacy fallback)
try:
    import openai  # type: ignore

    HAVE_OPENAI_LEGACY = True
except Exception:
    HAVE_OPENAI_LEGACY = False


# -----------------------------
# Defaults / paths
# -----------------------------
DEFAULT_INPUT_ROOT = (
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/Evaluation/"
    "01_pseudoprofile(s)/time_series_data/pseudodata"
)
DEFAULT_OUTPUT_ROOT = (
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/Evaluation/"
    "04_initial_observation_analysis/00_readiness_check"
)
DEFAULT_FILENAME = "pseudodata_wide.csv"


# -----------------------------
# Thresholds & weights
# -----------------------------
@dataclass
class Thresholds:
    # Missingness (per variable)
    hard_drop_missing_pct: float = 0.70
    soft_drop_missing_pct: float = 0.40

    # Variance / near-constant
    hard_drop_unique_values_max: int = 1
    soft_drop_unique_values_max: int = 2
    near_constant_std_epsilon: float = 1e-8
    dominance_soft_drop_pct: float = 0.92  # one value dominates

    # Rare events (binary)
    binary_rare_event_min_pct: float = 0.05  # min(p, 1-p) below this => unstable edges likely

    # Outliers (robust z using MAD)
    robust_z_outlier_threshold: float = 4.0
    outlier_soft_drop_pct: float = 0.20
    outlier_hard_drop_pct: float = 0.35

    # Time regularity
    irregular_time_soft_threshold: float = 0.15
    irregular_time_hard_threshold: float = 0.35
    duplicate_time_soft_threshold: float = 0.02
    duplicate_time_hard_threshold: float = 0.10

    # Stationarity test alphas
    adf_alpha: float = 0.05
    kpss_alpha: float = 0.05

    # Collinearity
    high_corr_threshold: float = 0.95
    moderate_corr_threshold: float = 0.85

    # Tier feasibility: effective sample size thresholds (STATIC Tier 3)
    tier3_static_min_n_eff_base: int = 50
    tier3_static_min_n_eff_per_var: int = 8
    tier3_static_min_n_eff_q25_floor: int = 35
    max_var_to_n_ratio_tier3_static: float = 0.12

    # Tier 3 TIME-VARYING gVAR (more demanding overall)
    # We approximate feasibility as: enough usable lagged information to support smoothing/windows.
    tier3_tv_min_n_eff_q25_floor: int = 60
    tier3_tv_min_windows: int = 3
    tier3_tv_window_min: int = 30
    tier3_tv_window_per_var: int = 6  # window >= 6*k is a rough stabilizer
    max_var_to_n_ratio_tier3_tv: float = 0.08

    # Tier 2 / Tier 1
    tier2_min_pairwise_n_q25: int = 20
    tier2_min_pairwise_n_median: int = 25
    tier1_min_pairwise_n_q25: int = 10
    tier1_min_pairwise_n_median: int = 12

    # Practical maximum variables for given N
    max_var_to_n_ratio_tier2: float = 0.25

    # Minimum variables to attempt tiers
    tier3_min_k: int = 3
    tier2_min_k: int = 3
    tier1_min_k: int = 2


THR = Thresholds()


@dataclass
class ScoreWeights:
    """
    Interpretable weights for the base score (0–100). Tier/variant choice is a separate decision.
    """
    sample_size: float = 0.30
    missingness: float = 0.25
    variable_quality: float = 0.25
    time_regular: float = 0.10
    model_assumptions: float = 0.10  # collinearity/duplicates + (stationarity only when relevant)


W = ScoreWeights()


# -----------------------------
# LLM settings (fixed; no CLI args)
# -----------------------------
LLM_MODEL = "gpt-5-nano"


# -----------------------------
# Utility helpers
# -----------------------------
def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _score_clip(x: float) -> float:
    return float(max(0.0, min(100.0, x)))


def _bool_from_str(x: str) -> bool:
    return str(x).strip().lower() in {"1", "true", "t", "yes", "y"}


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (float, int, np.floating, np.integer)):
            if np.isnan(x):
                return None
            return float(x)
        return float(x)
    except Exception:
        return None


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (pd.Timedelta,)):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    if dataclasses.is_dataclass(obj):
        return _jsonify(dataclasses.asdict(obj))
    return obj


def _label_from_score(score: float) -> str:
    if score < 20:
        return "NotReady"
    if score < 40:
        return "PartiallyReady_Low"
    if score < 60:
        return "PartiallyReady_High"
    if score < 75:
        return "Ready_Low"
    if score < 88:
        return "Ready_High"
    return "FullyReady"


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


def _infer_time_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    cols = [c for c in df.columns]
    lower = {c: str(c).strip().lower() for c in cols}

    time_candidates = [c for c in cols if lower[c] in {"t_index", "t", "time_index", "timeidx", "index"}]
    date_candidates = [c for c in cols if lower[c] in {"date", "datetime", "timestamp", "time"}]

    time_col = time_candidates[0] if time_candidates else None
    date_col = date_candidates[0] if date_candidates else None

    # if no explicit date col, find parseable
    if date_col is None:
        for c in cols:
            if c == time_col:
                continue
            if df[c].dtype == object or "date" in lower[c] or "time" in lower[c]:
                try:
                    parsed = pd.to_datetime(df[c], errors="coerce")
                    if parsed.notna().mean() > 0.80:
                        date_col = c
                        break
                except Exception:
                    continue

    # if no explicit numeric time col, find mostly-numeric monotone
    if time_col is None:
        for c in cols:
            if c == date_col:
                continue
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().mean() > 0.95:
                diffs = s.dropna().diff()
                if diffs.empty or (diffs >= 0).mean() > 0.98:
                    time_col = c
                    break

    return time_col, date_col


def _sort_by_time(df: pd.DataFrame, time_col: Optional[str], date_col: Optional[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Returns a time-sorted copy (if possible) and ordering info.
    Stable mergesort to preserve relative order for ties.
    """
    info = {"sorted": False, "sort_key": None, "time_parse_ok_fraction": None, "notes": []}

    if date_col is not None:
        t = pd.to_datetime(df[date_col], errors="coerce")
        ok = float(t.notna().mean())
        info["time_parse_ok_fraction"] = ok
        if ok >= 0.80:
            out = df.copy()
            out["_tmp_datetime_sort_key_"] = t
            out = out.sort_values("_tmp_datetime_sort_key_", kind="mergesort").drop(columns=["_tmp_datetime_sort_key_"])
            info["sorted"] = True
            info["sort_key"] = date_col
            return out, info
        info["notes"].append("Date column present but parsing success <80%; not sorting by date.")

    if time_col is not None:
        t = pd.to_numeric(df[time_col], errors="coerce")
        ok = float(t.notna().mean())
        info["time_parse_ok_fraction"] = ok
        if ok >= 0.80:
            out = df.copy()
            out["_tmp_time_sort_key_"] = t
            out = out.sort_values("_tmp_time_sort_key_", kind="mergesort").drop(columns=["_tmp_time_sort_key_"])
            info["sorted"] = True
            info["sort_key"] = time_col
            return out, info
        info["notes"].append("Numeric time column present but coercion success <80%; not sorting by time.")

    info["notes"].append("No reliable time column detected; keeping original row order.")
    return df.copy(), info


def _coerce_numeric_series(s: pd.Series) -> Tuple[pd.Series, Dict[str, Any]]:
    info: Dict[str, Any] = {"original_dtype": str(s.dtype), "coerced": False, "non_numeric_rate": None}
    if pd.api.types.is_numeric_dtype(s):
        out = s.astype(float)
        out = out.replace([np.inf, -np.inf], np.nan)
        return out, info
    coerced = pd.to_numeric(s, errors="coerce").astype(float)
    coerced = coerced.replace([np.inf, -np.inf], np.nan)
    non_num_rate = coerced.isna().mean() - s.isna().mean()
    info["coerced"] = True
    info["non_numeric_rate"] = float(max(non_num_rate, 0.0))
    return coerced, info


def _infer_variable_type(s: pd.Series) -> str:
    s2 = s.dropna()
    if len(s2) == 0:
        return "unknown"
    if not pd.api.types.is_numeric_dtype(s2):
        return "categorical"

    x = s2.to_numpy(dtype=float)
    uniq = np.unique(x)

    if uniq.size <= 2 and set(np.round(uniq, 6)).issubset({0.0, 1.0}):
        return "binary"
    if np.min(x) >= 0.0 and np.max(x) <= 1.0 and uniq.size > 2:
        return "proportion"

    is_int_like = np.mean(np.isclose(x, np.round(x))) > 0.98
    if is_int_like:
        if uniq.size <= 7:
            return "ordinal"
        if np.min(x) >= 0.0:
            return "count"

    return "continuous"


def _dominance_pct(s: pd.Series) -> float:
    s2 = s.dropna()
    if len(s2) == 0:
        return 1.0
    vc = s2.value_counts(dropna=True)
    if vc.empty:
        return 1.0
    return float(vc.iloc[0] / len(s2))


def _mad(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def _robust_outlier_pct(s: pd.Series, z_thresh: float) -> Optional[float]:
    x = s.dropna().to_numpy(dtype=float)
    if x.size < 8:
        return None
    mad = _mad(x)
    if mad <= 1e-12:
        return 0.0
    z = 0.6745 * (x - np.median(x)) / mad
    return float(np.mean(np.abs(z) >= z_thresh))


def _longest_missing_streak(s: pd.Series) -> int:
    m = s.isna().to_numpy(dtype=bool)
    best = 0
    cur = 0
    for v in m:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def _normality_tests(x: np.ndarray) -> Dict[str, Any]:
    out: Dict[str, Any] = {"jarque_bera_p": None, "shapiro_p": None, "skew": None, "kurtosis": None}
    if x.size < 8 or not HAVE_SCIPY:
        return out
    try:
        jb = sps.jarque_bera(x)
        out["jarque_bera_p"] = float(jb.pvalue)
    except Exception:
        pass
    try:
        if x.size <= 5000:
            sh = sps.shapiro(x)
            out["shapiro_p"] = float(sh.pvalue)
    except Exception:
        pass
    try:
        out["skew"] = float(sps.skew(x, nan_policy="omit"))
        out["kurtosis"] = float(sps.kurtosis(x, nan_policy="omit", fisher=True))
    except Exception:
        pass
    return out


def _stationarity_tests(s: pd.Series) -> Dict[str, Any]:
    """
    ADF: H0 = unit root (non-stationary) -> want p < alpha
    KPSS: H0 = stationary -> want p >= alpha
    Suppresses statsmodels InterpolationWarning spam; records it in notes.
    """
    out: Dict[str, Any] = {"adf_p": None, "kpss_p": None, "stationary_flag": None, "notes": []}
    x = s.dropna().to_numpy(dtype=float)
    if x.size < 15 or not HAVE_STATSMODELS:
        out["notes"].append("Stationarity tests skipped (need statsmodels and >=15 non-missing).")
        return out

    try:
        adf_res = adfuller(x, autolag="AIC")
        out["adf_p"] = float(adf_res[1])
    except Exception as e:
        out["notes"].append(f"ADF failed: {repr(e)}")

    try:
        with warnings.catch_warnings(record=True) as wrec:
            if InterpolationWarning is not None:
                warnings.filterwarnings("ignore", category=InterpolationWarning)
            kpss_res = kpss(x, regression="c", nlags="auto")
            out["kpss_p"] = float(kpss_res[1])
            if wrec:
                out["notes"].append(
                    "KPSS warning: p-value outside lookup table (InterpolationWarning); reported p-value is bounded/approx."
                )
    except Exception as e:
        out["notes"].append(f"KPSS failed: {repr(e)}")

    adf_p = out["adf_p"]
    kpss_p = out["kpss_p"]

    if adf_p is None and kpss_p is None:
        out["stationary_flag"] = None
    elif adf_p is not None and kpss_p is not None:
        out["stationary_flag"] = bool((adf_p < THR.adf_alpha) and (kpss_p >= THR.kpss_alpha))
    elif adf_p is not None:
        out["stationary_flag"] = bool(adf_p < THR.adf_alpha)
        out["notes"].append("Stationarity decided using ADF only.")
    else:
        out["stationary_flag"] = bool(kpss_p >= THR.kpss_alpha)
        out["notes"].append("Stationarity decided using KPSS only.")

    return out


def _ljung_box_p(s: pd.Series, lags: int = 10) -> Optional[float]:
    if not HAVE_STATSMODELS:
        return None
    x = s.dropna().to_numpy(dtype=float)
    if x.size < (lags + 8):
        return None
    try:
        lb = acorr_ljungbox(x, lags=[min(lags, x.size - 1)], return_df=True)
        return float(lb["lb_pvalue"].iloc[0])
    except Exception:
        return None


def _time_regularity(df: pd.DataFrame, time_col: Optional[str], date_col: Optional[str]) -> Dict[str, Any]:
    """
    Computes modal step + fraction of non-modal steps (irregular_fraction),
    and duplicates fraction.
    """
    out: Dict[str, Any] = {
        "time_col": time_col,
        "date_col": date_col,
        "n_rows": int(df.shape[0]),
        "regularity": None,
        "modal_step": None,
        "irregular_fraction": None,
        "duplicate_time_points": None,
        "duplicate_fraction": None,
        "notes": [],
    }

    if date_col is not None:
        t = pd.to_datetime(df[date_col], errors="coerce")
        valid = t.notna()
        if valid.mean() < 0.80:
            out["notes"].append("Date column exists but parsing success <80%; time regularity uncertain.")
        t = t[valid].sort_values()
        if len(t) < 5:
            out["notes"].append("Too few valid datetime points to assess regularity.")
            return out
        diffs = t.diff().dropna()
        if diffs.empty:
            out["notes"].append("No diffs available for datetime regularity.")
            return out

        secs = diffs.dt.total_seconds().to_numpy()
        rounded = np.round(secs, -1)
        mode_val = float(pd.Series(rounded).value_counts().idxmax())
        irregular = float(np.mean(np.abs(rounded - mode_val) > 0.0))

        dup_ct = int(t.duplicated().sum())
        out["modal_step"] = mode_val
        out["irregular_fraction"] = irregular
        out["duplicate_time_points"] = dup_ct
        out["duplicate_fraction"] = float(dup_ct / max(1, len(t)))

        if irregular <= THR.irregular_time_soft_threshold:
            out["regularity"] = "regular_or_quasi_regular"
        elif irregular <= THR.irregular_time_hard_threshold:
            out["regularity"] = "moderately_irregular"
        else:
            out["regularity"] = "highly_irregular"
        return out

    if time_col is not None:
        t = pd.to_numeric(df[time_col], errors="coerce")
        valid = t.notna()
        t = t[valid].sort_values()
        if len(t) < 5:
            out["notes"].append("Too few valid numeric time points to assess regularity.")
            return out
        diffs = t.diff().dropna()
        if diffs.empty:
            out["notes"].append("No diffs available for numeric time regularity.")
            return out

        d = diffs.to_numpy(dtype=float)
        rounded = np.round(d, 6)
        mode_val = float(pd.Series(rounded).value_counts().idxmax())
        irregular = float(np.mean(np.abs(rounded - mode_val) > 0.0))

        dup_ct = int(t.duplicated().sum())
        out["modal_step"] = mode_val
        out["irregular_fraction"] = irregular
        out["duplicate_time_points"] = dup_ct
        out["duplicate_fraction"] = float(dup_ct / max(1, len(t)))

        if irregular <= THR.irregular_time_soft_threshold:
            out["regularity"] = "regular_or_quasi_regular"
        elif irregular <= THR.irregular_time_hard_threshold:
            out["regularity"] = "moderately_irregular"
        else:
            out["regularity"] = "highly_irregular"
        return out

    out["notes"].append("No time/date column detected; time regularity unknown.")
    return out


def _pairwise_complete_n(df: pd.DataFrame, vars_: List[str]) -> Dict[Tuple[str, str], int]:
    out: Dict[Tuple[str, str], int] = {}
    for i in range(len(vars_)):
        for j in range(i + 1, len(vars_)):
            a, b = vars_[i], vars_[j]
            out[(a, b)] = int(df[[a, b]].dropna().shape[0])
    return out


def _pairwise_n_summary(pairwise: Dict[Tuple[str, str], int]) -> Dict[str, Any]:
    if not pairwise:
        return {"median": None, "q25": None, "min": None}
    vals = np.array(list(pairwise.values()), dtype=float)
    return {
        "median": int(np.median(vals)),
        "q25": int(np.quantile(vals, 0.25)),
        "min": int(np.min(vals)),
    }


def _corr_matrix(df: pd.DataFrame, vars_: List[str], method: str) -> pd.DataFrame:
    return df[vars_].corr(method=method, min_periods=max(3, THR.tier1_min_pairwise_n_q25))


def _effective_n_lagged_multivariate_strict(df: pd.DataFrame, vars_: List[str], lag: int) -> int:
    """
    Very conservative: counts rows where ALL vars are present at t and t-lag.
    """
    if not vars_:
        return 0
    cur = df[vars_].notna().all(axis=1)
    lagged = df[vars_].shift(lag).notna().all(axis=1)
    ok = cur & lagged
    return int(ok.sum())


def _effective_n_lagged_per_variable(df: pd.DataFrame, vars_: List[str], lag: int) -> Dict[str, int]:
    """
    For each variable, counts rows where var present at t and t-lag.
    """
    out: Dict[str, int] = {}
    for c in vars_:
        ok = df[c].notna() & df[c].shift(lag).notna()
        out[c] = int(ok.sum())
    return out


def _choose_corr_method(variables: Dict[str, Any], cols: List[str]) -> str:
    if not cols:
        return "pearson"
    heavy_outlier = 0
    nonnormal = 0
    n = 0
    for c in cols:
        v = variables[c]
        n += 1
        op = v.get("outlier_pct_robust_z")
        if op is not None and op >= 0.10:
            heavy_outlier += 1
        jb = v.get("normality", {}).get("jarque_bera_p")
        if jb is not None and jb < 0.01:
            nonnormal += 1
    if n == 0:
        return "pearson"
    if (heavy_outlier / n) >= 0.25 or (nonnormal / n) >= 0.40:
        return "spearman"
    return "pearson"


# -----------------------------
# LLM finalization
# -----------------------------
def _build_llm_payload(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Condensed payload to keep token size stable:
    - includes tier/variant evidence + score breakdown + aggregated variable issues (top items)
    """
    overall = report.get("overall", {})
    vars_dict = report.get("variables", {})
    dropped = overall.get("dropped_variables", {})

    hard = []
    soft = []
    for name, info in dropped.items():
        lvl = info.get("level")
        reasons = info.get("reasons", []) or []
        item = {"var": name, "level": lvl, "reasons": reasons[:2]}
        if lvl == "hard":
            hard.append(item)
        else:
            soft.append(item)

    outlier_vars = []
    missing_vars = []
    dominance_vars = []
    nonstationary_vars = []
    for name, v in vars_dict.items():
        op = v.get("outlier_pct_robust_z")
        if op is not None and op >= 0.20:
            outlier_vars.append({"var": name, "outlier_pct": float(op)})
        mp = float(v.get("missing_pct", 0.0))
        if mp >= 0.40:
            missing_vars.append({"var": name, "missing_pct": mp})
        dom = float(v.get("dominance_pct", 0.0))
        if dom >= 0.92:
            dominance_vars.append({"var": name, "dominance_pct": dom})
        st = v.get("stationarity", {}).get("stationary_flag")
        if st is False:
            nonstationary_vars.append(
                {"var": name, "adf_p": v.get("stationarity", {}).get("adf_p"), "kpss_p": v.get("stationarity", {}).get("kpss_p")}
            )

    outlier_vars = sorted(outlier_vars, key=lambda d: d["outlier_pct"], reverse=True)[:8]
    missing_vars = sorted(missing_vars, key=lambda d: d["missing_pct"], reverse=True)[:8]
    dominance_vars = sorted(dominance_vars, key=lambda d: d["dominance_pct"], reverse=True)[:8]
    nonstationary_vars = nonstationary_vars[:10]

    payload = {
        "meta": report.get("meta", {}),
        "dataset_overview": report.get("dataset_overview", {}),
        "tiers": report.get("tiers", {}),
        "overall": {
            "recommended_tier": overall.get("recommended_tier"),
            "tier3_variant": overall.get("tier3_variant"),
            "readiness_score_0_100": overall.get("readiness_score_0_100"),
            "readiness_label": overall.get("readiness_label"),
            "score_breakdown": overall.get("score_breakdown", {}),
            "why": (overall.get("why", []) or [])[:12],
            "technical_summary": overall.get("technical_summary", ""),
            "client_friendly_summary": overall.get("client_friendly_summary", ""),
            "next_steps": (overall.get("next_steps", []) or [])[:6],
            "caveats": (overall.get("caveats", []) or [])[:8],
        },
        "variable_issue_aggregates": {
            "n_variables_total": int(len(vars_dict)),
            "n_dropped_total": int(len(dropped)),
            "hard_drops_top": hard[:12],
            "soft_drops_top": soft[:12],
            "outlier_vars_top": outlier_vars,
            "missing_vars_top": missing_vars,
            "dominance_vars_top": dominance_vars,
            "nonstationary_vars_top": nonstationary_vars,
        },
        "note_on_nonstationarity": (
            "Non-stationarity is a concern for static lagged VAR/gVAR, but time-varying gVAR can be appropriate "
            "if there is enough data to support smoothing/windows."
        ),
    }
    return _jsonify(payload)


def _llm_prompt(condensed_payload: Dict[str, Any]) -> str:
    return f"""
You are a statistical QA reviewer for single-subject time-series network readiness reports.

You will receive a condensed JSON payload produced by an automated pipeline.

IMPORTANT DOMAIN NOTE
- Non-stationarity: treat as a blocker mainly for STATIC lagged VAR/gVAR. If TIME_VARYING_gVAR is feasible,
  non-stationarity is NOT a blocker (but may increase data needs and requires smoothing/window choices).

TASK
1) Validate whether the recommended tier and (if Tier 3) the recommended variant are coherent with the evidence.
2) If you adjust anything, keep it conservative and explain precisely why with explicit evidence strings.
3) Produce two summaries:
   - technical_summary: for analysts/developers
   - client_friendly_summary: for a non-technical app user, motivating continued data collection ; so do not use jargon here but rather user-friendly actionable summary
4) Produce actionable next_steps: up to 6 short items, each starting with a verb.
5) Produce caveats: up to 8 short items.

RULES
- Output MUST be valid JSON only. No markdown. No extra keys beyond the schema.
- Do not hallucinate data. Use only the payload.
- Allowed adjustments only:
    score_adjustment in [-10, +10] (integer)
    tier_override: one of
        "Tier3_LaggedDynamicNetwork",
        "Tier2_ContemporaneousPartialCorrelation",
        "Tier1_CorrelationMatrix",
        "Tier0_DescriptivesOnly"
    tier3_variant_override: one of
        "STATIC_gVAR",
        "TIME_VARYING_gVAR",
        null
  If you override tier/variant, justify with specific evidence from the payload.
- If fine, set score_adjustment=0 and overrides=null.
- Keep summaries compact: technical_summary <= 1100 chars; client_friendly_summary <= 800 chars.

SCHEMA (return exactly these keys)
{{
  "tier_override": string|null,
  "tier3_variant_override": string|null,
  "score_adjustment": int,
  "final_readiness_label": string,
  "key_evidence": [string, ...],
  "technical_summary": string,
  "client_friendly_summary": string,
  "next_steps": [string, ...],
  "caveats": [string, ...]
}}

PAYLOAD JSON
{json.dumps(condensed_payload, ensure_ascii=False)}
""".strip()


def _extract_openai_text(resp: Any) -> str:
    if hasattr(resp, "output_text") and isinstance(resp.output_text, str):
        return resp.output_text
    try:
        parts = []
        for item in getattr(resp, "output", []) or []:
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", None) == "output_text":
                    parts.append(getattr(c, "text", ""))
        txt = "".join(parts).strip()
        if txt:
            return txt
    except Exception:
        pass
    return ""


def _call_llm_finalize(report: Dict[str, Any]) -> Dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set; cannot run LLM finalization.")

    condensed_payload = _build_llm_payload(report)
    prompt = _llm_prompt(condensed_payload)

    if HAVE_OPENAI_NEW:
        client = OpenAI(api_key=api_key)
        resp = client.responses.create(
            model=LLM_MODEL,
            input=[{"role": "user", "content": prompt}],
        )
        text = _extract_openai_text(resp)
        if not text:
            raise RuntimeError("LLM returned empty output_text.")
        try:
            return json.loads(text)
        except Exception as e:
            raise RuntimeError(f"LLM output was not valid JSON: {repr(e)} ; raw={text[:1200]}")

    if HAVE_OPENAI_LEGACY:
        openai.api_key = api_key
        try:
            resp = openai.ChatCompletion.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp["choices"][0]["message"]["content"]
            return json.loads(text)
        except Exception as e:
            raise RuntimeError(f"Legacy OpenAI call failed: {repr(e)}")

    raise RuntimeError("No OpenAI client available (install openai>=1.0 recommended).")


def _apply_llm_adjustments(report: Dict[str, Any], llm_out: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    notes: List[str] = []
    overall = report["overall"]
    base_score = float(overall["readiness_score_0_100"])
    base_tier = str(overall["recommended_tier"])
    base_variant = overall.get("tier3_variant", None)
    base_label = str(overall["readiness_label"])

    needed = {
        "tier_override",
        "tier3_variant_override",
        "score_adjustment",
        "final_readiness_label",
        "key_evidence",
        "technical_summary",
        "client_friendly_summary",
        "next_steps",
        "caveats",
    }
    missing = needed - set(llm_out.keys())
    if missing:
        notes.append(f"LLM output missing keys: {sorted(list(missing))}. Ignoring LLM.")
        return report, notes

    try:
        adj = int(llm_out.get("score_adjustment", 0))
    except Exception:
        adj = 0
        notes.append("LLM score_adjustment not int; set to 0.")
    adj = max(-10, min(10, adj))
    new_score = _score_clip(base_score + adj)

    allowed_tiers = {
        "Tier3_LaggedDynamicNetwork",
        "Tier2_ContemporaneousPartialCorrelation",
        "Tier1_CorrelationMatrix",
        "Tier0_DescriptivesOnly",
        None,
    }
    tier_override = llm_out.get("tier_override", None)
    if tier_override not in allowed_tiers:
        notes.append("LLM tier_override invalid; ignored.")
        tier_override = None
    final_tier = tier_override if tier_override is not None else base_tier

    allowed_variants = {"STATIC_gVAR", "TIME_VARYING_gVAR", None}
    variant_override = llm_out.get("tier3_variant_override", None)
    if variant_override not in allowed_variants:
        notes.append("LLM tier3_variant_override invalid; ignored.")
        variant_override = None

    # Only apply variant if Tier3 (otherwise keep None)
    if final_tier == "Tier3_LaggedDynamicNetwork":
        final_variant = variant_override if variant_override is not None else base_variant
    else:
        final_variant = None

    allowed_labels = {
        "NotReady",
        "PartiallyReady_Low",
        "PartiallyReady_High",
        "Ready_Low",
        "Ready_High",
        "FullyReady",
    }
    final_label = llm_out.get("final_readiness_label", None)
    if final_label not in allowed_labels:
        final_label = _label_from_score(new_score)
        notes.append("LLM label invalid; recomputed from adjusted score.")

    report["llm_finalization"] = {
        "enabled": True,
        "model": LLM_MODEL,
        "applied_at": _now_iso(),
        "outputs": llm_out,
        "applied": {
            "score_adjustment": adj,
            "final_score": new_score,
            "tier_override": tier_override,
            "final_tier": final_tier,
            "tier3_variant_override": variant_override,
            "final_tier3_variant": final_variant,
            "final_label": final_label,
            "notes": notes,
        },
    }

    overall["readiness_score_0_100_raw"] = base_score
    overall["readiness_label_raw"] = base_label
    overall["recommended_tier_raw"] = base_tier
    overall["tier3_variant_raw"] = base_variant

    overall["readiness_score_0_100"] = new_score
    overall["readiness_label"] = final_label
    overall["recommended_tier"] = final_tier
    overall["tier3_variant"] = final_variant

    overall["technical_summary"] = str(llm_out.get("technical_summary", overall.get("technical_summary", ""))).strip()
    overall["client_friendly_summary"] = str(llm_out.get("client_friendly_summary", overall.get("client_friendly_summary", ""))).strip()

    ns = llm_out.get("next_steps", [])
    cv = llm_out.get("caveats", [])
    ke = llm_out.get("key_evidence", [])
    overall["next_steps"] = ns if isinstance(ns, list) else overall.get("next_steps", [])
    overall["caveats"] = cv if isinstance(cv, list) else overall.get("caveats", [])
    overall["key_evidence"] = ke if isinstance(ke, list) else []

    return report, notes


# -----------------------------
# Core analysis
# -----------------------------
def analyze_profile(
    csv_path: Path,
    output_root: Path,
    lag: int = 1,
    llm_finalize: bool = True,
    verbose: bool = True,
    prefer_time_varying: bool = True,
) -> Dict[str, Any]:
    profile_id = csv_path.parent.name
    if verbose:
        print(f"    [LOAD] Reading CSV…")

    df_raw = _read_csv_robust(csv_path)
    df_raw.columns = [str(c).strip() for c in df_raw.columns]

    time_col, date_col = _infer_time_columns(df_raw)
    df, ordering_info = _sort_by_time(df_raw, time_col, date_col)
    time_info = _time_regularity(df, time_col, date_col)

    excluded = set([c for c in [time_col, date_col] if c is not None])
    for c in list(df.columns):
        if str(c).lower() in {"idx", "index", "row", "subject", "id"}:
            excluded.add(c)

    candidate_cols = [c for c in df.columns if c not in excluded]
    pc_cols = [c for c in candidate_cols if str(c).startswith("P") or str(c).startswith("C")]
    var_cols = pc_cols if len(pc_cols) >= 2 else candidate_cols

    coercion: Dict[str, Dict[str, Any]] = {}
    numeric_df = pd.DataFrame(index=df.index)
    non_numeric_cols: List[str] = []

    if verbose:
        print(f"    [SCAN] Coercing variables to numeric… (candidates={len(var_cols)})")

    for c in var_cols:
        s_num, info = _coerce_numeric_series(df[c])
        coercion[c] = info
        added_nan_rate = info.get("non_numeric_rate")
        if added_nan_rate is not None and added_nan_rate > 0.25:
            non_numeric_cols.append(c)
            continue
        numeric_df[c] = s_num

    # sanitize infinities
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)

    model_cols = [c for c in numeric_df.columns if c not in non_numeric_cols]

    variables: Dict[str, Any] = {}
    hard_drop: Dict[str, List[str]] = {}
    soft_drop: Dict[str, List[str]] = {}
    transform_suggestions: Dict[str, List[str]] = {}

    dup_frac = _safe_float(time_info.get("duplicate_fraction"))
    dup_frac_val = dup_frac if dup_frac is not None else 0.0

    if verbose:
        print(f"    [QC] Variable-level diagnostics… (numeric={len(model_cols)}; non-numeric-excluded={len(non_numeric_cols)})")

    for c in model_cols:
        s = numeric_df[c]
        n_total = int(len(s))
        n_nonmissing = int(s.notna().sum())
        miss_pct = float(1.0 - (n_nonmissing / max(1, n_total)))
        uniq = int(s.dropna().nunique())
        dom = _dominance_pct(s)
        vtype = _infer_variable_type(s)

        x = s.dropna().to_numpy(dtype=float)
        mean = float(np.mean(x)) if x.size else None
        std = float(np.std(x, ddof=1)) if x.size >= 2 else None
        med = float(np.median(x)) if x.size else None

        outlier_pct = _robust_outlier_pct(s, THR.robust_z_outlier_threshold)
        miss_streak = _longest_missing_streak(s)

        norm = _normality_tests(x) if x.size else {"jarque_bera_p": None, "shapiro_p": None, "skew": None, "kurtosis": None}
        stat = _stationarity_tests(s)

        ac1 = None
        try:
            if x.size >= 3:
                ac1 = float(pd.Series(x).autocorr(lag=1))
        except Exception:
            ac1 = None
        lb_p = _ljung_box_p(s, lags=min(10, max(2, x.size // 6)))

        rare_event_flag = None
        zero_inflation_flag = None
        if x.size > 0:
            if vtype == "binary":
                p = float(np.mean(x))
                rare_event_flag = bool(min(p, 1.0 - p) < THR.binary_rare_event_min_pct)
            if vtype in {"count", "continuous"}:
                zero_rate = float(np.mean(np.isclose(x, 0.0)))
                if zero_rate >= 0.70 and np.max(x) > 0:
                    zero_inflation_flag = True

        t_sug: List[str] = []
        if vtype == "count":
            t_sug.append("Consider variance-stabilizing transform: sqrt(x) or log1p(x).")
        if vtype in {"continuous", "proportion"} and norm.get("skew") is not None and x.size:
            skew = float(norm["skew"])
            if abs(skew) > 1.25 and np.min(x) >= 0:
                t_sug.append("Right-skew detected; consider log1p(x) (if non-negative).")
        if outlier_pct is not None and outlier_pct >= 0.10:
            t_sug.append("Many outliers; consider winsorization or robust estimators (Spearman/robust partial corr).")
        if stat.get("stationary_flag") is False:
            t_sug.append("Non-stationary suspected; static lagged VAR may need detrending/differencing; time-varying gVAR can be an alternative if N is high.")
        if zero_inflation_flag:
            t_sug.append("High zero-rate; consider zero-inflated / hurdle-style modeling or discretization for some tiers.")
        transform_suggestions[c] = t_sug

        hd: List[str] = []
        sd: List[str] = []

        if n_nonmissing == 0:
            hd.append("All values missing.")
        if miss_pct >= THR.hard_drop_missing_pct:
            hd.append(f"Too much missingness ({miss_pct:.0%} >= {THR.hard_drop_missing_pct:.0%}).")
        elif miss_pct >= THR.soft_drop_missing_pct:
            sd.append(f"High missingness ({miss_pct:.0%} >= {THR.soft_drop_missing_pct:.0%}).")

        if uniq <= THR.hard_drop_unique_values_max:
            hd.append(f"No variability (unique={uniq}).")
        elif uniq <= THR.soft_drop_unique_values_max:
            sd.append(f"Very low variability (unique={uniq}).")

        if std is not None and std <= THR.near_constant_std_epsilon:
            hd.append("Near-constant series (std≈0).")

        if dom >= THR.dominance_soft_drop_pct and uniq >= 2:
            sd.append(f"One value dominates ({dom:.0%}); may destabilize estimation.")

        if vtype == "binary" and rare_event_flag:
            sd.append("Binary rare-event imbalance; unstable edges likely (especially Tier 2/3).")

        if outlier_pct is not None and outlier_pct >= THR.outlier_hard_drop_pct:
            sd.append(f"Extremely outlier-heavy ({outlier_pct:.0%}); consider drop or strong robustification.")
        elif outlier_pct is not None and outlier_pct >= THR.outlier_soft_drop_pct:
            sd.append(f"Outlier-heavy ({outlier_pct:.0%}); robust methods recommended.")

        if stat.get("stationary_flag") is False:
            sd.append("Non-stationary by tests: concern for STATIC lagged VAR; acceptable under TIME-VARYING gVAR if other readiness is high.")

        if miss_streak >= max(5, int(0.10 * n_total)):
            sd.append(f"Long missing streak (max consecutive NA={miss_streak}).")

        if zero_inflation_flag:
            sd.append("High zero-rate (potential zero inflation); may distort Gaussian assumptions.")

        if hd:
            hard_drop[c] = hd
        elif sd:
            soft_drop[c] = sd

        variables[c] = {
            "name": c,
            "role": "predictor" if str(c).startswith("P") else ("criterion" if str(c).startswith("C") else "unknown"),
            "coercion": coercion.get(c, {}),
            "type_inferred": vtype,
            "n_total": n_total,
            "n_nonmissing": n_nonmissing,
            "missing_pct": miss_pct,
            "n_unique_nonmissing": uniq,
            "dominance_pct": dom,
            "mean": mean,
            "std": std,
            "median": med,
            "outlier_pct_robust_z": outlier_pct,
            "max_consecutive_missing": miss_streak,
            "normality": norm,
            "stationarity": stat,
            "autocorr_lag1": ac1,
            "ljung_box_p": lb_p,
            "rare_event_flag": rare_event_flag,
            "zero_inflation_flag": zero_inflation_flag,
            "drop_hard_reasons": hd,
            "drop_soft_reasons": sd,
            "transform_suggestions": t_sug,
        }

    kept_after_hard = [c for c in model_cols if c not in hard_drop and c in variables]

    corr_method = _choose_corr_method(variables, kept_after_hard)

    collinearity_notes: List[str] = []
    high_corr_pairs: List[Dict[str, Any]] = []
    collinearity_drop: Dict[str, List[str]] = {}

    cols_for_corr = [c for c in kept_after_hard if variables[c]["n_nonmissing"] >= THR.tier1_min_pairwise_n_q25]
    if len(cols_for_corr) >= 2:
        if verbose:
            print(f"    [ASSUME] Correlation scan… (method={corr_method}, vars={len(cols_for_corr)})")
        cm = _corr_matrix(numeric_df, cols_for_corr, method=corr_method)
        for i, a in enumerate(cols_for_corr):
            for j in range(i + 1, len(cols_for_corr)):
                b = cols_for_corr[j]
                val = cm.loc[a, b]
                if pd.isna(val):
                    continue
                av = float(val)
                if abs(av) >= THR.high_corr_threshold:
                    high_corr_pairs.append({"a": a, "b": b, "corr": av, "method": corr_method})
        if high_corr_pairs:
            collinearity_notes.append(f"Found {len(high_corr_pairs)} highly collinear pairs (|r| >= {THR.high_corr_threshold}).")

    if high_corr_pairs:
        dropped = set()
        pairs_sorted = sorted(high_corr_pairs, key=lambda d: abs(d["corr"]), reverse=True)

        def badness(vn: str) -> Tuple[float, float, float, int]:
            v = variables[vn]
            miss = float(v["missing_pct"])
            outp = v.get("outlier_pct_robust_z")
            outv = float(outp) if outp is not None else 0.0
            dom = float(v["dominance_pct"])
            uniq = int(v["n_unique_nonmissing"])
            return (miss, outv, dom, -uniq)

        for p in pairs_sorted:
            a, b = p["a"], p["b"]
            if a in dropped or b in dropped:
                continue
            if badness(a) > badness(b):
                drop_c, keep_c = a, b
            else:
                drop_c, keep_c = b, a
            dropped.add(drop_c)
            collinearity_drop.setdefault(drop_c, []).append(
                f"High collinearity with {keep_c} (|corr|={abs(p['corr']):.3f}, {corr_method}); drop one for identifiability."
            )

        for c, reasons in collinearity_drop.items():
            soft_drop.setdefault(c, [])
            soft_drop[c].extend(reasons)
            variables[c]["drop_soft_reasons"] = list(set(variables[c]["drop_soft_reasons"] + reasons))

    # -----------------------------
    # Tier readiness sets
    # -----------------------------
    tier1_ready = [
        c for c in kept_after_hard
        if variables[c]["n_nonmissing"] >= THR.tier1_min_pairwise_n_q25 and variables[c]["n_unique_nonmissing"] >= 2
    ]
    tier2_ready = [c for c in tier1_ready if c not in collinearity_drop]

    # Tier 3 candidates base: exclude binary (default Gaussian heuristic) and keep numeric
    tier3_base: List[str] = []
    for c in tier2_ready:
        vtype = variables[c]["type_inferred"]
        if vtype in {"continuous", "count", "proportion", "ordinal"}:
            tier3_base.append(c)

    # Static candidate: remove clearly non-stationary
    tier3_static_candidates = [c for c in tier3_base if variables[c]["stationarity"]["stationary_flag"] is not False]

    # Time-varying candidate: allow non-stationary; still keep quality filters implicitly (hard drops already removed)
    tier3_tv_candidates = list(tier3_base)

    # Time gating (Tier 3 sensitive)
    reg = time_info.get("regularity")
    irregular_frac = _safe_float(time_info.get("irregular_fraction"))
    irr = irregular_frac if irregular_frac is not None else 0.20

    tier3_time_ok = True
    tier3_time_notes: List[str] = []
    if reg == "highly_irregular":
        tier3_time_ok = False
        tier3_time_notes.append("Sampling is highly irregular; Tier 3 lagged models are risky without resampling/CT methods.")
    elif reg == "moderately_irregular":
        tier3_time_notes.append("Sampling is moderately irregular; consider resampling or continuous-time modeling.")

    dup_notes: List[str] = []
    if dup_frac_val >= THR.duplicate_time_hard_threshold:
        tier3_time_ok = False
        dup_notes.append(f"Many duplicate timestamps (dup_fraction={dup_frac_val:.1%}); aggregate duplicates before lagged modeling.")
    elif dup_frac_val >= THR.duplicate_time_soft_threshold:
        dup_notes.append(f"Some duplicate timestamps (dup_fraction={dup_frac_val:.1%}); consider aggregating duplicates.")

    if verbose:
        print(f"    [FEAS] Effective-N calculations… (lag={lag})")

    # STATIC Tier 3 effective N
    n_eff_strict_static = _effective_n_lagged_multivariate_strict(numeric_df, tier3_static_candidates, lag=lag) if tier3_static_candidates else 0
    n_eff_per_var_static = _effective_n_lagged_per_variable(numeric_df, tier3_static_candidates, lag=lag) if tier3_static_candidates else {}
    n_eff_q25_static = int(np.quantile(list(n_eff_per_var_static.values()), 0.25)) if n_eff_per_var_static else 0

    # TV Tier 3 effective N
    n_eff_strict_tv = _effective_n_lagged_multivariate_strict(numeric_df, tier3_tv_candidates, lag=lag) if tier3_tv_candidates else 0
    n_eff_per_var_tv = _effective_n_lagged_per_variable(numeric_df, tier3_tv_candidates, lag=lag) if tier3_tv_candidates else {}
    n_eff_q25_tv = int(np.quantile(list(n_eff_per_var_tv.values()), 0.25)) if n_eff_per_var_tv else 0

    # Pairwise N summary for Tier 1/2
    pairwise_all = _pairwise_complete_n(numeric_df, tier1_ready)
    pairwise_summary = _pairwise_n_summary(pairwise_all)

    k1, k2 = len(tier1_ready), len(tier2_ready)
    k3_static, k3_tv = len(tier3_static_candidates), len(tier3_tv_candidates)

    # Tier 3 STATIC feasibility
    tier3_required_static = THR.tier3_static_min_n_eff_base + THR.tier3_static_min_n_eff_per_var * max(0, k3_static - 1)
    tier3_static_ok = (
        k3_static >= THR.tier3_min_k
        and tier3_time_ok
        and (n_eff_q25_static >= max(tier3_required_static, THR.tier3_static_min_n_eff_q25_floor))
    )
    if tier3_static_ok and k3_static > int(THR.max_var_to_n_ratio_tier3_static * max(1, n_eff_q25_static)):
        tier3_static_ok = False
        tier3_time_notes.append(f"Too many variables for STATIC Tier 3 (k={k3_static}, n_eff_q25={n_eff_q25_static}); reduce variable set.")

    # Tier 3 TIME-VARYING feasibility
    # Window heuristic (very rough): window >= max(min_window, per_var*k)
    tv_window = int(max(THR.tier3_tv_window_min, THR.tier3_tv_window_per_var * max(1, k3_tv)))
    tv_required_points = int(max(THR.tier3_tv_min_n_eff_q25_floor, THR.tier3_tv_min_windows * tv_window))
    tier3_tv_ok = (
        k3_tv >= THR.tier3_min_k
        and tier3_time_ok
        and (n_eff_q25_tv >= tv_required_points)
    )
    if tier3_tv_ok and k3_tv > int(THR.max_var_to_n_ratio_tier3_tv * max(1, n_eff_q25_tv)):
        tier3_tv_ok = False
        tier3_time_notes.append(f"Too many variables for TIME-VARYING Tier 3 (k={k3_tv}, n_eff_q25={n_eff_q25_tv}); reduce variable set.")

    # Tier 2 feasibility
    tier2_ok = (
        k2 >= THR.tier2_min_k
        and pairwise_summary["median"] is not None
        and pairwise_summary["q25"] is not None
        and pairwise_summary["median"] >= THR.tier2_min_pairwise_n_median
        and pairwise_summary["q25"] >= THR.tier2_min_pairwise_n_q25
    )
    if tier2_ok and k2 > int(THR.max_var_to_n_ratio_tier2 * max(1, pairwise_summary["median"] or 0)):
        tier2_ok = False

    # Tier 1 feasibility
    tier1_ok = (
        k1 >= THR.tier1_min_k
        and pairwise_summary["median"] is not None
        and pairwise_summary["q25"] is not None
        and pairwise_summary["median"] >= THR.tier1_min_pairwise_n_median
        and pairwise_summary["q25"] >= THR.tier1_min_pairwise_n_q25
    )

    # Decide Tier 3 variant (if Tier 3 feasible at all)
    tier3_variant: Optional[str] = None
    if prefer_time_varying and tier3_tv_ok:
        tier3_variant = "TIME_VARYING_gVAR"
    elif tier3_static_ok:
        tier3_variant = "STATIC_gVAR"
    elif tier3_tv_ok:
        tier3_variant = "TIME_VARYING_gVAR"

    # Choose overall recommended tier
    if tier3_variant is not None:
        recommended_tier = "Tier3_LaggedDynamicNetwork"
        ready_vars = tier3_tv_candidates if tier3_variant == "TIME_VARYING_gVAR" else tier3_static_candidates
    elif tier2_ok:
        recommended_tier = "Tier2_ContemporaneousPartialCorrelation"
        ready_vars = tier2_ready
    elif tier1_ok:
        recommended_tier = "Tier1_CorrelationMatrix"
        ready_vars = tier1_ready
    else:
        recommended_tier = "Tier0_DescriptivesOnly"
        ready_vars = []
        tier3_variant = None

    # -----------------------------
    # Dropped variables (tier-specific + hard)
    # -----------------------------
    dropped_vars: Dict[str, Any] = {}
    for c in model_cols:
        if c not in variables:
            continue
        if c in hard_drop:
            dropped_vars[c] = {"level": "hard", "reasons": hard_drop[c]}
        elif c not in ready_vars:
            reasons = list(variables[c]["drop_soft_reasons"])
            extra: List[str] = []
            if recommended_tier == "Tier3_LaggedDynamicNetwork":
                if tier3_variant == "STATIC_gVAR":
                    if variables[c]["stationarity"]["stationary_flag"] is False:
                        extra.append("Excluded for Tier 3 STATIC due to non-stationarity signals.")
                if variables[c]["type_inferred"] == "binary":
                    extra.append("Excluded for Tier 3 by default (binary; Gaussian gVAR heuristic).")
            if recommended_tier == "Tier2_ContemporaneousPartialCorrelation" and c in collinearity_drop:
                extra.append("Excluded for Tier 2 due to high collinearity.")
            if recommended_tier == "Tier0_DescriptivesOnly":
                extra.append("Insufficient data quality/quantity for network estimation.")
            reasons = list(dict.fromkeys(reasons + extra))
            dropped_vars[c] = {"level": "soft_or_tier_exclusion", "reasons": reasons}

    # -----------------------------
    # Scoring (0–100), stable weights
    # -----------------------------
    n_rows = int(numeric_df.shape[0])
    base_cols = ready_vars if ready_vars else kept_after_hard

    # sample size score reference differs by tier/variant
    if recommended_tier == "Tier3_LaggedDynamicNetwork" and tier3_variant == "TIME_VARYING_gVAR":
        n_ref = float(n_eff_q25_tv)
        n_req = float(max(THR.tier3_tv_min_n_eff_q25_floor, tv_required_points))
    elif recommended_tier == "Tier3_LaggedDynamicNetwork" and tier3_variant == "STATIC_gVAR":
        n_ref = float(n_eff_q25_static)
        n_req = float(max(tier3_required_static, THR.tier3_static_min_n_eff_q25_floor))
    elif recommended_tier == "Tier2_ContemporaneousPartialCorrelation":
        n_ref = float(pairwise_summary["median"] or 0)
        n_req = float(THR.tier2_min_pairwise_n_median)
    elif recommended_tier == "Tier1_CorrelationMatrix":
        n_ref = float(pairwise_summary["median"] or 0)
        n_req = float(THR.tier1_min_pairwise_n_median)
    else:
        n_ref = float(pairwise_summary["median"] or 0)
        n_req = 20.0

    sample_score = 100.0 * min(1.0, n_ref / max(1.0, n_req))

    if base_cols:
        mean_miss = float(np.mean([variables[c]["missing_pct"] for c in base_cols]))
        missing_score = 100.0 * (1.0 - mean_miss)
    else:
        mean_miss = 1.0
        missing_score = 0.0

    def var_quality_score(c: str) -> float:
        v = variables[c]
        score = 100.0
        dom = float(v["dominance_pct"])
        if dom >= 0.95:
            score -= 30.0
        elif dom >= THR.dominance_soft_drop_pct:
            score -= 15.0

        outp = v.get("outlier_pct_robust_z")
        if outp is not None:
            score -= min(35.0, 100.0 * float(outp))

        uniq = int(v["n_unique_nonmissing"])
        if uniq <= 2:
            score -= 20.0

        streak = int(v["max_consecutive_missing"])
        if n_rows > 0 and streak >= max(5, int(0.10 * n_rows)):
            score -= 10.0

        if v.get("zero_inflation_flag") is True:
            score -= 8.0

        return _score_clip(score)

    vq_score = float(np.mean([var_quality_score(c) for c in base_cols])) if base_cols else 0.0

    time_score = 100.0 * max(0.0, 1.0 - (irr / 0.80))

    # assumptions score: stationarity weight depends on whether it's actually required
    flags = [variables[c]["stationarity"]["stationary_flag"] for c in base_cols] if base_cols else []
    usable = [f for f in flags if f is not None]
    stat_frac = float(np.mean([1.0 if f else 0.0 for f in usable])) if usable else 0.7

    if base_cols and len(base_cols) >= 2:
        total_pairs = (len(base_cols) * (len(base_cols) - 1)) / 2.0
        high_corr_rate = min(1.0, len(high_corr_pairs) / max(1.0, total_pairs))
    else:
        high_corr_rate = 0.0

    dup_pen = 0.0
    if dup_frac_val >= THR.duplicate_time_hard_threshold:
        dup_pen = 0.35
    elif dup_frac_val >= THR.duplicate_time_soft_threshold:
        dup_pen = 0.15

    # stationarity relevance:
    stationarity_required = (recommended_tier == "Tier3_LaggedDynamicNetwork" and tier3_variant == "STATIC_gVAR")
    stationarity_weight = 0.65 if stationarity_required else (0.20 if (recommended_tier == "Tier3_LaggedDynamicNetwork" and tier3_variant == "TIME_VARYING_gVAR") else 0.10)

    # normalize weights inside assumptions component to sum to 1
    # (keep interpretability; avoid negative sums if stationarity_weight changes)
    col_w = 0.70 - (stationarity_weight - 0.10)  # shift weight from stationarity to collinearity for tv-gVAR
    col_w = max(0.20, min(0.75, col_w))
    dup_w = 1.0 - stationarity_weight - col_w
    if dup_w < 0.05:
        dup_w = 0.05
        col_w = 1.0 - stationarity_weight - dup_w

    assumptions_score = _score_clip(
        100.0 * (
            stationarity_weight * stat_frac
            + col_w * (1.0 - high_corr_rate)
            + dup_w * (1.0 - dup_pen)
        )
    )

    readiness_score = _score_clip(
        W.sample_size * sample_score
        + W.missingness * missing_score
        + W.variable_quality * vq_score
        + W.time_regular * time_score
        + W.model_assumptions * assumptions_score
    )

    # small tier-specific gates (explainable)
    tier_gate_notes: List[str] = []
    if recommended_tier == "Tier3_LaggedDynamicNetwork":
        if reg == "moderately_irregular":
            readiness_score = _score_clip(readiness_score - 5.0)
            tier_gate_notes.append("Tier 3 selected but sampling is moderately irregular; small penalty applied.")
        if reg == "highly_irregular":
            readiness_score = _score_clip(readiness_score - 12.0)
            tier_gate_notes.append("Tier 3 selected but sampling is highly irregular; penalty applied.")
        if dup_frac_val >= THR.duplicate_time_soft_threshold:
            readiness_score = _score_clip(readiness_score - 4.0)
            tier_gate_notes.append("Duplicate timestamps present; small penalty applied.")
        if tier3_variant == "TIME_VARYING_gVAR" and (n_ref < n_req):
            tier_gate_notes.append("Time-varying gVAR selected; data amount is near the lower feasibility boundary (consider collecting more).")

    readiness_label = _label_from_score(readiness_score)

    # -----------------------------
    # Summaries (pre-LLM)
    # -----------------------------
    technical_lines: List[str] = []
    if recommended_tier == "Tier3_LaggedDynamicNetwork":
        technical_lines.append(f"Tier={recommended_tier} ({tier3_variant}); score={readiness_score:.1f}/100 ({readiness_label}).")
    else:
        technical_lines.append(f"Tier={recommended_tier}; score={readiness_score:.1f}/100 ({readiness_label}).")

    technical_lines.append(
        f"Vars: candidates={len(model_cols)}, ready={len(ready_vars)}, hard_drop={len(hard_drop)}, soft_drop={len(soft_drop)}."
    )
    technical_lines.append(
        f"Time: regularity={time_info.get('regularity')}, irregular_frac={irr:.2f}, dup_frac={dup_frac_val:.2f}, sorted={ordering_info.get('sorted')}."
    )

    if recommended_tier == "Tier3_LaggedDynamicNetwork":
        technical_lines.append(
            f"Tier3 STATIC: k={k3_static}, n_eff_q25={n_eff_q25_static}, required≈{max(tier3_required_static, THR.tier3_static_min_n_eff_q25_floor)}; feasible={tier3_static_ok}."
        )
        technical_lines.append(
            f"Tier3 TV: k={k3_tv}, n_eff_q25={n_eff_q25_tv}, tv_window≈{tv_window}, tv_required≈{tv_required_points}; feasible={tier3_tv_ok}."
        )
        if tier3_variant == "TIME_VARYING_gVAR":
            technical_lines.append("Non-stationarity is not treated as a blocker under TIME_VARYING_gVAR; feasibility is driven mainly by usable lagged N and time structure.")
        if tier3_time_notes:
            technical_lines.extend(tier3_time_notes)
        if dup_notes:
            technical_lines.extend(dup_notes)
    else:
        technical_lines.append(
            f"Pairwise N: median={pairwise_summary['median']}, q25={pairwise_summary['q25']}, min={pairwise_summary['min']}; corr_method={corr_method}."
        )

    if high_corr_pairs:
        technical_lines.append(f"High collinearity pairs={len(high_corr_pairs)} (|r|>={THR.high_corr_threshold}).")
    if ordering_info.get("sorted") is not True:
        technical_lines.append("Note: data not time-sorted (no reliable time key).")
    if tier_gate_notes:
        technical_lines.extend(tier_gate_notes)

    technical_summary = " ".join([s for s in technical_lines if s])

    client_lines: List[str] = []
    if recommended_tier == "Tier3_LaggedDynamicNetwork" and tier3_variant == "TIME_VARYING_gVAR":
        client_lines.append("Your data is suitable for a time-varying dynamic network model (relationships can change over time).")
        client_lines.append("Even if some signals drift or trend, time-varying modeling can still work when there are enough check-ins.")
    elif recommended_tier == "Tier3_LaggedDynamicNetwork" and tier3_variant == "STATIC_gVAR":
        client_lines.append("Your data is close to supporting a dynamic (time-lagged) network model.")
        client_lines.append("To keep results reliable, regular timing and fewer missing entries are important.")
    elif recommended_tier == "Tier2_ContemporaneousPartialCorrelation":
        client_lines.append("Your data supports a network of same-moment relationships (how variables move together).")
        client_lines.append("More check-ins and fewer missing entries will strengthen the conclusions.")
    elif recommended_tier == "Tier1_CorrelationMatrix":
        client_lines.append("Right now, the most reliable analysis is a correlation overview (basic associations).")
        client_lines.append("More observations and fewer gaps will unlock more advanced network models.")
    else:
        client_lines.append("At the moment, there is not enough consistent data for a reliable network analysis.")
        client_lines.append("Collecting more observations and completing entries more consistently will unlock stronger insights.")

    if mean_miss > 0.35:
        client_lines.append("A key limitation is that many measurements are missing; completing more check-ins will help.")
    if reg in {"moderately_irregular", "highly_irregular"}:
        client_lines.append("Inconsistent timing between measurements reduces model quality; more regular check-ins help.")
    if dup_frac_val >= THR.duplicate_time_soft_threshold:
        client_lines.append("Some measurements share the same timestamp; combining duplicates will improve time-series modeling.")
    if len(ready_vars) < 4 and recommended_tier != "Tier0_DescriptivesOnly":
        client_lines.append("Only a small set of variables is currently usable; capturing more stable signals can help.")

    client_friendly_summary = " ".join(client_lines)

    next_steps: List[str] = []
    if recommended_tier in {"Tier0_DescriptivesOnly", "Tier1_CorrelationMatrix"}:
        next_steps.append("Collect more time points (increase the number of check-ins).")
    if mean_miss > 0.20:
        next_steps.append("Reduce missingness by completing more items per check-in.")
    if reg in {"moderately_irregular", "highly_irregular"}:
        next_steps.append("Increase measurement regularity (aim for a stable schedule).")
    if dup_frac_val >= THR.duplicate_time_soft_threshold:
        next_steps.append("Aggregate duplicate timestamps (e.g., average within same time point).")
    if high_corr_pairs:
        next_steps.append("Review highly redundant items and consider removing duplicates.")
    if any(variables[c]["type_inferred"] == "binary" for c in ready_vars):
        next_steps.append("If possible, measure key states on a scale instead of only yes/no to improve sensitivity.")
    if soft_drop:
        next_steps.append("Prioritize variables that vary over time (avoid items that are almost always the same).")
    if recommended_tier == "Tier3_LaggedDynamicNetwork" and tier3_variant == "TIME_VARYING_gVAR":
        next_steps.append("Plan smoothing/windows for time-varying estimation (and collect more points if close to minimum).")
    next_steps = next_steps[:6]

    caveats: List[str] = []
    if not HAVE_STATSMODELS:
        caveats.append("Stationarity/whiteness tests were skipped (statsmodels not available).")
    if not HAVE_SCIPY:
        caveats.append("Some distribution diagnostics were skipped (scipy not available).")
    if time_info.get("regularity") is None:
        caveats.append("Sampling regularity could not be assessed (no reliable time column detected).")
    if recommended_tier in {"Tier2_ContemporaneousPartialCorrelation", "Tier3_LaggedDynamicNetwork"}:
        caveats.append("Network models can be sensitive to collinearity and variable-to-sample-size ratio.")
    if recommended_tier == "Tier3_LaggedDynamicNetwork" and tier3_variant == "STATIC_gVAR":
        caveats.append("STATIC lagged gVAR typically expects approximate stationarity; trends/drifts may require preprocessing.")
    if recommended_tier == "Tier3_LaggedDynamicNetwork" and tier3_variant == "TIME_VARYING_gVAR":
        caveats.append("TIME-VARYING gVAR relaxes stationarity, but needs more data and requires choosing a smoothing/window strategy.")

    why: List[str] = []
    why.append(f"Recommended tier: {recommended_tier}")
    if recommended_tier == "Tier3_LaggedDynamicNetwork":
        why.append(f"Tier3 variant: {tier3_variant} (stationarity_required={stationarity_required})")
    why.append(f"Ready variables: {len(ready_vars)} / model candidates: {len(model_cols)}")
    why.append(f"Time regularity: {time_info.get('regularity')} (irregular_fraction={irr:.2f}, dup_fraction={dup_frac_val:.2f})")
    if recommended_tier == "Tier3_LaggedDynamicNetwork":
        why.append(f"Tier3 TV feasible={tier3_tv_ok}; n_eff_q25_tv={n_eff_q25_tv} vs required≈{tv_required_points} (tv_window≈{tv_window})")
        why.append(f"Tier3 STATIC feasible={tier3_static_ok}; n_eff_q25_static={n_eff_q25_static} vs required≈{max(tier3_required_static, THR.tier3_static_min_n_eff_q25_floor)}")
        why.append("Non-stationarity is not a blocker if TIME_VARYING_gVAR is feasible; it mainly affects STATIC feasibility.")
        if tier3_time_notes:
            why.extend(tier3_time_notes)
        if dup_notes:
            why.extend(dup_notes)
    else:
        why.append(f"Pairwise N median/q25/min: {pairwise_summary['median']}/{pairwise_summary['q25']}/{pairwise_summary['min']}")
    why.append(f"Mean missingness (base set): {mean_miss:.1%}" if base_cols else "No usable variables after filtering.")
    if high_corr_pairs:
        why.append(f"High collinearity pairs detected: {len(high_corr_pairs)} (method={corr_method})")
    if non_numeric_cols:
        why.append(f"Excluded non-numeric columns: {len(non_numeric_cols)}")
    if ordering_info.get("sorted") is not True:
        why.append("Data not time-sorted due to missing/invalid time key; results depend on original row order.")

    # -----------------------------
    # Report object
    # -----------------------------
    report: Dict[str, Any] = {
        "meta": {
            "profile_id": profile_id,
            "input_file": str(csv_path),
            "generated_at": _now_iso(),
            "script": "apply_readiness_check.py",
            "dependencies": {
                "scipy": HAVE_SCIPY,
                "statsmodels": HAVE_STATSMODELS,
                "openai_new": HAVE_OPENAI_NEW,
                "openai_legacy": HAVE_OPENAI_LEGACY,
            },
            "lag_used": lag,
        },
        "dataset_overview": {
            "n_rows": int(df.shape[0]),
            "n_columns_raw": int(df_raw.shape[1]),
            "time_info": time_info,
            "ordering": ordering_info,
            "excluded_columns": sorted(list(excluded)),
            "candidate_cols_count": int(len(var_cols)),
            "n_numeric_model_candidate_columns": int(len(model_cols)),
            "non_numeric_columns_excluded": non_numeric_cols,
            "corr_method_selected": corr_method,
        },
        "thresholds": dataclasses.asdict(THR),
        "score_weights": dataclasses.asdict(W),
        "variables": variables,
        "drop_recommendations": {
            "hard_drop": hard_drop,
            "soft_drop": soft_drop,
            "tier_specific_dropped": dropped_vars,
            "transform_suggestions": transform_suggestions,
            "collinearity": {
                "high_corr_pairs": high_corr_pairs,
                "collinearity_notes": collinearity_notes,
                "proposed_collinearity_drops": collinearity_drop,
            },
        },
        "tiers": {
            "tier1": {
                "name": "CorrelationMatrix",
                "feasible": tier1_ok,
                "ready_variables": tier1_ready,
                "pairwise_n": pairwise_summary,
                "min_pairwise_required": {
                    "median": THR.tier1_min_pairwise_n_median,
                    "q25": THR.tier1_min_pairwise_n_q25,
                },
                "recommended_corr_method": corr_method,
            },
            "tier2": {
                "name": "ContemporaneousPartialCorrelation",
                "feasible": tier2_ok,
                "ready_variables": tier2_ready,
                "pairwise_n": pairwise_summary,
                "min_pairwise_required": {
                    "median": THR.tier2_min_pairwise_n_median,
                    "q25": THR.tier2_min_pairwise_n_q25,
                },
            },
            "tier3": {
                "name": "LaggedDynamicNetwork_gVAR_heuristic",
                "variant_static": {
                    "feasible": tier3_static_ok,
                    "candidate_variables": tier3_static_candidates,
                    "n_eff_lagged": {
                        "strict_all_vars_complete": n_eff_strict_static,
                        "per_variable": n_eff_per_var_static,
                        "q25_per_variable": n_eff_q25_static,
                    },
                    "required_n_eff_heuristic": max(tier3_required_static, THR.tier3_static_min_n_eff_q25_floor),
                    "stationarity_required": True,
                },
                "variant_time_varying": {
                    "feasible": tier3_tv_ok,
                    "candidate_variables": tier3_tv_candidates,
                    "n_eff_lagged": {
                        "strict_all_vars_complete": n_eff_strict_tv,
                        "per_variable": n_eff_per_var_tv,
                        "q25_per_variable": n_eff_q25_tv,
                    },
                    "tv_window_heuristic": tv_window,
                    "tv_required_points_heuristic": tv_required_points,
                    "stationarity_required": False,
                    "note": "Non-stationarity is acceptable under time-varying gVAR if the dataset is large enough to support smoothing/windows.",
                },
                "time_regular_ok": tier3_time_ok,
                "time_notes": tier3_time_notes + dup_notes,
            },
        },
        "overall": {
            "recommended_tier": recommended_tier,
            "tier3_variant": tier3_variant,
            "ready_variables": ready_vars,
            "dropped_variables": dropped_vars,
            "readiness_score_0_100": readiness_score,
            "readiness_label": readiness_label,
            "score_breakdown": {
                "components": {
                    "sample_score": _score_clip(sample_score),
                    "missing_score": _score_clip(missing_score),
                    "variable_quality_score": _score_clip(vq_score),
                    "time_score": _score_clip(time_score),
                    "assumptions_score": _score_clip(assumptions_score),
                },
                "weights": dataclasses.asdict(W),
                "tier_gates": tier_gate_notes,
                "evidence": {
                    "n_ref_used": n_ref,
                    "n_req_used": n_req,
                    "mean_missingness_base": mean_miss,
                    "irregular_fraction": irr,
                    "duplicate_fraction": dup_frac_val,
                    "stationarity_fraction": stat_frac,
                    "stationarity_required": stationarity_required,
                    "high_collinearity_rate": high_corr_rate,
                    "tier3_tv_window": tv_window if recommended_tier == "Tier3_LaggedDynamicNetwork" else None,
                    "tier3_tv_required_points": tv_required_points if recommended_tier == "Tier3_LaggedDynamicNetwork" else None,
                },
            },
            "why": why,
            "technical_summary": technical_summary,
            "client_friendly_summary": client_friendly_summary,
            "next_steps": next_steps,
            "caveats": caveats,
            "nonstationarity_handling": {
                "statement": "Non-stationarity is not a problem if TIME_VARYING_gVAR is feasible; it mainly limits STATIC_gVAR feasibility.",
                "recommended_variant": tier3_variant,
                "stationarity_required_for_recommended": stationarity_required if recommended_tier == "Tier3_LaggedDynamicNetwork" else False,
            },
        },
        "llm_finalization": {
            "enabled": False,
            "model": LLM_MODEL,
            "notes": ["LLM finalization not run (yet)."],
        },
    }

    # -----------------------------
    # Optional LLM finalization (default ON here)
    # -----------------------------
    if llm_finalize:
        if verbose:
            print(f"    [LLM] Finalizing summaries with {LLM_MODEL}…")
        try:
            llm_out = _call_llm_finalize(report)
            report, llm_notes = _apply_llm_adjustments(report, llm_out)
            if llm_notes and "applied" in report.get("llm_finalization", {}):
                report["llm_finalization"]["applied"]["notes"] = report["llm_finalization"]["applied"]["notes"] + llm_notes
        except Exception as e:
            report["llm_finalization"] = {
                "enabled": True,
                "model": LLM_MODEL,
                "error": repr(e),
                "notes": ["LLM finalization failed; base (non-LLM) results retained."],
            }
            if verbose:
                print(f"    [LLM] Failed: {repr(e)}")

    # -----------------------------
    # Write outputs
    # -----------------------------
    out_dir = output_root / profile_id
    _ensure_dir(out_dir)

    json_path = out_dir / "readiness_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_jsonify(report), f, indent=2, ensure_ascii=False)

    summary_path = out_dir / "readiness_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Profile: {profile_id}\n")
        f.write(f"Input: {csv_path}\n")
        f.write(f"Generated: {report['meta']['generated_at']}\n")
        f.write(f"Tier: {report['overall']['recommended_tier']}\n")
        if report["overall"].get("tier3_variant"):
            f.write(f"Tier3 variant: {report['overall']['tier3_variant']}\n")
        f.write(f"Score: {report['overall']['readiness_score_0_100']:.1f} / 100 ({report['overall']['readiness_label']})\n")
        f.write(
            f"Ready variables ({len(report['overall']['ready_variables'])}): "
            f"{', '.join(report['overall']['ready_variables']) if report['overall']['ready_variables'] else '(none)'}\n\n"
        )

        f.write("Client-friendly summary\n")
        f.write(report["overall"]["client_friendly_summary"].strip() + "\n\n")

        f.write("Technical summary\n")
        f.write(report["overall"]["technical_summary"].strip() + "\n\n")

        f.write("Non-stationarity handling\n")
        f.write(report["overall"]["nonstationarity_handling"]["statement"].strip() + "\n\n")

        f.write("Next steps\n")
        for s in report["overall"].get("next_steps", [])[:6]:
            f.write(f"- {s}\n")
        f.write("\n")

        f.write("Key caveats\n")
        for c in report["overall"].get("caveats", [])[:8]:
            f.write(f"- {c}\n")
        f.write("\n")

        f.write("Why (evidence)\n")
        for line in report["overall"]["why"][:12]:
            f.write(f"- {line}\n")
        if len(report["overall"]["why"]) > 12:
            f.write("- ...\n")

    return report


def discover_profiles(input_root: Path, filename: str) -> List[Path]:
    return sorted(input_root.rglob(filename))


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run readiness checks for pseudoprofile time-series network analysis.")
    p.add_argument("--input-root", type=str, default=DEFAULT_INPUT_ROOT, help="Root folder containing pseudoprofile subfolders.")
    p.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT, help="Root output folder for readiness results.")
    p.add_argument("--filename", type=str, default=DEFAULT_FILENAME, help="CSV filename to find within pseudoprofiles.")
    p.add_argument("--lag", type=int, default=1, help="Lag to use for lagged effective-N calculations.")
    p.add_argument("--max-profiles", type=int, default=0, help="If >0, only process first N discovered profiles.")
    p.add_argument("--llm-finalize", type=str, default="True", help="Enable GPT-5-nano finalization (True/False). Default True.")
    p.add_argument("--quiet", type=str, default="False", help="Reduce console output (True/False). Default False.")
    p.add_argument("--prefer-time-varying", type=str, default="True", help="Prefer TIME_VARYING_gVAR when feasible (True/False). Default True.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    input_root = Path(args.input_root).expanduser()
    output_root = Path(args.output_root).expanduser()
    filename = args.filename
    lag = int(args.lag)
    llm_finalize = _bool_from_str(args.llm_finalize)
    verbose = not _bool_from_str(args.quiet)
    prefer_time_varying = _bool_from_str(args.prefer_time_varying)

    if not input_root.exists():
        print(f"[ERROR] input-root not found: {input_root}", file=sys.stderr)
        return 2

    _ensure_dir(output_root)

    csv_paths = discover_profiles(input_root, filename)
    if args.max_profiles and args.max_profiles > 0:
        csv_paths = csv_paths[: args.max_profiles]

    if not csv_paths:
        print(f"[WARN] No files named '{filename}' found under: {input_root}", file=sys.stderr)
        return 1

    # reduce KPSS spam globally; we still record per-variable note inside tests
    if HAVE_STATSMODELS and InterpolationWarning is not None:
        warnings.filterwarnings("ignore", category=InterpolationWarning)

    if verbose:
        print(f"[INFO] Found {len(csv_paths)} profile files.")
        print(f"[INFO] Output root: {output_root}")
        print(f"[INFO] Optional deps: scipy={HAVE_SCIPY}, statsmodels={HAVE_STATSMODELS}, openai_new={HAVE_OPENAI_NEW}, openai_legacy={HAVE_OPENAI_LEGACY}")
        print(f"[INFO] LLM finalization enabled: {llm_finalize} (model={LLM_MODEL})")
        print(f"[INFO] Prefer time-varying gVAR when feasible: {prefer_time_varying}")
        if llm_finalize and not os.environ.get("OPENAI_API_KEY"):
            print("[WARN] OPENAI_API_KEY is not set. LLM finalization will fail and fall back to base summaries.")
        print("")

    n_ok = 0
    n_fail = 0

    for i, csv_path in enumerate(csv_paths, start=1):
        try:
            print(f"[{i}/{len(csv_paths)}] Processing: {csv_path}")
            report = analyze_profile(
                csv_path=csv_path,
                output_root=output_root,
                lag=lag,
                llm_finalize=llm_finalize,
                verbose=verbose,
                prefer_time_varying=prefer_time_varying,
            )

            tier = report["overall"]["recommended_tier"]
            variant = report["overall"].get("tier3_variant", None)
            score = report["overall"]["readiness_score_0_100"]
            label = report["overall"]["readiness_label"]

            if variant:
                print(f"    [RESULT] tier={tier} ({variant}) | score={score:.1f}/100 | label={label}")
            else:
                print(f"    [RESULT] tier={tier} | score={score:.1f}/100 | label={label}")

            print(f"    [TECH] {report['overall']['technical_summary']}")
            print(f"    [USER] {report['overall']['client_friendly_summary']}")
            print(f"    [NOTE] {report['overall']['nonstationarity_handling']['statement']}")

            if report.get("llm_finalization", {}).get("enabled") is True:
                if "error" in report["llm_finalization"]:
                    print(f"    [LLM] error={report['llm_finalization']['error']}")
                else:
                    adj = report["llm_finalization"]["applied"]["score_adjustment"]
                    ov = report["llm_finalization"]["applied"]["tier_override"]
                    vov = report["llm_finalization"]["applied"].get("tier3_variant_override")
                    print(f"    [LLM] applied score_adjustment={adj}, tier_override={ov}, tier3_variant_override={vov}")
            print("")

            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f"[ERROR] Failed on {csv_path}: {repr(e)}", file=sys.stderr)

    print(f"\n[DONE] Success: {n_ok}  Failed: {n_fail}")
    return 0 if n_fail == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())

# TODO: later load LLM-generated json-configurated observation plan in this pipeline
