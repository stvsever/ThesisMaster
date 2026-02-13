#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_regular_time_series_analysis.py

Regular (non-network) time-series analysis per pseudoprofile, driven by readiness_report.json.

Scope (THIS script)
- Uses readiness outputs to select analyzable variables (Tier1-ready by default).
- Loads the original raw wide CSV per profile.
- Performs "regular" time-series analytics WITHOUT any visualization:
    * basic descriptives + data quality recap
    * imputation (Kalman/State Space when appropriate + fallbacks)
    * trend / progress metrics (including "regressed end - regressed start" % change)
    * autocorrelation + Ljung-Box (if available)
    * comprehensive timestamp-driven aggregation features:
        - per-weekday means/SD/median/IQR/count/missingness
        - weekday omnibus tests
        - weekday vs weekend split (means/SD/median/IQR/count/missingness + tests)
        - intra-day (hour-of-day) effects if timestamp granularity supports it
        - optional coarse time-of-day bins (night/morning/afternoon/evening)
    * generic nominal-factor effects (if additional nominal columns exist)

Not included (explicitly out-of-scope here)
- Correlation matrix, partial correlations, network estimation
- Time-varying gVAR / dynamic networks (that will be another script)

Outputs per profile (under OUTPUT_ROOT/<profile_id>/)
- regular_ts_report.json
- regular_ts_summary.txt
- regular_ts_variables_overview.csv

Defaults assume your directory layout; override via CLI if needed.

Usage
    python apply_regular_time_series_analysis.py
    python apply_regular_time_series_analysis.py --max-profiles 3
    python apply_regular_time_series_analysis.py --readiness-root ".../00_readiness_check"

Notes on non-stationarity
- This script does NOT treat non-stationarity as a disqualifier.
- It reports stationarity diagnostics and trend/regime metrics, but does not gate analyses.

Author: generated; tune thresholds as needed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# -----------------------------
# Optional dependencies
# -----------------------------
HAVE_SCIPY = False
HAVE_STATSMODELS = False

try:
    import scipy.stats as sps  # type: ignore
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

try:
    import statsmodels.api as sm  # type: ignore
    from statsmodels.stats.diagnostic import acorr_ljungbox  # type: ignore
    from statsmodels.tsa.statespace.structural import UnobservedComponents  # type: ignore
    from statsmodels.tsa.stattools import adfuller, kpss  # type: ignore

    try:
        from statsmodels.tools.sm_exceptions import InterpolationWarning  # type: ignore
    except Exception:
        InterpolationWarning = None  # type: ignore

    HAVE_STATSMODELS = True
except Exception:
    HAVE_STATSMODELS = False
    InterpolationWarning = None  # type: ignore


# -----------------------------
# Defaults / paths
# -----------------------------
def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "Evaluation").exists() and (candidate / "README.md").exists():
            return candidate
    raise RuntimeError("Could not locate repository root from 01_run_regular_ts_analysis.py")


REPO_ROOT = _find_repo_root()

DEFAULT_INPUT_ROOT = str(
    REPO_ROOT / "Evaluation/01_pseudoprofile(s)/time_series_data/pseudodata"
)
DEFAULT_READINESS_ROOT = str(
    REPO_ROOT / "Evaluation/04_initial_observation_analysis/00_readiness_check"
)
DEFAULT_OUTPUT_ROOT = str(
    REPO_ROOT / "Evaluation/04_initial_observation_analysis/01_time_series_analysis/regular"
)
DEFAULT_DATA_FILENAME = "pseudodata_wide.csv"
DEFAULT_READINESS_FILENAME = "readiness_report.json"


# -----------------------------
# Config / thresholds
# -----------------------------
@dataclass
class Config:
    # variable selection
    prefer_tier: str = "tier1"  # "tier1" | "overall_ready" | "all_non_hard"
    min_nonmissing_for_any_analysis: int = 8

    # imputation
    enable_imputation: bool = True
    kalman_min_nonmissing: int = 20
    kalman_max_missing_pct: float = 0.60
    interpolation_max_missing_pct: float = 0.85

    # trend
    trend_min_points: int = 8

    # nominal effects
    nominal_max_levels: int = 12
    nominal_min_per_level: int = 5

    # time aggregation feature thresholds
    weekday_min_total_n: int = 10
    hour_min_total_n: int = 20
    hour_min_per_bin: int = 5

    # significance alpha (reporting only)
    alpha: float = 0.05

    # stationarity tests (reporting only)
    adf_alpha: float = 0.05
    kpss_alpha: float = 0.05


CFG = Config()


# -----------------------------
# Helpers
# -----------------------------
def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


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


def _infer_time_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    cols = [c for c in df.columns]
    lower = {c: str(c).strip().lower() for c in cols}

    # user guarantee: timestamp always present; still infer robustly
    date_candidates = [c for c in cols if lower[c] in {"timestamp", "datetime", "date_time", "time", "date"}]
    time_candidates = [c for c in cols if lower[c] in {"t_index", "t", "time_index", "timeidx", "index"}]

    date_col = date_candidates[0] if date_candidates else None
    time_col = time_candidates[0] if time_candidates else None

    if date_col is None:
        for c in cols:
            if "time" in lower[c] or "date" in lower[c]:
                try:
                    parsed = pd.to_datetime(df[c], errors="coerce")
                    if parsed.notna().mean() > 0.80:
                        date_col = c
                        break
                except Exception:
                    continue

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
    info = {"sorted": False, "sort_key": None, "time_parse_ok_fraction": None, "notes": []}
    if date_col is not None:
        t = pd.to_datetime(df[date_col], errors="coerce")
        ok = float(t.notna().mean())
        info["time_parse_ok_fraction"] = ok
        if ok >= 0.80:
            out = df.copy()
            out["_tmp_dt_"] = t
            out = out.sort_values("_tmp_dt_", kind="mergesort").drop(columns=["_tmp_dt_"])
            info["sorted"] = True
            info["sort_key"] = date_col
            return out, info
        info["notes"].append("Date col present but parsing success <80%; not sorting by date.")
    if time_col is not None:
        t = pd.to_numeric(df[time_col], errors="coerce")
        ok = float(t.notna().mean())
        info["time_parse_ok_fraction"] = ok
        if ok >= 0.80:
            out = df.copy()
            out["_tmp_t_"] = t
            out = out.sort_values("_tmp_t_", kind="mergesort").drop(columns=["_tmp_t_"])
            info["sorted"] = True
            info["sort_key"] = time_col
            return out, info
        info["notes"].append("Numeric time col present but coercion success <80%; not sorting by time.")
    info["notes"].append("No reliable time column; keeping original row order.")
    return df.copy(), info


def _make_time_index(df: pd.DataFrame, time_col: Optional[str], date_col: Optional[str]) -> Tuple[pd.Index, Dict[str, Any]]:
    """
    Returns an index suitable for interpolation + time-based grouping.
    Preference: DatetimeIndex if parseable; else RangeIndex (0..n-1).
    """
    meta = {"index_type": None, "used_col": None, "parse_ok_fraction": None, "notes": []}

    if date_col is not None:
        dt = pd.to_datetime(df[date_col], errors="coerce")
        ok = float(dt.notna().mean())
        meta["parse_ok_fraction"] = ok
        if ok >= 0.80:
            meta["index_type"] = "datetime"
            meta["used_col"] = date_col
            if ok < 1.0:
                meta["notes"].append("Some datetime values unparseable; ffill/bfill for index continuity.")
                dt = dt.fillna(method="ffill").fillna(method="bfill")
            return pd.DatetimeIndex(dt), meta

    if time_col is not None:
        t = pd.to_numeric(df[time_col], errors="coerce")
        ok = float(t.notna().mean())
        meta["parse_ok_fraction"] = ok
        if ok >= 0.80:
            meta["index_type"] = "numeric"
            meta["used_col"] = time_col
            if ok < 1.0:
                meta["notes"].append("Some numeric time values missing/unparseable; falling back to RangeIndex.")
                return pd.RangeIndex(start=0, stop=len(df)), {"index_type": "range", "used_col": None, "parse_ok_fraction": ok, "notes": meta["notes"]}
            return pd.Index(t.astype(float).to_numpy()), meta

    meta["index_type"] = "range"
    return pd.RangeIndex(start=0, stop=len(df)), meta


def _infer_var_type_from_readiness(readiness: Dict[str, Any], var: str) -> str:
    try:
        return str(readiness.get("variables", {}).get(var, {}).get("type_inferred", "unknown"))
    except Exception:
        return "unknown"


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in cols:
        out[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return out


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


def _robust_mad(x: np.ndarray) -> float:
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def _series_summary_stats(x: np.ndarray) -> Dict[str, Any]:
    """
    Summary stats for a numeric vector (already filtered to non-NaN).
    """
    n = int(x.size)
    if n == 0:
        return {
            "n": 0, "mean": None, "std": None, "median": None, "min": None, "max": None,
            "iqr": None, "mad": None
        }
    q25 = float(np.quantile(x, 0.25)) if n >= 4 else float(np.min(x))
    q75 = float(np.quantile(x, 0.75)) if n >= 4 else float(np.max(x))
    iqr = float(q75 - q25)
    return {
        "n": n,
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if n >= 2 else None,
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "iqr": iqr,
        "mad": float(_robust_mad(x)),
    }


def _stationarity_tests(x: np.ndarray, adf_alpha: float, kpss_alpha: float) -> Dict[str, Any]:
    out = {"adf_p": None, "kpss_p": None, "stationary_flag": None, "notes": []}
    if not HAVE_STATSMODELS or x.size < 15:
        out["notes"].append("Stationarity tests skipped (need statsmodels and >=15 points).")
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
                out["notes"].append("KPSS warning: p-value outside lookup table; p is approximate/bounded.")
    except Exception as e:
        out["notes"].append(f"KPSS failed: {repr(e)}")

    ap = out["adf_p"]
    kp = out["kpss_p"]
    if ap is not None and kp is not None:
        out["stationary_flag"] = bool((ap < adf_alpha) and (kp >= kpss_alpha))
    elif ap is not None:
        out["stationary_flag"] = bool(ap < adf_alpha)
        out["notes"].append("Stationarity decided using ADF only.")
    elif kp is not None:
        out["stationary_flag"] = bool(kp >= kpss_alpha)
        out["notes"].append("Stationarity decided using KPSS only.")
    return out


def _ljung_box_p(x: np.ndarray, max_lag: int = 10) -> Optional[float]:
    if not HAVE_STATSMODELS or x.size < (max_lag + 8):
        return None
    try:
        lag = int(min(max_lag, x.size - 1))
        df_lb = acorr_ljungbox(x, lags=[lag], return_df=True)
        return float(df_lb["lb_pvalue"].iloc[0])
    except Exception:
        return None


# -----------------------------
# Imputation
# -----------------------------
def _kalman_impute_ucm(s: pd.Series, var_type: str) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Kalman/state-space imputation via statsmodels UnobservedComponents.
    """
    meta: Dict[str, Any] = {"method": "kalman_ucm", "ok": False, "notes": []}
    if not HAVE_STATSMODELS:
        meta["notes"].append("statsmodels not available; cannot run Kalman imputation.")
        return s.copy(), meta

    y = s.astype(float).copy()

    back_transform = None
    clip_low = None
    clip_high = None

    if var_type == "count":
        y = np.log1p(y)
        back_transform = lambda z: np.expm1(z)
        clip_low = 0.0
        meta["notes"].append("Applied log1p for count; expm1 back-transform after smoothing.")
    elif var_type == "proportion":
        clip_low, clip_high = 0.0, 1.0
        meta["notes"].append("Proportion: kept scale; clip to [0,1] after smoothing.")

    try:
        model = UnobservedComponents(y, level="local linear trend")
        res = model.fit(disp=False)
        smoothed = res.get_prediction().predicted_mean

        out = y.copy()
        miss = out.isna()
        out.loc[miss] = smoothed.loc[miss]

        if back_transform is not None:
            out = back_transform(out)
        if clip_low is not None:
            out = out.clip(lower=clip_low)
        if clip_high is not None:
            out = out.clip(upper=clip_high)

        meta["ok"] = True
        meta["aic"] = float(res.aic) if hasattr(res, "aic") else None
        return out.astype(float), meta
    except Exception as e:
        meta["notes"].append(f"UCM fit failed: {repr(e)}")
        return s.copy(), meta


def _interpolate_impute(s: pd.Series, time_index: pd.Index) -> Tuple[pd.Series, Dict[str, Any]]:
    meta: Dict[str, Any] = {"method": "interpolate", "ok": False, "notes": []}
    out = s.astype(float).copy()
    try:
        tmp = out.copy()
        tmp.index = time_index
        if isinstance(time_index, pd.DatetimeIndex):
            tmp2 = tmp.interpolate(method="time", limit_direction="both")
            meta["notes"].append("Used time interpolation.")
        else:
            tmp2 = tmp.interpolate(method="linear", limit_direction="both")
            meta["notes"].append("Used linear interpolation.")
        out2 = tmp2.reset_index(drop=True)
        meta["ok"] = True
        return out2.astype(float), meta
    except Exception as e:
        meta["notes"].append(f"Interpolation failed: {repr(e)}")
        return out, meta


def _binary_fill_impute(s: pd.Series) -> Tuple[pd.Series, Dict[str, Any]]:
    meta: Dict[str, Any] = {"method": "binary_ffill_bfill", "ok": False, "notes": []}
    out = s.copy()
    try:
        out2 = out.fillna(method="ffill").fillna(method="bfill")
        meta["ok"] = True
        meta["notes"].append("Binary: ffill then bfill.")
        return out2.astype(float), meta
    except Exception as e:
        meta["notes"].append(f"Binary fill failed: {repr(e)}")
        return out.astype(float), meta


def impute_series(
    s: pd.Series,
    var_type: str,
    readiness_time_regularity: Optional[str],
    time_index: pd.Index,
) -> Tuple[pd.Series, Dict[str, Any]]:
    meta: Dict[str, Any] = {"enabled": CFG.enable_imputation, "chosen": None, "notes": []}
    if not CFG.enable_imputation:
        meta["chosen"] = "none"
        return s.astype(float).copy(), meta

    n_total = int(len(s))
    n_nonmissing = int(s.notna().sum())
    miss_pct = float(1.0 - (n_nonmissing / max(1, n_total)))

    if n_nonmissing < CFG.min_nonmissing_for_any_analysis:
        meta["chosen"] = "none"
        meta["notes"].append("Too few non-missing values for imputation.")
        return s.astype(float).copy(), meta

    if miss_pct <= 0.02:
        meta["chosen"] = "none"
        meta["notes"].append("Missingness very low; no imputation.")
        return s.astype(float).copy(), meta

    if miss_pct > CFG.interpolation_max_missing_pct:
        meta["chosen"] = "none"
        meta["notes"].append("Missingness extremely high; skip imputation.")
        return s.astype(float).copy(), meta

    if var_type == "binary":
        imp, m = _binary_fill_impute(s)
        meta["chosen"] = m["method"]
        meta["notes"].extend(m.get("notes", []))
        meta["ok"] = m.get("ok", False)
        return imp, meta

    regular_ok = readiness_time_regularity in {"regular_or_quasi_regular", None}
    moderate_ok = readiness_time_regularity == "moderately_irregular"

    if HAVE_STATSMODELS and (regular_ok or moderate_ok):
        if n_nonmissing >= CFG.kalman_min_nonmissing and miss_pct <= CFG.kalman_max_missing_pct:
            imp, m = _kalman_impute_ucm(s, var_type=var_type)
            if m.get("ok"):
                meta["chosen"] = m["method"]
                meta["ok"] = True
                meta["notes"].extend(m.get("notes", []))
                return imp, meta
            meta["notes"].append("Kalman attempted but failed; fallback to interpolation.")

    imp, m = _interpolate_impute(s, time_index=time_index)
    meta["chosen"] = m["method"]
    meta["ok"] = m.get("ok", False)
    meta["notes"].extend(m.get("notes", []))
    return imp, meta


# -----------------------------
# Trend + progress
# -----------------------------
def _ols_trend(y: np.ndarray) -> Dict[str, Any]:
    n = int(y.size)
    t = np.arange(n, dtype=float)
    out: Dict[str, Any] = {"model": "ols", "n": n, "slope": None, "intercept": None, "p_slope": None, "r2": None, "notes": []}
    if n < 3:
        out["notes"].append("Too few points for OLS.")
        return out

    if HAVE_STATSMODELS:
        try:
            X = sm.add_constant(t)
            m = sm.OLS(y, X, missing="drop")
            res = m.fit(cov_type="HC3")
            out["intercept"] = float(res.params[0])
            out["slope"] = float(res.params[1])
            out["p_slope"] = float(res.pvalues[1])
            out["r2"] = float(res.rsquared)
            return out
        except Exception as e:
            out["notes"].append(f"statsmodels OLS failed; fallback to scipy. err={repr(e)}")

    if HAVE_SCIPY:
        try:
            lr = sps.linregress(t, y)
            out["intercept"] = float(lr.intercept)
            out["slope"] = float(lr.slope)
            out["p_slope"] = float(lr.pvalue)
            out["r2"] = float(lr.rvalue ** 2)
            out["notes"].append("Used scipy linregress (non-robust).")
            return out
        except Exception as e:
            out["notes"].append(f"scipy linregress failed: {repr(e)}")
    return out


def _logistic_trend_binary(y: np.ndarray) -> Dict[str, Any]:
    n = int(y.size)
    t = np.arange(n, dtype=float)
    out: Dict[str, Any] = {"model": "logit", "n": n, "beta_time": None, "or_per_step": None, "p_time": None, "notes": []}
    if n < 8:
        out["notes"].append("Too few points for logistic trend.")
        return out
    if not HAVE_STATSMODELS:
        out["notes"].append("statsmodels not available; logistic trend skipped.")
        return out

    try:
        X = sm.add_constant(t)
        m = sm.GLM(y, X, family=sm.families.Binomial())
        res = m.fit()
        beta = float(res.params[1])
        out["beta_time"] = beta
        out["or_per_step"] = float(math.exp(beta))
        out["p_time"] = float(res.pvalues[1])
        return out
    except Exception as e:
        out["notes"].append(f"Logistic trend failed: {repr(e)}")
        return out


def _progress_metrics_from_trend(trend: Dict[str, Any], n_points: int, y_scale_ref: float) -> Dict[str, Any]:
    out = {
        "regressed_start": None,
        "regressed_end": None,
        "regressed_delta": None,
        "regressed_pct_change_vs_scale": None,
        "scale_reference": y_scale_ref,
    }
    slope = _safe_float(trend.get("slope"))
    intercept = _safe_float(trend.get("intercept"))
    if slope is None or intercept is None or n_points <= 1:
        return out
    y0 = intercept + slope * 0.0
    y1 = intercept + slope * float(n_points - 1)
    delta = y1 - y0
    out["regressed_start"] = float(y0)
    out["regressed_end"] = float(y1)
    out["regressed_delta"] = float(delta)
    if y_scale_ref is not None and abs(y_scale_ref) > 1e-12:
        out["regressed_pct_change_vs_scale"] = float(100.0 * (delta / y_scale_ref))
    return out


# -----------------------------
# Timestamp-driven aggregation features
# -----------------------------
def _weekday_labels_nl() -> Dict[int, str]:
    return {0: "monday", 1: "tuesday", 2: "wednesday", 3: "thursday", 4: "friday", 5: "saturday", 6: "sunday"}


def _time_of_day_bin(hour: int) -> str:
    # coarse bins (customize if you want)
    if 0 <= hour <= 5:
        return "Night"
    if 6 <= hour <= 11:
        return "Morning"
    if 12 <= hour <= 17:
        return "Afternoon"
    return "Evening"


def _group_summary_with_missingness(series: pd.Series, group: pd.Series) -> Dict[str, Any]:
    """
    Returns group-wise summary including missingness metrics, using original series.
    """
    tmp = pd.DataFrame({"y": series, "g": group})
    out: Dict[str, Any] = {}

    # counts include missing rows too (to get missing rate per group)
    counts_all = tmp.groupby("g")["y"].size()
    counts_nonmiss = tmp.groupby("g")["y"].apply(lambda x: int(x.notna().sum()))
    miss_rate = (counts_all - counts_nonmiss) / counts_all.replace(0, np.nan)

    # stats on non-missing only
    stats = tmp.dropna(subset=["y"]).groupby("g")["y"].apply(lambda x: pd.Series(_series_summary_stats(x.to_numpy(dtype=float))))

    # assemble
    for gval in counts_all.index.tolist():
        gk = str(gval)
        out[gk] = {
            "n_total": int(counts_all.loc[gval]),
            "n_nonmissing": int(counts_nonmiss.loc[gval]),
            "missing_rate": float(miss_rate.loc[gval]) if pd.notna(miss_rate.loc[gval]) else None,
        }
        if gval in stats.index:
            # stats is multiindex-like (g, statname)
            for stat_name in ["mean", "std", "median", "min", "max", "iqr", "mad", "n"]:
                try:
                    val = stats.loc[gval, stat_name]
                    out[gk][stat_name] = float(val) if pd.notna(val) else None
                except Exception:
                    pass

    return out


def _omnibus_test_groups(groups: List[np.ndarray]) -> Dict[str, Any]:
    """
    Omnibus test for k groups. Chooses ANOVA vs Kruskal by quick assumptions.
    Returns test, statistic, p_value, effect_size.
    """
    out: Dict[str, Any] = {"test": None, "statistic": None, "p_value": None, "effect_size": None, "notes": []}
    if not HAVE_SCIPY:
        out["notes"].append("scipy not available.")
        return out

    k = len(groups)
    if k < 2 or min([g.size for g in groups]) < 2:
        out["notes"].append("Not enough groups/points.")
        return out

    normal_ok = True
    try:
        for g in groups:
            if g.size >= 3 and g.size <= 5000:
                _, psh = sps.shapiro(g)
                if psh < 0.05:
                    normal_ok = False
                    break
            else:
                normal_ok = False
                break
    except Exception:
        normal_ok = False

    hom_ok = True
    try:
        _, plev = sps.levene(*groups)
        if plev < 0.05:
            hom_ok = False
    except Exception:
        hom_ok = False

    n = int(sum(g.size for g in groups))

    if normal_ok and hom_ok:
        stat, p = sps.f_oneway(*groups)
        out["test"] = "ANOVA"
        out["statistic"] = float(stat)
        out["p_value"] = float(p)
        # eta^2 approximate from H-stat not; do variance decomposition externally? keep simple:
        out["effect_size"] = {"eta_squared": None}
        out["notes"].append("ANOVA used (assumption checks passed).")
        return out

    stat, p = sps.kruskal(*groups)
    out["test"] = "Kruskal-Wallis"
    out["statistic"] = float(stat)
    out["p_value"] = float(p)
    eps2 = float((stat - k + 1) / (n - k)) if (n - k) > 0 else None
    out["effect_size"] = {"epsilon_squared": eps2}
    out["notes"].append("Kruskal-Wallis used (assumptions not met).")
    return out


def _two_group_test(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    """
    Two-group test + effect size. Chooses Welch t vs Mann-Whitney.
    """
    out: Dict[str, Any] = {"test": None, "statistic": None, "p_value": None, "effect_size": None, "notes": []}
    if not HAVE_SCIPY:
        out["notes"].append("scipy not available.")
        return out
    if a.size < 2 or b.size < 2:
        out["notes"].append("Not enough points.")
        return out

    def _norm_ok(x: np.ndarray) -> bool:
        if x.size < 3 or x.size > 5000:
            return False
        try:
            _, psh = sps.shapiro(x)
            return bool(psh >= 0.05)
        except Exception:
            return False

    if _norm_ok(a) and _norm_ok(b):
        stat, p = sps.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        out["test"] = "Welch_t"
        out["statistic"] = float(stat)
        out["p_value"] = float(p)
        # Cohen's d
        sa = float(np.var(a, ddof=1))
        sb = float(np.var(b, ddof=1))
        sp = math.sqrt(((a.size - 1) * sa + (b.size - 1) * sb) / max(1.0, (a.size + b.size - 2)))
        d = float((np.mean(b) - np.mean(a)) / sp) if sp > 1e-12 else None
        out["effect_size"] = {"cohens_d_b_minus_a": d}
        return out

    stat, p = sps.mannwhitneyu(a, b, alternative="two-sided")
    out["test"] = "Mann_Whitney_U"
    out["statistic"] = float(stat)
    out["p_value"] = float(p)
    rbc = float(1.0 - (2.0 * float(stat)) / (a.size * b.size)) if (a.size * b.size) > 0 else None
    out["effect_size"] = {"rank_biserial_b_minus_a": rbc}
    return out


def _timestamp_feature_block(
    df_num: pd.DataFrame,
    dt_index: pd.DatetimeIndex,
    variables: List[str],
) -> Dict[str, Any]:
    """
    Produces comprehensive timestamp-driven features:
    - weekday group summaries + omnibus test
    - weekend split summaries + two-group test
    - hour-of-day summaries + omnibus test (if sufficient support)
    - time-of-day bin summaries + omnibus test
    """
    out: Dict[str, Any] = {"enabled": True, "notes": [], "weekday": {}, "weekend_split": {}, "hour_of_day": {}, "time_of_day_bins": {}}

    if not isinstance(dt_index, pd.DatetimeIndex):
        out["enabled"] = False
        out["notes"].append("No DatetimeIndex; timestamp features skipped.")
        return out

    wd = pd.Series(dt_index.weekday, index=df_num.index).map(_weekday_labels_nl())
    wk = pd.Series(dt_index.weekday, index=df_num.index).isin([5, 6]).map({True: "Weekend", False: "Weekday"})
    hour = pd.Series(dt_index.hour, index=df_num.index)
    tod = hour.map(_time_of_day_bin)

    # --- weekday and weekend split for each variable ---
    for v in variables:
        s = df_num[v]

        # weekday summaries
        tmp = pd.DataFrame({"y": s, "wd": wd, "wk": wk, "hour": hour, "tod": tod})
        if tmp["y"].notna().sum() < CFG.weekday_min_total_n:
            continue

        out["weekday"][v] = {
            "group_summaries": _group_summary_with_missingness(series=s, group=wd),
        }
        # weekday omnibus
        if HAVE_SCIPY:
            g_nonmiss = tmp.dropna(subset=["y"]).groupby("wd")["y"].apply(lambda x: x.to_numpy(dtype=float)).tolist()
            if len(g_nonmiss) >= 2 and min([g.size for g in g_nonmiss]) >= 2:
                out["weekday"][v]["omnibus_test"] = _omnibus_test_groups(g_nonmiss)

        # weekend split summaries + test
        out["weekend_split"][v] = {
            "group_summaries": _group_summary_with_missingness(series=s, group=wk),
        }
        if HAVE_SCIPY:
            a = tmp[tmp["wk"] == "Weekday"]["y"].dropna().to_numpy(dtype=float)
            b = tmp[tmp["wk"] == "Weekend"]["y"].dropna().to_numpy(dtype=float)
            if a.size >= 2 and b.size >= 2:
                out["weekend_split"][v]["two_group_test"] = _two_group_test(a, b)

        # hour-of-day summaries (only if enough observations and multiple hours present)
        if tmp["hour"].notna().sum() >= CFG.hour_min_total_n:
            hour_levels = tmp["hour"].dropna().unique().tolist()
            if len(hour_levels) >= 3:
                # keep only hours with enough non-missing per bin
                counts = tmp.dropna(subset=["y"]).groupby("hour")["y"].size()
                ok_hours = counts[counts >= CFG.hour_min_per_bin].index.tolist()
                if len(ok_hours) >= 3:
                    tmp_h = tmp[tmp["hour"].isin(ok_hours)]
                    out["hour_of_day"][v] = {
                        "group_summaries": _group_summary_with_missingness(series=tmp_h["y"], group=tmp_h["hour"]),
                    }
                    g2 = tmp_h.dropna(subset=["y"]).groupby("hour")["y"].apply(lambda x: x.to_numpy(dtype=float)).tolist()
                    if len(g2) >= 2 and min([g.size for g in g2]) >= 2:
                        out["hour_of_day"][v]["omnibus_test"] = _omnibus_test_groups(g2)

        # time-of-day bins summaries (4 bins)
        counts_tod = tmp.dropna(subset=["y"]).groupby("tod")["y"].size()
        ok_tod = counts_tod[counts_tod >= CFG.hour_min_per_bin].index.tolist()
        if len(ok_tod) >= 2:
            tmp_t = tmp[tmp["tod"].isin(ok_tod)]
            out["time_of_day_bins"][v] = {
                "group_summaries": _group_summary_with_missingness(series=tmp_t["y"], group=tmp_t["tod"]),
            }
            g3 = tmp_t.dropna(subset=["y"]).groupby("tod")["y"].apply(lambda x: x.to_numpy(dtype=float)).tolist()
            if len(g3) >= 2 and min([g.size for g in g3]) >= 2:
                out["time_of_day_bins"][v]["omnibus_test"] = _omnibus_test_groups(g3)

    return out


# -----------------------------
# Nominal effects (non-time factors)
# -----------------------------
def _detect_nominal_columns(df: pd.DataFrame, exclude: set) -> List[str]:
    nominals: List[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if str(c).startswith(("P", "C")):
            continue
        s = df[c]
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s):
            nu = int(s.dropna().nunique())
            if 2 <= nu <= CFG.nominal_max_levels:
                nominals.append(c)
            continue
        if pd.api.types.is_numeric_dtype(s):
            nu = int(pd.Series(s).dropna().nunique())
            if 2 <= nu <= min(6, CFG.nominal_max_levels):
                nominals.append(c)
    return nominals


def _nominal_effects(
    df_raw: pd.DataFrame,
    df_num: pd.DataFrame,
    nominal_cols: List[str],
    variables: List[str],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"enabled": True, "notes": [], "factors": {}}
    if not HAVE_SCIPY:
        out["enabled"] = False
        out["notes"].append("scipy not available; nominal tests skipped.")
        return out

    for fac in nominal_cols:
        s_fac = df_raw[fac]
        levels = [lv for lv in s_fac.dropna().unique().tolist() if lv is not None]
        levels = levels[: CFG.nominal_max_levels]
        if len(levels) < 2:
            continue

        fac_block: Dict[str, Any] = {"levels": [str(x) for x in levels], "variables": {}}

        for v in variables:
            tmp = pd.DataFrame({"y": df_num[v], "fac": s_fac}).dropna()
            if tmp.shape[0] < (2 * CFG.nominal_min_per_level):
                continue

            counts = tmp.groupby("fac")["y"].size()
            ok_levels = counts[counts >= CFG.nominal_min_per_level].index.tolist()
            if len(ok_levels) < 2:
                continue
            tmp = tmp[tmp["fac"].isin(ok_levels)]

            fac_block["variables"][v] = {
                "level_summaries": _group_summary_with_missingness(series=tmp["y"], group=tmp["fac"]),
            }

            # omnibus test
            groups = tmp.groupby("fac")["y"].apply(lambda x: x.to_numpy(dtype=float)).tolist()
            if len(groups) >= 2 and min([g.size for g in groups]) >= 2:
                fac_block["variables"][v]["omnibus_test"] = _omnibus_test_groups(groups)

        if fac_block["variables"]:
            out["factors"][fac] = fac_block

    return out


# -----------------------------
# Variable selection
# -----------------------------
def _select_analysis_variables(readiness: Dict[str, Any], df_raw: pd.DataFrame) -> Tuple[List[str], Dict[str, Any]]:
    info: Dict[str, Any] = {"strategy": CFG.prefer_tier, "notes": []}
    vars_meta = readiness.get("variables", {}) or {}
    all_in_df = [c for c in df_raw.columns if c in vars_meta]

    chosen: List[str] = []
    if CFG.prefer_tier == "overall_ready":
        chosen = list(readiness.get("overall", {}).get("ready_variables", []) or [])
    elif CFG.prefer_tier == "tier1":
        chosen = list(readiness.get("tiers", {}).get("tier1", {}).get("ready_variables", []) or [])
    else:
        for c in all_in_df:
            if (vars_meta.get(c, {}).get("drop_hard_reasons") or []):
                continue
            chosen.append(c)

    chosen = [c for c in chosen if c in df_raw.columns]
    if not chosen:
        info["notes"].append("No variables from preferred selection; fallback to non-hard-drop with enough data.")
        for c in df_raw.columns:
            if c in vars_meta and not (vars_meta.get(c, {}).get("drop_hard_reasons") or []):
                chosen.append(c)
        chosen = [c for c in chosen if c in df_raw.columns]

    chosen2: List[str] = []
    for c in chosen:
        nn = int(pd.to_numeric(df_raw[c], errors="coerce").notna().sum())
        if nn >= CFG.min_nonmissing_for_any_analysis:
            chosen2.append(c)
    if len(chosen2) < len(chosen):
        info["notes"].append(f"Removed {len(chosen)-len(chosen2)} variables due to insufficient non-missing values.")
    return chosen2, info


# -----------------------------
# Per-profile analysis
# -----------------------------
def analyze_profile(
    readiness_path: Path,
    input_root: Path,
    output_root: Path,
    data_filename: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    readiness = _read_json(readiness_path)
    profile_id = str(readiness.get("meta", {}).get("profile_id", readiness_path.parent.name))

    # resolve raw CSV path (prefer readiness meta)
    raw_csv_str = readiness.get("meta", {}).get("input_file")
    raw_csv_path = Path(raw_csv_str) if raw_csv_str else (input_root / profile_id / data_filename)
    if not raw_csv_path.exists():
        raw_csv_path = input_root / profile_id / data_filename
    if not raw_csv_path.exists():
        raise FileNotFoundError(f"Raw CSV not found for {profile_id}: tried {raw_csv_str} and {raw_csv_path}")

    if verbose:
        print(f"    [LOAD] raw={raw_csv_path}")

    df_raw = _read_csv_robust(raw_csv_path)
    df_raw.columns = [str(c).strip() for c in df_raw.columns]

    # time columns: prefer readiness time_info; infer as fallback
    time_info = readiness.get("dataset_overview", {}).get("time_info", {}) or {}
    time_col = time_info.get("time_col") or None
    date_col = time_info.get("date_col") or None

    if date_col not in df_raw.columns:
        _, d2 = _infer_time_columns(df_raw)
        date_col = date_col if (date_col in df_raw.columns) else d2
    if time_col is not None and time_col not in df_raw.columns:
        t2, _ = _infer_time_columns(df_raw)
        time_col = t2

    if date_col is None:
        # user guarantee says it exists; still fail loudly if not found
        raise ValueError(f"{profile_id}: could not infer timestamp column. Ensure a datetime/timestamp column exists.")

    df_sorted, ordering_info = _sort_by_time(df_raw, time_col=time_col, date_col=date_col)
    time_index, time_index_meta = _make_time_index(df_sorted, time_col=time_col, date_col=date_col)

    # enforce timestamp present: require datetime index
    if not isinstance(time_index, pd.DatetimeIndex):
        raise ValueError(f"{profile_id}: timestamp present but could not form a DatetimeIndex (parse failure?). date_col={date_col}")

    readiness_regularity = time_info.get("regularity") or readiness.get("dataset_overview", {}).get("time_info", {}).get("regularity")

    exclude_cols = set([c for c in [time_col, date_col] if c is not None])

    vars_to_analyze, selection_info = _select_analysis_variables(readiness, df_sorted)
    if verbose:
        print(f"    [VARS] selected={len(vars_to_analyze)} (strategy={selection_info.get('strategy')})")

    df_num = _coerce_numeric(df_sorted, vars_to_analyze)

    # nominal columns (excluding time columns)
    nominal_cols = _detect_nominal_columns(df_sorted, exclude=exclude_cols)

    # per-variable analysis
    var_results: Dict[str, Any] = {}
    overview_rows: List[Dict[str, Any]] = []

    for v in vars_to_analyze:
        s_raw = df_num[v]
        var_type = _infer_var_type_from_readiness(readiness, v)

        n_total = int(len(s_raw))
        n_nonmissing = int(s_raw.notna().sum())
        miss_pct = float(1.0 - (n_nonmissing / max(1, n_total)))
        streak = _longest_missing_streak(s_raw)

        s_imp, imp_meta = impute_series(
            s_raw,
            var_type=var_type,
            readiness_time_regularity=readiness_regularity,
            time_index=time_index,
        )

        use_imp = bool(imp_meta.get("ok")) and (s_imp.notna().sum() >= s_raw.notna().sum())
        s_use = s_imp if use_imp else s_raw

        x = s_use.dropna().to_numpy(dtype=float)
        n_use = int(x.size)

        desc = _series_summary_stats(x)
        if HAVE_SCIPY and n_use >= 8:
            desc["skew"] = float(sps.skew(x, nan_policy="omit"))
            desc["kurtosis_fisher"] = float(sps.kurtosis(x, nan_policy="omit", fisher=True))
        else:
            desc["skew"] = None
            desc["kurtosis_fisher"] = None

        ac1 = None
        try:
            if n_use >= 3:
                ac1 = float(pd.Series(x).autocorr(lag=1))
        except Exception:
            ac1 = None

        lb_p = _ljung_box_p(x, max_lag=min(10, max(2, n_use // 6)))
        st = _stationarity_tests(x, adf_alpha=CFG.adf_alpha, kpss_alpha=CFG.kpss_alpha)

        if n_use >= CFG.trend_min_points:
            if var_type == "binary":
                yb = np.clip(np.round(x), 0, 1)
                trend = _logistic_trend_binary(yb)
            else:
                trend = _ols_trend(x)
        else:
            trend = {"notes": ["Too few points for trend analysis."], "n": n_use}

        # scale ref for percent change
        if n_use:
            q25 = float(np.quantile(x, 0.25)) if n_use >= 4 else float(np.min(x))
            q75 = float(np.quantile(x, 0.75)) if n_use >= 4 else float(np.max(x))
            iqr = float(q75 - q25)
            rng = float(np.max(x) - np.min(x))
            med_abs = float(abs(np.median(x)))
            scale_ref = max(med_abs, iqr, rng, 1e-12)
        else:
            scale_ref = 1.0

        progress = _progress_metrics_from_trend(trend, n_points=n_use, y_scale_ref=float(scale_ref))

        raw_first = None
        raw_last = None
        try:
            sr = s_use.dropna()
            if len(sr) >= 2:
                raw_first = float(sr.iloc[0])
                raw_last = float(sr.iloc[-1])
        except Exception:
            raw_first, raw_last = None, None

        raw_delta = (raw_last - raw_first) if (raw_first is not None and raw_last is not None) else None
        raw_pct_vs_scale = (100.0 * raw_delta / scale_ref) if (raw_delta is not None and scale_ref > 1e-12) else None

        var_results[v] = {
            "type_inferred": var_type,
            "n_total": n_total,
            "n_nonmissing_raw": n_nonmissing,
            "missing_pct_raw": miss_pct,
            "max_consecutive_missing_raw": streak,
            "imputation": {"used": bool(use_imp), "details": imp_meta},
            "n_used_for_analysis": n_use,
            "descriptives": desc,
            "autocorrelation": {"autocorr_lag1": ac1, "ljung_box_p": lb_p},
            "stationarity": st,
            "trend": trend,
            "progress": {
                "regressed": progress,
                "raw_first": raw_first,
                "raw_last": raw_last,
                "raw_delta": raw_delta,
                "raw_pct_change_vs_scale": raw_pct_vs_scale,
                "scale_reference": scale_ref,
            },
        }

        overview_rows.append(
            {
                "variable": v,
                "type_inferred": var_type,
                "n_total": n_total,
                "n_nonmissing_raw": n_nonmissing,
                "missing_pct_raw": miss_pct,
                "imputed_used": bool(use_imp),
                "trend_slope_or_beta": float(trend.get("slope")) if trend.get("slope") is not None else (float(trend.get("beta_time")) if trend.get("beta_time") is not None else np.nan),
                "trend_p": float(trend.get("p_slope")) if trend.get("p_slope") is not None else (float(trend.get("p_time")) if trend.get("p_time") is not None else np.nan),
                "regressed_pct_change_vs_scale": float(progress.get("regressed_pct_change_vs_scale")) if progress.get("regressed_pct_change_vs_scale") is not None else np.nan,
                "autocorr_lag1": ac1 if ac1 is not None else np.nan,
                "ljung_box_p": lb_p if lb_p is not None else np.nan,
                "adf_p": st.get("adf_p") if st.get("adf_p") is not None else np.nan,
                "kpss_p": st.get("kpss_p") if st.get("kpss_p") is not None else np.nan,
            }
        )

    # timestamp-driven feature block (COMPREHENSIVE)
    if verbose:
        print("    [TIME] Computing comprehensive timestamp aggregation features…")
    time_features = _timestamp_feature_block(df_num=df_num, dt_index=time_index, variables=vars_to_analyze)

    # nominal-factor effects (non-time factors)
    nominal_block = None
    if nominal_cols and len(vars_to_analyze) > 0:
        if verbose:
            print(f"    [NOM] Nominal factors detected: {len(nominal_cols)} — computing summaries + omnibus tests…")
        nominal_block = _nominal_effects(df_raw=df_sorted, df_num=df_num, nominal_cols=nominal_cols, variables=vars_to_analyze)

    # summary highlights
    ov = pd.DataFrame(overview_rows)
    top_trends: List[str] = []
    top_prog: List[str] = []

    if not ov.empty:
        try:
            ov2 = ov.copy()
            ov2["abs_regressed_pct"] = ov2["regressed_pct_change_vs_scale"].abs()
            top_prog_vars = ov2.sort_values("abs_regressed_pct", ascending=False).head(5)["variable"].tolist()
            for v in top_prog_vars:
                val = float(ov2[ov2["variable"] == v]["regressed_pct_change_vs_scale"].iloc[0])
                top_prog.append(f"{v}: {val:.1f}% (regressed vs scale)")
        except Exception:
            pass

        try:
            ov2 = ov.copy()
            ov2["abs_trend"] = ov2["trend_slope_or_beta"].abs()
            top_trend_vars = ov2.sort_values("abs_trend", ascending=False).head(5)["variable"].tolist()
            for v in top_trend_vars:
                val = float(ov2[ov2["variable"] == v]["trend_slope_or_beta"].iloc[0])
                p = float(ov2[ov2["variable"] == v]["trend_p"].iloc[0])
                top_trends.append(f"{v}: coef={val:.4g}, p={p:.3g}")
        except Exception:
            pass

    # collect significant time effects (weekday + weekend + hour + tod)
    sig_time_effects: Dict[str, List[str]] = {"weekday": [], "weekend_split": [], "hour_of_day": [], "time_of_day_bins": []}
    if time_features.get("enabled"):
        for v, blk in (time_features.get("weekday") or {}).items():
            p = _safe_float((blk.get("omnibus_test") or {}).get("p_value"))
            if p is not None and p < CFG.alpha:
                sig_time_effects["weekday"].append(f"{v} (p={p:.3g}, {blk['omnibus_test'].get('test')})")
        for v, blk in (time_features.get("weekend_split") or {}).items():
            p = _safe_float((blk.get("two_group_test") or {}).get("p_value"))
            if p is not None and p < CFG.alpha:
                sig_time_effects["weekend_split"].append(f"{v} (p={p:.3g}, {blk['two_group_test'].get('test')})")
        for v, blk in (time_features.get("hour_of_day") or {}).items():
            p = _safe_float((blk.get("omnibus_test") or {}).get("p_value"))
            if p is not None and p < CFG.alpha:
                sig_time_effects["hour_of_day"].append(f"{v} (p={p:.3g}, {blk['omnibus_test'].get('test')})")
        for v, blk in (time_features.get("time_of_day_bins") or {}).items():
            p = _safe_float((blk.get("omnibus_test") or {}).get("p_value"))
            if p is not None and p < CFG.alpha:
                sig_time_effects["time_of_day_bins"].append(f"{v} (p={p:.3g}, {blk['omnibus_test'].get('test')})")

    report: Dict[str, Any] = {
        "meta": {
            "profile_id": profile_id,
            "generated_at": _now_iso(),
            "script": "apply_regular_time_series_analysis.py",
            "paths": {"readiness_report": str(readiness_path), "raw_csv": str(raw_csv_path)},
            "dependencies": {"scipy": HAVE_SCIPY, "statsmodels": HAVE_STATSMODELS},
            "config": _jsonify(CFG.__dict__),
        },
        "inputs_from_readiness": {
            "recommended_tier_from_readiness": readiness.get("overall", {}).get("recommended_tier"),
            "readiness_score_from_readiness": readiness.get("overall", {}).get("readiness_score_0_100"),
            "time_regularity_from_readiness": readiness.get("dataset_overview", {}).get("time_info", {}).get("regularity"),
            "selection_info": selection_info,
        },
        "dataset_overview": {
            "n_rows": int(df_sorted.shape[0]),
            "n_cols": int(df_sorted.shape[1]),
            "time_col": time_col,
            "date_col": date_col,
            "ordering": ordering_info,
            "time_index": time_index_meta,
            "nominal_columns_detected": nominal_cols,
        },
        "variables_analyzed": vars_to_analyze,
        "variable_results": var_results,
        "timestamp_feature_analysis": time_features,
        "nominal_factor_effects": nominal_block if nominal_block is not None else {"enabled": False, "notes": ["No nominal factors detected (or scipy unavailable)."]},
        "summary_highlights": {
            "top_trends": top_trends,
            "top_progress_regressed_pct": top_prog,
            "significant_time_effects": {k: v[:12] for k, v in sig_time_effects.items()},
        },
    }

    # write outputs
    out_dir = output_root / profile_id
    _ensure_dir(out_dir)

    json_path = out_dir / "regular_ts_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_jsonify(report), f, indent=2, ensure_ascii=False)

    overview_path = out_dir / "regular_ts_variables_overview.csv"
    if overview_rows:
        pd.DataFrame(overview_rows).to_csv(overview_path, index=False)

    summary_path = out_dir / "regular_ts_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Profile: {profile_id}\n")
        f.write(f"Generated: {report['meta']['generated_at']}\n")
        f.write(f"Raw CSV: {raw_csv_path}\n")
        f.write(f"Readiness report: {readiness_path}\n\n")

        f.write("Inputs from readiness\n")
        f.write(f"- Recommended tier (readiness): {report['inputs_from_readiness']['recommended_tier_from_readiness']}\n")
        f.write(f"- Readiness score (readiness): {report['inputs_from_readiness']['readiness_score_from_readiness']}\n")
        f.write(f"- Time regularity (readiness): {report['inputs_from_readiness']['time_regularity_from_readiness']}\n\n")

        f.write("Data overview\n")
        f.write(f"- Rows: {report['dataset_overview']['n_rows']}\n")
        f.write(f"- Timestamp col: {date_col}\n")
        f.write(f"- Nominal columns detected: {', '.join(nominal_cols) if nominal_cols else '(none)'}\n\n")

        f.write("Variables analyzed\n")
        f.write(f"- Count: {len(vars_to_analyze)}\n")
        if vars_to_analyze:
            f.write(f"- Variables: {', '.join(vars_to_analyze)}\n")
        f.write("\n")

        f.write("Highlights\n")
        if top_trends:
            f.write("- Top trends (coef, p):\n")
            for s in top_trends:
                f.write(f"  * {s}\n")
        else:
            f.write("- Top trends: (none)\n")

        if top_prog:
            f.write("- Top progress (regressed % vs scale):\n")
            for s in top_prog:
                f.write(f"  * {s}\n")
        else:
            f.write("- Top progress: (none)\n")

        f.write("\nSignificant timestamp effects (alpha)\n")
        for k, lst in sig_time_effects.items():
            f.write(f"- {k}: {len(lst)} significant\n")
            for s in lst[:12]:
                f.write(f"  * {s}\n")
        f.write("\n")

        f.write("Outputs\n")
        f.write(f"- JSON report: {json_path}\n")
        f.write(f"- Overview CSV: {overview_path}\n")
        f.write(f"- Summary TXT: {summary_path}\n")

    return report


# -----------------------------
# Discovery
# -----------------------------
def discover_readiness_reports(readiness_root: Path, readiness_filename: str) -> List[Path]:
    return sorted(readiness_root.rglob(readiness_filename))


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run regular time-series analyses per pseudoprofile (no networks).")
    p.add_argument("--input-root", type=str, default=DEFAULT_INPUT_ROOT, help="Root containing pseudoprofile subfolders with raw CSV.")
    p.add_argument("--readiness-root", type=str, default=DEFAULT_READINESS_ROOT, help="Root containing readiness_check outputs.")
    p.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT, help="Root output folder for regular TS analysis results.")
    p.add_argument("--data-filename", type=str, default=DEFAULT_DATA_FILENAME, help="Raw CSV filename inside each profile folder.")
    p.add_argument("--readiness-filename", type=str, default=DEFAULT_READINESS_FILENAME, help="Readiness JSON filename to find.")
    p.add_argument("--max-profiles", type=int, default=0, help="If >0, only process first N readiness reports.")
    p.add_argument("--quiet", type=str, default="False", help="Reduce console output (True/False).")
    p.add_argument("--prefer-tier", type=str, default=CFG.prefer_tier, choices=["tier1", "overall_ready", "all_non_hard"], help="Variable selection strategy.")
    p.add_argument("--impute", type=str, default=str(CFG.enable_imputation), help="Enable imputation (True/False).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root).expanduser()
    readiness_root = Path(args.readiness_root).expanduser()
    output_root = Path(args.output_root).expanduser()

    verbose = not _bool_from_str(args.quiet)

    CFG.prefer_tier = str(args.prefer_tier)
    CFG.enable_imputation = _bool_from_str(args.impute)

    if not readiness_root.exists():
        print(f"[ERROR] readiness-root not found: {readiness_root}", file=sys.stderr)
        return 2
    if not input_root.exists():
        print(f"[ERROR] input-root not found: {input_root}", file=sys.stderr)
        return 2

    _ensure_dir(output_root)

    if HAVE_STATSMODELS and InterpolationWarning is not None:
        warnings.filterwarnings("ignore", category=InterpolationWarning)

    readiness_paths = discover_readiness_reports(readiness_root, args.readiness_filename)
    if args.max_profiles and args.max_profiles > 0:
        readiness_paths = readiness_paths[: args.max_profiles]

    if not readiness_paths:
        print(f"[WARN] No readiness reports named '{args.readiness_filename}' found under: {readiness_root}", file=sys.stderr)
        return 1

    if verbose:
        print(f"[INFO] Found {len(readiness_paths)} readiness reports.")
        print(f"[INFO] Output root: {output_root}")
        print(f"[INFO] Optional deps: scipy={HAVE_SCIPY}, statsmodels={HAVE_STATSMODELS}")
        print(f"[INFO] Config: prefer_tier={CFG.prefer_tier}, impute={CFG.enable_imputation}")
        print("")

    n_ok = 0
    n_fail = 0

    for i, rp in enumerate(readiness_paths, start=1):
        try:
            profile_id = rp.parent.name
            print(f"[{i}/{len(readiness_paths)}] Processing: {profile_id}")
            analyze_profile(
                readiness_path=rp,
                input_root=input_root,
                output_root=output_root,
                data_filename=args.data_filename,
                verbose=verbose,
            )
            print("    [OK] regular_ts_report.json + summary + overview.csv written.\n")
            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f"[ERROR] Failed on {rp}: {repr(e)}", file=sys.stderr)

    print(f"\n[DONE] Success: {n_ok}  Failed: {n_fail}")
    return 0 if n_fail == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
