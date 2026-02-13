#!/usr/bin/env python3
"""
generate_pseudodata_from_profiles.py

FIXED VERSION (robust mapped-file discovery)

Why you got:
  "No mapped files found under .../runs/<latest>/profiles (looking for llm_observation_model_mapped.txt)"

Because the script previously searched ONLY for an exact filename:
  llm_observation_model_mapped.txt

In practice, your pipeline may produce variants like:
  llm_observation_model_mapped_v2.txt
  llm_observation_model_mapped_cleaned.txt
  llm_observation_model_mapped.md
  observation_model_mapped.txt
or place them slightly differently under the run folder.

This version:
- Defaults to a GLOB PATTERN: "llm_observation_model_mapped*.txt" (matches the exact file too)
- Falls back to broader patterns if nothing is found
- As a last resort, scans .txt files and keeps those that *look like* mapped models
  (contain CRITERIA/PREDICTORS and at least one "- P.." / "- C.." item line)
- Prints a clear diagnostic of what it tried (so you can still fix paths quickly)

Outputs per pseudoprofile:
  <out_root>/<pseudoprofile_id>/
    - variables_metadata.csv
    - pseudodata_long.csv
    - pseudodata_wide.csv
    - generation_summary.json
    - data_pattern_spec.txt
    - pseudodata_index.json (at out_root)

Default runs root:
  /Users/stijnvanseveren/PythonProjects/MASTERPROEF/Evaluation/03_construction_initial_observation_model/constructed_PC_models/runs

Default output root:
  /Users/stijnvanseveren/PythonProjects/MASTERPROEF/Evaluation/01_pseudoprofile(s)/time_series_data/pseudodata

Usage:
  python generate_pseudodata_from_profiles.py --overwrite

Optional:
  python generate_pseudodata_from_profiles.py --runs-root "/path/to/runs" --overwrite
  python generate_pseudodata_from_profiles.py --run-id "2026-01-18_19-37-23" --overwrite
  python generate_pseudodata_from_profiles.py --mapped-pattern "llm_observation_model_mapped*.txt" --overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Defaults (your paths)
# =============================================================================
def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "Evaluation").exists() and (candidate / "README.md").exists():
            return candidate
    raise RuntimeError("Could not locate repository root from create_pseudodata.py")


REPO_ROOT = _find_repo_root()

DEFAULT_RUNS_ROOT = (
    REPO_ROOT / "Evaluation/03_construction_initial_observation_model/constructed_PC_models/runs"
)

DEFAULT_OUT_ROOT = (
    REPO_ROOT / "Evaluation/01_pseudoprofile(s)/time_series_data/pseudodata"
)

# IMPORTANT FIX: use a glob that still matches the exact filename
MAPPED_PATTERN_DEFAULT = "llm_observation_model_mapped*.txt"


# =============================================================================
# Data structures
# =============================================================================
@dataclass(frozen=True)
class VariableDef:
    code: str                 # e.g. "P01" / "C03"
    role: str                 # "PREDICTOR" / "CRITERION"
    label: str                # left of "->"
    ontology_id: str          # last slash segment
    conf: Optional[float]     # from "(conf=...)"
    freq_hint: Optional[str]  # parsed from line (freq=weekly, ...)


@dataclass
class ProfileModel:
    pseudoprofile_id: str
    source_raw_model: Optional[str]
    variables: List[VariableDef]
    global_freq_hint: Optional[str]
    global_n_hint: Optional[int]
    mapped_file: Path


# =============================================================================
# Regex + parsing
# =============================================================================
_RX_PROFILE_ID = re.compile(r"^\s*pseudoprofile_id\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_RX_SOURCE_RAW = re.compile(r"^\s*source_raw_model\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)

_RX_GLOBAL_FREQ = re.compile(r"^\s*(?:sampling_)?frequency\s*[:=]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_RX_GLOBAL_N = re.compile(r"^\s*(?:n_timepoints|timepoints|n_samples)\s*[:=]\s*(\d+)\s*$", re.IGNORECASE | re.MULTILINE)

_RX_ITEM = re.compile(r"^\s*-\s*(?P<code>[PC]\d{2})\s*:\s*(?P<label>.*?)\s*->\s*(?P<rest>.+?)\s*$")
_RX_CONF = re.compile(r"\(\s*conf\s*=\s*([0-9]*\.?[0-9]+)\s*\)", re.IGNORECASE)
_RX_FREQ_INLINE = re.compile(r"(?:freq(?:uency)?)\s*[:=]\s*([A-Za-z_ -]+)", re.IGNORECASE)

_RX_RUN_DIRNAME = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")


def _normalize_freq_token(s: str) -> str:
    s2 = re.sub(r"\s+", " ", s.strip().lower())
    aliases = {
        "day": "daily",
        "daily": "daily",
        "d": "daily",
        "week": "weekly",
        "weekly": "weekly",
        "w": "weekly",
        "month": "monthly",
        "monthly": "monthly",
        "m": "monthly",
        "hour": "hourly",
        "hourly": "hourly",
        "h": "hourly",
    }
    head = s2.split(" ")[0]
    return aliases.get(head, head)


def looks_like_mapped_model_txt(p: Path) -> bool:
    """
    Heuristic: keep .txt files that resemble the mapped model format.
    """
    try:
        txt = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False

    up = txt.upper()
    if ("CRITERIA" not in up) or ("PREDICTORS" not in up):
        return False

    # must contain at least one "- P.." or "- C.." item line
    if re.search(r"^\s*-\s*[PC]\d{2}\s*:", txt, flags=re.MULTILINE) is None:
        return False

    return True


def parse_mapped_txt(path: Path) -> ProfileModel:
    txt = path.read_text(encoding="utf-8", errors="replace")

    pid_m = _RX_PROFILE_ID.search(txt)
    pseudoprofile_id = pid_m.group(1).strip() if pid_m else path.parent.name

    src_m = _RX_SOURCE_RAW.search(txt)
    source_raw_model = src_m.group(1).strip() if src_m else None

    gf_m = _RX_GLOBAL_FREQ.search(txt)
    global_freq_hint = _normalize_freq_token(gf_m.group(1)) if gf_m else None

    gn_m = _RX_GLOBAL_N.search(txt)
    global_n_hint = int(gn_m.group(1)) if gn_m else None

    lines = txt.splitlines()
    section: Optional[str] = None
    variables: List[VariableDef] = []

    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if s.upper().startswith("CRITERIA"):
            section = "CRITERION"
            continue
        if s.upper().startswith("PREDICTORS"):
            section = "PREDICTOR"
            continue

        m = _RX_ITEM.match(ln)
        if not m or section is None:
            continue

        code = m.group("code").strip()
        label = m.group("label").strip()
        rest = m.group("rest").strip()

        ontology_id = "unknown"
        segs = re.findall(r"/\s*([A-Za-z0-9_]+)", rest)
        if segs:
            ontology_id = segs[-1].strip()

        conf = None
        cm = _RX_CONF.search(rest)
        if cm:
            conf = float(cm.group(1))

        freq_hint = None
        fm = _RX_FREQ_INLINE.search(ln)
        if fm:
            freq_hint = _normalize_freq_token(fm.group(1))

        variables.append(
            VariableDef(
                code=code,
                role=section,
                label=label,
                ontology_id=ontology_id,
                conf=conf,
                freq_hint=freq_hint,
            )
        )

    def _sort_key(v: VariableDef) -> Tuple[int, str]:
        return (0 if v.role == "PREDICTOR" else 1, v.code)

    variables = sorted(variables, key=_sort_key)

    return ProfileModel(
        pseudoprofile_id=pseudoprofile_id,
        source_raw_model=source_raw_model,
        variables=variables,
        global_freq_hint=global_freq_hint,
        global_n_hint=global_n_hint,
        mapped_file=path,
    )


# =============================================================================
# Run discovery
# =============================================================================
def find_latest_run_dir(runs_root: Path) -> Path:
    if not runs_root.exists() or not runs_root.is_dir():
        raise SystemExit(f"runs_root does not exist or is not a directory: {runs_root}")

    candidates: List[Path] = []
    for p in runs_root.iterdir():
        if p.is_dir() and _RX_RUN_DIRNAME.match(p.name):
            candidates.append(p)

    if not candidates:
        raise SystemExit(f"No run directories found in: {runs_root}")

    return sorted(candidates, key=lambda x: x.name)[-1]


def find_run_dir(runs_root: Path, run_id: str) -> Path:
    p = runs_root / run_id
    if not p.exists() or not p.is_dir():
        raise SystemExit(f"Specified run-id directory not found: {p}")
    return p


def find_profile_mapped_files(run_dir: Path, mapped_pattern: str) -> List[Path]:
    """
    Robust discovery:
    1) Try under <run_dir>/profiles with mapped_pattern (glob)
    2) Try under run_dir directly (glob)
    3) Fallback patterns if still empty
    4) Last resort: scan .txt files and keep those that look like mapped model format
    """
    tried: List[str] = []
    profiles_dir = run_dir / "profiles"

    candidates: List[Path] = []

    if profiles_dir.exists():
        tried.append(f"{profiles_dir} rglob({mapped_pattern})")
        candidates = sorted(profiles_dir.rglob(mapped_pattern))
    else:
        tried.append(f"{profiles_dir} (missing)")

    if not candidates:
        tried.append(f"{run_dir} rglob({mapped_pattern})")
        candidates = sorted(run_dir.rglob(mapped_pattern))

    # fallback patterns (common variants)
    if not candidates:
        fallback_patterns = [
            "llm_observation_model_mapped*.txt",
            "*observation_model*mapped*.txt",
            "*mapped*.txt",
            "*mapped*.md",
        ]
        for pat in fallback_patterns:
            if pat == mapped_pattern:
                continue
            if profiles_dir.exists():
                tried.append(f"{profiles_dir} rglob({pat})")
                candidates = sorted(profiles_dir.rglob(pat))
            if not candidates:
                tried.append(f"{run_dir} rglob({pat})")
                candidates = sorted(run_dir.rglob(pat))
            if candidates:
                break

    # last resort: scan .txt and filter by content
    if not candidates:
        scanned: List[Path] = []
        if profiles_dir.exists():
            tried.append(f"{profiles_dir} rglob(*.txt) + content filter")
            scanned = sorted(profiles_dir.rglob("*.txt"))
        else:
            tried.append(f"{run_dir} rglob(*.txt) + content filter")
            scanned = sorted(run_dir.rglob("*.txt"))

        candidates = [p for p in scanned if looks_like_mapped_model_txt(p)]

    if not candidates:
        # Provide helpful diagnostics: show some filenames we *did* see
        sample_txt = []
        if profiles_dir.exists():
            sample_txt = [p.name for p in sorted(profiles_dir.rglob("*.txt"))[:15]]
        sample_any = [p.name for p in sorted(run_dir.iterdir())[:15]]

        msg = (
            f"No mapped files found for run: {run_dir}\n"
            f"Tried:\n  - " + "\n  - ".join(tried) + "\n\n"
            f"Found in run_dir (top-level sample): {sample_any}\n"
            f"Found in profiles_dir (*.txt sample): {sample_txt}\n\n"
            f"Fix options:\n"
            f"  1) Set --mapped-pattern to match your actual filename, e.g.\n"
            f"       --mapped-pattern '*mapped*.txt'\n"
            f"  2) Or pass --run-id to point at a different run\n"
        )
        raise SystemExit(msg)

    # de-duplicate
    uniq: List[Path] = []
    seen = set()
    for p in candidates:
        rp = str(p.resolve())
        if rp not in seen:
            uniq.append(p)
            seen.add(rp)

    return uniq


# =============================================================================
# Repro + math helpers
# =============================================================================
def stable_int_seed(*parts: str, mod: int = 2**31 - 1) -> int:
    h = hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()
    return int(h[:12], 16) % mod


def sigmoid(x: np.ndarray, k: float = 10.0) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-k * x))


def smooth_step(t: np.ndarray, center: float, width: float) -> np.ndarray:
    width = max(width, 1e-6)
    return sigmoid((t - center) / width, k=1.0)


def ar1_process(n: int, phi: float, rng: np.random.Generator, noise_sd: float) -> np.ndarray:
    x = np.zeros(n, dtype=float)
    eps = rng.normal(0.0, noise_sd, size=n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return x


def lagged_weighted_sum(x: np.ndarray, lags: List[int], weights: List[float]) -> np.ndarray:
    n = x.size
    out = np.zeros(n, dtype=float)
    for lag, w in zip(lags, weights):
        lag = int(max(0, lag))
        if lag == 0:
            out += w * x
        else:
            out[lag:] += w * x[:-lag]
    return out


def clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def clip010(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 10.0)


def to_pandas_freq(freq_token: str) -> str:
    token = str(freq_token).strip().upper()
    return "H" if token == "H" else "D"


def schedule_mask_from_freq_hint(n: int, hint: Optional[str], rng: np.random.Generator) -> np.ndarray:
    if hint is None:
        return np.ones(n, dtype=bool)

    h = _normalize_freq_token(hint)
    if h == "weekly":
        k = int(rng.integers(0, 7))
        idx = np.arange(n)
        return (idx % 7) == k
    if h == "monthly":
        step = int(rng.integers(28, 33))
        mask = np.zeros(n, dtype=bool)
        mask[0::step] = True
        return mask
    return np.ones(n, dtype=bool)


# =============================================================================
# Core generation
# =============================================================================
def generate_profile_timeseries(
    model: ProfileModel,
    n: int,
    freq: str,
    start_date: str,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, str]:
    rng = np.random.default_rng(seed)

    t_idx = np.arange(n, dtype=int)
    t = np.linspace(0.0, 1.0, n)

    weekly_sin = np.sin(2 * np.pi * t_idx / 7.0)
    weekend = ((t_idx % 7) >= 5).astype(float)

    center = float(np.clip(rng.normal(0.55, 0.06), 0.25, 0.85))
    w_step = (t >= center).astype(float)
    w_smooth = smooth_step(t, center=center, width=0.04)

    stress_base = ar1_process(n, phi=0.93, rng=rng, noise_sd=0.25)
    sleep_disruption_base = ar1_process(n, phi=0.90, rng=rng, noise_sd=0.22)

    drift = 0.6 * (t - 0.5)

    pulse = np.zeros(n, dtype=float)
    n_pulses = int(rng.integers(2, 5))
    pulse_specs: List[Dict[str, float]] = []
    for _ in range(n_pulses):
        start = int(rng.integers(0, max(1, n - 10)))
        length = int(rng.integers(2, 8))
        amp = float(rng.normal(1.0, 0.25))
        pulse[start : start + length] += amp * np.exp(-np.linspace(0, 1.6, length))
        pulse_specs.append({"start_idx": start, "length": length, "amp": amp})

    phase = float(rng.uniform(0, 2 * np.pi))
    shared_freq = float(rng.choice([0.8, 1.0, 1.3]))
    shared_osc = np.sin(2 * np.pi * t * shared_freq + phase)

    stress = 0.9 * stress_base + 0.9 * w_smooth + 0.25 * weekly_sin + 0.25 * drift + 0.45 * pulse
    stress = (stress - stress.mean()) / (stress.std() + 1e-12)

    sleep_disruption = (
        0.9 * sleep_disruption_base
        + 0.40 * (1.0 - w_smooth)
        + 0.25 * (-weekly_sin)
        - 0.18 * drift
        + 0.25 * (pulse * (1.0 - 0.6 * w_smooth))
    )
    sleep_disruption = (sleep_disruption - sleep_disruption.mean()) / (sleep_disruption.std() + 1e-12)

    preds = [v for v in model.variables if v.role == "PREDICTOR"]
    crits = [v for v in model.variables if v.role == "CRITERION"]

    P_series: Dict[str, np.ndarray] = {}

    def predictor_template(v: VariableDef) -> Tuple[np.ndarray, Dict]:
        lab = v.label.lower()
        base = 0.2 + 0.08 * rng.normal()
        adopt = 0.55 * w_smooth + 0.15 * w_step

        profile = "generic"
        if any(k in lab for k in ["sleep", "circadian", "daylight", "light", "environment"]):
            profile = "sleep_light"
            s = base + 0.55 * adopt + 0.12 * (1.0 - weekend) + 0.10 * weekly_sin
        elif any(k in lab for k in ["rumination", "mindfulness", "meditat", "self-regulation"]):
            profile = "cognitive_regulation"
            s = base + 0.35 * adopt + 0.18 * clip01(0.5 + 0.35 * (-stress)) + 0.10 * shared_osc
            s += 0.10 * (pulse > 0).astype(float)
        elif any(k in lab for k in ["social", "support", "belong", "community", "network"]):
            profile = "social_support"
            s = base + 0.30 * adopt + 0.12 * (t - 0.2) + 0.06 * (1.0 - 0.5 * weekend)
        else:
            s = base + 0.40 * adopt + 0.10 * weekly_sin

        noise_sd = 0.08 + 0.06 * clip01((stress - np.percentile(stress, 20)) / 2.0)
        eps = rng.normal(0.0, noise_sd, size=n)
        x = np.zeros(n, dtype=float)
        x[0] = clip01(0.5 + 0.2 * rng.normal())
        phi = float(np.clip(rng.normal(0.85, 0.06), 0.70, 0.95))

        for i in range(1, n):
            x[i] = phi * x[i - 1] + (1.0 - phi) * s[i] + eps[i]

        dropout_spec = None
        if rng.random() < 0.6:
            d0 = int(rng.integers(int(0.15 * n), int(0.85 * n)))
            dlen = int(rng.integers(2, 7))
            mult = float(rng.uniform(0.2, 0.6))
            x[d0 : d0 + dlen] *= mult
            dropout_spec = {"start_idx": d0, "length": dlen, "multiplier": mult}

        meta = {"template": profile, "phi": phi, "noise_sd_mean": float(np.mean(noise_sd)), "dropout": dropout_spec}
        return clip01(x), meta

    predictor_meta: Dict[str, Dict] = {}
    for v in preds:
        x, meta = predictor_template(v)
        P_series[v.code] = x
        predictor_meta[v.code] = meta

    driver_pred_code = str(rng.choice([v.code for v in preds])) if preds else None

    C_series: Dict[str, np.ndarray] = {}

    def relevant_predictor_weights_for_criterion(v: VariableDef) -> Dict[str, float]:
        lab = v.label.lower()
        w: Dict[str, float] = {}
        if any(k in lab for k in ["sleep", "insomnia", "circadian", "waking", "fatigue", "alert"]):
            for p in preds:
                if any(k in p.label.lower() for k in ["sleep", "circadian", "daylight", "light", "environment"]):
                    w[p.code] = w.get(p.code, 0.0) + 1.0
        if any(k in lab for k in ["rumination", "worry", "anxiety", "catastroph", "stress"]):
            for p in preds:
                if any(k in p.label.lower() for k in ["rumination", "mindfulness", "meditat", "self-regulation"]):
                    w[p.code] = w.get(p.code, 0.0) + 1.0
        if any(k in lab for k in ["anger", "criticism", "emotion"]):
            for p in preds:
                if any(k in p.label.lower() for k in ["mindfulness", "self-regulation", "support", "belong"]):
                    w[p.code] = w.get(p.code, 0.0) + 0.8
        if any(k in lab for k in ["support", "belong", "social", "community"]):
            for p in preds:
                if any(k in p.label.lower() for k in ["support", "belong", "social", "community"]):
                    w[p.code] = w.get(p.code, 0.0) + 1.0

        if not w and preds:
            for p in preds:
                w[p.code] = 0.35

        if w:
            s = sum(w.values())
            for k in list(w.keys()):
                w[k] = float(w[k] / (s + 1e-12))
        return w

    criterion_meta: Dict[str, Dict] = {}

    def criterion_template(v: VariableDef) -> Tuple[np.ndarray, Dict]:
        lab = v.label.lower()
        base = 5.0 + 0.35 * rng.normal()
        seasonal = 0.55 * (-weekly_sin) + 0.25 * weekend

        w_sleep = 1.3 if any(k in lab for k in ["sleep", "fatigue", "alert", "waking", "insomnia"]) else 0.7
        w_stress = 1.2 if any(k in lab for k in ["rumination", "worry", "anxiety", "anger"]) else 0.8

        latent = base + 1.2 * drift + seasonal + w_sleep * sleep_disruption + w_stress * stress

        uses_shared = bool(rng.random() < 0.7)
        if uses_shared:
            latent += 0.35 * shared_osc

        weights = relevant_predictor_weights_for_criterion(v)
        lags = [2, 4, 7]
        lag_w = [0.50, 0.35, 0.15]
        eff = np.zeros(n, dtype=float)
        if weights:
            for pc, ww in weights.items():
                eff += ww * lagged_weighted_sum(P_series[pc], lags=lags, weights=lag_w)
            latent += -3.0 * eff

        driver = None
        if driver_pred_code is not None and rng.random() < 0.9:
            drv = P_series[driver_pred_code]
            drv_lag = lagged_weighted_sum(drv, lags=[3], weights=[1.0])
            a1 = float(rng.uniform(-1.2, 0.6))
            a2 = float(rng.uniform(-2.0, 1.4))
            coef_t = (1.0 - w_smooth) * a1 + w_smooth * a2
            latent += coef_t * drv_lag
            driver = {"code": driver_pred_code, "a1_pre": a1, "a2_post": a2}

        latent += 0.65 * (pulse > 0).astype(float) * float(rng.uniform(0.5, 1.2))

        phi = float(np.clip(rng.normal(0.82, 0.07), 0.65, 0.95))
        x = np.zeros(n, dtype=float)
        x[0] = clip010(np.array([base + rng.normal(0, 0.8)]))[0]

        noise_sd = 0.45 + 0.35 * clip01((stress - np.percentile(stress, 30)) / 2.0)
        eps = rng.normal(0.0, noise_sd, size=n)

        for i in range(1, n):
            x[i] = phi * x[i - 1] + (1.0 - phi) * latent[i] + eps[i]

        outlier_points: List[int] = []
        if rng.random() < 0.7:
            k_out = int(rng.integers(1, 4))
            for _ in range(k_out):
                j = int(rng.integers(0, n))
                x[j] += float(rng.normal(0.0, 2.2))
                outlier_points.append(j)

        meta = {
            "phi": phi,
            "noise_sd_mean": float(np.mean(noise_sd)),
            "uses_shared_osc": uses_shared,
            "predictor_weights": weights,
            "lags": lags,
            "lag_weights": lag_w,
            "driver_effect": driver,
            "outlier_indices": outlier_points,
        }
        return clip010(x), meta

    for v in crits:
        x, meta = criterion_template(v)
        C_series[v.code] = x
        criterion_meta[v.code] = meta

    all_codes = [v.code for v in model.variables]
    df_wide = pd.DataFrame({"t_index": t_idx.copy()})

    try:
        dr = pd.date_range(start=start_date, periods=n, freq=to_pandas_freq(freq))
        df_wide["date"] = dr.astype(str)
    except Exception:
        df_wide["date"] = [str(date.today())] * n

    for v in model.variables:
        df_wide[v.code] = P_series.get(v.code, np.full(n, np.nan)) if v.role == "PREDICTOR" else C_series.get(v.code, np.full(n, np.nan))

    missing_stats: Dict[str, float] = {}
    stress01 = clip01(
        (stress - np.percentile(stress, 10))
        / (np.percentile(stress, 90) - np.percentile(stress, 10) + 1e-12)
    )

    missing_meta: Dict[str, Dict] = {}
    for v in model.variables:
        col = v.code
        x = df_wide[col].to_numpy(dtype=float)

        schedule = schedule_mask_from_freq_hint(n=n, hint=v.freq_hint, rng=rng)
        base_target = float(rng.uniform(0.05, 0.22) if v.role == "PREDICTOR" else rng.uniform(0.08, 0.35))

        mcar = rng.random(n) < (0.55 * base_target)
        mar_prob = 0.90 * base_target * (stress01 ** 1.6)
        mar = rng.random(n) < mar_prob

        block = np.zeros(n, dtype=bool)
        block_specs = []
        if rng.random() < 0.75:
            n_blocks = int(rng.integers(1, 3))
            for _ in range(n_blocks):
                bstart = int(rng.integers(0, max(1, n - 10)))
                blen = int(rng.integers(3, 11))
                block[bstart : bstart + blen] = True
                block_specs.append({"start_idx": bstart, "length": blen})

        missing = (mcar | mar | block | (~schedule))

        if missing.mean() > 0.92:
            missing[:] = False
            missing[rng.integers(0, n, size=int(0.15 * n))] = True

        x[missing] = np.nan
        df_wide[col] = x

        missing_stats[col] = float(np.mean(np.isnan(x)))
        missing_meta[col] = {
            "base_target": base_target,
            "schedule_hint": v.freq_hint,
            "schedule_observed_fraction": float(np.mean(schedule)),
            "block_segments": block_specs,
            "mcar_fraction": float(np.mean(mcar)),
            "mar_fraction": float(np.mean(mar)),
        }

    df_meta = pd.DataFrame(
        [
            {
                "code": v.code,
                "role": v.role,
                "label": v.label,
                "ontology_id": v.ontology_id,
                "conf": v.conf,
                "freq_hint": v.freq_hint,
            }
            for v in model.variables
        ]
    )

    df_long = df_wide.melt(
        id_vars=["t_index", "date"],
        value_vars=all_codes,
        var_name="code",
        value_name="value",
    )
    df_long = df_long.merge(df_meta, on="code", how="left")
    df_long.insert(0, "pseudoprofile_id", model.pseudoprofile_id)

    summary = {
        "pseudoprofile_id": model.pseudoprofile_id,
        "source_raw_model": model.source_raw_model,
        "mapped_file": str(model.mapped_file),
        "n_timepoints": int(n),
        "freq": str(freq),
        "start_date": str(start_date),
        "regime_center_t_norm": float(center),
        "driver_predictor_code": driver_pred_code,
        "shared_osc_frequency": float(shared_freq),
        "shared_osc_phase": float(phase),
        "pulse_specs": pulse_specs,
        "missing_rate_by_code": missing_stats,
        "missing_rate_overall": float(np.mean(list(missing_stats.values())) if missing_stats else 0.0),
        "predictor_generation_meta": predictor_meta,
        "criterion_generation_meta": criterion_meta,
        "missingness_meta": missing_meta,
        "patterns": [
            "AR(1) inertia",
            "weekly seasonality",
            "slow drift trend",
            "abrupt step change",
            "smooth sigmoid transition",
            "shared latent sinusoidal co-fluctuation",
            "lagged predictor->criterion effects",
            "bursts/pulses + decay",
            "outliers/shocks",
            "missingness: MCAR+MAR+block+schedule",
        ],
        "variables": df_meta.to_dict(orient="records"),
    }

    spec_lines = []
    spec_lines.append(f"pseudoprofile_id: {model.pseudoprofile_id}")
    spec_lines.append(f"mapped_file: {model.mapped_file}")
    spec_lines.append(f"source_raw_model: {model.source_raw_model}")
    spec_lines.append("")
    spec_lines.append("TIME AXIS")
    spec_lines.append(f"- n_timepoints: {n}")
    spec_lines.append(f"- freq (base axis): {freq}  (measurement schedules may enforce extra missingness)")
    spec_lines.append(f"- start_date: {start_date}")
    spec_lines.append("")
    spec_lines.append("REGIME SHIFT (Patterns 4 & 5)")
    spec_lines.append(f"- center (t_norm): {center:.4f}")
    spec_lines.append("- abrupt step: activates at t >= center")
    spec_lines.append("- smooth transition: sigmoid around center (width ~0.04 in t_norm)")
    spec_lines.append("")
    spec_lines.append("LATENT DRIVERS")
    spec_lines.append("- stress: AR(1) + smooth shift + weekly sinusoid + drift + pulses")
    spec_lines.append("- sleep_disruption: AR(1) + improvement after shift + weekly sinusoid + drift + pulses")
    spec_lines.append("")
    spec_lines.append("SHARED CO-FLUCTUATION (Pattern 6)")
    spec_lines.append(f"- shared_osc: sin(2π * {shared_freq:.2f} * t + phase={phase:.2f})")
    spec_lines.append("")
    spec_lines.append("PULSES / EVENTS (Pattern 8)")
    spec_lines.append(f"- number_of_pulses: {n_pulses}")
    for i, ps in enumerate(pulse_specs, start=1):
        spec_lines.append(f"  * pulse_{i}: start_idx={ps['start_idx']}, length={ps['length']}, amp={ps['amp']:.3f}")
    spec_lines.append("")
    spec_lines.append("CROSS-VARIABLE EFFECTS (Pattern 7)")
    spec_lines.append(f"- driver_predictor_code (regime-dependent effect): {driver_pred_code}")
    spec_lines.append("- criteria receive lagged predictor influence via lags [2,4,7] (weights [0.50,0.35,0.15])")
    spec_lines.append("")
    spec_lines.append("MISSINGNESS (Pattern 10)")
    spec_lines.append("- MCAR + MAR(stress-related) + block-missing + measurement schedule (freq hints)")
    spec_lines.append("- realized missingness rates by code:")
    for code in all_codes:
        fh = df_meta.set_index("code").loc[code, "freq_hint"]
        spec_lines.append(f"  * {code}: missing_rate={missing_stats.get(code, float('nan')):.3f} | freq_hint={fh}")
    spec_lines.append("")
    spec_lines.append("VARIABLES")
    for v in model.variables:
        spec_lines.append(f"- {v.code} ({v.role}): {v.label} | ontology_id={v.ontology_id} | conf={v.conf} | freq_hint={v.freq_hint}")
    spec_lines.append("")
    spec_lines.append("PATTERN CHECKLIST")
    for k, name in enumerate(summary["patterns"], start=1):
        spec_lines.append(f"{k}) {name}")

    spec_txt = "\n".join(spec_lines)
    return df_long, df_wide, summary, spec_txt


# =============================================================================
# IO helpers
# =============================================================================
def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_outputs(
    out_dir: Path,
    df_long: pd.DataFrame,
    df_wide: pd.DataFrame,
    summary: Dict,
    spec_txt: str,
    overwrite: bool,
) -> None:
    safe_mkdir(out_dir)

    if overwrite:
        for fn in [
            "pseudodata_long.csv",
            "pseudodata_wide.csv",
            "variables_metadata.csv",
            "generation_summary.json",
            "data_pattern_spec.txt",
        ]:
            fp = out_dir / fn
            if fp.exists():
                fp.unlink()

    meta_cols = ["code", "role", "label", "ontology_id", "conf", "freq_hint"]
    df_meta = df_long[meta_cols].drop_duplicates().sort_values(["role", "code"])
    df_meta.to_csv(out_dir / "variables_metadata.csv", index=False)

    df_long.to_csv(out_dir / "pseudodata_long.csv", index=False)
    df_wide.to_csv(out_dir / "pseudodata_wide.csv", index=False)

    (out_dir / "generation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "data_pattern_spec.txt").write_text(spec_txt, encoding="utf-8")


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--runs-root",
        type=str,
        default=str(DEFAULT_RUNS_ROOT),
        help="Root directory containing runs/<YYYY-MM-DD_HH-MM-SS>/",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional: explicitly select a run folder name (e.g., 2026-01-18_19-37-23). If omitted, uses latest.",
    )
    parser.add_argument(
        "--mapped-pattern",
        type=str,
        default=MAPPED_PATTERN_DEFAULT,
        help="Glob pattern to find mapped files (default matches llm_observation_model_mapped.txt and suffixed variants).",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default=str(DEFAULT_OUT_ROOT),
        help="Output root. Each pseudoprofile gets its own subdirectory here.",
    )

    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Override number of timepoints. If omitted: use global hint in txt if present, else 180.",
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="D",
        help="Base time axis frequency for date index: 'D' or 'H'.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2025-01-01",
        help="Start date (YYYY-MM-DD) for the generated index.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master seed; per-profile seed derived from this + pseudoprofile_id + mapped file path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite outputs in each pseudoprofile subdirectory if they exist.",
    )

    args = parser.parse_args()

    runs_root = Path(args.runs_root).expanduser()
    out_root = Path(args.out_root).expanduser()

    run_dir = find_run_dir(runs_root, args.run_id) if args.run_id else find_latest_run_dir(runs_root)
    mapped_files = find_profile_mapped_files(run_dir, args.mapped_pattern)

    safe_mkdir(out_root)

    print(f"[INFO] runs_root:       {runs_root}")
    print(f"[INFO] selected run:    {run_dir}")
    print(f"[INFO] mapped_pattern:  {args.mapped_pattern}")
    print(f"[INFO] out_root:        {out_root}")
    print(f"[INFO] found {len(mapped_files)} mapped files")

    generated_ids: List[str] = []

    for mp in mapped_files:
        model = parse_mapped_txt(mp)

        n_eff = args.n if args.n is not None else (model.global_n_hint if model.global_n_hint is not None else 180)
        prof_seed = stable_int_seed(str(args.seed), model.pseudoprofile_id, str(mp))

        df_long, df_wide, summary, spec_txt = generate_profile_timeseries(
            model=model,
            n=int(n_eff),
            freq=str(args.freq).strip().upper(),
            start_date=str(args.start_date),
            seed=int(prof_seed),
        )

        out_dir = out_root / model.pseudoprofile_id
        write_outputs(out_dir, df_long, df_wide, summary, spec_txt, overwrite=bool(args.overwrite))

        generated_ids.append(model.pseudoprofile_id)
        print(f"[OK] {model.pseudoprofile_id} -> {out_dir}")

    index = {
        "runs_root": str(runs_root),
        "run_dir": str(run_dir),
        "mapped_pattern": str(args.mapped_pattern),
        "out_root": str(out_root),
        "profiles_generated": generated_ids,
    }
    (out_root / "pseudodata_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    print("Done.")


if __name__ == "__main__":
    main()
