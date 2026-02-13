#!/usr/bin/env python3
"""
visualize_pseudodata_profiles.py

FIXED for:
  ValueError: The truth value of a Series is ambiguous

Root cause:
- variables_metadata.csv can contain duplicate "code" rows (e.g., if written from long data without
  enforcing uniqueness, or if the same code appears twice).
- Then meta.set_index("code").loc[c, "role"] returns a Series (multiple rows), not a scalar,
  and `== "PREDICTOR"` produces a boolean Series, which breaks inside an `if`.

Fixes implemented:
1) Robust metadata normalization: enforce one row per code (keep first) and normalize role strings.
2) Avoid repeated meta.set_index inside list comprehensions (performance + clarity).
3) Add progress print statements so you can see where the script is during execution.
4) Extra guards for very small n (rolling window) and correlation heatmap with constant/NaN columns.

Outputs per profile:
  <profile_dir>/visuals/*.png

Run:
  python visualize_pseudodata_profiles.py
  python visualize_pseudodata_profiles.py --root "/path/to/pseudodata" --overwrite

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# =============================================================================
# Configuration defaults
# =============================================================================
DEFAULT_ROOT_PSEUDODATA = Path(
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/Evaluation/"
    "01_pseudoprofile(s)/time_series_data/pseudodata"
)

WIDE_NAME = "pseudodata_wide.csv"
META_NAME = "variables_metadata.csv"
VIS_DIRNAME = "visuals"

ROLLING_WINDOW_DEFAULT = 30  # later adapted to series length


# =============================================================================
# Publication style
# =============================================================================
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#7F7F7F",
}


def set_pub_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.2,
            "lines.solid_capstyle": "round",
            "axes.grid": True,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.30,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# =============================================================================
# Utilities
# =============================================================================
def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, out_base: Path) -> None:
    ensure_dir(out_base.parent)
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    # fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")  # enable if you want vector PDFs


def zscore_df(X: pd.DataFrame) -> pd.DataFrame:
    mu = X.mean(axis=0, skipna=True)
    sd = X.std(axis=0, ddof=0, skipna=True).replace(0, np.nan)
    return (X - mu) / sd


def robust_limits(x: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> Tuple[float, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (-1.0, 1.0)
    return (float(np.percentile(x, lo)), float(np.percentile(x, hi)))


def pick_time_axis(df_wide: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Returns:
      t_num: numeric axis for plotting
      t_label: label series (dates if present, otherwise same as numeric)
    """
    if "date" in df_wide.columns:
        t_num = pd.Series(np.arange(len(df_wide)), name="t")
        return t_num, df_wide["date"].astype(str)
    if "t_index" in df_wide.columns:
        t = pd.to_numeric(df_wide["t_index"], errors="coerce")
        if t.isna().all():
            t_num = pd.Series(np.arange(len(df_wide)), name="t")
            return t_num, t_num
        return t, t
    t_num = pd.Series(np.arange(len(df_wide)), name="t")
    return t_num, t_num


def role_palette() -> Dict[str, str]:
    return {"PREDICTOR": OKABE_ITO["blue"], "CRITERION": OKABE_ITO["black"]}


def normalize_meta(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce one row per code and normalize fields to avoid ambiguous Series lookups.
    """
    m = meta.copy()

    # required columns
    for col in ["code", "role", "label"]:
        if col not in m.columns:
            raise ValueError(f"{META_NAME} must contain column '{col}'. Found: {list(m.columns)}")

    m["code"] = m["code"].astype(str).str.strip()
    m["role"] = m["role"].astype(str).str.strip().str.upper()
    m["label"] = m["label"].astype(str)

    # de-duplicate by code (keep first)
    if m["code"].duplicated().any():
        dup = m.loc[m["code"].duplicated(), "code"].unique().tolist()
        log(f"[WARN] Duplicated codes in metadata; keeping first occurrence. Duplicates: {dup[:10]}{'...' if len(dup)>10 else ''}")
        m = m.drop_duplicates(subset=["code"], keep="first")

    return m


def build_role_map(meta: pd.DataFrame) -> Dict[str, str]:
    return dict(zip(meta["code"].astype(str), meta["role"].astype(str)))


def code_color_map(codes: List[str], role_map: Dict[str, str]) -> Dict[str, str]:
    pred_colors = [
        OKABE_ITO["blue"],
        OKABE_ITO["sky"],
        OKABE_ITO["green"],
        OKABE_ITO["orange"],
        OKABE_ITO["purple"],
        OKABE_ITO["vermillion"],
        OKABE_ITO["yellow"],
        OKABE_ITO["gray"],
    ]
    out: Dict[str, str] = {}

    preds = [c for c in codes if role_map.get(c, "").upper() == "PREDICTOR"]
    crits = [c for c in codes if role_map.get(c, "").upper() == "CRITERION"]

    for i, c in enumerate(preds):
        out[c] = pred_colors[i % len(pred_colors)]
    for c in crits:
        out[c] = OKABE_ITO["black"]

    # unknowns
    unknowns = [c for c in codes if c not in out]
    for c in unknowns:
        out[c] = OKABE_ITO["gray"]

    return out


def criterion_linestyles(codes: List[str], role_map: Dict[str, str]) -> Dict[str, str]:
    crit_styles = ["-", "--", "-.", ":"]
    out: Dict[str, str] = {}
    crits = [c for c in codes if role_map.get(c, "").upper() == "CRITERION"]
    for i, c in enumerate(crits):
        out[c] = crit_styles[i % len(crit_styles)]
    return out


def load_profile_inputs(profile_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Dict[str, str], Dict[str, str]]:
    wide_path = profile_dir / WIDE_NAME
    meta_path = profile_dir / META_NAME

    if not wide_path.exists():
        raise FileNotFoundError(f"Missing {WIDE_NAME} in {profile_dir}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {META_NAME} in {profile_dir}")

    df_wide = pd.read_csv(wide_path)
    meta_raw = pd.read_csv(meta_path)

    meta = normalize_meta(meta_raw)

    codes = meta["code"].astype(str).tolist()
    missing_cols = [c for c in codes if c not in df_wide.columns]
    if missing_cols:
        raise ValueError(
            f"{WIDE_NAME} missing expected variable columns: {missing_cols}. "
            f"Available: {list(df_wide.columns)}"
        )

    role_map = build_role_map(meta)
    label_map = dict(zip(meta["code"].astype(str), meta["label"].astype(str)))
    return df_wide, meta, codes, role_map, label_map


# =============================================================================
# Plot 1: Small multiples (raw)
# =============================================================================
def plot_small_multiples(
    t_num: pd.Series,
    X: pd.DataFrame,
    role_map: Dict[str, str],
    label_map: Dict[str, str],
    out_dir: Path,
    title: str,
) -> None:
    colors = code_color_map(list(X.columns), role_map)
    ls_map = criterion_linestyles(list(X.columns), role_map)

    n_vars = X.shape[1]
    ncols = 2
    nrows = int(np.ceil(n_vars / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8.4, 2.0 * nrows + 0.5), sharex=True)
    axes = np.array(axes).ravel()

    for ax, code in zip(axes, X.columns):
        y = X[code].to_numpy(dtype=float)
        ylo, yhi = robust_limits(y, 1.0, 99.0)
        pad = 0.08 * (yhi - ylo) if np.isfinite(yhi - ylo) else 1.0

        ax.plot(
            t_num.values,
            y,
            color=colors.get(code, OKABE_ITO["black"]),
            linestyle=ls_map.get(code, "-"),
            alpha=0.95,
        )

        lab = label_map.get(code, "")
        ax.set_title(f"{code}: {lab}", loc="left", pad=2, fontsize=9)
        ax.set_ylim(ylo - pad, yhi + pad)
        ax.grid(True, alpha=0.25)

    for j in range(n_vars, len(axes)):
        axes[j].set_axis_off()

    fig.suptitle(f"{title} — Time series (small multiples; raw scale)", y=0.995)
    fig.supxlabel("Time")
    fig.supylabel("Value")

    fig.tight_layout(rect=[0, 0.0, 1, 0.985])
    save_figure(fig, out_dir / "01_small_multiples_raw")
    plt.close(fig)


# =============================================================================
# Plot 2: Overlay (z-scored)
# =============================================================================
def plot_overlay_zscore(
    t_num: pd.Series,
    X: pd.DataFrame,
    role_map: Dict[str, str],
    out_dir: Path,
    title: str,
) -> None:
    Xz = zscore_df(X)

    colors = code_color_map(list(X.columns), role_map)
    ls_map = criterion_linestyles(list(X.columns), role_map)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for code in X.columns:
        ax.plot(
            t_num.values,
            Xz[code].values,
            label=code,
            color=colors.get(code, OKABE_ITO["black"]),
            linestyle=ls_map.get(code, "-"),
            alpha=0.9,
        )

    ax.set_title(f"{title} — Z-scored overlay (single axis)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Z-score")

    ncols = min(6, max(2, int(np.ceil(X.shape[1] / 2))))

    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncols=ncols,
        frameon=False,
    )

    ax.set_xlim(np.nanmin(t_num.values), np.nanmax(t_num.values))
    fig.tight_layout()
    save_figure(fig, out_dir / "02_overlay_zscore")
    plt.close(fig)


# =============================================================================
# Plot 3: Stacked offsets (z-scored)
# =============================================================================
def plot_stacked_offset(
    t_num: pd.Series,
    X: pd.DataFrame,
    role_map: Dict[str, str],
    out_dir: Path,
    title: str,
) -> None:
    Xz = zscore_df(X)
    colors = code_color_map(list(X.columns), role_map)
    ls_map = criterion_linestyles(list(X.columns), role_map)

    fig, ax = plt.subplots(figsize=(8.4, 0.45 * X.shape[1] + 3.0))

    offset_step = 3.6
    offsets = np.arange(X.shape[1])[::-1] * offset_step

    for code, off in zip(X.columns, offsets):
        alpha = 0.95 if role_map.get(code, "").upper() == "PREDICTOR" else 0.9
        ax.plot(
            t_num.values,
            Xz[code].values + off,
            color=colors.get(code, OKABE_ITO["black"]),
            linestyle=ls_map.get(code, "-"),
            alpha=alpha,
        )

    ax.set_yticks(offsets)
    ax.set_yticklabels(list(X.columns))

    for off in offsets:
        ax.axhline(off, linewidth=0.6, alpha=0.15)

    ax.set_title(f"{title} — Stacked offsets (z-scored; single axis)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Variable")
    ax.set_xlim(np.nanmin(t_num.values), np.nanmax(t_num.values))

    fig.tight_layout()
    save_figure(fig, out_dir / "03_stacked_offsets_zscore")
    plt.close(fig)


# =============================================================================
# Plot 4: Rolling coupling heatmap (P vs C or role-based)
# =============================================================================
def rolling_mean_abs_corr_between_sets(
    X: pd.DataFrame, cols_a: List[str], cols_b: List[str], window: int
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(X)
    if n < 6:
        raise ValueError(f"Time series too short for rolling correlation (n={n}).")
    if window < 5 or window > n:
        raise ValueError(f"Invalid window={window} for n={n}")

    Xa = X[cols_a].to_numpy(dtype=float)
    Xb = X[cols_b].to_numpy(dtype=float)

    half = window // 2
    centers: List[int] = []
    vals: List[float] = []

    for center in range(half, n - half):
        seg_a = Xa[center - half : center + half]
        seg_b = Xb[center - half : center + half]

        seg_a = (seg_a - np.nanmean(seg_a, axis=0)) / (np.nanstd(seg_a, axis=0, ddof=0) + 1e-12)
        seg_b = (seg_b - np.nanmean(seg_b, axis=0)) / (np.nanstd(seg_b, axis=0, ddof=0) + 1e-12)

        corr_ab = (seg_a.T @ seg_b) / max(seg_a.shape[0] - 1, 1)
        centers.append(center)
        vals.append(float(np.nanmean(np.abs(corr_ab))))

    centers_arr = np.array(centers, dtype=int)
    heat = np.array(vals, dtype=float)[:, None]
    return centers_arr, heat


def plot_rolling_coupling_heatmap(
    t_num: pd.Series,
    X_for_corr: pd.DataFrame,
    role_map: Dict[str, str],
    out_dir: Path,
    title: str,
) -> None:
    preds = [c for c in X_for_corr.columns if role_map.get(c, "").upper() == "PREDICTOR"]
    crits = [c for c in X_for_corr.columns if role_map.get(c, "").upper() == "CRITERION"]

    if preds and crits:
        cols_a, cols_b = preds, crits
        label = "PREDICTOR ↔ CRITERION"
    else:
        k = max(1, X_for_corr.shape[1] // 2)
        cols_a, cols_b = list(X_for_corr.columns[:k]), list(X_for_corr.columns[k:])
        label = "group A ↔ group B (fallback)"

    n = len(X_for_corr)
    window = min(ROLLING_WINDOW_DEFAULT, max(8, n // 3))
    if window >= n:
        window = max(5, n - 1)
    if window < 5:
        window = 5

    centers, heat = rolling_mean_abs_corr_between_sets(X_for_corr, cols_a, cols_b, window=window)

    fig, ax = plt.subplots(figsize=(8.4, 2.6))
    im = ax.imshow(
        heat.T,
        aspect="auto",
        interpolation="nearest",
        extent=[t_num.iloc[centers[0]], t_num.iloc[centers[-1]], 0, 1],
        origin="lower",
    )

    ax.set_title(f"{title} — Rolling coupling ({label}; mean |corr|; window={window})")
    ax.set_xlabel("Time")
    ax.set_yticks([])

    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cbar.set_label("Mean |corr|")

    fig.tight_layout()
    save_figure(fig, out_dir / "04_rolling_group_coupling_heatmap")
    plt.close(fig)


# =============================================================================
# Plot 5: Violin distributions (z-scored; role colored)
# =============================================================================
def plot_violins(X: pd.DataFrame, role_map: Dict[str, str], out_dir: Path, title: str) -> None:
    Xz = zscore_df(X)

    codes = list(X.columns)
    data = [Xz[c].to_numpy(dtype=float) for c in codes]

    fig, ax = plt.subplots(figsize=(8.4, 4.6))

    parts = ax.violinplot(
        data,
        positions=np.arange(1, len(codes) + 1),
        widths=0.85,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )

    pal = role_palette()

    for i, body in enumerate(parts["bodies"]):
        code = codes[i]
        role = role_map.get(code, "UNKNOWN").upper()
        col = pal.get(role, OKABE_ITO["gray"])
        body.set_alpha(0.30 if role == "PREDICTOR" else 0.22)
        body.set_edgecolor(col)
        body.set_linewidth(1.0)

    if "cmedians" in parts:
        parts["cmedians"].set_linewidth(1.2)
        parts["cmedians"].set_color(OKABE_ITO["black"])

    rng = np.random.default_rng(123)
    max_points = 120
    for i, code in enumerate(codes, start=1):
        y = Xz[code].to_numpy(dtype=float)
        y = y[np.isfinite(y)]
        if y.size == 0:
            continue
        if y.size > max_points:
            idx = rng.choice(y.size, size=max_points, replace=False)
            y = y[idx]
        x = i + rng.normal(0, 0.04, size=y.size)
        ax.scatter(x, y, s=8, alpha=0.20, linewidths=0)

    ax.set_title(f"{title} — Distributions (violin; z-scored)")
    ax.set_xlabel("Variable (code)")
    ax.set_ylabel("Z-score")
    ax.set_xticks(np.arange(1, len(codes) + 1))
    ax.set_xticklabels(codes, rotation=45, ha="right")

    fig.tight_layout()
    save_figure(fig, out_dir / "05_violin_distributions_zscore")
    plt.close(fig)


# =============================================================================
# Plot 6: Missingness heatmap (variables x time; no imputation)
# =============================================================================
def plot_missingness_heatmap(X_raw: pd.DataFrame, out_dir: Path, title: str) -> None:
    miss = X_raw.isna().to_numpy(dtype=float)  # 1 if missing else 0
    fig, ax = plt.subplots(figsize=(8.4, 0.35 * X_raw.shape[1] + 2.2))
    im = ax.imshow(miss.T, aspect="auto", interpolation="nearest", origin="lower")
    ax.set_title(f"{title} — Missingness map (1=missing)")
    ax.set_xlabel("Time index")
    ax.set_yticks(np.arange(X_raw.shape[1]))
    ax.set_yticklabels(list(X_raw.columns))
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cbar.set_label("Missing (0/1)")
    fig.tight_layout()
    save_figure(fig, out_dir / "06_missingness_heatmap")
    plt.close(fig)


# =============================================================================
# Plot 7: Correlation heatmap (pairwise; impute for viz only)
# =============================================================================
def plot_correlation_heatmap(X_for_corr: pd.DataFrame, out_dir: Path, title: str) -> None:
    # correlation can produce NaNs if constant columns; fill NaNs to 0 for clean plotting
    corr = X_for_corr.corr()
    corr = corr.fillna(0.0).to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    im = ax.imshow(corr, aspect="auto", interpolation="nearest")
    ax.set_title(f"{title} — Pairwise correlation (viz-imputed; NaNs→0)")
    ax.set_xticks(np.arange(X_for_corr.shape[1]))
    ax.set_yticks(np.arange(X_for_corr.shape[1]))
    ax.set_xticklabels(list(X_for_corr.columns), rotation=45, ha="right")
    ax.set_yticklabels(list(X_for_corr.columns))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("corr")
    fig.tight_layout()
    save_figure(fig, out_dir / "07_correlation_heatmap")
    plt.close(fig)


# =============================================================================
# Orchestration
# =============================================================================
def visualize_profile(profile_dir: Path, overwrite: bool = False) -> None:
    log(f"\n[PROFILE] {profile_dir.name}")
    log(f"  - loading inputs...")

    df_wide, meta, codes, role_map, label_map = load_profile_inputs(profile_dir)

    out_dir = profile_dir / VIS_DIRNAME
    if out_dir.exists() and (out_dir / "01_small_multiples_raw.png").exists() and not overwrite:
        log(f"  - visuals already exist (use --overwrite to regenerate): {out_dir}")
        return

    ensure_dir(out_dir)
    log(f"  - output dir: {out_dir}")

    X_raw = df_wide[codes].apply(pd.to_numeric, errors="coerce")
    t_num, _t_label = pick_time_axis(df_wide)

    # For plots needing continuous data (lines/corr): visualization-only fill
    X_viz = X_raw.ffill().bfill()

    title = profile_dir.name

    log("  - plot 01: small multiples...")
    plot_small_multiples(t_num, X_raw, role_map, label_map, out_dir, title)

    log("  - plot 02: overlay zscore...")
    plot_overlay_zscore(t_num, X_raw, role_map, out_dir, title)

    log("  - plot 03: stacked offsets...")
    plot_stacked_offset(t_num, X_raw, role_map, out_dir, title)

    log("  - plot 04: rolling coupling heatmap...")
    plot_rolling_coupling_heatmap(t_num, X_viz, role_map, out_dir, title)

    log("  - plot 05: violin distributions...")
    plot_violins(X_raw, role_map, out_dir, title)

    log("  - plot 06: missingness heatmap...")
    plot_missingness_heatmap(X_raw, out_dir, title)

    log("  - plot 07: correlation heatmap...")
    plot_correlation_heatmap(X_viz, out_dir, title)

    log("  - DONE profile ✓")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=str(DEFAULT_ROOT_PSEUDODATA), help="Root pseudodata directory.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate visuals even if they exist.")
    args = parser.parse_args()

    set_pub_style()

    root = Path(args.root).expanduser()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Root not found or not a directory: {root}")

    profile_dirs = sorted([p for p in root.iterdir() if p.is_dir() and (p / WIDE_NAME).exists()])
    if not profile_dirs:
        raise SystemExit(f"No pseudoprofile directories found under: {root} (expected {WIDE_NAME})")

    log(f"[INFO] root: {root}")
    log(f"[INFO] profiles found: {len(profile_dirs)}")
    log(f"[INFO] overwrite: {bool(args.overwrite)}")

    for i, prof_dir in enumerate(profile_dirs, start=1):
        log(f"\n[INFO] ({i}/{len(profile_dirs)}) processing {prof_dir.name}")
        try:
            visualize_profile(prof_dir, overwrite=bool(args.overwrite))
        except Exception as e:
            log(f"[ERROR] Failed on profile {prof_dir.name}: {type(e).__name__}: {e}")
            raise  # fail fast as requested

    log("\nAll done.")


if __name__ == "__main__":
    main()
