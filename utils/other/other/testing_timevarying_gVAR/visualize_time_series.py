#!/usr/bin/env python3
"""
Publication-ready visualization for 3 scenarios of 10-variable time series.

Creates 5 complementary, journal-friendly plots PER SCENARIO and saves BOTH PNG (300dpi)
and vector PDF into each scenario's /data/visuals directory.

Scenarios (given):
- scenario_scenario1_smooth/data/X_raw.csv
- scenario_scenario2_abrupt/data/X_raw.csv
- scenario_scenario3_periodic/data/X_raw.csv

Variables:
P predictors: P1..P6
C criterions: C1..C4

Plots (5):
1) Small multiples (10 panels) time series (raw units)  [most interpretable]
2) Z-scored overlay time series (single axis)           [comparability]
3) Stacked offset (z-scored; single axis)              [decluttered]
4) Rolling correlation heatmap (avg P vs C; windowed)  [structure over time]
5) Violin plots (distribution; P vs C; robust)         [distributional view]

Only visualization outputs: no prints, no tables saved.
Errors raise exceptions (fail fast for reproducibility).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# =============================
# Configuration
# =============================

ROOT_RESULTS = Path(
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/other/other/"
    "testing_timevarying_gVAR/results"
)

SCENARIOS = [
    "scenario_scenario1_smooth",
    "scenario_scenario2_abrupt",
    "scenario_scenario3_periodic",
]

VARS = ["P1", "P2", "P3", "P4", "P5", "P6", "C1", "C2", "C3", "C4"]
P_VARS = ["P1", "P2", "P3", "P4", "P5", "P6"]
C_VARS = ["C1", "C2", "C3", "C4"]

CSV_NAME = "X_raw.csv"
VIS_DIRNAME = "visuals"

# Rolling window for correlation heatmap (tune as needed)
ROLLING_WINDOW = 50  # ~6.25% of 800 points


# =============================
# Publication style
# =============================

# Okabe–Ito colorblind-friendly palette
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
    # Conservative defaults that export cleanly to PDF and look good in print.
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
            "pdf.fonttype": 42,  # TrueType in PDF (better text handling)
            "ps.fonttype": 42,
        }
    )


# =============================
# Utilities
# =============================

def zscore(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean(axis=0, skipna=True)
    sd = df.std(axis=0, ddof=0, skipna=True).replace(0, np.nan)
    return (df - mu) / sd


def robust_limits(x: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> Tuple[float, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (-1.0, 1.0)
    return (np.percentile(x, lo), np.percentile(x, hi))


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, out_base: Path) -> None:
    """
    Save figure as PNG and PDF.
    out_base should be a path WITHOUT suffix.
    """
    ensure_dir(out_base.parent)
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    #fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")  # vector PDF


def load_data(csv_path: Path, vars_: List[str]) -> Tuple[pd.Series, pd.DataFrame]:
    df = pd.read_csv(csv_path)

    missing = [c for c in vars_ if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {csv_path}: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # Prefer a time column if present; else integer index
    time_col = None
    for cand in ["time", "t", "Time", "T"]:
        if cand in df.columns:
            time_col = cand
            break

    if time_col is not None:
        t = pd.to_numeric(df[time_col], errors="coerce")
    else:
        t = pd.Series(np.arange(len(df)), name="t")

    X = df[vars_].apply(pd.to_numeric, errors="coerce")

    # Visualization-only handling: fill small gaps to avoid broken lines.
    X = X.ffill().bfill()

    return t, X


def scenario_paths(root: Path, scenario_name: str) -> Tuple[Path, Path]:
    data_dir = root / scenario_name / "data"
    csv_path = data_dir / CSV_NAME
    out_dir = data_dir / VIS_DIRNAME
    return csv_path, out_dir


def var_colors() -> Dict[str, str]:
    # Predictors in distinct colors; criterions in black with line styles.
    return {
        "P1": OKABE_ITO["blue"],
        "P2": OKABE_ITO["sky"],
        "P3": OKABE_ITO["green"],
        "P4": OKABE_ITO["orange"],
        "P5": OKABE_ITO["purple"],
        "P6": OKABE_ITO["vermillion"],
        "C1": OKABE_ITO["black"],
        "C2": OKABE_ITO["black"],
        "C3": OKABE_ITO["black"],
        "C4": OKABE_ITO["black"],
    }


def c_linestyles() -> Dict[str, str]:
    return {"C1": "-", "C2": "--", "C3": "-.", "C4": ":"}


# =============================
# Plot 1: Small multiples (10 panels)
# =============================

def plot_small_multiples(t: pd.Series, X: pd.DataFrame, out_dir: Path, title: str) -> None:
    colors = var_colors()
    ls_map = c_linestyles()

    fig, axes = plt.subplots(
        nrows=5, ncols=2, figsize=(8.4, 10.2), sharex=True
    )
    axes = axes.ravel()

    # Robust y-limits per subplot for readability (reduces impact of spikes)
    for ax, v in zip(axes, VARS):
        y = X[v].to_numpy(dtype=float)
        ylo, yhi = robust_limits(y, 1.0, 99.0)
        pad = 0.08 * (yhi - ylo) if np.isfinite(yhi - ylo) else 1.0

        ax.plot(
            t.values,
            y,
            color=colors.get(v, OKABE_ITO["black"]),
            linestyle=ls_map.get(v, "-"),
            alpha=0.95,
        )
        ax.set_title(v, loc="left", pad=2)
        ax.set_ylim(ylo - pad, yhi + pad)
        ax.grid(True, alpha=0.25)

    # Remove unused axes (none here) and set common labels
    fig.suptitle(f"{title} — Time series (small multiples; raw scale)", y=0.995)
    fig.supxlabel("Time")
    fig.supylabel("Value")

    fig.tight_layout(rect=[0, 0.0, 1, 0.985])
    save_figure(fig, out_dir / "01_small_multiples_raw")
    plt.close(fig)


# =============================
# Plot 2: Overlay (z-scored, single axis)
# =============================

def plot_overlay_zscore(t: pd.Series, X: pd.DataFrame, out_dir: Path, title: str) -> None:
    Xz = zscore(X)

    colors = var_colors()
    ls_map = c_linestyles()

    fig, ax = plt.subplots(figsize=(8.4, 4.8))

    for v in VARS:
        ax.plot(
            t.values,
            Xz[v].values,
            label=v,
            color=colors.get(v, OKABE_ITO["black"]),
            linestyle=ls_map.get(v, "-"),
            alpha=0.9,
        )

    ax.set_title(f"{title} — Z-scored overlay (single axis)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Z-score")

    # Legend outside for publication cleanliness
    ax.legend(
        ncols=5,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        columnspacing=1.0,
        handlelength=2.0,
    )

    ax.set_xlim(np.nanmin(t.values), np.nanmax(t.values))
    fig.tight_layout()
    save_figure(fig, out_dir / "02_overlay_zscore")
    plt.close(fig)


# =============================
# Plot 3: Stacked offsets (z-scored, single axis)
# =============================

def plot_stacked_offset(t: pd.Series, X: pd.DataFrame, out_dir: Path, title: str) -> None:
    Xz = zscore(X)

    colors = var_colors()
    ls_map = c_linestyles()

    fig, ax = plt.subplots(figsize=(8.4, 6.0))

    offset_step = 3.6
    offsets = np.arange(len(VARS))[::-1] * offset_step

    for v, off in zip(VARS, offsets):
        ax.plot(
            t.values,
            Xz[v].values + off,
            color=colors.get(v, OKABE_ITO["black"]),
            linestyle=ls_map.get(v, "-"),
            alpha=0.95 if v.startswith("P") else 0.9,
        )

    ax.set_yticks(offsets)
    ax.set_yticklabels(VARS)

    # Light reference baselines
    for off in offsets:
        ax.axhline(off, linewidth=0.6, alpha=0.15)

    ax.set_title(f"{title} — Stacked offsets (z-scored; single axis)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Variable")
    ax.set_xlim(np.nanmin(t.values), np.nanmax(t.values))

    fig.tight_layout()
    save_figure(fig, out_dir / "03_stacked_offsets_zscore")
    plt.close(fig)


# =============================
# Plot 4: Rolling correlation heatmap (P vs C summary)
# =============================

def rolling_corr_heatmap_pc(
    X: pd.DataFrame,
    window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      centers: time index centers (int)
      heat: (len(centers), 1) average absolute corr between P and C sets
    """
    n = len(X)
    if window < 5 or window > n:
        raise ValueError(f"Invalid window={window} for n={n}")

    # Precompute arrays
    Xp = X[P_VARS].to_numpy(dtype=float)
    Xc = X[C_VARS].to_numpy(dtype=float)

    centers = []
    vals = []

    # Rolling window correlation between all Pi-Cj pairs, averaged (abs)
    # Complexity: O(n * window * p*c) is fine here (800*50*24 = 960k ops approx)
    half = window // 2
    for center in range(half, n - half):
        seg_p = Xp[center - half : center + half]
        seg_c = Xc[center - half : center + half]

        # z-score within window for stable correlations
        seg_p = (seg_p - np.nanmean(seg_p, axis=0)) / (np.nanstd(seg_p, axis=0, ddof=0) + 1e-12)
        seg_c = (seg_c - np.nanmean(seg_c, axis=0)) / (np.nanstd(seg_c, axis=0, ddof=0) + 1e-12)

        # corr matrix between P (6) and C (4): (6 x 4)
        # corr = (P^T C) / (window-1) approximately; we use mean of products
        corr_pc = (seg_p.T @ seg_c) / max(seg_p.shape[0] - 1, 1)

        centers.append(center)
        vals.append(np.nanmean(np.abs(corr_pc)))

    centers_arr = np.array(centers, dtype=int)
    heat = np.array(vals, dtype=float)[:, None]  # make it 2D for imshow
    return centers_arr, heat


def plot_rolling_corr_heatmap(t: pd.Series, X: pd.DataFrame, out_dir: Path, title: str) -> None:
    centers, heat = rolling_corr_heatmap_pc(X, window=ROLLING_WINDOW)

    fig, ax = plt.subplots(figsize=(8.4, 2.6))

    # Use a neutral colormap; value in [0,1] typical
    im = ax.imshow(
        heat.T,
        aspect="auto",
        interpolation="nearest",
        extent=[t.iloc[centers[0]], t.iloc[centers[-1]], 0, 1],
        origin="lower",
    )

    ax.set_title(f"{title} — Rolling P↔C coupling (mean |corr|; window={ROLLING_WINDOW})")
    ax.set_xlabel("Time")
    ax.set_yticks([])

    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cbar.set_label("Mean |corr(P,C)|")

    fig.tight_layout()
    save_figure(fig, out_dir / "04_rolling_pc_coupling_heatmap")
    plt.close(fig)


# =============================
# Plot 5: Violin plots (distribution; P vs C; z-scored)
# =============================

def plot_violins(t: pd.Series, X: pd.DataFrame, out_dir: Path, title: str) -> None:
    """
    Violin plots show marginal distributions; for time series this complements
    trajectory plots (publication common: distribution + dynamics).
    """
    Xz = zscore(X)

    data = [Xz[v].to_numpy(dtype=float) for v in VARS]

    fig, ax = plt.subplots(figsize=(8.4, 4.4))

    parts = ax.violinplot(
        data,
        positions=np.arange(1, len(VARS) + 1),
        widths=0.85,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )

    # Style violins: predictors tinted, criterions grayscale
    for i, body in enumerate(parts["bodies"]):
        v = VARS[i]
        if v.startswith("P"):
            body.set_alpha(0.35)
            body.set_edgecolor(var_colors()[v])
            body.set_linewidth(1.0)
        else:
            body.set_alpha(0.25)
            body.set_edgecolor(OKABE_ITO["black"])
            body.set_linewidth(1.0)

    if "cmedians" in parts:
        parts["cmedians"].set_linewidth(1.2)
        parts["cmedians"].set_color(OKABE_ITO["black"])

    # Add subtle jittered points (downsample for clarity)
    rng = np.random.default_rng(123)
    max_points_per_var = 120
    for i, v in enumerate(VARS, start=1):
        y = Xz[v].to_numpy(dtype=float)
        y = y[np.isfinite(y)]
        if y.size == 0:
            continue
        if y.size > max_points_per_var:
            idx = rng.choice(y.size, size=max_points_per_var, replace=False)
            y = y[idx]
        x = i + rng.normal(0, 0.04, size=y.size)
        ax.scatter(x, y, s=8, alpha=0.20, linewidths=0)

    ax.set_title(f"{title} — Distributions (violin; z-scored)")
    ax.set_xlabel("Variable")
    ax.set_ylabel("Z-score")
    ax.set_xticks(np.arange(1, len(VARS) + 1))
    ax.set_xticklabels(VARS)

    # Visual separator between P and C
    ax.axvline(len(P_VARS) + 0.5, linewidth=1.0, alpha=0.35)

    fig.tight_layout()
    save_figure(fig, out_dir / "05_violin_distributions_zscore")
    plt.close(fig)


# =============================
# Orchestration
# =============================

def visualize_scenario(scenario_name: str) -> None:
    csv_path, out_dir = scenario_paths(ROOT_RESULTS, scenario_name)
    t, X = load_data(csv_path, VARS)
    ensure_dir(out_dir)

    title = scenario_name.replace("_", " ")

    plot_small_multiples(t, X, out_dir, title=title)
    plot_overlay_zscore(t, X, out_dir, title=title)
    plot_stacked_offset(t, X, out_dir, title=title)
    plot_rolling_corr_heatmap(t, X, out_dir, title=title)
    plot_violins(t, X, out_dir, title=title)


def main() -> None:
    set_pub_style()
    for scen in SCENARIOS:
        visualize_scenario(scen)


if __name__ == "__main__":
    main()
