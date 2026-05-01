"""
Shared statistical and visualization utilities for the PHOENIX evaluation framework.

Design philosophy
-----------------
* Every figure function returns (fig, ax/axes) or accepts an existing Axes and returns it,
  so callers can combine panels freely.
* Raincloud plot (Allen et al. 2019) is the primary within-subjects comparison figure;
  it exposes 01_raw data, density, and summary statistics in one compact panel.
* Forest plot includes Cohen's d effect-size bands and significance annotations.
* TOST equivalence testing is provided alongside conventional NHST because the primary
  research question is "does PHOENIX perform at least as well as separate_HCPs?" — equivalence
  testing is more appropriate than failing to reject a null of difference.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from matplotlib.colors import to_rgba
from matplotlib.font_manager import FontProperties, findfont

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────
PALETTE: Dict[str, str] = {
    "primary":   "#4f46e5",   # indigo   — PHOENIX / AI output
    "secondary": "#10b981",   # emerald  — HCP / human comparator
    "tertiary":  "#f59e0b",   # amber    — third group / accent
    "neutral":   "#9ca3af",   # mid-gray — non-significant / trivial
    "ref_line":  "#6b7280",   # dark-gray reference lines
    "danger":    "#ef4444",   # red      — below-reference
    "equiv":     "#3b82f6",   # blue     — equivalence band
}

COLOR_PHOENIX = PALETTE["primary"]
COLOR_HCP     = PALETTE["secondary"]

# Dimension-level color rotation (for multi-panel dimension strips)
DIM_COLORS: List[str] = [
    "#4f46e5", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4",
]

# ─────────────────────────────────────────────────────────────────────────────
# Global rcParams — publication quality
# ─────────────────────────────────────────────────────────────────────────────
RCPARAMS: Dict[str, Any] = {
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.labelsize":     12,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         150,
    "axes.grid":          True,
    "grid.alpha":         0.18,
    "grid.linestyle":     "--",
}


def apply_rcparams() -> None:
    plt.rcParams.update(RCPARAMS)


apply_rcparams()


def get_preferred_font() -> str:
    for name in ["Inter", "Arial", "DejaVu Sans"]:
        fp = FontProperties(family=name)
        path = findfont(fp, fallback_to_default=False)
        if path and name.lower() in path.lower():
            return name
    return "DejaVu Sans"


# ─────────────────────────────────────────────────────────────────────────────
# Statistical helpers
# ─────────────────────────────────────────────────────────────────────────────

def shapiro_wilk_test(
    values: np.ndarray,
) -> Tuple[Optional[float], Optional[float], bool]:
    if len(values) < 3:
        return None, None, True
    try:
        stat, p = stats.shapiro(values)
        return float(stat), float(p), bool(p > 0.05)
    except Exception:
        return None, None, True


def cohen_d(group1: np.ndarray, group2: np.ndarray) -> float:
    n1, n2 = len(group1), len(group2)
    v1 = np.var(group1, ddof=1)
    v2 = np.var(group2, ddof=1)
    pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    return float((np.mean(group1) - np.mean(group2)) / pooled) if pooled else 0.0


def rank_biserial_r(U: float, n1: int, n2: int) -> float:
    return float(1 - 2 * U / (n1 * n2)) if n1 * n2 > 0 else 0.0


def effect_size_r(z_stat: float, n_total: int) -> float:
    return float(z_stat / np.sqrt(n_total)) if n_total > 0 else 0.0


def ci_mean(
    values: np.ndarray, confidence: float = 0.95
) -> Tuple[float, float, float]:
    n = len(values)
    if n < 2:
        m = float(np.mean(values))
        return m, m, m
    se = stats.sem(values)
    t_crit = stats.t.ppf((1 + confidence) / 2.0, df=n - 1)
    m = float(np.mean(values))
    return m, float(m - t_crit * se), float(m + t_crit * se)


def bootstrap_ci_mean(
    values: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    boot = [
        np.mean(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ]
    lo = float(np.percentile(boot, (1 - confidence) / 2 * 100))
    hi = float(np.percentile(boot, (1 + confidence) / 2 * 100))
    return lo, hi


def bootstrap_cohend_ci(
    g1: np.ndarray,
    g2: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    ds = [
        cohen_d(
            rng.choice(g1, size=len(g1), replace=True),
            rng.choice(g2, size=len(g2), replace=True),
        )
        for _ in range(n_boot)
    ]
    lo = float(np.percentile(ds, (1 - confidence) / 2 * 100))
    hi = float(np.percentile(ds, (1 + confidence) / 2 * 100))
    return lo, hi


def tost_test(
    vals_phoenix: np.ndarray,
    vals_hcp: np.ndarray,
    delta: float = 0.5,
) -> Dict[str, Any]:
    """
    Two One-Sided Tests (TOST) for equivalence.

    Tests whether the true mean difference falls within the interval [-delta, +delta].
    A significant result (p_tost < 0.05) supports equivalence: PHOENIX performs
    within `delta` of HCP performance in both directions.

    Parameters
    ----------
    delta : float
        Equivalence margin in rating-scale units. Default 0.5 corresponds to
        half a Likert step on the 1..7 anchored scale used by the LLM judge,
        approximating a minimal clinically important difference.
    """
    n1, n2 = len(vals_phoenix), len(vals_hcp)
    mean_diff = float(np.mean(vals_phoenix) - np.mean(vals_hcp))
    pooled_se = float(
        np.sqrt(
            np.var(vals_phoenix, ddof=1) / n1 + np.var(vals_hcp, ddof=1) / n2
        )
    )
    df = n1 + n2 - 2

    # Upper one-sided test: H0: diff >= delta  (reject means diff < delta)
    t_upper = (mean_diff - delta) / pooled_se if pooled_se > 0 else 0.0
    p_upper = float(stats.t.cdf(t_upper, df=df))

    # Lower one-sided test: H0: diff <= -delta  (reject means diff > -delta)
    t_lower = (mean_diff + delta) / pooled_se if pooled_se > 0 else 0.0
    p_lower = float(stats.t.sf(t_lower, df=df))

    p_tost = max(p_upper, p_lower)
    return {
        "p_tost":        p_tost,
        "p_upper":       p_upper,
        "p_lower":       p_lower,
        "t_upper":       float(t_upper),
        "t_lower":       float(t_lower),
        "delta":         float(delta),
        "observed_diff": mean_diff,
        "pooled_se":     float(pooled_se),
        "equivalent":    bool(p_tost < 0.05),
    }


def tost_test_one_sample(
    values: np.ndarray,
    delta: float = 1.0,
) -> Dict[str, Any]:
    """
    One-sample TOST for signed PHOENIX-vs-HCP preference scores.

    Tests whether the mean signed score lies inside [-delta, +delta].
    A significant result supports practical equivalence; a positive mean
    outside the band favours PHOENIX, and a negative mean outside the band
    favours HCP.
    """
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    n = len(vals)
    mean_diff = float(np.mean(vals)) if n else 0.0
    if n < 2:
        return {
            "p_tost": 1.0,
            "p_upper": 1.0,
            "p_lower": 1.0,
            "t_upper": 0.0,
            "t_lower": 0.0,
            "delta": float(delta),
            "observed_diff": mean_diff,
            "pooled_se": 0.0,
            "equivalent": False,
        }
    se = float(stats.sem(vals))
    df = n - 1
    if se <= 0:
        inside = abs(mean_diff) < float(delta)
        return {
            "p_tost": 0.0 if inside else 1.0,
            "p_upper": 0.0 if mean_diff < float(delta) else 1.0,
            "p_lower": 0.0 if mean_diff > -float(delta) else 1.0,
            "t_upper": float("-inf") if mean_diff < float(delta) else float("inf"),
            "t_lower": float("inf") if mean_diff > -float(delta) else float("-inf"),
            "delta": float(delta),
            "observed_diff": mean_diff,
            "pooled_se": se,
            "equivalent": bool(inside),
        }
    t_upper = (mean_diff - delta) / se
    p_upper = float(stats.t.cdf(t_upper, df=df))
    t_lower = (mean_diff + delta) / se
    p_lower = float(stats.t.sf(t_lower, df=df))
    p_tost = max(p_upper, p_lower)
    return {
        "p_tost": float(p_tost),
        "p_upper": p_upper,
        "p_lower": p_lower,
        "t_upper": float(t_upper),
        "t_lower": float(t_lower),
        "delta": float(delta),
        "observed_diff": mean_diff,
        "pooled_se": se,
        "equivalent": bool(p_tost < 0.05),
    }


def cohen_d_one_sample(values: np.ndarray, mu: float = 0.0) -> float:
    """Standardised one-sample effect size for signed scores."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2:
        return 0.0
    sd = float(np.std(vals, ddof=1))
    return float((np.mean(vals) - mu) / sd) if sd else 0.0


def bonferroni_correct(p_values: List[float]) -> List[float]:
    n = len(p_values)
    return [min(p * n, 1.0) for p in p_values]


def holm_correct(p_values: List[float]) -> List[float]:
    """
    Holm-Bonferroni step-down correction.

    Returns adjusted p-values in the original order. More powerful than
    Bonferroni while still controlling the family-wise error rate.
    """
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda kv: kv[1])
    adjusted = [1.0] * n
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed, start=1):
        scaled = (n - rank + 1) * float(p)
        running_max = max(running_max, scaled)
        adjusted[orig_idx] = min(running_max, 1.0)
    return adjusted


def p_to_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _candidate_vc_formula(
    variance_components: Optional[Dict[str, str]],
) -> Optional[Dict[str, str]]:
    if not variance_components:
        return None
    vc = {
        name: f"0 + C({col})"
        for name, col in variance_components.items()
        if col
    }
    return vc or None


def fit_crossed_mixedlm(
    data: pd.DataFrame,
    formula: str,
    effect_term: str,
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Try a sequence of mixed-model specifications and return the first converged fit.
    """
    errors: List[str] = []
    for candidate in candidates:
        group_col = candidate.get("group_col")
        if not group_col or group_col not in data.columns:
            continue
        if data[group_col].nunique() < 2:
            continue
        vc_formula = _candidate_vc_formula(candidate.get("variance_components"))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model = smf.mixedlm(
                    formula,
                    data=data,
                    groups=data[group_col],
                    re_formula=candidate.get("re_formula"),
                    vc_formula=vc_formula,
                ).fit(reml=True, method="lbfgs")
            coef = float(model.params.get(effect_term, 0.0))
            se   = float(model.bse.get(effect_term, 0.0))
            residuals = np.asarray(model.resid, dtype=float)
            residuals = residuals[np.isfinite(residuals)]
            if residuals.size >= 3:
                try:
                    sw, sp = stats.shapiro(residuals[:5000])
                    shapiro_w, shapiro_p = float(sw), float(sp)
                except Exception:
                    shapiro_w = shapiro_p = float("nan")
            else:
                shapiro_w = shapiro_p = float("nan")
            return {
                "method":             candidate["label"],
                "group_col":          group_col,
                "variance_components": candidate.get("variance_components", {}),
                "coefficient":        coef,
                "se":                 se,
                "ci_lower":           coef - 1.96 * se,
                "ci_upper":           coef + 1.96 * se,
                "p_value":            float(model.pvalues.get(effect_term, 1.0)),
                "converged":          bool(model.converged),
                "shapiro_w":          shapiro_w,
                "shapiro_p":          shapiro_p,
            }
        except Exception as exc:
            errors.append(f"{candidate['label']}: {exc}")
    return {
        "method":             "No converged mixed model",
        "group_col":          None,
        "variance_components": {},
        "coefficient":        0.0,
        "se":                 0.0,
        "ci_lower":           0.0,
        "ci_upper":           0.0,
        "p_value":            1.0,
        "converged":          False,
        "shapiro_w":          float("nan"),
        "shapiro_p":          float("nan"),
        "error":              " | ".join(errors) or "No valid candidate.",
    }


def run_mann_whitney(g1: np.ndarray, g2: np.ndarray) -> Dict[str, Any]:
    try:
        stat, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
        n = len(g1) + len(g2)
        z = float(stats.norm.ppf(p / 2))
        return {
            "test":          "Mann-Whitney U",
            "statistic":     float(stat),
            "z_approx":      z,
            "p_value":       float(p),
            "effect_size_r": abs(effect_size_r(z, n)),
            "n_total":       n,
        }
    except Exception as exc:
        return {"test": "Mann-Whitney U", "error": str(exc), "p_value": 1.0,
                "statistic": 0.0, "effect_size_r": 0.0}


# ─────────────────────────────────────────────────────────────────────────────
# Figure helpers — save and annotation
# ─────────────────────────────────────────────────────────────────────────────

def save_figure(fig: plt.Figure, path: Path, dpi: int = 300) -> None:
    fig.patch.set_facecolor("white")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved figure: {path}")


def add_significance_bracket(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y: float,
    p_value: float,
    h: float = 0.15,
    color: str = "black",
    lw: float = 1.2,
) -> None:
    stars = p_to_stars(p_value)
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=lw, color=color)
    ax.text(
        (x1 + x2) / 2, y + h + 0.02, stars,
        ha="center", va="bottom", fontsize=12, color=color,
        fontweight="bold" if stars != "ns" else "normal",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Raincloud plot  (Allen et al. 2019, doi:10.12688/wellcomeopenres.15191.2)
# ─────────────────────────────────────────────────────────────────────────────

def raincloud_plot(
    ax: plt.Axes,
    data_dict: Dict[str, np.ndarray],
    title: str,
    ylabel: str,
    colors: Optional[List[str]] = None,
    adj_p: Optional[float] = None,
    ylim: Optional[Tuple[float, float]] = None,
    show_tost: Optional[Dict[str, Any]] = None,
) -> plt.Axes:
    """
    Raincloud plot: half-violin KDE + narrow boxplot + value-gradient jitter scatter.

    Layout per group (left → right)
    --------------------------------
    ← KDE half-violin │ IQR box (narrow) │ jitter scatter →

    Parameters
    ----------
    show_tost : dict or None
        If provided, draws an equivalence bracket. Expected keys:
        ``equivalent`` (bool), ``delta`` (float), ``p_tost`` (float).
    """
    from scipy.stats import gaussian_kde

    apply_rcparams()
    labels    = list(data_dict.keys())
    n_groups  = len(labels)
    positions = np.arange(1, n_groups + 1, dtype=float) * 1.6

    if colors is None:
        default = [PALETTE["primary"], PALETTE["secondary"]]
        colors  = [default[i % len(default)] for i in range(n_groups)]

    rng = np.random.default_rng(42)

    for idx, (label, pos) in enumerate(zip(labels, positions)):
        vals  = np.asarray(data_dict[label], dtype=float)
        vals  = vals[np.isfinite(vals)]
        color = colors[idx]
        rgba  = to_rgba(color)

        # ── 1. Half-violin (KDE on LEFT side) ──────────────────────────────
        if len(vals) >= 3:
            bandwidth = max(0.3, np.std(vals, ddof=1) * len(vals) ** (-0.2))
            kde = gaussian_kde(vals, bw_method=bandwidth)
            y_grid   = np.linspace(vals.min() - 0.6, vals.max() + 0.6, 300)
            density  = kde(y_grid)
            # normalize to half-width of 0.30
            d_scaled = density / density.max() * 0.30
            ax.fill_betweenx(
                y_grid, pos - d_scaled, pos,
                color=color, alpha=0.38, linewidth=0,
            )
            ax.plot(pos - d_scaled, y_grid, color=color, lw=1.0, alpha=0.55)

        # ── 2. Boxplot (narrow, centered) ───────────────────────────────────
        q25, med, q75 = np.percentile(vals, [25, 50, 75])
        iqr = q75 - q25
        wlo = max(vals.min(), q25 - 1.5 * iqr)
        whi = min(vals.max(), q75 + 1.5 * iqr)
        bw  = 0.07
        box = mpatches.FancyBboxPatch(
            (pos - bw / 2, q25), bw, iqr,
            boxstyle="round,pad=0.01",
            facecolor=color, edgecolor=color, alpha=0.72, zorder=4,
        )
        ax.add_patch(box)
        ax.plot([pos - bw / 2, pos + bw / 2], [med, med],
                color="white", lw=2.2, zorder=5)
        ax.plot([pos, pos], [wlo, q25], color=color, lw=1.2, zorder=3)
        ax.plot([pos, pos], [q75, whi], color=color, lw=1.2, zorder=3)
        for end in [wlo, whi]:
            ax.plot([pos - bw / 4, pos + bw / 4], [end, end],
                    color=color, lw=1.0, zorder=3)

        # ── 3. Gradient scatter (RIGHT side) ────────────────────────────────
        jitter = rng.uniform(0.12, 0.38, size=len(vals))
        # Colour each point by its normalized value for visual density gradient
        v_norm  = (vals - vals.min()) / (np.ptp(vals) + 1e-9)
        pt_rgba = np.array([cm.coolwarm(v) for v in v_norm])
        # Blend with group colour (50/50)
        pt_rgba[:, :3] = 0.5 * pt_rgba[:, :3] + 0.5 * np.array(rgba[:3])
        pt_rgba[:, 3]  = 0.65
        ax.scatter(
            pos + jitter, vals,
            color=pt_rgba, s=16,
            edgecolors="white", linewidths=0.3, zorder=6,
        )

        # ── 4. Mean + 95% bootstrap CI (diamond) ────────────────────────────
        m = float(np.mean(vals))
        ci_lo, ci_hi = bootstrap_ci_mean(vals, n_boot=1000, seed=42 + idx)
        ax.vlines(pos + 0.03, ci_lo, ci_hi,
                  color=color, lw=2.5, zorder=7, alpha=0.85)
        ax.plot(pos + 0.03, m, "D", ms=7, color="white",
                markeredgecolor=color, markeredgewidth=1.6, zorder=8)

    # ── Significance bracket ─────────────────────────────────────────────────
    if adj_p is not None and n_groups == 2:
        all_vals  = np.concatenate([
            np.asarray(v, dtype=float) for v in data_dict.values()
        ])
        y_top    = float(np.nanmax(all_vals))
        y_rng    = y_top - float(np.nanmin(all_vals))
        y_brack  = y_top + y_rng * 0.09
        h_brack  = y_rng * 0.04
        add_significance_bracket(
            ax, positions[0], positions[1], y_brack, adj_p, h=h_brack,
        )

    # ── TOST equivalence annotation ─────────────────────────────────────────
    if show_tost is not None and n_groups == 2:
        eq   = show_tost.get("equivalent", False)
        pt   = show_tost.get("p_tost", 1.0)
        dlt  = show_tost.get("delta", 0.5)
        col  = "#15803d" if eq else PALETTE["neutral"]
        label = (
            f"TOST: p={pt:.3f} — equiv. (±{dlt})"
            if eq
            else f"TOST: p={pt:.3f} — not equiv. (±{dlt})"
        )
        ax.annotate(
            label, xy=(0.5, -0.14), xycoords="axes fraction",
            ha="center", va="top", fontsize=8, color=col,
            fontstyle="italic",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlim(positions[0] - 0.75, positions[-1] + 0.65)
    if ylim is not None:
        ax.set_ylim(ylim)
    return ax


# Keep backward-compat alias
def violin_with_scatter(
    ax: plt.Axes,
    data_dict: Dict[str, np.ndarray],
    title: str,
    ylabel: str,
    colors: Optional[List[str]] = None,
    p_value: Optional[float] = None,
    adj_p: Optional[float] = None,
    ref_line: Optional[float] = None,
    ref_label: Optional[str] = None,
    ylim: Optional[Tuple[float, float]] = None,
    add_mean_sd: bool = True,
) -> plt.Axes:
    """Backward-compatible wrapper — delegates to raincloud_plot."""
    ax = raincloud_plot(
        ax, data_dict, title, ylabel,
        colors=colors,
        adj_p=adj_p if adj_p is not None else p_value,
        ylim=ylim,
    )
    if ref_line is not None:
        ax.axhline(
            ref_line, color=PALETTE["ref_line"], linestyle="--",
            linewidth=1.0, alpha=0.8,
            label=ref_label if ref_label else f"y = {ref_line}",
        )
        ax.legend(fontsize=9)
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# Forest plot — enhanced with effect-size bands and star annotations
# ─────────────────────────────────────────────────────────────────────────────

def forest_plot(
    ax: plt.Axes,
    dimensions: Sequence[str],
    effects: Sequence[float],
    ci_lowers: Sequence[float],
    ci_uppers: Sequence[float],
    title: str,
    xlabel: str = "Effect Size (coefficient)",
    ref_line: float = 0.0,
    p_values: Optional[Sequence[float]] = None,
    tost_results: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
) -> plt.Axes:
    """
    Horizontal forest plot with:
    • Shaded Cohen's-d magnitude bands (trivial / small / medium / large)
    • Gradient-width CI lines (thicker near point estimate)
    • Significance stars and TOST equivalence indicators
    • Colour coding by direction (PHOENIX > HCP vs <)
    """
    apply_rcparams()
    n     = len(dimensions)
    y_pos = np.arange(n, dtype=float)

    # Effect-size bands (scale-agnostic shading)
    for lo, hi, alph, col in [
        (-0.2, 0.2,  0.05, "#6b7280"),   # trivial
        ( 0.2, 0.5,  0.07, "#3b82f6"),   # small positive
        (-0.5, -0.2, 0.07, "#ef4444"),   # small negative
        ( 0.5, 0.8,  0.10, "#10b981"),   # medium positive
        (-0.8, -0.5, 0.10, "#f59e0b"),   # medium negative
    ]:
        ax.axvspan(lo, hi, alpha=alph, color=col, zorder=0)

    band_patches = [
        mpatches.Patch(color="#6b7280", alpha=0.20, label="Trivial (<0.2)"),
        mpatches.Patch(color="#3b82f6", alpha=0.25, label="Small (0.2–0.5)"),
        mpatches.Patch(color="#10b981", alpha=0.28, label="Medium (0.5–0.8)"),
    ]
    ax.legend(handles=band_patches, loc="lower right", fontsize=7,
              framealpha=0.7, borderpad=0.5)

    for i, (dim, eff, lo, hi) in enumerate(
        zip(dimensions, effects, ci_lowers, ci_uppers)
    ):
        color = PALETTE["primary"] if eff >= 0 else PALETTE["secondary"]
        # Thick outer CI, thin inner CI (gradient-width effect)
        ax.plot([lo, hi], [i, i], color=color, lw=3.0, alpha=0.20,
                solid_capstyle="round")
        ax.plot([lo, hi], [i, i], color=color, lw=1.5, alpha=0.70,
                solid_capstyle="round")
        ax.plot(eff, i, "o", color=color, ms=9, zorder=5,
                markeredgecolor="white", markeredgewidth=1.2)

        # Numeric annotation
        stars = p_to_stars(p_values[i]) if p_values else ""
        annot = f"{eff:+.3f} [{lo:.3f}, {hi:.3f}]"
        if stars:
            annot += f"  {stars}"
        ax.text(
            max(ci_uppers) * 1.05 + 0.02, i, annot,
            va="center", ha="left", fontsize=8, color=color,
        )

        # TOST indicator (small eq/ne badge)
        if tost_results and tost_results[i] is not None:
            eq   = tost_results[i].get("equivalent", False)
            badge_col = "#15803d" if eq else "#9ca3af"
            badge_txt = "≡" if eq else "≠"
            ax.text(
                min(ci_lowers) - 0.08, i, badge_txt,
                va="center", ha="right", fontsize=10,
                color=badge_col, fontweight="bold",
            )

    ax.axvline(ref_line, color="black", linestyle="--", lw=1.0, alpha=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(dimensions, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontweight="bold", pad=12)
    ax.invert_yaxis()
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# Horizontal bar chart
# ─────────────────────────────────────────────────────────────────────────────

def horizontal_bar_chart(
    ax: plt.Axes,
    labels: Sequence[str],
    means: Sequence[float],
    sds: Sequence[float],
    sig_flags: Sequence[bool],
    title: str,
    xlabel: str,
    ref_line: Optional[float] = None,
    ref_label: Optional[str] = None,
    primary_color: Optional[str] = None,
    neutral_color: Optional[str] = None,
) -> plt.Axes:
    apply_rcparams()
    if primary_color is None:
        primary_color = PALETTE["primary"]
    if neutral_color is None:
        neutral_color = PALETTE["neutral"]
    means_arr = np.array(means, dtype=float)
    order = np.argsort(means_arr)[::-1]
    s_labels = [labels[i] for i in order]
    s_means  = [means_arr[i] for i in order]
    s_sds    = [sds[i] for i in order]
    s_sig    = [sig_flags[i] for i in order]
    colors   = [primary_color if s else neutral_color for s in s_sig]
    y_pos    = np.arange(len(s_labels))
    ax.barh(y_pos, s_means, xerr=s_sds, color=colors, alpha=0.82, capsize=4,
            error_kw={"elinewidth": 1.2, "capthick": 1.2}, height=0.65)
    for i, (m, sd) in enumerate(zip(s_means, s_sds)):
        ax.text(m + sd + 0.05, i, f"{m:.2f}", va="center", fontsize=9,
                color="#374151")
    if ref_line is not None:
        ax.axvline(ref_line, color=PALETTE["ref_line"], linestyle="--",
                   linewidth=1.0, alpha=0.8,
                   label=ref_label or f"x = {ref_line}")
        ax.legend(fontsize=9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(s_labels, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontweight="bold", pad=12)
    ax.invert_yaxis()
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# Publication heatmap
# ─────────────────────────────────────────────────────────────────────────────

def publication_heatmap(
    ax: plt.Axes,
    data_matrix: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str,
    cmap: str = "RdYlGn",
    vmin: float = 0.0,
    vmax: float = 1.0,
    colorbar_label: str = "Normalized Score (0–1)",
    star_matrix: Optional[np.ndarray] = None,
) -> plt.Axes:
    """Heatmap with optional per-cell significance stars."""
    apply_rcparams()
    im = ax.imshow(data_matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(colorbar_label, fontsize=10)
    n_rows, n_cols = data_matrix.shape
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(
        [c.replace("_", "\n") for c in col_labels], fontsize=9
    )
    ax.set_yticklabels(row_labels, fontsize=9)
    for i in range(n_rows):
        for j in range(n_cols):
            val = data_matrix[i, j]
            if np.isnan(val):
                continue
            tc = "white" if val < 0.30 or val > 0.80 else "black"
            cell_txt = f"{val:.3f}"
            if star_matrix is not None:
                cell_txt += f"\n{p_to_stars(star_matrix[i, j])}"
            ax.text(j, i, cell_txt, ha="center", va="center",
                    fontsize=7, color=tc, fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=12)
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# TOST equivalence panel (standalone figure with two-sided annotation)
# ─────────────────────────────────────────────────────────────────────────────

def tost_panel(
    ax: plt.Axes,
    dimensions: Sequence[str],
    tost_results: Sequence[Dict[str, Any]],
    title: str = "Equivalence Test Results (TOST)",
) -> plt.Axes:
    """
    Dot-and-bar chart showing the observed mean difference and equivalence
    bounds for each dimension. Green fill = equivalence demonstrated.
    """
    apply_rcparams()
    n     = len(dimensions)
    y_pos = np.arange(n, dtype=float)

    for i, (dim, res) in enumerate(zip(dimensions, tost_results)):
        dlt  = res.get("delta", 0.5)
        diff = res.get("observed_diff", 0.0)
        se   = res.get("pooled_se", 0.05)
        eq   = res.get("equivalent", False)
        pt   = res.get("p_tost", 1.0)

        # Equivalence band
        ax.barh(i, 2 * dlt, left=-dlt, height=0.35,
                color="#bbf7d0" if eq else "#fee2e2", alpha=0.60, zorder=1)

        # 95% CI of difference
        ci_lo = diff - 1.96 * se
        ci_hi = diff + 1.96 * se
        ax.plot([ci_lo, ci_hi], [i, i], color="#1e293b", lw=2.0,
                solid_capstyle="round", zorder=3)
        ax.plot(diff, i, "s", color="#15803d" if eq else "#dc2626",
                ms=8, zorder=4, markeredgecolor="white", markeredgewidth=1.0)

        stars = p_to_stars(pt)
        ax.text(
            max(r.get("delta", 0.5) for r in tost_results) * 1.15, i,
            f"p_TOST={pt:.3f} {stars}",
            va="center", ha="left", fontsize=8,
            color="#15803d" if eq else "#dc2626",
        )

    ax.axvline(0, color="black", lw=0.8, alpha=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(dimensions, fontsize=10)
    ax.set_xlabel("Mean difference (PHOENIX − HCP)")
    ax.set_title(title, fontweight="bold", pad=12)
    ax.invert_yaxis()
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compat stubs
# ─────────────────────────────────────────────────────────────────────────────

def violin_gradient_scatter(ax, data_dict, title, ylabel, p_value=None,
                            adj_p=None, cmap="viridis"):
    return raincloud_plot(ax, data_dict, title, ylabel,
                          adj_p=adj_p if adj_p is not None else p_value)


def boxwhisker_with_points(ax, data_dict, title, ylabel, cmap="viridis"):
    labels   = list(data_dict.keys())
    all_data = [np.array(data_dict[l], dtype=float) for l in labels]
    pos      = list(range(1, len(labels) + 1))
    bp = ax.boxplot(all_data, positions=pos, patch_artist=True, widths=0.4,
                    notch=False)
    for idx, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(DIM_COLORS[idx % len(DIM_COLORS)])
        patch.set_alpha(0.5)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2)
    rng = np.random.default_rng(42)
    for idx, (vals, p) in enumerate(zip(all_data, pos)):
        jit = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.full(len(vals), p) + jit, vals,
                   color=DIM_COLORS[idx % len(DIM_COLORS)], alpha=0.5, s=12, zorder=3)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", pad=12)
    return ax


def correlation_heatmap(ax, df, dimensions, title="Correlation Heatmap"):
    corr  = df[dimensions].dropna().corr(method="pearson")
    cmap  = plt.get_cmap("RdBu_r")
    im    = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8)
    n = len(dimensions)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short = [d[:15] for d in dimensions]
    ax.set_xticklabels(short, rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(short, fontsize=8)
    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(val) > 0.6 else "black")
    ax.set_title(title, fontweight="bold", pad=12)
    return ax


def random_effects_histogram(ax, blup_values, title="Random Effects (BLUPs)"):
    if not blup_values:
        ax.text(0.5, 0.5, "No BLUPs available", transform=ax.transAxes,
                ha="center", va="center")
        ax.set_title(title, fontweight="bold")
        return ax
    ax.hist(blup_values, bins=min(len(blup_values), 15),
            color=PALETTE["primary"], edgecolor="white", alpha=0.75)
    ax.axvline(0, color="#ef4444", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("BLUP Estimate")
    ax.set_ylabel("Count")
    ax.set_title(title, fontweight="bold", pad=12)
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# Report-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def interpret_result(dimension, test_result, group_ref, group_test,
                     adj_p, is_lme=True):
    if is_lme:
        coef = test_result.get("coefficient", 0)
        direction = "outperforms" if coef > 0 else "underperforms relative to"
        actor = group_test if coef > 0 else group_ref
        other = group_ref if coef > 0 else group_test
        return (f"{actor} {direction} {other} on {dimension} "
                f"(p_adj={adj_p:.4f}, coef={coef:.3f})")
    r = test_result.get("effect_size_r", 0)
    return (f"Non-parametric: p_adj={adj_p:.4f}, r={r:.3f}")


def build_dim_result(dimension, normality, lme_or_mw, adj_p, is_lme=True,
                     cohen_d_val=None, cohen_d_ci=None,
                     group_ref="expert", group_test="LLM"):
    res = {
        "dimension":       dimension,
        "normality":       {"shapiro_stat": normality[0], "shapiro_p": normality[1],
                            "is_normal": normality[2]},
        "test_used":       lme_or_mw.get("test", "unknown"),
        "p_value":         lme_or_mw.get("p_value"),
        "p_value_adjusted": adj_p,
        "interpretation":  interpret_result(
            dimension, lme_or_mw, group_ref, group_test, adj_p, is_lme=is_lme),
    }
    if is_lme:
        res.update({"coefficient": lme_or_mw.get("coefficient"),
                    "se": lme_or_mw.get("se"),
                    "t_value": lme_or_mw.get("t_value"),
                    "ci_95": [lme_or_mw.get("ci_lower"), lme_or_mw.get("ci_upper")]})
        if cohen_d_val is not None:
            res["effect_size"] = {"type": "Cohen's d", "value": cohen_d_val,
                                  "ci_95": cohen_d_ci}
    else:
        res["effect_size"] = {"type": "r", "value": lme_or_mw.get("effect_size_r")}
    return res


def save_json_report(report: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Saved JSON report: {path}")


def save_markdown_report(report: dict, path: Path) -> None:
    lines = []
    meta = report.get("metadata", {})
    lines.append(f"# {meta.get('study_name', 'Study Report')}\n")
    lines.append(f"**Study:** {meta.get('study_id', '')}  ")
    lines.append(f"**N:** {meta.get('n', '')}  ")
    lines.append(f"**Design:** {meta.get('design', '')}  \n")
    lines.append("## Statistical Results\n")
    for dr in report.get("results", []):
        dim = dr.get("dimension", "")
        lines.append(f"### {dim}\n")
        p = dr.get("p_value")
        if p:
            lines.append(f"- **p-value:** {p:.4f}  ")
        ap = dr.get("p_value_adjusted")
        if ap:
            lines.append(f"- **Adjusted p (Bonferroni):** {ap:.4f}  ")
        if "coefficient" in dr:
            lines.append(f"- **Coefficient:** {dr['coefficient']:.3f}  ")
            ci = dr.get("ci_95", [None, None])
            if ci[0] is not None:
                lines.append(f"- **95% CI:** [{ci[0]:.3f}, {ci[1]:.3f}]  ")
        if "effect_size" in dr:
            es = dr["effect_size"]
            v  = es.get("value")
            lines.append(f"- **Effect size ({es.get('type', '')}):** "
                         f"{v:.3f if v is not None else 'N/A'}  ")
        lines.append(f"\n**Interpretation:** {dr.get('interpretation', '')}  \n")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved Markdown report: {path}")
