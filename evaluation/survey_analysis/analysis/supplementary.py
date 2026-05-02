"""
Supplementary analyses for the LLM-as-judge evaluation pipeline.

Four analyses are implemented:

  Supp A – Judge-run stability (ICC)
      For each (case, part, dimension, entity) cell, compute variance across
      the 3 judge runs. Report ICC(2,1) — two-way random effects, single
      measures — per dimension per part as a reliability index.

  Supp B – Score calibration diagnostics (ceiling / floor effects)
      For each part × entity cell: ceiling rate (% score = +10), floor rate
      (% score = −10), and mean ± SD on the bipolar −10..+10 scale.
      Answers: does PHOENIX or HCP show ceiling/floor compression?

  Supp C – Confidence-weighted sensitivity analysis
      Rerun PHOENIX–HCP effect estimates weighted by judge confidence score.
      Compare weighted vs unweighted effects to check whether high-confidence
      ratings tell a materially different story (forest plot).

  Supp D – Per-case heterogeneity
      For each case × part, compute the mean PHOENIX–HCP quality gap.
      Heatmap shows which cases drive the aggregate effects.

All figures are saved to
    evaluation/survey_analysis/results/supplementary/visuals/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import scipy.stats as stats

from .shared.shared_stats import (
    PALETTE,
    COLOR_HCP,
    COLOR_PHOENIX,
    apply_rcparams,
    p_to_stars,
    save_figure,
)
from .shared.survey_paths import ensure_study_dirs, judgments_csv


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

PART_LABELS: Dict[str, str] = {
    "part1": "Part 1",
    "part2": "Part 2",
    "part3": "Part 3",
    "part4": "Part 4",
    "part5": "Part 5",
}

def _display_part(p: str) -> str:
    return PART_LABELS.get(str(p), str(p))


def _display_dim(d: str) -> str:
    return str(d).replace("_", " ").title()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Judgments CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"case_id", "part", "dimension", "judge_run", "entity", "quality_score"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Judgments CSV missing columns: {sorted(missing)}")
    df = df.copy()
    df["quality_score"] = pd.to_numeric(df["quality_score"], errors="coerce")
    if "confidence" not in df.columns:
        df["confidence"] = 3.0
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(3.0)
    df["judge_run"] = pd.to_numeric(df["judge_run"], errors="coerce")
    df = df.loc[df["entity"].isin(["phoenix", "hcp"])].dropna(subset=["quality_score"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Supp A: ICC(2,1) — two-way random effects, single measures
# ─────────────────────────────────────────────────────────────────────────────


def _icc_21_grouped(cell_scores: pd.Series, cell_ids: pd.Series) -> float:
    """
    Compute ICC(2,1) from grouped scores.

    Parameters
    ----------
    cell_scores : values (quality scores)
    cell_ids    : group identifier (one per cell, repeated k times)

    ICC(2,1) = (MS_B - MS_W) / (MS_B + (k-1)*MS_W)
    """
    groups = {}
    for cid, score in zip(cell_ids, cell_scores):
        groups.setdefault(cid, []).append(float(score))
    groups = {k: np.array(v) for k, v in groups.items() if len(v) >= 2}
    if len(groups) < 2:
        return float("nan")

    k_vals = [len(v) for v in groups.values()]
    k = float(np.mean(k_vals))  # harmonic mean approximation
    if k < 1:
        return float("nan")

    all_vals = np.concatenate(list(groups.values()))
    grand_mean = float(np.mean(all_vals))
    n_groups = len(groups)

    # Between-groups SS
    ss_b = sum(len(v) * (np.mean(v) - grand_mean) ** 2 for v in groups.values())
    ms_b = ss_b / max(n_groups - 1, 1)

    # Within-groups SS
    ss_w = sum(np.sum((v - np.mean(v)) ** 2) for v in groups.values())
    n_total = sum(len(v) for v in groups.values())
    df_w = n_total - n_groups
    ms_w = ss_w / max(df_w, 1) if df_w > 0 else 0.0

    if ms_b + (k - 1) * ms_w < 1e-12:
        return 1.0 if ms_w < 1e-12 else float("nan")
    return float((ms_b - ms_w) / (ms_b + (k - 1) * ms_w))


def compute_icc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ICC(2,1) per (part, dimension).

    For each (part, dimension) stratum:
      - subjects = (case_id, entity) cells
      - raters   = judge_run
      - ratings  = quality_score
    """
    rows: List[Dict[str, Any]] = []
    for (part, dim), sub in df.groupby(["part", "dimension"], observed=True):
        # cell_id = (case_id, entity) — each cell is a subject
        sub = sub.copy()
        sub["cell_id"] = sub["case_id"].astype(str) + "_" + sub["entity"].astype(str)
        icc = _icc_21_grouped(sub["quality_score"], sub["cell_id"])
        rows.append({
            "part": part,
            "dimension": dim,
            "icc": float(icc) if np.isfinite(icc) else float("nan"),
            "n_cells": int(sub["cell_id"].nunique()),
            "n_obs": int(len(sub)),
        })
    return pd.DataFrame(rows)


def _plot_icc(
    icc_df: pd.DataFrame,
    visuals_dir: Path,
) -> Path:
    apply_rcparams()
    parts = [p for p in ("part1", "part2", "part3", "part4", "part5")
             if p in set(icc_df["part"])]
    dims = list(dict.fromkeys(icc_df["dimension"].tolist()))

    # Build heatmap matrix
    matrix = np.full((len(parts), len(dims)), np.nan)
    for i, part in enumerate(parts):
        for j, dim in enumerate(dims):
            v = icc_df.loc[(icc_df["part"] == part) & (icc_df["dimension"] == dim), "icc"]
            if not v.empty and np.isfinite(float(v.iloc[0])):
                matrix[i, j] = float(v.iloc[0])

    fig, axes = plt.subplots(1, 2, figsize=(16, max(4, 0.6 * len(parts) + 2)),
                             gridspec_kw={"width_ratios": [3, 1]})

    # Left panel: heatmap
    ax = axes[0]
    finite_vals = matrix[np.isfinite(matrix)]
    vmin = float(np.nanmin(finite_vals)) if finite_vals.size else 0.0
    vmax = 1.0
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(dims)))
    ax.set_xticklabels([_display_dim(d) for d in dims],
                       rotation=38, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(parts)))
    ax.set_yticklabels([_display_part(p) for p in parts], fontsize=9)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if not np.isfinite(v):
                continue
            tc = "white" if v < 0.4 or v > 0.85 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color=tc)
    plt.colorbar(im, ax=ax, label="ICC(2,1)")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Part")

    # Right panel: violin / strip of ICC values per part
    ax2 = axes[1]
    icc_by_part = []
    part_tick_labels = []
    for part in parts:
        vals = icc_df.loc[icc_df["part"] == part, "icc"].dropna().to_numpy()
        vals = vals[np.isfinite(vals)]
        if len(vals):
            icc_by_part.append(vals)
            part_tick_labels.append(_display_part(part))

    if icc_by_part:
        vp = ax2.violinplot(
            icc_by_part,
            positions=np.arange(len(icc_by_part)),
            showmedians=True,
            widths=0.6,
        )
        for body in vp["bodies"]:
            body.set_facecolor(PALETTE["primary"])
            body.set_alpha(0.55)
        for key in ("cmedians", "cbars", "cmins", "cmaxes"):
            if key in vp:
                vp[key].set_color(PALETTE["ref_line"])
        # Overlay individual points
        rng = np.random.default_rng(42)
        for i, vals in enumerate(icc_by_part):
            jit = rng.uniform(-0.15, 0.15, size=len(vals))
            ax2.scatter(i + jit, vals, color=PALETTE["primary"],
                        s=18, alpha=0.7, zorder=3, edgecolors="white", linewidths=0.3)
    ax2.set_xticks(np.arange(len(part_tick_labels)))
    ax2.set_xticklabels(part_tick_labels, rotation=35, ha="right", fontsize=8)
    ax2.set_ylabel("ICC(2,1)")
    ax2.set_ylim(-0.1, 1.05)
    ax2.axhline(0.75, color=PALETTE["ref_line"], linestyle="--", linewidth=0.9, alpha=0.6)
    ax2.axhline(0.5, color=PALETTE["danger"], linestyle=":", linewidth=0.9, alpha=0.5)
    ax2.text(len(part_tick_labels) - 0.4, 0.76, "0.75", fontsize=7,
             color=PALETTE["ref_line"], va="bottom")

    plt.tight_layout()
    out = visuals_dir / "suppA_icc_stability.png"
    save_figure(fig, out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Supp B: Score calibration diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def compute_calibration(df: pd.DataFrame) -> pd.DataFrame:
    """Ceiling (% = +10), floor (% = −10), mean, SD by part × entity.

    On the bipolar −10..+10 scale, ceiling = score of +10 (outstanding),
    floor = score of −10 (catastrophic failure).
    """
    rows: List[Dict[str, Any]] = []
    for (part, entity), sub in df.groupby(["part", "entity"], observed=True):
        scores = sub["quality_score"].dropna().to_numpy(dtype=float)
        n = len(scores)
        if n == 0:
            continue
        rows.append({
            "part": part,
            "entity": entity,
            "n": n,
            "mean": float(np.mean(scores)),
            "sd": float(np.std(scores, ddof=1)) if n > 1 else 0.0,
            "ceiling_pct": float(np.sum(scores == 10) / n * 100),
            "floor_pct": float(np.sum(scores == -10) / n * 100),
        })
    return pd.DataFrame(rows)


def compute_score_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score distribution data: for each part × entity, report % of ratings
    at each individual integer score value on the bipolar −10..+10 scale.
    Returns a long-format DataFrame with columns:
        part, entity, score_value (int), pct (float).
    """
    SCORE_RANGE = list(range(-10, 11))   # −10, −9, …, +10 (21 values)
    rows: List[Dict[str, Any]] = []
    for (part, entity), sub in df.groupby(["part", "entity"], observed=True):
        scores = sub["quality_score"].dropna().to_numpy(dtype=float)
        n = len(scores)
        if n == 0:
            continue
        for sv in SCORE_RANGE:
            rows.append({
                "part": part,
                "entity": entity,
                "score_value": int(sv),
                "pct": float(np.sum(scores == sv) / n * 100),
            })
    return pd.DataFrame(rows)


def _plot_calibration(
    calib_df: pd.DataFrame,
    dist_df: pd.DataFrame,
    visuals_dir: Path,
) -> Path:
    """
    Two-panel calibration figure.

    Top panel — grouped bar chart: ceiling rate (% = +10) and floor rate
    (% = −10) side-by-side per part, separately for PHOENIX and HCP.

    Bottom panel — probability mass function (PMF) line plots: one subplot
    per survey part; PHOENIX vs HCP overlaid as filled-area + line graphs
    across all 21 individual integer score values (−10 to +10).
    """
    apply_rcparams()
    SCORE_RANGE = list(range(-10, 11))   # 21 integer values
    parts = [p for p in ("part1", "part2", "part3", "part4", "part5")
             if p in set(calib_df["part"])]
    n_parts = len(parts)

    # ── Layout: top row (bar chart) + bottom row (PMF grid, 1×n_parts) ──────
    fig = plt.figure(figsize=(max(12, 2.8 * n_parts), 10))
    gs = fig.add_gridspec(
        2, n_parts,
        height_ratios=[1, 2],
        hspace=0.45,
        wspace=0.35,
    )
    ax_top = fig.add_subplot(gs[0, :])   # spans full top row
    pmf_axes = [fig.add_subplot(gs[1, j]) for j in range(n_parts)]

    # ── Top panel: ceiling + floor rates ───────────��────────────────────────
    x = np.arange(len(parts))
    width = 0.22
    offsets = {"ceiling_pct": -0.33, "floor_pct": 0.33}
    hatches = {"ceiling_pct": "", "floor_pct": "///"}
    rate_labels = {"ceiling_pct": "Ceiling (+10)", "floor_pct": "Floor (−10)"}

    for idx, (entity, color, ent_label) in enumerate([
        ("phoenix", COLOR_PHOENIX, "PHOENIX"),
        ("hcp",     COLOR_HCP,     "HCP"),
    ]):
        sub = calib_df.loc[calib_df["entity"] == entity]
        offset_sign = -1 if entity == "phoenix" else +1
        for rate_col, base_offset in offsets.items():
            vals = [
                float(sub.loc[sub["part"] == p, rate_col].mean())
                if p in sub["part"].values else 0.0
                for p in parts
            ]
            ax_top.bar(
                x + offset_sign * width / 2 + base_offset * width,
                vals,
                width,
                label=f"{ent_label} {rate_labels[rate_col]}" if idx == 0 else f"HCP {rate_labels[rate_col]}",
                color=color,
                alpha=0.75 if rate_col == "ceiling_pct" else 0.45,
                hatch=hatches[rate_col],
                edgecolor="white",
            )

    ax_top.set_ylabel("Rate (%)")
    ax_top.set_xticks(x)
    ax_top.set_xticklabels([_display_part(p) for p in parts], fontsize=9)
    ax_top.set_ylim(0, 110)
    ax_top.axhline(20, color=PALETTE["ref_line"], linestyle="--",
                   linewidth=0.8, alpha=0.5)
    ax_top.text(len(parts) - 0.45, 21.5, "20%", fontsize=7.5,
                color=PALETTE["ref_line"], va="bottom")
    ax_top.legend(
        loc="upper right", fontsize=7.5, framealpha=0.7,
        ncol=2,
    )

    # ── Bottom panel: PMF per part ────────────────────────────���──────────────
    if dist_df.empty:
        for ax in pmf_axes:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")
    else:
        sv_arr = np.array(SCORE_RANGE, dtype=float)
        for ax, part in zip(pmf_axes, parts):
            for entity, color, alpha_fill in [
                ("phoenix", COLOR_PHOENIX, 0.20),
                ("hcp",     COLOR_HCP,     0.15),
            ]:
                sub = dist_df.loc[
                    (dist_df["part"] == part) & (dist_df["entity"] == entity)
                ].sort_values("score_value")
                if sub.empty:
                    continue
                sv_vals = sub["score_value"].to_numpy(dtype=float)
                pct_vals = sub["pct"].to_numpy(dtype=float)
                # Align to full -10..+10 grid (fill missing with 0)
                pct_grid = np.zeros(len(SCORE_RANGE), dtype=float)
                for sv, pv in zip(sv_vals, pct_vals):
                    idx_sv = int(sv) + 10
                    if 0 <= idx_sv < len(SCORE_RANGE):
                        pct_grid[idx_sv] = pv

                ent_label = "PHOENIX" if entity == "phoenix" else "HCP"
                ax.plot(sv_arr, pct_grid, color=color, linewidth=1.6,
                        label=ent_label, alpha=0.9, zorder=3)
                ax.fill_between(sv_arr, pct_grid, alpha=alpha_fill, color=color, zorder=2)

            ax.axvline(0, color=PALETTE["ref_line"], linestyle="--",
                       linewidth=0.8, alpha=0.5)
            ax.set_xlim(-11, 11)
            ax.set_xticks([-10, -5, 0, 5, 10])
            ax.set_xticklabels(["-10", "-5", "0", "+5", "+10"], fontsize=7.5)
            ax.set_xlabel("Score", fontsize=8)
            ax.set_ylabel("% of ratings", fontsize=8)
            ax.tick_params(axis="y", labelsize=7.5)
            ax.text(
                0.04, 0.97, _display_part(part),
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=8.5, fontweight="bold",
            )
            if part == parts[0]:
                ax.legend(loc="upper left", fontsize=7.5, framealpha=0.7)

    out = visuals_dir / "suppB_calibration_diagnostics.png"
    save_figure(fig, out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Supp C: Confidence-weighted sensitivity (forest plot)
# ─────────────────────────────────────────────────────────────────────────────

def compute_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each part: compute unweighted and confidence-weighted PHOENIX–HCP gaps
    using a weighted least squares approach (weight = confidence score).
    """
    rows: List[Dict[str, Any]] = []
    parts = [p for p in ("part1", "part2", "part3", "part4", "part5")
             if p in set(df["part"])]

    for part in parts:
        sub = df.loc[df["part"] == part].copy()

        # Unweighted mean gap
        ph = sub.loc[sub["entity"] == "phoenix", "quality_score"].dropna()
        hc = sub.loc[sub["entity"] == "hcp", "quality_score"].dropna()
        unweighted_gap = float(ph.mean() - hc.mean()) if (len(ph) and len(hc)) else 0.0

        # SE of unweighted gap
        if len(ph) > 1 and len(hc) > 1:
            se_uw = float(np.sqrt(
                ph.var(ddof=1) / len(ph) + hc.var(ddof=1) / len(hc)
            ))
        else:
            se_uw = 0.0

        # Confidence-weighted: WLS with weight = confidence
        ph_sub = sub.loc[sub["entity"] == "phoenix"].dropna(subset=["quality_score", "confidence"])
        hc_sub = sub.loc[sub["entity"] == "hcp"].dropna(subset=["quality_score", "confidence"])

        if len(ph_sub) and len(hc_sub):
            w_ph = ph_sub["confidence"].to_numpy(dtype=float)
            w_hc = hc_sub["confidence"].to_numpy(dtype=float)
            s_ph = ph_sub["quality_score"].to_numpy(dtype=float)
            s_hc = hc_sub["quality_score"].to_numpy(dtype=float)
            wm_ph = float(np.average(s_ph, weights=w_ph))
            wm_hc = float(np.average(s_hc, weights=w_hc))
            weighted_gap = wm_ph - wm_hc
            # SE via weighted variance
            def _wse(s: np.ndarray, w: np.ndarray) -> float:
                w = w / w.sum()
                wm = float(np.dot(w, s))
                n_eff = 1.0 / np.sum(w ** 2)
                wvar = float(np.dot(w, (s - wm) ** 2)) * n_eff / max(n_eff - 1, 1)
                return float(np.sqrt(wvar / max(len(s), 1)))
            se_w = float(np.sqrt(_wse(s_ph, w_ph) ** 2 + _wse(s_hc, w_hc) ** 2))
        else:
            weighted_gap = unweighted_gap
            se_w = se_uw

        rows.append({
            "part": part,
            "part_label": _display_part(part),
            "unweighted_gap": unweighted_gap,
            "se_unweighted": se_uw,
            "weighted_gap": weighted_gap,
            "se_weighted": se_w,
            "absolute_change": abs(weighted_gap - unweighted_gap),
        })
    return pd.DataFrame(rows)


def _plot_sensitivity(
    sens_df: pd.DataFrame,
    visuals_dir: Path,
) -> Path:
    apply_rcparams()
    if sens_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No sensitivity data", ha="center", va="center",
                transform=ax.transAxes)
        out = visuals_dir / "suppC_sensitivity_forest.png"
        save_figure(fig, out)
        return out

    parts = sens_df["part_label"].tolist()
    n = len(parts)
    y_pos = np.arange(n, dtype=float)

    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.8 * n + 1.5)))

    offset = 0.22
    for i, row in sens_df.iterrows():
        yi = float(y_pos[list(sens_df.index).index(i)])

        # Unweighted (grey)
        uw = float(row["unweighted_gap"])
        se_uw = float(row["se_unweighted"])
        ci_lo_uw = uw - 1.96 * se_uw
        ci_hi_uw = uw + 1.96 * se_uw
        ax.plot([ci_lo_uw, ci_hi_uw], [yi + offset, yi + offset],
                color=PALETTE["neutral"], lw=2.2, alpha=0.85, solid_capstyle="round")
        ax.plot(uw, yi + offset, "o", ms=8, color=PALETTE["neutral"],
                markeredgecolor="white", markeredgewidth=1.2, zorder=5)

        # Weighted (colored)
        wg = float(row["weighted_gap"])
        se_wg = float(row["se_weighted"])
        ci_lo_wg = wg - 1.96 * se_wg
        ci_hi_wg = wg + 1.96 * se_wg
        color = COLOR_PHOENIX if wg >= 0 else COLOR_HCP
        ax.plot([ci_lo_wg, ci_hi_wg], [yi - offset, yi - offset],
                color=color, lw=2.2, alpha=0.85, solid_capstyle="round")
        ax.plot(wg, yi - offset, "D", ms=8, color=color,
                markeredgecolor="white", markeredgewidth=1.2, zorder=5)

        # Annotate absolute change
        ax.text(
            max(ci_hi_uw, ci_hi_wg) + 0.08, yi,
            f"Δ={row['absolute_change']:.3f}",
            va="center", ha="left", fontsize=7.5, color="#374151",
        )

    ax.axvline(0, color="black", linestyle="--", lw=1.0, alpha=0.55)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(parts, fontsize=10)
    ax.set_xlabel("PHOENIX − HCP quality gap (scale −10 to +10)")
    ax.invert_yaxis()

    legend_elems = [
        mpatches.Patch(color=PALETTE["neutral"], alpha=0.85, label="Unweighted"),
        mpatches.Patch(color=COLOR_PHOENIX, alpha=0.85, label="Confidence-weighted"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=9, framealpha=0.7)

    plt.tight_layout()
    out = visuals_dir / "suppC_sensitivity_forest.png"
    save_figure(fig, out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Supp D: Per-case heterogeneity
# ─────────────────────────────────────────────────────────────────────────────

def compute_case_heterogeneity(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (case_id, part): mean PHOENIX − HCP quality gap
    (averaged over dimensions and judge runs).
    """
    rows: List[Dict[str, Any]] = []
    for (case_id, part), sub in df.groupby(["case_id", "part"], observed=True):
        ph_mean = float(sub.loc[sub["entity"] == "phoenix", "quality_score"].mean())
        hc_mean = float(sub.loc[sub["entity"] == "hcp", "quality_score"].mean())
        if np.isfinite(ph_mean) and np.isfinite(hc_mean):
            rows.append({
                "case_id": case_id,
                "part": part,
                "gap": ph_mean - hc_mean,
                "n_phoenix": int((sub["entity"] == "phoenix").sum()),
                "n_hcp": int((sub["entity"] == "hcp").sum()),
            })
    return pd.DataFrame(rows)


def _plot_case_heterogeneity(
    het_df: pd.DataFrame,
    visuals_dir: Path,
) -> Path:
    apply_rcparams()
    parts = [p for p in ("part1", "part2", "part3", "part4", "part5")
             if p in set(het_df["part"])]
    cases = sorted(het_df["case_id"].unique().tolist())

    matrix = np.full((len(cases), len(parts)), np.nan)
    for i, case in enumerate(cases):
        for j, part in enumerate(parts):
            val = het_df.loc[
                (het_df["case_id"] == case) & (het_df["part"] == part), "gap"
            ]
            if not val.empty:
                matrix[i, j] = float(val.iloc[0])

    finite_vals = matrix[np.isfinite(matrix)]
    vabs = max(0.3, float(np.max(np.abs(finite_vals)))) if finite_vals.size else 0.5

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(parts)), max(5, 0.55 * len(cases) + 1.5)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vabs, vmax=vabs)

    ax.set_xticks(np.arange(len(parts)))
    ax.set_xticklabels([_display_part(p) for p in parts], fontsize=9)
    ax.set_yticks(np.arange(len(cases)))
    ax.set_yticklabels(cases, fontsize=9)
    ax.set_xlabel("Part")
    ax.set_ylabel("Case")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if not np.isfinite(v):
                continue
            tc = "white" if abs(v) > 0.55 * vabs else "black"
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    fontsize=8, color=tc, fontweight="bold")

    plt.colorbar(im, ax=ax, label="PHOENIX − HCP quality gap (−10 to +10 scale)")
    plt.tight_layout()
    out = visuals_dir / "suppD_case_heterogeneity.png"
    save_figure(fig, out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Combined multi-panel figures
# ─────────────────────────────────────────────────────────────────────────────

def _plot_overview_dashboard(
    df: pd.DataFrame,
    icc_df: pd.DataFrame,
    calib_df: pd.DataFrame,
    sens_df: pd.DataFrame,
    het_df: pd.DataFrame,
    visuals_dir: Path,
) -> Path:
    """
    Create one compact 2x2 supplementary dashboard.

    The dashboard avoids zero-heavy diagnostics by showing: part-level
    reliability, scale-use bands, confidence-weighting robustness, and
    case-level heterogeneity. Titles are intentionally omitted; panel
    interpretation belongs in the markdown caption.
    """
    apply_rcparams()
    parts = [p for p in ("part1", "part2", "part3", "part4", "part5")
             if p in set(df["part"].astype(str))]
    part_labels = [_display_part(p) for p in parts]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # A. Judge-run reliability by part with dimension-level points.
    x = np.arange(len(parts), dtype=float)
    mean_iccs = [
        float(icc_df.loc[icc_df["part"] == p, "icc"].mean(skipna=True))
        for p in parts
    ]
    ax_a.bar(x, mean_iccs, color=PALETTE["primary"], alpha=0.30, width=0.62)
    rng = np.random.default_rng(42)
    for i, part in enumerate(parts):
        vals = icc_df.loc[icc_df["part"] == part, "icc"].dropna().to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals):
            ax_a.scatter(
                np.full(len(vals), i) + rng.uniform(-0.18, 0.18, size=len(vals)),
                vals,
                s=26,
                color=PALETTE["primary"],
                edgecolor="white",
                linewidth=0.5,
                alpha=0.78,
                zorder=3,
            )
    ax_a.axhline(0.75, color=PALETTE["ref_line"], linestyle="--", linewidth=1.0, alpha=0.65)
    ax_a.axhline(0.50, color=PALETTE["danger"], linestyle=":", linewidth=1.0, alpha=0.55)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(part_labels, rotation=25, ha="right")
    ax_a.set_ylabel("ICC(2,1)")
    ax_a.set_ylim(-0.05, 1.05)
    ax_a.text(-0.08, 1.03, "A", transform=ax_a.transAxes, fontsize=12, fontweight="bold")

    # B. Scale-use bands: negative, acceptable-zone, and high-quality ratings.
    categories = [
        ("Negative (≤ -5)", lambda s: s <= -5, COLOR_HCP),
        ("Acceptable zone (-4..+4)", lambda s: (s > -5) & (s < 5), PALETTE["neutral"]),
        ("High quality (≥ +5)", lambda s: s >= 5, COLOR_PHOENIX),
    ]
    width = 0.36
    for entity, offset, hatch in [("phoenix", -width / 2, ""), ("hcp", width / 2, "///")]:
        bottoms = np.zeros(len(parts), dtype=float)
        for label, mask_fn, color in categories:
            vals = []
            for part in parts:
                scores = df.loc[
                    (df["part"] == part) & (df["entity"] == entity),
                    "quality_score",
                ].to_numpy(dtype=float)
                vals.append(float(mask_fn(scores).mean() * 100) if len(scores) else 0.0)
            ax_b.bar(
                x + offset,
                vals,
                width,
                bottom=bottoms,
                color=color,
                alpha=0.72,
                hatch=hatch,
                edgecolor="white",
                linewidth=0.4,
                label=label if entity == "phoenix" else None,
            )
            bottoms += np.asarray(vals)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(part_labels, rotation=25, ha="right")
    ax_b.set_ylabel("Ratings (%)")
    ax_b.set_ylim(0, 100)
    ax_b.legend(loc="upper left", fontsize=8, framealpha=0.75)
    ax_b.text(0.98, 0.96, "solid = PHOENIX\nhatched = HCP",
              transform=ax_b.transAxes, ha="right", va="top", fontsize=8,
              color=PALETTE["ref_line"])
    ax_b.text(-0.08, 1.03, "B", transform=ax_b.transAxes, fontsize=12, fontweight="bold")

    # C. Confidence-weighting sensitivity: slope graph.
    if not sens_df.empty:
        y_min = min(sens_df["unweighted_gap"].min(), sens_df["weighted_gap"].min())
        y_max = max(sens_df["unweighted_gap"].max(), sens_df["weighted_gap"].max())
        label_rows = []
        for _, row in sens_df.iterrows():
            y0 = float(row["unweighted_gap"])
            y1 = float(row["weighted_gap"])
            color = COLOR_PHOENIX if y1 >= 0 else COLOR_HCP
            ax_c.plot([0, 1], [y0, y1], color=color, alpha=0.72, linewidth=2.0)
            ax_c.scatter([0, 1], [y0, y1], color=color, edgecolor="white", linewidth=0.8, s=46, zorder=3)
            label_rows.append((y1, str(row["part_label"]), color))
        pad = max(0.4, 0.12 * (y_max - y_min))
        ax_c.set_ylim(y_min - pad, y_max + pad)
        label_rows.sort(key=lambda row: row[0])
        label_min_sep = max(0.25, 0.06 * (y_max - y_min))
        adjusted = []
        for y1, label, color in label_rows:
            y_adj = y1
            if adjusted and y_adj - adjusted[-1][0] < label_min_sep:
                y_adj = adjusted[-1][0] + label_min_sep
            adjusted.append((y_adj, y1, label, color))
        ylim_lo, ylim_hi = ax_c.get_ylim()
        for y_adj, y1, label, color in adjusted:
            y_adj = min(max(y_adj, ylim_lo + 0.15), ylim_hi - 0.15)
            ax_c.plot([1.01, 1.04], [y1, y_adj], color=color, alpha=0.45, linewidth=0.8)
            ax_c.text(1.06, y_adj, label, va="center", fontsize=8, color=color)
    ax_c.axhline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.55)
    ax_c.set_xlim(-0.15, 1.55)
    ax_c.set_xticks([0, 1])
    ax_c.set_xticklabels(["Unweighted", "Confidence-weighted"])
    ax_c.set_ylabel("PHOENIX - HCP quality gap")
    ax_c.text(-0.08, 1.03, "C", transform=ax_c.transAxes, fontsize=12, fontweight="bold")

    # D. Case-by-part heterogeneity heatmap.
    cases = sorted(het_df["case_id"].unique().tolist()) if not het_df.empty else []
    matrix = np.full((len(cases), len(parts)), np.nan)
    for i, case in enumerate(cases):
        for j, part in enumerate(parts):
            val = het_df.loc[
                (het_df["case_id"] == case) & (het_df["part"] == part), "gap"
            ]
            if not val.empty:
                matrix[i, j] = float(val.iloc[0])
    finite = matrix[np.isfinite(matrix)]
    vabs = max(0.5, float(np.max(np.abs(finite)))) if finite.size else 1.0
    im = ax_d.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vabs, vmax=vabs)
    ax_d.set_xticks(np.arange(len(parts)))
    ax_d.set_xticklabels(part_labels, rotation=25, ha="right")
    ax_d.set_yticks(np.arange(len(cases)))
    ax_d.set_yticklabels(cases)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isfinite(val):
                ax_d.text(j, i, f"{val:+.1f}", ha="center", va="center",
                          fontsize=7.5, color="white" if abs(val) > 0.55 * vabs else "black")
    cbar = plt.colorbar(im, ax=ax_d, shrink=0.84, pad=0.02)
    cbar.set_label("Gap")
    ax_d.set_xlabel("Part")
    ax_d.set_ylabel("Case")
    ax_d.text(-0.08, 1.03, "D", transform=ax_d.transAxes, fontsize=12, fontweight="bold")

    plt.tight_layout()
    out = visuals_dir / "supplementary_overview_dashboard.png"
    save_figure(fig, out)
    return out

def _write_report(
    paths: Dict[str, Path],
    icc_df: pd.DataFrame,
    calib_df: pd.DataFrame,
    sens_df: pd.DataFrame,
    het_df: pd.DataFrame,
    n_rows: int,
) -> Path:
    lines: List[str] = [
        "Supplementary Analyses Report",
        "=" * 60,
        "",
        f"Rows analysed: {n_rows:,}",
        "",
        "Supp A: Judge-run stability (ICC)",
        "-" * 40,
    ]
    if not icc_df.empty:
        global_icc = float(icc_df["icc"].mean(skipna=True))
        lines.append(f"  Global mean ICC(2,1) across all part×dimension strata: {global_icc:.3f}")
        for part in ("part1", "part2", "part3", "part4", "part5"):
            sub = icc_df.loc[icc_df["part"] == part, "icc"].dropna()
            if not sub.empty:
                lines.append(f"  {_display_part(part)}: mean ICC = {sub.mean():.3f}, "
                             f"min = {sub.min():.3f}, max = {sub.max():.3f}")
        lines.append("")
        lines.append("  Benchmarks: ICC ≥ 0.75 = good, 0.50–0.75 = moderate, < 0.50 = poor.")

    lines += [
        "",
        "Supp B: Score calibration diagnostics",
        "-" * 40,
    ]
    if not calib_df.empty:
        for (part, entity), row in calib_df.groupby(["part", "entity"], observed=True):
            r = row.iloc[0]
            lines.append(
                f"  {_display_part(part)} [{entity.upper():6s}]: "
                f"mean={r['mean']:.3f}, SD={r['sd']:.3f}, "
                f"ceiling={r['ceiling_pct']:.1f}%, floor={r['floor_pct']:.1f}%"
            )

    lines += [
        "",
        "Supp C: Confidence-weighted sensitivity",
        "-" * 40,
    ]
    if not sens_df.empty:
        max_change = float(sens_df["absolute_change"].max())
        lines.append(
            f"  Max absolute change after confidence weighting: {max_change:.3f} quality points."
        )
        for _, row in sens_df.iterrows():
            lines.append(
                f"  {row['part_label']}: "
                f"unweighted={row['unweighted_gap']:+.3f}, "
                f"weighted={row['weighted_gap']:+.3f}, "
                f"Δ={row['absolute_change']:.3f}"
            )

    lines += [
        "",
        "Supp D: Per-case heterogeneity",
        "-" * 40,
    ]
    if not het_df.empty:
        overall_gap = float(het_df["gap"].mean())
        gap_sd = float(het_df["gap"].std(ddof=1))
        lines.append(f"  Grand mean PHOENIX−HCP gap: {overall_gap:+.3f} (SD={gap_sd:.3f})")
        driving = het_df.loc[het_df["gap"].abs() > 2.0, ["case_id", "part", "gap"]]
        if not driving.empty:
            lines.append(f"  Cells with |gap| > 2.0: {len(driving)}")
        for case_id in sorted(het_df["case_id"].unique()):
            case_mean = float(het_df.loc[het_df["case_id"] == case_id, "gap"].mean())
            lines.append(f"  {case_id}: mean gap = {case_mean:+.3f}")

    out = paths["report_dir"] / "supplementary_report.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_supplementary_analyses(csv_path: Path) -> Dict[str, Any]:
    """
    Run all four supplementary analyses and save figures.

    Parameters
    ----------
    csv_path : Path
        Path to the long-format judgments CSV.

    Returns
    -------
    dict with keys: icc, calibration, sensitivity, heterogeneity, figures.
    """
    apply_rcparams()
    paths = ensure_study_dirs("supplementary")
    visuals_dir = paths["visuals_dir"]

    df = _load(csv_path)

    # A: ICC
    icc_df = compute_icc(df)

    # B: Calibration
    calib_df = compute_calibration(df)
    dist_df = compute_score_distribution(df)

    # C: Sensitivity
    sens_df = compute_sensitivity(df)

    # D: Per-case heterogeneity
    het_df = compute_case_heterogeneity(df)

    # Save CSV outputs
    icc_df.to_csv(paths["report_dir"] / "suppA_icc.csv", index=False)
    calib_df.to_csv(paths["report_dir"] / "suppB_calibration.csv", index=False)
    sens_df.to_csv(paths["report_dir"] / "suppC_sensitivity.csv", index=False)
    het_df.to_csv(paths["report_dir"] / "suppD_heterogeneity.csv", index=False)

    # Generate figures
    fig_icc = _plot_icc(icc_df, visuals_dir)
    fig_calib = _plot_calibration(calib_df, dist_df, visuals_dir)
    fig_sens = _plot_sensitivity(sens_df, visuals_dir)
    fig_het = _plot_case_heterogeneity(het_df, visuals_dir)
    fig_dashboard = _plot_overview_dashboard(
        df, icc_df, calib_df, sens_df, het_df, visuals_dir
    )

    # Text report
    report_path = _write_report(
        paths, icc_df, calib_df, sens_df, het_df, n_rows=len(df)
    )

    return {
        "icc": icc_df,
        "calibration": calib_df,
        "sensitivity": sens_df,
        "heterogeneity": het_df,
        "figures": [fig_dashboard, fig_icc, fig_calib, fig_sens, fig_het],
        "report_path": report_path,
        "global_icc_mean": float(icc_df["icc"].mean(skipna=True)) if not icc_df.empty else float("nan"),
        "max_sensitivity_change": float(sens_df["absolute_change"].max()) if not sens_df.empty else 0.0,
        "grand_mean_gap": float(het_df["gap"].mean()) if not het_df.empty else 0.0,
    }


def run(config=None) -> Dict[str, Any]:
    """
    Entry point called by the pipeline.

    Accepts an optional legacy config argument (ignored) for backward
    compatibility with the old SupplementaryStudyConfig interface.
    """
    csv_path = judgments_csv()
    return run_supplementary_analyses(csv_path)


if __name__ == "__main__":
    run()
