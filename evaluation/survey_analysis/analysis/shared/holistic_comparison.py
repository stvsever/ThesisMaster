"""
Cross-part synthesis for absolute per-output quality scores (design).

Statistical model
-----------------
For each part and globally, fits:

    quality_score ~ entity_ec + (1 | case_id) + (1 | judge_run)

where entity_ec is effect-coded: PHOENIX = +0.5, HCP = −0.5.

The coefficient on entity_ec estimates the PHOENIX − HCP quality gap on
the bipolar −10..+10 scale; positive values indicate PHOENIX outperforms HCP.
Uncertainty is quantified with 95% CIs; multiplicity correction uses the
Holm–Bonferroni procedure across parts.

TOST equivalence testing on difference scores (phoenix − hcp per cell)
uses delta = ±1.5 quality points (7.5% of the −10..+10 scale range).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from .shared_stats import (
    PALETTE,
    COLOR_HCP,
    COLOR_PHOENIX,
    apply_rcparams,
    bootstrap_cohend_one_sample_ci,
    cohen_d_one_sample,
    effect_size_category,
    forest_plot,
    holm_correct,
    p_to_stars,
    publication_heatmap,
    raincloud_plot,
    save_figure,
    standardized_forest_plot,
    tost_panel,
    tost_test_one_sample,
)
from .survey_paths import ensure_study_dirs, judgments_csv


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HolisticStudyConfig:
    study_slug: str = "synthesis"
    title: str = "Cross-part PHOENIX vs HCP quality synthesis"
    report_name: str = "synthesis_report.txt"
    judgments_path: Optional[Path] = None
    quality_min: float = -10.0
    quality_max: float = 10.0
    tost_delta: float = 1.5       # ±1.5 pts on the −10..+10 scale (7.5% of range)
    part_order: Sequence[str] = (
        "part1", "part2", "part3", "part4", "part5",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _display_part(p: str) -> str:
    table = {
        "part1": "Part 1: Symptom Labels",
        "part2": "Part 2: Treatment Options",
        "part3": "Part 3: Target Ranking",
        "part4": "Part 4: EMA Items",
        "part5": "Part 5: Coaching Message",
    }
    return table.get(p, p)


def _display_dim(key: str) -> str:
    return key.replace("_", " ").title()


def _effect_code(df: pd.DataFrame) -> pd.DataFrame:
    """Add entity_ec: PHOENIX=+0.5, HCP=−0.5."""
    df = df.copy()
    df["entity_ec"] = df["entity"].map({"phoenix": 0.5, "hcp": -0.5}).fillna(0.0)
    return df


def _fit_entity_lmm(
    data: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Fit quality_score ~ entity_ec with case_id / judge_run random effects.

    Tries a sequence of mixed-model specs; falls back to Welch t-test.
    Returns a dict with coefficient, CI, p_value, method.
    """
    data = _effect_code(data)
    formula = "quality_score ~ entity_ec"
    effect_term = "entity_ec"

    candidates = [
        {
            "label": "LMM (case_id groups + judge_run VC)",
            "group_col": "case_id",
            "re_formula": None,
            "vc_formula": {"judge_run": "0 + C(judge_run)"},
        },
        {
            "label": "LMM (judge_run groups + case_id VC)",
            "group_col": "judge_run",
            "re_formula": None,
            "vc_formula": {"case_id": "0 + C(case_id)"},
        },
        {
            "label": "LMM (case_id groups, no VC)",
            "group_col": "case_id",
            "re_formula": None,
            "vc_formula": None,
        },
    ]

    errors: List[str] = []
    for cand in candidates:
        gc = cand["group_col"]
        if gc not in data.columns or data[gc].nunique() < 2:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model = smf.mixedlm(
                    formula,
                    data=data,
                    groups=data[gc],
                    vc_formula=cand.get("vc_formula"),
                ).fit(reml=True, method="lbfgs")
            coef = float(model.params.get(effect_term, np.nan))
            se   = float(model.bse.get(effect_term, np.nan))
            pval = float(model.pvalues.get(effect_term, 1.0))
            if np.isfinite(coef) and np.isfinite(se) and se > 1e-6:
                return {
                    "method":     cand["label"],
                    "coefficient": coef,
                    "se":         se,
                    "ci_lower":   coef - 1.96 * se,
                    "ci_upper":   coef + 1.96 * se,
                    "p_value":    pval,
                    "converged":  bool(model.converged),
                }
        except Exception as exc:
            errors.append(f"{cand['label']}: {exc}")

    # Welch t-test fallback
    ph = data.loc[data["entity"] == "phoenix", "quality_score"].astype(float).to_numpy()
    hc = data.loc[data["entity"] == "hcp",     "quality_score"].astype(float).to_numpy()
    ph = ph[np.isfinite(ph)]
    hc = hc[np.isfinite(hc)]
    if len(ph) >= 2 and len(hc) >= 2:
        coef = float(np.mean(ph) - np.mean(hc))
        se_w = float(np.sqrt(np.var(ph, ddof=1) / len(ph) + np.var(hc, ddof=1) / len(hc)))
        try:
            _, pval = stats.ttest_ind(ph, hc, equal_var=False)
        except Exception:
            pval = 1.0
        return {
            "method":     "Welch t-test (LMM fallback)",
            "coefficient": coef,
            "se":         se_w,
            "ci_lower":   coef - 1.96 * se_w,
            "ci_upper":   coef + 1.96 * se_w,
            "p_value":    float(pval),
            "converged":  True,
        }
    return {
        "method":     "Insufficient data",
        "coefficient": 0.0,
        "se":         0.0,
        "ci_lower":   0.0,
        "ci_upper":   0.0,
        "p_value":    1.0,
        "converged":  False,
        "error":      " | ".join(errors),
    }


def _paired_differences(data: pd.DataFrame) -> np.ndarray:
    """Return PHOENIX minus HCP differences paired by case, part, dimension, run."""
    index_cols = [
        c for c in ("case_id", "part", "dimension", "judge_run")
        if c in data.columns
    ]
    paired = (
        data.pivot_table(
            index=index_cols,
            columns="entity",
            values="quality_score",
            aggfunc="mean",
        )
        .dropna(subset=["phoenix", "hcp"], how="any")
    )
    if paired.empty:
        return np.asarray([], dtype=float)
    return (paired["phoenix"].astype(float) - paired["hcp"].astype(float)).to_numpy()


def _per_part_tost(
    data: pd.DataFrame,
    delta: float,
) -> Dict[str, Any]:
    """TOST on paired PHOENIX minus HCP difference scores."""
    diffs = _paired_differences(data)
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) < 2:
        return {
            "p_tost": 1.0, "p_upper": 1.0, "p_lower": 1.0,
            "delta": delta, "observed_diff": 0.0,
            "pooled_se": 0.0, "equivalent": False,
        }
    return tost_test_one_sample(diffs, delta=delta)


def _standardized_paired_effect(data: pd.DataFrame, seed: int = 42) -> Dict[str, Any]:
    """Return paired Cohen's dz and bootstrap CI for PHOENIX-HCP differences."""
    diffs = _paired_differences(data)
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) < 2:
        return {
            "cohen_dz": 0.0,
            "cohen_dz_ci_lower": 0.0,
            "cohen_dz_ci_upper": 0.0,
            "cohen_dz_category": "trivial",
            "n_pairs": int(len(diffs)),
        }
    d = cohen_d_one_sample(diffs)
    lo, hi = bootstrap_cohend_one_sample_ci(diffs, seed=seed)
    return {
        "cohen_dz": float(d),
        "cohen_dz_ci_lower": float(lo),
        "cohen_dz_ci_upper": float(hi),
        "cohen_dz_category": effect_size_category(d),
        "n_pairs": int(len(diffs)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_holistic_synthesis(config: HolisticStudyConfig) -> Dict[str, Any]:
    """Build cross-part absolute-quality synthesis artefacts (design)."""
    apply_rcparams()
    paths    = ensure_study_dirs(config.study_slug)
    csv_path = config.judgments_path or judgments_csv()
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Judgments CSV not found: {csv_path}. Run the judge first."
        )
    df = pd.read_csv(csv_path).copy()

    # ── Column normalisation (support legacy "score" column) ─────────────────
    if "quality_score" not in df.columns:
        if "score" in df.columns:
            # Legacy signed score: already on −10..+10, just clamp
            df["quality_score"] = df["score"].astype(float).apply(
                lambda s: max(-10, min(10, int(round(float(s)))))
            )
        else:
            raise ValueError("Judgments CSV missing 'quality_score' column.")

    if "entity" not in df.columns:
        # Try to infer from source_label: A/B blinding no longer in use for pseudo
        df["entity"] = "phoenix"

    required_cols = {"case_id", "part", "dimension", "judge_run", "quality_score", "entity"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Judgments CSV missing columns: {sorted(missing)}")

    df["quality_score"] = pd.to_numeric(df["quality_score"], errors="coerce")
    df["judge_run"]     = df["judge_run"].astype(str)
    df["case_id"]       = df["case_id"].astype(str)

    parts_present = [
        p for p in config.part_order
        if p in set(df["part"].astype(str))
    ]
    if not parts_present:
        raise ValueError("No configured parts present in judgments CSV.")

    work = df.loc[df["part"].astype(str).isin(parts_present)].copy()

    # ── Global entity model ───────────────────────────────────────────────────
    global_result = _fit_entity_lmm(work)
    global_tost   = _per_part_tost(work, config.tost_delta)
    global_std    = _standardized_paired_effect(work, seed=9000)

    # ── Per-part models ───────────────────────────────────────────────────────
    part_effects: Dict[str, Dict[str, Any]] = {}
    part_tosts:   Dict[str, Dict[str, Any]] = {}
    part_standardized: Dict[str, Dict[str, Any]] = {}
    raw_p: List[float] = []
    for idx, part in enumerate(parts_present):
        sub = work.loc[work["part"].astype(str) == part].copy()
        part_effects[part] = _fit_entity_lmm(sub)
        part_tosts[part]   = _per_part_tost(sub, config.tost_delta)
        part_standardized[part] = _standardized_paired_effect(sub, seed=9100 + idx)
        raw_p.append(float(part_effects[part]["p_value"]))

    holm_ps = holm_correct(raw_p)
    for part, padj in zip(parts_present, holm_ps):
        part_effects[part]["p_value_holm"] = float(padj)

    # ── Text report ──────────────────────────────────────────────────────────
    lines: List[str] = [
        "=" * 72,
        config.title,
        "Positive effect = PHOENIX outperforms HCP on the −10..+10 quality scale.",
        "=" * 72,
        "",
        f"Scale: −10..+10 (0 = acceptable, +10 = outstanding); TOST delta = ±{config.tost_delta} pts.",
        "Effect-coding: PHOENIX=+0.5, HCP=−0.5.",
        "Multiplicity correction (per-part): Holm–Bonferroni.",
        "",
        "--- Global entity model (pooled across all parts) ---",
        f"Method: {global_result['method']}",
    ]
    coef = global_result["coefficient"]
    p_g  = global_result["p_value"]
    lines.extend([
        f"PHOENIX − HCP effect: {coef:+.4f}",
        f"95% CI: [{global_result['ci_lower']:+.4f}, {global_result['ci_upper']:+.4f}]",
        f"Standardized paired effect: dz={global_std['cohen_dz']:+.4f} "
        f"[{global_std['cohen_dz_ci_lower']:+.4f}, {global_std['cohen_dz_ci_upper']:+.4f}] "
        f"({global_std['cohen_dz_category']})",
        f"p-value: {p_g:.4f} ({p_to_stars(p_g)})",
        "",
        f"Global TOST (delta=±{config.tost_delta}): "
        + (f"EQUIVALENT (p_TOST={global_tost['p_tost']:.4f})"
           if global_tost["equivalent"]
           else f"not equivalent (p_TOST={global_tost['p_tost']:.4f})"),
        f"  observed diff: {global_tost['observed_diff']:+.4f}",
        "",
        "--- Per-part entity effects (Holm-corrected) ---",
    ])
    for part in parts_present:
        e = part_effects[part]
        t = part_tosts[part]
        tost_str = (
            f"EQUIV (p_TOST={t['p_tost']:.4f})"
            if t["equivalent"] else
            f"not equiv (p_TOST={t['p_tost']:.4f})"
        )
        lines.append(
            f"  {_display_part(part)}: "
            f"Δ={e['coefficient']:+.4f}, "
            f"95% CI=[{e['ci_lower']:+.4f}, {e['ci_upper']:+.4f}], "
            f"dz={part_standardized[part]['cohen_dz']:+.4f}, "
            f"p_raw={e['p_value']:.4f}, p_holm={e['p_value_holm']:.4f} "
            f"({p_to_stars(e['p_value_holm'])}), TOST: {tost_str}"
        )

    # PHOENIX/HCP grand means per part
    lines.append("")
    lines.append("--- Quality means by entity and part ---")
    for part in parts_present:
        sub = work.loc[work["part"].astype(str) == part]
        for ent in ("phoenix", "hcp"):
            vals = sub.loc[sub["entity"] == ent, "quality_score"].dropna()
            if len(vals):
                lines.append(
                    f"  {_display_part(part)} [{ent.upper():6s}]: "
                    f"M={vals.mean():.3f}, SD={vals.std(ddof=1):.3f}, n={len(vals)}"
                )

    (paths["report_dir"] / config.report_name).write_text(
        "\n".join(lines), encoding="utf-8"
    )

    summary_rows = []
    for part in parts_present:
        sub = work.loc[work["part"].astype(str) == part]
        ph = sub.loc[sub["entity"] == "phoenix", "quality_score"].dropna()
        hc = sub.loc[sub["entity"] == "hcp", "quality_score"].dropna()
        e = part_effects[part]
        t = part_tosts[part]
        s = part_standardized[part]
        summary_rows.append({
            "part": part,
            "part_label": _display_part(part),
            "phoenix_mean": float(ph.mean()) if len(ph) else float("nan"),
            "hcp_mean": float(hc.mean()) if len(hc) else float("nan"),
            "effect_phoenix_minus_hcp": float(e["coefficient"]),
            "ci_lower": float(e["ci_lower"]),
            "ci_upper": float(e["ci_upper"]),
            "p_value_raw": float(e["p_value"]),
            "p_value_holm": float(e.get("p_value_holm", 1.0)),
            "cohen_dz": float(s["cohen_dz"]),
            "cohen_dz_ci_lower": float(s["cohen_dz_ci_lower"]),
            "cohen_dz_ci_upper": float(s["cohen_dz_ci_upper"]),
            "cohen_dz_category": str(s["cohen_dz_category"]),
            "n_pairs": int(s["n_pairs"]),
            "p_tost": float(t.get("p_tost", 1.0)),
            "tost_equivalent": bool(t.get("equivalent", False)),
            "method": e["method"],
            "converged": bool(e.get("converged", False)),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(paths["report_dir"] / "synthesis_part_summary.csv", index=False)

    # ─────────────────────────────────────────────────────────────────────────
    # Figure 1: entity-effect forest plot (PHOENIX − HCP Δ per part)
    # ─────────────────────────────────────────────────────────────────────────
    fig1, ax1 = plt.subplots(
        figsize=(9, max(4, 0.85 * len(parts_present) + 1.8))
    )
    forest_plot(
        ax1,
        dimensions=[_display_part(p) for p in parts_present],
        effects=[part_effects[p]["coefficient"] for p in parts_present],
        ci_lowers=[part_effects[p]["ci_lower"] for p in parts_present],
        ci_uppers=[part_effects[p]["ci_upper"] for p in parts_present],
        xlabel="Raw quality-point gap (PHOENIX − HCP; possible range −20 to +20)",
        ref_line=0.0,
        p_values=[part_effects[p]["p_value_holm"] for p in parts_present],
        tost_results=[part_tosts[p] for p in parts_present],
    )
    raw_lo = min([part_effects[p]["ci_lower"] for p in parts_present] + [-config.tost_delta])
    raw_hi = max([part_effects[p]["ci_upper"] for p in parts_present] + [config.tost_delta])
    raw_pad = max(0.4, 0.08 * (raw_hi - raw_lo))
    ax1.set_xlim(
        max(config.quality_min - config.quality_max, raw_lo - raw_pad),
        min(config.quality_max - config.quality_min, raw_hi + raw_pad),
    )
    plt.tight_layout()
    save_figure(fig1, paths["visuals_dir"] / "synthesis_part_forest.png")

    # ─────────────────────────────────────────────────────────────────────────
    # Figure 1b: standardized entity-effect forest plot (paired Cohen's dz)
    # ─────────────────────────────────────────────────────────────────────────
    fig1b, ax1b = plt.subplots(
        figsize=(9, max(4, 0.85 * len(parts_present) + 1.8))
    )
    standardized_forest_plot(
        ax1b,
        dimensions=[_display_part(p) for p in parts_present],
        effects=[part_standardized[p]["cohen_dz"] for p in parts_present],
        ci_lowers=[part_standardized[p]["cohen_dz_ci_lower"] for p in parts_present],
        ci_uppers=[part_standardized[p]["cohen_dz_ci_upper"] for p in parts_present],
        xlabel="Standardized paired effect (Cohen's dz; PHOENIX − HCP)",
        ref_line=0.0,
        p_values=[part_effects[p]["p_value_holm"] for p in parts_present],
    )
    d_lo = min([part_standardized[p]["cohen_dz_ci_lower"] for p in parts_present] + [-2.0])
    d_hi = max([part_standardized[p]["cohen_dz_ci_upper"] for p in parts_present] + [2.0])
    d_pad = max(0.25, 0.08 * (d_hi - d_lo))
    ax1b.set_xlim(d_lo - d_pad, d_hi + d_pad + 0.8)
    plt.tight_layout()
    save_figure(fig1b, paths["visuals_dir"] / "synthesis_standardized_effect_forest.png")

    # ─────────────────────────────────────────────────────────────────────────
    # Figure 2: Grouped raincloud PHOENIX vs HCP per part
    # ─────────────────────────────────────────────────────────────────────────
    fig2, axes = plt.subplots(
        1, len(parts_present),
        figsize=(max(7, 3.8 * len(parts_present)), 6.0),
        sharey=True,
    )
    if len(parts_present) == 1:
        axes = [axes]
    for ax, part in zip(axes, parts_present):
        sub  = work.loc[work["part"].astype(str) == part]
        ph_v = sub.loc[sub["entity"] == "phoenix", "quality_score"].to_numpy()
        hc_v = sub.loc[sub["entity"] == "hcp",     "quality_score"].to_numpy()
        holm_p = part_effects[part].get("p_value_holm", 1.0)
        raincloud_plot(
            ax,
            data_dict={"PHOENIX": ph_v, "HCP": hc_v},
            ylabel="Quality score (−10 to +10)",
            colors=[COLOR_PHOENIX, COLOR_HCP],
            adj_p=holm_p,
            ylim=(-10.5, 10.5),
            show_tost=part_tosts[part],
        )
        ax.axhline(0.0, color=PALETTE["ref_line"], linestyle="--",
                   linewidth=0.8, alpha=0.5)
    plt.tight_layout()
    save_figure(fig2, paths["visuals_dir"] / "synthesis_part_raincloud.png")

    # ─────────────────────────────────────────────────────────────────────────
    # Figure 3: Heatmap — mean quality by part × dimension (PHOENIX − HCP gap)
    # ─────────────────────────────────────────────────────────────────────────
    delta_df = (
        work.groupby(["part", "dimension", "entity"], as_index=False)["quality_score"]
        .mean()
    )
    # Pivot to wide: for each (part, dim) compute phoenix_mean − hcp_mean
    ph_df  = delta_df.loc[delta_df["entity"] == "phoenix"].rename(columns={"quality_score": "q_ph"})
    hcp_df = delta_df.loc[delta_df["entity"] == "hcp"].rename(columns={"quality_score": "q_hcp"})
    wide   = ph_df.merge(hcp_df, on=["part", "dimension"], how="outer")
    wide["gap"] = wide["q_ph"].fillna(0.0) - wide["q_hcp"].fillna(0.0)

    all_dims = list(dict.fromkeys(wide["dimension"].astype(str).tolist()))
    matrix   = np.full((len(parts_present), len(all_dims)), np.nan, dtype=float)
    for i, part in enumerate(parts_present):
        for j, dim in enumerate(all_dims):
            row = wide.loc[(wide["part"] == part) & (wide["dimension"] == dim)]
            if not row.empty:
                matrix[i, j] = float(row["gap"].iloc[0])

    fig3, ax3 = plt.subplots(
        figsize=(max(8, 0.75 * len(all_dims) + 2.5),
                 max(4, 0.65 * len(parts_present) + 1.5))
    )
    finite = np.abs(matrix[np.isfinite(matrix)])
    heat_lim = float(max(3.0, np.max(finite))) if finite.size else 3.0
    heat_lim = min(20.0, heat_lim)
    im = ax3.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-heat_lim, vmax=heat_lim)
    ax3.set_xticks(np.arange(len(all_dims)))
    ax3.set_xticklabels(
        [_display_dim(d) for d in all_dims],
        rotation=38, ha="right", fontsize=8,
    )
    ax3.set_yticks(np.arange(len(parts_present)))
    ax3.set_yticklabels([_display_part(p) for p in parts_present], fontsize=9)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isnan(v):
                continue
            ax3.text(
                j, i, f"{v:+.2f}",
                ha="center", va="center", fontsize=7,
                color="white" if abs(v) > 1.7 else "black",
            )
    plt.colorbar(im, ax=ax3, label="Raw quality-point gap (PHOENIX − HCP; possible range −20 to +20)")
    plt.tight_layout()
    save_figure(fig3, paths["visuals_dir"] / "synthesis_gap_heatmap.png")

    # ─────────────────────────────────────────────────────────────────────────
    # Figure 4: TOST equivalence panel
    # ─────────────────────────────────────────────────────────────────────────
    fig4, ax4 = plt.subplots(
        figsize=(8, max(3.5, 0.8 * len(parts_present) + 1.2))
    )
    tost_panel(
        ax4,
        dimensions=[_display_part(p) for p in parts_present],
        tost_results=[part_tosts[p] for p in parts_present],
    )
    plt.tight_layout()
    save_figure(fig4, paths["visuals_dir"] / "synthesis_tost.png")

    return {
        "global": global_result,
        "global_tost": global_tost,
        "global_standardized": global_std,
        "part_effects": part_effects,
        "part_tosts": part_tosts,
        "part_standardized": part_standardized,
    }
