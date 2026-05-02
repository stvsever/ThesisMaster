"""
Generic per-part analysis for the absolute-quality LLM-as-judge design.

Statistical model
-----------------
For each (part, dimension) the long-format CSV contains one row per
``(case_id, part, dimension, judge_run, entity)`` with:

    quality_score  bipolar −10..+10 quality rating (0 = acceptable)
    entity         "phoenix" | "hcp"

The primary model estimates the PHOENIX–HCP quality gap:

    quality_score ~ entity + (1 | case_id) + (1 | judge_run)

Where ``entity`` is effect-coded (HCP = -0.5, PHOENIX = +0.5) so the
intercept = grand mean quality and the entity coefficient = PHOENIX − HCP.

Equivalence testing uses a one-sample TOST on the entity-difference scores
(phoenix_score − hcp_score per cell) with a default margin of ±1.5 quality
points (7.5% of the −10..+10 range).

Outputs
-------
Per-part report text, summary CSV, grouped raincloud plot, forest plot, and
TOST equivalence panel.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

from .shared_stats import (
    PALETTE,
    apply_rcparams,
    bootstrap_ci_mean,
    bootstrap_cohend_one_sample_ci,
    cohen_d_one_sample,
    effect_size_category,
    fit_crossed_mixedlm,
    forest_plot,
    holm_correct,
    p_to_stars,
    raincloud_plot,
    save_figure,
    standardized_forest_plot,
    tost_panel,
    tost_test_one_sample,
)
from .survey_paths import ensure_study_dirs, judgments_csv


@dataclass(frozen=True)
class ComparisonStudyConfig:
    """Configuration for a single absolute-quality per-part comparison study."""

    study_slug: str
    part: str
    title: str
    report_name: str
    dimension_order: Sequence[str]
    judgments_path: Optional[Path] = None
    score_col: str = "quality_score"
    entity_col: str = "entity"
    case_col: str = "case_id"
    run_col: str = "judge_run"
    quality_min: int = -10
    quality_max: int = 10
    y_label: str = "Quality score (−10 = catastrophic, 0 = acceptable, +10 = outstanding)"
    tost_delta: float = 1.5   # equivalence margin: 7.5% of the −10..+10 range


def _display_label(value: str) -> str:
    return str(value).replace("_", " ").title()


def _mean_ci(values: np.ndarray) -> Tuple[float, float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2:
        m = float(np.mean(vals)) if len(vals) else 0.0
        return m, m, m
    mean = float(np.mean(vals))
    lo, hi = bootstrap_ci_mean(vals, n_boot=2000, seed=42)
    return mean, lo, hi


def _paired_entity_differences(
    sub: pd.DataFrame,
    *,
    case_col: str,
    run_col: str,
) -> np.ndarray:
    """Return PHOENIX minus HCP differences paired by case and judge run."""
    paired = (
        sub.pivot_table(
            index=[case_col, run_col],
            columns="entity",
            values="quality_score",
            aggfunc="mean",
        )
        .dropna(subset=["phoenix", "hcp"], how="any")
    )
    if paired.empty:
        return np.asarray([], dtype=float)
    return (paired["phoenix"].astype(float) - paired["hcp"].astype(float)).to_numpy()


def _fit_entity_lmm(
    sub: pd.DataFrame,
    *,
    entity_col: str,
    case_col: str,
    run_col: str,
) -> Dict[str, Any]:
    """
    Fit the entity-predictor LMM and fall back to a Welch t-test.

    The entity predictor is effect-coded so the coefficient gives the
    PHOENIX − HCP quality gap on the −10..+10 scale.
    """
    work = sub.copy()
    # Effect coding: PHOENIX = +0.5, HCP = -0.5
    work["entity_ec"] = work[entity_col].map({"phoenix": 0.5, "hcp": -0.5}).fillna(0.0)

    result = fit_crossed_mixedlm(
        work,
        formula="quality_score ~ entity_ec",
        effect_term="entity_ec",
        candidates=[
            {
                "label": "Crossed LMM (case intercept + judge_run variance component)",
                "group_col": case_col,
                "variance_components": {"judge_run": run_col},
            },
            {
                "label": "Crossed LMM (judge_run intercept + case variance component)",
                "group_col": run_col,
                "variance_components": {"case": case_col},
            },
            {
                "label": "LMM (case intercept fallback)",
                "group_col": case_col,
                "variance_components": {},
            },
        ],
    )

    se = float(result.get("se", 0.0) or 0.0)
    ci_width = abs(
        float(result.get("ci_upper", 0.0)) - float(result.get("ci_lower", 0.0))
    )
    if (
        result["method"] != "No converged mixed model"
        and np.isfinite(se)
        and se > 1e-6
        and ci_width > 1e-6
    ):
        return result

    # Fallback: Welch t-test on per-entity means
    phoenix_vals = work.loc[work[entity_col] == "phoenix", "quality_score"].astype(float).dropna().values
    hcp_vals = work.loc[work[entity_col] == "hcp", "quality_score"].astype(float).dropna().values
    diff = float(np.mean(phoenix_vals) - np.mean(hcp_vals)) if (len(phoenix_vals) and len(hcp_vals)) else 0.0
    try:
        t_stat, p_val = stats.ttest_ind(phoenix_vals, hcp_vals, equal_var=False)
    except Exception:
        t_stat, p_val = float("nan"), 1.0
    se_diff = (
        float(np.sqrt(np.var(phoenix_vals, ddof=1) / len(phoenix_vals)
                      + np.var(hcp_vals, ddof=1) / len(hcp_vals)))
        if len(phoenix_vals) > 1 and len(hcp_vals) > 1
        else 0.0
    )
    return {
        "method": "Welch t-test fallback",
        "group_col": None,
        "variance_components": {},
        "coefficient": diff,
        "se": se_diff,
        "ci_lower": diff - 1.96 * se_diff,
        "ci_upper": diff + 1.96 * se_diff,
        "p_value": float(p_val) if np.isfinite(p_val) else 1.0,
        "converged": False,
        "t_statistic": float(t_stat) if np.isfinite(t_stat) else float("nan"),
        "error": result.get("error", "mixed model produced degenerate standard error"),
    }


def run_comparison_study(config: ComparisonStudyConfig) -> Dict[str, Any]:
    """Run one per-part absolute-quality PHOENIX–HCP comparison study."""
    apply_rcparams()
    paths = ensure_study_dirs(config.study_slug)
    csv_path = config.judgments_path or judgments_csv()
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Judgments CSV not found: {csv_path}. Run the judge first."
        )

    df_all = pd.read_csv(csv_path)

    # Handle both old (score) and new (quality_score) column names
    if "quality_score" not in df_all.columns and "score" in df_all.columns:
        df_all = df_all.rename(columns={"score": "quality_score"})
    if "entity" not in df_all.columns:
        # Legacy: derive entity from source columns if present
        if "winner_source" in df_all.columns:
            # Not meaningful for absolute quality; skip gracefully
            df_all["entity"] = "unknown"
        else:
            raise ValueError(
                "Judgments CSV has no 'entity' column. "
                "Re-run the judge with the absolute-quality schema."
            )

    required = {"case_id", "part", "dimension", "judge_run", "quality_score", "entity"}
    missing = required.difference(df_all.columns)
    if missing:
        raise ValueError(f"Judgments CSV is missing required columns: {sorted(missing)}")

    df = df_all.loc[df_all["part"].astype(str) == config.part].copy()
    if df.empty:
        raise ValueError(f"No rows for part={config.part!r} in {csv_path}.")
    df["quality_score"] = df["quality_score"].astype(float)

    available = set(df["dimension"].astype(str).unique().tolist())
    dimensions = [d for d in config.dimension_order if d in available]
    if not dimensions:
        raise ValueError(
            f"None of the configured dimensions {list(config.dimension_order)} "
            f"are present in the data for {config.part}."
        )

    # ── Report header ────────────────────────────────────────────────────────
    report_lines: List[str] = [
        "=" * 72,
        config.title,
        "LLM-as-judge: absolute quality comparison (PHOENIX vs HCP)",
        "=" * 72,
        "",
        f"Scale: {config.quality_min}..{config.quality_max} "
        f"(0 = acceptable baseline, +10 = outstanding, −10 = catastrophic failure).",
        f"Equivalence margin (TOST): ± {config.tost_delta} quality points.",
        "Multiplicity correction: Holm–Bonferroni across dimensions.",
        f"Primary model: quality_score ~ entity + (1|{config.case_col}) + "
        f"(1|{config.run_col})",
        "",
    ]

    dim_results: Dict[str, Dict[str, Any]] = {}
    raw_p_values: List[float] = []

    for dim in dimensions:
        sub = df.loc[df["dimension"].astype(str) == dim].copy()
        if sub.empty:
            continue

        phoenix_vals = (
            sub.loc[sub["entity"] == "phoenix", "quality_score"]
            .astype(float).dropna().to_numpy()
        )
        hcp_vals = (
            sub.loc[sub["entity"] == "hcp", "quality_score"]
            .astype(float).dropna().to_numpy()
        )

        # Difference scores for TOST and Cohen's d, paired by case/run.
        diff_scores = _paired_entity_differences(
            sub, case_col=config.case_col, run_col=config.run_col
        )

        model_result = _fit_entity_lmm(
            sub, entity_col="entity", case_col=config.case_col, run_col=config.run_col
        )
        raw_p_values.append(float(model_result["p_value"]))
        tost_result = tost_test_one_sample(diff_scores, delta=config.tost_delta) if len(diff_scores) >= 2 else {
            "equivalent": False, "p_tost": 1.0, "observed_diff": 0.0, "pooled_se": 0.0
        }

        phoenix_mean, phoenix_ci_lo, phoenix_ci_hi = _mean_ci(phoenix_vals)
        hcp_mean, hcp_ci_lo, hcp_ci_hi = _mean_ci(hcp_vals)

        effect_d = cohen_d_one_sample(diff_scores) if len(diff_scores) >= 2 else 0.0
        d_ci_lo, d_ci_hi = (
            bootstrap_cohend_one_sample_ci(diff_scores, seed=1000 + len(dim_results))
            if len(diff_scores) >= 2 else (0.0, 0.0)
        )

        dim_results[dim] = {
            "phoenix_vals": phoenix_vals,
            "hcp_vals": hcp_vals,
            "diff_scores": diff_scores,
            "result": model_result,
            "tost": tost_result,
            "phoenix_mean": phoenix_mean,
            "phoenix_ci": (phoenix_ci_lo, phoenix_ci_hi),
            "hcp_mean": hcp_mean,
            "hcp_ci": (hcp_ci_lo, hcp_ci_hi),
            "effect_d": effect_d,
            "effect_d_ci": (d_ci_lo, d_ci_hi),
            "effect_d_category": effect_size_category(effect_d),
        }

    corrected = holm_correct(raw_p_values)
    for dim, p_adj in zip(list(dim_results.keys()), corrected):
        dim_results[dim]["result"]["p_value_adj"] = float(p_adj)

    # ── Per-dimension report lines ───────────────────────────────────────────
    for dim, r in dim_results.items():
        m, t = r["result"], r["tost"]
        coef = float(m.get("coefficient", 0.0))
        equiv = (
            f"EQUIVALENT (p_TOST={t['p_tost']:.4f})"
            if t.get("equivalent")
            else f"not equivalent (p_TOST={t['p_tost']:.4f})"
        )
        report_lines.extend([
            f"Dimension: {dim}",
            f"  PHOENIX: mean={r['phoenix_mean']:+.3f} (n={len(r['phoenix_vals'])}), "
            f"  HCP: mean={r['hcp_mean']:+.3f} (n={len(r['hcp_vals'])})",
            f"  PHOENIX − HCP effect: {coef:+.4f} quality points",
            f"  95% CI: [{m['ci_lower']:+.4f}, {m['ci_upper']:+.4f}]",
            f"  p-value (raw): {m['p_value']:.4f}",
            f"  p-value (Holm): {m['p_value_adj']:.4f} "
            f"({p_to_stars(m['p_value_adj'])})",
            f"  Cohen's dz (paired difference): {r['effect_d']:+.4f} "
            f"[{r['effect_d_ci'][0]:+.4f}, {r['effect_d_ci'][1]:+.4f}] "
            f"({r['effect_d_category']})",
            f"  TOST (+/- {config.tost_delta}): {equiv}",
            f"  Method: {m['method']}",
            "",
        ])

    # ── Save report + summary ────────────────────────────────────────────────
    (paths["report_dir"] / config.report_name).write_text(
        "\n".join(report_lines), encoding="utf-8",
    )

    summary_rows = []
    for dim, r in dim_results.items():
        m, t = r["result"], r["tost"]
        summary_rows.append({
            "part": config.part,
            "dimension": dim,
            "n_phoenix": int(len(r["phoenix_vals"])),
            "n_hcp": int(len(r["hcp_vals"])),
            "phoenix_mean": float(r["phoenix_mean"]),
            "hcp_mean": float(r["hcp_mean"]),
            "effect_phoenix_minus_hcp": float(m.get("coefficient", 0.0)),
            "ci_lower": float(m["ci_lower"]),
            "ci_upper": float(m["ci_upper"]),
            "p_value_raw": float(m["p_value"]),
            "p_value_holm": float(m.get("p_value_adj", 1.0)),
            "cohen_d_diff": float(r["effect_d"]),
            "cohen_dz": float(r["effect_d"]),
            "cohen_dz_ci_lower": float(r["effect_d_ci"][0]),
            "cohen_dz_ci_upper": float(r["effect_d_ci"][1]),
            "cohen_dz_category": str(r["effect_d_category"]),
            "p_tost": float(t.get("p_tost", 1.0)),
            "tost_equivalent": bool(t.get("equivalent", False)),
            "method": m["method"],
            "converged": bool(m.get("converged", False)),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = paths["report_dir"] / f"{config.study_slug}_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # ── Visualisations ───────────────────────────────────────────────────────
    n_dims = len(dim_results)

    # 1. Grouped raincloud (PHOENIX vs HCP per dimension)
    fig, axes = plt.subplots(
        1, n_dims, figsize=(max(5, 3.8 * n_dims), 6.0), sharey=True,
    )
    if n_dims == 1:
        axes = [axes]
    for ax, dim in zip(axes, dim_results.keys()):
        r = dim_results[dim]
        raincloud_plot(
            ax,
            data_dict={"PHOENIX": r["phoenix_vals"], "HCP": r["hcp_vals"]},
            ylabel=config.y_label,
            colors=[PALETTE["primary"], PALETTE["secondary"]],
            ylim=(config.quality_min - 0.5, config.quality_max + 0.5),
        )
        ax.text(
            0.5, -0.16,
            f"Δ={r['result'].get('coefficient', 0):+.2f}; "
            f"Holm p={r['result'].get('p_value_adj', 1.0):.3f}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            color=PALETTE["ref_line"],
        )
    plt.tight_layout()
    save_figure(
        fig,
        paths["visuals_dir"] / f"{config.study_slug}_quality_raincloud.png",
    )

    # 2. Forest plot of entity effects
    fig2, ax2 = plt.subplots(figsize=(10, max(4, 0.9 * n_dims + 1.5)))
    forest_plot(
        ax2,
        dimensions=[_display_label(d) for d in dim_results.keys()],
        effects=[r["result"].get("coefficient", 0) for r in dim_results.values()],
        ci_lowers=[r["result"]["ci_lower"] for r in dim_results.values()],
        ci_uppers=[r["result"]["ci_upper"] for r in dim_results.values()],
        xlabel="Raw quality-point gap (PHOENIX − HCP; possible range −20 to +20)",
        ref_line=0.0,
        p_values=[r["result"].get("p_value_adj", 1.0) for r in dim_results.values()],
        tost_results=[r["tost"] for r in dim_results.values()],
    )
    all_lo = [r["result"]["ci_lower"] for r in dim_results.values()]
    all_hi = [r["result"]["ci_upper"] for r in dim_results.values()]
    lo = min(all_lo + [-config.tost_delta])
    hi = max(all_hi + [config.tost_delta])
    pad = max(0.2, 0.10 * (hi - lo))
    diff_min = config.quality_min - config.quality_max
    diff_max = config.quality_max - config.quality_min
    ax2.set_xlim(
        max(diff_min, lo - pad),
        min(diff_max, hi + pad),
    )
    plt.tight_layout()
    save_figure(fig2, paths["visuals_dir"] / f"{config.study_slug}_effect_forest.png")

    # 3. Standardized forest plot of paired Cohen's dz
    fig3, ax3 = plt.subplots(figsize=(8, max(3.5, 0.8 * n_dims + 1.2)))
    standardized_forest_plot(
        ax3,
        dimensions=[_display_label(d) for d in dim_results.keys()],
        effects=[r["effect_d"] for r in dim_results.values()],
        ci_lowers=[r["effect_d_ci"][0] for r in dim_results.values()],
        ci_uppers=[r["effect_d_ci"][1] for r in dim_results.values()],
        xlabel="Standardized paired effect (Cohen's dz; PHOENIX − HCP)",
        ref_line=0.0,
        p_values=[r["result"].get("p_value_adj", 1.0) for r in dim_results.values()],
    )
    all_d_lo = [r["effect_d_ci"][0] for r in dim_results.values()]
    all_d_hi = [r["effect_d_ci"][1] for r in dim_results.values()]
    d_lo = min(all_d_lo + [-2.0])
    d_hi = max(all_d_hi + [2.0])
    d_pad = max(0.2, 0.08 * (d_hi - d_lo))
    ax3.set_xlim(d_lo - d_pad, d_hi + d_pad + 0.8)
    plt.tight_layout()
    save_figure(
        fig3,
        paths["visuals_dir"] / f"{config.study_slug}_standardized_effect_forest.png",
    )

    # 4. TOST equivalence panel
    fig4, ax4 = plt.subplots(figsize=(8, max(3.5, 0.8 * n_dims + 1.2)))
    tost_panel(
        ax4,
        dimensions=[_display_label(d) for d in dim_results.keys()],
        tost_results=[r["tost"] for r in dim_results.values()],
    )
    plt.tight_layout()
    save_figure(fig4, paths["visuals_dir"] / f"{config.study_slug}_tost_equivalence.png")

    return {
        "study_slug": config.study_slug,
        "part": config.part,
        "dim_results": dim_results,
        "summary_df": summary_df,
        "report_path": paths["report_dir"] / config.report_name,
        "summary_path": summary_path,
    }
