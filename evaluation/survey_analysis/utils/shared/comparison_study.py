from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf

from .shared_stats import (
    PALETTE,
    apply_rcparams,
    bonferroni_correct,
    bootstrap_cohend_ci,
    cohen_d,
    fit_crossed_mixedlm,
    forest_plot,
    p_to_stars,
    raincloud_plot,
    save_figure,
    tost_panel,
    tost_test,
)
from .survey_paths import data_file, ensure_study_dirs


@dataclass(frozen=True)
class ComparisonStudyConfig:
    study_slug: str
    title: str
    report_name: str
    data_filename: str
    item_col: str
    dimension_order: Sequence[str]
    source_col: str = "source"
    rating_col: str = "rating"
    participant_col: str = "participant_ID"
    source_reference: str = "hcp"
    source_test: str = "phoenix"
    source_labels: Dict[str, str] = field(
        default_factory=lambda: {"hcp": "Healthcare Expert", "phoenix": "PHOENIX"}
    )
    y_label: str = "Rating (1–9 Likert)"
    tost_delta: float = 0.5    # equivalence margin in scale units (half a Likert step on 1–9)


def _display_label(value: str) -> str:
    return str(value).replace("_", " ").title()


def _structure_note(method: str) -> str:
    if method.startswith("Crossed LME"):
        return "participant-level pairing with crossed item variability"
    if method.startswith("LME (item intercept fallback)"):
        return "item-level random intercept only after richer crossed models failed"
    return "fallback comparison after mixed-model convergence issues"


def _run_source_mixed_model(
    sub: pd.DataFrame,
    *,
    item_col: str,
    source_col: str,
    participant_col: str,
) -> Dict[str, Any]:
    work = sub.copy()
    work["source_bin"] = (work[source_col].astype(str) == "phoenix").astype(int)
    predictors = ["source_bin"]
    if "shift_regime" in work.columns and work["shift_regime"].nunique() > 1:
        predictors.append("C(shift_regime)")
    formula = "rating ~ " + " + ".join(predictors)
    candidates = [
        {
            "label": "Crossed LME (participant intercept + source slope, item intercept)",
            "group_col": participant_col,
            "re_formula": "~source_bin",
            "variance_components": {"item": item_col},
        },
        {
            "label": "Crossed LME (participant intercept, item intercept)",
            "group_col": participant_col,
            "variance_components": {"item": item_col},
        },
        {
            "label": "Crossed LME (item intercept, participant intercept)",
            "group_col": item_col,
            "variance_components": {"participant": participant_col},
        },
    ]
    result = fit_crossed_mixedlm(
        work,
        formula=formula,
        effect_term="source_bin",
        candidates=candidates,
    )
    if result["method"] != "No converged mixed model":
        return result
    exc = result.get("error", "No converged mixed model.")
    try:
        model = smf.mixedlm(
            formula,
            data=work,
            groups=work[item_col],
        ).fit(reml=True, method="lbfgs")
        coef = float(model.params.get("source_bin", 0.0))
        se = float(model.bse.get("source_bin", 0.0))
        p_val = float(model.pvalues.get("source_bin", 1.0))
        ci_lo = coef - 1.96 * se
        ci_hi = coef + 1.96 * se
        return {
            "method": "LME (item intercept fallback)",
            "coefficient": coef,
            "se": se,
            "p_value": p_val,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "converged": bool(model.converged),
            "error": exc,
            "shapiro_w": np.nan,
            "shapiro_p": np.nan,
        }
    except Exception as exc2:
        exc = f"{exc} | item intercept fallback: {exc2}"
        phoenix_vals = work.loc[work[source_col].astype(str) == "phoenix", "rating"].astype(float).to_numpy()
        hcp_vals = work.loc[work[source_col].astype(str) == "hcp", "rating"].astype(float).to_numpy()
        stat, p_val = stats.mannwhitneyu(phoenix_vals, hcp_vals, alternative="two-sided")
        diff = float(np.mean(phoenix_vals) - np.mean(hcp_vals))
        return {
            "method": "Mann-Whitney U fallback",
            "coefficient": diff,
            "se": 0.0,
            "p_value": float(p_val),
            "ci_lower": diff,
            "ci_upper": diff,
            "error": str(exc),
            "statistic": float(stat),
            "shapiro_w": np.nan,
            "shapiro_p": np.nan,
        }


def _shift_delta_table(df: pd.DataFrame, *, source_col: str, rating_col: str) -> pd.DataFrame:
    grouped = (
        df.groupby(["dimension", "shift_regime", source_col], as_index=False)[rating_col]
        .mean()
        .pivot_table(index="dimension", columns=["shift_regime", source_col], values=rating_col)
    )
    delta: Dict[str, pd.Series] = {}
    for shift_regime in sorted({col[0] for col in grouped.columns}):
        phoenix_col = (shift_regime, "phoenix")
        hcp_col = (shift_regime, "hcp")
        if phoenix_col not in grouped.columns or hcp_col not in grouped.columns:
            continue
        delta[shift_regime] = grouped[phoenix_col] - grouped[hcp_col]
    if not delta:
        return pd.DataFrame()
    table = pd.DataFrame(delta)
    table.index.name = "dimension"
    return table


def run_comparison_study(config: ComparisonStudyConfig) -> None:
    apply_rcparams()
    paths = ensure_study_dirs(config.study_slug)
    df = pd.read_csv(data_file(config.data_filename))
    dimensions = [dim for dim in config.dimension_order if dim in set(df["dimension"].astype(str))]
    rating_values = df[config.rating_col].astype(float).to_numpy()
    scale_min = float(np.nanmin(rating_values))
    scale_max = float(np.nanmax(rating_values))

    report_lines = [
        "=" * 68,
        config.title,
        f"Comparison: {config.source_labels[config.source_reference]} vs {config.source_labels[config.source_test]}",
        "=" * 68,
    ]

    dim_results: Dict[str, Dict[str, Any]] = {}
    raw_p_values = []

    # ── Primary mixed-model loop ─────────────────────────────────────────────
    for dim in dimensions:
        sub = df.loc[df["dimension"].astype(str) == dim].copy()
        ref_vals  = sub.loc[sub[config.source_col].astype(str) == config.source_reference,
                            config.rating_col].astype(float).to_numpy()
        test_vals = sub.loc[sub[config.source_col].astype(str) == config.source_test,
                            config.rating_col].astype(float).to_numpy()
        model_result = _run_source_mixed_model(
            sub.rename(columns={config.rating_col: "rating"}),
            item_col=config.item_col,
            source_col=config.source_col,
            participant_col=config.participant_col,
        )
        raw_p_values.append(float(model_result["p_value"]))

        # TOST equivalence test
        tost_result = tost_test(test_vals, ref_vals, delta=config.tost_delta)

        effect_d = cohen_d(test_vals, ref_vals)
        d_ci_lo, d_ci_hi = bootstrap_cohend_ci(test_vals, ref_vals)
        dim_results[dim] = {
            "reference_vals": ref_vals,
            "test_vals":      test_vals,
            "result":         model_result,
            "tost":           tost_result,
            "effect_d":       effect_d,
            "effect_d_ci":    (d_ci_lo, d_ci_hi),
        }

    corrected_p_values = bonferroni_correct(raw_p_values)
    for dim, corrected_p in zip(dimensions, corrected_p_values):
        dim_results[dim]["result"]["p_value_adj"] = float(corrected_p)

    # ── Report text ─────────────────────────────────────────────────────────
    report_lines.append(
        "Multiplicity control: Bonferroni correction across within-study dimensions."
    )
    report_lines.append(
        f"Equivalence testing: TOST with margin ±{config.tost_delta} scale units "
        f"(~half a minimal detectable difference on a 1–9 Likert scale)."
    )

    for dim in dimensions:
        result    = dim_results[dim]
        ref_vals  = result["reference_vals"]
        test_vals = result["test_vals"]
        model_result  = result["result"]
        tost_result   = result["tost"]
        effect_d  = result["effect_d"]
        d_ci_lo, d_ci_hi = result["effect_d_ci"]
        sub = df.loc[df["dimension"].astype(str) == dim].copy()

        equiv_str = (
            f"EQUIVALENT (p_TOST={tost_result['p_tost']:.4f})"
            if tost_result["equivalent"]
            else f"not equivalent (p_TOST={tost_result['p_tost']:.4f})"
        )
        report_lines.extend(
            [
                "",
                f"Dimension: {dim}",
                f"  {config.source_labels[config.source_reference]} "
                f"mean={np.mean(ref_vals):.3f}, SD={np.std(ref_vals, ddof=1):.3f}, "
                f"n={len(ref_vals)}",
                f"  {config.source_labels[config.source_test]} "
                f"mean={np.mean(test_vals):.3f}, SD={np.std(test_vals, ddof=1):.3f}, "
                f"n={len(test_vals)}",
                f"  Method: {model_result['method']}",
                f"  Model structure: {_structure_note(model_result['method'])}",
                f"  PHOENIX - HCP coefficient: {model_result['coefficient']:.4f}",
                f"  p-value (raw): {model_result['p_value']:.4f}",
                f"  p-value (Bonferroni): {model_result['p_value_adj']:.4f} "
                f"({p_to_stars(model_result['p_value_adj'])})",
                f"  Cohen's d: {effect_d:.4f} [{d_ci_lo:.4f}, {d_ci_hi:.4f}]",
                f"  TOST (±{config.tost_delta}): {equiv_str}",
                f"    observed diff={tost_result['observed_diff']:.4f}, "
                f"SE={tost_result['pooled_se']:.4f}",
                f"    t_upper={tost_result['t_upper']:.4f} "
                f"(p={tost_result['p_upper']:.4f}), "
                f"t_lower={tost_result['t_lower']:.4f} "
                f"(p={tost_result['p_lower']:.4f})",
            ]
        )
        if not np.isnan(model_result.get("shapiro_w", np.nan)):
            report_lines.append(
                f"  Residual Shapiro-Wilk: "
                f"W={model_result['shapiro_w']:.4f}, p={model_result['shapiro_p']:.4f}"
            )
        if "shift_regime" in sub.columns:
            regime_delta = (
                sub.groupby(["shift_regime", config.source_col], as_index=False)[
                    config.rating_col
                ]
                .mean()
                .pivot(
                    index="shift_regime",
                    columns=config.source_col,
                    values=config.rating_col,
                )
            )
            if {"hcp", "phoenix"}.issubset(regime_delta.columns):
                report_lines.append("  Shift-regime deltas (PHOENIX - HCP):")
                for regime_name, row in regime_delta.iterrows():
                    report_lines.append(
                        f"    {regime_name}: {float(row['phoenix'] - row['hcp']):.3f}"
                    )

    report_path = paths["report_dir"] / config.report_name
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    # ── Summary CSV ─────────────────────────────────────────────────────────
    pd.DataFrame(
        [
            {
                "dimension":          dim,
                "mean_hcp":           float(np.mean(dim_results[dim]["reference_vals"])),
                "mean_phoenix":       float(np.mean(dim_results[dim]["test_vals"])),
                "phoenix_minus_hcp":  float(dim_results[dim]["result"]["coefficient"]),
                "p_value_raw":        float(dim_results[dim]["result"]["p_value"]),
                "p_value_bonferroni": float(dim_results[dim]["result"]["p_value_adj"]),
                "effect_size_d":      float(dim_results[dim]["effect_d"]),
                "p_tost":             float(dim_results[dim]["tost"]["p_tost"]),
                "tost_equivalent":    bool(dim_results[dim]["tost"]["equivalent"]),
            }
            for dim in dimensions
        ]
    ).to_csv(paths["report_dir"] / f"{config.study_slug}_summary.csv", index=False)

    # ────────────────────────────────────────────────────────────────────────
    # Figure 1 — Raincloud plots (one panel per dimension)
    # ────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        1,
        len(dimensions),
        figsize=(max(5, 4.4 * len(dimensions)), 6.0),
        sharey=True,
    )
    if len(dimensions) == 1:
        axes = [axes]

    for ax, dim in zip(axes, dimensions):
        result = dim_results[dim]
        data_dict = {
            config.source_labels[config.source_reference]: result["reference_vals"],
            config.source_labels[config.source_test]:      result["test_vals"],
        }
        raincloud_plot(
            ax,
            data_dict=data_dict,
            title=_display_label(dim),
            ylabel=config.y_label,
            colors=[PALETTE["secondary"], PALETTE["primary"]],
            adj_p=result["result"]["p_value_adj"],
            ylim=(scale_min - 0.5, scale_max + 1.2),
            show_tost=result["tost"],
        )

    fig.suptitle(config.title, fontsize=13, y=1.02)
    plt.tight_layout()
    save_figure(fig, paths["visuals_dir"] / f"{config.study_slug}_ratings_raincloud.png")

    # ────────────────────────────────────────────────────────────────────────
    # Figure 2 — Forest plot with TOST badges and effect-size bands
    # ────────────────────────────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(
        figsize=(10, max(4, 0.9 * len(dimensions) + 1.5))
    )
    forest_plot(
        ax2,
        dimensions=[_display_label(dim) for dim in dimensions],
        effects=[dim_results[dim]["result"]["coefficient"] for dim in dimensions],
        ci_lowers=[dim_results[dim]["result"]["ci_lower"] for dim in dimensions],
        ci_uppers=[dim_results[dim]["result"]["ci_upper"] for dim in dimensions],
        title=f"{config.title}: PHOENIX − HCP effect by dimension",
        xlabel="Model coefficient / mean difference (rating units)",
        ref_line=0.0,
        p_values=[dim_results[dim]["result"]["p_value_adj"] for dim in dimensions],
        tost_results=[dim_results[dim]["tost"] for dim in dimensions],
    )
    plt.tight_layout()
    save_figure(fig2, paths["visuals_dir"] / f"{config.study_slug}_effect_forest.png")

    # ────────────────────────────────────────────────────────────────────────
    # Figure 3 — Standalone TOST equivalence panel
    # ────────────────────────────────────────────────────────────────────────
    fig3, ax3 = plt.subplots(
        figsize=(8, max(3.5, 0.8 * len(dimensions) + 1.2))
    )
    tost_panel(
        ax3,
        dimensions=[_display_label(dim) for dim in dimensions],
        tost_results=[dim_results[dim]["tost"] for dim in dimensions],
        title=f"{config.title}: Equivalence Test (TOST, δ = ±{config.tost_delta})",
    )
    plt.tight_layout()
    save_figure(fig3, paths["visuals_dir"] / f"{config.study_slug}_tost_equivalence.png")

    # ────────────────────────────────────────────────────────────────────────
    # Figure 4 (optional) — Distribution-shift sensitivity heatmap
    # ────────────────────────────────────────────────────────────────────────
    if "shift_regime" in df.columns:
        delta_table = _shift_delta_table(
            df, source_col=config.source_col, rating_col=config.rating_col
        )
        if not delta_table.empty:
            fig4, ax4 = plt.subplots(
                figsize=(
                    max(7, 1.8 * len(delta_table.columns)),
                    max(4, 0.75 * len(delta_table.index) + 1.5),
                )
            )
            im = ax4.imshow(delta_table.values, cmap="RdBu_r", aspect="auto")
            ax4.set_xticks(np.arange(len(delta_table.columns)))
            ax4.set_xticklabels(
                [_display_label(col) for col in delta_table.columns],
                rotation=25, ha="right",
            )
            ax4.set_yticks(np.arange(len(delta_table.index)))
            ax4.set_yticklabels(
                [_display_label(idx) for idx in delta_table.index]
            )
            ax4.set_title(
                f"{config.title}: distribution-shift sensitivity (PHOENIX − HCP)"
            )
            for i in range(delta_table.shape[0]):
                for j in range(delta_table.shape[1]):
                    ax4.text(
                        j, i, f"{delta_table.iloc[i, j]:.2f}",
                        ha="center", va="center", fontsize=8,
                    )
            plt.colorbar(im, ax=ax4, label="Mean rating delta")
            plt.tight_layout()
            save_figure(
                fig4,
                paths["visuals_dir"] / f"{config.study_slug}_shift_heatmap.png",
            )
