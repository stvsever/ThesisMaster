from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

from .shared_stats import (
    PALETTE,
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
class HolisticStudyConfig:
    study_slug: str
    title: str
    report_name: str
    data_filename: str
    tost_delta: float = 0.05    # normalised score scale (0–1); ~5 pp equivalence margin


def _display_label(value: str) -> str:
    return str(value).replace("_", " ").title()


def _holm_correct(p_values: list[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * len(p_values)
    running_max = 0.0
    n_tests = len(p_values)
    for rank, (original_idx, p_value) in enumerate(indexed, start=1):
        scaled = (n_tests - rank + 1) * p_value
        running_max = max(running_max, scaled)
        adjusted[original_idx] = min(running_max, 1.0)
    return adjusted


def run_holistic_reasoner_comparison(config: HolisticStudyConfig) -> None:
    paths = ensure_study_dirs(config.study_slug)
    df = pd.read_csv(data_file(config.data_filename))
    studies = sorted(df["study_id"].astype(str).unique().tolist())

    report_lines = [
        "=" * 72,
        config.title,
        "Human-in-the-loop comparison focused on PHOENIX vs healthcare expert outputs",
        "=" * 72,
    ]

    work = df.copy()
    work["reasoner_bin"] = (work["reasoner_group"].astype(str) == "phoenix").astype(int)
    formula_terms = ["reasoner_bin", "C(study_id)", "C(dimension)"]
    if "shift_regime" in work.columns and work["shift_regime"].nunique() > 1:
        formula_terms.append("C(shift_regime)")

    # ── Unified cross-study mixed model ──────────────────────────────────────
    unified_result = fit_crossed_mixedlm(
        work,
        formula="normalized_score ~ " + " + ".join(formula_terms),
        effect_term="reasoner_bin",
        candidates=[
            {
                "label": "Crossed LME (participant intercept + reasoner slope, task intercept, answer-block intercept)",
                "group_col": "participant_ID",
                "re_formula": "~reasoner_bin",
                "variance_components": {"task": "task_key", "answer_block": "response_run_id"},
            },
            {
                "label": "Crossed LME (participant intercept, task intercept, answer-block intercept)",
                "group_col": "participant_ID",
                "variance_components": {"task": "task_key", "answer_block": "response_run_id"},
            },
            {
                "label": "Crossed LME (task intercept, participant intercept)",
                "group_col": "task_key",
                "variance_components": {"participant": "participant_ID"},
            },
        ],
    )

    # Global TOST on normalised scores
    phoenix_all = work.loc[work["reasoner_group"].astype(str) == "phoenix", "normalized_score"].to_numpy()
    hcp_all     = work.loc[work["reasoner_group"].astype(str) == "hcp",     "normalized_score"].to_numpy()
    global_tost = tost_test(phoenix_all, hcp_all, delta=config.tost_delta)

    if unified_result["method"] != "No converged mixed model":
        report_lines.extend(
            [
                "",
                "--- Unified mixed model ---",
                f"Method: {unified_result['method']}",
                f"PHOENIX - HCP coefficient: {unified_result['coefficient']:.4f}",
                f"95% CI: [{unified_result['ci_lower']:.4f}, {unified_result['ci_upper']:.4f}]",
                f"p-value: {unified_result['p_value']:.4f} ({p_to_stars(unified_result['p_value'])})",
                "Primary estimand adjusts for study and dimension while accounting for "
                "participant severity, task difficulty, and participant-specific answer-block clustering.",
            ]
        )
        if not np.isnan(unified_result.get("shapiro_w", np.nan)):
            report_lines.append(
                f"Residual Shapiro-Wilk: W={unified_result['shapiro_w']:.4f}, "
                f"p={unified_result['shapiro_p']:.4f}"
            )
    else:
        stat, p_val = stats.mannwhitneyu(phoenix_all, hcp_all, alternative="two-sided")
        report_lines.extend(
            [
                "",
                "--- Unified mixed model ---",
                "Method: Mann-Whitney fallback",
                f"U-statistic: {float(stat):.4f}",
                f"p-value: {float(p_val):.4f}",
                f"Fallback reason: {unified_result.get('error', 'No converged mixed model.')}",
            ]
        )

    equiv_str = (
        f"EQUIVALENT (p_TOST={global_tost['p_tost']:.4f})"
        if global_tost["equivalent"]
        else f"not equivalent (p_TOST={global_tost['p_tost']:.4f})"
    )
    report_lines.extend(
        [
            "",
            f"Global TOST (δ=±{config.tost_delta}): {equiv_str}",
            f"  observed diff={global_tost['observed_diff']:.4f}, SE={global_tost['pooled_se']:.4f}",
        ]
    )

    # ── Study-level follow-up models ─────────────────────────────────────────
    study_effects: Dict[str, Dict[str, Any]] = {}
    study_tosts:   Dict[str, Dict[str, Any]] = {}
    raw_study_p_values = []

    for study_id in studies:
        sub = work.loc[work["study_id"].astype(str) == study_id].copy()
        phoenix_vals = sub.loc[
            sub["reasoner_group"].astype(str) == "phoenix", "normalized_score"
        ].to_numpy()
        hcp_vals = sub.loc[
            sub["reasoner_group"].astype(str) == "hcp", "normalized_score"
        ].to_numpy()
        diff = float(np.mean(phoenix_vals) - np.mean(hcp_vals))

        result = fit_crossed_mixedlm(
            sub,
            formula="normalized_score ~ reasoner_bin",
            effect_term="reasoner_bin",
            candidates=[
                {
                    "label": "Crossed LME (participant intercept + reasoner slope, task intercept, answer-block intercept)",
                    "group_col": "participant_ID",
                    "re_formula": "~reasoner_bin",
                    "variance_components": {"task": "task_key", "answer_block": "response_run_id"},
                },
                {
                    "label": "Crossed LME (participant intercept, task intercept, answer-block intercept)",
                    "group_col": "participant_ID",
                    "variance_components": {"task": "task_key", "answer_block": "response_run_id"},
                },
                {
                    "label": "Crossed LME (task intercept, participant intercept)",
                    "group_col": "task_key",
                    "variance_components": {"participant": "participant_ID"},
                },
            ],
        )
        if result["method"] == "No converged mixed model":
            stat, p_val = stats.mannwhitneyu(phoenix_vals, hcp_vals, alternative="two-sided")
            coef, ci_lo, ci_hi = diff, diff, diff
            method = "Mann-Whitney fallback"
            p_val  = float(p_val)
        else:
            coef   = float(result["coefficient"])
            ci_lo  = float(result["ci_lower"])
            ci_hi  = float(result["ci_upper"])
            p_val  = float(result["p_value"])
            method = result["method"]

        study_tosts[study_id]   = tost_test(phoenix_vals, hcp_vals, delta=config.tost_delta)
        study_effects[study_id] = {
            "coef": coef, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "p_value": p_val, "method": method,
        }
        raw_study_p_values.append(p_val)

    corrected_study_p_values = _holm_correct(raw_study_p_values)
    report_lines.append("")
    report_lines.append("--- Study-specific follow-up models ---")
    report_lines.append(
        "Secondary study-level contrasts are Holm-adjusted across studies."
    )
    for study_id, corrected_p in zip(studies, corrected_study_p_values):
        study_effects[study_id]["p_value_holm"] = corrected_p
        tost_r = study_tosts[study_id]
        equiv_str = (
            f"EQUIVALENT (p_TOST={tost_r['p_tost']:.4f})"
            if tost_r["equivalent"]
            else f"not equivalent (p_TOST={tost_r['p_tost']:.4f})"
        )
        report_lines.extend(
            [
                "",
                f"Study: {study_id}",
                f"  HCP mean={np.mean(work.loc[(work['study_id'].astype(str) == study_id) & (work['reasoner_group'].astype(str) == 'hcp'), 'normalized_score'].to_numpy()):.4f}, "
                f"PHOENIX mean={np.mean(work.loc[(work['study_id'].astype(str) == study_id) & (work['reasoner_group'].astype(str) == 'phoenix'), 'normalized_score'].to_numpy()):.4f}",
                f"  PHOENIX - HCP={study_effects[study_id]['coef']:.4f}",
                f"  95% CI: [{study_effects[study_id]['ci_lo']:.4f}, {study_effects[study_id]['ci_hi']:.4f}]",
                f"  p-value (raw)={study_effects[study_id]['p_value']:.4f}, "
                f"p-value (Holm)={corrected_p:.4f} ({p_to_stars(corrected_p)})",
                f"  TOST (δ=±{config.tost_delta}): {equiv_str}",
                f"  Method={study_effects[study_id]['method']}",
            ]
        )

    report_path = paths["report_dir"] / config.report_name
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    # ────────────────────────────────────────────────────────────────────────
    # Figure 1 — Raincloud per study (normalised score)
    # ────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        1, len(studies),
        figsize=(max(7, 4.4 * len(studies)), 6.0),
        sharey=True,
    )
    if len(studies) == 1:
        axes = [axes]

    for ax, study_id in zip(axes, studies):
        sub = work.loc[work["study_id"].astype(str) == study_id].copy()
        hcp_vals     = sub.loc[sub["reasoner_group"].astype(str) == "hcp",     "normalized_score"].to_numpy()
        phoenix_vals = sub.loc[sub["reasoner_group"].astype(str) == "phoenix", "normalized_score"].to_numpy()
        raincloud_plot(
            ax,
            data_dict={"HCP": hcp_vals, "PHOENIX": phoenix_vals},
            title=_display_label(study_id),
            ylabel="Normalized score (0–1)",
            colors=[PALETTE["secondary"], PALETTE["primary"]],
            adj_p=study_effects[study_id].get("p_value_holm"),
            ylim=(-0.08, 1.08),
            show_tost=study_tosts[study_id],
        )

    fig.suptitle(config.title, fontsize=13, y=1.02)
    plt.tight_layout()
    save_figure(fig, paths["visuals_dir"] / f"{config.study_slug}_reasoner_raincloud.png")

    # ────────────────────────────────────────────────────────────────────────
    # Figure 2 — TOST equivalence panel per study
    # ────────────────────────────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, max(3.5, 0.8 * len(studies) + 1.2)))
    tost_panel(
        ax2,
        dimensions=[_display_label(s) for s in studies],
        tost_results=[study_tosts[s] for s in studies],
        title=f"Holistic Equivalence (TOST, δ = ±{config.tost_delta}): per study",
    )
    plt.tight_layout()
    save_figure(fig2, paths["visuals_dir"] / f"{config.study_slug}_tost_equivalence.png")

    # ────────────────────────────────────────────────────────────────────────
    # Figure 3 — Dimension × study delta heatmap
    # ────────────────────────────────────────────────────────────────────────
    delta_table = (
        work.groupby(["study_id", "dimension", "reasoner_group"], as_index=False)["normalized_score"]
        .mean()
        .pivot_table(
            index=["study_id", "dimension"],
            columns="reasoner_group",
            values="normalized_score",
        )
        .reset_index()
    )
    if {"hcp", "phoenix"}.issubset(delta_table.columns):
        delta_table["phoenix_minus_hcp"] = delta_table["phoenix"] - delta_table["hcp"]
        heatmap_table = delta_table.pivot(
            index="dimension", columns="study_id", values="phoenix_minus_hcp"
        )
        fig3, ax3 = plt.subplots(
            figsize=(
                max(7, 1.8 * len(heatmap_table.columns)),
                max(4, 0.6 * len(heatmap_table.index) + 1.5),
            )
        )
        im = ax3.imshow(heatmap_table.values, cmap="RdBu_r", aspect="auto",
                        vmin=-0.3, vmax=0.3)
        ax3.set_xticks(np.arange(len(heatmap_table.columns)))
        ax3.set_xticklabels(
            [_display_label(c) for c in heatmap_table.columns],
            rotation=20, ha="right",
        )
        ax3.set_yticks(np.arange(len(heatmap_table.index)))
        ax3.set_yticklabels([_display_label(r) for r in heatmap_table.index])
        ax3.set_title("PHOENIX − HCP normalized score by study and dimension")
        for i in range(heatmap_table.shape[0]):
            for j in range(heatmap_table.shape[1]):
                ax3.text(
                    j, i, f"{heatmap_table.iloc[i, j]:.2f}",
                    ha="center", va="center", fontsize=8,
                )
        plt.colorbar(im, ax=ax3, label="Δ (PHOENIX − HCP)")
        plt.tight_layout()
        save_figure(fig3, paths["visuals_dir"] / f"{config.study_slug}_dimension_heatmap.png")

    # ────────────────────────────────────────────────────────────────────────
    # Figure 4 — Forest plot with TOST badges per study
    # ────────────────────────────────────────────────────────────────────────
    fig4, ax4 = plt.subplots(figsize=(9, max(4, 0.85 * len(studies) + 1.2)))
    forest_plot(
        ax4,
        dimensions=[_display_label(s) for s in studies],
        effects=[study_effects[s]["coef"] for s in studies],
        ci_lowers=[study_effects[s]["ci_lo"] for s in studies],
        ci_uppers=[study_effects[s]["ci_hi"] for s in studies],
        title="Holistic PHOENIX − HCP effect by study (normalised score)",
        xlabel="Normalized score difference",
        ref_line=0.0,
        p_values=[study_effects[s]["p_value_holm"] for s in studies],
        tost_results=[study_tosts[s] for s in studies],
    )
    plt.tight_layout()
    save_figure(fig4, paths["visuals_dir"] / f"{config.study_slug}_study_forest.png")
