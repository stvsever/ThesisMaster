"""
Cross-part synthesis for signed PHOENIX-vs-HCP preference scores.

The synthesis pools all per-dimension signed judge scores, normalises them to
[-1, +1] by dividing by 9, and fits:

    score_norm ~ 1 + (1|case_id) + (1|judge_run) + (1|dimension)

The intercept tests the average PHOENIX preference across the whole
evaluation. Per-part follow-up models use the same signed one-sample
structure within each part.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

from .shared_stats import (
    PALETTE,
    apply_rcparams,
    fit_crossed_mixedlm,
    forest_plot,
    holm_correct,
    p_to_stars,
    raincloud_plot,
    save_figure,
    tost_panel,
    tost_test_one_sample,
)
from .survey_paths import ensure_study_dirs, judgments_csv


@dataclass(frozen=True)
class HolisticStudyConfig:
    study_slug: str = "synthesis"
    title: str = "Cross-part signed preference synthesis"
    report_name: str = "synthesis_report.txt"
    judgments_path: Optional[Path] = None
    scale_max_abs: float = 9.0
    tost_delta: float = 0.10
    part_order: Sequence[str] = (
        "part1", "part2", "part3", "part4", "part5",
    )


def _display_part(p: str) -> str:
    table = {
        "part1": "Part 1: Symptoms",
        "part2": "Part 2: Treatment Options",
        "part3": "Part 3: Target Ranking",
        "part4": "Part 4: EMA Items",
        "part5": "Part 5: Coaching Message",
    }
    return table.get(p, p)


def _display_label(value: str) -> str:
    return str(value).replace("_", " ").title()


def _fit_signed_model(data: pd.DataFrame, formula: str, effect_term: str) -> Dict[str, Any]:
    result = fit_crossed_mixedlm(
        data,
        formula=formula,
        effect_term=effect_term,
        candidates=[
            {
                "label": "Crossed LMM (case intercept + judge_run + dimension VCs)",
                "group_col": "case_id",
                "variance_components": {
                    "judge_run": "judge_run",
                    "dimension": "dimension",
                },
            },
            {
                "label": "Crossed LMM (judge_run intercept + case + dimension VCs)",
                "group_col": "judge_run",
                "variance_components": {
                    "case": "case_id",
                    "dimension": "dimension",
                },
            },
            {
                "label": "LMM (case intercept fallback)",
                "group_col": "case_id",
                "variance_components": {},
            },
        ],
    )
    se = float(result.get("se", 0.0) or 0.0)
    ci_width = abs(float(result.get("ci_upper", 0.0)) - float(result.get("ci_lower", 0.0)))
    if result["method"] != "No converged mixed model" and np.isfinite(se) and se > 1e-6 and ci_width > 1e-6:
        return result
    vals = data["score_norm"].astype(float).to_numpy()
    mean = float(np.mean(vals)) if len(vals) else 0.0
    se = float(stats.sem(vals)) if len(vals) > 1 else 0.0
    ci = 1.96 * se
    try:
        _t, p_val = stats.ttest_1samp(vals, popmean=0.0)
    except Exception:
        p_val = 1.0
    return {
        **result,
        "method": "One-sample t-test fallback",
        "coefficient": mean,
        "se": se,
        "ci_lower": mean - ci,
        "ci_upper": mean + ci,
        "p_value": float(p_val) if np.isfinite(p_val) else 1.0,
    }


def run_holistic_synthesis(config: HolisticStudyConfig) -> Dict[str, Any]:
    """Build cross-part signed synthesis artefacts."""
    apply_rcparams()
    paths = ensure_study_dirs(config.study_slug)
    csv_path = config.judgments_path or judgments_csv()
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Judgments CSV not found: {csv_path}. Run the judge first."
        )
    df = pd.read_csv(csv_path).copy()
    required = {"case_id", "part", "dimension", "judge_run", "score"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Judgments CSV is missing required columns: {sorted(missing)}")
    df["score"] = df["score"].astype(float)
    df["score_norm"] = df["score"] / float(config.scale_max_abs)

    parts_present = [p for p in config.part_order if p in set(df["part"].astype(str))]
    if not parts_present:
        raise ValueError("No configured parts present in judgments CSV.")
    work = df.loc[df["part"].astype(str).isin(parts_present)].copy()

    unified = _fit_signed_model(work, "score_norm ~ 1", "Intercept")
    global_tost = tost_test_one_sample(
        work["score_norm"].astype(float).to_numpy(),
        delta=config.tost_delta,
    )

    part_effects: Dict[str, Dict[str, Any]] = {}
    part_tosts: Dict[str, Dict[str, Any]] = {}
    raw_part_ps: List[float] = []
    for part in parts_present:
        sub = work.loc[work["part"].astype(str) == part].copy()
        result = _fit_signed_model(sub, "score_norm ~ 1", "Intercept")
        part_effects[part] = result
        part_tosts[part] = tost_test_one_sample(
            sub["score_norm"].astype(float).to_numpy(),
            delta=config.tost_delta,
        )
        raw_part_ps.append(float(result["p_value"]))

    holm_part_ps = holm_correct(raw_part_ps)
    for part, padj in zip(parts_present, holm_part_ps):
        part_effects[part]["p_value_holm"] = float(padj)

    lines: List[str] = [
        "=" * 72,
        config.title,
        "Positive normalised scores favour PHOENIX; negative scores favour HCP.",
        "=" * 72,
        "",
        "Signed scores are normalised by dividing by 9, yielding [-1,+1].",
        f"TOST equivalence margin: +/- {config.tost_delta} on the normalised scale.",
        "Multiplicity correction (per-part follow-ups): Holm-Bonferroni.",
        "",
        "--- Unified grand-mean model ---",
        f"Method: {unified['method']}",
    ]
    if unified["method"] != "No converged mixed model":
        lines.extend([
            f"Global PHOENIX-HCP intercept: {unified['coefficient']:+.4f}",
            f"95% CI: [{unified['ci_lower']:+.4f}, {unified['ci_upper']:+.4f}]",
            f"p-value: {unified['p_value']:.4f} ({p_to_stars(unified['p_value'])})",
        ])
    else:
        lines.append(f"Fallback reason: {unified.get('error', '')}")

    equiv = (
        f"EQUIVALENT (p_TOST={global_tost['p_tost']:.4f})"
        if global_tost["equivalent"]
        else f"not equivalent (p_TOST={global_tost['p_tost']:.4f})"
    )
    lines.extend([
        "",
        f"Global TOST (delta=+/- {config.tost_delta}): {equiv}",
        f"  observed mean signed score: {global_tost['observed_diff']:+.4f}",
        "",
        "--- Per-part signed effects (Holm-corrected) ---",
    ])
    for part in parts_present:
        e = part_effects[part]
        t = part_tosts[part]
        equiv_p = (
            f"EQUIV (p_TOST={t['p_tost']:.4f})"
            if t["equivalent"]
            else f"not equiv (p_TOST={t['p_tost']:.4f})"
        )
        lines.append(
            f"  {_display_part(part)}: coef={e['coefficient']:+.4f}, "
            f"95% CI=[{e['ci_lower']:+.4f}, {e['ci_upper']:+.4f}], "
            f"p_raw={e['p_value']:.4f}, p_holm={e['p_value_holm']:.4f} "
            f"({p_to_stars(e['p_value_holm'])}), TOST: {equiv_p}"
        )

    (paths["report_dir"] / config.report_name).write_text(
        "\n".join(lines), encoding="utf-8",
    )

    fig1, ax1 = plt.subplots(figsize=(9, max(4, 0.85 * len(parts_present) + 1.2)))
    forest_plot(
        ax1,
        dimensions=[_display_part(p) for p in parts_present],
        effects=[part_effects[p]["coefficient"] for p in parts_present],
        ci_lowers=[part_effects[p]["ci_lower"] for p in parts_present],
        ci_uppers=[part_effects[p]["ci_upper"] for p in parts_present],
        title="Signed PHOENIX - HCP preference by part",
        xlabel="Normalised signed score (-1 HCP, +1 PHOENIX)",
        ref_line=0.0,
        p_values=[part_effects[p]["p_value_holm"] for p in parts_present],
        tost_results=[part_tosts[p] for p in parts_present],
    )
    ax1.set_xlim(-0.5, 0.5)
    plt.tight_layout()
    save_figure(fig1, paths["visuals_dir"] / "synthesis_part_forest.png")

    fig2, axes = plt.subplots(
        1, len(parts_present),
        figsize=(max(7, 3.8 * len(parts_present)), 6.0),
        sharey=True,
    )
    if len(parts_present) == 1:
        axes = [axes]
    for ax, part in zip(axes, parts_present):
        vals = work.loc[work["part"].astype(str) == part, "score_norm"].to_numpy()
        raincloud_plot(
            ax,
            data_dict={"PHOENIX-HCP": vals},
            title=_display_part(part),
            ylabel="Normalised signed score",
            colors=[PALETTE["primary"]],
            ylim=(-1.08, 1.08),
        )
        ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.65)
    fig2.suptitle(config.title, fontsize=13, y=1.02)
    plt.tight_layout()
    save_figure(fig2, paths["visuals_dir"] / "synthesis_part_signed_raincloud.png")

    delta = (
        work.groupby(["part", "dimension"], as_index=False)["score_norm"]
        .mean()
    )
    all_dims = list(dict.fromkeys(delta["dimension"].astype(str).tolist()))
    matrix = np.full((len(parts_present), len(all_dims)), np.nan, dtype=float)
    for i, part in enumerate(parts_present):
        for j, dim in enumerate(all_dims):
            row = delta.loc[(delta["part"] == part) & (delta["dimension"] == dim)]
            if not row.empty:
                matrix[i, j] = float(row["score_norm"].iloc[0])

    fig3, ax3 = plt.subplots(
        figsize=(max(8, 0.8 * len(all_dims) + 2),
                 max(4, 0.6 * len(parts_present) + 1.5)),
    )
    cmap = plt.get_cmap("RdBu_r")
    im = ax3.imshow(matrix, cmap=cmap, aspect="auto", vmin=-1.0, vmax=1.0)
    ax3.set_xticks(np.arange(len(all_dims)))
    ax3.set_xticklabels(
        [_display_label(d) for d in all_dims],
        rotation=35,
        ha="right",
        fontsize=8,
    )
    ax3.set_yticks(np.arange(len(parts_present)))
    ax3.set_yticklabels([_display_part(p) for p in parts_present])
    ax3.set_title("Heatmap: mean signed preference by part and dimension")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isnan(v):
                continue
            ax3.text(
                j,
                i,
                f"{v:+.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(v) > 0.55 else "black",
            )
    plt.colorbar(im, ax=ax3, label="Mean signed score (-1 HCP, +1 PHOENIX)")
    plt.tight_layout()
    save_figure(fig3, paths["visuals_dir"] / "synthesis_heatmap.png")

    fig4, ax4 = plt.subplots(figsize=(8, max(3.5, 0.8 * len(parts_present) + 1.2)))
    tost_panel(
        ax4,
        dimensions=[_display_part(p) for p in parts_present],
        tost_results=[part_tosts[p] for p in parts_present],
        title=f"Synthesis TOST (delta=+/- {config.tost_delta})",
    )
    plt.tight_layout()
    save_figure(fig4, paths["visuals_dir"] / "synthesis_tost.png")

    return {
        "unified": unified,
        "global_tost": global_tost,
        "part_effects": part_effects,
        "part_tosts": part_tosts,
    }
