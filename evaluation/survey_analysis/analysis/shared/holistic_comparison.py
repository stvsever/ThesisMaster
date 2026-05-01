"""
Cross-part holistic comparison.

Pools per-(part, dimension) ratings from the long-format judgments CSV,
normalises ratings to [0, 1] via ``(rating - 1) / (likert_max - 1)``, fits
a single mixed model

    score_norm ~ source * part + (1|case_id) + (1|judge_run) + (1|dimension)

with three fallback specifications, and produces three figures:

    1. Forest plot of per-part main effects with TOST badges.
    2. Raincloud per part of normalised scores.
    3. Heatmap rows=parts, cols=dimensions, cell = PHOENIX - HCP delta.
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
    tost_test,
)
from .survey_paths import ensure_study_dirs, judgments_csv


@dataclass(frozen=True)
class HolisticStudyConfig:
    study_slug: str = "synthesis"
    title: str = "Cross-part holistic comparison"
    report_name: str = "synthesis_report.txt"
    judgments_path: Optional[Path] = None
    likert_min: int = 1
    likert_max: int = 7
    tost_delta: float = 0.05    # 5% on the 0..1 normalised scale
    part_order: Sequence[str] = (
        "part1", "part2", "part3", "part4", "part5",
    )


def _display_part(p: str) -> str:
    table = {
        "part1": "Part 1: Operationalisation",
        "part2": "Part 2: Initial Model",
        "part3": "Part 3: Treatment Targets",
        "part4": "Part 4: Updated Model",
        "part5": "Part 5: Intervention",
    }
    return table.get(p, p)


def _display_label(value: str) -> str:
    return str(value).replace("_", " ").title()


def run_holistic_synthesis(config: HolisticStudyConfig) -> Dict[str, Any]:
    """
    Build the cross-part holistic comparison artefacts.
    """
    apply_rcparams()
    paths = ensure_study_dirs(config.study_slug)
    csv_path = config.judgments_path or judgments_csv()
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Judgments CSV not found: {csv_path}. Run the judge first."
        )
    df = pd.read_csv(csv_path).copy()

    # Normalise ratings to [0, 1] on the configured Likert scale.
    span = float(config.likert_max - config.likert_min)
    if span <= 0:
        raise ValueError("likert_max must be greater than likert_min")
    df["score_norm"] = (df["rating"].astype(float) - config.likert_min) / span

    parts_present = [p for p in config.part_order if p in set(df["part"].astype(str))]
    if not parts_present:
        raise ValueError("No configured parts present in judgments CSV.")

    work = df.loc[df["part"].astype(str).isin(parts_present)].copy()
    work["source_bin"] = (work["source"].astype(str) == "phoenix").astype(int)

    # ── Unified cross-part model ────────────────────────────────────────────
    formula = "score_norm ~ source_bin * C(part)"
    unified = fit_crossed_mixedlm(
        work,
        formula=formula,
        effect_term="source_bin",
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

    # Global TOST on normalised scores.
    phoenix_all = work.loc[work["source"].astype(str) == "phoenix",
                           "score_norm"].astype(float).to_numpy()
    hcp_all = work.loc[work["source"].astype(str) == "hcp",
                       "score_norm"].astype(float).to_numpy()
    global_tost = tost_test(phoenix_all, hcp_all, delta=config.tost_delta)

    # ── Per-part follow-up models ───────────────────────────────────────────
    part_effects: Dict[str, Dict[str, Any]] = {}
    part_tosts: Dict[str, Dict[str, Any]] = {}
    raw_part_ps: List[float] = []
    for part in parts_present:
        sub = work.loc[work["part"].astype(str) == part].copy()
        phoenix_vals = sub.loc[sub["source"].astype(str) == "phoenix",
                               "score_norm"].astype(float).to_numpy()
        hcp_vals = sub.loc[sub["source"].astype(str) == "hcp",
                           "score_norm"].astype(float).to_numpy()
        part_tosts[part] = tost_test(phoenix_vals, hcp_vals, delta=config.tost_delta)
        result = fit_crossed_mixedlm(
            sub,
            formula="score_norm ~ source_bin",
            effect_term="source_bin",
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
                    "label": "Crossed LMM (judge_run intercept + case VC)",
                    "group_col": "judge_run",
                    "variance_components": {"case": "case_id"},
                },
                {
                    "label": "LMM (case intercept fallback)",
                    "group_col": "case_id",
                    "variance_components": {},
                },
            ],
        )
        if result["method"] == "No converged mixed model":
            diff = float(np.mean(phoenix_vals) - np.mean(hcp_vals))
            result = {
                **result,
                "coefficient": diff,
                "ci_lower": diff,
                "ci_upper": diff,
                "method": "Mean-difference fallback",
            }
        part_effects[part] = result
        raw_part_ps.append(float(result["p_value"]))

    holm_part_ps = holm_correct(raw_part_ps)
    for part, padj in zip(parts_present, holm_part_ps):
        part_effects[part]["p_value_holm"] = float(padj)

    # ── Textual report ──────────────────────────────────────────────────────
    lines: List[str] = [
        "=" * 72,
        config.title,
        "Cross-part synthesis of LLM-as-judge ratings (normalised to [0,1])",
        "=" * 72,
        "",
        f"TOST equivalence margin: +/- {config.tost_delta} on the 0..1 scale",
        "Multiplicity correction (per-part main effects): Holm-Bonferroni",
        "",
        "--- Unified cross-part model ---",
        f"Method: {unified['method']}",
    ]
    if unified["method"] != "No converged mixed model":
        lines.extend([
            f"PHOENIX - HCP main coefficient: {unified['coefficient']:.4f}",
            f"95% CI: [{unified['ci_lower']:.4f}, {unified['ci_upper']:.4f}]",
            f"p-value: {unified['p_value']:.4f} "
            f"({p_to_stars(unified['p_value'])})",
        ])
    else:
        lines.append(f"  Fallback reason: {unified.get('error', '')}")

    equiv = (
        f"EQUIVALENT (p_TOST={global_tost['p_tost']:.4f})"
        if global_tost["equivalent"]
        else f"not equivalent (p_TOST={global_tost['p_tost']:.4f})"
    )
    lines.extend([
        "",
        f"Global TOST (delta=+/- {config.tost_delta}): {equiv}",
        f"  observed diff (PHOENIX - HCP) on [0,1]: "
        f"{global_tost['observed_diff']:.4f}",
        "",
        "--- Per-part main effects (Holm-corrected) ---",
    ])
    for part in parts_present:
        e = part_effects[part]
        t = part_tosts[part]
        equiv_p = (
            f"EQUIV (p_TOST={t['p_tost']:.4f})"
            if t["equivalent"]
            else f"not equiv (p_TOST={t['p_tost']:.4f})"
        )
        lines.extend([
            f"  {_display_part(part)}: coef={e['coefficient']:+.4f}, "
            f"95% CI=[{e['ci_lower']:.4f}, {e['ci_upper']:.4f}], "
            f"p_raw={e['p_value']:.4f}, "
            f"p_holm={e['p_value_holm']:.4f} "
            f"({p_to_stars(e['p_value_holm'])}), TOST: {equiv_p}",
        ])

    (paths["report_dir"] / config.report_name).write_text(
        "\n".join(lines), encoding="utf-8",
    )

    # ── Figure 1 — Per-part forest plot ─────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(9, max(4, 0.85 * len(parts_present) + 1.2)))
    forest_plot(
        ax1,
        dimensions=[_display_part(p) for p in parts_present],
        effects=[part_effects[p]["coefficient"] for p in parts_present],
        ci_lowers=[part_effects[p]["ci_lower"] for p in parts_present],
        ci_uppers=[part_effects[p]["ci_upper"] for p in parts_present],
        title="Holistic PHOENIX - HCP effect by part (normalised score)",
        xlabel="Normalised score difference (0..1 units)",
        ref_line=0.0,
        p_values=[part_effects[p]["p_value_holm"] for p in parts_present],
        tost_results=[part_tosts[p] for p in parts_present],
    )
    plt.tight_layout()
    save_figure(fig1, paths["visuals_dir"] / "synthesis_part_forest.png")

    # ── Figure 2 — Raincloud per part ───────────────────────────────────────
    fig2, axes = plt.subplots(
        1, len(parts_present),
        figsize=(max(7, 4.0 * len(parts_present)), 6.0), sharey=True,
    )
    if len(parts_present) == 1:
        axes = [axes]
    for ax, part in zip(axes, parts_present):
        sub = work.loc[work["part"].astype(str) == part]
        hcp = sub.loc[sub["source"].astype(str) == "hcp", "score_norm"].to_numpy()
        ph = sub.loc[sub["source"].astype(str) == "phoenix", "score_norm"].to_numpy()
        raincloud_plot(
            ax,
            data_dict={"HCP": hcp, "PHOENIX": ph},
            title=_display_part(part),
            ylabel="Normalised score (0..1)",
            colors=[PALETTE["secondary"], PALETTE["primary"]],
            adj_p=part_effects[part]["p_value_holm"],
            ylim=(-0.08, 1.08),
            show_tost=part_tosts[part],
        )
    fig2.suptitle(config.title, fontsize=13, y=1.02)
    plt.tight_layout()
    save_figure(fig2, paths["visuals_dir"] / "synthesis_part_raincloud.png")

    # ── Figure 3 — Part x dimension delta heatmap ──────────────────────────
    delta = (
        work.groupby(["part", "dimension", "source"], as_index=False)["score_norm"]
        .mean()
        .pivot_table(
            index=["part", "dimension"], columns="source", values="score_norm",
        )
        .reset_index()
    )
    if {"hcp", "phoenix"}.issubset(delta.columns):
        delta["delta"] = delta["phoenix"] - delta["hcp"]
        # Build matrix with parts as rows and union of dimensions as columns
        # (some dimensions only exist within one part; cells will be NaN for
        # those that don't apply).
        all_dims = list(dict.fromkeys(delta["dimension"].astype(str).tolist()))
        matrix = np.full((len(parts_present), len(all_dims)), np.nan, dtype=float)
        for i, part in enumerate(parts_present):
            for j, dim in enumerate(all_dims):
                row = delta.loc[(delta["part"] == part) & (delta["dimension"] == dim)]
                if not row.empty:
                    matrix[i, j] = float(row["delta"].iloc[0])

        fig3, ax3 = plt.subplots(
            figsize=(max(8, 0.8 * len(all_dims) + 2),
                     max(4, 0.6 * len(parts_present) + 1.5)),
        )
        cmap = plt.get_cmap("RdBu_r")
        vmax = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 0.3
        vmax = max(vmax, 0.05)
        im = ax3.imshow(matrix, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)
        ax3.set_xticks(np.arange(len(all_dims)))
        ax3.set_xticklabels(
            [_display_label(d) for d in all_dims],
            rotation=35, ha="right", fontsize=8,
        )
        ax3.set_yticks(np.arange(len(parts_present)))
        ax3.set_yticklabels([_display_part(p) for p in parts_present])
        ax3.set_title("Heatmap: PHOENIX - HCP delta on normalised score")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                v = matrix[i, j]
                if np.isnan(v):
                    continue
                ax3.text(
                    j, i, f"{v:+.2f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.6 * vmax else "black",
                )
        plt.colorbar(im, ax=ax3, label="Delta (normalised)")
        plt.tight_layout()
        save_figure(fig3, paths["visuals_dir"] / "synthesis_heatmap.png")

    # ── Figure 4 — Per-part TOST panel ─────────────────────────────────────
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
