"""
Generic per-part LMM/TOST comparison runner for the LLM-as-judge design.

Input data (long format) is expected to live at
``data/04_judgments/judgments_long.csv`` with columns:

    case_id, part, dimension, judge_run, source, rating,
    justification, prompt_version, model, timestamp

For each part p, this module fits one Linear Mixed Model per dimension,

    rating ~ source + (1 | case_id) + (1 | judge_run)

with ``source = phoenix`` contrasted against the reference ``hcp``,
runs a TOST equivalence test (delta defaults to 0.5 Likert points,
i.e. half a step on the 1..7 anchored scale), Holm-corrects p-values
within the part, and produces three figures:

    1. Per-dimension raincloud plot (small-multiples).
    2. Forest plot of effect estimates with TOST badges.
    3. Standalone TOST equivalence panel.

It also writes a textual report and a tidy summary CSV.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    bootstrap_cohend_ci,
    cohen_d,
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
class ComparisonStudyConfig:
    """Configuration for a single per-part comparison study."""

    study_slug: str            # e.g. "part1_operationalization"
    part: str                  # e.g. "part1"
    title: str                 # human-readable
    report_name: str           # txt filename in report_dir
    dimension_order: Sequence[str]
    judgments_path: Optional[Path] = None  # override the default
    source_col: str = "source"
    rating_col: str = "rating"
    case_col: str = "case_id"
    run_col: str = "judge_run"
    source_reference: str = "hcp"
    source_test: str = "phoenix"
    source_labels: Dict[str, str] = field(
        default_factory=lambda: {"hcp": "HCP", "phoenix": "PHOENIX"}
    )
    likert_min: int = 1
    likert_max: int = 7
    y_label: str = "Rating (1..7 Likert)"
    tost_delta: float = 0.5    # half a Likert step on 1..7


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _display_label(value: str) -> str:
    return str(value).replace("_", " ").title()


def _structure_note(method: str) -> str:
    method_lower = method.lower()
    if "case intercept" in method_lower and "judge_run" in method_lower:
        return "case-level random intercept with judge_run variance component"
    if "judge_run" in method_lower:
        return "judge_run-level random intercept with case_id variance component"
    if "fallback" in method_lower:
        return "fallback after the richer crossed models failed to converge"
    return "crossed random intercepts for case_id and judge_run"


def _fit_dimension_lmm(sub: pd.DataFrame, *, case_col: str, run_col: str) -> Dict[str, Any]:
    """
    Fit the per-dimension LMM:
        rating ~ source + (1|case_id) + (1|judge_run)
    via :func:`fit_crossed_mixedlm` with three fallback specifications.
    """
    work = sub.copy()
    work["source_bin"] = (work["source"].astype(str) == "phoenix").astype(int)
    formula = "rating ~ source_bin"
    candidates = [
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
    ]
    result = fit_crossed_mixedlm(
        work,
        formula=formula,
        effect_term="source_bin",
        candidates=candidates,
    )
    if result["method"] != "No converged mixed model":
        return result

    err = result.get("error", "No converged mixed model.")
    phoenix = work.loc[work["source"].astype(str) == "phoenix", "rating"].astype(float).to_numpy()
    hcp = work.loc[work["source"].astype(str) == "hcp", "rating"].astype(float).to_numpy()
    try:
        u_stat, p_val = stats.mannwhitneyu(phoenix, hcp, alternative="two-sided")
    except Exception as exc:
        u_stat, p_val = float("nan"), 1.0
        err = f"{err} | mann-whitney fallback failed: {exc}"
    diff = float(np.mean(phoenix) - np.mean(hcp))
    return {
        "method": "Mann-Whitney U fallback",
        "group_col": None,
        "variance_components": {},
        "coefficient": diff,
        "se": 0.0,
        "ci_lower": diff,
        "ci_upper": diff,
        "p_value": float(p_val),
        "converged": False,
        "shapiro_w": float("nan"),
        "shapiro_p": float("nan"),
        "u_statistic": float(u_stat),
        "error": err,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_comparison_study(config: ComparisonStudyConfig) -> Dict[str, Any]:
    """
    Run a per-part comparison study.

    Returns a dict summarising per-dimension results so that the caller may
    aggregate them into the cross-part synthesis without re-reading files.
    """
    apply_rcparams()
    paths = ensure_study_dirs(config.study_slug)
    csv_path = config.judgments_path or judgments_csv()
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Judgments CSV not found: {csv_path}. "
            "Run the judge_runner (real or pseudo) first."
        )
    df_all = pd.read_csv(csv_path)
    df = df_all.loc[df_all["part"].astype(str) == config.part].copy()
    if df.empty:
        raise ValueError(f"No rows for part={config.part!r} in {csv_path}.")

    available = set(df["dimension"].astype(str).unique().tolist())
    dimensions = [d for d in config.dimension_order if d in available]
    if not dimensions:
        raise ValueError(
            f"None of the configured dimensions {list(config.dimension_order)} "
            f"are present in the data for {config.part}."
        )

    report_lines: List[str] = [
        "=" * 72,
        config.title,
        f"Comparison: {config.source_labels[config.source_reference]} "
        f"vs {config.source_labels[config.source_test]} "
        f"(LLM-as-judge, double-blind)",
        "=" * 72,
        "",
        f"Equivalence margin (TOST): +/- {config.tost_delta} Likert points "
        f"on a {config.likert_min}..{config.likert_max} anchored scale.",
        "Multiplicity correction: Holm-Bonferroni across dimensions within this part.",
        f"LMM specification: rating ~ source + (1|{config.case_col}) "
        f"+ (1|{config.run_col}); reference source = {config.source_reference}.",
        "",
    ]

    dim_results: Dict[str, Dict[str, Any]] = {}
    raw_p_values: List[float] = []

    for dim in dimensions:
        sub = df.loc[df["dimension"].astype(str) == dim].copy()
        sub = sub.rename(columns={config.rating_col: "rating",
                                  config.source_col: "source"})
        ref_vals = sub.loc[sub["source"].astype(str) == config.source_reference,
                           "rating"].astype(float).to_numpy()
        test_vals = sub.loc[sub["source"].astype(str) == config.source_test,
                            "rating"].astype(float).to_numpy()
        if len(ref_vals) == 0 or len(test_vals) == 0:
            continue

        model_result = _fit_dimension_lmm(
            sub, case_col=config.case_col, run_col=config.run_col,
        )
        raw_p_values.append(float(model_result["p_value"]))

        tost_result = tost_test(test_vals, ref_vals, delta=config.tost_delta)
        effect_d = cohen_d(test_vals, ref_vals)
        d_ci_lo, d_ci_hi = bootstrap_cohend_ci(test_vals, ref_vals)
        dim_results[dim] = {
            "ref_vals": ref_vals,
            "test_vals": test_vals,
            "result": model_result,
            "tost": tost_result,
            "effect_d": effect_d,
            "effect_d_ci": (d_ci_lo, d_ci_hi),
        }

    corrected = holm_correct(raw_p_values)
    for dim, p_adj in zip(list(dim_results.keys()), corrected):
        dim_results[dim]["result"]["p_value_adj"] = float(p_adj)

    # ── Textual report ──────────────────────────────────────────────────────
    for dim, r in dim_results.items():
        ref_vals, test_vals = r["ref_vals"], r["test_vals"]
        m, t = r["result"], r["tost"]
        equiv = (
            f"EQUIVALENT (p_TOST={t['p_tost']:.4f})"
            if t["equivalent"]
            else f"not equivalent (p_TOST={t['p_tost']:.4f})"
        )
        report_lines.extend([
            f"Dimension: {dim}",
            f"  {config.source_labels[config.source_reference]} mean="
            f"{np.mean(ref_vals):.3f}, SD={np.std(ref_vals, ddof=1):.3f}, "
            f"n={len(ref_vals)}",
            f"  {config.source_labels[config.source_test]} mean="
            f"{np.mean(test_vals):.3f}, SD={np.std(test_vals, ddof=1):.3f}, "
            f"n={len(test_vals)}",
            f"  Method: {m['method']}",
            f"  Random structure: {_structure_note(m['method'])}",
            f"  PHOENIX - HCP coefficient: {m['coefficient']:.4f}",
            f"  95% CI: [{m['ci_lower']:.4f}, {m['ci_upper']:.4f}]",
            f"  p-value (raw): {m['p_value']:.4f}",
            f"  p-value (Holm): {m['p_value_adj']:.4f} "
            f"({p_to_stars(m['p_value_adj'])})",
            f"  Cohen's d: {r['effect_d']:.4f} "
            f"[{r['effect_d_ci'][0]:.4f}, {r['effect_d_ci'][1]:.4f}]",
            f"  TOST (+/- {config.tost_delta}): {equiv}",
            f"    observed diff={t['observed_diff']:.4f}, SE={t['pooled_se']:.4f}",
            "",
        ])

    (paths["report_dir"] / config.report_name).write_text(
        "\n".join(report_lines), encoding="utf-8",
    )

    # ── Tidy summary CSV ────────────────────────────────────────────────────
    summary_rows = []
    for dim, r in dim_results.items():
        m, t = r["result"], r["tost"]
        summary_rows.append({
            "part": config.part,
            "dimension": dim,
            "n_hcp": int(len(r["ref_vals"])),
            "n_phoenix": int(len(r["test_vals"])),
            "mean_hcp": float(np.mean(r["ref_vals"])),
            "mean_phoenix": float(np.mean(r["test_vals"])),
            "coef_phoenix_minus_hcp": float(m["coefficient"]),
            "ci_lower": float(m["ci_lower"]),
            "ci_upper": float(m["ci_upper"]),
            "p_value_raw": float(m["p_value"]),
            "p_value_holm": float(m["p_value_adj"]),
            "cohen_d": float(r["effect_d"]),
            "p_tost": float(t["p_tost"]),
            "tost_equivalent": bool(t["equivalent"]),
            "method": m["method"],
            "converged": bool(m.get("converged", False)),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = paths["report_dir"] / f"{config.study_slug}_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # ── Figure 1 — Raincloud per dimension ──────────────────────────────────
    n_dims = len(dim_results)
    fig, axes = plt.subplots(
        1, n_dims, figsize=(max(5, 4.4 * n_dims), 6.0), sharey=True,
    )
    if n_dims == 1:
        axes = [axes]
    for ax, dim in zip(axes, dim_results.keys()):
        r = dim_results[dim]
        data_dict = {
            config.source_labels[config.source_reference]: r["ref_vals"],
            config.source_labels[config.source_test]: r["test_vals"],
        }
        raincloud_plot(
            ax,
            data_dict=data_dict,
            title=_display_label(dim),
            ylabel=config.y_label,
            colors=[PALETTE["secondary"], PALETTE["primary"]],
            adj_p=r["result"]["p_value_adj"],
            ylim=(config.likert_min - 0.5, config.likert_max + 0.6),
            show_tost=r["tost"],
        )
    fig.suptitle(config.title, fontsize=13, y=1.02)
    plt.tight_layout()
    save_figure(fig, paths["visuals_dir"] / f"{config.study_slug}_ratings_raincloud.png")

    # ── Figure 2 — Forest plot ──────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, max(4, 0.9 * n_dims + 1.5)))
    forest_plot(
        ax2,
        dimensions=[_display_label(d) for d in dim_results.keys()],
        effects=[r["result"]["coefficient"] for r in dim_results.values()],
        ci_lowers=[r["result"]["ci_lower"] for r in dim_results.values()],
        ci_uppers=[r["result"]["ci_upper"] for r in dim_results.values()],
        title=f"{config.title}: PHOENIX - HCP effect by dimension",
        xlabel="Model coefficient (Likert units)",
        ref_line=0.0,
        p_values=[r["result"]["p_value_adj"] for r in dim_results.values()],
        tost_results=[r["tost"] for r in dim_results.values()],
    )
    plt.tight_layout()
    save_figure(fig2, paths["visuals_dir"] / f"{config.study_slug}_effect_forest.png")

    # ── Figure 3 — TOST panel ───────────────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(8, max(3.5, 0.8 * n_dims + 1.2)))
    tost_panel(
        ax3,
        dimensions=[_display_label(d) for d in dim_results.keys()],
        tost_results=[r["tost"] for r in dim_results.values()],
        title=f"{config.title}: TOST equivalence (delta = +/- {config.tost_delta})",
    )
    plt.tight_layout()
    save_figure(fig3, paths["visuals_dir"] / f"{config.study_slug}_tost_equivalence.png")

    return {
        "study_slug": config.study_slug,
        "part": config.part,
        "dim_results": dim_results,
        "summary_df": summary_df,
        "report_path": paths["report_dir"] / config.report_name,
        "summary_path": summary_path,
    }
