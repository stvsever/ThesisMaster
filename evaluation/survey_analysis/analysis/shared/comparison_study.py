"""
Generic per-part analysis for the signed LLM-as-judge design.

The long-format judgments CSV contains one row per
``(case_id, part, dimension, judge_run)`` with:

    score = signed PHOENIX-over-HCP preference on -9..+9

Positive scores favour PHOENIX; negative scores favour the HCP output; zero
means no meaningful difference. For each part and dimension this module fits:

    score ~ 1 + (1 | case_id) + (1 | judge_run)

The intercept is the estimated mean PHOENIX-HCP preference. We also run a
one-sample TOST around zero with a default equivalence margin of +/-1 signed
scale point, Holm-correct p-values within each part, and create a signed
raincloud, forest plot, and TOST panel.
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
    bootstrap_ci_mean,
    cohen_d_one_sample,
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
class ComparisonStudyConfig:
    """Configuration for a single signed per-part comparison study."""

    study_slug: str
    part: str
    title: str
    report_name: str
    dimension_order: Sequence[str]
    judgments_path: Optional[Path] = None
    score_col: str = "score"
    case_col: str = "case_id"
    run_col: str = "judge_run"
    scale_min: int = -9
    scale_max: int = 9
    y_label: str = "Signed preference score (-9 HCP, +9 PHOENIX)"
    tost_delta: float = 1.0


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


def _fit_signed_lmm(sub: pd.DataFrame, *, case_col: str, run_col: str) -> Dict[str, Any]:
    """Fit the signed one-sample LMM and fall back to a one-sample t-test."""
    work = sub.copy()
    formula = "score ~ 1"
    result = fit_crossed_mixedlm(
        work,
        formula=formula,
        effect_term="Intercept",
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
    ci_width = abs(float(result.get("ci_upper", 0.0)) - float(result.get("ci_lower", 0.0)))
    if result["method"] != "No converged mixed model" and np.isfinite(se) and se > 1e-6 and ci_width > 1e-6:
        return result

    vals = work["score"].astype(float).to_numpy()
    mean_score, ci_lo, ci_hi = _mean_ci(vals)
    try:
        t_stat, p_val = stats.ttest_1samp(vals, popmean=0.0)
    except Exception:
        t_stat, p_val = float("nan"), 1.0
    return {
        "method": "One-sample t-test fallback",
        "group_col": None,
        "variance_components": {},
        "coefficient": mean_score,
        "se": float(stats.sem(vals)) if len(vals) > 1 else 0.0,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "p_value": float(p_val) if np.isfinite(p_val) else 1.0,
        "converged": False,
        "shapiro_w": float("nan"),
        "shapiro_p": float("nan"),
        "t_statistic": float(t_stat) if np.isfinite(t_stat) else float("nan"),
        "error": result.get("error", "mixed model produced degenerate standard error"),
    }


def _mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2:
        m = float(np.mean(vals)) if len(vals) else 0.0
        return m, m, m
    mean = float(np.mean(vals))
    lo, hi = bootstrap_ci_mean(vals, n_boot=2000, seed=42)
    return mean, lo, hi


def run_comparison_study(config: ComparisonStudyConfig) -> Dict[str, Any]:
    """Run one per-part signed preference analysis."""
    apply_rcparams()
    paths = ensure_study_dirs(config.study_slug)
    csv_path = config.judgments_path or judgments_csv()
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Judgments CSV not found: {csv_path}. Run the judge first."
        )

    df_all = pd.read_csv(csv_path)
    required = {"case_id", "part", "dimension", "judge_run", config.score_col}
    missing = required.difference(df_all.columns)
    if missing:
        raise ValueError(f"Judgments CSV is missing required columns: {sorted(missing)}")
    df = df_all.loc[df_all["part"].astype(str) == config.part].copy()
    if df.empty:
        raise ValueError(f"No rows for part={config.part!r} in {csv_path}.")
    df = df.rename(columns={config.score_col: "score"})
    df["score"] = df["score"].astype(float)

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
        "Signed LLM-as-judge comparison: positive scores favour PHOENIX",
        "=" * 72,
        "",
        f"Scale: {config.scale_min}..{config.scale_max}; 0 = no meaningful difference.",
        f"Equivalence margin (one-sample TOST): +/- {config.tost_delta} signed score point.",
        "Multiplicity correction: Holm-Bonferroni across dimensions within this part.",
        f"LMM: score ~ 1 + (1|{config.case_col}) + (1|{config.run_col}).",
        "",
    ]

    dim_results: Dict[str, Dict[str, Any]] = {}
    raw_p_values: List[float] = []

    for dim in dimensions:
        sub = df.loc[df["dimension"].astype(str) == dim].copy()
        vals = sub["score"].astype(float).to_numpy()
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        model_result = _fit_signed_lmm(sub, case_col=config.case_col, run_col=config.run_col)
        raw_p_values.append(float(model_result["p_value"]))
        tost_result = tost_test_one_sample(vals, delta=config.tost_delta)
        mean_score, mean_ci_lo, mean_ci_hi = _mean_ci(vals)
        dim_results[dim] = {
            "values": vals,
            "result": model_result,
            "tost": tost_result,
            "mean_score": mean_score,
            "mean_ci": (mean_ci_lo, mean_ci_hi),
            "effect_d": cohen_d_one_sample(vals),
            "pct_phoenix_preferred": float(np.mean(vals > 0)),
            "pct_hcp_preferred": float(np.mean(vals < 0)),
            "pct_tie": float(np.mean(vals == 0)),
        }

    corrected = holm_correct(raw_p_values)
    for dim, p_adj in zip(list(dim_results.keys()), corrected):
        dim_results[dim]["result"]["p_value_adj"] = float(p_adj)

    for dim, r in dim_results.items():
        vals = r["values"]
        m = r["result"]
        t = r["tost"]
        equiv = (
            f"EQUIVALENT (p_TOST={t['p_tost']:.4f})"
            if t["equivalent"]
            else f"not equivalent (p_TOST={t['p_tost']:.4f})"
        )
        report_lines.extend([
            f"Dimension: {dim}",
            f"  Mean signed score={np.mean(vals):+.3f}, SD={np.std(vals, ddof=1):.3f}, "
            f"median={np.median(vals):+.3f}, n={len(vals)}",
            f"  Preference split: PHOENIX>{r['pct_phoenix_preferred']:.1%}, "
            f"HCP>{r['pct_hcp_preferred']:.1%}, tie={r['pct_tie']:.1%}",
            f"  Method: {m['method']}",
            f"  Random structure: {_structure_note(m['method'])}",
            f"  Intercept (PHOENIX - HCP): {m['coefficient']:+.4f}",
            f"  95% CI: [{m['ci_lower']:+.4f}, {m['ci_upper']:+.4f}]",
            f"  p-value (raw): {m['p_value']:.4f}",
            f"  p-value (Holm): {m['p_value_adj']:.4f} "
            f"({p_to_stars(m['p_value_adj'])})",
            f"  One-sample Cohen's d: {r['effect_d']:+.4f}",
            f"  TOST (+/- {config.tost_delta}): {equiv}",
            f"    observed mean={t['observed_diff']:+.4f}, SE={t['pooled_se']:.4f}",
            "",
        ])

    (paths["report_dir"] / config.report_name).write_text(
        "\n".join(report_lines), encoding="utf-8",
    )

    summary_rows = []
    for dim, r in dim_results.items():
        m, t = r["result"], r["tost"]
        vals = r["values"]
        summary_rows.append({
            "part": config.part,
            "dimension": dim,
            "n": int(len(vals)),
            "mean_score": float(np.mean(vals)),
            "median_score": float(np.median(vals)),
            "sd_score": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "pct_phoenix_preferred": r["pct_phoenix_preferred"],
            "pct_hcp_preferred": r["pct_hcp_preferred"],
            "pct_tie": r["pct_tie"],
            "coef_phoenix_minus_hcp": float(m["coefficient"]),
            "ci_lower": float(m["ci_lower"]),
            "ci_upper": float(m["ci_upper"]),
            "p_value_raw": float(m["p_value"]),
            "p_value_holm": float(m["p_value_adj"]),
            "cohen_d_one_sample": float(r["effect_d"]),
            "p_tost": float(t["p_tost"]),
            "tost_equivalent": bool(t["equivalent"]),
            "method": m["method"],
            "converged": bool(m.get("converged", False)),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = paths["report_dir"] / f"{config.study_slug}_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    n_dims = len(dim_results)
    fig, axes = plt.subplots(
        1, n_dims, figsize=(max(5, 3.8 * n_dims), 6.0), sharey=True,
    )
    if n_dims == 1:
        axes = [axes]
    for ax, dim in zip(axes, dim_results.keys()):
        vals = dim_results[dim]["values"]
        raincloud_plot(
            ax,
            data_dict={"PHOENIX-HCP": vals},
            title=_display_label(dim),
            ylabel=config.y_label,
            colors=[PALETTE["primary"]],
            ylim=(config.scale_min - 1, config.scale_max + 1),
        )
        ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.65)
        ax.text(
            0.5, -0.16,
            f"Holm p={dim_results[dim]['result']['p_value_adj']:.3f}; "
            f"TOST p={dim_results[dim]['tost']['p_tost']:.3f}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            color=PALETTE["ref_line"],
        )
    fig.suptitle(config.title, fontsize=13, y=1.02)
    plt.tight_layout()
    save_figure(fig, paths["visuals_dir"] / f"{config.study_slug}_signed_preference_raincloud.png")

    fig2, ax2 = plt.subplots(figsize=(10, max(4, 0.9 * n_dims + 1.5)))
    forest_plot(
        ax2,
        dimensions=[_display_label(d) for d in dim_results.keys()],
        effects=[r["result"]["coefficient"] for r in dim_results.values()],
        ci_lowers=[r["result"]["ci_lower"] for r in dim_results.values()],
        ci_uppers=[r["result"]["ci_upper"] for r in dim_results.values()],
        title=f"{config.title}: PHOENIX - HCP signed preference",
        xlabel="Signed score (negative = HCP preferred, positive = PHOENIX preferred)",
        ref_line=0.0,
        p_values=[r["result"]["p_value_adj"] for r in dim_results.values()],
        tost_results=[r["tost"] for r in dim_results.values()],
    )
    lo = min([r["result"]["ci_lower"] for r in dim_results.values()] + [-config.tost_delta])
    hi = max([r["result"]["ci_upper"] for r in dim_results.values()] + [config.tost_delta])
    pad = max(0.8, 0.10 * (hi - lo))
    ax2.set_xlim(max(config.scale_min, lo - pad), min(config.scale_max, hi + pad))
    plt.tight_layout()
    save_figure(fig2, paths["visuals_dir"] / f"{config.study_slug}_effect_forest.png")

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
