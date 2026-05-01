"""
Cross-part synthesis for absolute per-output quality scores (design).

Statistical model
-----------------
For each part and globally, fits:

    quality_score ~ entity_ec + (1 | case_id) + (1 | judge_run)

where entity_ec is effect-coded: PHOENIX = +0.5, HCP = −0.5.

The coefficient on entity_ec estimates the PHOENIX − HCP quality gap on
the 1–5 Likert scale; positive values indicate PHOENIX outperforms HCP.
Uncertainty is quantified with 95% CIs; multiplicity correction uses the
Holm–Bonferroni procedure across parts.

TOST equivalence testing on difference scores (phoenix − hcp per cell)
uses delta = ±0.3 quality points (7.5% of the 1–5 scale range).
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
    forest_plot,
    holm_correct,
    p_to_stars,
    publication_heatmap,
    raincloud_plot,
    save_figure,
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
    quality_min: float = 1.0
    quality_max: float = 5.0
    tost_delta: float = 0.3       # ±0.3 pts on the 1–5 scale
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
            # Legacy signed score: map to absolute via  abs_score = round(3 + (s/9)*2)
            df["quality_score"] = df["score"].astype(float).apply(
                lambda s: max(1, min(5, round(3.0 + (s / 9.0) * 2.0)))
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

    # ── Per-part models ───────────────────────────────────────────────────────
    part_effects: Dict[str, Dict[str, Any]] = {}
    part_tosts:   Dict[str, Dict[str, Any]] = {}
    raw_p: List[float] = []
    for part in parts_present:
        sub = work.loc[work["part"].astype(str) == part].copy()
        part_effects[part] = _fit_entity_lmm(sub)
        part_tosts[part]   = _per_part_tost(sub, config.tost_delta)
        raw_p.append(float(part_effects[part]["p_value"]))

    holm_ps = holm_correct(raw_p)
    for part, padj in zip(parts_present, holm_ps):
        part_effects[part]["p_value_holm"] = float(padj)

    # ── Text report ──────────────────────────────────────────────────────────
    lines: List[str] = [
        "=" * 72,
        config.title,
        "Positive effect = PHOENIX outperforms HCP on the 1–5 quality scale.",
        "=" * 72,
        "",
        f"Scale: 1 (Poor) – 5 (Excellent); TOST delta = ±{config.tost_delta} pts.",
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
        title="PHOENIX − HCP quality effect per part\n(entity_ec coefficient, 1–5 scale)",
        xlabel="Mean quality difference (PHOENIX − HCP, scale 1–5)",
        ref_line=0.0,
        p_values=[part_effects[p]["p_value_holm"] for p in parts_present],
        tost_results=[part_tosts[p] for p in parts_present],
    )
    ax1.set_xlim(-1.5, 1.5)
    plt.tight_layout()
    save_figure(fig1, paths["visuals_dir"] / "synthesis_part_forest.png")

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
            title=_display_part(part),
            ylabel="Quality score (1–5)",
            colors=[COLOR_PHOENIX, COLOR_HCP],
            adj_p=holm_p,
            ylim=(0.7, 5.3),
            show_tost=part_tosts[part],
        )
        ax.axhline(3.0, color=PALETTE["ref_line"], linestyle="--",
                   linewidth=0.8, alpha=0.5)
    fig2.suptitle(config.title, fontsize=13, y=1.01, fontweight="bold")
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
    wide["gap"] = wide["q_ph"].fillna(3.0) - wide["q_hcp"].fillna(3.0)

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
    im = ax3.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-1.0, vmax=1.0)
    ax3.set_xticks(np.arange(len(all_dims)))
    ax3.set_xticklabels(
        [_display_dim(d) for d in all_dims],
        rotation=38, ha="right", fontsize=8,
    )
    ax3.set_yticks(np.arange(len(parts_present)))
    ax3.set_yticklabels([_display_part(p) for p in parts_present], fontsize=9)
    ax3.set_title(
        "PHOENIX − HCP quality gap per dimension\n"
        "(blue = PHOENIX higher, red = HCP higher)",
        fontsize=11, fontweight="bold",
    )
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isnan(v):
                continue
            ax3.text(
                j, i, f"{v:+.2f}",
                ha="center", va="center", fontsize=7,
                color="white" if abs(v) > 0.55 else "black",
            )
    plt.colorbar(im, ax=ax3, label="Mean quality gap (PHOENIX − HCP, 1–5 scale)")
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
        title=f"Synthesis TOST equivalence (δ = ±{config.tost_delta} pts)",
    )
    plt.tight_layout()
    save_figure(fig4, paths["visuals_dir"] / "synthesis_tost.png")

    return {
        "global": global_result,
        "global_tost": global_tost,
        "part_effects": part_effects,
        "part_tosts": part_tosts,
    }
