"""
Supplementary stability and sensitivity analyses for the LLM-as-judge pipeline.

These analyses are not additional hypothesis tests. They quantify whether the
three repeated judge runs are stable enough to support the primary mixed-model
comparison and whether conclusions are sensitive to judge confidence weighting.
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

from .shared.shared_stats import PALETTE, apply_rcparams, save_figure
from .shared.survey_paths import ensure_study_dirs, judgments_csv


@dataclass(frozen=True)
class SupplementaryStudyConfig:
    study_slug: str = "supplementary"
    title: str = "Supplementary judge-run stability and sensitivity"
    report_name: str = "supplementary_report.txt"
    judgments_path: Optional[Path] = None
    part_order: Sequence[str] = ("part1", "part2", "part3", "part4", "part5")


def _display_part(part: str) -> str:
    return {
        "part1": "Part 1",
        "part2": "Part 2",
        "part3": "Part 3",
        "part4": "Part 4",
        "part5": "Part 5",
    }.get(str(part), str(part))


def _display_dimension(key: str) -> str:
    return str(key).replace("_", " ").title()


def _load_judgments(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Judgments CSV not found: {path}")
    df = pd.read_csv(path)
    required = {
        "case_id",
        "part",
        "dimension",
        "judge_run",
        "entity",
        "quality_score",
        "confidence",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Judgments CSV missing columns: {sorted(missing)}")
    df = df.copy()
    df["quality_score"] = pd.to_numeric(df["quality_score"], errors="coerce")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(3)
    df["judge_run"] = df["judge_run"].astype(int)
    df = df.loc[df["entity"].isin(["phoenix", "hcp"])].dropna(subset=["quality_score"])
    return df


def _paired_scores(df: pd.DataFrame) -> pd.DataFrame:
    score = df.pivot_table(
        index=["case_id", "part", "dimension", "judge_run"],
        columns="entity",
        values="quality_score",
        aggfunc="mean",
    )
    conf = df.pivot_table(
        index=["case_id", "part", "dimension", "judge_run"],
        columns="entity",
        values="confidence",
        aggfunc="mean",
    )
    paired = score.join(conf, lsuffix="_score", rsuffix="_confidence")
    paired = paired.dropna(subset=["phoenix_score", "hcp_score"], how="any")
    paired = paired.reset_index()
    paired["gap"] = paired["phoenix_score"] - paired["hcp_score"]
    paired["mean_confidence"] = paired[["phoenix_confidence", "hcp_confidence"]].mean(axis=1)
    paired["gap_sign"] = np.sign(paired["gap"]).astype(int)
    return paired


def _pairwise_agreement(values: np.ndarray, tolerance: float) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2:
        return np.nan
    total = 0
    agreed = 0
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            total += 1
            agreed += int(abs(vals[i] - vals[j]) <= tolerance)
    return agreed / total if total else np.nan


def _sign_consistency(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan
    signs = np.sign(vals).astype(int)
    _, counts = np.unique(signs, return_counts=True)
    return float(np.max(counts) / len(signs))


def _compute_stability(df: pd.DataFrame, paired: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    within = (
        df.groupby(["case_id", "part", "dimension", "entity"], observed=True)
        .agg(
            n_runs=("quality_score", "size"),
            mean_score=("quality_score", "mean"),
            sd_score=("quality_score", "std"),
            min_score=("quality_score", "min"),
            max_score=("quality_score", "max"),
            mean_confidence=("confidence", "mean"),
        )
        .reset_index()
    )
    within["sd_score"] = within["sd_score"].fillna(0.0)
    within["range_score"] = within["max_score"] - within["min_score"]

    agreements = []
    for keys, sub in df.groupby(["case_id", "part", "dimension", "entity"], observed=True):
        vals = sub.sort_values("judge_run")["quality_score"].to_numpy()
        agreements.append({
            "case_id": keys[0],
            "part": keys[1],
            "dimension": keys[2],
            "entity": keys[3],
            "exact_pairwise_agreement": _pairwise_agreement(vals, tolerance=0.0),
            "within_one_point_agreement": _pairwise_agreement(vals, tolerance=1.0),
        })
    agreement_df = pd.DataFrame(agreements)
    within = within.merge(
        agreement_df,
        on=["case_id", "part", "dimension", "entity"],
        how="left",
    )

    rating_stability = (
        within.groupby(["part", "dimension", "entity"], observed=True)
        .agg(
            cells=("case_id", "count"),
            mean_rating_sd=("sd_score", "mean"),
            median_rating_sd=("sd_score", "median"),
            mean_rating_range=("range_score", "mean"),
            exact_agreement_rate=("exact_pairwise_agreement", "mean"),
            within_one_point_rate=("within_one_point_agreement", "mean"),
            mean_confidence=("mean_confidence", "mean"),
        )
        .reset_index()
    )

    gap_cells = (
        paired.groupby(["case_id", "part", "dimension"], observed=True)
        .agg(
            n_runs=("gap", "size"),
            mean_gap=("gap", "mean"),
            sd_gap=("gap", "std"),
            min_gap=("gap", "min"),
            max_gap=("gap", "max"),
            mean_confidence=("mean_confidence", "mean"),
        )
        .reset_index()
    )
    gap_cells["sd_gap"] = gap_cells["sd_gap"].fillna(0.0)
    gap_cells["gap_range"] = gap_cells["max_gap"] - gap_cells["min_gap"]

    sign_rows = []
    for keys, sub in paired.groupby(["case_id", "part", "dimension"], observed=True):
        sign_rows.append({
            "case_id": keys[0],
            "part": keys[1],
            "dimension": keys[2],
            "sign_consistency": _sign_consistency(sub["gap"].to_numpy()),
        })
    sign_df = pd.DataFrame(sign_rows)
    gap_cells = gap_cells.merge(sign_df, on=["case_id", "part", "dimension"], how="left")

    gap_stability = (
        gap_cells.groupby(["part", "dimension"], observed=True)
        .agg(
            cells=("case_id", "count"),
            mean_gap=("mean_gap", "mean"),
            mean_gap_sd=("sd_gap", "mean"),
            median_gap_sd=("sd_gap", "median"),
            mean_gap_range=("gap_range", "mean"),
            sign_consistency=("sign_consistency", "mean"),
            mean_confidence=("mean_confidence", "mean"),
        )
        .reset_index()
    )
    return {
        "cell_stability": within,
        "rating_stability": rating_stability,
        "gap_cell_stability": gap_cells,
        "gap_stability": gap_stability,
    }


def _compute_confidence_sensitivity(df: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (part, dimension), sub in df.groupby(["part", "dimension"], observed=True):
        ph = sub.loc[sub["entity"] == "phoenix"]
        hc = sub.loc[sub["entity"] == "hcp"]
        if ph.empty or hc.empty:
            continue
        ph_w = np.average(ph["quality_score"], weights=ph["confidence"])
        hc_w = np.average(hc["quality_score"], weights=hc["confidence"])
        unweighted = float(
            paired.loc[
                (paired["part"] == part) & (paired["dimension"] == dimension),
                "gap",
            ].mean()
        )
        weighted = float(ph_w - hc_w)
        rows.append({
            "part": part,
            "dimension": dimension,
            "unweighted_gap": unweighted,
            "confidence_weighted_gap": weighted,
            "absolute_change": abs(weighted - unweighted),
        })
    return pd.DataFrame(rows)


def _compute_scale_use(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["part", "entity"], observed=True)
        .agg(
            n=("quality_score", "size"),
            mean_score=("quality_score", "mean"),
            sd_score=("quality_score", "std"),
            mean_confidence=("confidence", "mean"),
            floor_rate=("quality_score", lambda s: float((s == 1).mean())),
            low_rate=("quality_score", lambda s: float((s <= 2).mean())),
            acceptable_rate=("quality_score", lambda s: float((s == 3).mean())),
            high_rate=("quality_score", lambda s: float((s >= 4).mean())),
            ceiling_rate=("quality_score", lambda s: float((s == 5).mean())),
        )
        .reset_index()
    )
    return grouped


def _compute_run_level(paired: pd.DataFrame) -> pd.DataFrame:
    return (
        paired.groupby("judge_run", observed=True)
        .agg(
            n=("gap", "size"),
            mean_gap=("gap", "mean"),
            sd_gap=("gap", "std"),
            mean_confidence=("mean_confidence", "mean"),
        )
        .reset_index()
    )


def _plot_stability_dashboard(
    paths: Dict[str, Path],
    rating_stability: pd.DataFrame,
    gap_stability: pd.DataFrame,
    run_level: pd.DataFrame,
) -> Path:
    apply_rcparams()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Figure S1. Judge-run stability across three repeated ratings",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    ax = axes[0, 0]
    part_entity = (
        rating_stability.groupby(["part", "entity"], observed=True)["mean_rating_sd"]
        .mean()
        .reset_index()
    )
    x = np.arange(len(part_entity["part"].unique()))
    width = 0.36
    for idx, entity in enumerate(["phoenix", "hcp"]):
        sub = part_entity.loc[part_entity["entity"] == entity]
        y = [
            float(sub.loc[sub["part"] == p, "mean_rating_sd"].mean())
            for p in sorted(part_entity["part"].unique())
        ]
        ax.bar(
            x + (idx - 0.5) * width,
            y,
            width,
            label=entity.upper() if entity == "hcp" else "PHOENIX",
            color=PALETTE["primary"] if entity == "phoenix" else PALETTE["secondary"],
            alpha=0.85,
        )
    ax.set_title("A. Rating variability by source")
    ax.set_ylabel("Mean within-cell SD")
    ax.set_xticks(x)
    ax.set_xticklabels([_display_part(p) for p in sorted(part_entity["part"].unique())])
    ax.legend(frameon=False)

    ax = axes[0, 1]
    part_gap = gap_stability.groupby("part", observed=True)["mean_gap_sd"].mean()
    ax.bar(
        np.arange(len(part_gap)),
        part_gap.values,
        color=PALETTE["tertiary"],
        alpha=0.85,
    )
    ax.set_title("B. PHOENIX - HCP gap variability")
    ax.set_ylabel("Mean SD of paired gap")
    ax.set_xticks(np.arange(len(part_gap)))
    ax.set_xticklabels([_display_part(p) for p in part_gap.index])

    ax = axes[1, 0]
    part_sign = gap_stability.groupby("part", observed=True)["sign_consistency"].mean()
    ax.bar(
        np.arange(len(part_sign)),
        part_sign.values,
        color=PALETTE["equiv"],
        alpha=0.85,
    )
    ax.set_ylim(0, 1.02)
    ax.set_title("C. Directional consistency")
    ax.set_ylabel("Majority sign proportion")
    ax.set_xticks(np.arange(len(part_sign)))
    ax.set_xticklabels([_display_part(p) for p in part_sign.index])
    ax.axhline(2 / 3, color=PALETTE["ref_line"], linestyle="--", linewidth=1)

    ax = axes[1, 1]
    ax.plot(
        run_level["judge_run"],
        run_level["mean_gap"],
        marker="o",
        color=PALETTE["primary"],
        linewidth=2,
    )
    ax.axhline(0, color=PALETTE["ref_line"], linestyle="--", linewidth=1)
    ax.set_title("D. Global gap by judge run")
    ax.set_xlabel("Judge run")
    ax.set_ylabel("Mean PHOENIX - HCP gap")
    ax.set_xticks(run_level["judge_run"].tolist())

    fig.text(
        0.01,
        0.01,
        "Note. Stability is computed after source unblinding. Lower SD and range values indicate more stable judge ratings. "
        "Directional consistency is the proportion of repeated runs sharing the majority PHOENIX - HCP sign.",
        ha="left",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    out = paths["visuals_dir"] / "supplementary_stability_dashboard.png"
    save_figure(fig, out)
    return out


def _plot_sensitivity_dashboard(
    paths: Dict[str, Path],
    sensitivity: pd.DataFrame,
    scale_use: pd.DataFrame,
) -> Path:
    apply_rcparams()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    fig.suptitle(
        "Figure S2. Confidence weighting and scale-use diagnostics",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    ax = axes[0]
    ax.scatter(
        sensitivity["unweighted_gap"],
        sensitivity["confidence_weighted_gap"],
        s=52,
        color=PALETTE["primary"],
        alpha=0.8,
        edgecolor="white",
        linewidth=0.6,
    )
    lim = max(
        0.4,
        float(np.nanmax(np.abs(sensitivity[["unweighted_gap", "confidence_weighted_gap"]].to_numpy()))) + 0.1,
    )
    ax.plot([-lim, lim], [-lim, lim], color=PALETTE["ref_line"], linestyle="--", linewidth=1)
    ax.axhline(0, color=PALETTE["neutral"], linewidth=0.8)
    ax.axvline(0, color=PALETTE["neutral"], linewidth=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_title("A. Confidence-weighted sensitivity")
    ax.set_xlabel("Unweighted gap")
    ax.set_ylabel("Confidence-weighted gap")

    ax = axes[1]
    parts = sorted(scale_use["part"].unique())
    x = np.arange(len(parts))
    width = 0.36
    for idx, entity in enumerate(["phoenix", "hcp"]):
        sub = scale_use.loc[scale_use["entity"] == entity]
        y = [
            float(sub.loc[sub["part"] == p, "ceiling_rate"].mean())
            for p in parts
        ]
        ax.bar(
            x + (idx - 0.5) * width,
            y,
            width,
            label=entity.upper() if entity == "hcp" else "PHOENIX",
            color=PALETTE["primary"] if entity == "phoenix" else PALETTE["secondary"],
            alpha=0.85,
        )
    ax.set_ylim(0, 1.02)
    ax.set_title("B. Ceiling-rate diagnostics")
    ax.set_ylabel("Proportion scored 5")
    ax.set_xticks(x)
    ax.set_xticklabels([_display_part(p) for p in parts])
    ax.legend(frameon=False)

    fig.text(
        0.01,
        0.01,
        "Note. Panel A tests whether conclusions depend on judge confidence. Panel B reports ceiling compression by source and survey part.",
        ha="left",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    out = paths["visuals_dir"] / "supplementary_sensitivity_dashboard.png"
    save_figure(fig, out)
    return out


def _plot_dimension_heatmap(paths: Dict[str, Path], gap_stability: pd.DataFrame) -> Path:
    apply_rcparams()
    part_order = [p for p in ("part1", "part2", "part3", "part4", "part5") if p in set(gap_stability["part"])]
    dim_order = list(dict.fromkeys(gap_stability["dimension"].tolist()))
    matrix = np.full((len(part_order), len(dim_order)), np.nan)
    for i, part in enumerate(part_order):
        for j, dim in enumerate(dim_order):
            vals = gap_stability.loc[
                (gap_stability["part"] == part) & (gap_stability["dimension"] == dim),
                "mean_gap_sd",
            ]
            if not vals.empty:
                matrix[i, j] = float(vals.iloc[0])

    fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(dim_order)), 5.2))
    vmax = max(0.05, float(np.nanmax(matrix)) if np.isfinite(matrix).any() else 0.5)
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrBr", vmin=0, vmax=vmax)
    ax.set_title("Figure S3. Dimension-level stability of PHOENIX - HCP gaps", fontweight="bold")
    ax.set_yticks(np.arange(len(part_order)))
    ax.set_yticklabels([_display_part(p) for p in part_order])
    ax.set_xticks(np.arange(len(dim_order)))
    ax.set_xticklabels([_display_dimension(d) for d in dim_order], rotation=40, ha="right", fontsize=7)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mean within-cell SD of paired gap")
    fig.text(
        0.01,
        0.01,
        "Note. Each cell aggregates three repeated judge runs within case and dimension. Lower values indicate more stable PHOENIX - HCP estimates.",
        ha="left",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    out = paths["visuals_dir"] / "supplementary_dimension_stability_heatmap.png"
    save_figure(fig, out)
    return out


def _write_report(
    paths: Dict[str, Path],
    config: SupplementaryStudyConfig,
    df: pd.DataFrame,
    paired: pd.DataFrame,
    rating_stability: pd.DataFrame,
    gap_stability: pd.DataFrame,
    sensitivity: pd.DataFrame,
    scale_use: pd.DataFrame,
    run_level: pd.DataFrame,
) -> Path:
    n_runs = int(df["judge_run"].nunique())
    global_rating_sd = float(rating_stability["mean_rating_sd"].mean())
    global_gap_sd = float(gap_stability["mean_gap_sd"].mean())
    global_sign = float(gap_stability["sign_consistency"].mean())
    max_sens = float(sensitivity["absolute_change"].max()) if not sensitivity.empty else 0.0
    ceiling = (
        scale_use.pivot(index="part", columns="entity", values="ceiling_rate")
        .reset_index()
        if not scale_use.empty
        else pd.DataFrame()
    )

    lines: List[str] = [
        config.title,
        "=" * len(config.title),
        "",
        f"Repeated judge runs: {n_runs}",
        f"Rows analysed: {len(df):,}",
        f"Paired PHOENIX-HCP cells: {len(paired):,}",
        "",
        "Primary stability metrics",
        "-------------------------",
        f"Mean within-cell rating SD: {global_rating_sd:.3f} quality points.",
        f"Mean within-cell PHOENIX - HCP gap SD: {global_gap_sd:.3f} quality points.",
        f"Mean directional consistency across repeated runs: {global_sign:.3f}.",
        f"Maximum absolute change after confidence weighting: {max_sens:.3f} quality points.",
        "",
        "Interpretation",
        "--------------",
        "The supplementary analyses quantify run-to-run stability rather than re-test the primary hypothesis. "
        "For three repeated runs, low rating SD, low gap SD, and stable sign direction support the reliability "
        "of the primary PHOENIX versus HCP estimates.",
        "",
        "Figure S1. Judge-run stability across three repeated ratings.",
        "Note. Panels show rating variability, paired-gap variability, sign consistency, and run-level global gap.",
        "",
        "Figure S2. Confidence weighting and scale-use diagnostics.",
        "Note. The sensitivity panel checks whether conclusions change when high-confidence ratings receive greater weight. "
        "The ceiling panel checks whether the 1 to 5 quality scale is compressed at the top end.",
        "",
        "Figure S3. Dimension-level stability of PHOENIX - HCP gaps.",
        "Note. Lower heatmap values indicate more stable repeated-judge estimates.",
        "",
        "Part-level ceiling rates",
        "------------------------",
    ]
    if ceiling.empty:
        lines.append("(no ceiling-rate table available)")
    else:
        for _, row in ceiling.iterrows():
            ph = row.get("phoenix", np.nan)
            hc = row.get("hcp", np.nan)
            lines.append(f"{_display_part(row['part'])}: PHOENIX={ph:.3f}, HCP={hc:.3f}")

    out = paths["report_dir"] / config.report_name
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def run(config: SupplementaryStudyConfig | None = None) -> Dict[str, Any]:
    config = config or SupplementaryStudyConfig()
    apply_rcparams()
    paths = ensure_study_dirs(config.study_slug)
    csv_path = config.judgments_path or judgments_csv()

    df = _load_judgments(csv_path)
    df = df.loc[df["part"].isin(config.part_order)].copy()
    paired = _paired_scores(df)

    stability = _compute_stability(df, paired)
    sensitivity = _compute_confidence_sensitivity(df, paired)
    scale_use = _compute_scale_use(df)
    run_level = _compute_run_level(paired)

    stability["cell_stability"].to_csv(paths["report_dir"] / "cell_stability.csv", index=False)
    stability["rating_stability"].to_csv(paths["report_dir"] / "rating_stability_by_dimension.csv", index=False)
    stability["gap_cell_stability"].to_csv(paths["report_dir"] / "gap_cell_stability.csv", index=False)
    stability["gap_stability"].to_csv(paths["report_dir"] / "gap_stability_by_dimension.csv", index=False)
    sensitivity.to_csv(paths["report_dir"] / "confidence_weighted_sensitivity.csv", index=False)
    scale_use.to_csv(paths["report_dir"] / "scale_use_by_part.csv", index=False)
    run_level.to_csv(paths["report_dir"] / "run_level_stability.csv", index=False)

    figs = [
        _plot_stability_dashboard(
            paths,
            stability["rating_stability"],
            stability["gap_stability"],
            run_level,
        ),
        _plot_sensitivity_dashboard(paths, sensitivity, scale_use),
        _plot_dimension_heatmap(paths, stability["gap_stability"]),
    ]
    report_path = _write_report(
        paths,
        config,
        df,
        paired,
        stability["rating_stability"],
        stability["gap_stability"],
        sensitivity,
        scale_use,
        run_level,
    )
    return {
        "study_slug": config.study_slug,
        "report_path": report_path,
        "figures": figs,
        "rating_stability": stability["rating_stability"],
        "gap_stability": stability["gap_stability"],
        "sensitivity": sensitivity,
        "scale_use": scale_use,
        "run_level": run_level,
    }


if __name__ == "__main__":
    run()
