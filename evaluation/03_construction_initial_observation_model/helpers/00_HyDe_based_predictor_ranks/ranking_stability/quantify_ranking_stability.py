#!/usr/bin/env python3
"""
quantify_ranking_stability.py

Goal
----
Quantify ranking stability across multiple HyDe fused-ranking runs for each pseudoprofile,
focused on NUMERICAL RELEVANCE SCORES (global_XXX_score) for the top-K=200 features, and
visualize the top-N=100 most stable / most frequent features *as score matrices*.

Key changes vs earlier draft (per your request)
-----------------------------------------------
- Score semantics: treated as relevance of predictor-solution in recommendation context.
- No PDF outputs (PNG only).
- No run-by-run similarity matrices (ignored entirely).
- Visualization:
    1) "Top 100 numerical scores" plot: run-by-feature heatmap of scores.
    2) "3D version": run-by-feature 3D surface (scores).
- Domain-splitting:
    Features have a prefix: BIO / ...   PSYCHO / ...   SOCIAL / ...
    The script creates a clean subdirectory structure and runs the same analysis
    separately for BIO, PSYCHO, SOCIAL, plus ALL (combined).

Output Structure (under base_output_dir)
---------------------------------------
ranking_stability_v2/
  summaries/
    overall_summary.csv
    per_pseudoprofile/
      <pseudoprofile_id>/
        ALL__top100_features.csv
        BIO__top100_features.csv
        PSYCHO__top100_features.csv
        SOCIAL__top100_features.csv
        summary.txt
  plots/
    ALL/
      heatmaps/<pseudoprofile_id>__ALL__scores_top100.png
      surfaces3d/<pseudoprofile_id>__ALL__surface_top100.png
    BIO/
      heatmaps/...
      surfaces3d/...
    PSYCHO/
      heatmaps/...
      surfaces3d/...
    SOCIAL/
      heatmaps/...
      surfaces3d/...

Selection logic for top-100 features
------------------------------------
For each pseudoprofile and each domain subset:
- Build union of features across runs (within topK).
- Compute presence frequency (fraction of runs where present in topK).
- Compute mean_rank_present (lower is better) and std_rank_present.
- Select top N=100 by:
      (presence_frequency DESC, mean_rank_present ASC, std_rank_present ASC, mean_score_present DESC)
This yields frequently occurring + consistently high-ranked features.
(You can adjust sorting via CLI).

3D Plot
-------
Uses matplotlib's mplot3d to render a surface:
- x-axis: feature index (0..N-1)
- y-axis: run index (0..R-1)
- z-axis: score

Note: for many runs (large R), the 3D surface can be visually dense.
Heatmap remains the primary “paper-ready” figure.

Author: (generated)
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 3D support
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)


# -----------------------------
# Utilities
# -----------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sanitize_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")


def wrap_label(s: str, width: int = 42) -> str:
    s = str(s)
    if len(s) <= width:
        return s
    return "\n".join(textwrap.wrap(s, width=width, break_long_words=False, replace_whitespace=False))


def domain_of_feature(path_full: str) -> str:
    """
    Extract domain prefix from 'BIO / ...' or 'PSYCHO / ...' or 'SOCIAL / ...'
    Falls back to 'OTHER' if no match.
    """
    if not path_full:
        return "OTHER"
    s = str(path_full).strip()
    # common patterns: "BIO / ..." (space slash space)
    m = re.match(r"^(BIO|PSYCHO|SOCIAL)\s*/", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # sometimes could be "BIO/" etc
    m2 = re.match(r"^(BIO|PSYCHO|SOCIAL)\s*/?", s, flags=re.IGNORECASE)
    if m2:
        return m2.group(1).upper()
    return "OTHER"


def choose_figsize(nrows: int, ncols: int,
                   base: Tuple[float, float] = (7.0, 4.0),
                   cell: Tuple[float, float] = (0.20, 0.28),
                   max_size: Tuple[float, float] = (24.0, 16.0)) -> Tuple[float, float]:
    w = min(max_size[0], base[0] + cell[0] * ncols)
    h = min(max_size[1], base[1] + cell[1] * nrows)
    return (w, h)


# -----------------------------
# Data loading
# -----------------------------

GLOBAL_PATH_FULL_RE = re.compile(r"^global_(\d{3})_path_full$")
GLOBAL_SCORE_RE = re.compile(r"^global_(\d{3})_score$")


@dataclass
class RunProfileRanking:
    paths: List[str]
    scores: List[float]
    ranks: List[int]


def infer_global_columns(columns: List[str], topk: int) -> Tuple[List[int], List[str], List[str]]:
    idxs = []
    for c in columns:
        m = GLOBAL_PATH_FULL_RE.match(c)
        if m:
            i = int(m.group(1))
            if 1 <= i <= topk:
                idxs.append(i)
    idxs = sorted(set(idxs))
    path_cols = [f"global_{i:03d}_path_full" for i in idxs]
    score_cols = [f"global_{i:03d}_score" for i in idxs]
    return idxs, path_cols, score_cols


def load_dense_profiles_csv(csv_path: Path, topk: int) -> pd.DataFrame:
    header = pd.read_csv(csv_path, nrows=0)
    cols = list(header.columns)

    idxs, path_cols, score_cols = infer_global_columns(cols, topk)

    usecols = ["run_id", "pseudoprofile_id"] + path_cols
    usecols += [c for c in score_cols if c in cols]

    dtypes = {c: "string" for c in ["run_id", "pseudoprofile_id"] + path_cols}
    for c in score_cols:
        if c in cols:
            dtypes[c] = "float32"

    df = pd.read_csv(
        csv_path,
        usecols=usecols,
        dtype=dtypes,
        low_memory=False,
        on_bad_lines="warn",
    )

    for c in score_cols:
        if c not in df.columns:
            df[c] = np.nan

    for c in path_cols + ["run_id", "pseudoprofile_id"]:
        df[c] = df[c].astype("string").str.strip()

    return df


def extract_rankings_per_run(df: pd.DataFrame, topk: int) -> Dict[str, RunProfileRanking]:
    cols = list(df.columns)
    idxs, path_cols, score_cols = infer_global_columns(cols, topk)

    path_mat = df[path_cols].to_numpy(dtype=object)
    score_mat = df[score_cols].to_numpy(dtype=np.float32)
    pseudo_ids = df["pseudoprofile_id"].to_numpy(dtype=object)
    ranks = np.array(idxs, dtype=int)

    out: Dict[str, RunProfileRanking] = {}
    for i in range(df.shape[0]):
        pid = str(pseudo_ids[i])
        paths = path_mat[i]
        scores = score_mat[i]

        valid_paths: List[str] = []
        valid_scores: List[float] = []
        valid_ranks: List[int] = []

        for p, sc, rk in zip(paths, scores, ranks):
            if p is None:
                continue
            p_str = str(p).strip()
            if p_str == "" or p_str.lower() == "nan":
                continue
            valid_paths.append(p_str)
            if sc is None or (isinstance(sc, float) and math.isnan(sc)):
                valid_scores.append(float("nan"))
            else:
                valid_scores.append(float(sc))
            valid_ranks.append(int(rk))

        # Deduplicate by first occurrence (best rank)
        if valid_paths:
            seen = set()
            d_paths, d_scores, d_ranks = [], [], []
            for p_str, sc, rk in zip(valid_paths, valid_scores, valid_ranks):
                if p_str in seen:
                    continue
                seen.add(p_str)
                d_paths.append(p_str)
                d_scores.append(sc)
                d_ranks.append(rk)
            out[pid] = RunProfileRanking(d_paths, d_scores, d_ranks)
        else:
            out[pid] = RunProfileRanking([], [], [])

    return out


def discover_runs(runs_dir: Path) -> List[Path]:
    run_dirs = []
    if not runs_dir.exists():
        return run_dirs
    for d in sorted(runs_dir.iterdir()):
        if d.is_dir() and (d / "dense_profiles.csv").exists():
            run_dirs.append(d)
    return run_dirs


# -----------------------------
# Matrix building + feature stats
# -----------------------------

def build_matrices_for_features(
    features: List[str],
    run_ids: List[str],
    per_run: Dict[str, RunProfileRanking],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      B: presence (n_runs x n_features) 0/1
      S: scores   (n_runs x n_features) 0 if absent, else score
      R: ranks    (n_runs x n_features) NaN if absent, else rank (1..topk)
    """
    n_runs = len(run_ids)
    n_feat = len(features)
    feat_to_idx = {f: i for i, f in enumerate(features)}

    B = np.zeros((n_runs, n_feat), dtype=np.uint8)
    S = np.zeros((n_runs, n_feat), dtype=np.float32)
    R = np.full((n_runs, n_feat), np.nan, dtype=np.float32)

    for ri, run_id in enumerate(run_ids):
        rp = per_run[run_id]
        if not rp.paths:
            continue
        for p, sc, rk in zip(rp.paths, rp.scores, rp.ranks):
            j = feat_to_idx.get(p)
            if j is None:
                continue
            B[ri, j] = 1
            if sc is not None and not (isinstance(sc, float) and math.isnan(sc)):
                S[ri, j] = float(sc)
            R[ri, j] = float(rk)

    return B, S, R


def per_feature_table(features: List[str], B: np.ndarray, S: np.ndarray, R: np.ndarray) -> pd.DataFrame:
    present = (B == 1)

    S_mask = np.where(present, S, np.nan)
    R_mask = np.where(present, R, np.nan)

    freq = present.mean(axis=0).astype(float)
    n_present = present.sum(axis=0).astype(int)

    mean_rank = np.nanmean(R_mask, axis=0)
    std_rank = np.nanstd(R_mask, axis=0)

    mean_score = np.nanmean(S_mask, axis=0)
    std_score = np.nanstd(S_mask, axis=0)

    df = pd.DataFrame({
        "feature_path_full": features,
        "domain": [domain_of_feature(f) for f in features],
        "n_present_runs": n_present,
        "presence_frequency": freq,
        "mean_rank_present": mean_rank,
        "std_rank_present": std_rank,
        "mean_score_present": mean_score,
        "std_score_present": std_score,
    })

    # stable + high
    df = df.sort_values(
        by=["presence_frequency", "mean_rank_present", "std_rank_present", "mean_score_present"],
        ascending=[False, True, True, False],
        kind="mergesort",
    ).reset_index(drop=True)
    return df


def select_top_features(df: pd.DataFrame, topn: int) -> pd.DataFrame:
    # already sorted in a sensible stability order
    return df.head(topn).copy()


# -----------------------------
# Plotting
# -----------------------------

def plot_scores_heatmap_png(
    S_sel: np.ndarray,
    run_labels: List[str],
    feat_labels: List[str],
    title: str,
    out_png: Path,
    dpi: int = 300,
) -> None:
    plt.close("all")
    n_runs, n_feat = S_sel.shape
    figsize = choose_figsize(n_runs, n_feat, base=(7.0, 4.0), cell=(0.20, 0.28), max_size=(24.0, 16.0))
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Use robust scaling for readability (avoid a few outliers washing everything out)
    finite = np.isfinite(S_sel)
    if np.any(finite):
        vmin = float(np.nanpercentile(S_sel[finite], 2))
        vmax = float(np.nanpercentile(S_sel[finite], 98))
        if vmin == vmax:
            vmin = float(np.nanmin(S_sel[finite]))
            vmax = float(np.nanmax(S_sel[finite]))
    else:
        vmin, vmax = 0.0, 1.0

    im = ax.imshow(S_sel, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)

    ax.set_title(title, pad=12)

    ax.set_yticks(np.arange(len(run_labels)))
    ax.set_yticklabels([wrap_label(r, 28) for r in run_labels], fontsize=8)

    ax.set_xticks(np.arange(len(feat_labels)))
    ax.set_xticklabels([wrap_label(f, 44) for f in feat_labels], fontsize=7)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", va="top")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Relevance score", fontsize=9)

    ax.set_xlabel("")
    ax.set_ylabel("Run")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_scores_surface3d_png(
    S_sel: np.ndarray,
    title: str,
    out_png: Path,
    dpi: int = 300,
) -> None:
    plt.close("all")
    n_runs, n_feat = S_sel.shape

    X = np.arange(n_feat)
    Y = np.arange(n_runs)
    XX, YY = np.meshgrid(X, Y)
    ZZ = S_sel.astype(float)

    finite = np.isfinite(ZZ)
    zmin = float(np.nanpercentile(ZZ[finite], 2))
    zmax = float(np.nanpercentile(ZZ[finite], 98))

    fig = plt.figure(figsize=(11.5, 8.5), dpi=dpi)

    # IMPORTANT: reserve space for colorbar explicitly
    fig.subplots_adjust(left=0.05, right=0.88, bottom=0.08, top=0.92)

    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        XX, YY, ZZ,
        cmap="viridis",
        linewidth=0,
        antialiased=True
    )

    ax.set_title(title, pad=18)
    ax.set_xlabel("Feature index (top-N ordered)", labelpad=12)
    ax.set_ylabel("Run index", labelpad=12)
    ax.set_zlabel("Relevance score", labelpad=10)

    ax.set_zlim(zmin, zmax)
    ax.view_init(elev=30, azim=-55)

    # --- Colorbar: fully controlled placement ---
    cbar = fig.colorbar(
        surf,
        ax=ax,
        shrink=0.65,
        aspect=18,
        pad=0.02
    )

    cbar.set_label("Relevance score", fontsize=10, labelpad=8)
    cbar.ax.tick_params(labelsize=9)

    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)



# -----------------------------
# Pipeline
# -----------------------------

DOMAINS = ["ALL", "BIO", "PSYCHO", "SOCIAL"]

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantify ranking stability using numerical relevance scores and plot top-100 score matrices per domain.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--runs_dir",
        type=str,
        default="/Users/stijnvanseveren/PythonProjects/MASTERPROEF/Evaluation/03_construction_initial_observation_model/helpers/00_HyDe_based_predictor_ranks/runs",
        help="Directory containing run subfolders (each with dense_profiles.csv).",
    )

    parser.add_argument(
        "--base_output_dir",
        type=str,
        default="/Users/stijnvanseveren/PythonProjects/MASTERPROEF/Evaluation/03_construction_initial_observation_model/helpers/00_HyDe_based_predictor_ranks/ranking_stability",
        help="Base output directory for summaries and plots.",
    )

    parser.add_argument("--topk", type=int, default=200, help="Top-K features in each run per pseudoprofile.")
    parser.add_argument("--topn_plot", type=int, default=100, help="Top-N features to visualize (scores).")
    parser.add_argument("--min_runs", type=int, default=2, help="Minimum runs required per pseudoprofile.")

    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    base_out = Path(args.base_output_dir)

    # Output structure
    summaries_dir = base_out / "summaries"
    per_pp_dir = summaries_dir / "per_pseudoprofile"
    plots_dir = base_out / "plots"

    ensure_dir(summaries_dir)
    ensure_dir(per_pp_dir)
    ensure_dir(plots_dir)

    for dom in DOMAINS:
        ensure_dir(plots_dir / dom / "heatmaps")
        ensure_dir(plots_dir / dom / "surfaces3d")

    # Discover runs
    run_dirs = discover_runs(runs_dir)
    if not run_dirs:
        raise FileNotFoundError(f"No run subfolders with dense_profiles.csv found under: {runs_dir}")

    # Load all runs into memory (per run: pseudoprofile -> ranking)
    runs_rankings: Dict[str, Dict[str, RunProfileRanking]] = {}
    for d in run_dirs:
        run_id = d.name
        csv_path = d / "dense_profiles.csv"
        df = load_dense_profiles_csv(csv_path, topk=args.topk)
        runs_rankings[run_id] = extract_rankings_per_run(df, topk=args.topk)

    all_runs = sorted(runs_rankings.keys())
    all_pseudos = sorted(set(itertools.chain.from_iterable(runs_rankings[r].keys() for r in all_runs)))

    overall_rows: List[dict] = []

    for pseudo_id in all_pseudos:
        run_ids = [r for r in all_runs if pseudo_id in runs_rankings[r]]
        if len(run_ids) < args.min_runs:
            continue

        per_run = {r: runs_rankings[r][pseudo_id] for r in run_ids}

        # Union of features across runs (topK union)
        union_features = set()
        for r in run_ids:
            union_features.update(per_run[r].paths[:args.topk])

        union_features = [f for f in union_features if f and str(f).strip() != ""]
        if not union_features:
            continue

        # Pre-build full matrices once for ALL, then filter by domain
        features_all = sorted(union_features)
        B_all, S_all, R_all = build_matrices_for_features(features_all, run_ids, per_run)
        table_all = per_feature_table(features_all, B_all, S_all, R_all)

        # Save a per-pseudoprofile summary text
        pp_out = per_pp_dir / sanitize_filename(pseudo_id)
        ensure_dir(pp_out)

        summary_lines = []
        summary_lines.append(f"PSEUDOPROFILE: {pseudo_id}")
        summary_lines.append(f"Runs included: {len(run_ids)}")
        summary_lines.append("Run IDs:")
        for r in run_ids:
            summary_lines.append(f"  - {r}")
        summary_lines.append("")
        summary_lines.append(f"TopK used per run: {args.topk}")
        summary_lines.append(f"Union features across runs (within topK): {len(features_all)}")
        summary_lines.append(f"TopN visualized: {args.topn_plot}")
        summary_lines.append("")
        (pp_out / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

        # Helper for domain filtering
        def domain_filter_df(df: pd.DataFrame, dom: str) -> pd.DataFrame:
            if dom == "ALL":
                return df
            return df[df["domain"] == dom].copy()

        # For each domain: select topN, export CSV, plot heatmap + 3D surface
        for dom in DOMAINS:
            dom_df = domain_filter_df(table_all, dom)
            if dom_df.empty:
                # still write an empty CSV for traceability
                empty_csv = pp_out / f"{dom}__top{args.topn_plot}_features.csv"
                dom_df.to_csv(empty_csv, index=False)
                continue

            top_df = select_top_features(dom_df, topn=args.topn_plot)

            # Export table
            out_csv = pp_out / f"{dom}__top{args.topn_plot}_features.csv"
            top_df.to_csv(out_csv, index=False)

            # Build score matrix restricted to selected features (preserve order)
            feat_sel = top_df["feature_path_full"].tolist()
            idx_map = {f: i for i, f in enumerate(features_all)}
            sel_idx = [idx_map[f] for f in feat_sel if f in idx_map]

            if not sel_idx:
                continue

            S_sel = S_all[:, sel_idx].astype(float)

            run_labels = run_ids
            feat_labels = feat_sel

            # Heatmap (paper-ready)
            heat_png = plots_dir / dom / "heatmaps" / f"{sanitize_filename(pseudo_id)}__{dom}__scores_top{len(sel_idx)}.png"
            title = f"{pseudo_id} — {dom} — relevance scores (top {len(sel_idx)} stable/frequent features)"
            plot_scores_heatmap_png(S_sel, run_labels, feat_labels, title, heat_png)

            # 3D surface
            surf_png = plots_dir / dom / "surfaces3d" / f"{sanitize_filename(pseudo_id)}__{dom}__surface_top{len(sel_idx)}.png"
            title3d = f"{pseudo_id} — {dom} — 3D surface of relevance scores (top {len(sel_idx)})"
            plot_scores_surface3d_png(S_sel, title3d, surf_png)

            # Add overall row (domain-level) for later ranking
            overall_rows.append({
                "pseudoprofile_id": pseudo_id,
                "domain": dom,
                "n_runs": len(run_ids),
                "n_union_features_domain": int(dom_df.shape[0]),
                "topN_exported": int(len(sel_idx)),
                "mean_presence_frequency_topN": float(np.nanmean(top_df["presence_frequency"].to_numpy(dtype=float))),
                "mean_mean_rank_topN": float(np.nanmean(top_df["mean_rank_present"].to_numpy(dtype=float))),
                "mean_mean_score_topN": float(np.nanmean(top_df["mean_score_present"].to_numpy(dtype=float))),
            })

    # Overall CSV
    if overall_rows:
        overall_df = pd.DataFrame(overall_rows)
        overall_df = overall_df.sort_values(
            by=["domain", "mean_presence_frequency_topN", "mean_mean_rank_topN"],
            ascending=[True, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        overall_df.to_csv(summaries_dir / "overall_summary.csv", index=False)

    print(f"Done.\nOutputs written under: {base_out}")


if __name__ == "__main__":
    main()

# TODO: label 'relevance score' is just outside the 3D surface plot
