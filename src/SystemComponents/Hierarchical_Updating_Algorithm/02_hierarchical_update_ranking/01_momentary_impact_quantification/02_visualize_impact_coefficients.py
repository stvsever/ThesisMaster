#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_impact_coefficients.py

Goal
-----
For each profile folder produced by compute_momentary_impact_coefficients.py, read:
    <profile_dir>/impact_matrix.csv

Treat each matrix entry M[criterion, predictor] as the (unsigned) impact of:
    predictor  ->  criterion

Build an edge-weighted directed network where edge weights are PROPORTIONS of the
final impact matrix, i.e.:

    w_global(p -> c) = M[c,p] / sum_{all edges} M[c,p]

Additionally compute two useful normalized variants (for optional visuals):
    w_out_share(p -> c) = M[c,p] / sum_{c} M[c,p]         (within predictor)
    w_in_share(p -> c)  = M[c,p] / sum_{p} M[c,p]         (within criterion)

Outputs per profile
-------------------
Creates:
    <profile_dir>/visuals/   (if missing)

Saves publish-ready PNGs:
  1) heatmap_impact_matrix.png
  2) heatmap_impact_matrix_top.png
  3) network_bipartite_global_topK.png
  4) network_circular_global_topK.png
  5) bars_predictor_outgoing.png
  6) bars_criterion_incoming.png
  7) edge_weight_distribution.png

Also exports:
  - <profile_dir>/visuals/network_edges_proportion.csv
  - <profile_dir>/visuals/network_nodes_strength.csv

Defaults follow the current evaluation pipeline:
  root:
    /Users/stijnvanseveren/PythonProjects/MASTERPROEF/Evaluation/
    04_initial_observation_analysis/02_momentary_impact_coefficients
  integrated runs:
    /Users/stijnvanseveren/PythonProjects/MASTERPROEF/Evaluation/
    05_integrated_pipeline_runs/<run_id>/02_momentary_impact_coefficients
  pattern: pseudoprofile_FTC_

Usage
-----
python visualize_impact_proportion_networks.py
python visualize_impact_proportion_networks.py --root "<...>" --pattern "pseudoprofile_FTC_"

Notes on “publish-ready”
------------------------
- High DPI (default 300)
- Vector-like crisp linework via matplotlib
- Conservative labeling to avoid unreadable hairballs:
  network plots show only top-K edges by global proportion
"""

from __future__ import annotations

import argparse
import math
import os
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


# ---------------------------------------------------------------------
# Defaults (match your pipeline roots)
# ---------------------------------------------------------------------
def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        has_eval = (candidate / "evaluation").exists() or (candidate / "Evaluation").exists()
        if has_eval and (candidate / "README.md").exists():
            return candidate
    raise RuntimeError("Could not locate repository root from 02_visualize_impact_coefficients.py")


REPO_ROOT = _find_repo_root()
DEFAULT_ROOT = str(
    REPO_ROOT / "evaluation/04_initial_observation_analysis/02_momentary_impact_coefficients"
)
DEFAULT_PATTERN = "pseudoprofile_FTC_"
DEFAULT_MATRIX_NAME = "impact_matrix.csv"


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _init_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def safe_read_csv(path: Path) -> pd.DataFrame:
    # Try common separators
    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            if df.shape[1] > 1 or sep == ",":
                return df
        except Exception:
            pass
    return pd.read_csv(path, engine="python")


def as_numeric_matrix(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.fillna(0.0)
    return out


def robust_sort_labels_by_strength(mat: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Order predictors/criteria by total mass for cleaner visuals."""
    # rows = criteria, cols = predictors
    row_strength = mat.sum(axis=1).sort_values(ascending=False)
    col_strength = mat.sum(axis=0).sort_values(ascending=False)
    return row_strength.index.tolist(), col_strength.index.tolist()


@dataclass
class NetworkTables:
    edges: pd.DataFrame
    nodes: pd.DataFrame
    total_mass: float


def build_network_tables(mat: pd.DataFrame) -> NetworkTables:
    """
    mat: rows=criteria, cols=predictors, values>=0 (impact)
    """
    mat = mat.copy()

    # Ensure non-negative (impact_matrix from your pipeline is unsigned, but stay safe)
    mat[mat < 0] = 0.0

    total = float(mat.to_numpy(dtype=float).sum())
    if total <= 1e-12:
        return NetworkTables(edges=pd.DataFrame(), nodes=pd.DataFrame(), total_mass=0.0)

    # Totals
    predictor_out = mat.sum(axis=0)  # col sums
    criterion_in = mat.sum(axis=1)   # row sums

    # Edge list
    edges = mat.stack().reset_index()
    edges.columns = ["criterion", "predictor", "impact"]

    edges["impact"] = pd.to_numeric(edges["impact"], errors="coerce").fillna(0.0)
    edges = edges[edges["impact"] > 0].copy()

    # Proportions
    edges["w_global"] = edges["impact"] / total

    # within-predictor share
    pred_out_map = predictor_out.to_dict()
    edges["w_out_share"] = edges.apply(
        lambda r: (r["impact"] / pred_out_map.get(r["predictor"], 0.0)) if pred_out_map.get(r["predictor"], 0.0) > 1e-12 else 0.0,
        axis=1,
    )

    # within-criterion share
    crit_in_map = criterion_in.to_dict()
    edges["w_in_share"] = edges.apply(
        lambda r: (r["impact"] / crit_in_map.get(r["criterion"], 0.0)) if crit_in_map.get(r["criterion"], 0.0) > 1e-12 else 0.0,
        axis=1,
    )

    edges = edges.sort_values("w_global", ascending=False).reset_index(drop=True)

    # Node table
    nodes_pred = pd.DataFrame({
        "node": predictor_out.index.astype(str),
        "type": "predictor",
        "strength": predictor_out.values.astype(float),
        "strength_prop_global": (predictor_out.values.astype(float) / total),
    })

    nodes_crit = pd.DataFrame({
        "node": criterion_in.index.astype(str),
        "type": "criterion",
        "strength": criterion_in.values.astype(float),
        "strength_prop_global": (criterion_in.values.astype(float) / total),
    })

    nodes = pd.concat([nodes_pred, nodes_crit], ignore_index=True)
    nodes = nodes.sort_values(["type", "strength"], ascending=[True, False]).reset_index(drop=True)

    return NetworkTables(edges=edges, nodes=nodes, total_mass=total)


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------
def _savefig(fig: plt.Figure, out_path: Path, dpi: int = 300, metadata: Dict[str, Any] | None = None) -> None:
    ensure_dir(out_path.parent)
    svg_path = out_path.with_suffix(".svg")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    figure_meta = dict(metadata or {})
    figure_meta["generated_at"] = datetime.now().isoformat(timespec="seconds")
    figure_meta["files"] = [str(out_path), str(svg_path), str(pdf_path)]
    out_path.with_suffix(".figure.json").write_text(
        json.dumps(figure_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _safe_read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def plot_heatmap(mat: pd.DataFrame, out_path: Path, title: str, dpi: int = 300) -> None:
    """
    Publish-ready heatmap with sensible ordering and labels.
    """
    # Order by strengths
    row_order, col_order = robust_sort_labels_by_strength(mat)
    mat2 = mat.reindex(index=row_order, columns=col_order)

    fig = plt.figure(figsize=(max(6, 0.35 * mat2.shape[1]), max(5, 0.35 * mat2.shape[0])))
    ax = fig.add_subplot(111)

    im = ax.imshow(mat2.to_numpy(dtype=float), aspect="auto")
    ax.set_title(title, fontsize=12)

    ax.set_xticks(np.arange(mat2.shape[1]))
    ax.set_xticklabels(mat2.columns.tolist(), rotation=90, fontsize=7)
    ax.set_yticks(np.arange(mat2.shape[0]))
    ax.set_yticklabels(mat2.index.tolist(), fontsize=7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Impact (edge_impact)", rotation=90, fontsize=9)

    _savefig(fig, out_path, dpi=dpi)


def plot_heatmap_top(mat: pd.DataFrame, out_path: Path, title: str, top_n: int = 25, dpi: int = 300) -> None:
    """
    Heatmap restricted to top-N predictors and criteria by total mass for readability.
    """
    row_strength = mat.sum(axis=1).sort_values(ascending=False)
    col_strength = mat.sum(axis=0).sort_values(ascending=False)
    rows = row_strength.head(top_n).index.tolist()
    cols = col_strength.head(top_n).index.tolist()

    mat2 = mat.reindex(index=rows, columns=cols).copy()

    fig = plt.figure(figsize=(max(6, 0.4 * mat2.shape[1]), max(5, 0.4 * mat2.shape[0])))
    ax = fig.add_subplot(111)
    im = ax.imshow(mat2.to_numpy(dtype=float), aspect="auto")
    ax.set_title(title, fontsize=12)

    ax.set_xticks(np.arange(mat2.shape[1]))
    ax.set_xticklabels(mat2.columns.tolist(), rotation=90, fontsize=8)
    ax.set_yticks(np.arange(mat2.shape[0]))
    ax.set_yticklabels(mat2.index.tolist(), fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Impact (edge_impact)", rotation=90, fontsize=9)

    _savefig(fig, out_path, dpi=dpi)


def plot_readiness_diagnostics(readiness_payload: Dict[str, Any], out_path: Path, title: str, dpi: int = 300) -> None:
    overall = readiness_payload.get("overall", {}) or {}
    score = float(overall.get("readiness_score_0_100", 0.0) or 0.0)
    label = str(overall.get("readiness_label") or "unknown")
    variant = str(overall.get("tier3_variant") or "none")
    tv_conf = bool(overall.get("tv_full_confidence", False))
    components = ((overall.get("score_breakdown", {}) or {}).get("components", {}) or {})
    names = ["sample_score", "missing_score", "variable_quality_score", "time_score", "assumptions_score"]
    vals = [float(components.get(k, 0.0) or 0.0) for k in names]

    fig = plt.figure(figsize=(10, 4.8))
    ax1 = fig.add_subplot(121)
    ax1.bar(["Readiness"], [score], color="#457b9d")
    ax1.set_ylim(0, 100)
    ax1.set_title(f"{title}\nlabel={label} variant={variant} tv_full_conf={tv_conf}")
    ax1.set_ylabel("Score (0-100)")

    ax2 = fig.add_subplot(122)
    ax2.barh(names, vals, color="#2a9d8f")
    ax2.set_xlim(0, 100)
    ax2.set_title("Score Components")
    _savefig(
        fig,
        out_path,
        dpi=dpi,
        metadata={"plot_type": "readiness_diagnostics_panel", "readiness_label": label, "tier3_variant": variant},
    )


def plot_network_execution_summary(comparison_payload: Dict[str, Any], out_path: Path, title: str, dpi: int = 300) -> None:
    execution = comparison_payload.get("execution_plan", {}) or {}
    status = comparison_payload.get("method_status", {}) or {}
    labels = ["tv", "stationary", "corr"]
    vals = [1.0 if str(status.get(k, "")).lower() == "executed" else 0.0 for k in labels]
    fig = plt.figure(figsize=(8, 4.6))
    ax = fig.add_subplot(111)
    colors = ["#1b9e77" if v > 0.5 else "#e76f51" for v in vals]
    ax.bar(labels, vals, color=colors)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["skipped", "executed"])
    ax.set_title(
        f"{title}\nanalysis_set={execution.get('analysis_set')} tv_full_conf={execution.get('tv_full_confidence')}"
    )
    _savefig(
        fig,
        out_path,
        dpi=dpi,
        metadata={"plot_type": "network_method_execution_summary", "analysis_set": execution.get("analysis_set")},
    )


def _scaled_node_sizes(strength_prop: np.ndarray, min_size: float = 80.0, max_size: float = 900.0) -> np.ndarray:
    x = np.asarray(strength_prop, dtype=float)
    if x.size == 0:
        return x
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi <= lo + 1e-12:
        return np.full_like(x, (min_size + max_size) / 2.0)
    y = (x - lo) / (hi - lo)
    return min_size + y * (max_size - min_size)


def _edge_lw(weights: np.ndarray, min_lw: float = 0.4, max_lw: float = 6.0) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        return w
    lo, hi = float(np.min(w)), float(np.max(w))
    if hi <= lo + 1e-12:
        return np.full_like(w, (min_lw + max_lw) / 2.0)
    y = (w - lo) / (hi - lo)
    return min_lw + y * (max_lw - min_lw)


def plot_bipartite_network(
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    out_path: Path,
    title: str,
    top_k_edges: int = 60,
    dpi: int = 300,
) -> None:
    """
    Bipartite-ish layout: predictors left, criteria right.
    Edges are top-K by global proportion w_global.
    """
    if edges.empty or nodes.empty:
        return

    ed = edges.sort_values("w_global", ascending=False).head(int(top_k_edges)).copy()

    predictors = (
        nodes[nodes["type"] == "predictor"]
        .sort_values("strength", ascending=False)["node"]
        .astype(str).tolist()
    )
    criteria = (
        nodes[nodes["type"] == "criterion"]
        .sort_values("strength", ascending=False)["node"]
        .astype(str).tolist()
    )

    # Keep only nodes actually used in top edges (for readability)
    used_pred = sorted(set(ed["predictor"].astype(str)))
    used_crit = sorted(set(ed["criterion"].astype(str)))

    predictors = [p for p in predictors if p in used_pred]
    criteria = [c for c in criteria if c in used_crit]

    # y positions (top to bottom)
    def y_positions(items: List[str]) -> Dict[str, float]:
        n = len(items)
        if n <= 1:
            return {items[0]: 0.5} if n == 1 else {}
        return {it: 1.0 - (i / (n - 1)) for i, it in enumerate(items)}

    y_pred = y_positions(predictors)
    y_crit = y_positions(criteria)

    # node strength proportions for sizing
    node_map = nodes.set_index("node")
    pred_props = np.array([float(node_map.loc[p, "strength_prop_global"]) for p in predictors], dtype=float) if predictors else np.array([])
    crit_props = np.array([float(node_map.loc[c, "strength_prop_global"]) for c in criteria], dtype=float) if criteria else np.array([])

    pred_sizes = _scaled_node_sizes(pred_props) if pred_props.size else np.array([])
    crit_sizes = _scaled_node_sizes(crit_props) if crit_props.size else np.array([])

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.set_title(title, fontsize=12)

    # Draw edges
    lws = _edge_lw(ed["w_global"].to_numpy(dtype=float))
    for (idx, r), lw in zip(ed.iterrows(), lws):
        p = str(r["predictor"])
        c = str(r["criterion"])
        if p not in y_pred or c not in y_crit:
            continue
        x1, y1 = 0.05, y_pred[p]
        x2, y2 = 0.95, y_crit[c]

        # Curvature based on relative ordering to reduce overlap
        rad = 0.12 * np.sign(y2 - y1)

        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=float(lw),
            alpha=0.65,
            connectionstyle=f"arc3,rad={rad}",
        )
        ax.add_patch(arrow)

    # Draw nodes (simple grayscale so it prints well)
    # predictors
    if predictors:
        ax.scatter(
            np.full(len(predictors), 0.05),
            [y_pred[p] for p in predictors],
            s=pred_sizes,
            marker="o",
            edgecolors="black",
            linewidths=0.6,
            alpha=0.95,
        )
        for p in predictors:
            ax.text(0.02, y_pred[p], p, ha="right", va="center", fontsize=9)

    # criteria
    if criteria:
        ax.scatter(
            np.full(len(criteria), 0.95),
            [y_crit[c] for c in criteria],
            s=crit_sizes,
            marker="o",
            edgecolors="black",
            linewidths=0.6,
            alpha=0.95,
        )
        for c in criteria:
            ax.text(0.98, y_crit[c], c, ha="left", va="center", fontsize=9)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")

    _savefig(fig, out_path, dpi=dpi)


def plot_circular_network(
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    out_path: Path,
    title: str,
    top_k_edges: int = 80,
    dpi: int = 300,
) -> None:
    """
    Circular layout of predictors+criteria.
    Edges are top-K by global proportion (w_global) with curved arrows.
    """
    if edges.empty or nodes.empty:
        return

    ed = edges.sort_values("w_global", ascending=False).head(int(top_k_edges)).copy()

    # Node ordering: predictors then criteria, each sorted by strength
    preds = nodes[nodes["type"] == "predictor"].sort_values("strength", ascending=False)["node"].astype(str).tolist()
    crits = nodes[nodes["type"] == "criterion"].sort_values("strength", ascending=False)["node"].astype(str).tolist()

    # Keep only nodes used in edges
    used = sorted(set(ed["predictor"].astype(str)).union(set(ed["criterion"].astype(str))))
    preds = [n for n in preds if n in used]
    crits = [n for n in crits if n in used]
    order = preds + crits
    n = len(order)
    if n == 0:
        return

    # positions on a circle
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {node: (math.cos(a), math.sin(a)) for node, a in zip(order, angles)}

    node_map = nodes.set_index("node")
    props = np.array([float(node_map.loc[node, "strength_prop_global"]) for node in order], dtype=float)
    sizes = _scaled_node_sizes(props, min_size=70, max_size=750)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_title(title, fontsize=12)

    # Draw edges
    lws = _edge_lw(ed["w_global"].to_numpy(dtype=float), min_lw=0.3, max_lw=5.5)
    for (idx, r), lw in zip(ed.iterrows(), lws):
        p = str(r["predictor"])
        c = str(r["criterion"])
        if p not in pos or c not in pos:
            continue
        x1, y1 = pos[p]
        x2, y2 = pos[c]

        # curvature depends on angular distance
        # choose a moderate consistent arc
        rad = 0.25

        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=float(lw),
            alpha=0.55,
            connectionstyle=f"arc3,rad={rad}",
        )
        ax.add_patch(arrow)

    # Draw nodes
    xs = [pos[node][0] for node in order]
    ys = [pos[node][1] for node in order]
    ax.scatter(xs, ys, s=sizes, edgecolors="black", linewidths=0.6, alpha=0.95)

    # Labels (keep readable: label only top nodes by strength)
    strength_sorted = nodes[nodes["node"].isin(order)].sort_values("strength", ascending=False)
    top_label_nodes = strength_sorted.head(min(18, len(strength_sorted)))["node"].astype(str).tolist()
    for node in order:
        if node not in top_label_nodes:
            continue
        x, y = pos[node]
        ax.text(1.10 * x, 1.10 * y, node, ha="center", va="center", fontsize=9)

    ax.set_aspect("equal")
    ax.axis("off")

    _savefig(fig, out_path, dpi=dpi)


def plot_bars_predictors(nodes: pd.DataFrame, out_path: Path, title: str, top_n: int = 25, dpi: int = 300) -> None:
    df = nodes[nodes["type"] == "predictor"].sort_values("strength", ascending=False).head(int(top_n)).copy()
    if df.empty:
        return

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.set_title(title, fontsize=12)

    ax.bar(df["node"].astype(str).tolist(), df["strength_prop_global"].to_numpy(dtype=float))
    ax.set_ylabel("Global proportion of total impact mass")
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(df["node"].astype(str).tolist(), rotation=60, ha="right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)

    _savefig(fig, out_path, dpi=dpi)


def plot_bars_criteria(nodes: pd.DataFrame, out_path: Path, title: str, top_n: int = 25, dpi: int = 300) -> None:
    df = nodes[nodes["type"] == "criterion"].sort_values("strength", ascending=False).head(int(top_n)).copy()
    if df.empty:
        return

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.set_title(title, fontsize=12)

    ax.bar(df["node"].astype(str).tolist(), df["strength_prop_global"].to_numpy(dtype=float))
    ax.set_ylabel("Global proportion of total impact mass")
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(df["node"].astype(str).tolist(), rotation=60, ha="right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)

    _savefig(fig, out_path, dpi=dpi)


def plot_edge_weight_distribution(edges: pd.DataFrame, out_path: Path, title: str, dpi: int = 300) -> None:
    if edges.empty:
        return

    w = edges["w_global"].to_numpy(dtype=float)
    w = np.sort(w)[::-1]
    cum = np.cumsum(w)

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)
    ax.set_title(title, fontsize=12)

    ax.plot(np.arange(1, len(w) + 1), w, label="Edge proportion (sorted)")
    ax.plot(np.arange(1, len(w) + 1), cum, label="Cumulative proportion")

    ax.set_xlabel("Edge rank (descending)")
    ax.set_ylabel("Proportion")
    ax.grid(True, alpha=0.25)
    ax.legend()

    _savefig(fig, out_path, dpi=dpi)


# ---------------------------------------------------------------------
# Profile loop
# ---------------------------------------------------------------------
def list_profile_dirs(root: Path, pattern: str) -> List[Path]:
    if not root.exists():
        return []
    dirs = [p for p in root.iterdir() if p.is_dir()]
    if pattern:
        dirs = [d for d in dirs if pattern in d.name]
    return sorted(dirs)


def process_profile(profile_dir: Path, matrix_name: str, dpi: int, top_k_edges: int, top_n_heatmap: int, top_n_bars: int) -> str:
    mat_path = profile_dir / matrix_name
    if not mat_path.exists():
        log(f"[SKIP] {profile_dir.name}: missing {mat_path}")
        return "skipped"

    # Read matrix (criteria rows x predictors cols)
    mat_raw = safe_read_csv(mat_path)

    # impact_matrix.csv from your pipeline has first column as index if saved by pandas
    # Typical format: first column unnamed with row labels. Handle that.
    if mat_raw.columns[0].lower() in ["criterion", "unnamed: 0", ""]:
        # likely pivot export; set first column as index
        mat_raw = mat_raw.set_index(mat_raw.columns[0])

    mat = as_numeric_matrix(mat_raw)
    # Ensure row index and col labels are strings
    mat.index = mat.index.astype(str)
    mat.columns = mat.columns.astype(str)

    tables = build_network_tables(mat)
    if tables.total_mass <= 1e-12 or tables.edges.empty:
        log(f"[SKIP] {profile_dir.name}: matrix has zero total mass (all zeros).")
        return "skipped"

    profile_id = profile_dir.name
    visuals_dir = ensure_dir(profile_dir / "visuals")
    run_root = profile_dir.parent.parent
    readiness_payload = _safe_read_json(run_root / "00_readiness_check" / profile_id / "readiness_report.json")
    comparison_payload = _safe_read_json(run_root / "01_time_series_analysis" / "network" / profile_id / "comparison_summary.json")

    # Save network tables
    tables.edges.to_csv(visuals_dir / "network_edges_proportion.csv", index=False)
    tables.nodes.to_csv(visuals_dir / "network_nodes_strength.csv", index=False)

    # Plot set

    plot_heatmap(
        mat,
        visuals_dir / "heatmap_impact_matrix.png",
        title=f"{profile_id} — Impact matrix (edge_impact)",
        dpi=dpi,
    )

    plot_heatmap_top(
        mat,
        visuals_dir / "heatmap_impact_matrix_top.png",
        title=f"{profile_id} — Impact matrix (top by mass)",
        top_n=top_n_heatmap,
        dpi=dpi,
    )

    plot_bipartite_network(
        tables.edges,
        tables.nodes,
        visuals_dir / "network_bipartite_global_topK.png",
        title=f"{profile_id} — Predictor→Criterion network (top-{top_k_edges} edges by global proportion)",
        top_k_edges=top_k_edges,
        dpi=dpi,
    )

    plot_circular_network(
        tables.edges,
        tables.nodes,
        visuals_dir / "network_circular_global_topK.png",
        title=f"{profile_id} — Circular network (top-{top_k_edges} edges by global proportion)",
        top_k_edges=max(top_k_edges, 80),
        dpi=dpi,
    )

    plot_bars_predictors(
        tables.nodes,
        visuals_dir / "bars_predictor_outgoing.png",
        title=f"{profile_id} — Predictors: share of total impact mass (top-{top_n_bars})",
        top_n=top_n_bars,
        dpi=dpi,
    )

    plot_bars_criteria(
        tables.nodes,
        visuals_dir / "bars_criterion_incoming.png",
        title=f"{profile_id} — Criteria: share of total impact mass (top-{top_n_bars})",
        top_n=top_n_bars,
        dpi=dpi,
    )

    plot_edge_weight_distribution(
        tables.edges,
        visuals_dir / "edge_weight_distribution.png",
        title=f"{profile_id} — Edge proportion concentration",
        dpi=dpi,
    )

    if readiness_payload:
        plot_readiness_diagnostics(
            readiness_payload,
            visuals_dir / "readiness_diagnostics_panel.png",
            title=f"{profile_id} — Readiness Diagnostics",
            dpi=dpi,
        )
    if comparison_payload:
        plot_network_execution_summary(
            comparison_payload,
            visuals_dir / "network_method_execution_summary.png",
            title=f"{profile_id} — Network Method Execution",
            dpi=dpi,
        )

    log(f"[OK] {profile_id} -> {visuals_dir}")
    return "ok"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build proportion-weighted impact networks and publish-ready visualizations for all profiles."
    )
    p.add_argument("--root", type=str, default=DEFAULT_ROOT, help="Root folder containing pseudoprofile_* folders.")
    p.add_argument("--pattern", type=str, default=DEFAULT_PATTERN, help="Substring filter for profile directories.")
    p.add_argument("--matrix-name", type=str, default=DEFAULT_MATRIX_NAME, help="Name of the impact matrix CSV inside each profile dir.")
    p.add_argument("--dpi", type=int, default=300, help="PNG DPI.")
    p.add_argument("--top-k-edges", type=int, default=60, help="Top-K edges (by global proportion) to show in network plots.")
    p.add_argument("--top-n-heatmap", type=int, default=25, help="Top-N predictors/criteria to show in the 'top' heatmap.")
    p.add_argument("--top-n-bars", type=int, default=25, help="Top-N to show in bar charts.")
    p.add_argument("--max-profiles", type=int, default=0, help="0 = all; otherwise process first N.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    _init_plot_style()
    root = Path(args.root).expanduser()

    prof_dirs = list_profile_dirs(root, pattern=str(args.pattern or ""))
    if args.max_profiles and int(args.max_profiles) > 0:
        prof_dirs = prof_dirs[: int(args.max_profiles)]

    if not prof_dirs:
        log(f"No profile directories found in: {root} (pattern={args.pattern!r})")
        return 2

    log("========== IMPACT PROPORTION NETWORK VISUALS START ==========")
    log(f"root:        {root}")
    log(f"pattern:     {args.pattern!r}")
    log(f"profiles:    {len(prof_dirs)}")
    log(f"matrix:      {args.matrix_name!r}")
    log(f"dpi:         {args.dpi}")
    log(f"top_k_edges: {args.top_k_edges}")
    log("")

    n_ok, n_skip, n_fail = 0, 0, 0
    for d in prof_dirs:
        try:
            status = process_profile(
                profile_dir=d,
                matrix_name=str(args.matrix_name),
                dpi=int(args.dpi),
                top_k_edges=int(args.top_k_edges),
                top_n_heatmap=int(args.top_n_heatmap),
                top_n_bars=int(args.top_n_bars),
            )
            if status == "ok":
                n_ok += 1
            else:
                n_skip += 1
        except Exception as e:
            n_fail += 1
            log(f"[ERROR] {d.name}: {repr(e)}")

    log("")
    log("========== IMPACT PROPORTION NETWORK VISUALS COMPLETE ==========")
    log(f"Success: {n_ok}  Skipped: {n_skip}  Failed: {n_fail}")
    (root / "visualization_run_summary.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "root": str(root),
                "pattern": str(args.pattern),
                "matrix_name": str(args.matrix_name),
                "n_profiles_total": int(len(prof_dirs)),
                "n_profiles_success": int(n_ok),
                "n_profiles_skipped": int(n_skip),
                "n_profiles_failed": int(n_fail),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0 if n_fail == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
