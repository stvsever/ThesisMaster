#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch heatmaps + (SOTA) clustering + network visualizations for ontology mapping edge lists.

Per mapping:
  1) Read *_edges_long.csv
  2) Build dense score matrix (left labels x right labels), save
  3) Publication-ready heatmaps (PNG ONLY) with ALL labels (full + shortened)
  4) SOTA clustering:
       - Primary: Spectral Co-Clustering (biclustering) on the matrix (sklearn)
       - Enhancement: TruncatedSVD -> UMAP -> HDBSCAN on row/col embeddings (auto-install optional)
       - Fallback: KMeans on SVD embeddings (used for choosing k for biclustering)
  5) Clear bipartite networks (filtered edges + per-bicluster subnetworks)
  6) Save everything under:
       BASE_OUT/<mapping_name>/{results,plots,data}/...

Special requirement (predictor_to_criterion):
  - Y-axis (predictors) must reflect the predictor hierarchy stored in predictors_list.txt.
  - The label region includes an additional hierarchy visualization via row-annotation bars + grouped labels.
  - Criterion clusters (cluster_id) are kept "as is" (no dendrogram reordering).

Core deps:
  pip install pandas numpy matplotlib scipy scikit-learn networkx

Optional (recommended for “SOTA”):
  pip install umap-learn hdbscan

Run:
  /usr/local/bin/python3.11 path/to/cluster_mapping_scores.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.transforms import blended_transform_factory

import networkx as nx

from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans, SpectralCoclustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# SciPy clustering for dendrogram ordering (optional)
try:
    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy import linkage, leaves_list
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# -----------------------------
# USER PATHS (as given)
# -----------------------------

INPUTS = [
    # 1) CRITERION: predictor_to_criterion (clustered criterion side)
    dict(
        name="predictor_to_criterion",
        csv_path="/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/ontology_mappings/CRITERION/predictor_to_criterion/results/gpt-5-nano/predictor_to_criterion_edges_long.csv",
        left_label_col="predictor_full_path",
        right_label_col="cluster_id",
        score_col="score",
    ),
    # 2) HAPA: context_to_barrier
    dict(
        name="context_to_barrier",
        csv_path="/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/ontology_mappings/HAPA/context_to_barrier/results/gpt-5-nano/context_to_barrier_edges_long.csv",
        left_label_col="context_full_path",
        right_label_col="barrier_full_path",
        score_col="score",
    ),
    # 3) HAPA: coping_to_barrier (infer label columns)
    dict(
        name="coping_to_barrier",
        csv_path="/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/ontology_mappings/HAPA/coping_to_barrier/results/gpt-5-nano/coping_to_barrier_edges_long.csv",
        left_label_col=None,
        right_label_col=None,
        score_col="score",
    ),
    # 4) HAPA: profile_to_barrier
    dict(
        name="profile_to_barrier",
        csv_path="/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/ontology_mappings/HAPA/profile_to_barrier/results/gpt-5-nano/profile_to_barrier_edges_long.csv",
        left_label_col="profile_full_path",
        right_label_col="barrier_full_path",
        score_col="score",
    ),
    # 5) PREDICTOR: barrier_to_predictor (file name is predictor_to_barrier_edges_long.csv)
    dict(
        name="barrier_to_predictor",
        csv_path="/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/ontology_mappings/PREDICTOR/barrier_to_predictor/results/gpt-5-nano/predictor_to_barrier_edges_long.csv",
        left_label_col="barrier_full_path",
        right_label_col="predictor_full_path",
        score_col="score",
    ),
    # 6) PREDICTOR: context_to_predictor (file name is predictor_to_context_edges_long.csv)
    dict(
        name="context_to_predictor",
        csv_path="/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/ontology_mappings/PREDICTOR/context_to_predictor/results/gpt-5-nano/predictor_to_context_edges_long.csv",
        left_label_col="context_full_path",
        right_label_col="predictor_full_path",
        score_col="score",
    ),
    # 7) PREDICTOR: profile_to_predictor (file name is predictor_to_profile_edges_long.csv)
    dict(
        name="profile_to_predictor",
        csv_path="/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/ontology_mappings/PREDICTOR/profile_to_predictor/results/gpt-5-nano/predictor_to_profile_edges_long.csv",
        left_label_col="profile_full_path",
        right_label_col="predictor_full_path",
        score_col="score",
    ),
]

BASE_OUT = Path(
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/overall_mapping_analyses/utils/clustering_analysis/results"
).expanduser()

PREDICTOR_HIERARCHY_PATH = Path(
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/ontology_mappings/PREDICTOR/profile_to_predictor/input_lists/predictors_list.txt"
).expanduser()


# -----------------------------
# CONFIG / TUNABLES
# -----------------------------
RANDOM_SEED = 42

# Heatmap appearance
DPI = 300
HEATMAP_MAX_ANNOTATE_CELLS = 40 * 40  # annotate only if <= this many cells
LABEL_WRAP_WIDTH = 40
MAX_FIGSIZE_INCH = 40  # cap overall figure width/height

# Network filtering (for clarity)
EDGE_KEEP_QUANTILE = 0.90
EDGE_MIN_ABS = 0.10
TOPK_EDGES_PER_NODE = 10
MAX_EDGES_PLOTTED = 7000
NETWORK_NODE_LABEL_LIMIT = 80

# Clustering (biclustering)
MAX_CLUSTERS = 8
K_SEARCH_MAX = 12
MIN_DIM_FOR_CLUSTERING = 6
SVD_COMPONENTS = 32

# Optional SOTA path
TRY_UMAP_HDBSCAN = True
AUTO_INSTALL_UMAP_HDBSCAN = True  # installs with pins to avoid sklearn-compat breakage

# Predictor hierarchy visualization
DOMAIN_ORDER = {"BIO": 0, "PSYCHO": 1, "SOCIAL": 2, "UNMAPPED": 9}
DOMAIN_COLORS = {
    "BIO": "#1f77b4",     # tab:blue
    "PSYCHO": "#ff7f0e",  # tab:orange
    "SOCIAL": "#2ca02c",  # tab:green
    "UNMAPPED": "#7f7f7f"
}

# Domain label display (plot only) — user requested BIO PSYCHO SOCIAL on the far left
DOMAIN_DISPLAY = {
    "BIO": "BIO",
    "PSYCHO": "PSYCHO",
    "SOCIAL": "SOCIAL",
    "UNMAPPED": "UNMAPPED",
}


# -----------------------------
# UTILITIES
# -----------------------------

def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def wrap_label(s: str, width: int = LABEL_WRAP_WIDTH) -> str:
    return "\n".join(textwrap.wrap(str(s), width=width, break_long_words=True, break_on_hyphens=False))

def shorten_label(s: str, max_len: int = 45) -> str:
    s = str(s)
    parts = re.split(r"[/>|]+", s)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) >= 2:
        candidate = f"{parts[-2]} / {parts[-1]}"
    elif len(parts) == 1:
        candidate = parts[0]
    else:
        candidate = s
    candidate = re.sub(r"\s+", " ", candidate).strip()
    if len(candidate) <= max_len:
        return candidate
    return candidate[: max_len - 1] + "…"

def norm_key(s: str) -> str:
    s = str(s)
    s = re.sub(r"\(ID:\s*\d+\)", "", s)
    s = s.strip().lower()
    return re.sub(r"[^a-z0-9]+", "", s)

def distinct_colors(n: int, cmap_name: str = "gist_ncar") -> List[tuple]:
    """
    Return n visually distinct RGBA colors without repeating.
    Uses a continuous matplotlib colormap and samples evenly.
    """
    if n <= 0:
        return []
    cmap = plt.get_cmap(cmap_name)
    xs = np.linspace(0.05, 0.95, n)
    return [cmap(float(x)) for x in xs]

def infer_two_full_path_cols(df: pd.DataFrame, preferred_left_hint: Optional[str] = None) -> Tuple[str, str]:
    full_path_cols = [c for c in df.columns if c.lower().endswith("full_path")]
    if len(full_path_cols) >= 2:
        if preferred_left_hint:
            left_candidates = [c for c in full_path_cols if preferred_left_hint.lower() in c.lower()]
            if left_candidates:
                left = left_candidates[0]
                right = [c for c in full_path_cols if c != left][0]
                return left, right
        return full_path_cols[0], full_path_cols[1]

    name_cols = [c for c in df.columns if c.lower().endswith("name")]
    if len(name_cols) >= 2:
        return name_cols[0], name_cols[1]

    raise ValueError(f"Could not infer two label columns. Columns: {list(df.columns)}")

def densify_matrix(df: pd.DataFrame, left_col: str, right_col: str, score_col: str, agg: str = "max") -> pd.DataFrame:
    dfx = df[[left_col, right_col, score_col]].copy()

    dfx[left_col] = dfx[left_col].astype("string").str.strip()
    dfx[right_col] = dfx[right_col].astype("string").str.strip()

    dfx = dfx.dropna(subset=[left_col, right_col, score_col])

    dfx[score_col] = pd.to_numeric(dfx[score_col], errors="coerce")
    dfx = dfx.dropna(subset=[score_col])

    mat = dfx.pivot_table(
        index=left_col,
        columns=right_col,
        values=score_col,
        aggfunc=agg,
        fill_value=0.0,
    )

    mat = mat.sort_index(axis=0).sort_index(axis=1)
    return mat

def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def _safe_cosine_like_pdist_rows(mat: np.ndarray) -> np.ndarray:
    eps = 1e-12
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    normed = mat / np.maximum(norms, eps)
    dist = pdist(normed, metric="euclidean")
    dist = np.nan_to_num(dist, nan=0.0, posinf=np.finfo(float).max, neginf=0.0)
    return dist

def hierarchical_order(mat: np.ndarray, axis: int = 0, metric: str = "cosine", method: str = "average") -> np.ndarray:
    if not SCIPY_OK:
        return np.arange(mat.shape[axis], dtype=int)

    mat = np.asarray(mat, dtype=float)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)

    if axis == 1:
        mat = mat.T
    n = mat.shape[0]
    if n < 2:
        return np.arange(n, dtype=int)

    if np.allclose(mat, 0.0):
        return np.arange(n, dtype=int)

    try:
        if metric.lower() == "cosine":
            dist = _safe_cosine_like_pdist_rows(mat)
        else:
            dist = pdist(mat, metric=metric)
            dist = np.nan_to_num(dist, nan=0.0, posinf=np.finfo(float).max, neginf=0.0)

        if not np.all(np.isfinite(dist)):
            dist = pdist(mat, metric="euclidean")
            dist = np.nan_to_num(dist, nan=0.0, posinf=np.finfo(float).max, neginf=0.0)

        if dist.size == 0 or np.allclose(dist, 0.0):
            return np.arange(n, dtype=int)

        Z = linkage(dist, method=method)
        return leaves_list(Z)

    except Exception:
        return np.arange(n, dtype=int)

def choose_k_by_silhouette(emb: np.ndarray, k_max: int) -> int:
    n = emb.shape[0]
    if n < 5:
        return 2
    k_max = min(k_max, n - 1)
    if k_max < 2:
        return 2

    best_k, best_score = 2, -1.0
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init="auto")
        labels = km.fit_predict(emb)
        if len(set(labels)) < 2 or len(set(labels)) >= n:
            continue
        try:
            s = silhouette_score(emb, labels, metric="euclidean")
        except Exception:
            continue
        if s > best_score:
            best_score = s
            best_k = k
    return best_k


# -----------------------------
# PREDICTOR HIERARCHY PARSING
# -----------------------------

def load_predictor_hierarchy(path: Path) -> Dict[str, Any]:
    leaf_by_norm: Dict[str, Dict[str, Any]] = {}
    parent_order: Dict[Tuple[str, str], int] = {}

    current_domain: Optional[str] = None
    current_parent: Optional[str] = None
    parent_counter = 0

    if not path.exists():
        return {"leaf_by_norm": leaf_by_norm, "parent_order": parent_order}

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for raw in lines:
        line = raw.rstrip("\n")
        s = line.strip()

        if not s:
            continue

        m_dom = re.match(r"^\[(.+?)\]$", s)
        if m_dom:
            current_domain = m_dom.group(1).strip()
            current_parent = None
            continue

        if "└─" in s and "(ID:" not in s:
            parent = re.sub(r"^└─\s*", "", s).strip()
            current_parent = parent
            if current_domain is None:
                current_domain = "UNMAPPED"
            key = (current_domain, current_parent)
            if key not in parent_order:
                parent_order[key] = parent_counter
                parent_counter += 1
            continue

        if "(ID:" in s and "└─" in s:
            leaf_part = re.sub(r"^.*└─\s*", "", s).strip()
            m_id = re.search(r"\(ID:\s*(\d+)\)", leaf_part)
            if not m_id:
                continue
            leaf = re.sub(r"\s*\(ID:\s*\d+\)\s*$", "", leaf_part).strip()
            leaf_id = int(m_id.group(1))

            dom = current_domain or "UNMAPPED"
            par = current_parent or "UNMAPPED"
            pord = parent_order.get((dom, par), 10_000_000)

            meta = {
                "domain": dom,
                "parent": par,
                "leaf": leaf,
                "id": leaf_id,
                "parent_order": int(pord),
            }

            leaf_by_norm[norm_key(leaf)] = meta
            continue

    return {"leaf_by_norm": leaf_by_norm, "parent_order": parent_order}

def map_predictor_label_to_meta(label: str, hierarchy: Dict[str, Any]) -> Dict[str, Any]:
    leaf_by_norm: Dict[str, Dict[str, Any]] = hierarchy.get("leaf_by_norm", {})

    segments = re.split(r"[/>|]+", str(label))
    segments = [re.sub(r"\(ID:\s*\d+\)", "", seg).strip() for seg in segments if seg.strip()]
    if not segments:
        return {"domain": "UNMAPPED", "parent": "UNMAPPED", "leaf": str(label), "id": 10_000_000, "parent_order": 10_000_000}

    k = norm_key(segments[-1])
    if k in leaf_by_norm:
        return leaf_by_norm[k]

    for seg in reversed(segments):
        k2 = norm_key(seg)
        if k2 in leaf_by_norm:
            return leaf_by_norm[k2]

    return {"domain": "UNMAPPED", "parent": "UNMAPPED", "leaf": segments[-1], "id": 10_000_000, "parent_order": 10_000_000}

def order_predictor_rows_by_hierarchy(index_labels: List[str], hierarchy: Dict[str, Any]) -> Tuple[List[str], pd.DataFrame]:
    metas = []
    for lbl in index_labels:
        meta = map_predictor_label_to_meta(lbl, hierarchy)
        metas.append({
            "row_label_full": str(lbl),
            "domain": meta.get("domain", "UNMAPPED"),
            "parent": meta.get("parent", "UNMAPPED"),
            "leaf": meta.get("leaf", str(lbl)),
            "id": int(meta.get("id", 10_000_000)),
            "parent_order": int(meta.get("parent_order", 10_000_000)),
            "domain_order": int(DOMAIN_ORDER.get(meta.get("domain", "UNMAPPED"), 9)),
        })

    meta_df = pd.DataFrame(metas)

    meta_df = meta_df.sort_values(
        by=["domain_order", "parent_order", "id", "leaf", "row_label_full"],
        ascending=[True, True, True, True, True],
        kind="mergesort"
    ).reset_index(drop=True)

    ordered = meta_df["row_label_full"].tolist()
    return ordered, meta_df


# -----------------------------
# HEATMAP PLOTTING (PNG ONLY)
# -----------------------------

def plot_heatmap_basic(
    matrix_df: pd.DataFrame,
    out_png: Path,
    title: str,
    wrap: bool = True,
    annotate: bool = False,
    reorder_rows: bool = True,
    reorder_cols: bool = True,
) -> None:
    M = np.nan_to_num(matrix_df.values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)

    if (reorder_rows or reorder_cols) and matrix_df.shape[0] >= 2 and matrix_df.shape[1] >= 2:
        try:
            rord = hierarchical_order(M, axis=0, metric="cosine") if reorder_rows else np.arange(matrix_df.shape[0])
            cord = hierarchical_order(M, axis=1, metric="cosine") if reorder_cols else np.arange(matrix_df.shape[1])
            matrix_df = matrix_df.iloc[rord, :].iloc[:, cord]
            M = np.nan_to_num(matrix_df.values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            pass

    n_rows, n_cols = matrix_df.shape

    w = min(MAX_FIGSIZE_INCH, max(8.0, 0.26 * n_cols))
    h = min(MAX_FIGSIZE_INCH, max(6.0, 0.26 * n_rows))

    fig, ax = plt.subplots(figsize=(w, h))
    im = ax.imshow(M, aspect="auto", interpolation="nearest")

    ax.set_title(title, fontsize=14, pad=12)

    xlabels = matrix_df.columns.tolist()
    ylabels = matrix_df.index.tolist()
    if wrap:
        xlabels = [wrap_label(x) for x in xlabels]
        ylabels = [wrap_label(y) for y in ylabels]

    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))

    x_fs = max(4, min(10, int(120 / max(1, n_cols))))
    y_fs = max(4, min(10, int(120 / max(1, n_rows))))

    ax.set_xticklabels(xlabels, fontsize=x_fs, rotation=90, ha="center", va="top")
    ax.set_yticklabels(ylabels, fontsize=y_fs)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Relevance score", rotation=90)

    if annotate and (n_rows * n_cols) <= HEATMAP_MAX_ANNOTATE_CELLS:
        vmin, vmax = float(np.min(M)), float(np.max(M))
        mid = (vmin + vmax) / 2.0 if vmax > vmin else vmin
        for i in range(n_rows):
            for j in range(n_cols):
                val = M[i, j]
                if val == 0:
                    continue
                color = "white" if val > mid else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)

    ax.set_xlabel(matrix_df.columns.name or "Right variable")
    ax.set_ylabel(matrix_df.index.name or "Left variable")

    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)

def plot_heatmap_predictor_hierarchy(
    matrix_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    out_png: Path,
    title: str,
    xlabels_wrap: bool = True,
    annotate: bool = False,
) -> None:
    """
    FIXES:
      - Restores clear BLOCKS for domain + parent bars (not thin strips).
      - Makes sizing robust: annotation widths are preserved; when capped, heatmap width shrinks first.
      - Places BIO / PSYCHO / SOCIAL labels on the FAR LEFT, spanning their full blocks.
    """
    # --- Align meta_df to matrix_df row order robustly (prevents striped bars from accidental misalignment) ---
    meta = meta_df.copy()
    row_idx = matrix_df.index.astype(str).tolist()
    if "row_label_full" in meta.columns:
        meta_rows = meta["row_label_full"].astype(str).tolist()
        if meta_rows != row_idx and len(meta_rows) == len(row_idx):
            tmp = pd.DataFrame({"row_label_full": row_idx, "_pos": np.arange(len(row_idx))})
            meta = tmp.merge(meta, on="row_label_full", how="left", sort=False)
            meta = meta.sort_values("_pos").drop(columns=["_pos"]).reset_index(drop=True)

    # Ensure required columns exist (safe defaults)
    if "domain" not in meta.columns:
        meta["domain"] = "UNMAPPED"
    if "parent" not in meta.columns:
        meta["parent"] = "UNMAPPED"
    if "leaf" not in meta.columns:
        meta["leaf"] = meta.get("row_label_full", pd.Series(row_idx)).astype(str)
    if "id" not in meta.columns:
        meta["id"] = 10_000_000

    meta["domain"] = meta["domain"].fillna("UNMAPPED").astype(str)
    meta["parent"] = meta["parent"].fillna("UNMAPPED").astype(str)
    meta["leaf"] = meta["leaf"].fillna(meta.get("row_label_full", pd.Series(row_idx))).astype(str)
    meta["id"] = pd.to_numeric(meta["id"], errors="coerce").fillna(10_000_000).astype(int)

    M = np.nan_to_num(matrix_df.values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    n_rows, n_cols = matrix_df.shape

    domains = meta["domain"].astype(str).tolist()
    parents = meta["parent"].astype(str).tolist()

    # --------------------------
    # Dynamic horizontal sizing (PRESERVE annotation widths; shrink heatmap first if capped)
    # --------------------------
    PARENT_FONTSIZE = 8
    DOMAIN_FONTSIZE = 10

    max_parent_chars = max((len(str(p)) for p in parents), default=10)

    # character width approximation: ~0.65*fontsize points; 72pt = 1 inch
    char_inch = (0.65 * PARENT_FONTSIZE) / 72.0
    par_inch_desired = max(3.6, (char_inch * max_parent_chars + 0.9) * 1.05)  # +5% requirement

    # Make domain labels clearly readable at the FAR LEFT
    dom_lbl_inch = 1.9
    dom_bar_inch = 0.50

    cb_inch = 0.65

    hm_desired = max(6.0, 0.26 * n_cols)
    hm_min = 4.5

    # Fit within MAX_FIGSIZE_INCH by shrinking heatmap first
    fixed = dom_lbl_inch + dom_bar_inch + par_inch_desired + cb_inch
    if fixed + hm_desired <= MAX_FIGSIZE_INCH:
        par_inch = par_inch_desired
        hm_inch = hm_desired
        fig_w = fixed + hm_inch
    else:
        par_inch = par_inch_desired
        hm_inch = max(hm_min, MAX_FIGSIZE_INCH - fixed)
        # If still impossible, now shrink parent width (last resort) but keep it non-tiny
        if hm_inch <= hm_min and (dom_lbl_inch + dom_bar_inch + par_inch + cb_inch + hm_inch) > MAX_FIGSIZE_INCH:
            par_min = 2.6
            par_inch = max(par_min, MAX_FIGSIZE_INCH - (dom_lbl_inch + dom_bar_inch + cb_inch + hm_inch))
        fig_w = dom_lbl_inch + dom_bar_inch + par_inch + hm_inch + cb_inch

    # Vertical sizing
    fig_h = min(MAX_FIGSIZE_INCH, max(8.0, 0.24 * n_rows))

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        1, 5,
        width_ratios=[dom_lbl_inch, dom_bar_inch, par_inch, hm_inch, cb_inch],
        wspace=0.02
    )

    # Create bar axis first (sharey baseline)
    ax_dom_bar = fig.add_subplot(gs[0, 1])
    ax_dom_lbl = fig.add_subplot(gs[0, 0], sharey=ax_dom_bar)
    ax_par = fig.add_subplot(gs[0, 2], sharey=ax_dom_bar)
    ax_hm = fig.add_subplot(gs[0, 3], sharey=ax_dom_bar)
    ax_cb = fig.add_subplot(gs[0, 4])

    # --------------------------
    # Domain bar (BLOCKS)
    # --------------------------
    dom_list = ["BIO", "PSYCHO", "SOCIAL", "UNMAPPED"]
    dom_to_i = {d: i for i, d in enumerate(dom_list)}
    dom_ids = np.array([dom_to_i.get(d, dom_to_i["UNMAPPED"]) for d in domains], dtype=int).reshape(-1, 1)
    dom_cmap = ListedColormap([DOMAIN_COLORS[d] for d in dom_list])

    ax_dom_bar.imshow(dom_ids, aspect="auto", interpolation="nearest", cmap=dom_cmap)

    # --------------------------
    # Parent bar (BLOCKS, many colors, no cycling)
    # --------------------------
    parent_unique = []
    seen = set()
    for d, p in zip(domains, parents):
        key = f"{d}::{p}"
        if key not in seen:
            seen.add(key)
            parent_unique.append(key)

    par_to_i = {k: i for i, k in enumerate(parent_unique)}
    par_ids = np.array([par_to_i.get(f"{d}::{p}", -1) for d, p in zip(domains, parents)], dtype=int).reshape(-1, 1)

    par_colors = distinct_colors(len(parent_unique), cmap_name="gist_ncar")
    par_cmap = ListedColormap(par_colors if par_colors else [(0.5, 0.5, 0.5, 1.0)])

    ax_par.imshow(par_ids, aspect="auto", interpolation="nearest", cmap=par_cmap)

    # Hide frames/ticks for annotation axes; domain-label axis is text-only
    for ax in (ax_dom_lbl, ax_dom_bar, ax_par):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

    # --------------------------
    # Heatmap
    # --------------------------
    im = ax_hm.imshow(M, aspect="auto", interpolation="nearest")
    ax_hm.set_title(title, fontsize=14, pad=12)

    # X labels
    xlabels = matrix_df.columns.astype(str).tolist()
    if xlabels_wrap:
        xlabels = [wrap_label(x) for x in xlabels]

    # --- Y labels: ONLY leaf on heatmap axis (no repetition in hierarchy bars) ---
    ylabels = []
    for _, r in meta.iterrows():
        leaf = str(r["leaf"])
        pid = int(r["id"])
        if pid >= 10_000_000:
            ylabels.append(shorten_label(leaf, max_len=70))
        else:
            ylabels.append(f"{leaf} (ID:{pid})")

    ax_hm.set_xticks(np.arange(n_cols))
    ax_hm.set_yticks(np.arange(n_rows))

    x_fs = max(4, min(10, int(120 / max(1, n_cols))))
    y_fs = max(5, min(10, int(150 / max(1, n_rows))))

    ax_hm.set_xticklabels(xlabels, fontsize=x_fs, rotation=90, ha="center", va="top")
    ax_hm.set_yticklabels([wrap_label(y, 55) for y in ylabels], fontsize=y_fs)

    ax_hm.set_xlabel(matrix_df.columns.name or "Criterion cluster_id")
    #ax_hm.set_ylabel("Predictors (hierarchy-ordered)")

    cbar = fig.colorbar(im, cax=ax_cb)
    cbar.set_label("Relevance score", rotation=90)

    # Force consistent y-limits (shared axes can get overwritten by later imshow calls)
    ax_dom_bar.set_ylim(n_rows - 0.5, -0.5)

    # --------------------------
    # Block helpers
    # --------------------------
    def blocks(vals: List[str]) -> List[Tuple[int, int, str]]:
        out = []
        if not vals:
            return out
        start = 0
        cur = vals[0]
        for i in range(1, len(vals)):
            if vals[i] != cur:
                out.append((start, i - 1, cur))
                start = i
                cur = vals[i]
        out.append((start, len(vals) - 1, cur))
        return out

    parent_keys = [f"{d}::{p}" for d, p in zip(domains, parents)]
    parent_blocks = blocks(parent_keys)
    domain_blocks = blocks(domains)

    # separators at parent boundaries (draw across all annotation + heatmap)
    for (s, _e, _k) in parent_blocks:
        if s > 0:
            y = s - 0.5
            ax_hm.axhline(y=y, color="white", linewidth=0.8, alpha=0.9)
            ax_par.axhline(y=y, color="white", linewidth=0.8, alpha=0.9)
            ax_dom_bar.axhline(y=y, color="white", linewidth=0.8, alpha=0.9)
            ax_dom_lbl.axhline(y=y, color="white", linewidth=0.8, alpha=0.0)  # invisible (keeps alignment)

    # --------------------------
    # Parent labels (second layer)
    # --------------------------
    MIN_ROWS_FOR_PARENT_LABEL = 4
    tr_par = blended_transform_factory(ax_par.transAxes, ax_par.transData)
    for (s, e, k) in parent_blocks:
        if (e - s + 1) < MIN_ROWS_FOR_PARENT_LABEL:
            continue
        mid = (s + e) / 2.0
        try:
            _d, _p = k.split("::", 1)
        except Exception:
            _p = k

        ax_par.text(
            0.02, mid,
            _p,
            ha="left", va="center",
            fontsize=PARENT_FONTSIZE,
            color="black",
            transform=tr_par,
            clip_on=False,
            bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1.0)
        )

    # --------------------------
    # Domain labels on FAR LEFT (first layer)
    # --------------------------
    tr_domlbl = blended_transform_factory(ax_dom_lbl.transAxes, ax_dom_lbl.transData)
    for (s, e, d) in domain_blocks:
        mid = (s + e) / 2.0
        d_disp = DOMAIN_DISPLAY.get(d, d)
        ax_dom_lbl.text(
            0.02, mid,
            d_disp,
            ha="left", va="center",
            fontsize=DOMAIN_FONTSIZE,
            fontweight="bold",
            rotation=0,
            color="black",
            transform=tr_domlbl,
            clip_on=False,
        )


    # keep domain-label axis completely clean
    ax_dom_lbl.set_axis_off()

    # --------------------------
    # Optional cell annotations
    # --------------------------
    if annotate and (n_rows * n_cols) <= HEATMAP_MAX_ANNOTATE_CELLS:
        vmin, vmax = float(np.min(M)), float(np.max(M))
        midv = (vmin + vmax) / 2.0 if vmax > vmin else vmin
        for i in range(n_rows):
            for j in range(n_cols):
                val = M[i, j]
                if val == 0:
                    continue
                color = "white" if val > midv else "black"
                ax_hm.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)

    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)


# -----------------------------
# CLUSTERING
# -----------------------------

def bicluster_matrix(matrix_df: pd.DataFrame, out_dir: Path) -> Dict[str, object]:
    n_rows, n_cols = matrix_df.shape
    if n_rows < MIN_DIM_FOR_CLUSTERING or n_cols < MIN_DIM_FOR_CLUSTERING:
        return {"skipped": True, "reason": f"Matrix too small for clustering ({n_rows}x{n_cols})."}

    M = np.nan_to_num(matrix_df.values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)

    eps = 1e-12
    M2 = M + eps

    svd_dim = int(min(SVD_COMPONENTS, max(2, min(n_rows, n_cols) - 1)))
    svd = TruncatedSVD(n_components=svd_dim, random_state=RANDOM_SEED)
    row_emb = svd.fit_transform(M2)
    row_emb = np.nan_to_num(row_emb, nan=0.0, posinf=0.0, neginf=0.0)
    row_emb = StandardScaler().fit_transform(row_emb)

    k_max = min(K_SEARCH_MAX, min(n_rows, n_cols) - 1)
    k = choose_k_by_silhouette(row_emb, k_max=k_max)
    k = max(2, min(MAX_CLUSTERS, k))

    model = SpectralCoclustering(n_clusters=k, random_state=RANDOM_SEED)
    model.fit(M2)

    row_labels = model.row_labels_.astype(int)
    col_labels = model.column_labels_.astype(int)

    row_df = pd.DataFrame({"row_label": matrix_df.index.astype(str), "cluster": row_labels})
    col_df = pd.DataFrame({"col_label": matrix_df.columns.astype(str), "cluster": col_labels})

    row_df.to_csv(out_dir / "row_clusters.csv", index=False)
    col_df.to_csv(out_dir / "col_clusters.csv", index=False)

    row_order = np.argsort(row_labels, kind="stable")
    col_order = np.argsort(col_labels, kind="stable")

    clustered = matrix_df.iloc[row_order, :].iloc[:, col_order]
    clustered.to_csv(out_dir / "clustered_matrix.csv")

    return {
        "skipped": False,
        "n_clusters": int(k),
        "row_order": row_order.tolist(),
        "col_order": col_order.tolist(),
    }


# -----------------------------
# OPTIONAL UMAP + HDBSCAN
# -----------------------------

def _pip_install(packages: List[str]) -> bool:
    try:
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + packages
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return r.returncode == 0
    except Exception:
        return False

def ensure_umap_hdbscan_available() -> bool:
    if not TRY_UMAP_HDBSCAN:
        return False

    try:
        import umap  # noqa: F401
        import hdbscan  # noqa: F401
        return True
    except Exception:
        if not AUTO_INSTALL_UMAP_HDBSCAN:
            return False

        ok = _pip_install([
            "umap-learn>=0.5.5",
            "hdbscan>=0.8.40",
            "scikit-learn<1.8",
        ])
        if not ok:
            return False

        try:
            import umap  # noqa: F401
            import hdbscan  # noqa: F401
            return True
        except Exception:
            return False

def try_umap_hdbscan(
    matrix_df: pd.DataFrame,
    results_dir: Path,
    plots_dir: Path,
    mapping_name: str,
) -> Optional[Dict[str, object]]:
    if not TRY_UMAP_HDBSCAN:
        return None
    if not ensure_umap_hdbscan_available():
        return None

    import umap
    import hdbscan

    n_rows, n_cols = matrix_df.shape
    if n_rows < MIN_DIM_FOR_CLUSTERING or n_cols < MIN_DIM_FOR_CLUSTERING:
        return None

    M = np.nan_to_num(matrix_df.values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    eps = 1e-12
    M2 = M + eps

    svd_dim = int(min(SVD_COMPONENTS, max(2, min(n_rows, n_cols) - 1)))
    svd = TruncatedSVD(n_components=svd_dim, random_state=RANDOM_SEED)
    row_emb = svd.fit_transform(M2)
    col_emb = svd.components_.T * svd.singular_values_

    row_emb = np.nan_to_num(row_emb, nan=0.0, posinf=0.0, neginf=0.0)
    col_emb = np.nan_to_num(col_emb, nan=0.0, posinf=0.0, neginf=0.0)

    row_emb = StandardScaler().fit_transform(row_emb)
    col_emb = StandardScaler().fit_transform(col_emb)

    row_reducer = umap.UMAP(
        n_neighbors=min(30, max(5, n_rows // 10)),
        min_dist=0.05,
        n_components=2,
        metric="euclidean",
        random_state=RANDOM_SEED,
    )
    col_reducer = umap.UMAP(
        n_neighbors=min(30, max(5, n_cols // 10)),
        min_dist=0.05,
        n_components=2,
        metric="euclidean",
        random_state=RANDOM_SEED,
    )

    row_2d = row_reducer.fit_transform(row_emb)
    col_2d = col_reducer.fit_transform(col_emb)

    row_min_cs = max(5, n_rows // 25)
    col_min_cs = max(5, n_cols // 25)

    row_clusterer = hdbscan.HDBSCAN(min_cluster_size=row_min_cs)
    col_clusterer = hdbscan.HDBSCAN(min_cluster_size=col_min_cs)

    row_labels = row_clusterer.fit_predict(row_2d)
    col_labels = col_clusterer.fit_predict(col_2d)

    row_df = pd.DataFrame({
        "row_label": matrix_df.index.astype(str),
        "umap_x": row_2d[:, 0],
        "umap_y": row_2d[:, 1],
        "cluster": row_labels,
    })
    col_df = pd.DataFrame({
        "col_label": matrix_df.columns.astype(str),
        "umap_x": col_2d[:, 0],
        "umap_y": col_2d[:, 1],
        "cluster": col_labels,
    })

    row_df.to_csv(results_dir / "row_umap_hdbscan.csv", index=False)
    col_df.to_csv(results_dir / "col_umap_hdbscan.csv", index=False)

    def scatter(df_sc: pd.DataFrame, label_col: str, out_png: Path, title: str) -> None:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(df_sc["umap_x"], df_sc["umap_y"], s=12, c=df_sc["cluster"].astype(int), alpha=0.9)
        ax.set_title(title, fontsize=14, pad=10)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")

        try:
            for cl in sorted(df_sc["cluster"].unique()):
                if cl == -1:
                    continue
                sub = df_sc[df_sc["cluster"] == cl].copy()
                if len(sub) == 0:
                    continue
                cx, cy = sub["umap_x"].mean(), sub["umap_y"].mean()
                sub["d2"] = (sub["umap_x"] - cx) ** 2 + (sub["umap_y"] - cy) ** 2
                pick = sub.nsmallest(1, "d2")
                for _, r in pick.iterrows():
                    ax.text(r["umap_x"], r["umap_y"], shorten_label(r[label_col], 35), fontsize=7)
        except Exception:
            pass

        fig.tight_layout()
        fig.savefig(out_png, dpi=DPI)
        plt.close(fig)

    scatter(row_df, "row_label", plots_dir / "umap_hdbscan_rows_scatter.png", f"{mapping_name}: UMAP+HDBSCAN (rows)")
    scatter(col_df, "col_label", plots_dir / "umap_hdbscan_cols_scatter.png", f"{mapping_name}: UMAP+HDBSCAN (cols)")

    return {
        "method": "svd+umap+hdbscan",
        "svd_dim": int(svd_dim),
        "row_min_cluster_size": int(row_min_cs),
        "col_min_cluster_size": int(col_min_cs),
        "row_n_clusters_excl_noise": int(len(set(row_labels)) - (1 if -1 in set(row_labels) else 0)),
        "col_n_clusters_excl_noise": int(len(set(col_labels)) - (1 if -1 in set(col_labels) else 0)),
    }


# -----------------------------
# NETWORKS (PNG ONLY)
# -----------------------------

def build_filtered_edge_list_from_matrix(
    matrix_df: pd.DataFrame,
    quantile: float = EDGE_KEEP_QUANTILE,
    min_abs: float = EDGE_MIN_ABS,
    topk_per_node: int = TOPK_EDGES_PER_NODE,
) -> pd.DataFrame:
    M = np.nan_to_num(matrix_df.values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    rows = matrix_df.index.to_numpy()
    cols = matrix_df.columns.to_numpy()

    nz = np.argwhere(M > 0)
    if nz.size == 0:
        return pd.DataFrame(columns=["left", "right", "score"])

    scores = M[nz[:, 0], nz[:, 1]]
    thr = max(float(np.quantile(scores, quantile)), float(min_abs))

    keep = scores >= thr
    nz_keep = nz[keep]
    scores_keep = scores[keep]

    edges = pd.DataFrame({
        "left": rows[nz_keep[:, 0]].astype(str),
        "right": cols[nz_keep[:, 1]].astype(str),
        "score": scores_keep.astype(float),
    })

    if edges.empty:
        flat = [(str(rows[i]), str(cols[j]), float(M[i, j])) for i, j in nz]
        top = pd.DataFrame(flat, columns=["left", "right", "score"]).sort_values("score", ascending=False)
        return top.head(min(2000, len(top))).reset_index(drop=True)

    edges_top_left = edges.sort_values("score", ascending=False).groupby("left", as_index=False).head(topk_per_node)
    edges_top_right = edges.sort_values("score", ascending=False).groupby("right", as_index=False).head(topk_per_node)

    edges2 = pd.concat([edges_top_left, edges_top_right], ignore_index=True).drop_duplicates()
    edges2 = edges2.sort_values("score", ascending=False).reset_index(drop=True)
    return edges2

def plot_bipartite_network(
    edges_df: pd.DataFrame,
    left_clusters: Optional[Dict[str, int]],
    right_clusters: Optional[Dict[str, int]],
    out_png: Path,
    title: str,
) -> None:
    if edges_df.empty:
        return

    if len(edges_df) > MAX_EDGES_PLOTTED:
        edges_df = edges_df.nlargest(MAX_EDGES_PLOTTED, "score").copy()

    G = nx.Graph()
    left_nodes = sorted(edges_df["left"].astype(str).unique().tolist())
    right_nodes = sorted(edges_df["right"].astype(str).unique().tolist())

    for n in left_nodes:
        G.add_node(f"L::{n}", bipartite=0, label_full=n, label_short=shorten_label(n))
    for n in right_nodes:
        G.add_node(f"R::{n}", bipartite=1, label_full=n, label_short=shorten_label(n))

    for _, r in edges_df.iterrows():
        G.add_edge(f"L::{str(r['left'])}", f"R::{str(r['right'])}", weight=float(r["score"]))

    left_strength = {n: 0.0 for n in left_nodes}
    right_strength = {n: 0.0 for n in right_nodes}
    for _, r in edges_df.iterrows():
        left_strength[str(r["left"])] += float(r["score"])
        right_strength[str(r["right"])] += float(r["score"])

    left_sorted = sorted(left_nodes, key=lambda x: (-left_strength.get(x, 0.0), x))
    right_sorted = sorted(right_nodes, key=lambda x: (-right_strength.get(x, 0.0), x))

    pos = {}
    for i, n in enumerate(left_sorted):
        pos[f"L::{n}"] = (0.0, -i)
    for i, n in enumerate(right_sorted):
        pos[f"R::{n}"] = (1.0, -i)

    def node_cluster(node_key: str) -> int:
        if node_key.startswith("L::") and left_clusters is not None:
            return int(left_clusters.get(node_key[3:], -1))
        if node_key.startswith("R::") and right_clusters is not None:
            return int(right_clusters.get(node_key[3:], -1))
        return -1

    nodes = list(G.nodes())
    clusters = np.array([node_cluster(n) for n in nodes], dtype=int)

    cmap = plt.get_cmap("tab20")
    unique_clusters = sorted(set(clusters.tolist()))
    cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
    color_vals = [cluster_to_idx[c] for c in clusters]
    node_colors = [cmap((v % 20) / 20.0) for v in color_vals]

    weights = np.array([G[u][v]["weight"] for u, v in G.edges()], dtype=float)
    wmin, wmax = float(weights.min()), float(weights.max())
    if wmax > wmin:
        widths = 0.2 + 3.0 * (weights - wmin) / (wmax - wmin)
        alphas = 0.1 + 0.7 * (weights - wmin) / (wmax - wmin)
    else:
        widths = np.full_like(weights, 1.0)
        alphas = np.full_like(weights, 0.5)

    nL, nR = len(left_nodes), len(right_nodes)
    h = min(MAX_FIGSIZE_INCH, max(8.0, 0.12 * (nL + nR)))
    w = 14.0
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_title(title, fontsize=14, pad=12)

    for (u, v), lw, a in zip(G.edges(), widths, alphas):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], linewidth=float(lw), alpha=float(a))

    xs = [pos[n][0] for n in nodes]
    ys = [pos[n][1] for n in nodes]
    ax.scatter(xs, ys, s=24, c=node_colors, edgecolors="none", alpha=0.9)

    left_top = set(left_sorted[: max(5, min(NETWORK_NODE_LABEL_LIMIT // 2, len(left_sorted)))])
    right_top = set(right_sorted[: max(5, min(NETWORK_NODE_LABEL_LIMIT // 2, len(right_sorted)))])

    for n in nodes:
        full = G.nodes[n]["label_full"]
        short = G.nodes[n]["label_short"]
        if n.startswith("L::") and full in left_top:
            ax.text(pos[n][0] - 0.02, pos[n][1], short, ha="right", va="center", fontsize=7)
        elif n.startswith("R::") and full in right_top:
            ax.text(pos[n][0] + 0.02, pos[n][1], short, ha="left", va="center", fontsize=7)

    ax.set_axis_off()
    ax.set_xlim(-0.35, 1.35)
    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)

def plot_cluster_subnetworks(
    matrix_df: pd.DataFrame,
    row_clusters: Dict[str, int],
    col_clusters: Dict[str, int],
    n_clusters: int,
    plots_dir: Path,
    title_prefix: str,
) -> None:
    for c in range(n_clusters):
        rows = [r for r, rc in row_clusters.items() if rc == c]
        cols = [k for k, cc in col_clusters.items() if cc == c]
        if len(rows) == 0 or len(cols) == 0:
            continue

        sub = matrix_df.loc[rows, cols]
        edges = build_filtered_edge_list_from_matrix(
            sub,
            quantile=0.80,
            min_abs=EDGE_MIN_ABS,
            topk_per_node=max(5, TOPK_EDGES_PER_NODE // 2),
        )
        if edges.empty:
            continue

        out_png = plots_dir / f"network_cluster_{c}.png"
        plot_bipartite_network(
            edges_df=edges,
            left_clusters=None,
            right_clusters=None,
            out_png=out_png,
            title=f"{title_prefix} — bicluster {c} (rows={len(rows)}, cols={len(cols)})",
        )


# -----------------------------
# PIPELINE PER MAPPING
# -----------------------------

def process_mapping(cfg: dict, base_out: Path, predictor_hierarchy: Dict[str, Any]) -> None:
    name = cfg["name"]
    in_path = Path(cfg["csv_path"]).expanduser()

    mapping_dir = base_out / name
    results_dir = mapping_dir / "results"
    plots_dir = mapping_dir / "plots"
    data_dir = mapping_dir / "data"
    for d in [results_dir, plots_dir, data_dir]:
        safe_mkdir(d)

    log_path = results_dir / "run_log.txt"

    def log(msg: str) -> None:
        print(f"[{name}] {msg}")
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log(f"Input: {in_path}")

    if not in_path.exists():
        log("ERROR: input file does not exist. Skipping.")
        return

    try:
        df = pd.read_csv(in_path)
    except Exception as e:
        log(f"ERROR: could not read CSV: {e}. Skipping.")
        return

    left_col = cfg.get("left_label_col", None)
    right_col = cfg.get("right_label_col", None)
    score_col = cfg.get("score_col", "score")

    if score_col not in df.columns:
        candidates = [c for c in df.columns if "score" in c.lower()]
        if candidates:
            score_col = candidates[0]
            log(f"Inferred score column: {score_col}")
        else:
            log(f"ERROR: score column not found. Columns: {list(df.columns)}. Skipping.")
            return

    if left_col is None or right_col is None:
        hint = None
        if "coping" in name:
            hint = "coping"
        elif "context" in name:
            hint = "context"
        elif "profile" in name:
            hint = "profile"
        elif "barrier" in name:
            hint = "barrier"
        left_col, right_col = infer_two_full_path_cols(df, preferred_left_hint=hint)
        log(f"Inferred label columns: left={left_col}, right={right_col}")

    if left_col not in df.columns or right_col not in df.columns:
        log(f"ERROR: label columns missing. left={left_col}, right={right_col}. Columns: {list(df.columns)}. Skipping.")
        return

    df.to_csv(data_dir / "edges_raw.csv", index=False)

    try:
        matrix_df = densify_matrix(df, left_col, right_col, score_col, agg="max")
    except Exception as e:
        log(f"ERROR: could not build matrix: {e}. Skipping.")
        return

    if name == "predictor_to_criterion":
        try:
            cols_sorted = sorted(matrix_df.columns.tolist(), key=lambda x: int(str(x)))
        except Exception:
            cols_sorted = sorted(matrix_df.columns.astype(str).tolist())
        matrix_df = matrix_df.loc[:, cols_sorted]

        ordered_rows, meta_df = order_predictor_rows_by_hierarchy(matrix_df.index.astype(str).tolist(), predictor_hierarchy)
        matrix_df = matrix_df.loc[ordered_rows, :]
        meta_df.to_csv(results_dir / "predictor_hierarchy_row_meta.csv", index=False)

    matrix_df.to_csv(data_dir / "matrix.csv")

    M = np.nan_to_num(matrix_df.values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    n_rows, n_cols = matrix_df.shape
    nnz = int(np.sum(M > 0))
    density = float(nnz / max(1, n_rows * n_cols))
    stats = {
        "mapping": name,
        "input_csv": str(in_path),
        "left_label_col": left_col,
        "right_label_col": right_col,
        "score_col": score_col,
        "n_left": n_rows,
        "n_right": n_cols,
        "nnz": nnz,
        "density": density,
        "score_min": float(np.min(M)) if M.size else 0.0,
        "score_max": float(np.max(M)) if M.size else 0.0,
        "score_mean_nonzero": float(np.mean(M[M > 0])) if nnz else 0.0,
    }
    save_json(results_dir / "summary.json", stats)

    nz = np.argwhere(M > 0)
    if nz.size:
        top = [(str(matrix_df.index[i]), str(matrix_df.columns[j]), float(M[i, j])) for i, j in nz]
        top_df = pd.DataFrame(top, columns=["left_label", "right_label", "score"]).sort_values("score", ascending=False)
        top_df.head(300).to_csv(results_dir / "top_edges_300.csv", index=False)
    else:
        pd.DataFrame(columns=["left_label", "right_label", "score"]).to_csv(results_dir / "top_edges_300.csv", index=False)

    annotate = (n_rows * n_cols) <= HEATMAP_MAX_ANNOTATE_CELLS

    if name == "predictor_to_criterion":
        meta_df = pd.read_csv(results_dir / "predictor_hierarchy_row_meta.csv")
        plot_heatmap_predictor_hierarchy(
            matrix_df=matrix_df,
            meta_df=meta_df,
            out_png=plots_dir / "heatmap_predictor_hierarchy_full_labels.png",
            title=f"{name}: predictors (hierarchy) × criterion clusters (scores)",
            xlabels_wrap=True,
            annotate=annotate,
        )
        plot_heatmap_predictor_hierarchy(
            matrix_df=matrix_df,
            meta_df=meta_df,
            out_png=plots_dir / "heatmap_predictor_hierarchy_compact.png",
            title=f"{name}: hierarchy heatmap (compact labels)",
            xlabels_wrap=False,
            annotate=False,
        )
    else:
        plot_heatmap_basic(
            matrix_df,
            out_png=plots_dir / "heatmap_hclust_full_labels.png",
            title=f"{name}: {left_col} × {right_col} (scores)",
            wrap=True,
            annotate=annotate,
            reorder_rows=True,
            reorder_cols=True,
        )

        short_df = matrix_df.copy()
        short_df.index = [shorten_label(x, max_len=60) for x in short_df.index]
        short_df.columns = [shorten_label(x, max_len=60) for x in short_df.columns]
        plot_heatmap_basic(
            short_df,
            out_png=plots_dir / "heatmap_hclust_short_labels.png",
            title=f"{name}: shortened labels (scores)",
            wrap=True,
            annotate=annotate,
            reorder_rows=True,
            reorder_cols=True,
        )

    cluster_out = bicluster_matrix(matrix_df, out_dir=results_dir)
    save_json(results_dir / "clustering_bicluster.json", cluster_out)

    if not cluster_out.get("skipped", True):
        clustered_df = pd.read_csv(results_dir / "clustered_matrix.csv", index_col=0)
        plot_heatmap_basic(
            clustered_df,
            out_png=plots_dir / "heatmap_clustered_blocks_full_labels.png",
            title=f"{name}: Spectral Co-Clustering blocks (k={cluster_out['n_clusters']})",
            wrap=True,
            annotate=False,
            reorder_rows=False,
            reorder_cols=False,
        )

    umap_out = try_umap_hdbscan(matrix_df, results_dir=results_dir, plots_dir=plots_dir, mapping_name=name)
    if umap_out is not None:
        save_json(results_dir / "clustering_umap_hdbscan.json", umap_out)
        log("UMAP+HDBSCAN clustering saved.")
    else:
        log("UMAP+HDBSCAN not available (install failed or disabled).")

    edges_net = build_filtered_edge_list_from_matrix(matrix_df)
    edges_net.to_csv(data_dir / "edges_for_network.csv", index=False)

    G_out = nx.Graph()
    for _, r in edges_net.iterrows():
        G_out.add_edge(str(r["left"]), str(r["right"]), weight=float(r["score"]))
    nx.write_graphml(G_out, data_dir / "network.graphml")

    left_clusters = None
    right_clusters = None
    if not cluster_out.get("skipped", True):
        row_clusters_df = pd.read_csv(results_dir / "row_clusters.csv")
        col_clusters_df = pd.read_csv(results_dir / "col_clusters.csv")
        left_clusters = {str(r["row_label"]): int(r["cluster"]) for _, r in row_clusters_df.iterrows()}
        right_clusters = {str(r["col_label"]): int(r["cluster"]) for _, r in col_clusters_df.iterrows()}

    plot_bipartite_network(
        edges_df=edges_net,
        left_clusters=left_clusters,
        right_clusters=right_clusters,
        out_png=plots_dir / "network_bipartite.png",
        title=f"{name}: bipartite network (filtered edges)",
    )

    if left_clusters is not None and right_clusters is not None:
        plot_cluster_subnetworks(
            matrix_df=matrix_df,
            row_clusters=left_clusters,
            col_clusters=right_clusters,
            n_clusters=int(cluster_out["n_clusters"]),
            plots_dir=plots_dir,
            title_prefix=name,
        )

    node_labels = []
    for lbl in matrix_df.index.astype(str).tolist():
        node_labels.append({"side": "left", "label_full": lbl, "label_short": shorten_label(lbl)})
    for lbl in matrix_df.columns.astype(str).tolist():
        node_labels.append({"side": "right", "label_full": lbl, "label_short": shorten_label(lbl)})
    pd.DataFrame(node_labels).to_csv(data_dir / "node_labels_full_to_short.csv", index=False)

    log("DONE.")


def main() -> None:
    np.random.seed(RANDOM_SEED)
    safe_mkdir(BASE_OUT)

    predictor_hierarchy = load_predictor_hierarchy(PREDICTOR_HIERARCHY_PATH)
    if not predictor_hierarchy.get("leaf_by_norm"):
        print(f"[GLOBAL] Predictor hierarchy not loaded or empty: {PREDICTOR_HIERARCHY_PATH}")
        print("[GLOBAL] predictor_to_criterion will still run, but without hierarchy-based grouping.")

    if TRY_UMAP_HDBSCAN:
        ok = ensure_umap_hdbscan_available()
        if not ok:
            print("[GLOBAL] UMAP+HDBSCAN not available (could not import or auto-install).")

    for cfg in INPUTS:
        process_mapping(cfg, BASE_OUT, predictor_hierarchy)

    print(f"\nAll outputs saved under:\n  {BASE_OUT}\n")


if __name__ == "__main__":
    main()

#
# TODO: later apply similar cluster analyses to embedding-based heatmaps
# TODO: later ensure to use some hierarchical visualization logic --> automate later for ALL maps based on full PHOENIX ontology ; instead of only explicit PC
# TODO: optionally evaluate seaborn.clustermap row_colors for hierarchy bars (adds dependency; avoid unless needed)
# TODO: fix ongoing plotting issues with the hierarchical visualization
