"""
Intra-ontological mapping visualisations for the PHOENIX ontology.

Generates three thesis-grade composite figures (and individual sub-plots):

  • figure_ontology_diagnostics.png      (Fig 12, 4 panels: A-D)
        A  Cross-ontology mapping diagram (from src/backend/SystemComponents/...)
        B  Depth distribution per sub-ontology
        C  Sub-tree sizes and leaf ratio per sub-ontology
        D  Branching factor distribution (aggregated, log-y)

  • figure_predictor_criterion_diagnostics.png  (Fig 21, 4 panels: A-D)
        A  Hierarchical BPS → secondary-domain density
        B  Score histogram of observed (non-zero) edges
        C  Full predictor × criterion-cluster relevance heatmap (dendrograms)
        D  Score distributions across the seven cross-ontology mappings

  • figure_intra_ontological_mappings.png    (Fig 22, 8 panels: A-H)
        A  Predictor → Criterion        E  Barrier  → Predictor
        B  Context   → Barrier          F  Context  → Predictor
        C  Coping    → Barrier          G  Profile  → Predictor
        D  Profile   → Barrier          H  Sparsity & coverage diagnostic

For each "mapping" panel a Ward-linkage dendrogram is mounted on the top
(column tree) and on the right (row tree); rows and columns are reordered
by optimal-leaf-ordering so semantically similar entries cluster together.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.cluster.hierarchy import dendrogram, linkage, optimal_leaf_ordering
from scipy.spatial.distance import pdist


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[5]
MAP_ROOT = REPO_ROOT / "src/backend/utils/official/ontology_mappings"
CLUSTER_ROOT = REPO_ROOT / "src/backend/utils/official/cluster_criterions/results"
OUT_DIR = (
    REPO_ROOT
    / "src/backend/utils/official/overall_mapping_analyses/results/intra_ontological_mappings"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Existing image assets that we re-use in the composite figures
PRIMARY_DIR = (
    REPO_ROOT
    / "src/backend/utils/official/overall_mapping_analyses/results/overall_mapping_analysis/plots/PRIMARY"
)
ONTO_SUMMARY_DIR = (
    REPO_ROOT
    / "src/backend/utils/other/exploratory/summarize_ontology/plots/20260119_231403"
)
ONTOLOGY_MAPPING_DIAGRAM = (
    REPO_ROOT
    / "src/backend/SystemComponents/PHOENIX_ontology/aggretated/ontology_with_mapping.png"
)

DENSITY_PNG = PRIMARY_DIR / "hierarchical_BPS_secondary_density_points.png"
HISTOGRAM_PNG = PRIMARY_DIR / "score_histogram_observed_edges.png"


MAPPINGS = [
    {
        "key": "pred_to_crit",
        "dense_csv": "CRITERION/predictor_to_criterion/results/gpt-5-nano/predictor_to_criterion_map_dense.csv",
        "row_labels": "cluster_id",
        "col_labels": "CRITERION/predictor_to_criterion/input_lists/predictors_list.txt",
        "title": "Predictor → Criterion",
        "row_role": "Criterion cluster",
        "col_role": "Predictor leaf",
        "row_cap": 80, "col_cap": 80,
    },
    {
        "key": "ctx_to_barr",
        "dense_csv": "HAPA/context_to_barrier/results/gpt-5-nano/context_to_barrier_map_dense.csv",
        "row_labels": "HAPA/context_to_barrier/input_lists/barriers_list.txt",
        "col_labels": "HAPA/context_to_barrier/input_lists/contextual_factors_list.txt",
        "title": "Context → Barrier",
        "row_role": "Barrier", "col_role": "Context",
        "row_cap": 60, "col_cap": 60,
    },
    {
        "key": "cop_to_barr",
        "dense_csv": "HAPA/coping_to_barrier/results/gpt-5-nano/coping_to_barrier_map_dense.csv",
        "row_labels": "HAPA/coping_to_barrier/input_lists/coping_options_list.txt",
        "col_labels": "HAPA/coping_to_barrier/input_lists/barriers_list.txt",
        "title": "Coping → Barrier",
        "row_role": "Coping strategy", "col_role": "Barrier",
        "row_cap": 60, "col_cap": 60,
    },
    {
        "key": "prof_to_barr",
        "dense_csv": "HAPA/profile_to_barrier/results/gpt-5-nano/profile_to_barrier_map_dense.csv",
        "row_labels": "HAPA/profile_to_barrier/input_lists/barriers_list.txt",
        "col_labels": "HAPA/profile_to_barrier/input_lists/person_factors_list.txt",
        "title": "Profile → Barrier",
        "row_role": "Barrier", "col_role": "Profile attribute",
        "row_cap": 60, "col_cap": 80,
    },
    {
        "key": "barr_to_pred",
        "dense_csv": "PREDICTOR/barrier_to_predictor/results/gpt-5-nano/predictor_to_barrier_map_dense.csv",
        "row_labels": "PREDICTOR/barrier_to_predictor/input_lists/predictors_list.txt",
        "col_labels": "PREDICTOR/barrier_to_predictor/input_lists/barriers_list.txt",
        "title": "Barrier → Predictor",
        "row_role": "Predictor leaf", "col_role": "Barrier",
        "row_cap": 80, "col_cap": 60,
    },
    {
        "key": "ctx_to_pred",
        "dense_csv": "PREDICTOR/context_to_predictor/results/gpt-5-nano/predictor_to_context_map_dense.csv",
        "row_labels": "PREDICTOR/context_to_predictor/input_lists/predictors_list.txt",
        "col_labels": "PREDICTOR/context_to_predictor/input_lists/contextual_factors_list.txt",
        "title": "Context → Predictor",
        "row_role": "Predictor leaf", "col_role": "Context",
        "row_cap": 80, "col_cap": 60,
    },
    {
        "key": "prof_to_pred",
        "dense_csv": "PREDICTOR/profile_to_predictor/results/gpt-5-nano/predictor_to_profile_map_dense.csv",
        "row_labels": "PREDICTOR/profile_to_predictor/input_lists/predictors_list.txt",
        "col_labels": "PREDICTOR/profile_to_predictor/input_lists/person_factors_list.txt",
        "title": "Profile → Predictor",
        "row_role": "Predictor leaf", "col_role": "Profile attribute",
        "row_cap": 80, "col_cap": 80,
    },
]


CMAP = LinearSegmentedColormap.from_list(
    "phoenix_mapping",
    ["#1f3a68", "#3b6ea8", "#8eb6d8", "#dde9f4", "#f7f7f7",
     "#fae0c1", "#e89c6a", "#c7693d", "#8a3e1f"],
)


# ---------------------------------------------------------------------------
# Label parsing helpers
# ---------------------------------------------------------------------------

LEAF_LINE = re.compile(r"\(ID:(\d+)\)\s*$")
INLINE_ID = re.compile(r"^\s*(\d+)[\s.\-:]\s*(.+?)\s*$")


def _short(label: str, n: int = 4) -> str:
    if label is None:
        return ""
    s = str(label).replace("_", " ").strip()
    s = re.sub(r"^\[[A-Z]+\]\s*", "", s)
    tokens = s.split()
    return " ".join(tokens[-n:]) if len(tokens) > n else s


def _load_id_label_map_from_tree(path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = LEAF_LINE.search(raw)
        if not m:
            continue
        leaf_id = int(m.group(1))
        prefix = raw[: m.start()].strip()
        prefix = re.sub(r"^[\s│├└─]*", "", prefix)
        out[leaf_id] = prefix
    return out


def _load_id_label_map_from_flat(path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    if not path.exists():
        return out
    lines = [ln for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
    matched = False
    for ln in lines:
        m = INLINE_ID.match(ln)
        if m:
            out[int(m.group(1))] = m.group(2)
            matched = True
    if matched:
        return out
    for i, ln in enumerate(lines):
        out[i] = ln.strip()
    return out


def _load_labels(rel_path: str) -> dict[int, str]:
    fpath = MAP_ROOT / rel_path
    out = _load_id_label_map_from_tree(fpath)
    return out if out else _load_id_label_map_from_flat(fpath)


def _load_criterion_cluster_labels() -> dict[str, str]:
    fpath = CLUSTER_ROOT / "04_semantically_clustered_items.json"
    labels: dict[str, str] = {}
    if not fpath.exists():
        return labels
    data = json.loads(fpath.read_text(encoding="utf-8", errors="ignore"))
    clusters = data.get("clusters", {})
    gpt_lbls = CLUSTER_ROOT / "05_GPT_labeled_split_items.txt"
    if gpt_lbls.exists():
        for i, ln in enumerate(
            [ln.strip() for ln in gpt_lbls.read_text(encoding="utf-8").splitlines() if ln.strip()]
        ):
            labels[f"c{i}"] = ln[:60]
    for cid, info in clusters.items():
        if cid in labels:
            continue
        items = info.get("items") or []
        if items:
            first = items[0].split(" (context")[0].split(" — criterion")[0]
            labels[cid] = first.replace("_", " ")[:60]
        else:
            labels[cid] = cid
    return labels


# ---------------------------------------------------------------------------
# Matrix loading
# ---------------------------------------------------------------------------

def _load_dense(rel_path: str) -> pd.DataFrame:
    df = pd.read_csv(MAP_ROOT / rel_path, index_col=0)
    return df.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _apply_labels(df: pd.DataFrame, row_spec: str, col_spec: str) -> pd.DataFrame:
    if col_spec and col_spec != "cluster_id":
        col_map = _load_labels(col_spec)
        if col_map:
            df.columns = [col_map.get(int(c), str(c)) if str(c).isdigit() else str(c) for c in df.columns]
    if row_spec == "cluster_id":
        cluster_lbls = _load_criterion_cluster_labels()
        df.index = [cluster_lbls.get(str(idx), str(idx)) for idx in df.index]
    elif row_spec:
        row_map = _load_labels(row_spec)
        if row_map:
            df.index = [row_map.get(int(r), str(r)) if str(r).isdigit() else str(r) for r in df.index]
    return df


def _trim_top_k(df: pd.DataFrame, row_cap: int, col_cap: int) -> pd.DataFrame:
    df = df.loc[df.abs().sum(axis=1) > 0, df.abs().sum(axis=0) > 0]
    if row_cap and df.shape[0] > row_cap:
        df = df.loc[df.abs().sum(axis=1).sort_values(ascending=False).head(row_cap).index]
    if col_cap and df.shape[1] > col_cap:
        df = df.loc[:, df.abs().sum(axis=0).sort_values(ascending=False).head(col_cap).index]
    return df


# ---------------------------------------------------------------------------
# Plot primitives (clustermap)
# ---------------------------------------------------------------------------

def _render_clustermap_into_gridcell(
    fig: plt.Figure,
    subplot_spec: gridspec.SubplotSpec,
    df: pd.DataFrame,
    *,
    panel_letter: Optional[str] = None,
    panel_title: Optional[str] = None,
    show_x_labels: bool = False,
    show_y_labels: bool = False,
    vmax: Optional[float] = None,
) -> "plt.AxesImage":
    """
    Render a single clustermap (dendrogram-top + dendrogram-right + heatmap)
    inside a grid cell of a parent figure. Returns the AxesImage for the
    heatmap (so a shared colourbar can be created later).
    """
    inner = gridspec.GridSpecFromSubplotSpec(
        nrows=2, ncols=2, subplot_spec=subplot_spec,
        height_ratios=[0.10, 1.0],
        width_ratios=[1.0, 0.07],
        hspace=0.01, wspace=0.01,
    )
    ax_top = fig.add_subplot(inner[0, 0])
    ax_heat = fig.add_subplot(inner[1, 0])
    ax_right = fig.add_subplot(inner[1, 1])

    # Column dendrogram (top)
    col_dist = pdist(df.T.values, metric="euclidean")
    Z_col = optimal_leaf_ordering(linkage(col_dist, method="ward"), col_dist)
    dend_col = dendrogram(
        Z_col, ax=ax_top, orientation="top",
        color_threshold=0, above_threshold_color="#6c6c6c", no_labels=True,
    )
    col_order = dend_col["leaves"]
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    for s in ax_top.spines.values():
        s.set_visible(False)

    # Row dendrogram (right)
    if df.shape[0] > 2:
        row_dist = pdist(df.values, metric="euclidean")
        Z_row = optimal_leaf_ordering(linkage(row_dist, method="ward"), row_dist)
        dend_row = dendrogram(
            Z_row, ax=ax_right, orientation="right",
            color_threshold=0, above_threshold_color="#6c6c6c", no_labels=True,
        )
        row_order = dend_row["leaves"]
    else:
        row_order = list(range(df.shape[0]))
    ax_right.set_xticks([])
    ax_right.set_yticks([])
    for s in ax_right.spines.values():
        s.set_visible(False)

    ordered = df.iloc[row_order, col_order]
    n_rows, n_cols = ordered.shape
    if vmax is None:
        vmax = max(50.0, float(np.nanpercentile(np.abs(ordered.values), 99)))
    norm = TwoSlopeNorm(vmin=0.0, vcenter=vmax / 2.0, vmax=vmax)
    im = ax_heat.imshow(
        ordered.values, cmap=CMAP, norm=norm, aspect="auto",
        interpolation="nearest", extent=(0, n_cols * 10, n_rows * 10, 0),
    )

    # Ticks: only the most informative ones, never overlapping
    if show_x_labels and n_cols <= 80:
        x_centers = np.arange(n_cols) * 10 + 5
        ax_heat.set_xticks(x_centers)
        ax_heat.set_xticklabels(
            [_short(c, 3) for c in ordered.columns],
            rotation=70, ha="right", fontsize=4.2,
        )
        ax_heat.tick_params(axis="x", pad=1, length=0)
    else:
        ax_heat.set_xticks([])
    if show_y_labels and n_rows <= 60:
        y_centers = np.arange(n_rows) * 10 + 5
        ax_heat.set_yticks(y_centers)
        ax_heat.set_yticklabels(
            [_short(r, 3) for r in ordered.index], fontsize=4.5,
        )
        ax_heat.tick_params(axis="y", pad=1, length=0)
    else:
        ax_heat.set_yticks([])
    for s in ax_heat.spines.values():
        s.set_visible(False)

    # Align dendrograms exactly to the heatmap extents
    ax_top.set_xlim(0, n_cols * 10)
    ax_top.margins(x=0, y=0)
    ax_top.set_ylim(bottom=0)
    if df.shape[0] > 2:
        ax_right.set_ylim(ax_heat.get_ylim())
        ax_right.set_xlim(left=0)
        ax_right.margins(x=0, y=0)

    # Panel title: place BELOW the panel (on the heatmap axis) so it never
    # collides with the column dendrogram above.
    if panel_title:
        ax_heat.set_xlabel(
            f"{panel_title}   (n = {df.shape[0]} × {df.shape[1]})",
            fontsize=8.6, fontweight="bold", labelpad=4,
        )
    if panel_letter:
        # Letter at upper-left of the dendrogram axis (outside the heatmap)
        ax_top.text(
            -0.015, 1.55, panel_letter, transform=ax_top.transAxes,
            fontsize=15, fontweight="bold", va="top", ha="left",
        )
    return im


# ---------------------------------------------------------------------------
# Figure 22: composite 8-panel (A-H) intra-ontological mappings
# ---------------------------------------------------------------------------

def render_fig22_combined(trimmed: dict[str, pd.DataFrame]) -> Path:
    """Produces the 4 x 2 panelled figure A-H. Each clustermap panel now has
    its OWN colour-bar (so panel H is clearly a different diagnostic rather
    than appearing as a shared colour scale), and the inter-panel spacing is
    tightened along both axes."""
    letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
    panels = [
        ("pred_to_crit", "Predictor → Criterion"),
        ("ctx_to_barr",  "Context → Barrier"),
        ("cop_to_barr",  "Coping → Barrier"),
        ("prof_to_barr", "Profile → Barrier"),
        ("barr_to_pred", "Barrier → Predictor"),
        ("ctx_to_pred",  "Context → Predictor"),
        ("prof_to_pred", "Profile → Predictor"),
    ]

    fig = plt.figure(figsize=(13.6, 17.4), dpi=140)
    outer = gridspec.GridSpec(
        nrows=4, ncols=2, figure=fig,
        left=0.045, right=0.985, top=0.985, bottom=0.022,
        hspace=0.30, wspace=0.10,
    )

    for i, (key, title) in enumerate(panels):
        df = trimmed.get(key)
        if df is None or df.empty:
            continue
        _render_clustermap_into_gridcell_with_own_cbar(
            fig, outer[i // 2, i % 2], df,
            panel_letter=letters[i], panel_title=title,
        )

    # Panel H: sparsity & coverage diagnostic across all 7 mappings
    ax_h_outer = fig.add_subplot(outer[3, 1])
    _render_sparsity_panel(ax_h_outer, trimmed, panel_letter=letters[7])

    # No in-image suptitle: the Word caption above the image is the title.
    out_path = OUT_DIR / "figure_intra_ontological_mappings.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok]   Fig 22 → {out_path.name}")
    return out_path


def _render_clustermap_into_gridcell_with_own_cbar(
    fig: plt.Figure,
    subplot_spec: gridspec.SubplotSpec,
    df: pd.DataFrame,
    *,
    panel_letter: Optional[str] = None,
    panel_title: Optional[str] = None,
) -> "plt.AxesImage":
    """Variant of _render_clustermap_into_gridcell that adds a per-panel
    colour-bar to the right of the heatmap (so each mapping in Fig 18/22 has
    its own gradient indicator)."""
    inner = gridspec.GridSpecFromSubplotSpec(
        nrows=2, ncols=3, subplot_spec=subplot_spec,
        height_ratios=[0.10, 1.0],
        width_ratios=[1.0, 0.07, 0.04],
        hspace=0.01, wspace=0.04,
    )
    ax_top = fig.add_subplot(inner[0, 0])
    ax_heat = fig.add_subplot(inner[1, 0])
    ax_right = fig.add_subplot(inner[1, 1])
    ax_cbar = fig.add_subplot(inner[1, 2])

    # Column dendrogram (top)
    col_dist = pdist(df.T.values, metric="euclidean")
    Z_col = optimal_leaf_ordering(linkage(col_dist, method="ward"), col_dist)
    dend_col = dendrogram(
        Z_col, ax=ax_top, orientation="top",
        color_threshold=0, above_threshold_color="#6c6c6c", no_labels=True,
    )
    col_order = dend_col["leaves"]
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    for s in ax_top.spines.values():
        s.set_visible(False)

    # Row dendrogram (right)
    if df.shape[0] > 2:
        row_dist = pdist(df.values, metric="euclidean")
        Z_row = optimal_leaf_ordering(linkage(row_dist, method="ward"), row_dist)
        dend_row = dendrogram(
            Z_row, ax=ax_right, orientation="right",
            color_threshold=0, above_threshold_color="#6c6c6c", no_labels=True,
        )
        row_order = dend_row["leaves"]
    else:
        row_order = list(range(df.shape[0]))
    ax_right.set_xticks([])
    ax_right.set_yticks([])
    for s in ax_right.spines.values():
        s.set_visible(False)

    ordered = df.iloc[row_order, col_order]
    n_rows, n_cols = ordered.shape
    vmax = max(50.0, float(np.nanpercentile(np.abs(ordered.values), 99)))
    norm = TwoSlopeNorm(vmin=0.0, vcenter=vmax / 2.0, vmax=vmax)
    im = ax_heat.imshow(
        ordered.values, cmap=CMAP, norm=norm, aspect="auto",
        interpolation="nearest", extent=(0, n_cols * 10, n_rows * 10, 0),
    )

    ax_heat.set_xticks([])
    ax_heat.set_yticks([])
    for s in ax_heat.spines.values():
        s.set_visible(False)

    ax_top.set_xlim(0, n_cols * 10)
    ax_top.margins(x=0, y=0)
    ax_top.set_ylim(bottom=0)
    if df.shape[0] > 2:
        ax_right.set_ylim(ax_heat.get_ylim())
        ax_right.set_xlim(left=0)
        ax_right.margins(x=0, y=0)

    if panel_title:
        ax_heat.set_xlabel(
            f"{panel_title}   (n = {df.shape[0]} × {df.shape[1]})",
            fontsize=8.6, fontweight="bold", labelpad=4,
        )
    if panel_letter:
        ax_top.text(
            -0.022, 1.55, panel_letter, transform=ax_top.transAxes,
            fontsize=15, fontweight="bold", va="top", ha="left",
        )

    # Per-panel colourbar
    cbar = fig.colorbar(im, cax=ax_cbar, orientation="vertical")
    cbar.ax.tick_params(labelsize=6.0)
    cbar.outline.set_visible(False)
    return im


def _render_sparsity_panel(ax_outer, trimmed: dict[str, pd.DataFrame], panel_letter: str) -> None:
    """Panel H: per-mapping sparsity, coverage and mean score (grouped bars).
    The bar-chart axes are inset within an inner sub-grid that leaves a wider
    right-margin so the previous panel's colour-bar (Panel G) does not collide
    with the leftmost Y-labels of this panel."""
    # Hide the parent axis
    ax_outer.set_axis_off()
    inner = gridspec.GridSpecFromSubplotSpec(
        nrows=2, ncols=3, subplot_spec=ax_outer.get_subplotspec(),
        height_ratios=[0.10, 1.0],
        # Reserve LEFT margin so Panel G's right-side colorbar (in the
        # neighbouring outer cell) has clearance from this panel's Y-labels.
        width_ratios=[0.13, 1.0, 0.04],
        hspace=0.01, wspace=0.02,
    )
    ax_top_blank = ax_outer.figure.add_subplot(inner[0, 1])
    ax_top_blank.set_axis_off()
    ax_top_blank.text(
        -0.015, 1.55, panel_letter, transform=ax_top_blank.transAxes,
        fontsize=15, fontweight="bold", va="top", ha="left",
    )

    ax = ax_outer.figure.add_subplot(inner[1, 1])

    # Compute metrics on the trimmed matrices (consistent with what is plotted)
    rows = []
    titles = []
    short_titles = {
        "pred_to_crit": "P→C",
        "ctx_to_barr":  "Ctx→B",
        "cop_to_barr":  "Cop→B",
        "prof_to_barr": "Pf→B",
        "barr_to_pred": "B→P",
        "ctx_to_pred":  "Ctx→P",
        "prof_to_pred": "Pf→P",
    }
    for spec in MAPPINGS:
        key = spec["key"]
        df = trimmed.get(key)
        if df is None or df.empty:
            continue
        arr = df.values
        density = float(np.mean(arr > 0))            # share of nonzero cells
        mean_nz = float(np.mean(arr[arr > 0])) if np.any(arr > 0) else 0.0
        max_v = float(arr.max())
        rows.append((density, mean_nz / 1000.0, max_v / 1000.0))
        titles.append(short_titles.get(key, key))

    if not rows:
        return
    arr = np.array(rows)               # (n_mappings, 3)
    n = arr.shape[0]
    x = np.arange(n)
    width = 0.27

    ax.bar(x - width, arr[:, 0], width=width, color="#1f3a68", label="Edge density (% of cells with score > 0)")
    ax.bar(x,         arr[:, 1], width=width, color="#3b6ea8", label="Mean non-zero score (scaled 0–1)")
    ax.bar(x + width, arr[:, 2], width=width, color="#e89c6a", label="Max score (scaled 0–1)")

    ax.set_xticks(x)
    ax.set_xticklabels(titles, fontsize=7.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mapping diagnostic value", fontsize=8.0)
    ax.set_title("Sparsity, mean and max score per mapping",
                 fontsize=9.0, fontweight="bold", pad=4)
    ax.legend(loc="upper right", fontsize=6.5, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#cccccc", linestyle="-", linewidth=0.4, alpha=0.6)


# ---------------------------------------------------------------------------
# Figure 12: ontology diagnostics (A-D)
# ---------------------------------------------------------------------------

def render_fig12_ontology_diagnostics() -> Path:
    """4-panel ontology diagnostics figure (Fig 12). No in-image title; tight
    layout so panels A-B and C-D sit visually paired."""
    fig = plt.figure(figsize=(13.5, 8.0), dpi=170)
    gs = gridspec.GridSpec(
        nrows=2, ncols=2, figure=fig,
        left=0.04, right=0.98, top=0.97, bottom=0.03,
        hspace=0.08, wspace=0.08,         # tight vertical spacing
    )
    _place_image_panel(fig, gs[0, 0], ONTOLOGY_MAPPING_DIAGRAM, "A",
                       "Cross-ontology mapping diagram")
    # Panel B uses aspect='auto' so the depth-distribution strip fills the
    # cell vertically instead of leaving white space below.
    _place_image_panel(fig, gs[0, 1],
                       ONTO_SUMMARY_DIR / "06_secondary" / "01_secondary_depth_panel_relative_logy.png",
                       "B", "Depth distribution per sub-ontology",
                       aspect="auto")
    _place_image_panel(fig, gs[1, 0],
                       ONTO_SUMMARY_DIR / "06_secondary" / "02_secondary_sizes_and_leaf_ratio.png",
                       "C", "Sub-tree sizes and leaf ratio per sub-ontology")
    _place_image_panel(fig, gs[1, 1],
                       ONTO_SUMMARY_DIR / "03_branching" / "02_branching_distribution_logy.png",
                       "D", "Branching-factor distribution (log-y)")

    # No in-image suptitle: the Word caption above the image is the title.
    out_path = OUT_DIR / "figure_ontology_diagnostics.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok]   Fig 12 → {out_path.name}")
    return out_path


def _place_image_panel(fig, subplot_spec, image_path: Path, letter: str, title: str,
                       *, aspect: str = "equal") -> None:
    ax = fig.add_subplot(subplot_spec)
    if image_path.exists():
        img = plt.imread(image_path)
        ax.imshow(img, aspect=aspect)
    else:
        ax.text(0.5, 0.5, f"missing: {image_path.name}", ha="center", va="center")
    ax.set_axis_off()
    ax.text(-0.02, 1.06, letter, transform=ax.transAxes,
            fontsize=17, fontweight="bold", va="top", ha="left")
    ax.set_title(title, fontsize=9.0, fontweight="bold", pad=8)


# ---------------------------------------------------------------------------
# Figure 21: Predictor → Criterion mapping diagnostics (A-D)
# ---------------------------------------------------------------------------

def render_fig21_pred_to_crit_diagnostics(trimmed: dict[str, pd.DataFrame]) -> Path:
    """Fig 21: Predictor → Criterion mapping diagnostics, 4 panels (A-D).
    No in-image title; tight vertical spacing so AB pair sits visually close
    to CD pair. Panel D is a per-mapping violin with raw scatter overlay,
    coloured by mapping."""
    # Reduce the figure's vertical extent and bring the AB row visually close
    # to the CD row. Use aspect='auto' on the image panels so they fill their
    # cells (this stretches A vertically as the user requested).
    fig = plt.figure(figsize=(14.0, 9.8), dpi=170)
    outer = gridspec.GridSpec(
        nrows=2, ncols=2, figure=fig,
        left=0.05, right=0.97, top=0.97, bottom=0.045,
        hspace=0.04, wspace=0.14,         # bring AB ↔ CD rows closer
        height_ratios=[1.0, 1.0],
    )

    _place_image_panel(fig, outer[0, 0], DENSITY_PNG,
                       "A", "BPS → secondary-domain mean-score density",
                       aspect="auto")
    _place_image_panel(fig, outer[0, 1], HISTOGRAM_PNG,
                       "B", "Score histogram of observed non-zero edges",
                       aspect="auto")

    df_pc = trimmed.get("pred_to_crit")
    if df_pc is not None and not df_pc.empty:
        _render_clustermap_into_gridcell(
            fig, outer[1, 0], df_pc,
            panel_letter="C",
            panel_title="Predictor × Criterion-cluster relevance heatmap",
            show_x_labels=False, show_y_labels=False,
        )

    _render_mapping_violin_panel(fig, outer[1, 1], trimmed, panel_letter="D")

    # No in-image suptitle.
    out_path = OUT_DIR / "figure_predictor_criterion_diagnostics.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok]   Fig 21 → {out_path.name}")
    return out_path


def _render_mapping_violin_panel(fig, subplot_spec, trimmed, panel_letter: str) -> None:
    """Violin plot of observed (non-zero) score distributions across the seven
    cross-ontology mappings, with raw-score scatter overlay and a distinct
    colour per mapping (informative encoding rather than redundant)."""
    inner = gridspec.GridSpecFromSubplotSpec(
        nrows=2, ncols=1, subplot_spec=subplot_spec,
        height_ratios=[0.10, 1.0], hspace=0.01,
    )
    ax_blank = fig.add_subplot(inner[0, 0])
    ax_blank.set_axis_off()
    ax_blank.text(-0.02, 1.30, panel_letter, transform=ax_blank.transAxes,
                  fontsize=17, fontweight="bold", va="top", ha="left")
    ax = fig.add_subplot(inner[1, 0])

    short_titles = {
        "pred_to_crit": "P→C",
        "ctx_to_barr":  "Ctx→B",
        "cop_to_barr":  "Cop→B",
        "prof_to_barr": "Pf→B",
        "barr_to_pred": "B→P",
        "ctx_to_pred":  "Ctx→P",
        "prof_to_pred": "Pf→P",
    }
    # Distinct, ordered palette — one colour per mapping (qualitative)
    palette = {
        "pred_to_crit": "#1f3a68",
        "ctx_to_barr":  "#3b6ea8",
        "cop_to_barr":  "#2f8f74",
        "prof_to_barr": "#c79a3e",
        "barr_to_pred": "#c7693d",
        "ctx_to_pred":  "#8a3e1f",
        "prof_to_pred": "#5e3a82",
    }

    data = []
    labels = []
    colours = []
    keys_used = []
    for spec in MAPPINGS:
        key = spec["key"]
        df = trimmed.get(key)
        if df is None or df.empty:
            continue
        nz = df.values[df.values > 0]
        if nz.size == 0:
            continue
        data.append(nz)
        labels.append(short_titles.get(key, key))
        colours.append(palette.get(key, "#3b6ea8"))
        keys_used.append(key)

    parts = ax.violinplot(
        data, showmeans=False, showmedians=True,
        quantiles=[[0.25, 0.75]] * len(data),
        widths=0.78,
    )
    for body, c in zip(parts["bodies"], colours):
        body.set_facecolor(c)
        body.set_edgecolor(c)
        body.set_alpha(0.55)
    for key in ("cmins", "cmaxes", "cbars", "cmedians", "cquantiles"):
        if key in parts:
            parts[key].set_color("#222222")
            parts[key].set_linewidth(0.8)

    # Raw-score scatter overlay: jitter the x-positions within each violin
    rng = np.random.default_rng(0)
    for i, (vals, c) in enumerate(zip(data, colours)):
        x_pos = (i + 1) + rng.uniform(-0.18, 0.18, size=vals.size)
        ax.scatter(x_pos, vals, s=2.6, color=c, alpha=0.32, linewidths=0)

    # Annotate sample counts above each violin
    y_top = max(np.max(v) for v in data) if data else 1
    for i, vals in enumerate(data):
        ax.text(i + 1, y_top * 1.02, f"n={vals.size:,}",
                ha="center", va="bottom", fontsize=6.5, color="#444444")

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Observed (non-zero) edge score", fontsize=8.5)
    ax.set_title(
        "Distribution of observed edge scores across the seven cross-ontology mappings",
        fontsize=9.5, fontweight="bold", pad=4,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#cccccc", linestyle="-", linewidth=0.4, alpha=0.6)
    ax.set_ylim(top=y_top * 1.08)


# ---------------------------------------------------------------------------
# Per-mapping individual standalone figure (used for Fig 21 panel C source +
# optional supplementary figures).
# ---------------------------------------------------------------------------

def render_individual_clustermap(spec: dict, df: pd.DataFrame) -> Path:
    fig = plt.figure(figsize=(11.0, 9.0), dpi=140)
    outer = gridspec.GridSpec(
        nrows=1, ncols=1, figure=fig,
        left=0.10, right=0.96, top=0.92, bottom=0.10,
    )
    _render_clustermap_into_gridcell(
        fig, outer[0, 0], df,
        panel_letter=None,
        panel_title=f"{spec['title']}   (n={df.shape[0]} × {df.shape[1]})",
        show_x_labels=True, show_y_labels=True,
    )
    fig.suptitle(
        f"Intra-ontological mapping: {spec['title']}",
        fontsize=11.5, fontweight="bold", y=0.98,
    )
    out_path = OUT_DIR / f"fig_{spec['key']}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Output directory: {OUT_DIR}")
    trimmed: dict[str, pd.DataFrame] = {}
    for spec in MAPPINGS:
        df = _load_dense(spec["dense_csv"])
        df = _apply_labels(df, spec["row_labels"], spec["col_labels"])
        df = _trim_top_k(df, spec["row_cap"], spec["col_cap"])
        trimmed[spec["key"]] = df
        render_individual_clustermap(spec, df)
        print(f"  [ok]   individual {spec['key']:<14} shape={df.shape}")

    render_fig12_ontology_diagnostics()
    render_fig21_pred_to_crit_diagnostics(trimmed)
    render_fig22_combined(trimmed)
    print("\nDone.")


if __name__ == "__main__":
    main()
