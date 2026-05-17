"""
Intra-ontological mapping visualisations for the PHOENIX ontology.

For each of the seven cross-ontology mappings in
`src/backend/utils/official/ontology_mappings/...`, this script reads the dense
relevance matrix, optionally derives compact display labels from the source
input lists, then renders a clustermap-style figure where:

    - a Ward-linkage dendrogram is mounted on the TOP   axis  (column tree)
    - a Ward-linkage dendrogram is mounted on the RIGHT axis  (row    tree)
    - the central heatmap is reordered by optimal leaf ordering on both sides
    - a horizontal colourbar sits below the heatmap
    - axis labels are short, readable, APA-style.

Adapted from the clustermap / ECBSS heatmap pattern provided.

It writes individual figures per mapping plus a combined panelled figure
(panels A through G) with all seven mappings side by side, sized for thesis
inclusion (US Letter content width, 9360 DXA ≈ 6.5 inch).

USAGE:
    python visualize_intra_ontological_mappings.py

Outputs:
    /src/backend/utils/official/overall_mapping_analyses/results/
        intra_ontological_mappings/
            ./fig_pred_to_crit.png
            ./fig_ctx_to_barr.png
            ./fig_cop_to_barr.png
            ./fig_prof_to_barr.png
            ./fig_barr_to_pred.png
            ./fig_ctx_to_pred.png
            ./fig_prof_to_pred.png
            ./combined_intra_ontological_mappings.png      (panels A-G)
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
# Paths & layout constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[5]
MAP_ROOT = REPO_ROOT / "src/backend/utils/official/ontology_mappings"
CLUSTER_ROOT = REPO_ROOT / "src/backend/utils/official/cluster_criterions/results"
OUT_DIR = (
    REPO_ROOT
    / "src/backend/utils/official/overall_mapping_analyses/results/intra_ontological_mappings"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Spec for each of the seven mappings.
# - relative path of the dense CSV
# - input-list file for the column labels (or None to use numeric ids)
# - input-list file for the row labels    (or None to use numeric ids)
# - mapping caption used in the thesis figure
# - max rows / cols to retain on the heatmap (top-K by row/col sum)
MAPPINGS = [
    {
        "key": "pred_to_crit",
        "dense_csv": "CRITERION/predictor_to_criterion/results/gpt-5-nano/predictor_to_criterion_map_dense.csv",
        "row_labels": "cluster_id",   # special: derive from cluster JSON
        "col_labels": "CRITERION/predictor_to_criterion/input_lists/predictors_list.txt",
        "row_role": "Criterion cluster",
        "col_role": "Predictor leaf",
        "title": "Predictor → Criterion",
        "row_cap": 80,
        "col_cap": 80,
    },
    {
        "key": "ctx_to_barr",
        "dense_csv": "HAPA/context_to_barrier/results/gpt-5-nano/context_to_barrier_map_dense.csv",
        "row_labels": "HAPA/context_to_barrier/input_lists/barriers_list.txt",
        "col_labels": "HAPA/context_to_barrier/input_lists/contextual_factors_list.txt",
        "row_role": "Barrier",
        "col_role": "Context",
        "title": "Context → Barrier",
        "row_cap": 60,
        "col_cap": 60,
    },
    {
        "key": "cop_to_barr",
        "dense_csv": "HAPA/coping_to_barrier/results/gpt-5-nano/coping_to_barrier_map_dense.csv",
        "row_labels": "HAPA/coping_to_barrier/input_lists/coping_options_list.txt",
        "col_labels": "HAPA/coping_to_barrier/input_lists/barriers_list.txt",
        "row_role": "Coping strategy",
        "col_role": "Barrier",
        "title": "Coping → Barrier",
        "row_cap": 60,
        "col_cap": 60,
    },
    {
        "key": "prof_to_barr",
        "dense_csv": "HAPA/profile_to_barrier/results/gpt-5-nano/profile_to_barrier_map_dense.csv",
        "row_labels": "HAPA/profile_to_barrier/input_lists/barriers_list.txt",
        "col_labels": "HAPA/profile_to_barrier/input_lists/person_factors_list.txt",
        "row_role": "Barrier",
        "col_role": "Profile attribute",
        "title": "Profile → Barrier",
        "row_cap": 60,
        "col_cap": 80,
    },
    {
        "key": "barr_to_pred",
        "dense_csv": "PREDICTOR/barrier_to_predictor/results/gpt-5-nano/predictor_to_barrier_map_dense.csv",
        "row_labels": "PREDICTOR/barrier_to_predictor/input_lists/predictors_list.txt",
        "col_labels": "PREDICTOR/barrier_to_predictor/input_lists/barriers_list.txt",
        "row_role": "Predictor leaf",
        "col_role": "Barrier",
        "title": "Barrier → Predictor",
        "row_cap": 80,
        "col_cap": 60,
    },
    {
        "key": "ctx_to_pred",
        "dense_csv": "PREDICTOR/context_to_predictor/results/gpt-5-nano/predictor_to_context_map_dense.csv",
        "row_labels": "PREDICTOR/context_to_predictor/input_lists/predictors_list.txt",
        "col_labels": "PREDICTOR/context_to_predictor/input_lists/contextual_factors_list.txt",
        "row_role": "Predictor leaf",
        "col_role": "Context",
        "title": "Context → Predictor",
        "row_cap": 80,
        "col_cap": 60,
    },
    {
        "key": "prof_to_pred",
        "dense_csv": "PREDICTOR/profile_to_predictor/results/gpt-5-nano/predictor_to_profile_map_dense.csv",
        "row_labels": "PREDICTOR/profile_to_predictor/input_lists/predictors_list.txt",
        "col_labels": "PREDICTOR/profile_to_predictor/input_lists/person_factors_list.txt",
        "row_role": "Predictor leaf",
        "col_role": "Profile attribute",
        "title": "Profile → Predictor",
        "row_cap": 80,
        "col_cap": 80,
    },
]


# Colour scheme: cool→neutral→warm diverging (works well for 0-1000 scores
# treating 500 as neutral). Score range observed ≈ 0–1000.
CMAP = LinearSegmentedColormap.from_list(
    "phoenix_mapping",
    ["#1a3a6c", "#5b8dc5", "#cfe1f2", "#f7f7f7", "#fae0c1", "#e89c6a", "#a14a2a"],
)


# ---------------------------------------------------------------------------
# Label parsing helpers
# ---------------------------------------------------------------------------

LEAF_LINE = re.compile(r"\(ID:(\d+)\)\s*$")
INLINE_ID = re.compile(r"^\s*(\d+)[\s.\-:]\s*(.+?)\s*$")


def _short(label: str, n: int = 4) -> str:
    """Compact a long label to its last 1-2 informative tokens."""
    if label is None:
        return ""
    s = str(label)
    s = s.replace("_", " ").strip()
    # Drop bracketed prefixes like [BIO], [PSYCHO], [SOCIAL]
    s = re.sub(r"^\[[A-Z]+\]\s*", "", s)
    # Take last n words
    tokens = s.split()
    if len(tokens) <= n:
        return s
    return " ".join(tokens[-n:])


def _load_id_label_map_from_tree(path: Path) -> dict[int, str]:
    """
    Parse files of the form:
        [BIO]
        └─ Cognitive_Capacity_and_Brain_Health
          └─ Cerebrovascular_and_Oxygenation_Support (ID:0)

    Returns {id: label_at_leaf}.
    """
    out: dict[int, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = LEAF_LINE.search(raw)
        if not m:
            continue
        leaf_id = int(m.group(1))
        # The label is the token chain just before "(ID:n)"; take last
        # "Name_with_underscores" before "(ID:n)".
        prefix = raw[: m.start()].strip()
        # Strip leading tree characters and indentation
        prefix = re.sub(r"^[\s│├└─]*", "", prefix)
        out[leaf_id] = prefix
    return out


def _load_id_label_map_from_flat(path: Path) -> dict[int, str]:
    """
    Parse files where each line begins with an integer id followed by
    a label, or where labels are emitted positionally (one per line) and
    the id is the line index.
    """
    out: dict[int, str] = {}
    if not path.exists():
        return out
    lines = [ln for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
    # Try the "id<sep>label" form first
    matched_any = False
    for ln in lines:
        m = INLINE_ID.match(ln)
        if m:
            out[int(m.group(1))] = m.group(2)
            matched_any = True
    if matched_any:
        return out
    # Fall back to positional indexing
    for i, ln in enumerate(lines):
        out[i] = ln.strip()
    return out


def _load_labels(input_list_rel: str) -> dict[int, str]:
    """Load a label registry by parsing the input list file."""
    fpath = MAP_ROOT / input_list_rel
    # First try the tree form
    out = _load_id_label_map_from_tree(fpath)
    if out:
        return out
    return _load_id_label_map_from_flat(fpath)


def _load_criterion_cluster_labels() -> dict[str, str]:
    """
    Load representative labels for the 3,111 criterion clusters from the
    cluster-criterions output JSON. Each cluster's label is the most
    common short form of the first item in its `items` list.
    """
    fpath = CLUSTER_ROOT / "04_semantically_clustered_items.json"
    labels: dict[str, str] = {}
    if not fpath.exists():
        return labels
    data = json.loads(fpath.read_text(encoding="utf-8", errors="ignore"))
    clusters = data.get("clusters", {})
    # Try the GPT-labeled file first for richer names
    gpt_labels_path = CLUSTER_ROOT / "05_GPT_labeled_split_items.txt"
    if gpt_labels_path.exists():
        gpt_lines = [ln.strip() for ln in gpt_labels_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        for i, ln in enumerate(gpt_lines):
            labels[f"c{i}"] = ln[:60]
    # Fall back to first item of each cluster
    for cid, info in clusters.items():
        if cid in labels:
            continue
        items = info.get("items") or []
        if items:
            first = items[0]
            # Strip context annotations like " (context: ...)"
            label = first.split(" (context")[0].split(" — criterion")[0]
            labels[cid] = label.replace("_", " ")[:60]
        else:
            labels[cid] = cid
    return labels


# ---------------------------------------------------------------------------
# Matrix loading & preprocessing
# ---------------------------------------------------------------------------

def _load_dense(rel_path: str) -> pd.DataFrame:
    full = MAP_ROOT / rel_path
    df = pd.read_csv(full, index_col=0)
    # Coerce all numeric, NaN treated as 0 (no edge emitted)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df


def _apply_labels(
    df: pd.DataFrame,
    row_labels_spec: str,
    col_labels_spec: str,
) -> pd.DataFrame:
    """Apply human-readable labels to rows and columns if available."""
    # Columns: input_list file maps id -> label
    if col_labels_spec and not col_labels_spec.startswith("cluster_id"):
        col_map = _load_labels(col_labels_spec)
        if col_map:
            new_cols = []
            for c in df.columns:
                try:
                    cid = int(c)
                except (ValueError, TypeError):
                    new_cols.append(str(c))
                    continue
                new_cols.append(col_map.get(cid, str(c)))
            df.columns = new_cols
    # Rows
    if row_labels_spec == "cluster_id":
        cluster_lbls = _load_criterion_cluster_labels()
        df.index = [cluster_lbls.get(str(idx), str(idx)) for idx in df.index]
    elif row_labels_spec:
        row_map = _load_labels(row_labels_spec)
        if row_map:
            new_idx = []
            for r in df.index:
                try:
                    rid = int(r)
                except (ValueError, TypeError):
                    new_idx.append(str(r))
                    continue
                new_idx.append(row_map.get(rid, str(r)))
            df.index = new_idx
    return df


def _trim_top_k(df: pd.DataFrame, row_cap: int, col_cap: int) -> pd.DataFrame:
    """Retain top-K rows and columns by aggregate emission strength."""
    df = df.copy()
    # Drop fully-zero rows / cols
    df = df.loc[df.abs().sum(axis=1) > 0, df.abs().sum(axis=0) > 0]
    if row_cap and df.shape[0] > row_cap:
        row_strength = df.abs().sum(axis=1).sort_values(ascending=False)
        df = df.loc[row_strength.head(row_cap).index]
    if col_cap and df.shape[1] > col_cap:
        col_strength = df.abs().sum(axis=0).sort_values(ascending=False)
        df = df.loc[:, col_strength.head(col_cap).index]
    return df


# ---------------------------------------------------------------------------
# Plot primitives
# ---------------------------------------------------------------------------

def _plot_clustermap(
    df: pd.DataFrame,
    title: str,
    row_role: str,
    col_role: str,
    out_path: Path,
    figsize: tuple[float, float] = (11.0, 9.0),
    show_row_labels: bool = True,
    show_col_labels: bool = True,
    annot_threshold: int = 30,
) -> None:
    """Render a dendrogram-mounted clustermap for one mapping."""
    if df.empty or df.shape[0] < 2 or df.shape[1] < 2:
        print(f"  [skip] {out_path.name}: matrix too small ({df.shape})")
        return

    fig = plt.figure(figsize=figsize, dpi=140)
    gs = gridspec.GridSpec(
        nrows=3,
        ncols=2,
        figure=fig,
        height_ratios=[0.16, 1.0, 0.06],
        width_ratios=[1.0, 0.14],
        left=0.20, right=0.92, top=0.92, bottom=0.22,
        hspace=0.02, wspace=0.02,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1])
    ax_cbar = fig.add_axes([0.20, 0.10, 0.55, 0.018])

    # Column dendrogram (top)
    col_dist = pdist(df.T.values, metric="euclidean")
    Z_col = optimal_leaf_ordering(linkage(col_dist, method="ward"), col_dist)
    dend_col = dendrogram(
        Z_col, ax=ax_top, orientation="top",
        color_threshold=0, above_threshold_color="#8d8d8d", no_labels=True,
    )
    col_order = dend_col["leaves"]
    ax_top.set_axis_off()

    # Row dendrogram (right)
    if df.shape[0] > 2:
        row_dist = pdist(df.values, metric="euclidean")
        Z_row = optimal_leaf_ordering(linkage(row_dist, method="ward"), row_dist)
        dend_row = dendrogram(
            Z_row, ax=ax_right, orientation="right",
            color_threshold=0, above_threshold_color="#8d8d8d", no_labels=True,
        )
        row_order = dend_row["leaves"]
    else:
        row_order = list(range(df.shape[0]))
    ax_right.set_axis_off()

    ordered = df.iloc[row_order, col_order]
    n_rows, n_cols = ordered.shape

    # Heatmap
    vmax = max(50.0, float(np.nanpercentile(np.abs(ordered.values), 99)))
    norm = TwoSlopeNorm(vmin=0.0, vcenter=vmax / 2, vmax=vmax)
    im = ax_heat.imshow(
        ordered.values,
        cmap=CMAP, norm=norm, aspect="auto", interpolation="nearest",
        extent=(0, n_cols * 10, n_rows * 10, 0),
    )

    # Axis labels (rows on left, cols on bottom rotated)
    x_centers = np.arange(n_cols) * 10 + 5
    y_centers = np.arange(n_rows) * 10 + 5
    if show_col_labels:
        col_labels = [_short(c, 4) for c in ordered.columns]
        ax_heat.set_xticks(x_centers)
        ax_heat.set_xticklabels(col_labels, rotation=70, ha="right", fontsize=6.5)
    else:
        ax_heat.set_xticks([])
    if show_row_labels and n_rows <= annot_threshold:
        row_labels = [_short(r, 4) for r in ordered.index]
        ax_heat.set_yticks(y_centers)
        ax_heat.set_yticklabels(row_labels, fontsize=6.5)
    elif show_row_labels and n_rows > annot_threshold:
        # too many rows for individual labels — show every k-th
        k = max(1, n_rows // annot_threshold)
        idx = list(range(0, n_rows, k))
        ax_heat.set_yticks([y_centers[i] for i in idx])
        ax_heat.set_yticklabels([_short(ordered.index[i], 3) for i in idx], fontsize=5.5)
    else:
        ax_heat.set_yticks([])
    ax_heat.tick_params(length=0)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)

    # Align right dendrogram exactly to heatmap y
    if df.shape[0] > 2:
        ax_right.set_ylim(ax_heat.get_ylim())
        ax_right.set_xlim(left=0)
        ax_right.margins(x=0, y=0)
    # Align top dendrogram to heatmap x
    ax_top.set_xlim(0, n_cols * 10)
    ax_top.margins(x=0, y=0)
    ax_top.set_ylim(bottom=0)

    # Colourbar
    cbar = fig.colorbar(im, cax=ax_cbar, orientation="horizontal")
    cbar.set_label(
        f"Relevance score (0–1000) — rows: {row_role}; columns: {col_role}",
        fontsize=7.2,
    )
    cbar.ax.tick_params(labelsize=6.5)

    # Suptitle
    fig.suptitle(
        f"Intra-ontological mapping: {title}\n"
        f"(rows reordered by Ward linkage on emission profiles; columns reordered similarly)",
        fontsize=10.0, fontweight="bold", y=0.985,
    )

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok]   {out_path.name}  shape={df.shape}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_individuals() -> dict[str, pd.DataFrame]:
    """Render seven individual mapping figures; return their trimmed DataFrames."""
    trimmed = {}
    for spec in MAPPINGS:
        print(f"[load] {spec['key']:>15}  ←  {spec['dense_csv']}")
        df = _load_dense(spec["dense_csv"])
        df = _apply_labels(df, spec["row_labels"], spec["col_labels"])
        df = _trim_top_k(df, spec["row_cap"], spec["col_cap"])
        trimmed[spec["key"]] = df

        out_path = OUT_DIR / f"fig_{spec['key']}.png"
        _plot_clustermap(
            df=df,
            title=spec["title"],
            row_role=spec["row_role"],
            col_role=spec["col_role"],
            out_path=out_path,
            figsize=(11.0, 9.0),
            show_row_labels=df.shape[0] <= 60,
            show_col_labels=df.shape[1] <= 80,
        )
    return trimmed


def run_combined_panel(trimmed: dict[str, pd.DataFrame]) -> None:
    """Combine the seven mappings into a single panelled figure (A-G)."""
    panel_specs = [(s["key"], s["title"]) for s in MAPPINGS]
    panel_letters = ["A", "B", "C", "D", "E", "F", "G"]

    fig = plt.figure(figsize=(16.0, 22.0), dpi=140)
    outer = gridspec.GridSpec(
        nrows=4, ncols=2, figure=fig,
        left=0.06, right=0.97, top=0.97, bottom=0.03,
        hspace=0.38, wspace=0.22,
    )

    for i, (key, title) in enumerate(panel_specs):
        if key not in trimmed:
            continue
        df = trimmed[key]
        if df.empty:
            continue

        # Each panel gets its own mini-gridspec for top-dendro + heatmap + right-dendro
        cell = outer[i // 2, i % 2]
        inner = gridspec.GridSpecFromSubplotSpec(
            nrows=2, ncols=2, subplot_spec=cell,
            height_ratios=[0.14, 1.0],
            width_ratios=[1.0, 0.10],
            hspace=0.02, wspace=0.02,
        )
        ax_top = fig.add_subplot(inner[0, 0])
        ax_heat = fig.add_subplot(inner[1, 0])
        ax_right = fig.add_subplot(inner[1, 1])

        # Compute dendrograms
        col_dist = pdist(df.T.values, metric="euclidean")
        Z_col = optimal_leaf_ordering(linkage(col_dist, method="ward"), col_dist)
        dend_col = dendrogram(
            Z_col, ax=ax_top, orientation="top",
            color_threshold=0, above_threshold_color="#8d8d8d", no_labels=True,
        )
        col_order = dend_col["leaves"]
        ax_top.set_axis_off()

        if df.shape[0] > 2:
            row_dist = pdist(df.values, metric="euclidean")
            Z_row = optimal_leaf_ordering(linkage(row_dist, method="ward"), row_dist)
            dend_row = dendrogram(
                Z_row, ax=ax_right, orientation="right",
                color_threshold=0, above_threshold_color="#8d8d8d", no_labels=True,
            )
            row_order = dend_row["leaves"]
        else:
            row_order = list(range(df.shape[0]))
        ax_right.set_axis_off()

        ordered = df.iloc[row_order, col_order]
        n_rows, n_cols = ordered.shape
        vmax = max(50.0, float(np.nanpercentile(np.abs(ordered.values), 99)))
        norm = TwoSlopeNorm(vmin=0.0, vcenter=vmax / 2, vmax=vmax)
        im = ax_heat.imshow(
            ordered.values, cmap=CMAP, norm=norm, aspect="auto",
            interpolation="nearest", extent=(0, n_cols * 10, n_rows * 10, 0),
        )
        ax_heat.set_xticks([])
        ax_heat.set_yticks([])
        for spine in ax_heat.spines.values():
            spine.set_visible(False)

        # Align dendrograms exactly
        ax_top.set_xlim(0, n_cols * 10)
        ax_top.margins(x=0, y=0)
        ax_top.set_ylim(bottom=0)
        if df.shape[0] > 2:
            ax_right.set_ylim(ax_heat.get_ylim())
            ax_right.set_xlim(left=0)
            ax_right.margins(x=0, y=0)

        # Panel label and title (with shape annotation)
        letter = panel_letters[i]
        ax_heat.set_title(f"{title}  (n={df.shape[0]}×{df.shape[1]})",
                          fontsize=9.5, fontweight="normal", loc="center", pad=4)
        ax_top.text(
            -0.07, 1.30, letter, transform=ax_top.transAxes,
            fontsize=18, fontweight="bold", va="top", ha="left",
        )

    # Single shared horizontal colourbar at the bottom of the figure.
    cax = fig.add_axes([0.20, 0.005, 0.6, 0.012])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label(
        "Relevance score (0–1000); rows/cols reordered by Ward-linkage optimal leaf ordering",
        fontsize=9,
    )
    cbar.ax.tick_params(labelsize=8)

    out_path = OUT_DIR / "combined_intra_ontological_mappings.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok]   combined panel  →  {out_path.name}")


def run_density_histogram_combo() -> None:
    """Stack the existing PRIMARY density and histogram into one A/B figure."""
    primary = (
        REPO_ROOT
        / "src/backend/utils/official/overall_mapping_analyses/results/overall_mapping_analysis/plots/PRIMARY"
    )
    density_path = primary / "hierarchical_BPS_secondary_density_points.png"
    hist_path = primary / "score_histogram_observed_edges.png"
    if not (density_path.exists() and hist_path.exists()):
        print(f"  [skip] missing density/histogram source files")
        return
    img_density = plt.imread(density_path)
    img_hist = plt.imread(hist_path)

    fig = plt.figure(figsize=(13.5, 9.8), dpi=160)
    gs = gridspec.GridSpec(
        2, 1, figure=fig, height_ratios=[1.0, 1.05],
        left=0.04, right=0.98, top=0.96, bottom=0.04, hspace=0.10,
    )
    axA = fig.add_subplot(gs[0])
    axA.imshow(img_density)
    axA.set_axis_off()
    axA.text(-0.01, 1.04, "A", transform=axA.transAxes,
             fontsize=20, fontweight="bold", va="top", ha="left")

    axB = fig.add_subplot(gs[1])
    axB.imshow(img_hist)
    axB.set_axis_off()
    axB.text(-0.01, 1.04, "B", transform=axB.transAxes,
             fontsize=20, fontweight="bold", va="top", ha="left")

    out_path = OUT_DIR / "fig_mapping_overall_density_and_histogram.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok]   density+histogram combo  →  {out_path.name}")


def main() -> None:
    print(f"Output directory: {OUT_DIR}")
    run_density_histogram_combo()
    trimmed = run_individuals()
    run_combined_panel(trimmed)
    print("\nDone.")


if __name__ == "__main__":
    main()
