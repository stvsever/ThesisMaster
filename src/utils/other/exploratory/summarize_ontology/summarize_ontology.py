#!/usr/bin/env python3
"""
JSON Subclass Ontology Summarizer + Publish-ready Plots (Robust / Error-free)

Fixes included (relative to the previous version)
-------------------------------------------------
1) FIXED: Matplotlib ValueError "x and y must have same first dimension" in the secondary depth panel.
   Root cause: gaussian smoothing with 'same' convolution can return a vector longer than x when
   the gaussian kernel is larger than the number of bins (short distributions).
   Fix: robust smoothing that ALWAYS returns same length as input.

2) SAVE SUMMARY: Always saves the summary to ONE file format (EITHER .txt OR .json, not both),
   in: /Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/other/exploratory/summarize_ontology/summary
   - Default: JSON (machine readable)
   - If --summary-format txt: writes txt
   - If --summary-format json: writes json
   - If --summary-format none: don't save

3) Professional visualization improvements:
   - Better layout/spacing, consistent formatting, and clearer titles.
   - Log-scale variants where long-tails exist.
   - Additional "raw datapoint distribution" plot: depth KDE-like line (smoothed) + histogram.
   - Additional plot: subtree size distribution across secondary nodes (if many).
   - All plots saved as PNG + PDF at publication quality.

Output directories
-----------------
Plots root:
  /Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/other/exploratory/summarize_ontology/plots/<run_id>/
    01_overview/
    02_depth/
    03_branching/
    04_labels/
    05_text/
    06_secondary/

Summary output:
  /Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/other/exploratory/summarize_ontology/summary/
    <base>__<run_id>.summary.json   (default)
    OR
    <base>__<run_id>.summary.txt

Expected JSON shape (typical):
{
  "RootA": {
    "ChildA1": {
      "GrandchildA1a": {}
    },
    "ChildA2": {}
  }
}

Tolerates:
- empty dicts / None as leaves
- lists of strings or dicts as children (best-effort)
"""

import argparse
import json
import os
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# =========================
# Paths (project defaults)
# =========================

def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "Evaluation").exists() and (candidate / "README.md").exists():
            return candidate
    raise RuntimeError("Could not locate repository root from summarize_ontology.py")


REPO_ROOT = _find_repo_root()
DEFAULT_INPUT_JSON = str(REPO_ROOT / "SystemComponents/PHOENIX_ontology/aggretated/01_raw/ontology_aggregated.json")
DEFAULT_PLOT_DIR = str(REPO_ROOT / "utils/other/exploratory/summarize_ontology/plots")
DEFAULT_SUMMARY_DIR = str(REPO_ROOT / "utils/other/exploratory/summarize_ontology/summary")


# =========================
# Helpers
# =========================

def normalize_label(label: str) -> str:
    return " ".join(str(label).strip().split())


def sanitize_fragment(text: str) -> str:
    """Make a stable ID fragment for paths (not meant as an IRI, just a safe identifier)."""
    s = normalize_label(text).lower().replace(" ", "_").replace("-", "_")
    s = "".join(ch for ch in s if ch.isalnum() or ch == "_")
    return s or "unnamed"


def make_node_id(path: Sequence[str]) -> str:
    # Include the full path to disambiguate same labels appearing in different branches
    return "__".join(sanitize_fragment(p) for p in path)


def load_json_hierarchy(json_path: str) -> Dict[str, Any]:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("JSON root must be a dictionary/object (dict-of-subclasses).")
        return data
    except Exception as e:
        raise RuntimeError(f"Failed to load JSON hierarchy: {json_path}") from e


def iter_children(children: Any) -> Iterable[Tuple[str, Any]]:
    """
    Best-effort normalizer to iterate over children.

    Supported:
    - dict: {label: subclasses}
    - list: ["A", "B"] or [{"A": {...}}, {"B": {...}}]
    - None/empty: no children
    """
    if children is None:
        return []
    if isinstance(children, dict):
        return list(children.items())
    if isinstance(children, list):
        out: List[Tuple[str, Any]] = []
        for item in children:
            if isinstance(item, str):
                out.append((item, {}))
            elif isinstance(item, dict):
                out.extend(list(item.items()))
            else:
                continue
        return out
    return []


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def human_int(x: int) -> str:
    return f"{int(x):,}"


def safe_filename(s: str, max_len: int = 80) -> str:
    s2 = sanitize_fragment(s)
    return s2[:max_len].strip("_") or "plot"


def shorten_path_for_plot(path: str, max_chars: int = 95) -> str:
    if len(path) <= max_chars:
        return path
    keep = max_chars - 5
    a = int(keep * 0.55)
    b = keep - a
    return f"{path[:a]} ... {path[-b:]}"


# =========================
# Robust smoothing (FIX)
# =========================

def gaussian_smooth_1d(y: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    """
    Robust gaussian smoothing without SciPy, GUARANTEED same length as input.

    Previous bug source:
    - np.convolve(y, kernel, mode="same") can return length == max(len(y), len(kernel)),
      which can exceed len(y) if kernel is longer than y.

    Fix strategy:
    - Build kernel radius capped relative to len(y).
    - Use explicit padding + 'valid' convolution to force output length == len(y).
    """
    y = np.asarray(y, dtype=float)
    n = int(y.size)
    if n <= 2 or sigma <= 0:
        return y.copy()

    # Cap radius so kernel isn't longer than the signal.
    # Keep kernel length <= n (odd length preferred).
    max_radius = max(1, (n - 1) // 2)
    radius = int(max(1, min(max_radius, round(3 * sigma))))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()

    # Pad and convolve in a way that yields exactly n samples.
    # Reflect padding helps maintain edge behavior.
    pad = radius
    y_pad = np.pad(y, pad_width=pad, mode="reflect")
    sm = np.convolve(y_pad, kernel, mode="valid")  # length = n
    if sm.size != n:
        # Ultra safety: trim or pad to n
        sm = sm[:n] if sm.size > n else np.pad(sm, (0, n - sm.size), mode="edge")
    return sm


# =========================
# Plot styling + saving
# =========================

def set_plot_style() -> None:
    # Professional, consistent, clean.
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 320,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.titlepad": 10,
        "axes.labelpad": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _int_formatter(_x, _pos) -> str:
    try:
        return f"{int(_x):,}"
    except Exception:
        return str(_x)


def save_figure(fig: plt.Figure, out_basepath_no_ext: str) -> List[str]:
    """
    Saves both PNG and PDF, returns written paths.
    """
    out_paths = []
    png_path = out_basepath_no_ext + ".png"
    #pdf_path = out_basepath_no_ext + ".pdf"
    fig.savefig(png_path, bbox_inches="tight")
    #fig.savefig(pdf_path, bbox_inches="tight")
    out_paths.extend([png_path])
    plt.close(fig)
    return out_paths


# =========================
# Core model
# =========================

@dataclass(frozen=True)
class Node:
    node_id: str
    label: str
    path: Tuple[str, ...]
    depth: int
    parent_id: Optional[str]


def build_index(hierarchy: Dict[str, Any]) -> Tuple[Dict[str, Node], Dict[str, List[str]], List[str], List[str]]:
    """
    Traverse the JSON hierarchy and build:
    - nodes: node_id -> Node
    - children_map: node_id -> [child_node_id, ...]
    - anomalies: list of warnings
    - root_ids: list of root node IDs
    """
    nodes: Dict[str, Node] = {}
    children_map: Dict[str, List[str]] = defaultdict(list)
    anomalies: List[str] = []
    root_ids: List[str] = []

    def walk(subtree: Any, parent_path: Tuple[str, ...], parent_id: Optional[str]) -> None:
        items = subtree.items() if isinstance(subtree, dict) else []
        for raw_label, raw_children in items:
            if raw_label is None:
                anomalies.append("Found null label at some node; skipping.")
                continue

            label = normalize_label(str(raw_label))
            if not label:
                anomalies.append(f"Found empty/whitespace label under parent path {parent_path}; using 'Unnamed'.")
                label = "Unnamed"

            path = parent_path + (label,)
            node_id = make_node_id(path)
            depth = len(path)

            if node_id in nodes:
                anomalies.append(
                    f"ID collision detected for path {path} (node_id={node_id}). "
                    f"Consider adjusting sanitize_fragment."
                )

            nodes[node_id] = Node(
                node_id=node_id,
                label=label,
                path=path,
                depth=depth,
                parent_id=parent_id,
            )

            if parent_id is None:
                root_ids.append(node_id)
            else:
                children_map[parent_id].append(node_id)

            child_pairs = iter_children(raw_children)
            if raw_children not in (None, {}) and not isinstance(raw_children, (dict, list)):
                anomalies.append(
                    f"Non-dict/list children value at path {path}: type={type(raw_children).__name__}; treated as leaf."
                )

            # Deduplicate identical child labels under same parent (quality hint)
            seen_child_labels = set()
            for c_label, _ in child_pairs:
                nl = normalize_label(str(c_label))
                if nl in seen_child_labels:
                    anomalies.append(f"Duplicate sibling label '{nl}' under parent path {path}.")
                else:
                    seen_child_labels.add(nl)

            if isinstance(raw_children, dict):
                walk(raw_children, path, node_id)
            elif isinstance(raw_children, list):
                for c_label, c_children in child_pairs:
                    c_label_n = normalize_label(str(c_label)) or "Unnamed"
                    c_path = path + (c_label_n,)
                    c_id = make_node_id(c_path)

                    # Create child node from list
                    nodes[c_id] = Node(
                        node_id=c_id,
                        label=c_label_n,
                        path=c_path,
                        depth=len(c_path),
                        parent_id=node_id,
                    )
                    children_map[node_id].append(c_id)

                    if isinstance(c_children, dict):
                        walk(c_children, c_path, c_id)
                    elif isinstance(c_children, list):
                        for gc_label, gc_children in iter_children(c_children):
                            if isinstance(gc_children, dict):
                                walk({gc_label: gc_children}, c_path, c_id)

    walk(hierarchy, tuple(), None)
    return nodes, children_map, anomalies, root_ids


def compute_subtree_sizes(root_ids: List[str], children_map: Dict[str, List[str]]) -> Dict[str, int]:
    sizes: Dict[str, int] = {}

    def dfs(n: str) -> int:
        if n in sizes:
            return sizes[n]
        total = 1
        for c in children_map.get(n, []):
            total += dfs(c)
        sizes[n] = total
        return total

    for r in root_ids:
        dfs(r)
    return sizes


def collect_subtree_nodes(root_id: str, children_map: Dict[str, List[str]]) -> List[str]:
    out: List[str] = []
    stack = [root_id]
    while stack:
        n = stack.pop()
        out.append(n)
        stack.extend(children_map.get(n, []))
    return out


# =========================
# Report rendering
# =========================

def render_report(
    input_path: str,
    nodes: Dict[str, Node],
    children_map: Dict[str, List[str]],
    anomalies: List[str],
    root_ids: List[str],
    top_k: int = 10,
) -> str:
    total = len(nodes)
    all_ids = list(nodes.keys())
    roots = root_ids
    leaves = [nid for nid in all_ids if len(children_map.get(nid, [])) == 0]
    internal = [nid for nid in all_ids if len(children_map.get(nid, [])) > 0]

    depths = [nodes[nid].depth for nid in all_ids] or [0]
    max_depth = max(depths) if depths else 0
    min_depth = min(depths) if depths else 0
    avg_depth = (sum(depths) / len(depths)) if depths else 0.0
    med_depth = statistics.median(depths) if depths else 0.0
    depth_dist = Counter(depths)

    level_counts = Counter(depths)
    widest_depth, widest_count = (None, 0)
    if level_counts:
        widest_depth, widest_count = max(level_counts.items(), key=lambda kv: kv[1])

    child_counts = [len(children_map.get(nid, [])) for nid in internal]
    avg_branching = (sum(child_counts) / len(child_counts)) if child_counts else 0.0
    max_branching = max(child_counts) if child_counts else 0

    top_branch_nodes = sorted(
        internal,
        key=lambda nid: len(children_map.get(nid, [])),
        reverse=True
    )[:top_k]

    subtree_sizes = compute_subtree_sizes(roots, children_map)
    root_sizes = sorted(roots, key=lambda r: subtree_sizes.get(r, 1), reverse=True)

    deepest_leaves = sorted(leaves, key=lambda nid: nodes[nid].depth, reverse=True)[:top_k]

    label_to_nodes: Dict[str, List[str]] = defaultdict(list)
    label_to_depths: Dict[str, set] = defaultdict(set)
    for nid, node in nodes.items():
        label_to_nodes[node.label].append(nid)
        label_to_depths[node.label].add(node.depth)

    dup_labels = [(lbl, ids) for lbl, ids in label_to_nodes.items() if len(ids) > 1]
    dup_labels.sort(key=lambda x: len(x[1]), reverse=True)
    multi_depth_dups = [(lbl, ids) for lbl, ids in dup_labels if len(label_to_depths[lbl]) > 1]

    leaf_ratio = (len(leaves) / total) if total else 0.0
    root_ratio = (len(roots) / total) if total else 0.0

    def fmt_path(nid: str) -> str:
        return " > ".join(nodes[nid].path)

    def fmt_node_short(nid: str) -> str:
        n = nodes[nid]
        return f"{n.label} (depth {n.depth})"

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = []
    lines.append("=== JSON Subclass Ontology Summary Report ===")
    lines.append(f"Generated: {ts}")
    lines.append(f"Input: {input_path}")
    lines.append("")

    lines.append("1) High-level structure")
    lines.append(f"- Total classes/nodes: {total}")
    lines.append(f"- Root classes: {len(roots)} ({root_ratio:.3%} of all nodes)")
    lines.append(f"- Leaf classes: {len(leaves)} ({leaf_ratio:.3%} of all nodes)")
    lines.append(f"- Internal classes (non-leaf): {len(internal)}")
    lines.append("")

    lines.append("2) Root overview (largest subtrees)")
    if roots:
        lines.append(f"- Showing top {min(top_k, len(root_sizes))} roots by subtree size (node count incl. root)")
        for r in root_sizes[:top_k]:
            lines.append(f"  • {fmt_node_short(r)} | subtree size: {subtree_sizes.get(r, 1)} | path: {fmt_path(r)}")
    else:
        lines.append("- No roots detected (unexpected for a dict-root hierarchy).")
    lines.append("")

    lines.append("3) Depth analysis")
    lines.append(f"- Min depth: {min_depth}")
    lines.append(f"- Max depth: {max_depth}")
    lines.append(f"- Avg depth: {avg_depth:.2f}")
    lines.append(f"- Median depth: {med_depth:.2f}")
    if widest_depth is not None:
        lines.append(f"- Widest level: depth {widest_depth} with {widest_count} nodes")
    lines.append("")
    lines.append("Depth distribution (depth -> node count):")
    for d in sorted(depth_dist.keys()):
        lines.append(f"  - {d}: {depth_dist[d]}")
    lines.append("")

    lines.append("4) Branching / breadth analysis")
    lines.append(f"- Avg children per internal node: {avg_branching:.2f}")
    lines.append(f"- Max children on a single node: {max_branching}")
    if top_branch_nodes:
        lines.append(f"- Top {min(top_k, len(top_branch_nodes))} highest-branching nodes:")
        for nid in top_branch_nodes:
            cc = len(children_map.get(nid, []))
            lines.append(f"  • children={cc} | {fmt_path(nid)}")
    lines.append("")

    lines.append("5) Deepest leaf paths (examples)")
    if deepest_leaves:
        for nid in deepest_leaves:
            lines.append(f"  • depth {nodes[nid].depth}: {fmt_path(nid)}")
    else:
        lines.append("  - No leaves found (unexpected).")
    lines.append("")

    lines.append("6) Duplicate label analysis (same label used in multiple places)")
    lines.append(f"- Labels that appear >1 time: {len(dup_labels)}")
    lines.append(f"- Labels that appear at multiple depths: {len(multi_depth_dups)}")
    lines.append("")
    if dup_labels:
        lines.append(f"Top {min(top_k, len(dup_labels))} duplicated labels (count + sample paths):")
        for lbl, ids in dup_labels[:top_k]:
            count = len(ids)
            depths_here = sorted(label_to_depths[lbl])
            lines.append(f"  • '{lbl}' -> {count} occurrences | depths: {depths_here}")
            for sample_id in ids[:3]:
                lines.append(f"      - {fmt_path(sample_id)}")
    else:
        lines.append("- No duplicated labels detected.")
    lines.append("")

    lines.append("7) Data quality / anomalies (best-effort checks)")
    if anomalies:
        lines.append(f"- {len(anomalies)} potential issues detected (showing up to {top_k}):")
        for a in anomalies[:top_k]:
            lines.append(f"  • {a}")
        if len(anomalies) > top_k:
            lines.append(f"  ... (+{len(anomalies) - top_k} more)")
    else:
        lines.append("- No obvious anomalies detected in structure/labels.")
    lines.append("")

    lines.append("8) Interpretation hints")
    lines.append("- Many roots can mean multiple independent taxonomies or an unmerged top level.")
    lines.append("- Many duplicated labels often suggests reuse of generic terms (e.g., 'Other', 'Mixed', 'Unspecified') "
                 "or the need for disambiguating context in naming.")
    lines.append("- Very high max depth can indicate over-granularity or path-encoding rather than conceptual hierarchy.")
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# =========================
# Plotting primitives
# =========================

def plot_overview_counts(out_dir: str, title_prefix: str, total_nodes: int, roots: int, internal: int, leaves: int) -> List[str]:
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    labels = ["Total", "Internal", "Leaves", "Roots"]
    values = [total_nodes, internal, leaves, roots]
    bars = ax.bar(labels, values, alpha=0.9)
    ax.set_title(f"{title_prefix} — Node Counts")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(FuncFormatter(_int_formatter))
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), human_int(v), ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, max(values) * 1.14 if values else 1)
    fig.tight_layout()
    return save_figure(fig, os.path.join(out_dir, "01_overview_counts"))


def plot_depth_distribution_overall(out_dir: str, title_prefix: str, depth_dist: Counter) -> List[str]:
    depths = np.array(sorted(depth_dist.keys()), dtype=int)
    counts = np.array([depth_dist[d] for d in depths], dtype=float)

    fig, ax = plt.subplots(figsize=(10.8, 6.0))
    ax.bar(depths, counts, width=0.85, alpha=0.55, label="Count per depth")
    ax.plot(depths, gaussian_smooth_1d(counts, sigma=1.25), linewidth=2.2, label="Smoothed trend")
    ax.set_title(f"{title_prefix} — Depth Distribution (Overall)")
    ax.set_xlabel("Depth (absolute)")
    ax.set_ylabel("Node count")
    ax.yaxis.set_major_formatter(FuncFormatter(_int_formatter))
    ax.legend(loc="best")
    fig.tight_layout()
    return save_figure(fig, os.path.join(out_dir, "02_depth_distribution_overall"))


def plot_depth_distribution_overall_logy(out_dir: str, title_prefix: str, depth_dist: Counter) -> List[str]:
    depths = np.array(sorted(depth_dist.keys()), dtype=int)
    counts = np.array([depth_dist[d] for d in depths], dtype=float)

    fig, ax = plt.subplots(figsize=(10.8, 6.0))
    ax.bar(depths, counts, width=0.85, alpha=0.55, label="Count per depth")
    ax.plot(depths, gaussian_smooth_1d(counts, sigma=1.25), linewidth=2.2, label="Smoothed trend")
    ax.set_yscale("log")
    ax.set_title(f"{title_prefix} — Depth Distribution (Overall, log y)")
    ax.set_xlabel("Depth (absolute)")
    ax.set_ylabel("Node count (log)")
    ax.legend(loc="best")
    fig.tight_layout()
    return save_figure(fig, os.path.join(out_dir, "03_depth_distribution_overall_logy"))


def plot_width_by_depth(out_dir: str, title_prefix: str, level_counts: Counter) -> List[str]:
    depths = np.array(sorted(level_counts.keys()), dtype=int)
    counts = np.array([level_counts[d] for d in depths], dtype=float)
    cum = np.cumsum(counts)
    pct = (cum / cum[-1]) * 100.0 if cum.size else cum

    fig, ax = plt.subplots(figsize=(10.8, 6.0))
    ax.plot(depths, counts, marker="o", linewidth=2.1, label="Width (nodes at depth)")
    ax2 = ax.twinx()
    ax2.plot(depths, pct, linestyle="--", marker=".", linewidth=2.1, label="Cumulative %")

    ax.set_title(f"{title_prefix} — Width by Depth + Cumulative Coverage")
    ax.set_xlabel("Depth (absolute)")
    ax.set_ylabel("Nodes at depth")
    ax.yaxis.set_major_formatter(FuncFormatter(_int_formatter))
    ax2.set_ylabel("Cumulative % of nodes")
    ax2.set_ylim(0, 102)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    return save_figure(fig, os.path.join(out_dir, "04_width_by_depth"))


def plot_depth_hist_with_kde_like(out_dir: str, title_prefix: str, depths_all: List[int]) -> List[str]:
    """
    "Raw datapoint distribution": histogram + smoothed curve (KDE-like) over integer depths.
    """
    if not depths_all:
        return []

    depths = np.array(depths_all, dtype=int)
    dmin, dmax = int(depths.min()), int(depths.max())
    bins = np.arange(dmin, dmax + 2) - 0.5

    fig, ax = plt.subplots(figsize=(10.8, 6.0))
    hist_counts, edges, _ = ax.hist(depths, bins=bins, alpha=0.35, label="Depth samples")
    centers = (edges[:-1] + edges[1:]) / 2
    smooth = gaussian_smooth_1d(hist_counts.astype(float), sigma=1.2)
    ax.plot(centers, smooth, linewidth=2.3, label="Smoothed density-like trend")
    ax.set_title(f"{title_prefix} — Raw Depth Distribution (Histogram + Smoothed)")
    ax.set_xlabel("Depth (absolute)")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(FuncFormatter(_int_formatter))
    ax.legend(loc="best")
    fig.tight_layout()
    return save_figure(fig, os.path.join(out_dir, "05_depth_histogram_kdelike"))


def plot_branching_distribution(out_dir: str, title_prefix: str, child_counts_internal: List[int]) -> List[str]:
    arr = np.array(child_counts_internal, dtype=float)
    if arr.size == 0:
        return []

    fig, ax = plt.subplots(figsize=(10.8, 6.0))
    bins = np.arange(0, int(arr.max()) + 2) - 0.5
    hist_counts, hist_edges, _ = ax.hist(arr, bins=bins, alpha=0.45, label="Internal nodes")
    centers = (hist_edges[:-1] + hist_edges[1:]) / 2
    ax.plot(centers, gaussian_smooth_1d(hist_counts.astype(float), sigma=1.05), linewidth=2.2, label="Smoothed trend")

    ax.set_title(f"{title_prefix} — Branching Distribution (Children per Internal Node)")
    ax.set_xlabel("Children count")
    ax.set_ylabel("Number of internal nodes")
    ax.yaxis.set_major_formatter(FuncFormatter(_int_formatter))
    ax.legend(loc="best")
    fig.tight_layout()
    return save_figure(fig, os.path.join(out_dir, "01_branching_distribution"))


def plot_branching_distribution_logy(out_dir: str, title_prefix: str, child_counts_internal: List[int]) -> List[str]:
    arr = np.array(child_counts_internal, dtype=float)
    if arr.size == 0:
        return []

    fig, ax = plt.subplots(figsize=(10.8, 6.0))
    bins = np.arange(0, int(arr.max()) + 2) - 0.5
    hist_counts, hist_edges, _ = ax.hist(arr, bins=bins, alpha=0.45, label="Internal nodes")
    centers = (hist_edges[:-1] + hist_edges[1:]) / 2
    ax.plot(centers, gaussian_smooth_1d(hist_counts.astype(float), sigma=1.05), linewidth=2.2, label="Smoothed trend")
    ax.set_yscale("log")

    ax.set_title(f"{title_prefix} — Branching Distribution (log y)")
    ax.set_xlabel("Children count")
    ax.set_ylabel("Internal nodes (log)")
    ax.legend(loc="best")
    fig.tight_layout()
    return save_figure(fig, os.path.join(out_dir, "02_branching_distribution_logy"))


def plot_top_branching_nodes(out_dir: str, title_prefix: str, nodes: Dict[str, Node],
                            children_map: Dict[str, List[str]], top_ids: List[str]) -> List[str]:
    if not top_ids:
        return []
    labels = []
    values = []
    for nid in top_ids:
        values.append(len(children_map.get(nid, [])))
        labels.append(shorten_path_for_plot(" > ".join(nodes[nid].path), max_chars=105))

    labels = labels[::-1]
    values = values[::-1]

    fig, ax = plt.subplots(figsize=(13.2, 7.2))
    ax.barh(labels, values, alpha=0.9)
    ax.set_title(f"{title_prefix} — Top Branching Nodes")
    ax.set_xlabel("Children count")
    ax.xaxis.set_major_formatter(FuncFormatter(_int_formatter))
    fig.tight_layout()
    return save_figure(fig, os.path.join(out_dir, "03_top_branching_nodes"))


def plot_depth_vs_children_scatter(out_dir: str, title_prefix: str, nodes: Dict[str, Node],
                                  children_map: Dict[str, List[str]]) -> List[str]:
    internal_ids = [nid for nid in nodes if len(children_map.get(nid, [])) > 0]
    if not internal_ids:
        return []

    depths = np.array([nodes[nid].depth for nid in internal_ids], dtype=float)
    childs = np.array([len(children_map.get(nid, [])) for nid in internal_ids], dtype=float)

    fig, ax = plt.subplots(figsize=(10.8, 6.0))
    ax.scatter(depths, childs, alpha=0.20, s=18)
    ax.set_title(f"{title_prefix} — Depth vs Branching (Internal Nodes)")
    ax.set_xlabel("Depth (absolute)")
    ax.set_ylabel("Children count")
    ax.yaxis.set_major_formatter(FuncFormatter(_int_formatter))
    fig.tight_layout()
    return save_figure(fig, os.path.join(out_dir, "04_depth_vs_children_scatter"))


def plot_duplicates_top_labels(out_dir: str, title_prefix: str, dup_labels_sorted: List[Tuple[str, List[str]]], top_k: int = 15) -> List[str]:
    if not dup_labels_sorted:
        return []

    top = dup_labels_sorted[:top_k]
    labels = [t[0] for t in top][::-1]
    counts = [len(t[1]) for t in top][::-1]

    fig, ax = plt.subplots(figsize=(10.8, 6.3))
    ax.barh(labels, counts, alpha=0.9)
    ax.set_title(f"{title_prefix} — Top Duplicated Labels")
    ax.set_xlabel("Occurrences (same label in multiple places)")
    ax.xaxis.set_major_formatter(FuncFormatter(_int_formatter))
    fig.tight_layout()
    return save_figure(fig, os.path.join(out_dir, "01_top_duplicated_labels"))


def plot_duplicate_label_frequency(out_dir: str, title_prefix: str, label_counts: Counter) -> List[str]:
    counts = np.array(list(label_counts.values()), dtype=int)
    if counts.size == 0:
        return []
    dup_counts = counts[counts > 1]

    fig, ax = plt.subplots(figsize=(10.8, 6.0))
    if dup_counts.size == 0:
        ax.text(0.5, 0.5, "No duplicated labels found.", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        return save_figure(fig, os.path.join(out_dir, "02_duplicate_label_frequency"))

    max_c = int(dup_counts.max())
    bins = np.arange(1, max_c + 2) - 0.5
    hist_counts, hist_edges, _ = ax.hist(dup_counts, bins=bins, alpha=0.45, label="Duplicated labels only")
    centers = (hist_edges[:-1] + hist_edges[1:]) / 2
    ax.plot(centers, gaussian_smooth_1d(hist_counts.astype(float), sigma=1.0), linewidth=2.2, label="Smoothed trend")
    ax.set_yscale("log")

    ax.set_title(f"{title_prefix} — Duplicate Label Frequency (log y)")
    ax.set_xlabel("Occurrences per duplicated label")
    ax.set_ylabel("Number of labels (log)")
    ax.legend(loc="best")
    fig.tight_layout()
    return save_figure(fig, os.path.join(out_dir, "02_duplicate_label_frequency_logy"))


def _depth_distribution_for_subtree(nodes: Dict[str, Node], subtree_ids: List[str], base_depth: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    if not subtree_ids:
        return np.array([], dtype=int), np.array([], dtype=int)

    if base_depth is not None:
        rel_depths = [nodes[nid].depth - base_depth + 1 for nid in subtree_ids]
        dist = Counter(rel_depths)
    else:
        abs_depths = [nodes[nid].depth for nid in subtree_ids]
        dist = Counter(abs_depths)

    x = np.array(sorted(dist.keys()), dtype=int)
    y = np.array([dist[d] for d in x], dtype=int)
    return x, y


def plot_secondary_depth_distributions_panel(out_dir: str, title_prefix: str, nodes: Dict[str, Node],
                                             children_map: Dict[str, List[str]], secondary_ids: List[str],
                                             use_relative_depth: bool = True, log_y: bool = False) -> List[str]:
    """
    REQUIRED: one figure with 5 horizontal subplots (1x5).
    This function is now robust to very short depth distributions.
    """
    n = len(secondary_ids)
    if n == 0:
        return []

    fig_w = max(16.0, 3.9 * n)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 5.1), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, sid in zip(axes, secondary_ids):
        subtree = collect_subtree_nodes(sid, children_map)
        base_depth = nodes[sid].depth if use_relative_depth else None
        x, y = _depth_distribution_for_subtree(nodes, subtree, base_depth=base_depth)

        # Safe smoothing: always same length as y
        y_float = y.astype(float)
        y_smooth = gaussian_smooth_1d(y_float, sigma=1.15)

        ax.bar(x, y_float, width=0.85, alpha=0.45)
        ax.plot(x, y_smooth, linewidth=2.3)

        ax.set_title(nodes[sid].label, fontsize=12)
        ax.set_xlabel("Depth (rel.)" if use_relative_depth else "Depth (abs.)")
        if log_y:
            ax.set_yscale("log")

        total = len(subtree)
        leaves = sum(1 for nid in subtree if len(children_map.get(nid, [])) == 0)
        max_abs_depth = max((nodes[nid].depth for nid in subtree), default=nodes[sid].depth)
        max_depth = (max_abs_depth - nodes[sid].depth + 1) if use_relative_depth else max_abs_depth

        ax.text(
            0.02, 0.98,
            f"nodes: {human_int(total)}\nleaves: {human_int(leaves)}\nmax depth: {max_depth}",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", alpha=0.10, linewidth=0.0)
        )

    axes[0].set_ylabel("Node count" + (" (log)" if log_y else ""))
    axes[0].yaxis.set_major_formatter(FuncFormatter(_int_formatter))

    sup = f"{title_prefix} — Depth Distributions by Secondary Node"
    sup += " (relative)" if use_relative_depth else " (absolute)"
    if log_y:
        sup += " — log y"

    fig.suptitle(sup, fontsize=14, y=1.04)
    fig.tight_layout()

    suffix = "relative" if use_relative_depth else "absolute"
    suffix += "_logy" if log_y else "_linear"
    return save_figure(fig, os.path.join(out_dir, f"01_secondary_depth_panel_{suffix}"))


def plot_secondary_size_and_leaf_ratio(out_dir: str, title_prefix: str, nodes: Dict[str, Node],
                                       children_map: Dict[str, List[str]], secondary_ids_all: List[str]) -> List[str]:
    if not secondary_ids_all:
        return []

    rows = []
    for sid in secondary_ids_all:
        subtree = collect_subtree_nodes(sid, children_map)
        total = len(subtree)
        leaves = sum(1 for nid in subtree if len(children_map.get(nid, [])) == 0)
        rows.append((nodes[sid].label, total, leaves / total if total else 0.0))

    rows.sort(key=lambda t: t[1], reverse=True)

    labels = [r[0] for r in rows]
    sizes = np.array([r[1] for r in rows], dtype=float)
    leaf_ratios = np.array([r[2] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(max(11.0, 0.55 * len(labels) + 6), 6.4))
    ax.bar(labels, sizes, alpha=0.55, label="Subtree size (nodes)")
    ax.set_title(f"{title_prefix} — Secondary Node Subtree Sizes")
    ax.set_ylabel("Nodes in subtree")
    ax.yaxis.set_major_formatter(FuncFormatter(_int_formatter))
    ax.tick_params(axis="x", rotation=28, labelsize=10)

    ax2 = ax.twinx()
    ax2.plot(labels, leaf_ratios * 100.0, linestyle="--", marker="o", linewidth=2.1, label="Leaf ratio (%)")
    ax2.set_ylabel("Leaf ratio (%)")
    ax2.set_ylim(0, 105)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.tight_layout()
    return save_figure(fig, os.path.join(out_dir, "02_secondary_sizes_and_leaf_ratio"))


def plot_secondary_subtree_size_distribution(out_dir: str, title_prefix: str, nodes: Dict[str, Node],
                                             children_map: Dict[str, List[str]], secondary_ids_all: List[str]) -> List[str]:
    """
    Distribution of secondary subtree sizes (useful when there are many secondary nodes).
    """
    if not secondary_ids_all:
        return []
    sizes = np.array([len(collect_subtree_nodes(sid, children_map)) for sid in secondary_ids_all], dtype=float)
    if sizes.size == 0:
        return []

    fig, ax = plt.subplots(figsize=(10.8, 6.0))
    ax.hist(sizes, bins=25, alpha=0.45)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"{title_prefix} — Secondary Subtree Size Distribution (log-log)")
    ax.set_xlabel("Subtree size (nodes, log)")
    ax.set_ylabel("Count (log)")
    fig.tight_layout()
    return save_figure(fig, os.path.join(out_dir, "03_secondary_subtree_size_distribution_loglog"))


def plot_text_panel(out_dir: str, title: str, lines: List[str], filename_base: str) -> List[str]:
    fig_h = max(5.2, 0.26 * len(lines) + 2.2)
    fig, ax = plt.subplots(figsize=(13.2, fig_h))
    ax.set_title(title, fontsize=14, loc="left")
    ax.axis("off")
    text = "\n".join(lines) if lines else "(none)"
    ax.text(0.01, 0.98, text, ha="left", va="top", family="monospace", fontsize=10)
    fig.tight_layout()
    return save_figure(fig, os.path.join(out_dir, filename_base))


# =========================
# Plot suite generation
# =========================

def generate_plots(
    input_json_path: str,
    nodes: Dict[str, Node],
    children_map: Dict[str, List[str]],
    anomalies: List[str],
    root_ids: List[str],
    top_k: int,
    plot_root_dir: str,
    secondary_panel_n: int = 5,
) -> Dict[str, Any]:
    set_plot_style()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = os.path.join(plot_root_dir, run_id)
    subdirs = {
        "overview": os.path.join(run_dir, "01_overview"),
        "depth": os.path.join(run_dir, "02_depth"),
        "branching": os.path.join(run_dir, "03_branching"),
        "labels": os.path.join(run_dir, "04_labels"),
        "text": os.path.join(run_dir, "05_text"),
        "secondary": os.path.join(run_dir, "06_secondary"),
    }
    for p in subdirs.values():
        ensure_dir(p)

    base_name = os.path.splitext(os.path.basename(input_json_path))[0]
    title_prefix = f"{base_name} ({ts})"

    all_ids = list(nodes.keys())
    total = len(nodes)
    leaves = [nid for nid in all_ids if len(children_map.get(nid, [])) == 0]
    internal = [nid for nid in all_ids if len(children_map.get(nid, [])) > 0]
    depths_all = [nodes[nid].depth for nid in all_ids] or [0]
    depth_dist = Counter(depths_all)
    level_counts = Counter(depths_all)

    child_counts_internal = [len(children_map.get(nid, [])) for nid in internal]

    # Duplicates
    label_counts = Counter(n.label for n in nodes.values())
    label_to_nodes = defaultdict(list)
    label_to_depths = defaultdict(set)
    for nid, node in nodes.items():
        label_to_nodes[node.label].append(nid)
        label_to_depths[node.label].add(node.depth)
    dup_labels_sorted = [(lbl, ids) for lbl, ids in label_to_nodes.items() if len(ids) > 1]
    dup_labels_sorted.sort(key=lambda x: len(x[1]), reverse=True)

    # Root + secondary detection
    main_root_id = None
    secondary_ids_all: List[str] = []
    if root_ids:
        subtree_sizes = compute_subtree_sizes(root_ids, children_map)
        main_root_id = max(root_ids, key=lambda rid: subtree_sizes.get(rid, 1))
        secondary_ids_all = children_map.get(main_root_id, [])

    # Choose panel secondary nodes (top N by subtree size)
    secondary_choice_note = ""
    secondary_ids_for_panel: List[str] = []
    if secondary_ids_all:
        subtree_sizes_all = {sid: len(collect_subtree_nodes(sid, children_map)) for sid in secondary_ids_all}
        secondary_sorted = sorted(secondary_ids_all, key=lambda sid: subtree_sizes_all.get(sid, 1), reverse=True)
        secondary_ids_for_panel = secondary_sorted[:secondary_panel_n]
        if len(secondary_ids_all) != secondary_panel_n:
            secondary_choice_note = (
                f"NOTE: Found {len(secondary_ids_all)} secondary nodes; using top {secondary_panel_n} by subtree size for the panel."
            )

    # Top branching nodes
    top_branch_nodes = sorted(internal, key=lambda nid: len(children_map.get(nid, [])), reverse=True)[:top_k]

    # Deepest leaf paths
    deepest_leaves = sorted(leaves, key=lambda nid: nodes[nid].depth, reverse=True)[:top_k]
    deepest_lines = [f"depth {nodes[nid].depth:>2}: {' > '.join(nodes[nid].path)}" for nid in deepest_leaves]

    # Branching text lines
    branching_lines = [f"children={len(children_map.get(nid, [])):>3}: {' > '.join(nodes[nid].path)}"
                      for nid in top_branch_nodes]

    anomaly_lines = [f"- {a}" for a in anomalies[:max(60, top_k)]]

    written: List[str] = []

    # Overview
    written += plot_overview_counts(subdirs["overview"], title_prefix, total, len(root_ids), len(internal), len(leaves))

    # Depth
    written += plot_depth_distribution_overall(subdirs["depth"], title_prefix, depth_dist)
    written += plot_depth_distribution_overall_logy(subdirs["depth"], title_prefix, depth_dist)
    written += plot_width_by_depth(subdirs["depth"], title_prefix, level_counts)
    written += plot_depth_hist_with_kde_like(subdirs["depth"], title_prefix, depths_all)

    # Branching
    written += plot_branching_distribution(subdirs["branching"], title_prefix, child_counts_internal)
    written += plot_branching_distribution_logy(subdirs["branching"], title_prefix, child_counts_internal)
    written += plot_top_branching_nodes(subdirs["branching"], title_prefix, nodes, children_map, top_branch_nodes)
    written += plot_depth_vs_children_scatter(subdirs["branching"], title_prefix, nodes, children_map)

    # Labels
    written += plot_duplicates_top_labels(subdirs["labels"], title_prefix, dup_labels_sorted, top_k=min(15, max(10, top_k)))
    written += plot_duplicate_label_frequency(subdirs["labels"], title_prefix, label_counts)

    # Secondary (requested)
    if secondary_ids_all:
        written += plot_secondary_size_and_leaf_ratio(subdirs["secondary"], title_prefix, nodes, children_map, secondary_ids_all)
        written += plot_secondary_subtree_size_distribution(subdirs["secondary"], title_prefix, nodes, children_map, secondary_ids_all)

    # Required 5-horizontal panel
    written += plot_secondary_depth_distributions_panel(
        out_dir=subdirs["secondary"],
        title_prefix=title_prefix,
        nodes=nodes,
        children_map=children_map,
        secondary_ids=secondary_ids_for_panel,
        use_relative_depth=True,
        log_y=False,
    )
    written += plot_secondary_depth_distributions_panel(
        out_dir=subdirs["secondary"],
        title_prefix=title_prefix,
        nodes=nodes,
        children_map=children_map,
        secondary_ids=secondary_ids_for_panel,
        use_relative_depth=True,
        log_y=True,
    )

    # Text panels
    if secondary_choice_note:
        written += plot_text_panel(subdirs["text"], f"{title_prefix} — Notes", [secondary_choice_note], "01_notes")
    written += plot_text_panel(subdirs["text"], f"{title_prefix} — Deepest Leaf Paths (Top {top_k})", deepest_lines, "02_deepest_leaf_paths")
    written += plot_text_panel(subdirs["text"], f"{title_prefix} — Top Branching Nodes (Top {top_k})", branching_lines, "03_top_branching_nodes_text")
    if anomalies:
        written += plot_text_panel(subdirs["text"], f"{title_prefix} — Anomalies (showing up to {len(anomaly_lines)})", anomaly_lines, "04_anomalies")

    # README
    readme_path = os.path.join(run_dir, "README.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("Ontology plot suite\n")
        f.write(f"Input: {input_json_path}\n")
        f.write(f"Generated: {ts}\n\n")
        f.write("Key figures:\n")
        f.write("  - 06_secondary/01_secondary_depth_panel_relative_linear.(png|pdf)\n")
        f.write("  - 06_secondary/01_secondary_depth_panel_relative_logy.(png|pdf)\n")
        f.write("  - 02_depth/02_depth_distribution_overall.(png|pdf)\n")
        f.write("  - 02_depth/05_depth_histogram_kdelike.(png|pdf)\n")
        f.write("  - 03_branching/01_branching_distribution.(png|pdf)\n")
        f.write("  - 04_labels/01_top_duplicated_labels.(png|pdf)\n")
        if secondary_choice_note:
            f.write("\nNOTE:\n")
            f.write(secondary_choice_note + "\n")

    # Manifest
    manifest = {
        "run_id": run_id,
        "run_dir": run_dir,
        "input": input_json_path,
        "generated_at": ts,
        "main_root": nodes[main_root_id].label if main_root_id else None,
        "secondary_nodes_detected": [nodes[sid].label for sid in secondary_ids_all] if secondary_ids_all else [],
        "secondary_panel_nodes": [nodes[sid].label for sid in secondary_ids_for_panel] if secondary_ids_for_panel else [],
        "secondary_choice_note": secondary_choice_note or None,
        "counts": {
            "total_nodes": total,
            "roots": len(root_ids),
            "internal": len(internal),
            "leaves": len(leaves),
            "max_depth": int(max(depths_all)) if depths_all else 0,
            "avg_branching_internal": float(np.mean(child_counts_internal)) if child_counts_internal else 0.0,
            "max_branching_internal": int(max(child_counts_internal)) if child_counts_internal else 0,
            "duplicate_labels": int(sum(1 for c in label_counts.values() if c > 1)),
        },
        "files_written": written + [readme_path],
        "subdirs": subdirs,
    }
    manifest_path = os.path.join(run_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


# =========================
# Summary saving (TXT or JSON, not both)
# =========================

def save_summary_one_format(
    summary_dir: str,
    base_name: str,
    run_id: str,
    report_txt: str,
    summary_json: Dict[str, Any],
    summary_format: str
) -> Optional[str]:
    """
    Save either .txt OR .json, not both.
    """
    ensure_dir(summary_dir)
    summary_format = summary_format.lower().strip()

    if summary_format == "none":
        return None

    if summary_format not in ("txt", "json"):
        raise ValueError(f"Unsupported summary_format='{summary_format}'. Use one of: txt, json, none.")

    if summary_format == "txt":
        out_path = os.path.join(summary_dir, f"{base_name}__{run_id}.summary.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report_txt)
        return out_path

    out_path = os.path.join(summary_dir, f"{base_name}__{run_id}.summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)
    return out_path


# =========================
# Main callable
# =========================

def summarize_json_ontology(
    input_json_path: str,
    top_k: int = 10,
    make_plots: bool = True,
    plot_dir: Optional[str] = None,
    secondary_panel_n: int = 5,
    summary_dir: str = DEFAULT_SUMMARY_DIR,
    summary_format: str = "json",
) -> Dict[str, Any]:
    """
    - Prints report to console
    - Generates plots
    - Saves ONE summary file (.json default OR .txt if requested) in summary_dir
    - Returns a dict including paths
    """
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(input_json_path))[0]

    hierarchy = load_json_hierarchy(input_json_path)
    nodes, children_map, anomalies, root_ids = build_index(hierarchy)

    report = render_report(
        input_path=input_json_path,
        nodes=nodes,
        children_map=children_map,
        anomalies=anomalies,
        root_ids=root_ids,
        top_k=top_k,
    )
    print(report, end="")

    # Compact stats
    all_ids = list(nodes.keys())
    depths_all = [nodes[nid].depth for nid in all_ids] or [0]
    leaves = [nid for nid in all_ids if len(children_map.get(nid, [])) == 0]
    internal = [nid for nid in all_ids if len(children_map.get(nid, [])) > 0]
    label_counts = Counter(n.label for n in nodes.values())
    dup_label_count = sum(1 for c in label_counts.values() if c > 1)

    summary_json = {
        "run_id": run_id,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input": input_json_path,
        "total_nodes": len(nodes),
        "roots": len(root_ids),
        "leaves": len(leaves),
        "internal": len(internal),
        "max_depth": int(max(depths_all)) if depths_all else 0,
        "min_depth": int(min(depths_all)) if depths_all else 0,
        "avg_depth": float((sum(depths_all) / len(depths_all)) if depths_all else 0.0),
        "median_depth": float(statistics.median(depths_all)) if depths_all else 0.0,
        "duplicate_labels": int(dup_label_count),
        "anomalies_count": int(len(anomalies)),
        "top_k": int(top_k),
    }

    saved_summary_path = save_summary_one_format(
        summary_dir=summary_dir,
        base_name=base_name,
        run_id=run_id,
        report_txt=report,
        summary_json=summary_json,
        summary_format=summary_format,
    )
    if saved_summary_path:
        print(f"[Saved summary ({summary_format}) to]: {saved_summary_path}")

    plot_manifest = None
    if make_plots:
        if plot_dir is None:
            plot_dir = DEFAULT_PLOT_DIR
        ensure_dir(plot_dir)
        plot_manifest = generate_plots(
            input_json_path=input_json_path,
            nodes=nodes,
            children_map=children_map,
            anomalies=anomalies,
            root_ids=root_ids,
            top_k=top_k,
            plot_root_dir=plot_dir,
            secondary_panel_n=secondary_panel_n,
        )
        print(f"[Plots written to]: {plot_manifest['run_dir']}")

    # Return all relevant info
    out = dict(summary_json)
    out.update({
        "saved_summary_path": saved_summary_path,
        "plot_manifest": plot_manifest,
    })
    return out


# =========================
# CLI
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize a JSON subclass ontology (nested dict hierarchy) and generate publish-ready plots."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT_JSON, help="Path to the JSON file containing the subclass hierarchy.")
    parser.add_argument("--topk", type=int, default=10, help="How many items to show in ranked sections. Default=10.")
    parser.add_argument("--no-plots", action="store_true", help="Disable plot generation (default: enabled).")
    parser.add_argument("--plotdir", default=DEFAULT_PLOT_DIR, help="Base directory to write plots into (timestamped run folder inside).")
    parser.add_argument("--secondary-n", type=int, default=5, help="How many secondary nodes in the horizontal depth panel (default=5).")

    # Summary saving behavior (ONE format)
    parser.add_argument("--summarydir", default=DEFAULT_SUMMARY_DIR, help="Directory to save the summary file.")
    parser.add_argument(
        "--summary-format",
        default="json",
        choices=["json", "txt", "none"],
        help="Save summary as ONE file: json (default), txt, or none.",
    )

    args = parser.parse_args()

    summarize_json_ontology(
        input_json_path=args.input,
        top_k=max(1, args.topk),
        make_plots=(not args.no_plots),
        plot_dir=args.plotdir,
        secondary_panel_n=max(1, args.secondary_n),
        summary_dir=args.summarydir,
        summary_format=args.summary_format,
    )


if __name__ == "__main__":
    main()
