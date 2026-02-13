#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
overall_mapping_analyses_fastviz.py
Fast + robust + plot-heavy analysis of predictor-ontology → criterion-cluster relevance mappings.

Key fixes vs your current script
-------------------------------
1) Speed: NO dense (clusters × predictors) expansion.
   - All “missing treated as 0” means/SDs are computed analytically from sparse sums.
   - Medians with zeros are computed exactly per group (sorting only observed values).

2) Robustness:
   - Strict column validation + dtypes + duplicate (cluster,predictor) resolution (keep max score).
   - Predictor hierarchy comes from the tree file via merge (no slow row-wise apply).

3) Much richer visualization suite:
   - Primary-domain summary bar (+ SEM)
   - BIO/PSYCHO/SOCIAL secondary “hierarchical” panels (bar + density-transparent raw points)
   - Ternary (triplot) per cluster: BIO vs PSYCHO vs SOCIAL composition
   - Per-primary secondary ranked bars (+ SEM)
   - Per-primary secondary boxplots across clusters (zeros included)
   - Heatmaps:
       * clusters × primary (3 columns)
       * clusters × secondary (per primary; top-K clusters & top-K secondaries)
   - Optional per-cluster “profile” plots (top clusters):
       * primary means + top secondary means for that cluster

Outputs
-------
- summary/*.tsv + summary/*.txt
- plots/*.png (+ optional *.pdf if enabled)

Assumptions
-----------
- Scores are in [1,1000] for emitted edges; missing edges are treated as 0 if enabled.
- Predictor universe (denominators) is defined by predictors_list.txt leaves with (ID:xxx).

Dependencies
------------
Required: numpy, pandas, matplotlib
Optional: scipy (for Friedman + Wilcoxon). If absent, uses permutation sign-flip for pairwise.

"""

from __future__ import annotations

import argparse
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Headless-safe
import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402


# -------------------------
# Optional stats (SciPy if available)
# -------------------------
try:
    from scipy.stats import friedmanchisquare, wilcoxon  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    friedmanchisquare = None
    wilcoxon = None
    _HAVE_SCIPY = False


# -------------------------
# Defaults (your paths)
# -------------------------
DEFAULT_EDGES_CSV = (
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/"
    "ontology_mappings/CRITERION/predictor_to_criterion/results/gpt-5-nano/"
    "predictor_to_criterion_edges_long.csv"
)

DEFAULT_PREDICTORS_TREE = (
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/"
    "ontology_mappings/CRITERION/predictor_to_criterion/input_lists/predictors_list.txt"
)

DEFAULT_OUT_BASE = (
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/"
    "overall_mapping_analyses/results/overall_mapping_analysis"
)


# -------------------------
# Plot style (publication-ish)
# -------------------------
def set_pub_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
        }
    )


def save_plot_dual(path_base_no_ext: str, save_pdf: bool = False) -> None:
    out_png = path_base_no_ext + ".png"
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    if save_pdf:
        out_pdf = path_base_no_ext + ".pdf"
        plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()


def ensure_dirs(out_base: str) -> Tuple[str, str]:
    out_base = os.path.abspath(out_base)
    summary_dir = os.path.join(out_base, "summary")
    plots_dir = os.path.join(out_base, "plots")
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    return summary_dir, plots_dir


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def _fmt(x) -> str:
    try:
        return f"{float(x):.4f}"
    except Exception:
        return str(x)


# -------------------------
# Predictor tree parsing
# -------------------------
_SECTION_RE = re.compile(r"^\s*\[(?P<section>[^\]]+)\]\s*$")
_NODE_RE = re.compile(
    r"^(?P<prefix>[\s│]*)(?:[└├]─)\s*(?P<name>.+?)\s*"
    r"(?:\(\s*ID\s*:\s*(?P<id>\d+)\s*\))?\s*$",
    re.UNICODE,
)

_PRIMARY_BRACKET_RE = re.compile(r"^\s*\[(BIO|PSYCHO|SOCIAL)\]\s*$", re.IGNORECASE)


def _compute_depth(prefix: str) -> int:
    if not prefix:
        return 0
    p = prefix.replace("│", " ").replace("\t", "  ")
    return max(0, len(p) // 2)


@dataclass(frozen=True)
class PredLeaf:
    predictor_id: int
    primary: str
    secondary: str
    tertiary: str
    full_path: str


def load_predictor_leaves_from_tree_txt(path: str) -> Dict[int, PredLeaf]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Predictors tree not found: {path}")

    leaves: Dict[int, PredLeaf] = {}
    current_section = ""
    stack: List[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")

            msec = _SECTION_RE.match(line)
            if msec:
                current_section = msec.group("section").strip().upper()
                stack = []
                continue

            m = _NODE_RE.match(line)
            if not m:
                continue

            name = (m.group("name") or "").strip()
            id_str = m.group("id")
            depth = _compute_depth(m.group("prefix") or "")

            if depth < len(stack):
                stack = stack[:depth]

            if not id_str:
                # category node
                if depth == len(stack):
                    stack.append(name)
                else:
                    if depth < len(stack):
                        stack[depth] = name
                    else:
                        stack.append(name)
                continue

            pid = int(id_str)
            primary = current_section if current_section else "UNKNOWN_PRIMARY"
            secondary = stack[0] if len(stack) >= 1 else "UNKNOWN_SECONDARY"
            tertiary = name if name else "UNKNOWN_TERTIARY"
            full_path = f"[{primary}] > {secondary} > {tertiary}"
            leaves[pid] = PredLeaf(pid, primary, secondary, tertiary, full_path)

    if not leaves:
        raise ValueError("No predictor IDs parsed from predictors_list.txt (expected '(ID:xxx)' leaves).")
    return leaves


def parse_hierarchy_from_full_path(full_path: str) -> Tuple[str, str, str]:
    if not isinstance(full_path, str) or not full_path.strip():
        return ("UNKNOWN_PRIMARY", "UNKNOWN_SECONDARY", "UNKNOWN_TERTIARY")

    parts = [p.strip() for p in full_path.split(">") if p.strip()]
    if not parts:
        return ("UNKNOWN_PRIMARY", "UNKNOWN_SECONDARY", "UNKNOWN_TERTIARY")

    m = _PRIMARY_BRACKET_RE.match(parts[0])
    primary = (m.group(1).upper() if m else "UNKNOWN_PRIMARY")
    secondary = parts[1] if len(parts) >= 2 else "UNKNOWN_SECONDARY"
    tertiary = parts[2] if len(parts) >= 3 else "UNKNOWN_TERTIARY"
    return (primary, secondary, tertiary)


# -------------------------
# Core CSV loading
# -------------------------
REQUIRED_COLS = [
    "cluster_id",
    "domain",
    "domain_why",
    "predictor_id",
    "score",
    "predictor_full_path",
    "cluster_item_count",
]


def load_edges_fast(edges_csv: str) -> pd.DataFrame:
    if not os.path.exists(edges_csv):
        raise FileNotFoundError(f"Edges CSV not found: {edges_csv}")

    usecols = REQUIRED_COLS
    dtype = {
        "cluster_id": "string",
        "domain": "string",
        "domain_why": "string",
        "predictor_id": "string",
        "predictor_full_path": "string",
        "cluster_item_count": "string",  # parse numeric later robustly
        "score": "string",               # parse numeric later robustly
    }
    df = pd.read_csv(edges_csv, usecols=usecols, dtype=dtype, low_memory=False)

    # Clean
    df["cluster_id"] = df["cluster_id"].astype("string").fillna("").str.strip()
    df["predictor_id"] = df["predictor_id"].astype("string").fillna("").str.strip()
    df["domain"] = df["domain"].astype("string").fillna("").str.strip()
    df["domain_why"] = df["domain_why"].astype("string").fillna("").str.strip()
    df["predictor_full_path"] = df["predictor_full_path"].astype("string").fillna("").str.strip()

    df["cluster_item_count"] = pd.to_numeric(df["cluster_item_count"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    # Keep valid score rows only
    df = df.dropna(subset=["score"])
    df = df[(df["score"] >= 1) & (df["score"] <= 1000)]

    # Parse predictor_id to int
    df["predictor_id_int"] = pd.to_numeric(df["predictor_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["predictor_id_int"])
    df["predictor_id_int"] = df["predictor_id_int"].astype("int32")

    # Resolve duplicates: keep row with max score for each (cluster,predictor)
    # (fast via idxmax)
    g = df.groupby(["cluster_id", "predictor_id_int"], sort=False)["score"]
    idx = g.idxmax()
    df = df.loc[idx].copy()

    # Normalize types
    df["score"] = df["score"].astype("float32")
    df["cluster_item_count"] = df["cluster_item_count"].astype("float32")

    return df


# -------------------------
# Fast “zeros included” statistics helpers
# -------------------------
def _sample_std_from_sums(sum_x: float, sum_x2: float, n: int) -> float:
    if n <= 1:
        return float("nan")
    # sample variance: (Σx² - (Σx)²/n) / (n-1)
    var = (sum_x2 - (sum_x * sum_x) / float(n)) / float(n - 1)
    return float(np.sqrt(var)) if var > 0 else 0.0


def median_with_zeros(sorted_nonzero: np.ndarray, zeros: int, total_n: int) -> float:
    """
    Exact median of a multiset consisting of `zeros` zeros + `sorted_nonzero` positives.
    `sorted_nonzero` MUST be sorted ascending.
    """
    if total_n <= 0:
        return float("nan")
    if zeros >= total_n:
        return 0.0

    k = int(sorted_nonzero.size)
    if zeros + k != total_n:
        # keep robust if something slightly off
        k = min(k, max(0, total_n - zeros))
        sorted_nonzero = sorted_nonzero[:k]

    if total_n % 2 == 1:
        mid = total_n // 2  # 0-index
        if mid < zeros:
            return 0.0
        j = mid - zeros
        j = min(max(j, 0), sorted_nonzero.size - 1)
        return float(sorted_nonzero[j])
    else:
        mid2 = total_n // 2
        mid1 = mid2 - 1

        def _val(pos: int) -> float:
            if pos < zeros:
                return 0.0
            j = pos - zeros
            j = min(max(j, 0), sorted_nonzero.size - 1)
            return float(sorted_nonzero[j])

        return 0.5 * (_val(mid1) + _val(mid2))


def is_integer_like(scores: np.ndarray, tol: float = 1e-6) -> bool:
    if scores.size == 0:
        return False
    return bool(np.all(np.abs(scores - np.round(scores)) <= tol))


# -------------------------
# Aggregations (no dense matrix)
# -------------------------
def agg_by_cluster(df_edges: pd.DataFrame, n_predictors_total: int, treat_missing_as_zero: bool) -> pd.DataFrame:
    """
    Cluster stats across predictors.
    - If treat_missing_as_zero: denominator is n_predictors_total for means/SD; missing predictors contribute 0.
    - Else: stats are computed across observed edges only (no implicit zeros).
    """
    if not treat_missing_as_zero:
        g = (
            df_edges.groupby("cluster_id", dropna=False)["score"]
            .agg(
                mean_score="mean",
                median_score="median",
                std_score=lambda s: s.std(ddof=1),
                max_score="max",
                min_score="min",
                pct_nonzero=lambda s: float((s > 0).mean() * 100.0),
                n_edges="size",
            )
            .reset_index()
            .sort_values("mean_score", ascending=False)
        )
        return g

    gb = df_edges.groupby("cluster_id", dropna=False)
    sum_x = gb["score"].sum().astype(float)
    sum_x2 = gb["score"].apply(lambda s: float(np.square(s.astype(float)).sum()))
    k_nonzero = gb["score"].size().astype(int)

    denom = int(n_predictors_total)
    zeros = denom - k_nonzero

    # Median: needs sorting the observed values per cluster
    med = []
    mn = []
    mx = []
    for cid, sub in gb:
        arr = np.sort(sub["score"].to_numpy(dtype=float))
        z = int(denom - arr.size)
        med.append(median_with_zeros(arr, z, denom))
        mn.append(0.0 if z > 0 else float(arr[0]) if arr.size else 0.0)
        mx.append(float(arr[-1]) if arr.size else 0.0)

    mean = (sum_x / float(denom)).to_numpy()
    std = np.array([_sample_std_from_sums(float(s), float(s2), denom) for s, s2 in zip(sum_x.to_numpy(), sum_x2.to_numpy())])

    out = pd.DataFrame(
        {
            "cluster_id": sum_x.index.astype(str),
            "mean_score": mean,
            "median_score": np.array(med, dtype=float),
            "std_score": std,
            "max_score": np.array(mx, dtype=float),
            "min_score": np.array(mn, dtype=float),
            "pct_nonzero": (k_nonzero.to_numpy(dtype=float) / float(denom) * 100.0),
            "n_edges": k_nonzero.to_numpy(dtype=int),
        }
    ).sort_values("mean_score", ascending=False)
    return out


def agg_by_predictor(df_edges: pd.DataFrame, pred_map: pd.DataFrame, n_clusters: int, treat_missing_as_zero: bool) -> pd.DataFrame:
    """
    Predictor stats across clusters.
    - If treat_missing_as_zero: denominator is n_clusters for means/SD; missing clusters contribute 0.
    - Else: stats computed across observed edges only.
    """
    if not treat_missing_as_zero:
        g = (
            df_edges.groupby(["predictor_id_int", "primary", "secondary", "tertiary"], dropna=False)["score"]
            .agg(
                mean_score="mean",
                median_score="median",
                std_score=lambda s: s.std(ddof=1),
                max_score="max",
                min_score="min",
                pct_nonzero=lambda s: float((s > 0).mean() * 100.0),
                n_edges="size",
            )
            .reset_index()
            .rename(columns={"predictor_id_int": "predictor_id"})
            .sort_values(["mean_score", "pct_nonzero", "max_score"], ascending=[False, False, False])
        )
        return g

    denom = int(n_clusters)

    gb = df_edges.groupby(["predictor_id_int", "primary", "secondary", "tertiary"], dropna=False)
    sum_x = gb["score"].sum().astype(float)
    sum_x2 = gb["score"].apply(lambda s: float(np.square(s.astype(float)).sum()))
    k_nonzero = gb["score"].size().astype(int)

    med = []
    mn = []
    mx = []
    idx_rows = []
    for key, sub in gb:
        pid, prim, sec, ter = key
        arr = np.sort(sub["score"].to_numpy(dtype=float))
        z = int(denom - arr.size)
        med.append(median_with_zeros(arr, z, denom))
        mn.append(0.0 if z > 0 else float(arr[0]) if arr.size else 0.0)
        mx.append(float(arr[-1]) if arr.size else 0.0)
        idx_rows.append((pid, prim, sec, ter))

    mean = (sum_x / float(denom)).to_numpy()
    std = np.array([_sample_std_from_sums(float(s), float(s2), denom) for s, s2 in zip(sum_x.to_numpy(), sum_x2.to_numpy())])

    out = pd.DataFrame(
        {
            "predictor_id": [int(r[0]) for r in idx_rows],
            "primary": [str(r[1]) for r in idx_rows],
            "secondary": [str(r[2]) for r in idx_rows],
            "tertiary": [str(r[3]) for r in idx_rows],
            "mean_score": mean,
            "median_score": np.array(med, dtype=float),
            "std_score": std,
            "max_score": np.array(mx, dtype=float),
            "min_score": np.array(mn, dtype=float),
            "pct_nonzero": (k_nonzero.to_numpy(dtype=float) / float(denom) * 100.0),
            "n_edges": k_nonzero.to_numpy(dtype=int),
        }
    ).sort_values(["mean_score", "pct_nonzero", "max_score"], ascending=[False, False, False])

    return out


def build_per_cluster_primary(
    df_edges: pd.DataFrame,
    n_pred_primary: Dict[str, int],
    clusters: List[str],
    treat_missing_as_zero: bool,
) -> pd.DataFrame:
    primaries = [p for p in ["BIO", "PSYCHO", "SOCIAL"] if p in n_pred_primary]

    if not primaries:
        return pd.DataFrame(columns=["cluster_id", "primary", "cluster_mean_score"])

    if not treat_missing_as_zero:
        per = (
            df_edges.groupby(["cluster_id", "primary"], dropna=False)["score"]
            .mean()
            .reset_index(name="cluster_mean_score")
        )
        # Ensure all primaries appear per cluster? (not necessary for no-zero mode)
        return per

    sums = (
        df_edges.groupby(["cluster_id", "primary"], dropna=False)["score"]
        .sum()
        .reset_index(name="sum_score")
    )

    base = pd.MultiIndex.from_product([clusters, primaries], names=["cluster_id", "primary"]).to_frame(index=False)
    out = base.merge(sums, on=["cluster_id", "primary"], how="left")
    out["sum_score"] = out["sum_score"].fillna(0.0).astype(float)

    # Divide by primary predictor count
    out["denom"] = out["primary"].map(lambda p: float(n_pred_primary.get(p, 1)))
    out["cluster_mean_score"] = out["sum_score"] / out["denom"]
    out = out.drop(columns=["sum_score", "denom"])
    return out


def build_per_cluster_secondary(
    df_edges: pd.DataFrame,
    n_pred_secondary: Dict[Tuple[str, str], int],
    treat_missing_as_zero: bool,
) -> pd.DataFrame:
    if not treat_missing_as_zero:
        per = (
            df_edges.groupby(["cluster_id", "primary", "secondary"], dropna=False)["score"]
            .mean()
            .reset_index(name="cluster_mean_score")
        )
        return per

    sums = (
        df_edges.groupby(["cluster_id", "primary", "secondary"], dropna=False)["score"]
        .sum()
        .reset_index(name="sum_score")
    )

    # Divide by (primary,secondary) predictor count
    def _den(row) -> float:
        return float(n_pred_secondary.get((row["primary"], row["secondary"]), 1))

    denom = sums.apply(_den, axis=1)
    sums["cluster_mean_score"] = sums["sum_score"].astype(float) / denom.astype(float)
    return sums.drop(columns=["sum_score"])


def summarize_secondary_across_clusters(
    per_cluster_secondary_nonzero: pd.DataFrame,
    n_clusters: int,
    treat_missing_as_zero: bool,
) -> pd.DataFrame:
    """
    Summary per (primary, secondary) across clusters.
    If treat_missing_as_zero: missing cluster-secondary combos count as 0.
    """
    if not treat_missing_as_zero:
        g = (
            per_cluster_secondary_nonzero.groupby(["primary", "secondary"], dropna=False)["cluster_mean_score"]
            .agg(
                n_clusters="size",
                mean_across_clusters="mean",
                median_across_clusters="median",
                std_across_clusters=lambda s: s.std(ddof=1),
                pct_clusters_nonzero=lambda s: float((s > 0).mean() * 100.0),
                mean_nonzero_clusters=lambda s: float(s[s > 0].mean()) if (s > 0).any() else 0.0,
                max_cluster_mean="max",
            )
            .reset_index()
            .sort_values(["primary", "mean_across_clusters"], ascending=[True, False])
        )
        return g

    denom = int(n_clusters)
    gb = per_cluster_secondary_nonzero.groupby(["primary", "secondary"], dropna=False)
    sum_x = gb["cluster_mean_score"].sum().astype(float)
    sum_x2 = gb["cluster_mean_score"].apply(lambda s: float(np.square(s.astype(float)).sum()))
    k = gb["cluster_mean_score"].size().astype(int)

    # Median needs sorting observed values per (primary,secondary)
    med = []
    mx = []
    for key, sub in gb:
        arr = np.sort(sub["cluster_mean_score"].to_numpy(dtype=float))
        z = int(denom - arr.size)
        med.append(median_with_zeros(arr, z, denom))
        mx.append(float(arr[-1]) if arr.size else 0.0)

    mean = (sum_x / float(denom)).to_numpy()
    std = np.array([_sample_std_from_sums(float(s), float(s2), denom) for s, s2 in zip(sum_x.to_numpy(), sum_x2.to_numpy())])

    out = pd.DataFrame(
        {
            "primary": [str(k0[0]) for k0 in sum_x.index],
            "secondary": [str(k0[1]) for k0 in sum_x.index],
            "n_clusters": np.full(shape=(len(sum_x),), fill_value=denom, dtype=int),
            "mean_across_clusters": mean,
            "median_across_clusters": np.array(med, dtype=float),
            "std_across_clusters": std,
            "pct_clusters_nonzero": (k.to_numpy(dtype=float) / float(denom) * 100.0),
            "mean_nonzero_clusters": (sum_x.to_numpy(dtype=float) / np.maximum(k.to_numpy(dtype=float), 1.0)),
            "max_cluster_mean": np.array(mx, dtype=float),
        }
    ).sort_values(["primary", "mean_across_clusters"], ascending=[True, False])

    return out


def summarize_primary_across_clusters(
    per_cluster_primary: pd.DataFrame,
    n_clusters: int,
    treat_missing_as_zero: bool,
) -> pd.DataFrame:
    if not treat_missing_as_zero:
        g = (
            per_cluster_primary.groupby(["primary"], dropna=False)["cluster_mean_score"]
            .agg(
                n_clusters="size",
                mean_across_clusters="mean",
                median_across_clusters="median",
                std_across_clusters=lambda s: s.std(ddof=1),
                pct_clusters_nonzero=lambda s: float((s > 0).mean() * 100.0),
                mean_nonzero_clusters=lambda s: float(s[s > 0].mean()) if (s > 0).any() else 0.0,
                max_cluster_mean="max",
            )
            .reset_index()
            .sort_values("mean_across_clusters", ascending=False)
        )
        return g

    # In zero-mode, per_cluster_primary is complete (cluster×primary). So normal groupby is fine.
    g = (
        per_cluster_primary.groupby(["primary"], dropna=False)["cluster_mean_score"]
        .agg(
            n_clusters="size",
            mean_across_clusters="mean",
            median_across_clusters="median",
            std_across_clusters=lambda s: s.std(ddof=1),
            pct_clusters_nonzero=lambda s: float((s > 0).mean() * 100.0),
            mean_nonzero_clusters=lambda s: float(s[s > 0].mean()) if (s > 0).any() else 0.0,
            max_cluster_mean="max",
        )
        .reset_index()
        .sort_values("mean_across_clusters", ascending=False)
    )
    return g


# -------------------------
# Primary-domain significance
# -------------------------
def _holm_correction(pvals: List[float]) -> List[float]:
    m = len(pvals)
    idx = np.argsort(pvals)
    p_sorted = np.array([pvals[i] for i in idx], dtype=float)
    adj_sorted = np.empty(m, dtype=float)

    running_max = 0.0
    for i in range(m):
        factor = (m - i)
        val = factor * p_sorted[i]
        running_max = max(running_max, val)
        adj_sorted[i] = min(1.0, running_max)

    adj = np.empty(m, dtype=float)
    for rank_pos, original_i in enumerate(idx):
        adj[original_i] = adj_sorted[rank_pos]
    return adj.tolist()


def _paired_permutation_pvalue(diff: np.ndarray, n_perm: int = 20000, seed: int = 0) -> float:
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    obs = abs(diff.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, diff.size))
    perm = abs((signs * diff).mean(axis=1))
    return float((perm >= obs).mean())


def compute_primary_domain_significance(
    per_cluster_primary: pd.DataFrame,
    summary_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    pv = per_cluster_primary.pivot(index="cluster_id", columns="primary", values="cluster_mean_score")

    present = [c for c in ["BIO", "PSYCHO", "SOCIAL"] if c in pv.columns]
    pv = pv[present].copy()

    # If any missing in non-zero mode, drop NAs; in zero-mode this should be complete already.
    pv = pv.dropna(axis=0, how="any")

    desc = []
    for p in present:
        s = pv[p].astype(float)
        desc.append(
            {
                "primary": p,
                "n_clusters": int(s.shape[0]),
                "mean": float(s.mean()),
                "median": float(s.median()),
                "std": float(s.std(ddof=1)),
                "min": float(s.min()),
                "max": float(s.max()),
            }
        )
    df_desc = pd.DataFrame(desc).sort_values("mean", ascending=False)

    global_lines = []
    global_p = float("nan")
    global_stat = float("nan")
    if len(present) == 3 and pv.shape[0] >= 5:
        if _HAVE_SCIPY:
            stat, pval = friedmanchisquare(pv["BIO"].values, pv["PSYCHO"].values, pv["SOCIAL"].values)
            global_stat, global_p = float(stat), float(pval)
            global_lines.append(f"Friedman chi2={_fmt(global_stat)}  p={_fmt(global_p)}  (paired; n={pv.shape[0]} clusters)")
        else:
            global_lines.append("Friedman: SciPy not available; global test skipped; pairwise permutation tests reported instead.")
    else:
        global_lines.append("Friedman: not run (need BIO+PSYCHO+SOCIAL and >=5 paired clusters).")

    pairs = []
    pvals_raw = []
    for a, b in [("BIO", "PSYCHO"), ("BIO", "SOCIAL"), ("PSYCHO", "SOCIAL")]:
        if a not in present or b not in present:
            continue

        diff = (pv[a] - pv[b]).astype(float).to_numpy()
        diff = diff[np.isfinite(diff)]

        if diff.size < 5:
            p_raw = float("nan")
            method = "insufficient_n"
        else:
            if _HAVE_SCIPY:
                try:
                    res = wilcoxon(pv[a].values, pv[b].values, alternative="two-sided", zero_method="wilcox")
                    p_raw = float(res.pvalue)
                    method = "wilcoxon"
                except Exception:
                    p_raw = _paired_permutation_pvalue(diff, n_perm=20000, seed=0)
                    method = "perm_signflip"
            else:
                p_raw = _paired_permutation_pvalue(diff, n_perm=20000, seed=0)
                method = "perm_signflip"

        pvals_raw.append(p_raw)
        pairs.append(
            {
                "A": a,
                "B": b,
                "n_clusters": int(diff.size),
                "mean_diff_A_minus_B": float(np.mean(diff)) if diff.size else float("nan"),
                "median_diff_A_minus_B": float(np.median(diff)) if diff.size else float("nan"),
                "p_raw": p_raw,
                "method": method,
            }
        )

    p_raw_list = [r["p_raw"] for r in pairs]
    finite_idx = [i for i, p in enumerate(p_raw_list) if np.isfinite(p)]
    p_finite = [p_raw_list[i] for i in finite_idx]
    p_adj = _holm_correction(p_finite) if p_finite else []
    for k, i in enumerate(finite_idx):
        pairs[i]["p_holm"] = p_adj[k]
    for i in range(len(pairs)):
        if "p_holm" not in pairs[i]:
            pairs[i]["p_holm"] = float("nan")

    df_pairs = pd.DataFrame(pairs)

    best_mean = df_desc.sort_values("mean", ascending=False).iloc[0]["primary"] if not df_desc.empty else "UNKNOWN"
    best_median = df_desc.sort_values("median", ascending=False).iloc[0]["primary"] if not df_desc.empty else "UNKNOWN"

    lines = []
    lines.append("PRIMARY DOMAIN SIGNIFICANCE (paired per cluster)")
    lines.append("-" * 80)
    lines.append(f"clusters used (complete cases): {pv.shape[0]}")
    lines.append("")
    lines.append("Descriptives (per-cluster primary means):")
    lines.append(df_desc.to_csv(sep="\t", index=False))
    lines.append("")
    lines.append("Global test:")
    lines.extend(global_lines)
    lines.append("")
    lines.append("Pairwise tests (A minus B):")
    lines.append(df_pairs.to_csv(sep="\t", index=False) if not df_pairs.empty else "(no pairwise tests run)")
    lines.append("")
    lines.append(f"Best by MEAN:   {best_mean}")
    lines.append(f"Best by MEDIAN: {best_median}")

    txt = "\n".join(lines)
    write_text(os.path.join(summary_dir, "primary_domain_significance.txt"), txt)
    return df_desc, df_pairs, txt


# -------------------------
# Plot helpers
# -------------------------
def _style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", length=4, width=0.8)
    ax.tick_params(axis="both", which="minor", length=2, width=0.6)
    ax.grid(True, which="major", axis="x", linewidth=0.6, alpha=0.18)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_axisbelow(True)


def _density_alphas_1d(x: np.ndarray,
                       x_min: float,
                       x_max: float,
                       bins: int = 35,
                       alpha_min: float = 0.06,
                       alpha_max: float = 0.65,
                       gamma: float = 0.55) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([]), np.array([])

    counts, edges = np.histogram(x, bins=bins, range=(x_min, x_max))
    idx = np.clip(np.digitize(x, edges) - 1, 0, len(counts) - 1)
    c = counts[idx].astype(float)

    cmax = c.max() if c.size else 1.0
    norm = np.ones_like(c) if cmax <= 0 else (c / cmax) ** gamma

    alphas = alpha_min + (alpha_max - alpha_min) * norm
    sizes = 8.0 + 18.0 * norm
    return alphas, sizes


def barh_ranked_with_error(
    labels: List[str],
    values: List[float],
    errors: List[float],
    title: str,
    xlabel: str,
    out_base_no_ext: str,
    xlim: Optional[Tuple[float, float]] = None,
    save_pdf: bool = False,
) -> None:
    n = len(labels)
    height = max(4.5, 0.35 * n + 1.2)
    plt.figure(figsize=(10.5, height))
    y = list(range(n))
    plt.barh(y, values, xerr=errors, capsize=3, linewidth=0.0)
    plt.gca().invert_yaxis()
    plt.yticks(y, labels)
    plt.title(title)
    plt.xlabel(xlabel)
    if xlim is not None:
        plt.xlim(*xlim)
    _style_axis(plt.gca())
    save_plot_dual(out_base_no_ext, save_pdf=save_pdf)


# -------------------------
# Plots (richer set)
# -------------------------
def plot_primary_summary(by_primary: pd.DataFrame, plots_dir: str, save_pdf: bool = False) -> None:
    gp = by_primary.sort_values("mean_across_clusters", ascending=False).copy()
    labels = gp["primary"].astype(str).tolist()
    means = gp["mean_across_clusters"].astype(float).tolist()

    std = gp["std_across_clusters"].fillna(0.0).astype(float).tolist()
    n = gp["n_clusters"].astype(float).replace(0, 1).tolist()
    sem = [s / (nn ** 0.5) for s, nn in zip(std, n)]

    out_base = os.path.join(plots_dir, "primary_domains_mean_scores")
    barh_ranked_with_error(
        labels=labels,
        values=means,
        errors=sem,
        title="Primary domains: mean relevance across criterion clusters",
        xlabel="Mean score per cluster (missing treated as 0 if enabled)",
        out_base_no_ext=out_base,
        xlim=(0, 1000),
        save_pdf=save_pdf,
    )


def plot_hierarchical_bps_secondary_with_raw_density(
    by_secondary: pd.DataFrame,
    per_cluster_secondary: pd.DataFrame,
    plots_dir: str,
    topn_per_primary: int = 28,
    save_pdf: bool = False,
) -> None:
    primaries = [p for p in ["BIO", "PSYCHO", "SOCIAL"] if (by_secondary["primary"] == p).any()]
    if not primaries:
        return

    x_min, x_max = 0.0, 1000.0
    rng = np.random.default_rng(0)

    max_n = 1
    for p in primaries:
        n = by_secondary[by_secondary["primary"] == p].shape[0]
        max_n = max(max_n, min(int(topn_per_primary), int(n)))
    fig_h = max(4.2, 0.28 * max_n + 1.6)
    fig_w = 6.0 * len(primaries)
    fig, axes = plt.subplots(1, len(primaries), figsize=(fig_w, fig_h), sharex=True)
    if len(primaries) == 1:
        axes = [axes]

    for ax, primary in zip(axes, primaries):
        g = by_secondary[by_secondary["primary"] == primary].copy()
        g = g.sort_values("mean_across_clusters", ascending=False).head(int(topn_per_primary))
        if g.empty:
            ax.axis("off")
            continue

        labels = g["secondary"].astype(str).tolist()
        means = g["mean_across_clusters"].astype(float).to_numpy()
        std = g["std_across_clusters"].fillna(0.0).astype(float).to_numpy()
        ncl = g["n_clusters"].astype(float).replace(0, 1).to_numpy()
        sem = std / np.sqrt(ncl)

        y = np.arange(len(labels))
        ax.barh(y, means, xerr=sem, capsize=2.8, linewidth=0.0, alpha=0.55)
        ax.invert_yaxis()

        raw = per_cluster_secondary[per_cluster_secondary["primary"] == primary].copy()
        raw = raw[raw["secondary"].astype(str).isin(labels)].copy()
        raw["secondary"] = raw["secondary"].astype(str)

        y_map = {sec: i for i, sec in enumerate(labels)}

        for sec in labels:
            x = raw.loc[raw["secondary"] == sec, "cluster_mean_score"].astype(float).to_numpy()
            x = x[np.isfinite(x)]
            if x.size == 0:
                continue

            alphas, sizes = _density_alphas_1d(x, x_min, x_max, bins=34, alpha_min=0.05, alpha_max=0.62, gamma=0.55)
            y0 = y_map[sec]
            jitter = rng.normal(0.0, 0.075, size=x.size)
            yy = y0 + jitter

            if not hasattr(ax, "_hier_color"):
                ax._hier_color = ax._get_lines.get_next_color()
            color = ax._hier_color

            for xi, yi, ai, si in zip(x, yy, alphas, sizes):
                ax.scatter([xi], [yi], s=float(si), alpha=float(ai), edgecolors="none", color=color)

        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlim(x_min, x_max)
        ax.set_title(f"{primary} → secondary domains\n(bars=mean across clusters; points=per-cluster means)")
        ax.set_xlabel("Mean score per cluster (missing treated as 0 if enabled)")
        _style_axis(ax)
        for v in [100, 400, 700, 900]:
            ax.axvline(v, linewidth=0.6, alpha=0.10)

    plt.suptitle("Hierarchical relevance: BIO / PSYCHO / SOCIAL → secondary domains", y=1.02)
    save_plot_dual(os.path.join(plots_dir, "hierarchical_BPS_secondary_density_points"), save_pdf=save_pdf)


def plot_score_histograms(
    df_edges: pd.DataFrame,
    n_clusters: int,
    n_predictors_total: int,
    treat_missing_as_zero: bool,
    plots_dir: str,
    save_pdf: bool = False,
) -> None:
    scores = df_edges["score"].astype(float).to_numpy()
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return

    xmin, xmax = 0.0, 1000.0

    def _hist(ax, data, bins, x_range):
        data = np.asarray(data, dtype=float)
        data = data[np.isfinite(data)]
        if data.size == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return

        counts, edges = np.histogram(data, bins=bins, range=x_range)
        widths = np.diff(edges)
        lefts = edges[:-1]

        hmax = counts.max() if counts.size else 0
        gamma = 0.65
        alpha_min, alpha_max = 0.12, 0.92
        alphas = np.full_like(counts, 0.2, dtype=float) if hmax <= 0 else alpha_min + (alpha_max - alpha_min) * ((counts / hmax) ** gamma)

        color = ax._get_lines.get_next_color()
        for l, w, c, a in zip(lefts, widths, counts, alphas):
            if c == 0:
                continue
            ax.bar(l, c, width=w, align="edge", color=color, edgecolor=color, linewidth=0.6, alpha=float(a))

    def _style_ax(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", which="major", linewidth=0.6, alpha=0.22)
        ax.tick_params(axis="both", which="major", length=4, width=0.8)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.set_axisbelow(True)

    # 1) Histogram of observed nonzero edges (always)
    plt.figure(figsize=(6.8, 3.8))
    ax = plt.gca()
    _hist(ax, scores, bins=60, x_range=(xmin, xmax))
    ax.set_title("Emitted edge scores histogram (observed non-zero edges)", pad=10)
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    _style_ax(ax)
    save_plot_dual(os.path.join(plots_dir, "score_histogram_observed_edges"), save_pdf=save_pdf)

    # 2) Histogram of full matrix (including zeros) if enabled
    if treat_missing_as_zero:
        total_rows = int(n_clusters) * int(n_predictors_total)
        zeros = max(0, total_rows - int(scores.size))

        # If integer-like, build exact discrete histogram 0..1000
        if is_integer_like(scores):
            ints = np.round(scores).astype(int)
            counts = np.bincount(ints, minlength=1001).astype(np.int64)
            counts[0] += zeros

            plt.figure(figsize=(6.8, 3.8))
            ax = plt.gca()
            x = np.arange(0, 1001)
            # Plot as bars with default style (dense but OK)
            ax.bar(x, counts, width=1.0, align="center", linewidth=0.0)
            ax.set_xlim(0, 1000)
            ax.set_title("Full matrix score histogram (including implicit zeros)", pad=10)
            ax.set_xlabel("Score (0..1000)")
            ax.set_ylabel("Count")
            _style_ax(ax)
            save_plot_dual(os.path.join(plots_dir, "score_histogram_full_matrix_including_zeros"), save_pdf=save_pdf)
        else:
            # Approx: show zeros spike separately + nonzero histogram
            plt.figure(figsize=(6.8, 3.8))
            ax = plt.gca()
            _hist(ax, scores, bins=60, x_range=(xmin, xmax))
            ax.set_title("Full matrix: nonzero histogram (zeros shown as annotation)", pad=10)
            ax.set_xlabel("Score")
            ax.set_ylabel("Count")
            ax.text(0.02, 0.95, f"Implicit zeros: {zeros:,}", transform=ax.transAxes, ha="left", va="top")
            _style_ax(ax)
            save_plot_dual(os.path.join(plots_dir, "score_histogram_full_matrix_nonzero_plus_zero_annotation"), save_pdf=save_pdf)


def plot_secondary_ranked_bars_per_primary(
    by_secondary: pd.DataFrame,
    plots_dir: str,
    topn: int = 25,
    save_pdf: bool = False,
) -> None:
    for primary in ["BIO", "PSYCHO", "SOCIAL"]:
        g = by_secondary[by_secondary["primary"] == primary].copy()
        if g.empty:
            continue
        g = g.sort_values("mean_across_clusters", ascending=False).head(int(topn))

        labels = g["secondary"].astype(str).tolist()
        means = g["mean_across_clusters"].astype(float).tolist()
        std = g["std_across_clusters"].fillna(0.0).astype(float).tolist()
        n = g["n_clusters"].astype(float).replace(0, 1).tolist()
        sem = [s / (nn ** 0.5) for s, nn in zip(std, n)]

        barh_ranked_with_error(
            labels=labels,
            values=means,
            errors=sem,
            title=f"{primary}: secondary domains ranked (mean across clusters)",
            xlabel="Mean score per cluster",
            out_base_no_ext=os.path.join(plots_dir, f"secondary_ranked_bar_{primary}"),
            xlim=(0, 1000),
            save_pdf=save_pdf,
        )


def plot_secondary_boxplots_per_primary(
    per_cluster_secondary_nonzero: pd.DataFrame,
    by_secondary: pd.DataFrame,
    clusters: List[str],
    treat_missing_as_zero: bool,
    plots_dir: str,
    topn: int = 20,
    save_pdf: bool = False,
) -> None:
    """
    Boxplots across clusters for top secondaries per primary.
    In zero-mode: missing clusters are filled with 0 so the distribution is correct.
    """
    if not treat_missing_as_zero:
        # In no-zero mode, boxplots reflect only observed combos; still useful but different meaning.
        pass

    cluster_set = set(clusters)
    n_clusters = len(clusters)

    for primary in ["BIO", "PSYCHO", "SOCIAL"]:
        top = by_secondary[by_secondary["primary"] == primary].copy()
        if top.empty:
            continue
        top = top.sort_values("mean_across_clusters", ascending=False).head(int(topn))
        secs = top["secondary"].astype(str).tolist()
        if not secs:
            continue

        raw = per_cluster_secondary_nonzero[
            (per_cluster_secondary_nonzero["primary"] == primary)
            & (per_cluster_secondary_nonzero["secondary"].astype(str).isin(secs))
        ].copy()
        raw["secondary"] = raw["secondary"].astype(str)

        data = []
        labels = []
        for sec in secs:
            sub = raw[raw["secondary"] == sec]
            vals = sub["cluster_mean_score"].astype(float).to_numpy()
            vals = vals[np.isfinite(vals)]
            if treat_missing_as_zero:
                # Fill implicit zeros for clusters not present
                k = len(set(sub["cluster_id"].astype(str).tolist()))
                zeros = max(0, n_clusters - k)
                if zeros > 0:
                    vals = np.concatenate([vals, np.zeros(zeros, dtype=float)])
            data.append(vals)
            labels.append(sec)

        height = max(4.8, 0.34 * len(labels) + 1.5)
        plt.figure(figsize=(11.0, height))
        ax = plt.gca()
        ax.boxplot(data, vert=False, labels=labels, showfliers=False)
        ax.invert_yaxis()
        ax.set_title(f"{primary}: secondary distributions across clusters (boxplots; zeros included if enabled)")
        ax.set_xlabel("Mean score per cluster (secondary)")
        ax.set_xlim(0, 1000)
        ax.grid(True, axis="x", linewidth=0.6, alpha=0.18)
        ax.set_axisbelow(True)
        save_plot_dual(os.path.join(plots_dir, f"secondary_boxplots_{primary}"), save_pdf=save_pdf)


def plot_primary_heatmap_clusters(
    per_cluster_primary: pd.DataFrame,
    by_cluster: pd.DataFrame,
    plots_dir: str,
    top_clusters: int = 60,
    save_pdf: bool = False,
) -> None:
    """
    Heatmap: clusters × primary (3 columns).
    Clusters are sorted by overall mean_score.
    """
    top = by_cluster.sort_values("mean_score", ascending=False).head(int(top_clusters))
    cids = top["cluster_id"].astype(str).tolist()
    piv = per_cluster_primary.pivot(index="cluster_id", columns="primary", values="cluster_mean_score").copy()
    piv = piv.reindex(index=cids)

    cols = [c for c in ["BIO", "PSYCHO", "SOCIAL"] if c in piv.columns]
    if not cols:
        return
    mat = piv[cols].to_numpy(dtype=float)

    plt.figure(figsize=(6.4, max(4.5, 0.18 * len(cids) + 1.8)))
    ax = plt.gca()
    im = ax.imshow(mat, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    ax.set_yticks(np.arange(len(cids)))
    ax.set_yticklabels(cids)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_title("Cluster × primary mean scores (top clusters)")
    ax.set_xlabel("Primary domain")
    ax.set_ylabel("Cluster (sorted by overall mean)")
    save_plot_dual(os.path.join(plots_dir, "heatmap_clusters_by_primary"), save_pdf=save_pdf)


def plot_secondary_heatmaps_per_primary(
    per_cluster_secondary_nonzero: pd.DataFrame,
    by_secondary: pd.DataFrame,
    by_cluster: pd.DataFrame,
    clusters: List[str],
    treat_missing_as_zero: bool,
    plots_dir: str,
    top_clusters: int = 40,
    top_secondaries: int = 30,
    save_pdf: bool = False,
) -> None:
    """
    Heatmap: clusters × secondary (per primary).
    In zero-mode, missing combos are filled with 0 (correct matrix).
    """
    cluster_rank = by_cluster.sort_values("mean_score", ascending=False).head(int(top_clusters))
    cids = cluster_rank["cluster_id"].astype(str).tolist()
    n_clusters = len(cids)

    for primary in ["BIO", "PSYCHO", "SOCIAL"]:
        sec_rank = by_secondary[by_secondary["primary"] == primary].copy()
        if sec_rank.empty:
            continue
        sec_rank = sec_rank.sort_values("mean_across_clusters", ascending=False).head(int(top_secondaries))
        secs = sec_rank["secondary"].astype(str).tolist()
        if not secs:
            continue

        raw = per_cluster_secondary_nonzero[
            (per_cluster_secondary_nonzero["primary"] == primary)
            & (per_cluster_secondary_nonzero["secondary"].astype(str).isin(secs))
            & (per_cluster_secondary_nonzero["cluster_id"].astype(str).isin(cids))
        ].copy()
        raw["secondary"] = raw["secondary"].astype(str)
        raw["cluster_id"] = raw["cluster_id"].astype(str)

        piv = raw.pivot(index="cluster_id", columns="secondary", values="cluster_mean_score")
        piv = piv.reindex(index=cids, columns=secs)

        if treat_missing_as_zero:
            piv = piv.fillna(0.0)

        mat = piv.to_numpy(dtype=float)

        plt.figure(figsize=(max(7.5, 0.22 * len(secs) + 2.5), max(4.8, 0.18 * n_clusters + 2.2)))
        ax = plt.gca()
        im = ax.imshow(mat, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        ax.set_yticks(np.arange(len(cids)))
        ax.set_yticklabels(cids)
        ax.set_xticks(np.arange(len(secs)))
        ax.set_xticklabels(secs, rotation=90)
        ax.set_title(f"{primary}: cluster × secondary mean scores (top clusters & top secondaries)")
        ax.set_xlabel("Secondary domain")
        ax.set_ylabel("Cluster")
        save_plot_dual(os.path.join(plots_dir, f"heatmap_clusters_by_secondary_{primary}"), save_pdf=save_pdf)


def plot_ternary_primary_triplot(
    per_cluster_primary: pd.DataFrame,
    by_cluster: pd.DataFrame,
    plots_dir: str,
    top_clusters: int = 120,
    save_pdf: bool = False,
) -> None:
    """
    Ternary plot of BIO/PSYCHO/SOCIAL composition per cluster.
    Uses barycentric coordinates in Matplotlib (no extra deps).
    Points: top clusters by overall mean_score.
    """
    top = by_cluster.sort_values("mean_score", ascending=False).head(int(top_clusters))
    cids = top["cluster_id"].astype(str).tolist()

    piv = per_cluster_primary.pivot(index="cluster_id", columns="primary", values="cluster_mean_score").copy()
    piv = piv.reindex(index=cids).fillna(0.0)

    for col in ["BIO", "PSYCHO", "SOCIAL"]:
        if col not in piv.columns:
            piv[col] = 0.0
    b = piv["BIO"].to_numpy(dtype=float)
    p = piv["PSYCHO"].to_numpy(dtype=float)
    s = piv["SOCIAL"].to_numpy(dtype=float)

    tot = b + p + s
    tot = np.where(tot <= 0, 1.0, tot)
    bn = b / tot
    pn = p / tot
    sn = s / tot

    # Triangle vertices: BIO=(0,0), PSYCHO=(1,0), SOCIAL=(0.5, sqrt(3)/2)
    h = np.sqrt(3) / 2.0
    x = pn + 0.5 * sn
    y = h * sn

    # Point sizes based on overall mean_score
    ms = top["mean_score"].astype(float).to_numpy()
    ms = np.where(np.isfinite(ms), ms, 0.0)
    # mild scaling
    sizes = 18.0 + 70.0 * (ms / max(ms.max(), 1.0)) ** 0.6

    plt.figure(figsize=(7.2, 6.6))
    ax = plt.gca()

    # Draw triangle
    tri_x = [0.0, 1.0, 0.5, 0.0]
    tri_y = [0.0, 0.0, h, 0.0]
    ax.plot(tri_x, tri_y, linewidth=1.2)

    # Light grid lines (optional; subtle)
    for t in [0.2, 0.4, 0.6, 0.8]:
        # Lines parallel to base: sn = t
        ax.plot([0.5 * t, 1.0 - 0.5 * t], [h * t, h * t], linewidth=0.7, alpha=0.12)

    ax.scatter(x, y, s=sizes, alpha=0.55, edgecolors="none")

    # Labels
    ax.text(-0.03, -0.03, "BIO", ha="left", va="top")
    ax.text(1.03, -0.03, "PSYCHO", ha="right", va="top")
    ax.text(0.5, h + 0.04, "SOCIAL", ha="center", va="bottom")

    ax.set_title("Primary-domain composition per cluster (ternary / triplot)\n(points sized by overall cluster mean)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, h + 0.08)
    ax.set_aspect("equal", "box")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    save_plot_dual(os.path.join(plots_dir, "ternary_triplot_primary_composition_clusters"), save_pdf=save_pdf)


def plot_cluster_profiles(
    df_edges: pd.DataFrame,
    per_cluster_primary: pd.DataFrame,
    n_pred_secondary: Dict[Tuple[str, str], int],
    by_cluster: pd.DataFrame,
    plots_dir: str,
    treat_missing_as_zero: bool,
    n_profiles: int = 12,
    topn_secondaries: int = 12,
    save_pdf: bool = False,
) -> None:
    """
    For top clusters, create compact “profile” plots:
      - Primary means (BIO/PSYCHO/SOCIAL)
      - Top secondary means for that cluster (across all primaries)
    """
    top = by_cluster.sort_values("mean_score", ascending=False).head(int(n_profiles))
    cids = top["cluster_id"].astype(str).tolist()
    if not cids:
        return

    # primary pivot
    piv_p = per_cluster_primary.pivot(index="cluster_id", columns="primary", values="cluster_mean_score").fillna(0.0)

    for cid in cids:
        # Primary means
        prim_vals = {p: 0.0 for p in ["BIO", "PSYCHO", "SOCIAL"]}
        if cid in piv_p.index:
            for p in prim_vals.keys():
                if p in piv_p.columns:
                    prim_vals[p] = float(piv_p.loc[cid, p])

        # Secondary means for this cluster (computed from sparse sums; zeros ignored in top list)
        sub = df_edges[df_edges["cluster_id"].astype(str) == cid]
        if sub.empty:
            continue
        sums = (
            sub.groupby(["primary", "secondary"], dropna=False)["score"]
            .sum()
            .reset_index(name="sum_score")
        )

        if treat_missing_as_zero:
            denom = sums.apply(lambda r: float(n_pred_secondary.get((r["primary"], r["secondary"]), 1)), axis=1)
            sums["cluster_mean_score"] = sums["sum_score"].astype(float) / denom.astype(float)
        else:
            # no-zero mode: just mean of observed scores for that cluster-secondary
            means = (
                sub.groupby(["primary", "secondary"], dropna=False)["score"]
                .mean()
                .reset_index(name="cluster_mean_score")
            )
            sums = means

        sums = sums.sort_values("cluster_mean_score", ascending=False).head(int(topn_secondaries))
        labels = [f"{r.primary}:{r.secondary}" for r in sums.itertuples(index=False)]
        vals = sums["cluster_mean_score"].astype(float).tolist()

        # Plot: two horizontal bar charts stacked vertically
        fig_h = 6.2
        plt.figure(figsize=(10.8, fig_h))

        # Top: primary
        ax1 = plt.subplot(2, 1, 1)
        prim_order = ["BIO", "PSYCHO", "SOCIAL"]
        y = np.arange(len(prim_order))
        ax1.barh(y, [prim_vals[p] for p in prim_order], linewidth=0.0)
        ax1.set_yticks(y)
        ax1.set_yticklabels(prim_order)
        ax1.invert_yaxis()
        ax1.set_xlim(0, 1000)
        ax1.set_title(f"Cluster {cid}: primary means")
        ax1.set_xlabel("Mean score per cluster (primary)")
        ax1.grid(True, axis="x", linewidth=0.6, alpha=0.18)
        ax1.set_axisbelow(True)

        # Bottom: top secondaries
        ax2 = plt.subplot(2, 1, 2)
        yy = np.arange(len(labels))
        ax2.barh(yy, vals, linewidth=0.0)
        ax2.set_yticks(yy)
        ax2.set_yticklabels(labels)
        ax2.invert_yaxis()
        ax2.set_xlim(0, 1000)
        ax2.set_title(f"Cluster {cid}: top secondary means (nonzero)")
        ax2.set_xlabel("Mean score per cluster (secondary)")
        ax2.grid(True, axis="x", linewidth=0.6, alpha=0.18)
        ax2.set_axisbelow(True)

        save_plot_dual(os.path.join(plots_dir, f"cluster_profile_{cid}"), save_pdf=save_pdf)


# -------------------------
# Overview text (fast; no dense)
# -------------------------
def compute_overview_fast(
    df_edges: pd.DataFrame,
    n_clusters: int,
    n_predictors_total: int,
    treat_missing_as_zero: bool,
) -> str:
    scores = df_edges["score"].astype(float).to_numpy()
    scores = scores[np.isfinite(scores)]

    if not treat_missing_as_zero:
        n_rows = int(scores.size)
        if n_rows == 0:
            return "OVERALL MAPPING ANALYSIS\n(no valid scores)"
        mean = float(scores.mean())
        med = float(np.median(scores))
        std = float(scores.std(ddof=1)) if n_rows > 1 else float("nan")
        mn = float(scores.min())
        mx = float(scores.max())

        lines = [
            "OVERALL MAPPING ANALYSIS",
            "-" * 80,
            f"treat_missing_as_zero: {treat_missing_as_zero}",
            f"clusters_total: {n_clusters}",
            f"predictors_total: {n_predictors_total}",
            f"rows_used: {n_rows} (observed edges only)",
            "",
            "Score summary (observed edges only):",
            f"  mean:   {_fmt(mean)}",
            f"  median: {_fmt(med)}",
            f"  std:    {_fmt(std)}",
            f"  min:    {_fmt(mn)}",
            f"  max:    {_fmt(mx)}",
        ]
        return "\n".join(lines)

    total_rows = int(n_clusters) * int(n_predictors_total)
    nonzero_n = int(scores.size)
    zeros_n = max(0, total_rows - nonzero_n)

    sum_x = float(scores.sum())
    sum_x2 = float(np.square(scores).sum())
    mean = sum_x / float(total_rows) if total_rows > 0 else float("nan")
    std = _sample_std_from_sums(sum_x, sum_x2, total_rows) if total_rows > 1 else float("nan")
    mn = 0.0 if zeros_n > 0 else float(scores.min()) if nonzero_n > 0 else 0.0
    mx = float(scores.max()) if nonzero_n > 0 else 0.0

    # Median: exact if integer-like, otherwise exact via “zeros + sorted nonzero”
    if nonzero_n == 0:
        med = 0.0
    else:
        arr = np.sort(scores.astype(float))
        med = median_with_zeros(arr, zeros_n, total_rows)

    # Buckets (exact, integer-friendly)
    bucket_lines = []
    if nonzero_n > 0 and is_integer_like(scores):
        ints = np.round(scores).astype(int)
        counts = np.bincount(ints, minlength=1001).astype(np.int64)
        counts[0] += zeros_n
        pct = lambda c: (float(c) / float(total_rows) * 100.0) if total_rows > 0 else float("nan")

        bucket_lines = [
            "Score buckets (% of rows):",
            f"  1000:      {_fmt(pct(counts[1000]))}%",
            f"  900–999:   {_fmt(pct(counts[900:1000].sum()))}%",
            f"  700–899:   {_fmt(pct(counts[700:900].sum()))}%",
            f"  400–699:   {_fmt(pct(counts[400:700].sum()))}%",
            f"  100–399:   {_fmt(pct(counts[100:400].sum()))}%",
            f"  1–99:      {_fmt(pct(counts[1:100].sum()))}%",
            f"  0:         {_fmt(pct(counts[0]))}%",
        ]
    else:
        # fallback approximate buckets
        pct = lambda x: (float(x) / float(total_rows) * 100.0) if total_rows > 0 else float("nan")
        bucket_lines = [
            "Score buckets (% of rows) [approx; non-integer scores]:",
            f"  1000:      {_fmt(pct(int(np.sum(scores == 1000)) ))}%",
            f"  900–999:   {_fmt(pct(int(np.sum((scores >= 900) & (scores <= 999))) ))}%",
            f"  700–899:   {_fmt(pct(int(np.sum((scores >= 700) & (scores <= 899))) ))}%",
            f"  400–699:   {_fmt(pct(int(np.sum((scores >= 400) & (scores <= 699))) ))}%",
            f"  100–399:   {_fmt(pct(int(np.sum((scores >= 100) & (scores <= 399))) ))}%",
            f"  1–99:      {_fmt(pct(int(np.sum((scores >= 1) & (scores <= 99))) ))}%",
            f"  0:         {_fmt(pct(zeros_n))}%",
        ]

    lines = [
        "OVERALL MAPPING ANALYSIS",
        "-" * 80,
        f"treat_missing_as_zero: {treat_missing_as_zero}",
        f"clusters_total: {n_clusters}",
        f"predictors_total: {n_predictors_total}",
        f"matrix_rows_total: {total_rows}",
        f"observed_edges_rows: {nonzero_n}",
        f"implicit_zero_rows:  {zeros_n}",
        "",
        "Score summary (full matrix, zeros included):",
        f"  mean:   {_fmt(mean)}",
        f"  median: {_fmt(med)}",
        f"  std:    {_fmt(std)}",
        f"  min:    {_fmt(mn)}",
        f"  max:    {_fmt(mx)}",
        "",
        f"nonzero_rate: {_fmt((nonzero_n / float(total_rows) * 100.0) if total_rows else float('nan'))}%",
        "",
        *bucket_lines,
    ]
    return "\n".join(lines)


# -------------------------
# Main
# -------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges-csv", type=str, default=DEFAULT_EDGES_CSV)
    parser.add_argument("--predictors-tree", type=str, default=DEFAULT_PREDICTORS_TREE)
    parser.add_argument("--out-base", type=str, default=DEFAULT_OUT_BASE)

    parser.add_argument("--treat-missing-as-zero", type=int, default=1)
    parser.add_argument("--save-pdf", type=int, default=0)

    # Plot richness controls
    parser.add_argument("--topn-secondary-per-primary-hier", type=int, default=28)
    parser.add_argument("--topn-secondary-bars", type=int, default=25)
    parser.add_argument("--topn-secondary-box", type=int, default=20)

    parser.add_argument("--heatmap-top-clusters", type=int, default=40)
    parser.add_argument("--heatmap-top-secondaries", type=int, default=30)

    parser.add_argument("--ternary-top-clusters", type=int, default=120)

    parser.add_argument("--cluster-profiles", type=int, default=1)
    parser.add_argument("--n-cluster-profiles", type=int, default=12)
    parser.add_argument("--cluster-profile-topn-secondaries", type=int, default=12)

    # Edge listing
    parser.add_argument("--topk-edges-global", type=int, default=200)
    parser.add_argument("--topk-edges-per-cluster", type=int, default=30)

    args = parser.parse_args()
    treat_missing_as_zero = bool(int(args.treat_missing_as_zero))
    save_pdf = bool(int(args.save_pdf))
    do_cluster_profiles = bool(int(args.cluster_profiles))

    set_pub_style()
    summary_dir, plots_dir = ensure_dirs(args.out_base)

    t0 = time.perf_counter()

    print("=== CONFIG ===")
    print(f"edges_csv: {args.edges_csv}")
    print(f"predictors_tree: {args.predictors_tree}")
    print(f"out_base: {os.path.abspath(args.out_base)}")
    print(f"treat_missing_as_zero: {treat_missing_as_zero}")
    print(f"scipy_available: {_HAVE_SCIPY}")
    print("==============")

    # Predictor ontology
    pid2leaf = load_predictor_leaves_from_tree_txt(args.predictors_tree)
    pred_map = pd.DataFrame(
        [
            {
                "predictor_id_int": int(pid),
                "predictor_id": str(int(pid)),
                "primary": leaf.primary,
                "secondary": leaf.secondary,
                "tertiary": leaf.tertiary,
                "full_path_tree": leaf.full_path,
            }
            for pid, leaf in pid2leaf.items()
        ]
    )
    n_predictors_total = int(pred_map.shape[0])
    n_pred_primary = pred_map.groupby("primary").size().to_dict()
    n_pred_secondary = pred_map.groupby(["primary", "secondary"]).size().to_dict()

    print(f"[load] predictor leaves parsed: {n_predictors_total}")

    # Load sparse edges
    df_edges = load_edges_fast(args.edges_csv)
    print(f"[load] sparse edges loaded (deduped): {len(df_edges)}")

    # Attach hierarchy via fast merge
    df_edges = df_edges.merge(
        pred_map[["predictor_id_int", "primary", "secondary", "tertiary"]],
        on="predictor_id_int",
        how="left",
        validate="many_to_one",
    )
    df_edges["hierarchy_source"] = np.where(df_edges["primary"].notna(), "tree", "path")

    # Fallback for rare missing predictor IDs
    miss = df_edges["primary"].isna()
    if bool(miss.any()):
        # Parse from predictor_full_path only for those rows
        parsed = df_edges.loc[miss, "predictor_full_path"].apply(parse_hierarchy_from_full_path)
        df_edges.loc[miss, "primary"] = [p[0] for p in parsed]
        df_edges.loc[miss, "secondary"] = [p[1] for p in parsed]
        df_edges.loc[miss, "tertiary"] = [p[2] for p in parsed]

    # Normalize hierarchy strings
    for c in ["primary", "secondary", "tertiary"]:
        df_edges[c] = df_edges[c].astype("string").fillna(f"UNKNOWN_{c.upper()}").str.strip()

    clusters = sorted(df_edges["cluster_id"].astype(str).unique().tolist())
    n_clusters = len(clusters)

    print(f"[ok] clusters observed: {n_clusters}")

    # Overview
    overview_txt = compute_overview_fast(df_edges, n_clusters, n_predictors_total, treat_missing_as_zero)
    write_text(os.path.join(summary_dir, "overview.txt"), overview_txt)

    # Aggregations
    by_cluster = agg_by_cluster(df_edges, n_predictors_total, treat_missing_as_zero=treat_missing_as_zero)
    by_predictor = agg_by_predictor(df_edges, pred_map, n_clusters, treat_missing_as_zero=treat_missing_as_zero)

    per_cluster_primary = build_per_cluster_primary(df_edges, n_pred_primary, clusters, treat_missing_as_zero=treat_missing_as_zero)
    by_primary = summarize_primary_across_clusters(per_cluster_primary, n_clusters, treat_missing_as_zero=treat_missing_as_zero)

    per_cluster_secondary = build_per_cluster_secondary(df_edges, n_pred_secondary, treat_missing_as_zero=treat_missing_as_zero)
    by_secondary = summarize_secondary_across_clusters(per_cluster_secondary, n_clusters, treat_missing_as_zero=treat_missing_as_zero)

    # Primary significance
    df_primary_desc, df_primary_pairs, _ = compute_primary_domain_significance(per_cluster_primary, summary_dir)

    # Write tables
    by_predictor_out = by_predictor.copy()
    by_predictor_out.to_csv(os.path.join(summary_dir, "by_predictor.tsv"), sep="\t", index=False)
    by_secondary.to_csv(os.path.join(summary_dir, "by_secondary.tsv"), sep="\t", index=False)
    by_primary.to_csv(os.path.join(summary_dir, "by_primary.tsv"), sep="\t", index=False)
    by_cluster.to_csv(os.path.join(summary_dir, "by_cluster.tsv"), sep="\t", index=False)
    per_cluster_secondary.to_csv(os.path.join(summary_dir, "per_cluster_secondary.tsv"), sep="\t", index=False)
    per_cluster_primary.to_csv(os.path.join(summary_dir, "per_cluster_primary.tsv"), sep="\t", index=False)
    df_primary_desc.to_csv(os.path.join(summary_dir, "primary_desc.tsv"), sep="\t", index=False)
    df_primary_pairs.to_csv(os.path.join(summary_dir, "primary_pairwise_tests.tsv"), sep="\t", index=False)

    # Ranked summaries
    def write_ranked_text(title: str, df: pd.DataFrame, cols: List[str], topn: int, fname: str) -> None:
        head = df.loc[:, cols].head(int(topn))
        txt = title + "\n" + ("-" * 80) + "\n" + head.to_csv(sep="\t", index=False)
        write_text(os.path.join(summary_dir, fname), txt)

    write_ranked_text(
        title="TOP PREDICTOR LEAVES BY MEAN SCORE (across clusters; missing treated as 0 if enabled)",
        df=by_predictor_out,
        cols=["predictor_id", "primary", "secondary", "tertiary", "mean_score", "pct_nonzero", "max_score"],
        topn=100,
        fname="top_predictors_by_mean_score.txt",
    )

    write_ranked_text(
        title="TOP SECONDARY DOMAINS BY MEAN SCORE (mean across clusters; missing treated as 0 if enabled)",
        df=by_secondary.sort_values("mean_across_clusters", ascending=False),
        cols=["primary", "secondary", "mean_across_clusters", "pct_clusters_nonzero", "mean_nonzero_clusters", "max_cluster_mean"],
        topn=200,
        fname="top_secondary_domains_by_mean_score.txt",
    )

    write_ranked_text(
        title="TOP CRITERION CLUSTERS BY MEAN SCORE (across all predictors; missing treated as 0 if enabled)",
        df=by_cluster,
        cols=["cluster_id", "mean_score", "median_score", "pct_nonzero", "max_score"],
        topn=200,
        fname="top_clusters_by_mean_score.txt",
    )

    # Top emitted edges (global + per cluster)
    top_global = (
        df_edges.sort_values("score", ascending=False)
        .head(int(args.topk_edges_global))
        .loc[:, ["cluster_id", "score", "predictor_id", "predictor_id_int", "primary", "secondary", "tertiary", "predictor_full_path", "domain"]]
    )
    write_text(
        os.path.join(summary_dir, "top_edges_global.txt"),
        "TOP EDGES (GLOBAL; emitted by LLM)\n" + ("-" * 80) + "\n" + top_global.to_csv(sep="\t", index=False),
    )

    chunks: List[str] = []
    for cid, g in df_edges.groupby("cluster_id", dropna=False):
        g2 = g.sort_values("score", ascending=False).head(int(args.topk_edges_per_cluster))
        chunks.append(
            f"CLUSTER {cid}\n"
            + g2.loc[:, ["score", "predictor_id_int", "primary", "secondary", "tertiary", "predictor_full_path", "domain"]].to_csv(sep="\t", index=False)
        )
        chunks.append("")
    write_text(
        os.path.join(summary_dir, "top_edges_per_cluster.txt"),
        "TOP EDGES PER CLUSTER (emitted by LLM)\n" + ("=" * 80) + "\n\n" + "\n".join(chunks),
    )

    # QC
    qc = []
    qc.append("QUALITY CHECKS")
    qc.append("-" * 80)
    qc.append(f"hierarchy_source (edges):\n{df_edges['hierarchy_source'].value_counts(dropna=False).to_string()}")
    qc.append("")
    for col in ["primary", "secondary", "tertiary"]:
        qc.append(f"UNKNOWN {col}: {int((df_edges[col].astype(str).str.startswith('UNKNOWN')).sum())} / {len(df_edges)}")
    qc.append("")
    if treat_missing_as_zero:
        total_rows = n_clusters * n_predictors_total
        qc.append(f"zero_fraction (implicit): {_fmt((1.0 - (len(df_edges) / float(total_rows))) * 100.0)}%")
        qc.append(f"nonzero_fraction (observed): {_fmt((len(df_edges) / float(total_rows)) * 100.0)}%")
    write_text(os.path.join(summary_dir, "quality_checks.txt"), "\n".join(qc))

    # Plots
    plot_primary_summary(by_primary, plots_dir, save_pdf=save_pdf)
    plot_hierarchical_bps_secondary_with_raw_density(
        by_secondary=by_secondary,
        per_cluster_secondary=per_cluster_secondary,
        plots_dir=plots_dir,
        topn_per_primary=int(args.topn_secondary_per_primary_hier),
        save_pdf=save_pdf,
    )
    plot_score_histograms(df_edges, n_clusters, n_predictors_total, treat_missing_as_zero, plots_dir, save_pdf=save_pdf)

    plot_secondary_ranked_bars_per_primary(by_secondary, plots_dir, topn=int(args.topn_secondary_bars), save_pdf=save_pdf)
    plot_secondary_boxplots_per_primary(
        per_cluster_secondary_nonzero=per_cluster_secondary,
        by_secondary=by_secondary,
        clusters=clusters,
        treat_missing_as_zero=treat_missing_as_zero,
        plots_dir=plots_dir,
        topn=int(args.topn_secondary_box),
        save_pdf=save_pdf,
    )

    plot_primary_heatmap_clusters(
        per_cluster_primary=per_cluster_primary,
        by_cluster=by_cluster,
        plots_dir=plots_dir,
        top_clusters=int(args.heatmap_top_clusters),
        save_pdf=save_pdf,
    )
    plot_secondary_heatmaps_per_primary(
        per_cluster_secondary_nonzero=per_cluster_secondary,
        by_secondary=by_secondary,
        by_cluster=by_cluster,
        clusters=clusters,
        treat_missing_as_zero=treat_missing_as_zero,
        plots_dir=plots_dir,
        top_clusters=int(args.heatmap_top_clusters),
        top_secondaries=int(args.heatmap_top_secondaries),
        save_pdf=save_pdf,
    )

    plot_ternary_primary_triplot(
        per_cluster_primary=per_cluster_primary,
        by_cluster=by_cluster,
        plots_dir=plots_dir,
        top_clusters=int(args.ternary_top_clusters),
        save_pdf=save_pdf,
    )

    if do_cluster_profiles:
        plot_cluster_profiles(
            df_edges=df_edges,
            per_cluster_primary=per_cluster_primary,
            n_pred_secondary=n_pred_secondary,
            by_cluster=by_cluster,
            plots_dir=plots_dir,
            treat_missing_as_zero=treat_missing_as_zero,
            n_profiles=int(args.n_cluster_profiles),
            topn_secondaries=int(args.cluster_profile_topn_secondaries),
            save_pdf=save_pdf,
        )

    t1 = time.perf_counter()
    print("\n=== DONE ===")
    print(f"[ok] summaries → {summary_dir}")
    print(f"[ok] plots     → {plots_dir}")
    print(f"[time] total seconds: {t1 - t0:.2f}")


if __name__ == "__main__":
    main()
