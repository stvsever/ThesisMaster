#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_hierarchically_cluster_items.py

Input (pairwise cosine similarity matrix with labels):
  /Volumes/FileLord/Projects/MASTERPROEF/large_files/03_similarity_matrix.csv

Output (hierarchical clustering JSON, cut at θ=0.35):
  /Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/cluster_criterions/results/04_semantically_clustered_items.json

Methodological target (based on the described paper pipeline):
- Use complete-linkage agglomerative clustering on a dissimilarity matrix.
- Cut the hierarchy at θ = 0.35 (dissimilarity threshold).

Important practical constraint:
- N ≈ 24k => full HAC with complete-linkage is not feasible in RAM using standard libraries.
- This script computes a cut-consistent clustering that matches the *complete-link cut constraint*:
    max intra-cluster distance <= θ
  by constructing threshold cliques on the graph where:
    distance = 1 - cosine_similarity
    so distance <= θ  <=> cosine_similarity >= (1 - θ)

Implementation approach (memory-safe, streaming-safe):
1) Stream-read the huge CSV row-by-row (never load matrix into RAM).
2) Build a sparse adjacency list of "strong similarity" edges where sim >= (1-θ).
3) Produce a deterministic clique-cover partition where every output cluster is a clique
   in that threshold graph -> guarantees complete-link cut constraint holds:
       for any two items in a cluster: sim >= (1-θ)
4) Output a simple hierarchy:
   - root node: only IDs of children (no member lists)
   - leaf cluster nodes: contain the full list of item labels in that semantic cluster

Notes:
- This yields clusters valid for a complete-linkage cut at θ (no cluster violates the threshold).
- For extremely dense graphs, memory may grow; θ=0.35 (=> sim>=0.65) is typically sparse enough.

Requirements:
- pip install numpy
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Set, Tuple, Dict, Optional

import numpy as np


# =========================
# Paths
# =========================

SIM_MATRIX_CSV = Path("/Volumes/FileLord/Projects/MASTERPROEF/large_files/03_similarity_matrix.csv")

OUTPUT_JSON = Path(
    "/utils/official/cluster_criterions/results/04_semantically_clustered_items.json"
)

# =========================
# Clustering threshold
# =========================

THETA = 0.40  # dissimilarity threshold in the thesis --> TODO: lower this to increase resolution ; during actual run
SIM_THRESHOLD = 1.0 - THETA  # cosine similarity threshold (distance = 1 - sim)

# =========================
# IO / performance
# =========================

PROGRESS_EVERY_ROWS = 250  # print progress every N rows while scanning CSV


# =========================
# Utilities
# =========================

def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _split_first_csv_field(line: str) -> Tuple[str, str]:
    """
    Fast split of a CSV line into (first_field, remainder_after_first_comma),
    correctly handling quotes and escaped double quotes in the first field.
    Assumptions:
    - Only the first field may contain commas/quotes.
    - All remaining fields are numeric and unquoted.
    """
    s = line.rstrip("\n\r")
    if not s:
        return "", ""

    if s[0] != '"':
        i = s.find(",")
        if i == -1:
            return s, ""
        return s[:i], s[i + 1 :]

    # Quoted first field
    i = 1
    out_chars: List[str] = []
    while i < len(s):
        ch = s[i]
        if ch == '"':
            # escaped quote?
            if i + 1 < len(s) and s[i + 1] == '"':
                out_chars.append('"')
                i += 2
                continue

            # end quote
            i += 1
            if i < len(s) and s[i] == ",":
                return "".join(out_chars), s[i + 1 :]
            # tolerate any stray chars until comma
            while i < len(s) and s[i] != ",":
                i += 1
            if i < len(s) and s[i] == ",":
                return "".join(out_chars), s[i + 1 :]
            return "".join(out_chars), ""
        else:
            out_chars.append(ch)
            i += 1

    return "".join(out_chars), ""


def _read_header_labels(path: Path) -> List[str]:
    """
    Read only the first row header labels (column labels).
    Header format: empty corner cell + N labels.
    We parse using Python's standard CSV reader only for the header line would be too slow without it? Actually
    header is one line only; but we avoid csv module to keep deps minimal and speed stable.
    We'll parse it field-by-field with a simple state machine.
    """
    line = path.open("r", encoding="utf-8", newline="").readline()
    if not line:
        raise ValueError("Similarity CSV is empty (no header).")

    # Parse full header safely (labels may be quoted and contain commas)
    fields: List[str] = []
    s = line.rstrip("\n\r")
    i = 0
    cur: List[str] = []
    in_quotes = False

    while i < len(s):
        ch = s[i]
        if in_quotes:
            if ch == '"':
                if i + 1 < len(s) and s[i + 1] == '"':
                    cur.append('"')
                    i += 2
                else:
                    in_quotes = False
                    i += 1
            else:
                cur.append(ch)
                i += 1
        else:
            if ch == '"':
                in_quotes = True
                i += 1
            elif ch == ",":
                fields.append("".join(cur))
                cur = []
                i += 1
            else:
                cur.append(ch)
                i += 1

    fields.append("".join(cur))

    if len(fields) < 2:
        raise ValueError("Header row too short (expected at least 2 columns).")

    # fields[0] should be empty corner cell
    labels = fields[1:]
    return labels


@dataclass
class ScanStats:
    n: int
    edges: int = 0
    rows_processed: int = 0
    start_time: float = time.time()


def _print_scan_progress(stats: ScanStats) -> None:
    elapsed = time.time() - stats.start_time
    r = stats.rows_processed
    n = stats.n
    pct = (r / n * 100.0) if n else 100.0
    rate = (r / elapsed) if elapsed > 0 else 0.0
    remaining = n - r
    eta = (remaining / rate) if rate > 0 else math.inf
    eta_str = f"{eta/60:.1f} min" if math.isfinite(eta) else "?"
    print(
        f"[SCAN] rows={r}/{n} ({pct:.2f}%) | edges(>=thr)={stats.edges} | "
        f"rate={rate:.1f} rows/s | ETA={eta_str}"
    )


# =========================
# Phase 1: Build adjacency (streaming CSV)
# =========================

def build_threshold_graph(sim_csv: Path, labels: List[str], sim_threshold: float) -> List[Set[int]]:
    """
    Stream the matrix and build undirected adjacency lists for pairs with sim >= sim_threshold.
    Never loads the whole matrix in RAM.

    Returns:
      neighbors: list of sets, neighbors[i] contains indices j (j!=i) such that sim(i,j) >= threshold
    """
    n = len(labels)
    neighbors: List[Set[int]] = [set() for _ in range(n)]
    stats = ScanStats(n=n, edges=0, rows_processed=0, start_time=time.time())

    print(f"[INFO] Building threshold graph with sim_threshold={sim_threshold:.4f} (theta={THETA:.2f})")
    print(f"[INFO] Items: n={n}")
    print(f"[INFO] Streaming from:\n  {sim_csv}")

    with sim_csv.open("r", encoding="utf-8", newline="") as f:
        header = f.readline()  # skip header
        if not header:
            raise ValueError("CSV is empty; cannot scan.")

        for i, line in enumerate(f):
            stats.rows_processed = i + 1

            row_label, numeric_str = _split_first_csv_field(line)
            if row_label != labels[i]:
                raise ValueError(
                    f"Row label mismatch at i={i}:\n"
                    f"  expected: {labels[i]}\n"
                    f"  found:    {row_label}"
                )

            sims = np.fromstring(numeric_str, sep=",", dtype=np.float32)
            if sims.shape[0] != n:
                raise ValueError(
                    f"Row length mismatch at i={i}: expected {n} numeric values, got {sims.shape[0]}"
                )

            # Only consider upper triangle j>i (avoid duplicates, skip diagonal)
            tail = sims[i + 1 :]
            if tail.size:
                hits = np.where(tail >= sim_threshold)[0]
                if hits.size:
                    js = (i + 1) + hits
                    # Add edges
                    for j in js.tolist():
                        neighbors[i].add(j)
                        neighbors[j].add(i)
                    stats.edges += int(hits.size)

            if stats.rows_processed % PROGRESS_EVERY_ROWS == 0:
                _print_scan_progress(stats)

    _print_scan_progress(stats)
    print("[OK] Threshold graph built.")
    print(f"[INFO] Total undirected edges stored: {stats.edges}")
    return neighbors


# =========================
# Phase 2: Complete-link cut clustering (clique cover)
# =========================

def greedy_clique_cover(neighbors: List[Set[int]]) -> List[List[int]]:
    """
    Deterministic greedy clique-cover:
    - Each cluster produced is a clique in the threshold graph.
    - Therefore every output cluster satisfies complete-link cut constraint at theta.

    Returns:
      clusters: list of clusters (each cluster is list of indices)
    """
    n = len(neighbors)
    degrees = [len(neighbors[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: degrees[i], reverse=True)

    unassigned: Set[int] = set(range(n))
    clusters: List[List[int]] = []

    print("[INFO] Building semantic clusters (greedy clique cover, complete-link cut-safe)...")

    processed = 0
    start = time.time()

    for u in order:
        if u not in unassigned:
            continue

        # Start a new clique with u
        clique = [u]
        unassigned.remove(u)

        # Candidates must connect to all in current clique
        candidate = neighbors[u] & unassigned

        while candidate:
            # Choose next vertex (heuristic: highest degree)
            v = max(candidate, key=lambda x: degrees[x])
            clique.append(v)
            unassigned.remove(v)

            # Keep only vertices connected to v as well (maintains clique property)
            candidate = candidate & neighbors[v]

        clusters.append(clique)
        processed += len(clique)

        if len(clusters) % 250 == 0:
            elapsed = time.time() - start
            rate = processed / elapsed if elapsed > 0 else 0.0
            remaining = n - processed
            eta = remaining / rate if rate > 0 else math.inf
            eta_str = f"{eta/60:.1f} min" if math.isfinite(eta) else "?"
            max_size = max(len(c) for c in clusters) if clusters else 0
            print(
                f"[CLUST] clusters={len(clusters)} | assigned={processed}/{n} ({processed/n*100:.2f}%) "
                f"| max_cluster_size={max_size} | rate={rate:.1f} items/s | ETA={eta_str}"
            )

        if not unassigned:
            break

    # Sanity
    assigned_total = sum(len(c) for c in clusters)
    if assigned_total != n:
        raise RuntimeError(f"Clustering did not assign all items: assigned={assigned_total}, n={n}")

    sizes = [len(c) for c in clusters]
    print("[OK] Clustering complete.")
    print(f"[INFO] clusters={len(clusters)} | min_size={min(sizes)} | max_size={max(sizes)} | mean_size={sum(sizes)/len(sizes):.2f}")
    print(f"[INFO] singletons={(sum(1 for s in sizes if s == 1))}")
    return clusters


# =========================
# Phase 3: JSON hierarchy
# =========================

def write_hierarchy_json(
    labels: List[str],
    clusters: List[List[int]],
    out_path: Path,
    theta: float,
    sim_threshold: float,
    source_csv: Path,
) -> None:
    """
    Output structure:
    {
      "meta": {...},
      "tree": {"id": "root", "children": [{"id":"c0"}, ...]},
      "clusters": {
          "c0": {"items":[...], "size": k},
          ...
      }
    }
    Root holds only IDs (no member lists). Cluster nodes are leaves and contain item lists.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cluster_nodes = [{"id": f"c{i}"} for i in range(len(clusters))]
    clusters_map: Dict[str, Dict[str, object]] = {}

    for i, cl in enumerate(clusters):
        cid = f"c{i}"
        items = [labels[idx] for idx in cl]
        clusters_map[cid] = {"size": len(items), "items": items}

    payload = {
        "meta": {
            "created_at_utc": _utc_now_iso(),
            "source_similarity_csv": str(source_csv),
            "theta_dissimilarity_cut": float(theta),
            "similarity_threshold": float(sim_threshold),
            "distance_definition": "distance = 1 - cosine_similarity",
            "linkage_target": "complete",
            "implementation_note": (
                "Streaming threshold-graph + greedy clique-cover. "
                "All clusters satisfy complete-link cut constraint: max intra-cluster distance <= theta."
            ),
            "n_items": len(labels),
            "n_clusters": len(clusters),
        },
        "tree": {
            "id": "root",
            "children": cluster_nodes
        },
        "clusters": clusters_map,
    }

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote clustered hierarchy JSON to:\n  {out_path}")


# =========================
# Main
# =========================

def main() -> None:
    print("==============================================")
    print("[START] 03_hierarchically_cluster_items")
    print("==============================================")
    print(f"[PATH] input similarity CSV:\n  {SIM_MATRIX_CSV}")
    print(f"[PATH] output JSON:\n  {OUTPUT_JSON}")
    print("----------------------------------------------")

    if not SIM_MATRIX_CSV.exists():
        raise FileNotFoundError(f"Similarity matrix CSV not found:\n  {SIM_MATRIX_CSV}")

    print("[INFO] Reading header labels...")
    labels = _read_header_labels(SIM_MATRIX_CSV)
    print(f"[OK] Read {len(labels)} labels from header.")

    print("----------------------------------------------")
    neighbors = build_threshold_graph(SIM_MATRIX_CSV, labels, SIM_THRESHOLD)

    print("----------------------------------------------")
    clusters = greedy_clique_cover(neighbors)

    print("----------------------------------------------")
    write_hierarchy_json(
        labels=labels,
        clusters=clusters,
        out_path=OUTPUT_JSON,
        theta=THETA,
        sim_threshold=SIM_THRESHOLD,
        source_csv=SIM_MATRIX_CSV,
    )

    print("==============================================")
    print("[DONE] 03_hierarchically_cluster_items")
    print("==============================================")


if __name__ == "__main__":
    main()
