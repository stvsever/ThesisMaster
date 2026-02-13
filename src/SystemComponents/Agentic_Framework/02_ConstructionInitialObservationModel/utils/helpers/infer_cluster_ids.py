#!/usr/bin/env python3
"""
infer_cluster_ids.py

Infer cluster IDs for ontology leaf nodes from a JSON file containing clustered criterion strings.

Why this exists:
- Leaf IDs can repeat in an ontology (duplicates), so we disambiguate using the *two nearest context nodes*
  above the leaf (the immediate parent and grandparent in the path).
- Cluster items in your JSON look like:
    "<leaf> (context: <ctx1> < <ctx2>) — criterion"

This script:
1) Loads the clustering JSON.
2) Builds an index keyed by (leaf, ctx1, ctx2) with robust normalization.
3) Infers the cluster id for each provided leaf-path string.
4) Tests on your semicolon-separated list in __main__ and prints a dict to console.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

# Normalize a variety of unicode dashes to ASCII "-" for matching stability.
_DASHES = {
    "\u2010": "-",  # hyphen
    "\u2011": "-",  # non-breaking hyphen
    "\u2012": "-",  # figure dash
    "\u2013": "-",  # en dash
    "\u2014": "-",  # em dash
    "\u2212": "-",  # minus sign
}
_DASH_TRANSLATION = str.maketrans(_DASHES)

# Matches: "<leaf> (context: <ctx1> < <ctx2>) — criterion"
# Keep it permissive; ctx2 may contain parentheses, punctuation, etc.
_ITEM_RE = re.compile(
    r"^\s*(?P<leaf>.*?)\s*"
    r"\(context:\s*(?P<ctx1>.*?)\s*<\s*(?P<ctx2>.*?)\)\s*"
    r"(?:—|-)\s*criterion\s*$",
    re.IGNORECASE,
)


def _norm(text: str) -> str:
    """Casefold + dash-normalize + whitespace-normalize."""
    text = (text or "").translate(_DASH_TRANSLATION)
    # Collapse whitespace (including tabs/newlines) to single spaces
    text = " ".join(text.split())
    return text.casefold().strip()


def parse_leaf_path(path_str: str) -> Tuple[str, str, str]:
    """
    Parse an ontology path like:
      "A / B / C / *leaf_id"

    Returns (leaf_id, ctx1, ctx2) where:
      ctx1 = immediate parent segment (one above leaf)
      ctx2 = grandparent segment (two above leaf)

    Uses only the last two context nodes, per your constraint.
    """
    parts = [p.strip() for p in path_str.split("/") if p.strip()]
    if len(parts) < 3:
        raise ValueError(
            f"Path must have at least 3 segments (ctx2/ctx1/leaf). Got: {path_str!r}"
        )

    leaf = parts[-1].lstrip("*").strip()
    ctx1 = parts[-2].strip()
    ctx2 = parts[-3].strip()
    return leaf, ctx1, ctx2


def parse_cluster_item(item_str: str) -> Optional[Tuple[str, str, str]]:
    """
    From a cluster item string like:
      "self_referential_ruminative_content (context: with cognitive rumination < generalized anxiety disorder) — criterion"
    extract (leaf, ctx1, ctx2). Returns None if it doesn't match expected pattern.
    """
    m = _ITEM_RE.match(item_str or "")
    if not m:
        return None
    leaf = m.group("leaf").strip()
    ctx1 = m.group("ctx1").strip()
    ctx2 = m.group("ctx2").strip()
    return leaf, ctx1, ctx2


def _find_cluster_container(data: dict) -> Dict[str, dict]:
    """
    The JSON you described starts with {"meta":..., "tree":...} and (somewhere) cluster payloads keyed by "c###".
    This function tries common containers first, then falls back to selecting top-level "c\\d+" keys.
    """
    # Common container keys (defensive)
    for key in (
        "clusters",
        "cluster_items",
        "items_by_cluster",
        "cluster_to_items",
        "clusters_by_id",
        "by_cluster",
    ):
        v = data.get(key)
        if isinstance(v, dict) and any(re.fullmatch(r"c\d+", str(k)) for k in v.keys()):
            return v

    # Fallback: clusters at top-level alongside "meta"/"tree"
    cluster_like = {
        k: v
        for k, v in data.items()
        if isinstance(k, str) and re.fullmatch(r"c\d+", k) and isinstance(v, dict)
    }
    if cluster_like:
        return cluster_like

    raise ValueError(
        "Could not locate a cluster mapping. Expected a dict keyed by 'c###' "
        "either at top-level or under a key like 'clusters' / 'items_by_cluster'."
    )


def build_index(clusters: Dict[str, dict]) -> Dict[str, dict]:
    """
    Build lookup indexes:
      exact[(leaf, ctx1, ctx2)] -> set(cluster_ids)
      leaf_ctx2[(leaf, ctx2)]   -> set(cluster_ids)
      leaf_ctx1[(leaf, ctx1)]   -> set(cluster_ids)
      leaf_only[leaf]           -> set(cluster_ids)
    All keys are normalized via _norm().
    """
    exact: Dict[Tuple[str, str, str], Set[str]] = defaultdict(set)
    leaf_ctx2: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    leaf_ctx1: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    leaf_only: Dict[str, Set[str]] = defaultdict(set)

    for cid, payload in clusters.items():
        items = payload.get("items")
        if not isinstance(items, list):
            continue

        for raw_item in items:
            if not isinstance(raw_item, str):
                continue
            parsed = parse_cluster_item(raw_item)
            if not parsed:
                continue

            leaf, ctx1, ctx2 = parsed
            k_exact = (_norm(leaf), _norm(ctx1), _norm(ctx2))
            exact[k_exact].add(cid)
            leaf_ctx2[(_norm(leaf), _norm(ctx2))].add(cid)
            leaf_ctx1[(_norm(leaf), _norm(ctx1))].add(cid)
            leaf_only[_norm(leaf)].add(cid)

    return {
        "exact": exact,
        "leaf_ctx2": leaf_ctx2,
        "leaf_ctx1": leaf_ctx1,
        "leaf_only": leaf_only,
    }


def infer_cluster_id_for_path(
    path_str: str, idx: Dict[str, dict]
) -> Union[str, List[str], None]:
    """
    Infer cluster ID for a single leaf path.

    Priority:
      1) Exact match on (leaf, ctx1, ctx2)
      2) Match on (leaf, ctx2) if uniquely identifies one cluster
      3) Match on (leaf, ctx1) if unique
      4) Match on leaf-only if unique
      else None (or list of candidates if ambiguous)

    Returns:
      - cluster id string (e.g., "c586") if uniquely inferred
      - list[str] if ambiguous candidates remain
      - None if not found
    """
    leaf, ctx1, ctx2 = parse_leaf_path(path_str)
    k_exact = (_norm(leaf), _norm(ctx1), _norm(ctx2))

    exact_hits: Set[str] = idx["exact"].get(k_exact, set())
    if len(exact_hits) == 1:
        return next(iter(exact_hits))
    if len(exact_hits) > 1:
        return sorted(exact_hits)

    hits_ctx2: Set[str] = idx["leaf_ctx2"].get((_norm(leaf), _norm(ctx2)), set())
    if len(hits_ctx2) == 1:
        return next(iter(hits_ctx2))
    if len(hits_ctx2) > 1:
        return sorted(hits_ctx2)

    hits_ctx1: Set[str] = idx["leaf_ctx1"].get((_norm(leaf), _norm(ctx1)), set())
    if len(hits_ctx1) == 1:
        return next(iter(hits_ctx1))
    if len(hits_ctx1) > 1:
        return sorted(hits_ctx1)

    hits_leaf: Set[str] = idx["leaf_only"].get(_norm(leaf), set())
    if len(hits_leaf) == 1:
        return next(iter(hits_leaf))
    if len(hits_leaf) > 1:
        return sorted(hits_leaf)

    return None


def infer_cluster_ids(
    json_path: Union[str, Path], leaf_paths: Iterable[str]
) -> Dict[str, Union[str, List[str], None]]:
    """Load clustering JSON, build indexes, and infer cluster IDs for each leaf path."""
    json_path = Path(json_path).expanduser()

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    clusters = _find_cluster_container(data)
    idx = build_index(clusters)

    out: Dict[str, Union[str, List[str], None]] = {}
    for p in leaf_paths:
        p = p.strip()
        if not p:
            continue
        out[p] = infer_cluster_id_for_path(p, idx)
    return out


def _parse_semicolon_list(s: str) -> List[str]:
    return [x.strip() for x in (s or "").split(";") if x.strip()]


if __name__ == "__main__":
    default_json = "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/cluster_criterions/results/04_semantically_clustered_items.json"

    parser = argparse.ArgumentParser(
        description="Infer cluster IDs for ontology leaf paths using leaf + two context nodes."
    )
    parser.add_argument(
        "--json",
        type=str,
        default=default_json,
        help="Path to 04_semantically_clustered_items.json",
    )
    parser.add_argument(
        "--paths",
        type=str,
        default="",
        help="Semicolon-separated leaf paths. If omitted, uses the built-in test list.",
    )
    args = parser.parse_args()

    # Built-in test list (from your message). This is used if --paths is not provided.
    test_paths_raw = (
        "sleep-wake disorders / circadian rhythm sleep–wake disorders / delayed sleep–wake phase disorder / *subjective_nonrestorative_sleep ; "
        "sleep-wake disorders / insomnia disorders / short-term (acute) insomnia disorder / *daytime_fatigue ; "
        "cognitive systems / attention / behavioral indicators / sustained attention lapses / units_of_analysis / self_report / *sustained_attention_lapses_frequency_self_report ; "
        "negative valence systems / loss / cognitive-affective indicators / guilt and self-blame focus / units_of_analysis / self_report / *guilt_selfblame_scale_score ; "
        "anxiety disorders / generalized anxiety disorder / with cognitive rumination / *self_referential_ruminative_content"
    )

    leaf_paths = _parse_semicolon_list(args.paths) if args.paths.strip() else _parse_semicolon_list(test_paths_raw)

    result = infer_cluster_ids(args.json, leaf_paths)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=False))

# NOTE: RDoC was not used during semantical clustering --> so will result in unmapped/null values for some of the leaf nodes...
