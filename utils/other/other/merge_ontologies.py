#!/usr/bin/env python3
"""
merge_ontologies.py

Merges all ontology JSON other from INPUT_DIR into a single JSON at OUTPUT_PATH.

- Keeps the full input structure (no layer is dropped).
- Dicts are merged recursively (dict-vs-dict merges; otherwise right side wins).
- Sorting is OPTIONAL and controlled in main via two vars:
    SORT_ENABLED: bool
        If True, sort keys within each hierarchical cluster (siblings only).
        If False, preserve merge order.
    USE_INITIAL_WORDS_FIRST_LAYER: bool
        Only applies if SORT_ENABLED is True.
        When True: top-level keys sorted A→Z by their 01_pre_generation word; deeper layers by full key A→Z.
        When False: all layers sorted by full key A→Z.

Additionally prints metadata about the merged ontology:
- File count
- Node counts (total, internal, leaves)
- Unique key count and duplicate keys with occurrence counts
- Max depth and depth histogram
- Average branching factor
- Leaves per top-level key
"""

import json
import re
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, List


# ========= EXPLICIT PATHS =========
INPUT_DIR = Path(
    "/SystemComponents/PHOENIX_ontology/separate/CRITERION/steps/01_raw/separate/non_clinical/age_specific/separate")
OUTPUT_PATH = Path(
    "/SystemComponents/PHOENIX_ontology/separate/CRITERION/steps/01_raw/separate/non_clinical/age_specific/aggregated/idiosyncratic_nonclinical.json")
# ==================================


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge src into dst.
    - If both values are dicts, merge recursively.
    - Otherwise, src overwrites dst.
    """
    for k, v in src.items():
        if k in dst and isinstance(dst[k], dict) and isinstance(v, dict):
            deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def iter_json_files(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*.json")):
        if p.is_file():
            yield p


# Camel/PascalCase tokenizer (e.g., "CulturalLineageTracing" -> ["Cultural","Lineage","Tracing"])
_CAMEL_RE = re.compile(r'[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+')


def initial_word(key: str) -> str:
    """
    Initial word for sorting:
      1) Replace any non-alphanumeric with a space and take the first token.
      2) If single token, split CamelCase/PascalCase and take the first token.
    Lowercased for case-insensitive comparison.
    """
    if not isinstance(key, str):
        key = str(key)
    s = key.strip()
    if not s:
        return ""

    s_norm = re.sub(r'[^0-9A-Za-z]+', ' ', s).strip()
    parts = s_norm.split()

    first = parts[0] if parts else s_norm
    if len(parts) == 1:
        tokens = _CAMEL_RE.findall(first)
        if tokens:
            first = tokens[0]

    return first.lower()


def _sorted_items_for_layer(
    d: Dict[str, Any],
    depth: int,
    use_initial_word_top: bool
) -> List[Tuple[str, Any]]:
    """
    Return [(key, value_sorted), ...] for dict d, sorted among siblings.
    Top layer (depth==0) uses 01_pre_generation-word key if use_initial_word_top=True,
    otherwise full-key. Deeper layers always sort by full-key.
    """
    def key_func(k: str) -> Tuple[str, str]:
        if depth == 0 and use_initial_word_top:
            return (initial_word(k), k.lower())
        return (k.lower(), k.lower())

    sortable = []
    for k, v in d.items():
        v_sorted = sort_structure(d=v, depth=depth+1, use_initial_word_top=use_initial_word_top) if isinstance(v, dict) else v
        sortable.append((key_func(k), k, v_sorted))

    sortable.sort(key=lambda t: t[0])
    return [(k, v) for _, k, v in sortable]


def sort_structure(d: Dict[str, Any], depth: int, use_initial_word_top: bool) -> Dict[str, Any]:
    """
    Recursively sort a dict's children within each parent only.
    """
    out: Dict[str, Any] = {}
    for k, v in _sorted_items_for_layer(d, depth, use_initial_word_top):
        out[k] = v
    return out


# ---------- Metadata helpers ----------

def is_leaf(value: Any) -> bool:
    """
    Leaf if:
    - value is a dict with no keys
    - or value is not a dict
    """
    if isinstance(value, dict):
        return len(value) == 0
    return True


def traverse_collect(
    node: Any,
    path: Tuple[str, ...],
    *,
    key_occurrences: Dict[str, List[Tuple[str, ...]]],
    depth_hist: Counter,
) -> Tuple[int, int, int, int, int]:
    """
    Traverse and collect metadata.
    Returns a tuple:
      total_nodes, internal_nodes, leaf_nodes, total_children_links, max_depth
    Also fills key_occurrences and depth_hist as side effects.
    """
    if not isinstance(node, dict):
        # Non-dict treated as a leaf node at current path
        depth = len(path)
        depth_hist[depth] += 1
        return 1, 0, 1, 0, depth

    # Count this node
    depth = len(path)
    depth_hist[depth] += 1

    if len(node) == 0:
        # Empty dict leaf
        return 1, 0, 1, 0, depth

    total_nodes = 1
    internal_nodes = 1
    leaf_nodes = 0
    total_children_links = len(node)
    max_depth = depth

    for k, v in node.items():
        key_occurrences[k].append(path + (k,))
        t_nodes, i_nodes, l_nodes, t_children, m_depth = traverse_collect(
            v, path + (k,), key_occurrences=key_occurrences, depth_hist=depth_hist
        )
        total_nodes += t_nodes
        internal_nodes += i_nodes
        leaf_nodes += l_nodes
        total_children_links += t_children
        if m_depth > max_depth:
            max_depth = m_depth

    return total_nodes, internal_nodes, leaf_nodes, total_children_links, max_depth


def leaves_per_top_key(tree: Dict[str, Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for k, v in tree.items():
        out[k] = count_leaves(v)
    return out


def count_leaves(node: Any) -> int:
    if is_leaf(node):
        return 1
    total = 0
    for _, v in node.items():  # type: ignore[union-attr]
        total += count_leaves(v)
    return total


def print_metadata(merged: Dict[str, Any], files_count: int) -> None:
    key_occurrences: Dict[str, List[Tuple[str, ...]]] = defaultdict(list)
    depth_hist: Counter = Counter()

    total_nodes, internal_nodes, leaf_nodes, total_children_links, max_depth = traverse_collect(
        merged, tuple(), key_occurrences=key_occurrences, depth_hist=depth_hist
    )

    unique_keys = len(key_occurrences)
    duplicate_keys = {k: v for k, v in key_occurrences.items() if len(v) > 1}
    avg_branching = (total_children_links / internal_nodes) if internal_nodes > 0 else 0.0

    # Leaves per top-level key
    leaves_top = leaves_per_top_key(merged)

    print("\n=== PHOENIX_ontology Merge Report ===")
    print(f"Files merged: {files_count}")
    print(f"Total nodes: {total_nodes}")
    print(f"Internal nodes: {internal_nodes}")
    print(f"Leaf nodes: {leaf_nodes}")
    print(f"Unique keys: {unique_keys}")
    print(f"Max depth: {max_depth}")
    print(f"Average branching factor: {avg_branching:.3f}")

    # Depth histogram
    print("\nDepth histogram (depth -> node count):")
    for d in sorted(depth_hist.keys()):
        print(f"  {d}: {depth_hist[d]}")

    # Leaves per top-level key
    print("\nLeaves per top-level key:")
    for k in sorted(leaves_top.keys(), key=str.lower):
        print(f"  {k}: {leaves_top[k]}")

    # Duplicate keys and sample paths
    if duplicate_keys:
        print("\nDuplicate key names across hierarchy (key -> occurrence count and sample paths):")
    #    # Show top 30 by occurrence count
    #    items = sorted(duplicate_keys.items(), key=lambda kv: len(kv[1]), reverse=True)[:30]
    #    for k, paths in items:
    #        print(f"  {k}: {len(paths)}")
    #        # Show up to 5 example paths
    #        for p in paths[:5]:
    #            print(f"    - /{'/'.join(p)}")
    #        if len(paths) > 5:
    #            print(f"    ... (+{len(paths) - 5} more)")
    else:
        print("\nNo duplicate key names found across the hierarchy.")

    print("=== End of Report ===\n")


# -------------- main --------------

def main() -> None:
    # ---------- Your toggles ----------
    SORT_ENABLED = False  # set True to enable sorting
    USE_INITIAL_WORDS_FIRST_LAYER = True  # only applies if SORT_ENABLED is True
    # ----------------------------------

    if not INPUT_DIR.exists() or not INPUT_DIR.is_dir():
        raise SystemExit(f"Input directory not found or not a directory: {INPUT_DIR}")

    files = [p for p in iter_json_files(INPUT_DIR)]
    if not files:
        raise SystemExit(f"No JSON other found under: {INPUT_DIR}")

    merged: Dict[str, Any] = {}

    for fp in files:
        try:
            data = load_json(fp)
        except Exception as e:
            raise SystemExit(f"Failed to parse JSON: {fp}\n{e}")

        if not isinstance(data, dict):
            raise SystemExit(f"Top-level JSON must be an object in {fp}")

        deep_merge(merged, data)

    if SORT_ENABLED:
        merged = sort_structure(merged, depth=0, use_initial_word_top=USE_INITIAL_WORDS_FIRST_LAYER)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"Merged {len(files)} other")
    if SORT_ENABLED:
        print(f"Sorted: yes (top layer by {'01_pre_generation word' if USE_INITIAL_WORDS_FIRST_LAYER else 'full key'}, deeper layers by full key)")
    else:
        print("Sorted: no")
    print(f"Output: {OUTPUT_PATH}")

    # Print metadata
    print_metadata(merged, files_count=len(files))


if __name__ == "__main__":
    main()
