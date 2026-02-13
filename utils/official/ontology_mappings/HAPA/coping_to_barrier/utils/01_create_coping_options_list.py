#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_create_context_list.py

- Loads HAPA.json
- Extracts ONLY the "COPING_STRATEGIES" sub-ontology
- Collects ALL nodes at TARGET_LEVEL, where:
    level 1 = "COPING_STRATEGIES"
    level 2 = primary nodes (e.g., Assessment_and_Formulation, Planning_and_Implementation, ...)
    level 3 = secondary nodes (e.g., Functional_Analysis, Action_Planning, ...)
    level 4+ = deeper layers / leaf-level coping options (depending on your taxonomy depth)

- Writes a .txt file with a hierarchical outline (structure-preserving; compact)
- Adds global monotonically increasing IDs (0 -> last) across all nodes at TARGET_LEVEL
- Lets you choose depth via CLI: --target_level N
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Iterable


# -----------------------
# PATHS (edit if needed)
# -----------------------

INPUT_JSON = (
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/SystemComponents/"
    "PHOENIX_ontology/separate/01_raw/HAPA/HAPA.json"
)

OUTPUT_TXT = (
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/"
    "ontology_mappings/HAPA/coping_to_barrier/input_lists/"
    "coping_options_list.txt"
)

# Default depth: adjust to your typical "coping option" leaf depth.
# If your COPING_STRATEGIES now has deeper layers, you may set this higher.
DEFAULT_TARGET_LEVEL = 3


# -----------------------
# HELPERS
# -----------------------

def _ordered_items(d: Dict[str, Any], sort_keys: bool) -> Iterable[Tuple[str, Any]]:
    """Iterate dict items in insertion order (default) or sorted order."""
    if not isinstance(d, dict):
        return []
    if sort_keys:
        for k in sorted(d.keys()):
            yield k, d[k]
    else:
        for k, v in d.items():
            yield k, v


def load_coping_subontology(hapa_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return the dict under "COPING_STRATEGIES", robust to a few common container shapes:
      - {"COPING_STRATEGIES": {...}, ...}
      - {"HAPA": {"COPING_STRATEGIES": {...}, ...}}
    """
    if (
        isinstance(hapa_json, dict)
        and "COPING_STRATEGIES" in hapa_json
        and isinstance(hapa_json["COPING_STRATEGIES"], dict)
    ):
        return hapa_json["COPING_STRATEGIES"]

    if (
        isinstance(hapa_json, dict)
        and "HAPA" in hapa_json
        and isinstance(hapa_json["HAPA"], dict)
        and "COPING_STRATEGIES" in hapa_json["HAPA"]
        and isinstance(hapa_json["HAPA"]["COPING_STRATEGIES"], dict)
    ):
        return hapa_json["HAPA"]["COPING_STRATEGIES"]

    raise KeyError(
        "Could not find a dict at key 'COPING_STRATEGIES' (or 'HAPA' -> 'COPING_STRATEGIES') in the input JSON."
    )


def iter_level_paths_from_named_root(
    root_name: str,
    root_child: Dict[str, Any],
    target_level: int,
    *,
    sort_keys: bool = False
) -> Tuple[List[List[str]], Dict[int, int]]:
    """
    Iterative DFS to collect all paths that end at `target_level`.
    Here, level 1 is the named root (root_name).

    Returns:
      - collected_paths: list of paths (each path is a list[str]) at exactly target_level
      - level_counts: dict mapping level -> number of nodes encountered at that level
    """
    collected_paths: List[List[str]] = []
    level_counts: Dict[int, int] = defaultdict(int)

    # Stack entries: (node, path, level)
    stack: List[Tuple[Any, List[str], int]] = []

    # Seed with named root at level 1
    level_counts[1] += 1
    stack.append((root_child, [root_name], 1))

    while stack:
        node, path, level = stack.pop()

        if level == target_level:
            collected_paths.append(path)
            continue

        if not isinstance(node, dict) or len(node) == 0:
            continue

        next_level = level + 1

        # Push children; reverse push to preserve iteration order for DFS stack pop
        items = list(_ordered_items(node, sort_keys=sort_keys))
        for key, child in reversed(items):
            level_counts[next_level] += 1
            stack.append((child, path + [key], next_level))

    return collected_paths, level_counts


def build_prefix_to_leaves(
    paths: List[List[str]],
    target_level: int,
    *,
    sort_leaves: bool = False
) -> Dict[Tuple[str, ...], List[str]]:
    """
    For paths of length == target_level:
      prefix = tuple(path[:-1])  # ancestors only
      leaf   = path[-1]

    Returns:
      mapping[prefix] = leaves (unique; insertion-preserving unless sort_leaves=True)
    """
    mapping: Dict[Tuple[str, ...], List[str]] = defaultdict(list)
    seen: Dict[Tuple[str, ...], set] = defaultdict(set)

    for p in paths:
        if len(p) != target_level:
            continue
        prefix = tuple(p[:-1])
        leaf = p[-1]
        if leaf not in seen[prefix]:
            mapping[prefix].append(leaf)
            seen[prefix].add(leaf)

    if sort_leaves:
        for prefix in list(mapping.keys()):
            mapping[prefix] = sorted(mapping[prefix])

    return mapping


def write_hierarchy_txt(
    output_txt: str,
    prefix_to_leaves: Dict[Tuple[str, ...], List[str]],
    *,
    sort_prefixes: bool = False
) -> int:
    """
    Writes a compact hierarchy:
      [COPING_STRATEGIES]
      └─ Assessment_and_Formulation
         └─ Functional_Analysis
            └─ Trigger_And_Context_Mapping (ID:0)

    Returns total leaf rows written (== total nodes at chosen target_level).
    """
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)

    all_prefixes = list(prefix_to_leaves.keys())
    if not all_prefixes:
        with open(output_txt, "w", encoding="utf-8") as f:
            pass
        return 0

    root_label = all_prefixes[0][0]  # expected "COPING_STRATEGIES"

    if sort_prefixes:
        prefixes_ordered = sorted(prefix_to_leaves.keys())
    else:
        prefixes_ordered = all_prefixes

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(f"[{root_label}]\n")

        global_id = 0
        last_at_depth: Dict[int, str] = {}

        for prefix in prefixes_ordered:
            # prefix includes root at index 0: (COPING_STRATEGIES, L2, ..., L_{target-1})
            for depth_idx in range(1, len(prefix)):  # skip root at index 0
                level_num = depth_idx + 1
                label = prefix[depth_idx]

                if last_at_depth.get(level_num) != label:
                    for k in list(last_at_depth.keys()):
                        if k >= level_num:
                            del last_at_depth[k]

                    indent = "  " * (level_num - 2)  # level2 => 0 indents
                    f.write(f"{indent}└─ {label}\n")
                    last_at_depth[level_num] = label

            leaves = prefix_to_leaves[prefix]
            leaf_indent = "  " * (len(prefix) - 1)

            for leaf in leaves:
                f.write(f"{leaf_indent}└─ {leaf} (ID:{global_id})\n")
                global_id += 1

        return global_id


# -----------------------
# MAIN
# -----------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a hierarchical .txt list of nodes from HAPA['COPING_STRATEGIES'] at a chosen depth."
    )
    parser.add_argument(
        "--target_level",
        type=int,
        default=DEFAULT_TARGET_LEVEL,
        help=(
            "Depth to extract (levels count from 'COPING_STRATEGIES' as level 1). "
            "Examples: 2=primary nodes, 3=secondary nodes, 4+=deeper coping option layers."
        ),
    )
    parser.add_argument(
        "--input_json",
        type=str,
        default=INPUT_JSON,
        help="Path to HAPA.json"
    )
    parser.add_argument(
        "--output_txt",
        type=str,
        default=OUTPUT_TXT,
        help="Path to write coping_options_list.txt"
    )
    parser.add_argument(
        "--sort_keys",
        action="store_true",
        help="Sort ontology keys alphabetically during traversal (default preserves JSON insertion order)."
    )
    parser.add_argument(
        "--sort_prefixes",
        action="store_true",
        help="Sort printed prefixes alphabetically (default preserves traversal-derived order)."
    )
    parser.add_argument(
        "--sort_leaves",
        action="store_true",
        help="Sort leaf rows alphabetically under each prefix (default preserves traversal-derived order)."
    )

    args = parser.parse_args()

    if args.target_level < 2:
        raise ValueError("target_level must be >= 2 (level 1 is the root 'COPING_STRATEGIES').")

    with open(args.input_json, "r", encoding="utf-8") as f:
        hapa = json.load(f)

    coping_sub = load_coping_subontology(hapa)

    level_paths, _level_counts = iter_level_paths_from_named_root(
        "COPING_STRATEGIES",
        coping_sub,
        args.target_level,
        sort_keys=args.sort_keys
    )

    prefix_to_leaves = build_prefix_to_leaves(
        level_paths,
        args.target_level,
        sort_leaves=args.sort_leaves
    )

    _total_rows = write_hierarchy_txt(
        args.output_txt,
        prefix_to_leaves,
        sort_prefixes=args.sort_prefixes
    )

    # Optional console output:
    # print(f"[ok] wrote {_total_rows} rows at level {args.target_level} -> {args.output_txt}")
    # if _total_rows == 0:
    #     print("[warning] No nodes found at the requested level. Ontology may not reach that depth.")


if __name__ == "__main__":
    main()
