#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
print_levelN_predictor_hierarchy.py

- Loads PREDICTOR_ontology.json
- Extracts ALL predictor nodes at TARGET_LEVEL (root keys are level 1)
- Writes a .txt file with a hierarchical outline to preserve structure and reduce token space
- Avoids repeating the leaf name in the path display:
    LeafName (path: 'Ancestor1 > Ancestor2 > ...')
  (leaf appears only once: on the left)

- Adds global monotonically increasing IDs (0 -> last) across all leaves
- Reports metadata (counts per level visited, totals, etc.)
"""

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple


# -----------------------
# PATHS
# -----------------------

INPUT_JSON = (
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/SystemComponents/"
    "PHOENIX_ontology/separate/01_raw/PREDICTOR/steps/01_raw/aggregated/"
    "PREDICTOR_ontology.json"
)

OUTPUT_TXT = (
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/ontology_mappings/CRITERION/predictor_to_criterion/input_lists/predictors_list.txt"
)

TARGET_LEVEL = 3  # root keys are level 1 ; aim for 4 during actual runs
#NOTE: due to SOCIAL sub-ontology with combinatorial participation nodes --> large divergence from target level '6'


# -----------------------
# HELPERS
# -----------------------

def iter_level_paths(
    ontology: Dict[str, Any],
    target_level: int
) -> Tuple[List[List[str]], Dict[int, int]]:
    """
    Iterative DFS traversal to collect all paths that end at `target_level`.

    Returns:
      - collected_paths: list of paths (each path is a list[str]) at exactly target_level
      - level_counts: dict mapping level -> number of nodes encountered at that level
    """
    collected_paths: List[List[str]] = []
    level_counts: Dict[int, int] = defaultdict(int)

    stack: List[Tuple[Any, List[str], int]] = []

    # Seed stack with root keys at level 1
    for root_key, root_child in ontology.items():
        level_counts[1] += 1
        stack.append((root_child, [root_key], 1))

    while stack:
        node, path, level = stack.pop()

        if level == target_level:
            collected_paths.append(path)
            continue

        if not isinstance(node, dict):
            continue

        next_level = level + 1
        for key, child in node.items():
            level_counts[next_level] += 1
            stack.append((child, path + [key], next_level))

    return collected_paths, level_counts


def build_prefix_to_leaves(
    paths: List[List[str]],
    target_level: int
) -> Dict[Tuple[str, ...], List[str]]:
    """
    For paths of length == target_level:
      prefix = tuple(path[:-1])  # ancestors only
      leaf   = path[-1]

    Returns:
      mapping[prefix] = sorted unique leaves

    Example for target_level=4:
      prefix = (L1, L2, L3)
      leaves = [L4a, L4b, ...]
    """
    mapping: Dict[Tuple[str, ...], List[str]] = defaultdict(list)

    for p in paths:
        if len(p) != target_level:
            continue
        prefix = tuple(p[:-1])
        leaf = p[-1]
        mapping[prefix].append(leaf)

    for prefix in list(mapping.keys()):
        mapping[prefix] = sorted(set(mapping[prefix]))

    return mapping


def root_sort_key(root: str) -> Tuple[int, str]:
    """
    Prefer BIO, PSYCHO, SOCIAL if present; otherwise alphabetical.
    """
    preferred = {"BIO": 0, "PSYCHO": 1, "SOCIAL": 2}
    return (preferred.get(root, 999), root)


# -----------------------
# MAIN
# -----------------------

def main() -> None:
    if TARGET_LEVEL < 2:
        raise ValueError("TARGET_LEVEL must be >= 2 (root is level 1).")

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        ontology = json.load(f)

    level_paths, level_counts = iter_level_paths(ontology, TARGET_LEVEL)

    # Group leaves by their ancestor prefix
    prefix_to_leaves = build_prefix_to_leaves(level_paths, TARGET_LEVEL)

    # -------- Write TXT (hierarchical) --------
    os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        #f.write(f"PHOENIX Predictor Nodes (Level {TARGET_LEVEL})\n")
        #f.write(f"Hierarchical format: Level1 -> ... -> Level{TARGET_LEVEL}\n")
        #f.write("Leaf rows avoid repetition: LEAF (path: 'ancestors only')\n\n")

        # Global monotonically increasing ID (0 -> last)
        global_id = 0

        # Build a hierarchy by iterating prefixes in sorted order, but printing grouped.
        # We print:
        #   [L1]
        #     └─ L2
        #        └─ L3
        #           └─ Leaf (ID:n) (path: 'L1 > L2 > L3')
        #
        # This naturally supports any TARGET_LEVEL. Indentation depth depends on prefix length.
        # For TARGET_LEVEL=4, prefix length is 3 -> prints L1, L2, L3 headings then leaves.

        # Collect all distinct roots (level1)
        roots = sorted({prefix[0] for prefix in prefix_to_leaves.keys()}, key=root_sort_key)

        for l1 in roots:
            f.write(f"[{l1}]\n")

            # Get all prefixes under this root
            prefixes_under_root = sorted([p for p in prefix_to_leaves.keys() if p[0] == l1])

            # We want to print intermediate headings without repeating them excessively.
            # Track last seen prefix at each depth.
            last_at_depth: Dict[int, str] = {}

            for prefix in prefixes_under_root:
                # prefix is (L1, L2, ..., L_{TARGET_LEVEL-1})
                # iterate through levels 2..TARGET_LEVEL-1 to print headings when they change
                for depth_idx in range(1, len(prefix)):  # index into prefix
                    level_num = depth_idx + 1  # because prefix[0] is level1
                    label = prefix[depth_idx]

                    if last_at_depth.get(level_num) != label:
                        # when a higher-level label changes, clear deeper cached labels
                        for k in list(last_at_depth.keys()):
                            if k >= level_num:
                                del last_at_depth[k]

                        indent = "  " * (level_num - 2)  # level2 => 0 indents, level3 => 1, ...
                        branch = "└─ "
                        f.write(f"{indent}{branch}{label}\n")
                        last_at_depth[level_num] = label

                # Now write leaves under this prefix
                leaves = prefix_to_leaves[prefix]
                ancestors_str = " > ".join(prefix)  # ancestors only (no leaf)

                leaf_indent = "  " * (len(prefix) - 1)  # aligns leaves under deepest heading
                for leaf in leaves:
                    f.write(f"{leaf_indent}└─ {leaf} (ID:{global_id})\n")
                    global_id += 1

            f.write("\n")

        total_leaves = global_id

        # -------- Metadata --------
        #f.write("--- METADATA ---\n")
        #f.write(f"Target level: {TARGET_LEVEL}\n")
        #f.write(f"Total nodes at level {TARGET_LEVEL}: {total_leaves}\n")
        #f.write(f"Max level encountered (with at least 1 node): {max(level_counts) if level_counts else 0}\n\n")
        #f.write("Counts per level encountered:\n")
        #for lvl in sorted(level_counts):
        #    f.write(f"level{lvl}: {level_counts[lvl]}\n")

    # -------- Console metadata --------
    #print("\n=== METADATA ===")
    #print(f"Target level: {TARGET_LEVEL}")
    #print(f"Nodes at level {TARGET_LEVEL}: {total_leaves}")
    #if level_counts:
    #    print(f"Max level encountered: {max(level_counts)}")
    #print(f"\n[ok] wrote file → {OUTPUT_TXT}")

    #if total_leaves == 0:
    #    print(
    #        "\n[warning] No nodes found at the requested level. "
    #        "Either the ontology does not reach that depth, or the root structure differs."
    #    )


if __name__ == "__main__":
    main()
