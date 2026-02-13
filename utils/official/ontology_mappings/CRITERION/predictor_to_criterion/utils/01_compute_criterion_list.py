#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
print_all_leaf_criterion_nodes.py

- Loads CRITERION_ontology.json
- Traverses the full ontology tree (no depth cap)
- Extracts ALL leaf nodes (nodes whose value is an empty dict, or non-dict)
- Writes a .txt file where EACH ROW is ONE leaf node with its full path

Output format (one per line):
    LEAF_NODE (path: 'ROOT > ... > LEAF_NODE')

Notes:
- If a node's value is not a dict, it is treated as a leaf.
- If a node is an empty dict {}, it is treated as a leaf.
"""

import json
import os
from typing import Any, Dict, List, Tuple


# -----------------------
# PATHS
# -----------------------

INPUT_JSON = (
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/SystemComponents/"
    "PHOENIX_ontology/separate/01_raw/CRITERION/steps/01_raw/aggregated/"
    "CRITERION_ontology.json"
)

OUTPUT_TXT = (
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/"
    "ontology_mappings/CRITERION/predictor_to_criterion/other/input_lists/"
    "criterions_list.txt"
)


# -----------------------
# HELPERS
# -----------------------

def collect_leaf_paths(ontology: Dict[str, Any]) -> List[List[str]]:
    """
    Iterative DFS traversal to collect all leaf paths.

    A node is considered a leaf if:
      - its value is not a dict, OR
      - its value is an empty dict {}

    Returns:
      - list of paths (each path is a list[str]) from root to leaf key
    """
    leaf_paths: List[List[str]] = []
    stack: List[Tuple[Any, List[str]]] = []

    # Seed stack with root keys
    for root_key, root_child in ontology.items():
        stack.append((root_child, [root_key]))

    while stack:
        node, path = stack.pop()

        # Non-dict => leaf at current path
        if not isinstance(node, dict):
            leaf_paths.append(path)
            continue

        # Empty dict => leaf at current path
        if len(node) == 0:
            leaf_paths.append(path)
            continue

        # Otherwise descend
        for key, child in node.items():
            stack.append((child, path + [key]))

    return leaf_paths


# -----------------------
# MAIN
# -----------------------

def main() -> None:
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        ontology = json.load(f)

    leaf_paths = collect_leaf_paths(ontology)

    # Sort deterministically by full path string
    leaf_paths_sorted = sorted(leaf_paths, key=lambda p: " > ".join(p))

    os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        #f.write("PHOENIX Criterion Leaf Nodes (All Leaves)\n")
        #f.write("Format:    LEAF (path: 'ROOT > ... > LEAF')\n\n")

        for p in leaf_paths_sorted:
            leaf = p[-1]
            f.write(f"{leaf} (path: '{' > '.join(p)}')\n")

        #f.write("\n--- METADATA ---\n")
        #f.write(f"Total leaf nodes: {len(leaf_paths_sorted)}\n")

    print("\n=== METADATA ===")
    print(f"Total leaf nodes: {len(leaf_paths_sorted)}")
    print(f"\n[ok] wrote file → {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
