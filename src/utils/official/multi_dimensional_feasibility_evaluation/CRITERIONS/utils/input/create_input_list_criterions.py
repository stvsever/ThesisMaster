#!/usr/bin/env python3
"""
Extract leaf-node paths from a nested CRITERION ontology JSON and write them to a .txt file.
Each line in the output file is a full path from the root to a leaf, delimited by " | ".

Leaf node definition:
- A leaf is a dict with no keys: {}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List


# --- Paths you provided ---
CRITERION_ONTOLOGY = Path(
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/SystemComponents/PHOENIX_ontology/"
    "separate/01_raw/CRITERION/steps/01_raw/aggregated/CRITERION_ontology.json"
)

OUTPUT_DIR = Path(
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/"
    "multi_dimensional_feasibility_evaluation/CRITERIONS/utils/input"
)

DELIM = " | "
OUTPUT_FILENAME = "CRITERION_leaf_paths.txt"


def is_leaf(node: Any) -> bool:
    return isinstance(node, dict) and len(node) == 0


def collect_leaf_paths(node: Any, prefix: List[str] | None = None) -> List[List[str]]:
    """
    Traverse a nested dict-of-dicts structure and return a list of leaf paths (as lists of keys).
    """
    if prefix is None:
        prefix = []

    # Leaf node: empty dict
    if is_leaf(node):
        return [prefix]

    # Non-leaf dict: recurse into children
    if isinstance(node, dict):
        paths: List[List[str]] = []
        for key, child in node.items():
            paths.extend(collect_leaf_paths(child, prefix + [str(key)]))
        return paths

    # Unexpected non-dict node: treat as terminal value (still count as a "leaf-like" endpoint)
    # If you want to strictly ignore these, return [] instead.
    return [prefix]


def main() -> None:
    if not CRITERION_ONTOLOGY.exists():
        raise FileNotFoundError(f"Ontology JSON not found: {CRITERION_ONTOLOGY}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / OUTPUT_FILENAME

    with CRITERION_ONTOLOGY.open("r", encoding="utf-8") as f:
        data = json.load(f)

    leaf_paths = collect_leaf_paths(data)

    # Convert to "A | B | C" strings
    lines = [DELIM.join(path) for path in leaf_paths if path]

    # Optional: stable ordering
    lines.sort()

    with out_path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    print(f"Wrote {len(lines)} leaf paths to: {out_path}")


if __name__ == "__main__":
    main()
