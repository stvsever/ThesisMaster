#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Read CRITERION_ontology.json and write a .txt file containing ALL leaf nodes,
one line per leaf, with the path reversed and normalized for semantic embedding.

CHANGE REQUESTED:
- Keep ONLY: leaf + parent + grandparent (3 last nodes of the cleaned path),
  then append " — criterion"
  i.e. output is always:
      leaf < parent < grandparent — criterion
  (if fewer than 3 nodes exist after cleanup, keep what exists)

Leaf definition used:
- A dict with no keys ({}), i.e., an explicit leaf marker in your ontology
- OR any non-dict / non-list value (scalar leaf); then we append " :: <value>" to the leaf

Normalization & removal rules:
- Everything is lowercased
- Remove the node "unspecified_stage" wherever it appears in the path
- Remove the following prefixes entirely:
  * CRITERION > NON-CLINICAL > AGE-SPECIFIC >
  * CRITERION > NON-CLINICAL > AGE-GENERAL >
  * CRITERION > CLINICAL > DSM-5TR >
  * CRITERION > CLINICAL > ICD-10 > <chapter> >
- Output ends with "— criterion" (not "< criterion")
"""

import json
from pathlib import Path
from typing import Any, List, Optional, Tuple


# =========================
# Paths (as specified)
# =========================

input_file = Path(
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/SystemComponents/PHOENIX_ontology/separate/01_raw/CRITERION/steps/01_raw/aggregated/CRITERION_ontology.json"
)

output_file = Path(
    "/utils/official/cluster_criterions/results/01_all_leaf_nodes_with_paths.txt"
)


UP_SEP = " < "
ROOT_SEP = " — "
VALUE_SEP = " :: "


def _walk(node: Any, path: List[str], out_items: List[Tuple[List[str], Optional[str]]]) -> None:
    """Depth-first traversal collecting full paths to leaves."""
    if isinstance(node, dict):
        if not node:
            out_items.append((path, None))
            return
        for k, v in node.items():
            _walk(v, path + [str(k)], out_items)
        return

    if isinstance(node, list):
        for i, item in enumerate(node):
            _walk(item, path + [f"[{i}]"], out_items)
        return

    scalar = "null" if node is None else str(node)
    out_items.append((path, scalar))


def _strip_prefix_and_cleanup(full_path: List[str]) -> Tuple[str, List[str]]:
    """
    Apply prefix stripping and node cleanup.
    Returns (anchor_root, cleaned_remainder_without_anchor).
    """
    if not full_path:
        return ("CRITERION", [])

    anchor = full_path[0]
    p = full_path

    def starts_with(prefix: List[str]) -> bool:
        return len(p) >= len(prefix) and p[: len(prefix)] == prefix

    # Prefix removals
    if starts_with([anchor, "NON-CLINICAL", "AGE-SPECIFIC"]):
        p = p[3:]
    elif starts_with([anchor, "NON-CLINICAL", "AGE-GENERAL"]):
        p = p[3:]
    elif starts_with([anchor, "CLINICAL", "DSM-5TR"]):
        p = p[3:]
    elif starts_with([anchor, "CLINICAL", "ICD-10"]):
        # remove ICD-10 + one node underneath if present (chapter)
        p = p[4:] if len(p) >= 4 else []
    else:
        # Default: remove anchor only
        p = p[1:]

    # Remove "unspecified_stage" wherever it appears
    p = [x for x in p if str(x).lower() != "unspecified_stage"]

    return anchor, p


def _keep_last_three_nodes(remainder: List[str]) -> List[str]:
    """
    Keep only the last 3 nodes from the cleaned forward path:
      [..., grandparent, parent, leaf]
    """
    if not remainder:
        return []
    return remainder[-3:]


def _format_reversed_line(anchor: str, remainder: List[str], scalar_value: Optional[str]) -> Optional[str]:
    """
    Build output:
      leaf < parent < grandparent — criterion
    with lowercasing.
    """
    if not remainder:
        return None

    trimmed = _keep_last_three_nodes(remainder)
    if not trimmed:
        return None

    reversed_parts = list(reversed(trimmed))  # leaf-first

    if scalar_value is not None:
        reversed_parts[0] = f"{reversed_parts[0]}{VALUE_SEP}{scalar_value}"

    line = f"{reversed_parts[0]} (context: {' < '.join(reversed_parts[1:])}){ROOT_SEP}{anchor}"
    return line.lower()


def main() -> None:
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with input_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise TypeError("Input JSON root must be an object (dict).")

    collected: List[Tuple[List[str], Optional[str]]] = []
    for top_key, top_val in data.items():
        _walk(top_val, [str(top_key)], collected)

    lines: List[str] = []
    for full_path, scalar_value in collected:
        anchor, remainder = _strip_prefix_and_cleanup(full_path)
        line = _format_reversed_line(anchor, remainder, scalar_value)
        if line:
            lines.append(line)

    # Deduplicate while preserving order
    seen = set()
    deduped: List[str] = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            deduped.append(line)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for line in deduped:
            f.write(line + "\n")

    print(f"[OK] Wrote {len(deduped)} normalized reversed leaf paths (leaf+parent+grandparent) to:\n{output_file}")


if __name__ == "__main__":
    main()

# TODO: improve the strings structure/syntax such that emphasis is put on the leaf node itself instead of context
