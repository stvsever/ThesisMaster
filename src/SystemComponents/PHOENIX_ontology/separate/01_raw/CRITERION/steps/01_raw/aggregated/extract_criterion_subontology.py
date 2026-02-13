#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract CRITERION primary node from the aggregated PHOENIX ontology.

Handles common layouts:
1) {"ONTOLOGY": {"CRITERION": {...}}}
2) {"ONTOLOGY": {"CRITION": {...}}}  (typo)
3) {"CRITERION": {...}}              (flat)
4) {"CRITION": {...}}                (flat typo)

Temporary exclusions (requested):
- Skip the full RDoC node(s): any dict key equal to "RDoC" or starting with "RDoC-"
  (e.g., "RDoC-701": {...} is removed entirely).
- Remove any "units_of" key anywhere in the subtree.
"""

import json
from pathlib import Path


# =========================
# Paths (as specified)
# =========================

INPUT_PATH = Path(
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/SystemComponents/PHOENIX_ontology/aggretated/01_raw/ontology_aggregated.json"
)

OUTPUT_DIR = Path(
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/SystemComponents/PHOENIX_ontology/separate/01_raw/CRITERION/steps/01_raw/aggregated"
)

OUTPUT_PATH = OUTPUT_DIR / "CRITERION_ontology.json"


def _get_criterion_node(root: dict) -> dict:
    """Return the CRITERION dict from any supported layout, else raise KeyError."""
    # Case A: nested under ONTOLOGY
    if isinstance(root.get("ONTOLOGY"), dict):
        onto = root["ONTOLOGY"]
        if "CRITERION" in onto:
            return onto["CRITERION"]
        if "CRITION" in onto:
            return onto["CRITION"]

    # Case B: flat
    if "CRITERION" in root:
        return root["CRITERION"]
    if "CRITION" in root:
        return root["CRITION"]

    raise KeyError(
        "CRITERION primary node not found (checked ONTOLOGY/CRITERION and top-level keys)"
    )


def _should_skip_key(key: object) -> bool:
    """Keys to skip anywhere in the extracted subtree."""
    if not isinstance(key, str):
        return False
    if key == "units_of":
        return True
    if key == "RDoC" or key.startswith("RDoC-"):
        return True
    return False


def _prune_subtree(obj):
    """
    Recursively remove:
      - any key == 'units_of'
      - any key == 'RDoC' or starting with 'RDoC-'
    from dicts, and process lists deeply.
    """
    if isinstance(obj, dict):
        pruned = {}
        for k, v in obj.items():
            if _should_skip_key(k):
                continue
            pruned[k] = _prune_subtree(v)
        return pruned
    if isinstance(obj, list):
        return [_prune_subtree(x) for x in obj]
    return obj


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise TypeError("Aggregated ontology root must be a JSON object (dict).")

    criterion = _get_criterion_node(data)

    if not isinstance(criterion, dict):
        raise TypeError("CRITERION primary node must be a JSON object (dict).")

    # Apply requested temporary pruning
    criterion = _prune_subtree(criterion)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump({"CRITERION": criterion}, f, ensure_ascii=False, indent=2)

    print(f"[OK] Extracted CRITERION ontology to:\n{OUTPUT_PATH}")
    print(
        "[NOTE] Temporary pruning applied: skipped any 'RDoC'/'RDoC-*' nodes and removed all 'units_of' keys."
    )


if __name__ == "__main__":
    main()
