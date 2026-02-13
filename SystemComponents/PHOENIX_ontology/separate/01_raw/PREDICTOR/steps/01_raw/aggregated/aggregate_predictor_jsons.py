#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_aggregate_predictor_jsons.py

Goal:
    Recursively load and hierarchically merge all JSON other found under:

        PREDICTOR/steps/01_raw/separate/

    into a single aggregated JSON written to:

        PREDICTOR/steps/01_raw/aggregated/

Assumptions:
    - All JSON other contain hierarchical (tree-like) dict structures
    - Leaf nodes are empty dicts {}
    - Keys do not conflict semantically; identical keys are recursively merged

Behavior:
    - Recursively walks the input directory
    - Loads all *.json other
    - Deep-merges dictionaries (no overwriting unless necessary)
    - Writes a single aggregated JSON file

No schema inference. No validation. No normalization.
Pure structural aggregation.

"""

from __future__ import annotations

import os
import json
from typing import Dict, Any


# --------------------------
# CONFIG
# --------------------------

INPUT_DIR = (
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/SystemComponents/PHOENIX_ontology/separate/01_raw/PREDICTOR/steps/01_raw/separate"
)

OUTPUT_DIR = (
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/SystemComponents/PHOENIX_ontology/separate/01_raw/PREDICTOR/steps/01_raw/aggregated"
)

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "PREDICTOR_ontology.json")


# --------------------------
# Helpers
# --------------------------

def deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge src into dst.
    - dict + dict => recurse
    - otherwise: src overwrites dst
    """
    for k, v in src.items():
        if k in dst and isinstance(dst[k], dict) and isinstance(v, dict):
            deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_json_files(root: str):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".json"):
                yield os.path.join(dirpath, fn)


# --------------------------
# Main aggregation logic
# --------------------------

def main() -> None:
    if not os.path.isdir(INPUT_DIR):
        raise RuntimeError(f"Input directory not found: {INPUT_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    aggregated: Dict[str, Any] = {}

    json_files = sorted(find_json_files(INPUT_DIR))
    if not json_files:
        raise RuntimeError(f"No JSON other found under: {INPUT_DIR}")

    print(f"[info] Found {len(json_files)} JSON other. Aggregating...")

    for path in json_files:
        print(f"[load] {path}")
        data = load_json(path)
        if not isinstance(data, dict):
            raise ValueError(f"Top-level JSON must be object: {path}")
        deep_merge(aggregated, data)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, ensure_ascii=False, indent=2)

    print(f"[done] Aggregated ontology written to:")
    print(f"       {OUTPUT_FILE}")


# --------------------------
# Entry point
# --------------------------

if __name__ == "__main__":
    main()

# TODO: make ensure not some predictor-domains dominate due to count (only if truly needed)
