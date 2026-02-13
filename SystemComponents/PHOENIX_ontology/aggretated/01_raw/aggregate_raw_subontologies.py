#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Integrate many JSON ontologies into one BIG hierarchical JSON ontology.

Key requirements implemented:
- Root: ONTOLOGY
- Primary nodes: CRITERION, PREDICTOR, PERSON, CONTEXT, HAPA
- CRITERION subdivides first into CLINICAL (3) and NON-CLINICAL (2) with specified names.
- For the 3 CLINICAL ontologies (DSM-5TR / ICD-10 / RDoC-701):
    - Remove any "estimated prevalence" field (robust matching: spaces/underscores/case).
    - If a disorder node contains a "criteria" field (robust matching), DO NOT keep an explicit "criteria" node.
      Instead, "de-list" criteria (list -> dict leaf nodes) and merge directly under the disorder node.
    - Also: any leaf list (even without an explicit "criteria" label) is "de-listed" into leaf nodes.
- PREDICTOR:
    - Load a SINGLE already-aggregated predictor JSON from PREDICTOR_PATH.
    - That predictor JSON is assumed to NOT have "PREDICTOR" as its primary root node (but we safely unwrap if it does).
    - The predictor JSON is expected to contain BIO / PSYCHO / SOCIAL at its top level (robust matching);
      we standardize them to exact LABEL_BIO/LABEL_PSYCHO/LABEL_SOCIAL under the PREDICTOR primary node.
- PERSON / CONTEXT / HAPA:
    - Files already include their own primary key in-file; paste under ONTOLOGY (unwrap if wrapped).

Run:
  python integrate_phoenix_ontology.py

Output:
  Writes a single aggregated JSON to:
    /Users/stijnvanseveren/PythonProjects/MASTERPROEF/SystemComponents/PHOENIX_ontology/aggretated/01_raw/ontology_aggregated.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict


# =========================
# Paths (as you specified)
# =========================

OUTPUT_DIR_DEFAULT = Path(
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/SystemComponents/PHOENIX_ontology/aggretated/01_raw"
)

# CRITERION -> CLINICAL (special handling: skip prevalence, flatten criteria, de-list lists)
DSM_PATH = Path(
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/SystemComponents/PHOENIX_ontology/separate/01_raw/CRITERION/steps/01_raw/separate/clinical/02_post_generation/DSM_pg/DSM5TR_criteria_merged.json"
)
ICD_PATH = Path(
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/SystemComponents/PHOENIX_ontology/separate/01_raw/CRITERION/steps/01_raw/separate/clinical/02_post_generation/ICD_pg/ICD_criteria_merged.json"
)
RDOC_PATH = Path(
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/SystemComponents/PHOENIX_ontology/separate/01_raw/CRITERION/steps/01_raw/separate/clinical/02_post_generation/RDoC_pg/RDoC_criteria_merged.json"
)

# CRITERION -> NON-CLINICAL (simple paste)
AGE_GENERAL_PATH = Path(
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/SystemComponents/PHOENIX_ontology/separate/01_raw/CRITERION/steps/01_raw/separate/non_clinical/age_general/aggregated/general_nonclinical.json"
)
AGE_SPECIFIC_PATH = Path(
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/SystemComponents/PHOENIX_ontology/separate/01_raw/CRITERION/steps/01_raw/separate/non_clinical/age_specific/aggregated/idiosyncratic_nonclinical.json"
)

# PREDICTOR (ALREADY aggregated into BIO/PSYCHO/SOCIAL, WITHOUT PREDICTOR as primary node)
PREDICTOR_PATH = Path(
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/SystemComponents/PHOENIX_ontology/separate/01_raw/PREDICTOR/steps/01_raw/aggregated/PREDICTOR_ontology.json"
)

# PERSON / CONTEXT / HAPA (already include their own primary key in-file; paste under ONTOLOGY)
PERSON_PATH = Path(
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/SystemComponents/PHOENIX_ontology/separate/01_raw/PERSON/PERSON.json"
)
CONTEXT_PATH = Path(
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/SystemComponents/PHOENIX_ontology/separate/01_raw/CONTEXT/CONTEXT.json"
)
HAPA_PATH = Path(
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/SystemComponents/PHOENIX_ontology/separate/01_raw/HAPA/HAPA.json"
)


# =========================
# Naming (as you specified)
# =========================

LABEL_ONTOLOGY = "ONTOLOGY"

LABEL_CRITERION = "CRITERION"
LABEL_PREDICTOR = "PREDICTOR"
LABEL_PERSON = "PERSON"
LABEL_CONTEXT = "CONTEXT"
LABEL_HAPA = "HAPA"

LABEL_CLINICAL = "CLINICAL"
LABEL_NONCLINICAL = "NON-CLINICAL"

# Names specified after "(this is ...)"
LABEL_DSM = "DSM-5TR"
LABEL_ICD = "ICD-10"
LABEL_RDOC = "RDoC-701"
LABEL_AGE_GENERAL = "AGE-GENERAL"
LABEL_AGE_SPECIFIC = "AGE-SPECIFIC"

LABEL_BIO = "BIO"
LABEL_PSYCHO = "PSYCHO"
LABEL_SOCIAL = "SOCIAL"


# =========================
# Normalization controls
# =========================

# Your rule: BIG NAMES in CAPITAL; sub-leaf names in small letters.
# Default: normalize all non-protected keys to lowercase.
NORMALIZE_NON_BIG_KEYS_TO_LOWER_DEFAULT = True

PROTECTED_KEYS = {
    LABEL_ONTOLOGY,
    LABEL_CRITERION,
    LABEL_PREDICTOR,
    LABEL_PERSON,
    LABEL_CONTEXT,
    LABEL_HAPA,
    LABEL_CLINICAL,
    LABEL_NONCLINICAL,
    LABEL_DSM,
    LABEL_ICD,
    LABEL_RDOC,
    LABEL_AGE_GENERAL,
    LABEL_AGE_SPECIFIC,
    LABEL_BIO,
    LABEL_PSYCHO,
    LABEL_SOCIAL,
}


# =========================
# Helpers
# =========================

def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def canon_key(s: str) -> str:
    """Canonical key for robust matching: remove non-alphanum, lowercase."""
    return re.sub(r"[^A-Za-z0-9]+", "", s).lower()


def unwrap_single_root(d: Any, expected_root_key: str | None = None) -> Any:
    """
    If a JSON file is shaped like { "<LABEL>": {...} } and expected_root_key matches,
    return inner {...}. Otherwise return as-is.
    """
    if isinstance(d, dict) and len(d) == 1:
        (k, v), = d.items()
        if expected_root_key is None:
            return v
        if k == expected_root_key:
            return v
        if canon_key(k) == canon_key(expected_root_key):
            return v
    return d


def ensure_unique_key(existing: set[str], candidate: str) -> str:
    if candidate not in existing:
        existing.add(candidate)
        return candidate
    i = 2
    while True:
        k = f"{candidate}__{i}"
        if k not in existing:
            existing.add(k)
            return k
        i += 1


def delist_list_to_leafdict(lst: list[Any]) -> Dict[str, dict]:
    """
    Convert a list of criteria items into dict leaf-nodes: { "<criterion>": {} }.
    Keys are lowercased, duplicates disambiguated.
    """
    out: Dict[str, dict] = {}
    used: set[str] = set()
    for item in lst:
        if item is None:
            key = "null"
        elif isinstance(item, str):
            key = item.strip().lower()
        else:
            key = json.dumps(item, ensure_ascii=False).strip().lower()

        if not key:
            key = "empty"

        key = ensure_unique_key(used, key)
        out[key] = {}
    return out


def is_estimated_prevalence_key(k: str) -> bool:
    """
    Robust matcher for fields like:
      - estimated_prevalence_percent
      - estimated prevalence percent
      - estimatedPrevalencePercent
      - prevalence_percent (if ever present)
    We intentionally do NOT remove other numeric fields unless they match this pattern.
    """
    ck = canon_key(k)
    if "prevalence" not in ck:
        return False
    # must contain "estimated" OR contain "percent"/"percentage"/"rate"
    return ("estimated" in ck) or ("percent" in ck) or ("percentage" in ck) or ("rate" in ck)


def is_criteria_like_key(k: str) -> bool:
    """
    Robust matcher for 'criteria' keys in many formats:
      - criteria / Criteria / CRITERIA
      - criteria_list / criteriaList / diagnostic_criteria / etc.
    """
    ck = canon_key(k)
    return ck == "criteria" or ck.startswith("criteria") or ck.endswith("criteria") or ("criteria" in ck)


def transform_clinical(obj: Any) -> Any:
    """
    Clinical-specific transformation:
    - Remove "estimated prevalence" fields (robust matching).
    - Flatten any 'criteria' node (robust matching): do NOT keep explicit 'criteria' key.
      Merge de-listed criteria directly under the disorder node.
    - De-list ANY leaf list into leaf nodes.
    - Recurse through dicts.
    """
    # Lists become leaf dicts
    if isinstance(obj, list):
        return delist_list_to_leafdict(obj)

    # Scalars unchanged
    if not isinstance(obj, dict):
        return obj

    # Recurse and transform children first
    transformed: Dict[str, Any] = {}
    for k, v in obj.items():
        transformed[k] = transform_clinical(v)

    # Drop estimated prevalence fields
    for k in list(transformed.keys()):
        if is_estimated_prevalence_key(k):
            transformed.pop(k, None)

    # Flatten criteria-like keys (robust)
    criteria_keys = [k for k in list(transformed.keys()) if is_criteria_like_key(k)]
    for ck in criteria_keys:
        criteria_val = transformed.pop(ck, None)
        if criteria_val is None:
            continue

        # Ensure criteria becomes dict leaf nodes
        if isinstance(criteria_val, list):
            criteria_dict = delist_list_to_leafdict(criteria_val)
        elif isinstance(criteria_val, dict):
            criteria_dict = criteria_val
        else:
            # Unexpected scalar, ignore
            continue

        # Merge into disorder node without overwriting existing keys
        used = set(transformed.keys())
        for crit_k, crit_v in criteria_dict.items():
            mk = ensure_unique_key(used, crit_k)
            transformed[mk] = crit_v

    return transformed


def normalize_keys(obj: Any, protected: set[str]) -> Any:
    """
    Lowercase dict keys unless protected. Also avoids collisions created by lowercasing.
    """
    if isinstance(obj, list):
        return [normalize_keys(x, protected) for x in obj]
    if not isinstance(obj, dict):
        return obj

    new: Dict[str, Any] = {}
    used: set[str] = set()
    for k, v in obj.items():
        nk = k if (k in protected) else k.lower()
        nk = ensure_unique_key(used, nk)
        new[nk] = normalize_keys(v, protected)
    return new


def extract_predictor_parts(predictor_obj: Any) -> Dict[str, Any]:
    """
    Expect an already-aggregated predictor object whose top-level contains BIO/PSYCHO/SOCIAL
    (robust matching). Standardize those keys to LABEL_BIO/LABEL_PSYCHO/LABEL_SOCIAL.

    We also safely handle the case where the predictor file *does* include { "PREDICTOR": {...} }.
    """
    predictor_obj = unwrap_single_root(predictor_obj, expected_root_key=LABEL_PREDICTOR)

    if not isinstance(predictor_obj, dict):
        raise ValueError(
            f"PREDICTOR JSON must be an object/dict at root (after optional unwrapping). "
            f"Got: {type(predictor_obj).__name__}"
        )

    # Find BIO/PSYCHO/SOCIAL keys robustly
    canon_to_actual: Dict[str, str] = {canon_key(k): k for k in predictor_obj.keys()}

    def get_by_label(label: str) -> Any:
        # Prefer exact match, then canonical match, then search by substring
        if label in predictor_obj:
            return predictor_obj[label]
        c = canon_key(label)
        if c in canon_to_actual:
            return predictor_obj[canon_to_actual[c]]
        # fallback: find any key containing label canon (e.g., "bio_factors")
        for k in predictor_obj.keys():
            if canon_key(label) in canon_key(k):
                return predictor_obj[k]
        raise KeyError(label)

    missing: list[str] = []
    out: Dict[str, Any] = {}

    for label in (LABEL_BIO, LABEL_PSYCHO, LABEL_SOCIAL):
        try:
            out[label] = get_by_label(label)
        except KeyError:
            missing.append(label)

    if missing:
        raise ValueError(
            "PREDICTOR JSON is expected to contain top-level BIO/PSYCHO/SOCIAL (robust matching). "
            f"Missing: {missing}. Top-level keys found: {list(predictor_obj.keys())}"
        )

    return out


# =========================
# Build ontology
# =========================

def build_ontology(normalize_non_big_keys_to_lower: bool) -> Dict[str, Any]:
    # --- Load clinical (special transform) ---
    dsm_raw = read_json(DSM_PATH)
    icd_raw = read_json(ICD_PATH)
    rdoc_raw = read_json(RDOC_PATH)

    dsm = unwrap_single_root(dsm_raw, expected_root_key=LABEL_DSM)
    icd = unwrap_single_root(icd_raw, expected_root_key=LABEL_ICD)
    rdoc = unwrap_single_root(rdoc_raw, expected_root_key=LABEL_RDOC)

    dsm = transform_clinical(dsm)
    icd = transform_clinical(icd)
    rdoc = transform_clinical(rdoc)

    # --- Load non-clinical (simple paste) ---
    age_general_raw = read_json(AGE_GENERAL_PATH)
    age_specific_raw = read_json(AGE_SPECIFIC_PATH)

    age_general = unwrap_single_root(age_general_raw, expected_root_key=LABEL_AGE_GENERAL)
    age_specific = unwrap_single_root(age_specific_raw, expected_root_key=LABEL_AGE_SPECIFIC)

    # --- Load PREDICTOR (already aggregated, WITHOUT PREDICTOR primary key) ---
    predictor_raw = read_json(PREDICTOR_PATH)
    predictor = extract_predictor_parts(predictor_raw)

    # --- Load PERSON / CONTEXT / HAPA (unwrap if file is { "PERSON": {...} } etc.) ---
    person_raw = read_json(PERSON_PATH)
    context_raw = read_json(CONTEXT_PATH)
    hapa_raw = read_json(HAPA_PATH)

    person = unwrap_single_root(person_raw, expected_root_key=LABEL_PERSON)
    context = unwrap_single_root(context_raw, expected_root_key=LABEL_CONTEXT)
    hapa = unwrap_single_root(hapa_raw, expected_root_key=LABEL_HAPA)

    big: Dict[str, Any] = {
        LABEL_ONTOLOGY: {
            LABEL_CRITERION: {
                LABEL_CLINICAL: {
                    LABEL_DSM: dsm,
                    LABEL_ICD: icd,
                    LABEL_RDOC: rdoc,
                },
                LABEL_NONCLINICAL: {
                    LABEL_AGE_GENERAL: age_general,
                    LABEL_AGE_SPECIFIC: age_specific,
                },
            },
            # Wrap predictor under PREDICTOR primary node (but do NOT require it in-file)
            LABEL_PREDICTOR: predictor,
            LABEL_PERSON: person,
            LABEL_CONTEXT: context,
            LABEL_HAPA: hapa,
        }
    }

    if normalize_non_big_keys_to_lower:
        big = normalize_keys(big, protected=PROTECTED_KEYS)

    return big


def write_json(data: Any, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# =========================
# Main
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate PHOENIX ontology JSONs into one file.")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=OUTPUT_DIR_DEFAULT,
        help=f"Output directory (default: {OUTPUT_DIR_DEFAULT})",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default="ontology_aggregated.json",
        help="Output filename (default: ontology_aggregated.json)",
    )
    parser.add_argument(
        "--preserve-case",
        action="store_true",
        help="Do NOT lowercase non-BIG keys (overrides the default normalization).",
    )
    args = parser.parse_args()

    normalize_non_big = not args.preserve_case and NORMALIZE_NON_BIG_KEYS_TO_LOWER_DEFAULT

    ontology = build_ontology(normalize_non_big_keys_to_lower=normalize_non_big)
    out_path = args.outdir / args.outfile
    write_json(ontology, out_path)

    print(f"[OK] Wrote aggregated ontology to: {out_path}")

    # Optional: print a short preview
    preview = json.dumps(ontology, ensure_ascii=False, indent=2)
    print("\n[Preview - first 2000 chars]\n")
    print(preview[:2000])


if __name__ == "__main__":
    main()
