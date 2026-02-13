#!/usr/bin/env python3
"""
merge_ontologies.py

Merges all ontology JSON other from INPUT_DIR into a single JSON at OUTPUT_PATH.

What this version adds
1) Per-file schema normalization before merge:
   - If a file's top level looks like the four category keys
     {cognitive|social|emotional|physical}_development, it is wrapped under a
     detected life stage taken from the filename or parent folders. If no stage
     hint is found, it is wrapped under "UNSPECIFIED_STAGE".
   - Top-level life-stage keys are canonicalized to UPPER_SNAKE (e.g. early_childhood -> EARLY_CHILDHOOD).
   - Prevents category keys from ending up at the global root after merge.

2) Post-merge cleanup:
   - Any stray category keys that somehow still reached the root are moved
     under UNSPECIFIED_STAGE.

3) Schema validation report:
   - Confirms that the secondary layer under each stage contains only the four categories.
   - Lists any violations and missing categories per stage, without altering content.

Other behavior is unchanged:
- Recursive deep merge
- Optional sibling-only sorting
- Metadata summary
"""

import json
import re
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, List, Optional


# ========= EXPLICIT PATHS =========
INPUT_DIR = Path(
    "/SystemComponents/PHOENIX_ontology/separate/CRITERION/steps/01_raw/separate/non_clinical/age_specific/separate")
OUTPUT_PATH = Path(
    "/SystemComponents/PHOENIX_ontology/separate/CRITERION/steps/01_raw/separate/non_clinical/age_specific/aggregated/idiosyncratic_nonclinical.json")
# ==================================


# Canonical stage and category vocab
STAGE_KEYS = {
    "NEONATAL",
    "INFANCY",
    "TODDLERHOOD",
    "EARLY_CHILDHOOD",
    "MIDDLE_CHILDHOOD",
    "EARLY_ADOLESCENCE",
    "LATE_ADOLESCENCE",
    "EARLY_ADULTHOOD",
    "MIDDLE_ADULTHOOD",
    "LATE_ADULTHOOD",
    "END_OF_LIFE",
    "VERY_OLD",
    "UNSPECIFIED_STAGE",  # used when we cannot infer a stage
}
CATEGORY_KEYS = {
    "cognitive_development",
    "social_development",
    "emotional_development",
    "physical_development",
}

# Stage detection hints for filenames and folder names
_STAGE_HINTS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bneonat(al|e)?\b", re.I), "NEONATAL"),
    (re.compile(r"\binfanc(y|e)\b", re.I), "INFANCY"),
    (re.compile(r"\btoddler(s|hood)?\b", re.I), "TODDLERHOOD"),
    (re.compile(r"\bearly[_\-\s]?child(hood)?\b", re.I), "EARLY_CHILDHOOD"),
    (re.compile(r"\bmiddle[_\-\s]?child(hood)?\b", re.I), "MIDDLE_CHILDHOOD"),
    (re.compile(r"\bearly[_\-\s]?adolescen(ce|t)\b", re.I), "EARLY_ADOLESCENCE"),
    (re.compile(r"\blate[_\-\s]?adolescen(ce|t)\b", re.I), "LATE_ADOLESCENCE"),
    (re.compile(r"\bearly[_\-\s]?adult(hood)?\b", re.I), "EARLY_ADULTHOOD"),
    (re.compile(r"\bmiddle[_\-\s]?adult(hood)?\b", re.I), "MIDDLE_ADULTHOOD"),
    (re.compile(r"\blate[_\-\s]?adult(hood)?\b", re.I), "LATE_ADULTHOOD"),
    (re.compile(r"\bend[_\-\s]?of[_\-\s]?life\b", re.I), "END_OF_LIFE"),
    (re.compile(r"\bvery[_\-\s]?old\b", re.I), "VERY_OLD"),
]

# Camel/PascalCase tokenizer (e.g., "CulturalLineageTracing" -> ["Cultural","Lineage","Tracing"])
_CAMEL_RE = re.compile(r'[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+')


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


def initial_word(key: str) -> str:
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
    out: Dict[str, Any] = {}
    for k, v in _sorted_items_for_layer(d, depth, use_initial_word_top):
        out[k] = v
    return out


# ---------- Normalization helpers ----------

def _looks_like_categories_top(data: Dict[str, Any]) -> bool:
    if not isinstance(data, dict):
        return False
    if not data:
        return False
    keys = set(data.keys())
    # If there is any category key at the top, we consider it category-rooted
    return any(k in CATEGORY_KEYS for k in keys) and not any(k.upper() in STAGE_KEYS for k in keys)


def _canonicalize_stage_key_once(k: str) -> Optional[str]:
    """
    If k looks like a stage key in any reasonable casing or spacing, return canonical UPPER_SNAKE.
    Otherwise return None.
    """
    raw = re.sub(r"[^A-Za-z]", "_", k).strip("_").upper()
    # Collapse multiple underscores
    raw = re.sub(r"_+", "_", raw)
    # Common aliases
    aliases = {
        "EARLY_CHILD": "EARLY_CHILDHOOD",
        "MIDDLE_CHILD": "MIDDLE_CHILDHOOD",
        "EARLY_ADOLESCENT": "EARLY_ADOLESCENCE",
        "LATE_ADOLESCENT": "LATE_ADOLESCENCE",
        "END_OF_LIFE": "END_OF_LIFE",
        "VERY_OLD": "VERY_OLD",
        "INFANCY": "INFANCY",
        "NEONATAL": "NEONATAL",
        "TODDLER": "TODDLERHOOD",
        "TODDLERHOOD": "TODDLERHOOD",
        "EARLY_CHILDHOOD": "EARLY_CHILDHOOD",
        "MIDDLE_CHILDHOOD": "MIDDLE_CHILDHOOD",
        "EARLY_ADULTHOOD": "EARLY_ADULTHOOD",
        "MIDDLE_ADULTHOOD": "MIDDLE_ADULTHOOD",
        "LATE_ADULTHOOD": "LATE_ADULTHOOD",
        "VERY_OLD_AGE": "VERY_OLD",
        "SENIOR": "VERY_OLD",
    }
    # Direct hit
    if raw in STAGE_KEYS:
        return raw
    if raw in aliases:
        return aliases[raw]
    return None


def _detect_stage_from_path(p: Path) -> Optional[str]:
    haystack = " / ".join([p.stem] + [q for q in p.parts])
    for rx, stage in _STAGE_HINTS:
        if rx.search(haystack):
            return stage
    return None


def _canonicalize_top_stage_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        canon = _canonicalize_stage_key_once(k)
        if canon:
            # Merge if multiple variants map to the same canonical key
            if canon not in out:
                out[canon] = v
            else:
                if isinstance(out[canon], dict) and isinstance(v, dict):
                    deep_merge(out[canon], v)
                else:
                    out[canon] = v
        else:
            out[k] = v
    return out


def normalize_one_file_dict(data: Dict[str, Any], fp: Path) -> Dict[str, Any]:
    """
    Normalize a single file's dict to enforce:
    - top level is life stages
    - within a stage, second level is categories (not enforced here, just validated later)
    """
    if not isinstance(data, dict):
        return data

    # Canonicalize any top-level stage keys to UPPER_SNAKE
    data = _canonicalize_top_stage_keys(data)

    # If the file already has a stage at top, keep it
    if any((k in STAGE_KEYS) for k in data.keys()):
        return data

    # If top looks like categories, wrap under detected or unspecified stage
    if _looks_like_categories_top(data):
        stage = _detect_stage_from_path(fp) or "UNSPECIFIED_STAGE"
        return {stage: data}

    # Otherwise, leave as-is. Validation will flag if needed.
    return data


def post_merge_cleanup(merged: Dict[str, Any]) -> None:
    """
    After merging, if any of the four category keys leaked into the global root,
    move them under UNSPECIFIED_STAGE.
    """
    stray_keys = [k for k in list(merged.keys()) if k in CATEGORY_KEYS]
    if not stray_keys:
        return

    if "UNSPECIFIED_STAGE" not in merged or not isinstance(merged["UNSPECIFIED_STAGE"], dict):
        merged["UNSPECIFIED_STAGE"] = {}

    for k in stray_keys:
        node = merged.pop(k)
        if k not in merged["UNSPECIFIED_STAGE"]:
            merged["UNSPECIFIED_STAGE"][k] = node
        else:
            # Merge with existing category content under UNSPECIFIED_STAGE
            if isinstance(merged["UNSPECIFIED_STAGE"][k], dict) and isinstance(node, dict):
                deep_merge(merged["UNSPECIFIED_STAGE"][k], node)
            else:
                merged["UNSPECIFIED_STAGE"][k] = node


# ---------- Metadata and validation ----------

def is_leaf(value: Any) -> bool:
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
    if not isinstance(node, dict):
        depth = len(path)
        depth_hist[depth] += 1
        return 1, 0, 1, 0, depth

    depth = len(path)
    depth_hist[depth] += 1

    if len(node) == 0:
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


def validate_schema(merged: Dict[str, Any]) -> None:
    print("\n=== Schema Validation (life stage -> secondary-layer categories) ===")
    root_keys = set(merged.keys())
    non_stage_at_root = sorted([k for k in root_keys if k not in STAGE_KEYS])
    if non_stage_at_root:
        print("Root contains non-stage keys:", ", ".join(non_stage_at_root))
    else:
        print("Root OK. Only stage keys present.")

    for stage in sorted([k for k in root_keys if k in STAGE_KEYS]):
        node = merged.get(stage, {})
        if not isinstance(node, dict):
            print(f"- {stage}: not a dict at stage root")
            continue
        second_layer = set(node.keys())
        bad = sorted([k for k in second_layer if k not in CATEGORY_KEYS])
        missing = sorted([k for k in CATEGORY_KEYS if k not in second_layer])
        if not bad and not missing:
            print(f"- {stage}: OK")
        else:
            if bad:
                print(f"- {stage}: unexpected secondary keys -> {', '.join(bad)}")
            if missing:
                print(f"- {stage}: missing categories -> {', '.join(missing)}")


def print_metadata(merged: Dict[str, Any], files_count: int) -> None:
    key_occurrences: Dict[str, List[Tuple[str, ...]]] = defaultdict(list)
    depth_hist: Counter = Counter()

    total_nodes, internal_nodes, leaf_nodes, total_children_links, max_depth = traverse_collect(
        merged, tuple(), key_occurrences=key_occurrences, depth_hist=depth_hist
    )

    unique_keys = len(key_occurrences)
    duplicate_keys = {k: v for k, v in key_occurrences.items() if len(v) > 1}
    avg_branching = (total_children_links / internal_nodes) if internal_nodes > 0 else 0.0

    leaves_top = leaves_per_top_key(merged)

    print("\n=== PHOENIX_ontology Merge Report ===")
    print(f"Files merged: {files_count}")
    print(f"Total nodes: {total_nodes}")
    print(f"Internal nodes: {internal_nodes}")
    print(f"Leaf nodes: {leaf_nodes}")
    print(f"Unique keys: {unique_keys}")
    print(f"Max depth: {max_depth}")
    print(f"Average branching factor: {avg_branching:.3f}")

    print("\nDepth histogram (depth -> node count):")
    for d in sorted(depth_hist.keys()):
        print(f"  {d}: {depth_hist[d]}")

    print("\nLeaves per top-level key:")
    for k in sorted(leaves_top.keys(), key=str.lower):
        print(f"  {k}: {leaves_top[k]}")

    if duplicate_keys:
        print("\nDuplicate key names across hierarchy (key -> occurrence count and sample paths):")
    else:
        print("\nNo duplicate key names found across the hierarchy.")
    print("=== End of Report ===\n")


# -------------- main --------------

def main() -> None:
    # ---------- Your toggles ----------
    SORT_ENABLED = False
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

        # Normalize this file's schema first
        data = normalize_one_file_dict(data, fp)

        deep_merge(merged, data)

    # Ensure no category keys remain at global root
    post_merge_cleanup(merged)

    if SORT_ENABLED:
        merged = sort_structure(merged, depth=0, use_initial_word_top=USE_INITIAL_WORDS_FIRST_LAYER)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"Merged {len(files)} other")
    print(f"Sorted: {'yes (top layer by ' + ('01_pre_generation word' if USE_INITIAL_WORDS_FIRST_LAYER else 'full key') + ', deeper layers by full key)' if SORT_ENABLED else 'no'}")
    print(f"Output: {OUTPUT_PATH}")

    # Validation and metadata
    validate_schema(merged)
    print_metadata(merged, files_count=len(files))


if __name__ == "__main__":
    main()
