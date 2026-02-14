#!/usr/bin/env python3
"""
apply_multi_rank_fusion.py

Goal
----
Create per-pseudoprofile criterion×predictor relevance matrices by multi-rank fusion
(linear weighted average; equal weights by default) using:

(A) LLM-based criterion→predictor mapping ranks CSV:
    all_pseudoprofiles__predictor_ranks_dense.csv
    Required fields:
      pseudoprofile_id,part,criterion_path,criterion_leaf,rank,predictor_path,relevance_score

(B) HyDe-based GLOBAL predictor ranks (with scores) from dense_profiles.csv:
    dense_profiles.csv (one row per pseudoprofile_id for the HyDe run)
    Columns include (up to 200):
      global_001_rank, global_001_score, global_001_leaf_index,
      global_001_path_embedtext, global_001_path_full, global_001_path_lextext, ...
      ... repeated up to global_200_*

Important fixes vs previous version
-----------------------------------
1) Mapping relevance_score is on ~[0, 1000] (NOT [0,1]).
   The previous script clipped to [0,1], causing near-binary outputs (1 or NaN).
   This script NEVER clips mapping scores; instead it optionally scales HyDe scores
   onto the same scale as the mapping scores.

2) HyDe scores are taken from dense_profiles.csv (global_*_score) rather than fused_rankings.json.
   The HyDe side is GLOBAL (not per-criterion). We use it as a global prior per predictor leaf.

3) Sparse matrix output:
   By default, HyDe does NOT create new criterion×predictor edges.
   It only adjusts scores where a mapping edge exists (hyde_requires_mapping=True by default).

Output structure (clear)
------------------------
<output_dir>/
  run_<timestamp>/
    run_summary.json
    profiles_meta.csv                     (if --write_meta_json)
    failures.csv                          (if any)
    <pseudoprofile_id>/
      matrix.csv                          (wide: predictors x criteria; sparse with NaNs)
      matrix_sparse.csv                   (long: only non-null cells)
      meta.json                           (debug + counts + paths + settings)
      debug/
        mapping_raw_rows.csv
        mapping_agg_rows.csv
        mapping_predictor_cluster_to_hyde_matches.csv
        hyde_leaf_scores.csv              (raw + scaled)
        matrix_preview_top50.csv

Defaults
--------
- mapping scores assumed 0..1000 (set via --mapping_score_max).
- HyDe scores assumed cosine-like 0..1 and scaled by mapping_score_max
  (set via --hyde_scale_to_mapping_max; on by default).
- Criteria columns default to criterion_path (criterion clusters), as requested.

Usage
-----
python apply_multi_rank_fusion.py
python apply_multi_rank_fusion.py --pseudoprofile_id pseudoprofile_FTC_ID005
python apply_multi_rank_fusion.py --w_mapping 0.7 --w_hyde 0.3
python apply_multi_rank_fusion.py --no_expand_mapping_to_hyde_leaves
python apply_multi_rank_fusion.py --criterion_columns criterion_leaf
python apply_multi_rank_fusion.py --mapping_score_mode unit   # maps to 0..1, HyDe stays 0..1

"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict, OrderedDict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ============================================================
# Defaults (paths you provided)
# ============================================================
DEFAULT_MAPPING_RANKS_CSV = (
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/Evaluation/"
    "03_construction_initial_observation_model/helpers/"
    "00_LLM_based_mapping_based_predictor_ranks/all_pseudoprofiles__predictor_ranks_dense.csv"
)

DEFAULT_HYDE_PROFILES_DIR = (
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/Evaluation/"
    "03_construction_initial_observation_model/helpers/"
    "00_HyDe_based_predictor_ranks/runs/2026-01-15_19-51-58/profiles"
)

# dense_profiles.csv lives one directory above the "profiles" directory by convention
DEFAULT_HYDE_DENSE_PROFILES_CSV = os.path.join(os.path.dirname(DEFAULT_HYDE_PROFILES_DIR), "dense_profiles.csv")

DEFAULT_OUTPUT_DIR = (
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/Evaluation/"
    "03_construction_initial_observation_model/helpers/"
    "01_fused_ranks/results"
)

DEFAULT_MAPPING_PART = "post_per_criterion"

# Fusion weights
DEFAULT_W_MAPPING = 0.5
DEFAULT_W_HYDE = 0.5

# Score scaling defaults (mapping is 0..1000)
DEFAULT_MAPPING_SCORE_MAX = 1000.0


# ============================================================
# Utilities
# ============================================================
SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
BRACKET_DOMAIN_RE = re.compile(r"^\s*\[([A-Za-z]+)\]\s*$")


def _now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _safe_filename(s: str) -> str:
    s = SAFE_FILENAME_RE.sub("_", str(s).strip())
    return s[:180] if len(s) > 180 else s


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        if isinstance(x, float) and pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        if isinstance(x, float) and pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def _require_exists(path: str, what: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{what} not found: {path}")


def _is_nan(x: Any) -> bool:
    try:
        return pd.isna(x)
    except Exception:
        return False


# ============================================================
# Path normalization + matching
# ============================================================
_SPLIT_ANY_SEP_RE = re.compile(r"\s*(?:>|/)\s*")


def split_path_any(path: str) -> List[str]:
    """
    Robust splitter for ontology-like paths that may use:
      - "BIO / X / Y"
      - "[BIO] > X > Y"
      - "BIO > X > Y"
    Also strips:
      - '*' markers
      - bracketed domains like "[BIO]" -> "BIO"
    """
    s = _safe_str(path).strip()
    if not s:
        return []

    s = s.replace("*", "").strip()
    s = re.sub(r"\s+", " ", s)
    parts = [p.strip() for p in _SPLIT_ANY_SEP_RE.split(s) if p.strip()]

    norm_parts: List[str] = []
    for p in parts:
        m = BRACKET_DOMAIN_RE.match(p)
        if m:
            norm_parts.append(m.group(1).strip())
        else:
            p2 = re.sub(r"^\[([A-Za-z]+)\]\s*", r"\1 ", p).strip()
            norm_parts.append(p2)
    return norm_parts


def canonical_path(path: str) -> str:
    return " / ".join(split_path_any(path))


def path_depth(path: str) -> int:
    return len(split_path_any(path))


def is_prefix_path(prefix: str, full: str) -> bool:
    p = canonical_path(prefix)
    f = canonical_path(full)
    if not p or not f:
        return False
    if f == p:
        return True
    return f.startswith(p + " / ")


# ============================================================
# HyDe GLOBAL loader from dense_profiles.csv
# ============================================================
GLOBAL_SCORE_COL_RE = re.compile(r"^global_(\d{3})_score$")


def _pick_path_for_global_i(row: pd.Series, i: int) -> str:
    """
    Prefer *_path_full, else embedtext, else lextext.
    """
    cols = [
        f"global_{i:03d}_path_full",
        f"global_{i:03d}_path_embedtext",
        f"global_{i:03d}_path_lextext",
    ]
    for c in cols:
        if c in row.index:
            v = row[c]
            if not _is_nan(v):
                s = _safe_str(v).strip()
                if s:
                    return s
    return ""


def load_hyde_dense_profiles_scores(
    dense_profiles_csv: str,
    *,
    max_global: int = 200,
    top_k: Optional[int] = None,
    duplicate_policy: str = "max",
) -> Dict[str, Dict[str, float]]:
    """
    Returns:
      dict[pseudoprofile_id][leaf_path_full] = global_score (0..1-ish)

    duplicate_policy:
      - "max"  : keep max score per leaf_path
      - "mean" : mean across duplicates
    """
    _require_exists(dense_profiles_csv, "HyDe dense_profiles.csv")

    df = pd.read_csv(dense_profiles_csv)
    if "pseudoprofile_id" not in df.columns:
        raise ValueError("dense_profiles.csv missing column: pseudoprofile_id")

    # Determine which indices exist from columns (robust to fewer than 200)
    idxs: List[int] = []
    for c in df.columns:
        m = GLOBAL_SCORE_COL_RE.match(c)
        if m:
            idxs.append(int(m.group(1)))
    idxs = sorted(set(i for i in idxs if 1 <= i <= max_global))
    if not idxs:
        raise ValueError("dense_profiles.csv has no global_###_score columns (cannot load HyDe scores).")

    out: Dict[str, Dict[str, float]] = {}

    for _, row in df.iterrows():
        pid = _safe_str(row.get("pseudoprofile_id", "")).strip()
        if not pid:
            continue

        # If duplicates exist in file, later rows overwrite; that's typically fine for a single run export.
        leaf_to_scores: Dict[str, List[float]] = defaultdict(list)

        # Optional: respect global_top_n_computed if present
        n_comp = row.get("global_top_n_computed", None)
        n_comp_i = None
        if n_comp is not None and not _is_nan(n_comp):
            try:
                n_comp_i = int(n_comp)
            except Exception:
                n_comp_i = None

        effective_max = max_global
        if n_comp_i is not None and n_comp_i > 0:
            effective_max = min(effective_max, n_comp_i)
        if top_k is not None:
            effective_max = min(effective_max, int(top_k))

        for i in idxs:
            if i > effective_max:
                break
            score_col = f"global_{i:03d}_score"
            if score_col not in row.index:
                continue
            s = _safe_float(row[score_col])
            if s is None:
                continue

            p = _pick_path_for_global_i(row, i)
            if not p:
                continue

            leaf_to_scores[p].append(float(s))

        if not leaf_to_scores:
            out[pid] = {}
            continue

        pid_scores: Dict[str, float] = {}
        for leaf_path, scores in leaf_to_scores.items():
            if not scores:
                continue
            if duplicate_policy == "mean":
                pid_scores[leaf_path] = float(sum(scores) / len(scores))
            else:
                pid_scores[leaf_path] = float(max(scores))

        out[pid] = pid_scores

    return out


# ============================================================
# Mapping loader + aggregation
# ============================================================
REQUIRED_MAPPING_COLS = [
    "pseudoprofile_id",
    "part",
    "criterion_path",
    "criterion_leaf",
    "rank",
    "predictor_path",
    "relevance_score",
]


def load_mapping_ranks_df(mapping_csv: str) -> pd.DataFrame:
    _require_exists(mapping_csv, "Mapping ranks CSV")

    df = pd.read_csv(mapping_csv)

    missing = [c for c in REQUIRED_MAPPING_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Mapping CSV missing required columns: {missing}")

    df["pseudoprofile_id"] = df["pseudoprofile_id"].astype(str)
    df["part"] = df["part"].astype(str)
    df["criterion_path"] = df["criterion_path"].fillna("").astype(str)
    df["criterion_leaf"] = df["criterion_leaf"].fillna("").astype(str)
    df["predictor_path"] = df["predictor_path"].fillna("").astype(str)

    df["relevance_score"] = pd.to_numeric(df["relevance_score"], errors="coerce")
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")

    return df


def dedupe_names(names: List[str]) -> List[str]:
    seen: Dict[str, int] = {}
    out: List[str] = []
    for n in names:
        base = n
        if base not in seen:
            seen[base] = 1
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base} ({seen[base]})")
    return out


def make_criterion_label(
    criterion_leaf: str,
    criterion_path: str,
    *,
    mode: str,
    fallback_i: int,
) -> str:
    """
    mode:
      - "criterion_path"     => use criterion_path (cluster)
      - "criterion_leaf"     => use criterion_leaf
      - "leaf_or_path"       => leaf if present else path
      - "path_or_leaf"       => path if present else leaf
    """
    leaf = _safe_str(criterion_leaf).strip()
    path = _safe_str(criterion_path).strip()

    if mode == "criterion_leaf":
        return leaf if leaf else (path if path else f"UNMAPPED_{fallback_i:03d}")
    if mode == "leaf_or_path":
        return leaf if leaf else (path if path else f"UNMAPPED_{fallback_i:03d}")
    if mode == "path_or_leaf":
        return path if path else (leaf if leaf else f"UNMAPPED_{fallback_i:03d}")

    # default: criterion_path
    return path if path else (leaf if leaf else f"UNMAPPED_{fallback_i:03d}")


# ============================================================
# Fusion logic (no clipping; supports scaling)
# ============================================================
def fuse_scores_linear(
    mapping_score: Optional[float],
    hyde_score: Optional[float],
    *,
    w_mapping: float,
    w_hyde: float,
    missing_as_zero: bool,
) -> Optional[float]:
    """
    Linear fusion with optional missing handling.

    If missing_as_zero:
      treat missing component scores as 0 but still include its weight.
    Else:
      ignore missing components by removing their weight from denominator.

    Returns None if both missing and missing_as_zero=False.
    """
    m = _safe_float(mapping_score)
    h = _safe_float(hyde_score)

    wm = float(w_mapping)
    wh = float(w_hyde)

    if wm < 0 or wh < 0:
        raise ValueError("Fusion weights must be non-negative.")

    if missing_as_zero:
        denom = wm + wh
        if denom <= 0:
            return None
        m_val = 0.0 if m is None else float(m)
        h_val = 0.0 if h is None else float(h)
        return (wm * m_val + wh * h_val) / denom

    num = 0.0
    denom = 0.0
    if m is not None and wm > 0:
        num += wm * float(m)
        denom += wm
    if h is not None and wh > 0:
        num += wh * float(h)
        denom += wh
    if denom <= 0:
        return None
    return num / denom


# ============================================================
# Mapping predictor cluster expansion to HyDe leaves
# ============================================================
def build_hyde_leaf_index(hyde_scores: Dict[str, float]) -> Dict[str, List[str]]:
    idx: Dict[str, List[str]] = defaultdict(list)
    for lp in hyde_scores.keys():
        segs = split_path_any(lp)
        root = segs[0] if segs else ""
        idx[root].append(lp)
    return idx


def match_cluster_to_hyde_leaves(cluster_path: str, *, hyde_index_by_root: Dict[str, List[str]]) -> List[str]:
    segs = split_path_any(cluster_path)
    if not segs:
        return []
    root = segs[0]
    candidates = hyde_index_by_root.get(root, [])
    return [lp for lp in candidates if is_prefix_path(cluster_path, lp)]


# ============================================================
# Output directory structure helpers
# ============================================================
def profile_output_dirs(base_run_dir: str, pseudoprofile_id: str) -> Dict[str, str]:
    pdir = _ensure_dir(os.path.join(base_run_dir, _safe_filename(pseudoprofile_id)))
    debug_dir = _ensure_dir(os.path.join(pdir, "debug"))
    return {
        "profile_dir": pdir,
        "debug_dir": debug_dir,
        "matrix_csv": os.path.join(pdir, "matrix.csv"),
        "matrix_sparse_csv": os.path.join(pdir, "matrix_sparse.csv"),
        "meta_json": os.path.join(pdir, "meta.json"),
        "debug_mapping_raw": os.path.join(debug_dir, "mapping_raw_rows.csv"),
        "debug_mapping_agg": os.path.join(debug_dir, "mapping_agg_rows.csv"),
        "debug_cluster_matches": os.path.join(debug_dir, "mapping_predictor_cluster_to_hyde_matches.csv"),
        "debug_hyde_scores": os.path.join(debug_dir, "hyde_leaf_scores.csv"),
        "debug_matrix_preview": os.path.join(debug_dir, "matrix_preview_top50.csv"),
    }


# ============================================================
# Core per-profile builder
# ============================================================
def build_fused_matrix_for_profile(
    mapping_df: pd.DataFrame,
    hyde_scores_by_pid: Dict[str, Dict[str, float]],
    pseudoprofile_id: str,
    *,
    mapping_part: str,
    max_rank_mapping: Optional[int],
    expand_mapping_to_hyde_leaves: bool,
    keep_unmatched_cluster_rows: bool,
    criterion_columns_mode: str,
    w_mapping: float,
    w_hyde: float,
    missing_as_zero: bool,
    mapping_score_mode: str,               # "raw" or "unit"
    mapping_score_max: float,              # scaling constant
    hyde_scale_to_mapping_max: bool,       # scale HyDe(0..1) to mapping scale
    hyde_requires_mapping: bool,           # keep matrix sparse (default True)
    write_debug: bool,
    debug_paths: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Returns:
      (matrix_wide_df, matrix_sparse_df, meta)
    """
    pid = str(pseudoprofile_id)

    # ---- mapping subset
    sub = mapping_df[(mapping_df["pseudoprofile_id"] == pid) & (mapping_df["part"] == str(mapping_part))].copy()
    if sub.empty:
        raise ValueError(f"No mapping rows found for pseudoprofile_id={pid} with part={mapping_part}")

    if max_rank_mapping is not None:
        sub = sub[pd.to_numeric(sub["rank"], errors="coerce").fillna(10**9) <= int(max_rank_mapping)].copy()
        if sub.empty:
            raise ValueError(f"After max_rank_mapping={max_rank_mapping}, no mapping rows remain for pid={pid}")

    sub["predictor_path"] = sub["predictor_path"].fillna("").astype(str)
    sub = sub[sub["predictor_path"].str.strip() != ""].copy()
    sub = sub[pd.to_numeric(sub["relevance_score"], errors="coerce").notna()].copy()
    if sub.empty:
        raise ValueError(f"Mapping rows exist but none have predictor_path + relevance_score for pid={pid}")

    if write_debug:
        sub.to_csv(debug_paths["debug_mapping_raw"], index=False)

    # ---- mapping score scaling
    if mapping_score_mode not in ("raw", "unit"):
        raise ValueError("--mapping_score_mode must be 'raw' or 'unit'")

    # create a working column
    sub["_mapping_score"] = pd.to_numeric(sub["relevance_score"], errors="coerce")
    if mapping_score_mode == "unit":
        if mapping_score_max <= 0:
            raise ValueError("mapping_score_max must be > 0 for mapping_score_mode=unit")
        sub["_mapping_score"] = sub["_mapping_score"] / float(mapping_score_max)

    # ---- build criterion labels in insertion order
    crit_key_to_col: "OrderedDict[Tuple[str, str], str]" = OrderedDict()
    fallback_i = 1
    for _, r in sub.iterrows():
        c_leaf = _safe_str(r.get("criterion_leaf", "")).strip()
        c_path = _safe_str(r.get("criterion_path", "")).strip()
        key = (c_leaf, c_path)
        if key not in crit_key_to_col:
            crit_key_to_col[key] = make_criterion_label(
                c_leaf,
                c_path,
                mode=criterion_columns_mode,
                fallback_i=fallback_i,
            )
            fallback_i += 1

    crit_keys = list(crit_key_to_col.keys())
    crit_cols = dedupe_names(list(crit_key_to_col.values()))

    # re-bind after dedupe
    crit_key_to_col = OrderedDict((k, crit_cols[i]) for i, k in enumerate(crit_keys))

    # ---- aggregate mapping duplicates by mean: (criterion_leaf, criterion_path, predictor_path)
    grp = (
        sub.groupby(["criterion_leaf", "criterion_path", "predictor_path"], dropna=False)["_mapping_score"]
        .mean()
        .reset_index()
    )
    if write_debug:
        grp.to_csv(debug_paths["debug_mapping_agg"], index=False)

    mapping_scores_by_crit: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(dict)
    for _, r in grp.iterrows():
        c_leaf = _safe_str(r.get("criterion_leaf", "")).strip()
        c_path = _safe_str(r.get("criterion_path", "")).strip()
        p_path = _safe_str(r.get("predictor_path", "")).strip()
        s = _safe_float(r.get("_mapping_score", None))
        if p_path and s is not None:
            mapping_scores_by_crit[(c_leaf, c_path)][p_path] = float(s)

    # ---- HyDe GLOBAL scores for this pid (0..1-ish)
    hyde_raw = hyde_scores_by_pid.get(pid, {}) or {}

    # scale HyDe scores to match mapping scale if requested
    hyde_scale = 1.0
    if mapping_score_mode == "raw" and hyde_scale_to_mapping_max:
        hyde_scale = float(mapping_score_max)
    elif mapping_score_mode == "unit":
        hyde_scale = 1.0

    hyde_scores_scaled: Dict[str, float] = {lp: float(s) * hyde_scale for lp, s in hyde_raw.items()}

    if write_debug:
        pd.DataFrame(
            [
                {
                    "leaf_path": lp,
                    "leaf_path_canonical": canonical_path(lp),
                    "hyde_score_raw": hyde_raw.get(lp, None),
                    "hyde_score_scaled": hyde_scores_scaled.get(lp, None),
                }
                for lp in sorted(hyde_raw.keys(), key=lambda x: canonical_path(x))
            ]
        ).to_csv(debug_paths["debug_hyde_scores"], index=False)

    # ---- expand mapping predictor clusters to HyDe leaves (optional)
    final_mapping_by_crit: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(dict)
    predictor_rows: set = set()
    cluster_match_rows: List[Dict[str, Any]] = []

    if expand_mapping_to_hyde_leaves and hyde_scores_scaled:
        hyde_index = build_hyde_leaf_index(hyde_scores_scaled)

        # canonical->leaf for quick direct matches
        hyde_can_to_leaf: Dict[str, str] = {}
        for lp in hyde_scores_scaled.keys():
            can = canonical_path(lp)
            if can not in hyde_can_to_leaf:
                hyde_can_to_leaf[can] = lp

        all_mapping_predictor_paths = sorted({p for d in mapping_scores_by_crit.values() for p in d.keys()})

        cluster_to_leaves: Dict[str, List[str]] = {}
        for cluster in all_mapping_predictor_paths:
            can = canonical_path(cluster)
            direct = hyde_can_to_leaf.get(can)
            if direct is not None:
                leaves = [direct]
            else:
                leaves = match_cluster_to_hyde_leaves(cluster, hyde_index_by_root=hyde_index)
            cluster_to_leaves[cluster] = leaves

            cluster_match_rows.append(
                {
                    "mapping_predictor_path": cluster,
                    "mapping_predictor_path_canonical": can,
                    "n_matched_hyde_leaves": len(leaves),
                    "matched_hyde_leaves_joined": " || ".join(leaves)[:20000],
                }
            )

        if write_debug:
            pd.DataFrame(cluster_match_rows).to_csv(debug_paths["debug_cluster_matches"], index=False)

        # Most-specific-wins propagation (cluster score -> matched leaves)
        for ckey, p_to_s in mapping_scores_by_crit.items():
            leaf_best: Dict[str, Tuple[int, float]] = {}  # leaf -> (depth, score)
            unmatched_clusters: Dict[str, float] = {}

            for cluster_path, score in p_to_s.items():
                leaves = cluster_to_leaves.get(cluster_path, []) or []
                depth = path_depth(cluster_path)

                if leaves:
                    for leaf_path in leaves:
                        prev = leaf_best.get(leaf_path)
                        if prev is None:
                            leaf_best[leaf_path] = (depth, score)
                        else:
                            prev_depth, prev_score = prev
                            if (depth > prev_depth) or (depth == prev_depth and score > prev_score):
                                leaf_best[leaf_path] = (depth, score)
                else:
                    if keep_unmatched_cluster_rows:
                        unmatched_clusters[cluster_path] = score

            for leaf_path, (_d, s) in leaf_best.items():
                final_mapping_by_crit[ckey][leaf_path] = float(s)
                predictor_rows.add(leaf_path)
            for cluster_path, s in unmatched_clusters.items():
                final_mapping_by_crit[ckey][cluster_path] = float(s)
                predictor_rows.add(cluster_path)

    else:
        # no expansion (or no HyDe leaves known): keep predictor_path as-is
        if write_debug:
            pd.DataFrame(
                [
                    {
                        "mapping_predictor_path": "",
                        "mapping_predictor_path_canonical": "",
                        "n_matched_hyde_leaves": 0,
                        "matched_hyde_leaves_joined": "",
                        "note": "Expansion disabled or HyDe scores unavailable for this profile.",
                    }
                ]
            ).to_csv(debug_paths["debug_cluster_matches"], index=False)

        for ckey, p_to_s in mapping_scores_by_crit.items():
            for p, s in p_to_s.items():
                final_mapping_by_crit[ckey][p] = float(s)
                predictor_rows.add(p)

    if not predictor_rows:
        raise ValueError(f"No predictor rows could be constructed for pid={pid}")

    # ---- build fused matrix (sparse by default)
    predictors = sorted(list(predictor_rows), key=lambda x: canonical_path(x))
    ckey_to_col = dict(crit_key_to_col)

    # Prepare wide matrix data
    data: Dict[str, List[Optional[float]]] = {ckey_to_col[ck]: [] for ck in crit_keys}
    row_labels: List[str] = []

    # For sparse output
    sparse_rows: List[Dict[str, Any]] = []

    for pred in predictors:
        row_labels.append(pred)
        pred_hyde = hyde_scores_scaled.get(pred, None)

        for ckey in crit_keys:
            col = ckey_to_col[ckey]
            mapping_score = final_mapping_by_crit.get(ckey, {}).get(pred, None)

            if hyde_requires_mapping and mapping_score is None:
                fused = None
            else:
                fused = fuse_scores_linear(
                    mapping_score,
                    pred_hyde,
                    w_mapping=w_mapping,
                    w_hyde=w_hyde,
                    missing_as_zero=missing_as_zero,
                )

            data[col].append(fused)

            if fused is not None:
                sparse_rows.append(
                    {
                        "pseudoprofile_id": pid,
                        "predictor_path": pred,
                        "predictor_path_canonical": canonical_path(pred),
                        "criterion": col,
                        "criterion_leaf": ckey[0],
                        "criterion_path": ckey[1],
                        "mapping_score": mapping_score,
                        "hyde_score_scaled": pred_hyde,
                        "fused_score": fused,
                    }
                )

    mat = pd.DataFrame(data, index=row_labels)

    # row ordering by mean fused (non-null) then canonical path
    row_mean = mat.mean(axis=1, skipna=True)
    mat.insert(0, "__row_mean__", row_mean)
    mat.insert(1, "__row_canonical__", [canonical_path(x) for x in row_labels])
    mat = mat.sort_values(["__row_mean__", "__row_canonical__"], ascending=[False, True]).drop(
        columns=["__row_mean__", "__row_canonical__"]
    )

    matrix_wide = mat.reset_index().rename(columns={"index": "predictor_path"})

    matrix_sparse = pd.DataFrame(sparse_rows)
    if not matrix_sparse.empty:
        matrix_sparse = matrix_sparse.sort_values(["fused_score"], ascending=False)

    if write_debug:
        matrix_wide.head(50).to_csv(debug_paths["debug_matrix_preview"], index=False)

    meta = {
        "pseudoprofile_id": pid,
        "mapping_part": mapping_part,
        "max_rank_mapping": max_rank_mapping,
        "criterion_columns_mode": criterion_columns_mode,
        "n_mapping_rows_raw": int(len(sub)),
        "n_mapping_rows_agg": int(len(grp)),
        "n_criteria": int(len(crit_cols)),
        "n_predictors_rows": int(matrix_wide.shape[0]),
        "n_sparse_cells": int(len(matrix_sparse)),
        "weights": {"w_mapping": float(w_mapping), "w_hyde": float(w_hyde)},
        "missing_as_zero": bool(missing_as_zero),
        "mapping_score_mode": mapping_score_mode,
        "mapping_score_max": float(mapping_score_max),
        "hyde_scale_to_mapping_max": bool(hyde_scale_to_mapping_max),
        "hyde_scale_factor_used": float(hyde_scale),
        "hyde_requires_mapping": bool(hyde_requires_mapping),
        "expand_mapping_to_hyde_leaves": bool(expand_mapping_to_hyde_leaves and bool(hyde_scores_scaled)),
        "keep_unmatched_cluster_rows": bool(keep_unmatched_cluster_rows),
        "n_hyde_leaves_scored_raw": int(len(hyde_raw)),
        "notes": (
            "Mapping scores are NOT clipped. If mapping_score_mode=raw, HyDe scores are scaled by mapping_score_max "
            "when hyde_scale_to_mapping_max=True."
        ),
    }

    return matrix_wide, matrix_sparse, meta


# ============================================================
# Main
# ============================================================
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Apply multi-rank fusion (LLM mapping + HyDe GLOBAL) to produce per-pseudoprofile criterion×predictor matrices."
    )

    ap.add_argument("--mapping_ranks_csv", default=DEFAULT_MAPPING_RANKS_CSV)
    ap.add_argument("--hyde_profiles_dir", default=DEFAULT_HYDE_PROFILES_DIR, help="HyDe run 'profiles' directory (used to infer dense_profiles.csv default).")
    ap.add_argument("--hyde_dense_profiles_csv", default=None, help="Path to dense_profiles.csv (default inferred from hyde_profiles_dir).")
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)

    ap.add_argument("--mapping_part", default=DEFAULT_MAPPING_PART)
    ap.add_argument("--max_rank_mapping", type=int, default=None)

    ap.add_argument("--pseudoprofile_id", default=None)

    ap.add_argument("--w_mapping", type=float, default=DEFAULT_W_MAPPING)
    ap.add_argument("--w_hyde", type=float, default=DEFAULT_W_HYDE)
    ap.add_argument("--missing_as_zero", action="store_true")

    ap.add_argument("--expand_mapping_to_hyde_leaves", action="store_true", default=True)
    ap.add_argument("--no_expand_mapping_to_hyde_leaves", action="store_true", default=False)
    ap.add_argument("--keep_unmatched_cluster_rows", action="store_true", default=True)

    ap.add_argument(
        "--criterion_columns",
        choices=["criterion_path", "criterion_leaf", "leaf_or_path", "path_or_leaf"],
        default="criterion_path",
        help="Which criterion label to use as matrix columns (default: criterion_path for criterion clusters).",
    )

    ap.add_argument("--write_meta_json", action="store_true", default=True)
    ap.add_argument("--no_write_meta_json", action="store_true", default=False)

    ap.add_argument("--write_debug", action="store_true", default=True)
    ap.add_argument("--no_write_debug", action="store_true", default=False)

    ap.add_argument("--run_name", default=None)

    # Score scaling / sparsity options
    ap.add_argument(
        "--mapping_score_mode",
        choices=["raw", "unit"],
        default="raw",
        help="raw: keep mapping scores as-is (e.g., 0..1000). unit: divide mapping by mapping_score_max (0..1).",
    )
    ap.add_argument("--mapping_score_max", type=float, default=DEFAULT_MAPPING_SCORE_MAX, help="Used for scaling when mapping_score_mode=unit, and for scaling HyDe to mapping when enabled.")
    ap.add_argument(
        "--hyde_scale_to_mapping_max",
        action="store_true",
        default=True,
        help="If mapping_score_mode=raw, scale HyDe scores (0..1) by mapping_score_max (default: on).",
    )
    ap.add_argument(
        "--no_hyde_scale_to_mapping_max",
        action="store_true",
        default=False,
        help="Disable scaling HyDe by mapping_score_max in raw mode.",
    )
    ap.add_argument(
        "--hyde_requires_mapping",
        action="store_true",
        default=True,
        help="Keep matrix sparse: only compute fused scores where a mapping edge exists (default: on).",
    )
    ap.add_argument(
        "--no_hyde_requires_mapping",
        action="store_true",
        default=False,
        help="Allow HyDe to fill cells without mapping (less sparse).",
    )

    # HyDe dense parsing
    ap.add_argument("--hyde_top_k_global", type=int, default=None, help="Only use top-k global predictors from dense_profiles.csv (by index).")
    ap.add_argument("--hyde_duplicate_policy", choices=["max", "mean"], default="max", help="How to combine duplicate leaf paths in dense_profiles.csv for a profile.")

    args = ap.parse_args()

    # Validate critical inputs early
    _require_exists(args.mapping_ranks_csv, "Mapping ranks CSV")
    _require_exists(args.hyde_profiles_dir, "HyDe profiles dir")

    if float(args.w_mapping) == 0.0 and float(args.w_hyde) == 0.0:
        raise ValueError("At least one fusion weight must be > 0.")

    expand = bool(args.expand_mapping_to_hyde_leaves) and (not bool(args.no_expand_mapping_to_hyde_leaves))
    write_meta = bool(args.write_meta_json) and (not bool(args.no_write_meta_json))
    write_debug = bool(args.write_debug) and (not bool(args.no_write_debug))
    hyde_scale_to_mapping_max = bool(args.hyde_scale_to_mapping_max) and (not bool(args.no_hyde_scale_to_mapping_max))
    hyde_requires_mapping = bool(args.hyde_requires_mapping) and (not bool(args.no_hyde_requires_mapping))

    # Resolve dense_profiles.csv path
    dense_csv = args.hyde_dense_profiles_csv
    if not dense_csv:
        dense_csv = os.path.join(os.path.dirname(args.hyde_profiles_dir), "dense_profiles.csv")
    _require_exists(dense_csv, "HyDe dense_profiles.csv")

    base_out = _ensure_dir(args.output_dir)
    run_id = _safe_filename(args.run_name) if args.run_name else f"run_{_now_stamp()}"
    run_dir = _ensure_dir(os.path.join(base_out, run_id))

    _log("====================================")
    _log("RUN START: apply_multi_rank_fusion")
    _log(f"run_dir                  : {run_dir}")
    _log(f"mapping_ranks_csv        : {args.mapping_ranks_csv}")
    _log(f"hyde_profiles_dir        : {args.hyde_profiles_dir}")
    _log(f"hyde_dense_profiles_csv  : {dense_csv}")
    _log(f"mapping_part             : {args.mapping_part}")
    _log(f"max_rank_mapping         : {args.max_rank_mapping}")
    _log(f"criterion_columns        : {args.criterion_columns}")
    _log(f"weights                  : w_mapping={args.w_mapping} | w_hyde={args.w_hyde}")
    _log(f"missing_as_zero          : {bool(args.missing_as_zero)}")
    _log(f"expand_to_leaves         : {expand}")
    _log(f"keep_unmatched_clusters  : {bool(args.keep_unmatched_cluster_rows)}")
    _log(f"mapping_score_mode       : {args.mapping_score_mode}")
    _log(f"mapping_score_max        : {args.mapping_score_max}")
    _log(f"hyde_scale_to_mapping_max: {hyde_scale_to_mapping_max}")
    _log(f"hyde_requires_mapping    : {hyde_requires_mapping}")
    _log(f"hyde_top_k_global        : {args.hyde_top_k_global}")
    _log(f"hyde_duplicate_policy    : {args.hyde_duplicate_policy}")
    _log(f"write_debug              : {write_debug}")
    _log("====================================")

    mapping_df = load_mapping_ranks_df(args.mapping_ranks_csv)

    # Load HyDe dense scores once (fast + consistent)
    _log("Loading HyDe dense profiles scores...")
    hyde_scores_by_pid = load_hyde_dense_profiles_scores(
        dense_csv,
        max_global=200,
        top_k=args.hyde_top_k_global,
        duplicate_policy=str(args.hyde_duplicate_policy),
    )
    _log(f"Loaded HyDe scores for {len(hyde_scores_by_pid)} profiles from dense_profiles.csv")

    # Determine profiles to run
    if args.pseudoprofile_id:
        pids = [str(args.pseudoprofile_id)]
    else:
        pids = (
            mapping_df[mapping_df["part"] == str(args.mapping_part)]["pseudoprofile_id"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        pids.sort()

    if not pids:
        _log(f"No pseudoprofiles found for mapping_part={args.mapping_part}. Exiting.")
        return 1

    _log(f"Profiles to process: {len(pids)}")

    metas: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    n_ok = 0

    for i, pid in enumerate(pids, start=1):
        try:
            _log(f"[{i}/{len(pids)}] Processing {pid} ...")

            paths = profile_output_dirs(run_dir, pid)

            matrix_wide, matrix_sparse, meta = build_fused_matrix_for_profile(
                mapping_df=mapping_df,
                hyde_scores_by_pid=hyde_scores_by_pid,
                pseudoprofile_id=pid,
                mapping_part=str(args.mapping_part),
                max_rank_mapping=args.max_rank_mapping,
                expand_mapping_to_hyde_leaves=expand,
                keep_unmatched_cluster_rows=bool(args.keep_unmatched_cluster_rows),
                criterion_columns_mode=str(args.criterion_columns),
                w_mapping=float(args.w_mapping),
                w_hyde=float(args.w_hyde),
                missing_as_zero=bool(args.missing_as_zero),
                mapping_score_mode=str(args.mapping_score_mode),
                mapping_score_max=float(args.mapping_score_max),
                hyde_scale_to_mapping_max=bool(hyde_scale_to_mapping_max),
                hyde_requires_mapping=bool(hyde_requires_mapping),
                write_debug=write_debug,
                debug_paths=paths,
            )

            matrix_wide.to_csv(paths["matrix_csv"], index=False)
            matrix_sparse.to_csv(paths["matrix_sparse_csv"], index=False)

            if write_meta:
                with open(paths["meta_json"], "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)

            metas.append(meta)
            n_ok += 1

            _log(
                f"OK  {pid} | predictors={meta['n_predictors_rows']} criteria={meta['n_criteria']} "
                f"| sparse_cells={meta['n_sparse_cells']} | out={paths['matrix_csv']}"
            )

        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            failures.append({"pseudoprofile_id": str(pid), "error": err})
            _log(f"FAIL {pid} | {err}")

    # Summary files at run root
    summary = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mapping_ranks_csv": str(args.mapping_ranks_csv),
        "hyde_profiles_dir": str(args.hyde_profiles_dir),
        "hyde_dense_profiles_csv": str(dense_csv),
        "output_dir": str(run_dir),
        "mapping_part": str(args.mapping_part),
        "max_rank_mapping": args.max_rank_mapping,
        "criterion_columns": str(args.criterion_columns),
        "weights": {"w_mapping": float(args.w_mapping), "w_hyde": float(args.w_hyde)},
        "missing_as_zero": bool(args.missing_as_zero),
        "expand_mapping_to_hyde_leaves": bool(expand),
        "keep_unmatched_cluster_rows": bool(args.keep_unmatched_cluster_rows),
        "mapping_score_mode": str(args.mapping_score_mode),
        "mapping_score_max": float(args.mapping_score_max),
        "hyde_scale_to_mapping_max": bool(hyde_scale_to_mapping_max),
        "hyde_requires_mapping": bool(hyde_requires_mapping),
        "hyde_top_k_global": args.hyde_top_k_global,
        "hyde_duplicate_policy": str(args.hyde_duplicate_policy),
        "write_debug": bool(write_debug),
        "n_profiles_total": int(len(pids)),
        "n_ok": int(n_ok),
        "n_fail": int(len(failures)),
    }

    with open(os.path.join(run_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if metas and write_meta:
        pd.DataFrame(metas).to_csv(os.path.join(run_dir, "profiles_meta.csv"), index=False)

    if failures:
        pd.DataFrame(failures).to_csv(os.path.join(run_dir, "failures.csv"), index=False)

    _log("====================================")
    _log(f"RUN END | ok={n_ok} fail={len(failures)}")
    _log(f"Run dir: {run_dir}")
    _log("====================================")

    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())

#TODO: currently does not work — fix the HyDe-based logic to obtain the rel_scores ; for now just use LLM-based mapping scores
