#!/usr/bin/env python3
"""
bipartite_model_visualization.py

Creates publication-oriented visuals per pseudoprofile, saved into:
  profiles/<pseudoprofile_id>/visuals/

This script is updated to be compatible with the NEW observation-model output schema
from 01_construct_observation_model.py (dense relevance grids + sparse edges for P->C, P->P, C->C).

Key compatibility updates
-------------------------
1) Uses the new dense relevance grids when available:
   - predictor_criterion_relevance            (P -> C)  [FULL grid]
   - predictor_predictor_relevance            (P -> P)  [FULL directed grid, i!=j]
   - criterion_criterion_relevance            (C -> C)  [FULL directed grid, i!=j]
   Each entry contains: relevance_score_0_1_comma5 like "0,12345".

2) Uses the new sparse edge lists when available:
   - edges     (P -> C)
   - edges_pp  (P -> P)
   - edges_cc  (C -> C)
   Each edge contains:
     expected_sign (NEW enum), relation_interpretation, lag_spec, estimated_relevance_0_1, tier, notes

3) expected_sign enum mapping is handled:
   - positive-ish:  monotonic_positive / threshold_positive / saturation_positive  -> "+"
   - negative-ish:  monotonic_negative / threshold_negative / saturation_negative  -> "-"
   - nonlinear-ish: inverted_U / U_shaped / oscillatory / interaction_moderation   -> "NL"
   - no-effect:     no_effect_expected                                            -> "0"
   - otherwise (proxy/confounded/unknown/insufficient/etc.)                        -> "U"

Outputs (per pseudoprofile)
---------------------------
P -> C (keeps your original 5 filenames for backward compatibility):
  1) nomothetic_estimation_bipartite_full.png
     - SPARSE network (edges list if present; otherwise falls back)
       Edge thickness ∝ score; color by expected_sign; optional score annotation.

  2) nomothetic_estimation_bubble_matrix.png
     - Dense dot-matrix from predictor_criterion_relevance (FULL grid).
       Optionally masks cells below --mask_dense_below (default 0.00 = show all).
       Row/col labels inside plot.

  3) nomothetic_estimation_heatmap.png
     - Dense heatmap from predictor_criterion_relevance (FULL grid), optional masking.

  4) nomothetic_estimation_signed_heatmap.png
     - Signed heatmap based on SPARSE edges only:
         blue=positive, red=negative, WHITE=everything else (incl. NL/U/0 or not selected).
       (Because sign is not defined for all dense pairs in the schema.)

  5) nomothetic_estimation_bipartite_numeric_only.png
     - Same as (1) but only numeric (always true for sparse edges; kept for naming symmetry).

Additional (recommended) visuals to represent EVERYTHING (P->P and C->C):
  6) predictor_predictor_relevance_heatmap.png
  7) predictor_predictor_signed_sparse_heatmap.png
  8) predictor_predictor_network_sparse.png

  9) criterion_criterion_relevance_heatmap.png
 10) criterion_criterion_signed_sparse_heatmap.png
 11) criterion_criterion_network_sparse.png

Usage
-----
python bipartite_model_visualization.py
python bipartite_model_visualization.py --run_dir /.../constructed_PC_models/runs/2026-01-18_13-01-41
python bipartite_model_visualization.py --annotate_numeric_scores
python bipartite_model_visualization.py --mask_dense_below 0.10
python bipartite_model_visualization.py --no_pp_cc

Notes
-----
- Dense grids are ideal for magnitude-only visuals (dot-matrix / heatmap).
- Signs/lag/interpretation exist only on sparse edges; therefore signed heatmaps are sparse-by-design.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import textwrap
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


# -----------------------------
# Aesthetics (muted / paper-ish)
# -----------------------------
COLOR_EDGE_POS = "#5C9E7B"        # muted green
COLOR_EDGE_NEG = "#C46A6A"        # muted red
COLOR_EDGE_NONLINEAR = "#8B84B8"  # muted purple
COLOR_EDGE_UNKNOWN = "#7A7A7A"    # muted dark grey
COLOR_EDGE_NOEFFECT = "#BDBDBD"   # grey for "0"/no-effect
COLOR_EDGE_NULL = "#D7D7D7"       # light grey (null-score edges fallback)

COLOR_NODE_PRED = "#274C77"       # muted blue
COLOR_NODE_CRIT = "#5A3E2B"       # muted brown

COLOR_NUMERIC_ONLY = "#2B2B2B"    # light black for numeric-only lines
COLOR_GRID = "#EFEFEF"
COLOR_TEXT_SOFT = "#333333"

FONT_NODE = 9
FONT_EDGE = 7
FONT_AXIS = 8
FONT_TITLE = 14

EDGE_W_MIN = 0.7
EDGE_W_MAX = 6.8

WRAP_NODE_CHARS = 36

NODE_SIZE_HIGH = 78
NODE_SIZE_MED = 62
NODE_SIZE_LOW = 50
NODE_SIZE_UNKNOWN = 56

LEFT_X = 0.0
RIGHT_X = 1.0

# If there are many null edges, keep labels shorter to avoid clutter
NULL_LABEL_DENSITY_THRESHOLD = 45

# Matrix label spaces (in "cell" units)
MATRIX_LEFT_LABEL_SPACE = 2.8
MATRIX_TOP_LABEL_SPACE = 2.0

# run folder pattern: YYYY-MM-DD_HH-MM-SS
RUN_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")


# -----------------------------
# Helpers
# -----------------------------
def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)

def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None

def _priority_key(p: str) -> int:
    p = (p or "").strip().upper()
    if p == "HIGH":
        return 0
    if p == "MED":
        return 1
    if p == "LOW":
        return 2
    return 3

def _node_size(priority: str) -> int:
    p = (priority or "").strip().upper()
    if p == "HIGH":
        return NODE_SIZE_HIGH
    if p == "MED":
        return NODE_SIZE_MED
    if p == "LOW":
        return NODE_SIZE_LOW
    return NODE_SIZE_UNKNOWN

def _is_number_in_0_1(v: Any) -> bool:
    x = _safe_float(v)
    return x is not None and 0.0 <= x <= 1.0

def _format_score_5(v: float) -> str:
    return f"{v:.5f}"

def _format_cell_score(v: float, n_cells: int, signed: bool = False) -> str:
    # adaptive precision to avoid unreadable grids
    if n_cells >= 360:
        s = f"{v:.2f}"
    elif n_cells >= 200:
        s = f"{v:.3f}"
    else:
        s = f"{v:.4f}"
    if signed and v > 0:
        return f"+{s}"
    return s

def _parse_comma_decimal_0_1(s: Any) -> Optional[float]:
    """
    Parses values like "0,70000" or "0.70000" into float.
    Returns None if parsing fails or outside [0,1].
    """
    if s is None:
        return None
    if isinstance(s, (int, float)):
        v = float(s)
        return v if 0.0 <= v <= 1.0 else None
    txt = str(s).strip()
    if not txt:
        return None
    txt = txt.replace(",", ".")
    try:
        v = float(txt)
        return v if 0.0 <= v <= 1.0 else None
    except Exception:
        return None

def _wrap_label(s: str, width: int = WRAP_NODE_CHARS) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    return "\n".join(textwrap.wrap(s, width=width, break_long_words=False, replace_whitespace=False))

def _wrap_label_max_lines(s: str, width: int, max_lines: int) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    lines = textwrap.wrap(s, width=width, break_long_words=False, replace_whitespace=False)
    if len(lines) <= max_lines:
        return "\n".join(lines)
    kept = lines[:max_lines]
    if kept[-1] and not kept[-1].endswith("…"):
        kept[-1] = (kept[-1][:-1] + "…") if len(kept[-1]) >= 2 else (kept[-1] + "…")
    return "\n".join(kept)

def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# -----------------------------
# Sign + interpretation mappings (NEW schema compatible)
# -----------------------------
_POS_SIGNS = {
    "+",
    "monotonic_positive",
    "threshold_positive",
    "saturation_positive",
}
_NEG_SIGNS = {
    "-",
    "monotonic_negative",
    "threshold_negative",
    "saturation_negative",
}
_NONLINEAR_SIGNS = {
    "nonlinear",
    "inverted_u",
    "u_shaped",
    "oscillatory",
    "interaction_moderation",
}
_NOEFFECT_SIGNS = {
    "0",
    "no_effect_expected",
}

def _normalize_sign(s: str) -> str:
    return (s or "").strip().lower()

def _sign_class(expected_sign: str) -> str:
    """
    Returns: "pos" | "neg" | "nonlinear" | "noeffect" | "unknown"
    """
    s = _normalize_sign(expected_sign)
    if s in _POS_SIGNS:
        return "pos"
    if s in _NEG_SIGNS:
        return "neg"
    if s in _NONLINEAR_SIGNS:
        return "nonlinear"
    if s in _NOEFFECT_SIGNS:
        return "noeffect"
    return "unknown"

def _abbr_sign(expected_sign: str) -> str:
    c = _sign_class(expected_sign)
    if c == "pos":
        return "+"
    if c == "neg":
        return "-"
    if c == "nonlinear":
        return "NL"
    if c == "noeffect":
        return "0"
    return "U"

def _edge_color_from_class(cls: str) -> str:
    if cls == "pos":
        return COLOR_EDGE_POS
    if cls == "neg":
        return COLOR_EDGE_NEG
    if cls == "nonlinear":
        return COLOR_EDGE_NONLINEAR
    if cls == "noeffect":
        return COLOR_EDGE_NOEFFECT
    return COLOR_EDGE_UNKNOWN

def _edge_color(expected_sign: str, score: Optional[float]) -> str:
    if score is None:
        return COLOR_EDGE_NULL
    return _edge_color_from_class(_sign_class(expected_sign))

def _edge_width(score: Optional[float]) -> float:
    if score is None:
        return 1.2
    return EDGE_W_MIN + float(score) * (EDGE_W_MAX - EDGE_W_MIN)

# --- color/lightness modulation (weight-dependent "edge saturation") ---
def _hex_to_rgb01(h: str) -> Tuple[float, float, float]:
    h = (h or "").strip().lstrip("#")
    if len(h) != 6:
        return (0.0, 0.0, 0.0)
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    return (r, g, b)

def _rgb01_to_hex(rgb: Tuple[float, float, float]) -> str:
    r = int(max(0.0, min(1.0, rgb[0])) * 255.0)
    g = int(max(0.0, min(1.0, rgb[1])) * 255.0)
    b = int(max(0.0, min(1.0, rgb[2])) * 255.0)
    return f"#{r:02X}{g:02X}{b:02X}"

def _mix_with_white(hex_color: str, strength_0_1: float) -> str:
    """
    strength_0_1 = 0 -> white
    strength_0_1 = 1 -> original color
    """
    s = float(max(0.0, min(1.0, strength_0_1)))
    r, g, b = _hex_to_rgb01(hex_color)
    rr = (1.0 - s) * 1.0 + s * r
    gg = (1.0 - s) * 1.0 + s * g
    bb = (1.0 - s) * 1.0 + s * b
    return _rgb01_to_hex((rr, gg, bb))

def _edge_color_weighted(expected_sign: str, score: Optional[float]) -> str:
    base = _edge_color(expected_sign, score)
    if score is None:
        return base
    s = float(max(0.0, min(1.0, score)))
    strength = 0.18 + 0.82 * s
    return _mix_with_white(base, strength)

def _edge_alpha_weighted(score: Optional[float], *, a_min: float, a_max: float) -> float:
    if score is None:
        return a_min
    s = float(max(0.0, min(1.0, score)))
    return a_min + s * (a_max - a_min)

def _linestyle_for_relation_interpretation(rel: str) -> str:
    """
    Optional: encode 'relation_interpretation' via line style.
    Kept conservative to avoid clutter.
    """
    r = (rel or "").strip().lower()
    if not r:
        return "solid"
    if r in ("hypothesized_direct_causal", "hypothesized_indirect_or_mediated"):
        return "solid"
    if r in ("proxy_or_indicator",):
        return "dotted"
    if r in ("confounding_likely", "reverse_causality_plausible", "measurement_artifact_possible"):
        return "dashed"
    if r in ("temporal_association_noncausal_possible", "bidirectional_feedback_possible", "unknown", "insufficient_knowledge"):
        return "dashed"
    return "solid"


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class NodeInfo:
    var_id: str
    label: str
    priority: str

@dataclass
class EffectInfo:
    expected_sign: str
    lag_spec: str  # "min-maxh"

@dataclass
class SparseEdge:
    from_id: str
    to_id: str
    score: Optional[float]          # numeric in [0,1] or None
    expected_sign: str              # NEW enum supported
    lag_spec: str                   # "0-6h", ...
    relation_interpretation: str
    tier: str
    notes: str
    edge_kind: str                  # "pc" | "pp" | "cc"


# -----------------------------
# Extract nodes
# -----------------------------
def _extract_nodes(model: Dict[str, Any]) -> Tuple[List[NodeInfo], List[NodeInfo]]:
    criteria: List[NodeInfo] = []
    predictors: List[NodeInfo] = []

    for c in (model.get("criteria_variables") or []):
        criteria.append(
            NodeInfo(
                var_id=_safe_str(c.get("var_id", "")).strip(),
                label=_safe_str(c.get("label", "")).strip(),
                priority=_safe_str(c.get("include_priority", "")).strip(),
            )
        )

    for p in (model.get("predictor_variables") or []):
        predictors.append(
            NodeInfo(
                var_id=_safe_str(p.get("var_id", "")).strip(),
                label=_safe_str(p.get("label", "")).strip(),
                priority=_safe_str(p.get("include_priority", "")).strip(),
            )
        )

    criteria = [n for n in criteria if n.var_id]
    predictors = [n for n in predictors if n.var_id]

    criteria.sort(key=lambda n: (_priority_key(n.priority), n.var_id))
    predictors.sort(key=lambda n: (_priority_key(n.priority), n.var_id))

    return predictors, criteria


# -----------------------------
# Expected effects (predictor->criterion) fallback
# -----------------------------
def _extract_expected_effects_pc(model: Dict[str, Any]) -> Dict[Tuple[str, str], EffectInfo]:
    out: Dict[Tuple[str, str], EffectInfo] = {}
    for p in (model.get("predictor_variables") or []):
        pid = _safe_str(p.get("var_id", "")).strip()
        if not pid:
            continue
        for eff in (p.get("expected_effects") or []):
            cid = _safe_str(eff.get("criterion_var_id", "")).strip()
            if not cid:
                continue
            sign = _safe_str(eff.get("expected_sign", "unknown")).strip() or "unknown"
            try:
                lag_min = int(eff.get("typical_lag_hours_min", 0) or 0)
                lag_max = int(eff.get("typical_lag_hours_max", 0) or 0)
                if lag_max < lag_min:
                    lag_min, lag_max = lag_max, lag_min
                lag_spec = f"{lag_min}-{lag_max}h"
            except Exception:
                lag_spec = ""
            out[(pid, cid)] = EffectInfo(expected_sign=sign, lag_spec=lag_spec)
    return out


# -----------------------------
# Dense relevance extraction (NEW schema)
# -----------------------------
def _extract_dense_lookup_pc(model: Dict[str, Any]) -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    arr = model.get("predictor_criterion_relevance") or []
    for it in arr:
        if not isinstance(it, dict):
            continue
        pid = _safe_str(it.get("predictor_var_id", "")).strip()
        cid = _safe_str(it.get("criterion_var_id", "")).strip()
        sc = _parse_comma_decimal_0_1(it.get("relevance_score_0_1_comma5", None))
        if pid and cid and sc is not None:
            out[(pid, cid)] = float(sc)
    return out

def _extract_dense_lookup_pp(model: Dict[str, Any]) -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    arr = model.get("predictor_predictor_relevance") or []
    for it in arr:
        if not isinstance(it, dict):
            continue
        fr = _safe_str(it.get("from_predictor_var_id", "")).strip()
        to = _safe_str(it.get("to_predictor_var_id", "")).strip()
        sc = _parse_comma_decimal_0_1(it.get("relevance_score_0_1_comma5", None))
        if fr and to and fr != to and sc is not None:
            out[(fr, to)] = float(sc)
    return out

def _extract_dense_lookup_cc(model: Dict[str, Any]) -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    arr = model.get("criterion_criterion_relevance") or []
    for it in arr:
        if not isinstance(it, dict):
            continue
        fr = _safe_str(it.get("from_criterion_var_id", "")).strip()
        to = _safe_str(it.get("to_criterion_var_id", "")).strip()
        sc = _parse_comma_decimal_0_1(it.get("relevance_score_0_1_comma5", None))
        if fr and to and fr != to and sc is not None:
            out[(fr, to)] = float(sc)
    return out


# -----------------------------
# Sparse edges extraction (NEW schema)
# -----------------------------
def _extract_sparse_edges_pc(model: Dict[str, Any]) -> List[SparseEdge]:
    out: List[SparseEdge] = []
    for e in (model.get("edges") or []):
        if not isinstance(e, dict):
            continue
        fr = _safe_str(e.get("from_predictor_var_id", "")).strip()
        to = _safe_str(e.get("to_criterion_var_id", "")).strip()
        if not fr or not to:
            continue
        score = e.get("estimated_relevance_0_1", None)
        score_f = float(score) if isinstance(score, (int, float)) and 0.0 <= float(score) <= 1.0 else None
        out.append(
            SparseEdge(
                from_id=fr,
                to_id=to,
                score=score_f,
                expected_sign=_safe_str(e.get("expected_sign", "unknown")).strip() or "unknown",
                lag_spec=_safe_str(e.get("lag_spec", "")).strip(),
                relation_interpretation=_safe_str(e.get("relation_interpretation", "")).strip(),
                tier=_safe_str(e.get("estimated_relevance_tier", "")).strip(),
                notes=_safe_str(e.get("notes", "")).strip(),
                edge_kind="pc",
            )
        )
    return out

def _extract_sparse_edges_pp(model: Dict[str, Any]) -> List[SparseEdge]:
    out: List[SparseEdge] = []
    for e in (model.get("edges_pp") or []):
        if not isinstance(e, dict):
            continue
        fr = _safe_str(e.get("from_predictor_var_id", "")).strip()
        to = _safe_str(e.get("to_predictor_var_id", "")).strip()
        if not fr or not to or fr == to:
            continue
        score = e.get("estimated_relevance_0_1", None)
        score_f = float(score) if isinstance(score, (int, float)) and 0.0 <= float(score) <= 1.0 else None
        out.append(
            SparseEdge(
                from_id=fr,
                to_id=to,
                score=score_f,
                expected_sign=_safe_str(e.get("expected_sign", "unknown")).strip() or "unknown",
                lag_spec=_safe_str(e.get("lag_spec", "")).strip(),
                relation_interpretation=_safe_str(e.get("relation_interpretation", "")).strip(),
                tier=_safe_str(e.get("estimated_relevance_tier", "")).strip(),
                notes=_safe_str(e.get("notes", "")).strip(),
                edge_kind="pp",
            )
        )
    return out

def _extract_sparse_edges_cc(model: Dict[str, Any]) -> List[SparseEdge]:
    out: List[SparseEdge] = []
    for e in (model.get("edges_cc") or []):
        if not isinstance(e, dict):
            continue
        fr = _safe_str(e.get("from_criterion_var_id", "")).strip()
        to = _safe_str(e.get("to_criterion_var_id", "")).strip()
        if not fr or not to or fr == to:
            continue
        score = e.get("estimated_relevance_0_1", None)
        score_f = float(score) if isinstance(score, (int, float)) and 0.0 <= float(score) <= 1.0 else None
        out.append(
            SparseEdge(
                from_id=fr,
                to_id=to,
                score=score_f,
                expected_sign=_safe_str(e.get("expected_sign", "unknown")).strip() or "unknown",
                lag_spec=_safe_str(e.get("lag_spec", "")).strip(),
                relation_interpretation=_safe_str(e.get("relation_interpretation", "")).strip(),
                tier=_safe_str(e.get("estimated_relevance_tier", "")).strip(),
                notes=_safe_str(e.get("notes", "")).strip(),
                edge_kind="cc",
            )
        )
    return out


# -----------------------------
# Matrix builders
# -----------------------------
def _matrix_from_lookup(
    row_nodes: List[NodeInfo],
    col_nodes: List[NodeInfo],
    lookup: Dict[Tuple[str, str], float],
    *,
    exclude_diag: bool = False,
) -> np.ndarray:
    r_index = {n.var_id: i for i, n in enumerate(row_nodes)}
    c_index = {n.var_id: j for j, n in enumerate(col_nodes)}
    mat = np.full((len(row_nodes), len(col_nodes)), np.nan, dtype=float)

    for (a, b), v in lookup.items():
        if a not in r_index or b not in c_index:
            continue
        i = r_index[a]
        j = c_index[b]
        if exclude_diag and i == j:
            continue
        mat[i, j] = float(v)

    if exclude_diag and len(row_nodes) == len(col_nodes):
        for k in range(len(row_nodes)):
            mat[k, k] = np.nan
    return mat

def _sign_matrix_from_sparse_edges(
    row_nodes: List[NodeInfo],
    col_nodes: List[NodeInfo],
    edges: List[SparseEdge],
    *,
    exclude_diag: bool = False,
) -> np.ndarray:
    r_index = {n.var_id: i for i, n in enumerate(row_nodes)}
    c_index = {n.var_id: j for j, n in enumerate(col_nodes)}
    mat = np.full((len(row_nodes), len(col_nodes)), "", dtype=object)

    for e in edges:
        if e.from_id not in r_index or e.to_id not in c_index:
            continue
        i = r_index[e.from_id]
        j = c_index[e.to_id]
        if exclude_diag and i == j:
            continue
        mat[i, j] = _abbr_sign(e.expected_sign)

    if exclude_diag and len(row_nodes) == len(col_nodes):
        for k in range(len(row_nodes)):
            mat[k, k] = ""
    return mat


# -----------------------------
# Matrix drawing helpers (labels INSIDE plot)
# -----------------------------
def _draw_matrix_frame_and_labels(
    ax: plt.Axes,
    row_nodes: List[NodeInfo],
    col_nodes: List[NodeInfo],
    *,
    row_label_prefix: str = "",
    col_label_prefix: str = "",
    left_space: float = MATRIX_LEFT_LABEL_SPACE,
    top_space: float = MATRIX_TOP_LABEL_SPACE,
    label_wrap_w_row: int = 26,
    label_wrap_w_col: int = 20,
) -> None:
    n_r = len(row_nodes)
    n_c = len(col_nodes)

    ax.set_xlim(-left_space, n_c - 0.5)
    ax.set_ylim(n_r - 0.5, -top_space)

    # Cell grid
    for j in range(n_c + 1):
        x = j - 0.5
        ax.plot([x, x], [-0.5, n_r - 0.5], color=COLOR_GRID, linewidth=0.8, zorder=5)
    for i in range(n_r + 1):
        y = i - 0.5
        ax.plot([-0.5, n_c - 0.5], [y, y], color=COLOR_GRID, linewidth=0.8, zorder=5)

    # Row labels
    for i, n in enumerate(row_nodes):
        lab = f"{row_label_prefix}{n.var_id}"
        if n.label:
            lab = f"{row_label_prefix}{n.var_id}\n{_wrap_label_max_lines(n.label, width=label_wrap_w_row, max_lines=2)}"
        ax.text(
            -left_space + 0.10,
            i,
            lab,
            ha="left",
            va="center",
            fontsize=FONT_AXIS,
            color=COLOR_TEXT_SOFT,
            zorder=10,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85),
        )

    # Column labels (top band)
    header_y = -top_space + 0.20
    for j, n in enumerate(col_nodes):
        lab = f"{col_label_prefix}{n.var_id}"
        if n.label:
            lab = f"{col_label_prefix}{n.var_id}\n{_wrap_label_max_lines(n.label, width=label_wrap_w_col, max_lines=2)}"
        ax.text(
            j,
            header_y,
            lab,
            ha="center",
            va="top",
            fontsize=FONT_AXIS,
            color=COLOR_TEXT_SOFT,
            zorder=10,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85),
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


# -----------------------------
# Layout helpers
# -----------------------------
def _positions_bipartite(predictors: List[NodeInfo], criteria: List[NodeInfo]) -> Dict[str, Tuple[float, float]]:
    pos: Dict[str, Tuple[float, float]] = {}
    n_left = max(1, len(predictors))
    n_right = max(1, len(criteria))
    n = max(n_left, n_right)

    def y_for_index(i: int, total: int) -> float:
        if total <= 1:
            return 0.0
        return (n - 1) - (i * (n - 1) / (total - 1))

    for i, node in enumerate(predictors):
        pos[node.var_id] = (LEFT_X, y_for_index(i, n_left))
    for i, node in enumerate(criteria):
        pos[node.var_id] = (RIGHT_X, y_for_index(i, n_right))
    return pos

def _positions_circle(nodes: List[NodeInfo]) -> Dict[str, Tuple[float, float]]:
    pos: Dict[str, Tuple[float, float]] = {}
    n = max(1, len(nodes))
    for i, node in enumerate(nodes):
        ang = 2.0 * math.pi * (i / n)
        pos[node.var_id] = (math.cos(ang), math.sin(ang))
    return pos


# -----------------------------
# 1) P->C Bipartite sparse network
# -----------------------------
def _draw_bipartite_sparse_pc(
    model: Dict[str, Any],
    output_png: str,
    *,
    title: str,
    annotate_numeric_scores: bool,
) -> None:
    predictors, criteria = _extract_nodes(model)
    pos = _positions_bipartite(predictors, criteria)

    pred_ids = {n.var_id for n in predictors}
    crit_ids = {n.var_id for n in criteria}

    sparse_edges = _extract_sparse_edges_pc(model)
    sparse_edges = [e for e in sparse_edges if e.from_id in pred_ids and e.to_id in crit_ids]

    # Fallback: if sparse edges missing, try to reconstruct from expected_effects (null-score) + dense lookup if any
    if not sparse_edges:
        exp = _extract_expected_effects_pc(model)
        dense = _extract_dense_lookup_pc(model)
        for (pid, cid), ei in exp.items():
            if pid in pred_ids and cid in crit_ids:
                score = dense.get((pid, cid), None)
                sparse_edges.append(
                    SparseEdge(
                        from_id=pid,
                        to_id=cid,
                        score=score,
                        expected_sign=ei.expected_sign,
                        lag_spec=ei.lag_spec,
                        relation_interpretation="",
                        tier="",
                        notes="(fallback from expected_effects + dense grid)",
                        edge_kind="pc",
                    )
                )

    n = max(1, len(predictors), len(criteria))
    fig_w = 16
    fig_h = max(7.0, min(0.55 * n + 3.0, 30.0))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    numeric_edges = [e for e in sparse_edges if e.score is not None]
    null_edges = [e for e in sparse_edges if e.score is None]
    dense_null = len(null_edges) > NULL_LABEL_DENSITY_THRESHOLD

    # Draw null-score edges first (rare in new schema, but kept robust)
    for e in null_edges:
        x1, y1 = pos[e.from_id]
        x2, y2 = pos[e.to_id]
        dy = (y2 - y1)
        rad = max(-0.28, min(0.28, 0.11 * (dy / max(1.0, (n - 1)))))

        patch = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-",
            mutation_scale=10,
            connectionstyle=f"arc3,rad={rad}",
            linewidth=_edge_width(None),
            color=COLOR_EDGE_NULL,
            alpha=0.70,
            zorder=1,
        )
        ax.add_patch(patch)

        sign_abbr = _abbr_sign(e.expected_sign)
        label = sign_abbr if dense_null else f"{sign_abbr} | {e.lag_spec}".strip(" |")
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        off = 0.16 * rad
        ax.text(
            mx,
            my + off,
            label,
            fontsize=FONT_EDGE,
            ha="center",
            va="center",
            color="#505050",
            zorder=2,
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.70),
        )

    # Draw numeric sparse edges
    for e in numeric_edges:
        x1, y1 = pos[e.from_id]
        x2, y2 = pos[e.to_id]
        dy = (y2 - y1)
        rad = max(-0.34, min(0.34, 0.13 * (dy / max(1.0, (n - 1)))))

        patch = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-",
            mutation_scale=10,
            connectionstyle=f"arc3,rad={rad}",
            linewidth=_edge_width(e.score),
            color=_edge_color_weighted(e.expected_sign, e.score),
            alpha=_edge_alpha_weighted(e.score, a_min=0.30, a_max=0.95),
            linestyle=_linestyle_for_relation_interpretation(e.relation_interpretation),
            zorder=3,
        )
        ax.add_patch(patch)

        if annotate_numeric_scores and e.score is not None:
            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            off = 0.16 * rad
            ax.text(
                mx,
                my + off,
                _format_score_5(float(e.score)),
                fontsize=FONT_EDGE,
                ha="center",
                va="center",
                color="#2F2F2F",
                zorder=4,
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.75),
            )

    # Nodes
    pred_x = [pos[nn.var_id][0] for nn in predictors]
    pred_y = [pos[nn.var_id][1] for nn in predictors]
    pred_s = [_node_size(nn.priority) for nn in predictors]

    crit_x = [pos[nn.var_id][0] for nn in criteria]
    crit_y = [pos[nn.var_id][1] for nn in criteria]
    crit_s = [_node_size(nn.priority) for nn in criteria]

    ax.scatter(pred_x, pred_y, s=pred_s, color=COLOR_NODE_PRED, zorder=5)
    ax.scatter(crit_x, crit_y, s=crit_s, color=COLOR_NODE_CRIT, zorder=5)

    # Labels (boxed)
    for nn in predictors:
        x, y = pos[nn.var_id]
        text = f"{nn.var_id}: {_wrap_label(nn.label)}"
        ax.annotate(
            text,
            (x, y),
            xytext=(-10, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=FONT_NODE,
            color="black",
            bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="#E6E6E6", alpha=0.98),
            zorder=6,
        )

    for nn in criteria:
        x, y = pos[nn.var_id]
        text = f"{nn.var_id}: {_wrap_label(nn.label)}"
        ax.annotate(
            text,
            (x, y),
            xytext=(10, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=FONT_NODE,
            color="black",
            bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="#E6E6E6", alpha=0.98),
            zorder=6,
        )

    ax.set_title(title, fontsize=FONT_TITLE, pad=16)

    # Legend (compact)
    legend_y = -1.10
    ax.text(LEFT_X, legend_y, "Predictors", ha="center", va="center", fontsize=10, color=COLOR_NODE_PRED)
    ax.text(RIGHT_X, legend_y, "Criteria", ha="center", va="center", fontsize=10, color=COLOR_NODE_CRIT)

    x0, x1 = 0.40, 0.55
    ax.plot([x0, x1], [legend_y, legend_y], color=COLOR_EDGE_POS, linewidth=2.8)
    ax.text(x1 + 0.01, legend_y, "+ (positive family)", ha="left", va="center", fontsize=9, color="#444444")

    ax.plot([x0, x1], [legend_y - 0.35, legend_y - 0.35], color=COLOR_EDGE_NEG, linewidth=2.8)
    ax.text(x1 + 0.01, legend_y - 0.35, "- (negative family)", ha="left", va="center", fontsize=9, color="#444444")

    ax.plot([x0, x1], [legend_y - 0.70, legend_y - 0.70], color=COLOR_EDGE_NONLINEAR, linewidth=2.8)
    ax.text(x1 + 0.01, legend_y - 0.70, "NL (nonlinear family)", ha="left", va="center", fontsize=9, color="#444444")

    ax.plot([x0, x1], [legend_y - 1.05, legend_y - 1.05], color=COLOR_EDGE_UNKNOWN, linewidth=2.8)
    ax.text(x1 + 0.01, legend_y - 1.05, "U (unknown/other)", ha="left", va="center", fontsize=9, color="#444444")

    ax.plot([x0, x1], [legend_y - 1.40, legend_y - 1.40], color=COLOR_EDGE_NOEFFECT, linewidth=2.8)
    ax.text(x1 + 0.01, legend_y - 1.40, "0 (no-effect)", ha="left", va="center", fontsize=9, color="#444444")

    ax.set_xlim(-0.40, 1.40)
    ax.set_ylim(-1.95, max(1.0, n - 1) + 1.0)
    ax.axis("off")

    _ensure_dir(os.path.dirname(output_png))
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# 2) Dense dot matrix (P->C)
# -----------------------------
def _draw_dense_dot_matrix(
    row_nodes: List[NodeInfo],
    col_nodes: List[NodeInfo],
    score_mat: np.ndarray,
    *,
    output_png: str,
    title: str,
    mask_below: float,
) -> None:
    n_r = len(row_nodes)
    n_c = len(col_nodes)
    n_cells = max(1, n_r * n_c)

    fig_w = max(11.0, min(0.72 * n_c + 6.5, 26.0))
    fig_h = max(7.0, min(0.52 * n_r + 5.0, 22.0))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # White cell backgrounds
    for i in range(n_r):
        for j in range(n_c):
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1.0, 1.0, facecolor="white", edgecolor="none", zorder=0))

    xs: List[int] = []
    ys: List[int] = []
    sizes: List[float] = []
    colors: List[str] = []

    for i in range(n_r):
        for j in range(n_c):
            sc = score_mat[i, j]
            if np.isnan(sc):
                continue
            if float(sc) < float(mask_below):
                continue
            s = float(sc)
            xs.append(j)
            ys.append(i)
            sizes.append(45 + 720 * (s ** 0.85))
            # magnitude-only dots -> use a perceptually uniform colormap
            # (keep neutral edge colors, because sign is not defined for dense grid)
            colors.append(_mix_with_white("#2B2B2B", 0.10 + 0.90 * s))

    if xs:
        ax.scatter(xs, ys, s=sizes, c=colors, alpha=0.78, edgecolors="white", linewidths=0.8, zorder=3)

        # In-dot labels only when not too dense and dot is large enough
        if n_cells <= 220:
            for x, y, sc in zip(xs, ys, [score_mat[i, j] for i, j in zip(ys, xs)]):
                s = float(sc)
                if s < max(mask_below, 0.25):
                    continue
                txt = _format_cell_score(s, n_cells=n_cells, signed=False)
                ax.text(
                    x, y, txt,
                    ha="center", va="center",
                    fontsize=7,
                    color="#111111",
                    zorder=4,
                    bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.60),
                )

    _draw_matrix_frame_and_labels(ax, row_nodes, col_nodes)
    ax.set_title(title, fontsize=FONT_TITLE, pad=12)

    # Note about masking
    ax.text(
        -0.45, -1.20,
        f"Dot size ∝ relevance\nMask: show only scores ≥ {mask_below:.2f}",
        ha="left", va="top",
        fontsize=9,
        color=COLOR_TEXT_SOFT,
        bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="none", alpha=0.85),
        zorder=30,
    )

    _ensure_dir(os.path.dirname(output_png))
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# 3) Dense heatmap (generic)
# -----------------------------
def _draw_dense_heatmap(
    row_nodes: List[NodeInfo],
    col_nodes: List[NodeInfo],
    score_mat: np.ndarray,
    *,
    output_png: str,
    title: str,
    mask_below: float,
    add_cell_labels_if_small: bool = True,
) -> None:
    n_r = len(row_nodes)
    n_c = len(col_nodes)
    n_cells = max(1, n_r * n_c)

    fig_w = max(11.0, min(0.78 * n_c + 6.8, 28.0))
    fig_h = max(7.0, min(0.56 * n_r + 5.2, 24.0))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    mat = np.array(score_mat, dtype=float, copy=True)
    # mask below threshold + NaNs -> WHITE
    for i in range(n_r):
        for j in range(n_c):
            if np.isnan(mat[i, j]) or float(mat[i, j]) < float(mask_below):
                mat[i, j] = np.nan

    cmap = copy.copy(plt.get_cmap("cividis"))
    cmap.set_bad(color="white")
    masked = np.ma.masked_invalid(mat)

    im = ax.imshow(
        masked,
        interpolation="nearest",
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        cmap=cmap,
        origin="upper",
        extent=(-0.5, n_c - 0.5, n_r - 0.5, -0.5),
        zorder=1,
    )

    if add_cell_labels_if_small and n_cells <= 260:
        for i in range(n_r):
            for j in range(n_c):
                sc = mat[i, j]
                if np.isnan(sc):
                    continue
                txt = _format_cell_score(float(sc), n_cells=n_cells, signed=False)
                ax.text(
                    j, i, txt,
                    ha="center", va="center",
                    fontsize=7,
                    color="#111111",
                    zorder=4,
                    bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.60),
                )

    _draw_matrix_frame_and_labels(ax, row_nodes, col_nodes)
    ax.set_title(title, fontsize=FONT_TITLE, pad=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(f"Relevance (0–1), masked below {mask_below:.2f}", fontsize=10)

    _ensure_dir(os.path.dirname(output_png))
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# 4) Signed sparse heatmap (generic)
# -----------------------------
def _draw_signed_sparse_heatmap(
    row_nodes: List[NodeInfo],
    col_nodes: List[NodeInfo],
    dense_scores: np.ndarray,
    sparse_signs: np.ndarray,
    *,
    output_png: str,
    title: str,
    exclude_diag: bool,
) -> None:
    n_r = len(row_nodes)
    n_c = len(col_nodes)
    n_cells = max(1, n_r * n_c)

    signed = np.full((n_r, n_c), np.nan, dtype=float)
    for i in range(n_r):
        for j in range(n_c):
            if exclude_diag and n_r == n_c and i == j:
                continue
            sc = dense_scores[i, j]
            if np.isnan(sc):
                continue
            ab = (sparse_signs[i, j] or "").strip()
            if ab == "+":
                signed[i, j] = float(sc)
            elif ab == "-":
                signed[i, j] = -float(sc)
            else:
                signed[i, j] = np.nan  # NL/U/0 or not selected -> white

    fig_w = max(11.0, min(0.78 * n_c + 6.8, 28.0))
    fig_h = max(7.0, min(0.56 * n_r + 5.2, 24.0))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = copy.copy(plt.get_cmap("bwr_r"))  # red at -1, blue at +1
    cmap.set_bad(color="white")
    masked = np.ma.masked_invalid(signed)

    im = ax.imshow(
        masked,
        interpolation="nearest",
        aspect="auto",
        vmin=-1.0,
        vmax=1.0,
        cmap=cmap,
        origin="upper",
        extent=(-0.5, n_c - 0.5, n_r - 0.5, -0.5),
        zorder=1,
    )

    if n_cells <= 260:
        for i in range(n_r):
            for j in range(n_c):
                v = signed[i, j]
                if np.isnan(v):
                    continue
                txt = _format_cell_score(float(v), n_cells=n_cells, signed=True)
                ax.text(
                    j, i, txt,
                    ha="center", va="center",
                    fontsize=7,
                    color="#111111",
                    zorder=4,
                    bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.60),
                )

    _draw_matrix_frame_and_labels(ax, row_nodes, col_nodes)
    ax.set_title(title, fontsize=FONT_TITLE, pad=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Signed relevance (SPARSE edges only): red = -, blue = +", fontsize=10)

    ax.text(
        -0.45, -1.20,
        "WHITE = not selected as sparse edge\n(or NL/U/0 sign family)",
        ha="left", va="top",
        fontsize=9,
        color=COLOR_TEXT_SOFT,
        bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="none", alpha=0.85),
        zorder=30,
    )

    _ensure_dir(os.path.dirname(output_png))
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Within-set sparse network (P->P or C->C)
# -----------------------------
def _draw_within_set_sparse_network(
    nodes: List[NodeInfo],
    edges: List[SparseEdge],
    *,
    output_png: str,
    title: str,
) -> None:
    ids = {n.var_id for n in nodes}
    edges = [e for e in edges if e.from_id in ids and e.to_id in ids and e.from_id != e.to_id]

    pos = _positions_circle(nodes)

    n = max(1, len(nodes))
    fig_w = max(9.0, min(0.75 * n + 8.0, 18.0))
    fig_h = fig_w
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # edges first
    for e in edges:
        x1, y1 = pos[e.from_id]
        x2, y2 = pos[e.to_id]
        # curvature based on direction to reduce overlap a bit
        rad = 0.18 if (x1 * y2 - y1 * x2) >= 0 else -0.18

        patch = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=10,
            connectionstyle=f"arc3,rad={rad}",
            linewidth=_edge_width(e.score),
            color=_edge_color_weighted(e.expected_sign, e.score),
            alpha=_edge_alpha_weighted(e.score, a_min=0.18, a_max=0.85),
            linestyle=_linestyle_for_relation_interpretation(e.relation_interpretation),
            zorder=2,
        )
        ax.add_patch(patch)

    # nodes
    xs = [pos[nn.var_id][0] for nn in nodes]
    ys = [pos[nn.var_id][1] for nn in nodes]
    ss = [_node_size(nn.priority) for nn in nodes]
    ax.scatter(xs, ys, s=ss, color="#4B4B4B", zorder=3)

    # labels
    for nn in nodes:
        x, y = pos[nn.var_id]
        ax.annotate(
            f"{nn.var_id}\n{_wrap_label_max_lines(nn.label, width=28, max_lines=2)}",
            (x, y),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#E6E6E6", alpha=0.96),
            zorder=4,
        )

    ax.set_title(title, fontsize=FONT_TITLE, pad=14)

    # legend
    lx, ly = -1.35, -1.25
    legend = [
        ("+ (positive family)", COLOR_EDGE_POS),
        ("- (negative family)", COLOR_EDGE_NEG),
        ("NL (nonlinear family)", COLOR_EDGE_NONLINEAR),
        ("0 (no-effect)", COLOR_EDGE_NOEFFECT),
        ("U (unknown/other)", COLOR_EDGE_UNKNOWN),
    ]
    for k, (lab, col) in enumerate(legend):
        y = ly + k * 0.16
        ax.add_patch(Rectangle((lx, y - 0.05), 0.12, 0.10, facecolor=col, edgecolor="none", zorder=10))
        ax.text(lx + 0.16, y, lab, ha="left", va="center", fontsize=9, color=COLOR_TEXT_SOFT, zorder=11,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.80))

    ax.set_xlim(-1.65, 1.65)
    ax.set_ylim(-1.55, 1.65)
    ax.axis("off")

    _ensure_dir(os.path.dirname(output_png))
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Run directory discovery
# -----------------------------
def _parse_run_dir_datetime(name: str) -> Optional[datetime]:
    if not RUN_DIR_RE.match(name):
        return None
    try:
        return datetime.strptime(name, "%Y-%m-%d_%H-%M-%S")
    except Exception:
        return None

def _pick_latest_run_dir(runs_root: str) -> Optional[str]:
    if not os.path.isdir(runs_root):
        return None

    candidates: List[Tuple[int, str, float]] = []
    for name in os.listdir(runs_root):
        p = os.path.join(runs_root, name)
        if not os.path.isdir(p):
            continue
        dt = _parse_run_dir_datetime(name)
        if dt is not None:
            candidates.append((1, p, dt.timestamp()))
        else:
            try:
                candidates.append((0, p, os.path.getmtime(p)))
            except Exception:
                continue

    if not candidates:
        return None

    candidates.sort(key=lambda t: (t[0], t[2]))
    return candidates[-1][1]

def _find_profile_jsons(run_dir: str) -> List[Tuple[str, str]]:
    profiles_dir = os.path.join(run_dir, "profiles")
    if not os.path.isdir(profiles_dir):
        return []
    out: List[Tuple[str, str]] = []
    for name in sorted(os.listdir(profiles_dir)):
        pdir = os.path.join(profiles_dir, name)
        if not os.path.isdir(pdir):
            continue
        jpath = os.path.join(pdir, "llm_observation_model_final.json")
        if os.path.exists(jpath):
            out.append((name, jpath))
    return out


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create publication-oriented visuals per pseudoprofile (latest run by default)."
    )

    parser.add_argument(
        "--runs_root",
        default="/Users/stijnvanseveren/PythonProjects/MASTERPROEF/Evaluation/03_construction_initial_observation_model/constructed_PC_models/runs",
        help="Root directory containing run folders. Latest run is selected by default.",
    )
    parser.add_argument(
        "--run_dir",
        default=None,
        help="Optional explicit run directory (overrides latest-run selection).",
    )
    parser.add_argument(
        "--annotate_numeric_scores",
        action="store_true",
        help="If set, annotate numeric scores on P->C bipartite sparse edges (can clutter).",
    )
    parser.add_argument(
        "--mask_dense_below",
        type=float,
        default=0.00,
        help="Mask dense grid cells with relevance < this threshold (render as white). "
             "Recommended: 0.10 to hide 'none/very_low' per your calibration anchors.",
    )
    parser.add_argument(
        "--no_pp_cc",
        action="store_true",
        help="If set, do NOT generate the additional P->P and C->C visuals.",
    )

    args = parser.parse_args()

    run_dir = args.run_dir or _pick_latest_run_dir(args.runs_root)
    if not run_dir or not os.path.isdir(run_dir):
        print(f"No valid run_dir found. runs_root={args.runs_root} run_dir={args.run_dir}")
        return 2

    pairs = _find_profile_jsons(run_dir)
    if not pairs:
        print(f"No profiles found under: {run_dir}")
        return 2

    print(f"Run dir: {run_dir}")
    print(f"Found {len(pairs)} profile(s)")

    for pseudoprofile_id, jpath in pairs:
        try:
            model = _load_json(jpath)
            pid = _safe_str(model.get("pseudoprofile_id", pseudoprofile_id)).strip() or pseudoprofile_id

            profile_dir = os.path.dirname(jpath)
            visuals_dir = os.path.join(profile_dir, "visuals")
            _ensure_dir(visuals_dir)

            predictors, criteria = _extract_nodes(model)

            # ---------- P -> C ----------
            # Dense matrices (from NEW schema)
            dense_pc = _extract_dense_lookup_pc(model)
            pc_mat = _matrix_from_lookup(predictors, criteria, dense_pc, exclude_diag=False)

            # Sparse edges (for network + signed sparse heatmap)
            sparse_pc = _extract_sparse_edges_pc(model)
            pc_signs = _sign_matrix_from_sparse_edges(predictors, criteria, sparse_pc, exclude_diag=False)

            # 1) Bipartite sparse network (keeps old filename)
            out1 = os.path.join(visuals_dir, "nomothetic_estimation_bipartite_full.png")
            title1 = (
                f"Nomothetic estimation (P→C sparse network) | {pid}\n"
                f"P={len(predictors)}, C={len(criteria)}, sparse_edges={len(sparse_pc)}"
            )
            _draw_bipartite_sparse_pc(
                model=model,
                output_png=out1,
                title=title1,
                annotate_numeric_scores=bool(args.annotate_numeric_scores),
            )

            # 2) Dense dot matrix (keeps old filename)
            out2 = os.path.join(visuals_dir, "nomothetic_estimation_bubble_matrix.png")
            title2 = (
                f"Nomothetic estimation (P→C dense dot-matrix) | {pid}\n"
                f"Dot size ∝ dense relevance; white = masked (<{args.mask_dense_below:.2f})"
            )
            _draw_dense_dot_matrix(
                predictors,
                criteria,
                pc_mat,
                output_png=out2,
                title=title2,
                mask_below=float(args.mask_dense_below),
            )

            # 3) Dense heatmap (keeps old filename)
            out3 = os.path.join(visuals_dir, "nomothetic_estimation_heatmap.png")
            title3 = (
                f"Nomothetic estimation (P→C dense heatmap) | {pid}\n"
                f"WHITE = masked (<{args.mask_dense_below:.2f})"
            )
            _draw_dense_heatmap(
                predictors,
                criteria,
                pc_mat,
                output_png=out3,
                title=title3,
                mask_below=float(args.mask_dense_below),
            )

            # 4) Signed sparse heatmap (keeps old filename)
            out4 = os.path.join(visuals_dir, "nomothetic_estimation_signed_heatmap.png")
            title4 = (
                f"Nomothetic estimation (P→C signed; sparse edges only) | {pid}\n"
                f"BLUE=positive, RED=negative, WHITE=not selected / NL/U/0"
            )
            _draw_signed_sparse_heatmap(
                predictors,
                criteria,
                pc_mat,
                pc_signs,
                output_png=out4,
                title=title4,
                exclude_diag=False,
            )

            # 5) Bipartite numeric-only (keeps old filename)
            # In new schema sparse edges are numeric; keep this as a stylistic variant.
            out5 = os.path.join(visuals_dir, "nomothetic_estimation_bipartite_numeric_only.png")
            title5 = (
                f"Nomothetic estimation (P→C sparse network, numeric-only) | {pid}\n"
                f"Thickness/lightness ∝ relevance"
            )
            _draw_bipartite_sparse_pc(
                model=model,
                output_png=out5,
                title=title5,
                annotate_numeric_scores=False,
            )

            # ---------- P -> P and C -> C ----------
            if not bool(args.no_pp_cc):
                # P -> P dense
                dense_pp = _extract_dense_lookup_pp(model)
                pp_mat = _matrix_from_lookup(predictors, predictors, dense_pp, exclude_diag=True)
                sparse_pp = _extract_sparse_edges_pp(model)
                pp_signs = _sign_matrix_from_sparse_edges(predictors, predictors, sparse_pp, exclude_diag=True)

                out6 = os.path.join(visuals_dir, "predictor_predictor_relevance_heatmap.png")
                title6 = f"Predictor→Predictor dense heatmap | {pid}\nWHITE = masked (<{args.mask_dense_below:.2f}); diagonal blank"
                _draw_dense_heatmap(
                    predictors,
                    predictors,
                    pp_mat,
                    output_png=out6,
                    title=title6,
                    mask_below=float(args.mask_dense_below),
                )

                out7 = os.path.join(visuals_dir, "predictor_predictor_signed_sparse_heatmap.png")
                title7 = f"Predictor→Predictor signed (sparse edges only) | {pid}\nBLUE=+, RED=-, WHITE=not selected / NL/U/0"
                _draw_signed_sparse_heatmap(
                    predictors,
                    predictors,
                    pp_mat,
                    pp_signs,
                    output_png=out7,
                    title=title7,
                    exclude_diag=True,
                )

                out8 = os.path.join(visuals_dir, "predictor_predictor_network_sparse.png")
                title8 = f"Predictor→Predictor sparse network | {pid}\nEdges={len(sparse_pp)}"
                _draw_within_set_sparse_network(
                    predictors,
                    sparse_pp,
                    output_png=out8,
                    title=title8,
                )

                # C -> C dense
                dense_cc = _extract_dense_lookup_cc(model)
                cc_mat = _matrix_from_lookup(criteria, criteria, dense_cc, exclude_diag=True)
                sparse_cc = _extract_sparse_edges_cc(model)
                cc_signs = _sign_matrix_from_sparse_edges(criteria, criteria, sparse_cc, exclude_diag=True)

                out9 = os.path.join(visuals_dir, "criterion_criterion_relevance_heatmap.png")
                title9 = f"Criterion→Criterion dense heatmap | {pid}\nWHITE = masked (<{args.mask_dense_below:.2f}); diagonal blank"
                _draw_dense_heatmap(
                    criteria,
                    criteria,
                    cc_mat,
                    output_png=out9,
                    title=title9,
                    mask_below=float(args.mask_dense_below),
                )

                out10 = os.path.join(visuals_dir, "criterion_criterion_signed_sparse_heatmap.png")
                title10 = f"Criterion→Criterion signed (sparse edges only) | {pid}\nBLUE=+, RED=-, WHITE=not selected / NL/U/0"
                _draw_signed_sparse_heatmap(
                    criteria,
                    criteria,
                    cc_mat,
                    cc_signs,
                    output_png=out10,
                    title=title10,
                    exclude_diag=True,
                )

                out11 = os.path.join(visuals_dir, "criterion_criterion_network_sparse.png")
                title11 = f"Criterion→Criterion sparse network | {pid}\nEdges={len(sparse_cc)}"
                _draw_within_set_sparse_network(
                    criteria,
                    sparse_cc,
                    output_png=out11,
                    title=title11,
                )

            print(f"OK  {pid}")
            print(f"  - {out1}")
            print(f"  - {out2}")
            print(f"  - {out3}")
            print(f"  - {out4}")
            print(f"  - {out5}")
            if not bool(args.no_pp_cc):
                print(f"  - {out6}")
                print(f"  - {out7}")
                print(f"  - {out8}")
                print(f"  - {out9}")
                print(f"  - {out10}")
                print(f"  - {out11}")

        except Exception as e:
            print(f"FAIL {pseudoprofile_id}: {type(e).__name__}: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#TODO: still not proper logic ; like deal better with the possible types of relationships