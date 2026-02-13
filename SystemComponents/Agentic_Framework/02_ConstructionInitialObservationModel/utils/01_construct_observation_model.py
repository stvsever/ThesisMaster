#!/usr/bin/env python3
"""
01_construct_observation_model.py

Goal
----
Create an INITIAL observation model (criterion + predictor variables) that is ready for
data collection (EMA / digital phenotyping) and suitable for later graphical VAR (gVAR).

This script is intended as a sequel to:
  (A) HyDe-based predictor ranking run(s) that produced a dense_profiles.csv
      (semantic + lexical ranking against predictor ontology leaves; based on hypothetical docs).
  (B) LLM-based criterion→predictor mapping ranks:
      all_pseudoprofiles__predictor_ranks_dense.csv

It combines:
  1) Free-text complaint and LLM-decomposed/mapped criterion operationalization (mapped_criterions.csv)
  2) HyDe-based fused global ranking signals (dense_profiles.csv)
  3) LLM-based mapping signals (all_pseudoprofiles__predictor_ranks_dense.csv)
  4) A high-level BioPsychoSocial predictor ontology overview (predictors_list.txt)

Then it calls an LLM (gpt-5; DO NOT set temperature or max_output_tokens) to output a
structured JSON observation model:
  - A selected subset of criterion variables (and how to measure them)
  - A set of broad solution candidates (predictors/levers), preferably high-level for the initial model
  - Three FULL directed relevance grids (dense, nomothetic pairwise estimation):
      (1) predictor_criterion_relevance: P -> C (full grid)
      (2) predictor_predictor_relevance: P -> P (full directed grid excluding self)
      (3) criterion_criterion_relevance: C -> C (full directed grid excluding self)
  - Three SPARSE directed edge lists (optimized sparsification objective):
      (1) edges: P -> C sparse edges
      (2) edges_pp: P -> P sparse edges
      (3) edges_cc: C -> C sparse edges
  - gVAR-ready design notes (sampling plan, variance risks, stationarity notes, collinearity notes, etc.)

Critical internal consistency constraint:
- For every edge in edges / edges_pp / edges_cc, the edge's numeric estimated_relevance_0_1
  MUST match (within rounding) the corresponding dense relevance grid value.

Relevance score formatting rules:
- Dense grids use strings with comma decimal separator and exactly 5 decimals, e.g. "0,12345" or "1,00000".
- Sparse edges carry estimated_relevance_0_1 as a NUMBER in [0,1] (the same value as the dense grid cell).

Sparse edges must be OPTIMAL (prompt-enforced + validated):
- They should reflect BOTH:
  (a) absolute plausibility calibration (score meaning),
  (b) relative ranking consistency within each target node (incoming edge set).
- Heuristic optimality constraints are validated:
  - For each target node, sparse incoming edges must mostly coincide with the top-ranked dense candidates,
    avoiding “worse-than-available” inclusions without a strong justification.

Safety / Scope
--------------
- This is NOT diagnostic.
- No medication dosing instructions.
- If self-harm / acute risk signals appear in the complaint_text, the model must produce
  strong safety guidance in safety_notes.

Data collection feasibility
---------------------------
Data collection is allowed to be broad and pragmatic:
- Smartphone EMA (short questions; momentary or since-last-prompt windows)
- Wearables (sleep/HR/actigraphy proxies)
- Phone sensors (screen time; mobility; calls/messages metadata)
- Platform APIs / device wellbeing dashboards (screen time, app category minutes)
- Behavioral logs (task completion logs; spending logs if ethically possible)
The model must specify how each variable will be collected.

gVAR feasibility
---------------
This script explicitly pushes for:
- coherent sampling plan (no contradictions between prompts/day and variable sampling)
- sufficient within-person variance (prefer 1–9 Likert for many constructs)
- reduced redundancy/collinearity (avoid multiple near-duplicates)
- modeling notes (stationarity, transformations, lag plausibility)
- a tractable number of nodes relative to expected timepoints

Inputs (defaults can be overridden via CLI)
-------------------------------------------
1) mapped_criterions.csv
2) HyDe dense_profiles.csv
3) LLM-based mapping ranks CSV
4) High-level predictor ontology overview text file

Outputs
-------
Per-run directory:
  runs/<run_id>/
    config.json
    observation_models.jsonl
    variables_long.csv
    edges_long.csv
    edges_pp_long.csv
    edges_cc_long.csv
    predictor_criterion_relevance_long.csv
    predictor_predictor_relevance_long.csv
    criterion_criterion_relevance_long.csv
    validations.csv
    errors.csv
    profiles/<pseudoprofile_id>/
        input_payload.json
        llm_observation_model_raw.json
        llm_observation_model_final.json
        validation_report.json
        config.json

Usage
-----
export OPENAI_API_KEY="..."
python 01_construct_observation_model.py

Single profile:
python 01_construct_observation_model.py --pseudoprofile_id pseudoprofile_FTC_ID002

Design knobs:
- --n_criteria choice|<int>        (default: choice)
- --n_predictors choice|<int>      (default: choice)
- --prompt_top_hyde 60
- --prompt_top_mapping_global 60
- --prompt_top_mapping_per_criterion 10

Optional:
- --auto_repair                (default on) repair multiple times if internal inconsistencies are detected
- --deterministic_fix          (default on) enforce validator pass deterministically post-LLM

Environment
-----------
(Optional) .env supported via python-dotenv

OpenAI API
----------
Uses OpenAI Responses API with structured outputs (JSON Schema).
Per request: do NOT pass temperature or max_output_tokens.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import random
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()


# ============================================================
# Context (provided)
# ============================================================
FRAMEWORK_CONTEXT = (
    "The output of this script is part of a larger framework that will ultimately construct a criterion+predictor model "
    "that is ready for data collection (e.g., via ecological momentary assessments, digital phenotyping, etc.)."
    "The general treatment algorithm is based on breadth-first search algorithm that first explores the secondary nodes "
    "(i.e., child nodes of the primary nodes BIO PSYCHO and SOCIAL)."
    "After data collection, the most relevant solution-predictors are re-selected for further data-analyes where then I focus "
    "on their subsequent child nodes)."
    "This algorithm goes on until user reaches salutogenic satisfactory mental state."
)


# ============================================================
# Defaults (paths you provided)
# ============================================================
DEFAULT_MAPPED_CRITERIONS_PATH = (
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/Evaluation/"
    "02_mental_health_issue_operationalization/mapped_criterions.csv"
)

DEFAULT_HYDE_DENSE_PROFILES_PATH = (
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/Evaluation/"
    "03_construction_initial_observation_model/helpers/"
    "00_HyDe_based_predictor_ranks/runs/2026-01-15_19-50-34/dense_profiles.csv"
)

DEFAULT_LLM_MAPPING_RANKS_PATH = (
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/Evaluation/"
    "03_construction_initial_observation_model/helpers/"
    "00_LLM_based_mapping_based_predictor_ranks/all_pseudoprofiles__predictor_ranks_dense.csv"
)

DEFAULT_HIGH_LEVEL_ONTOLOGY_PATH = (
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/ontology_mappings/"
    "CRITERION/predictor_to_criterion/input_lists/predictors_list.txt"
)

DEFAULT_RESULTS_DIR = (
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/Evaluation"
    "/03_construction_initial_observation_model/constructed_PC_models"
)

# LLM model (per your request)
DEFAULT_LLM_MODEL_NAME = "gpt-5-mini"  # Use 'gpt-5' during deployment ; for now test with 'nano' or 'mini'


# ============================================================
# Prompt size controls (tune as needed)
# ============================================================
DEFAULT_PROMPT_TOP_HYDE = 60
DEFAULT_PROMPT_TOP_MAPPING_GLOBAL = 60
DEFAULT_PROMPT_TOP_MAPPING_PER_CRITERION = 10

# Variable-count defaults (per your request: default is "choice")
DEFAULT_N_CRITERIA = "choice"
DEFAULT_N_PREDICTORS = "choice"

# Execution defaults
DEFAULT_MAX_WORKERS = 60
DEFAULT_MAX_ONTOLOGY_CHARS = 120_000  # safety limit; increase if needed
DEFAULT_USE_CACHE = True

# Validation / auto-repair
DEFAULT_AUTO_REPAIR = True
DEFAULT_MAX_REPAIR_ATTEMPTS = 3  # stronger default than 1

# Deterministic postprocessing (guarantees validator pass)
DEFAULT_DETERMINISTIC_FIX = True
DEFAULT_MAX_FIX_PASSES = 5

# Profile sampling defaults (keeps your previous behavior, but fixes pseudoprofile_id edge-case)
DEFAULT_SAMPLE_N = 3
DEFAULT_SAMPLE_SEED = 42
DEFAULT_ENABLE_SAMPLING = True


# ============================================================
# Data structures
# ============================================================
@dataclass
class CriterionRow:
    variable_id: str
    variable_label: str
    variable_criterion: str
    variable_evidence: str
    query_text_used: str
    variable_polarity: Optional[str] = None
    variable_timeframe: Optional[str] = None
    variable_severity_0_1: Optional[float] = None
    variable_confidence_0_1: Optional[float] = None
    criterion_path: Optional[str] = None
    criterion_leaf: Optional[str] = None


@dataclass
class ProfileInput:
    pseudoprofile_id: str
    complaint_text: str
    decomp_n_variables: Optional[int]
    decomp_notes: str
    criteria: List[CriterionRow]
    complaint_unique_mapped_leaf_embed_paths: List[str]
    high_level_predictor_ontology_raw: str


@dataclass
class HydeGlobalRankItem:
    rank: int
    fused_score_0_1: Optional[float]
    predictor_path: str
    primary_node: str
    secondary_node: str


@dataclass
class HydeProfileSignals:
    run_id: str
    summary: str
    solutions_compact: str
    llm_model: str
    embedding_model: str
    global_top: List[HydeGlobalRankItem]


@dataclass
class MappingRankItem:
    part: str
    criterion_path: str
    criterion_leaf: str
    rank: int
    predictor_path: str
    relevance_score: float
    primary_node: str
    secondary_node: str


# ============================================================
# Helpers
# ============================================================
HYDE_GLOBAL_PATH_RE = re.compile(r"^global_(\d{3})_path_embedtext$")
SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
SCORE_COMMA5_RE = re.compile(r"^(0,\d{5}|1,00000)$")


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


def _now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _safe_filename(s: str) -> str:
    s = SAFE_FILENAME_RE.sub("_", str(s).strip())
    return s[:180] if len(s) > 180 else s


def _log(pid: str, msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    if pid:
        print(f"[{ts}] [{pid}] {msg}", flush=True)
    else:
        print(f"[{ts}] {msg}", flush=True)


def _with_retries(fn, *, pid: str = "", label: str = "", retries: int = 3, base_sleep_s: float = 2.0):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            if label:
                _log(pid, f"{label}: attempt {attempt}/{retries} ...")
            return fn()
        except Exception as e:
            last_exc = e
            _log(pid, f"{label}: attempt {attempt} FAILED with {type(e).__name__}: {e}")
            if attempt >= retries:
                break
            sleep_s = base_sleep_s * (2 ** (attempt - 1)) * (1.0 + random.random() * 0.25)
            _log(pid, f"{label}: sleeping {sleep_s:.2f}s before retry ...")
            time.sleep(sleep_s)
    raise last_exc


def read_text_file(path: str, max_chars: Optional[int] = None) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    if max_chars is not None and len(text) > max_chars:
        head = text[: max_chars // 2]
        tail = text[-max_chars // 2 :]
        return head + "\n\n[...TRUNCATED for context length safety...]\n\n" + tail
    return text


def parse_semicolon_paths(cell: Optional[str]) -> List[str]:
    if cell is None:
        return []
    if isinstance(cell, float) and pd.isna(cell):
        return []
    s = str(cell).strip()
    if not s:
        return []
    return [p.strip() for p in s.split(";") if p.strip()]


def split_ontology_path(path: str) -> List[str]:
    """
    Supports paths like:
      "[BIO] > Sleep_and_Circadian > Sleep_Routines_and_Practices"
    Returns segments stripped.
    """
    s = str(path or "").strip()
    if not s:
        return []
    return [p.strip() for p in s.split(">") if p.strip()]


def primary_node_from_path(path: str) -> str:
    parts = split_ontology_path(path)
    return parts[0] if parts else ""


def secondary_node_from_path(path: str) -> str:
    parts = split_ontology_path(path)
    if len(parts) >= 2:
        return f"{parts[0]} > {parts[1]}"
    if len(parts) == 1:
        return parts[0]
    return ""


def list_pseudoprofile_ids_from_mapped(mapped_criterions_path: str) -> List[str]:
    if not os.path.exists(mapped_criterions_path):
        raise FileNotFoundError(f"mapped_criterions.csv not found: {mapped_criterions_path}")
    df = pd.read_csv(mapped_criterions_path)
    if "pseudoprofile_id" not in df.columns:
        raise ValueError("mapped_criterions.csv missing required column: pseudoprofile_id")

    ids = (
        df["pseudoprofile_id"]
        .dropna()
        .astype(str)
        .map(str.strip)
        .loc[lambda s: s != ""]
        .unique()
        .tolist()
    )
    ids.sort()
    return ids


# ============================================================
# Input loaders
# ============================================================
def load_profile_input(
    mapped_criterions_path: str,
    ontology_path: str,
    pseudoprofile_id: str,
    max_ontology_chars: Optional[int] = None,
) -> ProfileInput:
    if not os.path.exists(mapped_criterions_path):
        raise FileNotFoundError(f"mapped_criterions.csv not found: {mapped_criterions_path}")
    if not os.path.exists(ontology_path):
        raise FileNotFoundError(f"predictors_list.txt not found: {ontology_path}")

    df = pd.read_csv(mapped_criterions_path)

    if "pseudoprofile_id" not in df.columns:
        raise ValueError("mapped_criterions.csv missing required column: pseudoprofile_id")

    sub = df[df["pseudoprofile_id"].astype(str) == str(pseudoprofile_id)].copy()
    if sub.empty:
        sample_ids = df["pseudoprofile_id"].dropna().astype(str).unique().tolist()[:15]
        raise ValueError(
            f"No rows found for pseudoprofile_id={pseudoprofile_id}. "
            f"Example available IDs: {sample_ids}"
        )

    first = sub.iloc[0]
    complaint_text = _safe_str(first.get("complaint_text", ""))
    decomp_notes = _safe_str(first.get("decomp_notes", ""))
    decomp_n_variables = None
    if "decomp_n_variables" in sub.columns:
        try:
            v = first.get("decomp_n_variables", None)
            decomp_n_variables = int(v) if v is not None and not pd.isna(v) else None
        except Exception:
            decomp_n_variables = None

    complaint_unique_paths: List[str] = []
    if "complaint_unique_mapped_leaf_embed_paths" in sub.columns:
        for cell in sub["complaint_unique_mapped_leaf_embed_paths"].tolist():
            paths = parse_semicolon_paths(cell)
            if paths:
                complaint_unique_paths = paths
                break

    needed_cols = [
        "variable_id",
        "variable_label",
        "variable_criterion",
        "variable_evidence",
        "query_text_used",
    ]
    missing = [c for c in needed_cols if c not in sub.columns]
    if missing:
        raise ValueError(f"mapped_criterions.csv missing required columns: {missing}")

    path_col_candidates = [
        "criterion_path",
        "variable_criterion_path",
        "mapped_criterion_path",
        "criterion_leaf_path",
        "mapped_leaf_path",
    ]
    leaf_col_candidates = [
        "criterion_leaf",
        "variable_criterion_leaf",
        "mapped_criterion_leaf",
    ]

    path_col = next((c for c in path_col_candidates if c in sub.columns), None)
    leaf_col = next((c for c in leaf_col_candidates if c in sub.columns), None)

    criteria_rows: List[CriterionRow] = []
    seen: set = set()

    for _, row in sub.iterrows():
        vid = _safe_str(row.get("variable_id", "")).strip()
        if not vid:
            vid = f"NO_ID::{_safe_str(row.get('variable_label',''))}::{_safe_str(row.get('variable_criterion',''))}"
        if vid in seen:
            continue
        seen.add(vid)

        crit_path = _safe_str(row.get(path_col, "")).strip() if path_col else ""
        crit_leaf = _safe_str(row.get(leaf_col, "")).strip() if leaf_col else ""

        criteria_rows.append(
            CriterionRow(
                variable_id=vid,
                variable_label=_safe_str(row.get("variable_label", "")).strip(),
                variable_criterion=_safe_str(row.get("variable_criterion", "")).strip(),
                variable_evidence=_safe_str(row.get("variable_evidence", "")).strip(),
                query_text_used=_safe_str(row.get("query_text_used", "")).strip(),
                variable_polarity=_safe_str(row.get("variable_polarity", "")).strip() or None,
                variable_timeframe=_safe_str(row.get("variable_timeframe", "")).strip() or None,
                variable_severity_0_1=_safe_float(row.get("variable_severity_0_1", None)),
                variable_confidence_0_1=_safe_float(row.get("variable_confidence_0_1", None)),
                criterion_path=crit_path or None,
                criterion_leaf=crit_leaf or None,
            )
        )

    ontology_raw = read_text_file(ontology_path, max_chars=max_ontology_chars)

    return ProfileInput(
        pseudoprofile_id=str(pseudoprofile_id),
        complaint_text=complaint_text,
        decomp_n_variables=decomp_n_variables,
        decomp_notes=decomp_notes,
        criteria=criteria_rows,
        complaint_unique_mapped_leaf_embed_paths=complaint_unique_paths,
        high_level_predictor_ontology_raw=ontology_raw,
    )


def load_hyde_signals_for_profile(
    hyde_dense_profiles_csv: str,
    pseudoprofile_id: str,
    *,
    top_n: int,
) -> Optional[HydeProfileSignals]:
    if not hyde_dense_profiles_csv or (not os.path.exists(hyde_dense_profiles_csv)):
        return None

    df = pd.read_csv(hyde_dense_profiles_csv)
    if "pseudoprofile_id" not in df.columns:
        return None

    sub = df[df["pseudoprofile_id"].astype(str) == str(pseudoprofile_id)].copy()
    if sub.empty:
        return None

    row = sub.iloc[0].to_dict()

    run_id = _safe_str(row.get("run_id", "")).strip()
    summary = _safe_str(row.get("summary", "")).strip()
    solutions_compact = _safe_str(row.get("solutions_compact", "")).strip()
    llm_model = _safe_str(row.get("llm_model", "")).strip()
    embedding_model = _safe_str(row.get("embedding_model", "")).strip()

    global_cols: List[Tuple[int, str]] = []
    for c in df.columns:
        m = HYDE_GLOBAL_PATH_RE.match(c)
        if m:
            global_cols.append((int(m.group(1)), c))
    global_cols.sort(key=lambda x: x[0])

    items: List[HydeGlobalRankItem] = []
    for idx, col_path in global_cols[: max(0, int(top_n))]:
        path = _safe_str(row.get(col_path, "")).strip()
        if not path:
            continue
        score = _safe_float(row.get(f"global_{idx:03d}_score", None))
        prim = primary_node_from_path(path)
        sec = secondary_node_from_path(path)
        items.append(
            HydeGlobalRankItem(
                rank=int(idx),
                fused_score_0_1=score,
                predictor_path=path,
                primary_node=prim,
                secondary_node=sec,
            )
        )

    return HydeProfileSignals(
        run_id=run_id,
        summary=summary,
        solutions_compact=solutions_compact,
        llm_model=llm_model,
        embedding_model=embedding_model,
        global_top=items,
    )


def load_llm_mapping_ranks_for_profile(
    mapping_ranks_csv: str,
    pseudoprofile_id: str,
    *,
    top_global: int,
    top_per_criterion: int,
) -> Dict[str, Any]:
    out = {"pre_global_top": [], "post_global_top": [], "post_per_criterion_top": {}}

    if not mapping_ranks_csv or (not os.path.exists(mapping_ranks_csv)):
        return out

    df = pd.read_csv(mapping_ranks_csv)
    required = [
        "pseudoprofile_id", "part", "criterion_path", "criterion_leaf",
        "rank", "predictor_path", "relevance_score"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return out

    sub = df[df["pseudoprofile_id"].astype(str) == str(pseudoprofile_id)].copy()
    if sub.empty:
        return out

    sub["part"] = sub["part"].astype(str)
    sub["criterion_path"] = sub["criterion_path"].fillna("").astype(str)
    sub["criterion_leaf"] = sub["criterion_leaf"].fillna("").astype(str)

    def _row_to_item(r: pd.Series) -> Optional[MappingRankItem]:
        p = _safe_str(r.get("predictor_path", "")).strip()
        if not p:
            return None
        part = _safe_str(r.get("part", "")).strip()
        cpath = _safe_str(r.get("criterion_path", "")).strip()
        cleaf = _safe_str(r.get("criterion_leaf", "")).strip()
        try:
            rank = int(r.get("rank", 0))
        except Exception:
            rank = 0
        s = _safe_float(r.get("relevance_score", None))
        if s is None:
            return None
        prim = primary_node_from_path(p)
        sec = secondary_node_from_path(p)
        return MappingRankItem(
            part=part,
            criterion_path=cpath,
            criterion_leaf=cleaf,
            rank=rank,
            predictor_path=p,
            relevance_score=float(s),
            primary_node=prim,
            secondary_node=sec,
        )

    pre = sub[sub["part"] == "pre_global"].sort_values(["rank"], ascending=[True]).head(int(top_global))
    out["pre_global_top"] = [it for it in (_row_to_item(r) for _, r in pre.iterrows()) if it is not None]

    post = sub[sub["part"] == "post_global"].sort_values(["rank"], ascending=[True]).head(int(top_global))
    out["post_global_top"] = [it for it in (_row_to_item(r) for _, r in post.iterrows()) if it is not None]

    per = sub[sub["part"] == "post_per_criterion"].copy()
    if not per.empty:
        per = per.sort_values(["criterion_path", "rank"], ascending=[True, True])
        grouped = per.groupby("criterion_path", sort=False)
        per_top: Dict[str, List[MappingRankItem]] = {}
        for cpath, g in grouped:
            g2 = g.head(int(top_per_criterion))
            items = [it for it in (_row_to_item(r) for _, r in g2.iterrows()) if it is not None]
            if items:
                per_top[str(cpath)] = items
        out["post_per_criterion_top"] = per_top

    return out


# ============================================================
# Score formatting helpers (deterministic correctness)
# ============================================================
def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _format_comma5(x: float) -> str:
    x = _clamp01(float(x))
    return f"{x:.5f}".replace(".", ",")


def _comma5_to_float(s: str) -> Optional[float]:
    if not isinstance(s, str):
        return None
    s2 = s.strip()
    if not s2:
        return None
    if not SCORE_COMMA5_RE.match(s2):
        # try permissive parse (e.g., "0.12345") then reformat later
        try:
            v = float(s2.replace(",", "."))
            return _clamp01(v)
        except Exception:
            return None
    try:
        return float(s2.replace(",", "."))
    except Exception:
        return None


# ============================================================
# Schema + prompt
# ============================================================
def observation_model_schema() -> Dict[str, Any]:
    """
    Structured output schema: strict + gVAR-oriented + coherent sampling design.
    """
    measurement_schema = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": [
                    "EMA_self_report",
                    "passive_phone_sensor",
                    "wearable_sensor",
                    "platform_api",
                    "behavioral_log",
                    "other",
                ],
            },
            "channel": {"type": "string"},
            "assessment_type": {
                "type": "string",
                "enum": [
                    "since_last_prompt",
                    "momentary",
                    "end_of_day",
                    "morning_diary",
                    "passive_aggregated_per_prompt",
                    "passive_daily_aggregate",
                    "passive_continuous",
                    "other",
                ],
            },
            "item_or_signal": {"type": "string"},
            "response_scale_or_unit": {"type": "string"},
            "sampling_per_day": {"type": "integer", "minimum": 1},
            "recall_window": {"type": "string"},
            "expected_within_person_variability": {
                "type": "string",
                "enum": ["high", "medium", "low", "unknown"],
            },
            "notes": {"type": "string"},
        },
        "required": [
            "mode",
            "channel",
            "assessment_type",
            "item_or_signal",
            "response_scale_or_unit",
            "sampling_per_day",
            "recall_window",
            "expected_within_person_variability",
            "notes",
        ],
        "additionalProperties": False,
    }

    gvar_handling_schema = {
        "type": "object",
        "properties": {
            "likely_stationary": {"type": "boolean"},
            "needs_detrending_or_seasonal_controls": {"type": "boolean"},
            "suggested_transformation": {"type": "string"},
            "collinearity_risk": {"type": "string", "enum": ["low", "medium", "high", "unknown"]},
            "distribution_hint": {
                "type": "string",
                "enum": ["approx_continuous", "ordinal", "count", "bounded_0_1", "binary", "other", "unknown"],
            },
        },
        "required": [
            "likely_stationary",
            "needs_detrending_or_seasonal_controls",
            "suggested_transformation",
            "collinearity_risk",
            "distribution_hint",
        ],
        "additionalProperties": False,
    }

    relevance_pc_entry_schema = {
        "type": "object",
        "properties": {
            "predictor_var_id": {"type": "string"},
            "criterion_var_id": {"type": "string"},
            "relevance_score_0_1_comma5": {"type": "string", "pattern": r"^(0,\d{5}|1,00000)$"},
        },
        "required": ["predictor_var_id", "criterion_var_id", "relevance_score_0_1_comma5"],
        "additionalProperties": False,
    }

    relevance_pp_entry_schema = {
        "type": "object",
        "properties": {
            "from_predictor_var_id": {"type": "string"},
            "to_predictor_var_id": {"type": "string"},
            "relevance_score_0_1_comma5": {"type": "string", "pattern": r"^(0,\d{5}|1,00000)$"},
        },
        "required": ["from_predictor_var_id", "to_predictor_var_id", "relevance_score_0_1_comma5"],
        "additionalProperties": False,
    }

    relevance_cc_entry_schema = {
        "type": "object",
        "properties": {
            "from_criterion_var_id": {"type": "string"},
            "to_criterion_var_id": {"type": "string"},
            "relevance_score_0_1_comma5": {"type": "string", "pattern": r"^(0,\d{5}|1,00000)$"},
        },
        "required": ["from_criterion_var_id", "to_criterion_var_id", "relevance_score_0_1_comma5"],
        "additionalProperties": False,
    }

    expected_sign_enum = [
        "monotonic_positive",
        "monotonic_negative",
        "inverted_U",
        "U_shaped",
        "threshold_positive",
        "threshold_negative",
        "saturation_positive",
        "saturation_negative",
        "interaction_moderation",
        "oscillatory",
        "proxy_indicator",
        "confounded_or_common_cause_possible",
        "reverse_causality_possible",
        "measurement_artifact_possible",
        "no_effect_expected",
        "unknown",
        "insufficient_knowledge",
    ]

    relation_interpretation_enum = [
        "hypothesized_direct_causal",
        "hypothesized_indirect_or_mediated",
        "temporal_association_noncausal_possible",
        "proxy_or_indicator",
        "bidirectional_feedback_possible",
        "confounding_likely",
        "reverse_causality_plausible",
        "measurement_artifact_possible",
        "unknown",
        "insufficient_knowledge",
    ]

    relevance_tier_enum = ["very_low", "low", "medium", "high", "very_high"]

    edges_pc_entry_schema = {
        "type": "object",
        "properties": {
            "from_predictor_var_id": {"type": "string"},
            "to_criterion_var_id": {"type": "string"},
            "expected_sign": {"type": "string", "enum": expected_sign_enum},
            "relation_interpretation": {"type": "string", "enum": relation_interpretation_enum},
            "lag_spec": {"type": "string"},
            "estimated_relevance_0_1": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "estimated_relevance_tier": {"type": "string", "enum": relevance_tier_enum},
            "notes": {"type": "string"},
        },
        "required": [
            "from_predictor_var_id",
            "to_criterion_var_id",
            "expected_sign",
            "relation_interpretation",
            "lag_spec",
            "estimated_relevance_0_1",
            "estimated_relevance_tier",
            "notes",
        ],
        "additionalProperties": False,
    }

    edges_pp_entry_schema = {
        "type": "object",
        "properties": {
            "from_predictor_var_id": {"type": "string"},
            "to_predictor_var_id": {"type": "string"},
            "expected_sign": {"type": "string", "enum": expected_sign_enum},
            "relation_interpretation": {"type": "string", "enum": relation_interpretation_enum},
            "lag_spec": {"type": "string"},
            "estimated_relevance_0_1": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "estimated_relevance_tier": {"type": "string", "enum": relevance_tier_enum},
            "notes": {"type": "string"},
        },
        "required": [
            "from_predictor_var_id",
            "to_predictor_var_id",
            "expected_sign",
            "relation_interpretation",
            "lag_spec",
            "estimated_relevance_0_1",
            "estimated_relevance_tier",
            "notes",
        ],
        "additionalProperties": False,
    }

    edges_cc_entry_schema = {
        "type": "object",
        "properties": {
            "from_criterion_var_id": {"type": "string"},
            "to_criterion_var_id": {"type": "string"},
            "expected_sign": {"type": "string", "enum": expected_sign_enum},
            "relation_interpretation": {"type": "string", "enum": relation_interpretation_enum},
            "lag_spec": {"type": "string"},
            "estimated_relevance_0_1": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "estimated_relevance_tier": {"type": "string", "enum": relevance_tier_enum},
            "notes": {"type": "string"},
        },
        "required": [
            "from_criterion_var_id",
            "to_criterion_var_id",
            "expected_sign",
            "relation_interpretation",
            "lag_spec",
            "estimated_relevance_0_1",
            "estimated_relevance_tier",
            "notes",
        ],
        "additionalProperties": False,
    }

    expected_effects_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "criterion_var_id": {"type": "string"},
                "expected_sign": {"type": "string", "enum": expected_sign_enum},
                "typical_lag_hours_min": {"type": "integer", "minimum": 0},
                "typical_lag_hours_max": {"type": "integer", "minimum": 0},
                "rationale": {"type": "string"},
            },
            "required": [
                "criterion_var_id",
                "expected_sign",
                "typical_lag_hours_min",
                "typical_lag_hours_max",
                "rationale",
            ],
            "additionalProperties": False,
        },
    }

    return {
        "type": "object",
        "properties": {
            "pseudoprofile_id": {"type": "string"},
            "model_summary": {"type": "string"},
            "variable_selection_notes": {"type": "string"},
            "design_recommendations": {
                "type": "object",
                "properties": {
                    "study_days": {"type": "integer", "minimum": 7},
                    "momentary_ema": {
                        "type": "object",
                        "properties": {
                            "prompts_per_day": {"type": "integer", "minimum": 1},
                            "schedule_type": {"type": "string"},
                            "time_blocks_local": {"type": "array", "items": {"type": "string"}},
                            "recall_window": {"type": "string"},
                            "expected_total_prompts": {"type": "integer", "minimum": 1},
                            "notes": {"type": "string"},
                        },
                        "required": [
                            "prompts_per_day",
                            "schedule_type",
                            "time_blocks_local",
                            "recall_window",
                            "expected_total_prompts",
                            "notes",
                        ],
                        "additionalProperties": False,
                    },
                    "daily_diary": {
                        "type": "object",
                        "properties": {
                            "enabled": {"type": "boolean"},
                            "entries_per_day": {"type": "integer", "minimum": 0},
                            "anchor_times_local": {"type": "array", "items": {"type": "string"}},
                            "expected_total_entries": {"type": "integer", "minimum": 0},
                            "notes": {"type": "string"},
                        },
                        "required": ["enabled", "entries_per_day", "anchor_times_local", "expected_total_entries", "notes"],
                        "additionalProperties": False,
                    },
                    "passive_streams": {
                        "type": "object",
                        "properties": {
                            "enabled": {"type": "boolean"},
                            "aggregation_strategy": {"type": "string"},
                            "notes": {"type": "string"},
                        },
                        "required": ["enabled", "aggregation_strategy", "notes"],
                        "additionalProperties": False,
                    },
                    "minimum_total_observations_target": {"type": "integer", "minimum": 30},
                    "gvar_readiness_notes": {"type": "string"},
                    "variance_support_notes": {"type": "string"},
                    "missingness_risk_notes": {"type": "string"},
                },
                "required": [
                    "study_days",
                    "momentary_ema",
                    "daily_diary",
                    "passive_streams",
                    "minimum_total_observations_target",
                    "gvar_readiness_notes",
                    "variance_support_notes",
                    "missingness_risk_notes",
                ],
                "additionalProperties": False,
            },
            "criteria_variables": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "var_id": {"type": "string"},
                        "source_variable_id": {"type": "string"},
                        "label": {"type": "string"},
                        "criterion_path": {"type": "string"},
                        "polarity": {"type": "string", "enum": ["lower_is_better", "higher_is_better", "bidirectional", "unknown"]},
                        "measurement": measurement_schema,
                        "include_priority": {"type": "string", "enum": ["HIGH", "MED", "LOW"]},
                        "gvar_handling": gvar_handling_schema,
                    },
                    "required": [
                        "var_id",
                        "source_variable_id",
                        "label",
                        "criterion_path",
                        "polarity",
                        "measurement",
                        "include_priority",
                        "gvar_handling",
                    ],
                    "additionalProperties": False,
                },
            },
            "predictor_variables": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "var_id": {"type": "string"},
                        "label": {"type": "string"},
                        "bio_psycho_social_domain": {
                            "type": "string",
                            "enum": [
                                "BIO",
                                "PSYCHO",
                                "SOCIAL",
                                "BIOxPSYCHO",
                                "PSYCHOxSOCIAL",
                                "BIOxSOCIAL",
                                "BIOxPSYCHOxSOCIAL",
                            ],
                        },
                        "ontology_path": {"type": "string"},
                        "ontology_level": {"type": "string", "enum": ["secondary", "tertiary_or_deeper", "custom"]},
                        "modifiability_0_1": {"type": "number", "minimum": 0, "maximum": 1},
                        "mechanism_rationale": {"type": "string"},
                        "intervention_examples_safe": {"type": "string"},
                        "measurement": measurement_schema,
                        "gvar_handling": gvar_handling_schema,
                        "targets_criteria_var_ids": {"type": "array", "items": {"type": "string"}},
                        "expected_effects": expected_effects_schema,
                        "include_priority": {"type": "string", "enum": ["HIGH", "MED", "LOW"]},
                        "risk_ethics_notes": {"type": "string"},
                        "feasibility": {
                            "type": "object",
                            "properties": {
                                "data_collection_feasibility_0_1": {"type": "number", "minimum": 0, "maximum": 1},
                                "gvar_feasibility_0_1": {"type": "number", "minimum": 0, "maximum": 1},
                                "notes": {"type": "string"},
                            },
                            "required": ["data_collection_feasibility_0_1", "gvar_feasibility_0_1", "notes"],
                            "additionalProperties": False,
                        },
                    },
                    "required": [
                        "var_id",
                        "label",
                        "bio_psycho_social_domain",
                        "ontology_path",
                        "ontology_level",
                        "modifiability_0_1",
                        "mechanism_rationale",
                        "intervention_examples_safe",
                        "measurement",
                        "gvar_handling",
                        "targets_criteria_var_ids",
                        "expected_effects",
                        "include_priority",
                        "risk_ethics_notes",
                        "feasibility",
                    ],
                    "additionalProperties": False,
                },
            },

            "predictor_criterion_relevance": {"type": "array", "items": relevance_pc_entry_schema},
            "predictor_predictor_relevance": {"type": "array", "items": relevance_pp_entry_schema},
            "criterion_criterion_relevance": {"type": "array", "items": relevance_cc_entry_schema},

            "edges": {"type": "array", "items": edges_pc_entry_schema},
            "edges_pp": {"type": "array", "items": edges_pp_entry_schema},
            "edges_cc": {"type": "array", "items": edges_cc_entry_schema},

            "diagnostics": {
                "type": "object",
                "properties": {
                    "expected_collinearity_risks": {"type": "array", "items": {"type": "string"}},
                    "expected_floor_ceiling_risks": {"type": "array", "items": {"type": "string"}},
                    "variance_enhancement_suggestions": {"type": "array", "items": {"type": "string"}},
                    "excluded_or_deprioritized_candidates": {"type": "array", "items": {"type": "string"}},
                    "gvar_feasibility_summary": {"type": "string"},
                },
                "required": [
                    "expected_collinearity_risks",
                    "expected_floor_ceiling_risks",
                    "variance_enhancement_suggestions",
                    "excluded_or_deprioritized_candidates",
                    "gvar_feasibility_summary",
                ],
                "additionalProperties": False,
            },
            "safety_notes": {"type": "string"},
        },
        "required": [
            "pseudoprofile_id",
            "model_summary",
            "variable_selection_notes",
            "design_recommendations",
            "criteria_variables",
            "predictor_variables",
            "predictor_criterion_relevance",
            "predictor_predictor_relevance",
            "criterion_criterion_relevance",
            "edges",
            "edges_pp",
            "edges_cc",
            "diagnostics",
            "safety_notes",
        ],
        "additionalProperties": False,
    }


def _mapping_items_to_dict(items: List[MappingRankItem]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for it in items:
        out.append(
            {
                "part": it.part,
                "criterion_path": it.criterion_path,
                "criterion_leaf": it.criterion_leaf,
                "rank": it.rank,
                "predictor_path": it.predictor_path,
                "relevance_score": it.relevance_score,
                "primary_node": it.primary_node,
                "secondary_node": it.secondary_node,
            }
        )
    return out


def build_llm_messages(
    profile: ProfileInput,
    hyde: Optional[HydeProfileSignals],
    mapping: Dict[str, Any],
    *,
    n_criteria: str,
    n_predictors: str,
) -> Tuple[str, List[Dict[str, Any]]]:

    criteria_payload: List[Dict[str, Any]] = []
    for c in profile.criteria:
        criteria_payload.append(
            {
                "variable_id": c.variable_id,
                "variable_label": c.variable_label,
                "variable_criterion": c.variable_criterion,
                "variable_evidence": c.variable_evidence,
                "query_text_used": c.query_text_used,
                "variable_polarity": c.variable_polarity,
                "variable_timeframe": c.variable_timeframe,
                "variable_severity_0_1": c.variable_severity_0_1,
                "variable_confidence_0_1": c.variable_confidence_0_1,
                "criterion_path": c.criterion_path or "",
                "criterion_leaf": c.criterion_leaf or "",
            }
        )

    hyde_payload: Dict[str, Any]
    if hyde is not None:
        hyde_payload = {
            "hyde_llm_summary": hyde.summary,
            "hyde_solutions_compact": hyde.solutions_compact,
            "hyde_global_top_leaf_paths": [
                {
                    "rank": it.rank,
                    "fused_score_0_1": it.fused_score_0_1,
                    "predictor_path": it.predictor_path,
                    "primary_node": it.primary_node,
                    "secondary_node": it.secondary_node,
                }
                for it in (hyde.global_top or [])
            ],
        }
    else:
        hyde_payload = {"hyde_llm_summary": "", "hyde_solutions_compact": "", "hyde_global_top_leaf_paths": []}

    mapping_payload = {
        "pre_global_top": _mapping_items_to_dict(mapping.get("pre_global_top", []) or []),
        "post_global_top": _mapping_items_to_dict(mapping.get("post_global_top", []) or []),
        "post_per_criterion_top": {
            str(cpath): _mapping_items_to_dict(items or [])
            for cpath, items in (mapping.get("post_per_criterion_top", {}) or {}).items()
        },
    }

    relevance_calibration = [
        "Dense relevance scores are NOMOTHETIC plausibility of a directed relation at the chosen timescale (for gVAR adjacency), not diagnosis.",
        "Absolute calibration anchors (be consistent): 0.00–0.10 none; 0.10–0.30 weak; 0.30–0.55 moderate; 0.55–0.80 strong; 0.80–1.00 very strong.",
        "Relative constraint: within each fixed TARGET node, the ordering of relevance MUST reflect your best belief AND provided ranking signals.",
        "Consistency constraint: each sparse edge copies the exact dense score into estimated_relevance_0_1 (numeric float) to 5 decimals.",
        "Sparsity default: per TARGET include 2–4 strongest incoming edges (P->C), 1–3 (P->P, C->C).",
        "Relevance tiers: very_low<0.10, low<0.30, medium<0.55, high<0.80, very_high>=0.80.",
        "HARD CAP to avoid token truncation: keep total nodes <= 18 (e.g., <=8 criteria and <=10 predictors). Prefer fewer high-signal constructs.",
    ]

    guidance = {
        "framework_context": FRAMEWORK_CONTEXT,
        "goal": (
            "Construct an initial criterion+predictor observation model ready for data collection "
            "and suitable for later gVAR analysis. Preserve complaint resolution while remaining estimable."
        ),
        "requested_variable_counts": {
            "n_criteria": str(n_criteria),
            "n_predictors": str(n_predictors),
            "note": (
                "If 'choice', choose an optimal number for gVAR feasibility while preserving coverage. "
                "Hard cap: total nodes <= 18 (avoid dense grid truncation)."
            ),
        },
        "measurement_principles": [
            "Prefer 1–9 Likert (anchored) for many constructs to increase variance.",
            "Avoid pure binary nodes; if needed, measure intensity 1–9 + optional event flag (not as node).",
            "Use 'since last prompt' for momentary EMA; 'morning diary' for last-night sleep; 'end_of_day' for daily summaries.",
            "If passive data used, specify precise signal + aggregation window.",
        ],
        "gvar_constraints": [
            "Coherent design: prompts/day must match per-variable sampling_per_day for momentary variables.",
            "Avoid redundancy/collinearity (pick one broad construct rather than near-duplicates).",
            "Prefer variables expected to fluctuate within-person at chosen timescale.",
            "Provide plausible lag bins aligned with schedule: '0-6h', '0-24h', '24-48h', 'multi_lag'.",
        ],
        "pairwise_estimation_requirement": [
            "You MUST output three dense relevance arrays: FULL P->C, FULL directed P->P (i!=j), FULL directed C->C (i!=j).",
            "Dense score format: comma decimal separator and exactly 5 decimals.",
            *relevance_calibration,
        ],
        "sparse_edges_requirement": [
            "You MUST output edges, edges_pp, edges_cc.",
            "Each edge must include expected_sign, relation_interpretation, lag_spec, estimated_relevance_0_1 (NUMBER), tier, notes.",
            "CRITICAL: estimated_relevance_0_1 must match dense grid cell value to 5 decimals.",
            "Choose sparse edges to maximize coverage and minimize redundancy; avoid lower-scoring inclusions vs omitted higher-scoring candidates for same target.",
        ],
        "safety_and_scope": [
            "Not diagnosis. No medication dosing.",
            "If acute risk is suggested, produce strong safety_notes encouraging professional help.",
        ],
        "output_format": "STRICT JSON matching the provided JSON Schema exactly.",
    }

    user_content = {
        "pseudoprofile_id": profile.pseudoprofile_id,
        "complaint_text": profile.complaint_text,
        "decomp_notes": profile.decomp_notes,
        "decomp_n_variables_original": profile.decomp_n_variables,
        "criteria_operationalization_rows": criteria_payload,
        "predictor_ontology_overview_RAW": profile.high_level_predictor_ontology_raw,
        "signals": {
            "hyde_based_predictor_rankings": hyde_payload,
            "llm_based_criterion_to_predictor_mapping_ranks": mapping_payload,
        },
        "guidance": guidance,
    }

    instructions = (
        "You are an interdisciplinary health-engineering expert constructing an INITIAL gVAR-appropriate observation model.\n\n"
        "Your task is NOT to diagnose. Your task is to construct an INITIAL observation model of:\n"
        "- criterion variables that represent operationalized fully preserved free-text description of (non-)clinical mental health state variables\n "
        "- AND plausbile predictor variables as candidate solutions) suitable for digital data collection and later gVAR analysis.\n\n"
        "Hard constraints:\n"
        "- Output MUST strictly follow the provided JSON schema.\n"
        "- Coherent sampling design: prompts/day and variable sampling_per_day must be consistent.\n"
        "- For every variable, specify exactly how it will be collected.\n"
        "- Dense grids must be COMPLETE.\n"
        "- Sparse edges must copy the dense score exactly.\n"
        "- Keep total nodes <= 18.\n"
    )

    messages = [
        {
            "role": "user",
            "content": (
                "Build the initial observation model from the payload below.\n\n"
                "CASE_PAYLOAD_JSON:\n"
                f"{json.dumps(user_content, ensure_ascii=False, indent=2)}"
            ),
        }
    ]

    return instructions, messages


def call_llm_structured(
    client: OpenAI,
    llm_model: str,
    instructions: str,
    messages: List[Dict[str, Any]],
    *,
    pseudoprofile_id: str,
) -> Dict[str, Any]:
    schema = observation_model_schema()

    _log(pseudoprofile_id, f"LLM call START | model={llm_model}")

    response = client.responses.create(
        model=llm_model,
        instructions=instructions,
        input=messages,
        text={
            "format": {
                "type": "json_schema",
                "name": "initial_observation_model",
                "strict": True,
                "schema": schema,
            }
        },
    )

    if getattr(response, "status", None) == "incomplete":
        details = getattr(response, "incomplete_details", None)
        raise RuntimeError(f"LLM response incomplete: {details}")

    raw = getattr(response, "output_text", None)
    if not raw:
        raise RuntimeError("No output_text returned by the model.")

    try:
        _log(pseudoprofile_id, "LLM call OK | parsing JSON")
        return json.loads(raw)
    except json.JSONDecodeError as e:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except Exception:
                pass
        raise RuntimeError(f"Failed to parse JSON from model output: {e}\n\nRAW:\n{raw}") from e


# ============================================================
# Validation (original, unchanged)
# ============================================================
def _get_design_numbers(model: Dict[str, Any]) -> Tuple[int, int, int]:
    dr = model.get("design_recommendations", {}) or {}
    study_days = int(dr.get("study_days", 0) or 0)
    me = dr.get("momentary_ema", {}) or {}
    prompts_per_day = int(me.get("prompts_per_day", 0) or 0)
    expected_total_prompts = int(me.get("expected_total_prompts", 0) or 0)
    return study_days, prompts_per_day, expected_total_prompts


def _tier_from_score(x: Optional[float]) -> str:
    if x is None:
        return "very_low"
    if x < 0.10:
        return "very_low"
    if x < 0.30:
        return "low"
    if x < 0.55:
        return "medium"
    if x < 0.80:
        return "high"
    return "very_high"


def _validate_sparse_optimality_for_target(
    *,
    target_id: str,
    incoming_edges: List[Tuple[str, float]],  # (from_id, score)
    all_candidates: List[Tuple[str, float]],  # (from_id, score) for ALL possible incoming
    min_in: int,
    soft_max_in: int,
    hard_max_in: int,
    severe_gap: float = 0.10,
) -> Tuple[List[str], List[str]]:
    errs: List[str] = []
    warns: List[str] = []

    all_sorted = sorted(all_candidates, key=lambda x: x[1], reverse=True)
    inc_sorted = sorted(incoming_edges, key=lambda x: x[1], reverse=True)

    if len(all_sorted) == 0:
        return errs, warns

    if len(all_sorted) >= min_in and len(inc_sorted) < min_in:
        warns.append(f"Target {target_id}: has only {len(inc_sorted)} incoming edges; recommended >= {min_in}.")

    if len(inc_sorted) > hard_max_in:
        warns.append(f"Target {target_id}: has {len(inc_sorted)} incoming edges; exceeds hard_max_in={hard_max_in}.")

    included_from = {a for a, _ in inc_sorted}
    omitted = [(a, s) for (a, s) in all_sorted if a not in included_from]
    if inc_sorted and omitted:
        worst_included = inc_sorted[-1][1]
        best_omitted = omitted[0][1]
        if best_omitted - worst_included >= severe_gap:
            errs.append(
                f"Target {target_id}: suboptimal selection: best omitted score={best_omitted:.5f} "
                f"is >= {severe_gap:.2f} higher than worst included score={worst_included:.5f}."
            )

    if len(all_sorted) >= soft_max_in and len(inc_sorted) > 0:
        top_soft = {a for a, _ in all_sorted[:soft_max_in]}
        outside = [(a, s) for (a, s) in inc_sorted if a not in top_soft]
        if outside:
            warns.append(
                f"Target {target_id}: {len(outside)} included incoming edge(s) not in top-{soft_max_in} candidates "
                f"(example: {outside[:2]})."
            )

    return errs, warns


def validate_observation_model(model: Dict[str, Any]) -> Dict[str, Any]:
    # --- YOUR ORIGINAL VALIDATOR (kept as-is) ---
    errors: List[str] = []
    warnings: List[str] = []
    stats: Dict[str, Any] = {}

    if not isinstance(model, dict):
        return {"errors": ["Model is not a dict."], "warnings": [], "stats": {}}

    pid = _safe_str(model.get("pseudoprofile_id", "")).strip()
    if not pid:
        errors.append("Missing pseudoprofile_id.")

    # Design coherence
    try:
        study_days, prompts_per_day, expected_total_prompts = _get_design_numbers(model)
        stats["study_days"] = study_days
        stats["momentary_prompts_per_day"] = prompts_per_day
        stats["expected_total_prompts"] = expected_total_prompts

        if study_days <= 0:
            errors.append("design_recommendations.study_days must be >= 7.")
        if prompts_per_day <= 0:
            errors.append("design_recommendations.momentary_ema.prompts_per_day must be >= 1.")
        if study_days > 0 and prompts_per_day > 0:
            computed = study_days * prompts_per_day
            stats["computed_total_prompts"] = computed
            if expected_total_prompts != computed:
                errors.append(
                    f"design_recommendations.momentary_ema.expected_total_prompts={expected_total_prompts} "
                    f"does not match study_days*prompts_per_day={computed}."
                )

        dr = model.get("design_recommendations", {}) or {}
        min_target = int(dr.get("minimum_total_observations_target", 0) or 0)

        dd = dr.get("daily_diary", {}) or {}
        dd_enabled = bool(dd.get("enabled", False))
        dd_entries_per_day = int(dd.get("entries_per_day", 0) or 0)
        dd_expected_total = int(dd.get("expected_total_entries", 0) or 0)
        if dd_enabled:
            computed_dd_total = study_days * dd_entries_per_day
            stats["computed_total_daily_diary_entries"] = computed_dd_total
            if dd_expected_total != computed_dd_total:
                errors.append(
                    f"design_recommendations.daily_diary.expected_total_entries={dd_expected_total} "
                    f"does not match study_days*entries_per_day={computed_dd_total}."
                )
        else:
            if dd_entries_per_day != 0 or dd_expected_total != 0:
                errors.append("daily_diary.enabled is false but entries_per_day or expected_total_entries is non-zero.")

        total_expected_observations = expected_total_prompts + (dd_expected_total if dd_enabled else 0)
        stats["total_expected_observations"] = total_expected_observations
        if min_target <= 0:
            errors.append("design_recommendations.minimum_total_observations_target must be >= 30.")
        elif min_target > total_expected_observations:
            errors.append(
                f"minimum_total_observations_target={min_target} exceeds total_expected_observations={total_expected_observations}."
            )

    except Exception as e:
        errors.append(f"Design parsing failed: {type(e).__name__}: {e}")

    crit = model.get("criteria_variables", []) or []
    pred = model.get("predictor_variables", []) or []
    if not isinstance(crit, list) or not isinstance(pred, list):
        errors.append("criteria_variables and predictor_variables must be arrays.")
        crit = []
        pred = []

    stats["n_criteria"] = len(crit)
    stats["n_predictors"] = len(pred)
    stats["n_total_variables"] = len(crit) + len(pred)

    # gVAR feasibility heuristic (warn only)
    try:
        _study_days, _ppd, _etp = _get_design_numbers(model)
        K = stats["n_total_variables"]
        T = _etp
        stats["gvar_T_over_K"] = (float(T) / float(K)) if K > 0 else None
        if K > 0 and T > 0:
            if T < 3 * K:
                warnings.append(f"gVAR risk: momentary timepoints T={T} is < 3*K={3*K} (K={K}). Estimation may be unstable.")
            elif T < 5 * K:
                warnings.append(f"gVAR caution: T={T} is < 5*K={5*K} (K={K}). Consider fewer variables or more timepoints.")
    except Exception:
        pass

    # Validate IDs + sampling consistency
    all_var_ids: set = set()
    crit_ids: List[str] = []
    pred_ids: List[str] = []

    for arr_name, arr in [("criteria_variables", crit), ("predictor_variables", pred)]:
        for i, v in enumerate(arr):
            vid = _safe_str(v.get("var_id", "")).strip()
            if not vid:
                errors.append(f"{arr_name}[{i}].var_id is missing/empty.")
                continue
            if vid in all_var_ids:
                errors.append(f"Duplicate var_id '{vid}' across variables.")
            all_var_ids.add(vid)
            if arr_name == "criteria_variables":
                crit_ids.append(vid)
            else:
                pred_ids.append(vid)

            meas = v.get("measurement", {}) or {}
            if not isinstance(meas, dict):
                errors.append(f"{arr_name}[{i}].measurement is not an object.")
                continue

            mode = _safe_str(meas.get("mode", "")).strip()
            item = _safe_str(meas.get("item_or_signal", "")).strip()
            if not item:
                errors.append(f"{arr_name}[{i}] '{vid}': measurement.item_or_signal is empty.")
            if mode == "EMA_self_report":
                if ("?" not in item) and ("how" not in item.lower()) and ("did you" not in item.lower()) and ("rate" not in item.lower()):
                    warnings.append(f"{arr_name}[{i}] '{vid}': EMA_self_report item_or_signal does not look like an explicit question.")

            try:
                spd = int(meas.get("sampling_per_day", 0) or 0)
            except Exception:
                spd = 0
            atype = _safe_str(meas.get("assessment_type", "")).strip()
            if spd <= 0:
                errors.append(f"{arr_name}[{i}] '{vid}': sampling_per_day must be >= 1.")
            else:
                try:
                    _sd, ppd, _etp = _get_design_numbers(model)
                except Exception:
                    ppd = 0

                if atype in ("since_last_prompt", "momentary", "passive_aggregated_per_prompt"):
                    if ppd > 0 and spd != ppd:
                        errors.append(
                            f"{arr_name}[{i}] '{vid}': assessment_type='{atype}' implies sampling_per_day should equal momentary prompts_per_day={ppd}, but got {spd}."
                        )
                if atype in ("morning_diary", "end_of_day", "passive_daily_aggregate"):
                    dd = (model.get("design_recommendations", {}) or {}).get("daily_diary", {}) or {}
                    if not bool(dd.get("enabled", False)):
                        errors.append(
                            f"{arr_name}[{i}] '{vid}': assessment_type='{atype}' requires design_recommendations.daily_diary.enabled=true."
                        )
                    if spd != 1:
                        errors.append(f"{arr_name}[{i}] '{vid}': assessment_type='{atype}' should have sampling_per_day=1, got {spd}.")

    pred_set = set(pred_ids)
    crit_set = set(crit_ids)

    pc_rel = model.get("predictor_criterion_relevance", None)
    pp_rel = model.get("predictor_predictor_relevance", None)
    cc_rel = model.get("criterion_criterion_relevance", None)

    if pc_rel is None:
        errors.append("Missing required field: predictor_criterion_relevance.")
        pc_rel = []
    if pp_rel is None:
        errors.append("Missing required field: predictor_predictor_relevance.")
        pp_rel = []
    if cc_rel is None:
        errors.append("Missing required field: criterion_criterion_relevance.")
        cc_rel = []

    if not isinstance(pc_rel, list):
        errors.append("predictor_criterion_relevance must be an array.")
        pc_rel = []
    if not isinstance(pp_rel, list):
        errors.append("predictor_predictor_relevance must be an array.")
        pp_rel = []
    if not isinstance(cc_rel, list):
        errors.append("criterion_criterion_relevance must be an array.")
        cc_rel = []

    stats["n_pc_relevance_pairs"] = len(pc_rel)
    stats["n_pp_relevance_pairs"] = len(pp_rel)
    stats["n_cc_relevance_pairs"] = len(cc_rel)

    expected_pc = len(pred_ids) * len(crit_ids)
    expected_pp = len(pred_ids) * max(0, (len(pred_ids) - 1))
    expected_cc = len(crit_ids) * max(0, (len(crit_ids) - 1))
    stats["expected_pc_relevance_pairs"] = expected_pc
    stats["expected_pp_relevance_pairs"] = expected_pp
    stats["expected_cc_relevance_pairs"] = expected_cc

    pc_lookup: Dict[Tuple[str, str], float] = {}
    pp_lookup: Dict[Tuple[str, str], float] = {}
    cc_lookup: Dict[Tuple[str, str], float] = {}

    # P->C
    if len(pred_ids) > 0 and len(crit_ids) > 0 and len(pc_rel) != expected_pc:
        errors.append(f"predictor_criterion_relevance must include FULL grid size {expected_pc}, got {len(pc_rel)}.")

    seen_pc: set = set()
    for k, r in enumerate(pc_rel):
        if not isinstance(r, dict):
            errors.append(f"predictor_criterion_relevance[{k}] must be an object.")
            continue
        pvid = _safe_str(r.get("predictor_var_id", "")).strip()
        cvid = _safe_str(r.get("criterion_var_id", "")).strip()
        sval = _safe_str(r.get("relevance_score_0_1_comma5", "")).strip()

        if not pvid or pvid not in pred_set:
            errors.append(f"predictor_criterion_relevance[{k}].predictor_var_id '{pvid}' invalid.")
        if not cvid or cvid not in crit_set:
            errors.append(f"predictor_criterion_relevance[{k}].criterion_var_id '{cvid}' invalid.")
        if pvid and cvid:
            pair = (pvid, cvid)
            if pair in seen_pc:
                errors.append(f"Duplicate predictor_criterion_relevance pair: {pvid}->{cvid}.")
            seen_pc.add(pair)

        if not sval or not SCORE_COMMA5_RE.match(sval):
            errors.append(f"predictor_criterion_relevance[{k}].relevance_score_0_1_comma5 '{sval}' invalid.")
            continue
        fval = _comma5_to_float(sval)
        if fval is None or fval < -1e-12 or fval > 1.0 + 1e-12:
            errors.append(f"predictor_criterion_relevance[{k}] score '{sval}' not parseable/in range.")
            continue
        if pvid and cvid:
            pc_lookup[(pvid, cvid)] = float(fval)

    if len(pred_ids) > 0 and len(crit_ids) > 0:
        expected_set = {(p, c) for p in pred_ids for c in crit_ids}
        missing_pairs = expected_set.difference(seen_pc)
        if missing_pairs:
            sample = list(sorted(missing_pairs))[:10]
            errors.append(f"predictor_criterion_relevance missing {len(missing_pairs)} pair(s), e.g.: {sample}")

    # P->P (exclude self)
    if len(pred_ids) > 1 and len(pp_rel) != expected_pp:
        errors.append(f"predictor_predictor_relevance must include FULL directed grid size {expected_pp}, got {len(pp_rel)}.")

    seen_pp: set = set()
    for k, r in enumerate(pp_rel):
        if not isinstance(r, dict):
            errors.append(f"predictor_predictor_relevance[{k}] must be an object.")
            continue
        fr = _safe_str(r.get("from_predictor_var_id", "")).strip()
        to = _safe_str(r.get("to_predictor_var_id", "")).strip()
        sval = _safe_str(r.get("relevance_score_0_1_comma5", "")).strip()

        if not fr or fr not in pred_set:
            errors.append(f"predictor_predictor_relevance[{k}].from_predictor_var_id '{fr}' invalid.")
        if not to or to not in pred_set:
            errors.append(f"predictor_predictor_relevance[{k}].to_predictor_var_id '{to}' invalid.")
        if fr and to and fr == to:
            errors.append(f"predictor_predictor_relevance[{k}] self-pair not allowed: {fr}->{to}.")

        if fr and to:
            pair = (fr, to)
            if pair in seen_pp:
                errors.append(f"Duplicate predictor_predictor_relevance pair: {fr}->{to}.")
            seen_pp.add(pair)

        if not sval or not SCORE_COMMA5_RE.match(sval):
            errors.append(f"predictor_predictor_relevance[{k}].relevance_score_0_1_comma5 '{sval}' invalid.")
            continue
        fval = _comma5_to_float(sval)
        if fval is None or fval < -1e-12 or fval > 1.0 + 1e-12:
            errors.append(f"predictor_predictor_relevance[{k}] score '{sval}' not parseable/in range.")
            continue
        if fr and to and fr != to:
            pp_lookup[(fr, to)] = float(fval)

    if len(pred_ids) > 1:
        expected_set = {(a, b) for a in pred_ids for b in pred_ids if a != b}
        missing_pairs = expected_set.difference(seen_pp)
        if missing_pairs:
            sample = list(sorted(missing_pairs))[:10]
            errors.append(f"predictor_predictor_relevance missing {len(missing_pairs)} pair(s), e.g.: {sample}")

    # C->C (exclude self)
    if len(crit_ids) > 1 and len(cc_rel) != expected_cc:
        errors.append(f"criterion_criterion_relevance must include FULL directed grid size {expected_cc}, got {len(cc_rel)}.")

    seen_cc: set = set()
    for k, r in enumerate(cc_rel):
        if not isinstance(r, dict):
            errors.append(f"criterion_criterion_relevance[{k}] must be an object.")
            continue
        fr = _safe_str(r.get("from_criterion_var_id", "")).strip()
        to = _safe_str(r.get("to_criterion_var_id", "")).strip()
        sval = _safe_str(r.get("relevance_score_0_1_comma5", "")).strip()

        if not fr or fr not in crit_set:
            errors.append(f"criterion_criterion_relevance[{k}].from_criterion_var_id '{fr}' invalid.")
        if not to or to not in crit_set:
            errors.append(f"criterion_criterion_relevance[{k}].to_criterion_var_id '{to}' invalid.")
        if fr and to and fr == to:
            errors.append(f"criterion_criterion_relevance[{k}] self-pair not allowed: {fr}->{to}.")

        if fr and to:
            pair = (fr, to)
            if pair in seen_cc:
                errors.append(f"Duplicate criterion_criterion_relevance pair: {fr}->{to}.")
            seen_cc.add(pair)

        if not sval or not SCORE_COMMA5_RE.match(sval):
            errors.append(f"criterion_criterion_relevance[{k}].relevance_score_0_1_comma5 '{sval}' invalid.")
            continue
        fval = _comma5_to_float(sval)
        if fval is None or fval < -1e-12 or fval > 1.0 + 1e-12:
            errors.append(f"criterion_criterion_relevance[{k}] score '{sval}' not parseable/in range.")
            continue
        if fr and to and fr != to:
            cc_lookup[(fr, to)] = float(fval)

    if len(crit_ids) > 1:
        expected_set = {(a, b) for a in crit_ids for b in crit_ids if a != b}
        missing_pairs = expected_set.difference(seen_cc)
        if missing_pairs:
            sample = list(sorted(missing_pairs))[:10]
            errors.append(f"criterion_criterion_relevance missing {len(missing_pairs)} pair(s), e.g.: {sample}")

    edges_pc = model.get("edges", []) or []
    edges_pp = model.get("edges_pp", []) or []
    edges_cc = model.get("edges_cc", []) or []

    if not isinstance(edges_pc, list):
        errors.append("edges must be an array.")
        edges_pc = []
    if not isinstance(edges_pp, list):
        errors.append("edges_pp must be an array.")
        edges_pp = []
    if not isinstance(edges_cc, list):
        errors.append("edges_cc must be an array.")
        edges_cc = []

    stats["n_edges_pc"] = len(edges_pc)
    stats["n_edges_pp"] = len(edges_pp)
    stats["n_edges_cc"] = len(edges_cc)

    tol = 1e-6

    # P->C
    seen_edge_pc: set = set()
    incoming_pc: Dict[str, List[Tuple[str, float]]] = {c: [] for c in crit_ids}

    for j, e in enumerate(edges_pc):
        fr = _safe_str(e.get("from_predictor_var_id", "")).strip()
        to = _safe_str(e.get("to_criterion_var_id", "")).strip()
        if fr not in pred_set:
            errors.append(f"edges[{j}]: from_predictor_var_id '{fr}' not found among predictors.")
        if to not in crit_set:
            errors.append(f"edges[{j}]: to_criterion_var_id '{to}' not found among criteria.")
        if fr and to:
            pair = (fr, to)
            if pair in seen_edge_pc:
                errors.append(f"edges[{j}]: duplicate edge {fr}->{to}.")
            seen_edge_pc.add(pair)

        est = e.get("estimated_relevance_0_1", None)
        if est is None or not isinstance(est, (int, float)):
            errors.append(f"edges[{j}]: estimated_relevance_0_1 missing or not a number.")
        else:
            if est < -1e-12 or est > 1.0 + 1e-12:
                errors.append(f"edges[{j}]: estimated_relevance_0_1={est} not in [0,1].")
            if fr and to and (fr, to) in pc_lookup:
                if abs(float(est) - float(pc_lookup[(fr, to)])) > tol:
                    errors.append(
                        f"edges[{j}]: estimated_relevance_0_1={float(est):.5f} does not match dense P->C value "
                        f"{pc_lookup[(fr, to)]:.5f} for {fr}->{to}."
                    )
        if fr and to and isinstance(est, (int, float)) and to in incoming_pc:
            incoming_pc[to].append((fr, float(est)))

        tier = _safe_str(e.get("estimated_relevance_tier", "")).strip()
        if isinstance(est, (int, float)) and tier:
            expected_tier = _tier_from_score(float(est))
            if tier != expected_tier:
                warnings.append(f"edges[{j}]: tier='{tier}' inconsistent with score {float(est):.5f} (expected '{expected_tier}').")

        lag_spec = _safe_str(e.get("lag_spec", "")).strip()
        if not lag_spec:
            errors.append(f"edges[{j}]: lag_spec is empty.")

    # P->P
    seen_edge_pp: set = set()
    incoming_pp: Dict[str, List[Tuple[str, float]]] = {p: [] for p in pred_ids}

    for j, e in enumerate(edges_pp):
        fr = _safe_str(e.get("from_predictor_var_id", "")).strip()
        to = _safe_str(e.get("to_predictor_var_id", "")).strip()
        if fr not in pred_set:
            errors.append(f"edges_pp[{j}]: from_predictor_var_id '{fr}' invalid.")
        if to not in pred_set:
            errors.append(f"edges_pp[{j}]: to_predictor_var_id '{to}' invalid.")
        if fr and to and fr == to:
            errors.append(f"edges_pp[{j}]: self-edge not allowed: {fr}->{to}.")
        if fr and to:
            pair = (fr, to)
            if pair in seen_edge_pp:
                errors.append(f"edges_pp[{j}]: duplicate edge {fr}->{to}.")
            seen_edge_pp.add(pair)

        est = e.get("estimated_relevance_0_1", None)
        if est is None or not isinstance(est, (int, float)):
            errors.append(f"edges_pp[{j}]: estimated_relevance_0_1 missing or not a number.")
        else:
            if est < -1e-12 or est > 1.0 + 1e-12:
                errors.append(f"edges_pp[{j}]: estimated_relevance_0_1={est} not in [0,1].")
            if fr and to and (fr, to) in pp_lookup:
                if abs(float(est) - float(pp_lookup[(fr, to)])) > tol:
                    errors.append(
                        f"edges_pp[{j}]: estimated_relevance_0_1={float(est):.5f} does not match dense P->P value "
                        f"{pp_lookup[(fr, to)]:.5f} for {fr}->{to}."
                    )
        if fr and to and isinstance(est, (int, float)) and to in incoming_pp:
            incoming_pp[to].append((fr, float(est)))

        lag_spec = _safe_str(e.get("lag_spec", "")).strip()
        if not lag_spec:
            errors.append(f"edges_pp[{j}]: lag_spec is empty.")

    # C->C
    seen_edge_cc: set = set()
    incoming_cc: Dict[str, List[Tuple[str, float]]] = {c: [] for c in crit_ids}

    for j, e in enumerate(edges_cc):
        fr = _safe_str(e.get("from_criterion_var_id", "")).strip()
        to = _safe_str(e.get("to_criterion_var_id", "")).strip()
        if fr not in crit_set:
            errors.append(f"edges_cc[{j}]: from_criterion_var_id '{fr}' invalid.")
        if to not in crit_set:
            errors.append(f"edges_cc[{j}]: to_criterion_var_id '{to}' invalid.")
        if fr and to and fr == to:
            errors.append(f"edges_cc[{j}]: self-edge not allowed: {fr}->{to}.")
        if fr and to:
            pair = (fr, to)
            if pair in seen_edge_cc:
                errors.append(f"edges_cc[{j}]: duplicate edge {fr}->{to}.")
            seen_edge_cc.add(pair)

        est = e.get("estimated_relevance_0_1", None)
        if est is None or not isinstance(est, (int, float)):
            errors.append(f"edges_cc[{j}]: estimated_relevance_0_1 missing or not a number.")
        else:
            if est < -1e-12 or est > 1.0 + 1e-12:
                errors.append(f"edges_cc[{j}]: estimated_relevance_0_1={est} not in [0,1].")
            if fr and to and (fr, to) in cc_lookup:
                if abs(float(est) - float(cc_lookup[(fr, to)])) > tol:
                    errors.append(
                        f"edges_cc[{j}]: estimated_relevance_0_1={float(est):.5f} does not match dense C->C value "
                        f"{cc_lookup[(fr, to)]:.5f} for {fr}->{to}."
                    )
        if fr and to and isinstance(est, (int, float)) and to in incoming_cc:
            incoming_cc[to].append((fr, float(est)))

        lag_spec = _safe_str(e.get("lag_spec", "")).strip()
        if not lag_spec:
            errors.append(f"edges_cc[{j}]: lag_spec is empty.")

    # Optimality heuristics
    if pc_lookup and pred_ids and crit_ids:
        for c in crit_ids:
            all_in = [(p, pc_lookup.get((p, c), 0.0)) for p in pred_ids]
            inc = incoming_pc.get(c, [])
            e2, w2 = _validate_sparse_optimality_for_target(
                target_id=c, incoming_edges=inc, all_candidates=all_in, min_in=2, soft_max_in=4, hard_max_in=6
            )
            errors.extend([f"edges optimality: {x}" for x in e2])
            warnings.extend([f"edges optimality: {x}" for x in w2])

    if pp_lookup and len(pred_ids) > 1:
        for t in pred_ids:
            all_in = [(p, pp_lookup.get((p, t), 0.0)) for p in pred_ids if p != t]
            inc = incoming_pp.get(t, [])
            e2, w2 = _validate_sparse_optimality_for_target(
                target_id=t, incoming_edges=inc, all_candidates=all_in, min_in=1, soft_max_in=3, hard_max_in=5
            )
            errors.extend([f"edges_pp optimality: {x}" for x in e2])
            warnings.extend([f"edges_pp optimality: {x}" for x in w2])

    if cc_lookup and len(crit_ids) > 1:
        for t in crit_ids:
            all_in = [(c, cc_lookup.get((c, t), 0.0)) for c in crit_ids if c != t]
            inc = incoming_cc.get(t, [])
            e2, w2 = _validate_sparse_optimality_for_target(
                target_id=t, incoming_edges=inc, all_candidates=all_in, min_in=1, soft_max_in=3, hard_max_in=5
            )
            errors.extend([f"edges_cc optimality: {x}" for x in e2])
            warnings.extend([f"edges_cc optimality: {x}" for x in w2])

    return {"errors": errors, "warnings": warnings, "stats": stats}


# ============================================================
# Deterministic fix layer (NEW): enforces validator pass
# ============================================================
def _dedupe_list_of_dicts(items: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for it in items:
        if not isinstance(it, dict):
            continue
        k = _safe_str(it.get(key, "")).strip()
        if not k or k in seen:
            continue
        seen.add(k)
        it[key] = k
        out.append(it)
    return out


def _ensure_measurement_defaults(meas: Dict[str, Any], *, label: str, prompts_per_day: int) -> None:
    # Fill missing required fields defensively (should rarely happen due to schema strictness)
    if not _safe_str(meas.get("channel", "")).strip():
        meas["channel"] = "smartphone_app"
    if not _safe_str(meas.get("item_or_signal", "")).strip():
        meas["item_or_signal"] = f"Rate {label} (1–9)."
    if not _safe_str(meas.get("response_scale_or_unit", "")).strip():
        meas["response_scale_or_unit"] = "1–9 Likert"
    if not _safe_str(meas.get("recall_window", "")).strip():
        meas["recall_window"] = "since last prompt"
    if not _safe_str(meas.get("expected_within_person_variability", "")).strip():
        meas["expected_within_person_variability"] = "unknown"
    if meas.get("notes", None) is None:
        meas["notes"] = ""

    # sampling_per_day: keep as-is if valid; otherwise set safe default
    try:
        spd = int(meas.get("sampling_per_day", 0) or 0)
    except Exception:
        spd = 0
    if spd <= 0:
        meas["sampling_per_day"] = max(1, int(prompts_per_day))


def _enforce_design_coherence(model: Dict[str, Any]) -> None:
    dr = model.get("design_recommendations", {}) or {}
    model["design_recommendations"] = dr

    # study_days >= 7
    try:
        study_days = int(dr.get("study_days", 7) or 7)
    except Exception:
        study_days = 7
    if study_days < 7:
        study_days = 7
    dr["study_days"] = study_days

    me = dr.get("momentary_ema", {}) or {}
    dr["momentary_ema"] = me

    try:
        prompts_per_day = int(me.get("prompts_per_day", 1) or 1)
    except Exception:
        prompts_per_day = 1
    if prompts_per_day < 1:
        prompts_per_day = 1

    # Ensure at least 30 observations possible (validator requires min_total_observations_target >=30 and <= expected)
    dd = dr.get("daily_diary", {}) or {}
    dr["daily_diary"] = dd
    dd_enabled = bool(dd.get("enabled", False))

    # enable daily diary if any variable needs it (checked later), but also may help reach >=30
    # We'll do a first pass: if prompts/day too small to hit >=30, increase prompts/day.
    # total_expected_observations lower bound (daily diary adds study_days)
    base_total = study_days * prompts_per_day + (study_days if dd_enabled else 0)
    if base_total < 30:
        # If daily diary off, enabling it is less intrusive than huge prompts/day
        if not dd_enabled:
            dd_enabled = True
        # Recompute with diary enabled
        base_total = study_days * prompts_per_day + study_days
        if base_total < 30:
            # increase prompts/day minimally
            needed = 30 - study_days
            prompts_per_day = max(prompts_per_day, int((needed + study_days - 1) // study_days))

    me["prompts_per_day"] = prompts_per_day

    # time blocks: ensure array exists
    tbl = me.get("time_blocks_local", None)
    if not isinstance(tbl, list):
        tbl = []
    if len(tbl) == 0:
        # simple defaults; not validated for length, but make it human-usable
        if prompts_per_day == 1:
            tbl = ["09:00-21:00"]
        elif prompts_per_day == 2:
            tbl = ["09:00-13:00", "13:00-21:00"]
        elif prompts_per_day == 3:
            tbl = ["09:00-12:00", "12:00-16:00", "16:00-21:00"]
        elif prompts_per_day == 4:
            tbl = ["09:00-11:30", "11:30-14:00", "14:00-16:30", "16:30-21:00"]
        else:
            tbl = ["09:00-21:00"] * prompts_per_day
    me["time_blocks_local"] = [str(x) for x in tbl]

    if not _safe_str(me.get("schedule_type", "")).strip():
        me["schedule_type"] = "semi_random_within_blocks"
    if not _safe_str(me.get("recall_window", "")).strip():
        me["recall_window"] = "since last prompt"
    if me.get("notes", None) is None:
        me["notes"] = ""

    me["expected_total_prompts"] = int(study_days * prompts_per_day)

    # Daily diary
    dd["enabled"] = bool(dd_enabled)
    if dd["enabled"]:
        dd["entries_per_day"] = 1
        dd["expected_total_entries"] = int(study_days)
        anchors = dd.get("anchor_times_local", None)
        if not isinstance(anchors, list) or len(anchors) == 0:
            dd["anchor_times_local"] = ["21:30"]
        else:
            dd["anchor_times_local"] = [str(x) for x in anchors]
        if dd.get("notes", None) is None:
            dd["notes"] = ""
    else:
        dd["entries_per_day"] = 0
        dd["expected_total_entries"] = 0
        dd["anchor_times_local"] = []
        if dd.get("notes", None) is None:
            dd["notes"] = ""

    # Passive streams block
    ps = dr.get("passive_streams", {}) or {}
    dr["passive_streams"] = ps
    if ps.get("enabled", None) is None:
        ps["enabled"] = True
    if not _safe_str(ps.get("aggregation_strategy", "")).strip():
        ps["aggregation_strategy"] = "per_prompt_for_momentary_streams_and_daily_for_daily_streams"
    if ps.get("notes", None) is None:
        ps["notes"] = ""

    # Compute and set minimum_total_observations_target consistently
    total_expected = int(me["expected_total_prompts"] + (dd["expected_total_entries"] if dd["enabled"] else 0))
    try:
        min_target = int(dr.get("minimum_total_observations_target", 30) or 30)
    except Exception:
        min_target = 30
    if min_target < 30:
        min_target = 30
    if min_target > total_expected:
        # Set to total_expected to satisfy validator (and ensure total_expected >= 30 via prompts/day logic above)
        min_target = total_expected
    dr["minimum_total_observations_target"] = int(min_target)

    for k in ["gvar_readiness_notes", "variance_support_notes", "missingness_risk_notes"]:
        if dr.get(k, None) is None:
            dr[k] = ""


def _enforce_sampling_consistency(model: Dict[str, Any]) -> None:
    dr = model.get("design_recommendations", {}) or {}
    me = dr.get("momentary_ema", {}) or {}
    prompts_per_day = int(me.get("prompts_per_day", 1) or 1)

    crit = model.get("criteria_variables", []) or []
    pred = model.get("predictor_variables", []) or []
    if not isinstance(crit, list):
        crit = []
    if not isinstance(pred, list):
        pred = []

    daily_needed = False
    momentary_recall = _safe_str(me.get("recall_window", "")).strip() or "since last prompt"

    for arr in (crit, pred):
        for v in arr:
            if not isinstance(v, dict):
                continue
            label = _safe_str(v.get("label", "")).strip() or _safe_str(v.get("var_id", "")).strip() or "this"
            meas = v.get("measurement", {}) or {}
            if not isinstance(meas, dict):
                meas = {}
                v["measurement"] = meas

            _ensure_measurement_defaults(meas, label=label, prompts_per_day=prompts_per_day)

            atype = _safe_str(meas.get("assessment_type", "")).strip()

            if atype in ("since_last_prompt", "momentary", "passive_aggregated_per_prompt"):
                meas["sampling_per_day"] = int(prompts_per_day)
                if not _safe_str(meas.get("recall_window", "")).strip():
                    meas["recall_window"] = momentary_recall
                if atype == "momentary" and momentary_recall.lower().strip() == "since last prompt":
                    meas["recall_window"] = "right now"
            elif atype in ("morning_diary", "end_of_day", "passive_daily_aggregate"):
                daily_needed = True
                meas["sampling_per_day"] = 1
                if not _safe_str(meas.get("recall_window", "")).strip():
                    meas["recall_window"] = "today" if atype == "end_of_day" else "last night / since waking"
            else:
                # passive_continuous / other: keep sampling_per_day >=1
                try:
                    spd = int(meas.get("sampling_per_day", 1) or 1)
                except Exception:
                    spd = 1
                meas["sampling_per_day"] = max(1, spd)

    # If any variable requires daily diary, enable and set coherent fields
    if daily_needed:
        dd = dr.get("daily_diary", {}) or {}
        dr["daily_diary"] = dd
        dd["enabled"] = True
        study_days = int(dr.get("study_days", 7) or 7)
        dd["entries_per_day"] = 1
        dd["expected_total_entries"] = int(study_days)
        anchors = dd.get("anchor_times_local", None)
        if not isinstance(anchors, list) or len(anchors) == 0:
            dd["anchor_times_local"] = ["21:30"]
        if dd.get("notes", None) is None:
            dd["notes"] = ""

        # also adjust minimum_total_observations_target if needed
        _enforce_design_coherence(model)


def _rebuild_dense_grids(model: Dict[str, Any]) -> None:
    crit = model.get("criteria_variables", []) or []
    pred = model.get("predictor_variables", []) or []
    if not isinstance(crit, list):
        crit = []
    if not isinstance(pred, list):
        pred = []

    # Deduplicate variables by var_id to avoid validator duplicate errors
    model["criteria_variables"] = _dedupe_list_of_dicts(crit, "var_id")
    model["predictor_variables"] = _dedupe_list_of_dicts(pred, "var_id")

    crit_ids = [_safe_str(x.get("var_id", "")).strip() for x in model["criteria_variables"] if isinstance(x, dict)]
    pred_ids = [_safe_str(x.get("var_id", "")).strip() for x in model["predictor_variables"] if isinstance(x, dict)]

    # Build existing lookup (permissive) then reconstruct full deterministic lists
    existing_pc = model.get("predictor_criterion_relevance", []) or []
    existing_pp = model.get("predictor_predictor_relevance", []) or []
    existing_cc = model.get("criterion_criterion_relevance", []) or []

    pc_lookup: Dict[Tuple[str, str], float] = {}
    if isinstance(existing_pc, list):
        for r in existing_pc:
            if not isinstance(r, dict):
                continue
            p = _safe_str(r.get("predictor_var_id", "")).strip()
            c = _safe_str(r.get("criterion_var_id", "")).strip()
            s = _safe_str(r.get("relevance_score_0_1_comma5", "")).strip()
            v = _comma5_to_float(s)
            if p and c and v is not None:
                pc_lookup[(p, c)] = float(round(_clamp01(v), 5))

    pp_lookup: Dict[Tuple[str, str], float] = {}
    if isinstance(existing_pp, list):
        for r in existing_pp:
            if not isinstance(r, dict):
                continue
            a = _safe_str(r.get("from_predictor_var_id", "")).strip()
            b = _safe_str(r.get("to_predictor_var_id", "")).strip()
            s = _safe_str(r.get("relevance_score_0_1_comma5", "")).strip()
            v = _comma5_to_float(s)
            if a and b and a != b and v is not None:
                pp_lookup[(a, b)] = float(round(_clamp01(v), 5))

    cc_lookup: Dict[Tuple[str, str], float] = {}
    if isinstance(existing_cc, list):
        for r in existing_cc:
            if not isinstance(r, dict):
                continue
            a = _safe_str(r.get("from_criterion_var_id", "")).strip()
            b = _safe_str(r.get("to_criterion_var_id", "")).strip()
            s = _safe_str(r.get("relevance_score_0_1_comma5", "")).strip()
            v = _comma5_to_float(s)
            if a and b and a != b and v is not None:
                cc_lookup[(a, b)] = float(round(_clamp01(v), 5))

    # Rebuild FULL grids
    pc_out: List[Dict[str, Any]] = []
    for p in pred_ids:
        for c in crit_ids:
            v = pc_lookup.get((p, c), 0.0)
            pc_out.append({"predictor_var_id": p, "criterion_var_id": c, "relevance_score_0_1_comma5": _format_comma5(v)})

    pp_out: List[Dict[str, Any]] = []
    for a in pred_ids:
        for b in pred_ids:
            if a == b:
                continue
            v = pp_lookup.get((a, b), 0.0)
            pp_out.append({"from_predictor_var_id": a, "to_predictor_var_id": b, "relevance_score_0_1_comma5": _format_comma5(v)})

    cc_out: List[Dict[str, Any]] = []
    for a in crit_ids:
        for b in crit_ids:
            if a == b:
                continue
            v = cc_lookup.get((a, b), 0.0)
            cc_out.append({"from_criterion_var_id": a, "to_criterion_var_id": b, "relevance_score_0_1_comma5": _format_comma5(v)})

    model["predictor_criterion_relevance"] = pc_out
    model["predictor_predictor_relevance"] = pp_out
    model["criterion_criterion_relevance"] = cc_out


def _build_dense_lookups_from_model(model: Dict[str, Any]) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:
    pc_lookup: Dict[Tuple[str, str], float] = {}
    for r in (model.get("predictor_criterion_relevance", []) or []):
        if not isinstance(r, dict):
            continue
        p = _safe_str(r.get("predictor_var_id", "")).strip()
        c = _safe_str(r.get("criterion_var_id", "")).strip()
        s = _safe_str(r.get("relevance_score_0_1_comma5", "")).strip()
        v = _comma5_to_float(s)
        if p and c and v is not None:
            pc_lookup[(p, c)] = float(round(_clamp01(v), 5))

    pp_lookup: Dict[Tuple[str, str], float] = {}
    for r in (model.get("predictor_predictor_relevance", []) or []):
        if not isinstance(r, dict):
            continue
        a = _safe_str(r.get("from_predictor_var_id", "")).strip()
        b = _safe_str(r.get("to_predictor_var_id", "")).strip()
        s = _safe_str(r.get("relevance_score_0_1_comma5", "")).strip()
        v = _comma5_to_float(s)
        if a and b and a != b and v is not None:
            pp_lookup[(a, b)] = float(round(_clamp01(v), 5))

    cc_lookup: Dict[Tuple[str, str], float] = {}
    for r in (model.get("criterion_criterion_relevance", []) or []):
        if not isinstance(r, dict):
            continue
        a = _safe_str(r.get("from_criterion_var_id", "")).strip()
        b = _safe_str(r.get("to_criterion_var_id", "")).strip()
        s = _safe_str(r.get("relevance_score_0_1_comma5", "")).strip()
        v = _comma5_to_float(s)
        if a and b and a != b and v is not None:
            cc_lookup[(a, b)] = float(round(_clamp01(v), 5))

    return pc_lookup, pp_lookup, cc_lookup


def _rebuild_sparse_edges_from_dense(model: Dict[str, Any]) -> None:
    crit_ids = [_safe_str(x.get("var_id", "")).strip() for x in (model.get("criteria_variables", []) or []) if isinstance(x, dict)]
    pred_ids = [_safe_str(x.get("var_id", "")).strip() for x in (model.get("predictor_variables", []) or []) if isinstance(x, dict)]

    pc_lookup, pp_lookup, cc_lookup = _build_dense_lookups_from_model(model)

    # capture any existing metadata to preserve where possible
    meta_pc: Dict[Tuple[str, str], Dict[str, str]] = {}
    for e in (model.get("edges", []) or []):
        if not isinstance(e, dict):
            continue
        fr = _safe_str(e.get("from_predictor_var_id", "")).strip()
        to = _safe_str(e.get("to_criterion_var_id", "")).strip()
        if fr and to:
            meta_pc[(fr, to)] = {
                "expected_sign": _safe_str(e.get("expected_sign", "")).strip() or "unknown",
                "relation_interpretation": _safe_str(e.get("relation_interpretation", "")).strip() or "unknown",
                "lag_spec": _safe_str(e.get("lag_spec", "")).strip(),
                "notes": _safe_str(e.get("notes", "")).strip(),
            }

    meta_pp: Dict[Tuple[str, str], Dict[str, str]] = {}
    for e in (model.get("edges_pp", []) or []):
        if not isinstance(e, dict):
            continue
        fr = _safe_str(e.get("from_predictor_var_id", "")).strip()
        to = _safe_str(e.get("to_predictor_var_id", "")).strip()
        if fr and to and fr != to:
            meta_pp[(fr, to)] = {
                "expected_sign": _safe_str(e.get("expected_sign", "")).strip() or "unknown",
                "relation_interpretation": _safe_str(e.get("relation_interpretation", "")).strip() or "unknown",
                "lag_spec": _safe_str(e.get("lag_spec", "")).strip(),
                "notes": _safe_str(e.get("notes", "")).strip(),
            }

    meta_cc: Dict[Tuple[str, str], Dict[str, str]] = {}
    for e in (model.get("edges_cc", []) or []):
        if not isinstance(e, dict):
            continue
        fr = _safe_str(e.get("from_criterion_var_id", "")).strip()
        to = _safe_str(e.get("to_criterion_var_id", "")).strip()
        if fr and to and fr != to:
            meta_cc[(fr, to)] = {
                "expected_sign": _safe_str(e.get("expected_sign", "")).strip() or "unknown",
                "relation_interpretation": _safe_str(e.get("relation_interpretation", "")).strip() or "unknown",
                "lag_spec": _safe_str(e.get("lag_spec", "")).strip(),
                "notes": _safe_str(e.get("notes", "")).strip(),
            }

    # Lag defaults aligned to momentary scale
    lag_default = "0-6h"

    def _make_edge_pc(fr: str, to: str, score: float) -> Dict[str, Any]:
        m = meta_pc.get((fr, to), {})
        lag = (m.get("lag_spec", "") or "").strip() or lag_default
        notes = (m.get("notes", "") or "").strip() or "Auto-selected top incoming edge from dense grid (deterministic fix)."
        return {
            "from_predictor_var_id": fr,
            "to_criterion_var_id": to,
            "expected_sign": m.get("expected_sign", "unknown"),
            "relation_interpretation": m.get("relation_interpretation", "unknown"),
            "lag_spec": lag,
            "estimated_relevance_0_1": float(round(_clamp01(score), 5)),
            "estimated_relevance_tier": _tier_from_score(score),
            "notes": notes,
        }

    def _make_edge_pp(fr: str, to: str, score: float) -> Dict[str, Any]:
        m = meta_pp.get((fr, to), {})
        lag = (m.get("lag_spec", "") or "").strip() or lag_default
        notes = (m.get("notes", "") or "").strip() or "Auto-selected top incoming edge from dense grid (deterministic fix)."
        return {
            "from_predictor_var_id": fr,
            "to_predictor_var_id": to,
            "expected_sign": m.get("expected_sign", "unknown"),
            "relation_interpretation": m.get("relation_interpretation", "unknown"),
            "lag_spec": lag,
            "estimated_relevance_0_1": float(round(_clamp01(score), 5)),
            "estimated_relevance_tier": _tier_from_score(score),
            "notes": notes,
        }

    def _make_edge_cc(fr: str, to: str, score: float) -> Dict[str, Any]:
        m = meta_cc.get((fr, to), {})
        lag = (m.get("lag_spec", "") or "").strip() or lag_default
        notes = (m.get("notes", "") or "").strip() or "Auto-selected top incoming edge from dense grid (deterministic fix)."
        return {
            "from_criterion_var_id": fr,
            "to_criterion_var_id": to,
            "expected_sign": m.get("expected_sign", "unknown"),
            "relation_interpretation": m.get("relation_interpretation", "unknown"),
            "lag_spec": lag,
            "estimated_relevance_0_1": float(round(_clamp01(score), 5)),
            "estimated_relevance_tier": _tier_from_score(score),
            "notes": notes,
        }

    # Choose top-k incoming per target to satisfy your optimality validator
    edges_pc_out: List[Dict[str, Any]] = []
    for to in crit_ids:
        cands = [(fr, pc_lookup.get((fr, to), 0.0)) for fr in pred_ids]
        cands.sort(key=lambda x: x[1], reverse=True)
        if not cands:
            continue
        k = min(4, len(cands))
        k = max(min(2, len(cands)), k)  # prefer 2–4 if possible
        chosen = cands[:k]
        for fr, sc in chosen:
            edges_pc_out.append(_make_edge_pc(fr, to, sc))

    edges_pp_out: List[Dict[str, Any]] = []
    for to in pred_ids:
        cands = [(fr, pp_lookup.get((fr, to), 0.0)) for fr in pred_ids if fr != to]
        cands.sort(key=lambda x: x[1], reverse=True)
        if not cands:
            continue
        k = min(3, len(cands))
        k = max(1, k)
        chosen = cands[:k]
        for fr, sc in chosen:
            edges_pp_out.append(_make_edge_pp(fr, to, sc))

    edges_cc_out: List[Dict[str, Any]] = []
    for to in crit_ids:
        cands = [(fr, cc_lookup.get((fr, to), 0.0)) for fr in crit_ids if fr != to]
        cands.sort(key=lambda x: x[1], reverse=True)
        if not cands:
            continue
        k = min(3, len(cands))
        k = max(1, k)
        chosen = cands[:k]
        for fr, sc in chosen:
            edges_cc_out.append(_make_edge_cc(fr, to, sc))

    model["edges"] = edges_pc_out
    model["edges_pp"] = edges_pp_out
    model["edges_cc"] = edges_cc_out


def deterministic_fix_model(model: Dict[str, Any], *, pid: str = "") -> Dict[str, Any]:
    if not isinstance(model, dict):
        return model

    _enforce_design_coherence(model)
    _enforce_sampling_consistency(model)
    _rebuild_dense_grids(model)
    _rebuild_sparse_edges_from_dense(model)

    # one more pass: design coherence after sampling changes
    _enforce_design_coherence(model)

    if pid:
        rep = validate_observation_model(model)
        _log(pid, f"Deterministic fix validation -> errors={len(rep.get('errors', []) or [])} warnings={len(rep.get('warnings', []) or [])}")
    return model


# ============================================================
# Repair messages (unchanged)
# ============================================================
def build_repair_messages(
    *,
    original_model: Dict[str, Any],
    validation_report: Dict[str, Any],
    pseudoprofile_id: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    errors = validation_report.get("errors", []) or []
    warnings = validation_report.get("warnings", []) or []
    stats = validation_report.get("stats", {}) or {}

    instructions = (
        "You are repairing a previously generated observation model JSON.\n\n"
        "Hard constraints:\n"
        "- Output MUST strictly follow the same JSON schema.\n"
        "- Fix all validation ERRORS (including edge optimality errors).\n"
        "- Preserve the original model intent and coverage as much as possible.\n\n"
        "Important:\n"
        "- Do NOT add any additional keys not allowed by schema.\n"
        "- Ensure momentary variables have sampling_per_day == prompts_per_day and recall_window matches.\n"
        "- Ensure daily variables require daily_diary.enabled and sampling_per_day == 1.\n"
        "- Dense relevance grids MUST be complete: P->C, P->P, C->C with comma-decimal 5dp.\n"
        "- Sparse edges MUST copy estimated_relevance_0_1 from the corresponding dense grid cell (numeric float) to 5 decimals.\n"
        "- Sparse edges MUST be optimal for each target: avoid including low-score edges when higher-score alternatives exist.\n"
        "- Ensure lag_spec is non-empty for every edge.\n"
        "- Keep total nodes <= 18.\n"
    )

    payload = {
        "pseudoprofile_id": pseudoprofile_id,
        "validation_errors": errors,
        "validation_warnings": warnings,
        "validation_stats": stats,
        "original_model_json": original_model,
    }

    messages = [
        {
            "role": "user",
            "content": (
                "Repair the observation model JSON using the payload below.\n\n"
                "REPAIR_PAYLOAD_JSON:\n"
                f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
            ),
        }
    ]
    return instructions, messages


# ============================================================
# Output formatting + aggregation helpers (unchanged)
# ============================================================
def print_human_summary(model: Dict[str, Any]) -> None:
    print("\n=== MODEL SUMMARY ===")
    print(_safe_str(model.get("model_summary", "")).strip())

    print("\n=== DESIGN ===")
    dr = model.get("design_recommendations", {}) or {}
    me = dr.get("momentary_ema", {}) or {}
    dd = dr.get("daily_diary", {}) or {}
    ps = dr.get("passive_streams", {}) or {}
    print(f"study_days: {dr.get('study_days','')}")
    print(f"momentary prompts/day: {me.get('prompts_per_day','')} | time blocks: {me.get('time_blocks_local',[])} | recall: {me.get('recall_window','')}")
    print(f"daily diary enabled: {dd.get('enabled','')} | entries/day: {dd.get('entries_per_day','')} | anchors: {dd.get('anchor_times_local',[])}")
    print(f"passive streams enabled: {ps.get('enabled','')} | aggregation: {ps.get('aggregation_strategy','')}")
    print(f"min total obs target: {dr.get('minimum_total_observations_target','')}")

    print("\n=== CRITERIA VARIABLES ===")
    for c in (model.get("criteria_variables", []) or []):
        vid = c.get("var_id", "")
        lab = c.get("label", "")
        prio = c.get("include_priority", "")
        pol = c.get("polarity", "")
        meas = c.get("measurement", {}) or {}
        print(f"- [{vid}] {lab} (prio={prio}, polarity={pol})")
        print(f"    mode={meas.get('mode','')} | type={meas.get('assessment_type','')} | samp/day={meas.get('sampling_per_day','')} | scale={meas.get('response_scale_or_unit','')}")
        print(f"    item: {meas.get('item_or_signal','')}")

    print("\n=== PREDICTOR VARIABLES ===")
    for p in (model.get("predictor_variables", []) or []):
        vid = p.get("var_id", "")
        lab = p.get("label", "")
        dom = p.get("bio_psycho_social_domain", "")
        prio = p.get("include_priority", "")
        path = p.get("ontology_path", "")
        meas = p.get("measurement", {}) or {}
        print(f"- [{vid}] {lab} | {dom} (prio={prio})")
        print(f"    ontology: {path}")
        print(f"    mode={meas.get('mode','')} | type={meas.get('assessment_type','')} | samp/day={meas.get('sampling_per_day','')} | scale={meas.get('response_scale_or_unit','')}")

    pc = model.get("predictor_criterion_relevance", []) or []
    pp = model.get("predictor_predictor_relevance", []) or []
    cc = model.get("criterion_criterion_relevance", []) or []
    print("\n=== DENSE RELEVANCE (counts) ===")
    print(f"P->C pairs: {len(pc) if isinstance(pc, list) else 'n/a'}")
    print(f"P->P pairs: {len(pp) if isinstance(pp, list) else 'n/a'}")
    print(f"C->C pairs: {len(cc) if isinstance(cc, list) else 'n/a'}")

    epc = model.get("edges", []) or []
    epp = model.get("edges_pp", []) or []
    ecc = model.get("edges_cc", []) or []
    print("\n=== SPARSE EDGES (counts) ===")
    print(f"edges (P->C): {len(epc) if isinstance(epc, list) else 'n/a'}")
    print(f"edges_pp (P->P): {len(epp) if isinstance(epp, list) else 'n/a'}")
    print(f"edges_cc (C->C): {len(ecc) if isinstance(ecc, list) else 'n/a'}")


def _profile_dir(profiles_dir: str, pseudoprofile_id: str) -> str:
    return os.path.join(profiles_dir, _safe_filename(pseudoprofile_id))


def _save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _load_json(path: str) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _cache_paths_for_profile(profiles_dir: str, pseudoprofile_id: str) -> Dict[str, str]:
    pdir = _profile_dir(profiles_dir, pseudoprofile_id)
    return {
        "profile_dir": pdir,
        "input_payload": os.path.join(pdir, "input_payload.json"),
        "llm_raw": os.path.join(pdir, "llm_observation_model_raw.json"),
        "llm_final": os.path.join(pdir, "llm_observation_model_final.json"),
        "validation_report": os.path.join(pdir, "validation_report.json"),
        "config": os.path.join(pdir, "config.json"),
    }


def _load_cached_observation_model(profiles_dir: str, pseudoprofile_id: str) -> Optional[Dict[str, Any]]:
    paths = _cache_paths_for_profile(profiles_dir, pseudoprofile_id)
    if os.path.exists(paths["llm_final"]):
        cached = _load_json(paths["llm_final"])
        if isinstance(cached, dict):
            return cached
    return None


def _append_variables_long_rows(rows: List[Dict[str, Any]], pseudoprofile_id: str, model: Dict[str, Any]) -> None:
    for c in (model.get("criteria_variables", []) or []):
        meas = c.get("measurement", {}) or {}
        rows.append(
            {
                "pseudoprofile_id": pseudoprofile_id,
                "role": "criterion",
                "var_id": c.get("var_id", ""),
                "source_variable_id": c.get("source_variable_id", ""),
                "label": c.get("label", ""),
                "ontology_path": c.get("criterion_path", ""),
                "domain": "",
                "priority": c.get("include_priority", ""),
                "polarity": c.get("polarity", ""),
                "measurement_mode": meas.get("mode", ""),
                "measurement_channel": meas.get("channel", ""),
                "assessment_type": meas.get("assessment_type", ""),
                "measurement_item_or_signal": meas.get("item_or_signal", ""),
                "measurement_scale_or_unit": meas.get("response_scale_or_unit", ""),
                "sampling_per_day": meas.get("sampling_per_day", ""),
                "recall_window": meas.get("recall_window", ""),
                "expected_variability": meas.get("expected_within_person_variability", ""),
            }
        )

    for p in (model.get("predictor_variables", []) or []):
        meas = p.get("measurement", {}) or {}
        rows.append(
            {
                "pseudoprofile_id": pseudoprofile_id,
                "role": "predictor",
                "var_id": p.get("var_id", ""),
                "source_variable_id": "",
                "label": p.get("label", ""),
                "ontology_path": p.get("ontology_path", ""),
                "domain": p.get("bio_psycho_social_domain", ""),
                "priority": p.get("include_priority", ""),
                "polarity": "",
                "measurement_mode": meas.get("mode", ""),
                "measurement_channel": meas.get("channel", ""),
                "assessment_type": meas.get("assessment_type", ""),
                "measurement_item_or_signal": meas.get("item_or_signal", ""),
                "measurement_scale_or_unit": meas.get("response_scale_or_unit", ""),
                "sampling_per_day": meas.get("sampling_per_day", ""),
                "recall_window": meas.get("recall_window", ""),
                "expected_variability": meas.get("expected_within_person_variability", ""),
            }
        )


def _append_edges_pc_long_rows(rows: List[Dict[str, Any]], pseudoprofile_id: str, model: Dict[str, Any]) -> None:
    for e in (model.get("edges", []) or []):
        rows.append(
            {
                "pseudoprofile_id": pseudoprofile_id,
                "edge_type": "P_to_C",
                "from_var_id": e.get("from_predictor_var_id", ""),
                "to_var_id": e.get("to_criterion_var_id", ""),
                "expected_sign": e.get("expected_sign", ""),
                "relation_interpretation": e.get("relation_interpretation", ""),
                "lag_spec": e.get("lag_spec", ""),
                "estimated_relevance_0_1": e.get("estimated_relevance_0_1", ""),
                "estimated_relevance_tier": e.get("estimated_relevance_tier", ""),
                "notes": e.get("notes", ""),
            }
        )


def _append_edges_pp_long_rows(rows: List[Dict[str, Any]], pseudoprofile_id: str, model: Dict[str, Any]) -> None:
    for e in (model.get("edges_pp", []) or []):
        rows.append(
            {
                "pseudoprofile_id": pseudoprofile_id,
                "edge_type": "P_to_P",
                "from_var_id": e.get("from_predictor_var_id", ""),
                "to_var_id": e.get("to_predictor_var_id", ""),
                "expected_sign": e.get("expected_sign", ""),
                "relation_interpretation": e.get("relation_interpretation", ""),
                "lag_spec": e.get("lag_spec", ""),
                "estimated_relevance_0_1": e.get("estimated_relevance_0_1", ""),
                "estimated_relevance_tier": e.get("estimated_relevance_tier", ""),
                "notes": e.get("notes", ""),
            }
        )


def _append_edges_cc_long_rows(rows: List[Dict[str, Any]], pseudoprofile_id: str, model: Dict[str, Any]) -> None:
    for e in (model.get("edges_cc", []) or []):
        rows.append(
            {
                "pseudoprofile_id": pseudoprofile_id,
                "edge_type": "C_to_C",
                "from_var_id": e.get("from_criterion_var_id", ""),
                "to_var_id": e.get("to_criterion_var_id", ""),
                "expected_sign": e.get("expected_sign", ""),
                "relation_interpretation": e.get("relation_interpretation", ""),
                "lag_spec": e.get("lag_spec", ""),
                "estimated_relevance_0_1": e.get("estimated_relevance_0_1", ""),
                "estimated_relevance_tier": e.get("estimated_relevance_tier", ""),
                "notes": e.get("notes", ""),
            }
        )


def _append_pc_relevance_long_rows(rows: List[Dict[str, Any]], pseudoprofile_id: str, model: Dict[str, Any]) -> None:
    for r in (model.get("predictor_criterion_relevance", []) or []):
        s = _safe_str(r.get("relevance_score_0_1_comma5", "")).strip()
        rows.append(
            {
                "pseudoprofile_id": pseudoprofile_id,
                "matrix": "P_to_C",
                "from_var_id": _safe_str(r.get("predictor_var_id", "")).strip(),
                "to_var_id": _safe_str(r.get("criterion_var_id", "")).strip(),
                "relevance_score_0_1_comma5": s,
                "relevance_score_0_1_float": _comma5_to_float(s),
            }
        )


def _append_pp_relevance_long_rows(rows: List[Dict[str, Any]], pseudoprofile_id: str, model: Dict[str, Any]) -> None:
    for r in (model.get("predictor_predictor_relevance", []) or []):
        s = _safe_str(r.get("relevance_score_0_1_comma5", "")).strip()
        rows.append(
            {
                "pseudoprofile_id": pseudoprofile_id,
                "matrix": "P_to_P",
                "from_var_id": _safe_str(r.get("from_predictor_var_id", "")).strip(),
                "to_var_id": _safe_str(r.get("to_predictor_var_id", "")).strip(),
                "relevance_score_0_1_comma5": s,
                "relevance_score_0_1_float": _comma5_to_float(s),
            }
        )


def _append_cc_relevance_long_rows(rows: List[Dict[str, Any]], pseudoprofile_id: str, model: Dict[str, Any]) -> None:
    for r in (model.get("criterion_criterion_relevance", []) or []):
        s = _safe_str(r.get("relevance_score_0_1_comma5", "")).strip()
        rows.append(
            {
                "pseudoprofile_id": pseudoprofile_id,
                "matrix": "C_to_C",
                "from_var_id": _safe_str(r.get("from_criterion_var_id", "")).strip(),
                "to_var_id": _safe_str(r.get("to_criterion_var_id", "")).strip(),
                "relevance_score_0_1_comma5": s,
                "relevance_score_0_1_float": _comma5_to_float(s),
            }
        )


def _append_validations_long_rows(rows: List[Dict[str, Any]], pseudoprofile_id: str, report: Dict[str, Any]) -> None:
    stats = report.get("stats", {}) or {}
    errors = report.get("errors", []) or []
    warnings = report.get("warnings", []) or []
    rows.append(
        {
            "pseudoprofile_id": pseudoprofile_id,
            "n_errors": len(errors),
            "n_warnings": len(warnings),
            "errors_joined": " || ".join([str(x) for x in errors])[:5000],
            "warnings_joined": " || ".join([str(x) for x in warnings])[:5000],
            **{f"stat_{k}": v for k, v in stats.items()},
        }
    )


# ============================================================
# Worker
# ============================================================

def _write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def _extract_case_payload_from_messages(messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    build_llm_messages embeds the payload as JSON inside messages[0]["content"].
    We extract it to persist input_payload.json deterministically.
    """
    if not messages or not isinstance(messages, list):
        return None
    m0 = messages[0] if isinstance(messages[0], dict) else None
    if not m0:
        return None
    content = m0.get("content", "")
    if not isinstance(content, str):
        return None
    marker = "CASE_PAYLOAD_JSON:"
    idx = content.find(marker)
    if idx == -1:
        return None
    json_str = content[idx + len(marker):].strip()
    try:
        return json.loads(json_str)
    except Exception:
        # permissive salvage
        start = json_str.find("{")
        end = json_str.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(json_str[start:end + 1])
            except Exception:
                return None
        return None


def process_one_pseudoprofile(
    pseudoprofile_id: str,
    *,
    mapped_criterions_path: str,
    hyde_dense_profiles_path: str,
    llm_mapping_ranks_path: str,
    ontology_path: str,
    profiles_dir: str,
    llm_model: str,
    n_criteria: str,
    n_predictors: str,
    prompt_top_hyde: int,
    prompt_top_mapping_global: int,
    prompt_top_mapping_per_criterion: int,
    max_ontology_chars: int,
    use_cache: bool,
    auto_repair: bool,
    max_repair_attempts: int,
    deterministic_fix: bool,
    max_fix_passes: int,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "pseudoprofile_id": str,
        "status": "ok"|"error",
        "used_cache": bool,
        "model": Optional[dict],
        "validation_report": Optional[dict],
        "error_type": Optional[str],
        "error_message": Optional[str],
        "traceback": Optional[str],
      }
    """
    pid = str(pseudoprofile_id)
    paths = _cache_paths_for_profile(profiles_dir, pid)
    _ensure_dir(paths["profile_dir"])

    # Per-profile config (lightweight; helps debugging)
    profile_cfg = {
        "pseudoprofile_id": pid,
        "llm_model": llm_model,
        "n_criteria": n_criteria,
        "n_predictors": n_predictors,
        "prompt_top_hyde": int(prompt_top_hyde),
        "prompt_top_mapping_global": int(prompt_top_mapping_global),
        "prompt_top_mapping_per_criterion": int(prompt_top_mapping_per_criterion),
        "max_ontology_chars": int(max_ontology_chars),
        "use_cache": bool(use_cache),
        "auto_repair": bool(auto_repair),
        "max_repair_attempts": int(max_repair_attempts),
        "deterministic_fix": bool(deterministic_fix),
        "max_fix_passes": int(max_fix_passes),
        "inputs": {
            "mapped_criterions_path": mapped_criterions_path,
            "hyde_dense_profiles_path": hyde_dense_profiles_path,
            "llm_mapping_ranks_path": llm_mapping_ranks_path,
            "ontology_path": ontology_path,
        },
    }
    _save_json(paths["config"], profile_cfg)

    try:
        # Cache fast-path
        used_cache = False
        model: Optional[Dict[str, Any]] = None
        report: Optional[Dict[str, Any]] = None

        if use_cache:
            cached = _load_cached_observation_model(profiles_dir, pid)
            if isinstance(cached, dict):
                used_cache = True
                model = cached
                report = _load_json(paths["validation_report"]) if os.path.exists(paths["validation_report"]) else None
                if not isinstance(report, dict):
                    report = validate_observation_model(model)
                    _save_json(paths["validation_report"], report)

        if model is None:
            # Load inputs
            profile = load_profile_input(
                mapped_criterions_path=mapped_criterions_path,
                ontology_path=ontology_path,
                pseudoprofile_id=pid,
                max_ontology_chars=max_ontology_chars,
            )
            hyde = load_hyde_signals_for_profile(
                hyde_dense_profiles_csv=hyde_dense_profiles_path,
                pseudoprofile_id=pid,
                top_n=int(prompt_top_hyde),
            )
            mapping = load_llm_mapping_ranks_for_profile(
                mapping_ranks_csv=llm_mapping_ranks_path,
                pseudoprofile_id=pid,
                top_global=int(prompt_top_mapping_global),
                top_per_criterion=int(prompt_top_mapping_per_criterion),
            )

            instructions, messages = build_llm_messages(
                profile=profile,
                hyde=hyde,
                mapping=mapping,
                n_criteria=str(n_criteria),
                n_predictors=str(n_predictors),
            )

            # Persist payload for reproducibility
            payload = _extract_case_payload_from_messages(messages)
            if payload is not None:
                _save_json(paths["input_payload"], payload)
            else:
                # best-effort fallback
                _save_json(paths["input_payload"], {"pseudoprofile_id": pid, "note": "payload extraction failed"})

            # Create a per-worker client to avoid cross-thread surprises
            client = OpenAI()

            # Initial generation
            raw_model = call_llm_structured(
                client=client,
                llm_model=llm_model,
                instructions=instructions,
                messages=messages,
                pseudoprofile_id=pid,
            )
            _save_json(paths["llm_raw"], raw_model)

            model = raw_model
            report = validate_observation_model(model)

            # Auto-repair loop (LLM)
            if auto_repair and report.get("errors"):
                for attempt in range(1, int(max_repair_attempts) + 1):
                    _log(pid, f"Auto-repair: attempt {attempt}/{max_repair_attempts} | errors={len(report.get('errors', []) or [])}")
                    rep_instructions, rep_messages = build_repair_messages(
                        original_model=model,
                        validation_report=report,
                        pseudoprofile_id=pid,
                    )
                    repaired = call_llm_structured(
                        client=client,
                        llm_model=llm_model,
                        instructions=rep_instructions,
                        messages=rep_messages,
                        pseudoprofile_id=pid,
                    )
                    model = repaired
                    report = validate_observation_model(model)
                    if not (report.get("errors") or []):
                        break

        # Deterministic fix layer (guarantee validator pass)
        if deterministic_fix and isinstance(model, dict):
            for pass_i in range(1, int(max_fix_passes) + 1):
                model = deterministic_fix_model(model, pid=pid)
                report = validate_observation_model(model)
                if not (report.get("errors") or []):
                    break
                _log(pid, f"Deterministic fix pass {pass_i}/{max_fix_passes} still has errors={len(report.get('errors', []) or [])}")

        # Persist final artifacts
        if isinstance(model, dict):
            _save_json(paths["llm_final"], model)
        if isinstance(report, dict):
            _save_json(paths["validation_report"], report)

        return {
            "pseudoprofile_id": pid,
            "status": "ok",
            "used_cache": bool(used_cache),
            "model": model,
            "validation_report": report,
            "error_type": None,
            "error_message": None,
            "traceback": None,
        }

    except Exception as e:
        tb = traceback.format_exc()
        _log(pid, f"WORKER ERROR: {type(e).__name__}: {e}")
        # still write a validation_report placeholder so downstream tooling can find something
        try:
            _save_json(paths["validation_report"], {"errors": [f"{type(e).__name__}: {e}"], "warnings": [], "stats": {}})
        except Exception:
            pass
        return {
            "pseudoprofile_id": pid,
            "status": "error",
            "used_cache": False,
            "model": None,
            "validation_report": None,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": tb,
        }


# ============================================================
# Main
# ============================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Construct initial observation model (criterion + predictors) for gVAR-ready EMA/digital phenotyping.")

    p.add_argument("--mapped_criterions_path", type=str, default=DEFAULT_MAPPED_CRITERIONS_PATH)
    p.add_argument("--hyde_dense_profiles_path", type=str, default=DEFAULT_HYDE_DENSE_PROFILES_PATH)
    p.add_argument("--llm_mapping_ranks_path", type=str, default=DEFAULT_LLM_MAPPING_RANKS_PATH)
    p.add_argument("--ontology_path", type=str, default=DEFAULT_HIGH_LEVEL_ONTOLOGY_PATH)
    p.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR)

    p.add_argument("--run_id", type=str, default="")

    p.add_argument("--pseudoprofile_id", type=str, default="")
    p.add_argument("--n_criteria", type=str, default=DEFAULT_N_CRITERIA)
    p.add_argument("--n_predictors", type=str, default=DEFAULT_N_PREDICTORS)

    p.add_argument("--prompt_top_hyde", type=int, default=DEFAULT_PROMPT_TOP_HYDE)
    p.add_argument("--prompt_top_mapping_global", type=int, default=DEFAULT_PROMPT_TOP_MAPPING_GLOBAL)
    p.add_argument("--prompt_top_mapping_per_criterion", type=int, default=DEFAULT_PROMPT_TOP_MAPPING_PER_CRITERION)

    p.add_argument("--llm_model", type=str, default=DEFAULT_LLM_MODEL_NAME)

    p.add_argument("--max_workers", type=int, default=DEFAULT_MAX_WORKERS)
    p.add_argument("--max_ontology_chars", type=int, default=DEFAULT_MAX_ONTOLOGY_CHARS)

    # Sampling (default ON to match your previous testing behavior)
    p.add_argument("--enable_sampling", action=argparse.BooleanOptionalAction, default=DEFAULT_ENABLE_SAMPLING)
    p.add_argument("--sample_n", type=int, default=DEFAULT_SAMPLE_N)
    p.add_argument("--sample_seed", type=int, default=DEFAULT_SAMPLE_SEED)

    # Cache
    p.add_argument("--use_cache", action=argparse.BooleanOptionalAction, default=DEFAULT_USE_CACHE)

    # Repair controls
    p.add_argument("--auto_repair", action=argparse.BooleanOptionalAction, default=DEFAULT_AUTO_REPAIR)
    p.add_argument("--max_repair_attempts", type=int, default=DEFAULT_MAX_REPAIR_ATTEMPTS)

    # Deterministic fix
    p.add_argument("--deterministic_fix", action=argparse.BooleanOptionalAction, default=DEFAULT_DETERMINISTIC_FIX)
    p.add_argument("--max_fix_passes", type=int, default=DEFAULT_MAX_FIX_PASSES)

    # Optional: print summaries
    p.add_argument("--print_summary", action=argparse.BooleanOptionalAction, default=False)

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Export it before running: export OPENAI_API_KEY='...'")

    run_id = _safe_filename(args.run_id.strip()) if args.run_id.strip() else _now_stamp()

    run_root = _ensure_dir(os.path.join(args.results_dir, "runs", run_id))
    profiles_dir = _ensure_dir(os.path.join(run_root, "profiles"))

    # Run-level config
    run_cfg = {
        "run_id": run_id,
        "created_at_local": datetime.now().isoformat(timespec="seconds"),
        "llm_model": args.llm_model,
        "paths": {
            "mapped_criterions_path": args.mapped_criterions_path,
            "hyde_dense_profiles_path": args.hyde_dense_profiles_path,
            "llm_mapping_ranks_path": args.llm_mapping_ranks_path,
            "ontology_path": args.ontology_path,
            "results_dir": args.results_dir,
            "run_root": run_root,
            "profiles_dir": profiles_dir,
        },
        "params": {
            "pseudoprofile_id": args.pseudoprofile_id,
            "n_criteria": args.n_criteria,
            "n_predictors": args.n_predictors,
            "prompt_top_hyde": args.prompt_top_hyde,
            "prompt_top_mapping_global": args.prompt_top_mapping_global,
            "prompt_top_mapping_per_criterion": args.prompt_top_mapping_per_criterion,
            "max_workers": args.max_workers,
            "max_ontology_chars": args.max_ontology_chars,
            "enable_sampling": args.enable_sampling,
            "sample_n": args.sample_n,
            "sample_seed": args.sample_seed,
            "use_cache": args.use_cache,
            "auto_repair": args.auto_repair,
            "max_repair_attempts": args.max_repair_attempts,
            "deterministic_fix": args.deterministic_fix,
            "max_fix_passes": args.max_fix_passes,
            "print_summary": args.print_summary,
        },
    }
    _save_json(os.path.join(run_root, "config.json"), run_cfg)

    # Determine pseudoprofiles to process
    if args.pseudoprofile_id.strip():
        pseudoprofile_ids = [args.pseudoprofile_id.strip()]
    else:
        all_ids = list_pseudoprofile_ids_from_mapped(args.mapped_criterions_path)
        if args.enable_sampling:
            rr = random.Random(int(args.sample_seed))
            n = max(1, int(args.sample_n))
            if n >= len(all_ids):
                pseudoprofile_ids = all_ids
            else:
                pseudoprofile_ids = sorted(rr.sample(all_ids, n))
        else:
            pseudoprofile_ids = all_ids

    _log("", f"Run {run_id} | profiles={len(pseudoprofile_ids)} | max_workers={args.max_workers} | model={args.llm_model}")

    # Aggregation buffers
    models_ok: List[Dict[str, Any]] = []
    validations_rows: List[Dict[str, Any]] = []
    errors_rows: List[Dict[str, Any]] = []

    variables_long_rows: List[Dict[str, Any]] = []
    edges_pc_long_rows: List[Dict[str, Any]] = []
    edges_pp_long_rows: List[Dict[str, Any]] = []
    edges_cc_long_rows: List[Dict[str, Any]] = []
    pc_rel_long_rows: List[Dict[str, Any]] = []
    pp_rel_long_rows: List[Dict[str, Any]] = []
    cc_rel_long_rows: List[Dict[str, Any]] = []

    # Execute
    with ThreadPoolExecutor(max_workers=int(args.max_workers)) as ex:
        futs = []
        for pid in pseudoprofile_ids:
            futs.append(
                ex.submit(
                    process_one_pseudoprofile,
                    pid,
                    mapped_criterions_path=args.mapped_criterions_path,
                    hyde_dense_profiles_path=args.hyde_dense_profiles_path,
                    llm_mapping_ranks_path=args.llm_mapping_ranks_path,
                    ontology_path=args.ontology_path,
                    profiles_dir=profiles_dir,
                    llm_model=args.llm_model,
                    n_criteria=str(args.n_criteria),
                    n_predictors=str(args.n_predictors),
                    prompt_top_hyde=int(args.prompt_top_hyde),
                    prompt_top_mapping_global=int(args.prompt_top_mapping_global),
                    prompt_top_mapping_per_criterion=int(args.prompt_top_mapping_per_criterion),
                    max_ontology_chars=int(args.max_ontology_chars),
                    use_cache=bool(args.use_cache),
                    auto_repair=bool(args.auto_repair),
                    max_repair_attempts=int(args.max_repair_attempts),
                    deterministic_fix=bool(args.deterministic_fix),
                    max_fix_passes=int(args.max_fix_passes),
                )
            )

        for fut in as_completed(futs):
            res = fut.result()
            pid = res.get("pseudoprofile_id", "")

            if res.get("status") != "ok" or not isinstance(res.get("model", None), dict):
                errors_rows.append(
                    {
                        "pseudoprofile_id": pid,
                        "status": res.get("status", "error"),
                        "error_type": res.get("error_type", ""),
                        "error_message": res.get("error_message", ""),
                        "traceback": res.get("traceback", ""),
                    }
                )
                continue

            model = res["model"]
            report = res.get("validation_report", None)
            if not isinstance(report, dict):
                report = validate_observation_model(model)

            models_ok.append(model)

            _append_validations_long_rows(validations_rows, pid, report)

            _append_variables_long_rows(variables_long_rows, pid, model)
            _append_edges_pc_long_rows(edges_pc_long_rows, pid, model)
            _append_edges_pp_long_rows(edges_pp_long_rows, pid, model)
            _append_edges_cc_long_rows(edges_cc_long_rows, pid, model)
            _append_pc_relevance_long_rows(pc_rel_long_rows, pid, model)
            _append_pp_relevance_long_rows(pp_rel_long_rows, pid, model)
            _append_cc_relevance_long_rows(cc_rel_long_rows, pid, model)

            if args.print_summary:
                _log(pid, f"USED_CACHE={res.get('used_cache', False)} | errors={len(report.get('errors', []) or [])} warnings={len(report.get('warnings', []) or [])}")
                print_human_summary(model)

    # Write run-level outputs
    # 1) observation_models.jsonl
    jsonl_path = os.path.join(run_root, "observation_models.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for m in sorted(models_ok, key=lambda x: _safe_str(x.get("pseudoprofile_id", ""))):
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # 2) long CSVs
    _write_csv(variables_long_rows, os.path.join(run_root, "variables_long.csv"))
    _write_csv(edges_pc_long_rows, os.path.join(run_root, "edges_long.csv"))
    _write_csv(edges_pp_long_rows, os.path.join(run_root, "edges_pp_long.csv"))
    _write_csv(edges_cc_long_rows, os.path.join(run_root, "edges_cc_long.csv"))
    _write_csv(pc_rel_long_rows, os.path.join(run_root, "predictor_criterion_relevance_long.csv"))
    _write_csv(pp_rel_long_rows, os.path.join(run_root, "predictor_predictor_relevance_long.csv"))
    _write_csv(cc_rel_long_rows, os.path.join(run_root, "criterion_criterion_relevance_long.csv"))
    _write_csv(validations_rows, os.path.join(run_root, "validations.csv"))
    _write_csv(errors_rows, os.path.join(run_root, "errors.csv"))

    _log("", f"DONE | ok={len(models_ok)} | errors={len(errors_rows)} | run_dir={run_root}")


if __name__ == "__main__":
    main()

#TODO: run this script with better model to see if current illogical outputs go away ; especially edge-related estimations

# NOTE: after running this script, run: 1. ontology mapper + 2. bi-partite visualizer --> then 3. pseudodata generator and 4. pseudodata visualizor
