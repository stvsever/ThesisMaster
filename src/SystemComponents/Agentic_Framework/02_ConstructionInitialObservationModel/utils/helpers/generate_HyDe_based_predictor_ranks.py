#!/usr/bin/env python3
"""
generate_HyDe_based_predictor_ranks.py

Goal
----
Given:
  - a free-text mental health / mental state description (complaint_text)
  - an LLM-decomposed, ontology-mapped set of criteria rows (from mapped_criterions.csv)
  - a high-level BioPsychoSocial predictor ontology overview (predictors_list.txt)

1) Call an LLM (gpt-5-nano/mini; no temperature) to produce expert-style candidate solution variables
   (modifiable predictors/levers) tailored to the criteria.

2) Extend: compute fused ontology-leaf rankings using:
   - 0.8 weight: semantic cosine similarity between:
       a) embeddings of solution-derived query strings (multiple components per solution)
       b) precomputed ontology leaf embeddings (PREDICTOR_leaf_embeddings.npy)
   - 0.2 weight: lexical fusion of TWO textual ranks:
       a) BM25 on LEXTTEXT
       b) token-overlap cosine on LEXTTEXT
     (lexical = 0.5*bm25_norm + 0.5*overlap_norm)

Batch Mode
----------
Runs ALL pseudoprofiles present in mapped_criterions.csv (unless --pseudoprofile_id provided),
in parallel with ThreadPoolExecutor.

Caching
-------
Per-pseudoprofile caching is used to avoid re-calling the LLM and/or re-computing rankings if artifacts
already exist on disk (in the run's profiles/<pseudoprofile_id>/ directory). Artifacts are saved
immediately after each pseudoprofile completes.

Outputs
-------
Per-run directory:
  runs/<run_id>/
    config.json
    dense_profiles.csv          (wide format, top global N=200 columns)
    long_rankings.csv           (long format, ALL global top 200 + per-solution top 50)
    errors.csv
    profiles/<pseudoprofile_id>/
        llm_result.json
        fused_rankings.json
        combined.json
        config.json

Usage
-----
python generate_solution_predictors.py
python generate_solution_predictors.py --pseudoprofile_id pseudoprofile_FTC_ID002

Environment
-----------
export OPENAI_API_KEY="..."
(Optional) .env via python-dotenv

Files expected in embeddings_dir
-------------------------------
PREDICTOR_leaf_embeddings.npy
PREDICTOR_leaf_embedding_norms.npy              (optional but recommended)
PREDICTOR_leaf_paths_EMBEDTEXT.json
PREDICTOR_leaf_paths_LEXTEXT.json
PREDICTOR_leaf_paths_FULL.json                  (optional for display)
PREDICTOR_leaf_embeddings_meta.json             (optional integrity check)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
import random
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env if available


# -----------------------------
# Defaults (as provided by you)
# -----------------------------
def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        has_eval = (candidate / "evaluation").exists() or (candidate / "Evaluation").exists()
        if has_eval and (candidate / "README.md").exists():
            return candidate
    raise RuntimeError("Could not locate repository root from generate_HyDe_based_predictor_ranks.py")


REPO_ROOT = _find_repo_root()

DEFAULT_MAPPED_CRITERIONS_PATH = (
    str(REPO_ROOT / "evaluation/02_mental_health_issue_operationalization/mapped_criterions.csv")
)
DEFAULT_HIGH_LEVEL_ONTOLOGY_PATH = (
    str(REPO_ROOT / "src/utils/official/ontology_mappings/CRITERION/predictor_to_criterion/input_lists/predictors_list.txt")
)

DEFAULT_PREDICTOR_EMBEDDINGS_DIR = (
    str(REPO_ROOT / "src/SystemComponents/Agentic_Framework/02_ConstructionInitialObservationModel/utils/helpers/embeddings")
)

LLM_MODEL_NAME = "gpt-5-nano"  # later use 'gpt-5' ; just for testing use 'nano' or 'mini'
EMBEDDING_MODEL_NAME = "text-embedding-3-small"  # per your request

# Ranking parameters (per your request)
GLOBAL_TOP_N = 200          # multi-predictor fuse global ranking
PER_SOLUTION_TOP_K = 50     # per-solution top-k ranking

SEMANTIC_WEIGHT = 0.8
LEXICAL_WEIGHT = 0.2
BM25_WEIGHT_WITHIN_LEXICAL = 0.5
OVERLAP_WEIGHT_WITHIN_LEXICAL = 0.5

# BM25 parameters
BM25_K1 = 1.5
BM25_B = 0.75

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


# -----------------------------
# Data structures
# -----------------------------
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


@dataclass
class ProfileInput:
    pseudoprofile_id: str
    complaint_text: str
    decomp_n_variables: Optional[int]
    decomp_notes: str
    criteria: List[CriterionRow]
    high_level_predictor_ontology_raw: str


# -----------------------------
# Helpers
# -----------------------------
def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
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
    # e.g., 2026-01-15_14-35-01
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _safe_filename(s: str) -> str:
    # conservative, cross-platform safe
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s).strip())
    return s[:180] if len(s) > 180 else s


def _log(pid: str, msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    prefix = f"[{ts}]"
    if pid:
        print(f"{prefix} [{pid}] {msg}", flush=True)
    else:
        print(f"{prefix} {msg}", flush=True)


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

    sub = df[df["pseudoprofile_id"] == pseudoprofile_id].copy()
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
        decomp_n_variables = int(first["decomp_n_variables"]) if not pd.isna(first["decomp_n_variables"]) else None

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

    criteria_rows: List[CriterionRow] = []
    seen: set = set()

    for _, row in sub.iterrows():
        vid = _safe_str(row.get("variable_id", "")).strip()
        if not vid:
            vid = f"NO_ID::{_safe_str(row.get('variable_label',''))}::{_safe_str(row.get('variable_criterion',''))}"
        if vid in seen:
            continue
        seen.add(vid)

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
            )
        )

    ontology_raw = read_text_file(ontology_path, max_chars=max_ontology_chars)

    return ProfileInput(
        pseudoprofile_id=pseudoprofile_id,
        complaint_text=complaint_text,
        decomp_n_variables=decomp_n_variables,
        decomp_notes=decomp_notes,
        criteria=criteria_rows,
        high_level_predictor_ontology_raw=ontology_raw,
    )


def build_llm_messages(profile: ProfileInput) -> Tuple[str, List[Dict[str, Any]]]:
    instructions = (
        "You are an expert interdisciplinary mental health optimization planner (clinical psychology, "
        "behavioral medicine, psychiatry-adjacent, neuroscience, coaching, and social determinants). "
        "Your job is NOT to diagnose. Your job is to propose high-quality, *modifiable* candidate "
        "BIO-PSYCHO-SOCIAL solution variables (predictors/levers) that improve the given operationalized (non-)clinical mental health state.\n\n"
        "Key constraints:\n"
        "- The complaint may be clinical OR non-clinical. Calibrate suggestions accordingly.\n"
        "- Produce solutions across Bio / Psycho / Social domains (and their interactions).\n"
        "- Each candidate must be a changeable predictor (a lever), not a symptom re-label.\n"
        # "- Prefer: high-leverage, feasible, low-regret, measurable, and ethically/safely actionable.\n"
        "- Avoid illegal substances or anything unsafe; do not give medication dosing instructions.\n"
        "- If the input suggests risk of self-harm or other acute severe deterioration, include a strong "
        "seek-professional-help note in safety_notes.\n"
        "- Output MUST strictly follow the JSON schema provided.\n\n"
        "What 'optimal' means here:\n"
        "- Target mechanisms that plausibly move single-specific or joint-multiple (non-)clinical mental health criteria.\n"
        "- Include both (non-)therapeutic interventions: look at BIO-PSYCHO-SOCIAL high-level structure to obtain overview of possibilities.\n"
        "clinical pathways.\n"
        "- Provide a tight mapping from each candidate predictor to the criteria it is expected to affect.\n"
        "- Provide measurement ideas (simple metrics) for each predictor.\n\n"
    )

    criteria_payload = []
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
            }
        )

    user_content = {
        # "pseudoprofile_id": profile.pseudoprofile_id,
        "complaint_text": profile.complaint_text,
        "decomp_n_variables": profile.decomp_n_variables,
        "decomp_notes": profile.decomp_notes,
        "criteria_operationalization_rows": criteria_payload,
        "high_level_BioPsychoSocial_predictor_ontology_overview_RAW": profile.high_level_predictor_ontology_raw,
    }

    messages = [
        {
            "role": "user",
            "content": (
                "Use the following case payload to generate candidate solution predictors.\n\n"
                "CASE_PAYLOAD_JSON:\n"
                f"{json.dumps(user_content, ensure_ascii=False, indent=2)}"
            ),
        }
    ]
    return instructions, messages


def solution_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "pseudoprofile_id": {"type": "string"},
            "summary": {
                "type": "string",
                "description": "One-paragraph expert summary of the key mechanisms and constraints.",
            },
            "solutions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "solution_id": {"type": "string", "description": "S01, S02, ..."},
                        "predictor_variable": {"type": "string"},
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
                        "solution_candidate": {"type": "string"},
                        "targets_criteria_ids": {"type": "array", "items": {"type": "string"}},
                        "mechanism_rationale": {"type": "string"},
                        "measurement": {"type": "string"},
                        "time_horizon": {"type": "string"},
                        "feasibility_0_1": {"type": "number", "minimum": 0, "maximum": 1},
                        "expected_impact_0_1": {"type": "number", "minimum": 0, "maximum": 1},
                        "priority": {"type": "string", "enum": ["HIGH", "MED", "LOW"]},
                        "safety_notes": {"type": "string"},
                        "mapping_query": {"type": "string"},
                    },
                    "required": [
                        "solution_id",
                        "predictor_variable",
                        "bio_psycho_social_domain",
                        "solution_candidate",
                        "targets_criteria_ids",
                        "mechanism_rationale",
                        "measurement",
                        "time_horizon",
                        "feasibility_0_1",
                        "expected_impact_0_1",
                        "priority",
                        "safety_notes",
                        "mapping_query",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["pseudoprofile_id", "summary", "solutions"],
        "additionalProperties": False,
    }


def call_llm_structured(
    instructions: str,
    messages: List[Dict[str, Any]],
    max_output_tokens: int = 1800,
    pseudoprofile_id: Optional[str] = None,
) -> Dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set. Export it before running this script.")

    client = OpenAI()
    schema = solution_schema()

    _log(pseudoprofile_id or "", f"LLM call START | model={LLM_MODEL_NAME}")

    response = client.responses.create(
        model=LLM_MODEL_NAME,
        instructions=instructions,
        input=messages,
        # max_output_tokens=max_output_tokens,
        text={
            "format": {
                "type": "json_schema",
                "name": "bipsysoc_solutions",
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
        _log(pseudoprofile_id or "", "LLM call OK | parsing JSON")
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


# -----------------------------
# Output formatting
# -----------------------------
def print_human_summary(result: Dict[str, Any]) -> None:
    print("\n=== SUMMARY ===")
    print(result.get("summary", "").strip())
    print("\n=== CANDIDATE SOLUTION PREDICTORS ===")

    sols = result.get("solutions", [])
    for s in sols:
        sid = s.get("solution_id", "")
        pred = s.get("predictor_variable", "")
        dom = s.get("bio_psycho_social_domain", "")
        prio = s.get("priority", "")
        targets = ", ".join(s.get("targets_criteria_ids", []) or [])
        print(f"\n[{sid}] {pred}")
        print(f"  Domain   : {dom} | Priority: {prio}")
        print(f"  Targets  : {targets}")
        print(f"  Plan     : {s.get('solution_candidate','')}")
        print(f"  Measure  : {s.get('measurement','')}")
        print(f"  Horizon  : {s.get('time_horizon','')}")
        print(f"  Feas/Imp : {s.get('feasibility_0_1','')} / {s.get('expected_impact_0_1','')}")
        print(f"  MapQuery : {s.get('mapping_query','')}")


# ============================================================
# Ontology ranking via semantic + lexical fused ranking
# ============================================================
@dataclass
class PredictorOntologyDB:
    leaf_paths_embed: List[str]   # from PREDICTOR_leaf_paths_EMBEDTEXT.json (starred leaf)
    leaf_paths_lex: List[str]     # from PREDICTOR_leaf_paths_LEXTEXT.json (token-friendly)
    leaf_paths_full: Optional[List[str]]  # from PREDICTOR_leaf_paths_FULL.json (human display)
    embeddings: np.ndarray        # memmap or ndarray [N, D]
    norms: np.ndarray             # [N]
    embedding_dim: int


def load_predictor_ontology_db(embeddings_dir: str) -> PredictorOntologyDB:
    embed_paths = os.path.join(embeddings_dir, "PREDICTOR_leaf_paths_EMBEDTEXT.json")
    lex_paths = os.path.join(embeddings_dir, "PREDICTOR_leaf_paths_LEXTEXT.json")
    full_paths = os.path.join(embeddings_dir, "PREDICTOR_leaf_paths_FULL.json")
    emb_npy = os.path.join(embeddings_dir, "PREDICTOR_leaf_embeddings.npy")
    norms_npy = os.path.join(embeddings_dir, "PREDICTOR_leaf_embedding_norms.npy")

    if not os.path.exists(embed_paths):
        raise FileNotFoundError(f"Missing: {embed_paths}")
    if not os.path.exists(lex_paths):
        raise FileNotFoundError(f"Missing: {lex_paths}")
    if not os.path.exists(emb_npy):
        raise FileNotFoundError(f"Missing: {emb_npy}")

    with open(embed_paths, "r", encoding="utf-8") as f:
        leaf_paths_embed = json.load(f)
    with open(lex_paths, "r", encoding="utf-8") as f:
        leaf_paths_lex = json.load(f)

    leaf_paths_full_list: Optional[List[str]] = None
    if os.path.exists(full_paths):
        with open(full_paths, "r", encoding="utf-8") as f:
            leaf_paths_full_list = json.load(f)

    embeddings = np.load(emb_npy, mmap_mode="r")
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D array; got shape={embeddings.shape}")

    n, d = embeddings.shape
    if len(leaf_paths_embed) != n or len(leaf_paths_lex) != n:
        raise ValueError(
            "Ontology path list length mismatch with embeddings.\n"
            f"len(leaf_paths_embed)={len(leaf_paths_embed)} len(leaf_paths_lex)={len(leaf_paths_lex)} "
            f"embeddings_rows={n}"
        )

    if os.path.exists(norms_npy):
        norms = np.load(norms_npy, mmap_mode="r")
        if norms.shape[0] != n:
            raise ValueError(f"Norms length mismatch: norms={norms.shape} embeddings_rows={n}")
    else:
        norms = np.linalg.norm(np.asarray(embeddings), axis=1).astype(np.float32)

    return PredictorOntologyDB(
        leaf_paths_embed=leaf_paths_embed,
        leaf_paths_lex=leaf_paths_lex,
        leaf_paths_full=leaf_paths_full_list,
        embeddings=embeddings,
        norms=norms,
        embedding_dim=d,
    )


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


@dataclass
class BM25Index:
    n_docs: int
    avgdl: float
    doc_len: np.ndarray               # [N]
    doc_unique_len: np.ndarray        # [N]
    idf: Dict[str, float]
    postings: Dict[str, List[Tuple[int, int]]]  # term -> [(doc_idx, tf), ...]


def build_bm25_index(docs_lex: List[str]) -> BM25Index:
    n = len(docs_lex)
    doc_len = np.zeros(n, dtype=np.int32)
    doc_unique_len = np.zeros(n, dtype=np.int32)
    df: Dict[str, int] = {}
    postings: Dict[str, List[Tuple[int, int]]] = {}

    for i, doc in enumerate(docs_lex):
        toks = tokenize(doc)
        doc_len[i] = len(toks)
        tf: Dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        doc_unique_len[i] = len(tf)

        for t, c in tf.items():
            df[t] = df.get(t, 0) + 1
            postings.setdefault(t, []).append((i, c))

    avgdl = float(doc_len.mean()) if n > 0 else 0.0
    idf: Dict[str, float] = {}
    for t, dfi in df.items():
        idf[t] = math.log((n - dfi + 0.5) / (dfi + 0.5) + 1.0)

    return BM25Index(
        n_docs=n,
        avgdl=avgdl,
        doc_len=doc_len,
        doc_unique_len=doc_unique_len,
        idf=idf,
        postings=postings,
    )


def bm25_scores(index: BM25Index, query: str, k1: float = BM25_K1, b: float = BM25_B) -> np.ndarray:
    q_toks = tokenize(query)
    if not q_toks:
        return np.zeros(index.n_docs, dtype=np.float32)

    scores = np.zeros(index.n_docs, dtype=np.float32)
    avgdl = index.avgdl if index.avgdl > 0 else 1.0

    for t in q_toks:
        if t not in index.postings:
            continue
        idf = index.idf.get(t, 0.0)
        for doc_idx, tf in index.postings[t]:
            dl = float(index.doc_len[doc_idx])
            denom = tf + k1 * (1.0 - b + b * (dl / avgdl))
            scores[doc_idx] += float(idf) * (tf * (k1 + 1.0) / (denom + 1e-12))

    return scores


def overlap_cosine_scores(index: BM25Index, query: str) -> np.ndarray:
    q_set = set(tokenize(query))
    if not q_set:
        return np.zeros(index.n_docs, dtype=np.float32)

    counts = np.zeros(index.n_docs, dtype=np.float32)
    for t in q_set:
        if t not in index.postings:
            continue
        for doc_idx, _tf in index.postings[t]:
            counts[doc_idx] += 1.0

    q_len = float(len(q_set))
    denom = np.sqrt(index.doc_unique_len.astype(np.float32) * q_len) + 1e-12
    return counts / denom


def normalize_0_1(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    x_max = float(np.max(x)) if x.size else 0.0
    x_min = float(np.min(x)) if x.size else 0.0
    if abs(x_max - x_min) < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x - x_min) / (x_max - x_min)


def normalize_by_max(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    x_max = float(np.max(x)) if x.size else 0.0
    if x_max <= 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return x / x_max


def embed_texts_openai(client: OpenAI, texts: List[str], model: str = EMBEDDING_MODEL_NAME) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)

    batch_size = 64
    all_vecs: List[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model, input=chunk)
        vecs = [np.array(item.embedding, dtype=np.float32) for item in resp.data]
        all_vecs.append(np.vstack(vecs))

    return np.vstack(all_vecs)


def cosine_sim_to_all_leaves(
    leaf_embeddings: np.ndarray,
    leaf_norms: np.ndarray,
    q_vec: np.ndarray,
    block_rows: int = 20000,
) -> np.ndarray:
    q_vec = q_vec.astype(np.float32, copy=False)
    q_norm = float(np.linalg.norm(q_vec)) + 1e-12

    n = leaf_embeddings.shape[0]
    sims = np.empty(n, dtype=np.float32)

    for start in range(0, n, block_rows):
        end = min(start + block_rows, n)
        block = np.asarray(leaf_embeddings[start:end]).astype(np.float32, copy=False)
        dots = block @ q_vec
        denom = (leaf_norms[start:end].astype(np.float32) * q_norm) + 1e-12
        sims[start:end] = dots / denom

    return sims


def build_criteria_lookup(profile: ProfileInput) -> Dict[str, CriterionRow]:
    return {c.variable_id: c for c in profile.criteria}


def build_solution_component_texts_for_embedding(
    solution: Dict[str, Any],
    profile: ProfileInput,
    criteria_lookup: Dict[str, CriterionRow],
) -> List[str]:
    """
    Several embeddings per solution output component.
    Includes criterion_mapped.csv info (labels/criteria/evidence/query_text_used) for targeted criteria.

    IMPORTANT: Leave t2-t4 commented (per your request).
    """
    pred = _safe_str(solution.get("predictor_variable", "")).strip()
    plan = _safe_str(solution.get("solution_candidate", "")).strip()
    mech = _safe_str(solution.get("mechanism_rationale", "")).strip()
    meas = _safe_str(solution.get("measurement", "")).strip()
    horizon = _safe_str(solution.get("time_horizon", "")).strip()
    domain = _safe_str(solution.get("bio_psycho_social_domain", "")).strip()
    mapping_query = _safe_str(solution.get("mapping_query", "")).strip()
    targets = solution.get("targets_criteria_ids", []) or []

    # Pull targeted criterion detail from mapped_criterions.csv rows
    crit_chunks: List[str] = []
    for cid in targets:
        if cid in criteria_lookup:
            c = criteria_lookup[cid]
            crit_chunks.append(
                f"[{c.variable_id}] {c.variable_label}. Criterion: {c.variable_criterion} "
                f"Evidence: {c.variable_evidence} QueryUsed: {c.query_text_used}"
            )
        else:
            crit_chunks.append(f"[{cid}] (criterion details not found in lookup)")

    criteria_block = "\n".join(crit_chunks).strip()

    complaint = profile.complaint_text.strip()
    decomp_notes = profile.decomp_notes.strip()

    # Component 1: direct lever + plan
    t1 = (
        f"solution domain: {domain} ; intervention variable: {pred} ; plan: {plan} \n"
    ).strip()

    # Component 2: mechanism
    #t2 = (
    #    f"Mechanism rationale for lever '{pred}': {mech}\n"
    #    f"Targets criteria: {', '.join([str(x) for x in targets])}"
    #).strip()

    # Component 3: measurement/horizon
    #t3 = (
    #    f"Measurement plan for lever '{pred}': {meas}\n"
    #    f"Expected time horizon: {horizon}"
    #).strip()

    # Component 4: criterion context + original complaint (anchors semantics to operationalization)
    #t4 = (
    #    f"Original complaint text: {complaint}\n"
    #    f"Decomposition notes: {decomp_notes}\n"
    #    f"Targeted operationalized criteria:\n{criteria_block}"
    #).strip()

    #return [t1, t2, t3, t4]
    return [t1]


def build_solution_component_queries_for_lexical(
    solution: Dict[str, Any],
    profile: ProfileInput,
    criteria_lookup: Dict[str, CriterionRow],
) -> List[str]:
    pred = _safe_str(solution.get("predictor_variable", "")).strip()
    mapping_query = _safe_str(solution.get("mapping_query", "")).strip()
    targets = solution.get("targets_criteria_ids", []) or []

    crit_labels = []
    for cid in targets:
        c = criteria_lookup.get(cid)
        if c:
            crit_labels.append(c.variable_label)
    crit_label_text = " ".join(crit_labels)

    q1 = f"{mapping_query} {pred} {crit_label_text}".strip()
    q2 = f"{pred} {crit_label_text} {profile.complaint_text}".strip()
    return [q1, q2]


def fused_scores_for_queries(
    ontology: PredictorOntologyDB,
    bm25_index: BM25Index,
    semantic_sims_per_query: List[np.ndarray],
    lexical_queries: List[str],
) -> np.ndarray:
    fused_acc = np.zeros(ontology.embeddings.shape[0], dtype=np.float32)
    n_q = 0

    for sem_sims in semantic_sims_per_query:
        sem_norm = normalize_0_1(sem_sims)
        fused_acc += SEMANTIC_WEIGHT * sem_norm
        n_q += 1

    for q in lexical_queries:
        bm25 = bm25_scores(bm25_index, q)
        ovlp = overlap_cosine_scores(bm25_index, q)
        bm25_n = normalize_by_max(bm25)
        ovlp_n = normalize_by_max(ovlp)
        lex = (BM25_WEIGHT_WITHIN_LEXICAL * bm25_n) + (OVERLAP_WEIGHT_WITHIN_LEXICAL * ovlp_n)
        fused_acc += LEXICAL_WEIGHT * lex
        n_q += 1

    if n_q == 0:
        return fused_acc
    return fused_acc / float(n_q)


def compute_fused_rankings(
    client: OpenAI,
    profile: ProfileInput,
    llm_result: Dict[str, Any],
    ontology: PredictorOntologyDB,
    global_top_n: int = GLOBAL_TOP_N,
    per_solution_top_k: int = PER_SOLUTION_TOP_K,
) -> Dict[str, Any]:
    criteria_lookup = build_criteria_lookup(profile)
    bm25_index = build_bm25_index(ontology.leaf_paths_lex)

    solutions = llm_result.get("solutions", []) or []
    if not solutions:
        return {"global_top": [], "per_solution_top": {}}

    global_embed_texts: List[str] = []
    global_lex_queries: List[str] = []
    per_solution_payload: List[Dict[str, Any]] = []

    for s in solutions:
        embed_texts = build_solution_component_texts_for_embedding(s, profile, criteria_lookup)
        lex_queries = build_solution_component_queries_for_lexical(s, profile, criteria_lookup)

        per_solution_payload.append(
            {
                "solution_id": s.get("solution_id", ""),
                "predictor_variable": s.get("predictor_variable", ""),
                "embed_texts": embed_texts,
                "lex_queries": lex_queries,
            }
        )
        global_embed_texts.extend(embed_texts)
        global_lex_queries.extend(lex_queries)

    _log(profile.pseudoprofile_id, f"Ranking: embedding {len(global_embed_texts)} semantic query-text(s)")
    global_embed_vecs = embed_texts_openai(client, global_embed_texts, model=EMBEDDING_MODEL_NAME)

    if global_embed_vecs.shape[1] != ontology.embedding_dim:
        raise ValueError(
            f"Embedding dim mismatch: query_dim={global_embed_vecs.shape[1]} leaf_dim={ontology.embedding_dim}. "
            f"(Leaf embeddings computed with a different embedding model?)"
        )

    _log(profile.pseudoprofile_id, f"Ranking: computing semantic cosine sims for {global_embed_vecs.shape[0]} query vector(s)")
    global_sem_sims: List[np.ndarray] = []
    for i in range(global_embed_vecs.shape[0]):
        sims = cosine_sim_to_all_leaves(ontology.embeddings, ontology.norms, global_embed_vecs[i])
        global_sem_sims.append(sims)

    _log(profile.pseudoprofile_id, "Ranking: fusing semantic + lexical scores (GLOBAL)")
    global_fused = fused_scores_for_queries(
        ontology=ontology,
        bm25_index=bm25_index,
        semantic_sims_per_query=global_sem_sims,
        lexical_queries=global_lex_queries,
    )

    top_idx = np.argsort(-global_fused)[:global_top_n]
    global_top = []
    for rank, idx in enumerate(top_idx, start=1):
        global_top.append(
            {
                "rank": rank,
                "leaf_index": int(idx),
                "fused_score_0_1": float(global_fused[idx]),
                "leaf_path_embedtext": ontology.leaf_paths_embed[idx],
                "leaf_path_full": (ontology.leaf_paths_full[idx] if ontology.leaf_paths_full else None),
                "leaf_path_lextext": ontology.leaf_paths_lex[idx],
            }
        )

    _log(profile.pseudoprofile_id, f"Ranking: GLOBAL top-{global_top_n} computed")

    per_solution_top: Dict[str, List[Dict[str, Any]]] = {}
    cursor = 0
    for payload in per_solution_payload:
        embed_texts = payload["embed_texts"]
        lex_queries = payload["lex_queries"]
        sid = str(payload.get("solution_id", "")).strip()
        pred = str(payload.get("predictor_variable", "")).strip()
        sol_key = f"{sid} | {pred}" if sid and pred else (sid or pred or f"solution_{len(per_solution_top) + 1}")

        n_texts = len(embed_texts)
        vecs = global_embed_vecs[cursor : cursor + n_texts]
        cursor += n_texts

        sem_sims_list: List[np.ndarray] = []
        for i in range(vecs.shape[0]):
            sims = cosine_sim_to_all_leaves(ontology.embeddings, ontology.norms, vecs[i])
            sem_sims_list.append(sims)

        fused = fused_scores_for_queries(
            ontology=ontology,
            bm25_index=bm25_index,
            semantic_sims_per_query=sem_sims_list,
            lexical_queries=lex_queries,
        )

        topk_idx = np.argsort(-fused)[:per_solution_top_k]
        per_solution_top[sol_key] = [
            {
                "rank": r + 1,
                "leaf_index": int(ix),
                "fused_score_0_1": float(fused[ix]),
                "leaf_path_embedtext": ontology.leaf_paths_embed[ix],
                "leaf_path_full": (ontology.leaf_paths_full[ix] if ontology.leaf_paths_full else None),
                "leaf_path_lextext": ontology.leaf_paths_lex[ix],
            }
            for r, ix in enumerate(topk_idx)
        ]

    _log(profile.pseudoprofile_id, f"Ranking: PER-SOLUTION computed for {len(per_solution_top)} solution(s) (top-{per_solution_top_k} each)")
    return {"global_top": global_top, "per_solution_top": per_solution_top}


def print_fused_rankings(rankings: Dict[str, Any], *, global_print_n: int = 50, per_solution_print_n: int = 10) -> None:
    print(f"\n=== GLOBAL FUSED RANKING (Top {global_print_n}) ===")
    global_top = rankings.get("global_top", []) or []
    for item in global_top[:global_print_n]:
        rank = item["rank"]
        score = item["fused_score_0_1"]
        path = item["leaf_path_embedtext"]
        print(f"{rank:02d}. {score:.4f}  {path}")

    print(f"\n=== PER-SOLUTION FUSED RANKINGS (Top {per_solution_print_n} each) ===")
    per_sol = rankings.get("per_solution_top", {}) or {}
    for sol_key, items in per_sol.items():
        print(f"\n--- {sol_key} ---")
        for it in (items or [])[:per_solution_print_n]:
            print(f"{it['rank']:02d}. {it['fused_score_0_1']:.4f}  {it['leaf_path_embedtext']}")


# ============================================================
# Batch runner + caching + CSV builders
# ============================================================
def _extract_solution_meta_map(llm_result: Dict[str, Any]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for s in (llm_result.get("solutions", []) or []):
        sid = str(s.get("solution_id", "")).strip()
        pred = str(s.get("predictor_variable", "")).strip()
        out[(sid, pred)] = s
    return out


def _profile_dir(profiles_dir: str, pseudoprofile_id: str) -> str:
    return os.path.join(profiles_dir, _safe_filename(pseudoprofile_id))


def _cache_paths_for_profile(profiles_dir: str, pseudoprofile_id: str) -> Dict[str, str]:
    pdir = _profile_dir(profiles_dir, pseudoprofile_id)
    return {
        "profile_dir": pdir,
        "llm_result": os.path.join(pdir, "llm_result.json"),
        "fused_rankings": os.path.join(pdir, "fused_rankings.json"),
        "combined": os.path.join(pdir, "combined.json"),
        "config": os.path.join(pdir, "config.json"),
    }


def _load_cached_profile_artifacts(profiles_dir: str, pseudoprofile_id: str) -> Optional[Dict[str, Any]]:
    paths = _cache_paths_for_profile(profiles_dir, pseudoprofile_id)
    if os.path.exists(paths["combined"]):
        try:
            with open(paths["combined"], "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    if os.path.exists(paths["llm_result"]) and os.path.exists(paths["fused_rankings"]):
        try:
            with open(paths["llm_result"], "r", encoding="utf-8") as f:
                llm_result = json.load(f)
            with open(paths["fused_rankings"], "r", encoding="utf-8") as f:
                rankings = json.load(f)
            return {"llm_result": llm_result, "fused_rankings": rankings}
        except Exception:
            return None
    return None


def _save_per_profile_artifacts(
    profiles_dir: str,
    run_id: str,
    profile: ProfileInput,
    llm_result: Dict[str, Any],
    rankings: Dict[str, Any],
    config: Dict[str, Any],
) -> str:
    prof_dir = _ensure_dir(_profile_dir(profiles_dir, profile.pseudoprofile_id))

    with open(os.path.join(prof_dir, "llm_result.json"), "w", encoding="utf-8") as f:
        json.dump(llm_result, f, ensure_ascii=False, indent=2)

    with open(os.path.join(prof_dir, "fused_rankings.json"), "w", encoding="utf-8") as f:
        json.dump(rankings, f, ensure_ascii=False, indent=2)

    combined = {
        "run_id": run_id,
        "pseudoprofile_id": profile.pseudoprofile_id,
        "llm_result": llm_result,
        "fused_rankings": rankings,
        "config": config,
    }
    with open(os.path.join(prof_dir, "combined.json"), "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    with open(os.path.join(prof_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    return prof_dir


def _make_dense_row(
    run_id: str,
    profile: ProfileInput,
    llm_result: Dict[str, Any],
    rankings: Dict[str, Any],
    *,
    dense_global_top_n: int,
) -> Dict[str, Any]:
    sols = llm_result.get("solutions", []) or []
    global_top = rankings.get("global_top", []) or []

    row: Dict[str, Any] = {
        "run_id": run_id,
        "pseudoprofile_id": profile.pseudoprofile_id,
        "n_solutions": len(sols),
        "complaint_text": profile.complaint_text,
        "summary": llm_result.get("summary", ""),
        "solutions_compact": " | ".join([f"{s.get('solution_id','')}:{s.get('predictor_variable','')}" for s in sols]),
        "llm_model": LLM_MODEL_NAME,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "semantic_weight": SEMANTIC_WEIGHT,
        "lexical_weight": LEXICAL_WEIGHT,
        "bm25_weight_within_lexical": BM25_WEIGHT_WITHIN_LEXICAL,
        "overlap_weight_within_lexical": OVERLAP_WEIGHT_WITHIN_LEXICAL,
        "global_top_n_requested": dense_global_top_n,
        "global_top_n_computed": len(global_top),
    }

    # Wide columns: GLOBAL top 200 (or fewer if fewer computed)
    for i in range(min(dense_global_top_n, len(global_top))):
        it = global_top[i]
        k = i + 1
        row[f"global_{k:03d}_rank"] = it.get("rank")
        row[f"global_{k:03d}_score"] = it.get("fused_score_0_1")
        row[f"global_{k:03d}_leaf_index"] = it.get("leaf_index")
        row[f"global_{k:03d}_path_embedtext"] = it.get("leaf_path_embedtext")
        row[f"global_{k:03d}_path_full"] = it.get("leaf_path_full")
        row[f"global_{k:03d}_path_lextext"] = it.get("leaf_path_lextext")

    return row


def _append_long_rows(
    long_rows: List[Dict[str, Any]],
    run_id: str,
    profile: ProfileInput,
    llm_result: Dict[str, Any],
    rankings: Dict[str, Any],
) -> None:
    """
    Long rows:
      - GLOBAL: all rows in global_top (top 200)
      - PER_SOLUTION: all rows in per_solution_top (top 50 per solution)

    This stores ALL computed ranking info in CSV for later detailed analyses.
    """
    sol_meta = _extract_solution_meta_map(llm_result)

    # GLOBAL
    for it in (rankings.get("global_top", []) or []):
        long_rows.append(
            {
                "run_id": run_id,
                "pseudoprofile_id": profile.pseudoprofile_id,
                "scope": "GLOBAL",
                "solution_key": "",
                "solution_id": "",
                "predictor_variable": "",
                "bio_psycho_social_domain": "",
                "priority": "",
                "targets_criteria_ids": "",
                "solution_candidate": "",
                "mechanism_rationale": "",
                "measurement": "",
                "time_horizon": "",
                "feasibility_0_1": "",
                "expected_impact_0_1": "",
                "mapping_query": "",
                "leaf_rank": it.get("rank"),
                "leaf_index": it.get("leaf_index"),
                "fused_score_0_1": it.get("fused_score_0_1"),
                "leaf_path_embedtext": it.get("leaf_path_embedtext"),
                "leaf_path_full": it.get("leaf_path_full"),
                "leaf_path_lextext": it.get("leaf_path_lextext"),
            }
        )

    # PER-SOLUTION
    per_sol = rankings.get("per_solution_top", {}) or {}
    for sol_key, items in per_sol.items():
        sid = ""
        pred = ""
        if " | " in sol_key:
            parts = sol_key.split(" | ", 1)
            sid = parts[0].strip()
            pred = parts[1].strip()
        else:
            sid = sol_key.strip()

        meta = sol_meta.get((sid, pred), {})
        targets = meta.get("targets_criteria_ids", []) or []
        targets_joined = "|".join([str(x) for x in targets])

        for it in (items or []):
            long_rows.append(
                {
                    "run_id": run_id,
                    "pseudoprofile_id": profile.pseudoprofile_id,
                    "scope": "PER_SOLUTION",
                    "solution_key": sol_key,
                    "solution_id": sid,
                    "predictor_variable": pred,
                    "bio_psycho_social_domain": meta.get("bio_psycho_social_domain", ""),
                    "priority": meta.get("priority", ""),
                    "targets_criteria_ids": targets_joined,
                    "solution_candidate": meta.get("solution_candidate", ""),
                    "mechanism_rationale": meta.get("mechanism_rationale", ""),
                    "measurement": meta.get("measurement", ""),
                    "time_horizon": meta.get("time_horizon", ""),
                    "feasibility_0_1": meta.get("feasibility_0_1", ""),
                    "expected_impact_0_1": meta.get("expected_impact_0_1", ""),
                    "mapping_query": meta.get("mapping_query", ""),
                    "leaf_rank": it.get("rank"),
                    "leaf_index": it.get("leaf_index"),
                    "fused_score_0_1": it.get("fused_score_0_1"),
                    "leaf_path_embedtext": it.get("leaf_path_embedtext"),
                    "leaf_path_full": it.get("leaf_path_full"),
                    "leaf_path_lextext": it.get("leaf_path_lextext"),
                }
            )


def _process_one_pseudoprofile(
    pseudoprofile_id: str,
    *,
    run_id: str,
    profiles_dir: str,
    mapped_criterions_path: str,
    ontology_path: str,
    max_ontology_chars: Optional[int],
    max_output_tokens: int,
    ontology_db: PredictorOntologyDB,
    global_top_n: int,
    per_solution_top_k: int,
    config: Dict[str, Any],
    use_cache: bool = True,
    recompute_rankings_if_missing: bool = True,
) -> Dict[str, Any]:
    """
    Worker for ThreadPoolExecutor.

    Caching logic:
      - If combined.json exists -> load and return (skip LLM + skip rankings).
      - Else if llm_result.json exists:
            - load llm_result
            - if fused_rankings.json exists -> load, return
            - else compute rankings (if recompute_rankings_if_missing=True), save, return
      - Else compute from scratch (LLM + rankings) and save immediately.

    Returns dict with keys:
      ok(bool), pseudoprofile_id, profile(ProfileInput|None), llm_result, rankings, from_cache(bool), error(str)
    """
    pid = pseudoprofile_id

    try:
        _log(pid, "START processing")

        profile = load_profile_input(
            mapped_criterions_path=mapped_criterions_path,
            ontology_path=ontology_path,
            pseudoprofile_id=pid,
            max_ontology_chars=max_ontology_chars,
        )

        # Cache check
        if use_cache:
            cached = _load_cached_profile_artifacts(profiles_dir, pid)
            if cached is not None:
                llm_cached = cached.get("llm_result")
                rank_cached = cached.get("fused_rankings")
                if llm_cached is not None and rank_cached is not None:
                    _log(pid, "CACHE HIT: combined artifacts found -> skipping LLM and ranking")
                    return {
                        "ok": True,
                        "pseudoprofile_id": pid,
                        "profile": profile,
                        "llm_result": llm_cached,
                        "rankings": rank_cached,
                        "from_cache": True,
                        "error": "",
                    }
                if llm_cached is not None and (rank_cached is None) and recompute_rankings_if_missing:
                    _log(pid, "CACHE PARTIAL: llm_result present but rankings missing -> computing rankings only")
                    client = OpenAI()
                    rankings = _with_retries(
                        lambda: compute_fused_rankings(
                            client=client,
                            profile=profile,
                            llm_result=llm_cached,
                            ontology=ontology_db,
                            global_top_n=global_top_n,
                            per_solution_top_k=per_solution_top_k,
                        ),
                        pid=pid,
                        label="RANKINGS",
                        retries=3,
                        base_sleep_s=2.0,
                    )
                    prof_dir = _save_per_profile_artifacts(
                        profiles_dir=profiles_dir,
                        run_id=run_id,
                        profile=profile,
                        llm_result=llm_cached,
                        rankings=rankings,
                        config=config,
                    )
                    _log(pid, f"SAVED (cache-fill) -> {prof_dir}")
                    return {
                        "ok": True,
                        "pseudoprofile_id": pid,
                        "profile": profile,
                        "llm_result": llm_cached,
                        "rankings": rankings,
                        "from_cache": False,
                        "error": "",
                    }

        # Fresh run (no cache)
        _log(pid, f"LOAD OK | criteria_rows={len(profile.criteria)} | complaint_chars={len(profile.complaint_text)}")

        instructions, messages = build_llm_messages(profile)

        llm_result = _with_retries(
            lambda: call_llm_structured(
                instructions=instructions,
                messages=messages,
                max_output_tokens=max_output_tokens,
                pseudoprofile_id=pid,
            ),
            pid=pid,
            label="LLM",
            retries=3,
            base_sleep_s=2.0,
        )

        _log(pid, f"LLM OK | n_solutions={(len(llm_result.get('solutions', []) or []))}")

        client = OpenAI()
        rankings = _with_retries(
            lambda: compute_fused_rankings(
                client=client,
                profile=profile,
                llm_result=llm_result,
                ontology=ontology_db,
                global_top_n=global_top_n,
                per_solution_top_k=per_solution_top_k,
            ),
            pid=pid,
            label="RANKINGS",
            retries=3,
            base_sleep_s=2.0,
        )

        _log(pid, f"RANKINGS OK | global_top={len(rankings.get('global_top', []) or [])} | per_solution={len(rankings.get('per_solution_top', {}) or {})}")

        # Save immediately (critical for caching + crash-safety)
        prof_dir = _save_per_profile_artifacts(
            profiles_dir=profiles_dir,
            run_id=run_id,
            profile=profile,
            llm_result=llm_result,
            rankings=rankings,
            config=config,
        )
        _log(pid, f"SAVED -> {prof_dir}")

        return {
            "ok": True,
            "pseudoprofile_id": pid,
            "profile": profile,
            "llm_result": llm_result,
            "rankings": rankings,
            "from_cache": False,
            "error": "",
        }

    except Exception as e:
        err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        _log(pid, f"FAIL: {type(e).__name__}: {e}")
        return {
            "ok": False,
            "pseudoprofile_id": pid,
            "profile": None,
            "llm_result": None,
            "rankings": None,
            "from_cache": False,
            "error": err,
        }


# -----------------------------
# Main (batch + caching + full ranking storage)
# -----------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch-generate BioPsychoSocial solution predictors + fused ontology-leaf rankings for ALL pseudoprofiles."
    )

    parser.add_argument("--mapped_criterions_path", default=DEFAULT_MAPPED_CRITERIONS_PATH)
    parser.add_argument("--high_level_BioPsychoSocial_ontology", default=DEFAULT_HIGH_LEVEL_ONTOLOGY_PATH)
    parser.add_argument("--max_ontology_chars", type=int, default=None)
    parser.add_argument("--max_output_tokens", type=int, default=1800)

    parser.add_argument("--predictor_embeddings_dir", default=DEFAULT_PREDICTOR_EMBEDDINGS_DIR)
    parser.add_argument("--global_top_n", type=int, default=GLOBAL_TOP_N)              # top 200 global fuse
    parser.add_argument("--per_solution_top_k", type=int, default=PER_SOLUTION_TOP_K)  # top 50 per solution

    parser.add_argument(
        "--pseudoprofile_id",
        default=None,
        help="Optional: run only this pseudoprofile_id. If omitted, runs ALL pseudoprofiles found in mapped_criterions.csv.",
    )
    parser.add_argument("--max_workers", type=int, default=20)

    parser.add_argument(
        "--results_dir",
        default=str(REPO_ROOT / "evaluation/03_construction_initial_observation_model/00_HyDe_based_predictor_ranks"),
        help="Base directory for results. A timestamped run directory will be created inside.",
    )
    parser.add_argument("--run_name", default=None, help="Optional run folder name. Default: timestamp.")
    parser.add_argument("--use_cache", action="store_true", help="If set, use per-profile cache to skip already processed pseudoprofiles.")
    parser.add_argument("--recompute_rankings_if_missing", action="store_true", help="If set, compute rankings when llm_result exists but rankings are missing.")
    parser.add_argument("--print_global_top_n", type=int, default=50, help="How many GLOBAL items to print to console per profile.")
    parser.add_argument("--print_per_solution_top_n", type=int, default=10, help="How many PER-SOLUTION items to print to console per profile.")

    # CSV controls
    parser.add_argument(
        "--dense_global_top_n",
        type=int,
        default=200,
        help="How many GLOBAL top leaves to store as wide columns in dense_profiles.csv (default=200).",
    )

    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set. Export it before running this script.")

    # Discover pseudoprofiles
    all_ids = list_pseudoprofile_ids_from_mapped(args.mapped_criterions_path)
    if args.pseudoprofile_id:
        if args.pseudoprofile_id not in all_ids:
            raise ValueError(
                f"pseudoprofile_id={args.pseudoprofile_id} not found in mapped_criterions.csv. "
                f"Example IDs: {all_ids[:15]}"
            )
        pseudoprofile_ids = [args.pseudoprofile_id]
    else:
        pseudoprofile_ids = all_ids

    base_dir = _ensure_dir(args.results_dir)
    run_id = _safe_filename(args.run_name) if args.run_name else _now_stamp()
    run_dir = _ensure_dir(os.path.join(base_dir, "runs", run_id))
    profiles_dir = _ensure_dir(os.path.join(run_dir, "profiles"))
    _ensure_dir(os.path.join(run_dir, "logs"))  # reserved for future, but keeps structure clear

    _log("", "==============================")
    _log("", "BATCH RUN START")
    _log("", f"Run ID           : {run_id}")
    _log("", f"Run dir          : {run_dir}")
    _log("", f"mapped_criterions: {args.mapped_criterions_path}")
    _log("", f"ontology txt     : {args.high_level_BioPsychoSocial_ontology}")
    _log("", f"embeddings dir   : {args.predictor_embeddings_dir}")
    _log("", f"pseudoprofiles   : {len(pseudoprofile_ids)}")
    _log("", f"max_workers      : {args.max_workers}")
    _log("", f"use_cache        : {args.use_cache}")
    _log("", f"global_top_n     : {args.global_top_n}")
    _log("", f"per_sol_top_k    : {args.per_solution_top_k}")
    _log("", "==============================\n")

    _log("", "Loading ontology DB (embeddings + paths) ...")
    ontology_db = load_predictor_ontology_db(args.predictor_embeddings_dir)
    _log("", f"Ontology loaded | leaves={len(ontology_db.leaf_paths_embed)} | emb_shape={ontology_db.embeddings.shape}")

    # Config snapshot for reproducibility
    config = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "llm_model": LLM_MODEL_NAME,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "semantic_weight": SEMANTIC_WEIGHT,
        "lexical_weight": LEXICAL_WEIGHT,
        "bm25_weight_within_lexical": BM25_WEIGHT_WITHIN_LEXICAL,
        "overlap_weight_within_lexical": OVERLAP_WEIGHT_WITHIN_LEXICAL,
        "global_top_n": args.global_top_n,
        "per_solution_top_k": args.per_solution_top_k,
        "max_output_tokens": args.max_output_tokens,
        "max_ontology_chars": args.max_ontology_chars,
        "mapped_criterions_path": args.mapped_criterions_path,
        "high_level_BioPsychoSocial_ontology": args.high_level_BioPsychoSocial_ontology,
        "predictor_embeddings_dir": args.predictor_embeddings_dir,
        "n_pseudoprofiles": len(pseudoprofile_ids),
        "max_workers": args.max_workers,
        "dense_global_top_n": args.dense_global_top_n,
        "print_global_top_n": args.print_global_top_n,
        "print_per_solution_top_n": args.print_per_solution_top_n,
        "use_cache": bool(args.use_cache),
        "recompute_rankings_if_missing": bool(args.recompute_rankings_if_missing),
        "ontology_embeddings_shape": list(ontology_db.embeddings.shape),
        "ontology_embedding_dim": ontology_db.embedding_dim,
        "paths_embedtext_n": len(ontology_db.leaf_paths_embed),
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    _log("", "Saved config.json")

    dense_rows: List[Dict[str, Any]] = []
    long_rows: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []

    started_at = time.time()

    with ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as ex:
        fut_map = {}
        for pid in pseudoprofile_ids:
            fut = ex.submit(
                _process_one_pseudoprofile,
                pid,
                run_id=run_id,
                profiles_dir=profiles_dir,
                mapped_criterions_path=args.mapped_criterions_path,
                ontology_path=args.high_level_BioPsychoSocial_ontology,
                max_ontology_chars=args.max_ontology_chars,
                max_output_tokens=args.max_output_tokens,
                ontology_db=ontology_db,
                global_top_n=args.global_top_n,
                per_solution_top_k=args.per_solution_top_k,
                config=config,
                use_cache=bool(args.use_cache),
                recompute_rankings_if_missing=bool(args.recompute_rankings_if_missing),
            )
            fut_map[fut] = pid

        _log("", f"Submitted {len(fut_map)} job(s) to ThreadPoolExecutor")

        done_count = 0
        for fut in as_completed(fut_map):
            pid = fut_map[fut]
            res = fut.result()
            done_count += 1

            if not res.get("ok"):
                err = res.get("error", "")
                error_rows.append({"run_id": run_id, "pseudoprofile_id": pid, "error": err})
                _log("", f"[{done_count}/{len(pseudoprofile_ids)}] FAIL {pid}")
                continue

            profile: ProfileInput = res["profile"]
            llm_result: Dict[str, Any] = res["llm_result"]
            rankings: Dict[str, Any] = res["rankings"]
            from_cache = bool(res.get("from_cache", False))

            _log("", f"[{done_count}/{len(pseudoprofile_ids)}] OK   {pid} (from_cache={from_cache})")

            # Optional printing for debugging/inspection (prints are heavy; tune via args)
            if (not from_cache) or args.print_global_top_n > 0 or args.print_per_solution_top_n > 0:
                try:
                    print_human_summary(llm_result)
                    print_fused_rankings(
                        rankings,
                        global_print_n=int(args.print_global_top_n),
                        per_solution_print_n=int(args.print_per_solution_top_n),
                    )
                except Exception as e:
                    _log(pid, f"WARNING: print block failed: {type(e).__name__}: {e}")

            dense_rows.append(
                _make_dense_row(
                    run_id,
                    profile,
                    llm_result,
                    rankings,
                    dense_global_top_n=int(args.dense_global_top_n),
                )
            )
            _append_long_rows(long_rows, run_id, profile, llm_result, rankings)

    elapsed = time.time() - started_at
    _log("", f"All futures completed | elapsed={elapsed:.1f}s")

    dense_csv_path = os.path.join(run_dir, "dense_profiles.csv")
    long_csv_path = os.path.join(run_dir, "long_rankings.csv")
    errors_csv_path = os.path.join(run_dir, "errors.csv")

    _log("", "Writing CSV outputs ...")

    if dense_rows:
        pd.DataFrame(dense_rows).sort_values(["pseudoprofile_id"]).to_csv(dense_csv_path, index=False)
        _log("", f"Saved dense CSV : {dense_csv_path}")
    else:
        _log("", "No dense rows to write.")

    if long_rows:
        pd.DataFrame(long_rows).sort_values(["pseudoprofile_id", "scope", "solution_key", "leaf_rank"]).to_csv(long_csv_path, index=False)
        _log("", f"Saved long CSV  : {long_csv_path}")
    else:
        _log("", "No long rows to write.")

    if error_rows:
        pd.DataFrame(error_rows).sort_values(["pseudoprofile_id"]).to_csv(errors_csv_path, index=False)
        _log("", f"Saved errors CSV: {errors_csv_path}")
    else:
        pd.DataFrame(columns=["run_id", "pseudoprofile_id", "error"]).to_csv(errors_csv_path, index=False)
        _log("", f"Saved errors CSV (empty): {errors_csv_path}")

    _log("", "==============================")
    _log("", "BATCH RUN END")
    _log("", f"Run dir: {run_dir}")
    _log("", "==============================\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#TODO: include weight-based logic inside output of LLM itself (i.e., predictor-solution prioritization)
