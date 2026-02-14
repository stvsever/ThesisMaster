#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_operationalize_freetext_complaints.py

Goal:
    Operationalize MANY free-text mental health complaints into MULTIPLE ontology leaf nodes by:
      (1) ALWAYS running an LLM-based clinical-expert decomposition into distinct criteria/variables (ontology-agnostic).
      (2) Mapping EACH decomposed criterion to the best matching CRITERION ontology leaf node via hybrid retrieval.
      (3) BY DEFAULT running an LLM-based per-criterion adjudication step to select the single best leaf from top-N candidates,
          with an explicit possibility to return UNMAPPED.

This script loads caches created by embed_leaf_nodes.py:
  - EMBEDTEXT paths
  - FULL paths
  - LEXTEXT paths
  - embeddings .npy + norms .npy + meta .json

Batch pipeline:
  free_text_complaints.txt (pseudoprofile_FTC_IDxxx blocks)
        ↓
  For each complaint:
    LLM decomposition (ontology-agnostic) -> list of criteria variables
        ↓
    Batch embeddings (one call) for each criterion query text
        ↓
    For each criterion:
      Dense semantic retrieval (cosine over leaf EMBEDTEXT)
      Sparse lexical grounding (BM25 / token overlap / fuzzy match) on LEXTEXT
      Candidate pooling + fusion
      LLM adjudication -> single best leaf OR UNMAPPED
        ↓
    Save per-complaint run JSON artifact to OUTPUT_DIR/operationalizations/
        ↓
  Write a SINGLE CSV with one row per decomposed variable (including unmapped)

Outputs:
  - CSV: mapped_criterions.csv (default path below)
  - JSON artifacts: OUTPUT_DIR/operationalizations/run_*.json

Notes:
  - GUI removed; no interactive interface.
  - Uses ThreadPoolExecutor across complaints to speed up processing (configurable).
  - Keeps the SAME decomposition prompt (verbatim) as provided.

Prerequisites:
  pip install openai numpy
Optional:
  pip install python-dotenv
  pip install rapidfuzz
"""

from __future__ import annotations

import os
import re
import json
import time
import math
import csv
import hashlib
import threading
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import heapq
import argparse
import traceback
from pathlib import Path

import numpy as np

from threading import Lock
_csv_lock = Lock()

# --------------------------
# CONFIG (EDIT THESE)
# --------------------------
def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        has_eval = (candidate / "evaluation").exists() or (candidate / "Evaluation").exists()
        if has_eval and (candidate / "README.md").exists():
            return candidate
    raise RuntimeError("Could not locate repository root from 02_operationalize_freetext_complaints.py")


REPO_ROOT = _find_repo_root()

# Input: the free-text complaints file (pseudoprofile blocks)
DEFAULT_FREE_TEXT_PATH = str(
    REPO_ROOT / "evaluation/01_pseudoprofile(s)/free_text/free_text_complaints.txt"
)

# Output: the mapped CSV
DEFAULT_OUTPUT_CSV = str(
    REPO_ROOT / "evaluation/02_mental_health_issue_operationalization/mapped_criterions.csv"
)

# Ontology input file is not required here (we use cached leaf text/embeddings),
# but kept for parity / provenance logs.
input_file = str(
    REPO_ROOT / "src/SystemComponents/PHOENIX_ontology/separate/01_raw/CRITERION/steps/01_raw/aggregated/CRITERION_ontology.json"
)

# Cache directory containing outputs from embed_leaf_nodes.py
# You can override via env var CRITERION_CACHE_DIR.
DEFAULT_OUTPUT_DIR = str(
    REPO_ROOT / "src/SystemComponents/Agentic_Framework/01_OperationalizationMentalHealthProblem/tmp"
)
OUTPUT_DIR = os.environ.get("CRITERION_CACHE_DIR", DEFAULT_OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cache other (produced by embed_leaf_nodes.py) ; TODO: if RDoC needs to be skipped --> remove all embeddings first --> then re-run script cs now it still uses them
PATHS_JSON = os.path.join(OUTPUT_DIR, "CRITERION_leaf_paths_EMBEDTEXT.json")
PATHS_FULL_JSON = os.path.join(OUTPUT_DIR, "CRITERION_leaf_paths_FULL.json")
PATHS_LEX_JSON = os.path.join(OUTPUT_DIR, "CRITERION_leaf_paths_LEXTEXT.json")

EMB_NPY = os.path.join(OUTPUT_DIR, "CRITERION_leaf_embeddings.npy")
NORM_NPY = os.path.join(OUTPUT_DIR, "CRITERION_leaf_embedding_norms.npy")
META_JSON = os.path.join(OUTPUT_DIR, "CRITERION_leaf_embeddings_meta.json")

# Embeddings
EMBED_MODEL = os.environ.get("CRITERION_EMBED_MODEL", "text-embedding-3-small")
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("OPENAI_TIMEOUT_S", "90.0"))

# LLM Decomposition (ALWAYS ON)
DECOMP_MODEL = os.environ.get("CRITERION_DECOMP_MODEL", "gpt-5-mini")  # NOTE: change to gpt-5 for actual non-test run
DECOMP_TEMPERATURE = float(os.environ.get("CRITERION_DECOMP_T", "1.0"))

# Per-criterion optional LLM picker (DEFAULT ON, as requested)
ENABLE_LLM_RERANKER_DEFAULT = True
LLM_RERANK_MODEL = os.environ.get("CRITERION_RERANK_MODEL", "gpt-5-nano")
LLM_RERANK_TOPN = int(os.environ.get("CRITERION_RERANK_TOPN", "200"))  # 50 --> 200
LLM_RERANK_TEMPERATURE = float(os.environ.get("CRITERION_RERANK_T", "1.0"))

# Retrieval weights (must sum to 1)
WEIGHT_EMBED = 0.80
WEIGHT_BM25 = 0.12
WEIGHT_TOKEN_OVERLAP = 0.05
WEIGHT_FUZZY = 0.03
assert abs((WEIGHT_EMBED + WEIGHT_BM25 + WEIGHT_TOKEN_OVERLAP + WEIGHT_FUZZY) - 1.0) < 1e-9

# Fusion behavior
# "htssf" = Hybrid Temperature-Scaled Softmax Fusion + RRF backstop (recommended)
# "rrf"   = weighted RRF only
# "scoresum" = minmax-normalized weighted sum (baseline)
FUSION_METHOD = "htssf"
RRF_K = 60

# HTSSF params
HTSSF_ALPHA = 0.90
HTSSF_TEMPS = (0.07, 1.00, 0.35, 0.35)

# Candidate settings
CANDIDATES_PER_METHOD = 600
TOP_K_RESULTS = 200  # 50 --> 200
CANDIDATE_POOL = 8000

# Decomposition -> retrieval query building
# If True, retrieval query text = "<criterion> | evidence: <evidence>"
INCLUDE_EVIDENCE_IN_QUERY = True

# Parallelism: per-complaint (outer) and per-criterion (inner)
MAX_PARALLEL_COMPLAINTS = int(os.environ.get("CRITERION_MAX_PARALLEL_COMPLAINTS", "2"))
MAX_PARALLEL_CRITERIA = int(os.environ.get("CRITERION_MAX_PARALLEL", "3"))

# Retry/backoff
MAX_RETRIES = 8
BACKOFF_BASE_SECONDS = 0.8
BACKOFF_MAX_SECONDS = 20.0

# Norm computation chunking
NORM_CHUNK_ROWS = 4096

# Output artifacts
RUNS_DIR = os.path.join(OUTPUT_DIR, "operationalizations")
os.makedirs(RUNS_DIR, exist_ok=True)

# --------------------------
# Pseudoprofile-level cache (deterministic)
# --------------------------

CACHE_RUNS_DIR = os.path.join(OUTPUT_DIR, "operationalizations_cache")
os.makedirs(CACHE_RUNS_DIR, exist_ok=True)

def _canonical_text(s: str) -> str:
    # Normalize line endings and trim; keeps semantics stable across OS differences.
    return s.replace("\r\n", "\n").replace("\r", "\n").strip()

def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _config_fingerprint(enable_llm_reranker: bool) -> str:
    cfg = {
        "embed_model": EMBED_MODEL,
        "decomp_model": DECOMP_MODEL,
        "decomp_temperature": DECOMP_TEMPERATURE,
        "enable_llm_reranker": bool(enable_llm_reranker),
        "rerank_model": (LLM_RERANK_MODEL if enable_llm_reranker else None),
        "rerank_temperature": LLM_RERANK_TEMPERATURE,
        "rerank_topn": LLM_RERANK_TOPN,
        "fusion_method": FUSION_METHOD,
        "weights": {
            "embed": WEIGHT_EMBED,
            "bm25": WEIGHT_BM25,
            "token_overlap": WEIGHT_TOKEN_OVERLAP,
            "fuzzy": WEIGHT_FUZZY,
        },
        "htssf": {"alpha": HTSSF_ALPHA, "temps": HTSSF_TEMPS, "rrf_k": RRF_K},
        "candidate_pool": CANDIDATE_POOL,
        "top_k_results": TOP_K_RESULTS,
        "include_evidence_in_query": INCLUDE_EVIDENCE_IN_QUERY,
        "unmapped_enabled": True,
    }
    blob = json.dumps(cfg, ensure_ascii=False, sort_keys=True)
    return _sha256_hex(blob)[:12]

def pseudoprofile_cache_path(pseudoprofile_id: str, complaint_text: str, enable_llm_reranker: bool) -> str:
    cfg_h = _config_fingerprint(enable_llm_reranker)
    txt_h = _sha256_hex(_canonical_text(complaint_text))[:12]
    fname = f"{pseudoprofile_id}__cfg{cfg_h}__txt{txt_h}.json"
    return os.path.join(CACHE_RUNS_DIR, fname)

def atomic_write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_completed_pids_from_csv(csv_path: str) -> set:
    """
    Resume support: if output CSV already has pseudoprofile_id rows, skip them.

    IMPORTANT:
      Do NOT treat pseudoprofiles with ERROR rows as completed (so you can re-run them after fixes).
    """
    if not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0:
        return set()
    done = set()
    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            rdr = csv.DictReader(f)
            if "pseudoprofile_id" not in (rdr.fieldnames or []):
                return set()
            for row in rdr:
                pid = (row.get("pseudoprofile_id") or "").strip()
                if not pid:
                    continue
                mapping_status = (row.get("mapping_status") or "").strip().upper()
                err = (row.get("error") or "").strip()
                # Only mark as "done" when not an ERROR row
                if mapping_status != "ERROR" and not err:
                    done.add(pid)
    except Exception:
        return set()
    return done

# CSV columns (pseudoprofile_id MUST be first column, per request)
CSV_FIELDNAMES: List[str] = [
    "pseudoprofile_id",
    "complaint_text",
    "decomp_n_variables",
    "decomp_notes",
    "variable_id",
    "variable_label",
    "variable_criterion",
    "variable_evidence",
    "variable_polarity",
    "variable_timeframe",
    "variable_severity_0_1",
    "variable_confidence_0_1",
    "query_text_used",
    "mapping_status",  # MAPPED | UNMAPPED | NO_RESULTS | ERROR
    "chosen_method",   # llm | top_fused | top_fused_fallback | none
    "chosen_idx",
    "chosen_leaf_embed_path",
    "chosen_leaf_full_path",
    "chosen_confidence",
    "chosen_rationale",
    "top_fused_idx",
    "top_fused_leaf_embed_path",
    "top_fused_leaf_full_path",
    "top_fused_score",
    "top_fused_emb",
    "top_fused_bm25",
    "top_fused_tok",
    "top_fused_fuz",
    "debug_pool_size",
    "debug_emb_rank_n",
    "debug_bm25_rank_n",
    "top50_candidates",
    "complaint_unique_mapped_leaf_indices",
    "complaint_unique_mapped_leaf_embed_paths",
    "run_json_path",
    "error",
]

# --------------------------
# Logging helpers
# --------------------------

def ts() -> str:
    return time.strftime("%H:%M:%S")

def log(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)

# --------------------------
# Detailed progress logging
# --------------------------

def log_stage(
    stage: str,
    *,
    pseudoprofile_id: Optional[str] = None,
    complaint_preview: Optional[str] = None,
    criterion_id: Optional[str] = None,
    criterion_label: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Structured, human-readable progress logging.
    Safe to call from multiple threads.
    """
    parts = [f"[stage={stage}]"]

    if pseudoprofile_id:
        parts.append(f"id={pseudoprofile_id}")

    if criterion_id:
        parts.append(f"criterion={criterion_id}")

    if criterion_label:
        parts.append(f"label='{criterion_label}'")

    if complaint_preview:
        preview = complaint_preview.strip().replace("\n", " ")
        if len(preview) > 120:
            preview = preview[:117] + "..."
        parts.append(f"text='{preview}'")

    if extra:
        for k, v in extra.items():
            parts.append(f"{k}={v}")

    print(f"[{ts()}] " + " | ".join(parts), flush=True)

def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024.0:
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{x:.2f} PB"

# --------------------------
# Optional dotenv loading
# --------------------------

def load_env_if_possible() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
        return
    except Exception:
        pass

    for p in [os.path.join(os.getcwd(), ".env"), os.path.join(os.path.dirname(__file__), ".env")]:
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k and k not in os.environ:
                            os.environ[k] = v
            except Exception:
                pass


# --------------------------
# Token normalization + light stemming
# --------------------------

WORD_RE = re.compile(r"[A-Za-z0-9]+")

_SUFFIXES = [
    "ingly", "edly", "ably",
    "ments", "ment",
    "able", "ibly",
    "ation", "tions", "tion",
    "ness", "less", "ful",
    "ing", "ed",
    "ies", "es", "s",
    "ive", "ity", "al",
    "er", "or",
]

def normalize_token(tok: str) -> str:
    t = tok.lower().strip()
    t = re.sub(r"[^a-z0-9]+", "", t)
    if len(t) <= 2:
        return t
    for suf in _SUFFIXES:
        if len(t) > (len(suf) + 2) and t.endswith(suf):
            if suf == "ies":
                t = t[:-3] + "y"
            else:
                t = t[:-len(suf)]
            break
    return t

def tokenize_norm(text: str) -> List[str]:
    s = text.replace("_", " ").replace("-", " ").replace("/", " ").replace("*", " ")
    s = re.sub(r"\s+", " ", s)
    toks = [normalize_token(m.group(0)) for m in WORD_RE.finditer(s)]
    toks = [t for t in toks if t and len(t) > 1]
    return toks


# --------------------------
# BM25 (sparse inverted index)
# --------------------------

@dataclass
class BM25Sparse:
    doc_len: np.ndarray
    avgdl: float
    idf: Dict[str, float]
    postings: Dict[str, List[Tuple[int, int]]]
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def build(cls, docs: List[str], k1: float = 1.5, b: float = 0.75) -> "BM25Sparse":
        log("[index] Tokenizing + building BM25 postings (sparse inverted index)...")
        toks_per_doc: List[List[str]] = [tokenize_norm(d) for d in docs]
        doc_len = np.array([len(t) for t in toks_per_doc], dtype=np.float32)
        avgdl = float(doc_len.mean()) if len(doc_len) else 0.0

        df: Dict[str, int] = {}
        postings: Dict[str, List[Tuple[int, int]]] = {}

        for i, toks in enumerate(toks_per_doc):
            if not toks:
                continue
            tf: Dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            for t, f in tf.items():
                postings.setdefault(t, []).append((i, f))
            for t in tf.keys():
                df[t] = df.get(t, 0) + 1

        N = len(docs)
        idf: Dict[str, float] = {}
        for t, n_q in df.items():
            idf[t] = math.log(1.0 + (N - n_q + 0.5) / (n_q + 0.5))

        log(f"[index] BM25 built. docs={N}, vocab={len(idf)}")
        return cls(doc_len=doc_len, avgdl=avgdl, idf=idf, postings=postings, k1=k1, b=b)

    def score_topk(self, query: str, k: int) -> Tuple[np.ndarray, Dict[int, float]]:
        q_tokens = tokenize_norm(query)
        if not q_tokens or self.avgdl == 0.0:
            return np.array([], dtype=np.int64), {}

        scores: Dict[int, float] = {}
        for qt in q_tokens:
            idf = self.idf.get(qt)
            if idf is None:
                continue
            plist = self.postings.get(qt)
            if not plist:
                continue
            for doc_id, f in plist:
                dl = float(self.doc_len[doc_id])
                denom_norm = self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
                s = idf * (f * (self.k1 + 1.0)) / (f + denom_norm)
                scores[doc_id] = scores.get(doc_id, 0.0) + s

        if not scores:
            return np.array([], dtype=np.int64), {}

        top = heapq.nlargest(k, scores.items(), key=lambda kv: kv[1])
        ranked = np.array([doc_id for doc_id, _ in top], dtype=np.int64)
        return ranked, scores


def token_overlap_score_subset(query: str, docs: List[str], idxs: np.ndarray) -> Dict[int, float]:
    q = set(tokenize_norm(query))
    out: Dict[int, float] = {}
    if not q or idxs.size == 0:
        return out
    for i in idxs.tolist():
        dt = set(tokenize_norm(docs[i]))
        if not dt:
            continue
        inter = len(q & dt)
        union = len(q | dt)
        out[i] = (float(inter) / float(union)) if union else 0.0
    return out


def fuzzy_score_subset(query: str, docs: List[str], idxs: np.ndarray) -> Dict[int, float]:
    q = query.strip().lower()
    out: Dict[int, float] = {}
    if not q or idxs.size == 0:
        return out

    try:
        from rapidfuzz.fuzz import ratio  # type: ignore
        for i in idxs.tolist():
            out[i] = float(ratio(q, docs[i].lower())) / 100.0
        return out
    except Exception:
        import difflib
        for i in idxs.tolist():
            out[i] = difflib.SequenceMatcher(None, q, docs[i].lower()).ratio()
        return out


# --------------------------
# Embeddings helpers (batch + norms)
# --------------------------

def stable_hash_of_paths(paths: List[str]) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


_thread_local = threading.local()

def make_openai_client():
    from openai import OpenAI
    try:
        return OpenAI(timeout=REQUEST_TIMEOUT_SECONDS, max_retries=0)
    except TypeError:
        try:
            return OpenAI(timeout=REQUEST_TIMEOUT_SECONDS)
        except TypeError:
            return OpenAI()

def get_thread_client():
    c = getattr(_thread_local, "client", None)
    if c is None:
        c = make_openai_client()
        _thread_local.client = c
    return c


def embed_texts_with_retry(texts: List[str], model: str) -> List[List[float]]:
    cleaned = [t.replace("\n", " ") for t in texts]

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            client = get_thread_client()
            try:
                resp = client.embeddings.create(
                    model=model,
                    input=cleaned,
                    encoding_format="float",
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
            except TypeError:
                resp = client.embeddings.create(
                    model=model,
                    input=cleaned,
                    encoding_format="float",
                )
            return [item.embedding for item in resp.data]
        except Exception as e:
            msg = str(e)
            sleep_s = min(BACKOFF_MAX_SECONDS, BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))
            sleep_s = sleep_s * (0.9 + 0.2 * ((time.time() * 997) % 1.0))
            log(f"[embed] attempt {attempt}/{MAX_RETRIES} failed: {type(e).__name__}: {msg}")
            if attempt >= MAX_RETRIES:
                raise
            log(f"[embed] backing off {sleep_s:.2f}s...")
            time.sleep(sleep_s)

    raise RuntimeError("Unreachable: retry loop ended unexpectedly.")


def compute_and_cache_norms(emb_path: str, norms_path: str) -> np.ndarray:
    log("[norm] Computing embedding row norms (cached, chunked)...")
    emb = np.load(emb_path, mmap_mode="r")
    n = emb.shape[0]
    norms = np.empty((n,), dtype=np.float32)

    t0 = time.time()
    done = 0
    for start in range(0, n, NORM_CHUNK_ROWS):
        end = min(n, start + NORM_CHUNK_ROWS)
        chunk = np.asarray(emb[start:end, :], dtype=np.float32)
        norms[start:end] = np.linalg.norm(chunk, axis=1).astype(np.float32)
        done = end
        if (done == n) or ((done // NORM_CHUNK_ROWS) % 10 == 0):
            pct = (done / n) * 100.0
            rate = done / max(1e-9, (time.time() - t0))
            log(f"[norm] progress: {done}/{n} ({pct:.2f}%) | {rate:.1f} rows/s")

    tmp = norms_path + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, norms)
    os.replace(tmp, norms_path)

    log(f"[norm] Saved norms -> {norms_path} ({human_bytes(os.path.getsize(norms_path))})")
    return norms


def topk_from_scores(scores: np.ndarray, k: int) -> np.ndarray:
    n = scores.shape[0]
    if k <= 0 or n == 0:
        return np.array([], dtype=np.int64)
    k = min(k, n)
    part = np.argpartition(-scores, k - 1)[:k]
    part = part[np.argsort(-scores[part], kind="mergesort")]
    return part.astype(np.int64)


def weighted_rrf_fusion(
    rankings: List[np.ndarray],
    weights: List[float],
    rrf_k: int,
    topn_per_method: int,
) -> Dict[int, float]:
    fused: Dict[int, float] = {}
    for r, w in zip(rankings, weights):
        limit = min(topn_per_method, len(r))
        for rank_pos in range(limit):
            idx = int(r[rank_pos])
            fused[idx] = fused.get(idx, 0.0) + (w / float(rrf_k + rank_pos + 1))
    return fused


# --------------------------
# Fusion helpers
# --------------------------

def safe_minmax_norm(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    mn = float(np.min(x))
    mx = float(np.max(x))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)

def fuse_scoresum_on_pool(
    pool_ids: np.ndarray,
    emb_scores_all: np.ndarray,
    bm25_map: Dict[int, float],
    tok_map: Dict[int, float],
    fuz_map: Dict[int, float],
) -> Dict[int, float]:
    pool_ids = pool_ids.astype(np.int64)
    emb_pool = emb_scores_all[pool_ids].astype(np.float32)

    bm25_pool = np.array([bm25_map.get(int(i), 0.0) for i in pool_ids], dtype=np.float32)
    tok_pool = np.array([tok_map.get(int(i), 0.0) for i in pool_ids], dtype=np.float32)
    fuz_pool = np.array([fuz_map.get(int(i), 0.0) for i in pool_ids], dtype=np.float32)

    emb_n = safe_minmax_norm(emb_pool)
    bm25_n = safe_minmax_norm(bm25_pool)
    tok_n = safe_minmax_norm(tok_pool)
    fuz_n = safe_minmax_norm(fuz_pool)

    fused: Dict[int, float] = {}
    for j, doc_id in enumerate(pool_ids.tolist()):
        fused[int(doc_id)] = (
            WEIGHT_EMBED * float(emb_n[j]) +
            WEIGHT_BM25 * float(bm25_n[j]) +
            WEIGHT_TOKEN_OVERLAP * float(tok_n[j]) +
            WEIGHT_FUZZY * float(fuz_n[j])
        )
    return fused

def softmax_probs(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if scores.size == 0:
        return scores
    t = max(1e-6, float(temperature))
    x = scores.astype(np.float64, copy=False) / t
    x = x - float(np.max(x))
    e = np.exp(x)
    s = float(e.sum())
    return (e / (s + 1e-12)).astype(np.float32)

def hybrid_softmax_rrf_fusion(
    pool_ids: np.ndarray,
    emb_scores: np.ndarray,
    bm25_map: Dict[int, float],
    tok_map: Dict[int, float],
    fuz_map: Dict[int, float],
    weights: Tuple[float, float, float, float],
    temps: Tuple[float, float, float, float] = HTSSF_TEMPS,
    alpha: float = HTSSF_ALPHA,
    rrf_k: int = RRF_K,
    rankings_for_rrf: Optional[List[np.ndarray]] = None,
    topn_per_method: int = CANDIDATES_PER_METHOD,
) -> Dict[int, float]:
    """
    HTSSF: Temperature-scaled softmax score fusion over candidate pool + small RRF backstop.
    - Softmax converts each method into a distribution over pool (scale-free after temperature).
    - BM25 is heavy-tailed: log1p before softmax reduces domination by spikes.
    - RRF backstop provides stability when score distributions are weird.
    """
    w_embed, w_bm25, w_tok, w_fuz = weights

    ids = pool_ids.astype(np.int64, copy=False)

    s_embed = emb_scores[ids].astype(np.float32, copy=False)

    s_bm25 = np.array([bm25_map.get(int(i), 0.0) for i in ids], dtype=np.float32)
    s_bm25 = np.log1p(np.maximum(s_bm25, 0.0))

    s_tok = np.array([tok_map.get(int(i), 0.0) for i in ids], dtype=np.float32)
    s_fuz = np.array([fuz_map.get(int(i), 0.0) for i in ids], dtype=np.float32)

    p_embed = softmax_probs(s_embed, temperature=temps[0])
    p_bm25  = softmax_probs(s_bm25,  temperature=temps[1])
    p_tok   = softmax_probs(s_tok,   temperature=temps[2])
    p_fuz   = softmax_probs(s_fuz,   temperature=temps[3])

    fused_score = (w_embed * p_embed + w_bm25 * p_bm25 + w_tok * p_tok + w_fuz * p_fuz)

    fused: Dict[int, float] = {int(i): float(alpha * s) for i, s in zip(ids.tolist(), fused_score.tolist())}

    beta = (1.0 - float(alpha))
    if rankings_for_rrf is not None and beta > 1e-9:
        rrf = weighted_rrf_fusion(
            rankings=rankings_for_rrf,
            weights=[w_embed, w_bm25, w_tok, w_fuz],
            rrf_k=rrf_k,
            topn_per_method=topn_per_method,
        )
        for i, s in rrf.items():
            fused[i] = fused.get(i, 0.0) + beta * float(s)

    return fused


# --------------------------
# LLM JSON helpers + decomposition
# --------------------------

def _extract_json_object(text: str) -> str:
    text = text.strip()
    # Prefer the largest JSON object if the model added extra prose.
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in LLM output.")
    return m.group(0)

def _repair_json_text_best_effort(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    # Normalize common “smart quotes”
    cleaned = cleaned.replace("“", "\"").replace("”", "\"").replace("’", "'").replace("‘", "'")
    # Try to isolate JSON
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start:end+1]
    # Remove trailing commas
    prev = None
    while prev != cleaned:
        prev = cleaned
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    return cleaned.strip()

def _safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        cleaned = _repair_json_text_best_effort(text)
        return json.loads(cleaned)

def _repair_json_with_llm(raw_text: str, model: str) -> str:
    """
    If an LLM produced almost-JSON but invalid JSON (missing commas, bad escaping, etc),
    ask the model to re-emit a STRICT valid JSON object.
    """
    client = make_openai_client()
    sys_msg = (
        "You are a strict JSON repair tool.\n"
        "Convert the user's provided text into ONE valid JSON object.\n"
        "Rules:\n"
        " - Output JSON ONLY (no markdown, no code fences, no commentary).\n"
        " - Preserve the meaning and fields from the original as much as possible.\n"
        " - Ensure the result is valid JSON parsable by Python json.loads.\n"
    )
    user_msg = "Repair into valid JSON object:\n\n" + raw_text

    text = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            try:
                try:
                    resp = client.responses.create(
                        model=model,
                        input=[
                            {"role": "system", "content": sys_msg},
                            {"role": "user", "content": user_msg},
                        ],
                        #temperature=0.0,
                        response_format={"type": "json_object"},
                    )
                    text = resp.output_text
                except Exception:
                    resp = client.responses.create(
                        model=model,
                        input=[
                            {"role": "system", "content": sys_msg},
                            {"role": "user", "content": user_msg},
                        ],
                        #temperature=0.0,
                    )
                    text = resp.output_text
            except Exception:
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": sys_msg},
                            {"role": "user", "content": user_msg},
                        ],
                        #temperature=0.0,
                        response_format={"type": "json_object"},
                    )
                    text = resp.choices[0].message.content
                except Exception:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": sys_msg},
                            {"role": "user", "content": user_msg},
                        ],
                        #temperature=0.0,
                    )
                    text = resp.choices[0].message.content
            break
        except Exception as e:
            msg = str(e)
            sleep_s = min(BACKOFF_MAX_SECONDS, BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))
            sleep_s = sleep_s * (0.9 + 0.2 * ((time.time() * 997) % 1.0))
            log(f"[json-repair] attempt {attempt}/{MAX_RETRIES} failed: {type(e).__name__}: {msg}")
            if attempt >= MAX_RETRIES:
                raise
            log(f"[json-repair] backing off {sleep_s:.2f}s...")
            time.sleep(sleep_s)

    if not text:
        raise RuntimeError("JSON repair LLM returned empty output.")
    return text


def llm_decompose_complaint(complaint: str) -> Dict[str, Any]:
    """
    Always-on decomposition that DOES NOT use the ontology.
    Output format (strict JSON object):
      {
        "meta": {"n_variables": int, "notes": str},
        "variables": [
          {
            "id": "C01",
            "label": "Sleep-maintenance insomnia",
            "criterion": "Wakes up repeatedly and cannot return to sleep",
            "evidence": "wake up too early, can't fall back asleep",
            "polarity": "present" | "absent" | "unclear",
            "timeframe": "e.g., weeks/months/unspecified",
            "severity_0_1": 0.0-1.0,
            "confidence_0_1": 0.0-1.0
          }, ...
        ]
      }
    """
    complaint = complaint.strip()
    if not complaint:
        raise ValueError("Complaint is empty.")

    client = make_openai_client()

    # IMPORTANT: keep prompt EXACTLY as provided
    sys_msg = (
        "You are a clinical-experimental healthcare expert.\n\n"
        "Task: Decompose a free-text mental health complaint into a optimal set of distinct, operational variables that preserve the overall resolution of the (non-)clinical mental health complain.\n"
        "Important constraints:\n"
        " - DO NOT use or reference any ontology, categories, diagnostic manuals, or leaf node names.\n"
        " - The variables must be state-optimizable factors that can be targeted in (non-)clinical interventions ; with focus on CURRENT issues"
        " - This is purely reasoning-based decomposition of what is described; be specific in your decomposition so that the full complaint is preserved in its full practical resolution.\n"
        " - Variables must be atomic and non-overlapping as much as possible.\n"
        " - Choose the GRANULARITY that matches the input: do not over-fragment short descriptions; do not under-specify rich ones.\n"
        " - Prefer 3–12 variables (but can be 0–20 if strongly justified by the text).\n"
        " - IMPORTANT: only decompose into set of variables that are changeable (e.g., no genetic, immutable, or distant factors).\n"
        "Output STRICT JSON ONLY (no markdown, no extra text):\n"
        "{\n"
        '  "meta": {"n_variables": <int>, "notes": <string>},\n'
        '  "variables": [\n'
        "    {\n"
        '      "id": "C01",\n'
        '      "label": <short human label>,\n'
        '      "criterion": <one-sentence operational statement>,\n'
        '      "evidence": <short quote/paraphrase grounded in the complaint>,\n'
        '      "polarity": "present"|"absent"|"unclear",\n'
        '      "timeframe": <short>,\n'
        '      "severity_0_1": <float 0-1>,\n'
        '      "confidence_0_1": <float 0-1>\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )

    user_msg = f"Complaint:\n'{complaint}'"

    text = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            try:
                try:
                    resp = client.responses.create(
                        model=DECOMP_MODEL,
                        input=[
                            {"role": "system", "content": sys_msg},
                            {"role": "user", "content": user_msg},
                        ],
                        #temperature=DECOMP_TEMPERATURE,
                        response_format={"type": "json_object"},
                    )
                    text = resp.output_text
                except Exception:
                    resp = client.responses.create(
                        model=DECOMP_MODEL,
                        input=[
                            {"role": "system", "content": sys_msg},
                            {"role": "user", "content": user_msg},
                        ],
                        #temperature=DECOMP_TEMPERATURE,
                    )
                    text = resp.output_text
            except Exception:
                try:
                    resp = client.chat.completions.create(
                        model=DECOMP_MODEL,
                        messages=[
                            {"role": "system", "content": sys_msg},
                            {"role": "user", "content": user_msg},
                        ],
                        #temperature=DECOMP_TEMPERATURE,
                        response_format={"type": "json_object"},
                    )
                    text = resp.choices[0].message.content
                except Exception:
                    resp = client.chat.completions.create(
                        model=DECOMP_MODEL,
                        messages=[
                            {"role": "system", "content": sys_msg},
                            {"role": "user", "content": user_msg},
                        ],
                        #temperature=DECOMP_TEMPERATURE,
                    )
                    text = resp.choices[0].message.content
            break
        except Exception as e:
            msg = str(e)
            sleep_s = min(BACKOFF_MAX_SECONDS, BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))
            sleep_s = sleep_s * (0.9 + 0.2 * ((time.time() * 997) % 1.0))
            log(f"[decomp] attempt {attempt}/{MAX_RETRIES} failed: {type(e).__name__}: {msg}")
            if attempt >= MAX_RETRIES:
                raise
            log(f"[decomp] backing off {sleep_s:.2f}s...")
            time.sleep(sleep_s)

    if not text:
        raise RuntimeError("Decomposition LLM returned empty output.")

    # Parse JSON (with repair fallback if needed)
    try:
        obj_text = _extract_json_object(text)
        data = _safe_json_loads(obj_text)
    except Exception as e:
        log(f"[decomp] JSON parse failed ({type(e).__name__}: {e}); attempting JSON repair...")
        repaired = _repair_json_with_llm(text, model=DECOMP_MODEL)
        obj_text = _extract_json_object(repaired)
        data = _safe_json_loads(obj_text)

    if not isinstance(data, dict) or "variables" not in data:
        raise RuntimeError(f"Decomposition JSON malformed. Raw:\n{obj_text}")

    vars_ = data.get("variables", [])
    if not isinstance(vars_, list) or not vars_:
        raise RuntimeError(f"Decomposition returned no variables. Raw:\n{obj_text}")

    # Basic normalization
    out_vars: List[Dict[str, Any]] = []
    for i, v in enumerate(vars_):
        if not isinstance(v, dict):
            continue
        vid = str(v.get("id") or f"C{i+1:02d}")
        label = str(v.get("label") or "").strip() or f"Variable {i+1}"
        crit = str(v.get("criterion") or "").strip()
        evid = str(v.get("evidence") or "").strip()
        polarity = str(v.get("polarity") or "unclear").strip().lower()
        if polarity not in {"present", "absent", "unclear"}:
            polarity = "unclear"
        timeframe = str(v.get("timeframe") or "unspecified").strip()
        try:
            sev = float(v.get("severity_0_1", 0.5))
        except Exception:
            sev = 0.5
        sev = max(0.0, min(1.0, sev))
        try:
            conf = float(v.get("confidence_0_1", 0.5))
        except Exception:
            conf = 0.5
        conf = max(0.0, min(1.0, conf))

        if not crit:
            # fallback: if criterion missing, use label/evidence
            crit = label if not evid else f"{label}: {evid}"

        out_vars.append({
            "id": vid,
            "label": label,
            "criterion": crit,
            "evidence": evid,
            "polarity": polarity,
            "timeframe": timeframe,
            "severity_0_1": sev,
            "confidence_0_1": conf,
        })

    meta = data.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    meta.setdefault("n_variables", len(out_vars))
    meta.setdefault("notes", "")

    return {"meta": meta, "variables": out_vars}


# --------------------------
# Optional LLM leaf picker (per criterion)
# --------------------------

def llm_pick_best_leaf_for_criterion(
    criterion_payload: Dict[str, Any],
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Pick the single best CRITERION leaf node for ONE decomposed criterion variable.
    Output STRICT JSON: {idx:int, confidence:float, rationale:str}

    NOTE:
      We enable UNMAPPED by passing a special candidate with idx=-1 and embed/full = "UNMAPPED".
      The prompt itself is kept unchanged; the model can select idx=-1 if appropriate.
    """
    client = make_openai_client()

    # Keep the original wording here (unchanged), and enable UNMAPPED via candidate list.
    sys_msg = (
        "You are a clinical ontology mapping assistant.\n"
        "Task: choose the SINGLE best-matching CRITERION ontology leaf node for ONE decomposed criterion variable.\n"
        "Rules:\n"
        " - Base decision primarily on semantic match to the criterion statement; use evidence/timeframe/polarity as secondary.\n"
        " - Prefer specific operational constructs over broad/general ones when clearly supported.\n"
        " - Choose exactly one leaf node that optimally (both 'semantically' and 'in description-resolution') matches the de criterion variable.\n"
        "Output STRICT JSON with keys: idx (int), confidence (0-1 float), rationale (string).\n"
    )

    crit_lines = [
        f"id: {criterion_payload.get('id')}",
        f"label: {criterion_payload.get('label')}",
        f"criterion: {criterion_payload.get('criterion')}",
        f"evidence: {criterion_payload.get('evidence')}",
        f"polarity: {criterion_payload.get('polarity')}",
        f"timeframe: {criterion_payload.get('timeframe')}",
        f"severity_0_1: {criterion_payload.get('severity_0_1')}",
        f"confidence_0_1: {criterion_payload.get('confidence_0_1')}",
    ]

    lines = []
    for c in candidates:
        lines.append(
            f"- idx={c['idx']} fused={c['fused_score']:.4f} "
            f"| embed={c['embed_path']} "
            f"| full={c['full_path']}"
        )

    user_msg = (
        "Decomposed criterion variable:\n"
        + "\n".join(crit_lines)
        + "\n\nCandidate leaf nodes:\n"
        + "\n".join(lines)
        + "\n\nPick the best idx."
    )

    text = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            try:
                try:
                    resp = client.responses.create(
                        model=LLM_RERANK_MODEL,
                        input=[
                            {"role": "system", "content": sys_msg},
                            {"role": "user", "content": user_msg},
                        ],
                        #temperature=LLM_RERANK_TEMPERATURE,
                        response_format={"type": "json_object"},
                    )
                    text = resp.output_text
                except Exception:
                    resp = client.responses.create(
                        model=LLM_RERANK_MODEL,
                        input=[
                            {"role": "system", "content": sys_msg},
                            {"role": "user", "content": user_msg},
                        ],
                        #temperature=LLM_RERANK_TEMPERATURE,
                    )
                    text = resp.output_text
            except Exception:
                try:
                    resp = client.chat.completions.create(
                        model=LLM_RERANK_MODEL,
                        messages=[
                            {"role": "system", "content": sys_msg},
                            {"role": "user", "content": user_msg},
                        ],
                        #temperature=LLM_RERANK_TEMPERATURE,
                        response_format={"type": "json_object"},
                    )
                    text = resp.choices[0].message.content
                except Exception:
                    resp = client.chat.completions.create(
                        model=LLM_RERANK_MODEL,
                        messages=[
                            {"role": "system", "content": sys_msg},
                            {"role": "user", "content": user_msg},
                        ],
                        #temperature=LLM_RERANK_TEMPERATURE,
                    )
                    text = resp.choices[0].message.content
            break
        except Exception as e:
            msg = str(e)
            sleep_s = min(BACKOFF_MAX_SECONDS, BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))
            sleep_s = sleep_s * (0.9 + 0.2 * ((time.time() * 997) % 1.0))
            log(f"[rerank] attempt {attempt}/{MAX_RETRIES} failed: {type(e).__name__}: {msg}")
            if attempt >= MAX_RETRIES:
                raise
            log(f"[rerank] backing off {sleep_s:.2f}s...")
            time.sleep(sleep_s)

    if not text:
        raise RuntimeError("LLM reranker returned empty output.")

    # Parse JSON (with repair fallback if needed)
    try:
        obj_text = _extract_json_object(text)
        obj = _safe_json_loads(obj_text)
    except Exception as e:
        log(f"[rerank] JSON parse failed ({type(e).__name__}: {e}); attempting JSON repair...")
        repaired = _repair_json_with_llm(text, model=LLM_RERANK_MODEL)
        obj_text = _extract_json_object(repaired)
        obj = _safe_json_loads(obj_text)

    if not isinstance(obj, dict) or "idx" not in obj:
        raise RuntimeError(f"LLM reranker JSON missing idx. Raw:\n{obj_text}")

    obj["idx"] = int(obj["idx"])
    try:
        obj["confidence"] = float(obj.get("confidence", 0.0))
    except Exception:
        obj["confidence"] = 0.0
    obj.setdefault("rationale", "")
    return obj


# --------------------------
# Search engine wrapper
# --------------------------

@dataclass
class SearchResult:
    idx: int
    fused: float
    emb: float
    bm25: float
    tok: float
    fuz: float
    embed_path: str
    full_path: str
    lex_text: str


class CriterionSearcher:
    def __init__(
        self,
        full_paths: List[str],
        embed_paths: List[str],
        lex_texts: List[str],
        emb: np.ndarray,
        norms: np.ndarray,
        bm25: BM25Sparse,
    ):
        self.full_paths = full_paths
        self.embed_paths = embed_paths
        self.lex_texts = lex_texts
        self.emb = emb
        self.norms = norms
        self.bm25 = bm25

        if self.emb.ndim != 2 or self.emb.shape[0] != len(self.embed_paths):
            raise RuntimeError(f"Embedding shape mismatch: emb={self.emb.shape}, paths={len(self.embed_paths)}")
        if self.norms.shape[0] != len(self.embed_paths):
            raise RuntimeError(f"Norms length mismatch: norms={self.norms.shape}, paths={len(self.embed_paths)}")

    def query_with_vec(self, query: str, q: np.ndarray) -> Tuple[List[SearchResult], Dict[str, Any]]:
        """
        Query using a PRECOMPUTED unit-norm query embedding q (np.float32).
        Returns:
            results: ranked SearchResult list
            debug: intermediate debug dict (scores, pool sizes)
        """
        query = query.strip()
        if not query:
            return [], {}

        q = q.astype(np.float32, copy=False)
        q = q / (float(np.linalg.norm(q)) + 1e-12)

        # Embedding similarity (cosine)
        dots = self.emb @ q
        emb_scores = dots / (self.norms + 1e-12)
        emb_rank = topk_from_scores(emb_scores, k=max(CANDIDATES_PER_METHOD, CANDIDATE_POOL))

        # BM25
        bm25_rank, bm25_map = self.bm25.score_topk(query, k=max(CANDIDATES_PER_METHOD, CANDIDATE_POOL))

        # Candidate pool = union(embedding, bm25), trimmed to CANDIDATE_POOL by embedding score
        pool_set = set(emb_rank.tolist()) | set(bm25_rank.tolist())
        pool_ids = np.fromiter(pool_set, dtype=np.int64)
        if pool_ids.size > CANDIDATE_POOL:
            local_scores = emb_scores[pool_ids]
            keep_local = topk_from_scores(local_scores, k=CANDIDATE_POOL)
            pool_ids = pool_ids[keep_local]

        # Token overlap + fuzzy on pool
        tok_map = token_overlap_score_subset(query, self.lex_texts, pool_ids)
        fuz_map = fuzzy_score_subset(query, self.lex_texts, pool_ids)

        # Rankings for RRF backstop
        tok_rank = np.array(
            [i for i, _ in heapq.nlargest(CANDIDATES_PER_METHOD, tok_map.items(), key=lambda kv: kv[1])],
            dtype=np.int64,
        )
        fuz_rank = np.array(
            [i for i, _ in heapq.nlargest(CANDIDATES_PER_METHOD, fuz_map.items(), key=lambda kv: kv[1])],
            dtype=np.int64,
        )

        # Fusion selection
        if FUSION_METHOD.lower() == "rrf":
            fused_map = weighted_rrf_fusion(
                rankings=[
                    emb_rank[:CANDIDATES_PER_METHOD],
                    bm25_rank[:CANDIDATES_PER_METHOD],
                    tok_rank,
                    fuz_rank,
                ],
                weights=[WEIGHT_EMBED, WEIGHT_BM25, WEIGHT_TOKEN_OVERLAP, WEIGHT_FUZZY],
                rrf_k=RRF_K,
                topn_per_method=CANDIDATES_PER_METHOD,
            )
        elif FUSION_METHOD.lower() == "scoresum":
            fused_map = fuse_scoresum_on_pool(
                pool_ids=pool_ids,
                emb_scores_all=emb_scores,
                bm25_map=bm25_map,
                tok_map=tok_map,
                fuz_map=fuz_map,
            )
        else:
            fused_map = hybrid_softmax_rrf_fusion(
                pool_ids=pool_ids,
                emb_scores=emb_scores,
                bm25_map=bm25_map,
                tok_map=tok_map,
                fuz_map=fuz_map,
                weights=(WEIGHT_EMBED, WEIGHT_BM25, WEIGHT_TOKEN_OVERLAP, WEIGHT_FUZZY),
                temps=HTSSF_TEMPS,
                alpha=HTSSF_ALPHA,
                rrf_k=RRF_K,
                rankings_for_rrf=[
                    emb_rank[:CANDIDATES_PER_METHOD],
                    bm25_rank[:CANDIDATES_PER_METHOD],
                    tok_rank,
                    fuz_rank,
                ],
                topn_per_method=CANDIDATES_PER_METHOD,
            )

        fused_items = sorted(fused_map.items(), key=lambda kv: kv[1], reverse=True)[:TOP_K_RESULTS]

        results: List[SearchResult] = []
        for idx, fscore in fused_items:
            i = int(idx)
            results.append(SearchResult(
                idx=i,
                fused=float(fscore),
                emb=float(emb_scores[i]),
                bm25=float(bm25_map.get(i, 0.0)),
                tok=float(tok_map.get(i, 0.0)),
                fuz=float(fuz_map.get(i, 0.0)),
                embed_path=self.embed_paths[i],
                full_path=self.full_paths[i],
                lex_text=self.lex_texts[i],
            ))

        debug = {
            "pool_size": int(pool_ids.size),
            "emb_rank_n": int(emb_rank.size),
            "bm25_rank_n": int(bm25_rank.size),
        }
        return results, debug


# --------------------------
# Multi-criteria mapping structures
# --------------------------

@dataclass
class CriterionVariable:
    id: str
    label: str
    criterion: str
    evidence: str
    polarity: str
    timeframe: str
    severity_0_1: float
    confidence_0_1: float

    def to_query_text(self) -> str:
        base = self.criterion.strip()
        if INCLUDE_EVIDENCE_IN_QUERY and self.evidence.strip():
            # return f"{base} | evidence: {self.evidence.strip()}" # ignore 'evidence' for now to reduce possible semantic noise
            return f"{base}"
        return base


@dataclass
class CriterionMapping:
    variable: CriterionVariable
    query_text_used: str
    results: List[SearchResult]
    chosen_method: str  # "top_fused" or "llm" or "top_fused_fallback" or "none"
    chosen_idx: int     # -1 => UNMAPPED
    chosen_confidence: float
    chosen_rationale: str
    debug: Dict[str, Any]

    @property
    def chosen_leaf_embed_path(self) -> str:
        if self.chosen_idx == -1:
            return "UNMAPPED"
        r = next((x for x in self.results if x.idx == self.chosen_idx), None)
        return r.embed_path if r else ""

    @property
    def chosen_leaf_full_path(self) -> str:
        if self.chosen_idx == -1:
            return "UNMAPPED"
        r = next((x for x in self.results if x.idx == self.chosen_idx), None)
        return r.full_path if r else ""


# --------------------------
# Load cached index
# --------------------------

def _require_file(path: str, hint: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing required cache file: {path}\nHint: {hint}")

def build_searcher_from_cache() -> Tuple[CriterionSearcher, Dict[str, Any]]:
    load_env_if_possible()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not found. Put it in .env as OPENAI_API_KEY=... or export it in your shell.")

    _require_file(PATHS_JSON, "Run: python embed_leaf_nodes.py")
    _require_file(PATHS_FULL_JSON, "Run: python embed_leaf_nodes.py")
    _require_file(PATHS_LEX_JSON, "Run: python embed_leaf_nodes.py")
    _require_file(EMB_NPY, "Run: python embed_leaf_nodes.py")
    _require_file(META_JSON, "Run: python embed_leaf_nodes.py (embeddings/meta)")

    log(f"[config] OUTPUT_DIR = {OUTPUT_DIR}")
    log(f"[config] input_file  = {input_file}")
    log(f"[config] EMBED_MODEL = {EMBED_MODEL}")
    log(f"[config] DECOMP_MODEL = {DECOMP_MODEL}")
    log(f"[config] RERANK_MODEL = {LLM_RERANK_MODEL}")
    log(f"[config] FUSION_METHOD = {FUSION_METHOD}")
    log(f"[config] weights: emb={WEIGHT_EMBED:.2f} bm25={WEIGHT_BM25:.2f} tok={WEIGHT_TOKEN_OVERLAP:.2f} fuz={WEIGHT_FUZZY:.2f}")
    if FUSION_METHOD.lower() == "htssf":
        log(f"[config] HTSSF: alpha={HTSSF_ALPHA:.2f}, temps={HTSSF_TEMPS}, rrf_k={RRF_K}")
    log("")

    embed_paths = json.load(open(PATHS_JSON, "r", encoding="utf-8"))
    full_paths = json.load(open(PATHS_FULL_JSON, "r", encoding="utf-8"))
    lex_texts = json.load(open(PATHS_LEX_JSON, "r", encoding="utf-8"))

    if not (isinstance(embed_paths, list) and isinstance(full_paths, list) and isinstance(lex_texts, list)):
        raise RuntimeError("Path caches are not lists (corrupted JSON).")

    if not (len(embed_paths) == len(full_paths) == len(lex_texts)):
        raise RuntimeError(
            f"Cache length mismatch: embed={len(embed_paths)} full={len(full_paths)} lex={len(lex_texts)}"
        )

    meta = json.load(open(META_JSON, "r", encoding="utf-8"))
    paths_hash = stable_hash_of_paths(embed_paths)
    if meta.get("paths_hash") != paths_hash:
        raise RuntimeError("Cache mismatch: paths_hash differs from META_JSON (paths changed or cache stale).")
    if meta.get("model") != EMBED_MODEL:
        raise RuntimeError(f"Cache mismatch: META_JSON model={meta.get('model')} but config EMBED_MODEL={EMBED_MODEL}.")
    if meta.get("status") != "complete":
        raise RuntimeError(f"Embeddings cache is not complete (status={meta.get('status')}). Re-run embed_leaf_nodes.py")

    emb = np.load(EMB_NPY, mmap_mode="r")
    if emb.shape[0] != len(embed_paths):
        raise RuntimeError(f"Embedding rows mismatch: emb_rows={emb.shape[0]} paths={len(embed_paths)}")

    if os.path.isfile(NORM_NPY):
        norms = np.load(NORM_NPY).astype(np.float32)
        if norms.shape[0] != len(embed_paths):
            log("[norm] Norm cache wrong length; recomputing.")
            norms = compute_and_cache_norms(EMB_NPY, NORM_NPY)
        else:
            log(f"[norm] Loaded norms: {norms.shape} ({human_bytes(os.path.getsize(NORM_NPY))})")
    else:
        norms = compute_and_cache_norms(EMB_NPY, NORM_NPY)

    bm25 = BM25Sparse.build(lex_texts)

    searcher = CriterionSearcher(
        full_paths=full_paths,
        embed_paths=embed_paths,
        lex_texts=lex_texts,
        emb=emb,
        norms=norms,
        bm25=bm25,
    )

    cache_info = {
        "output_dir": OUTPUT_DIR,
        "embed_model": EMBED_MODEL,
        "n_leaf_nodes": len(embed_paths),
        "meta": meta,
    }
    return searcher, cache_info


# --------------------------
# End-to-end operationalization (multi-criteria)
# --------------------------

def operationalize_complaint(
    searcher: CriterionSearcher,
    complaint: str,
    enable_llm_reranker: bool,
) -> Tuple[Dict[str, Any], List[CriterionMapping], str]:
    """
    Returns:
      decomposition_data, mappings, saved_json_path
    """
    t0 = time.time()

    log_stage(
        "DECOMPOSITION_START",
        complaint_preview=complaint,
    )

    # 1) Decompose (always-on)
    decomp = llm_decompose_complaint(complaint)
    variables_raw = decomp.get("variables", [])

    log_stage(
        "DECOMPOSITION_DONE",
        extra={"n_variables": len(decomp.get("variables", []))}
    )

    variables: List[CriterionVariable] = []
    for v in variables_raw:
        variables.append(CriterionVariable(
            id=str(v["id"]),
            label=str(v["label"]),
            criterion=str(v["criterion"]),
            evidence=str(v.get("evidence", "")),
            polarity=str(v.get("polarity", "unclear")),
            timeframe=str(v.get("timeframe", "unspecified")),
            severity_0_1=float(v.get("severity_0_1", 0.5)),
            confidence_0_1=float(v.get("confidence_0_1", 0.5)),
        ))

    # 2) Batch-embed each criterion query
    query_texts = [v.to_query_text() for v in variables]

    log_stage(
        "EMBED_CRITERIA_BATCH",
        extra={"n_criteria": len(query_texts)}
    )

    q_vecs = embed_texts_with_retry(query_texts, EMBED_MODEL)
    Q = np.array(q_vecs, dtype=np.float32)
    Q_norm = np.linalg.norm(Q, axis=1, keepdims=True).astype(np.float32) + 1e-12
    Q = Q / Q_norm

    # 3) For each criterion: retrieve (parallel modestly)
    mappings: List[CriterionMapping] = []

    def retrieve_one(i: int) -> CriterionMapping:
        var = variables[i]
        qtxt = query_texts[i]
        q = Q[i]

        results, debug = searcher.query_with_vec(qtxt, q)

        if not results:
            return CriterionMapping(
                variable=var,
                query_text_used=qtxt,
                results=[],
                chosen_method="none",
                chosen_idx=-1,  # treat as unmapped when no results
                chosen_confidence=0.0,
                chosen_rationale="No retrieval results.",
                debug=debug,
            )

        best = results[0]
        return CriterionMapping(
            variable=var,
            query_text_used=qtxt,
            results=results,
            chosen_method="top_fused",
            chosen_idx=best.idx,
            chosen_confidence=0.0,
            chosen_rationale="Selected top fused score.",
            debug=debug,
        )

    if len(variables) <= 1 or MAX_PARALLEL_CRITERIA <= 1:
        for i in range(len(variables)):
            mappings.append(retrieve_one(i))
    else:
        with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_CRITERIA, len(variables))) as ex:
            futs = {ex.submit(retrieve_one, i): i for i in range(len(variables))}
            for fut in as_completed(futs):
                mappings.append(fut.result())
        # keep deterministic order
        mappings.sort(key=lambda m: m.variable.id)

    # 4) Optional LLM reranking: run per-criterion pick in parallel (after decomposition + retrieval)
    if enable_llm_reranker:
        def rerank_one(m: CriterionMapping) -> CriterionMapping:
            if not m.results:
                # keep unmapped/no-results as is
                return m

            topn = min(LLM_RERANK_TOPN, len(m.results))

            # IMPORTANT: enable UNMAPPED by injecting a special candidate idx=-1
            cands = [{"idx": -1, "fused_score": 0.0, "embed_path": "UNMAPPED", "full_path": "UNMAPPED"}]
            cands += [
                {"idx": r.idx, "fused_score": r.fused, "embed_path": r.embed_path, "full_path": r.full_path}
                for r in m.results[:topn]
            ]

            log_stage(
                "LLM_ADJUDICATION_START",
                criterion_id=m.variable.id,
                criterion_label=m.variable.label,
                extra={"n_candidates": len(cands)},
            )

            try:
                pick = llm_pick_best_leaf_for_criterion(asdict(m.variable), cands)
            except Exception as e:
                best = m.results[0]
                log_stage(
                    "LLM_ADJUDICATION_ERROR",
                    criterion_id=m.variable.id,
                    criterion_label=m.variable.label,
                    extra={"error": f"{type(e).__name__}: {e}", "fallback": "top_fused"},
                )
                return CriterionMapping(
                    variable=m.variable,
                    query_text_used=m.query_text_used,
                    results=m.results,
                    chosen_method="top_fused_fallback",
                    chosen_idx=best.idx,
                    chosen_confidence=0.0,
                    chosen_rationale=f"LLM adjudication failed ({type(e).__name__}: {e}); fell back to top fused.",
                    debug=m.debug,
                )

            chosen_idx = int(pick["idx"])
            chosen_conf = float(pick.get("confidence", 0.0))
            chosen_rat = str(pick.get("rationale", ""))

            log_stage(
                "LLM_ADJUDICATION_DONE",
                criterion_id=m.variable.id,
                extra={
                    "chosen_idx": chosen_idx,
                    "confidence": f"{chosen_conf:.2f}",
                    "status": "UNMAPPED" if chosen_idx == -1 else "MAPPED",
                },
            )

            # Safety: if LLM returns an idx not in candidates, fall back to top fused
            cand_ids = {c["idx"] for c in cands}
            if chosen_idx not in cand_ids:
                best = m.results[0]
                return CriterionMapping(
                    variable=m.variable,
                    query_text_used=m.query_text_used,
                    results=m.results,
                    chosen_method="top_fused_fallback",
                    chosen_idx=best.idx,
                    chosen_confidence=0.0,
                    chosen_rationale=f"LLM returned idx not in candidate set; fell back to top fused. Raw pick={pick}",
                    debug=m.debug,
                )

            return CriterionMapping(
                variable=m.variable,
                query_text_used=m.query_text_used,
                results=m.results,
                chosen_method="llm",
                chosen_idx=chosen_idx,
                chosen_confidence=chosen_conf,
                chosen_rationale=chosen_rat,
                debug=m.debug,
            )

        if len(mappings) <= 1 or MAX_PARALLEL_CRITERIA <= 1:
            mappings = [rerank_one(m) for m in mappings]
        else:
            new_mappings: List[CriterionMapping] = []
            with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_CRITERIA, len(mappings))) as ex:
                futs = {ex.submit(rerank_one, m): m.variable.id for m in mappings}
                for fut in as_completed(futs):
                    new_mappings.append(fut.result())
            new_mappings.sort(key=lambda m: m.variable.id)
            mappings = new_mappings

    # 5) Save run artifact
    run_obj = {
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "embed_model": EMBED_MODEL,
            "decomp_model": DECOMP_MODEL,
            "rerank_model": LLM_RERANK_MODEL,
            "enable_llm_reranker": bool(enable_llm_reranker),
            "fusion_method": FUSION_METHOD,
            "weights": {
                "embed": WEIGHT_EMBED,
                "bm25": WEIGHT_BM25,
                "token_overlap": WEIGHT_TOKEN_OVERLAP,
                "fuzzy": WEIGHT_FUZZY,
            },
            "candidate_pool": CANDIDATE_POOL,
            "top_k_results": TOP_K_RESULTS,
            "llm_rerank_topn": LLM_RERANK_TOPN,
        },
        "complaint": complaint,
        "decomposition": decomp,
        "mappings": [
            {
                "variable": asdict(m.variable),
                "query_text_used": m.query_text_used,
                "chosen": {
                    "method": m.chosen_method,
                    "idx": m.chosen_idx,
                    "confidence": m.chosen_confidence,
                    "rationale": m.chosen_rationale,
                    "leaf_embed_path": m.chosen_leaf_embed_path,
                    "leaf_full_path": m.chosen_leaf_full_path,
                },
                "debug": m.debug,
                "top_results": [
                    {
                        "idx": r.idx,
                        "fused": r.fused,
                        "emb": r.emb,
                        "bm25": r.bm25,
                        "tok": r.tok,
                        "fuz": r.fuz,
                        "embed_path": r.embed_path,
                        "full_path": r.full_path,
                        "lex_text": r.lex_text,
                    }
                    for r in m.results
                ],
            }
            for m in mappings
        ],
        "runtime_seconds": float(time.time() - t0),
    }

    # deterministic-ish filename using hash of complaint + timestamp
    h = hashlib.sha256(complaint.encode("utf-8")).hexdigest()[:12]
    fname = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{h}.json"
    out_path = os.path.join(RUNS_DIR, fname)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(run_obj, f, ensure_ascii=False, indent=2)

    return decomp, mappings, out_path


# --------------------------
# Parsing free_text_complaints.txt
# --------------------------

ID_RE = re.compile(r"^\s*(pseudoprofile_FTC_ID\d{3})\s*$", flags=re.IGNORECASE)

def parse_free_text_complaints(path: str) -> List[Tuple[str, str]]:
    """
    Parse a file containing blocks like:

      pseudoprofile_FTC_ID001
      <free text ...>
      (blank lines allowed)
      pseudoprofile_FTC_ID002
      <...>

    Returns list of (pseudoprofile_id, complaint_text).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Free-text complaints file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    items: List[Tuple[str, str]] = []
    cur_id: Optional[str] = None
    buf: List[str] = []

    def flush():
        nonlocal cur_id, buf
        if cur_id is None:
            return
        text = "\n".join(buf).strip()
        # Light cleanup: strip wrapping quotes if present
        text = text.strip().strip('"').strip("'").strip()
        items.append((cur_id, text))
        cur_id = None
        buf = []

    for line in lines:
        m = ID_RE.match(line)
        if m:
            # start of a new block
            flush()
            cur_id = m.group(1)
            buf = []
            continue

        if cur_id is not None:
            buf.append(line)

    flush()

    # Filter empty complaint bodies (in case of malformed blocks)
    items = [(pid, txt) for pid, txt in items if txt.strip()]

    # Sort by numeric id if possible
    def key_fn(x: Tuple[str, str]) -> Tuple[int, str]:
        pid = x[0]
        mm = re.search(r"ID(\d{3})", pid, flags=re.IGNORECASE)
        if mm:
            return (int(mm.group(1)), pid)
        return (10**9, pid)

    items.sort(key=key_fn)
    return items


# --------------------------
# CSV row construction
# --------------------------

def _safe_join(parts: List[str], sep: str = "; ") -> str:
    parts2 = [p for p in parts if p]
    return sep.join(parts2)

def mappings_to_csv_rows(
    searcher: CriterionSearcher,
    pseudoprofile_id: str,
    complaint_text: str,
    decomp: Dict[str, Any],
    mappings: List[CriterionMapping],
    run_json_path: str,
    error: str = "",
) -> List[Dict[str, Any]]:
    meta = decomp.get("meta", {}) if isinstance(decomp, dict) else {}
    decomp_n = meta.get("n_variables", "")
    decomp_notes = meta.get("notes", "")

    # Compute unique mapped leaves per complaint (exclude -1 / unmapped)
    uniq_idxs: List[int] = []
    uniq_paths: List[str] = []
    seen = set()

    for m in mappings:
        if m.chosen_idx is None or int(m.chosen_idx) < 0:
            continue
        idx = int(m.chosen_idx)
        if idx in seen:
            continue
        seen.add(idx)
        uniq_idxs.append(idx)
        if 0 <= idx < len(searcher.embed_paths):
            uniq_paths.append(searcher.embed_paths[idx])

    uniq_idxs_str = _safe_join([str(i) for i in uniq_idxs], sep=";")
    uniq_paths_str = _safe_join(uniq_paths, sep=" ; ")

    rows: List[Dict[str, Any]] = []

    # If we had an error before decomposition/mapping, still produce one row.
    if error and not mappings:
        rows.append({
            "pseudoprofile_id": pseudoprofile_id,
            "complaint_text": complaint_text,
            "decomp_n_variables": "",
            "decomp_notes": "",
            "variable_id": "",
            "variable_label": "",
            "variable_criterion": "",
            "variable_evidence": "",
            "variable_polarity": "",
            "variable_timeframe": "",
            "variable_severity_0_1": "",
            "variable_confidence_0_1": "",
            "query_text_used": "",
            "mapping_status": "ERROR",
            "chosen_method": "none",
            "chosen_idx": "",
            "chosen_leaf_embed_path": "",
            "chosen_leaf_full_path": "",
            "chosen_confidence": "",
            "chosen_rationale": "",
            "top_fused_idx": "",
            "top_fused_leaf_embed_path": "",
            "top_fused_leaf_full_path": "",
            "top_fused_score": "",
            "top_fused_emb": "",
            "top_fused_bm25": "",
            "top_fused_tok": "",
            "top_fused_fuz": "",
            "debug_pool_size": "",
            "debug_emb_rank_n": "",
            "debug_bm25_rank_n": "",
            "top50_candidates": "",
            "complaint_unique_mapped_leaf_indices": "",
            "complaint_unique_mapped_leaf_embed_paths": "",
            "run_json_path": run_json_path,
            "error": error,
        })
        return rows

    for m in mappings:
        v = m.variable

        # Top fused (retrieval) info
        top = m.results[0] if m.results else None
        top_fused_idx = top.idx if top else ""
        top_fused_leaf_embed_path = top.embed_path if top else ""
        top_fused_leaf_full_path = top.full_path if top else ""
        top_fused_score = f"{top.fused:.6f}" if top else ""
        top_fused_emb = f"{top.emb:.6f}" if top else ""
        top_fused_bm25 = f"{top.bm25:.6f}" if top else ""
        top_fused_tok = f"{top.tok:.6f}" if top else ""
        top_fused_fuz = f"{top.fuz:.6f}" if top else ""

        # Chosen leaf (LLM or top_fused)
        chosen_idx = int(m.chosen_idx) if m.chosen_idx is not None else -1

        if chosen_idx == -1:
            chosen_leaf_embed_path = "UNMAPPED"
            chosen_leaf_full_path = "UNMAPPED"
            mapping_status = "UNMAPPED" if m.results else "NO_RESULTS"
        else:
            # Try mapping object first (if idx in top results)
            chosen_leaf_embed_path = m.chosen_leaf_embed_path
            chosen_leaf_full_path = m.chosen_leaf_full_path

            # If not found in results, fall back to cache arrays if idx is in range
            if (not chosen_leaf_embed_path) and (0 <= chosen_idx < len(searcher.embed_paths)):
                chosen_leaf_embed_path = searcher.embed_paths[chosen_idx]
            if (not chosen_leaf_full_path) and (0 <= chosen_idx < len(searcher.full_paths)):
                chosen_leaf_full_path = searcher.full_paths[chosen_idx]

            mapping_status = "MAPPED"

        # Top-5 candidate summary string
        top50 = []
        for r in (m.results[:50] if m.results else []):
            top50.append(f"idx={r.idx}|fused={r.fused:.4f}|leaf={r.embed_path}")
        top50_candidates = _safe_join(top50, sep=" || ")

        debug_pool_size = m.debug.get("pool_size", "")
        debug_emb_rank_n = m.debug.get("emb_rank_n", "")
        debug_bm25_rank_n = m.debug.get("bm25_rank_n", "")

        rows.append({
            "pseudoprofile_id": pseudoprofile_id,
            "complaint_text": complaint_text,
            "decomp_n_variables": decomp_n,
            "decomp_notes": decomp_notes,
            "variable_id": v.id,
            "variable_label": v.label,
            "variable_criterion": v.criterion,
            "variable_evidence": v.evidence,
            "variable_polarity": v.polarity,
            "variable_timeframe": v.timeframe,
            "variable_severity_0_1": f"{v.severity_0_1:.3f}",
            "variable_confidence_0_1": f"{v.confidence_0_1:.3f}",
            "query_text_used": m.query_text_used,
            "mapping_status": mapping_status,
            "chosen_method": m.chosen_method,
            "chosen_idx": str(chosen_idx),
            "chosen_leaf_embed_path": chosen_leaf_embed_path,
            "chosen_leaf_full_path": chosen_leaf_full_path,
            "chosen_confidence": f"{m.chosen_confidence:.3f}" if m.chosen_method == "llm" else "",
            "chosen_rationale": m.chosen_rationale,
            "top_fused_idx": str(top_fused_idx),
            "top_fused_leaf_embed_path": top_fused_leaf_embed_path,
            "top_fused_leaf_full_path": top_fused_leaf_full_path,
            "top_fused_score": top_fused_score,
            "top_fused_emb": top_fused_emb,
            "top_fused_bm25": top_fused_bm25,
            "top_fused_tok": top_fused_tok,
            "top_fused_fuz": top_fused_fuz,
            "debug_pool_size": str(debug_pool_size),
            "debug_emb_rank_n": str(debug_emb_rank_n),
            "debug_bm25_rank_n": str(debug_bm25_rank_n),
            "top50_candidates": top50_candidates,
            "complaint_unique_mapped_leaf_indices": uniq_idxs_str,
            "complaint_unique_mapped_leaf_embed_paths": uniq_paths_str,
            "run_json_path": run_json_path,
            "error": error,
        })

    return rows


def write_csv(rows: List[Dict[str, Any]], csv_path: str) -> None:
    out_dir = os.path.dirname(csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            # Ensure all expected keys exist
            for k in CSV_FIELDNAMES:
                if k not in row:
                    row[k] = ""
            writer.writerow(row)


# --------------------------
# Batch runner
# --------------------------

def process_one_pseudoprofile(
    searcher: CriterionSearcher,
    pseudoprofile_id: str,
    complaint_text: str,
    enable_llm_reranker: bool,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Returns (pseudoprofile_id, rows_for_csv).
    Uses deterministic cache JSON if available.
    """
    log_stage(
        "START_PSEUDOPROFILE",
        pseudoprofile_id=pseudoprofile_id,
        complaint_preview=complaint_text,
    )

    cache_path = pseudoprofile_cache_path(pseudoprofile_id, complaint_text, enable_llm_reranker)

    # 1) If cached JSON exists → load and return rows (skip compute)
    if os.path.isfile(cache_path) and os.path.getsize(cache_path) > 0:
        try:
            run_obj = load_json(cache_path)
            log_stage(
                "SKIP_CACHED_PSEUDOPROFILE",
                pseudoprofile_id=pseudoprofile_id,
                extra={"cache": os.path.basename(cache_path)},
            )
            log(f"[cache] skipped {pseudoprofile_id} bcs already cached")
            rows = run_obj_to_csv_rows_from_cache(
                searcher=searcher,
                pseudoprofile_id=pseudoprofile_id,
                complaint_text=complaint_text,
                run_obj=run_obj,
                run_json_path=cache_path,
            )
            return pseudoprofile_id, rows
        except Exception as e:
            log(f"[cache-warning] Failed to load cache for {pseudoprofile_id} ({type(e).__name__}: {e}). Recomputing...")

    # 2) Otherwise compute normally
    try:
        decomp, mappings, run_path = operationalize_complaint(
            searcher=searcher,
            complaint=complaint_text,
            enable_llm_reranker=enable_llm_reranker,
        )

        # Copy the timestamped run JSON to deterministic cache path
        try:
            run_obj = load_json(run_path)
            atomic_write_json(cache_path, run_obj)
        except Exception as e:
            log(f"[cache-warning] Could not write cache for {pseudoprofile_id}: {type(e).__name__}: {e}")

        # Use cache_path in CSV so it points to stable artifact
        rows = mappings_to_csv_rows(
            searcher=searcher,
            pseudoprofile_id=pseudoprofile_id,
            complaint_text=complaint_text,
            decomp=decomp,
            mappings=mappings,
            run_json_path=cache_path if os.path.isfile(cache_path) else run_path,
        )
        return pseudoprofile_id, rows

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        tb = traceback.format_exc()
        log(f"[error] {pseudoprofile_id} failed: {err}\n{tb}")
        rows = mappings_to_csv_rows(
            searcher=searcher,
            pseudoprofile_id=pseudoprofile_id,
            complaint_text=complaint_text,
            decomp={"meta": {}, "variables": []},
            mappings=[],
            run_json_path="",
            error=err,
        )
        return pseudoprofile_id, rows

def run_obj_to_csv_rows_from_cache(
    searcher: CriterionSearcher,
    pseudoprofile_id: str,
    complaint_text: str,
    run_obj: Dict[str, Any],
    run_json_path: str,
) -> List[Dict[str, Any]]:
    """
    Convert a cached run JSON (same schema as operationalize_complaint run_obj)
    into CSV rows without re-running LLMs.
    """
    decomp = run_obj.get("decomposition", {}) if isinstance(run_obj, dict) else {}
    meta = decomp.get("meta", {}) if isinstance(decomp, dict) else {}
    decomp_n = meta.get("n_variables", "")
    decomp_notes = meta.get("notes", "")

    mappings = run_obj.get("mappings", [])
    if not isinstance(mappings, list):
        mappings = []

    # unique mapped leaves per complaint
    uniq_idxs: List[int] = []
    uniq_paths: List[str] = []
    seen = set()
    for m in mappings:
        chosen = (m.get("chosen") or {}) if isinstance(m, dict) else {}
        try:
            idx = int(chosen.get("idx", -1))
        except Exception:
            idx = -1
        if idx >= 0 and idx not in seen:
            seen.add(idx)
            uniq_idxs.append(idx)
            p = str(chosen.get("leaf_embed_path") or "")
            if not p and 0 <= idx < len(searcher.embed_paths):
                p = searcher.embed_paths[idx]
            if p:
                uniq_paths.append(p)

    uniq_idxs_str = _safe_join([str(i) for i in uniq_idxs], sep=";")
    uniq_paths_str = _safe_join(uniq_paths, sep=" ; ")

    rows: List[Dict[str, Any]] = []

    for m in mappings:
        if not isinstance(m, dict):
            continue

        var = (m.get("variable") or {}) if isinstance(m.get("variable"), dict) else {}
        chosen = (m.get("chosen") or {}) if isinstance(m.get("chosen"), dict) else {}
        debug = (m.get("debug") or {}) if isinstance(m.get("debug"), dict) else {}
        top_results = m.get("top_results", [])
        if not isinstance(top_results, list):
            top_results = []

        qtxt = str(m.get("query_text_used") or "")

        # chosen
        chosen_method = str(chosen.get("method") or "none")
        try:
            chosen_idx = int(chosen.get("idx", -1))
        except Exception:
            chosen_idx = -1
        chosen_leaf_embed_path = str(chosen.get("leaf_embed_path") or ("UNMAPPED" if chosen_idx == -1 else ""))
        chosen_leaf_full_path = str(chosen.get("leaf_full_path") or ("UNMAPPED" if chosen_idx == -1 else ""))
        chosen_confidence = chosen.get("confidence", "")
        chosen_rationale = str(chosen.get("rationale") or "")

        if chosen_idx == -1:
            mapping_status = "UNMAPPED" if len(top_results) > 0 else "NO_RESULTS"
        else:
            mapping_status = "MAPPED"

        # top fused (= top_results[0])
        top0 = top_results[0] if top_results else {}
        def _getf(d, k, default=""):
            return d.get(k, default) if isinstance(d, dict) else default

        top_fused_idx = _getf(top0, "idx", "")
        top_fused_leaf_embed_path = _getf(top0, "embed_path", "")
        top_fused_leaf_full_path = _getf(top0, "full_path", "")
        top_fused_score = _getf(top0, "fused", "")
        top_fused_emb = _getf(top0, "emb", "")
        top_fused_bm25 = _getf(top0, "bm25", "")
        top_fused_tok = _getf(top0, "tok", "")
        top_fused_fuz = _getf(top0, "fuz", "")

        # top50 candidate summary
        top50 = []
        for r in top_results[:5]:
            if not isinstance(r, dict):
                continue
            top50.append(f"idx={r.get('idx')}|fused={float(r.get('fused', 0.0)):.4f}|leaf={r.get('embed_path')}")
        top50_candidates = _safe_join(top50, sep=" || ")

        rows.append({
            "pseudoprofile_id": pseudoprofile_id,
            "complaint_text": complaint_text,
            "decomp_n_variables": decomp_n,
            "decomp_notes": decomp_notes,
            "variable_id": str(var.get("id", "")),
            "variable_label": str(var.get("label", "")),
            "variable_criterion": str(var.get("criterion", "")),
            "variable_evidence": str(var.get("evidence", "")),
            "variable_polarity": str(var.get("polarity", "")),
            "variable_timeframe": str(var.get("timeframe", "")),
            "variable_severity_0_1": str(var.get("severity_0_1", "")),
            "variable_confidence_0_1": str(var.get("confidence_0_1", "")),
            "query_text_used": qtxt,
            "mapping_status": mapping_status,
            "chosen_method": chosen_method,
            "chosen_idx": str(chosen_idx),
            "chosen_leaf_embed_path": chosen_leaf_embed_path,
            "chosen_leaf_full_path": chosen_leaf_full_path,
            "chosen_confidence": str(chosen_confidence) if chosen_method == "llm" else "",
            "chosen_rationale": chosen_rationale,
            "top_fused_idx": str(top_fused_idx),
            "top_fused_leaf_embed_path": str(top_fused_leaf_embed_path),
            "top_fused_leaf_full_path": str(top_fused_leaf_full_path),
            "top_fused_score": str(top_fused_score),
            "top_fused_emb": str(top_fused_emb),
            "top_fused_bm25": str(top_fused_bm25),
            "top_fused_tok": str(top_fused_tok),
            "top_fused_fuz": str(top_fused_fuz),
            "debug_pool_size": str(debug.get("pool_size", "")),
            "debug_emb_rank_n": str(debug.get("emb_rank_n", "")),
            "debug_bm25_rank_n": str(debug.get("bm25_rank_n", "")),
            "top50_candidates": top50_candidates,
            "complaint_unique_mapped_leaf_indices": uniq_idxs_str,
            "complaint_unique_mapped_leaf_embed_paths": uniq_paths_str,
            "run_json_path": run_json_path,
            "error": "",
        })

    # If mappings empty, emit one error-like row
    if not rows:
        rows = mappings_to_csv_rows(
            searcher=searcher,
            pseudoprofile_id=pseudoprofile_id,
            complaint_text=complaint_text,
            decomp=decomp if isinstance(decomp, dict) else {"meta": {}, "variables": []},
            mappings=[],
            run_json_path=run_json_path,
            error="Cached run JSON had no mappings.",
        )
    return rows

def run_batch(
    free_text_path: str,
    output_csv_path: str,
    max_workers: int,
    enable_llm_reranker: bool,
    limit: int = 0,
) -> None:
    def _append_rows_to_csv(rows: List[Dict[str, Any]]) -> None:
        with _csv_lock:
            need_header = (not os.path.isfile(output_csv_path)) or (os.path.getsize(output_csv_path) == 0)
            with open(output_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
                if need_header:
                    writer.writeheader()
                for row in rows:
                    for k in CSV_FIELDNAMES:
                        if k not in row:
                            row[k] = ""
                    writer.writerow(row)

    searcher, _cache_info = build_searcher_from_cache()

    # --- TEMPORARY TEST LIMIT (will be changed later) ---
    MAX_PSEUDOPROFILES = 50  # NOTE: testing-only cap; adjust/remove later

    items = parse_free_text_complaints(free_text_path)

    if len(items) > MAX_PSEUDOPROFILES:
        log(
            f"[batch] Limiting pseudoprofiles to {MAX_PSEUDOPROFILES} "
            f"(skipping {len(items) - MAX_PSEUDOPROFILES} for now; testing purpose)"
        )
        items = items[:MAX_PSEUDOPROFILES]

    if limit and limit > 0:
        items = items[:limit]

    log(f"[batch] Loaded {len(items)} pseudoprofiles from: {free_text_path}")
    log(f"[batch] Output CSV: {output_csv_path}")
    log(f"[batch] max_workers={max_workers} | enable_llm_reranker={enable_llm_reranker}")
    log("")

    # Resume: if CSV already has rows, skip those pseudoprofiles (except ERROR rows)
    already_done = load_completed_pids_from_csv(output_csv_path)
    if already_done:
        new_items = []
        for pid, txt in items:
            if pid in already_done:
                log_stage("SKIP_ALREADY_IN_CSV", pseudoprofile_id=pid, extra={"reason": "already written"})
            else:
                new_items.append((pid, txt))
        items = new_items

    total_rows_written = 0

    if max_workers <= 1 or len(items) <= 1:
        for i, (pid, txt) in enumerate(items, start=1):
            log(f"[batch] ({i}/{len(items)}) Processing {pid} ...")
            _, rows = process_one_pseudoprofile(searcher, pid, txt, enable_llm_reranker)
            _append_rows_to_csv(rows)
            total_rows_written += len(rows)
            log(f"[batch] Progress: {i}/{len(items)} done.")
    else:
        mw = min(max_workers, len(items))
        log(f"[batch] Processing in parallel with {mw} workers...")
        done = 0
        with ThreadPoolExecutor(max_workers=mw) as ex:
            futs = {
                ex.submit(process_one_pseudoprofile, searcher, pid, txt, enable_llm_reranker): pid
                for pid, txt in items
            }
            for fut in as_completed(futs):
                pid = futs[fut]
                try:
                    _, rows = fut.result()
                    _append_rows_to_csv(rows)
                    total_rows_written += len(rows)
                except Exception as e:
                    # Should not happen because worker catches, but guard anyway
                    err = f"{type(e).__name__}: {e}"
                    log(f"[batch-error] Unexpected failure for {pid}: {err}")
                done += 1
                log(f"[batch] Progress: {done}/{len(items)} done.")

    log("")
    log(f"[done] Appended {total_rows_written} rows -> {output_csv_path}")
    log(f"[done] Per-complaint JSON artifacts -> {RUNS_DIR}")
    log(f"[done] Deterministic cache JSON -> {CACHE_RUNS_DIR}")
    log("")


# --------------------------
# Main
# --------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-txt", type=str, default=DEFAULT_FREE_TEXT_PATH, help="Path to free_text_complaints.txt")
    parser.add_argument("--output-csv", type=str, default=DEFAULT_OUTPUT_CSV, help="Path to mapped_criterions.csv")
    parser.add_argument("--max-workers", type=int, default=MAX_PARALLEL_COMPLAINTS, help="Parallel complaints workers")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N pseudoprofiles (0 = all)")
    parser.add_argument("--no-llm-rerank", action="store_true", help="Disable per-criterion LLM best-leaf adjudication")
    args = parser.parse_args()

    enable_llm_reranker = ENABLE_LLM_RERANKER_DEFAULT and (not args.no_llm_rerank)

    run_batch(
        free_text_path=args.input_txt,
        output_csv_path=args.output_csv,
        max_workers=max(1, int(args.max_workers)),
        enable_llm_reranker=enable_llm_reranker,
        limit=int(args.limit),
    )


if __name__ == "__main__":
    main()

# TODO: ensure that only selected mapped leaf is registered per cell instead of in all the cells the full list string delisted by ';'
# TODO: optimize forced-selection mapping logic --> avoid single-picks ; allow for more probabilistic/broad-candidate version in architecture

# NOTE: current script takes only 50 pseudoprofiles for now...
# NOTE: changed top5 --> top50
