#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_map_observation_model_to_ontology.py

Goal:
    Take the already-constructed observation model per pseudoprofile (raw JSON) and map:
      - criteria_variables  -> CRITERION ontology LEAF nodes
      - predictor_variables -> PREDICTOR ontology LEAF nodes

    Mapping pipeline (NO decomposition step):
      (1) Build a query text per variable (from label + measurement + rationale, etc.)
      (2) Retrieve top-K leaf-node candidates via hybrid retrieval:
            - Dense cosine similarity over cached leaf embeddings (EMBEDTEXT)
            - Sparse lexical grounding via BM25 on cached LEXTEXT
            - Token overlap + fuzzy match on LEXTEXT
            - Fusion (HTSSF by default)
      (3) LLM adjudication (DEFAULT model: gpt-5-mini) chooses:
            - idx of best leaf node from top-K candidates, OR
            - idx = -1 (UNMAPPED) with an explicit reason

Outputs:
    For each pseudoprofile directory inside the LATEST run:
      - llm_observation_model_mapped.json  (single aggregated JSON for the full model mapping)
      - llm_observation_model_mapped.txt   (very concise mapping overview)

Requirements:
    pip install openai numpy
Optional:
    pip install python-dotenv
    pip install rapidfuzz

Notes:
    - Uses ThreadPoolExecutor (pseudoprofile-level + variable-level).
    - Many console logs for traceability.
    - Top-K candidates default: 200 for both criteria and predictors.
"""

from __future__ import annotations

import os
import re
import json
import time
import math
import heapq
import hashlib
import argparse
import traceback
import threading
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# =============================================================================
# CONFIG DEFAULTS (EDIT THESE IF NEEDED)
# =============================================================================
def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "Evaluation").exists() and (candidate / "README.md").exists():
            return candidate
    raise RuntimeError("Could not locate repository root from 02_map_observation_model_to_ontology.py")


REPO_ROOT = _find_repo_root()

# --- Observation model runs root (script will pick latest run dir by default) ---
DEFAULT_RUNS_ROOT = str(
    REPO_ROOT / "Evaluation/03_construction_initial_observation_model/constructed_PC_models/runs"
)

# --- Input filename inside each pseudoprofile directory ---
RAW_MODEL_FILENAME = "llm_observation_model_raw.json"

# --- Output filenames inside each pseudoprofile directory ---
OUT_MAPPED_JSON_FILENAME = "llm_observation_model_mapped.json"
OUT_MAPPED_TXT_FILENAME = "llm_observation_model_mapped.txt"

# --- Embeddings cache directories (leaf nodes already embedded) ---
DEFAULT_CRITERION_CACHE_DIR = (
    str(
        REPO_ROOT / "SystemComponents/Agentic_Framework/02_ConstructionInitialObservationModel/utils/helpers/embeddings/CRITERIONS"
    )
)
DEFAULT_PREDICTOR_CACHE_DIR = (
    str(
        REPO_ROOT / "SystemComponents/Agentic_Framework/02_ConstructionInitialObservationModel/utils/helpers/embeddings/PREDICTORS"
    )
)

CRITERION_CACHE_DIR = os.environ.get("CRITERION_CACHE_DIR", DEFAULT_CRITERION_CACHE_DIR)
PREDICTOR_CACHE_DIR = os.environ.get("PREDICTOR_CACHE_DIR", DEFAULT_PREDICTOR_CACHE_DIR)

# --- OpenAI config ---
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("OPENAI_TIMEOUT_S", "90.0"))

# Embedding models:
# If not set, we will use each cache META_JSON["model"] to ensure consistency.
CRITERION_EMBED_MODEL_ENV = os.environ.get("CRITERION_EMBED_MODEL", "").strip()
PREDICTOR_EMBED_MODEL_ENV = os.environ.get("PREDICTOR_EMBED_MODEL", "").strip()

# LLM adjudicator (DEFAULT requested)
LLM_MAP_MODEL = os.environ.get("OBS_MAP_MODEL", "gpt-5-nano") # set on 'gpt-5-mini' or 'gpt-5' during deployment
LLM_MAP_TEMPERATURE = float(os.environ.get("OBS_MAP_T", "0.2"))

# --- Retrieval & fusion settings ---
TOP_K_RESULTS = int(os.environ.get("OBS_MAP_TOPK", "100"))  # requested default = 200
LLM_CANDIDATES_TOPN = int(os.environ.get("OBS_MAP_LLM_TOPN", "100"))  # requested default = 200

# Candidate pooling:
CANDIDATES_PER_METHOD = int(os.environ.get("OBS_MAP_CAND_PER_METHOD", "500"))
CANDIDATE_POOL = int(os.environ.get("OBS_MAP_CAND_POOL", "100000"))

# Retrieval weights (sum to 1)
WEIGHT_EMBED = float(os.environ.get("OBS_MAP_W_EMB", "0.80"))
WEIGHT_BM25 = float(os.environ.get("OBS_MAP_W_BM25", "0.12"))
WEIGHT_TOKEN_OVERLAP = float(os.environ.get("OBS_MAP_W_TOK", "0.05"))
WEIGHT_FUZZY = float(os.environ.get("OBS_MAP_W_FUZ", "0.03"))
if abs((WEIGHT_EMBED + WEIGHT_BM25 + WEIGHT_TOKEN_OVERLAP + WEIGHT_FUZZY) - 1.0) > 1e-9:
    raise ValueError("Retrieval weights must sum to 1.0")

# Fusion behavior:
# "htssf" recommended; also supports "rrf" or "scoresum"
FUSION_METHOD = os.environ.get("OBS_MAP_FUSION", "htssf").strip().lower()
RRF_K = int(os.environ.get("OBS_MAP_RRF_K", "60"))

# HTSSF params
HTSSF_ALPHA = float(os.environ.get("OBS_MAP_HTSSF_ALPHA", "0.90"))
# temps: (embed, bm25, token_overlap, fuzzy)
HTSSF_TEMPS = (
    float(os.environ.get("OBS_MAP_TEMB", "0.07")),
    float(os.environ.get("OBS_MAP_TBM25", "1.00")),
    float(os.environ.get("OBS_MAP_TTOK", "0.35")),
    float(os.environ.get("OBS_MAP_TFUZ", "0.35")),
)

# Norm computation chunking
NORM_CHUNK_ROWS = int(os.environ.get("OBS_MAP_NORM_CHUNK", "4096"))

# Parallelism
MAX_PARALLEL_PSEUDOPROFILES = int(os.environ.get("OBS_MAP_MAX_PP", "50"))
MAX_PARALLEL_VARIABLES = int(os.environ.get("OBS_MAP_MAX_VARS", "1000"))

# Retry/backoff
MAX_RETRIES = int(os.environ.get("OBS_MAP_MAX_RETRIES", "8"))
BACKOFF_BASE_SECONDS = float(os.environ.get("OBS_MAP_BACKOFF_BASE", "0.8"))
BACKOFF_MAX_SECONDS = float(os.environ.get("OBS_MAP_BACKOFF_MAX", "20.0"))

# =============================================================================
# LOGGING
# =============================================================================

def ts() -> str:
    return time.strftime("%H:%M:%S")

def log(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)

def log_stage(
    stage: str,
    *,
    pseudoprofile_id: Optional[str] = None,
    var_type: Optional[str] = None,
    var_id: Optional[str] = None,
    var_label: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    parts = [f"[stage={stage}]"]
    if pseudoprofile_id:
        parts.append(f"pid={pseudoprofile_id}")
    if var_type:
        parts.append(f"type={var_type}")
    if var_id:
        parts.append(f"var={var_id}")
    if var_label:
        lbl = var_label.strip().replace("\n", " ")
        if len(lbl) > 120:
            lbl = lbl[:117] + "..."
        parts.append(f"label='{lbl}'")
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

# =============================================================================
# CLEANING FILTER: exclude RDoC leaf nodes from candidate list (keep DSM/ICD only)
# =============================================================================

# Put this near other regex/constants (top-level, e.g. close to RUN_DIR_RE)
EXCLUDE_RDOC_IN_LEAF = os.environ.get("OBS_MAP_EXCLUDE_RDOC", "1").strip().lower() in ("1", "true", "yes", "y")

# Case-insensitive match for "RDoC" anywhere in a path string
RDOC_RE = re.compile(r"\brdoc\b", flags=re.IGNORECASE)

def _is_allowed_leaf_path(full_path: str, embed_path: str = "") -> bool:
    """
    Return True if this ontology leaf candidate should be considered.
    Requirement: if 'RDoC' appears anywhere in the embedded string/path, exclude.
    (We check both full_path and embed_path to be safe.)
    """
    if not EXCLUDE_RDOC_IN_LEAF:
        return True
    hay = f"{full_path} | {embed_path}"
    return RDOC_RE.search(hay) is None

# =============================================================================
# OPTIONAL dotenv loading
# =============================================================================

def load_env_if_possible() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
        return
    except Exception:
        pass

    # very light fallback .env parse
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

# =============================================================================
# TOKENIZATION + LIGHT STEMMING
# =============================================================================

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

# =============================================================================
# BM25 (SPARSE INVERTED INDEX)
# =============================================================================

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
        log("[index] Building BM25 sparse index...")
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

# =============================================================================
# EMBEDDINGS HELPERS + NORMS
# =============================================================================

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
    log(f"[norm] Computing norms for {os.path.basename(emb_path)} ...")
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

# =============================================================================
# FUSION HELPERS
# =============================================================================

def safe_minmax_norm(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    mn = float(np.min(x))
    mx = float(np.max(x))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)

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

# =============================================================================
# JSON HELPERS (LLM output parsing + repair)
# =============================================================================

def _extract_json_object(text: str) -> str:
    text = text.strip()
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in LLM output.")
    return m.group(0)

def _repair_json_text_best_effort(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.replace("“", "\"").replace("”", "\"").replace("’", "'").replace("‘", "'")
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start:end+1]
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
                resp = client.responses.create(
                    model=model,
                    input=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    response_format={"type": "json_object"},
                )
                text = resp.output_text
            except Exception:
                # fallback
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    response_format={"type": "json_object"},
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

# =============================================================================
# ONTOLOGY SEARCHER
# =============================================================================

def stable_hash_of_paths(paths: List[str]) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()

def _require_file(path: str, hint: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing required cache file: {path}\nHint: {hint}")

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

class OntologySearcher:
    def __init__(
        self,
        *,
        name: str,
        embed_model: str,
        full_paths: List[str],
        embed_paths: List[str],
        lex_texts: List[str],
        emb: np.ndarray,
        norms: np.ndarray,
        bm25: BM25Sparse,
    ):
        self.name = name
        self.embed_model = embed_model
        self.full_paths = full_paths
        self.embed_paths = embed_paths
        self.lex_texts = lex_texts
        self.emb = emb
        self.norms = norms
        self.bm25 = bm25

        if self.emb.ndim != 2 or self.emb.shape[0] != len(self.embed_paths):
            raise RuntimeError(f"[{self.name}] Embedding shape mismatch: emb={self.emb.shape}, paths={len(self.embed_paths)}")
        if self.norms.shape[0] != len(self.embed_paths):
            raise RuntimeError(f"[{self.name}] Norms length mismatch: norms={self.norms.shape}, paths={len(self.embed_paths)}")

    def query_with_vec(self, query: str, q: np.ndarray) -> Tuple[List[SearchResult], Dict[str, Any]]:
        query = query.strip()
        if not query:
            return [], {}

        q = q.astype(np.float32, copy=False)
        q = q / (float(np.linalg.norm(q)) + 1e-12)

        # Dense cosine
        dots = self.emb @ q
        emb_scores = dots / (self.norms + 1e-12)
        emb_rank = topk_from_scores(emb_scores, k=max(CANDIDATES_PER_METHOD, CANDIDATE_POOL))

        # BM25
        bm25_rank, bm25_map = self.bm25.score_topk(query, k=max(CANDIDATES_PER_METHOD, CANDIDATE_POOL))

        # Pool union
        pool_set = set(emb_rank.tolist()) | set(bm25_rank.tolist())
        pool_ids = np.fromiter(pool_set, dtype=np.int64)
        if pool_ids.size > CANDIDATE_POOL:
            local_scores = emb_scores[pool_ids]
            keep_local = topk_from_scores(local_scores, k=CANDIDATE_POOL)
            pool_ids = pool_ids[keep_local]

        # Token overlap + fuzzy
        tok_map = token_overlap_score_subset(query, self.lex_texts, pool_ids)
        fuz_map = fuzzy_score_subset(query, self.lex_texts, pool_ids)

        tok_rank = np.array(
            [i for i, _ in heapq.nlargest(CANDIDATES_PER_METHOD, tok_map.items(), key=lambda kv: kv[1])],
            dtype=np.int64,
        )
        fuz_rank = np.array(
            [i for i, _ in heapq.nlargest(CANDIDATES_PER_METHOD, fuz_map.items(), key=lambda kv: kv[1])],
            dtype=np.int64,
        )

        # Fusion
        if FUSION_METHOD == "rrf":
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
        elif FUSION_METHOD == "scoresum":
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

        fused_sorted = sorted(fused_map.items(), key=lambda kv: kv[1], reverse=True)

        # --- NEW: filter out RDoC leaves before selecting TOP_K_RESULTS ---
        fused_items: List[Tuple[int, float]] = []
        for idx, fscore in fused_sorted:
            i = int(idx)
            full_p = self.full_paths[i]
            emb_p = self.embed_paths[i]
            if _is_allowed_leaf_path(full_p, emb_p):
                fused_items.append((i, float(fscore)))
                if len(fused_items) >= TOP_K_RESULTS:
                    break

        results: List[SearchResult] = []
        for i, fscore in fused_items:
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
            "fusion_method": FUSION_METHOD,
            "excluded_rdoc": bool(EXCLUDE_RDOC_IN_LEAF),
        }
        return results, debug

def build_searcher_from_cache(cache_dir: str, prefix: str, name: str, embed_model_override: str = "") -> OntologySearcher:
    load_env_if_possible()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not found. Put it in .env as OPENAI_API_KEY=... or export it in your shell.")

    paths_embed = os.path.join(cache_dir, f"{prefix}_leaf_paths_EMBEDTEXT.json")
    paths_full = os.path.join(cache_dir, f"{prefix}_leaf_paths_FULL.json")
    paths_lex = os.path.join(cache_dir, f"{prefix}_leaf_paths_LEXTEXT.json")

    emb_npy = os.path.join(cache_dir, f"{prefix}_leaf_embeddings.npy")
    meta_json = os.path.join(cache_dir, f"{prefix}_leaf_embeddings_meta.json")

    # For predictors, norms exists; for criteria it may not.
    norms_npy = os.path.join(cache_dir, f"{prefix}_leaf_embedding_norms.npy")

    _require_file(paths_embed, f"Re-run your leaf embedding builder for {name}.")
    _require_file(paths_full,  f"Re-run your leaf embedding builder for {name}.")
    _require_file(paths_lex,   f"Re-run your leaf embedding builder for {name}.")
    _require_file(emb_npy,     f"Re-run your leaf embedding builder for {name}.")
    _require_file(meta_json,   f"Re-run your leaf embedding builder for {name} (meta).")

    embed_paths = json.load(open(paths_embed, "r", encoding="utf-8"))
    full_paths = json.load(open(paths_full, "r", encoding="utf-8"))
    lex_texts = json.load(open(paths_lex, "r", encoding="utf-8"))

    if not (isinstance(embed_paths, list) and isinstance(full_paths, list) and isinstance(lex_texts, list)):
        raise RuntimeError(f"[{name}] Path caches are not lists (corrupted JSON).")

    if not (len(embed_paths) == len(full_paths) == len(lex_texts)):
        raise RuntimeError(
            f"[{name}] Cache length mismatch: embed={len(embed_paths)} full={len(full_paths)} lex={len(lex_texts)}"
        )

    meta = json.load(open(meta_json, "r", encoding="utf-8"))
    paths_hash = stable_hash_of_paths(embed_paths)
    if meta.get("paths_hash") != paths_hash:
        raise RuntimeError(f"[{name}] Cache mismatch: paths_hash differs from META_JSON (paths changed or cache stale).")
    if meta.get("status") != "complete":
        raise RuntimeError(f"[{name}] Embeddings cache is not complete (status={meta.get('status')}). Rebuild embeddings.")

    embed_model = embed_model_override.strip() or str(meta.get("model") or "").strip()
    if not embed_model:
        raise RuntimeError(f"[{name}] META_JSON missing 'model' field and no override provided.")

    emb = np.load(emb_npy, mmap_mode="r")
    if emb.shape[0] != len(embed_paths):
        raise RuntimeError(f"[{name}] Embedding rows mismatch: emb_rows={emb.shape[0]} paths={len(embed_paths)}")

    if os.path.isfile(norms_npy):
        norms = np.load(norms_npy).astype(np.float32)
        if norms.shape[0] != len(embed_paths):
            log(f"[{name}] Norm cache wrong length; recomputing.")
            norms = compute_and_cache_norms(emb_npy, norms_npy)
        else:
            log(f"[{name}] Loaded norms: {norms.shape} ({human_bytes(os.path.getsize(norms_npy))})")
    else:
        log(f"[{name}] Norms file missing; computing: {norms_npy}")
        norms = compute_and_cache_norms(emb_npy, norms_npy)

    bm25 = BM25Sparse.build(lex_texts)

    log(f"[{name}] cache_dir={cache_dir}")
    log(f"[{name}] n_leaf_nodes={len(embed_paths)} | embed_model={embed_model}")
    log("")

    return OntologySearcher(
        name=name,
        embed_model=embed_model,
        full_paths=full_paths,
        embed_paths=embed_paths,
        lex_texts=lex_texts,
        emb=emb,
        norms=norms,
        bm25=bm25,
    )

# =============================================================================
# LLM ADJUDICATION
# =============================================================================

def llm_pick_best_leaf(
    *,
    ontology_name: str,
    pseudoprofile_context: Dict[str, Any],
    variable_payload: Dict[str, Any],
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    candidates: list of {idx, fused_score, embed_path, full_path}
    Must include UNMAPPED candidate idx=-1.
    Output STRICT JSON:
      {
        "idx": <int>,
        "confidence": <float 0-1>,
        "rationale": <string>,
        "unmapped_reason": <string>
      }
    """
    client = make_openai_client()

    sys_msg = (
        "You are an health-engineering ontology mapping expert.\n"
        f"Task: choose the SINGLE best-matching {ontology_name} ontology LEAF node for the given variable.\n"
        "Rules:\n"
        " - Choose exactly ONE candidate idx.\n"
        " - Prefer the most specific operational construct that matches the variable as intended.\n"
        " - Use variable label + measurement item/signal + mechanism rationale as primary evidence.\n"
        " - Use pseudoprofile context as supporting context (do NOT overfit to it).\n"
        " - If there is NO clear counterpart among candidates, choose idx=-1 (UNMAPPED) and explain why.\n"
        "Output STRICT JSON only with keys: idx (int), confidence (0-1 float), rationale (string), unmapped_reason (string).\n"
        "If idx != -1, set unmapped_reason to an empty string.\n"
    )

    # Context lines (keep compact but informative)
    pid = str(pseudoprofile_context.get("pseudoprofile_id") or "")
    model_summary = str(pseudoprofile_context.get("model_summary") or "")
    var_notes = str(pseudoprofile_context.get("variable_selection_notes") or "")

    ctx_lines = []
    if pid:
        ctx_lines.append(f"pseudoprofile_id: {pid}")
    if model_summary:
        ctx_lines.append(f"model_summary: {model_summary}")
    if var_notes:
        ctx_lines.append(f"variable_selection_notes: {var_notes}")

    var_lines = []
    for k in ["var_id", "id", "label", "polarity", "bio_psycho_social_domain", "ontology_path", "criterion_path", "mechanism_rationale", "intervention_examples_safe"]:
        if k in variable_payload and variable_payload.get(k) not in (None, "", []):
            v = variable_payload.get(k)
            if isinstance(v, (dict, list)):
                v = json.dumps(v, ensure_ascii=False)
            var_lines.append(f"{k}: {v}")

    # Measurement details (if present)
    measurement = variable_payload.get("measurement", {})
    if isinstance(measurement, dict):
        for mk in ["mode", "channel", "assessment_type", "item_or_signal", "response_scale_or_unit", "sampling_per_day", "recall_window", "notes"]:
            if mk in measurement and measurement.get(mk) not in (None, "", []):
                mv = measurement.get(mk)
                var_lines.append(f"measurement.{mk}: {mv}")

    # Candidate list (top-N already preselected)
    cand_lines = []
    for c in candidates:
        cand_lines.append(
            f"- idx={c['idx']} fused={float(c.get('fused_score', 0.0)):.4f} | full={c.get('full_path','')} | embed={c.get('embed_path','')}"
        )

    user_msg = (
        "Pseudoprofile context:\n"
        + ("\n".join(ctx_lines) if ctx_lines else "(none)")
        + "\n\nVariable to map:\n"
        + ("\n".join(var_lines) if var_lines else json.dumps(variable_payload, ensure_ascii=False, indent=2))
        + "\n\nCandidate leaf nodes (choose ONE idx; idx=-1 means UNMAPPED):\n"
        + "\n".join(cand_lines)
        + "\n\nReturn JSON only."
    )

    text = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            try:
                resp = client.responses.create(
                    model=LLM_MAP_MODEL,
                    input=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    # NOTE: keep low temp for deterministic mapping
                    #temperature=LLM_MAP_TEMPERATURE,
                    response_format={"type": "json_object"},
                )
                text = resp.output_text
            except Exception:
                resp = client.chat.completions.create(
                    model=LLM_MAP_MODEL,
                    messages=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    #temperature=LLM_MAP_TEMPERATURE,
                    response_format={"type": "json_object"},
                )
                text = resp.choices[0].message.content
            break
        except Exception as e:
            msg = str(e)
            sleep_s = min(BACKOFF_MAX_SECONDS, BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))
            sleep_s = sleep_s * (0.9 + 0.2 * ((time.time() * 997) % 1.0))
            log(f"[llm-map] attempt {attempt}/{MAX_RETRIES} failed: {type(e).__name__}: {msg}")
            if attempt >= MAX_RETRIES:
                raise
            log(f"[llm-map] backing off {sleep_s:.2f}s...")
            time.sleep(sleep_s)

    if not text:
        raise RuntimeError("LLM mapping returned empty output.")

    # Parse JSON (with repair fallback)
    try:
        obj_text = _extract_json_object(text)
        obj = _safe_json_loads(obj_text)
    except Exception as e:
        log(f"[llm-map] JSON parse failed ({type(e).__name__}: {e}); attempting JSON repair...")
        repaired = _repair_json_with_llm(text, model=LLM_MAP_MODEL)
        obj_text = _extract_json_object(repaired)
        obj = _safe_json_loads(obj_text)

    if not isinstance(obj, dict) or "idx" not in obj:
        raise RuntimeError(f"LLM mapping JSON missing idx. Raw:\n{obj_text}")

    # normalize
    try:
        obj["idx"] = int(obj.get("idx"))
    except Exception:
        obj["idx"] = -1

    try:
        obj["confidence"] = float(obj.get("confidence", 0.0))
    except Exception:
        obj["confidence"] = 0.0
    obj["confidence"] = max(0.0, min(1.0, obj["confidence"]))

    obj["rationale"] = str(obj.get("rationale", "") or "")
    obj["unmapped_reason"] = str(obj.get("unmapped_reason", "") or "")

    if obj["idx"] != -1:
        obj["unmapped_reason"] = ""

    return obj

# =============================================================================
# QUERY BUILDERS (for retrieval)
# =============================================================================

def _safe_get(d: Dict[str, Any], path: str, default: str = "") -> str:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    if cur is None:
        return default
    return str(cur)

def build_query_text_for_criterion(var: Dict[str, Any]) -> str:
    """
    Criterion variables are complaint-derived outcomes.
    Use label + measurement item/signal + any notes.
    """
    label = str(var.get("label") or "").strip()
    item = _safe_get(var, "measurement.item_or_signal", "")
    notes = _safe_get(var, "measurement.notes", "")
    polarity = str(var.get("polarity") or "").strip()

    parts = []
    if label:
        parts.append(label)
    if item:
        parts.append(f"Operational item/signal: {item}")
    if polarity:
        parts.append(f"Polarity: {polarity}")
    if notes:
        parts.append(f"Notes: {notes}")

    q = " | ".join(parts).strip()
    return q if q else label

def build_query_text_for_predictor(var: Dict[str, Any]) -> str:
    """
    Predictor variables are modifiable bio-psycho-social levers.
    Use label + mechanism rationale + intervention examples + measurement item/signal.
    """
    label = str(var.get("label") or "").strip()
    mech = str(var.get("mechanism_rationale") or "").strip()
    inter = str(var.get("intervention_examples_safe") or "").strip()
    domain = str(var.get("bio_psycho_social_domain") or "").strip()
    item = _safe_get(var, "measurement.item_or_signal", "")
    mode = _safe_get(var, "measurement.mode", "")
    prior_path = str(var.get("ontology_path") or "").strip()

    parts = []
    if label:
        parts.append(label)
    if domain:
        parts.append(f"Domain: {domain}")
    if mech:
        parts.append(f"Mechanism: {mech}")
    if inter:
        parts.append(f"Intervention examples: {inter}")
    if mode or item:
        parts.append(f"Measurement: {mode} | {item}".strip())
    if prior_path:
        parts.append(f"Prior ontology hint: {prior_path}")

    q = " | ".join([p for p in parts if p]).strip()
    return q if q else label

# =============================================================================
# IO HELPERS
# =============================================================================

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def atomic_write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def atomic_write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)

# =============================================================================
# RUN DISCOVERY
# =============================================================================

RUN_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")

def find_latest_run_dir(runs_root: str) -> str:
    if not os.path.isdir(runs_root):
        raise FileNotFoundError(f"Runs root not found: {runs_root}")

    dirs = [d for d in os.listdir(runs_root) if os.path.isdir(os.path.join(runs_root, d))]
    good = [d for d in dirs if RUN_DIR_RE.match(d)]
    if good:
        good.sort()
        latest = good[-1]
        return os.path.join(runs_root, latest)

    # fallback: newest mtime
    dirs_full = [os.path.join(runs_root, d) for d in dirs]
    latest = max(dirs_full, key=lambda p: os.path.getmtime(p))
    return latest

def discover_raw_models(run_dir: str) -> List[str]:
    profiles_dir = os.path.join(run_dir, "profiles")
    if not os.path.isdir(profiles_dir):
        raise FileNotFoundError(f"profiles/ dir not found in run: {profiles_dir}")

    raw_paths: List[str] = []
    for name in sorted(os.listdir(profiles_dir)):
        pdir = os.path.join(profiles_dir, name)
        if not os.path.isdir(pdir):
            continue
        raw = os.path.join(pdir, RAW_MODEL_FILENAME)
        if os.path.isfile(raw):
            raw_paths.append(raw)

    return raw_paths

# =============================================================================
# CORE MAPPING PER PSEUDOPROFILE
# =============================================================================

def map_one_pseudoprofile(
    *,
    raw_model_path: str,
    criterion_searcher: OntologySearcher,
    predictor_searcher: OntologySearcher,
    overwrite: bool,
) -> Tuple[str, bool]:
    """
    Returns (pseudoprofile_id, success).
    """
    pseudoprofile_dir = os.path.dirname(raw_model_path)
    out_json_path = os.path.join(pseudoprofile_dir, OUT_MAPPED_JSON_FILENAME)
    out_txt_path = os.path.join(pseudoprofile_dir, OUT_MAPPED_TXT_FILENAME)

    raw = load_json(raw_model_path)
    pid = str(raw.get("pseudoprofile_id") or os.path.basename(pseudoprofile_dir)).strip() or os.path.basename(pseudoprofile_dir)

    if (not overwrite) and os.path.isfile(out_json_path) and os.path.getsize(out_json_path) > 0:
        log_stage("SKIP_ALREADY_MAPPED", pseudoprofile_id=pid, extra={"file": OUT_MAPPED_JSON_FILENAME})
        return pid, True

    log_stage("START_PSEUDOPROFILE", pseudoprofile_id=pid, extra={"raw": os.path.basename(raw_model_path)})

    ctx = {
        "pseudoprofile_id": pid,
        "model_summary": raw.get("model_summary", ""),
        "variable_selection_notes": raw.get("variable_selection_notes", ""),
    }

    criteria_vars = raw.get("criteria_variables", [])
    predictor_vars = raw.get("predictor_variables", [])
    if not isinstance(criteria_vars, list):
        criteria_vars = []
    if not isinstance(predictor_vars, list):
        predictor_vars = []

    log_stage(
        "RAW_LOADED",
        pseudoprofile_id=pid,
        extra={"n_criteria": len(criteria_vars), "n_predictors": len(predictor_vars)}
    )

    # ---- Build query texts ----
    crit_queries = [build_query_text_for_criterion(v) for v in criteria_vars]
    pred_queries = [build_query_text_for_predictor(v) for v in predictor_vars]

    # ---- Embed queries (two batch calls, one per ontology model) ----
    crit_Q = np.zeros((0, 1), dtype=np.float32)
    pred_Q = np.zeros((0, 1), dtype=np.float32)

    if crit_queries:
        log_stage("EMBED_CRITERIA_BATCH", pseudoprofile_id=pid, extra={"n": len(crit_queries), "model": criterion_searcher.embed_model})
        crit_vecs = embed_texts_with_retry(crit_queries, criterion_searcher.embed_model)
        crit_Q = np.array(crit_vecs, dtype=np.float32)
        crit_Q = crit_Q / (np.linalg.norm(crit_Q, axis=1, keepdims=True).astype(np.float32) + 1e-12)

    if pred_queries:
        log_stage("EMBED_PREDICTORS_BATCH", pseudoprofile_id=pid, extra={"n": len(pred_queries), "model": predictor_searcher.embed_model})
        pred_vecs = embed_texts_with_retry(pred_queries, predictor_searcher.embed_model)
        pred_Q = np.array(pred_vecs, dtype=np.float32)
        pred_Q = pred_Q / (np.linalg.norm(pred_Q, axis=1, keepdims=True).astype(np.float32) + 1e-12)

    # ---- Per-variable mapping (retrieval + LLM pick) ----
    def map_variable(
        var_type: str,
        idx_in_list: int,
        var: Dict[str, Any],
        query_text: str,
        qvec: np.ndarray,
        searcher: OntologySearcher,
    ) -> Dict[str, Any]:
        var_id = str(var.get("var_id") or var.get("id") or f"{var_type}_{idx_in_list+1}")
        var_label = str(var.get("label") or "")

        log_stage("RETRIEVAL_START", pseudoprofile_id=pid, var_type=var_type, var_id=var_id, var_label=var_label)

        results, debug = searcher.query_with_vec(query_text, qvec)

        if not results:
            log_stage("RETRIEVAL_NO_RESULTS", pseudoprofile_id=pid, var_type=var_type, var_id=var_id)
            # still do LLM with only UNMAPPED, but not needed; directly unmapped
            return {
                "var_id": var_id,
                "var_type": var_type,
                "label": var_label,
                "query_text_used": query_text,
                "mapping": {
                    "status": "UNMAPPED",
                    "idx": -1,
                    "confidence": 0.0,
                    "rationale": "No retrieval results.",
                    "unmapped_reason": "No retrieval results from ontology search.",
                    "leaf_embed_path": "UNMAPPED",
                    "leaf_full_path": "UNMAPPED",
                },
                "debug": debug,
                "top_candidates": [],
                "source_variable": var,
            }

        # Candidates for LLM
        topn = min(LLM_CANDIDATES_TOPN, len(results))
        cands = [{"idx": -1, "fused_score": 0.0, "embed_path": "UNMAPPED", "full_path": "UNMAPPED"}]
        cands += [{"idx": r.idx, "fused_score": r.fused, "embed_path": r.embed_path, "full_path": r.full_path} for r in results[:topn]]

        log_stage(
            "LLM_PICK_START",
            pseudoprofile_id=pid,
            var_type=var_type,
            var_id=var_id,
            var_label=var_label,
            extra={"n_candidates": len(cands), "llm": LLM_MAP_MODEL}
        )

        try:
            pick = llm_pick_best_leaf(
                ontology_name=searcher.name,
                pseudoprofile_context=ctx,
                variable_payload=var,
                candidates=cands,
            )
        except Exception as e:
            # fallback: top fused
            best = results[0]
            err = f"{type(e).__name__}: {e}"
            log_stage(
                "LLM_PICK_ERROR_FALLBACK_TOP1",
                pseudoprofile_id=pid,
                var_type=var_type,
                var_id=var_id,
                extra={"error": err, "fallback_idx": best.idx}
            )
            pick = {
                "idx": int(best.idx),
                "confidence": 0.0,
                "rationale": f"LLM mapping failed ({err}); fell back to top fused retrieval result.",
                "unmapped_reason": "",
            }

        chosen_idx = int(pick.get("idx", -1))
        chosen_conf = float(pick.get("confidence", 0.0))
        chosen_rat = str(pick.get("rationale", "") or "")
        unmapped_reason = str(pick.get("unmapped_reason", "") or "")

        # Safety: if chosen_idx not in candidates, fallback to top fused
        cand_ids = {c["idx"] for c in cands}
        if chosen_idx not in cand_ids:
            best = results[0]
            log_stage(
                "LLM_PICK_IDX_NOT_IN_CANDS_FALLBACK_TOP1",
                pseudoprofile_id=pid,
                var_type=var_type,
                var_id=var_id,
                extra={"returned_idx": chosen_idx, "fallback_idx": best.idx}
            )
            chosen_idx = int(best.idx)
            chosen_conf = 0.0
            chosen_rat = f"LLM returned idx not in candidate set; fell back to top fused. Raw pick={pick}"
            unmapped_reason = ""

        if chosen_idx == -1:
            status = "UNMAPPED"
            leaf_embed_path = "UNMAPPED"
            leaf_full_path = "UNMAPPED"
        else:
            status = "MAPPED"
            leaf_embed_path = searcher.embed_paths[chosen_idx] if 0 <= chosen_idx < len(searcher.embed_paths) else ""
            leaf_full_path = searcher.full_paths[chosen_idx] if 0 <= chosen_idx < len(searcher.full_paths) else ""
            unmapped_reason = ""

        log_stage(
            "LLM_PICK_DONE",
            pseudoprofile_id=pid,
            var_type=var_type,
            var_id=var_id,
            extra={"status": status, "idx": chosen_idx, "conf": f"{chosen_conf:.2f}"}
        )

        # Save top candidates (full TOP_K_RESULTS already computed by retrieval)
        top_candidates = [
            {
                "idx": r.idx,
                "fused": r.fused,
                "emb": r.emb,
                "bm25": r.bm25,
                "tok": r.tok,
                "fuz": r.fuz,
                "embed_path": r.embed_path,
                "full_path": r.full_path,
            }
            for r in results
        ]

        return {
            "var_id": var_id,
            "var_type": var_type,
            "label": var_label,
            "query_text_used": query_text,
            "mapping": {
                "status": status,
                "idx": chosen_idx,
                "confidence": chosen_conf,
                "rationale": chosen_rat,
                "unmapped_reason": unmapped_reason,
                "leaf_embed_path": leaf_embed_path,
                "leaf_full_path": leaf_full_path,
            },
            "debug": debug,
            "top_candidates": top_candidates,  # top-K fused list (K=TOP_K_RESULTS)
            "source_variable": var,
        }

    # Map all vars (variable-level threadpool)
    criterion_mappings: List[Dict[str, Any]] = []
    predictor_mappings: List[Dict[str, Any]] = []

    # --- criteria ---
    if criteria_vars:
        log_stage("MAP_CRITERIA_START", pseudoprofile_id=pid, extra={"n": len(criteria_vars)})
        if len(criteria_vars) <= 1 or MAX_PARALLEL_VARIABLES <= 1:
            for i, v in enumerate(criteria_vars):
                criterion_mappings.append(map_variable("CRITERION", i, v, crit_queries[i], crit_Q[i], criterion_searcher))
        else:
            with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_VARIABLES, len(criteria_vars))) as ex:
                futs = {
                    ex.submit(map_variable, "CRITERION", i, v, crit_queries[i], crit_Q[i], criterion_searcher): i
                    for i, v in enumerate(criteria_vars)
                }
                for fut in as_completed(futs):
                    criterion_mappings.append(fut.result())
            # keep stable order by original list index (var_id sort can be misleading if IDs differ)
            criterion_mappings.sort(key=lambda x: (criteria_vars.index(x["source_variable"]) if x.get("source_variable") in criteria_vars else 10**9))
        log_stage("MAP_CRITERIA_DONE", pseudoprofile_id=pid, extra={"n": len(criterion_mappings)})

    # --- predictors ---
    if predictor_vars:
        log_stage("MAP_PREDICTORS_START", pseudoprofile_id=pid, extra={"n": len(predictor_vars)})
        if len(predictor_vars) <= 1 or MAX_PARALLEL_VARIABLES <= 1:
            for i, v in enumerate(predictor_vars):
                predictor_mappings.append(map_variable("PREDICTOR", i, v, pred_queries[i], pred_Q[i], predictor_searcher))
        else:
            with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_VARIABLES, len(predictor_vars))) as ex:
                futs = {
                    ex.submit(map_variable, "PREDICTOR", i, v, pred_queries[i], pred_Q[i], predictor_searcher): i
                    for i, v in enumerate(predictor_vars)
                }
                for fut in as_completed(futs):
                    predictor_mappings.append(fut.result())
            predictor_mappings.sort(key=lambda x: (predictor_vars.index(x["source_variable"]) if x.get("source_variable") in predictor_vars else 10**9))
        log_stage("MAP_PREDICTORS_DONE", pseudoprofile_id=pid, extra={"n": len(predictor_mappings)})

    # ---- Build output JSON: raw model + mapping augmentation ----
    def _augment_variables(original_vars: List[Dict[str, Any]], maps: List[Dict[str, Any]], kind: str) -> List[Dict[str, Any]]:
        # map by var_id primarily; fallback to index
        by_id: Dict[str, Dict[str, Any]] = {}
        for m in maps:
            by_id[str(m.get("var_id", ""))] = m

        out = []
        for i, v in enumerate(original_vars):
            vid = str(v.get("var_id") or v.get("id") or f"{kind}_{i+1}")
            m = by_id.get(vid)
            v2 = dict(v)
            if m:
                v2["mapping_status"] = m["mapping"]["status"]
                v2["mapped_leaf_idx"] = m["mapping"]["idx"]
                v2["mapped_leaf_full_path"] = m["mapping"]["leaf_full_path"]
                v2["mapped_leaf_embed_path"] = m["mapping"]["leaf_embed_path"]
                v2["mapped_confidence"] = m["mapping"]["confidence"]
                v2["mapped_rationale"] = m["mapping"]["rationale"]
                v2["unmapped_reason"] = m["mapping"]["unmapped_reason"]
                v2["mapping_query_text_used"] = m["query_text_used"]
            else:
                v2["mapping_status"] = "UNMAPPED"
                v2["mapped_leaf_idx"] = -1
                v2["mapped_leaf_full_path"] = "UNMAPPED"
                v2["mapped_leaf_embed_path"] = "UNMAPPED"
                v2["mapped_confidence"] = 0.0
                v2["mapped_rationale"] = "No mapping record found (unexpected)."
                v2["unmapped_reason"] = "No mapping record found (unexpected)."
                v2["mapping_query_text_used"] = ""
            out.append(v2)
        return out

    mapped_criteria_vars = _augment_variables(criteria_vars, criterion_mappings, "CRITERION")
    mapped_predictor_vars = _augment_variables(predictor_vars, predictor_mappings, "PREDICTOR")

    # Stats
    n_crit_mapped = sum(1 for v in mapped_criteria_vars if v.get("mapping_status") == "MAPPED")
    n_crit_unmapped = sum(1 for v in mapped_criteria_vars if v.get("mapping_status") != "MAPPED")
    n_pred_mapped = sum(1 for v in mapped_predictor_vars if v.get("mapping_status") == "MAPPED")
    n_pred_unmapped = sum(1 for v in mapped_predictor_vars if v.get("mapping_status") != "MAPPED")

    out_obj = {
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_raw_model_path": raw_model_path,
        "pseudoprofile_id": pid,
        "config": {
            "criterion_cache_dir": CRITERION_CACHE_DIR,
            "predictor_cache_dir": PREDICTOR_CACHE_DIR,
            "criterion_embed_model": criterion_searcher.embed_model,
            "predictor_embed_model": predictor_searcher.embed_model,
            "llm_map_model": LLM_MAP_MODEL,
            "llm_temperature": LLM_MAP_TEMPERATURE,
            "top_k_results": TOP_K_RESULTS,
            "llm_candidates_topn": LLM_CANDIDATES_TOPN,
            "fusion_method": FUSION_METHOD,
            "weights": {
                "embed": WEIGHT_EMBED,
                "bm25": WEIGHT_BM25,
                "token_overlap": WEIGHT_TOKEN_OVERLAP,
                "fuzzy": WEIGHT_FUZZY,
            },
            "htssf": {"alpha": HTSSF_ALPHA, "temps": HTSSF_TEMPS, "rrf_k": RRF_K},
            "candidate_pool": CANDIDATE_POOL,
            "candidates_per_method": CANDIDATES_PER_METHOD,
            "parallelism": {
                "max_parallel_pseudoprofiles": MAX_PARALLEL_PSEUDOPROFILES,
                "max_parallel_variables": MAX_PARALLEL_VARIABLES,
            },
        },
        # preserve high-level context fields
        "model_summary": raw.get("model_summary", ""),
        "variable_selection_notes": raw.get("variable_selection_notes", ""),
        "design_recommendations": raw.get("design_recommendations", {}),
        "safety_notes": raw.get("safety_notes", ""),
        # mapped variables
        "criteria_variables": mapped_criteria_vars,
        "predictor_variables": mapped_predictor_vars,
        # preserve edges and other components as-is
        "predictor_criterion_relevance": raw.get("predictor_criterion_relevance", []),
        "edges": raw.get("edges", []),
        "diagnostics": raw.get("diagnostics", {}),
        # include raw mapping records (with candidate lists) for debugging/auditing
        "mapping_records": {
            "criteria": criterion_mappings,
            "predictors": predictor_mappings,
        },
        "mapping_stats": {
            "criteria": {"n_total": len(mapped_criteria_vars), "n_mapped": n_crit_mapped, "n_unmapped": n_crit_unmapped},
            "predictors": {"n_total": len(mapped_predictor_vars), "n_mapped": n_pred_mapped, "n_unmapped": n_pred_unmapped},
        },
    }

    # ---- Write outputs ----
    atomic_write_json(out_json_path, out_obj)

    # concise txt
    lines: List[str] = []
    lines.append(f"pseudoprofile_id: {pid}")
    lines.append(f"source_raw_model: {os.path.basename(raw_model_path)}")
    lines.append("")
    lines.append("CRITERIA:")
    if mapped_criteria_vars:
        for v in mapped_criteria_vars:
            vid = str(v.get("var_id") or v.get("id") or "")
            lbl = str(v.get("label") or "")
            status = str(v.get("mapping_status") or "")
            leaf = str(v.get("mapped_leaf_full_path") or "")
            conf = v.get("mapped_confidence", 0.0)
            if status == "MAPPED":
                lines.append(f"- {vid}: {lbl} -> {leaf} (conf={conf:.2f})")
            else:
                reason = str(v.get("unmapped_reason") or "")
                lines.append(f"- {vid}: {lbl} -> UNMAPPED (reason={reason})")
    else:
        lines.append("- (none)")

    lines.append("")
    lines.append("PREDICTORS:")
    if mapped_predictor_vars:
        for v in mapped_predictor_vars:
            vid = str(v.get("var_id") or v.get("id") or "")
            lbl = str(v.get("label") or "")
            status = str(v.get("mapping_status") or "")
            leaf = str(v.get("mapped_leaf_full_path") or "")
            conf = v.get("mapped_confidence", 0.0)
            if status == "MAPPED":
                lines.append(f"- {vid}: {lbl} -> {leaf} (conf={conf:.2f})")
            else:
                reason = str(v.get("unmapped_reason") or "")
                lines.append(f"- {vid}: {lbl} -> UNMAPPED (reason={reason})")
    else:
        lines.append("- (none)")

    atomic_write_text(out_txt_path, "\n".join(lines) + "\n")

    log_stage(
        "DONE_PSEUDOPROFILE",
        pseudoprofile_id=pid,
        extra={
            "out_json": OUT_MAPPED_JSON_FILENAME,
            "out_txt": OUT_MAPPED_TXT_FILENAME,
            "criteria_mapped": f"{n_crit_mapped}/{len(mapped_criteria_vars)}",
            "predictors_mapped": f"{n_pred_mapped}/{len(mapped_predictor_vars)}",
        }
    )

    return pid, True

# =============================================================================
# BATCH RUNNER
# =============================================================================

def run_batch(
    *,
    runs_root: str,
    run_dir: str,
    criterion_searcher: OntologySearcher,
    predictor_searcher: OntologySearcher,
    max_workers: int,
    limit: int,
    overwrite: bool,
) -> None:
    if not run_dir:
        run_dir = find_latest_run_dir(runs_root)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    log(f"[batch] Using run_dir: {run_dir}")

    raw_paths = discover_raw_models(run_dir)
    if not raw_paths:
        raise FileNotFoundError(f"No {RAW_MODEL_FILENAME} files found in {run_dir}/profiles/*/")

    if limit and limit > 0:
        raw_paths = raw_paths[:limit]
        log(f"[batch] Limit active: {limit} pseudoprofiles")

    log(f"[batch] Found {len(raw_paths)} pseudoprofiles to map.")
    log(f"[batch] max_workers={max_workers} | overwrite={overwrite}")
    log("")

    ok = 0
    fail = 0

    if max_workers <= 1 or len(raw_paths) <= 1:
        for i, p in enumerate(raw_paths, start=1):
            log(f"[batch] ({i}/{len(raw_paths)}) Mapping: {p}")
            try:
                _, success = map_one_pseudoprofile(
                    raw_model_path=p,
                    criterion_searcher=criterion_searcher,
                    predictor_searcher=predictor_searcher,
                    overwrite=overwrite,
                )
                ok += 1 if success else 0
                fail += 0 if success else 1
            except Exception as e:
                fail += 1
                err = f"{type(e).__name__}: {e}"
                tb = traceback.format_exc()
                log(f"[batch-error] {p} failed: {err}\n{tb}")
    else:
        mw = min(max_workers, len(raw_paths))
        log(f"[batch] Parallel pseudoprofile mapping with {mw} workers...")
        with ThreadPoolExecutor(max_workers=mw) as ex:
            futs = {
                ex.submit(
                    map_one_pseudoprofile,
                    raw_model_path=p,
                    criterion_searcher=criterion_searcher,
                    predictor_searcher=predictor_searcher,
                    overwrite=overwrite,
                ): p
                for p in raw_paths
            }
            done = 0
            for fut in as_completed(futs):
                p = futs[fut]
                done += 1
                try:
                    pid, success = fut.result()
                    if success:
                        ok += 1
                    else:
                        fail += 1
                    log(f"[batch] Progress: {done}/{len(raw_paths)} done (last={pid})")
                except Exception as e:
                    fail += 1
                    err = f"{type(e).__name__}: {e}"
                    tb = traceback.format_exc()
                    log(f"[batch-error] {p} failed: {err}\n{tb}")
                    log(f"[batch] Progress: {done}/{len(raw_paths)} done")

    log("")
    log(f"[done] OK={ok} | FAIL={fail}")
    log(f"[done] Outputs written inside each pseudoprofile directory under: {os.path.join(run_dir, 'profiles')}")
    log("")

# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=str, default=DEFAULT_RUNS_ROOT, help="Path to runs/ directory")
    parser.add_argument("--run-dir", type=str, default="", help="Explicit run directory. If empty, picks latest inside --runs-root.")
    parser.add_argument("--criterion-cache", type=str, default=CRITERION_CACHE_DIR, help="CRITERION embeddings cache dir")
    parser.add_argument("--predictor-cache", type=str, default=PREDICTOR_CACHE_DIR, help="PREDICTOR embeddings cache dir")
    parser.add_argument("--max-workers", type=int, default=MAX_PARALLEL_PSEUDOPROFILES, help="Parallel pseudoprofile workers")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N pseudoprofiles (0 = all)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing mapped outputs")
    args = parser.parse_args()

    # Build ontology searchers
    log("[init] Loading ontology caches + building searchers...")

    criterion_embed_model = CRITERION_EMBED_MODEL_ENV  # env override if set (else meta will be used)
    predictor_embed_model = PREDICTOR_EMBED_MODEL_ENV

    criterion_searcher = build_searcher_from_cache(
        cache_dir=args.criterion_cache,
        prefix="CRITERION",
        name="CRITERION",
        embed_model_override=criterion_embed_model,
    )

    predictor_searcher = build_searcher_from_cache(
        cache_dir=args.predictor_cache,
        prefix="PREDICTOR",
        name="PREDICTOR",
        embed_model_override=predictor_embed_model,
    )

    run_batch(
        runs_root=args.runs_root,
        run_dir=args.run_dir,
        criterion_searcher=criterion_searcher,
        predictor_searcher=predictor_searcher,
        max_workers=max(1, int(args.max_workers)),
        limit=int(args.limit),
        overwrite=bool(args.overwrite),
    )

if __name__ == "__main__":
    main()

#TODO: re-think whether RDoC mapping is useful ; ignore the path for now
#TODO: do not force leaf-node resolution mapping — also take ontology pathway of them
#TODO: script creates new run?? — not supposed to happen ; should just create two files in each pseudoprofile sub-dir from input directory
