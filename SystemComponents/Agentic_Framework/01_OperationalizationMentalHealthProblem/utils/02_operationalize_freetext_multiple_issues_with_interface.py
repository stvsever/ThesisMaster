#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_operationalize_freetext_multiple_issues_with_interface.py

Goal:
    Operationalize a single free-text mental health complaint into MULTIPLE ontology leaf nodes by:
      (1) ALWAYS running an LLM-based clinical-expert decomposition into distinct criteria/variables (ontology-agnostic).
      (2) Mapping EACH decomposed criterion to the best matching CRITERION ontology leaf node via hybrid retrieval.
      (3) Optionally running an LLM-based per-criterion adjudication step to select the single best leaf from top-N candidates.

This script loads caches created by embed_leaf_nodes.py:
  - EMBEDTEXT paths
  - FULL paths
  - LEXTEXT paths
  - embeddings .npy + norms .npy + meta .json

Pipeline:
  Free-text complaint
        ↓
  LLM decomposition (ontology-agnostic) -> list of criteria variables
        ↓
  Batch embeddings (one call) for each criterion query text
        ↓
  For each criterion:
        Dense semantic retrieval (cosine over leaf EMBEDTEXT)
        Sparse lexical grounding (BM25 / token overlap / fuzzy match) on LEXTEXT
        Candidate pooling + fusion
        [Optional] LLM adjudication -> single best leaf
        ↓
  Final set of (criterion_variable -> ontology_leaf) mappings

Output:
  - Console logs: decomposition JSON + mapping summary
  - Auto-saved run JSON artifact to OUTPUT_DIR/operationalizations/
  - GUI: criteria list + per-criterion matches + overall mapping table

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
import hashlib
import threading
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, Optional
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, as_completed
import heapq
import argparse
import traceback

import numpy as np

# --------------------------
# CONFIG (EDIT THESE)
# --------------------------

# Ontology input file is not required here (we use cached leaf text/embeddings),
# but kept for parity / provenance logs.
input_file = "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/SystemComponents/PHOENIX_ontology/separate/01_raw/CRITERION/steps/01_raw/aggregated/CRITERION_ontology.json"

# Cache directory containing outputs from embed_leaf_nodes.py
# You can override via env var CRITERION_CACHE_DIR.
DEFAULT_OUTPUT_DIR = "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/SystemComponents/Agentic_Framework/01_OperationalizationMentalHealthProblem/tmp"
OUTPUT_DIR = os.environ.get("CRITERION_CACHE_DIR", DEFAULT_OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cache other (produced by embed_leaf_nodes.py)
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
DECOMP_MODEL = os.environ.get("CRITERION_DECOMP_MODEL", "gpt-5")
DECOMP_TEMPERATURE = float(os.environ.get("CRITERION_DECOMP_T", "1.0"))

# Per-criterion optional LLM picker
ENABLE_LLM_RERANKER_DEFAULT = True
LLM_RERANK_MODEL = os.environ.get("CRITERION_RERANK_MODEL", "gpt-5-nano")
LLM_RERANK_TOPN = int(os.environ.get("CRITERION_RERANK_TOPN", "200")) # 50 --> 200
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
TOP_K_RESULTS = 200 # 50 --> 200
CANDIDATE_POOL = 8000

# Decomposition -> retrieval query building
# If True, retrieval query text = "<criterion> | evidence: <evidence>"
INCLUDE_EVIDENCE_IN_QUERY = True

# Parallelism
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

# GUI defaults
GUI_TITLE = "CRITERION Operationalizer (PHOENIX) — Multi-criteria"
GUI_GEOMETRY = "1280x820"


# --------------------------
# Logging helpers
# --------------------------

def ts() -> str:
    return time.strftime("%H:%M:%S")

def log(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)

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

def _safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Sometimes models include trailing commas or code fences; attempt minimal cleanup
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()
        return json.loads(cleaned)

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
                resp = client.responses.create(
                    model=DECOMP_MODEL,
                    input=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=DECOMP_TEMPERATURE,
                )
                text = resp.output_text
            except Exception:
                resp = client.chat.completions.create(
                    model=DECOMP_MODEL,
                    messages=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=DECOMP_TEMPERATURE,
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

    obj_text = _extract_json_object(text)
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
    """
    client = make_openai_client()

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
        # embed_path is typically the "EMBEDTEXT" readable path; full_path gives full lineage
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
                resp = client.responses.create(
                    model=LLM_RERANK_MODEL,
                    input=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=LLM_RERANK_TEMPERATURE,
                )
                text = resp.output_text
            except Exception:
                resp = client.chat.completions.create(
                    model=LLM_RERANK_MODEL,
                    messages=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=LLM_RERANK_TEMPERATURE,
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

    obj_text = _extract_json_object(text)
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
            #return f"{base} | evidence: {self.evidence.strip()}" # ignore 'evidence' for now to reduce possible semantic noise
            return f"{base}"
        return base


@dataclass
class CriterionMapping:
    variable: CriterionVariable
    query_text_used: str
    results: List[SearchResult]
    chosen_method: str  # "top_fused" or "llm"
    chosen_idx: int
    chosen_confidence: float
    chosen_rationale: str
    debug: Dict[str, Any]

    @property
    def chosen_leaf_embed_path(self) -> str:
        r = next((x for x in self.results if x.idx == self.chosen_idx), None)
        return r.embed_path if r else ""

    @property
    def chosen_leaf_full_path(self) -> str:
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

    # 1) Decompose (always-on)
    decomp = llm_decompose_complaint(complaint)
    meta = decomp.get("meta", {})
    variables_raw = decomp.get("variables", [])

    log("[decomp] Decomposition output (JSON):")
    log(json.dumps(decomp, ensure_ascii=False, indent=2))

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
    log(f"[embed] Embedding {len(query_texts)} decomposed criteria (batch)...")
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
                chosen_method="top_fused",
                chosen_idx=-1,
                chosen_confidence=0.0,
                chosen_rationale="No results.",
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
                return m

            topn = min(LLM_RERANK_TOPN, len(m.results))
            cands = [
                {"idx": r.idx, "fused_score": r.fused, "embed_path": r.embed_path, "full_path": r.full_path}
                for r in m.results[:topn]
            ]
            pick = llm_pick_best_leaf_for_criterion(asdict(m.variable), cands)

            chosen_idx = int(pick["idx"])
            chosen_conf = float(pick.get("confidence", 0.0))
            chosen_rat = str(pick.get("rationale", ""))

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

    # Console summary
    log("")
    log("[summary] Mapped decomposed criteria -> ontology leaves:")
    for m in mappings:
        v = m.variable
        leaf = m.chosen_leaf_embed_path or "(none)"
        log(f"  {v.id} | {v.label}: {v.criterion}")
        log(f"      -> {leaf}")
    log(f"[saved] {out_path}")
    log("")

    return decomp, mappings, out_path


# --------------------------
# CLI Interface
# --------------------------

def run_cli(searcher: CriterionSearcher, enable_llm_reranker: bool) -> None:
    log("")
    log("[ready] CLI operationalizer is ready.")
    log("        Paste a complaint; type 'exit' to quit.")
    log("")

    while True:
        complaint = input("Complaint> ").strip()
        if not complaint:
            continue
        if complaint.lower() in {"exit", "quit", "q"}:
            log("Bye.")
            break

        try:
            t0 = time.time()
            _, mappings, out_path = operationalize_complaint(
                searcher=searcher,
                complaint=complaint,
                enable_llm_reranker=enable_llm_reranker,
            )
            dt = time.time() - t0
            log(f"[done] Full run in {dt:.2f}s | saved={out_path}")

            for m in mappings:
                best = next((r for r in m.results if r.idx == m.chosen_idx), None)
                if best:
                    log(f"- {m.variable.id} {m.variable.label} -> {best.embed_path} (fused={best.fused:.4f})")
                else:
                    log(f"- {m.variable.id} {m.variable.label} -> (none)")
            log("-" * 88)
            log("")
        except Exception as e:
            log(f"[error] {type(e).__name__}: {e}")
            log(traceback.format_exc())


# --------------------------
# Tkinter GUI
# --------------------------

def run_gui(searcher: CriterionSearcher, enable_llm_reranker_default: bool) -> None:
    import tkinter as tk
    from tkinter import ttk, messagebox

    root = tk.Tk()
    root.title(GUI_TITLE)
    root.geometry(GUI_GEOMETRY)

    ui_q: "Queue[Tuple[str, Any]]" = Queue()

    # --- Layout frames
    top = ttk.Frame(root, padding=10)
    top.pack(fill="x")

    mid = ttk.Frame(root, padding=(10, 0, 10, 10))
    mid.pack(fill="both", expand=True)

    bottom = ttk.Frame(root, padding=(10, 0, 10, 10))
    bottom.pack(fill="x")

    left = ttk.Frame(mid)
    left.pack(side="left", fill="y")

    center = ttk.Frame(mid)
    center.pack(side="left", fill="both", expand=True, padx=(10, 0))

    right = ttk.Frame(mid)
    right.pack(side="left", fill="both", expand=True, padx=(10, 0))

    # --- Top: complaint input + controls
    ttk.Label(top, text="Free-text complaint (multi-criteria operationalization):").pack(anchor="w")
    complaint_box = tk.Text(top, height=6, wrap="word")
    complaint_box.pack(fill="x", expand=True, pady=(4, 8))

    controls = ttk.Frame(top)
    controls.pack(fill="x")

    enable_llm_var = tk.BooleanVar(value=bool(enable_llm_reranker_default))
    ttk.Checkbutton(controls, text="Use LLM per-criterion best-leaf adjudication", variable=enable_llm_var).pack(side="left")

    status_var = tk.StringVar(value="Ready.")
    ttk.Label(top, textvariable=status_var).pack(anchor="w", pady=(6, 0))

    # --- Left: decomposed criteria list
    ttk.Label(left, text="Decomposed criteria").pack(anchor="w")
    criteria_list = tk.Listbox(left, height=24, width=40)
    criteria_list.pack(fill="y", expand=False, pady=(4, 8))

    # --- Center: per-criterion matches table
    ttk.Label(center, text="Matches for selected criterion").pack(anchor="w")
    cols = ("rank", "fused", "embed", "bm25", "tok", "fuz", "leaf")
    matches_tree = ttk.Treeview(center, columns=cols, show="headings", height=18)
    for c in cols:
        matches_tree.heading(c, text=c)
    matches_tree.column("rank", width=45, stretch=False)
    matches_tree.column("fused", width=70, stretch=False)
    matches_tree.column("embed", width=70, stretch=False)
    matches_tree.column("bm25", width=70, stretch=False)
    matches_tree.column("tok", width=70, stretch=False)
    matches_tree.column("fuz", width=70, stretch=False)
    matches_tree.column("leaf", width=520, stretch=True)
    matches_tree.pack(fill="both", expand=True, pady=(4, 8))

    # --- Right: criterion details + chosen leaf
    ttk.Label(right, text="Criterion details").pack(anchor="w")
    crit_detail = tk.Text(right, wrap="word", height=12)
    crit_detail.pack(fill="both", expand=False, pady=(4, 8))

    ttk.Label(right, text="Chosen leaf (for selected criterion)").pack(anchor="w")
    chosen_box = tk.Text(right, wrap="word", height=10)
    chosen_box.pack(fill="both", expand=False, pady=(4, 0))

    # --- Bottom: overall mapping summary
    ttk.Label(bottom, text="Overall mapping (criterion → chosen leaf)").pack(anchor="w")
    sum_cols = ("id", "label", "criterion", "chosen_leaf")
    summary_tree = ttk.Treeview(bottom, columns=sum_cols, show="headings", height=7)
    for c in sum_cols:
        summary_tree.heading(c, text=c)
    summary_tree.column("id", width=60, stretch=False)
    summary_tree.column("label", width=180, stretch=False)
    summary_tree.column("criterion", width=540, stretch=True)
    summary_tree.column("chosen_leaf", width=420, stretch=True)
    summary_tree.pack(fill="x", expand=True, pady=(4, 8))

    btn_row = ttk.Frame(bottom)
    btn_row.pack(fill="x")

    # --- State
    current_decomp: Optional[Dict[str, Any]] = None
    current_mappings: List[CriterionMapping] = []
    criteria_index_to_mapping: Dict[int, CriterionMapping] = {}

    def clear_tree(tree: ttk.Treeview) -> None:
        for row in tree.get_children():
            tree.delete(row)

    def render_selected(mapping: CriterionMapping) -> None:
        # criterion details
        crit_detail.delete("1.0", "end")
        v = mapping.variable
        crit_detail.insert("end", f"{v.id} — {v.label}\n\n")
        crit_detail.insert("end", f"criterion: {v.criterion}\n")
        crit_detail.insert("end", f"evidence: {v.evidence}\n")
        crit_detail.insert("end", f"polarity: {v.polarity}\n")
        crit_detail.insert("end", f"timeframe: {v.timeframe}\n")
        crit_detail.insert("end", f"severity_0_1: {v.severity_0_1:.2f}\n")
        crit_detail.insert("end", f"confidence_0_1: {v.confidence_0_1:.2f}\n\n")
        crit_detail.insert("end", f"retrieval_query_used: {mapping.query_text_used}\n")
        crit_detail.insert("end", f"debug: {json.dumps(mapping.debug)}\n")

        # matches table
        clear_tree(matches_tree)
        for i, r in enumerate(mapping.results, start=1):
            matches_tree.insert("", "end", values=(
                i,
                f"{r.fused:.4f}",
                f"{r.emb:.4f}",
                f"{r.bm25:.4f}",
                f"{r.tok:.4f}",
                f"{r.fuz:.4f}",
                r.embed_path,
            ))

        # chosen leaf box
        chosen_box.delete("1.0", "end")
        best = next((r for r in mapping.results if r.idx == mapping.chosen_idx), None)
        chosen_box.insert("end", f"method: {mapping.chosen_method}\n")
        chosen_box.insert("end", f"idx: {mapping.chosen_idx}\n")
        if mapping.chosen_method == "llm":
            chosen_box.insert("end", f"llm_confidence: {mapping.chosen_confidence:.2f}\n")
        if best:
            chosen_box.insert("end", f"fused: {best.fused:.6f}\n\n")
            chosen_box.insert("end", f"EMBED PATH:\n{best.embed_path}\n\n")
            chosen_box.insert("end", f"FULL PATH:\n{best.full_path}\n\n")
            chosen_box.insert("end", f"LEX TEXT:\n{best.lex_text}\n\n")
        if mapping.chosen_rationale:
            chosen_box.insert("end", f"rationale:\n{mapping.chosen_rationale}\n")

    def render_summary(mappings: List[CriterionMapping]) -> None:
        clear_tree(summary_tree)
        for m in mappings:
            v = m.variable
            leaf = m.chosen_leaf_embed_path or "(none)"
            summary_tree.insert("", "end", values=(v.id, v.label, v.criterion, leaf))

    def do_run() -> None:
        complaint = complaint_box.get("1.0", "end").strip()
        if not complaint:
            messagebox.showinfo("Missing input", "Please enter a complaint.")
            return

        run_btn.config(state="disabled")
        status_var.set("Running decomposition + ontology mapping...")

        use_llm = bool(enable_llm_var.get())

        def worker():
            ui_q.put(("status", "Running decomposition + mapping..."))
            try:
                t0 = time.time()
                decomp, mappings, out_path = operationalize_complaint(
                    searcher=searcher,
                    complaint=complaint,
                    enable_llm_reranker=use_llm,
                )
                dt = time.time() - t0
                ui_q.put(("done", (decomp, mappings, out_path, dt)))
            except Exception as e:
                ui_q.put(("error", (str(e), traceback.format_exc())))

        threading.Thread(target=worker, daemon=True).start()

    def on_select_criterion(_evt=None):
        sel = criteria_list.curselection()
        if not sel:
            return
        idx = int(sel[0])
        m = criteria_index_to_mapping.get(idx)
        if m:
            render_selected(m)

    def pump_ui() -> None:
        nonlocal current_decomp, current_mappings, criteria_index_to_mapping
        try:
            while True:
                kind, payload = ui_q.get_nowait()
                if kind == "status":
                    status_var.set(str(payload))
                elif kind == "error":
                    msg, tb = payload
                    status_var.set("Error.")
                    messagebox.showerror("Error", f"{msg}\n\n{tb}")
                elif kind == "done":

                    run_btn.config(state="normal")

                    decomp, mappings, out_path, dt = payload
                    current_decomp = decomp
                    current_mappings = mappings

                    # Fill criteria list
                    criteria_list.delete(0, "end")
                    criteria_index_to_mapping = {}
                    for i, m in enumerate(mappings):
                        v = m.variable
                        criteria_list.insert("end", f"{v.id} — {v.label}")
                        criteria_index_to_mapping[i] = m

                    # Render summary + select first
                    render_summary(mappings)
                    if mappings:
                        criteria_list.selection_clear(0, "end")
                        criteria_list.selection_set(0)
                        criteria_list.activate(0)
                        render_selected(mappings[0])

                    status_var.set(f"Done. {len(mappings)} criteria mapped in {dt:.2f}s. Saved: {out_path}")
        except Empty:
            pass
        root.after(80, pump_ui)

    run_btn = ttk.Button(
        btn_row,
        text="▶ READY — Run Operationalization",
        command=do_run
    )
    run_btn.pack(side="left", padx=(0, 10))
    ttk.Button(btn_row, text="Quit", command=root.destroy).pack(side="right")

    criteria_list.bind("<<ListboxSelect>>", on_select_criterion)

    pump_ui()
    root.mainloop()


# --------------------------
# Main
# --------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", action="store_true", help="Run in terminal mode instead of Tkinter GUI.")
    parser.add_argument("--no-llm-rerank", action="store_true", help="Disable per-criterion LLM best-leaf adjudication.")
    args = parser.parse_args()

    searcher, _cache_info = build_searcher_from_cache()

    enable_llm_reranker = ENABLE_LLM_RERANKER_DEFAULT and (not args.no_llm_rerank)

    if args.cli:
        run_cli(searcher, enable_llm_reranker=enable_llm_reranker)
    else:
        try:
            run_gui(searcher, enable_llm_reranker_default=enable_llm_reranker)
        except Exception as e:
            log(f"[gui] Failed to start GUI ({type(e).__name__}: {e}). Falling back to CLI.")
            run_cli(searcher, enable_llm_reranker=enable_llm_reranker)


if __name__ == "__main__":
    main()

# TODO: allow for possible that 'NO MATCH' is found with ontology during LLM-based adjunction
# TODO: further prompt engineer the current decomposition instruction --> allow for reasoned exploration instead of DIRECT decomposition
# NOTE: two versions of this; one callable function that can loop through large list of free text descriptions ; and one CLI/GUI interface for interactive use
