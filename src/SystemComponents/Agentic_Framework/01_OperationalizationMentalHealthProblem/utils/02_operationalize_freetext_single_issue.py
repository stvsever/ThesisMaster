#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_operationalize_freetext_single_issue.py

Goal:
    Zero-shot operationalization of a free-text mental health complaint into a single, best-matching CRITERION ontology leaf node.

This script loads caches created by embed_leaf_nodes.py:
  - EMBEDTEXT paths
  - FULL paths
  - LEXTEXT paths
  - embeddings .npy + norms .npy + meta .json

Pipeline:
  Free-text complaint
        ↓
  Query embedding
        ↓
  Dense semantic retrieval (cosine over leaf EMBEDTEXT)
        ↓
  Sparse lexical grounding (BM25 / token overlap / fuzzy match) on LEXTEXT
        ↓
  Recall-safe candidate pooling (union of dense and sparse results)
        ↓
  Temperature-scaled late score fusion + RRF regularization (optional)
        ↓
  Ranked CRITERION leaf list
        ↓
  [Optional] LLM-based semantic adjudication (single best)
        ↓
  Single operationalized CRITERION variable

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
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from collections import deque
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
import heapq
import argparse
from pathlib import Path

import numpy as np

# --------------------------
# CONFIG (EDIT THESE)
# --------------------------
def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        has_eval = (candidate / "evaluation").exists() or (candidate / "Evaluation").exists()
        if has_eval and (candidate / "README.md").exists():
            return candidate
    raise RuntimeError("Could not locate repository root from 02_operationalize_freetext_single_issue.py")


REPO_ROOT = _find_repo_root()

input_file = str(
    REPO_ROOT / "src/SystemComponents/PHOENIX_ontology/separate/01_raw/CRITERION/steps/01_raw/aggregated/CRITERION_ontology.json"
)

OUTPUT_DIR = str(
    REPO_ROOT / "src/SystemComponents/Agentic_Framework/01_OperationalizationMentalHealthProblem/tmp"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cache other (produced by embed_leaf_nodes.py)
PATHS_JSON = os.path.join(OUTPUT_DIR, "CRITERION_leaf_paths_EMBEDTEXT.json")
PATHS_FULL_JSON = os.path.join(OUTPUT_DIR, "CRITERION_leaf_paths_FULL.json")
PATHS_LEX_JSON = os.path.join(OUTPUT_DIR, "CRITERION_leaf_paths_LEXTEXT.json")

EMB_NPY = os.path.join(OUTPUT_DIR, "CRITERION_leaf_embeddings.npy")
DONE_NPY = os.path.join(OUTPUT_DIR, "CRITERION_leaf_embeddings_done.npy")  # not required for query, but kept for parity
NORM_NPY = os.path.join(OUTPUT_DIR, "CRITERION_leaf_embedding_norms.npy")
META_JSON = os.path.join(OUTPUT_DIR, "CRITERION_leaf_embeddings_meta.json")

# Embeddings
EMBED_MODEL = "text-embedding-3-small"
REQUEST_TIMEOUT_SECONDS = 90.0

# Retrieval weights (must sum to 1)
WEIGHT_EMBED = 0.80
WEIGHT_BM25 = 0.12
WEIGHT_TOKEN_OVERLAP = 0.05
WEIGHT_FUZZY = 0.03
assert abs((WEIGHT_EMBED + WEIGHT_BM25 + WEIGHT_TOKEN_OVERLAP + WEIGHT_FUZZY) - 1.0) < 1e-9

# Fusion behavior
# "htssf" = Hybrid Temperature-Scaled Softmax Fusion + RRF backstop (recommended)
# "rrf"   = weighted RRF only
# "scoresum" = minmax-normalized weighted sum (kept as baseline option)
FUSION_METHOD = "htssf"
RRF_K = 60

# HTSSF params
HTSSF_ALPHA = 0.90
HTSSF_TEMPS = (0.07, 1.00, 0.35, 0.35)

# Candidate settings
CANDIDATES_PER_METHOD = 600
TOP_K_RESULTS = 50
CANDIDATE_POOL = 8000

# Optional: a final decision layer that picks a single best leaf from top-N candidates.
ENABLE_LLM_RERANKER = True
LLM_RERANK_MODEL = "gpt-5-mini"
LLM_RERANK_TOPN = 50 # TODO: find optimal number
LLM_RERANK_TEMPERATURE = 1.0 # note: 0.0 is not supported for many models of OpenAI — so I use their default 1.0

# Retry/backoff
MAX_RETRIES = 8
BACKOFF_BASE_SECONDS = 0.8
BACKOFF_MAX_SECONDS = 20.0

# Norm computation chunking
NORM_CHUNK_ROWS = 4096

# GUI defaults
GUI_TITLE = "CRITERION Leaf Matcher (PHOENIX)"
GUI_GEOMETRY = "1100x720"


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
# Embeddings helpers (query + norms)
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
# Optional LLM reranker (final single-best decision)
# --------------------------

def llm_pick_best_leaf(query: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    client = make_openai_client()

    sys_msg = (
        "You are a clinical ontology mapping assistant.\n"
        "Task: choose the SINGLE best-matching CRITERION leaf node that operationalizes a user's free-text mental health complaint.\n"
        "Rules:\n"
        " - Prefer construct-specific symptom/criterion leaves over broad affect or generic wellbeing, if clearly implied.\n"
        " - If the complaint matches a diagnostic criterion leaf (e.g., anhedonia), prefer that over correlated states.\n"
        " - Choose exactly one leaf.\n"
        "Output STRICT JSON with keys: idx (int), confidence (0-1 float), rationale (string).\n"
    )

    lines = []
    for c in candidates:
        lines.append(f"- idx={c['idx']} fused={c['fused_score']:.4f} embed={c['embed_path']} | full={c['full_path']}")

    user_msg = (
        f"Complaint: {query}\n\n"
        f"Candidate leaf nodes:\n" + "\n".join(lines) + "\n\n"
        "Pick the best idx."
    )

    text = None
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

    if not text:
        raise RuntimeError("LLM reranker returned empty output.")

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise RuntimeError(f"LLM reranker did not return JSON. Raw:\n{text}")

    obj = json.loads(m.group(0))
    if "idx" not in obj:
        raise RuntimeError(f"LLM reranker JSON missing idx. Raw:\n{text}")

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

    def query(self, query: str) -> Tuple[List[SearchResult], Optional[Dict[str, Any]]]:
        query = query.strip()
        if not query:
            return [], None

        q_vec = embed_texts_with_retry([query], EMBED_MODEL)[0]
        q = np.array(q_vec, dtype=np.float32)
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

        llm_pick = None
        if ENABLE_LLM_RERANKER and results:
            topn = min(LLM_RERANK_TOPN, len(results))
            cands = [
                {"idx": r.idx, "fused_score": r.fused, "embed_path": r.embed_path, "full_path": r.full_path}
                for r in results[:topn]
            ]
            llm_pick = llm_pick_best_leaf(query=query, candidates=cands)

        return results, llm_pick


# --------------------------
# Load cached index
# --------------------------

def _require_file(path: str, hint: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing required cache file: {path}\nHint: {hint}")

def build_searcher_from_cache() -> CriterionSearcher:
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

    # Validate meta vs hash/model (strong safety check)
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

    return CriterionSearcher(
        full_paths=full_paths,
        embed_paths=embed_paths,
        lex_texts=lex_texts,
        emb=emb,
        norms=norms,
        bm25=bm25,
    )


# --------------------------
# CLI Interface
# --------------------------

def run_cli(searcher: CriterionSearcher) -> None:
    log("")
    log("[ready] CLI search is ready.")
    log("        Enter a mental health complaint; type 'exit' to quit.")
    log("")

    while True:
        query = input("Query> ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            log("Bye.")
            break

        t0 = time.time()
        results, llm_pick = searcher.query(query)
        log(f"[done] Query processed in {time.time()-t0:.2f}s")
        if not results:
            log("[results] No results.")
            continue

        best = results[0]
        log(f"[best] fused={best.fused:.4f} | {best.embed_path}")
        log(f"       FULL: {best.full_path}")

        if llm_pick is not None:
            picked_idx = llm_pick["idx"]
            conf = llm_pick.get("confidence", 0.0)
            rationale = llm_pick.get("rationale", "")
            picked = next((r for r in results if r.idx == picked_idx), None)
            if picked:
                log(f"[LLM BEST] idx={picked_idx} conf={conf:.2f} | {picked.embed_path}")
                log(f"          FULL: {picked.full_path}")
                if rationale:
                    log(f"          why: {rationale}")

        log(f"[results] Top {len(results)}")
        for rank_pos, r in enumerate(results, start=1):
            log(
                f"  {rank_pos:>2}. fused={r.fused:.6f} | emb={r.emb:.4f} bm25={r.bm25:.4f} tok={r.tok:.4f} fuz={r.fuz:.4f} | {r.embed_path}"
            )
        log("-" * 88)
        log("")


# --------------------------
# Tkinter GUI
# --------------------------

def run_gui(searcher: CriterionSearcher) -> None:
    import tkinter as tk
    from tkinter import ttk, messagebox

    root = tk.Tk()
    root.title(GUI_TITLE)
    root.geometry(GUI_GEOMETRY)

    ui_q: "Queue[Tuple[str, Any]]" = Queue()

    top = ttk.Frame(root, padding=10)
    top.pack(fill="x")

    mid = ttk.Frame(root, padding=(10, 0, 10, 10))
    mid.pack(fill="both", expand=True)

    left = ttk.Frame(mid)
    left.pack(side="left", fill="both", expand=True)

    right = ttk.Frame(mid)
    right.pack(side="right", fill="both", expand=True)

    ttk.Label(top, text="Mental health complaint / symptom description:").pack(anchor="w")
    query_var = tk.StringVar()
    query_entry = ttk.Entry(top, textvariable=query_var)
    query_entry.pack(fill="x", expand=True, pady=(4, 8))

    btn_row = ttk.Frame(top)
    btn_row.pack(fill="x")

    status_var = tk.StringVar(value="Ready.")
    ttk.Label(top, textvariable=status_var).pack(anchor="w", pady=(6, 0))

    ttk.Label(left, text="Top matches").pack(anchor="w")
    cols = ("rank", "fused", "embed", "bm25", "tok", "fuz", "leaf")
    tree = ttk.Treeview(left, columns=cols, show="headings", height=18)
    for c in cols:
        tree.heading(c, text=c)
    tree.column("rank", width=45, stretch=False)
    tree.column("fused", width=70, stretch=False)
    tree.column("embed", width=70, stretch=False)
    tree.column("bm25", width=70, stretch=False)
    tree.column("tok", width=70, stretch=False)
    tree.column("fuz", width=70, stretch=False)
    tree.column("leaf", width=520, stretch=True)
    tree.pack(fill="both", expand=True, pady=(4, 8))

    ttk.Label(right, text="Details").pack(anchor="w")
    detail = tk.Text(right, wrap="word", height=18)
    detail.pack(fill="both", expand=True, pady=(4, 8))

    ttk.Label(right, text="LLM single-best pick (optional)").pack(anchor="w")
    llm_box = tk.Text(right, wrap="word", height=8)
    llm_box.pack(fill="both", expand=False, pady=(4, 0))

    results_cache: List[SearchResult] = []
    llm_cache: Optional[Dict[str, Any]] = None

    def render_detail(r: SearchResult) -> None:
        detail.delete("1.0", "end")
        detail.insert("end", f"IDX: {r.idx}\n")
        detail.insert("end", f"Fused: {r.fused:.6f}\n")
        detail.insert("end", f"Scores: emb={r.emb:.4f} bm25={r.bm25:.4f} tok={r.tok:.4f} fuz={r.fuz:.4f}\n\n")
        detail.insert("end", f"EMBED PATH:\n{r.embed_path}\n\n")
        detail.insert("end", f"FULL PATH:\n{r.full_path}\n\n")
        detail.insert("end", f"LEX TEXT (BM25):\n{r.lex_text}\n")

    def render_llm_pick() -> None:
        llm_box.delete("1.0", "end")
        if llm_cache is None:
            llm_box.insert("end", "LLM reranker disabled or not run.\n")
            return
        idx = llm_cache.get("idx")
        conf = llm_cache.get("confidence", 0.0)
        rationale = llm_cache.get("rationale", "")
        llm_box.insert("end", f"idx={idx}  confidence={conf:.2f}\n\n")
        picked = next((r for r in results_cache if r.idx == idx), None)
        if picked:
            llm_box.insert("end", f"EMBED PATH:\n{picked.embed_path}\n\n")
            llm_box.insert("end", f"FULL PATH:\n{picked.full_path}\n\n")
        if rationale:
            llm_box.insert("end", f"Why:\n{rationale}\n")

    def do_search(include_llm: bool) -> None:
        q = query_var.get().strip()
        if not q:
            messagebox.showinfo("Missing input", "Please enter a complaint description.")
            return

        def worker():
            ui_q.put(("status", "Searching..."))
            try:
                global ENABLE_LLM_RERANKER
                prev = ENABLE_LLM_RERANKER
                ENABLE_LLM_RERANKER = bool(include_llm)

                t0 = time.time()
                res, llm = searcher.query(q)
                dt = time.time() - t0

                ENABLE_LLM_RERANKER = prev
                ui_q.put(("results", (res, llm, dt)))
            except Exception as e:
                ui_q.put(("error", str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def on_search():
        do_search(include_llm=False)

    def on_search_llm():
        do_search(include_llm=True)

    def on_select(_evt=None):
        sel = tree.selection()
        if not sel:
            return
        item = sel[0]
        values = tree.item(item, "values")
        if not values:
            return
        try:
            rank = int(values[0]) - 1
        except Exception:
            return
        if 0 <= rank < len(results_cache):
            render_detail(results_cache[rank])

    def clear_table():
        for row in tree.get_children():
            tree.delete(row)

    def pump_ui():
        nonlocal results_cache, llm_cache
        try:
            while True:
                kind, payload = ui_q.get_nowait()
                if kind == "status":
                    status_var.set(str(payload))
                elif kind == "error":
                    status_var.set("Error.")
                    messagebox.showerror("Error", str(payload))
                elif kind == "results":
                    res, llm, dt = payload
                    results_cache = res
                    llm_cache = llm
                    clear_table()
                    for i, r in enumerate(res, start=1):
                        tree.insert("", "end", values=(
                            i,
                            f"{r.fused:.4f}",
                            f"{r.emb:.4f}",
                            f"{r.bm25:.4f}",
                            f"{r.tok:.4f}",
                            f"{r.fuz:.4f}",
                            r.embed_path,
                        ))
                    status_var.set(f"Done. {len(res)} results in {dt:.2f}s. (Fusion={FUSION_METHOD})")
                    if res:
                        tree.selection_set(tree.get_children()[0])
                        render_detail(res[0])
                    render_llm_pick()
        except Empty:
            pass
        root.after(60, pump_ui)

    ttk.Button(btn_row, text="Search", command=on_search).pack(side="left")
    ttk.Button(btn_row, text="Search + Pick Best (LLM)", command=on_search_llm).pack(side="left", padx=(8, 0))
    ttk.Button(btn_row, text="Quit", command=root.destroy).pack(side="right")

    tree.bind("<<TreeviewSelect>>", on_select)

    def on_enter(_evt):
        on_search()
    query_entry.bind("<Return>", on_enter)
    query_entry.focus_set()

    pump_ui()
    root.mainloop()


# --------------------------
# Main
# --------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", action="store_true", help="Run in terminal mode instead of Tkinter GUI.")
    args = parser.parse_args()

    searcher = build_searcher_from_cache()
    if args.cli:
        run_cli(searcher)
    else:
        try:
            run_gui(searcher)
        except Exception as e:
            log(f"[gui] Failed to start GUI ({type(e).__name__}: {e}). Falling back to CLI.")
            run_cli(searcher)

if __name__ == "__main__":
    main()

# TODO: allow for optimal resolution pick --> right now only takes leaf nodes --> often too much depth
# TODO: then later implement similar logic for predictor search --> with ADDITIONAL relevance ranking logic of plausible relatedness estimates
