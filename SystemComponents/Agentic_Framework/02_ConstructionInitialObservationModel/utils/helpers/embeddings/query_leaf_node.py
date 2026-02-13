#!/usr/bin/env python3

"""
query_leaf_node.py

Semantic search over pre-embedded ontology leaf nodes using cosine similarity.

It searches in TWO indices:
- CRITERIONS (criterion ontology leaf embeddings)
- PREDICTORS (predictor ontology leaf embeddings)

Given a free-text query, it embeds the query (must match the embedding model used to build the DB),
then returns the top-k most similar items for each index.

DB layout expected (based on your screenshot):
<db_dir>/
  CRITERIONS/
    CRITERION_leaf_embeddings.npy
    CRITERION_leaf_embedding_norms.npy              (optional but supported)
    CRITERION_leaf_embeddings_meta.json             (optional but supported)
    CRITERION_leaf_paths_EMBEDTEXT.json             (preferred for display)
    CRITERION_leaf_paths_LEXTEXT.json               (fallback)
    CRITERION_leaf_paths_FULL.json                  (fallback)
  PREDICTORS/
    PREDICTOR_leaf_embeddings.npy
    PREDICTOR_leaf_embedding_norms.npy              (optional but supported)
    PREDICTOR_leaf_embeddings_meta.json             (optional but supported)
    PREDICTOR_leaf_paths_EMBEDTEXT.json             (preferred for display)
    PREDICTOR_leaf_paths_LEXTEXT.json               (fallback)
    PREDICTOR_leaf_paths_FULL.json                  (fallback)

Usage examples:
  python semantic_search_embeddings.py \
    --db_dir "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/SystemComponents/Agentic_Framework/02_ConstructionInitialObservationModel/utils/helpers/embeddings" \
    --top_k 50

If you embedded with OpenAI, export:
  export OPENAI_API_KEY="..."

Optionally force backend/model:
  python semantic_search_embeddings.py --backend openai --model text-embedding-3-large ...
  python semantic_search_embeddings.py --backend sentence_transformers --model all-MiniLM-L6-v2 ...

Notes:
- The query embedding dimension MUST match stored embeddings.
- Cosine similarity is computed as (E·q) / (||E|| * ||q||), using stored norms if present.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from dotenv import load_dotenv

# ----------------------------
# Data loading
# ----------------------------
# Load environment variable from .env file
load_dotenv()

@dataclass
class EmbeddingIndex:
    name: str
    base_dir: str
    embeddings: np.ndarray          # shape (N, D)
    norms: np.ndarray              # shape (N,)
    texts: List[str]               # length N, display text per row
    meta: Dict[str, Any]


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _coerce_text_list(obj: Any, expected_len: int, *, fallback_label: str) -> List[str]:
    """
    Make a best-effort to turn a JSON structure into a list[str] of length expected_len.
    """
    if isinstance(obj, list):
        out = [str(x) for x in obj]
        if len(out) != expected_len:
            raise ValueError(f"{fallback_label} is a list but length {len(out)} != expected {expected_len}")
        return out

    if isinstance(obj, dict):
        # common patterns
        for k in ("paths", "items", "data", "texts", "labels"):
            if k in obj and isinstance(obj[k], list):
                out = [str(x) for x in obj[k]]
                if len(out) != expected_len:
                    raise ValueError(
                        f"{fallback_label}.{k} length {len(out)} != expected {expected_len}"
                    )
                return out

        # maybe dict keyed by index -> text
        if all(isinstance(k, str) and k.isdigit() for k in obj.keys()):
            pairs = sorted(((int(k), v) for k, v in obj.items()), key=lambda x: x[0])
            if len(pairs) != expected_len:
                raise ValueError(f"{fallback_label} has {len(pairs)} indexed entries != expected {expected_len}")
            return [str(v) for _, v in pairs]

    raise ValueError(f"Unsupported JSON structure in {fallback_label}: {type(obj)}")


def _pick_paths_file(domain_dir: str, prefix: str, preference: str) -> str:
    """
    Pick which leaf_paths_*.json to use for display.
    preference in {"EMBEDTEXT","LEXTEXT","FULL"} or "AUTO"
    """
    candidates = []
    if preference == "AUTO":
        candidates = ["EMBEDTEXT", "LEXTEXT", "FULL"]
    else:
        candidates = [preference, "EMBEDTEXT", "LEXTEXT", "FULL"]

    for suf in candidates:
        p = os.path.join(domain_dir, f"{prefix}_leaf_paths_{suf}.json")
        if os.path.exists(p):
            return p

    raise FileNotFoundError(
        f"No paths JSON found. Tried: {', '.join([f'{prefix}_leaf_paths_{c}.json' for c in candidates])} in {domain_dir}"
    )


def load_index(db_dir: str, domain: str, prefix: str, display_preference: str, mmap: bool) -> EmbeddingIndex:
    domain_dir = os.path.join(db_dir, domain)
    if not os.path.isdir(domain_dir):
        raise FileNotFoundError(f"Missing directory: {domain_dir}")

    emb_path = os.path.join(domain_dir, f"{prefix}_leaf_embeddings.npy")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Missing embeddings file: {emb_path}")

    embeddings = np.load(emb_path, mmap_mode="r" if mmap else None)
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings at {emb_path} must be 2D (N,D). Got shape {embeddings.shape}.")

    n, d = embeddings.shape

    norms_path = os.path.join(domain_dir, f"{prefix}_leaf_embedding_norms.npy")
    if os.path.exists(norms_path):
        norms = np.load(norms_path, mmap_mode="r" if mmap else None)
        norms = np.asarray(norms).reshape(-1)
        if norms.shape[0] != n:
            raise ValueError(f"Norms length {norms.shape[0]} != embeddings rows {n} for {domain}")
    else:
        # compute once (in memory) for correctness; if mmap and huge, you can store this yourself for speed
        norms = np.linalg.norm(np.asarray(embeddings), axis=1)

    meta_path = os.path.join(domain_dir, f"{prefix}_leaf_embeddings_meta.json")
    meta: Dict[str, Any] = {}
    if os.path.exists(meta_path):
        try:
            meta = _read_json(meta_path)
        except Exception as e:
            meta = {"_meta_read_error": str(e), "_meta_path": meta_path}

    paths_path = _pick_paths_file(domain_dir, prefix, display_preference)
    paths_obj = _read_json(paths_path)
    texts = _coerce_text_list(paths_obj, expected_len=n, fallback_label=os.path.basename(paths_path))

    return EmbeddingIndex(
        name=domain,
        base_dir=domain_dir,
        embeddings=embeddings,
        norms=np.asarray(norms),
        texts=texts,
        meta=meta,
    )


# ----------------------------
# Embedding backends
# ----------------------------

def infer_model_from_meta(meta: Dict[str, Any]) -> Optional[str]:
    """
    Try to infer embedding model name from meta.json (best-effort).
    """
    if not meta:
        return None
    for k in ("embedding_model", "model", "openai_model", "embed_model", "embeddingModel"):
        v = meta.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def embed_query_openai(text: str, model: str) -> np.ndarray:
    """
    OpenAI embeddings via the official SDK.
    Requires: pip install openai
    Requires env: OPENAI_API_KEY
    """
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError("OpenAI backend selected but 'openai' package is not installed. Run: pip install openai") from e

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

    client = OpenAI()
    resp = client.embeddings.create(model=model, input=text)
    vec = resp.data[0].embedding
    return np.asarray(vec, dtype=np.float32)


def embed_query_sentence_transformers(text: str, model: str) -> np.ndarray:
    """
    Local embeddings via sentence-transformers.
    Requires: pip install sentence-transformers
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "sentence_transformers backend selected but 'sentence-transformers' is not installed. "
            "Run: pip install sentence-transformers"
        ) from e

    st = SentenceTransformer(model)
    vec = st.encode([text], normalize_embeddings=False)[0]
    return np.asarray(vec, dtype=np.float32)


def embed_query(text: str, backend: str, model: str) -> np.ndarray:
    backend = backend.lower().strip()
    if backend == "openai":
        return embed_query_openai(text, model)
    if backend in ("sentence_transformers", "st", "sbert"):
        return embed_query_sentence_transformers(text, model)
    raise ValueError(f"Unknown backend: {backend}. Use 'openai' or 'sentence_transformers'.")


# ----------------------------
# Similarity search
# ----------------------------

def topk_cosine(
    embeddings: np.ndarray,
    norms: np.ndarray,
    q: np.ndarray,
    k: int,
    chunk_size: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (indices, scores) of top-k cosine similarities.
    Uses stored norms if available.
    If chunk_size is set, processes in chunks to reduce peak memory.
    """
    eps = 1e-12
    q = np.asarray(q, dtype=np.float32).reshape(-1)
    qnorm = float(np.linalg.norm(q) + eps)
    qn = q / qnorm

    n = embeddings.shape[0]
    k = min(k, n)

    if chunk_size is None or chunk_size <= 0 or chunk_size >= n:
        # single pass
        dots = np.asarray(embeddings @ qn, dtype=np.float32)  # (N,)
        sims = dots / (norms.astype(np.float32) + eps)

        # top-k via argpartition then sort
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return idx, sims[idx]

    # chunked top-k
    best_idx = np.empty((0,), dtype=np.int64)
    best_sim = np.empty((0,), dtype=np.float32)

    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        emb_chunk = embeddings[start:end]
        norm_chunk = norms[start:end].astype(np.float32)

        dots = np.asarray(emb_chunk @ qn, dtype=np.float32)
        sims = dots / (norm_chunk + eps)

        kk = min(k, sims.shape[0])
        local_idx = np.argpartition(-sims, kk - 1)[:kk]
        local_idx = local_idx[np.argsort(-sims[local_idx])]
        local_sim = sims[local_idx]
        local_idx = local_idx.astype(np.int64) + start

        # merge with global
        best_idx = np.concatenate([best_idx, local_idx])
        best_sim = np.concatenate([best_sim, local_sim])

        if best_sim.shape[0] > k:
            keep = np.argpartition(-best_sim, k - 1)[:k]
            keep = keep[np.argsort(-best_sim[keep])]
            best_idx = best_idx[keep]
            best_sim = best_sim[keep]

    return best_idx, best_sim


def print_results(index: EmbeddingIndex, idx: np.ndarray, sims: np.ndarray, top_k: int) -> None:
    print(f"\n=== {index.name} (top {min(top_k, len(idx))}) ===")
    for rank, (i, s) in enumerate(zip(idx.tolist(), sims.tolist()), start=1):
        text = index.texts[i]
        print(f"{rank:02d}. {s: .6f}  |  {text}")


# ----------------------------
# CLI / main loop
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cosine-sim semantic search over precomputed embeddings (CRITERIONS + PREDICTORS).")
    p.add_argument(
        "--db_dir",
        default="/Users/stijnvanseveren/PythonProjects/MASTERPROEF/SystemComponents/Agentic_Framework/02_ConstructionInitialObservationModel/utils/helpers/embeddings",
        help="Path to embeddings DB directory.",
    )
    p.add_argument("--top_k", type=int, default=50, help="How many results to show per index.")
    p.add_argument(
        "--display",
        choices=["AUTO", "EMBEDTEXT", "LEXTEXT", "FULL"],
        default="AUTO",
        help="Which paths JSON to display per result (AUTO prefers EMBEDTEXT > LEXTEXT > FULL).",
    )
    p.add_argument(
        "--backend",
        choices=["openai", "sentence_transformers"],
        default="openai",
        help="Embedding backend to embed your query text.",
    )
    p.add_argument(
        "--model",
        default="",
        help="Embedding model name. If empty, tries to infer from meta.json; otherwise uses a backend default.",
    )
    p.add_argument(
        "--mmap",
        action="store_true",
        help="Memory-map .npy arrays (useful for large embeddings).",
    )
    p.add_argument(
        "--chunk_size",
        type=int,
        default=0,
        help="If >0, compute similarities in chunks of this many rows (reduces peak memory).",
    )
    p.add_argument(
        "--query",
        default="",
        help="If provided, run once and exit. Otherwise runs an interactive prompt loop.",
    )
    return p.parse_args()


def choose_default_model(backend: str) -> str:
    if backend == "openai":
        # Safe default; change if your meta indicates another.
        return "text-embedding-3-small"
    # sentence_transformers default
    return "all-MiniLM-L6-v2"


def main() -> int:
    args = parse_args()
    db_dir = args.db_dir

    crit = load_index(db_dir, "CRITERIONS", "CRITERION", args.display, args.mmap)
    pred = load_index(db_dir, "PREDICTORS", "PREDICTOR", args.display, args.mmap)

    inferred = infer_model_from_meta(crit.meta) or infer_model_from_meta(pred.meta)
    model = args.model.strip() or inferred or choose_default_model(args.backend)

    # Sanity: embedding dimension must match DB dimension.
    db_dim = int(crit.embeddings.shape[1])
    if int(pred.embeddings.shape[1]) != db_dim:
        raise ValueError(f"DB dimension mismatch: CRITERIONS dim {db_dim} vs PREDICTORS dim {pred.embeddings.shape[1]}")

    def run_one(query: str) -> None:
        q = embed_query(query, backend=args.backend, model=model)
        if q.shape[0] != db_dim:
            raise ValueError(
                f"Query embedding dim {q.shape[0]} != DB dim {db_dim}. "
                f"This usually means you're using the wrong embedding model/backend."
            )

        chunk_size = args.chunk_size if args.chunk_size and args.chunk_size > 0 else None

        c_idx, c_sim = topk_cosine(crit.embeddings, crit.norms, q, args.top_k, chunk_size=chunk_size)
        p_idx, p_sim = topk_cosine(pred.embeddings, pred.norms, q, args.top_k, chunk_size=chunk_size)

        print(f"\nQuery: {query}")
        print(f"Embedding backend: {args.backend} | model: {model} | dim: {db_dim}")
        print_results(crit, c_idx, c_sim, args.top_k)
        print_results(pred, p_idx, p_sim, args.top_k)

    if args.query.strip():
        run_one(args.query.strip())
        return 0

    # Interactive mode
    print("Interactive semantic search. Press Enter on an empty line to quit.")
    while True:
        try:
            q = input("\nQuery: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not q:
            break
        run_one(q)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
