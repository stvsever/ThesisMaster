#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
embed_leaf_nodes.py

Goal:
    Precompute and cache CRITERION leaf-node representations + embeddings for later zero-shot operationalization.

This script performs:
  1) Load CRITERION ontology JSON
  2) Extract ALL leaf paths (empty dicts)
  3) Create:
       - FULL paths (human inspection)
       - EMBEDTEXT paths (leaf-focused text for embeddings)
       - LEXTEXT paths (lexicalized text used later for BM25/token/fuzzy)
  4) Build + cache embeddings with resumable batching:
       - embeddings .npy memmap
       - done-state .npy
       - norms .npy (row norms for cosine sim)
       - meta .json with hash/model/dim/status

Notes:
  - This script does NOT run search/GUI. That is done in 02_operationalize_freetext_single_issue.py
  - Requires OPENAI_API_KEY, provided via env or .env

Prerequisites:
  pip install openai numpy
Optional:
  pip install python-dotenv
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
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from collections import Counter, deque
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
    raise RuntimeError("Could not locate repository root from 01_embed_leaf_nodes.py")


REPO_ROOT = _find_repo_root()
input_file = str(
    REPO_ROOT
    / "src/SystemComponents/PHOENIX_ontology/separate/01_raw/CRITERION/steps/01_raw/aggregated/CRITERION_ontology.json"
)

OUTPUT_DIR = str(
    REPO_ROOT / "src/SystemComponents/Agentic_Framework/01_OperationalizationMentalHealthProblem/tmp"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cache other
PATHS_JSON = os.path.join(OUTPUT_DIR, "CRITERION_leaf_paths_EMBEDTEXT.json")
PATHS_FULL_JSON = os.path.join(OUTPUT_DIR, "CRITERION_leaf_paths_FULL.json")
PATHS_LEX_JSON = os.path.join(OUTPUT_DIR, "CRITERION_leaf_paths_LEXTEXT.json")

EMB_NPY = os.path.join(OUTPUT_DIR, "CRITERION_leaf_embeddings.npy")
DONE_NPY = os.path.join(OUTPUT_DIR, "CRITERION_leaf_embeddings_done.npy")
NORM_NPY = os.path.join(OUTPUT_DIR, "CRITERION_leaf_embedding_norms.npy")
META_JSON = os.path.join(OUTPUT_DIR, "CRITERION_leaf_embeddings_meta.json")

USE_CACHE_IF_AVAILABLE = True

# Embeddings
EMBED_MODEL = "text-embedding-3-small"
EMBED_BATCH_SIZE = 96
MAX_WORKERS = 12
MAX_INFLIGHT = 24
STATUS_EVERY_SECONDS = 8.0
CHECKPOINT_EVERY_SECONDS = 25.0
CHECKPOINT_EVERY_BATCHES = 20
REQUEST_TIMEOUT_SECONDS = 90.0

# Path shaping (for embedding text)
DROP_PREFIX_LEVELS_AFTER_CRITERION = 2
LEAF_STAR = "*"
KEEP_LAST_N_NODES: Optional[int] = None

# Retry/backoff
MAX_RETRIES = 8
BACKOFF_BASE_SECONDS = 0.8
BACKOFF_MAX_SECONDS = 20.0

# Norm computation chunking
NORM_CHUNK_ROWS = 4096


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

def atomic_write_json(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)


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
# CRITERION extraction logic
# --------------------------

def find_criterion_root(obj: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return None

    if "CRITERION" in obj and isinstance(obj["CRITERION"], dict):
        return obj["CRITERION"]

    if "ONTOLOGY" in obj and isinstance(obj["ONTOLOGY"], dict):
        onto = obj["ONTOLOGY"]
        if "CRITERION" in onto and isinstance(onto["CRITERION"], dict):
            return onto["CRITERION"]

    for _, v in obj.items():
        if isinstance(v, dict):
            found = find_criterion_root(v)
            if found is not None:
                return found
        elif isinstance(v, list):
            for item in v:
                found = find_criterion_root(item)
                if found is not None:
                    return found
    return None


def iter_leaf_paths_iterative(root: Any) -> List[Tuple[str, ...]]:
    """
    Leaf = empty dict {}.
    Returns full paths as tuples, starting AFTER CRITERION (root children).
    """
    out: List[Tuple[str, ...]] = []
    stack: List[Tuple[Any, Tuple[str, ...]]] = [(root, tuple())]
    while stack:
        node, prefix = stack.pop()
        if isinstance(node, dict):
            if len(node) == 0:
                out.append(prefix)
            else:
                for k, v in node.items():
                    stack.append((v, prefix + (str(k),)))
        else:
            out.append(prefix)
    return out


def path_tuple_to_text_full(path: Tuple[str, ...]) -> str:
    return " / ".join(path)


def path_tuple_to_text_embed(path: Tuple[str, ...]) -> str:
    """
    Leaf-focused embedding text:
      - drops first DROP_PREFIX_LEVELS_AFTER_CRITERION nodes
      - optionally keeps only last KEEP_LAST_N_NODES nodes
      - marks the leaf with a star prefix
    """
    p = list(path)

    if DROP_PREFIX_LEVELS_AFTER_CRITERION > 0 and len(p) > DROP_PREFIX_LEVELS_AFTER_CRITERION:
        p = p[DROP_PREFIX_LEVELS_AFTER_CRITERION:]
    elif DROP_PREFIX_LEVELS_AFTER_CRITERION > 0 and len(p) <= DROP_PREFIX_LEVELS_AFTER_CRITERION:
        p = p[-1:] if p else p

    if KEEP_LAST_N_NODES is not None and KEEP_LAST_N_NODES > 0 and len(p) > KEEP_LAST_N_NODES:
        p = p[-KEEP_LAST_N_NODES:]

    if not p:
        return f"{LEAF_STAR}<EMPTY_PATH>"

    leaf = p[-1]
    ctx = p[:-1]
    if ctx:
        return " / ".join(ctx + [f"{LEAF_STAR}{leaf}"])
    return f"{LEAF_STAR}{leaf}"


def path_tuple_to_text_lex(path: Tuple[str, ...]) -> str:
    """
    Lexical text used later for BM25/token-overlap:
    - Uses trimmed context + leaf as SPACE-separated terms
    - Removes underscores/slashes/stars
    - Includes parent context and leaf for better match to free text
    """
    embed = path_tuple_to_text_embed(path)
    s = embed.replace("/", " ").replace("*", " ").replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


# --------------------------
# Embeddings helpers (resumable + bounded inflight + heartbeat)
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


def load_or_build_embeddings(paths_embed: List[str], paths_hash: str) -> Tuple[str, np.ndarray]:
    n = len(paths_embed)
    if n == 0:
        raise RuntimeError("No paths to embed.")

    if USE_CACHE_IF_AVAILABLE and os.path.isfile(EMB_NPY) and os.path.isfile(PATHS_JSON) and os.path.isfile(META_JSON):
        try:
            meta = json.load(open(META_JSON, "r", encoding="utf-8"))
            if (
                meta.get("paths_hash") == paths_hash
                and meta.get("model") == EMBED_MODEL
                and meta.get("num_paths") == n
                and meta.get("status") == "complete"
            ):
                log("[cache] Complete cache found. Verifying embed-paths + loading norms...")
                cached_paths = json.load(open(PATHS_JSON, "r", encoding="utf-8"))
                if cached_paths != paths_embed:
                    raise ValueError("Cached EMBEDTEXT paths mismatch (ordering/content).")

                if os.path.isfile(NORM_NPY):
                    norms = np.load(NORM_NPY).astype(np.float32)
                    if norms.shape[0] != n:
                        raise ValueError("Norm cache wrong length.")
                    log(f"[cache] Loaded norms: {norms.shape} ({human_bytes(os.path.getsize(NORM_NPY))})")
                else:
                    norms = compute_and_cache_norms(EMB_NPY, NORM_NPY)

                log("[cache] Ready.")
                return EMB_NPY, norms
        except Exception as e:
            log(f"[cache] Complete-cache load failed: {type(e).__name__}: {e}")
            log("[cache] Will attempt resume or rebuild...")

    if not os.path.isfile(PATHS_JSON):
        log(f"[cache] Writing embed-paths -> {PATHS_JSON}")
        atomic_write_json(PATHS_JSON, paths_embed)
        log(f"[cache] Embed-paths saved ({human_bytes(os.path.getsize(PATHS_JSON))})")

    batches: List[Tuple[int, int]] = []
    for start in range(0, n, EMBED_BATCH_SIZE):
        end = min(n, start + EMBED_BATCH_SIZE)
        batches.append((start, end))
    num_batches = len(batches)
    log(f"[embed] Total paths={n}, batch_size={EMBED_BATCH_SIZE}, batches={num_batches}")

    meta: Dict[str, Any] = {}
    if os.path.isfile(META_JSON):
        try:
            meta = json.load(open(META_JSON, "r", encoding="utf-8"))
        except Exception:
            meta = {}

    can_resume = (
        os.path.isfile(EMB_NPY)
        and os.path.isfile(DONE_NPY)
        and meta.get("paths_hash") == paths_hash
        and meta.get("model") == EMBED_MODEL
        and meta.get("num_paths") == n
        and meta.get("status") in {"partial", "embedding"}
        and meta.get("num_batches") == num_batches
        and meta.get("embed_batch_size") == EMBED_BATCH_SIZE
        and isinstance(meta.get("embedding_dim"), int)
    )

    embedding_dim: Optional[int] = None

    if can_resume:
        embedding_dim = int(meta["embedding_dim"])
        log(f"[resume] Resuming embedding. dim={embedding_dim}, file={EMB_NPY}")
        done = np.load(DONE_NPY).astype(bool)
        if done.shape[0] != num_batches:
            log("[resume] DONE_NPY has wrong shape -> rebuild from scratch.")
            can_resume = False
    else:
        log("[embed] No usable resume state. Starting fresh embedding.")

    if not can_resume:
        for stale_path in [EMB_NPY, DONE_NPY, NORM_NPY]:
            if os.path.isfile(stale_path):
                log(f"[embed] Removing stale file: {stale_path}")
                try:
                    os.remove(stale_path)
                except Exception as e:
                    log(f"[embed] Warning: could not remove {stale_path}: {type(e).__name__}: {e}")

        log("[embed] Embedding first batch to determine vector dimension...")
        first_start, first_end = batches[0]
        first_vecs = embed_texts_with_retry(paths_embed[first_start:first_end], EMBED_MODEL)
        first_arr = np.array(first_vecs, dtype=np.float32)
        embedding_dim = int(first_arr.shape[1])
        log(f"[embed] Determined embedding_dim={embedding_dim}")

        log(f"[embed] Creating memmap .npy for embeddings -> {EMB_NPY}")
        emb_mm = np.lib.format.open_memmap(
            EMB_NPY, mode="w+", dtype=np.float32, shape=(n, embedding_dim)
        )
        emb_mm[first_start:first_end, :] = first_arr
        emb_mm.flush()
        log(f"[embed] Wrote batch 1/{num_batches} ({first_end-first_start} items).")

        done = np.zeros((num_batches,), dtype=bool)
        done[0] = True
        np.save(DONE_NPY, done)
        log(f"[embed] Saved done-state -> {DONE_NPY}")

        meta = {
            "status": "partial",
            "model": EMBED_MODEL,
            "num_paths": n,
            "paths_hash": paths_hash,
            "embedding_dim": embedding_dim,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "num_batches": num_batches,
            "embed_batch_size": EMBED_BATCH_SIZE,
            "completed_batches": int(done.sum()),
        }
        atomic_write_json(META_JSON, meta)
        log(f"[embed] Wrote meta -> {META_JSON}")

    assert embedding_dim is not None
    emb_mm = np.load(EMB_NPY, mmap_mode="r+")
    done = np.load(DONE_NPY).astype(bool)

    missing_batch_ids = [i for i in range(num_batches) if not done[i]]
    if not missing_batch_ids:
        log("[embed] All batches already complete. Skipping embedding.")
    else:
        log(f"[embed] Missing batches: {len(missing_batch_ids)}/{num_batches}")
        meta["status"] = "embedding"
        atomic_write_json(META_JSON, meta)

        in_flight: Dict[Any, int] = {}
        batch_start_time: Dict[int, float] = {}
        completed_items = int(sum((batches[i][1] - batches[i][0]) for i in range(num_batches) if done[i]))
        t0 = time.time()
        last_checkpoint_t = time.time()
        batches_since_checkpoint = 0
        retry_queue = deque()

        def submit_batch(ex: ThreadPoolExecutor, bid: int) -> None:
            start, end = batches[bid]
            fut = ex.submit(embed_texts_with_retry, paths_embed[start:end], EMBED_MODEL)
            in_flight[fut] = bid
            batch_start_time[bid] = time.time()

        log(f"[embed] Executor max_workers={MAX_WORKERS}, MAX_INFLIGHT={MAX_INFLIGHT}")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            miss_iter = iter(missing_batch_ids)

            def top_up() -> None:
                while len(in_flight) < MAX_INFLIGHT:
                    if retry_queue:
                        bid = retry_queue.popleft()
                        if not done[bid]:
                            submit_batch(ex, bid)
                        continue
                    try:
                        bid = next(miss_iter)
                    except StopIteration:
                        break
                    if done[bid]:
                        continue
                    submit_batch(ex, bid)

            top_up()

            while in_flight:
                done_futs, _ = wait(in_flight.keys(), timeout=STATUS_EVERY_SECONDS, return_when=FIRST_COMPLETED)
                now = time.time()

                if not done_futs:
                    pending = len(in_flight)
                    elapsed = now - t0
                    rate = completed_items / elapsed if elapsed > 0 else 0.0
                    pct = (completed_items / n) * 100.0
                    oldest_bid = min(batch_start_time, key=lambda k: batch_start_time[k]) if batch_start_time else None
                    oldest_age = (now - batch_start_time[oldest_bid]) if oldest_bid is not None else 0.0
                    log(
                        f"[embed] heartbeat: {completed_items}/{n} ({pct:.2f}%) | "
                        f"{rate:.2f} items/s | pending_futures={pending} | oldest_batch_age={oldest_age:.1f}s"
                    )
                    continue

                for fut in done_futs:
                    bid = in_flight.pop(fut)
                    start, end = batches[bid]
                    batch_len = end - start

                    try:
                        vecs = fut.result()
                        arr = np.array(vecs, dtype=np.float32)
                        if arr.ndim != 2 or arr.shape[1] != embedding_dim or arr.shape[0] != batch_len:
                            raise RuntimeError(
                                f"Batch {bid} wrong shape: got {arr.shape}, expected ({batch_len},{embedding_dim})"
                            )

                        emb_mm[start:end, :] = arr
                        done[bid] = True
                        completed_items += batch_len
                        batches_since_checkpoint += 1
                        batch_start_time.pop(bid, None)

                    except Exception as e:
                        done[bid] = False
                        batch_start_time.pop(bid, None)
                        log(f"[embed] ERROR batch {bid+1}/{num_batches}: {type(e).__name__}: {e}")
                        retry_queue.append(bid)

                    if ((now - last_checkpoint_t) >= CHECKPOINT_EVERY_SECONDS) or (batches_since_checkpoint >= CHECKPOINT_EVERY_BATCHES):
                        log("[ckpt] Flushing embeddings + saving DONE + META ...")
                        emb_mm.flush()
                        np.save(DONE_NPY, done)
                        meta["status"] = "partial"
                        meta["completed_batches"] = int(done.sum())
                        meta["completed_items"] = int(completed_items)
                        meta["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                        atomic_write_json(META_JSON, meta)
                        last_checkpoint_t = now
                        batches_since_checkpoint = 0
                        log("[ckpt] Done.")

                    top_up()

        log("[embed] Final flush + finalize metadata...")
        emb_mm.flush()
        np.save(DONE_NPY, done)

        if not bool(done.all()):
            meta["status"] = "partial"
            meta["completed_batches"] = int(done.sum())
            meta["completed_items"] = int(sum((batches[i][1] - batches[i][0]) for i in range(num_batches) if done[i]))
            atomic_write_json(META_JSON, meta)
            raise RuntimeError(
                f"Embedding did not fully complete: {int(done.sum())}/{num_batches} batches done. Re-run to resume."
            )

        meta["status"] = "complete"
        meta["completed_batches"] = int(done.sum())
        meta["completed_items"] = n
        meta["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        atomic_write_json(META_JSON, meta)
        log(f"[embed] COMPLETE. embeddings file size: {human_bytes(os.path.getsize(EMB_NPY))}")

    if os.path.isfile(NORM_NPY):
        norms = np.load(NORM_NPY).astype(np.float32)
        if norms.shape[0] != n:
            log("[norm] Norm cache wrong length; recomputing.")
            norms = compute_and_cache_norms(EMB_NPY, NORM_NPY)
        else:
            log(f"[norm] Loaded norms: {norms.shape} ({human_bytes(os.path.getsize(NORM_NPY))})")
    else:
        norms = compute_and_cache_norms(EMB_NPY, NORM_NPY)

    return EMB_NPY, norms


# --------------------------
# Build / Cache leaf nodes + embeddings
# --------------------------

def build_leaf_cache() -> None:
    load_env_if_possible()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not found. Put it in .env as OPENAI_API_KEY=... or export it in your shell.")

    log(f"[config] OUTPUT_DIR = {OUTPUT_DIR}")
    log(f"[config] input_file  = {input_file}")
    log(f"[config] EMBED_MODEL = {EMBED_MODEL}, BATCH={EMBED_BATCH_SIZE}, WORKERS={MAX_WORKERS}, INFLIGHT={MAX_INFLIGHT}")
    log("")

    log(f"[load] Reading ontology JSON: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    criterion_root = find_criterion_root(data)
    if criterion_root is None:
        raise KeyError("Could not find CRITERION node in the input JSON.")

    log("[extract] Extracting leaf nodes (empty dicts)...")
    t0 = time.time()
    leaf_paths_tuples = iter_leaf_paths_iterative(criterion_root)
    leaf_paths_full = [path_tuple_to_text_full(p) for p in leaf_paths_tuples]
    leaf_paths_embed = [path_tuple_to_text_embed(p) for p in leaf_paths_tuples]
    leaf_paths_lex = [path_tuple_to_text_lex(p) for p in leaf_paths_tuples]
    log(f"[extract] Total CRITERION leaf nodes found: {len(leaf_paths_embed)} in {time.time()-t0:.2f}s")

    if leaf_paths_embed:
        log("[extract] Example leaf path (FULL):")
        log(f"         {leaf_paths_full[0]}")
        log("[extract] Example leaf path (EMBEDTEXT):")
        log(f"         {leaf_paths_embed[0]}")
        log("[extract] Example leaf path (LEXTEXT):")
        log(f"         {leaf_paths_lex[0]}")

    if not os.path.isfile(PATHS_FULL_JSON):
        log(f"[cache] Writing FULL leaf paths -> {PATHS_FULL_JSON}")
        atomic_write_json(PATHS_FULL_JSON, leaf_paths_full)
    if not os.path.isfile(PATHS_LEX_JSON):
        log(f"[cache] Writing LEX leaf texts -> {PATHS_LEX_JSON}")
        atomic_write_json(PATHS_LEX_JSON, leaf_paths_lex)

    c = Counter(leaf_paths_embed)
    dups = sum(1 for v in c.values() if v > 1)
    if dups > 0:
        log(f"[warn] {dups} EMBEDTEXT paths are duplicated after trimming.")

    log("[cache] Hashing EMBEDTEXT paths for cache validation...")
    paths_hash = stable_hash_of_paths(leaf_paths_embed)
    log(f"[cache] paths_hash={paths_hash[:16]}...")

    emb_path, norms = load_or_build_embeddings(leaf_paths_embed, paths_hash)
    log(f"[done] Embeddings ready: {emb_path} | norms={norms.shape} | {human_bytes(os.path.getsize(emb_path))}")


def main() -> None:
    build_leaf_cache()


if __name__ == "__main__":
    main()

# TODO: allow for optimal resolution pick --> right now only takes leaf nodes --> often too much depth
# TODO: then later implement similar logic for predictor search --> with ADDITIONAL relevance ranking logic of plausible relatedness estimates
