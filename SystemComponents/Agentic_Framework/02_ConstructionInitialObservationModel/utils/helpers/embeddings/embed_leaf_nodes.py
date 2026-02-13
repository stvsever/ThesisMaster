#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
embed_leaf_nodes.py  (PREDICTOR + CRITERION)

Goal:
    Precompute and cache BOTH PREDICTOR and CRITERION leaf-node representations + embeddings
    for later zero-shot operationalization.

This script performs (for each ontology root: PREDICTOR, CRITERION):
  1) Load ontology JSON
  2) Extract ALL leaf paths (LEAF = empty dict {})
  3) Create:
       - FULL paths (human inspection)
       - EMBEDTEXT paths (leaf-focused text for embeddings)
       - LEXTEXT paths (lexicalized text used later for BM25/token/fuzzy)
  4) Build + cache embeddings with resumable batching:
       - embeddings .npy memmap
       - done-state .npy  (per-batch completion)
       - norms .npy (row norms for cosine sim)
       - meta .json with hash/model/dim/status

Directory layout (requested):
  <OUTPUT_DIR>/
    PREDICTORS/
      PREDICTOR_leaf_paths_*.json
      PREDICTOR_leaf_embeddings*.npy
      PREDICTOR_leaf_embeddings_meta.json
    CRITERIONS/
      CRITERION_leaf_paths_*.json
      CRITERION_leaf_embeddings*.npy
      CRITERION_leaf_embeddings_meta.json

Notes:
  - Leaf definition is STRICT: {} (empty dict).
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
import hashlib
import threading
import argparse
from typing import Any, Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from collections import Counter, deque

import numpy as np


# --------------------------
# CONFIG (EDIT THESE)
# --------------------------

input_file = "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/SystemComponents/PHOENIX_ontology/aggretated/01_raw/ontology_aggregated.json"

# Base embeddings directory (script will create two subdirs inside it)
OUTPUT_DIR = "/SystemComponents/Agentic_Framework/02_ConstructionInitialObservationModel/utils/helpers/embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Requested two subdirs
OUTPUT_PRED_DIR = os.path.join(OUTPUT_DIR, "PREDICTORS")
OUTPUT_CRIT_DIR = os.path.join(OUTPUT_DIR, "CRITERIONS")
os.makedirs(OUTPUT_PRED_DIR, exist_ok=True)
os.makedirs(OUTPUT_CRIT_DIR, exist_ok=True)

# Cache behavior
USE_CACHE_IF_AVAILABLE = True

# Embeddings
EMBED_MODEL = "text-embedding-3-small"
EMBED_BATCH_SIZE = 96

# Requested: ThreadPoolExecutor max 100 workers
MAX_WORKERS = 100

# Bounded inflight futures (keeps throughput high but prevents runaway submission)
MAX_INFLIGHT = 200

STATUS_EVERY_SECONDS = 8.0
CHECKPOINT_EVERY_SECONDS = 25.0
CHECKPOINT_EVERY_BATCHES = 20
REQUEST_TIMEOUT_SECONDS = 90.0

# Path shaping (for embedding text)
# Paths are extracted starting AFTER the root (children of PREDICTOR/CRITERION).
DROP_PREFIX_LEVELS_AFTER_ROOT = 0

LEAF_STAR = "*"
KEEP_LAST_N_NODES: Optional[int] = None  # e.g., 6 to only keep last 6 nodes

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
# Ontology root finding + leaf extraction
# --------------------------

def find_root(obj: Any, root_key: str) -> Optional[Dict[str, Any]]:
    """
    Finds the dict that contains the subtree at key == root_key.

    Common patterns supported:
      - {root_key: {...}}
      - {"ONTOLOGY": {root_key: {...}}}
      - root_key nested deeper in dict/list structure
    """
    if not isinstance(obj, dict):
        return None

    if root_key in obj and isinstance(obj[root_key], dict):
        return obj[root_key]

    if "ONTOLOGY" in obj and isinstance(obj["ONTOLOGY"], dict):
        onto = obj["ONTOLOGY"]
        if root_key in onto and isinstance(onto[root_key], dict):
            return onto[root_key]

    for _, v in obj.items():
        if isinstance(v, dict):
            found = find_root(v, root_key)
            if found is not None:
                return found
        elif isinstance(v, list):
            for item in v:
                found = find_root(item, root_key)
                if found is not None:
                    return found

    return None


def iter_leaf_paths_iterative(root: Any) -> List[Tuple[str, ...]]:
    """
    Leaf = empty dict {}.
    Returns full paths as tuples, starting AFTER the root (root children).
    """
    out: List[Tuple[str, ...]] = []
    stack: List[Tuple[Any, Tuple[str, ...]]] = [(root, tuple())]

    while stack:
        node, prefix = stack.pop()

        if isinstance(node, dict):
            if len(node) == 0:
                # strict leaf
                out.append(prefix)
            else:
                for k, v in node.items():
                    stack.append((v, prefix + (str(k),)))
        else:
            # If the structure ever contains non-dict at a terminal, treat as leaf at this prefix.
            out.append(prefix)

    return out


def path_tuple_to_text_full(path: Tuple[str, ...]) -> str:
    return " / ".join(path)


def path_tuple_to_text_embed(path: Tuple[str, ...]) -> str:
    """
    Leaf-focused embedding text:
      - optionally drops first DROP_PREFIX_LEVELS_AFTER_ROOT nodes
      - optionally keeps only last KEEP_LAST_N_NODES nodes
      - marks the leaf with a star prefix
    """
    p = list(path)

    # Trim early prefix levels (optional)
    if DROP_PREFIX_LEVELS_AFTER_ROOT > 0 and len(p) > DROP_PREFIX_LEVELS_AFTER_ROOT:
        p = p[DROP_PREFIX_LEVELS_AFTER_ROOT:]
    elif DROP_PREFIX_LEVELS_AFTER_ROOT > 0 and len(p) <= DROP_PREFIX_LEVELS_AFTER_ROOT:
        p = p[-1:] if p else p

    # Keep last N nodes (optional)
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
    # Keep compatibility across SDK versions (some accept timeout/max_retries args)
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
    # Keep embeddings stable: flatten newlines
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
            # jitter
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
    done_rows = 0
    for start in range(0, n, NORM_CHUNK_ROWS):
        end = min(n, start + NORM_CHUNK_ROWS)
        chunk = np.asarray(emb[start:end, :], dtype=np.float32)
        norms[start:end] = np.linalg.norm(chunk, axis=1).astype(np.float32)
        done_rows = end
        if (done_rows == n) or ((done_rows // NORM_CHUNK_ROWS) % 10 == 0):
            pct = (done_rows / n) * 100.0
            rate = done_rows / max(1e-9, (time.time() - t0))
            log(f"[norm] progress: {done_rows}/{n} ({pct:.2f}%) | {rate:.1f} rows/s")

    tmp = norms_path + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, norms)
    os.replace(tmp, norms_path)

    log(f"[norm] Saved norms -> {norms_path} ({human_bytes(os.path.getsize(norms_path))})")
    return norms


def load_or_build_embeddings(
    *,
    root_label: str,
    paths_embed: List[str],
    paths_hash: str,
    out_dir: str,
) -> Tuple[str, np.ndarray]:
    """
    Per-root embedding cache builder/resumer.
    Files are written into out_dir with root_label prefixes.
    """
    n = len(paths_embed)
    if n == 0:
        raise RuntimeError(f"No paths to embed for {root_label}.")

    PATHS_JSON      = os.path.join(out_dir, f"{root_label}_leaf_paths_EMBEDTEXT.json")
    PATHS_FULL_JSON = os.path.join(out_dir, f"{root_label}_leaf_paths_FULL.json")
    PATHS_LEX_JSON  = os.path.join(out_dir, f"{root_label}_leaf_paths_LEXTEXT.json")

    EMB_NPY   = os.path.join(out_dir, f"{root_label}_leaf_embeddings.npy")
    DONE_NPY  = os.path.join(out_dir, f"{root_label}_leaf_embeddings_done.npy")
    NORM_NPY  = os.path.join(out_dir, f"{root_label}_leaf_embedding_norms.npy")
    META_JSON = os.path.join(out_dir, f"{root_label}_leaf_embeddings_meta.json")

    # 1) If complete cache exists and validates, load norms and return
    if USE_CACHE_IF_AVAILABLE and os.path.isfile(EMB_NPY) and os.path.isfile(PATHS_JSON) and os.path.isfile(META_JSON):
        try:
            meta = json.load(open(META_JSON, "r", encoding="utf-8"))
            if (
                meta.get("paths_hash") == paths_hash
                and meta.get("model") == EMBED_MODEL
                and meta.get("num_paths") == n
                and meta.get("status") == "complete"
            ):
                log(f"[{root_label}][cache] Complete cache found. Verifying embed-paths + loading norms...")
                cached_paths = json.load(open(PATHS_JSON, "r", encoding="utf-8"))
                if cached_paths != paths_embed:
                    raise ValueError("Cached EMBEDTEXT paths mismatch (ordering/content).")

                if os.path.isfile(NORM_NPY):
                    norms = np.load(NORM_NPY).astype(np.float32)
                    if norms.shape[0] != n:
                        raise ValueError("Norm cache wrong length.")
                    log(f"[{root_label}][cache] Loaded norms: {norms.shape} ({human_bytes(os.path.getsize(NORM_NPY))})")
                else:
                    norms = compute_and_cache_norms(EMB_NPY, NORM_NPY)

                log(f"[{root_label}][cache] Ready.")
                return EMB_NPY, norms

        except Exception as e:
            log(f"[{root_label}][cache] Complete-cache load failed: {type(e).__name__}: {e}")
            log(f"[{root_label}][cache] Will attempt resume or rebuild...")

    # 2) Ensure we have embed-paths on disk (for validation/resume)
    if not os.path.isfile(PATHS_JSON):
        log(f"[{root_label}][cache] Writing embed-paths -> {PATHS_JSON}")
        atomic_write_json(PATHS_JSON, paths_embed)
        log(f"[{root_label}][cache] Embed-paths saved ({human_bytes(os.path.getsize(PATHS_JSON))})")

    # Prepare batch spans
    batches: List[Tuple[int, int]] = []
    for start in range(0, n, EMBED_BATCH_SIZE):
        end = min(n, start + EMBED_BATCH_SIZE)
        batches.append((start, end))
    num_batches = len(batches)
    log(f"[{root_label}][embed] Total paths={n}, batch_size={EMBED_BATCH_SIZE}, batches={num_batches}")

    # Check if we can resume
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
        log(f"[{root_label}][resume] Resuming embedding. dim={embedding_dim}, file={EMB_NPY}")
        done = np.load(DONE_NPY).astype(bool)
        if done.shape[0] != num_batches:
            log(f"[{root_label}][resume] DONE_NPY has wrong shape -> rebuild from scratch.")
            can_resume = False
    else:
        log(f"[{root_label}][embed] No usable resume state. Starting fresh embedding.")

    # 3) If not resuming, build fresh memmap and seed first batch to learn dim
    if not can_resume:
        for stale_path in [EMB_NPY, DONE_NPY, NORM_NPY]:
            if os.path.isfile(stale_path):
                log(f"[{root_label}][embed] Removing stale file: {stale_path}")
                try:
                    os.remove(stale_path)
                except Exception as e:
                    log(f"[{root_label}][embed] Warning: could not remove {stale_path}: {type(e).__name__}: {e}")

        log(f"[{root_label}][embed] Embedding first batch to determine vector dimension...")
        first_start, first_end = batches[0]
        first_vecs = embed_texts_with_retry(paths_embed[first_start:first_end], EMBED_MODEL)
        first_arr = np.array(first_vecs, dtype=np.float32)
        embedding_dim = int(first_arr.shape[1])
        log(f"[{root_label}][embed] Determined embedding_dim={embedding_dim}")

        log(f"[{root_label}][embed] Creating memmap .npy for embeddings -> {EMB_NPY}")
        emb_mm = np.lib.format.open_memmap(
            EMB_NPY, mode="w+", dtype=np.float32, shape=(n, embedding_dim)
        )
        emb_mm[first_start:first_end, :] = first_arr
        emb_mm.flush()
        log(f"[{root_label}][embed] Wrote batch 1/{num_batches} ({first_end-first_start} items).")

        done = np.zeros((num_batches,), dtype=bool)
        done[0] = True
        np.save(DONE_NPY, done)
        log(f"[{root_label}][embed] Saved done-state -> {DONE_NPY}")

        meta = {
            "status": "partial",
            "root": root_label,
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
        log(f"[{root_label}][embed] Wrote meta -> {META_JSON}")

    assert embedding_dim is not None

    # 4) Resume remaining batches (bounded inflight, checkpointing, retry queue)
    emb_mm = np.load(EMB_NPY, mmap_mode="r+")
    done = np.load(DONE_NPY).astype(bool)

    missing_batch_ids = [i for i in range(num_batches) if not done[i]]
    if not missing_batch_ids:
        log(f"[{root_label}][embed] All batches already complete. Skipping embedding.")
    else:
        log(f"[{root_label}][embed] Missing batches: {len(missing_batch_ids)}/{num_batches}")
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

        log(f"[{root_label}][embed] Executor max_workers={MAX_WORKERS}, MAX_INFLIGHT={MAX_INFLIGHT}")

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
                        f"[{root_label}][embed] heartbeat: {completed_items}/{n} ({pct:.2f}%) | "
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
                        log(f"[{root_label}][embed] ERROR batch {bid+1}/{num_batches}: {type(e).__name__}: {e}")
                        retry_queue.append(bid)

                    # checkpoint periodically
                    if ((now - last_checkpoint_t) >= CHECKPOINT_EVERY_SECONDS) or (batches_since_checkpoint >= CHECKPOINT_EVERY_BATCHES):
                        log(f"[{root_label}][ckpt] Flushing embeddings + saving DONE + META ...")
                        emb_mm.flush()
                        np.save(DONE_NPY, done)
                        meta["status"] = "partial"
                        meta["completed_batches"] = int(done.sum())
                        meta["completed_items"] = int(completed_items)
                        meta["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                        atomic_write_json(META_JSON, meta)
                        last_checkpoint_t = now
                        batches_since_checkpoint = 0
                        log(f"[{root_label}][ckpt] Done.")

                    top_up()

        log(f"[{root_label}][embed] Final flush + finalize metadata...")
        emb_mm.flush()
        np.save(DONE_NPY, done)

        if not bool(done.all()):
            meta["status"] = "partial"
            meta["completed_batches"] = int(done.sum())
            meta["completed_items"] = int(sum((batches[i][1] - batches[i][0]) for i in range(num_batches) if done[i]))
            atomic_write_json(META_JSON, meta)
            raise RuntimeError(
                f"[{root_label}] Embedding did not fully complete: {int(done.sum())}/{num_batches} batches done. Re-run to resume."
            )

        meta["status"] = "complete"
        meta["completed_batches"] = int(done.sum())
        meta["completed_items"] = n
        meta["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        atomic_write_json(META_JSON, meta)
        log(f"[{root_label}][embed] COMPLETE. embeddings file size: {human_bytes(os.path.getsize(EMB_NPY))}")

    # 5) Norms cache
    if os.path.isfile(NORM_NPY):
        norms = np.load(NORM_NPY).astype(np.float32)
        if norms.shape[0] != n:
            log(f"[{root_label}][norm] Norm cache wrong length; recomputing.")
            norms = compute_and_cache_norms(EMB_NPY, NORM_NPY)
        else:
            log(f"[{root_label}][norm] Loaded norms: {norms.shape} ({human_bytes(os.path.getsize(NORM_NPY))})")
    else:
        norms = compute_and_cache_norms(EMB_NPY, NORM_NPY)

    return EMB_NPY, norms


# --------------------------
# Build / Cache leaf nodes + embeddings (per root)
# --------------------------

def build_leaf_cache_for_root(*, root_key: str, out_dir: str, data: Any) -> None:
    PATHS_JSON      = os.path.join(out_dir, f"{root_key}_leaf_paths_EMBEDTEXT.json")
    PATHS_FULL_JSON = os.path.join(out_dir, f"{root_key}_leaf_paths_FULL.json")
    PATHS_LEX_JSON  = os.path.join(out_dir, f"{root_key}_leaf_paths_LEXTEXT.json")

    log(f"[{root_key}] ====== START ROOT ======")
    log(f"[{root_key}][config] out_dir={out_dir}")

    root = find_root(data, root_key)
    if root is None:
        raise KeyError(f"Could not find {root_key} node in the input JSON.")

    log(f"[{root_key}][load] Found root with {len(root)} top-level keys.")
    if len(root) > 0:
        sample_keys = list(root.keys())[:10]
        log(f"[{root_key}][load] Example top-level keys: {sample_keys}")

    log(f"[{root_key}][extract] Extracting leaf nodes (STRICT leaf = empty dict {{}}) ...")
    t0 = time.time()
    leaf_paths_tuples = iter_leaf_paths_iterative(root)

    leaf_paths_full  = [path_tuple_to_text_full(p) for p in leaf_paths_tuples]
    leaf_paths_embed = [path_tuple_to_text_embed(p) for p in leaf_paths_tuples]
    leaf_paths_lex   = [path_tuple_to_text_lex(p) for p in leaf_paths_tuples]

    log(f"[{root_key}][extract] Total leaf nodes found: {len(leaf_paths_embed)} in {time.time()-t0:.2f}s")

    if leaf_paths_embed:
        log(f"[{root_key}][extract] Example leaf path (FULL):     {leaf_paths_full[0]}")
        log(f"[{root_key}][extract] Example leaf path (EMBEDTEXT): {leaf_paths_embed[0]}")
        log(f"[{root_key}][extract] Example leaf path (LEXTEXT):   {leaf_paths_lex[0]}")

    # Persist FULL + LEX paths (useful for inspection/search)
    if not os.path.isfile(PATHS_FULL_JSON):
        log(f"[{root_key}][cache] Writing FULL leaf paths -> {PATHS_FULL_JSON}")
        atomic_write_json(PATHS_FULL_JSON, leaf_paths_full)
        log(f"[{root_key}][cache] FULL paths saved ({human_bytes(os.path.getsize(PATHS_FULL_JSON))})")
    else:
        log(f"[{root_key}][cache] FULL paths already exists -> {PATHS_FULL_JSON}")

    if not os.path.isfile(PATHS_LEX_JSON):
        log(f"[{root_key}][cache] Writing LEX leaf texts -> {PATHS_LEX_JSON}")
        atomic_write_json(PATHS_LEX_JSON, leaf_paths_lex)
        log(f"[{root_key}][cache] LEX texts saved ({human_bytes(os.path.getsize(PATHS_LEX_JSON))})")
    else:
        log(f"[{root_key}][cache] LEX texts already exists -> {PATHS_LEX_JSON}")

    # Duplicate warning after trimming
    c = Counter(leaf_paths_embed)
    dups = sum(1 for v in c.values() if v > 1)
    if dups > 0:
        log(f"[{root_key}][warn] {dups} EMBEDTEXT paths are duplicated after trimming. (May be OK; check trimming settings.)")

    log(f"[{root_key}][cache] Hashing EMBEDTEXT paths for cache validation...")
    paths_hash = stable_hash_of_paths(leaf_paths_embed)
    log(f"[{root_key}][cache] paths_hash={paths_hash[:16]}...")

    # Write EMBEDTEXT paths if missing (used for resume/cache validation)
    if not os.path.isfile(PATHS_JSON):
        log(f"[{root_key}][cache] Writing EMBEDTEXT leaf paths -> {PATHS_JSON}")
        atomic_write_json(PATHS_JSON, leaf_paths_embed)
        log(f"[{root_key}][cache] EMBEDTEXT paths saved ({human_bytes(os.path.getsize(PATHS_JSON))})")
    else:
        log(f"[{root_key}][cache] EMBEDTEXT paths already exists -> {PATHS_JSON}")

    emb_path, norms = load_or_build_embeddings(
        root_label=root_key,
        paths_embed=leaf_paths_embed,
        paths_hash=paths_hash,
        out_dir=out_dir,
    )
    log(f"[{root_key}][done] Embeddings ready: {emb_path} | norms={norms.shape} | {human_bytes(os.path.getsize(emb_path))}")
    log(f"[{root_key}] ====== END ROOT ======")


def build_all_leaf_caches(
    *,
    roots: List[str],
    input_path: str,
) -> None:
    load_env_if_possible()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not found. Put it in .env as OPENAI_API_KEY=... or export it in your shell.")

    log("========== START ==========")
    log(f"[config] OUTPUT_DIR = {OUTPUT_DIR}")
    log(f"[config] input_file  = {input_path}")
    log(f"[config] EMBED_MODEL = {EMBED_MODEL}, BATCH={EMBED_BATCH_SIZE}, WORKERS={MAX_WORKERS}, INFLIGHT={MAX_INFLIGHT}")
    log(f"[config] DROP_PREFIX_LEVELS_AFTER_ROOT = {DROP_PREFIX_LEVELS_AFTER_ROOT}")
    log(f"[config] KEEP_LAST_N_NODES = {KEEP_LAST_N_NODES}")
    log(f"[config] ROOTS = {roots}")
    log("")

    log(f"[load] Reading ontology JSON: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for root_key in roots:
        if root_key == "PREDICTOR":
            out_dir = OUTPUT_PRED_DIR
        elif root_key == "CRITERION":
            out_dir = OUTPUT_CRIT_DIR
        else:
            raise ValueError(f"Unsupported root_key={root_key}. Expected PREDICTOR or CRITERION.")

        build_leaf_cache_for_root(root_key=root_key, out_dir=out_dir, data=data)

    log("========== END ==========")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Embed PREDICTOR and/or CRITERION leaf nodes from ontology JSON.")
    p.add_argument("--input", default=input_file, help="Path to ontology JSON.")
    p.add_argument(
        "--roots",
        default="PREDICTOR,CRITERION",
        help="Comma-separated roots to process (default: PREDICTOR,CRITERION).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    roots = [r.strip().upper() for r in str(args.roots).split(",") if r.strip()]
    build_all_leaf_caches(roots=roots, input_path=str(args.input))


if __name__ == "__main__":
    main()

# TODO: allow for optimal resolution pick --> right now only takes leaf nodes --> often too much depth
# TODO: then later implement similar logic for predictor/criterion search --> with ADDITIONAL relevance ranking logic of plausible relatedness estimates
