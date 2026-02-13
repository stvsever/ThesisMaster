#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_compute_semantic_similarity_matrix.py

Goal:
1) Read leaf-path lines from:
   /Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/cluster_criterions/results/01_all_leaf_nodes_with_paths.txt

2) Compute OpenAI embeddings (with caching to avoid redundant computation) using a ThreadPoolExecutor
   (max_workers=50) and store an "embeddings DB" at:
   /Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/cluster_criterions/results/02_embeddings_db

   This directory will contain:
   - cache.sqlite                 (persistent cache)
   - items.jsonl                  (index->text->hash mapping)
   - embeddings.npy               (embeddings array aligned to input order)
   - meta.json                    (model, dims, counts)

3) Compute a cosine similarity matrix WITHOUT holding the full matrix in RAM:
   - compute one row-block at a time
   - stream-write to CSV directly (with header row + first column labels)
   - store on external SSD:
     /Volumes/FileLord/Projects/MASTERPROEF/large_files/03_similarity_matrix.csv

IMPORTANT:
- The output CSV includes:
  * First row: empty cell + column labels (full path strings)
  * First column: row label (full path string)
  * Remaining cells: cosine similarity values formatted to 6 decimals

Requirements:
- pip install openai numpy python-dotenv
- Put OPENAI_API_KEY in:
  /Users/stijnvanseveren/PythonProjects/MASTERPROEF/.env
  like: OPENAI_API_KEY="sk-..."
Optionally:
- export OPENAI_EMBED_MODEL="text-embedding-3-small"
"""

import hashlib
import json
import os
import random
import sqlite3
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# =========================
# LOAD ENV
# =========================
load_dotenv("/Users/stijnvanseveren/PythonProjects/MASTERPROEF/.env", override=True)

# =========================
# Paths (as specified)
# =========================

INPUT_TXT = Path(
    "/utils/official/cluster_criterions/results/01_all_leaf_nodes_with_paths.txt"
)

# Fallback (in case the file is still named as in previous script output)
FALLBACK_INPUT_TXT = Path(
    "/utils/official/cluster_criterions/results/all_leaf_nodes_with_paths.txt"
)

EMBED_DB_DIR = Path(
    "/utils/official/cluster_criterions/results/02_embeddings_db"
)

# NEW: external SSD output for huge CSV
SIM_MATRIX_CSV = Path(
    "/Volumes/FileLord/Projects/MASTERPROEF/large_files/03_similarity_matrix.csv"
)

# =========================
# Embedding config
# =========================

MAX_WORKERS = 50
BATCH_SIZE = 64

# official small embedding model
REQUESTED_MODEL_DEFAULT = "text-embedding-ada-small"
MODEL_ENV = os.getenv("OPENAI_EMBED_MODEL", REQUESTED_MODEL_DEFAULT)


def _normalize_model_name(raw: str) -> str: # skip this
    cleaned = (raw or "").strip()
    cleaned_no_spaces = cleaned.replace(" ", "")
    if cleaned_no_spaces.lower() in {
        "text-embedding-ada-small",
        "text-embedding-adasmall",
        "textembeddingadasmall",
        "text-embedding-ada_small",
    }:
        return "text-embedding-3-small"
    return cleaned


EMBED_MODEL = _normalize_model_name(MODEL_ENV)

# =========================
# Similarity computation config
# =========================

# Smaller block reduces RAM usage further (block_rows * n float32)
SIM_BLOCK_ROWS = 128

# CSV float formatting
CSV_FLOAT_FMT = "%.6f"

# Cosine stabilization epsilon (must match formula used)
EPS = 1e-12

# =========================
# SQLite cache config
# =========================

CACHE_DB_PATH = EMBED_DB_DIR / "cache.sqlite"
ITEMS_JSONL = EMBED_DB_DIR / "items.jsonl"
EMBEDDINGS_NPY = EMBED_DB_DIR / "embeddings.npy"
META_JSON = EMBED_DB_DIR / "meta.json"

_thread_local = threading.local()


def _get_client() -> OpenAI:
    if not hasattr(_thread_local, "client"):
        _thread_local.client = OpenAI()
    return _thread_local.client


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _csv_escape(field: str) -> str:
    """
    CSV-escape a single field (RFC 4180-style):
    - If field contains comma/quote/newline, wrap in quotes and double quotes inside.
    """
    if any(ch in field for ch in [",", '"', "\n", "\r"]):
        return '"' + field.replace('"', '""') + '"'
    return field


def _read_input_lines() -> List[str]:
    if INPUT_TXT.exists():
        path = INPUT_TXT
    elif FALLBACK_INPUT_TXT.exists():
        print(f"[WARN] Input not found at:\n  {INPUT_TXT}\nUsing fallback:\n  {FALLBACK_INPUT_TXT}")
        path = FALLBACK_INPUT_TXT
    else:
        raise FileNotFoundError(
            f"Input file not found:\n  {INPUT_TXT}\n(and fallback not found:\n  {FALLBACK_INPUT_TXT}\n)"
        )

    lines: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                lines.append(s)

    return lines


def _init_cache_db() -> None:
    EMBED_DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CACHE_DB_PATH))
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                hash TEXT NOT NULL,
                model TEXT NOT NULL,
                dim  INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (hash, model)
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model);")
        conn.commit()
    finally:
        conn.close()


def _fetch_cached_embeddings(hashes: Sequence[str], model: str) -> Dict[str, np.ndarray]:
    cached: Dict[str, np.ndarray] = {}
    if not hashes:
        return cached

    conn = sqlite3.connect(str(CACHE_DB_PATH))
    try:
        CHUNK = 900
        for i in range(0, len(hashes), CHUNK):
            chunk = hashes[i : i + CHUNK]
            qmarks = ",".join(["?"] * len(chunk))
            rows = conn.execute(
                f"SELECT hash, dim, embedding FROM embeddings WHERE model = ? AND hash IN ({qmarks})",
                [model, *chunk],
            ).fetchall()

            for h, dim, blob in rows:
                vec = np.frombuffer(blob, dtype=np.float32)
                if int(dim) != vec.shape[0]:
                    continue
                cached[h] = vec
    finally:
        conn.close()

    return cached


def _insert_embeddings_into_cache(entries: List[Tuple[str, str, np.ndarray]], model: str) -> None:
    if not entries:
        return

    conn = sqlite3.connect(str(CACHE_DB_PATH))
    try:
        now = _utc_now_iso()
        rows = []
        for h, text, vec in entries:
            vec32 = np.asarray(vec, dtype=np.float32)
            rows.append((h, model, int(vec32.shape[0]), vec32.tobytes(), text, now))

        conn.executemany(
            """
            INSERT OR REPLACE INTO embeddings (hash, model, dim, embedding, text, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def _embed_batch_with_retries(texts: List[str], model: str, max_retries: int = 8) -> List[np.ndarray]:
    if not texts:
        return []

    for t in texts:
        if not t.strip():
            raise ValueError("Encountered empty string in embedding batch input.")

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            client = _get_client()
            resp = client.embeddings.create(model=model, input=texts)
            out = [np.array(item.embedding, dtype=np.float32) for item in resp.data]
            if len(out) != len(texts):
                raise RuntimeError(f"Embedding API returned {len(out)} vectors for {len(texts)} inputs.")
            return out
        except Exception as e:
            last_err = e
            sleep_s = min(60.0, (2 ** (attempt - 1)) * 0.5) + random.uniform(0.0, 0.25)
            print(
                f"[WARN] Embedding batch failed (attempt {attempt}/{max_retries}). "
                f"Sleeping {sleep_s:.2f}s. Error: {type(e).__name__}: {e}"
            )
            time.sleep(sleep_s)

    raise RuntimeError(f"Embedding batch failed after {max_retries} retries. Last error: {last_err}")


@dataclass
class BatchJob:
    batch_id: int
    indices: List[int]
    texts: List[str]
    hashes: List[str]


def _maybe_load_existing_embeddings(texts: List[str], model: str) -> Optional[np.ndarray]:
    if not EMBEDDINGS_NPY.exists() or not ITEMS_JSONL.exists():
        return None

    try:
        saved_hashes: List[str] = []
        with ITEMS_JSONL.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                saved_hashes.append(obj["hash"])

        current_hashes = [_sha256_text(t) for t in texts]
        if saved_hashes != current_hashes:
            print("[INFO] Existing embeddings cache does not match current input. Rebuilding.")
            return None

        emb = np.load(str(EMBEDDINGS_NPY))
        if emb.shape[0] != len(texts):
            print("[INFO] Existing embeddings row count mismatch. Rebuilding.")
            return None

        if META_JSON.exists():
            try:
                meta = json.loads(META_JSON.read_text(encoding="utf-8"))
                if meta.get("model") != model:
                    print(f"[INFO] Existing embeddings model differs. Rebuilding.")
                    return None
            except Exception:
                pass

        print(f"[OK] Reusing existing embeddings from:\n  {EMBEDDINGS_NPY}")
        return emb.astype(np.float32, copy=False)
    except Exception as e:
        print(f"[WARN] Failed to reuse existing embeddings; will recompute. Reason: {type(e).__name__}: {e}")
        return None


def _write_items_jsonl(texts: List[str], hashes: List[str]) -> None:
    with ITEMS_JSONL.open("w", encoding="utf-8") as f:
        for i, (t, h) in enumerate(zip(texts, hashes)):
            f.write(json.dumps({"index": i, "hash": h, "text": t}, ensure_ascii=False) + "\n")


def _write_meta(model: str, n_items: int, dim: int) -> None:
    meta = {
        "model": model,
        "n_items": n_items,
        "dim": dim,
        "created_at_utc": _utc_now_iso(),
        "input_file": str(INPUT_TXT),
        "embeddings_npy": str(EMBEDDINGS_NPY),
        "cache_db": str(CACHE_DB_PATH),
    }
    META_JSON.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def compute_embeddings() -> Tuple[List[str], List[str], np.ndarray]:
    texts = _read_input_lines()
    print(f"[INFO] Loaded {len(texts)} non-empty rows from input .txt")
    texts = [t.strip() for t in texts]

    if MODEL_ENV.strip() != EMBED_MODEL:
        print(f"[INFO] Model alias detected: '{MODEL_ENV}' -> using '{EMBED_MODEL}'")
    else:
        print(f"[INFO] Using embedding model: {EMBED_MODEL}")

    reused = _maybe_load_existing_embeddings(texts, EMBED_MODEL)
    if reused is not None:
        hashes = [_sha256_text(t) for t in texts]
        EMBED_DB_DIR.mkdir(parents=True, exist_ok=True)
        _write_items_jsonl(texts, hashes)
        _write_meta(EMBED_MODEL, len(texts), int(reused.shape[1]))
        return texts, hashes, reused

    _init_cache_db()

    hashes = [_sha256_text(t) for t in texts]

    print("[INFO] Checking embedding cache (sqlite)...")
    cached = _fetch_cached_embeddings(hashes, EMBED_MODEL)
    n_cached = len(cached)
    n_total = len(texts)
    n_missing = n_total - n_cached
    print(f"[INFO] Cache hit: {n_cached}/{n_total} ({(n_cached/n_total*100):.2f}%)")
    print(f"[INFO] To compute via API: {n_missing}")

    embeddings: Optional[np.ndarray] = None
    dim: Optional[int] = None

    if n_cached > 0:
        dim = next(iter(cached.values())).shape[0]
        embeddings = np.zeros((n_total, dim), dtype=np.float32)
        for i, h in enumerate(hashes):
            if h in cached:
                embeddings[i, :] = cached[h]

    missing_indices = [i for i, h in enumerate(hashes) if h not in cached]
    print(f"[INFO] Missing indices collected: {len(missing_indices)}")

    if not missing_indices:
        assert embeddings is not None and dim is not None
        EMBED_DB_DIR.mkdir(parents=True, exist_ok=True)
        _write_items_jsonl(texts, hashes)
        np.save(str(EMBEDDINGS_NPY), embeddings)
        _write_meta(EMBED_MODEL, n_total, dim)
        print(f"[OK] All embeddings were cached; saved embeddings.npy to:\n  {EMBEDDINGS_NPY}")
        return texts, hashes, embeddings

    jobs: List[BatchJob] = []
    batch_id = 0
    for start in range(0, len(missing_indices), BATCH_SIZE):
        idxs = missing_indices[start : start + BATCH_SIZE]
        jobs.append(
            BatchJob(
                batch_id=batch_id,
                indices=idxs,
                texts=[texts[i] for i in idxs],
                hashes=[hashes[i] for i in idxs],
            )
        )
        batch_id += 1

    print(f"[INFO] Created {len(jobs)} API batch jobs (batch_size={BATCH_SIZE}, max_workers={MAX_WORKERS})")

    processed_missing = 0
    staged_inserts: List[Tuple[str, str, np.ndarray]] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_job = {
            executor.submit(_embed_batch_with_retries, job.texts, EMBED_MODEL): job
            for job in jobs
        }

        total_batches = len(jobs)
        done_batches = 0

        for future in as_completed(future_to_job):
            job = future_to_job[future]
            done_batches += 1

            batch_vectors = future.result()
            if not batch_vectors:
                raise RuntimeError(f"Empty embedding result for batch_id={job.batch_id}")

            if embeddings is None:
                dim = int(batch_vectors[0].shape[0])
                embeddings = np.zeros((n_total, dim), dtype=np.float32)
                if n_cached > 0:
                    for i, h in enumerate(hashes):
                        if h in cached:
                            embeddings[i, :] = cached[h]

            assert embeddings is not None and dim is not None

            for vec in batch_vectors:
                if int(vec.shape[0]) != dim:
                    raise RuntimeError(f"Embedding dim mismatch: expected {dim} got {vec.shape[0]}")

            for i, h, t, vec in zip(job.indices, job.hashes, job.texts, batch_vectors):
                embeddings[i, :] = vec
                staged_inserts.append((h, t, vec))

            if len(staged_inserts) >= 512:
                _insert_embeddings_into_cache(staged_inserts, EMBED_MODEL)
                staged_inserts.clear()

            processed_missing += len(job.indices)
            remaining = n_missing - processed_missing
            pct = (processed_missing / n_missing * 100.0) if n_missing else 100.0

            print(
                f"[EMBED] batches_done={done_batches}/{total_batches} | "
                f"embedded={processed_missing}/{n_missing} ({pct:.2f}%) | remaining={remaining}"
            )

    if staged_inserts:
        _insert_embeddings_into_cache(staged_inserts, EMBED_MODEL)
        staged_inserts.clear()

    assert embeddings is not None and dim is not None
    EMBED_DB_DIR.mkdir(parents=True, exist_ok=True)
    _write_items_jsonl(texts, hashes)
    np.save(str(EMBEDDINGS_NPY), embeddings)
    _write_meta(EMBED_MODEL, n_total, dim)

    print(f"[OK] Saved embeddings.npy:\n  {EMBEDDINGS_NPY}")
    print(f"[OK] Updated cache DB:\n  {CACHE_DB_PATH}")
    print(f"[OK] Saved items mapping:\n  {ITEMS_JSONL}")
    print(f"[OK] Saved meta:\n  {META_JSON}")

    return texts, hashes, embeddings


def _estimate_csv_size_bytes(n: int) -> float:
    # Rough estimate: n*n floats, each ~8-12 bytes incl delimiter at 6 decimals.
    return float(n) * float(n) * 10.0


def compute_similarity_matrix_streaming(texts: List[str], embeddings_npy_path: Path) -> None:
    """
    Stream-write full cosine similarity matrix to CSV:
    - writes header row with column labels
    - then writes each row as: row_label, sim_0, sim_1, ..., sim_{n-1}
    - does NOT keep the full matrix in RAM
    """
    if not SIM_MATRIX_CSV.parent.exists():
        raise FileNotFoundError(
            f"Output directory does not exist:\n  {SIM_MATRIX_CSV.parent}\n"
            f"Create it (and ensure SSD is mounted) before running."
        )

    # Use memmap to avoid duplicating large arrays in RAM
    E = np.load(str(embeddings_npy_path), mmap_mode="r")  # shape (n, d)
    if E.ndim != 2:
        raise ValueError("embeddings.npy is not 2D.")

    n, d = E.shape
    print(f"[INFO] Similarity: n={n}, dim={d}")
    est_gb = _estimate_csv_size_bytes(n) / (1024 ** 3)
    print(f"[INFO] Rough estimated numeric CSV payload: ~{est_gb:.2f} GB (labels add more)")
    print(f"[INFO] Writing to external SSD:\n  {SIM_MATRIX_CSV}")
    print(f"[INFO] Row block size: {SIM_BLOCK_ROWS}")
    print("[INFO] Computing norms...")
    norms = np.linalg.norm(E, axis=1).astype(np.float32)  # (n,)
    norms_eps = norms + EPS

    # Prepare output file (streaming)
    print("[INFO] Writing CSV header row (this is huge)...")
    with SIM_MATRIX_CSV.open("w", encoding="utf-8", newline="") as out_f:
        # Header: first empty cell, then labels
        out_f.write("," + ",".join(_csv_escape(t) for t in texts) + "\n")
        out_f.flush()

        total_blocks = (n + SIM_BLOCK_ROWS - 1) // SIM_BLOCK_ROWS
        written_rows = 0

        print("[INFO] Starting blockwise cosine computation + streaming write...")
        for b in range(total_blocks):
            start = b * SIM_BLOCK_ROWS
            end = min(n, start + SIM_BLOCK_ROWS)
            block_rows = end - start

            print(
                f"[SIM] block={b+1}/{total_blocks} | computing rows {start}..{end-1} "
                f"(block_rows={block_rows}) | written_rows={written_rows}/{n}"
            )

            # Dot products: (block_rows, d) @ (d, n) -> (block_rows, n)
            # Keeps only the block in RAM.
            dot_block = (E[start:end, :] @ E.T).astype(np.float32)

            # Cosine denominator to match normalized formula:
            # sim = dot / ((||a||+EPS)*(||b||+EPS))
            denom = (norms_eps[start:end, None] * norms_eps[None, :]).astype(np.float32)
            sim_block = dot_block / denom

            # Numeric stability clamp
            np.clip(sim_block, -1.0, 1.0, out=sim_block)

            # Write numeric block using np.savetxt into a temp file, then prefix row labels while streaming to output.
            # This avoids python-level formatting loops over 24k floats per row.
            with tempfile.NamedTemporaryFile(
                mode="w+",
                encoding="utf-8",
                newline="",
                delete=False,
                dir=str(SIM_MATRIX_CSV.parent),
                suffix=".tmp_block_numeric.csv",
            ) as tmp_f:
                tmp_path = Path(tmp_f.name)
                np.savetxt(tmp_f, sim_block, delimiter=",", fmt=CSV_FLOAT_FMT)
                tmp_f.flush()
                tmp_f.seek(0)

                for i, line in enumerate(tmp_f):
                    row_label = _csv_escape(texts[start + i])
                    # line already ends with newline
                    out_f.write(row_label + "," + line)

            # Cleanup temp file
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception as e:
                print(f"[WARN] Could not delete temp file: {tmp_path} ({type(e).__name__}: {e})")

            written_rows += block_rows
            remaining = n - written_rows
            pct = (written_rows / n) * 100.0
            print(
                f"[SIM] block={b+1}/{total_blocks} | rows_written={written_rows}/{n} ({pct:.2f}%) | remaining={remaining}"
            )

            out_f.flush()

    print(f"[OK] Similarity matrix written with labels to:\n  {SIM_MATRIX_CSV}")


def main() -> None:
    print("==============================================")
    print("[START] 02_compute_semantic_similarity_matrix")
    print("==============================================")
    print(f"[PATH] input_txt:\n  {INPUT_TXT}")
    print(f"[PATH] embeddings_db_dir:\n  {EMBED_DB_DIR}")
    print(f"[PATH] similarity_csv (SSD):\n  {SIM_MATRIX_CSV}")
    print("----------------------------------------------")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.startswith("sk-"):
        print("[ERROR] OPENAI_API_KEY is not loaded (or invalid).")
        print("        Check .env at:")
        print("        /Users/stijnvanseveren/PythonProjects/MASTERPROEF/.env")
        sys.exit(1)

    print("[OK] OPENAI_API_KEY loaded.")
    print("----------------------------------------------")

    texts, hashes, embeddings = compute_embeddings()

    # Ensure we compute similarity from the saved .npy (memmap) to reduce RAM duplication
    if not EMBEDDINGS_NPY.exists():
        raise FileNotFoundError(f"Expected embeddings file not found:\n  {EMBEDDINGS_NPY}")

    # Free large in-memory array (best effort)
    try:
        del embeddings
    except Exception:
        pass

    print("----------------------------------------------")
    compute_similarity_matrix_streaming(texts=texts, embeddings_npy_path=EMBEDDINGS_NPY)
    print("==============================================")
    print("[DONE] Embeddings + labeled similarity CSV generated.")
    print("==============================================")


if __name__ == "__main__":
    main()
