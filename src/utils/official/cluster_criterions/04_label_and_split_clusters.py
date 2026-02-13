#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
04_label_and_split_clusters.py

Goal
----
Read semantic clusters from:
  /Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/cluster_criterions/results/04_semantically_clustered_items.json

For each cluster (or chunk if huge), call an LLM to:
  1) Produce ONE high-resolution canonical label if items are redundant / near-duplicates (action="keep")
  2) OR (rarely) split into 2+ distinct constructs (action="split"), each with its own high-resolution label

Output (COST + SIMPLE)
----------------------
- The main output file contains ONLY labels, one per line:
  /Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/cluster_criterions/results/05_GPT_labeled_split_items.txt

- A separate cache JSONL file is used for resume/skip:
  /Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/cluster_criterions/results/05_GPT_labeled_split_items.cache.jsonl

Deduplication
-------------
After processing, the labels TXT file is de-duplicated (preserve first occurrence).

Concurrency
-----------
ThreadPoolExecutor (default max_workers=10; set via --max-workers)

Testing
-------
Default: test mode ON (process first N clusters; default N=10)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from openai import BadRequestError


# =========================
# ENV
# =========================

ENV_PATH = Path("/Users/stijnvanseveren/PythonProjects/MASTERPROEF/.env")
load_dotenv(ENV_PATH, override=True)


# =========================
# PATHS
# =========================

INPUT_JSON = Path(
    "/utils/official/cluster_criterions/results/04_semantically_clustered_items.json"
)

# Main output: labels only (one per line)
OUTPUT_LABELS_TXT = Path(
    "/utils/official/cluster_criterions/results/05_GPT_labeled_split_items.txt"
)

# Resume cache (JSONL)
OUTPUT_CACHE_JSONL = Path(
    "/utils/official/cluster_criterions/results/05_GPT_labeled_split_items.cache.jsonl"
)


# =========================
# MODEL / CONFIG
# =========================

MODEL = os.getenv("OPENAI_LABEL_MODEL", "gpt-5-mini")

DEFAULT_MAX_WORKERS = int(os.getenv("LABEL_MAX_WORKERS", "10"))
MAX_ITEMS_PER_CALL = int(os.getenv("LABEL_MAX_ITEMS_PER_CALL", "120"))
MAX_OUTPUT_TOKENS = int(os.getenv("LABEL_MAX_OUTPUT_TOKENS", "250"))
MAX_RETRIES = int(os.getenv("LABEL_MAX_RETRIES", "6"))
PRINT_EVERY_N_TASKS = int(os.getenv("LABEL_PRINT_EVERY_N_TASKS", "10"))

# If you want stricter “prefer keep” behavior, increase this:
# (Used only in instructions; no API param)
PREFER_KEEP_BIAS = os.getenv("LABEL_PREFER_KEEP_BIAS", "high").strip().lower()


# =========================
# THREAD-LOCAL CLIENT + LOCKS
# =========================

_thread_local = threading.local()
_write_lock = threading.Lock()
_progress_lock = threading.Lock()


def _get_client() -> OpenAI:
    if not hasattr(_thread_local, "client"):
        _thread_local.client = OpenAI()
    return _thread_local.client


# =========================
# UTILS
# =========================

def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _cluster_hash(items: List[str]) -> str:
    return _sha256("\n".join(items))


def _load_clusters(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Input cluster JSON not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "clusters" not in data or not isinstance(data["clusters"], dict):
        raise TypeError("Unexpected cluster JSON format: expected top-level key 'clusters' as an object.")

    clusters: Dict[str, List[str]] = {}
    for cid, obj in data["clusters"].items():
        if isinstance(obj, dict) and isinstance(obj.get("items"), list):
            items = [str(x).strip() for x in obj["items"] if str(x).strip()]
            clusters[str(cid)] = items
    return clusters


def _load_already_processed(cache_jsonl: Path) -> Dict[str, str]:
    """
    Reads JSONL cache and returns mapping:
      cluster_key -> cluster_hash
    """
    processed: Dict[str, str] = {}
    if not cache_jsonl.exists():
        return processed

    with cache_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                key = obj.get("cluster_key")
                h = obj.get("cluster_hash")
                if isinstance(key, str) and isinstance(h, str):
                    processed[key] = h
            except Exception:
                continue
    return processed


def _extract_leaf_and_context(item: str) -> Tuple[str, str]:
    """
    For strings like:
      "<leaf> (context: parent < grandparent) — criterion"
    Return:
      (leaf, context_text)
    """
    leaf = item.strip()
    ctx = ""
    m = re.match(r"^(.*?)\s*\(context:\s*(.*?)\)\s*—\s*criterion\s*$", leaf)
    if m:
        leaf = m.group(1).strip()
        ctx = m.group(2).strip()
    return leaf, ctx


def _chunk_items(items: List[str], max_items: int) -> List[Tuple[str, List[str]]]:
    """
    Cost-oriented chunking:
    - If <= max_items: one call
    - Else: chunk sequentially
    """
    if len(items) <= max_items:
        return [("full", items)]

    out: List[Tuple[str, List[str]]] = []
    for start in range(0, len(items), max_items):
        chunk = items[start:start + max_items]
        out.append((f"chunk{start // max_items}", chunk))
    return out


def _dedupe_labels_file(path: Path) -> None:
    """
    De-duplicate label lines (preserve first occurrence).
    Uses casefold() for dedupe key but preserves original first-seen spelling.
    """
    if not path.exists():
        return

    with _write_lock:
        lines = path.read_text(encoding="utf-8").splitlines()

        seen = set()
        kept: List[str] = []
        for line in lines:
            lbl = line.strip()
            if not lbl:
                continue
            key = re.sub(r"\s+", " ", lbl).strip().casefold()
            if key in seen:
                continue
            seen.add(key)
            kept.append(lbl)

        path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")


# =========================
# PROMPT + SCHEMA (labels only)
# =========================

def _build_instructions() -> str:
    """
    Strongly biased toward KEEP unless clearly distinct constructs.
    Output is minimal: action + groups[{label}].
    """
    return (
        "You are a clinical ontology engineer working on PHOENIX: an OWL ontology used by a personalized "
        "optimization engine to select treatment targets (biopsychosocial variables) for mental well-being.\n\n"

        "Input:\n"
        "- You receive ONE semantic cluster of leaf-node strings (symptom/criterion-like labels with minimal context).\n"
        "- The cluster was created by embedding similarity; therefore, MOST clusters are already redundant.\n\n"

        "Goal:\n"
        "- Canonicalize redundancy: collapse non-distinct / near-duplicate items into a SINGLE atomic ontology label.\n"
        "- Split ONLY when the cluster contains clearly different constructs that should be separate OWL leaf nodes.\n\n"

        "Decision rule (IMPORTANT: strong keep bias):\n"
        "- action='keep' if items refer to the SAME underlying construct and differ only by wording, severity, timing, "
        "specifier, population, or trivial context.\n"
        "- action='split' ONLY if there are 2+ DISTINCT constructs (different symptom modality/entity) that would be "
        "separate targets in an optimization engine.\n\n"

        "What counts as DISTINCT (split triggers):\n"
        "- Different perceptual modality: auditory vs visual vs tactile vs olfactory hallucinations.\n"
        "- Different phenomenon type: hallucination vs delusion vs disorganization vs attention impairment vs agitation. ; if for instance distinct delusion type are present —> split them \n"
        "- Opposites or clinically distinct targets: insomnia vs hypersomnia; increased vs decreased sleep.\n\n"

        "Label construction (STRICT):\n"
        "- Produce ATOMIC, ontology-ready labels: one construct per label.\n"
        "- DO NOT enumerate symptoms or variants inside a label.\n"
        "- DO NOT use conjunctions: the label MUST NOT contain the words 'and' or 'or'.\n"
        "- DO NOT use these words: 'including', 'with', 'such as'.\n"
        "- DO NOT use punctuation/symbols: no parentheses, slashes, commas, semicolons, colons, quotes.\n"
        "- Keep labels short: 1–6 words (absolute max 10).\n"
        "- Prefer a clean noun phrase naming the optimization variable (e.g., 'Sleep-wake disturbance', "
        "'Auditory hallucinations', 'Disorganized thinking', 'Impaired attention').\n\n"
        
        "In many cases ; not splitting will be necessary — the actual redundancy clustering logic should already have worked fairly well."
        
        "All leaf nodes must reflect variables that in any (non-)clinical context should want to be optimized in some way.\n\n"
        
        "Output requirements:\n"
        "- Return ONLY JSON matching the schema.\n"
        "- If action='keep', output exactly ONE group with one label.\n"
        "- If action='split', output 2–6 groups (avoid over-splitting).\n"
    )


def _json_schema() -> Dict[str, Any]:
    """
    Structured Outputs schema via Responses API.
    Minimal: action + groups[{label}].
    """
    return {
        "name": "cluster_labeling_and_splitting",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "action": {"type": "string", "enum": ["keep", "split"]},
                "groups": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "label": {"type": "string"}
                            # "description": {"type": "string"},
                        },
                        "required": ["label"]
                    }
                }
            },
            "required": ["action", "groups"]
        },
        "strict": True
    }


def _build_user_input(cluster_key: str, items: List[str]) -> str:
    """
    Compact input: numbered list (no need for indices in output).
    """
    lines = [
        f"Cluster key: {cluster_key}",
        f"Number of items: {len(items)}",
        "",
        "Items:"
    ]
    for i, it in enumerate(items, start=1):
        leaf, ctx = _extract_leaf_and_context(it)
        if ctx:
            lines.append(f"{i}. {leaf} | context: {ctx}")
        else:
            lines.append(f"{i}. {leaf}")
    return "\n".join(lines)


# =========================
# GPT CALL (Structured Outputs; robust to unsupported params)
# =========================

def _extract_output_text(resp: Any) -> str:
    """
    Best-effort extraction of text from Responses API results.
    """
    raw = getattr(resp, "output_text", None)
    if isinstance(raw, str) and raw.strip():
        return raw

    # fallback: try common response shapes
    try:
        out = getattr(resp, "output", None)
        if isinstance(out, list) and out:
            c = out[0].get("content") if isinstance(out[0], dict) else getattr(out[0], "content", None)
            if isinstance(c, list) and c:
                t = c[0].get("text") if isinstance(c[0], dict) else getattr(c[0], "text", None)
                if isinstance(t, str) and t.strip():
                    return t
    except Exception:
        pass

    return ""


def _call_gpt(cluster_key: str, items: List[str]) -> Dict[str, Any]:
    client = _get_client()
    instructions = _build_instructions()
    user_input = _build_user_input(cluster_key, items)

    # Start minimal: omit temperature (some gpt-5 variants reject it). Include reasoning but auto-remove if unsupported.
    req: Dict[str, Any] = {
        "model": MODEL,
        "instructions": instructions,
        "input": user_input,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "reasoning": {"effort": "minimal"},
        "text": {
            "format": {
                "type": "json_schema",
                **_json_schema()
            }
        },
    }

    last_err: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.responses.create(**req)
            raw = _extract_output_text(resp)
            if not raw:
                raise ValueError("No output_text returned by Responses API.")

            obj = json.loads(raw)

            # normalize occasional list-wrapping (rare, but cheap to handle)
            if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
                obj = obj[0]

            if not isinstance(obj, dict) or "action" not in obj or "groups" not in obj:
                raise ValueError("Parsed JSON missing required keys ('action', 'groups').")

            return obj

        except BadRequestError as e:
            msg = str(e)
            # Remove unsupported params and retry immediately
            if "Unsupported parameter" in msg:
                if "reasoning" in msg and "reasoning" in req:
                    req.pop("reasoning", None)
                    print(f"[ADAPT] Removed unsupported 'reasoning' for model={MODEL}")
                    continue
            raise

        except Exception as e:
            last_err = e
            sleep_s = min(20.0, (2 ** (attempt - 1)) * 0.6) + random.uniform(0.0, 0.35)
            print(f"[WARN] GPT call failed attempt {attempt}/{MAX_RETRIES} cluster={cluster_key} sleep={sleep_s:.2f}s ({type(e).__name__}: {e})")
            time.sleep(sleep_s)

    raise RuntimeError(f"GPT call failed after {MAX_RETRIES} retries for cluster={cluster_key}. Last error: {last_err}")


def _post_validate_labels(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensures output has:
      - action in {keep, split}
      - groups: list of {label}
      - at least 1 label
    Enforces: if action=='keep' => exactly one group.
    """
    action = result.get("action")
    groups = result.get("groups")

    if not isinstance(groups, list) or not groups:
        # hard fallback
        return {"action": "keep", "groups": [{"label": "Unlabeled construct"}]}

    cleaned: List[Dict[str, str]] = []
    for g in groups:
        if not isinstance(g, dict):
            continue
        lbl = str(g.get("label", "")).strip()
        if lbl:
            cleaned.append({"label": lbl})

    if not cleaned:
        return {"action": "keep", "groups": [{"label": "Unlabeled construct"}]}

    if action not in ("keep", "split"):
        action = "split" if len(cleaned) >= 2 else "keep"

    # enforce: keep => one label
    if action == "keep" and len(cleaned) > 1:
        cleaned = cleaned[:1]

    # enforce: split => at least two groups if possible
    if action == "split" and len(cleaned) == 1:
        action = "keep"

    return {"action": action, "groups": cleaned}


# =========================
# WORKER + IO
# =========================

@dataclass
class TaskSpec:
    cluster_id: str
    cluster_key: str
    items: List[str]
    cluster_hash: str


def _append_labels_txt(path: Path, labels: List[str]) -> None:
    """
    Append labels to TXT file, one label per line.
    """
    if not labels:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with _write_lock:
        with path.open("a", encoding="utf-8") as f:
            for lbl in labels:
                s = lbl.strip()
                if s:
                    f.write(s + "\n")
            f.flush()


def _append_cache_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    """
    Append processing record for resume/skip.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False)
    with _write_lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()


def _process_one(task: TaskSpec) -> None:
    t0 = time.time()

    gpt_obj = _call_gpt(task.cluster_key, task.items)
    gpt_obj = _post_validate_labels(gpt_obj)

    labels = [g["label"] for g in gpt_obj["groups"] if g.get("label")]
    _append_labels_txt(OUTPUT_LABELS_TXT, labels)

    elapsed = time.time() - t0
    cache_record = {
        "timestamp_utc": _utc_now_iso(),
        "model": MODEL,
        "cluster_id": task.cluster_id,
        "cluster_key": task.cluster_key,
        "cluster_hash": task.cluster_hash,
        "n_items": len(task.items),
        "action": gpt_obj["action"],
        "labels": labels,
        "elapsed_sec": round(elapsed, 3),
    }
    _append_cache_jsonl(OUTPUT_CACHE_JSONL, cache_record)


# =========================
# MAIN
# =========================

def main() -> None:
    test_mode = False  # requested: default ON

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--test", default=test_mode, action="store_true", help="Process only a small number of clusters.")
    parser.add_argument("--test-n", type=int, default=5, help="How many clusters to process in --test mode.")
    parser.add_argument("--only-cluster-id", type=str, default=None, help="Process only a specific cluster id (e.g., c30).")
    parser.add_argument("--max-items-per-call", type=int, default=MAX_ITEMS_PER_CALL)
    args = parser.parse_args()

    print("==============================================")
    print("[START] 04_label_and_split_clusters")
    print("==============================================")
    print(f"[PATH] input_json:\n  {INPUT_JSON}")
    print(f"[PATH] output_labels_txt:\n  {OUTPUT_LABELS_TXT}")
    print(f"[PATH] cache_jsonl (resume):\n  {OUTPUT_CACHE_JSONL}")
    print("----------------------------------------------")
    print(f"[MODEL] {MODEL}")
    print(f"[CFG] max_workers={args.max_workers} | max_items_per_call={args.max_items_per_call} | max_out={MAX_OUTPUT_TOKENS}")
    print("----------------------------------------------")

    if not os.getenv("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY is not set. Ensure it's present in your environment or .env.")
        sys.exit(1)

    clusters = _load_clusters(INPUT_JSON)
    print(f"[OK] Loaded {len(clusters)} clusters from JSON")

    processed = _load_already_processed(OUTPUT_CACHE_JSONL)
    print(f"[OK] Loaded {len(processed)} processed entries from cache (resume)")

    cluster_ids = sorted(clusters.keys(), key=lambda x: (len(x), x))

    if args.only_cluster_id:
        if args.only_cluster_id not in clusters:
            print(f"[ERROR] Cluster id not found: {args.only_cluster_id}")
            sys.exit(1)
        cluster_ids = [args.only_cluster_id]

    if args.test:
        cluster_ids = cluster_ids[: args.test_n]
        print(f"[TEST] Enabled: processing only first {len(cluster_ids)} clusters")

    tasks: List[TaskSpec] = []
    for cid in cluster_ids:
        items = clusters[cid]
        if not items:
            continue

        chunks = _chunk_items(items, args.max_items_per_call)
        for sub_id, sub_items in chunks:
            key = cid if sub_id == "full" else f"{cid}::{sub_id}"
            h = _cluster_hash(sub_items)
            if processed.get(key) == h:
                continue
            tasks.append(TaskSpec(cluster_id=cid, cluster_key=key, items=sub_items, cluster_hash=h))

    total = len(tasks)
    print("----------------------------------------------")
    print(f"[PLAN] Tasks to run: {total}")
    if total == 0:
        print("[DONE] Nothing to do (all cached).")
        _dedupe_labels_file(OUTPUT_LABELS_TXT)
        return

    from concurrent.futures import ThreadPoolExecutor, as_completed

    done = 0
    failed = 0
    t_all = time.time()

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {ex.submit(_process_one, t): t for t in tasks}

        for fut in as_completed(futures):
            task = futures[fut]
            try:
                fut.result()
            except Exception as e:
                failed += 1
                print(f"[FAIL] {task.cluster_key} | {type(e).__name__}: {e}")
            finally:
                with _progress_lock:
                    done += 1
                    if done % PRINT_EVERY_N_TASKS == 0 or done == total:
                        elapsed = time.time() - t_all
                        print(f"[PROGRESS] {done}/{total} | failed={failed} | elapsed={elapsed/60.0:.2f} min")

    # Final dedupe of labels file
    _dedupe_labels_file(OUTPUT_LABELS_TXT)

    print("==============================================")
    print("[DONE] 04_label_and_split_clusters finished")
    print("==============================================")
    print(f"[SUMMARY] tasks={total} | failed={failed}")
    print(f"[OUT] labels_txt={OUTPUT_LABELS_TXT}")
    print(f"[OUT] cache_jsonl={OUTPUT_CACHE_JSONL}")


if __name__ == "__main__":
    main()

#TODO: ensure to skip the singletons --> later provide proper name for them with batching logic
#TODO: PERHAPS change logic --> is easier: just use the RAW context strings for operationalization...