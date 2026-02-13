#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_score_barriers_for_coping_options.py  (dense coping→barrier relevance)

Goal
----
For each COPING OPTION leaf node (from coping_options_list.txt), compute a DENSE relevance score (0..1000)
for EVERY BARRIER leaf node (from barriers_list.txt).

Interpretation:
- Barrier = obstacle domain to behavior change (HAPA-aligned)
- Coping option = intervention component / strategy domain
- Score(barrier_id) answers: "How effectively would this coping option help mitigate this barrier domain?"
  - 1000 = best-in-class, direct + strong mitigation
  - 0 = no plausible mitigation

Key requirements implemented
----------------------------
- Loops through ALL coping options (one LLM call per coping option).
- Prompts include the full BARRIERS hierarchy raw text (verbatim from barriers_list.txt).
- Output is DENSE: must contain ALL barrier IDs; missing IDs trigger retry and ultimately fail that coping option.
- Uses JSON mode (text.format.type="json_object") for Structured Output compatibility.
- Strict local validation + normalization:
  - requires all barrier IDs present
  - drops unknown/extra IDs
  - coerces ints
  - clamps scores into [0,1000]
- Resume-capable caching: re-runs skip completed coping options.

Outputs (under out_base/<model>/)
---------------------------------
- coping_to_barrier_map.json          (coping option key -> dense scores + stats)
- coping_to_barrier_map_dense.csv     (coping options x barriers; dense matrix)
- coping_to_barrier_edges_long.csv    (tidy/long form; one row per coping-barrier edge)

Usage examples
--------------
# Test mode (first N coping options):
python 02_score_barriers_for_coping_options.py --test-mode --n-test 5

# Full run:
python 02_score_barriers_for_coping_options.py --full-run

Notes
-----
- One successful LLM output per coping option (no batching).
- Retries happen only on failures (including missing barrier IDs).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI


# ======================
# CONFIG DEFAULTS
# ======================

load_dotenv()

DEFAULT_MODEL = "gpt-5-nano"  # override via --model (e.g., gpt-5-nano, gpt-5, etc.)
DEFAULT_REASONING_EFFORT: Optional[str] = "low"  # None to omit

DEFAULT_BARRIERS_LIST_TXT = (
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/"
    "ontology_mappings/HAPA/coping_to_barrier/input_lists/barriers_list.txt"
)
DEFAULT_COPING_LIST_TXT = (
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/"
    "ontology_mappings/HAPA/coping_to_barrier/input_lists/coping_options_list.txt"
)

DEFAULT_OUT_BASE = (
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/"
    "ontology_mappings/HAPA/coping_to_barrier/results"
)

TEST_MODE_DEFAULT = False
N_TEST_DEFAULT = 5

MAX_WORKERS_DEFAULT = 120
MAX_INFLIGHT_LLM_CALLS_DEFAULT = 60

MAX_RETRIES_DEFAULT = 3
RETRY_BASE_SLEEP_DEFAULT = 1.0


# ======================
# DATA STRUCTURES
# ======================

@dataclass(frozen=True)
class LeafNode:
    id: int
    name: str
    full_path: str
    ancestors_path: str
    section: str


# ======================
# PARSING: GENERIC TREE TXT
# ======================

_SECTION_RE = re.compile(r"^\s*\[(?P<section>[^\]]+)\]\s*$")
_NODE_RE = re.compile(
    r"^(?P<prefix>[\s│]*)(?:[└├]─)\s*(?P<name>.+?)\s*"
    r"(?:\(\s*ID\s*:\s*(?P<id>\d+)\s*\))?\s*"
    r"(?:\(\s*path\s*:\s*'(?P<path>[^']*)'\s*\))?\s*$",
    re.UNICODE,
)

def _compute_depth(prefix: str) -> int:
    if not prefix:
        return 0
    p = prefix.replace("│", " ")
    p = p.replace("\t", "  ")
    return max(0, len(p) // 2)

def load_leaves_from_tree_txt(path: str) -> List[LeafNode]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"List not found: {path}")

    leaves: List[LeafNode] = []
    seen_ids: set[int] = set()

    current_section: str = ""
    stack: List[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            msec = _SECTION_RE.match(line)
            if msec:
                current_section = msec.group("section").strip()
                stack = []
                continue

            m = _NODE_RE.match(line)
            if not m:
                continue

            name = (m.group("name") or "").strip()
            id_str = m.group("id")
            path_str = (m.group("path") or "").strip()
            depth = _compute_depth(m.group("prefix") or "")

            if depth < len(stack):
                stack = stack[:depth]

            # Category node: no ID => update stack and continue
            if not id_str:
                if depth == len(stack):
                    stack.append(name)
                else:
                    if depth < len(stack):
                        stack[depth] = name
                    else:
                        stack.append(name)
                continue

            node_id = int(id_str)
            if node_id in seen_ids:
                raise ValueError(f"Duplicate leaf ID detected in {path}: {node_id}")
            seen_ids.add(node_id)

            if path_str:
                full_path = path_str
            else:
                parts: List[str] = []
                if current_section:
                    parts.append(f"[{current_section}]")
                parts.extend([s for s in stack if s])
                parts.append(name)
                full_path = " > ".join(parts).strip()

            ancestors_path = ""
            if " > " in full_path:
                ancestors_path = " > ".join(full_path.split(" > ")[:-1]).strip()

            leaves.append(
                LeafNode(
                    id=node_id,
                    name=name,
                    full_path=full_path,
                    ancestors_path=ancestors_path,
                    section=current_section,
                )
            )

    leaves.sort(key=lambda x: x.id)
    if not leaves:
        raise ValueError(f"No leaf IDs parsed from {path}. Expected lines like '└─ Name (ID:0)'.")
    return leaves

def load_raw_text(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read().rstrip()


# ======================
# PROMPTING
# ======================

def build_system_prompt() -> str:
    return (
        "You are a clinical-ontology mapping engineer for a personalization-focused digital intervention system.\n"
        "Task: given ONE COPING OPTION domain (an intervention strategy), score how effectively it mitigates EACH BARRIER domain.\n\n"

        "DEFINITIONS\n"
        "- Barrier domain: obstacle to behavior change / adherence (HAPA-aligned ; Health Action Process Approach).\n"
        "- Coping option domain: strategy or intervention component that could reduce or bypass barriers.\n\n"

        "SCORING (DENSE)\n"
        "You MUST output a relevance score for EVERY barrier ID.\n"
        "- 1000: best-in-class mitigation; direct and strong, high specificity + plausibility.\n"
        "- 700–999: strong mitigation.\n"
        "- 400–699: moderate mitigation.\n"
        "- 1–399: weak but non-zero mitigation.\n"
        "- 0: no plausible mitigation.\n"
        "Use integer unit steps only.\n\n"

        "INTERNAL REASONING (DO NOT OUTPUT)\n"
        "1) Interpret the coping option mechanisms (what it changes/targets).\n"
        "2) For each barrier domain, judge whether this coping option addresses it directly/indirectly.\n"
        "3) Score comparatively across all barriers for this coping option.\n\n"

        "OUTPUT FORMAT (STRICT JSON ONLY)\n"
        "Return ONLY a single JSON object with EXACTLY this top-level shape:\n"
        "{\n"
        "  \"scores\": {\n"
        "    \"<barrier_id>\": <integer 0..1000>,\n"
        "    ... (ALL barrier IDs present)\n"
        "  }\n"
        "}\n"
        "Rules:\n"
        "1) Include EVERY barrier ID exactly once as a string key.\n"
        "2) Values must be integers in [0,1000].\n"
        "3) Do NOT include any extra top-level keys besides \"scores\".\n"
        "4) Do NOT output any text outside the JSON object.\n"
    )

def build_user_prompt(
    coping: LeafNode,
    barriers_raw_text: str,
    barriers: List[LeafNode],
) -> str:
    # A compact “flat list” helps prevent missing IDs, while still including the required raw hierarchy verbatim.
    flat = "\n".join([f"- {b.id}\t{b.name}\t({b.full_path})" for b in barriers])

    return (
        "TARGET COPING OPTION (score barriers for THIS coping option)\n"
        f"- coping_id: {coping.id}\n"
        f"- coping_name: {coping.name}\n"
        f"- coping_full_path: {coping.full_path}\n\n"
        "BARRIERS HIERARCHY (RAW TEXT; PROVIDED VERBATIM)\n"
        f"{barriers_raw_text}\n\n"
        "BARRIERS FLAT LIST (IDs to score; you MUST include ALL of them)\n"
        f"{flat}\n\n"
        "OUTPUT FORMAT (STRICT JSON ONLY)\n"
        "{\n"
        "  \"scores\": {\n"
        "    \"0\": <int 0..1000>,\n"
        "    \"1\": <int 0..1000>,\n"
        "    ...\n"
        "  }\n"
        "}\n"
        "Constraints:\n"
        "- You MUST output scores for ALL barrier IDs listed above.\n"
        "- Integer unit steps only; values in [0,1000].\n"
        "- No extra keys besides \"scores\".\n"
        "- No text outside JSON.\n"
    )


# ======================
# JSON EXTRACTION + NORMALIZATION
# ======================

def _coerce_int(x: Any) -> Optional[int]:
    try:
        if isinstance(x, bool):
            return None
        return int(x)
    except Exception:
        return None

def _extract_json_object(text: str) -> Dict[str, Any]:
    t = (text or "").strip()
    if not t:
        raise ValueError("Empty model output (expected JSON object).")

    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = t[start : end + 1]
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj

    raise ValueError("Model output did not parse as a JSON object.")

def normalize_dense_scores(
    obj: Dict[str, Any],
    expected_barrier_ids: List[int],
) -> Tuple[Dict[str, int], Dict[str, Any]]:
    """
    Accepts either:
      {"scores": {"0": 123, ...}}
    or (tolerant) a direct dict {"0":123, ...}.

    Enforces:
    - ALL expected IDs present (missing => error)
    - Drops extra IDs
    - Coerces to ints; invalid => treated as missing => error
    - Clamps values to [0,1000]
    """
    top_keys = list(obj.keys())
    scores_any = obj.get("scores", None)
    if scores_any is None:
        # tolerant fallback: treat whole object as scores dict if it looks like it
        scores_any = obj

    if not isinstance(scores_any, dict):
        raise ValueError("Output JSON must contain 'scores' as an object/dict.")

    expected_set = set(expected_barrier_ids)

    parsed: Dict[int, int] = {}
    dropped_extra = 0
    invalid_entries = 0
    clamped = 0

    for k, v in scores_any.items():
        kid = _coerce_int(k)
        if kid is None:
            invalid_entries += 1
            continue
        if kid not in expected_set:
            dropped_extra += 1
            continue

        sval = _coerce_int(v)
        if sval is None:
            invalid_entries += 1
            continue

        if sval < 0:
            sval = 0
            clamped += 1
        elif sval > 1000:
            sval = 1000
            clamped += 1

        parsed[int(kid)] = int(sval)

    missing = sorted(list(expected_set.difference(parsed.keys())))
    if missing:
        raise ValueError(f"Missing barrier IDs in model output: {missing}")

    # Build ordered output with string keys
    out_scores: Dict[str, int] = {str(i): int(parsed[i]) for i in expected_barrier_ids}

    stats = {
        "top_level_keys": top_keys,
        "returned_count": int(len(scores_any)),
        "expected_count": int(len(expected_barrier_ids)),
        "dropped_extra_ids": int(dropped_extra),
        "invalid_entries": int(invalid_entries),
        "clamped_scores": int(clamped),
    }
    return out_scores, stats


# ======================
# RESULTS I/O (THREAD SAFE)
# ======================

_results_lock = threading.Lock()

def load_results_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}

    if os.path.getsize(path) == 0:
        os.rename(path, path + ".empty")
        print("[cache] Empty cache detected – renamed and starting fresh.")
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        os.rename(path, path + ".corrupt")
        print("[cache] Corrupt cache detected – renamed and starting fresh.")
        return {}

def atomic_write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ======================
# LLM CALLS (ONE CALL PER COPING OPTION)
# ======================

def call_llm_dense_scores(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    coping_key: str,
    expected_barrier_ids: List[int],
    reasoning_effort: Optional[str],
    semaphore: threading.Semaphore,
    max_retries: int,
    retry_base_sleep: float,
) -> Tuple[Dict[str, int], Dict[str, Any]]:
    """
    Returns (scores_out, stats)

    Uses JSON mode (json_object).
    Retries on any failure, including missing barrier IDs.
    """
    last_err: Optional[Exception] = None
    disable_reasoning = False
    disable_json_mode = False

    semaphore.acquire()
    try:
        for attempt in range(1, max_retries + 1):
            try:
                req: Dict[str, Any] = dict(
                    model=model,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    metadata={"coping_key": coping_key, "purpose": "coping_to_barrier_dense_scoring"},
                )

                if (reasoning_effort is not None) and (not disable_reasoning):
                    req["reasoning"] = {"effort": reasoning_effort}

                if not disable_json_mode:
                    req["text"] = {"format": {"type": "json_object"}}

                resp = client.responses.create(**req)
                raw_text = getattr(resp, "output_text", "") or ""
                obj = _extract_json_object(raw_text)

                scores_out, stats = normalize_dense_scores(obj, expected_barrier_ids=expected_barrier_ids)

                stats["json_mode_used"] = bool(not disable_json_mode)
                stats["reasoning_effort_used"] = (
                    reasoning_effort if (reasoning_effort is not None and not disable_reasoning) else None
                )
                return scores_out, stats

            except Exception as e:
                last_err = e
                msg = str(e)
                low = msg.lower()

                # Disable incompatible params if needed
                if ("unknown parameter" in low or "unknown_parameter" in low or "unexpected keyword" in low):
                    if "reasoning" in low:
                        disable_reasoning = True
                    if "text" in low or "format" in low or "json_object" in low:
                        disable_json_mode = True

                sleep_s = retry_base_sleep * (2 ** (attempt - 1))
                print(
                    f"[LLM][{coping_key}] attempt {attempt}/{max_retries} failed: {type(e).__name__}: {msg}\n"
                    f"  -> sleeping {sleep_s:.1f}s then retrying..."
                )
                time.sleep(sleep_s)

        raise RuntimeError(f"LLM call failed after {max_retries} attempts for {coping_key}: {last_err}")

    finally:
        semaphore.release()


# ======================
# PROCESSING
# ======================

def process_one_coping_option(
    coping: LeafNode,
    barriers_raw_text: str,
    barriers: List[LeafNode],
    client: OpenAI,
    results_path: str,
    results: Dict[str, Any],
    semaphore: threading.Semaphore,
    model: str,
    system_prompt: str,
    reasoning_effort: Optional[str],
    max_retries: int,
    retry_base_sleep: float,
) -> None:
    coping_key = f"co{coping.id}"

    # Fast skip
    with _results_lock:
        existing = results.get(coping_key, {}) or {}
        if existing.get("complete") is True:
            print(f"[SKIP] {coping_key} already complete.")
            return

    expected_barrier_ids = [b.id for b in barriers]

    print(f"\n[START] {coping_key} | coping='{coping.name}' | barriers={len(expected_barrier_ids)} | calls=1")
    user_prompt = build_user_prompt(coping=coping, barriers_raw_text=barriers_raw_text, barriers=barriers)

    try:
        scores_out, stats = call_llm_dense_scores(
            client=client,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            coping_key=coping_key,
            expected_barrier_ids=expected_barrier_ids,
            reasoning_effort=reasoning_effort,
            semaphore=semaphore,
            max_retries=max_retries,
            retry_base_sleep=retry_base_sleep,
        )

        payload_out = {
            "complete": True,
            "model": model,
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "coping_option": {
                "id": coping.id,
                "name": coping.name,
                "full_path": coping.full_path,
                "ancestors_path": coping.ancestors_path,
                "section": coping.section,
            },
            "scores": scores_out,  # dense: barrier_id(str) -> int score
            "stats": stats,
        }

        with _results_lock:
            results[coping_key] = payload_out
            atomic_write_json(results_path, results)

        print(
            f"[DONE] {coping_key} scores={len(scores_out)} "
            f"dropped_extra={stats.get('dropped_extra_ids', 0)} "
            f"invalid_entries={stats.get('invalid_entries', 0)} "
            f"clamped={stats.get('clamped_scores', 0)}"
        )

    except Exception as e:
        # Record failure but keep pipeline running
        err = f"{type(e).__name__}: {e}"
        print(f"[ERROR] {coping_key} failed: {err}")
        traceback.print_exc()

        with _results_lock:
            results[coping_key] = {
                "complete": False,
                "model": model,
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "coping_option": {
                    "id": coping.id,
                    "name": coping.name,
                    "full_path": coping.full_path,
                    "ancestors_path": coping.ancestors_path,
                    "section": coping.section,
                },
                "scores": {},
                "stats": {"error": err},
            }
            atomic_write_json(results_path, results)


# ======================
# CSV EXPORTS
# ======================

def write_dense_csv_from_results(
    results: Dict[str, Any],
    coping_options: List[LeafNode],
    barriers: List[LeafNode],
    out_csv: str,
) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    barrier_ids = [str(b.id) for b in barriers]
    headers = ["coping_id"] + barrier_ids

    coping_by_id = {c.id: c for c in coping_options}
    coping_ids_sorted = sorted(coping_by_id.keys())

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)

        for cid in coping_ids_sorted:
            key = f"co{cid}"
            payload = results.get(key, {}) or {}
            scores: Dict[str, Any] = payload.get("scores", {}) or {}

            row = [str(cid)]
            for bid in barrier_ids:
                v = scores.get(bid, "")
                try:
                    row.append(str(int(v)) if v != "" else "")
                except Exception:
                    row.append("")
            w.writerow(row)

    print(f"[ok] wrote dense CSV → {out_csv}")

def write_long_edges_csv(
    results: Dict[str, Any],
    coping_options: List[LeafNode],
    barriers: List[LeafNode],
    out_csv: str,
) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    coping_by_id = {c.id: c for c in coping_options}
    barrier_by_id = {b.id: b for b in barriers}

    fieldnames = [
        "coping_id",
        "coping_name",
        "coping_full_path",
        "barrier_id",
        "barrier_name",
        "barrier_full_path",
        "score",
    ]

    coping_ids_sorted = sorted(coping_by_id.keys())

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for cid in coping_ids_sorted:
            key = f"co{cid}"
            payload = results.get(key, {}) or {}
            scores: Dict[str, Any] = payload.get("scores", {}) or {}

            coping = coping_by_id[cid]
            for bid_str, score_any in scores.items():
                bid = _coerce_int(bid_str)
                if bid is None:
                    continue
                bnode = barrier_by_id.get(int(bid))
                if bnode is None:
                    continue
                try:
                    score = int(score_any)
                except Exception:
                    continue

                w.writerow(
                    {
                        "coping_id": cid,
                        "coping_name": coping.name,
                        "coping_full_path": coping.full_path,
                        "barrier_id": bnode.id,
                        "barrier_name": bnode.name,
                        "barrier_full_path": bnode.full_path,
                        "score": score,
                    }
                )

    print(f"[ok] wrote long edges CSV → {out_csv}")


# ======================
# MAIN
# ======================

def _resolve_out_dir(out_base: str, model: str) -> str:
    out_base = os.path.abspath(out_base)
    if os.path.basename(out_base) == model:
        return out_base
    return os.path.join(out_base, model)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", type=str, default=DEFAULT_REASONING_EFFORT)

    parser.add_argument("--barriers-list", type=str, default=DEFAULT_BARRIERS_LIST_TXT)
    parser.add_argument("--coping-list", type=str, default=DEFAULT_COPING_LIST_TXT)
    parser.add_argument("--out-base", type=str, default=DEFAULT_OUT_BASE)

    parser.add_argument("--test-mode", action="store_true", default=TEST_MODE_DEFAULT)
    parser.add_argument("--full-run", action="store_true", default=False)
    parser.add_argument("--n-test", type=int, default=N_TEST_DEFAULT)

    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS_DEFAULT)
    parser.add_argument("--max-inflight-llm", type=int, default=MAX_INFLIGHT_LLM_CALLS_DEFAULT)

    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES_DEFAULT)
    parser.add_argument("--retry-base-sleep", type=float, default=RETRY_BASE_SLEEP_DEFAULT)

    args = parser.parse_args()

    model = args.model.strip()
    reasoning_effort = (args.reasoning_effort.strip() if args.reasoning_effort is not None else None)
    if reasoning_effort == "" or reasoning_effort.lower() == "none":
        reasoning_effort = None

    test_mode = args.test_mode and not args.full_run
    n_test = int(args.n_test)
    max_workers = max(1, int(args.max_workers))
    max_inflight = max(1, int(args.max_inflight_llm))
    max_retries = int(args.max_retries)
    retry_base_sleep = float(args.retry_base_sleep)

    out_dir = _resolve_out_dir(args.out_base, model)
    out_json = os.path.join(out_dir, "coping_to_barrier_map.json")
    out_csv_dense = os.path.join(out_dir, "coping_to_barrier_map_dense.csv")
    out_csv_long = os.path.join(out_dir, "coping_to_barrier_edges_long.csv")

    print("=== CONFIG ===")
    print(f"model: {model}")
    print(f"reasoning_effort: {reasoning_effort}")
    print(f"barriers_list: {args.barriers_list}")
    print(f"coping_list: {args.coping_list}")
    print(f"out_dir: {out_dir}")
    print(f"out_json: {out_json}")
    print(f"out_csv_dense: {out_csv_dense}")
    print(f"out_csv_long: {out_csv_long}")
    print(f"test_mode: {test_mode} | n_test: {n_test}")
    print(f"max_workers: {max_workers}")
    print(f"max_inflight_llm_calls: {max_inflight}")
    print(f"max_retries: {max_retries} | retry_base_sleep: {retry_base_sleep}")
    print("================\n")

    os.makedirs(out_dir, exist_ok=True)

    barriers = load_leaves_from_tree_txt(args.barriers_list)
    barriers_raw_text = load_raw_text(args.barriers_list)
    print(f"[load] barriers parsed: {len(barriers)} (IDs: {barriers[0].id}..{barriers[-1].id})")

    # Sanity: ensure barrier IDs are exactly 0..max without gaps if that's intended
    barrier_ids = [b.id for b in barriers]
    if len(set(barrier_ids)) != len(barrier_ids):
        raise ValueError("Barrier IDs are not unique.")
    # We enforce “ALL IDs generated” against whatever is parsed (authoritative).

    coping_options = load_leaves_from_tree_txt(args.coping_list)
    print(f"[load] coping options parsed: {len(coping_options)} (IDs: {coping_options[0].id}..{coping_options[-1].id})")

    coping_run_list = coping_options
    if test_mode:
        coping_run_list = coping_options[: min(n_test, len(coping_options))]
        print(f"[test] limiting coping options to first {len(coping_run_list)} IDs: {[c.id for c in coping_run_list]}")

    results = load_results_json(out_json)
    if not isinstance(results, dict):
        print("[warn] existing results JSON is not a dict; resetting.")
        results = {}
    print(f"[cache] existing coping options in results JSON: {len(results)}")

    client = OpenAI()
    semaphore = threading.Semaphore(max_inflight)
    system_prompt = build_system_prompt()

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import itertools

    total = len(coping_run_list)
    started_at = time.time()
    done_counter = 0
    done_lock = threading.Lock()

    def _worker(coping: LeafNode) -> int:
        nonlocal done_counter
        try:
            process_one_coping_option(
                coping=coping,
                barriers_raw_text=barriers_raw_text,
                barriers=barriers,
                client=client,
                results_path=out_json,
                results=results,
                semaphore=semaphore,
                model=model,
                system_prompt=system_prompt,
                reasoning_effort=reasoning_effort,
                max_retries=max_retries,
                retry_base_sleep=retry_base_sleep,
            )
        finally:
            with done_lock:
                done_counter += 1
                elapsed = time.time() - started_at
                print(f"[progress] coping_done={done_counter}/{total} | elapsed={elapsed/60:.1f} min")
        return coping.id

    print("\n=== RUN ===")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        it = iter(coping_run_list)
        active: Dict[Any, int] = {}

        for c in itertools.islice(it, max_workers):
            fut = ex.submit(_worker, c)
            active[fut] = c.id

        while active:
            for fut in as_completed(list(active.keys())):
                _ = fut.result()
                active.pop(fut, None)
                try:
                    nxt = next(it)
                except StopIteration:
                    continue
                nfut = ex.submit(_worker, nxt)
                active[nfut] = nxt.id

    # Reload results from disk (latest)
    results = load_results_json(out_json)

    print("\n=== WRITE CSVs ===")
    write_dense_csv_from_results(results=results, coping_options=coping_options, barriers=barriers, out_csv=out_csv_dense)
    write_long_edges_csv(results=results, coping_options=coping_options, barriers=barriers, out_csv=out_csv_long)

    print("\n=== DONE ===")
    print(f"[ok] results JSON → {out_json}")
    print(f"[ok] dense CSV   → {out_csv_dense}")
    print(f"[ok] long CSV    → {out_csv_long}")


if __name__ == "__main__":
    main()

#NOTE: currently running the full inference pipeline:
# - with GPT nano-version --> gpt-5
# - with low resolution predictor ontology (i.e., of target level '3') --> 4
# - with LOW reasoning efforts --> high
