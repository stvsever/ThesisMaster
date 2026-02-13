#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_map_predictors_to_criterions.py  (optimized + dashboard)

Goal
----
For each CRITERION cluster (from 04_semantically_clustered_items.json), score the relevance of ALL PREDICTOR leaf nodes
(from predictors_list.txt) for improving/optimizing that criterion cluster.

Important compatibility note (why your run crashed)
---------------------------------------------------
Your crash:
  Invalid schema for response_format 'ClusterMapping' ... array schema missing items

Cause:
- Pydantic v2 encodes Tuple schemas using `prefixItems` (JSON Schema 2020-12 style).
- The OpenAI `text.format: {type:"json_schema"}` validator for Structured Outputs rejected that schema.

Fix in this script:
- Uses **JSON mode** (`text.format.type="json_object"`) for maximum compatibility.
- Performs strict local validation + normalization (dedupe, drop unknown IDs, clamp scores, tag cleanup, sorting).

Outputs
-------
- predictor_to_criterion_map.json          (cluster -> domains + flattened scores + stats)
- predictor_to_criterion_map_dense.csv     (clusters x predictors; dense matrix)
- predictor_to_criterion_edges_long.csv    (tidy/long form; one row per edge)
- dashboard/ (index.html + bundle.json)    for interactive browsing

Usage examples
--------------
# Test mode (first N clusters), build dashboard:
python 02_map_predictors_to_criterions.py --test-mode --n-test 5 --build-dashboard

# Full run:
python 02_map_predictors_to_criterions.py --full-run

# Serve dashboard after a run:
python 02_map_predictors_to_criterions.py --serve-dashboard --port 8787

Notes
-----
- One successful LLM output per cluster (no batching). Retries happen only on failures.
- If a cluster output is suspiciously small (coverage low), it is flagged in stats and logs.
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
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI


# ======================
# CONFIG DEFAULTS
# ======================

load_dotenv()

DEFAULT_MODEL = "gpt-5-nano"  # override via --model (e.g., gpt-5-nano, gpt-5, etc.)
DEFAULT_REASONING_EFFORT: Optional[str] = "low"  # None to omit (other: 'low', 'high')

DEFAULT_CLUSTERS_JSON = (
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/"
    "cluster_criterions/results/04_semantically_clustered_items.json"
)
DEFAULT_PREDICTORS_LIST_TXT = (
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/"
    "ontology_mappings/CRITERION/predictor_to_criterion/input_lists/predictors_list.txt"
)

# Base output directory. A model subdir will be created under this.
DEFAULT_OUT_BASE = (
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/"
    "ontology_mappings/CRITERION/predictor_to_criterion/results"
)

TEST_MODE_DEFAULT = False
N_TEST_DEFAULT = 3

MAX_WORKERS_DEFAULT = 120
MAX_INFLIGHT_LLM_CALLS_DEFAULT = 60

MAX_RETRIES_DEFAULT = 3
RETRY_BASE_SLEEP_DEFAULT = 1.0

# Dashboard
DEFAULT_DASHBOARD_PORT = 8787


# ======================
# DATA STRUCTURES
# ======================

@dataclass(frozen=True)
class Predictor:
    id: int
    name: str
    full_path: str
    ancestors_path: str
    section: str


# ======================
# PARSING: PREDICTORS TREE
# ======================

_SECTION_RE = re.compile(r"^\s*\[(?P<section>[^\]]+)\]\s*$")
# Node line: prefix + branch marker + name + optional (ID:x) + optional (path:'...')  (path is optional)
_NODE_RE = re.compile(
    r"^(?P<prefix>[\s│]*)(?:[└├]─)\s*(?P<name>.+?)\s*"
    r"(?:\(\s*ID\s*:\s*(?P<id>\d+)\s*\))?\s*"
    r"(?:\(\s*path\s*:\s*'(?P<path>[^']*)'\s*\))?\s*$",
    re.UNICODE,
)


def _compute_depth(prefix: str) -> int:
    """
    Robust-ish indentation depth estimate for ASCII/Unicode trees.
    This is a best-effort heuristic; if indentation is absent, depth=0.
    """
    if not prefix:
        return 0
    # Replace vertical bars with spaces so we only measure whitespace width
    p = prefix.replace("│", " ")
    # Tabs -> 2 spaces
    p = p.replace("\t", "  ")
    # Count leading spaces; assume 2 spaces per depth step (common).
    return max(0, len(p) // 2)


def load_predictors_from_tree_txt(path: str) -> List[Predictor]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Predictors list not found: {path}")

    predictors: List[Predictor] = []
    seen_ids: set[int] = set()

    current_section: str = ""
    stack: List[str] = []  # category nodes by depth (within a section)

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

            # Maintain stack for categories
            if depth < len(stack):
                stack = stack[:depth]

            # If this is a category node (no ID): update stack and continue
            if not id_str:
                if depth == len(stack):
                    stack.append(name)
                else:
                    # Defensive: depth might be > len(stack) if indentation is irregular
                    if depth < len(stack):
                        stack[depth] = name
                    else:
                        stack.append(name)
                continue

            pid = int(id_str)
            if pid in seen_ids:
                continue
            seen_ids.add(pid)

            # Determine path:
            if path_str:
                full_path = path_str
            else:
                parts: List[str] = []
                if current_section:
                    parts.append(f"[{current_section}]")
                parts.extend([s for s in stack if s])
                parts.append(name)
                full_path = " > ".join(parts).strip()

            # Ancestors path: everything before the leaf name
            ancestors_path = ""
            if " > " in full_path:
                ancestors_path = " > ".join(full_path.split(" > ")[:-1]).strip()

            predictors.append(
                Predictor(
                    id=pid,
                    name=name,
                    full_path=full_path,
                    ancestors_path=ancestors_path,
                    section=current_section,
                )
            )

    predictors.sort(key=lambda p: p.id)
    if not predictors:
        raise ValueError(
            "No predictors parsed. Expected lines with unicode tree markers and '(ID:123)' or '(ID : 123)'."
        )
    return predictors


def build_predictor_candidates_block(predictors: List[Predictor]) -> str:
    """
    Compact representation: one line per predictor.
    Format: <id>\\t<full_path>
    """
    lines = [f"{p.id}\t{p.full_path}" for p in predictors]
    return "\n".join(lines)


# ======================
# PARSING: CLUSTERS
# ======================

def load_clusters(path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # expected wrapper: { "clusters": {...}, ... }
    if isinstance(raw, dict) and "clusters" in raw and isinstance(raw["clusters"], dict):
        return raw["clusters"]

    if isinstance(raw, dict):
        return raw

    raise ValueError("Clusters JSON must be a dict or a wrapper dict with key 'clusters'.")


def _cluster_sort_key(cid: str) -> Tuple[int, str]:
    m = re.match(r"^c(\d+)$", str(cid))
    if not m:
        return (10**18, str(cid))
    return (int(m.group(1)), str(cid))


# ======================
# PROMPTING (DO NOT SHORTEN)
# ======================

def build_system_prompt() -> str:
    return (
        "You are a clinical-ontology knowledge engineer and causal relevance scorer.\n"
        "Your job: score how relevant each SOLUTION DOMAIN (predictor) is for improving the CRITERION CLUSTER — "
        "the target CRITERION cluster (symptoms, impairments, risks, or patient-valued outcomes).\n\n"

        "INTERNAL REASONING PROCEDURE (MANDATORY, DO NOT OUTPUT):\n"
        "You MUST internally follow this pipeline before producing scores. Do NOT output any of these steps.\n"
        "A) PARSE & SUMMARIZE THE TARGET CLUSTER\n"
        "   1) Extract dominant FUNCTIONAL DEFICIT DOMAINS present in the cluster items.\n"
        "   2) Extract dominant ETIOLOGY / CONTEXT patterns if provided.\n"
        "   3) Convert this into a short internal MECHANISM PROFILE: ranked mechanisms/systems likely driving deficits.\n\n"

        "B) DEFINE EXPECTED RELEVANCE DOMAINS (INTERNAL ONLY)\n"
        "   Based on the mechanism profile, internally define a small set of EXPECTED relevance domains/mechanistic buckets\n"
        "   (e.g., executive control networks, sleep/circadian, mood/anxiety drivers, metabolic/cardiovascular risk,\n"
        "   sensory/perception, social/ADL supports, adherence/engagement scaffolding, safety/risk management).\n\n"

        "C) SCREEN EACH PREDICTOR FOR CAUSAL RELEVANCE (HIGH-RECALL)\n"
        "   For EVERY predictor, internally decide if it has ANY plausible pathway to improve the cluster outcomes.\n"
        "   - If there is a plausible direct OR indirect link (even weak), KEEP it and assign a low score (1..200).\n"
        "   - Omit ONLY predictors that are truly irrelevant (no plausible link).\n"
        "   IMPORTANT: Do NOT artificially limit the number of kept predictors. If hundreds are plausibly linked, output hundreds.\n\n"

        "D) SCORE USING A COMPARATIVE 1–1000 SCALE\n"
        "   Assign integer scores 1..1000 with unit steps only, calibrated comparatively across all kept predictors.\n"
        "   Higher scores require overall higher relevance: stronger causal proximity, larger expected effect size, higher specificity, stronger empirical/clinical support,\n"
        "   and feasibility when relevant.\n"
        "   Enforce separation (use full scale):\n"
        "   - 900–1000: best-in-class, mechanistically direct, high plausibility/support\n"
        "   - 700–899: strongly relevant\n"
        "   - 400–699: moderate relevance\n"
        "   - 100–399: low but non-zero\n"
        "   - 1–99: extremely low but non-zero; keep only if plausible\n\n"

        "E) CLUSTER OUTPUT BY RELEVANCE DOMAIN (OUTPUT REQUIRED)\n"
        "   Group kept predictors into relevance domains/mechanistic buckets.\n"
        "   Each domain must have:\n"
        "   - domain: a clean short name\n"
        "   - why: 1 short sentence explaining why this domain is relevant to the cluster\n"
        "   - edges: a list of compact tuples for predictors in this domain\n\n"

        "OUTPUT FORMAT (STRICT JSON ONLY)\n"
        "Return ONLY a single JSON object with exactly this top-level shape:\n"
        "{\n"
        "  \"domains\": [\n"
        "    {\n"
        "      \"domain\": \"...\",\n"
        "      \"why\": \"...\",\n"
        "      \"edges\": [ [\"<id>\", <score>], ... ]\n"
        "    },\n"
        "    ...\n"
        "  ]\n"
        "}\n"
        "Rules:\n"
        "1) IDs must be the correct predictor IDs; do NOT mix IDs.\n"
        "2) score must be integer 1..1000 (unit steps).\n"
        "3) Omit only truly irrelevant predictors; do NOT cap counts.\n"
        "4) Sort edges within each domain by score descending.\n"
        "5) Do NOT output any extra keys besides \"domains\".\n"
    )


def build_user_prompt(cluster_id: str, cluster_items: List[str], predictors_txt_raw: str) -> str:
    items_block = "\n".join([f"- {it}" for it in cluster_items])

    return (
        f"TARGET CRITERION CLUSTER\n"
        f"- cluster_id: {cluster_id}\n"
        f"- cluster_items:\n{items_block}\n\n"
        f"ALL SOLUTION CANDIDATES (predictors)\n"
        f"{predictors_txt_raw}\n\n"
        "OUTPUT FORMAT (STRICT JSON ONLY):\n"
        "{\n"
        "  \"domains\": [\n"
        "    {\n"
        "      \"domain\": \"<clean short name>\",\n"
        "      \"why\": \"<1 short sentence>\",\n"
        "      \"edges\": [ [\"<id>\", <score 1..1000>] , ... ]\n"
        "    },\n"
        "    ...\n"
        "  ]\n"
        "}\n"
        "Constraints:\n"
        "- Keep ONLY truly irrelevant predictors omitted; do NOT artificially limit number of edges.\n"
        "- Use integer scores with increments of 1 ONLY.\n"
        "- Do not include any explanatory text outside JSON.\n"
        "- Do not include any wrapper keys besides \"domains\".\n"
    )


# ======================
# OUTPUT NORMALIZATION (LOCAL VALIDATION)
# ======================

def _coerce_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def _clean_domain_name(s: Any) -> str:
    t = str(s or "").strip()
    t = re.sub(r"\s+", " ", t)
    return t[:80] if t else "Uncategorized"


def _clean_why(s: Any) -> str:
    t = str(s or "").strip()
    t = re.sub(r"\s+", " ", t)
    return t[:240]


def _extract_json_object(text: str) -> Dict[str, Any]:
    """
    Best-effort JSON object extraction.
    In JSON mode, output should already be a single JSON object; this is defensive.
    """
    t = (text or "").strip()
    if not t:
        raise ValueError("Empty model output (expected JSON object).")

    # Try strict parse first.
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Fallback: locate the outermost {...}
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = t[start : end + 1]
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj

    raise ValueError("Model output did not parse as a JSON object.")


def normalize_mapping_from_obj(
    obj: Dict[str, Any],
    valid_ids: set[int],
) -> Tuple[List[Dict[str, Any]], Dict[int, int], Dict[str, int]]:
    """
    Returns (domains_clean, flat_scores_best, stats)

    Accepts:
      {"domains":[{"domain":..., "why":..., "edges":[["id",score], ...]}, ...]}

    Defensive behavior:
    - Drops unknown IDs
    - Drops invalid edges/scores
    - Dedupes IDs across domains (keeps max score; ties keep first)
    - Sorts edges within each domain by score desc then id asc
    - Drops empty domains
    """
    dropped_unknown = 0
    dropped_invalid = 0
    duplicates_resolved = 0
    returned_edges = 0

    domains_any = obj.get("domains", [])
    if not isinstance(domains_any, list):
        raise ValueError("Output JSON must contain key 'domains' as a list.")

    domains_raw = [d for d in domains_any if isinstance(d, dict)]

    # best_map: pid -> (score, domain_index)
    best_map: Dict[int, Tuple[int, int]] = {}
    domain_meta: List[Tuple[str, str]] = []

    for di, d in enumerate(domains_raw):
        domain_name = _clean_domain_name(d.get("domain"))
        why = _clean_why(d.get("why"))
        domain_meta.append((domain_name, why))

        edges_any = d.get("edges", [])
        if not isinstance(edges_any, list):
            continue

        for e in edges_any:
            if not isinstance(e, (list, tuple)) or len(e) < 2:
                dropped_invalid += 1
                continue

            returned_edges += 1

            pid = _coerce_int(e[0])
            score = _coerce_int(e[1])

            if pid is None or score is None:
                dropped_invalid += 1
                continue
            if pid not in valid_ids:
                dropped_unknown += 1
                continue
            if score < 1 or score > 1000:
                dropped_invalid += 1
                continue

            prev = best_map.get(pid)
            if prev is None:
                best_map[pid] = (int(score), int(di))
            else:
                prev_score, prev_di = prev
                if int(score) > int(prev_score):
                    best_map[pid] = (int(score), int(di))
                duplicates_resolved += 1

    # rebuild domain edges
    domains_edges: List[List[Tuple[int, int]]] = [[] for _ in range(len(domain_meta))]
    for pid, (score, di) in best_map.items():
        if 0 <= di < len(domains_edges):
            domains_edges[di].append((pid, int(score)))
        else:
            domains_edges[0].append((pid, int(score)))

    domains_clean: List[Dict[str, Any]] = []
    for di, (domain_name, why) in enumerate(domain_meta):
        edges_sorted = sorted(domains_edges[di], key=lambda x: (-x[1], x[0]))
        if not edges_sorted:
            continue
        edges_out = [[str(pid), int(score)] for pid, score in edges_sorted]
        domains_clean.append({"domain": domain_name, "why": why, "edges": edges_out})

    flat_scores_best = {pid: score for pid, (score, _) in best_map.items()}

    stats = {
        "returned_domains": int(len(domains_raw)),
        "returned_edges": int(returned_edges),
        "kept_edges": int(len(flat_scores_best)),
        "dropped_unknown_ids": int(dropped_unknown),
        "dropped_invalid_entries": int(dropped_invalid),
        "duplicates_resolved": int(duplicates_resolved),
    }
    return domains_clean, flat_scores_best, stats


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
# LLM CALLS (ONE CALL PER CLUSTER)
# ======================

def call_llm_mapping(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    cluster_id: str,
    valid_ids: set[int],
    reasoning_effort: Optional[str],
    semaphore: threading.Semaphore,
    max_retries: int,
    retry_base_sleep: float,
) -> Tuple[List[Dict[str, Any]], Dict[int, int], Dict[str, int]]:
    """
    Returns (domains_clean, flat_scores_best, stats)

    Uses JSON mode for broad compatibility.
    Retries only on failures; no repair follow-up calls.
    """
    last_err: Optional[Exception] = None
    disable_reasoning = False
    disable_json_mode = False

    semaphore.acquire()
    try:
        for attempt in range(1, max_retries + 1):
            try:
                req_common: Dict[str, Any] = dict(
                    model=model,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    metadata={"cluster_id": cluster_id, "purpose": "predictor_to_criterion_mapping"},
                )

                if (reasoning_effort is not None) and (not disable_reasoning):
                    req_common["reasoning"] = {"effort": reasoning_effort}

                if not disable_json_mode:
                    req_common["text"] = {"format": {"type": "json_object"}}

                resp = client.responses.create(**req_common)
                raw_text = getattr(resp, "output_text", "") or ""
                obj = _extract_json_object(raw_text)

                domains_clean, flat_scores_best, stats = normalize_mapping_from_obj(obj, valid_ids=valid_ids)

                # Add call meta
                stats["json_mode_used"] = bool(not disable_json_mode)
                stats["reasoning_effort_used"] = (
                    reasoning_effort if (reasoning_effort is not None and not disable_reasoning) else None
                )
                return domains_clean, flat_scores_best, stats

            except Exception as e:
                last_err = e
                msg = str(e)
                low = msg.lower()

                # If API rejects reasoning, disable and retry
                if ("unknown parameter" in low or "unknown_parameter" in low or "unexpected keyword" in low):
                    if "reasoning" in low:
                        disable_reasoning = True
                    if "text" in low or "format" in low or "json_object" in low:
                        disable_json_mode = True

                sleep_s = retry_base_sleep * (2 ** (attempt - 1))
                print(
                    f"[LLM][{cluster_id}] attempt {attempt}/{max_retries} failed: {type(e).__name__}: {msg}\n"
                    f"  -> sleeping {sleep_s:.1f}s then retrying..."
                )
                time.sleep(sleep_s)

        raise RuntimeError(f"LLM call failed after {max_retries} attempts for cluster {cluster_id}: {last_err}")

    finally:
        semaphore.release()


# ======================
# PROCESSING
# ======================

def process_one_cluster(
    cluster_id: str,
    cluster_payload: Dict[str, Any],
    predictors: List[Predictor],
    predictors_txt_raw: str,
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
    # Fast skip (thread-safe read)
    with _results_lock:
        existing = results.get(cluster_id, {}) or {}
        if existing.get("complete") is True:
            print(f"[SKIP] cluster {cluster_id} already complete.")
            return

    items = cluster_payload.get("items", [])
    if not isinstance(items, list) or not items:
        print(f"[WARN] cluster {cluster_id} has no items. Marking complete with empty scores.")
        with _results_lock:
            results[cluster_id] = {
                "complete": True,
                "model": model,
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "domains": [],
                "scores": {},
                "stats": {
                    "returned_domains": 0,
                    "returned_edges": 0,
                    "kept_edges": 0,
                    "dropped_unknown_ids": 0,
                    "dropped_invalid_entries": 0,
                    "duplicates_resolved": 0,
                    "coverage_ratio": 0.0,
                },
            }
            atomic_write_json(results_path, results)
        return

    valid_ids = {p.id for p in predictors}
    n_predictors = len(predictors)

    print(f"\n[START] cluster {cluster_id} | items={len(items)} | predictors={n_predictors} | calls=1")
    user_prompt = build_user_prompt(cluster_id=cluster_id, cluster_items=items, predictors_txt_raw=predictors_txt_raw)

    domains_clean, flat_scores_best, stats = call_llm_mapping(
        client=client,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        cluster_id=cluster_id,
        valid_ids=valid_ids,
        reasoning_effort=reasoning_effort,
        semaphore=semaphore,
        max_retries=max_retries,
        retry_base_sleep=retry_base_sleep,
    )

    # Flatten as string keys for JSON compatibility
    scores_out: Dict[str, int] = {str(pid): int(score) for pid, score in flat_scores_best.items()}

    coverage_ratio = (len(scores_out) / float(n_predictors)) if n_predictors else 0.0
    stats["coverage_ratio"] = float(round(coverage_ratio, 6))

    # Warn if suspiciously small
    if len(scores_out) < 200:
        print(
            f"[WARN] cluster {cluster_id}: kept_edges={len(scores_out)} looks low (coverage={coverage_ratio:.4f})."
        )

    payload_out = {
        "complete": True,
        "model": model,
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "domains": domains_clean,
        "scores": scores_out,
        "stats": stats,
    }

    # Thread-safe write
    with _results_lock:
        results[cluster_id] = payload_out
        atomic_write_json(results_path, results)

    print(
        f"[DONE] cluster {cluster_id} kept={len(scores_out)} "
        f"coverage={coverage_ratio:.4f} "
        f"dropped_unknown={stats.get('dropped_unknown_ids', 0)} "
        f"dropped_invalid={stats.get('dropped_invalid_entries', 0)} "
        f"dupes_resolved={stats.get('duplicates_resolved', 0)}"
    )


# ======================
# CSV EXPORTS
# ======================

def write_dense_csv_from_results(results: Dict[str, Any], predictors: List[Predictor], out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # Predictor names removed from output; use IDs only
    headers = ["cluster_id"] + [str(p.id) for p in predictors]
    id_to_col = {str(p.id): idx + 1 for idx, p in enumerate(predictors)}

    cluster_ids = sorted([cid for cid in results.keys() if str(cid).startswith("c")], key=_cluster_sort_key)

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)

        for cid in cluster_ids:
            payload = results.get(cid, {}) or {}
            scores: Dict[str, Any] = payload.get("scores", {}) or {}

            row = [""] * (len(predictors) + 1)
            row[0] = cid

            for pid_str, score in scores.items():
                col = id_to_col.get(str(pid_str))
                if col is None:
                    continue
                try:
                    row[col] = str(int(score))
                except Exception:
                    continue

            w.writerow(row)

    print(f"[ok] wrote dense CSV → {out_csv}")


def write_long_edges_csv(
    results: Dict[str, Any],
    clusters: Dict[str, Dict[str, Any]],
    predictors: List[Predictor],
    out_csv: str,
) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    pred_by_id: Dict[int, Predictor] = {p.id: p for p in predictors}

    fieldnames = [
        "cluster_id",
        "domain",
        "domain_why",
        "predictor_id",
        "score",
        "predictor_full_path",
        "cluster_item_count",
    ]

    cluster_ids = sorted([cid for cid in results.keys() if str(cid).startswith("c")], key=_cluster_sort_key)

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for cid in cluster_ids:
            payload = results.get(cid, {}) or {}
            domains = payload.get("domains", []) or []
            items = (clusters.get(cid, {}) or {}).get("items", [])
            item_count = len(items) if isinstance(items, list) else 0

            for d in domains:
                domain = str(d.get("domain", "") or "")
                why = str(d.get("why", "") or "")
                edges = d.get("edges", []) or []
                if not isinstance(edges, list):
                    continue
                for e in edges:
                    if not isinstance(e, (list, tuple)) or len(e) < 2:
                        continue
                    try:
                        pid = int(e[0])
                        score = int(e[1])
                    except Exception:
                        continue

                    pred = pred_by_id.get(pid)
                    w.writerow(
                        {
                            "cluster_id": cid,
                            "domain": domain,
                            "domain_why": why,
                            "predictor_id": pid,
                            "score": score,
                            "predictor_full_path": (pred.full_path if pred else ""),
                            "cluster_item_count": item_count,
                        }
                    )

    print(f"[ok] wrote long edges CSV → {out_csv}")


# ======================
# DASHBOARD (HTML + JS, NO DEPENDENCIES)
# ======================

_DASHBOARD_INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Predictor → Criterion Mapping Dashboard</title>
  <style>
    :root {
      --bg: #0b0f14;
      --panel: #111826;
      --panel2: #0f1723;
      --text: #e6edf3;
      --muted: #93a4b8;
      --accent: #6ea8fe;
      --border: rgba(255,255,255,0.08);
      --danger: #ff6b6b;
    }
    html, body { height: 100%; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji";
      background: var(--bg);
      color: var(--text);
    }
    a { color: var(--accent); text-decoration: none; }
    .app {
      display: grid;
      grid-template-columns: 320px 1fr;
      height: 100vh;
    }
    .sidebar {
      border-right: 1px solid var(--border);
      background: var(--panel2);
      padding: 14px;
      overflow: auto;
    }
    .main {
      padding: 14px;
      overflow: auto;
    }
    .h1 {
      font-size: 16px;
      font-weight: 700;
      margin: 0 0 8px 0;
      letter-spacing: 0.2px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px;
      margin: 10px 0;
    }
    .row { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
    .control { display: flex; flex-direction: column; gap: 6px; }
    label { font-size: 12px; color: var(--muted); }
    select, input[type="text"], input[type="number"] {
      background: #0c1320;
      border: 1px solid var(--border);
      border-radius: 8px;
      color: var(--text);
      padding: 8px 10px;
      outline: none;
      width: 100%;
      box-sizing: border-box;
    }
    input[type="range"] { width: 220px; }
    .pill {
      display: inline-flex;
      gap: 6px;
      align-items: center;
      border: 1px solid var(--border);
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(255,255,255,0.03);
      font-size: 12px;
      color: var(--muted);
    }
    .domain-list { display: flex; flex-direction: column; gap: 6px; }
    .domain-btn {
      text-align: left;
      background: rgba(255,255,255,0.03);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 10px;
      border-radius: 10px;
      cursor: pointer;
      transition: transform 0.04s ease;
    }
    .domain-btn:hover { transform: translateY(-1px); border-color: rgba(110,168,254,0.35); }
    .domain-btn.active { border-color: rgba(110,168,254,0.6); box-shadow: 0 0 0 2px rgba(110,168,254,0.12) inset; }
    .small { font-size: 12px; color: var(--muted); }
    .items { margin: 0; padding-left: 18px; color: var(--text); }
    .items li { margin: 4px 0; color: var(--muted); }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      border-bottom: 1px solid var(--border);
      padding: 8px 8px;
      vertical-align: top;
    }
    th {
      text-align: left;
      color: var(--muted);
      font-weight: 600;
      cursor: pointer;
      user-select: none;
      white-space: nowrap;
    }
    tr:hover td { background: rgba(255,255,255,0.02); }
    .tag { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 12px; color: #cdd9e5; }
    .warn { color: var(--danger); font-weight: 700; }
    .muted { color: var(--muted); }
    .footer { margin-top: 10px; font-size: 12px; color: var(--muted); }
    .btn {
      background: rgba(110,168,254,0.12);
      border: 1px solid rgba(110,168,254,0.35);
      color: var(--text);
      padding: 8px 10px;
      border-radius: 10px;
      cursor: pointer;
    }
    .btn:hover { border-color: rgba(110,168,254,0.6); }
    .split { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    @media (max-width: 980px) {
      .app { grid-template-columns: 1fr; }
      .sidebar { border-right: none; border-bottom: 1px solid var(--border); }
      input[type="range"] { width: 100%; }
      .split { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
<div class="app">
  <div class="sidebar">
    <div class="h1">Mapping Dashboard</div>
    <div class="card">
      <div class="control">
        <label for="clusterSelect">Cluster</label>
        <select id="clusterSelect"></select>
      </div>
      <div class="row" style="margin-top:10px;">
        <span class="pill" id="modelPill">model: …</span>
        <span class="pill" id="statsPill">edges: …</span>
      </div>
      <div class="footer" id="generatedAt"></div>
    </div>

    <div class="card">
      <div class="h1">Domain filter</div>
      <div class="small">Click a domain to filter. Click again to clear.</div>
      <div style="height:10px;"></div>
      <div class="domain-list" id="domainList"></div>
    </div>
  </div>

  <div class="main">
    <div class="split">
      <div class="card">
        <div class="h1">Cluster items</div>
        <ul class="items" id="clusterItems"></ul>
      </div>

      <div class="card">
        <div class="h1">Filters</div>
        <div class="row" style="align-items:flex-end;">
          <div class="control" style="flex:1; min-width:260px;">
            <label for="searchBox">Search (path/domain/id)</label>
            <input id="searchBox" type="text" placeholder="e.g., sleep, antipsychotic, (ID) 407" />
          </div>
          <div class="control">
            <label for="minScore">Min score</label>
            <input id="minScore" type="number" min="1" max="1000" value="1" />
          </div>
          <div class="control">
            <label for="maxScore">Max score</label>
            <input id="maxScore" type="number" min="1" max="1000" value="1000" />
          </div>
          <div class="control">
            <button class="btn" id="exportBtn">Export filtered CSV</button>
          </div>
        </div>
        <div class="row" style="margin-top:10px; gap:12px;">
          <span class="pill">Sort: <span id="sortLabel">score ↓</span></span>
          <span class="pill">Domain: <span id="domainLabel" class="muted">all</span></span>
          <span class="pill">Rows: <span id="rowsLabel">0</span></span>
          <span class="pill" id="coveragePill">coverage: …</span>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="h1">Edges</div>
      <div class="small" id="whyHint"></div>
      <div style="height:10px;"></div>
      <table id="edgesTable">
        <thead>
          <tr>
            <th data-sort="score">Score</th>
            <th data-sort="id">ID</th>
            <th data-sort="path">Path</th>
            <th data-sort="domain">Domain</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
      <div class="footer">
        Tip: Use <span class="tag">domain</span> filter on the left to inspect mechanistic buckets quickly.
      </div>
    </div>
  </div>
</div>

<script>
(function(){
  const el = (id) => document.getElementById(id);

  let bundle = null;
  let activeDomain = null;
  let sortKey = "score";
  let sortDir = -1; // -1 desc, +1 asc

  function clusterKeysSorted(keys){
    return keys.slice().sort((a,b) => {
      const ma = /^c(\d+)$/.exec(a); const mb = /^c(\d+)$/.exec(b);
      if(ma && mb) return parseInt(ma[1],10) - parseInt(mb[1],10);
      return a.localeCompare(b);
    });
  }

  function setSortLabel(){
    const arrow = sortDir === -1 ? "↓" : "↑";
    el("sortLabel").textContent = `${sortKey} ${arrow}`;
  }

  function setDomainLabel(){
    el("domainLabel").textContent = activeDomain ? activeDomain : "all";
  }

  function normalizeStr(s){
    return (s || "").toString().toLowerCase();
  }

  function buildClusterEdges(clusterId){
    const c = bundle.clusters[clusterId];
    const preds = bundle.predictors;
    const out = [];
    const domains = c.domains || [];
    for(const d of domains){
      const domainName = d.domain || "";
      const why = d.why || "";
      const edges = d.edges || [];
      for(const e of edges){
        const pid = parseInt(e[0], 10);
        const score = parseInt(e[1], 10);
        const p = preds[String(pid)] || {};
        out.push({
          score, id: pid,
          path: p.full_path || "",
          domain: domainName,
          domainWhy: why
        });
      }
    }
    return out;
  }

  function buildDomainButtons(clusterId, edges){
    const c = bundle.clusters[clusterId];
    const list = el("domainList");
    list.innerHTML = "";

    const domains = c.domains || [];
    for(const d of domains){
      const domainName = d.domain || "Uncategorized";
      const count = (d.edges || []).length;

      const btn = document.createElement("button");
      btn.className = "domain-btn" + (activeDomain === domainName ? " active" : "");
      btn.innerHTML = `<div style="display:flex;justify-content:space-between;gap:10px;">
          <div>${domainName}</div><div class="small">${count}</div></div>
          <div class="small muted" style="margin-top:6px;line-height:1.3;">${d.why || ""}</div>`;
      btn.onclick = () => {
        if(activeDomain === domainName) activeDomain = null;
        else activeDomain = domainName;
        render(clusterId);
      };
      list.appendChild(btn);
    }

    // If activeDomain no longer exists, clear it.
    if(activeDomain && !domains.some(d => (d.domain || "") === activeDomain)){
      activeDomain = null;
    }
    setDomainLabel();
  }

  function applyFilters(edges){
    const q = normalizeStr(el("searchBox").value);
    const minScore = parseInt(el("minScore").value || "1", 10);
    const maxScore = parseInt(el("maxScore").value || "1000", 10);

    let filtered = edges.filter(e => {
      if(e.score < minScore || e.score > maxScore) return false;
      if(activeDomain && e.domain !== activeDomain) return false;
      if(!q) return true;
      const hay = [
        e.score, e.id, e.path, e.domain, e.domainWhy
      ].map(x => normalizeStr(x)).join(" | ");
      return hay.includes(q);
    });

    filtered.sort((a,b) => {
      let va = a[sortKey], vb = b[sortKey];
      if(sortKey === "path" || sortKey === "domain"){
        va = normalizeStr(va); vb = normalizeStr(vb);
        return sortDir * va.localeCompare(vb);
      }
      // numeric
      return sortDir * (va - vb);
    });

    return filtered;
  }

  function render(clusterId){
    const c = bundle.clusters[clusterId];

    el("modelPill").textContent = `model: ${c.model || bundle.model || "?"}`;
    const kept = (c.stats && c.stats.kept_edges) ? c.stats.kept_edges : 0;
    el("statsPill").textContent = `edges: ${kept}`;
    el("coveragePill").textContent = `coverage: ${(c.stats && c.stats.coverage_ratio != null) ? c.stats.coverage_ratio : "?"}`;

    // items
    const items = c.items || [];
    const ul = el("clusterItems");
    ul.innerHTML = "";
    for(const it of items){
      const li = document.createElement("li");
      li.textContent = it;
      ul.appendChild(li);
    }

    // edges
    const edges = buildClusterEdges(clusterId);
    buildDomainButtons(clusterId, edges);

    const filtered = applyFilters(edges);
    el("rowsLabel").textContent = filtered.length.toString();

    // show why hint if domain filter active
    if(activeDomain){
      const d = (c.domains || []).find(x => (x.domain || "") === activeDomain);
      el("whyHint").textContent = d ? (d.why || "") : "";
    } else {
      el("whyHint").textContent = "";
    }

    const tbody = el("edgesTable").querySelector("tbody");
    tbody.innerHTML = "";
    for(const e of filtered){
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${e.score}</td>
        <td>${e.id}</td>
        <td class="muted">${e.path || ""}</td>
        <td>${e.domain || ""}</td>
      `;
      tbody.appendChild(tr);
    }

    // update active button style
    [...document.querySelectorAll(".domain-btn")].forEach(btn => {
      if(activeDomain && btn.textContent.includes(activeDomain)) btn.classList.add("active");
      else btn.classList.remove("active");
    });

    setDomainLabel();
    setSortLabel();
  }

  function exportCSV(clusterId){
    const edges = buildClusterEdges(clusterId);
    const filtered = applyFilters(edges);

    const header = ["score","id","predictor_path","domain","domain_why"];
    const rows = [header.join(",")];
    for(const e of filtered){
      const cols = [
        e.score, e.id,
        `"${(e.path || "").replace(/"/g,'""')}"`,
        `"${(e.domain || "").replace(/"/g,'""')}"`,
        `"${(e.domainWhy || "").replace(/"/g,'""')}"`
      ];
      rows.push(cols.join(","));
    }

    const blob = new Blob([rows.join("\n")], {type: "text/csv;charset=utf-8"});
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `cluster_${clusterId}_filtered_edges.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  function wireTableSorting(clusterId){
    const ths = document.querySelectorAll("#edgesTable th[data-sort]");
    ths.forEach(th => {
      th.onclick = () => {
        const k = th.getAttribute("data-sort");
        if(sortKey === k) sortDir *= -1;
        else { sortKey = k; sortDir = (k === "score") ? -1 : 1; }
        render(clusterId);
      };
    });
  }

  async function init(){
    const resp = await fetch("bundle.json", {cache: "no-store"});
    bundle = await resp.json();

    el("generatedAt").textContent = `generated_at: ${bundle.generated_at || "?"}`;

    const clusterIds = clusterKeysSorted(Object.keys(bundle.clusters || {}));
    const select = el("clusterSelect");
    select.innerHTML = "";
    for(const cid of clusterIds){
      const opt = document.createElement("option");
      opt.value = cid; opt.textContent = cid;
      select.appendChild(opt);
    }

    const first = clusterIds[0];
    select.value = first;
    select.onchange = () => render(select.value);

    el("searchBox").oninput = () => render(select.value);
    el("minScore").oninput = () => render(select.value);
    el("maxScore").oninput = () => render(select.value);
    el("exportBtn").onclick = () => exportCSV(select.value);

    wireTableSorting(first);
    render(first);
  }

  init().catch(err => {
    console.error(err);
    alert("Failed to load dashboard data. If you opened index.html via file://, use --serve-dashboard instead.");
  });
})();
</script>
</body>
</html>
"""


def build_dashboard_bundle(
    out_dir: str,
    model: str,
    predictors: List[Predictor],
    clusters: Dict[str, Dict[str, Any]],
    results: Dict[str, Any],
) -> Dict[str, Any]:
    pred_meta = {
        str(p.id): {
            "id": p.id,
            "full_path": p.full_path,
            "ancestors_path": p.ancestors_path,
            "section": p.section,
        }
        for p in predictors
    }

    cluster_ids = sorted([cid for cid in results.keys() if str(cid).startswith("c")], key=_cluster_sort_key)

    clusters_out: Dict[str, Any] = {}
    for cid in cluster_ids:
        r = results.get(cid, {}) or {}
        c = clusters.get(cid, {}) or {}
        clusters_out[cid] = {
            "model": r.get("model", model),
            "items": c.get("items", []),
            "domains": r.get("domains", []),
            "stats": r.get("stats", {}),
        }

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "model": model,
        "predictors": pred_meta,
        "clusters": clusters_out,
    }


def write_dashboard(out_dir: str, bundle: Dict[str, Any]) -> str:
    dash_dir = os.path.join(out_dir, "dashboard")
    os.makedirs(dash_dir, exist_ok=True)

    index_path = os.path.join(dash_dir, "index.html")
    bundle_path = os.path.join(dash_dir, "bundle.json")

    with open(index_path, "w", encoding="utf-8") as f:
        f.write(_DASHBOARD_INDEX_HTML)

    with open(bundle_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False)

    print(f"[ok] dashboard written → {dash_dir}")
    return dash_dir


def serve_dashboard(dash_dir: str, port: int) -> None:
    # Serve the dashboard directory
    cwd = os.getcwd()
    os.chdir(dash_dir)
    try:
        handler = SimpleHTTPRequestHandler
        httpd = ThreadingHTTPServer(("127.0.0.1", int(port)), handler)
        url = f"http://127.0.0.1:{int(port)}/index.html"
        print(f"[serve] dashboard at {url}")
        try:
            webbrowser.open(url)
        except Exception:
            pass
        httpd.serve_forever()
    finally:
        os.chdir(cwd)


# ======================
# MAIN
# ======================

def _resolve_out_dir(out_base: str, model: str) -> str:
    """
    Avoids accidental duplicate nesting like .../results/gpt-5-nano/gpt-5-nano
    when out_base already ends with the model name.
    """
    out_base = os.path.abspath(out_base)
    if os.path.basename(out_base) == model:
        return out_base
    return os.path.join(out_base, model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", type=str, default=DEFAULT_REASONING_EFFORT)

    parser.add_argument("--clusters-json", type=str, default=DEFAULT_CLUSTERS_JSON)
    parser.add_argument("--predictors-list", type=str, default=DEFAULT_PREDICTORS_LIST_TXT)
    parser.add_argument("--out-base", type=str, default=DEFAULT_OUT_BASE)

    parser.add_argument("--test-mode", action="store_true", default=TEST_MODE_DEFAULT)
    parser.add_argument("--full-run", action="store_true", default=False)
    parser.add_argument("--n-test", type=int, default=N_TEST_DEFAULT)

    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS_DEFAULT)
    parser.add_argument("--max-inflight-llm", type=int, default=MAX_INFLIGHT_LLM_CALLS_DEFAULT)

    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES_DEFAULT)
    parser.add_argument("--retry-base-sleep", type=float, default=RETRY_BASE_SLEEP_DEFAULT)

    parser.add_argument("--build-dashboard", action="store_true", default=False)
    parser.add_argument("--serve-dashboard", action="store_true", default=False)
    parser.add_argument("--port", type=int, default=DEFAULT_DASHBOARD_PORT)

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
    out_json = os.path.join(out_dir, "predictor_to_criterion_map.json")
    out_csv_dense = os.path.join(out_dir, "predictor_to_criterion_map_dense.csv")
    out_csv_long = os.path.join(out_dir, "predictor_to_criterion_edges_long.csv")

    print("=== CONFIG ===")
    print(f"model: {model}")
    print(f"reasoning_effort: {reasoning_effort}")
    print(f"clusters_json: {args.clusters_json}")
    print(f"predictors_list: {args.predictors_list}")
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

    predictors = load_predictors_from_tree_txt(args.predictors_list)
    predictors_txt_raw = build_predictor_candidates_block(predictors)
    print(f"[load] predictors parsed: {len(predictors)} (IDs: {predictors[0].id}..{predictors[-1].id})")

    clusters = load_clusters(args.clusters_json)
    cluster_ids_all = sorted([k for k in clusters.keys() if str(k).startswith("c")], key=_cluster_sort_key)
    print(f"[load] clusters loaded: {len(cluster_ids_all)}")

    cluster_ids = cluster_ids_all
    if test_mode:
        cluster_ids = cluster_ids_all[: min(n_test, len(cluster_ids_all))]
        print(f"[test] limiting clusters to first {len(cluster_ids)} IDs: {cluster_ids}")

    results = load_results_json(out_json)
    if not isinstance(results, dict):
        print("[warn] existing results JSON is not a dict; resetting.")
        results = {}
    print(f"[cache] existing clusters in results JSON: {len(results)}")

    # OpenAI client
    client = OpenAI()
    semaphore = threading.Semaphore(max_inflight)
    system_prompt = build_system_prompt()

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import itertools

    total = len(cluster_ids)
    started_at = time.time()
    done_counter = 0
    done_lock = threading.Lock()

    def _worker(cid: str) -> str:
        nonlocal done_counter
        try:
            process_one_cluster(
                cluster_id=cid,
                cluster_payload=clusters[cid],
                predictors=predictors,
                predictors_txt_raw=predictors_txt_raw,
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
        except Exception as e:
            print(f"[ERROR] cluster {cid} failed: {type(e).__name__}: {e}")
            traceback.print_exc()
        finally:
            with done_lock:
                done_counter += 1
                elapsed = time.time() - started_at
                print(f"[progress] clusters_done={done_counter}/{total} | elapsed={elapsed/60:.1f} min")
        return cid

    print("\n=== RUN ===")

    # Submit at most ~max_workers tasks at once to avoid huge pending-futures memory on full runs.
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        cid_iter = iter(cluster_ids)
        active: Dict[Any, str] = {}

        for cid in itertools.islice(cid_iter, max_workers):
            fut = ex.submit(_worker, cid)
            active[fut] = cid

        while active:
            for fut in as_completed(list(active.keys())):
                _ = fut.result()
                active.pop(fut, None)
                try:
                    next_cid = next(cid_iter)
                except StopIteration:
                    continue
                nfut = ex.submit(_worker, next_cid)
                active[nfut] = next_cid

    # Reload results from disk to ensure we have the latest saved version
    results = load_results_json(out_json)

    print("\n=== WRITE CSVs ===")
    write_dense_csv_from_results(results=results, predictors=predictors, out_csv=out_csv_dense)
    write_long_edges_csv(results=results, clusters=clusters, predictors=predictors, out_csv=out_csv_long)

    if args.build_dashboard or args.serve_dashboard:
        print("\n=== DASHBOARD ===")
        bundle = build_dashboard_bundle(
            out_dir=out_dir,
            model=model,
            predictors=predictors,
            clusters=clusters,
            results=results,
        )
        dash_dir = write_dashboard(out_dir, bundle)
        if args.serve_dashboard:
            serve_dashboard(dash_dir, port=int(args.port))

    print("\n=== DONE ===")
    print(f"[ok] results JSON → {out_json}")
    print(f"[ok] dense CSV   → {out_csv_dense}")
    print(f"[ok] long CSV    → {out_csv_long}")
    if args.build_dashboard or args.serve_dashboard:
        print(f"[ok] dashboard   → {os.path.join(out_dir, 'dashboard')}")


if __name__ == "__main__":
    main()

# DO NOT RUN YET; optimize it cost-wise

#NOTE: currently running the full inference pipeline:
# - with GPT nano-version --> gpt-5
# - with low resolution of predictor ontology (i.e., of target level '3') --> 4
# - with 'LOW' reasoning efforts --> high
# - with low resolution of criterion ontology (i.e., 0.4 tau parameter during hierarchical clustering) --> 0.25

# TODO: fix directory saving issue
# TODO: ensure ALL of them are mapped (sequel script: 'clusters: 2528' --> there were 3K+ clusters...)

# TODO: make paper out of this ; including LLM-based mapping comparison (and embeddings) to actual feature importance estimates
# TODO: allow for a sub-type based mapping logic ; NOTE: still needs to be compatible for future pipelines --> so single estimation preference; but known sub-type specification
# TODO: use flex tier to save costs; argument: 'service_tier="flex"'

