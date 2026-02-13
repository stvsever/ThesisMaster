#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
map_profiles_to_predictors.py  (dense predictor→profile relevance)

Goal
----
For each PREDICTOR (aka "solution/intervention lever") leaf node, compute a DENSE relevance score (0..1000)
for EVERY PROFILE/PERSON domain leaf node.

Interpretation (for profile-aware recommendation)
-------------------------------------------------
- Profile domain: relatively stable person factors (demographics, resources, skills, preferences, access, etc.).
- Predictor domain: an intervention lever / solution component that can be recommended by a digital system.

Score(profile_id) answers:
"If the system KNOWS this PROFILE factor (trait/background/access/preference), how important is that knowledge
for deciding whether this PREDICTOR should be recommended (or how to tailor/deliver it), including safety,
feasibility, modality choice, personalization, equity/access, and expected adherence/effectiveness?"

Scoring guidance
----------------
- 1000 = critical determinant for recommendation (often gating/contraindicating or strongly required for feasibility/safety)
- 700–999 = strong relevance (frequently changes decision or strongly affects delivery/personalization)
- 400–699 = moderate relevance (meaningfully improves personalization/effectiveness but not strictly required)
- 1–399 = weak but non-zero relevance (occasionally useful; indirect/secondary)
- 0 = no plausible relevance to the recommendation decision or tailoring

Key requirements implemented
----------------------------
- Loops through ALL predictors (one LLM call per predictor leaf).
- Output is DENSE: must contain ALL profile IDs for each predictor; missing IDs trigger retry and ultimately fail that predictor.
- Uses JSON mode (text.format.type="json_object") when available.
- Strict local validation + minimal normalization only:
  - requires all profile IDs present
  - drops unknown/extra IDs
  - coerces ints
  - clamps into [0,1000] (no rescaling)
- Resume-capable caching: re-runs skip completed predictors.

Inputs (defaults)
-----------------
- Profile list:
  /Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/ontology_mappings/PREDICTOR/profile_to_predictor/input_lists/person_factors_list.txt
- Predictors list:
  /Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/ontology_mappings/PREDICTOR/profile_to_predictor/input_lists/predictors_list.txt

Outputs (under out_base/<model>/)
---------------------------------
- predictor_to_profile_map.json          (predictor key -> dense profile scores + stats)
- predictor_to_profile_map_dense.csv     (predictors x profile_ids; dense matrix)
- predictor_to_profile_edges_long.csv    (tidy/long form; one row per predictor-profile edge)

Usage examples
--------------
# Test mode (first N predictors):
python map_profiles_to_predictors.py --test-mode --n-test 20

# Full run:
python map_profiles_to_predictors.py --full-run

Optional prompt size reduction
------------------------------
If your predictor tree file is extremely large, you can use:
  --compact-prompts
This omits the full raw hierarchies and only provides:
- target predictor descriptor
- profile flat list (IDs/names/paths)

Notes
-----
- One successful LLM output per predictor (no batching).
- Retries happen only on failures (including missing profile IDs).
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

# Input lists (tree txt). You can override via CLI args.
DEFAULT_PROFILES_LIST_TXT = (
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/"
    "ontology_mappings/PREDICTOR/profile_to_predictor/input_lists/person_factors_list.txt"
)
DEFAULT_PREDICTORS_LIST_TXT = (
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/"
    "ontology_mappings/PREDICTOR/profile_to_predictor/input_lists/predictors_list.txt"
)

# Output base (requested)
DEFAULT_OUT_BASE = (
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/"
    "ontology_mappings/PREDICTOR/profile_to_predictor/results"
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
        "You are a clinical-ontology mapping engineer for a profile-aware digital mental health intervention system.\n"
        "Task: given ONE solution/intervention lever for optimizing (non-)clinical mental health issues, score how important each PERSON/PROFILE domain knowledge is for deciding\n"
        "whether that predictor/solution should be recommended (or how it should be tailored/delivered).\n\n"
        "DEFINITIONS\n"
        "- Predictor/solution domain: a solution/intervention lever or component that the system may recommend to a user for (non-)clinical mental health issues.\n"
        "- Profile domain: relatively stable person factors (demographics, resources, skills, preferences, access, background, etc.).\n\n"
        "RELEVANCE DEFINITION (THIS IS WHAT YOU SCORE)\n"
        "Score(profile_id) quantifies: If the system KNOWS this profile factor, how much would that knowledge matter for deciding\n"
        "whether to recommend the predictor (or how to tailor/deliver it)? Consider:\n"
        "  - Safety/contraindications and risk management\n"
        "  - Feasibility/constraints and required resources (access, time, money, transportation, device/internet)\n"
        "  - Modality fit (text/audio/video, self-guided vs coached)\n"
        "  - Communication/language/literacy considerations\n"
        "  - Personalization (tone, intensity, pacing, goal selection)\n"
        "  - Equity/access (avoiding recommendations that are unusable)\n"
        "  - Expected adherence and effectiveness given stable factors\n\n"
        "SCORING (DENSE)\n"
        "You MUST output a score for EVERY profile ID.\n"
        "- 1000: critical determinant (often gating/contraindicating or strongly required for feasibility/safety)\n"
        "- 700–999: strong relevance (frequently changes decision or strongly affects delivery/personalization)\n"
        "- 400–699: moderate relevance (meaningfully improves personalization/effectiveness but not strictly required)\n"
        "- 1–399: weak but non-zero relevance (occasionally useful; indirect/secondary)\n"
        "- 0: no plausible relevance to the recommendation decision or tailoring\n"
        "Use integer unit steps only.\n\n"
        "CALIBRATION\n"
        "Ensure scores are well-calibrated both ABSOLUTELY and RELATIVELY across profile domains for THIS predictor.\n"
        "Use very high scores only when the profile factor plausibly changes the recommendation decision in a substantial way.\n\n"
        "OUTPUT FORMAT (STRICT JSON ONLY)\n"
        "Return ONLY a single JSON object with EXACTLY this top-level shape:\n"
        "{\n"
        "  \"scores\": {\n"
        "    \"<profile_id>\": <integer 0..1000>,\n"
        "    ... (ALL profile IDs present)\n"
        "  }\n"
        "}\n"
        "Rules:\n"
        "1) Include EVERY profile ID exactly once as a string key.\n"
        "2) Values must be integers in [0,1000].\n"
        "3) Do NOT include any extra top-level keys besides \"scores\".\n"
        "4) Do NOT output any text outside the JSON object.\n"
    )

def build_user_prompt(
    predictor: LeafNode,
    profiles_raw_text: str,
    predictors_raw_text: str,
    profiles: List[LeafNode],
    compact_prompts: bool,
) -> str:
    flat_profiles = "\n".join([f"- {p.id}\t{p.name}\t({p.full_path})" for p in profiles])

    core = (
        "TARGET SOLUTION/PREDICTOR (score profile relevance for recommending THIS predictor)\n"
        f"- predictor_id: {predictor.id}\n"
        f"- predictor_name: {predictor.name}\n"
        f"- predictor_full_path: {predictor.full_path}\n\n"
        "PROFILE DOMAINS FLAT LIST (IDs to score; you MUST include ALL of them)\n"
        f"{flat_profiles}\n\n"
    )

    if not compact_prompts:
        core += (
            "PROFILE DOMAINS HIERARCHY (RAW TEXT; PROVIDED VERBATIM)\n"
            f"{profiles_raw_text}\n\n"
            # If you want, you can uncomment to include the full predictor hierarchy:
            # "PREDICTORS HIERARCHY (RAW TEXT; PROVIDED VERBATIM)\n"
            # f"{predictors_raw_text}\n\n"
        )

    core += (
        "OUTPUT FORMAT (STRICT JSON ONLY)\n"
        "{\n"
        "  \"scores\": {\n"
        "    \"0\": <int 0..1000>,\n"
        "    \"1\": <int 0..1000>,\n"
        "    ...\n"
        "  }\n"
        "}\n"
        "Constraints:\n"
        "- You MUST output scores for ALL profile IDs listed above.\n"
        "- Integer unit steps only; values in [0,1000].\n"
        "- No extra keys besides \"scores\".\n"
        "- No text outside JSON.\n"
    )
    return core


# ======================
# JSON EXTRACTION + VALIDATION
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
    expected_profile_ids: List[int],
) -> Tuple[Dict[str, int], Dict[str, Any]]:
    """
    Accepts:
      {"scores": {"0": 123, ...}}
    (tolerant) or a direct dict {"0":123, ...}.

    Enforces:
    - ALL expected IDs present (missing => error)
    - Drops extra IDs
    - Coerces to ints; invalid => treated as missing => error
    - Clamps into [0,1000] (no rescaling or other normalization)
    """
    top_keys = list(obj.keys())
    scores_any = obj.get("scores", None)
    if scores_any is None:
        scores_any = obj

    if not isinstance(scores_any, dict):
        raise ValueError("Output JSON must contain 'scores' as an object/dict.")

    expected_set = set(expected_profile_ids)

    parsed: Dict[int, int] = {}
    dropped_extra = 0
    invalid_entries = 0
    clamped = 0

    for k, v in scores_any.items():
        pid = _coerce_int(k)
        if pid is None:
            invalid_entries += 1
            continue
        if pid not in expected_set:
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

        parsed[int(pid)] = int(sval)

    missing = sorted(list(expected_set.difference(parsed.keys())))
    if missing:
        raise ValueError(f"Missing profile IDs in model output: {missing}")

    out_scores: Dict[str, int] = {str(i): int(parsed[i]) for i in expected_profile_ids}

    stats = {
        "top_level_keys": top_keys,
        "returned_count": int(len(scores_any)),
        "expected_count": int(len(expected_profile_ids)),
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
# LLM CALLS (ONE CALL PER PREDICTOR)
# ======================

def call_llm_dense_scores(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    predictor_key: str,
    expected_profile_ids: List[int],
    reasoning_effort: Optional[str],
    semaphore: threading.Semaphore,
    max_retries: int,
    retry_base_sleep: float,
) -> Tuple[Dict[str, int], Dict[str, Any]]:
    """
    Returns (scores_out, stats)

    Uses JSON mode (json_object) when supported.
    Retries on any failure, including missing profile IDs.
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
                    metadata={"predictor_key": predictor_key, "purpose": "predictor_to_profile_dense_scoring"},
                )

                if (reasoning_effort is not None) and (not disable_reasoning):
                    req["reasoning"] = {"effort": reasoning_effort}

                if not disable_json_mode:
                    req["text"] = {"format": {"type": "json_object"}}

                resp = client.responses.create(**req)
                raw_text = getattr(resp, "output_text", "") or ""
                obj = _extract_json_object(raw_text)

                scores_out, stats = normalize_dense_scores(obj, expected_profile_ids=expected_profile_ids)
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
                    f"[LLM][{predictor_key}] attempt {attempt}/{max_retries} failed: {type(e).__name__}: {msg}\n"
                    f"  -> sleeping {sleep_s:.1f}s then retrying..."
                )
                time.sleep(sleep_s)

        raise RuntimeError(f"LLM call failed after {max_retries} attempts for {predictor_key}: {last_err}")

    finally:
        semaphore.release()


# ======================
# PROCESSING
# ======================

def process_one_predictor(
    predictor: LeafNode,
    profiles_raw_text: str,
    predictors_raw_text: str,
    profiles: List[LeafNode],
    client: OpenAI,
    results_path: str,
    results: Dict[str, Any],
    semaphore: threading.Semaphore,
    model: str,
    system_prompt: str,
    reasoning_effort: Optional[str],
    max_retries: int,
    retry_base_sleep: float,
    compact_prompts: bool,
) -> None:
    predictor_key = f"p{predictor.id}"

    # Fast skip
    with _results_lock:
        existing = results.get(predictor_key, {}) or {}
        if existing.get("complete") is True:
            print(f"[SKIP] {predictor_key} already complete.")
            return

    expected_profile_ids = [p.id for p in profiles]

    print(f"\n[START] {predictor_key} | predictor='{predictor.name}' | profiles={len(expected_profile_ids)} | calls=1")

    user_prompt = build_user_prompt(
        predictor=predictor,
        profiles_raw_text=profiles_raw_text,
        predictors_raw_text=predictors_raw_text,
        profiles=profiles,
        compact_prompts=compact_prompts,
    )

    try:
        scores_out, stats = call_llm_dense_scores(
            client=client,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            predictor_key=predictor_key,
            expected_profile_ids=expected_profile_ids,
            reasoning_effort=reasoning_effort,
            semaphore=semaphore,
            max_retries=max_retries,
            retry_base_sleep=retry_base_sleep,
        )

        payload_out = {
            "complete": True,
            "model": model,
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "predictor": {
                "id": predictor.id,
                "name": predictor.name,
                "full_path": predictor.full_path,
                "ancestors_path": predictor.ancestors_path,
                "section": predictor.section,
            },
            "scores": scores_out,  # dense: profile_id(str) -> int score
            "stats": stats,
        }

        with _results_lock:
            results[predictor_key] = payload_out
            atomic_write_json(results_path, results)

        print(
            f"[DONE] {predictor_key} scores={len(scores_out)} "
            f"dropped_extra={stats.get('dropped_extra_ids', 0)} "
            f"invalid_entries={stats.get('invalid_entries', 0)} "
            f"clamped={stats.get('clamped_scores', 0)}"
        )

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"[ERROR] {predictor_key} failed: {err}")
        traceback.print_exc()

        with _results_lock:
            results[predictor_key] = {
                "complete": False,
                "model": model,
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "predictor": {
                    "id": predictor.id,
                    "name": predictor.name,
                    "full_path": predictor.full_path,
                    "ancestors_path": predictor.ancestors_path,
                    "section": predictor.section,
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
    predictors: List[LeafNode],
    profiles: List[LeafNode],
    out_csv: str,
) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    profile_ids = [str(p.id) for p in profiles]
    headers = ["predictor_id"] + profile_ids

    predictor_by_id = {p.id: p for p in predictors}
    predictor_ids_sorted = sorted(predictor_by_id.keys())

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)

        for pid in predictor_ids_sorted:
            key = f"p{pid}"
            payload = results.get(key, {}) or {}
            scores: Dict[str, Any] = payload.get("scores", {}) or {}

            row = [str(pid)]
            for prid in profile_ids:
                v = scores.get(prid, "")
                try:
                    row.append(str(int(v)) if v != "" else "")
                except Exception:
                    row.append("")
            w.writerow(row)

    print(f"[ok] wrote dense CSV → {out_csv}")

def write_long_edges_csv(
    results: Dict[str, Any],
    predictors: List[LeafNode],
    profiles: List[LeafNode],
    out_csv: str,
) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    predictor_by_id = {p.id: p for p in predictors}
    profile_by_id = {p.id: p for p in profiles}

    fieldnames = [
        "predictor_id",
        "predictor_name",
        "predictor_full_path",
        "profile_id",
        "profile_name",
        "profile_full_path",
        "score",
    ]

    predictor_ids_sorted = sorted(predictor_by_id.keys())

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for pid in predictor_ids_sorted:
            key = f"p{pid}"
            payload = results.get(key, {}) or {}
            scores: Dict[str, Any] = payload.get("scores", {}) or {}

            pnode = predictor_by_id[pid]
            for prid_str, score_any in scores.items():
                prid = _coerce_int(prid_str)
                if prid is None:
                    continue
                prof_node = profile_by_id.get(int(prid))
                if prof_node is None:
                    continue
                try:
                    score = int(score_any)
                except Exception:
                    continue

                w.writerow(
                    {
                        "predictor_id": pnode.id,
                        "predictor_name": pnode.name,
                        "predictor_full_path": pnode.full_path,
                        "profile_id": prof_node.id,
                        "profile_name": prof_node.name,
                        "profile_full_path": prof_node.full_path,
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

    parser.add_argument("--predictors-list", type=str, default=DEFAULT_PREDICTORS_LIST_TXT)
    parser.add_argument("--profiles-list", type=str, default=DEFAULT_PROFILES_LIST_TXT)
    parser.add_argument("--out-base", type=str, default=DEFAULT_OUT_BASE)

    parser.add_argument("--test-mode", action="store_true", default=TEST_MODE_DEFAULT)
    parser.add_argument("--full-run", action="store_true", default=False)
    parser.add_argument("--n-test", type=int, default=N_TEST_DEFAULT)

    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS_DEFAULT)
    parser.add_argument("--max-inflight-llm", type=int, default=MAX_INFLIGHT_LLM_CALLS_DEFAULT)

    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES_DEFAULT)
    parser.add_argument("--retry-base-sleep", type=float, default=RETRY_BASE_SLEEP_DEFAULT)

    parser.add_argument(
        "--compact-prompts",
        action="store_true",
        default=False,
        help="Omit full raw hierarchies from prompts (smaller/cheaper).",
    )

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
    compact_prompts = bool(args.compact_prompts)

    out_dir = _resolve_out_dir(args.out_base, model)
    out_json = os.path.join(out_dir, "predictor_to_profile_map.json")
    out_csv_dense = os.path.join(out_dir, "predictor_to_profile_map_dense.csv")
    out_csv_long = os.path.join(out_dir, "predictor_to_profile_edges_long.csv")

    print("=== CONFIG ===")
    print(f"model: {model}")
    print(f"reasoning_effort: {reasoning_effort}")
    print(f"predictors_list: {args.predictors_list}")
    print(f"profiles_list: {args.profiles_list}")
    print(f"out_dir: {out_dir}")
    print(f"out_json: {out_json}")
    print(f"out_csv_dense: {out_csv_dense}")
    print(f"out_csv_long: {out_csv_long}")
    print(f"test_mode: {test_mode} | n_test: {n_test}")
    print(f"max_workers: {max_workers}")
    print(f"max_inflight_llm_calls: {max_inflight}")
    print(f"max_retries: {max_retries} | retry_base_sleep: {retry_base_sleep}")
    print(f"compact_prompts: {compact_prompts}")
    print("================\n")

    os.makedirs(out_dir, exist_ok=True)

    predictors = load_leaves_from_tree_txt(args.predictors_list)
    profiles = load_leaves_from_tree_txt(args.profiles_list)

    predictors_raw_text = "" if compact_prompts else load_raw_text(args.predictors_list)
    profiles_raw_text = "" if compact_prompts else load_raw_text(args.profiles_list)

    print(f"[load] predictors parsed: {len(predictors)} (IDs: {predictors[0].id}..{predictors[-1].id})")
    print(f"[load] profiles parsed: {len(profiles)} (IDs: {profiles[0].id}..{profiles[-1].id})")

    # Sanity: unique IDs
    pred_ids = [p.id for p in predictors]
    prof_ids = [p.id for p in profiles]
    if len(set(pred_ids)) != len(pred_ids):
        raise ValueError("Predictor IDs are not unique.")
    if len(set(prof_ids)) != len(prof_ids):
        raise ValueError("Profile IDs are not unique.")

    predictors_run_list = predictors
    if test_mode:
        predictors_run_list = predictors[: min(n_test, len(predictors))]
        print(f"[test] limiting predictors to first {len(predictors_run_list)} IDs: {[p.id for p in predictors_run_list]}")

    results = load_results_json(out_json)
    if not isinstance(results, dict):
        print("[warn] existing results JSON is not a dict; resetting.")
        results = {}
    print(f"[cache] existing predictors in results JSON: {len(results)}")

    client = OpenAI()
    semaphore = threading.Semaphore(max_inflight)
    system_prompt = build_system_prompt()

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import itertools

    total = len(predictors_run_list)
    started_at = time.time()
    done_counter = 0
    done_lock = threading.Lock()

    def _worker(predictor: LeafNode) -> int:
        nonlocal done_counter
        try:
            process_one_predictor(
                predictor=predictor,
                profiles_raw_text=profiles_raw_text,
                predictors_raw_text=predictors_raw_text,
                profiles=profiles,
                client=client,
                results_path=out_json,
                results=results,
                semaphore=semaphore,
                model=model,
                system_prompt=system_prompt,
                reasoning_effort=reasoning_effort,
                max_retries=max_retries,
                retry_base_sleep=retry_base_sleep,
                compact_prompts=compact_prompts,
            )
        finally:
            with done_lock:
                done_counter += 1
                elapsed = time.time() - started_at
                print(f"[progress] predictors_done={done_counter}/{total} | elapsed={elapsed/60:.1f} min")
        return predictor.id

    print("\n=== RUN ===")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        it = iter(predictors_run_list)
        active: Dict[Any, int] = {}

        for p in itertools.islice(it, max_workers):
            fut = ex.submit(_worker, p)
            active[fut] = p.id

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

    # Reload latest results
    results = load_results_json(out_json)

    print("\n=== WRITE CSVs ===")
    write_dense_csv_from_results(results=results, predictors=predictors, profiles=profiles, out_csv=out_csv_dense)
    write_long_edges_csv(results=results, predictors=predictors, profiles=profiles, out_csv=out_csv_long)

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
