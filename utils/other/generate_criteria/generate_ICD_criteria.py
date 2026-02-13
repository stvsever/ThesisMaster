#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_ICD_criteria.py

Generates clinical expression criteria + estimated prevalence for all ICD-related
leaf disorders (where mental functioning is implicated) in a nested JSON.

- Uses OpenAI structured outputs (Pydantic) via client.beta.chat.completions.parse
- Concurrency + progress logging
- Robust caching of per-disorder results to allow safe restarts
- Merges results back into the original ontology:
    leaf {}  ->  { "criteria": [...], "estimated_prevalence_percent": <float> }

Requirements:
    pip install openai pydantic python-dotenv

Environment:
    OPENAI_API_KEY must be set (e.g., via .env or environment)

Author: (your name)
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import re
import sys
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI

# ==============================================================================
# TESTING SWITCH (explicit variable)
# Set to True to run in test mode (process a small subset, fewer workers)
# You can also pass --test on the CLI; the CLI flag overrides this variable.
# ==============================================================================
TEST_MODE: bool = False
TEST_MAX_LEAVES: int = 3  # if TEST_MODE is True, limit to this many leaves
TEST_MAX_WORKERS: int = 4

# ==============================================================================
# Pydantic models for LLM output
# ==============================================================================
class Clinical_Criterion(BaseModel):
    criterium: str = Field(..., description="Short variable-like name, medical jargon, no description.")

class Clinical_Information(BaseModel):
    clinical_expression: List[Clinical_Criterion] = Field(
        ..., description="List of concrete observable criteria (6–15 items)."
    )
    estimated_prevalence_percent: float = Field(
        ..., description="Estimated point prevalence in percent (float, decimals allowed)."
    )

# TODO: include other metadata ; and direct age-specific prevalence distribution

# ==============================================================================
# Helpers
# ==============================================================================
def sanitize_filename(name: str) -> str:
    """Replace non-alphanumeric sequences with underscores."""
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")

def get_leaf_nodes(
    nested: Dict[str, Union[dict, list, str]],
    parent_path: List[str] | None = None,
) -> List[Dict[str, List[str]]]:
    """
    Traverse nested dict; return leaf disorders with their path (excluding the leaf itself).
    Leaf := value is not a non-empty dict.
    """
    if parent_path is None:
        parent_path = []
    leaf_nodes: List[Dict[str, List[str]]] = []
    for key, value in (nested or {}).items():
        current_path = parent_path + [key]
        if isinstance(value, dict) and len(value) > 0:
            leaf_nodes.extend(get_leaf_nodes(value, current_path))
        else:
            # leaf; parent is everything before key
            leaf_nodes.append({"disorder": key, "path": parent_path})
    return leaf_nodes

def create_directories(base_dir: Path, path_segments: List[str]) -> Path:
    """Ensure directory exists for a path under base_dir."""
    directory_path = base_dir.joinpath(*[sanitize_filename(p) for p in path_segments])
    directory_path.mkdir(parents=True, exist_ok=True)
    return directory_path

# ==============================================================================
# LLM call (structured parse)
# ==============================================================================
def build_user_query(disorder: str, path: List[str]) -> str:
    full_path = " > ".join(path + [disorder]) if path else disorder
    return (
        f"This is the ICD-related disorder where mental functioning is implicated: '{disorder}'. "
        f"Full path for context: '{full_path}'. "
        "Return ONLY a JSON object matching the schema with two fields:\n"
        "1) clinical_expression: a list of 6–15 items, each an object with one string field 'criterium' "
        "(short, variable-like medical-jargon name; no descriptions; each item distinct and observable).\n"
        "2) estimated_prevalence_percent: a single FLOAT number representing best-estimate POINT PREVALENCE in the general population, "
        "expressed as PERCENT (e.g., 12.4 for 12.4%). Decimals allowed. Use subtype context from the path if relevant.\n\n"
        "CRITICAL CONTENT RULES:\n"
        "- clinical_expression: focus strictly on clinical phenomena (idiosyncratic symptoms/expressions), not rules, durations, or treatments.\n"
        "- Each criterium must be atomic and unambiguous, never combining two different phenomena (i.e., no disjunctions like 'A or B').\n"
        "- No prevalence narratives; ONLY the numeric float in the field provided.\n"
        "- Output MUST match the requested JSON schema."
    )

def call_llm_parse(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_query: str,
    pydantic_model: type[BaseModel],
) -> Clinical_Information:
    """
    Call OpenAI with structured outputs (Pydantic model).
    Returns parsed Clinical_Information.
    """
    # NOTE: Do NOT pass temperature; some models reject non-default temperatures.
    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_query},
        ],
        response_format=pydantic_model,  # type-safe parse
    )
    # Parse to Pydantic via SDK helper; validate locally again just in case:
    parsed = response.choices[0].message.parsed  # type: ignore
    if not isinstance(parsed, pydantic_model):
        # Extra safeguard (rare)
        parsed = pydantic_model.model_validate(parsed)
    return parsed  # type: ignore[return-value]

# ==============================================================================
# Per-disorder processing
# ==============================================================================
def process_disorder(
    client: OpenAI,
    model: str,
    system_prompt: str,
    disorder_info: Dict[str, List[str]],
    tmp_dir: Path,
    logger: logging.Logger,
    progress: Tuple[threading.Lock, Dict[str, int]],
    print_prompts: bool = False,
):
    disorder = disorder_info["disorder"]
    path = disorder_info["path"]
    rel_dir = create_directories(tmp_dir, path)

    # Unique filename incorporating path + disorder
    safe_path = "__".join([sanitize_filename(p) for p in (path + [disorder])]) or "root"
    tmp_file = rel_dir / f"{safe_path}__criteria.json"

    # Cache hit -> skip
    if tmp_file.exists():
        logger.info(f"[CACHE] Skipping already-generated: {disorder}")
        with progress[0]:
            progress[1]["done"] += 1
        return

    try:
        user_query = build_user_query(disorder, path)

        # In test mode, print prompts and other relevant info
        if print_prompts:
            logger.info("----- BEGIN TEST PROMPT -----")
            logger.info(f"[DISORDER] {disorder}")
            logger.info(f"[PATH] {' > '.join(path) if path else '(root)'}")
            logger.info("[SYSTEM PROMPT]")
            logger.info(system_prompt)
            logger.info("[USER QUERY]")
            logger.info(user_query)
            logger.info("------ END TEST PROMPT ------")

        logger.info(f"[LLM] Generating for '{disorder}' (path: {' > '.join(path)})")
        parsed = call_llm_parse(
            client=client,
            model=model,
            system_prompt=system_prompt,
            user_query=user_query,
            pydantic_model=Clinical_Information,
        )

        # Persist the parsed structure (flat, as returned by Clinical_Information)
        tmp_file.parent.mkdir(parents=True, exist_ok=True)
        with tmp_file.open("w", encoding="utf-8") as f:
            json.dump(parsed.model_dump(), f, ensure_ascii=False, indent=2)

        logger.info(f"[OK] Saved: {tmp_file}")
    except ValidationError as ve:
        logger.error(f"[VALIDATION] '{disorder}': {ve}")
    except Exception as e:
        logger.error(f"[ERROR] '{disorder}': {e}")
    finally:
        with progress[0]:
            progress[1]["done"] += 1
            if progress[1]["total"] > 0:
                pct = 100.0 * progress[1]["done"] / progress[1]["total"]
            else:
                pct = 100.0
            logger.info(f"[PROGRESS] {progress[1]['done']}/{progress[1]['total']} ({pct:.1f}%)")

# ==============================================================================
# Merge results back into ontology
# ==============================================================================
def load_tmp_result(
    tmp_dir: Path,
    path: List[str],
    leaf: str
) -> Tuple[List[str], Optional[float]]:
    """
    Read cached other results and return (criteria_list, estimated_prevalence_percent).
    If missing/unreadable, returns ([], None).
    """
    rel_dir = tmp_dir.joinpath(*[sanitize_filename(p) for p in path])
    safe_path = "__".join([sanitize_filename(p) for p in (path + [leaf])]) or "root"
    fp = rel_dir / f"{safe_path}__criteria.json"
    if not fp.exists():
        return ([], None)
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
        items = data.get("clinical_expression", [])
        crits = [it.get("criterium", "").strip() for it in items if it.get("criterium")]
        prev = data.get("estimated_prevalence_percent", None)
        # Normalize prevalence to float if it's a string
        if isinstance(prev, str):
            try:
                prev = float(prev.strip().replace("%", ""))
            except Exception:
                prev = None
        return (crits, prev if isinstance(prev, (int, float)) else None)
    except Exception:
        return ([], None)

def merge_into_ontology(original: Dict, tmp_dir: Path) -> Dict:
    """
    Produce a new dict where each leaf {} becomes:
      {
        "criteria": [...],
        "estimated_prevalence_percent": <float or null>
      }
    """
    def _merge(node: Dict, path: List[str]) -> Dict:
        merged = {}
        for k, v in node.items():
            if isinstance(v, dict) and len(v) > 0:
                merged[k] = _merge(v, path + [k])
            else:
                crits, prev = load_tmp_result(tmp_dir, path, k)
                merged[k] = {
                    "criteria": crits,
                    "estimated_prevalence_percent": prev,
                }
        return merged
    return _merge(original, [])

# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate ICD-related clinical criteria + estimated prevalence for all leaf disorders where mental functioning is implicated.")
    parser.add_argument(
        "--input-json",
        default="/Users/stijnvanseveren/PythonProjects/MASTERPROEF/SystemComponents/PHOENIX_ontology/separate/CRITERION/steps/01_raw/clinical/01_pre_generation/ICD_adjusted/ICD_additional_disorders.json",
        help="Path to input nested ICD JSON (disorders where mental functioning is implicated).",
    )
    parser.add_argument(
        "--output-dir",
        default="/Users/stijnvanseveren/PythonProjects/MASTERPROEF/SystemComponents/PHOENIX_ontology/separate/CRITERION/steps/01_raw/clinical/02_post_generation/ICD_pg",
        help="Directory to store results (other responses + merged JSON).",
    )
    parser.add_argument("--model", default="gpt-5-nano", help="LLM model to use (default: gpt-5-nano).")
    parser.add_argument("--max-workers", type=int, default=50, help="Max workers for ThreadPoolExecutor.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    parser.add_argument("--test", action="store_true", help="Run in test mode (limit leaves and workers, print prompts).")
    args = parser.parse_args()

    # Effective test mode (CLI flag overrides constant)
    effective_test_mode = args.test or TEST_MODE

    # Logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logger = logging.getLogger("ICD")

    # Load env (OPENAI_API_KEY)
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        logger.critical("No OPENAI_API_KEY found in env or .env")
        sys.exit(1)

    # Paths
    input_path = Path(args.input_json).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    tmp_dir = output_dir / "_tmp_responses"
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # OpenAI client
    client = OpenAI()

    # Load ontology
    if not input_path.is_file():
        logger.critical(f"Input JSON not found: {input_path}")
        sys.exit(1)
    try:
        with input_path.open("r", encoding="utf-8") as f:
            ontology = json.load(f)
        logger.info(f"Loaded ontology from {input_path}")
    except json.JSONDecodeError as e:
        logger.critical(f"Invalid JSON at {input_path}: {e}")
        sys.exit(1)

    # Extract leaves
    leaves = get_leaf_nodes(ontology)
    total_disorders = len(leaves)

    # Apply test-mode limits
    max_workers = args.max_workers
    if effective_test_mode:
        leaves = leaves[:TEST_MAX_LEAVES]
        max_workers = min(max_workers, TEST_MAX_WORKERS)
        logger.info(f"TEST MODE: limiting to first {len(leaves)} leaves and {max_workers} workers")

    logger.info(f"Total disorders to process: {len(leaves)} (out of {total_disorders})")

    # By top-level category counts (for visibility)
    category_counts = defaultdict(int)
    for info in leaves:
        cat = info["path"][0] if info["path"] else "Uncategorized"
        category_counts[cat] += 1
    for cat, cnt in category_counts.items():
        logger.info(f"Category '{cat}': {cnt} leaf disorders in this run")

    # System prompt adapted for ICD context
    system_prompt = (
        "You are a highly experienced interdisciplinary medical-clinical psychologist tasked with generating the full high-resolution clinical expression "
        "of a provided ICD-related disorder where mental functioning is implicated. Provide a list of observable (internal and/or external) criteria that can be used "
        "to monitor/evaluate the ongoing disorder; use concise, correct medical jargon. The criteria must be scientifically valid "
        "and clearly indicative of the disorder. Do NOT include diagnostic rules, treatment options, or duration or any other non-idiosyncratic specifiers. "
        "Return between 6 and 15 criteria. Output only the short variable-like name for each criterium (no descriptions). "
        "Each criterium must be atomic and unambiguous, never combining two different phenomena (i.e., no disjunctions like 'A or B'). "
        "Return JSON matching the Pydantic schema."
    )

    # Concurrency + progress
    progress_lock = threading.Lock()
    progress_state = {"done": 0, "total": len(leaves)}

    logger.info(f"Starting generation with up to {max_workers} workers (model: {args.model})")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_disorder,
                client,
                args.model,
                system_prompt,
                info,
                tmp_dir,
                logger,
                (progress_lock, progress_state),
                print_prompts=effective_test_mode,  # print prompts only in test mode
            )
            for info in leaves
        ]
        # Iterate to surface exceptions early
        for fut in concurrent.futures.as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                logger.error(f"Unhandled exception in worker: {e}")

    logger.info("All requested generations attempted. Beginning merge step...")

    # Merge back into ontology
    merged = merge_into_ontology(ontology, tmp_dir)
    merged_outfile = output_dir / f"ICD_criteria_merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with merged_outfile.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    logger.info(f"Merged criteria written to: {merged_outfile}")

    logger.info("DONE.")

# ==============================================================================
if __name__ == "__main__":
    main()

# TODO: replace to 'mini' instead of 'nano' to save costs when actually running outside tests
