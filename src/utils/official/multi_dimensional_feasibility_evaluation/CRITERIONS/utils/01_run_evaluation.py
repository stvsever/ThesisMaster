#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_run_cluster_evaluation.py

Cluster-level evaluation runner for semantically-clustered ontology criterions.

IMPORTANT CHANGE (vs per-criterion evaluation)
- This script performs ONE multi-dimensional suitability evaluation PER CLUSTER (not per item).
- The LLM produces a single `CriterionEvaluation` object that summarizes the cluster as an aggregate "variable family".
- This prevents nonsensical "Parsed evaluations length mismatch" errors and matches the intended workflow when
  the leaf-node set is too large.

Structured outputs + schema compatibility
- Uses responses.create() with Structured Outputs (json_schema, strict=true).
- Compatible with scores-only hierarchical schema module:

    00_hierarchical_criterion_evaluation_modules.py

  (weights are NOT part of the LLM-facing output; weights are internal utilities in the module)

Caching & outputs
- Primary cache is the output CSV keyed by `cluster_hash` (membership signature).
- Optional per-cluster JSON cache file is written in cache_evaluations/ for traceability (full member lists).
- A "wide" CSV is also produced (flattened evaluation_json) for quick inspection.

Threading
- One task per cluster; thread-safe appends to output CSV.

NOTE ABOUT YOUR JSON
Your clusters JSON has the structure:
{
  "meta": {...},
  "tree": {...},          # optional
  "clusters": {
      "c3104": {"size": 1, "items": [ ... ]},
      ...
  }
}
This runner uses a SIMPLE extraction: it iterates payload["clusters"] and reads each block's "items".
It does NOT rely on the "tree" for discovering cluster blocks (tree is used only to compute an optional path).
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Type, cast

import pandas as pd

# OpenAI SDK (expects you have installed the official python package: pip install openai)
from openai import OpenAI  # type: ignore

from pydantic import BaseModel, Field

# Optional: dotenv
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

sys.setrecursionlimit(20_000)

# -----------------------------------------------------------------------------
# User paths (defaults; override via CLI flags)
# -----------------------------------------------------------------------------

# Prefer the UPDATED "criterion" schema module by default.
# If you still want the older file, pass --eval-module explicitly.
EVAL_MODULE_PATH = Path(
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/multi_dimensional_feasibility_evaluation/"
    "CRITERIONS/utils/00_hierarchical_evaluation_modules.py"
)

CLUSTERS_JSON_PATH = Path(
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/"
    "cluster_criterions/results/04_semantically_clustered_items.json"
)

RESULTS_DIR = Path(
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/"
    "multi_dimensional_feasibility_evaluation/CRITERIONS/results/responses"
)

# -----------------------------------------------------------------------------
# Tunables
# -----------------------------------------------------------------------------

MODEL_NAME = "gpt-5-nano"

DEFAULT_MAX_WORKERS = 50
DEFAULT_CLUSTER_SAMPLE_MAX = 250   # how many raw items to include in the prompt (sampling if larger)
DEFAULT_RETRIES = 2

MAX_ITEM_TEXT_CHARS = 10_000
MAX_CONTEXT_ITEM_CHARS = 600
MAX_CLUSTER_PROMPT_CHARS = 220_000

# Kept for backwards compatibility with earlier patchers; harmless if absent.
WEIGHT_KEYS_TO_REMOVE = {"weights", "dimension_weights", "method_weights"}

# -----------------------------------------------------------------------------
# Prompt engineering
# -----------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert in:
- clinical-experimental health engineering,
- psychometrics and measurement validity,
- intensive longitudinal (time-series) modeling (e.g., time-varying gVAR),
- digital phenotyping / EMA / wearable sensing / ETL labeling pipelines,
- EU data protection & health-tech regulatory constraints.

Task
You will be given ONE semantic cluster containing multiple criterion leaf-nodes.
You must evaluate the CLUSTER AS A WHOLE (an aggregate "variable family"), NOT each item separately.

Goal
Return a single multi-dimensional suitability profile for this cluster, i.e., estimate the likelihood that
problems will arise when using this cluster as a candidate observation variable family in a digital
momentary mental-state optimization engine.

Scoring scale (9-point Likert; ALWAYS integers 1..9)
Each score = PROBLEM likelihood for that feature:
    1 = Negligible likelihood
    2 = Very low likelihood
    3 = Low likelihood
    4 = Mild likelihood
    5 = Moderate / uncertain likelihood
    6 = Elevated likelihood
    7 = High likelihood
    8 = Very high likelihood
    9 = Near-certain / critical likelihood

Important conventions
- Output MUST include all features for all dimensions; do not skip fields.
- If a data-collection method is not realistically feasible, still output that method with high problem likelihood
  (typically 8–9), rather than omitting it.
- You may use the member list to infer what this cluster represents; calibrate scores to the overall family.
- No prose outside the structured output.
"""

USER_PROMPT_TEMPLATE = """\
Cluster information
- cluster_id: {cluster_id}
- cluster_hash: {cluster_hash}
- cluster_tree_path: {cluster_tree_path}
- cluster_size: {cluster_size}

Member items (sample; truncated; may be sampled if very large):
{cluster_items_block}

Now evaluate this CLUSTER AS A WHOLE (aggregate) according to a multi-domain suitability framework.

Output requirements
- Return a single object with key "evaluation"
- The "evaluation" must follow the CriterionEvaluation schema (scores-only).
- metadata:
    - criterion_id: "cluster:{cluster_id}"
    - label: "cluster:{cluster_id}"
    - definition: null
    - ontology_name: null
    - age_group: null
- Provide scores for:
    - mathematical_restrictions.scores
    - data_collection_feasibility:
        - aggregation: "best_available"
        - self_report_ema.scores
        - third_party_ema.scores
        - wearable.scores
        - user_device_data.scores
        - etl_pipeline.scores
        - third_party_api.scores
    - validity_threats:
        - response_bias.scores
        - insight_capacity.scores
        - measurement_validity.scores
    - eu_regulatory_risk:
        - gdpr.scores
        - eu_ai_act.scores
        - medical_device.scores
        - eprivacy.scores
        - cybersecurity.scores
    - general_importance.scores
    - scientific_utility.scores

Strict rules
- Every score must be an integer 1..9.
- No additional keys.
- Do not include weights.
"""

# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def log_and_print(log_path: Path, msg: str) -> None:
    line = f"[{utc_now_iso()}] {msg}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")

def human_path(p: Path) -> str:
    try:
        return str(p.resolve())
    except Exception:
        return str(p)

# -----------------------------------------------------------------------------
# Dynamic import of evaluation module (absolute path)
# -----------------------------------------------------------------------------

def import_eval_module(path: Path):
    import importlib.util

    if not path.exists():
        raise FileNotFoundError(f"Evaluation module not found: {path}")

    spec = importlib.util.spec_from_file_location("hier_eval", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for module: {path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

def rebuild_pydantic_models_in_module(mod: Any, log_path: Path) -> None:
    """
    Rebuild pydantic models to resolve forward refs after dynamic import.
    """
    from pydantic import BaseModel as PydBaseModel

    ns: Dict[str, Any] = {name: getattr(mod, name) for name in dir(mod)}
    built = 0
    failed: List[str] = []

    for name, obj in ns.items():
        if not isinstance(obj, type):
            continue
        try:
            if issubclass(obj, PydBaseModel) and obj is not PydBaseModel:
                try:
                    obj.model_rebuild(force=True, _types_namespace=ns)  # type: ignore[arg-type]
                except TypeError:
                    obj.model_rebuild(force=True)
                built += 1
        except Exception as e:
            failed.append(f"{name}: {repr(e)}")

    log_and_print(log_path, f"Pydantic rebuild: rebuilt_models={built} failed={len(failed)}")
    for s in failed[:10]:
        log_and_print(log_path, f"WARNING: model_rebuild failed for {s}")

# -----------------------------------------------------------------------------
# Cluster parsing utilities (SIMPLE cluster discovery; tree optional)
# -----------------------------------------------------------------------------

# NOTE: In a previous iteration a bad regex (r"^c\\d+$") caused 0 matches.
# Here we do NOT filter keys by regex at all; we only require the shape {"items": [...]}.
TREE_NODE_ID_RE = re.compile(r"^c\d+$")  # only used for optional tree-path computation

CRIT_SUFFIX_RE = re.compile(r"\s+—\s*criterion\s*$", re.IGNORECASE)
CONTEXT_RE = re.compile(r"\(context:\s*(.*?)\)\s*$", re.IGNORECASE)

def short(s: str, n: int) -> str:
    s = s.replace("\n", " ").strip()
    return s[: n - 1] + "…" if len(s) > n else s

def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def load_clusters_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Clusters JSON not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Clusters JSON is empty: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level JSON object (dict). Got: {type(data)}")
    return data

def extract_tree(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    t = payload.get("tree")
    if isinstance(t, dict) and "id" in t and "children" in t:
        return t
    return None

def build_parent_map(tree_node: Dict[str, Any]) -> Dict[str, str]:
    parent: Dict[str, str] = {}

    def _walk(node: Dict[str, Any], parent_id: Optional[str]) -> None:
        node_id = node.get("id")
        if isinstance(node_id, str) and parent_id is not None:
            parent[node_id] = parent_id
        children = node.get("children", [])
        if not isinstance(children, list):
            return
        for ch in children:
            if isinstance(ch, dict):
                _walk(ch, node_id)

    _walk(tree_node, None)
    return parent

def compute_tree_path(cluster_id: str, parent_map: Dict[str, str]) -> str:
    if not parent_map or cluster_id not in parent_map:
        return cluster_id
    parts = [cluster_id]
    cur = cluster_id
    seen = set()
    while cur in parent_map:
        if cur in seen:
            break
        seen.add(cur)
        cur = parent_map[cur]
        parts.append(cur)
    parts.reverse()
    return " > ".join(parts)

def parse_item_text(item_text: str) -> Tuple[str, Optional[str]]:
    s = item_text.strip()
    s = CRIT_SUFFIX_RE.sub("", s).strip()

    ontology_context = None
    m = CONTEXT_RE.search(s)
    if m:
        ontology_context = m.group(1).strip()
        s = CONTEXT_RE.sub("", s).strip()

    criterion_id = s.split()[0].strip() if s.split() else s
    return criterion_id, ontology_context

def iter_cluster_blocks(payload: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """
    SIMPLE extraction logic:
    - Read payload["clusters"] (must be a dict)
    - Yield every (cluster_id, block) where block is a dict and block["items"] is a list
    No regex filtering.
    """
    clusters = payload.get("clusters")
    if not isinstance(clusters, dict):
        return
    for k, v in clusters.items():
        if isinstance(k, str) and isinstance(v, dict) and isinstance(v.get("items"), list):
            yield k, v

def debug_payload_summary(payload: Dict[str, Any]) -> str:
    top_keys = list(payload.keys())
    meta = payload.get("meta")
    meta_bits = ""
    if isinstance(meta, dict):
        meta_bits = f"meta n_items={meta.get('n_items')} n_clusters={meta.get('n_clusters')}"
    clusters = payload.get("clusters")
    clusters_bits = ""
    if isinstance(clusters, dict):
        clusters_bits = f"clusters dict keys={len(clusters)} (sample={list(clusters.keys())[:3]})"
    return f"Top-level keys: {top_keys}\n{meta_bits}\n{clusters_bits}".strip()

# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ClusterItem:
    raw_item_text: str
    criterion_id: str
    ontology_context: Optional[str]
    item_hash: str

@dataclasses.dataclass(frozen=True)
class ClusterBlock:
    cluster_id: str
    cluster_tree_path: str
    cluster_size: int
    items: List[ClusterItem]

# -----------------------------------------------------------------------------
# Directories
# -----------------------------------------------------------------------------

def ensure_dirs(base: Path) -> Dict[str, Path]:
    base.mkdir(parents=True, exist_ok=True)
    dirs = {
        "base": base,
        "cache": base / "cache_evaluations",   # per-cluster JSON cache (traceability)
        "errors": base / "errors",
        "tables": base / "tables",
        "logs": base / "logs",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs

def atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

# -----------------------------------------------------------------------------
# OpenAI response helpers
# -----------------------------------------------------------------------------

def response_text(resp: Any) -> str:
    t = getattr(resp, "output_text", None)
    if isinstance(t, str) and t.strip():
        return t.strip()

    out: List[str] = []
    for item in getattr(resp, "output", []) or []:
        content = getattr(item, "content", None)
        if not content:
            continue
        for c in content:
            tt = getattr(c, "text", None)
            if isinstance(tt, str):
                out.append(tt)
    return "".join(out).strip()

def parse_json_strict(s: str) -> Any:
    s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        i = s.find("{")
        j = s.rfind("}")
        if i != -1 and j != -1 and j > i:
            return json.loads(s[i : j + 1])
        raise

def backoff_sleep(attempt: int) -> None:
    base = min(30.0, 0.6 * (2**attempt))
    time.sleep(base + random.random() * 0.25)

# -----------------------------------------------------------------------------
# Structured Outputs schema patcher (ITERATIVE, NO RECURSION)
# -----------------------------------------------------------------------------

def patch_schema_for_structured_outputs(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Iteratively patch a Pydantic JSON schema to satisfy OpenAI Structured Outputs constraints.

    Guarantees:
    - Any dict that has "$ref" becomes ONLY {"$ref": "..."} (no siblings)
    - allOf with single $ref collapses to $ref
    - For objects: additionalProperties=false and required includes every remaining property key
      (forces Optional[...] fields to be present, which is desired here)
    - Removes defaults/titles/descriptions globally
    - Removes weight-bearing keys from schema entirely
    - Produces a pure-JSON deep copy at the end (no Python reference cycles)
    """
    stack: List[Any] = [schema]
    seen: set[int] = set()

    while stack:
        node = stack.pop()
        oid = id(node)
        if oid in seen:
            continue
        seen.add(oid)

        if isinstance(node, dict):
            if "$ref" in node:
                ref = node.get("$ref")
                node.clear()
                node["$ref"] = ref
                continue

            for k in ("default", "title", "description", "examples"):
                if k in node:
                    node.pop(k, None)

            allof = node.get("allOf")
            if isinstance(allof, list) and len(allof) == 1:
                only = allof[0]
                if isinstance(only, dict) and "$ref" in only:
                    node.clear()
                    node["$ref"] = only["$ref"]
                    continue

            props = node.get("properties")
            if isinstance(props, dict):
                for wk in WEIGHT_KEYS_TO_REMOVE:
                    props.pop(wk, None)
                node["properties"] = props
                node["required"] = list(props.keys())
                node["additionalProperties"] = False

            for v in node.values():
                if isinstance(v, (dict, list)):
                    stack.append(v)

        elif isinstance(node, list):
            for v in node:
                if isinstance(v, (dict, list)):
                    stack.append(v)

    return cast(Dict[str, Any], json.loads(json.dumps(schema)))

# -----------------------------------------------------------------------------
# Output schema bundle (cluster-level)
# -----------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class OutputSchemaBundle:
    wrapper_model: Type[BaseModel]
    json_schema: Dict[str, Any]
    schema_name: str

def build_output_schema(mod: Any, log_path: Path) -> OutputSchemaBundle:
    CriterionEvaluation = mod.CriterionEvaluation  # type: ignore[attr-defined]

    class ClusterEvaluationWrapper(BaseModel):
        evaluation: CriterionEvaluation = Field(...)

    ns = {name: getattr(mod, name) for name in dir(mod)}
    try:
        ClusterEvaluationWrapper.model_rebuild(force=True, _types_namespace=ns)  # type: ignore[arg-type]
    except TypeError:
        ClusterEvaluationWrapper.model_rebuild(force=True)

    schema = ClusterEvaluationWrapper.model_json_schema()
    schema = patch_schema_for_structured_outputs(cast(Dict[str, Any], schema))

    schema_name = "cluster_evaluation_wrapper"
    log_and_print(log_path, f"Built + patched Structured Output schema once. name='{schema_name}'")
    return OutputSchemaBundle(wrapper_model=ClusterEvaluationWrapper, json_schema=schema, schema_name=schema_name)

def validate_schema_once(client: OpenAI, schema_bundle: OutputSchemaBundle, log_path: Path) -> None:
    """
    Validate that the API accepts the patched JSON schema.
    """
    log_and_print(log_path, "Validating Structured Outputs schema with a tiny API call ...")
    _ = client.responses.create(
        model=MODEL_NAME,
        input=[
            {"role": "system", "content": "Return any valid JSON object that matches the provided json_schema exactly. No extra keys."},
            {"role": "user", "content": "Produce a minimal but fully valid object for this schema."},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": schema_bundle.schema_name,
                "strict": True,
                "schema": schema_bundle.json_schema,
            }
        },
    )
    log_and_print(log_path, "Schema validation call succeeded.")

# -----------------------------------------------------------------------------
# Prompt building (cluster-level, sampling-aware)
# -----------------------------------------------------------------------------

def compute_cluster_hash(cluster_items: List[ClusterItem]) -> str:
    """
    Stable signature for the cluster membership/order.
    Uses item_hashes derived from raw strings (stable across runs).
    """
    joined = "\n".join(ci.item_hash for ci in cluster_items)
    return sha1_text(joined)

def sample_cluster_items(items: List[ClusterItem], max_n: int, seed: int, cluster_id: str) -> List[ClusterItem]:
    """
    Deterministic sampling: always include first 10, last 10, plus a deterministic random sample of the remainder.
    """
    if max_n <= 0 or len(items) <= max_n:
        return items

    head = items[:10]
    tail = items[-10:] if len(items) > 10 else []
    fixed = {ci.item_hash: ci for ci in (head + tail)}  # dedupe

    remaining = [ci for ci in items if ci.item_hash not in fixed]
    need = max(0, max_n - len(fixed))

    rng = random.Random(f"{seed}:{cluster_id}")
    if need >= len(remaining):
        sampled = remaining
    else:
        sampled = rng.sample(remaining, k=need)

    # Preserve readability: head (in order) + sampled (random) + tail (in order)
    out: List[ClusterItem] = []
    for ci in head:
        if ci not in out:
            out.append(ci)
    for ci in sampled:
        if ci not in out:
            out.append(ci)
    for ci in tail:
        if ci not in out:
            out.append(ci)

    return out[:max_n]

def make_cluster_items_block(items: List[ClusterItem]) -> str:
    lines: List[str] = []
    for i, ci in enumerate(items, 1):
        lines.append(
            f"- {i}. criterion_id: {ci.criterion_id} | "
            f"text: {short(ci.raw_item_text, MAX_CONTEXT_ITEM_CHARS)} | "
            f"context: {ci.ontology_context if ci.ontology_context is not None else 'null'}"
        )
    return "\n".join(lines) if lines else "- (none)"

def build_user_prompt(
    cluster_id: str,
    cluster_hash: str,
    cluster_tree_path: str,
    cluster_size: int,
    cluster_items_block: str,
) -> str:
    return USER_PROMPT_TEMPLATE.format(
        cluster_id=cluster_id,
        cluster_hash=cluster_hash,
        cluster_tree_path=cluster_tree_path,
        cluster_size=cluster_size,
        cluster_items_block=cluster_items_block,
    )

def ensure_prompt_fits(prompt: str, cluster_id: str, log_path: Path) -> str:
    """
    Hard guard: if prompt still exceeds MAX_CLUSTER_PROMPT_CHARS, truncate from the bottom.
    """
    if len(prompt) <= MAX_CLUSTER_PROMPT_CHARS:
        return prompt

    lines = prompt.splitlines()
    while lines and len("\n".join(lines)) > MAX_CLUSTER_PROMPT_CHARS:
        lines.pop()

    new_prompt = "\n".join(lines).rstrip() + "\n"
    log_and_print(
        log_path,
        f"WARNING: prompt too large for cluster={cluster_id}; truncated to len={len(new_prompt)} chars.",
    )
    return new_prompt

# -----------------------------------------------------------------------------
# LLM evaluation logic (cluster-level)
# -----------------------------------------------------------------------------

def eval_cluster(
    client: OpenAI,
    schema_bundle: OutputSchemaBundle,
    cluster_id: str,
    cluster_hash: str,
    cluster_tree_path: str,
    cluster_size: int,
    cluster_items_block: str,
    retries: int,
    log_path: Path,
) -> Any:
    user_prompt = build_user_prompt(
        cluster_id=cluster_id,
        cluster_hash=cluster_hash,
        cluster_tree_path=cluster_tree_path,
        cluster_size=cluster_size,
        cluster_items_block=cluster_items_block,
    )
    user_prompt = ensure_prompt_fits(user_prompt, cluster_id=cluster_id, log_path=log_path)

    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            if attempt == 0:
                log_and_print(log_path, f"API CALL: cluster={cluster_id} (cluster-level) retries={retries}")

            resp = client.responses.create(
                model=MODEL_NAME,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": schema_bundle.schema_name,
                        "strict": True,
                        "schema": schema_bundle.json_schema,
                    }
                },
            )

            txt = response_text(resp)
            data = parse_json_strict(txt)

            parsed = schema_bundle.wrapper_model.model_validate(data)
            ev = parsed.evaluation  # type: ignore[attr-defined]
            return ev

        except Exception as e:
            last_err = e
            log_and_print(log_path, f"API ERROR (attempt {attempt+1}/{retries}) cluster={cluster_id}: {repr(e)}")
            backoff_sleep(attempt)

    raise RuntimeError(f"Failed after {retries} attempts. Last error: {last_err}") from last_err

# -----------------------------------------------------------------------------
# Orchestration helpers
# -----------------------------------------------------------------------------

def collect_clusters(payload: Dict[str, Any], log_path: Path) -> Dict[str, ClusterBlock]:
    """
    Build cluster blocks from payload["clusters"].
    Tree is OPTIONAL; it is used only to compute a path if present.
    """
    tree = extract_tree(payload)
    parent_map: Dict[str, str] = build_parent_map(tree) if tree is not None else {}

    if tree is None:
        log_and_print(log_path, "No 'tree' found (or not in expected shape). cluster_tree_path will be cluster_id.")
    else:
        log_and_print(log_path, "Tree found; computing parent map (optional cluster_tree_path).")

    out: Dict[str, ClusterBlock] = {}
    n_blocks = 0

    for cluster_id, block in iter_cluster_blocks(payload):
        n_blocks += 1
        raw_items = block.get("items", [])
        if not isinstance(raw_items, list):
            continue

        cluster_size = int(block.get("size", len(raw_items)))

        # Optional: compute tree path only if cluster_id is a canonical tree-node id.
        if TREE_NODE_ID_RE.match(cluster_id):
            cluster_tree_path = compute_tree_path(cluster_id, parent_map)
        else:
            cluster_tree_path = cluster_id

        items: List[ClusterItem] = []
        for it in raw_items:
            if not isinstance(it, str):
                continue
            criterion_id, ontology_context = parse_item_text(it)
            ih = sha1_text(it)
            items.append(
                ClusterItem(
                    raw_item_text=it,
                    criterion_id=criterion_id,
                    ontology_context=ontology_context,
                    item_hash=ih,
                )
            )

        if items:
            out[cluster_id] = ClusterBlock(
                cluster_id=cluster_id,
                cluster_tree_path=cluster_tree_path,
                cluster_size=cluster_size,
                items=items,
            )

    if n_blocks == 0:
        summary = debug_payload_summary(payload)
        log_and_print(log_path, "ERROR: No cluster blocks discovered. Payload summary:\n" + summary)
        raise AssertionError("No cluster blocks discovered.\n\n" + summary)

    if not out:
        summary = debug_payload_summary(payload)
        log_and_print(log_path, "ERROR: 0 clusters extracted. Payload summary:\n" + summary)
        raise AssertionError("0 clusters extracted.\n\n" + summary)

    # Log a sample
    first_key = sorted(out.keys(), key=lambda x: int(x[1:]) if len(x) > 1 and x[1:].isdigit() else 10**9)[0]
    sample_cluster = out[first_key]
    log_and_print(log_path, f"Extracted clusters: {len(out)} (sample cluster={first_key} n_items={len(sample_cluster.items)})")
    if sample_cluster.items:
        log_and_print(log_path, f"  sample member: {short(sample_cluster.items[0].raw_item_text, 140)}")

    return out

def sort_cluster_ids(cluster_ids: List[str]) -> List[str]:
    return sorted(cluster_ids, key=lambda x: int(x[1:]) if len(x) > 1 and x[1:].isdigit() else 10**9)

# -----------------------------------------------------------------------------
# CSV cache IO (cluster-level)
# -----------------------------------------------------------------------------

OUTPUT_COLUMNS = [
    "run_id",
    "model_name",
    "cached_at_utc",
    "cluster_id",
    "cluster_hash",
    "cluster_tree_path",
    "cluster_size",
    "n_members",
    "member_hashes_json",
    "member_criterion_ids_json",
    "member_sample_text",
    "evaluation_json",
]

def load_cached_cluster_hashes_from_csv(csv_path: Path, log_path: Path) -> set[str]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(csv_path, usecols=["cluster_hash"], dtype={"cluster_hash": "string"})
        hs = set(df["cluster_hash"].dropna().astype(str).tolist())
        log_and_print(log_path, f"Loaded cached cluster_hashes from CSV: {len(hs)} ({human_path(csv_path)})")
        return hs
    except Exception as e:
        log_and_print(log_path, f"WARNING: failed reading cached cluster hashes from CSV: {repr(e)} ({human_path(csv_path)})")
        return set()

def append_rows_to_csv(csv_path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0

    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in OUTPUT_COLUMNS})

def finalize_csv_dedup_sort(csv_path: Path, log_path: Path) -> None:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return
    try:
        df = pd.read_csv(csv_path, dtype="string")
        if "cluster_hash" not in df.columns:
            log_and_print(log_path, f"WARNING: cannot finalize; missing cluster_hash column in {human_path(csv_path)}")
            return

        before = len(df)
        df = df.drop_duplicates(subset=["cluster_hash"], keep="last")

        if "cluster_id" in df.columns:
            def _cid_num(x: str) -> int:
                try:
                    s = str(x)
                    return int(s[1:]) if len(s) > 1 and s[1:].isdigit() else 10**9
                except Exception:
                    return 10**9
            df["_cluster_num"] = df["cluster_id"].astype(str).map(_cid_num)
            df = df.sort_values(["_cluster_num"]).drop(columns=["_cluster_num"])

        tmp = csv_path.with_suffix(".tmp")
        df.to_csv(tmp, index=False, encoding="utf-8")
        os.replace(tmp, csv_path)

        after = len(df)
        log_and_print(log_path, f"Finalized CSV: dedup {before}->{after} rows | {human_path(csv_path)}")
    except Exception as e:
        log_and_print(log_path, f"WARNING: finalize_csv_dedup_sort failed: {repr(e)}")

def flatten_dict(d: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_dict(v, key))
    elif isinstance(d, list):
        out[prefix] = json.dumps(d, ensure_ascii=False)
    else:
        out[prefix] = d
    return out

def build_wide_csv(from_csv: Path, to_csv: Path, log_path: Path) -> None:
    if not from_csv.exists() or from_csv.stat().st_size == 0:
        return
    try:
        df = pd.read_csv(from_csv, dtype="string")
        if "evaluation_json" not in df.columns:
            return

        wide_rows: List[Dict[str, Any]] = []
        base_cols = [c for c in df.columns if c != "evaluation_json"]

        for _, row in df.iterrows():
            base = {c: row.get(c, "") for c in base_cols}
            ej = row.get("evaluation_json", "")
            try:
                ev = json.loads(ej) if isinstance(ej, str) and ej.strip() else {}
            except Exception:
                ev = {"_evaluation_json_parse_error": True}
            flat = flatten_dict(ev, prefix="evaluation")
            merged = {**base, **flat}
            wide_rows.append(merged)

        wide_df = pd.DataFrame(wide_rows)
        wide_df.to_csv(to_csv, index=False, encoding="utf-8")
        log_and_print(log_path, f"Wrote wide CSV: {human_path(to_csv)} (rows={len(wide_df)} cols={len(wide_df.columns)})")
    except Exception as e:
        log_and_print(log_path, f"WARNING: build_wide_csv failed: {repr(e)}")

# -----------------------------------------------------------------------------
# Task iterator: ONE task per cluster if not cached
# -----------------------------------------------------------------------------

def iter_cluster_tasks(
    cluster_ids: List[str],
    clusters: Dict[str, ClusterBlock],
    cached_cluster_hashes: set[str],
    overwrite: bool,
) -> Iterator[str]:
    for cid in cluster_ids:
        cb = clusters[cid]
        chash = compute_cluster_hash(cb.items)
        if overwrite or (chash not in cached_cluster_hashes):
            yield cid

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--clusters-json", type=str, default=str(CLUSTERS_JSON_PATH))
    ap.add_argument("--eval-module", type=str, default=str(EVAL_MODULE_PATH))
    ap.add_argument("--results-dir", type=str, default=str(RESULTS_DIR))

    ap.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    ap.add_argument("--cluster-sample-max", type=int, default=DEFAULT_CLUSTER_SAMPLE_MAX)
    ap.add_argument("--retries", type=int, default=DEFAULT_RETRIES)

    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="Re-evaluate clusters even if cached in output CSV.")
    ap.add_argument("--no-json-cache", action="store_true", help="Disable per-cluster JSON cache files (cache_evaluations/).")

    # TEST MODE (cluster sampling)
    ap.add_argument("--test-n-clusters", type=int, default=10, help="Evaluate N random clusters (0 disables).")
    ap.add_argument("--test-seed", type=int, default=42, help="Random seed for --test-n-clusters (also used for item sampling).")

    # Output control
    ap.add_argument("--out-csv", type=str, default="", help="Optional explicit output CSV path. If empty, auto-named.")
    ap.add_argument("--run-tag", type=str, default="", help="Optional tag inserted into output filenames.")

    args = ap.parse_args()

    clusters_json_path = Path(args.clusters_json)
    eval_module_path = Path(args.eval_module)
    results_dir = Path(args.results_dir)

    dirs = ensure_dirs(results_dir)
    log_path = dirs["logs"] / "run.log"

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_tag = str(args.run_tag).strip()
    if run_tag:
        run_tag = re.sub(r"[^A-Za-z0-9_\-]+", "_", run_tag)

    # Output filenames
    if args.out_csv:
        out_csv = Path(args.out_csv)
    else:
        base = "cluster_criterion_evaluations"
        if run_tag:
            base += f"_{run_tag}"
        if args.test_n_clusters and int(args.test_n_clusters) > 0:
            base += f"_TESTn{int(args.test_n_clusters)}_seed{int(args.test_seed)}"
        out_csv = dirs["tables"] / f"{base}.csv"

    wide_csv = out_csv.with_name(out_csv.stem + "_wide.csv")

    log_and_print(log_path, "==== RUN START (CLUSTER-LEVEL) ====")
    log_and_print(log_path, f"run_id:             {run_id}")
    log_and_print(log_path, f"clusters_json_path:  {human_path(clusters_json_path)}")
    log_and_print(log_path, f"eval_module_path:    {human_path(eval_module_path)}")
    log_and_print(log_path, f"results_dir:         {human_path(results_dir)}")
    log_and_print(log_path, f"output_csv:          {human_path(out_csv)}")
    log_and_print(log_path, f"wide_csv:            {human_path(wide_csv)}")

    log_and_print(
        log_path,
        f"clusters_json exists={clusters_json_path.exists()} size_bytes={clusters_json_path.stat().st_size if clusters_json_path.exists() else 'NA'}",
    )
    log_and_print(
        log_path,
        f"eval_module exists={eval_module_path.exists()} size_bytes={eval_module_path.stat().st_size if eval_module_path.exists() else 'NA'}",
    )

    log_and_print(log_path, "Loading evaluation module...")
    mod = import_eval_module(eval_module_path)
    assert hasattr(mod, "CriterionEvaluation"), "Evaluation module missing 'CriterionEvaluation'."
    assert hasattr(mod, "compute_overall_suitability"), "Evaluation module missing 'compute_overall_suitability'."
    rebuild_pydantic_models_in_module(mod, log_path=log_path)

    schema_bundle = build_output_schema(mod, log_path=log_path)

    log_and_print(log_path, "Loading clusters JSON...")
    payload = load_clusters_json(clusters_json_path)
    log_and_print(log_path, "Payload quick summary:\n" + debug_payload_summary(payload))

    log_and_print(log_path, "Collecting clusters...")
    clusters = collect_clusters(payload, log_path=log_path)
    cluster_ids_full = sort_cluster_ids(list(clusters.keys()))
    log_and_print(log_path, f"Clusters present: {len(cluster_ids_full)} (sample {cluster_ids_full[:10]})")

    # TEST MODE: sample N clusters
    cluster_ids = cluster_ids_full
    if args.test_n_clusters and int(args.test_n_clusters) > 0 and len(cluster_ids_full) > int(args.test_n_clusters):
        rng = random.Random(int(args.test_seed))
        sampled = rng.sample(cluster_ids_full, k=int(args.test_n_clusters))
        cluster_ids = sort_cluster_ids(sampled)
        log_and_print(log_path, f"TEST MODE: selected {len(cluster_ids)} clusters (seed={args.test_seed})")

    if args.dry_run:
        log_and_print(log_path, "DRY RUN enabled; not calling OpenAI.")
        log_and_print(log_path, "==== RUN END (DRY RUN) ====")
        return 0

    # Primary cache: output CSV
    cached_cluster_hashes = load_cached_cluster_hashes_from_csv(out_csv, log_path=log_path)
    log_and_print(log_path, f"Cached cluster_hashes (from output CSV): {len(cached_cluster_hashes)}")

    client = OpenAI()
    validate_schema_once(client, schema_bundle, log_path=log_path)

    write_lock = threading.Lock()
    completed_clusters = 0
    failed_clusters = 0

    def _run_one_cluster(cluster_id: str) -> None:
        nonlocal completed_clusters, failed_clusters, cached_cluster_hashes

        cb = clusters[cluster_id]
        chash = compute_cluster_hash(cb.items)
        cpath = cb.cluster_tree_path
        csize = cb.cluster_size

        # Skip if cached (unless overwrite)
        if (not args.overwrite) and (chash in cached_cluster_hashes):
            with write_lock:
                completed_clusters += 1
            return

        # Deterministic sampling for prompt size control
        sampled_items = sample_cluster_items(
            cb.items,
            max_n=int(args.cluster_sample_max),
            seed=int(args.test_seed),
            cluster_id=cluster_id,
        )
        items_block = make_cluster_items_block(sampled_items)

        ev = eval_cluster(
            client=client,
            schema_bundle=schema_bundle,
            cluster_id=cluster_id,
            cluster_hash=chash,
            cluster_tree_path=cpath,
            cluster_size=csize,
            cluster_items_block=items_block,
            retries=int(args.retries),
            log_path=log_path,
        )

        # Prepare row
        member_hashes = [ci.item_hash for ci in cb.items]
        member_ids = [ci.criterion_id for ci in cb.items]
        member_sample_text = " | ".join([short(ci.raw_item_text, 120) for ci in sampled_items[:15]])

        row = {
            "run_id": run_id,
            "model_name": MODEL_NAME,
            "cached_at_utc": utc_now_iso(),
            "cluster_id": cluster_id,
            "cluster_hash": chash,
            "cluster_tree_path": cpath,
            "cluster_size": str(csize),
            "n_members": str(len(cb.items)),
            "member_hashes_json": json.dumps(member_hashes, ensure_ascii=False),
            "member_criterion_ids_json": json.dumps(member_ids, ensure_ascii=False),
            "member_sample_text": member_sample_text,
            "evaluation_json": json.dumps(ev.model_dump(), ensure_ascii=False),
        }

        # Write CSV + update cache set
        with write_lock:
            append_rows_to_csv(out_csv, [row])
            cached_cluster_hashes.add(chash)
            completed_clusters += 1
            if completed_clusters % 5 == 0:
                log_and_print(
                    log_path,
                    f"PROGRESS: completed_clusters={completed_clusters} failed_clusters={failed_clusters} cached_clusters={len(cached_cluster_hashes)}",
                )

        # Optional JSON cache (full, with all members)
        if not args.no_json_cache:
            entry = {
                "run_id": run_id,
                "model_name": MODEL_NAME,
                "cluster_id": cluster_id,
                "cluster_hash": chash,
                "cluster_tree_path": cpath,
                "cluster_size": csize,
                "n_members": len(cb.items),
                "members": [
                    {
                        "criterion_id": ci.criterion_id,
                        "ontology_context": ci.ontology_context,
                        "raw_item_text": ci.raw_item_text,
                        "item_hash": ci.item_hash,
                    }
                    for ci in cb.items
                ],
                "prompt_members_sampled": [ci.item_hash for ci in sampled_items],
                "evaluation": ev.model_dump(),
                "cached_at_utc": utc_now_iso(),
            }
            cp = dirs["cache"] / f"cluster_{cluster_id}_{chash}.json"
            atomic_write_json(cp, entry)

    log_and_print(log_path, f"Starting ThreadPoolExecutor max_workers={args.max_workers} ...")

    task_iter = iter_cluster_tasks(
        cluster_ids=cluster_ids,
        clusters=clusters,
        cached_cluster_hashes=cached_cluster_hashes,
        overwrite=bool(args.overwrite),
    )

    in_flight = set()
    with ThreadPoolExecutor(max_workers=int(args.max_workers)) as ex:
        for _ in range(int(args.max_workers)):
            try:
                cid = next(task_iter)
            except StopIteration:
                break
            in_flight.add(ex.submit(_run_one_cluster, cid))

        while in_flight:
            done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for fut in done:
                try:
                    fut.result()
                except Exception as e:
                    with write_lock:
                        failed_clusters += 1
                        err_obj = {"error": repr(e), "created_at_utc": utc_now_iso()}
                        err_file = dirs["errors"] / f"error_{failed_clusters}_{int(time.time())}.json"
                        atomic_write_json(err_file, err_obj)
                        log_and_print(log_path, f"CLUSTER FAILED ({failed_clusters}): {repr(e)} | saved: {human_path(err_file)}")

                try:
                    cid = next(task_iter)
                except StopIteration:
                    cid = None
                if cid is not None:
                    in_flight.add(ex.submit(_run_one_cluster, cid))

    log_and_print(log_path, f"Finished clusters. completed={completed_clusters} failed={failed_clusters}")

    # Finalize output CSV for clarity + safety (dedupe/sort atomically)
    finalize_csv_dedup_sort(out_csv, log_path=log_path)

    # Build wide CSV for readability
    build_wide_csv(from_csv=out_csv, to_csv=wide_csv, log_path=log_path)

    log_and_print(log_path, "==== RUN END ====")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
