#!/usr/bin/env python3
"""
check_concistency.py

Purpose
-------
Evaluate ONLY *vertical* (parent → child) subclass relations inside a hierarchical BIO–PSYCHO–SOCIAL ontology.

IMPORTANT (scope guarantees)
----------------------------
1) This script evaluates ONLY IMMEDIATE parent→child edges (tree edges).
   - No sibling/same-depth comparisons.
   - No cousin comparisons.
   - No undirected pairs.
   - No multi-hop ancestor–descendant pairs beyond one edge.

2) Evaluation is restricted to edges that lie INSIDE the set of selected, non-overlapping subtrees ("selected domains")
   constructed by a leaf-budget partition (max_leaves_per_domain).
   - Edges are never evaluated across two different selected domains.
   - Each vertical edge belongs to at most one selected domain (non-overlap guarantee).

3) For scalability:
   - For each selected domain, we sample up to max_pairs_per_domain vertical edges.
   - The LLM only sees (parent, child) pairs, and returns only violated ones.

Fix for Responses JSON mode (400 error)
--------------------------------------
If using Responses API with:
    text={"format": {"type": "json_object"}}
some deployments require the *input* to contain the substring "json" (lowercase).
This script includes "json" (lowercase) in BOTH system prompt and user message.

Concurrency requirement (PARALLEL SUBTREES)
-------------------------------------------
This version runs SELECTED SUBTREES IN PARALLEL.
- One global ThreadPoolExecutor with max_workers=120
- One global inflight semaphore with max_inflight=100 (default) to cap concurrent API calls

Caching requirement
-------------------
- After each subtree is processed, it writes a per-subtree JSON cache file immediately in output_dir/cache/.
- On the next run, cached subtrees are skipped automatically if:
  - cache exists, and
  - cache "config_fingerprint" matches current run settings (model/max_leaves/max_pairs/batch_size/seed),
    and
  - cache reports status == "ok"

At the end, the script aggregates cached + newly computed subtree results and writes a final report.

Defaults requested
------------------
--max-leaves 200
--max-pairs 200
--batch-size 100
--seed 42
--model gpt-5-nano
--max-workers 120
--max-inflight 100

Output
------
- Console: prints plan + parallel scheduling stats + live completion updates.
- File:
  - Per-subtree cache: output_dir/cache/subtree__<hash>.json
  - Final report: output_dir/<report-prefix>_<timestamp>.txt

Usage
-----
export OPENAI_API_KEY="..."
python check_concistency.py

Optional overrides:
python check_concistency.py --model gpt-5-mini --max-leaves 200 --max-pairs 200 --batch-size 100
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore, Lock

from dotenv import load_dotenv

load_dotenv()


# -----------------------------
# Defaults (your provided paths)
# -----------------------------
DEFAULT_ONTOLOGY_PATH = (
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/SystemComponents/"
    "PHOENIX_ontology/separate/01_raw/PREDICTOR/steps/01_raw/aggregated/"
    "PREDICTOR_ontology.json"
)

DEFAULT_OUTPUT_DIR = (
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/"
    "vertical_migration_scheme/result"
)

DEFAULT_MODEL = "gpt-5-nano"

# Requested parallelism defaults
DEFAULT_MAX_WORKERS = 120
DEFAULT_MAX_INFLIGHT = 100


# -----------------------------
# Data structures
# -----------------------------
@dataclass(frozen=True)
class Edge:
    """A vertical (parent → child) edge; ALWAYS immediate (one hop)."""
    parent: str
    child: str
    parent_path: Tuple[str, ...]
    child_path: Tuple[str, ...]
    parent_depth: int
    child_is_leaf: bool


@dataclass
class DomainSelection:
    """A selected subtree ("domain") used as a local evaluation unit."""
    root_path: Tuple[str, ...]
    leaf_count: int
    all_edges: List[Edge]
    sampled_edges: List[Edge]


# -----------------------------
# Ontology helpers
# -----------------------------
def _is_mapping(x: Any) -> bool:
    return isinstance(x, dict)


def load_ontology_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ontology JSON not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not _is_mapping(data):
        raise ValueError("Top-level ontology JSON must be a dict/object.")
    return data


def iter_children(node: Any) -> Iterable[Tuple[str, Any]]:
    if not _is_mapping(node):
        return
    for k, v in node.items():
        yield str(k), v


def is_leaf(node: Any) -> bool:
    if not _is_mapping(node):
        return True
    return len(node) == 0


def compute_leaf_count(node: Any) -> int:
    if is_leaf(node):
        return 1
    return sum(compute_leaf_count(child) for _, child in iter_children(node))


def get_subtree(root: Dict[str, Any], path: Tuple[str, ...]) -> Any:
    cur: Any = root
    for p in path:
        if not _is_mapping(cur) or p not in cur:
            raise KeyError(f"Path not found in ontology: {'/'.join(path)}")
        cur = cur[p]
    return cur


def path_str(path: Tuple[str, ...]) -> str:
    return " / ".join(path) if path else "<ROOT>"


def top_domain_of_path(path: Tuple[str, ...]) -> str:
    return path[0] if path else "<ROOT>"


# -----------------------------
# Vertical-edge extraction (STRICT)
# -----------------------------
def list_vertical_edges_in_subtree(
    root: Dict[str, Any],
    domain_root_path: Tuple[str, ...],
) -> List[Edge]:
    """
    STRICT VERTICAL-ONLY extraction:
    Returns ONLY immediate parent→child edges inside the given subtree.
    """
    subtree = get_subtree(root, domain_root_path)
    edges: List[Edge] = []

    def dfs(node: Any, cur_path: Tuple[str, ...], rel_depth: int) -> None:
        if not _is_mapping(node) or len(node) == 0:
            return
        for child_name, child_node in iter_children(node):
            parent_name = cur_path[-1] if cur_path else "<ROOT>"
            child_path = cur_path + (child_name,)

            edges.append(
                Edge(
                    parent=parent_name,
                    child=child_name,
                    parent_path=cur_path,
                    child_path=child_path,
                    parent_depth=rel_depth,
                    child_is_leaf=is_leaf(child_node),
                )
            )
            dfs(child_node, child_path, rel_depth + 1)

    if len(domain_root_path) == 0:
        for top_name, top_node in iter_children(subtree):
            top_path = (top_name,)
            edges.append(
                Edge(
                    parent="<ROOT>",
                    child=top_name,
                    parent_path=(),
                    child_path=top_path,
                    parent_depth=0,
                    child_is_leaf=is_leaf(top_node),
                )
            )
            dfs(top_node, top_path, 1)
    else:
        dfs(subtree, domain_root_path, 0)

    return edges


# -----------------------------
# Domain partitioning (non-overlapping)
# -----------------------------
def partition_into_domains_by_leaf_limit(
    root: Dict[str, Any],
    max_leaves_per_domain: int,
) -> List[Tuple[str, ...]]:
    """Leaf-budget splitting into non-overlapping selected subtree roots."""
    domains: List[Tuple[str, ...]] = []

    for top_name, top_node in iter_children(root):
        stack: List[Tuple[Tuple[str, ...], Any]] = [((top_name,), top_node)]
        while stack:
            path, node = stack.pop()
            lc = compute_leaf_count(node)

            if lc <= max_leaves_per_domain or is_leaf(node):
                domains.append(path)
                continue

            if _is_mapping(node) and len(node) > 0:
                for child_name, child_node in iter_children(node):
                    stack.append((path + (child_name,), child_node))
            else:
                domains.append(path)

    domains.sort(key=lambda p: (len(p), p))
    return domains


# -----------------------------
# Sampling edges within a domain
# -----------------------------
def weighted_sample_without_replacement(
    rng: random.Random,
    items: List[Edge],
    k: int,
    weight_fn,
) -> List[Edge]:
    """
    Efraimidis–Spirakis (WRS-N-WOR):
    key_i = U_i^(1/w_i), select top-k keys
    """
    if k >= len(items):
        return list(items)

    keys: List[Tuple[float, Edge]] = []
    for it in items:
        w = max(1e-9, float(weight_fn(it)))
        u = rng.random()
        key = u ** (1.0 / w)
        keys.append((key, it))

    keys.sort(key=lambda x: x[0], reverse=True)
    return [it for _, it in keys[:k]]


def sample_edges_depth_aware(
    rng: random.Random,
    edges: List[Edge],
    max_pairs: int,
) -> List[Edge]:
    """
    Depth-aware sampling of VERTICAL edges only.
    Keeps parent_depth<=1 if possible, then samples deeper edges (prefers leaf children).
    """
    if len(edges) <= max_pairs:
        out = list(edges)
        out.sort(key=lambda e: (e.parent_depth, e.parent, e.child))
        return out

    shallow = [e for e in edges if e.parent_depth <= 1]
    deep = [e for e in edges if e.parent_depth >= 2]

    if len(shallow) >= max_pairs:
        def w_shallow(e: Edge) -> float:
            return 2.0 if e.parent_depth == 0 else 1.0
        out = weighted_sample_without_replacement(rng, shallow, max_pairs, w_shallow)
        out.sort(key=lambda e: (e.parent_depth, e.parent, e.child))
        return out

    selected: List[Edge] = list(shallow)
    remaining = max_pairs - len(selected)

    if remaining > 0 and deep:
        def w_deep(e: Edge) -> float:
            return 2.0 if e.child_is_leaf else 1.0
        selected.extend(weighted_sample_without_replacement(rng, deep, remaining, w_deep))

    selected = selected[:max_pairs]
    selected.sort(key=lambda e: (e.parent_depth, e.parent, e.child))
    return selected


# -----------------------------
# OpenAI call wrapper
# -----------------------------
# NOTE: include lowercase "json" explicitly to satisfy Responses API heuristic.
LLM_SYSTEM_PROMPT = """You are an ontology quality auditor.

You will be given ONLY vertical parent→child subclass pairs (immediate edges) from a biopsychosocial predictor ontology.
Your job is to identify which pairs violate the abstract→concrete gradient.

Meaning of abstract→concrete gradient:
- Parent must be MORE abstract/general than the child.
- Child must be a MORE specific subtype (a plausible 'is-a') of the parent.
- Underscores '_' should be read as spaces.

Mark a pair as VIOLATION if:
1) Child is broader/more abstract than parent, OR
2) Parent and child are at the same abstraction level (no clear specialization), OR
3) Relation is not plausibly 'child is-a parent'.

Be conservative:
- If uncertain, do NOT mark as a violation.

Output format (STRICT):
Return valid json only with schema:
{"violations": ["PARENT > CHILD", ...]}

Use exact names from input. If no violations:
{"violations": []}
"""


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s


def _extract_first_json_object(text: str) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text

    start = text.find("{")
    if start == -1:
        raise ValueError("No '{' found in model output; cannot parse json.")

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1].strip()

    raise ValueError("Unbalanced braces; could not extract json object.")


def _parse_violations_json(text: str) -> List[str]:
    text = _strip_code_fences(text)
    try:
        obj = json.loads(text)
    except Exception:
        obj = json.loads(_extract_first_json_object(text))

    viol = obj.get("violations", [])
    if not isinstance(viol, list):
        raise ValueError("Parsed json but 'violations' is not a list.")

    out: List[str] = []
    for v in viol:
        if isinstance(v, str):
            out.append(v.strip())
    return out


def _responses_create(
    client,
    model: str,
    system_prompt: str,
    user_text: str,
    use_json_mode: bool,
):
    """
    Tries both common Responses API call shapes:
    - input=[{role,content}, ...]
    - instructions=..., input="..."
    """
    kwargs: Dict[str, Any] = {}
    if use_json_mode:
        kwargs["text"] = {"format": {"type": "json_object"}}

    try:
        return client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            **kwargs,
        )
    except TypeError:
        return client.responses.create(
            model=model,
            instructions=system_prompt,
            input=user_text,
            **kwargs,
        )


def call_openai_violations(
    model: str,
    pairs: List[Tuple[str, str]],
    max_retries: int = 6,
    base_sleep: float = 1.0,
) -> List[str]:
    """
    Sends a batch of VERTICAL parent→child pairs and returns violations as "PARENT > CHILD" strings.
    No temperature is passed.

    JSON mode robustness:
    - Try Responses json_object mode
    - If error complains about json heuristic, retry without json_object mode (prompt-only)
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError(
            "Python package 'openai' not installed or failed to import. "
            "Install with: pip install --upgrade openai"
        ) from e

    client = OpenAI(api_key=api_key)

    payload = {"pairs": [{"parent": p, "child": c} for (p, c) in pairs]}

    # Include lowercase "json" in user message
    user_text = (
        "Evaluate these parent→child subclass pairs.\n"
        "Return valid json only (no extra text).\n\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )

    last_err: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            if hasattr(client, "responses") and hasattr(client.responses, "create"):
                try:
                    resp = _responses_create(
                        client=client,
                        model=model,
                        system_prompt=LLM_SYSTEM_PROMPT,
                        user_text=user_text,
                        use_json_mode=True,
                    )
                except Exception as e:
                    msg = str(e).lower()
                    if "must contain the word 'json'" in msg and "json_object" in msg:
                        resp = _responses_create(
                            client=client,
                            model=model,
                            system_prompt=LLM_SYSTEM_PROMPT,
                            user_text=user_text,
                            use_json_mode=False,
                        )
                    else:
                        raise

                text_out = getattr(resp, "output_text", None)
                if not text_out:
                    chunks: List[str] = []
                    try:
                        for item in getattr(resp, "output", []):
                            content = getattr(item, "content", None)
                            if not content:
                                continue
                            for c in content:
                                t = getattr(c, "text", None)
                                if t:
                                    chunks.append(t)
                    except Exception:
                        pass
                    text_out = "".join(chunks).strip()

                if not text_out:
                    text_out = str(resp)

                return _parse_violations_json(text_out)

            if hasattr(client, "chat") and hasattr(client.chat, "completions"):
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": LLM_SYSTEM_PROMPT},
                            {"role": "user", "content": user_text},
                        ],
                        response_format={"type": "json_object"},
                    )
                except TypeError:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": LLM_SYSTEM_PROMPT},
                            {"role": "user", "content": user_text},
                        ],
                    )

                text_out = resp.choices[0].message.content or ""
                return _parse_violations_json(text_out)

            raise RuntimeError("No supported OpenAI API method found in this SDK (responses or chat).")

        except Exception as e:
            last_err = e
            sleep_s = base_sleep * (2 ** attempt) + random.random() * 0.2
            time.sleep(min(sleep_s, 20.0))

    raise RuntimeError(f"OpenAI call failed after {max_retries} attempts: {last_err}")


# -----------------------------
# Caching helpers
# -----------------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def compute_config_fingerprint(args: argparse.Namespace) -> str:
    """
    Fingerprint of settings that affect evaluation outputs.
    If any of these change, we treat caches as stale by default.
    """
    cfg = {
        "model": args.model,
        "max_leaves": args.max_leaves,
        "max_pairs": args.max_pairs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "ontology_path": args.ontology_path,
        # NOTE: include prompt versioning so prompt edits invalidate cache
        "prompt_sha1": hashlib.sha1(LLM_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
    }
    return hashlib.sha1(_stable_json_dumps(cfg).encode("utf-8")).hexdigest()


def domain_cache_key(domain_root_path: Tuple[str, ...], config_fingerprint: str) -> str:
    """
    Stable per-domain cache key (does not depend on run timestamp).
    """
    s = f"{config_fingerprint}||{'/'.join(domain_root_path)}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def cache_paths(output_dir: str) -> Tuple[str, str]:
    cache_dir = os.path.join(output_dir, "cache")
    ensure_dir(cache_dir)
    return cache_dir, os.path.join(cache_dir, "_index.json")


def cache_file_for_domain(output_dir: str, key: str) -> str:
    cache_dir, _ = cache_paths(output_dir)
    return os.path.join(cache_dir, f"subtree__{key}.json")


def load_domain_cache(output_dir: str, key: str) -> Optional[Dict[str, Any]]:
    p = cache_file_for_domain(output_dir, key)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def write_domain_cache_atomic(path: str, obj: Dict[str, Any]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# -----------------------------
# Reporting (final report)
# -----------------------------
def write_report_txt(
    output_dir: str,
    report_name: str,
    config: Dict[str, Any],
    domain_results: List[Dict[str, Any]],
    aggregate_results: Dict[str, Any],
) -> str:
    ensure_dir(output_dir)
    out_path = os.path.join(output_dir, report_name)

    lines: List[str] = []
    lines.append("VERTICAL CONSISTENCY REPORT (Immediate Parent→Child Edges Only)\n")
    lines.append(f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}\n\n")

    lines.append("CONFIG\n")
    for k in sorted(config.keys()):
        lines.append(f"- {k}: {config[k]}\n")
    lines.append("\n")

    lines.append("AGGREGATE RESULTS (Top-level domains)\n")
    for dom in ["BIO", "PSYCHO", "SOCIAL", "ALL"]:
        if dom in aggregate_results:
            r = aggregate_results[dom]
            lines.append(
                f"- {dom}: violations={r['violations']} / evaluated={r['evaluated']} "
                f"({r['rate']:.3f})\n"
            )

    lines.append("\nDOMAIN DETAILS (each is a selected, non-overlapping subtree)\n")
    for dr in domain_results:
        lines.append("\n" + "=" * 80 + "\n")
        lines.append(f"Selected domain subtree: {dr['domain_path']}\n")
        lines.append(f"Leaf count (subtree): {dr['leaf_count']}\n")
        lines.append(f"Vertical edges in subtree (total): {dr['total_edges_in_subtree']}\n")
        lines.append(f"Vertical edges evaluated (sampled): {dr['evaluated_pairs']}\n")
        lines.append(f"Violations: {dr['violations_count']} ({dr['violation_rate']:.3f})\n")
        lines.append(f"LLM calls used: {dr['llm_calls']}\n")
        lines.append(f"Cache key: {dr.get('cache_key','')}\n")
        lines.append(f"Source: {dr.get('source','')}\n")

        if dr.get("violations"):
            lines.append("\nViolated parent→child edges:\n")
            for v in dr["violations"]:
                lines.append(f"- {v}\n")
        else:
            lines.append("\nViolated parent→child edges: none\n")

        if dr.get("violations_with_paths"):
            lines.append("\nTraceback paths for violated edges:\n")
            for item in dr["violations_with_paths"]:
                lines.append(f"- {item['pair']}\n")
                lines.append(f"  parent_path: {item['parent_path']}\n")
                lines.append(f"  child_path : {item['child_path']}\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    return out_path


# -----------------------------
# Subtree processing (parallel unit)
# -----------------------------
def build_subtree_result_from_edges(
    dpath: Tuple[str, ...],
    leaf_count: int,
    all_edges: List[Edge],
    sampled_edges: List[Edge],
    violated_pairs: List[str],
) -> Dict[str, Any]:
    violated_set = set(v.strip() for v in violated_pairs if v.strip())

    violations_with_paths: List[Dict[str, str]] = []
    for e in sampled_edges:
        key = f"{e.parent} > {e.child}"
        if key in violated_set:
            violations_with_paths.append(
                {
                    "pair": key,
                    "parent_path": path_str(e.parent_path),
                    "child_path": path_str(e.child_path),
                }
            )

    n_viol = len(violations_with_paths)
    n_eval = len(sampled_edges)
    rate = (n_viol / n_eval) if n_eval else 0.0

    return {
        "domain_path": path_str(dpath),
        "domain_root_path": list(dpath),  # JSON-serializable
        "leaf_count": leaf_count,
        "total_edges_in_subtree": len(all_edges),
        "evaluated_pairs": n_eval,
        "violations_count": n_viol,
        "violation_rate": rate,
        "llm_calls": math.ceil(n_eval / 1) if n_eval else 0,  # overwritten below by real count
        "violations": sorted(set(item["pair"] for item in violations_with_paths)),
        "violations_with_paths": violations_with_paths,
    }


def process_one_subtree(
    *,
    cache_key: str,
    config_fingerprint: str,
    output_dir: str,
    model: str,
    batch_size: int,
    inflight_sem: Semaphore,
    selection: DomainSelection,
    print_lock: Lock,
) -> Dict[str, Any]:
    """
    Processes ONE selected subtree: runs required LLM calls (batched), returns a result dict.
    Writes cache immediately (ok or error).
    """
    dpath = selection.root_path
    cache_path = cache_file_for_domain(output_dir, cache_key)

    # quick per-subtree header (parallel-safe)
    with print_lock:
        print(f"[SUBTREE] start: {path_str(dpath)} | sampled_edges={len(selection.sampled_edges)}")

    edges = selection.sampled_edges
    pairs = [(e.parent, e.child) for e in edges]
    batches = [pairs[j:j + batch_size] for j in range(0, len(pairs), batch_size)]

    violated_pairs: List[str] = []
    t0 = time.time()

    # Execute this subtree's LLM calls sequentially *within* subtree to reduce burstiness,
    # BUT many subtrees run in parallel; max_inflight caps overall concurrency.
    calls_ok = 0
    try:
        for bi, b in enumerate(batches, start=1):
            inflight_sem.acquire()
            try:
                with print_lock:
                    print(f"  [SUBTREE] {path_str(dpath)} | LLM {bi}/{len(batches)} | batch={len(b)} | inflight<=cap")
                viol = call_openai_violations(model=model, pairs=b)
                violated_pairs.extend(viol)
                calls_ok += 1
            finally:
                inflight_sem.release()

        # Build result
        result = build_subtree_result_from_edges(
            dpath=dpath,
            leaf_count=selection.leaf_count,
            all_edges=selection.all_edges,
            sampled_edges=selection.sampled_edges,
            violated_pairs=violated_pairs,
        )
        result["llm_calls"] = len(batches)
        result["cache_key"] = cache_key
        result["config_fingerprint"] = config_fingerprint
        result["status"] = "ok"
        result["source"] = "computed"
        result["elapsed_s"] = round(time.time() - t0, 3)

        # Cache immediately
        write_domain_cache_atomic(cache_path, result)

        with print_lock:
            print(f"[SUBTREE] done:  {path_str(dpath)} | violations={result['violations_count']}/{result['evaluated_pairs']} | cached")
        return result

    except Exception as e:
        err_obj = {
            "domain_path": path_str(dpath),
            "domain_root_path": list(dpath),
            "leaf_count": selection.leaf_count,
            "total_edges_in_subtree": len(selection.all_edges),
            "evaluated_pairs": len(selection.sampled_edges),
            "llm_calls": len(batches),
            "cache_key": cache_key,
            "config_fingerprint": config_fingerprint,
            "status": "error",
            "source": "computed",
            "error": str(e),
            "calls_completed": calls_ok,
            "elapsed_s": round(time.time() - t0, 3),
        }
        # Cache the error too (so you can inspect), but it will not be reused as ok.
        write_domain_cache_atomic(cache_path, err_obj)
        with print_lock:
            print(f"[SUBTREE] ERROR: {path_str(dpath)} | cached error | {e}")
        raise


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ontology-path", type=str, default=DEFAULT_ONTOLOGY_PATH)
    ap.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)

    # Requested defaults
    ap.add_argument("--max-leaves", type=int, default=200)
    ap.add_argument("--max-pairs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report-prefix", type=str, default="vertical_pair_consistency_report")

    # Parallelism defaults
    ap.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    ap.add_argument("--max-inflight", type=int, default=DEFAULT_MAX_INFLIGHT)

    args = ap.parse_args()
    rng = random.Random(args.seed)

    ensure_dir(args.output_dir)
    cache_dir, _ = cache_paths(args.output_dir)

    ontology = load_ontology_json(args.ontology_path)

    selected_domain_paths = partition_into_domains_by_leaf_limit(
        ontology, max_leaves_per_domain=args.max_leaves
    )

    # Build selections (must be done before parallel step)
    selections: List[DomainSelection] = []
    total_domains = 0
    total_edges_in_selected_subtrees = 0
    total_edges_to_evaluate = 0
    total_calls_planned = 0

    for dpath in selected_domain_paths:
        subtree = get_subtree(ontology, dpath)
        leaf_cnt = compute_leaf_count(subtree)

        all_edges = list_vertical_edges_in_subtree(ontology, dpath)
        if not all_edges:
            continue

        sampled = sample_edges_depth_aware(rng, all_edges, args.max_pairs)
        if not sampled:
            continue

        selections.append(
            DomainSelection(
                root_path=dpath,
                leaf_count=leaf_cnt,
                all_edges=all_edges,
                sampled_edges=sampled,
            )
        )

        total_domains += 1
        total_edges_in_selected_subtrees += len(all_edges)
        total_edges_to_evaluate += len(sampled)
        total_calls_planned += math.ceil(len(sampled) / args.batch_size)

    config_fingerprint = compute_config_fingerprint(args)

    # Determine caching: which subtrees can be skipped
    to_compute: List[Tuple[str, DomainSelection]] = []
    cached_ok_results: List[Dict[str, Any]] = []
    cached_skipped = 0

    for sel in selections:
        key = domain_cache_key(sel.root_path, config_fingerprint)
        cached = load_domain_cache(args.output_dir, key)
        if cached and cached.get("status") == "ok" and cached.get("config_fingerprint") == config_fingerprint:
            cached["source"] = "cache"
            cached_ok_results.append(cached)
            cached_skipped += 1
        else:
            to_compute.append((key, sel))

    calls_planned_compute_only = 0
    for _, sel in to_compute:
        calls_planned_compute_only += math.ceil(len(sel.sampled_edges) / args.batch_size)

    print("\n" + "=" * 108)
    print("PLAN (computed BEFORE LLM calls)")
    print(f"Ontology path: {args.ontology_path}")
    print(f"Model: {args.model}")
    print(f"Leaf budget (max_leaves_per_domain): {args.max_leaves}")
    print(f"Per-domain evaluation cap (max_pairs_per_domain): {args.max_pairs}")
    print(f"Batch size (pairs/call): {args.batch_size}")
    print(f"Random seed: {args.seed}")
    print(f"Concurrency: ThreadPoolExecutor max_workers={args.max_workers} | max_inflight={args.max_inflight}")
    print(f"Caching: cache_dir={cache_dir}")
    print(f"Config fingerprint: {config_fingerprint}")
    print("-" * 108)
    print(f"Selected domains (non-overlapping subtrees): {total_domains}")
    print(f"TOTAL vertical edges INSIDE selected domains (no sampling): {total_edges_in_selected_subtrees}")
    print(f"TOTAL vertical edges that WILL be evaluated (after sampling/caps): {total_edges_to_evaluate}")
    print(f"TOTAL planned LLM calls (if no cache): {total_calls_planned}")
    print(f"Cached subtrees skipped: {cached_skipped}")
    print(f"Subtrees to compute now: {len(to_compute)}")
    print(f"Planned LLM calls (compute-only): {calls_planned_compute_only}")
    print("=" * 108)
    print("Scope guarantee: ONLY immediate parent→child edges (vertical). No horizontal pairs.\n")

    if total_edges_to_evaluate == 0:
        print("Nothing to evaluate (no vertical edges found in selected domains). Exiting.")
        return

    # Global concurrency controls
    inflight_sem = Semaphore(args.max_inflight)
    print_lock = Lock()

    # Run subtrees in parallel
    computed_results: List[Dict[str, Any]] = []
    errors: List[str] = []

    if to_compute:
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futures = []
            for key, sel in to_compute:
                futures.append(
                    ex.submit(
                        process_one_subtree,
                        cache_key=key,
                        config_fingerprint=config_fingerprint,
                        output_dir=args.output_dir,
                        model=args.model,
                        batch_size=args.batch_size,
                        inflight_sem=inflight_sem,
                        selection=sel,
                        print_lock=print_lock,
                    )
                )

            done_n = 0
            total_n = len(futures)
            for fut in as_completed(futures):
                done_n += 1
                try:
                    res = fut.result()
                    computed_results.append(res)
                except Exception as e:
                    errors.append(str(e))
                if done_n % 1 == 0:
                    with print_lock:
                        print(f"[PROGRESS] completed subtrees: {done_n}/{total_n} | errors={len(errors)}")

    # Combine cached + computed (ok only)
    all_ok_results: List[Dict[str, Any]] = []
    all_ok_results.extend(cached_ok_results)
    all_ok_results.extend([r for r in computed_results if r.get("status") == "ok"])

    # Aggregate rates
    agg = {
        "BIO": {"evaluated": 0, "violations": 0},
        "PSYCHO": {"evaluated": 0, "violations": 0},
        "SOCIAL": {"evaluated": 0, "violations": 0},
        "ALL": {"evaluated": 0, "violations": 0},
    }

    total_llm_calls_used = 0
    total_edges_evaluated = 0

    # Normalize domain_results for report
    domain_results: List[Dict[str, Any]] = []
    for r in all_ok_results:
        # Ensure required keys exist
        dom_root = r.get("domain_root_path", [])
        dom_name = dom_root[0] if dom_root else "<ROOT>"

        evaluated = int(r.get("evaluated_pairs", 0))
        violations = int(r.get("violations_count", 0))
        llm_calls = int(r.get("llm_calls", 0))

        if dom_name in agg:
            agg[dom_name]["evaluated"] += evaluated
            agg[dom_name]["violations"] += violations
        agg["ALL"]["evaluated"] += evaluated
        agg["ALL"]["violations"] += violations

        total_llm_calls_used += llm_calls
        total_edges_evaluated += evaluated

        domain_results.append(r)

    # Sort for stable report
    domain_results.sort(key=lambda x: (len(x.get("domain_root_path", [])), x.get("domain_path", "")))

    aggregate_results: Dict[str, Any] = {}
    for k, v in agg.items():
        evaluated = v["evaluated"]
        violations = v["violations"]
        aggregate_results[k] = {
            "evaluated": evaluated,
            "violations": violations,
            "rate": (violations / evaluated) if evaluated else 0.0,
        }

    # Write final report
    now = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = f"{args.report_prefix}_{now}.txt"

    config = {
        "ontology_path": args.ontology_path,
        "output_dir": args.output_dir,
        "model": args.model,
        "max_leaves_per_domain": args.max_leaves,
        "max_pairs_per_domain": args.max_pairs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "max_workers": args.max_workers,
        "max_inflight": args.max_inflight,
        "config_fingerprint": config_fingerprint,
        "selected_domains": total_domains,
        "cached_subtrees_skipped": cached_skipped,
        "subtrees_computed_now": len(to_compute),
        "total_vertical_edges_in_selected_domains": total_edges_in_selected_subtrees,
        "total_vertical_edges_targeted_for_eval": total_edges_to_evaluate,
        "total_vertical_edges_evaluated_ok": total_edges_evaluated,
        "total_llm_calls_planned_if_no_cache": total_calls_planned,
        "total_llm_calls_planned_compute_only": calls_planned_compute_only,
        "total_llm_calls_used_ok": total_llm_calls_used,
        "errors_count": len(errors),
    }

    out_path = write_report_txt(
        output_dir=args.output_dir,
        report_name=report_name,
        config=config,
        domain_results=domain_results,
        aggregate_results=aggregate_results,
    )

    print("\n" + "=" * 108)
    print("DONE")
    print(f"Subtrees cached+ok included: {len(all_ok_results)} / {total_domains}")
    print(f"Subtrees skipped due to cache: {cached_skipped}")
    print(f"Subtrees computed now: {len(to_compute)}")
    print(f"Total vertical edges evaluated (ok): {total_edges_evaluated}")
    print(f"Total LLM calls used (ok): {total_llm_calls_used}")
    print(f"Errors: {len(errors)}")
    if errors:
        print("First error (for quick debugging):")
        print(f"  - {errors[0]}")
    print(f"Final report written to: {out_path}")
    print("=" * 108 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)

#TODO: fix social part --> many 'wrong'?? answers:
    # VERSION 1 of BIO-PSYCHO-SOCIAL ontology
        #- BIO: violations=23 / evaluated=1496 (0.015) --> OKE — sufficient
        #- PSYCHO: violations=4 / evaluated=1338 (0.003) --> OKE — sufficient
        #- SOCIAL: violations=22366 / evaluated=26732 (0.837) --> inherent to architecture due to tri-combinatorial participation candidates ; so ignore --> OKE — sufficient
    # VERSION 2 of BIO-PSYCHO-SOCIAL ontology
        #- BIO: violations=29 / evaluated=2334 (0.012) --> OKE — sufficient
        #- PSYCHO: violations = 18 / evaluated = 2146(0.008) --> OKE — sufficient
        #- SOCIAL: violations = 7 / evaluated = 970(0.007) --> OKE — sufficient