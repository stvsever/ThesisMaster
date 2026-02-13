#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_run_predictor_evaluation.py

Predictor-level evaluation runner for predictor ontology leaf-nodes ("solution candidates").

What this does
- Performs ONE multi-dimensional suitability evaluation PER PREDICTOR leaf-node (not clustered).
- Predictors are treated as actionable solution candidates that can be used to optimize a given
  (non-)clinical mental state within a proximal zone of responsibility.
- Uses Structured Outputs (json_schema, strict=true) to force a validated PredictorEvaluation object.

Inputs
- A plain-text file with one predictor leaf path per line (pipe-separated), e.g.:
    BIO | ... | Blood_Pressure_Optimization_Framework

Schema module
- Uses the scores-only predictor schema module:
    /.../PREDICTORS/utils/00_hierarchical_evaluation_modules.py
  (LLM-facing models contain scores only; weights are internal utilities in the module)

Caching & outputs
- Primary cache is the output CSV keyed by `predictor_hash` (hash of the full leaf-path line).
- Optional per-predictor JSON cache file is written in cache_evaluations/ (traceability).
- A "wide" CSV is also produced (flattened evaluation_json) for quick inspection.

Threading
- One task per predictor; thread-safe appends to output CSV.
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

# OpenAI SDK (expects: pip install openai)
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

EVAL_MODULE_PATH = Path(
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/"
    "multi_dimensional_feasibility_evaluation/PREDICTORS/utils/00_hierarchical_evaluation_modules.py"
)

PREDICTOR_LEAF_PATHS_TXT = Path(
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/"
    "multi_dimensional_feasibility_evaluation/PREDICTORS/utils/input/PREDICTOR_leaf_paths.txt"
)

RESULTS_DIR = Path(
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/"
    "multi_dimensional_feasibility_evaluation/PREDICTORS/results/responses"
)

# -----------------------------------------------------------------------------
# Tunables
# -----------------------------------------------------------------------------

MODEL_NAME = "gpt-5-nano"

DEFAULT_MAX_WORKERS = 50
DEFAULT_RETRIES = 2

MAX_CONTEXT_ITEM_CHARS = 800
MAX_PREDICTOR_PROMPT_CHARS = 80_000

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
- EU data protection & health-tech regulatory constraints,
- translating predictors into safe and actionable intervention targets.

Task
You will be given ONE predictor leaf-node (a "solution candidate") from a predictor ontology.
Evaluate that predictor as a candidate input variable/intervention target for optimizing a given
(non-)clinical mental state, within a proximal zone of responsibility.

Goal
Return a single multi-dimensional suitability profile for this predictor:
estimate the unsuitable nature (i.e., where problems might arise) of the soluction candidate inside digital application framework to optimize (non-)clinical mental health 

Scoring scale (9-point Likert; ALWAYS integers 1..9)
Each score = PROBLEM likelihood for that feature:
    1 = Negligible likelihood --> SUITABLE
    2 = Very low likelihood
    3 = Low likelihood
    4 = Mild likelihood
    5 = Moderate / uncertain likelihood
    6 = Elevated likelihood
    7 = High likelihood
    8 = Very high likelihood
    9 = Near-certain / critical likelihood --> UNSUITABLE

Important conventions
- Output MUST include all features for all dimensions; do not skip fields.
- Data collection feasibility:
  Even if a method is not realistically feasible, still output that method module with high problem likelihood
  (typically 8–9), rather than omitting it or setting it to null.
- Validity threats + EU regulatory submodules:
  Always output all submodules with scores (do not set them to null).
- No prose outside the structured output.
"""

USER_PROMPT_TEMPLATE = """\
Predictor leaf-node information
- predictor_id: {predictor_id}
- label: {label}
- biopsychosocial_layer: {biopsychosocial_layer}
- full_path: {full_path}
- path_segments: {path_segments_json}

Now evaluate this PREDICTOR leaf-node as a SOLUTION CANDIDATE (input variable + actionable target) using the predictor suitability framework.

Output requirements
- Return a single object with key "evaluation"
- The "evaluation" must follow the PredictorEvaluation schema (scores-only).
- metadata:
    - predictor_id: "{predictor_id}"
    - label: "{label}"
    - definition: null
    - ontology_name: null
    - age_group: null
    - biopsychosocial_layer: {biopsychosocial_layer_or_null}

- Provide scores for ALL modules (no nulls):
    - mathematical_suitability.scores #NOTE: for time-varying graphical vector auto-regression
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
    - treatment_translation.scores
    - eu_regulatory_risk:
        - gdpr.scores
        - eu_ai_act.scores
        - medical_device.scores
        - eprivacy.scores
        - cybersecurity.scores
    - scientific_utility.scores

Strict rules
- Every score must be an integer 1..9.
- HIGHER scores reflect HIGHER PROBLEM LIKELIHOOD (i.e., 1=SUITABLE variable, 9=UNSUITABLE variable).
- No additional keys.
- Do not include weights (none at any nesting level).
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
# Dynamic import of evaluation module (absolute path) + resolver
# -----------------------------------------------------------------------------

def resolve_eval_module(path: Path) -> Path:
    if path.exists():
        return path
    # Common alternates in the same directory
    alts = [
        path.with_name("01_hierarchical_predictor_evaluation_modules.py"),
        path.with_name("00_hierarchical_predictor_evaluation_modules.py"),
        path.with_name("01_hierarchical_evaluation_modules.py"),
    ]
    for a in alts:
        if a.exists():
            return a
    raise FileNotFoundError(f"Evaluation module not found: {path} (or alternates: {alts})")

def import_eval_module(path: Path):
    import importlib.util

    if not path.exists():
        raise FileNotFoundError(f"Evaluation module not found: {path}")

    spec = importlib.util.spec_from_file_location("hier_eval_predictors", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for module: {path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

def rebuild_pydantic_models_in_module(mod: Any, log_path: Path) -> None:
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
# Predictor leaf-path parsing
# -----------------------------------------------------------------------------

def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def short(s: str, n: int) -> str:
    s = s.replace("\n", " ").strip()
    return s[: n - 1] + "…" if len(s) > n else s

def load_predictor_leaf_paths(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Predictor leaf paths file not found: {path}")
    lines: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            if s.startswith("#"):
                continue
            lines.append(s)
    if not lines:
        raise ValueError(f"No predictor leaf paths found in: {path}")
    return lines

def parse_leaf_path_line(line: str) -> Tuple[Optional[str], List[str], str]:
    """
    Expected shape:
      BIO | A | B | C | LeafName
    Returns:
      (biopsychosocial_layer, segments, leaf_label)
    """
    parts = [p.strip() for p in line.split("|")]
    parts = [p for p in parts if p != ""]
    if not parts:
        return None, [], line.strip()

    layer = parts[0] if parts[0] in {"BIO", "PSYCHO", "SOCIAL"} else None
    segments = parts[1:] if layer is not None else parts
    leaf_label = segments[-1] if segments else (parts[-1] if parts else line.strip())
    return layer, segments, leaf_label

# -----------------------------------------------------------------------------
# Directories
# -----------------------------------------------------------------------------

def ensure_dirs(base: Path) -> Dict[str, Path]:
    base.mkdir(parents=True, exist_ok=True)
    dirs = {
        "base": base,
        "cache": base / "cache_evaluations",
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
    Patches Pydantic JSON schema to satisfy OpenAI Structured Outputs constraints.

    Guarantees:
    - Any dict that has "$ref" becomes ONLY {"$ref": "..."} (no siblings)
    - allOf with single $ref collapses to $ref
    - For objects: additionalProperties=false and required includes every remaining property key
      (forces Optional[...] fields to be present; prompt instructs to avoid nulls)
    - Removes defaults/titles/descriptions globally
    - Removes weight-bearing keys from schema entirely
    - Produces a pure-JSON deep copy at the end
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
# Output schema bundle (predictor-level)
# -----------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class OutputSchemaBundle:
    wrapper_model: Type[BaseModel]
    json_schema: Dict[str, Any]
    schema_name: str

def build_output_schema(mod: Any, log_path: Path) -> OutputSchemaBundle:
    PredictorEvaluation = mod.PredictorEvaluation  # type: ignore[attr-defined]

    class PredictorEvaluationWrapper(BaseModel):
        evaluation: PredictorEvaluation = Field(...)

    ns = {name: getattr(mod, name) for name in dir(mod)}
    try:
        PredictorEvaluationWrapper.model_rebuild(force=True, _types_namespace=ns)  # type: ignore[arg-type]
    except TypeError:
        PredictorEvaluationWrapper.model_rebuild(force=True)

    schema = PredictorEvaluationWrapper.model_json_schema()
    schema = patch_schema_for_structured_outputs(cast(Dict[str, Any], schema))

    schema_name = "predictor_evaluation_wrapper"
    log_and_print(log_path, f"Built + patched Structured Output schema once. name='{schema_name}'")
    return OutputSchemaBundle(wrapper_model=PredictorEvaluationWrapper, json_schema=schema, schema_name=schema_name)

def validate_schema_once(client: OpenAI, schema_bundle: OutputSchemaBundle, log_path: Path) -> None:
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
# Prompt building
# -----------------------------------------------------------------------------

def build_user_prompt(
    predictor_id: str,
    label: str,
    biopsychosocial_layer: Optional[str],
    full_path: str,
    segments: List[str],
) -> str:
    return USER_PROMPT_TEMPLATE.format(
        predictor_id=predictor_id,
        label=label,
        biopsychosocial_layer=(biopsychosocial_layer if biopsychosocial_layer is not None else "null"),
        full_path=short(full_path, MAX_CONTEXT_ITEM_CHARS),
        path_segments_json=json.dumps(segments, ensure_ascii=False),
        biopsychosocial_layer_or_null=(f"\"{biopsychosocial_layer}\"" if biopsychosocial_layer is not None else "null"),
    )

def ensure_prompt_fits(prompt: str, predictor_id: str, log_path: Path) -> str:
    if len(prompt) <= MAX_PREDICTOR_PROMPT_CHARS:
        return prompt
    # truncate from bottom
    lines = prompt.splitlines()
    while lines and len("\n".join(lines)) > MAX_PREDICTOR_PROMPT_CHARS:
        lines.pop()
    new_prompt = "\n".join(lines).rstrip() + "\n"
    log_and_print(log_path, f"WARNING: prompt too large for predictor={predictor_id}; truncated to len={len(new_prompt)} chars.")
    return new_prompt

# -----------------------------------------------------------------------------
# LLM evaluation logic (predictor-level)
# -----------------------------------------------------------------------------

def eval_predictor(
    client: OpenAI,
    schema_bundle: OutputSchemaBundle,
    predictor_id: str,
    label: str,
    biopsychosocial_layer: Optional[str],
    full_path: str,
    segments: List[str],
    retries: int,
    log_path: Path,
) -> Any:
    user_prompt = build_user_prompt(
        predictor_id=predictor_id,
        label=label,
        biopsychosocial_layer=biopsychosocial_layer,
        full_path=full_path,
        segments=segments,
    )
    user_prompt = ensure_prompt_fits(user_prompt, predictor_id=predictor_id, log_path=log_path)

    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            if attempt == 0:
                log_and_print(log_path, f"API CALL: predictor={predictor_id} retries={retries}")

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
            log_and_print(log_path, f"API ERROR (attempt {attempt+1}/{retries}) predictor={predictor_id}: {repr(e)}")
            backoff_sleep(attempt)

    raise RuntimeError(f"Failed after {retries} attempts. Last error: {last_err}") from last_err

# -----------------------------------------------------------------------------
# CSV cache IO
# -----------------------------------------------------------------------------

OUTPUT_COLUMNS = [
    "run_id",
    "model_name",
    "cached_at_utc",
    "predictor_id",
    "predictor_label",
    "biopsychosocial_layer",
    "predictor_hash",
    "full_path",
    "path_segments_json",
    "evaluation_json",
]

def load_cached_predictor_hashes_from_csv(csv_path: Path, log_path: Path) -> set[str]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(csv_path, usecols=["predictor_hash"], dtype={"predictor_hash": "string"})
        hs = set(df["predictor_hash"].dropna().astype(str).tolist())
        log_and_print(log_path, f"Loaded cached predictor_hashes from CSV: {len(hs)} ({human_path(csv_path)})")
        return hs
    except Exception as e:
        log_and_print(log_path, f"WARNING: failed reading cached predictor hashes from CSV: {repr(e)} ({human_path(csv_path)})")
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
        if "predictor_hash" not in df.columns:
            log_and_print(log_path, f"WARNING: cannot finalize; missing predictor_hash column in {human_path(csv_path)}")
            return

        before = len(df)
        df = df.drop_duplicates(subset=["predictor_hash"], keep="last")

        # stable sort by predictor_id
        if "predictor_id" in df.columns:
            df = df.sort_values(["predictor_id"])

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
            wide_rows.append({**base, **flat})

        wide_df = pd.DataFrame(wide_rows)
        wide_df.to_csv(to_csv, index=False, encoding="utf-8")
        log_and_print(log_path, f"Wrote wide CSV: {human_path(to_csv)} (rows={len(wide_df)} cols={len(wide_df.columns)})")
    except Exception as e:
        log_and_print(log_path, f"WARNING: build_wide_csv failed: {repr(e)}")

# -----------------------------------------------------------------------------
# Task iterator: ONE task per predictor if not cached
# -----------------------------------------------------------------------------

def iter_predictor_tasks(
    indices: List[int],
    leaf_lines: List[str],
    cached_hashes: set[str],
    overwrite: bool,
) -> Iterator[int]:
    for idx in indices:
        h = sha1_text(leaf_lines[idx])
        if overwrite or (h not in cached_hashes):
            yield idx

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--predictor-leaf-paths", type=str, default=str(PREDICTOR_LEAF_PATHS_TXT))
    ap.add_argument("--eval-module", type=str, default=str(EVAL_MODULE_PATH))
    ap.add_argument("--results-dir", type=str, default=str(RESULTS_DIR))

    ap.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    ap.add_argument("--retries", type=int, default=DEFAULT_RETRIES)

    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="Re-evaluate predictors even if cached in output CSV.")
    ap.add_argument("--no-json-cache", action="store_true", help="Disable per-predictor JSON cache files (cache_evaluations/).")

    # TEST MODE (predictor sampling)
    ap.add_argument("--test-n-predictors", type=int, default=0, help="Evaluate N random predictors (0 disables).")
    ap.add_argument("--test-seed", type=int, default=42, help="Random seed for --test-n-predictors.")

    # Output control
    ap.add_argument("--out-csv", type=str, default="", help="Optional explicit output CSV path. If empty, auto-named.")
    ap.add_argument("--run-tag", type=str, default="", help="Optional tag inserted into output filenames.")

    args = ap.parse_args()

    predictor_leaf_paths_path = Path(args.predictor_leaf_paths)
    eval_module_path = resolve_eval_module(Path(args.eval_module))
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
        base = "predictor_evaluations"
        if run_tag:
            base += f"_{run_tag}"
        if args.test_n_predictors and int(args.test_n_predictors) > 0:
            base += f"_TESTn{int(args.test_n_predictors)}_seed{int(args.test_seed)}"
        out_csv = dirs["tables"] / f"{base}.csv"

    wide_csv = out_csv.with_name(out_csv.stem + "_wide.csv")

    log_and_print(log_path, "==== RUN START (PREDICTOR-LEVEL) ====")
    log_and_print(log_path, f"run_id:                {run_id}")
    log_and_print(log_path, f"predictor_leaf_paths:   {human_path(predictor_leaf_paths_path)}")
    log_and_print(log_path, f"eval_module_path:       {human_path(eval_module_path)}")
    log_and_print(log_path, f"results_dir:            {human_path(results_dir)}")
    log_and_print(log_path, f"output_csv:             {human_path(out_csv)}")
    log_and_print(log_path, f"wide_csv:               {human_path(wide_csv)}")

    log_and_print(
        log_path,
        f"predictor_leaf_paths exists={predictor_leaf_paths_path.exists()} size_bytes={predictor_leaf_paths_path.stat().st_size if predictor_leaf_paths_path.exists() else 'NA'}",
    )
    log_and_print(
        log_path,
        f"eval_module exists={eval_module_path.exists()} size_bytes={eval_module_path.stat().st_size if eval_module_path.exists() else 'NA'}",
    )

    leaf_lines = load_predictor_leaf_paths(predictor_leaf_paths_path)
    log_and_print(log_path, f"Loaded predictor leaf paths: n={len(leaf_lines)} (sample='{short(leaf_lines[0], 120)}')")

    log_and_print(log_path, "Loading predictor evaluation module...")
    mod = import_eval_module(eval_module_path)
    assert hasattr(mod, "PredictorEvaluation"), "Evaluation module missing 'PredictorEvaluation'."
    assert hasattr(mod, "compute_overall_suitability"), "Evaluation module missing 'compute_overall_suitability'."
    rebuild_pydantic_models_in_module(mod, log_path=log_path)

    schema_bundle = build_output_schema(mod, log_path=log_path)

    # Select indices (test mode optional)
    all_indices = list(range(len(leaf_lines)))
    indices = all_indices
    if args.test_n_predictors and int(args.test_n_predictors) > 0 and len(all_indices) > int(args.test_n_predictors):
        rng = random.Random(int(args.test_seed))
        indices = rng.sample(all_indices, k=int(args.test_n_predictors))
        indices.sort()
        log_and_print(log_path, f"TEST MODE: selected {len(indices)} predictors (seed={args.test_seed})")

    if args.dry_run:
        log_and_print(log_path, "DRY RUN enabled; not calling OpenAI.")
        log_and_print(log_path, "==== RUN END (DRY RUN) ====")
        return 0

    cached_hashes = load_cached_predictor_hashes_from_csv(out_csv, log_path=log_path)
    log_and_print(log_path, f"Cached predictor_hashes (from output CSV): {len(cached_hashes)}")

    client = OpenAI()
    validate_schema_once(client, schema_bundle, log_path=log_path)

    write_lock = threading.Lock()
    completed = 0
    failed = 0

    def _run_one(idx: int) -> None:
        nonlocal completed, failed, cached_hashes

        line = leaf_lines[idx]
        phash = sha1_text(line)

        if (not args.overwrite) and (phash in cached_hashes):
            with write_lock:
                completed += 1
            return

        layer, segments, leaf_label = parse_leaf_path_line(line)
        # Disambiguated id: label + short hash suffix (stable, unique across duplicates)
        predictor_id = f"{leaf_label}::{phash[:8]}"

        ev = eval_predictor(
            client=client,
            schema_bundle=schema_bundle,
            predictor_id=predictor_id,
            label=leaf_label,
            biopsychosocial_layer=layer,
            full_path=line,
            segments=segments,
            retries=int(args.retries),
            log_path=log_path,
        )

        row = {
            "run_id": run_id,
            "model_name": MODEL_NAME,
            "cached_at_utc": utc_now_iso(),
            "predictor_id": predictor_id,
            "predictor_label": leaf_label,
            "biopsychosocial_layer": (layer if layer is not None else ""),
            "predictor_hash": phash,
            "full_path": line,
            "path_segments_json": json.dumps(segments, ensure_ascii=False),
            "evaluation_json": json.dumps(ev.model_dump(), ensure_ascii=False),
        }

        with write_lock:
            append_rows_to_csv(out_csv, [row])
            cached_hashes.add(phash)
            completed += 1
            if completed % 10 == 0:
                log_and_print(log_path, f"PROGRESS: completed={completed} failed={failed} cached={len(cached_hashes)}")

        if not args.no_json_cache:
            entry = {
                "run_id": run_id,
                "model_name": MODEL_NAME,
                "predictor_hash": phash,
                "predictor_id": predictor_id,
                "label": leaf_label,
                "biopsychosocial_layer": layer,
                "full_path": line,
                "path_segments": segments,
                "evaluation": ev.model_dump(),
                "cached_at_utc": utc_now_iso(),
            }
            cp = dirs["cache"] / f"predictor_{predictor_id}_{phash}.json"
            atomic_write_json(cp, entry)

    log_and_print(log_path, f"Starting ThreadPoolExecutor max_workers={args.max_workers} ...")

    task_iter = iter_predictor_tasks(
        indices=indices,
        leaf_lines=leaf_lines,
        cached_hashes=cached_hashes,
        overwrite=bool(args.overwrite),
    )

    in_flight = set()
    with ThreadPoolExecutor(max_workers=int(args.max_workers)) as ex:
        for _ in range(int(args.max_workers)):
            try:
                idx = next(task_iter)
            except StopIteration:
                break
            in_flight.add(ex.submit(_run_one, idx))

        while in_flight:
            done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for fut in done:
                try:
                    fut.result()
                except Exception as e:
                    with write_lock:
                        failed += 1
                        err_obj = {"error": repr(e), "created_at_utc": utc_now_iso()}
                        err_file = dirs["errors"] / f"error_{failed}_{int(time.time())}.json"
                        atomic_write_json(err_file, err_obj)
                        log_and_print(log_path, f"PREDICTOR FAILED ({failed}): {repr(e)} | saved: {human_path(err_file)}")

                try:
                    idx = next(task_iter)
                except StopIteration:
                    idx = None
                if idx is not None:
                    in_flight.add(ex.submit(_run_one, idx))

    log_and_print(log_path, f"Finished predictors. completed={completed} failed={failed}")

    finalize_csv_dedup_sort(out_csv, log_path=log_path)
    build_wide_csv(from_csv=out_csv, to_csv=wide_csv, log_path=log_path)

    log_and_print(log_path, "==== RUN END ====")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
