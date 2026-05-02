"""
End-to-end judge runner: blinding, prompt rendering, API call, JSON parsing,
and persistence into the long-format CSV.

Design
------
For each (case, part, run) cell the runner makes **two independent LLM
calls** — one for each output — evaluating them on separate, non-comparative
requests.  The judge rates ONE anonymous output at a time on a bipolar -10..+10
absolute quality scale per dimension.  The entity (phoenix / hcp) is identified only
after unblinding for the statistical analysis.

Concurrency
-----------
All evaluation tasks are dispatched through ThreadPoolExecutor so that
multiple (case, part, run, source) jobs run in parallel.  The default
max_workers is 20.  CSV writes are serialised through a threading.Lock.

Long-format CSV columns
-----------------------
case_id, part, dimension, judge_run, entity, quality_score, source_label,
confidence, justification, prompt_version, model, timestamp
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from analysis.shared.survey_paths import (
    JUDGMENTS_DIR,
    judgments_csv,
    raw_judgment_path,
)

from .dimensions import (
    DIMENSIONS_BY_PART,
    PART_TITLES,
    PROMPT_VERSION,
    QUALITY_SCALE_ANCHORS,
    Dimension,
    dimensions_for,
)
from .openrouter_client import (
    DEFAULT_MODEL,
    JSON_RETRY_INSTRUCTION,
    OpenRouterClient,
    OpenRouterError,
)
from .output_schema import JudgeResponse, parse_judge_json
from .prompts import load_prompt, load_system_prompt
from .pseudo_judge import generate_pseudo_response, serialize_pseudo_response

logger = logging.getLogger(__name__)

PSEUDO_MODEL_TAG: str = "pseudo-judge-absolute-quality"

JUDGMENTS_CSV_HEADER: List[str] = [
    "case_id",
    "part",
    "dimension",
    "judge_run",
    "entity",
    "quality_score",
    "source_label",
    "confidence",
    "justification",
    "prompt_version",
    "model",
    "timestamp",
]


# ──────────────────────────────────────────────────────────────────────────────
# Blinding
# ──────────────────────────────────────────────────────────────────────────────

_INT32_MAX: int = 2_147_483_647   # Gemini requires seeds in the INT32 range


def _blind_seed(case_id: str, part: str, run_idx: int) -> int:
    """Deterministic seed for A/B label assignment (clamped to INT32 for Gemini)."""
    h = hashlib.sha256(f"{case_id}|{part}|{run_idx}".encode("utf-8")).hexdigest()
    raw = int(h[:16], 16)
    return raw % (_INT32_MAX + 1)   # keeps determinism, fits in INT32


def _call_seed(case_id: str, part: str, run_idx: int, source_label: str) -> int:
    """Deterministic generation seed for one blinded judge call."""
    h = hashlib.sha256(
        f"{case_id}|{part}|{run_idx}|{source_label}".encode("utf-8")
    ).hexdigest()
    raw = int(h[:16], 16)
    return raw % (_INT32_MAX + 1)


def assign_blind_labels(
    case_id: str,
    part: str,
    run_idx: int,
) -> Dict[str, str]:
    """
    Decide which source maps to label A vs B.

    Returns e.g. ``{"A": "phoenix", "B": "hcp"}``.  Deterministic so the
    unblinding can be replayed from the CSV alone.
    """
    seed = _blind_seed(case_id, part, run_idx)
    a_is_phoenix = bool(seed & 1)
    return {
        "A": "phoenix" if a_is_phoenix else "hcp",
        "B": "hcp" if a_is_phoenix else "phoenix",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Prompt rendering (single-output absolute quality)
# ──────────────────────────────────────────────────────────────────────────────

def _render_dimensions_block(part: str) -> str:
    lines: List[str] = [
        "Absolute quality scale:",
        *[f"  {anchor}" for anchor in QUALITY_SCALE_ANCHORS],
        "",
        "Dimension-specific criteria:",
        "",
    ]
    for dim in dimensions_for(part):
        lines.append(f"### {dim.key}: {dim.display_label}")
        lines.append(f"Goal: {dim.goal_description}")
        lines.append(f"Why this matters: {dim.rationale}")
        lines.append("Quality anchors:")
        lines.append(dim.anchor_block())
        lines.append("")
    return "\n".join(lines).rstrip()


def _format_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def render_user_prompt(
    part: str,
    case_context: Dict[str, Any],
    the_output: Dict[str, Any],
) -> str:
    """
    Render the per-part user prompt for ONE anonymous output.

    ``the_output`` is the canonical output to be rated.  The judge never
    sees both outputs at once.
    """
    template = load_prompt(part)
    placeholders: Dict[str, str] = {
        # Case context placeholders
        "vignette": case_context.get("vignette", "(no vignette provided)"),
        "case_notes_json": _format_json(case_context.get("case_notes", {})),
        "standardized_symptoms_json": _format_json(
            case_context.get("standardized_symptoms", [])
        ),
        "standardized_treatment_options_json": _format_json(
            case_context.get("standardized_treatment_options", [])
        ),
        "treatment_targets_json": _format_json(case_context.get("treatment_targets", [])),
        "candidate_ema_items_json": _format_json(
            case_context.get("candidate_ema_items", [])
        ),
        "primary_problem": str(case_context.get("primary_problem", "(not provided)")),
        "treatment_goal": str(case_context.get("treatment_goal", "(not provided)")),
        "barrier": str(case_context.get("barrier", "(not provided)")),
        "coping_strategy": str(case_context.get("coping_strategy", "(not provided)")),
        "operationalisation_json": _format_json(
            case_context.get("operationalisation", {})
        ),
        "initial_model_json": _format_json(case_context.get("initial_model", {})),
        "network_summary_json": _format_json(case_context.get("network_summary", {})),
        "ema_summary_json": _format_json(case_context.get("ema_summary", {})),
        "treatment_options_json": _format_json(
            case_context.get("treatment_options", [])
        ),
        "ranking_json": _format_json(case_context.get("ranking", {})),
        "ema_item_context_json": _format_json(case_context.get("ema_item_context", {})),
        "assigned_hapa_phase": str(case_context.get("hapa_phase", "(unknown)")),
        # Single-output placeholders for the absolute-quality design.
        "the_output_json": _format_json(the_output),
        # Legacy comparative placeholders kept so old prompt templates still
        # render without KeyError; they will be replaced with the output
        # content or empty strings.
        "output_a_json": _format_json(the_output),   # legacy compat
        "output_b_json": "{}",                          # legacy compat (empty)
        "dimensions_block": _render_dimensions_block(part),
    }
    rendered = template
    for key, value in placeholders.items():
        rendered = rendered.replace("{{" + key + "}}", str(value))
    return rendered


def build_messages(
    part: str,
    case_context: Dict[str, Any],
    the_output: Dict[str, Any],
) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": load_system_prompt()},
        {
            "role": "user",
            "content": render_user_prompt(part, case_context, the_output),
        },
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Persistence helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(JUDGMENTS_CSV_HEADER)
        return
    with path.open(newline="", encoding="utf-8") as f:
        existing = next(csv.reader(f), [])
    if existing != JUDGMENTS_CSV_HEADER:
        raise ValueError(
            f"Existing judgments CSV at {path} has an incompatible header. "
            "Archive or delete it before running the absolute-quality judge."
        )


def _append_rows(path: Path, rows: Iterable[List[Any]], lock: threading.Lock) -> None:
    with lock:
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for row in rows:
                writer.writerow(row)


def _save_raw_response(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Runner config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class JudgeRunConfig:
    """Configuration for one full judging run."""

    cases: List[str]
    parts: List[str]
    n_runs: int = 3
    mode: str = "pseudo"        # "pseudo" or "real"
    model: str = DEFAULT_MODEL
    n_retries: int = 3
    json_retries: int = 1
    max_workers: int = 20       # thread-pool size for parallel evaluation
    out_csv: Optional[Path] = None
    raw_dir_root: Optional[Path] = None
    case_context_provider: Optional[Any] = None
    """
    Optional callable ``(case_id, part) -> dict`` that fills prompt
    placeholders.  The pipeline orchestrator wires up the real provider.
    """


def _now_iso() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# ──────────────────────────────────────────────────────────────────────────────
# Single evaluation task
# ──────────────────────────────────────────────────────────────────────────────

def _evaluate_single(
    *,
    case_id: str,
    part: str,
    run_idx: int,
    source_label: str,          # "A" or "B"
    entity: str,                # "phoenix" or "hcp"
    output: Dict[str, Any],
    case_context: Dict[str, Any],
    expected_dims: List[str],
    client: Optional[OpenRouterClient],
    mode: str,
    n_retries: int,
    json_retries: int,
    seed: int,
    blinding: Dict[str, str],
    raw_root: Path,
) -> Tuple[JudgeResponse, str, Dict[str, Any]]:
    """
    Evaluate one output (identified only as source_label during the call).

    Returns (JudgeResponse, model_used, raw_payload).
    """
    if mode == "pseudo":
        # Pseudo mode: generate absolute quality ratings for this entity
        response = generate_pseudo_response(
            case_id=case_id,
            part=part,
            judge_run=run_idx,
            a_is_phoenix=(blinding["A"] == "phoenix"),
            entity=entity,
        )
        raw_payload = {"pseudo": True, "source_label": source_label, "entity": entity}
        model_used = PSEUDO_MODEL_TAG
    else:
        assert client is not None
        messages = build_messages(part, case_context, output)
        response, raw_payload, model_used = _call_real_judge(
            client=client,
            messages=messages,
            expected_keys=expected_dims,
            n_retries=n_retries,
            json_retries=json_retries,
            seed=seed,
        )

    # Save raw response
    rp = raw_root / part / f"case_{case_id}_run_{run_idx}_{source_label}.json"
    _save_raw_response(rp, {
        "case_id": case_id,
        "part": part,
        "judge_run": run_idx,
        "source_label": source_label,
        "entity": entity,
        "blinding": blinding,
        "model": model_used,
        "prompt_version": PROMPT_VERSION,
        "response": raw_payload,
    })

    return response, model_used, raw_payload


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_judge(
    *,
    hcp_outputs: Dict[str, Dict[str, Any]],
    system_outputs: Dict[str, Dict[str, Any]],
    config: JudgeRunConfig,
) -> Path:
    """
    Run the judging loop and append rows to ``judgments_long.csv``.

    For each (case, part, run) cell the judge is called independently for
    the PHOENIX output and for the HCP output.  Both calls are dispatched
    concurrently through ThreadPoolExecutor.

    Parameters
    ----------
    hcp_outputs     ``{case_id: {part: canonical}}``
    system_outputs  ``{case_id: {part: canonical}}``

    Returns
    -------
    Path to the long-format CSV.
    """
    csv_path = config.out_csv or judgments_csv()
    _ensure_csv(csv_path)

    raw_root = config.raw_dir_root or JUDGMENTS_DIR / "raw"
    csv_lock = threading.Lock()

    client: Optional[OpenRouterClient] = None
    if config.mode == "real":
        client = OpenRouterClient(model=config.model)
    elif config.mode != "pseudo":
        raise ValueError(f"Unknown judge mode {config.mode!r}; expected pseudo|real")

    # Build the flat list of evaluation tasks.
    # Each task is one output evaluated by the judge.
    tasks: List[Dict[str, Any]] = []
    for case_id in config.cases:
        case_hcp = hcp_outputs.get(case_id, {})
        case_sys = system_outputs.get(case_id, {})
        for part in config.parts:
            if part not in DIMENSIONS_BY_PART:
                raise ValueError(f"Unknown part {part!r}")
            expected_dims = [d.key for d in dimensions_for(part)]
            for run_idx in range(config.n_runs):
                blinding = assign_blind_labels(case_id, part, run_idx)
                ctx = (
                    config.case_context_provider(case_id, part)
                    if config.case_context_provider else {}
                )
                for source_label in ("A", "B"):
                    entity = blinding[source_label]
                    output = (
                        case_sys.get(part, {})
                        if entity == "phoenix"
                        else case_hcp.get(part, {})
                    )
                    tasks.append({
                        "case_id": case_id,
                        "part": part,
                        "run_idx": run_idx,
                        "source_label": source_label,
                        "entity": entity,
                        "output": output,
                        "ctx": ctx,
                        "expected_dims": expected_dims,
                        "blinding": blinding,
                        "seed": _call_seed(case_id, part, run_idx, source_label),
                    })

    total = len(tasks)
    logger.info(
        "Starting judge run: mode=%s cases=%d parts=%d n_runs=%d "
        "tasks=%d max_workers=%d",
        config.mode, len(config.cases), len(config.parts),
        config.n_runs, total, config.max_workers,
    )

    completed_count = 0

    def _run_task(task: Dict[str, Any]) -> None:
        nonlocal completed_count
        case_id = task["case_id"]
        part = task["part"]
        run_idx = task["run_idx"]
        source_label = task["source_label"]
        entity = task["entity"]

        response, model_used, _ = _evaluate_single(
            case_id=case_id,
            part=part,
            run_idx=run_idx,
            source_label=source_label,
            entity=entity,
            output=task["output"],
            case_context=task["ctx"],
            expected_dims=task["expected_dims"],
            client=client,
            mode=config.mode,
            n_retries=config.n_retries,
            json_retries=config.json_retries,
            seed=task["seed"],
            blinding=task["blinding"],
            raw_root=raw_root,
        )

        ts = _now_iso()
        rows: List[List[Any]] = []
        for rating in response.ratings:
            rows.append([
                case_id,
                part,
                rating.dimension,
                run_idx,
                entity,
                int(rating.score),
                source_label,
                int(rating.confidence),
                rating.justification,
                PROMPT_VERSION,
                model_used,
                ts,
            ])

        _append_rows(csv_path, rows, csv_lock)
        with csv_lock:
            completed_count += 1
            if completed_count % 20 == 0 or completed_count == total:
                logger.info(
                    "Judge progress: %d / %d tasks (%s/%s run=%d %s=%s)",
                    completed_count, total,
                    case_id, part, run_idx, source_label, entity,
                )

    with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
        futures = {pool.submit(_run_task, task): task for task in tasks}
        errors = []
        for future in as_completed(futures):
            task = futures[future]
            try:
                future.result()
            except Exception as exc:
                logger.error(
                    "Judge task FAILED %s/%s run=%d %s: %s",
                    task["case_id"], task["part"], task["run_idx"],
                    task["source_label"], exc,
                )
                errors.append((task["case_id"], task["part"], task["run_idx"], str(exc)))

    if errors:
        logger.warning(
            "%d judge task(s) failed (see logs above). "
            "CSV may be incomplete.", len(errors)
        )
    else:
        logger.info("Judge run complete. Long-format CSV: %s", csv_path)

    return csv_path


def _call_real_judge(
    client: OpenRouterClient,
    messages: List[Dict[str, str]],
    expected_keys: List[str],
    n_retries: int,
    json_retries: int,
    seed: Optional[int] = None,
) -> Tuple[JudgeResponse, Dict[str, Any], str]:
    """Call the real judge with JSON-repair retry."""
    last_text: Optional[str] = None
    raw_payload: Dict[str, Any] = {}
    last_msgs = list(messages)
    for attempt in range(json_retries + 1):
        completion = client.chat_with_retry(last_msgs, n_retries=n_retries, seed=seed)
        last_text = completion.text
        raw_payload = {"text": completion.text, "raw": completion.raw}
        try:
            response = parse_judge_json(completion.text, expected_dimensions=expected_keys)
            return response, raw_payload, completion.model
        except ValueError as exc:
            logger.warning("Judge JSON parse failed: %s", exc)
            if attempt >= json_retries:
                raise
            last_msgs = list(messages) + [
                {"role": "assistant", "content": completion.text or ""},
                {"role": "user", "content": JSON_RETRY_INSTRUCTION},
            ]
    raise OpenRouterError(
        f"Judge produced unparseable output after {json_retries + 1} attempts; "
        f"last text: {last_text!r}"
    )


__all__ = [
    "JudgeRunConfig",
    "JUDGMENTS_CSV_HEADER",
    "PSEUDO_MODEL_TAG",
    "assign_blind_labels",
    "build_messages",
    "render_user_prompt",
    "run_judge",
]
