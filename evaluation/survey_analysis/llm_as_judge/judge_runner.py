"""
End-to-end judge runner: blinding, prompt rendering, API call (or pseudo),
JSON parsing, persistence into the long-format CSV.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import logging
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
    SIGNED_SCALE_ANCHORS,
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

PSEUDO_MODEL_TAG: str = "pseudo-judge-v1"


JUDGMENTS_CSV_HEADER: List[str] = [
    "case_id",
    "part",
    "dimension",
    "judge_run",
    "score",
    "raw_score_a_over_b",
    "source_a",
    "source_b",
    "winner_blind",
    "winner_source",
    "confidence",
    "justification",
    "prompt_version",
    "model",
    "timestamp",
]


# ──────────────────────────────────────────────────────────────────────────────
# Blinding
# ──────────────────────────────────────────────────────────────────────────────

def _blind_seed(case_id: str, part: str, run_idx: int) -> int:
    """Deterministic seed for A/B label assignment."""
    h = hashlib.sha256(f"{case_id}|{part}|{run_idx}".encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def assign_blind_labels(
    case_id: str,
    part: str,
    run_idx: int,
) -> Dict[str, str]:
    """
    Decide which source goes to label A vs B.

    Returns a dict like ``{"A": "phoenix", "B": "hcp"}``. Deterministic on
    ``(case_id, part, run_idx)`` so the unblinding can always be replayed.
    """
    seed = _blind_seed(case_id, part, run_idx)
    a_is_phoenix = bool(seed & 1)
    return {
        "A": "phoenix" if a_is_phoenix else "hcp",
        "B": "hcp" if a_is_phoenix else "phoenix",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Prompt rendering
# ──────────────────────────────────────────────────────────────────────────────

def _render_dimensions_block(part: str) -> str:
    lines: List[str] = [
        "Global signed comparative scale:",
        *[f"- {anchor}" for anchor in SIGNED_SCALE_ANCHORS],
        "",
        "Dimension-specific criteria:",
        "",
    ]
    for dim in dimensions_for(part):
        lines.append(f"### {dim.key}: {dim.display_label}")
        lines.append(f"Goal: {dim.goal_description}")
        lines.append(f"Why this matters: {dim.rationale}")
        lines.append("Comparative examples:")
        lines.append(dim.anchor_block())
        lines.append("")
    return "\n".join(lines).rstrip()


def _format_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def render_user_prompt(
    part: str,
    case_context: Dict[str, Any],
    output_a: Dict[str, Any],
    output_b: Dict[str, Any],
) -> str:
    """
    Render the per-part user prompt.

    ``case_context`` is part-aware: it should provide the placeholders
    referenced in the prompt template for this part (vignette, ema_summary
    etc). Missing placeholders are replaced with ``"(not provided)"``.
    """
    template = load_prompt(part)
    placeholders: Dict[str, str] = {
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
        "updated_model_json": _format_json(case_context.get("updated_model", {})),
        "assigned_hapa_phase": str(case_context.get("hapa_phase", "(unknown)")),
        "output_a_json": _format_json(output_a),
        "output_b_json": _format_json(output_b),
        "dimensions_block": _render_dimensions_block(part),
    }
    rendered = template
    for key, value in placeholders.items():
        rendered = rendered.replace("{{" + key + "}}", str(value))
    return rendered


def build_messages(
    part: str,
    case_context: Dict[str, Any],
    output_a: Dict[str, Any],
    output_b: Dict[str, Any],
) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": load_system_prompt()},
        {
            "role": "user",
            "content": render_user_prompt(part, case_context, output_a, output_b),
        },
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Persistence helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(JUDGMENTS_CSV_HEADER)
        return
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        existing = next(reader, [])
    if existing != JUDGMENTS_CSV_HEADER:
        raise ValueError(
            f"Existing judgments CSV at {path} has an incompatible header. "
            "Archive or delete it before running the v2 signed judge."
        )


def _append_rows(path: Path, rows: Iterable[List[Any]]) -> None:
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
    n_runs: int = 5
    mode: str = "pseudo"        # "pseudo" or "real"
    model: str = DEFAULT_MODEL
    n_retries: int = 3
    json_retries: int = 1
    out_csv: Optional[Path] = None
    raw_dir_root: Optional[Path] = None
    case_context_provider: Optional[Any] = None
    """
    Optional callable ``(case_id, part) -> dict`` used to fill the prompt
    placeholders. If absent, an empty placeholder dict is used. The
    pipeline orchestrator wires up the real provider.
    """


def _now_iso() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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

    Parameters
    ----------
    hcp_outputs : ``{case_id: {part: canonical}}``
    system_outputs : ``{case_id: {part: canonical}}``

    Returns
    -------
    Path
        Path to the long-format CSV that was written.
    """
    csv_path = config.out_csv or judgments_csv()
    _ensure_csv(csv_path)

    raw_root = config.raw_dir_root or JUDGMENTS_DIR / "raw"

    client: Optional[OpenRouterClient] = None
    if config.mode == "real":
        client = OpenRouterClient(model=config.model)
    elif config.mode != "pseudo":
        raise ValueError(f"Unknown judge mode {config.mode!r}; expected pseudo|real")

    total_calls = len(config.cases) * len(config.parts) * config.n_runs
    logger.info(
        "Starting judge run: mode=%s cases=%d parts=%d n_runs=%d total=%d",
        config.mode, len(config.cases), len(config.parts),
        config.n_runs, total_calls,
    )
    call_idx = 0

    for case_id in config.cases:
        case_hcp = hcp_outputs.get(case_id, {})
        case_sys = system_outputs.get(case_id, {})
        for part in config.parts:
            if part not in DIMENSIONS_BY_PART:
                raise ValueError(f"Unknown part {part!r}")
            expected_keys = [d.key for d in dimensions_for(part)]
            for run_idx in range(config.n_runs):
                call_idx += 1
                blinding = assign_blind_labels(case_id, part, run_idx)
                a_is_phoenix = blinding["A"] == "phoenix"
                hcp_payload = case_hcp.get(part, {})
                sys_payload = case_sys.get(part, {})
                output_a = sys_payload if a_is_phoenix else hcp_payload
                output_b = hcp_payload if a_is_phoenix else sys_payload

                ctx = (
                    config.case_context_provider(case_id, part)
                    if config.case_context_provider
                    else {}
                )

                logger.info(
                    "[%d/%d] case=%s part=%s run=%d A=%s",
                    call_idx, total_calls, case_id, part, run_idx, blinding["A"],
                )

                if config.mode == "pseudo":
                    response = generate_pseudo_response(
                        case_id=case_id,
                        part=part,
                        judge_run=run_idx,
                        a_is_phoenix=a_is_phoenix,
                    )
                    raw_payload = json.loads(serialize_pseudo_response(response))
                    model_used = PSEUDO_MODEL_TAG
                else:
                    assert client is not None
                    messages = build_messages(part, ctx, output_a, output_b)
                    response, raw_payload, model_used = _call_real_judge(
                        client=client,
                        messages=messages,
                        expected_keys=expected_keys,
                        n_retries=config.n_retries,
                        json_retries=config.json_retries,
                        seed=_blind_seed(case_id, part, run_idx),
                    )

                # Persist raw response.
                rp = raw_judgment_path(part, case_id, run_idx)
                if config.raw_dir_root is not None:
                    rp = (config.raw_dir_root / part /
                          f"case_{case_id}_run_{run_idx}.json")
                _save_raw_response(rp, {
                    "case_id": case_id,
                    "part": part,
                    "judge_run": run_idx,
                    "blinding": blinding,
                    "model": model_used,
                    "prompt_version": PROMPT_VERSION,
                    "response": raw_payload,
                })

                # Map A/B back to a PHOENIX-over-HCP signed score and append rows.
                rows: List[List[Any]] = []
                ts = _now_iso()
                for comp in response.comparisons:
                    raw_score = int(comp.score)
                    phoenix_score = raw_score if blinding["A"] == "phoenix" else -raw_score
                    if phoenix_score > 0:
                        winner_source = "phoenix"
                    elif phoenix_score < 0:
                        winner_source = "hcp"
                    else:
                        winner_source = "tie"
                    rows.append([
                        case_id,
                        part,
                        comp.dimension,
                        run_idx,
                        phoenix_score,
                        raw_score,
                        blinding["A"],
                        blinding["B"],
                        comp.winner,
                        winner_source,
                        int(comp.confidence),
                        comp.justification,
                        PROMPT_VERSION,
                        model_used,
                        ts,
                    ])
                _append_rows(csv_path, rows)

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
    """Call the real judge; if JSON parse fails, re-prompt up to json_retries."""
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
            # Re-prompt the model with the parse error context.
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
