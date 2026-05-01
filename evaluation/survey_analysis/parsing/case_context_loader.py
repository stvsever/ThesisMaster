"""
Load per-case context shown to both the HCPs and PHOENIX.

The LLM judge must receive the same case inputs for both anonymous outputs.
This loader expects a JSON file shaped as:

{
  "C01": {
    "vignette": "...",
    "standardized_symptoms": ["..."],
    "standardized_treatment_options": [{"option_id": "BO-1", "label": "..."}],
    "network_summary": {...},
    "ema_summary": {...},
    "treatment_targets": ["..."],
    "candidate_ema_items": ["..."],
    "primary_problem": "...",
    "treatment_goal": "...",
    "barrier": "...",
    "coping_strategy": "...",
    "hapa_phase": "intentional"
  }
}

Missing keys are tolerated; the prompt renderer inserts explicit
"not provided" placeholders.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict

from analysis.shared.survey_paths import RAW_DIR

DEFAULT_CASE_CONTEXTS_PATH: Path = RAW_DIR / "case_contexts.json"


def load_case_contexts(path: Path | str = DEFAULT_CASE_CONTEXTS_PATH) -> Dict[str, Dict[str, Any]]:
    """Load case contexts from JSON; return an empty dict if the file is absent."""
    path = Path(path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected case-context JSON object at {path}, got {type(data)!r}")
    return {
        str(case_id): payload if isinstance(payload, dict) else {}
        for case_id, payload in data.items()
    }


def make_case_context_provider(
    contexts: Dict[str, Dict[str, Any]],
) -> Callable[[str, str], Dict[str, Any]]:
    """
    Return the callable expected by ``JudgeRunConfig``.

    The part argument is accepted so this can later become part-specific
    without changing the runner interface.
    """
    def provider(case_id: str, part: str) -> Dict[str, Any]:
        ctx = dict(contexts.get(case_id, {}))
        if part == "part2" and "operationalisation" not in ctx:
            ctx["operationalisation"] = {"items": ctx.get("standardized_symptoms", [])}
        if part == "part4" and "ranking" not in ctx:
            ctx["ranking"] = {"ranking": ctx.get("standardized_treatment_options", [])}
        return ctx

    return provider


__all__ = [
    "DEFAULT_CASE_CONTEXTS_PATH",
    "load_case_contexts",
    "make_case_context_provider",
]
