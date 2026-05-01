"""Deterministic system-output fixture for pipeline testing.

This is not a thesis result and not a replacement for the PHOENIX engine.
It exists so the full HCP/system/judge/statistics pipeline can be tested
before the real PHOENIX run is available.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

try:
    from .paths import CASE_INPUTS_JSON, REPO_ROOT, SYSTEM_OUTPUTS_JSON, ensure_dirs
except ImportError:  # direct script execution from this folder
    from paths import CASE_INPUTS_JSON, REPO_ROOT, SYSTEM_OUTPUTS_JSON, ensure_dirs  # type: ignore

SURVEY_ANALYSIS_DIR = REPO_ROOT / "evaluation" / "survey_analysis"
if str(SURVEY_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(SURVEY_ANALYSIS_DIR))

from parsing.canonical_schemas import canonical_for_judge  # noqa: E402


def _short_sentence(text: str, max_words: int = 18) -> str:
    words = re.findall(r"[\wÀ-ÿ-]+", str(text), flags=re.UNICODE)
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).rstrip()


def _score_options(edges: Iterable[Mapping[str, Any]]) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for edge in edges:
        option_id = str(edge.get("from_option_id", "")).strip()
        if not option_id:
            continue
        weight = float(edge.get("weight", 0.0))
        # Both protective and risk edges indicate clinical leverage; absolute
        # strength is the transparent baseline priority signal.
        scores[option_id] = scores.get(option_id, 0.0) + abs(weight)
    return scores


def _rank_treatment_options(case: Mapping[str, Any]) -> List[Dict[str, Any]]:
    options = case.get("part3", {}).get("treatment_options", [])
    scores = _score_options(case.get("part3", {}).get("network_edges", []))
    ranked = sorted(
        options,
        key=lambda item: (
            -scores.get(str(item.get("option_id", "")), 0.0),
            str(item.get("option_id", "")),
        ),
    )
    return [
        {"rank": idx, "option_id": str(item.get("option_id", ""))}
        for idx, item in enumerate(ranked, start=1)
    ]


def _token_set(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[\wÀ-ÿ-]+", str(text), flags=re.UNICODE)
        if len(token) > 2
    }


def _select_ema_items(case: Mapping[str, Any]) -> List[str]:
    targets = case.get("part4", {}).get("treatment_targets", [])
    items = case.get("part4", {}).get("candidate_ema_items", [])
    selected: List[str] = []
    used: set[str] = set()
    for target in targets[:3]:
        target_tokens = _token_set(str(target))
        ranked: List[Tuple[int, str]] = []
        for item in items:
            label = str(item.get("label", ""))
            if label in used:
                continue
            overlap = len(target_tokens.intersection(_token_set(label)))
            ranked.append((-overlap, label))
        for _neg_overlap, label in sorted(ranked)[:2]:
            if label and label not in used:
                selected.append(label)
                used.add(label)
    if len(selected) < 6:
        for item in items:
            label = str(item.get("label", ""))
            if label and label not in used:
                selected.append(label)
                used.add(label)
            if len(selected) == 6:
                break
    return selected[:6]


def build_rule_based_outputs(case_inputs: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Build a deterministic, canonical-compatible fixture output bundle."""
    outputs: Dict[str, Dict[str, Any]] = {}
    for case_id, case in case_inputs.get("cases", {}).items():
        symptoms = case.get("part2", {}).get("standardized_symptoms", [])
        options = case.get("part3", {}).get("treatment_options", [])
        p5 = case.get("part5", {})
        target = str(p5.get("treatment_goal", "")).strip()
        barrier = str(p5.get("barrier", "")).strip()
        coping = str(p5.get("coping_strategy", "")).strip()
        message = (
            f"Vandaag focus je op {_short_sentence(target, 16).lower()}. "
            f"De hindernis is {_short_sentence(barrier, 16).lower()}; maak de stap daarom klein en concreet. "
            f"Start met {_short_sentence(coping, 18).lower()}."
        )
        raw_parts = {
            "part1": {"items": [{"label": str(s)} for s in symptoms[:6]]},
            "part2": {"items": [{"label": str(o.get("label", ""))} for o in options[:5]]},
            "part3": {"ranking": _rank_treatment_options(case)},
            "part4": {"selected_options": _select_ema_items(case)},
            "part5": {"message": message},
        }
        outputs[str(case_id)] = {
            part: canonical_for_judge(part, raw_parts.get(part, {}))
            for part in ("part1", "part2", "part3", "part4", "part5")
        }
    return outputs


def write_rule_based_outputs(
    case_inputs_path: Path = CASE_INPUTS_JSON,
    output_path: Path = SYSTEM_OUTPUTS_JSON,
) -> Path:
    """Write deterministic fixture outputs to the canonical system-output path."""
    ensure_dirs()
    case_inputs = json.loads(case_inputs_path.read_text(encoding="utf-8"))
    outputs = build_rule_based_outputs(case_inputs)
    output_path.write_text(
        json.dumps(outputs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


__all__ = ["build_rule_based_outputs", "write_rule_based_outputs"]
