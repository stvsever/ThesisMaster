"""
Local stand-in for the signed comparative LLM judge.

Pseudo mode generates plausible PHOENIX-over-HCP effects on the same -9..+9
scale used by the real judge. It lets the full parse -> judge -> analysis ->
visualisation pipeline run without an OpenRouter API key.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from .dimensions import DIMENSIONS_BY_PART, PROMPT_VERSION, SCALE_MAX, SCALE_MIN
from .output_schema import DimensionComparison, JudgeResponse

logger = logging.getLogger(__name__)


# Ground-truth pseudo effects (PHOENIX - HCP) in signed preference points.
# Positive values mean the pseudo judge should prefer PHOENIX on average.
GROUND_TRUTH_EFFECTS: Dict[str, Dict[str, float]] = {
    "part1": {
        "complaint_coverage": +1.2,
        "symptom_boundary_validity": +0.5,
        "granularity_resolution": -0.2,
        "nonredundancy_discriminability": +0.4,
        "clinical_interoperability": +1.4,
        "ema_measurability": +0.8,
    },
    "part2": {
        "modifiability_actionability": +0.6,
        "symptom_relevance": +0.3,
        "causal_plausibility": +1.1,
        "daily_ema_feasibility": +1.4,
        "symptom_option_separation": +0.7,
        "option_diversity_complementarity": -0.4,
        "label_precision": +0.2,
    },
    "part3": {
        "network_weight_alignment": +1.6,
        "current_state_integration": +1.0,
        "edge_direction_interpretation": +0.9,
        "top_target_defensibility": +0.5,
        "modifiability_feasibility_weighting": -0.3,
        "rank_order_coherence": +0.7,
    },
    "part4": {
        "target_item_mapping_accuracy": +0.9,
        "coverage_balance": +1.3,
        "measurement_concreteness": +0.8,
        "directness_specificity": +0.6,
        "dynamic_informativeness": +0.4,
        "monitoring_burden_parsimony": -0.2,
        "feedback_value_for_coaching": +1.0,
    },
    "part5": {
        "treatment_goal_alignment": +0.7,
        "barrier_responsiveness": +0.9,
        "action_specificity_feasibility": +1.2,
        "behaviour_change_potential": +1.1,
        "tone_empathy_professionalism": -0.5,
        "mobile_concision_readability": +0.1,
        "personalisation_specificity": +1.3,
        "clinical_safety_nonjudgment": +0.2,
    },
}

SIGMA_CASE: float = 0.45
SIGMA_RUN: float = 0.25
SIGMA_RESID: float = 1.55


def _hash_seed(*parts: Any) -> int:
    s = "|".join(str(p) for p in parts)
    return int.from_bytes(s.encode("utf-8")[:8].ljust(8, b"\0"), "little")


@dataclass
class PseudoJudgeContext:
    case_id: str
    part: str
    judge_run: int
    seed: int


def _winner(score: int) -> str:
    if score > 0:
        return "A"
    if score < 0:
        return "B"
    return "TIE"


def _confidence(abs_score: int) -> int:
    if abs_score >= 7:
        return 5
    if abs_score >= 5:
        return 4
    if abs_score >= 2:
        return 3
    return 2


def generate_pseudo_response(
    case_id: str,
    part: str,
    judge_run: int,
    a_is_phoenix: bool,
) -> JudgeResponse:
    """Return a deterministic pseudo response for one blinded comparison."""
    if part not in DIMENSIONS_BY_PART:
        raise ValueError(f"Unknown part {part!r}")
    seed = _hash_seed(case_id, part, judge_run, "pseudo-v2-signed")
    rng = np.random.default_rng(seed)
    case_intercept = rng.normal(0.0, SIGMA_CASE)
    run_intercept = rng.normal(0.0, SIGMA_RUN)

    comparisons: List[DimensionComparison] = []
    for dim in DIMENSIONS_BY_PART[part]:
        phoenix_minus_hcp = GROUND_TRUTH_EFFECTS[part].get(dim.key, 0.0)
        a_minus_b_center = phoenix_minus_hcp if a_is_phoenix else -phoenix_minus_hcp
        raw = a_minus_b_center + case_intercept + run_intercept + rng.normal(0, SIGMA_RESID)
        score = int(np.clip(round(raw), SCALE_MIN, SCALE_MAX))
        comparisons.append(DimensionComparison(
            dimension=dim.key,
            score=score,
            winner=_winner(score),
            confidence=_confidence(abs(score)),
            justification=(
                f"pseudo-judge: simulated signed comparison with "
                f"PHOENIX-HCP effect {phoenix_minus_hcp:+.1f}."
            ),
        ))

    extra: Dict[str, Any] = {}
    if part == "part5":
        phases = ["pre_intentional", "intentional", "action", "maintenance"]
        case_phase = phases[_hash_seed(case_id, "phase") % len(phases)]

        def classify() -> str:
            return case_phase if rng.random() < 0.8 else str(rng.choice(phases))

        extra["hapa_phase_a"] = classify()
        extra["hapa_phase_b"] = classify()

    return JudgeResponse(comparisons=comparisons, extra=extra)


def serialize_pseudo_response(resp: JudgeResponse) -> str:
    """Render a pseudo response as JSON for caching alongside real runs."""
    return json.dumps({
        "comparisons": [c.to_dict() for c in resp.comparisons],
        "extra": resp.extra,
        "_pseudo": True,
        "_prompt_version": PROMPT_VERSION,
    }, ensure_ascii=False, indent=2)


__all__ = [
    "GROUND_TRUTH_EFFECTS",
    "PseudoJudgeContext",
    "generate_pseudo_response",
    "serialize_pseudo_response",
]
