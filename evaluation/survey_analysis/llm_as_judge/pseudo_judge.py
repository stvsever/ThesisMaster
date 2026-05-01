"""
Local stand-in for the absolute-quality LLM judge.

Pseudo mode generates plausible per-output quality scores on the 1..5
absolute Likert scale used by the judge.  PHOENIX and HCP outputs
receive scores drawn from different distributions to simulate realistic
PHOENIX-advantage effects.

This lets the full pipeline run without an OpenRouter API key.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from .dimensions import DIMENSIONS_BY_PART, PROMPT_VERSION
from .output_schema import DimensionRating, JudgeResponse, QUALITY_MAX, QUALITY_MIN

logger = logging.getLogger(__name__)


# Ground-truth mean quality for each entity on the 1..5 scale.
# PHOENIX_MEAN and HCP_MEAN are the true underlying quality levels.
# The difference (PHOENIX - HCP) represents the effect that the statistical
# model should recover.

PHOENIX_MEAN_QUALITY: Dict[str, Dict[str, float]] = {
    "part1": {
        "task_adherence_label_format": 4.2,
        "complaint_coverage": 4.5,
        "symptom_boundary_validity": 4.1,
        "granularity_resolution": 3.8,
        "nonredundancy_discriminability": 4.0,
        "clinical_interoperability": 4.3,
        "ema_measurability": 4.2,
    },
    "part2": {
        "task_adherence_label_format": 4.3,
        "modifiability_actionability": 4.2,
        "symptom_relevance": 4.1,
        "causal_plausibility": 4.4,
        "daily_ema_feasibility": 4.5,
        "symptom_option_separation": 4.2,
        "option_diversity_complementarity": 3.6,
        "label_precision": 4.0,
    },
    "part3": {
        "ranking_validity_completeness": 4.7,
        "network_weight_alignment": 4.4,
        "current_state_integration": 4.2,
        "edge_direction_interpretation": 4.3,
        "top_target_defensibility": 4.1,
        "modifiability_feasibility_weighting": 3.8,
        "rank_order_coherence": 4.2,
    },
    "part4": {
        "valid_candidate_selection": 4.6,
        "target_item_mapping_accuracy": 4.3,
        "coverage_balance": 4.5,
        "measurement_concreteness": 4.3,
        "directness_specificity": 4.2,
        "dynamic_informativeness": 4.0,
        "monitoring_burden_parsimony": 3.8,
        "feedback_value_for_coaching": 4.2,
    },
    "part5": {
        "message_format_direct_address": 4.1,
        "treatment_goal_alignment": 4.3,
        "barrier_responsiveness": 4.4,
        "action_specificity_feasibility": 4.5,
        "behaviour_change_potential": 4.3,
        "tone_empathy_professionalism": 3.7,
        "mobile_concision_readability": 4.0,
        "personalisation_specificity": 4.4,
        "clinical_safety_nonjudgment": 4.2,
    },
}

HCP_MEAN_QUALITY: Dict[str, Dict[str, float]] = {
    "part1": {
        "task_adherence_label_format": 3.9,
        "complaint_coverage": 3.9,
        "symptom_boundary_validity": 3.8,
        "granularity_resolution": 3.7,
        "nonredundancy_discriminability": 3.8,
        "clinical_interoperability": 3.7,
        "ema_measurability": 3.9,
    },
    "part2": {
        "task_adherence_label_format": 3.8,
        "modifiability_actionability": 3.7,
        "symptom_relevance": 3.9,
        "causal_plausibility": 3.8,
        "daily_ema_feasibility": 3.7,
        "symptom_option_separation": 3.8,
        "option_diversity_complementarity": 3.8,
        "label_precision": 3.9,
    },
    "part3": {
        "ranking_validity_completeness": 4.1,
        "network_weight_alignment": 3.6,
        "current_state_integration": 3.7,
        "edge_direction_interpretation": 3.8,
        "top_target_defensibility": 3.9,
        "modifiability_feasibility_weighting": 3.9,
        "rank_order_coherence": 3.8,
    },
    "part4": {
        "valid_candidate_selection": 3.9,
        "target_item_mapping_accuracy": 3.7,
        "coverage_balance": 3.7,
        "measurement_concreteness": 3.8,
        "directness_specificity": 3.8,
        "dynamic_informativeness": 3.7,
        "monitoring_burden_parsimony": 3.9,
        "feedback_value_for_coaching": 3.7,
    },
    "part5": {
        "message_format_direct_address": 3.9,
        "treatment_goal_alignment": 3.9,
        "barrier_responsiveness": 3.8,
        "action_specificity_feasibility": 3.7,
        "behaviour_change_potential": 3.8,
        "tone_empathy_professionalism": 3.9,
        "mobile_concision_readability": 3.9,
        "personalisation_specificity": 3.7,
        "clinical_safety_nonjudgment": 4.0,
    },
}

SIGMA_CASE: float = 0.30
SIGMA_RUN: float = 0.15
SIGMA_RESID: float = 0.65


def _hash_seed(*parts: Any) -> int:
    s = "|".join(str(p) for p in parts)
    return int.from_bytes(s.encode("utf-8")[:8].ljust(8, b"\0"), "little")


def _confidence_from_score(score: int) -> int:
    """Map absolute quality score to rater confidence."""
    if score <= 1 or score >= 5:
        return 5
    if score in (2, 4):
        return 4
    return 3


def generate_pseudo_response(
    case_id: str,
    part: str,
    judge_run: int,
    a_is_phoenix: bool,  # kept for API compat; entity is inferred from context
    entity: str = "phoenix",  # "phoenix" or "hcp"
) -> JudgeResponse:
    """
    Return a deterministic pseudo absolute-quality response for one output.

    ``entity`` drives the expected quality mean.
    """
    if part not in DIMENSIONS_BY_PART:
        raise ValueError(f"Unknown part {part!r}")

    seed = _hash_seed(case_id, part, judge_run, entity, "pseudo-absolute-quality")
    rng = np.random.default_rng(seed)
    case_intercept = rng.normal(0.0, SIGMA_CASE)
    run_intercept = rng.normal(0.0, SIGMA_RUN)

    means = (
        PHOENIX_MEAN_QUALITY.get(part, {})
        if entity == "phoenix"
        else HCP_MEAN_QUALITY.get(part, {})
    )

    ratings: List[DimensionRating] = []
    for dim in DIMENSIONS_BY_PART[part]:
        mu = means.get(dim.key, 3.5)
        raw = mu + case_intercept + run_intercept + rng.normal(0, SIGMA_RESID)
        score = int(np.clip(round(raw), QUALITY_MIN, QUALITY_MAX))
        ratings.append(DimensionRating(
            dimension=dim.key,
            score=score,
            confidence=_confidence_from_score(score),
            justification=(
                f"pseudo-judge ({entity}): simulated quality "
                f"score around μ={mu:.1f}."
            ),
        ))

    return JudgeResponse(ratings=ratings)


def serialize_pseudo_response(resp: JudgeResponse) -> str:
    """Render a pseudo response as JSON (for caching alongside real runs)."""
    return json.dumps({
        "ratings": [r.to_dict() for r in resp.ratings],
        "_pseudo": True,
        "_prompt_version": PROMPT_VERSION,
    }, ensure_ascii=False, indent=2)


__all__ = [
    "PHOENIX_MEAN_QUALITY",
    "HCP_MEAN_QUALITY",
    "generate_pseudo_response",
    "serialize_pseudo_response",
]
