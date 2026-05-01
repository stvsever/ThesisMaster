"""
Local stand-in for the absolute-quality LLM judge.

Pseudo mode generates plausible per-output quality scores on the bipolar
−10..+10 semantic differential scale used by the judge.  0 = acceptable
baseline; positive = above baseline; negative = below baseline.

PHOENIX and HCP outputs receive scores drawn from different distributions
to simulate realistic PHOENIX-advantage effects.

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


# Ground-truth mean quality for each entity on the bipolar −10..+10 scale.
# Values rescaled from prior 1–5 estimates via: new = (old − 3) × 5
# PHOENIX_MEAN > HCP_MEAN on most dimensions; difference represents the
# effect that the statistical model should recover.

PHOENIX_MEAN_QUALITY: Dict[str, Dict[str, float]] = {
    "part1": {
        "task_adherence_label_format": 6.0,
        "complaint_coverage": 7.5,
        "symptom_boundary_validity": 5.5,
        "granularity_resolution": 4.0,
        "nonredundancy_discriminability": 5.0,
        "clinical_interoperability": 6.5,
        "ema_measurability": 6.0,
    },
    "part2": {
        "task_adherence_label_format": 6.5,
        "modifiability_actionability": 6.0,
        "symptom_relevance": 5.5,
        "causal_plausibility": 7.0,
        "daily_ema_feasibility": 7.5,
        "symptom_option_separation": 6.0,
        "option_diversity_complementarity": 3.0,
        "label_precision": 5.0,
    },
    "part3": {
        "ranking_validity_completeness": 8.5,
        "network_weight_alignment": 7.0,
        "current_state_integration": 6.0,
        "edge_direction_interpretation": 6.5,
        "top_target_defensibility": 5.5,
        "modifiability_feasibility_weighting": 4.0,
        "rank_order_coherence": 6.0,
    },
    "part4": {
        "valid_candidate_selection": 8.0,
        "target_item_mapping_accuracy": 6.5,
        "coverage_balance": 7.5,
        "measurement_concreteness": 6.5,
        "directness_specificity": 6.0,
        "dynamic_informativeness": 5.0,
        "monitoring_burden_parsimony": 4.0,
        "feedback_value_for_coaching": 6.0,
    },
    "part5": {
        "message_format_direct_address": 5.5,
        "treatment_goal_alignment": 6.5,
        "barrier_responsiveness": 7.0,
        "action_specificity_feasibility": 7.5,
        "behaviour_change_potential": 6.5,
        "tone_empathy_professionalism": 3.5,
        "mobile_concision_readability": 5.0,
        "personalisation_specificity": 7.0,
        "clinical_safety_nonjudgment": 6.0,
    },
}

HCP_MEAN_QUALITY: Dict[str, Dict[str, float]] = {
    "part1": {
        "task_adherence_label_format": 4.5,
        "complaint_coverage": 4.5,
        "symptom_boundary_validity": 4.0,
        "granularity_resolution": 3.5,
        "nonredundancy_discriminability": 4.0,
        "clinical_interoperability": 3.5,
        "ema_measurability": 4.5,
    },
    "part2": {
        "task_adherence_label_format": 4.0,
        "modifiability_actionability": 3.5,
        "symptom_relevance": 4.5,
        "causal_plausibility": 4.0,
        "daily_ema_feasibility": 3.5,
        "symptom_option_separation": 4.0,
        "option_diversity_complementarity": 4.0,
        "label_precision": 4.5,
    },
    "part3": {
        "ranking_validity_completeness": 5.5,
        "network_weight_alignment": 3.0,
        "current_state_integration": 3.5,
        "edge_direction_interpretation": 4.0,
        "top_target_defensibility": 4.5,
        "modifiability_feasibility_weighting": 4.5,
        "rank_order_coherence": 4.0,
    },
    "part4": {
        "valid_candidate_selection": 4.5,
        "target_item_mapping_accuracy": 3.5,
        "coverage_balance": 3.5,
        "measurement_concreteness": 4.0,
        "directness_specificity": 4.0,
        "dynamic_informativeness": 3.5,
        "monitoring_burden_parsimony": 4.5,
        "feedback_value_for_coaching": 3.5,
    },
    "part5": {
        "message_format_direct_address": 4.5,
        "treatment_goal_alignment": 4.5,
        "barrier_responsiveness": 4.0,
        "action_specificity_feasibility": 3.5,
        "behaviour_change_potential": 4.0,
        "tone_empathy_professionalism": 4.5,
        "mobile_concision_readability": 4.5,
        "personalisation_specificity": 3.5,
        "clinical_safety_nonjudgment": 5.0,
    },
}

# Variance components on the −10..+10 scale (rescaled from 1–5 estimates × 5)
SIGMA_CASE: float = 1.50
SIGMA_RUN: float = 0.75
SIGMA_RESID: float = 3.25


def _hash_seed(*parts: Any) -> int:
    s = "|".join(str(p) for p in parts)
    return int.from_bytes(s.encode("utf-8")[:8].ljust(8, b"\0"), "little")


def _confidence_from_score(score: int) -> int:
    """Map bipolar −10..+10 quality score to rater confidence (1–5).

    Extreme scores are typically easier to defend (high confidence);
    near-zero scores sit in an ambiguous region (lower confidence).
    """
    abs_s = abs(score)
    if abs_s >= 8:
        return 5   # clearly outstanding or catastrophic — unambiguous
    if abs_s >= 5:
        return 4   # strong positive or negative — well-supported
    if abs_s >= 2:
        return 3   # moderate — some ambiguity
    return 2       # near-zero — close to the acceptable baseline, most uncertain


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
        mu = means.get(dim.key, 0.0)
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
