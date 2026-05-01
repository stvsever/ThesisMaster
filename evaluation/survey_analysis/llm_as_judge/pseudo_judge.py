"""
Local stand-in for the LLM judge.

Generates plausible per-(case, part, dimension, judge_run) ratings without
any API call so that the full pipeline (parsing -> judge -> stats -> plots)
can be exercised end-to-end on pseudodata.

The injected effects per dimension are documented in
:data:`GROUND_TRUTH_EFFECTS` so the analysis stage can be checked against
known truth.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .dimensions import (
    DIMENSIONS_BY_PART,
    LIKERT_MAX,
    LIKERT_MIN,
    LIKERT_NEUTRAL,
    PROMPT_VERSION,
)
from .output_schema import DimensionRating, JudgeResponse

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Ground-truth pseudo effects (PHOENIX - HCP, in Likert points)
#
# Positive => PHOENIX rated higher than HCP on that dimension.
# Magnitudes chosen so that the analysis recovers a realistic mix of
# significantly better (~+0.6), equivalent (small), and slightly worse (-0.3)
# dimensions.
# ──────────────────────────────────────────────────────────────────────────────

GROUND_TRUTH_EFFECTS: Dict[str, Dict[str, float]] = {
    "part1": {
        "clinical_accuracy":         +0.20,
        "construct_interoperability":+0.60,
        "resolution_preservation":   -0.10,
        "behavioural_specificity":   +0.05,
        "internal_consistency":      +0.40,
        "completeness":              +0.55,
        "conciseness_redundancy":    -0.30,
    },
    "part2": {
        "clinical_appropriateness":  +0.10,
        "network_validity":          +0.55,
        "ema_feasibility":           +0.20,
        "predictor_diversity":       +0.65,
        "measurement_specificity":   +0.40,
        "intervention_potential":    +0.05,
        "construct_coverage":        -0.20,
    },
    "part3": {
        "top_target_appropriateness":+0.10,
        "evidence_alignment":        +0.50,
        "rank_coherence":            +0.60,
        "network_impact_awareness":  +0.45,
        "monitoring_integration":    +0.30,
        "modifiability_weighting":   -0.20,
    },
    "part4": {
        "adaptive_reasoning":        +0.55,
        "target_alignment":          +0.40,
        "personalisation":           +0.60,
        "measurement_quality":       +0.10,
        "parsimony":                 -0.25,
        "theoretical_coherence":     +0.05,
    },
    "part5": {
        "hapa_phase_appropriateness":+0.45,
        "behavioural_change_potential":+0.50,
        "personalisation_specificity":+0.65,
        "professional_tone":         -0.10,
        "empathy_warmth":            -0.30,
        "clarity_actionability":     +0.30,
        "message_appropriateness_length": -0.05,
    },
}

DEFAULT_HCP_BASELINE: float = 5.2     # HCP mean rating across dimensions
SIGMA_CASE: float = 0.40              # case-level random intercept SD
SIGMA_RUN: float = 0.20               # judge_run-level intercept SD
SIGMA_RESID: float = 0.55             # within-cell residual SD


def _hash_seed(*parts: Any) -> int:
    """Stable seed from heterogeneous arguments."""
    s = "|".join(str(p) for p in parts)
    return int.from_bytes(s.encode("utf-8")[:8].ljust(8, b"\0"), "little")


@dataclass
class PseudoJudgeContext:
    """All the state needed to generate one response."""
    case_id: str
    part: str
    judge_run: int
    seed: int


def _draw_rating(
    base: float,
    effect: float,
    is_phoenix: bool,
    rng: np.random.Generator,
    case_intercept: float,
    run_intercept: float,
) -> int:
    side_shift = (effect / 2.0) * (1.0 if is_phoenix else -1.0)
    raw = base + case_intercept + run_intercept + side_shift + rng.normal(0, SIGMA_RESID)
    return int(np.clip(round(raw), LIKERT_MIN, LIKERT_MAX))


def generate_pseudo_response(
    case_id: str,
    part: str,
    judge_run: int,
    a_is_phoenix: bool,
) -> JudgeResponse:
    """
    Return a :class:`JudgeResponse` for one (case, part, judge_run).

    Both ratings are drawn from the same generative model with deterministic
    seed so multiple runs are reproducible.
    """
    if part not in DIMENSIONS_BY_PART:
        raise ValueError(f"Unknown part {part!r}")
    seed = _hash_seed(case_id, part, judge_run, "pseudo-v1")
    rng = np.random.default_rng(seed)
    case_intercept = rng.normal(0.0, SIGMA_CASE)
    run_intercept = rng.normal(0.0, SIGMA_RUN)

    ratings: List[DimensionRating] = []
    for dim in DIMENSIONS_BY_PART[part]:
        eff = GROUND_TRUTH_EFFECTS[part].get(dim.key, 0.0)
        rating_a = _draw_rating(
            DEFAULT_HCP_BASELINE, eff,
            is_phoenix=a_is_phoenix,
            rng=rng,
            case_intercept=case_intercept,
            run_intercept=run_intercept,
        )
        rating_b = _draw_rating(
            DEFAULT_HCP_BASELINE, eff,
            is_phoenix=not a_is_phoenix,
            rng=rng,
            case_intercept=case_intercept,
            run_intercept=run_intercept,
        )
        ratings.append(DimensionRating(
            dimension=dim.key,
            rating_a=rating_a,
            rating_b=rating_b,
            justification=(
                f"pseudo-judge: simulated rating with effect {eff:+.2f} "
                f"on dimension {dim.key}."
            ),
        ))

    extra: Dict[str, Any] = {}
    if part == "part5":
        # For the pseudo judge we just classify both messages with the
        # case-deterministic phase to keep accuracy high but realistic.
        phases = ["pre_intentional", "intentional", "action", "maintenance"]
        case_phase = phases[hash(case_id) % len(phases)]
        # 80% chance the judge classifies correctly.
        def _classify() -> str:
            return case_phase if rng.random() < 0.8 else rng.choice(phases)
        extra["hapa_phase_a"] = _classify()
        extra["hapa_phase_b"] = _classify()

    return JudgeResponse(ratings=ratings, extra=extra)


def serialize_pseudo_response(resp: JudgeResponse) -> str:
    """Render a pseudo response as a JSON string for caching alongside real runs."""
    return json.dumps({
        "ratings": [r.to_dict() for r in resp.ratings],
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
