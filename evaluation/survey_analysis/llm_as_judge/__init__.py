"""
LLM-as-judge package for the PHOENIX evaluation (absolute-quality design).

The judge rates each output independently on a 1–5 absolute quality scale per
dimension.  Comparisons (PHOENIX vs HCP) are made in the downstream mixed-
model analysis, not inside the judge call.

Modules
-------
* :mod:`dimensions`        — per-part dimension specifications
* :mod:`output_schema`     — JSON schema the judge must emit 
* :mod:`openrouter_client` — thin OpenAI-SDK wrapper for OpenRouter
* :mod:`judge_runner`      — orchestration: blinding, retries, persistence
* :mod:`pseudo_judge`      — deterministic stand-in that emits plausible scores
"""

from .dimensions import (
    PROMPT_VERSION,
    Dimension,
    DIMENSIONS_BY_PART,
    PART_TITLES,
    dimensions_for,
)
from .output_schema import (
    DimensionRating,
    JudgeResponse,
    parse_judge_json,
    QUALITY_MIN,
    QUALITY_MAX,
    QUALITY_NEUTRAL,
)

__all__ = [
    "PROMPT_VERSION",
    "Dimension",
    "DIMENSIONS_BY_PART",
    "PART_TITLES",
    "dimensions_for",
    "DimensionRating",
    "JudgeResponse",
    "parse_judge_json",
    "QUALITY_MIN",
    "QUALITY_MAX",
    "QUALITY_NEUTRAL",
]
