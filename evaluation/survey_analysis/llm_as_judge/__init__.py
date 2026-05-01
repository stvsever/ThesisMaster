"""
LLM-as-judge package for the PHOENIX evaluation.

The judge consumes pairs of canonical (HCP, PHOENIX) outputs and emits
structured per-dimension Likert ratings. The implementation is split into:

* :mod:`dimensions`         — per-part dimension specifications.
* :mod:`output_schema`      — JSON schema the judge must emit.
* :mod:`openrouter_client`  — thin OpenAI-SDK wrapper for OpenRouter.
* :mod:`judge_runner`       — orchestration: blinding, retries, persistence.
* :mod:`pseudo_judge`       — local stand-in that emits plausible ratings.
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
]
