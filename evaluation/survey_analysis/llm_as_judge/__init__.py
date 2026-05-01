"""
LLM-as-judge package for the PHOENIX evaluation.

The judge consumes pairs of canonical (HCP, PHOENIX) outputs and emits
structured per-dimension signed A-vs-B comparisons. The runner unblinds
those into PHOENIX-vs-HCP scores.

* :mod:`dimensions`         — per-part dimension specifications.
* :mod:`output_schema`      — JSON schema the judge must emit.
* :mod:`openrouter_client`  — thin OpenAI-SDK wrapper for OpenRouter.
* :mod:`judge_runner`       — orchestration: blinding, retries, persistence.
* :mod:`pseudo_judge`       — local stand-in that emits plausible scores.
"""

from .dimensions import (
    PROMPT_VERSION,
    Dimension,
    DIMENSIONS_BY_PART,
    PART_TITLES,
    dimensions_for,
)
from .output_schema import (
    DimensionComparison,
    JudgeResponse,
    parse_judge_json,
)

__all__ = [
    "PROMPT_VERSION",
    "Dimension",
    "DIMENSIONS_BY_PART",
    "PART_TITLES",
    "dimensions_for",
    "DimensionComparison",
    "JudgeResponse",
    "parse_judge_json",
]
