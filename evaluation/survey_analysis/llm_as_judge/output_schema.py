"""
JSON schema for the judge response.

Pure dataclasses (no pydantic dep) so the package is portable.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .dimensions import LIKERT_MAX, LIKERT_MIN


@dataclass
class DimensionRating:
    dimension: str
    rating_a: int
    rating_b: int
    justification: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "rating_a": int(self.rating_a),
            "rating_b": int(self.rating_b),
            "justification": str(self.justification),
        }


@dataclass
class JudgeResponse:
    ratings: List[DimensionRating] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ratings": [r.to_dict() for r in self.ratings],
            "extra": dict(self.extra),
        }


_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}", flags=re.MULTILINE)


def _strip_to_json(text: str) -> str:
    """
    Best-effort recovery of a JSON object from a chatty LLM response.

    Strips markdown code fences and isolates the largest top-level
    ``{...}`` substring.
    """
    if text is None:
        return "{}"
    s = text.strip()
    # Drop fenced blocks like ```json\n...\n```.
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", s)
        s = re.sub(r"\n```\s*$", "", s)
    match = _JSON_OBJECT_RE.search(s)
    if match:
        return match.group(0)
    return s


def _clamp(rating: Any) -> int:
    try:
        v = int(round(float(rating)))
    except (TypeError, ValueError):
        v = LIKERT_MIN
    return max(LIKERT_MIN, min(LIKERT_MAX, v))


def parse_judge_json(
    raw_text: str,
    expected_dimensions: Optional[List[str]] = None,
) -> JudgeResponse:
    """
    Parse the judge's raw text into :class:`JudgeResponse`.

    Raises :class:`ValueError` if the response cannot be parsed at all.
    Tolerates extra fields and clamps ratings to ``[LIKERT_MIN, LIKERT_MAX]``.

    Parameters
    ----------
    expected_dimensions
        If provided, the parser ensures every expected dimension is present;
        missing dimensions get rating 4 (neutral) with a placeholder
        justification, so the analysis stage never fails on partial output.
    """
    text = _strip_to_json(raw_text or "")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse judge JSON: {exc}; got: {raw_text!r}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Judge response is not a JSON object: {raw_text!r}")

    raw_ratings = data.get("ratings", [])
    if not isinstance(raw_ratings, list):
        raw_ratings = []
    seen: Dict[str, DimensionRating] = {}
    for entry in raw_ratings:
        if not isinstance(entry, dict):
            continue
        key = str(entry.get("dimension", "")).strip()
        if not key:
            continue
        seen[key] = DimensionRating(
            dimension=key,
            rating_a=_clamp(entry.get("rating_a", LIKERT_MIN)),
            rating_b=_clamp(entry.get("rating_b", LIKERT_MIN)),
            justification=str(entry.get("justification", "")).strip(),
        )

    if expected_dimensions is not None:
        for dim_key in expected_dimensions:
            if dim_key not in seen:
                seen[dim_key] = DimensionRating(
                    dimension=dim_key,
                    rating_a=4,
                    rating_b=4,
                    justification="missing-from-judge-response",
                )
        ordered = [seen[k] for k in expected_dimensions]
    else:
        ordered = list(seen.values())

    extra = data.get("extra")
    if not isinstance(extra, dict):
        extra = {}
    return JudgeResponse(ratings=ordered, extra=extra)


__all__ = [
    "DimensionRating",
    "JudgeResponse",
    "parse_judge_json",
]
