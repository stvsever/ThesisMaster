"""
JSON schema and parser for absolute per-output quality ratings.

Design (absolute-quality)
-----------------------------
The judge rates ONE output at a time on a 1..5 absolute Likert quality
scale per dimension.  Comparisons between PHOENIX and HCP are never made
inside the judge call; instead the entity predictor (phoenix vs hcp) is
estimated in the downstream mixed-model analysis.

Expected JSON shape
-------------------
{
  "ratings": [
    {
      "dimension": "complaint_coverage",
      "score": 4,
      "confidence": 4,
      "justification": "Covers all three major complaint domains."
    }
  ],
  "extra": {}
}

Scale anchors
-------------
1 = Poor      — fails the criterion significantly
2 = Below average — partially meets the criterion
3 = Acceptable — meets the criterion adequately
4 = Good      — clearly meets the criterion well
5 = Excellent  — exceptional on this criterion
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Absolute quality scale constants
QUALITY_MIN: int = 1
QUALITY_MAX: int = 5
QUALITY_NEUTRAL: int = 3


@dataclass
class DimensionRating:
    """Absolute quality rating for one dimension of one output."""

    dimension: str
    score: int
    confidence: int
    justification: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "score": int(self.score),
            "confidence": int(self.confidence),
            "justification": str(self.justification),
        }


@dataclass
class JudgeResponse:
    """Parsed response: one absolute rating per requested dimension."""

    ratings: List[DimensionRating] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ratings": [r.to_dict() for r in self.ratings],
            "extra": dict(self.extra),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}", flags=re.MULTILINE)


def _strip_to_json(text: str) -> str:
    """Recover a JSON object from a chatty or fenced LLM response."""
    if text is None:
        return "{}"
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
    match = _JSON_OBJECT_RE.search(s)
    if match:
        return match.group(0)
    return s


def _clamp_quality(score: Any) -> int:
    try:
        v = int(round(float(score)))
    except (TypeError, ValueError):
        v = QUALITY_NEUTRAL
    return max(QUALITY_MIN, min(QUALITY_MAX, v))


def _clamp_confidence(confidence: Any) -> int:
    try:
        v = int(round(float(confidence)))
    except (TypeError, ValueError):
        v = 3
    return max(1, min(5, v))


def parse_judge_json(
    raw_text: str,
    expected_dimensions: Optional[List[str]] = None,
) -> JudgeResponse:
    """
    Parse an absolute-quality judge response.

    Accepts the current ``ratings`` schema and, as a migration aid, the
    older ``comparisons`` signed schema (converts by mapping the sign to
    quality: positive → >=3, negative → <=3).

    Missing expected dimensions are filled with QUALITY_NEUTRAL (3) so the
    analysis stage always has a rectangular data frame.
    """
    text = _strip_to_json(raw_text or "")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Could not parse judge JSON: {exc}; got: {raw_text!r}"
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(f"Judge response is not a JSON object: {raw_text!r}")

    # Primary schema: "ratings" list
    raw_ratings: list = data.get("ratings") or []

    # Migration path from old signed comparative schema ("comparisons")
    if not raw_ratings and isinstance(data.get("comparisons"), list):
        for entry in data["comparisons"]:
            if not isinstance(entry, dict):
                continue
            try:
                signed = float(entry.get("score", 0))
            except (TypeError, ValueError):
                signed = 0.0
            # Map signed -9..+9 to absolute 1..5:
            # -9 → 1 (very poor), 0 → 3 (neutral), +9 → 5 (excellent)
            absolute = round(3 + (signed / 9.0) * 2)
            raw_ratings.append({
                "dimension": entry.get("dimension", ""),
                "score": absolute,
                "confidence": entry.get("confidence", 3),
                "justification": entry.get("justification", ""),
            })

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
            score=_clamp_quality(entry.get("score", QUALITY_NEUTRAL)),
            confidence=_clamp_confidence(entry.get("confidence", 3)),
            justification=str(entry.get("justification", "")).strip(),
        )

    if expected_dimensions is not None:
        for dim_key in expected_dimensions:
            if dim_key not in seen:
                seen[dim_key] = DimensionRating(
                    dimension=dim_key,
                    score=QUALITY_NEUTRAL,
                    confidence=1,
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
    "QUALITY_MIN",
    "QUALITY_MAX",
    "QUALITY_NEUTRAL",
    "parse_judge_json",
]
