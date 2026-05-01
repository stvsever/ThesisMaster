"""
JSON schema and parser for signed A-vs-B judge responses.

The expected strict JSON shape is:

{
  "comparisons": [
    {
      "dimension": "complaint_coverage",
      "score": 4,
      "winner": "A",
      "confidence": 4,
      "justification": "A covers sleep and social withdrawal while B misses social withdrawal."
    }
  ],
  "extra": {}
}

``score`` is signed A-over-B on the -9..+9 scale. The runner later unblinds
it into a PHOENIX-over-HCP score.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .dimensions import SCALE_MAX, SCALE_MIN, SCALE_NEUTRAL


@dataclass
class DimensionComparison:
    dimension: str
    score: int
    winner: str
    confidence: int
    justification: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "score": int(self.score),
            "winner": self.winner,
            "confidence": int(self.confidence),
            "justification": str(self.justification),
        }


@dataclass
class JudgeResponse:
    comparisons: List[DimensionComparison] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "comparisons": [c.to_dict() for c in self.comparisons],
            "extra": dict(self.extra),
        }


_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}", flags=re.MULTILINE)


def _strip_to_json(text: str) -> str:
    """Recover a JSON object from a chatty or fenced LLM response."""
    if text is None:
        return "{}"
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", s)
        s = re.sub(r"\n```\s*$", "", s)
    match = _JSON_OBJECT_RE.search(s)
    if match:
        return match.group(0)
    return s


def _clamp_score(score: Any) -> int:
    try:
        v = int(round(float(score)))
    except (TypeError, ValueError):
        v = SCALE_NEUTRAL
    return max(SCALE_MIN, min(SCALE_MAX, v))


def _clamp_confidence(confidence: Any) -> int:
    try:
        v = int(round(float(confidence)))
    except (TypeError, ValueError):
        v = 3
    return max(1, min(5, v))


def _winner_from_score(score: int, raw_winner: Any = None) -> str:
    winner = str(raw_winner or "").strip().upper()
    if winner in {"A", "B", "TIE"}:
        return winner
    if score > 0:
        return "A"
    if score < 0:
        return "B"
    return "TIE"


def parse_judge_json(
    raw_text: str,
    expected_dimensions: Optional[List[str]] = None,
) -> JudgeResponse:
    """
    Parse the judge response.

    The parser accepts the current ``comparisons`` schema and, as a migration
    aid, the older ``ratings`` schema by converting ``rating_a - rating_b``
    into a signed comparative score. Missing expected dimensions are filled
    with a neutral comparison so the analysis stage remains rectangular.
    """
    text = _strip_to_json(raw_text or "")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse judge JSON: {exc}; got: {raw_text!r}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Judge response is not a JSON object: {raw_text!r}")

    raw_comparisons = data.get("comparisons")
    if raw_comparisons is None and isinstance(data.get("ratings"), list):
        raw_comparisons = []
        for entry in data.get("ratings", []):
            if not isinstance(entry, dict):
                continue
            try:
                # Migration path from the old 1..7 independent ratings:
                # each absolute rating step becomes three signed preference points.
                converted = (float(entry.get("rating_a", 4)) - float(entry.get("rating_b", 4))) * 3
            except (TypeError, ValueError):
                converted = 0
            raw_comparisons.append({
                "dimension": entry.get("dimension", ""),
                "score": converted,
                "winner": None,
                "confidence": 3,
                "justification": entry.get("justification", ""),
            })
    if not isinstance(raw_comparisons, list):
        raw_comparisons = []

    seen: Dict[str, DimensionComparison] = {}
    for entry in raw_comparisons:
        if not isinstance(entry, dict):
            continue
        key = str(entry.get("dimension", "")).strip()
        if not key:
            continue
        score = _clamp_score(entry.get("score", SCALE_NEUTRAL))
        seen[key] = DimensionComparison(
            dimension=key,
            score=score,
            winner=_winner_from_score(score, entry.get("winner")),
            confidence=_clamp_confidence(entry.get("confidence", 3)),
            justification=str(entry.get("justification", "")).strip(),
        )

    if expected_dimensions is not None:
        for dim_key in expected_dimensions:
            if dim_key not in seen:
                seen[dim_key] = DimensionComparison(
                    dimension=dim_key,
                    score=0,
                    winner="TIE",
                    confidence=1,
                    justification="missing-from-judge-response",
                )
        ordered = [seen[k] for k in expected_dimensions]
    else:
        ordered = list(seen.values())

    extra = data.get("extra")
    if not isinstance(extra, dict):
        extra = {}
    return JudgeResponse(comparisons=ordered, extra=extra)


__all__ = [
    "DimensionComparison",
    "JudgeResponse",
    "parse_judge_json",
]
