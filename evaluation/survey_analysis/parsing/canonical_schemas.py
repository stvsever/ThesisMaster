"""
Canonical per-part output shapes.

Both HCP outputs (parsed from Qualtrics) and PHOENIX outputs (loaded from
JSON) must be coerced into ONE canonical shape per part before being shown
to the LLM judge. This is what guarantees the blinding: the judge sees two
JSON blobs of identical structure and cannot tell which side is human.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, TypedDict

PART_KEYS: tuple[str, ...] = ("part1", "part2", "part3", "part4", "part5")

# Maximum field counts per part (matches Qualtrics blueprint).
CANONICAL_FIELD_COUNTS: Dict[str, int] = {
    "part1": 6,
    "part2": 5,
    "part3": 5,
    "part4": 1,
    "part5": 1,
}

HapaPhase = Literal[
    "pre_intentional", "intentional", "action", "maintenance",
]
HAPA_PHASES: tuple[str, ...] = (
    "pre_intentional", "intentional", "action", "maintenance",
)


# ──────────────────────────────────────────────────────────────────────────────
# Dataclass schemas
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Part1Item:
    label: str
    description: str

    def to_dict(self) -> Dict[str, str]:
        return {"label": self.label, "description": self.description}


@dataclass
class Part1Output:
    items: List[Part1Item] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"items": [it.to_dict() for it in self.items]}


@dataclass
class Part2Item:
    label: str

    def to_dict(self) -> Dict[str, str]:
        return {"label": self.label}


@dataclass
class Part2Output:
    items: List[Part2Item] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"items": [it.to_dict() for it in self.items]}


@dataclass
class Part3Item:
    rank: int
    option_id: str

    def to_dict(self) -> Dict[str, Any]:
        return {"rank": self.rank, "option_id": self.option_id}


@dataclass
class Part3Output:
    ranking: List[Part3Item] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"ranking": [it.to_dict() for it in self.ranking]}


@dataclass
class Part4Output:
    selected_options: List[str] = field(default_factory=list)
    note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"selected_options": self.selected_options, "note": self.note}


@dataclass
class Part5Output:
    message: str = ""
    hapa_phase: Optional[HapaPhase] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"message": self.message, "hapa_phase": self.hapa_phase}


# ──────────────────────────────────────────────────────────────────────────────
# Coercion helpers
# ──────────────────────────────────────────────────────────────────────────────

def _split_label_description(text: str) -> tuple[str, str]:
    """
    Split a free-text Part 1 cell into ``(label, description)``.

    The blueprint asked respondents for ``label | description`` but real
    submissions are highly inconsistent: some only provide a label, some use
    semicolons, colons, dashes, or no separator at all. We try each candidate
    separator in turn and fall back to ``label = description = text``.
    """
    if text is None:
        return "", ""
    s = str(text).strip()
    if not s:
        return "", ""
    for sep in (" | ", "|", " :: ", "::", " - ", ":"):
        if sep in s:
            left, _, right = s.partition(sep)
            left, right = left.strip(), right.strip()
            if left and right:
                return left, right
    return s, s


def coerce_part1(payload: Any) -> Part1Output:
    """
    Coerce a ``payload`` (raw cells list, dict, or already-canonical dict) to
    :class:`Part1Output`.
    """
    if isinstance(payload, Part1Output):
        return payload
    if isinstance(payload, dict) and "items" in payload:
        items: List[Part1Item] = []
        for it in payload["items"]:
            if isinstance(it, Part1Item):
                items.append(it)
            elif isinstance(it, dict):
                items.append(Part1Item(
                    label=str(it.get("label", "")).strip(),
                    description=str(it.get("description", "")).strip(),
                ))
        return Part1Output(items=items)
    if isinstance(payload, list):
        items = []
        for cell in payload:
            label, descr = _split_label_description(cell)
            if label or descr:
                items.append(Part1Item(label=label, description=descr))
        return Part1Output(items=items)
    return Part1Output()


def _split_predictor_cell(text: str) -> Part2Item:
    """
    Parse a Part 2 cell.

    The current Qualtrics survey asks only for short treatment-option labels.
    If an older PHOENIX output contains ``label | measurement | criteria``,
    keep only the first field so both sources stay structurally identical
    and the judge cannot infer source identity from extra detail.
    """
    s = str(text or "").strip()
    if not s:
        return Part2Item("")
    label = re.split(r"\s*\|\s*", s, maxsplit=1)[0].strip()
    return Part2Item(label=label)


def coerce_part2(payload: Any) -> Part2Output:
    if isinstance(payload, Part2Output):
        return payload
    if isinstance(payload, dict) and "items" in payload:
        items = []
        for it in payload["items"]:
            if isinstance(it, Part2Item):
                items.append(it)
            elif isinstance(it, dict):
                label = (
                    it.get("label")
                    or it.get("predictor")
                    or it.get("treatment_option")
                    or it.get("option")
                    or ""
                )
                label = re.split(r"\s*\|\s*", str(label), maxsplit=1)[0].strip()
                items.append(Part2Item(label=label))
        return Part2Output(items=items)
    if isinstance(payload, dict) and "treatment_options" in payload:
        return Part2Output(items=[
            _split_predictor_cell(c)
            for c in payload["treatment_options"]
            if str(c or "").strip()
        ])
    if isinstance(payload, list):
        items = [_split_predictor_cell(c) for c in payload if str(c or "").strip()]
        return Part2Output(items=items)
    return Part2Output()


def coerce_part3(payload: Any) -> Part3Output:
    """
    Coerce Part 3 ranking input.

    Accepts:
      - a list of length 5 of rank values where the i-th element is the rank
        assigned to option ``BO-{i+1}`` (Qualtrics matrix shape);
      - a dict ``{"ranking": [...]}`` already in canonical form;
      - a list of dicts with ``rank`` and ``option_id``.
    """
    if isinstance(payload, Part3Output):
        return payload
    if isinstance(payload, dict) and "ranking" in payload:
        items = []
        for it in payload["ranking"]:
            if isinstance(it, Part3Item):
                items.append(it)
            elif isinstance(it, dict):
                rank = int(it.get("rank", 0)) or 0
                opt = str(it.get("option_id", "")).strip()
                if opt and rank:
                    items.append(Part3Item(rank=rank, option_id=opt))
        items.sort(key=lambda x: x.rank)
        return Part3Output(ranking=items)
    if isinstance(payload, list):
        ranking: List[Part3Item] = []
        # Detect whether this is a matrix-shape (rank per option index) or
        # a list-of-dicts ranking.
        if payload and isinstance(payload[0], dict):
            for it in payload:
                rank = int(it.get("rank", 0)) or 0
                opt = str(it.get("option_id", "")).strip()
                if opt and rank:
                    ranking.append(Part3Item(rank=rank, option_id=opt))
        else:
            for i, raw in enumerate(payload, start=1):
                if raw is None or str(raw).strip() == "":
                    continue
                try:
                    rank = int(float(str(raw).strip()))
                except ValueError:
                    continue
                if rank > 0:
                    ranking.append(Part3Item(rank=rank, option_id=f"BO-{i}"))
        ranking.sort(key=lambda x: x.rank)
        return Part3Output(ranking=ranking)
    return Part3Output()


_NOTE_SENTINEL = re.compile(r"//\s*note\s*:", flags=re.IGNORECASE)


def coerce_part4(payload: Any) -> Part4Output:
    """
    Coerce Part 4 multi-select.

    Accepts a single string with options separated by commas or newlines and
    an optional trailing ``//note: ...`` annotation, OR a dict already in
    canonical form, OR a list of strings.
    """
    if isinstance(payload, Part4Output):
        return payload
    if isinstance(payload, dict) and "selected_options" in payload:
        opts = [str(s).strip() for s in payload["selected_options"] if str(s).strip()]
        note = payload.get("note")
        return Part4Output(selected_options=opts, note=str(note).strip() if note else None)
    if isinstance(payload, list):
        opts = [str(s).strip() for s in payload if str(s).strip()]
        return Part4Output(selected_options=opts, note=None)
    if isinstance(payload, str):
        s = payload.strip()
        note: Optional[str] = None
        if _NOTE_SENTINEL.search(s):
            head, _, tail = _NOTE_SENTINEL.split(s, maxsplit=1)[0], None, None
            # Re-split because re.split with capturing groups drops the match.
            match = _NOTE_SENTINEL.search(s)
            if match:
                head = s[: match.start()].rstrip(" ,;\n")
                note = s[match.end():].strip()
                s = head
        # Split on newlines or commas (commas are common in Qualtrics).
        raw_opts = re.split(r"[,\n]+", s)
        opts = [o.strip() for o in raw_opts if o.strip()]
        return Part4Output(selected_options=opts, note=note)
    return Part4Output()


def coerce_part5(payload: Any) -> Part5Output:
    """
    Coerce Part 5 free text. The HAPA phase is left ``None`` for HCP outputs
    (the judge classifies it). System outputs may already include it.
    """
    if isinstance(payload, Part5Output):
        return payload
    if isinstance(payload, dict):
        msg = str(payload.get("message", "")).strip()
        phase = payload.get("hapa_phase")
        if phase is not None:
            phase = str(phase).strip().lower()
            phase = phase if phase in HAPA_PHASES else None
        return Part5Output(message=msg, hapa_phase=phase)
    if isinstance(payload, str):
        return Part5Output(message=payload.strip(), hapa_phase=None)
    return Part5Output()


# ──────────────────────────────────────────────────────────────────────────────
# Public conversion: per-part canonical dict ready for the judge prompt
# ──────────────────────────────────────────────────────────────────────────────

PART_COERCERS = {
    "part1": coerce_part1,
    "part2": coerce_part2,
    "part3": coerce_part3,
    "part4": coerce_part4,
    "part5": coerce_part5,
}


def canonical_for_judge(part: str, payload: Any) -> Dict[str, Any]:
    """
    Return the canonical per-part dict that goes into the judge prompt.

    Always emits a JSON-serialisable dict; ``payload`` may be the raw
    Qualtrics-shaped value, a system JSON dict, or an already-coerced
    dataclass instance.
    """
    if part not in PART_COERCERS:
        raise ValueError(f"Unknown part {part!r}; expected one of {PART_KEYS}")
    obj = PART_COERCERS[part](payload)
    return obj.to_dict()


def dump_canonical(part: str, payload: Any) -> str:
    """Render the canonical dict as a stable, indent-2 JSON string."""
    return json.dumps(canonical_for_judge(part, payload), ensure_ascii=False, indent=2)


def empty_canonical(part: str) -> Dict[str, Any]:
    """Return the empty canonical shape for a part (used as placeholder)."""
    table = {
        "part1": Part1Output(),
        "part2": Part2Output(),
        "part3": Part3Output(),
        "part4": Part4Output(),
        "part5": Part5Output(),
    }
    return table[part].to_dict()


__all__ = [
    "PART_KEYS",
    "CANONICAL_FIELD_COUNTS",
    "HAPA_PHASES",
    "HapaPhase",
    "Part1Item", "Part1Output",
    "Part2Item", "Part2Output",
    "Part3Item", "Part3Output",
    "Part4Output", "Part5Output",
    "canonical_for_judge", "dump_canonical", "empty_canonical",
    "coerce_part1", "coerce_part2", "coerce_part3", "coerce_part4", "coerce_part5",
]
