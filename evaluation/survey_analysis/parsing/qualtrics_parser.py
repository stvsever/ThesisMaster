"""
Defensive parser for the Qualtrics export of the PHOENIX PRE survey.

Each row of the export is one HCP submission. Columns are namespaced with
``HCP##_C##_PART#``. Only the columns of the HCP that submitted are filled
on any given row; the embedded data column ``hcp`` (e.g. ``HCP03``) tells
us which block to read.

The parser turns each row into a :class:`HCPResponse` and the full export
into ``{case_id: {part: canonical_dict}}`` mappings, one per HCP.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .canonical_schemas import (
    PART_KEYS,
    canonical_for_judge,
)

logger = logging.getLogger(__name__)

HCP_PREFIX_RE = re.compile(r"^(HCP\d{2})_C(\d{2})_(PART[1-5])(?:_(\d+))?$")
HCP_INTAKE_RE = re.compile(r"^(HCP\d{2})_(CONSENT_DECISION|INTAKE_[A-Z_0-9]+)$")
HCP_FEEDBACK_RE = re.compile(r"^(HCP\d{2})_PART6_([ABC])$")

# Expected counts per part block (matches the survey blueprint).
PART_FIELD_RANGE: Dict[str, Optional[range]] = {
    "PART1": range(1, 7),
    "PART2": range(1, 6),
    "PART3": range(1, 6),
    "PART4": None,   # single column, no suffix
    "PART5": None,   # single column, no suffix
}


@dataclass
class HCPResponse:
    """One HCP's full response — case + intake + outputs + feedback."""

    hcp_id: str
    case_id: str                         # e.g. "C03" -> stored as "C03"
    response_id: str
    duration_seconds: Optional[int] = None
    intake: Dict[str, str] = field(default_factory=dict)
    raw_parts: Dict[str, Any] = field(default_factory=dict)
    feedback: Dict[str, str] = field(default_factory=dict)

    def canonical(self) -> Dict[str, Any]:
        """Return the canonical per-part dict for this HCP's case."""
        return {
            part: canonical_for_judge(part, self.raw_parts.get(part, None))
            for part in PART_KEYS
        }


# Public aliases used in callers.
HCP_RESPONSE = HCPResponse


def _resolve_hcp_for_row(
    header: List[str],
    row: List[str],
    embedded_col: Optional[int],
) -> Optional[str]:
    """
    Determine which HCP submitted this row.

    Strategy (in order):
      1. Embedded data column ``hcp`` (last named column, e.g. ``HCP03``).
      2. The first ``HCP##_C##_PART1_1`` cell that is non-empty.
    """
    if embedded_col is not None and embedded_col < len(row):
        v = row[embedded_col].strip()
        if re.fullmatch(r"HCP\d{2}", v):
            return v
    # Fallback: scan for the first non-empty HCP##_C##_PART1_1.
    for i, name in enumerate(header):
        m = HCP_PREFIX_RE.match(name)
        if not m:
            continue
        if m.group(3) == "PART1" and m.group(4) == "1" and i < len(row):
            if row[i].strip():
                return m.group(1)
    return None


def _build_part_payload(
    hcp_id: str,
    case_id: str,
    part: str,
    row: List[str],
    header_index: Dict[str, int],
) -> Any:
    """Collect the raw cells for one (hcp, case, part) into a list/string."""
    if part in ("PART1", "PART2", "PART3"):
        items: List[str] = []
        for i in PART_FIELD_RANGE[part]:
            col = f"{hcp_id}_{case_id}_{part}_{i}"
            idx = header_index.get(col)
            if idx is not None and idx < len(row):
                items.append(row[idx])
            else:
                items.append("")
        return items
    if part in ("PART4", "PART5"):
        col = f"{hcp_id}_{case_id}_{part}"
        idx = header_index.get(col)
        if idx is not None and idx < len(row):
            return row[idx]
        return ""
    raise ValueError(f"Unknown part {part!r}")


def parse_qualtrics_csv(path: Path | str) -> List[HCPResponse]:
    """
    Parse a Qualtrics export and return one :class:`HCPResponse` per submission.

    Skips header rows (Qualtrics inserts two: human label, ImportId mapping)
    and rows that look like incomplete previews.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Qualtrics CSV not found: {path}")

    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if len(rows) < 4:
        logger.warning("Qualtrics CSV has fewer than 4 rows; nothing to parse.")
        return []

    header = rows[0]
    header_index: Dict[str, int] = {name: i for i, name in enumerate(header)}
    embedded_col: Optional[int] = header_index.get("hcp")

    duration_idx = header_index.get("Duration (in seconds)")
    response_id_idx = header_index.get("ResponseId")
    finished_idx = header_index.get("Finished")

    out: List[HCPResponse] = []
    for r_idx, row in enumerate(rows[3:], start=3):
        if not any(cell.strip() for cell in row):
            continue

        hcp_id = _resolve_hcp_for_row(header, row, embedded_col)
        if hcp_id is None:
            logger.debug("Row %d: could not resolve HCP id; skipped.", r_idx)
            continue
        # Find the case id this HCP responded about. There is exactly one
        # C## suffix per HCP block in the survey design.
        case_id: Optional[str] = None
        for col_name in header:
            m = HCP_PREFIX_RE.match(col_name)
            if m and m.group(1) == hcp_id:
                case_id = f"C{m.group(2)}"
                break
        if case_id is None:
            logger.debug("Row %d: could not resolve case_id for %s; skipped.",
                         r_idx, hcp_id)
            continue

        # Sanity-check: at least one of the part columns must be non-empty.
        any_data = False
        for part in ("PART1", "PART2", "PART3", "PART4", "PART5"):
            payload = _build_part_payload(hcp_id, case_id, part, row, header_index)
            if isinstance(payload, list):
                any_data = any_data or any(str(x).strip() for x in payload)
            else:
                any_data = any_data or bool(str(payload).strip())
            if any_data:
                break
        if not any_data:
            logger.debug("Row %d: %s/%s has no PART data; skipped.",
                         r_idx, hcp_id, case_id)
            continue

        duration = None
        if duration_idx is not None and duration_idx < len(row):
            try:
                duration = int(float(row[duration_idx])) if row[duration_idx].strip() else None
            except ValueError:
                duration = None

        response_id = ""
        if response_id_idx is not None and response_id_idx < len(row):
            response_id = row[response_id_idx]

        intake: Dict[str, str] = {}
        feedback: Dict[str, str] = {}
        for col_name, idx in header_index.items():
            m_intake = HCP_INTAKE_RE.match(col_name)
            if m_intake and m_intake.group(1) == hcp_id and idx < len(row):
                intake[m_intake.group(2)] = row[idx]
                continue
            m_fb = HCP_FEEDBACK_RE.match(col_name)
            if m_fb and m_fb.group(1) == hcp_id and idx < len(row):
                feedback[m_fb.group(2)] = row[idx]

        raw_parts: Dict[str, Any] = {}
        for part in ("PART1", "PART2", "PART3", "PART4", "PART5"):
            raw_parts[part.lower()] = _build_part_payload(
                hcp_id, case_id, part, row, header_index,
            )

        out.append(HCPResponse(
            hcp_id=hcp_id,
            case_id=case_id,
            response_id=response_id,
            duration_seconds=duration,
            intake=intake,
            raw_parts=raw_parts,
            feedback=feedback,
        ))

    logger.info("Parsed %d HCP responses from %s.", len(out), path)
    return out


def save_parsed_outputs(
    responses: Iterable[HCPResponse],
    out_dir: Path,
) -> Dict[str, Path]:
    """
    Save parsed outputs.

    Writes:
      - ``hcp_outputs.json``: ``{case_id: {part: canonical}}`` aggregating
        the outputs across all HCPs (one HCP per case in the new design).
      - ``hcp_intake.json``: per-HCP intake + feedback, useful for
        reporting.
      - ``hcp_responses_detailed.json``: per-HCP raw parts and case mapping.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    canonical_by_case: Dict[str, Dict[str, Any]] = {}
    intake_by_hcp: Dict[str, Dict[str, Any]] = {}
    detailed: Dict[str, Dict[str, Any]] = {}

    for resp in responses:
        canonical_by_case.setdefault(resp.case_id, resp.canonical())
        intake_by_hcp[resp.hcp_id] = {
            "case_id": resp.case_id,
            "response_id": resp.response_id,
            "duration_seconds": resp.duration_seconds,
            "intake": resp.intake,
            "feedback": resp.feedback,
        }
        detailed[resp.hcp_id] = {
            "case_id": resp.case_id,
            "raw_parts": resp.raw_parts,
            "canonical": resp.canonical(),
        }

    paths = {
        "hcp_outputs": out_dir / "hcp_outputs.json",
        "hcp_intake": out_dir / "hcp_intake.json",
        "hcp_detailed": out_dir / "hcp_responses_detailed.json",
    }
    paths["hcp_outputs"].write_text(
        json.dumps(canonical_by_case, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths["hcp_intake"].write_text(
        json.dumps(intake_by_hcp, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths["hcp_detailed"].write_text(
        json.dumps(detailed, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return paths


__all__ = [
    "HCPResponse",
    "HCP_RESPONSE",
    "parse_qualtrics_csv",
    "save_parsed_outputs",
]
