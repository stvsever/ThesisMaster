"""
Loader / writer for PHOENIX (system) outputs.

System outputs are produced by a separate component of the project; this
module only wraps the I/O so that the judge runner has a single, stable
interface. The expected on-disk shape is::

    data/03_system/system_outputs.json
    {
        "C01": {"part1": {...canonical}, "part2": {...}, ...},
        ...
        "C10": {...}
    }
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .canonical_schemas import PART_KEYS, canonical_for_judge, empty_canonical

logger = logging.getLogger(__name__)


@dataclass
class SystemOutputBundle:
    """Container for the per-case system outputs of all parts."""

    by_case: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def cases(self) -> List[str]:
        return sorted(self.by_case.keys())

    def get(self, case_id: str, part: str) -> Dict[str, Any]:
        case = self.by_case.get(case_id)
        if not case:
            return empty_canonical(part)
        payload = case.get(part)
        if payload is None:
            return empty_canonical(part)
        return canonical_for_judge(part, payload)

    def has(self, case_id: str, part: str) -> bool:
        case = self.by_case.get(case_id)
        return bool(case and case.get(part))


def load_system_outputs(path: Path | str) -> SystemOutputBundle:
    """
    Load PHOENIX outputs.

    Missing files raise ``FileNotFoundError`` so the orchestrator can fall
    back to pseudo system outputs explicitly.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"System outputs not found at {path}. "
            "In pseudo mode, run pseudodata.generate_phoenix_outputs first."
        )
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a dict at top level of {path}, got {type(raw)!r}.")
    by_case: Dict[str, Dict[str, Any]] = {}
    for case_id, parts in raw.items():
        if not isinstance(parts, dict):
            logger.warning("Case %s does not map to a dict; skipping.", case_id)
            continue
        out_parts: Dict[str, Any] = {}
        for part in PART_KEYS:
            payload = parts.get(part)
            if payload is None:
                logger.debug("Case %s missing %s; canonicalising as empty.",
                             case_id, part)
                out_parts[part] = empty_canonical(part)
            else:
                out_parts[part] = canonical_for_judge(part, payload)
        by_case[case_id] = out_parts
    return SystemOutputBundle(by_case=by_case)


def save_system_outputs(
    bundle: SystemOutputBundle | Dict[str, Dict[str, Any]],
    path: Path | str,
) -> Path:
    """Write a ``SystemOutputBundle`` (or raw dict) to ``path`` as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(bundle, SystemOutputBundle):
        payload = bundle.by_case
    else:
        payload = bundle
    canonical: Dict[str, Dict[str, Any]] = {}
    for case_id, parts in payload.items():
        canonical[case_id] = {
            part: canonical_for_judge(part, parts.get(part))
            for part in PART_KEYS
        }
    path.write_text(
        json.dumps(canonical, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


__all__ = [
    "SystemOutputBundle",
    "load_system_outputs",
    "save_system_outputs",
]
