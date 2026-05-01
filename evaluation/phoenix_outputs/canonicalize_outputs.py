"""Canonicalise and validate PHOENIX outputs for double-blind judging."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

try:
    from .paths import (
        CASE_INPUTS_JSON,
        REPO_ROOT,
        SYSTEM_OUTPUTS_JSON,
        SYSTEM_OUTPUTS_TEMPLATE_JSON,
        VALIDATION_REPORT_JSON,
        ensure_dirs,
    )
except ImportError:  # direct script execution from this folder
    from paths import (  # type: ignore
        CASE_INPUTS_JSON,
        REPO_ROOT,
        SYSTEM_OUTPUTS_JSON,
        SYSTEM_OUTPUTS_TEMPLATE_JSON,
        VALIDATION_REPORT_JSON,
        ensure_dirs,
    )

SURVEY_ANALYSIS_DIR = REPO_ROOT / "evaluation" / "survey_analysis"
if str(SURVEY_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(SURVEY_ANALYSIS_DIR))

from parsing.canonical_schemas import PART_KEYS, canonical_for_judge, empty_canonical  # noqa: E402


@dataclass
class BundleValidation:
    """Validation report for a canonical output bundle."""

    ok: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        self.ok = False
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _expected_cases(case_inputs: Mapping[str, Any]) -> List[str]:
    return sorted(str(case_id) for case_id in case_inputs.get("cases", {}).keys())


def _item_labels(items: Iterable[Mapping[str, Any]]) -> List[str]:
    return [str(item.get("label", "")).strip() for item in items if str(item.get("label", "")).strip()]


def _valid_part4_labels(case_payload: Mapping[str, Any]) -> set[str]:
    return set(_item_labels(case_payload.get("part4", {}).get("candidate_ema_items", [])))


def canonicalize_system_outputs(
    raw_outputs: Mapping[str, Mapping[str, Any]],
    *,
    expected_cases: Sequence[str] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Coerce arbitrary PHOENIX outputs into the judge-ready canonical shape."""
    cases = list(expected_cases or sorted(raw_outputs.keys()))
    canonical: Dict[str, Dict[str, Any]] = {}
    for case_id in cases:
        raw_case = raw_outputs.get(case_id, {})
        canonical[str(case_id)] = {
            part: canonical_for_judge(part, raw_case.get(part, empty_canonical(part)))
            for part in PART_KEYS
        }
    return canonical


def validate_canonical_bundle(
    bundle: Mapping[str, Mapping[str, Any]],
    case_inputs: Mapping[str, Any],
    *,
    require_all_cases: bool = True,
) -> BundleValidation:
    """Validate counts, part presence, identifiers, and Part 4 candidate membership."""
    report = BundleValidation()
    expected = _expected_cases(case_inputs)
    cases_payload = case_inputs.get("cases", {})

    if require_all_cases:
        missing_cases = [case_id for case_id in expected if case_id not in bundle]
        if missing_cases:
            report.add_error(f"Missing cases: {missing_cases}")

    extra_cases = [case_id for case_id in bundle.keys() if case_id not in expected]
    if extra_cases:
        report.add_warning(f"Unexpected cases not present in Qualtrics inputs: {extra_cases}")

    for case_id in expected:
        case = bundle.get(case_id, {})
        source_case = cases_payload.get(case_id, {})
        for part in PART_KEYS:
            if part not in case:
                report.add_error(f"{case_id}/{part}: missing part")
                continue
            payload = case[part]
            try:
                canonical = canonical_for_judge(part, payload)
            except Exception as exc:
                report.add_error(f"{case_id}/{part}: cannot canonicalize: {exc}")
                continue
            if part == "part1":
                n = len(canonical.get("items", []))
                if not 2 <= n <= 6:
                    report.add_warning(f"{case_id}/part1: expected 2..6 labels, got {n}")
            elif part == "part2":
                n = len(canonical.get("items", []))
                if not 3 <= n <= 5:
                    report.add_warning(f"{case_id}/part2: expected 3..5 labels, got {n}")
            elif part == "part3":
                ranking = canonical.get("ranking", [])
                option_ids = [str(item.get("option_id", "")) for item in ranking]
                ranks = [int(item.get("rank", 0)) for item in ranking]
                expected_options = {
                    str(item.get("option_id", ""))
                    for item in source_case.get("part3", {}).get("treatment_options", [])
                }
                if len(ranking) != 5 or sorted(ranks) != [1, 2, 3, 4, 5]:
                    report.add_warning(f"{case_id}/part3: expected complete 1..5 ranking")
                if set(option_ids) != expected_options:
                    report.add_warning(
                        f"{case_id}/part3: option ids {sorted(option_ids)} "
                        f"do not match expected {sorted(expected_options)}"
                    )
            elif part == "part4":
                selected = [str(x).strip() for x in canonical.get("selected_options", [])]
                valid = _valid_part4_labels(source_case)
                if len(selected) != 6:
                    report.add_warning(f"{case_id}/part4: expected exactly 6 selections, got {len(selected)}")
                invalid = sorted({x for x in selected if x not in valid})
                if invalid:
                    report.add_warning(f"{case_id}/part4: selections not in candidate list: {invalid}")
            elif part == "part5":
                message = str(canonical.get("message", "")).strip()
                if not message:
                    report.add_warning(f"{case_id}/part5: empty message")
    return report


def write_template_outputs(
    case_inputs_path: Path = CASE_INPUTS_JSON,
    output_path: Path = SYSTEM_OUTPUTS_TEMPLATE_JSON,
) -> Path:
    """Write an empty canonical output template with all cases and parts."""
    ensure_dirs()
    case_inputs = _load_json(case_inputs_path)
    template = {
        case_id: {part: empty_canonical(part) for part in PART_KEYS}
        for case_id in _expected_cases(case_inputs)
    }
    output_path.write_text(
        json.dumps(template, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def canonicalize_file(
    raw_path: Path,
    *,
    case_inputs_path: Path = CASE_INPUTS_JSON,
    output_path: Path = SYSTEM_OUTPUTS_JSON,
    report_path: Path = VALIDATION_REPORT_JSON,
) -> Dict[str, Path]:
    """Canonicalise a PHOENIX output JSON file and write validation artifacts."""
    ensure_dirs()
    raw = _load_json(raw_path)
    case_inputs = _load_json(case_inputs_path)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected top-level object in {raw_path}")
    canonical = canonicalize_system_outputs(raw, expected_cases=_expected_cases(case_inputs))
    report = validate_canonical_bundle(canonical, case_inputs)
    output_path.write_text(
        json.dumps(canonical, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not report.ok:
        raise ValueError(f"Output validation failed; see {report_path}")
    return {"system_outputs": output_path, "validation_report": report_path}


__all__ = [
    "BundleValidation",
    "canonicalize_system_outputs",
    "validate_canonical_bundle",
    "write_template_outputs",
    "canonicalize_file",
]

