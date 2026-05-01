"""Agentic PHOENIX output quality gate before LLM-as-judge evaluation.

The quality gate is a deterministic PHOENIX-side refinement step. It repairs
format drift and strengthens outputs against the same task constraints used in
the survey: compact labels, complete rankings, valid EMA item choices, and
mobile-safe coaching messages.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

try:
    from .canonicalize_outputs import validate_canonical_bundle
    from .paths import (
        CASE_INPUTS_JSON,
        REPO_ROOT,
        SYSTEM_OUTPUTS_JSON,
        OUTPUTS_DIR,
        VALIDATION_REPORT_JSON,
        SURVEY_ANALYSIS_SYSTEM_OUTPUTS_JSON,
        ensure_dirs,
    )
except ImportError:  # direct script execution from this folder
    from canonicalize_outputs import validate_canonical_bundle  # type: ignore
    from paths import (  # type: ignore
        CASE_INPUTS_JSON,
        REPO_ROOT,
        SYSTEM_OUTPUTS_JSON,
        OUTPUTS_DIR,
        VALIDATION_REPORT_JSON,
        SURVEY_ANALYSIS_SYSTEM_OUTPUTS_JSON,
        ensure_dirs,
    )

SURVEY_ANALYSIS_DIR = REPO_ROOT / "evaluation" / "survey_analysis"
if str(SURVEY_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(SURVEY_ANALYSIS_DIR))

from parsing.canonical_schemas import canonical_for_judge  # noqa: E402


REFINED_OUTPUTS_JSON: Path = OUTPUTS_DIR / "system_outputs_refined.json"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _label(text: Any, max_words: int = 5) -> str:
    s = str(text or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^(aantal|minuten|mate van|frequentie van)\s+", "", s, flags=re.I)
    s = re.sub(r"\s+(per dag|per week|vandaag|vanavond)$", "", s, flags=re.I)
    s = s.strip(" .;:,")
    words = s.split()
    if len(words) > max_words:
        s = " ".join(words[:max_words])
    return s[:1].upper() + s[1:] if s else ""


def _item_labels(items: Iterable[Mapping[str, Any]]) -> List[str]:
    labels = []
    for item in items:
        label = str(item.get("label", "")).strip()
        if label:
            labels.append(label)
    return labels


def _tokens(text: Any) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[\wÀ-ÿ-]+", str(text), flags=re.UNICODE)
        if len(token) > 2
    }


def _unique(labels: Iterable[str], min_items: int, max_items: int) -> List[Dict[str, str]]:
    seen: set[str] = set()
    out: List[Dict[str, str]] = []
    for raw in labels:
        lab = _label(raw)
        key = lab.lower()
        if lab and key not in seen:
            out.append({"label": lab})
            seen.add(key)
        if len(out) >= max_items:
            break
    return out[:max_items] if len(out) >= min_items else out


def _fallback_symptom_labels(case_payload: Mapping[str, Any]) -> List[str]:
    symptoms = case_payload.get("part2", {}).get("standardized_symptoms", [])
    return [_label(x) for x in symptoms]


def _fallback_option_labels(case_payload: Mapping[str, Any]) -> List[str]:
    options = case_payload.get("part3", {}).get("treatment_options", [])
    return [_label(item.get("label", "")) for item in options]


def _refine_part1(raw: Mapping[str, Any], case_payload: Mapping[str, Any]) -> Dict[str, Any]:
    labels = []
    for item in raw.get("items", []) if isinstance(raw, Mapping) else []:
        labels.append(item.get("label", item) if isinstance(item, Mapping) else item)
    labels.extend(_fallback_symptom_labels(case_payload))
    return {"items": _unique(labels, min_items=3, max_items=6)}


def _refine_part2(raw: Mapping[str, Any], case_payload: Mapping[str, Any]) -> Dict[str, Any]:
    labels = []
    for item in raw.get("items", []) if isinstance(raw, Mapping) else []:
        labels.append(item.get("label", item) if isinstance(item, Mapping) else item)
    labels.extend(_fallback_option_labels(case_payload))
    return {"items": _unique(labels, min_items=3, max_items=5)}


def _rank_options(case_payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    options = case_payload.get("part3", {}).get("treatment_options", [])
    edges = case_payload.get("part3", {}).get("network_edges", [])
    scores: Dict[str, float] = {}
    for edge in edges:
        oid = str(edge.get("from_option_id", "")).strip()
        if not oid:
            continue
        scores[oid] = scores.get(oid, 0.0) + abs(float(edge.get("weight", 0.0)))
    ranked = sorted(
        options,
        key=lambda item: (
            -scores.get(str(item.get("option_id", "")), 0.0),
            str(item.get("option_id", "")),
        ),
    )
    return [
        {"rank": idx, "option_id": str(item.get("option_id", "")).strip()}
        for idx, item in enumerate(ranked, start=1)
    ]


def _refine_part3(raw: Mapping[str, Any], case_payload: Mapping[str, Any]) -> Dict[str, Any]:
    expected_ids = {
        str(item.get("option_id", "")).strip()
        for item in case_payload.get("part3", {}).get("treatment_options", [])
    }
    ranking = raw.get("ranking", []) if isinstance(raw, Mapping) else []
    ids = [str(item.get("option_id", "")).strip() for item in ranking if isinstance(item, Mapping)]
    ranks = [int(item.get("rank", 0)) for item in ranking if isinstance(item, Mapping) and str(item.get("rank", "")).isdigit()]
    if len(ranking) == 5 and set(ids) == expected_ids and sorted(ranks) == [1, 2, 3, 4, 5]:
        return {"ranking": sorted(
            [{"rank": int(item["rank"]), "option_id": str(item["option_id"]).strip()} for item in ranking],
            key=lambda x: x["rank"],
        )}
    return {"ranking": _rank_options(case_payload)}


def _target_item_scores(target: str, labels: Sequence[str]) -> List[Tuple[int, int, str]]:
    target_tokens = _tokens(target)
    scored = []
    for idx, label in enumerate(labels):
        item_tokens = _tokens(label)
        overlap = len(target_tokens & item_tokens)
        directness = int(any(t in item_tokens for t in target_tokens))
        scored.append((-overlap, -directness, idx, label))
    return sorted(scored)


def _refine_part4(raw: Mapping[str, Any], case_payload: Mapping[str, Any]) -> Dict[str, Any]:
    candidates = _item_labels(case_payload.get("part4", {}).get("candidate_ema_items", []))
    valid = set(candidates)
    selected = [
        str(item).strip()
        for item in (raw.get("selected_options", []) if isinstance(raw, Mapping) else [])
        if str(item).strip() in valid
    ]
    if len(dict.fromkeys(selected)) == 6:
        return {"selected_options": list(dict.fromkeys(selected))}
    targets = [str(t) for t in case_payload.get("part4", {}).get("treatment_targets", [])]
    out: List[str] = []
    used: set[str] = set()
    for target in targets[:3]:
        count = 0
        target_selected = [x for x in selected if x not in used and (_tokens(target) & _tokens(x))]
        for item in target_selected[:2]:
            out.append(item)
            used.add(item)
            count += 1
        if count < 2:
            for _overlap, _directness, _idx, label in _target_item_scores(target, candidates):
                if label not in used:
                    out.append(label)
                    used.add(label)
                    count += 1
                if count == 2:
                    break
    for label in candidates:
        if len(out) >= 6:
            break
        if label not in used:
            out.append(label)
            used.add(label)
    return {"selected_options": out[:6]}


def _sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip(" .;:")
    return text[:1].lower() + text[1:] if text else ""


def _refine_part5(raw: Mapping[str, Any], case_payload: Mapping[str, Any]) -> Dict[str, Any]:
    p5 = case_payload.get("part5", {})
    goal = _sentence(str(p5.get("treatment_goal", "")))
    barrier = _sentence(str(p5.get("barrier", "")))
    coping = _sentence(str(p5.get("coping_strategy", "")))
    message = (
        f"Vandaag kies je één kleine stap richting {goal}. "
        f"Als {barrier}, maak je de actie bewust kort en haalbaar. "
        f"Gebruik {coping} en plan meteen wanneer je dit doet. "
        "Merk daarna kort op wat het effect was, zonder jezelf te beoordelen."
    )
    return {"message": message}


def refine_outputs(
    raw_outputs: Mapping[str, Mapping[str, Any]],
    case_inputs: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Return refined canonical PHOENIX outputs for all cases."""
    refined: Dict[str, Dict[str, Any]] = {}
    for case_id, case_payload in case_inputs.get("cases", {}).items():
        raw_case = raw_outputs.get(case_id, {})
        raw_case = raw_case if isinstance(raw_case, Mapping) else {}
        parts = {
            "part1": _refine_part1(raw_case.get("part1", {}), case_payload),
            "part2": _refine_part2(raw_case.get("part2", {}), case_payload),
            "part3": _refine_part3(raw_case.get("part3", {}), case_payload),
            "part4": _refine_part4(raw_case.get("part4", {}), case_payload),
            "part5": _refine_part5(raw_case.get("part5", {}), case_payload),
        }
        refined[str(case_id)] = {
            part: canonical_for_judge(part, payload)
            for part, payload in parts.items()
        }
    return refined


def refine_file(
    raw_path: Path,
    *,
    case_inputs_path: Path = CASE_INPUTS_JSON,
    output_path: Path = REFINED_OUTPUTS_JSON,
    report_path: Path = VALIDATION_REPORT_JSON,
    sync_to_pipeline: bool = False,
) -> Dict[str, Path]:
    """Refine a PHOENIX output bundle and optionally sync it to the pipeline."""
    ensure_dirs()
    raw_outputs = _load_json(raw_path)
    case_inputs = _load_json(case_inputs_path)
    if not isinstance(raw_outputs, Mapping):
        raise ValueError(f"Expected top-level object in {raw_path}")
    refined = refine_outputs(raw_outputs, case_inputs)
    report = validate_canonical_bundle(refined, case_inputs)
    output_path.write_text(
        json.dumps(refined, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths: Dict[str, Path] = {
        "refined_outputs": output_path,
        "validation_report": report_path,
    }
    if not report.ok:
        raise ValueError(f"Refined output validation failed; see {report_path}")
    if sync_to_pipeline:
        shutil.copy2(output_path, SYSTEM_OUTPUTS_JSON)
        SURVEY_ANALYSIS_SYSTEM_OUTPUTS_JSON.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_path, SURVEY_ANALYSIS_SYSTEM_OUTPUTS_JSON)
        paths["system_outputs"] = SYSTEM_OUTPUTS_JSON
        paths["pipeline_system_outputs"] = SURVEY_ANALYSIS_SYSTEM_OUTPUTS_JSON
    return paths


__all__ = [
    "REFINED_OUTPUTS_JSON",
    "refine_outputs",
    "refine_file",
]
