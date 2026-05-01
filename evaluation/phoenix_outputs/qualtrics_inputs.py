"""Extract the exact PHOENIX evaluation inputs from the Qualtrics sources.

The live HCP survey uses rendered network figures in Part 3. PHOENIX cannot
consume images, so this module reconstructs the same non-image information:
case text, standardised symptom labels, treatment-option labels, monitoring
summary, abstract targets, candidate EMA items, coaching context, and the
network edges with numeric weights.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, List, Mapping, Tuple

try:
    from .paths import (
        CASE_CONTEXTS_JSON,
        CASE_INPUTS_JSON,
        QUALTRICS_CASE_SOURCE_ROOT,
        QUALTRICS_EDGE_WEIGHTS,
        QUALTRICS_GENERATE_QSF,
        REPO_ROOT,
        ensure_dirs,
    )
except ImportError:  # direct script execution from this folder
    from paths import (  # type: ignore
        CASE_CONTEXTS_JSON,
        CASE_INPUTS_JSON,
        QUALTRICS_CASE_SOURCE_ROOT,
        QUALTRICS_EDGE_WEIGHTS,
        QUALTRICS_GENERATE_QSF,
        REPO_ROOT,
        ensure_dirs,
    )

CONTRACT_VERSION = "phoenix-survey-inputs-v1"


def _load_qsf_generator() -> ModuleType:
    """Load the existing Qualtrics parser without regenerating survey files."""
    if not QUALTRICS_GENERATE_QSF.exists():
        raise FileNotFoundError(f"Missing Qualtrics generator: {QUALTRICS_GENERATE_QSF}")
    spec = importlib.util.spec_from_file_location(
        "phoenix_qualtrics_generate_qsf",
        QUALTRICS_GENERATE_QSF,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {QUALTRICS_GENERATE_QSF}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_example_network(qsf_module: ModuleType) -> Any:
    template_tex = (QUALTRICS_CASE_SOURCE_ROOT / "HCP_01" / "main.tex").read_text(
        encoding="utf-8",
    )
    p3_shared = qsf_module.extract_between(
        template_tex,
        r"\\section\{Deel 3:.*?\}",
        r"\\subsection\*\{Casus 1:",
    )
    example_tikz = qsf_module.extract_first(
        r"\\begin\{voorbeeldbox\}\{Deel 3:.*?\\begin\{tikzpicture\}"
        r"(.*?)\\end\{tikzpicture\}.*?\\end\{voorbeeldbox\}",
        p3_shared,
    )
    return qsf_module.parse_network(
        example_tikz,
        "Behandelingsopties",
        "Symptomen",
        "shared_example",
    )


def _case_dirs() -> List[Path]:
    paths = [QUALTRICS_CASE_SOURCE_ROOT / f"HCP_{idx:02d}" for idx in range(1, 11)]
    missing = [p for p in paths if not (p / "main.tex").exists()]
    if missing:
        raise FileNotFoundError(
            "Missing expected Qualtrics source files: "
            + ", ".join(str(p / "main.tex") for p in missing)
        )
    return paths


def _parse_option(text: str) -> Dict[str, str]:
    match = re.match(r"^(?:BO|P)-?(\d+)\s*:\s*(.+)$", str(text).strip())
    if not match:
        return {"option_id": "", "label": str(text).strip()}
    return {"option_id": f"BO-{int(match.group(1))}", "label": match.group(2).strip()}


def _parse_weight_endpoint(text: str, *, prefix: str) -> Tuple[str, str]:
    pattern = rf"^{re.escape(prefix)}\s*(\d+)\s*\((.*?)\)$"
    match = re.match(pattern, str(text).strip())
    if not match:
        return "", str(text).strip()
    number = int(match.group(1))
    out_prefix = "BO" if prefix == "P" else prefix.rstrip("-")
    return f"{out_prefix}-{number}", match.group(2).strip()


def _normalise_edge_weights(raw: Mapping[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for case_id, payload in raw.items():
        edges: List[Dict[str, Any]] = []
        for edge in payload.get("edges", []):
            predictor_id, predictor_label = _parse_weight_endpoint(
                edge.get("predictor", ""),
                prefix="P",
            )
            criterion_id, criterion_label = _parse_weight_endpoint(
                edge.get("criterion", ""),
                prefix="CR-",
            )
            edges.append({
                "from_option_id": predictor_id,
                "from_label": predictor_label,
                "to_symptom_id": criterion_id,
                "to_label": criterion_label,
                "weight": float(edge.get("weight", 0.0)),
                "direction": str(edge.get("direction", "")).strip(),
            })
        out[str(case_id)] = edges
    return out


def _load_edge_weights() -> Dict[str, List[Dict[str, Any]]]:
    if not QUALTRICS_EDGE_WEIGHTS.exists():
        return {}
    raw = json.loads(QUALTRICS_EDGE_WEIGHTS.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object in {QUALTRICS_EDGE_WEIGHTS}")
    return _normalise_edge_weights(raw)


def _nodes_from_network(case: Any) -> Dict[str, List[Dict[str, str]]]:
    treatment_options: List[Dict[str, str]] = []
    symptoms: List[Dict[str, str]] = []
    for node in case.network.nodes:
        if node.kind == "left":
            parsed = _parse_option(f"{node.prefix}: {node.label}")
            treatment_options.append(parsed)
        else:
            symptoms.append({
                "symptom_id": str(node.prefix).replace("CR", "CR-")
                if re.match(r"^CR\d+$", str(node.prefix))
                else str(node.prefix),
                "label": str(node.label).strip(),
            })
    treatment_options.sort(
        key=lambda x: int(re.search(r"\d+", x["option_id"]).group(0))
        if re.search(r"\d+", x["option_id"]) else 999
    )
    symptoms.sort(
        key=lambda x: int(re.search(r"\d+", x["symptom_id"]).group(0))
        if re.search(r"\d+", x["symptom_id"]) else 999
    )
    return {"treatment_options": treatment_options, "symptoms": symptoms}


def _ema_items(case: Any) -> List[Dict[str, str]]:
    return [
        {"item_id": f"EMA-{idx:02d}", "label": str(label).strip()}
        for idx, label in enumerate(case.part4_items, start=1)
    ]


def _case_to_input(case: Any, edges_by_case: Mapping[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    nodes = _nodes_from_network(case)
    case_id = str(case.case_code)
    return {
        "case_id": case_id,
        "hcp_code": str(case.hcp_code),
        "profile": str(case.profile),
        "duration": str(case.duration),
        "part1": {
            "vignette": str(case.complaint_vignette),
        },
        "part2": {
            "case_summary": str(case.short_summary),
            "standardized_symptoms": list(case.part2_symptoms),
        },
        "part3": {
            "case_summary": str(case.short_summary),
            "monitoring_summary": str(case.monitoring),
            "treatment_options": nodes["treatment_options"],
            "network_symptoms": nodes["symptoms"],
            "network_edges": list(edges_by_case.get(case_id, [])),
        },
        "part4": {
            "treatment_targets": list(case.part4_targets),
            "candidate_ema_items": _ema_items(case),
        },
        "part5": {
            "primary_problem": str(case.part5_primary_problem),
            "treatment_goal": str(case.part5_target),
            "barrier": str(case.part5_barrier),
            "coping_strategy": str(case.part5_coping),
        },
    }


def extract_case_inputs() -> Dict[str, Any]:
    """Return the complete case-input bundle derived from the Qualtrics source."""
    qsf_module = _load_qsf_generator()
    example_network = _load_example_network(qsf_module)
    edges_by_case = _load_edge_weights()
    cases: Dict[str, Any] = {}
    for case_dir in _case_dirs():
        tex = (case_dir / "main.tex").read_text(encoding="utf-8")
        case = qsf_module.parse_case_survey(tex, example_network)
        cases[str(case.case_code)] = _case_to_input(case, edges_by_case)
    source = QUALTRICS_CASE_SOURCE_ROOT.relative_to(REPO_ROOT)
    edge_source = QUALTRICS_EDGE_WEIGHTS.relative_to(REPO_ROOT)
    return {
        "metadata": {
            "contract_version": CONTRACT_VERSION,
            "source": str(source),
            "edge_weights_source": str(edge_source),
            "case_count": len(cases),
        },
        "cases": dict(sorted(cases.items())),
    }


def case_contexts_from_inputs(bundle: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Build the exact case-context JSON consumed by the LLM judge prompts."""
    contexts: Dict[str, Dict[str, Any]] = {}
    for case_id, case in bundle.get("cases", {}).items():
        p1 = case.get("part1", {})
        p2 = case.get("part2", {})
        p3 = case.get("part3", {})
        p4 = case.get("part4", {})
        p5 = case.get("part5", {})
        contexts[str(case_id)] = {
            "vignette": p1.get("vignette", ""),
            "case_notes": {
                "profile": case.get("profile", ""),
                "duration": case.get("duration", ""),
                "case_summary": p2.get("case_summary", ""),
            },
            "standardized_symptoms": p2.get("standardized_symptoms", []),
            "standardized_treatment_options": p3.get("treatment_options", []),
            "network_summary": {
                "monitoring_summary": p3.get("monitoring_summary", ""),
                "symptoms": p3.get("network_symptoms", []),
                "edges": p3.get("network_edges", []),
                "edge_interpretation": {
                    "positive": "more of treatment option is associated with more symptom burden",
                    "negative": "more of treatment option is associated with less symptom burden",
                },
            },
            "ema_summary": {
                "monitoring_summary": p3.get("monitoring_summary", ""),
            },
            "treatment_targets": p4.get("treatment_targets", []),
            "candidate_ema_items": p4.get("candidate_ema_items", []),
            "primary_problem": p5.get("primary_problem", ""),
            "treatment_goal": p5.get("treatment_goal", ""),
            "barrier": p5.get("barrier", ""),
            "coping_strategy": p5.get("coping_strategy", ""),
        }
    return contexts


def write_case_inputs(
    inputs_path: Path = CASE_INPUTS_JSON,
    contexts_path: Path = CASE_CONTEXTS_JSON,
) -> Dict[str, Path]:
    """Extract and write the PHOENIX input bundle plus judge contexts."""
    ensure_dirs()
    bundle = extract_case_inputs()
    contexts = case_contexts_from_inputs(bundle)
    inputs_path.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    contexts_path.write_text(
        json.dumps(contexts, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"case_inputs": inputs_path, "case_contexts": contexts_path}


def load_case_inputs(path: Path = CASE_INPUTS_JSON) -> Dict[str, Any]:
    """Load a previously extracted case-input bundle."""
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "CONTRACT_VERSION",
    "extract_case_inputs",
    "case_contexts_from_inputs",
    "write_case_inputs",
    "load_case_inputs",
]
