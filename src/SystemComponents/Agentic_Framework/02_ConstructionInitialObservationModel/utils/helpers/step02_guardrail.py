from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

try:
    from shared import (
        PromptSection,
        StructuredLLMClient,
        best_path_match,
        decision_from_score,
        normalize_path_text,
        pack_prompt_sections,
        load_prompt,
        render_prompt,
        weighted_composite,
    )
except ModuleNotFoundError:
    _THIS_DIR = Path(__file__).resolve().parent
    for _candidate in [_THIS_DIR, *_THIS_DIR.parents]:
        _agentic_root = _candidate / "src" / "utils" / "agentic_core"
        if (_agentic_root / "shared" / "__init__.py").exists():
            if str(_agentic_root) not in sys.path:
                sys.path.insert(0, str(_agentic_root))
            break
    from shared import (
        PromptSection,
        StructuredLLMClient,
        best_path_match,
        decision_from_score,
        normalize_path_text,
        pack_prompt_sections,
        load_prompt,
        render_prompt,
        weighted_composite,
    )


class Step02CriticReviewModel(BaseModel):
    decision: str = Field(pattern="^(PASS|REVISE)$")
    composite_score_0_1: float = Field(ge=0.0, le=1.0)
    weighted_dimensions_0_1: Dict[str, float]
    weighted_dimensions_used: Dict[str, float]
    critical_issues: List[str]
    actionable_feedback: List[str]
    rationale: str
    hard_ontology_constraint_applied: bool
    feasibility_alignment_summary: Dict[str, Any]


def collect_allowed_predictor_paths(
    *,
    complaint_unique_mapped_leaf_embed_paths: Sequence[str],
    hyde_global_top: Sequence[Any],
    mapping_payload: Dict[str, Any],
) -> List[str]:
    candidates: List[str] = []
    for raw in complaint_unique_mapped_leaf_embed_paths:
        path = normalize_path_text(str(raw or ""))
        if path:
            candidates.append(path)
    for item in hyde_global_top:
        path = normalize_path_text(str(getattr(item, "predictor_path", "") or ""))
        if path:
            candidates.append(path)
    for key in ("pre_global_top", "post_global_top"):
        for item in (mapping_payload.get(key, []) or []):
            path = normalize_path_text(str(getattr(item, "predictor_path", "") or (item or {}).get("predictor_path", "")))
            if path:
                candidates.append(path)
    per_criterion = mapping_payload.get("post_per_criterion_top", {}) or {}
    if isinstance(per_criterion, dict):
        for rows in per_criterion.values():
            for item in (rows or []):
                path = normalize_path_text(
                    str(getattr(item, "predictor_path", "") or (item or {}).get("predictor_path", ""))
                )
                if path:
                    candidates.append(path)
    out: List[str] = []
    seen: set = set()
    for path in candidates:
        key = path.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def collect_allowed_criterion_paths(criteria_rows: Sequence[Any]) -> List[str]:
    out: List[str] = []
    seen: set = set()
    for item in criteria_rows:
        raw = getattr(item, "criterion_path", None)
        path = normalize_path_text(str(raw or ""))
        if not path:
            continue
        key = path.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def extract_model_predictor_paths(model: Dict[str, Any]) -> List[str]:
    rows = model.get("predictor_variables", []) or []
    out: List[str] = []
    seen: set = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        path = normalize_path_text(str(row.get("ontology_path", "") or ""))
        if not path:
            continue
        key = path.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def critic_weights() -> Dict[str, float]:
    return {
        "schema_integrity": 0.25,
        "gvar_design_quality": 0.18,
        "evidence_grounding": 0.22,
        "feasibility_alignment": 0.20,
        "safety_scope": 0.15,
    }


def enforce_step02_hard_ontology(
    *,
    model: Dict[str, Any],
    allowed_predictor_paths: Iterable[str],
    allowed_criterion_paths: Iterable[str],
) -> Dict[str, Any]:
    predictor_replacements: List[Dict[str, Any]] = []
    criterion_replacements: List[Dict[str, Any]] = []
    predictor_violations: List[str] = []
    criterion_violations: List[str] = []

    predictor_allowed = [normalize_path_text(str(path or "")) for path in allowed_predictor_paths]
    predictor_allowed = [path for path in predictor_allowed if path]
    criterion_allowed = [normalize_path_text(str(path or "")) for path in allowed_criterion_paths]
    criterion_allowed = [path for path in criterion_allowed if path]

    predictor_vars = model.get("predictor_variables", []) or []
    for row in predictor_vars:
        if not isinstance(row, dict):
            continue
        current = normalize_path_text(str(row.get("ontology_path", "") or ""))
        if not current:
            predictor_violations.append(str(row.get("var_id") or "unknown_predictor"))
            continue
        best_match, best_score = best_path_match(current, predictor_allowed)
        if best_match and best_score >= 0.66:
            if best_match.lower() != current.lower():
                predictor_replacements.append(
                    {
                        "var_id": str(row.get("var_id") or ""),
                        "from": current,
                        "to": best_match,
                        "similarity_0_1": round(float(best_score), 4),
                    }
                )
                row["ontology_path"] = best_match
        else:
            predictor_violations.append(str(row.get("var_id") or "unknown_predictor"))
            if predictor_allowed:
                fallback = predictor_allowed[min(len(predictor_replacements), len(predictor_allowed) - 1)]
                row["ontology_path"] = fallback
                predictor_replacements.append(
                    {
                        "var_id": str(row.get("var_id") or ""),
                        "from": current,
                        "to": fallback,
                        "similarity_0_1": round(float(best_score), 4),
                    }
                )

    criterion_vars = model.get("criteria_variables", []) or []
    for row in criterion_vars:
        if not isinstance(row, dict):
            continue
        current = normalize_path_text(str(row.get("criterion_path", "") or ""))
        if not current:
            criterion_violations.append(str(row.get("var_id") or "unknown_criterion"))
            continue
        best_match, best_score = best_path_match(current, criterion_allowed)
        if best_match and best_score >= 0.66:
            if best_match.lower() != current.lower():
                criterion_replacements.append(
                    {
                        "var_id": str(row.get("var_id") or ""),
                        "from": current,
                        "to": best_match,
                        "similarity_0_1": round(float(best_score), 4),
                    }
                )
                row["criterion_path"] = best_match
        else:
            criterion_violations.append(str(row.get("var_id") or "unknown_criterion"))
            if criterion_allowed:
                fallback = criterion_allowed[min(len(criterion_replacements), len(criterion_allowed) - 1)]
                row["criterion_path"] = fallback
                criterion_replacements.append(
                    {
                        "var_id": str(row.get("var_id") or ""),
                        "from": current,
                        "to": fallback,
                        "similarity_0_1": round(float(best_score), 4),
                    }
                )

    return {
        "predictor_replacements": predictor_replacements,
        "criterion_replacements": criterion_replacements,
        "predictor_violations": predictor_violations,
        "criterion_violations": criterion_violations,
        "applied": bool(
            predictor_replacements
            or criterion_replacements
            or predictor_violations
            or criterion_violations
        ),
    }


def heuristic_step02_critic(
    *,
    model: Dict[str, Any],
    validation_report: Dict[str, Any],
    predictor_parent_feasibility: Dict[str, Any],
    hard_ontology_constraint_applied: bool,
    ontology_constraint_summary: Optional[Dict[str, Any]],
    pass_threshold: float,
) -> Step02CriticReviewModel:
    errors = [str(item) for item in (validation_report.get("errors", []) or [])]
    warnings = [str(item) for item in (validation_report.get("warnings", []) or [])]
    stats = validation_report.get("stats", {}) or {}

    predictor_rows = model.get("predictor_variables", []) or []
    criteria_rows = model.get("criteria_variables", []) or []
    n_predictors = len(predictor_rows) if isinstance(predictor_rows, list) else 0
    n_criteria = len(criteria_rows) if isinstance(criteria_rows, list) else 0
    n_total = n_predictors + n_criteria

    schema_integrity = max(0.0, 1.0 - min(1.0, (len(errors) * 0.12 + len(warnings) * 0.03)))
    t_over_k = float(stats.get("gvar_T_over_K", 0.0) or 0.0)
    gvar_design_quality = min(1.0, max(0.0, t_over_k / 5.0))
    if n_total < 6:
        gvar_design_quality = max(0.0, gvar_design_quality - 0.12)
    if n_total > 14:
        gvar_design_quality = max(0.0, gvar_design_quality - 0.08)

    if n_predictors < 2 or n_criteria < 3:
        evidence_grounding = 0.35
    else:
        evidence_grounding = 0.75
        if n_predictors >= 5 and n_criteria >= 4:
            evidence_grounding = 0.88

    top_parent = predictor_parent_feasibility.get("top_parent_domains", []) if isinstance(predictor_parent_feasibility, dict) else []
    if isinstance(top_parent, list) and top_parent:
        scores: List[float] = []
        for row in top_parent[:10]:
            try:
                scores.append(float(row.get("mean_composite_score_0_1", 0.0) or 0.0))
            except Exception:
                scores.append(0.0)
        feasibility_alignment = float(max(0.0, min(1.0, sum(scores) / max(1, len(scores)))))
    else:
        feasibility_alignment = 0.52

    safety_notes = str(model.get("safety_notes", "") or "").strip()
    safety_scope = 0.70 if safety_notes else 0.35
    if re.search(r"(emergency|urgent|professional|safety)", safety_notes, flags=re.IGNORECASE):
        safety_scope = min(1.0, safety_scope + 0.18)

    weights = critic_weights()
    composite_bundle = weighted_composite(
        subscores={
            "schema_integrity": schema_integrity,
            "gvar_design_quality": gvar_design_quality,
            "evidence_grounding": evidence_grounding,
            "feasibility_alignment": feasibility_alignment,
            "safety_scope": safety_scope,
        },
        weights=weights,
    )

    critical_issues: List[str] = []
    actionable_feedback: List[str] = []
    if errors:
        critical_issues.append(f"validator_errors:{len(errors)}")
        actionable_feedback.append(
            "Resolve validator errors first; ensure dense relevance grids and sparse edges are internally consistent."
        )
    if hard_ontology_constraint_applied and ontology_constraint_summary:
        n_violations = len(ontology_constraint_summary.get("predictor_violations", []) or []) + len(
            ontology_constraint_summary.get("criterion_violations", []) or []
        )
        if n_violations > 0:
            critical_issues.append(f"hard_ontology_violations:{n_violations}")
            actionable_feedback.append(
                "Align predictor ontology_path and criterion_path fields to valid PHOENIX ontology paths."
            )
    if feasibility_alignment < 0.45:
        actionable_feedback.append(
            "Prefer predictors with stronger multi-domain feasibility scores (data collection + translational suitability)."
        )
    if n_predictors < 4:
        actionable_feedback.append("Increase predictor breadth toward ~6 predictors when evidence supports it.")
    if n_criteria < 4:
        actionable_feedback.append("Increase criterion coverage toward ~4 criteria when evidence supports it.")

    decision = decision_from_score(
        score_0_1=float(composite_bundle.get("composite_score_0_1", 0.0)),
        threshold_0_1=float(pass_threshold),
        critical_issues=critical_issues,
    )
    if decision == "REVISE" and not actionable_feedback:
        actionable_feedback.append("Revise variable set and edge structure for stronger gVAR readiness and grounding.")

    return Step02CriticReviewModel(
        decision=decision,
        composite_score_0_1=float(composite_bundle.get("composite_score_0_1", 0.0)),
        weighted_dimensions_0_1={
            key: float(value)
            for key, value in (composite_bundle.get("subscores_0_1", {}) or {}).items()
        },
        weighted_dimensions_used={
            key: float(value)
            for key, value in (composite_bundle.get("normalized_weights", {}) or {}).items()
        },
        critical_issues=critical_issues,
        actionable_feedback=actionable_feedback[:8],
        rationale=(
            "Composite critic score combines schema integrity, gVAR design quality, evidence grounding, "
            "feasibility alignment, and safety/scope consistency."
        ),
        hard_ontology_constraint_applied=bool(hard_ontology_constraint_applied),
        feasibility_alignment_summary={
            "n_parent_domains_considered": int(len(top_parent) if isinstance(top_parent, list) else 0),
            "mean_feasibility_0_1": round(float(feasibility_alignment), 6),
        },
    )


def run_llm_step02_critic(
    *,
    llm_model: str,
    profile_id: str,
    evidence_bundle: Dict[str, Any],
    prompt_budget_tokens: int,
    timeout_seconds: float,
) -> Tuple[Optional[Step02CriticReviewModel], Dict[str, Any]]:
    try:
        system_prompt = load_prompt("step02_initial_model_critic_system.md")
        user_template = load_prompt("step02_initial_model_critic_user_template.md")
    except Exception as exc:
        return None, {"provider": "none", "reason": f"prompt_load_failed:{repr(exc)}", "profile_id": profile_id}

    serialized = json.dumps(evidence_bundle, ensure_ascii=False, indent=2)
    rendered_user = render_prompt(user_template, {"EVIDENCE_BUNDLE_JSON": serialized})
    pack = pack_prompt_sections(
        [
            PromptSection(name="step02_critic_payload", text=rendered_user, priority=1),
        ],
        max_tokens=max(4096, int(prompt_budget_tokens)),
        reserve_tokens=2200,
        model=str(llm_model),
    )
    runtime = StructuredLLMClient(
        model=str(llm_model),
        timeout_seconds=float(timeout_seconds),
        max_attempts=2,
        repair_attempts=1,
    )
    result = runtime.generate_structured(
        system_prompt=system_prompt,
        user_prompt=pack.text,
        schema_model=Step02CriticReviewModel,
    )
    trace = {
        "provider": result.provider,
        "success": bool(result.success),
        "failure_reason": result.failure_reason,
        "used_repair": bool(result.used_repair),
        "usage": result.usage,
        "pack_estimated_tokens": int(pack.estimated_tokens),
        "pack_truncated_sections": list(pack.truncated_sections),
        "profile_id": profile_id,
    }
    if not result.success or not isinstance(result.parsed, dict):
        return None, trace
    try:
        parsed = Step02CriticReviewModel.model_validate(result.parsed)
    except Exception as exc:
        trace["failure_reason"] = f"parsed_validation_failed:{repr(exc)}"
        return None, trace
    return parsed, trace
