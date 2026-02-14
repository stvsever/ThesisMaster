from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    return []


def _safe_str(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def build_cycle_summary(
    *,
    profile_id: str,
    cycle_index: int,
    step03_payload: Dict[str, Any],
    step04_payload: Dict[str, Any],
    step05_payload: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    step05_payload = step05_payload or {}
    recommended_targets = _as_list(step03_payload.get("recommended_targets"))
    ranked_predictors = _as_list(step03_payload.get("ranked_predictors"))
    updated_predictors = _as_list(step04_payload.get("recommended_next_observation_predictors"))
    retained_criteria = _as_list(step04_payload.get("retained_criteria_ids"))
    intervention_targets = _as_list(step05_payload.get("selected_treatment_targets"))
    barrier_priorities = _as_list(step05_payload.get("barrier_priorities"))
    coping_priorities = _as_list(step05_payload.get("coping_priorities"))

    selected_target_paths = [
        _safe_str(item.get("mapped_leaf_path") if isinstance(item, dict) else "")
        for item in recommended_targets
    ]
    selected_target_paths = [path for path in selected_target_paths if path]

    return {
        "profile_id": _safe_str(profile_id),
        "cycle_index": int(cycle_index),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "step03": {
            "recommended_target_count": int(len(recommended_targets)),
            "ranked_predictor_count": int(len(ranked_predictors)),
            "recommended_target_paths": selected_target_paths,
        },
        "step04": {
            "retained_criteria_ids": [_safe_str(item) for item in retained_criteria if _safe_str(item)],
            "updated_predictors": [_safe_str(item) for item in updated_predictors if _safe_str(item)],
            "updated_predictor_count": int(len(updated_predictors)),
            "rationale": _safe_str(step04_payload.get("rationale")),
        },
        "step05": {
            "selected_treatment_target_count": int(len(intervention_targets)),
            "barrier_count": int(len(barrier_priorities)),
            "coping_count": int(len(coping_priorities)),
        },
        "lineage": {
            "previous_model_reference": _safe_str(step04_payload.get("previous_model_reference")),
            "memory_used": bool(step04_payload.get("memory_used", False)),
        },
    }
