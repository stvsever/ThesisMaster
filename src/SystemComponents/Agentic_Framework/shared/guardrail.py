from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .target_refinement import normalize_path_text, path_similarity


def clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def normalize_score(raw: Any) -> float:
    try:
        value = float(raw)
    except Exception:
        return 0.0
    if value > 1.0:
        value = value / 100.0 if value <= 100.0 else value / 1000.0
    return clamp01(value)


def weighted_composite(
    *,
    subscores: Dict[str, Any],
    weights: Dict[str, float],
) -> Dict[str, Any]:
    clean_weights: Dict[str, float] = {}
    for key, value in weights.items():
        try:
            clean_weights[str(key)] = max(0.0, float(value))
        except Exception:
            clean_weights[str(key)] = 0.0
    total_weight = float(sum(clean_weights.values()))
    if total_weight <= 0.0:
        total_weight = float(max(1, len(clean_weights)))
        clean_weights = {key: 1.0 for key in clean_weights}
    normalized_weights = {key: float(value / total_weight) for key, value in clean_weights.items()}
    clean_scores = {str(key): normalize_score(value) for key, value in subscores.items()}

    composite = 0.0
    coverage_missing: List[str] = []
    for key, norm_weight in normalized_weights.items():
        if key not in clean_scores:
            coverage_missing.append(key)
            continue
        composite += float(norm_weight * clean_scores[key])
    return {
        "composite_score_0_1": round(clamp01(composite), 6),
        "normalized_weights": normalized_weights,
        "subscores_0_1": clean_scores,
        "missing_weighted_dimensions": coverage_missing,
    }


def decision_from_score(
    *,
    score_0_1: float,
    threshold_0_1: float,
    critical_issues: Sequence[str] | None = None,
) -> str:
    critical_count = len(list(critical_issues or []))
    if critical_count > 0:
        return "REVISE"
    return "PASS" if float(score_0_1) >= float(threshold_0_1) else "REVISE"


def best_path_match(
    candidate: str,
    allowed_paths: Iterable[str],
) -> Tuple[str, float]:
    normalized_candidate = normalize_path_text(str(candidate or ""))
    if not normalized_candidate:
        return "", 0.0
    allowed = [normalize_path_text(str(path or "")) for path in allowed_paths]
    allowed = [path for path in allowed if path]
    if not allowed:
        return "", 0.0
    if normalized_candidate in allowed:
        return normalized_candidate, 1.0
    best_path = ""
    best_score = 0.0
    for path in allowed:
        score = path_similarity(normalized_candidate, path)
        if score > best_score:
            best_score = score
            best_path = path
    return best_path, float(best_score)

