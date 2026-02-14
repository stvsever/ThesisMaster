from __future__ import annotations

import json
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


LogFn = Callable[[str], None]


def _log(log_fn: Optional[LogFn], message: str) -> None:
    if log_fn is not None:
        log_fn(message)


def _to_scale_range(response_scale: str) -> Tuple[float, float]:
    text = str(response_scale or "").strip().lower()
    if not text:
        return 0.0, 9.0
    compact = text.replace(" ", "")
    match = re.search(r"(-?\d+(?:[.,]\d+)?)\s*(?:-|to|–)\s*(-?\d+(?:[.,]\d+)?)", compact)
    if not match:
        if "%" in compact:
            return 0.0, 100.0
        return 0.0, 9.0
    left = float(match.group(1).replace(",", "."))
    right = float(match.group(2).replace(",", "."))
    if left == right:
        return left, right + 1.0
    if left < right:
        return left, right
    return right, left


def _default_baseline(role: str, polarity: str) -> float:
    role_norm = str(role).upper()
    polarity_norm = str(polarity).lower()
    if role_norm == "PREDICTOR":
        return 0.55
    if polarity_norm == "higher_is_worse":
        return 0.65
    if polarity_norm == "higher_is_better":
        return 0.35
    return 0.50


def _extract_variables(model_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    criteria = list(model_payload.get("criteria_variables", []) or [])
    predictors = list(model_payload.get("predictor_variables", []) or [])

    for item in criteria:
        var_id = str(item.get("var_id") or "").strip()
        if not var_id:
            continue
        measurement = item.get("measurement", {}) if isinstance(item.get("measurement"), dict) else {}
        rows.append(
            {
                "var_id": var_id,
                "role": "CRITERION",
                "label": str(item.get("label") or var_id),
                "ontology_path": str(item.get("criterion_path") or ""),
                "polarity": str(item.get("polarity") or ""),
                "measurement_item": str(measurement.get("item_or_signal") or ""),
                "response_scale": str(measurement.get("response_scale_or_unit") or "0-9"),
                "sampling_per_day": int(measurement.get("sampling_per_day") or 1),
                "confidence": float(item.get("variable_confidence_0_1") or 0.75)
                if str(item.get("variable_confidence_0_1") or "").strip()
                else 0.75,
            }
        )

    for item in predictors:
        var_id = str(item.get("var_id") or "").strip()
        if not var_id:
            continue
        measurement = item.get("measurement", {}) if isinstance(item.get("measurement"), dict) else {}
        feasibility = item.get("feasibility", {}) if isinstance(item.get("feasibility"), dict) else {}
        rows.append(
            {
                "var_id": var_id,
                "role": "PREDICTOR",
                "label": str(item.get("label") or var_id),
                "ontology_path": str(item.get("ontology_path") or ""),
                "polarity": str(item.get("expected_direction") or ""),
                "measurement_item": str(measurement.get("item_or_signal") or ""),
                "response_scale": str(measurement.get("response_scale_or_unit") or "0-9"),
                "sampling_per_day": int(measurement.get("sampling_per_day") or 1),
                "confidence": float(feasibility.get("data_collection_feasibility_0_1") or 0.75)
                if str(feasibility.get("data_collection_feasibility_0_1") or "").strip()
                else 0.75,
            }
        )

    rows = sorted(rows, key=lambda row: str(row["var_id"]))
    return rows


def build_collection_schema(model_payload: Dict[str, Any]) -> Dict[str, Any]:
    rows = _extract_variables(model_payload)
    schema_rows: List[Dict[str, Any]] = []
    for row in rows:
        scale_min, scale_max = _to_scale_range(row.get("response_scale", ""))
        schema_rows.append(
            {
                "var_id": row["var_id"],
                "role": row["role"],
                "label": row["label"],
                "ontology_path": row["ontology_path"],
                "question": row["measurement_item"] or row["label"],
                "response_scale": row["response_scale"],
                "scale_min": scale_min,
                "scale_max": scale_max,
                "default_baseline_0_1": _default_baseline(row["role"], row.get("polarity", "")),
                "sampling_per_day": int(row.get("sampling_per_day") or 1),
            }
        )
    return {
        "variable_count": len(schema_rows),
        "criteria_count": int(sum(1 for item in schema_rows if item["role"] == "CRITERION")),
        "predictor_count": int(sum(1 for item in schema_rows if item["role"] == "PREDICTOR")),
        "variables": schema_rows,
    }


def _generate_signal(
    *,
    n_points: int,
    baseline: float,
    rng: np.random.Generator,
    ar: float = 0.72,
    noise_std: float = 0.08,
    seasonal_period: int = 14,
) -> np.ndarray:
    values = np.zeros(n_points, dtype=float)
    values[0] = float(np.clip(rng.normal(loc=baseline, scale=noise_std), 0.0, 1.0))
    phase = rng.uniform(0.0, np.pi * 2.0)
    seasonal = 0.05 * np.sin((np.arange(n_points) * 2.0 * np.pi / max(2, seasonal_period)) + phase)
    for idx in range(1, n_points):
        innovation = float(rng.normal(loc=0.0, scale=noise_std))
        values[idx] = (ar * values[idx - 1]) + ((1.0 - ar) * baseline) + innovation + seasonal[idx]
        values[idx] = float(np.clip(values[idx], 0.0, 1.0))
    return values


def _apply_scale(signal: np.ndarray, lower: float, upper: float) -> np.ndarray:
    width = float(upper - lower)
    if width <= 1e-9:
        return np.full_like(signal, fill_value=lower)
    scaled = lower + (signal * width)
    return np.round(scaled, 4)


def synthesize_pseudodata(
    *,
    model_payload: Dict[str, Any],
    profile_id: str,
    output_profile_root: Path,
    n_points: int = 84,
    missing_rate: float = 0.10,
    seed: int = 42,
    baseline_overrides: Optional[Dict[str, float]] = None,
    log_fn: Optional[LogFn] = None,
) -> Dict[str, Any]:
    output_profile_root.mkdir(parents=True, exist_ok=True)
    schema = build_collection_schema(model_payload)
    variables = list(schema.get("variables", []))
    if not variables:
        raise RuntimeError("Observation model has no variables; cannot synthesize pseudodata.")

    rng = np.random.default_rng(int(seed))
    n_points = max(10, int(n_points))
    missing_rate = float(max(0.0, min(0.70, missing_rate)))
    overrides = baseline_overrides or {}

    _log(log_fn, f"Synthesizing pseudodata for {profile_id} with n_points={n_points}, missing_rate={missing_rate:.2f}.")

    start_day = date.today() - timedelta(days=max(1, n_points - 1))
    dates = [start_day + timedelta(days=index) for index in range(n_points)]
    wide = pd.DataFrame(
        {
            "t_index": np.arange(n_points, dtype=int),
            "date": [day.isoformat() for day in dates],
        }
    )
    metadata_rows: List[Dict[str, Any]] = []
    generation_details: List[Dict[str, Any]] = []

    for variable in variables:
        var_id = str(variable["var_id"])
        role = str(variable["role"])
        scale_min = float(variable.get("scale_min") or 0.0)
        scale_max = float(variable.get("scale_max") or 9.0)
        baseline = float(overrides.get(var_id, variable.get("default_baseline_0_1", 0.5)))
        baseline = float(max(0.05, min(0.95, baseline)))
        signal = _generate_signal(
            n_points=n_points,
            baseline=baseline,
            rng=rng,
            ar=0.70 if role == "CRITERION" else 0.78,
            noise_std=0.10 if role == "CRITERION" else 0.07,
            seasonal_period=14,
        )
        scaled = _apply_scale(signal, scale_min, scale_max)
        miss_mask = rng.uniform(size=n_points) < missing_rate
        scaled = scaled.astype(float)
        scaled[miss_mask] = np.nan
        wide[var_id] = scaled

        metadata_rows.append(
            {
                "code": var_id,
                "role": role,
                "label": variable.get("label", var_id),
                "ontology_id": variable.get("ontology_path", ""),
                "conf": round(float(variable.get("default_baseline_0_1", 0.5)), 3),
                "freq_hint": variable.get("sampling_per_day", 1),
            }
        )
        generation_details.append(
            {
                "var_id": var_id,
                "role": role,
                "baseline_0_1": round(baseline, 4),
                "scale_min": scale_min,
                "scale_max": scale_max,
                "missing_rate_empirical": round(float(np.mean(np.isnan(scaled))), 4),
            }
        )

    long_df = wide.melt(
        id_vars=["t_index", "date"],
        var_name="variable",
        value_name="value",
    ).sort_values(["t_index", "variable"])
    metadata_df = pd.DataFrame(metadata_rows)

    wide_path = output_profile_root / "pseudodata_wide.csv"
    long_path = output_profile_root / "pseudodata_long.csv"
    metadata_path = output_profile_root / "variables_metadata.csv"
    summary_path = output_profile_root / "generation_summary.json"
    spec_path = output_profile_root / "data_pattern_spec.txt"

    wide.to_csv(wide_path, index=False)
    long_df.to_csv(long_path, index=False)
    metadata_df.to_csv(metadata_path, index=False)

    summary_payload = {
        "profile_id": profile_id,
        "n_points": n_points,
        "missing_rate_target": missing_rate,
        "variables": generation_details,
        "criteria_count": schema["criteria_count"],
        "predictor_count": schema["predictor_count"],
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    spec_lines = [
        f"profile_id={profile_id}",
        f"n_points={n_points}",
        f"missing_rate_target={missing_rate}",
        f"criteria_count={schema['criteria_count']}",
        f"predictor_count={schema['predictor_count']}",
        "",
        "variables:",
    ]
    for row in generation_details:
        spec_lines.append(
            f"- {row['var_id']} ({row['role']}): baseline={row['baseline_0_1']}, "
            f"scale=[{row['scale_min']},{row['scale_max']}], missing_emp={row['missing_rate_empirical']}"
        )
    spec_path.write_text("\n".join(spec_lines) + "\n", encoding="utf-8")

    _log(log_fn, f"Wrote pseudodata files under: {output_profile_root}")

    return {
        "profile_id": profile_id,
        "n_points": n_points,
        "missing_rate_target": missing_rate,
        "wide_csv": str(wide_path),
        "long_csv": str(long_path),
        "metadata_csv": str(metadata_path),
        "summary_json": str(summary_path),
        "spec_txt": str(spec_path),
        "criteria_count": schema["criteria_count"],
        "predictor_count": schema["predictor_count"],
        "variable_count": schema["variable_count"],
    }


def parse_baseline_overrides(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for row in rows:
        var_id = str(row.get("var_id") or "").strip()
        if not var_id:
            continue
        try:
            baseline = float(row.get("baseline_0_1"))
        except Exception:
            continue
        out[var_id] = max(0.0, min(1.0, baseline))
    return out
