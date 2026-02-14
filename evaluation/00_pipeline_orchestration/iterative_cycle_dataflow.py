from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_read_csv(path: Path) -> pd.DataFrame:
    for sep in [",", ";", "\t", "|"]:
        try:
            frame = pd.read_csv(path, sep=sep, engine="python")
            if frame.shape[1] > 1 or sep == ",":
                return frame
        except Exception:
            pass
    return pd.read_csv(path, engine="python")


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(number):
        return float(default)
    return float(number)


def _normalize_path(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "").strip())
    cleaned = cleaned.replace(">", "/").replace("\\", "/")
    cleaned = re.sub(r"\s*/\s*", " / ", cleaned)
    return cleaned


def _path_tokens(text: str) -> List[str]:
    return [token.lower() for token in re.findall(r"[a-zA-Z0-9_]+", _normalize_path(text))]


def path_similarity_jaccard(path_a: str, path_b: str) -> float:
    left = set(_path_tokens(path_a))
    right = set(_path_tokens(path_b))
    if not left or not right:
        return 0.0
    overlap = len(left.intersection(right))
    universe = len(left.union(right))
    if universe <= 0:
        return 0.0
    return float(overlap / universe)


def discover_profiles(root: Path, filename: str, pattern: str, max_profiles: int) -> List[str]:
    if not root.exists():
        return []
    profiles: List[str] = []
    for csv_path in sorted(root.rglob(filename)):
        profile_id = csv_path.parent.name
        if pattern and pattern not in profile_id:
            continue
        profiles.append(profile_id)
    if max_profiles > 0:
        profiles = profiles[:max_profiles]
    return profiles


def resolve_cycle_run_root(*, output_root: Path, run_id: str, cycle_index: int) -> Path:
    base = output_root / run_id
    if int(cycle_index) <= 1:
        return base
    return base / "cycles" / f"cycle_{int(cycle_index):02d}"


def resolve_previous_cycle_root(
    *,
    output_root: Path,
    run_id: str,
    cycle_index: int,
    resume_from_run: str = "",
) -> Optional[Path]:
    idx = int(cycle_index)
    if idx > 1:
        candidate = resolve_cycle_run_root(output_root=output_root, run_id=run_id, cycle_index=idx - 1)
        return candidate if candidate.exists() else None

    resume_id = str(resume_from_run).strip()
    if not resume_id:
        return None
    base = output_root / resume_id
    if not base.exists():
        return None
    cycles_root = base / "cycles"
    if not cycles_root.exists():
        return base
    cycle_dirs = sorted([p for p in cycles_root.iterdir() if p.is_dir() and p.name.startswith("cycle_")], key=lambda p: p.name)
    if not cycle_dirs:
        return base
    return cycle_dirs[-1]


def _infer_metadata_from_wide(profile_id: str, wide: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for column in wide.columns:
        if column in {"t_index", "date"}:
            continue
        role = "UNKNOWN"
        if re.match(r"^P\d{2}$", str(column)):
            role = "PREDICTOR"
        elif re.match(r"^C\d{2}$", str(column)):
            role = "CRITERION"
        rows.append(
            {
                "code": str(column),
                "role": role,
                "label": str(column),
                "ontology_id": str(column).lower(),
                "conf": "",
                "freq_hint": "",
            }
        )
    frame = pd.DataFrame(rows)
    frame.insert(0, "profile_id", profile_id)
    return frame


def _metadata_codes(frame: pd.DataFrame, role: str, available_columns: Sequence[str]) -> List[str]:
    if frame.empty:
        return []
    role_norm = str(role).upper().strip()
    subset = frame.copy()
    if "role" in subset.columns:
        subset["role_norm"] = subset["role"].astype(str).str.upper().str.strip()
        subset = subset[subset["role_norm"] == role_norm]
    codes = [str(code).strip() for code in subset.get("code", [])]
    valid = [code for code in codes if code in available_columns]
    if valid:
        return valid
    if role_norm == "PREDICTOR":
        return [col for col in available_columns if re.match(r"^P\d{2}$", str(col))]
    if role_norm == "CRITERION":
        return [col for col in available_columns if re.match(r"^C\d{2}$", str(col))]
    return []


def _load_predictor_path_map(
    *,
    step03_evidence: Dict[str, Any],
    step03_target: Dict[str, Any],
    step05_payload: Dict[str, Any],
) -> Dict[str, str]:
    mapping: Dict[str, str] = {}

    initial_model = step03_evidence.get("initial_model", {}) if isinstance(step03_evidence, dict) else {}
    for row in initial_model.get("predictor_summary", []) or []:
        if not isinstance(row, dict):
            continue
        code = str(row.get("var_id") or "").strip()
        path = _normalize_path(str(row.get("mapped_leaf_full_path") or ""))
        if code and path:
            mapping[code] = path

    for row in (step03_target.get("ranked_predictors", []) or []):
        if not isinstance(row, dict):
            continue
        code = str(row.get("predictor") or "").strip()
        path = _normalize_path(str(row.get("mapped_leaf_path") or ""))
        if code and path and code not in mapping:
            mapping[code] = path

    for row in (step05_payload.get("selected_treatment_targets", []) or []):
        if not isinstance(row, dict):
            continue
        code = str(row.get("predictor") or "").strip()
        path = _normalize_path(str(row.get("predictor_path") or ""))
        if code and path and code not in mapping:
            mapping[code] = path

    return mapping


def _load_impact_scores(impact_csv: Path) -> Dict[str, float]:
    if not impact_csv.exists():
        return {}
    frame = _safe_read_csv(impact_csv)
    if frame.empty or "predictor" not in frame.columns:
        return {}
    score_col = "predictor_impact"
    if score_col not in frame.columns and "predictor_impact_pct" in frame.columns:
        score_col = "predictor_impact_pct"
    if score_col not in frame.columns:
        return {}
    out: Dict[str, float] = {}
    for _, row in frame.iterrows():
        code = str(row.get("predictor") or "").strip()
        if not code:
            continue
        score = _to_float(row.get(score_col), default=0.0)
        if score > 1.0:
            score = score / 1000.0
        out[code] = max(0.0, min(1.0, float(score)))
    return out


def _load_step03_scores(step03_target: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for row in (step03_target.get("ranked_predictors", []) or []):
        if not isinstance(row, dict):
            continue
        code = str(row.get("predictor") or "").strip()
        if not code:
            continue
        out[code] = max(0.0, min(1.0, _to_float(row.get("score_0_1"), default=0.0)))
    return out


def _load_step05_priorities(step05_payload: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    priorities: Dict[str, float] = {}
    linked_criteria: Dict[str, List[str]] = {}
    for row in (step05_payload.get("selected_treatment_targets", []) or []):
        if not isinstance(row, dict):
            continue
        code = str(row.get("predictor") or "").strip()
        if not code:
            continue
        priorities[code] = max(0.0, min(1.0, _to_float(row.get("priority_0_1"), default=0.0)))
        linked = [str(item).strip() for item in (row.get("linked_criteria_ids", []) or []) if str(item).strip()]
        if linked:
            linked_criteria[code] = linked
    return priorities, linked_criteria


def choose_iterative_predictors(
    *,
    available_predictors: Sequence[str],
    predictor_path_map: Dict[str, str],
    recommended_paths: Sequence[str],
    impact_scores: Dict[str, float],
    step03_scores: Dict[str, float],
    step05_priorities: Dict[str, float],
    quality_scores: Optional[Dict[str, float]] = None,
    preferred_count: int,
    min_count: int,
    max_count: int,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    candidates = [str(code).strip() for code in available_predictors if str(code).strip()]
    if not candidates:
        return [], []

    rec_paths = [_normalize_path(path) for path in recommended_paths if _normalize_path(path)]
    preferred = max(int(min_count), min(int(max_count), int(preferred_count)))

    ranking_rows: List[Dict[str, Any]] = []
    quality = quality_scores or {}
    for code in candidates:
        mapped_path = predictor_path_map.get(code, "")
        similarity_best = 0.0
        for idx, rec_path in enumerate(rec_paths[:80]):
            rank_weight = 1.0 / float(idx + 1)
            similarity_best = max(similarity_best, rank_weight * path_similarity_jaccard(mapped_path, rec_path))
        impact_component = impact_scores.get(code, 0.0)
        step03_component = step03_scores.get(code, 0.0)
        step05_component = step05_priorities.get(code, 0.0)
        quality_component = quality.get(code, 0.0)
        composite = (
            0.35 * similarity_best
            + 0.25 * impact_component
            + 0.15 * step03_component
            + 0.10 * step05_component
            + 0.15 * quality_component
        )
        ranking_rows.append(
            {
                "predictor": code,
                "mapped_path": mapped_path,
                "similarity_component": round(float(similarity_best), 6),
                "impact_component": round(float(impact_component), 6),
                "step03_component": round(float(step03_component), 6),
                "step05_component": round(float(step05_component), 6),
                "quality_component": round(float(quality_component), 6),
                "composite_score_0_1": round(float(max(0.0, min(1.0, composite))), 6),
            }
        )

    ranking_rows = sorted(
        ranking_rows,
        key=lambda row: (
            float(row.get("composite_score_0_1", 0.0)),
            float(row.get("impact_component", 0.0)),
            float(row.get("step05_component", 0.0)),
            str(row.get("predictor", "")),
        ),
        reverse=True,
    )

    selected = [str(row["predictor"]) for row in ranking_rows[:preferred]]
    if len(selected) < min_count:
        for row in ranking_rows:
            code = str(row["predictor"])
            if code in selected:
                continue
            selected.append(code)
            if len(selected) >= min_count:
                break
    return selected[: max_count], ranking_rows


def choose_iterative_criteria(
    *,
    available_criteria: Sequence[str],
    retained_criteria: Sequence[str],
    severity_scores: Dict[str, float],
    quality_scores: Optional[Dict[str, float]] = None,
    preferred_count: int,
    min_count: int,
    max_count: int,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    available = [str(code).strip() for code in available_criteria if str(code).strip()]
    retained = [str(code).strip() for code in retained_criteria if str(code).strip()]
    preferred = max(int(min_count), min(int(max_count), int(preferred_count)))
    if not available:
        return [], []

    quality = quality_scores or {}
    retained_sorted = sorted(
        [code for code in retained if code in available],
        key=lambda code: (quality.get(code, 0.0), severity_scores.get(code, 0.0), code),
        reverse=True,
    )
    selected: List[str] = list(retained_sorted)
    fallback_rank = sorted(
        available,
        key=lambda code: (quality.get(code, 0.0), severity_scores.get(code, 0.0), code),
        reverse=True,
    )
    for code in fallback_rank:
        if len(selected) >= preferred:
            break
        if code in selected:
            continue
        selected.append(code)
    if len(selected) < min_count:
        for code in fallback_rank:
            if code in selected:
                continue
            selected.append(code)
            if len(selected) >= min_count:
                break

    ranking_rows = [
        {
            "criterion": code,
            "severity_score": round(float(severity_scores.get(code, 0.0)), 6),
            "quality_score": round(float(quality.get(code, 0.0)), 6),
            "retained_by_step04": bool(code in retained),
            "selected": bool(code in selected),
        }
        for code in fallback_rank
    ]
    return selected[: max_count], ranking_rows


def _column_stats(series: pd.Series, *, is_predictor: bool) -> Dict[str, float]:
    observed = pd.to_numeric(series, errors="coerce")
    non_missing = observed.dropna()
    if non_missing.empty:
        return {
            "median": 0.5 if is_predictor else 5.0,
            "std": 0.05 if is_predictor else 0.5,
            "low": 0.0 if is_predictor else 0.0,
            "high": 1.0 if is_predictor else 10.0,
            "missing_rate": 0.0,
        }
    low = float(np.nanpercentile(non_missing, 2.0))
    high = float(np.nanpercentile(non_missing, 98.0))
    if is_predictor and low >= -0.10 and high <= 1.30:
        low, high = 0.0, 1.0
    else:
        spread = max(1e-6, high - low)
        low = low - 0.05 * spread
        high = high + 0.05 * spread
        if low > 0.0:
            low = 0.0
    return {
        "median": float(np.nanmedian(non_missing)),
        "std": float(np.nanstd(non_missing, ddof=0)),
        "low": float(low),
        "high": float(high),
        "missing_rate": float(np.mean(observed.isna())),
    }


def _next_time_values(base: pd.DataFrame, step: int) -> Tuple[Optional[int], Optional[str]]:
    t_val: Optional[int] = None
    date_val: Optional[str] = None
    if "t_index" in base.columns:
        t_series = pd.to_numeric(base["t_index"], errors="coerce")
        base_t = int(np.nanmax(t_series.to_numpy())) if np.isfinite(np.nanmax(t_series.to_numpy())) else (len(base) - 1)
        t_val = int(base_t + step + 1)
    if "date" in base.columns:
        parsed = pd.to_datetime(base["date"], errors="coerce")
        if parsed.notna().any():
            last_date = parsed.dropna().iloc[-1]
            date_val = (last_date + timedelta(days=step + 1)).date().isoformat()
    return t_val, date_val


def simulate_iterative_cycle_wide(
    *,
    history_wide: pd.DataFrame,
    predictor_codes: Sequence[str],
    criterion_codes: Sequence[str],
    targeted_predictors: Sequence[str],
    targeted_criteria: Sequence[str],
    new_points: int,
    noise_scale: float,
    improvement_strength: float,
    seed: int,
) -> pd.DataFrame:
    predictors = [code for code in predictor_codes if code in history_wide.columns]
    criteria = [code for code in criterion_codes if code in history_wide.columns]
    if not predictors and not criteria:
        return history_wide.copy()

    frame = history_wide.copy()
    for column in predictors + criteria:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    stats: Dict[str, Dict[str, float]] = {}
    for code in predictors:
        stats[code] = _column_stats(frame[code], is_predictor=True)
    for code in criteria:
        stats[code] = _column_stats(frame[code], is_predictor=False)

    rng = np.random.default_rng(int(seed))
    targeted_predictor_set = set(str(code) for code in targeted_predictors)
    targeted_criteria_set = set(str(code) for code in targeted_criteria)

    last_values: Dict[str, float] = {}
    for code in predictors + criteria:
        series = frame[code].dropna()
        last_values[code] = float(series.iloc[-1]) if not series.empty else float(stats[code]["median"])

    new_rows: List[Dict[str, Any]] = []
    for step in range(max(0, int(new_points))):
        row: Dict[str, Any] = {}
        t_val, date_val = _next_time_values(frame, step)
        if "t_index" in frame.columns:
            row["t_index"] = t_val
        if "date" in frame.columns:
            row["date"] = date_val

        predictor_level_values: List[float] = []
        for code in predictors:
            cfg = stats[code]
            prev = last_values.get(code, cfg["median"])
            drift = 0.015 + (0.035 if code in targeted_predictor_set else 0.0)
            value = 0.78 * prev + 0.22 * cfg["median"] + drift + rng.normal(0.0, max(1e-6, float(noise_scale)))
            value = float(np.clip(value, cfg["low"], cfg["high"]))
            if rng.random() < min(0.12, max(0.0, cfg["missing_rate"] * 0.45)):
                row[code] = np.nan
            else:
                row[code] = value
                last_values[code] = value
                if cfg["high"] > cfg["low"]:
                    predictor_level_values.append((value - cfg["low"]) / (cfg["high"] - cfg["low"]))

        influence = float(np.mean(predictor_level_values)) if predictor_level_values else 0.5
        for code in criteria:
            cfg = stats[code]
            prev = last_values.get(code, cfg["median"])
            multiplier = 1.20 if code in targeted_criteria_set else 1.0
            improvement = float(improvement_strength) * influence * multiplier
            value = (
                0.82 * prev
                + 0.18 * cfg["median"]
                - improvement
                + rng.normal(0.0, max(1e-6, float(noise_scale) * 1.6))
            )
            value = float(np.clip(value, cfg["low"], cfg["high"]))
            if rng.random() < min(0.15, max(0.0, cfg["missing_rate"] * 0.50)):
                row[code] = np.nan
            else:
                row[code] = value
                last_values[code] = value

        new_rows.append(row)

    new_frame = pd.DataFrame(new_rows)
    if new_frame.empty:
        return frame
    ordered_cols = [col for col in frame.columns if col in new_frame.columns] + [col for col in new_frame.columns if col not in frame.columns]
    combined = pd.concat([frame, new_frame], ignore_index=True)
    return combined.reindex(columns=ordered_cols)


def _build_long_table(profile_id: str, wide: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    meta_rows = {
        str(row.get("code")): {
            "role": row.get("role", ""),
            "label": row.get("label", ""),
            "ontology_id": row.get("ontology_id", ""),
            "conf": row.get("conf", ""),
            "freq_hint": row.get("freq_hint", ""),
        }
        for row in metadata.to_dict(orient="records")
    }
    long_rows: List[Dict[str, Any]] = []
    time_cols = [col for col in ["t_index", "date"] if col in wide.columns]
    for code, info in meta_rows.items():
        if code not in wide.columns:
            continue
        for _, row in wide[time_cols + [code]].iterrows():
            payload = {
                "pseudoprofile_id": profile_id,
                "code": code,
                "value": row.get(code),
                "role": info.get("role", ""),
                "label": info.get("label", ""),
                "ontology_id": info.get("ontology_id", ""),
                "conf": info.get("conf", ""),
                "freq_hint": info.get("freq_hint", ""),
            }
            if "t_index" in time_cols:
                payload["t_index"] = row.get("t_index")
            if "date" in time_cols:
                payload["date"] = row.get("date")
            long_rows.append(payload)
    if not long_rows:
        return pd.DataFrame(columns=["pseudoprofile_id", "t_index", "date", "code", "value", "role", "label", "ontology_id", "conf", "freq_hint"])
    ordered = ["pseudoprofile_id", "t_index", "date", "code", "value", "role", "label", "ontology_id", "conf", "freq_hint"]
    return pd.DataFrame(long_rows).reindex(columns=[col for col in ordered if col in pd.DataFrame(long_rows).columns])


@dataclass
class IterativeCycleInputResult:
    active_pseudodata_root: Path
    used_previous_cycle: bool
    source_cycle_root: str
    mode: str
    profile_count: int
    profiles: List[str]
    summary_json: str


def build_iterative_cycle_input_root(
    *,
    output_root: Path,
    run_id: str,
    run_root: Path,
    cycle_index: int,
    base_pseudodata_root: Path,
    data_filename: str,
    pattern: str,
    max_profiles: int,
    enable_iterative_memory: bool,
    resume_from_run: str,
    preferred_predictor_count: int,
    preferred_criterion_count: int,
    iterative_min_predictors: int,
    iterative_max_predictors: int,
    iterative_min_criteria: int,
    iterative_max_criteria: int,
    iterative_history_points: int,
    iterative_new_points: int,
    iterative_noise_scale: float,
    iterative_improvement_strength: float,
    logger: Any,
) -> IterativeCycleInputResult:
    default_summary_path = run_root / "iterative_dataflow" / "iterative_cycle_input_summary.json"
    _ensure_dir(default_summary_path.parent)

    if not bool(enable_iterative_memory):
        summary = {
            "mode": "base_pseudodata",
            "reason": "iterative_memory_disabled",
            "cycle_index": int(cycle_index),
            "active_pseudodata_root": str(base_pseudodata_root),
        }
        default_summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return IterativeCycleInputResult(
            active_pseudodata_root=base_pseudodata_root,
            used_previous_cycle=False,
            source_cycle_root="",
            mode="base_pseudodata",
            profile_count=0,
            profiles=[],
            summary_json=str(default_summary_path),
        )

    source_cycle_root = resolve_previous_cycle_root(
        output_root=output_root,
        run_id=run_id,
        cycle_index=cycle_index,
        resume_from_run=resume_from_run,
    )
    if source_cycle_root is None:
        summary = {
            "mode": "base_pseudodata",
            "reason": "no_previous_cycle_root",
            "cycle_index": int(cycle_index),
            "active_pseudodata_root": str(base_pseudodata_root),
        }
        default_summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return IterativeCycleInputResult(
            active_pseudodata_root=base_pseudodata_root,
            used_previous_cycle=False,
            source_cycle_root="",
            mode="base_pseudodata",
            profile_count=0,
            profiles=[],
            summary_json=str(default_summary_path),
        )

    source_data_root = source_cycle_root / "_input_subset"
    fallback_data_root = base_pseudodata_root
    candidate_input_root = source_data_root if source_data_root.exists() else fallback_data_root
    profiles = discover_profiles(candidate_input_root, filename=data_filename, pattern=pattern, max_profiles=max_profiles)
    if not profiles:
        profiles = discover_profiles(fallback_data_root, filename=data_filename, pattern=pattern, max_profiles=max_profiles)
        candidate_input_root = fallback_data_root

    generated_root = run_root / "_iterative_cycle_input"
    if generated_root.exists():
        shutil.rmtree(generated_root)
    _ensure_dir(generated_root)

    summary_rows: List[Dict[str, Any]] = []
    generated_profiles: List[str] = []
    for profile_id in profiles:
        source_profile_root = candidate_input_root / profile_id
        fallback_profile_root = fallback_data_root / profile_id

        source_wide = source_profile_root / data_filename
        if not source_wide.exists():
            source_wide = fallback_profile_root / data_filename
        if not source_wide.exists():
            if hasattr(logger, "log"):
                logger.log(
                    "WARNING",
                    "iterative_dataflow.profile_skipped",
                    "Skipping profile due to missing pseudodata input.",
                    stage="iterative_dataflow",
                    profile_id=profile_id,
                    artifacts={"source_wide": str(source_wide)},
                )
            continue

        source_meta = source_profile_root / "variables_metadata.csv"
        if not source_meta.exists():
            source_meta = fallback_profile_root / "variables_metadata.csv"

        source_wide_frame = _safe_read_csv(source_wide)
        source_meta_frame = _safe_read_csv(source_meta) if source_meta.exists() else _infer_metadata_from_wide(profile_id, source_wide_frame)

        step04_path = source_cycle_root / "03_treatment_target_handoff" / profile_id / "step04_updated_observation_model.json"
        step03_bundle_path = source_cycle_root / "03_treatment_target_handoff" / profile_id / "step03_evidence_bundle.json"
        step03_target_path = source_cycle_root / "03_treatment_target_handoff" / profile_id / "step03_target_selection.json"
        step05_path = source_cycle_root / "03b_translation_digital_intervention" / profile_id / "step05_hapa_intervention.json"
        readiness_path = source_cycle_root / "00_readiness_check" / profile_id / "readiness_report.json"
        impact_csv_path = source_cycle_root / "02_momentary_impact_coefficients" / profile_id / "predictor_composite.csv"

        step04_payload = _read_json(step04_path)
        step03_bundle_payload = _read_json(step03_bundle_path)
        step03_target_payload = _read_json(step03_target_path)
        step05_payload = _read_json(step05_path)
        readiness_payload = _read_json(readiness_path)

        available_columns = list(source_wide_frame.columns)
        predictor_codes = _metadata_codes(source_meta_frame, role="PREDICTOR", available_columns=available_columns)
        criterion_codes = _metadata_codes(source_meta_frame, role="CRITERION", available_columns=available_columns)

        ready_variables = {
            str(code).strip()
            for code in ((readiness_payload.get("overall", {}) or {}).get("ready_variables", []) or [])
            if str(code).strip()
        }
        readiness_vars = readiness_payload.get("variables", {}) if isinstance(readiness_payload, dict) else {}

        predictor_quality: Dict[str, float] = {}
        for code in predictor_codes:
            var_row = readiness_vars.get(code, {}) if isinstance(readiness_vars, dict) else {}
            miss = _to_float(var_row.get("missing_pct"), default=float(pd.to_numeric(source_wide_frame.get(code), errors="coerce").isna().mean()))
            std = abs(_to_float(var_row.get("std"), default=float(pd.to_numeric(source_wide_frame.get(code), errors="coerce").std(skipna=True))))
            miss_component = max(0.0, min(1.0, 1.0 - miss))
            std_component = max(0.0, min(1.0, std / 0.15))
            ready_component = 1.0 if code in ready_variables else 0.0
            predictor_quality[code] = max(0.0, min(1.0, 0.55 * miss_component + 0.20 * std_component + 0.25 * ready_component))

        criteria_quality: Dict[str, float] = {}
        for code in criterion_codes:
            var_row = readiness_vars.get(code, {}) if isinstance(readiness_vars, dict) else {}
            miss = _to_float(var_row.get("missing_pct"), default=float(pd.to_numeric(source_wide_frame.get(code), errors="coerce").isna().mean()))
            std = abs(_to_float(var_row.get("std"), default=float(pd.to_numeric(source_wide_frame.get(code), errors="coerce").std(skipna=True))))
            miss_component = max(0.0, min(1.0, 1.0 - miss))
            std_component = max(0.0, min(1.0, std / 0.90))
            ready_component = 1.0 if code in ready_variables else 0.0
            criteria_quality[code] = max(0.0, min(1.0, 0.55 * miss_component + 0.20 * std_component + 0.25 * ready_component))

        path_map = _load_predictor_path_map(
            step03_evidence=step03_bundle_payload,
            step03_target=step03_target_payload,
            step05_payload=step05_payload,
        )
        recommended_paths = [
            _normalize_path(path)
            for path in (step04_payload.get("recommended_next_observation_predictors", []) or [])
            if _normalize_path(path)
        ]
        if not recommended_paths:
            recommended_paths = [
                _normalize_path((row or {}).get("predictor_path", ""))
                for row in (step04_payload.get("refined_predictor_shortlist", []) or [])[:40]
                if isinstance(row, dict)
            ]

        impact_scores = _load_impact_scores(impact_csv_path)
        step03_scores = _load_step03_scores(step03_target_payload)
        step05_priorities, linked_criteria_map = _load_step05_priorities(step05_payload)

        chosen_predictors, predictor_ranking = choose_iterative_predictors(
            available_predictors=predictor_codes,
            predictor_path_map=path_map,
            recommended_paths=recommended_paths,
            impact_scores=impact_scores,
            step03_scores=step03_scores,
            step05_priorities=step05_priorities,
            quality_scores=predictor_quality,
            preferred_count=int(preferred_predictor_count),
            min_count=int(iterative_min_predictors),
            max_count=int(iterative_max_predictors),
        )

        retained_criteria = [str(code).strip() for code in (step04_payload.get("retained_criteria_ids", []) or []) if str(code).strip()]
        severity_scores: Dict[str, float] = {}
        for code in criterion_codes:
            if code not in source_wide_frame.columns:
                continue
            series = pd.to_numeric(source_wide_frame[code], errors="coerce")
            severity_scores[code] = _to_float(series.mean(skipna=True), default=0.0)

        chosen_criteria, criteria_ranking = choose_iterative_criteria(
            available_criteria=criterion_codes,
            retained_criteria=retained_criteria,
            severity_scores=severity_scores,
            quality_scores=criteria_quality,
            preferred_count=int(preferred_criterion_count),
            min_count=int(iterative_min_criteria),
            max_count=int(iterative_max_criteria),
        )

        estimated_points = min(int(iterative_history_points), int(len(source_wide_frame))) + max(0, int(iterative_new_points))
        hard_min_total = int(iterative_min_predictors) + int(iterative_min_criteria)
        dynamic_cap = int(max(hard_min_total, min(int(iterative_max_predictors) + int(iterative_max_criteria), max(5, estimated_points // 15))))
        if len(chosen_predictors) + len(chosen_criteria) > dynamic_cap:
            criteria_target = max(
                int(iterative_min_criteria),
                min(len(chosen_criteria), min(int(preferred_criterion_count), dynamic_cap - int(iterative_min_predictors))),
            )
            chosen_criteria = chosen_criteria[:criteria_target]
            predictor_target = max(int(iterative_min_predictors), dynamic_cap - len(chosen_criteria))
            chosen_predictors = chosen_predictors[: max(1, predictor_target)]

        selected_codes = chosen_predictors + chosen_criteria
        selected_time_cols = [column for column in ["t_index", "date"] if column in source_wide_frame.columns]
        baseline_slice = source_wide_frame[selected_time_cols + selected_codes].copy()
        if int(iterative_history_points) > 0 and len(baseline_slice) > int(iterative_history_points):
            baseline_slice = baseline_slice.tail(int(iterative_history_points)).reset_index(drop=True)

        targeted_predictors = [code for code in step05_priorities.keys() if code in chosen_predictors]
        targeted_criteria = sorted(
            {
                criterion
                for predictor in targeted_predictors
                for criterion in linked_criteria_map.get(predictor, [])
                if criterion in chosen_criteria
            }
        )
        seed = abs(hash((profile_id, int(cycle_index), str(run_id)))) % (2**32 - 1)
        generated_wide = simulate_iterative_cycle_wide(
            history_wide=baseline_slice,
            predictor_codes=chosen_predictors,
            criterion_codes=chosen_criteria,
            targeted_predictors=targeted_predictors,
            targeted_criteria=targeted_criteria,
            new_points=int(iterative_new_points),
            noise_scale=float(iterative_noise_scale),
            improvement_strength=float(iterative_improvement_strength),
            seed=int(seed),
        )

        metadata_filtered = source_meta_frame.copy()
        if "code" in metadata_filtered.columns:
            metadata_filtered["code"] = metadata_filtered["code"].astype(str)
            metadata_filtered = metadata_filtered[metadata_filtered["code"].isin(selected_codes)]
            present_codes = set(metadata_filtered["code"].tolist())
            missing_codes = [code for code in selected_codes if code not in present_codes]
            if missing_codes:
                extras = []
                for code in missing_codes:
                    extras.append(
                        {
                            "code": code,
                            "role": "PREDICTOR" if code in chosen_predictors else "CRITERION",
                            "label": code,
                            "ontology_id": code.lower(),
                            "conf": "",
                            "freq_hint": "",
                        }
                    )
                metadata_filtered = pd.concat([metadata_filtered, pd.DataFrame(extras)], ignore_index=True)
        else:
            metadata_filtered = _infer_metadata_from_wide(profile_id, generated_wide)

        long_frame = _build_long_table(profile_id=profile_id, wide=generated_wide, metadata=metadata_filtered)

        out_profile_root = _ensure_dir(generated_root / profile_id)
        generated_wide.to_csv(out_profile_root / data_filename, index=False)
        metadata_filtered.to_csv(out_profile_root / "variables_metadata.csv", index=False)
        long_frame.to_csv(out_profile_root / "pseudodata_long.csv", index=False)

        generation_summary = {
            "profile_id": profile_id,
            "cycle_index": int(cycle_index),
            "source_cycle_root": str(source_cycle_root),
            "source_input_root": str(candidate_input_root),
            "selected_predictors": chosen_predictors,
            "selected_criteria": chosen_criteria,
            "targeted_predictors": targeted_predictors,
            "targeted_criteria": targeted_criteria,
            "history_points_used": int(len(baseline_slice)),
            "new_points_generated": int(max(0, iterative_new_points)),
            "final_points": int(len(generated_wide)),
            "data_filename": data_filename,
        }
        (out_profile_root / "generation_summary.json").write_text(
            json.dumps(generation_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (out_profile_root / "data_pattern_spec.txt").write_text(
            "\n".join(
                [
                    f"profile_id: {profile_id}",
                    f"cycle_index: {cycle_index}",
                    f"history_points_used: {len(baseline_slice)}",
                    f"new_points_generated: {max(0, iterative_new_points)}",
                    f"selected_predictors: {', '.join(chosen_predictors)}",
                    f"selected_criteria: {', '.join(chosen_criteria)}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        trace_payload = {
            "profile_id": profile_id,
            "cycle_index": int(cycle_index),
            "source_cycle_root": str(source_cycle_root),
            "source_input_root": str(candidate_input_root),
            "source_wide_path": str(source_wide),
            "source_metadata_path": str(source_meta),
            "step04_path": str(step04_path),
            "step03_bundle_path": str(step03_bundle_path),
            "step03_target_path": str(step03_target_path),
            "step05_path": str(step05_path),
            "readiness_path": str(readiness_path),
            "impact_csv_path": str(impact_csv_path),
            "ready_variables_from_previous_cycle": sorted(list(ready_variables)),
            "predictor_ranking_top20": predictor_ranking[:20],
            "criteria_ranking": criteria_ranking,
            "selected_predictors": chosen_predictors,
            "selected_criteria": chosen_criteria,
            "targeted_predictors": targeted_predictors,
            "targeted_criteria": targeted_criteria,
            "history_points_used": int(len(baseline_slice)),
            "new_points_generated": int(max(0, iterative_new_points)),
            "final_points": int(len(generated_wide)),
        }
        (out_profile_root / "iterative_dataflow_trace.json").write_text(
            json.dumps(trace_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        summary_rows.append(
            {
                "profile_id": profile_id,
                "selected_predictor_count": len(chosen_predictors),
                "selected_criterion_count": len(chosen_criteria),
                "history_points_used": int(len(baseline_slice)),
                "new_points_generated": int(max(0, iterative_new_points)),
                "final_points": int(len(generated_wide)),
            }
        )
        generated_profiles.append(profile_id)

    index_payload = {
        "mode": "iterative_cycle_generated",
        "run_id": run_id,
        "cycle_index": int(cycle_index),
        "source_cycle_root": str(source_cycle_root),
        "source_input_root": str(candidate_input_root),
        "out_root": str(generated_root),
        "profiles_generated": generated_profiles,
    }
    (generated_root / "pseudodata_index.json").write_text(
        json.dumps(index_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary_payload = {
        "mode": "iterative_cycle_generated",
        "run_id": run_id,
        "cycle_index": int(cycle_index),
        "source_cycle_root": str(source_cycle_root),
        "source_input_root": str(candidate_input_root),
        "active_pseudodata_root": str(generated_root),
        "profiles_generated": generated_profiles,
        "profile_rows": summary_rows,
    }
    default_summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(default_summary_path.with_suffix(".csv"), index=False)

    if hasattr(logger, "log"):
        logger.log(
            "INFO",
            "iterative_dataflow.generated",
            "Generated cycle input pseudodata from previous-cycle outputs.",
            stage="iterative_dataflow",
            metrics={
                "cycle_index": int(cycle_index),
                "profiles": int(len(generated_profiles)),
                "new_points_per_profile": int(max(0, iterative_new_points)),
            },
            artifacts={
                "source_cycle_root": str(source_cycle_root),
                "active_pseudodata_root": str(generated_root),
                "summary_json": str(default_summary_path),
            },
        )

    return IterativeCycleInputResult(
        active_pseudodata_root=generated_root if generated_profiles else base_pseudodata_root,
        used_previous_cycle=bool(generated_profiles),
        source_cycle_root=str(source_cycle_root),
        mode="iterative_cycle_generated" if generated_profiles else "base_pseudodata",
        profile_count=len(generated_profiles),
        profiles=generated_profiles,
        summary_json=str(default_summary_path),
    )
