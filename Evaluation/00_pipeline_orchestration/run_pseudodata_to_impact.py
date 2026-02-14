#!/usr/bin/env python3
"""
run_pseudodata_to_impact.py

End-to-end integration runner for:
1) Readiness check
2) Network time-series analysis (tv-gVAR + baselines)
3) Momentary impact coefficient quantification
4) Treatment-target identification (Agentic step 03)
5) Updated observation-model suggestion (Agentic step 04)
6) HAPA-based digital intervention generation (Agentic step 05)
7) Impact visualization generation
8) Research communication report generation

Scope:
- Starts from existing pseudodata input folders.
- Does NOT modify ontology structure.
- Produces run-scoped outputs under Evaluation/05_integrated_pipeline_runs/.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _run_id_now() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def _bool_str(value: bool) -> str:
    return "True" if value else "False"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class StageResult:
    stage: str
    command: List[str]
    return_code: int
    duration_seconds: float
    log_path: str


class PipelineLogger:
    def __init__(self, jsonl_path: Path, run_id: str = "") -> None:
        self.jsonl_path = jsonl_path
        self.run_id = str(run_id).strip() or "unknown_run"
        _ensure_dir(self.jsonl_path.parent)

    def log(
        self,
        level: str,
        event: str,
        message: str,
        *,
        stage: str = "pipeline",
        profile_id: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        artifacts: Optional[Dict[str, Any]] = None,
        exception: Optional[str] = None,
        **fields: object,
    ) -> None:
        payload: Dict[str, object] = {
            "timestamp_local": _ts(),
            "run_id": self.run_id,
            "profile_id": profile_id,
            "stage": stage,
            "level": level.upper(),
            "event": event,
            "message": message,
            "metrics": metrics or {},
            "artifacts": artifacts or {},
            "exception": exception,
        }
        if fields:
            payload["fields"] = fields

        print(f"[{payload['timestamp_local']}] [{payload['level']}] {event}: {message}", flush=True)
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_stage_event(
    event_path: Path,
    *,
    run_id: str,
    profile_id: Optional[str],
    stage: str,
    event: str,
    level: str,
    message: str,
    metrics: Optional[Dict[str, Any]] = None,
    artifacts: Optional[Dict[str, Any]] = None,
    exception: Optional[str] = None,
) -> None:
    _ensure_dir(event_path.parent)
    payload = {
        "timestamp_local": _ts(),
        "run_id": run_id,
        "profile_id": profile_id,
        "stage": stage,
        "event": event,
        "level": level.upper(),
        "message": message,
        "metrics": metrics or {},
        "artifacts": artifacts or {},
        "exception": exception,
    }
    with event_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_stage_trace(
    trace_path: Path,
    *,
    run_id: str,
    cycle_index: int,
    stage: str,
    result: StageResult,
    profile_count: int,
    contract_results: Optional[List[Dict[str, Any]]] = None,
) -> None:
    payload = {
        "timestamp_local": _ts(),
        "run_id": run_id,
        "cycle_index": int(cycle_index),
        "stage": stage,
        "duration_seconds": round(float(result.duration_seconds), 3),
        "return_code": int(result.return_code),
        "command": list(result.command),
        "profile_count": int(profile_count),
        "contract_results": contract_results or [],
    }
    trace_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def discover_profiles(pseudodata_root: Path, filename: str, pattern: str, max_profiles: int) -> List[str]:
    profiles: List[str] = []
    for csv_path in sorted(pseudodata_root.rglob(filename)):
        profile_id = csv_path.parent.name
        if pattern and pattern not in profile_id:
            continue
        profiles.append(profile_id)
    if max_profiles > 0:
        profiles = profiles[:max_profiles]
    return profiles


def materialize_profile_subset_root(
    pseudodata_root: Path,
    selected_profiles: Sequence[str],
    subset_root: Path,
    logger: PipelineLogger,
) -> Path:
    """
    Some component scripts only support --max-profiles and do not accept a --pattern filter.
    To ensure those scripts operate on the intended set, we create a run-scoped subset root
    containing only the selected profile folders (copy).
    """
    if subset_root.exists():
        shutil.rmtree(subset_root)
    _ensure_dir(subset_root)

    for profile_id in selected_profiles:
        src = pseudodata_root / profile_id
        dst = subset_root / profile_id
        if not src.exists():
            logger.log("WARNING", "pipeline.subset_missing", "Profile folder missing in pseudodata root.", profile_id=profile_id, src=str(src))
            continue
        shutil.copytree(src, dst, dirs_exist_ok=True)

    logger.log(
        "INFO",
        "pipeline.subset_ready",
        "Materialized run-scoped pseudodata subset root.",
        subset_root=str(subset_root),
        profile_count=len(list(selected_profiles)),
    )
    return subset_root


def discover_output_profiles(stage_root: Path, required_filename: str, pattern: str, max_profiles: int) -> List[str]:
    profiles: List[str] = []
    if not stage_root.exists():
        return profiles
    for fpath in sorted(stage_root.rglob(required_filename)):
        profile_id = fpath.parent.name
        if pattern and pattern not in profile_id:
            continue
        profiles.append(profile_id)
    if max_profiles > 0:
        profiles = profiles[:max_profiles]
    return profiles


def run_stage(
    stage: str,
    cmd: Sequence[str],
    stage_log_path: Path,
    logger: PipelineLogger,
    run_id: str,
    cycle_index: int,
    profile_count: int,
    stage_events_path: Path,
    stage_trace_path: Path,
    profile_id: Optional[str] = None,
    contract_results: Optional[List[Dict[str, Any]]] = None,
    env_overrides: Optional[Dict[str, str]] = None,
) -> StageResult:
    start = time.time()
    logger.log("INFO", f"{stage}.start", "Running stage command.", stage=stage, command=" ".join(cmd))
    _write_stage_event(
        stage_events_path,
        run_id=run_id,
        profile_id=profile_id,
        stage=stage,
        event=f"{stage}.start",
        level="INFO",
        message="Running stage command.",
        metrics={"profile_count": profile_count, "cycle_index": cycle_index},
        artifacts={"command": " ".join(cmd)},
    )

    process_env = os.environ.copy()
    if env_overrides:
        process_env.update({str(k): str(v) for k, v in env_overrides.items()})

    _ensure_dir(stage_log_path.parent)
    with stage_log_path.open("w", encoding="utf-8") as stage_log:
        process = subprocess.Popen(
            list(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=process_env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            clean = line.rstrip("\n")
            prefixed = f"[{stage}] {clean}"
            print(prefixed, flush=True)
            stage_log.write(prefixed + "\n")
        return_code = process.wait()

    duration = time.time() - start
    result = StageResult(
        stage=stage,
        command=list(cmd),
        return_code=return_code,
        duration_seconds=duration,
        log_path=str(stage_log_path),
    )

    if return_code != 0:
        logger.log(
            "ERROR",
            f"{stage}.failed",
            "Stage returned non-zero exit code.",
            stage=stage,
            return_code=return_code,
            duration_seconds=round(duration, 3),
            log_path=str(stage_log_path),
        )
        _write_stage_event(
            stage_events_path,
            run_id=run_id,
            profile_id=profile_id,
            stage=stage,
            event=f"{stage}.failed",
            level="ERROR",
            message="Stage returned non-zero exit code.",
            metrics={"return_code": return_code, "duration_seconds": round(duration, 3), "cycle_index": cycle_index},
            artifacts={"log_path": str(stage_log_path)},
        )
        _write_stage_trace(
            stage_trace_path,
            run_id=run_id,
            cycle_index=cycle_index,
            stage=stage,
            result=result,
            profile_count=profile_count,
            contract_results=contract_results,
        )
        raise RuntimeError(f"Stage '{stage}' failed with exit code {return_code}.")

    logger.log(
        "INFO",
        f"{stage}.done",
        "Stage completed successfully.",
        stage=stage,
        duration_seconds=round(duration, 3),
        log_path=str(stage_log_path),
    )
    _write_stage_event(
        stage_events_path,
        run_id=run_id,
        profile_id=profile_id,
        stage=stage,
        event=f"{stage}.done",
        level="INFO",
        message="Stage completed successfully.",
        metrics={"duration_seconds": round(duration, 3), "return_code": int(return_code), "cycle_index": cycle_index},
        artifacts={"log_path": str(stage_log_path)},
    )
    _write_stage_trace(
        stage_trace_path,
        run_id=run_id,
        cycle_index=cycle_index,
        stage=stage,
        result=result,
        profile_count=profile_count,
        contract_results=contract_results,
    )
    return result


def validate_stage_outputs(
    stage_name: str,
    root: Path,
    profiles: Sequence[str],
    required_relpaths: Sequence[str],
    logger: PipelineLogger,
) -> None:
    if not root.exists():
        logger.log(
            "ERROR",
            f"{stage_name}.validation_failed",
            "Stage root does not exist.",
            stage_root=str(root),
        )
        raise RuntimeError(f"Validation failed for stage '{stage_name}'. Stage root does not exist: {root}")

    missing: List[str] = []
    for profile_id in profiles:
        for relpath in required_relpaths:
            expected = root / profile_id / relpath
            if not expected.exists():
                missing.append(str(expected))

    if missing:
        logger.log(
            "ERROR",
            f"{stage_name}.validation_failed",
            "Required outputs are missing.",
            missing_count=len(missing),
            first_missing=missing[:20],
        )
        raise RuntimeError(f"Validation failed for stage '{stage_name}'. Missing files: {len(missing)}")

    logger.log(
        "INFO",
        f"{stage_name}.validation_ok",
        "All required outputs found.",
        profile_count=len(profiles),
        required_files_per_profile=len(required_relpaths),
    )


def validate_root_outputs(
    stage_name: str,
    root: Path,
    required_relpaths: Sequence[str],
    logger: PipelineLogger,
) -> None:
    if not root.exists():
        logger.log(
            "ERROR",
            f"{stage_name}.validation_failed",
            "Stage root does not exist.",
            stage_root=str(root),
        )
        raise RuntimeError(f"Validation failed for stage '{stage_name}'. Stage root does not exist: {root}")

    missing: List[str] = []
    for relpath in required_relpaths:
        expected = root / relpath
        if not expected.exists():
            missing.append(str(expected))

    if missing:
        logger.log(
            "ERROR",
            f"{stage_name}.validation_failed",
            "Required root-level outputs are missing.",
            missing_count=len(missing),
            first_missing=missing[:20],
        )
        raise RuntimeError(f"Validation failed for stage '{stage_name}'. Missing files: {len(missing)}")

    logger.log(
        "INFO",
        f"{stage_name}.validation_ok",
        "All required root-level outputs found.",
        required_files=len(required_relpaths),
    )


def _load_contract_validator(repo_root: Path):
    agentic_core_root = repo_root / "src/utils/agentic_core"
    if str(agentic_core_root) not in sys.path:
        sys.path.insert(0, str(agentic_core_root))
    from shared import ContractValidator  # type: ignore

    return ContractValidator()


def validate_contract_files(
    *,
    validator,
    contract_name: str,
    root: Path,
    profiles: Sequence[str],
    relpath: str,
    logger: PipelineLogger,
    stage: str,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for profile_id in profiles:
        payload_path = root / profile_id / relpath
        result = validator.validate_file(contract_name=contract_name, payload_path=payload_path)
        results.append(
            {
                "profile_id": profile_id,
                "contract_name": contract_name,
                "success": bool(result.success),
                "path": result.path,
                "message": result.message,
                "errors": result.errors or [],
            }
        )
        if not result.success:
            logger.log(
                "ERROR",
                f"{stage}.contract_failed",
                "Contract validation failed.",
                stage=stage,
                profile_id=profile_id,
                artifacts={"path": result.path},
                metrics={"contract_name": contract_name},
                exception="; ".join(result.errors or [])[:1000],
            )
            raise RuntimeError(f"Contract validation failed for {contract_name}: {payload_path}")
    logger.log(
        "INFO",
        f"{stage}.contract_ok",
        "Contract validation passed.",
        stage=stage,
        metrics={"profiles": len(profiles), "contract_name": contract_name},
    )
    return results


def append_history_ledger(
    *,
    history_root: Path,
    run_id: str,
    cycle_index: int,
    profile_ids: Sequence[str],
    step04_root: Path,
    step05_root: Path,
    impact_root: Path,
    logger: PipelineLogger,
) -> None:
    _ensure_dir(history_root)
    profile_events_path = history_root / "profile_events.jsonl"
    snapshot_rows: List[Dict[str, Any]] = []
    lineage_rows: List[Dict[str, Any]] = []

    for profile_id in profile_ids:
        step04_path = step04_root / profile_id / "step04_updated_observation_model.json"
        step05_path = step05_root / profile_id / "step05_hapa_intervention.json"
        impact_path = impact_root / profile_id / "momentary_impact.json"
        top_predictors: List[str] = []
        if impact_path.exists():
            impact_payload = json.loads(impact_path.read_text(encoding="utf-8"))
            overall = impact_payload.get("overall", {}) or {}
            candidates = overall.get("ranked_predictors", []) or overall.get("predictors", []) or []
            if isinstance(candidates, list):
                for row in candidates[:8]:
                    if isinstance(row, dict):
                        pid = str(row.get("predictor") or row.get("predictor_id") or "").strip()
                        if pid:
                            top_predictors.append(pid)

        next_predictors: List[str] = []
        retained_criteria: List[str] = []
        if step04_path.exists():
            step04_payload = json.loads(step04_path.read_text(encoding="utf-8"))
            next_predictors = list(step04_payload.get("recommended_next_observation_predictors", []) or [])
            retained_criteria = list(step04_payload.get("retained_criteria_ids", []) or [])

        barrier_count = 0
        coping_count = 0
        if step05_path.exists():
            step05_payload = json.loads(step05_path.read_text(encoding="utf-8"))
            barrier_count = int(len(step05_payload.get("selected_barriers", []) or []))
            coping_count = int(len(step05_payload.get("selected_coping_strategies", []) or []))

        event_payload = {
            "timestamp_local": _ts(),
            "run_id": run_id,
            "cycle_index": int(cycle_index),
            "profile_id": profile_id,
            "event": "cycle_summary",
            "top_predictors": top_predictors[:8],
            "next_predictors": next_predictors[:20],
            "retained_criteria": retained_criteria[:10],
            "barrier_count": barrier_count,
            "coping_count": coping_count,
        }
        with profile_events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event_payload, ensure_ascii=False) + "\n")

        snapshot_rows.append(
            {
                "run_id": run_id,
                "cycle_index": int(cycle_index),
                "profile_id": profile_id,
                "top_predictors": "|".join(top_predictors[:8]),
                "next_predictors": "|".join(next_predictors[:20]),
                "retained_criteria": "|".join(retained_criteria[:10]),
                "barrier_count": barrier_count,
                "coping_count": coping_count,
            }
        )
        lineage_rows.append(
            {
                "run_id": run_id,
                "cycle_index": int(cycle_index),
                "profile_id": profile_id,
                "from_model": f"cycle_{max(0, cycle_index - 1):02d}",
                "to_model": f"cycle_{cycle_index:02d}",
                "n_recommended_predictors": len(next_predictors),
                "n_retained_criteria": len(retained_criteria),
            }
        )

    if snapshot_rows:
        snapshot_df = pd.DataFrame(snapshot_rows)
        lineage_df = pd.DataFrame(lineage_rows)
        snapshots_path = history_root / "feature_snapshots.parquet"
        lineage_path = history_root / "model_lineage.parquet"
        if snapshots_path.exists():
            prev = pd.read_parquet(snapshots_path)
            snapshot_df = pd.concat([prev, snapshot_df], ignore_index=True)
        if lineage_path.exists():
            prev = pd.read_parquet(lineage_path)
            lineage_df = pd.concat([prev, lineage_df], ignore_index=True)
        snapshot_df.to_parquet(snapshots_path, index=False)
        lineage_df.to_parquet(lineage_path, index=False)

    logger.log(
        "INFO",
        "history.ledger_updated",
        "Iterative memory history ledger updated.",
        stage="pipeline",
        metrics={"profiles": len(profile_ids), "cycle_index": int(cycle_index)},
        artifacts={
            "profile_events_jsonl": str(profile_events_path),
            "feature_snapshots_parquet": str(history_root / "feature_snapshots.parquet"),
            "model_lineage_parquet": str(history_root / "model_lineage.parquet"),
        },
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    repo_root_default = _repo_root_from_here()
    pseudodata_default = repo_root_default / "Evaluation/01_pseudoprofile(s)/time_series_data/pseudodata"
    runs_default = repo_root_default / "Evaluation/05_integrated_pipeline_runs"

    parser = argparse.ArgumentParser(
        description="Integrated pseudodata->impact pipeline with clear run-scoped logs and validation."
    )
    parser.add_argument("--repo-root", type=str, default=str(repo_root_default))
    parser.add_argument("--python-exe", type=str, default=sys.executable)

    parser.add_argument("--pseudodata-root", type=str, default=str(pseudodata_default))
    parser.add_argument("--output-root", type=str, default=str(runs_default))
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument(
        "--readiness-source-root",
        type=str,
        default="",
        help="Optional existing readiness output root to reuse when --skip-readiness is enabled.",
    )
    parser.add_argument(
        "--network-source-root",
        type=str,
        default="",
        help="Optional existing network output root to reuse when --skip-network is enabled.",
    )
    parser.add_argument(
        "--impact-source-root",
        type=str,
        default="",
        help="Optional existing impact output root to reuse when --skip-impact is enabled.",
    )
    parser.add_argument("--pattern", type=str, default="pseudoprofile_FTC_")
    parser.add_argument("--max-profiles", type=int, default=0)
    parser.add_argument("--data-filename", type=str, default="pseudodata_wide.csv")

    parser.add_argument("--lag", type=int, default=1)
    parser.add_argument("--llm-finalize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prefer-time-varying", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--readiness-quiet", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument(
        "--network-prefer-tier",
        type=str,
        default="overall_ready",
        choices=["tier1", "overall_ready", "all_non_hard"],
    )
    parser.add_argument(
        "--network-execution-policy",
        type=str,
        default="readiness_aligned",
        choices=["readiness_aligned", "all_methods"],
    )
    parser.add_argument("--network-boot", type=int, default=80)
    parser.add_argument("--network-block-len", type=int, default=20)
    parser.add_argument("--network-jobs", type=int, default=1)
    parser.add_argument("--network-verbose", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--impact-half-life", type=float, default=0.20)
    parser.add_argument("--impact-top-k-edges", type=int, default=200)

    parser.add_argument("--handoff-top-k", type=int, default=15)
    parser.add_argument("--handoff-min-impact", type=float, default=0.10)
    parser.add_argument("--handoff-max-candidate-predictors", type=int, default=200)
    parser.add_argument("--handoff-llm-model", type=str, default="gpt-5-nano")
    parser.add_argument("--handoff-disable-llm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--handoff-prompt-budget-tokens", type=int, default=400000)
    parser.add_argument("--handoff-critic-max-iterations", type=int, default=2)
    parser.add_argument("--handoff-critic-pass-threshold", type=float, default=0.74)
    parser.add_argument("--handoff-preferred-predictor-count", type=int, default=6)
    parser.add_argument("--handoff-preferred-criterion-count", type=int, default=4)
    parser.add_argument("--run-intervention-step", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--intervention-llm-model", type=str, default="gpt-5-nano")
    parser.add_argument("--intervention-disable-llm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--intervention-prompt-budget-tokens", type=int, default=400000)
    parser.add_argument("--intervention-critic-max-iterations", type=int, default=2)
    parser.add_argument("--intervention-critic-pass-threshold", type=float, default=0.74)
    parser.add_argument(
        "--predictor-feasibility-csv",
        type=str,
        default="",
        help="Optional predictor feasibility summary for critic grounding.",
    )
    parser.add_argument("--parent-feasibility-top-k", type=int, default=30)
    parser.add_argument("--hard-ontology-constraint", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--run-impact-visualizations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visualization-dpi", type=int, default=300)
    parser.add_argument("--visualization-top-k-edges", type=int, default=60)
    parser.add_argument("--visualization-top-n-heatmap", type=int, default=25)
    parser.add_argument("--visualization-top-n-bars", type=int, default=25)

    parser.add_argument("--run-component-report", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-iterative-memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cycle-index", type=int, default=1)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument(
        "--memory-policy",
        type=str,
        default="v1_weighted_fusion",
        choices=["v1_weighted_fusion", "none"],
    )
    parser.add_argument(
        "--resume-from-run",
        type=str,
        default="",
        help="Optional prior run_id used as iterative-memory seed metadata.",
    )
    parser.add_argument(
        "--history-root",
        type=str,
        default="",
        help="Path for persistent cross-run history ledger. Defaults to <output-root>/_history.",
    )
    parser.add_argument(
        "--profile-memory-window",
        type=int,
        default=3,
        help="Max prior cycle windows to consider in memory metadata.",
    )

    parser.add_argument("--skip-readiness", action="store_true")
    parser.add_argument("--skip-network", action="store_true")
    parser.add_argument("--skip-impact", action="store_true")
    parser.add_argument("--skip-handoff", action="store_true")
    parser.add_argument("--skip-intervention", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    repo_root = Path(args.repo_root).expanduser().resolve()
    pseudodata_root = Path(args.pseudodata_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    run_id = args.run_id.strip() or _run_id_now()
    cycle_index = max(1, int(args.cycle_index))
    cycles = max(1, int(args.cycles))
    run_root = _ensure_dir(output_root / run_id)
    if cycle_index > 1:
        run_root = _ensure_dir(run_root / "cycles" / f"cycle_{cycle_index:02d}")
    logs_root = _ensure_dir(run_root / "logs")
    history_root = _ensure_dir(
        Path(args.history_root).expanduser().resolve()
        if str(args.history_root).strip()
        else (output_root / "_history")
    )
    runtime_env = {
        "MPLCONFIGDIR": str(_ensure_dir(run_root / ".mplconfig")),
        "XDG_CACHE_HOME": str(_ensure_dir(run_root / ".cache")),
    }

    readiness_root = _ensure_dir(run_root / "00_readiness_check")
    network_root = _ensure_dir(run_root / "01_time_series_analysis" / "network")
    impact_root = _ensure_dir(run_root / "02_momentary_impact_coefficients")
    handoff_root = _ensure_dir(run_root / "03_treatment_target_handoff")
    intervention_root = _ensure_dir(run_root / "03b_translation_digital_intervention")
    visualization_root = _ensure_dir(run_root / "04_impact_visualizations")
    reporting_root = _ensure_dir(run_root / "05_research_reports")
    readiness_source_root = (
        Path(args.readiness_source_root).expanduser().resolve()
        if str(args.readiness_source_root).strip()
        else readiness_root
    )
    network_source_root = (
        Path(args.network_source_root).expanduser().resolve()
        if str(args.network_source_root).strip()
        else network_root
    )
    impact_source_root = (
        Path(args.impact_source_root).expanduser().resolve()
        if str(args.impact_source_root).strip()
        else impact_root
    )
    readiness_runtime_root = readiness_root if not args.skip_readiness else readiness_source_root
    network_runtime_root = network_root if not args.skip_network else network_source_root
    impact_runtime_root = impact_root if not args.skip_impact else impact_source_root

    logger = PipelineLogger(logs_root / "pipeline.jsonl", run_id=run_id)
    logger.log(
        "INFO",
        "pipeline.start",
        "Pipeline started.",
        stage="pipeline",
        metrics={"cycle_index": cycle_index, "cycles": cycles, "iterative_memory": bool(args.enable_iterative_memory)},
        artifacts={"repo_root": str(repo_root), "run_root": str(run_root), "history_root": str(history_root)},
    )
    contract_validator = _load_contract_validator(repo_root=repo_root)

    if not pseudodata_root.exists():
        logger.log("ERROR", "pipeline.input_missing", "Pseudodata root not found.", pseudodata_root=str(pseudodata_root))
        return 2

    profiles = discover_profiles(
        pseudodata_root=pseudodata_root,
        filename=args.data_filename,
        pattern=args.pattern.strip(),
        max_profiles=int(args.max_profiles),
    )
    if not profiles:
        logger.log(
            "ERROR",
            "pipeline.no_profiles",
            "No pseudodata profiles found for the provided pattern/filename.",
            pseudodata_root=str(pseudodata_root),
            filename=args.data_filename,
            pattern=args.pattern,
        )
        return 3

    logger.log("INFO", "pipeline.profiles", "Profiles discovered.", profile_count=len(profiles), profiles=profiles[:25])
    logger.log(
        "INFO",
        "pipeline.stage_roots",
        "Resolved stage roots.",
        readiness_runtime_root=str(readiness_runtime_root),
        network_runtime_root=str(network_runtime_root),
        impact_runtime_root=str(impact_runtime_root),
    )

    subset_root = materialize_profile_subset_root(
        pseudodata_root=pseudodata_root,
        selected_profiles=profiles,
        subset_root=run_root / "_input_subset",
        logger=logger,
    )

    readiness_script = (
        repo_root
        / "src/SystemComponents/Hierarchical_Updating_Algorithm/01_time_series_analysis/01_check_readiness/apply_readiness_check.py"
    )
    network_script = (
        repo_root
        / "src/SystemComponents/Hierarchical_Updating_Algorithm/01_time_series_analysis/02_network_time_series_analysis/01_run_network_ts_analysis.py"
    )
    impact_script = (
        repo_root
        / "src/SystemComponents/Hierarchical_Updating_Algorithm/02_hierarchical_update_ranking/01_momentary_impact_quantification/01_compute_momentary_impact_coefficients.py"
    )
    handoff_script = (
        repo_root
        / "src/SystemComponents/Agentic_Framework/03_TreatmentTargetIdentification/01_prepare_targets_from_impact.py"
    )
    visualization_script = (
        repo_root
        / "src/SystemComponents/Hierarchical_Updating_Algorithm/02_hierarchical_update_ranking/01_momentary_impact_quantification/02_visualize_impact_coefficients.py"
    )
    intervention_script = (
        repo_root
        / "src/SystemComponents/Agentic_Framework/05_TranslationDigitalIntervention/01_generate_hapa_digital_intervention.py"
    )
    report_script = repo_root / "Evaluation/07_research_communication/generate_pipeline_research_report.py"

    for script_path in [readiness_script, network_script, impact_script, handoff_script, intervention_script, visualization_script]:
        if not script_path.exists():
            logger.log("ERROR", "pipeline.script_missing", "Required script not found.", script=str(script_path))
            return 4
    if bool(args.run_component_report) and not report_script.exists():
        logger.log("ERROR", "pipeline.script_missing", "Research report script not found.", script=str(report_script))
        return 4

    stage_results: List[StageResult] = []
    impact_profiles: List[str] = []
    stage_output_roots: Dict[str, Path] = {
        "readiness": readiness_root,
        "network": network_root,
        "impact": impact_root,
        "handoff": handoff_root,
        "intervention": intervention_root,
        "visualization": visualization_root,
        "reporting": reporting_root,
    }

    try:
        if not args.skip_readiness:
            cmd_readiness = [
                args.python_exe,
                str(readiness_script),
                "--input-root",
                str(subset_root),
                "--output-root",
                str(readiness_root),
                "--filename",
                args.data_filename,
                "--lag",
                str(args.lag),
                "--max-profiles",
                str(args.max_profiles),
                "--llm-finalize",
                _bool_str(bool(args.llm_finalize)),
                "--prefer-time-varying",
                _bool_str(bool(args.prefer_time_varying)),
                "--quiet",
                _bool_str(bool(args.readiness_quiet)),
            ]
            if args.dry_run:
                logger.log("INFO", "readiness.dry_run", "Skipping execution (dry-run).", command=" ".join(cmd_readiness))
            else:
                stage_results.append(
                    run_stage(
                        "readiness",
                        cmd_readiness,
                        stage_output_roots["readiness"] / "stage.log",
                        logger,
                        run_id=run_id,
                        cycle_index=cycle_index,
                        profile_count=len(profiles),
                        stage_events_path=stage_output_roots["readiness"] / "stage_events.jsonl",
                        stage_trace_path=stage_output_roots["readiness"] / "stage_trace.json",
                        env_overrides=runtime_env,
                    )
                )
                validate_stage_outputs(
                    stage_name="readiness",
                    root=readiness_root,
                    profiles=profiles,
                    required_relpaths=["readiness_report.json", "readiness_summary.txt"],
                    logger=logger,
                )
                readiness_contracts = validate_contract_files(
                    validator=contract_validator,
                    contract_name="readiness_report",
                    root=readiness_root,
                    profiles=profiles,
                    relpath="readiness_report.json",
                    logger=logger,
                    stage="readiness",
                )
                _write_stage_trace(
                    stage_output_roots["readiness"] / "stage_trace.json",
                    run_id=run_id,
                    cycle_index=cycle_index,
                    stage="readiness",
                    result=stage_results[-1],
                    profile_count=len(profiles),
                    contract_results=readiness_contracts,
                )
        elif not args.skip_network and not args.dry_run:
            validate_stage_outputs(
                stage_name="readiness",
                root=readiness_runtime_root,
                profiles=profiles,
                required_relpaths=["readiness_report.json", "readiness_summary.txt"],
                logger=logger,
            )

        if not args.skip_network:
            cmd_network = [
                args.python_exe,
                str(network_script),
                "--input-root",
                str(subset_root),
                "--readiness-root",
                str(readiness_runtime_root),
                "--output-root",
                str(network_root),
                "--data-filename",
                args.data_filename,
                "--metadata-filename",
                "variables_metadata.csv",
                "--readiness-filename",
                "readiness_report.json",
                "--pattern",
                args.pattern,
                "--max-profiles",
                str(args.max_profiles),
                "--prefer-tier",
                args.network_prefer_tier,
                "--execution-policy",
                args.network_execution_policy,
                "--boot",
                str(args.network_boot),
                "--block-len",
                str(args.network_block_len),
                "--jobs",
                str(args.network_jobs),
            ]
            if not bool(args.network_verbose):
                cmd_network.append("--no-verbose")
            if args.dry_run:
                logger.log("INFO", "network.dry_run", "Skipping execution (dry-run).", command=" ".join(cmd_network))
            else:
                stage_results.append(
                    run_stage(
                        "network",
                        cmd_network,
                        stage_output_roots["network"] / "stage.log",
                        logger,
                        run_id=run_id,
                        cycle_index=cycle_index,
                        profile_count=len(profiles),
                        stage_events_path=stage_output_roots["network"] / "stage_events.jsonl",
                        stage_trace_path=stage_output_roots["network"] / "stage_trace.json",
                        env_overrides=runtime_env,
                    )
                )
                validate_stage_outputs(
                    stage_name="network",
                    root=network_root,
                    profiles=profiles,
                    required_relpaths=["comparison_summary.json", "network_metrics/predictor_importance_tv.csv"],
                    logger=logger,
                )
                network_contracts = validate_contract_files(
                    validator=contract_validator,
                    contract_name="network_comparison_summary",
                    root=network_root,
                    profiles=profiles,
                    relpath="comparison_summary.json",
                    logger=logger,
                    stage="network",
                )
                _write_stage_trace(
                    stage_output_roots["network"] / "stage_trace.json",
                    run_id=run_id,
                    cycle_index=cycle_index,
                    stage="network",
                    result=stage_results[-1],
                    profile_count=len(profiles),
                    contract_results=network_contracts,
                )
        elif not args.skip_impact and not args.dry_run:
            validate_stage_outputs(
                stage_name="network",
                root=network_runtime_root,
                profiles=profiles,
                required_relpaths=["comparison_summary.json", "network_metrics/predictor_importance_tv.csv"],
                logger=logger,
            )

        if not args.skip_impact:
            cmd_impact = [
                args.python_exe,
                str(impact_script),
                "--input-root",
                str(network_runtime_root),
                "--output-root",
                str(impact_root),
                "--pattern",
                args.pattern,
                "--max-profiles",
                str(args.max_profiles),
                "--half-life",
                str(args.impact_half_life),
                "--top-k-edges",
                str(args.impact_top_k_edges),
            ]
            if args.dry_run:
                logger.log("INFO", "impact.dry_run", "Skipping execution (dry-run).", command=" ".join(cmd_impact))
            else:
                stage_results.append(
                    run_stage(
                        "impact",
                        cmd_impact,
                        stage_output_roots["impact"] / "stage.log",
                        logger,
                        run_id=run_id,
                        cycle_index=cycle_index,
                        profile_count=len(profiles),
                        stage_events_path=stage_output_roots["impact"] / "stage_events.jsonl",
                        stage_trace_path=stage_output_roots["impact"] / "stage_trace.json",
                        env_overrides=runtime_env,
                    )
                )
                impact_profiles = discover_output_profiles(
                    stage_root=impact_root,
                    required_filename="predictor_composite.csv",
                    pattern=args.pattern,
                    max_profiles=int(args.max_profiles),
                )
                if impact_profiles:
                    validate_stage_outputs(
                        stage_name="impact",
                        root=impact_root,
                        profiles=impact_profiles,
                        required_relpaths=["overall_predictor_impact.json", "momentary_impact.json", "predictor_composite.csv"],
                        logger=logger,
                    )
                    impact_contracts = validate_contract_files(
                        validator=contract_validator,
                        contract_name="momentary_impact",
                        root=impact_root,
                        profiles=impact_profiles,
                        relpath="momentary_impact.json",
                        logger=logger,
                        stage="impact",
                    )
                    _write_stage_trace(
                        stage_output_roots["impact"] / "stage_trace.json",
                        run_id=run_id,
                        cycle_index=cycle_index,
                        stage="impact",
                        result=stage_results[-1],
                        profile_count=len(impact_profiles),
                        contract_results=impact_contracts,
                    )
                else:
                    logger.log(
                        "WARNING",
                        "impact.no_profiles",
                        "No impact outputs generated. This can happen when readiness does not permit lagged analysis.",
                    )
        elif not args.skip_handoff and not args.dry_run:
            impact_profiles = discover_output_profiles(
                stage_root=impact_runtime_root,
                required_filename="predictor_composite.csv",
                pattern=args.pattern,
                max_profiles=int(args.max_profiles),
            )
            if impact_profiles:
                validate_stage_outputs(
                    stage_name="impact",
                    root=impact_runtime_root,
                    profiles=impact_profiles,
                    required_relpaths=["overall_predictor_impact.json", "momentary_impact.json", "predictor_composite.csv"],
                    logger=logger,
                )
            else:
                logger.log(
                    "WARNING",
                    "impact.no_profiles",
                    "No impact outputs found in source impact root. Handoff stage will be skipped.",
                )

        if not impact_profiles and not args.skip_handoff:
            logger.log("WARNING", "handoff.skipped", "Skipping handoff because no impact profiles are available.")
        elif not args.skip_handoff:
            initial_model_runs_root = (
                repo_root / "Evaluation/03_construction_initial_observation_model/constructed_PC_models/runs"
            )
            free_text_root = repo_root / "Evaluation/01_pseudoprofile(s)/free_text"
            predictor_feasibility_csv = (
                Path(str(args.predictor_feasibility_csv)).expanduser().resolve()
                if str(args.predictor_feasibility_csv).strip()
                else (
                    repo_root
                    / "src/utils/official/multi_dimensional_feasibility_evaluation/PREDICTORS/results/summary/predictor_rankings.csv"
                ).resolve()
            )
            predictor_list_path = (
                repo_root
                / "src/utils/official/ontology_mappings/CRITERION/predictor_to_criterion/input_lists/predictors_list.txt"
            )
            cmd_handoff = [
                args.python_exe,
                str(handoff_script),
                "--impact-root",
                str(impact_runtime_root),
                "--output-root",
                str(handoff_root),
                "--readiness-root",
                str(readiness_runtime_root),
                "--network-root",
                str(network_runtime_root),
                "--initial-model-runs-root",
                str(initial_model_runs_root),
                "--free-text-root",
                str(free_text_root),
                "--predictor-list-path",
                str(predictor_list_path),
                "--predictor-feasibility-csv",
                str(predictor_feasibility_csv),
                "--pattern",
                args.pattern,
                "--max-profiles",
                str(args.max_profiles),
                "--top-k",
                str(args.handoff_top_k),
                "--min-impact",
                str(args.handoff_min_impact),
                "--max-candidate-predictors",
                str(args.handoff_max_candidate_predictors),
                "--llm-model",
                str(args.handoff_llm_model),
                "--prompt-budget-tokens",
                str(args.handoff_prompt_budget_tokens),
                "--critic-max-iterations",
                str(args.handoff_critic_max_iterations),
                "--critic-pass-threshold",
                str(args.handoff_critic_pass_threshold),
                "--preferred-predictor-count",
                str(args.handoff_preferred_predictor_count),
                "--preferred-criterion-count",
                str(args.handoff_preferred_criterion_count),
                "--parent-feasibility-top-k",
                str(args.parent_feasibility_top_k),
                "--contract-version",
                "1.0.0",
                "--trace-output",
                str(handoff_root / "stage_trace_component.json"),
            ]
            if bool(args.hard_ontology_constraint):
                cmd_handoff.append("--hard-ontology-constraint")
            if bool(args.enable_iterative_memory):
                cmd_handoff.extend(
                    [
                        "--history-snapshots-path",
                        str(history_root / "feature_snapshots.parquet"),
                        "--profile-memory-window",
                        str(max(1, int(args.profile_memory_window))),
                    ]
                )
            if bool(args.handoff_disable_llm):
                cmd_handoff.append("--disable-llm")
            if args.dry_run:
                logger.log("INFO", "handoff.dry_run", "Skipping execution (dry-run).", command=" ".join(cmd_handoff))
            else:
                stage_results.append(
                    run_stage(
                        "handoff",
                        cmd_handoff,
                        stage_output_roots["handoff"] / "stage.log",
                        logger,
                        run_id=run_id,
                        cycle_index=cycle_index,
                        profile_count=len(impact_profiles),
                        stage_events_path=stage_output_roots["handoff"] / "stage_events.jsonl",
                        stage_trace_path=stage_output_roots["handoff"] / "stage_trace.json",
                        env_overrides=runtime_env,
                    )
                )
                validate_stage_outputs(
                    stage_name="handoff",
                    root=handoff_root,
                    profiles=impact_profiles,
                    required_relpaths=[
                        "top_treatment_target_candidates.csv",
                        "step04_updated_observation_model.json",
                        "step04_nomothetic_idiographic_fusion.json",
                        "step03_guardrail_review.json",
                        "step04_guardrail_review.json",
                        "visuals/updated_model_fused_heatmap.png",
                    ],
                    logger=logger,
                )
                step03_contracts = validate_contract_files(
                    validator=contract_validator,
                    contract_name="step03_target_selection",
                    root=handoff_root,
                    profiles=impact_profiles,
                    relpath="step03_target_selection.json",
                    logger=logger,
                    stage="handoff",
                )
                step04_contracts = validate_contract_files(
                    validator=contract_validator,
                    contract_name="step04_updated_model",
                    root=handoff_root,
                    profiles=impact_profiles,
                    relpath="step04_updated_observation_model.json",
                    logger=logger,
                    stage="handoff",
                )
                _write_stage_trace(
                    stage_output_roots["handoff"] / "stage_trace.json",
                    run_id=run_id,
                    cycle_index=cycle_index,
                    stage="handoff",
                    result=stage_results[-1],
                    profile_count=len(impact_profiles),
                    contract_results=[*step03_contracts, *step04_contracts],
                )

        if bool(args.run_intervention_step) and not bool(args.skip_intervention):
            if args.skip_handoff:
                logger.log(
                    "WARNING",
                    "intervention.skipped",
                    "Skipping Step-05 intervention because --skip-handoff is enabled.",
                )
            elif not impact_profiles:
                logger.log(
                    "WARNING",
                    "intervention.skipped",
                    "Skipping Step-05 intervention because no impact/handoff profiles are available.",
                )
            else:
                cmd_intervention = [
                    args.python_exe,
                    str(intervention_script),
                    "--handoff-root",
                    str(handoff_root),
                    "--output-root",
                    str(intervention_root),
                    "--readiness-root",
                    str(readiness_runtime_root),
                    "--network-root",
                    str(network_runtime_root),
                    "--impact-root",
                    str(impact_runtime_root),
                    "--predictor-feasibility-csv",
                    str(predictor_feasibility_csv),
                    "--pattern",
                    args.pattern,
                    "--max-profiles",
                    str(args.max_profiles),
                    "--llm-model",
                    str(args.intervention_llm_model),
                    "--prompt-budget-tokens",
                    str(args.intervention_prompt_budget_tokens),
                    "--critic-max-iterations",
                    str(args.intervention_critic_max_iterations),
                    "--critic-pass-threshold",
                    str(args.intervention_critic_pass_threshold),
                    "--parent-feasibility-top-k",
                    str(args.parent_feasibility_top_k),
                    "--contract-version",
                    "1.0.0",
                    "--trace-output",
                    str(intervention_root / "stage_trace_component.json"),
                ]
                if bool(args.hard_ontology_constraint):
                    cmd_intervention.append("--hard-ontology-constraint")
                if bool(args.intervention_disable_llm):
                    cmd_intervention.append("--disable-llm")
                if args.dry_run:
                    logger.log(
                        "INFO",
                        "intervention.dry_run",
                        "Skipping execution (dry-run).",
                        command=" ".join(cmd_intervention),
                    )
                else:
                    stage_results.append(
                        run_stage(
                            "intervention",
                            cmd_intervention,
                            stage_output_roots["intervention"] / "stage.log",
                            logger,
                            run_id=run_id,
                            cycle_index=cycle_index,
                            profile_count=len(impact_profiles),
                            stage_events_path=stage_output_roots["intervention"] / "stage_events.jsonl",
                            stage_trace_path=stage_output_roots["intervention"] / "stage_trace.json",
                            env_overrides=runtime_env,
                        )
                    )
                    validate_stage_outputs(
                        stage_name="intervention",
                        root=intervention_root,
                        profiles=impact_profiles,
                        required_relpaths=[
                            "step05_hapa_intervention.json",
                            "step05_guardrail_review.json",
                            "step05_selected_barriers_top10.csv",
                            "step05_coping_candidates_ranked.csv",
                        ],
                        logger=logger,
                    )
                    step05_contracts = validate_contract_files(
                        validator=contract_validator,
                        contract_name="step05_hapa_intervention",
                        root=intervention_root,
                        profiles=impact_profiles,
                        relpath="step05_hapa_intervention.json",
                        logger=logger,
                        stage="intervention",
                    )
                    _write_stage_trace(
                        stage_output_roots["intervention"] / "stage_trace.json",
                        run_id=run_id,
                        cycle_index=cycle_index,
                        stage="intervention",
                        result=stage_results[-1],
                        profile_count=len(impact_profiles),
                        contract_results=step05_contracts,
                    )
        else:
            logger.log("INFO", "intervention.disabled", "Step-05 intervention stage disabled.")

        if bool(args.run_impact_visualizations):
            if not impact_profiles:
                logger.log(
                    "WARNING",
                    "visualization.skipped",
                    "Skipping visualizations because no impact profiles are available.",
                )
            else:
                cmd_visualization = [
                    args.python_exe,
                    str(visualization_script),
                    "--root",
                    str(impact_runtime_root),
                    "--pattern",
                    args.pattern,
                    "--matrix-name",
                    "impact_matrix.csv",
                    "--dpi",
                    str(args.visualization_dpi),
                    "--top-k-edges",
                    str(args.visualization_top_k_edges),
                    "--top-n-heatmap",
                    str(args.visualization_top_n_heatmap),
                    "--top-n-bars",
                    str(args.visualization_top_n_bars),
                    "--max-profiles",
                    str(args.max_profiles),
                ]
                if args.dry_run:
                    logger.log(
                        "INFO",
                        "visualization.dry_run",
                        "Skipping execution (dry-run).",
                        command=" ".join(cmd_visualization),
                    )
                else:
                    stage_results.append(
                        run_stage(
                            "visualization",
                            cmd_visualization,
                            stage_output_roots["visualization"] / "stage.log",
                            logger,
                            run_id=run_id,
                            cycle_index=cycle_index,
                            profile_count=len(impact_profiles),
                            stage_events_path=stage_output_roots["visualization"] / "stage_events.jsonl",
                            stage_trace_path=stage_output_roots["visualization"] / "stage_trace.json",
                            env_overrides=runtime_env,
                        )
                    )
                    viz_summary_src = impact_runtime_root / "visualization_run_summary.json"
                    if viz_summary_src.exists():
                        shutil.copy2(viz_summary_src, visualization_root / "visualization_run_summary.json")
                        for profile_id in impact_profiles:
                            if (impact_runtime_root / profile_id / "visuals").exists():
                                _ensure_dir(visualization_root / "profile_visuals_index").joinpath(f"{profile_id}.txt").write_text(
                                    str((impact_runtime_root / profile_id / "visuals").resolve()),
                                    encoding="utf-8",
                                )
                    else:
                        logger.log(
                            "WARNING",
                            "visualization.summary_missing",
                            "Visualization summary file was not found after stage execution.",
                            expected=str(viz_summary_src),
                        )
        else:
            logger.log("INFO", "visualization.disabled", "Visualization stage disabled by --no-run-impact-visualizations.")

        if bool(args.enable_iterative_memory) and impact_profiles:
            append_history_ledger(
                history_root=history_root,
                run_id=run_id,
                cycle_index=cycle_index,
                profile_ids=impact_profiles,
                step04_root=handoff_root,
                step05_root=intervention_root,
                impact_root=impact_runtime_root,
                logger=logger,
            )
            local_history_root = _ensure_dir(run_root / "history")
            if local_history_root.resolve() != history_root.resolve():
                append_history_ledger(
                    history_root=local_history_root,
                    run_id=run_id,
                    cycle_index=cycle_index,
                    profile_ids=impact_profiles,
                    step04_root=handoff_root,
                    step05_root=intervention_root,
                    impact_root=impact_runtime_root,
                    logger=logger,
                )

        interim_summary = {
            "status": "running",
            "contract_version": "1.0.0",
            "run_id": run_id,
            "cycle_index": cycle_index,
            "cycles": cycles,
            "hard_ontology_constraint": bool(args.hard_ontology_constraint),
            "critic_policy": {
                "handoff_max_iterations": int(args.handoff_critic_max_iterations),
                "handoff_pass_threshold": float(args.handoff_critic_pass_threshold),
                "intervention_max_iterations": int(args.intervention_critic_max_iterations),
                "intervention_pass_threshold": float(args.intervention_critic_pass_threshold),
            },
            "run_root": str(run_root),
            "repo_root": str(repo_root),
            "pseudodata_root": str(pseudodata_root),
            "iterative_memory": {
                "enabled": bool(args.enable_iterative_memory),
                "memory_policy": str(args.memory_policy),
                "resume_from_run": str(args.resume_from_run).strip(),
                "history_root": str(history_root),
                "run_history_root": str(run_root / "history"),
                "profile_memory_window": int(max(1, int(args.profile_memory_window))),
            },
            "profiles": profiles,
            "impact_profiles": impact_profiles,
            "outputs": {
                "readiness_root": str(readiness_root),
                "network_root": str(network_root),
                "impact_root": str(impact_root),
                "handoff_root": str(handoff_root),
                "intervention_root": str(intervention_root),
                "visualization_root": str(visualization_root),
                "reporting_root": str(reporting_root),
                "readiness_runtime_root": str(readiness_runtime_root),
                "network_runtime_root": str(network_runtime_root),
                "impact_runtime_root": str(impact_runtime_root),
                "logs_root": str(logs_root),
            },
            "stage_results": [asdict(result) for result in stage_results],
        }
        (run_root / "pipeline_summary.json").write_text(json.dumps(interim_summary, indent=2), encoding="utf-8")

        if bool(args.run_component_report):
            cmd_report = [
                args.python_exe,
                str(report_script),
                "--run-root",
                str(run_root),
                "--output-root",
                str(reporting_root),
                "--max-profiles",
                str(args.max_profiles),
            ]
            if args.dry_run:
                logger.log("INFO", "reporting.dry_run", "Skipping execution (dry-run).", command=" ".join(cmd_report))
            else:
                stage_results.append(
                    run_stage(
                        "reporting",
                        cmd_report,
                        stage_output_roots["reporting"] / "stage.log",
                        logger,
                        run_id=run_id,
                        cycle_index=cycle_index,
                        profile_count=len(profiles),
                        stage_events_path=stage_output_roots["reporting"] / "stage_events.jsonl",
                        stage_trace_path=stage_output_roots["reporting"] / "stage_trace.json",
                        env_overrides=runtime_env,
                    )
                )
                validate_root_outputs(
                    stage_name="reporting",
                    root=reporting_root,
                    required_relpaths=["run_report.md", "run_report.json"],
                    logger=logger,
                )
        else:
            logger.log("INFO", "reporting.disabled", "Research report generation disabled by --no-run-component-report.")

    except Exception as exc:
        logger.log("ERROR", "pipeline.failed", "Pipeline failed.", error=repr(exc))
        summary = {
            "status": "failed",
            "contract_version": "1.0.0",
            "error": repr(exc),
            "run_id": run_id,
            "cycle_index": cycle_index,
            "run_root": str(run_root),
            "profiles": profiles,
            "stage_results": [asdict(result) for result in stage_results],
        }
        (run_root / "pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return 5

    summary_ok = {
        "status": "ok",
        "contract_version": "1.0.0",
        "run_id": run_id,
        "cycle_index": cycle_index,
        "cycles": cycles,
        "hard_ontology_constraint": bool(args.hard_ontology_constraint),
        "critic_policy": {
            "handoff_max_iterations": int(args.handoff_critic_max_iterations),
            "handoff_pass_threshold": float(args.handoff_critic_pass_threshold),
            "intervention_max_iterations": int(args.intervention_critic_max_iterations),
            "intervention_pass_threshold": float(args.intervention_critic_pass_threshold),
        },
        "run_root": str(run_root),
        "repo_root": str(repo_root),
        "pseudodata_root": str(pseudodata_root),
        "iterative_memory": {
            "enabled": bool(args.enable_iterative_memory),
            "memory_policy": str(args.memory_policy),
            "resume_from_run": str(args.resume_from_run).strip(),
            "history_root": str(history_root),
            "run_history_root": str(run_root / "history"),
            "profile_memory_window": int(max(1, int(args.profile_memory_window))),
        },
        "profiles": profiles,
        "impact_profiles": impact_profiles,
        "outputs": {
            "readiness_root": str(readiness_root),
            "network_root": str(network_root),
            "impact_root": str(impact_root),
            "handoff_root": str(handoff_root),
            "intervention_root": str(intervention_root),
            "visualization_root": str(visualization_root),
            "reporting_root": str(reporting_root),
            "readiness_runtime_root": str(readiness_runtime_root),
            "network_runtime_root": str(network_runtime_root),
            "impact_runtime_root": str(impact_runtime_root),
            "logs_root": str(logs_root),
        },
        "stage_results": [asdict(result) for result in stage_results],
    }
    (run_root / "pipeline_summary.json").write_text(json.dumps(summary_ok, indent=2), encoding="utf-8")
    pipeline_contract = contract_validator.validate_payload(
        contract_name="pipeline_summary",
        payload=summary_ok,
        path=str(run_root / "pipeline_summary.json"),
    )
    if not pipeline_contract.success:
        logger.log(
            "ERROR",
            "pipeline.contract_failed",
            "Pipeline summary contract validation failed.",
            stage="pipeline",
            exception="; ".join(pipeline_contract.errors or [])[:1000],
        )
        return 6
    logger.log("INFO", "pipeline.done", "Pipeline completed successfully.", summary_path=str(run_root / "pipeline_summary.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
