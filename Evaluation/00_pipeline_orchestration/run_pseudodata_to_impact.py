#!/usr/bin/env python3
"""
run_pseudodata_to_impact.py

End-to-end integration runner for:
1) Readiness check
2) Network time-series analysis (tv-gVAR + baselines)
3) Momentary impact coefficient quantification
4) Treatment-target handoff preparation (Agentic step 03 input)
5) Impact visualization generation
6) Research communication report generation

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
from typing import Dict, List, Optional, Sequence


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
    def __init__(self, jsonl_path: Path) -> None:
        self.jsonl_path = jsonl_path
        _ensure_dir(self.jsonl_path.parent)

    def log(self, level: str, event: str, message: str, **fields: object) -> None:
        payload: Dict[str, object] = {
            "timestamp_local": _ts(),
            "level": level.upper(),
            "event": event,
            "message": message,
        }
        if fields:
            payload["fields"] = fields

        print(f"[{payload['timestamp_local']}] [{payload['level']}] {event}: {message}", flush=True)
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


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


def run_stage(stage: str, cmd: Sequence[str], stage_log_path: Path, logger: PipelineLogger) -> StageResult:
    start = time.time()
    logger.log("INFO", f"{stage}.start", "Running stage command.", command=" ".join(cmd))

    _ensure_dir(stage_log_path.parent)
    with stage_log_path.open("w", encoding="utf-8") as stage_log:
        process = subprocess.Popen(
            list(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
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
            return_code=return_code,
            duration_seconds=round(duration, 3),
            log_path=str(stage_log_path),
        )
        raise RuntimeError(f"Stage '{stage}' failed with exit code {return_code}.")

    logger.log(
        "INFO",
        f"{stage}.done",
        "Stage completed successfully.",
        duration_seconds=round(duration, 3),
        log_path=str(stage_log_path),
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

    parser.add_argument("--run-impact-visualizations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visualization-dpi", type=int, default=300)
    parser.add_argument("--visualization-top-k-edges", type=int, default=60)
    parser.add_argument("--visualization-top-n-heatmap", type=int, default=25)
    parser.add_argument("--visualization-top-n-bars", type=int, default=25)

    parser.add_argument("--run-component-report", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--skip-readiness", action="store_true")
    parser.add_argument("--skip-network", action="store_true")
    parser.add_argument("--skip-impact", action="store_true")
    parser.add_argument("--skip-handoff", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    repo_root = Path(args.repo_root).expanduser().resolve()
    pseudodata_root = Path(args.pseudodata_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    run_id = args.run_id.strip() or _run_id_now()
    run_root = _ensure_dir(output_root / run_id)
    logs_root = _ensure_dir(run_root / "logs")

    readiness_root = _ensure_dir(run_root / "00_readiness_check")
    network_root = _ensure_dir(run_root / "01_time_series_analysis" / "network")
    impact_root = _ensure_dir(run_root / "02_momentary_impact_coefficients")
    handoff_root = _ensure_dir(run_root / "03_treatment_target_handoff")
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

    logger = PipelineLogger(logs_root / "pipeline.jsonl")
    logger.log("INFO", "pipeline.start", "Pipeline started.", repo_root=str(repo_root), run_root=str(run_root))

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

    readiness_script = (
        repo_root
        / "SystemComponents/Hierarchical_Updating_Algorithm/01_time_series_analysis/01_check_readiness/apply_readiness_check.py"
    )
    network_script = (
        repo_root
        / "SystemComponents/Hierarchical_Updating_Algorithm/01_time_series_analysis/02_network_time_series_analysis/01_run_network_ts_analysis.py"
    )
    impact_script = (
        repo_root
        / "SystemComponents/Hierarchical_Updating_Algorithm/02_hierarchical_update_ranking/01_momentary_impact_quantification/01_compute_momentary_impact_coefficients.py"
    )
    handoff_script = repo_root / "SystemComponents/Agentic_Framework/03_TreatmentTargetIdentification/01_prepare_targets_from_impact.py"
    visualization_script = (
        repo_root
        / "SystemComponents/Hierarchical_Updating_Algorithm/02_hierarchical_update_ranking/01_momentary_impact_quantification/02_visualize_impact_coefficients.py"
    )
    report_script = repo_root / "Evaluation/07_research_communication/generate_pipeline_research_report.py"

    for script_path in [readiness_script, network_script, impact_script, handoff_script, visualization_script]:
        if not script_path.exists():
            logger.log("ERROR", "pipeline.script_missing", "Required script not found.", script=str(script_path))
            return 4
    if bool(args.run_component_report) and not report_script.exists():
        logger.log("ERROR", "pipeline.script_missing", "Research report script not found.", script=str(report_script))
        return 4

    stage_results: List[StageResult] = []
    impact_profiles: List[str] = []

    try:
        if not args.skip_readiness:
            cmd_readiness = [
                args.python_exe,
                str(readiness_script),
                "--input-root",
                str(pseudodata_root),
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
                stage_results.append(run_stage("readiness", cmd_readiness, logs_root / "readiness.log", logger))
                validate_stage_outputs(
                    stage_name="readiness",
                    root=readiness_root,
                    profiles=profiles,
                    required_relpaths=["readiness_report.json", "readiness_summary.txt"],
                    logger=logger,
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
                str(pseudodata_root),
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
                stage_results.append(run_stage("network", cmd_network, logs_root / "network.log", logger))
                validate_stage_outputs(
                    stage_name="network",
                    root=network_root,
                    profiles=profiles,
                    required_relpaths=["comparison_summary.json", "network_metrics/predictor_importance_tv.csv"],
                    logger=logger,
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
                stage_results.append(run_stage("impact", cmd_impact, logs_root / "impact.log", logger))
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
            cmd_handoff = [
                args.python_exe,
                str(handoff_script),
                "--impact-root",
                str(impact_runtime_root),
                "--output-root",
                str(handoff_root),
                "--pattern",
                args.pattern,
                "--max-profiles",
                str(args.max_profiles),
                "--top-k",
                str(args.handoff_top_k),
                "--min-impact",
                str(args.handoff_min_impact),
            ]
            if args.dry_run:
                logger.log("INFO", "handoff.dry_run", "Skipping execution (dry-run).", command=" ".join(cmd_handoff))
            else:
                stage_results.append(run_stage("handoff", cmd_handoff, logs_root / "handoff.log", logger))
                validate_stage_outputs(
                    stage_name="handoff",
                    root=handoff_root,
                    profiles=impact_profiles,
                    required_relpaths=["top_treatment_target_candidates.csv"],
                    logger=logger,
                )

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
                    stage_results.append(run_stage("visualization", cmd_visualization, logs_root / "visualization.log", logger))
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

        interim_summary = {
            "status": "running",
            "run_id": run_id,
            "run_root": str(run_root),
            "repo_root": str(repo_root),
            "pseudodata_root": str(pseudodata_root),
            "profiles": profiles,
            "impact_profiles": impact_profiles,
            "outputs": {
                "readiness_root": str(readiness_root),
                "network_root": str(network_root),
                "impact_root": str(impact_root),
                "handoff_root": str(handoff_root),
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
                stage_results.append(run_stage("reporting", cmd_report, logs_root / "reporting.log", logger))
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
            "error": repr(exc),
            "run_id": run_id,
            "run_root": str(run_root),
            "profiles": profiles,
            "stage_results": [asdict(result) for result in stage_results],
        }
        (run_root / "pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return 5

    summary_ok = {
        "status": "ok",
        "run_id": run_id,
        "run_root": str(run_root),
        "repo_root": str(repo_root),
        "pseudodata_root": str(pseudodata_root),
        "profiles": profiles,
        "impact_profiles": impact_profiles,
        "outputs": {
            "readiness_root": str(readiness_root),
            "network_root": str(network_root),
            "impact_root": str(impact_root),
            "handoff_root": str(handoff_root),
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
    logger.log("INFO", "pipeline.done", "Pipeline completed successfully.", summary_path=str(run_root / "pipeline_summary.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
