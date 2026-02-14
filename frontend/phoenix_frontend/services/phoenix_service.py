from __future__ import annotations

import json
import os
import queue
import shlex
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from .communication_agent import generate_communication_summary
from .pseudodata import build_collection_schema, parse_baseline_overrides, synthesize_pseudodata
from .session_store import SessionStore


LogFn = Callable[[str, str], None]


class PhoenixService:
    def __init__(self, *, repo_root: Path, python_exe: str, session_store: SessionStore) -> None:
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.python_exe = str(python_exe)
        self.session_store = session_store
        self.evaluation_root = self._resolve_evaluation_root()
        self.default_max_workers = max(1, int(os.getenv("PHOENIX_MAX_WORKERS", "12")))

        self.step01_script = (
            self.repo_root
            / "src/SystemComponents/Agentic_Framework/01_OperationalizationMentalHealthProblem/utils/02_operationalize_freetext_complaints.py"
        )
        self.step02_script = (
            self.repo_root
            / "src/SystemComponents/Agentic_Framework/02_ConstructionInitialObservationModel/utils/01_construct_observation_model.py"
        )
        self.step02_visual_script = (
            self.repo_root
            / "src/SystemComponents/Agentic_Framework/02_ConstructionInitialObservationModel/utils/helpers/bipartite_model_visualization.py"
        )
        self.integrated_pipeline_script = self.evaluation_root / "00_pipeline_orchestration/run_pseudodata_to_impact.py"
        self.predictor_feasibility_csv = (
            self.repo_root
            / "src/utils/official/multi_dimensional_feasibility_evaluation/PREDICTORS/results/summary/predictor_rankings.csv"
        )
        self.mapping_ranks_csv = (
            self.evaluation_root
            / "03_construction_initial_observation_model/helpers/00_LLM_based_mapping_based_predictor_ranks/all_pseudoprofiles__predictor_ranks_dense.csv"
        )
        self.ontology_predictor_list = (
            self.repo_root / "src/utils/official/ontology_mappings/CRITERION/predictor_to_criterion/input_lists/predictors_list.txt"
        )
        self.operationalization_cache_dir = (
            self.repo_root / "src/SystemComponents/Agentic_Framework/01_OperationalizationMentalHealthProblem/tmp"
        )

    def run_initial_model(
        self,
        *,
        session_id: str,
        llm_model: str = "gpt-5-nano",
        disable_llm: bool = False,
        hard_ontology_constraint: bool = False,
        prompt_budget_tokens: int = 400000,
        critic_max_iterations: int = 2,
        critic_pass_threshold: float = 0.74,
        max_workers: Optional[int] = None,
        log: LogFn,
    ) -> Dict[str, Any]:
        session = self.session_store.load_session(session_id)
        paths = self.session_store.session_paths(session_id)
        free_text_files = self.session_store.write_free_text_files(session)
        profile_id = session.profile_id
        effective_workers = max(1, int(max_workers if max_workers is not None else self.default_max_workers))

        run_id = f"frontend_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = paths["initial_model_runs_root"] / run_id
        profile_dir = run_dir / "profiles" / profile_id
        mapped_csv = paths["operationalization_root"] / "mapped_criterions.csv"
        mapped_csv.parent.mkdir(parents=True, exist_ok=True)
        if mapped_csv.exists():
            mapped_csv.unlink()

        if disable_llm:
            log("LLM disabled: constructing heuristic initial model fallback.", "WARNING")
            heuristic_model = self._build_heuristic_initial_model(profile_id=profile_id, complaint_text=session.complaint_text)
            profile_dir.mkdir(parents=True, exist_ok=True)
            model_final_json = profile_dir / "llm_observation_model_final.json"
            model_final_json.write_text(json.dumps(heuristic_model, ensure_ascii=False, indent=2), encoding="utf-8")
            model_mapped_json = profile_dir / "llm_observation_model_mapped.json"
            model_mapped_json.write_text(json.dumps(heuristic_model, ensure_ascii=False, indent=2), encoding="utf-8")
            mapped_csv.write_text("pseudoprofile_id,variable_id,variable_label,mapping_status\n", encoding="utf-8")
        else:
            self._assert_operationalization_cache()
            env_step01 = os.environ.copy()
            env_step01.setdefault("CRITERION_CACHE_DIR", str(self.operationalization_cache_dir))
            log("Running Step 01 operationalization from session-specific free-text input.", "INFO")
            self._run_command(
                cmd=[
                    self.python_exe,
                    str(self.step01_script),
                    "--input-txt",
                    str(free_text_files["free_text_complaints"]),
                    "--output-csv",
                    str(mapped_csv),
                    "--max-workers",
                    str(effective_workers),
                    "--limit",
                    "1",
                ],
                log=log,
                env=env_step01,
                component="step01_operationalization",
            )

            hyde_dense_profiles = self._discover_latest_hyde_dense_profiles()
            step02_cmd = [
                self.python_exe,
                str(self.step02_script),
                "--mapped_criterions_path",
                str(mapped_csv),
                "--hyde_dense_profiles_path",
                str(hyde_dense_profiles),
                "--llm_mapping_ranks_path",
                str(self.mapping_ranks_csv),
                "--ontology_path",
                str(self.ontology_predictor_list),
                "--predictor_feasibility_csv",
                str(self.predictor_feasibility_csv),
                "--results_dir",
                str(paths["initial_model_root"]),
                "--run_id",
                run_id,
                "--pseudoprofile_id",
                profile_id,
                "--llm_model",
                llm_model,
                "--prompt_budget_tokens",
                str(int(prompt_budget_tokens)),
                "--critic_max_iterations",
                str(int(critic_max_iterations)),
                "--critic_pass_threshold",
                str(float(critic_pass_threshold)),
                "--max_workers",
                str(effective_workers),
                "--no-enable_sampling",
            ]
            if hard_ontology_constraint:
                step02_cmd.append("--hard_ontology_constraint")

            log("Running Step 02 initial observation-model construction.", "INFO")
            self._run_command(
                cmd=step02_cmd,
                log=log,
                component="step02_initial_model",
                mark_success=False,
            )

        model_final_json = profile_dir / "llm_observation_model_final.json"
        if not model_final_json.exists():
            worker_error = self._extract_step02_worker_error(run_dir=run_dir, profile_id=profile_id)
            if worker_error:
                log(f"[component:step02_initial_model] status=failed {worker_error}", "ERROR")
                raise RuntimeError(
                    "Step 02 produced no model artifact. "
                    f"Worker error for {profile_id}: {worker_error}. "
                    f"Expected file: {model_final_json}"
                )
            log("[component:step02_initial_model] status=failed missing_model_json", "ERROR")
            raise RuntimeError(f"Step 02 finished but model JSON is missing: {model_final_json}")
        log("[component:step02_initial_model] status=succeeded", "INFO")

        model_mapped_json = profile_dir / "llm_observation_model_mapped.json"
        if not model_mapped_json.exists():
            model_mapped_json.write_text(model_final_json.read_text(encoding="utf-8"), encoding="utf-8")

        model_payload = json.loads(model_final_json.read_text(encoding="utf-8"))
        schema = build_collection_schema(model_payload)

        visuals: List[Dict[str, str]] = []
        communication: Dict[str, Any] = {}

        def _run_visualization_task() -> None:
            if self.step02_visual_script.exists():
                log("Generating Step 02 visualizations.", "INFO")
                try:
                    self._run_command(
                        cmd=[
                            self.python_exe,
                            str(self.step02_visual_script),
                            "--run_dir",
                            str(run_dir),
                            "--annotate_numeric_scores",
                        ],
                        log=log,
                        component="step02_visualization",
                    )
                except Exception as exc:
                    log(f"Visualization helper failed; continuing with model artifacts only ({exc}).", "WARNING")
            else:
                log("Step 02 visualization helper script not found; skipping visual generation.", "WARNING")

        def _run_communication_task() -> Dict[str, Any]:
            log("[component:communication_agent] status=running", "INFO")
            try:
                payload = self._write_communication_summary(
                    session_id=session_id,
                    stage="initial_model",
                    llm_model=llm_model,
                    disable_llm=disable_llm,
                    evidence={
                        "profile_id": profile_id,
                        "criteria_count": len(model_payload.get("criteria_variables", []) or []),
                        "predictor_count": len(model_payload.get("predictor_variables", []) or []),
                        "model_summary": str(model_payload.get("model_summary") or ""),
                        "collection_schema": schema,
                    },
                )
                log("[component:communication_agent] status=succeeded", "INFO")
                return payload
            except Exception as exc:
                log(f"[component:communication_agent] status=failed {exc}", "ERROR")
                raise

        with ThreadPoolExecutor(max_workers=2) as executor:
            vis_future = executor.submit(_run_visualization_task)
            comm_future = executor.submit(_run_communication_task)
            vis_future.result()
            communication = comm_future.result()

        visuals = self._collect_profile_visuals(profile_dir, session_id=session_id)

        self.session_store.update_session(
            session_id,
            initial_model_run_id=run_id,
            initial_model_run_dir=str(run_dir),
            latest_model_json=str(model_final_json),
            latest_model_mapped_json=str(model_mapped_json),
            notes={
                **session.notes,
                "initial_model_visual_count": len(visuals),
                "awaiting_fresh_acquisition": False,
            },
        )

        return {
            "profile_id": profile_id,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "mapped_criterions_csv": str(mapped_csv),
            "model_final_json": str(model_final_json),
            "model_mapped_json": str(model_mapped_json),
            "collection_schema": schema,
            "visuals": visuals,
            "communication_summary": communication,
        }

    def get_collection_schema(self, session_id: str) -> Dict[str, Any]:
        model_payload = self.load_model_payload(session_id)
        return build_collection_schema(model_payload)

    def synthesize_session_pseudodata(
        self,
        *,
        session_id: str,
        n_points: int,
        missing_rate: float,
        seed: int,
        baseline_rows: Optional[List[Dict[str, Any]]],
        log: LogFn,
    ) -> Dict[str, Any]:
        session = self.session_store.load_session(session_id)
        model_payload = self.load_model_payload(session_id)
        paths = self.session_store.session_paths(session_id)

        overrides = parse_baseline_overrides(baseline_rows or [])
        result = synthesize_pseudodata(
            model_payload=model_payload,
            profile_id=session.profile_id,
            output_profile_root=paths["pseudodata_profile_root"],
            n_points=int(n_points),
            missing_rate=float(missing_rate),
            seed=int(seed),
            baseline_overrides=overrides,
            log_fn=lambda line: log(line, "INFO"),
        )
        self.session_store.update_session(
            session_id,
            pseudodata_ready=True,
            pseudodata_root=str(paths["pseudodata_root"]),
            notes={
                **session.notes,
                "awaiting_fresh_acquisition": False,
            },
        )
        return result

    def save_manual_pseudodata(
        self,
        *,
        session_id: str,
        csv_text: str,
        log: LogFn,
    ) -> Dict[str, Any]:
        session = self.session_store.load_session(session_id)
        model_payload = self.load_model_payload(session_id)
        schema = build_collection_schema(model_payload)
        expected = [str(item["var_id"]) for item in schema["variables"]]
        paths = self.session_store.session_paths(session_id)

        frame = pd.read_csv(StringIO(csv_text))
        if frame.empty:
            raise RuntimeError("Uploaded manual pseudodata CSV is empty.")

        for col in expected:
            if col not in frame.columns:
                frame[col] = pd.NA
        if "t_index" not in frame.columns:
            frame.insert(0, "t_index", list(range(len(frame))))
        if "date" not in frame.columns:
            start = date.today() - timedelta(days=max(1, len(frame) - 1))
            frame.insert(1, "date", [(start + timedelta(days=idx)).isoformat() for idx in range(len(frame))])

        frame = frame[["t_index", "date", *expected]]
        profile_root = paths["pseudodata_profile_root"]
        profile_root.mkdir(parents=True, exist_ok=True)

        wide_path = profile_root / "pseudodata_wide.csv"
        long_path = profile_root / "pseudodata_long.csv"
        metadata_path = profile_root / "variables_metadata.csv"
        summary_path = profile_root / "generation_summary.json"
        spec_path = profile_root / "data_pattern_spec.txt"

        frame.to_csv(wide_path, index=False)
        long_df = frame.melt(id_vars=["t_index", "date"], var_name="variable", value_name="value")
        long_df.to_csv(long_path, index=False)
        metadata_df = pd.DataFrame(
            [
                {
                    "code": item["var_id"],
                    "role": item["role"],
                    "label": item["label"],
                    "ontology_id": item["ontology_path"],
                    "conf": item["default_baseline_0_1"],
                    "freq_hint": item["sampling_per_day"],
                }
                for item in schema["variables"]
            ]
        )
        metadata_df.to_csv(metadata_path, index=False)
        summary_payload = {
            "profile_id": session.profile_id,
            "mode": "manual",
            "n_points": int(len(frame)),
            "variable_count": len(expected),
            "missing_cells": int(frame[expected].isna().sum().sum()),
        }
        summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        spec_path.write_text(
            "manual_data_upload=true\n"
            f"profile_id={session.profile_id}\n"
            f"rows={len(frame)}\n"
            f"variables={len(expected)}\n",
            encoding="utf-8",
        )

        self.session_store.update_session(
            session_id,
            pseudodata_ready=True,
            pseudodata_root=str(paths["pseudodata_root"]),
            notes={
                **session.notes,
                "awaiting_fresh_acquisition": False,
            },
        )
        log("Saved manual pseudodata and metadata files.", "INFO")
        return {
            "profile_id": session.profile_id,
            "mode": "manual",
            "wide_csv": str(wide_path),
            "long_csv": str(long_path),
            "metadata_csv": str(metadata_path),
            "summary_json": str(summary_path),
            "spec_txt": str(spec_path),
        }

    def run_pipeline_cycle(
        self,
        *,
        session_id: str,
        hard_ontology_constraint: bool,
        llm_model: str,
        disable_llm: bool,
        include_intervention: bool,
        request_model_refinement: bool,
        profile_memory_window: int,
        handoff_critic_max_iterations: int,
        intervention_critic_max_iterations: int,
        log: LogFn,
    ) -> Dict[str, Any]:
        session = self.session_store.load_session(session_id)
        paths = self.session_store.session_paths(session_id)

        pseudodata_input = paths["pseudodata_profile_root"] / "pseudodata_wide.csv"
        if not pseudodata_input.exists():
            raise RuntimeError("No pseudodata found for this session. Generate pseudodata first.")
        if not session.latest_model_json:
            raise RuntimeError("No initial model found for this session. Run model construction first.")

        pipeline_run_id = session.pipeline_run_id.strip() or f"frontend_pipeline_{session.session_id}"
        cycle_index = int(session.current_cycle) + 1

        cmd = [
            self.python_exe,
            str(self.integrated_pipeline_script),
            "--repo-root",
            str(self.repo_root),
            "--pseudodata-root",
            str(paths["pseudodata_root"]),
            "--output-root",
            str(paths["pipeline_root"]),
            "--run-id",
            pipeline_run_id,
            "--pattern",
            session.profile_id,
            "--max-profiles",
            "1",
            "--data-filename",
            "pseudodata_wide.csv",
            "--enable-iterative-memory",
            "--cycle-index",
            str(cycle_index),
            "--cycles",
            str(cycle_index),
            "--memory-policy",
            "v1_weighted_fusion",
            "--history-root",
            str(paths["history_root"]),
            "--profile-memory-window",
            str(max(1, int(profile_memory_window))),
            "--initial-model-runs-root",
            str(paths["initial_model_runs_root"]),
            "--free-text-root",
            str(paths["free_text_root"]),
            "--predictor-feasibility-csv",
            str(self.predictor_feasibility_csv),
            "--handoff-llm-model",
            llm_model,
            "--intervention-llm-model",
            llm_model,
            "--handoff-critic-max-iterations",
            str(max(1, int(handoff_critic_max_iterations))),
            "--intervention-critic-max-iterations",
            str(max(1, int(intervention_critic_max_iterations))),
        ]
        cmd.append("--run-intervention-step" if include_intervention else "--no-run-intervention-step")

        if hard_ontology_constraint:
            cmd.append("--hard-ontology-constraint")
        if disable_llm:
            cmd.append("--disable-llm")

        log(f"Running integrated PHOENIX analysis cycle {cycle_index}.", "INFO")
        self._run_command(
            cmd=cmd,
            log=log,
            component="pipeline_cycle_orchestrator",
        )

        cycle_root = self._cycle_run_root(
            output_root=paths["pipeline_root"],
            run_id=pipeline_run_id,
            cycle_index=cycle_index,
        )
        summary_json = cycle_root / "pipeline_summary.json"
        if not summary_json.exists():
            raise RuntimeError(f"Integrated pipeline finished but no summary was produced: {summary_json}")
        payload = json.loads(summary_json.read_text(encoding="utf-8"))
        dashboard = self._extract_cycle_dashboard(
            cycle_root=cycle_root,
            profile_id=session.profile_id,
            session_id=session_id,
        )
        communication = self._write_communication_summary(
            session_id=session_id,
            stage=f"cycle_{cycle_index:02d}",
            llm_model=llm_model,
            disable_llm=disable_llm,
            evidence=dashboard,
        )

        self.session_store.update_session(
            session_id,
            current_cycle=cycle_index,
            pipeline_run_id=pipeline_run_id,
            latest_pipeline_cycle_root=str(cycle_root),
            latest_pipeline_summary=str(summary_json),
            notes={
                **session.notes,
                "latest_cycle_dashboard": dashboard,
                "latest_cycle_communication": communication,
                "awaiting_fresh_acquisition": bool(request_model_refinement),
                "cycle_requested_refinement": bool(request_model_refinement),
            },
        )
        return {
            "cycle_index": cycle_index,
            "run_id": pipeline_run_id,
            "cycle_root": str(cycle_root),
            "pipeline_summary_json": str(summary_json),
            "pipeline_summary": payload,
            "dashboard": dashboard,
            "communication_summary": communication,
        }

    def load_model_payload(self, session_id: str) -> Dict[str, Any]:
        session = self.session_store.load_session(session_id)
        model_path_raw = str(session.latest_model_json or "").strip()
        if not model_path_raw:
            raise RuntimeError("No initial observation model file found for this session.")
        model_path = Path(model_path_raw).expanduser().resolve()
        if not model_path.exists() or not model_path.is_file():
            raise RuntimeError("No initial observation model file found for this session.")
        return json.loads(model_path.read_text(encoding="utf-8"))

    def session_snapshot(self, session_id: str) -> Dict[str, Any]:
        session = self.session_store.load_session(session_id)
        paths = self.session_store.session_paths(session_id)

        collection_schema: Dict[str, Any] = {}
        model_summary: Dict[str, Any] = {}
        visuals: List[Dict[str, str]] = []
        model_path_raw = str(session.latest_model_json or "").strip()
        model_path = Path(model_path_raw).expanduser().resolve() if model_path_raw else None
        if model_path is not None and model_path.exists() and model_path.is_file():
            payload = json.loads(model_path.read_text(encoding="utf-8"))
            collection_schema = build_collection_schema(payload)
            model_summary = {
                "criteria_count": int(len(payload.get("criteria_variables", []) or [])),
                "predictor_count": int(len(payload.get("predictor_variables", []) or [])),
                "model_summary": str(payload.get("model_summary") or ""),
            }
            visuals = self._collect_profile_visuals(model_path.parent, session_id=session_id)

        pseudodata_summary: Dict[str, Any] = {}
        generation_summary_path = paths["pseudodata_profile_root"] / "generation_summary.json"
        if generation_summary_path.exists():
            pseudodata_summary = json.loads(generation_summary_path.read_text(encoding="utf-8"))

        pipeline_summary: Dict[str, Any] = {}
        pipeline_dashboard: Dict[str, Any] = {}
        pipeline_summary_raw = str(session.latest_pipeline_summary or "").strip()
        pipeline_summary_path = Path(pipeline_summary_raw).expanduser().resolve() if pipeline_summary_raw else None
        if pipeline_summary_path is not None and pipeline_summary_path.exists() and pipeline_summary_path.is_file():
            pipeline_summary = json.loads(pipeline_summary_path.read_text(encoding="utf-8"))
            cycle_root = Path(str(session.latest_pipeline_cycle_root or "")).expanduser().resolve()
            if cycle_root.exists():
                pipeline_dashboard = self._extract_cycle_dashboard(
                    cycle_root=cycle_root,
                    profile_id=session.profile_id,
                    session_id=session_id,
                )

        communication_summary = self._load_latest_communication_summary(session_id=session_id)

        return {
            "session": session.to_dict(),
            "paths": {
                key: str(value)
                for key, value in paths.items()
            },
            "model_summary": model_summary,
            "collection_schema": collection_schema,
            "visuals": visuals,
            "pseudodata_summary": pseudodata_summary,
            "pipeline_summary": pipeline_summary,
            "pipeline_dashboard": pipeline_dashboard,
            "communication_summary": communication_summary,
            "has_model": bool(model_path is not None and model_path.exists() and model_path.is_file()),
            "has_pseudodata": (paths["pseudodata_profile_root"] / "pseudodata_wide.csv").exists(),
            "has_pipeline_summary": bool(
                pipeline_summary_path is not None and pipeline_summary_path.exists() and pipeline_summary_path.is_file()
            ),
        }

    def _collect_profile_visuals(self, profile_dir: Path, *, session_id: str) -> List[Dict[str, str]]:
        visuals_root = profile_dir / "visuals"
        out: List[Dict[str, str]] = []
        if not visuals_root.exists():
            return out
        session_root = self.session_store.session_paths(session_id)["session_root"]
        for path in sorted(visuals_root.glob("*")):
            if path.suffix.lower() not in {".png", ".svg", ".pdf", ".gif"}:
                continue
            rel = str(path.resolve().relative_to(session_root.resolve()))
            out.append({"name": path.name, "relative_path": rel})
        return out

    def _run_command(
        self,
        *,
        cmd: List[str],
        log: LogFn,
        env: Optional[Dict[str, str]] = None,
        component: str = "",
        mark_success: bool = True,
    ) -> None:
        if not Path(cmd[1]).exists():
            raise RuntimeError(f"Required script not found: {cmd[1]}")
        log(f"$ {shlex.join(cmd)}", "DEBUG")
        process_env = os.environ.copy()
        if env:
            process_env.update(env)
        process_env.setdefault("PYTHONUNBUFFERED", "1")
        existing_pythonpath = process_env.get("PYTHONPATH", "")
        root_path = str(self.repo_root)
        path_parts = [part for part in existing_pythonpath.split(os.pathsep) if part]
        if root_path not in path_parts:
            process_env["PYTHONPATH"] = (
                f"{root_path}{os.pathsep}{existing_pythonpath}"
                if existing_pythonpath
                else root_path
            )
        if component:
            log(f"[component:{component}] status=running", "INFO")
        process = subprocess.Popen(
            cmd,
            cwd=str(self.repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=process_env,
        )
        assert process.stdout is not None
        line_queue: "queue.Queue[str]" = queue.Queue()

        def _reader() -> None:
            assert process.stdout is not None
            for raw_line in process.stdout:
                line_queue.put(raw_line.rstrip())
            line_queue.put("__PHOENIX_EOF__")

        reader_thread = threading.Thread(target=_reader, daemon=True)
        reader_thread.start()

        started = time.time()
        last_heartbeat = started
        rc: Optional[int] = None

        while True:
            try:
                line = line_queue.get(timeout=0.8)
            except queue.Empty:
                now = time.time()
                if component and now - last_heartbeat >= 10:
                    elapsed = int(now - started)
                    log(f"[component:{component}] heartbeat elapsed={elapsed}s", "INFO")
                    last_heartbeat = now
                rc = process.poll()
                if rc is not None:
                    break
                continue

            if line == "__PHOENIX_EOF__":
                rc = process.wait()
                break
            log(line, "INFO")

            now = time.time()
            if component and now - last_heartbeat >= 10:
                elapsed = int(now - started)
                log(f"[component:{component}] heartbeat elapsed={elapsed}s", "INFO")
                last_heartbeat = now

        if rc is None:
            rc = process.wait()
        reader_thread.join(timeout=1)
        if rc != 0:
            if component:
                log(f"[component:{component}] status=failed exit_code={rc}", "ERROR")
            raise RuntimeError(f"Command failed with exit code {rc}: {shlex.join(cmd)}")
        if component and mark_success:
            log(f"[component:{component}] status=succeeded", "INFO")

    def _extract_step02_worker_error(self, *, run_dir: Path, profile_id: str) -> str:
        errors_path = run_dir / "errors.csv"
        if not errors_path.exists():
            return ""
        try:
            errors_df = self._safe_read_csv(errors_path)
            if errors_df.empty:
                return ""
            scoped = errors_df
            if "pseudoprofile_id" in errors_df.columns:
                scoped = errors_df[errors_df["pseudoprofile_id"].astype(str) == str(profile_id)]
                if scoped.empty:
                    scoped = errors_df
            row = scoped.iloc[0].to_dict()
            if "error_message" in row:
                return str(row.get("error_message") or "").strip()
            if "error" in row:
                return str(row.get("error") or "").strip()
            compact = {str(k): str(v) for k, v in row.items() if str(v).strip()}
            return json.dumps(compact, ensure_ascii=False)
        except Exception:
            return ""

    def _discover_latest_hyde_dense_profiles(self) -> Path:
        root = self.evaluation_root / "03_construction_initial_observation_model/helpers/00_HyDe_based_predictor_ranks/runs"
        if not root.exists():
            return root / "dense_profiles.csv"
        candidates = sorted(root.glob("*/dense_profiles.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            return root / "dense_profiles.csv"
        return candidates[0]

    def _resolve_evaluation_root(self) -> Path:
        low = self.repo_root / "evaluation"
        if low.exists():
            return low
        up = self.repo_root / "Evaluation"
        if up.exists():
            return up
        return low

    @staticmethod
    def _safe_read_csv(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        for sep in [",", ";", "\t", "|"]:
            try:
                frame = pd.read_csv(path, sep=sep, engine="python")
                if frame.shape[1] > 1 or sep == ",":
                    return frame
            except Exception:
                pass
        return pd.read_csv(path, engine="python")

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _build_heuristic_initial_model(self, *, profile_id: str, complaint_text: str) -> Dict[str, Any]:
        source_runs = self.evaluation_root / "03_construction_initial_observation_model/constructed_PC_models/runs"
        sample_jsons = sorted(source_runs.glob("*/profiles/*/llm_observation_model_final.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if sample_jsons:
            payload = self._read_json(sample_jsons[0])
        else:
            payload = {}
        if not payload:
            payload = {
                "pseudoprofile_id": profile_id,
                "model_summary": "Heuristic fallback model generated with LLM disabled.",
                "variable_selection_notes": "Fallback model preserves PHOENIX flow without actor LLM.",
                "design_recommendations": {"study_days": 14},
                "criteria_variables": [],
                "predictor_variables": [],
            }
        payload["pseudoprofile_id"] = profile_id
        payload["model_summary"] = (
            "Heuristic fallback model generated with LLM disabled. "
            + str(payload.get("model_summary") or "")
        ).strip()
        payload["variable_selection_notes"] = (
            str(payload.get("variable_selection_notes") or "")
            + f"\nComplaint context: {complaint_text[:400]}"
        ).strip()
        return payload

    def _write_communication_summary(
        self,
        *,
        session_id: str,
        stage: str,
        llm_model: str,
        disable_llm: bool,
        evidence: Dict[str, Any],
    ) -> Dict[str, Any]:
        paths = self.session_store.session_paths(session_id)
        out = generate_communication_summary(
            stage=stage,
            evidence=evidence,
            llm_model=llm_model,
            disable_llm=disable_llm,
        )
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = paths["frontend_logs_root"] / f"communication_{stage}_{stamp}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "path": str(target),
            "payload": out,
        }

    def _load_latest_communication_summary(self, *, session_id: str) -> Dict[str, Any]:
        paths = self.session_store.session_paths(session_id)
        files = sorted(paths["frontend_logs_root"].glob("communication_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            return {}
        payload = self._read_json(files[0])
        return {
            "path": str(files[0]),
            "payload": payload,
        }

    def _extract_cycle_dashboard(self, *, cycle_root: Path, profile_id: str, session_id: str) -> Dict[str, Any]:
        readiness = self._read_json(cycle_root / "00_readiness_check" / profile_id / "readiness_report.json")
        network = self._read_json(cycle_root / "01_time_series_analysis/network" / profile_id / "comparison_summary.json")
        impact_df = self._safe_read_csv(cycle_root / "02_momentary_impact_coefficients" / profile_id / "predictor_composite.csv")
        step03_path = cycle_root / "03_treatment_target_handoff" / profile_id / "step03_target_selection.json"
        step04_path = cycle_root / "03_treatment_target_handoff" / profile_id / "step04_updated_observation_model.json"
        step05_path = cycle_root / "03b_translation_digital_intervention" / profile_id / "step05_hapa_intervention.json"
        step03 = self._read_json(step03_path)
        step04 = self._read_json(step04_path)
        step05 = self._read_json(step05_path)

        top_predictors: List[Dict[str, Any]] = []
        if not impact_df.empty:
            score_col = "predictor_impact"
            if score_col not in impact_df.columns and "predictor_impact_pct" in impact_df.columns:
                score_col = "predictor_impact_pct"
            if score_col in impact_df.columns:
                local = impact_df.copy()
                local["_score"] = pd.to_numeric(local[score_col], errors="coerce").fillna(0.0)
                if local["_score"].max() > 1.0:
                    local["_score"] = local["_score"] / 1000.0
                local = local.sort_values("_score", ascending=False).head(10)
                for _, row in local.iterrows():
                    top_predictors.append(
                        {
                            "predictor": str(row.get("predictor") or ""),
                            "score_0_1": float(max(0.0, min(1.0, row.get("_score", 0.0)))),
                        }
                    )

        recommended_targets = [
            {
                "predictor": str(item.get("predictor") or ""),
                "score_0_1": float(item.get("score_0_1") or 0.0),
                "rationale": str(item.get("rationale") or ""),
            }
            for item in (step03.get("recommended_targets") or [])
            if isinstance(item, dict)
        ]
        updated_predictors = [str(item) for item in (step04.get("recommended_next_observation_predictors") or []) if str(item).strip()]
        selected_barriers = [
            {
                "barrier_name": str(item.get("barrier_name") or ""),
                "score_0_1": float(item.get("score_0_1") or 0.0),
            }
            for item in (step05.get("selected_barriers") or [])
            if isinstance(item, dict)
        ]
        selected_coping = [
            {
                "coping_name": str(item.get("coping_name") or ""),
                "score_0_1": float(item.get("score_0_1") or 0.0),
            }
            for item in (step05.get("selected_coping_strategies") or [])
            if isinstance(item, dict)
        ]

        overall = readiness.get("overall", {}) if isinstance(readiness.get("overall"), dict) else {}
        execution_plan = network.get("execution_plan", {}) if isinstance(network.get("execution_plan"), dict) else {}
        network_notes: List[str] = []
        for note in execution_plan.get("notes", []) if isinstance(execution_plan.get("notes"), list) else []:
            txt = str(note).strip()
            if txt:
                network_notes.append(txt)
        for note in execution_plan.get("why_not_time_varying", []) if isinstance(execution_plan.get("why_not_time_varying"), list) else []:
            txt = str(note).strip()
            if txt:
                network_notes.append(txt)
        if not network_notes and isinstance(network.get("notes"), list):
            network_notes.extend([str(item) for item in network.get("notes") if str(item).strip()])
        if not network_notes and execution_plan:
            network_notes.append(
                f"Execution plan: {execution_plan.get('analysis_set', 'unknown')} "
                f"(time-varying={bool(execution_plan.get('run_tv_gvar'))}, "
                f"stationary={bool(execution_plan.get('run_stationary_gvar'))}, "
                f"correlation={bool(execution_plan.get('run_correlation_baseline'))})."
            )

        step03_status = "generated" if step03_path.exists() else "skipped"
        step04_status = "generated" if step04_path.exists() else "skipped"
        step05_status = "generated" if step05_path.exists() else "skipped"

        impact_status = "generated" if top_predictors else "skipped"
        impact_reason = ""
        if impact_status == "skipped":
            plan_block = execution_plan.get("can_compute_momentary_impact")
            if plan_block is False:
                impact_reason = "Impact was skipped because the selected readiness tier does not permit lag-based impact quantification."
            elif impact_df.empty:
                impact_reason = "No impact profiles were produced in this run."

        readiness_panel = {
            "label": str(overall.get("readiness_label") or ""),
            "score_0_100": float(overall.get("readiness_score_0_100") or 0.0),
            "tier": str(overall.get("recommended_tier") or ""),
            "tier3_variant": str(overall.get("tier3_variant") or ""),
            "analysis_execution_plan": overall.get("analysis_execution_plan", {}),
            "why": [str(item) for item in (overall.get("why") or [])[:8]],
        }
        network_panel = {
            "method_path": str(network.get("executed_path") or network.get("selected_method_path") or ""),
            "analysis_set": str(execution_plan.get("analysis_set") or ""),
            "notes": network_notes[:8],
        }
        visuals = self._collect_cycle_visuals(cycle_root=cycle_root, profile_id=profile_id, session_id=session_id)
        return {
            "profile_id": profile_id,
            "cycle_root": str(cycle_root),
            "readiness": readiness_panel,
            "network": network_panel,
            "impact": {"top_predictors": top_predictors, "status": impact_status, "status_reason": impact_reason},
            "step03": {"recommended_targets": recommended_targets, "status": step03_status},
            "step04": {
                "recommended_predictors": updated_predictors,
                "retained_criteria_ids": [str(item) for item in (step04.get("retained_criteria_ids") or [])],
                "reason_codes": [str(item) for item in (step04.get("range_policy_reason_codes") or [])],
                "status": step04_status,
            },
            "step05": {
                "selected_barriers": selected_barriers,
                "selected_coping": selected_coping,
                "user_summary": str(step05.get("user_friendly_summary") or step05.get("personalized_message") or ""),
                "status": step05_status,
            },
            "visuals": visuals,
        }

    def _collect_cycle_visuals(self, *, cycle_root: Path, profile_id: str, session_id: str) -> List[Dict[str, str]]:
        session_root = self.session_store.session_paths(session_id)["session_root"]
        out: List[Dict[str, str]] = []
        candidates: List[Path] = []
        for base in [
            cycle_root / "00_readiness_check" / profile_id,
            cycle_root / "01_time_series_analysis/network" / profile_id,
            cycle_root / "02_momentary_impact_coefficients" / profile_id / "visuals",
            cycle_root / "03_treatment_target_handoff" / profile_id / "visuals",
            cycle_root / "03b_translation_digital_intervention" / profile_id / "visuals",
            cycle_root / "04_impact_visualizations" / profile_id / "visuals",
        ]:
            if base.exists():
                candidates.extend([p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in {".png", ".svg", ".pdf"}])
        for path in sorted(candidates):
            try:
                rel = str(path.resolve().relative_to(session_root.resolve()))
            except Exception:
                continue
            out.append({"name": path.name, "relative_path": rel})
        return out

    def _assert_operationalization_cache(self) -> None:
        required = [
            "CRITERION_leaf_paths_EMBEDTEXT.json",
            "CRITERION_leaf_paths_FULL.json",
            "CRITERION_leaf_paths_LEXTEXT.json",
            "CRITERION_leaf_embeddings.npy",
            "CRITERION_leaf_embedding_norms.npy",
            "CRITERION_leaf_embeddings_meta.json",
        ]
        missing = [name for name in required if not (self.operationalization_cache_dir / name).exists()]
        if missing:
            raise RuntimeError(
                "Operationalization cache is incomplete. Missing: "
                + ", ".join(missing)
                + ". Run the embedding cache builder before frontend Step 01."
            )

    @staticmethod
    def _cycle_run_root(*, output_root: Path, run_id: str, cycle_index: int) -> Path:
        if int(cycle_index) <= 1:
            return output_root / run_id
        return output_root / run_id / "cycles" / f"cycle_{int(cycle_index):02d}"
