from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

FRONTEND_PACKAGE_ROOT = Path(__file__).resolve().parents[5] / "src" / "frontend" / "phoenix_frontend"

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        not FRONTEND_PACKAGE_ROOT.exists(),
        reason="src/frontend is intentionally absent from the public repository checkout",
    ),
]


def _load_service_module(repo_root):
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.frontend.phoenix_frontend.services import phoenix_service as module

    return module


def test_extract_step02_worker_error_prefers_profile_row(repo_root, tmp_path: Path):
    module = _load_service_module(repo_root)
    run_dir = tmp_path / "runs" / "frontend_run_x"
    run_dir.mkdir(parents=True, exist_ok=True)
    errors_csv = run_dir / "errors.csv"
    errors_csv.write_text(
        "pseudoprofile_id,error_message\n"
        "pseudoprofile_A,TypeError: mismatch\n"
        "pseudoprofile_B,RuntimeError: failed\n",
        encoding="utf-8",
    )

    service = module.PhoenixService.__new__(module.PhoenixService)
    message = module.PhoenixService._extract_step02_worker_error(
        service,
        run_dir=run_dir,
        profile_id="pseudoprofile_B",
    )
    assert "RuntimeError: failed" in message


def test_extract_step02_worker_error_empty_when_missing(repo_root, tmp_path: Path):
    module = _load_service_module(repo_root)
    run_dir = tmp_path / "runs" / "frontend_run_y"
    run_dir.mkdir(parents=True, exist_ok=True)

    service = module.PhoenixService.__new__(module.PhoenixService)
    message = module.PhoenixService._extract_step02_worker_error(
        service,
        run_dir=run_dir,
        profile_id="pseudoprofile_X",
    )
    assert message == ""


def test_load_latest_communication_prefers_cycle_stage(repo_root, tmp_path: Path):
    module = _load_service_module(repo_root)
    logs_root = tmp_path / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    (logs_root / "communication_initial_model_1.json").write_text(
        '{"stage":"initial_model","summary":{"headline":"init"}}',
        encoding="utf-8",
    )
    (logs_root / "communication_cycle_01_2.json").write_text(
        '{"stage":"cycle_01","summary":{"headline":"cycle"}}',
        encoding="utf-8",
    )

    class _Store:
        def session_paths(self, _session_id):
            return {"frontend_logs_root": logs_root}

    service = module.PhoenixService.__new__(module.PhoenixService)
    service.session_store = _Store()
    payload = module.PhoenixService._load_latest_communication_summary(service, session_id="s_test")
    assert payload
    assert payload["payload"]["stage"] == "cycle_01"


def test_run_pipeline_cycle_adds_start_from_pseudodata_after_cycle_one(repo_root, tmp_path: Path):
    module = _load_service_module(repo_root)
    from src.frontend.phoenix_frontend.services.session_store import SessionStore

    store = SessionStore(tmp_path / "sessions")
    session = store.create_session(
        complaint_text="test complaint",
        person_text="",
        context_text="",
        profile_id="pseudoprofile_FTC_ID001",
    )
    paths = store.session_paths(session.session_id)
    (paths["pseudodata_profile_root"] / "pseudodata_wide.csv").write_text(
        "t_index,date,P01\n0,2025-01-01,0.3\n",
        encoding="utf-8",
    )
    model_path = paths["outputs_root"] / "model.json"
    model_path.write_text(json.dumps({"model_summary": "ok"}), encoding="utf-8")
    store.update_session(
        session.session_id,
        latest_model_json=str(model_path),
        pipeline_run_id="unit_run",
        current_cycle=1,
    )

    expected_cycle_root = paths["pipeline_root"] / "unit_run" / "cycles" / "cycle_02"
    expected_cycle_root.mkdir(parents=True, exist_ok=True)
    (expected_cycle_root / "pipeline_summary.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "run_id": "unit_run",
                "cycle_index": 2,
                "engine_stage_flow": [],
                "quality_and_research_flow": [],
                "stage_results": [],
            }
        ),
        encoding="utf-8",
    )

    service = module.PhoenixService(
        repo_root=repo_root,
        python_exe=sys.executable,
        session_store=store,
    )
    captured = {"cmd": []}

    def _fake_run_command(*, cmd, log, env=None, component="", mark_success=True):  # noqa: ANN001
        captured["cmd"] = list(cmd)

    service._run_command = _fake_run_command  # type: ignore[method-assign]
    service._write_communication_summary = lambda **_: {"path": "", "payload": {}}  # type: ignore[assignment]

    result = service.run_pipeline_cycle(
        session_id=session.session_id,
        hard_ontology_constraint=False,
        llm_model="gpt-5-nano",
        disable_llm=True,
        include_intervention=True,
        request_model_refinement=False,
        profile_memory_window=3,
        handoff_critic_max_iterations=2,
        handoff_critic_pass_threshold=0.74,
        intervention_critic_max_iterations=2,
        intervention_critic_pass_threshold=0.74,
        network_boot=40,
        network_block_len=14,
        network_jobs=1,
        network_execution_policy="readiness_aligned",
        run_impact_visualizations=False,
        run_treatment_communication=False,
        parallel_branches=True,
        log=lambda *_: None,
    )

    assert "--start-from-pseudodata" in captured["cmd"]
    assert result["cycle_index"] == 2


def test_run_pipeline_cycle_respects_network_jobs_when_not_forced(repo_root, tmp_path: Path, monkeypatch):
    module = _load_service_module(repo_root)
    from src.frontend.phoenix_frontend.services.session_store import SessionStore

    monkeypatch.delenv("PHOENIX_FORCE_SINGLE_NETWORK_JOB", raising=False)
    store = SessionStore(tmp_path / "sessions")
    session = store.create_session(
        complaint_text="test complaint",
        person_text="",
        context_text="",
        profile_id="pseudoprofile_FTC_ID001",
    )
    paths = store.session_paths(session.session_id)
    (paths["pseudodata_profile_root"] / "pseudodata_wide.csv").write_text(
        "t_index,date,P01\n0,2025-01-01,0.3\n",
        encoding="utf-8",
    )
    model_path = paths["outputs_root"] / "model.json"
    model_path.write_text(json.dumps({"model_summary": "ok"}), encoding="utf-8")
    store.update_session(
        session.session_id,
        latest_model_json=str(model_path),
        pipeline_run_id="unit_run_parallel_jobs",
        current_cycle=0,
    )
    expected_cycle_root = paths["pipeline_root"] / "unit_run_parallel_jobs" / "pipeline_summary.json"
    expected_cycle_root.parent.mkdir(parents=True, exist_ok=True)
    expected_cycle_root.write_text(
        json.dumps(
            {
                "status": "ok",
                "run_id": "unit_run_parallel_jobs",
                "cycle_index": 1,
                "engine_stage_flow": [],
                "quality_and_research_flow": [],
                "stage_results": [],
            }
        ),
        encoding="utf-8",
    )

    service = module.PhoenixService(repo_root=repo_root, python_exe=sys.executable, session_store=store)
    captured = {"cmd": []}

    def _fake_run_command(*, cmd, log, env=None, component="", mark_success=True):  # noqa: ANN001
        captured["cmd"] = list(cmd)

    service._run_command = _fake_run_command  # type: ignore[method-assign]
    service._write_communication_summary = lambda **_: {"path": "", "payload": {}}  # type: ignore[assignment]
    service._extract_cycle_dashboard = lambda **_: {}  # type: ignore[assignment]

    service.run_pipeline_cycle(
        session_id=session.session_id,
        hard_ontology_constraint=False,
        llm_model="gpt-5-nano",
        disable_llm=True,
        include_intervention=True,
        request_model_refinement=False,
        profile_memory_window=3,
        handoff_critic_max_iterations=2,
        handoff_critic_pass_threshold=0.74,
        intervention_critic_max_iterations=2,
        intervention_critic_pass_threshold=0.74,
        network_boot=40,
        network_block_len=14,
        network_jobs=4,
        network_execution_policy="readiness_aligned",
        run_impact_visualizations=False,
        run_treatment_communication=False,
        parallel_branches=True,
        log=lambda *_: None,
    )
    idx = captured["cmd"].index("--network-jobs")
    assert captured["cmd"][idx + 1] == "4"


def test_run_pipeline_cycle_clamps_network_jobs_when_forced(repo_root, tmp_path: Path, monkeypatch):
    module = _load_service_module(repo_root)
    from src.frontend.phoenix_frontend.services.session_store import SessionStore

    monkeypatch.setenv("PHOENIX_FORCE_SINGLE_NETWORK_JOB", "1")
    store = SessionStore(tmp_path / "sessions")
    session = store.create_session(
        complaint_text="test complaint",
        person_text="",
        context_text="",
        profile_id="pseudoprofile_FTC_ID001",
    )
    paths = store.session_paths(session.session_id)
    (paths["pseudodata_profile_root"] / "pseudodata_wide.csv").write_text(
        "t_index,date,P01\n0,2025-01-01,0.3\n",
        encoding="utf-8",
    )
    model_path = paths["outputs_root"] / "model.json"
    model_path.write_text(json.dumps({"model_summary": "ok"}), encoding="utf-8")
    store.update_session(
        session.session_id,
        latest_model_json=str(model_path),
        pipeline_run_id="unit_run_forced_single_job",
        current_cycle=0,
    )
    expected_summary = paths["pipeline_root"] / "unit_run_forced_single_job" / "pipeline_summary.json"
    expected_summary.parent.mkdir(parents=True, exist_ok=True)
    expected_summary.write_text(
        json.dumps(
            {
                "status": "ok",
                "run_id": "unit_run_forced_single_job",
                "cycle_index": 1,
                "engine_stage_flow": [],
                "quality_and_research_flow": [],
                "stage_results": [],
            }
        ),
        encoding="utf-8",
    )

    service = module.PhoenixService(repo_root=repo_root, python_exe=sys.executable, session_store=store)
    captured = {"cmd": []}

    def _fake_run_command(*, cmd, log, env=None, component="", mark_success=True):  # noqa: ANN001
        captured["cmd"] = list(cmd)

    service._run_command = _fake_run_command  # type: ignore[method-assign]
    service._write_communication_summary = lambda **_: {"path": "", "payload": {}}  # type: ignore[assignment]
    service._extract_cycle_dashboard = lambda **_: {}  # type: ignore[assignment]

    service.run_pipeline_cycle(
        session_id=session.session_id,
        hard_ontology_constraint=False,
        llm_model="gpt-5-nano",
        disable_llm=True,
        include_intervention=True,
        request_model_refinement=False,
        profile_memory_window=3,
        handoff_critic_max_iterations=2,
        handoff_critic_pass_threshold=0.74,
        intervention_critic_max_iterations=2,
        intervention_critic_pass_threshold=0.74,
        network_boot=40,
        network_block_len=14,
        network_jobs=4,
        network_execution_policy="readiness_aligned",
        run_impact_visualizations=False,
        run_treatment_communication=False,
        parallel_branches=True,
        log=lambda *_: None,
    )
    idx = captured["cmd"].index("--network-jobs")
    assert captured["cmd"][idx + 1] == "1"


def test_run_pipeline_cycle_retries_with_single_network_job_on_failure(repo_root, tmp_path: Path, monkeypatch):
    module = _load_service_module(repo_root)
    from src.frontend.phoenix_frontend.services.session_store import SessionStore

    monkeypatch.delenv("PHOENIX_FORCE_SINGLE_NETWORK_JOB", raising=False)
    store = SessionStore(tmp_path / "sessions")
    session = store.create_session(
        complaint_text="test complaint",
        person_text="",
        context_text="",
        profile_id="pseudoprofile_FTC_ID001",
    )
    paths = store.session_paths(session.session_id)
    (paths["pseudodata_profile_root"] / "pseudodata_wide.csv").write_text(
        "t_index,date,P01\n0,2025-01-01,0.3\n",
        encoding="utf-8",
    )
    model_path = paths["outputs_root"] / "model.json"
    model_path.write_text(json.dumps({"model_summary": "ok"}), encoding="utf-8")
    store.update_session(
        session.session_id,
        latest_model_json=str(model_path),
        pipeline_run_id="unit_run_retry_single_job",
        current_cycle=0,
    )
    expected_summary = paths["pipeline_root"] / "unit_run_retry_single_job" / "pipeline_summary.json"
    expected_summary.parent.mkdir(parents=True, exist_ok=True)
    expected_summary.write_text(
        json.dumps(
            {
                "status": "ok",
                "run_id": "unit_run_retry_single_job",
                "cycle_index": 1,
                "engine_stage_flow": [],
                "quality_and_research_flow": [],
                "stage_results": [],
            }
        ),
        encoding="utf-8",
    )

    service = module.PhoenixService(repo_root=repo_root, python_exe=sys.executable, session_store=store)
    calls = []

    def _fake_run_command(*, cmd, log, env=None, component="", mark_success=True):  # noqa: ANN001
        calls.append(list(cmd))
        idx = cmd.index("--network-jobs")
        jobs = str(cmd[idx + 1])
        if len(calls) == 1 and jobs == "4":
            raise RuntimeError("simulated parallel network failure")

    service._run_command = _fake_run_command  # type: ignore[method-assign]
    service._write_communication_summary = lambda **_: {"path": "", "payload": {}}  # type: ignore[assignment]
    service._extract_cycle_dashboard = lambda **_: {}  # type: ignore[assignment]

    service.run_pipeline_cycle(
        session_id=session.session_id,
        hard_ontology_constraint=False,
        llm_model="gpt-5-nano",
        disable_llm=True,
        include_intervention=True,
        request_model_refinement=False,
        profile_memory_window=3,
        handoff_critic_max_iterations=2,
        handoff_critic_pass_threshold=0.74,
        intervention_critic_max_iterations=2,
        intervention_critic_pass_threshold=0.74,
        network_boot=40,
        network_block_len=14,
        network_jobs=4,
        network_execution_policy="readiness_aligned",
        run_impact_visualizations=False,
        run_treatment_communication=False,
        parallel_branches=True,
        log=lambda *_: None,
    )
    assert len(calls) == 2
    idx0 = calls[0].index("--network-jobs")
    idx1 = calls[1].index("--network-jobs")
    assert calls[0][idx0 + 1] == "4"
    assert calls[1][idx1 + 1] == "1"


def test_operationalization_summary_marks_complaint_match(repo_root, tmp_path: Path):
    module = _load_service_module(repo_root)
    mapped_csv = tmp_path / "mapped_criterions.csv"
    mapped_csv.write_text(
        "pseudoprofile_id,complaint_text,variable_id,variable_label,mapping_status,chosen_confidence,chosen_leaf_embed_path\n"
        "pseudoprofile_FTC_ID001,Unique complaint token,C01,Mood variability,MAPPED,0.85,negative valence systems / potential threat\n",
        encoding="utf-8",
    )
    service = module.PhoenixService.__new__(module.PhoenixService)
    summary = module.PhoenixService._load_operationalization_summary(
        service,
        mapped_csv_path=mapped_csv,
        profile_id="pseudoprofile_FTC_ID001",
        fallback_complaint_text="Unique complaint token",
    )
    assert summary["source"] == "mapped_csv"
    assert summary["criteria_count"] == 1
    assert summary["matches_session_complaint"] is True
    assert summary["mismatch_warning"] == ""
    assert summary["confidence_avg_0_1"] == pytest.approx(0.85, rel=1e-6)
    assert summary["ontology_domains"] == ["negative valence systems"]


def test_operationalization_summary_flags_complaint_mismatch(repo_root, tmp_path: Path):
    module = _load_service_module(repo_root)
    mapped_csv = tmp_path / "mapped_criterions.csv"
    mapped_csv.write_text(
        "pseudoprofile_id,complaint_text,variable_id,variable_label,mapping_status,chosen_confidence\n"
        "pseudoprofile_FTC_ID001,Old complaint text,C01,Mood variability,MAPPED,0.60\n",
        encoding="utf-8",
    )
    service = module.PhoenixService.__new__(module.PhoenixService)
    summary = module.PhoenixService._load_operationalization_summary(
        service,
        mapped_csv_path=mapped_csv,
        profile_id="pseudoprofile_FTC_ID001",
        fallback_complaint_text="Fresh complaint text",
    )
    assert summary["matches_session_complaint"] is False
    assert "differs" in str(summary["mismatch_warning"]).lower()


def test_operationalization_summary_ignores_nan_variable_rows_and_exposes_errors(repo_root, tmp_path: Path):
    module = _load_service_module(repo_root)
    mapped_csv = tmp_path / "mapped_criterions.csv"
    mapped_csv.write_text(
        "pseudoprofile_id,complaint_text,variable_id,variable_label,mapping_status,error\n"
        "pseudoprofile_FTC_ID001,Complaint text,, ,ERROR,APIConnectionError: Connection error.\n",
        encoding="utf-8",
    )
    service = module.PhoenixService.__new__(module.PhoenixService)
    summary = module.PhoenixService._load_operationalization_summary(
        service,
        mapped_csv_path=mapped_csv,
        profile_id="pseudoprofile_FTC_ID001",
        fallback_complaint_text="Complaint text",
    )
    assert summary["source"] == "mapped_csv"
    assert summary["criteria_count"] == 0
    assert summary["variables"] == []
    assert summary["errors"]
    assert "apiconnectionerror" in str(summary["errors"][0]).lower()


def test_run_initial_model_writes_and_uses_session_complaint(repo_root, tmp_path: Path):
    module = _load_service_module(repo_root)
    from src.frontend.phoenix_frontend.services.session_store import SessionStore

    complaint = "UNIQUE_FRONTEND_COMPLAINT_TOKEN_ABC123"
    store = SessionStore(tmp_path / "sessions")
    session = store.create_session(
        complaint_text=complaint,
        person_text="",
        context_text="",
        profile_id="pseudoprofile_FTC_ID321",
    )
    service = module.PhoenixService(
        repo_root=repo_root,
        python_exe=sys.executable,
        session_store=store,
    )

    fake_hyde = tmp_path / "helpers" / "dense_profiles.csv"
    fake_hyde.parent.mkdir(parents=True, exist_ok=True)
    fake_hyde.write_text("pseudoprofile_id,dense_profile\n", encoding="utf-8")
    service._discover_latest_hyde_dense_profiles = lambda: fake_hyde  # type: ignore[method-assign]
    service._collect_profile_visuals = lambda *_args, **_kwargs: []  # type: ignore[method-assign]
    service._assert_operationalization_cache = lambda: None  # type: ignore[method-assign]

    captured = {"input_seen": False}

    def _fake_run_command(*, cmd, log, env=None, component="", mark_success=True):  # noqa: ANN001
        if component == "step01_operationalization":
            input_path = Path(cmd[cmd.index("--input-txt") + 1])
            output_path = Path(cmd[cmd.index("--output-csv") + 1])
            payload = input_path.read_text(encoding="utf-8")
            assert complaint in payload
            captured["input_seen"] = True
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                "pseudoprofile_id,complaint_text,variable_id,variable_label,mapping_status,chosen_confidence\n"
                f"{session.profile_id},{complaint},C01,Energy instability,MAPPED,0.82\n",
                encoding="utf-8",
            )
            return
        if component == "step02_initial_model":
            results_dir = Path(cmd[cmd.index("--results_dir") + 1])
            run_id = str(cmd[cmd.index("--run_id") + 1])
            pseudoprofile_id = str(cmd[cmd.index("--pseudoprofile_id") + 1])
            profile_dir = results_dir / run_id / "profiles" / pseudoprofile_id
            profile_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "pseudoprofile_id": pseudoprofile_id,
                "model_summary": "unit-test model",
                "criteria_variables": [
                    {
                        "var_id": "C01",
                        "label": "Energy instability",
                        "criterion_path": "criteria/high_level/energy_instability",
                        "polarity": "higher_is_worse",
                        "measurement": {
                            "item_or_signal": "Daily energy instability",
                            "response_scale_or_unit": "0-10",
                            "sampling_per_day": 1,
                        },
                    }
                ],
                "predictor_variables": [
                    {
                        "var_id": "P01",
                        "label": "Sleep regularity",
                        "ontology_path": "predictors/high_level/sleep_regular",
                        "expected_direction": "higher_is_better",
                        "measurement": {
                            "item_or_signal": "How regular was your sleep schedule?",
                            "response_scale_or_unit": "0-10",
                            "sampling_per_day": 1,
                        },
                    }
                ],
            }
            (profile_dir / "llm_observation_model_final.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return
        raise AssertionError(f"Unexpected component in fake run: {component}")

    service._run_command = _fake_run_command  # type: ignore[method-assign]
    result = service.run_initial_model(
        session_id=session.session_id,
        llm_model="gpt-5-nano",
        disable_llm=False,
        hard_ontology_constraint=False,
        prompt_budget_tokens=10000,
        critic_max_iterations=1,
        critic_pass_threshold=0.7,
        max_workers=1,
        generate_communication=False,
        log=lambda *_: None,
    )

    assert captured["input_seen"] is True
    mapped_csv = Path(result["mapped_criterions_csv"])
    assert mapped_csv.exists()
    assert complaint in mapped_csv.read_text(encoding="utf-8")


def test_run_initial_model_forwards_operationalization_controls(repo_root, tmp_path: Path):
    module = _load_service_module(repo_root)
    from src.frontend.phoenix_frontend.services.session_store import SessionStore

    store = SessionStore(tmp_path / "sessions")
    session = store.create_session(
        complaint_text="Need frontend control passthrough validation.",
        person_text="",
        context_text="",
        profile_id="pseudoprofile_FTC_ID654",
    )
    service = module.PhoenixService(
        repo_root=repo_root,
        python_exe=sys.executable,
        session_store=store,
    )

    fake_hyde = tmp_path / "helpers" / "dense_profiles.csv"
    fake_hyde.parent.mkdir(parents=True, exist_ok=True)
    fake_hyde.write_text("pseudoprofile_id,dense_profile\n", encoding="utf-8")
    service._discover_latest_hyde_dense_profiles = lambda: fake_hyde  # type: ignore[method-assign]
    service._collect_profile_visuals = lambda *_args, **_kwargs: []  # type: ignore[method-assign]
    service._assert_operationalization_cache = lambda: None  # type: ignore[method-assign]

    captured: dict[str, object] = {}

    def _fake_run_command(*, cmd, log, env=None, component="", mark_success=True):  # noqa: ANN001
        if component == "step01_operationalization":
            captured["step01_cmd"] = list(cmd)
            captured["step01_env"] = dict(env or {})
            output_path = Path(cmd[cmd.index("--output-csv") + 1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                "pseudoprofile_id,complaint_text,variable_id,variable_label,mapping_status,chosen_confidence\n"
                f"{session.profile_id},Need frontend control passthrough validation.,C01,Stress load,MAPPED,0.79\n",
                encoding="utf-8",
            )
            return
        if component == "step02_initial_model":
            results_dir = Path(cmd[cmd.index("--results_dir") + 1])
            run_id = str(cmd[cmd.index("--run_id") + 1])
            pseudoprofile_id = str(cmd[cmd.index("--pseudoprofile_id") + 1])
            profile_dir = results_dir / run_id / "profiles" / pseudoprofile_id
            profile_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "pseudoprofile_id": pseudoprofile_id,
                "model_summary": "control passthrough",
                "criteria_variables": [],
                "predictor_variables": [],
            }
            (profile_dir / "llm_observation_model_final.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return
        raise AssertionError(f"Unexpected component in fake run: {component}")

    service._run_command = _fake_run_command  # type: ignore[method-assign]
    service.run_initial_model(
        session_id=session.session_id,
        llm_model="gpt-5-nano",
        disable_llm=False,
        hard_ontology_constraint=False,
        prompt_budget_tokens=10000,
        critic_max_iterations=1,
        critic_pass_threshold=0.7,
        max_workers=1,
        operationalization_enable_llm_rerank=False,
        operationalization_critic_max_iterations=4,
        operationalization_critic_pass_threshold=0.83,
        generate_communication=False,
        log=lambda *_: None,
    )

    step01_cmd = list(captured.get("step01_cmd") or [])
    step01_env = dict(captured.get("step01_env") or {})
    assert "--no-llm-rerank" in step01_cmd
    assert step01_env["CRITERION_DECOMP_MODEL"] == "gpt-5-nano"
    assert step01_env["CRITERION_DECOMP_CRITIC_MODEL"] == "gpt-5-nano"
    assert step01_env["CRITERION_RERANK_MODEL"] == "gpt-5-nano"
    assert step01_env["CRITERION_DECOMP_CRITIC_MAX_ITERATIONS"] == "4"
    assert step01_env["CRITERION_DECOMP_CRITIC_PASS_THRESHOLD"] == "0.8300"
