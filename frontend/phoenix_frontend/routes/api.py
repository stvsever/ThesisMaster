from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

from flask import Blueprint, Response, current_app, jsonify, request, send_file, stream_with_context

from ..services import JobManager, PhoenixService, SessionStore
from ..services.job_manager import TERMINAL_STATUSES


api_bp = Blueprint("api", __name__)


def _session_store() -> SessionStore:
    return current_app.extensions["phoenix.session_store"]


def _job_manager() -> JobManager:
    return current_app.extensions["phoenix.job_manager"]


def _service() -> PhoenixService:
    return current_app.extensions["phoenix.service"]


def _llm_globally_disabled() -> bool:
    return bool(current_app.config.get("PHOENIX_DISABLE_LLM", False))


def _error(message: str, code: int = 400):
    return jsonify({"status": "error", "message": message}), int(code)


@api_bp.post("/sessions")
def create_session():
    payload = request.get_json(silent=True) or {}
    complaint_text = str(payload.get("complaint_text") or "").strip()
    if not complaint_text:
        return _error("complaint_text is required.", 400)
    session = _session_store().create_session(
        complaint_text=complaint_text,
        person_text=str(payload.get("person_text") or ""),
        context_text=str(payload.get("context_text") or ""),
        profile_id=str(payload.get("profile_id") or ""),
    )
    return jsonify({"status": "ok", "session": session.to_dict()})


@api_bp.get("/sessions/<session_id>/snapshot")
def session_snapshot(session_id: str):
    try:
        snapshot = _service().session_snapshot(session_id)
        return jsonify({"status": "ok", "snapshot": snapshot})
    except FileNotFoundError:
        return _error("Session not found.", 404)
    except Exception as exc:
        return _error(str(exc), 500)


def _start_initial_model_job(session_id: str, payload: Dict[str, Any]) -> str:
    manager = _job_manager()
    service = _service()
    disable_llm = _llm_globally_disabled() or bool(payload.get("disable_llm") or False)
    return manager.start_job(
        session_id=session_id,
        kind="initial_model",
        target=lambda log: service.run_initial_model(
            session_id=session_id,
            llm_model=str(payload.get("llm_model") or "gpt-5-nano"),
            disable_llm=disable_llm,
            hard_ontology_constraint=bool(payload.get("hard_ontology_constraint") or False),
            prompt_budget_tokens=int(payload.get("prompt_budget_tokens") or 400000),
            critic_max_iterations=int(payload.get("critic_max_iterations") or 2),
            critic_pass_threshold=float(payload.get("critic_pass_threshold") or 0.74),
            max_workers=int(payload.get("max_workers") or 12),
            log=log,
        ),
    )


def _start_synthesize_job(session_id: str, payload: Dict[str, Any]) -> str:
    manager = _job_manager()
    service = _service()
    baseline_rows = payload.get("baselines")
    if not isinstance(baseline_rows, list):
        baseline_rows = []
    return manager.start_job(
        session_id=session_id,
        kind="synthesize_pseudodata",
        target=lambda log: service.synthesize_session_pseudodata(
            session_id=session_id,
            n_points=int(payload.get("n_points") or 84),
            missing_rate=float(payload.get("missing_rate") or 0.1),
            seed=int(payload.get("seed") or 42),
            baseline_rows=baseline_rows,
            log=log,
        ),
    )


def _start_manual_data_job(session_id: str, payload: Dict[str, Any]) -> str:
    csv_text = str(payload.get("csv_text") or "").strip()
    if not csv_text:
        raise RuntimeError("csv_text is required for manual data upload.")
    manager = _job_manager()
    service = _service()
    return manager.start_job(
        session_id=session_id,
        kind="manual_data_upload",
        target=lambda log: service.save_manual_pseudodata(
            session_id=session_id,
            csv_text=csv_text,
            log=log,
        ),
    )


def _start_cycle_job(session_id: str, payload: Dict[str, Any]) -> str:
    manager = _job_manager()
    service = _service()
    disable_llm = _llm_globally_disabled() or bool(payload.get("disable_llm") or False)
    return manager.start_job(
        session_id=session_id,
        kind="pipeline_cycle",
        target=lambda log: service.run_pipeline_cycle(
            session_id=session_id,
            hard_ontology_constraint=bool(payload.get("hard_ontology_constraint") or False),
            llm_model=str(payload.get("llm_model") or "gpt-5-nano"),
            disable_llm=disable_llm,
            include_intervention=bool(payload.get("include_intervention", True)),
            request_model_refinement=bool(payload.get("request_model_refinement") or False),
            profile_memory_window=int(payload.get("profile_memory_window") or 3),
            handoff_critic_max_iterations=int(payload.get("handoff_critic_max_iterations") or 2),
            intervention_critic_max_iterations=int(payload.get("intervention_critic_max_iterations") or 2),
            log=log,
        ),
    )


@api_bp.post("/sessions/<session_id>/jobs/initial-model")
def start_initial_model_job(session_id: str):
    try:
        payload = request.get_json(silent=True) or {}
        job_id = _start_initial_model_job(session_id, payload)
        return jsonify({"status": "ok", "job_id": job_id})
    except FileNotFoundError:
        return _error("Session not found.", 404)
    except Exception as exc:
        return _error(str(exc), 400)


@api_bp.post("/sessions/<session_id>/jobs/synthesize")
def start_synthesize_job(session_id: str):
    try:
        payload = request.get_json(silent=True) or {}
        job_id = _start_synthesize_job(session_id, payload)
        return jsonify({"status": "ok", "job_id": job_id})
    except FileNotFoundError:
        return _error("Session not found.", 404)
    except Exception as exc:
        return _error(str(exc), 400)


@api_bp.post("/sessions/<session_id>/jobs/manual-data")
def start_manual_data_job(session_id: str):
    try:
        payload = request.get_json(silent=True) or {}
        job_id = _start_manual_data_job(session_id, payload)
        return jsonify({"status": "ok", "job_id": job_id})
    except FileNotFoundError:
        return _error("Session not found.", 404)
    except Exception as exc:
        return _error(str(exc), 400)


@api_bp.post("/sessions/<session_id>/jobs/run-cycle")
def start_pipeline_cycle_job(session_id: str):
    try:
        payload = request.get_json(silent=True) or {}
        job_id = _start_cycle_job(session_id, payload)
        return jsonify({"status": "ok", "job_id": job_id})
    except FileNotFoundError:
        return _error("Session not found.", 404)
    except Exception as exc:
        return _error(str(exc), 400)


@api_bp.get("/jobs/<job_id>")
def get_job(job_id: str):
    try:
        job = _job_manager().get_job(job_id)
    except KeyError:
        return _error("Job not found.", 404)
    return jsonify({"status": "ok", "job": job.to_dict()})


@api_bp.get("/jobs/<job_id>/logs")
def get_job_logs(job_id: str):
    try:
        after = int(request.args.get("after", 0))
    except Exception:
        after = 0
    try:
        logs = _job_manager().logs_since(job_id, after=after)
        job = _job_manager().get_job(job_id)
    except KeyError:
        return _error("Job not found.", 404)
    return jsonify(
        {
            "status": "ok",
            "job_id": job_id,
            "job_status": job.status,
            "logs": logs,
        }
    )


@api_bp.get("/jobs/<job_id>/stream")
def stream_job_logs(job_id: str):
    manager = _job_manager()
    try:
        manager.get_job(job_id)
    except KeyError:
        return _error("Job not found.", 404)

    @stream_with_context
    def _event_stream():
        cursor = 0
        while True:
            try:
                job = manager.get_job(job_id)
            except KeyError:
                break
            rows = manager.logs_since(job_id, after=cursor)
            for row in rows:
                cursor = max(cursor, int(row["index"]))
                payload = {"type": "log", "job_id": job_id, "entry": row}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            if job.status in TERMINAL_STATUSES:
                payload = {
                    "type": "status",
                    "job_id": job_id,
                    "status": job.status,
                    "kind": job.kind,
                    "error": job.error,
                    "finished_at": job.finished_at,
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                break
            time.sleep(0.6)

    return Response(
        _event_stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@api_bp.get("/sessions/<session_id>/files/<path:rel_path>")
def serve_session_file(session_id: str, rel_path: str):
    try:
        target = _session_store().resolve_session_file(session_id, rel_path)
    except FileNotFoundError:
        return _error("Session not found.", 404)
    except Exception as exc:
        return _error(str(exc), 400)

    if not target.exists() or not target.is_file():
        return _error("File not found.", 404)
    return send_file(Path(target))
