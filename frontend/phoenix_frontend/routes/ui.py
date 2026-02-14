from __future__ import annotations

from flask import Blueprint, current_app, redirect, render_template, request, url_for

from ..services import PhoenixService, SessionStore


ui_bp = Blueprint("ui", __name__)


def _session_store() -> SessionStore:
    return current_app.extensions["phoenix.session_store"]


def _service() -> PhoenixService:
    return current_app.extensions["phoenix.service"]


@ui_bp.get("/")
def index():
    store = _session_store()
    sessions = [item.to_dict() for item in store.list_sessions(limit=30)]
    return render_template(
        "index.html",
        sessions=sessions,
        llm_globally_disabled=bool(current_app.config.get("PHOENIX_DISABLE_LLM", False)),
    )


@ui_bp.post("/sessions/new")
def create_session_from_form():
    complaint_text = str(request.form.get("complaint_text") or "").strip()
    person_text = str(request.form.get("person_text") or "").strip()
    context_text = str(request.form.get("context_text") or "").strip()
    profile_id = str(request.form.get("profile_id") or "").strip()
    if not complaint_text:
        sessions = [item.to_dict() for item in _session_store().list_sessions(limit=30)]
        return render_template(
            "index.html",
            sessions=sessions,
            error="A complaint text is required to create a session.",
            llm_globally_disabled=bool(current_app.config.get("PHOENIX_DISABLE_LLM", False)),
        )
    session = _session_store().create_session(
        complaint_text=complaint_text,
        person_text=person_text,
        context_text=context_text,
        profile_id=profile_id,
    )
    return redirect(url_for("ui.session_detail", session_id=session.session_id))


@ui_bp.get("/sessions/<session_id>")
def session_detail(session_id: str):
    try:
        snapshot = _service().session_snapshot(session_id)
    except FileNotFoundError:
        return render_template(
            "index.html",
            sessions=[item.to_dict() for item in _session_store().list_sessions(limit=30)],
            error="Session not found.",
            llm_globally_disabled=bool(current_app.config.get("PHOENIX_DISABLE_LLM", False)),
        )
    return render_template(
        "session.html",
        snapshot=snapshot,
        llm_globally_disabled=bool(current_app.config.get("PHOENIX_DISABLE_LLM", False)),
    )
