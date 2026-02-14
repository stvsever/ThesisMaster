from __future__ import annotations

import os

from flask import Flask

from .config import load_paths
from .routes.api import api_bp
from .routes.ui import ui_bp
from .services import JobManager, PhoenixService, SessionStore


def create_app() -> Flask:
    paths = load_paths()
    paths.workspace_root.mkdir(parents=True, exist_ok=True)
    paths.sessions_root.mkdir(parents=True, exist_ok=True)

    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.config["SECRET_KEY"] = "phoenix-frontend-dev"
    app.config["PHOENIX_REPO_ROOT"] = str(paths.repo_root)
    app.config["PHOENIX_WORKSPACE_ROOT"] = str(paths.workspace_root)
    app.config["PHOENIX_SESSIONS_ROOT"] = str(paths.sessions_root)
    app.config["PHOENIX_PYTHON_EXE"] = str(paths.python_exe)
    app.config["PHOENIX_DISABLE_LLM"] = str(os.environ.get("PHOENIX_DISABLE_LLM", "false")).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }

    session_store = SessionStore(paths.sessions_root)
    job_manager = JobManager()
    phoenix_service = PhoenixService(
        repo_root=paths.repo_root,
        python_exe=paths.python_exe,
        session_store=session_store,
    )

    app.extensions["phoenix.session_store"] = session_store
    app.extensions["phoenix.job_manager"] = job_manager
    app.extensions["phoenix.service"] = phoenix_service

    app.register_blueprint(ui_bp)
    app.register_blueprint(api_bp, url_prefix="/api")
    return app
