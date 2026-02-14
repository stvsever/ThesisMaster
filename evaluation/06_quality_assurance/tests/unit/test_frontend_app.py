from __future__ import annotations

import sys


def test_frontend_app_boots(repo_root):
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from frontend.phoenix_frontend import create_app

    app = create_app()
    client = app.test_client()
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"PHOENIX" in resp.data

    created = client.post(
        "/sessions/new",
        data={"complaint_text": "I feel persistently down and unmotivated."},
        follow_redirects=False,
    )
    assert created.status_code in {302, 303}
    location = created.headers.get("Location", "")
    assert "/sessions/" in location

    detail = client.get(location)
    assert detail.status_code == 200
    assert b"topbar-pipeline-nodes" in detail.data
    assert b"Workspace Control" in detail.data
    assert b"Runtime Components" in detail.data
    assert b"run-next-phase-btn" in detail.data
    assert b"Expand all sections" in detail.data
    assert b"Logs" in detail.data
    assert b"Realtime Process Logs" in detail.data
    assert b"cycle-request-refinement" in detail.data
    assert b"logs-drawer-open" in detail.data


def test_frontend_global_disable_llm_env(repo_root, monkeypatch):
    monkeypatch.setenv("PHOENIX_DISABLE_LLM", "true")
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from frontend.phoenix_frontend import create_app

    app = create_app()
    assert app.config.get("PHOENIX_DISABLE_LLM") is True
