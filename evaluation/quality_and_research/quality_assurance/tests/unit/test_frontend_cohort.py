from __future__ import annotations

import sys
import time
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


def test_build_cohort_cases_count_and_variants(repo_root):
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.frontend.phoenix_frontend.services.cohort import build_cohort_cases

    cases = build_cohort_cases(
        base_complaint="Mood is unstable.",
        base_person="",
        base_context="",
        patient_count=10,
    )
    assert len(cases) == 10
    assert cases[0].variant_label
    assert cases[0].complaint_text
    assert cases[1].variant_label != cases[0].variant_label



def test_full_cohort_api_job_starts_and_finishes(monkeypatch, repo_root):
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.frontend.phoenix_frontend import create_app

    app = create_app()
    session_store = app.extensions["phoenix.session_store"]
    service = app.extensions["phoenix.service"]
    session = session_store.create_session(complaint_text="Mood instability with highs and lows.")

    def _fake_run_full_cohort(**kwargs):
        log_fn = kwargs["log"]
        log_fn("[component:cohort_batch] status=running", "INFO")
        log_fn("[component:cohort_batch] status=succeeded", "INFO")
        return {
            "run_id": "cohort_test_run",
            "patient_count": 10,
            "completed_count": 10,
            "failed_count": 0,
        }

    monkeypatch.setattr(service, "run_full_cohort", _fake_run_full_cohort)

    client = app.test_client()
    resp = client.post(f"/api/sessions/{session.session_id}/jobs/full-cohort", json={"patient_count": 10})
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload and payload.get("status") == "ok"
    job_id = str(payload.get("job_id") or "")
    assert job_id.startswith("job_")

    terminal = None
    for _ in range(40):
        poll = client.get(f"/api/jobs/{job_id}")
        assert poll.status_code == 200
        job_payload = poll.get_json() or {}
        status = ((job_payload.get("job") or {}).get("status") or "").lower()
        if status in {"succeeded", "failed"}:
            terminal = status
            break
        time.sleep(0.05)

    assert terminal == "succeeded"
