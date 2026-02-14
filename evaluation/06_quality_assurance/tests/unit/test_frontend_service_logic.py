from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _load_service_module(repo_root):
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from frontend.phoenix_frontend.services import phoenix_service as module

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
