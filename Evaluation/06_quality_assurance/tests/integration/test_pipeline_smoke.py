from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.smoke]

def test_pipeline_smoke_single_profile(tmp_path: Path, repo_file_fn) -> None:
    if os.environ.get("PHOENIX_ENABLE_SMOKE", "0") != "1":
        pytest.skip("Set PHOENIX_ENABLE_SMOKE=1 to run the full smoke pipeline.")

    script = repo_file_fn("Evaluation/00_pipeline_orchestration/run_pipeline.py")
    output_root = tmp_path / "smoke_runs"
    run_id = "pytest_smoke_run"

    cmd = [
        sys.executable,
        str(script),
        "--mode",
        "synthetic_v1",
        "--repo-root",
        str(repo_file_fn(".")),
        "--output-root",
        str(output_root),
        "--run-id",
        run_id,
        "--pattern",
        "pseudoprofile_FTC_ID002",
        "--max-profiles",
        "1",
        "--network-boot",
        "3",
        "--network-block-len",
        "10",
        "--network-jobs",
        "1",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr

    summary_path = output_root / run_id / "pipeline_summary.json"
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert (output_root / run_id / "05_research_reports" / "run_report.md").exists()
