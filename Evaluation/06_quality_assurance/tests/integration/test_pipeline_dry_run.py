from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_pipeline_dry_run(tmp_path: Path, repo_file_fn) -> None:
    script = repo_file_fn("Evaluation/00_pipeline_orchestration/run_pseudodata_to_impact.py")
    pseudodata_root = tmp_path / "pseudodata"
    profile_dir = pseudodata_root / "pseudoprofile_test001"
    profile_dir.mkdir(parents=True)
    (profile_dir / "pseudodata_wide.csv").write_text("t_index,P01,C01\n1,1,1\n2,2,2\n", encoding="utf-8")

    output_root = tmp_path / "runs"
    run_id = "pytest_dry_run"
    cmd = [
        sys.executable,
        str(script),
        "--repo-root",
        str(repo_file_fn(".")),
        "--pseudodata-root",
        str(pseudodata_root),
        "--output-root",
        str(output_root),
        "--run-id",
        run_id,
        "--pattern",
        "pseudoprofile_test001",
        "--max-profiles",
        "1",
        "--dry-run",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr

    summary_path = output_root / run_id / "pipeline_summary.json"
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["run_id"] == run_id
