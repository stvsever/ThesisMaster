from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_visualization_component_generates_outputs(tmp_path: Path, repo_file_fn) -> None:
    script = repo_file_fn(
        "SystemComponents/Hierarchical_Updating_Algorithm/02_hierarchical_update_ranking/01_momentary_impact_quantification/02_visualize_impact_coefficients.py"
    )
    root = tmp_path / "impact_root"
    profile_dir = root / "pseudoprofile_test001"
    profile_dir.mkdir(parents=True)
    (profile_dir / "impact_matrix.csv").write_text(
        ",P01,P02\nC01,0.10,0.40\nC02,0.05,0.20\n",
        encoding="utf-8",
    )

    cmd = [
        sys.executable,
        str(script),
        "--root",
        str(root),
        "--pattern",
        "pseudoprofile_test001",
        "--max-profiles",
        "1",
        "--top-k-edges",
        "4",
    ]
    env = os.environ.copy()
    mpl_dir = tmp_path / "mplconfig"
    cache_dir = tmp_path / "xdg_cache"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    env["MPLCONFIGDIR"] = str(mpl_dir)
    env["XDG_CACHE_HOME"] = str(cache_dir)
    env["MPLBACKEND"] = "Agg"
    env["HOME"] = str(tmp_path)
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr

    summary_path = root / "visualization_run_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["n_profiles_success"] == 1
    assert (profile_dir / "visuals" / "heatmap_impact_matrix.png").exists()
