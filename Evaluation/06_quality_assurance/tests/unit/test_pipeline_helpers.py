from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _mod(module_loader, repo_file_fn):
    return module_loader(
        str(repo_file_fn("Evaluation/00_pipeline_orchestration/run_pseudodata_to_impact.py")),
        "phoenix_pipeline_module",
    )


def test_discover_profiles_filters_and_limits(tmp_path: Path, module_loader, repo_file_fn) -> None:
    module = _mod(module_loader, repo_file_fn)
    root = tmp_path / "pseudodata"
    (root / "pseudoprofile_A").mkdir(parents=True)
    (root / "pseudoprofile_A" / "pseudodata_wide.csv").write_text("t,P01,C01\n1,1,1\n", encoding="utf-8")
    (root / "pseudoprofile_B").mkdir(parents=True)
    (root / "pseudoprofile_B" / "pseudodata_wide.csv").write_text("t,P01,C01\n1,1,1\n", encoding="utf-8")

    profiles = module.discover_profiles(
        pseudodata_root=root,
        filename="pseudodata_wide.csv",
        pattern="pseudoprofile_A",
        max_profiles=10,
    )
    assert profiles == ["pseudoprofile_A"]

    profiles_limited = module.discover_profiles(
        pseudodata_root=root,
        filename="pseudodata_wide.csv",
        pattern="pseudoprofile_",
        max_profiles=1,
    )
    assert len(profiles_limited) == 1


def test_validate_root_outputs_success(tmp_path: Path, module_loader, repo_file_fn) -> None:
    module = _mod(module_loader, repo_file_fn)
    root = tmp_path / "report"
    root.mkdir(parents=True)
    (root / "run_report.md").write_text("# ok\n", encoding="utf-8")
    (root / "run_report.json").write_text("{}", encoding="utf-8")
    logger = module.PipelineLogger(tmp_path / "logs" / "pipeline.jsonl")

    module.validate_root_outputs(
        stage_name="reporting",
        root=root,
        required_relpaths=["run_report.md", "run_report.json"],
        logger=logger,
    )
