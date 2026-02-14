from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.unit


def _mod(module_loader, repo_file_fn):
    return module_loader(
        str(repo_file_fn("evaluation/00_pipeline_orchestration/iterative_cycle_dataflow.py")),
        "phoenix_iterative_dataflow_module",
    )


def test_choose_iterative_predictors_prefers_recommended_mapping(module_loader, repo_file_fn) -> None:
    module = _mod(module_loader, repo_file_fn)
    selected, ranking = module.choose_iterative_predictors(
        available_predictors=["P01", "P02", "P03"],
        predictor_path_map={
            "P01": "BIO / sleep / duration",
            "P02": "PSYCHO / activation / planning",
            "P03": "SOCIAL / support / interactions",
        },
        recommended_paths=[
            "PSYCHO / activation / planning / scheduling",
            "BIO / sleep / duration / consistency",
        ],
        impact_scores={"P01": 0.20, "P02": 0.55, "P03": 0.10},
        step03_scores={"P01": 0.30, "P02": 0.80, "P03": 0.10},
        step05_priorities={"P02": 0.85},
        preferred_count=2,
        min_count=2,
        max_count=8,
    )
    assert len(selected) == 2
    assert selected[0] == "P02"
    assert ranking[0]["predictor"] == "P02"


def test_simulate_iterative_cycle_wide_appends_points(module_loader, repo_file_fn) -> None:
    module = _mod(module_loader, repo_file_fn)
    history = pd.DataFrame(
        {
            "t_index": [0, 1, 2, 3],
            "date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
            "P01": [0.2, 0.3, 0.4, 0.5],
            "P02": [0.1, 0.15, 0.2, 0.25],
            "C01": [6.0, 5.7, 5.6, 5.5],
            "C02": [4.8, 4.7, 4.6, 4.5],
        }
    )
    simulated = module.simulate_iterative_cycle_wide(
        history_wide=history,
        predictor_codes=["P01", "P02"],
        criterion_codes=["C01", "C02"],
        targeted_predictors=["P01"],
        targeted_criteria=["C01"],
        new_points=5,
        noise_scale=0.01,
        improvement_strength=0.08,
        seed=1234,
    )
    assert len(simulated) == len(history) + 5
    assert list(simulated.columns) == list(history.columns)
    assert int(simulated["t_index"].iloc[-1]) == 8


def test_resolve_previous_cycle_root_prefers_prior_cycle(tmp_path: Path, module_loader, repo_file_fn) -> None:
    module = _mod(module_loader, repo_file_fn)
    output_root = tmp_path / "runs"
    (output_root / "run_a").mkdir(parents=True)
    (output_root / "run_a" / "cycles" / "cycle_02").mkdir(parents=True)

    previous = module.resolve_previous_cycle_root(
        output_root=output_root,
        run_id="run_a",
        cycle_index=3,
        resume_from_run="",
    )
    assert previous == output_root / "run_a" / "cycles" / "cycle_02"
