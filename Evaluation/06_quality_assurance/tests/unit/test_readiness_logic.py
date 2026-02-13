from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def _mod(module_loader, repo_file_fn):
    return module_loader(
        str(
            repo_file_fn(
                "SystemComponents/Hierarchical_Updating_Algorithm/01_time_series_analysis/01_check_readiness/apply_readiness_check.py"
            )
        ),
        "phoenix_readiness_module",
    )


def test_readiness_label_cap_for_static_tier3(module_loader, repo_file_fn) -> None:
    module = _mod(module_loader, repo_file_fn)
    label, note = module._cap_readiness_label_for_analysis(
        proposed_label="FullyReady",
        recommended_tier="Tier3_LaggedDynamicNetwork",
        tier3_variant="STATIC_gVAR",
    )
    assert label == "Ready_High"
    assert note is not None


def test_readiness_label_kept_for_tv_tier3(module_loader, repo_file_fn) -> None:
    module = _mod(module_loader, repo_file_fn)
    label, note = module._cap_readiness_label_for_analysis(
        proposed_label="FullyReady",
        recommended_tier="Tier3_LaggedDynamicNetwork",
        tier3_variant="TIME_VARYING_gVAR",
    )
    assert label == "FullyReady"
    assert note is None


def test_readiness_analysis_execution_plan_tier2(module_loader, repo_file_fn) -> None:
    module = _mod(module_loader, repo_file_fn)
    plan = module._analysis_execution_plan(
        recommended_tier="Tier2_ContemporaneousPartialCorrelation",
        tier3_variant=None,
    )
    assert plan["analysis_set"] == "tier2_contemporaneous"
    assert plan["run_tv_gvar"] is False
    assert plan["run_stationary_gvar"] is False
    assert plan["run_correlation_baseline"] is True
