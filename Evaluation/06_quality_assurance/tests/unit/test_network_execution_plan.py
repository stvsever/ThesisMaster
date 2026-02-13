from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def _mod(module_loader, repo_file_fn):
    return module_loader(
        str(
            repo_file_fn(
                "SystemComponents/Hierarchical_Updating_Algorithm/01_time_series_analysis/02_network_time_series_analysis/01_run_network_ts_analysis.py"
            )
        ),
        "phoenix_network_module",
    )


def test_execution_plan_respects_static_tier3(module_loader, repo_file_fn) -> None:
    module = _mod(module_loader, repo_file_fn)
    readiness = {
        "overall": {
            "recommended_tier": "Tier3_LaggedDynamicNetwork",
            "tier3_variant": "STATIC_gVAR",
        }
    }
    plan = module.build_execution_plan_from_readiness(readiness, execution_policy="readiness_aligned")
    assert plan["analysis_set"] == "tier3_static_lagged"
    assert plan["run_tv_gvar"] is False
    assert plan["run_stationary_gvar"] is True
    assert plan["run_correlation_baseline"] is True


def test_execution_plan_all_methods_override(module_loader, repo_file_fn) -> None:
    module = _mod(module_loader, repo_file_fn)
    readiness = {"overall": {"recommended_tier": "Tier0_DescriptivesOnly", "tier3_variant": None}}
    plan = module.build_execution_plan_from_readiness(readiness, execution_policy="all_methods")
    assert plan["analysis_set"] == "all_methods_for_diagnostics"
    assert plan["run_tv_gvar"] is True
    assert plan["run_stationary_gvar"] is True
    assert plan["run_correlation_baseline"] is True
