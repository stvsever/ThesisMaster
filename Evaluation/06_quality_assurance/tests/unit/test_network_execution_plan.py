from __future__ import annotations

import numpy as np
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


def test_multicollinearity_report_handles_single_variable(module_loader, repo_file_fn) -> None:
    module = _mod(module_loader, repo_file_fn)
    X = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=float)
    report = module.multicollinearity_report(X, labels=["P01"], ridge=1e-3, corr_thresholds=(0.8, 0.9))
    assert report["enabled"] is True
    assert report["p"] == 1
    assert report["vif"]["P01"] == pytest.approx(1.0, abs=1e-9)
