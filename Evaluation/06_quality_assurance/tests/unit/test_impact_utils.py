from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def _mod(module_loader, repo_file_fn):
    return module_loader(
        str(
            repo_file_fn(
                "SystemComponents/Hierarchical_Updating_Algorithm/02_hierarchical_update_ranking/01_momentary_impact_quantification/01_compute_momentary_impact_coefficients.py"
            )
        ),
        "phoenix_impact_module",
    )


def test_robust_minmax_handles_all_nan(module_loader, repo_file_fn) -> None:
    module = _mod(module_loader, repo_file_fn)
    out = module.robust_minmax_01(np.array([np.nan, np.nan, np.nan], dtype=float))
    assert np.allclose(out, np.array([0.5, 0.5, 0.5], dtype=float))


def test_exponential_recency_weights_are_normalized(module_loader, repo_file_fn) -> None:
    module = _mod(module_loader, repo_file_fn)
    weights = module.exponential_recency_weights(np.array([0.0, 0.5, 1.0], dtype=float), half_life=0.20)
    assert np.isclose(np.sum(weights), 1.0)
    assert weights[-1] > weights[0]
