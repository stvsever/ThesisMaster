from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def _mod(module_loader, repo_file_fn):
    return module_loader(
        str(repo_file_fn("SystemComponents/Agentic_Framework/03_TreatmentTargetIdentification/01_prepare_targets_from_impact.py")),
        "phoenix_handoff_module",
    )


def test_priority_mapping_boundaries(module_loader, repo_file_fn) -> None:
    module = _mod(module_loader, repo_file_fn)
    assert module.priority_from_impact(0.75) == "very_high"
    assert module.priority_from_impact(0.55) == "high"
    assert module.priority_from_impact(0.35) == "medium"
    assert module.priority_from_impact(0.15) == "low"
    assert module.priority_from_impact(0.05) == "very_low"
