from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_contract_validator_accepts_backward_compatible_payload(module_loader, repo_file_fn) -> None:
    module = module_loader(
        str(repo_file_fn("src/utils/agentic_core/shared/contracts/registry.py")),
        "phoenix_contract_registry",
    )
    validator = module.ContractValidator()
    payload = {"overall": {"readiness_label": "READY_HIGH"}}
    result = validator.validate_payload(contract_name="readiness_report", payload=payload)
    assert result.success is True
    assert result.contract_version == "1.0.0"


def test_prompt_manifest_contains_step_prompts(module_loader, repo_file_fn) -> None:
    module = module_loader(
        str(repo_file_fn("src/utils/agentic_core/shared/prompt_loader.py")),
        "phoenix_prompt_loader_module",
    )
    manifest = module.load_prompts_manifest()
    prompts = manifest.get("prompts", {})
    assert "step02_initial_model_system.md" in prompts
    assert "step02_initial_model_critic_system.md" in prompts
    assert "step03_target_selection_system.md" in prompts
    assert "step03_target_selection_critic_system.md" in prompts
    assert "step04_observation_update_system.md" in prompts
    assert "step04_observation_update_critic_system.md" in prompts
    assert "step05_hapa_intervention_system.md" in prompts
    assert "step05_hapa_intervention_critic_system.md" in prompts


def test_llm_runtime_uses_provider_unavailable_taxonomy(module_loader, repo_file_fn, monkeypatch) -> None:
    module = module_loader(
        str(repo_file_fn("src/utils/agentic_core/shared/llm_runtime.py")),
        "phoenix_llm_runtime_module",
    )

    class DemoSchema(module.BaseModel):
        value: int

    client = module.StructuredLLMClient(max_attempts=2)

    def _raise(*args, **kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(client, "_responses_api_call", _raise)
    result = client.generate_structured(
        system_prompt="sys",
        user_prompt="usr",
        schema_model=DemoSchema,
    )
    assert result.success is False
    assert result.failure_reason == "provider_unavailable"


def test_guardrail_weighted_composite_scoring(module_loader, repo_file_fn) -> None:
    module = module_loader(
        str(repo_file_fn("src/utils/agentic_core/shared/__init__.py")),
        "phoenix_shared_runtime_module",
    )
    scored = module.weighted_composite(
        subscores={"reasoning": 0.8, "grounding": 0.6},
        weights={"reasoning": 0.4, "grounding": 0.6},
    )
    assert 0.0 <= scored["composite_score_0_1"] <= 1.0
    assert scored["normalized_weights"]["grounding"] == pytest.approx(0.6, abs=1e-8)
    decision = module.decision_from_score(score_0_1=scored["composite_score_0_1"], threshold_0_1=0.65)
    assert decision in {"PASS", "REVISE"}
