from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def _mod(module_loader, repo_file_fn):
    return module_loader(
        str(
            repo_file_fn(
                "src/SystemComponents/Agentic_Framework/05_TranslationDigitalIntervention/01_generate_hapa_digital_intervention.py"
            )
        ),
        "phoenix_step05_intervention_module",
    )


def test_collect_predictor_candidates_merges_sources(module_loader, repo_file_fn) -> None:
    module = _mod(module_loader, repo_file_fn)
    step03_payload = {
        "recommended_targets": [
            {
                "predictor": "P01",
                "predictor_label": "Sleep quality",
                "score_0_1": 0.72,
                "mapped_leaf_path": "BIO / Sleep / Sleep_Quality",
            }
        ],
        "ranked_predictors": [
            {"predictor": "P01", "score_0_1": 0.72},
            {"predictor": "P02", "score_0_1": 0.55},
        ],
    }
    step04_payload = {
        "retained_criteria_ids": ["C01", "C02", "C03"],
        "recommended_next_observation_predictors": [
            "BIO / Sleep / Sleep_Quality / Daily_Rating",
            "PSYCHO / Activation / Behavioral_Activation / Activity_Scheduling",
        ],
    }
    fusion_payload = {
        "predictor_rankings": [
            {
                "predictor_path": "BIO / Sleep / Sleep_Quality / Daily_Rating",
                "fused_score_0_1": 0.81,
                "prior_score_0_1": 0.45,
            }
        ]
    }
    evidence_bundle = {
        "initial_model": {
            "predictor_summary": [
                {"var_id": "P01", "label": "Sleep quality", "mapped_leaf_full_path": "BIO / Sleep / Sleep_Quality"},
                {"var_id": "P02", "label": "Activation", "mapped_leaf_full_path": "PSYCHO / Activation / Behavioral_Activation"},
            ]
        }
    }
    rows = module.collect_predictor_candidates(
        profile_id="pseudoprofile_FTC_ID999",
        step03_payload=step03_payload,
        step04_payload=step04_payload,
        fusion_payload=fusion_payload,
        evidence_bundle=evidence_bundle,
    )
    assert len(rows) >= 2
    assert rows[0]["priority_0_1"] >= rows[-1]["priority_0_1"]
    assert any("BIO / Sleep" in str(row.get("predictor_path")) for row in rows)


def test_heuristic_step05_includes_llm_limitation(module_loader, repo_file_fn) -> None:
    module = _mod(module_loader, repo_file_fn)
    out = module.heuristic_step05_intervention(
        profile_id="pseudoprofile_FTC_ID111",
        free_text={
            "complaint_text": "Low mood and fatigue",
            "person_text": "Busy student with limited evenings",
            "context_text": "Noisy dorm and irregular schedule",
        },
        criteria_ids=["C01", "C02", "C03"],
        predictor_candidates=[
            {
                "predictor": "P01",
                "predictor_label": "Sleep regularity",
                "predictor_path": "BIO / Sleep / Regularity",
                "priority_0_1": 0.75,
            }
        ],
        selected_barriers=[
            {
                "barrier_name": "Planning",
                "barrier_path": "BARRIERS / Volition / Planning",
                "barrier_parent_domain": "BARRIERS / Volition",
                "total_score_0_1": 0.66,
            }
        ],
        coping_candidates=[
            {
                "coping_name": "Action_Planning",
                "coping_path": "COPING_STRATEGIES / Planning_and_Implementation / Action_Planning",
                "score_0_1": 0.71,
                "linked_barriers": ["Planning"],
            }
        ],
    )
    assert out.profile_id == "pseudoprofile_FTC_ID111"
    assert len(out.hapa_component_plan) >= 4
    assert module.LIMITATION_LLM_UNAVAILABLE in out.limitations


def test_step05_hard_ontology_constraint_filters_paths(module_loader, repo_file_fn) -> None:
    module = _mod(module_loader, repo_file_fn)
    out = module.heuristic_step05_intervention(
        profile_id="pseudoprofile_FTC_ID222",
        free_text={
            "complaint_text": "Stress and poor sleep",
            "person_text": "Working parent",
            "context_text": "Noisy evenings",
        },
        criteria_ids=["C01", "C02", "C03", "C04"],
        predictor_candidates=[
            {
                "predictor": "P01",
                "predictor_label": "Sleep regularity",
                "predictor_path": "BIO / Sleep / Sleep_Regularity / Fixed_Bedtime",
                "priority_0_1": 0.77,
            },
            {
                "predictor": "P02",
                "predictor_label": "Behavioral activation",
                "predictor_path": "PSYCHO / Activation / Behavioral_Activation / Activity_Scheduling",
                "priority_0_1": 0.71,
            },
        ],
        selected_barriers=[
            {
                "barrier_name": "Planning",
                "barrier_path": "BARRIERS / Volition / Planning",
                "barrier_parent_domain": "BARRIERS / Volition",
                "total_score_0_1": 0.63,
            }
        ],
        coping_candidates=[
            {
                "coping_name": "Action_Planning",
                "coping_path": "COPING_STRATEGIES / Planning_and_Implementation / Action_Planning",
                "score_0_1": 0.70,
                "linked_barriers": ["Planning"],
            }
        ],
    )
    report = module._enforce_step05_hard_ontology(
        out,
        allowed_predictor_paths=["BIO / Sleep / Sleep_Regularity / Fixed_Bedtime"],
        allowed_barrier_paths=["BARRIERS / Volition / Planning"],
        allowed_coping_paths=["COPING_STRATEGIES / Planning_and_Implementation / Action_Planning"],
    )
    assert report["dropped_targets"] >= 0
    assert out.selected_treatment_targets
    assert out.selected_treatment_targets[0].predictor_path == "BIO / Sleep / Sleep_Regularity / Fixed_Bedtime"
