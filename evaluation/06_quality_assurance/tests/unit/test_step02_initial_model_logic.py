from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def _mod(module_loader, repo_file_fn):
    return module_loader(
        str(
            repo_file_fn(
                "src/SystemComponents/Agentic_Framework/02_ConstructionInitialObservationModel/utils/01_construct_observation_model.py"
            )
        ),
        "phoenix_step02_initial_model_module",
    )


def test_step02_hard_ontology_enforcement_remaps_paths(module_loader, repo_file_fn) -> None:
    module = _mod(module_loader, repo_file_fn)
    model = {
        "predictor_variables": [
            {"var_id": "P01", "ontology_path": "BIO / Sleep / Unknown_Path"},
            {"var_id": "P02", "ontology_path": "PSYCHO / Activation / Behavioral_Activation"},
        ],
        "criteria_variables": [
            {"var_id": "C01", "criterion_path": "CRITERION / Mood / Sadness"},
            {"var_id": "C02", "criterion_path": "CRITERION / Energy / Fatigue"},
        ],
    }
    report = module._enforce_step02_hard_ontology(
        model=model,
        allowed_predictor_paths=[
            "BIO / Sleep / Sleep_Regularity / Fixed_Bedtime",
            "PSYCHO / Activation / Behavioral_Activation / Activity_Scheduling",
        ],
        allowed_criterion_paths=[
            "CRITERION / Mood / Sadness",
            "CRITERION / Energy / Fatigue",
        ],
    )
    assert report["applied"] is True
    predictor_paths = [str(row.get("ontology_path", "")) for row in model["predictor_variables"]]
    assert any("Fixed_Bedtime" in path for path in predictor_paths)


def test_step02_heuristic_critic_revise_on_validation_errors(module_loader, repo_file_fn) -> None:
    module = _mod(module_loader, repo_file_fn)
    model = {
        "criteria_variables": [{"var_id": "C01"}, {"var_id": "C02"}, {"var_id": "C03"}],
        "predictor_variables": [{"var_id": "P01"}, {"var_id": "P02"}],
        "safety_notes": "",
    }
    review = module._heuristic_step02_critic(
        model=model,
        validation_report={
            "errors": ["missing_dense_grid"],
            "warnings": ["gvar caution"],
            "stats": {"gvar_T_over_K": 1.2},
        },
        predictor_parent_feasibility={
            "top_parent_domains": [
                {"mean_composite_score_0_1": 0.31},
                {"mean_composite_score_0_1": 0.28},
            ]
        },
        hard_ontology_constraint_applied=True,
        ontology_constraint_summary={"predictor_violations": ["P01"], "criterion_violations": []},
        pass_threshold=0.74,
    )
    assert review.decision == "REVISE"
    assert review.critical_issues
    assert review.actionable_feedback


def test_step02_heuristic_critic_pass_on_strong_inputs(module_loader, repo_file_fn) -> None:
    module = _mod(module_loader, repo_file_fn)
    model = {
        "criteria_variables": [{"var_id": "C01"}, {"var_id": "C02"}, {"var_id": "C03"}, {"var_id": "C04"}],
        "predictor_variables": [
            {"var_id": "P01"},
            {"var_id": "P02"},
            {"var_id": "P03"},
            {"var_id": "P04"},
            {"var_id": "P05"},
            {"var_id": "P06"},
        ],
        "safety_notes": "If risk escalates, contact professional support urgently.",
    }
    review = module._heuristic_step02_critic(
        model=model,
        validation_report={
            "errors": [],
            "warnings": [],
            "stats": {"gvar_T_over_K": 6.1},
        },
        predictor_parent_feasibility={
            "top_parent_domains": [
                {"mean_composite_score_0_1": 0.82},
                {"mean_composite_score_0_1": 0.79},
                {"mean_composite_score_0_1": 0.76},
            ]
        },
        hard_ontology_constraint_applied=False,
        ontology_constraint_summary={"predictor_violations": [], "criterion_violations": []},
        pass_threshold=0.65,
    )
    assert review.decision == "PASS"
    assert review.composite_score_0_1 >= 0.65


def test_step02_derive_allowed_paths_uses_expected_fields(module_loader, repo_file_fn) -> None:
    module = _mod(module_loader, repo_file_fn)
    profile = module.ProfileInput(
        pseudoprofile_id="pseudoprofile_FTC_ID999",
        complaint_text="I feel exhausted and worried.",
        decomp_n_variables=2,
        decomp_notes="",
        criteria=[
            module.CriterionRow(
                variable_id="C01",
                variable_label="fatigue",
                variable_criterion="fatigue",
                variable_evidence="self-report",
                query_text_used="fatigue",
                criterion_path="CRITERION / Energy / Fatigue",
                criterion_leaf="Fatigue",
            )
        ],
        complaint_unique_mapped_leaf_embed_paths=[
            "BIO / Sleep / Sleep_Regularity / Fixed_Bedtime"
        ],
        high_level_predictor_ontology_raw="",
    )
    hyde = module.HydeProfileSignals(
        run_id="run_x",
        summary="",
        solutions_compact="",
        llm_model="gpt-5-nano",
        embedding_model="text-embedding-3-small",
        global_top=[
            module.HydeGlobalRankItem(
                rank=1,
                fused_score_0_1=0.91,
                predictor_path="PSYCHO / Emotion_Regulation / Cognitive_Reframing",
                primary_node="PSYCHO",
                secondary_node="Emotion_Regulation",
            )
        ],
    )
    mapping = {
        "pre_global_top": [],
        "post_global_top": [],
        "post_per_criterion_top": {
            "CRITERION / Energy / Fatigue": [
                module.MappingRankItem(
                    part="post_per_criterion",
                    criterion_path="CRITERION / Energy / Fatigue",
                    criterion_leaf="Fatigue",
                    rank=1,
                    predictor_path="BIO / Activity / Movement",
                    relevance_score=0.7,
                    primary_node="BIO",
                    secondary_node="Activity",
                )
            ]
        },
    }

    predictor_paths, criterion_paths = module._derive_allowed_paths(
        profile=profile,
        hyde=hyde,
        mapping=mapping,
    )
    assert "BIO / Sleep / Sleep_Regularity / Fixed_Bedtime" in predictor_paths
    assert "PSYCHO / Emotion_Regulation / Cognitive_Reframing" in predictor_paths
    assert "BIO / Activity / Movement" in predictor_paths
    assert "CRITERION / Energy / Fatigue" in criterion_paths
