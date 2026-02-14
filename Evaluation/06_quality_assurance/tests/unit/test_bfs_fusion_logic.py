from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.unit


def _mod(module_loader, repo_file_fn):
    return module_loader(
        str(repo_file_fn("src/SystemComponents/Agentic_Framework/shared/target_refinement.py")),
        "phoenix_target_refinement_module",
    )


def test_bfs_candidates_cover_domains_before_depth(tmp_path: Path, module_loader, repo_file_fn) -> None:
    module = _mod(module_loader, repo_file_fn)
    paths_json = tmp_path / "predictor_paths.json"
    paths_json.write_text(
        """[
  "BIO / Sleep / Sleep_Routines / Consistent_Bedtime",
  "BIO / Sleep / Sleep_Routines / Sleep_Window",
  "PSYCHO / Activation / Behavioral_Activation / Values_Action",
  "PSYCHO / Activation / Behavioral_Activation / Activity_Scheduling",
  "SOCIAL / Support / Family / Checkins"
]""",
        encoding="utf-8",
    )
    leaf_paths = module.load_predictor_leaf_paths(paths_json)
    mapping_rows = [
        {
            "part": "post_per_criterion",
            "criterion_path": "mood / low_mood",
            "predictor_path": "BIO / Sleep / Sleep_Routines",
            "relevance_score_0_1": 0.90,
        },
        {
            "part": "post_per_criterion",
            "criterion_path": "mood / low_mood",
            "predictor_path": "PSYCHO / Activation / Behavioral_Activation",
            "relevance_score_0_1": 0.85,
        },
    ]
    candidates = module.build_bfs_candidates(
        leaf_paths=leaf_paths,
        mapping_rows=mapping_rows,
        hyde_scores={},
        mapped_predictor_paths=["BIO / Sleep / Sleep_Routines / Consistent_Bedtime"],
        impact_by_predictor={"P01": 0.72},
        predictor_var_to_path={"P01": "BIO / Sleep / Sleep_Routines / Consistent_Bedtime"},
        max_candidates=5,
    )
    assert len(candidates) >= 3
    first_two = candidates[:2]
    assert first_two[0]["bfs_stage"] == "breadth_domain_coverage"
    assert first_two[1]["bfs_stage"] == "breadth_domain_coverage"
    assert first_two[0]["bfs_domain_key"] != first_two[1]["bfs_domain_key"]
    stages = [row["bfs_stage"] for row in candidates]
    if "depth_refinement" in stages:
        assert stages.index("depth_refinement") >= 2


def test_nomothetic_idiographic_fusion_builds_matrix(module_loader, repo_file_fn) -> None:
    module = _mod(module_loader, repo_file_fn)
    criteria = [
        {"var_id": "C01", "label": "Low mood", "mapped_leaf_full_path": "CLINICAL / Mood / low_mood"},
        {"var_id": "C02", "label": "Anhedonia", "mapped_leaf_full_path": "CLINICAL / Mood / anhedonia"},
    ]
    predictors = [
        {"var_id": "P01", "label": "Sleep duration", "mapped_leaf_full_path": "BIO / Sleep / Duration"},
        {"var_id": "P02", "label": "Activation", "mapped_leaf_full_path": "PSYCHO / Activation / Behavioral_Activation"},
    ]
    initial_model_payload = {
        "predictor_criterion_relevance": [
            {"predictor_var_id": "P01", "criterion_var_id": "C01", "relevance_score_0_1_comma5": "0,72000"},
            {"predictor_var_id": "P01", "criterion_var_id": "C02", "relevance_score_0_1_comma5": "0,54000"},
            {"predictor_var_id": "P02", "criterion_var_id": "C01", "relevance_score_0_1_comma5": "0,51000"},
            {"predictor_var_id": "P02", "criterion_var_id": "C02", "relevance_score_0_1_comma5": "0,76000"},
        ]
    }
    impact_matrix = pd.DataFrame(
        [[0.35, 0.22], [0.18, 0.41]],
        index=["C01", "C02"],
        columns=["P01", "P02"],
    )
    mapping_rows = [
        {
            "part": "post_per_criterion",
            "criterion_path": "mood / low_mood",
            "predictor_path": "BIO / Sleep / Sleep_Routines",
            "relevance_score_0_1": 0.83,
        },
        {
            "part": "post_per_criterion",
            "criterion_path": "mood / anhedonia",
            "predictor_path": "PSYCHO / Activation / Behavioral_Activation",
            "relevance_score_0_1": 0.87,
        },
    ]
    fusion = module.fuse_updated_model_matrix(
        criteria_summary=criteria,
        predictor_summary=predictors,
        initial_model_payload=initial_model_payload,
        impact_matrix=impact_matrix,
        candidate_paths=[
            "BIO / Sleep / Sleep_Routines / Consistent_Bedtime",
            "PSYCHO / Activation / Behavioral_Activation / Activity_Scheduling",
        ],
        candidate_prior_scores={
            "BIO / Sleep / Sleep_Routines / Consistent_Bedtime": 0.62,
            "PSYCHO / Activation / Behavioral_Activation / Activity_Scheduling": 0.58,
        },
        mapping_rows=mapping_rows,
        readiness_score_0_100=84.0,
        max_predictors=10,
    )
    assert "weights" in fusion
    assert 0.0 <= fusion["weights"]["nomothetic_weight"] <= 1.0
    assert 0.0 <= fusion["weights"]["idiographic_weight"] <= 1.0
    assert len(fusion["predictor_rankings"]) == 2
    assert len(fusion["edge_rows"]) == 4
    for row in fusion["edge_rows"]:
        assert 0.0 <= float(row["fused_score_0_1"]) <= 1.0


def test_fusion_supports_previous_cycle_scores(module_loader, repo_file_fn) -> None:
    module = _mod(module_loader, repo_file_fn)
    criteria = [{"var_id": "C01", "label": "Low mood", "mapped_leaf_full_path": "CLINICAL / Mood / low_mood"}]
    predictors = [{"var_id": "P01", "label": "Sleep duration", "mapped_leaf_full_path": "BIO / Sleep / Duration"}]
    initial_model_payload = {
        "predictor_criterion_relevance": [
            {"predictor_var_id": "P01", "criterion_var_id": "C01", "relevance_score_0_1_comma5": "0,70000"}
        ]
    }
    impact_matrix = pd.DataFrame([[0.34]], index=["C01"], columns=["P01"])
    candidate_path = "BIO / Sleep / Sleep_Routines / Consistent_Bedtime"
    fusion = module.fuse_updated_model_matrix(
        criteria_summary=criteria,
        predictor_summary=predictors,
        initial_model_payload=initial_model_payload,
        impact_matrix=impact_matrix,
        candidate_paths=[candidate_path],
        candidate_prior_scores={candidate_path: 0.62},
        mapping_rows=[],
        readiness_score_0_100=90.0,
        previous_cycle_scores={candidate_path.lower(): 0.80},
        max_predictors=5,
    )
    assert fusion["predictor_rankings"]
    row = fusion["predictor_rankings"][0]
    assert "fused_memory_score_0_1" in row
    assert "previous_cycle_impact_0_1" in row
    assert 0.0 <= float(row["fused_memory_score_0_1"]) <= 1.0
