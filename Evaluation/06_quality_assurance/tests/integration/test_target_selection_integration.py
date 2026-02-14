from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_target_selection_generates_step03_and_step04_outputs(tmp_path: Path, repo_file_fn) -> None:
    script = repo_file_fn(
        "src/SystemComponents/Agentic_Framework/03_TreatmentTargetIdentification/01_prepare_targets_from_impact.py"
    )
    profile_id = "pseudoprofile_FTC_ID002"

    impact_root = tmp_path / "impact"
    profile_impact = impact_root / profile_id
    profile_impact.mkdir(parents=True)
    (profile_impact / "predictor_composite.csv").write_text(
        "predictor,predictor_label,predictor_impact,predictor_impact_pct,predictor_rank\n"
        "P01,Sleep duration,0.78,78,1\n"
        "P02,Activation,0.54,54,2\n"
        "P03,Rumination,0.33,33,3\n",
        encoding="utf-8",
    )
    (profile_impact / "edge_composite.csv").write_text(
        "predictor,criterion,edge_impact\nP01,C01,0.42\nP02,C02,0.31\n",
        encoding="utf-8",
    )
    (profile_impact / "impact_matrix.csv").write_text(
        ",P01,P02\nC01,0.33,0.21\nC02,0.17,0.29\n",
        encoding="utf-8",
    )

    readiness_root = tmp_path / "readiness"
    _write_json(
        readiness_root / profile_id / "readiness_report.json",
        {
            "overall": {
                "readiness_label": "Ready_High",
                "readiness_score_0_100": 87.4,
                "recommended_tier": "Tier3_LaggedDynamicNetwork",
                "tier3_variant": "STATIC_gVAR",
                "ready_variables": ["P01", "P02", "C01", "C02"],
                "why": ["Sufficient effective sample size", "Regular time index"],
                "technical_summary": "Ready for static Tier-3 lagged analysis.",
                "client_friendly_summary": "Dataset quality is strong enough for lagged modeling.",
                "next_steps": ["Maintain data completeness"],
                "caveats": ["Track missingness drift"],
                "score_breakdown": {
                    "components": {
                        "missing_score": 82.0,
                        "time_score": 95.0,
                        "sample_score": 88.0,
                        "variable_quality_score": 84.0,
                    }
                },
            },
            "tiers": {
                "tier3": {
                    "variant_time_varying": {
                        "feasible": False,
                        "tv_required_points_heuristic": 240,
                        "n_eff_lagged": {"q25_per_variable": 110},
                    },
                    "variant_static": {
                        "feasible": True,
                        "required_n_eff_heuristic": 90,
                        "n_eff_lagged": {"q25_per_variable": 108},
                    },
                }
            },
        },
    )

    network_root = tmp_path / "network"
    metrics = network_root / profile_id / "network_metrics"
    metrics.mkdir(parents=True)
    (metrics / "predictor_importance_tv.csv").write_text(
        "predictor,out_strength_criteria_mean,delta_mse_criteria,nonzero_fraction_mean\n"
        "P01,0.61,0.03,0.40\nP02,0.43,0.02,0.33\n",
        encoding="utf-8",
    )
    (metrics / "criterion_dependence_tv.csv").write_text(
        "criterion,incoming_from_predictors_mean\nC01,0.40\nC02,0.35\n",
        encoding="utf-8",
    )

    model_runs_root = tmp_path / "model_runs"
    _write_json(
        model_runs_root / "2026-02-13_18-00-00" / "profiles" / profile_id / "llm_observation_model_mapped.json",
        {
            "criteria_variables": [
                {"var_id": "C01", "label": "Low mood", "mapped_leaf_full_path": "DSM / Mood"},
                {"var_id": "C02", "label": "Anhedonia", "mapped_leaf_full_path": "DSM / Interest"},
            ],
            "predictor_variables": [
                {
                    "var_id": "P01",
                    "label": "Sleep duration",
                    "mapped_leaf_full_path": "BIO / Sleep / Sleep_Duration",
                    "include_priority": "HIGH",
                    "mapped_confidence": 0.81,
                },
                {
                    "var_id": "P02",
                    "label": "Behavioral activation",
                    "mapped_leaf_full_path": "PSYCHO / Activation / Behavioral_Activation",
                    "include_priority": "HIGH",
                    "mapped_confidence": 0.75,
                },
            ],
            "edges": [
                {"source_var_id": "P01", "target_var_id": "C01", "estimated_relevance_0_1": 0.62},
                {"source_var_id": "P02", "target_var_id": "C02", "estimated_relevance_0_1": 0.55},
            ],
        },
    )

    free_text_root = tmp_path / "free_text"
    free_text_root.mkdir(parents=True)
    (free_text_root / "free_text_complaints.txt").write_text(
        "pseudoprofile_FTC_ID002\nI feel flat and exhausted.\n",
        encoding="utf-8",
    )
    (free_text_root / "free_text_person.txt").write_text(
        "pseudoprofile_person_ID002\nWorks full-time and has limited evening energy.\n",
        encoding="utf-8",
    )
    (free_text_root / "free_text_context.txt").write_text(
        "pseudoprofile_context_ID002\nLate-evening workload and noisy home context.\n",
        encoding="utf-8",
    )

    predictor_list = tmp_path / "predictors_list.txt"
    predictor_list.write_text(
        "[BIO]\n"
        "└─ Sleep\n"
        "  └─ Sleep_Duration (ID:11)\n"
        "[PSYCHO]\n"
        "└─ Activation\n"
        "  └─ Behavioral_Activation (ID:212)\n"
        "  └─ Rumination_Management (ID:213)\n",
        encoding="utf-8",
    )
    predictor_leaf_paths = tmp_path / "predictor_leaf_paths.json"
    predictor_leaf_paths.write_text(
        json.dumps(
            [
                "BIO / Sleep / Sleep_Duration / Consistent_Bedtime",
                "BIO / Sleep / Sleep_Duration / Sleep_Window",
                "PSYCHO / Activation / Behavioral_Activation / Activity_Scheduling",
                "PSYCHO / Activation / Behavioral_Activation / Values_Action",
            ]
        ),
        encoding="utf-8",
    )
    mapping_ranks = tmp_path / "mapping_ranks.csv"
    mapping_ranks.write_text(
        "pseudoprofile_id,part,criterion_path,criterion_leaf,rank,predictor_path,relevance_score\n"
        "pseudoprofile_FTC_ID002,post_per_criterion,mood / low_mood,low_mood,1,BIO / Sleep / Sleep_Duration,870\n"
        "pseudoprofile_FTC_ID002,post_per_criterion,mood / anhedonia,anhedonia,1,PSYCHO / Activation / Behavioral_Activation,840\n"
        "pseudoprofile_FTC_ID002,post_global,,,1,PSYCHO / Activation / Behavioral_Activation,800\n",
        encoding="utf-8",
    )

    output_root = tmp_path / "handoff_out"
    cmd = [
        sys.executable,
        str(script),
        "--impact-root",
        str(impact_root),
        "--output-root",
        str(output_root),
        "--readiness-root",
        str(readiness_root),
        "--network-root",
        str(network_root),
        "--initial-model-runs-root",
        str(model_runs_root),
        "--free-text-root",
        str(free_text_root),
        "--mapping-ranks-csv",
        str(mapping_ranks),
        "--predictor-leaf-paths-json",
        str(predictor_leaf_paths),
        "--predictor-list-path",
        str(predictor_list),
        "--pattern",
        profile_id,
        "--max-profiles",
        "1",
        "--disable-llm",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr

    profile_out = output_root / profile_id
    assert (profile_out / "top_treatment_target_candidates.csv").exists()
    assert (profile_out / "step03_target_selection.json").exists()
    assert (profile_out / "step04_updated_observation_model.json").exists()
    assert (profile_out / "ontology_subtree_candidates_top200.csv").exists()
    assert (profile_out / "step04_nomothetic_idiographic_fusion.json").exists()
    assert (profile_out / "step04_fusion_edges.csv").exists()
    assert (profile_out / "step04_fusion_predictor_rankings.csv").exists()
    assert (profile_out / "visuals" / "updated_model_fused_heatmap.png").exists()
    step04 = json.loads((profile_out / "step04_updated_observation_model.json").read_text(encoding="utf-8"))
    assert step04["profile_id"] == profile_id
    assert len(step04["retained_criteria_ids"]) >= 1
