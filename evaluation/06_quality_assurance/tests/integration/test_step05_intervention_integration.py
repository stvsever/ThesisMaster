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


def test_step05_intervention_generates_outputs(tmp_path: Path, repo_file_fn) -> None:
    script = repo_file_fn(
        "src/SystemComponents/Agentic_Framework/05_TranslationDigitalIntervention/01_generate_hapa_digital_intervention.py"
    )
    profile_id = "pseudoprofile_FTC_ID019"

    handoff_root = tmp_path / "handoff"
    profile_handoff = handoff_root / profile_id
    profile_handoff.mkdir(parents=True, exist_ok=True)

    _write_json(
        profile_handoff / "step03_target_selection.json",
        {
            "profile_id": profile_id,
            "recommended_targets": [
                {
                    "predictor": "P01",
                    "predictor_label": "Behavioral activation",
                    "score_0_1": 0.82,
                    "mapped_leaf_path": "PSYCHO / Activation / Behavioral_Activation",
                }
            ],
            "ranked_predictors": [
                {"predictor": "P01", "score_0_1": 0.82},
                {"predictor": "P02", "score_0_1": 0.61},
            ],
        },
    )
    _write_json(
        profile_handoff / "step04_updated_observation_model.json",
        {
            "profile_id": profile_id,
            "retained_criteria_ids": ["C01", "C02", "C03", "C04"],
            "recommended_next_observation_predictors": [
                "PSYCHO / Activation / Behavioral_Activation / Activity_Scheduling",
                "BIO / Sleep / Sleep_Quality / Daily_Rating",
            ],
        },
    )
    _write_json(
        profile_handoff / "step04_nomothetic_idiographic_fusion.json",
        {
            "predictor_rankings": [
                {
                    "predictor_path": "PSYCHO / Activation / Behavioral_Activation / Activity_Scheduling",
                    "fused_score_0_1": 0.76,
                    "prior_score_0_1": 0.53,
                },
                {
                    "predictor_path": "BIO / Sleep / Sleep_Quality / Daily_Rating",
                    "fused_score_0_1": 0.69,
                    "prior_score_0_1": 0.47,
                },
            ]
        },
    )
    _write_json(
        profile_handoff / "step03_evidence_bundle.json",
        {
            "free_text": {
                "complaint_text": "Low mood, fatigue, and poor focus",
                "person_text": "Works evening shifts and has limited routines",
                "context_text": "Noisy apartment and irregular timing",
            },
            "initial_model": {
                "criteria_summary": [
                    {"var_id": "C01", "label": "Low mood"},
                    {"var_id": "C02", "label": "Anhedonia"},
                    {"var_id": "C03", "label": "Fatigue"},
                ],
                "predictor_summary": [
                    {
                        "var_id": "P01",
                        "label": "Behavioral activation",
                        "mapped_leaf_full_path": "PSYCHO / Activation / Behavioral_Activation",
                    },
                    {"var_id": "P02", "label": "Sleep quality", "mapped_leaf_full_path": "BIO / Sleep / Sleep_Quality"},
                ],
            },
        },
    )

    readiness_root = tmp_path / "readiness"
    _write_json(
        readiness_root / profile_id / "readiness_report.json",
        {
            "overall": {
                "readiness_label": "Ready_High",
                "readiness_score_0_100": 86.7,
                "recommended_tier": "Tier3_LaggedDynamicNetwork",
                "tier3_variant": "STATIC_gVAR",
            }
        },
    )

    network_root = tmp_path / "network"
    _write_json(
        network_root / profile_id / "comparison_summary.json",
        {
            "profile": profile_id,
            "n_rows": 180,
            "n_vars": 8,
            "predictors": ["P01", "P02", "P03"],
            "criteria": ["C01", "C02", "C03", "C04"],
            "execution_plan": {"analysis_set": "tier3_static_lagged"},
            "method_status": {"tv": "skipped", "stationary": "executed", "corr": "executed"},
        },
    )
    metrics_dir = network_root / profile_id / "network_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "predictor_importance_tv.csv").write_text(
        "predictor,out_strength_criteria_mean\nP01,0.61\nP02,0.44\n",
        encoding="utf-8",
    )

    impact_root = tmp_path / "impact"
    impact_profile = impact_root / profile_id
    impact_profile.mkdir(parents=True, exist_ok=True)
    (impact_profile / "predictor_composite.csv").write_text(
        "predictor,predictor_label,predictor_impact\nP01,Behavioral activation,0.82\nP02,Sleep quality,0.61\n",
        encoding="utf-8",
    )
    (impact_profile / "edge_composite.csv").write_text(
        "predictor,criterion,edge_impact\nP01,C01,0.42\nP02,C03,0.37\n",
        encoding="utf-8",
    )

    free_text_root = tmp_path / "free_text"
    free_text_root.mkdir(parents=True, exist_ok=True)
    (free_text_root / "free_text_complaints.txt").write_text(
        "pseudoprofile_FTC_ID019\nLow mood and fatigue with poor concentration.\n",
        encoding="utf-8",
    )
    (free_text_root / "free_text_person.txt").write_text(
        "pseudoprofile_person_ID019\nShift worker with high workload.\n",
        encoding="utf-8",
    )
    (free_text_root / "free_text_context.txt").write_text(
        "pseudoprofile_context_ID019\nNoisy home and irregular evenings.\n",
        encoding="utf-8",
    )

    predictor_to_barrier_csv = tmp_path / "predictor_to_barrier.csv"
    predictor_to_barrier_csv.write_text(
        "predictor_id,predictor_name,predictor_full_path,barrier_id,barrier_name,barrier_full_path,score\n"
        "0,Behavioral_Activation,[PSYCHO] > Activation > Behavioral_Activation,7,Planning,[BARRIERS] > Volition > Planning,900\n"
        "1,Sleep_Quality,[BIO] > Sleep > Sleep_Quality,8,Time_And_Organization,[BARRIERS] > Volition > Time_And_Organization,850\n",
        encoding="utf-8",
    )
    profile_to_barrier_csv = tmp_path / "profile_to_barrier.csv"
    profile_to_barrier_csv.write_text(
        "barrier_id,barrier_name,barrier_full_path,profile_id,profile_name,profile_full_path,score\n"
        "7,Planning,[BARRIERS] > Volition > Planning,0,Work_Schedule,[PERSON] > Employment_and_Work > Work_Schedule_Type,900\n",
        encoding="utf-8",
    )
    context_to_barrier_csv = tmp_path / "context_to_barrier.csv"
    context_to_barrier_csv.write_text(
        "barrier_id,barrier_name,barrier_full_path,context_id,context_name,context_full_path,score\n"
        "8,Time_And_Organization,[BARRIERS] > Volition > Time_And_Organization,0,Daily_Structure,[CONTEXT] > Time > Daily_Structure,920\n",
        encoding="utf-8",
    )
    coping_to_barrier_csv = tmp_path / "coping_to_barrier.csv"
    coping_to_barrier_csv.write_text(
        "coping_id,coping_name,coping_full_path,barrier_id,barrier_name,barrier_full_path,score\n"
        "6,Action_Planning,[COPING_STRATEGIES] > Planning_and_Implementation > Action_Planning,7,Planning,[BARRIERS] > Volition > Planning,920\n"
        "7,Coping_Planning,[COPING_STRATEGIES] > Planning_and_Implementation > Coping_Planning,8,Time_And_Organization,[BARRIERS] > Volition > Time_And_Organization,900\n",
        encoding="utf-8",
    )

    output_root = tmp_path / "step05_out"
    cmd = [
        sys.executable,
        str(script),
        "--handoff-root",
        str(handoff_root),
        "--output-root",
        str(output_root),
        "--readiness-root",
        str(readiness_root),
        "--network-root",
        str(network_root),
        "--impact-root",
        str(impact_root),
        "--free-text-root",
        str(free_text_root),
        "--predictor-to-barrier-csv",
        str(predictor_to_barrier_csv),
        "--profile-to-barrier-csv",
        str(profile_to_barrier_csv),
        "--context-to-barrier-csv",
        str(context_to_barrier_csv),
        "--coping-to-barrier-csv",
        str(coping_to_barrier_csv),
        "--pattern",
        profile_id,
        "--max-profiles",
        "1",
        "--disable-llm",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr

    profile_out = output_root / profile_id
    assert (profile_out / "step05_hapa_intervention.json").exists()
    assert (profile_out / "step05_selected_barriers_top10.csv").exists()
    assert (profile_out / "step05_coping_candidates_ranked.csv").exists()
    payload = json.loads((profile_out / "step05_hapa_intervention.json").read_text(encoding="utf-8"))
    assert payload["profile_id"] == profile_id
    assert len(payload["selected_barriers"]) >= 1
    assert len(payload["selected_coping_strategies"]) >= 1
    assert "Limitation recorded: LLM unavailable, so it’s impact-driven only." in payload["limitations"]
