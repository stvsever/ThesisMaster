from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.unit


def _mod(module_loader, repo_file_fn):
    return module_loader(
        str(
            repo_file_fn(
                "src/SystemComponents/Agentic_Framework/03_TreatmentTargetIdentification/01_prepare_targets_from_impact.py"
            )
        ),
        "phoenix_target_selection_module",
    )


def test_heuristic_selection_limits_targets(module_loader, repo_file_fn) -> None:
    module = _mod(module_loader, repo_file_fn)
    impact_df = pd.DataFrame(
        [
            {"predictor": "P01", "predictor_label": "A", "predictor_impact": 0.82},
            {"predictor": "P02", "predictor_label": "B", "predictor_impact": 0.65},
            {"predictor": "P03", "predictor_label": "C", "predictor_impact": 0.31},
            {"predictor": "P04", "predictor_label": "D", "predictor_impact": 0.05},
        ]
    )
    out = module.heuristic_step03_selection(
        profile_id="pseudoprofile_FTC_ID999",
        impact_df=impact_df,
        mapped_predictor_paths={"P01": "BIO / Sleep / Duration"},
        top_k=10,
        min_impact=0.10,
    )
    assert out.profile_id == "pseudoprofile_FTC_ID999"
    assert len(out.ranked_predictors) == 4
    assert len(out.recommended_targets) <= 3
    assert out.recommended_targets[0].predictor == "P01"


def test_parse_predictor_catalog_extracts_leaf_paths(tmp_path: Path, module_loader, repo_file_fn) -> None:
    module = _mod(module_loader, repo_file_fn)
    predictor_txt = tmp_path / "predictors_list.txt"
    predictor_txt.write_text(
        "[BIO]\n"
        "└─ Sleep\n"
        "  └─ Sleep_Duration (ID:11)\n"
        "  └─ Sleep_Regularity (ID:12)\n"
        "[PSYCHO]\n"
        "└─ Emotion_Regulation\n"
        "  └─ Cognitive_Reappraisal (ID:201)\n",
        encoding="utf-8",
    )
    catalog = module.parse_predictor_catalog(predictor_txt)
    assert len(catalog) == 3
    assert catalog[0].full_path.startswith("BIO / Sleep /")
    assert catalog[2].root_domain == "PSYCHO"


def test_hard_ontology_enforcement_remaps_step03_paths(module_loader, repo_file_fn) -> None:
    module = _mod(module_loader, repo_file_fn)
    impact_df = pd.DataFrame(
        [
            {"predictor": "P01", "predictor_label": "Sleep quality", "predictor_impact": 0.72},
            {"predictor": "P02", "predictor_label": "Activation", "predictor_impact": 0.66},
        ]
    )
    out = module.heuristic_step03_selection(
        profile_id="pseudoprofile_FTC_ID001",
        impact_df=impact_df,
        mapped_predictor_paths={
            "P01": "BIO / Sleep / Sleep_Quality",
            "P02": "PSYCHO / Activation / Behavioral_Activation",
        },
        top_k=10,
        min_impact=0.10,
    )
    report = module._enforce_step03_hard_ontology(
        out,
        allowed_predictor_paths=[
            "BIO / Sleep / Sleep_Quality / Daily_Rating",
            "PSYCHO / Activation / Behavioral_Activation / Activity_Scheduling",
        ],
        predictor_var_to_path={
            "P01": "BIO / Sleep / Sleep_Quality",
            "P02": "PSYCHO / Activation / Behavioral_Activation",
        },
    )
    assert report["applied"] is True
    assert out.ranked_predictors
    assert all("/" in item.mapped_leaf_path for item in out.ranked_predictors if item.mapped_leaf_path)
