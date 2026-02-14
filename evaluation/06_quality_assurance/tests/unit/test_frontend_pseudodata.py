from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_frontend_pseudodata_synthesis_writes_expected_artifacts(module_loader, repo_root, tmp_path: Path):
    module = module_loader(
        str(repo_root / "frontend/phoenix_frontend/services/pseudodata.py"),
        "frontend_pseudodata_test_module",
    )

    payload = {
        "criteria_variables": [
            {
                "var_id": "C01",
                "label": "Depressed mood",
                "criterion_path": "mood / depression",
                "polarity": "higher_is_worse",
                "measurement": {
                    "item_or_signal": "mood_intensity",
                    "response_scale_or_unit": "0-9",
                    "sampling_per_day": 4,
                },
            }
        ],
        "predictor_variables": [
            {
                "var_id": "P01",
                "label": "Sleep quality",
                "ontology_path": "BIO / sleep / quality",
                "measurement": {
                    "item_or_signal": "sleep_quality",
                    "response_scale_or_unit": "0-9",
                    "sampling_per_day": 1,
                },
                "feasibility": {
                    "data_collection_feasibility_0_1": 0.8,
                },
            }
        ],
    }

    out = module.synthesize_pseudodata(
        model_payload=payload,
        profile_id="pseudoprofile_FTC_ID901",
        output_profile_root=tmp_path,
        n_points=32,
        missing_rate=0.15,
        seed=123,
    )

    assert Path(out["wide_csv"]).exists()
    assert Path(out["long_csv"]).exists()
    assert Path(out["metadata_csv"]).exists()
    assert Path(out["summary_json"]).exists()
    assert Path(out["spec_txt"]).exists()

    wide = pd.read_csv(out["wide_csv"])
    assert "C01" in wide.columns
    assert "P01" in wide.columns
    assert len(wide) == 32

    metadata = pd.read_csv(out["metadata_csv"])
    assert set(metadata["code"].tolist()) == {"C01", "P01"}
    assert set(metadata["role"].tolist()) == {"CRITERION", "PREDICTOR"}
