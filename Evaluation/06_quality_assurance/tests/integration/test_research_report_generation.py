from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_research_report_generation(tmp_path: Path, repo_file_fn) -> None:
    script = repo_file_fn("Evaluation/07_research_communication/generate_pipeline_research_report.py")
    run_root = tmp_path / "run_a"
    (run_root / "00_readiness_check" / "pseudoprofile_test001").mkdir(parents=True)
    (run_root / "01_time_series_analysis" / "network" / "pseudoprofile_test001").mkdir(parents=True)
    (run_root / "02_momentary_impact_coefficients" / "pseudoprofile_test001" / "visuals").mkdir(parents=True)
    (run_root / "03_treatment_target_handoff" / "pseudoprofile_test001").mkdir(parents=True)

    (run_root / "pipeline_summary.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "run_id": "run_a",
                "profiles": ["pseudoprofile_test001"],
                "outputs": {},
                "stage_results": [],
            }
        ),
        encoding="utf-8",
    )

    (run_root / "00_readiness_check" / "pseudoprofile_test001" / "readiness_report.json").write_text(
        json.dumps(
            {
                "overall": {
                    "readiness_label": "Ready_High",
                    "readiness_score_0_100": 88.1,
                    "recommended_tier": "Tier3_LaggedDynamicNetwork",
                    "tier3_variant": "STATIC_gVAR",
                    "analysis_execution_plan": {"analysis_set": "tier3_static_lagged"},
                }
            }
        ),
        encoding="utf-8",
    )
    (run_root / "01_time_series_analysis" / "network" / "pseudoprofile_test001" / "comparison_summary.json").write_text(
        json.dumps(
            {
                "execution_plan": {"analysis_set": "tier3_static_lagged"},
                "method_status": {"tv": "skipped", "stationary": "executed", "corr": "executed"},
            }
        ),
        encoding="utf-8",
    )
    (run_root / "02_momentary_impact_coefficients" / "pseudoprofile_test001" / "predictor_composite.csv").write_text(
        "predictor,predictor_label,predictor_impact,predictor_rank\nP01,Test predictor,0.72,1\n",
        encoding="utf-8",
    )
    (run_root / "02_momentary_impact_coefficients" / "pseudoprofile_test001" / "momentary_impact.json").write_text(
        json.dumps({"meta": {"impact_source": "stationary_gvar_fallback"}}),
        encoding="utf-8",
    )
    (run_root / "02_momentary_impact_coefficients" / "pseudoprofile_test001" / "visuals" / "heatmap_impact_matrix.png").write_bytes(
        b"png"
    )
    (run_root / "03_treatment_target_handoff" / "pseudoprofile_test001" / "top_treatment_target_candidates.csv").write_text(
        "predictor,predictor_impact\nP01,0.72\n",
        encoding="utf-8",
    )

    output_root = tmp_path / "reports"
    cmd = [
        sys.executable,
        str(script),
        "--run-root",
        str(run_root),
        "--output-root",
        str(output_root),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr

    assert (output_root / "run_report.md").exists()
    assert (output_root / "run_report.json").exists()
    report_json = json.loads((output_root / "run_report.json").read_text(encoding="utf-8"))
    assert report_json["component_status"]["n_profiles_reported"] == 1
