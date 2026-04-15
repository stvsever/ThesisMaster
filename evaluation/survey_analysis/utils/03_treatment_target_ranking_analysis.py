from __future__ import annotations

from shared.comparison_study import ComparisonStudyConfig, run_comparison_study


if __name__ == "__main__":
    run_comparison_study(
        ComparisonStudyConfig(
            study_slug="study_03_treatment_target",
            title="Study 03 — Treatment-Target Identification: PHOENIX vs Healthcare Expert",
            report_name="study_03_report.txt",
            data_filename="study_03_treatment_target.csv",
            item_col="task_ID",
            dimension_order=[
                "clinical_priority",
                "evidence_alignment",
                "rank_coherence",
            ],
        )
    )
