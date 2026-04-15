from __future__ import annotations

from shared.comparison_study import ComparisonStudyConfig, run_comparison_study


if __name__ == "__main__":
    run_comparison_study(
        ComparisonStudyConfig(
            study_slug="study_04_updated_model",
            title="Study 04 — Updated Observational Model: PHOENIX vs Healthcare Expert",
            report_name="study_04_report.txt",
            data_filename="study_04_updated_model.csv",
            item_col="task_ID",
            dimension_order=[
                "target_alignment",
                "measurement_selection",
            ],
        )
    )
