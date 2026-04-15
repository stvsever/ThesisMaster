from __future__ import annotations

from shared.comparison_study import ComparisonStudyConfig, run_comparison_study


if __name__ == "__main__":
    run_comparison_study(
        ComparisonStudyConfig(
            study_slug="study_01_operationalization",
            title="Study 01 — Operationalization: PHOENIX vs Healthcare Expert",
            report_name="study_01_report.txt",
            data_filename="study_01_operationalization.csv",
            item_col="text_ID",
            dimension_order=[
                "criterion_accuracy",
                "operationalization_quality",
                "completeness",
            ],
        )
    )
