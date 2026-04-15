from __future__ import annotations

from shared.comparison_study import ComparisonStudyConfig, run_comparison_study


if __name__ == "__main__":
    run_comparison_study(
        ComparisonStudyConfig(
            study_slug="study_02_initial_model",
            title="Study 02 — Initial Observational Model: PHOENIX vs Healthcare Expert",
            report_name="study_02_report.txt",
            data_filename="study_02_initial_model.csv",
            item_col="item_ID",
            dimension_order=[
                "clinical_appropriateness",
                "network_validity",
                "ema_feasibility",
                "intervention_potential",
            ],
        )
    )
