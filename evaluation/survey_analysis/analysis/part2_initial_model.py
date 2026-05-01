"""Part 2 — Initial observational model: per-dimension LMM, TOST, plots."""

from __future__ import annotations

from typing import Any, Dict

from llm_as_judge.dimensions import dimensions_for

from .shared import ComparisonStudyConfig, run_comparison_study


def run() -> Dict[str, Any]:
    config = ComparisonStudyConfig(
        study_slug="part2_initial_model",
        part="part2",
        title="Part 2: Initial observational model",
        report_name="part2_initial_model_report.txt",
        dimension_order=[d.key for d in dimensions_for("part2")],
    )
    return run_comparison_study(config)


if __name__ == "__main__":
    run()
