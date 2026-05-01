"""Part 4 - Concrete EMA item selection: signed LMM, TOST, plots."""

from __future__ import annotations

from typing import Any, Dict

from llm_as_judge.dimensions import dimensions_for

from .shared import ComparisonStudyConfig, run_comparison_study


def run() -> Dict[str, Any]:
    config = ComparisonStudyConfig(
        study_slug="part4_prompt",
        part="part4",
        title="Part 4: Concrete EMA item selection",
        report_name="part4_prompt_report.txt",
        dimension_order=[d.key for d in dimensions_for("part4")],
    )
    return run_comparison_study(config)


if __name__ == "__main__":
    run()
