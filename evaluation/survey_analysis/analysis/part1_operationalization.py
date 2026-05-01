"""Part 1 — Operationalisation: per-dimension LMM, TOST, plots."""

from __future__ import annotations

from typing import Any, Dict

from llm_as_judge.dimensions import dimensions_for

from .shared import ComparisonStudyConfig, run_comparison_study


def run() -> Dict[str, Any]:
    config = ComparisonStudyConfig(
        study_slug="part1_operationalization",
        part="part1",
        title="Part 1: Operationalisation of mental state",
        report_name="part1_operationalization_report.txt",
        dimension_order=[d.key for d in dimensions_for("part1")],
    )
    return run_comparison_study(config)


if __name__ == "__main__":
    run()
