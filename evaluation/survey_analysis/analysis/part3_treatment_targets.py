"""Part 3 — Treatment-target prioritisation: per-dimension LMM, TOST, plots.

The judge rates the QUALITY of the ranking (top-target appropriateness,
evidence alignment, coherence, network impact awareness, monitoring
integration, modifiability weighting); we do not run a Spearman rank-order
test — the rationale is that two valid clinical orderings can disagree on
exact rank but be equally defensible, and the dimensions encode that.
"""

from __future__ import annotations

from typing import Any, Dict

from llm_as_judge.dimensions import dimensions_for

from .shared import ComparisonStudyConfig, run_comparison_study


def run() -> Dict[str, Any]:
    config = ComparisonStudyConfig(
        study_slug="part3_treatment_targets",
        part="part3",
        title="Part 3: Treatment-target prioritisation",
        report_name="part3_treatment_targets_report.txt",
        dimension_order=[d.key for d in dimensions_for("part3")],
    )
    return run_comparison_study(config)


if __name__ == "__main__":
    run()
