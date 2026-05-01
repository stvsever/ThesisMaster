"""Part 5 - tailored mobile coaching message: signed analyses and plots.

The judge ALSO classifies each message's HAPA phase in the ``extra``
field of its JSON response. That classification is recorded in the raw
response files when HAPA context is available, but it is not part of the
per-dimension signed scores analysed here.
"""

from __future__ import annotations

from typing import Any, Dict

from llm_as_judge.dimensions import dimensions_for

from .shared import ComparisonStudyConfig, run_comparison_study


def run() -> Dict[str, Any]:
    config = ComparisonStudyConfig(
        study_slug="part5_prompt",
        part="part5",
        title="Part 5: Mobile coaching message",
        report_name="part5_prompt_report.txt",
        dimension_order=[d.key for d in dimensions_for("part5")],
    )
    return run_comparison_study(config)


if __name__ == "__main__":
    run()
