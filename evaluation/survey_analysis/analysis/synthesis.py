"""
Cross-part holistic synthesis driver for the absolute-quality design.

Fits quality_score ~ entity_ec (PHOENIX=+0.5, HCP=−0.5) with
case_id / judge_run random effects across all five PHOENIX evaluation parts.
"""

from __future__ import annotations

from typing import Any, Dict

from .shared import HolisticStudyConfig, run_holistic_synthesis


def run() -> Dict[str, Any]:
    config = HolisticStudyConfig(
        study_slug="synthesis",
        title="Cross-part PHOENIX vs HCP quality synthesis",
        report_name="synthesis_report.txt",
        tost_delta=0.3,   # ±0.3 pts on the 1–5 scale
    )
    return run_holistic_synthesis(config)


if __name__ == "__main__":
    run()
