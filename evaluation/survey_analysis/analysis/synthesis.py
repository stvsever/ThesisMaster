"""Cross-part holistic synthesis driver."""

from __future__ import annotations

from typing import Any, Dict

from .shared import HolisticStudyConfig, run_holistic_synthesis


def run() -> Dict[str, Any]:
    config = HolisticStudyConfig(
        study_slug="synthesis",
        title="Cross-part holistic comparison",
        report_name="synthesis_report.txt",
    )
    return run_holistic_synthesis(config)


if __name__ == "__main__":
    run()
