"""
End-to-end smoke test for the PHOENIX evaluation pipeline.

Runs the full pipeline in pseudo mode on a small subset and asserts that
the expected output files appear and the LMM converges for a healthy
fraction of dimensions.

Designed to be importable and runnable both via ``python -m unittest``
and as a plain ``python tests/test_smoke.py``.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from analysis.shared.survey_paths import (  # noqa: E402
    JUDGMENTS_DIR,
    RESULTS_DIR,
    judgments_csv,
)
from llm_as_judge.dimensions import DIMENSIONS_BY_PART  # noqa: E402
from pipeline import build_parser, main  # noqa: E402


CASE_SUBSET = ["C01", "C02", "C03", "C04", "C05"]
PART_SUBSET = list(DIMENSIONS_BY_PART.keys())   # all 5 parts


class SmokeTest(unittest.TestCase):
    """Run the full pipeline in pseudo mode and check the artefacts."""

    @classmethod
    def setUpClass(cls) -> None:
        argv = [
            "--mode", "pseudo",
            "--n-runs", "3",
            "--cases", *CASE_SUBSET,
            "--parts", *PART_SUBSET,
            "--log-level", "WARNING",
        ]
        rc = main(argv)
        cls.return_code = rc

    def test_return_code(self) -> None:
        self.assertEqual(self.return_code, 0)

    def test_judgments_csv_exists(self) -> None:
        path = judgments_csv()
        self.assertTrue(path.exists(), f"Missing {path}")
        df = pd.read_csv(path)
        self.assertGreater(len(df), 0, "judgments_long.csv is empty")
        # One row per (case, part, run, dimension, entity).
        for col in ("case_id", "part", "dimension", "judge_run", "entity",
                    "quality_score", "source_label", "confidence",
                    "prompt_version", "model"):
            self.assertIn(col, df.columns, f"missing column {col!r}")

    def test_per_part_results(self) -> None:
        for part in PART_SUBSET:
            slug = {
                "part1": "part1_prompt",
                "part2": "part2_prompt",
                "part3": "part3_prompt",
                "part4": "part4_prompt",
                "part5": "part5_prompt",
            }[part]
            report_dir = RESULTS_DIR / slug / "report"
            visuals_dir = RESULTS_DIR / slug / "visuals"
            self.assertTrue(
                (report_dir / f"{slug}_summary.csv").exists(),
                f"missing summary CSV for {part}",
            )
            self.assertTrue(
                (visuals_dir / f"{slug}_effect_forest.png").exists(),
                f"missing forest plot for {part}",
            )
            self.assertTrue(
                (visuals_dir / f"{slug}_standardized_effect_forest.png").exists(),
                f"missing standardized forest plot for {part}",
            )
            self.assertTrue(
                (visuals_dir / f"{slug}_quality_raincloud.png").exists(),
                f"missing raincloud plot for {part}",
            )
            self.assertTrue(
                (visuals_dir / f"{slug}_tost_equivalence.png").exists(),
                f"missing TOST panel for {part}",
            )

    def test_model_or_fallback_rate(self) -> None:
        # Aggregate successful model-or-documented-fallback rates across summaries.
        total = 0
        valid = 0
        slugs = [
            "part1_prompt",
            "part2_prompt",
            "part3_prompt",
            "part4_prompt",
            "part5_prompt",
        ]
        for slug in slugs:
            csv_path = RESULTS_DIR / slug / "report" / f"{slug}_summary.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            total += len(df)
            valid += int((df["method"] != "No converged mixed model").sum())
        self.assertGreater(total, 0, "no per-part summaries found")
        rate = valid / total
        self.assertGreaterEqual(
            rate, 0.80,
            f"valid model/fallback rate too low: {valid}/{total} = {rate:.2%}",
        )

    def test_synthesis_artefacts(self) -> None:
        synth_dir = RESULTS_DIR / "synthesis"
        report = synth_dir / "report" / "synthesis_report.txt"
        self.assertTrue(report.exists(), f"missing {report}")
        for fname in (
            "synthesis_part_forest.png",
            "synthesis_standardized_effect_forest.png",
            "synthesis_part_raincloud.png",
            "synthesis_tost.png",
            "synthesis_gap_heatmap.png",
        ):
            self.assertTrue(
                (synth_dir / "visuals" / fname).exists(),
                f"missing synthesis visual {fname}",
            )

    def test_supplementary_artefacts(self) -> None:
        supp_dir = RESULTS_DIR / "supplementary"
        report = supp_dir / "report" / "supplementary_report.txt"
        self.assertTrue(report.exists(), f"missing {report}")
        for fname in (
            "supplementary_overview_dashboard.png",
            "suppA_icc_stability.png",
            "suppB_calibration_diagnostics.png",
            "suppC_sensitivity_forest.png",
            "suppD_case_heterogeneity.png",
        ):
            self.assertTrue(
                (supp_dir / "visuals" / fname).exists(),
                f"missing supplementary visual {fname}",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
