"""
Generate pseudo judgments by invoking the pseudo-judge over all
``(case, part, run)`` cells using the same persistence pipeline as
``judge_runner.run_judge``.

GROUND TRUTH
------------
See ``llm_as_judge.pseudo_judge.PHOENIX_MEAN_QUALITY`` and
``HCP_MEAN_QUALITY``. The downstream analysis stage is expected to recover
the simulated PHOENIX - HCP quality gaps within uncertainty intervals.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from analysis.shared.survey_paths import (
    JUDGMENTS_DIR,
    PSEUDODATA_DIR,
    judgments_csv,
)
from llm_as_judge.dimensions import DIMENSIONS_BY_PART
from llm_as_judge.judge_runner import JudgeRunConfig, run_judge

logger = logging.getLogger(__name__)

# Default case set; mirrored from pseudodata.__init__ to avoid a circular import.
CASE_IDS: tuple[str, ...] = tuple(f"C{i:02d}" for i in range(1, 11))


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def generate_pseudo_judgments(
    *,
    n_runs: int = 3,
    cases: Optional[list[str]] = None,
    parts: Optional[list[str]] = None,
    out_csv: Optional[Path] = None,
    raw_dir_root: Optional[Path] = None,
    hcp_path: Optional[Path] = None,
    phoenix_path: Optional[Path] = None,
) -> Path:
    """
    Run the pseudo-judge over every ``(case, part, run)`` cell.

    Parameters mirror :class:`JudgeRunConfig`. By default writes both:
      - ``data/pseudodata/judgments_long.csv`` (canonical pseudo artefact)
      - ``data/04_judgments/judgments_long.csv`` (the analysis-stage entry)
    """
    cases = list(cases) if cases else list(CASE_IDS)
    parts = list(parts) if parts else list(DIMENSIONS_BY_PART.keys())

    hcp_path = hcp_path or PSEUDODATA_DIR / "hcp_outputs.json"
    phoenix_path = phoenix_path or PSEUDODATA_DIR / "phoenix_outputs.json"
    if not hcp_path.exists() or not phoenix_path.exists():
        raise FileNotFoundError(
            "Pseudo HCP/PHOENIX outputs not found. Run "
            "pseudodata.generate_hcp_outputs and "
            "pseudodata.generate_phoenix_outputs first."
        )

    hcp_outputs = _load_json(hcp_path)
    phoenix_outputs = _load_json(phoenix_path)

    primary_csv = out_csv or PSEUDODATA_DIR / "judgments_long.csv"
    raw_root = raw_dir_root or PSEUDODATA_DIR / "raw_judgments"

    # Reset target file so re-running is idempotent.
    if primary_csv.exists():
        primary_csv.unlink()

    config = JudgeRunConfig(
        cases=cases,
        parts=parts,
        n_runs=n_runs,
        mode="pseudo",
        out_csv=primary_csv,
        raw_dir_root=raw_root,
    )
    run_judge(
        hcp_outputs=hcp_outputs,
        system_outputs=phoenix_outputs,
        config=config,
    )

    # Mirror to the analysis-stage location.
    analysis_csv = judgments_csv()
    analysis_csv.parent.mkdir(parents=True, exist_ok=True)
    analysis_csv.write_bytes(primary_csv.read_bytes())
    logger.info("Mirrored pseudo judgments to %s", analysis_csv)
    return analysis_csv


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = generate_pseudo_judgments()
    print(f"Wrote pseudo judgments to {p}")
