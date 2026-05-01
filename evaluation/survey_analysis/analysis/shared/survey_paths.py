"""
Path helpers for the PHOENIX survey-analysis pipeline.

Layout (rooted at evaluation/survey_analysis/):
    data/01_raw/         Qualtrics CSV (gitignored, symlinked or referenced)
    data/02_parsed/      Parsed HCP outputs as JSON, one per (case, hcp)
    data/03_system/      PHOENIX (system) outputs as JSON, one per case
    data/04_judgments/   Long-format signed judge scores + raw responses
    data/pseudodata/     Pseudo HCP/PHOENIX/judgments for end-to-end testing
    results/<study>/     Per-study figures + reports
"""

from __future__ import annotations

from pathlib import Path

# analysis/shared/survey_paths.py -> survey_analysis/
SURVEY_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = SURVEY_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "01_raw"
PARSED_DIR: Path = DATA_DIR / "02_parsed"
SYSTEM_DIR: Path = DATA_DIR / "03_system"
JUDGMENTS_DIR: Path = DATA_DIR / "04_judgments"
PSEUDODATA_DIR: Path = DATA_DIR / "pseudodata"
RESULTS_DIR: Path = SURVEY_ROOT / "results"

# Qualtrics raw export (the example response CSV).
QUALTRICS_RAW_CSV: Path = (
    SURVEY_ROOT.parent / "qualtrics" / "data" / "01_raw"
    / "Masterproef_May 1, 2026_15.25.csv"
)


def ensure_data_dirs() -> None:
    """Create the data sub-folders if they do not yet exist."""
    for d in (RAW_DIR, PARSED_DIR, SYSTEM_DIR, JUDGMENTS_DIR, PSEUDODATA_DIR):
        d.mkdir(parents=True, exist_ok=True)


def ensure_study_dirs(study_slug: str) -> dict[str, Path]:
    """Return (and create) the per-study output directories."""
    study_root = RESULTS_DIR / study_slug
    report_dir = study_root / "report"
    visuals_dir = study_root / "visuals"
    report_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)
    return {
        "study_root": study_root,
        "report_dir": report_dir,
        "visuals_dir": visuals_dir,
    }


def data_file(filename: str) -> Path:
    """Resolve a filename within DATA_DIR (compat shim for legacy callers)."""
    return DATA_DIR / filename


def judgments_csv(part: str | None = None) -> Path:
    """
    Path to the long-format judgments CSV.

    If `part` is given, return the per-part split CSV under 04_judgments/by_part/.
    """
    if part is None:
        return JUDGMENTS_DIR / "judgments_long.csv"
    return JUDGMENTS_DIR / "by_part" / f"{part}_judgments.csv"


def raw_judgment_path(part: str, case_id: str, run_idx: int) -> Path:
    """Where to save the raw judge response JSON for one (part, case, run)."""
    return JUDGMENTS_DIR / "raw" / part / f"case_{case_id}_run_{run_idx}.json"
