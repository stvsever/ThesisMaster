"""Path constants for the PHOENIX survey-output preparation flow."""

from __future__ import annotations

from pathlib import Path

ROOT: Path = Path(__file__).resolve().parent
EVALUATION_ROOT: Path = ROOT.parent
REPO_ROOT: Path = EVALUATION_ROOT.parent

DATA_DIR: Path = ROOT / "data"
INPUTS_DIR: Path = DATA_DIR / "inputs"
OUTPUTS_DIR: Path = DATA_DIR / "outputs"
LOGS_DIR: Path = DATA_DIR / "logs"

QUALTRICS_SURVEY_AUTOMATED_DIR: Path = (
    EVALUATION_ROOT
    / "qualtrics"
    / "survey"
    / "01_HCPs_PRE"
    / "separate_HCPs"
    / "survey_AUTOMATED"
)
QUALTRICS_GENERATE_QSF: Path = QUALTRICS_SURVEY_AUTOMATED_DIR / "generate_qsf.py"
QUALTRICS_CASE_SOURCE_ROOT: Path = (
    QUALTRICS_SURVEY_AUTOMATED_DIR.parent / "1_case_per_HCP"
)
QUALTRICS_EDGE_WEIGHTS: Path = (
    EVALUATION_ROOT
    / "qualtrics"
    / "survey"
    / "01_HCPs_PRE"
    / "bipartite_edge_weights.json"
)

CASE_INPUTS_JSON: Path = INPUTS_DIR / "qualtrics_case_inputs.json"
CASE_CONTEXTS_JSON: Path = INPUTS_DIR / "case_contexts_for_judge.json"
SYSTEM_OUTPUTS_JSON: Path = OUTPUTS_DIR / "system_outputs.json"
SYSTEM_OUTPUTS_TEMPLATE_JSON: Path = OUTPUTS_DIR / "system_outputs_template.json"
VALIDATION_REPORT_JSON: Path = LOGS_DIR / "validation_report.json"

SURVEY_ANALYSIS_DIR: Path = EVALUATION_ROOT / "survey_analysis"
SURVEY_ANALYSIS_CONTEXTS_JSON: Path = (
    SURVEY_ANALYSIS_DIR / "data" / "01_raw" / "case_contexts.json"
)
SURVEY_ANALYSIS_SYSTEM_OUTPUTS_JSON: Path = (
    SURVEY_ANALYSIS_DIR / "data" / "03_system" / "system_outputs.json"
)


def ensure_dirs() -> None:
    """Create local data directories."""
    for path in (INPUTS_DIR, OUTPUTS_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)


__all__ = [
    "ROOT",
    "EVALUATION_ROOT",
    "REPO_ROOT",
    "DATA_DIR",
    "INPUTS_DIR",
    "OUTPUTS_DIR",
    "LOGS_DIR",
    "QUALTRICS_SURVEY_AUTOMATED_DIR",
    "QUALTRICS_GENERATE_QSF",
    "QUALTRICS_CASE_SOURCE_ROOT",
    "QUALTRICS_EDGE_WEIGHTS",
    "CASE_INPUTS_JSON",
    "CASE_CONTEXTS_JSON",
    "SYSTEM_OUTPUTS_JSON",
    "SYSTEM_OUTPUTS_TEMPLATE_JSON",
    "VALIDATION_REPORT_JSON",
    "SURVEY_ANALYSIS_CONTEXTS_JSON",
    "SURVEY_ANALYSIS_SYSTEM_OUTPUTS_JSON",
    "ensure_dirs",
]

