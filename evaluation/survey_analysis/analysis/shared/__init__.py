"""
Shared statistical and plotting helpers for the LLM-as-judge analysis.

Re-exports the most commonly used symbols so callers can write::

    from analysis.shared import (
        ComparisonStudyConfig, run_comparison_study,
        HolisticStudyConfig,   run_holistic_synthesis,
    )
"""

from .comparison_study import (
    ComparisonStudyConfig,
    run_comparison_study,
)
from .holistic_comparison import (
    HolisticStudyConfig,
    run_holistic_synthesis,
)
from .shared_stats import (
    PALETTE,
    apply_rcparams,
    bonferroni_correct,
    holm_correct,
    fit_crossed_mixedlm,
    forest_plot,
    raincloud_plot,
    save_figure,
    tost_panel,
    tost_test,
    tost_test_one_sample,
)
from .survey_paths import (
    DATA_DIR,
    JUDGMENTS_DIR,
    PARSED_DIR,
    PSEUDODATA_DIR,
    QUALTRICS_RAW_CSV,
    RAW_DIR,
    RESULTS_DIR,
    SURVEY_ROOT,
    SYSTEM_DIR,
    ensure_data_dirs,
    ensure_study_dirs,
    judgments_csv,
    raw_judgment_path,
)

__all__ = [
    "ComparisonStudyConfig",
    "run_comparison_study",
    "HolisticStudyConfig",
    "run_holistic_synthesis",
    "PALETTE",
    "apply_rcparams",
    "bonferroni_correct",
    "holm_correct",
    "fit_crossed_mixedlm",
    "forest_plot",
    "raincloud_plot",
    "save_figure",
    "tost_panel",
    "tost_test",
    "tost_test_one_sample",
    "DATA_DIR",
    "JUDGMENTS_DIR",
    "PARSED_DIR",
    "PSEUDODATA_DIR",
    "QUALTRICS_RAW_CSV",
    "RAW_DIR",
    "RESULTS_DIR",
    "SURVEY_ROOT",
    "SYSTEM_DIR",
    "ensure_data_dirs",
    "ensure_study_dirs",
    "judgments_csv",
    "raw_judgment_path",
]
