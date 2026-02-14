# Evaluation Workspace

This directory contains orchestration, synthetic inputs, run artifacts, QA suites, and research communication outputs.

## Core Directories

- `00_pipeline_orchestration/`: integrated pipeline entrypoints and run control.
- `01_pseudoprofile(s)/`: synthetic free-text and time-series profile inputs.
- `02_mental_health_issue_operationalization/`: operationalization-stage outputs.
- `03_construction_initial_observation_model/`: initial model construction outputs/helpers.
- `04_initial_observation_analysis/`: legacy/manual analysis artifacts.
- `05_integrated_pipeline_runs/`: standardized run-scoped outputs.
- `06_quality_assurance/`: unit/integration/smoke-oriented validation framework.
- `07_research_communication/`: report-generation utilities.

## Standard Run Output Layout

`evaluation/05_integrated_pipeline_runs/<run_id>/`
- `00_readiness_check/`
- `01_time_series_analysis/network/`
- `02_momentary_impact_coefficients/`
- `03_treatment_target_handoff/`
- `03b_translation_digital_intervention/`
- `04_impact_visualizations/`
- `05_research_reports/`
- `logs/`
- `pipeline_summary.json`

## Sequential Flow (Current Scope)

1. synthetic profile input
2. readiness and method-path decision
3. network and impact estimation
4. target-selection and updated-model handoff
5. HAPA intervention translation
6. visualization and research reporting
