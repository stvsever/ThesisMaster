# Hierarchical Updating Algorithm

This directory contains the quantitative analysis core for readiness estimation, network time-series analysis, and impact quantification.

## Modules

- `01_time_series_analysis/01_check_readiness/`
  - Readiness classification and diagnostics on pseudodata profiles.
- `01_time_series_analysis/02_network_time_series_analysis/`
  - Readiness-driven tv-gVAR, stationary baselines, and network metrics.
- `01_time_series_analysis/02_regular_time_series_analysis/`
  - Non-network regular time-series analyses.
- `02_hierarchical_update_ranking/01_momentary_impact_quantification/`
  - Edge-level and predictor-level impact coefficient computation.

## Integrated Execution

Use the integrated runner for consistent outputs, logs, and validation:

```bash
python Evaluation/00_pipeline_orchestration/run_pipeline.py --mode synthetic_v1
```

## Readiness-to-Analysis Consistency

- Readiness now exposes an explicit `analysis_execution_plan` in each `readiness_report.json`.
- The network analysis stage follows this plan by default (`--execution-policy readiness_aligned`).
- `FullyReady` is reserved for profiles where full time-varying gVAR is executable.
