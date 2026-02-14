# Hierarchical Updating Algorithm

Quantitative core for PHOENIX readiness estimation, network analysis, and impact scoring.

## Modules

- `01_time_series_analysis/01_check_readiness/`
  - readiness classification and diagnostics.
- `01_time_series_analysis/02_network_time_series_analysis/`
  - readiness-aligned tv-gVAR, stationary gVAR, and baseline analyses.
- `01_time_series_analysis/02_regular_time_series_analysis/`
  - regular non-network analyses.
- `02_hierarchical_update_ranking/01_momentary_impact_quantification/`
  - predictor-level impact coefficient construction.

## Execution

Use orchestrated execution for reproducibility:

```bash
python Evaluation/00_pipeline_orchestration/run_pipeline.py --mode synthetic_v1
```

## Readiness Consistency

`readiness_report.json` includes `analysis_execution_plan`, and downstream analysis follows this plan to keep method selection explicit and auditable.
