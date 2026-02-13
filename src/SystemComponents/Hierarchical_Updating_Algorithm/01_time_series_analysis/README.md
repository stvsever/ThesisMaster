# Time-Series Analysis

Core analysis layer for profile-level readiness and network modeling.

## Submodules

- `01_check_readiness/`
  - Produces `readiness_report.json` + `readiness_summary.txt` per profile.
- `02_network_time_series_analysis/`
  - Produces tv-gVAR/stationary/correlation outputs plus network metrics.
- `02_regular_time_series_analysis/`
  - Produces regular non-network time-series analytics.

## Recommended Execution

For end-to-end thesis workflow consistency, run through:

```bash
python Evaluation/00_pipeline_orchestration/run_pipeline.py --mode synthetic_v1
```

This keeps path management, logging, and stage validation centralized.
