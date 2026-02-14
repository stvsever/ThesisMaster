# Time-Series Analysis Layer

Profile-level readiness and network modeling modules.

## Submodules

- `01_check_readiness/`: generates readiness diagnostics and execution recommendations.
- `02_network_time_series_analysis/`: executes network analyses according to readiness tier.
- `02_regular_time_series_analysis/`: supports regular time-series alternatives.

## Recommended Usage

Run via the integrated orchestrator to preserve standardized paths, logs, and contracts:

```bash
python Evaluation/00_pipeline_orchestration/run_pipeline.py --mode synthetic_v1
```
