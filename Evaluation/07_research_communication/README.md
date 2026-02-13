# PHOENIX Research Communication

This folder standardizes communication artifacts for integrated pipeline runs.

## Script

- `generate_pipeline_research_report.py`
  - Input: one run folder (e.g. `Evaluation/05_integrated_pipeline_runs/<run_id>/`)
  - Output:
    - `run_report.md` (human-readable summary)
    - `run_report.json` (machine-readable summary)
    - `component_status.csv`
    - `profile_overview.csv`

## Example

```bash
python Evaluation/07_research_communication/generate_pipeline_research_report.py \
  --run-root "Evaluation/05_integrated_pipeline_runs/<run_id>" \
  --output-root "Evaluation/05_integrated_pipeline_runs/<run_id>/05_research_reports"
```

## Scope Note

Current reports summarize synthetic-data evaluation runs.  
The schema is intentionally forward-compatible with future real-world frontend/backend data ingestion.

