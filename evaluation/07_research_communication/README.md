# Research Communication

Utilities for generating publication-ready summaries from integrated PHOENIX runs.

## Script

- `generate_pipeline_research_report.py`

## Inputs and Outputs

Input: `evaluation/05_integrated_pipeline_runs/<run_id>/`

Outputs:
- `run_report.md`
- `run_report.json`
- `component_status.csv`
- `profile_overview.csv`

## Example

```bash
python evaluation/07_research_communication/generate_pipeline_research_report.py \
  --run-root "evaluation/05_integrated_pipeline_runs/<run_id>" \
  --output-root "evaluation/05_integrated_pipeline_runs/<run_id>/05_research_reports"
```

## Scope

Current reports cover synthetic-data runs, with schema compatibility for future real-world ingestion workflows.
