# Step-04 Construction Updated Observation Model

This is the canonical Step-04 module in PHOENIX.

## Purpose

- Build and persist cycle-level updated observation model summaries.
- Keep iterative lineage explicit across cycles.
- Support history append events for resumable runs.

## Main script

- `01_run_updated_model_cycle.py`

## Utilities

- `utils/cycle_summary.py`
- `utils/history_io.py`

## Inputs

- `step03_treatment_target_selection.json`
- `step04_updated_observation_model.json`
- `step05_hapa_intervention.json` (optional)

## Outputs

- `updated_model_cycle_summary.json`
- `history/profile_events.jsonl` (optional append)
