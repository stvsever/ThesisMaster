# Step 04 — Updated Observation Model

Canonical module for cycle-level updated observation-model persistence.

## Main Script

- `01_run_updated_model_cycle.py`

## Purpose

- consolidate Step-03/04/05 artifacts into a cycle summary,
- maintain model lineage across cycles,
- append profile-level history events for resumable iterative runs.

## Expected Inputs

- `step03_treatment_target_selection.json`
- `step04_updated_observation_model.json`
- `step05_hapa_intervention.json` (optional)

## Outputs

- `updated_model_cycle_summary.json`
- `history/profile_events.jsonl` (optional append)
