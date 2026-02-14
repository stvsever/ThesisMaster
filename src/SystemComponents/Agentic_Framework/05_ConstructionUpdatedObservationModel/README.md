# Step-05 Construction Updated Observation Model

This component hosts iterative observation-model update utilities for PHOENIX cycles.

## Current scope

- Consolidates Step-03/Step-04/Step-05 artifacts into a cycle-level updated-model summary.
- Persists cycle lineage events for iterative runs.
- Keeps criteria continuity and predictor update trace explicit for downstream analysis and reporting.

## Why this exists (and looked empty before)

The current integrated runtime computes Step-04 updates from the Step-03 module and Step-05 intervention module.  
This folder is now the dedicated home for iterative update-cycle orchestration so the architecture is explicit and thesis-aligned.

## Main script

- `01_run_updated_model_cycle.py`
  - Builds `updated_model_cycle_summary.json`
  - Optionally appends a compact JSONL history event

## Inputs expected

- Step-03 output: `step03_treatment_target_selection.json`
- Step-04 output: `step04_updated_observation_model.json`
- Step-05 output (optional): `step05_hapa_intervention.json`

## Output

- `updated_model_cycle_summary.json`
- optional history append (`profile_events.jsonl`)
