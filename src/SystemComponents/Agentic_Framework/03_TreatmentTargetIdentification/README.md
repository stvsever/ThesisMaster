# Step 03: Treatment Target Identification (Handoff Prep)

This folder currently contains preprocessing utilities that prepare step-03 inputs from impact coefficients.

## Script

- `01_prepare_targets_from_impact.py`
  - Reads `predictor_composite.csv` outputs from impact quantification.
  - Produces per-profile candidate tables (`top_treatment_target_candidates.csv`) and summary files.
  - Intended as handoff material for the later full treatment-target identification agent.
