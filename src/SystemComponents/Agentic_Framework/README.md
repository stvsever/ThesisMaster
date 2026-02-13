# Agentic Framework

This directory contains the thesis agentic workflow components around problem operationalization, model construction, and treatment-target handoff.

## Current Module Status

- `01_OperationalizationMentalHealthProblem/`
  - Converts free-text complaints into structured criterion variables mapped to ontology leaves.
- `02_ConstructionInitialObservationModel/`
  - Builds initial criterion-predictor observation models from mapped complaints and ranking signals.
- `03_TreatmentTargetIdentification/`
  - Contains handoff-preparation script that converts momentary-impact outputs into ranked treatment-target candidate tables.
- `04_TranslationDigitalIntervention/`
  - Reserved for later implementation.
- `05_ConstructionUpdatedObservationModel/`
  - Reserved for later implementation.

## Integration

The integrated execution from pseudodata to step-03 handoff is orchestrated via:

- `Evaluation/00_pipeline_orchestration/run_pipeline.py --mode synthetic_v1`
