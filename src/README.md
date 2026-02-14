# Source Code (`src`)

This directory contains the PHOENIX implementation layers and supporting utilities.

## Structure

- `SystemComponents/`
  - `Agentic_Framework/`: LLM-mediated decision stages (operationalization, modeling, targeting, intervention).
  - `Hierarchical_Updating_Algorithm/`: readiness, network analysis, and impact quantification.
  - `PHOENIX_ontology/`: ontology assets and mappings (kept structurally stable).
- `overview/`: architecture-level documentation.
- `utils/`: official utilities, shared agentic runtime assets, and non-core helper tooling.

## Usage Note

For reproducible execution, run through `Evaluation/00_pipeline_orchestration/run_pipeline.py` rather than calling stage scripts ad hoc.
