# Agentic Framework

This directory contains the thesis agentic workflow components around problem operationalization, model construction, and treatment-target handoff.

## Current Module Status

- `01_OperationalizationMentalHealthProblem/`
  - Converts free-text complaints into structured criterion variables mapped to ontology leaves.
- `02_ConstructionInitialObservationModel/`
  - Builds initial criterion-predictor observation models from mapped complaints and ranking signals.
  - Includes Step-02 guardrail critic review with weighted multi-domain feasibility grounding.
  - Supports optional hard ontology constraints (`--hard-ontology-constraint`) with path enforcement.
  - Writes per-profile guardrail artifacts (`step02_guardrail_review.json`, `step02_guardrail_trace.json`).
- `03_TreatmentTargetIdentification/`
  - Implements structured Step-03 target identification using integrated evidence (impact, readiness, network, free text, mapped model).
  - Produces Step-04 updated observation-model suggestions with ontology-constrained candidate sets.
  - Enforces breadth-first domain coverage and outputs readiness-weighted nomothetic+idiographic fusion artifacts and visuals.
  - Includes optional actor-critic guardrail review loops and hard ontology constraints (`--hard-ontology-constraint`).
- `04_ConstructionUpdatedObservationModel/` (canonical Step-04)
  - Hosts iterative cycle-level updated-model orchestration.
  - `01_run_updated_model_cycle.py` consolidates Step-03/04/05 artifacts into a lineage-aware cycle summary.
- `05_TranslationDigitalIntervention/` (canonical Step-05)
  - Implements Step-05 HAPA-based digital intervention generation from integrated outputs.
  - Produces barrier-domain rankings, coping-strategy rankings, and a personalized intervention JSON plan.
  - Includes guardrail critic scoring, revision feedback loops, and ontology-constrained target/barrier/coping enforcement.
- `04_TranslationDigitalIntervention/` and `05_ConstructionUpdatedObservationModel/` (legacy compatibility)
  - Kept as wrappers to preserve backward compatibility for older scripts.

## Integration

The integrated execution from pseudodata to Step-03/Step-04/Step-05 outputs is orchestrated via:

- `Evaluation/00_pipeline_orchestration/run_pipeline.py --mode synthetic_v1`

## Shared Runtime Assets

- `shared/contracts/`: JSON-schema contracts and validators for stage outputs.
- `shared/llm_runtime.py`: structured LLM client with retry/repair/fallback taxonomy.
- `shared/guardrail.py`: weighted critic scoring and decision helpers.
- `shared/feasibility.py`: predictor parent-domain feasibility aggregation/matching helpers.
- `prompts/prompts_manifest.json`: versioned prompt registry for Step-02/03/04/05.
- `prompts/full_engine_prompt_inventory.md`: centralized index of registry-backed and legacy inline prompt sources.
