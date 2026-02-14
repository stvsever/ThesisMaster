# Agentic Framework

This component contains the PHOENIX LLM-mediated stages for model construction, target selection, model updating, and intervention translation.

## Modules

- `01_OperationalizationMentalHealthProblem/`
  - free-text to structured criterion operationalization.
- `02_ConstructionInitialObservationModel/`
  - initial criterion-predictor model generation with optional critic review and ontology constraints.
- `03_TreatmentTargetIdentification/`
  - evidence-integrated target ranking and Step-04 handoff artifacts.
- `04_ConstructionUpdatedObservationModel/`
  - cycle-level updated model lineage and history append logic.
- `05_TranslationDigitalIntervention/`
  - HAPA-based intervention generation with barrier/coping ranking and guardrail validation.

## Shared Infrastructure

- `shared/contracts/`: schema contracts and validators.
- `shared/llm_runtime.py`: LLM execution, retry, repair, fallback.
- `shared/guardrail.py`: critic scoring and pass/revise policy.
- `shared/feasibility.py`: predictor feasibility aggregation and matching.
- `prompts/`: centralized prompt registry and manifest.

## Integration Entry

Use the orchestrator for full-stage consistency:

```bash
python Evaluation/00_pipeline_orchestration/run_pipeline.py --mode synthetic_v1
```
