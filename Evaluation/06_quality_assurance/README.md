# PHOENIX Evaluation QA

This folder contains the research-grade test framework for the PHOENIX Engine evaluation stack.

## Test Layers

- `tests/unit/`  
  Deterministic logic tests for readiness, network execution plans, impact scoring utilities, handoff ranking, and pipeline helpers.

- `tests/integration/`  
  Cross-component tests for:
  - dry-run orchestration (`run_pseudodata_to_impact.py`)
  - impact visualization generation
  - research communication report generation
  - optional full smoke run (`PHOENIX_ENABLE_SMOKE=1`)

## Run Commands

```bash
pytest -m "unit"
pytest -m "integration and not smoke"
PHOENIX_ENABLE_SMOKE=1 pytest -m "smoke"
```

## CI Alignment

Default CI executes unit + non-smoke integration tests.
Smoke runs are configured as manual/optional to keep normal CI fast and stable.

