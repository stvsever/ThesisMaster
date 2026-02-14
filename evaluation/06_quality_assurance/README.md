# Quality Assurance

Research-grade QA for PHOENIX pipeline reliability and contract stability.

## Test Layers

- `tests/unit/`: deterministic logic, scoring, gating, and utility tests.
- `tests/integration/`: cross-component behavior, artifact generation, and orchestrator checks.
- smoke markers: optional deeper runs for manual or scheduled validation.

## Commands

```bash
pytest -m "unit"
pytest -m "integration and not smoke"
PHOENIX_ENABLE_SMOKE=1 pytest -m "smoke"
```

## CI Alignment

Default CI enforces unit + non-smoke integration tests. Smoke runs remain optional to keep routine CI duration practical.
