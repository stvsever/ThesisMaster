# PHOENIX Engine 🔥🧠📈

Research-grade codebase for the **Ghent University master’s thesis** on personalized mental-health optimization.

## Academic Context

- Institution: Ghent University  
- Author: Stijn Van Severen  
- Supervisors: Geert Crombez, Annick De Paepe

## Thesis Scope

PHOENIX (**P**ersonalized **H**ierarchical **O**ptimization **E**ngine for **N**avigating **I**nsightful e**X**plorations) models mental states as iterative, data-driven optimization problems:

1. free-text operationalization of criterions,
2. initial criterion-predictor observation model construction,
3. readiness-aware time-series network analysis,
4. momentary impact quantification,
5. agentic treatment-target identification (BFS-guided),
6. updated observation-model refinement with readiness-weighted nomothetic×idiographic fusion,
7. HAPA-based digital intervention translation (barrier + coping + personalized message),
8. visualization and research communication outputs.

Current repository runs are based on **synthetic pseudodata**, with architecture prepared for future frontend/backend real-world deployment.

## Project Structure

```text
MASTERPROEF/
├── src/
│   ├── SystemComponents/      # Core thesis components (Agentic, HUA, Ontology)
│   └── utils/                 # Official and exploratory utilities
├── Evaluation/                # Orchestration, QA, communication, synthetic datasets
├── .github/workflows/         # CI and smoke workflows
├── pyproject.toml             # Project metadata
├── requirements.txt
└── requirements-dev.txt
```

## Pipeline Entry

```bash
python Evaluation/00_pipeline_orchestration/run_pipeline.py --mode synthetic_v1
```

Integrated synthetic run sequence:
`readiness → network analysis → impact quantification → Step-03 target identification → Step-04 fused updated model → Step-05 HAPA intervention generation → visuals → research report`.

Iterative mode (history-aware):

```bash
python Evaluation/00_pipeline_orchestration/run_pipeline.py --mode synthetic_v1 --cycles 2 --profile-memory-window 3
```

Guardrail + hard ontology mode:

```bash
python Evaluation/00_pipeline_orchestration/run_pipeline.py --mode synthetic_v1 \
  --hard-ontology-constraint \
  --handoff-critic-max-iterations 2 \
  --intervention-critic-max-iterations 2
```

## Architecture Flow

```mermaid
flowchart LR
    A["Free text + Pseudodata"] --> B["Step 01-02<br/>Readiness + Network Analysis"]
    B --> C["Step 03<br/>Momentary Impact"]
    C --> D["Step 04<br/>Target Selection + Updated Model"]
    D --> E["Step 05<br/>HAPA Intervention"]
    E --> F["Visuals + Research Report"]
    F --> G["History Ledger<br/>JSONL + Parquet"]
    G --> D
```

## Quality and CI

```bash
make qa-unit
make qa-integration
```

- CI: `.github/workflows/ci.yml`
- Smoke pipeline: `.github/workflows/smoke_pipeline.yml`
- Contract validation: `Evaluation/06_quality_assurance/validate_contract_schemas.py`

## Near-Term Roadmap

1. Strengthen closed-loop iterative updates toward full multi-cycle PHOENIX runs.
2. Extend adaptive intervention delivery into UI-native execution (`run_pipeline.py --mode full_engine` target path).
3. Integrate participant-facing frontend/backend data ingestion for real-world data streams.
4. Execute expert and participant evaluation studies with thesis-aligned protocols.

## Security and Data Hygiene

- `.env`, secrets, caches, and heavy generated artifacts are excluded via `.gitignore`.
- Ontology structure is preserved; generated run artifacts are kept local by default.
- Large generated files (e.g., visuals, run outputs, caches) are intentionally excluded from GitHub.

## LLM Reliability and Fallbacks

- Structured calls are routed through a shared runtime with retry + bounded repair.
- Failure taxonomy is explicit: `provider_unavailable`, `schema_validation_failed`, `repair_exhausted`, `budget_exceeded`.
- Step-03/04/05 now support actor-critic guardrail loops with weighted composite quality scoring and max 2 refinement rounds.
- Parent-domain predictor feasibility evidence is injected into critic evaluations for stronger grounding.
- `--hard-ontology-constraint` forces predictor/barrier/coping selections to PHOENIX ontology-matched nodes.
- Fallback outputs remain schema-valid and include the explicit limitation:
  - `Limitation recorded: LLM unavailable, so it’s impact-driven only.`

## License

`GPL-3.0` — see `LICENSE`.
