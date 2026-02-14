# PHOENIX Engine

Research-grade software for the Ghent University master’s thesis on personalized, iterative mental-health optimization.

## Academic Context

- **Institution:** Ghent University
- **Author:** Stijn Van Severen
- **Supervisors:** Geert Crombez, Annick De Paepe

## What PHOENIX Does

PHOENIX (Personalized Hierarchical Optimization Engine for Navigating Insightful eXplorations) operationalizes mental-state support as an iterative data-analysis and decision-support workflow:

1. operationalize free-text complaints into measurable criterions,
2. build an initial criterion-predictor observation model,
3. run readiness-aware time-series analysis,
4. quantify predictor momentary impact,
5. identify treatment targets,
6. construct an updated observation model,
7. generate a HAPA-based digital intervention,
8. produce standardized visual and research communication artifacts.

The repository currently uses synthetic pseudodata and is structured for future frontend/backend real-data integration.

## Repository Structure

```text
MASTERPROEF/
├── src/
│   ├── SystemComponents/
│   │   ├── Agentic_Framework/
│   │   ├── Hierarchical_Updating_Algorithm/
│   │   └── PHOENIX_ontology/
│   ├── overview/
│   └── utils/
├── Evaluation/
│   ├── 00_pipeline_orchestration/
│   ├── 05_integrated_pipeline_runs/
│   ├── 06_quality_assurance/
│   └── 07_research_communication/
├── .github/workflows/
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

## Quick Start

Run the integrated synthetic pipeline:

```bash
python Evaluation/00_pipeline_orchestration/run_pipeline.py --mode synthetic_v1
```

Run iterative cycles with memory:

```bash
python Evaluation/00_pipeline_orchestration/run_pipeline.py --mode synthetic_v1 --cycles 2 --profile-memory-window 3
```

Run with stricter ontology and guardrails:

```bash
python Evaluation/00_pipeline_orchestration/run_pipeline.py --mode synthetic_v1 \
  --hard-ontology-constraint \
  --handoff-critic-max-iterations 2 \
  --intervention-critic-max-iterations 2
```

## Pipeline Overview

```mermaid
flowchart LR
    A["Free text and pseudodata"] --> B["Readiness and network analysis"]
    B --> C["Momentary impact quantification"]
    C --> D["Target identification and updated model"]
    D --> E["HAPA intervention generation"]
    E --> F["Visualizations and run report"]
    F --> G["Run history ledger"]
    G --> D
```

## Quality Assurance

```bash
make qa-unit
make qa-integration
```

- CI workflow: `.github/workflows/ci.yml`
- Smoke workflow: `.github/workflows/smoke_pipeline.yml`
- Contract validation entrypoint: `Evaluation/06_quality_assurance/validate_contract_schemas.py`

## LLM Reliability and Fallbacks

- Shared runtime supports retry, bounded auto-repair, and structured validation.
- Error classes are explicit (provider, schema, repair, budget).
- Actor-critic loops are available in Step-03/04/05.
- `--hard-ontology-constraint` enforces ontology-matched outputs across key decisions.
- If LLM execution is unavailable, pipeline outputs remain schema-valid and include:
  - `Limitation recorded: LLM unavailable, so it’s impact-driven only.`

## Data and Security Hygiene

- `.env`, secrets, caches, and heavy generated artifacts are excluded by `.gitignore`.
- Ontology content remains versioned and protected.
- Large generated run outputs are intentionally kept out of version control.

## License

`GPL-3.0` (see `LICENSE`).
