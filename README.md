# PHOENIX Engine — MASTERPROEF

Professional research codebase for the **PHOENIX Engine** thesis project at **Ghent University**.

## Academic Context

- Institution: Ghent University
- Author: Stijn Van Severen
- Supervisors: Geert Crombez, Annick De Paepe

## Research Objective

PHOENIX (Personalized Hierarchical Optimization Engine for Navigating Insightful eXplorations) models mental health as a hierarchical optimization process:

1. operationalize criterion space from user complaints,
2. construct initial observation models,
3. estimate readiness and run multivariate time-series analyses,
4. quantify predictor momentary impact,
5. prepare treatment-target handoff for agentic intervention logic.

Current runs are based on **synthetic pseudodata**.  
The codebase is explicitly designed for transition to real frontend/backend data pipelines.

## Clean Root Structure

```text
MASTERPROEF/
├── src/                          # Core engine code
│   ├── SystemComponents/         # Thesis component architecture
│   └── utils/                    # Support and exploratory tooling
├── Evaluation/                   # Inputs, orchestration, QA, reports
├── .github/workflows/            # CI + smoke workflows
├── pyproject.toml                # Modern Python project metadata
├── requirements.txt              # Runtime dependencies
└── requirements-dev.txt          # QA/CI dependencies
```

Compatibility symlinks (`SystemComponents`, `utils`) are retained at root so existing scripts and paths keep working.

## Current Execution

Install dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Primary launcher (future-ready mode routing):

```bash
python Evaluation/00_pipeline_orchestration/run_pipeline.py --mode synthetic_v1
```

This executes readiness → network analysis → impact quantification → treatment-target handoff preparation → visualizations → research report generation.

## QA and CI

```bash
make qa-unit
make qa-integration
```

CI workflows:
- `.github/workflows/ci.yml` (unit + integration)
- `.github/workflows/smoke_pipeline.yml` (manual smoke run)

## Near-Future Thesis Directions

The current repository is ready for extension toward full PHOENIX deployment:

1. **Agentic Step 03 (full implementation):** treatment-target reasoning agent beyond handoff preparation.
2. **Agentic Step 04:** translation to tailored intervention generation.
3. **Agentic Step 05:** construction of updated observational models after intervention cycles.
4. **Real-world evaluation:** study execution with healthcare professionals and non-expert participants (currently planned after synthetic validation).
5. **UI integration:** migration from `run_pseudodata_to_impact.py` to full `run_pipeline.py` + future `main.py` web entrypoint.

## Security and Data Hygiene

- `.env`, credentials, caches, and generated heavy artifacts are excluded by `.gitignore`.
- Integrated run outputs are reproducible locally and intentionally not tracked by default.

## License

See `LICENSE`.
