# PHOENIX Frontend (Flask)

Interactive debugging UI for the PHOENIX engine, built to inspect and run the thesis pipeline with high-detail runtime logging.

## What It Covers

- Session-based intake: complaint text + optional person/context.
- Step 01→02 execution:
  - mental-state operationalization,
  - initial observation-model construction,
  - model visualization artifact collection.
- Data collection stage:
  - variable-level collection schema preview,
  - pseudodata synthesis with configurable points/missingness/seed,
  - manual CSV upload fallback.
- Iterative PHOENIX cycle trigger:
  - readiness/network/impact,
  - Step-03 treatment-target handoff,
  - Step-04 updated model,
  - Step-05 HAPA intervention,
  - run summaries and stage logs.
- Realtime streaming logs via SSE for every background job.

## Directory Layout

```text
frontend/
├── app.py
├── phoenix_frontend/
│   ├── config.py
│   ├── routes/
│   │   ├── ui.py
│   │   └── api.py
│   ├── services/
│   │   ├── job_manager.py
│   │   ├── phoenix_service.py
│   │   ├── pseudodata.py
│   │   └── session_store.py
│   ├── static/
│   └── templates/
└── workspace/
    └── sessions/   (runtime artifacts, per session)
```

## Run

```bash
python frontend/app.py
```

Or launch through the orchestrator:

```bash
python evaluation/00_pipeline_orchestration/run_pipeline.py --ui
```

Open:

- [http://127.0.0.1:5050](http://127.0.0.1:5050)

## Environment Notes

- `OPENAI_API_KEY` must be available for LLM-enabled runs.
- LLM execution is enabled by default in the UI; each run form includes a `Disable LLM` toggle.
- Optional overrides:
  - `PHOENIX_REPO_ROOT`
  - `PHOENIX_FRONTEND_WORKSPACE`
  - `PHOENIX_PYTHON_EXE`
  - `PHOENIX_DISABLE_LLM` (`true/false`, enforced globally in UI)

## Runtime Data Safety

- Frontend writes only under `frontend/workspace/sessions/<session_id>/`.
- No ontology structure/content is modified.
- Session files are isolated and can be inspected independently for reproducibility.
