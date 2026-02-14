# PHOENIX Architecture Overview

## Executive Summary

PHOENIX is a multi-agent research system that integrates ontology-guided reasoning, idiographic time-series evidence, and intervention translation into a reproducible pipeline.

Core sequence:
1. operationalize complaints,
2. construct initial observation model,
3. quantify dynamic impact,
4. refine targets and updated model,
5. generate HAPA-aligned intervention,
6. preserve iterative lineage for subsequent cycles.

## Multi-Agent Flow

```mermaid
flowchart TD
    A["Step 01 Operationalization Agent"] --> B["Step 02 Initial Model Agent"]
    B --> Bc["Step 02 Critic"]
    Bc -->|"Revise max 2"| B
    Bc -->|"Pass"| C["Data Collection (Current Model)"]
    C --> D["Step 03 Readiness and Network Analysis"]
    D --> E["Momentary Impact"]
    E --> F["Step 03 Target Actor"]
    F --> Fc["Step 03 Critic"]
    Fc -->|"Revise max 2"| F
    Fc -->|"Pass"| G["Step 04 Updated Model Actor"]
    G --> Gc["Step 04 Critic"]
    Gc -->|"Revise max 2"| G
    Gc -->|"Pass"| H["Step 05 Intervention Actor"]
    H --> Hc["Step 05 Critic"]
    Hc -->|"Revise max 2"| H
    Hc -->|"Pass"| I["Run Artifacts and Reports"]
    I --> J["History Ledger"]
    J --> K["Data Collection (Updated Model)"]
    K --> D
```

## Design Principles

- **Breadth-first solution search:** explore sibling predictor domains before deepening.
- **Nomothetic × idiographic fusion:** combine mapping priors and profile-specific evidence.
- **Explicit data-collection loop:** each initial/updated model is followed by collection before re-analysis.
- **Guardrail reviews:** critic agents score quality and request bounded revisions.
- **Feasibility grounding:** parent-domain suitability signals are injected into model reviews.
- **Controlled ontology strictness:** default ontology-driven mode with optional hard constraints.

## Output Guarantees

Each stage emits:
- `stage.log` (human-readable)
- `stage_events.jsonl` (event trace)
- `stage_trace.json` (summary, timings, counts)

Major artifacts are schema-validated to preserve backward-compatible contracts.

Iterative runs now include cycle-specific pseudodata regeneration from prior Step-04/Step-05 outputs before re-entering readiness/network analysis.
