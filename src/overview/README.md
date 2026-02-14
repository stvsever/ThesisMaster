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
    Bc -->|"Pass"| C["Readiness and Network Analysis"]
    C --> D["Momentary Impact"]
    D --> E["Step 03 Target Actor"]
    E --> Ec["Step 03 Critic"]
    Ec -->|"Revise max 2"| E
    Ec -->|"Pass"| F["Step 04 Updated Model Actor"]
    F --> Fc["Step 04 Critic"]
    Fc -->|"Revise max 2"| F
    Fc -->|"Pass"| G["Step 05 Intervention Actor"]
    G --> Gc["Step 05 Critic"]
    Gc -->|"Revise max 2"| G
    Gc -->|"Pass"| H["Run Artifacts and Reports"]
    H --> I["History Ledger"]
    I --> F
    I --> G
```

## Design Principles

- **Breadth-first solution search:** explore sibling predictor domains before deepening.
- **Nomothetic × idiographic fusion:** combine mapping priors and profile-specific evidence.
- **Guardrail reviews:** critic agents score quality and request bounded revisions.
- **Feasibility grounding:** parent-domain suitability signals are injected into model reviews.
- **Controlled ontology strictness:** default ontology-driven mode with optional hard constraints.

## Output Guarantees

Each stage emits:
- `stage.log` (human-readable)
- `stage_events.jsonl` (event trace)
- `stage_trace.json` (summary, timings, counts)

Major artifacts are schema-validated to preserve backward-compatible contracts.
