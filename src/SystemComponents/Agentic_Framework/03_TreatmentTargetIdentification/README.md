# Step 03: Treatment Target Identification

## Main Script

- `01_prepare_targets_from_impact.py`
  - Integrates evidence from:
    - momentary impact outputs,
    - readiness diagnostics,
    - network metrics,
    - free-text complaint/person/context,
    - mapped initial observation model,
    - predictor ontology candidate sets.
  - Runs structured target-selection reasoning (LLM mode or deterministic fallback).
  - Enforces breadth-first candidate expansion before subtree deepening.
  - Applies readiness-weighted nomothetic+idiographic fusion for Step-04 refinement.
  - Adds actor-critic guardrail validation loops (max 2 refinement iterations by default).
  - Supports `--hard-ontology-constraint` to force predictor outputs onto ontology-matched nodes.
  - Generates updated-model visuals for research communication.
  - Produces:
    - `top_treatment_target_candidates.csv`
    - `step03_target_selection.json`
    - `step04_updated_observation_model.json`
    - `step04_nomothetic_idiographic_fusion.json`
    - `step04_fusion_edges.csv`
    - `step04_fusion_matrix.csv`
    - `step04_fusion_predictor_rankings.csv`
    - `step03_guardrail_review.json`
    - `step04_guardrail_review.json`
    - `predictor_parent_feasibility_top30.json`
    - `visuals/updated_model_fused_heatmap.png`
    - `visuals/updated_model_fused_bipartite.png`
    - ontology-candidate trace files and profile-level prompt/validation traces.

## Prompt and LLM Modularity

- Prompts are centralized under `src/SystemComponents/Agentic_Framework/prompts/`.
- Reusable runtime modules live in `src/SystemComponents/Agentic_Framework/shared/`:
  - `llm_runtime.py` (structured output + auto-repair),
  - `token_budget.py` (prompt token budgeting),
  - `prompt_loader.py` (prompt loading/rendering),
  - `target_refinement.py` (BFS candidate planning + nomothetic/idiographic fusion),
  - `updated_model_visualization.py` (Step-04 fusion visual outputs).
