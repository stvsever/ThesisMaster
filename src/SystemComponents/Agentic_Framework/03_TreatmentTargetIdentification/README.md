# Step 03 — Treatment Target Identification

## Main Script

- `01_prepare_targets_from_impact.py`

## Responsibilities

- merges evidence from impact, readiness, network diagnostics, free text, and mapped observation-model context,
- generates ranked treatment target candidates,
- prepares Step-04 updated-model artifacts,
- supports ontology-aware breadth-first candidate exploration,
- supports optional actor-critic refinement,
- supports optional `--hard-ontology-constraint`.

## Key Outputs

- `top_treatment_target_candidates.csv`
- `step03_target_selection.json`
- `step04_updated_observation_model.json`
- `step04_nomothetic_idiographic_fusion.json`
- `step04_fusion_predictor_rankings.csv`
- `step03_guardrail_review.json`
- `step04_guardrail_review.json`
- `predictor_parent_feasibility_top30.json`
- `visuals/updated_model_fused_heatmap.png`
- `visuals/updated_model_fused_bipartite.png`

## Technical Dependencies

- shared prompt/runtime modules in `src/SystemComponents/Agentic_Framework/shared/`
- prompt registry in `src/SystemComponents/Agentic_Framework/prompts/`
