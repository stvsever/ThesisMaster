# PHOENIX Prompt Inventory (Full Engine)

This inventory centralizes known prompt sources across the Agentic Framework.

## Registry-backed prompts (active)

- `step02_initial_model_system.md`
- `step02_initial_model_user_template.md`
- `step02_initial_model_critic_system.md`
- `step02_initial_model_critic_user_template.md`
- `step03_target_selection_system.md`
- `step03_target_selection_user_template.md`
- `step03_target_selection_critic_system.md`
- `step03_target_selection_critic_user_template.md`
- `step04_observation_update_system.md`
- `step04_observation_update_user_template.md`
- `step04_observation_update_critic_system.md`
- `step04_observation_update_critic_user_template.md`
- `step05_hapa_intervention_system.md`
- `step05_hapa_intervention_user_template.md`
- `step05_hapa_intervention_critic_system.md`
- `step05_hapa_intervention_critic_user_template.md`

## Inline prompt locations (to externalize progressively)

- `src/SystemComponents/Agentic_Framework/02_ConstructionInitialObservationModel/utils/01_construct_observation_model.py`
  - repair prompt blocks
- `src/SystemComponents/Agentic_Framework/02_ConstructionInitialObservationModel/utils/02_map_observation_model_to_ontology.py`
  - mapping/adjudication prompt blocks
  - JSON repair prompt blocks
- `src/SystemComponents/Agentic_Framework/02_ConstructionInitialObservationModel/utils/helpers/generate_HyDe_based_predictor_ranks.py`
  - HyDe decomposition prompt blocks

## Policy

- Step-03/04/05 prompts are versioned via `prompts_manifest.json`.
- Inline prompts above remain backward-compatible for current thesis scripts and are tracked here for phased migration.
