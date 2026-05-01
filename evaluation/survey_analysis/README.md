# PHOENIX Survey Analysis

This folder runs the Phase 2 comparison between HCP survey outputs and PHOENIX
system outputs. The current design does not use a second human expert judging
phase. HCPs first produce reference outputs in Qualtrics; PHOENIX later receives
the same case inputs; a double-blind LLM judge compares both anonymous outputs.

## Core Design

The five judged tasks mirror the live Qualtrics PRE survey:

| Part | Task | Compared output shape |
| --- | --- | --- |
| 1 | Identify symptom labels | `{"items": [{"label": "..."}]}` |
| 2 | Generate modifiable treatment-option labels | `{"items": [{"label": "..."}]}` |
| 3 | Rank five treatment options | `{"ranking": [{"rank": 1, "option_id": "BO-1"}]}` |
| 4 | Select exactly six EMA items | `{"selected_options": ["..."]}` |
| 5 | Write a mobile coaching message | `{"message": "..."}` |

Only these minimal shapes are shown as Output A and Output B. All shared case
information, such as vignette, standardised symptoms, network weights,
candidate EMA items, treatment goal, barrier, and coping strategy, is provided
separately in the prompt context. This prevents the judge from inferring source
identity from richer PHOENIX metadata.

## Data Flow

```
Qualtrics CSV
  -> parsing/qualtrics_parser.py
  -> data/02_parsed/hcp_outputs.json

Exact Qualtrics-matched PHOENIX inputs
  -> evaluation/phoenix_outputs/data/inputs/qualtrics_case_inputs.json

PHOENIX canonical outputs
  -> data/03_system/system_outputs.json

Shared judge context
  -> data/01_raw/case_contexts.json

Double-blind LLM judge
  -> data/04_judgments/judgments_long.csv
  -> data/04_judgments/raw/<part>/case_<case>_run_<run>.json

Per-part analysis
  -> results/part1_prompt/ ... results/part5_prompt/

Cross-part synthesis
  -> results/synthesis/
```

Use `evaluation/phoenix_outputs/run.py` to extract exact PHOENIX inputs from
the Qualtrics source and to canonicalize actual PHOENIX outputs.

## Judge Scale

The judge gives one signed comparative score per dimension:

```text
-9 = Output B decisively better than Output A
-6 = Output B strongly better
-3 = Output B modestly better
  0 = no meaningful difference / tie
 +3 = Output A modestly better
 +6 = Output A strongly better
 +9 = Output A decisively better
```

The runner unblinds this into `score = PHOENIX - HCP`:

- `score > 0`: PHOENIX preferred;
- `score < 0`: HCP preferred;
- `score = 0`: no meaningful difference.

`PROMPT_VERSION` is `2026-05-01-v3-signed-comparison`.

## Evaluation Dimensions

Dimensions are part-specific. Each dimension has a definition, rationale, and
comparative anchors in `llm_as_judge/dimensions.py`.

| Part | Dimension keys |
| --- | --- |
| 1 | `task_adherence_label_format`, `complaint_coverage`, `symptom_boundary_validity`, `granularity_resolution`, `nonredundancy_discriminability`, `clinical_interoperability`, `ema_measurability` |
| 2 | `task_adherence_label_format`, `modifiability_actionability`, `symptom_relevance`, `causal_plausibility`, `daily_ema_feasibility`, `symptom_option_separation`, `option_diversity_complementarity`, `label_precision` |
| 3 | `ranking_validity_completeness`, `network_weight_alignment`, `current_state_integration`, `edge_direction_interpretation`, `top_target_defensibility`, `modifiability_feasibility_weighting`, `rank_order_coherence` |
| 4 | `valid_candidate_selection`, `target_item_mapping_accuracy`, `coverage_balance`, `measurement_concreteness`, `directness_specificity`, `dynamic_informativeness`, `monitoring_burden_parsimony`, `feedback_value_for_coaching` |
| 5 | `message_format_direct_address`, `treatment_goal_alignment`, `barrier_responsiveness`, `action_specificity_feasibility`, `behaviour_change_potential`, `tone_empathy_professionalism`, `mobile_concision_readability`, `personalisation_specificity`, `clinical_safety_nonjudgment` |

## Statistical Analysis

For each part and dimension, the primary model is:

```text
score ~ 1 + (1 | case_id) + (1 | judge_run)
```

The intercept is the estimated signed PHOENIX-HCP preference. The report gives
the intercept, 95% CI, raw p-value, Holm-corrected p-value within part,
one-sample Cohen's d, preference split, and one-sample TOST equivalence around
zero with `delta = +/-1` signed point.

The synthesis divides scores by 9 and fits a grand-mean model:

```text
score_norm ~ 1 + (1 | case_id) + (1 | judge_run) + (1 | dimension)
```

Per-part follow-ups use the same one-sample signed structure. Global and
per-part TOST use `delta = +/-0.10` on the normalised `[-1,+1]` scale.

## Real-Mode Inputs

HCP outputs come from the Qualtrics export, for example:

```text
evaluation/qualtrics/data/01_raw/Masterproef_May 1, 2026_15.25.csv
```

The parser reads the current columns such as:

```text
HCP03_C03_PART1_1 ... HCP03_C03_PART1_6
HCP03_C03_PART2_1 ... HCP03_C03_PART2_5
HCP03_C03_PART3_1 ... HCP03_C03_PART3_5
HCP03_C03_PART4
HCP03_C03_PART5
hcp
```

PHOENIX/system outputs must be canonicalized to:

```text
evaluation/survey_analysis/data/03_system/system_outputs.json
```

The exact PHOENIX inputs and judge contexts are prepared from:

```bash
python evaluation/phoenix_outputs/run.py extract-inputs
python evaluation/phoenix_outputs/run.py sync-to-analysis --skip-system-outputs
```

After the actual PHOENIX run:

```bash
python evaluation/phoenix_outputs/run.py canonicalize \
  --raw /path/to/raw_phoenix_outputs.json
python evaluation/phoenix_outputs/run.py sync-to-analysis
```

## CLI Usage

Pseudo end-to-end run, no API key:

```bash
python evaluation/survey_analysis/pipeline.py --mode pseudo
```

Real HCP parse plus pseudo judge for software validation on the current C03
example row:

```bash
python evaluation/phoenix_outputs/run.py prepare-fixture-analysis
python evaluation/survey_analysis/pipeline.py \
  --mode real \
  --judge pseudo \
  --cases C03 \
  --parts part1 part2 part3 part4 part5
```

Real LLM-as-judge run after all HCP and PHOENIX outputs are present:

```bash
export OPENROUTER_API_KEY=...
python evaluation/survey_analysis/pipeline.py \
  --mode real \
  --judge openrouter \
  --n-runs 5 \
  --system-outputs evaluation/survey_analysis/data/03_system/system_outputs.json
```

Re-run only analysis from an existing `judgments_long.csv`:

```bash
python evaluation/survey_analysis/pipeline.py \
  --mode real \
  --skip-parse \
  --skip-judge
```

Smoke test:

```bash
python -m unittest evaluation/survey_analysis/tests/test_smoke.py
```

## Notes

- Real mode is strict: requested cases and parts must exist for both HCP and
  PHOENIX outputs. This prevents accidental comparison against empty JSON.
- Generated analysis data under `data/` and generated figures under `results/`
  are ignored by git, except folder READMEs.
- The protected live Qualtrics image paths under
  `evaluation/qualtrics/01_HCPs_PRE/...` are not modified by this pipeline.

