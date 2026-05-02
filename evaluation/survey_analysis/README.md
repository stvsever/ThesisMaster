# PHOENIX Survey Analysis

This folder runs the Phase 2 evaluation of PHOENIX against healthcare expert
outputs collected through the Qualtrics PRE survey. The design no longer uses a
second human expert judging phase. HCPs produce outputs for the 10 clinical
cases, PHOENIX receives the same inputs, and an LLM judge rates anonymous
outputs under a double-blind contract.

## Core Design

The five judged tasks mirror the live Qualtrics survey:

| Part | Task | Judged output shape |
| --- | --- | --- |
| 1 | Identify symptom labels | `{"items": [{"label": "..."}]}` |
| 2 | Generate modifiable treatment-option labels | `{"items": [{"label": "..."}]}` |
| 3 | Rank five treatment options | `{"ranking": [{"rank": 1, "option_id": "BO-1"}]}` |
| 4 | Select six EMA items | `{"selected_options": ["..."]}` |
| 5 | Write a mobile coaching message | `{"message": "..."}` |

Only these minimal source-symmetric shapes are judged. The shared case context
is supplied separately: vignette, standardized symptoms, treatment options,
network edge weights, monitoring summary, candidate EMA items, treatment goal,
barrier, and coping strategy. This prevents the judge from identifying the
source from richer PHOENIX metadata.

## Data Flow

```text
Qualtrics HCP CSV
  -> parsing/qualtrics_parser.py
  -> data/02_parsed/hcp_outputs.json

Exact Qualtrics-matched PHOENIX inputs
  -> evaluation/phoenix_outputs/data/inputs/qualtrics_case_inputs.json

PHOENIX canonical outputs
  -> data/03_system/system_outputs.json

Shared judge context
  -> data/01_raw/case_contexts.json

Double-blind absolute-quality LLM judge
  -> data/04_judgments/judgments_long.csv
  -> data/04_judgments/raw/

Primary analysis
  -> results/part1_prompt/ ... results/part5_prompt/
  -> results/synthesis/

Supplementary diagnostics
  -> results/supplementary/
```

Use `evaluation/phoenix_outputs/run.py` to extract exact PHOENIX inputs from
the Qualtrics source, run or canonicalize PHOENIX outputs, and sync those
outputs into this analysis pipeline.

## Judge Method

The judge rates one anonymous output at a time. It never sees PHOENIX and HCP
side by side in the same call, which reduces direct preference bias, source
guessing, and style-based comparison effects. For each case, part, run, and
source, the judge receives the same task context plus `The Output`.

Quality scale:

```text
-10 = Clinically harmful or unusable
-5  = Substantially below acceptable quality
0   = Acceptable minimum clinical quality
+5  = Strong clinical quality
+10 = Exemplary clinical quality for the task
```

The default is three independent judge runs per `(case, part, source)` cell.
Three runs reduce stochastic rating noise while avoiding unnecessary token
cost. Stability across runs is quantified in the supplementary analysis.

## Evaluation Dimensions

Dimensions are part-specific. Each dimension has a definition, rationale, and
anchored examples spanning the bipolar -10..+10 scale in
`llm_as_judge/dimensions.py`.

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
quality_score ~ entity_ec + (1 | case_id) + (1 | judge_run)
```

`entity_ec` is effect coded as PHOENIX = +0.5 and HCP = -0.5. The coefficient
is the PHOENIX - HCP quality gap on the bipolar -10..+10 scale.

Each per-part report includes:

- PHOENIX and HCP means;
- PHOENIX - HCP mixed-model coefficient and 95% CI;
- raw and Holm-corrected p-values within part;
- Cohen's d on paired case-run differences;
- TOST equivalence with `delta = +/-1.5` quality points.

The synthesis fits the same entity-predictor model across all parts and runs
part-level follow-ups. Equivalence is evaluated on paired PHOENIX - HCP
difference scores.

## Supplementary Analyses

The supplementary module quantifies reliability and sensitivity without adding
new primary hypotheses:

- within-cell rating SD across the three judge runs;
- exact and within-one-point agreement across repeated ratings;
- SD and range of paired PHOENIX - HCP gaps;
- sign consistency of the PHOENIX - HCP gap;
- confidence-weighted sensitivity of effect estimates;
- ceiling and floor diagnostics for the bipolar -10..+10 quality scale.

Outputs are saved under `results/supplementary/` as CSV files and title-free
publication figures; figure titles and `Note.` captions are written in the
Markdown results report.

## Real-Mode Inputs

HCP outputs come from the Qualtrics export, for example:

```text
evaluation/qualtrics/data/01_raw/Masterproef_May 1, 2026_15.25.csv
```

The parser reads columns such as:

```text
HCP03_C03_PART1_1 ... HCP03_C03_PART1_6
HCP03_C03_PART2_1 ... HCP03_C03_PART2_5
HCP03_C03_PART3_1 ... HCP03_C03_PART3_5
HCP03_C03_PART4
HCP03_C03_PART5
hcp
```

PHOENIX outputs must be canonicalized to:

```text
evaluation/survey_analysis/data/03_system/system_outputs.json
```

Prepare exact PHOENIX inputs and judge contexts:

```bash
python evaluation/phoenix_outputs/run.py extract-inputs
python evaluation/phoenix_outputs/run.py sync-to-analysis --skip-system-outputs
```

Run the PHOENIX engine through OpenRouter:

```bash
export OPENROUTER_API_KEY=...
python evaluation/phoenix_outputs/run.py run-engine
python evaluation/phoenix_outputs/run.py quality-gate \
  --raw evaluation/phoenix_outputs/data/outputs/system_outputs_llm.json \
  --sync
```

Or canonicalize an externally produced PHOENIX output file:

```bash
python evaluation/phoenix_outputs/run.py canonicalize \
  --raw /path/to/raw_phoenix_outputs.json
python evaluation/phoenix_outputs/run.py sync-to-analysis
```

## CLI Usage

Pseudo end-to-end run, no API key:

```bash
python evaluation/survey_analysis/pipeline.py --mode pseudo --n-runs 3
```

Real HCP parse plus pseudo judge for the current C03 example row:

```bash
python evaluation/phoenix_outputs/run.py prepare-fixture-analysis
python evaluation/survey_analysis/pipeline.py \
  --mode real \
  --judge pseudo \
  --cases C03 \
  --parts part1 part2 part3 part4 part5 \
  --n-runs 3
```

Real LLM-as-judge run after all HCP and PHOENIX outputs are present:

```bash
export OPENROUTER_API_KEY=...
python evaluation/survey_analysis/pipeline.py \
  --mode real \
  --judge openrouter \
  --n-runs 3 \
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
  PHOENIX outputs.
- Generated analysis data under `data/` and generated figures under `results/`
  are ignored by git, except folder READMEs.
- The protected live Qualtrics image paths under
  `evaluation/qualtrics/01_HCPs_PRE/...` are not modified by this pipeline.
