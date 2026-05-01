# PHOENIX Survey Analysis

This folder contains the Phase 2 evaluation pipeline for comparing PHOENIX
system outputs against healthcare professional (HCP) outputs. The current
design no longer uses a second human expert POST phase. HCPs generate the
reference outputs in Qualtrics once; PHOENIX later receives the same case
inputs and produces outputs in the same canonical response shapes. A
double-blind LLM-as-judge then compares the two anonymous outputs per case,
part, and evaluation dimension.

## Current Design

The live Qualtrics PRE survey collects one HCP output per case for the five
main tasks:

| Part | Survey task | Canonical output |
| --- | --- | --- |
| 1 | Identify 3..6 symptom labels | `{"items": [{"label", "description"}]}` |
| 2 | Generate 3..5 modifiable treatment-option labels | `{"items": [{"label"}]}` |
| 3 | Rank five standardised treatment options | `{"ranking": [{"rank", "option_id"}]}` |
| 4 | Select exactly six concrete EMA items | `{"selected_options": [...], "note": ...}` |
| 5 | Write a 2..4 sentence mobile coaching message | `{"message": "...", "hapa_phase": ...}` |

The judge sees only `Output A` and `Output B` in these canonical forms. A
deterministic blinding seed decides whether PHOENIX or HCP is A for each
`(case_id, part, judge_run)` cell. The raw unblinding key is saved, but never
shown in the prompt.

## Signed Judge Scale

The judge makes one comparative score per dimension:

```
-9 = Output B is decisively better
-6 = Output B is strongly better
-3 = Output B is modestly better
 0 = no meaningful difference / tie
+3 = Output A is modestly better
+6 = Output A is strongly better
+9 = Output A is decisively better
```

The runner unblinds this to a PHOENIX-vs-HCP score:

- positive `score` = PHOENIX preferred;
- negative `score` = HCP preferred;
- zero `score` = no meaningful difference.

This avoids the calibration problem of asking an LLM to give two independent
absolute ratings. The statistical model directly tests whether the signed
preference differs from zero.

## Data Flow

```
Qualtrics CSV
  -> parsing/qualtrics_parser.py
  -> data/02_parsed/hcp_outputs.json

PHOENIX runs on the same 10 cases
  -> data/03_system/system_outputs.json

Shared case context shown to both sources
  -> data/01_raw/case_contexts.json

Double-blind LLM judge
  -> data/04_judgments/judgments_long.csv
  -> data/04_judgments/raw/<part>/case_<case>_run_<run>.json

Per-part analysis
  -> results/part*/report/*_summary.csv
  -> results/part*/visuals/*_signed_preference_raincloud.png
  -> results/part*/visuals/*_effect_forest.png
  -> results/part*/visuals/*_tost_equivalence.png

Cross-part synthesis
  -> results/synthesis/report/synthesis_report.txt
  -> results/synthesis/visuals/synthesis_part_forest.png
  -> results/synthesis/visuals/synthesis_part_signed_raincloud.png
  -> results/synthesis/visuals/synthesis_heatmap.png
  -> results/synthesis/visuals/synthesis_tost.png
```

## Judge Model

Default real judge:

```
google/gemini-3.1-flash-lite-preview
```

served through OpenRouter's OpenAI-compatible API. The model can be changed
in `llm_as_judge/openrouter_client.py` or by wiring a model argument into the
runner. Pseudo mode uses a deterministic local pseudo judge and does not call
OpenRouter.

## Evaluation Dimensions

Dimensions are part-specific because the five tasks ask for different forms
of clinical reasoning.

| Part | Dimensions |
| --- | --- |
| 1 Symptoms | `complaint_coverage`, `symptom_boundary_validity`, `granularity_resolution`, `nonredundancy_discriminability`, `clinical_interoperability`, `ema_measurability` |
| 2 Treatment options | `modifiability_actionability`, `symptom_relevance`, `causal_plausibility`, `daily_ema_feasibility`, `symptom_option_separation`, `option_diversity_complementarity`, `label_precision` |
| 3 Target ranking | `network_weight_alignment`, `current_state_integration`, `edge_direction_interpretation`, `top_target_defensibility`, `modifiability_feasibility_weighting`, `rank_order_coherence` |
| 4 EMA items | `target_item_mapping_accuracy`, `coverage_balance`, `measurement_concreteness`, `directness_specificity`, `dynamic_informativeness`, `monitoring_burden_parsimony`, `feedback_value_for_coaching` |
| 5 Coaching message | `treatment_goal_alignment`, `barrier_responsiveness`, `action_specificity_feasibility`, `behaviour_change_potential`, `tone_empathy_professionalism`, `mobile_concision_readability`, `personalisation_specificity`, `clinical_safety_nonjudgment` |

Full definitions, rationales, and comparative anchors are in
`llm_as_judge/dimensions.py`; part-specific prompt templates are in
`llm_as_judge/prompts/`.

## Statistical Analysis

For every part and dimension:

```
score ~ 1 + (1 | case_id) + (1 | judge_run)
```

The intercept is the estimated mean PHOENIX-HCP signed preference. It is
positive when PHOENIX is preferred and negative when the HCP output is
preferred.

The analysis reports:

- intercept estimate and 95% CI;
- raw and Holm-corrected p-values within each part;
- one-sample Cohen's d for the signed scores;
- preference split: PHOENIX preferred, HCP preferred, tie;
- one-sample TOST equivalence around zero with `delta = +/-1` signed score
  point;
- a documented one-sample t-test/bootstrap fallback when the intercept-only
  mixed model returns degenerate variance estimates.

The cross-part synthesis normalises signed scores by dividing by 9 and fits:

```
score_norm ~ 1 + C(part) + (1 | case_id) + (1 | judge_run) + (1 | dimension)
```

Global and per-part TOST use `delta = +/-0.10` on the normalised `[-1,+1]`
scale.

## Required Real-Mode Inputs

### HCP outputs

Use the raw Qualtrics export:

```
evaluation/qualtrics/data/01_raw/Masterproef_May 1, 2026_15.25.csv
```

The parser understands the current column structure:

```
HCP03_C03_PART1_1 ... HCP03_C03_PART1_6
HCP03_C03_PART2_1 ... HCP03_C03_PART2_5
HCP03_C03_PART3_1 ... HCP03_C03_PART3_5
HCP03_C03_PART4
HCP03_C03_PART5
hcp
```

### PHOENIX outputs

Place production system outputs here:

```
evaluation/survey_analysis/data/03_system/system_outputs.json
```

Shape:

```json
{
  "C01": {
    "part1": {"items": [{"label": "insomnia", "description": "sleep onset difficulty"}]},
    "part2": {"items": [{"label": "evening screen time"}]},
    "part3": {"ranking": [{"rank": 1, "option_id": "BO-1"}]},
    "part4": {"selected_options": ["screen-free interval before bed"], "note": null},
    "part5": {"message": "Tonight, put your phone away at 21:30.", "hapa_phase": "intentional"}
  }
}
```

### Case contexts

For real OpenRouter judging, create:

```
evaluation/survey_analysis/data/01_raw/case_contexts.json
```

This should contain the inputs shown to both HCPs and PHOENIX: vignette,
standardised symptoms, treatment options, network/EMA summaries, Part 4
candidate EMA items, and Part 5 treatment-goal/barrier/coaching context.
Missing fields are tolerated but weaken the judge prompt, so production
analysis should fill them.

## Running

Pseudo mode, no API key:

```bash
python evaluation/survey_analysis/pipeline.py --mode pseudo
```

Real mode:

```bash
export OPENROUTER_API_KEY=...
python evaluation/survey_analysis/pipeline.py --mode real --n-runs 5
```

Subset while debugging:

```bash
python evaluation/survey_analysis/pipeline.py \
  --mode pseudo --cases C01 C02 C03 --parts part1 part5 --n-runs 5
```

Re-run only analysis from existing judge scores:

```bash
python evaluation/survey_analysis/pipeline.py --mode pseudo --skip-parse --skip-judge
```

Smoke test:

```bash
python -m unittest evaluation/survey_analysis/tests/test_smoke.py
```

## Notes

- The legacy live Qualtrics image paths under
  `evaluation/qualtrics/01_HCPs_PRE/...` are not touched by this pipeline.
- Generated data and results are reproducible pseudo-mode artefacts; real
  OpenRouter outputs should be versioned by `prompt_version`, model, and raw
  response JSON.
- `PROMPT_VERSION` is currently `2026-05-01-v2-signed-comparison`.
