# PHOENIX LLM-as-Judge Results

This file summarizes the latest software-validation run of the LLM-as-judge
pipeline. The current run uses pseudo HCP outputs and the pseudo judge, so it
validates the data flow, model formulas, figures, and reports rather than
constituting the final thesis result.

## Run Configuration

| Component | Setting |
| --- | --- |
| Cases | C01 to C10 |
| Parts | Five Qualtrics-matched tasks |
| Judge runs | 3 independent runs per case, part, and source |
| Total rows | 2,340 long-format ratings |
| Rating scale | 1 to 5 absolute quality |
| Primary model | `quality_score ~ entity_ec + (1 | case_id) + (1 | judge_run)` |
| Equivalence margin | +/-0.3 quality points |
| Multiplicity correction | Holm correction within part |
| Supplementary analyses | stability, confidence weighting, and scale-use diagnostics |

## Global Synthesis

| Estimate | Value |
| --- | ---: |
| PHOENIX - HCP quality gap | +0.3385 |
| 95% CI | [+0.2906, +0.3863] |
| p-value | < .001 |
| TOST equivalence | Not equivalent |

In this validation run, PHOENIX shows a positive global quality gap relative to
the pseudo HCP comparator. This pattern is expected because the pseudo judge is
parameterized to test whether the analysis recovers known PHOENIX advantages.

## Per-Part Effects

| Part | PHOENIX - HCP gap | 95% CI | Holm p | TOST |
| --- | ---: | --- | ---: | --- |
| Part 1: Symptom labels | +0.3571 | [+0.2430, +0.4713] | < .001 | Not equivalent |
| Part 2: Treatment options | +0.2625 | [+0.1576, +0.3674] | < .001 | Not equivalent |
| Part 3: Target ranking | +0.3000 | [+0.1842, +0.4158] | < .001 | Not equivalent |
| Part 4: EMA items | +0.3875 | [+0.2770, +0.4980] | < .001 | Not equivalent |
| Part 5: Coaching message | +0.3778 | [+0.2757, +0.4799] | < .001 | Not equivalent |

## Supplementary Stability

| Metric | Value |
| --- | ---: |
| Mean within-cell rating SD | 0.000 |
| Mean within-cell PHOENIX - HCP gap SD | 0.000 |
| Mean directional consistency | 1.000 |
| Maximum confidence-weighted change | 0.092 |

The pseudo judge is deterministic for a fixed cell, so stability metrics are
near-perfect in this validation run. With the real OpenRouter judge, these
same outputs quantify run-to-run stochastic stability across the three judge
runs.

## Generated Figures

| Figure | File |
| --- | --- |
| Per-part quality rainclouds | `results/part{N}_prompt/visuals/*_quality_raincloud.png` |
| Per-part PHOENIX - HCP forests | `results/part{N}_prompt/visuals/*_effect_forest.png` |
| Per-part TOST panels | `results/part{N}_prompt/visuals/*_tost_equivalence.png` |
| Cross-part forest | `results/synthesis/visuals/synthesis_part_forest.png` |
| Cross-part raincloud | `results/synthesis/visuals/synthesis_part_raincloud.png` |
| Cross-part heatmap | `results/synthesis/visuals/synthesis_gap_heatmap.png` |
| Supplementary stability dashboard | `results/supplementary/visuals/supplementary_stability_dashboard.png` |
| Supplementary sensitivity dashboard | `results/supplementary/visuals/supplementary_sensitivity_dashboard.png` |
| Supplementary dimension stability heatmap | `results/supplementary/visuals/supplementary_dimension_stability_heatmap.png` |

## Reproduction

```bash
python evaluation/survey_analysis/pipeline.py --mode pseudo --n-runs 3
```

For the final real-data run, replace pseudo HCP data with the completed
Qualtrics export, use the PHOENIX output flow under `evaluation/phoenix_outputs`,
and run the pipeline with `--mode real --judge openrouter --n-runs 3`.
