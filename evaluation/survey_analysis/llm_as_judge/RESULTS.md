# PHOENIX Engine Evaluation Results

This document summarises the current PHOENIX engine evaluation run as a
research-paper results section. The present run is a software-validation run:
it uses LLM-generated pseudo HCP outputs and a real OpenRouter LLM judge to
validate the complete double-blind judging, mixed-model analysis, and reporting
workflow before the final Qualtrics HCP dataset is complete.

## Evaluation Sample

The evaluation covered 10 clinical cases across the five Qualtrics-matched
tasks: symptom-label generation, modifiable treatment-option generation,
treatment-target ranking, EMA item selection, and mobile coaching-message
generation. Each anonymous output was rated independently on a bipolar −10 to
+10 absolute quality scale across part-specific clinical and methodological
dimensions. Three independent judge runs were used per case, part, and source,
producing 2,340 long-format ratings.

| Component | Value |
| --- | ---: |
| Clinical cases | 10 |
| Survey parts | 5 |
| Evaluation dimensions | 39 configured, 38 unique dimension keys |
| Judge runs per source output | 3 |
| OpenRouter judge calls | 300 |
| Long-format ratings | 2,340 |
| Paired PHOENIX-HCP cells | 1,170 |
| Quality scale | −10 to +10 integer scale, 0 = acceptable |
| Primary model | `quality_score ~ entity_ec + (1 \| case_id) + (1 \| judge_run)` |
| Equivalence margin | ±1.5 quality points |

## Primary Outcome

Across all parts and dimensions, PHOENIX was rated higher than the HCP
comparator. The pooled mixed-effects model estimated a PHOENIX-HCP quality gap
of Δ = +3.68 points on the −10 to +10 scale, 95% CI [+3.35, +4.01], p < .001.
The global TOST analysis did not support practical equivalence,
p<sub>TOST</sub> = 1.000, because the observed effect exceeded the predefined
equivalence band in favour of PHOENIX.

<p align="center">
  <img src="../results/synthesis/visuals/synthesis_part_forest.png" alt="Cross-part PHOENIX versus HCP quality effects" width="900">
</p>

**Figure 1. Cross-part PHOENIX versus HCP quality effects.** Points show
mixed-model PHOENIX-HCP quality gaps with 95% confidence intervals. Positive
values favour PHOENIX. **Note.** Entity was effect coded as PHOENIX = +0.5 and
HCP = −0.5; per-part p-values are Holm corrected.

## Part-Level Effects

PHOENIX showed positive estimated effects in all five survey parts. The largest
advantage was observed for network-informed treatment-target ranking, followed
by modifiable treatment-option generation. Symptom labels and coaching messages
showed smaller but still positive effects; symptom labels were statistically
higher for PHOENIX while remaining practically equivalent within the ±1.5-point
margin.

| Part | PHOENIX M | HCP M | PHOENIX-HCP gap | 95% CI | Holm p | TOST |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| Symptom labels | +4.41 | +3.61 | +0.80 | [+0.28, +1.32] | .0037 | Equivalent |
| Treatment options | +6.16 | −0.14 | +6.30 | [+5.69, +6.91] | < .001 | Not equivalent |
| Target ranking | +8.19 | −1.41 | +9.60 | [+8.95, +10.26] | < .001 | Not equivalent |
| EMA items | +6.33 | +5.07 | +1.27 | [+0.74, +1.79] | < .001 | Not equivalent |
| Coaching message | +4.01 | +2.87 | +1.14 | [+0.42, +1.85] | .0037 | Not equivalent |

<p align="center">
  <img src="../results/synthesis/visuals/synthesis_gap_heatmap.png" alt="Dimension-level PHOENIX-HCP quality gaps across survey parts" width="950">
</p>

**Figure 2. Dimension-level PHOENIX-HCP quality gaps.** Heatmap cells show
estimated quality gaps by survey part and evaluation dimension. **Note.** Warm
positive cells indicate dimensions where PHOENIX was rated higher; cooler cells
indicate dimensions where HCP was rated higher.

<p align="center">
  <img src="../results/synthesis/visuals/synthesis_part_raincloud.png" alt="Quality score distributions by source and survey part" width="950">
</p>

**Figure 3. Quality score distributions by source and survey part.** Raincloud
plots show the full distribution of judge ratings for PHOENIX and HCP outputs.
**Note.** The dashed reference line at 0 marks the acceptable-quality baseline.

<p align="center">
  <img src="../results/synthesis/visuals/synthesis_tost.png" alt="TOST equivalence summary across survey parts" width="850">
</p>

**Figure 4. Equivalence-test summary.** The TOST panels evaluate whether
PHOENIX and HCP outputs fall inside the predefined ±1.5-point equivalence
margin. **Note.** Non-equivalence in Parts 2, 3, 4, and 5 reflects positive
PHOENIX effects outside the equivalence band, not HCP superiority.

## Dimension-Level Results

Part 1 showed a modest PHOENIX advantage. The clearest dimension-level gain was
task adherence and label format, Δ = +2.67, Holm p < .001. Other symptom-label
dimensions were small and mostly equivalent, indicating that the optimised
PHOENIX labels were concise and clinically usable without over-expanding beyond
the requested label-only format.

Part 2 showed the strongest improvement in clinical construct validity.
PHOENIX outperformed HCP especially on symptom-option separation,
Δ = +13.43; modifiability and actionability, Δ = +9.83; causal plausibility,
Δ = +7.73; option diversity, Δ = +5.80; symptom relevance, Δ = +5.53; label
precision, Δ = +4.00; and daily EMA feasibility, Δ = +2.07. This indicates
that the PHOENIX treatment-option set was judged as genuinely modifiable rather
than merely re-labelling symptoms as treatment targets.

Part 3 showed the largest PHOENIX advantage after adding numeric pseudo-network
context to the validation data. PHOENIX strongly outperformed HCP on
network-weight alignment, Δ = +13.73; current-state integration, Δ = +13.30;
rank-order coherence, Δ = +12.40; top-target defensibility, Δ = +12.17; edge
direction interpretation, Δ = +7.33; and modifiability-feasibility weighting,
Δ = +6.47. These effects confirm that the judge can recover the expected
advantage when structured network and EMA evidence is available.

Part 4 showed a smaller but consistent PHOENIX advantage. The largest gains
were observed for feedback value for coaching, Δ = +2.33; monitoring burden and
parsimony, Δ = +2.13; directness and specificity, Δ = +1.93; target-item
mapping accuracy, Δ = +1.47; and dynamic informativeness, Δ = +1.40. Both
sources achieved perfect valid-candidate selection, confirming that the
candidate-list contract is functioning.

Part 5 showed a positive PHOENIX effect while preserving phone-ready tone.
The strongest gains were behaviour-change potential, Δ = +2.70; personalisation
specificity, Δ = +2.47; treatment-goal alignment, Δ = +2.27; barrier
responsiveness, Δ = +2.00; and action specificity, Δ = +0.93. HCP remained
similar on tone and concision, indicating that PHOENIX's advantage came from
clinical targeting and behaviour-change structure rather than simply longer or
more technical messages.

<p align="center">
  <img src="../results/part1_prompt/visuals/part1_prompt_effect_forest.png" alt="Part 1 symptom-label dimension effects" width="900">
</p>

**Figure 5A. Part 1 symptom-label dimension effects.** Effects are
PHOENIX-HCP quality gaps with 95% confidence intervals.

<p align="center">
  <img src="../results/part2_prompt/visuals/part2_prompt_effect_forest.png" alt="Part 2 treatment-option dimension effects" width="900">
</p>

**Figure 5B. Part 2 treatment-option dimension effects.** The largest gains
occurred on dimensions testing whether outputs were genuinely modifiable,
clinically causal, and distinct from symptom labels.

<p align="center">
  <img src="../results/part3_prompt/visuals/part3_prompt_effect_forest.png" alt="Part 3 treatment-target ranking dimension effects" width="900">
</p>

**Figure 5C. Part 3 treatment-target ranking dimension effects.** The largest
gains occurred on network alignment, current-state integration, and rank-order
coherence.

<p align="center">
  <img src="../results/part4_prompt/visuals/part4_prompt_effect_forest.png" alt="Part 4 EMA item-selection dimension effects" width="900">
</p>

**Figure 5D. Part 4 EMA item-selection dimension effects.** PHOENIX showed its
clearest advantages on clinically useful monitoring value and parsimony while
matching HCP on valid-candidate selection.

<p align="center">
  <img src="../results/part5_prompt/visuals/part5_prompt_effect_forest.png" alt="Part 5 mobile coaching-message dimension effects" width="900">
</p>

**Figure 5E. Part 5 mobile coaching-message dimension effects.** PHOENIX gains
were concentrated in personalisation, treatment-goal alignment, and expected
behaviour-change value.

## Supplementary Reliability and Sensitivity

The three-run design showed adequate judge stability for a software-validation
run. Global mean ICC(2,1) was 0.684 across all part × dimension strata.
Reliability was highest for target ranking, ICC = 0.920, and treatment options,
ICC = 0.823. The lower Part 1 ICC, ICC = 0.484, reflects smaller between-source
effects and greater stochastic variation around otherwise acceptable symptom
labels. Confidence-weighted sensitivity analyses changed part-level effects by
at most 0.097 quality points, indicating that the primary conclusions were not
driven by low-confidence ratings.

| Supplementary diagnostic | Value |
| --- | ---: |
| Global mean ICC(2,1) | 0.684 |
| Highest part-level ICC | Part 3 = 0.920 |
| Lowest part-level ICC | Part 1 = 0.484 |
| Maximum confidence-weighted shift | 0.097 points |
| Grand mean case × part gap | +3.82 |
| Case × part cells with positive PHOENIX gap | 44 / 50 |

<p align="center">
  <img src="../results/supplementary/visuals/supplementary_overview_dashboard.png" alt="Supplementary reliability, calibration, sensitivity, and heterogeneity dashboard" width="950">
</p>

**Figure 6. Supplementary evaluation diagnostics.** Panel A shows judge-run
ICC(2,1) by part with dimension-level points. Panel B shows scale-use bands for
PHOENIX and HCP outputs. Panel C compares unweighted and confidence-weighted
PHOENIX-HCP gaps. Panel D shows case-by-part heterogeneity. **Note.** Figure
content is title-free by design; figure interpretation is provided in the
caption text.

<p align="center">
  <img src="../results/supplementary/visuals/suppB_calibration_diagnostics.png" alt="Score calibration diagnostics" width="950">
</p>

**Figure 7. Score calibration diagnostics.** The calibration plot verifies that
the judge used the bipolar scale rather than collapsing all ratings near zero.
**Note.** PHOENIX showed high-quality-range compression in Part 3, consistent
with near-optimal recovery of the supplied pseudo-network priority structure.

## Statistical Conclusion

This validation run supports the full PHOENIX double-blind evaluation workflow.
The current PHOENIX output artifact outperformed the HCP comparator globally
and in every survey part under real LLM judging. The strongest advantages were
observed in tasks where structured computational reasoning is central:
constructing modifiable treatment options and ranking treatment targets from
network and EMA evidence. Smaller but positive effects were observed for
symptom labels, EMA item selection, and mobile coaching messages. Supplementary
analyses showed that the three-run judge design was stable and that conclusions
were robust to confidence weighting.

The present estimates should be interpreted as software-validation results,
not final thesis findings. Final inference should be rerun after the complete
Qualtrics HCP dataset and production PHOENIX outputs are available.

## Reproduction

The current run was generated with:

```bash
set -a; source .env; set +a
rm -f evaluation/survey_analysis/data/04_judgments/judgments_long.csv
rm -rf evaluation/survey_analysis/data/04_judgments/raw
python3 evaluation/survey_analysis/pipeline.py \
  --mode pseudo \
  --judge openrouter \
  --n-runs 3 \
  --log-level INFO
```
