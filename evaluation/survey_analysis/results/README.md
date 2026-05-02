# Survey Analysis Results

This folder stores generated PHOENIX LLM-as-judge analysis artefacts. Generated
figures and CSV files are reproducible and ignored by git; this README defines
the expected structure.

Current artefacts use bipolar -10 to +10 absolute quality ratings, where 0 is
the acceptable clinical baseline. PHOENIX versus HCP effects are estimated
after unblinding as PHOENIX - HCP quality gaps.

| Folder | Meaning |
| --- | --- |
| `part1_prompt/` | Symptom-label evaluation |
| `part2_prompt/` | Modifiable treatment-option evaluation |
| `part3_prompt/` | Treatment-target ranking evaluation |
| `part4_prompt/` | EMA item-selection evaluation |
| `part5_prompt/` | Mobile coaching-message evaluation |
| `synthesis/` | Cross-part PHOENIX versus HCP synthesis |
| `supplementary/` | Judge-run stability and sensitivity diagnostics |

Each per-part folder contains:

- `report/*_report.txt`: textual statistical report;
- `report/*_summary.csv`: tidy dimension-level estimates;
- `visuals/*_quality_raincloud.png`: quality distributions by entity;
- `visuals/*_effect_forest.png`: PHOENIX - HCP quality gaps with 95% CI;
- `visuals/*_tost_equivalence.png`: equivalence-test panel.

The synthesis folder contains cross-part forest, raincloud, heatmap, and TOST
figures. The supplementary folder contains CSV diagnostics plus title-free
publication figures:

- `supplementary_overview_dashboard.png`: 2 by 2 overview of ICC reliability,
  scale use, confidence-weighted sensitivity, and case heterogeneity;
- `suppA_icc_stability.png`: judge-run ICC stability by part and dimension;
- `suppB_calibration_diagnostics.png`: floor, ceiling, and score-distribution diagnostics;
- `suppC_sensitivity_forest.png`: confidence-weighted sensitivity forest;
- `suppD_case_heterogeneity.png`: case by part PHOENIX - HCP gap heatmap.
