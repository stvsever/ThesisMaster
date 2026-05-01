# Survey Analysis Results

This folder stores generated LLM-as-judge analysis artefacts.

Current v2 artefacts use signed PHOENIX-vs-HCP preference scores:

- negative scores favour the HCP output;
- positive scores favour PHOENIX;
- zero means no meaningful difference.

Per-part folders:

| Folder | Meaning |
| --- | --- |
| `part1_operationalization/` | Symptom-label comparison |
| `part2_initial_model/` | Modifiable treatment-option comparison |
| `part3_treatment_targets/` | Treatment-target ranking comparison |
| `part4_updated_model/` | Concrete EMA item-selection comparison |
| `part5_intervention/` | Mobile coaching-message comparison |
| `synthesis/` | Cross-part signed preference synthesis |

Each per-part folder contains:

- `report/*_report.txt`: textual statistical report;
- `report/*_summary.csv`: tidy dimension-level estimates;
- `visuals/*_signed_preference_raincloud.png`: score distributions;
- `visuals/*_effect_forest.png`: intercept estimates with confidence intervals;
- `visuals/*_tost_equivalence.png`: equivalence-test panel.

Legacy `study_*` folders are older artefacts from the previous evaluation
design and are not used by the current signed LLM-as-judge pipeline.
