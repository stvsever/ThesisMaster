# PHOENIX Survey Analysis

End-to-end evaluation of the PHOENIX multi-agent personalised digital
mental-health pipeline by comparing its outputs against healthcare
professionals (HCPs) using a double-blind LLM-as-judge.

This folder is the home of:

- The Qualtrics CSV parser that turns raw HCP survey submissions into
  canonical per-(case, part) JSON.
- The LLM-as-judge module (Google Gemini 2.5 Flash via OpenRouter,
  `google/gemini-2.5-flash`).
- The per-part Linear Mixed Model + TOST analysis machinery.
- The cross-part holistic synthesis with a parts-by-dimensions effect
  heatmap.

The full pipeline runs end-to-end in pseudo mode without any real LLM
calls so the analysis can be developed and validated offline.

## Why the design changed

The original POST-survey design (5 expert HCPs each rating 100 outputs
blind via Qualtrics) had three problems that this redesign solves:

1. **Recruitment burden** for a second wave of clinicians. Even one wave
   was enormously expensive in person-hours; a second wave for every
   PHOENIX iteration was infeasible.
2. **Latency**: any change to the PHOENIX outputs forced weeks of waiting
   for the next survey wave.
3. **Drift**: 5 HCPs with different ratings produce a noisier signal than
   a stable, deterministic-ish judge with anchored Likert prompts.

The new design replaces the second wave with a single LLM judge. The
PRE-survey HCPs stay (10 unique HCPs, one per case), but their outputs
are now compared against PHOENIX outputs via the judge instead of being
re-rated by other HCPs.

## Pipeline

```
Qualtrics CSV
   |  (parsing/qualtrics_parser.py)
   v
data/02_parsed/hcp_outputs.json  -- canonical {case_id: {part: ...}}

PHOENIX runs (offline)
   |
   v
data/03_system/system_outputs.json  -- canonical {case_id: {part: ...}}

                |    |
                v    v
        llm_as_judge.judge_runner
        - deterministic A/B blinding per (case, part, run)
        - 5 stochastic runs per cell (default)
        - strict-JSON output with parse-retry
   |
   v
data/04_judgments/judgments_long.csv
columns: case_id, part, dimension, judge_run, source, rating,
         justification, prompt_version, model, timestamp

   |
   v
analysis/part{1..5}_*.py
- LMM: rating ~ source + (1|case_id) + (1|judge_run)
- TOST equivalence (delta = 0.5 Likert, half a step on 1..7)
- Holm correction across dimensions within each part
- Forest plot + raincloud plot + TOST panel

   |
   v
analysis/synthesis.py
- Normalise rating to [0, 1]: (rating - 1) / 6
- Pooled LMM: score_norm ~ source*part + (1|case_id) + (1|judge_run) + (1|dimension)
- Cross-part forest + per-part raincloud + parts x dimensions delta heatmap
- Global TOST with delta=0.05 on the 0..1 scale
```

## Folder structure

```
evaluation/survey_analysis/
├── README.md                       # this file
├── requirements.txt                # python deps
├── run_pipeline.sh                 # bash entry point
├── pipeline.py                     # python orchestrator
│
├── data/
│   ├── 01_raw/                     # Qualtrics CSV (gitignored)
│   ├── 02_parsed/                  # parsed HCP outputs (gitignored)
│   ├── 03_system/                  # PHOENIX outputs (gitignored)
│   ├── 04_judgments/               # long-format judgments + raw responses (gitignored)
│   └── pseudodata/                 # pseudo HCP/PHOENIX/judgments (gitignored)
│
├── parsing/
│   ├── canonical_schemas.py        # canonical per-part dataclasses + coercers
│   ├── qualtrics_parser.py         # Qualtrics CSV -> per-(case,hcp) JSON
│   └── system_output_loader.py     # loader for PHOENIX outputs
│
├── llm_as_judge/
│   ├── README.md                   # judge design + prompt-engineering rationale
│   ├── dimensions.py               # PROMPT_VERSION + per-part dimensions
│   ├── prompts/                    # system + per-part Markdown prompt templates
│   ├── openrouter_client.py        # OpenAI-SDK / urllib client for OpenRouter
│   ├── output_schema.py            # JSON schema + tolerant parser
│   ├── judge_runner.py             # blinding + retry + persistence
│   └── pseudo_judge.py             # local stand-in (no API call)
│
├── analysis/
│   ├── shared/                     # reusable LMM/TOST/plot helpers
│   │   ├── shared_stats.py         # raincloud, forest, TOST, LMM (kept from old code)
│   │   ├── comparison_study.py     # generic per-part runner (1..7 Likert)
│   │   ├── holistic_comparison.py  # cross-part synthesis runner
│   │   ├── plotting_extras.py      # parts x dimensions heatmap
│   │   └── survey_paths.py         # path helpers
│   ├── part1_operationalization.py
│   ├── part2_initial_model.py
│   ├── part3_treatment_targets.py
│   ├── part4_updated_model.py
│   ├── part5_intervention.py
│   └── synthesis.py
│
├── pseudodata/
│   ├── generate_hcp_outputs.py
│   ├── generate_phoenix_outputs.py
│   └── generate_judgments.py
│
└── tests/
    └── test_smoke.py               # end-to-end pseudo-mode smoke test
```

## Statistical specification

For each part p in {1, 2, 3, 4, 5} and each dimension d in
`DIMENSIONS_BY_PART[p]`:

```
rating ~ source + (1 | case_id) + (1 | judge_run)
```

- `source` is categorical with reference `hcp` and test `phoenix`.
- `case_id` ranges over C01..C10 (10 cases x 1 HCP each in the new design).
- `judge_run` ranges over the 5 stochastic samples per (case, part).

Implementation: `analysis.shared.shared_stats.fit_crossed_mixedlm` tries a
sequence of three random-effects specifications (case-intercept +
judge-run VC; judge-run intercept + case VC; case intercept fallback)
and returns the first that converges. If all fail, a Mann-Whitney U
fallback is reported instead, and the LMM artefacts are flagged as
non-converged in the per-part summary CSV.

Effect sizes: Cohen's d on the (PHOENIX vs HCP) split with bootstrap CIs.

Equivalence testing: TOST with delta = 0.5 Likert points on the 1..7
scale, mirroring the half-step minimal clinically important difference
convention. The cross-part synthesis additionally runs a global TOST on
normalised scores with delta = 0.05.

Multiplicity correction: Holm-Bonferroni across the 5..7 dimensions within
each part. The cross-part synthesis applies a separate Holm correction
across the 5 per-part main effects.

## Per-part dimensions (summary)

| Part | Goal | Dimensions |
| --- | --- | --- |
| 1 | Operationalisation | clinical_accuracy, construct_interoperability, resolution_preservation, behavioural_specificity, internal_consistency, completeness, conciseness_redundancy |
| 2 | Initial model | clinical_appropriateness, network_validity, ema_feasibility, predictor_diversity, measurement_specificity, intervention_potential, construct_coverage |
| 3 | Treatment targets | top_target_appropriateness, evidence_alignment, rank_coherence, network_impact_awareness, monitoring_integration, modifiability_weighting |
| 4 | Updated model | adaptive_reasoning, target_alignment, personalisation, measurement_quality, parsimony, theoretical_coherence |
| 5 | Intervention | hapa_phase_appropriateness, behavioural_change_potential, personalisation_specificity, professional_tone, empathy_warmth, clarity_actionability, message_appropriateness_length |

Full goal descriptions, anchors, and rationales live in
`llm_as_judge/dimensions.py`.

## Reproducibility

- Random seeds: every stochastic step (blinding, pseudo-judge effects,
  bootstrap CIs) is seeded from a deterministic hash of its inputs. The
  pseudo-judge is fully deterministic given (case_id, part, judge_run).
- Prompt versioning: `llm_as_judge.dimensions.PROMPT_VERSION` is recorded
  in every long-format row; bump it whenever the prompt or the dimension
  set changes so analyses can stratify or filter by version.
- Model pinning: `google/gemini-2.5-flash` via OpenRouter; bump the
  default in `openrouter_client.py` to switch.
- Retry policy: 3 transient retries with exponential backoff in the
  OpenRouter client, plus 1 JSON-parse re-prompt with a "fix your JSON"
  follow-up turn.

## How to run

Pseudo mode (no API key needed):

```
bash evaluation/survey_analysis/run_pipeline.sh --mode pseudo
```

Real mode:

```
export OPENROUTER_API_KEY=sk-...
bash evaluation/survey_analysis/run_pipeline.sh --mode real --n-runs 5
```

Subset of cases / parts (works in either mode):

```
bash evaluation/survey_analysis/run_pipeline.sh \
    --mode pseudo --cases C01 C02 C03 --parts part1 part5
```

Re-run only the analysis (skipping parsing and judging, e.g. while
iterating on plots):

```
bash evaluation/survey_analysis/run_pipeline.sh --mode pseudo \
    --skip-parse --skip-judge
```

## References

- Qualtrics PRE survey blueprint:
  `evaluation/qualtrics/survey/01_HCPs_PRE/`
- Allen et al. (2019). Raincloud plots: a multi-platform tool for robust
  data visualization. Wellcome Open Research.
- Lakens (2017). Equivalence tests: a practical primer for t-tests,
  correlations, and meta-analyses. Social Psychological and Personality
  Science.
- Schwarzer (2008). Modeling health behavior change (HAPA model).

## Limitations

- **Single-judge architecture**: the headline numbers depend on Gemini
  2.5 Flash. A sensitivity analysis with at least one alternative judge
  (e.g. Claude Sonnet, GPT-4o) is left as future work; the prompt
  templates and runner are agnostic to the model identifier so this is
  a one-line change.
- **Cohort size**: 10 HCPs x 5 runs gives 50 ratings per (part,
  dimension, source), which is adequate for LMM but tight for rare-effect
  detection. Confidence intervals reflect this.
- **Pseudodata content quality**: the in-tree pseudo HCP and PHOENIX
  outputs are deliberately rough; their purpose is shape-validation, not
  clinical evaluation. Real evaluation runs use the production PHOENIX
  outputs and the parsed Qualtrics responses.
