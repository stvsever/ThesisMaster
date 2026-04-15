# PHOENIX Survey Analysis

This folder contains the modular survey-analysis stack for the PHOENIX evaluation
framework: pseudodata generation, shared statistics utilities, study runners, and
publication-oriented outputs.

---

## Study Design

The PHOENIX evaluation uses a **two-phase expert elicitation and blind evaluation
design** that maps directly to the five-stage pipeline structure.

### Phase 1 — PRE (Expert Generation)

Five healthcare professionals (HCP-PRE-01 through HCP-PRE-05) independently
complete five clinical reasoning tasks for two assigned case vignettes each
(10 unique cases total, non-overlapping assignment). All five parts of the survey
correspond directly to the five PHOENIX pipeline steps. PRE respondents produce
the human expert comparison corpus.

**Assignment:** HCP-PRE-01 → C01,C02; HCP-PRE-02 → C03,C04;
HCP-PRE-03 → C05,C06; HCP-PRE-04 → C07,C08; HCP-PRE-05 → C09,C10.

**Per-case outputs per part:** 1 unique expert response (from the assigned HCP).

### Phase 2 — POST (Blind Evaluation)

Five different healthcare professionals (HCP-POST-01 through HCP-POST-05)
evaluate **all 10 cases × 2 outputs (PHOENIX + HCP, labelled A/B)** across all
five parts. Outputs are presented blind; source labels are revealed only after
data collection.

**Counterbalancing:** PHOENIX = Output A for C01–C05; PHOENIX = Output B for
C06–C10. This controls for any systematic A/B label preference.

**Statistical power:** 5 raters × 10 cases × 2 outputs = **100 ratings per
dimension per part**. The primary inferential model is a crossed mixed-effects
regression (participant intercept + task intercept) estimating the
PHOENIX-minus-HCP difference on each dimension.

---

## Study Registry

| Study ID | Pipeline stage | Evaluator group | Design | Primary outcome |
|----------|---------------|----------------|--------|-----------------|
| `study_00` | Momentary impact quantification | 30 lay users | PHOENIX vs. static estimator | Spearman footrule distance |
| `study_01` | Operationalization (Part 1) | 5 POST HCPs | Blind A/B | Likert: criterion accuracy; operationalization quality; completeness |
| `study_02` | Initial observational model (Part 2) | 5 POST HCPs | Blind A/B | Likert: clinical appropriateness; network validity; EMA feasibility; intervention potential |
| `study_03` | Treatment-target identification (Part 3) | 5 POST HCPs | Blind A/B | Likert: clinical priority; evidence alignment; rank coherence |
| `study_04` | Updated observational model (Part 4) | 5 POST HCPs | Blind A/B | Likert: target alignment; measurement selection quality |
| `study_05` | Intervention message (Part 5) | 5 POST HCPs | Blind A/B + HAPA kappa | Likert: HAPA phase appropriateness; message tailoring; actionability; professional tone; + Cohen's κ on phase classification |
| `study_06` | Holistic synthesis | All POST HCPs | Cross-study LME | Normalized score; TOST equivalence |

Studies `00` and `03` through `05` all use HCP raters. Study `06` pools
normalized scores from Studies `01`, `02`, `03`, `04`, and `05`.

---

## Folder Structure

- [`data/`](data/) — pseudodata entrypoint and generated CSV files
- [`utils/`](utils/) — numbered study entrypoints (`00`–`06`) plus a `shared/`
  package for reusable statistics, plotting, path, and pseudodata logic
- [`results/`](results/) — generated reports and figures
- [`run_all_studies.sh`](run_all_studies.sh) — end-to-end runner for all studies

---

## Statistical Principles

### Why mixed-effects models are the default

Across all survey studies, observations are not independent:
- the same POST evaluator rates multiple items
- the same item is rated by multiple evaluators
- for each case, the same evaluator rates both PHOENIX and HCP outputs

The primary analysis uses **crossed linear mixed-effects models** (participant
intercept + task/case intercept) rather than simple rank tests. Nonparametric
tests (Mann-Whitney U) serve only as fallbacks when a mixed model fails to
converge.

### Why Likert ratings are analyzed with linear mixed models

Outcomes in Studies 01–05 are ordinal 1–9 ratings. Linear mixed models are used
because they:
- support the crossed random-effects structure
- keep coefficients directly comparable across studies and dimensions
- perform well for bounded Likert outcomes when the comparison target is a
  mean performance difference rather than a latent threshold

### TOST equivalence testing

All per-dimension and holistic analyses include a Two One-Sided Tests (TOST)
equivalence test with:
- **Studies 01–05:** δ = ±0.5 Likert units (smallest effect of practical
  significance on a 1–9 scale)
- **Study 06 (holistic):** δ = ±0.05 normalized score units

This tests whether PHOENIX and HCP outputs are statistically indistinguishable
within a practically meaningful margin.

### HAPA Phase Classification (Study 05 only)

In addition to Likert ratings, Study 05 includes an inter-rater agreement
analysis for HAPA phase classification. Five POST evaluators and the PHOENIX
system each classify each case as pre-intentional / intentional / action-
maintenance. Agreement is quantified with:
- **Cohen's κ** (pairwise: PHOENIX vs. each POST rater)
- **Fleiss' κ** (multi-rater: all POST raters jointly)

This outcome is stored in `study_05_hapa_kappa.csv` and analyzed separately
from the Likert ratings.

---

## Study Families

### Ranking study: `00`

Study 00 uses **Spearman footrule distance** as the dependent variable (lower =
better agreement with the latent gold ranking). Primary model: fixed effect for
`estimator` + participant-level and task-level random effects.

### Expert-comparison Likert studies: `01`–`05`

Each study is run **per dimension** (not pooled across dimensions). Primary
model per dimension:
- fixed effect for `source` (PHOENIX vs. HCP)
- fixed effect for `shift_regime` when present
- participant-level random intercept
- participant-level random slope for source when estimable
- item-level random intercept (one per study)

Item-level random factors:
- Study 01: `text_ID`
- Study 02: `item_ID`
- Studies 03–04: `task_ID`
- Study 05: `intervention_ID`

Multiplicity control: **Holm correction** across dimensions within each study.

### Holistic synthesis: `06`

Study 06 pools **normalized (0–1) scores** from Studies 01–05. Primary fixed
effects: `reasoner_group` (PHOENIX vs. HCP), `study_id`, `dimension`,
`shift_regime`. Random structure: participant intercept + reasoner slope (when
estimable) + crossed task and response-block intercepts.

The holistic model is the primary inferential target; study-specific follow-up
effects are Holm-adjusted across studies.

---

## Pseudodata Design

The pseudodata generator (`utils/shared/pseudodata_factory.py`) is structured
to stress-test the analysis pipeline with realistic distributional properties:

- **Participant heterogeneity:** rater-level random effects (μ=0, σ=0.35)
- **Task heterogeneity:** item-level random effects (μ=0, σ=0.30)
- **Dimension-specific advantages:** each dimension encodes slightly different
  PHOENIX vs. HCP mean differences, reflecting expected real-world patterns
- **Shift regimes:** standard / ambiguous / implementation_shift / context_shift
  (one per 4-item cycle) test robustness to distribution shift
- **HAPA kappa pseudodata:** simulates classification agreement across 5 POST
  raters with realistic agreement rates (~72–80%) and plausible confusion
  patterns (adjacent phase confusions more likely than distal ones)

Dimension names in the factory **exactly match** the POST survey instrument
(`evaluation/qualtrics/02_HCPs_POST/main.pdf`) to ensure the analysis pipeline
can be validated against real data without renaming.

---

## Visual Outputs

Current output types per study:

| Figure type | Studies |
|-------------|---------|
| Raincloud plots (KDE + box + jitter) | 01–06 |
| Forest plots (effect sizes + CI + TOST badges) | 01–06 |
| TOST equivalence panel | 01–06 |
| Dimension × study delta heatmap | 06 only |
| Shift-regime robustness plots | 00–05 |

All figures are saved to `results/{study_slug}/visuals/`.

---

## Running The Full Pipeline

```bash
bash evaluation/survey_analysis/run_all_studies.sh
```

This will:
1. Regenerate all pseudodata (including HAPA kappa data for Study 05)
2. Run studies `00` through `06`
3. Write updated reports and figures to `evaluation/survey_analysis/results/`

---

## Tracked Artifacts

Generated survey CSV files and result directories are excluded from version
control (see `.gitignore`). This keeps commits focused on code and documentation
rather than reproducible heavy artifacts.
