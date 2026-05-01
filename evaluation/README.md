# Evaluation Workspace

Canonical execution, validation, and human evaluation workspace for the PHOENIX engine.
This directory contains two distinct but linked concerns: **engine execution** (running and
validating the PHOENIX pipeline) and **human evaluation** (the formal two-phase expert study
comparing PHOENIX outputs to independent clinician outputs).

---

## Directory Overview

```
evaluation/
├── integrated_pipeline/       # End-to-end automated runner (primary execution path)
├── sequential/                # Stage-by-stage manual runner (development / debugging)
├── example/                   # Minimal worked example with assets and a rendered report
├── quality_and_research/
│   ├── quality_assurance/     # pytest suites + contract validation
│   └── research_communication/# Research report generation utilities
├── artifacts/                 # Legacy outputs retained for traceability
├── qualtrics/                 # Survey instruments for the human evaluation study
│   ├── 01_HCPs_PRE/           # Phase 1: expert generation surveys (HCP-PRE-01 to -05)
│   │   ├── separate_HCPs/     # Individual HCP bundles (PDF + Word, one per participant)
│   │   │   ├── HCP_1/ … HCP_5/
│   │   │   ├── generate_hcp_surveys.py
│   │   │   └── bipartite_edge_weights.json
│   │   ├── total/             # Consolidated PRE blueprint
│   │   └── main.{tex,pdf}
│   └── 02_HCPs_POST/          # Phase 2: blind evaluation survey
│       └── main.{tex,pdf}
└── survey_analysis/           # Statistical analysis stack for the evaluation study
    ├── data/                  # Pseudodata entrypoint + generated CSVs
    ├── utils/                 # Study runners (00–06) + shared statistics/plotting
    └── results/               # Generated figures and reports
```

---

## Part 1 — Engine Execution

### Integrated Pipeline (`integrated_pipeline/`)

The primary way to run the full PHOENIX engine end-to-end. The runner wires all stages
together and writes structured output to a timestamped run directory.

**Entry points:**
- `run_pipeline.py` — standard run from a free-text pseudoprofile source
- `run_engine_pipeline.py` — run from existing iterative-start pseudodata (cycle 2+)

**Run output layout** (`integrated_pipeline/runs/<run_id>/`):

```
00_operationalization/
01_initial_observation_model/
02_pseudodata_generation/
03_readiness_check/
04_time_series_analysis/
   └── network/
05_momentary_impact_coefficients/
06_target_identification_and_model_update/
07_hapa_digital_intervention/
08_treatment_translation_communication/
09_impact_visualizations/          ← support output, not core engine
10_research_reports/               ← support output, not core engine
logs/
pipeline_summary.json
cycles/
   └── cycle_<NN>/                 ← same layout; cycles 2+ skip 00/01
```

### Sequential Runner (`sequential/`)

Manually runnable stage modules for development and debugging. Each stage has its own
`run_step.py` so individual steps can be executed and inspected in isolation.

Stages mirror the integrated pipeline:

| Directory | Stage |
|-----------|-------|
| `00_pseudoprofile_generation/` | Free-text ingestion and pseudoprofile construction |
| `01_operationalization/` | Criterion operationalization |
| `02_initial_observation_model/` | Initial biopsychosocial network model |
| `03_readiness_check/` | EMA data quality and readiness gating |
| `04_network_time_series_analysis/` | Time-series network estimation |
| `05_momentary_impact_quantification/` | Edge weight / momentary impact coefficients |
| `06_target_identification_and_model_update/` | Treatment-target selection + model refresh |
| `07_hapa_digital_intervention/` | HAPA-based digital intervention generation |
| `08_treatment_translation_communication/` | Clinical communication output |

### Example (`example/`)

A minimal worked example with pre-built assets and a rendered report. Useful as a
reference for expected output format and for demonstrating the full pipeline to
reviewers or collaborators.

---

## Part 2 — Quality and Research Support

### Quality Assurance (`quality_and_research/quality_assurance/`)

pytest suites and contract validation checks that verify:
- stage input/output schemas
- inter-stage data contract compliance
- regression tests on canonical run outputs

### Research Communication (`quality_and_research/research_communication/`)

Utilities for generating research reports from pipeline run outputs. Feeds into
the `10_research_reports/` stage of the integrated pipeline.

### Artifacts (`artifacts/`)

Legacy pipeline outputs migrated from earlier directory structures. Retained for
traceability and cross-run comparisons. Not actively updated.

---

## Part 3 — Human Evaluation Study

The human evaluation measures whether PHOENIX outputs are clinically comparable to
independent expert outputs across all five pipeline stages. It uses a two-phase
design: **Phase 1 (PRE)** collects human expert outputs; **Phase 2 (POST)** presents
PHOENIX and HCP outputs blind for rating.

### Study Design at a Glance

```
Phase 1 — PRE (Expert Generation)        Phase 2 — POST (Blind Evaluation)
──────────────────────────────────        ──────────────────────────────────
5 HCPs × 2 cases each (non-overlapping)  5 different HCPs
= 10 independent expert outputs           × 10 cases × 2 outputs (A / B, blind)
  (one human response per case, per part) × 5 rating dimensions per part
                                          = 100 ratings per dimension per part
```

**Case assignment (Phase 1):**

| Participant | Cases | Clinical themes |
|-------------|-------|-----------------|
| HCP-PRE-01 | C01, C02 | Sleep/cognitive stress; Panic/avoidance |
| HCP-PRE-02 | C03, C04 | Occupational burnout; Prolonged grief |
| HCP-PRE-03 | C05, C06 | OCD-type intrusions; Performance anxiety |
| HCP-PRE-04 | C07, C08 | Persistent depression; ADHD-type dysexecutive |
| HCP-PRE-05 | C09, C10 | BPD-type emotion dysregulation; Chronic work stress |

**Counterbalancing (Phase 2):** PHOENIX = Output A for C01–C05, Output B for C06–C10.
This controls for any systematic preference for label A or B. Sources are revealed only
after data collection.

**Study registry:**

| Study | Pipeline stage | Primary outcome |
|-------|---------------|-----------------|
| `00` | Momentary impact quantification | Spearman footrule distance vs. gold ranking |
| `01` | Operationalization (Part 1) | Criterion accuracy; operationalization quality; completeness |
| `02` | Initial observational model (Part 2) | Clinical appropriateness; network validity; EMA feasibility; intervention potential |
| `03` | Treatment-target identification (Part 3) | Clinical priority; evidence alignment; rank coherence |
| `04` | Updated observational model (Part 4) | Target alignment; measurement selection quality |
| `05` | Intervention message (Part 5) | HAPA phase appropriateness; tailoring; actionability; professional tone; Cohen's κ |
| `06` | Holistic synthesis | Normalized score across all dimensions; TOST equivalence |

Primary inferential model: **crossed linear mixed-effects regression** (participant
intercept + case intercept) estimating the PHOENIX-minus-HCP difference per dimension.
Multiplicity control: **Holm correction** within each study. Equivalence testing: **TOST**
with δ = ±0.5 Likert units (Studies 01–05) or δ = ±0.05 normalized units (Study 06).

---

### Qualtrics Survey Instruments (`qualtrics/`)

Contains the LaTeX-generated survey blueprints used to configure the Qualtrics instruments.

#### Phase 1 — PRE Survey (`qualtrics/survey/01_HCPs_PRE/`)

Each participant receives a **self-contained personal bundle** (PDF + Word) covering only
their two assigned cases, labelled simply *Casus 1* and *Casus 2*. This makes the personal
scope unambiguous — participants see no reference to other cases or the global C01–C10 numbering.

| File / folder | Purpose |
|---------------|---------|
| `separate_HCPs/HCP_1/` … `HCP_5/` | Individual participant bundles (PDF + Word, compiled from `main.tex`) |
| `separate_HCPs/generate_hcp_surveys.py` | Generator script — rebuilds all 5 bundles from the `CASES` dict |
| `separate_HCPs/bipartite_edge_weights.json` | All bipartite network edge weights (−1 to 1) for all 10 cases, saved for later analysis |
| `total/` | Consolidated PRE blueprint (all cases in one document) |
| `main.{tex,pdf}` | Parent-level PRE overview |

**Bipartite network in the PRE bundles (Part 3):**  
Each bundle shows a bipartite network with **predictors on the left** and **criteria on the
right**. Edge width is proportional to the absolute weight (|w|, scale 0–1); red edges are
risk factors (positive weight, predictor increases criterion severity), blue edges are
protective factors (negative weight, predictor decreases criterion severity). All edge weights
are stored in `bipartite_edge_weights.json`.

#### Phase 2 — POST Survey (`qualtrics/survey/02_HCPs_POST/`) [legacy]

A single Qualtrics survey presented to all 5 POST evaluators. Each evaluator rates both
outputs (A and B) for all 10 cases across all 5 parts, fully blind to source.

| File | Purpose |
|------|---------|
| `main.tex` | POST survey blueprint (all cases × outputs × rating dimensions) |
| `main.pdf` | Compiled PDF — use as Qualtrics configuration reference |

---

### Survey Analysis (`survey_analysis/`)

Modular analysis stack. Runs on pseudodata during development; switches to real Qualtrics
exports after data collection.

| Sub-folder | Contents |
|------------|----------|
| `data/` | Pseudodata generator input and generated CSVs |
| `utils/` | Numbered study runners (`00`–`06`) + `shared/` statistics, plotting, and pseudodata utilities |
| `results/` | Generated figures (raincloud, forest, TOST, heatmap) and reports per study |

**To run all studies:**
```bash
bash evaluation/survey_analysis/run_all_studies.sh
```

---

## PRE → POST Researcher Workflow

1. Collect Phase 1 responses via the five personal HCP bundles.
2. Extract each HCP's outputs from Qualtrics (one response per case per part).
3. Insert human HCP outputs into the POST survey at the positions specified in
   `02_HCPs_POST/main.tex` (counterbalancing table).
4. Run the PHOENIX pipeline on all 10 case vignettes to generate PHOENIX outputs.
5. Launch Phase 2 data collection. POST evaluators see only A/B labels.
6. After collection, unblind sources and run `survey_analysis/run_all_studies.sh`.
