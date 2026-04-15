# Phase 1 PRE Survey — HCP Generation Instrument (Total / Researcher Reference)

This folder contains the **complete Qualtrics configuration blueprint** for Phase 1 (PRE) of the PHOENIX evaluation study, covering **all ten case vignettes** across all five clinical reasoning parts. It is the canonical researcher-reference document and serves as the single source of truth for Qualtrics configuration, study design documentation, and archival purposes.

---

## Document Contents (`main.pdf` / `main.tex`)

The document is written in English and is intended for:
- Qualtrics survey setup (copy exact item prompts, display logic, and response format specs)
- Research documentation and archival (thesis appendix, ethics submission, study pre-registration)
- Reviewer reference during double-blind evaluation (Phase 2)

### Structure

| Section | Content |
|---------|---------|
| 1 — Introduction | Study background, PHOENIX system description, participant instructions, ethical notice |
| 2 — Case Assignment | Distributed HCP-to-case assignment table; Qualtrics Display Logic notes |
| 3 — Case Vignettes Reference | All 10 case vignettes with profiles and summary table |
| 4 — Part 1 | Operationalization items for C01–C10 (with Display Logic annotations) |
| 5 — Part 2 | Initial observational model items for C01–C10 |
| 6 — Part 3 | Treatment-target prioritisation + bipartite network diagrams for C01–C10 |
| 7 — Part 4 | Updated observational model (breadth-first EMA selection) for C01–C10 |
| 8 — Part 5 | HAPA-phase + coaching message items for C01–C10 |

---

## Case Coverage

All **10 case vignettes** (C01–C10) are present for all **5 parts**. Each case appears exactly once per part, gated by Display Logic to the assigned HCP code.

| Case | Profile | Assigned to |
|------|---------|-------------|
| C01 | 28-yr software developer — sleep onset insomnia, cognitive hyperarousal | HCP-PRE-01 |
| C02 | 34-yr healthcare administrator — recurrent panic, anticipatory anxiety, avoidance | HCP-PRE-01 |
| C03 | 41-yr secondary school teacher — professional burnout, exhaustion, social withdrawal | HCP-PRE-02 |
| C04 | 63-yr retired professional — complicated grief, insomnia, guilt rumination | HCP-PRE-02 |
| C05 | 25-yr postgraduate student — intrusive harm obsessions, compulsive reviewing, avoidance | HCP-PRE-03 |
| C06 | 32-yr marketing professional — performance anxiety, post-event rumination, advancement avoidance | HCP-PRE-03 |
| C07 | 46-yr accounts manager — persistent low mood, anergia, social isolation | HCP-PRE-04 |
| C08 | 38-yr project manager — ADHD-related executive dysfunction, shame, task avoidance | HCP-PRE-04 |
| C09 | 27-yr sales representative — emotional dysregulation, impulsivity, interpersonal instability | HCP-PRE-05 |
| C10 | 52-yr operations director — chronic work stress, sleep maintenance insomnia, alcohol coping | HCP-PRE-05 |

---

## Qualtrics Setup Instructions

### 1. Survey structure

Create a single Qualtrics survey with the following block structure:

```
Block: Consent and Participant Information
Block: Screening — Participant Code Assignment
Block A: Cases C01 + C02 (all 5 parts)    ← Display Logic: Code = HCP-PRE-01
Block B: Cases C03 + C04 (all 5 parts)    ← Display Logic: Code = HCP-PRE-02
Block C: Cases C05 + C06 (all 5 parts)    ← Display Logic: Code = HCP-PRE-03
Block D: Cases C07 + C08 (all 5 parts)    ← Display Logic: Code = HCP-PRE-04
Block E: Cases C09 + C10 (all 5 parts)    ← Display Logic: Code = HCP-PRE-05
Block: Completion and Debrief
```

Within each block (A–E), the 5 parts appear sequentially:
- Part 1 questions for the 2 cases
- Part 2 questions for the 2 cases
- Part 3 questions for the 2 cases
- Part 4 questions for the 2 cases
- Part 5 questions for the 2 cases

### 2. Screening question (`Q_ParticipantCode`)

Item type: **Multiple Choice (single answer)**

> "Please select your assigned participant code from the list below.
> If you have not received a code, contact the researcher before proceeding."
>
> ○ HCP-PRE-01  ○ HCP-PRE-02  ○ HCP-PRE-03  ○ HCP-PRE-04  ○ HCP-PRE-05

Apply **Display Logic** to each case block: `Show if Q_ParticipantCode = [relevant code]`.
The amber `Display Logic` boxes in `main.pdf` indicate the exact condition per case block.

### 3. Question types per part

| Part | Qualtrics item type | Notes |
|------|---------------------|-------|
| 1 | **Text Entry — Large** (Essay) | One item per case. Prompt: "List 2–6 criteria: Label \| Description" |
| 2 | **Text Entry — Large** (Essay) | One item per case. Prompt: "List 3–5 predictors: Label \| Measurement \| Criteria" |
| 3 | **Rank Order** | Rank all 5 predictor nodes from bipartite network; select 2–4 targets |
| 4 | **Multiple Choice — Multiple Answers** (checkbox) | Select exactly 5 EMA items from a list of 20 |
| 5a | **Multiple Choice (single answer)** | HAPA phase: Pre-intentional / Intentional / Action-maintenance |
| 5b | **Text Entry — Medium** | 2–3 sentence personalized digital coaching message |

### 4. Embedded data

At the start of the survey, set the following **Embedded Data** fields:
- `study_phase` = PRE
- `survey_version` = 1.0

### 5. Data export and file naming

After data collection, export Qualtrics responses as CSV with headers and rename:

```
study_01_operationalization.csv   ← Part 1 responses
study_02_initial_model.csv        ← Part 2 responses
study_03_treatment_target.csv     ← Part 3 responses
study_04_updated_model.csv        ← Part 4 responses
study_05_intervention.csv         ← Part 5 responses
```

These filenames match the pseudodata factory and study runners in `evaluation/survey_analysis/utils/`.

---

## Bipartite Network Diagrams (Part 3)

Each Part 3 case block contains a bipartite network diagram showing the monitoring-derived predictor–criterion structure. In the total reference document, the networks use a simplified three-level edge scheme (strong / moderate / weak) without directional colour coding, as this document serves as a researcher/configurator reference rather than a participant-facing instrument.

The **participant-facing** separate HCP bundles (see `../separate_HCPs/`) use the full colour scheme:
- **Red** edges = risk factor (positive weight: predictor increases the criterion)
- **Blue** edges = protective factor (negative weight: predictor decreases the criterion)
- **Line thickness** proportional to `|w|`, min-max normalised within each network (range 1–5 pt)
- A legend is embedded in each figure

---

## Relation to Separate HCP Bundles

The participant-facing instruments in `../separate_HCPs/` show only the two cases assigned to each HCP. This total document contains all ten and is used solely for:
1. Qualtrics survey configuration (copy item prompts and Display Logic from here)
2. Research documentation (thesis appendix, archival record)
3. Reviewer reference in Phase 2

Participants never see this document.

---

## Time Estimates (per participant, 2 cases)

| Part | Time |
|------|------|
| 1 — Operationalization | ~6 min |
| 2 — Initial observational model | ~6 min |
| 3 — Treatment targets | ~7 min |
| 4 — Updated observational model | ~5 min |
| 5 — Intervention message | ~8 min |
| **Total** | **~30–40 min** |
