# Phase 1 PRE Survey — HCP Generation Instrument

This folder contains the Qualtrics configuration blueprint for **Phase 1 (PRE)**
of the PHOENIX evaluation study. Five healthcare professionals independently
complete five clinical reasoning tasks for two assigned case vignettes each,
producing the expert comparison corpus for Phase 2.

---

## Purpose

Generate 10 unique human expert outputs (one per case, per clinical reasoning
part) that can be compared to PHOENIX outputs in a double-blind evaluation
(Phase 2 POST survey). The distributed design (2 cases per HCP rather than 10)
keeps individual participant burden under 40 minutes while covering all 10 cases.

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

### 2. Screening question (Q_ParticipantCode)

Item type: **Multiple Choice (single answer)**

> "Please select your assigned participant code from the list below.
> If you have not received a code, contact the researcher before proceeding."
>
> ○ HCP-PRE-01  ○ HCP-PRE-02  ○ HCP-PRE-03  ○ HCP-PRE-04  ○ HCP-PRE-05

Apply **Display Logic** to each case block: `Show if Q_ParticipantCode = [relevant code]`.

### 3. Question types per part

| Part | Qualtrics item type | Notes |
|------|---------------------|-------|
| 1 | **Text Entry — Large** (Essay) | One item per case. Prompt: "List 2–6 criteria using format: Label \| Description" |
| 2 | **Text Entry — Large** (Essay) | One item per case. Prompt: "List 3–6 predictors using format: Label \| Measurement \| Criteria" |
| 3 | **Rank Order** | Options = predictor labels from the bipartite network (5 options); respondent selects/ranks 2–4 |
| 4 | **Multiple Choice — Multiple Answers** (checkbox) + optional **Text Entry — Small** | Options = predictor labels; instruction: select 2–4 |
| 5a | **Multiple Choice (single answer)** | HAPA phase: Pre-intentional / Intentional / Action-maintenance |
| 5b | **Text Entry — Medium** | "Write a 2–3 sentence personalized digital coaching message for this person." |

### 4. Embedded data

At the start of the survey, set the following **Embedded Data** fields:
- `study_phase` = PRE
- `survey_version` = 1.0

These fields will appear in the CSV export and simplify data cleaning.

### 5. Data export and file naming

After data collection, export the Qualtrics responses as CSV with headers.
Rename the file following the convention:
```
study_01_operationalization.csv   ← Part 1 responses
study_02_initial_model.csv        ← Part 2 responses
study_03_treatment_target.csv     ← Part 3 responses
study_04_updated_model.csv        ← Part 4 responses
study_05_intervention.csv         ← Part 5 responses
```

These filenames match the pseudodata factory and study runners in
`evaluation/survey_analysis/utils/`.

---

## Case Assignment Reference

| Participant code | Case A | Case B |
|-----------------|--------|--------|
| HCP-PRE-01 | C01 — 28yr software developer (sleep/stress) | C02 — 34yr admin (panic/avoidance) |
| HCP-PRE-02 | C03 — 41yr teacher (burnout) | C04 — 63yr retiree (grief) |
| HCP-PRE-03 | C05 — 25yr student (OCD-type) | C06 — 32yr marketing (performance anxiety) |
| HCP-PRE-04 | C07 — 46yr manager (depression) | C08 — 38yr project manager (ADHD) |
| HCP-PRE-05 | C09 — 27yr sales rep (BPD-type) | C10 — 52yr director (stress/alcohol) |

---

## Time Estimates (per participant)

| Part | Time (2 cases) |
|------|----------------|
| 1 — Operationalization | ~6 min |
| 2 — Initial model | ~6 min |
| 3 — Treatment targets | ~7 min |
| 4 — Updated model | ~5 min |
| 5 — Intervention message | ~8 min |
| **Total** | **~30–40 min** |

---

## Survey Blueprint Reference

See `main.pdf` for the complete Qualtrics configuration reference, including:
- Full case vignette text for all 10 cases
- Exact item prompts for all parts
- Bipartite network diagrams for Part 3 (all 10 cases)
- Standardised context blocks for Parts 2–5
- Display Logic annotations (amber boxes) indicating which participant code
  triggers each case block
