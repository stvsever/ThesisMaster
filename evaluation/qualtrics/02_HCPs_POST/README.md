# Phase 2 POST Survey — HCP Blind Evaluation Instrument

This folder contains the Qualtrics configuration blueprint for **Phase 2 (POST)**
of the PHOENIX evaluation study. Five healthcare professionals (different from
Phase 1 respondents) evaluate all 10 cases × 2 anonymised outputs (labelled A/B)
across all 5 clinical reasoning parts, producing the primary outcome data for
the PHOENIX vs. HCP mixed-effects comparison.

---

## Purpose

Collect independent expert quality ratings for both PHOENIX outputs and human
expert outputs (from Phase 1) in a double-blind format. Each output receives
5 × n_evaluators ratings per dimension, enabling a crossed mixed-effects model
that controls for rater severity and case difficulty simultaneously.

---

## Prerequisites

Before configuring the POST survey, the following must be complete:

1. **Phase 1 (PRE) data collection is finished.** All 5 PRE respondents have
   submitted their outputs for all 5 parts.
2. **PHOENIX pipeline has been run** on all 10 case vignettes for all 5 parts.
3. **The researcher has compiled the output pairs** per case per part, following
   the counterbalancing table in `main.pdf` (Section 3).

---

## Output Insertion (Pre-Survey Setup)

For each case and each part, you need to insert the actual text of Output A and
Output B into the Qualtrics question body. Use the counterbalancing table:

| Cases | Output A source | Output B source |
|-------|----------------|----------------|
| C01–C05 | PHOENIX output | PRE HCP output |
| C06–C10 | PRE HCP output | PHOENIX output |

**How to insert outputs in Qualtrics:**
- Navigate to the question for Case Cxx, Part y, Output A.
- Replace the placeholder text with the actual output from the relevant source.
- Do not include any identifying information (e.g., do not say "AI system" or
  "healthcare professional"; just present the content).

---

## Qualtrics Setup Instructions

### 1. Survey structure

```
Block: Consent and Participant Information
Block: Study Overview and Blinding Instructions
Block: Rating Scale Guide (show Section 2 of main.pdf)
Block: Part 1 — Operationalization Evaluation (Cases C01–C10)
Block: Part 2 — Initial Model Evaluation (Cases C01–C10)
Block: Part 3 — Treatment Targets Evaluation (Cases C01–C10)
Block: Part 4 — Updated Model Evaluation (Cases C01–C10)
Block: Part 5 — Intervention Message Evaluation (Cases C01–C10)
Block: Completion and Debrief
```

**No Display Logic needed** — all POST evaluators see all 10 cases.

### 2. Rating item configuration

For each rating item, use a **Likert Matrix question** or individual
**Multiple Choice (single answer)** questions with options 1–9.

Recommended configuration per dimension:
- Item type: **Slider** (range 1–9, step 1) or **Multiple Choice (radio)**
- Show anchors: "1 = Poor/Absent" and "9 = Excellent/Optimal"
- Mark as required (force response)

Each case × part block contains:
- A **Text/Graphic** item showing the vignette context
- A **Text/Graphic** item showing Output A
- A **Text/Graphic** item showing Output B
- Rating items (matrix or individual sliders)

### 3. Part 5 HAPA classification item

**Critical:** The HAPA phase classification question must appear **before** the
Output A and Output B text blocks for each Part 5 case.

Item type: **Multiple Choice (single answer)**

> "Based only on the case information above (before reading any outputs), classify
> this person's current HAPA motivational phase:
>
> ○ Pre-intentional — has not yet formed a clear intention to change behaviour
> ○ Intentional — intends to change but has not yet started acting consistently
> ○ Action-maintenance — is actively and consistently engaged in behaviour change"

Use **Page Break** to ensure Output A and B only appear after the classification
item on a new page (prevents bias from sequential exposure).

### 4. Embedded data

Set at survey start:
- `study_phase` = POST
- `survey_version` = 1.0
- `participant_ID` = [set via survey link / Qualtrics panel]

### 5. Survey link personalisation

Create 5 unique survey links (one per POST participant) with embedded data
`participant_ID` pre-filled as HCP-POST-01 through HCP-POST-05. This removes
the need for a screening question and ensures clean ID assignment in the export.

---

## Data Export and Variable Naming

After data collection, export as CSV. The expected variable structure per row is:

| Column | Values | Description |
|--------|--------|-------------|
| `participant_ID` | HCP-POST-01..05 | Evaluator identifier |
| `case_ID` | C01..C10 | Case identifier |
| `part` | 1..5 | Survey part |
| `output_label` | A or B | Blinded label |
| `dimension` | see below | Rating dimension |
| `rating` | 1..9 | Likert rating |
| `hapa_class` | 1/2/3 | Part 5 only: 1=pre-intentional, 2=intentional, 3=action-maintenance |

**Dimension names** (must match `pseudodata_factory.py` keys):

| Part | Dimension name |
|------|---------------|
| 1 | `criterion_accuracy`, `operationalization_quality`, `completeness` |
| 2 | `clinical_appropriateness`, `network_validity`, `ema_feasibility`, `intervention_potential` |
| 3 | `clinical_priority`, `evidence_alignment`, `rank_coherence` |
| 4 | `target_alignment`, `measurement_selection` |
| 5 | `hapa_phase_appropriateness`, `message_tailoring`, `actionability`, `professional_tone` |

**File naming convention** (to match survey analysis pipeline):
```
study_01_operationalization.csv
study_02_initial_model.csv
study_03_treatment_target.csv
study_04_updated_model.csv
study_05_intervention.csv
```

---

## Sample Size and Statistical Power

With 5 POST evaluators × 10 cases × 2 outputs:

| Metric | Value |
|--------|-------|
| Ratings per dimension per part | 100 |
| Total rating items completed per evaluator | ~330 |
| Crossed random effects | participant (5 levels), case (10 levels) |
| Primary test | Mixed-effects model coefficient PHOENIX vs. HCP |
| Secondary | TOST equivalence (δ = 0.5 Likert units) |
| HAPA kappa N | 5 POST raters × 10 cases = 50 classifications |

This design provides adequate power to detect a 0.5-point mean difference (the
smallest effect of practical significance on a 1–9 scale) and to test
equivalence within a ±0.5 window.

---

## Timing Reference

| Part | Cases | Est. time |
|------|-------|-----------|
| 1 — Operationalization | 10 × 2 outputs × 3 dims | ~20 min |
| 2 — Initial model | 10 × 2 outputs × 4 dims | ~25 min |
| 3 — Treatment targets | 10 × 2 outputs × 3 dims | ~20 min |
| 4 — Updated model | 10 × 2 outputs × 2 dims | ~15 min |
| 5 — Intervention + HAPA | 10 × (classification + 2 outputs × 4 dims) | ~25 min |
| **Total** | | **~90–120 min** |

Recommended: split into **Session 1** (Parts 1–2, ~45 min) and
**Session 2** (Parts 3–5, ~55 min).

---

## Survey Blueprint Reference

See `main.pdf` for the complete configuration reference, including:
- Exact vignette and context text for each case and part
- Output A/B placeholder locations (with counterbalancing annotations)
- Rating item wording and Likert anchors for each dimension
- HAPA phase classification item placement instructions
- Appendix A: complete network context and treatment targets for Parts 3–4
