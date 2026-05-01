<!--
PROMPT_VERSION: 2026-05-02-absolute-quality-research-grade
PART_INDEX: 2
PART_TITLE: 02_Identifying_Modifiable_Treatment_Options
-->

# Part 2 — Evaluate: modifiable treatment-option labels

You are evaluating **The Output** — an anonymous response to a structured
clinical task in the PHOENIX survey.  You do **not** know whether The Output
was produced by a human clinician or by an AI system.

Rate The Output **independently** on each evaluation dimension using the
1–5 absolute quality scale from the system prompt.  Apply all mandatory
evaluation rules, especially the **anti-halo rule**: each dimension must be
rated solely on its own criterion evidence.

---

## The clinical task (what the respondent was asked to do)

> **Task:** Based on the standardised symptom labels and the patient
> background below, identify 3–5 **modifiable treatment options** — concrete
> behaviours, routines, or strategies the patient or therapist can realistically
> change.  Provide short labels only; no rationales or measurement
> definitions.

The respondent had access to:
1. Standardised symptom labels (the output of Part 1, provided below)
2. Case background notes (provided below)

The respondent did **not** have access to any network data, EMA data, or
pre-defined answer key for Part 2.

**Important framing for your evaluation:** The distinction between this task
and Part 1 is critical.  Part 1 identifies *symptom nodes*; Part 2 identifies
*modifiable behaviour/routine nodes* that can be connected to those symptoms
in a bipartite network.  An option that is actually a symptom re-labelled as
an action is a boundary violation.

---

## Case context available to the respondent

**Standardised symptom labels (output of Part 1, given to respondent):**
```json
{{standardized_symptoms_json}}
```

**Case background notes:**
```json
{{case_notes_json}}
```

---

## Canonical output format

Valid outputs for this task use the following structure:
```json
{"items": [{"label": "short treatment-option label"}, ...]}
```
Between 3 and 5 items are expected.

**The Output to evaluate:**
```json
{{the_output_json}}
```

---

## Evaluation dimensions

{{dimensions_block}}

---

## Response format

Return the strict `ratings` JSON schema defined in the system prompt.
Include exactly one entry per dimension key listed above.
