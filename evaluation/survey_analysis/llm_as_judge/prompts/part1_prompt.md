<!--
PROMPT_VERSION: 2026-05-02-absolute-quality-research-grade
PART_INDEX: 1
PART_TITLE: 01_Identifying_Symptoms
-->

# Part 1 — Evaluate: symptom label identification

You are evaluating **The Output** — an anonymous response to a structured
clinical task in the PHOENIX survey.  You do **not** know whether The Output
was produced by a human clinician or by an AI system.

Rate The Output **independently** on each evaluation dimension using the
bipolar −10 to +10 absolute quality scale from the system prompt.  Apply all mandatory
evaluation rules, especially the **anti-halo rule**: each dimension must be
rated solely on its own criterion evidence.

---

## The clinical task (what the respondent was asked to do)

> **Task:** Identify the current complaint and state dimensions expressed or
> strongly implied in the patient vignette below.  Provide 3–6 **short
> symptom labels** only — no diagnoses, no explanations, no treatment
> suggestions.

The respondent had access to:
1. The patient vignette (complaint text, provided below)
2. General case background notes (provided below)

The respondent did **not** have access to any standardised symptom list,
network data, or pre-defined answer key.

---

## Case context available to the respondent

**Patient vignette (free complaint text):**
```text
{{vignette}}
```

**Case background notes:**
```json
{{case_notes_json}}
```

---

## Canonical output format

Valid outputs for this task use the following structure:
```json
{"items": [{"label": "short symptom label"}, ...]}
```
Between 3 and 6 items are expected.

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
