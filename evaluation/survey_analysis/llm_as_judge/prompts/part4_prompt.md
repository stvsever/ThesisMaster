<!--
PROMPT_VERSION: 2026-05-02-absolute-quality-research-grade
PART_INDEX: 4
PART_TITLE: 04_Selecting_EMA_Measurement_Items
-->

# Part 4 — Evaluate: EMA item selection

You are evaluating **The Output** — an anonymous response to a structured
clinical task in the PHOENIX survey.  You do **not** know whether The Output
was produced by a human clinician or by an AI system.

Rate The Output **independently** on each evaluation dimension using the
1–5 absolute quality scale from the system prompt.  Apply all mandatory
evaluation rules, especially the **anti-halo rule**: each dimension must be
rated solely on its own criterion evidence.

---

## The clinical task (what the respondent was asked to do)

> **Task:** From the 20-item EMA candidate list below, select **exactly
> 6 items** — 2 items per treatment target — that best operationalise the
> three abstract treatment targets provided.  Choose the most **direct**
> available operationalisation; avoid side-path items even if they seem
> related.  Return only the 6 selected item labels, no explanations.

The respondent had access to:
1. The three abstract treatment targets (provided below)
2. The full 20-item EMA candidate list (provided below)

The respondent did **not** have access to a correct answer key.  This is
a constrained selection task: only items from the candidate list are valid
selections.

**Selection principles the respondent was given:**
- Prefer items that are *direct* operationalisations of the target (not
  tangential proxies)
- Prefer items that vary meaningfully day-to-day (not stable traits)
- Two items per target maintains coverage balance; distributing 3+1 or 4+2
  is a balance violation

---

## Case context available to the respondent

**Three abstract treatment targets:**
```json
{{treatment_targets_json}}
```

**20-item EMA candidate list (all valid selections must come from this list):**
```json
{{candidate_ema_items_json}}
```

---

## Canonical output format

Valid outputs for this task use the following structure:
```json
{"selected_options": ["Item label 1", "Item label 2", "Item label 3",
                      "Item label 4", "Item label 5", "Item label 6"]}
```
Exactly 6 items, each matching a label from the candidate list.

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
