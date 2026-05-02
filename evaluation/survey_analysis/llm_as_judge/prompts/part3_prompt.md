<!--
PROMPT_VERSION: 2026-05-02-absolute-quality-research-grade
PART_INDEX: 3
PART_TITLE: 03_Prioritising_Treatment_Targets
-->

# Part 3 — Evaluate: treatment-target priority ranking

You are evaluating **The Output** — an anonymous response to a structured
clinical task in the PHOENIX survey.  You do **not** know whether The Output
was produced by a human clinician or by an AI system.

Rate The Output **independently** on each evaluation dimension using the
bipolar −10 to +10 absolute quality scale from the system prompt.  Apply all mandatory
evaluation rules, especially the **anti-halo rule**: each dimension must be
rated solely on its own criterion evidence.

---

## The clinical task (what the respondent was asked to do)

> **Task:** Rank the **five standardised treatment options** (provided below)
> from highest priority (rank 1) to lowest priority (rank 5) for this patient,
> using the bipartite symptom-treatment network and the 21-day EMA monitoring
> data as primary inputs.  Assign each of the five options exactly one rank;
> no ties.

The respondent had access to:
1. The five standardised treatment options with their IDs (provided below)
2. The bipartite symptom-treatment network summary — edge weights (positive
   and negative), degree, and centrality (provided below)
3. The 21-day EMA monitoring summary — symptom burden, frequency, trends,
   and option engagement data (provided below)

**Key network interpretation rules the respondent was given:**
- A **positive edge** means increasing the treatment-option behaviour is
  associated with *increasing* a connected symptom (risk/maintaining factor).
  Reducing this option reduces the symptom.
- A **negative edge** means increasing the option is associated with
  *reducing* a connected symptom (protective factor).  Increasing this option
  benefits the patient.
- Edge weight magnitude reflects the strength of the symptom-treatment
  association.  Higher total connectivity (more edges, higher combined weight)
  generally indicates a higher priority target.
- Current EMA burden modifies priority: a strong edge to an already
  favourable, low-burden symptom is less urgent than the same edge to a
  currently high-burden symptom.

Your evaluation should assess whether **The Output** correctly applies
this logic to the data provided below.

---

## Case context available to the respondent

**Five standardised treatment options (fixed IDs):**
```json
{{standardized_treatment_options_json}}
```

**Bipartite symptom-treatment network summary:**
```json
{{network_summary_json}}
```

**21-day EMA monitoring summary:**
```json
{{ema_summary_json}}
```

---

## Canonical output format

Valid outputs for this task use the following structure:
```json
{"ranking": [{"rank": 1, "option_id": "BO-X"}, {"rank": 2, "option_id": "BO-Y"}, ...]}
```
All five options ranked from 1 (highest priority) to 5 (lowest priority).

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
