<!--
PROMPT_VERSION: 2026-05-02-absolute-quality-research-grade
PART_INDEX: 5
PART_TITLE: 05_Mobile_Coaching_Message
-->

# Part 5 — Evaluate: personalised mobile coaching message

You are evaluating **The Output** — an anonymous response to a structured
clinical task in the PHOENIX survey.  You do **not** know whether The Output
was produced by a human clinician or by an AI system.

Rate The Output **independently** on each evaluation dimension using the
1–5 absolute quality scale from the system prompt.  Apply all mandatory
evaluation rules, especially the **anti-halo rule**: each dimension must be
rated solely on its own criterion evidence.

---

## The clinical task (what the respondent was asked to do)

> **Task:** Write a **2–4 sentence mobile coaching message** directly
> addressed to the patient (second person: "you/your").  The message must:
> - Target the primary treatment goal
> - Acknowledge or work around the stated barrier
> - Suggest a concrete, feasible next action
> - Be phone-ready (not a clinical note; no jargon; warm but professional tone)

The respondent had access to:
1. The primary complaint or problem description (provided below)
2. The treatment goal for this patient (provided below)
3. The main identified barrier to that goal (provided below)
4. A suggested coping strategy or approach (provided below)
5. The patient's HAPA motivational phase (provided below)
6. A summary of the 21-day EMA monitoring data (provided below)

**HAPA phase context:**
- **Motivational phase** (pre-intention): the patient has not yet formed
  a firm intention to change — the message should build motivation and
  perceived relevance.
- **Volitional phase** (post-intention, pre-action): the patient intends to
  change but needs a specific plan — emphasise implementation intention.
- **Action/maintenance phase**: the patient is already engaging with the
  target behaviour — reinforce and troubleshoot barriers.

---

## Case context available to the respondent

**Primary problem:**
{{primary_problem}}

**Treatment goal:**
{{treatment_goal}}

**Main barrier to the treatment goal:**
{{barrier}}

**Suggested coping strategy:**
{{coping_strategy}}

**Patient's HAPA phase:**
{{assigned_hapa_phase}}

**21-day EMA monitoring summary:**
```json
{{ema_summary_json}}
```

---

## Canonical output format

Valid outputs for this task use the following structure:
```json
{"message": "2–4 sentence coaching message addressed directly to the patient."}
```

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
