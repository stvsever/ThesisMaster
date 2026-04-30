# PHOENIX Evaluation — Qualtrics Survey Blueprint

This folder contains the LaTeX-based survey blueprints used to configure the
Qualtrics instruments for the two-phase PHOENIX evaluation study. Each subfolder
contains a `main.tex` source and a compiled `main.pdf` that serves as the
authoritative configuration reference for the Qualtrics survey administrator.

---

## Study Design Overview

The PHOENIX evaluation uses a **two-phase expert elicitation and blind evaluation
design** in which Phase 1 (PRE) produces a corpus of human expert clinical
outputs, and Phase 2 (POST) evaluates those outputs alongside AI outputs in a
double-blind format.

### Design schematic

```
Phase 1 — PRE (Generation)           Phase 2 — POST (Blind Evaluation)
─────────────────────────────         ────────────────────────────────────────
5 HCPs × 2 cases/HCP                  5 different HCPs
= 10 unique expert outputs             × 10 cases × 2 sources (A/B blind)
  (one per case, per part)             × 5 parts
                                       = 100 ratings per dimension per part
```

---

## Phase 1 — PRE Survey (`01_HCPs_PRE/`)

**Participants:** 5 healthcare professionals (HCP-PRE-01 through HCP-PRE-05).
None of these participants takes part in Phase 2.

**Design:** Distributed case assignment. Each HCP is assigned exactly **2 of the
10 standardised case vignettes**. All 5 clinical reasoning parts are completed
for those 2 cases only. This keeps individual burden to approximately 30–40
minutes while ensuring all 10 cases receive a genuine independent expert output.

**Case assignment:**

| Participant code | Assigned cases | Approximate profiles |
|-----------------|----------------|----------------------|
| HCP-PRE-01 | C01, C02 | Sleep/stress; Panic/avoidance |
| HCP-PRE-02 | C03, C04 | Burnout; Grief |
| HCP-PRE-03 | C05, C06 | OCD-type; Performance anxiety |
| HCP-PRE-04 | C07, C08 | Depression; ADHD |
| HCP-PRE-05 | C09, C10 | BPD-type dysregulation; Work stress |

**Five clinical tasks per case:**

| Part | Task | Response format |
|------|------|-----------------|
| 1 | Operationalization of mental state | Structured text: label \| description |
| 2 | Initial observational model | Structured text: predictor \| measurement \| criteria |
| 3 | Treatment-target identification | Rank-order (Qualtrics Rank Order item) |
| 4 | Updated observational model | Multiple Choice (checkbox) + optional note |
| 5 | Tailored intervention message | HAPA phase (MC) + 2–3 sentence message (Text Entry) |

**Qualtrics implementation:** A single screening question at the start of the
survey asks for participant code (single-choice dropdown, HCP-PRE-01 to
HCP-PRE-05). Display Logic on each case block then shows only the two cases
belonging to that participant.

**Instructiepagina:** Elk HCP-blok bevat een instructiepagina ("Het AI-systeem
en de studiecontext") die het PHOENIX-systeem uitlegt. Die pagina toont
Figuur 1 (`qualtrics_overview.png`) — een schematisch overzicht van de volledige
pipeline van klachttekst tot gepersonaliseerde coaching — direct na de
theoretische introductie, vóór de uitleg van de afzonderlijke concepten (EMA,
netwerkanalyse).

---

## Phase 2 — POST Survey (`02_HCPs_POST/`)

**Participants:** 5 healthcare professionals (HCP-POST-01 through HCP-POST-05).
These are different individuals from the Phase 1 respondents.

**Design:** Each POST evaluator rates **all 10 cases × both outputs (A and B)**
for all 5 parts. Outputs are presented blind — sources are labelled A and B
without any indication of whether each was produced by a human or by PHOENIX.

**Counterbalancing scheme:**

| Cases | Output A source | Output B source |
|-------|----------------|----------------|
| C01–C05 | PHOENIX | Human HCP (Phase 1) |
| C06–C10 | Human HCP (Phase 1) | PHOENIX |

This 5/5 counterbalance controls for any systematic preference for label A or B.
In analysis, outputs are identified by their true source after unblinding.

**Rating dimensions per part:**

| Part | Dimensions rated (per output) |
|------|-------------------------------|
| 1 — Operationalization | criterion accuracy; operationalization quality; completeness |
| 2 — Initial model | clinical appropriateness; network validity; EMA feasibility; intervention potential |
| 3 — Treatment targets | clinical priority; evidence alignment; rank coherence |
| 4 — Updated model | target alignment; measurement selection quality |
| 5 — Intervention message | HAPA phase appropriateness; message tailoring; actionability; professional tone |

**Additional Part 5 item:** Before reading outputs, each POST evaluator
independently classifies the HAPA motivational phase (pre-intentional /
intentional / action-maintenance). This binary/ordinal outcome is used to
compute inter-rater agreement (Cohen's κ) across POST evaluators and Phase 1
respondents.

**Statistical power:** 5 POST raters × 10 cases × 2 outputs = 100 ratings per
dimension per part. The primary analysis (crossed mixed-effects model) uses this
full structure to estimate PHOENIX vs. HCP effects with participant and case
random effects.

---

## PRE → POST Connection

The link between Phase 1 and Phase 2 is managed by the researcher:

1. After Phase 1 data collection, each HCP's two responses (for all five parts)
   are extracted from Qualtrics.
2. The researcher inserts the human HCP outputs into the POST Qualtrics survey
   at the appropriate positions (per the counterbalancing table in `02_HCPs_POST/main.tex`).
3. PHOENIX outputs are generated separately by running the pipeline on each of
   the 10 case vignettes.
4. POST evaluators never see the PRE assignment table; blinding is enforced by
   label assignment only.

---

## Timeline

| Phase | When | Duration |
|-------|------|---------|
| PRE data collection | Before POST setup | ~1 week (flexible, async) |
| PHOENIX pipeline runs | Concurrent with PRE | ~1 day (automated) |
| POST Qualtrics setup | After PRE complete | Researcher: ~2 hours |
| POST data collection | After setup | ~1–2 weeks |
| Analysis | After POST complete | See `survey_analysis/` |

---

## File Index

| File | Description |
|------|-------------|
| `01_HCPs_PRE/main.tex` | PRE survey blueprint (all 10 cases with Display Logic annotations) |
| `01_HCPs_PRE/main.pdf` | Compiled PDF — use for Qualtrics configuration |
| `01_HCPs_PRE/separate_HCPs/survey_AUTOMATED/generated/PHOENIX_PRE_MERGED_ALL10_FINAL.txt` | Qualtrics TXT import — alle 10 HCP-blokken samengevoegd; klaar voor import |
| `01_HCPs_PRE/total/qualtrics_overview.png` | Systeemoverzichtsfiguur (Figuur 1) — getoond op de instructiepagina van elk HCP-blok |
| `02_HCPs_POST/main.tex` | POST survey blueprint (all 10 cases × 2 outputs × 5 parts) |
| `02_HCPs_POST/main.pdf` | Compiled PDF — use for Qualtrics configuration |
| `main.tex` / `main.pdf` | Parent-level overview document (optional) |
