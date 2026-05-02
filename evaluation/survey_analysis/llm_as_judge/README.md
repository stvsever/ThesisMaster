# LLM-as-Judge

This module implements the double-blind Phase 2 judge for the PHOENIX survey
analysis. The judge rates one anonymous output at a time using the same case
context for PHOENIX and HCP outputs. Source labels are hidden during judging
and restored only for statistical analysis.

## Backend

Default real backend:

```text
google/gemini-3.1-flash-lite-preview
```

via OpenRouter. Pseudo mode uses `pseudo_judge.py` and makes no network calls.

## Blinding Contract

For every `(case_id, part, judge_run)`:

1. `assign_blind_labels()` deterministically assigns HCP and PHOENIX to blind labels A/B.
2. Each source is judged in a separate single-output call.
3. Both outputs are canonicalized to exactly the same part-specific JSON shape.
4. Shared inputs are supplied in case context, not inside either candidate.
5. `judgments_long.csv` stores source labels only after unblinding.

Compared output shapes:

| Part | Shape |
| --- | --- |
| 1 | `{"items": [{"label": "..."}]}` |
| 2 | `{"items": [{"label": "..."}]}` |
| 3 | `{"ranking": [{"rank": 1, "option_id": "BO-1"}]}` |
| 4 | `{"selected_options": ["..."]}` |
| 5 | `{"message": "..."}` |

## Quality Scale

The judge returns one absolute quality rating per dimension:

```text
-10 = Clinically harmful or unusable
-5  = Substantially below acceptable quality
0   = Acceptable minimum clinical quality
+5  = Strong clinical quality
+10 = Exemplary clinical quality for the task
```

The judge is instructed to avoid halo bias, length bias, fluency bias, and
source guessing. A score of 0 means the output is adequate, not excellent.

## Required JSON

```json
{
  "ratings": [
    {
      "dimension": "complaint_coverage",
      "score": 7,
      "confidence": 4,
      "justification": "Includes sleep disruption, withdrawal, and low mood as separate symptom labels."
    }
  ],
  "extra": {}
}
```

`confidence` is recorded for supplementary sensitivity analyses. It is not an
inferential weight in the primary model.

## Prompt Files

| File | Purpose |
| --- | --- |
| `prompts/system_prompt.md` | Shared blinded judge role, anti-bias rules, scale, JSON schema |
| `prompts/part1_prompt.md` | Symptom-label rating |
| `prompts/part2_prompt.md` | Modifiable treatment-option rating |
| `prompts/part3_prompt.md` | Treatment-target ranking rating |
| `prompts/part4_prompt.md` | EMA item-selection rating |
| `prompts/part5_prompt.md` | Mobile coaching-message rating |

The prompt identifier is:

```text
2026-05-02-absolute-quality-research-grade
```

Update `PROMPT_VERSION` whenever dimension definitions, anchors, prompt
wording, model-facing schema, or scale definitions change.

## Dimensions

All dimension definitions live in `dimensions.py`. The prompt renderer injects
the goal, rationale, and bipolar -10..+10 anchor examples for every dimension.

- Part 1: label-format adherence, complaint coverage, symptom boundaries,
  granularity, non-redundancy, interoperability, EMA measurability.
- Part 2: label-format adherence, modifiability, relevance, causal
  plausibility, EMA feasibility, symptom-option separation, diversity,
  precision.
- Part 3: ranking validity, network alignment, current-state integration,
  edge-direction interpretation, top-target defensibility, feasibility,
  rank coherence.
- Part 4: valid candidate selection, target-item mapping, 2-per-target
  balance, concreteness, directness, dynamic informativeness, burden, coaching
  feedback value.
- Part 5: phone-message format, goal alignment, barrier responsiveness,
  feasible action, behaviour-change potential, tone, concision,
  personalisation, safety.

## Output Artefacts

Long-format scored data:

```text
evaluation/survey_analysis/data/04_judgments/judgments_long.csv
```

Main columns:

| Column | Meaning |
| --- | --- |
| `case_id`, `part`, `dimension`, `judge_run` | Evaluation cell |
| `entity` | `phoenix` or `hcp`, restored after blinding |
| `quality_score` | Bipolar -10 to +10 absolute quality rating |
| `source_label` | Blind label A or B during judging |
| `confidence` | Judge confidence, 1 to 5 |
| `justification` | Evidence-based sentence |
| `prompt_version`, `model`, `timestamp` | Reproducibility metadata |

Raw responses are saved under:

```text
evaluation/survey_analysis/data/04_judgments/raw/<part>/case_<case>_run_<run>_<label>.json
```

## Running

Pseudo mode:

```bash
python evaluation/survey_analysis/pipeline.py --mode pseudo --n-runs 3
```

Real OpenRouter mode:

```bash
export OPENROUTER_API_KEY=...
python evaluation/survey_analysis/pipeline.py --mode real --judge openrouter --n-runs 3
```

Real HCP parsing with pseudo judging for the current single-row C03 smoke path:

```bash
python evaluation/phoenix_outputs/run.py prepare-fixture-analysis
python evaluation/survey_analysis/pipeline.py --mode real --judge pseudo --cases C03 --n-runs 3
```
