# LLM-as-Judge

This module implements the double-blind Phase 2 judge. The model receives the
same case context for both candidates and compares anonymous `Output A` and
`Output B`. It never sees source labels.

## Backend

Default real backend:

```text
google/gemini-3.1-flash-lite-preview
```

through OpenRouter. Pseudo mode uses `pseudo_judge.py` and makes no network
calls.

## Blinding Contract

For every `(case_id, part, judge_run)`:

1. `assign_blind_labels()` deterministically assigns HCP and PHOENIX to A/B.
2. Both outputs are canonicalised to exactly the same part-specific JSON shape.
3. Shared inputs are supplied in case context, not inside either candidate.
4. The raw response stores the blinding key for reproducibility.
5. `judgments_long.csv` stores the unblinded signed PHOENIX-vs-HCP score.

Compared output shapes:

| Part | Shape |
| --- | --- |
| 1 | `{"items": [{"label": "..."}]}` |
| 2 | `{"items": [{"label": "..."}]}` |
| 3 | `{"ranking": [{"rank": 1, "option_id": "BO-1"}]}` |
| 4 | `{"selected_options": ["..."]}` |
| 5 | `{"message": "..."}` |

## Signed Scale

The judge returns one signed comparison per dimension:

```text
-9 = Output B decisively better than Output A
-6 = Output B strongly better
-3 = Output B modestly better
 0 = no meaningful difference / tie
+3 = Output A modestly better
+6 = Output A strongly better
+9 = Output A decisively better
```

The runner converts the A/B score to `score = PHOENIX - HCP`. Positive values
favour PHOENIX, negative values favour the HCP output.

## Required JSON

```json
{
  "comparisons": [
    {
      "dimension": "complaint_coverage",
      "score": 3,
      "winner": "A",
      "confidence": 4,
      "justification": "A covers sleep and withdrawal while B misses withdrawal."
    }
  ],
  "extra": {}
}
```

`winner` must match the score sign. `confidence` is descriptive and is not an
inferential weight.

## Prompt Files

| File | Purpose |
| --- | --- |
| `prompts/system_prompt.md` | Shared blind judge role, scale, JSON rules |
| `prompts/part1_prompt.md` | Symptom-label comparison |
| `prompts/part2_prompt.md` | Modifiable treatment-option comparison |
| `prompts/part3_prompt.md` | Treatment-target ranking comparison |
| `prompts/part4_prompt.md` | EMA item-selection comparison |
| `prompts/part5_prompt.md` | Mobile coaching-message comparison |

The current prompt version is:

```text
2026-05-01-v3-signed-comparison
```

Bump `PROMPT_VERSION` whenever dimension definitions, anchors, prompt wording,
or compared output schema change.

## Dimensions

All dimension definitions live in `dimensions.py`. The prompt renderer injects
the goal, rationale, and comparative anchor examples for every dimension.

The dimensions explicitly check both task validity and clinical quality:

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
| `score` | Unblinded PHOENIX-vs-HCP signed score |
| `raw_score_a_over_b` | Original A-vs-B model output |
| `source_a`, `source_b` | Blinding key |
| `winner_blind` | A/B/TIE before unblinding |
| `winner_source` | phoenix/hcp/tie after unblinding |
| `confidence` | Judge confidence, 1..5 |
| `justification` | One-sentence judge rationale |
| `prompt_version`, `model`, `timestamp` | Reproducibility metadata |

Raw responses are saved under:

```text
evaluation/survey_analysis/data/04_judgments/raw/<part>/case_<case>_run_<run>.json
```

## Running

Pseudo mode:

```bash
python evaluation/survey_analysis/pipeline.py --mode pseudo
```

Real OpenRouter mode:

```bash
export OPENROUTER_API_KEY=...
python evaluation/survey_analysis/pipeline.py --mode real --judge openrouter --n-runs 5
```

Real HCP parsing with pseudo judging for the current single-row C03 smoke path:

```bash
python evaluation/phoenix_outputs/run.py prepare-fixture-analysis
python evaluation/survey_analysis/pipeline.py --mode real --judge pseudo --cases C03
```

