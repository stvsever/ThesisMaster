# LLM-as-Judge

This module implements the Phase 2 double-blind judge. It compares an HCP
output and a PHOENIX output for the same case and survey part, but the model
only sees anonymous `Output A` and `Output B`.

## Judge Backend

Default real backend:

```
google/gemini-3.1-flash-lite-preview
```

through OpenRouter. Pseudo mode uses `pseudo_judge.py` and never calls the
network.

## Blinding

For every `(case_id, part, judge_run)`:

1. `judge_runner.assign_blind_labels()` deterministically assigns PHOENIX and
   HCP to A/B.
2. Both outputs are canonicalised to the same JSON shape before prompting.
3. The prompt contains no source metadata.
4. The raw response JSON stores the blinding key for reproducibility.
5. `judgments_long.csv` stores the unblinded signed PHOENIX-vs-HCP score.

## Scale

The judge returns one signed comparison per dimension:

```
-9 = Output B is decisively better
-6 = Output B is strongly better
-3 = Output B is modestly better
 0 = no meaningful difference / tie
+3 = Output A is modestly better
+6 = Output A is strongly better
+9 = Output A is decisively better
```

The runner converts this to `score`, where:

- `score > 0`: PHOENIX preferred;
- `score < 0`: HCP preferred;
- `score = 0`: tie / no meaningful difference.

This is intentionally pairwise. It prevents the judge from giving two
independent absolute ratings with inconsistent calibration across prompts.

## JSON Schema

The model must return strict JSON:

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

`winner` must match the sign. `confidence` is 1..5 and is descriptive; it is
not used as an inferential weight in the current statistics.

## Prompt Files

| File | Purpose |
| --- | --- |
| `prompts/system_prompt.md` | Shared blind judge role, scale, JSON rules |
| `prompts/part1_operationalization.md` | Symptom-label comparison |
| `prompts/part2_initial_model.md` | Modifiable treatment-option comparison |
| `prompts/part3_treatment_targets.md` | Treatment-target ranking comparison |
| `prompts/part4_updated_model.md` | EMA item-selection comparison |
| `prompts/part5_intervention.md` | Mobile coaching-message comparison |

The Part 2 and Part 4 filenames retain the older module names for import
compatibility, but the prompt content matches the current Qualtrics survey:
Part 2 is treatment-option generation and Part 4 is concrete EMA item
selection.

## Dimensions

All dimension definitions live in `dimensions.py`; the prompt renderer injects
their goal, rationale, and comparative examples into the part prompt.

The current prompt version is:

```
2026-05-01-v2-signed-comparison
```

Bump `PROMPT_VERSION` whenever dimensions, anchors, prompt wording, or output
schema change.

## Output Artefacts

Long-format scored data:

```
evaluation/survey_analysis/data/04_judgments/judgments_long.csv
```

Columns:

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

Raw judge responses:

```
evaluation/survey_analysis/data/04_judgments/raw/<part>/case_<case>_run_<run>.json
```

## Pseudo Mode

`pseudo_judge.py` injects known PHOENIX-HCP effects per dimension so the
analysis can be validated without an API key. These are not clinical claims;
they only test whether the pipeline recovers a realistic mix of PHOENIX
advantages, ties/equivalence, and slight HCP advantages.

Run:

```bash
python evaluation/survey_analysis/pipeline.py --mode pseudo
```

## Real Mode Checklist

Before real OpenRouter judging:

1. Export completed Qualtrics CSV to `evaluation/qualtrics/data/01_raw/`.
2. Save PHOENIX outputs to `data/03_system/system_outputs.json`.
3. Save complete shared case inputs to `data/01_raw/case_contexts.json`.
4. Set `OPENROUTER_API_KEY`.
5. Run `python evaluation/survey_analysis/pipeline.py --mode real --n-runs 5`.

If an old `judgments_long.csv` has the pre-v2 absolute-rating header, the
runner stops and asks you to archive/delete it. This avoids mixing 1..7
absolute ratings with -9..+9 signed comparisons.
