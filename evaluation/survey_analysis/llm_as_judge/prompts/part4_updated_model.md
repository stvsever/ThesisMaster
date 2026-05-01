<!--
PROMPT_VERSION: 2026-05-01-v1
PART_INDEX: 4
PART_TITLE: Updated observational model
MODEL: google/gemini-2.5-flash
TEMPERATURE: 0.7
-->

# Task — Part 4: Updated observational model

You are evaluating two candidate revisions of the observational model after a
21-day EMA monitoring period. Each output selects which predictor variables
to keep, drop, or add, with an optional free-text note.

## Case context

Patient vignette:
```
{{vignette}}
```

Initial Part 2 model (predictors before revision):
```json
{{initial_model_json}}
```

Treatment-target ranking from Part 3:
```json
{{ranking_json}}
```

EMA monitoring summary (the data driving the revision):
```json
{{ema_summary_json}}
```

## Outputs to compare

Both outputs follow the canonical shape
``{"selected_options": [str, ...], "note": Optional[str]}``.

### Output A
```json
{{output_a_json}}
```

### Output B
```json
{{output_b_json}}
```

## Dimensions to rate

{{dimensions_block}}

## Required JSON output

Use the exact dimension keys above. Return STRICT JSON only.
