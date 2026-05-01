<!--
PROMPT_VERSION: 2026-05-01-v1
PART_INDEX: 2
PART_TITLE: Initial observational model
MODEL: google/gemini-2.5-flash
TEMPERATURE: 0.7
-->

# Task — Part 2: Initial observational model

You are evaluating two candidate sets of EMA-feasible predictor variables,
their measurement schedule, and their decision criteria. The goal is a
predictor set that lets a 21-day momentary monitoring run identify leverage
points for intervention.

## Case context

Patient vignette:
```
{{vignette}}
```

Operationalised items from Part 1 (the predictors should map onto these):
```json
{{operationalisation_json}}
```

## Outputs to compare

Both outputs follow the canonical shape
``{"items": [{"predictor": str, "measurement": str, "criteria": str}, ...]}``.

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
