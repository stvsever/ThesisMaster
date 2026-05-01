<!--
PROMPT_VERSION: 2026-05-01-v1
PART_INDEX: 1
PART_TITLE: Operationalisation of mental state
MODEL: google/gemini-2.5-flash
TEMPERATURE: 0.7
-->

# Task — Part 1: Operationalisation of mental state

You are evaluating two candidate operationalisations of a patient's complaint
into a small set of clinically usable items, each shaped as
``label | description``.

## Case context

Patient vignette (free text):
```
{{vignette}}
```

## Outputs to compare

Both outputs follow the canonical shape
``{"items": [{"label": str, "description": str}, ...]}``.

### Output A
```json
{{output_a_json}}
```

### Output B
```json
{{output_b_json}}
```

## Dimensions to rate

For each dimension below, rate Output A and Output B on the 1..7 anchored
Likert scale and provide a one-sentence justification.

{{dimensions_block}}

## Required JSON output

Use the exact dimension keys above. Return STRICT JSON only.
