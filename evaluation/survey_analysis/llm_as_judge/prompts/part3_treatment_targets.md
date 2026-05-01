<!--
PROMPT_VERSION: 2026-05-01-v1
PART_INDEX: 3
PART_TITLE: Treatment-target prioritisation
MODEL: google/gemini-2.5-flash
TEMPERATURE: 0.7
-->

# Task — Part 3: Treatment-target prioritisation

You are evaluating two candidate rankings of five treatment-options, where
each option is a node in a bipartite predictor-criterion network.

## Case context

Patient vignette:
```
{{vignette}}
```

Bipartite network summary (predictor x criterion edges, weights in [0,1]):
```json
{{network_summary_json}}
```

EMA monitoring summary (21-day rolling means, peaks, trends):
```json
{{ema_summary_json}}
```

Treatment options (BO-1 .. BO-5):
```json
{{treatment_options_json}}
```

## Outputs to compare

Both outputs follow the canonical shape
``{"ranking": [{"rank": 1, "option_id": "BO-X"}, ..., {"rank": 5, "option_id": "BO-Y"}]}``
with rank 1 = highest priority.

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
