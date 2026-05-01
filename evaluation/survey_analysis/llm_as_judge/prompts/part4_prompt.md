<!--
PROMPT_VERSION: 2026-05-01-v3-signed-comparison
PART_INDEX: 4
PART_TITLE: 04_Selecting_EMA_Measurement_Items
MODEL: google/gemini-3.1-flash-lite-preview
-->

# Part 4 - Selecting concrete EMA measurement items

You are comparing two anonymous selections of EMA items. The task is to
select exactly six concrete daily mobile measurement items: two items for
each of three abstract treatment targets.

The best answer selects items that directly operationalise each target,
avoid tangential items, remain measurable on a phone, and provide data that
can guide later coaching.

## Case context

Free-text complaint / vignette:
```text
{{vignette}}
```

Abstract treatment targets for this case:
```json
{{treatment_targets_json}}
```

Candidate EMA items shown to the respondent, if available:
```json
{{candidate_ema_items_json}}
```

EMA monitoring summary, if available:
```json
{{ema_summary_json}}
```

## Outputs to compare

Both outputs are canonicalised to the same shape:
```json
{"selected_options": ["item label", "item label"]}
```

### Output A
```json
{{output_a_json}}
```

### Output B
```json
{{output_b_json}}
```

## Dimensions

For each dimension, return one signed A-over-B score on the -9..+9 scale.
Positive scores favour Output A; negative scores favour Output B; zero means
no meaningful difference.

{{dimensions_block}}

## JSON only

Return the strict `comparisons` JSON schema from the system prompt. Use every
dimension key exactly once.
