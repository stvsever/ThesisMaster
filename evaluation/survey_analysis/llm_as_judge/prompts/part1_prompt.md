<!--
PROMPT_VERSION: 2026-05-01-v3-signed-comparison
PART_INDEX: 1
PART_TITLE: 01_Identifying_Symptoms
MODEL: google/gemini-3.1-flash-lite-preview
-->

# Part 1 - Identifying the most important symptoms

You are comparing two anonymous answers to the same Qualtrics task. The
clinician and PHOENIX were both asked to identify 3..6 current complaint or
state dimensions from the same free-text case.

Judge symptom labels as labels only: concise, clinically meaningful,
currently present, non-overlapping, and suitable for later EMA monitoring.
Do not require explanatory text if the survey task only asked for labels.

## Case context

Free-text complaint / vignette:
```text
{{vignette}}
```

Standardised case notes, if available:
```json
{{case_notes_json}}
```

## Outputs to compare

Both outputs are canonicalised to the same shape:
```json
{"items": [{"label": "short symptom label"}]}
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
