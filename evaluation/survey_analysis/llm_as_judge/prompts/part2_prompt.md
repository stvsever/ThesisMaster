<!--
PROMPT_VERSION: 2026-05-01-v3-signed-comparison
PART_INDEX: 2
PART_TITLE: 02_Identifying_Modifiable_Treatment_Options
MODEL: google/gemini-3.1-flash-lite-preview
-->

# Part 2 - Identifying modifiable treatment options

You are comparing two anonymous answers to the same Qualtrics task. The
clinician and PHOENIX were both asked to generate 3..5 modifiable treatment
options: behaviours, routines, strategies, or process variables the patient
can realistically change and that can be monitored daily by a mobile app.

The key distinction is important: symptoms describe what is going wrong;
treatment options describe what the patient can change.

## Case context

Free-text complaint / vignette:
```text
{{vignette}}
```

Standardised symptoms supplied to the respondent:
```json
{{standardized_symptoms_json}}
```

## Outputs to compare

Both outputs are canonicalised to the same shape:
```json
{"items": [{"label": "short treatment-option label"}]}
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
