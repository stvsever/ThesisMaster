<!--
PROMPT_VERSION: 2026-05-01-v1
PART_INDEX: 5
PART_TITLE: Tailored intervention message
MODEL: google/gemini-2.5-flash
TEMPERATURE: 0.7
-->

# Task — Part 5: Tailored intervention message

You are evaluating two candidate 2..3 sentence motivational messages tailored
to the patient's HAPA phase (pre-intentional / intentional / action /
maintenance).

## Case context

Patient vignette:
```
{{vignette}}
```

Updated observational model (the cues the message should reference):
```json
{{updated_model_json}}
```

Top treatment targets (the message should serve these):
```json
{{ranking_json}}
```

Assigned HAPA phase for this case (from the survey blueprint):
```
{{assigned_hapa_phase}}
```

## Outputs to compare

Both outputs follow the canonical shape
``{"message": str, "hapa_phase": Optional[str]}``. The ``hapa_phase`` field
in the outputs is informational only; the assigned phase above is the
ground truth for this case.

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

In addition to ``ratings``, populate ``extra`` with your independent HAPA
classification of each message:

```
"extra": {
  "hapa_phase_a": "pre_intentional" | "intentional" | "action" | "maintenance",
  "hapa_phase_b": "pre_intentional" | "intentional" | "action" | "maintenance"
}
```

Use the exact dimension keys. Return STRICT JSON only.
