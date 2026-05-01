<!--
PROMPT_VERSION: 2026-05-01-v3-signed-comparison
PART_INDEX: 5
PART_TITLE: 05_Mobile_Coaching_Message
MODEL: google/gemini-3.1-flash-lite-preview
-->

# Part 5 - Mobile coaching message

You are comparing two anonymous mobile coaching messages for the same case.
The message should be short enough for a smartphone, written directly to the
patient, and designed to support a concrete next behaviour.

The survey task asks for 2..4 sentences with warm, direct, professional tone,
no clinical jargon or diagnostic labels, clear relation to the primary
treatment goal and main barrier, and one feasible next action.

## Case context

Free-text complaint / vignette:
```text
{{vignette}}
```

Primary problem:
```text
{{primary_problem}}
```

Treatment goal:
```text
{{treatment_goal}}
```

Main barrier:
```text
{{barrier}}
```

Recommended coping strategy / behaviour-shift logic, if available:
```text
{{coping_strategy}}
```

Assigned HAPA phase, if available:
```text
{{assigned_hapa_phase}}
```

## Outputs to compare

Both outputs are canonicalised to the same shape:
```json
{"message": "2..4 sentence mobile coaching message"}
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

## Extra field

If the assigned HAPA phase is available, add your independent phase
classification for each message:

```json
"extra": {
  "hapa_phase_a": "pre_intentional|intentional|action|maintenance|unknown",
  "hapa_phase_b": "pre_intentional|intentional|action|maintenance|unknown"
}
```

If HAPA phase is not available, use an empty object: `"extra": {}`.

## JSON only

Return the strict `comparisons` JSON schema from the system prompt. Use every
dimension key exactly once.
