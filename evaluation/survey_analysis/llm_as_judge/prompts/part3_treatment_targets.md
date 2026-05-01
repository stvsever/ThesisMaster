<!--
PROMPT_VERSION: 2026-05-01-v2-signed-comparison
PART_INDEX: 3
PART_TITLE: 03_Prioritising_Treatment_Targets
MODEL: google/gemini-3.1-flash-lite-preview
-->

# Part 3 - Prioritising treatment targets

You are comparing two anonymous rankings of the same five standardised
treatment options. Rank 1 is the highest treatment priority.

The correct reasoning combines three evidence streams:

- network impact: stronger edges matter more;
- current EMA state: high burden/frequency/trend matters more;
- modifiability: the target must be realistically changeable now.

Do not judge a ranking only by whether it follows the strongest edge. The
survey instructions explicitly state that strong network relations are
necessary but not sufficient when current state is already favourable.

## Case context

Free-text complaint / vignette:
```text
{{vignette}}
```

Treatment options being ranked:
```json
{{standardized_treatment_options_json}}
```

Bipartite network summary, if available:
```json
{{network_summary_json}}
```

EMA monitoring summary:
```json
{{ema_summary_json}}
```

## Outputs to compare

Both outputs are canonicalised to the same shape:
```json
{"ranking": [{"rank": 1, "option_id": "BO-1"}, {"rank": 2, "option_id": "BO-2"}]}
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
