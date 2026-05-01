# PHOENIX Outputs for Survey Evaluation

This folder prepares the system side of the Phase 2 double-blind comparison.
It has one job: make PHOENIX receive the same non-image information that HCPs
received in Qualtrics, then save PHOENIX answers in the same canonical shape as
parsed HCP answers.

The live Qualtrics survey shows Part 3 as images. PHOENIX should not receive
those images; it receives the exact labels and numeric edge weights behind the
figures instead.

## Folder Layout

| Path | Purpose |
| --- | --- |
| `qualtrics_inputs.py` | Extracts exact case inputs from the Qualtrics LaTeX sources. |
| `canonicalize_outputs.py` | Converts real PHOENIX output JSON to the judge-ready schema and validates it. |
| `rule_based_fixture.py` | Deterministic dry-run fixture only; not a thesis result. |
| `run.py` | CLI entry point. |
| `data/inputs/qualtrics_case_inputs.json` | Extracted exact inputs for all 10 cases. |
| `data/inputs/case_contexts_for_judge.json` | Case context copied into judge prompts. |
| `data/outputs/system_outputs.json` | Canonical PHOENIX/system outputs when generated. |

## Canonical Output Contract

PHOENIX outputs must be a JSON object keyed by case id:

```json
{
  "C03": {
    "part1": {"items": [{"label": "Emotionele uitputting"}]},
    "part2": {"items": [{"label": "Werk-privegrens"}]},
    "part3": {"ranking": [{"rank": 1, "option_id": "BO-2"}]},
    "part4": {"selected_options": ["Herstelactiviteit buiten het werk actief uitgevoerd (min)"]},
    "part5": {"message": "2..4 sentence mobile coaching message"}
  }
}
```

Do not include source-specific fields in the compared output. HAPA phase,
network edges, candidate item lists, and treatment targets belong in the
case-context JSON, not inside Output A or Output B.

## CLI Usage

Extract exact inputs from the current Qualtrics source:

```bash
python evaluation/phoenix_outputs/run.py extract-inputs
```

Write an empty output template for the real PHOENIX run:

```bash
python evaluation/phoenix_outputs/run.py write-template
```

Canonicalize actual PHOENIX output after the engine run:

```bash
python evaluation/phoenix_outputs/run.py canonicalize \
  --raw /path/to/raw_phoenix_outputs.json
```

Copy prepared contexts and canonical outputs into the analysis pipeline:

```bash
python evaluation/phoenix_outputs/run.py sync-to-analysis
```

Dry-run the full dataflow before real PHOENIX outputs exist:

```bash
python evaluation/phoenix_outputs/run.py prepare-fixture-analysis
python evaluation/survey_analysis/pipeline.py \
  --mode real \
  --judge pseudo \
  --qualtrics-csv evaluation/qualtrics/data/01_raw/Masterproef_May\ 1,\ 2026_15.25.csv \
  --cases C03 \
  --parts part1 part2 part3 part4 part5
```

The dry-run fixture is only for software validation. Replace
`data/outputs/system_outputs.json` with canonicalized real PHOENIX outputs
before the actual LLM-as-judge run.

## Real PHOENIX Run Inputs

For each case, `qualtrics_case_inputs.json` contains:

- Part 1: free-text complaint/vignette.
- Part 2: standardised symptoms supplied to HCPs.
- Part 3: treatment-option labels, monitoring summary, symptom labels, and
  numeric network edges.
- Part 4: three abstract treatment targets and the 20 candidate EMA items.
- Part 5: primary problem, treatment goal, barrier, and coping strategy.

That is the exact non-image information required for the five PHOENIX prompts.

