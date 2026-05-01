# Output Files

`system_outputs.json` is the canonical PHOENIX/system-side input for the
double-blind LLM-as-judge pipeline. Real PHOENIX outputs should be
canonicalized with:

```bash
python evaluation/phoenix_outputs/run.py canonicalize --raw /path/to/raw_phoenix_outputs.json
```

The dry-run fixture can write a compatible file for software testing, but that
fixture is not a thesis result.

