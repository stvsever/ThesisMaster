# LLM-as-Judge

Module that compares PHOENIX outputs to HCP outputs in a double-blind manner
using a single LLM judge (Google Gemini 2.5 Flash, served via OpenRouter).

## Why an LLM judge?

The original evaluation design called for a second round of HCP raters
(POST-survey). That design imposed a substantial recruitment burden, took
weeks to execute, and was hard to repeat when the underlying PHOENIX
outputs changed. A double-blind LLM judge gives us:

- **Speed** (full re-runs in minutes, not weeks).
- **Determinism (modulo temperature)** with reproducible seeds and
  prompt-version pinning.
- **Symmetric blinding**: by canonicalising both outputs to identical JSON
  shapes, the judge cannot tell which side is human.
- **Per-dimension granularity**: each output is rated on 5..7 dimensions per
  part, enabling LMM-based effects rather than a single overall score.

## Why Gemini 2.5 Flash?

- Strong instruction-following on JSON-mode prompts.
- Fast enough to run 5 stochastic samples per (case, part) without blowing
  out the time budget.
- Generally available through OpenRouter, so this evaluation doesn't pin
  the project to a specific vendor account.
- Cost is negligible at this evaluation's volume (10 cases x 5 parts x 5
  runs = 250 calls).

The choice is a hyperparameter; substituting any other OpenRouter-served
model is a one-line change in the orchestrator.

## Blinding protocol

For each ``(case_id, part, judge_run)`` triple:

1. We compute a deterministic seed from ``hash((case_id, part, judge_run))``.
2. The least significant bit of the seed decides whether ``A`` is the
   PHOENIX output or the HCP output.
3. The judge sees only ``Output A`` and ``Output B``. It receives no
   metadata about either side.
4. We persist the unblinding key alongside the raw response.

Across cases this gives roughly 50/50 A-vs-B PHOENIX assignment, balanced
by construction over the 50 (case x part) cells.

## Prompt-engineering choices

- 1..7 anchored Likert scale with concrete anchor examples baked into each
  dimension's prompt block (low/mid/high). Six points was rejected as
  forcing collapsed midpoints; nine points was rejected because Gemini's
  ratings tend to cluster in the 1..5 range without enough headroom.
- Per-dimension justification field forces the judge to commit to a reason
  for each rating, which improves rating stability across runs.
- Strict JSON output with explicit "do not output prose" rule. Combined
  with a JSON re-prompt step on parse failure, this gives near-100% parse
  success at temperature 0.7.
- Fixed `PROMPT_VERSION` constant (in ``dimensions.py``) is recorded in
  every long-format row so future analyses can stratify by prompt version.

## Adding a new dimension

1. Append a :class:`Dimension` to the relevant ``PARTn_DIMENSIONS`` list in
   :mod:`dimensions`.
2. Bump ``PROMPT_VERSION``.
3. Re-run the judge (``run_pipeline.sh --mode pseudo`` first to verify, then
   ``--mode real`` if the change should affect the headline numbers).
4. The analysis stage picks the new dimension up automatically — no
   per-dimension code change needed.

## Files

| File | Purpose |
| --- | --- |
| `dimensions.py`         | Per-part dimension specs + prompt-version tag. |
| `prompts/*.md`          | Per-part prompt templates with placeholders. |
| `output_schema.py`      | JSON schema + tolerant parser. |
| `openrouter_client.py`  | OpenAI-SDK / urllib wrapper for OpenRouter. |
| `pseudo_judge.py`       | Local stand-in that mirrors the API. |
| `judge_runner.py`       | Orchestration: blinding, retries, persistence. |

## Pseudo-mode ground truth

`pseudo_judge.GROUND_TRUTH_EFFECTS` documents the per-dimension PHOENIX-HCP
effect injected when running in pseudo mode. The downstream analysis is
expected to recover these effects within bootstrap CIs, providing a
self-test for the LMM and TOST machinery.
