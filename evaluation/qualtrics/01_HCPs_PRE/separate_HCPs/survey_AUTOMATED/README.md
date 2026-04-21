# PHOENIX PRE HCP Survey Automation

This folder contains the **chosen deployment path** for your HCP PRE evaluation survey:

- **Platform**: Qualtrics
- **Design**: **10 separate surveys**, exactly **1 case per HCP**
- **Construction model**: generate import-ready survey files directly from the current authoritative LaTeX/PDF source package
- **Hosting requirement**: **none**

This is the best fit for your constraints.

## Current account reality

Your current UGent Qualtrics account shows that the API is **not enabled**.

That changes the decision materially:

- the Qualtrics Survey API cannot be used as the production deployment path
- a hand-written `QSF` is still not a high-confidence route
- fragile browser automation against the Qualtrics web UI is not the quality-first choice for a thesis instrument

Therefore the best method for your actual account is:

- stay in Qualtrics
- generate the surveys locally from source
- import the generated survey files through the documented Qualtrics import workflow
- keep the remaining Qualtrics-side actions to the absolute minimum

Why this is the right choice:

- You explicitly do **not** want to host a custom website yourself.
- Your study design is **1 distinct case per HCP**, so separate surveys reduce routing risk and keep assignment clean.
- Qualtrics officially supports importing **Advanced TXT / DOCX** surveys and is already the natural target for this project.
- Building one giant branched survey would add avoidable logic risk for a sample of only 10 HCPs.

Official references used for this choice:

- Qualtrics import / export surveys:
  - [Import & Export Surveys](https://www.qualtrics.com/support/survey-platform/survey-module/survey-tools/import-and-export-surveys/)
- Qualtrics query strings / embedded data:
  - [Passing Information via Query Strings](https://www.qualtrics.com/support/survey-platform/survey-module/survey-flow/standard-elements/passing-information-through-query-strings/)
  - [Embedded Data](https://www.qualtrics.com/support/survey-platform/survey-module/survey-flow/standard-elements/embedded-data/)
- Qualtrics graphics in survey content:
  - [Insert a Graphic](https://www.qualtrics.com/support/survey-platform/survey-module/editing-questions/rich-content-editor/insert-a-graphic/)
- Qualtrics form-field question type:
  - [Form Field Question](https://www.qualtrics.com/support/survey-platform/survey-module/editing-questions/question-types-guide/standard-content/form-field-question/)
- Qualtrics rank order question type:
  - [Rank Order Question](https://www.qualtrics.com/support/survey-platform/survey-module/editing-questions/question-types-guide/standard-content/rank-order/)

## Why this is not a hand-written QSF or API-only deployment

I explicitly did **not** choose a hand-authored `QSF` generator as the production path.

Reason:

- Qualtrics officially supports **importing** QSF files, but their own support page frames QSF as an exported survey-transfer format and warns not to edit the file contents manually.
- By contrast, Qualtrics explicitly documents **TXT / DOCX** import as the supported authoring path for building surveys from formatted text files.
- An API-only deployment would tie this package to account-specific credentials, API entitlements, and live-brand testing. That is a weaker archival and handover format for a master's-thesis workflow than deterministic, source-controlled import artifacts.

So the chosen implementation is:

- generate Qualtrics-ready survey imports directly from the authoritative source files
- keep the package fully local, reproducible, and account-independent
- avoid relying on undocumented QSF internals

## What this package now does

The generator reads the current source-of-truth files in:

- `/Users/stijnvanseveren/PythonProjects/MASTERPROEF/evaluation/qualtrics/01_HCPs_PRE/separate_HCPs/1_case_per_HCP`

It then automatically builds:

- 10 Qualtrics Advanced TXT import files
- 11 reconstructed Part 3 network figures
- a zipped upload bundle
- manifests for case assignment and emailing
- a source coverage report

The generated surveys now include:

- title / study context
- explicit **informed consent**
- the requested **intake block**
- all shared instructions
- all examples
- all case-specific content
- Part 3 example + case-specific figures
- Parts 1 through 6
- closing / return information

## Important change from the earlier package

The Part 3 figures are now embedded **inline by default** as base64 HTML images inside the generated Qualtrics import files.

That means:

- you do **not** need to host the figures externally
- you do **not** need to manually insert the Part 3 images after import in the normal workflow

## Output locations

Main generator:

- [generate_qualtrics_pre_package.py](/Users/stijnvanseveren/PythonProjects/MASTERPROEF/evaluation/qualtrics/01_HCPs_PRE/separate_HCPs/survey_AUTOMATED/generate_qualtrics_pre_package.py)

Generated import files:

- `/Users/stijnvanseveren/PythonProjects/MASTERPROEF/evaluation/qualtrics/01_HCPs_PRE/separate_HCPs/survey_AUTOMATED/generated/qualtrics_advanced_txt`

Generated manifests:

- [hcp_case_manifest.csv](/Users/stijnvanseveren/PythonProjects/MASTERPROEF/evaluation/qualtrics/01_HCPs_PRE/separate_HCPs/survey_AUTOMATED/generated/manifests/hcp_case_manifest.csv)
- [email_send_sheet.csv](/Users/stijnvanseveren/PythonProjects/MASTERPROEF/evaluation/qualtrics/01_HCPs_PRE/separate_HCPs/survey_AUTOMATED/generated/manifests/email_send_sheet.csv)
- [image_manifest.csv](/Users/stijnvanseveren/PythonProjects/MASTERPROEF/evaluation/qualtrics/01_HCPs_PRE/separate_HCPs/survey_AUTOMATED/generated/manifests/image_manifest.csv)
- [source_coverage_report.md](/Users/stijnvanseveren/PythonProjects/MASTERPROEF/evaluation/qualtrics/01_HCPs_PRE/separate_HCPs/survey_AUTOMATED/generated/manifests/source_coverage_report.md)

Upload bundle:

- [qualtrics_upload_bundle.zip](/Users/stijnvanseveren/PythonProjects/MASTERPROEF/evaluation/qualtrics/01_HCPs_PRE/separate_HCPs/survey_AUTOMATED/generated/qualtrics_upload_bundle.zip)

Figures:

- `/Users/stijnvanseveren/PythonProjects/MASTERPROEF/evaluation/qualtrics/01_HCPs_PRE/separate_HCPs/survey_AUTOMATED/assets/networks`

## Intake block

The intake was designed around the three domains you requested:

1. clinical background / practical experience
2. stance toward agentic AI
3. familiarity / stance toward network analysis

Current implementation:

- 1 open background question
- 3 Likert items on agentic AI
- 3 Likert items on network analysis

Note:

- Qualtrics Advanced TXT officially supports `MC`, `Matrix`, `TE`, `CS`, `RO`, and `DB` in this import format, but **not native slider questions** in the documented Advanced TXT import syntax.
- Because of that platform constraint, the two 3-item intake attitude sets are generated as **7-point Likert matrix questions**.
- This is the best import-safe path without switching away from Qualtrics or manually rebuilding question types in the editor.

## Chosen workflow

This is the workflow to use.

### 1. Regenerate the package

```bash
python3 /Users/stijnvanseveren/PythonProjects/MASTERPROEF/evaluation/qualtrics/01_HCPs_PRE/separate_HCPs/survey_AUTOMATED/generate_qualtrics_pre_package.py
```

Default behavior:

- figures are embedded inline
- all 10 one-case survey files are rebuilt
- the bundle zip is rebuilt

### 2. Import the 10 surveys into Qualtrics

For each generated `.txt` file in:

- `/Users/stijnvanseveren/PythonProjects/MASTERPROEF/evaluation/qualtrics/01_HCPs_PRE/separate_HCPs/survey_AUTOMATED/generated/qualtrics_advanced_txt`

import it into Qualtrics via:

- `Survey tab -> Tools -> Import/Export -> Import survey`

This is still the only unavoidable repetitive step left on the Qualtrics side.

### 3. Perform the 2 remaining Qualtrics-side checks

Because these are not reliably encoded by the documented Advanced TXT import format, you still need to check:

1. **Deel 4 validation**:
   Set the Part 4 multiple-answer question so that respondents must select **exactly 6** options.
2. **Consent behavior**:
   Verify the informed-consent question appears correctly at the start.

## What is already automated vs. what is not

### Automated

- source parsing from the authoritative LaTeX files
- case assignment mapping
- title / context / instructions
- informed consent page content
- intake block content
- examples and shared instructions
- case-specific vignettes and context
- Part 3 figures
- Part 1, 2, 3, 4, 5, 6 structure
- closing text
- import files
- upload bundle
- manifests

### Not fully automatable in the current documented Qualtrics TXT import format

- forcing **exactly 6** selections in Part 4 directly at import time
- native slider rendering for the two intake attitude question sets
- full consent branching logic directly from the TXT import alone

These are platform/import-format limitations, not source-extraction limitations.

## Image modes

Default mode is correct for your use case:

```bash
python3 /Users/stijnvanseveren/PythonProjects/MASTERPROEF/evaluation/qualtrics/01_HCPs_PRE/separate_HCPs/survey_AUTOMATED/generate_qualtrics_pre_package.py --image-mode inline
```

Other supported modes exist only for fallback/debugging:

- `--image-mode base-url --image-base-url "https://..."`
- `--image-mode placeholder`

For your thesis workflow, use `inline`.

## Practical conclusion

Use this package as the production path:

- stay with **Qualtrics**
- keep **10 separate one-case surveys**
- import the generated files
- send one link per HCP yourself

This gives you the highest-fidelity, no-self-hosting workflow with the lowest operational risk for your study design.

With your current non-API account, this is also the **most automated supported method that remains realistically robust**.
