# Qualtrics Compatibility Notes

This note captures the current official-platform constraints that matter for this automation package.

## Official findings

- Qualtrics officially supports importing surveys from `QSF`, `TXT`, and `DOCX`.
- Advanced TXT officially supports these question families:
  - `MC`
  - `Matrix`
  - `TE`
  - `CS`
  - `RO`
  - `DB`
- Qualtrics explicitly documents that HTML coding can be applied to TXT documents, but custom coding is provided as-is.
- Qualtrics documents a Survey API that can create surveys, blocks, questions, flow elements, and upload library graphics.
- Qualtrics Public Workspace also states that API access is a paid feature tied to the license.

## What this means for this package

- The generated survey files use only documented Advanced TXT question families.
- The generated files therefore align with the officially documented import syntax for:
  - informed consent single-choice item
  - descriptive instruction blocks
  - essay text entry
  - form text entry
  - matrix Likert questions
  - multiple-answer checkbox questions
  - rank-order questions

## What Advanced TXT import does not fully solve by itself

- Survey Flow / branch logic / embedded data routing are not described in the Advanced TXT import syntax itself.
- Exact answer-count constraints such as "select exactly 6" are not documented in the TXT import syntax.
- Slider questions are not listed among the officially documented Advanced TXT-compatible question families.

## Graphics / HTML risk

- The current package embeds Part 3 figures via HTML inside descriptive text blocks.
- Qualtrics officially allows HTML in TXT imports, but custom HTML support is not support-assisted and some HTML behavior depends on account permissions.
- On free accounts, Qualtrics explicitly documents that API access, HTML markup, and image insertion are not available.

## Practical conclusion

- If your university Qualtrics brand has normal paid Research Core access plus the necessary HTML permissions, this package is the most automated no-self-hosting route currently available from documented import syntax.
- If your brand blocks the required HTML behavior, the survey text and response formats remain importable, but Part 3 figures will need a Qualtrics-side insertion workflow instead of pure import-only deployment.
- If the API is disabled for your account, the best supported deployment method is the documented Qualtrics import workflow rather than API construction.
