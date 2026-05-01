#!/usr/bin/env python3
"""
Split each 2-case HCP document into two 1-case documents.
Source: 2_cases_per_HCP/HCP_N/main.tex  (N = 1..5)
Output: 1_case_per_HCP/HCP_NN/main.tex  (NN = 01..10)
"""

import re
import os

BASE = os.path.dirname(__file__)
PARENT = os.path.dirname(BASE)
OUT_DIR = os.path.join(PARENT, "1_case_per_HCP")

# Case metadata: (source_hcp_N, new_hcp_A, new_hcp_B, caseA_code, caseB_code, caseA_desc, caseB_desc)
SPLITS = [
    (1, "01", "02", "C01", "C02",
     "28-jarige softwareontwikkelaar",
     "34-jarige administratief medewerker in de zorg"),
    (2, "03", "04", "C03", "C04",
     "41-jarige leraar secundair onderwijs",
     "63-jarige gepensioneerde professional"),
    (3, "05", "06", "C05", "C06",
     "25-jarige postgraduaatsstudent",
     "32-jarige marketingprofessional"),
    (4, "07", "08", "C07", "C08",
     "46-jarige accountmanager",
     "38-jarige projectmanager"),
    (5, "09", "10", "C09", "C10",
     "27-jarige salesvertegenwoordiger",
     "52-jarige operationeel directeur"),
]

# Time table for single-case version
SINGLE_CASE_TABLE = r"""\toprule
Stap & Klinische taak & Wat u doet & Richttijd \\
\midrule
-- & Instructies lezen & Lees de instructiepagina aandachtig door & $\approx$\,5 min \\
1 & Operationalisering & Identificeer 2--6 \textbf{symptoomlabels} (klachtdimensies) & $\approx$\,3 min \\
2 & Initieel observatiemodel & Genereer 3--5 \textbf{behandelingsoptielabels} (modificeerbaar, EMA-geschikt) & $\approx$\,3 min \\
3 & Behandeldoelprioritering & Rangschik alle 5 \textbf{behandelingsopties} van hoog naar laag behandelprioriteit & $\approx$\,4 min \\
4 & Verfijnd observatiemodel & Selecteer exact 6 \textbf{EMA-items} (2 per behandeldoel) uit de lijst van 20 & $\approx$\,4 min \\
5 & Mobiele coaching & Schrijf een korte patientgerichte boodschap voor de app & $\approx$\,4 min \\
\midrule
 & \textbf{Totaal (1 casus)} & & \textbf{20--25 min} \\
\bottomrule"""

TWO_CASE_TABLE = r"""\toprule
Stap & Klinische taak & Wat u doet & Richttijd \\
\midrule
-- & Instructies lezen & Lees de instructiepagina aandachtig door & $\approx$\,5 min \\
1 & Operationalisering & Identificeer 2--6 \textbf{symptoomlabels} (klachtdimensies) & $\approx$\,6 min \\
2 & Initieel observatiemodel & Genereer 3--5 \textbf{behandelingsoptielabels} (modificeerbaar, EMA-geschikt) & $\approx$\,6 min \\
3 & Behandeldoelprioritering & Rangschik alle 5 \textbf{behandelingsopties} van hoog naar laag behandelprioriteit & $\approx$\,7 min \\
4 & Verfijnd observatiemodel & Selecteer exact 6 \textbf{EMA-items} (2 per behandeldoel) uit de lijst van 20 & $\approx$\,8 min \\
5 & Mobiele coaching & Schrijf een korte patientgerichte boodschap voor de app & $\approx$\,8 min \\
\midrule
 & \textbf{Totaal (2 casussen)} & & \textbf{40--50 min} \\
\bottomrule"""


def remove_casus2_blocks(text: str) -> str:
    """Remove all 'Casus 2' subsections from each Deel, leaving Casus 1 intact."""
    # Match the separator newpage + optional blank line + subsection header + everything through
    # the closing \vspace + newpage of that Casus 2 block.
    # We use a non-greedy match with DOTALL.
    # Pattern covers both variants:
    #   (a) \newpage\n\n\subsection*{Casus 2:   (Deel 1 variant with blank line)
    #   (b) \newpage\n\subsection*{Casus 2:      (Deel 2-5 variant, no extra blank)
    text = re.sub(
        r'\\newpage\n\n?\\subsection\*\{Casus 2:.*?\\vspace\{0\.6em\}\n\n?\\newpage',
        r'\\newpage',
        text,
        flags=re.DOTALL
    )
    return text


def remove_casus1_blocks_and_rename(text: str) -> str:
    """
    For Casus 2 output files:
    - Remove all 'Casus 1' subsections
    - Rename 'Casus 2' → 'Casus 1' in subsection headers and responsebox opdrachten
    """
    # Each Deel has this structure (after example box):
    # \vspace{0.5em}\n\n\newpage\n\n\subsection*{Casus 1:...}
    # ...casus 1 content...
    # \vspace{0.6em}\n\newpage\n\subsection*{Casus 2:...}
    # ...casus 2 content...
    # \vspace{0.6em}\n\newpage\n\n\n%
    #
    # We want to:
    # 1. Remove from "\n\n\subsection*{Casus 1:" to just before "\n\newpage\n\subsection*{Casus 2:"
    #    (keeping the leading "\vspace{0.5em}\n\n\newpage")
    # 2. Then rename "Casus 2:" → "Casus 1:" in subsection headers

    # Step 1: Remove Casus 1 blocks.
    # The Casus 1 block starts with a blank line + \subsection*{Casus 1:
    # and ends just before the \newpage that precedes \subsection*{Casus 2:
    #
    # Pattern: after \vspace{0.5em}\n\n\newpage (which we keep),
    # we have \n\n?\subsection*{Casus 1:...[content]...\vspace{0.6em}
    # then \newpage (which belongs to separator), then \n?\subsection*{Casus 2:
    #
    # We remove: \n(blank?)\subsection*{Casus 1:...content...\vspace{0.6em}\n\newpage\n
    # so that \subsection*{Casus 2: follows immediately after \newpage\n\n\newpage or similar.
    #
    # Let me try a different approach: replace the casus1_section + separator_newpage
    # with just nothing, making the outer \newpage connect directly to \subsection*{Casus 2:

    # Remove Casus 1 block + the joining newpage before Casus 2
    text = re.sub(
        r'\n\n?\\subsection\*\{Casus 1:.*?\\vspace\{0\.6em\}\n\n?\\newpage\n',
        r'\n',
        text,
        flags=re.DOTALL
    )

    # Step 2: Rename "Casus 2" → "Casus 1" in subsection headers
    text = re.sub(
        r'\\subsection\*\{Casus 2:',
        r'\\subsection*{Casus 1:',
        text
    )

    # Also rename in complaintbox titles, contextbox references
    # e.g., title={Casus 2: verkorte klachtomschrijving}
    text = re.sub(r'(title=\{)Casus 2:', r'\1Casus 1:', text)

    return text


def adapt_for_single_case(text: str, new_hcp_code: str, case_code: str,
                           case_desc: str, casus_num: int,
                           source_hcp_code_padded: str) -> str:
    """Apply all metadata adaptations for a single-case document."""

    new_hcp_full = f"HCP-PRE-{new_hcp_code}"

    # 1. pdftitle
    text = re.sub(
        r'pdftitle=\{PHOENIX evaluatiestudie - HCP-PRE-\d+ \(.*?\)\}',
        f'pdftitle={{PHOENIX evaluatiestudie - {new_hcp_full} ({case_code})}}',
        text
    )

    # 2. fancyhead right: HCP code + case codes (use line-based match to handle nested braces)
    text = re.sub(
        r'\\fancyhead\[R\]\{[^\n]+\n',
        rf'\\fancyhead[R]{{\\small\\color{{SlateMid}}\\textbf{{{new_hcp_full}}}\\quad {case_code}}}\n',
        text
    )

    # 3. Deelnemerscode box on title page
    text = re.sub(
        r'\{\\large\\bfseries\\color\{DarkBlue\}Deelnemerscode: \\texttt\{HCP-PRE-\d+\}\}',
        rf'{{\\large\\bfseries\\color{{DarkBlue}}Deelnemerscode: \\texttt{{{new_hcp_full}}}}}',
        text
    )

    # 4. Case assignment box: replace multi-case block with single case
    # Original: Toegewezen casussen:\quad\n\textbf{...}\n\quad en \quad\n\textbf{...}\n}
    text = re.sub(
        r'Toegewezen casussen:\\quad\n\\textbf\{[^}]+\}[^\n]+\n\\quad en \\quad\n\\textbf\{[^}]+\}[^\n]+\n\}',
        rf'Toegewezen casus:\\quad\n\\textbf{{\\color{{PrimaryBlue}}{case_code}}} ({case_desc})\n}}',
        text
    )

    # 5. Geschatte duur
    text = text.replace(
        "Geschatte duur & Ongeveer 40--50 minuten voor beide casussen samen (inclusief leestijd instructies) \\\\",
        "Geschatte duur & Ongeveer 20--25 minuten voor de casus (inclusief leestijd instructies) \\\\"
    )

    # 6. Doel box: "Voor elk van uw twee casussen" → "Voor uw casus"
    text = text.replace(
        "Voor elk van uw twee casussen vult u dezelfde vijf delen in:",
        "Voor uw casus vult u dezelfde vijf delen in:"
    )

    # 7. Werkwijze: "voor beide casussen" → remove that phrase
    text = text.replace(
        "Ga pas naar een volgend deel wanneer het huidige deel volledig is afgewerkt voor beide casussen.",
        "Ga pas naar een volgend deel wanneer het huidige deel volledig is afgewerkt."
    )

    # 8. Time table: replace 2-case table with single-case table
    text = text.replace(TWO_CASE_TABLE, SINGLE_CASE_TABLE)

    # 9. Afronding: "uw twee toegewezen casussen (Casus 1 en Casus 2)" → "uw toegewezen casus"
    text = text.replace(
        "Dank u voor het invullen van alle vijf delen voor uw twee toegewezen casussen\n(Casus 1 en Casus 2).",
        "Dank u voor het invullen van alle vijf delen voor uw toegewezen casus."
    )

    # 10. Checklist: "voor beide casussen" → just "Ik heb in Deel N ..."
    text = re.sub(r' voor beide casussen', '', text)

    # 11. Email subject
    text = re.sub(
        r'\\texttt\{PHOENIX-PRE-HCP-PRE-\d+\}',
        rf'\\texttt{{PHOENIX-PRE-{new_hcp_full}}}',
        text
    )

    return text


def create_single_case_file(source_path: str, output_dir: str,
                             new_hcp_code: str, case_code: str,
                             case_desc: str, casus_to_keep: int,
                             source_hcp_padded: str) -> None:
    with open(source_path, encoding="utf-8") as f:
        text = f.read()

    if casus_to_keep == 1:
        text = remove_casus2_blocks(text)
    else:
        text = remove_casus1_blocks_and_rename(text)

    text = adapt_for_single_case(
        text, new_hcp_code, case_code, case_desc, casus_to_keep, source_hcp_padded
    )

    out_path = os.path.join(output_dir, f"HCP_{new_hcp_code}", "main.tex")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  ✓  Created: {out_path}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for (src_n, hcp_a, hcp_b, case_a, case_b, desc_a, desc_b) in SPLITS:
        src_path = os.path.join(BASE, f"HCP_{src_n}", "main.tex")
        src_padded = f"{src_n:02d}"
        print(f"\nSplitting HCP_{src_n} → HCP_{hcp_a} ({case_a}) + HCP_{hcp_b} ({case_b})")
        create_single_case_file(src_path, OUT_DIR, hcp_a, case_a, desc_a, 1, src_padded)
        create_single_case_file(src_path, OUT_DIR, hcp_b, case_b, desc_b, 2, src_padded)
    print("\nAll single-case docs created.")
