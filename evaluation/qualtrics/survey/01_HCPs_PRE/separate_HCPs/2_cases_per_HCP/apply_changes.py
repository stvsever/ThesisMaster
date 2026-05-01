#!/usr/bin/env python3
"""Apply all pending textual changes to HCP_1–5 main.tex files."""

import re
import os

BASE = os.path.dirname(__file__)
HCP_DIRS = [os.path.join(BASE, f"HCP_{i}") for i in range(1, 6)]

# ── helper ────────────────────────────────────────────────────────────────────
def apply_changes(path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    original = text  # keep for diff check

    # 1. Fix \\[3pt] → \\[1pt] in voorbeeldbox macro (HCP_1 already done, harmless to re-apply)
    text = text.replace(
        r"\\[3pt]%" + "\n" + r"    {\footnotesize\itshape\color{white}#3}},#1",
        r"\\[1pt]%" + "\n" + r"    {\footnotesize\itshape\color{white}#3}},#1",
    )

    # 2. Remove "(PHOENIX)" from study name in info table
    text = text.replace(
        "geestelijke gezondheidszorg (PHOENIX) \\\\",
        "geestelijke gezondheidszorg \\\\",
    )

    # 3. Change "als het PHOENIX-systeem" → "als een geautomatiseerd multi-agentsysteem"
    text = text.replace(
        "vijf klinische redeneerstappen uitvoeren als het PHOENIX-systeem. Uw antwoorden",
        "vijf klinische redeneerstappen uitvoeren als een geautomatiseerd multi-agentsysteem. Uw antwoorden",
    )

    # 4. Remove "Geen behandelingsoptie, geen symptoom" bullet (two-line block)
    text = re.sub(
        r"\\item \\textbf\{Geen behandelingsoptie, geen symptoom:\}[^\n]+\n[^\n]+\n",
        "",
        text,
    )

    # 5. Add "vaak" to EMA description
    text = text.replace(
        r"\textbf{EMA} is een methode waarbij personen meerdere keren per dag hun actuele",
        r"\textbf{EMA} is een methode waarbij personen vaak meerdere keren per dag hun actuele",
    )

    # 6. Add "bijv.\ " to "Dagelijks rapporteerbaar" bullet
    text = text.replace(
        r"\item \textbf{Dagelijks rapporteerbaar:} via een korte vraag op de smartphone (ja/nee, aantal, minuten of 0--10).",
        r"\item \textbf{Dagelijks rapporteerbaar:} via een korte vraag op de smartphone, bijv.\ (ja/nee, aantal, minuten of 0--10).",
    )

    # 7. Add Werkwijze item 6 (before \end{enumerate} inside instrbox Werkwijze)
    OLD_WERKWIJZE_END = (
        r"\item \textbf{Noteer of typ rechtstreeks in de voorziene antwoordzones.}" + "\n"
        r"Onleesbare of ambigu geformuleerde antwoorden bemoeilijken latere blinde beoordeling." + "\n"
        r"\end{enumerate}" + "\n"
        r"\end{instrbox}"
    )
    NEW_WERKWIJZE_END = (
        r"\item \textbf{Noteer of typ rechtstreeks in de voorziene antwoordzones.}" + "\n"
        r"Onleesbare of ambigu geformuleerde antwoorden bemoeilijken latere blinde beoordeling." + "\n"
        r"\item \textbf{Werk ook bij beperkte informatie.} Klinische teksten bevatten vaak onvoldoende informatie voor definitieve conclusies. Geef \emph{altijd} een antwoord op basis van de beschikbare informatie en uw klinisch oordeel. Laat geen velden blanco zonder reden." + "\n"
        r"\end{enumerate}" + "\n"
        r"\end{instrbox}"
    )
    text = text.replace(OLD_WERKWIJZE_END, NEW_WERKWIJZE_END)

    # 8a. Change S-2 label in Deel 1 example: "Vroeg ontwaken" → "Vroegochtendontwaak"
    text = text.replace(
        r"\item \textbf{S-2}\enspace Vroeg ontwaken" + "\n"
        r"  \hfill\textit{(``word vroeg wakker'' $\rightarrow$ onderscheidbaar van S-1)}",
        r"\item \textbf{S-2}\enspace Vroegochtendontwaak" + "\n"
        r"  \hfill\textit{(``word vroeg wakker'' $\rightarrow$ DSM-5-TR: early morning awakening; onderscheidbaar van S-1)}",
    )

    # 8b. Update S-2 in Deel 2 example reference list
    text = text.replace(
        r"S-1: Inslaapmoeilijkheden;\enspace S-2: Vroeg ontwaken;\enspace",
        r"S-1: Inslaapmoeilijkheden;\enspace S-2: Vroegochtendontwaak;\enspace",
    )

    # 9. Update time estimate table: add instruction reading row, update total
    OLD_TABLE = (
        r"\toprule" + "\n"
        r"Stap & Klinische taak & Wat u doet & Richttijd \\" + "\n"
        r"\midrule" + "\n"
        r"1 & Operationalisering & Identificeer 2--6 \textbf{symptoomlabels} (klachtdimensies) & $\approx$\,6 min \\" + "\n"
        r"2 & Initieel observatiemodel & Genereer 3--5 \textbf{behandelingsoptielabels} (modificeerbaar, EMA-geschikt) & $\approx$\,6 min \\" + "\n"
        r"3 & Behandeldoelprioritering & Rangschik alle 5 \textbf{behandelingsopties} van hoog naar laag behandelprioriteit & $\approx$\,7 min \\" + "\n"
        r"4 & Verfijnd observatiemodel & Selecteer exact 6 \textbf{EMA-items} (2 per behandeldoel) uit de lijst van 20 & $\approx$\,8 min \\" + "\n"
        r"5 & Mobiele coaching & Schrijf een korte patientgerichte boodschap voor de app & $\approx$\,8 min \\" + "\n"
        r"\midrule" + "\n"
        r" & \textbf{Totaal (2 casussen)} & & \textbf{35--45 min} \\" + "\n"
        r"\bottomrule"
    )
    NEW_TABLE = (
        r"\toprule" + "\n"
        r"Stap & Klinische taak & Wat u doet & Richttijd \\" + "\n"
        r"\midrule" + "\n"
        r"-- & Instructies lezen & Lees de instructiepagina aandachtig door & $\approx$\,5 min \\" + "\n"
        r"1 & Operationalisering & Identificeer 2--6 \textbf{symptoomlabels} (klachtdimensies) & $\approx$\,6 min \\" + "\n"
        r"2 & Initieel observatiemodel & Genereer 3--5 \textbf{behandelingsoptielabels} (modificeerbaar, EMA-geschikt) & $\approx$\,6 min \\" + "\n"
        r"3 & Behandeldoelprioritering & Rangschik alle 5 \textbf{behandelingsopties} van hoog naar laag behandelprioriteit & $\approx$\,7 min \\" + "\n"
        r"4 & Verfijnd observatiemodel & Selecteer exact 6 \textbf{EMA-items} (2 per behandeldoel) uit de lijst van 20 & $\approx$\,8 min \\" + "\n"
        r"5 & Mobiele coaching & Schrijf een korte patientgerichte boodschap voor de app & $\approx$\,8 min \\" + "\n"
        r"\midrule" + "\n"
        r" & \textbf{Totaal (2 casussen)} & & \textbf{40--50 min} \\" + "\n"
        r"\bottomrule"
    )
    text = text.replace(OLD_TABLE, NEW_TABLE)

    # 10. Update "Geschatte duur" on title page
    text = text.replace(
        "Geschatte duur & Ongeveer 35--45 minuten voor beide casussen samen \\\\",
        "Geschatte duur & Ongeveer 40--50 minuten voor beide casussen samen (inclusief leestijd instructies) \\\\",
    )

    # 11. Add Part 4 checkbox note to responsebox instructions in Deel 4
    # Find the exact line in both casus 1 and casus 2 Deel 4 responseboxes
    OLD_D4_INSTR = (
        r"\small\textbf{Opdracht:} selecteer \textbf{exact 6} EMA-items (2 per behandeldoel) die het best aansluiten als sub-behandelingsopties." + "\n"
        r"\textit{Alle 20 items zijn dagelijkse mobiele EMA-items (behandelingsoptie-type, geen symptomen).}"
    )
    NEW_D4_INSTR = (
        r"\small\textbf{Opdracht:} selecteer \textbf{exact 6} EMA-items (2 per behandeldoel) die het best aansluiten als sub-behandelingsopties." + "\n"
        r"\textit{Alle 20 items zijn dagelijkse mobiele EMA-items (behandelingsoptie-type, geen symptomen).}" + "\n"
        r"\textit{In het Word-document kunt u de vakjes ($\square$) omcirkelen of markeren in plaats van aanvinken.}"
    )
    text = text.replace(OLD_D4_INSTR, NEW_D4_INSTR)

    if text != original:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  ✓  Updated: {path}")
    else:
        print(f"  ⚠  No changes applied: {path}")


if __name__ == "__main__":
    for hcp_dir in HCP_DIRS:
        tex_path = os.path.join(hcp_dir, "main.tex")
        if os.path.exists(tex_path):
            print(f"\nProcessing {tex_path} …")
            apply_changes(tex_path)
        else:
            print(f"  ✗  Not found: {tex_path}")
    print("\nDone.")
