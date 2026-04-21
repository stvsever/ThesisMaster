#!/usr/bin/env python3
"""
Apply v2 changes to all 5 two-case HCP LaTeX files:
1. Voorbeeldbox: change title to "Voorbeeld --- #2", add toptitle/bottomtitle
2. Deel 4 goals: remove "(domein: ...)" domain annotations
3. "Na 21 dagen EMA-monitoring" → "Na voldoende EMA-monitoring"
"""

import re
import os
import subprocess

BASE = os.path.dirname(__file__)
FILES = [os.path.join(BASE, f"HCP_{i}", "main.tex") for i in range(1, 6)]

OLD_VBOX = """\
\\newtcolorbox{voorbeeldbox}[3][]{
  colback=AccentPurple,colframe=RichPurple,
  boxrule=0.8pt,arc=2.5mm,breakable,
  left=10pt,right=10pt,top=9pt,bottom=9pt,
  fonttitle=\\small\\bfseries\\color{RichPurple},
  halign title=center,
  title={Uitgewerkt voorbeeld -- #2 -- Nadia (38-jarige verpleegkundige)\\\\[1pt]%
    {\\footnotesize\\itshape\\color{white}#3}},#1
}"""

NEW_VBOX = """\
\\newtcolorbox{voorbeeldbox}[3][]{
  colback=AccentPurple,colframe=RichPurple,
  boxrule=0.8pt,arc=2.5mm,breakable,
  left=10pt,right=10pt,top=9pt,bottom=9pt,
  toptitle=3pt,bottomtitle=3pt,
  fonttitle=\\small\\bfseries\\color{RichPurple},
  halign title=center,
  title={Voorbeeld --- #2\\\\[1pt]%
    {\\footnotesize\\itshape\\color{white}#3}},#1
}"""


def apply_changes(text: str) -> str:
    # 1. Voorbeeldbox: fix title format and add top/bottom title spacing
    text = text.replace(OLD_VBOX, NEW_VBOX)

    # 2. Remove domain annotations from Deel 4 goals
    # Matches: \quad\textit{\small (domein: ...some text...)}
    text = re.sub(
        r'\\quad\\textit\{\\small \(domein:[^}]*\)\}',
        '',
        text
    )

    # 3. "Na 21 dagen EMA-monitoring" → "Na voldoende EMA-monitoring"
    text = text.replace(
        'Na 21 dagen EMA-monitoring construeert PHOENIX',
        'Na voldoende EMA-monitoring construeert PHOENIX'
    )

    return text


if __name__ == '__main__':
    for path in FILES:
        with open(path, encoding='utf-8') as f:
            text = f.read()
        new_text = apply_changes(text)
        if new_text == text:
            print(f'  ⚠  No changes in: {path}')
        else:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(new_text)
            print(f'  ✓  Updated: {path}')
    print('\nDone.')
