#!/usr/bin/env python3
"""
make_pirai_qsf.py
Generates a stripped-down QSF for upload to Pirai AI (<300 KB).
Full survey structure (all 241 questions, complete flow/branches) but:
  - No images
  - No QuestionText_Unsafe (saves 355 KB duplicate)
  - HTML stripped to plain text
  - DB (instructional) pages truncated to 400 chars — Pirai only needs the label
  - Interactive questions (MC, Matrix, TE, RO) keep full text
  - Redundant fields removed
"""
import json, re
from pathlib import Path

SRC  = Path(__file__).parent / "generated" / "qsf_files" / "PHOENIX_PRE_MERGED_ALL10.qsf"
DEST = Path(__file__).parent / "generated" / "qsf_files" / "PHOENIX_PRE_MERGED_PIRAI.qsf"

def strip_html(html: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = (text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<")
                .replace("&gt;", ">").replace("&quot;", '"').replace("&#8212;", "\u2014")
                .replace("&#8211;", "\u2013").replace("&#9744;", "\u2610"))
    return re.sub(r"\n{3,}", "\n\n", text.strip())

DROP_KEYS = {"QuestionText_Unsafe", "GradingData", "Language", "DataVisibility",
             "DefaultChoices", "NextChoiceId", "NextAnswerId", "Configuration"}

qsf = json.loads(SRC.read_text(encoding="utf-8"))

for el in qsf["SurveyElements"]:
    if el["Element"] != "SQ":
        continue
    pl = el["Payload"]
    q_type = pl.get("QuestionType", pl.get("Type", ""))
    qt_plain = strip_html(pl.get("QuestionText", ""))

    if q_type == "DB":
        # Instructional pages: truncate — Pirai only needs to know what section this is
        pl["QuestionText"] = qt_plain[:200] + ("..." if len(qt_plain) > 200 else "")
    else:
        # Interactive questions: keep full plain text so Pirai AI can process them
        pl["QuestionText"] = qt_plain

    for k in DROP_KEYS:
        pl.pop(k, None)
    if "QuestionDescription" in pl:
        pl["QuestionDescription"] = pl["QuestionDescription"][:120]

DEST.write_text(json.dumps(qsf, ensure_ascii=True), encoding="utf-8")

sz = DEST.stat().st_size
q2 = json.loads(DEST.read_text())
n_q = sum(1 for e in q2["SurveyElements"] if e["Element"] == "SQ")
n_b = len(q2["SurveyElements"][0]["Payload"])
print(f"Written: {DEST.name}")
print(f"Size:    {sz/1024:.0f} KB  ({sz:,} bytes)")
print(f"Blocks:  {n_b}  |  Questions: {n_q}")
