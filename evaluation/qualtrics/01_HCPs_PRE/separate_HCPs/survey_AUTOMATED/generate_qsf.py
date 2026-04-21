#!/usr/bin/env python3
"""
generate_qsf.py
Generates valid Qualtrics Survey Format (.qsf) files for all 10 PHOENIX PRE HCP surveys.
Each .qsf can be imported directly into Qualtrics via Tools → Import/Export → Import survey.

Survey structure per HCP (separate pages for each section):
  Title page → Consent → Instructions p.1 → Instructions p.2 → Intake (3 pages) →
  Deel 1 (instr | example | case+response) →
  Deel 2 (instr | example | case+response) →
  Deel 3 (instr | example | case+network+ranking) →
  Deel 4 (instr | example | targets+items) →
  Deel 5 (instr | example | context+essay) →
  Deel 6 (context | question-a | question-b | question-c) →
  Afronding

Run from survey_AUTOMATED/:
    python3 generate_qsf.py

Output: generated/qsf_files/PHOENIX_PRE_<HCP_CODE>_<CASE_CODE>.qsf
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT.parent / "1_case_per_HCP"
QSF_ROOT = ROOT / "generated" / "qsf_files"
HCP_DIRS = [SOURCE_ROOT / f"HCP_{i:02d}" for i in range(1, 11)]

SCALE_7PT = [
    "1 = Helemaal oneens",
    "2 = Oneens",
    "3 = Eerder oneens",
    "4 = Neutraal",
    "5 = Eerder eens",
    "6 = Eens",
    "7 = Helemaal eens",
]

# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Node:
    key: str; prefix: str; label: str; x: float; y: float; kind: str; compact: bool

@dataclass
class Edge:
    src: str; dst: str; width: float; sign: str

@dataclass
class NetworkFigure:
    title: str; left_header: str; right_header: str
    nodes: list[Node]; edges: list[Edge]

@dataclass
class CaseSurvey:
    hcp_code: str; case_code: str; profile: str; duration: str
    complaint_vignette: str; short_summary: str
    part2_symptoms: list[str]; monitoring: str
    part3_options: list[str]; part4_targets: list[str]
    part4_items: list[str]; part5_primary_problem: str
    part5_target: str; part5_barrier: str; part5_coping: str
    network: NetworkFigure; example_network: NetworkFigure

# ─────────────────────────────────────────────────────────────────────────────
# LATEX PARSING
# ─────────────────────────────────────────────────────────────────────────────

def latex_to_text(text: str) -> str:
    subs = [
        (r"(?m)^\s*%.*$", ""),
        (r"\\section\{([^}]*)\}", r"\n\1\n"),
        (r"\\subsection\*\{([^}]*)\}", r"\n\1\n"),
        (r"\\begin\{voorbeeldbox\}\{([^}]*)\}\{([^}]*)\}", r"\n\1\n\2\n"),
        (r"\\begin\{(?:tcolorbox|instrbox|responsebox|complaintbox|contextbox|monitorbox)\}(?:\[[^\]]*\])?", "\n"),
        (r"\\end\{(?:tcolorbox|instrbox|responsebox|complaintbox|contextbox|monitorbox)\}", "\n"),
        (r"\\begin\{itemize\}(?:\[[^\]]*\])?", "\n"), (r"\\end\{itemize\}", "\n"),
        (r"\\begin\{enumerate\}(?:\[[^\]]*\])?", "\n"), (r"\\end\{enumerate\}", "\n"),
        (r"\\begin\{center\}", "\n"), (r"\\end\{center\}", "\n"),
        (r"\\item\s*", "- "),
        (r"\\begin\{tabular\}(?:\[[^\]]*\])?\{.*?\}", "\n"),
        (r"\\begin\{tabularx\}(?:\{.*?\}){2}", "\n"),
        (r"\\end\{tabular\}", "\n"), (r"\\end\{tabularx\}", "\n"),
        (r"\\toprule", "\n"), (r"\\midrule", "\n"), (r"\\bottomrule", "\n"),
        (r"\\medskip", "\n"), (r"\\smallskip", "\n"), (r"\\bigskip", "\n"),
        (r"\\vspace\{[^}]*\}", "\n"), (r"\\nopagebreak\[[^\]]*\]", ""),
        (r"\\noindent", ""), (r"\\quad", " "), (r"\\enspace", " "),
        (r"\\hfill", " "), (r"\\\\(\[[^\]]*\])?", "\n"),
        (r"\\textbf\{([^}]*)\}", r"\1"), (r"\\textit\{([^}]*)\}", r"\1"),
        (r"\\emph\{([^}]*)\}", r"\1"), (r"\\texttt\{([^}]*)\}", r"\1"),
        (r"\\small", ""), (r"\\large", ""), (r"\\bfseries", ""),
        (r"\\itshape", ""), (r"\\renewcommand\{[^}]*\}\{[^}]*\}", ""),
        (r"\\color\{[^}]*\}", ""), (r"\\rule\{[^}]*\}\{[^}]*\}", ""),
        (r"\\checkmark", "✓"), (r"\\square", "☐"),
        (r"\\%", "%"), (r"\\&", "&"), (r"\\_", "_"), (r"\\,", ""), (r"\\\.", "."),
        (r"\\approx", "≈"), (r"\\rightarrow", "→"), (r"\$\\times\$", "×"),
        (r"\$\\rightarrow\$", "→"), (r"\$\\square\$", "☐"),
        (r"\$\\approx\\,\$", "≈ "), (r"\$\\approx\$", "≈"),
        (r"``", '"'), (r"''", '"'), (r"---", "—"), (r"--", "–"),
        (r"\\[a-zA-Z]+\*?\{([^}]*)\}", r"\1"),
        (r"\\[a-zA-Z]+\*?", ""), (r"[\{\}]", ""), (r"\$", ""),
        (r"\s&\s", ": "), (r"~", " "),
    ]
    out = text
    for p, r in subs:
        out = re.sub(p, r, out, flags=re.DOTALL)
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def extract_between(text: str, start: str, end: str) -> str:
    m = re.search(start + r"(.*?)" + end, text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"Could not extract between {start!r} and {end!r}")
    return m.group(1)


def extract_first(pattern: str, text: str) -> str:
    m = re.search(pattern, text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"Pattern not found: {pattern}")
    return m.group(1).strip()


def parse_prefixed_item_lines(block: str) -> list[str]:
    items = re.findall(
        r"\\item\s+\\textbf\{[^}]*\}\s*(.*?)\s*(?=(?:\\item|$))",
        block, flags=re.DOTALL,
    )
    return [latex_to_text(i).strip() for i in items]


def parse_enumerated_targets(block: str) -> list[str]:
    items = re.findall(r"\\item\s+\\textbf\{([^}]*)\}", block, flags=re.DOTALL)
    if not items:
        items = re.findall(r"\\item\s+([^\\\n]+)", block, flags=re.DOTALL)
    return [latex_to_text(i).strip() for i in items]


def parse_table_value(label: str, block: str) -> str:
    return latex_to_text(extract_first(rf"{re.escape(label)}\s*&\s*(.*?)\s*\\\\", block))


def parse_network(tikz: str, lh: str, rh: str, title: str) -> NetworkFigure:
    nodes: list[Node] = []
    edges: list[Edge] = []
    np = re.compile(
        r"\\node\[(?P<style>[a-zA-Z0-9]+)\]\s+\((?P<key>[a-zA-Z0-9]+)\)\s+at\s+"
        r"\((?P<x>-?[0-9.]+),\s*(?P<y>-?[0-9.]+)\)\s+\{(?P<body>.*?)\};",
        re.DOTALL,
    )
    ep = re.compile(
        r"\\draw\[line width=(?P<w>[0-9.]+)pt,\s*draw=(?P<col>[A-Za-z0-9!]+).*?\]\s+"
        r"\((?P<src>[a-zA-Z0-9]+)\.east\)\s+--\s+\((?P<dst>[a-zA-Z0-9]+)\.west\);"
    )
    for m in np.finditer(tikz):
        body = m.group("body").split("\\\\", 1)
        prefix = latex_to_text(body[0]).strip()
        label = latex_to_text(body[1] if len(body) > 1 else "").strip()
        key = m.group("key")
        kind = "left" if key.lower().startswith(("p", "bo")) else "right"
        nodes.append(Node(key=key, prefix=prefix, label=label,
                          x=float(m.group("x")), y=float(m.group("y")),
                          kind=kind, compact=m.group("style").startswith("sm")))
    known = {n.key for n in nodes}
    for m in ep.finditer(tikz):
        s, d = m.group("src"), m.group("dst")
        if s in known and d in known:
            sign = "positive" if "PrimaryBlue" in m.group("col") else "negative"
            edges.append(Edge(src=s, dst=d, width=float(m.group("w")), sign=sign))
    if not nodes or not edges:
        raise ValueError("parse_network: no nodes or edges found")
    return NetworkFigure(title=title, left_header=lh, right_header=rh, nodes=nodes, edges=edges)


def parse_case_survey(tex: str, example_network: NetworkFigure) -> CaseSurvey:
    hcp_code = extract_first(r"Deelnemerscode:\s*\\texttt\{(HCP-PRE-\d+)\}", tex)
    case_code = extract_first(r"\\textbf\{\\color\{PrimaryBlue\}(C\d+)\}", tex)

    p1c = extract_between(tex, r"\\subsection\*\{Casus 1:.*?\}\n", r"\\section\{Deel 2:")
    p2c = extract_between(tex, r"\\subsection\*\{Casus 1:.*?-- Deel 2\}", r"\\section\{Deel 3:")
    p3c = extract_between(tex, r"\\subsection\*\{Casus 1:.*?-- Deel 3\}", r"\\section\{Deel 4:")
    p4c = extract_between(tex, r"\\subsection\*\{Casus 1:.*?-- Deel 4\}", r"\\section\{Deel 5:")
    p5c = extract_between(tex, r"\\subsection\*\{Casus 1:.*?-- Deel 5\}", r"\\section\{Deel 6:")

    profile  = extract_first(r"title=\{Casus 1:\s*(.*?);\s*duur:", p1c).strip()
    duration = extract_first(r"title=\{Casus 1:.*?;\s*duur:\s*([^}]*)\}", p1c)
    complaint_vignette = latex_to_text(
        extract_first(r"\\begin\{complaintbox\}.*?\\small\s+(.*?)\n\\end\{complaintbox\}", p1c))
    short_summary = latex_to_text(
        extract_first(r"\\begin\{complaintbox\}.*?\\small\s+(.*?)\n\\end\{complaintbox\}", p2c))
    part2_symptoms = parse_prefixed_item_lines(
        extract_first(r"\\small\\textbf\{Gestandaardiseerde symptomen uit Deel 1:\}(.*?)\\end\{itemize\}", p2c))
    monitoring = latex_to_text(
        extract_first(r"\\begin\{monitorbox\}\s*\\small\\textbf\{21-daagse monitoring:\}\s*(.*?)\n\\end\{monitorbox\}", p3c))
    network_tikz = extract_first(r"\\begin\{tikzpicture\}(.*?)\\end\{tikzpicture\}", p3c)
    network = parse_network(network_tikz, "Behandelingsopties", "Symptomen", case_code)
    part3_options = [f"{n.prefix}: {n.label}" for n in network.nodes if n.kind == "left"]
    part4_targets = parse_enumerated_targets(
        extract_first(r"Gestandaardiseerde behandeldoelen \(abstract\) voor Deel 4:\}(.*?)\\end\{enumerate\}", p4c))
    part4_items = [latex_to_text(item)
                   for _, item in re.findall(r"\\checkitemBFS\{(\d+)\}\{([^}]*)\}", p4c)]
    return CaseSurvey(
        hcp_code=hcp_code, case_code=case_code, profile=profile, duration=duration,
        complaint_vignette=complaint_vignette, short_summary=short_summary,
        part2_symptoms=part2_symptoms, monitoring=monitoring,
        part3_options=part3_options, part4_targets=part4_targets, part4_items=part4_items,
        part5_primary_problem=parse_table_value("Primair probleem", p5c),
        part5_target=parse_table_value("Behandeldoel", p5c),
        part5_barrier=parse_table_value("Voornaamste barriere", p5c),
        part5_coping=parse_table_value("Copingstrategie", p5c),
        network=network, example_network=example_network,
    )


# ─────────────────────────────────────────────────────────────────────────────
# NETWORK FIGURE → base64 PNG
# ─────────────────────────────────────────────────────────────────────────────

def network_to_b64(figure: NetworkFigure, dpi: int = 180) -> str:
    compact = any(n.compact for n in figure.nodes)
    fw, fh = (7.4, 4.8) if compact else (8.5, 6.0)
    lw = 2.95 if compact else 3.45; rw = lw
    nh = 0.68 if compact else 0.76
    fp = 6.1 if compact else 6.4; fl = 5.9 if compact else 6.0

    xs = [n.x for n in figure.nodes]; ys = [n.y for n in figure.nodes]
    mnx, mxx, mny, mxy = min(xs), max(xs), min(ys), max(ys)

    fig, ax = plt.subplots(figsize=(fw, fh))
    bg, bdr = "#F8FAFC", "#CBD5E1"
    ax.set_facecolor(bg); fig.patch.set_facecolor(bg); ax.axis("off")
    xp, yp_t, yp_b = 2.3, 1.2, 2.2
    ax.set_xlim(mnx - xp, mxx + xp); ax.set_ylim(mny - yp_b, mxy + yp_t)

    ax.add_patch(mpatches.FancyBboxPatch(
        (mnx - xp + 0.1, mny - yp_b + 0.1),
        (mxx - mnx) + 2*xp - 0.2, (mxy - mny) + yp_t + yp_b - 0.2,
        boxstyle="round,pad=0.02", lw=0.8, edgecolor=bdr, facecolor=bg, zorder=0,
    ))
    ln = [n for n in figure.nodes if n.kind == "left"]
    rn = [n for n in figure.nodes if n.kind == "right"]
    if ln:
        ax.text(sum(n.x for n in ln)/len(ln), mxy+0.62, figure.left_header,
                ha="center", va="center", fontsize=8, fontweight="bold", color="#047857", style="italic")
    if rn:
        ax.text(sum(n.x for n in rn)/len(rn), mxy+0.62, figure.right_header,
                ha="center", va="center", fontsize=8, fontweight="bold", color="#1E3A5F", style="italic")

    nl = {n.key: n for n in figure.nodes}
    for e in figure.edges:
        s, d = nl[e.src], nl[e.dst]
        c = "#1D4ED8" if e.sign == "positive" else "#B91C1C"
        ax.plot([s.x+0.38, d.x-0.38], [s.y, d.y], color=c, linewidth=e.width, alpha=0.86, zorder=1,
                solid_capstyle="round")
    for n in figure.nodes:
        face = "#CCFBF1" if n.kind == "left" else "#DBEAFE"
        ec = "#047857" if n.kind == "left" else "#1D4ED8"
        tc = "#047857" if n.kind == "left" else "#1E3A5F"
        w = lw if n.kind == "left" else rw
        ax.add_patch(mpatches.FancyBboxPatch(
            (n.x-w/2, n.y-nh/2), w, nh,
            boxstyle="round,pad=0.05", lw=1.1, edgecolor=ec, facecolor=face, zorder=2,
        ))
        ax.text(n.x, n.y+0.10, n.prefix, ha="center", va="center",
                fontsize=fp, fontweight="bold", color=tc, zorder=3)
        ax.text(n.x, n.y-0.12, n.label, ha="center", va="center",
                fontsize=fl, color=tc, zorder=3)

    ly = mny - 1.0
    ax.axhline(ly+0.52, xmin=0.05, xmax=0.95, color=bdr, linewidth=0.8)
    ax.plot([mnx-0.6, mnx+0.2], [ly, ly], color="#1D4ED8", linewidth=2.4, solid_capstyle="round")
    ax.text(mnx+0.45, ly, "Blauw = positief verband (BO vergroot S)",
            ha="left", va="center", fontsize=6.1, color="#1E293B")
    ax.plot([mnx-0.6, mnx+0.2], [ly-0.42, ly-0.42], color="#B91C1C", linewidth=2.4, solid_capstyle="round")
    ax.text(mnx+0.45, ly-0.42, "Rood = negatief verband (BO verkleint S)",
            ha="left", va="center", fontsize=6.1, color="#1E293B")
    ax.text((mnx+mxx)/2, ly-0.92,
            "Lijndikte proportioneel aan |w|, genormaliseerd binnen netwerk (bereik 1–5 pt).",
            ha="center", va="center", fontsize=5.6, color="#64748B", style="italic")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=bg, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ─────────────────────────────────────────────────────────────────────────────
# HTML HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def _box(title: str, body: str, bg: str, bdr: str, title_color: str,
         body_color: str = "#1E293B", border_width: str = "2px") -> str:
    return (
        f'<div style="background:{bg};border:{border_width} solid {bdr};border-radius:6px;'
        f'padding:14px 18px;margin:10px 0;">'
        + (f'<div style="font-weight:bold;color:{title_color};margin-bottom:8px;font-size:13px;">'
           f'{title}</div>' if title else "")
        + f'<div style="font-size:13px;line-height:1.65;color:{body_color};">{body}</div></div>'
    )

def blue_box(title: str, body: str) -> str:
    return _box(title, body, "#EFF6FF", "#1D4ED8", "#1D4ED8")

def gray_box(title: str, body: str) -> str:
    return _box(title, body, "#F8FAFC", "#CBD5E1", "#475569", border_width="1.5px")

def green_box(title: str, body: str) -> str:
    return _box(title, body, "#F0FDF4", "#047857", "#047857")

def amber_box(title: str, body: str) -> str:
    return _box(title, body, "#FFFBEB", "#B45309", "#92400E", "#78350F")

def purple_box(title: str, subtitle: str, body: str) -> str:
    return (
        f'<div style="background:#EDE9FE;border:2px solid #6D28D9;border-radius:6px;'
        f'padding:14px 18px;margin:10px 0;">'
        f'<div style="font-weight:bold;color:#6D28D9;text-align:center;margin-bottom:4px;font-size:13px;">'
        f'Voorbeeld &#8212; {title}</div>'
        f'<div style="color:#7C3AED;text-align:center;font-style:italic;margin-bottom:10px;font-size:11px;">'
        f'{subtitle}</div>'
        f'<div style="font-size:13px;line-height:1.65;color:#2E1065;">{body}</div></div>'
    )

def fig_html(b64: str, alt: str) -> str:
    return (
        f'<div style="text-align:center;margin:14px 0;">'
        f'<img src="data:image/png;base64,{b64}" alt="{esc(alt)}" '
        f'style="max-width:100%;height:auto;border:1px solid #CBD5E1;border-radius:6px;" /></div>'
    )

def ul(items: list[str], style: str = "") -> str:
    return (f'<ul style="margin:4px 0;padding-left:22px;{style}">'
            + "".join(f'<li style="margin:3px 0;">{x}</li>' for x in items)
            + "</ul>")

def ol(items: list[str]) -> str:
    return ('<ol style="margin:4px 0;padding-left:22px;">'
            + "".join(f'<li style="margin:3px 0;">{x}</li>' for x in items)
            + "</ol>")

def p(t: str) -> str:
    return f'<p style="margin:5px 0;">{t}</p>'

def b(t: str) -> str:
    return f"<strong>{t}</strong>"

def i(t: str) -> str:
    return f"<em>{t}</em>"

def hr() -> str:
    return '<hr style="border:none;border-top:1px solid #E2E8F0;margin:10px 0;" />'

def h2(t: str, color: str = "#1E3A5F") -> str:
    return (f'<h2 style="color:{color};border-bottom:2px solid #1D4ED8;'
            f'padding-bottom:6px;margin-top:0;">{t}</h2>')

def h3(t: str, color: str = "#1D4ED8") -> str:
    return f'<h3 style="color:{color};margin-bottom:4px;">{t}</h3>'

def plain(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html)[:120].strip()

def meta_table(rows: list[tuple[str, str]]) -> str:
    body = ""
    for i, (label, value) in enumerate(rows):
        bg = "#F8FAFC" if i % 2 == 0 else "white"
        body += (f'<tr style="background:{bg};">'
                 f'<td style="padding:6px 10px;font-weight:bold;color:#475569;width:30%;vertical-align:top;">{label}</td>'
                 f'<td style="padding:6px 10px;">{value}</td></tr>')
    return f'<table style="width:100%;border-collapse:collapse;font-size:13px;">{body}</table>'

# ─────────────────────────────────────────────────────────────────────────────
# ID GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def gen_id(prefix: str, seed: str, length: int = 15) -> str:
    return prefix + hashlib.sha256(seed.encode()).hexdigest()[:length].upper()


# ─────────────────────────────────────────────────────────────────────────────
# QSF BUILDER
# ─────────────────────────────────────────────────────────────────────────────

class QSFBuilder:
    def __init__(self, survey_id: str, survey_name: str):
        self.survey_id = survey_id
        self.survey_name = survey_name
        self.questions: list[dict] = []
        self.blocks: list[dict] = []
        self._qid = 1
        self._bid = 1
        self._cur: dict | None = None

    def _qid_str(self) -> str:
        q = f"QID{self._qid}"; self._qid += 1; return q

    def _bid_str(self) -> str:
        b = gen_id("BL_", f"{self.survey_id}_b{self._bid}"); self._bid += 1; return b

    def block(self, name: str) -> "QSFBuilder":
        self._cur = {"id": self._bid_str(), "name": name, "elements": []}
        self.blocks.append(self._cur)
        return self

    def page_break(self) -> "QSFBuilder":
        if self._cur:
            self._cur["elements"].append({"Type": "PageBreak"})
        return self

    def _reg(self, payload: dict) -> str:
        q = self._qid_str()
        payload["QuestionID"] = q
        short = plain(payload.get("QuestionText", ""))
        self.questions.append({
            "SurveyID": self.survey_id, "Element": "SQ",
            "PrimaryAttribute": q, "SecondaryAttribute": short,
            "TertiaryAttribute": None, "Payload": payload,
        })
        if self._cur:
            self._cur["elements"].append({"Type": "Question", "QuestionID": q})
        return q

    def db(self, html: str, tag: str | None = None) -> str:
        t = tag or f"QID{self._qid}"
        return self._reg({
            "Type": "DB", "Selector": "TB", "SubSelector": "TX", "DataExportTag": t,
            "QuestionText": html, "QuestionDescription": plain(html),
            "ChoiceOrder": [], "Choices": {}, "GradingData": [], "Language": [],
            "NextChoiceId": 1, "NextAnswerId": 1, "QuestionText_Unsafe": html,
            "Configuration": {"QuestionDescriptionOption": "UseText"},
        })

    def mc_single(self, html: str, choices: list[str], tag: str | None = None,
                  skip_end_if: int | None = None, force: bool = True) -> str:
        t = tag or f"QID{self._qid}"
        payload: dict = {
            "Type": "MC", "Selector": "SAVR", "SubSelector": "TX", "DataExportTag": t,
            "QuestionText": html, "QuestionDescription": plain(html),
            "ChoiceOrder": list(range(1, len(choices)+1)),
            "Choices": {str(i+1): {"Display": c} for i, c in enumerate(choices)},
            "GradingData": [], "Language": [],
            "NextChoiceId": len(choices)+1, "NextAnswerId": 1,
            "QuestionText_Unsafe": html,
            "Configuration": {"QuestionDescriptionOption": "UseText"},
            "Validation": {"Settings": {
                "ForceResponse": "ON" if force else "OFF",
                "ForceResponseType": "ON" if force else "OFF",
                "Type": "None",
            }},
        }
        if skip_end_if is not None:
            q_ref = f"QID{self._qid}"
            payload["SkipLogic"] = [{
                "SkipLogicID": 1, "ConditionType": "Choice",
                "Condition": "Is Selected",
                "ChoiceLocator": f"q://{q_ref}/SelectableChoice/{skip_end_if}",
                "Locator": f"q://{q_ref}/SelectableChoice/{skip_end_if}",
                "SkipToDestination": "SURVEY_END",
                "Description": "Consent refused — end survey",
            }]
        return self._reg(payload)

    def mc_multi(self, html: str, choices: list[str], tag: str | None = None,
                 min_c: int | None = None, max_c: int | None = None) -> str:
        t = tag or f"QID{self._qid}"
        if min_c is not None or max_c is not None:
            validation = {"Settings": {
                "ForceResponse": "ON", "ForceResponseType": "ON",
                "Type": "SelectMany",
                "MinChoices": str(min_c or 0), "MaxChoices": str(max_c or 999),
            }}
        else:
            validation = {"Settings": {"ForceResponse": "OFF", "Type": "None"}}
        return self._reg({
            "Type": "MC", "Selector": "MAVR", "SubSelector": "TX", "DataExportTag": t,
            "QuestionText": html, "QuestionDescription": plain(html),
            "ChoiceOrder": list(range(1, len(choices)+1)),
            "Choices": {str(i+1): {"Display": c} for i, c in enumerate(choices)},
            "GradingData": [], "Language": [],
            "NextChoiceId": len(choices)+1, "NextAnswerId": 1,
            "QuestionText_Unsafe": html,
            "Configuration": {"QuestionDescriptionOption": "UseText"},
            "Validation": validation,
        })

    def te_essay(self, html: str, tag: str | None = None, force: bool = False) -> str:
        t = tag or f"QID{self._qid}"
        return self._reg({
            "Type": "TE", "Selector": "ML", "SubSelector": "NULL", "DataExportTag": t,
            "QuestionText": html, "QuestionDescription": plain(html),
            "ChoiceOrder": [], "Choices": {}, "GradingData": [], "Language": [],
            "NextChoiceId": 1, "NextAnswerId": 1, "QuestionText_Unsafe": html,
            "Configuration": {"QuestionDescriptionOption": "UseText"},
            "Validation": {"Settings": {
                "ForceResponse": "ON" if force else "OFF", "Type": "None",
            }},
        })

    def te_form(self, html: str, fields: list[str], tag: str | None = None) -> str:
        t = tag or f"QID{self._qid}"
        return self._reg({
            "Type": "TE", "Selector": "FORM", "SubSelector": "NULL", "DataExportTag": t,
            "QuestionText": html, "QuestionDescription": plain(html),
            "ChoiceOrder": list(range(1, len(fields)+1)),
            "Choices": {str(i+1): {"Display": f} for i, f in enumerate(fields)},
            "GradingData": [], "Language": [],
            "NextChoiceId": len(fields)+1, "NextAnswerId": 1,
            "QuestionText_Unsafe": html,
            "Configuration": {"QuestionDescriptionOption": "UseText"},
            "Validation": {"Settings": {"ForceResponse": "OFF", "Type": "None"}},
        })

    def matrix(self, html: str, statements: list[str], scale: list[str],
               tag: str | None = None) -> str:
        t = tag or f"QID{self._qid}"
        return self._reg({
            "Type": "Matrix", "Selector": "Likert", "SubSelector": "SingleAnswer",
            "DataExportTag": t,
            "QuestionText": html, "QuestionDescription": plain(html),
            "ChoiceOrder": list(range(1, len(statements)+1)),
            "Choices": {str(i+1): {"Display": s} for i, s in enumerate(statements)},
            "AnswerOrder": list(range(1, len(scale)+1)),
            "Answers": {str(i+1): {"Display": sp} for i, sp in enumerate(scale)},
            "GradingData": [], "Language": [],
            "NextChoiceId": len(statements)+1, "NextAnswerId": len(scale)+1,
            "QuestionText_Unsafe": html,
            "Configuration": {"QuestionDescriptionOption": "UseText"},
            "Validation": {"Settings": {"ForceResponse": "OFF", "Type": "None"}},
        })

    def rank_order(self, html: str, choices: list[str], tag: str | None = None) -> str:
        t = tag or f"QID{self._qid}"
        return self._reg({
            "Type": "RO", "Selector": "TX", "SubSelector": "NULL", "DataExportTag": t,
            "QuestionText": html, "QuestionDescription": plain(html),
            "ChoiceOrder": list(range(1, len(choices)+1)),
            "Choices": {str(i+1): {"Display": c} for i, c in enumerate(choices)},
            "GradingData": [], "Language": [],
            "NextChoiceId": len(choices)+1, "NextAnswerId": 1,
            "QuestionText_Unsafe": html,
            "Configuration": {"QuestionDescriptionOption": "UseText"},
            "Validation": {"Settings": {"ForceResponse": "OFF", "Type": "None"}},
        })

    def build(self) -> dict:
        rs_id = gen_id("RS_", self.survey_id)
        block_payload = {
            str(i): {
                "Type": "Default", "SubType": "", "Description": blk["name"],
                "ID": blk["id"], "BlockElements": blk["elements"],
                "Options": {"BlockLocking": "false", "RandomizeQuestions": "false",
                            "BlockVisibility": "Expanded"},
            }
            for i, blk in enumerate(self.blocks)
        }
        flow = [{"Type": "Block", "ID": blk["id"], "FlowID": f"FL_{i+1}"}
                for i, blk in enumerate(self.blocks)]
        flow.append({"Type": "EndSurvey", "FlowID": f"FL_{len(self.blocks)+1}"})
        elements: list[dict] = [
            {"SurveyID": self.survey_id, "Element": "BL", "PrimaryAttribute": "Survey Blocks",
             "SecondaryAttribute": None, "TertiaryAttribute": None, "Payload": block_payload},
            {"SurveyID": self.survey_id, "Element": "FL", "PrimaryAttribute": "Survey Flow",
             "SecondaryAttribute": None, "TertiaryAttribute": None, "Payload": {
                 "Type": "Root", "FlowID": "FL_0", "Flow": flow,
                 "Properties": {"Count": len(flow)}}},
            {"SurveyID": self.survey_id, "Element": "SO", "PrimaryAttribute": "Survey Options",
             "SecondaryAttribute": "Default Question Block", "TertiaryAttribute": None,
             "Payload": {
                 "BackButton": "false", "SaveAndContinue": "true",
                 "SurveyProtection": "PublicSurvey", "BallotBoxStuffingPrevention": "false",
                 "NoIndex": "No", "SecureResponseFiles": "true", "SurveyExpiration": "None",
                 "SurveyTermination": "DefaultMessage", "Header": "", "Footer": "",
                 "ProgressBarDisplay": "WithText", "PartialData": "+7 days",
                 "ValidationMessage": "", "InactiveSurvey": "DefaultMessage",
                 "AvailableLanguages": {"NL": "Dutch"}, "Language": "NL",
                 "CustomStyles": "", "HeaderMid": "", "FooterMid": ""}},
            {"SurveyID": self.survey_id, "Element": "PROJ", "PrimaryAttribute": "Survey Project",
             "SecondaryAttribute": None, "TertiaryAttribute": None,
             "Payload": {"ProjectCategory": "CORE", "SchemaVersion": "1.1.0"}},
            {"SurveyID": self.survey_id, "Element": "RS", "PrimaryAttribute": rs_id,
             "SecondaryAttribute": "Default Response Set", "TertiaryAttribute": None,
             "Payload": {"ID": rs_id, "Name": "Default Response Set", "IsDefault": True,
                         "CreationDate": "2026-04-21 10:00:00",
                         "LastModifiedDate": "2026-04-21 10:00:00"}},
        ]
        elements.extend(self.questions)
        return {
            "SurveyEntry": {
                "SurveyID": self.survey_id, "SurveyName": self.survey_name,
                "SurveyDescription": None, "SurveyOwnerID": "UR_00000000000000000",
                "SurveyBrandID": "ugent", "DivisionID": None, "SurveyLanguage": "NL",
                "SurveyActiveResponseSet": rs_id, "SurveyStatus": "Inactive",
                "SurveyStartDate": "0000-00-00 00:00:00",
                "SurveyExpirationDate": "0000-00-00 00:00:00",
                "SurveyCreationDate": "2026-04-21 10:00:00",
                "CreatorID": "UR_00000000000000000",
                "LastModified": "2026-04-21 10:00:00",
                "LastAccessed": "0000-00-00 00:00:00",
                "LastActivated": "0000-00-00 00:00:00",
                "Deleted": None,
            },
            "SurveyElements": elements,
        }


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 0: TITELPAGINA
# ─────────────────────────────────────────────────────────────────────────────

def page_title(case: CaseSurvey) -> str:
    return (
        f'<div style="background:#1E3A5F;border-radius:8px;padding:22px 28px;'
        f'margin-bottom:18px;text-align:center;">'
        f'<div style="font-size:24px;font-weight:bold;color:white;margin-bottom:8px;">'
        f'PHOENIX evaluatiestudie</div>'
        f'<div style="font-size:16px;color:#BFDBFE;margin-bottom:6px;">'
        f'Fase 1 &#8212; onafhankelijke expertgeneratie</div>'
        f'<div style="font-size:13px;color:white;font-style:italic;">'
        f'Instrument voor Zorgprofessionals</div></div>'
        + f'<div style="background:#DBEAFE;border:2px solid #1D4ED8;border-radius:6px;'
        f'padding:14px 20px;text-align:center;margin-bottom:14px;">'
        f'<div style="font-size:16px;font-weight:bold;color:#1E3A5F;">Deelnemerscode: '
        f'<span style="font-family:monospace;">{case.hcp_code}</span></div>'
        f'<div style="font-size:13px;color:#475569;margin-top:6px;">Toegewezen casus:&nbsp;'
        f'<strong style="color:#1D4ED8;">{case.case_code}</strong> '
        f'({esc(case.profile)})</div></div>'
        + gray_box("Studiegegevens", meta_table([
            ("Studie", "Evaluatie van de klinische kwaliteit van een ontologiegebaseerd "
                       "multi-agentsysteem voor gepersonaliseerde digitale geestelijke "
                       "gezondheidszorg (PHOENIX)"),
            ("Instelling", "Universiteit Gent — Faculteit Psychologie en Pedagogische Wetenschappen"),
            ("Onderzoeker", "Stijn Van Severen (masterproefstudent)"),
            ("Promotoren", "Prof. Dr. Geert Crombez; Dr. Annick De Paepe"),
            ("Contact", "stijn.vanseveren@ugent.be"),
            ("Geschatte duur", "Ongeveer 25–35 minuten voor de casus (inclusief leestijd "
                               "instructies en Deel 6)"),
        ]))
        + gray_box("Doel van deze bundel",
            p("U neemt deel aan een evaluatiestudie waarin zorgprofessionals onafhankelijk "
              "dezelfde vijf klinische redeneerstappen uitvoeren als het PHOENIX-systeem. "
              "Uw antwoorden vormen het menselijke referentiecorpus voor een latere "
              "<strong>dubbelblinde vergelijking</strong> met systeemoutput.")
            + p("Voor uw casus vult u dezelfde vijf inhoudelijke delen in: "
                "(1)&nbsp;operationalisering, (2)&nbsp;initieel observatiemodel, "
                "(3)&nbsp;prioritering van behandeldoelen, (4)&nbsp;verfijning van "
                "EMA-metingen en (5)&nbsp;een mobiele coachingsboodschap. Aansluitend "
                "beantwoordt u in <strong>Deel 6</strong> drie reflectievragen over uw "
                "digitale benaderingslogica.")
            + p("<strong>We vragen uw eigen klinische oordeelsvorming.</strong> Antwoord "
                "zoals u dat in een reële professionele context zou doen, maar werk strikt "
                "volgens de instructies op de volgende pagina.")
        )
        + amber_box("Vertrouwelijkheid",
            "Uw antwoorden worden voor analyse geanonimiseerd en uitsluitend gebruikt "
            "binnen deze masterproefstudie. Deelname is <strong>volledig vrijwillig</strong>; "
            "u kunt zich op elk moment terugtrekken door contact op te nemen met de onderzoeker "
            "via stijn.vanseveren@ugent.be."
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 0b: GEINFORMEERDE TOESTEMMING
# ─────────────────────────────────────────────────────────────────────────────

def page_consent(case: CaseSurvey) -> str:
    return (
        h2("Geïnformeerde toestemming")
        + blue_box("",
            p(f"U staat op het punt de evaluatiesurvey voor "
              f"<strong>{case.hcp_code}</strong> te starten.")
            + p("Lees onderstaande informatie zorgvuldig door voordat u verdergaat.")
        )
        + gray_box("Doel van het onderzoek",
            p("Deze survey verzamelt onafhankelijke expertantwoorden van zorgprofessionals "
              "op dezelfde klinische redeneerstappen die het PHOENIX-systeem later uitvoert. "
              "Uw antwoorden worden gebruikt als menselijk referentiecorpus voor een "
              "<strong>dubbelblinde vergelijking</strong> met systeemoutput in het kader van "
              "een masterproef aan de Universiteit Gent (promotoren: Prof. Dr. Geert Crombez; "
              "Dr. Annick De Paepe).")
        )
        + gray_box("Wat deelname inhoudt",
            ul([
                "U vult eerst een korte <strong>intake</strong> in (3 vragen: achtergrond, "
                "houding tegenover agentic AI, en kennis van netwerkanalyse).",
                "Vervolgens volgt uw toegewezen <strong>casus</strong> met 6 inhoudelijke delen "
                "(symptoomidentificatie, behandelingsopties, prioritering, EMA-itemselectie, "
                "coachingsboodschap, reflectie).",
                "De survey duurt <strong>ongeveer 25–35 minuten</strong> in totaal.",
                "We vragen uitsluitend uw <strong>eigen klinisch oordeel</strong> — "
                "<em>geen generatieve AI, collegaoverleg of andere externe hulpmiddelen</em>.",
            ])
        )
        + green_box("Vrijwilligheid en vertrouwelijkheid",
            ul([
                "Deelname is <strong>volledig vrijwillig</strong>.",
                "U kunt op <strong>elk moment stoppen</strong> door het venster te sluiten; "
                "uw gedeeltelijke antwoorden worden dan niet verwerkt.",
                "Uw antwoorden worden na ontvangst <strong>geanonimiseerd</strong> en "
                "uitsluitend voor deze masterproefanalyse gebruikt.",
                "De resultaten worden gerapporteerd op groepsniveau; individuele antwoorden "
                "zijn niet herleidbaar naar uw persoon.",
            ])
        )
        + amber_box("Contact",
            meta_table([
                ("Onderzoeker", "Stijn Van Severen"),
                ("E-mail", "stijn.vanseveren@ugent.be"),
                ("Instelling", "Universiteit Gent, Fac. Psychologie en Pedagogische Wetenschappen"),
            ])
        )
        + p("<em>Kies hieronder of u op basis van bovenstaande informatie wenst deel te nemen. "
            "Indien u <strong>niet</strong> wenst deel te nemen, kiest u optie 2 — de survey "
            "sluit dan automatisch af.</em>")
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1a: INSTRUCTIEPAGINA — deel 1 (PHOENIX + kerndefinities + EMA)
# ─────────────────────────────────────────────────────────────────────────────

def page_instr_p1() -> str:
    return (
        h2("Instructiepagina — het PHOENIX-systeem en kerndefinities")
        + p("<strong>PHOENIX</strong> (Personalized Hierarchical Optimization Engine for "
            "Navigating Insightful eXplorations) is een multi-agentsysteem ontwikkeld als "
            "onderdeel van een masterproef in de psychologie aan de Universiteit Gent. "
            "Het systeem verwerkt een <em>vrije klachttekst</em> en doorloopt vijf "
            "opeenvolgende klinische redeneerstappen die de kern vormen van een "
            "gepersonaliseerde, longitudinale digitale interventie.")
        + p("Het theoretische kader steunt op het <strong>netwerk-analytisch model van "
            "psychopathologie</strong>: psychische klachten worden niet als uitingen van "
            "een latente stoornis beschouwd, maar als een <em>dynamisch netwerk van "
            "onderling samenhangende klachtcomponenten</em>. Een sleutelbegrip daarbinnen "
            "is het onderscheid tussen <strong>symptomen</strong> en "
            "<strong>behandelingsopties</strong>.")
        + gray_box("Kerndefinities: symptoom versus behandelingsoptie",
            p("<strong>Symptoom (S):</strong> een klinisch relevante, momentaan aanwezige "
              "<em>klacht- of toestandsdimensie</em> van de persoon die herhaald meetbaar "
              "is via dagelijkse EMA. Symptomen zijn de <em>uitkomstvariabelen</em> in het "
              "netwerk — ze beschrijven <em>wat er mis gaat</em> (bijv. inslaapproblemen, "
              "paniekepisoden, sombere stemming). Symptomen zijn <u>geen</u> gedragingen of "
              "interventies.")
            + p("<strong>Behandelingsoptie (BO):</strong> een <em>modificeerbare gedrags- "
                "of procesvariabele</em> die plausibel causaal ingrijpt op een of meerdere "
                "symptomen en dagelijks meetbaar is via een korte EMA-vraag. "
                "Behandelingsopties zijn de <em>ingreepvariabelen</em> in het netwerk — "
                "ze beschrijven <em>wat veranderbaar is</em> (bijv. avondschermtijd, "
                "geplande exposure, aerobe beweging). Een behandelingsoptie is "
                "<u>nooit</u> een symptoom zelf, maar een gedrag, strategie of "
                "omgevingsfactor die het symptoom beïnvloedt.")
            + p("<strong>Voorbeelden ter verduidelijking:</strong>")
            + ul([
                b("Symptoom:") + " inslaapproblemen, anticipatieangst, anergie, "
                "piekeren voor het slapengaan, sombere stemming, emotionele uitputting.",
                b("Behandelingsoptie:") + " avondschermtijd (min), geplande "
                "exposurestap (ja/nee), bewegingsfrequentie (min), preslaap-offloading "
                "(ja/nee), werk-privégrens (ja/nee), herstelactiviteit (min).",
            ])
        )
        + gray_box("Ecological Momentary Assessment (EMA) — basisprincipes",
            p("<strong>EMA</strong> is een methode waarbij personen meerdere keren per dag "
              "hun actuele toestand of gedrag rapporteren via een mobiele applicatie. "
              "In PHOENIX wordt EMA gebruikt om zowel symptomen als behandelingsopties "
              "dagelijks te monitoren over een periode van meerdere weken.")
            + p("Een goede EMA-variabele voldoet aan vier vereisten:")
            + ul([
                b("Dagelijks rapporteerbaar:") + " via een korte vraag op de smartphone, "
                "bijv. (ja/nee, aantal, minuten of 0–10 schaal).",
                b("Dynamisch en veranderbaar:") + " geen vaste trek, diagnose of stabiel "
                "achtergrondkenmerk.",
                b("Klinisch relevant:") + " toont binnen-persoonsvariatie die therapeutisch "
                "informatief is.",
                b("Onderscheid symptoom vs. behandelingsoptie:") + " klachtdimensies zijn "
                "symptomen; modificeerbare gedragingen zijn behandelingsopties.",
            ])
            + p(i("De verzamelde EMA-data worden vervolgens ingezet voor "
                  + b("netwerkanalyse") + ": het kwantitatief in kaart brengen van de "
                  "empirische verbanden tussen dagelijkse gedragingen (behandelingsopties) "
                  "en klachtniveaus (symptomen) over meerdere weken. Het resulterende "
                  "tweedelig netwerk maakt behandelprioriteiten empirisch zichtbaar "
                  "(zie Deel 3)."))
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1b: INSTRUCTIEPAGINA — deel 2 (tweedelig netwerk + taken + werkwijze)
# ─────────────────────────────────────────────────────────────────────────────

def page_instr_p2() -> str:
    task_table = (
        '<table style="width:100%;border-collapse:collapse;font-size:12px;margin:8px 0;">'
        '<thead><tr style="background:#1D4ED8;color:white;">'
        + "".join(f'<th style="padding:7px;text-align:left;">{h}</th>'
                  for h in ["Stap", "Klinische taak", "Wat u doet", "Richttijd"])
        + '</tr></thead><tbody>'
        + "".join(
            f'<tr style="background:{"#F8FAFC" if i%2==0 else "white"};">'
            f'<td style="padding:6px;color:{c};font-weight:{fw};">{stp}</td>'
            f'<td style="padding:6px;">{task}</td>'
            f'<td style="padding:6px;">{desc}</td>'
            f'<td style="padding:6px;">{time}</td></tr>'
            for i, (stp, c, fw, task, desc, time) in enumerate([
                ("—", "#475569", "normal", "Instructies lezen",
                 "Lees de instructiepagina (deze pagina's) aandachtig door", "≈ 5 min"),
                ("1", "#1D4ED8", "bold", "Operationalisering",
                 "Identificeer 2–6 <strong>symptoomlabels</strong> (klachtdimensies)", "≈ 3 min"),
                ("2", "#1D4ED8", "bold", "Initieel observatiemodel",
                 "Genereer 3–5 <strong>behandelingsoptielabels</strong> (modificeerbaar, EMA-geschikt)", "≈ 3 min"),
                ("3", "#1D4ED8", "bold", "Behandeldoelprioritering",
                 "Rangschik alle 5 <strong>behandelingsopties</strong> van hoog naar laag prioriteit", "≈ 4 min"),
                ("4", "#1D4ED8", "bold", "Verfijnd observatiemodel",
                 "Selecteer exact 6 <strong>EMA-items</strong> (2 per behandeldoel) uit de lijst van 20", "≈ 4 min"),
                ("5", "#1D4ED8", "bold", "Mobiele coaching",
                 "Schrijf een korte patiëntgerichte boodschap voor de app", "≈ 4 min"),
                ("6", "#1D4ED8", "bold", "Reflectie digitale aanpak",
                 "Beantwoord drie reflectievragen over uw EMA-benaderingslogica", "≈ 5 min"),
            ])
        )
        + '<tr style="font-weight:bold;background:#DBEAFE;">'
        '<td style="padding:6px;"></td><td style="padding:6px;">Totaal (1 casus)</td>'
        '<td style="padding:6px;"></td><td style="padding:6px;">25–35 min</td></tr>'
        '</tbody></table>'
    )
    return (
        h2("Instructiepagina — netwerk, taakoverzicht en werkwijze")
        + gray_box("Het tweedelig netwerk: structuur van het observatiemodel",
            p("Na voldoende EMA-monitoring construeert PHOENIX een <strong>tweedelig netwerk</strong>: "
              "een graaf met twee kolommensets en gewogen verbindingen daartussen.")
            + ul([
                b("Linkse kolom — Behandelingsopties (BO):") + " de modificeerbare variabelen.",
                b("Rechtse kolom — Symptomen (S):") + " de klacht- en toestandsdimensies.",
                b("Kanten:") + " de richting loopt van behandelingsoptie naar symptoom (BO → S). "
                "Een <span style='color:#1D4ED8;font-weight:bold;'>blauwe</span> rand duidt op een "
                "<em>positief</em> verband (behandelingsoptie vergroot het symptoom), een "
                "<span style='color:#B91C1C;font-weight:bold;'>rode</span> rand op een "
                "<em>negatief</em> verband (behandelingsoptie verkleint het symptoom). "
                "Lijndikte is proportioneel aan de sterkte van het empirische verband uit de "
                "monitoringdata.",
            ])
            + p("In <strong>Deel 3</strong> rangschikt u de behandelingsopties op behandelprioriteit "
                "op basis van dit netwerk en de monitoringsamenvatting.")
        )
        + h3("Overzicht van de vijf taken")
        + task_table
        + gray_box("Werkwijze — strikt te volgen",
            ol([
                b("Werk sequentieel: Deel 1 → Deel 6.") + " Ga pas naar een volgend deel "
                "wanneer het huidige deel volledig is afgewerkt. Deel 6 is éénmalig, los van de casus.",
                b("Gebruik geen generatieve AI, schrijfhulpmiddelen, richtlijnen of collegaoverleg.") +
                " Extern gebruik ondermijnt de methodologische validiteit en de "
                "blind scoringswaarde van de studie.",
                b("Gebruik in latere delen uitsluitend de meegeleverde gestandaardiseerde context.") +
                " Die is bewust vastgezet zodat alle deelnemers op identieke input reageren "
                "en vergelijkbaar zijn.",
                b("Herwerk eerdere antwoorden niet retroactief") +
                " nadat u latere context hebt gezien.",
                b("Noteer of typ rechtstreeks in de voorziene antwoordvelden.") +
                " Onleesbare of ambigu geformuleerde antwoorden bemoeilijken latere blinde "
                "beoordeling.",
                b("Werk ook bij beperkte informatie.") + " Klinische teksten bevatten "
                + "vaak onvoldoende informatie voor definitieve conclusies. Geef "
                + i("altijd") + " een antwoord op basis van de beschikbare informatie "
                + "en uw klinisch oordeel. Laat geen velden blanco zonder reden.",
            ])
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# INTAKE PAGES
# ─────────────────────────────────────────────────────────────────────────────

def page_intake_intro() -> str:
    return (
        h2("Intake — achtergrond, kennis en houding")
        + gray_box("Doel van de intake",
            p("Deze intake verzamelt drie soorten context die later relevant zijn voor "
              "de interpretatie van uw expertantwoorden:")
            + ul([
                b("1. Algemene achtergrond en klinische ervaring:") + " om de professionele "
                "context van uw antwoorden te kunnen situeren (discipline, functie, jaren "
                "ervaring, mate van contact met psychische klachten).",
                b("2. Houding tegenover agentic AI in de klinische context:") + " om "
                "eventuele verwachtingsbias tegenover AI-systemen bij de latere blinde "
                "beoordeling mee te kunnen wegen.",
                b("3. Kennis van netwerk-analytisch denken:") + " omdat de casusopgaven "
                "expliciet vertrekken vanuit een netwerk-analytisch redeneermodel en deze "
                "kennis relevant is voor de interpretatie van uw antwoorden.",
            ])
            + p(i("De intake staat bewust vóór de casus, zodat uw achtergrondkennis niet "
                  "wordt gekleurd door het invullen van de inhoudelijke taken zelf."))
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# DEEL 1 — Operationalisering
# ─────────────────────────────────────────────────────────────────────────────

def page_d1_instr() -> str:
    return (
        h2("Deel 1 — Operationalisering van de mentale gezondheidstoestand",
           color="#1E3A5F")
        + blue_box("Instructies Deel 1",
            p(b("Opdracht:") + " identificeer de belangrijkste actuele klacht- en "
              "toestandsdimensies (<strong>symptomen</strong>) in de klachttekst en noteer "
              "voor elke dimensie uitsluitend een <strong>kort symptoomlabel</strong>.")
            + p(b("Wat is een symptoom?") + " Een symptoom is een "
                "<em>klacht- of toestandsdimensie</em> die beschrijft <em>wat er mis gaat</em> "
                "bij de persoon. Het is <u>geen</u> behandeling, geen gedrag en geen oorzaak, "
                "maar een actuele toestandsbeschrijving:")
            + ul([
                "Momenteel aanwezig bij de persoon (niet hypothetisch of anamnestisch).",
                "Onderscheidbaar van andere klachtdimensies (niet overlappend).",
                "In principe herhaald meetbaar via dagelijkse zelfrapportage (EMA).",
                "Op DSM-5-TR/ICD-11-niveau: concreet genoeg om afzonderlijk te meten "
                "(bijv. 'inslaapmoeilijkheden', niet het brede 'slaapstoornis'), maar niet "
                "te atomair (niet 'wakker om 3u00' als apart symptoom).",
            ])
            + p(b("Voorbeelden van goede symptoomlabels:") + " inslaapproblemen, "
                "paniekepisoden, anticipatieangst, sombere stemming, emotionele uitputting, "
                "anergie, interpersoonlijke instabiliteit, compulsief controleergedrag.")
            + p(b("Let op:") + " gedragingen (bijv. 'avondschermtijd'), oorzaken "
                "(bijv. 'werkstress') en behandelstrategieën zijn <em>geen</em> symptomen "
                "maar behandelingsopties — die horen in Deel 2.")
            + p(b("Antwoordformat:") + " label van 2–5 woorden, zonder beschrijvende zin. "
                "Noteer per casus <strong>2–6 symptomen</strong>. Laat ongebruikte velden leeg.")
            + p(i(b("Zie het uitgewerkte voorbeeld op de volgende pagina") + " voor een "
                  "concrete illustratie van wat een correct symptoomlabel is."))
        )
    )


def page_d1_example() -> str:
    body = (
        p(b("Klacht (verkorte versie):") + " " +
          i('"De afgelopen vijf maanden slaap ik slecht: ik lig lang wakker en word vroeg '
            'wakker. Overdag ben ik moe en prikkelbaar. Ik beweeg bijna niet meer en zie '
            'vrienden nauwelijks nog."'))
        + hr()
        + p(b("Correct ingevulde symptoomlabels") + " (elk één klachtdimensie, expliciet "
            "uit de tekst):")
        + ul([
            b("S-1") + "&nbsp; Inslaapmoeilijkheden "
            + i('("lig lang wakker" → afzonderlijk meetbaar)'),
            b("S-2") + "&nbsp; Vroegochtendontwaak "
            + i('("word vroeg wakker" → DSM-5-TR: early morning awakening; onderscheidbaar van S-1)'),
            b("S-3") + "&nbsp; Vermoeidheid overdag "
            + i('("moe overdag" → toestandsdimensie)'),
            b("S-4") + "&nbsp; Prikkelbaarheid "
            + i('("prikkelbaar" → emotioneel symptoom, los van S-3)'),
            b("S-5") + "&nbsp; Sociale terugtrekking "
            + i('("zie vrienden nauwelijks" → klachtpatroon)'),
        ])
        + hr()
        + p(f'<strong style="color:#B91C1C;">Dit zijn GEEN symptoomlabels</strong> '
            f'(horen in Deel 2 als behandelingsoptie):')
        + ul([
            i('"Weinig bewegen"') + " → modificeerbaar gedrag, geen klacht; hoort bij Deel 2.",
            i('"Slaapproblemen"') + " → te breed: splits naar meetbare dimensies "
            "(zie S-1 en S-2).",
        ])
    )
    return purple_box("Deel 1: Operationalisering",
                      "Hoe formuleert u een correct symptoomlabel?", body)


def page_d1_case(case: CaseSurvey) -> str:
    return (
        blue_box(f"Casus {case.case_code}: {esc(case.profile)} &nbsp;|&nbsp; "
                 f"Duur: {esc(case.duration)}",
                 f'<p style="font-style:italic;font-size:13px;">'
                 f'{esc(case.complaint_vignette)}</p>')
        + p(b("Opdracht:") + " noteer <strong>2–6 symptoomlabels</strong> voor deze casus. "
            "Gebruik enkel korte labels (2–5 woorden); voeg geen beschrijving toe. "
            "Laat ongebruikte velden leeg.")
    )


# ─────────────────────────────────────────────────────────────────────────────
# DEEL 2 — Initieel observatiemodel
# ─────────────────────────────────────────────────────────────────────────────

def page_d2_instr() -> str:
    return (
        h2("Deel 2 — Initieel observatiemodel (tweedelig netwerk)", color="#1E3A5F")
        + green_box("Instructies Deel 2",
            p(b("Opdracht:") + " genereer <strong>3–5 behandelingsoptielabels</strong> die "
              "samen het <strong>initieel observatiemodel</strong> voor deze casus vormen.")
            + p(b("Wat is een behandelingsoptie in deze context?") + " Een "
                "behandelingsoptie is een <em>modificeerbare gedrags- of procesvariabele</em> "
                "die beschrijft <em>wat de persoon kan veranderen</em>:")
            + ul([
                b("Modificeerbaar:") + " de persoon (of therapeut) kan er rechtstreeks "
                "op ingrijpen.",
                b("EMA-geschikt:") + " dagelijks meetbaar via een eenvoudige vraag op de "
                "smartphone (ja/nee, aantal, minuten of 0–10 schaal).",
                b("Causaal plausibel:") + " klinisch aannemelijk dat de behandelingsoptie "
                "de symptomen beïnvloedt.",
                b("Geen symptoom:") + " klachtniveaus of diagnostische trekken zijn "
                "<em>geen</em> behandelingsopties.",
            ])
            + p(b("Voorbeelden van goede behandelingsoptielabels:") + " avondschermtijd, "
                "geplande exposure, aerobe beweging, preslaap-offloading, veiligheidsgedrag, "
                "werk-privégrens, herstelactiviteit, compulsie-uitstel.")
            + p(b("Let op:") + " variabelen als 'eetlust', 'spierspanning' of 'moeheid' "
                "zijn klachtdimensies (symptomen) — geen behandelingsopties. Kies variabelen "
                "die de persoon actief kan uitvoeren of aanpassen.")
            + p(b("Antwoordformat:") + " enkel een label van 2–6 woorden, zonder "
                "meetdefinitie of toelichting. Laat ongebruikte velden leeg.")
            + p(i(b("Zie het uitgewerkte voorbeeld op de volgende pagina") + " voor een "
                  "concrete illustratie."))
        )
    )


def page_d2_example() -> str:
    body = (
        p(b("Gestandaardiseerde symptomen uit Deel 1 (voorbeeld):") + " S-1: "
          "Inslaapmoeilijkheden;&ensp;S-2: Vroegochtendontwaak;&ensp;S-3: Vermoeidheid "
          "overdag;&ensp;S-4: Prikkelbaarheid;&ensp;S-5: Sociale terugtrekking.")
        + hr()
        + p(b("Correct ingevulde behandelingsoptielabels:"))
        + ul([
            b("BO-1") + "&nbsp; Avondschermgebruik "
            + i("(modificeerbaar gedrag; EMA: minuten voor slapengaan)"),
            b("BO-2") + "&nbsp; Lichaamsbeweging "
            + i("(modificeerbaar; EMA: minuten actief bewegen per dag)"),
            b("BO-3") + "&nbsp; Sociaal contact "
            + i("(modificeerbaar; EMA: ja/nee contact gezocht op eigen initiatief)"),
        ])
        + hr()
        + p(f'<strong style="color:#B91C1C;">Dit zijn GEEN behandelingsopties</strong> '
            f'(zijn symptomen, horen in Deel 1):')
        + ul([
            i('"Vermoeidheid"') + " → klachtdimensie (= S-3), geen behandeldoel.",
            i('"Slechte slaap"') + " → klachtdimensie (= S-1), geen behandeldoel.",
        ])
    )
    return purple_box("Deel 2: Initieel observatiemodel",
                      "Hoe genereert u behandelingsoptielabels?", body)


def page_d2_case(case: CaseSurvey) -> str:
    return (
        blue_box(f"Casus {case.case_code}: verkorte klachtomschrijving",
                 f"<p>{esc(case.short_summary)}</p>")
        + green_box("Gestandaardiseerde symptomen uit Deel 1",
                    ul([f"<strong>{esc(s)}</strong>" for s in case.part2_symptoms]))
        + p(b("Opdracht:") + " noteer <strong>3–5 behandelingsoptielabels</strong> voor "
            "een initieel observatiemodel. Zorg dat elk label later dagelijks via een "
            "mobiele app meetbaar kan worden gemaakt. Laat ongebruikte velden leeg.")
    )


# ─────────────────────────────────────────────────────────────────────────────
# DEEL 3 — Behandeldoelprioritering
# ─────────────────────────────────────────────────────────────────────────────

def page_d3_instr() -> str:
    return (
        h2("Deel 3 — Behandeldoelprioritering via het tweedelig netwerk", color="#1E3A5F")
        + amber_box("Instructies Deel 3",
            p(b("Opdracht:") + " rangschik de <strong>5 gestandaardiseerde "
              "behandelingsopties</strong> van <strong>hoogste</strong> naar "
              "<strong>laagste</strong> behandelprioriteit.")
            + p(b("Wat is een behandeldoel?") + " Een behandeldoel is de "
                "behandelingsoptie die, als zij veranderd wordt, naar verwachting de "
                "sterkste vermindering van de symptomen oplevert. Prioriteer op basis van:")
            + ul([
                b("Monitoringbewijs:") + " frequentie en ernst van het gedragspatroon "
                "gedurende de 21-daagse monitoring.",
                b("Klinische modificeerbaarheid:") + " haalbaarheid en realiseerbaarheid "
                "voor deze specifieke persoon.",
                b("Netwerkimpact:") + " invloed op meerdere symptomen tegelijk "
                "(brede vs. smalle verbondenheid in het netwerk).",
            ])
            + p(b("Het tweedelig netwerk lezen:"))
            + ul([
                "Behandelingsopties staan in de <strong>linkerkolom "
                "(groen)</strong>; symptomen in de <strong>rechterkolom (blauw)</strong>.",
                "<span style='color:#1D4ED8;font-weight:bold;'>Blauwe rand:</span> "
                "behandelingsoptie heeft een <em>positief</em> verband met het symptoom "
                "(meer van BO → meer van S; bijv. meer veiligheidsgedrag → meer angst).",
                "<span style='color:#B91C1C;font-weight:bold;'>Rode rand:</span> "
                "behandelingsoptie heeft een <em>negatief</em> verband met het symptoom "
                "(meer van BO → minder van S; bijv. meer exposure → minder vermijding).",
                "Lijndikte: proportioneel aan de sterkte van het empirische verband "
                "(21-daagse EMA-data).",
            ])
            + p(b("Antwoordformat:") + " vul alle 5 prioriteitslijnen in, van rangorde 1 "
                "(hoogste prioriteit) tot en met 5 (laagste prioriteit).")
            + p(i(b("Zie het uitgewerkte voorbeeld op de volgende pagina") + " voor een "
                  "toelichting op het lezen van het netwerk en de prioriteringslogica."))
        )
    )


def page_d3_example(example_b64: str) -> str:
    body = (
        p(b("Monitoring (voorbeeld):") + " Schermtijd voor slapengaan: 18/21 avonden "
          "actief (gem. 65 min). Lichaamsbeweging: 0,8 sessies/week. Sociaal contact op "
          "eigen initiatief: 0,9/week.")
        + hr()
        + p(b("Tweedelig netwerk voor dit voorbeeld:"))
        + fig_html(example_b64, "Voorbeeldnetwerk Deel 3")
        + hr()
        + p(b("Prioriteringsredenering (voorbeeld):"))
        + ol([
            b("BO-1 (Avondschermgebruik)") + " — raakt S-1 en S-2 via sterke blauwe "
            + "verbanden; monitoring toont 18/21 avonden hoge schermtijd → "
            + b("prioriteit 1") + ".",
            b("BO-2 (Lichaamsbeweging)") + " — beschermend effect op S-1 en S-2 (rode "
            + "verbanden); frequentie extreem laag (0,8/week) → "
            + b("prioriteit 2") + ".",
            b("BO-3 (Sociaal contact)") + " — sterk negatief verband met S-3; eveneens "
            "laag (0,9/week) → " + b("prioriteit 3") + ".",
        ])
    )
    return purple_box("Deel 3: Behandeldoelprioritering",
                      "Hoe leest u het netwerk en rangschikt u de behandelingsopties?",
                      body)


def page_d3_case(case: CaseSurvey, case_b64: str) -> str:
    return (
        blue_box(f"Casus {case.case_code}: {esc(case.profile)}",
                 f"<p>{esc(case.short_summary)} &nbsp;|&nbsp; Duur: {esc(case.duration)}</p>")
        + gray_box("21-daagse monitoring",
                   f"<p>{esc(case.monitoring)}</p>")
        + fig_html(case_b64, f"Tweedelig netwerk — {case.case_code}")
        + green_box("Beschikbare behandelingsopties (te rangschikken)",
                    ul([f"<strong>{esc(o)}</strong>" for o in case.part3_options]))
    )


# ─────────────────────────────────────────────────────────────────────────────
# DEEL 4 — Verfijnd observatiemodel
# ─────────────────────────────────────────────────────────────────────────────

def page_d4_instr() -> str:
    return (
        h2("Deel 4 — Verfijnd observatiemodel: selectie van sub-behandelingsopties",
           color="#1E3A5F")
        + _box(
            "Instructies Deel 4",
            p(b("Opdracht:") + " selecteer per casus <strong>exact 6 EMA-items</strong>: "
              "<strong>2 sub-behandelingsopties per behandeldoel</strong> "
              "(3 behandeldoelen × 2 = 6 items).")
            + p(b("Wat zijn sub-behandelingsopties?") + " De gestandaardiseerde "
                "behandeldoelen in dit deel zijn <em>abstracte domeinen</em> (bijv. "
                "'Slaapkwaliteit' of 'Voedingskwaliteit'). Om dit domein dagelijks via EMA "
                "te monitoren, moet het vertaald worden naar <em>concrete, specifieke "
                "gedragsitems</em> — de <strong>sub-behandelingsopties</strong>.")
            + p(b("Hoe werkt de abstracte-naar-concrete vertaling?"))
            + ul([
                "Een abstract behandeldoel beschrijft een " + i("gedragsdomein") + " (bijv. "
                "Slaapkwaliteit, Voedingskwaliteit).",
                "Een sub-behandelingsoptie is de " + i("concrete gedragsoperationalisering") +
                " van dat domein: een specifiek, dagelijks meetbaar item dat het domein "
                "direct meet (bijv. 'schermvrij interval voor slapengaan (min)' als sub-optie "
                "van 'Slaapkwaliteit', of 'gezonde maaltijd genuttigd (ja/nee)' als sub-optie "
                "van 'Voedingskwaliteit').",
                "Selecteer per behandeldoel de 2 items die het domein het "
                + i("meest direct en precies") + " meten.",
                "Vermijd items die het domein slechts " + i("zijdelings") + " raken — "
                "zelfs als ze op het eerste zicht verwant lijken.",
            ])
            + p(b("Belangrijk:") + " alle 20 items in de lijst zijn behandelingsoptie-type "
                "EMA-items (modificeerbare gedragingen en strategieën — <em>geen</em> "
                "symptomen). Uw taak is te selecteren welke 6 items het best aansluiten bij "
                "de 3 gestandaardiseerde behandeldoelen, 2 per doel.")
            + p(b("Antwoordformat:") + " vink <strong>exact 6</strong> items aan. "
                "Geen toelichting vereist.")
            + p(i(b("Zie het uitgewerkte voorbeeld op de volgende pagina") + " voor een "
                  "concrete illustratie van de selectielogica."))
            ,
            "#F5F3FF", "#6D28D9", "#6D28D9",
        )
    )


def page_d4_example() -> str:
    body = (
        p(b("Gestandaardiseerde behandeldoelen (abstract, voorbeeld):"))
        + ol(["Slaapkwaliteit", "Lichamelijke activiteit", "Sociaal contact"])
        + hr()
        + p(b("Selectielogica — hoe kiest u de 2 beste items per doel?"))
        + p("Stel: doel 1 = " + i("Slaapkwaliteit") + ". U ziet de volgende opties in de lijst:")
        + ul([
            "&#10003;&nbsp; 'Schermvrij interval direct voor slapengaan (min)' → "
            + b("juist") + " (operationaliseert het domein direct)",
            "&#10003;&nbsp; 'Vaste bedtijd aangehouden (ja/nee)' → "
            + b("juist") + " (complementaire meting: regelmaat)",
            "&#10007;&nbsp; 'Cafeïne-inname na 15.00 uur (aantal)' → "
            + b("niet juist") + " (raakt slaap slechts zijdelings, hoort niet bij Slaapkwaliteit)",
            "&#10007;&nbsp; 'Maaltijd op vaste tijden genuttigd (ja/nee)' → "
            + b("niet juist") + " (hoort bij Voedingskwaliteit, niet bij Slaapkwaliteit)",
        ])
        + hr()
        + p(b("Correct ingevuld (6 items totaal voor de 3 doelen):"))
        + ul([
            "Doel 1 → 'Schermvrij interval voor slapengaan (min)' + "
            "'Vaste bedtijd aangehouden (ja/nee)'",
            "Doel 2 → 'Duur actieve beweging vandaag (min)' + "
            "'Beweegactiviteit uitgevoerd (ja/nee)'",
            "Doel 3 → 'Bewust sociaal contact gezocht vandaag (ja/nee)' + "
            "'Sociale activiteit bijgewoond of gepland (ja/nee)'",
        ])
    )
    return purple_box("Deel 4: Verfijnd observatiemodel",
                      "Hoe selecteert u de meest passende EMA-items per behandeldoel?",
                      body)


def page_d4_case(case: CaseSurvey) -> str:
    return (
        green_box(f"Gestandaardiseerde behandeldoelen (abstract) voor Deel 4 — {case.case_code}",
                  ol([f"<strong>{esc(t)}</strong>" for t in case.part4_targets]))
        + p(b("Opdracht:") + " selecteer <strong>exact 6 EMA-items</strong> "
            "(2 per behandeldoel) die het best aansluiten als sub-behandelingsopties. "
            "Alle items zijn dagelijkse mobiele EMA-items van het type behandelingsoptie "
            "(<em>geen</em> symptomen). Het systeem vereist exact 6 selecties.")
    )


# ─────────────────────────────────────────────────────────────────────────────
# DEEL 5 — Gepersonaliseerde mobiele coachingsboodschap
# ─────────────────────────────────────────────────────────────────────────────

def page_d5_instr() -> str:
    return (
        h2("Deel 5 — Gepersonaliseerde mobiele coachingsboodschap (HAPA-kader)",
           color="#1E3A5F")
        + green_box("Instructies Deel 5",
            p(b("Opdracht:") + " schrijf een korte, rechtstreeks tot de persoon gerichte "
              "coachingsboodschap die in de mobiele applicatie verschijnt.")
            + p(b("Theoretisch kader — HAPA (Health Action Process Approach; "
                  "Schwarzer, 1992):") + " PHOENIX gebruikt dit kader om de boodschap af "
                "te stemmen op de motivationele fase van de persoon:")
            + ul([
                b("Pre-intentionele fase:") + " de persoon is nog niet gemotiveerd om te "
                "veranderen → focus op " + i("risicobewustzijn en uitkomstverwachting") + ".",
                b("Intentionele fase:") + " de persoon wil veranderen maar heeft nog geen "
                "concreet plan → focus op " + i("doelstelling en actieplanning") + ".",
                b("Actie-/onderhoudsfase:") + " de persoon probeert al te veranderen → "
                "focus op " + i("copingplanning en zelfeffectiviteitsondersteuning") + ".",
            ])
            + p("U hoeft de fase niet expliciet te benoemen; gebruik het kader om de "
                "toon en inhoud van uw boodschap te sturen.")
            + p(b("De boodschap voldoet aan:"))
            + ul([
                b("Lengte:") + " 2–4 zinnen, compact genoeg voor een mobiel scherm.",
                b("Toon:") + " warm, direct, professioneel — geen klinisch jargon of "
                "diagnostische labels.",
                b("Inhoud:") + " adresseert het primaire behandeldoel en de voornaamste "
                "barrière; bevat een concrete, eerstvolgende actie.",
                b("Perspectief:") + " tweede persoon ('jij' informeel of 'u' formeel).",
            ])
            + p(i(b("Zie het uitgewerkte voorbeeld op de volgende pagina") + " voor een "
                  "volledige illustratie."))
        )
    )


def page_d5_example() -> str:
    body = (
        p(b("Context (voorbeeld):"))
        + ul([
            "Primair behandeldoel: avondschermgebruik verminderen",
            "Voornaamste barrière: gewoontekracht — schermgebruik is de enige manier om "
            "na het werk te ontspannen (lage zelfeffectiviteit)",
            "HAPA-fase: intentioneel (wil veranderen, maar heeft nog geen concreet plan)",
        ])
        + hr()
        + p(b("Voorbeeldboodschap:") + "<br>" +
            i('"Je weet dat je avondscherm je slaap in de weg staat, maar het voelt nog '
              'als de makkelijkste manier om te ontspannen na een drukke dag. Leg vanavond '
              'je telefoon om 21.30 uur in een andere kamer en vul die laatste twintig '
              'minuten met iets anders — een boek, muziek, of gewoon even niets. Eén avond '
              'is genoeg om te merken dat het kan."'))
        + hr()
        + p(b("Waarom werkt deze boodschap?") + " De boodschap erkent de barrière "
            "(gewoontekracht), geeft een concrete actie met tijdstip (21.30u), verlaagt de "
            "drempel ('gewoon even niets') en vergroot de zelfeffectiviteit "
            "('één avond is genoeg').")
    )
    return purple_box("Deel 5: Mobiele coachingsboodschap",
                      "Hoe formuleert u een effectieve coachingsboodschap?", body)


def page_d5_case(case: CaseSurvey) -> str:
    return green_box(
        f"Gestandaardiseerde context — {case.case_code}",
        meta_table([
            ("Primair probleem", esc(case.part5_primary_problem)),
            ("Behandeldoel", esc(case.part5_target)),
            ("Voornaamste barrière", esc(case.part5_barrier)),
            ("Copingstrategie", esc(case.part5_coping)),
        ]),
    )


# ─────────────────────────────────────────────────────────────────────────────
# DEEL 6 — Reflectie op digitale benaderingslogica
# ─────────────────────────────────────────────────────────────────────────────

def page_d6_context() -> str:
    return (
        h2("Deel 6 — Reflectie op digitale benaderingslogica", color="#1E3A5F")
        + amber_box("Instructies Deel 6 — Drie reflectievragen",
            p(b("Context — de PHOENIX-benaderingslogica:") + " PHOENIX verwerkt een vrije "
              "klachttekst via vijf opeenvolgende stappen: "
              "(1) identificatie van klacht- en toestandsdimensies als meetbare "
              + i("symptomen") + ", "
              "(2) generatie van modificeerbare gedragsvariabelen "
              "(" + i("behandelingsopties") + "), "
              "(3) prioritering op basis van een tweedelig netwerk dat op dagelijkse "
              "EMA-data wordt gefit, "
              "(4) selectie van concrete EMA-meetitems per behandeldoel, en "
              "(5) formulering van een gepersonaliseerde dagelijkse coachingsboodschap.")
            + p("De kern van deze aanpak is " + b("symptoomgedreven netwerkanalyse") +
                ": klachten worden niet als uitingen van een vaste stoornis beschouwd, maar "
                "als dynamische knooppunten in een netwerk van onderling samenhangende "
                "toestandsdimensies. Behandelprioriteiten worden " + i("niet") +
                " klinisch verondersteld, maar " + i("empirisch") + " bepaald op basis "
                "van de gemeten verbanden tussen dagelijks gedrag (behandelingsopties) en "
                "klachtniveaus (symptomen) over de tijd.")
            + hr()
            + p(b("Hypothetische situatie:") + " Stel u voor dat u als zorgprofessional "
                "een vrije klachttekst ontvangt van een persoon met psychische klachten. "
                "U beschikt " + i("uitsluitend") + " over twee middelen:")
            + ul([
                "Een <strong>mobiele applicatie</strong> waarmee de persoon elke dag een "
                "korte set vragen beantwoordt (bijv. ja/nee, 0–10 schaal, aantal, minuten) "
                "— een EMA-design met dagelijkse datacollectie.",
                "Een <strong>dagelijks digitaal bericht</strong> dat u — of een "
                "geautomatiseerd systeem — naar de persoon stuurt (zoals u in Deel 5 "
                "formuleerde).",
            ])
            + p("U heeft " + i("geen") + " directe consulten, telefoongesprekken of "
                "andere contactvormen ter beschikking. " + i("Alleen") + " dagelijkse "
                "datacollectie via de app en een dagelijkse geschreven boodschap.")
            + hr()
            + p(i("Er is geen enkel correct antwoord. Dit deel dient uitsluitend om uw "
                  "klinische benaderingslogica in kaart te brengen, zodat die naast die "
                  "van PHOENIX kan worden geplaatst. Neem voldoende ruimte om uw "
                  "redenering volledig en concreet te beschrijven. De drie vragen staan "
                  "elk op een afzonderlijke pagina."))
        )
    )


def q_d6_a(case: CaseSurvey) -> str:
    return amber_box(
        "Reflectievraag a",
        p("Welke informatie zou u via de dagelijkse EMA-vragen willen verzamelen, en hoe "
          "zou u die selectie bepalen op basis van de vrije klachttekst?")
        + p(i("Neem voldoende ruimte om uw redenering volledig en concreet te beschrijven."))
    )


def q_d6_b(case: CaseSurvey) -> str:
    return amber_box(
        "Reflectievraag b",
        p("Hoe zou u de dagelijkse boodschappen inhoudelijk afstemmen op wat u via de "
          "EMA-data geleidelijk leert over de toestand en het gedrag van de persoon?")
        + p(i("Neem voldoende ruimte om uw redenering volledig en concreet te beschrijven."))
    )


def q_d6_c(case: CaseSurvey) -> str:
    return amber_box(
        "Reflectievraag c",
        p("Zou u uw digitale aanpak systematisch bijsturen naarmate meer data beschikbaar "
          "wordt? Zo ja — op basis van welke criteria en via welk redeneerproces?")
        + p(i("Neem voldoende ruimte om uw redenering volledig en concreet te beschrijven."))
    )


# ─────────────────────────────────────────────────────────────────────────────
# AFRONDING
# ─────────────────────────────────────────────────────────────────────────────

def page_closing(case: CaseSurvey) -> str:
    return (
        h2("Afronding en terugbezorging")
        + p("Dank u voor het invullen van alle zes delen voor uw toegewezen casus "
            "en de afsluitende reflectievragen.")
        + gray_box("Checklist voor terugbezorging",
            ul([
                "&#9744;&nbsp; Ik heb in Deel 1 symptoomlabels ingevuld.",
                "&#9744;&nbsp; Ik heb in Deel 2 behandelingsoptielabels ingevuld.",
                "&#9744;&nbsp; Ik heb in Deel 3 alle 5 behandelingsopties gerangschikt.",
                "&#9744;&nbsp; Ik heb in Deel 4 exact 6 EMA-items geselecteerd.",
                "&#9744;&nbsp; Ik heb in Deel 5 een mobiele coachingsboodschap geschreven.",
                "&#9744;&nbsp; Ik heb in Deel 6 de drie reflectievragen beantwoord.",
                "&#9744;&nbsp; Mijn antwoorden weerspiegelen mijn eigen klinische "
                "oordeelsvorming zonder gebruik van generatieve AI of andere externe hulp.",
                "&#9744;&nbsp; Ik begrijp dat mijn antwoorden geanonimiseerd worden "
                "voor analyse.",
            ])
        )
        + blue_box("Uw antwoorden zijn opgeslagen",
            p("Na het indienen van deze survey worden uw antwoorden automatisch opgeslagen "
              "in het systeem. U hoeft niets terug te mailen.")
            + p("Na ontvangst worden uw antwoorden geanonimiseerd en opgenomen in het "
                "expertreferentiecorpus voor de latere blinde evaluatie van PHOENIX.")
            + p(b("Hartelijk dank voor uw waardevolle bijdrage aan dit onderzoek."))
            + p("Vragen of opmerkingen? Neem contact op via: "
                "<strong>stijn.vanseveren@ugent.be</strong>")
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# SURVEY ASSEMBLER
# ─────────────────────────────────────────────────────────────────────────────

def build_survey(case: CaseSurvey) -> dict:
    survey_id = gen_id("SV_", f"PHOENIX_PRE_{case.hcp_code}_{case.case_code}")
    survey_name = (f"PHOENIX PRE — {case.hcp_code} — {case.case_code} "
                   f"({case.profile})")

    print(f"  Rendering network figures ...")
    example_b64 = network_to_b64(case.example_network)
    case_b64    = network_to_b64(case.network)

    q = QSFBuilder(survey_id, survey_name)

    # ── Block 1: Titelpagina ─────────────────────────────────────────────────
    q.block("Titelpagina")
    q.db(page_title(case), tag="TITELPAGINA")
    q.page_break()

    # ── Block 2: Toestemming ─────────────────────────────────────────────────
    q.block("Toestemming")
    q.db(page_consent(case), tag="CONSENT_INFO")
    q.mc_single(
        p(b("Gaat u akkoord met deelname aan dit onderzoek?") + "<br>"
          "<em>Kies één optie. Indien u kiest voor optie 2 sluit de survey automatisch af.</em>"),
        choices=[
            "Ja — ik heb de informatie gelezen en geef geinformeerde toestemming "
            "om deel te nemen.",
            "Nee — ik wens niet deel te nemen.",
        ],
        tag="CONSENT_DECISION",
        skip_end_if=2,
        force=True,
    )
    q.page_break()

    # ── Block 3: Instructiepagina ────────────────────────────────────────────
    q.block("Instructiepagina")
    q.db(page_instr_p1(), tag="INSTR_P1")
    q.page_break()
    q.db(page_instr_p2(), tag="INSTR_P2")
    q.page_break()

    # ── Block 4: Intake ──────────────────────────────────────────────────────
    q.block("Intake")
    q.db(page_intake_intro(), tag="INTAKE_INTRO")
    q.te_essay(
        p(b("Vraag 1 — Algemene achtergrond en klinische ervaring") + "<br><br>"
          "Wat is uw professionele achtergrond en ervaring in de klinische praktijk? "
          "Beschrijf kort uw "
          + b("discipline") + " (bijv. klinische psychologie, psychiatrie, psychotherapie), "
          + b("functie") + " (bijv. psycholoog, psychiater, GGZ-verpleegkundige), "
          + b("aantal jaren klinische ervaring") + ", en de mate waarin u momenteel "
          + b("werkt met personen met psychische klachten") + "."),
        tag="INTAKE_BACKGROUND",
    )
    q.page_break()
    q.matrix(
        p(b("Vraag 2 — Houding tegenover agentic AI in de klinische context") + "<br><br>"
          "Geef aan in welke mate u akkoord gaat met onderstaande uitspraken. "
          "Gebruik de 7-punt schaal van <em>Helemaal oneens</em> tot "
          "<em>Helemaal eens</em>. Er zijn geen goede of foute antwoorden."),
        statements=[
            "Ik zie agentic AI als een potentieel zinvolle ondersteuning voor klinische "
            "besluitvorming in mentale gezondheidszorg.",
            "Ik zou agentic AI enkel vertrouwen wanneer de onderliggende redeneerstappen "
            "expliciet en controleerbaar worden getoond.",
            "Ik verwacht dat agentic AI binnen afzienbare tijd bruikbaar kan zijn voor "
            "gepersonaliseerde digitale geestelijke gezondheidszorg.",
        ],
        scale=SCALE_7PT,
        tag="INTAKE_AGENTIC_AI",
    )
    q.page_break()
    q.matrix(
        p(b("Vraag 3 — Kennis van en houding tegenover netwerk-analytisch denken") + "<br><br>"
          "Geef aan in welke mate u akkoord gaat met onderstaande uitspraken. "
          "Gebruik de 7-punt schaal van <em>Helemaal oneens</em> tot "
          "<em>Helemaal eens</em>."),
        statements=[
            "Ik ben voldoende vertrouwd met het netwerk-analytisch denken om de logica "
            "van de huidige surveyopgaven goed te kunnen volgen.",
            "Het in kaart brengen van dynamische relaties tussen klachten en modificeerbaar "
            "gedrag kan klinisch relevante behandelprioriteiten zichtbaar maken.",
            "Ik vind een netwerk-analytische benadering methodologisch bruikbaar als basis "
            "voor gepersonaliseerde EMA-monitoring en digitale interventies.",
        ],
        scale=SCALE_7PT,
        tag="INTAKE_NETWORK_ANALYSIS",
    )
    q.page_break()

    # ── Block 5: Deel 1 ──────────────────────────────────────────────────────
    q.block("Deel 1 — Operationalisering")
    q.db(page_d1_instr(), tag="D1_INSTR")
    q.page_break()
    q.db(page_d1_example(), tag="D1_VOORBEELD")
    q.page_break()
    q.db(page_d1_case(case), tag=f"{case.case_code}_D1_CASUS")
    q.te_form(
        p(b(f"Uw antwoord — Deel 1 ({case.case_code})") + "<br>"
          "Noteer <strong>2–6 symptoomlabels</strong> (2–5 woorden per label; "
          "geen beschrijvende zin; laat ongebruikte velden leeg)."),
        fields=[f"Symptoom {i}" for i in range(1, 7)],
        tag=f"{case.case_code}_D1_ANTWOORD",
    )
    q.page_break()

    # ── Block 6: Deel 2 ──────────────────────────────────────────────────────
    q.block("Deel 2 — Initieel observatiemodel")
    q.db(page_d2_instr(), tag="D2_INSTR")
    q.page_break()
    q.db(page_d2_example(), tag="D2_VOORBEELD")
    q.page_break()
    q.db(page_d2_case(case), tag=f"{case.case_code}_D2_CASUS")
    q.te_form(
        p(b(f"Uw antwoord — Deel 2 ({case.case_code})") + "<br>"
          "Noteer <strong>3–5 behandelingsoptielabels</strong> (2–6 woorden per label; "
          "laat ongebruikte velden leeg)."),
        fields=[f"Behandelingsoptie {i}" for i in range(1, 6)],
        tag=f"{case.case_code}_D2_ANTWOORD",
    )
    q.page_break()

    # ── Block 7: Deel 3 ──────────────────────────────────────────────────────
    q.block("Deel 3 — Behandeldoelprioritering")
    q.db(page_d3_instr(), tag="D3_INSTR")
    q.page_break()
    q.db(page_d3_example(example_b64), tag="D3_VOORBEELD")
    q.page_break()
    q.db(page_d3_case(case, case_b64), tag=f"{case.case_code}_D3_CASUS")
    q.rank_order(
        p(b(f"Uw antwoord — Deel 3 ({case.case_code})") + "<br>"
          + "Rangschik alle 5 behandelingsopties van hoogste (positie 1) naar laagste "
          + "(positie 5) behandelprioriteit.<br>"
          + i("Sleep de opties in de gewenste volgorde, of typ rangorde 1–5 in de velden.")),
        choices=case.part3_options,
        tag=f"{case.case_code}_D3_ANTWOORD",
    )
    q.page_break()

    # ── Block 8: Deel 4 ──────────────────────────────────────────────────────
    q.block("Deel 4 — Verfijnd observatiemodel")
    q.db(page_d4_instr(), tag="D4_INSTR")
    q.page_break()
    q.db(page_d4_example(), tag="D4_VOORBEELD")
    q.page_break()
    q.db(page_d4_case(case), tag=f"{case.case_code}_D4_CASUS")
    q.mc_multi(
        p(b(f"Uw antwoord — Deel 4 ({case.case_code})") + "<br>"
          "Selecteer <strong>exact 6</strong> EMA-items (2 per behandeldoel). "
          "Alle items zijn van het type behandelingsoptie (modificeerbaar gedrag, "
          "<em>geen</em> symptomen). Het systeem vereist exact 6 aangevinkte items."),
        choices=[f"{i}.&nbsp; {item}"
                 for i, item in enumerate(case.part4_items, start=1)],
        tag=f"{case.case_code}_D4_ANTWOORD",
        min_c=6, max_c=6,
    )
    q.page_break()

    # ── Block 9: Deel 5 ──────────────────────────────────────────────────────
    q.block("Deel 5 — Mobiele coachingsboodschap")
    q.db(page_d5_instr(), tag="D5_INSTR")
    q.page_break()
    q.db(page_d5_example(), tag="D5_VOORBEELD")
    q.page_break()
    q.db(page_d5_case(case), tag=f"{case.case_code}_D5_CASUS")
    q.te_essay(
        p(b(f"Uw antwoord — Deel 5 ({case.case_code})") + "<br>"
          "Schrijf hieronder de mobiele coachingsboodschap voor deze casus. "
          "Formuleer alsof de tekst morgen rechtstreeks op de smartphone van de "
          "persoon verschijnt (2–4 zinnen, warm, direct, concreet)."),
        tag=f"{case.case_code}_D5_ANTWOORD",
    )
    q.page_break()

    # ── Block 10: Deel 6 ─────────────────────────────────────────────────────
    q.block("Deel 6 — Reflectie op digitale benaderingslogica")
    q.db(page_d6_context(), tag="D6_CONTEXT")
    q.page_break()
    q.te_essay(q_d6_a(case), tag=f"{case.case_code}_D6_A", force=False)
    q.page_break()
    q.te_essay(q_d6_b(case), tag=f"{case.case_code}_D6_B", force=False)
    q.page_break()
    q.te_essay(q_d6_c(case), tag=f"{case.case_code}_D6_C", force=False)
    q.page_break()

    # ── Block 11: Afronding ──────────────────────────────────────────────────
    q.block("Afronding")
    q.db(page_closing(case), tag="AFRONDING")

    return q.build()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    QSF_ROOT.mkdir(parents=True, exist_ok=True)

    print("Reading shared template from HCP_01/main.tex ...")
    template_tex = (HCP_DIRS[0] / "main.tex").read_text(encoding="utf-8")

    p3_shared = extract_between(
        template_tex,
        r"\\section\{Deel 3:.*?\}",
        r"\\subsection\*\{Casus 1:",
    )
    example_tikz = extract_first(
        r"\\begin\{voorbeeldbox\}\{Deel 3:.*?\\begin\{tikzpicture\}"
        r"(.*?)\\end\{tikzpicture\}.*?\\end\{voorbeeldbox\}",
        p3_shared,
    )
    example_network = parse_network(example_tikz, "Behandelingsopties", "Symptomen",
                                    "shared_example")

    print(f"Generating {len(HCP_DIRS)} QSF files ...\n")
    for hcp_dir in HCP_DIRS:
        tex = (hcp_dir / "main.tex").read_text(encoding="utf-8")
        case = parse_case_survey(tex, example_network)
        print(f"Building {case.hcp_code} / {case.case_code} ({case.profile}) ...")
        qsf_data = build_survey(case)
        out = QSF_ROOT / f"PHOENIX_PRE_{case.hcp_code}_{case.case_code}.qsf"
        out.write_text(json.dumps(qsf_data, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        n_q = len([e for e in qsf_data["SurveyElements"] if e["Element"] == "SQ"])
        n_b = len(qsf_data["SurveyElements"][0]["Payload"])
        print(f"  ✓ {out.name}  ({out.stat().st_size // 1024} KB, "
              f"{n_b} blocks, {n_q} questions)\n")

    print(f"Done. QSF files are in:\n  {QSF_ROOT}\n")
    print("How to import into Qualtrics:")
    print("  1. Qualtrics → Survey tab → Tools → Import/Export → Import survey")
    print("  2. Choose File → select the .qsf → category: Research Core → Import")
    print("  3. The full survey (all blocks, figures, logic) is created automatically.")
    print("  4. Activate and generate personal links for each HCP.")


if __name__ == "__main__":
    main()
