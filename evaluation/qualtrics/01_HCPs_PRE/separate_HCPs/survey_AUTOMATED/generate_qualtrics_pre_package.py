#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import html
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT.parent / "1_case_per_HCP"
GENERATED_ROOT = ROOT / "generated"
IMPORT_ROOT = GENERATED_ROOT / "qualtrics_advanced_txt"
MANIFEST_ROOT = GENERATED_ROOT / "manifests"
ASSET_ROOT = ROOT / "assets" / "networks"
BUNDLE_PATH = GENERATED_ROOT / "qualtrics_upload_bundle.zip"

HCP_DIRS = [SOURCE_ROOT / f"HCP_{idx:02d}" for idx in range(1, 11)]

SCALE_7PT = [
    "1 = Helemaal oneens",
    "2 = Oneens",
    "3 = Eerder oneens",
    "4 = Neutraal",
    "5 = Eerder eens",
    "6 = Eens",
    "7 = Helemaal eens",
]


@dataclass
class Node:
    key: str
    prefix: str
    label: str
    x: float
    y: float
    kind: str
    compact: bool


@dataclass
class Edge:
    src: str
    dst: str
    width: float
    sign: str


@dataclass
class NetworkFigure:
    title: str
    left_header: str
    right_header: str
    nodes: list[Node]
    edges: list[Edge]


@dataclass
class CaseSurvey:
    hcp_code: str
    case_code: str
    profile: str
    duration: str
    complaint_vignette: str
    short_summary: str
    part2_symptoms: list[str]
    monitoring: str
    part3_options: list[str]
    part4_targets: list[str]
    part4_items: list[str]
    part5_primary_problem: str
    part5_target: str
    part5_barrier: str
    part5_coping: str
    network: NetworkFigure


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def ensure_dirs() -> None:
    for path in (GENERATED_ROOT, IMPORT_ROOT, MANIFEST_ROOT, ASSET_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def latex_to_text(text: str) -> str:
    subs = [
        (r"(?m)^\s*%.*$", ""),
        (r"\\section\{([^}]*)\}", r"\n\1\n"),
        (r"\\subsection\*\{([^}]*)\}", r"\n\1\n"),
        (r"\\begin\{voorbeeldbox\}\{([^}]*)\}\{([^}]*)\}", r"\n\1\n\2\n"),
        (r"\\begin\{(?:tcolorbox|instrbox|responsebox|complaintbox|contextbox|monitorbox)\}(?:\[[^\]]*\])?", "\n"),
        (r"\\end\{(?:tcolorbox|instrbox|responsebox|complaintbox|contextbox|monitorbox)\}", "\n"),
        (r"\\begin\{itemize\}(?:\[[^\]]*\])?", "\n"),
        (r"\\end\{itemize\}", "\n"),
        (r"\\begin\{enumerate\}(?:\[[^\]]*\])?", "\n"),
        (r"\\end\{enumerate\}", "\n"),
        (r"\\begin\{center\}", "\n"),
        (r"\\end\{center\}", "\n"),
        (r"\\item\s*", "- "),
        (r"\\begin\{tabular\}(?:\[[^\]]*\])?\{.*?\}", "\n"),
        (r"\\begin\{tabularx\}(?:\{.*?\}){2}", "\n"),
        (r"\\end\{tabular\}", "\n"),
        (r"\\end\{tabularx\}", "\n"),
        (r"\\toprule", "\n"),
        (r"\\midrule", "\n"),
        (r"\\bottomrule", "\n"),
        (r"\\medskip", "\n"),
        (r"\\smallskip", "\n"),
        (r"\\bigskip", "\n"),
        (r"\\vspace\{[^}]*\}", "\n"),
        (r"\\nopagebreak\[[^\]]*\]", ""),
        (r"\\noindent", ""),
        (r"\\quad", " "),
        (r"\\enspace", " "),
        (r"\\hfill", " "),
        (r"\\\\(\[[^\]]*\])?", "\n"),
        (r"\\textbf\{([^}]*)\}", r"\1"),
        (r"\\textit\{([^}]*)\}", r"\1"),
        (r"\\emph\{([^}]*)\}", r"\1"),
        (r"\\texttt\{([^}]*)\}", r"\1"),
        (r"\\small", ""),
        (r"\\large", ""),
        (r"\\bfseries", ""),
        (r"\\itshape", ""),
        (r"\\renewcommand\{[^}]*\}\{[^}]*\}", ""),
        (r"\\color\{[^}]*\}", ""),
        (r"\\rule\{[^}]*\}\{[^}]*\}", ""),
        (r"\\today", ""),
        (r"\\checkmark", "✓"),
        (r"\\square", "☐"),
        (r"\\%", "%"),
        (r"\\&", "&"),
        (r"\\_", "_"),
        (r"\\,", ""),
        (r"\\\.", "."),
        (r"\\approx", "≈"),
        (r"\\rightarrow", "→"),
        (r"\$\\times\$", "×"),
        (r"\$\\rightarrow\$", "→"),
        (r"\$\\square\$", "☐"),
        (r"\$\\approx\\,\$", "≈ "),
        (r"\$\\approx\$", "≈"),
        (r"``", '"'),
        (r"''", '"'),
        (r"---", "-"),
        (r"--", "-"),
        (r"\\[a-zA-Z]+\*?\{([^}]*)\}", r"\1"),
        (r"\\[a-zA-Z]+\*?", ""),
        (r"[\{\}]", ""),
        (r"\$", ""),
        (r"\s&\s", ": "),
        (r"~", " "),
    ]
    out = text
    for pattern, replacement in subs:
        out = re.sub(pattern, replacement, out, flags=re.DOTALL)
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def extract_between(text: str, start: str, end: str) -> str:
    match = re.search(start + r"(.*?)" + end, text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not extract text between {start!r} and {end!r}")
    return match.group(1)


def extract_first(pattern: str, text: str) -> str:
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Pattern not found: {pattern}")
    return match.group(1).strip()


def strip_comments(text: str) -> str:
    return re.sub(r"(?m)^\s*%.*$", "", text)


def qtext(text: str) -> str:
    cleaned = latex_to_text(text)
    return cleaned.replace("\n\n", "\n")


def render_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in qtext(text).splitlines():
        line = raw_line.strip().replace("\\", "")
        if not line:
            continue
        if line in {"center", "[4]"}:
            continue
        if line.startswith("tabular"):
            continue
        if re.fullmatch(r"[@>{}p0-9.X\\~,:-]+", line):
            continue
        lines.append(line)
    return lines


def parse_bullets(block: str) -> list[str]:
    items = re.findall(r"\\item\s+(.*?)(?=(?:\\item|\\end\{itemize\}|\\end\{enumerate\}))", block, flags=re.DOTALL)
    return [latex_to_text(item).strip() for item in items]


def parse_prefixed_item_lines(block: str) -> list[str]:
    items = re.findall(r"\\item\s+\\textbf\{[^}]*\}\s*(.*?)\s*(?=(?:\\item|$))", block, flags=re.DOTALL)
    return [latex_to_text(item).strip() for item in items]


def parse_enumerated_targets(block: str) -> list[str]:
    items = re.findall(r"\\item\s+\\textbf\{([^}]*)\}", block, flags=re.DOTALL)
    if not items:
        items = re.findall(r"\\item\s+([^\\\n]+)", block, flags=re.DOTALL)
    return [latex_to_text(item).strip() for item in items]


def parse_table_value(label: str, block: str) -> str:
    pattern = rf"{re.escape(label)}\s*&\s*(.*?)\s*\\\\"
    return latex_to_text(extract_first(pattern, block))


def parse_network(network_tex: str, left_header: str, right_header: str, title: str) -> NetworkFigure:
    nodes: list[Node] = []
    edges: list[Edge] = []

    node_pattern = re.compile(
        r"\\node\[(?P<style>[a-zA-Z0-9]+)\]\s+\((?P<key>[a-zA-Z0-9]+)\)\s+at\s+\((?P<x>-?[0-9.]+),\s*(?P<y>-?[0-9.]+)\)\s+\{(?P<body>.*?)\};",
        re.DOTALL,
    )
    edge_pattern = re.compile(
        r"\\draw\[line width=(?P<width>[0-9.]+)pt,\s*draw=(?P<color>[A-Za-z0-9!]+).*?\]\s+\((?P<src>[a-zA-Z0-9]+)\.east\)\s+--\s+\((?P<dst>[a-zA-Z0-9]+)\.west\);"
    )

    for match in node_pattern.finditer(network_tex):
        body = match.group("body").split("\\\\", 1)
        prefix = latex_to_text(body[0]).strip()
        label = latex_to_text(body[1] if len(body) > 1 else "")
        style = match.group("style")
        key = match.group("key")
        x = float(match.group("x"))
        y = float(match.group("y"))
        kind = "left" if key.lower().startswith(("p", "bo")) else "right"
        compact = style.startswith("sm")
        nodes.append(Node(key=key, prefix=prefix, label=label, x=x, y=y, kind=kind, compact=compact))

    known_keys = {node.key for node in nodes}
    for match in edge_pattern.finditer(network_tex):
        src = match.group("src")
        dst = match.group("dst")
        if src not in known_keys or dst not in known_keys:
            continue
        color = match.group("color")
        sign = "positive" if "PrimaryBlue" in color else "negative"
        edges.append(Edge(src=src, dst=dst, width=float(match.group("width")), sign=sign))

    if not nodes or not edges:
        raise ValueError("Could not parse network figure from tikz source")

    return NetworkFigure(title=title, left_header=left_header, right_header=right_header, nodes=nodes, edges=edges)


def draw_network_figure(figure: NetworkFigure, out_path: Path) -> None:
    compact = any(node.compact for node in figure.nodes)
    fig_w, fig_h = (7.4, 4.8) if compact else (8.5, 6.0)
    left_w = 2.95 if compact else 3.45
    right_w = 2.95 if compact else 3.45
    node_h = 0.68 if compact else 0.76
    font_prefix = 6.1 if compact else 6.4
    font_label = 5.9 if compact else 6.0

    min_x = min(node.x for node in figure.nodes)
    max_x = max(node.x for node in figure.nodes)
    min_y = min(node.y for node in figure.nodes)
    max_y = max(node.y for node in figure.nodes)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    bg = "#F8FAFC"
    border = "#CBD5E1"
    ax.set_facecolor(bg)
    fig.patch.set_facecolor(bg)
    ax.axis("off")

    x_pad = 2.3
    y_pad_top = 1.2
    y_pad_bottom = 2.2
    ax.set_xlim(min_x - x_pad, max_x + x_pad)
    ax.set_ylim(min_y - y_pad_bottom, max_y + y_pad_top)

    frame = mpatches.FancyBboxPatch(
        (min_x - x_pad + 0.1, min_y - y_pad_bottom + 0.1),
        (max_x - min_x) + (2 * x_pad) - 0.2,
        (max_y - min_y) + y_pad_top + y_pad_bottom - 0.2,
        boxstyle="round,pad=0.02",
        linewidth=0.8,
        edgecolor=border,
        facecolor=bg,
        zorder=0,
    )
    ax.add_patch(frame)

    ax.text(
        min(node.x for node in figure.nodes if node.kind == "left"),
        max_y + 0.62,
        figure.left_header,
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color="#047857",
        style="italic",
    )
    ax.text(
        max(node.x for node in figure.nodes if node.kind == "right"),
        max_y + 0.62,
        figure.right_header,
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color="#1E3A5F",
        style="italic",
    )

    node_lookup = {node.key: node for node in figure.nodes}
    for edge in figure.edges:
        src = node_lookup[edge.src]
        dst = node_lookup[edge.dst]
        color = "#1D4ED8" if edge.sign == "positive" else "#B91C1C"
        ax.plot(
            [src.x + 0.38, dst.x - 0.38],
            [src.y, dst.y],
            color=color,
            linewidth=edge.width,
            alpha=0.86,
            zorder=1,
            solid_capstyle="round",
        )

    for node in figure.nodes:
        face = "#CCFBF1" if node.kind == "left" else "#DBEAFE"
        edge_color = "#047857" if node.kind == "left" else "#1D4ED8"
        text_color = "#047857" if node.kind == "left" else "#1E3A5F"
        width = left_w if node.kind == "left" else right_w
        rect = mpatches.FancyBboxPatch(
            (node.x - width / 2, node.y - node_h / 2),
            width,
            node_h,
            boxstyle="round,pad=0.05",
            linewidth=1.1,
            edgecolor=edge_color,
            facecolor=face,
            zorder=2,
        )
        ax.add_patch(rect)
        ax.text(
            node.x,
            node.y + 0.10,
            node.prefix,
            ha="center",
            va="center",
            fontsize=font_prefix,
            fontweight="bold",
            color=text_color,
            zorder=3,
        )
        ax.text(
            node.x,
            node.y - 0.12,
            node.label,
            ha="center",
            va="center",
            fontsize=font_label,
            color=text_color,
            zorder=3,
            wrap=True,
        )

    legend_y = min_y - 1.0
    ax.axhline(legend_y + 0.52, xmin=0.05, xmax=0.95, color=border, linewidth=0.8)
    ax.plot([min_x - 0.6, min_x + 0.2], [legend_y, legend_y], color="#1D4ED8", linewidth=2.4, solid_capstyle="round")
    ax.text(min_x + 0.45, legend_y, "Blauw = positief verband (BO vergroot S)", ha="left", va="center", fontsize=6.1, color="#1E293B")
    ax.plot([min_x - 0.6, min_x + 0.2], [legend_y - 0.42, legend_y - 0.42], color="#B91C1C", linewidth=2.4, solid_capstyle="round")
    ax.text(min_x + 0.45, legend_y - 0.42, "Rood = negatief verband (BO verkleint S)", ha="left", va="center", fontsize=6.1, color="#1E293B")
    ax.text(
        (min_x + max_x) / 2,
        legend_y - 0.92,
        "Lijndikte proportioneel aan |w|, genormaliseerd binnen netwerk (bereik 1-5 pt).",
        ha="center",
        va="center",
        fontsize=5.6,
        color="#64748B",
        style="italic",
    )

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=bg, edgecolor="none")
    plt.close(fig)


def build_shared_content(template_tex: str) -> dict[str, str]:
    intro_block = extract_between(template_tex, r"\\section\{Instructiepagina\}", r"\\section\{Deel 1:")
    part1_shared = extract_between(template_tex, r"\\section\{Deel 1:.*?\}", r"\\subsection\*\{Casus 1:")
    part2_shared = extract_between(template_tex, r"\\section\{Deel 2:.*?\}", r"\\subsection\*\{Casus 1:")
    part3_shared = extract_between(template_tex, r"\\section\{Deel 3:.*?\}", r"\\subsection\*\{Casus 1:")
    part4_shared = extract_between(template_tex, r"\\section\{Deel 4:.*?\}", r"\\subsection\*\{Casus 1:")
    part5_shared = extract_between(template_tex, r"\\section\{Deel 5:.*?\}", r"\\subsection\*\{Casus 1:")
    part6_full = extract_between(template_tex, r"\\section\{Deel 6:.*?\}", r"\\section\{Afronding")
    closing_block = extract_between(template_tex, r"\\section\{Afronding.*?\}", r"\\end\{document\}")

    part3_example_network = extract_first(
        r"\\begin\{voorbeeldbox\}\{Deel 3:.*?\\begin\{tikzpicture\}(.*?)\\end\{tikzpicture\}.*?\\end\{voorbeeldbox\}",
        part3_shared,
    )

    part3_shared_no_figure = re.sub(
        r"\\begin\{center\}.*?\\end\{center\}",
        "\n[Figuur voorbeeldnetwerk]\n",
        part3_shared,
        count=1,
        flags=re.DOTALL,
    )

    part6_intro = re.sub(
        r"\\begin\{tcolorbox\}\[colback=AccentAmber.*",
        "",
        part6_full,
        flags=re.DOTALL,
    )

    part6_questions = {
        "a": latex_to_text(
            extract_first(
                r"Reflectievraag a\}.*?\n(.*?)\n\\end\{tcolorbox\}",
                part6_full,
            )
        ),
        "b": latex_to_text(
            extract_first(
                r"Reflectievraag b\}.*?\n(.*?)\n\\end\{tcolorbox\}",
                part6_full,
            )
        ),
        "c": latex_to_text(
            extract_first(
                r"Reflectievraag c\}.*?\n(.*?)\n\\end\{tcolorbox\}",
                part6_full,
            )
        ),
    }

    return {
        "intro": latex_to_text(intro_block),
        "part1_shared": latex_to_text(part1_shared),
        "part2_shared": latex_to_text(part2_shared),
        "part3_shared": latex_to_text(part3_shared_no_figure),
        "part4_shared": latex_to_text(part4_shared),
        "part5_shared": latex_to_text(part5_shared),
        "part6_intro": latex_to_text(part6_intro),
        "part6_q_a": part6_questions["a"],
        "part6_q_b": part6_questions["b"],
        "part6_q_c": part6_questions["c"],
        "closing": latex_to_text(closing_block),
        "part3_example_network_tikz": part3_example_network,
    }


def parse_case_survey(tex: str) -> CaseSurvey:
    hcp_code = extract_first(r"Deelnemerscode:\s*\\texttt\{(HCP-PRE-\d+)\}", tex)
    case_code = extract_first(r"\\textbf\{\\color\{PrimaryBlue\}(C\d+)\}", tex)

    part1_case = extract_between(tex, r"\\subsection\*\{Casus 1:.*?\}\n", r"\\section\{Deel 2:")
    part2_case = extract_between(tex, r"\\subsection\*\{Casus 1:.*?-- Deel 2\}", r"\\section\{Deel 3:")
    part3_case = extract_between(tex, r"\\subsection\*\{Casus 1:.*?-- Deel 3\}", r"\\section\{Deel 4:")
    part4_case = extract_between(tex, r"\\subsection\*\{Casus 1:.*?-- Deel 4\}", r"\\section\{Deel 5:")
    part5_case = extract_between(tex, r"\\subsection\*\{Casus 1:.*?-- Deel 5\}", r"\\section\{Deel 6:")

    part1_title = extract_first(r"title=\{Casus 1:\s*(.*?);\s*duur:", part1_case)
    profile = part1_title.strip()
    duration = extract_first(r"title=\{Casus 1:.*?;\s*duur:\s*([^}]*)\}", part1_case)
    complaint_vignette = latex_to_text(extract_first(r"\\begin\{complaintbox\}.*?\\small\s+(.*?)\n\\end\{complaintbox\}", part1_case))

    short_summary = latex_to_text(extract_first(r"\\begin\{complaintbox\}.*?\\small\s+(.*?)\n\\end\{complaintbox\}", part2_case))
    part2_symptoms = parse_prefixed_item_lines(extract_first(r"\\small\\textbf\{Gestandaardiseerde symptomen uit Deel 1:\}(.*?)\\end\{itemize\}", part2_case))

    monitoring = latex_to_text(extract_first(r"\\begin\{monitorbox\}\s*\\small\\textbf\{21-daagse monitoring:\}\s*(.*?)\n\\end\{monitorbox\}", part3_case))
    network_tikz = extract_first(r"\\begin\{tikzpicture\}(.*?)\\end\{tikzpicture\}", part3_case)
    part3_options = [
        f"{node.prefix}: {node.label}"
        for node in parse_network(network_tikz, "Behandelingsopties", "Symptomen", case_code).nodes
        if node.kind == "left"
    ]

    part4_targets = parse_enumerated_targets(extract_first(r"Gestandaardiseerde behandeldoelen \(abstract\) voor Deel 4:\}(.*?)\\end\{enumerate\}", part4_case))
    part4_items = [
        latex_to_text(item)
        for _, item in re.findall(r"\\checkitemBFS\{(\d+)\}\{([^}]*)\}", part4_case)
    ]

    part5_primary_problem = parse_table_value("Primair probleem", part5_case)
    part5_target = parse_table_value("Behandeldoel", part5_case)
    part5_barrier = parse_table_value("Voornaamste barriere", part5_case)
    part5_coping = parse_table_value("Copingstrategie", part5_case)

    network = parse_network(network_tikz, "Behandelingsopties", "Symptomen", case_code)

    return CaseSurvey(
        hcp_code=hcp_code,
        case_code=case_code,
        profile=profile,
        duration=duration,
        complaint_vignette=complaint_vignette,
        short_summary=short_summary,
        part2_symptoms=part2_symptoms,
        monitoring=monitoring,
        part3_options=part3_options,
        part4_targets=part4_targets,
        part4_items=part4_items,
        part5_primary_problem=part5_primary_problem,
        part5_target=part5_target,
        part5_barrier=part5_barrier,
        part5_coping=part5_coping,
        network=network,
    )


def build_intro_block(case: CaseSurvey, shared: dict[str, str]) -> list[str]:
    return [
        f"PHOENIX evaluatiestudie - Fase 1",
        "",
        f"Deelnemerscode: {case.hcp_code}",
        f"Toegewezen casus: {case.case_code} ({case.profile})",
        "",
        "Studie: Evaluatie van de klinische kwaliteit van een ontologiegebaseerd multi-agentsysteem voor gepersonaliseerde digitale geestelijke gezondheidszorg (PHOENIX)",
        "Instelling: Universiteit Gent - Faculteit Psychologie en Pedagogische Wetenschappen",
        "Onderzoeker: Stijn Van Severen (masterproefstudent)",
        "Promotoren: Prof. Dr. Geert Crombez; Dr. Annick De Paepe",
        "Contact: stijn.vanseveren@ugent.be",
        "Geschatte duur: Ongeveer 25-35 minuten voor de casus, plus intake",
        "",
        "U neemt deel aan een evaluatiestudie waarin zorgprofessionals onafhankelijk dezelfde vijf klinische redeneerstappen uitvoeren als het PHOENIX-systeem.",
        "Uw antwoorden vormen het menselijke referentiecorpus voor een latere dubbelblinde vergelijking met systeemoutput.",
        "",
        "Voor uw casus vult u dezelfde vijf inhoudelijke delen in: (1) operationalisering, (2) initieel observatiemodel, (3) prioritering van behandeldoelen, (4) verfijning van EMA-metingen en (5) een mobiele coachingsboodschap.",
        "Aansluitend beantwoordt u in Deel 6 drie reflectievragen over uw digitale benaderingslogica.",
        "",
        "We vragen uw eigen klinische oordeelsvorming. Antwoord zoals u dat in een reele professionele context zou doen, maar werk strikt volgens de instructies op de volgende pagina's.",
        "",
        "Vertrouwelijkheid: uw antwoorden worden voor analyse geanonimiseerd en uitsluitend gebruikt binnen deze masterproefstudie. Deelname is vrijwillig; u kan zich op elk moment terugtrekken door contact op te nemen met de onderzoeker.",
    ]


def build_consent_block(case: CaseSurvey) -> list[str]:
    return [
        "Geinformeerde toestemming",
        "",
        f"U staat op het punt de evaluatiesurvey voor {case.hcp_code} te starten.",
        "",
        "Lees onderstaande informatie zorgvuldig.",
        "",
        "Doel van het onderzoek: deze survey verzamelt onafhankelijke expertantwoorden van zorgprofessionals op dezelfde klinische redeneerstappen die later ook door het PHOENIX-systeem worden uitgevoerd.",
        "Uw antwoorden worden gebruikt als menselijk referentiecorpus voor een dubbelblinde vergelijking met systeemoutput in het kader van een masterproef aan de Universiteit Gent.",
        "",
        "Wat deelname inhoudt:",
        "- U vult eerst een korte intake in en daarna 1 toegewezen casus.",
        "- De survey bevat 6 delen en duurt ongeveer 25-35 minuten.",
        "- We vragen uitsluitend uw eigen klinische oordeel, zonder generatieve AI, collegaoverleg of andere externe hulpmiddelen.",
        "",
        "Vrijwilligheid en vertrouwelijkheid:",
        "- Deelname is vrijwillig.",
        "- U kan op elk moment stoppen door het venster te sluiten.",
        "- Uw antwoorden worden na ontvangst geanonimiseerd en alleen voor deze masterproefanalyse gebruikt.",
        "",
        "Contact:",
        "- Onderzoeker: Stijn Van Severen",
        "- E-mail: stijn.vanseveren@ugent.be",
        "",
        "Kies hieronder of u op basis van deze informatie wenst deel te nemen. Indien u niet wenst deel te nemen, sluit dan de survey na uw keuze.",
    ]


def build_intake_intro() -> list[str]:
    return [
        "Intake - achtergrond, kennis van netwerkanalyse en houding tegenover agentic AI",
        "",
        "Deze intake verzamelt drie soorten context die later relevant zijn voor interpretatie van de expertoutput:",
        "- Klinische achtergrond en ervaring: om de professionele context van het antwoord te kunnen situeren.",
        "- Kennis van netwerkanalyse: omdat de casusopgaven expliciet vertrekken vanuit een netwerk-analytisch redeneermodel.",
        "- Basishouding tegenover agentic AI: om eventuele verwachtingsbias tegenover AI-systemen later mee te kunnen wegen.",
        "",
        "De intake staat bewust voor de casus, zodat deze context niet wordt gekleurd door het invullen van de inhoudelijke taken zelf.",
    ]


def _image_html(src: str, alt_text: str) -> str:
    return (
        '<div style="margin-top:12px;margin-bottom:12px;">'
        f'<img src="{html.escape(src)}" alt="{html.escape(alt_text)}" '
        'style="max-width:100%;height:auto;border:1px solid #CBD5E1;border-radius:6px;" />'
        "</div>"
    )


def maybe_figure_lines(filename: str, alt_text: str, image_mode: str, image_base_url: str | None) -> list[str]:
    if image_mode == "base-url":
        if not image_base_url:
            raise ValueError("image_base_url is required when image_mode='base-url'")
        src = image_base_url.rstrip("/") + "/" + filename
        return [_image_html(src, alt_text)]
    if image_mode == "inline":
        image_path = ASSET_ROOT / filename
        payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return [_image_html(f"data:image/png;base64,{payload}", alt_text)]
    return [f"[VOEG HIER FIGUUR TOE: {filename}]"]


def question_block(question_type: str, question_id: str, prompt_lines: list[str], *, choices: list[str] | None = None, answers: list[str] | None = None) -> list[str]:
    lines = [question_type, f"[[ID:{question_id}]]", *prompt_lines]
    if choices is not None:
        lines.append("[[Choices]]")
        lines.extend(choices)
    if answers is not None:
        lines.append("[[Answers]]")
        lines.extend(answers)
    return lines


def build_survey_lines(case: CaseSurvey, shared: dict[str, str], image_mode: str, image_base_url: str | None) -> list[str]:
    case_image = f"{case.case_code.lower()}_part3_network.png"
    example_image = "shared_part3_example_network.png"
    lines: list[str] = ["[[AdvancedFormat]]", ""]

    lines.extend(["[[Block:Toestemming]]", ""])
    lines.extend(question_block("[[Question:DB]]", "CONSENT_INFO", build_consent_block(case)))
    lines.extend([""])
    lines.extend(
        question_block(
            "[[Question:MC:SingleAnswer:Vertical]]",
            "CONSENT_DECISION",
            [
                "Ik heb bovenstaande informatie gelezen.",
                "",
                "Duid aan of u geinformeerde toestemming geeft om deel te nemen aan deze survey.",
            ],
            choices=[
                "Ja, ik geef geinformeerde toestemming en wens deel te nemen.",
                "Nee, ik wens niet deel te nemen.",
            ],
        )
    )
    lines.extend(["", "[[PageBreak]]", ""])

    lines.extend(["[[Block:Introductie]]", ""])
    lines.extend(question_block("[[Question:DB]]", "INTRO", build_intro_block(case, shared)))
    lines.extend(["", "[[PageBreak]]", ""])

    lines.extend(["[[Block:Intake]]", ""])
    lines.extend(question_block("[[Question:DB]]", "INTAKE_INFO", build_intake_intro()))
    lines.extend([""])
    lines.extend(
        question_block(
            "[[Question:TE:Essay]]",
            "INTAKE_BACKGROUND",
            [
                "Wat is uw professionele achtergrond en ervaring in de klinische praktijk?",
                "",
                "Beschrijf kort uw discipline, functie, aantal jaren klinische ervaring en de mate waarin u momenteel werkt met personen met psychische klachten.",
            ],
        )
    )
    lines.extend([""])
    lines.extend(
        question_block(
            "[[Question:Matrix]]",
            "INTAKE_AGENTIC_AI",
            [
                "Geef aan in welke mate u akkoord gaat met onderstaande uitspraken over agentic AI in een klinische context.",
            ],
            choices=[
                "Ik zie agentic AI als een potentieel zinvolle ondersteuning voor klinische besluitvorming in mentale gezondheidszorg.",
                "Ik zou agentic AI enkel vertrouwen wanneer de onderliggende redeneerstappen expliciet en controleerbaar worden getoond.",
                "Ik verwacht dat agentic AI binnen afzienbare tijd bruikbaar kan zijn voor gepersonaliseerde digitale geestelijke gezondheidszorg.",
            ],
            answers=SCALE_7PT,
        )
    )
    lines.extend([""])
    lines.extend(
        question_block(
            "[[Question:Matrix]]",
            "INTAKE_NETWORK_ANALYSIS",
            [
                "Geef aan in welke mate u akkoord gaat met onderstaande uitspraken over netwerk-analyse van psychopathologie.",
            ],
            choices=[
                "Ik ben voldoende vertrouwd met het netwerk-analytisch denken om de logica van de huidige surveyopgaven goed te kunnen volgen.",
                "Het in kaart brengen van dynamische relaties tussen klachten en modificeerbaar gedrag kan klinisch relevante behandelprioriteiten zichtbaar maken.",
                "Ik vind een netwerk-analytische benadering methodologisch bruikbaar als basis voor gepersonaliseerde EMA-monitoring en digitale interventies.",
            ],
            answers=SCALE_7PT,
        )
    )
    lines.extend(["", "[[PageBreak]]", ""])

    lines.extend(["[[Block:Instructies]]", ""])
    lines.extend(question_block("[[Question:DB]]", "INSTRUCTION_PAGE", render_lines(shared["intro"])))
    lines.extend(["", "[[PageBreak]]", ""])

    lines.extend(["[[Block:Deel 1]]", ""])
    lines.extend(question_block("[[Question:DB]]", "PART1_INFO", render_lines(shared["part1_shared"])))
    lines.extend([""])
    lines.extend(
        question_block(
            "[[Question:TE:Form]]",
            f"{case.case_code}_PART1",
            [
                f"{case.case_code} - {case.profile}",
                f"Duur: {case.duration}",
                "",
                case.complaint_vignette,
                "",
                "Opdracht: noteer 2-6 symptoomlabels voor deze casus. Gebruik enkel korte labels (2-5 woorden); voeg geen beschrijving toe.",
            ],
            choices=[f"Symptoom {idx}" for idx in range(1, 7)],
        )
    )
    lines.extend(["", "[[PageBreak]]", ""])

    lines.extend(["[[Block:Deel 2]]", ""])
    lines.extend(question_block("[[Question:DB]]", "PART2_INFO", render_lines(shared["part2_shared"])))
    lines.extend([""])
    lines.extend(
        question_block(
            "[[Question:TE:Form]]",
            f"{case.case_code}_PART2",
            [
                f"{case.case_code} - {case.profile} - Deel 2",
                "",
                case.short_summary,
                "",
                "Gestandaardiseerde symptomen uit Deel 1:",
                *[f"- {item}" for item in case.part2_symptoms],
                "",
                "Opdracht: noteer 3-5 behandelingsoptielabels voor een initieel observatiemodel. Zorg dat elk label later dagelijks via een mobiele app meetbaar kan worden gemaakt.",
            ],
            choices=[f"Behandelingsoptie {idx}" for idx in range(1, 6)],
        )
    )
    lines.extend(["", "[[PageBreak]]", ""])

    lines.extend(["[[Block:Deel 3]]", ""])
    part3_info = render_lines(shared["part3_shared"])
    part3_info.extend([""])
    part3_info.extend(maybe_figure_lines(example_image, "Voorbeeldnetwerk voor Deel 3", image_mode, image_base_url))
    lines.extend(question_block("[[Question:DB]]", "PART3_INFO", part3_info))
    lines.extend([""])
    part3_case_info = [
        f"{case.case_code} - {case.profile} - Deel 3",
        "",
        case.short_summary,
        "",
        f"21-daagse monitoring: {case.monitoring}",
        "",
    ]
    part3_case_info.extend(maybe_figure_lines(case_image, f"Netwerkfiguur voor {case.case_code}", image_mode, image_base_url))
    part3_case_info.extend(
        [
            "",
            "Beschikbare behandelingsopties:",
            *[f"- {choice}" for choice in case.part3_options],
        ]
    )
    lines.extend(question_block("[[Question:DB]]", "PART3_CASE_INFO", part3_case_info))
    lines.extend([""])
    lines.extend(
        question_block(
            "[[Question:RO]]",
            f"{case.case_code}_PART3",
            [
                "Rangschik alle 5 behandelingsopties van hoogste naar laagste behandelprioriteit.",
                "",
                "Sleep of rangschik de opties van meest prioritair naar minst prioritair.",
            ],
            choices=case.part3_options,
        )
    )
    lines.extend(["", "[[PageBreak]]", ""])

    lines.extend(["[[Block:Deel 4]]", ""])
    lines.extend(question_block("[[Question:DB]]", "PART4_INFO", render_lines(shared["part4_shared"])))
    lines.extend([""])
    lines.extend(
        question_block(
            "[[Question:DB]]",
            "PART4_CASE_INFO",
            [
                f"{case.case_code} - {case.profile} - Deel 4",
                "",
                "Gestandaardiseerde behandeldoelen (abstract) voor Deel 4:",
                *[f"- {target}" for target in case.part4_targets],
            ],
        )
    )
    lines.extend([""])
    lines.extend(
        question_block(
            "[[Question:MC:MultipleAnswer]]",
            f"{case.case_code}_PART4",
            [
                "Selecteer exact 6 EMA-items (2 per behandeldoel) die het best aansluiten als sub-behandelingsopties.",
                "",
                "Alle 20 items zijn dagelijkse mobiele EMA-items van het type behandelingsoptie, geen symptomen.",
            ],
            choices=[f"{idx}. {item}" for idx, item in enumerate(case.part4_items, start=1)],
        )
    )
    lines.extend(["", "[[PageBreak]]", ""])

    lines.extend(["[[Block:Deel 5]]", ""])
    lines.extend(question_block("[[Question:DB]]", "PART5_INFO", render_lines(shared["part5_shared"])))
    lines.extend([""])
    lines.extend(
        question_block(
            "[[Question:DB]]",
            "PART5_CASE_INFO",
            [
                f"{case.case_code} - {case.profile} - Deel 5",
                "",
                f"Primair probleem: {case.part5_primary_problem}",
                f"Behandeldoel: {case.part5_target}",
                f"Voornaamste barriere: {case.part5_barrier}",
                f"Copingstrategie: {case.part5_coping}",
            ],
        )
    )
    lines.extend([""])
    lines.extend(
        question_block(
            "[[Question:TE:Essay]]",
            f"{case.case_code}_PART5",
            [
                "Schrijf hieronder de mobiele coachingsboodschap voor deze casus.",
                "",
                "Formuleer alsof de tekst morgen rechtstreeks op de smartphone van de persoon verschijnt.",
            ],
        )
    )
    lines.extend(["", "[[PageBreak]]", ""])

    lines.extend(["[[Block:Deel 6]]", ""])
    lines.extend(question_block("[[Question:DB]]", "PART6_INFO", render_lines(shared["part6_intro"])))
    lines.extend([""])
    lines.extend(question_block("[[Question:TE:Essay]]", "PART6_A", [shared["part6_q_a"]]))
    lines.extend([""])
    lines.extend(question_block("[[Question:TE:Essay]]", "PART6_B", [shared["part6_q_b"]]))
    lines.extend([""])
    lines.extend(question_block("[[Question:TE:Essay]]", "PART6_C", [shared["part6_q_c"]]))
    lines.extend(["", "[[PageBreak]]", ""])

    lines.extend(["[[Block:Afronding]]", ""])
    lines.extend(question_block("[[Question:DB]]", "CLOSING", render_lines(shared["closing"])))
    lines.append("")
    return lines


def write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_manifest(cases: list[CaseSurvey]) -> None:
    manifest_path = MANIFEST_ROOT / "hcp_case_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "survey_name",
                "hcp_code",
                "case_code",
                "profile",
                "import_file",
                "case_network_image",
                "shared_example_image",
                "inline_images_embedded",
                "post_import_note",
            ],
        )
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "survey_name": f"PHOENIX_PRE_{case.hcp_code}_{case.case_code}",
                    "hcp_code": case.hcp_code,
                    "case_code": case.case_code,
                    "profile": case.profile,
                    "import_file": f"{case.hcp_code}_{case.case_code}.txt",
                    "case_network_image": f"{case.case_code.lower()}_part3_network.png",
                    "shared_example_image": "shared_part3_example_network.png",
                    "inline_images_embedded": "yes",
                    "post_import_note": "Stel in Qualtrics nog exact-6 validatie in voor Deel 4 en controleer de toestemmingsvraag.",
                }
            )

    email_path = MANIFEST_ROOT / "email_send_sheet.csv"
    with email_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "hcp_code",
                "case_code",
                "survey_name",
                "survey_link",
                "email_subject",
                "email_body_template",
            ],
        )
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "hcp_code": case.hcp_code,
                    "case_code": case.case_code,
                    "survey_name": f"PHOENIX_PRE_{case.hcp_code}_{case.case_code}",
                    "survey_link": "PASTE_FINAL_QUALTRICS_LINK_HERE",
                    "email_subject": f"Uitnodiging PHOENIX-evaluatiestudie - {case.hcp_code}",
                    "email_body_template": (
                        f"Beste collega,\n\nHierbij stuur ik u de link naar uw toegewezen PHOENIX-evaluatiesurvey "
                        f"({case.case_code}; {case.profile}). De survey bevat eerst geinformeerde toestemming, daarna een korte intake en vervolgens de inhoudelijke casusopgaven. "
                        f"Gelieve de survey zelfstandig in te vullen op basis van uw eigen klinisch oordeel, zonder gebruik van generatieve AI of andere externe hulp.\n\n"
                        f"Surveylink: {{SURVEY_LINK}}\n\n"
                        "Hartelijk dank voor uw medewerking.\n\n"
                        "Stijn Van Severen"
                    ),
                }
            )

    image_path = MANIFEST_ROOT / "image_manifest.csv"
    with image_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["filename", "used_in", "description"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "filename": "shared_part3_example_network.png",
                "used_in": "Deel 3 voorbeeldblok in alle 10 surveys",
                "description": "Voorbeeldnetwerk dat de logica van Part 3 illustreert.",
            }
        )
        for case in cases:
            writer.writerow(
                {
                    "filename": f"{case.case_code.lower()}_part3_network.png",
                    "used_in": f"Deel 3 case block voor {case.hcp_code} / {case.case_code}",
                    "description": f"Case-specifiek netwerk voor {case.case_code} ({case.profile}).",
                }
            )


def write_coverage_report(cases: list[CaseSurvey], image_mode: str) -> None:
    report_path = MANIFEST_ROOT / "source_coverage_report.md"
    lines = [
        "# Source Coverage Report",
        "",
        "Deze package is gegenereerd vanuit de actuele `1_case_per_HCP` LaTeX-bronbestanden.",
        "",
        "Gedekte survey-onderdelen:",
        "- Titelpagina / studiecontext / deelnemerscode / toegewezen casus",
        "- Expliciete informed-consent pagina",
        "- Intakeblok (achtergrond, agentic AI, netwerkanalyse)",
        "- Instructiepagina",
        "- Deel 1 t/m Deel 6",
        "- Voorbeelden in Deel 1 t/m Deel 5",
        "- Part 3 voorbeeldfiguur en case-specifieke netwerkfiguur",
        "- Afronding en terugbezorging",
        "",
        f"Figuurmodus in huidige generatie: `{image_mode}`",
        "",
        "Case-overzicht:",
    ]
    for case in cases:
        lines.append(f"- {case.hcp_code}: {case.case_code} ({case.profile})")
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_bundle_archive() -> None:
    with zipfile.ZipFile(BUNDLE_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for root in (IMPORT_ROOT, MANIFEST_ROOT, ASSET_ROOT):
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(GENERATED_ROOT.parent))


def build_package(image_mode: str, image_base_url: str | None) -> list[CaseSurvey]:
    ensure_dirs()
    template_tex = read_text(HCP_DIRS[0] / "main.tex")
    shared = build_shared_content(template_tex)
    example_figure = parse_network(
        shared["part3_example_network_tikz"],
        left_header="Behandelingsopties",
        right_header="Symptomen",
        title="shared_part3_example",
    )
    draw_network_figure(example_figure, ASSET_ROOT / "shared_part3_example_network.png")

    cases: list[CaseSurvey] = []
    for hcp_dir in HCP_DIRS:
        tex = read_text(hcp_dir / "main.tex")
        case = parse_case_survey(tex)
        cases.append(case)
        draw_network_figure(case.network, ASSET_ROOT / f"{case.case_code.lower()}_part3_network.png")
        survey_lines = build_survey_lines(case, shared, image_mode, image_base_url)
        write_lines(IMPORT_ROOT / f"{case.hcp_code}_{case.case_code}.txt", survey_lines)

    write_manifest(cases)
    write_coverage_report(cases, image_mode=image_mode)
    write_bundle_archive()
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Qualtrics automation package for the PHOENIX PRE HCP surveys.")
    parser.add_argument(
        "--image-mode",
        choices=("inline", "base-url", "placeholder"),
        default="inline",
        help="How Part 3 figures should be inserted into the generated survey files. Default: inline base64 images.",
    )
    parser.add_argument(
        "--image-base-url",
        default=None,
        help="Public base URL for the generated network images when --image-mode=base-url.",
    )
    args = parser.parse_args()

    cases = build_package(args.image_mode, args.image_base_url)
    print(f"Generated {len(cases)} Qualtrics import files in {IMPORT_ROOT}")
    print(f"Generated network assets in {ASSET_ROOT}")
    print(f"Generated manifests in {MANIFEST_ROOT}")
    print(f"Generated upload bundle at {BUNDLE_PATH}")


if __name__ == "__main__":
    main()
