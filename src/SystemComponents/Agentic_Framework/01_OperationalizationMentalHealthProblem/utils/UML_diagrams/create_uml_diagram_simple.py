#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
_create_uml_diagram_simple.py

Creates a SIMPLE, publication-ready UML-style overview diagram (Graphviz/DOT) for the
CRITERION operationalizer pipeline (analytic flow only; no UI; no deep math blocks).

Design goals:
  - Clear on paper (A4/Letter), minimal but still technical
  - Bounded output size (Graphviz size="W,H!") to avoid "overblown" PNGs
  - No edge labels (they often collide); steps are numbered in node titles instead
  - Crucial steps present and correct w.r.t. 02_operationalize_freetext_multiple_issues_with_interface.py

Logic reflected (high level):
  - Startup: validate offline cache (paths/meta/embeddings), load or compute norms, build BM25 on cached LEXTEXT
  - Runtime per complaint:
      R1) LLM decomposition (always-on) -> criterion variables (JSON)
      R2) Batch-embed criterion queries (one call), unit-normalize q_i
      R3) For each criterion (modestly parallel):
            dense cosine over cached embeddings (EMBEDTEXT)
            sparse BM25 over LEXTEXT
            candidate pool = union(dense_topK, bm25_topK), trim to CANDIDATE_POOL by embedding score
            token-overlap + fuzzy computed on pool only
            fusion ranks pool, emit Top-K candidates
            optional LLM adjudication picks single best leaf from top-N
      R8) Outputs: multi-leaf mappings + saved run artifact JSON

Outputs:
  - PNG (DPI-controlled)
  - Optional SVG

Run:
  python _create_uml_diagram_simple.py \
      --out "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/SystemComponents/Agentic_Framework/01_OperationalizationMentalHealthProblem/utils/UML_diagrams/UML_diagram_simple.png" \
      --svg "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/SystemComponents/Agentic_Framework/01_OperationalizationMentalHealthProblem/utils/UML_diagrams/UML_diagram_simple.svg" \
      --dpi 600 \
      --page "11,8.5"
"""

from __future__ import annotations

import os
import argparse
from graphviz import Digraph


# ---------------------------------------------------------------------
# CONFIG SNAPSHOT (kept short; defaults match 02_operationalize_freetext_multiple_issues_with_interface.py)
# ---------------------------------------------------------------------

EMBED_MODEL = os.environ.get("CRITERION_EMBED_MODEL", "text-embedding-3-small")
DECOMP_MODEL = os.environ.get("CRITERION_DECOMP_MODEL", "gpt-5")
RERANK_MODEL = os.environ.get("CRITERION_RERANK_MODEL", "gpt-5-nano")

# Fusion mode shown (the pipeline supports htssf / rrf / scoresum; default is htssf)
FUSION_METHOD = os.environ.get("CRITERION_FUSION_METHOD", "htssf").lower().strip()

# Candidate settings (defaults reflect new logic)
CANDIDATES_PER_METHOD = int(os.environ.get("CRITERION_CAND_PER_METHOD", "600"))
CANDIDATE_POOL = int(os.environ.get("CRITERION_CAND_POOL", "8000"))
TOP_K_RESULTS = int(os.environ.get("CRITERION_TOP_K_RESULTS", "200"))
LLM_RERANK_TOPN = int(os.environ.get("CRITERION_LLM_RERANK_TOPN", "200"))

DEFAULT_OUT = "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/SystemComponents/Agentic_Framework/01_OperationalizationMentalHealthProblem/utils/UML_diagrams/UML_diagram_simple.png"


# ---------------------------------------------------------------------
# Diagram builder
# ---------------------------------------------------------------------

def build_diagram(dpi: int, page_in: str) -> Digraph:
    """
    dpi: rendering DPI for raster outputs
    page_in: "W,H" in inches; diagram scaled to fit within this box (Graphviz size="W,H!")
    """
    g = Digraph("CRITERION_Operationalizer_UML_SIMPLE", engine="dot")

    page_in = (page_in or "").strip()
    if "," not in page_in:
        # Landscape default for simple overview
        page_in = "11,8.5"
    W, H = [x.strip() for x in page_in.split(",", 1)]
    size_attr = f"{W},{H}!"

    # Graph-level styling: bounded and compact (avoid "overblown" layouts)
    g.attr(
        rankdir="LR",
        dpi=str(int(dpi)),
        size=size_attr,
        ratio="compress",
        charset="UTF-8",
        fontname="Helvetica",
        fontsize="12",
        labelloc="t",
        label="CRITERION Operationalizer — Simple Analytical Overview (UML-style)",
        bgcolor="white",

        # Layout tuning
        splines="ortho",
        overlap="false",
        newrank="true",
        pack="true",
        packmode="node",
        compound="true",

        # Compact but readable
        nodesep="0.18",
        ranksep="0.28",
        pad="0.08",
    )

    # Node defaults
    g.attr(
        "node",
        shape="box",
        style="rounded,filled",
        fillcolor="#F7F7F7",
        color="#3A3A3A",
        fontname="Helvetica",
        fontsize="10",
        margin="0.10,0.06",
        penwidth="1.05",
    )

    # Edge defaults
    g.attr(
        "edge",
        color="#444444",
        arrowsize="0.70",
        penwidth="0.95",
    )

    # Helper: Graphviz HTML labels
    def H(inner: str) -> str:
        return f"<{inner}>"

    # -----------------------------------------------------------------
    # Nodes (single main flow + minimal supporting nodes)
    # -----------------------------------------------------------------

    g.node(
        "cache",
        label=H(
            "<B>Offline cache</B><BR ALIGN='LEFT'/>"
            "leaf paths (EMBEDTEXT/LEXTEXT/FULL)<BR ALIGN='LEFT'/>"
            "embeddings + norms + meta"
        ),
        shape="note",
        fillcolor="#FFFFFF",
        color="#888888",
        fontsize="9",
    )

    g.node(
        "startup",
        label=H(
            "<B>S0: Startup init</B><BR ALIGN='LEFT'/>"
            "validate cache; load/compute norms;<BR ALIGN='LEFT'/>"
            "build BM25 on cached LEXTEXT"
        ),
        fillcolor="#FFFFFF",
    )

    g.node(
        "input",
        label=H("<B>R0: Free-text complaint</B>"),
        fillcolor="#FFFFFF",
    )

    g.node(
        "decomp",
        label=H(
            "<B>R1: LLM decomposition (always)</B><BR ALIGN='LEFT'/>"
            f"model: <I>{DECOMP_MODEL}</I><BR ALIGN='LEFT'/>"
            "→ criterion variables (strict JSON)"
        ),
    )

    g.node(
        "embed",
        label=H(
            "<B>R2: Batch-embed criteria</B><BR ALIGN='LEFT'/>"
            f"model: <I>{EMBED_MODEL}</I><BR ALIGN='LEFT'/>"
            "q_i per variable (unit-normalized)"
        ),
    )

    g.node(
        "loop",
        label=H(
            "<B>R3: Per-criterion mapping loop</B><BR ALIGN='LEFT'/>"
            "map_one(variable)  (modest parallelism)"
        ),
        fillcolor="#FFFFFF",
    )

    g.node(
        "hybrid",
        label=H(
            "<B>R4: Hybrid retrieval + pool scoring</B><BR ALIGN='LEFT'/>"
            "dense cosine (EMBEDTEXT) + BM25 (LEXTEXT)<BR ALIGN='LEFT'/>"
            f"pool=union(topK_dense, topK_bm25) → trim {CANDIDATE_POOL}<BR ALIGN='LEFT'/>"
            "token overlap + fuzzy on pool only"
        ),
    )

    g.node(
        "fusion",
        label=H(
            "<B>R5: Fusion rank → Top-K</B><BR ALIGN='LEFT'/>"
            f"method: <I>{FUSION_METHOD}</I><BR ALIGN='LEFT'/>"
            f"emit Top-{TOP_K_RESULTS} candidates"
        ),
    )

    g.node(
        "rerank",
        label=H(
            "<B>R6: Optional LLM adjudication</B><BR ALIGN='LEFT'/>"
            f"model: <I>{RERANK_MODEL}</I><BR ALIGN='LEFT'/>"
            f"pick best from Top-{LLM_RERANK_TOPN}"
        ),
        fillcolor="#FFFFFF",
    )

    g.node(
        "output",
        label=H(
            "<B>R7: Outputs</B><BR ALIGN='LEFT'/>"
            "criterion→leaf mappings (multi-leaf)<BR ALIGN='LEFT'/>"
            "saved run artifact JSON (operationalizations/)"
        ),
        fillcolor="#FFFFFF",
    )

    # Minimal legend
    g.node(
        "legend",
        label=H(
            "<B>Legend</B><BR ALIGN='LEFT'/>"
            "solid: required<BR ALIGN='LEFT'/>"
            "dashed: optional reranker<BR ALIGN='LEFT'/>"
            "dotted: default select"
        ),
        shape="note",
        fillcolor="#FFFFFF",
        color="#888888",
        fontsize="9",
    )

    # -----------------------------------------------------------------
    # Edges (no labels; style indicates optional path)
    # -----------------------------------------------------------------

    g.edge("cache", "startup")
    g.edge("startup", "loop")

    g.edge("input", "decomp")
    g.edge("decomp", "embed")
    g.edge("embed", "loop")

    g.edge("loop", "hybrid")
    g.edge("hybrid", "fusion")

    # Default when reranker disabled: choose top fused candidate per criterion
    g.edge("fusion", "output", style="dotted")

    # Optional reranker path (per criterion)
    g.edge("fusion", "rerank", style="dashed")
    g.edge("rerank", "output", style="dashed")

    g.edge("output", "legend", style="invis")

    return g


def render(out_png: str, out_svg: str | None, dpi: int, page_in: str) -> tuple[str, str | None]:
    out_png = os.path.abspath(out_png)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    g = build_diagram(dpi=dpi, page_in=page_in)

    png_bytes = g.pipe(format="png")
    with open(out_png, "wb") as f:
        f.write(png_bytes)

    out_svg_path = None
    if out_svg:
        out_svg_path = os.path.abspath(out_svg)
        os.makedirs(os.path.dirname(out_svg_path), exist_ok=True)
        svg_bytes = g.pipe(format="svg")
        with open(out_svg_path, "wb") as f:
            f.write(svg_bytes)

    return out_png, out_svg_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("UML_OUT", DEFAULT_OUT), help="Output PNG path")
    ap.add_argument("--svg", default=os.environ.get("UML_SVG_OUT", ""), help="Optional SVG output path")
    ap.add_argument("--dpi", type=int, default=int(os.environ.get("UML_DPI", "600")), help="Raster DPI")
    ap.add_argument(
        "--page",
        default=os.environ.get("UML_PAGE_IN", "11,8.5"),
        help='Page size in inches, "W,H" (default: 11,8.5 landscape)',
    )
    args = ap.parse_args()

    svg_path = args.svg.strip() or None
    out_png, out_svg = render(args.out, svg_path, dpi=args.dpi, page_in=args.page)

    print(f"[ok] PNG: {out_png}")
    if out_svg:
        print(f"[ok] SVG: {out_svg}")


if __name__ == "__main__":
    main()
