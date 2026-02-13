#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
create_uml_diagram_elaborate.py

Creates a publication-ready UML-style diagram (Graphviz/DOT) that mirrors the
ANALYTICAL backbone of:

    02_operationalize_freetext_complaints.py

Design goals (your requested fix):
  - Avoid “first step horizontal / later steps vertical” layouts that create big bottom-left whitespace.
  - Keep the analytical pipeline correct, but *collapse parallel branches* (dense vs BM25, etc.)
    into vertical “hybrid retrieval” steps with explicit notes about internal parallelism.
  - Keep the runtime mapping backbone visually dominant; keep offline artifacts compact.

Pipeline (analytic flow only; UI intentionally excluded):
    Offline caches (from embed_leaf_nodes.py)
        ↓
    Startup init (load/validate caches, norms, BM25 index, searcher)
        ↓
    Per complaint:
        Free-text complaint
            → LLM decomposition (ALWAYS ON) → criterion variables
            → build retrieval query text per variable
            → batch embed all query texts (one call) + unit-normalize
            → per criterion (parallel bounded):
                hybrid retrieval (dense cosine + BM25) → candidate pool
                pool-only scorers (token overlap + fuzzy) → fusion (htssf/rrf/scoresum) → Top-K
                optional per-criterion LLM adjudication → chosen leaf or UNMAPPED
            → save JSON run artifact + final mappings

Outputs:
    - PNG diagram (bounded physical size, DPI-controlled)
    - Optional SVG (recommended for papers/LaTeX: infinite zoom)

Requirements:
    - system graphviz:
        macOS: brew install graphviz
        Linux: apt-get install graphviz
    - python package:
        pip install graphviz

Run:
    python create_uml_diagram_elaborate.py \
        --out "/path/to/UML_diagram_elaborate.png" \
        --svg "/path/to/UML_diagram_elaborate.svg" \
        --dpi 600 \
        --page "11,14"
"""

from __future__ import annotations

import os
import argparse
from graphviz import Digraph


# ---------------------------------------------------------------------
# CONFIG SNAPSHOT (keep aligned with 02_operationalize_freetext_complaints.py)
# ---------------------------------------------------------------------

# Embeddings model
EMBED_MODEL = os.environ.get("CRITERION_EMBED_MODEL", "text-embedding-3-small")

# LLM Decomposition (ALWAYS ON)
DECOMP_MODEL = os.environ.get("CRITERION_DECOMP_MODEL", "gpt-5-mini")
DECOMP_TEMPERATURE = float(os.environ.get("CRITERION_DECOMP_T", "1.0"))

# Optional per-criterion LLM picker
RERANK_MODEL = os.environ.get("CRITERION_RERANK_MODEL", "gpt-5-nano")
RERANK_TEMPERATURE = float(os.environ.get("CRITERION_RERANK_T", "1.0"))
LLM_RERANK_TOPN = int(os.environ.get("CRITERION_RERANK_TOPN", "200"))

# Retrieval weights (must sum to 1)
WEIGHT_EMBED = float(os.environ.get("CRITERION_W_EMBED", "0.80"))
WEIGHT_BM25 = float(os.environ.get("CRITERION_W_BM25", "0.12"))
WEIGHT_TOK = float(os.environ.get("CRITERION_W_TOK", "0.05"))
WEIGHT_FUZ = float(os.environ.get("CRITERION_W_FUZ", "0.03"))

# Fusion behavior (script hard-sets htssf; allow env override for diagram label)
FUSION_METHOD = os.environ.get("CRITERION_FUSION_METHOD", "htssf").lower()
RRF_K = int(os.environ.get("CRITERION_RRF_K", "60"))

# HTSSF params
HTSSF_ALPHA = float(os.environ.get("CRITERION_HTSSF_ALPHA", "0.90"))
HTSSF_TEMPS = (
    float(os.environ.get("CRITERION_HTSSF_T_EMBED", "0.07")),
    float(os.environ.get("CRITERION_HTSSF_T_BM25", "1.00")),
    float(os.environ.get("CRITERION_HTSSF_T_TOK", "0.35")),
    float(os.environ.get("CRITERION_HTSSF_T_FUZ", "0.35")),
)

# Candidate settings
CANDIDATES_PER_METHOD = int(os.environ.get("CRITERION_CAND_PER_METHOD", "600"))
CANDIDATE_POOL = int(os.environ.get("CRITERION_CAND_POOL", "8000"))
TOP_K_RESULTS = int(os.environ.get("CRITERION_TOP_K_RESULTS", "200"))

# Parallelism
MAX_PARALLEL_CRITERIA = int(os.environ.get("CRITERION_MAX_PARALLEL", "3"))

# Query building
INCLUDE_EVIDENCE_IN_QUERY = os.environ.get("CRITERION_INCLUDE_EVIDENCE", "true").strip().lower() in {"1", "true", "yes"}
# Important nuance from your code: to_query_text() currently returns criterion only (evidence inclusion suppressed).
EVIDENCE_ACTUALLY_USED = False

# Norm computation chunking (depicted)
NORM_CHUNK_ROWS = int(os.environ.get("CRITERION_NORM_CHUNK_ROWS", "4096"))

# Default output path (adjust as needed)
DEFAULT_OUT = os.environ.get(
    "UML_OUT",
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/SystemComponents/Agentic_Framework/"
    "01_OperationalizationMentalHealthProblem/utils/UML_diagrams/UML_diagram_elaborate.png"
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _weights_ok() -> bool:
    s = WEIGHT_EMBED + WEIGHT_BM25 + WEIGHT_TOK + WEIGHT_FUZ
    return abs(s - 1.0) < 1e-6


def build_diagram(dpi: int, page_in: str) -> Digraph:
    """
    dpi: rendering DPI for raster outputs
    page_in: "W,H" in inches; diagram is scaled to fit within this box (Graphviz size="W,H!")
    """
    g = Digraph("CRITERION_Operationalizer_UML", engine="dot")

    page_in = (page_in or "").strip()
    if "," not in page_in:
        page_in = "11,14"
    W, H = [x.strip() for x in page_in.split(",", 1)]
    size_attr = f"{W},{H}!"

    # Graph-level styling (paper-friendly, bounded)
    g.attr(
        rankdir="TB",
        dpi=str(int(dpi)),
        size=size_attr,
        ratio="compress",
        charset="UTF-8",
        fontname="Helvetica",
        fontsize="11",
        labelloc="t",
        label="CRITERION Operationalizer (PHOENIX) — Analytical Mapping Backbone (UML-style)",
        splines="ortho",
        nodesep="0.30",
        ranksep="0.42",
        pad="0.16",
        bgcolor="white",
        newrank="true",
        compound="true",
        concentrate="true",
    )

    # Node defaults
    g.attr(
        "node",
        shape="box",
        style="rounded,filled",
        fillcolor="#F7F7F7",
        color="#3A3A3A",
        fontname="Helvetica",
        fontsize="9",
        margin="0.10,0.06",
        penwidth="1.10",
    )

    # Edge defaults
    g.attr(
        "edge",
        color="#444444",
        fontname="Helvetica",
        fontsize="8",
        arrowsize="0.75",
        penwidth="1.00",
    )

    # Helper: Graphviz HTML labels
    def H(inner: str) -> str:
        return f"<{inner}>"

    # -----------------------------------------------------------------
    # Cluster: Offline cached artifacts (compact single node to avoid wide layout)
    # -----------------------------------------------------------------
    with g.subgraph(name="cluster_offline") as off:
        off.attr(
            label="Offline artifacts (produced by embed_leaf_nodes.py)",
            style="rounded",
            color="#777777",
        )

        off.node(
            "offline_cache",
            label=H(
                "<B>Cached leaf-node data</B><BR ALIGN='LEFT'/>"
                "• EMBEDTEXT paths: CRITERION_leaf_paths_EMBEDTEXT.json<BR ALIGN='LEFT'/>"
                "• FULL paths: CRITERION_leaf_paths_FULL.json<BR ALIGN='LEFT'/>"
                "• LEXTEXT paths: CRITERION_leaf_paths_LEXTEXT.json<BR ALIGN='LEFT'/>"
                "• Embeddings: CRITERION_leaf_embeddings.npy (E: [n_leaf × d])<BR ALIGN='LEFT'/>"
                "• Norms: CRITERION_leaf_embedding_norms.npy (||E_i||: [n_leaf])<BR ALIGN='LEFT'/>"
                "• Meta: CRITERION_leaf_embeddings_meta.json (paths_hash/model/status)"
            ),
            shape="note",
            fillcolor="#FFFFFF",
        )

    # -----------------------------------------------------------------
    # Cluster: Startup / initialization (once per run)
    # -----------------------------------------------------------------
    with g.subgraph(name="cluster_startup") as st:
        st.attr(
            label="Startup initialization (once per run)",
            style="rounded",
            color="#777777",
        )

        st.node(
            "init_env",
            label=H(
                "<B>S0) Load environment</B><BR ALIGN='LEFT'/>"
                "• optional .env loading (dotenv/manual)<BR ALIGN='LEFT'/>"
                "• require OPENAI_API_KEY for LLM + embeddings"
            ),
            fillcolor="#FFFFFF",
        )

        st.node(
            "init_load",
            label=H(
                "<B>S1) Load + validate caches</B><BR ALIGN='LEFT'/>"
                "• required cache files exist<BR ALIGN='LEFT'/>"
                "• paths_hash matches meta<BR ALIGN='LEFT'/>"
                f"• embed model matches meta (<I>{EMBED_MODEL}</I>)<BR ALIGN='LEFT'/>"
                "• status==complete<BR ALIGN='LEFT'/>"
                "• mmap embeddings E"
            ),
            fillcolor="#FFFFFF",
        )

        st.node(
            "init_norms",
            label=H(
                "<B>S2) Ensure norms available</B><BR ALIGN='LEFT'/>"
                "if norms missing/wrong length:<BR ALIGN='LEFT'/>"
                f"• compute ||E_i|| chunked (rows={NORM_CHUNK_ROWS})<BR ALIGN='LEFT'/>"
                "• cache to norms.npy"
            ),
        )

        st.node(
            "init_bm25",
            label=H(
                "<B>S3) Build BM25Sparse over LEXTEXT</B><BR ALIGN='LEFT'/>"
                "tokenize_norm(): lower + light suffix stripping<BR ALIGN='LEFT'/>"
                "postings: token → [(doc_id, tf)]<BR ALIGN='LEFT'/>"
                "idf(t)=log(1 + (N-df+0.5)/(df+0.5))<BR ALIGN='LEFT'/>"
                "<I>k1=1.5, b=0.75</I>"
            ),
        )

        st.node(
            "init_searcher",
            label=H(
                "<B>S4) Instantiate searcher</B><BR ALIGN='LEFT'/>"
                "CriterionSearcher(full_paths, embed_paths, lex_texts, E, ||E||, bm25)"
            ),
            fillcolor="#FFFFFF",
        )

    # Force vertical flow (avoid side-by-side placement)
    g.edge("offline_cache", "init_env", xlabel="inputs")
    g.edge("init_env", "init_load", xlabel="config ready")
    g.edge("init_load", "init_norms", xlabel="after validation")
    g.edge("init_norms", "init_bm25", xlabel="LEXTEXT ready")
    g.edge("init_bm25", "init_searcher", xlabel="bm25 ready")

    # -----------------------------------------------------------------
    # Cluster: Runtime pipeline (per complaint) — collapsed vertical steps
    # -----------------------------------------------------------------
    with g.subgraph(name="cluster_runtime") as r:
        r.attr(
            label="Runtime analytic pipeline (per complaint)",
            style="rounded",
            color="#777777",
        )

        r.node(
            "input_txt",
            label=H("<B>R0) Input</B><BR/>Free-text complaint (single string)"),
            fillcolor="#FFFFFF",
        )

        r.node(
            "llm_decomp",
            label=H(
                "<B>R1) LLM decomposition</B> [ALWAYS ON]<BR ALIGN='LEFT'/>"
                f"model=<I>{DECOMP_MODEL}</I>, T={DECOMP_TEMPERATURE:.2f}<BR ALIGN='LEFT'/>"
                "clinical-expert reasoning; NO ontology usage<BR ALIGN='LEFT'/>"
                "atomic, minimally-overlapping variables (3–12 typical)<BR ALIGN='LEFT'/>"
                "<I>strict JSON</I>: meta{n_variables,notes}, variables[]<BR ALIGN='LEFT'/>"
                "{id,label,criterion,evidence,polarity,timeframe,severity_0_1,confidence_0_1}<BR ALIGN='LEFT'/>"
                "robust JSON extraction + retries/backoff"
            ),
        )

        r.node(
            "vars",
            label=H("<B>R1b) Criterion variables</B><BR ALIGN='LEFT'/>List[CriterionVariable]"),
        )

        evidence_mode = "criterion only (current code)" if not EVIDENCE_ACTUALLY_USED else "criterion + evidence"
        evidence_note = (
            "INCLUDE_EVIDENCE_IN_QUERY=True but to_query_text() returns criterion only"
            if (INCLUDE_EVIDENCE_IN_QUERY and not EVIDENCE_ACTUALLY_USED) else
            "query text = criterion; optionally append evidence"
        )

        r.node(
            "build_queries",
            label=H(
                "<B>R2) Build retrieval query text</B><BR ALIGN='LEFT'/>"
                f"mode: <I>{evidence_mode}</I><BR ALIGN='LEFT'/>"
                f"note: {evidence_note}"
            ),
            fillcolor="#FFFFFF",
        )

        r.node(
            "embed_batch",
            label=H(
                "<B>R3) Batch-embed criterion queries</B><BR ALIGN='LEFT'/>"
                f"model=<I>{EMBED_MODEL}</I><BR ALIGN='LEFT'/>"
                "one embeddings call for all queries (retry/backoff)<BR ALIGN='LEFT'/>"
                "Q[i] ← Q[i] / (||Q[i]|| + eps) (unit-normalize)"
            ),
        )

        # Collapsed per-criterion mapping (single vertical block, avoids branch whitespace)
        weights_lbl = f"W=[{WEIGHT_EMBED:.2f},{WEIGHT_BM25:.2f},{WEIGHT_TOK:.2f},{WEIGHT_FUZ:.2f}]"
        temps_lbl = f"T=[{HTSSF_TEMPS[0]:.2f},{HTSSF_TEMPS[1]:.2f},{HTSSF_TEMPS[2]:.2f},{HTSSF_TEMPS[3]:.2f}]"
        ok = "OK" if _weights_ok() else "NOT-OK"
        active = (FUSION_METHOD or "").strip().lower()
        active_txt = f"<B>ACTIVE fusion:</B> <I>{active}</I>" if active else "<B>ACTIVE fusion:</B> <I>htssf</I>"

        # Fusion math snippet (keep compact, but correct)
        fusion_math = (
            "<B>Fusion</B> (pool-scoped):<BR ALIGN='LEFT'/>"
            "• htssf: softmax per method (temp-scaled) + small RRF backstop<BR ALIGN='LEFT'/>"
            "  s_bm25 ← log1p(max(bm25,0)); p_m(i)=softmax(s_m(i)/T_m) over pool<BR ALIGN='LEFT'/>"
            f"  temps {temps_lbl}, α={HTSSF_ALPHA:.2f}, RRF_k={RRF_K}<BR ALIGN='LEFT'/>"
            "• rrf: Σ w_m/(RRF_k + rank_m(i) + 1)<BR ALIGN='LEFT'/>"
            "• scoresum: Σ w_m·minmax(s_m(i)) over pool"
        )

        r.node(
            "per_criterion",
            label=H(
                "<B>R4) Per-criterion mapping</B> (repeat for each variable)<BR ALIGN='LEFT'/>"
                f"<I>parallelizable</I>: bounded ThreadPoolExecutor (max={MAX_PARALLEL_CRITERIA})<BR ALIGN='LEFT'/>"
                "<BR ALIGN='LEFT'/>"
                "<B>R4a) Hybrid retrieval</B> (internal parallelism)<BR ALIGN='LEFT'/>"
                f"• Dense cosine: emb_scores[i]=(E·q)/(||E_i||+eps); topK=max({CANDIDATES_PER_METHOD},{CANDIDATE_POOL})<BR ALIGN='LEFT'/>"
                f"• Sparse BM25 on LEXTEXT; topK=max({CANDIDATES_PER_METHOD},{CANDIDATE_POOL})<BR ALIGN='LEFT'/>"
                "<BR ALIGN='LEFT'/>"
                "<B>R4b) Candidate pooling</B><BR ALIGN='LEFT'/>"
                "• pool = union(dense_topK, bm25_topK)<BR ALIGN='LEFT'/>"
                f"• if |pool|&gt;{CANDIDATE_POOL}: trim by emb_scores<BR ALIGN='LEFT'/>"
                "<BR ALIGN='LEFT'/>"
                "<B>R4c) Pool-only scorers</B><BR ALIGN='LEFT'/>"
                "• token overlap = Jaccard(tokens(query), tokens(LEXTEXT))<BR ALIGN='LEFT'/>"
                "• fuzzy score = rapidfuzz ratio (or difflib fallback)<BR ALIGN='LEFT'/>"
                "<BR ALIGN='LEFT'/>"
                f"{active_txt}<BR ALIGN='LEFT'/>"
                f"weights {weights_lbl} (sum={ok})<BR ALIGN='LEFT'/>"
                f"{fusion_math}<BR ALIGN='LEFT'/>"
                "<BR ALIGN='LEFT'/>"
                f"<B>R4d) Top-K</B>: keep TOP_K_RESULTS={TOP_K_RESULTS} candidates"
            ),
        )

        r.node(
            "llm_pick",
            label=H(
                "<B>R5) Optional LLM best-leaf adjudication</B> [per-criterion]<BR ALIGN='LEFT'/>"
                f"model=<I>{RERANK_MODEL}</I>, T={RERANK_TEMPERATURE:.2f}<BR ALIGN='LEFT'/>"
                f"consider top-N={LLM_RERANK_TOPN} (+ special idx=-1 UNMAPPED candidate)<BR ALIGN='LEFT'/>"
                "<I>strict JSON</I>: {idx:int, confidence:float, rationale:str}<BR ALIGN='LEFT'/>"
                "parallelizable (bounded) + retries/backoff"
            ),
        )

        r.node(
            "choose",
            label=H(
                "<B>R6) Choose leaf</B><BR ALIGN='LEFT'/>"
                "default: top_fused (rank 1)<BR ALIGN='LEFT'/>"
                "or: LLM-picked idx (if reranker enabled)<BR ALIGN='LEFT'/>"
                "idx=-1 means UNMAPPED"
            ),
            fillcolor="#FFFFFF",
        )

        r.node(
            "save",
            label=H(
                "<B>R7) Save run artifact (JSON)</B><BR ALIGN='LEFT'/>"
                "• timestamp_local + hashed complaint id<BR ALIGN='LEFT'/>"
                "• config snapshot + decomposition JSON<BR ALIGN='LEFT'/>"
                "• per-criterion: candidates + chosen leaf + debug<BR ALIGN='LEFT'/>"
                "<I>OUTPUT_DIR/operationalizations/run_*.json</I>"
            ),
            fillcolor="#FFFFFF",
        )

        r.node(
            "output",
            label=H(
                "<B>R8) Outputs</B><BR ALIGN='LEFT'/>"
                "1) mappings (criterion_variable → ontology_leaf or UNMAPPED)<BR ALIGN='LEFT'/>"
                "2) run artifact path (JSON)"
            ),
            fillcolor="#FFFFFF",
        )

        r.node(
            "legend",
            label=H(
                "<B>Legend</B><BR ALIGN='LEFT'/>"
                "solid edge = required step<BR ALIGN='LEFT'/>"
                "dashed edge = optional (if LLM adjudication enabled)<BR ALIGN='LEFT'/>"
                "dotted edge = default selection when adjudication disabled"
            ),
            shape="note",
            fillcolor="#FFFFFF",
            color="#888888",
        )

        # Runtime edges (strict vertical ordering)
        r.edge("init_searcher", "input_txt", xlabel="searcher ready")
        r.edge("input_txt", "llm_decomp", xlabel="R1")
        r.edge("llm_decomp", "vars", xlabel="variables[]")
        r.edge("vars", "build_queries", xlabel="to_query_text()")
        r.edge("build_queries", "embed_batch", xlabel="query_texts[]")
        r.edge("embed_batch", "per_criterion", xlabel="Q (unit)")

        # Optional adjudication + default fallback
        r.edge("per_criterion", "llm_pick", xlabel="top-N", style="dashed")
        r.edge("per_criterion", "choose", xlabel="top_fused", style="dotted")
        r.edge("llm_pick", "choose", xlabel="best idx", style="dashed")

        r.edge("choose", "save", xlabel="chosen leaf")
        r.edge("save", "output", xlabel="return mappings")

        # Keep legend anchored without impacting layout
        r.edge("output", "legend", style="invis")

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
    ap.add_argument("--out", default=DEFAULT_OUT, help="Output PNG path")
    ap.add_argument("--svg", default=os.environ.get("UML_SVG_OUT", ""), help="Optional SVG output path")
    ap.add_argument("--dpi", type=int, default=int(os.environ.get("UML_DPI", "600")), help="Raster DPI")
    ap.add_argument("--page", default=os.environ.get("UML_PAGE_IN", "11,14"), help='Page size in inches, "W,H"')
    args = ap.parse_args()

    svg_path = args.svg.strip() or None
    out_png, out_svg = render(args.out, svg_path, dpi=args.dpi, page_in=args.page)

    print(f"[ok] PNG: {out_png}")
    if out_svg:
        print(f"[ok] SVG: {out_svg}")


if __name__ == "__main__":
    main()
