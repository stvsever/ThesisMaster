#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""create_uml_diagram_apply_readiness_check_elaborate.py

Creates a publication-ready UML-style flow diagram (Graphviz/DOT) for:

    apply_readiness_check.py

Outputs
-------
- PNG diagram (DPI-controlled)
- Optional SVG (recommended for papers/LaTeX)

Requirements
------------
- system graphviz: brew install graphviz | apt-get install graphviz
- python: pip install graphviz

Run (local)
-----------
python create_uml_diagram_apply_readiness_check_elaborate.py \
  --out "/path/to/UML_diagram/UML_apply_readiness_check_elaborate.png" \
  --svg "/path/to/UML_diagram/UML_apply_readiness_check_elaborate.svg" \
  --dpi 600 --page "11,14"
"""

from __future__ import annotations

import os
import argparse
from graphviz import Digraph


# ---------------------------------------------------------------------
# CONFIG SNAPSHOT (keep aligned with apply_readiness_check.py)
# ---------------------------------------------------------------------

LLM_MODEL = "gpt-5-nano"

# Key feasibility floors that materially affect Tier 3 decisions (from Thresholds)
TIER3_STATIC_Q25_FLOOR = 35
TIER3_TV_Q25_FLOOR = 60

# Default output (your local machine). Override with --out.
DEFAULT_OUT = os.environ.get(
    "UML_OUT",
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/SystemComponents/Hierarchical_Updating_Algorithm/"
    "01_time_series_analysis/01_check_readiness/UML_diagram/"
    "UML_apply_readiness_check_elaborate.png",
)


# ---------------------------------------------------------------------
# Diagram builder
# ---------------------------------------------------------------------

def build_diagram(dpi: int, page_in: str) -> Digraph:
    """Builds a compact vertical DOT graph."""

    g = Digraph("Readiness_Check_UML", engine="dot")

    page_in = (page_in or "").strip()
    if "," not in page_in:
        page_in = "11,14"
    W, H = [x.strip() for x in page_in.split(",", 1)]
    size_attr = f"{W},{H}!"

    g.attr(
        rankdir="TB",
        dpi=str(int(dpi)),
        size=size_attr,
        ratio="compress",
        charset="UTF-8",
        fontname="Helvetica",
        fontsize="11",
        labelloc="t",
        label="apply_readiness_check.py — Readiness tiers for idiographic time-series network analysis (tv-gVAR supported)",
        splines="polyline",
        nodesep="0.30",
        ranksep="0.44",
        pad="0.16",
        bgcolor="white",
        newrank="true",
        compound="true",
        concentrate="true",
    )

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

    g.attr(
        "edge",
        color="#444444",
        fontname="Helvetica",
        fontsize="8",
        arrowsize="0.75",
        penwidth="1.00",
    )

    def H(inner: str) -> str:
        return f"<{inner}>"

    # --------------------------------------------------------------
    # Cluster: Inputs & configuration
    # --------------------------------------------------------------
    with g.subgraph(name="cluster_inputs") as c:
        c.attr(label="Inputs & configuration", style="rounded", color="#777777")

        c.node(
            "in_root",
            label=H(
                "<B>Input root</B><BR ALIGN='LEFT'/>"
                "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/Evaluation/01_pseudoprofile(s)/time_series_data/pseudodata<BR ALIGN='LEFT'/>"
                "Discover per-profile CSV: <I>pseudodata_wide.csv</I>"
            ),
            shape="note",
            fillcolor="#FFFFFF",
        )

        c.node(
            "out_root",
            label=H(
                "<B>Output root</B><BR ALIGN='LEFT'/>"
                "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/Evaluation/04_initial_observation_analysis/00_readiness_check<BR ALIGN='LEFT'/>"
                "Write per profile: readiness_report.json + readiness_summary.txt"
            ),
            shape="note",
            fillcolor="#FFFFFF",
        )

        c.node(
            "deps",
            label=H(
                "<B>Optional deps</B><BR ALIGN='LEFT'/>"
                "scipy (distribution tests), statsmodels (ADF/KPSS, Ljung-Box)<BR ALIGN='LEFT'/>"
                "openai client (LLM finalization)"
            ),
            shape="note",
            fillcolor="#FFFFFF",
        )

    # --------------------------------------------------------------
    # Cluster: Run init / discovery
    # --------------------------------------------------------------
    with g.subgraph(name="cluster_init") as s:
        s.attr(label="Run initialization (once)", style="rounded", color="#777777")

        s.node(
            "cli",
            label=H(
                "<B>S1) Parse CLI args</B><BR ALIGN='LEFT'/>"
                "--input-root, --output-root, --filename, --lag<BR ALIGN='LEFT'/>"
                "--max-profiles, --llm-finalize, --prefer-time-varying, --quiet"
            ),
            fillcolor="#FFFFFF",
        )

        s.node(
            "discover",
            label=H(
                "<B>S2) Discover profiles</B><BR ALIGN='LEFT'/>"
                "discover_profiles(): rglob(filename) under input_root<BR ALIGN='LEFT'/>"
                "Optional slice: first N (--max-profiles)"
            ),
            fillcolor="#FFFFFF",
        )

        s.node(
            "loop",
            label=H(
                "<B>S3) For each CSV</B><BR ALIGN='LEFT'/>"
                "analyze_profile(csv_path, output_root, lag, llm_finalize, prefer_time_varying)"
            ),
            fillcolor="#FFFFFF",
        )

    g.edge("in_root", "cli")
    g.edge("out_root", "cli")
    g.edge("deps", "cli")
    g.edge("cli", "discover")
    g.edge("discover", "loop")

    # --------------------------------------------------------------
    # Cluster: Per-profile pipeline
    # --------------------------------------------------------------
    with g.subgraph(name="cluster_profile") as w:
        w.attr(label="Per-profile readiness analysis (analyze_profile)", style="rounded", color="#777777")

        w.node(
            "load",
            label=H(
                "<B>P1) Load CSV robustly</B><BR ALIGN='LEFT'/>"
                "_read_csv_robust(): try separators [',',';','\\t','|']"
            ),
            fillcolor="#FFFFFF",
        )

        w.node(
            "time",
            label=H(
                "<B>P2) Time axis handling</B><BR ALIGN='LEFT'/>"
                "infer time/date columns → stable mergesort ordering<BR ALIGN='LEFT'/>"
                "assess regularity (modal step, irregular fraction) + duplicates"
            ),
            fillcolor="#FFFFFF",
        )

        w.node(
            "select_vars",
            label=H(
                "<B>P3) Select candidate variables</B><BR ALIGN='LEFT'/>"
                "Prefer columns starting with P/C; exclude time/id-like columns"
            ),
            fillcolor="#FFFFFF",
        )

        w.node(
            "coerce",
            label=H(
                "<B>P4) Coerce to numeric</B><BR ALIGN='LEFT'/>"
                "_coerce_numeric_series(); exclude columns with high non-numeric rate"
            ),
            fillcolor="#FFFFFF",
        )

        w.node(
            "qc",
            label=H(
                "<B>P5) Variable QC + diagnostics</B><BR ALIGN='LEFT'/>"
                "infer type (binary/ordinal/count/continuous/proportion)<BR ALIGN='LEFT'/>"
                "missingness + streaks, low variability, dominance, outliers (MAD z)<BR ALIGN='LEFT'/>"
                "optional: normality (scipy), stationarity ADF/KPSS + Ljung-Box (statsmodels)<BR ALIGN='LEFT'/>"
                "produce hard/soft drop recommendations + transform suggestions"
            ),
            fillcolor="#FFFFFF",
        )

        w.node(
            "corr",
            label=H(
                "<B>P6) Collinearity scan</B><BR ALIGN='LEFT'/>"
                "Choose corr method (Pearson/Spearman) from diagnostics<BR ALIGN='LEFT'/>"
                "Find abs(r) ≥ 0.95 pairs → drop worse-quality variable"
            ),
            fillcolor="#FFFFFF",
        )

        w.node(
            "feas",
            label=H(
                "<B>P7) Tier feasibility</B><BR ALIGN='LEFT'/>"
                "Tier 1/2: pairwise complete-N summary (median/q25)<BR ALIGN='LEFT'/>"
                "Tier 3: lagged effective N (strict + per-variable; use q25)<BR ALIGN='LEFT'/>"
                f"STATIC floor ~ q25 ≥ {TIER3_STATIC_Q25_FLOOR}, TV floor ~ q25 ≥ {TIER3_TV_Q25_FLOOR}<BR ALIGN='LEFT'/>"
                "TV feasibility also requires multiple windows (smoothing/window heuristic)"
            ),
            fillcolor="#FFFFFF",
        )

        w.node(
            "decide",
            label=H(
                "<B>P8) Decide Tier + Tier3 variant</B><BR ALIGN='LEFT'/>"
                "Prefer TIME_VARYING_gVAR when feasible (flag)<BR ALIGN='LEFT'/>"
                "Stationarity: required mainly for STATIC_gVAR; NOT a blocker for TIME_VARYING_gVAR"
            ),
            fillcolor="#FFFFFF",
        )

        w.node(
            "score",
            label=H(
                "<B>P9) Score + summaries</B><BR ALIGN='LEFT'/>"
                "Score components: sample size, missingness, variable quality, time regularity, assumptions<BR ALIGN='LEFT'/>"
                "Assumptions weighting shifts: stationarity emphasized only for STATIC Tier 3<BR ALIGN='LEFT'/>"
                "Generate technical_summary + client_friendly_summary + next_steps + caveats"
            ),
            fillcolor="#FFFFFF",
        )

        w.node(
            "report",
            label=H(
                "<B>P10) Build report dict</B><BR ALIGN='LEFT'/>"
                "meta + dataset_overview + thresholds + variables + tier evidence + overall decision"
            ),
            fillcolor="#FFFFFF",
        )

        w.node(
            "llm",
            label=H(
                "<B>P11) Optional LLM finalization</B><BR ALIGN='LEFT'/>"
                f"model=<I>{LLM_MODEL}</I><BR ALIGN='LEFT'/>"
                "condense payload → prompt → OpenAI call (new or legacy)<BR ALIGN='LEFT'/>"
                "allowed: conservative tier/variant override + score adjust [-10,+10] + improved summaries"
            ),
            fillcolor="#FFFFFF",
        )

        w.node(
            "write",
            label=H(
                "<B>P12) Write outputs</B><BR ALIGN='LEFT'/>"
                "output_root/PROFILE_ID/: readiness_report.json + readiness_summary.txt"
            ),
            fillcolor="#FFFFFF",
        )

        w.node(
            "ret",
            label=H(
                "<B>P13) Return</B> report to main loop<BR ALIGN='LEFT'/>"
                "main prints tier/variant/score + summaries; counts ok/fail"
            ),
            fillcolor="#FFFFFF",
        )

        w.edge("load", "time")
        w.edge("time", "select_vars")
        w.edge("select_vars", "coerce")
        w.edge("coerce", "qc")
        w.edge("qc", "corr")
        w.edge("corr", "feas")
        w.edge("feas", "decide")
        w.edge("decide", "score")
        w.edge("score", "report")
        w.edge("report", "llm", style="dashed", xlabel="if enabled")
        w.edge("llm", "write", style="dashed", xlabel="LLM ok")
        w.edge("report", "write", xlabel="base path")
        w.edge("write", "ret")

    g.edge("loop", "load", lhead="cluster_profile", xlabel="per profile")

    # --------------------------------------------------------------
    # Cluster: End
    # --------------------------------------------------------------
    with g.subgraph(name="cluster_done") as o:
        o.attr(label="End", style="rounded", color="#777777")

        o.node(
            "done",
            label=H(
                "<B>DONE</B><BR ALIGN='LEFT'/>"
                "Return exit code 0 if all succeeded; else 3<BR ALIGN='LEFT'/>"
                "Stores outputs per profile for downstream analysis"
            ),
            fillcolor="#FFFFFF",
        )

        o.node(
            "legend",
            label=H(
                "<B>Legend</B><BR ALIGN='LEFT'/>"
                "solid = always<BR ALIGN='LEFT'/>"
                "dashed = optional (LLM finalize)"
            ),
            shape="note",
            fillcolor="#FFFFFF",
            color="#888888",
        )

    g.edge("ret", "done")
    g.edge("done", "legend", style="invis")

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
