#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""create_uml_diagram_construct_observation_model_simple.py

Creates a slide-friendly *simple* UML-style flow diagram (Graphviz/DOT) for:

    01_construct_observation_model.py

Design goals
------------
- Minimal nodes + readable labels for a clear presentation slide.
- Correct high-level control flow:
    Inputs -> Run init -> Parallel execution -> Per-profile pipeline -> Aggregation -> Outputs
- Highlights the backbone:
    STRICT JSON Schema -> Validation -> Auto-repair (optional) -> Deterministic fix (guaranteed pass)

Outputs
-------
- PNG diagram (DPI-controlled)
- Optional SVG (recommended for slides/LaTeX)

Requirements
------------
- system graphviz: brew install graphviz | apt-get install graphviz
- python: pip install graphviz

Run (local)
-----------
python create_uml_diagram_construct_observation_model_simple.py \
  --out "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/SystemComponents/Agentic_Framework/02_ConstructionInitialObservationModel/utils/UML_diagrams/UML_01_construct_observation_model_simple.png" \
  --svg "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/SystemComponents/Agentic_Framework/02_ConstructionInitialObservationModel/utils/UML_diagrams/UML_01_construct_observation_model_simple.svg" \
  --dpi 600 --page "13.33,7.5"

"""

from __future__ import annotations

import os
import argparse
from graphviz import Digraph


# ---------------------------------------------------------------------
# CONFIG SNAPSHOT (keep aligned with 01_construct_observation_model.py)
# ---------------------------------------------------------------------

LLM_MODEL_DEFAULT = os.environ.get("OBS_LLM_MODEL", "gpt-5-mini")

PROMPT_TOP_HYDE = int(os.environ.get("OBS_PROMPT_TOP_HYDE", "60"))
PROMPT_TOP_MAPPING_GLOBAL = int(os.environ.get("OBS_PROMPT_TOP_MAPPING_GLOBAL", "60"))
PROMPT_TOP_MAPPING_PER_CRIT = int(os.environ.get("OBS_PROMPT_TOP_MAPPING_PER_CRITERION", "10"))

N_CRITERIA_DEFAULT = os.environ.get("OBS_N_CRITERIA", "choice")
N_PREDICTORS_DEFAULT = os.environ.get("OBS_N_PREDICTORS", "choice")

MAX_WORKERS_DEFAULT = int(os.environ.get("OBS_MAX_WORKERS", "60"))
MAX_ONTOLOGY_CHARS_DEFAULT = int(os.environ.get("OBS_MAX_ONTOLOGY_CHARS", "120000"))
USE_CACHE_DEFAULT = os.environ.get("OBS_USE_CACHE", "true").strip().lower() in {"1", "true", "yes"}

AUTO_REPAIR_DEFAULT = os.environ.get("OBS_AUTO_REPAIR", "true").strip().lower() in {"1", "true", "yes"}
MAX_REPAIR_ATTEMPTS_DEFAULT = int(os.environ.get("OBS_MAX_REPAIR_ATTEMPTS", "3"))

DETERMINISTIC_FIX_DEFAULT = os.environ.get("OBS_DETERMINISTIC_FIX", "true").strip().lower() in {"1", "true", "yes"}
MAX_FIX_PASSES_DEFAULT = int(os.environ.get("OBS_MAX_FIX_PASSES", "5"))

ENABLE_SAMPLING_DEFAULT = os.environ.get("OBS_ENABLE_SAMPLING", "true").strip().lower() in {"1", "true", "yes"}
SAMPLE_N_DEFAULT = int(os.environ.get("OBS_SAMPLE_N", "3"))
SAMPLE_SEED_DEFAULT = int(os.environ.get("OBS_SAMPLE_SEED", "42"))


# Your requested default output directory (local machine).
DEFAULT_OUT = os.environ.get(
    "UML_OUT",
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/SystemComponents/Agentic_Framework/"
    "02_ConstructionInitialObservationModel/utils/UML_diagrams/UML_01_construct_observation_model_simple.png",
)


# ---------------------------------------------------------------------
# Diagram builder
# ---------------------------------------------------------------------

def build_diagram(*, dpi: int, page_in: str) -> Digraph:
    """Builds a simple, slide-friendly DOT graph."""

    g = Digraph("Construct_Observation_Model_UML_SIMPLE", engine="dot")

    # Default to 16:9 slide canvas.
    page_in = (page_in or "").strip()
    if "," not in page_in:
        page_in = "13.33,7.5"
    W, H = [x.strip() for x in page_in.split(",", 1)]
    size_attr = f"{W},{H}!"

    g.attr(
        rankdir="LR",
        dpi=str(int(dpi)),
        size=size_attr,
        ratio="compress",
        charset="UTF-8",
        fontname="Helvetica",
        fontsize="14",
        labelloc="t",
        label="01_construct_observation_model.py — Simple flow (inputs → LLM → validation/repair → outputs)",
        splines="polyline",
        nodesep="0.40",
        ranksep="0.55",
        pad="0.22",
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
        fontsize="12",
        margin="0.15,0.10",
        penwidth="1.20",
    )

    g.attr(
        "edge",
        color="#444444",
        fontname="Helvetica",
        fontsize="11",
        arrowsize="0.85",
        penwidth="1.10",
    )

    def H(inner: str) -> str:
        return f"<{inner}>"

    # --------------------------------------------------------------
    # Inputs (collapsed)
    # --------------------------------------------------------------
    with g.subgraph(name="cluster_inputs") as c:
        c.attr(label="Inputs", style="rounded", color="#777777")

        c.node(
            "inputs",
            label=H(
                "<B>Inputs (per pseudoprofile)</B><BR ALIGN='LEFT'/>"
                "• mapped_criterions.csv (complaint + criterion vars)<BR ALIGN='LEFT'/>"
                "• dense_profiles.csv (HyDe predictor ranks)<BR ALIGN='LEFT'/>"
                "• predictor_ranks_dense.csv (LLM mapping ranks)<BR ALIGN='LEFT'/>"
                "• predictors_list.txt (BPS ontology; trunc @ max chars)"
            ),
            shape="note",
            fillcolor="#FFFFFF",
        )

    # --------------------------------------------------------------
    # Run init
    # --------------------------------------------------------------
    with g.subgraph(name="cluster_run") as r:
        r.attr(label="Run initialization", style="rounded", color="#777777")

        r.node(
            "init",
            label=H(
                "<B>Init</B><BR ALIGN='LEFT'/>"
                "dotenv (optional) + require OPENAI_API_KEY<BR ALIGN='LEFT'/>"
                "Parse CLI args + create run dirs + save config<BR ALIGN='LEFT'/>"
                "Select pseudoprofile_id list (optional sampling)"
            ),
            fillcolor="#FFFFFF",
        )

    # --------------------------------------------------------------
    # Parallel execution + per-profile pipeline
    # --------------------------------------------------------------
    with g.subgraph(name="cluster_parallel") as p:
        p.attr(label="Parallel execution", style="rounded", color="#777777")

        p.node(
            "parallel",
            label=H(
                "<B>ThreadPoolExecutor</B><BR ALIGN='LEFT'/>"
                f"max_workers={MAX_WORKERS_DEFAULT}<BR ALIGN='LEFT'/>"
                "process_one_pseudoprofile(pid) for each id"
            ),
            fillcolor="#FFFFFF",
        )

        p.node(
            "profile_pipeline",
            label=H(
                "<B>Per-profile pipeline</B><BR ALIGN='LEFT'/>"
                "1) Cache check (optional)<BR ALIGN='LEFT'/>"
                "2) Load inputs + build CASE_PAYLOAD_JSON<BR ALIGN='LEFT'/>"
                "3) LLM: strict JSON Schema output (no temp/max_tokens)<BR ALIGN='LEFT'/>"
                "4) Validate (design + IDs + FULL dense grids + edge consistency)<BR ALIGN='LEFT'/>"
                "5) Auto-repair loop (optional) + deterministic fix (guaranteed pass)<BR ALIGN='LEFT'/>"
                "6) Save per-profile artifacts + return result"
            ),
            fillcolor="#FFFFFF",
        )

    # --------------------------------------------------------------
    # Aggregation + outputs
    # --------------------------------------------------------------
    with g.subgraph(name="cluster_outputs") as o:
        o.attr(label="Aggregation & outputs", style="rounded", color="#777777")

        o.node(
            "aggregate",
            label=H(
                "<B>Aggregate results</B><BR ALIGN='LEFT'/>"
                "Collect ok models + errors<BR ALIGN='LEFT'/>"
                "Build long-format row buffers"
            ),
            fillcolor="#FFFFFF",
        )

        o.node(
            "exports",
            label=H(
                "<B>Write run-level exports</B><BR ALIGN='LEFT'/>"
                "observation_models.jsonl<BR ALIGN='LEFT'/>"
                "variables_long.csv + edges_*.csv + relevance_*.csv<BR ALIGN='LEFT'/>"
                "validations.csv + errors.csv"
            ),
            fillcolor="#FFFFFF",
        )

        o.node(
            "next",
            label=H(
                "<B>Next steps (manual)</B><BR ALIGN='LEFT'/>"
                "ontology mapper → bipartite visualizer → pseudodata gen → pseudodata viz"
            ),
            shape="note",
            fillcolor="#FFFFFF",
            color="#888888",
        )

    # --------------------------------------------------------------
    # Edges (simple left-to-right)
    # --------------------------------------------------------------
    g.edge("inputs", "init")
    g.edge("init", "parallel")
    g.edge("parallel", "profile_pipeline", label="per pid")
    g.edge("profile_pipeline", "aggregate")
    g.edge("aggregate", "exports")
    g.edge("exports", "next", style="dashed")

    # Small legend as footnote (invisible anchor to keep spacing stable)
    g.node(
        "legend",
        label=H(
            "<B>Legend</B><BR ALIGN='LEFT'/>"
            "dashed = follow-on / optional"
        ),
        shape="note",
        fillcolor="#FFFFFF",
        color="#FFFFFF",
        fontcolor="#777777",
        fontsize="10",
    )
    g.edge("next", "legend", style="invis")

    return g


def render(*, out_png: str, out_svg: str | None, dpi: int, page_in: str) -> tuple[str, str | None]:
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
    ap = argparse.ArgumentParser(description="Create a simple UML flow diagram for 01_construct_observation_model.py")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Output PNG path")
    ap.add_argument("--svg", default=os.environ.get("UML_SVG_OUT", ""), help="Optional SVG output path")
    ap.add_argument("--dpi", type=int, default=int(os.environ.get("UML_DPI", "600")), help="Raster DPI")
    ap.add_argument("--page", default=os.environ.get("UML_PAGE_IN", "13.33,7.5"), help='Page size in inches, "W,H"')
    args = ap.parse_args()

    svg_path = args.svg.strip() or None
    out_png, out_svg = render(out_png=args.out, out_svg=svg_path, dpi=args.dpi, page_in=args.page)

    print(f"[ok] PNG: {out_png}")
    if out_svg:
        print(f"[ok] SVG: {out_svg}")


if __name__ == "__main__":
    main()
