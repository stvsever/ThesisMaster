#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""create_uml_diagram_construct_observation_model_elaborate.py

Creates a publication-ready UML-style flow diagram (Graphviz/DOT) for:

    01_construct_observation_model.py

Design goals (mirrors your previous diagram generator):
  - Vertical, compact layout (avoids big whitespace from parallel branches).
  - Collapses optional/looping sub-steps into single blocks with clear notes.
  - Highlights strict-JSON + validation + auto-repair + deterministic-fix backbone.

Outputs:
  - PNG diagram (bounded physical size, DPI-controlled)
  - Optional SVG (recommended for papers/LaTeX)

Requirements:
  - system graphviz: brew install graphviz | apt-get install graphviz
  - python: pip install graphviz

Run (local):
  python create_uml_diagram_construct_observation_model_elaborate.py \
    --out "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/SystemComponents/Agentic_Framework/02_ConstructionInitialObservationModel/utils/UML_diagrams/UML_01_construct_observation_model_elaborate.png" \
    --svg "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/SystemComponents/Agentic_Framework/02_ConstructionInitialObservationModel/utils/UML_diagrams/UML_01_construct_observation_model_elaborate.svg" \
    --dpi 600 --page "11,14"

"""

from __future__ import annotations

import os
import argparse
from graphviz import Digraph


# ---------------------------------------------------------------------
# CONFIG SNAPSHOT (keep aligned with 01_construct_observation_model.py)
# ---------------------------------------------------------------------

# LLM model (script default)
LLM_MODEL_DEFAULT = os.environ.get("OBS_LLM_MODEL", "gpt-5-mini")

# Prompt size controls (defaults)
PROMPT_TOP_HYDE = int(os.environ.get("OBS_PROMPT_TOP_HYDE", "60"))
PROMPT_TOP_MAPPING_GLOBAL = int(os.environ.get("OBS_PROMPT_TOP_MAPPING_GLOBAL", "60"))
PROMPT_TOP_MAPPING_PER_CRIT = int(os.environ.get("OBS_PROMPT_TOP_MAPPING_PER_CRITERION", "10"))

# Variable selection defaults
N_CRITERIA_DEFAULT = os.environ.get("OBS_N_CRITERIA", "choice")
N_PREDICTORS_DEFAULT = os.environ.get("OBS_N_PREDICTORS", "choice")

# Execution defaults
MAX_WORKERS_DEFAULT = int(os.environ.get("OBS_MAX_WORKERS", "60"))
MAX_ONTOLOGY_CHARS_DEFAULT = int(os.environ.get("OBS_MAX_ONTOLOGY_CHARS", "120000"))
USE_CACHE_DEFAULT = os.environ.get("OBS_USE_CACHE", "true").strip().lower() in {"1", "true", "yes"}

# Repair defaults
AUTO_REPAIR_DEFAULT = os.environ.get("OBS_AUTO_REPAIR", "true").strip().lower() in {"1", "true", "yes"}
MAX_REPAIR_ATTEMPTS_DEFAULT = int(os.environ.get("OBS_MAX_REPAIR_ATTEMPTS", "3"))

# Deterministic fix defaults
DETERMINISTIC_FIX_DEFAULT = os.environ.get("OBS_DETERMINISTIC_FIX", "true").strip().lower() in {"1", "true", "yes"}
MAX_FIX_PASSES_DEFAULT = int(os.environ.get("OBS_MAX_FIX_PASSES", "5"))

# Sampling defaults
ENABLE_SAMPLING_DEFAULT = os.environ.get("OBS_ENABLE_SAMPLING", "true").strip().lower() in {"1", "true", "yes"}
SAMPLE_N_DEFAULT = int(os.environ.get("OBS_SAMPLE_N", "3"))
SAMPLE_SEED_DEFAULT = int(os.environ.get("OBS_SAMPLE_SEED", "42"))


# Your requested default output directory (local machine). In this environment you can override with --out.
DEFAULT_OUT = os.environ.get(
    "UML_OUT",
    "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/SystemComponents/Agentic_Framework/"
    "02_ConstructionInitialObservationModel/utils/UML_diagrams/UML_01_construct_observation_model_elaborate.png",
)


# ---------------------------------------------------------------------
# Diagram builder
# ---------------------------------------------------------------------

def build_diagram(dpi: int, page_in: str) -> Digraph:
    """Builds a compact vertical DOT graph."""

    g = Digraph("Construct_Observation_Model_UML", engine="dot")

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
        label="01_construct_observation_model.py — Initial Criterion+Predictor Observation Model (gVAR-ready)",
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
    # Cluster: Inputs
    # --------------------------------------------------------------
    with g.subgraph(name="cluster_inputs") as c:
        c.attr(label="Inputs (per run)", style="rounded", color="#777777")

        c.node(
            "in_mapped",
            label=H(
                "<B>mapped_criterions.csv</B><BR ALIGN='LEFT'/>"
                "LLM-decomposed criterion vars + mapped criterion leaf/path<BR ALIGN='LEFT'/>"
                "Per pseudoprofile_id: complaint_text, variables[], query_text_used"
            ),
            shape="note",
            fillcolor="#FFFFFF",
        )

        c.node(
            "in_hyde",
            label=H(
                "<B>dense_profiles.csv</B> (HyDe global ranks)<BR ALIGN='LEFT'/>"
                "summary, solutions_compact, global_###_path_embedtext + scores"
            ),
            shape="note",
            fillcolor="#FFFFFF",
        )

        c.node(
            "in_mapping",
            label=H(
                "<B>all_pseudoprofiles__predictor_ranks_dense.csv</B><BR ALIGN='LEFT'/>"
                "LLM-based criterion→predictor rank signals<BR ALIGN='LEFT'/>"
                "parts: pre_global, post_global, post_per_criterion"
            ),
            shape="note",
            fillcolor="#FFFFFF",
        )

        c.node(
            "in_ontology",
            label=H(
                "<B>predictors_list.txt</B><BR ALIGN='LEFT'/>"
                "High-level BioPsychoSocial predictor ontology overview<BR ALIGN='LEFT'/>"
                f"max chars: {MAX_ONTOLOGY_CHARS_DEFAULT:,} (truncate head+tail)"
            ),
            shape="note",
            fillcolor="#FFFFFF",
        )

    # --------------------------------------------------------------
    # Cluster: Run initialization
    # --------------------------------------------------------------
    with g.subgraph(name="cluster_startup") as s:
        s.attr(label="Run initialization (once)", style="rounded", color="#777777")

        s.node(
            "env",
            label=H(
                "<B>S0) Load environment</B><BR ALIGN='LEFT'/>"
                "• dotenv (.env) optional<BR ALIGN='LEFT'/>"
                "• require OPENAI_API_KEY"
            ),
            fillcolor="#FFFFFF",
        )

        s.node(
            "cli",
            label=H(
                "<B>S1) Parse CLI args</B><BR ALIGN='LEFT'/>"
                f"• llm_model={LLM_MODEL_DEFAULT}<BR ALIGN='LEFT'/>"
                f"• n_criteria={N_CRITERIA_DEFAULT}, n_predictors={N_PREDICTORS_DEFAULT}<BR ALIGN='LEFT'/>"
                f"• prompt_top: hyde={PROMPT_TOP_HYDE}, mapping_global={PROMPT_TOP_MAPPING_GLOBAL}, per_criterion={PROMPT_TOP_MAPPING_PER_CRIT}<BR ALIGN='LEFT'/>"
                f"• max_workers={MAX_WORKERS_DEFAULT}, use_cache={USE_CACHE_DEFAULT}<BR ALIGN='LEFT'/>"
                f"• auto_repair={AUTO_REPAIR_DEFAULT} (max={MAX_REPAIR_ATTEMPTS_DEFAULT})<BR ALIGN='LEFT'/>"
                f"• deterministic_fix={DETERMINISTIC_FIX_DEFAULT} (passes={MAX_FIX_PASSES_DEFAULT})"
            ),
            fillcolor="#FFFFFF",
        )

        s.node(
            "run_id",
            label=H(
                "<B>S2) Determine run_id + create run dirs</B><BR ALIGN='LEFT'/>"
                "• run_id = CLI or timestamp<BR ALIGN='LEFT'/>"
                "• run_root = results_dir/runs/run_id<BR ALIGN='LEFT'/>"
                "• profiles_dir = run_root/profiles"
            ),
        )

        s.node(
            "save_run_cfg",
            label=H(
                "<B>S3) Save run config</B><BR ALIGN='LEFT'/>"
                "run_root/config.json (paths + params)"
            ),
            fillcolor="#FFFFFF",
        )

        s.node(
            "select_ids",
            label=H(
                "<B>S4) Select pseudoprofile_id list</B><BR ALIGN='LEFT'/>"
                "• If CLI pseudoprofile_id: single profile<BR ALIGN='LEFT'/>"
                "• Else list from mapped_criterions.csv<BR ALIGN='LEFT'/>"
                f"• Optional sampling: enabled={ENABLE_SAMPLING_DEFAULT}, n={SAMPLE_N_DEFAULT}, seed={SAMPLE_SEED_DEFAULT}"
            ),
        )

    g.edge("in_mapped", "env", xlabel="paths")
    g.edge("in_hyde", "env")
    g.edge("in_mapping", "env")
    g.edge("in_ontology", "env")

    g.edge("env", "cli")
    g.edge("cli", "run_id")
    g.edge("run_id", "save_run_cfg")
    g.edge("save_run_cfg", "select_ids")

    # --------------------------------------------------------------
    # Cluster: Parallel execution
    # --------------------------------------------------------------
    with g.subgraph(name="cluster_parallel") as p:
        p.attr(label="Parallel execution", style="rounded", color="#777777")

        p.node(
            "threadpool",
            label=H(
                "<B>P0) ThreadPoolExecutor</B><BR ALIGN='LEFT'/>"
                f"max_workers = {MAX_WORKERS_DEFAULT}<BR ALIGN='LEFT'/>"
                "submit(process_one_pseudoprofile) for each id<BR ALIGN='LEFT'/>"
                "collect futures via as_completed"
            ),
            fillcolor="#FFFFFF",
        )

        p.node(
            "worker",
            label=H(
                "<B>P1) process_one_pseudoprofile(pid)</B><BR ALIGN='LEFT'/>"
                "Per-profile pipeline (see next cluster)"
            ),
        )

        p.node(
            "aggregate",
            label=H(
                "<B>P2) Aggregate results</B><BR ALIGN='LEFT'/>"
                "• models_ok[] (final JSON models)<BR ALIGN='LEFT'/>"
                "• errors_rows[] (worker failures)<BR ALIGN='LEFT'/>"
                "• long-format row buffers (variables, edges, relevance, validations)"
            ),
            fillcolor="#FFFFFF",
        )

    g.edge("select_ids", "threadpool", xlabel="pseudoprofile_ids")
    g.edge("threadpool", "worker", xlabel="futures")
    g.edge("worker", "aggregate", xlabel="res")

    # --------------------------------------------------------------
    # Cluster: Per-profile worker pipeline (collapsed vertical)
    # --------------------------------------------------------------
    with g.subgraph(name="cluster_worker") as w:
        w.attr(label="Per-profile pipeline (process_one_pseudoprofile)", style="rounded", color="#777777")

        w.node(
            "w_cfg",
            label=H(
                "<B>W0) Setup profile dir + config</B><BR ALIGN='LEFT'/>"
                "profiles/pid/config.json<BR ALIGN='LEFT'/>"
                "cache paths: input_payload.json, llm_raw.json, llm_final.json, validation_report.json"
            ),
            fillcolor="#FFFFFF",
        )

        w.node(
            "w_cache",
            label=H(
                "<B>W1) Cache fast-path</B> (optional)<BR ALIGN='LEFT'/>"
                "if use_cache and llm_final exists:<BR ALIGN='LEFT'/>"
                "• load model<BR ALIGN='LEFT'/>"
                "• load validation_report else validate now"
            ),
        )

        w.node(
            "w_load_inputs",
            label=H(
                "<B>W2) Load inputs (if no cache)</B><BR ALIGN='LEFT'/>"
                "• load_profile_input(mapped_criterions, predictors_list, pid)<BR ALIGN='LEFT'/>"
                "• load_hyde_signals_for_profile(dense_profiles, pid, top_n)<BR ALIGN='LEFT'/>"
                "• load_llm_mapping_ranks_for_profile(mapping_ranks, pid, top_global, top_per_criterion)"
            ),
            fillcolor="#FFFFFF",
        )

        w.node(
            "w_build_prompt",
            label=H(
                "<B>W3) Build LLM payload + prompt</B><BR ALIGN='LEFT'/>"
                "build_llm_messages():<BR ALIGN='LEFT'/>"
                "• CASE_PAYLOAD_JSON includes complaint + criteria operationalization + ontology overview + signals<BR ALIGN='LEFT'/>"
                "• guidance: node cap&lt;=18, dense grids complete, sparse edges copy dense values, coherent sampling<BR ALIGN='LEFT'/>"
                "persist input_payload.json (extract JSON from messages[0].content)"
            ),
        )

        w.node(
            "w_llm",
            label=H(
                "<B>W4) LLM structured generation</B><BR ALIGN='LEFT'/>"
                f"Responses API: model=<I>{LLM_MODEL_DEFAULT}</I><BR ALIGN='LEFT'/>"
                "• strict JSON Schema (json_schema; strict=true)<BR ALIGN='LEFT'/>"
                "• DO NOT pass temperature or max_output_tokens<BR ALIGN='LEFT'/>"
                "output: observation model JSON (criteria+predictors + dense grids + sparse edges + design notes)"
            ),
        )

        w.node(
            "w_validate",
            label=H(
                "<B>W5) Validate</B> validate_observation_model()<BR ALIGN='LEFT'/>"
                "checks:<BR ALIGN='LEFT'/>"
                "• design coherence (study_days, prompts/day, expected totals)<BR ALIGN='LEFT'/>"
                "• variable ID uniqueness + measurement feasibility + sampling consistency<BR ALIGN='LEFT'/>"
                "• dense grids FULL + score formatting (comma-decimal 5dp)<BR ALIGN='LEFT'/>"
                "• sparse edges match dense cell values (numeric vs string) + no self-edges<BR ALIGN='LEFT'/>"
                "• optimality heuristics per target (avoid worse-than-available edges)"
            ),
            fillcolor="#FFFFFF",
        )

        w.node(
            "w_repair",
            label=H(
                "<B>W6) Auto-repair loop</B> (optional; LLM)<BR ALIGN='LEFT'/>"
                f"enabled={AUTO_REPAIR_DEFAULT}, max_attempts={MAX_REPAIR_ATTEMPTS_DEFAULT}<BR ALIGN='LEFT'/>"
                "If validator errors:<BR ALIGN='LEFT'/>"
                "• build_repair_messages(original_model + validation_report)<BR ALIGN='LEFT'/>"
                "• call LLM (same strict schema)<BR ALIGN='LEFT'/>"
                "• validate; stop if no errors"
            ),
        )

        w.node(
            "w_fix",
            label=H(
                "<B>W7) Deterministic fix layer</B> (optional; guaranteed pass)<BR ALIGN='LEFT'/>"
                f"enabled={DETERMINISTIC_FIX_DEFAULT}, max_passes={MAX_FIX_PASSES_DEFAULT}<BR ALIGN='LEFT'/>"
                "deterministic_fix_model():<BR ALIGN='LEFT'/>"
                "• enforce design coherence + minimum observations (≥30)<BR ALIGN='LEFT'/>"
                "• enforce sampling consistency per assessment_type<BR ALIGN='LEFT'/>"
                "• rebuild FULL dense grids deterministically (fill missing with 0,00000)<BR ALIGN='LEFT'/>"
                "• rebuild sparse edges from dense (top-k incoming per target) with exact score copies"
            ),
            fillcolor="#FFFFFF",
        )

        w.node(
            "w_save",
            label=H(
                "<B>W8) Persist profile artifacts</B><BR ALIGN='LEFT'/>"
                "profiles/pid/:<BR ALIGN='LEFT'/>"
                "• llm_observation_model_raw.json<BR ALIGN='LEFT'/>"
                "• llm_observation_model_final.json<BR ALIGN='LEFT'/>"
                "• validation_report.json"
            ),
            fillcolor="#FFFFFF",
        )

        w.node(
            "w_return",
            label=H(
                "<B>W9) Return result</B><BR ALIGN='LEFT'/>"
                "{status, used_cache, model, validation_report} or error bundle"
            ),
            fillcolor="#FFFFFF",
        )

        # Worker edges (vertical)
        w.edge("w_cfg", "w_cache", xlabel="W1")
        w.edge("w_cache", "w_load_inputs", xlabel="cache miss", style="dotted")
        w.edge("w_load_inputs", "w_build_prompt", xlabel="payload")
        w.edge("w_build_prompt", "w_llm", xlabel="W4")
        w.edge("w_llm", "w_validate", xlabel="W5")
        w.edge("w_validate", "w_repair", xlabel="if errors", style="dashed")
        w.edge("w_repair", "w_validate", xlabel="re-validate", style="dashed")
        w.edge("w_validate", "w_fix", xlabel="enforce pass", style="dashed")
        w.edge("w_fix", "w_validate", xlabel="re-validate", style="dashed")
        w.edge("w_validate", "w_save", xlabel="final model")
        w.edge("w_save", "w_return", xlabel="done")

    # Connect parallel worker node to detailed pipeline cluster
    g.edge("worker", "w_cfg", lhead="cluster_worker", xlabel="per pid")
    g.edge("w_return", "aggregate", ltail="cluster_worker", xlabel="ok/error")

    # --------------------------------------------------------------
    # Cluster: Run-level outputs
    # --------------------------------------------------------------
    with g.subgraph(name="cluster_outputs") as o:
        o.attr(label="Run-level outputs (after aggregation)", style="rounded", color="#777777")

        o.node(
            "out_jsonl",
            label=H(
                "<B>O1) observation_models.jsonl</B><BR ALIGN='LEFT'/>"
                "One JSON model per line (sorted by pseudoprofile_id)"
            ),
            fillcolor="#FFFFFF",
        )

        o.node(
            "out_long",
            label=H(
                "<B>O2) Long-format CSV exports</B><BR ALIGN='LEFT'/>"
                "variables_long.csv<BR ALIGN='LEFT'/>"
                "edges_long.csv, edges_pp_long.csv, edges_cc_long.csv<BR ALIGN='LEFT'/>"
                "predictor_criterion_relevance_long.csv<BR ALIGN='LEFT'/>"
                "predictor_predictor_relevance_long.csv<BR ALIGN='LEFT'/>"
                "criterion_criterion_relevance_long.csv"
            ),
            fillcolor="#FFFFFF",
        )

        o.node(
            "out_reports",
            label=H(
                "<B>O3) QA summaries</B><BR ALIGN='LEFT'/>"
                "validations.csv (errors/warnings + stats)<BR ALIGN='LEFT'/>"
                "errors.csv (worker exceptions)"
            ),
            fillcolor="#FFFFFF",
        )

        o.node(
            "next",
            label=H(
                "<B>Next pipeline steps (manual)</B><BR ALIGN='LEFT'/>"
                "1) ontology mapper<BR ALIGN='LEFT'/>"
                "2) bi-partite visualizer<BR ALIGN='LEFT'/>"
                "3) pseudodata generator<BR ALIGN='LEFT'/>"
                "4) pseudodata visualizor"
            ),
            shape="note",
            fillcolor="#FFFFFF",
            color="#888888",
        )

        o.node(
            "legend",
            label=H(
                "<B>Legend</B><BR ALIGN='LEFT'/>"
                "solid = required step<BR ALIGN='LEFT'/>"
                "dashed = optional / loop (auto-repair, deterministic fix)<BR ALIGN='LEFT'/>"
                "dotted = conditional path (cache miss)"
            ),
            shape="note",
            fillcolor="#FFFFFF",
            color="#888888",
        )

    g.edge("aggregate", "out_jsonl", xlabel="write")
    g.edge("out_jsonl", "out_long", xlabel="write")
    g.edge("out_long", "out_reports", xlabel="write")
    g.edge("out_reports", "next", style="dashed", xlabel="follow-on")
    g.edge("next", "legend", style="invis")

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
