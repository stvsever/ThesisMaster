"""
Build the four multi-panel supplementary figures referenced by Appendix S1-S4.

S1: Extended ontology structural diagnostics       (6 panels)
S2: Multi-dimensional feasibility evaluation       (6 panels)
S3: Latency benchmark — detailed industrial view   (4 panels)
S4: HUA simulation — parameter-sweep heatmaps      (6 panels)

All four figures use the same visual grammar as the main-body figures: tight
gridspec layouts, dendrogram-mounted heatmaps where applicable, no per-panel
whitespace bleed, panel letters at upper-left, titles below each panel, and
matched colour palette.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "evaluation/supplementary_analyses/results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CMAP_DIV = LinearSegmentedColormap.from_list(
    "phoenix_div",
    ["#1f3a68", "#3b6ea8", "#cfe1f2", "#f7f7f7", "#fae0c1", "#e89c6a", "#c7693d"],
)
CMAP_SEQ = LinearSegmentedColormap.from_list(
    "phoenix_seq",
    ["#f7f7f7", "#dde9f4", "#8eb6d8", "#3b6ea8", "#1f3a68"],
)
METHOD_COLOR = {
    "tv-gVAR":         "#1f3a68",
    "stationary-gVAR": "#3b6ea8",
    "Ledoit-Wolf":     "#e89c6a",
}


# ---------------------------------------------------------------------------
# S1 — Extended ontology structural diagnostics
# ---------------------------------------------------------------------------

def _ontology_stats(json_path: Path) -> dict:
    """Compute per-node depth, branching, and leaf statistics from a nested-dict ontology."""
    if not json_path.exists():
        return {"depths": [], "branching": [], "leaves": 0, "internal": 0}
    data = json.loads(json_path.read_text())
    depths = []
    branching = []
    leaves = 0
    internal = 0

    def walk(node, d):
        nonlocal leaves, internal
        if not isinstance(node, dict) or not node:
            leaves += 1
            depths.append(d)
            return
        keys = list(node.keys())
        if keys:
            branching.append(len(keys))
            internal += 1
        depths.append(d)
        for k, v in node.items():
            walk(v, d + 1)

    if isinstance(data, dict):
        # Some ontology JSONs wrap content in a top-level key. Walk through children.
        for k, v in data.items():
            walk(v, 1)
    return {"depths": depths, "branching": branching,
            "leaves": leaves, "internal": internal}


SUBONTO_PATHS = {
    "CRITERION": REPO / "src/backend/SystemComponents/PHOENIX_ontology/separate/01_raw/CRITERION/steps/01_raw/aggregated/CRITERION_ontology.json",
    "PREDICTOR": REPO / "src/backend/SystemComponents/PHOENIX_ontology/separate/01_raw/PREDICTOR/steps/01_raw/aggregated/PREDICTOR_ontology.json",
    "PERSON":    REPO / "src/backend/SystemComponents/PHOENIX_ontology/separate/01_raw/PERSON/PERSON.json",
    "CONTEXT":   REPO / "src/backend/SystemComponents/PHOENIX_ontology/separate/01_raw/CONTEXT/CONTEXT.json",
    "HAPA":      REPO / "src/backend/SystemComponents/PHOENIX_ontology/separate/01_raw/HAPA/HAPA.json",
}
SUBONTO_COLOR = {
    "CRITERION": "#1f3a68",
    "PREDICTOR": "#3b6ea8",
    "PERSON":    "#5b8dc5",
    "CONTEXT":   "#e89c6a",
    "HAPA":      "#c7693d",
}


def build_S1() -> Path:
    """Six-panel native-redraw structural diagnostics of the PHOENIX ontology.

    All metrics are computed directly from the JSON sub-ontology files; no
    bitmap snapshots are reused, so the rendering is sharp at any zoom level.
    """
    sub_stats = {name: _ontology_stats(path) for name, path in SUBONTO_PATHS.items()}

    fig = plt.figure(figsize=(15.5, 10.5), dpi=160)
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        left=0.06, right=0.985, top=0.93, bottom=0.07,
        hspace=0.45, wspace=0.30,
    )

    # Panel A: leaf and internal-node counts per sub-ontology
    axA = fig.add_subplot(gs[0, 0])
    names = list(SUBONTO_PATHS.keys())
    x = np.arange(len(names))
    leaves = [sub_stats[n]["leaves"] for n in names]
    internal = [sub_stats[n]["internal"] for n in names]
    axA.bar(x - 0.18, leaves, 0.34, label="leaves",
            color=[SUBONTO_COLOR[n] for n in names], alpha=0.95, edgecolor="white")
    axA.bar(x + 0.18, internal, 0.34, label="internal",
            color=[SUBONTO_COLOR[n] for n in names], alpha=0.55, edgecolor="white")
    for i, (l, ip) in enumerate(zip(leaves, internal)):
        axA.text(i - 0.18, l + max(leaves) * 0.018, f"{l:,}", ha="center", fontsize=7)
        axA.text(i + 0.18, ip + max(leaves) * 0.018, f"{ip:,}", ha="center", fontsize=7)
    axA.set_xticks(x); axA.set_xticklabels(names, fontsize=8.5, rotation=20)
    axA.set_ylabel("Node count", fontsize=9)
    axA.set_title("A.  Leaf vs internal nodes per sub-ontology",
                  fontsize=9.5, fontweight="bold", loc="left", pad=4)
    axA.grid(axis="y", alpha=0.3)
    axA.spines["top"].set_visible(False); axA.spines["right"].set_visible(False)
    axA.legend(fontsize=8, framealpha=0.95, loc="upper right")

    # Panel B: depth distribution per sub-ontology (KDE-style histograms, normalized)
    axB = fig.add_subplot(gs[0, 1])
    max_depth = max((max(sub_stats[n]["depths"]) if sub_stats[n]["depths"] else 0)
                    for n in names)
    bins = np.arange(0, max_depth + 2) - 0.5
    for n in names:
        d = sub_stats[n]["depths"]
        if not d: continue
        axB.hist(d, bins=bins, density=True, alpha=0.45, label=n,
                 color=SUBONTO_COLOR[n], edgecolor="white", linewidth=0.4)
    axB.set_xlabel("Depth", fontsize=9); axB.set_ylabel("Density", fontsize=9)
    axB.set_title("B.  Depth distribution per sub-ontology",
                  fontsize=9.5, fontweight="bold", loc="left", pad=4)
    axB.legend(fontsize=8, framealpha=0.95, ncol=2)
    axB.grid(axis="y", alpha=0.3)
    axB.spines["top"].set_visible(False); axB.spines["right"].set_visible(False)

    # Panel C: branching factor distribution per sub-ontology (log-y)
    axC = fig.add_subplot(gs[0, 2])
    max_b = max((max(sub_stats[n]["branching"]) if sub_stats[n]["branching"] else 1)
                for n in names)
    bins_b = np.linspace(1, max_b + 1, 25)
    for n in names:
        b = sub_stats[n]["branching"]
        if not b: continue
        axC.hist(b, bins=bins_b, alpha=0.55, label=n,
                 color=SUBONTO_COLOR[n], edgecolor="white", linewidth=0.4)
    axC.set_yscale("log")
    axC.set_xlabel("Branching factor (children per internal node)", fontsize=9)
    axC.set_ylabel("Count (log)", fontsize=9)
    axC.set_title("C.  Branching distribution per sub-ontology",
                  fontsize=9.5, fontweight="bold", loc="left", pad=4)
    axC.legend(fontsize=8, framealpha=0.95)
    axC.grid(axis="y", which="both", alpha=0.3)
    axC.spines["top"].set_visible(False); axC.spines["right"].set_visible(False)

    # Panel D: leaf ratio per sub-ontology (bars, with annotation)
    axD = fig.add_subplot(gs[1, 0])
    ratios = []
    for n in names:
        total = sub_stats[n]["leaves"] + sub_stats[n]["internal"]
        ratios.append(sub_stats[n]["leaves"] / max(total, 1) * 100.0)
    bars = axD.bar(x, ratios, color=[SUBONTO_COLOR[n] for n in names],
                   alpha=0.88, edgecolor="white")
    for bar, r in zip(bars, ratios):
        axD.text(bar.get_x() + bar.get_width() / 2, r + 1, f"{r:.1f}%",
                 ha="center", fontsize=8)
    axD.set_xticks(x); axD.set_xticklabels(names, fontsize=8.5, rotation=20)
    axD.set_ylabel("Leaf fraction (%)", fontsize=9)
    axD.set_ylim(0, 100)
    axD.set_title("D.  Leaf ratio per sub-ontology",
                  fontsize=9.5, fontweight="bold", loc="left", pad=4)
    axD.grid(axis="y", alpha=0.3)
    axD.spines["top"].set_visible(False); axD.spines["right"].set_visible(False)

    # Panel E: cumulative depth curves (Lorenz-style)
    axE = fig.add_subplot(gs[1, 1])
    for n in names:
        d = sorted(sub_stats[n]["depths"])
        if not d: continue
        cum = np.arange(1, len(d) + 1) / len(d)
        axE.plot(d, cum, label=n, color=SUBONTO_COLOR[n], linewidth=1.6, alpha=0.9)
    axE.set_xlabel("Depth", fontsize=9)
    axE.set_ylabel("Cumulative fraction of nodes", fontsize=9)
    axE.set_title("E.  Cumulative depth curves",
                  fontsize=9.5, fontweight="bold", loc="left", pad=4)
    axE.legend(fontsize=8, framealpha=0.95)
    axE.grid(alpha=0.3)
    axE.spines["top"].set_visible(False); axE.spines["right"].set_visible(False)

    # Panel F: depth × branching scatter (per sub-ontology, jittered)
    axF = fig.add_subplot(gs[1, 2])
    rng = np.random.default_rng(7)
    for n in names:
        d = sub_stats[n]["depths"][: sub_stats[n]["internal"]] or []
        b = sub_stats[n]["branching"]
        m = min(len(d), len(b))
        if m == 0: continue
        d = np.array(d[:m]) + rng.uniform(-0.18, 0.18, m)
        b = np.array(b[:m]) + rng.uniform(-0.20, 0.20, m)
        axF.scatter(d, b, s=10, alpha=0.45, color=SUBONTO_COLOR[n],
                    edgecolor="white", linewidth=0.3, label=n)
    axF.set_yscale("log")
    axF.set_xlabel("Depth of parent node", fontsize=9)
    axF.set_ylabel("Children (log)", fontsize=9)
    axF.set_title("F.  Depth × branching scatter",
                  fontsize=9.5, fontweight="bold", loc="left", pad=4)
    axF.legend(fontsize=8, framealpha=0.95, ncol=2)
    axF.grid(alpha=0.3, which="both")
    axF.spines["top"].set_visible(False); axF.spines["right"].set_visible(False)

    # in-image suptitle removed (caption above the image is the title)
    out = OUT_DIR / "fig_S1_ontology_diagnostics.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok] S1 → {out.name}")
    return out


# ---------------------------------------------------------------------------
# S2 — Multi-dimensional feasibility evaluation
# ---------------------------------------------------------------------------

def build_S2() -> Path:
    """Per-dimension feasibility distributions + dimension covariance + top/bot rankings."""
    crit_csv = REPO / ("src/backend/utils/official/multi_dimensional_feasibility_evaluation/"
                       "CRITERIONS/results/summary/cluster_rankings.csv")
    pred_csv = REPO / ("src/backend/utils/official/multi_dimensional_feasibility_evaluation/"
                       "PREDICTORS/results/summary/predictor_rankings.csv")

    crit = pd.read_csv(crit_csv) if crit_csv.exists() else pd.DataFrame()
    pred = pd.read_csv(pred_csv) if pred_csv.exists() else pd.DataFrame()

    # Dimension columns
    crit_dims = [c for c in crit.columns if c.startswith("suitability.")]
    pred_dims = [c for c in pred.columns if c.startswith("suitability.")]

    fig = plt.figure(figsize=(15.5, 11.0), dpi=150)
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        left=0.06, right=0.985, top=0.95, bottom=0.06,
        hspace=0.45, wspace=0.40,        # widen wspace to prevent F-label overlap
    )

    # Panel A: per-dimension violin distribution — CRITERION
    axA = fig.add_subplot(gs[0, 0])
    if not crit.empty:
        data = [crit[d].dropna().values for d in crit_dims]
        labels = [d.replace("suitability.", "").replace("_", " ")[:18] for d in crit_dims]
        parts = axA.violinplot(data, positions=range(len(data)), showmeans=False,
                               showmedians=True, widths=0.7)
        for b in parts["bodies"]:
            b.set_facecolor("#3b6ea8"); b.set_alpha(0.65); b.set_edgecolor("#1f3a68")
        for k in ("cmins", "cmaxes", "cbars", "cmedians"):
            if k in parts: parts[k].set_color("#1f3a68")
        axA.set_xticks(range(len(labels)))
        axA.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
    axA.set_ylabel("Suitability score (0–1)", fontsize=9)
    axA.set_title("A.  CRITERION feasibility dimensions  (n = 100 clusters)",
                  fontsize=9.5, fontweight="bold", loc="left", pad=4)
    axA.grid(axis="y", alpha=0.3)
    axA.spines["top"].set_visible(False); axA.spines["right"].set_visible(False)
    axA.set_ylim(0, 1)

    # Panel B: per-dimension violin distribution — PREDICTOR
    axB = fig.add_subplot(gs[0, 1])
    if not pred.empty:
        data = [pred[d].dropna().values for d in pred_dims]
        labels = [d.replace("suitability.", "").replace("_", " ")[:18] for d in pred_dims]
        parts = axB.violinplot(data, positions=range(len(data)), showmeans=False,
                               showmedians=True, widths=0.7)
        for b in parts["bodies"]:
            b.set_facecolor("#e89c6a"); b.set_alpha(0.65); b.set_edgecolor("#c7693d")
        for k in ("cmins", "cmaxes", "cbars", "cmedians"):
            if k in parts: parts[k].set_color("#c7693d")
        axB.set_xticks(range(len(labels)))
        axB.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
    axB.set_ylabel("Suitability score (0–1)", fontsize=9)
    axB.set_title("B.  PREDICTOR feasibility dimensions  (n = 201 leaves)",
                  fontsize=9.5, fontweight="bold", loc="left", pad=4)
    axB.grid(axis="y", alpha=0.3)
    axB.spines["top"].set_visible(False); axB.spines["right"].set_visible(False)
    axB.set_ylim(0, 1)

    # Panel C: PREDICTOR overall suitability by BPS layer
    axC = fig.add_subplot(gs[0, 2])
    if not pred.empty and "layer" in pred.columns:
        for color, layer in zip(["#3b6ea8", "#e89c6a", "#1f3a68"], ["BIO", "PSYCHO", "SOCIAL"]):
            vals = pred[pred["layer"] == layer]["overall_suitability"].dropna().values
            if len(vals) == 0: continue
            axC.hist(vals, bins=25, alpha=0.55, label=f"{layer}  (n={len(vals)})",
                     color=color, edgecolor="white", linewidth=0.4)
    axC.set_xlabel("Overall suitability", fontsize=9)
    axC.set_ylabel("Number of predictor leaves", fontsize=9)
    axC.set_title("C.  PREDICTOR overall suitability by BPS layer",
                  fontsize=9.5, fontweight="bold", loc="left", pad=4)
    axC.legend(fontsize=8, framealpha=0.9)
    axC.grid(axis="y", alpha=0.3)
    axC.spines["top"].set_visible(False); axC.spines["right"].set_visible(False)

    # Panel D: dimension correlation matrix — CRITERION
    axD = fig.add_subplot(gs[1, 0])
    if not crit.empty and len(crit_dims) >= 2:
        corr = crit[crit_dims].corr().values
        labels = [d.replace("suitability.", "").replace("_", " ")[:12] for d in crit_dims]
        im = axD.imshow(corr, cmap=CMAP_DIV, vmin=-1, vmax=1)
        axD.set_xticks(range(len(labels)))
        axD.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
        axD.set_yticks(range(len(labels)))
        axD.set_yticklabels(labels, fontsize=7)
        for i in range(len(labels)):
            for j in range(len(labels)):
                v = corr[i, j]
                axD.text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=6.5, color="white" if abs(v) > 0.55 else "#222")
        fig.colorbar(im, ax=axD, fraction=0.045, pad=0.04)
    axD.set_title("D.  CRITERION dimension correlations  (Pearson r)",
                  fontsize=9.5, fontweight="bold", loc="left", pad=4)

    # Panel E: dimension correlation matrix — PREDICTOR
    axE = fig.add_subplot(gs[1, 1])
    if not pred.empty and len(pred_dims) >= 2:
        corr = pred[pred_dims].corr().values
        labels = [d.replace("suitability.", "").replace("_", " ")[:12] for d in pred_dims]
        im = axE.imshow(corr, cmap=CMAP_DIV, vmin=-1, vmax=1)
        axE.set_xticks(range(len(labels)))
        axE.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
        axE.set_yticks(range(len(labels)))
        axE.set_yticklabels(labels, fontsize=7)
        for i in range(len(labels)):
            for j in range(len(labels)):
                v = corr[i, j]
                axE.text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=6.5, color="white" if abs(v) > 0.55 else "#222")
        fig.colorbar(im, ax=axE, fraction=0.045, pad=0.04)
    axE.set_title("E.  PREDICTOR dimension correlations  (Pearson r)",
                  fontsize=9.5, fontweight="bold", loc="left", pad=4)

    # Panel F: top-10 and bottom-10 PREDICTOR suitability
    axF = fig.add_subplot(gs[1, 2])
    if not pred.empty:
        sorted_p = pred.sort_values("overall_suitability", ascending=False)
        top10 = sorted_p.head(10)[["label", "overall_suitability", "layer"]]
        bot10 = sorted_p.tail(10)[["label", "overall_suitability", "layer"]]
        combined = pd.concat([top10, bot10], ignore_index=True)
        combined["group"] = ["top"] * 10 + ["bot"] * 10
        y = np.arange(len(combined))
        layer_color = {"BIO": "#3b6ea8", "PSYCHO": "#e89c6a", "SOCIAL": "#1f3a68"}
        for idx, row in combined.iterrows():
            axF.barh(y[idx], row["overall_suitability"], color=layer_color.get(row["layer"], "#888"),
                     edgecolor="white", height=0.85)
        axF.set_yticks(y)
        axF.set_yticklabels([str(l)[:24] for l in combined["label"]], fontsize=6)
        axF.invert_yaxis()
        axF.axhline(9.5, color="#222", linewidth=0.6, linestyle="--")
        axF.text(0.97, 0.05, "TOP 10", transform=axF.transAxes, ha="right",
                  fontsize=8, color="#222", fontweight="bold")
        axF.text(0.97, 0.97, "BOTTOM 10", transform=axF.transAxes, ha="right",
                  va="top", fontsize=8, color="#222", fontweight="bold")
        axF.set_xlim(0, 1)
        axF.set_xlabel("Overall suitability", fontsize=9)
    axF.set_title("F.  Top/bottom-10 PREDICTOR leaves",
                  fontsize=9.5, fontweight="bold", loc="left", pad=4)
    axF.grid(axis="x", alpha=0.3)
    axF.spines["top"].set_visible(False); axF.spines["right"].set_visible(False)

    # in-image suptitle removed (caption above the image is the title)
    out = OUT_DIR / "fig_S2_feasibility.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok] S2 → {out.name}")
    return out


# ---------------------------------------------------------------------------
# S3 — Latency benchmark detail (industrial view)
# ---------------------------------------------------------------------------

def build_S3() -> Path:
    csv = REPO / "evaluation/ontology_evaluation/results/ontology_query_latency.csv"
    stats = json.loads((REPO / "evaluation/ontology_evaluation/results/ontology_query_latency_stats.json").read_text())
    df = pd.read_csv(csv)

    fig = plt.figure(figsize=(14.5, 10.5), dpi=150)
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        left=0.07, right=0.97, top=0.93, bottom=0.07,
        hspace=0.40, wspace=0.26,
    )

    backends = sorted(df["backend"].unique())
    op_short = {"top10_criteria_for_predictor": "Top-10 retrieval",
                "criterion_degree_for_predictor": "Degree count"}
    operations = sorted(df["operation"].unique())

    # Panel A: per-(operation, backend) boxplot on log-y, raw values in ms
    axA = fig.add_subplot(gs[0, 0])
    pos = 0
    xticks = []
    xtlabs = []
    cols = {"Adjacency dict (Python in-memory)": "#1f3a68", "SQL": "#3b6ea8", "SPARQL": "#e89c6a"}
    for op in operations:
        for b in backends:
            vals = df[(df["operation"] == op) & (df["backend"] == b)]["latency_ms"].values
            if len(vals) == 0: continue
            bp = axA.boxplot([vals], positions=[pos], widths=0.55, patch_artist=True,
                             showfliers=False,
                             boxprops={"facecolor": cols.get(b, "#999"), "alpha": 0.78,
                                       "edgecolor": "#222"},
                             medianprops={"color": "white", "linewidth": 1.4})
            xticks.append(pos); xtlabs.append(f"{b}\n{op_short.get(op, op)[:12]}")
            pos += 1
        pos += 0.6
    axA.set_yscale("log")
    axA.set_xticks(xticks); axA.set_xticklabels(xtlabs, fontsize=7, rotation=15)
    axA.set_ylabel("Latency (ms, log scale)", fontsize=9)
    axA.set_title("A.  Per-operation × backend latency distribution",
                  fontsize=10, fontweight="bold", loc="left")
    axA.grid(axis="y", which="both", alpha=0.3)
    axA.spines["top"].set_visible(False); axA.spines["right"].set_visible(False)

    # Panel B: percentile sweep p50/p90/p95/p99/p99.9 (lines)
    axB = fig.add_subplot(gs[0, 1])
    percs = [50, 90, 95, 99, 99.9]
    for b in backends:
        vals = df[df["backend"] == b]["latency_ms"].values
        if len(vals) == 0: continue
        ys = [np.percentile(vals, p) for p in percs]
        axB.plot(percs, ys, marker="o", color=cols.get(b, "#999"), linewidth=2,
                 label=b, alpha=0.9)
    axB.set_yscale("log")
    axB.set_xlabel("Percentile", fontsize=9)
    axB.set_ylabel("Latency (ms, log scale)", fontsize=9)
    axB.set_title("B.  Percentile-sweep tail behaviour",
                  fontsize=10, fontweight="bold", loc="left")
    axB.grid(alpha=0.3, which="both")
    axB.spines["top"].set_visible(False); axB.spines["right"].set_visible(False)
    axB.legend(fontsize=8, framealpha=0.9)

    # Panel C: throughput (qps) at median and at p95 — bars, log-y
    axC = fig.add_subplot(gs[1, 0])
    overall = stats["global"]["per_backend"]
    n_b = len(overall)
    x = np.arange(n_b)
    qmed = [overall[b]["throughput_qps_median"] for b in backends if b in overall]
    qp95 = [overall[b]["throughput_qps_p95"] for b in backends if b in overall]
    bw = 0.35
    axC.bar(x - bw/2, qmed, bw, label="QPS at median latency",
            color="#3b6ea8", alpha=0.85, edgecolor="white")
    axC.bar(x + bw/2, qp95, bw, label="QPS at p95 latency",
            color="#e89c6a", alpha=0.85, edgecolor="white")
    axC.set_yscale("log")
    axC.set_xticks(x); axC.set_xticklabels(backends, fontsize=8.5)
    axC.set_ylabel("Throughput (queries / second)", fontsize=9)
    axC.set_title("C.  Industrial-throughput at two operating points",
                  fontsize=10, fontweight="bold", loc="left")
    axC.grid(axis="y", which="both", alpha=0.3)
    axC.spines["top"].set_visible(False); axC.spines["right"].set_visible(False)
    axC.legend(fontsize=8, framealpha=0.9, loc="upper right")
    for i, (m, p) in enumerate(zip(qmed, qp95)):
        axC.text(i - bw/2, m * 1.2, f"{m:.0f}", ha="center", va="bottom", fontsize=7)
        axC.text(i + bw/2, p * 1.2, f"{p:.0f}", ha="center", va="bottom", fontsize=7)

    # Panel D: jitter density of warm-query latencies (μs scale)
    axD = fig.add_subplot(gs[1, 1])
    rng = np.random.default_rng(42)
    for i, b in enumerate(backends):
        vals = df[df["backend"] == b]["latency_ms"].values
        if len(vals) == 0: continue
        vals_us = vals * 1000.0
        xs = i + rng.uniform(-0.20, 0.20, size=len(vals_us))
        axD.scatter(xs, vals_us, color=cols.get(b, "#999"),
                    alpha=0.45, s=10, edgecolor="white", linewidth=0.3)
        axD.scatter(i, np.median(vals_us), color="black", marker="_", s=200, linewidth=2.0)
    axD.set_yscale("log")
    axD.set_xticks(range(len(backends)))
    axD.set_xticklabels(backends, fontsize=8.5)
    axD.set_ylabel("Latency (μs, log scale)", fontsize=9)
    axD.set_title("D.  Per-query density (1,296 warm queries; black bar = median)",
                  fontsize=10, fontweight="bold", loc="left")
    axD.grid(axis="y", which="both", alpha=0.3)
    axD.spines["top"].set_visible(False); axD.spines["right"].set_visible(False)

    # in-image suptitle removed (caption above the image is the title)
    out = OUT_DIR / "fig_S3_latency.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok] S3 → {out.name}")
    return out


# ---------------------------------------------------------------------------
# S4 — HUA simulation per-condition heatmaps
# ---------------------------------------------------------------------------

def build_S4() -> Path:
    csv = REPO / "evaluation/hua_evaluation/results/hua_study_a_per_run.csv"
    df = pd.read_csv(csv)
    regimes = ["stationary", "smooth", "abrupt", "periodic"]
    methods = ["tv-gVAR", "stationary-gVAR", "Ledoit-Wolf"]

    fig = plt.figure(figsize=(16.0, 11.0), dpi=150)
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        left=0.05, right=0.985, top=0.93, bottom=0.06,
        hspace=0.42, wspace=0.30,
    )

    # Panels A-C: per-method F1 heatmap with T on x and (p, noise) on y
    df["pn"] = df["p"].astype(str) + "_" + df["noise"].astype(str)
    pn_order = sorted(df["pn"].unique())
    pn_labels = [f"p={pn.split('_')[0]}, σ={pn.split('_')[1]}" for pn in pn_order]
    T_order = sorted(df["T"].unique())

    for i, m in enumerate(methods):
        ax = fig.add_subplot(gs[0, i])
        agg = df[df["method"] == m].pivot_table(
            index="pn", columns="T", values="f1", aggfunc="mean")
        agg = agg.reindex(index=pn_order, columns=T_order)
        im = ax.imshow(agg.values, cmap=CMAP_SEQ, aspect="auto", vmin=0, vmax=0.85)
        for r in range(agg.shape[0]):
            for c in range(agg.shape[1]):
                v = agg.values[r, c]
                if np.isnan(v): continue
                tc = "white" if v > 0.45 else "#222"
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=8, color=tc)
        ax.set_xticks(range(len(T_order)))
        ax.set_xticklabels(T_order, fontsize=8)
        ax.set_yticks(range(len(pn_order)))
        ax.set_yticklabels(pn_labels, fontsize=7.5)
        ax.set_xlabel("Sample size T", fontsize=8.5)
        ax.set_title(f"{chr(65+i)}.  {m} — F1 across (p × σ × T)",
                     fontsize=9.5, fontweight="bold", loc="left", pad=4)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    # Panel D: F1 by regime, all methods (grouped bar with SEM, not SD).
    # We use SEM (SD/sqrt(n)) so non-overlapping intervals genuinely reflect
    # estimator-mean differences rather than per-run variance, and we annotate
    # the bar heights so the tv-gVAR advantage in every non-stationary regime
    # is unambiguous to the reader.
    axD = fig.add_subplot(gs[1, 0])
    x = np.arange(len(regimes))
    width = 0.27
    for i, m in enumerate(methods):
        means = []
        sems = []
        for r in regimes:
            v = df[(df["method"] == m) & (df["regime"] == r)]["f1"].dropna().values
            means.append(np.mean(v))
            sems.append(np.std(v) / max(1, np.sqrt(len(v))))
        bars = axD.bar(x + (i - 1) * width, means, width, yerr=sems, capsize=2.5,
                       color=METHOD_COLOR[m], alpha=0.92, label=m, edgecolor="white",
                       error_kw=dict(ecolor="#333", lw=0.8))
        for xi, v in zip(x, means):
            axD.text(xi + (i - 1) * width, v + 0.012,
                     f"{v:.2f}", ha="center", fontsize=6.6, color="#222")
    # Mark which method wins each regime
    for j, r in enumerate(regimes):
        m_means = {m: df[(df["method"] == m) & (df["regime"] == r)]["f1"].mean()
                   for m in methods}
        winner = max(m_means, key=m_means.get)
        axD.text(x[j], max(m_means.values()) + 0.07,
                 f"best:\n{winner}", ha="center", va="bottom", fontsize=6.5,
                 color="#1f3a68", fontstyle="italic")
    axD.set_xticks(x); axD.set_xticklabels(regimes, fontsize=8.5)
    axD.set_ylabel("Edge-recovery F1 (mean ± SEM)", fontsize=9)
    axD.set_ylim(0, max([df[(df["method"] == m)]["f1"].mean() for m in methods]) * 1.6)
    axD.set_title("D.  Per-regime F1 (all methods; SEM whiskers, winner labelled)",
                  fontsize=10, fontweight="bold", loc="left")
    axD.grid(axis="y", alpha=0.3)
    axD.spines["top"].set_visible(False); axD.spines["right"].set_visible(False)
    axD.legend(fontsize=8, framealpha=0.9, loc="lower right")

    # Panel E: MSE by sample size (with separate lines per method)
    axE = fig.add_subplot(gs[1, 1])
    for m in methods:
        sub = df[df["method"] == m].groupby("T")["mse"].agg(["mean", "std"]).reset_index()
        axE.errorbar(sub["T"], sub["mean"], yerr=sub["std"], color=METHOD_COLOR[m],
                     marker="o", linewidth=1.8, capsize=3, alpha=0.9, label=m)
    axE.set_yscale("log")
    axE.set_xlabel("Sample size T", fontsize=9)
    axE.set_ylabel("Coefficient-recovery MSE (log)", fontsize=9)
    axE.set_title("E.  MSE scaling with sample size",
                  fontsize=10, fontweight="bold", loc="left")
    axE.grid(alpha=0.3, which="both")
    axE.spines["top"].set_visible(False); axE.spines["right"].set_visible(False)
    axE.legend(fontsize=8, framealpha=0.9)

    # Panel F: AUC heatmap (regime × method) — averaged
    axF = fig.add_subplot(gs[1, 2])
    auc_mat = np.zeros((len(methods), len(regimes)))
    for i, m in enumerate(methods):
        for j, r in enumerate(regimes):
            v = df[(df["method"] == m) & (df["regime"] == r)]["auc"].dropna().values
            auc_mat[i, j] = np.mean(v) if len(v) else np.nan
    im = axF.imshow(auc_mat, cmap=CMAP_SEQ, vmin=0.5, vmax=1.0)
    axF.set_xticks(range(len(regimes)))
    axF.set_xticklabels(regimes, fontsize=8.5)
    axF.set_yticks(range(len(methods)))
    axF.set_yticklabels(methods, fontsize=8.5)
    for i in range(len(methods)):
        for j in range(len(regimes)):
            v = auc_mat[i, j]
            tc = "white" if v > 0.78 else "#222"
            axF.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color=tc)
    axF.set_title("F.  Edge-ranking AUC (mean by regime × method)",
                  fontsize=10, fontweight="bold", loc="left")
    fig.colorbar(im, ax=axF, fraction=0.04, pad=0.02)

    # in-image suptitle removed (caption above the image is the title)
    out = OUT_DIR / "fig_S4_hua_parameter_sweep.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok] S4 → {out.name}")
    return out


def build_S5() -> Path:
    """Vertical-migration audit per-domain + cross-ontology mapping coverage."""
    # ---- 1. parse vertical-migration report ----
    vm_report = REPO / "src/backend/utils/official/vertical_migration_scheme/result/vertical_pair_consistency_report_20260114_201256.txt"
    domain_results = []  # list of (branch, domain, eval, violations, rate)
    if vm_report.exists():
        text = vm_report.read_text()
        pattern = re.compile(
            r"Selected domain subtree:\s*([A-Z]+)\s*/\s*([^\n]+)\n"
            r"Leaf count \(subtree\):\s*\d+\n"
            r"Vertical edges in subtree \(total\):\s*\d+\n"
            r"Vertical edges evaluated \(sampled\):\s*(\d+)\n"
            r"Violations:\s*(\d+)\s*\(([\d.]+)\)"
        )
        for m in pattern.finditer(text):
            branch, domain, evaluated, violations, rate = m.groups()
            domain_results.append({
                "branch": branch, "domain": domain.strip(),
                "evaluated": int(evaluated),
                "violations": int(violations),
                "rate": float(rate),
            })

    df_vm = pd.DataFrame(domain_results)

    # ---- 2. parse mapping summaries ----
    mapping_root = REPO / "src/backend/utils/official/overall_mapping_analyses/utils/clustering_analysis/results"
    mapping_rows = []
    for sub in sorted(p for p in mapping_root.iterdir() if p.is_dir()):
        f = sub / "results/summary.json"
        if not f.exists(): continue
        d = json.loads(f.read_text())
        mapping_rows.append(d)
    df_map = pd.DataFrame(mapping_rows)

    fig = plt.figure(figsize=(16.5, 11.5), dpi=160)
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        left=0.06, right=0.985, top=0.95, bottom=0.06,
        hspace=0.42, wspace=0.55,        # widen so panel-C horizontal-bar labels do not collide
    )

    # Panel A: per-branch summary bars (BIO / PSYCHO / SOCIAL)
    axA = fig.add_subplot(gs[0, 0])
    if not df_vm.empty:
        branch_stats = df_vm.groupby("branch").agg(
            evaluated=("evaluated", "sum"),
            violations=("violations", "sum"),
        ).reset_index()
        branch_stats["rate"] = branch_stats["violations"] / branch_stats["evaluated"] * 100
        colors = {"BIO": "#3b6ea8", "PSYCHO": "#e89c6a", "SOCIAL": "#1f3a68"}
        bars = axA.bar(branch_stats["branch"], branch_stats["rate"],
                       color=[colors.get(b, "#888") for b in branch_stats["branch"]],
                       edgecolor="white", alpha=0.88)
        for bar, rate, n in zip(bars, branch_stats["rate"], branch_stats["evaluated"]):
            axA.text(bar.get_x() + bar.get_width() / 2, rate + 0.04,
                     f"{rate:.2f}%\n(n={n:,})", ha="center", fontsize=8)
        axA.set_ylabel("Violation rate (%)", fontsize=9.5)
        axA.set_ylim(0, max(2.0, branch_stats["rate"].max() * 1.4))
    axA.set_title("A.  Vertical-migration violations by BPS branch",
                  fontsize=10.0, fontweight="bold", loc="left", pad=4)
    axA.grid(axis="y", alpha=0.3)
    axA.spines["top"].set_visible(False); axA.spines["right"].set_visible(False)

    # Panel B: per-domain violation rate scatter (size = # edges evaluated)
    axB = fig.add_subplot(gs[0, 1])
    if not df_vm.empty:
        colors = {"BIO": "#3b6ea8", "PSYCHO": "#e89c6a", "SOCIAL": "#1f3a68"}
        for branch, sub in df_vm.groupby("branch"):
            axB.scatter(sub["evaluated"], sub["rate"] * 100,
                        s=sub["evaluated"].clip(20, 300) * 0.6, alpha=0.55,
                        color=colors.get(branch, "#888"), label=branch,
                        edgecolor="white", linewidth=0.5)
        axB.set_xlabel("Edges evaluated in subtree", fontsize=9)
        axB.set_ylabel("Violation rate (%)", fontsize=9)
        axB.set_xscale("log")
        axB.axhline(1.0, color="#888", linestyle=":", linewidth=0.8)
        axB.text(axB.get_xlim()[1] * 0.85, 1.05, "1% reference",
                 fontsize=7, color="#666", ha="right")
        axB.legend(fontsize=8, framealpha=0.95)
    axB.set_title("B.  Per-domain audit (97 selected subtrees)",
                  fontsize=10.0, fontweight="bold", loc="left", pad=4)
    axB.grid(alpha=0.3, which="both")
    axB.spines["top"].set_visible(False); axB.spines["right"].set_visible(False)

    # Panel C: top-violating domains (horizontal bars)
    axC = fig.add_subplot(gs[0, 2])
    if not df_vm.empty:
        top = df_vm.sort_values("rate", ascending=False).head(12)
        colors = {"BIO": "#3b6ea8", "PSYCHO": "#e89c6a", "SOCIAL": "#1f3a68"}
        bars = axC.barh(range(len(top)), top["rate"] * 100,
                        color=[colors.get(b, "#888") for b in top["branch"]],
                        edgecolor="white", alpha=0.9)
        labels = [f"{r['branch']} / {str(r['domain'])[:32]}" for _, r in top.iterrows()]
        axC.set_yticks(range(len(top)))
        axC.set_yticklabels(labels, fontsize=7)
        axC.invert_yaxis()
        for bar, v in zip(bars, top["violations"]):
            axC.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
                     f"{v}", va="center", fontsize=7, color="#222")
        axC.set_xlabel("Violation rate (%)", fontsize=9)
    axC.set_title("C.  Top-12 domains by violation rate",
                  fontsize=10.0, fontweight="bold", loc="left", pad=4)
    axC.grid(axis="x", alpha=0.3)
    axC.spines["top"].set_visible(False); axC.spines["right"].set_visible(False)

    # Panel D: cross-mapping density (bars, log-y)
    axD = fig.add_subplot(gs[1, 0])
    if not df_map.empty:
        df_sorted = df_map.sort_values("density", ascending=False)
        short = {
            "predictor_to_criterion": "P → C",
            "context_to_predictor": "Ctx → P",
            "profile_to_predictor": "Pf → P",
            "barrier_to_predictor": "B → P",
            "profile_to_barrier": "Pf → B",
            "context_to_barrier": "Ctx → B",
            "coping_to_barrier": "Cop → B",
        }
        labels = [short.get(m, m) for m in df_sorted["mapping"]]
        bars = axD.bar(range(len(df_sorted)), df_sorted["density"] * 100,
                       color="#3b6ea8", edgecolor="white", alpha=0.88)
        for bar, dens in zip(bars, df_sorted["density"]):
            axD.text(bar.get_x() + bar.get_width() / 2, dens * 100 + 1.5,
                     f"{dens * 100:.1f}%", ha="center", fontsize=7.5)
        axD.set_xticks(range(len(labels)))
        axD.set_xticklabels(labels, fontsize=8)
        axD.set_ylabel("Edge density (% of cells with score > 0)", fontsize=9)
        axD.set_ylim(0, 110)
    axD.set_title("D.  Cross-mapping edge density",
                  fontsize=10.0, fontweight="bold", loc="left", pad=4)
    axD.grid(axis="y", alpha=0.3)
    axD.spines["top"].set_visible(False); axD.spines["right"].set_visible(False)

    # Panel E: scale comparison (n_left × n_right, log-log, bubble = nnz)
    axE = fig.add_subplot(gs[1, 1])
    if not df_map.empty:
        sizes = df_map["nnz"].clip(20, 8000) / 30.0
        for _, r in df_map.iterrows():
            axE.scatter(r["n_left"], r["n_right"], s=sizes[r.name],
                        alpha=0.6, color="#3b6ea8", edgecolor="white", linewidth=0.6)
            axE.annotate(short.get(r["mapping"], r["mapping"]),
                         (r["n_left"], r["n_right"]),
                         textcoords="offset points", xytext=(6, 4),
                         fontsize=7, color="#222")
        axE.set_xscale("log"); axE.set_yscale("log")
        axE.set_xlabel("Source nodes  (n_left)", fontsize=9)
        axE.set_ylabel("Target nodes  (n_right)", fontsize=9)
    axE.set_title("E.  Mapping scale  (bubble area ∝ non-zero cells)",
                  fontsize=10.0, fontweight="bold", loc="left", pad=4)
    axE.grid(alpha=0.3, which="both")
    axE.spines["top"].set_visible(False); axE.spines["right"].set_visible(False)

    # Panel F: mean non-zero score per mapping (horizontal bars)
    axF = fig.add_subplot(gs[1, 2])
    if not df_map.empty:
        df_sorted = df_map.sort_values("score_mean_nonzero")
        labels = [short.get(m, m) for m in df_sorted["mapping"]]
        bars = axF.barh(range(len(df_sorted)), df_sorted["score_mean_nonzero"],
                        color="#1f3a68", edgecolor="white", alpha=0.88)
        for bar, v in zip(bars, df_sorted["score_mean_nonzero"]):
            axF.text(bar.get_width() + 10, bar.get_y() + bar.get_height() / 2,
                     f"{v:.0f}", va="center", fontsize=8)
        axF.set_yticks(range(len(labels)))
        axF.set_yticklabels(labels, fontsize=8)
        axF.set_xlim(0, 750)
        axF.set_xlabel("Mean non-zero score (0–1000)", fontsize=9)
    axF.set_title("F.  Mean non-zero edge score per mapping",
                  fontsize=10.0, fontweight="bold", loc="left", pad=4)
    axF.grid(axis="x", alpha=0.3)
    axF.spines["top"].set_visible(False); axF.spines["right"].set_visible(False)

    # in-image suptitle removed (caption above the image is the title)
    out = OUT_DIR / "fig_S5_vertical_and_coverage.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok] S5 → {out.name}")
    return out


def main():
    print(f"Output: {OUT_DIR}")
    build_S1()
    build_S2()
    build_S3()
    build_S4()
    build_S5()
    print("Done.")


if __name__ == "__main__":
    main()
