#!/usr/bin/env python3
"""Benchmark ontology-mediated mapping queries across local representations.

The methods chapter describes latency checks for ontology-driven queries. This
script makes that evaluation reproducible from the local PHOENIX artefacts by
running equivalent predictor-to-criterion mapping queries through three
representations:

* RDFLib SPARQL over a generated mapping graph.
* SQLite SQL as a local relational proxy for PostgreSQL-style queries.
* In-memory adjacency lists as a local graph/Cypher-style proxy.

The benchmark intentionally excludes load/build time and measures warm query
execution after each representation has been initialized.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sqlite3
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, XSD


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EDGE_CSV = (
    REPO_ROOT
    / "src/backend/utils/official/overall_mapping_analyses/utils/clustering_analysis/"
    "results/predictor_to_criterion/data/edges_raw.csv"
)
DEFAULT_OUT_DIR = REPO_ROOT / "evaluation/ontology_evaluation/results"

PHX = Namespace("https://phoenix.local/ontology-evaluation/")


@dataclass(frozen=True)
class QueryCase:
    operation: str
    predictor_id: str
    predictor_full_path: str


def slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_")
    return cleaned or "missing"


def load_positive_edges(edge_csv: Path) -> pd.DataFrame:
    edges = pd.read_csv(edge_csv)
    expected = {"predictor_id", "predictor_full_path", "cluster_id", "score"}
    missing = expected.difference(edges.columns)
    if missing:
        raise ValueError(f"Missing expected columns in {edge_csv}: {sorted(missing)}")

    edges = edges.copy()
    edges["predictor_id"] = edges["predictor_id"].astype(str)
    edges["cluster_id"] = edges["cluster_id"].astype(str)
    edges["score"] = pd.to_numeric(edges["score"], errors="coerce").fillna(0.0)
    return edges[edges["score"] > 0].reset_index(drop=True)


def build_sqlite(edges: pd.DataFrame) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    edges.to_sql("predictor_to_criterion", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX idx_ptc_predictor ON predictor_to_criterion(predictor_id)")
    conn.execute("CREATE INDEX idx_ptc_cluster ON predictor_to_criterion(cluster_id)")
    return conn


def build_adjacency(edges: pd.DataFrame) -> dict[str, list[tuple[str, float]]]:
    adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for row in edges.itertuples(index=False):
        adjacency[str(row.predictor_id)].append((str(row.cluster_id), float(row.score)))
    for predictor_id, rows in adjacency.items():
        rows.sort(key=lambda item: item[1], reverse=True)
    return dict(adjacency)


def build_sparql_graph(edges: pd.DataFrame) -> tuple[Graph, dict[str, URIRef]]:
    graph = Graph()
    graph.bind("phx", PHX)
    predictor_nodes: dict[str, URIRef] = {}

    for row in edges.itertuples(index=False):
        predictor_id = str(row.predictor_id)
        cluster_id = str(row.cluster_id)
        predictor_uri = predictor_nodes.setdefault(
            predictor_id, URIRef(PHX[f"predictor/{slug(predictor_id)}"])
        )
        edge_uri = URIRef(PHX[f"edge/{slug(predictor_id)}_{slug(cluster_id)}"])
        criterion_uri = URIRef(PHX[f"criterion_cluster/{slug(cluster_id)}"])

        graph.add((predictor_uri, RDF.type, PHX.Predictor))
        graph.add((predictor_uri, PHX.path, Literal(str(row.predictor_full_path))))
        graph.add((edge_uri, RDF.type, PHX.MappingEdge))
        graph.add((edge_uri, PHX.sourcePredictor, predictor_uri))
        graph.add((edge_uri, PHX.targetCriterionCluster, criterion_uri))
        graph.add((edge_uri, PHX.score, Literal(float(row.score), datatype=XSD.double)))

    return graph, predictor_nodes


def make_cases(edges: pd.DataFrame, seed: int, n_predictors: int) -> list[QueryCase]:
    rng = random.Random(seed)
    predictors = (
        edges[["predictor_id", "predictor_full_path"]]
        .drop_duplicates()
        .sort_values("predictor_id")
        .to_dict("records")
    )
    sample_size = min(n_predictors, len(predictors))
    selected = rng.sample(predictors, sample_size)
    cases: list[QueryCase] = []
    for row in selected:
        cases.append(
            QueryCase(
                operation="top10_criteria_for_predictor",
                predictor_id=str(row["predictor_id"]),
                predictor_full_path=str(row["predictor_full_path"]),
            )
        )
        cases.append(
            QueryCase(
                operation="criterion_degree_for_predictor",
                predictor_id=str(row["predictor_id"]),
                predictor_full_path=str(row["predictor_full_path"]),
            )
        )
    return cases


def sql_runner(conn: sqlite3.Connection, case: QueryCase) -> object:
    if case.operation == "top10_criteria_for_predictor":
        return conn.execute(
            """
            SELECT cluster_id, score
            FROM predictor_to_criterion
            WHERE predictor_id = ?
            ORDER BY score DESC
            LIMIT 10
            """,
            (case.predictor_id,),
        ).fetchall()

    if case.operation == "criterion_degree_for_predictor":
        return conn.execute(
            """
            SELECT COUNT(DISTINCT cluster_id)
            FROM predictor_to_criterion
            WHERE predictor_id = ?
            """,
            (case.predictor_id,),
        ).fetchone()[0]

    raise ValueError(f"Unknown operation: {case.operation}")


def graph_runner(adjacency: dict[str, list[tuple[str, float]]], case: QueryCase) -> object:
    rows = adjacency.get(case.predictor_id, [])
    if case.operation == "top10_criteria_for_predictor":
        return rows[:10]
    if case.operation == "criterion_degree_for_predictor":
        return len(rows)
    raise ValueError(f"Unknown operation: {case.operation}")


def sparql_runner(graph: Graph, predictor_nodes: dict[str, URIRef], case: QueryCase) -> object:
    predictor_uri = predictor_nodes[case.predictor_id]
    if case.operation == "top10_criteria_for_predictor":
        query = f"""
        PREFIX phx: <{PHX}>
        SELECT ?cluster ?score WHERE {{
          ?edge phx:sourcePredictor <{predictor_uri}> ;
                phx:targetCriterionCluster ?cluster ;
                phx:score ?score .
        }}
        ORDER BY DESC(?score)
        LIMIT 10
        """
        return list(graph.query(query))

    if case.operation == "criterion_degree_for_predictor":
        query = f"""
        PREFIX phx: <{PHX}>
        SELECT (COUNT(DISTINCT ?cluster) AS ?n) WHERE {{
          ?edge phx:sourcePredictor <{predictor_uri}> ;
                phx:targetCriterionCluster ?cluster .
        }}
        """
        return list(graph.query(query))[0][0]

    raise ValueError(f"Unknown operation: {case.operation}")


def timed(
    backend: str,
    cases: Iterable[QueryCase],
    repeats: int,
    runner: Callable[[QueryCase], object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in cases:
        for repeat in range(repeats):
            started = time.perf_counter_ns()
            result = runner(case)
            elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
            rows.append(
                {
                    "backend": backend,
                    "operation": case.operation,
                    "predictor_id": case.predictor_id,
                    "predictor_full_path": case.predictor_full_path,
                    "repeat": repeat,
                    "latency_ms": elapsed_ms,
                    "result_size": len(result) if hasattr(result, "__len__") else 1,
                }
            )
    return rows


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    summary = (
        rows.groupby(["backend", "operation"], as_index=False)["latency_ms"]
        .agg(
            n="count",
            mean_ms="mean",
            median_ms="median",
            min_ms="min",
            max_ms="max",
            p95_ms=lambda values: float(np.percentile(values, 95)),
        )
        .sort_values(["operation", "median_ms"])
    )
    return summary


def plot_latency(rows: pd.DataFrame, summary: pd.DataFrame, out_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    backend_order = ["Adjacency graph", "SQL", "SPARQL"]
    colors = {
        "Adjacency graph": "#2F6F73",
        "SQL": "#8B5E34",
        "SPARQL": "#4E5D8A",
    }
    operations = [
        "top10_criteria_for_predictor",
        "criterion_degree_for_predictor",
    ]
    labels = {
        "top10_criteria_for_predictor": "Top-10 criterion clusters",
        "criterion_degree_for_predictor": "Criterion-cluster degree",
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.25), constrained_layout=True)
    for ax, operation, panel in zip(axes, operations, ["A", "B"]):
        data = [
            rows[(rows["operation"] == operation) & (rows["backend"] == backend)][
                "latency_ms"
            ].to_numpy()
            for backend in backend_order
        ]
        bplot = ax.boxplot(
            data,
            patch_artist=True,
            showfliers=False,
            widths=0.55,
            medianprops={"color": "#111111", "linewidth": 1.4},
            boxprops={"linewidth": 1.0},
            whiskerprops={"linewidth": 1.0},
            capprops={"linewidth": 1.0},
        )
        for patch, backend in zip(bplot["boxes"], backend_order):
            patch.set_facecolor(colors[backend])
            patch.set_alpha(0.78)

        medians = {
            row.backend: row.median_ms
            for row in summary[summary["operation"] == operation].itertuples(index=False)
        }
        for idx, backend in enumerate(backend_order, start=1):
            median = medians.get(backend, float("nan"))
            median_label = "<0.001 ms" if median < 0.001 else f"{median:.3f} ms"
            ax.text(
                idx,
                max(data[idx - 1]) * 1.08 if len(data[idx - 1]) else 1.0,
                median_label,
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#222222",
            )

        ax.set_title(labels[operation], loc="left", fontweight="bold")
        ax.text(
            0.0,
            1.08,
            panel,
            transform=ax.transAxes,
            fontsize=10.5,
            fontweight="bold",
            va="bottom",
            ha="left",
        )
        ax.set_xticks(range(1, len(backend_order) + 1), backend_order, rotation=20, ha="right")
        ax.set_yscale("log")
        ax.set_ylabel("Latency (ms, log scale)")
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Ontology-mediated query latency across local representations",
        x=0.02,
        ha="left",
        fontsize=10.5,
        fontweight="bold",
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_manifest(
    out_dir: Path,
    edge_csv: Path,
    edges: pd.DataFrame,
    cases: list[QueryCase],
    repeats: int,
) -> None:
    manifest = {
        "edge_csv": str(edge_csv),
        "positive_edges": int(len(edges)),
        "predictors": int(edges["predictor_id"].nunique()),
        "criterion_clusters": int(edges["cluster_id"].nunique()),
        "query_cases": len(cases),
        "repeats_per_case": repeats,
        "notes": [
            "Load/build time is excluded.",
            "SQL uses SQLite as a local proxy for PostgreSQL-style relational queries.",
            "Adjacency graph is a local proxy for graph/Cypher-style neighborhood queries.",
            "SPARQL uses RDFLib over a generated mapping graph.",
        ],
    }
    (out_dir / "ontology_query_latency_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-csv", type=Path, default=DEFAULT_EDGE_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-predictors", type=int, default=48)
    parser.add_argument("--repeats", type=int, default=8)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    edges = load_positive_edges(args.edge_csv)
    cases = make_cases(edges, args.seed, args.n_predictors)

    conn = build_sqlite(edges)
    adjacency = build_adjacency(edges)
    graph, predictor_nodes = build_sparql_graph(edges)

    # Warm representative operations before recording timings.
    for warm_case in cases[: min(8, len(cases))]:
        sql_runner(conn, warm_case)
        graph_runner(adjacency, warm_case)
        sparql_runner(graph, predictor_nodes, warm_case)

    rows: list[dict[str, object]] = []
    rows.extend(timed("SQL", cases, args.repeats, lambda case: sql_runner(conn, case)))
    rows.extend(
        timed(
            "Adjacency graph",
            cases,
            args.repeats,
            lambda case: graph_runner(adjacency, case),
        )
    )
    rows.extend(
        timed(
            "SPARQL",
            cases,
            args.repeats,
            lambda case: sparql_runner(graph, predictor_nodes, case),
        )
    )

    rows_df = pd.DataFrame(rows)
    summary_df = summarize(rows_df)

    latency_csv = args.out_dir / "ontology_query_latency.csv"
    summary_csv = args.out_dir / "ontology_query_latency_summary.csv"
    figure_png = args.out_dir / "ontology_query_latency_figure.png"
    rows_df.to_csv(latency_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    summary_df.to_csv(summary_csv, index=False)
    plot_latency(rows_df, summary_df, figure_png)
    write_manifest(args.out_dir, args.edge_csv, edges, cases, args.repeats)

    print(f"Wrote {latency_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {figure_png}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
