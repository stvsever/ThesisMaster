from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _savefig(fig: plt.Figure, out_path: Path, dpi: int, metadata: Dict[str, Any] | None = None) -> None:
    svg_path = out_path.with_suffix(".svg")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    payload = dict(metadata or {})
    payload["generated_at"] = datetime.now().isoformat(timespec="seconds")
    payload["files"] = [str(out_path), str(svg_path), str(pdf_path)]
    out_path.with_suffix(".figure.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _build_matrix_dataframe(
    fusion_payload: Dict[str, Any],
    *,
    top_predictors: int,
) -> pd.DataFrame:
    predictor_order = [str(item) for item in fusion_payload.get("predictor_order", [])][: max(1, int(top_predictors))]
    criterion_order = [str(item) for item in fusion_payload.get("criterion_order", [])]
    edge_rows = fusion_payload.get("edge_rows", []) or []

    if not predictor_order or not criterion_order or not edge_rows:
        return pd.DataFrame()

    lookup = {}
    for row in edge_rows:
        key = (str(row.get("criterion_var_id") or ""), str(row.get("predictor_path") or ""))
        value = float(row.get("fused_score_0_1") or 0.0)
        lookup[key] = value

    data: List[List[float]] = []
    for criterion_id in criterion_order:
        data.append([lookup.get((criterion_id, predictor_path), 0.0) for predictor_path in predictor_order])

    return pd.DataFrame(data, index=criterion_order, columns=predictor_order)


def _plot_heatmap(
    matrix: pd.DataFrame,
    out_path: Path,
    title: str,
    dpi: int,
) -> None:
    fig = plt.figure(figsize=(max(7, 0.42 * matrix.shape[1]), max(5, 0.42 * matrix.shape[0])))
    ax = fig.add_subplot(111)
    im = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=12)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns.tolist(), rotation=75, ha="right", fontsize=7)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index.tolist(), fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Fused relevance (0-1)", rotation=90, fontsize=9)
    _savefig(fig, out_path, dpi=dpi)


def _plot_bipartite_network(
    matrix: pd.DataFrame,
    out_path: Path,
    title: str,
    top_edges: int,
    dpi: int,
) -> None:
    edges = []
    for criterion in matrix.index:
        for predictor in matrix.columns:
            score = float(matrix.loc[criterion, predictor])
            if score <= 0:
                continue
            edges.append((criterion, predictor, score))
    if not edges:
        return

    edges = sorted(edges, key=lambda item: item[2], reverse=True)[: max(1, int(top_edges))]
    criteria = sorted(set(edge[0] for edge in edges))
    predictors = sorted(set(edge[1] for edge in edges))

    def _y_positions(items: List[str]) -> Dict[str, float]:
        if not items:
            return {}
        if len(items) == 1:
            return {items[0]: 0.5}
        return {item: 1.0 - (idx / (len(items) - 1)) for idx, item in enumerate(items)}

    y_criteria = _y_positions(criteria)
    y_predictors = _y_positions(predictors)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.set_title(title, fontsize=12)

    for criterion, predictor, score in edges:
        x1, y1 = 0.06, y_predictors[predictor]
        x2, y2 = 0.94, y_criteria[criterion]
        width = 0.8 + 5.2 * score
        alpha = 0.20 + 0.70 * score
        arc = 0.12 * np.sign(y2 - y1)
        patch = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=width,
            alpha=alpha,
            connectionstyle=f"arc3,rad={arc}",
            color="#274C77",
        )
        ax.add_patch(patch)

    ax.scatter(
        np.full(len(predictors), 0.06),
        [y_predictors[item] for item in predictors],
        s=130,
        color="#3D5A80",
        edgecolors="black",
        linewidths=0.6,
        zorder=3,
    )
    for predictor in predictors:
        ax.text(0.03, y_predictors[predictor], predictor, ha="right", va="center", fontsize=8)

    ax.scatter(
        np.full(len(criteria), 0.94),
        [y_criteria[item] for item in criteria],
        s=130,
        color="#7D4F50",
        edgecolors="black",
        linewidths=0.6,
        zorder=3,
    )
    for criterion in criteria:
        ax.text(0.97, y_criteria[criterion], criterion, ha="left", va="center", fontsize=8)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")
    _savefig(fig, out_path, dpi=dpi)


def _plot_predictor_bars(
    fusion_payload: Dict[str, Any],
    out_path: Path,
    title: str,
    top_predictors: int,
    dpi: int,
) -> None:
    rankings = fusion_payload.get("predictor_rankings", []) or []
    if not rankings:
        return
    frame = pd.DataFrame(rankings).head(max(1, int(top_predictors)))
    if frame.empty:
        return

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.set_title(title, fontsize=12)
    ax.bar(frame["predictor_path"].astype(str), frame["fused_score_0_1"].astype(float))
    ax.set_ylabel("Fused predictor score (0-1)")
    ax.set_xticks(np.arange(len(frame)))
    ax.set_xticklabels(frame["predictor_path"].astype(str).tolist(), rotation=75, ha="right", fontsize=7)
    ax.grid(True, axis="y", alpha=0.25)
    _savefig(fig, out_path, dpi=dpi)


def _plot_bfs_stage_counts(
    fusion_payload: Dict[str, Any],
    out_path: Path,
    title: str,
    dpi: int,
) -> None:
    rankings = fusion_payload.get("predictor_rankings", []) or []
    if not rankings:
        return
    stages = [str(item.get("bfs_stage") or "unknown") for item in rankings]
    stage_counts = pd.Series(stages).value_counts()
    if stage_counts.empty:
        return

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.set_title(title, fontsize=12)
    ax.bar(stage_counts.index.tolist(), stage_counts.values.astype(float))
    ax.set_ylabel("Count")
    ax.set_xticks(np.arange(len(stage_counts)))
    ax.set_xticklabels(stage_counts.index.tolist(), rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    _savefig(fig, out_path, dpi=dpi)


def generate_updated_model_visuals(
    *,
    fusion_payload: Dict[str, Any],
    output_dir: Path,
    profile_id: str,
    dpi: int = 300,
    top_predictors: int = 20,
    top_edges: int = 90,
) -> List[str]:
    visuals_dir = _ensure_dir(output_dir)
    matrix = _build_matrix_dataframe(fusion_payload, top_predictors=top_predictors)
    if matrix.empty:
        return []

    created: List[str] = []
    def _bundle(path: Path) -> List[str]:
        return [str(path), str(path.with_suffix(".svg")), str(path.with_suffix(".pdf")), str(path.with_suffix(".figure.json"))]

    heatmap_path = visuals_dir / "updated_model_fused_heatmap.png"
    _plot_heatmap(
        matrix=matrix,
        out_path=heatmap_path,
        title=f"Updated model fusion heatmap (nomothetic × idiographic) | {profile_id}",
        dpi=dpi,
    )
    created.extend(_bundle(heatmap_path))

    network_path = visuals_dir / "updated_model_fused_bipartite.png"
    _plot_bipartite_network(
        matrix=matrix,
        out_path=network_path,
        title=f"Updated model fusion network (top edges) | {profile_id}",
        top_edges=top_edges,
        dpi=dpi,
    )
    created.extend(_bundle(network_path))

    bar_path = visuals_dir / "updated_model_predictor_fused_scores.png"
    _plot_predictor_bars(
        fusion_payload=fusion_payload,
        out_path=bar_path,
        title=f"Updated model predictor fusion ranking | {profile_id}",
        top_predictors=top_predictors,
        dpi=dpi,
    )
    created.extend(_bundle(bar_path))

    stage_path = visuals_dir / "updated_model_bfs_stage_distribution.png"
    _plot_bfs_stage_counts(
        fusion_payload=fusion_payload,
        out_path=stage_path,
        title=f"Breadth-first candidate stage coverage | {profile_id}",
        dpi=dpi,
    )
    created.extend(_bundle(stage_path))

    return created
