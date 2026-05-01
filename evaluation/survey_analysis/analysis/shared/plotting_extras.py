"""
Extra plotting helpers specific to the LLM-as-judge synthesis.

Adds a parts-x-dimensions effect heatmap to complement the figures provided
by ``shared_stats``. Kept separate from ``shared_stats`` so that the latter
remains schema-agnostic.
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .shared_stats import apply_rcparams, p_to_stars


def parts_dimensions_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    part_labels: Sequence[str],
    dim_labels: Sequence[str],
    title: str = "Effect heatmap: parts x dimensions",
    pvals: Optional[np.ndarray] = None,
    cmap: str = "RdBu_r",
    vmax: Optional[float] = None,
    cbar_label: str = "Delta (PHOENIX - HCP)",
) -> plt.Axes:
    """
    Render a heatmap of effect sizes (rows = parts, cols = dimensions).

    Parameters
    ----------
    matrix : np.ndarray of shape (n_parts, n_dims)
        Effect values. NaN cells are rendered blank.
    pvals : np.ndarray of shape (n_parts, n_dims), optional
        Holm-adjusted p-values; if given, significance stars are written
        below each numeric value.
    """
    apply_rcparams()
    finite = matrix[np.isfinite(matrix)]
    if vmax is None:
        vmax = float(np.max(np.abs(finite))) if finite.size else 0.3
    vmax = max(vmax, 1e-3)
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(dim_labels)))
    ax.set_xticklabels(dim_labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(part_labels)))
    ax.set_yticklabels(part_labels, fontsize=10)
    ax.set_title(title, fontweight="bold", pad=10)

    n_rows, n_cols = matrix.shape
    for i in range(n_rows):
        for j in range(n_cols):
            v = matrix[i, j]
            if not np.isfinite(v):
                continue
            txt = f"{v:+.2f}"
            if pvals is not None and np.isfinite(pvals[i, j]):
                txt += f"\n{p_to_stars(float(pvals[i, j]))}"
            ax.text(
                j, i, txt,
                ha="center", va="center", fontsize=7,
                color="white" if abs(v) > 0.6 * vmax else "black",
            )
    plt.colorbar(im, ax=ax, label=cbar_label, shrink=0.85)
    return ax
