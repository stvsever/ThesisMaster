#!/usr/bin/env python3
"""
compare_time_series_analysis_methods_cv.py

Full pipeline with:
  Method 1: Time-varying gVAR via kernel-smoothed L1 VAR(1) ("tvKS")
            + 5-fold time-series CV for tv hyperparameters (bandwidth + alpha)
            + moving-block bootstrap CIs
            + automated "cluster creation" heuristic based on modelling-error ratio (stationary vs tv)
            + per-timepoint network frames + GIF
  Method 2: Stationary gVAR (L1 VAR(1) + residual partial correlations via graphical lasso)
  Method 3: Pearson correlation baseline

Key improvements vs your current version:
  Correct simulation time alignment (burn-in handled properly; t_norm matches X)
  Proper 5-fold TimeSeriesSplit CV for the tv model’s bandwidth and alpha
  Leakage-safe standardization during CV (fit on training only)
  Kernel weights normalized to constant sum (keeps alpha comparable across bandwidths)

This script simulates multivariate time series with **designed temporal structure** to benchmark
time-varying vs stationary gVAR methods.

CREATED DATA PATTERNS
---------------------
1) NODE STRUCTURE
   - 10 nodes total: 6 Predictors (P1–P6) and 4 Criteria (C1–C4).
   - Innovation noise has block-correlated covariance:
       * strong P–P correlations
       * moderate C–C correlations
       * weaker P–C correlations
   → induces visible contemporaneous correlation and partial-correlation networks.

2) BASELINE VAR(1) SKELETON
   - All nodes have autoregressive self-loops (AR(1)) that oscillate slowly over time.
   - Two fixed directed chains:
       * P1→P2→P3→P4→P5→P6
       * C1→C2→C3→C4
   → provides a stable, sparse temporal backbone.

3) DESIGNED TIME-VARYING EDGES (GROUND TRUTH SIGNAL)
   - A small set of lagged effects explicitly varies over time using:
       * smooth sigmoid transitions,
       * abrupt step changes,
       * slow and fast periodic oscillations.
   - Key cross-lagged edges change sign and/or strength around t≈0.55 or cyclically:
       * C2→P4, P3→C1, C4→C3
   - Additional edges fade in/out or become negative after the transition:
       * C2→P2 (fades out), C4→P6 (turns negative).

4) STABILITY CONTROL
   - All coefficient matrices are clipped and rescaled to keep spectral radius < 1,
     ensuring a stable VAR process.

NET EFFECT
----------
The data contain:
   - globally persistent but stable dynamics,
   - interpretable directed chains,
   - localized time-varying causal changes (smooth, abrupt, or periodic),
   - structured contemporaneous dependence via correlated innovations.

This setup is designed to test whether time-varying gVAR models recover true temporal
changes better than stationary VAR or correlation-based methods.

Notes:
- For computational tractability, tv hyperparameter CV is two-stage:
    1) Select bandwidth via CV using a fixed alpha (default: median(alpha_grid))
    2) Select alpha via CV using the selected bandwidth
  Both steps are 5-fold TimeSeriesSplit.

- If you want *more sparsity* than CV selects, increase --alpha-min / --alpha-max,
  or set --alpha-floor (hard lower bound after CV).

"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
from sklearn.model_selection import TimeSeriesSplit
from sklearn.covariance import GraphicalLassoCV, LedoitWolf

import networkx as nx  # only used for deterministic layout (circular)
from matplotlib.patches import FancyArrowPatch
from PIL import Image


# ============================================================
# Repro utilities
# ============================================================
def set_seed(seed: int) -> np.random.Generator:
    np.random.seed(seed)
    return np.random.default_rng(seed)


# ============================================================
# Node naming / groups (6 predictors + 4 criteria = 10 total)
# ============================================================
P_COUNT = 6
C_COUNT = 4
P_IDX = list(range(0, P_COUNT))
C_IDX = list(range(P_COUNT, P_COUNT + C_COUNT))


def node_names() -> List[str]:
    return [f"P{i+1}" for i in range(P_COUNT)] + [f"C{i+1}" for i in range(C_COUNT)]


def node_groups() -> List[str]:
    """Per node: 'P' or 'C'."""
    return ["P"] * P_COUNT + ["C"] * C_COUNT


# ============================================================
# Core math utilities
# ============================================================
def sigmoid(x: np.ndarray, k: float = 1.0) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-k * x))


def zscore_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return mu, sd


def zscore_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd


def zscore(X: np.ndarray) -> np.ndarray:
    mu, sd = zscore_fit(X)
    return zscore_apply(X, mu, sd)


def gaussian_kernel_weights(t: np.ndarray, te: float, bandwidth: float) -> np.ndarray:
    """Gaussian kernel weights for times t around te (bandwidth in normalized [0, 1])."""
    bw = max(float(bandwidth), 1e-9)
    z = (t - te) / bw
    # gaussian density (scale doesn't matter; we renormalize)
    w = np.exp(-0.5 * z * z) / (math.sqrt(2.0 * math.pi) * bw)
    return w


def normalize_weights_sum_to_n(w: np.ndarray) -> np.ndarray:
    """
    Normalize weights to have constant sum = n (number of observations),
    so the loss scale is comparable across bandwidth choices (important for alpha comparability).
    """
    w = np.asarray(w, dtype=float)
    n = float(w.size)
    s = float(np.sum(w)) + 1e-12
    return w * (n / s)


def weighted_mean(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    wsum = np.sum(w) + 1e-12
    return (w[:, None] * X).sum(axis=0) / wsum


def weighted_cov(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted covariance (denominator sum(w), not unbiased)."""
    mu = weighted_mean(X, w)
    Xc = X - mu[None, :]
    wsum = np.sum(w) + 1e-12
    return (Xc.T * w) @ Xc / wsum


def precision_to_partial_corr(Theta: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(Theta), 1e-12, np.inf))
    pc = -Theta / (d[:, None] * d[None, :])
    np.fill_diagonal(pc, 1.0)
    return pc


def safe_invert_cov(S: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    p = S.shape[0]
    return np.linalg.inv(S + ridge * np.eye(p))


def effective_sample_size(w: np.ndarray) -> float:
    """ESS for kernel weights: (sum w)^2 / sum(w^2). Invariant to scaling of w."""
    w = np.asarray(w, dtype=float)
    sw = float(np.sum(w)) + 1e-12
    return float((sw * sw) / (float(np.sum(w * w)) + 1e-12))


# ============================================================
# Stability tools (for simulation)
# ============================================================
def spectral_radius(A: np.ndarray) -> float:
    vals = np.linalg.eigvals(A)
    return float(np.max(np.abs(vals)))


def stabilize_by_scaling(B: np.ndarray, target_rho: float = 0.95) -> np.ndarray:
    rho = spectral_radius(B)
    if rho <= target_rho or rho == 0.0:
        return B
    return (target_rho / rho) * B


# ============================================================
# Pseudo-data scenarios (simulation) — FIXED time alignment
# ============================================================
def make_innovation_cov(p: int = 10, rho_p: float = 0.35, rho_c: float = 0.28, rho_pc: float = 0.18) -> np.ndarray:
    """
    Build a positive definite innovation covariance matrix with block correlations.
    This ensures correlation & partial-correlation networks have visible structure.
    """
    assert p == 10
    Sigma = np.eye(p)

    for i in P_IDX:
        for j in P_IDX:
            if i != j:
                Sigma[i, j] = rho_p

    for i in C_IDX:
        for j in C_IDX:
            if i != j:
                Sigma[i, j] = rho_c

    for i in P_IDX:
        for j in C_IDX:
            Sigma[i, j] = rho_pc
            Sigma[j, i] = rho_pc

    Sigma = Sigma + 1e-3 * np.eye(p)
    vals = np.linalg.eigvalsh(Sigma)
    if np.min(vals) <= 1e-8:
        Sigma = Sigma + (abs(np.min(vals)) + 1e-2) * np.eye(p)
    return Sigma


def scenario_B(t: float, scenario: str, p: int = 10, max_abs: float = 0.40) -> np.ndarray:
    """
    Three scenarios; each enforces multiple time-varying patterns.
    Indices use P1..P6 (0..5), C1..C4 (6..9).
    """
    assert p == 10
    B = np.zeros((p, p), dtype=float)

    base_diag = 0.30 + 0.06 * np.cos(2 * math.pi * t)
    for i in range(p):
        B[i, i] = base_diag * (1.0 if i in P_IDX else 0.85)

    # small chain structure
    for j in range(0, 5):
        B[j + 1, j] += 0.12
    for j in range(6, 9):
        B[j + 1, j] += 0.10

    w_smooth = float(sigmoid(np.array([t - 0.55]), k=18.0)[0])
    w_step = 1.0 if t >= 0.55 else 0.0
    w_period = 0.5 * (1.0 + math.sin(2 * math.pi * t * 1.0))
    w_fast = 0.5 * (1.0 + math.sin(2 * math.pi * t * 2.0))

    # showcased time-varying edges
    B[0, 0] += 0.08 * math.sin(2 * math.pi * t + 0.4)
    B[1, 4] += 0.24 * math.sin(2 * math.pi * t * (1.0 if scenario != "scenario2_abrupt" else 0.7))

    if scenario == "scenario1_smooth":
        B[3, 7] += 0.28 * (1.0 - w_smooth) - 0.18 * w_smooth
        B[6, 2] += 0.30 * w_smooth
        B[8, 9] += 0.10 + 0.18 * w_smooth
    elif scenario == "scenario2_abrupt":
        B[3, 7] += 0.30 * (1.0 - w_step) - 0.22 * w_step
        B[6, 2] += 0.32 * w_step
        B[8, 9] += 0.12 + 0.22 * w_step
    else:
        B[3, 7] += 0.26 * (1.0 - w_period) - 0.20 * w_period
        B[6, 2] += 0.28 * w_fast
        B[8, 9] += 0.12 + 0.20 * w_period

    B[7, 1] += 0.18 * (1.0 - w_smooth)
    B[9, 5] += -0.14 * w_smooth

    B = np.clip(B, -max_abs, max_abs)
    B = stabilize_by_scaling(B, target_rho=0.95)
    return B


def simulate_tvvar(
    n: int = 800,
    p: int = 10,
    burnin: int = 150,
    seed: int = 42,
    scenario: str = "scenario1_smooth",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      X_raw:      shape (n, p)  (NOT standardized; caller decides standardization)
      t_norm:     shape (n,)    aligned with X_raw
      B_true_all: shape (n, p, p) aligned with X_raw and t_norm
    """
    rng = set_seed(seed)

    n_total = n + burnin
    t_full = np.linspace(0.0, 1.0, n_total)

    B_full = np.zeros((n_total, p, p), dtype=float)
    for k in range(n_total):
        B_full[k] = scenario_B(float(t_full[k]), scenario=scenario, p=p)

    Sigma_eps = make_innovation_cov(p=p)
    L = np.linalg.cholesky(Sigma_eps)

    X = np.zeros((n_total, p), dtype=float)
    X[0] = rng.normal(0.0, 0.5, size=p)

    for k in range(1, n_total):
        B = B_full[k - 1]
        eps = (L @ rng.normal(0.0, 1.0, size=p))
        X[k] = B @ X[k - 1] + eps

    # drop burn-in and rescale time to [0,1] for the retained segment
    X_raw = X[burnin:]
    t_kept = t_full[burnin:]
    t_norm = (t_kept - t_kept[0]) / (t_kept[-1] - t_kept[0] + 1e-12)
    B_true_all = B_full[burnin:]

    return X_raw, t_norm, B_true_all


# ============================================================
# Weighted Lasso fitting
# ============================================================
@dataclass
class LassoFitResult:
    coef: np.ndarray
    intercept: float
    alpha: float


def weighted_lasso_fit(Z: np.ndarray, y: np.ndarray, w: np.ndarray, alpha: float, seed: int = 42) -> LassoFitResult:
    model = Lasso(alpha=float(alpha), fit_intercept=True, max_iter=20000, random_state=seed)
    model.fit(Z, y, sample_weight=w)
    return LassoFitResult(coef=model.coef_.copy(), intercept=float(model.intercept_), alpha=float(alpha))


# ============================================================
# tvKS estimator (time-varying)
# ============================================================
@dataclass
class TvKsResult:
    estpoints: np.ndarray
    bandwidth: float
    Bhat: np.ndarray
    intercepts: np.ndarray
    alphas: np.ndarray
    r2_node: np.ndarray
    partial_corr: np.ndarray
    mse_point: np.ndarray
    ess_point: np.ndarray  # effective sample size per estimation point


def estimate_tvKS(
    X: np.ndarray,
    t_norm: np.ndarray,
    estpoints: np.ndarray,
    bandwidth: float,
    alphas_node: np.ndarray,
    seed: int = 42,
    ridge_precision: float = 1e-3,
) -> TvKsResult:
    """
    Kernel-smoothed L1 VAR(1):
      For each estimation point te, fit weighted Lasso per node with kernel weights w(t_obs, te).
    """
    Y = X[1:]
    Z = X[:-1]
    t_obs = t_norm[1:]
    n_obs, p = Y.shape
    m = len(estpoints)

    Bhat = np.zeros((m, p, p))
    intercepts = np.zeros((m, p))
    r2_node = np.zeros((m, p))
    partial_corr = np.zeros((m, p, p))
    mse_point = np.zeros(m)
    ess_point = np.zeros(m)

    alphas = np.tile(alphas_node[None, :], (m, 1))

    for ei, te in enumerate(estpoints):
        w_raw = gaussian_kernel_weights(t_obs, float(te), float(bandwidth))
        w = normalize_weights_sum_to_n(w_raw)
        ess_point[ei] = effective_sample_size(w)

        for i in range(p):
            a = float(alphas_node[i])
            fit = weighted_lasso_fit(Z, Y[:, i], w, alpha=a, seed=seed)
            Bhat[ei, i, :] = fit.coef
            intercepts[ei, i] = fit.intercept

            y = Y[:, i]
            y_pred = (Z @ fit.coef) + fit.intercept
            wsum = float(np.sum(w))
            y_bar = float(np.sum(w * y) / (wsum + 1e-12))
            sst = float(np.sum(w * (y - y_bar) ** 2))
            sse = float(np.sum(w * (y - y_pred) ** 2))
            r2_node[ei, i] = 1.0 - (sse / (sst + 1e-12))

        Yhat = intercepts[ei][None, :] + (Z @ Bhat[ei].T)
        mse_point[ei] = float(np.sum(w[:, None] * (Y - Yhat) ** 2) / (np.sum(w) * p + 1e-12))

        E = Y - Yhat
        S = weighted_cov(E, w)
        Theta = safe_invert_cov(S, ridge=ridge_precision)
        partial_corr[ei] = precision_to_partial_corr(Theta)

    return TvKsResult(
        estpoints=estpoints.copy(),
        bandwidth=float(bandwidth),
        Bhat=Bhat,
        intercepts=intercepts,
        alphas=alphas,
        r2_node=r2_node,
        partial_corr=partial_corr,
        mse_point=mse_point,
        ess_point=ess_point,
    )


# ============================================================
# Proper 5-fold TimeSeriesSplit CV for tvKS hyperparameters
# ============================================================
@dataclass
class TvCvSummary:
    n_splits: int
    standardize_in_cv: bool
    alpha_fixed_for_bw: float
    bw_grid: List[float]
    bw_cv_mse: List[float]
    bw_selected: float
    alpha_grid: List[float]
    alpha_cv_mse: List[float]
    alpha_selected: float


def _tv_cv_mse(
    X_raw: np.ndarray,
    t_norm: np.ndarray,
    estpoints: np.ndarray,
    bandwidth: float,
    alpha: float,
    n_splits: int,
    seed: int,
    standardize_in_cv: bool,
    ridge_precision: float,
) -> float:
    """
    Leakage-safe forward-chaining CV MSE for tvKS:
      - Splits are TimeSeriesSplit on VAR observations (Z/Y rows)
      - For each fold, standardize using training prefix only (if enabled)
      - Fit tvKS on training prefix only
      - Predict validation block using nearest estpoint model
    """
    X_raw = np.asarray(X_raw, dtype=float)
    t_norm = np.asarray(t_norm, dtype=float)

    Y_full = X_raw[1:]
    Z_full = X_raw[:-1]
    t_obs_full = t_norm[1:]
    n_obs, p = Y_full.shape

    tscv = TimeSeriesSplit(n_splits=int(n_splits))

    fold_mses: List[float] = []

    for fold_id, (tr, va) in enumerate(tscv.split(Z_full)):
        if len(tr) == 0 or len(va) == 0:
            continue

        train_end = int(tr[-1])
        val_end = int(va[-1])

        # prefix needed to cover validation Y (needs X up to val_end+1)
        X_prefix_raw = X_raw[: val_end + 2]
        t_prefix = t_norm[: val_end + 2]

        if standardize_in_cv:
            mu, sd = zscore_fit(X_prefix_raw[: train_end + 2])
            X_prefix = zscore_apply(X_prefix_raw, mu, sd)
        else:
            X_prefix = X_prefix_raw

        X_train = X_prefix[: train_end + 2]
        t_train = t_prefix[: train_end + 2]

        alphas_node = np.full(p, float(alpha), dtype=float)
        tv = estimate_tvKS(
            X_train,
            t_train,
            estpoints=estpoints,
            bandwidth=float(bandwidth),
            alphas_node=alphas_node,
            seed=int(seed) + 17 * fold_id,
            ridge_precision=float(ridge_precision),
        )

        # standardized prefix for prediction
        Zp = X_prefix[:-1]
        Yp = X_prefix[1:]
        t_obs_p = t_prefix[1:]

        va_t = t_obs_p[va]
        idx = np.argmin(np.abs(estpoints[None, :] - va_t[:, None]), axis=1).astype(int)

        # Yhat = intercept + B @ x_prev
        B_sel = tv.Bhat[idx]            # (n_va, p, p)
        c_sel = tv.intercepts[idx]      # (n_va, p)
        x_sel = Zp[va]                  # (n_va, p)

        yhat = c_sel + np.einsum("nij,nj->ni", B_sel, x_sel)

        mse = float(np.mean((Yp[va] - yhat) ** 2))
        fold_mses.append(mse)

    return float(np.mean(fold_mses)) if fold_mses else float("inf")


def select_bandwidth_tv_cv(
    X_raw: np.ndarray,
    t_norm: np.ndarray,
    estpoints_cv: np.ndarray,
    bw_grid: np.ndarray,
    alpha_fixed: float,
    n_splits: int,
    seed: int,
    standardize_in_cv: bool,
    ridge_precision: float,
) -> Tuple[float, List[float]]:
    scores: List[float] = []
    for b in bw_grid:
        mse = _tv_cv_mse(
            X_raw=X_raw,
            t_norm=t_norm,
            estpoints=estpoints_cv,
            bandwidth=float(b),
            alpha=float(alpha_fixed),
            n_splits=n_splits,
            seed=seed,
            standardize_in_cv=standardize_in_cv,
            ridge_precision=ridge_precision,
        )
        scores.append(float(mse))
    best_idx = int(np.argmin(scores))
    return float(bw_grid[best_idx]), scores


def select_alpha_tv_cv(
    X_raw: np.ndarray,
    t_norm: np.ndarray,
    estpoints_cv: np.ndarray,
    bandwidth_fixed: float,
    alpha_grid: np.ndarray,
    n_splits: int,
    seed: int,
    standardize_in_cv: bool,
    ridge_precision: float,
) -> Tuple[float, List[float]]:
    scores: List[float] = []
    for a in alpha_grid:
        mse = _tv_cv_mse(
            X_raw=X_raw,
            t_norm=t_norm,
            estpoints=estpoints_cv,
            bandwidth=float(bandwidth_fixed),
            alpha=float(a),
            n_splits=n_splits,
            seed=seed,
            standardize_in_cv=standardize_in_cv,
            ridge_precision=ridge_precision,
        )
        scores.append(float(mse))
    best_idx = int(np.argmin(scores))
    return float(alpha_grid[best_idx]), scores


# ============================================================
# Moving-block bootstrap CIs (temporal B only)
# ============================================================
def moving_block_bootstrap_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    if block_len <= 1:
        return rng.integers(0, n, size=n)
    starts = rng.integers(0, max(1, n - block_len + 1), size=int(math.ceil(n / block_len)))
    idx = np.concatenate([np.arange(s, s + block_len) for s in starts], axis=0)[:n]
    return idx


@dataclass
class BootstrapCI:
    B_low: np.ndarray
    B_high: np.ndarray


def bootstrap_tvKS_CI(
    X: np.ndarray,
    t_norm: np.ndarray,
    tv_fit: TvKsResult,
    alphas_node: np.ndarray,
    n_boot: int = 100,
    block_len: int = 20,
    ci: Tuple[float, float] = (0.05, 0.95),
    seed: int = 42,
    n_jobs: int = 1,
) -> BootstrapCI:
    Y0 = X[1:]
    Z0 = X[:-1]
    t_obs0 = t_norm[1:]
    n_obs, p = Y0.shape
    m = len(tv_fit.estpoints)

    def one_boot(bi: int) -> np.ndarray:
        rng = np.random.default_rng(seed + 1000 + bi)
        idx = moving_block_bootstrap_indices(n_obs, block_len, rng)
        Y = Y0[idx]
        Z = Z0[idx]
        t_obs = t_obs0[idx]

        B_b = np.zeros((m, p, p), dtype=float)
        for ei, te in enumerate(tv_fit.estpoints):
            w_raw = gaussian_kernel_weights(t_obs, float(te), float(tv_fit.bandwidth))
            w = normalize_weights_sum_to_n(w_raw)
            for i in range(p):
                fit = weighted_lasso_fit(Z, Y[:, i], w, alpha=float(alphas_node[i]), seed=seed + bi)
                B_b[ei, i, :] = fit.coef
        return B_b

    if n_jobs != 1:
        try:
            from joblib import Parallel, delayed
            draws = Parallel(n_jobs=n_jobs, verbose=0)(delayed(one_boot)(bi) for bi in range(n_boot))
        except Exception:
            draws = [one_boot(bi) for bi in range(n_boot)]
    else:
        draws = [one_boot(bi) for bi in range(n_boot)]

    B_draws = np.stack(draws, axis=0)
    low = np.quantile(B_draws, ci[0], axis=0)
    high = np.quantile(B_draws, ci[1], axis=0)
    return BootstrapCI(B_low=low, B_high=high)


# ============================================================
# Stationary gVAR (kept; uses TimeSeriesSplit CV per node)
# ============================================================
@dataclass
class StationaryGvarResult:
    B: np.ndarray
    intercept: np.ndarray
    alphas: np.ndarray
    r2_node: np.ndarray
    residual_cov: np.ndarray
    partial_corr: np.ndarray


def fit_stationary_gvar(X: np.ndarray, alpha_grid: np.ndarray, seed: int = 42) -> StationaryGvarResult:
    Y = X[1:]
    Z = X[:-1]
    n_obs, p = Y.shape
    tscv = TimeSeriesSplit(n_splits=5)
    w = np.ones(n_obs)

    B = np.zeros((p, p))
    intercept = np.zeros(p)
    alphas = np.zeros(p)
    r2_node = np.zeros(p)

    for i in range(p):
        best_a = float(alpha_grid[0])
        best_mse = float("inf")
        y = Y[:, i]
        for a in alpha_grid:
            mses = []
            for tr, va in tscv.split(Z):
                fit = weighted_lasso_fit(Z[tr], y[tr], w[tr], alpha=float(a), seed=seed)
                pred = (Z[va] @ fit.coef) + fit.intercept
                mses.append(float(np.mean((y[va] - pred) ** 2)))
            m = float(np.mean(mses))
            if m < best_mse:
                best_mse = m
                best_a = float(a)

        fit = weighted_lasso_fit(Z, y, w, alpha=best_a, seed=seed)
        B[i, :] = fit.coef
        intercept[i] = fit.intercept
        alphas[i] = best_a

        y_pred = (Z @ fit.coef) + fit.intercept
        y_bar = float(np.mean(y))
        sst = float(np.sum((y - y_bar) ** 2))
        sse = float(np.sum((y - y_pred) ** 2))
        r2_node[i] = 1.0 - (sse / (sst + 1e-12))

    E = Y - (Z @ B.T + intercept[None, :])
    residual_cov = np.cov(E, rowvar=False)

    try:
        gl = GraphicalLassoCV()
        gl.fit(E)
        Theta = gl.precision_
    except Exception:
        lw = LedoitWolf().fit(E)
        Theta = safe_invert_cov(lw.covariance_, ridge=1e-3)

    partial_corr = precision_to_partial_corr(Theta)
    return StationaryGvarResult(
        B=B,
        intercept=intercept,
        alphas=alphas,
        r2_node=r2_node,
        residual_cov=residual_cov,
        partial_corr=partial_corr,
    )


# ============================================================
# Correlation baseline
# ============================================================
@dataclass
class CorrResult:
    corr: np.ndarray


def compute_corr(X: np.ndarray) -> CorrResult:
    return CorrResult(corr=np.corrcoef(X, rowvar=False))


# ============================================================
# Metrics helpers
# ============================================================
def nearest_true_B(B_true_all: np.ndarray, t_norm: np.ndarray, te: float) -> np.ndarray:
    idx = int(np.argmin(np.abs(t_norm - te)))
    return B_true_all[idx]


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def edge_recovery_scores(B_est: np.ndarray, B_true: np.ndarray, thr_true: float = 0.06, thr_est: float = 0.06) -> Dict[str, float]:
    p = B_true.shape[0]
    mask = ~np.eye(p, dtype=bool)
    true_pos = (np.abs(B_true) >= thr_true) & mask
    est_pos = (np.abs(B_est) >= thr_est) & mask
    tp = float(np.sum(true_pos & est_pos))
    fp = float(np.sum(~true_pos & est_pos))
    fn = float(np.sum(true_pos & ~est_pos))
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1)}


def one_step_prediction_metrics_tv(X: np.ndarray, t_norm: np.ndarray, tv: TvKsResult) -> Dict[str, float]:
    Y = X[1:]
    Z = X[:-1]
    t_obs = t_norm[1:]
    n_obs, p = Y.shape

    idx = np.argmin(np.abs(tv.estpoints[None, :] - t_obs[:, None]), axis=1).astype(int)
    B_sel = tv.Bhat[idx]
    c_sel = tv.intercepts[idx]
    Yhat = c_sel + np.einsum("nij,nj->ni", B_sel, Z)

    mse = float(np.mean((Y - Yhat) ** 2))
    y_bar = np.mean(Y, axis=0, keepdims=True)
    sst = float(np.sum((Y - y_bar) ** 2))
    sse = float(np.sum((Y - Yhat) ** 2))
    r2 = 1.0 - (sse / (sst + 1e-12))
    return {"mse": mse, "r2_overall": float(r2)}


def one_step_prediction_metrics_stationary(X: np.ndarray, st: StationaryGvarResult) -> Dict[str, float]:
    Y = X[1:]
    Z = X[:-1]
    Yhat = st.intercept[None, :] + (Z @ st.B.T)
    mse = float(np.mean((Y - Yhat) ** 2))
    y_bar = np.mean(Y, axis=0, keepdims=True)
    sst = float(np.sum((Y - y_bar) ** 2))
    sse = float(np.sum((Y - Yhat) ** 2))
    r2 = 1.0 - (sse / (sst + 1e-12))
    return {"mse": mse, "r2_overall": float(r2)}


# ============================================================
# Cluster heuristic for time-varying networks (tv only)
# ============================================================
@dataclass
class TvClusterResult:
    create_clusters: bool
    ratio_stationary_over_tv: np.ndarray
    threshold_ratio: float
    peaks: List[int]
    segments: List[Tuple[int, int]]


def detect_tv_clusters_by_error_ratio(
    X: np.ndarray,
    t_norm: np.ndarray,
    tv: TvKsResult,
    st: StationaryGvarResult,
    ratio_threshold: float = 3.0,
) -> TvClusterResult:
    Y = X[1:]
    Z = X[:-1]
    t_obs = t_norm[1:]
    m = len(tv.estpoints)
    p = Y.shape[1]

    mse_stat = np.zeros(m)
    Yhat_stat_full = st.intercept[None, :] + (Z @ st.B.T)

    for ei, te in enumerate(tv.estpoints):
        w_raw = gaussian_kernel_weights(t_obs, float(te), float(tv.bandwidth))
        w = normalize_weights_sum_to_n(w_raw)
        mse_stat[ei] = float(np.sum(w[:, None] * (Y - Yhat_stat_full) ** 2) / (np.sum(w) * p + 1e-12))

    ratio = mse_stat / (tv.mse_point + 1e-12)
    create = bool(np.max(ratio) >= ratio_threshold)

    peaks: List[int] = []
    segments: List[Tuple[int, int]] = []

    if not create:
        return TvClusterResult(
            create_clusters=False,
            ratio_stationary_over_tv=ratio,
            threshold_ratio=float(ratio_threshold),
            peaks=[],
            segments=[],
        )

    idx = np.where(ratio >= ratio_threshold)[0]
    runs: List[List[int]] = []
    cur: List[int] = []
    for ii in idx:
        if not cur or ii == cur[-1] + 1:
            cur.append(int(ii))
        else:
            runs.append(cur)
            cur = [int(ii)]
    if cur:
        runs.append(cur)

    for run in runs:
        peak = int(run[int(np.argmax(ratio[run]))])
        peaks.append(peak)

    peaks = sorted(set(peaks))
    boundaries = [pp for pp in peaks if 0 <= pp < m - 1]
    start = 0
    for b in boundaries:
        segments.append((start, b))
        start = b + 1
    segments.append((start, m - 1))

    return TvClusterResult(
        create_clusters=True,
        ratio_stationary_over_tv=ratio,
        threshold_ratio=float(ratio_threshold),
        peaks=peaks,
        segments=segments,
    )


# ============================================================
# Professional network visualization (Matplotlib-first)
# ============================================================
def _node_color_list() -> List[str]:
    return ["tab:blue" if g == "P" else "tab:orange" for g in node_groups()]


def _fixed_circular_pos(p: int) -> Dict[int, np.ndarray]:
    nodes = list(range(p))
    pos = nx.circular_layout(nodes)
    return {k: np.array(v, dtype=float) for k, v in pos.items()}


def _layout_limits(pos: Dict[int, np.ndarray], pad: float = 0.40) -> Tuple[float, float, float, float]:
    xs = np.array([pos[i][0] for i in pos], dtype=float)
    ys = np.array([pos[i][1] for i in pos], dtype=float)
    return float(xs.min() - pad), float(xs.max() + pad), float(ys.min() - pad), float(ys.max() + pad)


def _draw_nodes_and_labels(ax, pos: Dict[int, np.ndarray], node_sizes: List[float], node_colors: List[str]) -> None:
    labels = node_names()
    xs = np.array([pos[i][0] for i in range(len(labels))], dtype=float)
    ys = np.array([pos[i][1] for i in range(len(labels))], dtype=float)

    ax.scatter(xs, ys, s=node_sizes, c=node_colors, edgecolors="black", linewidths=0.9, zorder=3)

    for i, lab in enumerate(labels):
        ax.text(xs[i], ys[i], lab, ha="center", va="center", fontsize=9, zorder=4, color="black")


def _edge_widths_from_abs(abs_w: np.ndarray, w_min: float = 0.9, w_max: float = 4.5) -> np.ndarray:
    if abs_w.size == 0:
        return abs_w
    amax = float(np.max(abs_w)) + 1e-12
    return w_min + (w_max - w_min) * (abs_w / amax)


def _select_directed_edges(
    B: np.ndarray,
    thr_offdiag: float,
    thr_diag: float,
    topk_offdiag: int,
    min_offdiag_edges: int,
) -> Tuple[np.ndarray, float, int]:
    p = B.shape[0]
    mask_off = np.ones((p, p), dtype=bool)
    np.fill_diagonal(mask_off, False)
    abs_off = np.abs(B) * mask_off

    sel = (abs_off >= thr_offdiag)
    off_count = int(np.sum(sel))

    flat = abs_off.flatten()
    nonzero = flat[flat > 0]
    if nonzero.size == 0:
        used_thr = float(thr_offdiag)
        return sel, used_thr, off_count

    def mask_topk(k: int) -> Tuple[np.ndarray, float]:
        k = max(1, int(k))
        vals = np.sort(nonzero)
        k = min(k, vals.size)
        thr_k = float(vals[-k])
        return (abs_off >= thr_k), thr_k

    used_thr = float(thr_offdiag)

    if off_count > topk_offdiag:
        sel, used_thr = mask_topk(topk_offdiag)
        off_count = int(np.sum(sel))

    if off_count < min_offdiag_edges and min_offdiag_edges > 0:
        k = min(topk_offdiag, max(min_offdiag_edges, off_count))
        sel, used_thr = mask_topk(k)
        off_count = int(np.sum(sel))

    np.fill_diagonal(sel, False)
    return sel, float(used_thr), int(off_count)


def _draw_self_loop(ax, x: float, y: float, weight: float, scale: float = 0.12) -> None:
    lw = 1.0 + 4.2 * abs(weight)
    style = "-" if weight >= 0 else "--"
    rad = 0.65
    patch = FancyArrowPatch(
        (x + scale, y),
        (x, y + scale),
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>",
        mutation_scale=10 + 2.5 * lw,
        linewidth=lw,
        linestyle=style,
        color="black",
        alpha=0.9,
        zorder=2,
    )
    ax.add_patch(patch)


def draw_directed_network(
    ax,
    B: np.ndarray,
    node_score: Optional[np.ndarray],
    title: str,
    thr_offdiag: float,
    thr_diag: float,
    topk_offdiag: int,
    min_offdiag_edges: int,
    pos: Dict[int, np.ndarray],
    fixed_limits: Tuple[float, float, float, float],
    show_legend: bool = True,
) -> Dict[str, float]:
    p = B.shape[0]
    colors = _node_color_list()

    base = 780.0
    if node_score is None:
        node_sizes = [base] * p
    else:
        node_sizes = [base * (0.60 + float(np.clip(node_score[k], 0.0, 1.0))) for k in range(p)]

    _draw_nodes_and_labels(ax, pos, node_sizes, colors)

    sel_off, used_thr, off_count = _select_directed_edges(
        B=B,
        thr_offdiag=thr_offdiag,
        thr_diag=thr_diag,
        topk_offdiag=topk_offdiag,
        min_offdiag_edges=min_offdiag_edges,
    )

    abs_w = np.abs(B[sel_off])
    widths = _edge_widths_from_abs(abs_w, w_min=0.9, w_max=4.8)
    width_iter = iter(widths.tolist())

    for dst in range(p):
        for src in range(p):
            if src == dst:
                continue
            if not sel_off[dst, src]:
                continue
            w = float(B[dst, src])

            x1, y1 = float(pos[src][0]), float(pos[src][1])
            x2, y2 = float(pos[dst][0]), float(pos[dst][1])

            reciprocal = (abs(B[src, dst]) >= used_thr)
            rad = (0.18 if src < dst else -0.18) if reciprocal else 0.08

            lw = float(next(width_iter))
            style = "-" if w >= 0 else "--"
            alpha = 0.55 + 0.45 * min(1.0, abs(w) / (np.max(abs_w) + 1e-12)) if abs_w.size else 0.8

            patch = FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                mutation_scale=12 + 2.2 * lw,
                linewidth=lw,
                linestyle=style,
                color="black",
                alpha=alpha,
                zorder=1,
            )
            ax.add_patch(patch)

    diag_count = 0
    for i in range(p):
        w = float(B[i, i])
        if abs(w) >= thr_diag:
            diag_count += 1
            _draw_self_loop(ax, float(pos[i][0]), float(pos[i][1]), w, scale=0.12)

    ax.set_title(title, fontsize=11)
    ax.set_axis_off()
    x_min, x_max, y_min, y_max = fixed_limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")

    if show_legend:
        from matplotlib.lines import Line2D
        legend_elems = [
            Line2D([0], [0], marker="o", color="w", label="Predictor (P)", markerfacecolor="tab:blue",
                   markeredgecolor="black", markersize=9),
            Line2D([0], [0], marker="o", color="w", label="Criterion (C)", markerfacecolor="tab:orange",
                   markeredgecolor="black", markersize=9),
            Line2D([0], [0], color="black", lw=2, linestyle="-", label="Positive"),
            Line2D([0], [0], color="black", lw=2, linestyle="--", label="Negative"),
        ]
        ax.legend(handles=legend_elems, loc="lower left", frameon=False, fontsize=8)

    return {
        "offdiag_edges_drawn": float(off_count),
        "diag_edges_drawn": float(diag_count),
        "used_offdiag_threshold": float(used_thr),
    }


def select_threshold_topk(W: np.ndarray, top_k: int = 15, min_thr: float = 0.05) -> float:
    p = W.shape[0]
    vals = np.abs(W[np.triu_indices(p, 1)])
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float(min_thr)
    vals_sorted = np.sort(vals)
    k = min(top_k, len(vals_sorted))
    thr = float(vals_sorted[-k]) if k > 0 else float(np.max(vals_sorted))
    return float(max(min_thr, thr))


def draw_undirected_network(
    ax,
    W: np.ndarray,
    title: str,
    thr: Optional[float],
    top_k: int,
    min_thr: float,
    pos: Dict[int, np.ndarray],
    fixed_limits: Tuple[float, float, float, float],
    show_legend: bool = True,
) -> Dict[str, float]:
    p = W.shape[0]
    colors = _node_color_list()
    node_sizes = [760.0] * p

    _draw_nodes_and_labels(ax, pos, node_sizes, colors)

    thr_used = select_threshold_topk(W, top_k=top_k, min_thr=min_thr) if thr is None else float(thr)

    edges = []
    vals = []
    for i in range(p):
        for j in range(i + 1, p):
            w = float(W[i, j])
            if abs(w) >= thr_used:
                edges.append((i, j, w))
                vals.append(abs(w))

    vals_arr = np.array(vals, dtype=float)
    widths = _edge_widths_from_abs(vals_arr, w_min=0.8, w_max=4.2).tolist()
    for (i, j, w), lw in zip(edges, widths):
        x1, y1 = float(pos[i][0]), float(pos[i][1])
        x2, y2 = float(pos[j][0]), float(pos[j][1])
        style = "-" if w >= 0 else "--"
        alpha = 0.55 + 0.45 * min(1.0, abs(w) / (np.max(vals_arr) + 1e-12)) if vals_arr.size else 0.8
        rad = 0.06 if i < j else -0.06
        patch = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-",
            linewidth=float(lw),
            linestyle=style,
            color="black",
            alpha=alpha,
            zorder=1,
        )
        ax.add_patch(patch)

    ax.set_title(f"{title}\n(thr={thr_used:.3f}, top_k={top_k})", fontsize=11)
    ax.set_axis_off()
    x_min, x_max, y_min, y_max = fixed_limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")

    if show_legend:
        from matplotlib.lines import Line2D
        legend_elems = [
            Line2D([0], [0], marker="o", color="w", label="Predictor (P)", markerfacecolor="tab:blue",
                   markeredgecolor="black", markersize=9),
            Line2D([0], [0], marker="o", color="w", label="Criterion (C)", markerfacecolor="tab:orange",
                   markeredgecolor="black", markersize=9),
            Line2D([0], [0], color="black", lw=2, linestyle="-", label="Positive"),
            Line2D([0], [0], color="black", lw=2, linestyle="--", label="Negative"),
        ]
        ax.legend(handles=legend_elems, loc="lower left", frameon=False, fontsize=8)

    return {"edges_drawn": float(len(edges)), "thr_used": float(thr_used)}


def save_heatmap(path: Path, M: np.ndarray, title: str):
    fig = plt.figure(figsize=(8.2, 6.4), constrained_layout=True)
    ax = fig.add_subplot(111)
    im = ax.imshow(M, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    labels = node_names()
    p = len(labels)
    ax.set_xticks(np.arange(p))
    ax.set_yticks(np.arange(p))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    fig.savefig(path, dpi=220)
    plt.close(fig)


# ============================================================
# Saving helpers
# ============================================================
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_matrix_csv(path: Path, M: np.ndarray):
    labels = node_names()
    df = pd.DataFrame(M, index=labels, columns=labels)
    df.to_csv(path, index=True)


def save_tv_matrices(base: Path, tv: TvKsResult, ci: Optional[BootstrapCI]):
    num = ensure_dir(base / "numerical outputs")
    vis = ensure_dir(base / "visuals")

    np.savez_compressed(
        num / "tvKS_full_arrays.npz",
        estpoints=tv.estpoints,
        bandwidth=np.array([tv.bandwidth]),
        Bhat=tv.Bhat,
        intercepts=tv.intercepts,
        alphas=tv.alphas,
        r2_node=tv.r2_node,
        partial_corr=tv.partial_corr,
        mse_point=tv.mse_point,
        ess_point=tv.ess_point,
        ci_low=(ci.B_low if ci else np.full_like(tv.Bhat, np.nan)),
        ci_high=(ci.B_high if ci else np.full_like(tv.Bhat, np.nan)),
    )

    meta = []
    m, p, _ = tv.Bhat.shape
    for ei in range(m):
        te = float(tv.estpoints[ei])
        fn = f"Bhat_time_{ei:02d}_t_{te:.3f}.csv"
        save_matrix_csv(num / fn, tv.Bhat[ei])
        entry = {"time_index": ei, "t": te, "Bhat_csv": fn}

        fnc = f"partialcorr_time_{ei:02d}_t_{te:.3f}.csv"
        save_matrix_csv(num / fnc, tv.partial_corr[ei])
        entry["partialcorr_csv"] = fnc

        if ci is not None:
            fnl = f"CI_low_time_{ei:02d}_t_{te:.3f}.csv"
            fnh = f"CI_high_time_{ei:02d}_t_{te:.3f}.csv"
            save_matrix_csv(num / fnl, ci.B_low[ei])
            save_matrix_csv(num / fnh, ci.B_high[ei])
            entry["ci_low_csv"] = fnl
            entry["ci_high_csv"] = fnh

        meta.append(entry)

    (num / "time_index_map.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return num, vis


def save_stationary_outputs(base: Path, st: StationaryGvarResult):
    num = ensure_dir(base / "numerical outputs")
    vis = ensure_dir(base / "visuals")

    save_matrix_csv(num / "B_stationary.csv", st.B)
    save_matrix_csv(num / "residual_cov.csv", st.residual_cov)
    save_matrix_csv(num / "partial_corr.csv", st.partial_corr)

    np.savez_compressed(
        num / "stationary_full_arrays.npz",
        B=st.B,
        intercept=st.intercept,
        alphas=st.alphas,
        r2_node=st.r2_node,
        residual_cov=st.residual_cov,
        partial_corr=st.partial_corr,
    )
    return num, vis


def save_corr_outputs(base: Path, cr: CorrResult):
    num = ensure_dir(base / "numerical outputs")
    vis = ensure_dir(base / "visuals")
    save_matrix_csv(num / "correlation_matrix.csv", cr.corr)
    np.savez_compressed(num / "correlation_full_arrays.npz", corr=cr.corr)
    return num, vis


# ============================================================
# Debug plots
# ============================================================
def plot_debug_error_ratio(outpath: Path, cl: TvClusterResult):
    x = np.arange(1, len(cl.ratio_stationary_over_tv) + 1)
    fig = plt.figure(figsize=(9.2, 4.2), constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.plot(x, cl.ratio_stationary_over_tv, marker="o", linewidth=1.6)
    ax.axhline(cl.threshold_ratio, linestyle="--", linewidth=1.2)
    for pk in cl.peaks:
        ax.axvline(pk + 1, linestyle=":", linewidth=1.2)
    ax.set_title("Debug: modelling-error ratio (stationary / tv) over estimation points")
    ax.set_xlabel("Estimation point (1..m)")
    ax.set_ylabel("MSE ratio (stationary / tv)")
    ax.grid(True, linestyle=":", linewidth=0.8)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_debug_ess(outpath: Path, tv: TvKsResult):
    x = np.arange(1, len(tv.estpoints) + 1)
    fig = plt.figure(figsize=(8.8, 4.0), constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.plot(x, tv.ess_point, marker="o", linewidth=1.6)
    ax.set_title("Debug: effective sample size (kernel weights) per estimation point")
    ax.set_xlabel("Estimation point (1..m)")
    ax.set_ylabel("ESS")
    ax.grid(True, linestyle=":", linewidth=0.8)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_debug_spectral_radius(outpath: Path, tv: TvKsResult):
    x = np.arange(1, len(tv.estpoints) + 1)
    rhos = [spectral_radius(tv.Bhat[i]) for i in range(len(tv.estpoints))]
    fig = plt.figure(figsize=(8.8, 4.0), constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.plot(x, rhos, marker="o", linewidth=1.6)
    ax.axhline(1.0, linestyle="--", linewidth=1.2)
    ax.set_title("Debug: spectral radius of estimated B(t) (stability sanity)")
    ax.set_xlabel("Estimation point (1..m)")
    ax.set_ylabel("Spectral radius")
    ax.grid(True, linestyle=":", linewidth=0.8)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_debug_directed_strengths(outpath: Path, B: np.ndarray, thr_offdiag: float, used_thr: float, title: str):
    p = B.shape[0]
    mask = ~np.eye(p, dtype=bool)
    vals = np.abs(B[mask]).flatten()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return
    fig = plt.figure(figsize=(8.8, 4.2), constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.hist(vals, bins=30)
    ax.axvline(thr_offdiag, linestyle="--", linewidth=1.2)
    ax.axvline(used_thr, linestyle=":", linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel("|B_ij| (off-diagonal)")
    ax.set_ylabel("Count")
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_debug_edgecount_vs_threshold(outpath: Path, W: np.ndarray, title: str):
    p = W.shape[0]
    vals = np.abs(W[np.triu_indices(p, 1)])
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return
    thr_grid = np.linspace(np.min(vals), np.max(vals), 60)
    edge_counts = []
    for thr in thr_grid:
        edge_counts.append(int(np.sum(vals >= thr)))
    fig = plt.figure(figsize=(7.6, 4.2), constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.plot(thr_grid, edge_counts, marker="o", linewidth=1.2, markersize=3)
    ax.set_title(title)
    ax.set_xlabel("Threshold |w|")
    ax.set_ylabel("Edge count (upper triangle)")
    ax.grid(True, linestyle=":", linewidth=0.8)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


# ============================================================
# Plots per method
# ============================================================
def plot_tv_main(
    outpath: Path,
    tv: TvKsResult,
    ci: Optional[BootstrapCI],
    edge_specs: List[Tuple[str, int, int]],
    snap_idx: List[int],
    thr_offdiag: float,
    thr_diag: float,
    topk_directed: int,
    min_directed_edges: int,
):
    p = tv.Bhat.shape[1]
    pos = _fixed_circular_pos(p)
    lims = _layout_limits(pos, pad=0.42)

    titles = [
        f"tv-gVAR temporal snapshot (early)\n(estpoint={snap_idx[0]+1}, bw={tv.bandwidth:.2f})",
        f"tv-gVAR temporal snapshot (mid)\n(estpoint={snap_idx[1]+1}, bw={tv.bandwidth:.2f})",
        f"tv-gVAR temporal snapshot (late)\n(estpoint={snap_idx[2]+1}, bw={tv.bandwidth:.2f})",
    ]

    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0])

    debug_stats = []
    for col, si in enumerate(snap_idx):
        ax = fig.add_subplot(gs[0, col])
        st = draw_directed_network(
            ax=ax,
            B=tv.Bhat[si],
            node_score=tv.r2_node[si],
            title=titles[col],
            thr_offdiag=thr_offdiag,
            thr_diag=thr_diag,
            topk_offdiag=topk_directed,
            min_offdiag_edges=min_directed_edges,
            pos=pos,
            fixed_limits=lims,
            show_legend=(col == 0),
        )
        debug_stats.append(st)

    ax2 = fig.add_subplot(gs[1, :])
    x = np.arange(1, len(tv.estpoints) + 1)

    for label, src, dst in edge_specs:
        y = tv.Bhat[:, dst, src]
        ax2.plot(x, y, marker="o", linewidth=1.5, label=label)
        if ci is not None:
            lo = ci.B_low[:, dst, src]
            hi = ci.B_high[:, dst, src]
            ax2.fill_between(x, lo, hi, alpha=0.2)

    ax2.set_xlabel("Estimation point (1..m)")
    ax2.set_ylabel("Estimated lag-1 coefficient")
    ax2.set_title("Designed time-varying lagged effects with bootstrap CIs (tvKS)")
    ax2.legend(ncol=2, fontsize=9, frameon=False)
    ax2.grid(True, linestyle=":", linewidth=0.8)

    fig.savefig(outpath, dpi=220)
    plt.close(fig)

    return {"snap_debug": debug_stats}


def plot_tv_contemporaneous_snapshots(outpath: Path, tv: TvKsResult, snap_idx: List[int], top_k: int):
    p = tv.Bhat.shape[1]
    pos = _fixed_circular_pos(p)
    lims = _layout_limits(pos, pad=0.42)

    fig = plt.figure(figsize=(14, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 3)
    titles = [
        f"tv-gVAR contemporaneous (partial corr)\n(estpoint={snap_idx[0]+1})",
        f"tv-gVAR contemporaneous (partial corr)\n(estpoint={snap_idx[1]+1})",
        f"tv-gVAR contemporaneous (partial corr)\n(estpoint={snap_idx[2]+1})",
    ]
    for col, si in enumerate(snap_idx):
        ax = fig.add_subplot(gs[0, col])
        draw_undirected_network(
            ax=ax,
            W=tv.partial_corr[si],
            title=titles[col],
            thr=None,
            top_k=top_k,
            min_thr=0.05,
            pos=pos,
            fixed_limits=lims,
            show_legend=(col == 0),
        )
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_stationary_summary(
    outpath: Path,
    st: StationaryGvarResult,
    thr_offdiag: float,
    thr_diag: float,
    topk_directed: int,
    min_directed_edges: int,
    top_k_undirected: int,
):
    p = st.B.shape[0]
    pos = _fixed_circular_pos(p)
    lims = _layout_limits(pos, pad=0.42)

    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    gs = fig.add_gridspec(1, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    draw_directed_network(
        ax=ax1,
        B=st.B,
        node_score=st.r2_node,
        title="Stationary gVAR: temporal (VAR(1))",
        thr_offdiag=thr_offdiag,
        thr_diag=thr_diag,
        topk_offdiag=topk_directed,
        min_offdiag_edges=min_directed_edges,
        pos=pos,
        fixed_limits=lims,
        show_legend=True,
    )

    ax2 = fig.add_subplot(gs[0, 1])
    draw_undirected_network(
        ax=ax2,
        W=st.partial_corr,
        title="Stationary gVAR: residual partial corr",
        thr=None,
        top_k=top_k_undirected,
        min_thr=0.05,
        pos=pos,
        fixed_limits=lims,
        show_legend=True,
    )

    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_corr_outputs(heatmap_path: Path, network_path: Path, corr: np.ndarray, top_k: int):
    save_heatmap(heatmap_path, corr, "Pearson correlation matrix")

    p = corr.shape[0]
    pos = _fixed_circular_pos(p)
    lims = _layout_limits(pos, pad=0.42)

    fig = plt.figure(figsize=(6.8, 6.2))
    ax = fig.add_subplot(111)
    draw_undirected_network(
        ax=ax,
        W=corr,
        title="Correlation network",
        thr=None,
        top_k=top_k,
        min_thr=0.05,
        pos=pos,
        fixed_limits=lims,
        show_legend=True,
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.90)
    fig.savefig(network_path, dpi=220, facecolor="white")
    plt.close(fig)


# ============================================================
# GIF creation for tv networks
# ============================================================
def make_tv_gif(
    out_gif: Path,
    frame_dir: Path,
    tv: TvKsResult,
    cluster: TvClusterResult,
    thr_offdiag: float,
    thr_diag: float,
    topk_directed: int,
    min_directed_edges: int,
    fps: int,
):
    ensure_dir(frame_dir)
    p = tv.Bhat.shape[1]
    pos = _fixed_circular_pos(p)
    lims = _layout_limits(pos, pad=0.45)

    seg_id = [-1] * len(tv.estpoints)
    if cluster.create_clusters:
        for si, (a, b) in enumerate(cluster.segments):
            for e in range(a, b + 1):
                seg_id[e] = si

    frames: List[Image.Image] = []
    for ei, te in enumerate(tv.estpoints):
        fig = plt.figure(figsize=(6.8, 6.6))
        ax = fig.add_subplot(111)
        extra = f" seg={seg_id[ei]}" if cluster.create_clusters else ""
        title = (
            f"tv temporal network | estpoint={ei+1} t={te:.3f}{extra}\n"
            f"ratio(stat/tv)={cluster.ratio_stationary_over_tv[ei]:.2f}"
        )
        draw_directed_network(
            ax=ax,
            B=tv.Bhat[ei],
            node_score=tv.r2_node[ei],
            title=title,
            thr_offdiag=thr_offdiag,
            thr_diag=thr_diag,
            topk_offdiag=topk_directed,
            min_offdiag_edges=min_directed_edges,
            pos=pos,
            fixed_limits=lims,
            show_legend=True,
        )
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.90)

        fp = frame_dir / f"frame_{ei:03d}.png"
        fig.savefig(fp, dpi=220, facecolor="white")
        plt.close(fig)
        frames.append(Image.open(fp).convert("P", palette=Image.ADAPTIVE))

    duration_ms = int(1000 / max(1, fps))
    frames[0].save(
        out_gif,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


# ============================================================
# Optional: R/mgm inside Python (kept)
# ============================================================
R_MGM_SCRIPT = """\
args <- commandArgs(trailingOnly=TRUE)
data_csv <- args[1]
out_num  <- args[2]
out_vis  <- args[3]

suppressMessages({
  if (!requireNamespace('mgm', quietly=TRUE)) {
    install.packages('mgm', repos='https://cloud.r-project.org')
  }
  if (!requireNamespace('qgraph', quietly=TRUE)) {
    install.packages('qgraph', repos='https://cloud.r-project.org')
  }
})

library(mgm)
library(qgraph)

X <- as.matrix(read.csv(data_csv, header=FALSE))
p <- ncol(X)

set.seed(42)
bw_seq <- seq(0.03, 0.36, length.out=8)
bw_sel <- bwSelect(data=X, type='tvmvar', p=1, bwSeq=bw_seq, scale=TRUE)

m <- 20
estpoints <- seq(1, nrow(X), length.out=m)

fit <- tvmvar(data=X, p=1, scale=TRUE, bandwidth=bw_sel$bw, estpoints=estpoints, lambdaSel='CV')
res <- resample(fit, nB=50, quantiles=c(.05,.95), blocks=10)

wadj <- fit$wadj
signs <- fit$signs
Bhat <- wadj * signs

ci_wadj <- res$wadj
ci_signs <- res$signs
ci_lo <- ci_wadj[[1]] * ci_signs[[1]]
ci_hi <- ci_wadj[[2]] * ci_signs[[2]]

to_long <- function(A, name) {
  out <- data.frame()
  for (k in 1:dim(A)[3]) {
    M <- A[,,k]
    tmp <- expand.grid(dst=1:p, src=1:p)
    tmp$time_index <- k-1
    tmp$value <- as.vector(M)
    out <- rbind(out, tmp)
  }
  names(out)[4] <- name
  out
}

df_b <- to_long(Bhat, 'beta')
df_lo <- to_long(ci_lo, 'ci_low')
df_hi <- to_long(ci_hi, 'ci_high')
df <- merge(df_b, df_lo, by=c('dst','src','time_index'))
df <- merge(df, df_hi, by=c('dst','src','time_index'))

write.csv(df, file=file.path(out_num, 'r_mgm_tv_long.csv'), row.names=FALSE)
write.csv(data.frame(estpoints=estpoints, bw=bw_sel$bw), file=file.path(out_num, 'r_mgm_meta.csv'), row.names=FALSE)

png(file.path(out_vis, 'r_mgm_temporal_snapshots.png'), width=1400, height=450)
par(mfrow=c(1,3), mar=c(1,1,3,1))
snap <- c(2,10,18)
for (s in snap) {
  qgraph(Bhat[,,s], directed=TRUE, edge.labels=FALSE, layout='circle',
         title=paste0('R/mgm tvmvar temporal snapshot (estpoint=', s, ')'))
}
dev.off()
"""


def run_r_mgm_if_available(data_csv: Path, out_num: Path, out_vis: Path) -> Dict[str, str]:
    status: Dict[str, str] = {"ran": "false", "reason": ""}
    rscript = shutil.which("Rscript")
    if rscript is None:
        status["reason"] = "Rscript not found on PATH; skipped R/mgm step."
        return status

    r_file = out_num / "run_mgm_tv.R"
    r_file.write_text(R_MGM_SCRIPT, encoding="utf-8")

    try:
        proc = subprocess.run([rscript, str(r_file), str(data_csv), str(out_num), str(out_vis)],
                              capture_output=True, text=True, check=False)
        (out_num / "r_mgm_stdout.txt").write_text(proc.stdout, encoding="utf-8")
        (out_num / "r_mgm_stderr.txt").write_text(proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            status["reason"] = f"Rscript returned non-zero code: {proc.returncode}. See r_mgm_stderr.txt."
            return status
    except Exception as e:
        status["reason"] = f"Exception running Rscript: {e}"
        return status

    status["ran"] = "true"
    status["reason"] = "OK"
    return status


# ============================================================
# Outdir selection
# ============================================================
def pick_outdir(desired: str) -> Path:
    p = Path(desired).expanduser()
    try:
        p.mkdir(parents=True, exist_ok=True)
        testfile = p / ".write_test"
        testfile.write_text("ok", encoding="utf-8")
        testfile.unlink()
        return p
    except Exception:
        fallback = Path.cwd() / "results"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


# ============================================================
# Scenario runner
# ============================================================
def run_one_scenario(
    base_out: Path,
    scenario: str,
    n: int,
    m: int,
    cv_m: int,
    seed: int,
    boot: int,
    block_len: int,
    jobs: int,
    run_r: bool,
    ratio_threshold: float,
    gif_fps: int,
    topk_undirected: int,
    thr_offdiag: float,
    thr_diag: float,
    topk_directed: int,
    min_directed_edges: int,
    alpha_grid: np.ndarray,
    bw_grid: np.ndarray,
    cv_splits: int,
    standardize_in_cv: bool,
    ridge_precision: float,
    alpha_floor: float,
):
    p = 10
    scen_dir = ensure_dir(base_out / f"scenario_{scenario}")
    m1 = ensure_dir(scen_dir / "method 1")
    m2 = ensure_dir(scen_dir / "method 2")
    m3 = ensure_dir(scen_dir / "method 3")
    data_dir = ensure_dir(scen_dir / "data")

    # ---- simulate raw data
    X_raw, t_norm, B_true_all = simulate_tvvar(n=n, p=p, burnin=150, seed=seed, scenario=scenario)

    # save raw
    pd.DataFrame(X_raw, columns=node_names()).to_csv(data_dir / "X_raw.csv", index=False)
    pd.DataFrame({"t_norm": t_norm}).to_csv(data_dir / "t_norm.csv", index=False)

    # final-fit standardized data (common in VAR; for CV we do leakage-safe standardization per fold)
    X = zscore(X_raw)

    # ---- tv hyperparameter CV (5-fold TimeSeriesSplit)
    estpoints_cv = np.linspace(0.0, 1.0, int(cv_m))
    alpha_fixed_for_bw = float(np.median(alpha_grid))

    bw_sel, bw_scores = select_bandwidth_tv_cv(
        X_raw=X_raw,
        t_norm=t_norm,
        estpoints_cv=estpoints_cv,
        bw_grid=bw_grid,
        alpha_fixed=alpha_fixed_for_bw,
        n_splits=cv_splits,
        seed=seed,
        standardize_in_cv=standardize_in_cv,
        ridge_precision=ridge_precision,
    )

    alpha_sel, alpha_scores = select_alpha_tv_cv(
        X_raw=X_raw,
        t_norm=t_norm,
        estpoints_cv=estpoints_cv,
        bandwidth_fixed=bw_sel,
        alpha_grid=alpha_grid,
        n_splits=cv_splits,
        seed=seed,
        standardize_in_cv=standardize_in_cv,
        ridge_precision=ridge_precision,
    )

    # optional enforced sparsity
    alpha_sel_eff = float(max(alpha_sel, alpha_floor))

    cv_summary = TvCvSummary(
        n_splits=int(cv_splits),
        standardize_in_cv=bool(standardize_in_cv),
        alpha_fixed_for_bw=float(alpha_fixed_for_bw),
        bw_grid=[float(x) for x in bw_grid.tolist()],
        bw_cv_mse=[float(x) for x in bw_scores],
        bw_selected=float(bw_sel),
        alpha_grid=[float(x) for x in alpha_grid.tolist()],
        alpha_cv_mse=[float(x) for x in alpha_scores],
        alpha_selected=float(alpha_sel_eff),
    )
    (m1 / "tv_hyperparam_cv.json").write_text(json.dumps(cv_summary.__dict__, indent=2), encoding="utf-8")

    # ---- final tv fit with selected hyperparams
    estpoints = np.linspace(0.0, 1.0, m)
    alphas_node = np.full(p, float(alpha_sel_eff), dtype=float)

    tv = estimate_tvKS(
        X=X,
        t_norm=t_norm,
        estpoints=estpoints,
        bandwidth=float(bw_sel),
        alphas_node=alphas_node,
        seed=seed,
        ridge_precision=ridge_precision,
    )

    ci = bootstrap_tvKS_CI(
        X, t_norm, tv, alphas_node,
        n_boot=boot, block_len=block_len, ci=(0.05, 0.95), seed=seed, n_jobs=max(1, int(jobs))
    )

    num1, vis1 = save_tv_matrices(m1, tv, ci)

    edge_specs = [
        (f"{node_names()[0]}(t-1) → {node_names()[0]}(t)", 0, 0),
        (f"{node_names()[4]}(t-1) → {node_names()[1]}(t)", 4, 1),
        (f"{node_names()[7]}(t-1) → {node_names()[3]}(t)", 7, 3),
        (f"{node_names()[2]}(t-1) → {node_names()[6]}(t)", 2, 6),
        (f"{node_names()[9]}(t-1) → {node_names()[8]}(t)", 9, 8),
    ]

    snap_idx = [1, max(1, m // 2 - 1), max(2, m - 3)]
    snap_dbg = plot_tv_main(
        vis1 / "tv_dynamic_gvar_with_CI.png",
        tv, ci,
        edge_specs=edge_specs,
        snap_idx=snap_idx,
        thr_offdiag=thr_offdiag,
        thr_diag=thr_diag,
        topk_directed=topk_directed,
        min_directed_edges=min_directed_edges,
    )
    plot_tv_contemporaneous_snapshots(vis1 / "tv_contemporaneous_snapshots.png", tv, snap_idx=snap_idx, top_k=topk_undirected)

    mid = snap_idx[1]
    save_heatmap(vis1 / "tv_temporal_heatmap_mid.png", tv.Bhat[mid], f"tvKS temporal B (mid estpoint={mid+1})")
    save_heatmap(vis1 / "tv_contemp_partialcorr_mid.png", tv.partial_corr[mid], f"tvKS partial corr (mid estpoint={mid+1})")

    # long-format edges for downstream workflows
    rows = []
    names = node_names()
    for ei in range(m):
        te = float(tv.estpoints[ei])
        for dst in range(p):
            for src in range(p):
                rows.append({
                    "time_index": ei, "t": te,
                    "dst": names[dst], "src": names[src],
                    "beta": float(tv.Bhat[ei, dst, src]),
                    "ci_low": float(ci.B_low[ei, dst, src]),
                    "ci_high": float(ci.B_high[ei, dst, src]),
                })
    pd.DataFrame(rows).to_csv(num1 / "tv_temporal_long_with_CI.csv", index=False)

    # ---- stationary
    st = fit_stationary_gvar(X, alpha_grid=alpha_grid, seed=seed)
    num2, vis2 = save_stationary_outputs(m2, st)
    plot_stationary_summary(
        vis2 / "stationary_gvar_summary.png",
        st,
        thr_offdiag=thr_offdiag,
        thr_diag=thr_diag,
        topk_directed=topk_directed,
        min_directed_edges=min_directed_edges,
        top_k_undirected=topk_undirected,
    )
    save_heatmap(vis2 / "stationary_temporal_B_heatmap.png", st.B, "Stationary VAR(1) B")
    save_heatmap(vis2 / "stationary_partialcorr_heatmap.png", st.partial_corr, "Stationary residual partial corr")

    # ---- correlation
    cr = compute_corr(X)
    num3, vis3 = save_corr_outputs(m3, cr)
    plot_corr_outputs(vis3 / "correlation_heatmap.png", vis3 / "correlation_network.png", cr.corr, top_k=topk_undirected)

    # ---- cluster heuristic
    cl = detect_tv_clusters_by_error_ratio(X, t_norm, tv, st, ratio_threshold=ratio_threshold)
    (num1 / "tv_cluster_detection.json").write_text(json.dumps({
        "create_clusters": cl.create_clusters,
        "threshold_ratio": cl.threshold_ratio,
        "peaks": cl.peaks,
        "segments": cl.segments,
        "ratio_stationary_over_tv": cl.ratio_stationary_over_tv.tolist(),
    }, indent=2), encoding="utf-8")

    # debug plots
    plot_debug_error_ratio(vis1 / "debug_error_ratio.png", cl)
    plot_debug_ess(vis1 / "debug_effective_sample_size.png", tv)
    plot_debug_spectral_radius(vis1 / "debug_spectral_radius.png", tv)

    used_thr_mid = float(snap_dbg["snap_debug"][1]["used_offdiag_threshold"]) if "snap_debug" in snap_dbg else float(thr_offdiag)
    plot_debug_directed_strengths(
        vis1 / "debug_directed_strengths_mid.png",
        tv.Bhat[mid],
        thr_offdiag=float(thr_offdiag),
        used_thr=float(used_thr_mid),
        title="Debug: |B_ij| distribution (mid tvKS) + thresholds",
    )
    plot_debug_edgecount_vs_threshold(
        vis1 / "debug_edgecount_partialcorr_mid.png",
        tv.partial_corr[mid],
        "Debug: partial corr (mid) edge count vs threshold",
    )
    plot_debug_edgecount_vs_threshold(
        vis3 / "debug_edgecount_corr.png",
        cr.corr,
        "Debug: correlation edge count vs threshold",
    )

    # ---- gif
    gif_path = vis1 / "tv_temporal_network.gif"
    frame_dir = vis1 / "gif_frames"
    make_tv_gif(
        out_gif=gif_path,
        frame_dir=frame_dir,
        tv=tv,
        cluster=cl,
        thr_offdiag=thr_offdiag,
        thr_diag=thr_diag,
        topk_directed=topk_directed,
        min_directed_edges=min_directed_edges,
        fps=gif_fps,
    )

    # ---- metrics summary
    tv_rmse = []
    tv_mae = []
    tv_f1 = []
    for ei, te in enumerate(tv.estpoints):
        B_true = nearest_true_B(B_true_all, t_norm, float(te))
        tv_rmse.append(rmse(tv.Bhat[ei], B_true))
        tv_mae.append(mae(tv.Bhat[ei], B_true))
        tv_f1.append(edge_recovery_scores(tv.Bhat[ei], B_true)["f1"])

    m1_metrics = {
        "tv_cv": cv_summary.__dict__,
        "bandwidth_selected_final": float(tv.bandwidth),
        "alpha_selected_final": float(alpha_sel_eff),
        "bootstrap": {"n_boot": int(boot), "block_len": int(block_len), "ci": [0.05, 0.95]},
        "prediction": one_step_prediction_metrics_tv(X, t_norm, tv),
        "temporal_matrix_error_vs_true": {
            "rmse_mean": float(np.mean(tv_rmse)),
            "mae_mean": float(np.mean(tv_mae)),
            "f1_mean": float(np.mean(tv_f1)),
        },
        "tv_clusters": {"create_clusters": cl.create_clusters, "threshold_ratio": cl.threshold_ratio, "peaks": cl.peaks, "segments": cl.segments},
        "debug": {
            "ess_mean": float(np.mean(tv.ess_point)),
            "spectral_radius_mean": float(np.mean([spectral_radius(tv.Bhat[i]) for i in range(len(tv.estpoints))])),
            "nonzero_fraction": float(np.mean(np.abs(tv.Bhat) > 1e-6)),
        }
    }
    (num1 / "metrics.json").write_text(json.dumps(m1_metrics, indent=2), encoding="utf-8")

    B_true_avg = np.mean(B_true_all, axis=0)
    m2_metrics = {
        "prediction": one_step_prediction_metrics_stationary(X, st),
        "temporal_matrix_error_vs_true_avg": {
            "rmse": rmse(st.B, B_true_avg),
            "mae": mae(st.B, B_true_avg),
            **edge_recovery_scores(st.B, B_true_avg),
        },
        "r2_node_mean": float(np.mean(st.r2_node)),
        "alphas": st.alphas.tolist(),
    }
    (num2 / "metrics.json").write_text(json.dumps(m2_metrics, indent=2), encoding="utf-8")

    m3_metrics = {
        "avg_abs_corr_offdiag": float(np.mean(np.abs(cr.corr[~np.eye(p, dtype=bool)]))),
        "max_abs_corr_offdiag": float(np.max(np.abs(cr.corr[~np.eye(p, dtype=bool)]))),
        "threshold_topk": int(topk_undirected),
    }
    (num3 / "metrics.json").write_text(json.dumps(m3_metrics, indent=2), encoding="utf-8")

    # optional R/mgm
    r_status = {"ran": "false", "reason": "not requested"}
    if run_r:
        data_csv = data_dir / "X_noheader.csv"
        np.savetxt(data_csv, X, delimiter=",")
        r_status = run_r_mgm_if_available(data_csv, num1, vis1)
        (num1 / "r_mgm_status.json").write_text(json.dumps(r_status, indent=2), encoding="utf-8")

    summary = {
        "scenario": scenario,
        "base_outdir": str(scen_dir),
        "method 1": {"metrics": m1_metrics, "r_mgm_status": r_status},
        "method 2": {"metrics": m2_metrics},
        "method 3": {"metrics": m3_metrics},
        "files": {
            "method1_main_plot": str((vis1 / "tv_dynamic_gvar_with_CI.png").relative_to(scen_dir)),
            "method1_gif": str((vis1 / "tv_temporal_network.gif").relative_to(scen_dir)),
            "method2_main_plot": str((vis2 / "stationary_gvar_summary.png").relative_to(scen_dir)),
            "method3_main_plot": str((vis3 / "correlation_heatmap.png").relative_to(scen_dir)),
        },
    }
    (scen_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[{scenario}] done -> {scen_dir}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str,
                        default="/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/other/other/testing_timevarying_gVAR/results")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=800)
    parser.add_argument("--m", type=int, default=20)

    # CV for tv model
    parser.add_argument("--cv-m", type=int, default=12, help="Estimation points used during tv hyperparameter CV.")
    parser.add_argument("--cv-splits", type=int, default=5, help="TimeSeriesSplit folds for tv hyperparameter CV.")
    parser.add_argument("--standardize-in-cv", action="store_true", help="Leakage-safe standardization inside CV (recommended). Default off unless set.")
    parser.add_argument("--ridge-precision", type=float, default=1e-3, help="Ridge for precision inversion in tv partial correlations.")

    # Alpha grid (shared across tv and stationary by default)
    parser.add_argument("--alpha-min-exp", type=float, default=-2.0, help="log10(alpha_min) for grid")
    parser.add_argument("--alpha-max-exp", type=float, default=0.2, help="log10(alpha_max) for grid")
    parser.add_argument("--alpha-num", type=int, default=18, help="Number of alphas in grid")
    parser.add_argument("--alpha-floor", type=float, default=0.0, help="Hard lower bound on selected alpha (enforce more sparsity).")

    # Bandwidth grid
    parser.add_argument("--bw-grid", type=str, default="0.03,0.05,0.08,0.11,0.15,0.20,0.28,0.36",
                        help="Comma-separated bandwidth grid in [0,1].")

    parser.add_argument("--boot", type=int, default=80)
    parser.add_argument("--block-len", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--run-r", action="store_true")

    parser.add_argument("--ratio-threshold", type=float, default=3.0,
                        help="Cluster-creation threshold on MSE ratio (stationary / tv).")
    parser.add_argument("--gif-fps", type=int, default=3)

    parser.add_argument("--topk-undirected", type=int, default=15,
                        help="Undirected networks keep approximately top-k strongest edges.")
    parser.add_argument("--thr-offdiag", type=float, default=0.06,
                        help="Base threshold for off-diagonal VAR(1) edges (tv + stationary).")
    parser.add_argument("--thr-diag", type=float, default=0.08,
                        help="Threshold for autoregressive self loops.")

    parser.add_argument("--topk-directed", type=int, default=26,
                        help="Directed networks cap off-diagonal edges to top-k strongest (safety).")
    parser.add_argument("--min-directed-edges", type=int, default=10,
                        help="If threshold yields too few off-diagonal edges, expand up to at least this many via top-k. Use 0 to see true sparsity.")

    parser.add_argument("--scenarios", type=str, default="scenario1_smooth,scenario2_abrupt,scenario3_periodic",
                        help="Comma-separated scenario list.")

    args = parser.parse_args()
    base_out = pick_outdir(args.outdir)

    alpha_grid = np.logspace(float(args.alpha_min_exp), float(args.alpha_max_exp), int(args.alpha_num))
    bw_grid = np.array([float(x.strip()) for x in args.bw_grid.split(",") if x.strip()], dtype=float)

    scenario_list = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    for i, scen in enumerate(scenario_list):
        run_one_scenario(
            base_out=base_out,
            scenario=scen,
            n=int(args.n),
            m=int(args.m),
            cv_m=int(args.cv_m),
            seed=int(args.seed) + 11 * i,
            boot=int(args.boot),
            block_len=int(args.block_len),
            jobs=int(args.jobs),
            run_r=bool(args.run_r),
            ratio_threshold=float(args.ratio_threshold),
            gif_fps=int(args.gif_fps),
            topk_undirected=int(args.topk_undirected),
            thr_offdiag=float(args.thr_offdiag),
            thr_diag=float(args.thr_diag),
            topk_directed=int(args.topk_directed),
            min_directed_edges=int(args.min_directed_edges),
            alpha_grid=alpha_grid,
            bw_grid=bw_grid,
            cv_splits=int(args.cv_splits),
            standardize_in_cv=bool(args.standardize_in_cv),
            ridge_precision=float(args.ridge_precision),
            alpha_floor=float(args.alpha_floor),
        )

    print("All scenarios complete.")
    print("Output base:", base_out)


if __name__ == "__main__":
    main()

