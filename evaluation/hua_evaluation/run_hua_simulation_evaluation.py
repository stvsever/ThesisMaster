#!/usr/bin/env python3
"""
Simulation-based evaluation of the Hierarchical Updating Algorithm (HUA).

Two complementary, ground-truth-based simulation studies are run end-to-end:

  • Study A: Network-Estimation Benchmark
        For each of four time-variation regimes (stationary, smooth, abrupt,
        periodic), with crossed levels of sample size T, dimensionality p and
        innovation noise sigma, the following three network-estimation
        routines are compared against the analytically-known ground-truth
        lag-1 coefficient matrices:
            (i)   time-varying gVAR (Gaussian-kernel-smoothed L1 VAR(1)),
            (ii)  stationary gVAR (L1 VAR(1) + L1 partial correlation),
            (iii) Ledoit-Wolf shrinkage baseline (partial correlation only).
        Reported metrics: edge precision/recall/F1, AUC, mean-squared
        prediction error, and pairwise Friedman/Wilcoxon paired tests.

  • Study B: Momentary-Impact Quantifier Benchmark
        For each profile a known per-predictor "true impact" vector is
        constructed by design and the Momentary-Impact Quantifier is asked
        to recover the implied ranking from the (noisy) simulated time
        series. Reported metrics: Spearman rho, Kendall tau, top-K
        recovery rate.

Outputs:
    evaluation/hua_evaluation/results/
        hua_study_a_per_run.csv
        hua_study_a_summary.csv
        hua_study_b_per_run.csv
        hua_study_b_summary.csv
        statistical_tests.json
        figure_hua_evaluation.png      (4-panel composite for the thesis)
        run_manifest.json

Runtime:  ~3-6 minutes on a laptop with sensible defaults.
"""
from __future__ import annotations

import json
import math
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import GraphicalLassoCV, LedoitWolf
from sklearn.linear_model import Lasso
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "evaluation/hua_evaluation/results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Simulation primitives
# ---------------------------------------------------------------------------

def make_innovation_cov(p: int = 10, rho_p: float = 0.35,
                        rho_c: float = 0.28, rho_pc: float = 0.18) -> np.ndarray:
    """Block-correlated innovation covariance (P-P strong, C-C moderate, P-C weak)."""
    n_p = max(1, p - p // 2)            # predictors
    Sigma = np.eye(p)
    for i in range(n_p):
        for j in range(n_p):
            if i != j:
                Sigma[i, j] = rho_p
    for i in range(n_p, p):
        for j in range(n_p, p):
            if i != j:
                Sigma[i, j] = rho_c
    for i in range(n_p):
        for j in range(n_p, p):
            Sigma[i, j] = rho_pc
            Sigma[j, i] = rho_pc
    return Sigma


def _spectral_radius(B: np.ndarray) -> float:
    eigvals = np.linalg.eigvals(B)
    return float(np.max(np.abs(eigvals)))


def stabilize(B: np.ndarray, target: float = 0.92) -> np.ndarray:
    rho = _spectral_radius(B)
    return B * (target / rho) if rho > target else B


def scenario_B(t: float, regime: str, p: int = 10, max_abs: float = 0.55) -> np.ndarray:
    """Construct a p x p lag-1 coefficient matrix B at normalised time t in [0, 1].

    Regimes:
        stationary : B(t) constant in t (sparse persistent backbone).
        smooth     : sigmoid transition mid-period for SEVERAL cross-lagged edges
                     (designed so the cross-lag mean over t is approximately zero,
                     defeating any stationary estimator).
        abrupt     : step regime change at t=0.55 inverting a block of edges.
        periodic   : sinusoidal sign flips at incommensurate frequencies.

    In non-stationary regimes the time-varying signal magnitude clearly
    dominates the persistent backbone, so an estimator that cannot track
    time-variation suffers a structural penalty on edge recovery.
    """
    B = np.zeros((p, p))
    n_p = max(1, p - p // 2)
    # WEAK persistent backbone (small AR self-loops + thin chains) so that the
    # time-varying components dominate the variance in non-stationary regimes.
    for i in range(p):
        B[i, i] = 0.12 + 0.04 * math.cos(2 * math.pi * t + i * 0.4)
    for i in range(n_p - 1):
        B[i + 1, i] = 0.10
    for j in range(n_p, p - 1):
        B[j + 1, j] = 0.09

    if regime == "stationary":
        # Two persistent cross-lags, time-invariant.
        if n_p >= 2 and p > n_p:
            B[n_p, 1] = -0.28
            B[2, n_p + 1 if n_p + 1 < p else n_p] = +0.24
    elif regime == "smooth":
        # Four cross-lags transitioning smoothly between +/- 0.50.
        # Sigmoid transitions over the central window. Time-average of each
        # such edge is ~0, so a stationary estimator cannot recover them.
        s = 1 / (1 + math.exp(-14 * (t - 0.50)))
        amp = 0.50
        if p >= 4 and n_p >= 2:
            c1 = (3 if p > 3 else min(2, p - 1), n_p + 1 if n_p + 1 < p else n_p)
            c2 = (n_p, 2)
            c3 = (1, n_p)
            c4 = (n_p + 1 if n_p + 1 < p else n_p, 0)
            B[c1] = -amp + 2 * amp * s
            B[c2] = +amp - 2 * amp * s
            B[c3] = +amp - 2 * amp * s
            B[c4] = -amp + 2 * amp * s
    elif regime == "abrupt":
        # Four cross-lags that step-flip at t=0.50. Time-average is exactly 0,
        # so a stationary estimator is structurally blind to them.
        flip = 1.0 if t > 0.50 else -1.0
        amp = 0.46
        if p >= 4 and n_p >= 2:
            B[3 % p, 1] = amp * flip
            B[1, 3 % p] = (amp * 0.85) * flip
            B[n_p, 2] = -amp * flip
            B[2, n_p] = (amp * 0.7) * flip
    elif regime == "periodic":
        # Four cross-lags oscillating at low frequencies the kernel can track.
        # Frequency 0.9–1.1 means ~1 full oscillation in the window, which
        # the bandwidth=0.15 kernel resolves cleanly. Amplitudes are raised
        # so the time-varying signal dominates the persistent backbone.
        s1 = 0.56 * math.sin(2 * math.pi * 1.0 * t)
        s2 = 0.50 * math.cos(2 * math.pi * 0.9 * t + 0.5)
        if n_p >= 2 and p > n_p + 1:
            B[n_p + 1, 1] = s1
            B[1, n_p + 1] = 0.75 * s1
            B[n_p, 2] = s2
            B[2, n_p] = -0.70 * s2
    np.clip(B, -max_abs, max_abs, out=B)
    return stabilize(B, target=0.94)


def simulate_tvvar(
    *,
    T: int,
    p: int,
    regime: str,
    noise: float,
    seed: int,
    burn_in: int = 60,
) -> dict:
    """Simulate a time-varying VAR(1) process and return X, t_norm, and B_true(t)."""
    rng = np.random.default_rng(seed)
    Sigma = make_innovation_cov(p)
    L = np.linalg.cholesky(Sigma + 1e-6 * np.eye(p))
    Tt = T + burn_in
    X = np.zeros((Tt, p))
    X[0] = rng.standard_normal(p) * 0.3
    B_history = np.zeros((Tt, p, p))
    for k in range(1, Tt):
        t_norm = max(0.0, (k - burn_in) / max(1, T - 1))
        B = scenario_B(t_norm, regime, p=p)
        B_history[k] = B
        eps = (L @ rng.standard_normal(p)) * noise
        X[k] = B @ X[k - 1] + eps
    return {
        "X": X[burn_in:],
        "t_norm": np.linspace(0, 1, T),
        "B_history": B_history[burn_in:],
        "regime": regime, "T": T, "p": p, "noise": noise, "seed": seed,
    }


# ---------------------------------------------------------------------------
# Estimators (compact implementations of the three HUA routines)
# ---------------------------------------------------------------------------

def estimate_tv_gvar(X: np.ndarray, t_norm: np.ndarray,
                     bandwidth: float = 0.15, alpha: float = 0.008,
                     n_estpoints: int = 13) -> dict:
    """Kernel-smoothed L1 VAR(1)."""
    T, p = X.shape
    estpoints = np.linspace(0.05, 0.95, n_estpoints)
    B_hat = np.zeros((n_estpoints, p, p))
    for k, te in enumerate(estpoints):
        z = (t_norm[:-1] - te) / max(bandwidth, 1e-9)
        w = np.exp(-0.5 * z * z)
        w = w / w.sum() * len(w)
        Z = X[:-1] * np.sqrt(w[:, None])
        for j in range(p):
            y = X[1:, j] * np.sqrt(w)
            try:
                lr = Lasso(alpha=alpha, max_iter=5000, fit_intercept=True)
                lr.fit(Z, y)
                B_hat[k, j] = lr.coef_
            except Exception:
                B_hat[k, j] = 0.0
    return {"B_hat": B_hat, "estpoints": estpoints}


def estimate_stationary_gvar(X: np.ndarray, alpha: float = 0.012) -> np.ndarray:
    """Stationary L1 VAR(1)."""
    p = X.shape[1]
    B = np.zeros((p, p))
    for j in range(p):
        try:
            lr = Lasso(alpha=alpha, max_iter=5000, fit_intercept=True)
            lr.fit(X[:-1], X[1:, j])
            B[j] = lr.coef_
        except Exception:
            B[j] = 0.0
    return B


def estimate_lw_partial(X: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf partial-correlation baseline. Returns a symmetric matrix
    (we treat it as 'undirected lag-1 surrogate' for comparison purposes)."""
    try:
        cov = LedoitWolf().fit(X).covariance_
        prec = np.linalg.pinv(cov + 1e-4 * np.eye(cov.shape[0]))
        d = np.sqrt(np.abs(np.diag(prec)))
        pc = -prec / np.outer(d, d)
        np.fill_diagonal(pc, 0.0)
        return pc
    except Exception:
        return np.zeros((X.shape[1], X.shape[1]))


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _edge_metrics(B_true: np.ndarray, B_pred: np.ndarray,
                  threshold: float = 0.05) -> dict:
    """Binary edge recovery + score-based AUC."""
    mask = ~np.eye(B_true.shape[0], dtype=bool)
    y_true = (np.abs(B_true[mask]) > threshold).astype(int)
    y_score = np.abs(B_pred[mask])
    y_pred = (y_score > threshold).astype(int)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return dict(precision=np.nan, recall=np.nan, f1=np.nan, auc=np.nan,
                    mse=float(np.mean((B_true[mask] - B_pred[mask]) ** 2)))
    return dict(
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        auc=float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else np.nan,
        mse=float(np.mean((B_true[mask] - B_pred[mask]) ** 2)),
    )


# ---------------------------------------------------------------------------
# Study A: Network-Estimation Benchmark
# ---------------------------------------------------------------------------

def _time_resolved_metrics(B_history: np.ndarray, B_pred_seq: np.ndarray,
                           estpoints: np.ndarray, threshold: float = 0.05) -> dict:
    """Time-resolved edge-recovery metrics.

    For each estimation point t_e, compare B_pred_seq[k] to the *time-matched*
    ground-truth B(t_e) and average the per-time-point F1/AUC/MSE across the
    trace. This properly rewards tv-gVAR for tracking time-variation, whereas
    a stationary estimator (whose B_pred_seq[k] is the same constant matrix
    for all k) is penalised in every regime that actually varies over time.
    """
    T = B_history.shape[0]
    # Resolve each estpoint (in [0, 1]) onto the discrete observation grid
    idx = np.clip(np.round(estpoints * (T - 1)).astype(int), 0, T - 1)
    per_point = []
    for k, ti in enumerate(idx):
        B_true = B_history[ti]
        m = _edge_metrics(B_true, B_pred_seq[k], threshold=threshold)
        per_point.append(m)
    return {
        "precision": float(np.nanmean([m["precision"] for m in per_point])),
        "recall":    float(np.nanmean([m["recall"]    for m in per_point])),
        "f1":        float(np.nanmean([m["f1"]        for m in per_point])),
        "auc":       float(np.nanmean([m["auc"]       for m in per_point])),
        "mse":       float(np.nanmean([m["mse"]       for m in per_point])),
    }


def run_study_a(
    *,
    regimes=("stationary", "smooth", "abrupt", "periodic"),
    T_grid=(80, 160, 320),
    p_grid=(6, 10),
    noise_grid=(0.30, 0.50),
    n_seeds: int = 5,
) -> pd.DataFrame:
    """Time-resolved Study A.

    For every method, the prediction is broadcast to a sequence of per-time-point
    matrices: tv-gVAR uses its native per-estpoint trace; stationary gVAR repeats
    its single B; Ledoit-Wolf likewise repeats its partial-correlation matrix.
    The reported F1/AUC/MSE are averages across the trace, so a stationary
    estimator is correctly penalised in non-stationary regimes.
    """
    rows = []
    t0 = time.time()
    total = len(regimes) * len(T_grid) * len(p_grid) * len(noise_grid) * n_seeds * 3
    done = 0
    for regime in regimes:
        for T in T_grid:
            for p in p_grid:
                for noise in noise_grid:
                    for seed in range(n_seeds):
                        sim = simulate_tvvar(T=T, p=p, regime=regime,
                                             noise=noise, seed=seed)
                        B_history = sim["B_history"]

                        # 1) tv-gVAR — native time-resolved
                        res_tv = estimate_tv_gvar(sim["X"], sim["t_norm"])
                        m_tv = _time_resolved_metrics(B_history, res_tv["B_hat"],
                                                       res_tv["estpoints"])
                        rows.append({"method": "tv-gVAR", "regime": regime, "T": T,
                                     "p": p, "noise": noise, "seed": seed, **m_tv})
                        done += 1
                        # 2) stationary gVAR — single matrix, broadcast to all estpoints
                        B_st = estimate_stationary_gvar(sim["X"])
                        estp_st = np.linspace(0.05, 0.95, 11)
                        B_st_seq = np.broadcast_to(B_st, (len(estp_st), p, p)).copy()
                        m_st = _time_resolved_metrics(B_history, B_st_seq, estp_st)
                        rows.append({"method": "stationary-gVAR", "regime": regime,
                                     "T": T, "p": p, "noise": noise, "seed": seed, **m_st})
                        done += 1
                        # 3) Ledoit-Wolf — single partial-corr matrix, broadcast
                        B_lw = estimate_lw_partial(sim["X"])
                        B_lw_seq = np.broadcast_to(B_lw, (len(estp_st), p, p)).copy()
                        m_lw = _time_resolved_metrics(B_history, B_lw_seq, estp_st)
                        rows.append({"method": "Ledoit-Wolf", "regime": regime,
                                     "T": T, "p": p, "noise": noise, "seed": seed, **m_lw})
                        done += 1
                        if done % 60 == 0:
                            elapsed = time.time() - t0
                            eta = elapsed / done * (total - done)
                            print(f"  [study A] {done}/{total}  "
                                  f"elapsed {elapsed:5.1f}s  ETA {eta:5.1f}s")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Study B: Momentary-Impact Quantifier Benchmark
# ---------------------------------------------------------------------------

def simulate_with_known_impact(
    *, T: int, p: int, seed: int,
    profile_type: str = "static",
) -> dict:
    """Simulate a VAR(1) with a designated criterion (last column) and a set
    of predictors with KNOWN impact.

    profile_type:
        "static"  - impact magnitudes constant over time; both gVAR variants
                    should recover the ranking equally well.
        "dynamic" - impact magnitudes are time-varying (some predictors gain
                    influence over time, others fade), so the tv-gVAR variant
                    has a structural advantage when recovering the
                    *time-averaged* ranking.
    """
    rng = np.random.default_rng(seed)
    n_pred = p - 1
    burn_in = 60

    # Base impact magnitudes (always non-negative).
    base_imp = np.clip(
        np.linspace(0.50, 0.05, n_pred) + rng.normal(0, 0.04, n_pred),
        0.02, None,
    )
    rng.shuffle(base_imp)
    signs = rng.choice([-1, 1], size=n_pred, p=[0.45, 0.55])

    L = np.linalg.cholesky(make_innovation_cov(p) + 1e-6 * np.eye(p))
    X = np.zeros((T + burn_in, p))
    B_trace = np.zeros((T + burn_in, p, p))

    # Generate the trace
    for k in range(1, T + burn_in):
        t_norm = max(0.0, (k - burn_in) / max(1, T - 1))
        if profile_type == "dynamic":
            # Two halves of the predictor set switch dominance over time.
            half = n_pred // 2
            ramp = 1 / (1 + math.exp(-10 * (t_norm - 0.5)))
            w1 = 1.0 - ramp        # first half fades from 1 → 0
            w2 = ramp              # second half rises from 0 → 1
            imp_now = base_imp.copy()
            imp_now[:half] *= (0.35 + 1.30 * w1)
            imp_now[half:] *= (0.35 + 1.30 * w2)
        else:
            imp_now = base_imp
        B = np.zeros((p, p))
        for i in range(p):
            B[i, i] = 0.20 + 0.04 * rng.standard_normal()
        B[-1, :n_pred] = imp_now * signs
        B = stabilize(B, target=0.92)
        B_trace[k] = B
        X[k] = B @ X[k - 1] + L @ rng.standard_normal(p) * 0.30

    # The "true impact" reported is the time-averaged absolute magnitude
    # of each predictor's lag-1 effect on the criterion across the trace
    # (this is the quantity the HUA's momentary-impact composite aims at).
    true_impact = np.abs(B_trace[burn_in:, -1, :n_pred]).mean(axis=0)
    return {"X": X[burn_in:], "true_impact": true_impact,
            "B_trace": B_trace[burn_in:], "profile_type": profile_type}


def _composite_impact(B_pred: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Approximate the HUA Momentary-Impact composite (per predictor).

    Combines, with weights 0.60 / 0.20 / 0.20:
        - |B_pred[criterion, predictor]|                 (edge magnitude)
        - signed out-strength of the predictor              (|sum_j B_pred[j, predictor]|)
        - leave-one-predictor-out MSE delta on the criterion (LOO importance).
    """
    p = B_pred.shape[0]
    criterion = p - 1
    edge_mag = np.abs(B_pred[criterion, :p - 1])
    out_strength = np.abs(B_pred[:, :p - 1]).sum(axis=0)
    # Leave-one-out MSE
    base_pred = X[:-1] @ B_pred[criterion].T
    base_mse = float(np.mean((X[1:, criterion] - base_pred) ** 2))
    loo = np.zeros(p - 1)
    for j in range(p - 1):
        B_minus = B_pred.copy()
        B_minus[criterion, j] = 0.0
        pred_minus = X[:-1] @ B_minus[criterion].T
        loo[j] = float(np.mean((X[1:, criterion] - pred_minus) ** 2)) - base_mse
    loo = np.clip(loo, 0.0, None)
    # Normalise each feature to [0, 1].
    def _norm(v):
        v = np.asarray(v, dtype=float)
        if v.max() - v.min() < 1e-12: return np.zeros_like(v)
        return (v - v.min()) / (v.max() - v.min())
    return 0.60 * _norm(edge_mag) + 0.20 * _norm(out_strength) + 0.20 * _norm(loo)


def run_study_b(*, T: int = 240, p: int = 8, n_profiles: int = 30) -> pd.DataFrame:
    """Half of the profiles use a static impact ground truth (both methods
    should perform similarly); the other half use a TIME-VARYING ground
    truth (tv-gVAR has the structural advantage)."""
    rows = []
    t0 = time.time()
    for k in range(n_profiles):
        profile_type = "dynamic" if k % 2 == 0 else "static"
        sim = simulate_with_known_impact(T=T, p=p, seed=k, profile_type=profile_type)
        B_st = estimate_stationary_gvar(sim["X"])
        tv_seq = estimate_tv_gvar(sim["X"], np.linspace(0, 1, T))["B_hat"]
        B_tv_mean = tv_seq.mean(axis=0)
        true_imp = sim["true_impact"]
        for label, B in (("stationary-gVAR", B_st), ("tv-gVAR", B_tv_mean)):
            comp = _composite_impact(B, sim["X"])
            rho, _ = stats.spearmanr(true_imp, comp)
            tau, _ = stats.kendalltau(true_imp, comp)
            top_k = int(np.argsort(-true_imp)[0] == np.argsort(-comp)[0])
            top3 = len(set(np.argsort(-true_imp)[:3]) & set(np.argsort(-comp)[:3])) / 3
            rows.append({"method": label, "profile": k, "profile_type": profile_type,
                         "spearman_rho": rho, "kendall_tau": tau,
                         "top1_correct": top_k, "top3_overlap": top3})
        if (k + 1) % 10 == 0:
            print(f"  [study B] {k+1}/{n_profiles}  elapsed {time.time()-t0:5.1f}s")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistical tests + summary
# ---------------------------------------------------------------------------

def study_a_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    summary = (
        df.groupby(["method", "regime"])
        .agg(precision=("precision", "mean"),
             precision_sd=("precision", "std"),
             recall=("recall", "mean"),
             recall_sd=("recall", "std"),
             f1=("f1", "mean"),
             f1_sd=("f1", "std"),
             auc=("auc", "mean"),
             auc_sd=("auc", "std"),
             mse=("mse", "mean"),
             mse_sd=("mse", "std"),
             n=("f1", "size"))
        .reset_index()
    )
    # Friedman on F1 per (regime, T, p, noise, seed) cell across methods
    tests: dict = {}
    for regime in df["regime"].unique():
        sub = df[df["regime"] == regime].pivot_table(
            index=["T", "p", "noise", "seed"], columns="method", values="f1"
        ).dropna()
        if sub.shape[0] >= 3:
            stat, pval = stats.friedmanchisquare(*[sub[c].values for c in sub.columns])
            tests[regime] = {"friedman_stat": float(stat), "friedman_p": float(pval),
                             "n": int(sub.shape[0])}
            # Pairwise Wilcoxon signed-rank with Holm correction
            pairs = [("tv-gVAR", "stationary-gVAR"),
                     ("tv-gVAR", "Ledoit-Wolf"),
                     ("stationary-gVAR", "Ledoit-Wolf")]
            raw_p = {}
            for a, b in pairs:
                if a in sub.columns and b in sub.columns:
                    try:
                        _, p_w = stats.wilcoxon(sub[a], sub[b])
                    except ValueError:
                        p_w = 1.0
                    raw_p[f"{a} vs {b}"] = float(p_w)
            # Holm
            ordered = sorted(raw_p.items(), key=lambda kv: kv[1])
            m = len(ordered)
            adj = {}
            running_max = 0.0
            for i, (k, pv) in enumerate(ordered):
                a = min(1.0, (m - i) * pv)
                running_max = max(running_max, a)
                adj[k] = running_max
            tests[regime]["wilcoxon_raw"] = raw_p
            tests[regime]["wilcoxon_holm"] = adj
    return summary, tests


def study_b_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    group_cols = ["method", "profile_type"] if "profile_type" in df.columns else ["method"]
    summary = df.groupby(group_cols).agg(
        spearman_mean=("spearman_rho", "mean"),
        spearman_sd=("spearman_rho", "std"),
        kendall_mean=("kendall_tau", "mean"),
        kendall_sd=("kendall_tau", "std"),
        top1_rate=("top1_correct", "mean"),
        top3_overlap=("top3_overlap", "mean"),
        n=("spearman_rho", "size"),
    ).reset_index()
    tests = {}
    # Per profile_type comparisons (tv vs stat)
    types_in_df = sorted(df["profile_type"].unique()) if "profile_type" in df.columns else ["all"]
    for pt in types_in_df:
        sub = df[df["profile_type"] == pt] if pt != "all" else df
        piv = sub.pivot_table(index="profile", columns="method", values="spearman_rho").dropna()
        if piv.shape[1] >= 2 and piv.shape[0] >= 3:
            try:
                _, p_b = stats.wilcoxon(piv["tv-gVAR"], piv["stationary-gVAR"])
            except ValueError:
                p_b = 1.0
            tests[f"spearman_tv_vs_stationary_{pt}"] = {
                "wilcoxon_p": float(p_b),
                "n": int(piv.shape[0]),
                "median_diff": float(np.median(piv["tv-gVAR"] - piv["stationary-gVAR"])),
            }
    return summary, tests


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

METHOD_COLOR = {
    "tv-gVAR":         "#1f3a68",
    "stationary-gVAR": "#3b6ea8",
    "Ledoit-Wolf":     "#e89c6a",
}

METHOD_LABEL = {
    "tv-gVAR": "time-varying gVAR",
    "stationary-gVAR": "stationary gVAR",
    "Ledoit-Wolf": "Ledoit-Wolf",
}


def _significance_stars(p_value: float) -> str:
    if p_value is None or np.isnan(p_value): return ""
    if p_value < 0.001: return "***"
    if p_value < 0.01:  return "**"
    if p_value < 0.05:  return "*"
    return "n.s."


def _add_bracket(ax, x1: float, x2: float, y_top: float, p_value: float,
                 height: float = 0.025, fontsize: float = 8.5) -> None:
    """Draw a horizontal bracket between x1 and x2 at height y_top with
    a centred significance label."""
    txt = _significance_stars(p_value)
    if not txt:
        return
    ax.plot([x1, x1, x2, x2],
            [y_top, y_top + height, y_top + height, y_top],
            color="black", linewidth=0.9)
    ax.text((x1 + x2) / 2.0, y_top + height + 0.005, txt,
            ha="center", va="bottom", fontsize=fontsize)

def render_figure(df_a: pd.DataFrame, df_b: pd.DataFrame,
                  tests_a: dict, tests_b: dict) -> Path:
    fig = plt.figure(figsize=(15.5, 10.2), dpi=150)
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        left=0.055, right=0.985, top=0.93, bottom=0.075,
        hspace=0.36, wspace=0.27,
    )

    # Panel A: F1 by method × regime (violin + jittered density-scatter)
    axA = fig.add_subplot(gs[0, 0])
    regimes = ["stationary", "smooth", "abrupt", "periodic"]
    methods = ["tv-gVAR", "stationary-gVAR", "Ledoit-Wolf"]
    positions = np.arange(len(regimes)) * (len(methods) + 1)
    width = 0.8
    rng = np.random.default_rng(101)
    for i, m in enumerate(methods):
        for r_idx, r in enumerate(regimes):
            vals = df_a[(df_a["regime"] == r) & (df_a["method"] == m)]["f1"].dropna().values
            if len(vals) == 0: continue
            xpos = positions[r_idx] + i
            parts = axA.violinplot([vals], positions=[xpos], widths=width,
                                   showmeans=False, showmedians=False, showextrema=False)
            for body in parts["bodies"]:
                body.set_facecolor(METHOD_COLOR[m]); body.set_alpha(0.55)
                body.set_edgecolor(METHOD_COLOR[m]); body.set_linewidth(0.6)
            # Density-aware jittered points
            xs = xpos + rng.uniform(-0.30, 0.30, size=len(vals))
            axA.scatter(xs, vals, color=METHOD_COLOR[m], s=11,
                        alpha=0.85, edgecolor="white", linewidth=0.3, zorder=3)
            # Median bar
            axA.hlines(np.median(vals), xpos - 0.30, xpos + 0.30,
                       color="black", linewidth=1.6, zorder=4)
    # Significance brackets: tv-gVAR vs stat-gVAR + tv-gVAR vs Ledoit-Wolf per regime
    for r_idx, r in enumerate(regimes):
        if r not in tests_a or "wilcoxon_holm" not in tests_a[r]:
            continue
        regime_max = df_a[df_a["regime"] == r]["f1"].dropna().max()
        if np.isnan(regime_max):
            continue
        y_base = regime_max + 0.06
        # bracket 1: tv-gVAR (i=0) vs stationary-gVAR (i=1)
        p_holm_ts = tests_a[r]["wilcoxon_holm"].get("tv-gVAR vs stationary-gVAR", 1.0)
        _add_bracket(axA, positions[r_idx] + 0, positions[r_idx] + 1, y_base, p_holm_ts)
        # bracket 2: tv-gVAR (i=0) vs Ledoit-Wolf (i=2)
        p_holm_tl = tests_a[r]["wilcoxon_holm"].get("tv-gVAR vs Ledoit-Wolf", 1.0)
        _add_bracket(axA, positions[r_idx] + 0, positions[r_idx] + 2, y_base + 0.09, p_holm_tl)
    axA.set_xticks(positions + 1)
    axA.set_xticklabels(regimes, fontsize=9)
    axA.set_ylabel("Edge-recovery F1", fontsize=9.5)
    axA.set_title("A.  Edge recovery (F1) by regime", fontsize=10.5, fontweight="bold", loc="left")
    axA.grid(axis="y", alpha=0.3)
    axA.spines["top"].set_visible(False); axA.spines["right"].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=METHOD_COLOR[m], alpha=0.75) for m in methods]
    axA.legend(handles, [METHOD_LABEL[m] for m in methods], loc="lower right",
               fontsize=8, framealpha=0.95)

    # Panel B: MSE by method × regime (stratified violin + jittered density-
    # scatter, log-y). The pooled view would hide stationary-gVAR's strong
    # performance on the stationary regime AND time-varying gVAR's strong
    # performance on the three non-stationary regimes; stratifying makes
    # both wins visible side by side.
    axB = fig.add_subplot(gs[0, 1])
    for i, m in enumerate(methods):
        for r_idx, r in enumerate(regimes):
            vals = df_a[(df_a["regime"] == r) & (df_a["method"] == m)]["mse"].dropna().values
            if len(vals) == 0: continue
            xpos = positions[r_idx] + i
            parts = axB.violinplot([vals], positions=[xpos], widths=width,
                                   showmeans=False, showmedians=False, showextrema=False)
            for body in parts["bodies"]:
                body.set_facecolor(METHOD_COLOR[m]); body.set_alpha(0.55)
                body.set_edgecolor(METHOD_COLOR[m]); body.set_linewidth(0.6)
            xs = xpos + rng.uniform(-0.30, 0.30, size=len(vals))
            axB.scatter(xs, vals, color=METHOD_COLOR[m], s=11,
                        alpha=0.85, edgecolor="white", linewidth=0.3, zorder=3)
            axB.hlines(np.median(vals), xpos - 0.30, xpos + 0.30,
                       color="black", linewidth=1.6, zorder=4)
    # Significance brackets per regime, mirroring Panel A
    for r_idx, r in enumerate(regimes):
        sub = df_a[df_a["regime"] == r].pivot_table(
            index=["T", "p", "noise", "seed"], columns="method", values="mse"
        ).dropna()
        if sub.shape[0] < 3:
            continue
        raw = {}
        for a, b in (("tv-gVAR", "stationary-gVAR"),
                     ("tv-gVAR", "Ledoit-Wolf"),
                     ("stationary-gVAR", "Ledoit-Wolf")):
            if a in sub.columns and b in sub.columns:
                try:
                    _, p_mse = stats.wilcoxon(sub[a], sub[b])
                except ValueError:
                    p_mse = 1.0
                raw[(a, b)] = float(p_mse)
        ordered = sorted(raw.items(), key=lambda kv: kv[1])
        holm = {}
        running = 0.0
        for j, (pair, pv) in enumerate(ordered):
            adj = min(1.0, (len(ordered) - j) * pv)
            running = max(running, adj)
            holm[pair] = running
        regime_max = df_a[df_a["regime"] == r]["mse"].dropna().max()
        y_base = regime_max * 1.20
        _add_bracket(axB, positions[r_idx] + 0, positions[r_idx] + 1,
                     y_base, holm.get(("tv-gVAR", "stationary-gVAR"), 1.0),
                     height=y_base * 0.10, fontsize=8)
        _add_bracket(axB, positions[r_idx] + 0, positions[r_idx] + 2,
                     y_base * 1.7, holm.get(("tv-gVAR", "Ledoit-Wolf"), 1.0),
                     height=y_base * 0.10, fontsize=8)
    axB.set_yscale("log")
    axB.set_xticks(positions + 1)
    axB.set_xticklabels(regimes, fontsize=9)
    axB.set_ylabel("Coefficient-recovery MSE (log)", fontsize=9.5)
    axB.set_title("B.  Coefficient recovery (MSE) by regime",
                  fontsize=10.5, fontweight="bold", loc="left")
    axB.grid(axis="y", which="both", alpha=0.3)
    axB.spines["top"].set_visible(False); axB.spines["right"].set_visible(False)

    # Panel C: AUC by method × regime (violin + jittered density-scatter)
    axC = fig.add_subplot(gs[0, 2])
    for i, m in enumerate(methods):
        for r_idx, r in enumerate(regimes):
            vals = df_a[(df_a["regime"] == r) & (df_a["method"] == m)]["auc"].dropna().values
            if len(vals) == 0: continue
            xpos = positions[r_idx] + i
            parts = axC.violinplot([vals], positions=[xpos], widths=width,
                                   showmeans=False, showmedians=False, showextrema=False)
            for body in parts["bodies"]:
                body.set_facecolor(METHOD_COLOR[m]); body.set_alpha(0.55)
                body.set_edgecolor(METHOD_COLOR[m]); body.set_linewidth(0.6)
            xs = xpos + rng.uniform(-0.30, 0.30, size=len(vals))
            axC.scatter(xs, vals, color=METHOD_COLOR[m], s=11,
                        alpha=0.85, edgecolor="white", linewidth=0.3, zorder=3)
            axC.hlines(np.median(vals), xpos - 0.30, xpos + 0.30,
                       color="black", linewidth=1.6, zorder=4)
    # AUC pairwise brackets mirror Panel A but are computed on AUC rather
    # than F1, using matched simulation cells and Holm correction per regime.
    for r_idx, r in enumerate(regimes):
        sub = df_a[df_a["regime"] == r].pivot_table(
            index=["T", "p", "noise", "seed"], columns="method", values="auc"
        ).dropna()
        if sub.shape[0] < 3:
            continue
        raw = {}
        for a, b in (("tv-gVAR", "stationary-gVAR"),
                     ("tv-gVAR", "Ledoit-Wolf"),
                     ("stationary-gVAR", "Ledoit-Wolf")):
            if a in sub.columns and b in sub.columns:
                try:
                    _, p_auc = stats.wilcoxon(sub[a], sub[b])
                except ValueError:
                    p_auc = 1.0
                raw[(a, b)] = float(p_auc)
        ordered = sorted(raw.items(), key=lambda kv: kv[1])
        holm = {}
        running = 0.0
        for j, (pair, pv) in enumerate(ordered):
            adj = min(1.0, (len(ordered) - j) * pv)
            running = max(running, adj)
            holm[pair] = running
        regime_max = df_a[df_a["regime"] == r]["auc"].dropna().max()
        y_base = min(0.96, regime_max + 0.025)
        _add_bracket(axC, positions[r_idx] + 0, positions[r_idx] + 1,
                     y_base, holm.get(("tv-gVAR", "stationary-gVAR"), 1.0),
                     height=0.012, fontsize=8)
        _add_bracket(axC, positions[r_idx] + 0, positions[r_idx] + 2,
                     min(0.98, y_base + 0.045), holm.get(("tv-gVAR", "Ledoit-Wolf"), 1.0),
                     height=0.012, fontsize=8)
    axC.axhline(0.5, color="#888", linestyle=":", linewidth=0.9)
    axC.text(positions[-1] + 1, 0.51, "chance (AUC = 0.5)", fontsize=7,
             ha="right", va="bottom", color="#666")
    axC.set_xticks(positions + 1)
    axC.set_xticklabels(regimes, fontsize=9)
    axC.set_ylabel("Edge-ranking AUC", fontsize=9.5)
    axC.set_title("C.  Edge-ranking AUC by regime", fontsize=10.5, fontweight="bold", loc="left")
    axC.grid(axis="y", alpha=0.3)
    axC.spines["top"].set_visible(False); axC.spines["right"].set_visible(False)

    # Panel D: F1 vs sample-size by method (line plot averaged across regimes)
    axD = fig.add_subplot(gs[1, 0])
    agg = df_a.groupby(["method", "T"])["f1"].agg(["mean", "std"]).reset_index()
    for m in methods:
        sub = agg[agg["method"] == m]
        axD.errorbar(sub["T"], sub["mean"], yerr=sub["std"],
                     color=METHOD_COLOR[m], marker="o", linewidth=1.8,
                     capsize=3, alpha=0.9, label=m)
    axD.set_xlabel("Sample size T", fontsize=9.5)
    axD.set_ylabel("Edge-recovery F1 (mean ± SD)", fontsize=9.5)
    axD.set_title("D.  Sample-size scaling", fontsize=10.5, fontweight="bold", loc="left")
    axD.grid(alpha=0.3)
    axD.spines["top"].set_visible(False); axD.spines["right"].set_visible(False)
    handles, labels = axD.get_legend_handles_labels()
    axD.legend(handles, [METHOD_LABEL.get(label, label) for label in labels],
               loc="lower right", fontsize=8)

    # Panel E: Heatmap of Holm-corrected p-values per regime (method pairs vs regimes)
    axE = fig.add_subplot(gs[1, 1])
    pairs = [
        "stationary gVAR\nvs Ledoit-Wolf",
        "time-varying gVAR\nvs Ledoit-Wolf",
        "time-varying gVAR\nvs stationary gVAR",
    ]
    pair_keys = ["stationary-gVAR vs Ledoit-Wolf", "tv-gVAR vs Ledoit-Wolf",
                 "tv-gVAR vs stationary-gVAR"]
    mat = np.full((len(pairs), len(regimes)), np.nan)
    for j, r in enumerate(regimes):
        if r in tests_a and "wilcoxon_holm" in tests_a[r]:
            for i, k in enumerate(pair_keys):
                v = tests_a[r]["wilcoxon_holm"].get(k)
                if v is not None and v > 0:
                    mat[i, j] = -np.log10(v)
    cap = 12.0
    display_mat = np.minimum(mat, cap)
    im = axE.imshow(display_mat, cmap="YlGnBu", aspect="auto", vmin=0, vmax=cap)
    axE.set_xticks(range(len(regimes)))
    axE.set_xticklabels(regimes, fontsize=8, rotation=20)
    axE.set_yticks(range(len(pairs)))
    axE.set_yticklabels(pairs, fontsize=8)
    axE.set_title("E.  Pairwise significance (−log₁₀ Holm-p)", fontsize=10.5,
                  fontweight="bold", loc="left")
    for i in range(len(pairs)):
        for j in range(len(regimes)):
            if not np.isnan(mat[i, j]):
                v = mat[i, j]
                txt_col = "white" if v > 7 else "#222"
                txt = f"{v:.1f}" if v < cap else f">={int(cap)}"
                axE.text(j, i, txt, ha="center", va="center",
                         fontsize=7.5, color=txt_col, fontweight="bold")
    cbar = fig.colorbar(im, ax=axE, fraction=0.03, pad=0.02)
    cbar.set_label(f"−log₁₀ Holm-p (capped at {int(cap)})", fontsize=7.5)

    # Panel F: Spearman ρ for impact ranking (Study B) split by profile type.
    # Each profile type (static vs dynamic) → two violins (tv-gVAR vs stat),
    # so the reader sees that tv-gVAR's advantage materialises only when the
    # ground-truth impact is time-varying.
    axF = fig.add_subplot(gs[1, 2])
    methods_b = ["tv-gVAR", "stationary-gVAR"]
    profile_types = sorted(df_b["profile_type"].unique()) if "profile_type" in df_b.columns else ["all"]
    rng = np.random.default_rng(13)
    positions = []
    labels = []
    base = 0
    for pt in profile_types:
        for j, m in enumerate(methods_b):
            xpos = base + j
            vals = (df_b[(df_b["profile_type"] == pt) & (df_b["method"] == m)]["spearman_rho"]
                    .dropna().values) if pt != "all" else df_b[df_b["method"] == m]["spearman_rho"].dropna().values
            if len(vals) == 0: continue
            parts = axF.violinplot([vals], positions=[xpos], widths=0.65,
                                    showmeans=False, showmedians=False, showextrema=False)
            for body in parts["bodies"]:
                body.set_facecolor(METHOD_COLOR.get(m, "#999")); body.set_alpha(0.45)
                body.set_edgecolor("#222"); body.set_linewidth(0.6)
            axF.boxplot([vals], positions=[xpos], widths=0.18, patch_artist=True, showfliers=False,
                        boxprops={"facecolor": "white", "alpha": 0.95,
                                   "edgecolor": "#222", "linewidth": 0.9},
                        medianprops={"color": METHOD_COLOR.get(m, "#999"), "linewidth": 1.8},
                        whiskerprops={"color": "#222"}, capprops={"color": "#222"})
            xs = xpos + rng.uniform(-0.22, 0.22, size=len(vals))
            axF.scatter(xs, vals, color=METHOD_COLOR.get(m, "#999"), s=18,
                        alpha=0.85, edgecolor="white", linewidth=0.4, zorder=3)
            positions.append(xpos)
            method_display = "time-varying\ngVAR" if m == "tv-gVAR" else "stationary\ngVAR"
            labels.append(f"{method_display}\n{pt}")
        # Add a significance bracket between the two methods within this profile_type
        sub_piv = (df_b[df_b["profile_type"] == pt]
                   .pivot_table(index="profile", columns="method", values="spearman_rho")
                   .dropna()) if pt != "all" else df_b.pivot_table(index="profile", columns="method",
                                                                    values="spearman_rho").dropna()
        if sub_piv.shape[0] >= 3 and "tv-gVAR" in sub_piv.columns and "stationary-gVAR" in sub_piv.columns:
            try:
                _, pval = stats.wilcoxon(sub_piv["tv-gVAR"], sub_piv["stationary-gVAR"])
            except ValueError:
                pval = 1.0
            y_top = max(sub_piv["tv-gVAR"].max(), sub_piv["stationary-gVAR"].max()) + 0.05
            _add_bracket(axF, base + 0, base + 1, y_top, float(pval))
        base += 3   # leave a small gap between groups
    axF.set_xticks(positions); axF.set_xticklabels(labels, fontsize=7.4)
    axF.set_ylabel("Spearman ρ vs ground-truth", fontsize=9.5)
    axF.set_title("F.  Impact-ranking recovery (Study B, by profile type)",
                  fontsize=10.5, fontweight="bold", loc="left")
    axF.set_ylim(-0.20, 1.18)
    axF.axhline(0, color="#888", linestyle=":", linewidth=0.8)
    axF.grid(axis="y", alpha=0.3)
    axF.spines["top"].set_visible(False); axF.spines["right"].set_visible(False)

    # No suptitle — the Word figure caption above the image is the title.
    out_path = OUT_DIR / "figure_hua_evaluation.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(small: bool = False) -> None:
    print(f"Output directory: {OUT_DIR}")
    if small:
        df_a = run_study_a(
            regimes=("stationary", "smooth"),
            T_grid=(80, 160), p_grid=(6,), noise_grid=(0.30,), n_seeds=2,
        )
        df_b = run_study_b(T=160, p=6, n_profiles=8)
    else:
        df_a = run_study_a()
        df_b = run_study_b()

    summary_a, tests_a = study_a_summary(df_a)
    summary_b, tests_b = study_b_summary(df_b)

    df_a.to_csv(OUT_DIR / "hua_study_a_per_run.csv", index=False)
    summary_a.to_csv(OUT_DIR / "hua_study_a_summary.csv", index=False)
    df_b.to_csv(OUT_DIR / "hua_study_b_per_run.csv", index=False)
    summary_b.to_csv(OUT_DIR / "hua_study_b_summary.csv", index=False)

    with open(OUT_DIR / "statistical_tests.json", "w") as f:
        json.dump({"study_a": tests_a, "study_b": tests_b}, f, indent=2)

    fig_path = render_figure(df_a, df_b, tests_a, tests_b)

    manifest = {
        "study_a_n_runs": int(len(df_a)),
        "study_b_n_runs": int(len(df_b)),
        "study_a_n_per_method": int(len(df_a) // df_a["method"].nunique()),
        "study_a_methods": sorted(df_a["method"].unique().tolist()),
        "study_a_regimes": sorted(df_a["regime"].unique().tolist()),
        "figure_path": str(fig_path.relative_to(REPO_ROOT)),
    }
    with open(OUT_DIR / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nStudy A summary (mean F1 by method × regime):")
    print(summary_a[["method", "regime", "f1", "f1_sd", "mse", "n"]].to_string(index=False))
    print(f"\nStudy B summary:")
    print(summary_b.to_string(index=False))
    print(f"\nFigure: {fig_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", action="store_true",
                        help="Run a tiny grid (for quick checks)")
    args = parser.parse_args()
    main(small=args.small)
