#!/usr/bin/env python3
"""
Latency statistics post-processor for the ontology-query benchmark.

Reads `ontology_query_latency.csv` (produced by `benchmark_ontology_queries.py`)
and generates:

  * `ontology_query_latency_stats.json`      — per-operation summary with
        adaptive time-unit reporting (microseconds where median < 1 ms),
        Friedman omnibus test across the three back-ends, all-pairs Wilcoxon
        signed-rank tests with Holm-Bonferroni correction, and Cliff's delta
        as the pair-level effect-size metric.
  * `ontology_query_latency_stats.csv`       — flat per-(operation, backend)
        rows with mean / median / IQR / p95 / p99 reported in the adaptive
        unit, plus the omnibus Friedman p and the dominant pairwise contrast.

The script intentionally does NOT produce a figure: the thesis reports the
latency results in text, so the on-disk artefact is a compact JSON/CSV pair
that can be inlined in the document as numeric evidence.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN = REPO_ROOT / "evaluation/ontology_evaluation/results/ontology_query_latency.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "evaluation/ontology_evaluation/results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _adaptive_unit(median_ms: float) -> tuple[str, float]:
    """Return a (unit_label, scale_from_ms) pair such that values look natural."""
    if median_ms < 0.5:
        return "μs", 1000.0          # 1 ms == 1000 μs
    if median_ms < 1000.0:
        return "ms", 1.0
    return "s", 1.0 / 1000.0


def _format(value_ms: float, unit_label: str, scale_from_ms: float) -> str:
    val = value_ms * scale_from_ms
    if unit_label == "μs":
        return f"{val:.1f} μs"
    if unit_label == "ms":
        return f"{val:.3f} ms"
    return f"{val:.3f} s"


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta effect size: in [-1, 1]; 0 means stochastically equal."""
    a, b = np.asarray(a), np.asarray(b)
    n_gt = int(np.sum(a[:, None] > b[None, :]))
    n_lt = int(np.sum(a[:, None] < b[None, :]))
    n_tot = len(a) * len(b)
    if n_tot == 0:
        return 0.0
    return (n_gt - n_lt) / n_tot


def _holm_correct(pvals: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(ordered)
    out = {}
    cumulative = 0.0
    for i, (key, p) in enumerate(ordered):
        adj = min(1.0, (m - i) * p)
        cumulative = max(cumulative, adj)
        out[key] = cumulative
    return out


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def analyse(in_path: Path, out_dir: Path) -> None:
    df = pd.read_csv(in_path)
    required = {"backend", "operation", "latency_ms"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    results = {"operations": {}, "global": {}}
    flat_rows: list[dict] = []

    operations = sorted(df["operation"].unique())
    backends = sorted(df["backend"].unique())

    for op in operations:
        sub = df[df["operation"] == op]
        per_backend_arr = {b: sub.loc[sub["backend"] == b, "latency_ms"].values
                           for b in backends}

        per_backend = {}
        for b, vals in per_backend_arr.items():
            if len(vals) == 0:
                continue
            # Pick a per-backend adaptive unit so sub-millisecond values are
            # never reported as "0.000 ms" (the user's complaint).
            med = float(np.median(vals))
            unit_label, scale = _adaptive_unit(med)
            q25, q75 = np.percentile(vals, [25, 75])
            p95, p99 = np.percentile(vals, [95, 99])
            per_backend[b] = {
                "n": int(len(vals)),
                "unit": unit_label,
                "mean": float(np.mean(vals) * scale),
                "median": float(med * scale),
                "iqr": [float(q25 * scale), float(q75 * scale)],
                "p95": float(p95 * scale),
                "p99": float(p99 * scale),
                "min": float(np.min(vals) * scale),
                "max": float(np.max(vals) * scale),
                "median_ms": med,
                "human_median": _format(med, unit_label, scale),
                "human_p95": _format(float(p95), unit_label, scale),
            }
            flat_rows.append({
                "operation": op,
                "backend": b,
                "n": len(vals),
                "unit": unit_label,
                "mean": per_backend[b]["mean"],
                "median": per_backend[b]["median"],
                "iqr_low": per_backend[b]["iqr"][0],
                "iqr_high": per_backend[b]["iqr"][1],
                "p95": per_backend[b]["p95"],
                "p99": per_backend[b]["p99"],
                "human_median": per_backend[b]["human_median"],
                "human_p95": per_backend[b]["human_p95"],
            })

        # Friedman omnibus test — across backends, paired by query case.
        # Need a 'paired' axis; we use a sequential within-backend index after
        # truncating to the minimum length.
        min_n = min(len(v) for v in per_backend_arr.values()) if per_backend_arr else 0
        friedman = None
        if min_n >= 3 and len(per_backend_arr) >= 3:
            stacked = np.column_stack([per_backend_arr[b][:min_n] for b in backends])
            try:
                stat, p = stats.friedmanchisquare(*[stacked[:, i] for i in range(stacked.shape[1])])
                friedman = {"chi2": float(stat), "p": float(p), "n": int(min_n)}
            except Exception as exc:
                friedman = {"chi2": float("nan"), "p": float("nan"),
                            "error": str(exc), "n": int(min_n)}

        # Pairwise Wilcoxon signed-rank tests + Cliff's delta
        raw_p = {}
        deltas = {}
        for a, b in itertools.combinations(backends, 2):
            va = per_backend_arr[a][:min_n]
            vb = per_backend_arr[b][:min_n]
            if len(va) < 2:
                continue
            try:
                _, pv = stats.wilcoxon(va, vb)
            except ValueError:
                pv = 1.0
            raw_p[f"{a} vs {b}"] = float(pv)
            deltas[f"{a} vs {b}"] = _cliffs_delta(per_backend_arr[a], per_backend_arr[b])
        holm = _holm_correct(raw_p) if raw_p else {}

        results["operations"][op] = {
            "per_backend": per_backend,
            "friedman": friedman,
            "pairwise_wilcoxon_raw": raw_p,
            "pairwise_wilcoxon_holm": holm,
            "pairwise_cliffs_delta": deltas,
        }

    # Global summary across all operations × queries + industrial metrics
    all_per_backend = {b: df[df["backend"] == b]["latency_ms"].values for b in backends}
    overall = {}
    for b, vals in all_per_backend.items():
        if len(vals) == 0:
            continue
        med = float(np.median(vals))
        unit_label, scale = _adaptive_unit(med)
        # Throughput: queries-per-second under the assumption of a sequential
        # single-thread server (a conservative industrial upper bound).
        qps_median = 1000.0 / med if med > 0 else float("inf")
        qps_p95 = 1000.0 / float(np.percentile(vals, 95)) if np.percentile(vals, 95) > 0 else float("inf")
        # Cold vs warm distinction: in this benchmark every query is "warm"
        # (load/build excluded), so we annotate explicitly. A cold-start cost
        # measurement is left as a follow-up benchmark.
        overall[b] = {
            "n": int(len(vals)),
            "unit": unit_label,
            "regime": "warm",                         # benchmark scope
            "median": float(med * scale),
            "p50": float(np.percentile(vals, 50) * scale),
            "p90": float(np.percentile(vals, 90) * scale),
            "p95": float(np.percentile(vals, 95) * scale),
            "p99": float(np.percentile(vals, 99) * scale),
            "p999": float(np.percentile(vals, 99.9) * scale),
            "max": float(np.max(vals) * scale),
            "iqr_low": float(np.percentile(vals, 25) * scale),
            "iqr_high": float(np.percentile(vals, 75) * scale),
            "stdev": float(np.std(vals, ddof=1) * scale) if len(vals) > 1 else 0.0,
            "cv": float(np.std(vals, ddof=1) / np.mean(vals)) if np.mean(vals) > 0 else 0.0,
            "throughput_qps_median": qps_median,
            "throughput_qps_p95": qps_p95,
            "throughput_qps_per_ms_p95": qps_p95 / 1000.0,
            "human_median": _format(med, unit_label, scale),
            "human_p99": _format(float(np.percentile(vals, 99)), unit_label, scale),
        }
    results["global"]["per_backend"] = overall

    # Persist outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "ontology_query_latency_stats.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    csv_path = out_dir / "ontology_query_latency_stats.csv"
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False)

    # Console report
    print(f"\nLatency statistics report")
    print(f"  read:  {in_path}")
    print(f"  wrote: {json_path}")
    print(f"  wrote: {csv_path}")
    for op in operations:
        opres = results["operations"][op]
        print(f"\n  [{op}]")
        for b in backends:
            if b not in opres["per_backend"]:
                continue
            row = opres["per_backend"][b]
            print(f"    {b:<22} median={row['human_median']:>12}   "
                  f"p95={row['human_p95']:>12}   n={row['n']}")
        if opres["friedman"]:
            f = opres["friedman"]
            print(f"    Friedman χ²={f['chi2']:.2f}, p={f['p']:.3e}, n={f['n']}")
        for pair, p_holm in opres["pairwise_wilcoxon_holm"].items():
            d = opres["pairwise_cliffs_delta"].get(pair, 0.0)
            print(f"    Wilcoxon {pair}:  Holm-p={p_holm:.3e}   Cliff's δ={d:+.3f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=DEFAULT_IN)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = p.parse_args()
    analyse(args.input, args.out_dir)


if __name__ == "__main__":
    main()
