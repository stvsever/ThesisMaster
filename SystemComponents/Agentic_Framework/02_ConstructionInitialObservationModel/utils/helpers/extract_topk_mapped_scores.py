r"""
Utilities to textualize LLM-based field mappings between:
- predictors ↔ criteria (aka domain/barrier criterion clusters)
- predictors ↔ context
- predictors ↔ profile
- predictors ↔ barriers (HAPA)
- profiles ↔ barriers (HAPA)
- contexts ↔ barriers (HAPA)
- coping ↔ barriers (HAPA)

# NOTE: In this project, a "predictor" refers to a potential solution/treatment
#       to address a (non-)clinical mental health issue.
# NOTE: A "path" is the semantic context where the solution is hierarchically framed
#       inside a biopsychosocial framework.

IMPORTANT NOTE (missing score semantics):
- Empty / missing scores (parsed as None/NaN) mean "non-relevance", i.e. numerical 0.
- For aggregation we print BOTH:
    - avg_uncorrected: mean over PRESENT scores (NaNs ignored; pandas default)
    - avg_corrected:   mean where missing scores are treated as 0 (NaNs -> 0)
"""

from __future__ import annotations

from typing import Optional, Sequence, List, Union, Tuple

import pandas as pd

# -------------------------
# Configuration defaults
# -------------------------
# PREDICTOR-mapped
DEFAULT_TOP_K_CRITERION = 200
DEFAULT_TOP_K_CONTEXT = 200
DEFAULT_TOP_K_PROFILE = 200

# BARRIER-mapped
DEFAULT_TOP_K_PROFILE_BARRIER = 200
DEFAULT_TOP_K_CONTEXT_BARRIER = 200
DEFAULT_TOP_K_COPING_BARRIER = 200

# PREDICTOR-BARRIER-mapped
DEFAULT_TOP_K_PREDICTOR_BARRIER = 200


# -------------------------
# Helpers
# -------------------------
def _ensure_columns(df: pd.DataFrame, required: Sequence[str], csv_path: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV '{csv_path}' is missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )


def _coerce_score(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _apply_id_filter(
    df: pd.DataFrame,
    id_col: str,
    ids: Optional[Sequence[Union[str, int]]],
) -> pd.DataFrame:
    """
    Filter df by df[id_col] ∈ ids.
    - ids is None or []  -> no filtering (keeps current behaviour)
    """
    if ids is None or len(ids) == 0:
        return df
    wanted = set(str(x) for x in ids)
    return df[df[id_col].astype(str).isin(wanted)]


def _filter_entity(
    df: pd.DataFrame,
    *,
    id_col: str,
    full_path_col: str,
    full_paths: Optional[Sequence[str]] = None,
    ids: Optional[Sequence[Union[str, int]]] = None,
) -> pd.DataFrame:
    if full_paths:
        df = df[df[full_path_col].isin(full_paths)]
    df = _apply_id_filter(df, id_col=id_col, ids=ids)
    return df


def _fmt_score(x) -> str:
    """
    Formatting rule:
    - NaN/None means "non-relevance" => display as 0.0000
    """
    try:
        if pd.isna(x):
            return f"{0.0:.4f}"
        v = float(x)
        return f"{v:.4f}"
    except Exception:
        return str(x)


def _print_lines(header: str, lines: Sequence[str]) -> None:
    print("\n" + "=" * 80)
    print(header)
    print("=" * 80)
    for line in lines:
        print(line)


def _mean_uncorrected(series: pd.Series) -> float:
    # pandas mean ignores NaN by default
    return float(series.mean()) if len(series) else float("nan")


def _mean_corrected(series: pd.Series) -> float:
    # corrected: missing scores are 0 (non-relevance)
    return float(series.fillna(0).mean()) if len(series) else float("nan")


def _missing_count(series: pd.Series) -> int:
    return int(series.isna().sum()) if len(series) else 0


def _aggregate_overall(
    df: pd.DataFrame,
    *,
    score_col: str,
    left_id_col: str,
    right_id_col: str,
) -> dict:
    if len(df) == 0:
        return {
            "avg_uncorrected": float("nan"),
            "avg_corrected": float("nan"),
            "rows": 0,
            "missing_scores": 0,
            "left_unique": 0,
            "right_unique": 0,
        }

    s = df[score_col]
    return {
        "avg_uncorrected": _mean_uncorrected(s),
        "avg_corrected": _mean_corrected(s),
        "rows": int(len(df)),
        "missing_scores": _missing_count(s),
        "left_unique": int(df[left_id_col].astype(str).nunique()),
        "right_unique": int(df[right_id_col].astype(str).nunique()),
    }


def _first_nonnull(series: pd.Series) -> Optional[str]:
    """Return the first non-null, non-empty string value (best-effort representative label/path)."""
    for v in series:
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s:
            return s
    return None


# =============================================================================
# 1) Predictor ↔ Criterion-cluster mapping
# =============================================================================
def extract_topk_predictor_criterion_map(
    csv_path: str,
    top_k: int = DEFAULT_TOP_K_CRITERION,
    predictor_full_paths: Optional[Sequence[str]] = None,
    predictor_ids: Optional[Sequence[Union[str, int]]] = None,
    # explicit ID lists for both variables
    predictor_ids_filter: Optional[Sequence[Union[str, int]]] = None,
    criterion_cluster_ids_filter: Optional[Sequence[Union[str, int]]] = None,
    # aggregation toggles
    # Default requested for PC: aggregate_criterions=True, aggregate_predictors=False
    aggregate_criterions: bool = True,
    aggregate_predictors: bool = False,
    # NEW (ONLY new argument):
    # full universe size for inference of missing predictor×cluster pairs when the CSV is sparse
    full_matrix_shape: Tuple[int, int] = (143, 3110),
) -> List[str]:
    """
    Input columns (actual):
      cluster_id,domain,domain_why,predictor_id,score,predictor_full_path,cluster_item_count

    Interpretation:
      - predictor_id: predictor identifier
      - domain: predictor name (domain)  [NOTE: not guaranteed stable across rows]
      - cluster_id: criterion-cluster identifier (criterion side)

    Aggregation semantics:
      - aggregate_criterions=True  => per predictor_id aggregate across cluster_id
      - aggregate_predictors=True  => per cluster_id aggregate across predictor_id

    Missing score semantics:
      - avg_uncorrected: mean over PRESENT scores (NaNs ignored; pandas default)
      - avg_corrected:
          mean where missing scores are treated as 0, INCLUDING:
            (a) explicit NaN scores in the CSV, and
            (b) *implicit missing pairs* not present in a sparse CSV, inferred via:
                (#predictors_considered × #clusters_considered) - #pairs_present
    """

    def _safe_int(x, name: str) -> int:
        try:
            v = int(x)
        except Exception as e:
            raise ValueError(f"{name} must be an int; got {x!r}") from e
        if v < 0:
            raise ValueError(f"{name} must be >= 0; got {v}")
        return v

    def _unique_count_from_ids(ids: Optional[Sequence[Union[str, int]]]) -> int:
        if ids is None or len(ids) == 0:
            return 0
        return len(set(str(x) for x in ids))

    def _unique_count_from_paths(paths: Optional[Sequence[str]]) -> int:
        if not paths:
            return 0
        return len(set(str(p) for p in paths))

    def _infer_considered_counts(df_pairs: pd.DataFrame) -> tuple[int, int]:
        """
        Determine the denominator counts (predictors_considered, clusters_considered)
        given optional filters. If no explicit filters for a side are provided, fall back
        to the full universe size (full_matrix_shape).
        """
        full_pred, full_clu = full_matrix_shape
        full_pred = _safe_int(full_pred, "full_matrix_shape[0]")
        full_clu = _safe_int(full_clu, "full_matrix_shape[1]")

        # Predictor side: prefer explicit ID/path filters if provided.
        n_pred = (
            _unique_count_from_ids(predictor_ids_filter)
            or _unique_count_from_ids(predictor_ids)
            or _unique_count_from_paths(predictor_full_paths)
            or full_pred
        )

        # Cluster side: only explicit ID list exists at this level.
        n_clu = _unique_count_from_ids(criterion_cluster_ids_filter) or full_clu

        # Defensive: if filters yield 0, denominator becomes 0 => return 0 to avoid division.
        return int(n_pred), int(n_clu)

    def _infer_predictor_name_from_path(v) -> Optional[str]:
        if pd.isna(v):
            return None
        s = str(v).strip()
        if not s:
            return None
        parts = [p.strip() for p in s.split(">")]
        if not parts:
            return None
        last = parts[-1].strip()
        return last if last else None

    required = [
        "cluster_id",
        "domain",
        "domain_why",
        "predictor_id",
        "score",
        "predictor_full_path",
        "cluster_item_count",
    ]

    df = pd.read_csv(csv_path)
    _ensure_columns(df, required, csv_path)

    # Base predictor filters
    df = _filter_entity(
        df,
        id_col="predictor_id",
        full_path_col="predictor_full_path",
        full_paths=predictor_full_paths,
        ids=predictor_ids,
    )

    # Explicit ID list filters (both sides)
    df = _apply_id_filter(df, id_col="predictor_id", ids=predictor_ids_filter)
    df = _apply_id_filter(df, id_col="cluster_id", ids=criterion_cluster_ids_filter)

    df["score"] = _coerce_score(df["score"])

    # Collapse to unique predictor×cluster pairs (robust against accidental duplicates)
    # Pair-score: mean(score) over duplicate rows (NaN ignored by default for mean).
    df_pairs = (
        df.groupby(["predictor_id", "cluster_id"], dropna=False, as_index=False)
        .agg(
            score=("score", "mean"),
            predictor_domain=("predictor_full_path", lambda s: _first_nonnull(s.map(_infer_predictor_name_from_path))),
            predictor_path=("predictor_full_path", _first_nonnull),
        )
    )

    predictors_considered, clusters_considered = _infer_considered_counts(df_pairs)
    total_pairs_overall = predictors_considered * clusters_considered

    # Helper for inferred-missing accounting
    def _overall_stats() -> dict:
        if total_pairs_overall == 0:
            return {
                "avg_uncorrected": float("nan"),
                "avg_corrected": float("nan"),
                "pairs_present": int(len(df_pairs)),
                "pairs_total": 0,
                "missing_explicit": int(df_pairs["score"].isna().sum()) if len(df_pairs) else 0,
                "missing_inferred": 0,
                "missing_total": 0,
                "predictors_considered": predictors_considered,
                "clusters_considered": clusters_considered,
            }

        pairs_present = int(len(df_pairs))
        missing_explicit = int(df_pairs["score"].isna().sum()) if pairs_present else 0
        missing_inferred = max(0, int(total_pairs_overall - pairs_present))
        missing_total = missing_explicit + missing_inferred

        avg_uncorrected = float(df_pairs["score"].mean()) if pairs_present else float("nan")
        avg_corrected = float(df_pairs["score"].fillna(0).sum()) / float(total_pairs_overall)

        return {
            "avg_uncorrected": avg_uncorrected,
            "avg_corrected": avg_corrected,
            "pairs_present": pairs_present,
            "pairs_total": int(total_pairs_overall),
            "missing_explicit": missing_explicit,
            "missing_inferred": missing_inferred,
            "missing_total": missing_total,
            "predictors_considered": predictors_considered,
            "clusters_considered": clusters_considered,
        }

    if aggregate_criterions and aggregate_predictors:
        stats = _overall_stats()
        missing_pct = (100.0 * stats["missing_total"] / stats["pairs_total"]) if stats["pairs_total"] else 0.0
        return [
            " | ".join(
                [
                    "[01]",
                    f"avg_uncorrected={_fmt_score(stats['avg_uncorrected'])}",
                    f"avg_corrected={_fmt_score(stats['avg_corrected'])}",
                    f"pairs_present={stats['pairs_present']}",
                    f"pairs_total={stats['pairs_total']}",
                    f"missing_total={stats['missing_total']} ({missing_pct:.1f}%)",
                    f"missing_explicit={stats['missing_explicit']}",
                    f"missing_inferred={stats['missing_inferred']}",
                    f"predictors_considered={stats['predictors_considered']}",
                    f"criterion_clusters_considered={stats['clusters_considered']}",
                ]
            )
        ]

    if aggregate_criterions:
        # Denominator per predictor: clusters_considered
        denom = clusters_considered

        if denom == 0:
            return []

        df_agg = (
            df_pairs.groupby("predictor_id", dropna=False, as_index=False)
            .agg(
                sum_score=("score", lambda s: float(s.fillna(0).sum())),
                avg_uncorrected=("score", "mean"),  # NaNs ignored by pandas
                present_pairs=("cluster_id", "count"),
                missing_explicit=("score", lambda s: int(s.isna().sum())),
                predictor_domain=("predictor_domain", _first_nonnull),
                predictor_path=("predictor_path", _first_nonnull),
            )
        )

        # Inferred missing pairs per predictor (pairs absent from sparse CSV)
        df_agg["missing_inferred"] = (denom - df_agg["present_pairs"]).clip(lower=0).astype(int)
        df_agg["missing_scores"] = (df_agg["missing_explicit"] + df_agg["missing_inferred"]).astype(int)

        # Corrected average over FULL denominator (implicit missing treated as 0)
        df_agg["avg_corrected"] = df_agg["sum_score"] / float(denom)

        # If the user provided an explicit predictor list filter, ensure those IDs are represented
        # even when they have 0 present edges in the sparse CSV.
        if predictor_ids_filter is not None and len(predictor_ids_filter) > 0:
            wanted = [str(x) for x in predictor_ids_filter]
            have = set(df_agg["predictor_id"].astype(str).tolist())
            missing_ids = [pid for pid in wanted if pid not in have]
            if missing_ids:
                df_missing = pd.DataFrame(
                    {
                        "predictor_id": missing_ids,
                        "sum_score": 0.0,
                        "avg_uncorrected": float("nan"),
                        "present_pairs": 0,
                        "missing_explicit": 0,
                        "predictor_domain": None,
                        "predictor_path": None,
                        "missing_inferred": denom,
                        "missing_scores": denom,
                        "avg_corrected": 0.0,
                    }
                )
                df_agg = pd.concat([df_agg, df_missing], ignore_index=True)

        # Rank by corrected average
        df_agg = df_agg.sort_values("avg_corrected", ascending=False).head(int(top_k))

        lines: List[str] = []
        for i, row in enumerate(df_agg.itertuples(index=False), start=1):
            lines.append(
                " | ".join(
                    [
                        f"[{i:02d}]",
                        f"predictor(domain)='{row.predictor_domain}' (id={row.predictor_id})",
                        f"predictor_path='{row.predictor_path}'",
                        f"avg_uncorrected={_fmt_score(row.avg_uncorrected)}",
                        f"avg_corrected={_fmt_score(row.avg_corrected)}",
                        #f"present_pairs={int(row.present_pairs)}/{denom}",
                        #f"missing_scores={int(row.missing_scores)}",
                        #f"missing_explicit={int(row.missing_explicit)}",
                        #f"missing_inferred={int(row.missing_inferred)}",
                    ]
                )
            )
        return lines

    if aggregate_predictors:
        # Denominator per cluster: predictors_considered
        denom = predictors_considered

        if denom == 0:
            return []

        df_agg = (
            df_pairs.groupby("cluster_id", dropna=False, as_index=False)
            .agg(
                sum_score=("score", lambda s: float(s.fillna(0).sum())),
                avg_uncorrected=("score", "mean"),
                present_pairs=("predictor_id", "count"),
                missing_explicit=("score", lambda s: int(s.isna().sum())),
            )
        )

        df_agg["missing_inferred"] = (denom - df_agg["present_pairs"]).clip(lower=0).astype(int)
        df_agg["missing_scores"] = (df_agg["missing_explicit"] + df_agg["missing_inferred"]).astype(int)
        df_agg["avg_corrected"] = df_agg["sum_score"] / float(denom)

        # If the user provided an explicit cluster list filter, ensure those IDs are represented
        if criterion_cluster_ids_filter is not None and len(criterion_cluster_ids_filter) > 0:
            wanted = [str(x) for x in criterion_cluster_ids_filter]
            have = set(df_agg["cluster_id"].astype(str).tolist())
            missing_ids = [cid for cid in wanted if cid not in have]
            if missing_ids:
                df_missing = pd.DataFrame(
                    {
                        "cluster_id": missing_ids,
                        "sum_score": 0.0,
                        "avg_uncorrected": float("nan"),
                        "present_pairs": 0,
                        "missing_explicit": 0,
                        "missing_inferred": denom,
                        "missing_scores": denom,
                        "avg_corrected": 0.0,
                    }
                )
                df_agg = pd.concat([df_agg, df_missing], ignore_index=True)

        df_agg = df_agg.sort_values("avg_corrected", ascending=False).head(int(top_k))

        lines: List[str] = []
        for i, row in enumerate(df_agg.itertuples(index=False), start=1):
            lines.append(
                " | ".join(
                    [
                        f"[{i:02d}]",
                        f"cluster_id={row.cluster_id}",
                        f"avg_uncorrected={_fmt_score(row.avg_uncorrected)}",
                        f"avg_corrected={_fmt_score(row.avg_corrected)}",
                        #f"present_pairs={int(row.present_pairs)}/{denom}",
                        #f"missing_scores={int(row.missing_scores)}",
                        #f"missing_explicit={int(row.missing_explicit)}",
                        #f"missing_inferred={int(row.missing_inferred)}",
                    ]
                )
            )
        return lines

    # No aggregation (edge listing)
    # NOTE: display NaN as 0.0000 (non-relevance)
    df_pairs = df_pairs.assign(score_display=df_pairs["score"].fillna(0))
    df_pairs = df_pairs.sort_values("score_display", ascending=False).head(int(top_k))

    lines: List[str] = []
    for i, row in enumerate(df_pairs.itertuples(index=False), start=1):
        lines.append(
            " | ".join(
                [
                    f"[{i:02d}]",
                    f"predictor(domain)='{row.predictor_domain}' (id={row.predictor_id})",
                    f"predictor_path='{row.predictor_path}'",
                    f"cluster_id={row.cluster_id}",
                    f"score={_fmt_score(row.score_display)}",
                ]
            )
        )
    return lines


# =============================================================================
# 2) Predictor ↔ Context mapping
# =============================================================================
def extract_topk_predictor_context_map(
    csv_path: str,
    top_k: int = DEFAULT_TOP_K_CONTEXT,
    predictor_full_paths: Optional[Sequence[str]] = None,
    predictor_ids: Optional[Sequence[Union[str, int]]] = None,
    predictor_ids_filter: Optional[Sequence[Union[str, int]]] = None,
    context_ids_filter: Optional[Sequence[Union[str, int]]] = None,
    aggregate_contexts: bool = False,
    aggregate_predictors: bool = False,
) -> List[str]:
    """
    Input columns:
      predictor_id,predictor_name,predictor_full_path,context_id,context_name,context_full_path,score

    Missing score semantics:
      - avg_uncorrected: mean(score) ignoring missing (NaN)
      - avg_corrected:   mean(score) treating missing as 0 (NaN -> 0)
    """
    required = [
        "predictor_id",
        "predictor_name",
        "predictor_full_path",
        "context_id",
        "context_name",
        "context_full_path",
        "score",
    ]

    df = pd.read_csv(csv_path)
    _ensure_columns(df, required, csv_path)

    df = _filter_entity(
        df,
        id_col="predictor_id",
        full_path_col="predictor_full_path",
        full_paths=predictor_full_paths,
        ids=predictor_ids,
    )

    df = _apply_id_filter(df, id_col="predictor_id", ids=predictor_ids_filter)
    df = _apply_id_filter(df, id_col="context_id", ids=context_ids_filter)

    df["score"] = _coerce_score(df["score"])

    if aggregate_contexts and aggregate_predictors:
        stats = _aggregate_overall(df, score_col="score", left_id_col="predictor_id", right_id_col="context_id")
        missing_pct = (100.0 * stats["missing_scores"] / stats["rows"]) if stats["rows"] else 0.0
        return [
            " | ".join(
                [
                    "[01]",
                    f"avg_uncorrected={_fmt_score(stats['avg_uncorrected'])}",
                    #f"avg_corrected={_fmt_score(stats['avg_corrected'])}",
                    #f"rows_aggregated={stats['rows']}",
                    #f"missing_scores={stats['missing_scores']} ({missing_pct:.1f}%)",
                    f"predictors={stats['left_unique']}",
                    f"contexts={stats['right_unique']}",
                ]
            )
        ]

    if aggregate_contexts:
        df_agg = (
            df.groupby("predictor_id", dropna=False, as_index=False)
            .agg(
                avg_uncorrected=("score", "mean"),
                avg_corrected=("score", lambda s: s.fillna(0).mean()),
                missing_scores=("score", lambda s: int(s.isna().sum())),
                predictor_name=("predictor_name", _first_nonnull),
                predictor_path=("predictor_full_path", _first_nonnull),
            )
        )
        df_agg = df_agg.sort_values("avg_corrected", ascending=False).head(int(top_k))

        lines: List[str] = []
        for i, row in enumerate(df_agg.itertuples(index=False), start=1):
            lines.append(
                " | ".join(
                    [
                        f"[{i:02d}]",
                        f"predictor(domain)='{row.predictor_name}' (id={row.predictor_id})",
                        f"predictor_path='{row.predictor_path}'",
                        f"avg_uncorrected={_fmt_score(row.avg_uncorrected)}",
                        #f"avg_corrected={_fmt_score(row.avg_corrected)}",
                        #f"missing_scores={int(row.missing_scores)}",
                    ]
                )
            )
        return lines

    if aggregate_predictors:
        df_agg = (
            df.groupby("context_id", dropna=False, as_index=False)
            .agg(
                avg_uncorrected=("score", "mean"),
                avg_corrected=("score", lambda s: s.fillna(0).mean()),
                missing_scores=("score", lambda s: int(s.isna().sum())),
                context_name=("context_name", _first_nonnull),
                context_path=("context_full_path", _first_nonnull),
            )
        )
        df_agg = df_agg.sort_values("avg_corrected", ascending=False).head(int(top_k))

        lines: List[str] = []
        for i, row in enumerate(df_agg.itertuples(index=False), start=1):
            lines.append(
                " | ".join(
                    [
                        f"[{i:02d}]",
                        f"context(domain)='{row.context_name}' (id={row.context_id})",
                        f"context_path='{row.context_path}'",
                        f"avg_uncorrected={_fmt_score(row.avg_uncorrected)}",
                        #f"avg_corrected={_fmt_score(row.avg_corrected)}",
                        #f"missing_scores={int(row.missing_scores)}",
                    ]
                )
            )
        return lines

    # No aggregation (edge listing)
    df = df.assign(score_display=df["score"].fillna(0))
    df = df.sort_values("score_display", ascending=False).head(int(top_k))

    lines: List[str] = []
    for i, row in enumerate(df.itertuples(index=False), start=1):
        lines.append(
            " | ".join(
                [
                    f"[{i:02d}]",
                    f"predictor(domain)='{row.predictor_name}' (id={row.predictor_id})",
                    f"predictor_path='{row.predictor_full_path}'",
                    f"context(domain)='{row.context_name}' (id={row.context_id})",
                    f"context_path='{row.context_full_path}'",
                    f"score={_fmt_score(row.score_display)}",
                ]
            )
        )
    return lines


# =============================================================================
# 3) Predictor ↔ Profile mapping
# =============================================================================
def extract_topk_predictor_profile_map(
    csv_path: str,
    top_k: int = DEFAULT_TOP_K_PROFILE,
    predictor_full_paths: Optional[Sequence[str]] = None,
    predictor_ids: Optional[Sequence[Union[str, int]]] = None,
    predictor_ids_filter: Optional[Sequence[Union[str, int]]] = None,
    profile_ids_filter: Optional[Sequence[Union[str, int]]] = None,
    aggregate_profiles: bool = False,
    aggregate_predictors: bool = False,
) -> List[str]:
    """
    Input columns:
      predictor_id,predictor_name,predictor_full_path,profile_id,profile_name,profile_full_path,score

    Missing score semantics:
      - avg_uncorrected: mean(score) ignoring missing (NaN)
      - avg_corrected:   mean(score) treating missing as 0 (NaN -> 0)
    """
    required = [
        "predictor_id",
        "predictor_name",
        "predictor_full_path",
        "profile_id",
        "profile_name",
        "profile_full_path",
        "score",
    ]

    df = pd.read_csv(csv_path)
    _ensure_columns(df, required, csv_path)

    df = _filter_entity(
        df,
        id_col="predictor_id",
        full_path_col="predictor_full_path",
        full_paths=predictor_full_paths,
        ids=predictor_ids,
    )

    df = _apply_id_filter(df, id_col="predictor_id", ids=predictor_ids_filter)
    df = _apply_id_filter(df, id_col="profile_id", ids=profile_ids_filter)

    df["score"] = _coerce_score(df["score"])

    if aggregate_profiles and aggregate_predictors:
        stats = _aggregate_overall(df, score_col="score", left_id_col="predictor_id", right_id_col="profile_id")
        missing_pct = (100.0 * stats["missing_scores"] / stats["rows"]) if stats["rows"] else 0.0
        return [
            " | ".join(
                [
                    "[01]",
                    f"avg_uncorrected={_fmt_score(stats['avg_uncorrected'])}",
                    #f"avg_corrected={_fmt_score(stats['avg_corrected'])}",
                    #f"rows_aggregated={stats['rows']}",
                    #f"missing_scores={stats['missing_scores']} ({missing_pct:.1f}%)",
                    f"predictors={stats['left_unique']}",
                    f"profiles={stats['right_unique']}",
                ]
            )
        ]

    if aggregate_profiles:
        df_agg = (
            df.groupby("predictor_id", dropna=False, as_index=False)
            .agg(
                avg_uncorrected=("score", "mean"),
                avg_corrected=("score", lambda s: s.fillna(0).mean()),
                missing_scores=("score", lambda s: int(s.isna().sum())),
                predictor_name=("predictor_name", _first_nonnull),
                predictor_path=("predictor_full_path", _first_nonnull),
            )
        )
        df_agg = df_agg.sort_values("avg_corrected", ascending=False).head(int(top_k))

        lines: List[str] = []
        for i, row in enumerate(df_agg.itertuples(index=False), start=1):
            lines.append(
                " | ".join(
                    [
                        f"[{i:02d}]",
                        f"predictor(domain)='{row.predictor_name}' (id={row.predictor_id})",
                        f"predictor_path='{row.predictor_path}'",
                        f"avg_uncorrected={_fmt_score(row.avg_uncorrected)}",
                        f"avg_corrected={_fmt_score(row.avg_corrected)}",
                        f"missing_scores={int(row.missing_scores)}",
                    ]
                )
            )
        return lines

    if aggregate_predictors:
        df_agg = (
            df.groupby("profile_id", dropna=False, as_index=False)
            .agg(
                avg_uncorrected=("score", "mean"),
                avg_corrected=("score", lambda s: s.fillna(0).mean()),
                missing_scores=("score", lambda s: int(s.isna().sum())),
                profile_name=("profile_name", _first_nonnull),
                profile_path=("profile_full_path", _first_nonnull),
            )
        )
        df_agg = df_agg.sort_values("avg_corrected", ascending=False).head(int(top_k))

        lines: List[str] = []
        for i, row in enumerate(df_agg.itertuples(index=False), start=1):
            lines.append(
                " | ".join(
                    [
                        f"[{i:02d}]",
                        f"profile(domain)='{row.profile_name}' (id={row.profile_id})",
                        f"profile_path='{row.profile_path}'",
                        f"avg_uncorrected={_fmt_score(row.avg_uncorrected)}",
                        f"avg_corrected={_fmt_score(row.avg_corrected)}",
                        f"missing_scores={int(row.missing_scores)}",
                    ]
                )
            )
        return lines

    # No aggregation (edge listing)
    df = df.assign(score_display=df["score"].fillna(0))
    df = df.sort_values("score_display", ascending=False).head(int(top_k))

    lines: List[str] = []
    for i, row in enumerate(df.itertuples(index=False), start=1):
        lines.append(
            " | ".join(
                [
                    f"[{i:02d}]",
                    f"predictor(domain)='{row.predictor_name}' (id={row.predictor_id})",
                    f"predictor_path='{row.predictor_full_path}'",
                    f"profile(domain)='{row.profile_name}' (id={row.profile_id})",
                    f"profile_path='{row.profile_full_path}'",
                    f"score={_fmt_score(row.score_display)}",
                ]
            )
        )
    return lines


# =============================================================================
# 4) Profile ↔ Barrier mapping
# =============================================================================
def extract_topk_profile_barrier_map(
    csv_path: str,
    top_k: int = DEFAULT_TOP_K_PROFILE_BARRIER,
    barrier_full_paths: Optional[Sequence[str]] = None,
    barrier_ids: Optional[Sequence[Union[str, int]]] = None,
    profile_full_paths: Optional[Sequence[str]] = None,
    profile_ids: Optional[Sequence[Union[str, int]]] = None,
    profile_ids_filter: Optional[Sequence[Union[str, int]]] = None,
    barrier_ids_filter: Optional[Sequence[Union[str, int]]] = None,
    aggregate_barriers: bool = False,
    aggregate_profiles: bool = False,
) -> List[str]:
    """
    Input columns:
      barrier_id,barrier_name,barrier_full_path,profile_id,profile_name,profile_full_path,score

    Missing score semantics:
      - avg_uncorrected: mean(score) ignoring missing (NaN)
      - avg_corrected:   mean(score) treating missing as 0 (NaN -> 0)
    """
    required = [
        "barrier_id",
        "barrier_name",
        "barrier_full_path",
        "profile_id",
        "profile_name",
        "profile_full_path",
        "score",
    ]

    df = pd.read_csv(csv_path)
    _ensure_columns(df, required, csv_path)

    df = _filter_entity(
        df,
        id_col="barrier_id",
        full_path_col="barrier_full_path",
        full_paths=barrier_full_paths,
        ids=barrier_ids,
    )
    df = _filter_entity(
        df,
        id_col="profile_id",
        full_path_col="profile_full_path",
        full_paths=profile_full_paths,
        ids=profile_ids,
    )

    df = _apply_id_filter(df, id_col="profile_id", ids=profile_ids_filter)
    df = _apply_id_filter(df, id_col="barrier_id", ids=barrier_ids_filter)

    df["score"] = _coerce_score(df["score"])

    if aggregate_barriers and aggregate_profiles:
        stats = _aggregate_overall(df, score_col="score", left_id_col="profile_id", right_id_col="barrier_id")
        missing_pct = (100.0 * stats["missing_scores"] / stats["rows"]) if stats["rows"] else 0.0
        return [
            " | ".join(
                [
                    "[01]",
                    f"avg_uncorrected={_fmt_score(stats['avg_uncorrected'])}",
                    #f"avg_corrected={_fmt_score(stats['avg_corrected'])}",
                    #f"rows_aggregated={stats['rows']}",
                    #f"missing_scores={stats['missing_scores']} ({missing_pct:.1f}%)",
                    f"profiles={stats['left_unique']}",
                    f"barriers={stats['right_unique']}",
                ]
            )
        ]

    if aggregate_barriers:
        df_agg = (
            df.groupby("profile_id", dropna=False, as_index=False)
            .agg(
                avg_uncorrected=("score", "mean"),
                avg_corrected=("score", lambda s: s.fillna(0).mean()),
                missing_scores=("score", lambda s: int(s.isna().sum())),
                profile_name=("profile_name", _first_nonnull),
                profile_path=("profile_full_path", _first_nonnull),
            )
        )
        df_agg = df_agg.sort_values("avg_corrected", ascending=False).head(int(top_k))

        lines: List[str] = []
        for i, row in enumerate(df_agg.itertuples(index=False), start=1):
            lines.append(
                " | ".join(
                    [
                        f"[{i:02d}]",
                        f"profile(domain)='{row.profile_name}' (id={row.profile_id})",
                        f"profile_path='{row.profile_path}'",
                        f"avg_uncorrected={_fmt_score(row.avg_uncorrected)}",
                        f"avg_corrected={_fmt_score(row.avg_corrected)}",
                        f"missing_scores={int(row.missing_scores)}",
                    ]
                )
            )
        return lines

    if aggregate_profiles:
        df_agg = (
            df.groupby("barrier_id", dropna=False, as_index=False)
            .agg(
                avg_uncorrected=("score", "mean"),
                avg_corrected=("score", lambda s: s.fillna(0).mean()),
                missing_scores=("score", lambda s: int(s.isna().sum())),
                barrier_name=("barrier_name", _first_nonnull),
                barrier_path=("barrier_full_path", _first_nonnull),
            )
        )
        df_agg = df_agg.sort_values("avg_corrected", ascending=False).head(int(top_k))

        lines: List[str] = []
        for i, row in enumerate(df_agg.itertuples(index=False), start=1):
            lines.append(
                " | ".join(
                    [
                        f"[{i:02d}]",
                        f"barrier(domain)='{row.barrier_name}' (id={row.barrier_id})",
                        f"barrier_path='{row.barrier_path}'",
                        f"avg_uncorrected={_fmt_score(row.avg_uncorrected)}",
                        f"avg_corrected={_fmt_score(row.avg_corrected)}",
                        f"missing_scores={int(row.missing_scores)}",
                    ]
                )
            )
        return lines

    # No aggregation (edge listing)
    df = df.assign(score_display=df["score"].fillna(0))
    df = df.sort_values("score_display", ascending=False).head(int(top_k))

    lines: List[str] = []
    for i, row in enumerate(df.itertuples(index=False), start=1):
        lines.append(
            " | ".join(
                [
                    f"[{i:02d}]",
                    f"profile(domain)='{row.profile_name}' (id={row.profile_id})",
                    f"profile_path='{row.profile_full_path}'",
                    f"barrier(domain)='{row.barrier_name}' (id={row.barrier_id})",
                    f"barrier_path='{row.barrier_full_path}'",
                    f"score={_fmt_score(row.score_display)}",
                ]
            )
        )
    return lines


# =============================================================================
# 5) Coping ↔ Barrier mapping
# =============================================================================
def extract_topk_coping_barrier_map(
    csv_path: str,
    top_k: int = DEFAULT_TOP_K_COPING_BARRIER,
    coping_full_paths: Optional[Sequence[str]] = None,
    coping_ids: Optional[Sequence[Union[str, int]]] = None,
    barrier_full_paths: Optional[Sequence[str]] = None,
    barrier_ids: Optional[Sequence[Union[str, int]]] = None,
    coping_ids_filter: Optional[Sequence[Union[str, int]]] = None,
    barrier_ids_filter: Optional[Sequence[Union[str, int]]] = None,
    aggregate_barriers: bool = False,
    aggregate_copings: bool = False,
) -> List[str]:
    """
    Input columns:
      coping_id,coping_name,coping_full_path,barrier_id,barrier_name,barrier_full_path,score

    Missing score semantics:
      - avg_uncorrected: mean(score) ignoring missing (NaN)
      - avg_corrected:   mean(score) treating missing as 0 (NaN -> 0)
    """
    required = [
        "coping_id",
        "coping_name",
        "coping_full_path",
        "barrier_id",
        "barrier_name",
        "barrier_full_path",
        "score",
    ]

    df = pd.read_csv(csv_path)
    _ensure_columns(df, required, csv_path)

    df = _filter_entity(
        df,
        id_col="coping_id",
        full_path_col="coping_full_path",
        full_paths=coping_full_paths,
        ids=coping_ids,
    )
    df = _filter_entity(
        df,
        id_col="barrier_id",
        full_path_col="barrier_full_path",
        full_paths=barrier_full_paths,
        ids=barrier_ids,
    )

    df = _apply_id_filter(df, id_col="coping_id", ids=coping_ids_filter)
    df = _apply_id_filter(df, id_col="barrier_id", ids=barrier_ids_filter)

    df["score"] = _coerce_score(df["score"])

    if aggregate_barriers and aggregate_copings:
        stats = _aggregate_overall(df, score_col="score", left_id_col="coping_id", right_id_col="barrier_id")
        missing_pct = (100.0 * stats["missing_scores"] / stats["rows"]) if stats["rows"] else 0.0
        return [
            " | ".join(
                [
                    "[01]",
                    f"avg_uncorrected={_fmt_score(stats['avg_uncorrected'])}",
                    #f"avg_corrected={_fmt_score(stats['avg_corrected'])}",
                    #f"rows_aggregated={stats['rows']}",
                    #f"missing_scores={stats['missing_scores']} ({missing_pct:.1f}%)",
                    f"copings={stats['left_unique']}",
                    f"barriers={stats['right_unique']}",
                ]
            )
        ]

    if aggregate_barriers:
        df_agg = (
            df.groupby("coping_id", dropna=False, as_index=False)
            .agg(
                avg_uncorrected=("score", "mean"),
                avg_corrected=("score", lambda s: s.fillna(0).mean()),
                missing_scores=("score", lambda s: int(s.isna().sum())),
                coping_name=("coping_name", _first_nonnull),
                coping_path=("coping_full_path", _first_nonnull),
            )
        )
        df_agg = df_agg.sort_values("avg_corrected", ascending=False).head(int(top_k))

        lines: List[str] = []
        for i, row in enumerate(df_agg.itertuples(index=False), start=1):
            lines.append(
                " | ".join(
                    [
                        f"[{i:02d}]",
                        f"coping(domain)='{row.coping_name}' (id={row.coping_id})",
                        f"coping_path='{row.coping_path}'",
                        f"avg_uncorrected={_fmt_score(row.avg_uncorrected)}",
                        f"avg_corrected={_fmt_score(row.avg_corrected)}",
                        f"missing_scores={int(row.missing_scores)}",
                    ]
                )
            )
        return lines

    if aggregate_copings:
        df_agg = (
            df.groupby("barrier_id", dropna=False, as_index=False)
            .agg(
                avg_uncorrected=("score", "mean"),
                avg_corrected=("score", lambda s: s.fillna(0).mean()),
                missing_scores=("score", lambda s: int(s.isna().sum())),
                barrier_name=("barrier_name", _first_nonnull),
                barrier_path=("barrier_full_path", _first_nonnull),
            )
        )
        df_agg = df_agg.sort_values("avg_corrected", ascending=False).head(int(top_k))

        lines: List[str] = []
        for i, row in enumerate(df_agg.itertuples(index=False), start=1):
            lines.append(
                " | ".join(
                    [
                        f"[{i:02d}]",
                        f"barrier(domain)='{row.barrier_name}' (id={row.barrier_id})",
                        f"barrier_path='{row.barrier_path}'",
                        f"avg_uncorrected={_fmt_score(row.avg_uncorrected)}",
                        f"avg_corrected={_fmt_score(row.avg_corrected)}",
                        f"missing_scores={int(row.missing_scores)}",
                    ]
                )
            )
        return lines

    # No aggregation (edge listing)
    df = df.assign(score_display=df["score"].fillna(0))
    df = df.sort_values("score_display", ascending=False).head(int(top_k))

    lines: List[str] = []
    for i, row in enumerate(df.itertuples(index=False), start=1):
        lines.append(
            " | ".join(
                [
                    f"[{i:02d}]",
                    f"coping(domain)='{row.coping_name}' (id={row.coping_id})",
                    f"coping_path='{row.coping_full_path}'",
                    f"barrier(domain)='{row.barrier_name}' (id={row.barrier_id})",
                    f"barrier_path='{row.barrier_full_path}'",
                    f"score={_fmt_score(row.score_display)}",
                ]
            )
        )
    return lines


# =============================================================================
# 6) Context ↔ Barrier mapping
# =============================================================================
def extract_topk_context_barrier_map(
    csv_path: str,
    top_k: int = DEFAULT_TOP_K_CONTEXT_BARRIER,
    context_full_paths: Optional[Sequence[str]] = None,
    context_ids: Optional[Sequence[Union[str, int]]] = None,
    barrier_full_paths: Optional[Sequence[str]] = None,
    barrier_ids: Optional[Sequence[Union[str, int]]] = None,
    context_ids_filter: Optional[Sequence[Union[str, int]]] = None,
    barrier_ids_filter: Optional[Sequence[Union[str, int]]] = None,
    aggregate_barriers: bool = False,
    aggregate_contexts: bool = False,
) -> List[str]:
    """
    Input columns:
      barrier_id,barrier_name,barrier_full_path,context_id,context_name,context_full_path,score

    Missing score semantics:
      - avg_uncorrected: mean(score) ignoring missing (NaN)
      - avg_corrected:   mean(score) treating missing as 0 (NaN -> 0)
    """
    required = [
        "barrier_id",
        "barrier_name",
        "barrier_full_path",
        "context_id",
        "context_name",
        "context_full_path",
        "score",
    ]

    df = pd.read_csv(csv_path)
    _ensure_columns(df, required, csv_path)

    df = _filter_entity(
        df,
        id_col="context_id",
        full_path_col="context_full_path",
        full_paths=context_full_paths,
        ids=context_ids,
    )
    df = _filter_entity(
        df,
        id_col="barrier_id",
        full_path_col="barrier_full_path",
        full_paths=barrier_full_paths,
        ids=barrier_ids,
    )

    df = _apply_id_filter(df, id_col="context_id", ids=context_ids_filter)
    df = _apply_id_filter(df, id_col="barrier_id", ids=barrier_ids_filter)

    df["score"] = _coerce_score(df["score"])

    if aggregate_barriers and aggregate_contexts:
        stats = _aggregate_overall(df, score_col="score", left_id_col="context_id", right_id_col="barrier_id")
        missing_pct = (100.0 * stats["missing_scores"] / stats["rows"]) if stats["rows"] else 0.0
        return [
            " | ".join(
                [
                    "[01]",
                    f"avg_uncorrected={_fmt_score(stats['avg_uncorrected'])}",
                    #f"avg_corrected={_fmt_score(stats['avg_corrected'])}",
                    #f"rows_aggregated={stats['rows']}",
                    #f"missing_scores={stats['missing_scores']} ({missing_pct:.1f}%)",
                    f"contexts={stats['left_unique']}",
                    f"barriers={stats['right_unique']}",
                ]
            )
        ]

    if aggregate_barriers:
        df_agg = (
            df.groupby("context_id", dropna=False, as_index=False)
            .agg(
                avg_uncorrected=("score", "mean"),
                avg_corrected=("score", lambda s: s.fillna(0).mean()),
                missing_scores=("score", lambda s: int(s.isna().sum())),
                context_name=("context_name", _first_nonnull),
                context_path=("context_full_path", _first_nonnull),
            )
        )
        df_agg = df_agg.sort_values("avg_corrected", ascending=False).head(int(top_k))

        lines: List[str] = []
        for i, row in enumerate(df_agg.itertuples(index=False), start=1):
            lines.append(
                " | ".join(
                    [
                        f"[{i:02d}]",
                        f"context(domain)='{row.context_name}' (id={row.context_id})",
                        f"context_path='{row.context_path}'",
                        f"avg_uncorrected={_fmt_score(row.avg_uncorrected)}",
                        f"avg_corrected={_fmt_score(row.avg_corrected)}",
                        f"missing_scores={int(row.missing_scores)}",
                    ]
                )
            )
        return lines

    if aggregate_contexts:
        df_agg = (
            df.groupby("barrier_id", dropna=False, as_index=False)
            .agg(
                avg_uncorrected=("score", "mean"),
                avg_corrected=("score", lambda s: s.fillna(0).mean()),
                missing_scores=("score", lambda s: int(s.isna().sum())),
                barrier_name=("barrier_name", _first_nonnull),
                barrier_path=("barrier_full_path", _first_nonnull),
            )
        )
        df_agg = df_agg.sort_values("avg_corrected", ascending=False).head(int(top_k))

        lines: List[str] = []
        for i, row in enumerate(df_agg.itertuples(index=False), start=1):
            lines.append(
                " | ".join(
                    [
                        f"[{i:02d}]",
                        f"barrier(domain)='{row.barrier_name}' (id={row.barrier_id})",
                        f"barrier_path='{row.barrier_path}'",
                        f"avg_uncorrected={_fmt_score(row.avg_uncorrected)}",
                        f"avg_corrected={_fmt_score(row.avg_corrected)}",
                        f"missing_scores={int(row.missing_scores)}",
                    ]
                )
            )
        return lines

    # No aggregation (edge listing)
    df = df.assign(score_display=df["score"].fillna(0))
    df = df.sort_values("score_display", ascending=False).head(int(top_k))

    lines: List[str] = []
    for i, row in enumerate(df.itertuples(index=False), start=1):
        lines.append(
            " | ".join(
                [
                    f"[{i:02d}]",
                    f"barrier(domain)='{row.barrier_name}' (id={row.barrier_id})",
                    f"barrier_path='{row.barrier_full_path}'",
                    f"context(domain)='{row.context_name}' (id={row.context_id})",
                    f"context_path='{row.context_full_path}'",
                    f"score={_fmt_score(row.score_display)}",
                ]
            )
        )
    return lines


# =============================================================================
# 7) Predictor ↔ Barrier mapping
# =============================================================================
def extract_topk_predictor_barrier_map(
    csv_path: str,
    top_k: int = DEFAULT_TOP_K_PREDICTOR_BARRIER,
    predictor_full_paths: Optional[Sequence[str]] = None,
    predictor_ids: Optional[Sequence[Union[str, int]]] = None,
    barrier_full_paths: Optional[Sequence[str]] = None,
    barrier_ids: Optional[Sequence[Union[str, int]]] = None,
    predictor_ids_filter: Optional[Sequence[Union[str, int]]] = None,
    barrier_ids_filter: Optional[Sequence[Union[str, int]]] = None,
    aggregate_barriers: bool = False,
    aggregate_predictors: bool = False,
) -> List[str]:
    """
    Input columns:
      predictor_id,predictor_name,predictor_full_path,barrier_id,barrier_name,barrier_full_path,score

    Missing score semantics:
      - avg_uncorrected: mean(score) ignoring missing (NaN)
      - avg_corrected:   mean(score) treating missing as 0 (NaN -> 0)
    """
    required = [
        "predictor_id",
        "predictor_name",
        "predictor_full_path",
        "barrier_id",
        "barrier_name",
        "barrier_full_path",
        "score",
    ]

    df = pd.read_csv(csv_path)
    _ensure_columns(df, required, csv_path)

    df = _filter_entity(
        df,
        id_col="predictor_id",
        full_path_col="predictor_full_path",
        full_paths=predictor_full_paths,
        ids=predictor_ids,
    )
    df = _filter_entity(
        df,
        id_col="barrier_id",
        full_path_col="barrier_full_path",
        full_paths=barrier_full_paths,
        ids=barrier_ids,
    )

    df = _apply_id_filter(df, id_col="predictor_id", ids=predictor_ids_filter)
    df = _apply_id_filter(df, id_col="barrier_id", ids=barrier_ids_filter)

    df["score"] = _coerce_score(df["score"])

    if aggregate_barriers and aggregate_predictors:
        stats = _aggregate_overall(df, score_col="score", left_id_col="predictor_id", right_id_col="barrier_id")
        missing_pct = (100.0 * stats["missing_scores"] / stats["rows"]) if stats["rows"] else 0.0
        return [
            " | ".join(
                [
                    "[01]",
                    f"avg_uncorrected={_fmt_score(stats['avg_uncorrected'])}",
                    #f"avg_corrected={_fmt_score(stats['avg_corrected'])}",
                    #f"rows_aggregated={stats['rows']}",
                    #f"missing_scores={stats['missing_scores']} ({missing_pct:.1f}%)",
                    f"predictors={stats['left_unique']}",
                    f"barriers={stats['right_unique']}",
                ]
            )
        ]

    if aggregate_barriers:
        df_agg = (
            df.groupby("predictor_id", dropna=False, as_index=False)
            .agg(
                avg_uncorrected=("score", "mean"),
                avg_corrected=("score", lambda s: s.fillna(0).mean()),
                missing_scores=("score", lambda s: int(s.isna().sum())),
                predictor_name=("predictor_name", _first_nonnull),
                predictor_path=("predictor_full_path", _first_nonnull),
            )
        )
        df_agg = df_agg.sort_values("avg_corrected", ascending=False).head(int(top_k))

        lines: List[str] = []
        for i, row in enumerate(df_agg.itertuples(index=False), start=1):
            lines.append(
                " | ".join(
                    [
                        f"[{i:02d}]",
                        f"predictor(domain)='{row.predictor_name}' (id={row.predictor_id})",
                        f"predictor_path='{row.predictor_path}'",
                        f"avg_uncorrected={_fmt_score(row.avg_uncorrected)}",
                        f"avg_corrected={_fmt_score(row.avg_corrected)}",
                        f"missing_scores={int(row.missing_scores)}",
                    ]
                )
            )
        return lines

    if aggregate_predictors:
        df_agg = (
            df.groupby("barrier_id", dropna=False, as_index=False)
            .agg(
                avg_uncorrected=("score", "mean"),
                avg_corrected=("score", lambda s: s.fillna(0).mean()),
                missing_scores=("score", lambda s: int(s.isna().sum())),
                barrier_name=("barrier_name", _first_nonnull),
                barrier_path=("barrier_full_path", _first_nonnull),
            )
        )
        df_agg = df_agg.sort_values("avg_corrected", ascending=False).head(int(top_k))

        lines: List[str] = []
        for i, row in enumerate(df_agg.itertuples(index=False), start=1):
            lines.append(
                " | ".join(
                    [
                        f"[{i:02d}]",
                        f"barrier(domain)='{row.barrier_name}' (id={row.barrier_id})",
                        f"barrier_path='{row.barrier_path}'",
                        f"avg_uncorrected={_fmt_score(row.avg_uncorrected)}",
                        f"avg_corrected={_fmt_score(row.avg_corrected)}",
                        f"missing_scores={int(row.missing_scores)}",
                    ]
                )
            )
        return lines

    # No aggregation (edge listing)
    df = df.assign(score_display=df["score"].fillna(0))
    df = df.sort_values("score_display", ascending=False).head(int(top_k))

    lines: List[str] = []
    for i, row in enumerate(df.itertuples(index=False), start=1):
        lines.append(
            " | ".join(
                [
                    f"[{i:02d}]",
                    f"predictor(domain)='{row.predictor_name}' (id={row.predictor_id})",
                    f"predictor_path='{row.predictor_full_path}'",
                    f"barrier(domain)='{row.barrier_name}' (id={row.barrier_id})",
                    f"barrier_path='{row.barrier_full_path}'",
                    f"score={_fmt_score(row.score_display)}",
                ]
            )
        )
    return lines


# -------------------------
# Example console runner
# -------------------------
if __name__ == "__main__":
    # Predictor ↔ other mappings
    CRITERION_PREDICTOR_CSV = "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/ontology_mappings/CRITERION/predictor_to_criterion/results/gpt-5-nano/predictor_to_criterion_edges_long.csv"
    CONTEXT_PREDICTOR_CSV = "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/ontology_mappings/PREDICTOR/context_to_predictor/results/gpt-5-nano/predictor_to_context_edges_long.csv"
    PROFILE_PREDICTOR_CSV = "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/ontology_mappings/PREDICTOR/profile_to_predictor/results/gpt-5-nano/predictor_to_profile_edges_long.csv"

    # Barrier ↔ other mappings (HAPA-based translation)
    PROFILE_BARRIER_CSV = "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/ontology_mappings/HAPA/profile_to_barrier/results/gpt-5-nano/profile_to_barrier_edges_long.csv"
    COPING_BARRIER_CSV = "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/ontology_mappings/HAPA/coping_to_barrier/results/gpt-5-nano/coping_to_barrier_edges_long.csv"
    CONTEXT_BARRIER_CSV = "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/ontology_mappings/HAPA/context_to_barrier/results/gpt-5-nano/context_to_barrier_edges_long.csv"

    # Predictor ↔ Barrier mapping (HAPA-based translation)
    BARRIER_PREDICTOR_CSV = "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/ontology_mappings/PREDICTOR/barrier_to_predictor/results/gpt-5-nano/predictor_to_barrier_edges_long.csv"

    # Optional: restrict to a specific set of predictor paths
    predictor_paths = None

    # By default: leave the ID lists empty (no filtering)
    pc_predictor_ids_filter: List[int] = []
    pc_cluster_ids_filter: List[Union[str, int]] = []

    predictor_ids_filter: List[int] = []
    context_ids_filter: List[int] = []
    profile_ids_filter: List[int] = []
    barrier_ids_filter: List[int] = []
    coping_ids_filter: List[int] = []

    # --- Run one example (PC mapping) ---
    criterion_lines = extract_topk_predictor_criterion_map(
        CRITERION_PREDICTOR_CSV,
        top_k=DEFAULT_TOP_K_CRITERION,
        predictor_full_paths=predictor_paths,
        predictor_ids=None,
        predictor_ids_filter=pc_predictor_ids_filter,
        criterion_cluster_ids_filter=pc_cluster_ids_filter,
        aggregate_criterions=True,     # default requested for PC
        aggregate_predictors=False,    # default requested for PC
        # full_matrix_shape defaults to (143, 3110)
    )
    _print_lines("TOP Predictor ↔ Criterion-cluster mappings", criterion_lines)

    # Uncomment any other runs as needed.
    context_lines = extract_topk_predictor_context_map(
         CONTEXT_PREDICTOR_CSV,
         top_k=DEFAULT_TOP_K_CONTEXT,
         predictor_full_paths=predictor_paths,
         predictor_ids=None,
         predictor_ids_filter=predictor_ids_filter,
         context_ids_filter=context_ids_filter,
         aggregate_contexts=True,
         aggregate_predictors=False,
     )
    _print_lines("TOP Predictor ↔ Context mappings", context_lines)

    profile_lines = extract_topk_predictor_profile_map(
         PROFILE_PREDICTOR_CSV,
         top_k=DEFAULT_TOP_K_PROFILE,
         predictor_full_paths=predictor_paths,
         predictor_ids=None,
         predictor_ids_filter=predictor_ids_filter,
         profile_ids_filter=profile_ids_filter,
         aggregate_profiles=True,
         aggregate_predictors=False,
     )
    _print_lines("TOP Predictor ↔ Profile mappings", profile_lines)

    profile_barrier_lines = extract_topk_profile_barrier_map(
         PROFILE_BARRIER_CSV,
         top_k=DEFAULT_TOP_K_PROFILE_BARRIER,
         profile_ids_filter=profile_ids_filter,
         barrier_ids_filter=barrier_ids_filter,
         aggregate_barriers=False,
         aggregate_profiles=True,
     )
    _print_lines("TOP Profile ↔ Barrier mappings", profile_barrier_lines)

    context_barrier_lines = extract_topk_context_barrier_map(
         CONTEXT_BARRIER_CSV,
         top_k=DEFAULT_TOP_K_CONTEXT_BARRIER,
         context_ids_filter=context_ids_filter,
         barrier_ids_filter=barrier_ids_filter,
         aggregate_barriers=False,
         aggregate_contexts=True,
     )
    _print_lines("TOP Context ↔ Barrier mappings", context_barrier_lines)

    #predictor_barrier_lines = extract_topk_predictor_barrier_map(
    #     BARRIER_PREDICTOR_CSV,
    #     top_k=DEFAULT_TOP_K_PREDICTOR_BARRIER,
    #     predictor_full_paths=predictor_paths,
    #     predictor_ids=None,
    #     predictor_ids_filter=predictor_ids_filter,
    #     barrier_ids_filter=barrier_ids_filter,
    #     aggregate_barriers=False,
    #     aggregate_predictors=False,
    # )
    #_print_lines("TOP Predictor ↔ Barrier mappings", predictor_barrier_lines)

    coping_barrier_lines = extract_topk_coping_barrier_map(
         COPING_BARRIER_CSV,
         top_k=DEFAULT_TOP_K_COPING_BARRIER,
         coping_ids_filter=coping_ids_filter,
         barrier_ids_filter=barrier_ids_filter,
         aggregate_barriers=True,
         aggregate_copings=False,
     )
    _print_lines("TOP Coping ↔ Barrier mappings", coping_barrier_lines)