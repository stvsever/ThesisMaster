from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd

# Current script is dependent on two proprietary scripts in same /helpers directory
import extract_topk_mapped_scores as etms
import infer_cluster_ids as ici

# -------------------------
# Configuration defaults
# -------------------------
def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        has_eval = (candidate / "evaluation").exists() or (candidate / "Evaluation").exists()
        if has_eval and (candidate / "README.md").exists():
            return candidate
    raise RuntimeError("Could not locate repository root from rank_best_predictors.py")


REPO_ROOT = _find_repo_root()

DEFAULT_MAPPED_CRITERIONS_CSV = (
    str(REPO_ROOT / "evaluation/02_mental_health_issue_operationalization/mapped_criterions.csv")
)

DEFAULT_CLUSTER_JSON = (
    str(REPO_ROOT / "src/utils/official/cluster_criterions/results/04_semantically_clustered_items.json")
)

# Single output CSV (ALL pseudoprofiles, ALL parts) will be written here by default
DEFAULT_OUTPUT_DIR = (
    str(REPO_ROOT / "evaluation/03_construction_initial_observation_model/helpers/00_LLM_based_mapping_based_predictor_ranks")
)
DEFAULT_OUTPUT_CSV_NAME = "all_pseudoprofiles__predictor_ranks_dense.csv"

# Requested sizes
DEFAULT_TOP_K_AGGREGATE = 50          # 50 PRE-global + 50 POST-global
DEFAULT_TOP_K_PER_CRITERION = 10      # 10 POST-per-criterion

CRITERION_PREDICTOR_CSV = (
    str(
        REPO_ROOT
        / "src/utils/official/ontology_mappings/CRITERION/predictor_to_criterion/results/gpt-5-nano/predictor_to_criterion_edges_long.csv"
    )
)

# Optional: restrict to a specific set of predictor paths
predictor_paths = None

# Optional predictor id filters (leave empty for no filtering)
pc_predictor_ids_filter: List[int] = []

# If you want a fixed matrix shape for reproducibility/testing
FULL_MATRIX_SHAPE = (143, 3110)

# If empty, the script processes ALL pseudoprofiles in mapped_criterions.csv.
DEFAULT_PSEUDOPROFILES_TO_PROCESS: List[str] = []

# -------------------------
# Parsing helpers (mapped_criterions.csv)
# -------------------------

_TOP5_SPLIT_RE = re.compile(r"\s*\|\|\s*")
_LEAF_EXTRACT_RE = re.compile(r"(?:^|[|])\s*leaf\s*=\s*(?P<leaf>.*)\s*$", re.IGNORECASE)


def _parse_top5_candidates_cell(cell: Optional[str]) -> List[str]:
    """
    Parse a 'top5_candidates' cell like:
      'idx=11726|fused=0.0034|leaf=... || idx=...|leaf=... || ...'
    Returns extracted leaf paths.
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []

    s = str(cell).strip()
    if not s:
        return []

    chunks = [c.strip() for c in _TOP5_SPLIT_RE.split(s) if c.strip()]
    leaf_paths: List[str] = []

    for chunk in chunks:
        if "leaf=" in chunk:
            _, _, tail = chunk.partition("leaf=")
            leaf = tail.strip()
        else:
            m = _LEAF_EXTRACT_RE.search(chunk)
            leaf = (m.group("leaf").strip() if m else "")

        if leaf:
            leaf_paths.append(leaf)

    return leaf_paths


def _parse_semicolon_paths(cell: Optional[str]) -> List[str]:
    """
    Parse 'complaint_unique_mapped_leaf_embed_paths' cell like:
      'path1 ; path2 ; path3'
    Returns list of paths.
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    s = str(cell).strip()
    if not s:
        return []
    return [p.strip() for p in s.split(";") if p.strip()]


def _flatten_inferred_cluster_ids(
    inferred: Dict[str, Union[str, List[str], None]]
) -> List[str]:
    """
    Convert infer_cluster_ids(...) output to a de-duplicated list preserving order.
    """
    out: List[str] = []
    seen = set()

    for _path, cid in inferred.items():
        if cid is None:
            continue

        if isinstance(cid, str):
            cands = [cid]
        else:
            cands = list(cid)

        for c in cands:
            if c not in seen:
                seen.add(c)
                out.append(c)

    return out


def infer_cluster_ids_for_paths(
    cluster_json_path: Union[str, Path], leaf_paths: Iterable[str]
) -> List[str]:
    """
    Convenience wrapper: infer cluster ids for a list of leaf paths and return unique list.
    """
    leaf_paths = [p.strip() for p in leaf_paths if p and str(p).strip()]
    if not leaf_paths:
        return []
    inferred = ici.infer_cluster_ids(cluster_json_path, leaf_paths)
    return _flatten_inferred_cluster_ids(inferred)


def infer_cluster_ids_with_path_context(
    cluster_json_path: Union[str, Path], leaf_paths: List[str]
) -> Tuple[Dict[str, Union[str, List[str], None]], List[str]]:
    """
    Returns:
      inferred_map: {leaf_path: cluster_id_or_candidates_or_none}
      unique_cluster_ids: flattened + de-duplicated cluster ID list
    """
    leaf_paths = [p.strip() for p in leaf_paths if p and str(p).strip()]
    if not leaf_paths:
        return {}, []

    inferred_map = ici.infer_cluster_ids(cluster_json_path, leaf_paths)
    unique_ids = _flatten_inferred_cluster_ids(inferred_map)
    return inferred_map, unique_ids


def _criterion_leaf_label(criterion_path: str) -> str:
    """
    Human-friendly leaf label.
    Handles either "a / b / c" or "a/b/c".
    """
    if not criterion_path:
        return ""
    parts = [p.strip() for p in criterion_path.split("/") if p.strip()]
    return parts[-1] if parts else criterion_path.strip()


# -------------------------
# Score cleaning helper (keeps output stable; also helps parsing)
# -------------------------

_REL_SCORE_CLEAN_RE = re.compile(
    r"\s*\|\s*avg_uncorrected=(?P<u>\d+(?:\.\d+)?)\s*\|\s*avg_corrected=(?P<c>\d+(?:\.\d+)?)"
)


def clean_rel_scores(lines: List[str]) -> List[str]:
    """
    Replace '| avg_uncorrected=X | avg_corrected=X' with '| relevance_score=X'
    when both values are identical (numeric equality).

    Leaves the line unchanged if values differ or pattern isn't present.
    """
    cleaned: List[str] = []

    for line in lines:
        m = _REL_SCORE_CLEAN_RE.search(line)
        if not m:
            cleaned.append(line)
            continue

        u_s = m.group("u")
        c_s = m.group("c")

        try:
            u_v = float(u_s)
            c_v = float(c_s)
        except ValueError:
            cleaned.append(line)
            continue

        if abs(u_v - c_v) < 1e-12:
            line = _REL_SCORE_CLEAN_RE.sub(f" | relevance_score={u_s}", line)

        cleaned.append(line)

    return cleaned


# -------------------------
# Helper: parse etms textual lines -> structured rows
# -------------------------

_SCORE_RE_LIST = [
    re.compile(r"relevance_score\s*=\s*(?P<s>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"),
    re.compile(r"avg_corrected\s*=\s*(?P<s>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"),
    re.compile(r"avg_uncorrected\s*=\s*(?P<s>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"),
    re.compile(r"score\s*=\s*(?P<s>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"),
]

_PATH_RE_LIST = [
    re.compile(r"predictor_full_path\s*=\s*(?P<p>.+?)(?:\s*\|\s*|$)"),
    re.compile(r"predictor_path\s*=\s*(?P<p>.+?)(?:\s*\|\s*|$)"),
    re.compile(r"predictor\s*=\s*(?P<p>.+?)(?:\s*\|\s*|$)"),
]


def _extract_score(line: str) -> Optional[float]:
    for rx in _SCORE_RE_LIST:
        m = rx.search(line)
        if m:
            try:
                return float(m.group("s"))
            except ValueError:
                return None
    return None


def _extract_predictor_path(line: str) -> Optional[str]:
    # First try explicit regexes
    for rx in _PATH_RE_LIST:
        m = rx.search(line)
        if m:
            p = m.group("p").strip()
            return p if p else None

    # Fallback: token scan on "|" boundaries for something like "predictor*=..."
    toks = [t.strip() for t in str(line).split("|") if t.strip()]
    for t in toks:
        if "=" not in t:
            continue
        k, v = t.split("=", 1)
        k = k.strip().lower()
        if "predictor" in k:
            vv = v.strip()
            return vv if vv else None

    return None


def lines_to_dense_df(
    lines: List[str],
    *,
    top_k: int,
    pseudoprofile_id: str,
    part: str,
    criterion_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convert etms.extract_topk_predictor_criterion_map textual lines to a dense DataFrame.

    Output columns (SINGLE CSV schema):
      - pseudoprofile_id
      - part: one of {"pre_global", "post_global", "post_per_criterion"}
      - criterion_path (nullable)
      - criterion_leaf (nullable)
      - rank
      - predictor_path
      - relevance_score

    Non-redundant within each (pseudoprofile_id, part, criterion_path):
      - de-duplicate predictor_path (keep max relevance_score; tie -> best raw_rank)
      - then dense re-rank 1..N by relevance_score desc
    """
    rows: List[Dict[str, object]] = []
    for i, raw in enumerate(lines, start=1):
        line = str(raw).strip()
        if not line:
            continue

        p = _extract_predictor_path(line)
        s = _extract_score(line)
        if p is None or s is None:
            continue

        row: Dict[str, object] = {
            "pseudoprofile_id": str(pseudoprofile_id),
            "part": str(part),
            "criterion_path": (str(criterion_path) if criterion_path is not None else ""),
            "criterion_leaf": (_criterion_leaf_label(str(criterion_path)) if criterion_path is not None else ""),
            "raw_rank": int(i),
            "predictor_path": str(p),
            "relevance_score": float(s),
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=[
                "pseudoprofile_id",
                "part",
                "criterion_path",
                "criterion_leaf",
                "rank",
                "predictor_path",
                "relevance_score",
            ]
        )

    df = pd.DataFrame(rows)

    # Normalize criterion fields: keep empty string for globals to avoid NaN drift in CSV
    if "criterion_path" not in df.columns:
        df["criterion_path"] = ""
    if "criterion_leaf" not in df.columns:
        df["criterion_leaf"] = ""

    key_cols = ["pseudoprofile_id", "part", "criterion_path", "predictor_path"]

    # De-duplicate predictor_path within each key by best score, tie -> best raw_rank
    df = df.sort_values(["relevance_score", "raw_rank"], ascending=[False, True]).copy()
    df = df.drop_duplicates(subset=key_cols, keep="first")

    # Dense rank within each (pseudoprofile_id, part, criterion_path)
    df = df.sort_values(
        ["pseudoprofile_id", "part", "criterion_path", "relevance_score", "raw_rank"],
        ascending=[True, True, True, False, True],
    ).copy()

    df["rank"] = df.groupby(["pseudoprofile_id", "part", "criterion_path"], sort=False).cumcount() + 1
    df = df[df["rank"] <= int(top_k)].copy()

    df = df[
        [
            "pseudoprofile_id",
            "part",
            "criterion_path",
            "criterion_leaf",
            "rank",
            "predictor_path",
            "relevance_score",
        ]
    ].reset_index(drop=True)

    return df


# -------------------------
# Core logic (cluster filters + 3 parts per pseudoprofile)
# -------------------------

def build_pre_post_cluster_filters_for_pseudoprofile_df(
    sub: pd.DataFrame,
    cluster_json_path: Union[str, Path],
) -> Tuple[List[str], List[str]]:
    """
    Returns:
      pc_cluster_ids_filter_pre  - inferred from top5_candidates across rows (pre-selection)
      pc_cluster_ids_filter_post - inferred from complaint_unique_mapped_leaf_embed_paths (post-selection aggregate)

    IMPORTANT:
    - 'complaint_unique_mapped_leaf_embed_paths' repeats the *full selected list* across rows for same pseudoprofile,
      so we take the first non-empty occurrence.
    """
    if "top5_candidates" not in sub.columns:
        raise ValueError("mapped_criterions.csv must contain a 'top5_candidates' column.")

    pre_leaf_paths: List[str] = []
    for cell in sub["top5_candidates"].tolist():
        pre_leaf_paths.extend(_parse_top5_candidates_cell(cell))

    pc_cluster_ids_filter_pre = infer_cluster_ids_for_paths(cluster_json_path, pre_leaf_paths)

    if "complaint_unique_mapped_leaf_embed_paths" not in sub.columns:
        raise ValueError(
            "mapped_criterions.csv must contain a 'complaint_unique_mapped_leaf_embed_paths' column."
        )

    post_cell = None
    for cell in sub["complaint_unique_mapped_leaf_embed_paths"].tolist():
        if _parse_semicolon_paths(cell):
            post_cell = cell
            break

    post_leaf_paths = _parse_semicolon_paths(post_cell)
    pc_cluster_ids_filter_post = infer_cluster_ids_for_paths(cluster_json_path, post_leaf_paths)

    return pc_cluster_ids_filter_pre, pc_cluster_ids_filter_post


def get_post_selected_paths_for_pseudoprofile(sub: pd.DataFrame) -> List[str]:
    """
    Returns post-selected criterion leaf paths from 'complaint_unique_mapped_leaf_embed_paths'.

    NOTE:
      This column repeats the full selected list across rows for the pseudoprofile.
      We take the first non-empty occurrence.
    """
    if "complaint_unique_mapped_leaf_embed_paths" not in sub.columns:
        raise ValueError(
            "mapped_criterions.csv must contain a 'complaint_unique_mapped_leaf_embed_paths' column."
        )

    for cell in sub["complaint_unique_mapped_leaf_embed_paths"].tolist():
        paths = _parse_semicolon_paths(cell)
        if paths:
            return paths

    return []


def run_pre_selection_aggregate_lines(
    pc_cluster_ids_filter_pre: List[Union[str, int]],
    top_k_aggregate: int,
) -> List[str]:
    if not pc_cluster_ids_filter_pre:
        return []
    lines = etms.extract_topk_predictor_criterion_map(
        CRITERION_PREDICTOR_CSV,
        top_k=top_k_aggregate,
        predictor_full_paths=predictor_paths,
        predictor_ids=None,
        predictor_ids_filter=pc_predictor_ids_filter,
        criterion_cluster_ids_filter=pc_cluster_ids_filter_pre,
        aggregate_criterions=True,
        aggregate_predictors=False,
        full_matrix_shape=FULL_MATRIX_SHAPE,
    )
    return clean_rel_scores(lines)


def run_post_selection_aggregate_lines(
    pc_cluster_ids_filter_post: List[Union[str, int]],
    top_k_aggregate: int,
) -> List[str]:
    if not pc_cluster_ids_filter_post:
        return []
    lines = etms.extract_topk_predictor_criterion_map(
        CRITERION_PREDICTOR_CSV,
        top_k=top_k_aggregate,
        predictor_full_paths=predictor_paths,
        predictor_ids=None,
        predictor_ids_filter=pc_predictor_ids_filter,
        criterion_cluster_ids_filter=pc_cluster_ids_filter_post,
        aggregate_criterions=True,
        aggregate_predictors=False,
        full_matrix_shape=FULL_MATRIX_SHAPE,
    )
    return clean_rel_scores(lines)


def run_post_selection_per_criterion_dense_rows(
    pseudoprofile_id: str,
    cluster_json_path: Union[str, Path],
    post_selected_paths: List[str],
    top_k_per_criterion: int,
) -> pd.DataFrame:
    """
    POST-selection per criterion path:
      For each selected criterion leaf path:
        - infer its cluster ID(s)
        - run mapping restricted to those cluster IDs
        - merge across candidate cluster IDs (if multiple) by taking best score per predictor_path
        - then produce top_k_per_criterion rows per criterion.

    Returns DataFrame in SINGLE CSV schema (part='post_per_criterion').
    """
    if not post_selected_paths:
        return pd.DataFrame(
            columns=[
                "pseudoprofile_id",
                "part",
                "criterion_path",
                "criterion_leaf",
                "rank",
                "predictor_path",
                "relevance_score",
            ]
        )

    inferred_map, _ = infer_cluster_ids_with_path_context(cluster_json_path, post_selected_paths)
    out_tables: List[pd.DataFrame] = []

    for criterion_path in post_selected_paths:
        cid = inferred_map.get(criterion_path)
        if cid is None:
            continue

        candidate_cids = [cid] if isinstance(cid, str) else list(cid)
        candidate_cids = [c for c in candidate_cids if c is not None]
        if not candidate_cids:
            continue

        # collect per-candidate results, then merge best predictor scores across candidates for THIS criterion_path
        per_candidate_rows: List[pd.DataFrame] = []
        for one_cid in candidate_cids:
            lines = etms.extract_topk_predictor_criterion_map(
                CRITERION_PREDICTOR_CSV,
                top_k=top_k_per_criterion,
                predictor_full_paths=predictor_paths,
                predictor_ids=None,
                predictor_ids_filter=pc_predictor_ids_filter,
                criterion_cluster_ids_filter=[one_cid],
                aggregate_criterions=True,
                aggregate_predictors=False,
                full_matrix_shape=FULL_MATRIX_SHAPE,
            )
            lines = clean_rel_scores(lines)
            df_one = lines_to_dense_df(
                lines,
                top_k=top_k_per_criterion,
                pseudoprofile_id=pseudoprofile_id,
                part="post_per_criterion",
                criterion_path=criterion_path,
            )
            if not df_one.empty:
                per_candidate_rows.append(df_one)

        if not per_candidate_rows:
            continue

        merged = pd.concat(per_candidate_rows, ignore_index=True)

        # Non-redundant across candidate CIDs for same criterion:
        # keep best score per (pseudoprofile_id, part, criterion_path, predictor_path)
        merged = merged.sort_values(["relevance_score", "rank"], ascending=[False, True]).copy()
        merged = merged.drop_duplicates(
            subset=["pseudoprofile_id", "part", "criterion_path", "predictor_path"],
            keep="first",
        )

        # Re-rank within criterion after merge
        merged = merged.sort_values(
            ["pseudoprofile_id", "part", "criterion_path", "relevance_score"],
            ascending=[True, True, True, False],
        ).copy()
        merged["rank"] = merged.groupby(["pseudoprofile_id", "part", "criterion_path"], sort=False).cumcount() + 1
        merged = merged[merged["rank"] <= int(top_k_per_criterion)].copy()

        out_tables.append(merged.reset_index(drop=True))

    if not out_tables:
        return pd.DataFrame(
            columns=[
                "pseudoprofile_id",
                "part",
                "criterion_path",
                "criterion_leaf",
                "rank",
                "predictor_path",
                "relevance_score",
            ]
        )

    out = pd.concat(out_tables, ignore_index=True)
    out = out.sort_values(["pseudoprofile_id", "criterion_path", "rank"], ascending=[True, True, True]).reset_index(
        drop=True
    )
    return out


# -------------------------
# Single-CSV writer (append per pseudoprofile)
# -------------------------

def _safe_mkdir(p: Union[str, Path]) -> Path:
    pp = Path(p).expanduser()
    pp.mkdir(parents=True, exist_ok=True)
    return pp


def _init_output_csv(out_csv_path: Path) -> None:
    """
    Create/overwrite the output CSV with header only.
    """
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    empty = pd.DataFrame(
        columns=[
            "pseudoprofile_id",
            "part",
            "criterion_path",
            "criterion_leaf",
            "rank",
            "predictor_path",
            "relevance_score",
        ]
    )
    empty.to_csv(out_csv_path, index=False)


def _append_df_to_csv(out_csv_path: Path, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    df.to_csv(out_csv_path, mode="a", header=False, index=False)


# -------------------------
# Orchestrator
# -------------------------

def run_all_pseudoprofiles_single_csv(
    mapped_criterions_csv: Union[str, Path],
    cluster_json_path: Union[str, Path],
    out_csv_path: Union[str, Path],
    top_k_aggregate: int,
    top_k_per_criterion: int,
    pseudoprofiles_to_process: Optional[List[str]] = None,
    verbose: bool = True,
) -> Path:
    """
    Writes ONE single dense CSV containing:
      - 50 PRE-global rows per pseudoprofile (part='pre_global', criterion_path='')
      - 50 POST-global rows per pseudoprofile (part='post_global', criterion_path='')
      - 10 POST-per-criterion rows per selected criterion (part='post_per_criterion')

    Non-redundant:
      - within each (pseudoprofile_id, part, criterion_path): predictor_path unique, keep best relevance_score.
    """
    mapped_criterions_csv = Path(mapped_criterions_csv).expanduser()
    cluster_json_path = Path(cluster_json_path).expanduser()
    out_csv_path = Path(out_csv_path).expanduser()

    df = pd.read_csv(mapped_criterions_csv)
    if "pseudoprofile_id" not in df.columns:
        raise ValueError("mapped_criterions.csv must contain a 'pseudoprofile_id' column.")

    # Preserve order of first appearance
    all_ids: List[str] = []
    seen = set()
    for v in df["pseudoprofile_id"].dropna().astype(str).tolist():
        if v not in seen:
            seen.add(v)
            all_ids.append(v)

    if pseudoprofiles_to_process:
        wanted = set(str(x) for x in pseudoprofiles_to_process)
        ids_to_run = [pid for pid in all_ids if str(pid) in wanted]
    else:
        ids_to_run = all_ids

    if not ids_to_run:
        raise ValueError("No pseudoprofile_id values selected to run.")

    # Start fresh
    _init_output_csv(out_csv_path)

    if verbose:
        print(f"\n=== Writing single CSV to: {out_csv_path} ===")
        print(f"=== Running {len(ids_to_run)} pseudoprofiles ===")

    for pid in ids_to_run:
        sub = df[df["pseudoprofile_id"].astype(str) == str(pid)].copy()
        if sub.empty:
            continue

        post_paths = get_post_selected_paths_for_pseudoprofile(sub)

        pc_cluster_ids_filter_pre, pc_cluster_ids_filter_post = build_pre_post_cluster_filters_for_pseudoprofile_df(
            sub=sub,
            cluster_json_path=cluster_json_path,
        )

        # 1) PRE-global
        pre_lines = run_pre_selection_aggregate_lines(pc_cluster_ids_filter_pre, top_k_aggregate)
        pre_df = lines_to_dense_df(
            pre_lines,
            top_k=top_k_aggregate,
            pseudoprofile_id=str(pid),
            part="pre_global",
            criterion_path=None,
        )

        # 2) POST-global
        post_lines = run_post_selection_aggregate_lines(pc_cluster_ids_filter_post, top_k_aggregate)
        post_df = lines_to_dense_df(
            post_lines,
            top_k=top_k_aggregate,
            pseudoprofile_id=str(pid),
            part="post_global",
            criterion_path=None,
        )

        # 3) POST-per-criterion
        percrit_df = run_post_selection_per_criterion_dense_rows(
            pseudoprofile_id=str(pid),
            cluster_json_path=cluster_json_path,
            post_selected_paths=post_paths,
            top_k_per_criterion=top_k_per_criterion,
        )

        # Append in deterministic order
        _append_df_to_csv(out_csv_path, pre_df)
        _append_df_to_csv(out_csv_path, post_df)
        _append_df_to_csv(out_csv_path, percrit_df)

        if verbose:
            print(
                f"[{pid}] appended rows: "
                f"PRE={len(pre_df)} | POST={len(post_df)} | POST-per-criterion={len(percrit_df)} "
                f"(criteria={len(post_paths)})"
            )

    if verbose:
        print(f"\nDONE. Single CSV written: {out_csv_path}")

    return out_csv_path


# -------------------------
# CLI / main
# -------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Loop all pseudoprofiles and write ONE single dense CSV with:\n"
            "  (1) 50 PRE-global\n"
            "  (2) 50 POST-global\n"
            "  (3) 10 POST-per-criterion (per selected criterion)\n"
            "\nSchema:\n"
            "  pseudoprofile_id, part, criterion_path, criterion_leaf, rank, predictor_path, relevance_score\n"
        )
    )
    parser.add_argument(
        "--mapped_criterions_csv",
        type=str,
        default=DEFAULT_MAPPED_CRITERIONS_CSV,
        help="Path to mapped_criterions.csv",
    )
    parser.add_argument(
        "--cluster_json",
        type=str,
        default=DEFAULT_CLUSTER_JSON,
        help="Path to 04_semantically_clustered_items.json",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the single CSV will be written (default output dir).",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help=(
            "Optional explicit output CSV path. If omitted, uses --out_dir + "
            f"'{DEFAULT_OUTPUT_CSV_NAME}'."
        ),
    )
    parser.add_argument(
        "--top_k_aggregate",
        type=int,
        default=DEFAULT_TOP_K_AGGREGATE,
        help="Top-K for PRE-global and POST-global mapping calls (default=50).",
    )
    parser.add_argument(
        "--top_k_per_criterion",
        type=int,
        default=DEFAULT_TOP_K_PER_CRITERION,
        help="Top-K for each POST per-criterion mapping call (default=10).",
    )
    parser.add_argument(
        "--pseudoprofiles",
        nargs="*",
        default=None,  # handled in code
        help=(
            "Optional list of pseudoprofile IDs. If omitted, processes ALL pseudoprofiles in the CSV.\n"
            "Example: --pseudoprofiles pseudoprofile_FTC_ID001 pseudoprofile_FTC_ID002"
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress prints.",
    )

    args = parser.parse_args()

    # Decide output path
    out_dir = _safe_mkdir(args.out_dir)
    if args.out_csv:
        out_csv_path = Path(args.out_csv).expanduser()
    else:
        out_csv_path = out_dir / DEFAULT_OUTPUT_CSV_NAME

    # Priority: CLI --pseudoprofiles (non-empty) > DEFAULT_PSEUDOPROFILES_TO_PROCESS (non-empty) > all
    if args.pseudoprofiles is not None and len(args.pseudoprofiles) > 0:
        pseudoprofiles_to_process: List[str] = args.pseudoprofiles
    elif DEFAULT_PSEUDOPROFILES_TO_PROCESS:
        pseudoprofiles_to_process = list(DEFAULT_PSEUDOPROFILES_TO_PROCESS)
    else:
        pseudoprofiles_to_process = []

    run_all_pseudoprofiles_single_csv(
        mapped_criterions_csv=args.mapped_criterions_csv,
        cluster_json_path=args.cluster_json,
        out_csv_path=out_csv_path,
        top_k_aggregate=args.top_k_aggregate,
        top_k_per_criterion=args.top_k_per_criterion,
        pseudoprofiles_to_process=pseudoprofiles_to_process,
        verbose=(not args.quiet),
    )


if __name__ == "__main__":
    main()
