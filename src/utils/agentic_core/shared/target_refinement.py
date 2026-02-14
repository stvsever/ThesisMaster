from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def normalize_path_text(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    raw = raw.strip("'\"")
    raw = raw.replace(">", "/")
    raw = raw.replace("\\", "/")
    raw = raw.replace("[", "").replace("]", "")
    raw = re.sub(r"\s+", " ", raw)
    raw = re.sub(r"\s*/\s*", " / ", raw)
    raw = raw.replace("*", "")
    parts = [segment.strip() for segment in raw.split("/") if segment.strip()]
    return " / ".join(parts)


def path_segments(text: str) -> List[str]:
    normalized = normalize_path_text(text)
    if not normalized:
        return []
    return [segment.strip() for segment in normalized.split(" / ") if segment.strip()]


def _path_tokens(text: str) -> List[str]:
    normalized = normalize_path_text(text).lower()
    if not normalized:
        return []
    tokens = re.findall(r"[a-z0-9_]+", normalized.replace("/", " "))
    return [token for token in tokens if len(token) > 1]


def path_similarity(path_a: str, path_b: str) -> float:
    seg_a = [item.lower() for item in path_segments(path_a)]
    seg_b = [item.lower() for item in path_segments(path_b)]
    if not seg_a or not seg_b:
        return 0.0

    prefix = 0
    for left, right in zip(seg_a, seg_b):
        if left == right:
            prefix += 1
        else:
            break
    prefix_score = float(prefix / max(len(seg_a), len(seg_b)))

    tok_a = set(_path_tokens(path_a))
    tok_b = set(_path_tokens(path_b))
    token_score = 0.0
    if tok_a and tok_b:
        token_score = float(len(tok_a.intersection(tok_b)) / len(tok_a.union(tok_b)))

    subseq = 0.0
    if _is_subsequence(seg_a, seg_b) or _is_subsequence(seg_b, seg_a):
        subseq = 1.0

    return _clamp01(0.45 * token_score + 0.35 * prefix_score + 0.20 * subseq)


def _is_subsequence(shorter: Sequence[str], longer: Sequence[str]) -> bool:
    if not shorter:
        return False
    if len(shorter) > len(longer):
        return False
    i = 0
    for item in longer:
        if shorter[i] == item:
            i += 1
            if i == len(shorter):
                return True
    return False


@dataclass
class PredictorLeafPath:
    full_path: str
    segments: Tuple[str, ...]
    root_domain: str
    secondary_domain: str
    leaf_label: str
    parent_key: str


def load_predictor_leaf_paths(json_path: Path) -> List[PredictorLeafPath]:
    if not json_path.exists():
        return []
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    out: List[PredictorLeafPath] = []
    for raw_path in payload:
        normalized = normalize_path_text(str(raw_path))
        segments = path_segments(normalized)
        if len(segments) < 2:
            continue
        root = segments[0]
        secondary = segments[1] if len(segments) > 1 else root
        leaf = segments[-1]
        parent = " / ".join(segments[:3]) if len(segments) >= 3 else " / ".join(segments[:2])
        out.append(
            PredictorLeafPath(
                full_path=normalized,
                segments=tuple(item.lower() for item in segments),
                root_domain=root,
                secondary_domain=secondary,
                leaf_label=leaf,
                parent_key=parent.lower(),
            )
        )
    return out


def discover_latest_hyde_dense_profiles(runs_root: Path) -> Optional[Path]:
    if not runs_root.exists():
        return None
    candidates = sorted(
        [path for path in runs_root.glob("*/dense_profiles.csv") if path.exists()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_profile_mapping_rows(mapping_csv: Path, profile_id: str) -> List[Dict[str, Any]]:
    if not mapping_csv.exists():
        return []
    df = pd.read_csv(mapping_csv)
    required = {"pseudoprofile_id", "part", "criterion_path", "predictor_path", "relevance_score"}
    if not required.issubset(set(df.columns)):
        return []
    subset = df[df["pseudoprofile_id"].astype(str) == str(profile_id)].copy()
    if subset.empty:
        return []
    rows: List[Dict[str, Any]] = []
    for _, row in subset.iterrows():
        predictor_path = normalize_path_text(str(row.get("predictor_path") or ""))
        if not predictor_path:
            continue
        score_raw = pd.to_numeric(pd.Series([row.get("relevance_score")]), errors="coerce").iloc[0]
        if pd.isna(score_raw):
            continue
        score_0_1 = _clamp01(float(score_raw) / 1000.0 if float(score_raw) > 1.0 else float(score_raw))
        rows.append(
            {
                "part": str(row.get("part") or "").strip(),
                "criterion_path": normalize_path_text(str(row.get("criterion_path") or "")),
                "criterion_leaf": normalize_path_text(str(row.get("criterion_leaf") or "")),
                "predictor_path": predictor_path,
                "relevance_score_0_1": score_0_1,
            }
        )
    return rows


def load_profile_hyde_scores(hyde_dense_profiles_csv: Path, profile_id: str, top_k: int = 250) -> Dict[str, float]:
    if not hyde_dense_profiles_csv.exists():
        return {}
    df = pd.read_csv(hyde_dense_profiles_csv)
    if "pseudoprofile_id" not in df.columns:
        return {}
    sub = df[df["pseudoprofile_id"].astype(str) == str(profile_id)].copy()
    if sub.empty:
        return {}
    row = sub.iloc[0]
    out: Dict[str, float] = {}
    rank_cols = [column for column in df.columns if re.match(r"^global_\d{3}_path_full$", str(column))]
    rank_cols = sorted(rank_cols)[: max(0, int(top_k))]
    for path_col in rank_cols:
        score_col = path_col.replace("_path_full", "_score")
        path_value = normalize_path_text(str(row.get(path_col) or ""))
        if not path_value:
            continue
        score_raw = pd.to_numeric(pd.Series([row.get(score_col)]), errors="coerce").iloc[0]
        if pd.isna(score_raw):
            continue
        out[path_value.lower()] = max(out.get(path_value.lower(), 0.0), _clamp01(float(score_raw)))
    return out


def _domain_key_from_segments(segments: Sequence[str]) -> str:
    if not segments:
        return "unknown"
    if len(segments) == 1:
        return str(segments[0]).lower()
    return f"{segments[0].lower()} / {segments[1].lower()}"


def _mapping_score_for_leaf(
    leaf: PredictorLeafPath,
    mapping_rows: Sequence[Dict[str, Any]],
) -> Tuple[float, str]:
    best_score = 0.0
    source = ""
    leaf_segments = leaf.segments
    for row in mapping_rows:
        predictor_path = str(row.get("predictor_path") or "")
        score = float(row.get("relevance_score_0_1", 0.0))
        row_segments = tuple(item.lower() for item in path_segments(predictor_path))
        if not row_segments:
            continue
        matched = False
        if row_segments[0] == leaf_segments[0] and _is_subsequence(row_segments, leaf_segments):
            matched = True
            candidate = score
        else:
            sim = path_similarity(leaf.full_path, predictor_path)
            candidate = score * sim
            matched = sim >= 0.45
        if matched and candidate > best_score:
            best_score = candidate
            source = predictor_path
    return _clamp01(best_score), source


def build_bfs_candidates(
    *,
    leaf_paths: Sequence[PredictorLeafPath],
    mapping_rows: Sequence[Dict[str, Any]],
    hyde_scores: Dict[str, float],
    mapped_predictor_paths: Sequence[str],
    impact_by_predictor: Dict[str, float],
    predictor_var_to_path: Dict[str, str],
    max_candidates: int,
) -> List[Dict[str, Any]]:
    if not leaf_paths:
        return []

    mapped_paths_norm = [normalize_path_text(path) for path in mapped_predictor_paths if str(path).strip()]
    mapped_paths_norm = [path for path in mapped_paths_norm if path]

    path_impact: Dict[str, float] = {}
    for predictor_id, impact in impact_by_predictor.items():
        predictor_path = normalize_path_text(str(predictor_var_to_path.get(str(predictor_id), "")))
        if predictor_path:
            path_impact[predictor_path] = max(path_impact.get(predictor_path, 0.0), _clamp01(float(impact)))

    anchor_domains: set[str] = set()
    for path in mapped_paths_norm:
        segments = path_segments(path)
        anchor_domains.add(_domain_key_from_segments(segments))
    for row in mapping_rows:
        segments = path_segments(str(row.get("predictor_path") or ""))
        if segments:
            anchor_domains.add(_domain_key_from_segments(segments))

    scored: List[Dict[str, Any]] = []
    for leaf in leaf_paths:
        mapping_score, mapping_source = _mapping_score_for_leaf(leaf, mapping_rows)
        hyde_score = float(hyde_scores.get(leaf.full_path.lower(), 0.0))
        idiographic_anchor = 0.0
        for anchor_path, anchor_impact in path_impact.items():
            sim = path_similarity(leaf.full_path, anchor_path)
            idiographic_anchor = max(idiographic_anchor, _clamp01(sim * anchor_impact))
        domain_key = _domain_key_from_segments(leaf.segments)
        domain_bonus = 1.0 if domain_key in anchor_domains else 0.0
        total_score = _clamp01(
            0.45 * mapping_score
            + 0.25 * hyde_score
            + 0.20 * idiographic_anchor
            + 0.10 * domain_bonus
        )

        if total_score <= 0.015 and mapping_score <= 0.0 and hyde_score <= 0.0 and idiographic_anchor <= 0.0:
            continue

        scored.append(
            {
                "predictor_path": leaf.full_path,
                "predictor_id": "",
                "root_domain": leaf.root_domain,
                "secondary_node": leaf.secondary_domain,
                "leaf_label": leaf.leaf_label,
                "subtree_relevance_score_0_1": round(float(total_score), 6),
                "mapping_score_0_1": round(float(mapping_score), 6),
                "hyde_score_0_1": round(float(hyde_score), 6),
                "idiographic_anchor_score_0_1": round(float(idiographic_anchor), 6),
                "bfs_domain_key": domain_key,
                "mapping_anchor_path": mapping_source,
            }
        )

    if not scored:
        return []

    by_domain: Dict[str, List[Dict[str, Any]]] = {}
    for row in scored:
        by_domain.setdefault(str(row["bfs_domain_key"]), []).append(row)
    for domain_rows in by_domain.values():
        domain_rows.sort(key=lambda item: float(item["subtree_relevance_score_0_1"]), reverse=True)

    domain_priority = sorted(
        by_domain.keys(),
        key=lambda key: float(by_domain[key][0]["subtree_relevance_score_0_1"]),
        reverse=True,
    )

    selected: List[Dict[str, Any]] = []
    seen_paths: set[str] = set()
    target_n = max(1, int(max_candidates))

    for domain in domain_priority:
        if len(selected) >= target_n:
            break
        first = by_domain[domain][0]
        candidate_path = str(first["predictor_path"])
        if candidate_path in seen_paths:
            continue
        first = dict(first)
        first["bfs_stage"] = "breadth_domain_coverage"
        selected.append(first)
        seen_paths.add(candidate_path)

    breadth_limit = min(target_n, max(len(domain_priority) * 3, len(domain_priority)))
    domain_positions = {domain: 1 for domain in domain_priority}
    while len(selected) < breadth_limit:
        made_progress = False
        for domain in domain_priority:
            idx = domain_positions[domain]
            rows = by_domain[domain]
            if idx >= len(rows):
                continue
            row = dict(rows[idx])
            domain_positions[domain] = idx + 1
            candidate_path = str(row["predictor_path"])
            if candidate_path in seen_paths:
                continue
            row["bfs_stage"] = "breadth_round_robin"
            selected.append(row)
            seen_paths.add(candidate_path)
            made_progress = True
            if len(selected) >= breadth_limit:
                break
        if not made_progress:
            break

    remaining = sorted(scored, key=lambda item: float(item["subtree_relevance_score_0_1"]), reverse=True)
    for row in remaining:
        if len(selected) >= target_n:
            break
        candidate_path = str(row["predictor_path"])
        if candidate_path in seen_paths:
            continue
        row_copy = dict(row)
        row_copy["bfs_stage"] = "depth_refinement"
        selected.append(row_copy)
        seen_paths.add(candidate_path)

    for rank, row in enumerate(selected, start=1):
        row["bfs_rank"] = rank

    return selected


def load_impact_matrix(impact_matrix_path: Path) -> pd.DataFrame:
    if not impact_matrix_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(impact_matrix_path, index_col=0)
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.fillna(0.0)
    return df


def _parse_pc_dense_relevance(initial_model_payload: Dict[str, Any]) -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    for row in initial_model_payload.get("predictor_criterion_relevance", []) or []:
        predictor_id = str(row.get("predictor_var_id") or "").strip()
        criterion_id = str(row.get("criterion_var_id") or "").strip()
        if not predictor_id or not criterion_id:
            continue
        raw = str(row.get("relevance_score_0_1_comma5") or "").strip().replace(",", ".")
        try:
            value = float(raw)
        except Exception:
            continue
        out[(predictor_id, criterion_id)] = _clamp01(value)
    for row in initial_model_payload.get("edges", []) or []:
        predictor_id = str(row.get("from_predictor_var_id") or row.get("source_var_id") or "").strip()
        criterion_id = str(row.get("to_criterion_var_id") or row.get("target_var_id") or "").strip()
        if not predictor_id or not criterion_id:
            continue
        value = row.get("estimated_relevance_0_1")
        if not isinstance(value, (int, float)):
            continue
        out[(predictor_id, criterion_id)] = max(out.get((predictor_id, criterion_id), 0.0), _clamp01(float(value)))
    return out


def _criterion_weight_vector(criteria: Sequence[Dict[str, Any]], impact_matrix: pd.DataFrame) -> Dict[str, float]:
    if impact_matrix.empty:
        weight = 1.0 / max(1, len(criteria))
        return {str(item.get("var_id")): weight for item in criteria}
    row_sums = impact_matrix.sum(axis=1)
    if float(row_sums.sum()) <= 1e-12:
        weight = 1.0 / max(1, len(criteria))
        return {str(item.get("var_id")): weight for item in criteria}
    out: Dict[str, float] = {}
    for item in criteria:
        criterion_id = str(item.get("var_id") or "")
        if not criterion_id:
            continue
        out[criterion_id] = float(row_sums.get(criterion_id, 0.0))
    total = float(sum(out.values()))
    if total <= 1e-12:
        weight = 1.0 / max(1, len(criteria))
        return {str(item.get("var_id")): weight for item in criteria}
    return {key: float(value / total) for key, value in out.items()}


def fuse_updated_model_matrix(
    *,
    criteria_summary: Sequence[Dict[str, Any]],
    predictor_summary: Sequence[Dict[str, Any]],
    initial_model_payload: Dict[str, Any],
    impact_matrix: pd.DataFrame,
    candidate_paths: Sequence[str],
    candidate_prior_scores: Dict[str, float],
    mapping_rows: Sequence[Dict[str, Any]],
    readiness_score_0_100: Optional[float],
    previous_cycle_scores: Optional[Dict[str, float]] = None,
    max_predictors: int = 200,
) -> Dict[str, Any]:
    if not candidate_paths:
        return {
            "weights": {"nomothetic_weight": 0.7, "idiographic_weight": 0.3, "readiness_0_1": 0.0},
            "criterion_order": [],
            "predictor_order": [],
            "predictor_rankings": [],
            "edge_rows": [],
            "matrix": [],
        }

    readiness_0_1 = _clamp01(float(readiness_score_0_100 or 0.0) / 100.0)
    idiographic_weight = _clamp01(0.30 + 0.50 * readiness_0_1)
    nomothetic_weight = _clamp01(1.0 - idiographic_weight)

    pc_dense = _parse_pc_dense_relevance(initial_model_payload)
    criterion_weights = _criterion_weight_vector(criteria_summary, impact_matrix)

    current_predictor_paths: Dict[str, str] = {}
    for item in predictor_summary:
        predictor_id = str(item.get("var_id") or "").strip()
        predictor_path = normalize_path_text(str(item.get("mapped_leaf_full_path") or item.get("label") or ""))
        if predictor_id:
            current_predictor_paths[predictor_id] = predictor_path

    normalized_candidate_paths = [normalize_path_text(path) for path in candidate_paths if str(path).strip()]
    normalized_candidate_paths = [path for path in normalized_candidate_paths if path]

    parent_groups: Dict[str, List[str]] = {}
    for path in normalized_candidate_paths:
        segments = path_segments(path)
        parent_key = " / ".join(segments[:3]).lower() if len(segments) >= 3 else " / ".join(segments[:2]).lower()
        parent_groups.setdefault(parent_key, []).append(path)

    group_shares: Dict[str, float] = {}
    for parent_key, members in parent_groups.items():
        priors = [max(1e-6, float(candidate_prior_scores.get(member, 0.0))) for member in members]
        total = float(sum(priors))
        for member, prior in zip(members, priors):
            group_shares[member] = float(prior / total) if total > 1e-12 else 1.0 / max(1, len(members))

    edge_rows: List[Dict[str, Any]] = []
    predictor_to_edges: Dict[str, List[Dict[str, Any]]] = {}

    for candidate_path in normalized_candidate_paths:
        segments = path_segments(candidate_path)
        parent_key = " / ".join(segments[:3]).lower() if len(segments) >= 3 else " / ".join(segments[:2]).lower()
        members = parent_groups.get(parent_key, [])
        group_size = max(1, len(members))
        split_share = float(group_shares.get(candidate_path, 1.0 / group_size))
        split_factor = 1.0 if group_size <= 1 else 0.5 + 0.5 * split_share * float(group_size)

        for criterion in criteria_summary:
            criterion_id = str(criterion.get("var_id") or "").strip()
            if not criterion_id:
                continue
            criterion_ref = normalize_path_text(
                str(criterion.get("mapped_leaf_full_path") or criterion.get("label") or criterion_id)
            )

            dense_num = 0.0
            dense_den = 0.0
            idi_num = 0.0
            idi_den = 0.0
            for predictor_id, predictor_path in current_predictor_paths.items():
                similarity = path_similarity(candidate_path, predictor_path)
                if similarity <= 0.0:
                    continue
                dense_value = float(pc_dense.get((predictor_id, criterion_id), 0.0))
                if dense_value > 0.0:
                    dense_num += similarity * dense_value
                    dense_den += similarity
                if not impact_matrix.empty and criterion_id in impact_matrix.index and predictor_id in impact_matrix.columns:
                    impact_value = float(impact_matrix.loc[criterion_id, predictor_id])
                    idi_num += similarity * max(0.0, impact_value)
                    idi_den += similarity

            nomothetic_dense = float(dense_num / dense_den) if dense_den > 1e-12 else 0.0
            idiographic_raw = float(idi_num / idi_den) if idi_den > 1e-12 else 0.0
            idiographic_adjusted = _clamp01(idiographic_raw * split_factor)

            mapping_candidates: List[float] = []
            for row in mapping_rows:
                predictor_path = str(row.get("predictor_path") or "")
                relevance = float(row.get("relevance_score_0_1", 0.0))
                if relevance <= 0.0:
                    continue
                pred_sim = path_similarity(candidate_path, predictor_path)
                if pred_sim <= 0.20:
                    continue
                criterion_path = str(row.get("criterion_path") or row.get("criterion_leaf") or "")
                criterion_sim = path_similarity(criterion_ref, criterion_path) if criterion_path else 0.45
                mapping_candidates.append(relevance * pred_sim * (0.35 + 0.65 * criterion_sim))
            nomothetic_mapping = max(mapping_candidates) if mapping_candidates else 0.0
            nomothetic = _clamp01(0.60 * nomothetic_mapping + 0.40 * nomothetic_dense)

            fused = _clamp01(nomothetic_weight * nomothetic + idiographic_weight * idiographic_adjusted)

            row = {
                "predictor_path": candidate_path,
                "criterion_var_id": criterion_id,
                "criterion_label": str(criterion.get("label") or criterion_id),
                "nomothetic_score_0_1": round(float(nomothetic), 6),
                "idiographic_score_0_1": round(float(idiographic_adjusted), 6),
                "fused_score_0_1": round(float(fused), 6),
                "criterion_weight_0_1": round(float(criterion_weights.get(criterion_id, 0.0)), 6),
                "idiographic_split_factor_0_1": round(float(_clamp01(split_factor)), 6),
                "nomothetic_mapping_component_0_1": round(float(_clamp01(nomothetic_mapping)), 6),
                "nomothetic_transfer_component_0_1": round(float(_clamp01(nomothetic_dense)), 6),
                "bfs_parent_key": parent_key,
            }
            edge_rows.append(row)
            predictor_to_edges.setdefault(candidate_path, []).append(row)

    if previous_cycle_scores is None:
        previous_cycle_scores = {}

    rankings: List[Dict[str, Any]] = []
    for predictor_path, rows in predictor_to_edges.items():
        if not rows:
            continue
        fused_mean = float(sum(r["fused_score_0_1"] * r["criterion_weight_0_1"] for r in rows))
        nomo_mean = float(sum(r["nomothetic_score_0_1"] * r["criterion_weight_0_1"] for r in rows))
        idio_mean = float(sum(r["idiographic_score_0_1"] * r["criterion_weight_0_1"] for r in rows))
        prior = float(candidate_prior_scores.get(predictor_path, 0.0))
        segments = path_segments(predictor_path)
        domain_key = _domain_key_from_segments(tuple(segment.lower() for segment in segments))
        rankings.append(
            {
                "predictor_path": predictor_path,
                "fused_score_0_1": round(float(_clamp01(fused_mean)), 6),
                "nomothetic_mean_0_1": round(float(_clamp01(nomo_mean)), 6),
                "idiographic_mean_0_1": round(float(_clamp01(idio_mean)), 6),
                "prior_score_0_1": round(float(_clamp01(prior)), 6),
                "domain_key": domain_key,
                "root_domain": segments[0] if segments else "",
                "secondary_node": segments[1] if len(segments) > 1 else "",
            }
        )

    for row in rankings:
        predictor_key = str(row["predictor_path"]).lower()
        current_impact = float(row.get("idiographic_mean_0_1", 0.0))
        previous_impact = float(previous_cycle_scores.get(predictor_key, 0.0))
        ontology_mapping = float(row.get("nomothetic_mean_0_1", 0.0))
        sign_stable = 1.0 if previous_impact > 0.0 and current_impact > 0.0 else 0.0
        magnitude_stable = 1.0 - min(1.0, abs(current_impact - previous_impact))
        stability_bonus = _clamp01(0.6 * sign_stable + 0.4 * magnitude_stable)
        fused_memory = _clamp01(
            (0.45 * current_impact)
            + (0.25 * previous_impact)
            + (0.20 * ontology_mapping)
            + (0.10 * stability_bonus)
        )
        contradiction = previous_impact > 0.60 and current_impact < 0.20
        if contradiction:
            fused_memory = _clamp01(min(fused_memory, 0.40 * previous_impact + 0.60 * current_impact))
        row["fused_memory_score_0_1"] = round(float(fused_memory), 6)
        row["previous_cycle_impact_0_1"] = round(float(_clamp01(previous_impact)), 6)
        row["stability_bonus_0_1"] = round(float(stability_bonus), 6)
        row["contradiction_decay_applied"] = bool(contradiction)
        row["fused_score_0_1"] = round(float(_clamp01((0.65 * float(row["fused_score_0_1"])) + (0.35 * fused_memory))), 6)

    rankings = sorted(rankings, key=lambda row: float(row["fused_score_0_1"]), reverse=True)
    rankings = rankings[: max(1, int(max_predictors))]
    keep_paths = {str(item["predictor_path"]) for item in rankings}
    edge_rows = [row for row in edge_rows if str(row["predictor_path"]) in keep_paths]

    criterion_order = [str(item.get("var_id") or "") for item in criteria_summary if str(item.get("var_id") or "").strip()]
    predictor_order = [str(item["predictor_path"]) for item in rankings]

    matrix: List[List[float]] = []
    edge_lookup = {
        (str(row["criterion_var_id"]), str(row["predictor_path"])): float(row["fused_score_0_1"])
        for row in edge_rows
    }
    for criterion_id in criterion_order:
        matrix.append(
            [float(edge_lookup.get((criterion_id, predictor_path), 0.0)) for predictor_path in predictor_order]
        )

    return {
        "weights": {
            "nomothetic_weight": round(float(nomothetic_weight), 6),
            "idiographic_weight": round(float(idiographic_weight), 6),
            "readiness_0_1": round(float(readiness_0_1), 6),
        },
        "criterion_order": criterion_order,
        "predictor_order": predictor_order,
        "predictor_rankings": rankings,
        "edge_rows": edge_rows,
        "matrix": matrix,
    }
