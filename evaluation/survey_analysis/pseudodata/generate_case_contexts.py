"""Generate pseudo case contexts for prompt and pipeline development."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from analysis.shared.survey_paths import PSEUDODATA_DIR
from .generate_hcp_outputs import generate_hcp_outputs
from .generate_phoenix_outputs import OPTIMISED_PART2_LABELS, priority_order_for_case


def _load_hcp_bundle() -> Dict[str, Dict[str, Any]]:
    path = generate_hcp_outputs()
    return json.loads(path.read_text(encoding="utf-8"))


def _context_from_case(case_id: str, parts: Dict[str, Any]) -> Dict[str, Any]:
    symptoms = [item.get("label", "") for item in parts.get("part1", {}).get("items", [])]
    options = [
        {"option_id": f"BO-{idx}", "label": item.get("label", "")}
        for idx, item in enumerate(parts.get("part2", {}).get("items", []), start=1)
    ]
    options = _ensure_five_options(case_id, options)
    targets = [opt["label"] for opt in options[:3]]
    network_summary, ema_summary = _build_pseudo_network_context(
        case_id=case_id,
        symptoms=symptoms,
        options=options,
    )
    candidate_items = []
    for target in targets:
        if target:
            candidate_items.extend([
                f"{target} completed today (yes/no)",
                f"{target} duration today (minutes)",
                f"{target} difficulty today (0..10)",
            ])
    while len(candidate_items) < 20:
        candidate_items.append(f"Generic candidate EMA item {len(candidate_items) + 1}")

    return {
        "vignette": f"Pseudo vignette for {case_id}; use production case text in real mode.",
        "case_notes": {"case_id": case_id, "source": "pseudodata"},
        "standardized_symptoms": symptoms,
        "standardized_treatment_options": options,
        "network_summary": network_summary,
        "ema_summary": ema_summary,
        "treatment_targets": targets,
        "candidate_ema_items": candidate_items[:20],
        "primary_problem": "; ".join(symptoms[:3]),
        "treatment_goal": targets[0] if targets else "",
        "barrier": "Pseudo barrier; replace with production barrier context.",
        "coping_strategy": "Pseudo coping strategy; replace with production HAPA/coping output.",
        "hapa_phase": "intentional",
    }


def _ensure_five_options(
    case_id: str,
    options: list[Dict[str, str]],
) -> list[Dict[str, str]]:
    """Guarantee the five fixed Part 3 option IDs expected by the survey."""
    out = [dict(option) for option in options[:5]]
    optimised = OPTIMISED_PART2_LABELS.get(case_id, [])
    while len(out) < 5:
        idx = len(out) + 1
        fallback_label = (
            optimised[idx - 1]
            if idx - 1 < len(optimised)
            else f"Supplementary modifiable option {idx}"
        )
        out.append({"option_id": f"BO-{idx}", "label": fallback_label})
    return out


def _build_pseudo_network_context(
    *,
    case_id: str,
    symptoms: list[str],
    options: list[Dict[str, str]],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Build a transparent pseudo network with enough structure for Part 3.

    The real survey uses a rendered network figure. In pseudo validation we
    provide the equivalent numeric information directly: edge weights,
    symptom burden, and option-level priority scores. This prevents the judge
    from flooring Part 3 simply because no network evidence was available.
    """
    priority_order = priority_order_for_case(case_id)
    rank_pos = {option_id: idx for idx, option_id in enumerate(priority_order)}
    symptom_ids = [f"CR-{idx}" for idx, _ in enumerate(symptoms, start=1)]

    burden_rows = []
    for idx, symptom in enumerate(symptoms, start=1):
        burden_rows.append({
            "symptom_id": f"CR-{idx}",
            "label": symptom,
            "mean_burden_0_10": round(8.4 - 0.45 * min(idx - 1, 6), 2),
            "frequency_pct": int(max(42, 91 - 7 * (idx - 1))),
            "trend": "worsening" if idx <= 2 else "stable" if idx <= 4 else "improving",
        })

    edges = []
    option_table = []
    for option in options:
        option_id = str(option.get("option_id", ""))
        priority_idx = rank_pos.get(option_id, len(options))
        base_strength = max(0.22, 0.92 - 0.14 * priority_idx)
        connected_symptom_ids = symptom_ids[: min(3, len(symptom_ids))]
        total_abs_weight = 0.0
        burden_weighted_score = 0.0
        for edge_idx, symptom_id in enumerate(connected_symptom_ids):
            symptom_label = symptoms[edge_idx]
            weight = -round(base_strength - 0.08 * edge_idx, 2)
            burden = next(
                row["mean_burden_0_10"]
                for row in burden_rows
                if row["symptom_id"] == symptom_id
            )
            total_abs_weight += abs(weight)
            burden_weighted_score += abs(weight) * float(burden)
            edges.append({
                "from_option_id": option_id,
                "from_label": option.get("label", ""),
                "to_symptom_id": symptom_id,
                "to_label": symptom_label,
                "weight": weight,
                "direction": "protective",
                "interpretation": (
                    "Increasing this option is expected to reduce the connected symptom; "
                    "larger absolute weights imply higher clinical leverage."
                ),
            })
        option_table.append({
            "option_id": option_id,
            "label": option.get("label", ""),
            "total_absolute_edge_weight": round(total_abs_weight, 2),
            "burden_weighted_priority_score": round(burden_weighted_score, 2),
            "priority_signal": "highest" if priority_idx == 0 else "high" if priority_idx <= 2 else "lower",
        })

    option_table.sort(
        key=lambda row: (
            -float(row["burden_weighted_priority_score"]),
            str(row["option_id"]),
        )
    )
    network_summary = {
        "note": (
            "Pseudo numeric equivalent of the Part 3 network figure. Negative "
            "protective weights mean that increasing the option is expected to "
            "reduce connected symptoms; priority follows stronger absolute "
            "weights to higher-burden symptoms."
        ),
        "symptoms": burden_rows,
        "option_priority_table": option_table,
        "edges": edges,
    }
    ema_summary = {
        "window_days": 21,
        "symptom_burden": burden_rows,
        "option_engagement": [
            {
                "option_id": row["option_id"],
                "current_engagement": "low" if idx <= 2 else "moderate",
                "trend": "insufficient improvement" if idx <= 2 else "mixed",
            }
            for idx, row in enumerate(option_table, start=1)
        ],
    }
    return network_summary, ema_summary


def generate_case_contexts(out_path: Path | None = None) -> Path:
    """Write pseudo case contexts to disk."""
    out_path = Path(out_path) if out_path else PSEUDODATA_DIR / "case_contexts.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = _load_hcp_bundle()
    contexts = {case_id: _context_from_case(case_id, parts) for case_id, parts in bundle.items()}
    out_path.write_text(
        json.dumps(contexts, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_path


if __name__ == "__main__":
    p = generate_case_contexts()
    print(f"Wrote pseudo case contexts to {p}")
