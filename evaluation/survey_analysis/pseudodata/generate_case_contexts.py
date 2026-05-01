"""Generate pseudo case contexts for prompt and pipeline development."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from analysis.shared.survey_paths import PSEUDODATA_DIR
from .generate_hcp_outputs import generate_hcp_outputs


def _load_hcp_bundle() -> Dict[str, Dict[str, Any]]:
    path = generate_hcp_outputs()
    return json.loads(path.read_text(encoding="utf-8"))


def _context_from_case(case_id: str, parts: Dict[str, Any]) -> Dict[str, Any]:
    symptoms = [item.get("label", "") for item in parts.get("part1", {}).get("items", [])]
    options = [
        {"option_id": f"BO-{idx}", "label": item.get("label", "")}
        for idx, item in enumerate(parts.get("part2", {}).get("items", []), start=1)
    ]
    targets = [opt["label"] for opt in options[:3]]
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
        "network_summary": {
            "note": "Pseudo network; production runs should include edge direction and weight.",
            "edges": [],
        },
        "ema_summary": {
            "note": "Pseudo 21-day EMA summary; production runs should include frequency, burden, and trend.",
        },
        "treatment_targets": targets,
        "candidate_ema_items": candidate_items[:20],
        "primary_problem": "; ".join(symptoms[:3]),
        "treatment_goal": targets[0] if targets else "",
        "barrier": "Pseudo barrier; replace with production barrier context.",
        "coping_strategy": "Pseudo coping strategy; replace with production HAPA/coping output.",
        "hapa_phase": "intentional",
    }


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
