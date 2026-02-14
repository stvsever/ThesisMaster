#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    contracts_root = repo_root / "src/utils/agentic_core/shared/contracts"
    schemas_root = contracts_root / "schemas"
    required = [
        "readiness_report",
        "network_comparison_summary",
        "momentary_impact",
        "step03_target_selection",
        "step04_updated_model",
        "step05_hapa_intervention",
        "pipeline_summary",
    ]

    sys.path.insert(0, str((repo_root / "src/utils/agentic_core").resolve()))
    from shared import ContractValidator  # type: ignore

    validator = ContractValidator()
    for name in required:
        schema_path = schemas_root / f"{name}.schema.json"
        if not schema_path.exists():
            print(f"[ERROR] Missing schema: {schema_path}")
            return 2
        json.loads(schema_path.read_text(encoding="utf-8"))
        payload = {"contract_version": "1.0.0"}
        if name == "readiness_report":
            payload["overall"] = {}
        if name == "step03_target_selection":
            payload.update({"profile_id": "p1", "ranked_predictors": []})
        if name == "step04_updated_model":
            payload.update({"profile_id": "p1", "recommended_next_observation_predictors": []})
        if name == "step05_hapa_intervention":
            payload.update({"profile_id": "p1", "intervention_title": "x", "selected_treatment_targets": []})
        if name == "pipeline_summary":
            payload.update({"status": "ok", "run_id": "r1", "run_root": "/tmp/r1"})
        result = validator.validate_payload(contract_name=name, payload=payload)
        if not result.success:
            print(f"[ERROR] Validation failed for {name}: {result.errors}")
            return 3
    print("[OK] Contract schemas validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
