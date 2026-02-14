#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from utils import append_profile_event, build_cycle_summary


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a cycle-level updated-model summary from Step-03/04/05 artifacts."
    )
    parser.add_argument("--profile-id", type=str, required=True)
    parser.add_argument("--cycle-index", type=int, default=1)
    parser.add_argument("--step03-json", type=str, required=True)
    parser.add_argument("--step04-json", type=str, required=True)
    parser.add_argument("--step05-json", type=str, default="")
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--history-jsonl", type=str, default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    step03_payload = _read_json(Path(args.step03_json).expanduser().resolve())
    step04_payload = _read_json(Path(args.step04_json).expanduser().resolve())
    step05_payload = (
        _read_json(Path(args.step05_json).expanduser().resolve())
        if str(args.step05_json).strip()
        else {}
    )

    summary = build_cycle_summary(
        profile_id=str(args.profile_id),
        cycle_index=int(args.cycle_index),
        step03_payload=step03_payload,
        step04_payload=step04_payload,
        step05_payload=step05_payload,
    )

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if str(args.history_jsonl).strip():
        append_profile_event(
            history_jsonl_path=Path(args.history_jsonl).expanduser().resolve(),
            profile_id=str(args.profile_id),
            cycle_index=int(args.cycle_index),
            event="updated_model_cycle_summary_created",
            payload={
                "output_json": str(output_path),
                "updated_predictor_count": summary.get("step04", {}).get("updated_predictor_count", 0),
            },
        )

    print(f"[OK] Wrote cycle summary: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
