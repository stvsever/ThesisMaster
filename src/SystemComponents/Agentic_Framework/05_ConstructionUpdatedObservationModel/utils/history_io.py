from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def append_profile_event(
    *,
    history_jsonl_path: Path,
    profile_id: str,
    cycle_index: int,
    event: str,
    payload: Dict[str, Any],
) -> None:
    history_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp_local": datetime.now().isoformat(timespec="seconds"),
        "profile_id": str(profile_id),
        "cycle_index": int(cycle_index),
        "event": str(event),
        "payload": payload,
    }
    with history_jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
