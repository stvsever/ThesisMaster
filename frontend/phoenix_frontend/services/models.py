from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class SessionRecord:
    session_id: str
    profile_id: str
    created_at: str
    updated_at: str
    complaint_text: str
    person_text: str
    context_text: str
    current_cycle: int = 0
    initial_model_run_id: str = ""
    initial_model_run_dir: str = ""
    latest_model_json: str = ""
    latest_model_mapped_json: str = ""
    pseudodata_ready: bool = False
    pseudodata_root: str = ""
    pipeline_run_id: str = ""
    latest_pipeline_cycle_root: str = ""
    latest_pipeline_summary: str = ""
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SessionRecord":
        return cls(
            session_id=str(payload.get("session_id") or ""),
            profile_id=str(payload.get("profile_id") or ""),
            created_at=str(payload.get("created_at") or now_iso()),
            updated_at=str(payload.get("updated_at") or now_iso()),
            complaint_text=str(payload.get("complaint_text") or ""),
            person_text=str(payload.get("person_text") or ""),
            context_text=str(payload.get("context_text") or ""),
            current_cycle=int(payload.get("current_cycle") or 0),
            initial_model_run_id=str(payload.get("initial_model_run_id") or ""),
            initial_model_run_dir=str(payload.get("initial_model_run_dir") or ""),
            latest_model_json=str(payload.get("latest_model_json") or ""),
            latest_model_mapped_json=str(payload.get("latest_model_mapped_json") or ""),
            pseudodata_ready=bool(payload.get("pseudodata_ready") or False),
            pseudodata_root=str(payload.get("pseudodata_root") or ""),
            pipeline_run_id=str(payload.get("pipeline_run_id") or ""),
            latest_pipeline_cycle_root=str(payload.get("latest_pipeline_cycle_root") or ""),
            latest_pipeline_summary=str(payload.get("latest_pipeline_summary") or ""),
            notes=dict(payload.get("notes") or {}),
        )


@dataclass
class JobLogEntry:
    index: int
    timestamp: str
    line: str
    level: str = "INFO"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": int(self.index),
            "timestamp": str(self.timestamp),
            "line": str(self.line),
            "level": str(self.level),
        }


@dataclass
class JobRecord:
    job_id: str
    session_id: str
    kind: str
    status: str
    created_at: str
    started_at: str = ""
    finished_at: str = ""
    result: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    logs: List[JobLogEntry] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "session_id": self.session_id,
            "kind": self.kind,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result": self.result,
            "error": self.error,
            "log_count": len(self.logs),
        }
