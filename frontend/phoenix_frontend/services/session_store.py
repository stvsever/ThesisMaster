from __future__ import annotations

import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from .models import SessionRecord, now_iso


PROFILE_ID_RE = re.compile(r"ID(\d{3})$")


class SessionStore:
    def __init__(self, sessions_root: Path) -> None:
        self.sessions_root = Path(sessions_root).expanduser().resolve()
        self.sessions_root.mkdir(parents=True, exist_ok=True)

    def _session_file(self, session_id: str) -> Path:
        return self.sessions_root / session_id / "session.json"

    def session_paths(self, session_id: str) -> Dict[str, Path]:
        session = self.load_session(session_id)
        profile_id = session.profile_id
        session_root = self.sessions_root / session_id
        inputs_root = session_root / "inputs"
        free_text_root = inputs_root / "free_text"
        outputs_root = session_root / "outputs"
        operationalization_root = outputs_root / "operationalization"
        initial_model_root = outputs_root / "initial_model"
        pseudodata_root = outputs_root / "pseudodata"
        pseudodata_profile_root = pseudodata_root / profile_id
        pipeline_root = outputs_root / "pipeline"
        history_root = outputs_root / "history"
        frontend_logs_root = outputs_root / "frontend_logs"
        paths = {
            "session_root": session_root,
            "inputs_root": inputs_root,
            "free_text_root": free_text_root,
            "outputs_root": outputs_root,
            "operationalization_root": operationalization_root,
            "initial_model_root": initial_model_root,
            "initial_model_runs_root": initial_model_root / "runs",
            "pseudodata_root": pseudodata_root,
            "pseudodata_profile_root": pseudodata_profile_root,
            "pipeline_root": pipeline_root,
            "history_root": history_root,
            "frontend_logs_root": frontend_logs_root,
        }
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        return paths

    def list_sessions(self, limit: int = 20) -> List[SessionRecord]:
        rows: List[SessionRecord] = []
        for child in sorted(self.sessions_root.iterdir(), key=lambda p: p.name, reverse=True):
            if not child.is_dir():
                continue
            file_path = child / "session.json"
            if not file_path.exists():
                continue
            try:
                payload = json.loads(file_path.read_text(encoding="utf-8"))
                rows.append(SessionRecord.from_dict(payload))
            except Exception:
                continue
            if len(rows) >= max(1, int(limit)):
                break
        return rows

    def create_session(
        self,
        *,
        complaint_text: str,
        person_text: str = "",
        context_text: str = "",
        profile_id: str = "",
    ) -> SessionRecord:
        session_id = f"s_{now_iso().replace(':', '').replace('-', '').replace('T', '_')}_{uuid.uuid4().hex[:8]}"
        profile = profile_id.strip() or self._profile_id_for_session(session_id)
        record = SessionRecord(
            session_id=session_id,
            profile_id=profile,
            created_at=now_iso(),
            updated_at=now_iso(),
            complaint_text=complaint_text.strip(),
            person_text=person_text.strip(),
            context_text=context_text.strip(),
        )
        session_root = self.sessions_root / session_id
        session_root.mkdir(parents=True, exist_ok=True)
        self.save_session(record)
        self.session_paths(session_id)
        self.write_free_text_files(record)
        return record

    def load_session(self, session_id: str) -> SessionRecord:
        file_path = self._session_file(session_id)
        if not file_path.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        return SessionRecord.from_dict(payload)

    def save_session(self, record: SessionRecord) -> None:
        file_path = self._session_file(record.session_id)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        record.updated_at = now_iso()
        file_path.write_text(json.dumps(record.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def update_session(self, session_id: str, **updates: object) -> SessionRecord:
        record = self.load_session(session_id)
        for key, value in updates.items():
            if hasattr(record, key):
                setattr(record, key, value)  # type: ignore[arg-type]
            else:
                record.notes[str(key)] = value
        self.save_session(record)
        return record

    def write_free_text_files(self, record: SessionRecord) -> Dict[str, Path]:
        paths = self.session_paths(record.session_id)
        free_text_root = paths["free_text_root"]
        profile_number = self._profile_number(record.profile_id)
        complaint_key = f"pseudoprofile_FTC_ID{profile_number}"
        person_key = f"pseudoprofile_person_ID{profile_number}"
        context_key = f"pseudoprofile_context_ID{profile_number}"

        complaints_path = free_text_root / "free_text_complaints.txt"
        person_path = free_text_root / "free_text_person.txt"
        context_path = free_text_root / "free_text_context.txt"

        complaints_path.write_text(
            self._format_free_text_block(complaint_key, record.complaint_text),
            encoding="utf-8",
        )
        person_path.write_text(
            self._format_free_text_block(person_key, record.person_text),
            encoding="utf-8",
        )
        context_path.write_text(
            self._format_free_text_block(context_key, record.context_text),
            encoding="utf-8",
        )

        return {
            "free_text_complaints": complaints_path,
            "free_text_person": person_path,
            "free_text_context": context_path,
        }

    def resolve_session_file(self, session_id: str, rel_path: str) -> Path:
        session_root = self.sessions_root / session_id
        target = (session_root / rel_path).resolve()
        if not str(target).startswith(str(session_root.resolve())):
            raise ValueError("Requested path resolves outside of session root.")
        return target

    def _profile_id_for_session(self, session_id: str) -> str:
        digest = hashlib.sha1(session_id.encode("utf-8")).hexdigest()  # noqa: S324
        idx = (int(digest[:8], 16) % 900) + 100
        return f"pseudoprofile_FTC_ID{idx:03d}"

    def _profile_number(self, profile_id: str) -> str:
        match = PROFILE_ID_RE.search(profile_id)
        if match:
            return match.group(1)
        digits = re.sub(r"\D+", "", profile_id)
        if len(digits) >= 3:
            return digits[-3:]
        return "999"

    @staticmethod
    def _format_free_text_block(profile_key: str, text: str) -> str:
        cleaned = (text or "").strip()
        return f"{profile_key}\n{cleaned}\n"
