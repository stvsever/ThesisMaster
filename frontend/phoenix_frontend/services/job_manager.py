from __future__ import annotations

import threading
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .models import JobLogEntry, JobRecord


LogFn = Callable[[str, str], None]
JobTarget = Callable[[LogFn], Dict[str, Any]]

TERMINAL_STATUSES = {"succeeded", "failed"}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class JobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = threading.RLock()

    def start_job(self, *, session_id: str, kind: str, target: JobTarget) -> str:
        with self._lock:
            active = self.get_active_job_for_session(session_id)
            if active is not None:
                raise RuntimeError(
                    f"Session already has running job '{active.kind}' ({active.job_id}). Wait until it finishes."
                )
            job_id = f"job_{uuid.uuid4().hex[:12]}"
            record = JobRecord(
                job_id=job_id,
                session_id=session_id,
                kind=kind,
                status="queued",
                created_at=_now(),
            )
            self._jobs[job_id] = record

        thread = threading.Thread(
            target=self._run_job_thread,
            name=f"phoenix-{job_id}",
            args=(job_id, target),
            daemon=True,
        )
        thread.start()
        return job_id

    def _run_job_thread(self, job_id: str, target: JobTarget) -> None:
        self._set_status(job_id, "running")
        self._set_started(job_id, _now())

        def log(message: str, level: str = "INFO") -> None:
            self._append_log(job_id=job_id, line=message, level=level)

        try:
            result = target(log)
            self._set_result(job_id, result or {})
            self._set_status(job_id, "succeeded")
        except Exception as exc:
            self._append_log(job_id=job_id, line=f"{type(exc).__name__}: {exc}", level="ERROR")
            self._set_error(job_id, f"{type(exc).__name__}: {exc}")
            self._set_status(job_id, "failed")
        finally:
            self._set_finished(job_id, _now())

    def get_job(self, job_id: str) -> JobRecord:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return self._jobs[job_id]

    def get_active_job_for_session(self, session_id: str) -> Optional[JobRecord]:
        for record in self._jobs.values():
            if record.session_id != session_id:
                continue
            if record.status not in TERMINAL_STATUSES:
                return record
        return None

    def logs_since(self, job_id: str, after: int = 0) -> List[Dict[str, Any]]:
        with self._lock:
            record = self.get_job(job_id)
            return [item.to_dict() for item in record.logs if item.index > int(after)]

    def wait_until_complete(self, job_id: str, timeout_seconds: float = 0.0) -> JobRecord:
        start = time.time()
        while True:
            record = self.get_job(job_id)
            if record.status in TERMINAL_STATUSES:
                return record
            if timeout_seconds > 0 and (time.time() - start) > timeout_seconds:
                return record
            time.sleep(0.25)

    def _set_started(self, job_id: str, value: str) -> None:
        with self._lock:
            self._jobs[job_id].started_at = value

    def _set_finished(self, job_id: str, value: str) -> None:
        with self._lock:
            self._jobs[job_id].finished_at = value

    def _set_status(self, job_id: str, status: str) -> None:
        with self._lock:
            self._jobs[job_id].status = status

    def _set_result(self, job_id: str, result: Dict[str, Any]) -> None:
        with self._lock:
            self._jobs[job_id].result = dict(result)

    def _set_error(self, job_id: str, error: str) -> None:
        with self._lock:
            self._jobs[job_id].error = str(error)

    def _append_log(self, *, job_id: str, line: str, level: str) -> None:
        with self._lock:
            record = self._jobs[job_id]
            index = len(record.logs) + 1
            record.logs.append(
                JobLogEntry(
                    index=index,
                    timestamp=_now(),
                    line=str(line).rstrip(),
                    level=str(level).upper(),
                )
            )
