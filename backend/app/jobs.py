"""Job records, progress reporting, and the background worker pool."""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from .config import settings
from .pipeline.tracks import Track
from .schemas import Plan

STAGES = [
    ("plan", "Plan", "Llama 3.1 turns the instruction into tool calls"),
    ("segment", "Segment", "SAM 3.1 finds and tracks each concept"),
    ("compose", "Compose", "Mask logic applies selections and exceptions"),
    ("render", "Render", "Effects are applied frame by frame"),
    ("audit", "Audit", "Muse Spark checks output frames for leaks"),
    ("patch", "Patch", "Leaks are re-segmented and re-rendered"),
]


@dataclass
class Job:
    id: str
    video_id: str
    instruction: str
    parent_job_id: str | None = None
    status: str = "queued"  # queued | running | done | error
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    plan: Plan | None = None
    stages: dict[str, dict] = field(
        default_factory=lambda: {key: {"status": "pending", "detail": "", "started": None, "ended": None} for key, *_ in STAGES}
    )
    progress: float = 0.0
    events: list[dict] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)
    audit: dict[str, Any] = field(default_factory=dict)
    timeline: dict[str, Any] = field(default_factory=dict)
    modes: dict[str, str] = field(default_factory=dict)
    final_tracks: dict[str, list[Track]] = field(default_factory=dict, repr=False)

    def log(self, message: str, level: str = "info", stage: str | None = None) -> None:
        self.events.append({"t": round(time.time() - self.created_at, 2), "level": level, "stage": stage, "message": message})

    def start_stage(self, key: str, detail: str = "") -> None:
        self.stages[key].update(status="active", detail=detail, started=time.time())

    def end_stage(self, key: str, detail: str | None = None, status: str = "done") -> None:
        st = self.stages[key]
        st.update(status=status, ended=time.time())
        if detail is not None:
            st["detail"] = detail

    def public(self) -> dict:
        stages = []
        for key, title, blurb in STAGES:
            st = self.stages[key]
            dur = (st["ended"] or time.time()) - st["started"] if st["started"] else None
            stages.append({"key": key, "title": title, "blurb": blurb, "status": st["status"], "detail": st["detail"], "duration_s": round(dur, 2) if dur is not None else None})
        return {
            "id": self.id,
            "video_id": self.video_id,
            "instruction": self.instruction,
            "parent_job_id": self.parent_job_id,
            "status": self.status,
            "error": self.error,
            "progress": round(self.progress, 3),
            "elapsed_s": round((self.finished_at or time.time()) - self.created_at, 2),
            "plan": (
                {"summary": self.plan.summary, "source": self.plan.source, "warnings": self.plan.warnings, "steps": self.plan.as_view()}
                if self.plan
                else None
            ),
            "stages": stages,
            "events": list(self.events),
            "outputs": dict(self.outputs),
            "audit": self.audit,
            "timeline": self.timeline,
            "modes": dict(self.modes),
        }


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        # One worker: a single GPU-backed SAM service handles requests in order anyway.
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="veil-job")
        self.jobs_dir = settings.data_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    def submit(self, video_id: str, instruction: str, parent_job_id: str | None) -> Job:
        job = Job(id=uuid.uuid4().hex[:10], video_id=video_id, instruction=instruction.strip(), parent_job_id=parent_job_id)
        with self._lock:
            self._jobs[job.id] = job
        self._pool.submit(self._run, job)
        return job

    def _run(self, job: Job) -> None:
        from .orchestrator import run_job  # local import: orchestrator imports this module

        job.status = "running"
        try:
            run_job(job, self)
            job.status = "done"
        except Exception as exc:
            job.status = "error"
            job.error = str(exc) or exc.__class__.__name__
            for st in job.stages.values():
                if st["status"] == "active":
                    st.update(status="error", ended=time.time())
            job.log(job.error, level="error")
            traceback.print_exc()
        finally:
            job.finished_at = time.time()

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)


jobs = JobManager()
