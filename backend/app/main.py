"""Veil API: uploads, samples, plan previews, and redaction jobs."""

from __future__ import annotations

import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import settings
from .jobs import jobs
from .pipeline.video_io import VideoError
from .planner import make_plan
from .schemas import JobCreate
from .videos import store

ALLOWED_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    store.load_samples()
    yield


app = FastAPI(title="Veil", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _mode(live: bool, live_name: str, mock_name: str) -> dict:
    return {"live": live, "name": live_name if live else mock_name}


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "modes": {
            "planner": _mode(settings.planner_live, settings.planner_model, "mock planner"),
            "segmenter": _mode(settings.segmenter_live, "SAM 3.1 (local GPU)" if settings.sam_local else "SAM 3.1 on Modal", "mock SAM 3.1"),
            "auditor": _mode(settings.auditor_live, settings.auditor_model, "mock Muse Spark"),
        },
        "samples": len(store.samples()),
        "cache_only": settings.cache_only,
    }


@app.get("/api/samples")
def samples() -> list[dict]:
    return [v.public() for v in store.samples()]


@app.post("/api/videos")
async def upload_video(file: UploadFile = File(...)) -> dict:
    if settings.cache_only:
        raise HTTPException(409, "Uploads are off on this demo. Pick a sample clip to replay a precomputed run.")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(415, f"Veil accepts {', '.join(sorted(ALLOWED_SUFFIXES))} files. This one is {suffix or 'unknown'}.")
    limit = settings.max_upload_mb * 1024 * 1024
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        size = 0
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                tmp.close()
                Path(tmp.name).unlink(missing_ok=True)
                raise HTTPException(413, f"Videos can be up to {settings.max_upload_mb} MB.")
            tmp.write(chunk)
    try:
        rec = await run_in_threadpool(store.add_upload, Path(tmp.name), file.filename or "upload")
    except VideoError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        Path(tmp.name).unlink(missing_ok=True)
    return rec.public()


@app.get("/api/videos/{video_id}")
def get_video(video_id: str) -> dict:
    rec = store.get(video_id)
    if rec is None:
        raise HTTPException(404, "No video with that id. Upload it again or pick a sample.")
    return rec.public()


class PlanPreview(BaseModel):
    instruction: str = Field(min_length=3, max_length=2000)
    parent_job_id: str | None = None


@app.post("/api/plan")
def preview_plan(body: PlanPreview) -> dict:
    parent = jobs.get(body.parent_job_id) if body.parent_job_id else None
    try:
        plan, fallback = make_plan(body.instruction, parent.plan if parent else None)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"summary": plan.summary, "source": plan.source, "warnings": plan.warnings + ([fallback] if fallback else []), "steps": plan.as_view()}


@app.post("/api/jobs")
def create_job(body: JobCreate) -> dict:
    if store.get(body.video_id) is None:
        raise HTTPException(404, "No video with that id. Upload it again or pick a sample.")
    if body.parent_job_id and jobs.get(body.parent_job_id) is None:
        raise HTTPException(404, "The previous run expired. Run the full instruction again.")
    job = jobs.submit(body.video_id, body.instruction, body.parent_job_id, replay_only=settings.cache_only)
    if job is None:
        raise HTTPException(409, "This demo replays precomputed runs. Pick a sample and one of its suggested instructions.")
    return job.public()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "No job with that id.")
    return job.public()


app.mount("/media", StaticFiles(directory=settings.data_dir, check_dir=False), name="media")
app.mount("/samples", StaticFiles(directory=settings.samples_dir, check_dir=False), name="samples")
app.mount("/cached", StaticFiles(directory=settings.cache_dir, check_dir=False), name="cached")

# Serve the built frontend when present (npm run build), so one process can host the demo.
if settings.frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=settings.frontend_dist / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        target = settings.frontend_dist / path
        if path and target.is_file() and target.resolve().is_relative_to(settings.frontend_dist.resolve()):
            return FileResponse(target)
        return FileResponse(settings.frontend_dist / "index.html")

