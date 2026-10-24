"""End-to-end: plan -> segment -> render -> audit -> patch on the bundled street clip."""

import subprocess

import pytest

from app.jobs import Job, jobs
from app.orchestrator import run_job
from app.videos import store

DEMO = "Blur every face and license plate except the woman in the red jacket, and black out phone screens."


@pytest.fixture(scope="module")
def street():
    store.load_samples()
    video = store.get("sample-street_crossing")
    if video is None:
        pytest.skip("demo clips missing; run demo_assets/generate_demo_clips.py")
    return video


@pytest.fixture(scope="module")
def first_run(street):
    job = Job(id="t-first", video_id=street.id, instruction=DEMO)
    jobs._jobs[job.id] = job
    run_job(job, jobs)
    return job


def _by_id(job):
    return {e["track_id"]: e for e in job.audit["entries"]}


def test_exception_is_kept_visible(first_run):
    entries = _by_id(first_run)
    assert entries["face-02"]["status"] == "kept_visible"
    assert "woman in the red jacket" in entries["face-02"]["note"]


def test_audit_catches_and_patches_first_pass_misses(first_run):
    entries = _by_id(first_run)
    # face-07 (tiny, far sidewalk) and plate-05 (frame edge) are withheld on the first pass.
    assert entries["face-07"]["status"] == "patched"
    assert entries["plate-05"]["status"] == "patched"
    assert first_run.audit["counts"]["flagged"] == 0
    assert first_run.stages["patch"]["status"] == "done"


def test_everything_else_is_redacted_and_verified(first_run):
    entries = _by_id(first_run)
    for tid in ("face-01", "face-03", "plate-01", "phone-03", "phone-06"):
        assert entries[tid]["status"] == "verified"
    assert entries["phone-03"]["effect"] == "blackout"


def test_outputs_are_written(first_run):
    folder = jobs.jobs_dir / first_run.id
    for name in ("redacted.mp4", "masks.mp4", "audit.json", "plan.json"):
        assert (folder / name).stat().st_size > 0
    assert first_run.outputs["redacted_url"].endswith("/redacted.mp4")


def test_follow_up_reuses_tracks_and_carries_patches(street, first_run):
    job = Job(id="t-follow", video_id=street.id, instruction="Also hide the logo on the truck", parent_job_id=first_run.id)
    jobs._jobs[job.id] = job
    run_job(job, jobs)
    entries = _by_id(job)
    assert entries["logo-03"]["status"] == "verified"
    assert entries["face-07"]["status"] == "patched"  # carried over, not re-discovered
    assert entries["face-02"]["status"] == "kept_visible"
    assert any("Reused cached tracks" in e["message"] for e in job.events)


def test_writers_after_an_audio_mux_still_work(tmp_path):
    """imageio-ffmpeg appends audio flags to output_params in place; the shared H264_OUT must stay clean."""
    import numpy as np

    from app.pipeline import video_io

    src = tmp_path / "with_audio.mp4"
    subprocess.run(
        [video_io.FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=gray:s=64x48:r=10:d=1",
         "-f", "lavfi", "-i", "sine=d=1", "-shortest", "-pix_fmt", "yuv420p", str(src)],
        check=True,
    )  # fmt: skip
    meta = video_io.probe(src)
    assert meta.has_audio
    frame = np.zeros((48, 64, 3), np.uint8)
    for name, audio in (("redacted.mp4", src), ("masks.mp4", None)):
        w = video_io.H264Writer(tmp_path / name, meta, audio_from=audio)
        w.write(frame)
        w.close()
    assert video_io.H264_OUT == ["-preset", "veryfast", "-crf", "20", "-movflags", "+faststart"]
    assert video_io.probe(tmp_path / "masks.mp4").has_audio is False
