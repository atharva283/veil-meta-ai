"""SAM re-check of the rendered output: every frame, coverage-based leak test."""

from pathlib import Path

import numpy as np

from app.auditor import mock_auditor
from app.auditor.sam_recheck import recheck
from app.jobs import Job, jobs
from app.orchestrator import run_job
from app.pipeline.tracks import FrameMask, Track
from app.pipeline.video_io import VideoMeta
from app.videos import VideoRecord, store


def _fm(box):
    x0, y0, x1, y1 = box
    return FrameMask(box, np.ones((y1 - y0, x1 - x0), dtype=bool))


class FakeSam:
    name = "fake"

    def __init__(self, tracks):
        self.tracks, self.calls = tracks, []

    def segment(self, video, concepts, score_threshold=None, frame_stride=None):
        self.calls.append((video.path, score_threshold, frame_stride))
        return {c: self.tracks for c in concepts}


def test_flags_only_uncovered_non_exception_detections():
    video = VideoRecord("v", "v", Path("in.mp4"), VideoMeta(200, 100, 24.0, 24, False))
    face = Track("face-1", "face", {0: _fm((10, 10, 30, 30)), 5: _fm((10, 10, 30, 30)), 6: _fm((10, 10, 30, 30))})
    seg = FakeSam([face])
    hidden = {0: [_fm((5, 5, 35, 35))]}  # frame 0 was blurred; frame 5 (between samples) was not
    kept = {6: [_fm((0, 0, 60, 90))]}  # frame 6: she is the exception
    result = recheck(seg, video, Path("out.mp4"), ["face"], lambda f: hidden.get(f, []), lambda f: kept.get(f, []))
    assert [(f.frame_index, f.concept) for f in result.findings] == [(5, "face")]
    assert result.frames_checked == 24
    assert seg.calls == [(Path("out.mp4"), 0.2, 1)]  # the rendered output, low threshold, every frame


def test_recheck_alone_catches_first_pass_misses(monkeypatch):
    store.load_samples()
    street = store.get("sample-street_crossing")
    monkeypatch.setattr(mock_auditor, "audit", lambda *a, **k: [])  # the sampled auditor sees nothing
    job = Job(id="t-recheck", video_id=street.id, instruction="Blur every face and license plate.")
    jobs._jobs[job.id] = job
    run_job(job, jobs)
    entries = {e["track_id"]: e for e in job.audit["entries"]}
    assert entries["face-07"]["status"] == "patched"
    assert entries["plate-05"]["status"] == "patched"
    assert job.audit["counts"]["flagged"] == 0
    assert job.audit["rechecked_frames"] == job.audit["frame_count"] == street.meta.frame_count


def test_offline_recheck_skips_uploads_instead_of_claiming_coverage():
    from app.orchestrator import Composition, _recheck
    from app.segmenter import mock_segmenter

    upload = VideoRecord("u", "u", Path("missing.mp4"), VideoMeta(200, 100, 24.0, 24, False))
    job = Job(id="t-skip", video_id="u", instruction="Blur faces")
    result = _recheck(job, mock_segmenter, upload, Composition(), Path("missing.mp4"), ["face"])
    assert result.findings == [] and result.frames_checked == 0
    assert any("re-check skipped" in e["message"] for e in job.events)
