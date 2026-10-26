"""Redaction Report (hash chain) and instant replay of frozen sample runs."""

import dataclasses
import json

import pytest
from fastapi.testclient import TestClient

from app import cache as cache_mod
from app import main
from app.cache import SampleCache
from app.jobs import jobs
from app.report import verify
from app.videos import store

DEMO = "Blur every face and license plate except the woman in the red jacket, and black out phone screens."


@pytest.fixture(scope="module")
def street():
    store.load_samples()
    return store.get("sample-street_crossing")


@pytest.fixture(scope="module")
def run(street):
    job = jobs.run_now(street.id, DEMO)
    assert job.status == "done", job.error
    return job


def test_report_chain_verifies_and_detects_tampering(run, street):
    folder = jobs.jobs_dir / run.id
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    assert [a["name"] for a in report["artifacts"]] == ["input", "plan.json", "redacted.mp4", "masks.mp4", "audit.json"]
    assert report["digest"] == report["artifacts"][-1]["link"]
    assert verify(folder, street.path) == (True, None)

    html = (folder / "report.html").read_text(encoding="utf-8")
    assert report["digest"] in html and "does not certify compliance" in html
    assert html.count("<figure>") == run.audit["counts"]["patched"]  # a before/after per audit fix
    assert run.outputs["report_url"].endswith("/report.html")

    redacted = folder / "redacted.mp4"
    original = redacted.read_bytes()
    try:
        redacted.write_bytes(original + b"\0")
        assert verify(folder, street.path) == (False, "redacted.mp4")
    finally:
        redacted.write_bytes(original)


@pytest.fixture
def sample_cache(tmp_path, monkeypatch):
    c = SampleCache(tmp_path / "cache")
    monkeypatch.setattr(cache_mod, "sample_cache", c)
    return c


def test_mock_runs_are_not_cached_unless_allowed(run, sample_cache):
    with pytest.raises(ValueError, match="allow-mock"):
        sample_cache.freeze(jobs.jobs_dir / run.id)


def test_frozen_runs_replay_without_running_models(run, street, sample_cache):
    entry = sample_cache.freeze(jobs.jobs_dir / run.id, allow_mock=True)
    replay = jobs.submit(street.id, DEMO.upper().rstrip(".") + "  ", None)  # same instruction, different spelling
    assert replay.status == "done" and replay.cache_key == entry.key
    view = replay.public()
    assert view["replay"]["models"] == run.modes
    assert view["outputs"]["redacted_url"] == f"/cached/{entry.key}/redacted.mp4"
    assert (sample_cache.root / entry.key / "redacted.mp4").exists()
    assert view["audit"]["counts"] == run.audit["counts"]

    # Follow-ups replay too, keyed on the frozen parent.
    follow = jobs.run_now(street.id, "Also hide the logo on the truck", run.id)
    sample_cache.freeze(jobs.jobs_dir / follow.id, allow_mock=True)
    again = jobs.submit(street.id, "Also hide the logo on the truck.", replay.id)
    assert again.replay is not None and again.public()["parent_job_id"] == replay.id


def test_cache_only_mode_refuses_uncached_work(run, street, sample_cache, monkeypatch):
    sample_cache.freeze(jobs.jobs_dir / run.id, allow_mock=True)
    monkeypatch.setattr(main, "settings", dataclasses.replace(main.settings, cache_only=True))
    client = TestClient(main.app)
    assert client.post("/api/jobs", json={"video_id": street.id, "instruction": "Pixelate everything"}).status_code == 409
    ok = client.post("/api/jobs", json={"video_id": street.id, "instruction": DEMO})
    assert ok.status_code == 200 and ok.json()["replay"] is not None
    assert client.post("/api/videos", files={"file": ("x.mp4", b"0", "video/mp4")}).status_code == 409
    assert client.post("/api/plan", json={"instruction": DEMO}).status_code == 409  # no planner calls either
