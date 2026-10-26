"""Instant replay of finished runs on the bundled sample clips.

A run costs GPU time and model calls, and the same clip with the same
instruction gives the same result, so finished runs can be frozen and replayed
with no model calls at all. Frozen runs live in demo_assets/cache/:

    manifest.json           one entry per run: key, clip, instruction, parent key, models
    <key>/snapshot.json     the job as the API returned it, URLs pointing at /cached/<key>/
    <key>/redacted.mp4, masks.mp4, audit.json, plan.json, report.html, report.json

A request replays an entry when the clip, the normalized instruction and the
parent run all match. Build entries with the models you want judged:

    python -m app.cache build                # every sample's suggested and follow-up prompts
    python -m app.cache freeze <job dir>...  # runs you already made (parents before follow-ups)

Mock runs are refused unless --allow-mock, so a replay never passes off a mock
as a model run. Replays say so in the API ("replay") and the UI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import settings
from .schemas import Plan

FILES = ("redacted.mp4", "masks.mp4", "audit.json", "plan.json", "report.html", "report.json")


def normalize(instruction: str) -> str:
    return re.sub(r"\s+", " ", instruction.strip().lower()).rstrip(" .!")


def entry_key(video_id: str, instruction: str, parent_key: str | None) -> str:
    return hashlib.sha1(f"{video_id}\n{normalize(instruction)}\n{parent_key or ''}".encode()).hexdigest()[:12]


@dataclass
class Entry:
    key: str
    video_id: str
    instruction: str
    parent_key: str | None
    models: dict
    source_job: str
    frozen_at: str


class SampleCache:
    def __init__(self, root: Path):
        self.root = root

    def _entries(self) -> list[Entry]:
        path = self.root / "manifest.json"
        if not path.exists():
            return []
        return [Entry(**e) for e in json.loads(path.read_text(encoding="utf-8"))]

    def lookup(self, video_id: str, instruction: str, parent_key: str | None) -> Entry | None:
        key = entry_key(video_id, instruction, parent_key)
        return next((e for e in self._entries() if e.key == key), None)

    def snapshot(self, entry: Entry) -> dict:
        return json.loads((self.root / entry.key / "snapshot.json").read_text(encoding="utf-8"))

    def plan(self, entry: Entry) -> Plan:
        return Plan(**json.loads((self.root / entry.key / "plan.json").read_text(encoding="utf-8")))

    def freeze(self, job_dir: Path, allow_mock: bool = False) -> Entry:
        """Copy a finished job into the cache and record it in the manifest."""
        snap = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
        mocked = [m for m in snap["modes"].values() if "mock" in m]
        if mocked and not allow_mock:
            raise ValueError(f"{job_dir.name} ran on {', '.join(mocked)}; pass --allow-mock to cache it anyway")
        entries = self._entries()
        parent_key = None
        if snap["parent_job_id"]:
            parent = next((e for e in entries if e.source_job == snap["parent_job_id"]), None)
            if parent is None:
                raise ValueError(f"{job_dir.name} is a follow-up of {snap['parent_job_id']}; freeze that run first")
            parent_key = parent.key
        key = entry_key(snap["video_id"], snap["instruction"], parent_key)
        dest = self.root / key
        dest.mkdir(parents=True, exist_ok=True)
        for name in FILES:
            if (job_dir / name).exists():
                shutil.copy2(job_dir / name, dest / name)
        snap["outputs"] = {k: (v if k == "original_url" else f"/cached/{key}/{Path(v).name}") for k, v in snap["outputs"].items()}
        (dest / "snapshot.json").write_text(json.dumps(snap, indent=2), encoding="utf-8")
        entry = Entry(key, snap["video_id"], snap["instruction"], parent_key, snap["modes"], snap["id"], time.strftime("%Y-%m-%d"))
        entries = [e for e in entries if e.key != key] + [entry]
        (self.root / "manifest.json").write_text(json.dumps([asdict(e) for e in entries], indent=2), encoding="utf-8")
        return entry


sample_cache = SampleCache(settings.cache_dir)


def build(allow_mock: bool) -> None:
    """Run every real sample's suggested prompts (and follow-ups of the first) and freeze them."""
    from .jobs import jobs
    from .videos import store

    store.load_samples()
    for video in store.samples():
        if video.annotations is not None:
            continue  # synthetic clips are test fixtures
        first = None
        for i, prompt in enumerate(video.suggested_prompts):
            job = _run_and_freeze(jobs, video.id, prompt, None, allow_mock)
            if i == 0 and job.status == "done":
                first = job
        for prompt in video.follow_up_prompts if first else ():
            _run_and_freeze(jobs, video.id, prompt, first.id, allow_mock)


def _run_and_freeze(jobs, video_id: str, instruction: str, parent_job_id: str | None, allow_mock: bool):
    t0 = time.time()
    job = jobs.run_now(video_id, instruction, parent_job_id)
    if job.status != "done":
        print(f"FAILED  {video_id}: {instruction!r}: {job.error}")
        return job
    entry = sample_cache.freeze(jobs.jobs_dir / job.id, allow_mock=allow_mock)
    print(f"cached  {entry.key}  {video_id}: {instruction!r}  ({time.time() - t0:.0f}s, {', '.join(entry.models.values())})")
    return job


def main() -> None:
    ap = argparse.ArgumentParser(description="Build and manage frozen sample runs for instant replay.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="run and freeze every sample's suggested prompts")
    b.add_argument("--allow-mock", action="store_true")
    f = sub.add_parser("freeze", help="freeze finished job folders (parents first)")
    f.add_argument("job_dirs", type=Path, nargs="+")
    f.add_argument("--allow-mock", action="store_true")
    sub.add_parser("list")
    args = ap.parse_args()
    if args.cmd == "build":
        build(args.allow_mock)
    elif args.cmd == "freeze":
        for d in args.job_dirs:
            e = sample_cache.freeze(d, allow_mock=args.allow_mock)
            print(f"cached  {e.key}  {e.video_id}: {e.instruction!r}")
    else:
        for e in sample_cache._entries():
            print(f"{e.key}  {e.frozen_at}  {e.video_id}: {e.instruction!r}{'  (follow-up of ' + e.parent_key + ')' if e.parent_key else ''}  [{', '.join(e.models.values())}]")


if __name__ == "__main__":
    main()
