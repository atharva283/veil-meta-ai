"""Instant replay of finished runs on the bundled sample clips.

A run costs GPU time and model calls, and the same clip with the same
instruction gives the same result, so finished runs can be frozen and replayed
with no model calls at all. Frozen runs live in demo_assets/cache/:

    manifest.json           one entry per run: key, clip, instruction, parent key, models
    <key>/snapshot.json     the job as the API returned it, URLs pointing at /cached/<key>/
    <key>/redacted.mp4, masks.mp4, audit.json, plan.json, report.html, report.json
    <key>/comparison_grid.mp4, comparison_pair.mp4   (rendered by `build`)

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
import subprocess
import sys
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

    def freeze(self, job_dir: Path, allow_mock: bool = False, allow_mock_planner: bool = False) -> Entry:
        """Copy a finished job into the cache and record it in the manifest.

        Refuses runs whose masks or audit came from a mock. A rule-based plan is
        allowed with allow_mock_planner (it stays labelled as such).
        """
        snap = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
        mocked = [m for k, m in snap["modes"].items() if "mock" in m and not (allow_mock_planner and k == "planner")]
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


def build(allow_mock: bool, clips: list[str] | None = None, max_prompts: int | None = None, follow_ups: bool = True, allow_mock_planner: bool = False) -> None:
    """Run real samples' suggested prompts (and follow-ups of the first), freeze them, render comparisons."""
    from .jobs import jobs
    from .videos import store

    store.load_samples()
    for video in store.samples():
        if video.annotations is not None:
            continue  # synthetic clips are test fixtures
        if clips and video.id.removeprefix("sample-") not in clips:
            continue
        first = None
        for i, prompt in enumerate(video.suggested_prompts[:max_prompts]):
            job, entry = _run_and_freeze(jobs, video, prompt, None, allow_mock, allow_mock_planner)
            if i == 0 and entry is not None:
                first = job
        for prompt in video.follow_up_prompts if first and follow_ups else ():
            _run_and_freeze(jobs, video, prompt, first.id, allow_mock, allow_mock_planner)


def _run_and_freeze(jobs, video, instruction: str, parent_job_id: str | None, allow_mock: bool, allow_mock_planner: bool):
    t0 = time.time()
    job = jobs.run_now(video.id, instruction, parent_job_id)
    if job.status != "done":
        print(f"FAILED   {video.id}: {instruction!r}: {job.error}")
        return job, None
    try:
        entry = sample_cache.freeze(jobs.jobs_dir / job.id, allow_mock=allow_mock, allow_mock_planner=allow_mock_planner)
    except ValueError as exc:
        print(f"SKIPPED  {video.id}: {instruction!r}: {exc}")
        return job, None
    render_comparisons(video.path, sample_cache.root / entry.key)
    print(f"cached   {entry.key}  {video.id}: {instruction!r}  ({time.time() - t0:.0f}s; {', '.join(entry.models.values())})")
    return job, entry


def render_comparisons(original: Path, entry_dir: Path) -> None:
    """comparison_grid.mp4 and comparison_pair.mp4 next to the frozen run (demo_assets/make_comparison.py)."""
    script = settings.samples_dir.parent / "make_comparison.py"
    for layout in ("grid", "pair"):
        subprocess.run(
            [sys.executable, str(script), "--original", str(original), "--job", str(entry_dir),
             "--out", str(entry_dir / f"comparison_{layout}.mp4"), "--layout", layout],
            check=True, capture_output=True,
        )  # fmt: skip


def add_clip(src: Path, name: str, title: str, prompts: list[str], follow_ups: list[str]) -> Path:
    """Normalize a recording into demo_assets/clips/<name>.mp4 with a thumbnail and sidecar."""
    from .pipeline import video_io

    dest = settings.samples_dir / f"{name}.mp4"
    meta = video_io.normalize(src, dest)
    video_io.save_thumbnail(dest, dest.with_suffix(".jpg"))
    sidecar = dest.with_suffix(".json")
    if not sidecar.exists():
        sidecar.write_text(json.dumps({
            "title": title, "description": "", "suggested_prompts": prompts, "order": 0, "follow_up_prompts": follow_ups,
        }, indent=2) + "\n", encoding="utf-8")  # fmt: skip
    print(f"added    {dest.name}: {meta.width}x{meta.height}, {meta.fps:.2f} fps, {meta.frame_count} frames, {'with' if meta.has_audio else 'no'} audio")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description="Build and manage frozen sample runs for instant replay.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="run and freeze every sample's suggested prompts")
    b.add_argument("--allow-mock", action="store_true")
    b.add_argument("--allow-mock-planner", action="store_true", help="keep runs whose plan came from the rule-based planner")
    b.add_argument("--clips", help="comma-separated clip names (default: every real sample)")
    b.add_argument("--max-prompts", type=int, help="suggested prompts per clip (default: all)")
    b.add_argument("--no-follow-ups", action="store_true")
    a = sub.add_parser("add-clip", help="normalize a recording into demo_assets/clips/")
    a.add_argument("src", type=Path)
    a.add_argument("--name", required=True, help="file stem, e.g. desk_mine")
    a.add_argument("--title", required=True)
    a.add_argument("--prompt", action="append", default=[], help="suggested instruction (repeatable)")
    a.add_argument("--follow-up", action="append", default=[], help="follow-up instruction (repeatable)")
    f = sub.add_parser("freeze", help="freeze finished job folders (parents first)")
    f.add_argument("job_dirs", type=Path, nargs="+")
    f.add_argument("--allow-mock", action="store_true")
    sub.add_parser("list")
    args = ap.parse_args()
    if args.cmd == "build":
        build(args.allow_mock, args.clips.split(",") if args.clips else None, args.max_prompts, not args.no_follow_ups, args.allow_mock_planner)
    elif args.cmd == "add-clip":
        add_clip(args.src, args.name, args.title, args.prompt, args.follow_up)
    elif args.cmd == "freeze":
        for d in args.job_dirs:
            e = sample_cache.freeze(d, allow_mock=args.allow_mock)
            print(f"cached  {e.key}  {e.video_id}: {e.instruction!r}")
    else:
        for e in sample_cache._entries():
            print(f"{e.key}  {e.frozen_at}  {e.video_id}: {e.instruction!r}{'  (follow-up of ' + e.parent_key + ')' if e.parent_key else ''}  [{', '.join(e.models.values())}]")


if __name__ == "__main__":
    main()
