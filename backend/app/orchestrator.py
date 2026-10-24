"""The Veil agent loop: plan -> segment -> compose -> render -> audit -> patch."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

from .auditor import AuditPolicy, AuditSample, get_auditor, mock_auditor
from .auditor.sam_recheck import RecheckResult, recheck
from .config import settings
from .jobs import Job, JobManager
from .pipeline import video_io
from .pipeline.effects import CLASS_COLORS, apply_effect, concept_class, concept_color, draw_preview, obscured_mask
from .pipeline.tracks import FrameMask, Track, box_iou, clip_mask, select, subtract, tracks_overlap, union
from .planner import make_plan
from .schemas import AuditEntry, AuditFinding, Plan
from .segmenter import RefineRequest, get_segmenter, mock_segmenter
from .videos import VideoRecord, media_url, store

# (video id, segmenter name, concept) -> tracks. SAM results are deterministic
# for a given clip and prompt, so follow-up edits reuse them instantly.
_seg_cache: dict[tuple[str, str, str], list[Track]] = {}
_cache_lock = threading.Lock()


@dataclass
class Layer:
    effect: str
    concepts: list[str]  # every segment concept feeding this effect, found or not
    note: str
    tracks: list[Track]


@dataclass
class Composition:
    layers: list[Layer] = field(default_factory=list)
    kept: list[Track] = field(default_factory=list)
    kept_notes: dict[str, str] = field(default_factory=dict)  # track id -> why it stays visible


def _copy(tracks: list[Track]) -> list[Track]:
    return [Track(t.track_id, t.concept, dict(t.frames), t.source) for t in tracks]


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def _segment(job: Job, video: VideoRecord, plan: Plan, parent: Job | None) -> dict[str, list[Track]]:
    concepts = list(dict.fromkeys(s.concept for s in plan.steps if s.op == "segment" and s.concept))
    seg = get_segmenter()
    results: dict[str, list[Track]] = {}
    todo = []
    for c in concepts:
        if parent and c in parent.final_tracks:
            results[c] = _copy(parent.final_tracks[c])  # keeps the parent's audit patches
        elif (video.id, seg.name, c) in _seg_cache:
            results[c] = _copy(_seg_cache[(video.id, seg.name, c)])
        else:
            todo.append(c)
    reused = len(concepts) - len(todo)
    if reused:
        job.log(f"Reused cached tracks for {reused} concept(s).", stage="segment")
    if todo:
        try:
            fresh = seg.segment(video, todo)
        except Exception as exc:
            if seg is mock_segmenter:
                raise
            job.log(f"SAM 3.1 service failed ({exc}); using the offline segmenter.", level="warn", stage="segment")
            job.modes["segmenter"] = f"{mock_segmenter.name} (fallback)"
            seg = mock_segmenter
            fresh = seg.segment(video, todo)
        for c, tracks in fresh.items():
            with _cache_lock:
                _seg_cache[(video.id, seg.name, c)] = tracks
            results[c] = _copy(tracks)
    for c in concepts:
        n = len(results[c])
        if seg is mock_segmenter and not mock_segmenter.supports(video, c):
            job.log(f'"{c}" needs SAM 3.1 on uploaded clips (set MODAL_ENDPOINT); offline mode only detects faces and plates.', level="warn", stage="segment")
        else:
            job.log(f'Segment("{c}") found {n} tracked instance{"s" if n != 1 else ""}.', stage="segment")
    return results


def _compose(plan: Plan, seg: dict[str, list[Track]]) -> Composition:
    comp = Composition()
    values: dict[str, list[Track]] = {}
    roots: dict[str, list[str]] = {}
    by_id = {s.id: s for s in plan.steps}
    for step in plan.steps:
        if step.op == "segment":
            values[step.id], roots[step.id] = seg.get(step.concept, []), [step.concept]
        elif step.op == "union":
            values[step.id] = union(*(values[i] for i in step.inputs))
            roots[step.id] = list(dict.fromkeys(c for i in step.inputs for c in roots[i]))
        elif step.op == "select":
            values[step.id], roots[step.id] = select(values[step.source], values[step.reference]), roots[step.source]
        elif step.op == "subtract":
            src, ref = values[step.source], values[step.reference]
            kept = select(src, ref)
            values[step.id], roots[step.id] = subtract(src, ref), roots[step.source]
            # A track on both sides means tracking swapped identities partway
            # through; give the visible part its own id so the report shows both.
            redacted_ids = {t.track_id for t in values[step.id]}
            for t in kept:
                if t.track_id in redacted_ids:
                    t.track_id = f"{t.track_id}-visible"
            # Describe who is kept ("the woman in the red jacket"), not the step.
            why = by_id[step.reference].note or step.note or "excluded by instruction"
            for t in kept:
                if t.track_id not in comp.kept_notes:
                    comp.kept.append(t)
                    comp.kept_notes[t.track_id] = why
        elif step.op == "effect":
            concepts = roots[step.source]
            comp.layers.append(Layer(step.effect, concepts, step.note or ", ".join(concepts), list(values[step.source])))
    return comp


def _render(job: Job, video: VideoRecord, comp: Composition, job_dir, progress_span: tuple[float, float]) -> dict[int, AuditSample]:
    meta = video.meta
    every = max(1, round(meta.fps / max(0.1, settings.audit_sample_fps)))
    out = video_io.H264Writer(job_dir / "redacted.mp4", meta, audio_from=video.path)
    prev = video_io.H264Writer(job_dir / "masks.mp4", meta)
    samples: dict[int, AuditSample] = {}
    lo, hi = progress_span
    total = max(1, meta.frame_count)
    try:
        for i, frame in enumerate(video_io.iter_frames(video.path)):
            redacted_frame = frame.copy()
            preview = frame.copy()
            redacted = []
            for layer in comp.layers:
                for t in layer.tracks:
                    fm = t.frames.get(i)
                    fm = clip_mask(fm, meta.width, meta.height) if fm else None
                    if fm is None:
                        continue
                    apply_effect(redacted_frame, fm, layer.effect)
                    redacted.append(fm)
                    color = CLASS_COLORS["patched"] if t.source == "audit_patch" else concept_color(t.concept)
                    draw_preview(preview, fm, t.track_id, color)
            kept = [t.frames[i] for t in comp.kept if i in t.frames]
            for fm in kept:
                draw_preview(preview, fm, "kept visible", CLASS_COLORS["kept"], filled=False)
            out.write(redacted_frame)
            prev.write(preview)
            if i % every == 0:
                samples[i] = AuditSample(i, i / meta.fps, redacted_frame, redacted, kept)
            if i % 10 == 0:
                job.progress = lo + (hi - lo) * (i / total)
    finally:
        out.close()
        prev.close()
    return samples


def _absorb_patch(comp: Composition, finding: AuditFinding, track: Track) -> tuple[str, Track | None]:
    """Merge a re-segmented track into the layer redacting its concept.

    Returns ("added" | "extended", track) on success, ("kept", None) when the
    track is an intentional exception, or ("unfixable", None) otherwise.
    """
    layers = [l for l in comp.layers if finding.concept in l.concepts] or [
        l for l in comp.layers if any(concept_class(c) == concept_class(finding.concept) for c in l.concepts)
    ]
    if not layers:
        return "unfixable", None
    # Never override the user's exception, but only on the frames where it applies.
    outside = subtract([track], comp.kept)
    if not outside:
        return "kept", None
    track = outside[0]
    for layer in layers:
        for existing in layer.tracks:
            if existing.track_id == track.track_id or tracks_overlap(track, existing) or tracks_overlap(existing, track):
                added = 0
                for f, fm in track.frames.items():
                    if f not in existing.frames:
                        existing.frames[f] = fm
                        added += 1
                if not added:
                    return "unfixable", None  # already redacted there, yet still visible
                existing.source = "audit_patch"
                return "extended", existing
    track.concept = finding.concept
    track.source = "audit_patch"
    layers[0].tracks.append(track)
    return "added", track


def _group_findings(findings: list[AuditFinding], max_frame_gap: int) -> list[list[AuditFinding]]:
    """Cluster per-frame findings into distinct leaked objects.

    The auditor reports each sampled frame separately, so one missed face shows
    up many times. Findings chain into a group when they share a concept, are
    close in time, and the box hasn't jumped further than ~2.5 box diagonals.
    """
    groups: list[list[AuditFinding]] = []
    for f in sorted(findings, key=lambda f: f.frame_index):
        cx, cy = (f.box[0] + f.box[2]) / 2, (f.box[1] + f.box[3]) / 2
        diag = max(8.0, ((f.box[2] - f.box[0]) ** 2 + (f.box[3] - f.box[1]) ** 2) ** 0.5)
        for g in groups:
            last = g[-1]
            if last.concept != f.concept or not 0 <= f.frame_index - last.frame_index <= max_frame_gap:
                continue
            lx, ly = (last.box[0] + last.box[2]) / 2, (last.box[1] + last.box[3]) / 2
            if box_iou(last.box, f.box) > 0.1 or ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5 <= 2.5 * diag:
                if last.frame_index == f.frame_index:
                    break  # same object reported twice in one frame
                g.append(f)
                break
        else:
            groups.append([f])
    return groups


def _representative(group: list[AuditFinding]) -> AuditFinding:
    """The finding with the largest box: the clearest view to re-prompt SAM on."""
    return max(group, key=lambda f: (f.box[2] - f.box[0]) * (f.box[3] - f.box[1]))


def _recheck(job: Job, seg, video: VideoRecord, comp: Composition, output: Path, targets: list[str]) -> RecheckResult:
    """Re-segment the rendered output on every frame (see auditor.sam_recheck)."""
    W, H = video.meta.width, video.meta.height

    @cache
    def obscured_at(f: int) -> list[FrameMask]:
        out = []
        for layer in comp.layers:
            for t in layer.tracks:
                fm = clip_mask(t.frames[f], W, H) if f in t.frames else None
                hidden = obscured_mask(fm, layer.effect, W, H) if fm else None
                if hidden is not None:
                    out.append(hidden)
        return out

    def kept_at(f: int) -> list[FrameMask]:
        return [t.frames[f] for t in comp.kept if f in t.frames]

    def offline() -> RecheckResult:
        # Haar on every output frame flags pavement and doorways as faces, and
        # its patches blur them; without SAM or ground truth, skip and say so.
        if video.annotations is None:
            job.log("SAM re-check skipped: uploaded clips need SAM 3.1 for it (set MODAL_ENDPOINT).", level="warn", stage="audit")
            return RecheckResult([], 0)
        return recheck(mock_segmenter, video, output, targets, obscured_at, kept_at)

    if seg is mock_segmenter:
        return offline()
    try:
        return recheck(seg, video, output, targets, obscured_at, kept_at)
    except Exception as exc:
        job.log(f"SAM re-check failed ({exc}); using the offline segmenter.", level="warn", stage="audit")
        return offline()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _spans(frames: list[int], fps: float) -> list[list[float]]:
    spans: list[list[int]] = []
    for f in frames:
        if spans and f - spans[-1][1] <= 2:
            spans[-1][1] = f
        else:
            spans.append([f, f])
    return [[round(a / fps, 3), round((b + 1) / fps, 3)] for a, b in spans]


def _report(job: Job, video: VideoRecord, comp: Composition, patched: dict[str, dict], unresolved: list[AuditFinding], audited: list[int], rechecked: int, rounds: int) -> None:
    fps = video.meta.fps
    entries: list[AuditEntry] = []
    timeline: list[dict] = []

    def add(t: Track, concept: str, effect: str | None, status: str, note: str) -> None:
        frames = t.frame_indices
        if not frames:
            return
        entries.append(AuditEntry(
            track_id=t.track_id, concept=concept, effect=effect, status=status,
            start_s=round(frames[0] / fps, 3), end_s=round((frames[-1] + 1) / fps, 3),
            frames=len(frames), confidence=round(t.mean_score, 3), note=note,
        ))  # fmt: skip
        timeline.append({
            "track_id": t.track_id, "concept": concept, "class": concept_class(concept),
            "status": status, "effect": effect, "spans": _spans(frames, fps),
        })  # fmt: skip

    seen: set[str] = set()
    for layer in comp.layers:
        for t in layer.tracks:
            if t.track_id in seen:
                continue
            seen.add(t.track_id)
            span = set(range(t.frame_indices[0], t.frame_indices[-1] + 1)) if t.frames else set()
            checks = sum(1 for f in audited if f in span)
            if t.track_id in patched:
                p = patched[t.track_id]
                note = (
                    f"Missed by the first segmentation pass. Audit caught it at {p['at']:.2f}s and SAM re-segmented it."
                    if p["kind"] == "added"
                    else f"Tracking dropped out; audit caught the gap at {p['at']:.2f}s and the track was extended."
                )
                add(t, t.concept, layer.effect, "patched", note)
            elif t.source == "audit_patch":
                add(t, t.concept, layer.effect, "patched", "Caught by the audit in the previous pass and carried over.")
            else:
                note = f"No leak found in {checks} sampled frame{'s' if checks != 1 else ''}"
                if rechecked:
                    note += f" or by the SAM re-check of all {len(span)} frames it appears in"
                add(t, t.concept, layer.effect, "verified", note + ".")
    for t in comp.kept:
        add(t, t.concept, None, "kept_visible", f"Left visible on purpose: {comp.kept_notes.get(t.track_id, 'exception')}.")
    for i, f in enumerate(unresolved, start=1):
        entries.append(AuditEntry(
            track_id=f"leak-{i}", concept=f.concept, effect=None, status="flagged",
            start_s=round(f.frame_index / fps, 3), end_s=round((f.frame_index + 1) / fps, 3),
            frames=1, confidence=0.0, note=f"{f.reason or 'Possible leak'}. Couldn't re-segment it; review this moment manually.",
        ))  # fmt: skip
        timeline.append({"track_id": f"leak-{i}", "concept": f.concept, "class": concept_class(f.concept), "status": "flagged", "effect": None, "spans": [[round(f.frame_index / fps, 3), round((f.frame_index + 1) / fps, 3)]]})

    counts = {s: sum(1 for e in entries if e.status == s) for s in ("verified", "patched", "kept_visible", "flagged")}
    job.audit = {
        "entries": [e.model_dump() for e in sorted(entries, key=lambda e: (e.start_s, e.track_id))],
        "counts": counts,
        "rounds": rounds,
        "audited_frames": len(audited),
        "rechecked_frames": rechecked,
        "frame_count": video.meta.frame_count,
        "patches": [{"track_id": tid, **p} for tid, p in patched.items()],
    }
    job.timeline = {"duration_s": round(video.meta.duration_s, 3), "fps": fps, "tracks": timeline}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_job(job: Job, manager: JobManager) -> None:
    video = store.get(job.video_id)
    if video is None:
        raise ValueError("That video is no longer available. Upload it again or pick a sample.")
    parent = manager.get(job.parent_job_id) if job.parent_job_id else None
    if parent and (parent.video_id != job.video_id or parent.plan is None):
        parent = None
    job_dir = manager.jobs_dir / job.id
    job_dir.mkdir(parents=True, exist_ok=True)
    seg, auditor = get_segmenter(), get_auditor()
    job.modes = {
        "planner": settings.planner_model if settings.planner_live else "mock planner",
        "segmenter": seg.name,
        "auditor": auditor.name,
    }
    job.outputs["original_url"] = media_url(video.path)

    # 1. Plan
    job.start_stage("plan")
    plan, fallback_reason = make_plan(job.instruction, parent.plan if parent else None)
    if fallback_reason:
        job.log(fallback_reason, level="warn", stage="plan")
        job.modes["planner"] = plan.source
    job.plan = plan
    for w in plan.warnings:
        job.log(w, level="warn", stage="plan")
    job.log(f"{plan.source} produced {len(plan.steps)} steps.", stage="plan")
    (job_dir / "plan.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    job.end_stage("plan", f"{len(plan.steps)} steps")
    job.progress = 0.08

    # 2. Segment
    job.start_stage("segment")
    seg_results = _segment(job, video, plan, parent)
    job.end_stage("segment", f"{sum(len(v) for v in seg_results.values())} tracks across {len(seg_results)} concepts")
    job.progress = 0.3

    # 3. Compose
    job.start_stage("compose")
    comp = _compose(plan, seg_results)
    n_red = len({t.track_id for l in comp.layers for t in l.tracks})
    job.log(f"{n_red} instance(s) to redact, {len(comp.kept)} kept visible by exception.", stage="compose")
    job.end_stage("compose", f"{n_red} to redact, {len(comp.kept)} kept")

    # 4. Render
    job.start_stage("render")
    samples = _render(job, video, comp, job_dir, (0.32, 0.6))
    job.end_stage("render", f"{video.meta.frame_count} frames")

    # 5-6. Audit and patch until clean (bounded)
    targets = list(dict.fromkeys(c for l in comp.layers for c in l.concepts))
    policy = AuditPolicy(targets=targets, kept_descriptions=sorted(set(comp.kept_notes.values())))
    patched: dict[str, dict] = {}
    unresolved: list[AuditFinding] = []
    rounds = rechecked = 0
    job.start_stage("audit")
    for rounds in range(1, settings.audit_max_rounds + 1):
        sample_list = [samples[k] for k in sorted(samples)]
        try:
            findings = auditor.audit(video, sample_list, policy)
        except Exception as exc:
            if auditor is mock_auditor:
                raise
            job.log(f"Muse Spark audit failed ({exc}); using the offline auditor.", level="warn", stage="audit")
            auditor = mock_auditor
            job.modes["auditor"] = f"{mock_auditor.name} (fallback)"
            findings = auditor.audit(video, sample_list, policy)
        # Muse Spark judges sampled frames; SAM re-segments every frame in between.
        rc = _recheck(job, seg, video, comp, job_dir / "redacted.mp4", targets)
        rechecked = rc.frames_checked
        findings = findings + rc.findings
        sample_gap = max(1, round(video.meta.fps / max(0.1, settings.audit_sample_fps)))
        groups = _group_findings(findings, max_frame_gap=3 * sample_gap)
        findings = [_representative(g) for g in groups]
        job.log(
            f"Audit round {rounds}: checked {len(sample_list)} sampled frames"
            + (f" and re-segmented all {rechecked}" if rechecked else "")
            + f", found {len(findings)} leaked object(s)"
            + (f" across {sum(len(g) for g in groups)} frame reports." if groups else "."),
            stage="audit",
        )
        job.progress = 0.62 + 0.1 * rounds
        if not findings:
            unresolved = []
            break
        if rounds == 1:
            job.end_stage("audit", f"{len(findings)} leak(s) found")
            job.start_stage("patch")
        try:
            tracks = seg.refine(video, [RefineRequest(f.concept, f.frame_index, f.box) for f in findings])
        except Exception as exc:
            job.log(f"Re-segmentation failed ({exc}); using the offline segmenter.", level="warn", stage="patch")
            tracks = mock_segmenter.refine(video, [RefineRequest(f.concept, f.frame_index, f.box) for f in findings])
        unresolved = []
        changed = False
        for finding, track in zip(findings, tracks):
            kind, absorbed = _absorb_patch(comp, finding, track) if track else ("unfixable", None)
            if absorbed is None:
                if kind == "unfixable":
                    unresolved.append(finding)
                continue
            changed = True
            if absorbed.track_id not in patched:
                patched[absorbed.track_id] = {"kind": kind, "at": round(finding.frame_index / video.meta.fps, 2), "concept": finding.concept, "reason": finding.reason}
                job.log(f"Patched {finding.concept} at {finding.frame_index / video.meta.fps:.2f}s ({'new track' if kind == 'added' else 'extended track'}).", stage="patch")
        if not changed:
            break
        samples = _render(job, video, comp, job_dir, (0.75, 0.95))
    if job.stages["patch"]["status"] == "active":
        job.end_stage("patch", f"{len(patched)} patched" + (f", {len(unresolved)} flagged" if unresolved else ""))
    else:
        job.end_stage("patch", "Nothing to patch", status="skipped")
    if job.stages["audit"]["status"] == "active":
        job.end_stage("audit", "No leaks found")
    if unresolved:
        job.log(f"{len(unresolved)} possible leak(s) need manual review.", level="warn", stage="audit")

    # Follow-up edits start from these: segment results (extended in place by
    # patches) plus any tracks the audit added.
    job.final_tracks = {c: list(ts) for c, ts in seg_results.items()}
    for layer in comp.layers:
        for t in layer.tracks:
            if t.source != "audit_patch":
                continue
            bucket = job.final_tracks.setdefault(t.concept, [])
            same = next((x for x in bucket if x.track_id == t.track_id), None)
            if same is None:
                bucket.append(t)
            elif same is not t:
                # t is a per-frame fragment of a segment track (see tracks._split):
                # carry its patch frames back to the whole track.
                for f, fm in t.frames.items():
                    same.frames.setdefault(f, fm)
                same.source = "audit_patch"

    _report(job, video, comp, patched, unresolved, sorted(samples), rechecked, rounds)
    job.outputs["redacted_url"] = media_url(job_dir / "redacted.mp4")
    job.outputs["masks_url"] = media_url(job_dir / "masks.mp4")
    (job_dir / "audit.json").write_text(json.dumps({"instruction": job.instruction, "plan": plan.model_dump(), **job.audit}, indent=2), encoding="utf-8")
    job.outputs["audit_url"] = media_url(job_dir / "audit.json")
    job.progress = 1.0
    job.log("Done. Redacted video and audit log are ready.", stage="done")
