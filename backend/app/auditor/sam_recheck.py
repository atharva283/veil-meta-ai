"""Exhaustive output re-check: run the segmenter on the redacted video itself.

Muse Spark reads a few sampled frames and judges meaning (a reflection, a name
badge). This pass is the opposite: SAM 3.1 looks for every target concept
again, on every output frame, at a low threshold. A blurred face still looks
like a face, so a detection only counts as a leak when the pixels the effects
actually hid cover less than MIN_COVERAGE of it and it isn't the user's
exception on that frame.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from ..pipeline.tracks import OVERLAP_THRESHOLD, FrameMask, covered_fraction, overlap_fraction
from ..schemas import AuditFinding
from ..segmenter.base import Segmenter
from ..videos import VideoRecord

SCORE_THRESHOLD = 0.2
MIN_COVERAGE = 0.8
MIN_AREA = 40  # px; specks aren't identifiable


@dataclass
class RecheckResult:
    findings: list[AuditFinding]  # one per leaking frame; the orchestrator groups them
    frames_checked: int


def recheck(
    seg: Segmenter,
    video: VideoRecord,
    output: Path,
    targets: list[str],
    obscured_at: Callable[[int], list[FrameMask]],
    kept_at: Callable[[int], list[FrameMask]],
) -> RecheckResult:
    """Segment `targets` on every frame of `output` and report what isn't hidden.

    obscured_at(f) gives the regions the effects hid on frame f; kept_at(f) the
    instances left visible on purpose.
    """
    rendered = replace(video, id=f"{video.id}:redacted", path=output)
    found = seg.segment(rendered, targets, score_threshold=SCORE_THRESHOLD, frame_stride=1)
    fps = video.meta.fps
    findings: list[AuditFinding] = []
    for concept, tracks in found.items():
        for t in tracks:
            for f, fm in sorted(t.frames.items()):
                if fm.area < MIN_AREA:
                    continue
                if any(overlap_fraction(fm, k) >= OVERLAP_THRESHOLD for k in kept_at(f)):
                    continue
                hidden = covered_fraction(fm, obscured_at(f))
                if hidden >= MIN_COVERAGE:
                    continue
                findings.append(AuditFinding(
                    frame_index=f, concept=concept, box=fm.box,
                    reason=f"SAM re-check still finds a {concept} at {f / fps:.2f}s ({hidden:.0%} hidden)",
                ))  # fmt: skip
    return RecheckResult(findings, video.meta.frame_count)
