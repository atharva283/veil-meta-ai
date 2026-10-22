"""Offline auditor: checks rendered output for leaks without a vision model.

Samples: compares each sampled output frame with ground truth, so a target
the first pass missed shows up as a finding exactly where it leaked.
Uploads: re-runs Haar face/plate detection on the redacted frames; anything
still detectable outside the redacted and intentionally visible regions leaked.
"""

from __future__ import annotations

import numpy as np

from ..pipeline.effects import concept_class
from ..pipeline.tracks import FrameMask, overlap_fraction
from ..schemas import AuditFinding
from ..segmenter.mock import MockSegmenter
from ..videos import VideoRecord
from .base import AuditPolicy, AuditSample


def _covered(fm: FrameMask, masks: list[FrameMask]) -> float:
    """Fraction of fm covered by the union of masks (approximate: max single overlap
    plus a union pass when several masks touch it)."""
    touching = [m for m in masks if overlap_fraction(fm, m) > 0]
    if not touching:
        return 0.0
    x0, y0, x1, y1 = fm.box
    acc = np.zeros(fm.mask.shape, dtype=bool)
    for m in touching:
        ix0, iy0, ix1, iy1 = max(x0, m.box[0]), max(y0, m.box[1]), min(x1, m.box[2]), min(y1, m.box[3])
        acc[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0] |= m.mask[iy0 - m.box[1] : iy1 - m.box[1], ix0 - m.box[0] : ix1 - m.box[0]]
    return float(np.logical_and(acc, fm.mask).sum()) / max(1, fm.area)


class MockAuditor:
    name = "mock Muse Spark"

    def __init__(self, segmenter: MockSegmenter):
        self._seg = segmenter

    def audit(self, video: VideoRecord, samples: list[AuditSample], policy: AuditPolicy) -> list[AuditFinding]:
        findings: list[AuditFinding] = []
        if video.annotations:
            truth = {c: self._seg._gt_tracks(video, c, include_hard=True) for c in policy.targets}
            for s in samples:
                for concept, tracks in truth.items():
                    for t in tracks:
                        fm = t.frames.get(s.frame_index)
                        if fm is None or fm.area < 40:
                            continue
                        if any(overlap_fraction(fm, k) >= 0.35 for k in s.kept):
                            continue
                        if _covered(fm, s.redacted) < 0.6:
                            findings.append(AuditFinding(
                                frame_index=s.frame_index, concept=concept, box=fm.box,
                                reason=f"Unredacted {concept} visible at {s.time_s:.2f}s",
                            ))
            return findings

        for concept in policy.targets:
            if concept_class(concept) not in ("face", "plate"):
                continue
            for s in samples:
                for fm in self._seg.detect_frame(s.image, concept, min_neighbors=6):
                    if any(overlap_fraction(fm, k) >= 0.35 for k in s.kept):
                        continue
                    if _covered(fm, s.redacted) < 0.3:
                        findings.append(AuditFinding(
                            frame_index=s.frame_index, concept=concept, box=fm.box,
                            reason=f"Detector still finds a {concept} at {s.time_s:.2f}s",
                        ))
        return findings
