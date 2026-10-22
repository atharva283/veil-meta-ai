"""Tracked instance masks and the instance-level mask algebra used by plans.

Masks are stored cropped to their bounding box so a 30-second clip with a few
dozen tracked objects stays in tens of megabytes rather than gigabytes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

# An instance must cover this fraction of its own area inside the reference
# for select/subtract to treat the two as "overlapping".
OVERLAP_THRESHOLD = 0.35


@dataclass(eq=False)  # identity equality: masks hold numpy arrays
class FrameMask:
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 (exclusive), pixel coords
    mask: np.ndarray  # bool, shape (y1 - y0, x1 - x0)
    score: float = 1.0

    @property
    def area(self) -> int:
        return int(self.mask.sum())


@dataclass(eq=False)
class Track:
    track_id: str
    concept: str
    frames: dict[int, FrameMask] = field(default_factory=dict)
    source: str = "segment"  # "segment" or "audit_patch"

    @property
    def frame_indices(self) -> list[int]:
        return sorted(self.frames)

    @property
    def mean_score(self) -> float:
        if not self.frames:
            return 0.0
        return float(np.mean([fm.score for fm in self.frames.values()]))


def overlap_fraction(a: FrameMask, b: FrameMask) -> float:
    """Fraction of a's pixels that fall inside b."""
    x0, y0 = max(a.box[0], b.box[0]), max(a.box[1], b.box[1])
    x1, y1 = min(a.box[2], b.box[2]), min(a.box[3], b.box[3])
    if x0 >= x1 or y0 >= y1 or a.area == 0:
        return 0.0
    ca = a.mask[y0 - a.box[1] : y1 - a.box[1], x0 - a.box[0] : x1 - a.box[0]]
    cb = b.mask[y0 - b.box[1] : y1 - b.box[1], x0 - b.box[0] : x1 - b.box[0]]
    return float(np.logical_and(ca, cb).sum()) / a.area


def box_iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def tracks_overlap(a: Track, b: Track) -> bool:
    """True when a sits inside b on most of the frames they share."""
    shared = set(a.frames) & set(b.frames)
    if not shared:
        return False
    hits = sum(overlap_fraction(a.frames[f], b.frames[f]) >= OVERLAP_THRESHOLD for f in shared)
    return hits / len(shared) >= 0.5


def select(source: list[Track], reference: list[Track]) -> list[Track]:
    return [t for t in source if any(tracks_overlap(t, r) for r in reference)]


def subtract(source: list[Track], reference: list[Track]) -> list[Track]:
    return [t for t in source if not any(tracks_overlap(t, r) for r in reference)]


def union(*sets: Iterable[Track]) -> list[Track]:
    out: dict[str, Track] = {}
    for s in sets:
        for t in s:
            out.setdefault(t.track_id, t)
    return list(out.values())


def mask_from_full(full: np.ndarray, score: float = 1.0) -> FrameMask | None:
    """Crop a full-frame boolean mask to its bounding box."""
    ys, xs = np.nonzero(full)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    return FrameMask((x0, y0, x1, y1), full[y0:y1, x0:x1].copy(), score)


def link_detections(
    per_frame: dict[int, list[FrameMask]], concept: str, prefix: str, iou_min: float = 0.3
) -> list[Track]:
    """Greedy IoU linking of per-frame detections into tracks.

    Used when a segmenter returns detections without object ids (for example
    the Haar fallback or a per-frame SAM image pass).
    """
    tracks: list[Track] = []
    last_seen: dict[str, int] = {}
    for f in sorted(per_frame):
        candidates = sorted(per_frame[f], key=lambda m: -m.score)
        claimed: set[str] = set()
        for det in candidates:
            best, best_iou = None, iou_min
            for t in tracks:
                if t.track_id in claimed or f - last_seen[t.track_id] > 6:
                    continue
                prev = t.frames[last_seen[t.track_id]]
                iou = box_iou(prev.box, det.box)
                if iou > best_iou:
                    best, best_iou = t, iou
            if best is None:
                best = Track(f"{prefix}-{len(tracks) + 1}", concept)
                tracks.append(best)
            best.frames[f] = det
            last_seen[best.track_id] = f
            claimed.add(best.track_id)
    # Drop single-frame flickers; real objects persist for a few frames.
    return [t for t in tracks if len(t.frames) >= 3]


def fill_gaps(track: Track, max_gap: int) -> Track:
    """Bridge short detection dropouts so redactions don't flicker off.

    Each missing frame reuses the nearer neighbour's mask, shifted along the
    straight line between the two neighbouring boxes.
    """
    idx = track.frame_indices
    for a, b in zip(idx, idx[1:]):
        if not 1 < b - a <= max_gap + 1:
            continue
        fa, fb = track.frames[a], track.frames[b]
        for f in range(a + 1, b):
            t = (f - a) / (b - a)
            near = fa if t < 0.5 else fb
            dx = round((fb.box[0] - fa.box[0]) * t + fa.box[0] - near.box[0])
            dy = round((fb.box[1] - fa.box[1]) * t + fa.box[1] - near.box[1])
            x0, y0, x1, y1 = near.box
            track.frames[f] = FrameMask((x0 + dx, y0 + dy, x1 + dx, y1 + dy), near.mask, near.score)
    return track


def clip_mask(fm: FrameMask, width: int, height: int) -> FrameMask | None:
    """Clip a mask to the frame; returns None if nothing remains visible."""
    x0, y0, x1, y1 = fm.box
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
    if cx0 >= cx1 or cy0 >= cy1:
        return None
    if (cx0, cy0, cx1, cy1) == fm.box:
        return fm
    sub = fm.mask[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0]
    return FrameMask((cx0, cy0, cx1, cy1), sub, fm.score) if sub.any() else None


# ---------------------------------------------------------------------------
# RLE wire format shared with the Modal SAM 3.1 service.
# Row-major run lengths over the bounding-box crop, starting with a run of 0s.
# ---------------------------------------------------------------------------


def rle_encode(mask: np.ndarray) -> list[int]:
    flat = mask.astype(np.uint8).ravel()
    if flat.size == 0:
        return []
    changes = np.flatnonzero(np.diff(flat)) + 1
    bounds = np.concatenate(([0], changes, [flat.size]))
    runs = np.diff(bounds).tolist()
    return runs if flat[0] == 0 else [0, *runs]


def rle_decode(counts: list[int], shape: tuple[int, int]) -> np.ndarray:
    flat = np.zeros(shape[0] * shape[1], dtype=bool)
    pos, value = 0, False
    for run in counts:
        if value:
            flat[pos : pos + run] = True
        pos += run
        value = not value
    return flat.reshape(shape)
