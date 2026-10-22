"""Offline stand-in for SAM 3.1.

Samples ship with per-frame ground-truth shapes (demo_assets/clips/*.annotations.json).
On the first text-prompt pass, tracks marked "hard" are withheld to mimic
SAM's typical misses (profile faces, tiny distant plates), which gives the
audit loop something real to catch. A refine request around a flagged box
returns the matching track, as a box-prompted SAM pass would.

Uploads have no ground truth, so faces and plates fall back to OpenCV Haar
cascades; other concepts need the live SAM 3.1 service.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from ..pipeline import video_io
from ..pipeline.effects import concept_class
from ..pipeline.tracks import FrameMask, Track, box_iou, clip_mask, fill_gaps, link_detections
from ..videos import VideoRecord
from .base import RefineRequest, normalize_concept

SYNONYMS = {
    "woman": "person", "man": "person", "pedestrian": "person", "people": "person",
    "licence plate": "license plate", "number plate": "license plate", "plate": "license plate",
    "phone": "phone screen", "smartphone": "phone screen", "mobile screen": "phone screen",
    "head": "face", "vehicle": "car", "monitor": "screen", "display": "screen",
}  # fmt: skip


def _score(track_id: str, lo: float, hi: float) -> float:
    h = int(hashlib.sha1(track_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return round(lo + (hi - lo) * h, 3)


@lru_cache(maxsize=8)
def _load_annotations(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rasterize(shapes: list[list], width: int, height: int, score: float) -> FrameMask | None:
    boxes = []
    for s in shapes:
        if s[0] == "e":
            _, cx, cy, rx, ry = s
            boxes.append((cx - rx, cy - ry, cx + rx, cy + ry))
        elif s[0] == "r":
            boxes.append(tuple(s[1:5]))
        elif s[0] == "p":
            pts = np.array(s[1], dtype=np.int32)
            boxes.append((pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()))
    x0 = int(min(b[0] for b in boxes))
    y0 = int(min(b[1] for b in boxes))
    x1 = int(max(b[2] for b in boxes)) + 1
    y1 = int(max(b[3] for b in boxes)) + 1
    crop = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    for s in shapes:
        if s[0] == "e":
            _, cx, cy, rx, ry = s
            cv2.ellipse(crop, (round(cx - x0), round(cy - y0)), (round(rx), round(ry)), 0, 0, 360, 1, -1)
        elif s[0] == "r":
            cv2.rectangle(crop, (round(s[1] - x0), round(s[2] - y0)), (round(s[3] - x0), round(s[4] - y0)), 1, -1)
        elif s[0] == "p":
            cv2.fillPoly(crop, [np.array(s[1], dtype=np.int32) - [x0, y0]], 1)
    return clip_mask(FrameMask((x0, y0, x1, y1), crop.astype(bool), score), width, height)


class MockSegmenter:
    name = "mock SAM 3.1"

    def __init__(self) -> None:
        self._cascades: dict[str, cv2.CascadeClassifier] = {}

    # -- ground-truth samples ------------------------------------------------

    def _gt_tracks(self, video: VideoRecord, concept: str, include_hard: bool) -> list[Track]:
        ann = _load_annotations(video.annotations)
        want = normalize_concept(concept)
        want = SYNONYMS.get(want, want)
        out = []
        for t in ann["tracks"]:
            labels = {SYNONYMS.get(normalize_concept(lbl), normalize_concept(lbl)) for lbl in t["labels"]}
            if want not in labels or (t.get("hard") and not include_hard):
                continue
            score = _score(t["id"], 0.55, 0.7) if t.get("hard") else _score(t["id"], 0.81, 0.97)
            track = Track(t["id"], concept)
            for f, shapes in t["frames"].items():
                fm = _rasterize(shapes, ann["width"], ann["height"], score)
                if fm is not None:
                    track.frames[int(f)] = fm
            if track.frames:
                out.append(track)
        return out

    # -- Haar fallback for uploads -------------------------------------------

    def _cascade(self, name: str) -> cv2.CascadeClassifier:
        if name not in self._cascades:
            self._cascades[name] = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / name))
        return self._cascades[name]

    def detect_frame(self, frame: np.ndarray, concept: str, min_neighbors: int = 5) -> list[FrameMask]:
        """Haar detections for faces or plates in one BGR frame."""
        cls = concept_class(concept)
        if cls == "face":
            cascades, ellipse = ["haarcascade_frontalface_default.xml", "haarcascade_profileface.xml"], True
        elif cls == "plate":
            cascades, ellipse = ["haarcascade_russian_plate_number.xml"], False
        else:
            return []
        H, W = frame.shape[:2]
        scale = min(1.0, 640 / W)
        gray = cv2.equalizeHist(cv2.cvtColor(cv2.resize(frame, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY))
        dets: list[FrameMask] = []
        for name in cascades:
            rects, _, weights = self._cascade(name).detectMultiScale3(
                gray, scaleFactor=1.1, minNeighbors=min_neighbors, minSize=(18, 18), outputRejectLevels=True
            )
            for (x, y, w, h), wgt in zip(rects, np.ravel(weights) if len(rects) else []):
                box = tuple(int(v / scale) for v in (x, y, x + w, y + h))
                if any(box_iou(box, d.box) > 0.4 for d in dets):
                    continue
                bw, bh = box[2] - box[0], box[3] - box[1]
                m = np.zeros((bh, bw), dtype=np.uint8)
                if ellipse:
                    cv2.ellipse(m, (bw // 2, bh // 2), (bw // 2, int(bh * 0.6)), 0, 0, 360, 1, -1)
                else:
                    m[:] = 1
                score = float(np.clip(0.5 + float(wgt) / 20, 0.5, 0.99))
                fm = clip_mask(FrameMask(box, m.astype(bool), score), W, H)
                if fm:
                    dets.append(fm)
        return dets

    def _haar_tracks(self, video: VideoRecord, concept: str) -> list[Track]:
        if concept_class(concept) not in ("face", "plate"):
            return []
        stride = 2  # detect every other frame; fill_gaps bridges the rest
        per_frame = {
            i: self.detect_frame(frame, concept)
            for i, frame in enumerate(video_io.iter_frames(video.path))
            if i % stride == 0
        }
        tracks = link_detections(per_frame, concept, prefix=normalize_concept(concept).replace(" ", "_"), iou_min=0.2)
        return [fill_gaps(t, max_gap=stride * 3) for t in tracks]

    def supports(self, video: VideoRecord, concept: str) -> bool:
        return video.annotations is not None or concept_class(concept) in ("face", "plate")

    # -- Segmenter protocol --------------------------------------------------

    def segment(self, video: VideoRecord, concepts: list[str]) -> dict[str, list[Track]]:
        if video.annotations:
            return {c: self._gt_tracks(video, c, include_hard=False) for c in concepts}
        return {c: self._haar_tracks(video, c) for c in concepts}

    def refine(self, video: VideoRecord, requests: list[RefineRequest]) -> list[Track | None]:
        out: list[Track | None] = []
        for req in requests:
            if video.annotations:
                best, best_iou = None, 0.1
                for t in self._gt_tracks(video, req.concept, include_hard=True):
                    fm = t.frames.get(req.frame_index)
                    iou = box_iou(fm.box, req.box) if fm else 0.0
                    if iou > best_iou:
                        best, best_iou = t, iou
                out.append(best)
                continue
            # No ground truth: patch the flagged box for a short window, the way
            # an editor would cover a leak by hand.
            x0, y0, x1, y1 = (int(v) for v in req.box)
            win = max(3, int(video.meta.fps / 3))
            m = np.ones((y1 - y0, x1 - x0), dtype=bool)
            track = Track(f"patch-{req.frame_index}-{x0}-{y0}", req.concept)
            for f in range(max(0, req.frame_index - win), min(video.meta.frame_count, req.frame_index + win + 1)):
                fm = clip_mask(FrameMask((x0, y0, x1, y1), m, 0.5), video.meta.width, video.meta.height)
                if fm:
                    track.frames[f] = fm
            out.append(track if track.frames else None)
        return out
