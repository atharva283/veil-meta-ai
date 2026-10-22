"""HTTP client for the SAM 3.1 service on Modal (backend/modal_app/sam3_service.py).

Wire format (both endpoints answer with the same shape):

    {"tracked": true,
     "results": [{"id": "p0", "objects": [
         {"obj_id": 3, "detections": [
             {"f": 12, "score": 0.91, "box": [x0, y0, x1, y1], "rle": [z, o, z, o, ...]}]}]}]}

`rle` is row-major run lengths over the box crop, starting with a run of zeros
(see pipeline.tracks.rle_encode). When "tracked" is false the detections are
per-frame and the client links them by IoU.
"""

from __future__ import annotations

import json

import httpx

from ..config import settings
from ..pipeline.tracks import FrameMask, Track, clip_mask, fill_gaps, link_detections, rle_decode
from ..videos import VideoRecord
from .base import RefineRequest, normalize_concept


class SegmenterError(RuntimeError):
    pass


def _decode_result(result: dict, concept: str, tracked: bool, width: int, height: int) -> list[Track]:
    prefix = normalize_concept(concept).replace(" ", "_")
    per_frame: dict[int, list[FrameMask]] = {}
    tracks: list[Track] = []
    for obj in result.get("objects", []):
        track = Track(f"{prefix}-{obj.get('obj_id', len(tracks) + 1)}", concept)
        for det in obj.get("detections", []):
            x0, y0, x1, y1 = (int(round(v)) for v in det["box"])
            if x1 <= x0 or y1 <= y0:
                continue
            fm = clip_mask(FrameMask((x0, y0, x1, y1), rle_decode(det["rle"], (y1 - y0, x1 - x0)), float(det.get("score", 1.0))), width, height)
            if fm is None:
                continue
            if tracked:
                track.frames[int(det["f"])] = fm
            else:
                per_frame.setdefault(int(det["f"]), []).append(fm)
        if tracked and track.frames:
            tracks.append(fill_gaps(track, max_gap=4))
    if not tracked:
        tracks = [fill_gaps(t, max_gap=4) for t in link_detections(per_frame, concept, prefix)]
    return tracks


class ModalSamSegmenter:
    name = "SAM 3.1 on Modal"

    def _post(self, route: str, video: VideoRecord, payload: dict) -> dict:
        headers = {"Authorization": f"Bearer {settings.modal_auth_token}"} if settings.modal_auth_token else {}
        with video.path.open("rb") as fh, httpx.Client(timeout=settings.request_timeout_s * 3) as client:
            resp = client.post(
                f"{settings.modal_endpoint}/{route}",
                headers=headers,
                files={"video": (video.path.name, fh, "video/mp4")},
                data={"payload": json.dumps(payload)},
            )
        if resp.status_code >= 400:
            raise SegmenterError(f"SAM service /{route} returned {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    def segment(self, video: VideoRecord, concepts: list[str]) -> dict[str, list[Track]]:
        prompts = [{"id": f"p{i}", "text": c} for i, c in enumerate(concepts)]
        data = self._post("segment", video, {"prompts": prompts, "score_threshold": 0.45})
        by_id = {r["id"]: r for r in data.get("results", [])}
        tracked = bool(data.get("tracked", True))
        W, H = video.meta.width, video.meta.height
        return {c: _decode_result(by_id.get(f"p{i}", {}), c, tracked, W, H) for i, c in enumerate(concepts)}

    def refine(self, video: VideoRecord, requests: list[RefineRequest]) -> list[Track | None]:
        if not requests:
            return []
        payload = {
            "requests": [
                {"id": f"r{i}", "text": r.concept, "frame_index": r.frame_index, "box": list(r.box), "window": int(video.meta.fps * 2)}
                for i, r in enumerate(requests)
            ]
        }
        data = self._post("refine", video, payload)
        by_id = {r["id"]: r for r in data.get("results", [])}
        tracked = bool(data.get("tracked", True))
        out: list[Track | None] = []
        for i, req in enumerate(requests):
            tracks = _decode_result(by_id.get(f"r{i}", {}), req.concept, tracked, video.meta.width, video.meta.height)
            for t in tracks:
                t.track_id = f"patch-{t.track_id}-{req.frame_index}"
            out.append(max(tracks, key=lambda t: len(t.frames)) if tracks else None)
        return out
