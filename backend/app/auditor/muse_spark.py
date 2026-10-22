"""Muse Spark visual auditor over an OpenAI-compatible multimodal chat endpoint.

Each request carries a few sampled output frames. Intentionally visible
instances are boxed in green and labelled on the copy we send, so the model
can tell an exception from a leak. Boxes come back normalized to 0..1000.
"""

from __future__ import annotations

import base64
import json
import re
from concurrent.futures import ThreadPoolExecutor

import cv2
import httpx

from ..config import settings
from ..schemas import AuditFinding
from ..videos import VideoRecord
from .base import AuditPolicy, AuditSample

FRAMES_PER_REQUEST = 4
MAX_SIDE = 1024

PROMPT = """You are auditing redacted video frames for privacy leaks before publication.
Policy: every instance of these must be fully obscured: {targets}.
Regions outlined in green and labelled VISIBLE are intentionally left visible{kept}; ignore them.
For each image, report every policy item that is still clearly recognizable
(not blurred, pixelated, or blacked out). Partially hidden items count if a
person could still identify them.
Answer with JSON only:
{{"findings": [{{"image": <image number>, "concept": "<one of: {targets}>", "box": [x0, y0, x1, y1], "reason": "<short>"}}]}}
Boxes use 0-1000 normalized coordinates. If nothing leaked, answer {{"findings": []}}."""


class AuditorError(RuntimeError):
    pass


def _encode(sample: AuditSample) -> tuple[str, float]:
    img = sample.image.copy()
    for fm in sample.kept:
        x0, y0, x1, y1 = fm.box
        cv2.rectangle(img, (x0, y0), (x1, y1), (60, 200, 90), 3)
        cv2.putText(img, "VISIBLE", (x0, max(14, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 200, 90), 2)
    scale = min(1.0, MAX_SIDE / max(img.shape[:2]))
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise AuditorError("couldn't encode audit frame")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode(), scale


def _parse(content: str) -> list[dict]:
    match = re.search(r"\{.*\}", content or "", re.S)
    if not match:
        raise AuditorError(f"Muse Spark returned no JSON: {content[:200]!r}")
    return json.loads(match.group(0)).get("findings", [])


class MuseSparkAuditor:
    name = "Muse Spark"

    def _audit_batch(self, batch: list[AuditSample], policy: AuditPolicy, video: VideoRecord) -> list[AuditFinding]:
        kept = f" ({'; '.join(policy.kept_descriptions)})" if policy.kept_descriptions else ""
        content: list[dict] = [{"type": "text", "text": PROMPT.format(targets=", ".join(policy.targets), kept=kept)}]
        for i, s in enumerate(batch, start=1):
            url, _ = _encode(s)
            content.append({"type": "text", "text": f"Image {i} (t={s.time_s:.2f}s):"})
            content.append({"type": "image_url", "image_url": {"url": url}})
        with httpx.Client(timeout=settings.request_timeout_s) as client:
            resp = client.post(
                f"{settings.auditor_base}/chat/completions",
                headers={"Authorization": f"Bearer {settings.auditor_key}"},
                json={"model": settings.auditor_model, "messages": [{"role": "user", "content": content}], "temperature": 0},
            )
        if resp.status_code >= 400:
            raise AuditorError(f"Auditor API returned {resp.status_code}: {resp.text[:300]}")
        message = (resp.json().get("choices") or [{}])[0].get("message") or {}
        W, H = video.meta.width, video.meta.height
        out: list[AuditFinding] = []
        for item in _parse(message.get("content") or ""):
            idx = int(item.get("image", 0)) - 1
            if not 0 <= idx < len(batch) or len(item.get("box", [])) != 4:
                continue
            concept = str(item.get("concept", "")).strip().lower()
            if concept not in policy.targets:
                concept = next((t for t in policy.targets if t in concept or concept in t), None)
                if concept is None:
                    continue
            x0, y0, x1, y1 = (float(v) / 1000 for v in item["box"])
            box = (max(0, x0 * W), max(0, y0 * H), min(W, x1 * W), min(H, y1 * H))
            if box[2] - box[0] < 4 or box[3] - box[1] < 4:
                continue
            out.append(AuditFinding(frame_index=batch[idx].frame_index, concept=concept, box=box, reason=str(item.get("reason", ""))[:200]))
        return out

    def audit(self, video: VideoRecord, samples: list[AuditSample], policy: AuditPolicy) -> list[AuditFinding]:
        batches = [samples[i : i + FRAMES_PER_REQUEST] for i in range(0, len(samples), FRAMES_PER_REQUEST)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = pool.map(lambda b: self._audit_batch(b, policy, video), batches)
            return [f for batch in results for f in batch]
