"""Per-instance redaction effects and the mask-preview overlay."""

from __future__ import annotations

import hashlib

import cv2
import numpy as np

from .tracks import FrameMask

# BGR colors per concept class; the frontend uses the same hues.
CLASS_COLORS = {
    "face": (61, 179, 242),  # amber
    "plate": (217, 196, 94),  # cyan
    "screen": (242, 140, 180),  # violet
    "person": (230, 160, 90),  # blue
    "kept": (155, 211, 111),  # green: intentionally visible
    "patched": (102, 122, 255),  # coral: added by the audit loop
}
_FALLBACK = [(120, 200, 230), (200, 200, 120), (160, 140, 230), (110, 190, 170)]


def concept_class(concept: str) -> str:
    c = concept.lower()
    if "face" in c or "head" in c:
        return "face"
    if "plate" in c:
        return "plate"
    if any(k in c for k in ("screen", "monitor", "laptop", "phone", "display")):
        return "screen"
    if any(k in c for k in ("person", "people", "woman", "man", "child")):
        return "person"
    return "other"


def concept_color(concept: str) -> tuple[int, int, int]:
    cls = concept_class(concept)
    if cls in CLASS_COLORS:
        return CLASS_COLORS[cls]
    digest = int(hashlib.md5(concept.encode()).hexdigest(), 16)
    return _FALLBACK[digest % len(_FALLBACK)]


def _region(width: int, height: int, fm: FrameMask):
    """The padded, dilated area an effect covers around a mask (masks are always clipped to the frame)."""
    bw, bh = fm.box[2] - fm.box[0], fm.box[3] - fm.box[1]
    pad = max(4, int(0.12 * max(bw, bh)))
    x0, y0, x1, y1 = fm.box
    X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(width, x1 + pad), min(height, y1 + pad)
    local = np.zeros((Y1 - Y0, X1 - X0), dtype=np.uint8)
    local[y0 - Y0 : y1 - Y0, x0 - X0 : x1 - X0] = fm.mask.astype(np.uint8) * 255
    k = max(3, (pad // 2) * 2 + 1)
    local = cv2.dilate(local, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return (X0, Y0, X1, Y1), local, pad


def obscured_mask(fm: FrameMask, effect: str, width: int, height: int) -> FrameMask | None:
    """The pixels apply_effect actually hides for this instance (None for highlight)."""
    if effect == "highlight" or fm.box[2] <= fm.box[0] or fm.box[3] <= fm.box[1]:
        return None
    box, local, _ = _region(width, height, fm)
    return FrameMask(box, local > 0, fm.score)


def apply_effect(frame: np.ndarray, fm: FrameMask, effect: str) -> None:
    """Redact one instance in place. Masks are dilated so edges never leak."""
    bw, bh = fm.box[2] - fm.box[0], fm.box[3] - fm.box[1]
    if bw <= 0 or bh <= 0:
        return
    if effect == "highlight":
        _outline(frame, fm, (255, 255, 255), 2)
        return
    (X0, Y0, X1, Y1), local, pad = _region(frame.shape[1], frame.shape[0], fm)
    region = frame[Y0:Y1, X0:X1]
    rh, rw = region.shape[:2]

    if effect == "blackout":
        region[local > 0] = 0
        return
    if effect == "pixelate":
        block = max(6, max(bw, bh) // 8)
        small = cv2.resize(region, (max(1, rw // block), max(1, rh // block)), interpolation=cv2.INTER_AREA)
        processed = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
        region[local > 0] = processed[local > 0]
        return

    # Blur: downsample-blur-upsample is both strong and cheap; feather the edge.
    factor = max(4, max(bw, bh) // 10)
    small = cv2.resize(region, (max(1, rw // factor), max(1, rh // factor)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigmaX=1.5)
    processed = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_LINEAR)
    alpha = cv2.GaussianBlur(local, (0, 0), sigmaX=max(1.0, pad / 3)).astype(np.float32)[..., None] / 255.0
    alpha = np.maximum(alpha, (local > 0)[..., None].astype(np.float32))
    region[:] = (processed * alpha + region * (1 - alpha)).astype(np.uint8)


def _outline(frame: np.ndarray, fm: FrameMask, color, thickness: int) -> None:
    contours, _ = cv2.findContours(fm.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, color, thickness, lineType=cv2.LINE_AA, offset=fm.box[:2])


def draw_preview(frame: np.ndarray, fm: FrameMask, label: str, color, filled: bool = True) -> None:
    """Translucent mask + outline + tag, used for the 'Masks' view."""
    x0, y0, x1, y1 = fm.box
    if filled:
        region = frame[y0:y1, x0:x1]
        tint = np.empty_like(region)
        tint[:] = color
        m = fm.mask[: region.shape[0], : region.shape[1]]
        region[m] = (region[m] * 0.5 + tint[m] * 0.5).astype(np.uint8)
    _outline(frame, fm, color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    ty = max(th + 6, y0 - 4)
    cv2.rectangle(frame, (x0, ty - th - 6), (x0 + tw + 8, ty + 2), color, -1)
    cv2.putText(frame, label, (x0 + 4, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (24, 24, 24), 1, cv2.LINE_AA)
