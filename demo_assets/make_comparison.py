"""Render a before/after comparison video from a finished Veil job.

    backend/.venv/Scripts/python demo_assets/make_comparison.py \
        --original demo_assets/clips/manchester_street.mp4 \
        --job backend/data/jobs/<job id> --out comparison.mp4

With the job's masks.mp4 present the output is a 1920x1080 grid:

    +----------------+----------------+
    | Original       | Veil output    |
    +----------------+----------------+
    | Masks          | Instruction,   |
    | (segmenter)    | plan, audit    |
    +----------------+----------------+

`--layout pair` drops the bottom row (Original | Veil output, 1920x540 + caption
bar). Panel labels name the model modes recorded in the job's audit.json, so a
run on mocks is labelled as one. Audit patches get an on-screen callout from
just before the moment they were caught. Audio comes from the redacted video.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from app.pipeline import video_io  # noqa: E402
from app.schemas import Plan  # noqa: E402

FONT = cv2.FONT_HERSHEY_DUPLEX
BG, FG, MUTED = (28, 24, 22), (240, 240, 240), (160, 160, 160)
CORAL, GREEN, AMBER = (102, 122, 255), (155, 211, 111), (61, 179, 242)  # BGR, as in pipeline.effects
PW, PH = 960, 540  # panel size


def text(img, s, org, scale=0.6, color=FG, thick=1):
    s = s.replace("∩", "&").replace("−", "-")  # Hershey fonts are ASCII-only (plan labels use both)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def label(img, s, color=FG):
    (tw, th), _ = cv2.getTextSize(s, FONT, 0.55, 1)
    cv2.rectangle(img, (12, 12), (12 + tw + 16, 12 + th + 14), (0, 0, 0), -1)
    text(img, s, (20, 12 + th + 6), 0.55, color)


def fit(frame, w=PW, h=PH):
    """Letterbox a frame into a w x h panel."""
    scale = min(w / frame.shape[1], h / frame.shape[0])
    small = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    panel = np.zeros((h, w, 3), np.uint8)
    y, x = (h - small.shape[0]) // 2, (w - small.shape[1]) // 2
    panel[y : y + small.shape[0], x : x + small.shape[1]] = small
    return panel


def callouts(audit: dict, t: float) -> list[str]:
    """Patches caught near time t, shown from 0.25 s before to 2.5 s after."""
    return [
        f"Audit caught a missed {p['concept']} at {p['at']:.2f}s -> re-segmented and redacted"
        for p in audit.get("patches", [])
        if p["at"] - 0.25 <= t <= p["at"] + 2.5
    ]


def info_panel(audit: dict, plan: Plan, t: float, duration: float) -> np.ndarray:
    img = np.full((PH, PW, 3), BG, np.uint8)
    modes = audit.get("modes", {})
    text(img, "VEIL", (28, 46), 1.0, FG, 2)
    text(img, f"{t:5.2f}s / {duration:.2f}s", (PW - 210, 46), 0.6, MUTED)
    y = 84
    for line in textwrap.wrap(f'"{audit["instruction"]}"', 70)[:3]:
        text(img, line, (28, y), 0.6, AMBER)
        y += 26
    y += 8
    text(img, f"Plan ({modes.get('planner', 'planner')}):", (28, y), 0.5, MUTED)
    y += 24
    steps = plan.steps[:8]
    for i, s in enumerate(steps, start=1):
        text(img, f"{i}. {s.label}" + (f"  - {s.note}" if s.note else ""), (40, y), 0.5)
        y += 22
    if len(plan.steps) > len(steps):
        text(img, f"... {len(plan.steps) - len(steps)} more", (40, y), 0.5, MUTED)
        y += 22
    c = audit.get("counts", {})
    y += 10
    text(img, f"Redacted {c.get('verified', 0) + c.get('patched', 0)}   Patched by audit {c.get('patched', 0)}   "
              f"Kept visible {c.get('kept_visible', 0)}   Needs review {c.get('flagged', 0)}", (28, y), 0.5)  # fmt: skip
    y += 24
    audit_line = f"Audit ({modes.get('auditor', 'auditor')}): {audit.get('audited_frames', 0)} sampled frames"
    if audit.get("rechecked_frames"):
        pct = round(100 * audit["rechecked_frames"] / max(1, audit.get("frame_count") or audit["rechecked_frames"]))
        audit_line += f", re-segmented {audit['rechecked_frames']} frames ({pct}%)"
    text(img, audit_line, (28, y), 0.5)
    y = PH - 24
    for msg in reversed(callouts(audit, t)[-3:]):
        cv2.rectangle(img, (20, y - 22), (PW - 20, y + 8), CORAL, -1)
        text(img, msg, (30, y), 0.5, (20, 20, 20))
        y -= 36
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--original", type=Path, required=True)
    ap.add_argument("--job", type=Path, required=True, help="job folder with redacted.mp4, masks.mp4, audit.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layout", choices=["grid", "pair"], default="grid")
    args = ap.parse_args()

    audit = json.loads((args.job / "audit.json").read_text(encoding="utf-8"))
    plan = Plan(**audit["plan"])
    modes = audit.get("modes", {})
    redacted = args.job / "redacted.mp4"
    masks = args.job / "masks.mp4"
    grid = args.layout == "grid" and masks.exists()

    src = video_io.probe(args.original)
    out_h = 2 * PH if grid else PH + 90
    meta = video_io.VideoMeta(2 * PW, out_h, src.fps, src.frame_count, video_io.probe(redacted).has_audio)
    writer = video_io.H264Writer(args.out, meta, audio_from=redacted)
    streams = [video_io.iter_frames(args.original), video_io.iter_frames(redacted)]
    if grid:
        streams.append(video_io.iter_frames(masks))
    duration = src.frame_count / src.fps
    n = 0
    try:
        for i, frames in enumerate(zip(*streams)):
            t = i / src.fps
            before, after = fit(frames[0]), fit(frames[1])
            label(before, "ORIGINAL")
            label(after, "VEIL OUTPUT", GREEN)
            top = np.hstack([before, after])
            if grid:
                m = fit(frames[2])
                label(m, f"MASKS - {modes.get('segmenter', 'segmenter')}  (coral = caught by audit, green = kept visible)")
                canvas = np.vstack([top, np.hstack([m, info_panel(audit, plan, t, duration)])])
            else:
                bar = np.full((90, 2 * PW, 3), BG, np.uint8)
                text(bar, f'"{textwrap.shorten(audit["instruction"], 110)}"', (24, 36), 0.65, AMBER)
                msgs = callouts(audit, t)
                text(bar, msgs[-1] if msgs else f"{t:5.2f}s", (24, 72), 0.6, CORAL if msgs else MUTED)
                canvas = np.vstack([top, bar])
            writer.write(canvas)
            n += 1
    finally:
        writer.close()
    print(f"wrote {args.out} ({n} frames, {'grid' if grid else 'pair'} layout)")


if __name__ == "__main__":
    main()
