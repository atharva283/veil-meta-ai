"""Render synthetic demo clips with per-frame ground truth.

These exist so the full Veil pipeline runs offline before SAM 3.1 is wired up:
the mock segmenter reads the ground truth instead of running a model. Each
clip gets three files in demo_assets/clips/:

    <name>.mp4                 the video (H.264)
    <name>.annotations.json    per-frame shapes for every trackable object
    <name>.json                title, description, suggested prompts

Tracks marked "hard" are ones SAM commonly misses on a first text-prompt pass
(a tiny distant face, a partly occluded face, a plate at the frame edge). The
mock segmenter withholds them so the audit loop has real leaks to catch.

Run with the backend venv:  backend/.venv/Scripts/python demo_assets/generate_demo_clips.py
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

OUT = Path(__file__).resolve().parent / "clips"
W, H, FPS = 960, 540, 24
FONT = cv2.FONT_HERSHEY_SIMPLEX

SKIN = [(160, 190, 230), (120, 160, 205), (90, 125, 175), (70, 100, 145), (140, 175, 220)]
HAIR = [(30, 30, 40), (40, 60, 90), (20, 20, 25), (60, 90, 140), (90, 110, 120)]
JACKETS = {
    "red": (45, 45, 200), "blue": (160, 95, 40), "green": (70, 140, 60), "yellow": (40, 190, 225),
    "gray": (120, 120, 120), "black": (40, 40, 45), "teal": (140, 140, 40), "purple": (130, 60, 110),
}  # fmt: skip


class Recorder:
    def __init__(self) -> None:
        self.tracks: dict[str, dict] = {}

    def add(self, tid: str, labels: list[str], frame: int, shapes: list[list], hard: bool = False) -> None:
        def visible(s):
            if s[0] == "e":
                return s[1] + s[3] > 0 and s[1] - s[3] < W and s[2] + s[4] > 0 and s[2] - s[4] < H
            if s[0] == "p":
                xs, ys = [p[0] for p in s[1]], [p[1] for p in s[1]]
                return max(xs) > 0 and min(xs) < W and max(ys) > 0 and min(ys) < H
            return s[3] > 0 and s[1] < W and s[4] > 0 and s[2] < H

        def rounded(s):
            if s[0] == "p":
                return ["p", [[round(float(x), 1), round(float(y), 1)] for x, y in s[1]]]
            return [s[0], *[round(float(v), 1) for v in s[1:]]]

        shapes = [rounded(s) for s in shapes if visible(s)]
        if not shapes:
            return
        t = self.tracks.setdefault(tid, {"id": tid, "labels": labels, "hard": hard, "frames": {}})
        t["frames"][str(frame)] = shapes


def rrect(img, x0, y0, x1, y1, r, color):
    x0, y0, x1, y1, r = int(x0), int(y0), int(x1), int(y1), int(max(1, r))
    cv2.rectangle(img, (x0 + r, y0), (x1 - r, y1), color, -1)
    cv2.rectangle(img, (x0, y0 + r), (x1, y1 - r), color, -1)
    for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
        cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)


def shade(c, f):
    return tuple(int(max(0, min(255, v * f))) for v in c)


def draw_person(img, rec, f, pid, x, feet, s, jacket, skin, hair, phase, hard_face=False,
                noun="person", phone=False, seated=False, badge=None):
    """Front-facing figure. Returns nothing; records person/face/jacket/phone/badge tracks."""
    head_cy = feet - (96 if seated else 134) * s
    rx, ry = 13 * s, 16 * s
    torso = (x - 22 * s, head_cy + 18 * s, x + 22 * s, head_cy + 76 * s)
    pants = (40, 45, 55)
    body_shapes = [["e", x, head_cy, rx, ry], ["r", torso[0] - 6 * s, torso[1], torso[2] + 6 * s, torso[3]]]
    if not seated:
        swing = math.sin(phase) * 8 * s
        for side, sw in ((-1, swing), (1, -swing)):
            hip = (int(x + side * 9 * s), int(torso[3]))
            foot = (int(x + side * 9 * s + sw), int(feet))
            cv2.line(img, hip, foot, pants, max(2, int(11 * s)), cv2.LINE_AA)
            cv2.ellipse(img, (foot[0] + int(3 * s), foot[1]), (int(7 * s), int(3 * s)), 0, 0, 360, (30, 30, 30), -1)
        body_shapes.append(["r", x - 18 * s - abs(swing), torso[3], x + 18 * s + abs(swing), feet + 3 * s])
    # torso + arms
    rrect(img, *torso, 9 * s, jacket)
    arm = math.sin(phase + math.pi) * 6 * s if not seated else 0
    for side in (-1, 1):
        sh = (int(x + side * 20 * s), int(torso[1] + 8 * s))
        hand = (int(x + side * 26 * s + side * arm * 0.3), int(torso[1] + 52 * s + arm))
        if phone and side == 1:
            hand = (int(x + 6 * s), int(torso[1] + 22 * s))
        cv2.line(img, sh, hand, shade(jacket, 0.82), max(2, int(9 * s)), cv2.LINE_AA)
        cv2.circle(img, hand, max(2, int(4.5 * s)), skin, -1, cv2.LINE_AA)
    cv2.line(img, (int(x), int(torso[1] + 4 * s)), (int(x), int(torso[3] - 4 * s)), shade(jacket, 0.7), max(1, int(2 * s)))
    # head
    cv2.rectangle(img, (int(x - 5 * s), int(head_cy + 12 * s)), (int(x + 5 * s), int(torso[1] + 3 * s)), shade(skin, 0.9), -1)
    cv2.ellipse(img, (int(x), int(head_cy)), (int(rx), int(ry)), 0, 0, 360, skin, -1, cv2.LINE_AA)
    cv2.ellipse(img, (int(x), int(head_cy - 6 * s)), (int(rx + 1.5 * s), int(ry * 0.72)), 0, 180, 360, hair, -1, cv2.LINE_AA)
    if noun == "woman":
        cv2.ellipse(img, (int(x), int(head_cy + 2 * s)), (int(rx + 3 * s), int(ry + 4 * s)), 0, 160, 380, hair, max(2, int(5 * s)), cv2.LINE_AA)
    for ex in (-5, 5):
        cv2.circle(img, (int(x + ex * s), int(head_cy + 1 * s)), max(1, int(1.6 * s)), (35, 35, 35), -1, cv2.LINE_AA)
    cv2.ellipse(img, (int(x), int(head_cy + 8 * s)), (int(4 * s), max(1, int(1.5 * s))), 0, 0, 180, (60, 60, 120), max(1, int(1.2 * s)))

    color_name = next(k for k, v in JACKETS.items() if v == jacket)
    rec.add(f"person-{pid}", ["person", noun, "pedestrian"], f, body_shapes)
    rec.add(f"face-{pid}", ["face", "head"], f, [["e", x, head_cy + 1 * s, rx + 1 * s, ry + 1 * s]], hard=hard_face)
    rec.add(f"jacket-{pid}", ["jacket", f"{color_name} jacket", "coat", f"{color_name} coat", "clothing", f"{color_name} clothing"], f,
            [["r", torso[0], torso[1], torso[2], torso[3]]])
    if phone:
        px0, py0 = x - 4 * s, torso[1] + 4 * s
        px1, py1 = px0 + 20 * s, py0 + 34 * s
        rrect(img, px0 - 2 * s, py0 - 2 * s, px1 + 2 * s, py1 + 2 * s, 3 * s, (25, 25, 30))
        cv2.rectangle(img, (int(px0), int(py0)), (int(px1), int(py1)), (250, 235, 200), -1)
        for k in range(4):
            yy = int(py0 + (6 + k * 7) * s)
            cv2.line(img, (int(px0 + 3 * s), yy), (int(px1 - (3 + 5 * (k % 2)) * s), yy), (170, 120, 60), max(1, int(1.5 * s)))
        rec.add(f"phone-{pid}", ["phone screen", "screen", "phone"], f, [["r", px0, py0, px1, py1]])
    if badge:
        bx0, by0 = x - 12 * s, torso[1] + 22 * s
        cv2.line(img, (int(x - 10 * s), int(torso[1])), (int(x), int(by0)), (40, 40, 160), max(1, int(2 * s)))
        cv2.line(img, (int(x + 10 * s), int(torso[1])), (int(x), int(by0)), (40, 40, 160), max(1, int(2 * s)))
        cv2.rectangle(img, (int(bx0), int(by0)), (int(bx0 + 24 * s), int(by0 + 16 * s)), (245, 245, 245), -1)
        cv2.rectangle(img, (int(bx0), int(by0)), (int(bx0 + 24 * s), int(by0 + 5 * s)), (40, 40, 160), -1)
        cv2.putText(img, badge, (int(bx0 + 2 * s), int(by0 + 13 * s)), FONT, 0.22 * s, (30, 30, 30), 1, cv2.LINE_AA)
        rec.add(f"badge-{pid}", ["name badge", "badge"], f, [["r", bx0, by0, bx0 + 24 * s, by0 + 16 * s]])


def draw_vehicle(img, rec, f, vid, x, base, w, color, plate_text, truck=False, hard_plate=False, color_name="white"):
    plate_w, plate_h = 30, 13
    if truck:
        box = (x, base - 92, x + w * 0.72, base - 16)
        rrect(img, *box, 4, (235, 235, 230))
        cv2.ellipse(img, (int(x + w * 0.36), int(base - 56)), (24, 24), 0, 0, 360, (60, 140, 230), -1, cv2.LINE_AA)
        cv2.putText(img, "ACME", (int(x + w * 0.36 - 20), int(base - 51)), FONT, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
        cab = (x + w * 0.74, base - 64, x + w, base - 16)
        rrect(img, *cab, 6, color)
        cv2.rectangle(img, (int(cab[0] + 8), int(cab[1] + 8)), (int(cab[2] - 6), int(cab[1] + 28)), (220, 200, 160), -1)
        rec.add(f"truck-{vid}", ["truck", "vehicle", "van"], f, [["r", *box], ["r", *cab]])
        rec.add(f"logo-{vid}", ["logo", "brand logo"], f, [["e", x + w * 0.36, base - 56, 26, 26]])
    else:
        body = (x, base - 40, x + w, base - 14)
        cabin = [[x + w * 0.22, base - 40], [x + w * 0.32, base - 64], [x + w * 0.68, base - 64], [x + w * 0.8, base - 40]]
        cv2.fillPoly(img, [np.array(cabin, dtype=np.int32)], shade(color, 0.92), cv2.LINE_AA)
        rrect(img, *body, 8, color)
        win = [[x + w * 0.27, base - 42], [x + w * 0.34, base - 60], [x + w * 0.66, base - 60], [x + w * 0.75, base - 42]]
        cv2.fillPoly(img, [np.array(win, dtype=np.int32)], (215, 195, 160), cv2.LINE_AA)
        cv2.line(img, (int(x + w * 0.5), int(base - 60)), (int(x + w * 0.5), int(base - 42)), shade(color, 0.92), 3)
        rec.add(f"car-{vid}", ["car", "vehicle", f"{color_name} car"], f, [["r", *body], ["p", cabin]])
    for wx in (x + w * 0.18, x + w * 0.82):
        cv2.circle(img, (int(wx), int(base - 12)), 12, (30, 30, 30), -1, cv2.LINE_AA)
        cv2.circle(img, (int(wx), int(base - 12)), 5, (150, 150, 150), -1, cv2.LINE_AA)
    px = x + w - plate_w - 4
    py = base - 34
    cv2.rectangle(img, (int(px), int(py)), (int(px + plate_w), int(py + plate_h)), (245, 245, 245), -1)
    cv2.rectangle(img, (int(px), int(py)), (int(px + plate_w), int(py + plate_h)), (60, 60, 60), 1)
    cv2.putText(img, plate_text, (int(px + 2), int(py + 10)), FONT, 0.26, (40, 40, 40), 1, cv2.LINE_AA)
    rec.add(f"plate-{vid}", ["license plate", "plate", "number plate"], f, [["r", px, py, px + plate_w, py + plate_h]], hard=hard_plate)


def finish(img, rng, overlay: str | None, t: float):
    noise = rng.normal(0, 4.0, img.shape).astype(np.int16)
    img[:] = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    if overlay:
        stamp = f"2026-10-21  14:32:{7 + int(t):02d}"
        cv2.putText(img, overlay, (16, 28), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, stamp, (W - 222, 28), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------


def street_background() -> np.ndarray:
    bg = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(230):
        k = y / 230
        bg[y] = (int(235 - 30 * k), int(215 - 20 * k), int(170 - 10 * k))
    rng = random.Random(4)
    x = -20
    palette = [(95, 105, 160), (130, 140, 150), (150, 170, 190), (80, 90, 110), (110, 130, 175)]
    while x < W:
        bw, bh = rng.randint(90, 170), rng.randint(110, 210)
        c = rng.choice(palette)
        cv2.rectangle(bg, (x, 262 - bh), (x + bw, 262), c, -1)
        for wy in range(262 - bh + 14, 250, 26):
            for wx in range(x + 12, x + bw - 16, 24):
                lit = rng.random() < 0.25
                cv2.rectangle(bg, (wx, wy), (wx + 12, wy + 14), (170, 220, 240) if lit else shade(c, 0.6), -1)
        x += bw + rng.randint(0, 8)
    cv2.rectangle(bg, (0, 262), (W, 284), (175, 180, 185), -1)  # far sidewalk
    cv2.rectangle(bg, (0, 284), (W, 384), (75, 75, 80), -1)  # road
    for lx in range(0, W, 70):
        cv2.rectangle(bg, (lx, 332), (lx + 38, 336), (210, 210, 210), -1)
    cv2.rectangle(bg, (0, 384), (W, 392), (140, 145, 150), -1)  # curb
    cv2.rectangle(bg, (0, 392), (W, H), (165, 175, 185), -1)  # near sidewalk
    for gx in range(0, W, 80):
        cv2.line(bg, (gx, 392), (gx - 40, H), (150, 160, 170), 1)
    return bg


def make_street(rec: Recorder, frames: int):
    bg = street_background()
    rng = np.random.default_rng(1)
    people = [
        # pid, start x, speed px/frame, feet y, scale, jacket, noun, extras
        dict(pid="01", x=-40, v=2.6, feet=505, s=0.86, jacket="blue", noun="man"),
        dict(pid="02", x=980, v=-2.2, feet=532, s=1.0, jacket="red", noun="woman"),
        dict(pid="03", x=150, v=1.7, feet=532, s=1.0, jacket="gray", noun="man", phone=True),
        dict(pid="04", x=760, v=-1.5, feet=505, s=0.86, jacket="green", noun="woman"),
        dict(pid="05", x=-260, v=2.9, feet=532, s=1.0, jacket="yellow", noun="man"),
        dict(pid="06", x=1180, v=-2.4, feet=505, s=0.86, jacket="teal", noun="woman", phone=True),
        # tiny pedestrian on the far sidewalk: the face a first pass misses
        dict(pid="07", x=820, v=-0.9, feet=282, s=0.3, jacket="purple", noun="man", hard=True),
    ]
    cars = [
        dict(vid="01", x=-200, v=4.2, base=330, w=150, color=(60, 60, 190), name="red", plate="7KLM482"),
        dict(vid="02", x=1100, v=-3.6, base=378, w=160, color=(200, 200, 200), name="white", plate="4TRV915"),
        dict(vid="03", x=-700, v=3.4, base=330, w=230, color=(180, 120, 40), name="blue", plate="2BXA330", truck=True),
        dict(vid="04", x=-1250, v=4.0, base=378, w=150, color=(40, 40, 40), name="black", plate="9QWE771"),
    ]
    for f in range(frames):
        img = bg.copy()
        # parked car at the right edge, mostly out of frame: its plate is "hard"
        draw_vehicle(img, rec, f, "05", 822, 378, 150, (90, 160, 90), "5MNP208", hard_plate=True, color_name="green")
        for c in sorted(cars, key=lambda c: c["base"]):
            x = c["x"] + c["v"] * f
            draw_vehicle(img, rec, f, c["vid"], x, c["base"], c["w"], c["color"], c["plate"], truck=c.get("truck", False), color_name=c["name"])
        for p in sorted(people, key=lambda p: p["feet"]):
            x = p["x"] + p["v"] * f
            k = int(p["pid"])
            draw_person(img, rec, f, p["pid"], x, p["feet"], p["s"], JACKETS[p["jacket"]], SKIN[k % 5], HAIR[k % 5],
                        phase=f * 0.32 * (1 if p["v"] > 0 else -1) + k, hard_face=p.get("hard", False),
                        noun=p["noun"], phone=p.get("phone", False))
        finish(img, rng, "CAM 02  MARKET ST", f / FPS)
        yield img


def make_event(rec: Recorder, frames: int):
    bg = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        k = y / H
        bg[y] = (int(70 + 30 * k), int(58 + 22 * k), int(52 + 18 * k))
    cv2.rectangle(bg, (0, 300), (W, H), (60, 75, 95), -1)  # floor
    cv2.rectangle(bg, (90, 34), (520, 262), (250, 250, 248), -1)  # projector screen
    cv2.putText(bg, "Community Safety Meetup", (118, 84), FONT, 0.9, (90, 60, 40), 2, cv2.LINE_AA)
    for i, line in enumerate(["Updates from the city council", "Street lighting survey", "Q&A with residents"]):
        cv2.circle(bg, (130, 128 + i * 38), 5, (200, 140, 60), -1)
        cv2.putText(bg, line, (146, 134 + i * 38), FONT, 0.62, (60, 60, 60), 1, cv2.LINE_AA)
    rng = np.random.default_rng(2)
    audience_back = [("11", 90, "black"), ("12", 250, "teal"), ("13", 410, "gray"), ("14", 570, "purple"), ("15", 730, "blue"), ("16", 880, "yellow")]
    audience_front = [("21", 170, "gray", "SAM P."), ("22", 340, "blue", "MAYA R."), ("23", 530, "black", "LEO K."), ("24", 720, "teal", "ANA V.")]
    for f in range(frames):
        img = bg.copy()
        # podium + laptop with a private inbox on screen
        cv2.rectangle(img, (760, 250), (870, 360), (50, 70, 100), -1)
        cv2.rectangle(img, (772, 214), (858, 250), (40, 40, 40), -1)
        cv2.rectangle(img, (776, 218), (854, 246), (240, 225, 200), -1)
        for k in range(3):
            cv2.line(img, (780, 224 + k * 8), (848 - k * 14, 224 + k * 8), (150, 110, 70), 2)
        rec.add("laptop-01", ["laptop screen", "screen", "laptop", "monitor"], f, [["r", 776, 218, 854, 246]])
        # presenter in the green jacket walks across the stage
        px = 640 + 100 * math.sin(f / 30)
        draw_person(img, rec, f, "30", px, 352, 1.0, JACKETS["green"], SKIN[2], HAIR[0], phase=f * 0.25, noun="woman")
        for pid, x, jacket in audience_back:
            sway = 3 * math.sin(f / 15 + int(pid))
            hard = pid == "13"
            draw_person(img, rec, f, pid, x + sway, 470, 0.92, JACKETS[jacket], SKIN[int(pid) % 5], HAIR[int(pid) % 5],
                        phase=0, seated=True, hard_face=hard, noun="person",
                        phone=(pid == "15" and 30 < f < 150))
        # person 13 sits in the shadow under the screen: a low-contrast face (hard case)
        sx = int(410 + 3 * math.sin(f / 15 + 13))
        shadow = np.zeros((H, W), dtype=np.float32)
        cv2.ellipse(shadow, (sx, 384), (34, 40), 0, 0, 360, 1.0, -1)
        shadow = cv2.GaussianBlur(shadow, (0, 0), 9)[..., None]
        img[:] = (img * (1 - 0.55 * shadow)).astype(np.uint8)
        for pid, x, jacket, name in audience_front:
            sway = 2 * math.sin(f / 18 + int(pid))
            draw_person(img, rec, f, pid, x + sway, 575, 1.15, JACKETS[jacket], SKIN[int(pid) % 5], HAIR[int(pid) % 5],
                        phase=0, seated=True, noun="person", badge=name, phone=(pid == "23" and f > 60))
        finish(img, rng, None, f / FPS)
        yield img


def write_clip(name: str, gen, rec: Recorder, frames: int, meta: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(OUT / f"{name}.mp4"), (W, H), pix_fmt_in="bgr24", fps=FPS, codec="libx264", quality=None,
        macro_block_size=2, output_params=["-preset", "medium", "-crf", "20", "-movflags", "+faststart"],
    )
    writer.send(None)
    for i, img in enumerate(gen):
        writer.send(np.ascontiguousarray(img))
        if i == int(frames * 0.3):
            cv2.imwrite(str(OUT / f"{name}.jpg"), cv2.resize(img, (480, 270), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 85])
    writer.close()
    ann = {"width": W, "height": H, "fps": FPS, "frame_count": frames, "tracks": list(rec.tracks.values())}
    (OUT / f"{name}.annotations.json").write_text(json.dumps(ann, separators=(",", ":")), encoding="utf-8")
    (OUT / f"{name}.json").write_text(json.dumps({**meta, "frame_count": frames}, indent=2), encoding="utf-8")
    hard = [t["id"] for t in rec.tracks.values() if t["hard"]]
    print(f"{name}: {frames} frames, {len(rec.tracks)} tracks, hard={hard}")


def main() -> None:
    frames = FPS * 8
    rec = Recorder()
    write_clip("street_crossing", make_street(rec, frames), rec, frames, {
        "title": "Market Street crossing",
        "order": 1,
        "description": "CCTV-style street clip: pedestrians, passing cars and a delivery truck, two people on their phones.",
        "suggested_prompts": [
            "Blur every face and license plate except the woman in the red jacket, and black out phone screens.",
            "Pixelate all faces and black out license plates.",
        ],
        "follow_up_prompts": ["Also hide the logo on the truck."],
    })
    rec = Recorder()
    write_clip("community_meetup", make_event(rec, frames), rec, frames, {
        "title": "Community meetup",
        "order": 2,
        "description": "Event footage: a speaker on stage, an audience with name badges, phones recording, a laptop showing an inbox.",
        "suggested_prompts": [
            "Blur every face and name badge except the speaker in the green jacket, and black out the laptop screen.",
            "Black out all screens and pixelate the badges.",
        ],
        "follow_up_prompts": ["Also black out the phone screens."],
    })


if __name__ == "__main__":
    main()
