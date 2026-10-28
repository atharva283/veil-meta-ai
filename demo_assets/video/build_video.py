"""Build the 3-minute demo video: Gemini TTS narration over recorded runs, UI capture and cards.

    demo_assets/video/.venv/Scripts/python demo_assets/video/build_video.py
    ... --no-voice        silent draft, each scene timed from its word count (no API key needed)
    ... --voice Charon    another Gemini prebuilt voice (default Kore)

Needs GEMINI_API_KEY in the repo's .env (Google AI Studio). Output in demo_assets/video/build/:
final_submission.mp4 (1920x1080, 30 fps, under 180 s or the build fails) and final_submission.srt.

Every scene is voiced separately and lasts as long as its narration, so picture and sound
can't drift; narration audio is cached by its text, so re-runs only pay for changed lines.
The UI scenes are recorded headlessly from the live replay demo.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import imageio_ffmpeg
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "demo_assets" / "video" / "build"
CACHE = ROOT / "demo_assets" / "cache"
CLIPS = ROOT / "demo_assets" / "clips"
SITE = "https://atharva283--veil-demo-web.us-east.modal.run"
REPO = "github.com/atharva283/veil-meta-ai"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
W, H, FPS = 1920, 1080, 30
MAX_SECONDS = 179.5
LEAD, TAIL = 0.35, 0.65  # silence before and after each scene's narration

MAIN, FOLLOW = CACHE / "f1033d7cf4b3", CACHE / "df14e93c192b"  # Manchester: pink-coat run, bank-logo follow-up
DESK, DESK_FOLLOW = CACHE / "f57a67578775", CACHE / "6ba8326bb9dc"

BG, PANEL, LINE = (21, 24, 29), (32, 37, 44), (58, 65, 75)
TEXT, MUTED, FAINT = (236, 232, 224), (164, 172, 182), (118, 128, 139)
CORAL, GREEN, AMBER = (255, 112, 86), (72, 200, 130), (240, 180, 70)
FONTS = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"


def font(size: int, weight: str = "regular") -> ImageFont.FreeTypeFont:
    name = {"regular": "segoeui.ttf", "semibold": "seguisb.ttf", "bold": "segoeuib.ttf", "mono": "consola.ttf"}[weight]
    for path in (FONTS / name, Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size)


# -- scenes -------------------------------------------------------------------


@dataclass
class Scene:
    """One narration request, over one or more visuals.

    A visual is ("clip", path, start, speed, title, kicker) | ("card", name) | ("ui", name); each
    part is shown while its own words are spoken. The free Gemini tier allows 10 TTS requests a
    day per model, so the build makes exactly 10 (one per scene).
    """

    id: str
    parts: list[tuple[tuple, str]]  # (visual, the words spoken over it)
    min_seconds: float = 0.0
    hold: float = 0.0  # silent seconds on the last visual after the narration ends
    timing: dict = field(default_factory=dict)

    @property
    def say(self) -> str:
        return " ".join(text for _, text in self.parts)


def one(id: str, say: str, visual: tuple, **kw) -> Scene:
    return Scene(id, [(visual, say)], **kw)


SCENES = [
    one("hook", "Thirteen seconds on a Manchester high street. Every face in it belongs to someone who never agreed "
          "to be filmed. Before anyone can post it, those faces have to be hidden, frame by frame.",
          ("clip", CLIPS / "manchester_street.mp4", 0.0, 1.0, None, "Stock footage (Pexels, zahid Anwar)")),
    one("problem", "Journalists, creators, schools and event organisers, and anyone answering a records request for "
          "bodycam or CCTV footage, all face this. Blur tools either blur everything, or make you mask by hand. None of "
          "them understands: everyone, except her. And none of them tells you when it missed someone.",
          ("card", "problem")),
    one("instruction", "Veil takes the instruction in plain English. Blur every face, except the woman in the pink coat.",
          ("ui", "instruction"), min_seconds=8.5),
    one("architecture", "Llama 3.1 turns that sentence into a plan of tool calls. SAM 3.1 finds and tracks every face, "
          "person and coat in the clip, and mask logic applies the exception frame by frame. Then the part that matters "
          "most: SAM 3.1 re-checks every frame of Veil's own output, and anything that slipped through is re-segmented "
          "and patched.",
          ("card", "architecture")),
    one("plan", "This is Llama's actual plan. Segment faces, people and the pink coat. Select the person wearing it. "
          "Subtract her from the faces, and blur the rest. These are real tool calls, validated before anything runs.",
          ("ui", "plan")),
    one("result", "Drag across the frame to compare: every face is blurred, except hers. The masks view shows each "
          "instance SAM tracked.",
          ("ui", "compare")),
    one("audit", "Here, after the follow-up, also hide the bank logo, SAM's first pass missed a face at two seconds, "
          "and the logo in several frames. Veil's audit re-segmented all three hundred and ninety-seven output frames, "
          "caught the misses, and patched nineteen of them. Twelve frames it couldn't fix are flagged with timestamps "
          "for a person to review. Veil never claims a result it hasn't checked.",
          ("clip", FOLLOW / "comparison_grid.mp4", 0.0, 0.6, None, None)),
    Scene("desk", [
        (("clip", DESK / "comparison_pair.mp4", 0.0, 1.0, "\u201cBlack out the laptop screen.\u201d", "Recorded run \u00b7 Llama 3.1 8B + SAM 3.1"),
         "Follow-up edits build on the last run. Black out the laptop screen."),
        (("clip", DESK_FOLLOW / "comparison_pair.mp4", 0.0, 1.0, "\u201cAlso black out the phone.\u201d", "Follow-up \u00b7 earlier masks reused"),
         "Then: also black out the phone. Veil keeps the earlier masks, and only segments what's new."),
    ]),
    one("report", "Every run ends with a Redaction Report. The original, the plan, the redacted video, the masks and the "
          "audit log are chained with SHA-256 hashes. Change a single byte, and verification fails.",
          ("card", "report")),
    Scene("engineering", [
        (("card", "numbers"), "All of this runs on a free Colab T4. Fitting SAM 3.1 meant rewriting its precision for an "
         "older GPU, pruning its tracker memory, and tracking long clips in pieces. The public demo replays these "
         "recorded runs instantly, with no GPU at all."),
        (("card", "close"), "Veil. Privacy that checks its own work."),
    ], hold=2.5),
]  # fmt: skip


# -- narration ----------------------------------------------------------------

# No style prompt: gemini-3.8-flash-tts reads any instruction in the text aloud (verified by transcription).
STYLE = ""


def narrate(scene: Scene, model: str, voice: str, key: str) -> Path:
    """The scene's narration as a WAV, from Gemini TTS (cached by model, voice and text)."""
    digest = hashlib.sha1(f"{model}|{voice}|{STYLE}|{scene.say}".encode()).hexdigest()[:12]
    out = BUILD / "voice" / f"{scene.id}-{digest}.wav"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "contents": [{"parts": [{"text": f"{STYLE}\n\n{scene.say}"}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}},
        },
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    for attempt in range(4):
        wait = _LAST_REQUEST[0] + REQUEST_GAP - time.monotonic()
        if wait > 0:
            time.sleep(wait)  # stay under the free tier's requests per minute
        r = httpx.post(url, headers={"x-goog-api-key": key}, json=body, timeout=180)
        _LAST_REQUEST[0] = time.monotonic()
        if r.status_code == 429 and "per day" in r.text:
            # Retrying burns nothing useful: the day's quota for this model is gone.
            raise SystemExit(f"Daily TTS quota for {model} is used up ({_retry_delay(r)}). Other TTS models on this key: "
                             f"{', '.join(tts_models(key))}. Re-run with --model <one of them>; finished scenes are cached.")  # fmt: skip
        if r.status_code in (429, 500, 502, 503) and attempt < 3:
            time.sleep(20 * (attempt + 1))
            continue
        if r.status_code == 404:
            raise SystemExit(f"TTS model {model!r} not found. TTS models on this key: {', '.join(tts_models(key)) or 'none'}")
        r.raise_for_status()
        break
    part = r.json()["candidates"][0]["content"]["parts"][0]["inlineData"]
    rate = int(re.search(r"rate=(\d+)", part.get("mimeType", "")).group(1)) if "rate=" in part.get("mimeType", "") else 24000
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(base64.b64decode(part["data"]))
    return out


REQUEST_GAP = 21.0  # seconds between TTS requests
_LAST_REQUEST = [0.0]


def _retry_delay(r: httpx.Response) -> str:
    for d in r.json().get("error", {}).get("details", []):
        if "retryDelay" in d:
            return f"resets in {int(float(d['retryDelay'].rstrip('s'))) // 3600} h"
    return "see https://ai.dev/rate-limit"


def tts_models(key: str) -> list[str]:
    r = httpx.get("https://generativelanguage.googleapis.com/v1beta/models", headers={"x-goog-api-key": key}, params={"pageSize": 200}, timeout=60)
    return [m["name"].removeprefix("models/") for m in r.json().get("models", []) if "tts" in m["name"]]


def wav_seconds(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


# -- cards ----------------------------------------------------------------------


def canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), BG)
    return img, ImageDraw.Draw(img)


def wrap(draw: ImageDraw.ImageDraw, text: str, f: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        test = f"{line} {word}".strip()
        if draw.textlength(test, font=f) <= width:
            line = test
        else:
            lines.append(line)
            line = word
    return [*lines, line] if line else lines


def kicker(draw: ImageDraw.ImageDraw, text: str, y: int = 96) -> None:
    draw.text((140, y), text.upper(), font=font(26, "semibold"), fill=CORAL)


def card_problem() -> Image.Image:
    img, d = canvas()
    kicker(d, "The problem")
    d.text((140, 150), "Sharing footage means hiding the people in it.", font=font(64, "bold"), fill=TEXT)
    who = [("Journalists", "street and protest footage"), ("Creators", "vlogs full of strangers"),
           ("Schools and events", "recordings of children and attendees"), ("Records requests", "bodycam and CCTV releases")]  # fmt: skip
    for i, (a, b) in enumerate(who):
        x, y = 140 + (i % 2) * 820, 300 + (i // 2) * 150
        d.rounded_rectangle((x, y, x + 780, y + 120), 18, fill=PANEL)
        d.text((x + 36, y + 18), a, font=font(40, "semibold"), fill=TEXT)
        d.text((x + 36, y + 70), b, font=font(28), fill=MUTED)
    gaps = ["Blur everything, or mask by hand, frame by frame.", "No tool understands \u201ceveryone except her\u201d.",
            "None tells you what it missed."]  # fmt: skip
    for i, g in enumerate(gaps):
        d.ellipse((140, 664 + i * 74, 160, 684 + i * 74), fill=CORAL)
        d.text((186, 650 + i * 74), g, font=font(38), fill=TEXT)
    return img


def arrow(d: ImageDraw.ImageDraw, a: tuple[int, int], b: tuple[int, int], color=MUTED, width: int = 5) -> None:
    d.line((a, b), fill=color, width=width)
    (x0, y0), (x1, y1) = a, b
    length = max(1.0, ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    d.polygon([(x1, y1), (x1 - 22 * ux + 12 * uy, y1 - 22 * uy - 12 * ux), (x1 - 22 * ux - 12 * uy, y1 - 22 * uy + 12 * ux)], fill=color)


def box(d: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, title: str, sub: str, accent=LINE) -> None:
    d.rounded_rectangle((x, y, x + w, y + h), 20, fill=PANEL, outline=accent, width=4)
    d.text((x + w // 2, y + 46), title, font=font(32, "bold"), fill=TEXT, anchor="mm")
    for i, line in enumerate(wrap(d, sub, font(25), w - 40)):
        d.text((x + w // 2, y + 92 + i * 32), line, font=font(25), fill=MUTED, anchor="mm")


def card_architecture() -> Image.Image:
    img, d = canvas()
    kicker(d, "How Veil works")
    d.text((140, 150), "Two Meta models. One of them checks the other's work.", font=font(56, "bold"), fill=TEXT)
    bw, bh, y1, y2 = 290, 170, 330, 640
    xs = [100 + i * (bw + 67) for i in range(5)]  # 67 px gaps leave room for the arrows
    top = [("Instruction", "plain English", LINE), ("Llama 3.1", "plan of tool calls", AMBER), ("SAM 3.1", "segment and track every concept", AMBER),
           ("Mask logic", "except / only, frame by frame", LINE), ("Render", "blur, pixelate, black out", LINE)]  # fmt: skip
    for (t, s, c), x in zip(top, xs):
        box(d, x, y1, bw, bh, t, s, c)
    for a, b in zip(xs, xs[1:]):
        arrow(d, (a + bw + 8, y1 + bh // 2), (b - 10, y1 + bh // 2))
    rx = xs[4]
    box(d, rx, y2, bw, bh, "SAM 3.1 re-check", "every output frame", CORAL)
    arrow(d, (rx + bw // 2, y1 + bh + 8), (rx + bw // 2, y2 - 10), CORAL)
    box(d, xs[2] + 185, y2, bw, bh, "Patch", "SAM point prompt at the leak", CORAL)
    arrow(d, (rx - 8, y2 + bh // 2), (xs[2] + 185 + bw + 10, y2 + bh // 2), CORAL)
    d.line(((xs[2] + 185 + bw // 2, y2 - 8), (xs[2] + 185 + bw // 2, y2 - 60), (rx + bw // 2 - 70, y2 - 60)), fill=CORAL, width=5)
    arrow(d, (rx + bw // 2 - 70, y2 - 60), (rx + bw // 2 - 70, y1 + bh + 10), CORAL)
    box(d, xs[0] + 60, y2, bw + 120, bh, "Redaction Report", "audit log + SHA-256 hash chain", GREEN)
    arrow(d, (xs[2] + 185 - 8, y2 + bh // 2), (xs[0] + 60 + bw + 130, y2 + bh // 2), GREEN)
    d.text((140, 900), "Leaks still found after two rounds are flagged with timestamps for a person to review.", font=font(32), fill=MUTED)
    return img


def card_report() -> Image.Image:
    img, d = canvas()
    kicker(d, "Redaction Report")
    d.text((140, 150), "Tamper-evident by construction.", font=font(60, "bold"), fill=TEXT)
    shot = BUILD / "ui" / "report.png"
    if shot.exists():
        r = Image.open(shot).convert("RGB")
        r.thumbnail((760, 800))
        img.paste(r, (140, 250))
        d.rectangle((140, 250, 140 + r.width, 250 + r.height), outline=LINE, width=3)
    x, y = 960, 250
    d.rounded_rectangle((x, y, W - 120, y + 640), 20, fill=(12, 14, 17), outline=LINE, width=3)
    mono, small = font(27, "mono"), font(24, "mono")
    cols = int((W - 120 - x - 68) // d.textlength("0", font=mono))  # monospace: wrap by characters

    def chop(text: str) -> list[str]:
        """Wrap on words; cut only tokens longer than a line (the hex digest)."""
        lines, line = [], ""
        for word in text.split():
            while len(word) > cols:
                if line:
                    lines.append(line)
                    line = ""
                lines.append(word[:cols])
                word = word[cols:]
            if len(f"{line} {word}".strip()) <= cols:
                line = f"{line} {word}".strip()
            else:
                lines.append(line)
                line = word
        return [*lines, line] if line else lines

    lines = [("$ python -m app.report verify cache/df14e93c192b \\", MUTED), ("    --original clips/manchester_street.mp4", MUTED)]
    lines += [(ln, GREEN) for ln in chop(VERIFY["ok"])]
    lines += [("", TEXT), ("# change one byte of redacted.mp4, verify again", FAINT)]
    lines += [(ln, CORAL) for ln in chop(VERIFY["tampered"])]
    for i, (ln, c) in enumerate(lines):
        d.text((x + 34, y + 34 + i * 44), ln, font=mono, fill=c)
    d.text((x + 34, y + 540), "link_i = sha256(link_i-1 + name + sha256(file))", font=small, fill=FAINT)
    d.text((x + 34, y + 580), "over: original, plan, redacted, masks, audit log", font=small, fill=FAINT)
    return img


def card_numbers() -> Image.Image:
    img, d = canvas()
    kicker(d, "Measured on the recorded runs")
    d.text((140, 150), "Built for a free Colab T4.", font=font(60, "bold"), fill=TEXT)
    stats = [("397 / 397", "output frames re-checked by SAM 3.1"), ("19", "misses caught and patched by the audit"),
             ("12", "frames flagged, with timestamps, for review"), ("$0", "GPU to replay the public demo")]  # fmt: skip
    for i, (big, label) in enumerate(stats):
        x, y = 140 + (i % 2) * 820, 270 + (i // 2) * 220
        d.rounded_rectangle((x, y, x + 780, y + 190), 20, fill=PANEL)
        d.text((x + 40, y + 22), big, font=font(84, "bold"), fill=CORAL if i < 3 else GREEN)
        d.text((x + 40, y + 130), label, font=font(32), fill=MUTED)
    fits = ["fp16 rewrite for a GPU without bf16", "tracker memory pruned to 48 frames", "long clips tracked in 150-frame pieces"]
    for i, t in enumerate(fits):
        x = 140 + i * 560
        d.ellipse((x, 774, x + 16, 790), fill=GREEN)
        d.text((x + 30, 760), t, font=font(30), fill=TEXT)
    d.text((140, 860), "Planned by Llama 3.1 8B and segmented by SAM 3.1, both on one T4.  53 tests.", font=font(32), fill=MUTED)
    return img


def card_close() -> Image.Image:
    img, d = canvas()
    d.text((W // 2, 400), "Veil", font=font(180, "bold"), fill=TEXT, anchor="mm")
    d.text((W // 2, 530), "Privacy that checks its own work.", font=font(56, "semibold"), fill=MUTED, anchor="mm")
    d.text((W // 2, 700), SITE.removeprefix("https://"), font=font(36), fill=TEXT, anchor="mm")
    d.text((W // 2, 760), REPO, font=font(36), fill=TEXT, anchor="mm")
    d.text((W // 2, 900), "Built with Llama  \u00b7  SAM 3.1  \u00b7  Footage: Pexels", font=font(30), fill=FAINT, anchor="mm")
    return img


CARDS = {"problem": card_problem, "architecture": card_architecture, "report": card_report, "numbers": card_numbers, "close": card_close}
VERIFY: dict[str, str] = {}


def overlay(title: str | None, kick: str | None) -> Path:
    """Transparent 1920x1080 layer with a title in the top band and a note in the bottom band."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if title:
        d.text((W // 2, 120), title, font=font(60, "bold"), fill=TEXT, anchor="mm")
    if kick:
        f = font(30, "semibold")
        tw = d.textlength(kick, font=f)
        d.rounded_rectangle((W // 2 - tw / 2 - 26, H - 110, W // 2 + tw / 2 + 26, H - 50), 14, fill=(0, 0, 0, 170))
        d.text((W // 2, H - 80), kick, font=f, fill=TEXT, anchor="mm")
    out = BUILD / "overlays" / f"{hashlib.sha1(f'{title}|{kick}'.encode()).hexdigest()[:10]}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


# -- verify output and UI capture ------------------------------------------------


def run_verify() -> None:
    py = ROOT / "backend" / ".venv" / "Scripts" / "python.exe"
    py = py if py.exists() else ROOT / "backend" / ".venv" / "bin" / "python"
    clip = CLIPS / "manchester_street.mp4"

    def verify(run_dir: Path) -> str:
        r = subprocess.run([str(py), "-m", "app.report", "verify", str(run_dir), "--original", str(clip)],
                           cwd=ROOT / "backend", capture_output=True, text=True)  # fmt: skip
        return (r.stdout + r.stderr).strip().splitlines()[-1]

    VERIFY["ok"] = verify(FOLLOW)
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / FOLLOW.name
        shutil.copytree(FOLLOW, copy, ignore=shutil.ignore_patterns("comparison_*"))
        data = bytearray((copy / "redacted.mp4").read_bytes())
        data[len(data) // 2] ^= 1
        (copy / "redacted.mp4").write_bytes(data)
        VERIFY["tampered"] = verify(copy)
    if not VERIFY["ok"].startswith("OK") or VERIFY["tampered"].startswith("OK"):
        raise SystemExit(f"verify didn't behave: {VERIFY}")


def capture_ui() -> dict[str, tuple[Path, float]]:
    """Record the UI scenes from the live demo: {name: (webm, seconds to skip)}."""
    from playwright.sync_api import sync_playwright

    out = BUILD / "ui"
    out.mkdir(parents=True, exist_ok=True)
    view = {"width": 1536, "height": 864}  # scaled to 1080p, so the UI's text reads 25% larger
    shots: dict[str, tuple[Path, float]] = {}
    with sync_playwright() as p:
        browser = p.chromium.launch()

        def session(name: str, script) -> None:
            ctx = browser.new_context(viewport=view, record_video_dir=str(out / name), record_video_size=view)
            page = ctx.new_page()
            t0 = time.monotonic()
            page.goto(SITE, wait_until="networkidle")
            page.get_by_text("Replays of recorded runs").wait_for()
            start = script(page)
            ctx.close()
            video = Path(page.video.path())
            shots[name] = (video, start - t0)

        def pick_and_run(page, typed: bool) -> None:
            page.get_by_role("button", name=re.compile("^Manchester high street")).click()
            box = page.get_by_role("textbox").first
            if typed:
                box.fill("")
                page.wait_for_timeout(400)
                box.type("Blur every face except the woman in the pink coat.", delay=45)
                page.wait_for_timeout(700)
            page.get_by_role("button", name="Redact video").click()
            page.get_by_text("Instant replay").first.wait_for()

        def instruction(page) -> float:
            page.wait_for_timeout(600)
            start = time.monotonic()
            page.wait_for_timeout(900)
            pick_and_run(page, typed=True)
            page.wait_for_timeout(5000)
            return start

        def plan(page) -> float:
            pick_and_run(page, typed=False)
            page.wait_for_timeout(800)
            start = time.monotonic()
            page.wait_for_timeout(5000)
            page.get_by_text("Show tool call").click()
            page.wait_for_timeout(9000)
            page.get_by_text("Show steps").click()
            page.wait_for_timeout(4000)
            return start

        def compare(page) -> float:
            pick_and_run(page, typed=False)
            page.get_by_role("button", name="Play").click()
            page.wait_for_timeout(600)
            start = time.monotonic()
            handle = page.get_by_role("slider", name="Compare original and redacted").bounding_box()
            stage = page.get_by_role("region", name="Video").bounding_box()
            cy = handle["y"] + handle["height"] / 2
            page.mouse.move(handle["x"] + handle["width"] / 2, cy)
            page.mouse.down()
            for x in [stage["x"] + stage["width"] * f for f in (0.15, 0.85, 0.5)]:
                page.mouse.move(x, cy, steps=45)
                page.wait_for_timeout(500)
            page.mouse.up()
            page.wait_for_timeout(1500)
            page.get_by_role("button", name="Masks").click()
            page.wait_for_timeout(6000)
            return start

        session("instruction", instruction)
        session("plan", plan)
        session("compare", compare)

        page = browser.new_page(viewport={"width": 1100, "height": 1250})
        page.goto(f"{SITE}/cached/{FOLLOW.name}/report.html", wait_until="networkidle")
        page.screenshot(path=str(out / "report.png"))
        browser.close()
    return shots


# -- rendering ---------------------------------------------------------------------


def ffmpeg(*args: str) -> None:
    r = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"ffmpeg failed: {r.stderr[-2000:]}")


FADE = 0.03  # seconds of fade at each end of every narration clip


def render(scene: Scene, frames: int, voice: Path | None, ui: dict, out: Path) -> None:
    """The scene's picture (out, video only) and sound (out.wav, PCM), both exactly `frames` long.

    A multi-part scene shows each visual while its own words are spoken. The sound stays PCM until
    the whole film is joined and encoded once: joining separately encoded AAC segments leaves a
    click at every cut (each segment carries its own encoder padding).
    """
    seconds = frames / FPS
    if len(scene.parts) == 1:
        render_visual(scene.parts[0][0], scene.id, frames, ui, out)
    else:
        chars = [len(text) for _, text in scene.parts]
        spans = [round((scene.timing["voice"] * c / sum(chars)) * FPS) for c in chars]
        spans[0] += round(LEAD * FPS)
        spans[-1] = frames - sum(spans[:-1])
        pieces = []
        for i, ((visual, _), span) in enumerate(zip(scene.parts, spans)):
            piece = out.with_name(f"{out.stem}-part{i}.mp4")
            render_visual(visual, f"{scene.id}-{i}", span, ui, piece)
            pieces.append(piece)
        listing = out.with_name(f"{out.stem}-parts.txt")
        listing.write_text("".join(f"file '{p.as_posix()}'\n" for p in pieces), encoding="utf-8")
        ffmpeg("-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(out))
        for p in (*pieces, listing):
            p.unlink()
    wav = out.with_suffix(".wav")
    if voice:
        spoken = wav_seconds(voice)
        af = (f"aresample=48000,aformat=channel_layouts=stereo,afade=t=in:d={FADE},afade=t=out:st={spoken - FADE:.3f}:d={FADE},"
              f"adelay={int(LEAD * 1000)}:all=1,apad,atrim=duration={seconds:.6f}")  # fmt: skip
        ffmpeg("-i", str(voice), "-af", af, "-c:a", "pcm_s16le", str(wav))
    else:
        ffmpeg("-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000", "-t", f"{seconds:.6f}", "-c:a", "pcm_s16le", str(wav))


def render_visual(visual: tuple, name: str, frames: int, ui: dict, out: Path) -> None:
    seconds = frames / FPS
    kind = visual[0]
    fit = f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=0x{BG[0]:02x}{BG[1]:02x}{BG[2]:02x}"
    inputs: list[str] = []
    if kind == "card":
        png = BUILD / "cards" / f"{name}.png"
        png.parent.mkdir(parents=True, exist_ok=True)
        CARDS[visual[1]]().save(png)
        inputs += ["-loop", "1", "-framerate", str(FPS), "-t", f"{seconds:.3f}", "-i", str(png)]
        vf = f"[0:v]{fit},fps={FPS},format=yuv420p[v]"
    else:
        if kind == "clip":
            _, path, start, speed, title, kick = visual
        else:
            path, start = ui[visual[1]]
            speed, title, kick = 1.0, None, "Live demo \u00b7 replay of a recorded run"
        inputs += ["-ss", f"{start:.3f}", "-i", str(path)]
        vf = f"[0:v]setpts=PTS/{speed},{fit},fps={FPS},tpad=stop_mode=clone:stop_duration={seconds:.3f},trim=duration={seconds:.3f},setpts=PTS-STARTPTS"
        if title or kick:
            inputs += ["-i", str(overlay(title, kick))]
            vf += "[base];[base][1:v]overlay=0:0,format=yuv420p[v]"
        else:
            vf += ",format=yuv420p[v]"
    ffmpeg(*inputs, "-filter_complex", vf, "-map", "[v]", "-frames:v", str(frames), "-an",
           "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-r", str(FPS), str(out))  # fmt: skip


def srt(scenes: list[Scene], path: Path) -> None:
    """One cue per sentence, timed in proportion to its length within the scene's narration."""

    def stamp(t: float) -> str:
        ms = int(round(t * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"

    cues, t = [], 0.0
    for s in scenes:
        sentences = [x for x in re.split(r"(?<=[.!?])\s+", s.say) if x]
        span = s.timing["voice"]
        at = t + LEAD
        for x in sentences:
            dur = span * len(x) / sum(len(y) for y in sentences)
            cues.append((at, at + dur, x))
            at += dur
        t += s.timing["seconds"]
    path.write_text("\n".join(f"{i}\n{stamp(a)} --> {stamp(b)}\n{x}\n" for i, (a, b, x) in enumerate(cues, 1)), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-voice", action="store_true", help="silent draft, scenes timed from their word count")
    ap.add_argument("--voice", default="Kore", help="Gemini prebuilt voice (e.g. Kore, Charon, Puck, Aoede)")
    ap.add_argument("--model", default="gemini-3.8-flash-tts")
    ap.add_argument("--reuse-ui", action="store_true", help="reuse the last UI capture instead of recording again")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not args.no_voice and not key:
        raise SystemExit("GEMINI_API_KEY is empty in .env (or pass --no-voice for a silent draft)")
    BUILD.mkdir(parents=True, exist_ok=True)

    print("verifying the report chain for the report scene ...", flush=True)
    run_verify()
    marks = BUILD / "ui" / "marks.json"
    if args.reuse_ui and marks.exists():
        ui = {k: (Path(p), s) for k, (p, s) in json.loads(marks.read_text()).items()}
    else:
        print("recording the UI from the live demo ...", flush=True)
        ui = capture_ui()
        marks.write_text(json.dumps({k: [str(p), s] for k, (p, s) in ui.items()}))

    for s in SCENES:
        voice = None if args.no_voice else narrate(s, args.model, args.voice, key)
        spoken = wav_seconds(voice) if voice else len(s.say.split()) / 2.6
        frames = round(max(spoken + LEAD + TAIL + s.hold, s.min_seconds) * FPS)  # whole frames: picture and sound stay aligned
        s.timing = {"voice": spoken, "frames": frames, "seconds": frames / FPS, "wav": voice}
    total = sum(s.timing["seconds"] for s in SCENES)
    print("\n".join(f"  {s.id:<12} {s.timing['seconds']:6.1f}s" for s in SCENES) + f"\n  {'total':<12} {total:6.1f}s", flush=True)
    if total > MAX_SECONDS:
        raise SystemExit(f"{total:.1f}s is over the {MAX_SECONDS}s cap; shorten the narration")

    parts = BUILD / "parts"
    shutil.rmtree(parts, ignore_errors=True)
    parts.mkdir()
    for i, s in enumerate(SCENES):
        print(f"rendering {s.id} ...", flush=True)
        render(s, s.timing["frames"], s.timing["wav"], ui, parts / f"{i:02d}-{s.id}.mp4")
    video, audio = BUILD / "joined.mp4", BUILD / "joined.wav"
    for ext, out in (("mp4", video), ("wav", audio)):
        listing = parts / f"list-{ext}.txt"
        listing.write_text("".join(f"file '{p.as_posix()}'\n" for p in sorted(parts.glob(f"*.{ext}"))), encoding="utf-8")
        ffmpeg("-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(out))
    final = BUILD / "final_submission.mp4"
    # One AAC encode over the joined PCM; loudnorm upsamples internally, so pin the output back to 48 kHz.
    ffmpeg("-i", str(video), "-i", str(audio), "-map", "0:v", "-map", "1:a", "-c:v", "copy",
           "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "48000", "-c:a", "aac", "-b:a", "192k",
           "-t", f"{sum(s.timing['frames'] for s in SCENES) / FPS:.6f}", "-movflags", "+faststart", str(final))  # fmt: skip
    srt(SCENES, BUILD / "final_submission.srt")
    probe = subprocess.run([FFMPEG, "-hide_banner", "-i", str(final)], capture_output=True, text=True).stderr
    duration = re.search(r"Duration: (\d+):(\d+):([\d.]+)", probe)
    seconds = int(duration[1]) * 3600 + int(duration[2]) * 60 + float(duration[3])
    print(f"\n{final} : {seconds:.1f}s" + ("" if seconds <= MAX_SECONDS else "  OVER THE CAP"))
    if seconds > MAX_SECONDS:
        sys.exit(1)


if __name__ == "__main__":
    main()
