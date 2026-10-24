"""Video ingestion, frame iteration, and browser-safe H.264 encoding.

Uses the ffmpeg binary bundled with imageio-ffmpeg, so no system install is
needed. Every upload is normalized once into a working copy (even dimensions,
capped width/fps/duration, H.264 + AAC) that the rest of the pipeline and the
browser's "Before" view both use, which keeps frame indices in lockstep.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import imageio_ffmpeg
import numpy as np

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
MAX_WIDTH = 1280
MAX_FPS = 30.0
MAX_DURATION_S = 60.0
H264_OUT = ["-preset", "veryfast", "-crf", "20", "-movflags", "+faststart"]


@dataclass
class VideoMeta:
    width: int
    height: int
    fps: float
    frame_count: int
    has_audio: bool

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0

    def as_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 3),
            "frame_count": self.frame_count,
            "duration_s": round(self.duration_s, 3),
            "has_audio": self.has_audio,
        }


class VideoError(RuntimeError):
    pass


def _has_audio(path: Path) -> bool:
    proc = subprocess.run([FFMPEG, "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    return bool(re.search(r"Stream #\S+.*Audio:", proc.stderr))


def probe(path: Path) -> VideoMeta:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError(f"Can't open {path.name}: the file is not a readable video.")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if width <= 0 or height <= 0:
        raise VideoError(f"{path.name} has no video stream.")
    return VideoMeta(width, height, fps if 1 <= fps <= 240 else 25.0, count, _has_audio(path))


def normalize(src: Path, dst: Path) -> VideoMeta:
    """Transcode an upload into the pipeline's working format."""
    meta = probe(src)
    scale = min(1.0, MAX_WIDTH / meta.width)
    width = int(meta.width * scale) // 2 * 2
    height = int(meta.height * scale) // 2 * 2
    fps = min(meta.fps, MAX_FPS)
    cmd = [
        FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src), "-t", str(MAX_DURATION_S),
        "-vf", f"scale={width}:{height},fps={fps:.3f}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", *H264_OUT,
        "-c:a", "aac", "-b:a", "128k",
        str(dst),
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise VideoError(f"ffmpeg couldn't convert {src.name}: {proc.stderr.strip()[-400:]}")
    out = probe(dst)
    # CAP_PROP_FRAME_COUNT can be off by a frame or two; count exactly once.
    out.frame_count = sum(1 for _ in iter_frames(dst))
    return out


def iter_frames(path: Path) -> Iterator[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield frame
    finally:
        cap.release()


def read_frame(path: Path, index: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


class H264Writer:
    """Streams BGR frames into a browser-playable MP4, optionally muxing audio."""

    def __init__(self, path: Path, meta: VideoMeta, audio_from: Path | None = None):
        self.path = path
        self._gen = imageio_ffmpeg.write_frames(
            str(path),
            (meta.width, meta.height),
            pix_fmt_in="bgr24",
            fps=meta.fps,
            codec="libx264",
            quality=None,
            macro_block_size=2,
            output_params=list(H264_OUT),  # imageio-ffmpeg appends audio flags to this list in place
            audio_path=str(audio_from) if audio_from and meta.has_audio else None,
            audio_codec="aac" if audio_from and meta.has_audio else None,
        )
        self._gen.send(None)

    def write(self, frame: np.ndarray) -> None:
        self._gen.send(np.ascontiguousarray(frame))

    def close(self) -> None:
        self._gen.close()


def save_thumbnail(path: Path, dst: Path, at_fraction: float = 0.3) -> None:
    meta = probe(path)
    frame = read_frame(path, int(meta.frame_count * at_fraction)) if meta.frame_count else None
    if frame is None:
        frame = next(iter_frames(path), None)
    if frame is not None:
        thumb = cv2.resize(frame, (480, int(480 * frame.shape[0] / frame.shape[1])), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst), thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
