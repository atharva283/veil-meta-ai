"""Registry of videos the pipeline can run on: bundled samples and uploads."""

from __future__ import annotations

import json
import shutil
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

from .config import settings
from .pipeline import video_io
from .pipeline.video_io import VideoMeta


@dataclass
class VideoRecord:
    id: str
    title: str
    path: Path  # normalized working copy (H.264, even dims)
    meta: VideoMeta
    thumbnail: Path | None = None
    annotations: Path | None = None  # ground truth for synthetic samples (mock segmenter)
    suggested_prompts: tuple[str, ...] = ()
    follow_up_prompts: tuple[str, ...] = ()
    description: str = ""
    is_sample: bool = False
    order: int = 0

    def public(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "is_sample": self.is_sample,
            "url": media_url(self.path),
            "thumbnail_url": media_url(self.thumbnail) if self.thumbnail else None,
            "suggested_prompts": list(self.suggested_prompts),
            "follow_up_prompts": list(self.follow_up_prompts),
            "simulated": self.annotations is not None,
            **self.meta.as_dict(),
        }


def media_url(path: Path) -> str:
    """Files are served from /media (data dir) or /samples (demo assets)."""
    path = path.resolve()
    for root, prefix in ((settings.data_dir.resolve(), "/media"), (settings.samples_dir.resolve(), "/samples")):
        if path.is_relative_to(root):
            return f"{prefix}/{path.relative_to(root).as_posix()}"
    raise ValueError(f"{path} is outside the served directories")


class VideoStore:
    def __init__(self) -> None:
        self._videos: dict[str, VideoRecord] = {}
        self._lock = threading.Lock()
        self.uploads_dir = settings.data_dir / "uploads"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

    def load_samples(self) -> None:
        """Register every <name>.mp4 in demo_assets/clips that has a sidecar <name>.json."""
        for sidecar in sorted(settings.samples_dir.glob("*.json")):
            if sidecar.name.endswith(".annotations.json"):
                continue
            info = json.loads(sidecar.read_text(encoding="utf-8"))
            clip = sidecar.with_suffix(".mp4")
            if not clip.exists():
                continue
            ann = sidecar.with_name(f"{sidecar.stem}.annotations.json")
            thumb = sidecar.with_suffix(".jpg")
            if not thumb.exists():
                video_io.save_thumbnail(clip, thumb)
            meta = video_io.probe(clip)
            meta.frame_count = info.get("frame_count", meta.frame_count)
            rec = VideoRecord(
                id=f"sample-{sidecar.stem}",
                title=info.get("title", sidecar.stem),
                description=info.get("description", ""),
                path=clip,
                meta=meta,
                thumbnail=thumb,
                annotations=ann if ann.exists() else None,
                suggested_prompts=tuple(info.get("suggested_prompts", ())),
                follow_up_prompts=tuple(info.get("follow_up_prompts", ())),
                is_sample=True,
                order=int(info.get("order", 99)),
            )
            self._videos[rec.id] = rec

    def add_upload(self, tmp_path: Path, filename: str) -> VideoRecord:
        vid = uuid.uuid4().hex[:12]
        folder = self.uploads_dir / vid
        folder.mkdir(parents=True)
        try:
            meta = video_io.normalize(tmp_path, folder / "source.mp4")
            video_io.save_thumbnail(folder / "source.mp4", folder / "thumb.jpg")
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        rec = VideoRecord(
            id=vid,
            title=Path(filename).stem[:80] or "Uploaded clip",
            path=folder / "source.mp4",
            meta=meta,
            thumbnail=folder / "thumb.jpg",
        )
        with self._lock:
            self._videos[vid] = rec
        return rec

    def get(self, vid: str) -> VideoRecord | None:
        return self._videos.get(vid)

    def samples(self) -> list[VideoRecord]:
        return sorted((v for v in self._videos.values() if v.is_sample), key=lambda v: (v.order, v.title))


store = VideoStore()
