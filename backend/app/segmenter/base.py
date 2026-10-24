from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..pipeline.tracks import Track
from ..videos import VideoRecord


@dataclass
class RefineRequest:
    """Re-segment `concept` around a region the auditor flagged as leaked."""

    concept: str
    frame_index: int
    box: tuple[float, float, float, float]


class Segmenter(Protocol):
    name: str

    def segment(
        self, video: VideoRecord, concepts: list[str], score_threshold: float | None = None, frame_stride: int | None = None
    ) -> dict[str, list[Track]]:
        """Text-prompted segmentation + tracking for each concept.

        The defaults suit the first pass; the output re-check asks for a low
        threshold on every frame (frame_stride=1).
        """
        ...

    def refine(self, video: VideoRecord, requests: list[RefineRequest]) -> list[Track | None]:
        """One patched track (or None) per request, propagated through the clip."""
        ...


def normalize_concept(text: str) -> str:
    words = [w for w in text.lower().replace("-", " ").split() if w not in ("the", "a", "an", "any", "every", "all")]
    out = []
    for w in words:
        if w.endswith("ies") and len(w) > 4:
            w = w[:-3] + "y"
        elif w.endswith("s") and not w.endswith("ss") and len(w) > 3:
            w = w[:-1]
        out.append(w)
    return " ".join(out)
