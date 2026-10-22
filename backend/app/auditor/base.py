from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..pipeline.tracks import FrameMask
from ..schemas import AuditFinding
from ..videos import VideoRecord


@dataclass
class AuditSample:
    """One rendered output frame plus what the pipeline believes it did there."""

    frame_index: int
    time_s: float
    image: np.ndarray  # BGR, already redacted
    redacted: list[FrameMask] = field(default_factory=list)
    kept: list[FrameMask] = field(default_factory=list)  # intentionally visible instances


@dataclass
class AuditPolicy:
    targets: list[str]  # concepts that must be hidden, e.g. ["face", "license plate"]
    kept_descriptions: list[str] = field(default_factory=list)


class Auditor(Protocol):
    name: str

    def audit(self, video: VideoRecord, samples: list[AuditSample], policy: AuditPolicy) -> list[AuditFinding]:
        ...
