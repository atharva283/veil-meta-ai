"""SAM 3.1 in this process, for machines with their own GPU (Colab, Kaggle, a workstation).

Same requests and decoding as the Modal client; only the transport differs.
Enable with VEIL_SAM_LOCAL=1 (and VEIL_SAM_PRECISION=auto|bf16|fp16|fp32).
The model loads on first use and stays loaded for the process.
"""

from __future__ import annotations

import threading

from ..config import settings
from ..videos import VideoRecord
from . import sam3_core
from .modal_sam import ModalSamSegmenter


class LocalSam3Segmenter(ModalSamSegmenter):
    name = "SAM 3.1 (in-process)"

    def __init__(self) -> None:
        self._predictor = None
        self._lock = threading.Lock()

    def load(self):
        """Load the model (once); `name` then records the GPU and precision used."""
        with self._lock:
            if self._predictor is None:
                self._predictor, self.name = sam3_core.load_predictor(settings.sam_precision)
        return self._predictor

    def _post(self, route: str, video: VideoRecord, payload: dict) -> dict:
        predictor = self.load()
        if route == "segment":
            return sam3_core.segment(predictor, str(video.path), payload["prompts"], payload.get("score_threshold", 0.45))
        return sam3_core.refine(predictor, str(video.path), payload["requests"], video.meta.width, video.meta.height)
