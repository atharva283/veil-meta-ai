from __future__ import annotations

from ..config import settings
from .base import RefineRequest, Segmenter
from .mock import MockSegmenter
from .modal_sam import ModalSamSegmenter

mock_segmenter = MockSegmenter()
_local = None


def get_segmenter() -> Segmenter:
    global _local
    if not settings.segmenter_live:
        return mock_segmenter
    if settings.sam_local:
        if _local is None:
            from .local_sam import LocalSam3Segmenter  # imports torch/sam3 only when used

            _local = LocalSam3Segmenter()
            _local.load()  # before any job records the segmenter's name
        return _local
    return ModalSamSegmenter()


__all__ = ["RefineRequest", "Segmenter", "get_segmenter", "mock_segmenter"]
