from __future__ import annotations

from ..config import settings
from .base import RefineRequest, Segmenter
from .mock import MockSegmenter
from .modal_sam import ModalSamSegmenter

mock_segmenter = MockSegmenter()


def get_segmenter() -> Segmenter:
    return ModalSamSegmenter() if settings.segmenter_live else mock_segmenter


__all__ = ["RefineRequest", "Segmenter", "get_segmenter", "mock_segmenter"]
