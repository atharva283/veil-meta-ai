from __future__ import annotations

from ..config import settings
from ..segmenter import mock_segmenter
from .base import AuditPolicy, Auditor, AuditSample
from .mock import MockAuditor
from .muse_spark import MuseSparkAuditor

mock_auditor = MockAuditor(mock_segmenter)


class NoSampledAuditor:
    """VEIL_AUDITOR=none: no vision model on sampled frames; the SAM re-check of every frame still runs."""

    name = "none (SAM re-check only)"

    def audit(self, video, samples, policy) -> list:
        return []


def get_auditor() -> Auditor:
    if settings.auditor_off:
        return NoSampledAuditor()
    return MuseSparkAuditor() if settings.auditor_live else mock_auditor


__all__ = ["AuditPolicy", "AuditSample", "Auditor", "get_auditor", "mock_auditor"]
