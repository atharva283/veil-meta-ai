from __future__ import annotations

from ..config import settings
from ..segmenter import mock_segmenter
from .base import AuditPolicy, Auditor, AuditSample
from .mock import MockAuditor
from .muse_spark import MuseSparkAuditor

mock_auditor = MockAuditor(mock_segmenter)


def get_auditor() -> Auditor:
    return MuseSparkAuditor() if settings.auditor_live else mock_auditor


__all__ = ["AuditPolicy", "AuditSample", "Auditor", "get_auditor", "mock_auditor"]
