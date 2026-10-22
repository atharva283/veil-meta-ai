"""Runtime settings, read once from the repo-level .env file.

Each model integration runs "live" when its credentials are present and
falls back to a local mock otherwise, so the whole pipeline works offline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = BACKEND_DIR.parent

load_dotenv(REPO_DIR / ".env")


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # Meta Model API (Llama 3.1 planner + Muse Spark auditor).
    meta_api_key: str = field(default_factory=lambda: _env("META_MODEL_API_KEY"))
    meta_api_base: str = field(
        default_factory=lambda: _env("META_MODEL_API_BASE", "https://api.llama.com/compat/v1")
    )
    planner_model: str = field(
        default_factory=lambda: _env("PLANNER_MODEL", "Llama-3.1-70B-Instruct")
    )
    auditor_api_base: str = field(default_factory=lambda: _env("AUDITOR_API_BASE"))
    auditor_api_key: str = field(default_factory=lambda: _env("AUDITOR_API_KEY"))
    auditor_model: str = field(default_factory=lambda: _env("AUDITOR_MODEL", "muse-spark"))

    # SAM 3.1 service deployed on Modal (see backend/modal_app/sam3_service.py).
    modal_endpoint: str = field(default_factory=lambda: _env("MODAL_ENDPOINT").rstrip("/"))
    modal_auth_token: str = field(default_factory=lambda: _env("MODAL_AUTH_TOKEN"))

    # Comma-separated list of components to keep on mocks even when keys exist,
    # e.g. VEIL_FORCE_MOCK=auditor while Muse Spark access is pending.
    force_mock: frozenset[str] = field(
        default_factory=lambda: frozenset(
            p.strip().lower() for p in _env("VEIL_FORCE_MOCK").split(",") if p.strip()
        )
    )

    audit_sample_fps: float = field(default_factory=lambda: _env_float("AUDIT_SAMPLE_FPS", 2.0))
    audit_max_rounds: int = field(default_factory=lambda: _env_int("AUDIT_MAX_ROUNDS", 2))
    max_upload_mb: int = field(default_factory=lambda: _env_int("MAX_UPLOAD_MB", 200))
    request_timeout_s: float = field(default_factory=lambda: _env_float("REQUEST_TIMEOUT_S", 120.0))

    data_dir: Path = field(default_factory=lambda: Path(_env("VEIL_DATA_DIR") or BACKEND_DIR / "data"))
    samples_dir: Path = REPO_DIR / "demo_assets" / "clips"
    frontend_dist: Path = REPO_DIR / "frontend" / "dist"

    @property
    def planner_live(self) -> bool:
        return bool(self.meta_api_key) and "planner" not in self.force_mock

    @property
    def segmenter_live(self) -> bool:
        return bool(self.modal_endpoint) and "segmenter" not in self.force_mock

    @property
    def auditor_live(self) -> bool:
        return bool(self.auditor_api_key or self.meta_api_key) and "auditor" not in self.force_mock

    @property
    def auditor_base(self) -> str:
        return self.auditor_api_base or self.meta_api_base

    @property
    def auditor_key(self) -> str:
        return self.auditor_api_key or self.meta_api_key


settings = Settings()
