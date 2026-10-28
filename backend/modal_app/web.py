"""Public Veil demo on Modal: frozen runs only (no uploads, no model calls, no GPU).

The FastAPI app and the built frontend in one CPU container, serving the
sample clips that have frozen runs in demo_assets/cache. Judges replay those
runs instantly; any other instruction gets a 409 (see VEIL_CACHE_ONLY).

    cd frontend && npm run build && cd ..
    modal deploy backend/modal_app/web.py

Job state lives in memory, so the app runs as a single container (one is
plenty: a replay is a JSON copy plus static files).
"""

import json
from pathlib import Path, PurePosixPath

import modal

# Modal imports this file again inside the container (as /root/web.py), where the repo isn't
# around: everything that reads it runs only on the machine that deploys.
LOCAL = modal.is_local()
ROOT = Path(__file__).resolve().parents[2] if LOCAL else Path("/")
REMOTE = PurePosixPath("/root/veil")  # inside the Linux container, whatever OS deploys
CACHE = ROOT / "demo_assets" / "cache"


def _clip_files() -> list[Path]:
    """The clips with frozen runs, with their thumbnails and sidecars."""
    if not LOCAL:
        return []
    ids = {e["video_id"].removeprefix("sample-") for e in json.loads((CACHE / "manifest.json").read_text(encoding="utf-8"))}
    return [p for i in sorted(ids) for p in (ROOT / "demo_assets" / "clips").glob(f"{i}.*")]


def _requirements() -> list[str]:
    if not LOCAL:
        return []
    lines = (ROOT / "backend" / "requirements.txt").read_text(encoding="utf-8").splitlines()
    return [ln.split("#")[0].strip() for ln in lines if ln.split("#")[0].strip()]


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(*_requirements())
    .env({
        "VEIL_CACHE_ONLY": "1",
        "VEIL_FORCE_MOCK": "planner,segmenter,auditor",  # belt and braces: no key, no model, ever
        "VEIL_DATA_DIR": "/tmp/veil-data",
    })  # fmt: skip
    .add_local_dir(ROOT / "backend" / "app", str(REMOTE / "backend" / "app"), ignore=["**/__pycache__"])
    .add_local_dir(ROOT / "frontend" / "dist", str(REMOTE / "frontend" / "dist"))
    # The comparison videos are editing material for the demo video; the app doesn't serve them.
    .add_local_dir(CACHE, str(REMOTE / "demo_assets" / "cache"), ignore=["**/comparison_*.mp4"])
)
for path in _clip_files():
    image = image.add_local_file(path, str(REMOTE / "demo_assets" / "clips" / path.name))

app = modal.App("veil-demo", image=image)


@app.function(max_containers=1, scaledown_window=600, timeout=120)
@modal.concurrent(max_inputs=64)
@modal.asgi_app()
def web():
    import sys

    sys.path.insert(0, str(REMOTE / "backend"))
    from app.main import app as api

    return api
