"""SAM 3.1 segmentation service for Veil, deployed on Modal.

    modal deploy backend/modal_app/sam3_service.py

Then put the printed https://...modal.run URL in .env as MODAL_ENDPOINT.

Setup:
  1. Get access to the gated checkpoints at https://huggingface.co/facebook/sam3.1
     with the account whose token you use.
  2. modal secret create veil-sam3 HF_TOKEN=hf_xxx VEIL_TOKEN=<any string>
     (VEIL_TOKEN must match MODAL_AUTH_TOKEN in .env).
  3. modal deploy, then check $MODAL_ENDPOINT/health ("ok": true once weights load;
     send the bearer token to see load errors) and:
       curl -H "Authorization: Bearer $TOKEN" \
            -F video=@demo_assets/clips/street_crossing.mp4 \
            -F 'payload={"prompts":[{"id":"p0","text":"face"}]}' \
            $MODAL_ENDPOINT/segment

Weights download once into the "veil-sam3-cache" Volume, so later cold starts
skip the download. Modal needs a payment method on the account for any GPU;
without one, run SAM 3.1 in-process instead (notebooks/veil_colab_runner.ipynb).

API contract (mirrors backend/app/segmenter/modal_sam.py):
  POST /segment  multipart: video=<mp4>, payload={"prompts":[{"id","text"}], "score_threshold"}
  POST /refine   multipart: video=<mp4>, payload={"requests":[{"id","text","frame_index","box","point"?,"window"}]}
  -> {"tracked": true, "results": [{"id", "objects": [{"obj_id", "detections": [{"f","score","box","rle"}]}]}]}

The model calls live in backend/app/segmenter/sam3_core.py, shared with
in-process runs; this file only adds the GPU container and the HTTP layer.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import modal

CORE = Path(__file__).resolve().parent.parent / "app" / "segmenter" / "sam3_core.py"
SAM_COMMIT = "f1647e8d0c1d280991f4d1b7d974c2704537fe66"  # keep in step with sam3_core.SAM_COMMIT
GPU = "A10G"  # 24 GB Ampere: plenty for SAM 3.1, and native bf16

cache = modal.Volume.from_name("veil-sam3-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.10.0", "torchvision==0.25.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "numpy>=1.26,<2",
        "opencv-python-headless==4.10.0.84",
        "pillow",
        # Imported by sam3 but not declared in its dependencies.
        "einops",
        "scipy",
        "psutil",
        "setuptools<80",  # pkg_resources
        "fastapi[standard]",
        "python-multipart",
        "huggingface_hub",
    )
    .pip_install(f"git+https://github.com/facebookresearch/sam3.git@{SAM_COMMIT}")
    .env({"HF_HOME": "/cache/hf"})  # huggingface_hub reads HF_TOKEN from the secret
    .add_local_file(CORE, "/root/sam3_core.py")
)

app = modal.App("veil-sam3", image=image)


@app.cls(
    gpu=GPU,
    secrets=[modal.Secret.from_name("veil-sam3")],
    volumes={"/cache": cache},
    scaledown_window=120,  # scale to zero two minutes after the last request
    timeout=1200,
)
class Sam3Service:
    @modal.enter()
    def load(self):
        import sam3_core

        self.predictor, self.load_error = None, ""
        try:
            self.predictor, _ = sam3_core.load_predictor("auto")
            cache.commit()
        except Exception as exc:  # usually: this HF account isn't approved for facebook/sam3.1 yet
            self.load_error = f"{type(exc).__name__}: {exc}"[:600]

    @modal.asgi_app()
    def web(self):
        import cv2
        import sam3_core
        from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile

        api = FastAPI(title="veil-sam3")
        token = os.environ.get("VEIL_TOKEN", "")

        def check(auth: str | None):
            if token and auth != f"Bearer {token}":
                raise HTTPException(401, "missing or wrong bearer token")
            if self.predictor is None:
                raise HTTPException(503, f"SAM 3.1 isn't loaded: {self.load_error}")

        async def save(video: UploadFile) -> str:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(await video.read())
            tmp.close()
            return tmp.name

        @api.get("/health")
        def health(authorization: str | None = Header(None)):
            status = {"ok": self.predictor is not None}
            if not token or authorization == f"Bearer {token}":  # load errors only for the token holder
                status.update(model="sam3.1", commit=SAM_COMMIT, gpu=GPU, error=self.load_error)
            return status

        @api.post("/segment")
        async def segment(video: UploadFile = File(...), payload: str = Form(...), authorization: str | None = Header(None)):
            check(authorization)
            body = json.loads(payload)
            path = await save(video)
            try:
                return sam3_core.segment(self.predictor, path, body["prompts"], float(body.get("score_threshold", 0.45)))
            finally:
                os.unlink(path)

        @api.post("/refine")
        async def refine(video: UploadFile = File(...), payload: str = Form(...), authorization: str | None = Header(None)):
            check(authorization)
            path = await save(video)
            try:
                cap = cv2.VideoCapture(path)
                W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                return sam3_core.refine(self.predictor, path, json.loads(payload)["requests"], W, H)
            finally:
                os.unlink(path)

        return api
