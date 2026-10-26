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
skip the download.

API contract (mirrors backend/app/segmenter/modal_sam.py):
  POST /segment  multipart: video=<mp4>, payload={"prompts":[{"id","text"}], "score_threshold"}
  POST /refine   multipart: video=<mp4>, payload={"requests":[{"id","text","frame_index","box","point"?,"window"}]}
  -> {"tracked": true, "results": [{"id", "objects": [{"obj_id", "detections": [{"f","score","box","rle"}]}]}]}

/segment runs SAM 3.1's video predictor: a text prompt detects every instance
and the tracker follows each one with a stable id through the whole clip
(objects entering later are picked up as propagation reaches them).
/refine follows one object from a point prompt (the flagged box's center, or
an explicit "point" in pixels), `window` frames each way.
"""

from __future__ import annotations

import json
import os
import tempfile

import modal

SAM_COMMIT = "f1647e8d0c1d280991f4d1b7d974c2704537fe66"  # facebookresearch/sam3 main, 2026-10-23
GPU = "A10G"  # 24 GB Ampere: enough for SAM 3.1; Flash-Attention 3 kernels need Hopper, so they are off below
MAX_OBJECTS = 64  # per prompt; SAM 3.1's default of 16 is too few for a crowd

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
)

app = modal.App("veil-sam3", image=image)


def rle_encode(mask):
    """Row-major run lengths over the crop, starting with a run of zeros."""
    import numpy as np

    flat = mask.astype(np.uint8).ravel()
    if flat.size == 0:
        return []
    changes = np.flatnonzero(np.diff(flat)) + 1
    runs = np.diff(np.concatenate(([0], changes, [flat.size]))).tolist()
    return runs if flat[0] == 0 else [0, *runs]


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
        import torch

        self.predictor, self.load_error = None, ""
        try:
            from sam3.model_builder import build_sam3_predictor

            torch.backends.cuda.matmul.allow_tf32 = True
            self.predictor = build_sam3_predictor(version="sam3.1", use_fa3=False, max_num_objects=MAX_OBJECTS)
            cache.commit()
        except Exception as exc:  # usually: this HF account isn't approved for facebook/sam3.1 yet
            self.load_error = f"{type(exc).__name__}: {exc}"[:600]

    # -- model calls ---------------------------------------------------------

    def _request(self, **req):
        return self.predictor.handle_request(request=req)

    def _stream(self, **req):
        return self.predictor.handle_stream_request(request=req)

    @staticmethod
    def _collect(objects: dict, frame_idx: int, outputs: dict, threshold: float, only_obj: int | None = None) -> None:
        """Fold one frame of predictor output into {obj_id: {frame: detection}}."""
        import numpy as np

        ids, probs, masks = outputs["out_obj_ids"], outputs["out_probs"], outputs["out_binary_masks"]
        for k in range(len(ids)):
            oid, score = int(ids[k]), float(probs[k])
            if (only_obj is not None and oid != only_obj) or score < threshold:
                continue
            ys, xs = np.nonzero(masks[k])
            if len(xs) == 0:
                continue
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            objects.setdefault(oid, {})[int(frame_idx)] = {
                "f": int(frame_idx), "score": round(score, 4), "box": [x0, y0, x1, y1],
                "rle": rle_encode(masks[k][y0:y1, x0:x1]),
            }  # fmt: skip

    @staticmethod
    def _objects(found: dict) -> list[dict]:
        return [{"obj_id": oid, "detections": [d for _, d in sorted(frames.items())]} for oid, frames in sorted(found.items())]

    # -- HTTP ----------------------------------------------------------------

    @modal.asgi_app()
    def web(self):
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

        def frame_size(path: str) -> tuple[int, int]:
            import cv2

            cap = cv2.VideoCapture(path)
            size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            return size

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
            threshold = float(body.get("score_threshold", 0.45))
            path = await save(video)
            session = self._request(type="start_session", resource_path=path)["session_id"]
            results = []
            try:
                for p in body["prompts"]:
                    found: dict = {}
                    # A new text prompt resets the session's tracking state.
                    out = self._request(type="add_prompt", session_id=session, frame_index=0, text=p["text"], output_prob_thresh=threshold)
                    self._collect(found, out["frame_index"], out["outputs"], threshold)
                    for r in self._stream(type="propagate_in_video", session_id=session, propagation_direction="forward",
                                          start_frame_index=0, output_prob_thresh=threshold):  # fmt: skip
                        self._collect(found, r["frame_index"], r["outputs"], threshold)
                    results.append({"id": p["id"], "objects": self._objects(found)})
            finally:
                self._request(type="close_session", session_id=session)
                os.unlink(path)
            return {"tracked": True, "results": results}

        @api.post("/refine")
        async def refine(video: UploadFile = File(...), payload: str = Form(...), authorization: str | None = Header(None)):
            """Follow one object from a point prompt, `window` frames either side of frame_index."""
            check(authorization)
            reqs = json.loads(payload)["requests"]
            path = await save(video)
            W, H = frame_size(path)
            session = self._request(type="start_session", resource_path=path)["session_id"]
            results = []
            try:
                for r in reqs:
                    x, y = r.get("point") or ((r["box"][0] + r["box"][2]) / 2, (r["box"][1] + r["box"][3]) / 2)
                    self._request(type="reset_session", session_id=session)
                    found: dict = {}
                    out = self._request(type="add_prompt", session_id=session, frame_index=r["frame_index"], obj_id=1,
                                        points=[[x / W, y / H]], point_labels=[1], rel_coordinates=True)  # fmt: skip
                    self._collect(found, out["frame_index"], out["outputs"], 0.0, only_obj=1)
                    for o in self._stream(type="propagate_in_video", session_id=session, propagation_direction="both",
                                          start_frame_index=r["frame_index"], max_frame_num_to_track=r["window"]):  # fmt: skip
                        self._collect(found, o["frame_index"], o["outputs"], 0.0, only_obj=1)
                    results.append({"id": r["id"], "objects": self._objects(found)})
            finally:
                self._request(type="close_session", session_id=session)
                os.unlink(path)
            return {"tracked": True, "results": results}

        return api
