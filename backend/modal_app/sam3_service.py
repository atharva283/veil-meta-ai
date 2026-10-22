"""SAM 3.1 segmentation service for Veil, deployed on Modal.

    modal deploy backend/modal_app/sam3_service.py

Then put the printed https://...modal.run URL in .env as MODAL_ENDPOINT.

Day-1 checklist (lines marked VERIFY were written against the SAM 3 release
and could not be run before the hackathon):
  1. Accept the SAM 3.1 license on Hugging Face; create a read token.
  2. modal secret create veil-sam3 HF_TOKEN=hf_xxx VEIL_TOKEN=<any string>
     (VEIL_TOKEN must match MODAL_AUTH_TOKEN in .env).
  3. Check SAM_PACKAGE and the two model calls in Sam3Service.load / _detect.
  4. modal deploy, then smoke-test:
       curl -H "Authorization: Bearer $TOKEN" \
            -F video=@demo_assets/clips/street_crossing.mp4 \
            -F 'payload={"prompts":[{"id":"p0","text":"face"}]}' \
            $MODAL_ENDPOINT/segment

API contract (mirrors backend/app/segmenter/modal_sam.py):
  POST /segment  multipart: video=<mp4>, payload={"prompts":[{"id","text"}], "score_threshold", "frame_stride"}
  POST /refine   multipart: video=<mp4>, payload={"requests":[{"id","text","frame_index","box","window"}]}
  -> {"tracked": bool, "results": [{"id", "objects": [{"obj_id", "detections": [{"f","score","box","rle"}]}]}]}

This first version runs the SAM 3.1 *image* model frame by frame and lets the
client link detections (tracked=false). Next step once it works: switch
/segment to the video predictor so SAM tracks identities itself (tracked=true).
"""

from __future__ import annotations

import json
import os
import tempfile

import modal

SAM_PACKAGE = "git+https://github.com/facebookresearch/sam3.git"  # VERIFY: SAM 3.1 tag/branch
GPU = "L40S"  # 48 GB; ~half the price of an H100 and plenty for one SAM model

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.7.1",
        "torchvision==0.22.1",
        "numpy<2.3",
        "opencv-python-headless<5",
        "pillow",
        "fastapi[standard]",
        "python-multipart",
        "huggingface_hub",
    )
    .pip_install(SAM_PACKAGE)
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


def box_iou(a, b) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@app.cls(
    gpu=GPU,
    secrets=[modal.Secret.from_name("veil-sam3")],
    scaledown_window=300,  # stay warm between demo runs; set min_containers=1 on demo day
    timeout=900,
)
class Sam3Service:
    @modal.enter()
    def load(self):
        import torch
        from huggingface_hub import login

        if os.environ.get("HF_TOKEN"):
            login(token=os.environ["HF_TOKEN"])
        # VERIFY: builder name and checkpoint selection for SAM 3.1.
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        torch.backends.cuda.matmul.allow_tf32 = True
        self.model = build_sam3_image_model()
        self.processor = Sam3Processor(self.model)

    # -- model calls ---------------------------------------------------------

    def _detect(self, state, text: str, threshold: float):
        """Text-prompted detections on an image state: list of (box, mask, score)."""
        import numpy as np

        if hasattr(self.processor, "reset_all_prompts"):
            self.processor.reset_all_prompts(state)
        # VERIFY: output keys and tensor shapes (masks N×1×H×W, boxes N×4 xyxy px).
        out = self.processor.set_text_prompt(state=state, prompt=text)
        masks, boxes, scores = out["masks"], out["boxes"], out["scores"]
        dets = []
        for i in range(len(scores)):
            score = float(scores[i])
            if score < threshold:
                continue
            full = masks[i].squeeze().float().cpu().numpy() > 0.5
            ys, xs = np.nonzero(full)
            if len(xs) == 0:
                continue
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            dets.append(((x0, y0, x1, y1), full[y0:y1, x0:x1], score))
        return dets

    def _frames(self, path: str, indices=None, stride: int = 1):
        import cv2
        from PIL import Image

        cap = cv2.VideoCapture(path)
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if (indices is None and i % stride == 0) or (indices is not None and i in indices):
                yield i, Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            i += 1
        cap.release()

    # -- HTTP ----------------------------------------------------------------

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile

        api = FastAPI(title="veil-sam3")
        token = os.environ.get("VEIL_TOKEN", "")

        def check(auth: str | None):
            if token and auth != f"Bearer {token}":
                raise HTTPException(401, "missing or wrong bearer token")

        async def save(video: UploadFile) -> str:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(await video.read())
            tmp.close()
            return tmp.name

        @api.get("/health")
        def health():
            return {"ok": True, "gpu": GPU}

        @api.post("/segment")
        async def segment(video: UploadFile = File(...), payload: str = Form(...), authorization: str | None = Header(None)):
            check(authorization)
            body = json.loads(payload)
            prompts = body["prompts"]
            threshold = float(body.get("score_threshold", 0.45))
            stride = int(body.get("frame_stride", 2))  # the client bridges skipped frames
            path = await save(video)
            results = {p["id"]: [] for p in prompts}
            try:
                for f, img in self._frames(path, stride=stride):
                    state = self.processor.set_image(img)  # one backbone pass per frame
                    for p in prompts:
                        for box, crop, score in self._detect(state, p["text"], threshold):
                            results[p["id"]].append({"f": f, "score": round(score, 4), "box": list(box), "rle": rle_encode(crop)})
            finally:
                os.unlink(path)
            # Untracked: one object per detection; the client links them by IoU.
            return {
                "tracked": False,
                "results": [
                    {"id": pid, "objects": [{"obj_id": k, "detections": [d]} for k, d in enumerate(dets)]}
                    for pid, dets in results.items()
                ],
            }

        @api.post("/refine")
        async def refine(video: UploadFile = File(...), payload: str = Form(...), authorization: str | None = Header(None)):
            """Re-detect a leaked object at a low threshold and follow it from the seed box."""
            check(authorization)
            reqs = json.loads(payload)["requests"]
            path = await save(video)
            wanted = {f for r in reqs for f in range(max(0, r["frame_index"] - r["window"]), r["frame_index"] + r["window"] + 1)}
            found: list[dict[int, list]] = [{} for _ in reqs]
            try:
                for f, img in self._frames(path, indices=wanted):
                    state = self.processor.set_image(img)
                    for r, dets in zip(reqs, found):
                        if abs(f - r["frame_index"]) <= r["window"]:
                            dets[f] = self._detect(state, r["text"], 0.2)
            finally:
                os.unlink(path)
            results = []
            for r, dets in zip(reqs, found):
                chosen = {}
                for direction in (1, -1):
                    prev = tuple(r["box"])
                    f = r["frame_index"] if direction == 1 else r["frame_index"] - 1
                    while f in dets:
                        best = max(dets[f], key=lambda d: box_iou(prev, d[0]), default=None)
                        if best is None or box_iou(prev, best[0]) < 0.2:
                            break
                        chosen[f] = best
                        prev = best[0]
                        f += direction
                detections = [
                    {"f": f, "score": round(s, 4), "box": list(b), "rle": rle_encode(m)} for f, (b, m, s) in sorted(chosen.items())
                ]
                results.append({"id": r["id"], "objects": [{"obj_id": 1, "detections": detections}] if detections else []})
            return {"tracked": True, "results": results}

        return api
