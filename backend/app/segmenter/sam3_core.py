"""SAM 3.1 video segmentation, shared by the Modal service and in-process runs.

Wraps the official predictor (facebookresearch/sam3, `build_sam3_predictor`)
and speaks Veil's segmenter wire format:

    {"tracked": true, "results": [{"id", "objects": [{"obj_id", "detections": [{"f","score","box","rle"}]}]}]}

so backend/app/segmenter/modal_sam.py decodes both the same way. This module
imports torch and sam3 lazily and has no other dependency on the backend, so
the Modal image can ship it as a single file.

Precision: SAM 3.1 hardcodes bfloat16 (autocast contexts, some applied as
decorators at import time, plus explicit casts). GPUs before Ampere (the
Colab/Kaggle T4) have no native bf16, so `set_precision` rewrites the
installed package's `torch.bfloat16` to fp16 or fp32 before it's imported.
fp32 turns CUDA autocast off entirely (slowest, safest).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

SAM_COMMIT = "f1647e8d0c1d280991f4d1b7d974c2704537fe66"  # facebookresearch/sam3 main, 2026-10-23
MAX_OBJECTS = 64  # per prompt; SAM 3.1's default of 16 is too few for a crowd
# Tracker state kept per frame during a forward pass. SAM 3.1 reads memories from the last
# 7 frames and object pointers from the last 16; anything older is only used for
# interactive corrections, which `segment` doesn't do.
KEEP_FRAMES = 48
PRECISIONS = {"bf16": "torch.bfloat16", "fp16": "torch.float16", "fp32": "torch.float32"}
_PATCHED_DIRS = ("model", "perflib")


def rle_encode(mask):
    """Row-major run lengths over the crop, starting with a run of zeros."""
    import numpy as np

    flat = mask.astype(np.uint8).ravel()
    if flat.size == 0:
        return []
    changes = np.flatnonzero(np.diff(flat)) + 1
    runs = np.diff(np.concatenate(([0], changes, [flat.size]))).tolist()
    return runs if flat[0] == 0 else [0, *runs]


# -- precision ---------------------------------------------------------------


def set_precision(precision: str, package_dir: Path | None = None) -> int:
    """Point the installed sam3 package at `precision`; returns files changed.

    Must run before `import sam3` in this process. Keeps a pristine copy of each
    file (`.bf16.orig`) so switching back and forth is safe.
    """
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {', '.join(PRECISIONS)}")
    if package_dir is None:
        spec = importlib.util.find_spec("sam3")
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError("sam3 isn't installed")
        package_dir = Path(next(iter(spec.submodule_search_locations)))
    changed = 0
    for sub in _PATCHED_DIRS:
        for path in (package_dir / sub).glob("*.py"):
            orig = path.with_suffix(".py.bf16.orig")
            source = orig.read_text(encoding="utf-8") if orig.exists() else path.read_text(encoding="utf-8")
            if "torch.bfloat16" not in source:
                continue
            if not orig.exists():
                orig.write_text(source, encoding="utf-8")
            target = source.replace("torch.bfloat16", PRECISIONS[precision])
            if path.read_text(encoding="utf-8") != target:
                path.write_text(target, encoding="utf-8")
                changed += 1
    return changed


def auto_precision() -> str:
    import torch

    return "bf16" if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else "fp16"


def load_predictor(precision: str = "auto"):
    """Build the SAM 3.1 video predictor; returns (predictor, description)."""
    import os

    # Before the first CUDA allocation: lets freed blocks be reused across sizes (less fragmentation).
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    precision = auto_precision() if precision == "auto" else precision
    set_precision(precision)
    import torch
    from sam3.model_builder import build_sam3_predictor

    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
    # Flash-Attention 3 kernels need Hopper; everything else takes the PyTorch path.
    predictor = build_sam3_predictor(version="sam3.1", use_fa3=False, max_num_objects=MAX_OBJECTS)
    _drop_unknown_init_kwargs(predictor.model)
    if _offload():
        # Post-process one frame at a time instead of 16 (upstream: "set to 1 to disable batching");
        # a batch holds 16 frames of full-resolution masks for every object on the GPU at once.
        predictor.model.postprocess_batch_size = 1
    gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"
    return predictor, f"SAM 3.1 ({gpu}, {precision})"


def _drop_unknown_init_kwargs(model) -> None:
    """Make `model.init_state` ignore keyword arguments it doesn't take.

    Upstream bug (sam3 @ SAM_COMMIT): `Sam3BasePredictor.start_session` always passes
    `offload_state_to_cpu`, which the SAM 3.1 multiplex model's `init_state` doesn't
    accept, so every start_session raises TypeError. add_prompt and propagate_in_video
    already filter their kwargs by signature upstream; this does the same for init_state.
    """
    import functools
    import inspect

    init = model.init_state
    params = inspect.signature(init).parameters
    if any(p.kind is p.VAR_KEYWORD for p in params.values()):
        return

    @functools.wraps(init)
    def init_state(*args, **kwargs):
        return init(*args, **{k: v for k, v in kwargs.items() if k in params})

    model.init_state = init_state


# -- inference ---------------------------------------------------------------


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


def _offload() -> bool:
    """Keep video frames in CPU memory on GPUs under 20 GB (the 15 GB T4)."""
    try:
        import torch
    except ImportError:
        return False

    return torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 20 * 2**30


def _start(predictor, video_path: str) -> str:
    return predictor.handle_request(request={"type": "start_session", "resource_path": video_path, "offload_video_to_cpu": _offload()})["session_id"]


def _objects(found: dict) -> list[dict]:
    return [{"obj_id": oid, "detections": [d for _, d in sorted(frames.items())]} for oid, frames in sorted(found.items())]


def _prune(predictor, session: str, frame_idx: int) -> None:
    """Drop tracker state for the frame `KEEP_FRAMES` behind `frame_idx` (forward pass only).

    sam3 keeps every frame's memory features, image features and masks for the whole
    session, which outgrows a 15 GB GPU in a few seconds of a busy clip. Missing frames
    are skipped as padding when the tracker gathers its memory.
    """
    state = getattr(predictor, "_all_inference_states", {}).get(session, {}).get("state", {})
    old = frame_idx - KEEP_FRAMES
    for tracker in state.get("sam2_inference_states", []):
        tracker["output_dict"]["non_cond_frame_outputs"].pop(old, None)
        for per_obj in tracker["output_dict_per_obj"].values():
            per_obj["non_cond_frame_outputs"].pop(old, None)


def _free_cuda() -> None:
    """Hand the allocator's cached blocks back, e.g. for Ollama loading the planner between runs."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def segment(predictor, video_path: str, prompts: list[dict], threshold: float = 0.45, on_frame=None) -> dict:
    """Text-prompt each concept on frame 0 and track every instance through the clip.

    Objects that enter later are picked up as propagation reaches them. `on_frame(frame_idx)`
    is called after each frame (the smoke test uses it to report memory).
    """
    session = _start(predictor, video_path)
    results = []
    try:
        for p in prompts:
            found: dict = {}
            # A new text prompt resets the session's tracking state.
            out = predictor.handle_request(request={
                "type": "add_prompt", "session_id": session, "frame_index": 0, "text": p["text"], "output_prob_thresh": threshold,
            })  # fmt: skip
            _collect(found, out["frame_index"], out["outputs"], threshold)
            # evict_cached_frame_outputs: drop each frame's full-resolution masks once they're
            # streamed to us instead of keeping them on the GPU for the rest of the session.
            for r in predictor.handle_stream_request(request={
                "type": "propagate_in_video", "session_id": session, "propagation_direction": "forward",
                "start_frame_index": 0, "output_prob_thresh": threshold, "evict_cached_frame_outputs": True,
            }):  # fmt: skip
                _collect(found, r["frame_index"], r["outputs"], threshold)
                _prune(predictor, session, r["frame_index"])
                if on_frame:
                    on_frame(r["frame_index"])
            results.append({"id": p["id"], "objects": _objects(found)})
    finally:
        predictor.handle_request(request={"type": "close_session", "session_id": session})
        _free_cuda()
    return {"tracked": True, "results": results}


def refine(predictor, video_path: str, requests: list[dict], width: int, height: int) -> dict:
    """Follow one object per request from a point prompt, `window` frames either side.

    The point is the request's "point" (pixels) or the center of its "box".
    """
    session = _start(predictor, video_path)
    results = []
    try:
        for r in requests:
            x, y = r.get("point") or ((r["box"][0] + r["box"][2]) / 2, (r["box"][1] + r["box"][3]) / 2)
            predictor.handle_request(request={"type": "reset_session", "session_id": session})
            found: dict = {}
            out = predictor.handle_request(request={
                "type": "add_prompt", "session_id": session, "frame_index": r["frame_index"], "obj_id": 1,
                "points": [[x / width, y / height]], "point_labels": [1], "rel_coordinates": True,
            })  # fmt: skip
            _collect(found, out["frame_index"], out["outputs"], 0.0, only_obj=1)
            for o in predictor.handle_stream_request(request={
                "type": "propagate_in_video", "session_id": session, "propagation_direction": "both",
                "start_frame_index": r["frame_index"], "max_frame_num_to_track": r["window"],
            }):  # fmt: skip
                _collect(found, o["frame_index"], o["outputs"], 0.0, only_obj=1)
            results.append({"id": r["id"], "objects": _objects(found)})
    finally:
        predictor.handle_request(request={"type": "close_session", "session_id": session})
        _free_cuda()
    return {"tracked": True, "results": results}


def smoke(clip: str, concept: str, frames: int, precision: str) -> None:
    """Segment `concept` on the first `frames` frames and report speed, memory and what was found."""
    import tempfile
    import time

    import cv2
    import torch

    cap = cv2.VideoCapture(clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for _ in range(frames):
        ok, frame = cap.read()
        if not ok:
            break
        out.write(frame)
    out.release()
    cap.release()

    t0 = time.time()
    predictor, desc = load_predictor(precision)
    t1 = time.time()
    print(f"{desc}: loaded in {t1 - t0:.0f}s", flush=True)
    gib = lambda n: f"{n / 2**30:.1f} GiB"  # noqa: E731

    def report(f: int) -> None:
        if torch.cuda.is_available() and f % 15 == 0:
            print(f"  frame {f}: {gib(torch.cuda.memory_allocated())} allocated, {gib(torch.cuda.memory_reserved())} reserved", flush=True)

    data = segment(predictor, path, [{"id": "p0", "text": concept}], on_frame=report)
    t2 = time.time()
    objects = data["results"][0]["objects"]
    dets = sum(len(o["detections"]) for o in objects)
    print(f'"{concept}" on {frames} frames: {len(objects)} tracked objects, {dets} masks, {t2 - t1:.1f}s ({frames / max(t2 - t1, 1e-6):.1f} frames/s)')
    if torch.cuda.is_available():
        print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    if not objects:
        print("WARNING: nothing found. If the clip clearly shows this concept, try another precision (fp32 is the safest).")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Smoke-test SAM 3.1 on the first frames of a clip.")
    ap.add_argument("clip")
    ap.add_argument("--concept", default="face")
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--precision", default="auto", choices=["auto", *PRECISIONS])
    a = ap.parse_args()
    smoke(a.clip, a.concept, a.frames, a.precision)
