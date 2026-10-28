"""SAM 3.1 wrapper against a fake predictor that speaks the sam3 request API."""

from pathlib import Path

import numpy as np
import pytest

from app.cache import SampleCache
from app.jobs import jobs
from app.pipeline.video_io import VideoMeta
from app.segmenter import sam3_core
from app.segmenter.base import RefineRequest
from app.segmenter.local_sam import LocalSam3Segmenter
from app.videos import VideoRecord

W, H = 64, 48


def _out(objs):
    """One frame of predictor output for [(obj_id, prob, (x0, y0, x1, y1)), ...]."""
    masks = np.zeros((len(objs), H, W), dtype=bool)
    for k, (_, _, (x0, y0, x1, y1)) in enumerate(objs):
        masks[k, y0:y1, x0:x1] = True
    return {
        "out_obj_ids": np.array([o[0] for o in objs], dtype=np.int64),
        "out_probs": np.array([o[1] for o in objs], dtype=np.float32),
        "out_boxes_xywh": np.zeros((len(objs), 4), dtype=np.float32),
        "out_binary_masks": masks,
    }


class FakePredictor:
    """Two faces: id 1 on every frame, id 2 entering at frame 2 with a low score on frame 3."""

    def __init__(self):
        self.requests, self.closed = [], 0

    def handle_request(self, request):
        self.requests.append(request)
        t = request["type"]
        if t == "start_session":
            return {"session_id": "s1"}
        if t == "close_session":
            self.closed += 1
            return {"is_success": True}
        if t == "reset_session":
            return {"is_success": True}
        if t == "add_prompt":
            if "points" in request:
                return {"frame_index": request["frame_index"], "outputs": _out([(1, 0.9, (30, 20, 40, 30)), (7, 0.9, (0, 0, 5, 5))])}
            return {"frame_index": 0, "outputs": _out([(1, 0.95, (10, 10, 20, 20))])}
        raise AssertionError(t)

    def handle_stream_request(self, request):
        self.requests.append(request)
        for f in range(1, 4):
            objs = [(1, 0.95, (10 + f, 10, 20 + f, 20))]
            if f >= 2:
                objs.append((2, 0.9 if f == 2 else 0.1, (40, 5, 50, 15)))
            yield {"frame_index": f, "outputs": _out(objs)}


def test_segment_tracks_ids_through_the_clip_and_closes_the_session():
    pred = FakePredictor()
    data = sam3_core.segment(pred, "clip.mp4", [{"id": "p0", "text": "face"}], threshold=0.45)
    (res,) = data["results"]
    assert data["tracked"] and res["id"] == "p0"
    by_id = {o["obj_id"]: o["detections"] for o in res["objects"]}
    assert [d["f"] for d in by_id[1]] == [0, 1, 2, 3]
    assert by_id[1][3]["box"] == [13, 10, 23, 20]
    assert [d["f"] for d in by_id[2]] == [2]  # frame 3 is under the threshold
    prompt = next(r for r in pred.requests if r["type"] == "add_prompt")
    assert prompt["text"] == "face" and prompt["frame_index"] == 0
    assert pred.closed == 1


def test_segment_streams_without_caching_and_prunes_old_tracker_frames(monkeypatch):
    """Only the last KEEP_FRAMES frames of tracker state stay in memory during a forward pass."""
    monkeypatch.setattr(sam3_core, "KEEP_FRAMES", 2)
    pred = FakePredictor()
    frames = {f: object() for f in range(4)}
    tracker = {"output_dict": {"non_cond_frame_outputs": dict(frames)}, "output_dict_per_obj": {0: {"non_cond_frame_outputs": dict(frames)}}}
    pred._all_inference_states = {"s1": {"state": {"sam2_inference_states": [tracker]}}}
    seen = []
    sam3_core.segment(pred, "clip.mp4", [{"id": "p0", "text": "face"}], on_frame=seen.append)
    stream = next(r for r in pred.requests if r["type"] == "propagate_in_video")
    assert stream["evict_cached_frame_outputs"] is True
    assert seen == [1, 2, 3]
    assert sorted(tracker["output_dict"]["non_cond_frame_outputs"]) == [2, 3]  # 0 and 1 dropped at frames 2 and 3
    assert sorted(tracker["output_dict_per_obj"][0]["non_cond_frame_outputs"]) == [2, 3]


def _video(path, n):
    import cv2

    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (W, H))
    for i in range(n):
        w.write(np.full((H, W, 3), i * 20, dtype=np.uint8))
    w.release()
    return str(path)


def test_long_clip_is_tracked_in_pieces_joined_at_the_shared_frame(monkeypatch, tmp_path):
    """Pieces of 4 frames over 10 share frames 3 and 6; object 1 keeps one id across both seams."""
    monkeypatch.setenv("VEIL_SAM_PIECE", "4")
    clip = _video(tmp_path / "clip.mp4", 10)
    assert sam3_core._pieces(10, 4) == [(0, 4), (3, 7), (6, 10)]
    pred = FakePredictor()
    seen = []
    data = sam3_core.segment(pred, clip, [{"id": "p0", "text": "face"}], on_frame=seen.append)
    by_id = {o["obj_id"]: [d["f"] for d in o["detections"]] for o in data["results"][0]["objects"]}
    assert by_id[1] == list(range(10))
    # Object 2 first appears mid-piece, so each piece's copy is a new object (nothing to continue).
    assert sorted(by_id.values())[1:] == [[2], [5], [8]]
    assert seen == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    starts = [r for r in pred.requests if r["type"] == "start_session"]
    assert len(starts) == 3 and pred.closed == 3
    assert all(r["resource_path"].endswith(".avi") for r in starts)
    assert not any(Path(r["resource_path"]).exists() for r in starts)  # pieces are cleaned up


def test_refine_on_a_long_clip_loads_only_the_window(monkeypatch, tmp_path):
    monkeypatch.setenv("VEIL_SAM_PIECE", "4")
    clip = _video(tmp_path / "clip.mp4", 10)
    pred = FakePredictor()
    data = sam3_core.refine(pred, clip, [{"id": "r0", "frame_index": 6, "box": [28, 18, 44, 34], "window": 2}], W, H)
    prompt = next(r for r in pred.requests if r["type"] == "add_prompt")
    assert prompt["frame_index"] == 2  # frame 6 of the clip is frame 2 of the piece [4, 9)
    frames = [d["f"] for d in data["results"][0]["objects"][0]["detections"]]
    assert frames == [5, 6, 7]  # the fake answers on piece frames 1-3


def test_refine_prompts_the_box_center_as_a_relative_point():
    pred = FakePredictor()
    data = sam3_core.refine(pred, "clip.mp4", [{"id": "r0", "text": "face", "frame_index": 2, "box": [28, 18, 44, 34], "window": 5}], W, H)
    prompt = next(r for r in pred.requests if r["type"] == "add_prompt")
    assert prompt["points"] == [[36 / W, 26 / H]] and prompt["obj_id"] == 1 and prompt["point_labels"] == [1]
    stream = next(r for r in pred.requests if r["type"] == "propagate_in_video")
    assert stream["start_frame_index"] == 2 and stream["max_frame_num_to_track"] == 5
    (res,) = data["results"]
    assert [o["obj_id"] for o in res["objects"]] == [1]  # only the prompted object


def test_refine_on_the_last_frame_propagates_backward_only(monkeypatch, tmp_path):
    """A leak on a clip's last frame crashed sam3's backward pass (KeyError 58, Colab Oct 27)."""
    monkeypatch.setenv("VEIL_SAM_PIECE", "4")
    clip = _video(tmp_path / "clip.mp4", 10)
    pred = FakePredictor()
    sam3_core.refine(pred, clip, [
        {"id": "last", "frame_index": 9, "box": [28, 18, 44, 34], "window": 2},
        {"id": "first", "frame_index": 0, "box": [28, 18, 44, 34], "window": 2},
        {"id": "mid", "frame_index": 5, "box": [28, 18, 44, 34], "window": 2},
    ], W, H)
    directions = [r["propagation_direction"] for r in pred.requests if r["type"] == "propagate_in_video"]
    assert directions == ["backward", "forward", "both"]


def test_one_failing_refine_request_does_not_cancel_the_others(monkeypatch):
    pred = FakePredictor()
    real = pred.handle_request
    calls = {"n": 0}

    def flaky(request):
        if request["type"] == "add_prompt":
            calls["n"] += 1
            if calls["n"] == 1:
                raise KeyError(58)
        return real(request)

    monkeypatch.setattr(pred, "handle_request", flaky)
    box = [28, 18, 44, 34]
    data = sam3_core.refine(pred, "clip.mp4", [{"id": "a", "frame_index": 2, "box": box, "window": 5}, {"id": "b", "frame_index": 2, "box": box, "window": 5}], W, H)
    by_id = {r["id"]: r["objects"] for r in data["results"]}
    assert by_id["a"] == [] and by_id["b"]
    assert pred.closed == 2  # both sessions closed, including the failed one


def test_local_segmenter_decodes_tracks(monkeypatch):
    pred = FakePredictor()
    monkeypatch.setattr(sam3_core, "load_predictor", lambda precision: (pred, "SAM 3.1 (Fake GPU, fp16)"))
    seg = LocalSam3Segmenter()
    seg.load()
    video = VideoRecord("v", "v", Path("clip.mp4"), VideoMeta(W, H, 10.0, 4, False))
    (face,) = [t for t in seg.segment(video, ["face"])["face"] if 3 in t.frames]
    assert seg.name == "SAM 3.1 (Fake GPU, fp16)"
    assert face.frames[0].box == (10, 10, 20, 20) and face.frames[3].box == (13, 10, 23, 20)
    (patch,) = seg.refine(video, [RefineRequest("face", 2, (28, 18, 44, 34))])
    assert patch.track_id.startswith("patch-") and 2 in patch.frames


def test_init_state_ignores_kwargs_the_model_does_not_take():
    """sam3's start_session passes offload_state_to_cpu, which the SAM 3.1 model rejects."""

    class Model:
        def init_state(self, resource_path, offload_video_to_cpu=False, async_loading_frames=False):
            return {"path": resource_path, "offload": offload_video_to_cpu}

    model = Model()
    sam3_core._drop_unknown_init_kwargs(model)
    state = model.init_state(resource_path="a.mp4", offload_video_to_cpu=True, offload_state_to_cpu=False, async_loading_frames=True)
    assert state == {"path": "a.mp4", "offload": True}


def test_set_precision_rewrites_and_restores(tmp_path):
    pkg = tmp_path / "sam3"
    (pkg / "model").mkdir(parents=True)
    (pkg / "perflib").mkdir()
    src = "with torch.autocast(device_type='cuda', dtype=torch.bfloat16):\n    x = x.to(torch.bfloat16)\n"
    (pkg / "model" / "a.py").write_text(src)
    (pkg / "perflib" / "b.py").write_text("y = 1\n")
    assert sam3_core.set_precision("fp16", pkg) == 1
    assert "torch.bfloat16" not in (pkg / "model" / "a.py").read_text() and "torch.float16" in (pkg / "model" / "a.py").read_text()
    assert sam3_core.set_precision("fp16", pkg) == 0  # idempotent
    assert sam3_core.set_precision("fp32", pkg) == 1 and "torch.float32" in (pkg / "model" / "a.py").read_text()
    sam3_core.set_precision("bf16", pkg)
    assert (pkg / "model" / "a.py").read_text() == src
    with pytest.raises(ValueError):
        sam3_core.set_precision("int8", pkg)


def test_set_precision_lets_flash_only_attention_fall_back(tmp_path):
    """sam3's decoder pins SDPA to Flash, which a T4 (sm75) doesn't have."""
    pkg = tmp_path / "sam3"
    (pkg / "model").mkdir(parents=True)
    (pkg / "perflib").mkdir()
    src = "with sdpa_kernel(SDPBackend.FLASH_ATTENTION):\n    out = F.scaled_dot_product_attention(q, k, v)\n"
    (pkg / "model" / "decoder.py").write_text(src)
    assert sam3_core.set_precision("fp16", pkg) == 1
    patched = (pkg / "model" / "decoder.py").read_text()
    assert "SDPBackend.EFFICIENT_ATTENTION" in patched and "SDPBackend.MATH" in patched
    assert sam3_core.set_precision("fp16", pkg) == 0  # idempotent
    compile(patched, "decoder.py", "exec")


def test_rule_based_plan_can_be_cached_but_mock_masks_cannot(tmp_path):
    from app.videos import store

    store.load_samples()
    job = jobs.run_now("sample-street_crossing", "Blur every face.")
    snap_path = jobs.jobs_dir / job.id / "job.json"
    cache = SampleCache(tmp_path / "cache")
    with pytest.raises(ValueError, match="mock SAM"):
        cache.freeze(snap_path.parent, allow_mock_planner=True)  # masks are mock
    snap = snap_path.read_text(encoding="utf-8")
    snap_path.write_text(snap.replace('"mock SAM 3.1"', '"SAM 3.1 (Tesla T4, fp16)"').replace('"mock Muse Spark"', '"none (SAM re-check only)"'), encoding="utf-8")
    with pytest.raises(ValueError, match="mock planner"):
        cache.freeze(snap_path.parent)
    entry = cache.freeze(snap_path.parent, allow_mock_planner=True)
    assert entry.models["planner"] == "mock planner"
