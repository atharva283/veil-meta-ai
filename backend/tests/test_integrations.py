"""Contract tests for the live integrations, using fake HTTP responses.

These pin down the parsing we can't exercise against real APIs until the
hackathon keys arrive: Llama tool calls, Muse Spark findings, SAM wire format.
"""

import json

import httpx
import numpy as np
import pytest

from app.auditor import muse_spark
from app.auditor.base import AuditPolicy, AuditSample
from app.planner import llama
from app.planner.llama import EXAMPLE_PLAN
from app.pipeline.tracks import FrameMask, rle_encode
from app.segmenter.modal_sam import _decode_result


def _patch_http(monkeypatch, module, handler):
    real = httpx.Client
    monkeypatch.setattr(module.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))


def _chat(message: dict) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": message}]})


def test_llama_tool_call_is_parsed(monkeypatch):
    def handler(request: httpx.Request):
        body = json.loads(request.content)
        assert body["tool_choice"]["function"]["name"] == "submit_plan"
        call = {"function": {"name": "submit_plan", "arguments": json.dumps(EXAMPLE_PLAN)}}
        return _chat({"role": "assistant", "content": None, "tool_calls": [call]})

    _patch_http(monkeypatch, llama, handler)
    plan = llama.plan("Blur faces")
    assert [s.op for s in plan.steps][:2] == ["segment", "segment"]
    assert plan.source == llama.settings.planner_model


def test_llama_steps_sent_as_a_json_string_are_decoded(monkeypatch):
    """Llama 3.1 8B via Ollama encoded the steps array as a string (Colab, Oct 26)."""

    def handler(request: httpx.Request):
        args = {**EXAMPLE_PLAN, "steps": json.dumps(EXAMPLE_PLAN["steps"])}
        call = {"function": {"name": "submit_plan", "arguments": json.dumps(args)}}
        return _chat({"role": "assistant", "content": None, "tool_calls": [call]})

    _patch_http(monkeypatch, llama, handler)
    plan = llama.plan("Blur faces")
    assert len(plan.steps) == len(EXAMPLE_PLAN["steps"])
    assert plan.source == llama.settings.planner_model


def test_llama_plain_json_reply_and_self_repair(monkeypatch):
    replies = iter([
        _chat({"content": 'Here: {"summary": "x", "steps": [{"id": "s1", "op": "effect", "source": "s9", "effect": "blur"}]}'}),
        _chat({"content": json.dumps(EXAMPLE_PLAN)}),
    ])
    seen = []

    def handler(request: httpx.Request):
        seen.append(json.loads(request.content)["messages"][-1]["content"])
        return next(replies)

    _patch_http(monkeypatch, llama, handler)
    plan = llama.plan("Blur faces")
    assert len(plan.steps) == len(EXAMPLE_PLAN["steps"])
    assert "invalid" in seen[1]  # the second request carried the validation error


def test_muse_spark_findings_map_to_pixels_and_policy(monkeypatch):
    def handler(request: httpx.Request):
        content = json.loads(request.content)["messages"][0]["content"]
        assert sum(1 for c in content if c["type"] == "image_url") == 2
        findings = [
            {"image": 2, "concept": "Face", "box": [100, 200, 300, 400], "reason": "profile face"},
            {"image": 1, "concept": "dog", "box": [0, 0, 10, 10]},  # not in policy: dropped
        ]
        return _chat({"content": json.dumps({"findings": findings})})

    _patch_http(monkeypatch, muse_spark, handler)
    video = type("V", (), {"meta": type("M", (), {"width": 1000, "height": 500})()})()
    samples = [AuditSample(i, i / 24, np.zeros((500, 1000, 3), np.uint8)) for i in (0, 12)]
    out = muse_spark.MuseSparkAuditor().audit(video, samples, AuditPolicy(targets=["face"]))
    assert len(out) == 1
    assert out[0].frame_index == 12 and out[0].concept == "face"
    assert out[0].box == pytest.approx((100, 100, 300, 200))


def test_sam_wire_format_untracked_detections_are_linked():
    mask = np.zeros((20, 10), bool)
    mask[5:15, 2:8] = True
    dets = [{"f": f, "score": 0.9, "box": [10 + f, 10, 20 + f, 30], "rle": rle_encode(mask)} for f in range(0, 12, 2)]
    result = {"id": "p0", "objects": [{"obj_id": i, "detections": [d]} for i, d in enumerate(dets)]}
    tracks = _decode_result(result, "face", tracked=False, width=100, height=100)
    assert len(tracks) == 1
    t = tracks[0]
    assert t.frame_indices == list(range(0, 11))  # gaps bridged
    fm: FrameMask = t.frames[0]
    assert np.array_equal(fm.mask, mask)
