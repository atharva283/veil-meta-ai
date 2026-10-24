import numpy as np

from app.pipeline.tracks import FrameMask, Track, fill_gaps, rle_decode, rle_encode, select, subtract


def _box_track(tid: str, boxes: dict[int, tuple[int, int, int, int]]) -> Track:
    t = Track(tid, "x")
    for f, (x0, y0, x1, y1) in boxes.items():
        t.frames[f] = FrameMask((x0, y0, x1, y1), np.ones((y1 - y0, x1 - x0), dtype=bool))
    return t


def test_rle_roundtrip():
    rng = np.random.default_rng(0)
    for shape in [(1, 1), (7, 13), (40, 25)]:
        m = rng.random(shape) > 0.6
        assert np.array_equal(rle_decode(rle_encode(m), shape), m)
    starts_on = np.ones((3, 3), dtype=bool)
    assert rle_encode(starts_on)[0] == 0


def test_select_and_subtract_partition_by_overlap():
    face_a = _box_track("face-a", {0: (10, 10, 20, 20), 1: (11, 10, 21, 20)})
    face_b = _box_track("face-b", {0: (100, 10, 110, 20), 1: (101, 10, 111, 20)})
    her = _box_track("person", {0: (5, 5, 30, 80), 1: (6, 5, 31, 80)})
    assert [t.track_id for t in select([face_a, face_b], [her])] == ["face-a"]
    assert [t.track_id for t in subtract([face_a, face_b], [her])] == ["face-b"]


def test_fill_gaps_bridges_short_dropouts_only():
    t = _box_track("t", {0: (0, 0, 10, 10), 4: (40, 0, 50, 10), 20: (0, 0, 10, 10)})
    fill_gaps(t, max_gap=4)
    assert [f for f in range(5)] == [f for f in t.frame_indices if f <= 4]
    assert t.frames[2].box == (20, 0, 30, 10)
    assert 10 not in t.frames


def test_identity_swap_is_split_per_frame():
    # Tracking follows her face for frames 0-4, then swaps onto a passing stranger's face.
    swapped = _box_track("face-1", {f: (10, 10, 20, 20) if f < 5 else (200, 10, 210, 20) for f in range(10)})
    her = _box_track("person", {f: (5, 5, 30, 80) for f in range(10)})
    kept, redacted = select([swapped], [her]), subtract([swapped], [her])
    assert kept[0].frame_indices == [0, 1, 2, 3, 4]
    assert redacted[0].frame_indices == [5, 6, 7, 8, 9]  # the stranger is redacted, not kept with her


def test_unsplit_tracks_keep_their_identity():
    face = _box_track("face-1", {0: (10, 10, 20, 20), 1: (11, 10, 21, 20)})
    her = _box_track("person", {0: (5, 5, 30, 80), 1: (6, 5, 31, 80)})
    assert select([face], [her])[0] is face
    assert subtract([face], []) == [face] and subtract([face], [])[0] is face


def test_compose_redacts_the_swapped_frames_and_reports_both_parts():
    from app.orchestrator import _compose
    from app.schemas import Plan

    swapped = _box_track("face-1", {f: (10, 10, 20, 20) if f < 5 else (200, 10, 210, 20) for f in range(10)})
    swapped.concept = "face"
    her = _box_track("person-1", {f: (5, 5, 30, 80) for f in range(10)})
    plan = Plan(steps=[
        {"id": "s1", "op": "segment", "concept": "face"},
        {"id": "s2", "op": "segment", "concept": "person"},
        {"id": "s3", "op": "subtract", "source": "s1", "reference": "s2", "note": "faces except her"},
        {"id": "s4", "op": "effect", "source": "s3", "effect": "blur"},
    ])  # fmt: skip
    comp = _compose(plan, {"face": [swapped], "person": [her]})
    (layer,) = comp.layers
    assert [(t.track_id, t.frame_indices) for t in layer.tracks] == [("face-1", [5, 6, 7, 8, 9])]
    assert [(t.track_id, t.frame_indices) for t in comp.kept] == [("face-1-visible", [0, 1, 2, 3, 4])]
