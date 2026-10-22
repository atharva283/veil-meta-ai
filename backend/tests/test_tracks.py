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
