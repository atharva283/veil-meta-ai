"""The planner evaluation's checker (T7): right plans pass, wrong ones say why."""

from app.planner import eval as ev
from app.schemas import Plan, Step

CASES = {c.id: c for c in ev.CASES}


def _plan(*steps) -> Plan:
    return Plan(steps=[Step(id=f"s{i}", **s) for i, s in enumerate(steps, start=1)])


# Llama 3.1 8B's plan for the Manchester demo instruction (Colab, Oct 27).
PINK_COAT = _plan(
    {"op": "segment", "concept": "face"},
    {"op": "segment", "concept": "person"},
    {"op": "segment", "concept": "pink coat"},
    {"op": "select", "source": "s2", "reference": "s3"},
    {"op": "subtract", "source": "s1", "reference": "s4"},
    {"op": "effect", "source": "s5", "effect": "blur"},
)


def test_a_correct_exception_plan_passes():
    assert ev.check(CASES["pink-coat"], PINK_COAT) == []


def test_missing_exception_wrong_effect_and_invented_targets_are_caught():
    no_exception = _plan({"op": "segment", "concept": "face"}, {"op": "effect", "source": "s1", "effect": "blur"})
    assert ev.check(CASES["pink-coat"], no_exception) == ["face: exception (pink) not subtracted"]
    pixelated = _plan({"op": "segment", "concept": "faces"}, {"op": "effect", "source": "s1", "effect": "pixelate"})
    assert ev.check(CASES["faces"], pixelated) == ["face: pixelate instead of blur"]
    extra = _plan(
        {"op": "segment", "concept": "logo"}, {"op": "effect", "source": "s1", "effect": "blackout"},
        {"op": "segment", "concept": "person"}, {"op": "effect", "source": "s3", "effect": "blur"},
    )  # fmt: skip
    assert ev.check(CASES["logo"], extra) == ["invented: blur on 'person'"]


def test_a_person_exception_must_not_filter_plates():
    plan = _plan(
        {"op": "segment", "concept": "license plate"},
        {"op": "segment", "concept": "face"},
        {"op": "segment", "concept": "person"},
        {"op": "segment", "concept": "bicycle"},
        {"op": "select", "source": "s3", "reference": "s4"},
        {"op": "subtract", "source": "s1", "reference": "s5"},
        {"op": "subtract", "source": "s2", "reference": "s5"},
        {"op": "effect", "source": "s6", "effect": "blur"},
        {"op": "effect", "source": "s7", "effect": "blur"},
    )
    assert ev.check(CASES["bicycle"], plan) == ["plate: person exception wrongly applied"]


def test_union_and_selection_resolve_to_their_concepts():
    plan = _plan(
        {"op": "segment", "concept": "license plate"},
        {"op": "segment", "concept": "red car"},
        {"op": "select", "source": "s1", "reference": "s2"},
        {"op": "effect", "source": "s3", "effect": "blur"},
    )
    assert ev.check(CASES["red-car"], plan) == []
    screens = _plan(
        {"op": "segment", "concept": "laptop screen"}, {"op": "segment", "concept": "monitor"},
        {"op": "union", "inputs": ["s1", "s2"]}, {"op": "effect", "source": "s3", "effect": "blackout"},
        {"op": "segment", "concept": "name badge"}, {"op": "effect", "source": "s5", "effect": "pixelate"},
    )  # fmt: skip
    assert ev.check(CASES["screens-badges"], screens) == []


def test_every_case_runs_on_the_rule_based_planner():
    results = [ev.run_case(c, live=False) for c in ev.CASES]
    assert all(r.valid for r in results)
    assert {r.case for r in results if not r.correct} == {"but-not", "bicycle", "red-car"}
