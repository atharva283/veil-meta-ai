import pytest

from app.planner import mock
from app.schemas import Plan, Step

DEMO = "Blur every face and license plate except the woman in the red jacket, and black out phone screens."


def _ops(plan: Plan) -> list[str]:
    return [s.label for s in plan.steps]


def test_demo_instruction_decomposes_the_exception():
    plan = mock.plan(DEMO)
    segments = [s.concept for s in plan.steps if s.op == "segment"]
    assert segments == ["face", "license plate", "person", "red jacket", "phone screen"]
    select = next(s for s in plan.steps if s.op == "select")
    assert select.note == "the woman in the red jacket"
    # The person exception filters faces only; plates can't belong to her.
    subtracts = [s for s in plan.steps if s.op == "subtract"]
    assert len(subtracts) == 1 and subtracts[0].reference == select.id
    effects = {s.note: s.effect for s in plan.steps if s.op == "effect"}
    assert effects == {"faces": "blur", "license plates": "blur", "phone screens": "blackout"}
    assert plan.summary == "Blur faces and license plates, except the woman in the red jacket; black out phone screens"
    assert not plan.warnings


def test_vehicle_exception_only_filters_vehicle_targets():
    plan = mock.plan("Blur all faces and license plates except the red car")
    sub = [s for s in plan.steps if s.op == "subtract"]
    assert len(sub) == 1
    assert next(s for s in plan.steps if s.id == sub[0].source).concept == "license plate"


def test_follow_up_extends_the_previous_plan_and_reuses_segments():
    base = mock.plan(DEMO)
    plan = mock.plan("Also hide the logo on the truck", base)
    assert plan.steps[: len(base.steps)] == base.steps
    new = plan.steps[len(base.steps) :]
    assert [s.op for s in new] == ["segment", "segment", "select", "effect"]
    assert new[0].concept == "logo" and new[1].concept == "truck"
    assert plan.summary.startswith(base.summary)


def test_effect_words_map_to_effects():
    plan = mock.plan("Pixelate all faces and black out license plates.")
    assert {s.effect for s in plan.steps if s.op == "effect"} == {"pixelate", "blackout"}


def test_unresolvable_self_reference_warns_instead_of_guessing():
    plan = mock.plan("Blur everyone except me")
    assert any("you" in w for w in plan.warnings)
    assert not any(s.op == "subtract" for s in plan.steps)


def test_me_with_clothing_hint_resolves():
    plan = mock.plan("Blur every face except me (green hoodie)")
    assert any(s.op == "segment" and s.concept == "green hoodie" for s in plan.steps)


def test_instruction_without_targets_is_rejected():
    with pytest.raises(ValueError, match="doesn't name anything"):
        mock.plan("Make it look nice")


def test_plan_validation_rejects_forward_references():
    with pytest.raises(ValueError, match="before it is defined"):
        Plan(steps=[Step(id="s1", op="effect", source="s2", effect="blur"), Step(id="s2", op="segment", concept="face")])
