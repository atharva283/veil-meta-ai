"""Planner evaluation (T7): does the planner turn instructions into the right plan?

Schema validity alone says little (a valid plan can blur the wrong thing), so
each case states what a correct plan must render, and the plan is read as a
graph: every effect step resolves to the segment concepts it draws over, minus
the concepts subtracted from them and restricted to the concepts selected.

    python -m app.planner.eval              # the live planner (PLANNER_MODEL)
    python -m app.planner.eval --mock       # the rule-based planner
    python -m app.planner.eval --out results.json --repeat 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass, field

from ..config import settings
from ..schemas import Plan
from . import llama, mock

OBSCURING = {"blur", "pixelate", "blackout"}

FACE = ("face",)
PLATE = ("plate",)
LAPTOP = ("laptop", "screen", "monitor")
SCREEN = ("screen", "monitor", "display", "laptop")
BADGE = ("badge",)
LOGO = ("logo",)
PHONE = ("phone",)


@dataclass
class Case:
    id: str
    instruction: str
    # target name -> (accepted words in the segment concept, required effect or None for any obscuring one)
    targets: dict[str, tuple[tuple[str, ...], str | None]]
    # target name -> words, one of which must be in a concept subtracted from every effect on that target
    exceptions: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # targets that must not be filtered by the person exception (rule 2 of the prompt)
    unfiltered: tuple[str, ...] = ()
    # target name -> words, one of which must be in a concept the target is restricted to
    selections: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # a follow-up edit on the rule-based plan for this instruction
    base: str | None = None


CASES = [
    Case("faces", "Blur every face.", {"face": (FACE, "blur")}),
    Case("plates", "Pixelate all license plates.", {"plate": (PLATE, "pixelate")}),
    Case("laptop", "Black out the laptop screen.", {"laptop": (LAPTOP, "blackout")}),
    Case("pink-coat", "Blur every face except the woman in the pink coat.", {"face": (FACE, "blur")}, {"face": ("pink",)}),
    Case("everyone", "Blur everyone except the man in the yellow hat.", {"face": (FACE, "blur")}, {"face": ("yellow",)}),
    Case("but-not", "Blur faces but not the child holding the red balloon.", {"face": (FACE, "blur")}, {"face": ("balloon",)}),
    Case(
        "bicycle", "Blur every license plate and every face except the woman on the bicycle.",
        {"plate": (PLATE, "blur"), "face": (FACE, "blur")}, {"face": ("bicycle", "bike")}, unfiltered=("plate",),
    ),
    Case("two-effects", "Pixelate all faces and black out license plates.", {"face": (FACE, "pixelate"), "plate": (PLATE, "blackout")}),
    Case("screens-badges", "Black out all screens and pixelate the badges.", {"screen": (SCREEN, "blackout"), "badge": (BADGE, "pixelate")}),
    Case(
        "speaker", "Blur every face and name badge except the speaker in the green jacket, and black out the laptop screen.",
        {"face": (FACE, "blur"), "badge": (BADGE, "blur"), "laptop": (LAPTOP, "blackout")}, {"face": ("green",)},
    ),
    Case("logo", "Hide the bank logo.", {"logo": (LOGO, None)}),
    Case("red-car", "Blur only the license plate on the red car.", {"plate": (PLATE, "blur")}, selections={"plate": ("red", "car")}),
    Case(
        "follow-logo", "Also hide the bank logo.", {"face": (FACE, "blur"), "logo": (LOGO, None)}, {"face": ("pink",)},
        base="Blur every face except the woman in the pink coat.",
    ),
    Case(
        "follow-phone", "Also black out the phone.", {"laptop": (LAPTOP, "blackout"), "phone": (PHONE, "blackout")},
        base="Black out the laptop screen.",
    ),
]  # fmt: skip


@dataclass
class Rendered:
    """What one effect step draws over."""

    concept: str
    effect: str
    minus: set[str]
    only: set[str]


def rendered(plan: Plan) -> list[Rendered]:
    """Resolve every effect step to its segment concepts, with what was subtracted and selected."""
    steps = {s.id: s for s in plan.steps}

    def concepts(sid: str) -> set[str]:
        s = steps[sid]
        if s.op == "segment":
            return {(s.concept or "").lower()}
        if s.op == "union":
            return set().union(*(concepts(i) for i in s.inputs))
        return concepts(s.source)  # select/subtract keep the source's instances

    def resolve(sid: str) -> dict[str, tuple[set[str], set[str]]]:
        s = steps[sid]
        if s.op == "segment":
            return {(s.concept or "").lower(): (set(), set())}
        if s.op == "union":
            out: dict[str, tuple[set[str], set[str]]] = {}
            for i in s.inputs:
                for c, (minus, only) in resolve(i).items():
                    m, o = out.setdefault(c, (set(), set()))
                    m |= minus
                    o |= only
            return out
        out = {c: (set(m), set(o)) for c, (m, o) in resolve(s.source).items()}
        for m, o in out.values():
            (m if s.op == "subtract" else o).update(concepts(s.reference) | _reference_words(steps, s.reference))
        return out

    return [
        Rendered(c, s.effect or "blur", m, o)
        for s in plan.steps
        if s.op == "effect"
        for c, (m, o) in resolve(s.source).items()
    ]


def _reference_words(steps: dict, sid: str) -> set[str]:
    """Every concept anywhere under a reference (a selected person carries their clothing)."""
    s = steps[sid]
    if s.op == "segment":
        return {(s.concept or "").lower()}
    return set().union(*(_reference_words(steps, r) for r in s.refs()))


def _hits(text: str, words: tuple[str, ...]) -> bool:
    return any(w in text for w in words)


def check(case: Case, plan: Plan) -> list[str]:
    """Problems with `plan` for `case`; empty when it's right."""
    effects = rendered(plan)
    problems = []
    for name, (words, effect) in case.targets.items():
        on = [r for r in effects if _hits(r.concept, words)]
        if not on:
            problems.append(f"{name}: not redacted")
            continue
        if effect and not any(r.effect == effect for r in on):
            problems.append(f"{name}: {'/'.join(sorted({r.effect for r in on}))} instead of {effect}")
        elif not effect and not any(r.effect in OBSCURING for r in on):
            problems.append(f"{name}: only highlighted")
        if name in case.exceptions and not all(any(_hits(m, case.exceptions[name]) for m in r.minus) for r in on):
            problems.append(f"{name}: exception ({case.exceptions[name][0]}) not subtracted")
        if name in case.unfiltered and any(r.minus for r in on):
            problems.append(f"{name}: person exception wrongly applied")
        if name in case.selections and not all(any(_hits(o, case.selections[name]) for o in r.only) for r in on):
            problems.append(f"{name}: not restricted to the {case.selections[name][-1]}")
    wanted = [w for words, _ in case.targets.values() for w in words]
    for r in effects:
        if not _hits(r.concept, tuple(wanted)) and r.effect in OBSCURING:
            problems.append(f"invented: {r.effect} on {r.concept!r}")
    return problems


@dataclass
class Result:
    case: str
    instruction: str
    valid: bool
    correct: bool
    problems: list[str]
    seconds: float
    plan: list[str]


def run_case(case: Case, live: bool) -> Result:
    base = mock.plan(case.base) if case.base else None
    t0 = time.time()
    try:
        plan = llama.plan(case.instruction, base) if live else mock.plan(case.instruction, base)
    except Exception as exc:  # invalid twice, refused, or unreachable
        return Result(case.id, case.instruction, False, False, [f"no valid plan: {str(exc)[:160]}"], time.time() - t0, [])
    problems = check(case, plan)
    return Result(case.id, case.instruction, True, not problems, problems, round(time.time() - t0, 2), [s.label for s in plan.steps])


def main() -> None:
    ap = argparse.ArgumentParser(description="Score the planner on instructions with known correct plans.")
    ap.add_argument("--mock", action="store_true", help="evaluate the rule-based planner instead of the live one")
    ap.add_argument("--repeat", type=int, default=1, help="run every case N times (live models aren't deterministic)")
    ap.add_argument("--out", help="write the results as JSON")
    args = ap.parse_args()
    live = not args.mock
    if live and not settings.planner_live:
        raise SystemExit("no live planner configured (META_MODEL_API_KEY / PLANNER_MODEL); pass --mock for the rule-based one")
    name = settings.planner_model if live else "rule-based planner"
    results = [run_case(c, live) for _ in range(args.repeat) for c in CASES]
    for r in results:
        print(f"{'PASS' if r.correct else 'FAIL'}  {r.case:<15} {r.seconds:5.1f}s  {'; '.join(r.problems)}")
    n = len(results)
    valid, correct = sum(r.valid for r in results), sum(r.correct for r in results)
    secs = [r.seconds for r in results if r.valid]
    print(f"\n{name}: {valid}/{n} valid plans, {correct}/{n} correct" + (f", median {statistics.median(secs):.1f}s per plan" if secs else ""))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"planner": name, "valid": valid, "correct": correct, "runs": n, "results": [asdict(r) for r in results]}, f, indent=2)


if __name__ == "__main__":
    main()
