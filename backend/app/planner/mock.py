"""Rule-based planner used when no Meta Model API key is configured.

It understands the instruction shapes the demo relies on: effect verbs,
target lists, "except <descriptor>" exceptions, "<target> on the <thing>"
constraints, and follow-up edits that extend a previous plan. Anything it
can't resolve becomes a plan warning instead of a silent guess.
"""

from __future__ import annotations

import re

from ..schemas import Plan, Step

# Longest phrases first so "phone screens" wins over "screens".
CONCEPTS: list[tuple[str, str]] = [
    ("license plate", r"(?:license|licence|number|registration)\s+plates?|plates?"),
    ("phone screen", r"(?:phone|smartphone|mobile)\s+screens?|smart\s*phones?|phones?"),
    ("laptop screen", r"laptop\s+screens?|laptops?"),
    ("screen", r"(?:computer\s+|tv\s+)?screens?|monitors?|displays?"),
    ("face", r"faces?|heads?|everyone|everybody|strangers?|bystanders?|people|crowds?"),
    ("person", r"persons?|pedestrians?|bodies|whole\s+people"),
    ("child", r"kids?|children|child"),
    ("logo", r"logos?|brand\s+marks?"),
    ("name badge", r"(?:name\s+)?badges?|lanyards?"),
    ("tattoo", r"tattoos?"),
    ("street sign", r"(?:street\s+)?signs?"),
    ("document", r"documents?|papers?|letters?"),
    ("truck", r"trucks?|vans?"),
    ("car", r"cars?|vehicles?"),
    ("text", r"text|writing"),
]
CONCEPT_RE = re.compile(r"\b(?:" + "|".join(f"(?P<c{i}>{p})" for i, (_, p) in enumerate(CONCEPTS)) + r")\b")

EFFECTS: list[tuple[str, str]] = [
    ("blackout", r"black(?:\s|-)?out|black\s+bar|cover|redact|censor|block\s+out|mask\s+out|obscure"),
    ("pixelate", r"pixel(?:ate|ize|ise)|mosaic"),
    ("highlight", r"highlight|spotlight|outline"),
    ("blur", r"blur|hide|anonymi[sz]e|remove|scrub|obfuscate"),
]
EFFECT_RE = re.compile(r"\b(?:" + "|".join(f"(?P<e{i}>{p})" for i, (_, p) in enumerate(EFFECTS)) + r")\b")

EXCEPT_RE = re.compile(
    r"\b(?:except(?:\s+for)?|but\s+not|other\s+than|excluding|apart\s+from|aside\s+from|save\s+for|besides|leaving\s+out)\b"
)
RELATION_RE = re.compile(r"^\s*(?:on|in|of|from|inside|attached\s+to)\s+(?:the|a|an|that|this|any|every)?\s*$")
QUALIFIER_RE = re.compile(r"(?:showing|with|displaying|containing|that\s+show)\s+(?:any\s+)?$")

COLORS = r"red|orange|yellow|green|blue|purple|pink|black|white|gr[ae]y|brown|beige|navy|teal|maroon"
CLOTHING = r"jacket|coat|shirt|t-shirt|hoodie|sweater|dress|hat|cap|scarf|vest|top|jersey|uniform|backpack"
SUBJECTS = r"woman|man|person|guy|girl|boy|lady|gentleman|kid|child|presenter|speaker|cyclist|driver|officer"
ATTRIBUTE_RE = re.compile(rf"\b(?:(?P<color>{COLORS})\s+)?(?P<item>{CLOTHING})\b")
COLOR_ONLY_RE = re.compile(rf"\b(?:in|wearing)\s+(?P<color>{COLORS})\b")
SUBJECT_RE = re.compile(rf"\b(?P<subject>{SUBJECTS})\b")
SELF_RE = re.compile(r"\b(?:me|myself|my\s+face|i)\b")
VEHICLE_RE = re.compile(rf"\b(?P<color>{COLORS})\s+(?P<thing>car|truck|van|bike|bicycle)\b")


def _concept_of(m: re.Match) -> str:
    for i, (canonical, _) in enumerate(CONCEPTS):
        if m.group(f"c{i}"):
            return canonical
    raise AssertionError("unreachable")


def _effect_of(m: re.Match) -> str:
    for i, (canonical, _) in enumerate(EFFECTS):
        if m.group(f"e{i}"):
            return canonical
    raise AssertionError("unreachable")


class _Builder:
    def __init__(self, base: Plan | None):
        self.steps: list[Step] = list(base.steps) if base else []
        self.segments: dict[str, str] = {s.concept: s.id for s in self.steps if s.op == "segment" and s.concept}
        self.warnings: list[str] = []
        self.summary_parts: list[str] = []

    def _add(self, **kw) -> str:
        sid = f"s{len(self.steps) + 1}"
        self.steps.append(Step(id=sid, **kw))
        return sid

    def segment(self, concept: str) -> str:
        if concept not in self.segments:
            self.segments[concept] = self._add(op="segment", concept=concept)
        return self.segments[concept]

    def select(self, source: str, reference: str, note: str) -> str:
        return self._add(op="select", source=source, reference=reference, note=note)

    def subtract(self, source: str, reference: str, note: str) -> str:
        return self._add(op="subtract", source=source, reference=reference, note=note)

    def effect(self, source: str, effect: str, note: str) -> str:
        return self._add(op="effect", source=source, effect=effect, note=note)


def _resolve_exception(b: _Builder, text: str) -> tuple[str | None, str, str]:
    """Turn an exception descriptor into (step id of kept instances, description, kind).

    kind is "person", "vehicle" or "concept"; it decides which targets the
    exception can apply to (a person exception never filters license plates).
    """
    text = text.strip(" ,.;")
    if not text:
        return None, "", "concept"
    vehicle = VEHICLE_RE.search(text)
    if vehicle:
        phrase = f"{vehicle['color']} {vehicle['thing']}"
        return b.segment(phrase), f"the {phrase}", "vehicle"
    attr = ATTRIBUTE_RE.search(text)
    subject = SUBJECT_RE.search(text)
    color_only = COLOR_ONLY_RE.search(text)
    if attr and (attr["color"] or subject):
        clothing = f"{attr['color']} {attr['item']}" if attr["color"] else attr["item"]
        who = subject["subject"] if subject else "person"
        person = b.segment("person")
        wearing = b.segment(clothing)
        desc = f"the {who} in the {clothing}"
        return b.select(person, wearing, desc), desc, "person"
    if color_only:
        desc = f"the person in {color_only['color']}"
        b.warnings.append(f'Matched "{text}" by color only; SAM works best with a named garment, e.g. "red jacket".')
        person = b.segment("person")
        return b.select(person, b.segment(f"{color_only['color']} clothing"), desc), desc, "person"
    if SELF_RE.search(text):
        b.warnings.append(
            "Veil can't tell which person is \"you\". Describe what you are wearing, e.g. \"except me (red jacket)\"."
        )
        return None, "", "concept"
    concept = CONCEPT_RE.search(text)
    if concept:
        c = _concept_of(concept)
        return b.segment(c), f"the {c}", "concept"
    b.warnings.append(f"Couldn't resolve the exception \"{text}\"; nothing was excluded.")
    return None, "", "concept"


# Which targets an exception of each kind can meaningfully filter.
PERSONAL = {"face", "person", "child", "name badge", "tattoo", "phone screen", "laptop screen", "document"}
VEHICULAR = {"license plate", "car", "truck", "logo"}


def _applies(kind: str, concept: str) -> bool:
    return {"person": concept in PERSONAL, "vehicle": concept in VEHICULAR}.get(kind, True)


def _plural(name: str) -> str:
    head, sep, tail = name.partition(" on the ")
    head = head if head.endswith("s") else f"{head}s"
    return f"{head}{sep}{tail}"


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def _targets(text: str, b: _Builder) -> list[tuple[str, str | None]]:
    """Concepts named in a clause, each with an optional 'on the X' constraint."""
    found: list[tuple[str, str | None]] = []
    last_end = 0
    for m in CONCEPT_RE.finditer(text):
        between = text[last_end : m.start()]
        concept = _concept_of(m)
        if found and RELATION_RE.match(between) and found[-1][1] is None:
            found[-1] = (found[-1][0], concept)
        elif QUALIFIER_RE.search(text[: m.start()]):
            b.warnings.append(
                f'Treating "{text[: m.end()].strip()}" as all {found[-1][0] if found else concept}s; '
                "content-based filtering needs the live auditor."
            )
        elif concept not in [c for c, _ in found]:
            found.append((concept, None))
        last_end = m.end()
    return found


def plan(instruction: str, base: Plan | None = None) -> Plan:
    text = instruction.lower().replace("’", "'")
    b = _Builder(base)
    verbs = list(EFFECT_RE.finditer(text))
    if not verbs:
        b.warnings.append("No action word found (blur, pixelate, black out); defaulting to blur.")
        clauses = [("blur", text)]
    else:
        clauses = [
            (_effect_of(v), text[v.end() : verbs[i + 1].start() if i + 1 < len(verbs) else len(text)])
            for i, v in enumerate(verbs)
        ]

    for effect, body in clauses:
        exc = EXCEPT_RE.search(body)
        target_text = body[: exc.start()] if exc else body
        targets = [(concept, on, b.segment(concept)) for concept, on in _targets(target_text, b)]
        if not targets:
            continue
        keep_id, keep_desc, keep_kind = _resolve_exception(b, body[exc.end() :]) if exc else (None, "", "")
        names = []
        for concept, on, sid in targets:
            name = _plural(f"{concept} on the {on}" if on else concept)
            if on:
                sid = b.select(sid, b.segment(on), name)
            if keep_id and _applies(keep_kind, concept):
                sid = b.subtract(sid, keep_id, f"{name} except {keep_desc}")
            b.effect(sid, effect, name)
            names.append(name)
        verb = {"blackout": "Black out", "highlight": "Highlight"}.get(effect, effect.capitalize())
        b.summary_parts.append(f"{verb} {_join(names)}" + (f", except {keep_desc}" if keep_id else ""))

    if not any(s.op == "effect" for s in b.steps):
        raise ValueError(
            "The instruction doesn't name anything to redact. Try e.g. "
            '"Blur every face and license plate".'
        )
    parts = [p if i == 0 else p[0].lower() + p[1:] for i, p in enumerate(b.summary_parts)]
    summary = "; ".join(parts) or "No new edits"
    if base:
        summary = f"{base.summary}; then {summary[0].lower()}{summary[1:]}" if base.summary else summary
    return Plan(summary=summary, steps=b.steps, warnings=b.warnings, source="mock planner")
