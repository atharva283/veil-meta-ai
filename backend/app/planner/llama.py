"""Llama 3.1 planner over an OpenAI-compatible chat completions endpoint.

Works with the Meta Model API's compatibility endpoint and with Fireworks or
Baseten deployments of Llama 3.1; only META_MODEL_API_BASE / PLANNER_MODEL
change. The model must answer with a single `submit_plan` tool call, which is
validated against the Plan schema and repaired once if it fails.
"""

from __future__ import annotations

import json
import re

import httpx
from pydantic import ValidationError

from ..config import settings
from ..schemas import Plan

SYSTEM_PROMPT = """You are the planner inside Veil, a video privacy redaction tool.
Convert the user's instruction into a plan of tool calls that a segmentation
model (SAM 3.1, which takes short noun-phrase prompts like "face",
"license plate", "red jacket") and a renderer will execute.

Operations (each step has a unique id "s1", "s2", ... and may only reference
earlier steps):
- segment:  {"op":"segment","concept":"<short noun phrase>"}  -> tracked instances
- union:    {"op":"union","inputs":["sA","sB"]}
- select:   {"op":"select","source":"sA","reference":"sB"} -> instances of sA that overlap sB
- subtract: {"op":"subtract","source":"sA","reference":"sB"} -> instances of sA that do NOT overlap sB
- effect:   {"op":"effect","source":"sA","effect":"blur|pixelate|blackout|highlight"}

Rules:
1. SAM prompts must be short, concrete noun phrases. Decompose descriptions:
   "the woman in the red jacket" = select(segment("person"), segment("red jacket")).
2. Exceptions ("except", "but not") become subtract steps, applied only to the
   targets they can affect: a person exception filters faces, bodies, badges and
   the phones they hold, never license plates or logos.
3. "Everyone", "people", "strangers" mean faces unless whole bodies are requested.
4. Reuse a segment step instead of segmenting the same concept twice.
5. Add a short human-readable "note" to every non-segment step.
6. If something can't be expressed, still return the closest valid plan and
   explain the gap in "warnings". Never invent concepts the user didn't ask for.
"""

EXAMPLE_USER = "Blur every face and license plate except the woman in the red jacket, and black out phone screens."
EXAMPLE_PLAN = {
    "summary": "Blur faces and license plates, except the woman in the red jacket; black out phone screens",
    "steps": [
        {"id": "s1", "op": "segment", "concept": "face"},
        {"id": "s2", "op": "segment", "concept": "license plate"},
        {"id": "s3", "op": "segment", "concept": "person"},
        {"id": "s4", "op": "segment", "concept": "red jacket"},
        {"id": "s5", "op": "select", "source": "s3", "reference": "s4", "note": "the woman in the red jacket"},
        {"id": "s6", "op": "subtract", "source": "s1", "reference": "s5", "note": "faces except the woman in the red jacket"},
        {"id": "s7", "op": "effect", "source": "s6", "effect": "blur", "note": "faces"},
        {"id": "s8", "op": "effect", "source": "s2", "effect": "blur", "note": "license plates"},
        {"id": "s9", "op": "segment", "concept": "phone screen"},
        {"id": "s10", "op": "effect", "source": "s9", "effect": "blackout", "note": "phone screens"},
    ],
    "warnings": [],
}

STEP_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "op": {"type": "string", "enum": ["segment", "union", "select", "subtract", "effect"]},
        "concept": {"type": "string"},
        "source": {"type": "string"},
        "reference": {"type": "string"},
        "inputs": {"type": "array", "items": {"type": "string"}},
        "effect": {"type": "string", "enum": ["blur", "pixelate", "blackout", "highlight"]},
        "note": {"type": "string"},
    },
    "required": ["id", "op"],
}
TOOL = {
    "type": "function",
    "function": {
        "name": "submit_plan",
        "description": "Submit the redaction plan for execution.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "steps": {"type": "array", "items": STEP_SCHEMA},
                "warnings": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["summary", "steps"],
        },
    },
}


class PlannerError(RuntimeError):
    pass


def _messages(instruction: str, base: Plan | None) -> list[dict]:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": EXAMPLE_USER},
        {"role": "assistant", "content": json.dumps(EXAMPLE_PLAN)},
    ]
    if base:
        current = {"summary": base.summary, "steps": [s.model_dump(exclude_none=True) for s in base.steps]}
        msgs.append({
            "role": "user",
            "content": "This is a follow-up edit. The current plan is:\n"
            f"{json.dumps(current)}\n"
            "Return the FULL updated plan: keep existing steps and ids unless the user "
            f"changes them, and append new steps.\nFollow-up instruction: {instruction}",
        })
    else:
        msgs.append({"role": "user", "content": instruction})
    return msgs


def _extract_args(data: dict) -> dict:
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        if fn.get("name") == "submit_plan":
            args = fn.get("arguments")
            return _decode_nested(json.loads(args) if isinstance(args, str) else args)
    # Some providers ignore tool_choice and answer in plain JSON.
    content = message.get("content") or ""
    match = re.search(r"\{.*\}", content, re.S)
    if not match:
        raise PlannerError("Llama returned neither a submit_plan tool call nor JSON.")
    return _decode_nested(json.loads(match.group(0)))


def _decode_nested(args: dict) -> dict:
    """Llama 3.1 8B via Ollama sometimes sends the steps array as a JSON-encoded string."""
    for key in ("steps", "warnings"):
        if isinstance(args.get(key), str):
            try:
                args[key] = json.loads(args[key])
            except ValueError:
                pass  # left as is; validation reports it and Llama gets a retry
    return args


def plan(instruction: str, base: Plan | None = None) -> Plan:
    messages = _messages(instruction, base)
    last_error = ""
    with httpx.Client(timeout=settings.request_timeout_s) as client:
        for _attempt in range(2):
            resp = client.post(
                f"{settings.meta_api_base}/chat/completions",
                headers={"Authorization": f"Bearer {settings.meta_api_key}"},
                json={
                    "model": settings.planner_model,
                    "messages": messages,
                    "tools": [TOOL],
                    "tool_choice": {"type": "function", "function": {"name": "submit_plan"}},
                    "temperature": 0,
                },
            )
            if resp.status_code >= 400:
                raise PlannerError(f"Planner API returned {resp.status_code}: {resp.text[:300]}")
            args: dict | None = None
            try:
                args = _extract_args(resp.json())
                args.pop("source", None)
                return Plan(**args, source=settings.planner_model)
            except (ValidationError, ValueError, TypeError) as exc:
                last_error = str(exc)
                messages = [
                    *messages,
                    {"role": "assistant", "content": json.dumps(args) if args else ""},
                    {"role": "user", "content": f"That plan is invalid: {last_error}. Return a corrected plan."},
                ]
    raise PlannerError(f"Llama produced an invalid plan twice: {last_error}")
