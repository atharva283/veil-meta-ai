"""Wire-level schemas shared by the API, the planner, and the frontend."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

Op = Literal["segment", "union", "select", "subtract", "effect"]
Effect = Literal["blur", "pixelate", "blackout", "highlight"]


class Step(BaseModel):
    """One tool call in an execution plan.

    segment  -- run SAM 3.1 for `concept`; yields a set of tracked instances
    union    -- merge the instance sets in `inputs`
    select   -- keep instances of `source` that overlap any instance of `reference`
    subtract -- drop instances of `source` that overlap any instance of `reference`
    effect   -- render `effect` over every instance of `source`
    """

    id: str
    op: Op
    concept: Optional[str] = None
    source: Optional[str] = None
    reference: Optional[str] = None
    inputs: list[str] = Field(default_factory=list)
    effect: Optional[Effect] = None
    note: Optional[str] = None

    def refs(self) -> list[str]:
        return [r for r in (self.source, self.reference, *self.inputs) if r]

    @property
    def label(self) -> str:
        if self.op == "segment":
            return f'Segment("{self.concept}")'
        if self.op == "union":
            return f"Union({', '.join(self.inputs)})"
        if self.op == "select":
            return f"Select({self.source} ∩ {self.reference})"
        if self.op == "subtract":
            return f"Subtract({self.source} − {self.reference})"
        return f"{(self.effect or 'blur').capitalize()}({self.source})"


class Plan(BaseModel):
    summary: str = ""
    steps: list[Step]
    warnings: list[str] = Field(default_factory=list)
    source: str = "mock"  # which planner produced it, shown in the UI

    @model_validator(mode="after")
    def _check_graph(self) -> "Plan":
        seen: set[str] = set()
        for step in self.steps:
            if step.id in seen:
                raise ValueError(f"duplicate step id {step.id!r}")
            for ref in step.refs():
                if ref not in seen:
                    raise ValueError(f"step {step.id!r} references {ref!r} before it is defined")
            if step.op == "segment" and not (step.concept or "").strip():
                raise ValueError(f"segment step {step.id!r} needs a concept")
            if step.op in ("select", "subtract") and not (step.source and step.reference):
                raise ValueError(f"{step.op} step {step.id!r} needs source and reference")
            if step.op == "union" and len(step.inputs) < 2:
                raise ValueError(f"union step {step.id!r} needs at least two inputs")
            if step.op == "effect" and not (step.source and step.effect):
                raise ValueError(f"effect step {step.id!r} needs source and effect")
            seen.add(step.id)
        if not any(s.op == "effect" for s in self.steps):
            raise ValueError("plan has no effect step, so nothing would be redacted")
        return self

    def as_view(self) -> list[dict]:
        return [{**s.model_dump(), "label": s.label} for s in self.steps]


class AuditEntry(BaseModel):
    track_id: str
    concept: str
    effect: Optional[str]
    status: Literal["verified", "patched", "kept_visible", "flagged"]
    start_s: float
    end_s: float
    frames: int
    confidence: float
    note: str = ""


class AuditFinding(BaseModel):
    frame_index: int
    concept: str
    box: tuple[float, float, float, float]  # pixel coords x0, y0, x1, y1
    reason: str = ""


class JobCreate(BaseModel):
    video_id: str
    instruction: str = Field(min_length=3, max_length=2000)
    parent_job_id: Optional[str] = None
