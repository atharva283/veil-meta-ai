import { TriangleAlert } from "lucide-react";
import { useMemo, useState } from "react";
import { EFFECT_LABEL } from "../lib/format";
import type { Plan, PlanStep } from "../lib/types";

type Props = { plan: Plan | null; previousSteps: number; running: boolean };

const OP_GLYPH: Record<PlanStep["op"], string> = {
  segment: "◎",
  union: "∪",
  select: "∩",
  subtract: "−",
  effect: "▮",
};

/** Human-readable call using step numbers and concept names instead of ids. */
function display(step: PlanStep, index: Map<string, number>, byId: Map<string, PlanStep>): string {
  const ref = (id: string | null) => {
    if (!id) return "?";
    const s = byId.get(id);
    return s?.op === "segment" ? s.concept ?? id : `#${index.get(id)}`;
  };
  switch (step.op) {
    case "segment":
      return `Segment("${step.concept}")`;
    case "union":
      return `Union(${step.inputs.map(ref).join(", ")})`;
    case "select":
      return `Select(${ref(step.source)} ∩ ${ref(step.reference)})`;
    case "subtract":
      return `Subtract(${ref(step.source)} − ${ref(step.reference)})`;
    case "effect":
      return `${EFFECT_LABEL[step.effect ?? "blur"].replace(" ", "")}(${ref(step.source)})`;
  }
}

export function PlanView({ plan, previousSteps, running }: Props) {
  const [raw, setRaw] = useState(false);
  const maps = useMemo(() => {
    const index = new Map<string, number>();
    const byId = new Map<string, PlanStep>();
    plan?.steps.forEach((s, i) => {
      index.set(s.id, i + 1);
      byId.set(s.id, s);
    });
    return { index, byId };
  }, [plan]);

  if (!plan) {
    return (
      <section className="panel plan" aria-labelledby="plan-h">
        <h2 id="plan-h">Plan</h2>
        <p className="muted">{running ? "Planning…" : "Llama 3.1 turns your instruction into segmentation and mask steps. They show up here."}</p>
      </section>
    );
  }

  const source = plan.source.includes("mock") ? "rule-based planner (offline mock)" : plan.source;
  const toolCall = {
    name: "submit_plan",
    arguments: { summary: plan.summary, steps: plan.steps.map(({ label: _label, ...s }) => s), warnings: plan.warnings },
  };

  return (
    <section className="panel plan" aria-labelledby="plan-h">
      <div className="panel-head">
        <h2 id="plan-h">Plan</h2>
        <button className="link" onClick={() => setRaw((r) => !r)} aria-pressed={raw}>
          {raw ? "Show steps" : "Show tool call"}
        </button>
      </div>
      <p className="plan-summary">{plan.summary}</p>
      <p className="faint small">Planned by {source}</p>
      {raw ? (
        <pre className="tool-call">{JSON.stringify(toolCall, null, 2)}</pre>
      ) : (
        <ol className="steps">
          {plan.steps.map((s, i) => (
            <li
              key={s.id}
              className={`step op-${s.op}${i >= previousSteps && previousSteps > 0 ? " is-new" : ""}`}
              style={{ ["--i" as string]: i - previousSteps }}
            >
              <span className="step-n">{i + 1}</span>
              <span className="step-glyph" aria-hidden>
                {OP_GLYPH[s.op]}
              </span>
              <span className="step-body">
                <code>{display(s, maps.index, maps.byId)}</code>
                {s.note && s.op !== "segment" && <span className="step-note">{s.note}</span>}
              </span>
              {i >= previousSteps && previousSteps > 0 && <span className="new-tag">new</span>}
            </li>
          ))}
        </ol>
      )}
      {plan.warnings.length > 0 && (
        <ul className="warnings">
          {plan.warnings.map((w) => (
            <li key={w}>
              <TriangleAlert size={14} aria-hidden /> {w}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
