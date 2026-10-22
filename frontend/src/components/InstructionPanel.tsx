import { LoaderCircle } from "lucide-react";
import { useEffect, useRef } from "react";

type Props = {
  value: string;
  onChange: (v: string) => void;
  suggestions: string[];
  canFollowUp: boolean;
  followUp: boolean;
  onFollowUpChange: (v: boolean) => void;
  ready: boolean;
  busy: boolean;
  error: string | null;
  onRun: () => void;
};

export function InstructionPanel(p: Props) {
  const box = useRef<HTMLTextAreaElement>(null);
  const editing = p.canFollowUp && p.followUp;

  useEffect(() => {
    if (editing) box.current?.focus();
  }, [editing]);

  return (
    <section className="panel instruction" aria-labelledby="instruction-h">
      <h2 id="instruction-h">{editing ? "Describe a follow-up edit" : "What should be hidden?"}</h2>
      <textarea
        ref={box}
        value={p.value}
        rows={4}
        placeholder={
          editing
            ? "Also hide the logo on the truck."
            : "Blur every face and license plate except the woman in the red jacket, and black out phone screens."
        }
        onChange={(e) => p.onChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            if (p.ready && !p.busy && p.value.trim().length >= 3) p.onRun();
          }
        }}
        aria-describedby="instruction-hint"
      />
      {p.suggestions.length > 0 && (
        <div className="suggestions" aria-label="Suggested instructions">
          {p.suggestions.map((s) => (
            <button key={s} className="chip" onClick={() => p.onChange(s)}>
              {s}
            </button>
          ))}
        </div>
      )}
      {p.canFollowUp && (
        <label className="check">
          <input type="checkbox" checked={p.followUp} onChange={(e) => p.onFollowUpChange(e.target.checked)} />
          Build on the last result
        </label>
      )}
      <div className="run-row">
        <button className="btn btn-primary" onClick={p.onRun} disabled={!p.ready || p.busy || p.value.trim().length < 3}>
          {p.busy && <LoaderCircle className="spin" size={16} aria-hidden />}
          {editing ? "Apply edit" : "Redact video"}
        </button>
        <span id="instruction-hint" className="faint">
          {p.ready ? "Ctrl + Enter" : "Choose a clip first"}
        </span>
      </div>
      {p.error && (
        <p className="error" role="alert">
          {p.error}
        </p>
      )}
    </section>
  );
}
