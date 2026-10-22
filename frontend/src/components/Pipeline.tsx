import { Check, Circle, LoaderCircle, Minus, X } from "lucide-react";
import { seconds, timecode } from "../lib/format";
import type { Job, StageStatus } from "../lib/types";

const ICON: Record<StageStatus, React.ReactNode> = {
  pending: <Circle size={14} aria-hidden />,
  active: <LoaderCircle size={14} className="spin" aria-hidden />,
  done: <Check size={14} aria-hidden />,
  skipped: <Minus size={14} aria-hidden />,
  error: <X size={14} aria-hidden />,
};

const STATUS_TEXT: Record<StageStatus, string> = {
  pending: "Waiting",
  active: "Running",
  done: "Done",
  skipped: "Skipped",
  error: "Failed",
};

export function Pipeline({ job }: { job: Job | null }) {
  if (!job) return null;
  return (
    <section className="panel pipeline" aria-labelledby="pipeline-h">
      <div className="panel-head">
        <h2 id="pipeline-h">Run</h2>
        <span className="faint small">{job.status === "done" || job.status === "error" ? `Finished in ${seconds(job.elapsed_s)}` : `${seconds(job.elapsed_s)} elapsed`}</span>
      </div>
      <ol className="stages">
        {job.stages.map((s) => (
          <li key={s.key} className={`stage-row st-${s.status}`}>
            <span className="stage-icon" title={STATUS_TEXT[s.status]}>
              {ICON[s.status]}
            </span>
            <span className="stage-title">{s.title}</span>
            <span className="stage-detail">{s.status === "pending" ? s.blurb : s.detail || STATUS_TEXT[s.status]}</span>
            <span className="stage-time faint">{s.status === "done" ? seconds(s.duration_s) : ""}</span>
          </li>
        ))}
      </ol>
      {job.error && <p className="error" role="alert">{job.error}</p>}
      <details className="activity">
        <summary>Activity log ({job.events.length})</summary>
        <ol>
          {job.events.map((e, i) => (
            <li key={i} className={`ev-${e.level}`}>
              <code className="faint">{timecode(e.t)}</code> {e.message}
            </li>
          ))}
        </ol>
      </details>
    </section>
  );
}
