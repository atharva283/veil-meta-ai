import { CircleAlert, Download, Eye, ShieldCheck, Wrench } from "lucide-react";
import { useMemo, useState } from "react";
import { conceptClass, EFFECT_LABEL, STATUS_LABEL, timecode } from "../lib/format";
import type { AuditStatus, Job } from "../lib/types";

type Filter = "all" | AuditStatus;

const STATUS_ICON: Record<AuditStatus, React.ReactNode> = {
  verified: <ShieldCheck size={14} aria-hidden />,
  patched: <Wrench size={14} aria-hidden />,
  kept_visible: <Eye size={14} aria-hidden />,
  flagged: <CircleAlert size={14} aria-hidden />,
};

type Props = { job: Job; time: number; onSeek: (t: number) => void };

export function AuditLog({ job, time, onSeek }: Props) {
  const [filter, setFilter] = useState<Filter>("all");
  const entries = job.audit.entries ?? [];
  const counts = job.audit.counts ?? { verified: 0, patched: 0, kept_visible: 0, flagged: 0 };
  const shown = useMemo(() => (filter === "all" ? entries : entries.filter((e) => e.status === filter)), [entries, filter]);

  const summary = [
    `${counts.verified + counts.patched} redacted`,
    counts.patched ? `${counts.patched} of them caught by the audit` : null,
    counts.kept_visible ? `${counts.kept_visible} kept visible on purpose` : null,
    counts.flagged ? `${counts.flagged} need manual review` : null,
  ]
    .filter(Boolean)
    .join(", ");

  const filters: { key: Filter; label: string; n: number }[] = [
    { key: "all", label: "All", n: entries.length },
    { key: "patched", label: "Patched", n: counts.patched },
    { key: "kept_visible", label: "Kept visible", n: counts.kept_visible },
    { key: "flagged", label: "Needs review", n: counts.flagged },
  ];

  return (
    <section className="audit" aria-labelledby="audit-h">
      <div className="audit-head">
        <div>
          <h2 id="audit-h">Privacy audit log</h2>
          <p className="muted">
            {summary}. Checked {job.audit.audited_frames ?? 0} sampled frames
            {job.audit.rechecked_frames
              ? ` and re-segmented ${job.audit.rechecked_frames} of ${job.audit.frame_count ?? job.audit.rechecked_frames} output frames (${Math.round((100 * job.audit.rechecked_frames) / (job.audit.frame_count || job.audit.rechecked_frames))}%)`
              : ""}{" "}
            in {job.audit.rounds ?? 0} round{job.audit.rounds === 1 ? "" : "s"}.
          </p>
        </div>
        <div className="audit-actions">
          <div className="segmented" role="group" aria-label="Filter by status">
            {filters.map((f) => (
              <button key={f.key} aria-pressed={filter === f.key} onClick={() => setFilter(f.key)} disabled={f.key !== "all" && f.n === 0}>
                {f.label} <span className="faint">{f.n}</span>
              </button>
            ))}
          </div>
          {job.outputs.audit_url && (
            <a className="btn btn-quiet" href={job.outputs.audit_url} download={`veil-audit-${job.id}.json`}>
              <Download size={15} aria-hidden /> Download JSON
            </a>
          )}
        </div>
      </div>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th scope="col">Object</th>
              <th scope="col">Action</th>
              <th scope="col">On screen</th>
              <th scope="col">Confidence</th>
              <th scope="col">Status</th>
              <th scope="col">Details</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((e) => {
              const live = time >= e.start_s && time < e.end_s;
              return (
                <tr
                  key={e.track_id}
                  className={live ? "is-live" : undefined}
                  onClick={() => onSeek(e.start_s)}
                  tabIndex={0}
                  onKeyDown={(k) => k.key === "Enter" && onSeek(e.start_s)}
                  aria-label={`${e.concept} ${e.track_id}, ${STATUS_LABEL[e.status]}. Press Enter to jump to ${timecode(e.start_s)}.`}
                >
                  <td>
                    <span className={`swatch c-${conceptClass(e.concept)}`} aria-hidden />
                    {e.concept} <code className="faint">{e.track_id}</code>
                  </td>
                  <td>{e.effect ? EFFECT_LABEL[e.effect] : <span className="faint">None</span>}</td>
                  <td>
                    <code>
                      {timecode(e.start_s)}–{timecode(e.end_s)}
                    </code>
                  </td>
                  <td>
                    {e.status === "flagged" ? (
                      <span className="faint">n/a</span>
                    ) : (
                      <span className="confidence">
                        <span className="meter" aria-hidden>
                          <span style={{ width: `${Math.round(e.confidence * 100)}%` }} />
                        </span>
                        {Math.round(e.confidence * 100)}%
                      </span>
                    )}
                  </td>
                  <td>
                    <span className={`pill p-${e.status}`}>
                      {STATUS_ICON[e.status]} {STATUS_LABEL[e.status]}
                    </span>
                  </td>
                  <td className="note">{e.note}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {shown.length === 0 && <p className="muted empty-row">Nothing in this category.</p>}
      </div>
    </section>
  );
}
