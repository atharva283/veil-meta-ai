import type { Health, VideoInfo } from "../lib/types";

type Props = { health: Health | null; video: VideoInfo | null; onChangeClip: () => void; busy: boolean };

const MODELS = [
  { key: "planner", label: "Llama 3.1", role: "Plans the redaction" },
  { key: "segmenter", label: "SAM 3.1", role: "Segments and tracks objects" },
  { key: "auditor", label: "Muse Spark", role: "Audits the output" },
] as const;

export function TopBar({ health, video, onChangeClip, busy }: Props) {
  return (
    <header className="topbar">
      <div className="brand">
        <span className="wordmark">Veil</span>
        <span className="tagline">Video redaction from plain English</span>
      </div>
      {video && (
        <div className="clip-name">
          <span title={video.title}>{video.title}</span>
          <button className="link" onClick={onChangeClip} disabled={busy}>
            Change clip
          </button>
        </div>
      )}
      <ul className="models" aria-label="Model connections">
        {MODELS.map((m) => {
          const mode = health?.modes[m.key];
          // Replay mode: no sampled-frame auditor was used; SAM 3.1 re-checked the output instead.
          const label = mode?.recorded && m.key === "auditor" && mode.name.startsWith("none") ? "SAM 3.1 re-check" : m.label;
          const state = !health
            ? "Checking"
            : mode?.recorded
              ? `Recorded run: ${mode.name}`
              : mode?.live
                ? `Live: ${mode.name}`
                : "Offline mock";
          const tag = !health ? "…" : mode?.recorded ? "recorded" : mode?.live ? "live" : "mock";
          return (
            <li key={m.key} className={mode?.live ? "is-live" : "is-mock"} title={`${m.role}. ${state}.`}>
              <span className="dot" aria-hidden />
              {label}
              <span className="faint">{tag}</span>
            </li>
          );
        })}
      </ul>
    </header>
  );
}
