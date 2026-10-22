import { Pause, Play } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { timecode } from "../lib/format";
import type { Job, VideoInfo } from "../lib/types";
import { Timeline } from "./Timeline";

type View = "redacted" | "masks";

type Props = {
  video: VideoInfo;
  job: Job | null;
  time: number;
  onTime: (t: number) => void;
  registerSeek: (fn: (t: number) => void) => void;
};

const STAGE_COPY: Record<string, string> = {
  plan: "Planning with Llama 3.1",
  segment: "Segmenting and tracking with SAM 3.1",
  compose: "Applying selections and exceptions",
  render: "Rendering effects",
  audit: "Auditing output frames with Muse Spark",
  patch: "Re-segmenting leaks and re-rendering",
};

export function Stage({ video, job, time, onTime, registerSeek }: Props) {
  const master = useRef<HTMLVideoElement>(null);
  const redacted = useRef<HTMLVideoElement>(null);
  const masks = useRef<HTMLVideoElement>(null);
  const frame = useRef<HTMLDivElement>(null);
  const [playing, setPlaying] = useState(false);
  const [split, setSplit] = useState(50);
  const [view, setView] = useState<View>("redacted");
  const [dragging, setDragging] = useState(false);

  const done = job?.status === "done" && !!job.outputs.redacted_url;
  const running = job?.status === "queued" || job?.status === "running";
  const duration = video.duration_s;
  const activeStage = job?.stages.find((s) => s.status === "active");

  // Keep follower videos locked to the master clock; report time at ~30 fps.
  useEffect(() => {
    let raf = 0;
    let last = -1;
    const tick = () => {
      const m = master.current;
      if (m) {
        for (const f of [redacted.current, masks.current]) {
          if (!f || f.readyState < 1) continue;
          if (Math.abs(f.currentTime - m.currentTime) > 0.08) f.currentTime = m.currentTime;
          if (!m.paused && f.paused) void f.play().catch(() => {});
          if (m.paused && !f.paused) f.pause();
        }
        if (Math.abs(m.currentTime - last) > 0.03) {
          last = m.currentTime;
          onTime(last);
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [onTime]);

  const seek = useCallback((t: number) => {
    const clamped = Math.max(0, Math.min(t, duration - 0.001));
    for (const v of [master.current, redacted.current, masks.current]) if (v) v.currentTime = clamped;
    onTime(clamped);
  }, [duration, onTime]);

  useEffect(() => registerSeek(seek), [registerSeek, seek]);

  // A new clip or a new result starts paused at the beginning.
  const resultUrl = job?.outputs.redacted_url;
  useEffect(() => {
    master.current?.pause();
    setPlaying(false);
    seek(0);
  }, [video.id, resultUrl, seek]);

  const toggle = useCallback(() => {
    const m = master.current;
    if (!m) return;
    if (m.paused) void m.play();
    else m.pause();
  }, []);

  const splitFromEvent = (clientX: number) => {
    const rect = frame.current?.getBoundingClientRect();
    if (!rect) return;
    setSplit(Math.max(0, Math.min(100, ((clientX - rect.left) / rect.width) * 100)));
  };

  const onKey = (e: React.KeyboardEvent) => {
    if (e.target instanceof HTMLButtonElement) return;
    const step = 1 / (video.fps || 24);
    if (e.key === " ") {
      e.preventDefault();
      toggle();
    } else if (e.key === "," || e.key === ".") {
      master.current?.pause();
      seek(time + (e.key === "." ? step : -step));
    }
  };

  return (
    <section className="stage" aria-label="Video">
      <div
        ref={frame}
        className={`stage-frame${done ? " is-compare" : ""}${dragging ? " is-dragging" : ""}`}
        style={{ aspectRatio: `${video.width} / ${video.height}`, ["--split" as string]: `${split}%` }}
        tabIndex={0}
        onKeyDown={onKey}
        onClick={(e) => {
          if (!(e.target as HTMLElement).closest(".veil-handle")) toggle();
        }}
      >
        <video
          ref={master}
          src={video.url}
          playsInline
          preload="auto"
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onEnded={() => setPlaying(false)}
        />
        {done && (
          <>
            <video ref={redacted} className="after" src={job!.outputs.redacted_url} muted playsInline preload="auto" hidden={view !== "redacted"} />
            <video ref={masks} className="after" src={job!.outputs.masks_url} muted playsInline preload="auto" hidden={view !== "masks"} />
            <span className="stage-tag left">Original</span>
            <span className="stage-tag right">{view === "redacted" ? "Redacted" : "Masks"}</span>
            <div
              className="veil-handle"
              role="slider"
              tabIndex={0}
              aria-label="Compare original and redacted"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(split)}
              onPointerDown={(e) => {
                e.currentTarget.setPointerCapture(e.pointerId);
                setDragging(true);
              }}
              onPointerMove={(e) => dragging && splitFromEvent(e.clientX)}
              onPointerUp={() => setDragging(false)}
              onKeyDown={(e) => {
                if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
                  e.preventDefault();
                  e.stopPropagation();
                  setSplit((s) => Math.max(0, Math.min(100, s + (e.key === "ArrowRight" ? 5 : -5))));
                }
              }}
            >
              <span className="veil-knob" aria-hidden />
            </div>
          </>
        )}
        {running && (
          <div className="stage-progress" role="status">
            <span>{activeStage ? STAGE_COPY[activeStage.key] : "Queued"}</span>
            <div className="bar">
              <div style={{ width: `${Math.round((job?.progress ?? 0) * 100)}%` }} />
            </div>
          </div>
        )}
      </div>

      <div className="transport">
        <button className="icon-btn" onClick={toggle} aria-label={playing ? "Pause" : "Play"}>
          {playing ? <Pause size={18} /> : <Play size={18} />}
        </button>
        <span className="timecode">
          {timecode(time)} <span className="faint">/ {timecode(duration)}</span>
        </span>
        <span className="transport-hint faint">Space to play, comma and period to step frames</span>
        {done && (
          <div className="segmented" role="group" aria-label="Right side shows">
            <button aria-pressed={view === "redacted"} onClick={() => setView("redacted")}>Redacted</button>
            <button aria-pressed={view === "masks"} onClick={() => setView("masks")}>Masks</button>
          </div>
        )}
      </div>

      <Timeline duration={duration} time={time} tracks={done ? job!.timeline.tracks ?? [] : []} onSeek={seek} hasRun={done} />
    </section>
  );
}
