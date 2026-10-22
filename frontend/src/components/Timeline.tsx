import { useMemo, useRef } from "react";
import { CLASS_LABEL, CLASS_ORDER, STATUS_LABEL } from "../lib/format";
import type { ConceptClass, TimelineTrack } from "../lib/types";

type Props = {
  duration: number;
  time: number;
  tracks: TimelineTrack[];
  hasRun: boolean;
  onSeek: (t: number) => void;
};

/** NLE-style lanes: one thin row per tracked object, grouped by concept class. */
export function Timeline({ duration, time, tracks, hasRun, onSeek }: Props) {
  const area = useRef<HTMLDivElement>(null);

  const groups = useMemo(() => {
    const by = new Map<ConceptClass, TimelineTrack[]>();
    for (const t of tracks) by.set(t.class, [...(by.get(t.class) ?? []), t]);
    return CLASS_ORDER.filter((c) => by.has(c)).map((c) => ({
      cls: c,
      tracks: by.get(c)!.sort((a, b) => (a.spans[0]?.[0] ?? 0) - (b.spans[0]?.[0] ?? 0)),
    }));
  }, [tracks]);

  const ticks = useMemo(() => {
    const step = duration > 30 ? 5 : duration > 12 ? 2 : 1;
    return Array.from({ length: Math.floor(duration / step) + 1 }, (_, i) => i * step);
  }, [duration]);

  const pct = (t: number) => `${(Math.max(0, Math.min(t, duration)) / duration) * 100}%`;

  const seekFrom = (e: React.PointerEvent) => {
    const rect = area.current?.getBoundingClientRect();
    if (!rect || !duration) return;
    onSeek(((e.clientX - rect.left) / rect.width) * duration);
  };

  return (
    <div className="timeline">
      <div className="timeline-labels" aria-hidden>
        <div className="ruler-label" />
        {groups.map((g) => (
          <div key={g.cls} className="lane-label" style={{ height: `calc(${g.tracks.length} * var(--lane))` }}>
            <span className={`swatch c-${g.cls}`} />
            {CLASS_LABEL[g.cls]} <span className="faint">{g.tracks.length}</span>
          </div>
        ))}
      </div>
      <div
        ref={area}
        className="timeline-area"
        role="slider"
        aria-label="Seek"
        aria-valuemin={0}
        aria-valuemax={duration}
        aria-valuenow={Number(time.toFixed(2))}
        tabIndex={0}
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId);
          seekFrom(e);
        }}
        onPointerMove={(e) => e.buttons === 1 && seekFrom(e)}
        onKeyDown={(e) => {
          if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
            e.preventDefault();
            onSeek(time + (e.key === "ArrowRight" ? 0.5 : -0.5));
          }
        }}
      >
        <div className="ruler">
          {ticks.map((t) => (
            <span key={t} className={`tick${t >= duration - 0.05 ? " end" : ""}`} style={{ left: pct(t) }}>
              {t}s
            </span>
          ))}
        </div>
        {groups.map((g) => (
          <div key={g.cls} className="lane-group">
            {g.tracks.map((t) => (
              <div key={t.track_id} className="lane">
                {t.spans.map(([a, b], i) => (
                  <span
                    key={i}
                    className={`clip c-${t.class} s-${t.status}`}
                    style={{ left: pct(a), width: `max(3px, calc(${pct(b)} - ${pct(a)}))` }}
                    title={`${t.track_id}: ${t.concept}, ${STATUS_LABEL[t.status].toLowerCase()}`}
                  />
                ))}
              </div>
            ))}
          </div>
        ))}
        {hasRun && groups.length === 0 && <p className="timeline-empty">Nothing matched the instruction in this clip.</p>}
        {!hasRun && <p className="timeline-empty">Tracked objects appear here after a run.</p>}
        <div className="playhead" style={{ left: pct(time) }} />
      </div>
    </div>
  );
}
