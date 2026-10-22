export type ModeInfo = { live: boolean; name: string };

export type Health = {
  ok: boolean;
  modes: { planner: ModeInfo; segmenter: ModeInfo; auditor: ModeInfo };
  samples: number;
};

export type VideoInfo = {
  id: string;
  title: string;
  description: string;
  is_sample: boolean;
  url: string;
  thumbnail_url: string | null;
  suggested_prompts: string[];
  follow_up_prompts: string[];
  simulated: boolean;
  width: number;
  height: number;
  fps: number;
  frame_count: number;
  duration_s: number;
  has_audio: boolean;
};

export type Op = "segment" | "union" | "select" | "subtract" | "effect";
export type Effect = "blur" | "pixelate" | "blackout" | "highlight";

export type PlanStep = {
  id: string;
  op: Op;
  concept: string | null;
  source: string | null;
  reference: string | null;
  inputs: string[];
  effect: Effect | null;
  note: string | null;
  label: string;
};

export type Plan = { summary: string; source: string; warnings: string[]; steps: PlanStep[] };

export type StageStatus = "pending" | "active" | "done" | "skipped" | "error";
export type Stage = {
  key: "plan" | "segment" | "compose" | "render" | "audit" | "patch";
  title: string;
  blurb: string;
  status: StageStatus;
  detail: string;
  duration_s: number | null;
};

export type AuditStatus = "verified" | "patched" | "kept_visible" | "flagged";
export type AuditEntry = {
  track_id: string;
  concept: string;
  effect: Effect | null;
  status: AuditStatus;
  start_s: number;
  end_s: number;
  frames: number;
  confidence: number;
  note: string;
};

export type ConceptClass = "face" | "plate" | "screen" | "person" | "other";
export type TimelineTrack = {
  track_id: string;
  concept: string;
  class: ConceptClass;
  status: AuditStatus;
  effect: Effect | null;
  spans: [number, number][];
};

export type JobEvent = { t: number; level: "info" | "warn" | "error"; stage: string | null; message: string };

export type Job = {
  id: string;
  video_id: string;
  instruction: string;
  parent_job_id: string | null;
  status: "queued" | "running" | "done" | "error";
  error: string | null;
  progress: number;
  elapsed_s: number;
  plan: Plan | null;
  stages: Stage[];
  events: JobEvent[];
  outputs: { original_url?: string; redacted_url?: string; masks_url?: string; audit_url?: string };
  audit: {
    entries?: AuditEntry[];
    counts?: Record<AuditStatus, number>;
    rounds?: number;
    audited_frames?: number;
  };
  timeline: { duration_s?: number; fps?: number; tracks?: TimelineTrack[] };
  modes: { planner?: string; segmenter?: string; auditor?: string };
};
