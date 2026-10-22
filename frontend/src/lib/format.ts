import type { AuditStatus, ConceptClass, Effect } from "./types";

/** 00:03.42 style timecode. */
export function timecode(s: number): string {
  if (!Number.isFinite(s) || s < 0) s = 0;
  const m = Math.floor(s / 60);
  const sec = s - m * 60;
  return `${String(m).padStart(2, "0")}:${sec.toFixed(2).padStart(5, "0")}`;
}

export function seconds(s: number | null): string {
  if (s == null) return "";
  return s < 1 ? `${Math.round(s * 1000)} ms` : `${s.toFixed(1)} s`;
}

export const CLASS_ORDER: ConceptClass[] = ["face", "plate", "screen", "person", "other"];

export const CLASS_LABEL: Record<ConceptClass, string> = {
  face: "Faces",
  plate: "Plates",
  screen: "Screens",
  person: "People",
  other: "Other",
};

export function conceptClass(concept: string): ConceptClass {
  const c = concept.toLowerCase();
  if (c.includes("face") || c.includes("head")) return "face";
  if (c.includes("plate")) return "plate";
  if (["screen", "monitor", "laptop", "phone", "display"].some((k) => c.includes(k))) return "screen";
  if (["person", "people", "woman", "man", "child"].some((k) => c.includes(k))) return "person";
  return "other";
}

export const STATUS_LABEL: Record<AuditStatus, string> = {
  verified: "Verified",
  patched: "Patched by audit",
  kept_visible: "Kept visible",
  flagged: "Needs review",
};

export const EFFECT_LABEL: Record<Effect, string> = {
  blur: "Blur",
  pixelate: "Pixelate",
  blackout: "Black out",
  highlight: "Highlight",
};
