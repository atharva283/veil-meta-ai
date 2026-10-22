import type { Health, Job, VideoInfo } from "./types";

export class ApiError extends Error {}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, init);
  } catch {
    throw new ApiError("Can't reach the Veil server. Start it with scripts/dev and try again.");
  }
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) detail = body.detail.map((d: { msg: string }) => d.msg).join("; ");
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(detail);
  }
  return res.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => request<Health>("/api/health"),
  samples: () => request<VideoInfo[]>("/api/samples"),
  upload: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<VideoInfo>("/api/videos", { method: "POST", body: form });
  },
  createJob: (video_id: string, instruction: string, parent_job_id?: string | null) =>
    request<Job>("/api/jobs", json({ video_id, instruction, parent_job_id: parent_job_id ?? null })),
  job: (id: string) => request<Job>(`/api/jobs/${id}`),
};
