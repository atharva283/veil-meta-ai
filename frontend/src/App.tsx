import { useCallback, useEffect, useRef, useState } from "react";
import { AuditLog } from "./components/AuditLog";
import { InstructionPanel } from "./components/InstructionPanel";
import { Pipeline } from "./components/Pipeline";
import { PlanView } from "./components/PlanView";
import { SourcePicker } from "./components/SourcePicker";
import { Stage } from "./components/Stage";
import { TopBar } from "./components/TopBar";
import { api } from "./lib/api";
import type { Health, Job, VideoInfo } from "./lib/types";

const POLL_MS = 400;

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [samples, setSamples] = useState<VideoInfo[]>([]);
  const [video, setVideo] = useState<VideoInfo | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [previousSteps, setPreviousSteps] = useState(0);
  const [instruction, setInstruction] = useState("");
  const [followUp, setFollowUp] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [time, setTime] = useState(0);
  const seekFn = useRef<(t: number) => void>(() => {});

  useEffect(() => {
    api.health().then(setHealth).catch((e) => setError(e.message));
    api.samples().then(setSamples).catch(() => {});
  }, []);

  const running = job?.status === "queued" || job?.status === "running";

  // Poll the active job until it settles.
  useEffect(() => {
    if (!job || !running) return;
    const id = window.setTimeout(async () => {
      try {
        const next = await api.job(job.id);
        setJob(next);
        if (next.status === "done" && followUp) setInstruction("");
      } catch (e) {
        setError((e as Error).message);
      }
    }, POLL_MS);
    return () => window.clearTimeout(id);
  }, [job, running, followUp]);

  const pick = (v: VideoInfo | null) => {
    setVideo(v);
    setJob(null);
    setPreviousSteps(0);
    setError(null);
    setTime(0);
    setInstruction(v?.suggested_prompts[0] ?? "");
  };

  const upload = async (file: File) => {
    setUploading(true);
    setError(null);
    try {
      pick(await api.upload(file));
    } catch (e) {
      setError(`Upload failed: ${(e as Error).message}`);
    } finally {
      setUploading(false);
    }
  };

  const run = async () => {
    if (!video) return;
    const parent = followUp && job?.status === "done" ? job : null;
    setError(null);
    try {
      const next = await api.createJob(video.id, instruction.trim(), parent?.id);
      setPreviousSteps(parent?.plan?.steps.length ?? 0);
      setJob(next);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const onTime = useCallback((t: number) => setTime(t), []);
  const registerSeek = useCallback((fn: (t: number) => void) => {
    seekFn.current = fn;
  }, []);
  const seek = useCallback((t: number) => seekFn.current(t), []);

  const editing = followUp && job?.status === "done";
  const suggestions = (video ? (editing ? video.follow_up_prompts : video.suggested_prompts) : []).filter(
    (s) => s !== instruction,
  );

  return (
    <div className="app">
      <TopBar health={health} video={video} busy={running} onChangeClip={() => pick(null)} />
      <main className="workspace">
        <div className="main-col">
          {video ? (
            <>
              <Stage video={video} job={job} time={time} onTime={onTime} registerSeek={registerSeek} />
              <Pipeline job={job} />
            </>
          ) : (
            <SourcePicker samples={samples} uploading={uploading} onUpload={upload} onPick={pick} />
          )}
        </div>
        <aside className="side-col">
          <InstructionPanel
            value={instruction}
            onChange={setInstruction}
            suggestions={suggestions}
            canFollowUp={job?.status === "done"}
            followUp={followUp}
            onFollowUpChange={setFollowUp}
            ready={!!video}
            busy={running}
            error={error}
            onRun={run}
          />
          <PlanView plan={job?.plan ?? null} previousSteps={previousSteps} running={running} />
        </aside>
      </main>
      {job?.status === "done" && <AuditLog job={job} time={time} onSeek={seek} />}
    </div>
  );
}
