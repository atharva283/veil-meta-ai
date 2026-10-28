import { LoaderCircle, Upload } from "lucide-react";
import { useRef, useState } from "react";
import type { VideoInfo } from "../lib/types";

type Props = {
  samples: VideoInfo[];
  uploading: boolean;
  onUpload: (file: File) => void;
  onPick: (video: VideoInfo) => void;
  replayOnly?: boolean; // public demo: recorded runs only, no uploads
};

export function SourcePicker({ samples, uploading, onUpload, onPick, replayOnly }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  const take = (files: FileList | null) => {
    const file = files?.[0];
    if (file) onUpload(file);
  };

  return (
    <section className="picker" aria-label="Choose a video">
      {replayOnly ? (
        <div className="dropzone">
          <div className="dropzone-body">
            <h2>Replays of recorded runs</h2>
            <p className="muted">
              This demo replays runs recorded on a GPU with Llama 3.1 and SAM 3.1, so nothing is computed while you
              watch. Pick a clip, then one of its suggested instructions.
            </p>
          </div>
        </div>
      ) : (
      <div
        className={`dropzone${over ? " is-over" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          take(e.dataTransfer.files);
        }}
      >
        {uploading ? (
          <div className="dropzone-body">
            <LoaderCircle className="spin" size={28} aria-hidden />
            <h2>Uploading and converting</h2>
            <p className="muted">Veil makes a working copy so every frame lines up with the result.</p>
          </div>
        ) : (
          <div className="dropzone-body">
            <h2>Drop a video to redact</h2>
            <p className="muted">MP4, MOV or WebM. Veil uses the first 60 seconds.</p>
            <button className="btn btn-primary" onClick={() => input.current?.click()}>
              <Upload size={16} aria-hidden /> Choose a file
            </button>
            <input
              ref={input}
              type="file"
              accept="video/mp4,video/quicktime,video/webm,video/x-matroska,.mp4,.mov,.m4v,.webm,.mkv,.avi"
              hidden
              onChange={(e) => take(e.target.files)}
            />
          </div>
        )}
      </div>
      )}

      {samples.length > 0 && (
        <div className="samples">
          <h3>{replayOnly ? "Sample clips" : "Or start with a sample clip"}</h3>
          <ul>
            {samples.map((s) => (
              <li key={s.id}>
                <button className="sample" onClick={() => onPick(s)} disabled={uploading}>
                  {s.thumbnail_url && <img src={s.thumbnail_url} alt="" />}
                  <span className="sample-text">
                    <strong>{s.title}</strong>
                    <span className="muted">{s.description}</span>
                    <span className="faint">
                      {Math.round(s.duration_s)} seconds{s.simulated ? ", simulated detections" : ""}
                    </span>
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
