"""Redaction Report: a tamper-evident evidence trail for one finished job.

Every artifact is hashed (SHA-256) and folded into a chain:

    link_0 = sha256("veil-report-v1")
    link_i = sha256(link_{i-1} + name_i + sha256(artifact_i))

so changing, dropping or reordering any artifact changes every later link and
the final digest printed on the report. Recompute it with:

    python -m app.report verify <job dir> --original <source clip>

The report is an evidence trail for a data-minimization review. It does not
certify compliance with any law or standard.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .pipeline import video_io
from .pipeline.tracks import Track

CHAIN_SEED = "veil-report-v1"
ARTIFACTS = ("plan.json", "redacted.mp4", "masks.mp4", "audit.json")  # after the input clip, in chain order


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_chain(hashes: list[tuple[str, str]]) -> list[dict]:
    link = hashlib.sha256(CHAIN_SEED.encode()).hexdigest()
    out = []
    for name, digest in hashes:
        link = hashlib.sha256(f"{link}{name}{digest}".encode()).hexdigest()
        out.append({"name": name, "sha256": digest, "link": link})
    return out


def _artifact_hashes(job_dir: Path, original: Path) -> list[tuple[str, str]]:
    return [("input", file_sha256(original))] + [(n, file_sha256(job_dir / n)) for n in ARTIFACTS if (job_dir / n).exists()]


def verify(job_dir: Path, original: Path) -> tuple[bool, str | None]:
    """(True, None) when the files on disk reproduce the recorded chain, else (False, first mismatch)."""
    recorded = json.loads((job_dir / "report.json").read_text(encoding="utf-8"))["artifacts"]
    actual = build_chain(_artifact_hashes(job_dir, original))
    for want, got in zip(recorded, actual):
        if want != got:
            return False, want["name"]
    if len(recorded) != len(actual):
        return False, (recorded[len(actual)] if len(recorded) > len(actual) else actual[len(recorded)])["name"]
    return True, None


def _thumbnail(original: Path, redacted: Path, track: Track, frame_index: int) -> str | None:
    """Before/after crop around the patched object, as a JPEG data URI."""
    if not track.frames:
        return None
    f = min(track.frames, key=lambda k: abs(k - frame_index))
    before, after = video_io.read_frame(original, f), video_io.read_frame(redacted, f)
    if before is None or after is None:
        return None
    x0, y0, x1, y1 = track.frames[f].box
    pad = int(0.6 * max(x1 - x0, y1 - y0)) + 24
    H, W = before.shape[:2]
    X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    crops = []
    for img in (before, after):
        c = img[Y0:Y1, X0:X1]
        crops.append(cv2.resize(c, (max(1, int(c.shape[1] * 140 / c.shape[0])), 140), interpolation=cv2.INTER_AREA))
    gap = np.full((140, 6, 3), 255, np.uint8)
    ok, buf = cv2.imencode(".jpg", np.hstack([crops[0], gap, crops[1]]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else None


def write_report(job, video, patched_tracks: list[tuple[Track, int]], job_dir: Path) -> dict:
    """Write report.json and report.html into job_dir; returns the report dict."""
    audit = job.audit
    chain = build_chain(_artifact_hashes(job_dir, video.path))
    meta = video.meta
    report = {
        "version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "job_id": job.id,
        "instruction": job.instruction,
        "video": {"title": video.title, **meta.as_dict()},
        "models": dict(job.modes),
        "plan": [{"step": s.label, "note": s.note} for s in job.plan.steps] if job.plan else [],
        "coverage": {
            "sampled_frames": audit.get("audited_frames", 0),
            "rechecked_frames": audit.get("rechecked_frames", 0),
            "frame_count": meta.frame_count,
            "rounds": audit.get("rounds", 0),
        },
        "counts": audit.get("counts", {}),
        "objects": audit.get("entries", []),
        "artifacts": chain,
        "digest": chain[-1]["link"],
    }
    (job_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    thumbs = [(t, f, _thumbnail(video.path, job_dir / "redacted.mp4", t, f)) for t, f in patched_tracks]
    (job_dir / "report.html").write_text(_html(report, thumbs), encoding="utf-8")
    return report


def _html(r: dict, thumbs: list[tuple[Track, int, str | None]]) -> str:
    e = html.escape
    cov, counts, fps = r["coverage"], r["counts"], r["video"]["fps"] or 1
    rechecked = (
        f"SAM re-segmented {cov['rechecked_frames']} of {cov['frame_count']} output frames "
        f"({round(100 * cov['rechecked_frames'] / max(1, cov['frame_count']))}%)"
        if cov["rechecked_frames"]
        else "No full-frame re-check ran"
    )
    rows = "".join(
        f"<tr><td>{e(o['track_id'])}</td><td>{e(o['concept'])}</td><td>{e(o['effect'] or '—')}</td>"
        f"<td class='st-{e(o['status'])}'>{e(o['status'].replace('_', ' '))}</td>"
        f"<td class='nowrap'>{o['start_s']:.2f}–{o['end_s']:.2f}s</td><td>{o['frames']}</td><td>{e(o['note'])}</td></tr>"
        for o in r["objects"]
    )
    patches = "".join(
        f"<figure><img src='{src}' alt='Before and after for {e(t.track_id)}'>"
        f"<figcaption>{e(t.concept)} · {e(t.track_id)} · frame {f} ({f / fps:.2f}s) · before | after</figcaption></figure>"
        for t, f, src in thumbs
        if src
    ) or "<p class='muted'>No object needed an audit fix.</p>"
    plan = "".join(f"<li>{e(p['step'])}{' — ' + e(p['note']) if p['note'] else ''}</li>" for p in r["plan"])
    models = "".join(f"<li><b>{e(k)}</b>: {e(v)}</li>" for k, v in r["models"].items())
    chain = "".join(f"<tr><td>{e(a['name'])}</td><td><code>{a['sha256']}</code></td><td><code>{a['link']}</code></td></tr>" for a in r["artifacts"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Veil redaction report</title>
<style>
:root {{ --fg:#1d1d1f; --muted:#6b6b72; --line:#e3e3e8; --bg:#fff; --ok:#1f7a3f; --patch:#c2410c; --kept:#2563eb; --flag:#b91c1c; }}
body {{ font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--fg); background: var(--bg); max-width: 1000px; margin: 0 auto; padding: 32px 16px; }}
h1 {{ font-size: 24px; margin: 0 0 4px; }} h2 {{ font-size: 16px; margin: 28px 0 8px; }}
.muted {{ color: var(--muted); }} blockquote {{ margin: 12px 0; padding: 8px 14px; border-left: 3px solid var(--line); font-size: 16px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }} td, th {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }}
code {{ font-size: 11px; word-break: break-all; }} .nowrap {{ white-space: nowrap; }} .digest {{ font-size: 13px; padding: 10px 12px; border: 1px solid var(--line); border-radius: 6px; }}
.st-verified {{ color: var(--ok); }} .st-patched {{ color: var(--patch); }} .st-kept_visible {{ color: var(--kept); }} .st-flagged {{ color: var(--flag); font-weight: 600; }}
.figs {{ display: flex; flex-wrap: wrap; gap: 16px; }} figure {{ margin: 0; }} figure img {{ height: 140px; border: 1px solid var(--line); }} figcaption {{ font-size: 12px; color: var(--muted); }}
footer {{ margin-top: 32px; font-size: 12px; color: var(--muted); }}
</style></head><body>
<h1>Redaction report</h1>
<p class="muted">{e(r['video']['title'])} · {r['video']['width']}×{r['video']['height']}, {r['video']['duration_s']:.1f}s, {r['video']['frame_count']} frames · job {e(r['job_id'])} · generated {e(r['generated_at'])}</p>
<blockquote>“{e(r['instruction'])}”</blockquote>
<p class="digest"><b>Report digest</b> <code>{r['digest']}</code></p>

<h2>Outcome</h2>
<p>{counts.get('verified', 0) + counts.get('patched', 0)} objects redacted, {counts.get('patched', 0)} of them fixed by the audit,
{counts.get('kept_visible', 0)} kept visible on purpose, {counts.get('flagged', 0)} flagged for manual review.</p>
<p>Checks: {cov['sampled_frames']} sampled frames reviewed by the auditor; {rechecked}; {cov['rounds']} audit round{'s' if cov['rounds'] != 1 else ''}.</p>

<h2>Models</h2><ul>{models}</ul>
<h2>Plan</h2><ol>{plan}</ol>

<h2>Objects</h2>
<table><thead><tr><th>Id</th><th>Concept</th><th>Effect</th><th>Status</th><th>When</th><th>Frames</th><th>Note</th></tr></thead><tbody>{rows}</tbody></table>

<h2>Audit fixes</h2><div class="figs">{patches}</div>

<h2>Artifact chain</h2>
<table><thead><tr><th>Artifact</th><th>SHA-256</th><th>Chain link</th></tr></thead><tbody>{chain}</tbody></table>
<p class="muted">link₀ = sha256("{CHAIN_SEED}"), linkᵢ = sha256(linkᵢ₋₁ + name + sha256). The digest is the last link.
Recompute with <code>python -m app.report verify &lt;job dir&gt; --original &lt;clip&gt;</code>.</p>

<footer>This report is an evidence trail for a data-minimization review of one video. It does not certify compliance with any law or standard; automated redaction can miss things, so review flagged moments by hand.</footer>
</body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify a Veil redaction report's artifact chain.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify")
    v.add_argument("job_dir", type=Path)
    v.add_argument("--original", type=Path, required=True)
    args = ap.parse_args()
    ok, bad = verify(args.job_dir, args.original)
    digest = json.loads((args.job_dir / "report.json").read_text(encoding="utf-8"))["digest"]
    print(f"OK: chain reproduces digest {digest}" if ok else f"MISMATCH: {bad} differs from the recorded report")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
