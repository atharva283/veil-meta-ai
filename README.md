# Veil

Redact video from a plain-English instruction.

> "Blur every face and license plate except the woman in the red jacket, and black out phone screens."

**Llama 3.1** turns the instruction into a plan of tool calls. **SAM 3.1** segments and tracks each concept across every frame. A renderer applies blur, pixelate, or black-out. Then **Muse Spark** audits sampled output frames. If something leaked, the agent re-prompts SAM on that region, patches the track, and re-renders. Every object ends up in a timestamped audit log as verified, patched, kept visible, or needs review.

Built for the Meta Global AI Developer Hackathon (Indie Developer Track).

## Quick start (offline, no API keys)

Windows:

```powershell
scripts\setup.ps1
scripts\dev.ps1        # open http://localhost:5173
```

macOS / Linux:

```bash
scripts/setup.sh
scripts/dev.sh         # open http://localhost:5173
```

Pick **Market Street crossing** and press **Redact video**. With no keys, every model runs as a local mock, so the whole pipeline (including the audit catching two missed objects) works end to end. To serve everything from one port, run `npm run build` in `frontend/`; the API then serves the UI at http://localhost:8000.

Requirements: Python 3.11+, Node 20.19+. ffmpeg comes bundled through `imageio-ffmpeg`.

## How it works

```
instruction ──► Llama 3.1 planner ──► plan: segment / select / subtract / union / effect
                                          │ concepts
video ──► normalize ──► SAM 3.1 ──► tracks ──► compose ──► render ──► redacted.mp4 + masks.mp4
                          ▲                                   │ sampled frames
                          └──── refine(frame, box) ◄── Muse Spark audit ◄┘   (max 2 rounds)
```

| Plan op | Meaning |
|---|---|
| `segment(concept)` | SAM 3.1 text prompt, e.g. `"face"`, `"red jacket"` → tracked instances |
| `select(A, B)` | instances of A that overlap B (e.g. the person wearing the red jacket) |
| `subtract(A, B)` | instances of A that don't overlap B (faces except hers) |
| `union(A, B, …)` | merge instance sets |
| `effect(A, blur\|pixelate\|blackout\|highlight)` | render over every instance of A |

Follow-up edits ("Also hide the logo on the truck") extend the previous plan and reuse its tracks and audit patches, so they finish in seconds.

## Live vs mock

Each component goes live when its credentials are in `.env`. If a live call fails mid-run, Veil falls back to the mock and logs a warning in the run's activity log. The top bar shows which mode each model is in.

| Component | Live when | Offline mock |
|---|---|---|
| Planner (Llama 3.1) | `META_MODEL_API_KEY` set | Rule-based parser: effect verbs, target lists, `except …` descriptors, `X on the Y` |
| Segmenter (SAM 3.1) | `MODAL_ENDPOINT` set | Samples: per-frame ground truth, with "hard" objects withheld on the first pass to simulate SAM misses. Uploads: OpenCV Haar faces and plates only |
| Auditor (Muse Spark) | Meta key (or `AUDITOR_API_KEY`) set | Samples: compares output to ground truth. Uploads: re-runs Haar on redacted frames |

`VEIL_FORCE_MOCK=auditor` (or any comma list) keeps a component on its mock even with keys set. Use it to isolate a failing integration.

## Going live on Oct 26

1. `cp .env.example .env` and set `META_MODEL_API_KEY`. Confirm the base URL and model ids from the hackathon docs (`GET $META_MODEL_API_BASE/models`), then set `PLANNER_MODEL` and `AUDITOR_MODEL`.
2. Deploy SAM 3.1 on Modal:
   ```bash
   pip install modal && modal setup
   modal secret create veil-sam3 HF_TOKEN=hf_xxx VEIL_TOKEN=pick-a-secret
   modal deploy backend/modal_app/sam3_service.py
   ```
   Put the printed URL in `MODAL_ENDPOINT` and the secret in `MODAL_AUTH_TOKEN`.
3. Check the lines marked `VERIFY` in `backend/modal_app/sam3_service.py` against the SAM 3.1 release. They were written against the SAM 3 API and couldn't be run before the event.
4. Restart `scripts/dev`. All three chips in the top bar should read "live".
5. Add real demo footage: drop `clip.mp4` and a `clip.json` (title, description, `suggested_prompts`, `follow_up_prompts`, `order`) into `demo_assets/clips/`. No annotations are needed with live SAM. Use footage you have the rights to show.
6. On demo day, set `min_containers=1` on the Modal class so the first run doesn't wait for a cold start.

## Demo script (2 minutes)

1. **0:00** Raw street clip: "I want to post this, but it's full of strangers' faces, plates, and someone's phone."
2. **0:15** Type the instruction. The plan appears as numbered tool calls. Click **Show tool call** to show the raw `submit_plan` JSON.
3. **0:45** Drag the veil across the frame: everyone is blurred except the woman in red. Switch to **Masks** to show SAM's tracked instances.
4. **1:15** The audit log shows the far-sidewalk face and the edge-of-frame plate as **Patched by audit**. Click a row to jump to that moment.
5. **1:40** Follow-up: "Also hide the logo on the truck." It finishes in seconds and the new steps are marked **new**.

## API

| Method | Path | |
|---|---|---|
| GET | `/api/health` | mode of each component |
| GET | `/api/samples` | bundled clips |
| POST | `/api/videos` | multipart upload; normalized to H.264, ≤1280 px wide, ≤30 fps, ≤60 s |
| POST | `/api/plan` | `{instruction, parent_job_id?}` → plan preview without running |
| POST | `/api/jobs` | `{video_id, instruction, parent_job_id?}` → job (poll it) |
| GET | `/api/jobs/{id}` | stages, plan, events, outputs, audit log, timeline |

Outputs are written to `backend/data/jobs/<id>/`: `redacted.mp4`, `masks.mp4`, `plan.json`, `audit.json`.

## Layout

```
backend/
  app/
    main.py            FastAPI routes + static serving
    orchestrator.py    plan → segment → compose → render → audit → patch
    jobs.py            job records, stage timing, worker
    planner/           llama.py (live), mock.py (rule-based)
    segmenter/         modal_sam.py (live client), mock.py (ground truth / Haar)
    auditor/           muse_spark.py (live), mock.py
    pipeline/          video_io.py, tracks.py (mask algebra, RLE), effects.py
  modal_app/sam3_service.py   SAM 3.1 GPU service for Modal
  tests/                      planner, mask algebra, end-to-end, API contracts
frontend/src/
  App.tsx, components/  Stage (compare + veil), Timeline, PlanView, Pipeline, AuditLog
demo_assets/
  generate_demo_clips.py  synthetic clips with per-frame ground truth
  clips/                  street_crossing, community_meetup
```

## Tests

```bash
cd backend
.venv/Scripts/python -m pytest      # Windows
.venv/bin/python -m pytest          # macOS/Linux
```

These cover the planner grammar, mask algebra and RLE, a full end-to-end run on the street clip (exception kept, both planted misses patched, follow-up reuse), and contract tests for the Llama, Muse Spark, and SAM wire formats using fake HTTP responses.

## Known gaps

- The bundled clips are synthetic. With no keys, "segmentation" there reads ground truth, and the audit's catches are simulated misses. The UI labels these clips "simulated detections".
- Offline mode on uploaded clips can only find faces and license plates (Haar cascades), and it misses a lot. Everything else needs SAM 3.1.
- The SAM 3.1 service runs the image model frame by frame, and the client links detections by IoU. Switching `/segment` to SAM's video predictor (true identity tracking) is the first upgrade once it runs.
- The Muse Spark request assumes an OpenAI-compatible chat endpoint with `image_url` parts. Adjust `auditor/muse_spark.py` if the real API differs.
- Audio passes through untouched. Spoken names and voices are not redacted.
- Jobs live in memory and run one at a time. Restarting the API forgets job history; output files stay on disk.
