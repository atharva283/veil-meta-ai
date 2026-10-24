# Veil: hackathon plan

Source of truth: the self-audit review of 2026-10-24. Tasks are in priority order; work them top to bottom.
Status lives in [progress.md](progress.md).

**Keys** column:
- **none**: can be done now, no hackathon API keys.
- **you**: no hackathon keys, but needs an action or account only you can do.
- **keys**: needs the Meta Model API keys / Modal credits handed out at kickoff (Oct 26).

## Phase 1: before kickoff (Oct 24–25)

| # | Task | Keys | Done when |
|---|---|---|---|
| T1 | Check hackathon rules on pre-existing code (repo's first commit is Oct 22). If strict, get a written answer from organizers. | you | Rule read; decision recorded in progress.md |
| T2 | Request access to the gated SAM 3.1 weights on Hugging Face; create a read token. | you | Access granted |
| T3 | Real demo footage: 3 clips (street + exception + reflection, desk with laptop screen, crowd with two people crossing). Candidate stock clips below. Add each to `demo_assets/clips/` with a `clip.json`. | none | 3 real clips selectable in the UI |
| T4 | SAM 3.1 on Modal: fix the `VERIFY` lines in `backend/modal_app/sam3_service.py`, cache weights in a Modal Volume, switch `/segment` to the video predictor (`tracked=true`), add point prompts to `/refine`. | you (Modal account + T2) | Smoke test from the service docstring returns tracked objects |
| T5 | Frame-by-frame exceptions: `select`/`subtract` decide per frame, not per whole track, so an identity switch can't keep a stranger's face visible. Test with two people crossing. (`backend/app/pipeline/tracks.py`, `_compose` in `backend/app/orchestrator.py`) | none | New crossing test passes; existing tests pass |
| T6 | SAM re-check of the output: run SAM on `redacted.mp4` (same concepts, low threshold, every frame); flag detections less than ~80% covered by applied masks; feed findings into the existing patch loop next to Muse Spark. New `backend/app/auditor/sam_recheck.py`. | none (mock now; live after T4) | Leaks between audit samples are caught in tests; report states % of frames checked |
| T7 | Planner eval: 12 instructions, plan-validity score, run against Ollama (`http://localhost:11434/v1`, `llama3.1:8b`). | none | Script prints a score for the README |
| T8 | Redaction Report: SHA-256 of input/output/plan/audit in a hash chain, model ids, coverage (frames checked and how), per-object table, before/after thumbnail per patch. No "GDPR/HIPAA compliant" claim; wording is "an evidence trail for a data-minimization review". | none | Downloadable from the UI |
| T9 | Presets + instant sample library: preset chips that fill the instruction; cache finished results for the sample clips so the first click is instant. | none | Sample + preset returns instantly |
| T10 | Audio PII bleeping: transcription with per-word timestamps (faster-whisper; a Meta speech model instead if one gives word timestamps), Llama tags words to remove from the same instruction, ffmpeg bleeps spans; transcript strip with struck-through words. | none to start (live Llama tagging needs keys) | Spoken name in a test clip is bleeped |

## Phase 2: kickoff (Oct 26)

| # | Task | Keys | Done when |
|---|---|---|---|
| T11 | Fill `.env`; `GET /models`; set the newest allowed Llama as planner. If a multimodal Llama is available, give the planner a keyframe. | keys | Top bar shows planner "live" |
| T12 | Muse Spark contract check: send 4 frames with a known unblurred face, confirm box format (assumed 0–1000) and `image_url` input; fix `_audit_batch` if wrong. | keys | Boxes land on the face |
| T13 | Full live run: 3 clips × 4 instructions. Record wall-clock times and before/after residual leak counts. | keys | Numbers table filled in |
| T14 | Click-to-correct: `POST /api/jobs/{id}/corrections {frame, x, y, action: redact\|keep}` → SAM point refine → re-render only; click handler in `frontend/src/components/Stage.tsx`. | you (needs T4 live) | Clicking a person toggles their redaction |
| T15 | Deploy the FastAPI app to Modal with cached samples; keep a container warm only during judging. | you (Modal) | Public URL works cold |

## Phase 3: polish (Oct 27)

| # | Task | Keys | Done when |
|---|---|---|---|
| T16 | 2:00 demo video per the review's storyboard. Real wall-clock times; waits sped up and labelled; no mocks; voiceover recorded separately; run from cached results. | keys (live results) | Video exported |
| T17 | README rewrite: hero GIF on real footage, one-line pitch, "Why three Meta models", measured results table, architecture, honest limitations, how to run. Replace synthetic clips and remove the planted-miss mechanism from the demo path. | keys (needs T13 numbers) | README merged |

## T3 candidate footage

Checked 2026-10-25 on each clip's page. **None of these is CC0.** Pexels and Pixabay use their own licenses:

- **[Pexels License](https://www.pexels.com/license/)**: free to use and modify, no attribution required. Identifiable people may not appear in a bad light or offensively; don't imply endorsement; don't resell unaltered copies.
- **[Pixabay Content License](https://pixabay.com/service/license-summary/)**: free to use and modify, no attribution required. No immoral or illegal use, especially of content showing recognisable people; you are responsible for checking whether third-party consent is needed; no standalone redistribution.

A privacy-redaction demo doesn't show anyone in a bad light, so both licenses allow it. Credit the creators in the README anyway.

| Scenario | Chosen (in `demo_assets/clips/`) | Alternate |
|---|---|---|
| Street: faces toward camera, exception person, glass shopfronts | `manchester_street`: [People Walking on a Busy Street](https://www.pexels.com/video/people-walking-on-a-busy-street-5021555/), zahid Anwar, 0:13, Pexels License ("Free"). Exception: the woman in the pink coat. | [Pedestrians Passing by a Storefront](https://www.pexels.com/video/pedestrians-passing-by-a-storefront-3647789/) |
| Desk: laptop screen with email | `desk_inbox`: [Person Writing an Email](https://www.pexels.com/video/person-writing-an-email-6183189/), Taryn Elliott, 0:08, Pexels License. Gmail inbox on screen, phone on the books, no faces. | [A Man Typing on His Computer in the Cafe](https://www.pexels.com/video/a-man-typing-on-his-computer-in-the-cafe-4318511/), Pexels License. Rejected: screen is edge-on and unreadable. |
| Crowd: people crossing paths | `storefront_crossing`: [Pedestrians Passing by a Storefront](https://www.pexels.com/video/pedestrians-passing-by-a-storefront-3647789/), George Morina, 0:21, Pexels License. Many people crossing sideways with occlusions. | [Traffic, Crowd, Crossing (Shibuya)](https://pixabay.com/videos/traffic-crowd-crossing-crosswalk-181114/) and [Crosswalk, Crowd, People](https://pixabay.com/videos/crosswalk-crowd-people-many-3630/), Pixabay Content License. Not used: Pixabay returns 403 to scripted downloads. |

Caveats:
- Frames of each chosen clip were checked. None shows a readable license plate, and none has a clear face reflected in glass; the shop-window reflection catch still needs footage you shoot yourself.
- Stock clips don't contain someone saying a full name out loud, which T10 needs. Record that clip yourself, with consent.
