# Veil: hackathon plan

Sources: the self-audit review of 2026-10-24, and the Official Rules read on 2026-10-26 (text embedded in the [event details page](https://cerebralvalley.ai/e/meta-ai-developer-hackathon/details), dated October 25, 2026). Tasks are in execution order. Status lives in [progress.md](progress.md).

## What the Official Rules say

- **Dates.** Submissions due **Nov 4, 9:00 PM ET** at https://cerebralvalley.ai/e/meta-ai-developer-hackathon/hackathon/submit. Judging Nov 4–8; live finals and winners Nov 9. Dates can change; changes are posted on the event page.
- **Track.** Indie Developer: individuals and students, teams up to 4. 1st $150,000 · 2nd $100,000 · 3rd $75,000 · 4th $50,000 · 5th $25,000. Bonus prizes (community vote, most viral) up to $50,000 total.
- **Judging.**
  - Impact 30%: real-world potential, who benefits, how much it matters, whether people could use it, fit with a problem statement.
  - Use of Meta AI Tooling 25%: how creatively Meta's AI tooling is used, going beyond a basic integration, surfacing unexpected capabilities.
  - Technical Depth & Execution 20%: sound, thoughtfully refined engineering; real craft.
  - Demo 25%: a working, impressive demo that holds up live, and the quality of the demo video.
- **Prior work.** No rule requires code to be written during the event. The Submission must be "(a) … their … original work product; (b) … solely owned by the Participant … with no other person or entity having any right or interest in it; and (c) does not violate the intellectual property or other proprietary rights (including copyright, trademark, patent, contract, or privacy rights) of any other person or entity." Third-party technical help is allowed if the components remain your own work and you own all rights. Open source is allowed "provided applicable open source licenses are followed and the Submission enhances or builds upon the underlying open source product."
- **Meta tooling.** "Participants build with Meta AI products, including Muse Spark, Glimmer, SAM 3, and any other Meta AI products identified on the Event page." The event page lists the Muse Spark series, Muse Wonder, Muse Image and SAM 3.1 through Meta Model API, "run Meta open weights models", and Muse Code. The $150 Meta Model API credit is a perk for selected participants. No rule makes the API, or any partner's credits, a condition of entry or judging.
- **Not in the rules:** a demo-video length limit or required submission fields. Check the submit form before cutting the video.
- **Rights to watch.**
  - The Pexels clips are licensed, not owned, so clause (b) is a stretch for anything shipped inside the Submission. Use self-shot footage with consenting people for the hero clip; keep stock clips as secondary, credited material.
  - Follow model licenses: SAM 3.1 (Meta's SAM license), Muse Glimmer (Apache-2.0), and Llama 3.1 if used (Llama Community License: show "Built with Llama").

## Models and infrastructure (as of Oct 26)

- **Not available:**
  - Meta Model API: claiming credits at dev.meta.ai needs an international credit card. This rules out Muse Spark, which is API-only.
  - Modal and Baseten partner credits: applied for, unconfirmed.
- **Available:** Modal free tier (CLI logged in), Hugging Face access to SAM 3.1 and Llama 3.1, Groq.
- **Stack, all Meta open weights:**
  - **SAM 3.1** (`facebook/sam3.1`) on Modal: segmentation, tracking, and the output re-check.
  - **Muse Glimmer-30B** (`meta-models/Muse-Glimmer-30B`; Apache-2.0, ungated, released Aug 2026). The card says it is distilled from Muse Spark, with tool use, image input, and a 4-bit build that fits 24 GB. It is the planner (tool calls) and the visual auditor (sampled frames), served on Modal behind an OpenAI-compatible endpoint, so `planner/llama.py` and `auditor/muse_spark.py` only need `.env` changes.
  - **Fallbacks:**
    - Llama on Groq for the planner.
    - Muse Spark if API access appears.
    - The offline mocks, which already exist.
- **Cost rules:**
  - Scale to zero.
  - No warm containers except during recording and the Nov 4–8 judging window.
  - Cache finished results for the sample clips (T9).
  - Temperature 0 everywhere.
  - SAM results cached per clip and prompt.

## Tasks

**Needs** column: **none** = local only · **GPU** = Modal free tier · **you** = only you can do it.

### Phase 1: foundations (done or in flight)

| # | Task | Needs | Done when |
|---|---|---|---|
| T1 | Check the rules on pre-existing code. | none | Verified against the Official Rules (above) |
| T2 | Hugging Face access to SAM 3.1; read token. | you | Access granted |
| T3 | Real demo footage: 3 stock clips in `demo_assets/clips/` (see clip audit). | none | 3 real clips selectable in the UI |
| T5 | Frame-by-frame exceptions: an identity switch can't keep a stranger's face visible. | none | Crossing tests pass |
| T6 | SAM re-check of the output: every frame, coverage-based, feeds the patch loop. | none (live after T4) | Leaks between samples caught in tests; report states % of frames checked |
| T18 | Comparison renderer `demo_assets/make_comparison.py`: Original \| Veil output \| masks \| plan and audit panel, with audit-catch callouts. Added Oct 26 at your request. | none | Grid and pair videos render from any job |

### Phase 2: live Meta models (Oct 26–31)

| # | Task | Needs | Done when |
|---|---|---|---|
| T4 | SAM 3.1 on Modal: fix the `VERIFY` lines in `backend/modal_app/sam3_service.py`; cache weights in a Modal Volume; switch `/segment` to the video predictor (`tracked=true`); add point prompts to `/refine`. | GPU | Smoke test from the service docstring returns tracked objects |
| T4b | Zero-card fallback (added Oct 26, pending your go-ahead): a Colab notebook that runs SAM 3.1 in-process on the sample clips (bf16 → fp16/fp32 on the T4) and freezes the results with `python -m app.cache build`. Batch only; no tunnel. | you (HF access) + Colab | Frozen live-model runs in `demo_assets/cache/` |
| T11 | Planner on Muse Glimmer-30B: deploy an OpenAI-compatible endpoint on Modal (4-bit build on a 24 GB GPU, scale to zero) and point `PLANNER_MODEL`/`META_MODEL_API_BASE` at it. Fallback: Llama on Groq. | GPU | Top bar shows the planner live; demo instruction yields a valid plan |
| T12 | Auditor on Glimmer (vision) via `AUDITOR_API_BASE`/`AUDITOR_MODEL`: send 4 frames with a known unblurred face, confirm the box format (assumed 0–1000) and `image_url` input; fix `_audit_batch` if wrong. | GPU | Boxes land on the face |
| T3b | Self-shot hero clip(s), 20–30 s, consenting friends (see clip audit). Added Oct 26 from the clip audit. | you | Clip in `demo_assets/clips/` with sidecar |
| T13 | Full live run: hero clip + 3 stock clips × 4 instructions. Record wall-clock times, cost, and residual leaks before/after the audit loop. | GPU | Numbers table filled in |
| T7 | Planner eval: 12 instructions, plan-validity score, run against whichever planner is live (Glimmer; Groq/Ollama Llama for comparison). | GPU | Script prints a score for the README |
| T14 | Click-to-correct: `POST /api/jobs/{id}/corrections {frame, x, y, action: redact\|keep}` → SAM point refine → re-render only; click handler in `frontend/src/components/Stage.tsx`. | GPU (needs T4) | Clicking a person toggles their redaction |

### Phase 3: product and story (Nov 1–4; submit by Nov 3 to keep a buffer)

| # | Task | Needs | Done when |
|---|---|---|---|
| T8 | Redaction Report: SHA-256 of input/output/plan/audit in a hash chain, model ids, coverage, per-object table, before/after thumbnail per patch. No "GDPR/HIPAA compliant" claim; wording is "an evidence trail for a data-minimization review". | none | Downloadable from the UI |
| T9 | Presets + instant sample library: preset chips; cached results so the first click is instant. | none | Sample + preset returns instantly |
| T10 | Audio PII bleeping: per-word timestamps (faster-whisper), the planner tags words to remove under the same instruction, ffmpeg bleeps the spans; transcript strip with struck-through words. | GPU | Spoken name in the hero clip is bleeped |
| T15 | Deploy the FastAPI app to Modal with cached samples; warm only during judging. | GPU | Public URL works cold |
| T16 | 3:00 demo video per the storyboard below. Real wall-clock times; waits sped up and labelled; no mocks; voiceover recorded separately. | you + GPU | Video exported |
| T17 | README: hero GIF on real footage, pitch, "Why these Meta models", measured results, architecture, honest limitations, how to run, license notices and footage credits. Remove the planted-miss mechanism from the demo path. | none (needs T13 numbers) | README merged |

## Clip audit (Oct 26)

The three stock clips are competent B-roll, not world-class evidence. Judges have seen stock street footage many times, none of it is "a problem you know firsthand" (the Build for Your World theme), and none contains the moment the demo depends on: a real miss caught by the audit.

| Clip | Verdict | Role in the video |
|---|---|---|
| `manchester_street` | Good. Faces walk toward the camera; pink-coat exception; glass shopfronts. No plates, no confirmed reflection. | Exception walkthrough (60–100 s) if the hero clip falls short |
| `storefront_crossing` | Good stress test. People cross paths with occlusion, in low light and profile. Visually flat. | 5-second proof that identity holds through a crossing (T5) |
| `desk_inbox` | Fine for screens. Readable inbox and a phone, no people. 8 s. | Follow-up edit: "Also black out the laptop screen" |
| synthetic `street_crossing`, `community_meetup` | Never in the video. | Tests only |

**T3b: shoot this yourself.** About 30 s at 1080p, phone on a tripod or steady handheld, daylight. Every recognizable person must agree to appear. The checklist is the demo:
1. A friend in a distinctive jacket walks toward the camera (the exception).
2. Two or three other people pass, one crossing in front of her (T5).
3. A shop window or glass door where a passer-by's face shows **as a reflection** (the audit catch SAM's first pass is likely to miss).
4. A parked car with a readable plate.
5. Someone holding a phone with a visible notification.
6. Someone says a full name out loud: "Hey, Priya Sharma!" (T10).

Optional second clip: a laptop screen-share with names and email addresses for the follow-up edit.

## Demo video storyboard (3:00)

Produce with `make_comparison.py`: **pair** layout for the payoff shots, **grid** layout for the audit and architecture proof. Record the UI live at real speed and label any sped-up waits.

| Time | Shot | Proves (rubric) |
|---|---|---|
| **0:00–0:30 Hook** | Raw hero clip at full screen. VO: "I filmed this for my channel. It has 11 strangers' faces, a license plate, someone's phone, and my friend's full name. I can't post it." Type the instruction into Veil, press Redact, then cut to 3 s of the pair layout as the payoff. | Impact: a firsthand problem |
| **0:30–1:00 How it works** | Animated diagram: instruction → **Muse Glimmer** plans tool calls → **SAM 3.1** segments and tracks every instance → render → two auditors: **SAM 3.1 re-segments every output frame**, **Glimmer looks** at sampled frames → patch loop. Line: "SAM finds faces. It can't understand 'except'. Glimmer can, and it checks SAM's work." End on "All Meta open-weights models, running on Modal." | Use of Meta AI Tooling |
| **1:00–2:20 Walkthrough** | 1:00 The plan appears; flash the raw tool-call JSON (3 s). 1:15 Drag the veil: everyone is blurred except her (green "kept visible"). 1:30 Masks view on the crossing: her identity holds (storefront clip, 5 s). 1:45 **The moment**, in grid layout: the audit flags the reflection, the coral callout appears, SAM re-segments it, then a before/after zoom and "re-segmented 100% of frames". 2:00 Audio: the name is bleeped; the transcript shows it struck through. 2:10 Click a background person to redact them (T14), then type "Also black out the laptop screen" and it finishes in seconds. | Demo + Technical Depth |
| **2:20–3:00 Impact and depth** | Who it's for: creators, journalists, schools and event organisers, bodycam/dashcam footage requests. The numbers table from T13 and T7: residual leaks before → after the audit, planner validity, seconds and cost per minute of video. Download the Redaction Report and show the hash chain (T8). Close: "Veil. Privacy that checks its own work." Repo URL. | Impact + Technical Depth |

## Stock footage (T3)

Checked 2026-10-25. **None of these is CC0**: Pexels uses the [Pexels License](https://www.pexels.com/license/) (free use and modification, no attribution required; identifiable people may not appear in a bad light; no implied endorsement). Credit creators in the README.

| Clip | Source |
|---|---|
| `manchester_street` | [People Walking on a Busy Street](https://www.pexels.com/video/people-walking-on-a-busy-street-5021555/), zahid Anwar |
| `desk_inbox` | [Person Writing an Email](https://www.pexels.com/video/person-writing-an-email-6183189/), Taryn Elliott |
| `storefront_crossing` | [Pedestrians Passing by a Storefront](https://www.pexels.com/video/pedestrians-passing-by-a-storefront-3647789/), George Morina |
| `kl_traffic` (added Oct 26; 15 s cut from 0:03) | [Cars Stuck in Traffic](https://www.pexels.com/video/cars-stuck-in-traffic-3148319/), Nazim Zafri. Readable plate (BJJ 8840) throughout, riders facing the camera, a woman cycling toward the lens (exception). |
