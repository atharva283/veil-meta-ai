# Progress

Task details are in [PLAN.md](PLAN.md). Status: todo · in progress · blocked · done.

| Task | Status | Notes |
|---|---|---|
| T1 Rules check on pre-existing code | blocked | Needs you: read the official rules. |
| T2 SAM 3.1 Hugging Face access | done | Access granted and HF token ready (you, Oct 25). |
| T3 Real demo footage | done | 3 Pexels clips added (1280x720), listed before the synthetic ones; all run end to end offline. Fixed a bug found doing this: any clip with audio broke `masks.mp4` and later encodes (shared ffmpeg arg list mutated); regression test added, 21/21 pass. Gaps: no plate, no clear reflection, no spoken name in stock footage; you need to shoot those. |
| T4 SAM 3.1 on Modal | todo | Unblocked Oct 25: Modal CLI 1.6.2 logged in (profile atharva283), HF token ready. Deferred at your request so T6 goes first. |
| T5 Frame-by-frame exceptions | done | select/subtract split tracks per frame; a split exception shows as `<id>-visible` in the report; audit patches only skip frames that overlap the exception; patch frames on split fragments carry into follow-ups. Crossing tests fail on the old code and pass now; 24/24 pass. Side effect on the synthetic clip: her face is redacted on its first frame (frame 3), where her person mask has no head yet. |
| T6 SAM re-check of output | done (offline) | New `auditor/sam_recheck.py`: each audit round re-segments `redacted.mp4` (threshold 0.2, every frame); a detection is a leak if the pixels the effects actually hid cover <80% of it and it is not the exception on that frame. Findings join Muse Spark's in the patch loop. Report and audit panel state frames re-checked (e.g. 192/192). Tests: leak between samples caught; covered and exception detections ignored; with the sampled auditor silenced, the re-check alone patches face-07 and plate-05 (fails with re-check disabled). 27/27 pass. Offline it runs only on ground-truth clips: Haar on every frame of real footage patched pavement and doorways, so uploads/real clips skip it and report 0 frames re-checked. Live check pending T4; it adds one full-clip SAM pass per audit round. |
| T7 Planner eval (Ollama) | todo | Ollama is installed locally. |
| T8 Redaction Report | todo | |
| T9 Presets + cached samples | todo | |
| T10 Audio PII bleeping | todo | |
| T11 Kickoff: keys + newest Llama | todo | Needs keys. |
| T12 Muse Spark contract check | todo | Needs keys. |
| T13 Full live run + numbers | todo | Needs keys. |
| T14 Click-to-correct | todo | Needs T4 live. |
| T15 Deploy app to Modal | todo | Modal ready. |
| T16 Demo video | todo | Needs live results. |
| T17 README rewrite | todo | Needs T13 numbers. |
