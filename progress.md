# Progress

Task details are in [PLAN.md](PLAN.md), listed in execution order. Status: todo · in progress · blocked · done.

| Task | Status | Notes |
|---|---|---|
| T1 Rules check on pre-existing code | done | Passed Oct 26 against the Official Rules (dated Oct 25): no "built during the event" requirement; the Submission must be original, solely owned and non-infringing; open source allowed if licenses are followed. Caveat: stock clips are licensed, not owned; hero footage should be self-shot (T3b). |
| T2 SAM 3.1 Hugging Face access | done | Access granted and HF token ready (you, Oct 25). |
| T3 Real demo footage | done | 3 Pexels clips added (1280x720), listed before the synthetic ones. Fixed a bug found doing this: any clip with audio broke `masks.mp4` and later encodes (shared ffmpeg arg list mutated); regression test added. Clip audit Oct 26: secondary material only; see T3b. |
| T5 Frame-by-frame exceptions | done | select/subtract split tracks per frame; a split exception shows as `<id>-visible`; audit patches only skip frames that overlap the exception. Crossing tests fail on the old code and pass now. Side effect on the synthetic clip: her face is redacted on its first frame, where her person mask has no head yet. |
| T6 SAM re-check of output | done (offline) | `auditor/sam_recheck.py` re-segments `redacted.mp4` (threshold 0.2, every frame); leak = <80% of the detection hidden and not the exception on that frame. Report and UI show frames re-checked. With the sampled auditor silenced, the re-check alone patches both planted misses. Offline it runs only on ground-truth clips (Haar on real footage made false patches). Live check after T4; adds one full-clip SAM pass per audit round. |
| T18 Comparison renderer | done | Added Oct 26 at your request. `demo_assets/make_comparison.py --original … --job … --out … [--layout grid\|pair]`: 1920x1080 grid (original, output, masks, plan/audit panel with audit-catch callouts) or 1920x630 pair. Labels show the recorded model modes, so mock runs say "mock". `audit.json` now records `modes`. 27/27 tests pass. |
| T4 SAM 3.1 on Modal | blocked | Oct 26: service rewritten against the real SAM 3.1 API (sam3 @ f1647e8: video predictor with stable ids, point-prompt /refine, weights cached in Volume `veil-sam3-cache`, Flash-Attention 3 off, /health reports load errors); client follows Modal's 303 for long requests. Image builds. Secret `veil-sam3` created (HF_TOKEN + VEIL_TOKEN). Two blockers: (1) HF account Ath-333AB is not approved for facebook/sam3.1 or facebook/sam3 (403; Llama 3.1 access works); (2) Modal refuses every GPU type (L40S, L4, A10G, T4) without a payment method on the account. |
| T11 Planner on Muse Glimmer-30B | blocked | Self-hosting needs a Modal GPU (payment method). Hosted alternative: OpenRouter serves `meta/muse-glimmer-30b`, `meta/muse-spark-1.3` and `meta-llama/llama-4-scout` per token (needs a card or crypto). Card-free option: Llama 4 Scout on Groq's free tier, if Groq still lists it. |
| T12 Auditor on Glimmer (vision) | todo | Box-format check; Muse Spark only if API credits become claimable. |
| T3b Self-shot hero clip | todo | Needs you: the shot checklist is in PLAN.md (clip audit). Any day this week. |
| T13 Full live run + numbers | todo | Needs T4, T11, T12. |
| T7 Planner eval | todo | Against the live planner. |
| T14 Click-to-correct | todo | Needs T4 point prompts. |
| T8 Redaction Report | todo | |
| T9 Presets + cached samples | todo | |
| T10 Audio PII bleeping | todo | Needs the hero clip's spoken name. |
| T15 Deploy app to Modal | todo | |
| T16 Demo video (3:00) | todo | Storyboard in PLAN.md. Check the submit form for a length limit. |
| T17 README rewrite | todo | Needs T13 numbers; add license notices and footage credits. |
