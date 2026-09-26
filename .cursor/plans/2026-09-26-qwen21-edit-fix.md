# Qwen-Image-2.1 edit quality repair

User authorized fixing the 1024² teapot selective-color edit defect. Scope is Qwen-Image-2.1 only; preserve both v0.2.1 LoRAs, the requested Heretic encoder, existing local enhancements and other models.

- Isolate reference VAE encoding, visual/text conditioning, transformer and sampler using identical inputs and official pinned Diffusers code.
- Find the first divergent component; keep diagnostics and before/after images locally without replacing goldens.
- Apply the smallest evidenced correction and add regression coverage for that failure.
- Validate real r128/r256 edits and the affected base path, record actual settings and thermal context, then update the skill/report.

Prior controls: RGB versus RGBA inputs and KV cache ON/OFF retained the background artifact. Turbo uses the official 6/8-node schedule, CFG/scale 1 and unmerged adapters.

2026-09-27 outcome: source/edit initial-noise reuse is the reproduced cause. The full pinned Diffusers control also fails; changing only seed 42→43 succeeds. Independent edit noise is now default in the shared base/Turbo edit path, with legacy replay preserved. Both real r128/r256 WebUI tests pass at the original 1024²/6-step/seed-42 settings. Base-path noise routing is covered by a focused generation test; the diagnostic real 40-step base run was deliberately stopped at step 15 when the single-variable seed test became the decisive experiment. It is not counted as a completed quality check.
