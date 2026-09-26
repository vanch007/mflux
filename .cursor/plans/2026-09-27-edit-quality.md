# Qwen 2.1 edit quality

User authorized diagnosis, local fixes and retesting. Preserve unrelated changes and weights.

1. Audit official Viggle recipe and the recorded previous runs.
2. Compare safe, identical portrait color edits at 1600x960, seed 42, bf16, runtime LoRA and KV on. Isolate CFG, scale and step count; record temperature and separate load/generation time.
3. Separate reference encoding resolution from output sizing; retain legacy metadata replay. Remove WebUI's unnecessary whole-image output resizing and expose actual output dimensions and optional LoRA scale.
4. Run focused regression checks and real local generation. Publish a self-contained comparison HTML and document only supported findings.

Do not change the text encoder, model weights, unrelated models, or promote experimental CFG as a quality default. Explicit sexual edits are excluded from this validation.
