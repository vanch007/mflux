---
name: mflux-qwen-image-21
description: "Local Qwen-Image-2.1-Turbo generation/editing, 智能选择 VAE 与功能性 LoRA, mflux-paint maintenance, PE diagnosis and comparisons on Apple Silicon. Excludes other models, remote services and unrelated MLX administration."
metadata:
  last_verified: "2026-10-10"
---

# Qwen-Image-2.1 Turbo via mflux

Use official Turbo DiT + Heretic encoder. Read [selection policy](references/selection-policy.md) before choosing VAE/adapters and [catalog](references/lora-catalog.md) for exact files/triggers. Preserve user intent.

## Deployment and invariants

Core `/Users/vanch/mflux-pr736`; WebUI `/Users/vanch/mflux-paint`. Run from core with `uv run --no-sync`. Model directories under core `models/`:

- `Qwen-Image-2.1-Turbo-Heretic-mlx-bf16`: standard FP32 VAE, default.
- `Qwen-Image-2.1-Turbo-HDR-mlx-bf16`: shared `qwen21-hdr-vae-mlx-fp32`.

Always use **8 steps, guidance 1, qwen_turbo scheduler, strength 1, KV cache ON, StepCache OFF**, runtime `--no-bake-lora`. Functional LoRAs must not reset steps to 25/40. Original Heretic/NoctQ bases, quantized exports and acceleration LoRAs are retired; never automatically restore historical choices.

## Select, run, verify

1. Determine task from references; image1 is the edit source. Read preservation/style/size requirements and check selected files against [inventory](reports/deployment-snapshot-2026-10-10.json).
2. Default to standard VAE/no adapter. Standard favors natural skin, neutral/product colors and faithful edits; consider HDR for requested stronger color/contrast/detail, not universal quality improvement.
3. Choose adapters using the selection table. Start with one; add a second only for a distinct compatible goal. Independent scale 1 is a local starting point, not an optimum.
4. Preserve every active declared trigger and image role through PE. Optional style phrases are not mandatory triggers. Do not add unsolicited content from samples or adapter names.
5. State VAE, adapters/scales, reference order, dimensions and fixed Turbo recipe briefly, then execute [usage recipes](references/usage.md). Always pass the local model path. Ask only for missing essential input or material ambiguity.
6. Verify actual parameters and output; deliver files, seed, dimensions and elapsed time. Loader compatibility is not image-quality proof. Compare matched inputs and starting temperatures/cooling; separate loading/denoising/decoding/wall time.

Use image1 for canvas; see usage for reference budgets and CLI dimension caveats.

## UI and maintenance

WebUI has two VAE choices and two independent functional LoRA slots. Skill rules guide the agent, not automatic UI intent classification.

See [technical reference](references/technical-reference.md) for current PE model, unverified vision, old UI hints and runtime limits.

Maintain the canonical package and existing core mirror. Read project rules before code edits. No implied deletion/upload/publish/push authority. [Skill IR](references/skill-contract.md), [routing cases](evals/trigger_cases.json), [audit](reports/sync-2026-10-10.md).
