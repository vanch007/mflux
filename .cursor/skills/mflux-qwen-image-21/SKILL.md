---
name: mflux-qwen-image-21
description: "Run and maintain the local Qwen-Image-2.1 deployment on Apple Silicon via mflux/MLX and mflux-paint. Select Heretic or NoctQ with standard/HDR VAE, configure acceleration and effect LoRAs, and generate or edit images with reference conditioning, RGBA, masks and prompt enhancement."
metadata:
  last_verified: "2026-10-06"
---

# Qwen-Image-2.1 via mflux (MLX)

Use this skill for the local Qwen-Image-2.1 deployment, its CLI/Python usage, WebUI configuration and controlled image comparisons. Read the project rules before changing either repository. Other image-model families and remote image-generation services are outside this skill's scope.

## Local environment and model selection

- Core: `/Users/vanch/mflux-pr736`; WebUI: `/Users/vanch/mflux-paint`.
- Run core commands from the core directory with `uv run --no-sync`.
- Verified runtime: Python 3.13.5, mflux 0.20.0, MLX and mlx-metal 0.32.2 (2026-10-03).
- Canonical package: `/Users/vanch/.skills-manager/skills/mflux-qwen-image-21`; `/Users/vanch/.codex/skills/mflux-qwen-image-21` is its symlink. The package under the core repo's `.cursor/skills/` is a synchronized portable mirror.

| WebUI model option | Local model directory |
| --- | --- |
| Qwen-Image-2.1 Heretic · bf16 | `/Users/vanch/models/mlx/Qwen-Image-2.1-Heretic-mlx-bf16` |
| Qwen-Image-2.1 Heretic (HDR VAE) · bf16 | `/Users/vanch/models/mlx/Qwen-Image-2.1-Heretic-HDR-mlx-bf16` |
| Qwen-Image-2.1 Turbo (官方 8步) · bf16 | `/Users/vanch/models/mlx/Qwen-Image-2.1-Turbo-Heretic-mlx-bf16` |
| Qwen-Image-2.1 Turbo (官方 8步 · HDR VAE) · bf16 | `/Users/vanch/models/mlx/Qwen-Image-2.1-Turbo-HDR-mlx-bf16` |
| Qwen-Image-2.1 NoctQ V4 · bf16 | `/Users/vanch/models/mlx/Qwen-Image-2.1-NoctQ-bf16` |
| Qwen-Image-2.1 NoctQ V4 (HDR VAE) · bf16 | `/Users/vanch/models/mlx/Qwen-Image-2.1-NoctQ-HDR-mlx-bf16` |

Heretic replaces the text/vision encoder while retaining the original diffusion transformer. NoctQ V4 supplies a fine-tuned diffusion transformer and shares the deployed Heretic encoder/processor. Both use the original FP32 VAE unless an HDR combination is selected. HDR combinations share `/Users/vanch/models/mlx/qwen21-hdr-vae-mlx-fp32` through symlinks; they do not duplicate the base weights.

The local 8-bit/4-bit model exports were removed. Core quantization and remote aliases still exist; do not describe those aliases as downloaded models or recreate exports without a task requiring them.

Read [deployment and technical reference](references/technical-reference.md) for source revisions, component metadata, shared paths, HDR tradeoffs and current limitations.

## Acceleration and effect adapters

| Acceleration option | CLI preset | Steps | Scheduler | Default scale / guidance |
| --- | --- | --- | --- | --- |
| Viggle v0.3 r128 (local default) | `viggle-turbo` | 6; local 8-step option | `viggle_turbo` | 1.0 / 1.0 |
| Viggle v0.3 r256 | `viggle-turbo` + explicit r256 file | 6; local 8-step option | `viggle_turbo` | 1.0 / 1.0 |
| Turbo8 (chriswritescode) | `turbo8` | 8 | `linear` | 1.0 / 1.0 |
| No acceleration LoRA | omit preset and acceleration adapter | 40 default | `linear` | n/a / 1.0 |

Nine effect LoRAs are downloaded: **Radiance Chrome Voluptuous v1.0**, **Natural Exposure**, **RunningHub perspective blend v2.0 (pysj666)**, **SimpleTuner photo-aesthetics v2**, **f23gg NSFW**, **BFS Head V1.1**, **BFS Head V1.1 Alternative**, **BFS Body V1.0**, and **Outfit Swap Consistency (step 1000)**. Their default local scale is 1.0, independently adjustable from the acceleration scale; effect adapters are disabled by default. Their files, cache revisions, trigger evidence and author recipes are in [LoRA catalog](references/lora-catalog.md). Read it when selecting, stacking, updating or benchmarking adapters.

- Use **one acceleration adapter** at a time, optionally with effect adapters. Never stack Viggle r128, r256 and Turbo8 together.
- Keep adapters at runtime: `--no-bake-lora`. The Turbo presets reject bf16 baking; export adapter-free bases.
- When supplying explicit `--lora` files alongside a preset, include its acceleration file too. Preset auto-selection occurs only when no LoRA list is supplied.
- Scale 0.85 remains available as an experiment; default is 1.0. Do not claim a universal quality optimum from one prompt.
- The Viggle repository contains **both LoRAs and premerged GGUF/INT8/FP8 transformers**. Local deployment uses the v0.3 r128/r256 LoRAs, not those premerged checkpoints. v0.2.1/r64 are historical, not current local choices.
- Viggle's current author card describes a **9-step hybrid base tail**. Local presets support only 6/8 steps; the existing 8-step schedule is not the new 9-step algorithm.
- Preserve requested image content. Distinguish an author trigger from a locally added style phrase; never assume an NSFW adapter requires adding nudity to a clothed prompt.

## Generate and edit

Use `mflux-generate-qwen-2.1-edit` as the primary entry point for both tasks. Pass an explicit local model directory to use these deployed weights: omitting `--model` selects `Qwen/Qwen-Image-2.1`, not the local Heretic variant.

```sh
cd /Users/vanch/mflux-pr736
uv run --no-sync mflux-generate-qwen-2.1-edit \
  --model /Users/vanch/models/mlx/Qwen-Image-2.1-Heretic-mlx-bf16 \
  --lora-preset viggle-turbo --steps 6 --guidance 1 \
  --prompt "A red ceramic teapot on a wooden table" \
  --width 1024 --height 1024 --seed 42 \
  --mlx-cache-limit-gb 8 --no-bake-lora --metadata --output teapot.png
```

Add `--image-paths source.png` for editing, or up to 10 ordered reference paths for multi-reference editing. Image1 is the edit source and supplies the default output aspect ratio; subsequent references only supply conditioning. The same model handles text-only generation when no references are supplied. Explicit dimensions must be multiples of 32. Reference encoding stays at `--reference-resolution 1024`, independently of output dimensions.

KV cache and deterministic independent edit noise are ON by default. The latter hashes the requested seed and resized RGBA reference pixels/order to avoid reusing a generated reference's initial noise. Old metadata may replay its original shared resolution/noise policy; see [usage](references/usage.md).

Read [usage](references/usage.md) for r256/Turbo8, effect stacking, HDR/NoctQ, Python, resolution presets, masks, True CFG, RGBA, img2img, quantization and model saving.

## WebUI and prompt enhancement

Start `/Users/vanch/mflux-paint/start_webui.sh`; open `http://127.0.0.1:7866/`. Model precision/base/VAE selection is separate from acceleration and effect adapter choices. Each of the four Qwen entries uses `auto` mode: references present means editing; no references means text-to-image. Switching acceleration resets its steps/guidance/scale defaults; explicit scale edits are preserved for the run.

The current browser UI selects **one effect LoRA** in addition to its acceleration LoRA. The backend `extra_loras` list and core repeated `--lora` arguments accept multiple effect adapters with independent scales; do not claim the browser exposes a multi-effect selector.

Radiance Chrome selects independent author strength 1 and preserves acceleration; no-acceleration selection uses a local 25-step starting point based on sample filenames. All 448 tensors load directly, alone or stacked with r128. Its automatic `Reversal Film Gravure, analog film photography,` prefix is an editable sample style description: no mandatory trigger is declared. Preserve requested clothing/content and do not auto-insert sample white-balance defects. Its Euler/simple author recipe is not identical to mflux's timestep grid. See [integration/comparison evidence](reports/radiance-chrome-2026-10-06.md).

RunningHub lighting blend inserts the declared **`pysj666`** trigger at the beginning, including after WebUI prompt enhancement. It preserves the current acceleration recipe and initializes only its own effect strength to 1. Its card describes consistent object/background illumination, specular highlights and contact shadows but publishes no optimal inference scale/steps or required reference count. All 448 tensors load in MLX alone and with Viggle r128; generated-image quality is untested for this addition. See [integration evidence](reports/runninghub-lighting-2026-10-05.md).

BFS needs at least two ordered references: `<image1>` is the original scene and `<image2>` is the source face/person. Selection inserts the author's `head_swap:` or `body_swap:` prompt, defaults to effect strength 1 and no-acceleration 40 steps, and disables StepCache for that baseline. The browser and HTTP run handler reject fewer than two references. Head includes hair/identity; Body includes clothing/body proportions. Accelerated BFS combinations are selectable but do not claim their quality is validated. The body author's separate scene/person pixel budgets are not implemented; references currently share the 1024² encoding policy.

Outfit Swap needs image1 = the person to dress, image2 = outfit (product shot, flat lay or a wearer). It has **no trigger word**. The browser adds an editable author-based swap template, uses the pinned step-1000 file, and selects no acceleration, 25 steps, guidance/scale 1 and StepCache off. Specify the garment briefly. If compositions mix, try a product photo or crop to the garment: a same-noise local crop trial passed qualitatively after the full-scene pair failed. See [integration evidence](reports/outfit-swap-2026-10-05.md). Reference labels and first-image mask checks reflect these roles. Acceleration remains selectable independently; author-tested Viggle 8-step behavior is not proof of quality for all local combinations.

WebUI Qwen runs cap the MLX allocator cache at 8 GB. This is not an 8 GB total-memory limit. Output sizing always follows image1, even while another reference is active in the canvas. Auto preserves image1 dimensions up to 1536² area with 32-pixel alignment; larger sources are scaled to that budget. Whole-image edits preserve generated dimensions; masked edits composite at image1's canvas dimensions. Up to 10 original references are sent without a canvas re-encode. See [first-reference canvas evidence](reports/reference-canvas-2026-10-05.md).

The WebUI PE uses local oMLX at port 8000 with `Qwen3.6-35B-A3B-Uncensored-Genesis-Hermes-V13-dequantized-oQ6e-fp16-mtp`. Edit PE constructs image inputs for up to 10 references; it is not text-only. Rules live at `mflux-paint/prompts/system_prompt_{t2i,edit}.txt` and include the local photography-lighting additions. These are adapted official rules plus local changes, not the official PE model weights or guaranteed official-equivalent behavior. Inspect the actual outgoing request/logs when diagnosing image understanding.

Built-in CLI `--auto-mask`, `--enhance-prompt` and `--verify` use the deployed Qwen3-VL generation head, restored on 2026-10-01. They are a separate path from WebUI oMLX PE. Preserve `lm_head.weight`, final norm, lazy loading and the Heretic RoPE compatibility patch.

## Quality and performance evidence

Measure with identical prompts, seed, dimensions, references, precision, VAE and adapter scales. Match starting CPU/GPU temperatures, cooling/fan settings and background workload; separate loading, generation and decoding. Identify whether the run is a cold process, warm model or cold file cache.

Core `--use-step-cache` is OFF by default; WebUI's no-acceleration mode enables it at threshold 0.12. This is FBCache-style first-block residual reuse, not guaranteed identical pixels or a universally optimal threshold. Use `--no-use-step-cache` for an uncached 40-step baseline; record the setting in comparisons.

HDR VAE intentionally changes saturation, contrast and detail; it is a choice, not a quality-neutral default. Existing historical v0.2.1 results must not be relabeled as v0.3 tests. Micro-faces/crowds, dense text, complex edits, unusual alpha and NoctQ-plus-adapter combinations remain prompt-dependent.

See [technical evidence](references/technical-reference.md), the [2026-10-03 synchronization audit](reports/sync-2026-10-03.md) and [skill contract](references/skill-contract.md) for evidence boundaries and maintenance responsibilities. Skill use does not authorize deleting weights, uploading models, publishing repositories or pushing code.
