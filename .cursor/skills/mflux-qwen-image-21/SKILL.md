---
name: mflux-qwen-image-21
description: "Run Qwen-Image-2.1 text-to-image, image editing, and img2img locally on Apple Silicon via mflux (MLX). Covers optional Viggle Turbo acceleration LoRA, CLI/Python usage, quantization, True CFG, prefix KV caching, multi-reference editing, RGBA output, and model saving."
---

# Qwen-Image-2.1 via mflux (MLX)

Local deployment of Qwen-Image-2.1 through mflux, running natively on Apple Silicon with MLX. Based on [PR #741](https://github.com/mflux-community/mflux/pull/741) which adds prefix KV caching, multi-reference editing, and RGBA output on top of [PR #736](https://github.com/mflux-community/mflux/pull/736).

## Prerequisites

- **Project location**: `/Users/vanch/mflux-pr736`
- **Virtual env**: `/Users/vanch/mflux-pr736/.venv` (Python 3.13.5, MLX/Metal 0.32.2 (runtime and lock verified 2026-09-27))
- **Deployed weights**: `/Users/vanch/models/mlx/Qwen-Image-2.1-Heretic-mlx-bf16` (~33 GB bf16)
  - Transformer: 14.2 GB, Text encoder (Qwen3-VL Heretic): 17.5 GB, VAE: 1.4 GB
- **Text encoder**: Replaced with `pottokao/Qwen-Image-2.1-Text-Encoder-Heretic`; the local RoPE-config compatibility patch must be preserved.
- **Generation head**: The deployed export includes the original Heretic BF16 `lm_head.weight` and final norm (restored 2026-10-01). Built-in `--auto-mask`, `--enhance-prompt` and `--verify` passed real calls; ordinary conditioning is bit-identical. Older headless BF16 exports can use `uv run python scripts/restore_qwen21_heretic_head.py --model-path PATH`. Evidence: `docs/heretic-generation-head-2026-10-01.md`. The head stays lazy until an auxiliary text-generation call; WebUI oMLX PE remains separately configured.
- **Memory**: Peak ~46 GB at bf16. 64 GB machine recommended. Use `-q 8` for tighter memory.
- **All commands** must run from the project directory:

```sh
cd /Users/vanch/mflux-pr736
```

## Image Editing (edit) — Primary Use

Takes one or more reference images plus a text instruction. **KV cache is ON by default**, caching text and reference-image prefix attention across denoising steps. Measure the gain for the specific workload; reduced-precision cached and uncached outputs need not be identical.

**Independent edit noise is ON by default.** Reusing the seed and dimensions that generated a reference can cause severe oversharpening and background noise. The local edit pipeline now derives a deterministic noise seed from the requested seed plus the resized RGBA reference pixels (dimensions and order included). Same seed and references remain reproducible, chained edits receive new noise, and text-only generation is unchanged. This applies to the base model and both Turbo adapters, including WebUI calls.

### CLI

```sh
uv run --no-sync mflux-generate-qwen-2.1-edit \
  --image-paths /path/to/source.png \
  --prompt "Replace the Chinese text with Vietnamese text" \
  --steps 40 \
  --seed 42 \
  --output edited.png
```

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--image-paths` | none | One or more reference images (up to 10) |
| `--steps` | 40 | Inference steps |
| `--seed` | random | Reproducibility seed |
| `--width` / `--height` | auto | Output size (32x multiples). Auto-derives from last reference image |
| `--output-resolution` | 1024 | Pixel-area budget for automatic output dimensions |
| `--reference-resolution` | 1024 | Independent reference encoding budget; keep 1024 for Turbo training alignment |
| `--use-kv-cache` | ON | Prefix KV caching. Disable with `--no-use-kv-cache` |
| `--independent-edit-noise` | ON | Prevent reuse of a generated reference's initial noise; disable only for legacy reproduction/parity |
| `-q 8` or `-q 4` | None (bf16) | Quantize eligible transformer + text/vision layers; VAE stays FP32 |
| `--guidance` | 1.0 | CFG scale. >1.0 with `--negative-prompt` for True CFG |
| `--output` | image.png | Output path (must be .png, .webp, or .tiff for RGBA) |
| `--metadata` | off | Export JSON metadata sidecar |
| `--vae-tiling` | off | Tiled VAE decode to reduce peak memory |

**Multi-reference editing** (up to 10 images):

```sh
uv run --no-sync mflux-generate-qwen-2.1-edit \
  --image-paths subject.png scene.png \
  --prompt "Place the subject from image 1 in the setting of image 2." \
  --steps 40 --seed 42 \
  --output-resolution 512
```

**Transparent output (RGBA)**:

```sh
uv run --no-sync mflux-generate-qwen-2.1-edit \
  --prompt "This is an RGBA image with transparency. A cute sticker. The background is transparent." \
  --seed 42 \
  --output sticker.png
```

### Python API

```python
from mflux.models.qwen21.reference import QwenImage21Edit

model = QwenImage21Edit(quantize=None)

# Edit with KV cache (default)
image = model.generate_image(
    seed=42,
    prompt="Replace the Chinese text with Vietnamese text",
    image_paths=["/path/to/source.png"],
    num_inference_steps=40,
    width=512,
    height=736,
    guidance=1.0,
    use_kv_cache=True,  # default
)
image.save("edited.png")
```

Python exposes `independent_edit_noise=True`. Metadata records the requested `seed`, effective `noise_seed`, and this policy. Older metadata files lacking the policy replay with their original direct-seed behavior; add `--independent-edit-noise` to apply the fix when replaying an old sidecar. New sidecars restore the saved policy.

## Optional Viggle v0.3 acceleration LoRA

Only the v0.3 **r128** and **r256** adapters are supported by the Turbo preset. Default is r128, 6 steps, scale 1.0, guidance 1, no negative prompt, runtime LoRA, KV cache ON. The base model remains 40 steps. Verified against the official card on 2026-09-26.

```sh
uv run --no-sync mflux-generate-qwen-2.1-edit \
  --lora-preset viggle-turbo --steps 6 --guidance 1 \
  --prompt "A red ceramic teapot on a wooden table" \
  --width 1024 --height 1024 --seed 42 --metadata --output turbo.png
```

The preset automatically selects `viggle_turbo`; an explicit incompatible scheduler is rejected. Use `--steps 8` for the official dense-text schedule. Four steps and arbitrary step counts are no longer offered in this local preset.

- Six raw nodes: `[1, 0.9375, 0.875, 0.75, 0.5, 0.25]`.
- Eight raw nodes: `[1, 0.9375, 0.875, 0.75, 0.625, 0.5, 0.25, 0.125]`.
- Apply resolution-dependent shifting once; `sigma_shift_terminal=None`. Never substitute equally spaced 6/8-step nodes or bake into bf16.
- Both adapters are pinned to revision `009a44a895ef85f7e643c80fdca9543795248867` in `Viggle/Qwen-Image-2.1-viggle-turbo`. r128: 679,604,800 bytes; r256: 1,359,147,904 bytes. Base weights are reused.
- To select r256, add `--lora Viggle/Qwen-Image-2.1-viggle-turbo:Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r256.safetensors 1.0`.
- Add `--image-paths source.png` for editing. References use `--reference-resolution 1024` independently of output dimensions and `--output-resolution`. Prefer dimensions aligned to 32. Strength-based img2img is incompatible with the preset.
- Python: `QwenImage21Edit(lora_preset="viggle-turbo")`; omitted steps and scheduler resolve to 6 and `viggle_turbo`.

Scale 0.85 is supported as an experimental comparison setting in CLI/API/WebUI; it is not the official distilled recipe. CFG above 1 is allowed in CLI/API with a warning, but not promoted as an edit-quality improvement. Old metadata without `reference_resolution` replays the historical shared output/reference budget; pass `--reference-resolution 1024` explicitly to apply the updated default.

Other Qwen 2.1 adapters use `--lora PATH SCALE` or Python `lora_paths` / `lora_scales`. The local loader supports Diffusers, PEFT (including `.default`), common ComfyUI/Kohya names and fused `img_mlp.gate_up` tensors. Older Qwen Image 20B adapters are incompatible. Export base checkpoints without adapters, then load LoRAs at runtime.

Distillation changes quality; 6/8 steps cannot guarantee the 40-step model's complex-edit or text fidelity. Keep Heretic as requested. The local oMLX prompt rewriter is an optional approximation, not the official PE-T2I/PE-I2I models; its edit rewrite does not see the reference image.

Current configuration audit: `/Users/vanch/mflux-pr736/outputs/config-audit-2026-09-26/`. Historical outputs remain in `outputs/qwen21-turbo-2026-09-23/`; their settings and timings are historical evidence, not the current recipe.

Further edit-quality evidence (reference/output budget separation, original-input UI transport and native output): `/Users/vanch/mflux-pr736/outputs/qwen21-edit-quality-2026-09-27/`. These are one-prompt, single-seed comparisons; no general perceptual quality win is established.

The audit's 1024² selective-color defect was isolated on 2026-09-27 to source/edit noise reuse: the pinned official Diffusers implementation reproduced it, and changing only seed 42 to 43 removed it. The default independent-noise fix passed real WebUI tests for both v0.2.1 adapters with the original seed 42, 1024×1024 and 6 steps. Evidence and embedded-image comparison: `/Users/vanch/mflux-pr736/outputs/qwen21-edit-fix-2026-09-27/`. This verifies that failure case, not universal edit fidelity; unrelated historical panda samples have not been revalidated.

## WebUI (Browser & Canvas Interface)

A graphical WebUI with an interactive drawing canvas, image upload, inpaint/edit, and model picker is available via `/Users/vanch/mflux-paint`:

```sh
/Users/vanch/mflux-paint/start_webui.sh
```

Open `http://127.0.0.1:7866` in Safari / Chrome.

Integrated models in the top-left dropdown:
- `Qwen-Image-2.1 · Turbo Edit (r128 / r256)`: 6 steps by default, optional 8, guidance 1.0 and default LoRA strength 1.0 (editable, e.g. 0.85), KV cache ON.
- `Qwen-Image-2.1 · Quality Edit (40 steps)`: Full 40-step reference editing.
- `Qwen-Image-2.1 · Turbo T2I (r128 / r256)`: same Turbo recipe for text-only generation.
- `Qwen-Image-2.1 · Quality Text-to-Image`: Full 40-step text-to-image.


Qwen edit Auto preserves source dimensions up to a 1536² area cap with 32-pixel alignment. Explicit resolution options set area budgets. Whole-image edits export native generated pixels without scaling back to the input; masked edits keep canvas dimensions for compositing. The UI passes original image data without a canvas roundtrip, preventing pixel changes from altering the deterministic edit-noise seed. The reference encoding budget remains 1024². Text-to-image exposes explicit width/height. Quantization remains off by default.

## Text-to-Image (txt2img)

For text-only generation without reference images, both the edit entry point and the original txt2img work:

```sh
# Using the edit entry point (has KV cache)
uv run --no-sync mflux-generate-qwen-2.1-edit \
  --prompt "A serene mountain lake at golden hour" \
  --steps 40 --seed 42 \
  --width 1024 --height 1024 \
  --output output.png

# Using the original txt2img entry point (no KV cache)
uv run --no-sync mflux-generate-qwen-2.1 \
  --prompt "A serene mountain lake at golden hour" \
  --steps 40 --seed 42 \
  --output output.png
```

### Python API (txt2img)

```python
from mflux.models.common.config import ModelConfig
from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21

model = QwenImage21(
    quantize=None,
    model_config=ModelConfig.qwen_image_21(),
)
image = model.generate_image(
    seed=42,
    prompt="A serene mountain lake at golden hour",
    num_inference_steps=40,
    width=1024,
    height=1024,
)
image.save("output.png")
```

## Image-to-Image (img2img)

Uses the txt2img entry point with source image and strength:

```sh
uv run --no-sync mflux-generate-qwen-2.1 \
  --prompt "Oil painting version" \
  --image-path /path/to/source.jpg \
  --image-strength 0.7 \
  --steps 40 --seed 42
```

## True CFG (optional)

```sh
uv run --no-sync mflux-generate-qwen-2.1-edit \
  --prompt "A photorealistic portrait" \
  --negative-prompt "blurry, low quality, distorted" \
  --guidance 2.5 \
  --steps 40 --seed 42
```

## Quantization

In the primary edit pipeline, `-q` quantizes eligible transformer and text/vision encoder layers; the VAE remains FP32 across the bf16, 8-bit and 4-bit model variants. This pipeline also handles text-only generation.

The user requested removal of the VAE quantization experiment on 2026-09-27. Its CLI/API/config support and both experimental VAE exports have been deleted; use the original FP32 VAE. Historical comparison images and measurements are preserved at `/Users/vanch/mflux-pr736/outputs/qwen21-vae-quantization-2026-09-27/`, with the embedded report `/Users/vanch/Desktop/Qwen21_VAE量化冷机对比_1600x960.html`. Those experimental results do not describe the current deployment.

```sh
uv run --no-sync mflux-generate-qwen-2.1-edit --prompt "..." -q 8 --steps 40 --seed 42
```

Use the existing dense weights unless memory requires quantization; speed and quality depend on the workload. Do not retain additional model copies without need.

## Model Saving (edit variant)

```python
from mflux.models.qwen21.reference import QwenImage21Edit

model = QwenImage21Edit(quantize=8)
model.save_model("./qwen21-8bit")
```

Note: `mflux-save --model qwen-image-2.1` exports the text-only implementation without the vision tower. Use `QwenImage21Edit.save_model()` for the full reference-capable model.

## Performance evidence

Use controlled measurements with the same prompt, seed, dimensions, reference budget, precision and KV setting. Record temperature and separate model loading from image generation. Earlier approximate PR #736/#741 timings do not establish an isolated KV-cache speedup.

KV cache reuses the text prefix attention across denoising steps. The first step computes and caches; subsequent steps reuse. This works because `causal_condition` makes text activations step-independent.

## Architecture Overview

- **Transformer**: 7.1B single-stream block-causal DiT
- **Text encoder**: Qwen3-VL Heretic (~17.5 GB bf16; eligible layers can be quantized by the edit pipeline)
- **Vision tower**: Qwen3-VL vision model (for reference image understanding in edit mode)
- **VAE**: 64-channel causal VAE, 16x spatial compression, RGBA output
- **Prefix KV cache**: Text and reference image prefix cached across denoising steps
- Neural inference runs through MLX; image preprocessing, tokenization and file I/O also use CPU libraries.

## Current Limitations

- LoRA inference is supported locally; LoRA training, rsLoRA/DoRA, and nontrivial PEFT alpha/rank metadata patterns are not.
- PID decoding not yet supported
- No video generation
- Edit fidelity varies with prompt, reference and resolution; earlier failed examples do not establish a general size ranking.
- This is a local reference-edit branch, not a wholesale checkout of upstream main. Main and open-PR support are tracked separately in the audit report.
