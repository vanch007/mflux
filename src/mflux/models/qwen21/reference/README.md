# Qwen-Image-2.1 reference editing

Native MLX inference for [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1): text-to-image, editing with up to ten reference images, and RGBA output. This is a separate architecture from the earlier [Qwen Image models](../../qwen/README.md), with a 7B diffusion transformer, a Qwen3-VL text/vision encoder, and a 64-channel RGBA VAE.

The `mflux-generate-qwen-2.1-edit` entry point complements the existing `mflux-generate-qwen-2.1` text-to-image and strength-based img2img command. Both use the same model registry entry, Transformer blocks, VAE encoder/decoder, language decoder, component-loading mechanism, and LoRA mappings. This variant adds the vision tower and interleaved reference conditioning. Production implementations live under `qwen21/model`, `weights`, `latent_creator`, and `variants`; `qwen21/reference` retains compatibility imports, these validation notes, and the upstream license. Existing generation behavior and saved checkpoints keep their original entry point.

The shared components preserve each variant's numerical contract. Editing keeps RGBA single-frame VAE output with clipping, its normalization rounding, and pre-final-norm language features with DeepStack injection. Text generation keeps RGB output, its VAE attention arithmetic, padding mask, and final language RMSNorm, without loading a vision tower. Separate model objects retain separate parameters and request state.

The editing pipeline includes prefix KV caching, LoRA loading, quantization, local checkpoint export/reload, metadata replay, and the common generation callbacks. Training is not implemented for this variant.

## Acceleration LoRA (local integration)

The optional [Viggle Turbo v0.3](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo) preset downloads only its 680 MB rank-128 adapter (or the 1.36 GB rank-256 alternative), pinned to revision `009a44a895ef85f7e643c80fdca9543795248867`. It reuses the base model, including a locally substituted text encoder.

```sh
mflux-generate-qwen-2.1-edit \
  --lora-preset viggle-turbo --steps 6 --guidance 1 \
  --prompt "A red ceramic teapot on a wooden table, studio photograph" \
  --width 1024 --height 1024 --seed 42 --metadata --output turbo.png
```

Add `--image-paths source.png` for reference editing. The same preset is available for text-to-image on `mflux-generate-qwen-2.1`; use the edit entry point for KV caching. Turbo rejects legacy strength-based img2img because it skips part of the required distilled schedule. The preset disables terminal sigma stretching and defaults to six steps, guidance 1, adapter scale 1, and runtime adapters (`--no-bake-lora`, the default for both Qwen 2.1 commands). Omit the preset for normal 40-step generation. Original base defaults are unchanged.

```python
model = QwenImage21Edit(lora_preset="viggle-turbo")
image = model.generate_image(seed=42, prompt="A ceramic teapot", width=1024, height=1024)
image.save("turbo.png")  # preset defaults to six steps in the Python API
```

The scheduler is selected automatically as `viggle_turbo`. Its six raw nodes are `[1, 0.9375, 0.875, 0.75, 0.5, 0.25]`; the resolution-dependent time shift is applied once. `--steps 8` selects the official dense-text nodes `[1, 0.9375, 0.875, 0.75, 0.625, 0.5, 0.25, 0.125]`. Other step counts and an explicit `linear` scheduler are rejected. Scale 1 is the official recipe; custom scales such as 0.85 are available for local comparisons. Turbo CFG above 1 remains experimental and warns because it adds a second pass and can reduce quality. To choose r256, add `--lora Viggle/Qwen-Image-2.1-viggle-turbo:Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r256.safetensors 1.0`. Both adapters remain unquantized and unmerged by default.

For other compatible 2.1 adapters, use the shared `--lora PATH SCALE` option or the Python `lora_paths` / `lora_scales` arguments. Diffusers, PEFT, and common ComfyUI/Kohya naming conventions are mapped; unsupported keys, incomplete pairs, and incompatible dimensions are rejected before any adapter is applied. Legacy Qwen Image 20B adapters are incompatible. Nontrivial PEFT alpha/rank patterns, rsLoRA, and DoRA are not supported. Save base checkpoints without adapters and load adapters separately; exporting a LoRA-loaded model is intentionally rejected to prevent accidental baking or loss of the Turbo scheduler.

This is a lossy distillation preset, not a quality-preserving replacement. Its author reports weaker complex edits and rendered text; training used at most three references; RGBA/mask behavior has limited validation. Local tests must be assessed separately, especially when using a substituted text encoder or low resolutions.

## Generate an image

```sh
mflux-generate-qwen-2.1-edit \
  --prompt "A red panda reading a book under a cherry tree, watercolor illustration" \
  --seed 42 \
  --quantize 8 \
  --output qwen21.png
```

Defaults are **40 steps**, **guidance 1.0**, **1024×1024** for text-to-image, and **KV caching enabled**. The aliases `qwen-image-2.1`, `qwen-2.1`, and `qwen-image-21` all select this model; `--model` can be omitted for this command.

The original checkpoint downloads about 33.1 GB of weights, even when quantizing at load time. `--quantize` applies to eligible transformer and text/vision encoder layers. The VAE remains fp32 for bf16, 8-bit and 4-bit model variants. Peak memory depends on resolution, reference images, and quantization. Use `--vae-tiling` to reduce decode memory or `--low-ram` for the common memory-saving callbacks.

The experimental VAE quantization implementation and local quantized VAE exports were removed at the user's request on 2026-09-27. The original FP32 VAE remains in use. Historical images and measurements are retained in `outputs/qwen21-vae-quantization-2026-09-27/`.

Explicit width and height must be multiples of 32. The upstream model-card examples use 2048×2048; request that size explicitly:

```sh
mflux-generate-qwen-2.1-edit \
  --prompt "A detailed illustrated botanical poster with the title SPRING" \
  --width 2048 --height 2048 \
  --seed 42 --quantize 8 \
  --vae-tiling \
  --output poster.png
```

## Edit with reference images

![Two-reference clothing edit](../../../assets/qwen_image_21_reference_edit.png)

The 896×1152 example uses the two inputs and prompt from the [official ComfyUI editing workflow](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_qwen_image_2_1_image_edit.json), an 8-bit model, 25 steps, and guidance 1. The person retains the pose and background while wearing the second reference's denim shirt.

```sh
mflux-generate-qwen-2.1-edit \
  --image-paths portrait.png \
  --prompt "Change the jacket to dark green. Preserve the person's face and pose." \
  --seed 42 --quantize 8 \
  --output-resolution 512 \
  --output edited.png
```

```sh
mflux-generate-qwen-2.1-edit \
  --image-paths subject.png scene.png \
  --prompt "Place the subject from image 1 in the setting of image 2." \
  --seed 42 --quantize 8 \
  --output-resolution 512 \
  --output combined.png
```

The examples request a 512-pixel output area budget, with reference encoding independently kept at the default 1024². For historical reference-budget tests, pass `--reference-resolution 512` explicitly. See [validation results](VALIDATION.md) for historical samples and their limitations.

Reference order is significant. The first reference is the edit source and determines the output aspect ratio when width and height are omitted. Additional references supply conditioning without choosing the output canvas. `--output-resolution` defaults to 1024 and derives automatic output dimensions from that source ratio. `--reference-resolution` independently defaults to 1024, matching Viggle training even for 1536² output. `--width` and `--height` override output dimensions without changing reference encoding. Older metadata without `reference_resolution` replays its original shared budget; override `--reference-resolution 1024` to apply the new behavior. Reference images must remain within the checkpoint processor's pixel-area range after resizing; the default budget handles ordinary aspect ratios.

RGBA references retain alpha in the VAE. The vision encoder reads a separate copy composited over white, matching the upstream pipeline.

### Reproducible edit noise

Editing a generated image with its original seed and output dimensions can reuse the exact noise that created it. A local controlled test reproduced severe background texture and oversharpening in both MLX and Diffusers; changing only the edit seed removed the artifact.

The edit entry point now enables `--independent-edit-noise` by default: it derives the noise seed from the requested seed and the resized RGBA reference pixels, including their dimensions and order. The same seed and references remain reproducible; text-only generation keeps its existing noise. This applies to both the base model and Turbo adapters, including WebUI calls, without additional denoising steps.

Python exposes `independent_edit_noise=True`. Use `--no-independent-edit-noise` only for legacy reproduction or controlled reference comparisons. Metadata records `seed`, `noise_seed`, and `independent_edit_noise`. New sidecars restore the policy; older sidecars without the field replay the original direct-seed behavior. Add `--independent-edit-noise` explicitly when applying the fix to an old sidecar.

## Local edits, strength and self-checks

These options act on the first reference image. They are MFLUX additions, not part of the upstream pipeline.

| Option | Effect |
| --- | --- |
| `--mask-image mask.png` | Inpaint: white repaints, black keeps the source. Unmasked latents follow the source's own noise trajectory at every step, and a final pixel composite keeps them exact. |
| `--auto-mask "the red shirt"` | Asks the text encoder's Qwen3-VL to locate the object and turns its box into the mask. Ignored with `--mask-image`. |
| `--strength 0.6` | Skips the first `1 - strength` of the schedule and starts from the source noised to that point, for subtler edits. The default `1.0` starts from pure noise. |
| `--enhance-prompt` | Rewrites a terse instruction into a detailed prompt first, following the official prompt-rewrite recipe. An unparseable reply keeps the original. |
| `--verify` / `--verify-retries N` | Asks the Qwen3-VL whether the edit applied and the rest is unchanged, and regenerates with the next seed up to `N` times on a failed check. |
| `--use-step-cache` | Skips transformer blocks 1..N on steps whose first-block output barely changed (threshold `--step-cache-threshold`, default 0.12). Faster, and slightly different output. Needs `--use-kv-cache`. Runs under 10 steps, like the 6-step `viggle_turbo`, skip nothing. |

```sh
mflux-generate-qwen-2.1-edit \
  --image-paths portrait.png \
  --auto-mask "the jacket" \
  --prompt "Change the jacket to dark green." \
  --seed 42 --quantize 8 \
  --output edited.png
```

Auto-mask, prompt rewriting and verification decode greedily with the text encoder's own untied `lm_head`, so they need no extra download. The head is about 1.2 GB in bf16 and 0.66 GB at q8. The loader keeps it lazy and reads it only when one of these options first uses it. `save_model` still writes the head, so exports keep these options. Checkpoints saved before this support existed do not have the head. They still generate. Auto-mask and verification raise a missing-head error; prompt rewriting logs a warning and retains the original instruction.

Qwen-Image-2.1 reference edit documentation


```sh
uv run python scripts/restore_qwen21_heretic_head.py --model-path /path/to/Qwen-Image-2.1-Heretic-mlx-bf16
```

The utility retrieves only the required byte ranges from a pinned Heretic revision, validates tensor shapes, backs up the weight index, and records source hashes beside the new shard. The local BF16 export was restored and its grounding, rewriting and verification were tested on 2026-10-01; see [the validation record](../../../../../docs/heretic-generation-head-2026-10-01.md). The WebUI continues to use its configured oMLX model for prompt rewriting. Acceleration presets require the complete distilled schedule, so `--strength` must remain `1.0` with Viggle/Turbo8. Step skipping is disabled by default.

## Transparent output

```sh
mflux-generate-qwen-2.1-edit \
  --prompt "This is an RGBA image with transparency. A cute red panda sticker. The image has alpha channel and the background is transparent." \
  --seed 42 --quantize 8 \
  --output sticker.png
```

All outputs have four channels; the prompt determines whether the generated alpha is transparent or opaque. The CLI accepts PNG, WebP, and TIFF to retain alpha. PNG is recommended for lossless output and embedded metadata.

## Quantized checkpoints and reproducibility

Export the complete reference-capable model from Python:

```python
from mflux.models.qwen21.reference import QwenImage21Edit

model = QwenImage21Edit(quantize=8)
model.save_model("./qwen21-8bit")
```

The existing `mflux-save --model qwen-image-2.1` exports the original text-only implementation and omits the vision tower. Those exports cannot be used by this editing entry point. Use the native Hugging Face checkpoint or an export produced by `QwenImage21Edit.save_model`.

Exports created before component consolidation remain loadable through their original entry points. The loader translates the old `modulation.1` and text VAE convolution/norm parameter names to the shared layouts without requantizing tensors. The generation and editing variants retain their respective quantization defaults and export contents. Local `mflux-save --model qwen-image-2.1-heretic-bf16` (or the 8-bit/4-bit registry variants) uses the complete editing saver.

```sh
mflux-generate-qwen-2.1-edit \
  --model ./qwen21-8bit \
  --prompt "A red panda reading a book" \
  --seed 42 --make-conf \
  --output saved-model.png

mflux-generate-qwen-2.1-edit \
  --model ./qwen21-8bit \
  --config-from-conf saved-model.metadata.json \
  --output replay.png
```

Use `--no-use-kv-cache` to recompute the prefix each step. Keep this setting fixed when comparing samples: cached and uncached execution can produce different images in reduced precision. Metadata records the cache flag and both resolution budgets, and explicit CLI values override restored metadata. MLX and PyTorch use different random-number generators, so matching seeds alone does not imply matching images.

Optional classifier-free guidance uses `--guidance` greater than 1 and `--negative-prompt`. The trained default is 1.0. See the [common CLI documentation](../../common/README.md) for prompt files, seeds, step previews, and other shared flags.

## Python

```python
from mflux.models.qwen21.reference import QwenImage21Edit

model = QwenImage21Edit(quantize=8)
image = model.generate_image(
    seed=42,
    prompt="A red panda reading a book, transparent background",
    width=1024,
    height=1024,
)
image.save("panda.png")

edited = model.generate_image(
    seed=42,
    prompt="Give the panda a blue scarf.",
    image_paths=["panda.png"],
    output_resolution=512,
)
edited.save("panda-scarf.png")
```

`model.rewrite_prompt(prompt, images)` returns the prompt that `enhance_prompt=True` would encode, so you can read or change it and then pass it as `prompt` with `enhance_prompt=False`. Where `enhance_prompt=True` falls back to the original instruction, `rewrite_prompt` raises. The rewrite sees every reference image. `images`, like `image_paths` in `generate_image`, takes paths or PIL images; the image metadata then records `<in-memory image>` for them.

## Reference and validation

The port targets checkpoint revision [`b3179ad`](https://huggingface.co/Qwen/Qwen-Image-2.1/tree/b3179ad355be050328e483a9dfdd9e60cd62adfa) and the Diffusers implementation at [`80c7ed2`](https://github.com/huggingface/diffusers/tree/80c7ed262aeffbeb43ef13ae04baeb9b84515a69/src/diffusers/pipelines/qwenimage21). Adapted model code retains its Apache-2.0 attribution; the upstream license is included in [LICENSE.diffusers](LICENSE.diffusers). Checkpoint use is governed by the model's [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/b3179ad355be050328e483a9dfdd9e60cd62adfa/LICENSE).

Fast tests cover the registry, metadata, RGBA serialization, latent layout, prefix masks, and cached decoding. Optional reference tests compare transformer, VAE, and scheduler behavior against Diffusers with identical weights and inputs. The small PyTorch reference models run on CPU so the tests do not depend on the CI runner's MPS support; MLX still uses Metal. Diffusers comparisons skip when that reference implementation is unavailable:

```sh
MFLUX_PRESERVE_TEST_OUTPUT=1 uv run \
  --with "diffusers @ git+https://github.com/huggingface/diffusers@80c7ed262aeffbeb43ef13ae04baeb9b84515a69" \
  python -m pytest tests/image_generation/test_qwen_image21_parity.py
```

Real checkpoint tests have produced 1024-pixel text-to-image and transparent PNG outputs, 512-pixel single- and two-reference edits, a successful 896×1152 two-reference clothing edit using the official ComfyUI workflow inputs, and pixel-identical output after 8-bit checkpoint export/reload. A 2048×2048 text-generation case passed a full 40-step visual check with 8-bit weights and tiled VAE decoding on 2026-09-30. The original 1024×1024 panda edits missed the requested changes in both MLX and the controlled Diffusers reference run; expanding the prompts did not resolve those samples. The existing text-only golden also remains failing locally, including with its introducing code; that investigation and the bounded 2048 result are recorded in [VALIDATION.md](VALIDATION.md).


### Turbo8 and independent adapter strengths

Use `--lora-preset turbo8` for `chriswritescode/Turbo8-LoRA-Qwen-Image-2.1`.
CLI and Python calls default to 8 steps, linear Euler sampling with the base
resolution shift, terminal stretching disabled, and runtime LoRA. Guidance 1
and scale 1 are the initial recipe; an explicit scale such as 0.85 is preserved.
Pass the matching adapter explicitly when customizing its scale:

```sh
uv run --no-sync mflux-generate-qwen-2.1-edit \
  --model /path/to/Qwen-Image-2.1-Heretic-mlx-bf16 \
  --lora-preset turbo8 \
  --lora chriswritescode/Turbo8-LoRA-Qwen-Image-2.1:turbo8_lora_step2500.safetensors 0.85 \
  --prompt "A ceramic teapot on a wooden table" --seed 42 --output turbo8.png
```

Append `--image-paths source.png` for editing. Standard Hugging Face cached
adapters are found offline without making another copy. Additional ordinary
adapters use separate `--lora PATH SCALE` arguments; never stack two acceleration
students. The local WebUI reports the effective adapter, scale, sampler, mode,
steps and requested seed, including automatically generated seeds. A selected
recipe is not a guarantee of skin quality; compare identical prompts and seeds.
