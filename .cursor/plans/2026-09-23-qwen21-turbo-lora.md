# Qwen-Image-2.1 acceleration LoRA

User request: add local acceleration LoRA support and test it.

- Reuse mflux LoRA resolution/loading for both Qwen 2.1 entry points; keep the existing Heretic compatibility patch.
- Map Diffusers/PEFT LoRA keys against both transformer layouts and reject incompatible shapes/partial adapters.
- Add an opt-in `viggle-turbo` preset using the pinned 339,832,808-byte adapter at HF revision `bafc91e4cc934f5fb1406b22496a0bed9b99c548`. Use four steps, guidance 1, scale 1, runtime adapters, and no terminal sigma stretching. Preserve base defaults.
- Test matrix arithmetic, both layouts, parser/metadata wiring, incompatible adapters, and sigma parity. Run focused existing regressions and lint.
- Generate benign text-to-image and reference-edit samples with the actual local bf16 + Heretic model. Compare against 40-step base with matching settings and record timing/thermal measurements; inspect outputs and report quality limits.
- Document commands and evidence. No upstream push, full-model replacement, quantized-model export, or removal of existing files.
