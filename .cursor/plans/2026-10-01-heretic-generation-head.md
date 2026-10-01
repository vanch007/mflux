# Restore Heretic language generation weights

- Confirm the deployed MLX export omits `lm_head.weight` and `language_model.norm.weight`.
- Copy only those BF16 tensors from the pinned original Heretic checkpoint; preserve all conditioning weights and configs.
- Add a reproducible repair utility with an index backup, exact range checks, shape validation and provenance hashes.
- Verify lossless extraction, idempotency, runtime loading, object grounding, edit rewriting and image comparison.
- Keep large weights and generated evidence local; record the results in project documentation.
