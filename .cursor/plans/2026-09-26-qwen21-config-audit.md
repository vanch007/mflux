# Qwen 2.1 configuration audit

User request: verify current Viggle defaults and current mflux support/optimizations. Existing authorization covers the local LoRA integration, its fixes and necessary local tests.

- Verify official model-card settings, LoRA checksums and upstream main/open PR changes.
- Fix the existing preset across Python, both CLIs and WebUI: 6-step default, official optional 8-step text schedule, scale/CFG 1, runtime LoRA, no terminal stretch, prefix KV enabled. Reject incompatible settings early.
- Apply relevant upstream compatibility fixes while preserving reference editing, Heretic weights and prior local changes. Record remaining differences from main; no wholesale branch replacement.
- Add meaningful scheduler/CLI/WebUI regressions, run lint and fast tests in the installed inference environment, then run a benign real WebUI sample.
- Update current usage docs/skill and record reproducible evidence.
