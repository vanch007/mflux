# Upstream synchronization — 2026-10-01

## Authorized scope

Check the mflux-community/mflux and Amo643/mflux-paint upstream projects, integrate applicable changes into the local checkouts, and create semantic local commits with smart-commit. Preserve existing user work and model files.

## Implementation

1. Fetch upstream refs, recover the shallow PR history, and record upstream commits.
2. Back up current HEADs and tracked diffs; commit existing related changes by purpose before integration.
3. Merge mflux upstream main, resolving shared Qwen components while retaining Heretic configuration, runtime acceleration/effect LoRAs, independent edit noise, independent reference resolution, and FP32 VAE.
4. Keep approximate step skipping disabled by default. Preserve the WebUI's local oMLX PE, reference tray, per-adapter strengths, saved-image provenance, and existing other-model entries.
5. Run focused regressions, fast tests, lint, CLI compatibility checks and a local Qwen generation/edit smoke test where the updated model path requires it.
6. Update project records and an audit report; finish with local semantic commits.

## Evidence before integration

- mflux baseline focused tests: 103 passed.
- mflux-paint baseline tests: 54 passed.
- mflux-paint upstream `2d5db24942d50b0e8561ee5c4b7e11049b67302a` is already an ancestor of local HEAD; no missing upstream commits.
- mflux upstream `83f4d1d` contains new Qwen caches, shared components, edit helpers, scheduler and loader fixes.
- Backups: `.git/local-backups/upstream-sync-20261001-152732/` in both checkouts, plus local `backup/pre-upstream-sync-20261001-152732` branches.

## Verified result

- Community main `83f4d1d` merged in `6e01c0f`; frontend main `2d5db24` already included.
- Default regressions: 2271 passed, 13 skipped; frontend regressions: 54 passed. Tracked-code lint/format/type checks, CI extraction and package build passed.
- Actual 1024-square bf16/r128/6-step WebUI generation and blue-teapot reference edit passed; these runs are functional checks, not isolated speed measurements.
- Heretic exports lack the language generation head needed by the new built-in mask/verify helpers; WebUI PE keeps its existing oMLX backend.
- Evidence and remaining optional checks: `docs/upstream-sync-2026-10-01.md` and `outputs/upstream-sync-2026-10-01/`.
