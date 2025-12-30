# Follow-up: Validation steps for intertile pruning connectivity fix

## 1) Quick sanity check (log-level)
Run the same dataset that produced the patchwork seams.

In `zemosaic_worker.log`, locate `[Intertile] Pair pruning ...` section and verify:

- You see `Pair pruning summary` (new line), and:
  - `components=1` and `bridges_added > 0`
  OR
  - `PRUNE_FALLBACK_NO_PRUNING` present (explicit)

You must NOT see:
- `WARN - Could not connect all active components. Remaining components: ... Bridges added: 0`

## 2) Confirm TwoPass still executes (unchanged)
Verify existing `[TwoPass] ...` lines still appear:
- `Second pass requested`
- `Computed gains ...`
- `coverage-renorm merged`

## 3) Visual validation
Compare final mosaic before/after patch using the same stretch:
- Look specifically at tile boundaries (where you previously saw plate-like brightness jumps).
- Expect much smoother background continuity and fewer "plaque" transitions.

## 4) If seams persist (secondary tuning, NOT part of patch)
Try (runtime config) for extended/structured targets:
- `intertile_sky_percentile=(5,25)` or `(10,30)` instead of `(30,70)`
- keep `robust_clip_sigma` around 2.5–3.0

But do NOT change defaults in code unless requested.

## 5) Regression check (small dataset)
Run a smaller mosaic (few dozen tiles):
- Ensure no new warnings
- Ensure runtime not dramatically worse
- Ensure output remains at least as good as before
s