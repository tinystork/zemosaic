# followup.md — execution tracker (aesthetic-first mission)

## Mission status snapshot

### Implemented
- [x] Patchwork suppressor (A) wired + logged
- [x] Underconstrained intertile guardrail (B) wired + logged

### Not implemented yet
- [ ] Dual FITS export (`_science` + `_aesthetic`) (C)
- [ ] Aesthetic hole-fill / seam completion (D)

---

## Field observation (latest)

From TESTC/TESTD/TESTE style runs:
- clear visual progress vs older NGC6888 runs,
- but residual holes remain in specific regions,
- logs indicate most gains are photometric harmonization,
- geometric coverage mask changes remain small.

Interpretation:
- current A/B stack is working,
- final remaining issue is primarily **coverage hole completion in visual branch**.

---

## Active objective shift

Primary product target is now:
- aesthetic mosaic output that is smooth/homogeneous and editor-friendly,
- while preserving scientific output and high throughput.

---

## TODO by objective

### C — Dual FITS export
- [x] Add config keys (`export_aesthetic_fits`, suffixes)
- [ ] Add UI switch + help text
- [x] Export both files in one run when enabled
- [x] Add explicit logs + header provenance

### D — Aesthetic hole-fill
- [x] Add hole-mask detection from coverage/alpha
- [x] Add local inpainting/fill (aesthetic branch only)
- [ ] Add blend/feather controls
- [ ] Add star/detail protection guard
- [x] Add diagnostics (`[AestheticFill]` suggested)

### Performance guardrails
- [ ] Measure runtime overhead on sparse + dense runs
- [ ] Measure peak RAM delta
- [ ] Keep default mode fast (bounded pass count)

---

## Proposed config/profile defaults (draft)

Baseline fast aesthetic profile:
- `patchwork_suppressor_enabled=true`
- `patchwork_suppressor_strength=normal`
- `intertile_underconstrained_guard_enabled=true`
- `intertile_underconstrained_force_mode=offset_only`
- `export_aesthetic_fits=false` (default conservative)

Strong aesthetic profile (opt-in):
- `patchwork_suppressor_strength=strong`
- `aesthetic_hole_fill_enabled=true`

---

## Validation plan (next runs)

1. Sparse existing-master baseline
2. Sparse A+B
3. Sparse A+B+C
4. Sparse A+B+C+D
5. Dense sanity + perf check

Record for each:
- visual verdict,
- key log lines,
- runtime and memory overhead.

---

## Exit criteria

Mission considered complete when:
1. aesthetic output is consistently editable and visually smooth,
2. science output remains trustworthy and clearly separated,
3. throughput remains compatible with high-volume workflows.
