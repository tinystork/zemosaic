# agent.md — ZeMosaic mission brief (aesthetic-first, speed-safe)

## Product objective (updated)
Deliver a mosaic output that is:
1. **visually smooth and homogeneous** (background + nebulosity, minimal seams/patches/holes),
2. **easy to edit** in Siril / PixInsight / Seti Astro Suite,
3. while preserving ZeMosaic’s core advantage: **high-throughput processing of very large datasets**.

This mission explicitly prioritizes aesthetic usability for most users, **without sacrificing** scientific integrity or pipeline speed.

---

## Output contract (mandatory)

### Output A — Scientific
- canonical FITS, physically faithful as much as possible,
- never silently altered by aesthetic-only operations,
- remains the reference output.

### Output B — Aesthetic
- dedicated visually-optimized FITS,
- may include seam suppression + local hole-fill/inpainting,
- intended for downstream editing workflows.

Naming recommendation:
- `*_science.fits`
- `*_aesthetic.fits`

---

## Implemented foundations (already done)

### A) Patchwork suppressor
- config + pipeline hook + logs (`[Patchwork]`) available.

### B) Underconstrained intertile guardrail
- sparse-graph detection + safe fallback + logs (`[IntertileGuard]`) available.

These remain active as baseline protections.

---

## New Objective C — Dual FITS export
Status: **in progress (core runtime implemented)**

### Requirements
- [ ] Add `export_aesthetic_fits` switch.
- [ ] Export both science and aesthetic FITS when enabled.
- [ ] Guarantee no overwrite ambiguity (safe filenames).
- [ ] Add metadata/header tags identifying branch + parameters.

Suggested keys:
```json
"export_aesthetic_fits": false,
"scientific_fits_suffix": "_science",
"aesthetic_fits_suffix": "_aesthetic"
```

---

## New Objective D — Aesthetic hole-fill / seam completion
Status: **in progress (core runtime implemented)**

Purpose:
- remove residual “holes” and patch discontinuities in visual branch,
- keep edits local and low-frequency aware,
- avoid destructive star/core smearing.

### V1 behavior (aesthetic branch only)
- detect invalid/near-invalid holes from coverage/alpha maps,
- fill locally using seam-aware inpainting / low-frequency completion,
- feather transitions to avoid hard patches,
- protect compact high-frequency structures (stars, sharp filaments).

Suggested keys:
```json
"aesthetic_hole_fill_enabled": true,
"aesthetic_hole_fill_max_radius_px": 64,
"aesthetic_hole_fill_blend": 0.7,
"aesthetic_hole_fill_only_near_seams": true
```

---

## Throughput constraint (non-negotiable)

Any new aesthetic module must respect large-scale throughput:
- O(N) / tiled-friendly memory behavior,
- no heavy global optimization loops in default mode,
- one-pass or bounded multi-pass,
- optional stronger mode allowed only as explicit opt-in.

Default should stay **fast + robust**.

---

## Validation matrix (must complete)

### Sparse pathological case (existing master tiles)
- [ ] baseline (A/B off)
- [ ] A+B on
- [ ] A+B+C (dual export)
- [ ] A+B+C+D (with hole-fill)

### Dense normal case
- [ ] regression check runtime
- [ ] regression check visual integrity

### Performance checks
- [ ] runtime overhead delta (%)
- [ ] peak RAM delta
- [ ] output size delta

---

## Acceptance criteria

### Visual (aesthetic branch)
- visibly reduced seams/patches,
- problematic holes substantially reduced or visually neutralized,
- output judged “ready to edit” in Siril/PixInsight/Seti Astro Suite.

### Integrity (science branch)
- scientific FITS unchanged in semantics vs baseline branch.

### Performance
- no unacceptable slowdown on large runs,
- throughput profile remains compatible with thousands-of-frames workflows.

---

## Priority order
1. Objective C (dual FITS export)
2. Objective D (hole-fill visual completion)
3. tuning presets (Balanced / Strong)
4. finalize default profile for production.
