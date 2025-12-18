
## followup.md
```md
# Validation steps
1) Run the same dataset with the same settings.
2) In the log, confirm:
   - Phase 5 starts with GPU allowed, but then prints the existing warning/info about forcing CPU because per-pixel alpha weights are present.
   - A debug/info line like:
     `assemble_reproject_coadd: input_weights sample ... weight_source=alpha_weight2d`
     (not `coverage_mask`).
3) Visually inspect final mosaic:
   - black/zero bands at tile borders are gone,
   - overlaps blend normally.

# Regression checks
- Run a dataset with no ALPHA extension: Phase 5 should stay GPU-capable and unchanged.
- Confirm Two-Pass Coverage Renorm still runs and does not crash; coverage maps should now reflect ALPHA when present.

# Notes
This change is intentionally minimal: it only fixes ALPHA loading. No algorithm changes.