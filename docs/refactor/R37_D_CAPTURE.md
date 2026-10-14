# R37-D — Pre-offset/pre-feather diagnostic capture seam

**Mission:** ZM-ZEGRID-R37-D-CAPTURE · **Phase:** implementation · **Iteration:** r0
**Baseline:** `beta` @ `7e929735` (R37-C accepted by Junior + Nono).
**Status:** IMPLEMENTED. Diagnostic-only, default-OFF. No correction, no default
change, no science/output drift.

---

## 1. What this adds

An explicit, default-OFF capture seam at the **real pre-offset/pre-feather per-Cell
assembly point**. When deliberately requested it publishes the *accepted* R37-C
strict JSON artifact into **that run's actual output directory** so a future
offline R37 evaluation can run on the exact relations used at assembly time.

It does **not** apply any R37 correction, does **not** touch the offset graph,
additive offsets, feather weights, or any science/aesthetic/coverage array, and
never changes normal outputs or default behaviour.

Public opt-in (keyword-only, both default OFF/compat):

```python
run_zegrid_mode(
    ...,
    r37_diagnostic_capture: bool = False,
    r37_diagnostic_holdout_cap: int = 64,   # == background_diagnostics.HOLDOUT_CAP
)
```

Forwarded verbatim through every mount-segregated / single `_run_single` path.

---

## 2. Exact capture point

Inside `_assemble_per_cell_feathered`, **after** the complete `CellPatch` objects
coexist and **before** `build_cell_offset_graph`, the additive offsets and the
feather assembly. The published provenance says `capture_stage =
"pre_offset_pre_feather"` (module constant `R37_CAPTURE_STAGE`, mirrored in
`background_diagnostics.CAPTURE_STAGE`).

The payload is produced by the **unchanged** accepted
`background_diagnostics.build_payload` (which delegates all pixel/difference math
to `background_model.observe_edge` / `build_observations`). There is **no duplicate
pixel or difference math**.

* **Measurement band** is explicit and consistent with the real halo overlap:
  `measurement_px = 2 * halo_px` normally (`2 * 8 = 16 px`, the accepted default),
  falling back to the accepted default band for a halo-less patch. The halo and
  the effective gauge mode are recorded in the payload provenance and the
  manifest.
* The capture reads the `CellPatch` objects by reference; `observe_edge` copies
  to float64 for its own maths, so the patches, graph and offsets are unmodified.

---

## 3. Artifact publication (`background_diagnostics.publish_artifact`)

The **only** supported persistence path, with a **fixed** basename
(`zegrid_r37_pre_feather.json`); the caller selects only the directory, so there
is **no path-traversal surface**:

1. **Strict validation before publication** — `load_artifact(payload)` runs first;
   a malformed payload never touches the filesystem.
2. **Atomic, same-directory write** — temp file + `flush` + `fsync` + `os.replace`;
   the temp is always removed on failure (no partial file, no residue). A
   best-effort directory `fsync` follows.
3. **SHA-256 / byte count AFTER publication**, from the on-disk bytes.

An **explicit** request fails **loudly** — there is no silent missing/corrupt
artifact. The **disabled** path never imports, builds, or writes the diagnostic
(the import is lazy inside the capture branch only).

Compact capture metadata is recorded in `cell_feather_info` → manifest
(`cell_feather.r37_diagnostic_capture`) and the run log: `status`, `filename`,
`schema`/`schema_version`, `bytes`, `sha256`, `n_relations`, `n_dropped`,
`holdout_cap`, `holdout_capped`, `holdout_exact`, `qualification`, `capture_stage`,
`effective_cell_gauge_mode`, `halo_px`, `measurement_px`. When disabled the
manifest keeps its exact previous shape.

---

## 4. Mode requirement (loud, early)

Capture requires the **effective** `cell_gauge_mode == "per_cell"`:

* explicit `"global"` → fail loudly in `run_zegrid_mode` **before** input discovery;
* `"auto"` → deferred to the run-level resolution in `_run_single`; an
  `auto → global` resolution fails loudly **before** per-Cell stacking.

Argument types are validated strictly: `r37_diagnostic_capture` must be a real
`bool`; `r37_diagnostic_holdout_cap` must be a real `int` and `>= 4` when capture
is enabled. No silent coercion, no GUI/config activation.

---

## 5. R37-C LOW cleanup (Nono LOW-1)

The accepted R37-C `recompute_decision` previously injected non-`PASS`
diagnostic entries into the accepted `gates` mapping and left
`decision=INSUFFICIENT_EVIDENCE` while `gates["evidence"]/["heldout"]` stayed
`PASS`.

Cleanup (in `background_diagnostics.recompute_decision` only — the accepted
`background_model` logic is untouched):

* the accepted scientific **`gates` mapping is left strictly intact** (no
  diagnostic key is injected, so `all(v == "PASS" for v in gates.values())` keeps
  its accepted meaning);
* qualification lives in a dedicated, namespaced
  **`result.provenance["diagnostic"]`** block:
  `qualified`, `qualification` (`EXACT` | `RESEARCH_ONLY_HELDOUT_CAPPED`),
  `heldout_capped`, `heldout_exact`, `n_capped_relations`,
  `accepted_decision`, `accepted_model`, `effective_decision`, `effective_model`,
  `downgraded_from`, `downgrade_reason`, `downgrade_policy`, `note`;
* **decision / qualification are internally unambiguous**: the block records both
  the accepted decision and the conservative effective decision (with an explicit
  downgrade reason), and `to_json_dict()` preserves it.

The conservative policy is **not weakened**: a capped artifact is still
downgraded away from `ROBUST_PLAN` with an explicit refusal reason.

---

## 6. Scientific / output invariants (proven)

* Diagnostic-only: no correction, no modification of patches, graph, offsets,
  feather weights, science/aesthetic/coverage arrays or FITS headers.
* Capture ON vs OFF on a deterministic synthetic integration corpus yields
  **bit-identical** `mosaic_grid_science.fits`, `mosaic_grid_aesthetic.fits` and
  `mosaic_grid_coverage.fits` and **identical R34c offsets** (manifest
  `cell_feather.offsets`); only the diagnostic metadata differs.
* The pure `_assemble_per_cell_feathered` assembles **array-equal** results and an
  info dict equal once the capture metadata is set aside.

---

## 7. Tests

`tests/test_zegrid_r37d_diagnostic_capture.py` (19 tests): default OFF (no
artifact, module never called), explicit `per_cell` ON (strict-loadable artifact,
provenance/stage, relations match a fresh build, no temp residue, digest/bytes
metadata), cap large → `EXACT` / small → `RESEARCH_ONLY_HELDOUT_CAPPED` (cannot
qualify `ROBUST_PLAN`), `auto → per_cell` capture, explicit `global` and
`auto → global` loud failures, mount segregation (one artifact per output dir),
publication failure loudness (no partial file/residue), ON/OFF byte+offset
equality, pure-assembler equality, strict validation, and no config/GUI activation
/ no correction.

`tests/test_zegrid_r37c_diagnostics.py` was updated for the LOW cleanup
(namespaced `provenance["diagnostic"]`, gates untouched) and its
"no production import" check now parses the AST for **module-level** imports
(a lazy capture-branch import is the intended R37-D seam).

---

## 8. Limits / non-goals

* No R37 correction application / no `apply_plane_correction` wiring.
* No change to R34/R36 math/defaults, calibration, normalization, geometry,
  feather, FITS pixels, GUI/config defaults.
* No new dependency, no spline/nonlinear model, no reprojection, no
  cache/prefetch/multiprocessing.
* No write under `/home/tristan/caldwell11` or M106; no full data run.
* Producing a real-run capture is authorised by this seam, but running the real
  Caldwell/M106 corpus remains out of scope for this mission.
