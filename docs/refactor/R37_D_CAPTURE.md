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

* **Measurement + feather widths** are explicit and consistent with the real halo
  overlap: `measurement_px = 2 * halo_px` and `feather_px = halo_px` normally
  (`2 * 8 = 16 px` and `8 px`, the accepted defaults), falling back to the accepted
  defaults for a halo-less patch. The halo, the measurement band, the feather width
  and the effective gauge mode are recorded in the payload provenance/meta and the
  manifest (L1).
* The capture reads the `CellPatch` objects by reference; `observe_edge` copies
  to float64 for its own maths, so the patches, graph and offsets are unmodified.

---

## 3. Artifact publication (`background_diagnostics.publish_artifact`)

The **only** supported persistence path. The basename is **FIXED**
(`zegrid_r37_pre_feather.json`) and `publish_artifact(output_dir, payload)` exposes
**no basename override** (L2), so there is **no path-traversal surface** — only the
directory is chosen by the caller:

0. **Fail-closed invalidation FIRST** — any pre-existing fixed target is removed
   *before* validation/writing, so a stale artifact from a previous run can never be
   left looking like the current requested capture (M2). If the target cannot be
   removed, publication fails loudly.
1. **Strict validation before writing** — `load_artifact(payload)` runs after the
   invalidation; a malformed payload is never published (and the old target is
   already gone).
2. **Atomic, same-directory write** — temp file + `flush` + `fsync` + `os.replace`;
   the temp is always removed on failure. A best-effort directory `fsync` follows the
   unlink and the rename.
3. **SHA-256 / byte count AFTER publication**, from the on-disk bytes.

On **any** failure the fixed target is **absent** and no temp residue remains. An
**explicit** request fails **loudly** — there is no silent missing/corrupt artifact.
The **disabled** path never imports, builds, invalidates or writes anything.

Compact capture metadata is recorded in `cell_feather_info` → manifest
(`cell_feather.r37_diagnostic_capture`) and the run log: `status`, `filename`,
`capture_id`, `run_label`, `n_source_frames`, `source_set_digest`,
`schema`/`schema_version`, `bytes`, `sha256`, `n_relations`, `n_dropped`,
`holdout_cap`, `holdout_capped`, `holdout_exact`, `qualification`, `capture_stage`,
`effective_cell_gauge_mode`, `halo_px`, `measurement_px`, `feather_px`. When disabled
the manifest keeps its exact previous shape.

## 3b. Run identity, linkage and the uncommitted rule (M2)

* **Identity** — `_run_single` generates ONE stable opaque identity per requested
  capture pass *before* per-Cell stacking (single / EQ / ALTZ each distinct) and
  records it in the run log first:
  * `capture_id = capture_run_id(start_ts, pass_label, n_frames, source_set_digest)`
    — a deterministic sha256 (hex, strict-JSON friendly), **no hidden RNG**;
  * `source_set_digest = source_set_digest([frame_id.logical_path ...])` — a
    deterministic, order-invariant, de-duplicated sha256 of the SORTED canonical
    **relative logical** frame identifiers (never absolute private paths); the
    algorithm is declared as `SOURCE_SET_DIGEST_ALGO`;
  * `run_label` (pass) and `n_source_frames`.
  The same identity fields appear in the artifact provenance (`provenance.capture`)
  and in the manifest capture metadata.
* **Cross-link (commit rule)** — a capture artifact is **UNCOMMITTED** unless a
  *completed* `zegrid_manifest.json` links it on all four keys
  (`capture_id`, `filename`, `sha256`, `bytes`) — enforced by
  `manifest_links_artifact`. A **stale** prior manifest mismatches the new
  artifact's id/hash; a **missing** manifest is non-qualifying. The artifact is
  never rewritten after publication; the manifest is written later, so a later run
  failure legitimately leaves an *uncommitted* artifact as capture evidence.

## 3c. Truthful, single-valued provenance (M1)

An artifact published through the capture seam carries **only** the truthful
production-capture provenance built by `capture_provenance(...)`:
`mission = "ZM-ZEGRID-R37-D"`,
`status = CAPTURE_PROVENANCE_STATUS (production_capture_uncommitted_until_manifest)`,
`production_wired = True`, `capture_stage = "pre_offset_pre_feather"`,
`correction_applied = False`, plus a nested `capture` block (stage, identity, mode,
widths) that **agrees** and repeats **no** wiring claim. The pure/offline
`DIAGNOSTIC_PROVENANCE` (mission R37-C, `production_wired = False`) remains the
default for **standalone** `build_payload` use only and is **replaced** at the seam,
so a published production artifact can never say `production_wired` both false and
true. The module self-description was updated: it remains pure math/artifact code
but is now lazily imported by the explicit, default-OFF production capture seam.

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

`tests/test_zegrid_r37d_diagnostic_capture.py` (27 tests): default OFF (no
artifact, module never called), explicit `per_cell` ON (strict-loadable artifact,
truthful single-valued provenance, run identity in artifact+manifest+run log,
relations match a fresh build, no temp residue, digest/bytes metadata), cap large →
`EXACT` / small → `RESEARCH_ONLY_HELDOUT_CAPPED` (cannot qualify `ROBUST_PLAN`),
`auto → per_cell` capture, explicit `global` and `auto → global` loud failures,
mount segregation (one artifact per output dir), publication failure loudness with
fail-closed invalidation (validation failure AND `os.replace` failure leave the
fixed target absent, no residue), later-run-failure uncommitted rule, source-set
digest stability/sensitivity, `capture_id` distinctness, `manifest_links_artifact`,
non-default halo widths (L1), fixed basename / no override (L2), ON/OFF byte+offset
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
