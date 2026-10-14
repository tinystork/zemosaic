# R38-A — Bounded causal probe harness (offline, read-only) + Gate-A report

**Mission:** ZM-ZEGRID-R38-A-CAUSAL · **Phase:** implementation
**Baseline:** `beta` @ `e13d52d7ebc80ef40b1aa9d64681aefa79bffe4e` (R38 plan committed).
**Status:** IMPLEMENTED. New standalone offline tool + focused tests + this note.
**No** production wiring, default, config, GUI, manifest or FITS change.

Authoritative architecture/gates: `R38_LAYOUT_INDEPENDENT_CALIBRATION_PLAN.md`.
This document only describes the R38-A instrument and its honest limits.

---

## 1. Why

The R38 plan forbids implementing a new photometric solver before falsifying the
responsibility of the per-Cell gauge choices (frame set, reference, coefficients,
weights/exclusions). R38-A is that bounded first instrument: it must be able to
**confirm or falsify** that per-Cell gauge choices are a dominant cause of the
inter-Cell disagreement, using only the accepted Windows-816 pre-feather capture
and the read-only source FITS, and it must produce auditable compact evidence.

Static hypothesis under test (from `core/zegrid/photometric.py`):
`compute_cell_gauge` recomputes `compute_fixed_normalization` over each Cell's own
patch domain, so reference/coefficients/weights/exclusions can depend on the
layout. **This is a hypothesis, not a proven cause.**

## 2. What this adds

* `tools/r38_causal_probe.py` — the offline CLI (import side-effect-free; every
  engine import is lazy at call time).
* `tests/test_zegrid_r38a_causal_probe.py` — focused suite (`-W error`).
* this note + a status line in the R38 plan.

No new dependency, no network, no third-party code, no correction application.

## 3. CLI contract

```
python tools/r38_causal_probe.py \
    --capture-dir PATH        # R37E capture dir (READ-ONLY: artifact + manifest)
    --source-dir  PATH        # read-only source FITS
    --mode {inventory,gauge,factors}   # default inventory
    [--execute-real]          # required for any real-data pixel probe
    [--out PATH] [--report-dir PATH] [--scratch-dir PATH]
    [--expect-identity {checked,any}]
    [--max-cell-contributors N] [--max-frames-total N] [--max-seconds S]
    [--factor-max-contributors N] [--factor-band-px N] [--factor-max-tangent-px N]
    [--determinism-check]
```

Modes and their writes:

| mode | reads source pixels | writes | writes under the capture/source trees |
|---|---|---|---|
| `inventory` | no (FITS headers only) | the strict-JSON report | never |
| `gauge` | yes (banded, capped) | report + scratch aligned cache | never |
| `factors` | yes (gauge + bounded ROI) | report + scratch aligned cache | never |

Any `--out`, `--report-dir` or `--scratch-dir` inside a read-only tree
(`/home/tristan/caldwell11`, the capture dir, the source dir) is REFUSED before
any work.

## 4. Behaviour

### 4.1 Strict identity validation (always, before any probe)

`validate_identity` strict-loads (NaN/Infinity literals rejected) the artifact and
the manifest and independently verifies:

* artifact schema/schema_version, `capture_stage`, `correction_applied is False`,
  `production_wired is True`;
* manifest `engine == "zegrid"`, non-empty UNIQUE string `frame_ids`,
  `n_frames == len(frame_ids)`, `cell_feather.enabled is True`, effective gauge
  mode `per_cell`;
* the **source-set digest is recomputed** from the canonical relative logical ids
  and must equal BOTH the artifact provenance digest AND the manifest capture
  block digest (same algo tag), with the source count matching;
* the manifest capture block `sha256`/`bytes` must equal the **on-disk** artifact
  hash/size (non-tautological), plus capture id and filename;
* `qualification == "EXACT"` with `holdout_exact is True`, `holdout_capped is
  False` and `n_dropped == 0`;
* `--expect-identity checked` (default) additionally requires the recorded
  Windows-816 identity (816 frames, 6x4, canvas 4117x4377 / id `e8f1877f16f8cca4`,
  capture id `6c311af7…`, artifact sha256 `a260b330…`, digest `e3f4c16a…`).
  `--expect-identity any` accepts another valid artifact but records the computed
  identity (a generic-mode run is never presented as the checked run).

### 4.2 Geometry reconstruction (no duplicated pixel math)

Sources are resolved from the manifest ids (relative-path first, then a UNIQUE
basename; traversal/absolute/missing/ambiguous/duplicate ids are refused), then
the accepted primitives rebuild the run:

`zemosaic_zegrid_mode._build_frame_descriptors` → `geometry.build_canvas` →
`auto_layout.project_footprints` → `sweep.build_cell_contexts_from_footprints`.

On the checked capture this reproduces canvas `e8f1877f16f8cca4` (4117x4377) and
`Σ patch_ids = 6114`, matching the manifest cache entry count.

### 4.3 Frozen DEV relation panel

A deterministic, deliberately BIASED panel of six geometric relations is frozen
from the capture artifact **before** any factor result: the R37-declared worst
edge, the highest-burden vertical relation, the highest-burden relation, the
median-burden relation, the minimum-burden relation and the cheapest corner
relation (dedup in slot order). "Burden" is `max_channel |local_offset_train|`
(ADU) — a transparent level-step proxy. The panel is labelled
`NOT_QUALIFICATION` and its metrics never qualify a model.

### 4.4 Full-domain per-Cell gauge facts

For each selected Cell the tool streams its **FULL accepted patch domain**
(production decoder → source-ROI crop → `slice_wcs` → `reproject_cropped` →
`AlignedCacheBuilder`, one frame resident) and runs the FROZEN estimator
(`photometric.compute_cell_gauge`) on that provider. It records the ordered frame
ids, reference id/support, `(a,b)` coefficients, weights, active flags and
exclusions, plus the effective algorithm (`sky_mean` + `noise_variance`). The
gauge is never estimated from a seam strip.

Cross-side comparison then works on the common ACTIVE frames. For `sky_mean`
(`a=1`, additive) the same frame at the same pixel differs between the two Cells
by exactly `b_u - b_v`; the median of that difference is the gauge-induced
per-Cell **level step**, and its MAD measures how frame/mask-dependent it is.
Non-additive (`a != 1`) or empty-common comparisons ABSTAIN (never approximated).

### 4.5 Factor probe A/B/C/D (bounded ROI, fixed CPU backend)

On the shared global-pixel ROI along the relation's boundary (measurement band ±
`band_px`, capped tangent extent) each side is restacked with the accepted
streaming executor through a read-only ROI provider, holding a fixed gauge. The
ladder is a CHAIN in which every step changes exactly ONE variable:

* `BASE` own frames + own local gauge + own weights + declared rejection;
* `A` FRAME SET only (common intersection; each side keeps its OWN gauge subset);
* `G` GAUGE only (both sides use side-u's gauge over the common frames);
* `W` WEIGHTS only (coefficients kept, weights set uniform);
* `D` REJECTION only (`none`; DIAGNOSTIC ONLY);
* `B0` control: identity gauge (a=1,b=0) + uniform weights (a layout-independent
  gauge choice; **not presumed right**).

Metrics per channel: median/RMS/p90/MAD of the pair difference, tangent and
normal slopes, level correlation, block-median spread, contributors before and
after rejection, rejected fractions. A setting that cannot run returns `NOT_RUN`
with a reason — never a fabricated zero. ROI restacks reuse the executor on a
sub-window, so the footprint taper is computed within the ROI (documented edge
effect) and the gauge coefficients/weights come from the FULL domain.
Note: with `sky_mean` and a shared gauge the two sides are recombined from the
same frames, so a residual of numerical magnitude is the expected outcome; the
metric therefore isolates the LEVEL/gauge term and cannot, on its own, quantify a
spatial model. Interactions are reported per setting, never summed.

### 4.6 Bounded resources + strict output

Explicit caps (`--max-cell-contributors`, `--max-frames-total`, `--max-seconds`,
`--factor-*`); deterministic sorted order everywhere; no `N_frames x canvas`
cube; strict JSON with `allow_nan=False` and non-finite values emitted as `null`.
Panel relations that cannot fit the caps are listed explicitly under `not_run`.

### 4.7 Conclusion (pre-registered)

Exactly one of `DOMINANT_CAUSE_CONFIRMED`, `PARTIAL_CAUSE`, `CAUSE_FALSIFIED`,
`INCONCLUSIVE`, from frozen thresholds on the ratio `gauge_step / observed_step`,
its consistency, and the number of comparable probed relations. A small-band or
single-worst-edge result is never treated as causal; interactions are reported,
never summed into an additive attribution.

## 5. Tests

`tests/test_zegrid_r38a_causal_probe.py` (run under `-W error`) covers: the
module-scope import audit (fresh-interpreter check that importing the tool pulls
no `zemosaic` module), strict JSON (NaN/Infinity rejected, canonical dump,
non-finite → `None`), the digest's byte-compatibility with the accepted
algorithm, identity PASS/tamper (sha256, bytes, capture id, digest, capped
holdout, dropped relations, correction flag, gauge mode, engine), source
resolution refusals, panel determinism/slots (both orientations, corner,
declared worst edge), bounded planning, gauge cross-comparison (constant step
recovered; non-additive abstention), ROI metrics vs a hand-computed difference,
the read-only ROI provider over a real R6 aligned cache, factor-probe `NOT_RUN`
refusals, the conclusion rule, output safety, and an end-to-end synthetic
`inventory` run (strict JSON, capture tree untouched, deterministic repeat).

## 6. Limits / non-goals

* No production runtime/default/GUI/manifest change; diagnostic only.
* No R38-B solver, no gain/plan/spline correction, no `canonical_*`/Legacy/SDS
  modification.
* No full 816-frame stack or 3-hour Windows emulation; the probe is explicitly
  bounded and reports `NOT_RUN` instead of approximating.
* Not a qualification corpus: the panel is biased DEV and the R37 held-out has
  already been consumed for the R37 model decision.
