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

Any `--out`, `--report-dir` or `--scratch-dir` inside a read-only input tree (the
capture directory or the lights root passed to the tool, plus any root named in
`R38A_FORBIDDEN_ROOTS`) is REFUSED before any work. A symlinked capture or source
root passed on the command line is refused as well.

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

### 4.5 Factor probe A/G/W/D (bounded ROI, fixed CPU backend)

On the shared global-pixel ROI along the relation's boundary (measurement band ±
`band_px`, capped tangent extent) each side is restacked with the accepted
streaming executor through a read-only ROI provider, holding a fixed gauge. The
ladder is a CHAIN in which every step changes exactly ONE variable:

* `BASE` own frames + own local gauge + own weights + declared rejection;
* `A` FRAME SET only (common intersection; each side keeps its OWN gauge,
  faithfully subset with remapped exclusions);
* `G` GAUGE only (both sides use side-u's gauge over the common frames);
* `W` WEIGHTS only (identical coefficients/reference/frame order/`norm_active`/
  `weight_active`/exclusions; ONLY the active numeric weights become uniform);
* `D` REJECTION only (`none`; DIAGNOSTIC ONLY);
* `B0` CONTROL (multi-variable): identity gauge (a=1,b=0) AND uniform weights.
  It is NOT part of the one-variable ladder.

`G` is an EXPECTED IDENTITY sanity check, **not** independent causal evidence:
both sides recombine the SAME common frames with the SAME gauge over the SAME
global ROI, so a near-zero difference is expected by construction. The JSON
`contract.tautology.G` says so explicitly, and the conclusion rule does not
consume `factor_evidence`.

Metrics per channel: median/RMS/p90/MAD of the pair difference, tangent and
normal slopes, level correlation, block-median spread, contributors before and
after rejection, rejected fractions. A setting that cannot run returns `NOT_RUN`
with a reason — never a fabricated zero. ROI restacks reuse the executor on a
sub-window, so the footprint taper is computed within the ROI (documented edge
effect) and the gauge coefficients/weights come from the FULL domain.
Interactions are reported per setting, never summed.

Only DOCUMENTED domain/budget failures map to `NOT_RUN`; an unexpected exception
(programming error) FAILS the run with top-level `status=FAILED` instead of being
disguised as a scientific `NOT_RUN`.

### 4.6 Bounded resources, strict output, privacy boundary

Explicit caps (`--max-cell-contributors`, `--max-frames-total`, `--max-seconds`,
`--factor-*`); deterministic sorted order everywhere; no `N_frames x canvas`
cube; strict JSON with `allow_nan=False` and non-finite values emitted as `null`.
Panel relations that cannot fit the caps are listed explicitly under `not_run`.

`--max-seconds` is a **SOFT** wall-clock budget: it is checked between frames in
the cache build and between cells/settings, so a single canonical
decode/reproject/restack call cannot be interrupted and a run may overshoot by at
most one such call. This is stated in the artifact (`budgets.max_seconds_semantics`).

**Protected roots (write policy).** The run's own capture and source roots are
always protected. A `--expect-identity checked` run additionally derives the
nearest common ancestor of capture and source as a MANDATORY protected root
(refusing a filesystem/drive root, a too-shallow path, or home/its ancestors); the
path is never persisted — the artifact records only
`write_policy.dataset_root_protection` and a one-way identity hash. A generic
`--expect-identity any` run protects capture/source plus `R38A_FORBIDDEN_ROOTS`
(additive, `os.pathsep`-separated) only, and says so explicitly.

**Failure contract.** If preparation fails before output safety is established, the
tool writes NO report file anywhere (the configured output may live in a refused
tree); it emits only a compact, path-free status on stdout and returns nonzero.
The output is re-validated before parent creation, after the parent exists and
immediately before `os.replace`; the temporary file is created with
`tempfile.mkstemp` in the destination directory (`O_EXCL`, no following of a
pre-existing `.tmp` symlink, 0600 perms) and removed on failure.

The durable artifact carries NO absolute/private path: input/scratch paths are
replaced by stable logical labels (basenames / `cells/<cell_id>` keys), and a
recursive fail-closed boundary validator recognises POSIX-absolute, Windows
drive-absolute, UNC and `file://` forms (whole value and embedded). A surviving
leak sets `status=FAILED` and the report records only the offending JSON paths and
categories (never the values).

Every report carries a top-level `status` (`PASS`/`PARTIAL`/`FAILED`),
`qualification = DEV_NOT_QUALIFICATION`, `scientific_acceptance: false`,
`r38b_authorized: false` and an `evidence_scope` block (panel/relations/
orientations actually executed, relation-level aggregation, `correction_applied:
false`). `PASS` means "the harness ran every planned relation", never scientific
acceptance.

Resumable scratch caches are bound to the run by a persisted
`r38a_binding.json` (capture id, artifact sha256, source-set digest, source-root
LABEL, source file count, canvas/layout/patch facts, config tokens, ordered frame
ids + per-frame source fingerprints). A cache is reused only when the accepted R6
completeness predicate holds AND the binding matches exactly; otherwise it is
rebuilt within the same caps.

### 4.7 Conclusion (pre-registered, RELATION-LEVEL)

Exactly one of `DOMINANT_CAUSE_CONFIRMED`, `PARTIAL_CAUSE`, `CAUSE_FALSIFIED`,
`INCONCLUSIVE`. Channels are aggregated INSIDE each relation first (a relation
QUALIFIES when >= 1 channel has a stable step explaining >= 50 % of the observed
pair difference and is not negligible); `PARTIAL_CAUSE` then requires at least TWO
distinct qualifying relations covering BOTH orientations. One relation's
correlated channels can never satisfy it, and `DOMINANT_CAUSE_CONFIRMED` is
unreachable in this bounded harness (no restack-dominance criterion exists).
`INCONCLUSIVE` is the expected state when fewer than two relations (or only one
orientation) qualify. The report also exposes the executed/total panel fraction,
the qualifying relations/orientations and per-relation channel summaries. A
small-band or single-edge result is never treated as causal; interactions are
reported, never summed.

## 5. Tests

`tests/test_zegrid_r38a_causal_probe.py` (run under `-W error`) covers, beyond the
r0 set: the privacy boundary (path-leak categories incl. POSIX/Windows-drive/UNC/
`file://`/embedded, recursive scan, scrub), the resolver fail-closed cases
(Windows drive-absolute with slash/backslash/case variants, `file://`, symlinked
root and symlinked candidate), exclusion remap fidelity in the gauge subset
(in-subset remap + explicit out-of-subset record + refusal when the reference is
absent), the `W` single-variable invariant (only active weights change;
coefficients/reference/order/active masks/exclusions preserved),
`_uniform_weights_only`, the `G` tautology and `B0` multi-variable labelling, the
cache binding (absent / matching / stale mismatch / malformed, and a real binding
that is private-root-free), relation-level conclusion cases (one relation's three
channels insufficient; two relations in the same orientation insufficient; two
relations across h+v sufficient; falsified; too few), the gate
status/qualification/privacy block in the end-to-end inventory, and the resolver /
JSON / digest / identity / ROI / conclusion unit tests from r0.

## 6. Limits / non-goals

* No production runtime/default/GUI/manifest change; diagnostic only.
* No R38-B solver, no gain/plan/spline correction, no `canonical_*`/Legacy/SDS
  modification.
* No full 816-frame stack or 3-hour Windows emulation; the probe is explicitly
  bounded and reports `NOT_RUN` instead of approximating.
* Not a qualification corpus: the panel is biased DEV and the R37 held-out has
  already been consumed for the R37 model decision.
