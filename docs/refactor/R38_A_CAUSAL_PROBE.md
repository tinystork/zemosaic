# R38-A — Bounded causal probe harness (offline, read-only) + Gate-A report

**Mission:** ZM-ZEGRID-R38-A-CAUSAL · **Phase:** implementation
**Baseline:** `beta` @ `e13d52d7ebc80ef40b1aa9d64681aefa79bffe4e` (R38 plan committed).
**Final execution HEAD:** `beta` @ `ff3803b303864cac854d0bad756eada42da79f38`.
**Status:** IMPLEMENTED and EXECUTED on the full frozen DEV panel; **R38-A is CLOSED at Gate A** (§7).
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

---

## 7. Gate-A outcome — final full frozen DEV panel (R38-A closed)

This section records the FINAL R38-A execution and closes the diagnostic at Gate A.
It supersedes any earlier "pending / to be rerun" wording: there is no remaining
`NOT_RUN` relation and no outstanding step that would authorize R38-B.

### 7.1 Execution facts (independently verified)

* The complete frozen R38-A DEV panel ran over the exact Windows-816 R37E capture and
  the read-only original 816 IRCUT source frames.
* **6/6** panel relations executed, covering **both** orientations (`h` and `v`);
  `executed_fraction = 1.0`; **no** relation is `NOT_RUN` (`not_run[0].relations = []`).
* Durable artifact `r38a_factors_extended_full.json`: **1,168,958 bytes**, sha256
  `218187eb1ebb3bf806735cf160fca1580472b164acf0e411af137934bb2685e0`.
* Recorded `repo_head`: `ff3803b303864cac854d0bad756eada42da79f38`.
* Top-level `status = PASS` with `status_reason = "every planned relation executed"`.
* `runtime.seconds = 875.233`, `runtime.peak_rss_kib = 3,907,984`.
* `privacy.leaks = 0` (recursive boundary scan after known-root scrub).
* `evidence_scope`: `kind = diagnostic_dev_bounded`, `correction_applied = false`,
  `panel_relations_total = 6`, `relations_executed = 6`, `r37_heldout_consumed = true`.
* Determinism: `same_process_json_roundtrip_recompute`, `roundtrip_equivalent = true`,
  `independent_real_data_repeat = false` — this is **not** an independent real-data rerun.

### 7.2 Technical PASS vs scientific acceptance

These are different things and must not be conflated:

* `status = PASS` is a **technical** fact: the harness ran every planned relation and
  emitted a valid, strict-JSON, privacy-clean artifact.
* `qualification = DEV_NOT_QUALIFICATION`: the panel is a biased DEV diagnostic frozen
  before the results; it is not a qualification corpus.
* `scientific_acceptance = false` and `r38b_authorized = false`: no scientific claim is
  accepted by this run and no follow-up lot is authorized by it.

### 7.3 Conclusion and coverage

The pre-registered relation-level rule (§4.7) yields `conclusion.state = PARTIAL_CAUSE`:
**5 of 6** relations qualify a stable, near-constant local-gauge additive level
mechanism, across **both** orientations (gauge-step / observed ratio, per channel):

| relation | orientation | ratio |
|---|---|---|
| `r0003c0000|r0003c0001` | h | 0.999995–1.002271 |
| `r0001c0005|r0002c0005` | v | 0.994589–0.997655 |
| `r0003c0002|r0003c0003` | h (declared worst) | 0.999258–0.999496 |
| `r0000c0001|r0000c0002` | h (median burden) | 1.000713–1.001538 |
| `r0002c0004|r0002c0005` | h (high burden) | 1.000121–1.000305 |

The sole non-qualifying relation is the **minimum-burden vertical control**
`r0002c0000|r0003c0000`: its absolute observed pair step is nearly zero
(about **0.44 / 0.66 / 2.11 ADU**) and its MAD-to-step consistency is **1.63–3.23**, so
its ratio is not stable. Qualifying ratios span **0.994589–1.002271**.
`DOMINANT_CAUSE_CONFIRMED` is unreachable in this bounded harness (no pre-registered
restack-dominance criterion exists), so `PARTIAL_CAUSE` is the honest maximum here.

### 7.4 Why this does not authorize R38-B

* The measured mechanism is an almost **constant per-Cell additive level step** applied
  *before* offset. The accepted R34c graph already **estimates and removes additive Cell
  offsets**, so a new shared photometric calibration solver risks replacing an
  already-handled level term.
* This result does **not** explain the unresolved **POST-OFFSET, POST-FEATHER spatial
  residual** (the original inter-Cell disagreement); no dominant cause is demonstrated.
* Consequently `r38b_authorized = false`: R38-B is not scientifically justified by R38-A.

### 7.5 Closure at Gate A and next research question

* **R38 is CLOSED / STOPPED at Gate A.** B/C/D/E are **NOT_RUN / NOT_AUTHORIZED**, not
  "pending": they were not executed, and this evidence does not open them. Any future
  effort must be a **separately pre-registered diagnostic**, never an automatic
  continuation to R38-B.
* Next research question, to target directly: the **post-offset spatial residual** —
  constant vs slope/curvature; offset-estimation bias vs sky structure; support /
  geometry / taper; at **multiple distances** and in **both orientations** — evaluated
  on independent validation data.

### 7.6 No product impact

R38-A is a **diagnostic only**: no product/runtime/default/GUI/manifest/FITS behavior
was changed. No correction is applied anywhere (`correction_applied = false`); no
`canonical_*` / Legacy / SDS code path, threshold, default or user FITS was modified.
