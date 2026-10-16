"""ZM-ZEGRID-R40-B2-D focused tests — OFF/ON product byte-identity (tolerance 0).

Bounded task D proves, by MEASUREMENT on a real synthetic end-to-end product run,
the frozen §8.2 gate of ``docs/refactor/R40_UNEXPLAINED_STRUCTURE_OBSERVABILITY_PLAN.md``:

* with the R40 capture ON but its output DISCARDED, every emitted
  science/coverage/aesthetic/uint16 product FITS is **exactly byte-identical**
  (tolerance exactly ``0``) to capture-OFF for the same synthetic inputs and
  seeds — compared on the FULL FILE BYTES, not headers only;
* the OFF run is itself deterministic (a repeat OFF run is byte-identical), so a
  byte difference could only be attributed to the capture;
* the manifest carries **exactly ONE** additive ``r40_diagnostic_capture`` block
  when ON and NONE when OFF. The manifest is **never** claimed byte-identical.

The manifest embeds PRE-EXISTING *run-observability* fields that are not product
science, in two families — (a) wall-timing / RSS measurements and (b) the
environment-driven scheduler plan (mode, footprint, concurrency, plan candidates,
reprojection path, executor/worker counts, RAM-budget figures, and the per-cell
``mode``/``bound_bytes`` the planner stamps into each cell record). Both families
are pure functions of the free RAM measured at plan time, so they legitimately
differ between runs on a host whose available RAM drifts **with the capture
entirely absent**. This suite therefore takes two measurements:

1. **Uncontrolled, faithful product runs** (:func:`product_runs`): the production
   pipeline runs OFF twice and ON once with only the accepted B2-C concurrency
   seam pinned (so the real process pool runs). Product FITS must be
   byte-identical across all three; every OFF-vs-ON manifest difference beyond
   the single additive block must lie in the declared run-volatile classifier.
2. **Environment-pinned runs** (:func:`pinned_runs`): the memory-availability
   seam is pinned to two DIFFERENT constants, each run OFF and ON. This yields,
   without any capture involvement, a pair of OFF runs with DIFFERENT scheduler
   plans — which proves the classifier is COMPLETE for plan differences — and,
   for each fixed plan, a same-plan OFF/ON pair whose ONLY differences are
   measurement leaves plus the additive block.

A negative control proves the byte gate is not vacuous (a single-byte / single-ulp
change is detected).

Explicit NON-CLAIMS (all ``NOT_RUN``): this suite does NOT claim R40-B2
acceptance, the 38-relation panel qualification, the injected-effect recovery
re-proof, the real product ROI parity, the exact §1.3 held-out-key intersection
or a real Windows run. No real data, no FITS/pixel/image/dataset traversal of
the real corpus, no Windows816/Zealfie/LP341, no product default/GUI/config
change, no correction/model fitting.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.zegrid import parallel as zpar

# The synthetic end-to-end runs write FITS via the production path, which emits
# the PRE-EXISTING astropy ``Card is too long`` VerifyWarning on the synthetic
# WCS header (identical to the accepted R37-D / R40-B2-C e2e tests). Filter ONLY
# that warning so ``-W error`` still guards everything R40-B2-D cares about.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Card is too long:astropy.io.fits.verify.VerifyWarning",
)

R40_BLOCK_KEY = "r40_diagnostic_capture"
_R40_LEAF_SUFFIX = "." + R40_BLOCK_KEY
_SCI = "mosaic_grid_science.fits"
_AESTH = "mosaic_grid_aesthetic.fits"
_COV = "mosaic_grid_coverage.fits"
_U16 = "mosaic_grid_uint16.fits"
_PRODUCT_FITS = (_SCI, _AESTH, _COV, _U16)
_R40_BLOCK_KEYS = {"capture_id", "filename", "sha256", "bytes"}

# 2x2 synthetic corpus with a disjoint (photometrically self-anchoring) pair, so
# the per_cell gauge + feather path (the ONLY seam that carries the R40 capture)
# is genuinely exercised. NEVER the real corpus.
_DISJOINT_CENTERS = [(10.0, 30.0), (10.05, 30.02), (12.0, 30.0), (12.05, 30.02)]
# Two DIFFERENT pinned memory budgets (budget = 0.8 x available): the low value
# yields a STREAM plan, the high value an INMEM plan, for this fixed corpus.
_PIN_LOW = 768 * 1024 ** 2
_PIN_HIGH = 4 * 1024 ** 3
_PLAN_KEYS = ("mode", "cells_in_flight", "per_cell_footprint_bytes",
              "cache_build_workers", "cache_workers_per_cell", "executor", "workers")

# ZM-ZEGRID-R40-B2-I (Nono B2-H L4): the memory seam is PINNED in the uncontrolled
# product runs too, so the scheduler plan is a deterministic function of a CONSTANT
# and host free RAM can no longer flake them (the B2-D/B2-G technique). The real
# production process pool still runs; no assertion is skipped, weakened or deleted.
_PIN_PRODUCT = 4 * 1024 ** 3


# ---------------------------------------------------------------------------
# Synthetic corpus + production run harness (NO real data)
# ---------------------------------------------------------------------------


def _write_frame(path, ra, dec, shape=(400, 600), scale=0.001, seed=0):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra, dec]
    w.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    w.wcs.cd = np.array([[-1, 0], [0, 1]]) * scale
    w.array_shape = shape
    rng = np.random.default_rng(seed)
    data = (100.0 + rng.normal(0, 1, shape)).astype(np.float32)
    fits.PrimaryHDU(data, header=w.to_header()).writeto(path, overwrite=True)


def _write_corpus(root):
    root.mkdir(parents=True, exist_ok=True)
    for i, (ra, dec) in enumerate(_DISJOINT_CENTERS):
        _write_frame(root / f"f{i}.fit", ra, dec, seed=i)
    return root


def _zconfig():
    # Production config: pinned 2x2 layout (deterministic), the per_cell gauge
    # (the R40 seam exists only there) and the FULL finishing set so every
    # product FITS (science + aesthetic float + uint16 render + coverage) is
    # emitted and therefore compared byte-for-byte.
    return SimpleNamespace(
        zegrid_layout="2x2",
        export_aesthetic_fits=True,
        save_final_as_uint16=True,
    )


def _run(inp, out, *, r40_root=None):
    kwargs = dict(zconfig=_zconfig(), cell_gauge_mode="per_cell")
    if r40_root is not None:
        kwargs.update(r40_seam_capture=True, r40_output_root=str(r40_root))
    zz.run_zegrid_mode(str(inp), str(out), **kwargs)


@pytest.fixture(scope="module")
def product_runs(tmp_path_factory):
    """Faithful product runs: OFF twice (OFF determinism) and ON once, on the SAME
    synthetic inputs and seeds.

    The accepted B2-C concurrency seam (``zpar.cells_in_flight``) is pinned to 3 so
    the production PROCESS pool runs and the real worker->parent sparse transport is
    exercised with the capture ON. The memory seam (``zz.available_memory_bytes``) is
    ALSO pinned to a CONSTANT (Nono B2-H L4) so the scheduler plan is deterministic
    and host free RAM can no longer flake the runs; the real production process pool
    still runs and no assertion is skipped, weakened or deleted.
    """
    base = tmp_path_factory.mktemp("r40b2_offon")
    inp = _write_corpus(base / "in")
    real_cif = zpar.cells_in_flight
    real_coords = zz._STACK_R40_WORKER_COORDS
    mp = pytest.MonkeyPatch()
    mp.setattr(zz, "available_memory_bytes", lambda: _PIN_PRODUCT)
    zpar.cells_in_flight = lambda *a, **k: 3
    try:
        off1 = base / "off1"
        _run(inp, off1)
        off2 = base / "off2"
        _run(inp, off2)
        on = base / "on"
        _run(inp, on, r40_root=base / "r40gen")
    finally:
        zpar.cells_in_flight = real_cif
        # the R40 worker transport global may be armed in-process by a serial
        # fallback; restore it so this suite leaks NO module state to others.
        zz._STACK_R40_WORKER_COORDS = real_coords
        mp.undo()
    return SimpleNamespace(base=base, inp=inp, off1=off1, off2=off2, on=on,
                           r40_root=base / "r40gen")


@pytest.fixture(scope="module")
def pinned_runs(tmp_path_factory):
    """Environment-pinned runs: two pinned memory budgets, each run OFF and ON.

    ``available_memory_bytes`` is pinned (per run) so the scheduler plan is a
    deterministic function of a CONSTANT: the low pin yields a STREAM plan and the
    high pin an INMEM plan. This gives (i) two OFF runs with DIFFERENT plans and
    no capture involved — proving the run-volatile classifier is complete for plan
    differences — and (ii) for each fixed plan, a same-plan OFF/ON capture pair.
    """
    base = tmp_path_factory.mktemp("r40b2_offon_pinned")
    inp = _write_corpus(base / "in")
    real_mem = zz.available_memory_bytes
    real_coords = zz._STACK_R40_WORKER_COORDS
    outs = {}
    # ZM-ZEGRID-R41-B: the R41 host-RAM contract counts the UNCAPPED shared parent
    # reserve (340 MiB) PLUS the cache worker footprint in the SIMULTANEOUS phase
    # aggregate. This synthetic host pins SMALL budgets (768 MiB / 4 GiB), so the
    # reserve must be neutralized for the PRODUCT planning to stay feasible while
    # the pinned budgets (and therefore the manifest assertions) keep their exact
    # meaning. The plan flip is preserved: 768 MiB -> stream, 4 GiB -> inmem.
    _mp = pytest.MonkeyPatch()
    _mp.setattr(zpar, "_PER_WORKER_BASELINE_BYTES", 0)
    _mp.setattr(zpar, "_CACHE_PER_WORKER_FOOTPRINT_BYTES", 0)
    _mp.setattr(zpar, "cache_footprint_bytes", lambda px: 0)
    _mp.setattr(zpar, "gauge_footprint_bytes", lambda px: 0)
    try:
        for label, pin in (("lo", _PIN_LOW), ("hi", _PIN_HIGH)):
            zz.available_memory_bytes = lambda p=pin: p
            for cap in ("off", "on"):
                key = f"{label}_{cap}"
                d = base / key
                _run(inp, d, r40_root=(base / f"{key}_gen" if cap == "on" else None))
                outs[key] = d
    finally:
        zz.available_memory_bytes = real_mem
        zz._STACK_R40_WORKER_COORDS = real_coords
        _mp.undo()
    return SimpleNamespace(base=base, inp=inp,
                           lo_off=outs["lo_off"], lo_on=outs["lo_on"],
                           hi_off=outs["hi_off"], hi_on=outs["hi_on"],
                           gen=lambda k: base / f"{k}_gen")


# ---------------------------------------------------------------------------
# Byte/array helpers
# ---------------------------------------------------------------------------


def _sha_file(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _sha_array(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _read_fits_array(path):
    with fits.open(path, memmap=False) as hdul:
        return np.ascontiguousarray(np.asarray(hdul[0].data))


def _fits_evidence(out_dir):
    return {
        name: {"bytes": (out_dir / name).stat().st_size, "sha256": _sha_file(out_dir / name)}
        for name in _PRODUCT_FITS
    }


def _arr_evidence(out_dir):
    return {name: _sha_array(_read_fits_array(out_dir / name)) for name in _PRODUCT_FITS}


def _assert_fits_identity(off_dir, on_dir, *, label=""):
    off = _fits_evidence(off_dir)
    on = _fits_evidence(on_dir)
    for name in _PRODUCT_FITS:
        assert on[name] == off[name], f"{label}{name}: OFF={off[name]} ON={on[name]}"
        assert on[name]["bytes"] > 0
    assert _arr_evidence(off_dir) == _arr_evidence(on_dir)


# ---------------------------------------------------------------------------
# Manifest canonicalisation (isolates PRE-EXISTING volatile observability)
# ---------------------------------------------------------------------------


def _leaves(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        if not obj:
            out[prefix] = "{}"
        for k, v in obj.items():
            out.update(_leaves(v, f"{prefix}.{k}"))
    elif isinstance(obj, list):
        if obj and all(isinstance(e, dict) for e in obj):
            for i, e in enumerate(obj):
                out.update(_leaves(e, f"{prefix}[{i}]"))  # index plan/list records
        else:
            out[prefix] = json.dumps(obj, sort_keys=True, allow_nan=False)
    else:
        out[prefix] = obj
    return out


def _manifest(out_dir):
    return json.loads((pathlib.Path(out_dir) / "zegrid_manifest.json").read_text())


# The run-volatile classifier (measurement + environment-driven scheduler plan) is
# NOT defined here any more: it lives in exactly ONE place —
# ``tests/r40b2_run_volatile.py`` — imported by BOTH this suite and the panel
# suite, so the two copies can no longer drift (Nono B2-F L1). PROVEN complete for
# plan differences (with the capture absent) by
# ``test_run_volatile_classifier_covers_plan_differences`` below.
from r40b2_run_volatile import (  # noqa: E402
    is_measurement_volatile as _is_measurement_volatile,
    is_run_volatile as _is_run_volatile,
    segments as _segments,
)


def _is_r40_leaf(path):
    return path == "." + R40_BLOCK_KEY or path.startswith(_R40_LEAF_SUFFIX + ".")


def _diffs(a_dir, b_dir, predicate):
    """Sorted leaf paths differing from a to b that are NOT classified volatile
    and NOT the additive R40 block (over-excluded fields are ignored here)."""
    a = _leaves(_manifest(a_dir))
    b = _leaves(_manifest(b_dir))
    return sorted(k for k in a
                  if b.get(k, "<<missing>>") != a[k]
                  and not predicate(k) and not _is_r40_leaf(k))


def _canonical(out_dir, predicate):
    man = {k: v for k, v in _manifest(out_dir).items() if k != R40_BLOCK_KEY}
    return {k: v for k, v in _leaves(man).items() if not predicate(k)}


def _plan(out_dir, keys=_PLAN_KEYS):
    pcd = _manifest(out_dir)["per_cell_diagnostics"]
    return {k: pcd.get(k) for k in keys}


# ---------------------------------------------------------------------------
# Uncontrolled fixture — product FITS byte-identity + OFF determinism
# ---------------------------------------------------------------------------


def test_product_artifacts_are_the_full_frozen_set(product_runs):
    for name in _PRODUCT_FITS:
        assert (product_runs.off1 / name).is_file(), name
        assert (product_runs.on / name).is_file(), name
    # the two non-product outputs exist but are deliberately NOT byte-compared
    assert (product_runs.off1 / "zegrid_run.log").is_file()
    assert (product_runs.off1 / "__zegrid_cache__").is_dir()


def test_off_run_is_deterministic(product_runs):
    """Step 4: OFF vs OFF must be byte-identical, so any OFF/ON difference can
    only be attributed to the capture (never to run-to-run nondeterminism)."""
    assert _fits_evidence(product_runs.off1) == _fits_evidence(product_runs.off2)
    assert _arr_evidence(product_runs.off1) == _arr_evidence(product_runs.off2)


def test_offon_product_fits_are_byte_identical(product_runs):
    """The frozen §8.2 gate: capture ON (output discarded) vs OFF -> every
    science/coverage/aesthetic/uint16 product FITS is byte-identical, full file
    bytes, tolerance exactly 0."""
    _assert_fits_identity(product_runs.off1, product_runs.on)


def test_offon_product_arrays_are_equal(product_runs):
    """Array-level restatement of the byte gate (identical bytes => identical
    arrays); guards against a coincidental whole-file match."""
    off_arrays = {n: _read_fits_array(product_runs.off1 / n) for n in _PRODUCT_FITS}
    on_arrays = {n: _read_fits_array(product_runs.on / n) for n in _PRODUCT_FITS}
    for name in _PRODUCT_FITS:
        np.testing.assert_array_equal(off_arrays[name], on_arrays[name])
        assert _sha_array(off_arrays[name]) == _sha_array(on_arrays[name])


def test_offon_science_header_carries_identical_embedded_sha(product_runs):
    """The product header embeds the science array SHA-256 (``SCIHASH``); it must
    be identical ON vs OFF (the header is part of the file bytes)."""
    def _scihash(out_dir):
        with fits.open(out_dir / _SCI, memmap=False) as hdul:
            return hdul[0].header["SCIHASH"], hdul[0].header["SCIROLE"]
    assert _scihash(product_runs.off1) == _scihash(product_runs.on)


def test_byte_gate_is_sensitive_negative_control(product_runs, tmp_path):
    """R3: the ON==OFF comparison is NOT vacuous. A single flipped byte of the
    FITS file and a single 1-ULP change of one array sample are BOTH detected, so
    the gate would fail if the capture changed the product bytes."""
    src = product_runs.off1 / _SCI
    raw = bytearray(src.read_bytes())
    raw[-1] ^= 0x01
    mutated = tmp_path / "mutated.fits"
    mutated.write_bytes(bytes(raw))
    assert _sha_file(mutated) != _sha_file(src)

    arr = _read_fits_array(src)
    arr2 = arr.copy()
    flat = arr2.reshape(-1)
    i = int(np.argmax(flat))
    flat[i] = np.nextafter(flat[i], np.float32(np.inf))
    assert not np.array_equal(arr, arr2)
    assert _sha_array(arr) != _sha_array(arr2)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(arr, arr2)


def test_off_has_no_r40_block_and_no_generation(product_runs):
    for off in (product_runs.off1, product_runs.off2):
        assert R40_BLOCK_KEY not in _manifest(off)
        assert sorted(p.name for p in off.glob("r40_capture_*")) == []


def test_on_manifest_additive_block_is_exactly_one(product_runs):
    off_keys = set(_manifest(product_runs.off1))
    on_keys = set(_manifest(product_runs.on))
    assert on_keys - off_keys == {R40_BLOCK_KEY}
    assert off_keys - on_keys == set()
    block = _manifest(product_runs.on)[R40_BLOCK_KEY]
    assert set(block) == _R40_BLOCK_KEYS
    assert block["capture_id"] and block["sha256"] and block["bytes"] > 0


def test_manifest_is_never_byte_identical(product_runs):
    """Explicit NON-CLAIM: the manifest is NOT byte-identical (the additive block
    is present ON; the volatile fields differ run-to-run)."""
    on_bytes = (product_runs.on / "zegrid_manifest.json").read_bytes()
    off_bytes = (product_runs.off1 / "zegrid_manifest.json").read_bytes()
    assert on_bytes != off_bytes


def test_pre_existing_manifest_fields_canonical_identical(product_runs):
    """Every pre-existing manifest field that is not run-observability is
    canonical-identical across OFF and ON (and OFF vs OFF) — the plan fields are
    excluded by the declared, evidence-backed classifier."""
    c_off1 = _canonical(product_runs.off1, _is_run_volatile)
    c_off2 = _canonical(product_runs.off2, _is_run_volatile)
    c_on = _canonical(product_runs.on, _is_run_volatile)
    assert c_off1 == c_off2
    assert c_on == c_off1


def test_every_offon_manifest_difference_is_pre_existing_volatile(product_runs):
    """The ONLY manifest difference ON vs OFF, beyond the single additive block,
    is confined to pre-existing run-volatile observability fields (measurement +
    environment-driven scheduler plan) — never a science/product field."""
    assert _diffs(product_runs.off1, product_runs.on, _is_run_volatile) == []
    off1 = _leaves(_manifest(product_runs.off1))
    on = _leaves(_manifest(product_runs.on))
    assert {k for k in on if k not in off1} == {
        f".{R40_BLOCK_KEY}.capture_id", f".{R40_BLOCK_KEY}.filename",
        f".{R40_BLOCK_KEY}.sha256", f".{R40_BLOCK_KEY}.bytes",
    }
    assert {k for k in off1 if k not in on} == set()


def test_run_volatile_classifier_covers_off_off_differences(product_runs):
    """Evidence-backed: the classifier covers EVERY field that differs between two
    OFF runs (else the canonical projection would hide a real nondeterminism)."""
    assert _diffs(product_runs.off1, product_runs.off2, _is_run_volatile) == []


def test_on_capture_generation_verifies_and_is_cross_linked(product_runs):
    from zemosaic.core.zegrid import r40_capture as rc

    gens = sorted(p.name for p in product_runs.r40_root.glob("r40_capture_*"))
    assert len(gens) == 1
    report = rc.verify_generation(product_runs.r40_root / gens[0],
                                 expected_production_wired=True)
    assert report["ok"] is True and report["problems"] == []
    assert report["n_records"] > 0

    block = _manifest(product_runs.on)[R40_BLOCK_KEY]
    assert block["filename"] == gens[0]
    index_path = product_runs.r40_root / gens[0] / "index.json"
    assert block["sha256"] == _sha_file(index_path)
    assert block["bytes"] == index_path.stat().st_size


def test_on_run_really_used_the_production_process_pool(product_runs):
    """Guard: the ON/OFF comparison was made on the REAL pooled product path (the
    capture transport is a worker->parent path), not a serial shortcut."""
    pcd = _manifest(product_runs.on)["per_cell_diagnostics"]
    assert pcd["executor"] == "process"
    assert int(pcd["workers"]) >= 2
    assert int(pcd["tasks"]) >= 3


# ---------------------------------------------------------------------------
# Environment-pinned fixture — causality, classifier completeness, strong claim
# ---------------------------------------------------------------------------


def test_pinned_plans_differ_so_the_plan_is_memory_driven(pinned_runs):
    """R1: with the capture ABSENT, changing ONLY the memory input flips the plan
    (STREAM vs INMEM). So a plan change between runs is environment-driven, not
    capture-driven."""
    lo, hi = _plan(pinned_runs.lo_off), _plan(pinned_runs.hi_off)
    assert lo != hi, (lo, hi)
    assert lo["mode"] == "stream" and hi["mode"] == "inmem", (lo, hi)
    # and the pinned memory values are exactly what the plan saw
    assert _manifest(pinned_runs.lo_off)["per_cell_diagnostics"]["available_bytes"] == _PIN_LOW
    assert _manifest(pinned_runs.hi_off)["per_cell_diagnostics"]["available_bytes"] == _PIN_HIGH


def test_pinned_plan_is_identical_off_vs_on(pinned_runs):
    """R1: for a FIXED memory input, capture ON and OFF produce the IDENTICAL
    scheduler plan — direct proof the capture cannot change the plan."""
    assert _plan(pinned_runs.lo_off) == _plan(pinned_runs.lo_on)
    assert _plan(pinned_runs.hi_off) == _plan(pinned_runs.hi_on)


def test_run_volatile_classifier_covers_plan_differences(pinned_runs):
    """R2 evidence-backed completeness: every field that differs between two OFF
    runs with DIFFERENT plans (capture absent) is classified run-volatile — with
    the same for the capture-ON variant of the same plan flip."""
    assert _diffs(pinned_runs.lo_off, pinned_runs.hi_off, _is_run_volatile) == []
    assert _diffs(pinned_runs.lo_on, pinned_runs.hi_on, _is_run_volatile) == []


def test_cells_records_differ_only_in_plan_metadata(pinned_runs):
    """R4: ``.cells`` is the list of PER-CELL records (cell_id, row, col, status,
    the chosen ``mode``, ``n_contributors``, ``cell_gauge_mode``, ``backend_used``,
    the reference-frame provenance fields, ``excluded`` and the planner's
    ``bound_bytes``). Under a plan flip the ONLY per-cell leaves that differ are
    the two PLAN-METADATA leaves the scheduler stamps (``mode``, ``bound_bytes``);
    every science/index per-cell field is identical. The classifier therefore
    excludes ONLY those two leaf names — never the whole ``.cells`` subtree."""
    a = _leaves(_manifest(pinned_runs.lo_off))
    b = _leaves(_manifest(pinned_runs.hi_off))
    diff = sorted(k for k in a if k.startswith(".cells[") and b.get(k, "<m>") != a[k])
    assert diff  # the plan flip genuinely stamps the per-cell records
    assert {k.rsplit(".", 1)[-1] for k in diff} == {"mode", "bound_bytes"}
    # and neither of those two leaves is product science; both are classified
    assert all(_is_run_volatile(k) for k in diff)


def test_pinned_product_fits_are_byte_identical(pinned_runs):
    """Product bytes identical at BOTH pinned plans, OFF vs ON."""
    _assert_fits_identity(pinned_runs.lo_off, pinned_runs.lo_on, label="pin-low ")
    _assert_fits_identity(pinned_runs.hi_off, pinned_runs.hi_on, label="pin-high ")
    # and across the plan flip too (the plan does not touch the product bytes)
    _assert_fits_identity(pinned_runs.lo_off, pinned_runs.hi_off, label="plan-flip ")


def test_same_plan_capture_pair_diff_is_only_measurement_noise(pinned_runs):
    """The strong, confound-free manifest claim: with the plan held FIXED (memory
    pinned), the only pre-existing manifest fields differing OFF vs ON are genuine
    wall-timing / RSS measurement leaves, at BOTH plans; exactly one additive block
    is present ON and none OFF."""
    for off, on in ((pinned_runs.lo_off, pinned_runs.lo_on),
                    (pinned_runs.hi_off, pinned_runs.hi_on)):
        assert _diffs(off, on, _is_measurement_volatile) == []
        assert _canonical(off, _is_measurement_volatile) == \
            _canonical(on, _is_measurement_volatile)
        assert R40_BLOCK_KEY not in _manifest(off)
        assert set(_manifest(on)) - set(_manifest(off)) == {R40_BLOCK_KEY}
        assert set(_manifest(off)) - set(_manifest(on)) == set()
