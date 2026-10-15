"""ZM-ZEGRID-R40-B2-F focused tests — 38-relation PANEL OFF/ON qualification.

Bounded task F proves, by MEASUREMENT on a real synthetic end-to-end capture:

* the frozen adjacency rule yields **exactly the full 38-relation panel** for the
  frozen 6x4 Cell grid (20 horizontal + 18 vertical), asserted explicitly — never
  assumed;
* running the PRODUCTION per-cell path with the R40 capture ON and OFF on that
  grid: **(a)** every emitted science/coverage/aesthetic/uint16 product FITS and
  its array stay **exactly byte-identical** (tolerance exactly ``0``); **(b)** all
  38 relations are captured, complete (all 3 channels) and **independently
  verifiable** from on-disk bytes; **(c)** an INCOMPLETE panel (one relation
  removed) is **refused** by the frozen completeness gate and is **never
  promoted**; **(d)** the ON manifest carries **exactly one** additive
  ``r40_diagnostic_capture`` block and the OFF manifest **none**.

It additionally re-states, on the PANEL runs, the frozen run-volatile-classifier
obligation (Nono B2-E §3, now an explicit B2 acceptance criterion in §8): every
manifest leaf that differs between two capture-OFF runs AND between the capture
OFF/ON runs is confined to the declared, evidence-backed run-volatile classifier,
with a negative control proving the check **fails loudly** on an unclassified
differing leaf, that the canonical projection excludes only the enumerated leaves
(never a product field), and that the ``.cells`` subtree is never
blanket-excluded.

Explicit NON-CLAIMS (all ``NOT_RUN``): this is a **SYNTHETIC panel
qualification**. The real **Windows816 38-relation layout is NOT exercised** and
the real Windows run remains a **HUMAN GATE**. This suite does NOT claim R40-B2
acceptance, the injected-effect recovery re-proof, the real product ROI parity,
the exact §1.3 held-out-key intersection or a real Windows run. No real data; no
FITS/pixel/image/dataset traversal of the real corpus; no Windows816/Zealfie/
LP341; no product default/GUI/config change; no correction/model fitting; the
capture stays default-OFF.
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
from zemosaic.core.zegrid import cell_offset as zco
from zemosaic.core.zegrid import parallel as zpar
from zemosaic.core.zegrid import r40_capture as rc

# The synthetic end-to-end runs write FITS via the production path, which emits
# the PRE-EXISTING astropy ``Card is too long`` VerifyWarning on the synthetic
# WCS header (identical to the accepted R37-D / R40-B2-B/C/D e2e tests). Filter
# ONLY that warning so ``-W error`` still guards everything R40-B2-F cares about.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Card is too long:astropy.io.fits.verify.VerifyWarning",
)

# --- frozen panel geometry ---------------------------------------------------
_NX, _NY = 6, 4
_PANEL_RELATIONS = (_NX - 1) * _NY + _NX * (_NY - 1)  # 20 h + 18 v = 38
_PANEL_H = (_NX - 1) * _NY
_PANEL_V = _NX * (_NY - 1)

R40_BLOCK_KEY = "r40_diagnostic_capture"
_SCI = "mosaic_grid_science.fits"
_AESTH = "mosaic_grid_aesthetic.fits"
_COV = "mosaic_grid_coverage.fits"
_U16 = "mosaic_grid_uint16.fits"
_PRODUCT_FITS = (_SCI, _AESTH, _COV, _U16)
_R40_BLOCK_KEYS = {"capture_id", "filename", "sha256", "bytes"}

# 5 x 2 synthetic frames tiling the field so EVERY one of the 24 Cells of the
# pinned 6x4 grid is covered (a smaller corpus leaves uncovered Cells without an
# M9 reference frame, which correctly REFUSES). NEVER the real corpus.
_PANEL_RAS = [9.7 + 0.6 * i for i in range(5)]
_PANEL_DECS = [30.0, 30.02]
_PANEL_LAYOUT = "6x4"

# ZM-ZEGRID-R40-B2-I (Nono B2-H L4): the memory seam is PINNED so the scheduler
# plan is a deterministic function of a CONSTANT and host free RAM can no longer
# flake these runs (the B2-D/B2-G technique). The real production process pool
# still runs; no assertion is skipped, weakened or deleted.
_PIN_MEM_BYTES = 4 * 1024 ** 3


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


def _write_panel_corpus(root):
    root.mkdir(parents=True, exist_ok=True)
    k = 0
    for ra in _PANEL_RAS:
        for dec in _PANEL_DECS:
            _write_frame(root / f"f{k:03d}.fit", ra, dec, seed=k)
            k += 1
    return root


def _zconfig():
    # Pinned 6x4 layout (deterministic full 38-relation panel), the per_cell gauge
    # (the only seam that carries the R40 capture) and the FULL finishing set so
    # every product FITS (science + aesthetic + uint16 render + coverage) is
    # emitted and compared byte-for-byte.
    return SimpleNamespace(
        zegrid_layout=_PANEL_LAYOUT,
        export_aesthetic_fits=True,
        save_final_as_uint16=True,
    )


def _run(inp, out, *, r40_root=None):
    kwargs = dict(zconfig=_zconfig(), cell_gauge_mode="per_cell")
    if r40_root is not None:
        kwargs.update(r40_seam_capture=True, r40_output_root=str(r40_root))
    zz.run_zegrid_mode(str(inp), str(out), **kwargs)


@pytest.fixture(scope="module")
def panel_runs(tmp_path_factory):
    """Faithful PANEL runs: OFF twice (OFF determinism) and ON once.

    The accepted B2-C concurrency seam (``zpar.cells_in_flight``) is pinned to 3 so
    the production PROCESS pool runs and the real worker->parent sparse transport is
    exercised with the capture ON. The memory seam (``zz.available_memory_bytes``) is
    ALSO pinned to a CONSTANT (Nono B2-H L4) so the scheduler plan is deterministic
    and host free RAM can no longer flake the runs; the real production process pool
    still runs and no assertion is skipped, weakened or deleted.
    """
    base = tmp_path_factory.mktemp("r40b2_panel")
    inp = _write_panel_corpus(base / "in")
    real_cif = zpar.cells_in_flight
    real_coords = zz._STACK_R40_WORKER_COORDS
    # ZM-ZEGRID-R40-B2-I (Nono B2-H L4): pin the memory seam (monkeypatch) so the
    # scheduler plan is a deterministic function of a CONSTANT — host free RAM can
    # no longer flake the panel runs. The real production process pool still runs.
    mp = pytest.MonkeyPatch()
    mp.setattr(zz, "available_memory_bytes", lambda: _PIN_MEM_BYTES)
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


# ---------------------------------------------------------------------------
# Byte / array / manifest helpers
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


def _manifest(out_dir):
    return json.loads((pathlib.Path(out_dir) / "zegrid_manifest.json").read_text())


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


# The run-volatile classifier (measurement + environment-driven scheduler plan) is
# NOT defined here any more: it lives in exactly ONE place —
# ``tests/r40b2_run_volatile.py`` — imported by BOTH this suite and the B2-D suite,
# so the two copies can no longer drift (Nono B2-F L1). Its completeness for plan
# differences is proven by measurement in
# ``tests/test_zegrid_r40b2_offon_identity.py`` (two capture-OFF runs with FORCED
# DIFFERENT plans, capture absent) — the obligation frozen by §8 clause 12.
from r40b2_run_volatile import (  # noqa: E402
    is_measurement_volatile as _is_measurement_volatile,
    is_run_volatile as _is_run_volatile,
    segments as _segments,
)


def _is_r40_leaf(path):
    return path == "." + R40_BLOCK_KEY or path.startswith("." + R40_BLOCK_KEY + ".")


def _diffs_leaves(a_leaves, b_leaves, predicate):
    """Sorted differing leaf paths NOT classified volatile and NOT the additive
    R40 block (so an over-broad classifier is ignored *here*, but the completeness
    of the classifier itself is asserted separately)."""
    return sorted(k for k in a_leaves
                  if b_leaves.get(k, "<<missing>>") != a_leaves[k]
                  and not predicate(k) and not _is_r40_leaf(k))


def _diffs(a_dir, b_dir, predicate):
    return _diffs_leaves(_leaves(_manifest(a_dir)), _leaves(_manifest(b_dir)), predicate)


def _canonical(out_dir, predicate):
    man = {k: v for k, v in _manifest(out_dir).items() if k != R40_BLOCK_KEY}
    return {k: v for k, v in _leaves(man).items() if not predicate(k)}


# ---------------------------------------------------------------------------
# Generation / shard helpers (independent verification path)
# ---------------------------------------------------------------------------


def _generation_dirs(r40_root):
    return sorted(p for p in pathlib.Path(r40_root).glob("r40_capture_*") if p.is_dir())


def _on_generation(panel_runs):
    gens = _generation_dirs(panel_runs.r40_root)
    assert len(gens) == 1, gens
    return gens[0]


def _index(gen_dir):
    return json.loads((gen_dir / "index.json").read_text())


def _decode_records(raw):
    """Decode one shard's frozen structured bytes back to record mappings."""
    array = np.frombuffer(raw, dtype=rc.R40_RECORD_DTYPE)
    records = []
    for row in array:
        records.append({
            name: (row[name].decode("ascii") if name == "sample_key" else row[name].item())
            for name in rc.R40_RECORD_NAMES
        })
    return records


def _shard_from_entry(gen_dir, entry):
    raw = (gen_dir / entry["filename"]).read_bytes()
    return rc.RelationShard(
        u=entry["u"], v=entry["v"], orientation=entry["orientation"],
        header=entry["header"], records=tuple(_decode_records(raw)),
    )


def _shard_meta(entry, shard):
    data = shard.data()
    return {
        "relation_id": shard.relation_id,
        "count": len(shard.records),
        "bytes": len(data),
        "sha256": rc.sha256_bytes(data),
    }


def _panel_edges():
    return list(zco.grid_adjacency(_NX, _NY))


def _cell_row_col(cid):
    assert cid[0] == "r" and len(cid) == 10, cid
    return int(cid[1:5]), int(cid[6:10])


# ---------------------------------------------------------------------------
# 1. The exact frozen 38-relation count is ASSERTED, never assumed
# ---------------------------------------------------------------------------


def test_frozen_panel_relation_count_is_exactly_38():
    """The frozen adjacency rule on a 6x4 Cell grid yields EXACTLY 38 geometric
    relations = 20 horizontal + 18 vertical (explicit arithmetic AND the real
    production ``grid_adjacency`` helper — not an assumption)."""
    horizontal = (_NX - 1) * _NY
    vertical = _NX * (_NY - 1)
    assert (horizontal, vertical) == (20, 18)
    assert horizontal + vertical == 38 == _PANEL_RELATIONS

    edges = _panel_edges()
    assert len(edges) == 38
    # deterministic order, no duplicates, no self-loops
    assert len(set(edges)) == 38
    assert len(set(edges)) == len(edges)
    assert all(u != v for u, v in edges)

    # the production helper reproduces the frozen arithmetic and orientations
    def _orient(u, v):
        ru, cu = _cell_row_col(u)
        rv, cv = _cell_row_col(v)
        assert ru == rv or cu == cv
        return "h" if ru == rv else "v"

    kinds = [_orient(u, v) for u, v in edges]
    assert kinds.count("h") == 20 and kinds.count("v") == 18
    # every cell id is inside the 6x4 indexed rectangle
    for u, v in edges:
        for cid in (u, v):
            row, col = _cell_row_col(cid)
            assert 0 <= row < _NY and 0 <= col < _NX


# ---------------------------------------------------------------------------
# 2. The end-to-end synthetic capture yields EXACTLY the 38-relation panel
# ---------------------------------------------------------------------------


def test_e2e_capture_exposes_exactly_the_38_relation_panel(panel_runs):
    gen = _on_generation(panel_runs)
    index = _index(gen)
    expected = {rc.relation_id_for(u, v): ("h" if _cell_row_col(u)[0] == _cell_row_col(v)[0] else "v")
                for u, v in _panel_edges()}
    got = {e["relation_id"]: e["orientation"] for e in index["shards"]}
    assert len(index["shards"]) == index["n_shards"] == _PANEL_RELATIONS == 38
    assert set(got) == set(expected)
    assert got == expected
    assert sum(1 for o in got.values() if o == "h") == 20
    assert sum(1 for o in got.values() if o == "v") == 18


def test_every_captured_relation_is_complete_on_all_three_channels(panel_runs):
    """Completeness (§3.1): each of the 38 relations carries all 3 channels and a
    positive record count, and the panel totals are internally consistent."""
    gen = _on_generation(panel_runs)
    index = _index(gen)
    total = 0
    for entry in index["shards"]:
        raw = (gen / entry["filename"]).read_bytes()
        array = np.frombuffer(raw, dtype=rc.R40_RECORD_DTYPE)
        channels = {int(c) for c in np.unique(array["channel"])}
        assert channels == {0, 1, 2}, (entry["relation_id"], channels)
        assert entry["count"] == len(array) > 0
        assert entry["count"] % 3 == 0
        total += entry["count"]
    assert total == index["n_records"]
    # the 114/114 relation-channel panel (38 relations x 3 channels)
    assert index["n_shards"] * 3 == 114


# ---------------------------------------------------------------------------
# 3. Every relation is INDEPENDENTLY verifiable from on-disk bytes
# ---------------------------------------------------------------------------


def test_verify_generation_ok_for_the_panel(panel_runs):
    gen = _on_generation(panel_runs)
    report = rc.verify_generation(gen, expected_production_wired=True)
    assert report["ok"] is True and report["problems"] == []
    assert report["n_shards"] == 38
    assert report["checked_records"] == report["n_records"] > 0


def test_every_relation_is_independently_verifiable(panel_runs):
    """Independent re-derivation from the on-disk bytes: per-shard hash/bytes/
    count, frozen dtype decode, per-record sample key, pairset digest, frozen
    header validation, relation id and orientation — none of it trusted from the
    index alone."""
    gen = _on_generation(panel_runs)
    index = _index(gen)
    for entry in index["shards"]:
        raw = (gen / entry["filename"]).read_bytes()
        # --- hash / size / count binding -----------------------------------
        assert rc.sha256_bytes(raw) == entry["sha256"]
        assert len(raw) == entry["bytes"]
        assert entry["count"] * rc.R40_RECORD_ITEMSIZE == entry["bytes"]
        array = np.frombuffer(raw, dtype=rc.R40_RECORD_DTYPE)
        assert len(array) == entry["count"]
        # --- frozen record ordering + per-record identity -------------------
        order = list(zip(array["channel"].tolist(), array["gy"].tolist(), array["gx"].tolist()))
        assert order == sorted(order)
        keys = []
        for row in array:
            rid = entry["relation_id"]
            key = rc.sample_key(rid, int(row["channel"]), int(row["gx"]), int(row["gy"]))
            assert row["sample_key"].decode("ascii") == key
            keys.append(key)
            assert bool(np.isfinite(np.asarray(row["sci_u_pre"], dtype=np.float64)))
        assert len(set(keys)) == len(keys)  # duplicate sample key refused by the core
        assert rc.pairset_digest(keys) == entry["pairset_digest"]
        # --- frozen header + relation identity ------------------------------
        assert entry["relation_id"] == rc.relation_id_for(entry["u"], entry["v"])
        checked = rc.validate_relation_header(
            entry["header"], u=entry["u"], v=entry["v"], orientation=entry["orientation"]
        )
        assert checked["orientation"] == entry["orientation"]
        assert checked["relation_id"] == entry["relation_id"]


def test_captured_relations_match_the_frozen_grid_adjacency(panel_runs):
    """The captured panel is exactly the frozen adjacency set (no extra, no
    missing), and each shard's (u, v) is an edge of the production grid."""
    gen = _on_generation(panel_runs)
    index = _index(gen)
    edges = {(u, v) for u, v in _panel_edges()}
    captured = {(e["u"], e["v"]) for e in index["shards"]}
    assert captured == edges


# ---------------------------------------------------------------------------
# 4. OFF/ON product byte-identity (tolerance exactly 0) on the panel grid
# ---------------------------------------------------------------------------


def test_panel_product_artifacts_are_the_full_frozen_set(panel_runs):
    for name in _PRODUCT_FITS:
        assert (panel_runs.off1 / name).is_file(), name
        assert (panel_runs.on / name).is_file(), name


def test_off_run_is_deterministic_on_the_panel(panel_runs):
    """OFF vs OFF must be byte-identical, so any OFF/ON difference can only be
    attributed to the capture (never to run-to-run nondeterminism)."""
    assert _fits_evidence(panel_runs.off1) == _fits_evidence(panel_runs.off2)
    assert _arr_evidence(panel_runs.off1) == _arr_evidence(panel_runs.off2)


def test_panel_offon_product_fits_are_byte_identical(panel_runs):
    """The frozen §8.2 gate on the full 38-relation panel: capture ON (output
    discarded) vs OFF -> every product FITS is byte-identical, full file bytes,
    tolerance exactly 0."""
    off = _fits_evidence(panel_runs.off1)
    on = _fits_evidence(panel_runs.on)
    for name in _PRODUCT_FITS:
        assert on[name] == off[name], f"{name}: OFF={off[name]} ON={on[name]}"
        assert on[name]["bytes"] > 0
    assert _arr_evidence(panel_runs.off1) == _arr_evidence(panel_runs.on)


def test_panel_offon_product_arrays_are_equal(panel_runs):
    off_arrays = {n: _read_fits_array(panel_runs.off1 / n) for n in _PRODUCT_FITS}
    on_arrays = {n: _read_fits_array(panel_runs.on / n) for n in _PRODUCT_FITS}
    for name in _PRODUCT_FITS:
        np.testing.assert_array_equal(off_arrays[name], on_arrays[name])
        assert _sha_array(off_arrays[name]) == _sha_array(on_arrays[name])


def test_panel_byte_gate_is_sensitive_negative_control(panel_runs, tmp_path):
    """The ON==OFF comparison is NOT vacuous: a single flipped byte and a single
    1-ULP array change are BOTH detected."""
    src = panel_runs.off1 / _SCI
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
    assert _sha_array(arr) != _sha_array(arr2)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(arr, arr2)


# ---------------------------------------------------------------------------
# 5. Additive manifest block: exactly ONE ON / NONE OFF (+ cross-link)
# ---------------------------------------------------------------------------


def test_panel_off_has_no_r40_block_and_no_generation(panel_runs):
    for off in (panel_runs.off1, panel_runs.off2):
        assert R40_BLOCK_KEY not in _manifest(off)
        assert sorted(p.name for p in off.glob("r40_capture_*")) == []


def test_panel_on_manifest_additive_block_is_exactly_one(panel_runs):
    off_keys = set(_manifest(panel_runs.off1))
    off2_keys = set(_manifest(panel_runs.off2))
    on_keys = set(_manifest(panel_runs.on))
    assert off_keys == off2_keys
    assert on_keys - off_keys == {R40_BLOCK_KEY}
    assert off_keys - on_keys == set()
    block = _manifest(panel_runs.on)[R40_BLOCK_KEY]
    assert set(block) == _R40_BLOCK_KEYS
    assert block["capture_id"] and block["sha256"] and block["bytes"] > 0


def test_panel_on_generation_verifies_and_is_cross_linked(panel_runs):
    """The single additive block cross-links the published generation to the
    manifest on the frozen identity keys (sha256/bytes == the on-disk index)."""
    gen = _on_generation(panel_runs)
    index_path = gen / "index.json"
    block = _manifest(panel_runs.on)[R40_BLOCK_KEY]
    assert block["filename"] == gen.name
    assert block["sha256"] == _sha_file(index_path)
    assert block["bytes"] == index_path.stat().st_size


# ---------------------------------------------------------------------------
# 6. Frozen classifier obligation re-stated on the PANEL runs (+ fail-loud)
# ---------------------------------------------------------------------------


def test_panel_manifest_is_never_byte_identical(panel_runs):
    """Explicit NON-CLAIM: the manifest is NOT byte-identical (the additive block
    is present ON; the run-volatile fields differ run-to-run)."""
    on_bytes = (panel_runs.on / "zegrid_manifest.json").read_bytes()
    off_bytes = (panel_runs.off1 / "zegrid_manifest.json").read_bytes()
    assert on_bytes != off_bytes


def test_panel_offoff_and_offon_diff_is_confined_to_the_declared_classifier(panel_runs):
    """Frozen §8 clause 12 on the panel: every manifest leaf that differs between
    two capture-OFF runs AND between the capture OFF/ON runs is confined to the
    declared, evidence-backed run-volatile classifier; the ONLY added leaves are
    the four additive block leaves."""
    assert _diffs(panel_runs.off1, panel_runs.off2, _is_run_volatile) == []
    assert _diffs(panel_runs.off1, panel_runs.on, _is_run_volatile) == []

    off_leaves = _leaves(_manifest(panel_runs.off1))
    on_leaves = _leaves(_manifest(panel_runs.on))
    assert {k for k in on_leaves if k not in off_leaves} == {
        f".{R40_BLOCK_KEY}.capture_id", f".{R40_BLOCK_KEY}.filename",
        f".{R40_BLOCK_KEY}.sha256", f".{R40_BLOCK_KEY}.bytes",
    }
    assert {k for k in off_leaves if k not in on_leaves} == set()

    # canonical projection: identical across OFF/OFF and OFF/ON
    c_off1 = _canonical(panel_runs.off1, _is_run_volatile)
    c_off2 = _canonical(panel_runs.off2, _is_run_volatile)
    c_on = _canonical(panel_runs.on, _is_run_volatile)
    assert c_off1 == c_off2 == c_on


def test_panel_classifier_never_excludes_a_product_field_or_the_cells_subtree(panel_runs):
    """The canonical projection excludes ONLY the enumerated run-volatile leaves:
    no product field is excluded, and the ``.cells`` subtree is never
    blanket-excluded (its science/index/status per-cell leaves survive)."""
    canonical = _canonical(panel_runs.off1, _is_run_volatile)
    # `.cells` records are present and are NOT blanket-excluded
    cell_leaves = [k for k in canonical if k.startswith(".cells[")]
    assert cell_leaves, "the .cells subtree must survive the canonical projection"
    # only the two frozen plan-metadata leaf NAMES are ever classified volatile
    excluded_cell = sorted(k for k in _leaves(_manifest(panel_runs.off1))
                           if k.startswith(".cells[") and _is_run_volatile(k))
    assert {k.rsplit(".", 1)[-1] for k in excluded_cell} <= {"mode", "bound_bytes"}
    # a real product/science field is present in the canonical projection
    assert any("cell_feather" in k or "science" in k or "coverage" in k for k in canonical)


def test_panel_classifier_check_fails_loud_on_unclassified_differing_leaf():
    """Fail-loud negative control (frozen §8 clause 12): a differing leaf that is
    NOT classified volatile is DETECTED (so the check above cannot pass by
    silently normalising an unclassified diff), while a classified leaf is
    ignored."""
    base = {"cells": [{"cell_id": "r0000c0000", "mode": "stream",
                       "bound_bytes": 100, "n_contributors": 3}],
            "science_sha": "a" * 64}
    same = json.loads(json.dumps(base))
    assert _diffs_leaves(_leaves(base), _leaves(same), _is_run_volatile) == []

    # an UNCLASSIFIED product/science leaf differs -> detected (non-empty)
    unclassified = json.loads(json.dumps(base))
    unclassified["science_sha"] = "b" * 64
    diffs = _diffs_leaves(_leaves(base), _leaves(unclassified), _is_run_volatile)
    assert diffs == [".science_sha"]

    # a PLAN-METADATA leaf that DID differ ON vs OFF is classified -> ignored
    classified = json.loads(json.dumps(base))
    classified["cells"][0]["mode"] = "inmem"
    assert _diffs_leaves(_leaves(base), _leaves(classified), _is_run_volatile) == []

    # a differing per-cell SCIENCE field is never excused by the classifier
    science_diff = json.loads(json.dumps(base))
    science_diff["cells"][0]["n_contributors"] = 4
    assert _diffs_leaves(_leaves(base), _leaves(science_diff), _is_run_volatile) == \
        [".cells[0].n_contributors"]


# ---------------------------------------------------------------------------
# 7. An INCOMPLETE panel is REFUSED and is NEVER promoted
# ---------------------------------------------------------------------------


def test_incomplete_panel_is_refused_and_does_not_promote(panel_runs, tmp_path):
    """Remove ONE relation from the captured panel:
    * the frozen completeness gate (``assert_checkpoint_complete`` — the §9
      ``INCOMPLETE_PANEL`` / "no final COMMIT while partial" rule) REFUSES it and
      names the missing relation;
    * the exact-resume primitive reports the panel as ``RESUMABLE`` with the
      missing relation remaining — never ``COMPLETE``;
    * a physically written 37-shard generation is self-consistent on disk
      (``verify_generation`` ok) yet the panel gate still REFUSES it -> a partial
      panel is NOT promotable. The complete 38-relation panel passes.
    """
    gen = _on_generation(panel_runs)
    index = _index(gen)
    identity = index["identity"]
    entries = index["shards"]
    assert len(entries) == 38
    all_ids = [e["relation_id"] for e in entries]

    shards = {e["relation_id"]: _shard_from_entry(gen, e) for e in entries}
    # reconstruction is lossless (so the reduced generation is faithful)
    for e in entries:
        shard = shards[e["relation_id"]]
        assert rc.sha256_bytes(shard.data()) == e["sha256"]

    metas = [_shard_meta(e, shards[e["relation_id"]]) for e in entries]

    # --- complete panel: the gate PASSES ---------------------------------
    full_cp = rc.build_checkpoint(identity=identity, completed_shards=metas,
                                  expected_production_wired=True)
    rc.assert_checkpoint_complete(full_cp, expected_relation_ids=all_ids)
    full_resume = rc.check_resume(full_cp, identity=identity, completed_shards=metas,
                                  expected_relation_ids=all_ids,
                                  expected_production_wired=True)
    assert full_resume["state"] == rc.COMPLETE and full_resume["remaining"] == []

    # --- INCOMPLETE panel: drop ONE relation ------------------------------
    dropped = entries[-1]
    kept_entries = entries[:-1]
    kept_metas = [_shard_meta(e, shards[e["relation_id"]]) for e in kept_entries]
    assert len(kept_metas) == 37

    partial_cp = rc.build_checkpoint(identity=identity, completed_shards=kept_metas,
                                     expected_production_wired=True)
    with pytest.raises(rc.StaleGenerationError) as exc:
        rc.assert_checkpoint_complete(partial_cp, expected_relation_ids=all_ids)
    assert "INCOMPLETE_PANEL" in str(exc.value)
    assert dropped["relation_id"] in str(exc.value)

    partial_resume = rc.check_resume(partial_cp, identity=identity,
                                     completed_shards=kept_metas,
                                     expected_relation_ids=all_ids,
                                     expected_production_wired=True)
    assert partial_resume["state"] == rc.RESUMABLE
    assert partial_resume["remaining"] == [dropped["relation_id"]]
    assert partial_resume["n_completed"] == 37

    # --- a physically written 37-shard generation is NOT promoted ----------
    reduced_root = tmp_path / "reduced"
    written = rc.write_capture_generation(
        str(reduced_root),
        shards=[shards[e["relation_id"]] for e in kept_entries],
        identity=identity,
        config=rc.R40CaptureConfig(enabled=True, output_root=str(reduced_root),
                                   disk_budget_bytes=256 * 1024 * 1024),
        expected_production_wired=True,
    )
    assert written["n_shards"] == 37
    reduced_gen = reduced_root / written["generation_dir"]
    # the reduced generation is internally self-consistent ...
    report = rc.verify_generation(reduced_gen, expected_production_wired=True)
    assert report["ok"] is True and report["problems"] == []
    assert report["n_shards"] == 37
    # ... yet the PANEL completeness gate still refuses it: no promotion.
    with pytest.raises(rc.StaleGenerationError) as exc2:
        rc.assert_checkpoint_complete(partial_cp, expected_relation_ids=all_ids)
    assert "INCOMPLETE_PANEL" in str(exc2.value)
