"""Characterization witness: Classic cache/resume/checkpoint semantics (pre-R2).

Pins the *current* deterministic behavior of the legacy classic pipeline's
cache/resume/checkpoint layers before any R2 extraction:

* top-level ``zemosaic.zemosaic_worker._safe_load_cache`` (WinError-1455
  memmap fallback),
* the nested phase5 checkpoint writer/loader inside
  ``run_hierarchical_mosaic_classic_legacy``,
* the nested phase1 resume cache writer/loader inside the same entrypoint.

This is a **characterization**, not an endorsement. Where current behavior is
permissive (signature mismatch in "force" mode, optional coverage/alpha arrays,
singleton trailing-channel normalization) it is pinned exactly as observed and
*not* "improved".

Design / seam
-------------
``_safe_load_cache`` is a module-level function and is exercised directly.

The phase5 checkpoint and phase1 resume helpers are *nested* closures and are
not importable. They are exercised by reconstructing the original nested
``code`` objects from ``run_hierarchical_mosaic_classic_legacy.__wrapped__.__code__``
(the decorator ``@_close_zesolver_on_run_exit`` uses ``functools.wraps``, so
``__wrapped__`` is the undecorated function). ``types.FunctionType`` is used
with ``types.CellType`` closures to instantiate the *original* code objects
against the module's real ``__dict__`` (so ``np``/``Path``/``fits``/``WCS``/
``_path_exists``/etc. resolve to the real module globals). No formula or body is
copied or reimplemented; the extraction is identity-proven below
(``fn.__code__ is <original code object>``) and the helpers are called
behaviorally with tiny arrays/files under ``tmp_path``.

Brittleness: this seam depends on the *names* of the nested helpers and on them
remaining nested closures of ``run_hierarchical_mosaic_classic_legacy``. Any
future R2 extraction that lifts these helpers to module scope will require
updating this test to import them directly (the behavioral assertions remain
valid).
"""

from __future__ import annotations

import json
import os
import types

import numpy as np
import pytest

from astropy.io import fits
from astropy.wcs import WCS

from zemosaic import zemosaic_worker as zw


# ---------------------------------------------------------------------------
# Code-object extraction seam (identity-proven)
# ---------------------------------------------------------------------------

_OUTER_CODE = zw.run_hierarchical_mosaic_classic_legacy.__wrapped__.__code__


def _collect_nested_codes(code: types.CodeType) -> dict[str, types.CodeType]:
    """Recursively collect nested ``CodeType`` objects by ``co_name``."""
    out: dict[str, types.CodeType] = {}

    def walk(c: types.CodeType) -> None:
        for const in c.co_consts:
            if isinstance(const, types.CodeType):
                out[const.co_name] = const
                walk(const)

    walk(code)
    return out


_NESTED_CODES = _collect_nested_codes(_OUTER_CODE)


def _build_nested(name: str, overrides: dict | None = None, kwdefaults: dict | None = None) -> object:
    """Instantiate the original nested ``code`` object named ``name``.

    ``overrides`` supplies values for freevars that are *not* other nested
    functions (captured entrypoint locals/params); nested-function freevars are
    resolved recursively. ``kwdefaults`` restores keyword-only argument
    defaults that ``types.FunctionType`` does not carry over.
    """
    overrides = overrides or {}
    cache: dict[str, object] = {}

    def build(nm: str) -> object:
        if nm in cache:
            return cache[nm]
        code = _NESTED_CODES[nm]
        cells: list[object] = []
        for fv in code.co_freevars:
            if fv in overrides:
                cells.append(types.CellType(overrides[fv]))
            elif fv in _NESTED_CODES:
                cells.append(types.CellType(build(fv)))
            else:
                raise KeyError(f"unresolved freevar {fv!r} for nested function {nm!r}")
        fn = types.FunctionType(code, zw.__dict__, nm, None, tuple(cells))
        if kwdefaults and nm in kwdefaults:
            fn.__kwdefaults__ = kwdefaults[nm]
        cache[nm] = fn
        return fn

    return build(name)


# Phase5 checkpoint helpers (all freevars are other nested helpers).
_PHASE5_KWDEFAULTS = {
    "_try_load_phase5_checkpoint": {
        "expected_final_assembly_method": None,
        "expected_master_tiles_count": None,
        "expected_raw_count": None,
        "expected_output_shape_hw": None,
    },
}


def _write_phase5_checkpoint(output_root, run_signature, **kw):
    fn = _build_nested("_write_phase5_checkpoint")
    return fn(output_root, run_signature, **kw)


def _try_load_phase5_checkpoint(output_root, run_signature, **kw):
    fn = _build_nested("_try_load_phase5_checkpoint", kwdefaults=_PHASE5_KWDEFAULTS)
    return fn(output_root, run_signature, **kw)


def _state_paths(output_root):
    fn = _build_nested("_state_paths")
    return fn(output_root)


# Phase1 resume helpers.
def _write_phase1_resume_cache(cache_dir, entries, signature_payload, signature_current, *, input_folder, output_folder):
    overrides = {
        "input_folder": input_folder,
        "output_folder": output_folder,
        "worker_config_cache": {},
        "solver_settings": {},
        "astap_search_radius_config": 180.0,
        "astap_downsample_config": 2,
        "astap_sensitivity_config": 2.0,
    }
    fn = _build_nested("_write_phase1_resume_cache", overrides=overrides)
    return fn(cache_dir, entries, signature_payload, signature_current)


def _try_resume_phase1(cache_dir, resume_mode, signature_current, progress_callback=None):
    calls: list[tuple] = []

    def _recorder(*args, **kwargs):
        calls.append((args, kwargs))

    cb = _recorder if progress_callback is None else progress_callback
    overrides = {"progress_callback": cb}
    fn = _build_nested("_try_resume_phase1", overrides=overrides)
    result = fn(cache_dir, resume_mode, signature_current)
    return result, calls


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _wcs_header() -> fits.Header:
    hdr = fits.Header()
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = 16
    hdr["NAXIS2"] = 16
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CRPIX1"] = 8.5
    hdr["CRPIX2"] = 8.5
    hdr["CRVAL1"] = 84.0
    hdr["CRVAL2"] = -44.0
    hdr["CDELT1"] = -0.000694
    hdr["CDELT2"] = 0.000694
    hdr["EQUINOX"] = 2000.0
    return hdr


def _write_raw_fits(path: str) -> None:
    fits.writeto(path, np.zeros((16, 16), dtype=np.float32), header=_wcs_header(), overwrite=True)


def _write_cache_npy(path: str) -> None:
    np.save(path, np.zeros((16, 16), dtype=np.float32))


# ---------------------------------------------------------------------------
# A. top-level ``_safe_load_cache``
# ---------------------------------------------------------------------------


def test_safe_load_cache_normal_memmap(tmp_path):
    path = tmp_path / "cached.npy"
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.save(path, data, allow_pickle=False)

    loaded = zw._safe_load_cache(str(path))

    assert isinstance(loaded, np.memmap)
    assert loaded.mode == "r"
    np.testing.assert_array_equal(loaded, data)


def test_safe_load_cache_winerror1455_fallback(tmp_path, monkeypatch):
    path = tmp_path / "cached.npy"
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    np.save(path, data, allow_pickle=False)

    real_load = np.load
    state = {"calls": 0}
    pcb_calls: list[tuple] = []

    def fake_load(p, **kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            err = OSError("paging file is too small for this operation")
            err.winerror = 1455
            raise err
        return real_load(p, **kwargs)

    monkeypatch.setattr(zw.np, "load", fake_load)

    loaded = zw._safe_load_cache(
        str(path),
        pcb=lambda *a, **kw: pcb_calls.append((a, kw)),
        tile_id=7,
    )

    assert state["calls"] == 2
    np.testing.assert_array_equal(loaded, data)
    # fallback callback emitted once with the pinned message key
    assert len(pcb_calls) == 1
    args, kw = pcb_calls[0]
    assert args[0] == "stack_mem_fallback_memmap_to_ram"
    assert kw.get("lvl") == "WARN"
    assert kw.get("tile_id") == 7


def test_safe_load_cache_non_1455_oserror_reraised(tmp_path, monkeypatch):
    path = tmp_path / "cached.npy"
    np.save(path, np.arange(4, dtype=np.float32), allow_pickle=False)

    def fake_load(p, **kwargs):
        raise OSError("disk I/O error")

    monkeypatch.setattr(zw.np, "load", fake_load)

    with pytest.raises(OSError, match="disk I/O error"):
        zw._safe_load_cache(str(path))


def test_safe_load_cache_fallback_failure_reraised(tmp_path, monkeypatch):
    path = tmp_path / "cached.npy"
    np.save(path, np.arange(4, dtype=np.float32), allow_pickle=False)

    real_load = np.load
    state = {"calls": 0}

    def fake_load(p, **kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            err = OSError("paging file too small")
            err.winerror = 1455
            raise err
        # second attempt (no-memmap) also fails -> must propagate
        raise OSError("fallback read failed")

    monkeypatch.setattr(zw.np, "load", fake_load)

    with pytest.raises(OSError, match="fallback read failed"):
        zw._safe_load_cache(str(path), pcb=lambda *a, **kw: None)


# ---------------------------------------------------------------------------
# Extractor identity proof (required when using the code-object seam)
# ---------------------------------------------------------------------------


def test_nested_extractor_identity_and_freevars():
    assert _OUTER_CODE.co_name == "run_hierarchical_mosaic_classic_legacy"

    for name, expected_freevars in {
        "_write_phase5_checkpoint": ("_atomic_save_npy", "_atomic_write_json", "_state_paths"),
        "_try_load_phase5_checkpoint": ("_read_json_file", "_state_paths"),
        "_try_resume_phase1": ("pcb",),
        "_write_phase1_resume_cache": (
            "_compute_current_signature",
            "_normalize_manifest_path",
            "_serialize_phase1_entry",
            "input_folder",
            "output_folder",
        ),
    }.items():
        assert name in _NESTED_CODES, name
        code = _NESTED_CODES[name]
        assert code.co_name == name
        assert tuple(code.co_freevars) == expected_freevars

    # Built function must *be* the original code object (no copy).
    built = _build_nested("_write_phase5_checkpoint")
    assert built.__code__ is _NESTED_CODES["_write_phase5_checkpoint"]

    built_load = _build_nested("_try_load_phase5_checkpoint", kwdefaults=_PHASE5_KWDEFAULTS)
    assert built_load.__code__ is _NESTED_CODES["_try_load_phase5_checkpoint"]
    assert built_load.__kwdefaults__ == _PHASE5_KWDEFAULTS["_try_load_phase5_checkpoint"]


# ---------------------------------------------------------------------------
# B. phase5 checkpoint write/load (nested original implementation)
# ---------------------------------------------------------------------------


def test_phase5_checkpoint_roundtrip_contract(tmp_path):
    out = str(tmp_path)
    sig = "sig-phase5-001"
    mosaic = np.zeros((2, 3, 2), dtype=np.float32)
    mosaic[0, 0, 0] = 1.5
    mosaic[1, 2, 1] = -0.25
    coverage = np.full((2, 3), 0.75, dtype=np.float32)
    alpha = np.full((2, 3), 0.9, dtype=np.float32)

    _write_phase5_checkpoint(
        out, sig,
        final_assembly_method="mean",
        mosaic_hwc=mosaic,
        coverage_hw=coverage,
        alpha_hw=alpha,
        final_output_shape_hw=(2, 3),
        master_tiles_count=2,
        raw_count=5,
    )

    paths = _state_paths(out)
    # manifest contract
    manifest = json.loads((paths["phase5"] and open(paths["phase5"], "r", encoding="utf-8").read()))
    assert manifest["schema_version"] == 1
    assert manifest["pipeline"] == "classic_legacy"
    assert manifest["run_signature"] == sig
    assert manifest["final_assembly_method"] == "mean"
    assert manifest["master_tiles_count"] == 2
    assert manifest["raw_count"] == 5
    assert manifest["output_shape_hw"] == [2, 3]
    assert manifest["mosaic"]["dtype"] == "float32"

    result = _try_load_phase5_checkpoint(out, sig)
    assert result is not None
    np.testing.assert_array_equal(result["mosaic"], mosaic)
    assert result["mosaic"].dtype == np.float32
    assert result["mosaic"].shape == (2, 3, 2)
    np.testing.assert_array_equal(result["coverage"], coverage)
    assert result["coverage"].shape == (2, 3)
    np.testing.assert_array_equal(result["alpha"], alpha)
    assert result["alpha"].shape == (2, 3)
    assert result["output_shape_hw"] == [2, 3]
    assert result["final_assembly_method"] == "mean"


def test_phase5_checkpoint_signature_schema_pipeline_rejection(tmp_path):
    out = str(tmp_path)
    sig = "sig-phase5-002"
    mosaic = np.zeros((2, 2, 1), dtype=np.float32)
    _write_phase5_checkpoint(
        out, sig, final_assembly_method="mean", mosaic_hwc=mosaic,
        coverage_hw=None, alpha_hw=None, final_output_shape_hw=(2, 2),
        master_tiles_count=1, raw_count=1,
    )
    paths = _state_paths(out)

    # signature mismatch
    assert _try_load_phase5_checkpoint(out, "other-sig") is None

    # schema mismatch
    manifest = json.loads(open(paths["phase5"], "r", encoding="utf-8").read())
    manifest["schema_version"] = 2
    with open(paths["phase5"], "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)
    assert _try_load_phase5_checkpoint(out, sig) is None

    # pipeline mismatch
    manifest["schema_version"] = 1
    manifest["pipeline"] = "grid"
    with open(paths["phase5"], "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)
    assert _try_load_phase5_checkpoint(out, sig) is None


def test_phase5_checkpoint_expected_mismatches_rejected(tmp_path):
    out = str(tmp_path)
    sig = "sig-phase5-003"
    mosaic = np.zeros((2, 2, 1), dtype=np.float32)
    _write_phase5_checkpoint(
        out, sig, final_assembly_method="mean", mosaic_hwc=mosaic,
        coverage_hw=None, alpha_hw=None, final_output_shape_hw=(2, 2),
        master_tiles_count=3, raw_count=10,
    )

    assert _try_load_phase5_checkpoint(out, sig, expected_final_assembly_method="median") is None
    assert _try_load_phase5_checkpoint(out, sig, expected_master_tiles_count=4) is None
    assert _try_load_phase5_checkpoint(out, sig, expected_raw_count=11) is None
    assert _try_load_phase5_checkpoint(out, sig, expected_output_shape_hw=(3, 3)) is None
    # matching expectations still load
    assert _try_load_phase5_checkpoint(
        out, sig,
        expected_final_assembly_method="mean",
        expected_master_tiles_count=3,
        expected_raw_count=10,
        expected_output_shape_hw=(2, 2),
    ) is not None


def test_phase5_checkpoint_missing_corrupt_invalid_mosaic_rejected(tmp_path):
    out = str(tmp_path)
    sig = "sig-phase5-004"
    mosaic = np.zeros((2, 2, 1), dtype=np.float32)
    _write_phase5_checkpoint(
        out, sig, final_assembly_method="mean", mosaic_hwc=mosaic,
        coverage_hw=None, alpha_hw=None, final_output_shape_hw=(2, 2),
        master_tiles_count=1, raw_count=1,
    )
    paths = _state_paths(out)
    mosaic_path = paths["phase5_mosaic"]

    # missing mosaic
    os.remove(mosaic_path)
    assert _try_load_phase5_checkpoint(out, sig) is None

    # corrupt (not a valid .npy)
    with open(mosaic_path, "wb") as fh:
        fh.write(b"not a numpy file")
    assert _try_load_phase5_checkpoint(out, sig) is None

    # invalid dimension (2D instead of HWC 3D)
    np.save(mosaic_path, np.zeros((2, 2), dtype=np.float32), allow_pickle=False)
    assert _try_load_phase5_checkpoint(out, sig) is None


def test_phase5_checkpoint_optional_arrays_degrade_to_none(tmp_path):
    out = str(tmp_path)
    sig = "sig-phase5-005"
    mosaic = np.zeros((2, 3, 1), dtype=np.float32)
    coverage = np.full((2, 3), 0.5, dtype=np.float32)
    alpha = np.full((2, 3), 0.8, dtype=np.float32)
    _write_phase5_checkpoint(
        out, sig, final_assembly_method="mean", mosaic_hwc=mosaic,
        coverage_hw=coverage, alpha_hw=alpha, final_output_shape_hw=(2, 3),
        master_tiles_count=1, raw_count=1,
    )
    paths = _state_paths(out)

    # missing coverage -> coverage None, mosaic still loads
    os.remove(paths["phase5_coverage"])
    r = _try_load_phase5_checkpoint(out, sig)
    assert r is not None
    assert r["coverage"] is None
    np.testing.assert_array_equal(r["mosaic"], mosaic)
    np.testing.assert_array_equal(r["alpha"], alpha)

    # wrong-shape coverage -> coverage None
    _write_phase5_checkpoint(
        out, sig, final_assembly_method="mean", mosaic_hwc=mosaic,
        coverage_hw=coverage, alpha_hw=alpha, final_output_shape_hw=(2, 3),
        master_tiles_count=1, raw_count=1,
    )
    np.save(paths["phase5_coverage"], np.zeros((9, 9), dtype=np.float32), allow_pickle=False)
    r = _try_load_phase5_checkpoint(out, sig)
    assert r is not None
    assert r["coverage"] is None

    # missing alpha -> alpha None
    _write_phase5_checkpoint(
        out, sig, final_assembly_method="mean", mosaic_hwc=mosaic,
        coverage_hw=coverage, alpha_hw=alpha, final_output_shape_hw=(2, 3),
        master_tiles_count=1, raw_count=1,
    )
    os.remove(paths["phase5_alpha"])
    r = _try_load_phase5_checkpoint(out, sig)
    assert r is not None
    assert r["alpha"] is None
    np.testing.assert_array_equal(r["mosaic"], mosaic)
    np.testing.assert_array_equal(r["coverage"], coverage)


def test_phase5_checkpoint_writer_shape_validation(tmp_path):
    out = str(tmp_path)
    sig = "sig-phase5-006"

    # non-HWC mosaic rejected
    with pytest.raises(ValueError, match="mosaic must be HWC"):
        _write_phase5_checkpoint(
            out, sig, final_assembly_method="mean",
            mosaic_hwc=np.zeros((2, 2), dtype=np.float32),
            coverage_hw=None, alpha_hw=None, final_output_shape_hw=(2, 2),
            master_tiles_count=1, raw_count=1,
        )

    # non-HW coverage rejected (trailing channel != 1)
    with pytest.raises(ValueError, match="coverage must be HW"):
        _write_phase5_checkpoint(
            out, sig, final_assembly_method="mean",
            mosaic_hwc=np.zeros((2, 2, 1), dtype=np.float32),
            coverage_hw=np.zeros((2, 2, 2), dtype=np.float32),
            alpha_hw=None, final_output_shape_hw=(2, 2),
            master_tiles_count=1, raw_count=1,
        )

    # non-HW alpha rejected
    with pytest.raises(ValueError, match="alpha must be HW"):
        _write_phase5_checkpoint(
            out, sig, final_assembly_method="mean",
            mosaic_hwc=np.zeros((2, 2, 1), dtype=np.float32),
            coverage_hw=None,
            alpha_hw=np.zeros((2, 2, 2), dtype=np.float32),
            final_output_shape_hw=(2, 2),
            master_tiles_count=1, raw_count=1,
        )


def test_phase5_checkpoint_singleton_trailing_channel_normalized(tmp_path):
    out = str(tmp_path)
    sig = "sig-phase5-007"
    mosaic = np.zeros((2, 2, 1), dtype=np.float32)
    # (H, W, 1) coverage/alpha are normalized to (H, W)
    _write_phase5_checkpoint(
        out, sig, final_assembly_method="mean", mosaic_hwc=mosaic,
        coverage_hw=np.full((2, 2, 1), 0.4, dtype=np.float32),
        alpha_hw=np.full((2, 2, 1), 0.6, dtype=np.float32),
        final_output_shape_hw=(2, 2),
        master_tiles_count=1, raw_count=1,
    )
    r = _try_load_phase5_checkpoint(out, sig)
    assert r is not None
    assert r["coverage"].shape == (2, 2)
    assert r["alpha"].shape == (2, 2)
    np.testing.assert_array_equal(r["coverage"], np.full((2, 2), 0.4, dtype=np.float32))


def test_phase5_checkpoint_no_tmp_files_after_success(tmp_path):
    out = str(tmp_path)
    sig = "sig-phase5-008"
    _write_phase5_checkpoint(
        out, sig, final_assembly_method="mean",
        mosaic_hwc=np.zeros((2, 2, 1), dtype=np.float32),
        coverage_hw=np.zeros((2, 2), dtype=np.float32),
        alpha_hw=np.zeros((2, 2), dtype=np.float32),
        final_output_shape_hw=(2, 2),
        master_tiles_count=1, raw_count=1,
    )
    state_dir = _state_paths(out)["dir"]
    leftovers = [f for f in os.listdir(state_dir) if f.endswith(".tmp")]
    assert leftovers == []


# ---------------------------------------------------------------------------
# C. phase1 resume/write (nested original implementation)
# ---------------------------------------------------------------------------


def _write_full_phase1_cache(tmp_path, *, n_entries=1, sig="sig-phase1-001"):
    cache_dir = str(tmp_path / "cache")
    os.makedirs(cache_dir, exist_ok=True)
    input_folder = str(tmp_path / "input")
    output_folder = str(tmp_path / "output")
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    entries = []
    for i in range(n_entries):
        raw_path = os.path.join(input_folder, f"raw_{i}.fits")
        cache_path = os.path.join(cache_dir, f"preproc_{i}.npy")
        _write_raw_fits(raw_path)
        _write_cache_npy(cache_path)
        entries.append(
            {
                "path_raw": raw_path,
                "path_preprocessed_cache": cache_path,
                "header": _wcs_header(),
                "preprocessed_shape": (16, 16),
            }
        )

    _write_phase1_resume_cache(
        cache_dir, entries,
        {"pipeline": "classic_legacy"},
        sig,
        input_folder=input_folder,
        output_folder=output_folder,
    )
    return cache_dir, input_folder, entries


def test_phase1_write_cache_manifest_and_marker(tmp_path):
    sig = "sig-phase1-001"
    cache_dir, input_folder, entries = _write_full_phase1_cache(tmp_path, sig=sig)

    manifest_path = os.path.join(cache_dir, "cache_manifest.json")
    processed_path = os.path.join(cache_dir, "phase1_processed_info.json")
    done_path = os.path.join(cache_dir, "phase1.done")

    assert os.path.exists(manifest_path)
    assert os.path.exists(processed_path)
    assert os.path.exists(done_path)

    manifest = json.loads(open(manifest_path, "r", encoding="utf-8").read())
    assert manifest["schema_version"] == 1
    assert manifest["pipeline"] == "classic_legacy"
    assert manifest["run_signature"] == sig
    assert manifest["phase1"]["done"] is True
    assert manifest["phase1"]["done_marker"] == "phase1.done"
    assert manifest["phase1"]["processed_info_file"] == "phase1_processed_info.json"
    assert manifest["phase1"]["num_entries"] == 1
    # normalized absolute paths recorded
    assert manifest["input_folder_norm"]
    assert manifest["output_folder_norm"]

    processed = json.loads(open(processed_path, "r", encoding="utf-8").read())
    assert isinstance(processed, list) and len(processed) == 1
    rec = processed[0]
    assert rec["path_raw"] == entries[0]["path_raw"]
    assert rec["path_preprocessed_cache"] == entries[0]["path_preprocessed_cache"]
    assert rec["preprocessed_shape"] == [16, 16]
    assert rec["header_str"]  # serialized FITS header present

    assert open(done_path, "r", encoding="utf-8").read() == "done\n"


def test_phase1_resume_auto_exact_signature_ok(tmp_path):
    sig = "sig-phase1-002"
    cache_dir, _, _ = _write_full_phase1_cache(tmp_path, sig=sig)

    (ok, entries, reason), _ = _try_resume_phase1(cache_dir, "auto", sig)

    assert ok is True
    assert reason == "ok"
    assert isinstance(entries, list) and len(entries) == 1
    entry = entries[0]
    assert isinstance(entry["header"], fits.Header)
    assert isinstance(entry["wcs"], WCS)
    assert entry["preprocessed_shape"] == (16, 16)
    assert "header_str" not in entry  # header_str is popped after reconstruction


def test_phase1_resume_auto_mismatch_rejects_force_warns(tmp_path):
    sig = "sig-phase1-003"
    cache_dir, _, _ = _write_full_phase1_cache(tmp_path, sig=sig)

    # auto + signature mismatch -> rejected
    (ok, entries, reason), _ = _try_resume_phase1(cache_dir, "auto", "wrong-sig")
    assert ok is False
    assert entries is None
    assert reason == "signature mismatch"

    # force + signature mismatch -> proceeds and warns
    (ok, entries, reason), calls = _try_resume_phase1(cache_dir, "force", "wrong-sig")
    assert ok is True
    assert reason == "ok"
    assert len(entries) == 1
    # warning surfaced through the real ``pcb`` (message passes via _log_and_callback)
    warned = any(
        (isinstance(args[0], str) and "signature mismatch ignored" in args[0])
        for args, _ in calls
    )
    assert warned


def test_phase1_resume_partial_and_no_valid(tmp_path):
    sig = "sig-phase1-004"
    cache_dir, input_folder, entries = _write_full_phase1_cache(tmp_path, n_entries=2, sig=sig)

    # partial: delete one raw file -> one entry missing_raw
    os.remove(entries[1]["path_raw"])
    (ok, entries_out, reason), _ = _try_resume_phase1(cache_dir, "auto", sig)
    assert ok is False
    assert entries_out is not None and len(entries_out) == 1
    assert reason.startswith("phase1 partial cache")
    assert "valid=1/2" in reason

    # no valid entries -> False/None
    os.remove(entries[0]["path_raw"])
    (ok, entries_out, reason), _ = _try_resume_phase1(cache_dir, "auto", sig)
    assert ok is False
    assert entries_out is None
    assert reason.startswith("phase1 cache unusable")
    assert "valid=0/2" in reason


def test_phase1_resume_reason_pins(tmp_path):
    sig = "sig-phase1-005"
    cache_dir, _, _ = _write_full_phase1_cache(tmp_path, sig=sig)

    # manifest missing
    (ok, entries, reason), _ = _try_resume_phase1(str(tmp_path / "empty"), "auto", sig)
    assert (ok, entries, reason) == (False, None, "manifest missing")

    # manifest invalid (garbage JSON)
    bad_dir = str(tmp_path / "badjson")
    os.makedirs(bad_dir, exist_ok=True)
    with open(os.path.join(bad_dir, "cache_manifest.json"), "w", encoding="utf-8") as fh:
        fh.write("{ not valid json")
    (ok, entries, reason), _ = _try_resume_phase1(bad_dir, "auto", sig)
    assert ok is False and entries is None
    assert reason.startswith("manifest invalid")

    # schema_version mismatch
    manifest_path = os.path.join(cache_dir, "cache_manifest.json")
    manifest = json.loads(open(manifest_path, "r", encoding="utf-8").read())
    manifest["schema_version"] = 2
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)
    (ok, entries, reason), _ = _try_resume_phase1(cache_dir, "auto", sig)
    assert (ok, entries, reason) == (False, None, "schema_version mismatch")

    # pipeline mismatch
    manifest["schema_version"] = 1
    manifest["pipeline"] = "grid"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)
    (ok, entries, reason), _ = _try_resume_phase1(cache_dir, "auto", sig)
    assert (ok, entries, reason) == (False, None, "pipeline mismatch")

    # processed-info missing
    manifest["pipeline"] = "classic_legacy"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)
    os.remove(os.path.join(cache_dir, "phase1_processed_info.json"))
    (ok, entries, reason), _ = _try_resume_phase1(cache_dir, "auto", sig)
    assert (ok, entries, reason) == (False, None, "phase1_processed_info.json missing")
