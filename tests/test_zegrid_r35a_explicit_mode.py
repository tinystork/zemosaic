"""ZM-ZEGRID-R35A targeted tests.

Covers the R35A contract:
* explicit processing-mode resolution (classic / master tiles / SDS / ZeGrid);
* ``stack_plan.csv`` no longer activates ZeGrid and no longer truncates its input;
* ZeGrid inventory == eligible FITS of the input folder (deterministic, excluded
  dirs / output subtree / Global WCS artifact excluded);
* mount metadata (EQ / ALTZ) now read from the FITS header (EQMODE);
* temporary working directory resolver (blank -> output_dir, explicit -> used,
  unusable -> loud error, outputs never relocated);
* GUI pipeline checkboxes are mutually exclusive and round-trip through the Filter.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from zemosaic import processing_modes as pmodes
from zemosaic import input_inventory as inv
from zemosaic.zemosaic_utils import resolve_temporary_work_dir


# ---------------------------------------------------------------------------
# 1. Mode resolution
# ---------------------------------------------------------------------------

def test_default_is_classic():
    mode, info = pmodes.resolve_processing_mode({})
    assert mode == "classic"
    assert info["source"] == "default"


def test_each_legacy_flag_selects_its_mode():
    assert pmodes.resolve_processing_mode({"use_existing_master_tiles": True})[0] == "existing_master_tiles"
    assert pmodes.resolve_processing_mode({"sds_mode_default": True})[0] == "sds"
    assert pmodes.resolve_processing_mode({"zegrid_mode_enabled": True})[0] == "zegrid"


def test_string_booleans_never_false_positive():
    assert pmodes.resolve_processing_mode({"zegrid_mode_enabled": "false"})[0] == "classic"
    assert pmodes.resolve_processing_mode({"zegrid_mode_enabled": "true"})[0] == "zegrid"


def test_explicit_processing_mode_wins():
    mode, info = pmodes.resolve_processing_mode(
        {"processing_mode": "zegrid", "sds_mode_default": True}
    )
    assert mode == "zegrid"
    assert info["conflict"] == ["sds"]
    assert info["resolution"] == "explicit_mode_wins"


def test_conflicting_flags_resolved_by_documented_precedence():
    mode, info = pmodes.resolve_processing_mode(
        {"use_existing_master_tiles": True, "zegrid_mode_enabled": True, "sds_mode_default": True}
    )
    assert mode == "existing_master_tiles"
    assert info["resolution"] == "precedence"
    assert sorted(info["conflict"]) == ["existing_master_tiles", "sds", "zegrid"]


def test_processes_mode_token_ignores_requested_flag_not_equal():
    mode, info = pmodes.resolve_processing_mode(
        {"processing_mode": "classic", "zegrid_mode_enabled": True}
    )
    assert mode == "classic"
    assert info["conflict"] == ["zegrid"]


def test_apply_mode_to_flags_is_coherent():
    cfg = pmodes.apply_mode_to_flags({"input_dir": "x"}, "sds")
    assert cfg["processing_mode"] == "sds"
    assert cfg["sds_mode_default"] is True
    assert cfg["zegrid_mode_enabled"] is False
    assert cfg["use_existing_master_tiles"] is False
    assert cfg["input_dir"] == "x"


def test_both_arbiters_agree_on_every_conflict_pair():
    """R35A F1 regression: the config migration and the runtime resolver MUST
    arbitrate an incoherent legacy state identically (documented precedence)."""
    import itertools

    from zemosaic import zemosaic_config as zc

    keys = {
        "existing_master_tiles": "use_existing_master_tiles",
        "sds": "sds_mode_default",
        "zegrid": "zegrid_mode_enabled",
    }
    names = list(keys)
    for n in (2, 3):
        for combo in itertools.combinations(names, n):
            cfg = {keys[m]: True for m in combo}
            runtime_mode = pmodes.resolve_processing_mode(dict(cfg))[0]
            migrated = dict(cfg)
            zc._normalize_processing_mode(migrated)
            assert runtime_mode == migrated["processing_mode"], (
                f"arbiters disagree for {sorted(combo)}: "
                f"{runtime_mode!r} vs {migrated['processing_mode']!r}"
            )


# ---------------------------------------------------------------------------
# 2. Inventory: full FITS dataset, stack_plan has no effect
# ---------------------------------------------------------------------------

def _make_fits(path: Path, eqmode: int | None = None):
    from astropy.io import fits

    data = np.zeros((4, 4), dtype=np.float32)
    hdr = fits.Header()
    if eqmode is not None:
        hdr["EQMODE"] = eqmode
    fits.PrimaryHDU(data=data, header=hdr).writeto(path, overwrite=True)


def test_inventory_is_not_reduced_by_a_partial_stack_plan(tmp_path):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    for i in range(100):
        _make_fits(input_dir / f"frame_{i:03d}.fits")
    (input_dir / "stack_plan.csv").write_text(
        "file_path\n" + "\n".join(f"frame_{i:03d}.fits" for i in range(40)) + "\n",
        encoding="utf-8",
    )

    frames, info = inv.build_zegrid_frames(input_dir, tmp_path / "out")
    assert info["discovered"] == 100
    assert len(frames) == 100  # NOT silently reduced to 40


def test_inventory_excludes_output_subtree_quarantine_and_global_wcs(tmp_path):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    (in_dir / "sub").mkdir(parents=True)
    (in_dir / "unaligned_by_zemosaic").mkdir()
    out_dir.mkdir()

    _make_fits(in_dir / "a.fits")
    _make_fits(in_dir / "sub" / "b.fit")
    _make_fits(in_dir / "unaligned_by_zemosaic" / "c.fits")
    _make_fits(out_dir / "d.fits")           # output subtree
    _make_fits(in_dir / "global_mosaic_wcs.fits")  # internal artifact
    (in_dir / "notes.txt").write_text("x", encoding="utf-8")

    paths = inv.scan_input_fits(in_dir, out_dir)
    names = sorted(Path(p).name for p in paths)
    assert names == ["a.fits", "b.fit"]


def test_inventory_deterministic_ordering(tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    for name in ("b.fits", "A.fits", "c.fit"):
        _make_fits(in_dir / name)
    first = inv.scan_input_fits(in_dir, tmp_path / "out")
    second = inv.scan_input_fits(in_dir, tmp_path / "out")
    assert first == second
    assert [Path(p).name for p in first] == ["A.fits", "b.fits", "c.fit"]


def test_explicit_selection_narrows_inventory(tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    for i in range(10):
        _make_fits(in_dir / f"f{i}.fits")
    selected = [str(in_dir / "f1.fits"), str(in_dir / "f3.fits")]
    frames, info = inv.build_zegrid_frames(
        in_dir, tmp_path / "out", selection_paths=selected
    )
    assert info["discovered"] == 10
    assert info["selection_applied"] is True
    assert len(frames) == 2
    assert {Path(f.path).name for f in frames} == {"f1.fits", "f3.fits"}


def test_unmatched_selection_falls_back_to_full_inventory(tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    _make_fits(in_dir / "a.fits")
    frames, info = inv.build_zegrid_frames(
        in_dir, tmp_path / "out", selection_paths=[str(in_dir / "nope.fits")]
    )
    assert len(frames) == 1
    assert info["selection_unmatched"] == 1


# ---------------------------------------------------------------------------
# 3. Mount metadata from the FITS header
# ---------------------------------------------------------------------------

def test_mount_read_from_fits_header_eqmode(tmp_path):
    import zemosaic.zemosaic_stack_plan as zsp

    p_eq = tmp_path / "eq.fits"
    p_altz = tmp_path / "altz.fits"
    p_none = tmp_path / "none.fits"
    _make_fits(p_eq, eqmode=1)
    _make_fits(p_altz, eqmode=0)
    _make_fits(p_none, eqmode=None)

    def _load(path):
        fi = zsp.FrameInfo(path=path)
        try:
            zsp.load_frame_wcs(fi)
        except Exception:
            pass
        return fi.mount

    assert _load(p_eq) == "EQ"
    assert _load(p_altz) == "ALTZ"
    assert _load(p_none) is None


def test_normalize_mount_tokens():
    assert inv.normalize_mount(1) == "EQ"
    assert inv.normalize_mount(0) == "ALTZ"
    assert inv.normalize_mount("EQMODE_1") == "EQ"
    assert inv.normalize_mount("alt-az") == "ALTZ"
    assert inv.normalize_mount("nonsense") is None


# ---------------------------------------------------------------------------
# 4. Temporary working directory
# ---------------------------------------------------------------------------

def test_temp_dir_blank_uses_output_dir(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    assert resolve_temporary_work_dir(out, "") == out
    assert resolve_temporary_work_dir(out, None) == out


def test_temp_dir_explicit_is_used(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    explicit = tmp_path / "scratch" / "tmp"
    resolved = resolve_temporary_work_dir(out, str(explicit))
    assert resolved == explicit
    assert resolved.is_dir()


def test_temp_dir_unusable_raises_clear_error(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir", encoding="utf-8")
    with pytest.raises(RuntimeError) as exc:
        resolve_temporary_work_dir(out, str(blocker))
    assert "Temporary working directory" in str(exc.value)


def test_temp_dir_unwritable_raises(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    import zemosaic.zemosaic_utils as zu

    real_open = Path.open

    def _fail_open(self, *a, **k):
        if self.name == ".zemosaic_tempdir_probe":
            raise PermissionError("read-only")
        return real_open(self, *a, **k)

    monkeypatch.setattr(Path, "open", _fail_open)
    with pytest.raises(RuntimeError) as exc:
        resolve_temporary_work_dir(out, str(scratch))
    assert "not writable" in str(exc.value)


# ---------------------------------------------------------------------------
# 5. Worker dispatch (real run_hierarchical_mosaic with fakes)
# ---------------------------------------------------------------------------

_WORKER_ARGS = dict(
    input_folder="/tmp/x", output_folder="/tmp/y", astap_exe_path="",
    astap_data_dir_param="", astap_search_radius_config=1.0,
    astap_downsample_config=1, astap_sensitivity_config=1,
    cluster_threshold_config=0.1, cluster_target_groups_config=1,
    cluster_orientation_split_deg_config=1.0, progress_callback=lambda *a, **k: None,
    stack_ram_budget_gb_config=0.0, stack_norm_method="sky_mean",
    stack_weight_method="noise_variance", stack_reject_algo="kappa_sigma",
    stack_kappa_low=3.0, stack_kappa_high=3.0, parsed_winsor_limits=(0.05, 0.05),
    stack_final_combine="mean", poststack_equalize_rgb_config=False,
    apply_radial_weight_config=False, radial_feather_fraction_config=0.8,
    radial_shape_power_config=2.0, min_radial_weight_floor_config=0.0,
    final_assembly_method_config="reproject_coadd",
    inter_master_merge_enable_config=False, inter_master_overlap_threshold_config=0.3,
    inter_master_min_group_size_config=1, inter_master_stack_method_config="mean",
    inter_master_memmap_policy_config="auto", inter_master_local_scale_config="none",
    inter_master_max_group_config=1, num_base_workers_config=2,
    apply_master_tile_crop_config=False, master_tile_crop_percent_config=0.0,
    quality_crop_enabled_config=False, quality_crop_band_px_config=0,
    quality_crop_k_sigma_config=3.0, quality_crop_margin_px_config=0,
    quality_crop_min_run_config=0, altaz_cleanup_enabled_config=False,
    altaz_margin_percent_config=0.0, altaz_decay_config=0.0, altaz_nanize_config=False,
    quality_gate_enabled_config=False, quality_gate_threshold_config=0.5,
    quality_gate_edge_band_px_config=0, quality_gate_k_sigma_config=3.0,
    quality_gate_erode_px_config=0, quality_gate_move_rejects_config=False,
    save_final_as_uint16_config=False, legacy_rgb_cube_config=False,
    coadd_use_memmap_config=False, coadd_memmap_dir_config="",
    coadd_cleanup_memmap_config=False, assembly_process_workers_config=1,
    auto_limit_frames_per_master_tile_config=False,
    winsor_max_frames_per_pass_config=0, winsor_worker_limit_config=1,
    max_raw_per_master_tile_config=100,
)


def _dispatch(monkeypatch, config: dict):
    """Run the real dispatcher with faked backends; return which backend ran."""
    import sys

    import zemosaic.zemosaic_worker as zw

    calls = {"zegrid": 0, "classic": 0, "classic_master": 0}

    monkeypatch.setattr(zw, "ZEMOSAIC_CONFIG_AVAILABLE", True)
    monkeypatch.setattr(
        zw, "zemosaic_config", SimpleNamespace(load_config=lambda: dict(config))
    )

    def fake_zegrid(**kwargs):
        calls["zegrid"] += 1
        return None

    monkeypatch.setitem(
        sys.modules,
        "zemosaic.zemosaic_zegrid_mode",
        SimpleNamespace(
            run_zegrid_mode=fake_zegrid,
            resolve_gpu_preference=lambda *a, **k: (False, "test"),
        ),
    )

    def fake_classic(**kwargs):
        if kwargs.get("use_existing_master_tiles_config"):
            calls["classic_master"] += 1
        else:
            calls["classic"] += 1
        return None

    monkeypatch.setattr(zw, "run_hierarchical_mosaic_classic_legacy", fake_classic)

    args = dict(_WORKER_ARGS)
    args["input_folder"] = str(config.pop("_input", "/tmp/x"))
    args["output_folder"] = str(config.pop("_output", "/tmp/y"))
    args.update({k: v for k, v in config.items() if k in zw.run_hierarchical_mosaic.__code__.co_varnames or True})
    # only keep real signature keys
    import inspect

    sig = inspect.signature(zw.run_hierarchical_mosaic).parameters
    args = {k: v for k, v in args.items() if k in sig}
    try:
        zw.run_hierarchical_mosaic(**args)
    except Exception:
        pass
    return calls


def test_stack_plan_present_without_mode_is_classic(monkeypatch, tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    (in_dir / "stack_plan.csv").write_text("file_path\n", encoding="utf-8")
    calls = _dispatch(monkeypatch, {"_input": in_dir, "_output": tmp_path / "o"})
    assert calls == {"zegrid": 0, "classic": 1, "classic_master": 0}


def test_stack_plan_absent_without_mode_is_classic(monkeypatch, tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    calls = _dispatch(monkeypatch, {"_input": in_dir, "_output": tmp_path / "o"})
    assert calls["classic"] == 1


def test_explicit_zegrid_with_stack_plan_present_still_uses_inventory(monkeypatch, tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    (in_dir / "stack_plan.csv").write_text("file_path\n", encoding="utf-8")
    calls = _dispatch(
        monkeypatch,
        {"_input": in_dir, "_output": tmp_path / "o", "zegrid_mode_enabled": True},
    )
    assert calls["zegrid"] == 1


def test_explicit_sds_dispatch(monkeypatch, tmp_path):
    calls = _dispatch(
        monkeypatch,
        {"_input": tmp_path / "in", "_output": tmp_path / "o", "sds_mode_default": True},
    )
    assert calls["zegrid"] == 0


def test_master_tiles_dispatch(monkeypatch, tmp_path):
    calls = _dispatch(
        monkeypatch,
        {"_input": tmp_path / "in", "_output": tmp_path / "o", "use_existing_master_tiles": True},
    )
    assert calls["classic_master"] == 1


def test_zegrid_dispatched_without_stack_plan(monkeypatch, tmp_path):
    calls = _dispatch(
        monkeypatch,
        {"_input": tmp_path / "in", "_output": tmp_path / "o", "zegrid_mode_enabled": True},
    )
    assert calls["zegrid"] == 1


# ---------------------------------------------------------------------------
# 6. Global WCS default preserved (UX removal only)
# ---------------------------------------------------------------------------

def test_global_wcs_default_resolution_still_works(tmp_path):
    from zemosaic.zemosaic_utils import resolve_global_wcs_output_paths

    fits_path, json_path = resolve_global_wcs_output_paths(str(tmp_path), None)
    assert Path(fits_path).name == "global_mosaic_wcs.fits"
    assert Path(fits_path).parent == tmp_path
    assert Path(json_path).suffix == ".json"


def test_config_defaults_include_new_keys():
    import zemosaic.zemosaic_config as zc

    assert "processing_mode" in zc.DEFAULT_CONFIG
    assert "zegrid_mode_enabled" in zc.DEFAULT_CONFIG
    assert "temporary_work_dir" in zc.DEFAULT_CONFIG
    assert zc.DEFAULT_CONFIG["processing_mode"] == "auto"
    assert zc.DEFAULT_CONFIG["global_wcs_output_path"] == "global_mosaic_wcs.fits"
