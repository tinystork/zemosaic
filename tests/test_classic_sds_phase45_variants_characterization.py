"""Characterization witness: SCI-04 — Classic / SDS / Phase 4.5 variants inventory.

Pins the *current, deterministic* differences among the Classic legacy pipeline,
SDS mode, and the Phase 4.5 pre-stack path, plus low-N and chunking behavior —
**before any future consolidation**, without consolidating, unifying, refactoring,
or fixing anything.

Sections and evidence classes
-----------------------------
* **A. Route inventory** (STATIC via AST/import + dynamic existence proof): which
  module/function owns Classic vs SDS vs Grid vs Phase 4.5 stacking; the dispatcher
  ``run_hierarchical_mosaic`` and the legacy pipeline
  ``run_hierarchical_mosaic_classic_legacy`` are distinct module-level entry points.
  The SDS-mode flag resolution precedence (explicit filter override > plan override >
  config default > False) lives *inline* inside the two big pipeline functions, so it
  is pinned STATIC-only via a test-local reconstruction of the precedence chain using
  the *real* ``_coerce_bool_flag``; the real resolution code path is not hermetically
  reachable (it requires a full pipeline run) and is not overclaimed.
* **B. SDS pure-helper contracts** (DYNAMIC, deterministic, tiny arrays): the five
  module-level pure helpers ``_mask_sds_low_coverage_pixels``,
  ``_sanitize_sds_megatile_payload``, ``_sds_compute_tile_payload``,
  ``_sds_choose_reference_index``, ``_normalize_sds_megatiles_photometry`` are called
  directly on the real code and pinned for shape/dtype/value.
* **C. Phase 4.5 vs Classic stacking differences** (DYNAMIC entry existence +
  RECONSTRUCTION seam): the Phase 4.5 alpha-weighted branch (inline weighted-mean with
  ``nan_to_num`` + clip ``[0,1]``) vs the rejection-based branch (real
  ``zemosaic_align_stack`` wrappers) vs the Classic legacy wrapper route are
  distinguished. The active pre-stack normalization ``linear_fit`` (slope clipped
  ``0.25..4``, intercept from *unclipped* slope, skip when denominator <= 0 / non-finite,
  hard ``min_overlap_required >= 5000`` common pixels) vs ``sky_mean`` (percentile
  low/high clip then median delta, no pixel-count gate) is characterized via a faithful
  test-only reconstruction of the inline block — the production code is inline inside
  ``_run_phase4_5_inter_master_merge`` and not importable as a standalone callable, so
  this sub-item is labeled RECONSTRUCTION, not a direct production-call proof.
* **D. low-N route differences** (INVENTORY, cross-reference only): Classic kappa/winsor
  N<3 forcing CPU + valid stack ``rejected=0.0``; Grid CPU zero vs core NaN; classic
  N>=3 not covered — cross-referenced to ``test_stacking_low_n_all_invalid_witness.py``
  (TEST-04) and ``test_grid_mask_weight_characterization.py`` (SCI-03) rather than
  re-tested.
* **E. Chunking differences** (DYNAMIC for the cheap profile helper; STATIC inventory
  otherwise): ``_apply_safe_dynamic_chunk_profile`` for each mode and invalid fallback;
  Phase 4.5 ``max_group`` chunk loop, Phase 5 VRAM chunk-budget helper, and DBE chunked
  RBF evaluation are distinguished as three *different* chunking mechanisms.

Design notes
------------
* All corpora are tiny deterministic ``float32`` arrays built from explicit values; no
  random/unseeded data, no sleeps, no network, no profile writes, no GPU, no media.
* NaN/finite semantics are asserted with explicit masks (``np.isnan``/``np.isfinite``);
  ``np.testing.assert_equal``/``equal_nan`` are avoided so a real value is never
  confused with a NaN.
* Env is isolated via ``monkeypatch`` and never mutates real user config/profile.
* Route ownership is proven via ``ast`` call-site/``inspect`` presence, not source
  line numbers or source substrings.
* The Phase 4.5 inline normalization and inline alpha-weighted branch are exercised
  only through a clearly labeled test-only reconstruction; no formula is copied into
  production code.
"""

from __future__ import annotations

import ast
import inspect
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import zemosaic.zemosaic_worker as zw  # noqa: E402
from zemosaic import zemosaic_align_stack  # noqa: E402


# ---------------------------------------------------------------------------
# A. Route inventory helpers
# ---------------------------------------------------------------------------

_WORKER_SRC = REPO_ROOT / "src" / "zemosaic" / "zemosaic_worker.py"


def _function_def_names(source: str) -> set[str]:
    tree = ast.parse(source)
    return {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }


def _call_names(source: str) -> set[str]:
    """Return the set of bare/attr function names referenced by Call nodes."""
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            names.add(func.id)
        elif isinstance(func, ast.Attribute):
            names.add(func.attr)
    return names


# ---------------------------------------------------------------------------
# B. SDS pure-helper contracts
# ---------------------------------------------------------------------------


def _frame(*values: float) -> np.ndarray:
    """Build a 2x2x1 float32 HWC frame from four scalar pixel values (row-major)."""
    return np.array(
        [[[values[0]], [values[1]]], [[values[2]], [values[3]]]],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# C. Phase 4.5 test-only reconstruction seams
# ---------------------------------------------------------------------------


def _reconstruct_phase45_prestack_normalize(
    frames,
    *,
    norm_method: str,
    clip_sigma: float = 3.0,
    sky_low: float = 30.0,
    sky_high: float = 70.0,
) -> list[np.ndarray]:
    """Test-only reconstruction of the inline Phase 4.5 pre-stack normalization.

    Faithful mirror of the inline block inside ``_run_phase4_5_inter_master_merge``
    (``src/zemosaic/zemosaic_worker.py``). The production code is inline (not a
    standalone callable), so this is a reconstruction seam — not a direct call to
    production code. No formula is copied into production.
    """
    frames = [np.asarray(f, dtype=np.float32) for f in frames]
    if norm_method not in ("linear_fit", "sky_mean") or len(frames) < 2:
        return frames
    ref_arr = frames[0]
    if ref_arr.ndim == 2:
        ref_arr = ref_arr[..., np.newaxis]
    ref_channels = ref_arr.shape[-1]
    total_pixels = max(1, ref_arr.shape[0] * ref_arr.shape[1])
    min_overlap_required = max(5000, int(math.ceil(total_pixels * 0.01)))
    for frame_idx in range(1, len(frames)):
        src_arr = frames[frame_idx]
        if src_arr is None:
            continue
        src_arr = np.asarray(src_arr, dtype=np.float32)
        if src_arr.ndim == 2:
            src_arr = src_arr[..., np.newaxis]
        if src_arr.shape[-1] != ref_channels:
            continue
        if norm_method == "linear_fit":
            for ch in range(ref_channels):
                ref_chan = ref_arr[..., ch]
                src_chan = src_arr[..., ch]
                common_mask = np.isfinite(ref_chan) & np.isfinite(src_chan)
                common_pixels = int(common_mask.sum())
                if common_pixels < min_overlap_required:
                    continue
                x = src_chan[common_mask]
                y = ref_chan[common_mask]
                if clip_sigma > 0.0 and x.size and y.size:
                    clip_mask = np.ones(x.shape, dtype=bool)
                    if x.size > 4:
                        x_med = float(np.nanmedian(x))
                        x_std = float(np.nanstd(x))
                        if math.isfinite(x_std) and x_std > 0.0:
                            clip_mask &= np.abs(x - x_med) <= (clip_sigma * x_std)
                    if y.size > 4:
                        y_med = float(np.nanmedian(y))
                        y_std = float(np.nanstd(y))
                        if math.isfinite(y_std) and y_std > 0.0:
                            clip_mask &= np.abs(y - y_med) <= (clip_sigma * y_std)
                    if not np.any(clip_mask):
                        continue
                    x = x[clip_mask]
                    y = y[clip_mask]
                    if x.size < max(1000, min_overlap_required // 2):
                        continue
                if x.size < 2:
                    continue
                xm = float(np.mean(x))
                ym = float(np.mean(y))
                xv = x - xm
                yv = y - ym
                denom = float(np.dot(xv, xv))
                if denom <= 0.0 or not math.isfinite(denom):
                    continue
                slope = float(np.dot(xv, yv) / denom)
                intercept = float(ym - slope * xm)
                if not (math.isfinite(slope) and math.isfinite(intercept)):
                    continue
                slope = float(np.clip(slope, 0.25, 4.0))
                src_valid = np.isfinite(src_chan)
                if not np.any(src_valid):
                    continue
                src_values = src_chan[src_valid]
                src_values *= slope
                src_values += intercept
                src_chan[src_valid] = src_values
        else:  # sky_mean
            for ch in range(ref_channels):
                ref_chan = ref_arr[..., ch]
                src_chan = src_arr[..., ch]
                ref_mask = np.isfinite(ref_chan)
                src_mask = np.isfinite(src_chan)
                if not (np.any(ref_mask) and np.any(src_mask)):
                    continue
                ref_vals = ref_chan[ref_mask]
                src_vals = src_chan[src_mask]
                ref_range = np.nanpercentile(ref_vals, [sky_low, sky_high])
                src_range = np.nanpercentile(src_vals, [sky_low, sky_high])
                ref_sel = (ref_vals >= ref_range[0]) & (ref_vals <= ref_range[1])
                src_sel = (src_vals >= src_range[0]) & (src_vals <= src_range[1])
                ref_clip = ref_vals[ref_sel] if np.any(ref_sel) else ref_vals
                src_clip = src_vals[src_sel] if np.any(src_sel) else src_vals
                if ref_clip.size == 0 or src_clip.size == 0:
                    continue
                bg_ref = float(np.nanmedian(ref_clip))
                bg_src = float(np.nanmedian(src_clip))
                if not (math.isfinite(bg_ref) and math.isfinite(bg_src)):
                    continue
                delta = bg_ref - bg_src
                src_chan[src_mask] = src_chan[src_mask] + delta
    return frames


def _reconstruct_phase45_alpha_weighted_stack(
    frames, frame_weights
):
    """Test-only reconstruction of the inline Phase 4.5 alpha-weighted branch.

    Faithful mirror of the inline block inside ``_run_phase4_5_inter_master_merge``.
    Reconstruction seam, not a direct call to production code.
    """
    frames_np = np.stack(frames, axis=0).astype(np.float32, copy=False)
    reference_shape = frames_np.shape[1:3]
    weight_stack_list = []
    for wmap in frame_weights:
        if isinstance(wmap, np.ndarray) and wmap.shape == reference_shape:
            weight_stack_list.append(wmap.astype(np.float32, copy=False))
        else:
            weight_stack_list.append(np.ones(reference_shape, dtype=np.float32))
    weight_stack = np.stack(weight_stack_list, axis=0)
    weight_stack = np.clip(np.nan_to_num(weight_stack, nan=0.0), 0.0, 1.0)
    weight_expanded = weight_stack[..., None]
    num = np.nansum(frames_np * weight_expanded, axis=0)
    den = np.nansum(weight_expanded, axis=0)
    super_arr = np.where(den > 0, num / den, np.nan)
    alpha_out = (np.nanmax(weight_stack, axis=0) * 255.0).astype(np.uint8)
    return super_arr, alpha_out


# ---------------------------------------------------------------------------
# A. Route inventory
# ---------------------------------------------------------------------------


def test_route_dispatcher_and_legacy_are_distinct_entry_points():
    """The dispatcher and the legacy pipeline are distinct module-level functions."""
    assert callable(zw.run_hierarchical_mosaic)
    assert callable(zw.run_hierarchical_mosaic_classic_legacy)
    assert zw.run_hierarchical_mosaic is not zw.run_hierarchical_mosaic_classic_legacy
    assert zw.run_hierarchical_mosaic.__module__ == "zemosaic.zemosaic_worker"
    assert (
        zw.run_hierarchical_mosaic_classic_legacy.__module__
        == "zemosaic.zemosaic_worker"
    )


def test_route_dispatcher_calls_classic_and_zegrid_statically():
    """STATIC AST: the dispatcher references the legacy wrapper and ZeGrid runner."""
    dispatcher_src = inspect.getsource(zw.run_hierarchical_mosaic)
    names = _call_names(dispatcher_src)
    assert "run_hierarchical_mosaic_classic_legacy" in names
    assert "run_zegrid_mode" in names
    # The removed legacy Grid runner must NOT be referenced anymore.
    assert "run_grid_mode" not in names
    # The dispatcher itself must NOT be defined to *call itself* recursively as its
    # own name in a way that would collapse the two entry points.
    assert "run_hierarchical_mosaic_classic_legacy" in _function_def_names(
        _WORKER_SRC.read_text(encoding="utf-8")
    )


def test_route_module_ownership_static():
    """STATIC AST/import: each stacking route lives in its owning module/function."""
    align_src = (
        REPO_ROOT / "src" / "zemosaic" / "zemosaic_align_stack.py"
    ).read_text(encoding="utf-8")
    align_names = _function_def_names(align_src)
    for name in (
        "stack_winsorized_sigma_clip",
        "stack_kappa_sigma_clip",
        "stack_linear_fit_clip",
        "stack_aligned_images",
    ):
        assert name in align_names

    core_src = (
        REPO_ROOT / "src" / "zemosaic" / "zemosaic_align_stack.py"
    ).read_text(encoding="utf-8")
    assert "stack_aligned_images" in _function_def_names(core_src)

    worker_names = _function_def_names(_WORKER_SRC.read_text(encoding="utf-8"))
    for name in (
        "_mask_sds_low_coverage_pixels",
        "_sanitize_sds_megatile_payload",
        "_sds_compute_tile_payload",
        "_sds_choose_reference_index",
        "_normalize_sds_megatiles_photometry",
        "_finalize_sds_global_mosaic",
        "_apply_safe_dynamic_chunk_profile",
        "_run_phase4_5_inter_master_merge",
    ):
        assert name in worker_names


def test_route_sds_helpers_importable_and_callable():
    """DYNAMIC existence: the SDS pure helpers are importable module-level callables."""
    for name in (
        "_mask_sds_low_coverage_pixels",
        "_sanitize_sds_megatile_payload",
        "_sds_compute_tile_payload",
        "_sds_choose_reference_index",
        "_normalize_sds_megatiles_photometry",
        "_finalize_sds_global_mosaic",
    ):
        obj = getattr(zw, name)
        assert callable(obj), name


def test_route_phase45_rejection_entry_uses_real_wrappers_dynamically():
    """DYNAMIC: the Phase 4.5 rejection branch's named stack entries resolve.

    The inline Phase 4.5 branch first probes ``stack_kappa_sigma`` (which does NOT
    exist) and falls through to ``stack_kappa_sigma_clip``; winsorized goes straight
    to ``stack_winsorized_sigma_clip``. The Classic legacy route calls the same real
    ``zemosaic_align_stack`` wrappers plus ``stack_aligned_images``.
    """
    assert callable(zemosaic_align_stack.stack_winsorized_sigma_clip)
    assert callable(zemosaic_align_stack.stack_kappa_sigma_clip)
    assert callable(zemosaic_align_stack.stack_linear_fit_clip)
    assert callable(zemosaic_align_stack.stack_aligned_images)
    # Phase 4.5's first kappa candidate is absent -> the hasattr() gate falls through.
    assert not hasattr(zemosaic_align_stack, "stack_kappa_sigma")


def test_route_sds_flag_resolution_precedence_reconstructed():
    """RECONSTRUCTION of the inline SDS-mode flag resolution precedence chain.

    The real resolution is inline in ``run_hierarchical_mosaic`` and
    ``run_hierarchical_mosaic_classic_legacy`` (duplicated, ARCH-05) and is not
    hermetically reachable without a full pipeline run. This test reconstructs the
    exact precedence chain using the *real* ``_coerce_bool_flag`` helper, and labels
    the result STATIC/RECONSTRUCTION — not a direct production-call proof.
    """
    coerce = zw._coerce_bool_flag

    def resolve(filter_overrides, worker_config_cache):
        config_sds_default = coerce(worker_config_cache.get("sds_mode_default"))
        override_flag = None
        override_defined = False
        plan_override_flag = None
        plan_override_defined = False
        plan_override = None
        if isinstance(filter_overrides, dict):
            if "sds_mode" in filter_overrides:
                override_flag = coerce(filter_overrides.get("sds_mode"))
                override_defined = override_flag is not None
            plan_override = filter_overrides.get("global_wcs_plan_override")
        if isinstance(plan_override, dict) and "sds_mode" in plan_override:
            plan_override_flag = coerce(plan_override.get("sds_mode"))
            plan_override_defined = plan_override_flag is not None
        if override_defined:
            return override_flag
        if plan_override_defined:
            return plan_override_flag
        if config_sds_default is not None:
            return config_sds_default
        return False

    # explicit filter override wins over plan and config
    assert resolve({"sds_mode": True}, {"sds_mode_default": False}) is True
    assert resolve({"sds_mode": False}, {"sds_mode_default": True}) is False
    # plan override wins over config default
    assert (
        resolve(
            {"global_wcs_plan_override": {"sds_mode": True}},
            {"sds_mode_default": False},
        )
        is True
    )
    # config default
    assert resolve({}, {"sds_mode_default": True}) is True
    assert resolve({}, {"sds_mode_default": False}) is False
    # nothing -> False
    assert resolve({}, {}) is False
    # override_defined is only True when coercion yields a non-None bool
    assert resolve({"sds_mode": None}, {"sds_mode_default": True}) is True


# ---------------------------------------------------------------------------
# B. SDS pure-helper contracts
# ---------------------------------------------------------------------------


def test_mask_sds_low_coverage_pixels_normalizes_and_masks():
    cov = np.array([[1, 2], [3, 4]], dtype=np.float32)
    mos = _frame(10.0, 20.0, 30.0, 40.0)
    out_mos, out_cov, summary = zw._mask_sds_low_coverage_pixels(
        mos, cov, min_keep_fraction=0.5
    )
    # max_cov = 4; normalized [0.25,0.5,0.75,1.0]; only [0,0] (0.25) < 0.5 masked.
    assert summary == {"max_cov": 4.0, "masked_pixels": 1}
    assert out_cov.dtype == np.float32
    assert np.isnan(out_mos[0, 0, 0])
    assert not np.isnan(out_mos[1, 1, 0])
    assert out_cov[0, 0] == 0.0
    assert out_cov[1, 1] == 4.0
    # non-masked pixels retain original coverage values
    assert out_cov[0, 1] == 2.0
    assert out_cov[1, 0] == 3.0


def test_mask_sds_low_coverage_pixels_noop_when_none_empty_zero_or_frac_zero():
    mos = _frame(1.0, 2.0, 3.0, 4.0)
    # coverage None -> unchanged identity
    out_mos, out_cov, summary = zw._mask_sds_low_coverage_pixels(
        mos, None, min_keep_fraction=0.5
    )
    assert out_mos is mos
    assert out_cov is None
    assert summary == {"max_cov": 0.0, "masked_pixels": 0}
    # empty coverage -> max_cov 0 -> no masking
    out_mos, out_cov, summary = zw._mask_sds_low_coverage_pixels(
        mos, np.empty((0, 0), dtype=np.float32), min_keep_fraction=0.5
    )
    assert summary["masked_pixels"] == 0
    # max <= 0 -> no masking
    out_mos, out_cov, summary = zw._mask_sds_low_coverage_pixels(
        mos, np.zeros((2, 2), dtype=np.float32), min_keep_fraction=0.5
    )
    assert summary == {"max_cov": 0.0, "masked_pixels": 0}
    # frac <= 0 -> no masking even with positive coverage
    out_mos, out_cov, summary = zw._mask_sds_low_coverage_pixels(
        mos, np.ones((2, 2), dtype=np.float32), min_keep_fraction=0.0
    )
    assert summary["masked_pixels"] == 0


def test_mask_sds_low_coverage_pixels_target_hw_reshape():
    # 1D coverage of size 4 reshaped to (2,2) via target_hw.
    cov = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    mos = _frame(1.0, 2.0, 3.0, 4.0)
    out_mos, out_cov, summary = zw._mask_sds_low_coverage_pixels(
        mos, cov, min_keep_fraction=0.5, target_hw=(2, 2)
    )
    assert summary["masked_pixels"] == 1
    assert out_cov.shape == (2, 2)
    assert np.isnan(out_mos[0, 0, 0])


def test_sanitize_sds_megatile_payload_coverage_from_coverage():
    cov = np.array([[1, 2], [3, 4]], dtype=np.float32)
    mos = _frame(1.0, 2.0, 3.0, 4.0)
    out_mos, out_cov, out_alpha = zw._sanitize_sds_megatile_payload(mos, cov, None)
    assert out_cov.dtype == np.float32
    assert out_alpha.dtype == np.uint8
    # alpha derived from coverage/max*255, clipped to uint8
    assert out_alpha.tolist() == [[63, 127], [191, 255]]
    # all coverage > 1e-6 -> no NaN in mosaic
    assert not np.any(np.isnan(out_mos))


def test_sanitize_sds_megatile_payload_coverage_from_alpha_normalized():
    alpha = np.array([[255, 128], [64, 0]], dtype=np.float32)
    mos = _frame(1.0, 2.0, 3.0, 4.0)
    out_mos, out_cov, out_alpha = zw._sanitize_sds_megatile_payload(mos, None, alpha)
    # coverage = alpha / max_alpha (255)
    assert out_cov[0, 0] == 1.0
    assert np.isclose(out_cov[0, 1], 128.0 / 255.0)
    assert np.isclose(out_cov[1, 0], 64.0 / 255.0)
    assert out_cov[1, 1] == 0.0
    # zero-alpha pixel (<=1e-6) is masked -> NaN in mosaic, alpha 0
    assert np.isnan(out_mos[1, 1, 0])
    assert out_alpha[1, 1] == 0
    assert out_alpha.dtype == np.uint8


def test_sanitize_sds_megatile_payload_valid_mask_and_fallback_ones():
    # coverage with a zero pixel -> valid_mask excludes it
    cov = np.array([[1, 2], [0, 4]], dtype=np.float32)
    mos = _frame(1.0, 2.0, 3.0, 4.0)
    out_mos, out_cov, out_alpha = zw._sanitize_sds_megatile_payload(mos, cov, None)
    assert np.isnan(out_mos[1, 0, 0])
    assert out_cov[1, 0] == 0.0
    # both coverage and alpha None -> coverage fallback ones, no masking
    out_mos, out_cov, out_alpha = zw._sanitize_sds_megatile_payload(mos, None, None)
    assert np.array_equal(out_cov, np.ones((2, 2), dtype=np.float32))
    assert not np.any(np.isnan(out_mos))
    # mosaic None -> (None, coverage, None)
    out_mos, out_cov, out_alpha = zw._sanitize_sds_megatile_payload(
        None, np.ones((2, 2), dtype=np.float32), None
    )
    assert out_mos is None
    assert out_alpha is None


def test_sanitize_sds_megatile_payload_1d_drop_and_shape_coercion():
    mos = _frame(1.0, 2.0, 3.0, 4.0)
    # 1D coverage cannot be reshaped meaningfully -> dropped -> fallback ones
    cov_1d = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    out_mos, out_cov, out_alpha = zw._sanitize_sds_megatile_payload(mos, cov_1d, None)
    assert out_cov.shape == (2, 2)
    assert not np.any(np.isnan(out_mos))
    # mismatched 2D shape -> cropped/padded to target (2,2)
    cov_3x3 = np.ones((3, 3), dtype=np.float32)
    out_mos, out_cov, out_alpha = zw._sanitize_sds_megatile_payload(mos, cov_3x3, None)
    assert out_cov.shape == (2, 2)


def test_sds_compute_tile_payload_median_and_stats():
    tile = np.array([[1, 2], [3, 4]], dtype=np.float32)
    cov = np.ones((2, 2), dtype=np.float32)
    arr, median, stats = zw._sds_compute_tile_payload(tile, cov)
    assert median == 2.5
    assert arr.dtype == np.float32
    assert stats == {
        "coverage_weight": 4.0,
        "coverage_pixels": 4,
        "coverage_max": 1.0,
    }


def test_sds_compute_tile_payload_coverage_positive_mask():
    # F6 (D3): plain finite + explicit-support mask (no 1% derived-constant threshold).
    tile = np.array([[100, 2], [3, 4]], dtype=np.float32)
    cov = np.array([[0.005, 1.0], [1.0, 1.0]], dtype=np.float32)
    _, median, stats = zw._sds_compute_tile_payload(tile, cov)
    # mask = cov > 0.0 includes the low-coverage pixel [0,0]; median of [100,2,3,4] = 3.5.
    assert median == 3.5
    assert stats["coverage_max"] == 1.0
    assert stats["coverage_weight"] == pytest.approx(3.005)


def test_sds_compute_tile_payload_all_invalid_and_positive_abs():
    cov = np.ones((2, 2), dtype=np.float32)
    # all-NaN tile -> fallback median 1.0
    _, median, _ = zw._sds_compute_tile_payload(
        np.full((2, 2), np.nan, dtype=np.float32), cov
    )
    assert median == 1.0
    # negative median -> abs/positive normalization
    _, median, _ = zw._sds_compute_tile_payload(
        np.array([[-5, -2], [-1, -4]], dtype=np.float32), None
    )
    assert median == 3.0
    # no coverage -> coverage stats remain zeroed
    _, _, stats = zw._sds_compute_tile_payload(
        np.array([[1, 2], [3, 4]], dtype=np.float32), None
    )
    assert stats == {"coverage_weight": 0.0, "coverage_pixels": 0, "coverage_max": 0.0}


def test_sds_choose_reference_index_contracts():
    # F6 (D2): canonical N1 — max VALID SUPPORT COUNT (coverage_pixels), not coverage_weight/central.
    def payload(pixels):
        return (np.zeros((1, 1), dtype=np.float32), 1.0, {"coverage_pixels": pixels})

    pl = [payload(1), payload(9), payload(5)]
    # requested valid index wins
    assert zw._sds_choose_reference_index(pl, 2) == 2
    # max coverage_pixels selection
    assert zw._sds_choose_reference_index(pl, None) == 1
    assert zw._sds_choose_reference_index(pl, 99) == 1
    # no valid support -> first index (0)
    assert zw._sds_choose_reference_index([payload(0), payload(0)], None) == 0
    # empty list -> 0
    assert zw._sds_choose_reference_index([], None) == 0


def test_normalize_sds_megatiles_photometry_contracts():
    t_ref = np.full((2, 2, 1), 2.0, dtype=np.float32)
    t_bright = np.full((2, 2, 1), 4.0, dtype=np.float32)
    out = zw._normalize_sds_megatiles_photometry(
        [t_ref, t_bright], [None, None], ref_index=0
    )
    # reference untouched; bright tile scaled by ref_median/tile_median = 2/4 = 0.5
    assert out[0][0, 0, 0] == 2.0
    assert out[1][0, 0, 0] == pytest.approx(2.0)
    assert out[1].dtype == np.float32
    # empty -> []
    assert zw._normalize_sds_megatiles_photometry([], None) == []
    # zero-median tile -> gain non-finite/<=0 -> 1.0 (untouched)
    t_zero = np.zeros((2, 2, 1), dtype=np.float32)
    out2 = zw._normalize_sds_megatiles_photometry(
        [t_ref, t_zero], [None, None], ref_index=0
    )
    assert out2[1][0, 0, 0] == 0.0


# ---------------------------------------------------------------------------
# C. Phase 4.5 vs Classic stacking differences
# ---------------------------------------------------------------------------


def test_phase45_alpha_weighted_branch_is_distinct_from_rejection():
    """RECONSTRUCTION: alpha-weighted weighted-mean differs from the rejection path.

    The alpha-weighted branch (when per-frame weight maps are present) is an inline
    weighted-mean with ``nan_to_num`` + clip ``[0,1]``; it is NOT one of the
    ``stack_winsorized_sigma_clip``/``stack_kappa_sigma_clip`` rejection wrappers.
    """
    f1 = _frame(1.0, 2.0, 3.0, 4.0)
    f2 = _frame(3.0, 6.0, 1.0, 8.0)
    w = np.ones((2, 2), dtype=np.float32)
    super_arr, alpha = _reconstruct_phase45_alpha_weighted_stack(
        [f1, f2], [w, w]
    )
    # equal weights -> plain mean per pixel
    assert super_arr[0, 0, 0] == pytest.approx(2.0)
    assert super_arr[1, 1, 0] == pytest.approx(6.0)
    # alpha = max weight across frames * 255 (unit weights -> 255)
    assert alpha.dtype == np.uint8
    assert alpha[0, 0] == 255


def test_phase45_alpha_weighted_nan_to_num_and_clip():
    """RECONSTRUCTION: nan->0 and clip [0,1] on the weight stack."""
    f1 = _frame(10.0, 10.0, 10.0, 10.0)
    f2 = _frame(20.0, 20.0, 20.0, 20.0)
    # weights: frame1 has NaN and 5.0 (>1 -> clipped), frame2 zero
    w1 = np.array([[np.nan, 5.0], [1.0, 1.0]], dtype=np.float32)
    w2 = np.zeros((2, 2), dtype=np.float32)
    super_arr, alpha = _reconstruct_phase45_alpha_weighted_stack(
        [f1, f2], [w1, w2]
    )
    # [0,0]: w1 nan->0, w2 0 -> den 0 -> NaN
    assert np.isnan(super_arr[0, 0, 0])
    # [0,1]: w1 5.0 -> clipped to 1.0 -> mean = 10 (only frame1 contributes)
    assert super_arr[0, 1, 0] == pytest.approx(10.0)
    # alpha: max weight clipped -> [0,1] -> 255 at [0,1]
    assert alpha[0, 1] == 255


def test_phase45_prestack_linear_fit_affine_normalizes():
    """RECONSTRUCTION: linear_fit maps a known affine source back to the reference.

    Uses a mono 80x80 corpus so common_pixels (6400) >= min_overlap_required (5000).
    slope = cov(src,ref)/var(src) = 1/a for src = a*ref + b; intercept from unclipped
    slope; slope clipped to [0.25, 4.0]. For a=2 the slope 0.5 is within range, so the
    result maps src back to ref.
    """
    ref = np.arange(6400, dtype=np.float32).reshape(80, 80)
    src = (2.0 * ref + 10.0).astype(np.float32)
    out = _reconstruct_phase45_prestack_normalize(
        [ref, src], norm_method="linear_fit"
    )
    result = out[1]
    assert result.shape == (80, 80)
    np.testing.assert_allclose(result, ref, rtol=1e-3, atol=1.0)


def test_phase45_prestack_linear_fit_slope_clipped_0_25_to_4():
    """RECONSTRUCTION: slope outside [0.25,4] is clipped.

    For src = 0.1*ref, raw slope = 1/0.1 = 10 -> clipped to 4.0. The intercept is
    computed from the *unclipped* slope (10), so the result is NOT a clean map back to
    ref (it becomes 0.4*ref), proving the clip changes the transform.
    """
    ref = np.arange(6400, dtype=np.float32).reshape(80, 80)
    src = (0.1 * ref).astype(np.float32)
    out = _reconstruct_phase45_prestack_normalize(
        [ref, src], norm_method="linear_fit"
    )
    result = out[1]
    # 0.1*ref * 4.0 + 0.0 == 0.4*ref
    np.testing.assert_allclose(result, 0.4 * ref, rtol=1e-3, atol=1.0)
    # not equal to ref (the clip prevented full normalization)
    assert not np.allclose(result, ref, rtol=1e-2, atol=1.0)


def test_phase45_prestack_linear_fit_skips_on_tiny_or_constant():
    """RECONSTRUCTION: linear_fit is a no-op below min_overlap_required or for constant.

    A tiny 4x4 corpus (16 common pixels < 5000) is skipped entirely; a constant source
    yields denom <= 0 and is skipped.
    """
    ref = np.arange(16, dtype=np.float32).reshape(4, 4)
    src = (2.0 * ref).astype(np.float32)
    out = _reconstruct_phase45_prestack_normalize(
        [ref, src], norm_method="linear_fit"
    )
    # unchanged (no normalization applied)
    assert np.array_equal(out[1], src)
    # constant source -> denom <= 0 -> skip
    big_ref = np.arange(6400, dtype=np.float32).reshape(80, 80)
    const_src = np.full((80, 80), 7.0, dtype=np.float32)
    out2 = _reconstruct_phase45_prestack_normalize(
        [big_ref, const_src], norm_method="linear_fit"
    )
    assert np.array_equal(out2[1], const_src)


def test_phase45_prestack_sky_mean_percentile_delta():
    """RECONSTRUCTION: sky_mean removes a constant offset via percentile median delta.

    sky_mean has no pixel-count gate (unlike linear_fit), so even a tiny 4x4 corpus is
    normalized. Percentile [30,70] clip -> median, then delta = bg_ref - bg_src added.
    """
    ref = np.arange(16, dtype=np.float32).reshape(4, 4)
    src = (ref + 50.0).astype(np.float32)
    out = _reconstruct_phase45_prestack_normalize(
        [ref, src], norm_method="sky_mean", sky_low=30.0, sky_high=70.0
    )
    # offset removed: src -> ref
    np.testing.assert_allclose(out[1], ref, rtol=1e-3, atol=1.0)


def test_phase45_linear_fit_vs_sky_mean_distinct_behavior():
    """RECONSTRUCTION: linear_fit (affine, >=5000 px gate) vs sky_mean (offset, no gate).

    On a tiny corpus linear_fit is a no-op while sky_mean removes the offset — a
    concrete, discriminating behavioral difference between the two active pre-stack
    normalization methods. The reconstruction mutates the source frame in place
    (faithful to the production inline block), so the original offset value is
    captured before calling sky_mean.
    """
    ref = np.arange(16, dtype=np.float32).reshape(4, 4)
    # linear_fit: below the 5000-common-pixel gate -> no-op, source unchanged
    src_lf = (ref + 50.0).astype(np.float32)
    lf = _reconstruct_phase45_prestack_normalize(
        [ref, src_lf], norm_method="linear_fit"
    )
    assert np.array_equal(lf[1], src_lf)
    # sky_mean: no pixel-count gate -> removes the offset (mutates in place)
    src_sm = (ref + 50.0).astype(np.float32)
    original_offset = src_sm.copy()
    sm = _reconstruct_phase45_prestack_normalize(
        [ref, src_sm], norm_method="sky_mean"
    )
    assert not np.array_equal(sm[1], original_offset)  # sky_mean changed it
    np.testing.assert_allclose(sm[1], ref, rtol=1e-3, atol=1.0)


# ---------------------------------------------------------------------------
# E. Chunking differences
# ---------------------------------------------------------------------------


_SAFE_DYNAMIC_KEYS = {
    "parallel_autotune_enabled",
    "parallel_target_cpu_load",
    "parallel_target_ram_fraction",
    "parallel_gpu_vram_fraction",
    "phase5_chunk_auto",
    "phase3_ram_high_pct",
    "phase3_ram_critical_pct",
    "phase3_chunk_scale_high",
    "phase3_chunk_scale_critical",
}

_SAFE_DYNAMIC_VALUES = {
    "parallel_autotune_enabled": True,
    "parallel_target_cpu_load": 0.90,
    "parallel_target_ram_fraction": 0.82,
    "parallel_gpu_vram_fraction": 0.72,
    "phase5_chunk_auto": True,
    "phase3_ram_high_pct": 80.0,
    "phase3_ram_critical_pct": 86.0,
    "phase3_chunk_scale_high": 0.70,
    "phase3_chunk_scale_critical": 0.50,
}

_SAFE_DYNAMIC_PLUS_VALUES = {
    "parallel_autotune_enabled": True,
    "parallel_target_cpu_load": 0.94,
    "parallel_target_ram_fraction": 0.85,
    "parallel_gpu_vram_fraction": 0.75,
    "phase5_chunk_auto": True,
    "phase3_ram_high_pct": 82.0,
    "phase3_ram_critical_pct": 88.0,
    "phase3_chunk_scale_high": 0.72,
    "phase3_chunk_scale_critical": 0.55,
}

_AGGRESSIVE_VALUES = {
    "parallel_autotune_enabled": True,
    "parallel_target_cpu_load": 0.95,
    "parallel_target_ram_fraction": 0.90,
    "parallel_gpu_vram_fraction": 0.80,
    "phase5_chunk_auto": True,
    "phase3_ram_high_pct": 84.0,
    "phase3_ram_critical_pct": 90.0,
    "phase3_chunk_scale_high": 0.75,
    "phase3_chunk_scale_critical": 0.60,
}


@pytest.mark.parametrize(
    "mode,expected_values",
    [
        ("safe_dynamic", _SAFE_DYNAMIC_VALUES),
        ("safe_dynamic_plus", _SAFE_DYNAMIC_PLUS_VALUES),
        ("aggressive", _AGGRESSIVE_VALUES),
    ],
)
def test_chunk_profile_applies_keys_per_mode(mode, expected_values):
    """DYNAMIC: each named profile sets the full key set and correct values."""
    cfg = {"chunk_profile_mode": mode}
    zconfig = SimpleNamespace()
    ret = zw._apply_safe_dynamic_chunk_profile(cfg, zconfig, pcb=None)
    assert ret == mode
    set_keys = set(cfg.keys()) - {"chunk_profile_mode"}
    assert set_keys == _SAFE_DYNAMIC_KEYS
    for key, value in expected_values.items():
        assert cfg[key] == value
        assert getattr(zconfig, key) == value


def test_chunk_profile_baseline_keeps_raw_config():
    """DYNAMIC: baseline returns the raw mode string without setting profile keys."""
    cfg = {"chunk_profile_mode": "baseline"}
    zconfig = SimpleNamespace()
    ret = zw._apply_safe_dynamic_chunk_profile(cfg, zconfig, pcb=None)
    assert ret == "baseline"
    assert set(cfg.keys()) == {"chunk_profile_mode"}


def test_chunk_profile_invalid_falls_back_to_safe_dynamic():
    """DYNAMIC: invalid/None/empty mode falls back to safe_dynamic."""
    for mode in ("BOGUS", None, ""):
        cfg = {"chunk_profile_mode": mode}
        zconfig = SimpleNamespace()
        ret = zw._apply_safe_dynamic_chunk_profile(cfg, zconfig, pcb=None)
        assert ret == "safe_dynamic"
        assert cfg["parallel_target_cpu_load"] == 0.90
        assert cfg["phase5_chunk_auto"] is True


def test_chunk_profile_safe_when_cfg_missing_or_zconfig_none():
    """DYNAMIC: safe when cfg is None/non-dict or zconfig is None."""
    assert zw._apply_safe_dynamic_chunk_profile(None, None, pcb=None) == "safe_dynamic"
    assert (
        zw._apply_safe_dynamic_chunk_profile("not-a-dict", SimpleNamespace(), pcb=None)
        == "safe_dynamic"
    )
    # zconfig with chunk_profile_mode set but cfg empty -> reads from zconfig
    zconfig = SimpleNamespace(chunk_profile_mode="aggressive")
    cfg = {}
    ret = zw._apply_safe_dynamic_chunk_profile(cfg, zconfig, pcb=None)
    assert ret == "aggressive"
    assert cfg["parallel_target_cpu_load"] == 0.95


def test_phase5_chunk_budget_is_distinct_mechanism():
    """DYNAMIC: Phase 5 VRAM chunk-budget helper is a separate chunking mechanism.

    It computes a *byte budget* (not a config profile), keyed on power state /
    VRAM probe. This distinguishes it from ``_apply_safe_dynamic_chunk_profile`` and
    from the Phase 4.5 ``max_group`` chunk loop.
    """
    ac = SimpleNamespace(
        power_plugged=True,
        on_battery=False,
        is_hybrid_graphics=False,
        vram_free_bytes=6 * 1024**3,
    )
    budget, meta = zw._compute_phase5_vram_budget_bytes(ac, {}, False)
    assert meta["fraction"] == 0.80
    assert budget > 0
    assert isinstance(budget, int)
    battery = SimpleNamespace(
        power_plugged=False,
        on_battery=True,
        is_hybrid_graphics=True,
        vram_free_bytes=6 * 1024**3,
    )
    _, meta_b = zw._compute_phase5_vram_budget_bytes(battery, {}, False)
    assert meta_b["fraction"] == 0.25
    # unknown VRAM probe -> fallback budget path
    _, meta_unknown = zw._compute_phase5_vram_budget_bytes(None, {}, True)
    assert "probe_unknown" in meta_unknown["reasons"]
    assert meta_unknown["fallback_mb"] == 128


def test_phase45_chunk_loop_static_inventory():
    """STATIC: Phase 4.5 chunk loop is bounded by max_group and inline (not a helper).

    The chunking inside ``_run_phase4_5_inter_master_merge`` iterates
    ``range(0, len(members), max_group)`` and computes
    ``group_chunks = max(1, ceil(len(members)/max_group))``. It is a *loop* over tile
    members (bounded by ``max_group``), a distinct mechanism from the config profile
    helper and the Phase 5 byte budget. Heavy execution is NOT_RUN.
    """
    src = _WORKER_SRC.read_text(encoding="utf-8")
    merge_src = inspect.getsource(zw._run_phase4_5_inter_master_merge)
    names = _call_names(merge_src)
    assert "ceil" in names
    # the chunk loop slices members by max_group
    assert "range(0, len(members), max_group)" in merge_src


def test_dbe_chunked_rbf_static_inventory():
    """STATIC: DBE chunked RBF evaluation is a third, distinct chunking mechanism.

    ``_apply_final_mosaic_dbe_per_channel`` chunks the *evaluation points* of a scipy
    RBF model (``chunk_points = max(1024, max_eval_pairs_chunk // n_samples)``) to
    bound per-chunk memory; it does not touch the stacking chunk profile or the Phase 5
    byte budget. Heavy compute is NOT_RUN.
    """
    assert callable(zw._apply_final_mosaic_dbe_per_channel)
    dbe_src = inspect.getsource(zw._apply_final_mosaic_dbe_per_channel)
    assert "rbf_eval_chunked" in dbe_src or "chunk_points" in dbe_src
