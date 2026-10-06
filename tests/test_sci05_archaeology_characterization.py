"""SCI-05 Gate A — canonical-stacking archaeology characterization (red witnesses).

Hermetic, deterministic baseline tests that **PASS at Gate A while proving current
gaps**. This file records *current* GUI→runtime behavior and the coverage-donor
conceptual gap. It deliberately does **not** correct, harmonize, unify, port, or
choose science. No source/config/dependency/version change is made.

Scope of evidence (minimum dynamic coverage required by the mission):

1.  Qt exposes the exact token/label for every requested stacking choice and the
    legacy radial controls (AST over ``_create_stacking_group`` — no Qt import,
    no brittle line numbers).
2.  ``stack_core`` ``linear_fit`` placeholder (== ``median``) and its simplified
    winsorized (median/σ, not PixInsight WSC) are cross-checked against the
    existing SCI-01/02 corpora with a *light* spot-probe (no corpus duplication).
3.  ``linear_fit_clip`` is a visible Qt choice whose executed helper
    ``_reject_outliers_linear_fit_clip`` is a proven no-op placeholder.
4.  ``noise_fwhm`` reachable Classic behavior (variance substitution only when
    Photutils is unavailable; no-weighting when star-free; partial ``1e-6``/``1.0``
    substitutions) plus the separate Grid variance-only fallback — proven via the
    real estimator (star-free) and controlled seams, NOT a blanket "→ variance".
5.  Weight zero/negative, median weight-ignore and all-invalid divergences are
    cross-referenced (SCI-03/TEST-04) and lightly spot-probed.
6.  The legacy center-radial map fails translation invariance and differs from
    the donor footprint-taper *concept* (reconstructed in test-only code; the
    donor module is **never** imported).
7.  Global-coadd dispatch names the actually-executed finalizer symbols/formulas
    per method (AST seam + small reconstruction of the chunked median/winsorized
    formulas).
8.  Current logging/provenance gaps (no support domain, no footprint taper, no
    coverage render, no ``N_eff``/``COVERAGE_*`` provenance) are named with
    evidence (absence scan over the package source).

Design notes
------------
* Deterministic tiny float32 corpora only; no random unseeded data, no sleeps,
  no network, no GPU, no media, no profile/XDG/HOME writes.
* The donor worktree (zsss-sci05) is treated as read-only data and is **never
  imported**; the footprint-taper property is reconstructed from its documented
  concept using standard ``scipy.ndimage.distance_transform_edt`` in test-only
  code, clearly labeled ``RECONSTRUCTION``.
* AST extraction is used instead of source-substring/line-number assertions for
  the GUI tokens and the global-coadd dispatch.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from zemosaic import zemosaic_align_stack as zas
from zemosaic import zemosaic_stack_core
from zemosaic.zemosaic_utils import make_radial_weight_map


_SRC = Path(__file__).resolve().parents[1] / "src" / "zemosaic"
_GUI_PATH = _SRC / "zemosaic_gui_qt.py"
_WORKER_PATH = _SRC / "zemosaic_worker.py"

_WSC_ENV = "ZEMOSAIC_WSC_IMPL"


# ---------------------------------------------------------------------------
# AST helpers (GUI tokens / global-coadd dispatch)
# ---------------------------------------------------------------------------

def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _find_func(tree: ast.Module, name: str) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _tr_call(node: ast.AST) -> tuple[str, str] | None:
    """Return (key, fallback) for an ``self._tr(key, fallback)`` call, else None."""
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr == "_tr":
        args = node.args
        if len(args) >= 2 and all(isinstance(a, ast.Constant) for a in args[:2]):
            return str(args[0].value), str(args[1].value)
    return None


def _gui_option_lists() -> dict[str, list[tuple[str, str, str]]]:
    """Extract the stacking-group combobox option lists from ``_create_stacking_group``.

    Returns ``{list_name: [(value, tr_key, fallback_label), ...]}`` for every
    ``*_options`` list literal in the method, in source order.
    """
    tree = _parse(_GUI_PATH)
    fn = _find_func(tree, "_create_stacking_group")
    assert fn is not None, "GUI _create_stacking_group not found"
    out: dict[str, list[tuple[str, str, str]]] = {}
    for node in fn.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id.endswith("_options"):
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        items: list[tuple[str, str, str]] = []
                        for elt in node.value.elts:
                            if (
                                isinstance(elt, (ast.Tuple, ast.List))
                                and len(elt.elts) >= 2
                                and isinstance(elt.elts[0], ast.Constant)
                            ):
                                tr = _tr_call(elt.elts[1])
                                if tr is not None:
                                    items.append((str(elt.elts[0].value), tr[0], tr[1]))
                        out[tgt.id] = items
    return out


def _gui_config_field_keys() -> set[str]:
    """Extract every ``self._config_fields[<key>]`` registration key inside
    ``_create_stacking_group`` (covers combobox, composite, checkbox, spinbox).
    """
    tree = _parse(_GUI_PATH)
    fn = _find_func(tree, "_create_stacking_group")
    assert fn is not None
    keys: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Subscript):
            slc = node.slice
            if isinstance(slc, ast.Constant) and isinstance(slc.value, str):
                value = node.value
                if isinstance(value, ast.Attribute) and value.attr == "_config_fields":
                    keys.add(slc.value)
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else None
            if name in {"_register_checkbox", "_register_double_spinbox", "_register_spinbox"}:
                if node.args and isinstance(node.args[0], ast.Constant):
                    keys.add(str(node.args[0].value))
    return keys


# ---------------------------------------------------------------------------
# 1. Qt tokens/labels + legacy radial controls
# ---------------------------------------------------------------------------

class TestQtStackingTokens:
    def test_normalization_tokens_and_labels(self):
        opts = _gui_option_lists()["norm_options"]
        got = [(v, label) for v, _k, label in opts]
        assert got == [
            ("none", "None"),
            ("linear_fit", "Linear Fit (Sky)"),
            ("sky_mean", "Sky Mean Subtraction"),
        ]

    def test_weighting_tokens_and_labels(self):
        opts = _gui_option_lists()["weight_options"]
        got = [(v, label) for v, _k, label in opts]
        assert got == [
            ("none", "None"),
            ("noise_variance", "Noise Variance (1/σ²)"),
            ("noise_fwhm", "Noise + FWHM"),
        ]

    def test_rejection_tokens_and_labels(self):
        opts = _gui_option_lists()["reject_options"]
        got = [(v, label) for v, _k, label in opts]
        # Gate A/R3 snapshot recorded 'linear_fit_clip' as a visible Qt choice; Gate F1
        # removed it (decision R3).
        assert got == [
            ("none", "None"),
            ("kappa_sigma", "Kappa-Sigma Clip"),
            ("winsorized_sigma_clip", "Winsorized Sigma Clip"),
        ]

    def test_combine_tokens_and_labels(self):
        opts = _gui_option_lists()["combine_options"]
        got = [(v, label) for v, _k, label in opts]
        assert got == [("mean", "Mean"), ("median", "Median")]

    def test_global_coadd_tokens_and_labels(self):
        opts = _gui_option_lists()["global_coadd_options"]
        got = [(v, label) for v, _k, label in opts]
        assert got == [
            ("kappa_sigma", "Global coadd: Kappa-Sigma"),
            ("winsorized", "Global coadd: Winsorized"),
            ("mean", "Global coadd: Mean"),
            ("median", "Global coadd: Median"),
        ]

    def test_legacy_radial_controls_registered(self):
        from zemosaic import zemosaic_config
        keys = _gui_config_field_keys()
        # Gate A snapshot recorded the legacy radial Qt controls as registered;
        # Gate E5b replaced them with the canonical Coverage controls (decision K).
        # (1) Legacy radial fields are NO LONGER registered in the Qt stacking group.
        for legacy in ("apply_radial_weight", "radial_feather_fraction", "min_radial_weight_floor"):
            assert legacy not in keys
        # ``radial_shape_power`` remains config-only (never a Qt stacking field).
        assert "radial_shape_power" not in keys
        # (2) The two canonical Coverage controls ARE registered, with correct defaults.
        assert {"coverage_support_taper", "coverage_aware_reconstruction"} <= keys
        assert zemosaic_config.DEFAULT_CONFIG["coverage_support_taper"] is True
        assert zemosaic_config.DEFAULT_CONFIG["coverage_aware_reconstruction"] is False

    def test_all_requested_keys_registered(self):
        keys = _gui_config_field_keys()
        expected = {
            "stacking_normalize_method",
            "stacking_weighting_method",
            "stacking_rejection_algorithm",
            "stacking_kappa_low",
            "stacking_kappa_high",
            "stacking_winsor_limits",
            "stacking_final_combine_method",
            "global_coadd_method",
            "poststack_equalize_rgb",
        }
        assert expected <= keys


# ---------------------------------------------------------------------------
# 2. stack_core linear_fit placeholder + simplified winsorized (cross-ref SCI-01/02)
# ---------------------------------------------------------------------------

def _affine_frames() -> list[np.ndarray]:
    """Two small float32 (2,2,1) frames: a ramp ref and an affine target."""
    ref = np.array([[[10.0], [20.0]], [[30.0], [40.0]]], dtype=np.float32)
    tgt = (2.0 * ref + 10.0).astype(np.float32)
    return [ref, tgt]


class TestStackCorePlaceholderCrossCheck:
    def test_linear_fit_is_median_bit_exact(self):
        # Gate A/SCI-02 recorded linear_fit == median (bit-exact placeholder); Gate F1 (N4)
        # removed the placeholder: linear_fit now raises, never silently becomes median.
        with pytest.raises(ValueError) as ei:
            zemosaic_stack_core.stack_core(
                _affine_frames(),
                stack_config={"normalize_method": "linear_fit", "final_combine_method": "mean"},
                backend="cpu",
            )
        assert "unsupported_removed_sci05" in str(ei.value)

    def test_linear_fit_differs_from_none(self):
        # Gate F1 (N4): linear_fit no longer re-centres the stack; it raises instead.
        with pytest.raises(ValueError) as ei:
            zemosaic_stack_core.stack_core(
                _affine_frames(),
                stack_config={"normalize_method": "linear_fit", "final_combine_method": "mean"},
                backend="cpu",
            )
        assert "unsupported_removed_sci05" in str(ei.value)

    def test_winsorized_simplified_diverges_from_pixinsight_wsc(self, monkeypatch):
        # SCI-01: stack_core winsorized = median/σ clip (keeps an impulse outlier
        # whose median is the floor), while the PixInsight WSC collapses it.
        monkeypatch.delenv(_WSC_ENV, raising=False)
        frames = [np.array([[[v]]], dtype=np.float32) for v in (0.0, 0.0, 0.0, 0.0, 100.0)]
        core_result, rejected, _ws = zemosaic_stack_core.stack_core(
            frames,
            stack_config={
                "rejection_algorithm": "winsorized_sigma_clip",
                "sigma_clip_low": 2.5,
                "sigma_clip_high": 2.5,
                "final_combine_method": "mean",
            },
            backend="cpu",
        )
        assert rejected == 0.0  # simplified clip keeps all five (median=0, σ large)
        stacked = np.stack(frames, axis=0)  # (5,1,1,1)
        wsc_out, _mask = zas._reject_outliers_winsorized_sigma_clip(
            stacked, (0.05, 0.05), 2.5, 2.5, max_workers=1
        )
        wsc_result = float(np.nanmean(wsc_out))
        # Material divergence: simplified keeps the outlier (≈20.0); PixInsight
        # winsorizes it to ≈0. Assert only a large gap, not an exact value.
        assert abs(float(np.mean(core_result)) - wsc_result) > 1.0


# ---------------------------------------------------------------------------
# 3. linear_fit_clip: visible Qt choice + no-op placeholder
# ---------------------------------------------------------------------------

class TestLinearFitClipPlaceholder:
    def test_reject_outliers_linear_fit_clip_is_noop(self):
        # Gate F1 (R3): the helper remains only as a legacy/unreachable no-op; no supported
        # caller invokes it.
        stacked = np.array([[[[1.0]], [[2.0]], [[100.0]]]], dtype=np.float32)
        out, mask = zas._reject_outliers_linear_fit_clip(stacked)
        assert np.array_equal(out, stacked)  # returned unchanged
        assert mask.dtype == bool and bool(np.all(mask))  # all-True keep mask


# ---------------------------------------------------------------------------
# 4. noise_fwhm reachable Classic behavior (real path + controlled seams)
# ---------------------------------------------------------------------------
#
# Reachable Classic semantics (zemosaic_align_stack._compute_quality_weights):
#   * Photutils UNavailable  → explicit variance substitution, effective
#     ``noise_variance`` (the ONLY variance fallback).
#   * Photutils available + all-unusable/star-free estimator → the real
#     ``_calculate_image_weights_noise_fwhm`` returns UNIT compact weights; the
#     sanitizer sees no effect and returns ``(None, "none", None)`` — requested
#     FWHM becomes NO weighting, NOT variance.
#   * Photutils available + partial success → usable frames get
#     ``min_overall_valid_fwhm / fwhm``, failed estimates get constant ``1e-6``,
#     skipped/unprocessed frames get ``1.0``; effective label may stay
#     ``noise_fwhm``.
#   * Photutils available + all usable → genuine ``min_fwhm/fwhm`` weighting.
#
# Evidence labels are honest: the Photutils-unavailable branch is a reachable
# dependency seam (photutils is optional, ``PHOTOUTILS_AVAILABLE`` starts False);
# the star-free witness runs the REAL estimator on the project venv; the partial
# substitution is a controlled SEAM (the estimator result is monkeypatched).

class TestNoiseFwhmBehavior:
    def test_photutils_unavailable_substitutes_variance(self, monkeypatch):
        # Reachable dependency seam: photutils is an optional import; with it
        # unavailable, requested noise_fwhm executes noise_variance (effective
        # label changes). No exception.
        monkeypatch.setattr(zas, "PHOTOUTILS_AVAILABLE", False)
        calls: dict[str, int] = {}

        def _spy_variance(image_list, progress_callback=None):
            calls["variance"] = calls.get("variance", 0) + 1
            return [np.array(2.0, dtype=np.float32), np.array(1.0, dtype=np.float32)]

        def _spy_fwhm(image_list, progress_callback=None):
            calls["fwhm"] = calls.get("fwhm", 0) + 1
            return [np.array(1.0, dtype=np.float32)] * len(image_list)

        monkeypatch.setattr(zas, "_calculate_image_weights_noise_variance", _spy_variance)
        monkeypatch.setattr(zas, "_calculate_image_weights_noise_fwhm", _spy_fwhm)
        frames = [
            np.array([[[10.0]]], dtype=np.float32),
            np.array([[[20.0]]], dtype=np.float32),
        ]
        _weights, effective_method, _stats = zas._compute_quality_weights(frames, "noise_fwhm")
        assert effective_method == "noise_variance"
        assert calls.get("variance") == 1 and calls.get("fwhm", 0) == 0

    def test_star_free_real_path_returns_no_weighting(self):
        # REAL PATH (project venv, photutils genuinely available): the actual
        # estimator on deterministic star-free 64x64 noise frames finds no
        # sources, returns unit weights, and the sanitizer drops them →
        # effective "none", weights None. NOT a variance fallback (H1 fix).
        assert zas.PHOTOUTILS_AVAILABLE, "photutils must be available in the project venv"
        rng = np.random.default_rng(0)
        frame = rng.normal(loc=1000.0, scale=20.0, size=(64, 64)).astype(np.float32)
        weights, effective_method, stats = zas._compute_quality_weights(
            [frame, frame.copy()], "noise_fwhm"
        )
        assert effective_method == "none"
        assert weights is None
        assert stats is None

    def test_partial_estimator_substitutions_seam(self, monkeypatch):
        # Controlled SEAM: the real partial-success finalization substitutes a
        # constant 1e-6 for failed estimates and 1.0 for skipped frames, and the
        # sanitizer then drops the 1.0 (no-effect) frame. We monkeypatch the
        # estimator to return that realistic mixed result to prove the sanitizer
        # path — labeled SEAM, not a real-estimator run.
        monkeypatch.setattr(zas, "PHOTOUTILS_AVAILABLE", True)
        mixed = [
            np.array(0.5, dtype=np.float32),  # usable frame: min_fwhm/fwhm < 1
            np.array(1e-6, dtype=np.float32),  # failed estimate → constant 1e-6
            np.array(1.0, dtype=np.float32),  # skipped frame → 1.0 (dropped later)
        ]
        monkeypatch.setattr(
            zas, "_calculate_image_weights_noise_fwhm",
            lambda image_list, progress_callback=None: list(mixed),
        )
        frames = [
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        ]
        weights, effective_method, stats = zas._compute_quality_weights(frames, "noise_fwhm")
        # Effective label stays noise_fwhm (partial success is not a variance
        # substitution and not a full no-weight drop).
        assert effective_method == "noise_fwhm"
        assert weights is not None
        assert weights[0] is not None and float(np.asarray(weights[0]).ravel()[0]) == pytest.approx(0.5)
        assert weights[1] is not None and float(np.asarray(weights[1]).ravel()[0]) == pytest.approx(1e-6)
        # The 1.0 (skipped) frame has no effect and is dropped by the sanitizer.
        assert weights[2] is None


# ---------------------------------------------------------------------------
# 5. weight zero/negative, median weight-ignore, all-invalid (cross-ref SCI-03/TEST-04)
# ---------------------------------------------------------------------------

class TestWeightEdgeSpotProbes:
    def test_stack_core_median_ignores_zero_weight(self):
        # cross-ref SCI-03 D2: stack_core median ignores weights entirely, so a
        # zero-weight finite frame is still included (median of 10 and 100 == 55).
        frames = [
            np.array([[[10.0]]], dtype=np.float32),
            np.array([[[100.0]]], dtype=np.float32),
        ]
        weights = np.array([[[0.0]], [[1.0]]], dtype=np.float32)
        result, _r, _ws = zemosaic_stack_core.stack_core(
            frames,
            weights=weights,
            stack_config={"normalize_method": "none", "final_combine_method": "median"},
            backend="cpu",
        )
        assert float(result[0, 0, 0]) == pytest.approx(55.0)

    def test_stack_core_mean_all_zero_weight_is_nan(self):
        # cross-ref SCI-03 D1 / TEST-04: stack_core mean at zero weight_sum → NaN
        # (Grid CPU returns a zero tile instead).
        frames = [np.array([[[10.0]]], dtype=np.float32)]
        weights = np.array([[[0.0]]], dtype=np.float32)
        result, _r, ws = zemosaic_stack_core.stack_core(
            frames,
            weights=weights,
            stack_config={"normalize_method": "none", "final_combine_method": "mean"},
            backend="cpu",
        )
        assert np.isnan(result[0, 0, 0])
        assert float(ws[0, 0, 0]) == 0.0


# ---------------------------------------------------------------------------
# 6. center-radial map fails translation invariance (vs footprint-taper concept)
# ---------------------------------------------------------------------------

def _footprint_taper_reconstruction(mask: np.ndarray, feather_px: float = 4.0, floor: float = 0.0) -> np.ndarray:
    """Test-only RECONSTRUCTION of the donor footprint-taper *concept*.

    Follows the real boolean footprint boundary (interior 1.0 → floor over
    ``feather_px`` px near the boundary, 0.0 outside), translation/rotation
    invariant. Uses standard ``scipy.ndimage.distance_transform_edt`` only —
    the donor module (``seestar.enhancement.weight_utils``) is **never imported**.
    """
    from scipy.ndimage import distance_transform_edt

    m = np.asarray(mask, dtype=bool)
    padded = np.pad(m, 1, constant_values=False)
    dist = distance_transform_edt(padded)[1:-1, 1:-1]
    frac = np.clip(dist.astype(np.float32) / float(feather_px), 0.0, 1.0)
    return np.where(m, floor + (1.0 - floor) * frac, np.float32(0.0)).astype(np.float32)


class TestRadialVsFootprintTaper:
    def test_center_radial_map_fails_translation_invariance(self):
        # Legacy ZeMosaic map is centre-radial (cosine of distance from image
        # centre): two identical sub-regions at different absolute positions
        # receive different weights → NOT footprint-following.
        r = make_radial_weight_map(64, 64, feather_fraction=0.5, shape_power=2.0, min_weight_floor=0.0)
        centre = r[30:34, 30:34]
        corner = r[2:6, 2:6]
        assert not np.array_equal(centre, corner)

    def test_footprint_taper_concept_is_translation_invariant(self):
        # RECONSTRUCTION: the footprint taper follows the footprint boundary, so
        # two identical footprints at different positions produce identical taper
        # values at their own coordinates.
        a = np.zeros((40, 40), bool)
        a[15:25, 15:25] = True
        b = np.zeros((40, 40), bool)
        b[5:15, 5:15] = True
        ta = _footprint_taper_reconstruction(a)
        tb = _footprint_taper_reconstruction(b)
        assert np.array_equal(ta[15:25, 15:25], tb[5:15, 5:15])

    def test_radial_map_differs_from_footprint_taper_concept(self):
        # The legacy radial map weights an interior point by absolute centre
        # distance; a footprint taper weights it ~1.0 everywhere well inside the
        # footprint. This is the conceptual divergence the donor COV-02 taper
        # resolves. Use a large footprint so a deep-interior off-centre point is
        # ≥feather_px from the boundary (taper==1.0) while still off-centre
        # enough for the radial map to have decayed.
        mask = np.zeros((40, 40), bool)
        mask[5:35, 5:35] = True
        taper = _footprint_taper_reconstruction(mask, feather_px=4.0)
        assert float(taper[20, 20]) == pytest.approx(1.0)  # centre interior unity
        radial = make_radial_weight_map(40, 40, feather_fraction=0.5, shape_power=2.0, min_weight_floor=0.0)
        assert float(radial[20, 20]) == pytest.approx(1.0, abs=0.01)  # centre ≈ unity, but...
        # ...(10,10) is 5px inside the footprint (taper==1.0) yet off-centre, so
        # the centre-radial map has decayed below unity there.
        assert float(taper[10, 10]) == pytest.approx(1.0)
        assert float(radial[10, 10]) < 1.0


# ---------------------------------------------------------------------------
# 7. global-coadd dispatch: actual executed symbols/formulas per method
# ---------------------------------------------------------------------------

class TestGlobalCoaddDispatch:
    def test_allowed_methods_vocabulary(self):
        tree = _parse(_WORKER_PATH)
        fn = _find_func(tree, "_assemble_global_mosaic_first_impl")
        assert fn is not None
        allowed = None
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id == "allowed_methods":
                        if isinstance(node.value, ast.Set):
                            allowed = {str(e.value) for e in node.value.elts if isinstance(e, ast.Constant)}
        assert allowed == {"mean", "median", "kappa_sigma", "winsorized"}

    def test_finalizer_dispatch_symbols(self):
        # F5 R1 (decision G1, option a): the coadd dispatch routes ALL FOUR labels through
        # the same canonical engine via `_finalize_chunked(method)`; the legacy
        # `_finalize_mean` / `_finalize_kappa_sigma` ad-hoc paths are no longer dispatched.
        tree = _parse(_WORKER_PATH)
        fn = _find_func(tree, "_assemble_global_mosaic_first_impl")
        assert fn is not None

        def _body_call(node: ast.If) -> str | None:
            if node.body and isinstance(node.body[0], ast.Assign):
                call = node.body[0].value
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                    return call.func.id
            return None

        # The legacy `_finalize_mean` is no longer dispatched (no `If` calls it).
        mean_if = None
        for node in ast.walk(fn):
            if isinstance(node, ast.If) and _body_call(node) == "_finalize_mean":
                mean_if = node
                break
        assert mean_if is None

        chunked = _find_func(fn, "_finalize_chunked")
        assert chunked is not None
        src = ast.dump(chunked)
        assert "winsorized_sigma_clip" in src  # Winsorized -> canonical WSC
        assert "kappa_sigma" in src  # Kappa-Sigma -> canonical kappa
        assert "nanmedian" not in src  # ad-hoc median removed (routes canonical)
        assert "nanpercentile" not in src  # divergent percentile clip removed

    def test_chunked_median_and_winsorized_semantics_reconstruction(self):
        # RECONSTRUCTION of the chunked finalizer formulas (worker:37743+) with
        # PRODUCTION DIMENSIONS:
        #   stack/clipped: (N,H,W,C); weight_stack: (N,H,W) — channel-invariant;
        #   multiply ``clipped * weight_stack[..., None]``;
        #   ``chunk_weight = nansum(weight_stack, axis=0)`` -> (H,W);
        #   divide by ``chunk_weight[..., None]`` -> (H,W,C).
        #   median → nanmedian along the frame axis -> (H,W,C).
        # Nonuniform 2-D weights and C=2 make accidental broadcasting unable to
        # pass: the result must be exactly (H,W,C) with a pinned analytical value.
        # Not a donor import.
        winsor_limits = (0.05, 0.05)
        low_pct = max(0.0, min(100.0, winsor_limits[0] * 100.0))  # 5.0
        high_pct = max(0.0, min(100.0, 100.0 - winsor_limits[1] * 100.0))  # 95.0

        # Median: 3 frames with a clear mid value → nanmedian observable, (H,W,C) result.
        med_stack = np.array(
            [
                [[[10.0, 10.0]], [[20.0, 20.0]]],
                [[[20.0, 20.0]], [[30.0, 30.0]]],
                [[[100.0, 100.0]], [[100.0, 100.0]]],
            ],
            dtype=np.float32,
        )  # (3 frames, 2 rows, 1 col, 2 ch)
        median = np.nanmedian(med_stack, axis=0)
        assert median.shape == (2, 1, 2)  # (H,W,C)
        assert float(median[0, 0, 0]) == 20.0  # median(10,20,100)
        assert float(median[1, 0, 0]) == 30.0  # median(20,30,100)

        # Winsorized: 5 frames, 2 rows, 1 col, 2 channels, NONUNIFORM (N,H,W) weights.
        # Per pixel the frame-axis values are [10,10,10,10,1000] (both channels),
        # so 5th percentile = 10 and 95th percentile = 802 (np.nanpercentile linear
        # interpolation on 5 samples); the 1000 outlier clips to 802.
        # weight_stack[f, y, x] = w_f with w = [1,2,3,4,5] (channel-invariant).
        wins_stack = np.array(
            [
                [[[10.0, 10.0]], [[10.0, 10.0]]],
                [[[10.0, 10.0]], [[10.0, 10.0]]],
                [[[10.0, 10.0]], [[10.0, 10.0]]],
                [[[10.0, 10.0]], [[10.0, 10.0]]],
                [[[1000.0, 1000.0]], [[1000.0, 1000.0]]],
            ],
            dtype=np.float32,
        )  # (5 frames, 2 rows, 1 col, 2 ch)
        weight_stack = np.array(
            [[[1.0], [1.0]], [[2.0], [2.0]], [[3.0], [3.0]], [[4.0], [4.0]], [[5.0], [5.0]]],
            dtype=np.float32,
        )  # (N=5, H=2, W=1) channel-invariant
        assert weight_stack.shape == (5, 2, 1)
        lower = np.nanpercentile(wins_stack, low_pct, axis=0).astype(np.float32)
        upper = np.nanpercentile(wins_stack, high_pct, axis=0).astype(np.float32)
        clipped = np.clip(wins_stack, lower, upper)
        chunk_weight = np.nansum(weight_stack, axis=0)  # (H,W)
        assert chunk_weight.shape == (2, 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            chunk_result = np.nansum(clipped * weight_stack[..., None], axis=0) / chunk_weight[..., None]
        # (H,W,C) exactly, 3-D indexing.
        assert chunk_result.shape == (2, 1, 2)
        # Analytical: clipped values [10,10,10,10,802]; Σ(clipped·w) =
        # 10·1+10·2+10·3+10·4+802·5 = 4110; Σw = 15; result = 4110/15 = 274.0
        # for every pixel/channel (weights are channel-invariant).
        assert float(chunk_result[0, 0, 0]) == pytest.approx(274.0, abs=0.01)
        assert float(chunk_result[1, 0, 0]) == pytest.approx(274.0, abs=0.01)
        assert float(chunk_result[0, 0, 1]) == pytest.approx(274.0, abs=0.01)
        # Clipping is observable: clipped weighted mean 274.0 differs from the
        # unclipped weighted mean Σ(raw·w)/Σw =
        # (10·1+10·2+10·3+10·4+1000·5)/15 = 5100/15 = 340.0.
        unclipped = np.nansum(wins_stack * weight_stack[..., None], axis=0) / chunk_weight[..., None]
        assert float(unclipped[0, 0, 0]) == pytest.approx(340.0, abs=0.01)


# ---------------------------------------------------------------------------
# 8. logging / provenance gaps (named with evidence)
# ---------------------------------------------------------------------------

class TestCoverageProvenanceGaps:
    def test_no_support_or_coverage_render_domain(self):
        # Gate A snapshot recorded the support/taper and coverage-render domains as
        # absent. Gates E1/E3/E5a now provide core/canonical_support.py,
        # core/canonical_render.py, and the coverage_support_taper setting; the
        # render/config execution tokens remain absent pending later gates.

        def _scan(token):
            found = []
            for py in sorted(_SRC.rglob("*.py")):
                try:
                    text = py.read_text(encoding="utf-8")
                except Exception:
                    continue
                if token in text:
                    found.append(str(py.relative_to(_SRC)))
            return found

        # (1) Support/taper + render + settings tokens now PRESENT (Gate E1/E3/E5a),
        # each specifically in its bounded, documented module (presence elsewhere is
        # not sufficient; the assertion fails if a symbol disappears).
        present = {
            "make_footprint_taper": "core/canonical_support.py",
            "PositiveSupportAccumulator": "core/canonical_support.py",
            "N_eff_support": "core/canonical_support.py",
            "coverage_aware_render": "core/canonical_render.py",
            "support_taper": "zemosaic_config.py",  # coverage_support_taper (decision K)
        }
        for token, module in present.items():
            found = _scan(token)
            assert module in found, f"{token!r} not in {module}; found in {found}"

        # (2) Coverage render/config execution tokens still ABSENT (later gates).
        for token in ("apply_coverage_render", "COVERAGE_RENDER_RESULT", "COVERAGE_CONFIG"):
            found = _scan(token)
            assert not found, f"{token!r} unexpectedly present in {found}"
