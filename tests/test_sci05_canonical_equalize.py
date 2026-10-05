"""SCI-05 Gate E4 — canonical RGB equalizer (deterministic parity witnesses).

Deterministic, hermetic tests for the out-of-place canonical RGB equalizer in
``zemosaic.core.canonical_equalize``, proving parity with the existing robust
reference implementation (``zemosaic_align_stack.equalize_rgb_medians_inplace``)
without importing the heavy module into the core. No randomness with global
seeds, no network, no filesystem writes.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from zemosaic.core.canonical_equalize import (
    equalize_rgb_medians_canonical,
    equalize_rgb_medians_copy,
)

# The heavy reference (imported here only — never from the core).
from zemosaic.zemosaic_align_stack import equalize_rgb_medians_inplace


def _rgb(seed, h=100, w=100, means=(100.0, 120.0, 90.0), scale=20.0):
    rng = np.random.default_rng(seed)
    return np.stack(
        [rng.normal(m, scale, (h, w)) for m in means], axis=2
    ).astype(np.float32)


def _assert_parity(arr, **kw):
    """Assert the canonical out-of-place port matches the in-place reference."""
    ref_arr = arr.copy()
    ref_info = equalize_rgb_medians_inplace(ref_arr, return_info=True, **kw)
    can_arr, can_info = equalize_rgb_medians_copy(arr, **kw)

    for k in ("decision", "samples", "mask_coverage", "raw_gains",
              "clipped_gains", "gain_r", "gain_g", "gain_b"):
        a, b = ref_info[k], can_info[k]
        if isinstance(a, str):
            assert a == b, f"{k}: {a!r} != {b!r}"
        else:
            np.testing.assert_allclose(
                np.asarray(a, dtype=float), np.asarray(b, dtype=float),
                rtol=1e-6, atol=1e-6, equal_nan=True,
            )
    # target_median (NaN-aware)
    a, b = ref_info["target_median"], can_info["target_median"]
    if np.isnan(a) or np.isnan(b):
        assert np.isnan(a) and np.isnan(b), f"target_median {a} != {b}"
    else:
        assert a == pytest.approx(b, rel=1e-6, abs=1e-6)
    # equalized array (NaN-aware)
    np.testing.assert_array_equal(can_arr, ref_arr)
    return can_arr, can_info


# ---------------------------------------------------------------------------
# Parity vs reference
# ---------------------------------------------------------------------------

class TestParity:
    def test_normal_applied(self):
        arr = _rgb(0)
        _, info = _assert_parity(arr)
        assert info["decision"] == "applied"
        assert info["applied"] is True

    def test_not_rgb_mono(self):
        mono = np.ones((20, 20), dtype=np.float32)
        _, info = _assert_parity(mono)
        assert info["decision"] == "invalid_input"

    def test_not_rgb_4d(self):
        arr = np.ones((10, 10, 3, 2), dtype=np.float32)
        _, info = _assert_parity(arr)
        assert info["decision"] == "invalid_input"

    def test_all_nan(self):
        arr = np.full((30, 30, 3), np.nan, dtype=np.float32)
        _, info = _assert_parity(arr)
        assert info["decision"] == "no_valid_pixels"

    def test_non_positive_pixels(self):
        arr = np.full((30, 30, 3), -1.0, dtype=np.float32)
        _, info = _assert_parity(arr)
        assert info["decision"] == "no_valid_pixels"

    def test_insufficient_samples(self):
        arr = _rgb(1, h=40, w=40)
        _, info = _assert_parity(arr, min_samples=100000)
        assert info["decision"] == "insufficient_samples"

    def test_insufficient_coverage(self):
        # Large image so the samples gate passes; a high coverage threshold so
        # the coverage gate (which keeps only the ~5-85th percentile band) fires.
        arr = _rgb(2, h=100, w=100)
        _, info = _assert_parity(arr, min_coverage=0.99)
        assert info["decision"] == "insufficient_coverage"

    def test_clip_saturation(self):
        # widely-separated channel medians -> raw gains far from 1 -> clipped
        arr = _rgb(3, means=(50.0, 100.0, 150.0))
        _, info = _assert_parity(arr, gain_clip=(0.98, 1.02))
        assert info["decision"] == "applied"
        for g in info["clipped_gains"]:
            assert 0.98 - 1e-6 <= g <= 1.02 + 1e-6

    def test_bg_percentile_string(self):
        arr = _rgb(4)
        _, info = _assert_parity(arr, bg_percentile="5;85")
        assert info["decision"] == "applied"


# ---------------------------------------------------------------------------
# Out-of-place purity / validation
# ---------------------------------------------------------------------------

class TestPurity:
    def test_input_not_mutated(self):
        arr = _rgb(5)
        arr_before = arr.copy()
        equalize_rgb_medians_copy(arr)
        np.testing.assert_array_equal(arr, arr_before)

    def test_out_of_place_result_owned(self):
        arr = _rgb(6)
        out, _ = equalize_rgb_medians_copy(arr)
        assert out.flags["OWNDATA"]
        assert not np.shares_memory(out, arr)

    def test_non_rgb_neutral_no_change(self):
        mono = np.ones((20, 20), dtype=np.float32)
        out, info = equalize_rgb_medians_copy(mono)
        np.testing.assert_array_equal(out, mono.astype(np.float32))
        assert info["applied"] is False
        assert info["decision"] == "invalid_input"

    def test_core_return_info_toggle(self):
        arr = _rgb(7)
        # return_info=False -> array only
        out = equalize_rgb_medians_canonical(arr)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float32
        # return_info=True -> (array, info)
        out2, info = equalize_rgb_medians_canonical(arr, return_info=True)
        np.testing.assert_array_equal(out, out2)
        assert isinstance(info, dict)

    def test_determinism(self):
        arr = _rgb(8)
        a, ia = equalize_rgb_medians_copy(arr)
        b, ib = equalize_rgb_medians_copy(arr)
        np.testing.assert_array_equal(a, b)
        assert ia == ib


# ---------------------------------------------------------------------------
# Info JSON-serializable
# ---------------------------------------------------------------------------

class TestInfo:
    def test_json_serializable_applied(self):
        arr = _rgb(9)
        _, info = equalize_rgb_medians_copy(arr)
        assert info["decision"] == "applied"
        json.dumps(info)  # must not raise

    def test_json_serializable_neutral(self):
        mono = np.ones((20, 20), dtype=np.float32)
        _, info = equalize_rgb_medians_copy(mono)
        assert info["decision"] == "invalid_input"
        json.dumps(info, allow_nan=True)  # NaN target_median is the only non-strict value

    def test_decision_strings_stable(self):
        expected = {
            "invalid_input", "no_valid_pixels", "invalid_luminance",
            "percentile_error", "insufficient_samples", "insufficient_coverage",
            "invalid_channel_medians", "invalid_target", "applied",
        }
        # non-RGB and all-NaN produce the two reachable neutral decisions
        mono = np.ones((20, 20), dtype=np.float32)
        _, i1 = equalize_rgb_medians_copy(mono)
        assert i1["decision"] in expected
        allnan = np.full((30, 30, 3), np.nan, dtype=np.float32)
        _, i2 = equalize_rgb_medians_copy(allnan)
        assert i2["decision"] in expected
