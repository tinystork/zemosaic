"""SCI-05 Gate F3 — legacy radial weighting inert on all stacking paths (decision K).

Deterministic tests proving that ``apply_radial_weight=True`` is bit-identical to ``False``
(radial inert) on the covered paths, no ``make_radial_weight_map`` is ever called, the
deprecated config keys stay readable, and the GPU path remains inert (E5b).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from zemosaic import zemosaic_align_stack as zas
from zemosaic import zemosaic_config

_SRC = Path(zas.__file__).resolve().parent


def _frames(n=3, shape=(70, 70, 3)):
    rng = np.random.default_rng(0)
    return [rng.normal(100.0, 10.0, shape).astype(np.float32) for _ in range(n)]


def _legacy_call(frames, apply_radial_weight, **radial_params):
    return zas.stack_aligned_images(
        frames,
        normalize_method="none",
        weighting_method="none",
        rejection_algorithm="none",
        final_combine_method="mean",
        apply_radial_weight=apply_radial_weight,
        **radial_params,
    )


class TestRadialInert:
    def test_apply_radial_weight_true_equals_false(self):
        frames = _frames()
        res_true = _legacy_call(frames, True)
        res_false = _legacy_call(frames, False)
        assert res_true is not None and res_false is not None
        np.testing.assert_array_equal(res_true, res_false)

    def test_radial_params_ineffective(self):
        frames = _frames()
        a = _legacy_call(frames, True, radial_feather_fraction=0.1, radial_shape_power=1.0)
        b = _legacy_call(frames, True, radial_feather_fraction=0.9, radial_shape_power=6.0)
        assert a is not None and b is not None
        np.testing.assert_array_equal(a, b)

    def test_make_radial_weight_map_func_not_called(self, monkeypatch):
        frames = _frames()
        calls = []

        def detector(*args, **kwargs):
            calls.append((args, kwargs))
            raise AssertionError("make_radial_weight_map must not be called")

        monkeypatch.setattr(zas, "make_radial_weight_map_func", detector)
        monkeypatch.setattr(zas, "ZEMOSAIC_UTILS_AVAILABLE_FOR_RADIAL", True)
        _legacy_call(frames, True)
        assert calls == []

    def test_worker_no_radial_application(self):
        worker_src = (_SRC / "zemosaic_worker.py").read_text(encoding="utf-8-sig")
        # All worker radial application points were removed: no make_radial_weight_map call remains.
        assert "make_radial_weight_map" not in worker_src

    def test_gpu_path_still_inert(self):
        from zemosaic import zemosaic_align_stack_gpu as zasgpu
        assert zasgpu._compute_radial_weight_map(16, 16, 3, {"apply_radial_weight": True}, None) is None

    def test_deprecated_keys_readable(self):
        # The deprecated radial config keys remain readable (accepted-but-inert).
        cfg = zemosaic_config.load_config()
        for key in ("apply_radial_weight", "radial_feather_fraction", "radial_shape_power"):
            assert key in cfg

    def test_final_mosaic_headers_honest(self):
        # Both final-mosaic header sites are unconditional + honest: STK_RADW is always False
        # (inert), STK_RADFF/STK_RADPW/STK_RADFLR are omitted, and never a True "applied" claim.
        worker_src = (_SRC / "zemosaic_worker.py").read_text(encoding="utf-8-sig")
        assert "final_header['STK_RADW'] = (True" not in worker_src
        assert "final_header['STK_RADFF']" not in worker_src
        assert "final_header['STK_RADPW']" not in worker_src
        assert "final_header['STK_RADFLR']" not in worker_src
        assert (
            worker_src.count(
                "final_header['STK_RADW'] = (False, 'Stacking: Radial Weighting Applied (inert)')"
            )
            == 2
        )
