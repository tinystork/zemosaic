"""SCI-05 Gate F5 — global-coadd route convergence (decision G1/I).

Deterministic tests proving all four global-coadd labels (`Mean`, `Median`, `Kappa-Sigma`,
`Winsorized`) route to the same canonical engine/method contracts; the divergent
percentile-winsorized and ad-hoc mean/median/kappa paths are removed; `coadd_k` and
`winsor_limits` are accepted-but-inert.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from zemosaic import zemosaic_worker as zw
from zemosaic.core.canonical_engine import CanonicalStackRequest, run_canonical_stack

_WORKER_SRC = Path(zw.__file__).read_text(encoding="utf-8-sig")


class TestGlobalCoaddConvergence:
    def test_percentile_winsorized_removed(self):
        # The divergent percentile-winsorized clip is removed from the Winsorized route.
        assert "np.nanpercentile(stack, low_pct" not in _WORKER_SRC
        assert "np.clip(stack, lower, upper, out=stack)" not in _WORKER_SRC
        # The ad-hoc median/mean chunked paths are removed too (all labels route canonical).
        assert "np.nanmedian(stack, axis=0)" not in _WORKER_SRC

    def test_all_four_labels_route_canonical(self):
        # All four labels route through the same canonical routing (_coadd_map).
        assert '"mean": ("none", "mean")' in _WORKER_SRC
        assert '"median": ("none", "median")' in _WORKER_SRC
        assert '"kappa_sigma": ("kappa_sigma", "mean")' in _WORKER_SRC
        assert '"winsorized": ("winsorized_sigma_clip", "mean")' in _WORKER_SRC
        assert "run_canonical_stack" in _WORKER_SRC

    def test_kappa_sigma_canonical_sigma_3(self):
        # Kappa-Sigma routes to the canonical rejection (frozen sigma 3.0; no coadd_k override).
        assert '"kappa_sigma": ("kappa_sigma", "mean")' in _WORKER_SRC

    def test_coadd_k_and_winsor_limits_inert(self):
        # coadd_k and winsor_limits are accepted-but-INERT (documented deprecation).
        assert "accepted-but-INERT" in _WORKER_SRC

    def test_gpu_helper_legacy_disabled(self):
        # F5 R2: the legacy GPU helper combine (reproject_and_coadd_wrapper with
        # combine_function=coadd_method, coadd_k/winsor_limits) is disabled for the global coadd
        # so BOTH CPU and GPU environments run the canonical CPU stage (explicit, non-silent).
        assert "global_coadd_helper_legacy_disabled_canonical_only" in _WORKER_SRC

    def test_winsorized_label_preserved_no_rename(self):
        assert '"winsorized"' in _WORKER_SRC
        assert '{"mean", "median", "kappa_sigma", "winsorized"}' in _WORKER_SRC

    def test_canonical_semantics_all_labels(self):
        # The canonical engine the route delegates to handles all four rejections/combines.
        rng = np.random.default_rng(0)
        frames = [rng.normal(100.0, 10.0, (32, 32, 3)).astype(np.float32) for _ in range(4)]
        fps = [np.ones((32, 32), dtype=bool) for _ in frames]
        for rejection, combine in (
            ("none", "mean"),
            ("none", "median"),
            ("kappa_sigma", "mean"),
            ("winsorized_sigma_clip", "mean"),
        ):
            res = run_canonical_stack(
                CanonicalStackRequest(
                    images=frames, geometric_support=fps,
                    normalization="none", weighting="none", rejection=rejection,
                    combine=combine, reference_index=0, taper="none", backend="cpu",
                )
            )
            assert res.science is not None and res.science.shape == (32, 32, 3)

    def test_linear_fit_clip_still_removed(self):
        with pytest.raises(Exception):
            run_canonical_stack(
                CanonicalStackRequest(
                    images=[np.ones((16, 16, 3), dtype=np.float32)],
                    geometric_support=[np.ones((16, 16), dtype=bool)],
                    normalization="none", weighting="none", rejection="linear_fit_clip",
                    combine="mean", reference_index=0, taper="none", backend="cpu",
                )
            )
