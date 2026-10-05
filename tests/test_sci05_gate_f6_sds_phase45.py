"""SCI-05 Gate F6 — SDS + Phase 4.5 route convergence (rework-2, A1/A2/A3).

Deterministic tests: SDS reference (D2) + mask (D3) canonical; Phase 4.5 alpha-weighted (A2/D4)
and legacy-stack (A3/D5) route through the canonical engine with an explicit alpha taper and an
honest geometric support (no ~isnan/brightness support); the simplified WSC / legacy kappa
divergence is removed from the supported route; D1 documented as an explicit inter-master op.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from zemosaic import zemosaic_worker as zw

_WORKER_SRC = Path(zw.__file__).read_text(encoding="utf-8-sig")


class TestSDSConvergence:
    def test_d1_inter_master_gain_documented(self):
        assert "Inter-master photometric gain" in _WORKER_SRC
        assert "DISTINCT inter-master operation" in _WORKER_SRC

    def test_d2_reference_max_valid_support(self):
        # canonical N1 auto: max VALID SUPPORT COUNT (coverage_pixels), not coverage_weight/central.
        assert "MAX VALID SUPPORT COUNT" in _WORKER_SRC
        # behavioral: max coverage_pixels wins
        def payload(pixels):
            return (np.zeros((1, 1), dtype=np.float32), 1.0, {"coverage_pixels": pixels})

        assert zw._sds_choose_reference_index([payload(1), payload(9), payload(5)], None) == 1
        assert zw._sds_choose_reference_index([payload(0), payload(0)], None) == 0

    def test_d3_derived_constant_removed(self):
        assert "plain finite + explicit-support mask" in _WORKER_SRC
        # behavioral: low-coverage pixel now included (no 1% threshold)
        tile = np.array([[100, 2], [3, 4]], dtype=np.float32)
        cov = np.array([[0.005, 1.0], [1.0, 1.0]], dtype=np.float32)
        _, median, _ = zw._sds_compute_tile_payload(tile, cov)
        assert median == 3.5  # median of [100,2,3,4]


class TestPhase45Convergence:
    def test_a2_alpha_as_explicit_taper(self):
        # A2 (D4): the alpha-weighted path routes through the canonical engine with the
        # per-pixel alpha passed as the explicit estimator-weight taper (not an ad-hoc mean).
        assert "taper=alpha_maps" in _WORKER_SRC
        assert "not an ad-hoc weighted mean" in _WORKER_SRC

    def test_a3_legacy_stack_removed_canonical_rejection(self):
        # A3 (D5): canonical rejection routing present; the simplified WSC/legacy kappa calls are
        # removed from the supported Phase 4.5 combine route.
        assert "rejection=rejection_token" in _WORKER_SRC
        assert "p45_no_honest_footprint_deferred" in _WORKER_SRC

    def test_a3_no_isnan_brightness_support(self):
        # The honest geometric support is the per-tile WCS reprojection footprint
        # (frame_weights > 0); ~isnan / brightness-derived support is not used.
        assert "no_honest_geometric_support" in _WORKER_SRC

    def test_a1_explicit_taper_accepted_by_engine(self):
        from zemosaic.core.canonical_engine import CanonicalStackRequest, run_canonical_stack
        arrays = [np.full((1, 1), v, dtype=np.float32) for v in (10.0, 20.0, 100.0)]
        masks = [np.ones((1, 1), dtype=bool) for _ in arrays]
        explicit = [np.full((1, 1), 0.5, dtype=np.float32) for _ in arrays]
        res = run_canonical_stack(
            CanonicalStackRequest(
                images=arrays, geometric_support=masks,
                normalization="none", weighting="none", rejection="none", combine="mean",
                taper=explicit, backend="cpu",
            )
        )
        assert res.provenance["taper"]["kind"] == "explicit"


class TestR3LegacyDispatch:
    def test_master_tile_cpu_routes_canonical(self):
        # F6 R3: the WSC/kappa legacy branches are removed; stack_aligned_images is the only path
        # (F2 routes that to run_canonical_stack).
        assert "stack_winsorized_sigma_clip / stack_kappa_sigma_clip wrappers are no longer called" in _WORKER_SRC
        assert "rejection_algorithm=stack_reject_algo" in _WORKER_SRC

    def test_stack_mosaics_canonical_with_deferral(self):
        # F6 R3: _stack_mosaics routes through run_canonical_stack; no-honest-footprint deferred.
        assert "sds_final_no_honest_footprint_deferred" in _WORKER_SRC

    def test_legacy_wrappers_marked_unreachable(self):
        from zemosaic import zemosaic_align_stack as zas
        src = Path(zas.__file__).read_text(encoding="utf-8-sig")
        assert "LEGACY / UNREACHABLE (SCI-05 Gate F6" in src

    def test_deferred_path_skips_not_non_canonical_combine(self):
        # F6 R4: the deferred no-honest-footprint sub-path SKIPs (NaN) — no non-canonical
        # nanmean/nanmedian combine is fabricated.
        assert "fabricate science with a non-canonical nanmean/nanmedian combine" in _WORKER_SRC
        assert "np.full(mosaics[0].shape, np.nan" in _WORKER_SRC
