"""SCI-05 Gate F1 — placeholder removal (R3 ``linear_fit_clip`` removal).

Deterministic, hermetic tests for the frozen decision R3 (``linear_fit_clip`` removed from
supported choices, explicit ``unsupported_removed_sci05`` failure). No random data, no
network, no GPU.

(Historical note: the N4 ``stack_core`` ``linear_fit`` placeholder-removal assertions were
removed in ZM-ZEGRID-R9 — ``zemosaic_stack_core`` itself was dead code after the legacy Grid
removal and was deleted; the R3 assertions below guard shipped canonical behaviour and are
retained.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from zemosaic import zemosaic_align_stack

_SRC = Path(zemosaic_align_stack.__file__).resolve().parent

class TestLinearFitClipR3:
    def test_worker_raises_unsupported(self):
        # All three worker rejection sites raise the stable unsupported token (no call to
        # stack_linear_fit_clip): _stack_master_tile_cpu, the Phase 4.5 merge, and _stack_mosaics.
        worker_src = (_SRC / "zemosaic_worker.py").read_text(encoding="utf-8-sig")
        assert worker_src.count("unsupported_removed_sci05") == 3

    def test_gpu_error_token_aligned(self):
        gpu_src = (_SRC / "zemosaic_align_stack_gpu.py").read_text(encoding="utf-8-sig")
        assert "unsupported_removed_sci05" in gpu_src

    def test_qt_reject_options_absent(self):
        qt_src = (_SRC / "zemosaic_gui_qt.py").read_text(encoding="utf-8-sig")
        assert '"linear_fit_clip"' not in qt_src

    def test_tk_reject_keys_absent(self):
        tk_src = (_SRC / "zemosaic_gui.py").read_text(encoding="utf-8-sig")
        assert '"linear_fit_clip"' not in tk_src

    def test_legacy_helpers_still_present_but_unreachable(self):
        # The legacy helpers remain (marked unreachable); no supported caller uses them.
        assert callable(zemosaic_align_stack.stack_linear_fit_clip)
        assert callable(zemosaic_align_stack._reject_outliers_linear_fit_clip)


# ---------------------------------------------------------------------------
# R1 — centralized rejection-token guard (stack_aligned_images)
# ---------------------------------------------------------------------------

class TestRejectionTokenGuard:
    def _frames(self, shape=(70, 70, 3)):
        rng = np.random.default_rng(0)
        return [rng.normal(100.0, 10.0, shape).astype(np.float32) for _ in range(3)]

    def test_linear_fit_clip_raises(self):
        with pytest.raises(ValueError) as ei:
            zemosaic_align_stack.stack_aligned_images(
                self._frames(), rejection_algorithm="linear_fit_clip"
            )
        assert "unsupported_removed_sci05" in str(ei.value)

    def test_case_whitespace_variant_raises(self):
        with pytest.raises(ValueError) as ei:
            zemosaic_align_stack.stack_aligned_images(
                self._frames(), rejection_algorithm="  Linear_Fit_Clip "
            )
        assert "unsupported_removed_sci05" in str(ei.value)

    def test_unknown_token_raises(self):
        with pytest.raises(ValueError) as ei:
            zemosaic_align_stack.stack_aligned_images(
                self._frames(), rejection_algorithm="made_up_rejection"
            )
        assert "unknown" in str(ei.value)

    def test_supported_tokens_work(self):
        frames = self._frames()
        for token in ("none", "kappa_sigma", "winsorized_sigma_clip"):
            result = zemosaic_align_stack.stack_aligned_images(
                frames, rejection_algorithm=token
            )
            assert result is not None
            assert result.shape == (70, 70, 3)
            assert result.dtype == np.float32

    def test_guard_identity_for_supported_tokens(self):
        # The guard is a no-op (returns the same token) for the exact supported tokens,
        # so the science is bit-identical to the pre-rework behavior.
        assert zemosaic_align_stack._validate_rejection_token("none") == "none"
        assert zemosaic_align_stack._validate_rejection_token("kappa_sigma") == "kappa_sigma"
        assert (
            zemosaic_align_stack._validate_rejection_token("winsorized_sigma_clip")
            == "winsorized_sigma_clip"
        )
