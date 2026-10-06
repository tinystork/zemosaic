"""Direct ``zemosaic.core.robust_rejection`` shipped-behaviour contracts.

Restored (ZM-ZEGRID-R8 rework-1) from the deleted SCI-01 Grid WSC characterization
witness. The assertions below target the STILL-SHIPPED
``zemosaic.core.robust_rejection`` module directly (``resolve_wsc_impl`` and
``wsc_pixinsight_core``) and never depend on the removed ``grid_mode`` module.
``core/robust_rejection`` is used by ``zemosaic_align_stack`` /
``zemosaic_align_stack_gpu`` for winsorized-sigma-clip rejection.

Pinned semantics (as of the current tree):

* ``resolve_wsc_impl()`` resolves ``ZEMOSAIC_WSC_IMPL`` env (case-insensitive,
  trimmed) > config > default ``"pixinsight"``. Invalid env falls through to the
  default and never creates a new mode.
* ``wsc_pixinsight_core`` (PixInsight-like WSC) collapses a ``[0,0,0,0,100]``
  impulse to ~0 (the 100 outlier is winsorized), returns ~1.0 for
  ``[1,1,1,2,100]``, and ~6.5740323 for ``[1,2,3,4,5,100]`` — distinct from the
  simplified median/std clip in ``stack_core`` (see ``test_stack_core_contracts.py``).
"""

from __future__ import annotations

import numpy as np
import pytest

from zemosaic.core.robust_rejection import (
    WSC_IMPL_LEGACY,
    WSC_IMPL_PIXINSIGHT,
    resolve_wsc_impl,
    wsc_pixinsight_core,
)


_WSC_ENV = "ZEMOSAIC_WSC_IMPL"
# Matches zemosaic_align_stack._WSC_PIXINSIGHT_MAX_ITERS (pinned at 10).
_PIXINSIGHT_MAX_ITERS = 10

IMPULSE = [0.0, 0.0, 0.0, 0.0, 100.0]
NONDEGENERATE = [1.0, 1.0, 1.0, 2.0, 100.0]
GRADIENT = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]


def _data_stack(values):
    """(N,1,1,1) float32 stack from per-frame scalars."""
    patches = [np.array([[[float(v)]]], dtype=np.float32) for v in values]
    return np.stack(patches, axis=0)


def _px_core(values):
    return wsc_pixinsight_core(
        np,
        _data_stack(values),
        sigma_low=2.5,
        sigma_high=2.5,
        max_iters=_PIXINSIGHT_MAX_ITERS,
    )


# ---------------------------------------------------------------------------
# A. Resolver contract (hermetic env)
# ---------------------------------------------------------------------------


def test_resolver_env_absent_defaults_to_pixinsight(monkeypatch):
    monkeypatch.delenv(_WSC_ENV, raising=False)
    assert resolve_wsc_impl() == WSC_IMPL_PIXINSIGHT


def test_resolver_valid_legacy_env_overrides_default(monkeypatch):
    monkeypatch.setenv(_WSC_ENV, "legacy_quantile")
    assert resolve_wsc_impl() == WSC_IMPL_LEGACY


def test_resolver_valid_pixinsight_env(monkeypatch):
    monkeypatch.setenv(_WSC_ENV, "pixinsight")
    assert resolve_wsc_impl() == WSC_IMPL_PIXINSIGHT


def test_resolver_env_is_case_insensitive_and_trimmed(monkeypatch):
    monkeypatch.setenv(_WSC_ENV, " Legacy_Quantile ")
    assert resolve_wsc_impl() == WSC_IMPL_LEGACY
    monkeypatch.setenv(_WSC_ENV, "PIXINSIGHT")
    assert resolve_wsc_impl() == WSC_IMPL_PIXINSIGHT


def test_resolver_invalid_env_falls_through_to_default(monkeypatch):
    monkeypatch.setenv(_WSC_ENV, "bogus_impl")
    assert resolve_wsc_impl() == WSC_IMPL_PIXINSIGHT
    monkeypatch.setenv(_WSC_ENV, "")
    assert resolve_wsc_impl() == WSC_IMPL_PIXINSIGHT


def test_resolver_never_creates_new_mode_from_invalid_env(monkeypatch):
    monkeypatch.setenv(_WSC_ENV, "not-a-real-impl")
    got = resolve_wsc_impl()
    assert got in {WSC_IMPL_PIXINSIGHT, WSC_IMPL_LEGACY}
    assert got == WSC_IMPL_PIXINSIGHT


# ---------------------------------------------------------------------------
# B. PixInsight WSC core numerical contract
# ---------------------------------------------------------------------------


def test_wsc_pixinsight_core_impulse():
    pix = _px_core(IMPULSE)
    assert pix.shape == (1, 1, 1)
    # PixInsight WSC collapses [0,0,0,0,100] to ~0 (100 winsorized to ~2.5e-10).
    assert pix[0, 0, 0] == pytest.approx(5e-11, abs=1e-9)


def test_wsc_pixinsight_core_nondegenerate():
    pix = _px_core(NONDEGENERATE)
    assert pix.shape == (1, 1, 1)
    assert pix[0, 0, 0] == pytest.approx(1.0, abs=1e-6)


def test_wsc_pixinsight_core_gradient():
    pix = _px_core(GRADIENT)
    assert pix.shape == (1, 1, 1)
    assert pix[0, 0, 0] == pytest.approx(6.5740323, abs=1e-4)
