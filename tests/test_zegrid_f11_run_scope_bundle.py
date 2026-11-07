"""F11 — one run-scope executable bundle, explicit precedence, no stale name."""

from __future__ import annotations

import inspect

from zemosaic import zemosaic_zegrid_mode as mode


class _Falsy:
    def __bool__(self):
        return False


def test_active_bundle_wins_even_when_falsy():
    active = _Falsy()
    final = object()
    assert mode._run_scope_bundle(active, final) is active


def test_final_bundle_is_the_fallback():
    final = object()
    assert mode._run_scope_bundle(None, final) is final


def test_no_outer_reference_to_the_gpu_local_accessor():
    src = inspect.getsource(mode)
    # The old name must be gone; only the GPU-local accessor may exist.
    assert "_gpu_run_scope_bundle" not in src
    stripped = src.replace("_gpu_current_bundle(", "")
    assert "_current_bundle(" not in stripped
