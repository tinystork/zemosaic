"""ZM-ZEGRID-R12 — run instrumentation (timings, run log, GPU + ignored settings).

Pure observation, no science. Provides:

* :class:`Timings` — per-phase wall-clock accumulator (summed per named phase).
* :func:`ignored_settings_present` — the product settings the CPU-only ZeGrid
  engine currently reads but DOES NOT honour, surfaced so nothing is silently
  dropped (the user's real complaint: settings were silently ignored).
* :func:`describe_gpu_usage` — an explicit, truthful statement about GPU use.

Everything here is import-safe (stdlib only) so it can be reused by tests and by
the production orchestrator without pulling in the heavy decode/reproject stack.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

# Product settings the ZeGrid engine currently IGNORES (no post-stack
# processing). Kept as an explicit list so the run log + manifest can surface
# them and nothing is silently dropped.
#   * Post-processing (ZeGrid performs no DBE / inter-tile blend / normalization
#     rework / anchor review / coverage renorm).
#
# ZM-ZEGRID-R22: the GPU stack/grid routing flags (``use_gpu_stack`` /
# ``use_gpu_grid`` / ``stack_use_gpu``) are NO LONGER unconditionally ignored —
# they are now HONOURED by the GPU backend resolution. They are surfaced in
# ``ignored_settings`` ONLY when the GPU is requested but the backend could not be
# honoured (see ``GPU_SETTING_KEYS`` / ``ignored_settings_present(gpu_honoured=...)``).
IGNORED_SETTINGS: tuple[str, ...] = (
    "intertile_affine_blend",
    "center_out_normalization_p3",
    "enable_poststack_anchor_review",
    "two_pass_coverage_renorm",
)

# GPU flags that are honoured when the GPU backend is effective, and surfaced as
# ignored ONLY when the GPU is requested but NOT honoured (loud exact-CPU fallback).
GPU_SETTING_KEYS: tuple[str, ...] = (
    "use_gpu_stack",
    "use_gpu_grid",
    "stack_use_gpu",
)

# Settings matched by prefix (final_mosaic_dbe_* : enable / sigma / iterations / ...).
# ZM-ZEGRID-R18: ``final_mosaic_dbe_*`` is NO LONGER ignored — it is now honoured
# by the final-mosaic finishing step (see ``final_mosaic_finishing.py``), so the
# prefix list is empty.
IGNORED_SETTING_PREFIXES: tuple[str, ...] = ()

# ``run_zegrid_mode`` arguments the ZeGrid engine ACCEPTS (for backward
# compatibility with the removed legacy Grid) but does NOT honour, because
# ZeGrid uses the FROZEN science config (sky_mean / noise_variance / mean /
# footprint taper) and emits standard ``mosaic_grid.fits`` +
# ``mosaic_grid_coverage.fits``. Surfaced so nothing is silently dropped.
# NOTE (ZM-ZEGRID-R16): ``stack_reject_algo`` / ``stack_kappa_low`` /
# ``stack_kappa_high`` / ``winsor_limits`` are NO LONGER ignored — they are now
# mapped into the science config where the canonical engine supports them (see
# ``zemosaic_zegrid_mode.resolve_rejection_science``); unsupported values stay
# surfaced via that mapping's ``unhonoured`` dict, never silently dropped.
# NOTE (ZM-ZEGRID-R18): ``save_final_as_uint16`` and ``grid_rgb_equalize`` are
# NO LONGER ignored — they are now honoured by the final-mosaic finishing step.
# NOTE (ZM-ZEGRID-R22): ``use_gpu`` is NO LONGER ignored — it is now honoured by
# the GPU backend resolution (see ``zemosaic_zegrid_mode.resolve_gpu_preference``).
IGNORED_RUN_ARGS: tuple[str, ...] = (
    "stack_weight_method",
    "stack_final_combine",
    "apply_radial_weight",
    "radial_feather_fraction",
    "radial_shape_power",
    "legacy_rgb_cube",
)

# ZM-ZEGRID-R22: the GPU contract is now truthful. The rejection + combine stages
# CAN run on the GPU (opt-in); the gauge (normalization + weighting) and the
# support/taper construction stay on the CPU by the frozen canonical contract.
GPU_USAGE_NOTE = (
    "ZeGrid honours the user's GPU preference: when requested AND a CUDA device "
    "with sufficient free VRAM is available, the per-cell rejection + combine "
    "stages run on a VRAM-bounded tiled GPU path (one cell at a time); otherwise "
    "the engine degrades LOUDLY to the exact CPU backend with a recorded reason. "
    "The photometric gauge (normalization + weighting) and support/taper stages "
    "always run on the CPU. 'used' reports the EFFECTIVE per-cell backend, never "
    "CuPy initialisation alone."
)


class Timings:
    """Accumulate wall-clock seconds per named phase (repeated samples sum).

    ``subordinate=True`` marks a timing as a CHILD of another phase (e.g.
    ``layout.footprints`` under ``layout``): it still appears in ``to_dict()``
    (manifest + run-log listing) but is EXCLUDED from ``total()`` so the overall
    run wall-clock is never double-counted by summing a phase and its children.
    """

    def __init__(self) -> None:
        self._totals: dict[str, float] = {}
        self._order: list[str] = []
        self._subordinate: set[str] = set()

    def add(self, name: str, seconds: float, *, subordinate: bool = False) -> None:
        if name not in self._totals:
            self._order.append(name)
            self._totals[name] = 0.0
        if subordinate:
            self._subordinate.add(name)
        self._totals[name] += float(seconds)

    @contextmanager
    def timed(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - t0)

    def get(self, name: str) -> float:
        return self._totals.get(name, 0.0)

    def total(self) -> float:
        return float(sum(
            sec for name, sec in self._totals.items()
            if name not in self._subordinate
        ))

    def to_dict(self) -> dict:
        return {name: round(self._totals[name], 6) for name in self._order}

    def to_lines(self) -> list[str]:
        return [f"  {name}: {self._totals[name]:.3f}s" for name in self._order]


def _truthy(value) -> bool:
    """A setting is 'present' when it is a truthy, non-empty, non-default value."""
    if value is None:
        return False
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return value not in (0, 0.0)
    if isinstance(value, str):
        return value.strip() not in ("", "none", "false", "off", "0")
    return True


def ignored_settings_present(zconfig, *, gpu_honoured: bool = False) -> dict:
    """Return ``{setting_name: value}`` for every ignored setting present.

    ``zconfig`` is the product config object (a ``SimpleNamespace`` in the
    production path). ``None`` yields an empty dict. Values are read by name
    (exact) or by prefix (``final_mosaic_dbe_*``); a setting is reported only when
    it is truthy (so we warn about what is actually *set*, not a long default list).

    ``gpu_honoured`` (ZM-ZEGRID-R22): when True, the GPU stack/grid flags
    (``use_gpu_stack`` / ``use_gpu_grid`` / ``stack_use_gpu``) are NOT reported as
    ignored (they ARE honoured); when False they are reported when set, so a loud
    CPU fallback still surfaces the unhonoured user intent.
    """
    out: dict = {}
    if zconfig is None:
        return out
    for name in IGNORED_SETTINGS:
        value = getattr(zconfig, name, None)
        if _truthy(value):
            out[name] = value
    if not gpu_honoured:
        for name in GPU_SETTING_KEYS:
            value = getattr(zconfig, name, None)
            if _truthy(value):
                out[name] = value
    # Prefix matches (final_mosaic_dbe_*).
    for attr in dir(zconfig):
        for prefix in IGNORED_SETTING_PREFIXES:
            if attr.startswith(prefix):
                value = getattr(zconfig, attr, None)
                if _truthy(value):
                    out[attr] = value
    return out


def describe_gpu_usage() -> str:
    return GPU_USAGE_NOTE


def describe_ignored_run_args(ignored: dict) -> list[str]:
    """Human-readable lines listing ``run_zegrid_mode`` args accepted but ignored."""
    if not ignored:
        return ["No accepted-but-ignored run_zegrid_mode arguments detected."]
    lines = [
        "ZeGrid accepts but IGNORES the following stack/final-mosaic arguments "
        "(ZeGrid uses the frozen science config and standard FITS outputs):"
    ]
    for name in sorted(ignored):
        lines.append(f"  - {name} = {ignored[name]!r}")
    return lines


def ignored_settings_warning_lines(ignored: dict) -> list[str]:
    """Human-readable WARN lines listing ignored settings (or 'none set')."""
    if not ignored:
        return [
            "No ignored ZeGrid settings detected (no post-stack-processing flags "
            "set, and any GPU flags are honoured)."
        ]
    lines = [f"ZeGrid ignores the following product settings:"]
    for name in sorted(ignored):
        lines.append(f"  - {name} = {ignored[name]!r}")
    return lines
