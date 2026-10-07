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

# Product settings the ZeGrid engine currently IGNORES (CPU-only engine + no
# post-stack processing). Kept as an explicit list so the run log + manifest can
# surface them and nothing is silently dropped.
#   * GPU stack/grid routing: use_gpu_stack / use_gpu_grid / stack_use_gpu.
#   * Post-processing (ZeGrid performs no DBE / inter-tile blend / normalization
#     rework / anchor review / coverage renorm).
IGNORED_SETTINGS: tuple[str, ...] = (
    "use_gpu_stack",
    "use_gpu_grid",
    "stack_use_gpu",
    "intertile_affine_blend",
    "center_out_normalization_p3",
    "enable_poststack_anchor_review",
    "two_pass_coverage_renorm",
)

# Settings matched by prefix (final_mosaic_dbe_* : enable / sigma / iterations / ...).
IGNORED_SETTING_PREFIXES: tuple[str, ...] = ("final_mosaic_dbe_",)

# ``run_zegrid_mode`` arguments the ZeGrid engine ACCEPTS (for backward
# compatibility with the removed legacy Grid) but does NOT honour, because
# ZeGrid uses the FROZEN science config (sky_mean / noise_variance / kappa_sigma
# / mean / footprint taper) and emits standard ``mosaic_grid.fits`` +
# ``mosaic_grid_coverage.fits``. Surfaced so nothing is silently dropped.
IGNORED_RUN_ARGS: tuple[str, ...] = (
    "stack_weight_method",
    "stack_reject_algo",
    "stack_kappa_low",
    "stack_kappa_high",
    "winsor_limits",
    "stack_final_combine",
    "apply_radial_weight",
    "radial_feather_fraction",
    "radial_shape_power",
    "save_final_as_uint16",
    "legacy_rgb_cube",
    "grid_rgb_equalize",
    "use_gpu",
)

# ZM-ZEGRID-R12 F3: the precise answer to the user's observation that loading
# "uses the GPU". The ZeGrid ENGINE is CPU-only, but the PRODUCT worker still
# probes/initialises CuPy during its own (pre/post ZeGrid) phases.
GPU_USAGE_NOTE = (
    "The ZeGrid ENGINE is CPU-only: it performs all decode/reproject/stack work "
    "on the CPU and never touches the GPU; its use_gpu_* / stack_use_gpu flags "
    "are read but ignored. The PRODUCT worker may still initialise CuPy during "
    "its own phases (gpu_runtime probing / apply_gpu_safety_to_phase5_flag do "
    "cupy.cuda.Device().use() and cupy.is_available() in zemosaic_worker.py), so "
    "a GPU spike during loading most likely comes from that product-level "
    "initialisation, NOT from the ZeGrid engine."
)


class Timings:
    """Accumulate wall-clock seconds per named phase (repeated samples sum)."""

    def __init__(self) -> None:
        self._totals: dict[str, float] = {}
        self._order: list[str] = []

    def add(self, name: str, seconds: float) -> None:
        if name not in self._totals:
            self._order.append(name)
            self._totals[name] = 0.0
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
        return float(sum(self._totals.values()))

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


def ignored_settings_present(zconfig) -> dict:
    """Return ``{setting_name: value}`` for every ignored setting present.

    ``zconfig`` is the product config object (a ``SimpleNamespace`` in the
    production path). ``None`` yields an empty dict. Values are read by name
    (exact) or by prefix (``final_mosaic_dbe_*``); a setting is reported only when
    it is truthy (so we warn about what is actually *set*, not a long default list).
    """
    out: dict = {}
    if zconfig is None:
        return out
    for name in IGNORED_SETTINGS:
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
            "No ignored ZeGrid settings detected (no use_gpu_*/stack_use_gpu or "
            "post-stack-processing flags set)."
        ]
    lines = [f"ZeGrid ignores the following product settings (CPU-only engine):"]
    for name in sorted(ignored):
        lines.append(f"  - {name} = {ignored[name]!r}")
    return lines
