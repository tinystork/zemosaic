"""ZM-ZEGRID-R1 — isolated local MiniTile implementation (one real Cell).

Modules:
* ``geometry``   — manifest, canvas, layout, cell, patch, source ROI plans.
* ``execution``  — local section reads, local WCS reprojection, support map.
* ``science_adapter`` — build ONE CanonicalStackRequest, run canonical stack.
* ``assembly``   — core-slice extraction (no canvas/blend/DBE/equalize).
* ``adequacy``   — explicit witness-adequacy fields (closes Nono M1).
* ``sweep``      — deterministic multi-frame candidate sweep + memory guard.
* ``seam``       — adjacent-Cell continuity diagnostics (diagnostic, no blend).
* ``executor``   — R3 full-layout deterministic executor (row-major, sky_mean).
* ``mosaic``     — R3 deterministic canvas assembly (one-owner, no blend).

Not wired into production dispatch.

R55-C import-laziness note
-------------------------
This package used to import every submodule eagerly. That transitively imported
``reproject.adaptive`` → ``dask.array`` → ``cupy`` on a machine where CuPy was
installed, i.e. merely importing a CPU/ZeGrid module loaded CuPy. The package now
uses PEP 562 module ``__getattr__``: submodules are imported only when a caller
first asks for them. Historical imports keep working unchanged:

    from zemosaic.core.zegrid import assembly as za
    import zemosaic.core.zegrid as zg
    zg.assembly            # resolved lazily on attribute access

No CuPy/CUDA work is triggered by importing this package; only an explicit
``gpu_readiness.probe_gpu_readiness()`` (or a path that genuinely needs CuPy)
does that.
"""

from __future__ import annotations

import importlib

# Historical public surface: unchanged, so ``from ... import *`` is stable.
__all__ = [
    "geometry",
    "execution",
    "science_adapter",
    "assembly",
    "adequacy",
    "sweep",
    "seam",
    "executor",
    "mosaic",
    "auto_layout",
    "photometric",
    "observability",
    "final_mosaic_finishing",
]

# Every public submodule remains reachable as a package attribute, on demand.
_LAZY_SUBMODULES = frozenset(
    {
        "adequacy",
        "aesthetic_hole_fill",
        "assembly",
        "auto_layout",
        "background_diagnostics",
        "background_model",
        "calibration",
        "cell_feather",
        "cell_offset",
        "execution",
        "executor",
        "file_provider",
        "final_mosaic_finishing",
        "geometry",
        "gpu",
        "gpu_readiness",
        "instrumentation",
        "mosaic",
        "observability",
        "parallel",
        "photometric",
        # NOTE: the R40 diagnostic capture core is deliberately NOT listed here.
        # Its contract forbids wiring that core into the package init.
        "ram_contract",
        "resource_envelope",
        "scheduler",
        "science_adapter",
        "seam",
        "streaming",
        "sweep",
    }
)


def __getattr__(name: str):
    """Lazily import a public submodule on first attribute access (PEP 562)."""
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_SUBMODULES | set(__all__))
