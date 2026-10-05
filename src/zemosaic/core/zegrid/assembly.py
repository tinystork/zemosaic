"""ZM-ZEGRID-R1 — assembly stub: extract the Cell core slice from every plane.

Isolated R1 module. No production dispatch wiring. R1 does NOT assemble a
canvas, blend, DBE, or equalize. The MiniTile is the FULL ProcessingPatch
result; this module only extracts the deterministic core slice from every
science/support plane using the frozen ``core_slice``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .geometry import ProcessingPatch
from .science_adapter import MiniTileScienceResult


@dataclass(frozen=True)
class MiniTile:
    """One MiniTile = full ProcessingPatch science result + core slice crops."""

    cell_id: str
    patch_shape_hw: tuple[int, int]
    core_slice: tuple[int, int, int, int]  # (y0, y1, x0, x1)
    science: np.ndarray
    estimator_weight_sum: np.ndarray
    support_w1: np.ndarray
    support_w2: np.ndarray
    n_eff_support: np.ndarray
    valid_mask: np.ndarray
    surviving_sample_count: np.ndarray
    reference_frame_id: str | None
    excluded: tuple[tuple[str, str, str], ...]
    frame_order: tuple[str, ...]
    provenance: dict


def _crop_core(patch_plane: np.ndarray, core_slice: tuple[int, int, int, int]) -> np.ndarray:
    y0, y1, x0, x1 = core_slice
    return patch_plane[y0:y1, x0:x1]


def extract_minitile(
    patch: ProcessingPatch,
    science: MiniTileScienceResult,
) -> MiniTile:
    """Crop every result plane by the patch's core slice.

    The MiniTile keeps the full patch result as ``science``/``estimator_*``
    fields but also exposes the deterministic core crop in
    ``_core``-suffixed convenience fields. The caller must place the core at its
    global origin (no halo double-counting, no blend).
    """
    r = science.result
    cs = (patch.core_slice.y0, patch.core_slice.y1, patch.core_slice.x0, patch.core_slice.x1)
    return MiniTile(
        cell_id=patch.cell_id,
        patch_shape_hw=patch.patch_shape_hw,
        core_slice=cs,
        science=r.science,
        estimator_weight_sum=r.estimator_weight_sum,
        support_w1=r.support_w1,
        support_w2=r.support_w2,
        n_eff_support=r.n_eff_support,
        valid_mask=r.valid_mask,
        surviving_sample_count=r.surviving_sample_count,
        reference_frame_id=science.reference_frame_id,
        excluded=science.excluded,
        frame_order=science.frame_order,
        provenance=r.provenance,
    )


def crop_all_planes_to_core(minitile: MiniTile) -> dict[str, Any]:
    """Return the deterministic core crops of every plane (no placement)."""
    cs = minitile.core_slice
    y0, y1, x0, x1 = cs
    science_core = minitile.science[y0:y1, x0:x1] if minitile.science.ndim == 2 else minitile.science[y0:y1, x0:x1, :]
    return {
        "science_core": science_core,
        "estimator_weight_sum_core": _crop_core(minitile.estimator_weight_sum, cs),
        "support_w1_core": _crop_core(minitile.support_w1, cs),
        "support_w2_core": _crop_core(minitile.support_w2, cs),
        "n_eff_support_core": _crop_core(minitile.n_eff_support, cs),
        "valid_mask_core": _crop_core(minitile.valid_mask, cs),
        "surviving_sample_count_core": _crop_core(minitile.surviving_sample_count, cs),
    }
