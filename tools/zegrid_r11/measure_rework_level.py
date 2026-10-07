#!/usr/bin/env python3
"""ZM-ZEGRID-R11 rework-1 — synthetic MOSAIC level measurement (F1 branch).

Proves numerically that, after the F1 fix, a Cell WITHOUT the global reference R
lands on R's GLOBAL level (not a re-anchored local level). Reports, per Cell, the
mean science LEVEL and the residual step vs R for three regimes:

  * BEFORE          — per-Cell normalization (each Cell auto-selects its own
                      reference; fixed=None).
  * AFTER-r0-buggy  — the r0 behaviour reproduced by hand (re-anchor to the Cell's
                      highest-weight frame L via b' = b - b_L).
  * AFTER-rework    — the corrected gauge (keep global coefficients, placeholder
                      reference index).

The synthetic mosaic is frames at DIFFERENT sky offsets with partial support masks,
so the global reference R (greatest valid support) is absent from some Cells.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SRC = Path("/home/tristan/.openclaw/workspace/worktrees/zemosaic-zegrid-r1/src")
sys.path.insert(0, str(SRC))

from zemosaic.core.canonical_engine import CanonicalStackRequest  # noqa: E402
from zemosaic.core.canonical_streaming import (  # noqa: E402
    FixedNormalization,
    InMemoryCanonicalProvider,
    compute_fixed_normalization,
    run_canonical_stack_streaming,
    subset_fixed_normalization,
)


def _full_support(h, w):
    return np.ones((h, w), dtype=bool)


def _request(arrays, masks):
    return CanonicalStackRequest(
        images=arrays, geometric_support=masks,
        normalization="sky_mean", weighting="noise_variance",
        rejection="kappa_sigma", combine="mean", taper="footprint", taper_px=8.0,
    )


def _mean_level(res) -> float:
    return float(np.nanmean(res.science))


def _buggy_reanchor(gauge, global_ids, cell_ids) -> FixedNormalization:
    """Reproduce the r0 bug: re-anchor the subset to its highest-weight frame L."""
    sub = subset_fixed_normalization(gauge, global_ids, cell_ids)
    w = sub.weight_active
    anchor = int(np.argmax(np.where(w, sub.weights, -np.inf)))
    aL = sub.coefficients[anchor]  # (C, 2)
    new_coeff = np.full_like(sub.coefficients, np.nan)
    for j in range(len(cell_ids)):
        if not np.isfinite(sub.coefficients[j, 0, 0]):
            continue
        new_coeff[j, :, 0] = sub.coefficients[j, :, 0] / aL[:, 0]
        new_coeff[j, :, 1] = (sub.coefficients[j, :, 1] - aL[:, 1]) / aL[:, 0]
    return FixedNormalization(
        reference_index=anchor, coefficients=new_coeff,
        norm_active=sub.norm_active, weights=sub.weights,
        weight_active=sub.weight_active, exclusions=sub.exclusions,
        frame_ids=sub.frame_ids,
    )


def main() -> int:
    h, w = 48, 48
    rng = np.random.default_rng(7)
    offsets = np.array([0.0, 10.0, 20.0, -15.0, 35.0, -8.0])
    frames = [(100.0 + off + rng.normal(0.0, 0.5, (h, w))).astype(np.float32) for off in offsets]
    masks = [_full_support(h, w)] * len(frames)
    # Partial support: frame 2 loses its right half, frame 5 its bottom half, so the
    # greatest-valid-support reference is deterministic (frame 0, tie -> lowest).
    masks[2][:, w // 2:] = False
    masks[5][h // 2:, :] = False

    req = _request(frames, masks)
    gauge = compute_fixed_normalization(InMemoryCanonicalProvider(frames, masks), req)
    global_ids = list(range(len(frames)))
    R = int(gauge.reference_index)
    r_name = f"frame_{R}"

    # Cells: one containing R, several lacking R.
    cells = {
        "cell_with_R": [R, 1, 2],
        "cell_without_R_a": [1, 3, 4],
        "cell_without_R_b": [3, 4, 5],
    }

    def run(cell_ids, fixed):
        sub_arrays = [frames[i] for i in cell_ids]
        sub_masks = [masks[i] for i in cell_ids]
        return run_canonical_stack_streaming(
            InMemoryCanonicalProvider(sub_arrays, sub_masks),
            _request(sub_arrays, sub_masks), tile_size=16, fixed=fixed,
        )

    # R's global level = the corrected level of the cell containing R.
    r_level = _mean_level(run(cells["cell_with_R"], subset_fixed_normalization(gauge, global_ids, cells["cell_with_R"])))

    rows = []
    for name, cell in cells.items():
        has_R = R in cell
        # BEFORE: per-cell (fixed=None).
        before = _mean_level(run(cell, None))
        # AFTER-buggy: manual re-anchor.
        buggy = _mean_level(run(cell, _buggy_reanchor(gauge, global_ids, cell)))
        # AFTER-rework: corrected.
        after = _mean_level(run(cell, subset_fixed_normalization(gauge, global_ids, cell)))
        rows.append({
            "cell": name,
            "frame_ids": [f"frame_{i}" for i in cell],
            "contains_R": has_R,
            "BEFORE_level": round(before, 4),
            "BEFORE_step_vs_R": round(before - r_level, 4),
            "AFTER_buggy_level": round(buggy, 4),
            "AFTER_buggy_step_vs_R": round(buggy - r_level, 4),
            "AFTER_rework_level": round(after, 4),
            "AFTER_rework_step_vs_R": round(after - r_level, 4),
        })

    out = {
        "global_reference": r_name,
        "R_level": round(r_level, 4),
        "max_sky_offset_ADU": float(offsets.max()),
        "cells": rows,
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
