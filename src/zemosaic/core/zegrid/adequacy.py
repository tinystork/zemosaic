"""ZM-ZEGRID-R2 — witness-adequacy metrics (closes Nono M1's "silent" part).

Nono M1 (MEDIUM): the frozen corner Cell ``r0000c0000`` is a scientifically
weak, effectively **single-frame** witness — only 1 of its 7 patch contributors
survives normalization (valid fraction ~8.8%) — and R1 did not make that
limitation explicit or guard it.

This module makes witness adequacy **explicit and non-silent**. Every cell run
(sweep candidate, chosen witness, corner) persists the following scalar fields
in its provenance JSON:

* ``total_contributors``            — number of patch contributors fed to the stack.
* ``effective_contributor_count``   — number of frames that survive normalization
                                      (+ weighting) and actually contribute at least
                                      one accepted sample (``N - excluded_count``).
* ``max_surviving_sample_count``    — peak per-pixel stacking depth (max of the
                                      engine ``surviving_sample_count`` plane).
* ``valid_fraction``                — ``mean(valid_mask)`` over all pixels/channels.
* ``single_frame_witness``          — bool: ``effective_contributor_count <= 1``.

Documented limitation (non-silent): a cell whose
``effective_contributor_count == 1`` (or ``0``) is a **single-frame witness**; it
exercises the geometry/local-read/reprojection/exclusion paths but NOT genuine
multi-frame canonical stacking (weighted mean / rejection over >=3 frames). This
is a scientific adequacy statement, not a pipeline defect.

NOTE: ``effective_contributor_count`` counts *distinct surviving frames* (frames
that are not whole-frame excluded at normalization/weighting). It is **not** the
same as ``max_surviving_sample_count`` (the deepest per-pixel overlap); both are
recorded because they answer different questions (how many frames contributed
anywhere vs how deep the stack gets at any single pixel).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WitnessAdequacy:
    """Scalar witness-adequacy fields persisted for every cell run."""

    total_contributors: int
    effective_contributor_count: int
    max_surviving_sample_count: int
    valid_fraction: float
    single_frame_witness: bool
    excluded_count: int

    def to_dict(self) -> dict:
        return {
            "total_contributors": int(self.total_contributors),
            "effective_contributor_count": int(self.effective_contributor_count),
            "max_surviving_sample_count": int(self.max_surviving_sample_count),
            "valid_fraction": float(self.valid_fraction),
            "single_frame_witness": bool(self.single_frame_witness),
            "excluded_count": int(self.excluded_count),
        }


def compute_adequacy(
    total_contributors: int,
    excluded: tuple[tuple[str, str, str], ...],
    surviving_sample_count: np.ndarray,
    valid_mask: np.ndarray,
) -> WitnessAdequacy:
    """Derive witness adequacy from a canonical result's provenance + planes.

    ``excluded`` is the adapter's ``(frame_id, stage, reason)`` tuple; each entry
    is one distinct whole-frame exclusion (normalization or weighting). The
    reference frame is never excluded, so ``effective`` is in ``[1, N]`` for any
    non-degenerate stack.
    """
    excluded_count = int(len(excluded))
    effective = int(total_contributors) - excluded_count
    if surviving_sample_count is not None and surviving_sample_count.size:
        max_surv = int(np.nanmax(surviving_sample_count))
    else:
        max_surv = 0
    if valid_mask is not None and valid_mask.size:
        valid_fraction = float(np.mean(valid_mask))
    else:
        valid_fraction = 0.0
    return WitnessAdequacy(
        total_contributors=int(total_contributors),
        effective_contributor_count=int(effective),
        max_surviving_sample_count=int(max_surv),
        valid_fraction=float(valid_fraction),
        single_frame_witness=bool(effective <= 1),
        excluded_count=int(excluded_count),
    )
