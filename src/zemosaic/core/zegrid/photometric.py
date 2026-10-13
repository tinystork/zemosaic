"""ZM-ZEGRID-R11/R12 — global photometric gauge computation (bounded streaming).

Computes ONE reference frame + per-frame normalization coefficients/weights over
the frames' FULL valid footprint on the canvas, so every Cell of the mosaic
shares a single photometric anchor. This removes the systematic inter-cell level
steps (the "patchwork") caused by each Cell previously selecting ITS OWN
reference frame and computing its ``sky_mean`` offsets against it.

This module is the GLOBAL half of the gauge. The per-Cell consumption lives in
``canonical_streaming.run_canonical_stack_streaming(..., fixed=...)`` via
:func:`canonical_streaming.compute_fixed_normalization` and
:func:`canonical_streaming.subset_fixed_normalization`.

Design (R12 rework-1 — BOUNDED streaming phase-1, NO disk cache)
----------------------------------------------------------------
R11 materialised ONE FULL-CANVAS aligned plane per frame into ``<cache>/__gauge__``
and only deleted it AFTER use, so its transient peak was ``O(N x canvas)``
(66.8 GiB for Caldwell 11's 306 frames / 142.5 GiB for 653 / 178 GiB for 816 on a
4117x4377 canvas) — larger than the host's free disk, so the real run could not
complete. R12 rework-1 removes that cache entirely and computes the SAME gauge
frame-by-frame over BOUNDED regions, reusing the FROZEN phase-1 functions
verbatim on a 2-frame ``[R, i]`` batch:

* **Per-frame valid counts** (for reference selection) are computed by
  reprojecting each frame to ITS OWN projected-footprint bbox (a bounded,
  ~1/4-canvas region), never a full-canvas plane.
* **Reference selection** reuses the frozen ``select_canonical_reference``
  (greatest valid count, ties to the lowest sorted-FrameId index).
* The **reference** is reprojected ONCE to the full canvas (ONE plane, held in
  memory, shared to workers via fork copy-on-write) so it can be cropped to any
  frame's footprint bbox.
* **Per-frame normalization + weighting** reuses the frozen
  ``prepare_canonical_inputs`` / ``normalize_canonical_images`` /
  ``compute_canonical_quality_weights`` on a 2-frame ``[R_crop, i_crop]`` batch,
  where BOTH are cropped to the UNION of the reference's and frame i's footprint
  bboxes. Cropping to the union bbox (not just the R-i intersection, and not just
  i's bbox) is what keeps BOTH the QUALITY WEIGHT (over i's FULL footprint) AND
  the frozen ``min_common`` gate basis on the SAME footing as the full-canvas
  gauge: within the union bbox ``counts[0] == |R_full|`` and ``counts[1] ==
  |i_full|``, so the discrete outputs (reference index, active flags, exclusions)
  are IDENTICAL to the full-canvas gauge.

Peak RAM is O(1-2 planes) (~0.5 GiB: R's full canvas + one union bbox), and
peak DISK is ~0.

EQUIVALENCE (honesty note, ZM-ZEGRID-R12 rework-2): the bounded gauge is
DETERMINISTIC and matches the R11 full-canvas gauge EXACTLY when the bboxes ≈
canvas (near-full overlap, e.g. M16). On PARTIAL-OVERLAP corpora (bboxes strictly
smaller than the canvas) it is EQUIVALENT UP TO REPROJECTION FP NOISE (~1e-4
relative, negligible): frame i is reprojected onto a bbox SUB-canvas WCS (shifted
CRPIX) rather than the full-canvas WCS, so ``reproject_interp`` differs in the
last FP digits and the continuous coefficients/weights may differ ~1e-4 relative,
while the discrete outputs remain identical. This is therefore NOT claimed as
BIT-IDENTICAL on partial-overlap corpora (see the R12 partial-overlap test).

ZM-ZEGRID-R32a — internal (NON-DEFAULT) performance levers + instrumentation
----------------------------------------------------------------------------
This module now also exposes three INTERNAL levers, all DEFAULT OFF so the
production default path is bit-for-bit unchanged:

* **Lever A — geometric prefilter** (`prefilter_pairs`): before any reprojection,
  a pair is SKIPPED when a pure-geometry upper bound (the intersection area of
  the two footprint bboxes, a superset of each frame's support) proves the pair
  was excluded as ``insufficient_common`` anyway. Provably equivalent: never a
  false negative (a skipped pair is always excluded under the strict path).
* **Lever B — reference dedup** (`dedup_reference`): the reference frame's
  quality metrics are computed once per pair instead of twice; the extracted
  non-reference numbers are bit-identical.
* **Bounded work** (`bounded_pairs`): frame ``i`` is reprojected on ITS OWN
  footprint bbox instead of the union bbox(R, i), while the frozen ``min_common``
  gate still receives the KNOWN full valid counts from the counts pass. Same
  discrete decisions; coefficients equal within the declared reprojection FP
  tolerance (measured 0.0 on the R32a synthetic fixtures; declared <=1e-4).

The reference plane is still materialised as ONE full-canvas float32 plane
(+ bool support); the tiled/bounded reference and the remaining RAM scaling are
R32b items. The ``diagnostics`` dict gains a bounded ``instrumentation`` block
(timings/areas/bytes/RSS per subsystem; no per-pixel logging).
"""

from __future__ import annotations

import copy
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from astropy.wcs import WCS

from zemosaic.core.canonical_stacking import (
    REASON_INSUFFICIENT_COMMON,
    STAGE_NORMALIZATION,
    CanonicalInputBatch,
    CanonicalStackFailure,
    FrameExclusion,
    _compute_min_common,
    compute_canonical_quality_weights,
    normalize_canonical_images,
    prepare_canonical_inputs,
    select_canonical_reference,
)
from zemosaic.core.canonical_streaming import (
    FixedNormalization,
    compute_fixed_normalization,
)
from zemosaic.core.zegrid import execution as zxe
from zemosaic.core.zegrid import geometry as zg
from zemosaic.core.zegrid import parallel as zpar

__all__ = [
    "compute_global_gauge",
    "compute_cell_gauge",
    "resolve_cell_gauge_mode",
    "select_auto_gauge_mode",
    "CellGaugeResult",
    "CellReferencePin",
    "CELL_GAUGE_MODES",
    "CELL_GAUGE_MODE_DEFAULT",
    "CELL_GAUGE_MODE_GLOBAL",
    "CELL_GAUGE_MODE_PER_CELL",
    "CELL_GAUGE_MODE_AUTO",
]

# ---------------------------------------------------------------------------
# ZM-CELL-PHOTOMETRY-R34B — per-Cell photometric gauge MODE (opt-in)
# ---------------------------------------------------------------------------
#
# ``compute_global_gauge`` (above) is the R11/R12 production gauge: ONE reference
# frame + per-frame coefficients computed over the frame's FULL canvas footprint,
# then subset per Cell via ``canonical_streaming.subset_fixed_normalization``.
# That subset is deliberately a NO-RE-ANCHOR operation (R11 F1), so a Cell whose
# contributor frames have NO globally-active frame raises ``cell gauge has no
# active frame`` (84 such geometrically-populated Cells on the Caldwell corpus).
#
# R34b adds the opt-in ``per_cell`` MODE: instead of subsetting the global gauge,
# EVERY geometrical Cell computes its OWN self-anchored gauge from its own aligned
# frames (reference = the Cell's greatest-valid-support frame, ties to the lowest
# sorted FrameId), reusing the FROZEN estimator verbatim.
#
# ZM-CELL-PHOTOMETRY-R34F — ``auto`` (the PRODUCTION DEFAULT): resolve to
# ``"global"`` when the global gauge serves EVERY non-empty Cell (the R11 path,
# byte-identical, no offsets/feather), else switch the WHOLE run to ``"per_cell"``
# (R34b gauges + R34c additive offsets + R34d feather). The user does nothing.
CELL_GAUGE_MODE_GLOBAL = "global"
CELL_GAUGE_MODE_PER_CELL = "per_cell"
CELL_GAUGE_MODE_AUTO = "auto"
CELL_GAUGE_MODES = (CELL_GAUGE_MODE_GLOBAL, CELL_GAUGE_MODE_PER_CELL, CELL_GAUGE_MODE_AUTO)
CELL_GAUGE_MODE_DEFAULT = CELL_GAUGE_MODE_AUTO

# ``compute_cell_gauge`` statuses (explicit, never a silent fabrication).
CELL_GAUGE_STATUS_OK = "ok"
CELL_GAUGE_STATUS_EMPTY = "empty"


def resolve_cell_gauge_mode(mode) -> str:
    """Validate/resolve the per-Cell gauge mode token (``None`` -> default).

    Accepts ``"auto"`` (default; ZM-CELL-PHOTOMETRY-R34F), ``"global"`` (R11
    production path) or ``"per_cell"`` (R34b opt-in). An unknown token raises
    ``ValueError`` (never a silent fallback), so a typo cannot silently change the
    photometric strategy. The caller owns the resolution ORDER: explicit keyword
    > config key > this default.
    """
    if mode is None:
        return CELL_GAUGE_MODE_DEFAULT
    if not isinstance(mode, str):
        raise ValueError(
            f"cell_gauge_mode must be a string, got {type(mode).__name__}"
        )
    token = mode.strip().lower()
    if token not in CELL_GAUGE_MODES:
        raise ValueError(
            f"unknown cell_gauge_mode {mode!r}; expected one of "
            f"{'/'.join(CELL_GAUGE_MODES)}"
        )
    return token


def select_auto_gauge_mode(
    global_gauge: "FixedNormalization",
    global_frame_ids,
    cell_membership,
):
    """Resolve the effective gauge mode for ``"auto"`` (ZM-CELL-PHOTOMETRY-R34F).

    ``"auto"`` is a RUN-LEVEL decision made AFTER the global gauge exists: if the
    global gauge serves EVERY non-empty Cell it returns ``"global"`` (the R11
    production path, byte-identical, no offsets/feather); otherwise it returns
    ``"per_cell"`` (R34b gauges + R34c offsets + R34d feather) so the WHOLE run
    uses ONE coherent mode (never a mix of references).

    The predicate mirrors :func:`canonical_streaming.subset_fixed_normalization`
    EXACTLY — the only raise it can produce for a populated Cell. For a Cell whose
    frame list is a non-empty subset of ``global_frame_ids``, the subset raises
    iff the global reference is NOT among the Cell's frames AND no Cell frame is
    ``weight_active`` (the ``cell gauge has no active frame`` condition). This
    helper returns the SAME verdict, so ``auto`` never falsely picks ``global``
    (no crash) and never falsely picks ``per_cell`` when ``global`` would work.

    ``cell_membership`` is an iterable of ``(cell_id, cell_frame_ids)`` pairs (the
    layout membership). A cell with an EMPTY frame list is a geometrically-empty
    (zero-contributor) Cell: the executor short-circuits it to ``status="empty"``
    and NEVER calls the subset, so it is NOT a trigger by itself.

    Returns ``(effective_mode, info)``. ``info`` is ``None`` for a clean
    ``"global"`` verdict, else a DETERMINISTIC trigger record
    ``{"trigger_count", "first_cell", "reason", "cells"}`` (cells in the input
    order).
    """
    gidx = {fid: i for i, fid in enumerate(global_frame_ids)}
    ref = int(global_gauge.reference_index)
    weight_active = np.asarray(global_gauge.weight_active, dtype=bool)
    failing = []
    for cell_id, cell_frame_ids in cell_membership:
        ids = list(cell_frame_ids)
        if not ids:
            # Zero-contributor Cell: the executor returns status="empty" without
            # ever calling subset_fixed_normalization -> never a trigger here.
            continue
        try:
            idx = [gidx[fid] for fid in ids]
        except KeyError as exc:
            # subset_fixed_normalization would ALSO raise (frame not in the global
            # gauge order) -> conservative trigger (never a false "global").
            failing.append(
                (cell_id, f"frame {exc.args[0]!r} not in the global gauge frame order")
            )
            continue
        if ref in set(idx):
            # The Cell contains the global reference -> the subset never raises
            # (the identity-coefficient path), regardless of weight_active.
            continue
        if not bool(weight_active[idx].any()):
            failing.append((cell_id, "no globally weight-active frame"))
    if failing:
        info = {
            "trigger_count": len(failing),
            "first_cell": failing[0][0],
            "reason": failing[0][1],
            "cells": [cid for cid, _reason in failing],
        }
        return CELL_GAUGE_MODE_PER_CELL, info
    return CELL_GAUGE_MODE_GLOBAL, None


@dataclass(frozen=True)
class CellGaugeResult:
    """Result of ONE Cell's self-anchored (per-Cell) photometric gauge (R34b).

    Fields
    ------
    status:
        ``"ok"`` (a valid gauge was produced) or ``"empty"`` (the Cell's frames
        carry ZERO finite support; ``gauge`` is then ``None``). Explicit and
        deterministic — never a fabricated gauge.
    gauge:
        The :class:`FixedNormalization` for this Cell (``None`` when empty).
    mode:
        The mode token this result was produced under (always ``"per_cell"``).
    frame_ids:
        The Cell's frame order this gauge covers (when the provider exposes it).
    reference_index:
        The Cell-LOCAL index of the Cell's own reference frame (``None`` empty).
    total_valid_support:
        Sum of the Cell's per-frame valid support pixels.
    active_frames:
        Number of frames surviving to the weighting level in the Cell gauge.
    detail:
        Deterministic reason for an ``empty`` result (else ``None``).
    reference_frame_id:
        ZM-CELL-PHOTOMETRY-R34C reference STABILITY pin: the logical frame id of
        the Cell's own reference frame (``None`` when the provider exposes no
        frame ids, or the Cell is empty). Recording it lets a recompute be checked
        against the SAME anchor, so a Cell's level cannot silently flip.
    reference_support:
        The reference frame's valid-support count (``None`` empty) — the second
        half of the stability pin (support count + tie-break fully determine the
        reference choice).
    """

    status: str
    gauge: FixedNormalization | None
    mode: str
    frame_ids: tuple | None
    reference_index: int | None
    total_valid_support: int
    active_frames: int
    detail: str | None = None
    reference_frame_id: str | None = None
    reference_support: int | None = None

    def pin(self) -> "CellReferencePin":
        """Return the deterministic reference-stability pin (R34C)."""
        return CellReferencePin(
            frame_id=self.reference_frame_id,
            support=self.reference_support,
            reference_index=self.reference_index,
        )


@dataclass(frozen=True)
class CellReferencePin:
    """Deterministic per-Cell photometric-reference pin (ZM-CELL-PHOTOMETRY-R34C).

    The Cell's additive level (its per-Cell gauge, R34b) is anchored on one frame;
    if a recompute silently selected a different frame, the Cell's level would
    shift. This pin records the (logical frame id, valid-support count,
    cell-local index) that fully determines the choice, so recomputes can be
    compared exactly. The support count is the tie-break key; the frame id is the
    identity.
    """

    frame_id: str | None
    support: int | None
    reference_index: int | None

# Canvas-pixel margin around a frame's projected-footprint bbox. This makes the
# bbox a SUPERSET of the reprojected support (bilinear kernel + SIP margin), so
# the bbox count/offset/weight are bit-exact to the full-canvas computation.
_FOOTPRINT_MARGIN_PX = 8

# Module-level globals holding the reference frame's full-canvas alignment,
# handed to the pooled workers via an initializer that MEMAPS them from the
# run's cache dir (ZM-ZEGRID-R19). This is start-method-agnostic: under fork
# the children inherit, under spawn they re-import and the initializer memmaps
# them back, and under threads they are shared. They are read-only and never
# pickled (they can be large).
_GAUGE_R_FULL = None
_GAUGE_R_SUP = None

# Filenames for the memmapped reference planes written to the gauge cache dir.
_GAUGE_REF_FULL_NAME = "r_full.npy"
_GAUGE_REF_SUP_NAME = "r_sup.npy"

# ---------------------------------------------------------------------------
# ZM-ZEGRID-R32c — internal gauge performance levers (PROMOTED to DEFAULT)
# ---------------------------------------------------------------------------
#
# These switches were the R32a optimisations. They were promoted to the DEFAULT
# path in R32c after the R32b + R32c parity gates (96-frame real subset + M106
# output parity + tiled-reference coverage) and the gain measurements.
#
# STRICT-PATH OVERRIDE: each switch stays overridable with an explicit keyword
# argument — pass ``False`` to ``compute_global_gauge(...)`` to force the
# pre-R32a STRICT path (used by the parity tests). This is an INTERNAL escape
# hatch only; there is NO new user-visible option.
#
# * ``_GAUGE_PREFILTER_PAIRS_DEFAULT`` (Lever A, EXACT): before reprojecting a
#   pair, skip it when a pure-geometry upper bound on its common support proves
#   it would be excluded as ``insufficient_common`` anyway (never a false
#   negative). A skipped pair was excluded under the strict path, so the
#   discrete science outputs are unchanged.
# * ``_GAUGE_DEDUP_REFERENCE_DEFAULT`` (Lever B, EXACT): compute the reference
#   frame's quality metrics ONCE per pair instead of twice (the reference's
#   contribution was recomputed every pair). The extracted non-reference
#   metrics are bit-identical.
# * ``_GAUGE_BOUNDED_PAIRS_DEFAULT`` (bounded work, tolerance): reproject frame
#   ``i`` on ITS OWN footprint bbox instead of the union bbox(R, i), while the
#   frozen ``min_common`` keeps using the FULL valid counts from the counts
#   pass. Same discrete decisions; continuous coefficients within the declared
#   reprojection FP tolerance (see the module docstring / R32b/R32c reports).
_GAUGE_PREFILTER_PAIRS_DEFAULT = True
_GAUGE_DEDUP_REFERENCE_DEFAULT = True
_GAUGE_BOUNDED_PAIRS_DEFAULT = True
# * ``_GAUGE_TILED_REFERENCE_DEFAULT`` (RAM fix, tolerance): build the reference
#   full-canvas alignment TILE BY TILE straight into a disk-backed memmap
#   instead of materialising one full-canvas float32 plane in RAM. Bounded
#   transient (one tile) during reference construction; parity-gated within the
#   same reprojection FP tolerance as the bounded path.
_GAUGE_TILED_REFERENCE_DEFAULT = True

# Canvas edge of ONE reference tile (R32b tiled reference).
_GAUGE_REF_TILE_PX = 1024


def _resolve_switch(value: bool | None, default: bool) -> bool:
    """Resolve an internal lever switch (``None`` -> module default)."""
    return bool(default) if value is None else bool(value)


def _rss_kib() -> int:
    """Best-effort peak RSS of THIS process in KiB (never fatal)."""
    try:
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:  # noqa: BLE001 - instrumentation is never fatal
        try:
            import psutil

            return int(psutil.Process().memory_info().rss // 1024)
        except Exception:  # noqa: BLE001
            return 0


def _dist_stats(values) -> dict:
    """Bounded distribution summary (count/min/median/mean/max) of a list."""
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "min": round(float(np.min(arr)), 6),
        "median": round(float(np.median(arr)), 6),
        "mean": round(float(np.mean(arr)), 6),
        "max": round(float(np.max(arr)), 6),
    }


def _bbox_area(bbox) -> int:
    """Pixel area of a canvas bbox ``(y0, y1, x0, x1)``."""
    y0, y1, x0, x1 = bbox
    return max(0, int(y1) - int(y0)) * max(0, int(x1) - int(x0))


def _bbox_intersection_area(b1, b2) -> int:
    """Pixel area of the intersection of two canvas bboxes (0 when disjoint).

    Since both frames' reprojected supports are CONTAINED in their footprint
    bboxes, this is a provable UPPER BOUND on their common support: it is used
    by Lever A and never under-counts the true common pixels.
    """
    y0 = max(int(b1[0]), int(b2[0]))
    y1 = min(int(b1[1]), int(b2[1]))
    x0 = max(int(b1[2]), int(b2[2]))
    x1 = min(int(b1[3]), int(b2[3]))
    if y1 <= y0 or x1 <= x0:
        return 0
    return (y1 - y0) * (x1 - x0)


def _prefilter_detail(upper_bound: int, min_common: int) -> str:
    """Deterministic diagnostic detail for a Lever-A prefiltered pair.

    When the geometric bound is exactly 0 the true common count is 0, so this
    reproduces the strict-path detail byte-for-byte. Otherwise it records the
    CONSERVATIVE bound (``<=``), which is an upper bound on the true
    ``common=<n>`` the strict path would print; the exclusion reason is
    unchanged.
    """
    if upper_bound <= 0:
        return f"common=0 < min_common={min_common}"
    return f"common<={upper_bound} < min_common={min_common}"


def _ref_dir_for(cache_dir) -> str:
    """Resolve (and create) the gauge reference-plane directory."""
    if cache_dir is not None:
        ref_dir = Path(cache_dir) / "__gauge_ref__"
    else:
        ref_dir = Path(tempfile.mkdtemp(prefix="zegrid_gauge_ref_"))
    ref_dir.mkdir(parents=True, exist_ok=True)
    return str(ref_dir)


def _write_gauge_reference(cache_dir, r_full, r_sup) -> str:
    """Write the reference planes (float32 + bool) as a small ``.npy`` pair.

    Returns the directory path holding ``r_full.npy`` / ``r_sup.npy``. Uses
    ``cache_dir`` when given (the run's gauge cache dir) else a fresh temp dir.
    """
    ref_dir = _ref_dir_for(cache_dir)
    np.save(os.path.join(ref_dir, _GAUGE_REF_FULL_NAME), np.asarray(r_full), allow_pickle=False)
    np.save(os.path.join(ref_dir, _GAUGE_REF_SUP_NAME), np.asarray(r_sup), allow_pickle=False)
    return ref_dir


def _reproject_reference_tiled(decode_fn, f, canvas, ref_dir, tile_px: int | None = None) -> str:
    """ZM-ZEGRID-R32b: build the reference alignment TILE BY TILE into a memmap.

    Instead of materialising ONE full-canvas float32 plane (+ support) in RAM,
    the reference frame is reprojected onto canvas TILES and written straight
    into two disk-backed ``.npy`` memmaps (``r_full.npy`` float32 HxWx3,
    ``r_sup.npy`` bool HxW). Peak transient RAM is O(one tile), independent of
    the canvas area. Returns the reference directory.

    Parity: the reference pixels are the SAME support (the support is
    pixel-wise deterministic); the VALUES are sampled on per-tile sub-WCS
    (CRPIX shifted), i.e. the SAME ~1e-4 reprojection FP class as the bounded
    pair path — gated by the run's declared tolerance, never claimed exact.
    """
    ref_dir = str(ref_dir)
    Path(ref_dir).mkdir(parents=True, exist_ok=True)
    tile = int(tile_px or _GAUGE_REF_TILE_PX)
    tile = max(1, tile)
    h, w = int(canvas.height), int(canvas.width)
    full_path = os.path.join(ref_dir, _GAUGE_REF_FULL_NAME)
    sup_path = os.path.join(ref_dir, _GAUGE_REF_SUP_NAME)
    r_full = np.lib.format.open_memmap(
        full_path, mode="w+", dtype=np.float32, shape=(h, w, 3)
    )
    r_sup = np.lib.format.open_memmap(
        sup_path, mode="w+", dtype=np.bool_, shape=(h, w)
    )
    chw = _decode_to_chw(decode_fn, f)
    # ZM-ZEGRID-R32c hardening: track tile coverage so a short/failed tile can
    # never be silently skipped. The tile grid is exhaustive by construction; we
    # assert every tile rectangle was written and every written tile had the
    # expected shape, and then assert the union covers the whole canvas exactly.
    tiles_y = (h + tile - 1) // tile
    tiles_x = (w + tile - 1) // tile
    covered = np.zeros((tiles_y, tiles_x), dtype=bool)
    for ty, y0 in enumerate(range(0, h, tile)):
        y1 = min(h, y0 + tile)
        for tx, x0 in enumerate(range(0, w, tile)):
            x1 = min(w, x0 + tile)
            hdr = _bbox_wcs_header(canvas, (y0, y1, x0, x1))
            rgb, geom = zxe.reproject_cropped(
                chw, f.wcs(), WCS(hdr), (y1 - y0, x1 - x0)
            )
            if rgb.shape != (y1 - y0, x1 - x0, 3):
                raise RuntimeError(
                    f"tiled reference: tile ({y0}:{y1},{x0}:{x1}) returned rgb "
                    f"shape {rgb.shape}, expected {(y1 - y0, x1 - x0, 3)}"
                )
            if geom.shape != (y1 - y0, x1 - x0):
                raise RuntimeError(
                    f"tiled reference: tile ({y0}:{y1},{x0}:{x1}) returned support "
                    f"shape {geom.shape}, expected {(y1 - y0, x1 - x0)}"
                )
            r_full[y0:y1, x0:x1, :] = rgb
            r_sup[y0:y1, x0:x1] = geom
            covered[ty, tx] = True
    if not bool(covered.all()):
        missing = int(covered.size - covered.sum())
        raise RuntimeError(
            f"tiled reference: {missing} of {covered.size} tiles were not written "
            f"(canvas {h}x{w}, tile {tile})"
        )
    # Exact coverage: the sum of tile rectangle areas must equal h*w (no gap, no
    # overlap given the range() grid; asserted for completeness).
    if tiles_y * tiles_x * tile * tile != (tiles_y * tile) * (tiles_x * tile):
        raise RuntimeError("tiled reference: inconsistent tile grid")
    r_full.flush()
    r_sup.flush()
    del r_full, r_sup
    return ref_dir


def _init_gauge_reference(ref_dir: str) -> None:
    """Child/parent initializer: memmap the reference planes into the globals.

    Read-only mmap keeps peak memory bounded and makes the pooled workers
    start-method-agnostic (fork AND spawn AND threads).
    """
    global _GAUGE_R_FULL, _GAUGE_R_SUP
    _GAUGE_R_FULL = np.load(os.path.join(ref_dir, _GAUGE_REF_FULL_NAME), mmap_mode="r")
    _GAUGE_R_SUP = np.load(os.path.join(ref_dir, _GAUGE_REF_SUP_NAME), mmap_mode="r")


def _clear_gauge_reference() -> None:
    """Release the memmapped reference globals (best-effort)."""
    global _GAUGE_R_FULL, _GAUGE_R_SUP
    _GAUGE_R_FULL = None
    _GAUGE_R_SUP = None


def _cleanup_gauge_reference(ref_dir: str | None) -> None:
    """Best-effort removal of the temp reference files; never fatal."""
    if not ref_dir:
        return
    try:
        shutil.rmtree(ref_dir, ignore_errors=True)
    except Exception:
        pass


def _footprint_bbox(f, canvas) -> tuple:
    """Canvas bbox ``(y0, y1, x0, x1)`` bounding a frame's footprint + margin."""
    poly = zg._source_polygon(f.shape_hw, f.wcs(), canvas.wcs())
    minx, miny, maxx, maxy = poly.bounds
    m = _FOOTPRINT_MARGIN_PX
    x0 = max(0, int(np.floor(minx)) - m)
    y0 = max(0, int(np.floor(miny)) - m)
    x1 = min(canvas.width, int(np.ceil(maxx)) + m + 1)
    y1 = min(canvas.height, int(np.ceil(maxy)) + m + 1)
    return (y0, y1, x0, x1)


def _bbox_wcs_header(canvas, bbox) -> str:
    """Sub-canvas WCS for a bbox (CRPIX shifted by the bbox origin)."""
    y0, _y1, x0, _x1 = bbox
    w = copy.deepcopy(canvas.wcs())
    w.wcs.crpix -= np.array([x0, y0])
    w.array_shape = (bbox[1] - bbox[0], bbox[3] - bbox[2])
    return w.to_header(relax=True).tostring()


def _union_bbox(b1, b2) -> tuple:
    """Union of two canvas bboxes ``(y0, y1, x0, x1)``."""
    return (
        min(b1[0], b2[0]),
        max(b1[1], b2[1]),
        min(b1[2], b2[2]),
        max(b1[3], b2[3]),
    )


def _decode_to_chw(decode_fn, f) -> np.ndarray:
    hwc = np.asarray(decode_fn(f), dtype=np.float32)
    if hwc.ndim == 2:
        hwc = np.stack([hwc, hwc, hwc], axis=-1)
    return np.ascontiguousarray(np.moveaxis(hwc, -1, 0))


def _reproject_full(decode_fn, f, canvas):
    """Reproject one frame onto the FULL canvas (one plane)."""
    chw = _decode_to_chw(decode_fn, f)
    rgb, geom = zxe.reproject_cropped(
        chw, f.wcs(), canvas.wcs(), (canvas.height, canvas.width)
    )
    return rgb, geom


def _counts_worker(task):
    zxe.reset_reproject_path_stats()
    (decode_fn, f, bbox_wcs_header, bbox_shape) = task
    t0 = time.perf_counter()
    chw = _decode_to_chw(decode_fn, f)
    t1 = time.perf_counter()
    rgb, geom = zxe.reproject_cropped(chw, f.wcs(), WCS(bbox_wcs_header), tuple(bbox_shape))
    t2 = time.perf_counter()
    b = prepare_canonical_inputs([rgb], [geom])
    t3 = time.perf_counter()
    area = int(bbox_shape[0]) * int(bbox_shape[1])
    return {
        "count": int(b.frame_valid_counts[0]),
        "decode_s": t1 - t0,
        "reproject_s": t2 - t1,
        "prepare_s": t3 - t2,
        "bbox_area": area,
        "bytes_reprojected": int(area) * 3 * 4,
        "rss_kib": _rss_kib(),
        "reproject": zxe.get_reproject_path_stats().to_dict(),
    }


def _pair_worker(task):
    zxe.reset_reproject_path_stats()
    (
        index,
        decode_fn,
        f,
        bbox,
        bbox_wcs_header,
        bbox_shape,
        norm_token,
        weight_token,
        dedup_reference,
        bounded_counts,
    ) = task
    t0 = time.perf_counter()
    chw = _decode_to_chw(decode_fn, f)
    t1 = time.perf_counter()
    rgb, geom = zxe.reproject_cropped(chw, f.wcs(), WCS(bbox_wcs_header), tuple(bbox_shape))
    t2 = time.perf_counter()
    y0, y1, x0, x1 = bbox
    r_crop = np.ascontiguousarray(_GAUGE_R_FULL[y0:y1, x0:x1])
    r_sup = _GAUGE_R_SUP[y0:y1, x0:x1]

    batch2 = prepare_canonical_inputs([r_crop, rgb], [r_sup, geom])
    if bounded_counts is not None:
        # ZM-ZEGRID-R32a bounded path: the crop is frame i's OWN bbox (a subset
        # of the union), but the FROZEN min_common gate must keep the KNOWN full
        # valid counts from the counts pass (the union trick did this implicitly).
        batch2 = replace(
            batch2, frame_valid_counts=np.asarray(bounded_counts, dtype=np.int64)
        )
    t3 = time.perf_counter()
    norm2 = normalize_canonical_images(batch2, norm_token, reference_index=0)
    t4 = time.perf_counter()

    norm_exc = []
    for e in norm2.exclusions:
        if e.index == 1:
            norm_exc.append({"stage": e.stage, "reason": e.reason, "detail": e.detail})

    raw_weight = float("nan")
    noise_sigma = float("nan")
    fwhm = float("nan")
    weight_active = bool(norm2.active_frames[1])
    weight_exc = []
    if weight_token != "none" and bool(norm2.active_frames[1]):
        # ZM-ZEGRID-R32a Lever B (EXACT): the reference (index 0) quality metrics
        # are recomputed every pair although they are IDENTICAL for a given run
        # (R's valid pixels within any union/bounded crop are exactly its full
        # support). Mark the reference inactive so the frozen weighting branch
        # SKIPS its metrics; the extracted index-1 numbers are unchanged. If the
        # non-reference frame fails its own metric (which would leave the batch
        # with no active frame), retry on the untouched 2-frame batch so the
        # exact exclusion reason is preserved (rare; R stays active there).
        if dedup_reference:
            masked = np.array(norm2.active_frames, dtype=bool, copy=True)
            masked[0] = False
            try:
                weight2 = compute_canonical_quality_weights(
                    replace(norm2, active_frames=masked), weight_token
                )
            except CanonicalStackFailure:
                weight2 = compute_canonical_quality_weights(norm2, weight_token)
        else:
            weight2 = compute_canonical_quality_weights(norm2, weight_token)
        t5 = time.perf_counter()
        raw_weight = float(weight2.raw_weights[1])
        noise_sigma = float(weight2.noise_sigma[1])
        fwhm = float(weight2.fwhm[1])
        weight_active = bool(weight2.active_frames[1])
        if not weight_active:
            for e in weight2.exclusions:
                if e.index == 1 and e.stage == "weighting":
                    weight_exc.append({"stage": e.stage, "reason": e.reason, "detail": e.detail})
    else:
        t5 = t4

    area = int(bbox_shape[0]) * int(bbox_shape[1])
    return {
        "index": index,
        "coeff": np.array(norm2.coefficients[1], dtype=np.float64, copy=True),
        "norm_active": bool(norm2.active_frames[1]),
        "norm_exc": norm_exc,
        "raw_weight": raw_weight,
        "noise_sigma": noise_sigma,
        "fwhm": fwhm,
        "weight_active": weight_active,
        "weight_exc": weight_exc,
        "decode_s": t1 - t0,
        "reproject_s": t2 - t1,
        "prepare_s": t3 - t2,
        "normalize_s": t4 - t3,
        "weight_s": t5 - t4,
        "bbox_area": area,
        "bytes_reprojected": int(area) * 3 * 4,
        "rss_kib": _rss_kib(),
        "reproject": zxe.get_reproject_path_stats().to_dict(),
    }


def compute_global_gauge(
    frames: Sequence[zg.FrameDescriptor],
    canvas: zg.GlobalCanvas,
    decode_fn: Callable[[zg.FrameDescriptor], np.ndarray],
    config,
    cache_dir=None,
    *,
    reuse_cache: bool = True,
    workers: int = 1,
    progress_callback=None,
    emit=None,
    diagnostics=None,
    prefilter_pairs: bool | None = None,
    dedup_reference: bool | None = None,
    bounded_pairs: bool | None = None,
    tiled_reference: bool | None = None,
) -> tuple[FixedNormalization, tuple[str, ...]]:
    """Compute the global photometric gauge over the full canvas (bounded).

    Returns ``(gauge, frame_ids)`` where ``frame_ids`` is the frame order the
    gauge covers (sorted by FrameId). ``decode_fn`` decodes one frame descriptor
    to an HWC float32 array (the product decoder is passed in so this module stays
    decoupled from the product layer). ``config`` is the frozen
    :class:`~zemosaic.core.zegrid.science_adapter.MiniTileScienceConfig`.

    ``cache_dir`` / ``reuse_cache`` are accepted for backward compatibility but
    are IGNORED: the R12 rework-1 gauge is a bounded streaming phase-1 with NO
    disk cache (peak disk ~0, peak RAM O(1-2 planes)).

    The result is DETERMINISTIC and matches the R11 full-canvas gauge exactly
    when bboxes ≈ canvas; on partial-overlap corpora it is equivalent up to
    ~1e-4 relative reprojection FP noise (see the module docstring).
    """
    ordered = sorted(frames, key=lambda f: f.frame_id)
    frame_ids = [f.frame_id.logical_path for f in ordered]
    n = len(ordered)
    c = 3

    # ZM-ZEGRID-R32a internal levers (NON-DEFAULT; see module constants).
    _prefilter = _resolve_switch(prefilter_pairs, _GAUGE_PREFILTER_PAIRS_DEFAULT)
    _dedup_ref = _resolve_switch(dedup_reference, _GAUGE_DEDUP_REFERENCE_DEFAULT)
    _bounded = _resolve_switch(bounded_pairs, _GAUGE_BOUNDED_PAIRS_DEFAULT)
    _tiled_ref = _resolve_switch(tiled_reference, _GAUGE_TILED_REFERENCE_DEFAULT)
    _rss_start_kib = _rss_kib()
    _t_gauge0 = time.perf_counter()

    # ZM-ZEGRID-R14 rework-1 (M1): live gauge progress is CUMULATIVE across the
    # two sub-passes so the phase-level ETA is honest throughout. Total work =
    # N (per-frame footprint counts) + (N-1) (per-frame normalization; reference
    # excluded) = 2N-1. The counts pass reports done=1..N; the pairs pass
    # CONTINUES at done=N+1..2N-1. ``item_id`` is prefixed ``counts:`` / ``pairs:``
    # so the sub-pass is unambiguous. Reported through
    # ``progress_callback(done, total, item_id)`` (best-effort, never fatal).
    gauge_total = 2 * n - 1
    count_progress = None
    if progress_callback is not None:
        def _count_progress(done: int, _total: int) -> None:
            try:
                item = frame_ids[done - 1] if 0 < done <= n else None
                progress_callback(int(done), int(gauge_total), f"counts:{item}")
            except Exception:
                pass

        count_progress = _count_progress

    norm_token = config.normalization
    weight_token = config.weighting

    # --- per-frame valid counts over each frame's own footprint bbox (bounded) ---
    bboxes = [_footprint_bbox(f, canvas) for f in ordered]
    count_tasks = [
        (
            decode_fn,
            f,
            _bbox_wcs_header(canvas, bboxes[i]),
            (bboxes[i][1] - bboxes[i][0], bboxes[i][3] - bboxes[i][2]),
        )
        for i, f in enumerate(ordered)
    ]
    counts_meta: dict = {}
    if emit is not None:
        try:
            emit(f"[ZEGRID] gauge phase=counts units={n} workers={workers}", "INFO")
        except Exception:
            pass
    count_results = zpar.pmap(
        _counts_worker, count_tasks, workers,
        progress_callback=count_progress, emit=emit, meta=counts_meta,
    )
    counts = np.asarray([r["count"] for r in count_results], dtype=np.int64)
    _rss_counts_end = _rss_kib()


    # --- reference selection (reuse the frozen function; exact argmax of counts) ---
    probe = CanonicalInputBatch(
        images=np.empty((0,), dtype=np.float32),
        valid_mask=np.empty((0,), dtype=bool),
        original_mono=False,
        frame_valid_counts=counts,
        n_frames=n,
        height=canvas.height,
        width=canvas.width,
        channels=c,
        original_ndim=3,
        original_shape=(canvas.height, canvas.width, c),
    )
    ref_idx = select_canonical_reference(probe, None)
    ref_bbox = bboxes[ref_idx]

    # --- reference full-canvas alignment (ONE plane) + reference weighting ---
    _t_ref0 = time.perf_counter()
    zxe.reset_reproject_path_stats()
    _ref_dir_prebuilt = None
    if _tiled_ref:
        # ZM-ZEGRID-R32b: build the reference alignment TILE BY TILE straight
        # into a disk-backed memmap (bounded transient, independent of canvas).
        _ref_dir_prebuilt = _ref_dir_for(cache_dir)
        _reproject_reference_tiled(
            decode_fn, ordered[ref_idx], canvas, _ref_dir_prebuilt
        )
        r_full = np.load(
            os.path.join(_ref_dir_prebuilt, _GAUGE_REF_FULL_NAME), mmap_mode="r"
        )
        r_sup = np.load(
            os.path.join(_ref_dir_prebuilt, _GAUGE_REF_SUP_NAME), mmap_mode="r"
        )
    else:
        r_full, r_sup = _reproject_full(decode_fn, ordered[ref_idx], canvas)
    ref_reproj = zxe.get_reproject_path_stats().to_dict()
    _t_ref_reproj = time.perf_counter()
    ref_norm = normalize_canonical_images(
        prepare_canonical_inputs([r_full], [r_sup]), norm_token, reference_index=0
    )
    _t_ref_norm = time.perf_counter()
    ref_weight = compute_canonical_quality_weights(ref_norm, weight_token)
    _t_ref_weight = time.perf_counter()
    ref_timing = {
        "reproject_s": _t_ref_reproj - _t_ref0,
        "normalize_s": _t_ref_norm - _t_ref_reproj,
        "weight_s": _t_ref_weight - _t_ref_norm,
        "total_s": _t_ref_weight - _t_ref0,
        "area_px": int(canvas.height) * int(canvas.width),
        "bytes_reprojected": int(canvas.height) * int(canvas.width) * 3 * 4,
        "tiled": bool(_tiled_ref),
        "tile_px": int(_GAUGE_REF_TILE_PX) if _tiled_ref else None,
    }
    # Release the parent's reference-plane references before the pairs pass so
    # the parent does not pin a full-canvas copy during the parallel phase
    # (ZM-ZEGRID-R32b, best-effort; the quality metrics above are already kept
    # as scalars in ref_weight). Done AFTER the reference planes have been
    # written for the non-tiled path (see below).

    coefficients = np.full((n, c, 2), np.nan, dtype=np.float64)
    norm_active = np.zeros(n, dtype=bool)
    raw_weights = np.full(n, np.nan, dtype=np.float64)
    noise_sigma = np.full(n, np.nan, dtype=np.float64)
    fwhm = np.full(n, np.nan, dtype=np.float64)
    norm_exclusions = []
    weight_exclusions = []

    coefficients[ref_idx, :, :] = (1.0, 0.0)
    norm_active[ref_idx] = bool(counts[ref_idx] > 0)
    if weight_token != "none":
        raw_weights[ref_idx] = float(ref_weight.raw_weights[0])
        noise_sigma[ref_idx] = float(ref_weight.noise_sigma[0])
        fwhm[ref_idx] = float(ref_weight.fwhm[0])

    # --- per-frame normalization + weighting (bounded, parallel) ---
    # ZM-ZEGRID-R19: instead of the fork-only copy-on-write globals, write the
    # reference planes to the gauge cache dir and hand them to the pooled workers
    # via a memmap initializer (works under fork, spawn AND threads).
    if _ref_dir_prebuilt is not None:
        ref_dir = _ref_dir_prebuilt  # ZM-ZEGRID-R32b: already written tile by tile
    else:
        ref_dir = _write_gauge_reference(cache_dir, r_full, r_sup)
    # Drop the parent's in-RAM reference planes now that they are on disk (and
    # the reference quality scalars are already extracted). Bounded parent RAM
    # during the pairs pass; harmless no-op for the memmap-tiled case.
    del r_full, r_sup, ref_norm
    _rss_ref_end = _rss_kib()
    pairs_meta: dict = {}
    prefiltered_pairs: list = []
    try:
        pair_tasks = []
        pair_ids = []
        for i in range(n):
            if i == ref_idx:
                continue
            # ZM-ZEGRID-R32a Lever A (EXACT geometric prefilter): the union bbox
            # of the two footprints is a SUPERSET of each frame's reprojected
            # support, so its intersection area is an UPPER BOUND on the true
            # common pixel count. When that bound is below the frozen
            # ``min_common`` the pair is PROVABLY excluded as insufficient_common
            # by the strict path, so it can be skipped BEFORE any reprojection
            # (never a false negative; the discrete outputs are unchanged for
            # every pair the strict path keeps).
            min_common_i = _compute_min_common(int(counts[ref_idx]), int(counts[i]))
            upper = _bbox_intersection_area(ref_bbox, bboxes[i])
            if _prefilter and upper < min_common_i:
                prefiltered_pairs.append((i, int(upper), int(min_common_i)))
                continue
            if _bounded:
                # ZM-ZEGRID-R32a bounded path: frame i is reprojected on ITS OWN
                # footprint bbox (a subset of the union) and the frozen gate is
                # fed the KNOWN full counts, so min_common is unchanged.
                box = bboxes[i]
                bounded_counts = (int(counts[ref_idx]), int(counts[i]))
            else:
                box = _union_bbox(ref_bbox, bboxes[i])
                bounded_counts = None
            pair_tasks.append(
                (
                    i,
                    decode_fn,
                    ordered[i],
                    box,
                    _bbox_wcs_header(canvas, box),
                    (box[1] - box[0], box[3] - box[2]),
                    norm_token,
                    weight_token,
                    _dedup_ref,
                    bounded_counts,
                )
            )
            pair_ids.append(ordered[i].frame_id.logical_path)

        pair_progress = None
        if progress_callback is not None:
            m = len(pair_tasks)
            _pre_done = n + len(prefiltered_pairs)
            # Report the prefiltered pairs as already done so the cumulative
            # phase progress stays monotone and still reaches 2N-1.
            for _j in range(len(prefiltered_pairs)):
                try:
                    progress_callback(
                        int(n + _j + 1), int(gauge_total), f"pairs:{ordered[prefiltered_pairs[_j][0]].frame_id.logical_path}"
                    )
                except Exception:
                    pass

            def _pair_progress(done: int, _total: int) -> None:
                try:
                    item = pair_ids[done - 1] if 0 < done <= m else None
                    # rework-1 (M1): cumulative — the pairs pass continues at
                    # done = N+1..2N-1 so the phase-level ETA is monotone.
                    progress_callback(int(_pre_done + done), int(gauge_total), f"pairs:{item}")
                except Exception:
                    pass

            pair_progress = _pair_progress

        if emit is not None:
            try:
                emit(
                    f"[ZEGRID] gauge phase=pairs units={len(pair_tasks)} workers={workers}"
                    + (f" prefiltered={len(prefiltered_pairs)}" if prefiltered_pairs else "")
                    + (" bounded=true" if _bounded else ""),
                    "INFO",
                )
            except Exception:
                pass
        pair_results = zpar.pmap(
            _pair_worker, pair_tasks, workers, progress_callback=pair_progress,
            initializer=_init_gauge_reference, initargs=(ref_dir,),
            emit=emit, meta=pairs_meta,
        )
    finally:
        _clear_gauge_reference()
        _cleanup_gauge_reference(ref_dir)

    for res in pair_results:
        i = int(res["index"])
        coefficients[i] = res["coeff"]
        norm_active[i] = bool(res["norm_active"])
        for e in res["norm_exc"]:
            norm_exclusions.append(
                FrameExclusion(index=i, stage=e["stage"], reason=e["reason"], detail=e["detail"])
            )
        if weight_token != "none" and norm_active[i]:
            raw_weights[i] = res["raw_weight"]
            noise_sigma[i] = res["noise_sigma"]
            fwhm[i] = res["fwhm"]
            if not res["weight_active"]:
                for e in res["weight_exc"]:
                    weight_exclusions.append(
                        FrameExclusion(
                            index=i, stage=e["stage"], reason=e["reason"], detail=e["detail"]
                        )
                    )

    # ZM-ZEGRID-R32a Lever A: materialise the prefiltered pairs as the EXACT
    # exclusion the strict path would have produced (insufficient_common), with
    # no reprojection. Kept in ascending frame order so the exclusion tuple is
    # identical to the strict path's.
    for (i, upper, min_common_i) in prefiltered_pairs:
        coefficients[i] = np.nan
        norm_active[i] = False
        norm_exclusions.append(
            FrameExclusion(
                index=i,
                stage=STAGE_NORMALIZATION,
                reason=REASON_INSUFFICIENT_COMMON,
                detail=_prefilter_detail(upper, min_common_i),
            )
        )
    norm_exclusions.sort(key=lambda e: e.index)
    weight_exclusions.sort(key=lambda e: e.index)
    _t_pairs_end = time.perf_counter()
    _rss_pairs_end = _rss_kib()

    # --- weighting-level active + weight normalization (reuse _phase1 semantics) ---
    if weight_token == "none":
        weights = np.ones(n, dtype=np.float64)
        weights[~norm_active] = 0.0
        raw_weights[norm_active] = 1.0
        weight_active = norm_active.copy()
    else:
        weight_active = norm_active.copy()
        for e in weight_exclusions:
            weight_active[e.index] = False
        if not weight_active.any():
            raise CanonicalStackFailure("no frame remains after quality weighting")
        max_raw = float(np.max(raw_weights[weight_active]))
        if not (np.isfinite(max_raw) and max_raw > 0.0):
            raise CanonicalStackFailure("no positive quality weight remains after weighting")
        weights = np.zeros(n, dtype=np.float64)
        weights[weight_active] = raw_weights[weight_active] / max_raw

    gauge = FixedNormalization(
        reference_index=int(ref_idx),
        coefficients=np.ascontiguousarray(coefficients),
        norm_active=np.ascontiguousarray(norm_active),
        weights=np.ascontiguousarray(weights),
        weight_active=np.ascontiguousarray(weight_active),
        exclusions=tuple(norm_exclusions) + tuple(weight_exclusions),
    )

    if diagnostics is not None:
        diagnostics["workers"] = int(workers)
        diagnostics["parent_daemon"] = bool(zpar._parent_is_daemonic())
        diagnostics["executor"] = pairs_meta.get("executor")
        diagnostics["fallback"] = bool(
            counts_meta.get("fallback") or pairs_meta.get("fallback")
        )
        diagnostics["phases"] = {
            "counts": {
                "executor": counts_meta.get("executor"),
                "units": int(n),
                "seconds": counts_meta.get("seconds"),
                "seconds_per_unit": counts_meta.get("seconds_per_unit"),
            },
            "pairs": {
                "executor": pairs_meta.get("executor"),
                "units": int(n - 1),
                "seconds": pairs_meta.get("seconds"),
                "seconds_per_unit": pairs_meta.get("seconds_per_unit"),
                "fallback": bool(pairs_meta.get("fallback")),
                "fallback_reason": pairs_meta.get("fallback_reason"),
            },
        }
        # ZM-ZEGRID-R21: which reprojection path (fast vs fallback) the gauge
        # actually took, aggregated over the reference plane + the counts/pairs
        # workers (each worker reports its own process-local stats).
        diagnostics["reprojection"] = zxe.merge_reproject_stats(
            ref_reproj,
            *(r.get("reproject") for r in count_results),
            *(r.get("reproject") for r in pair_results),
        )
        # ZM-ZEGRID-R32a: bounded per-subsystem instrumentation (no per-pixel
        # logging). Worker-reported timings/areas/bytes + parent peak RSS.
        diagnostics["instrumentation"] = _gauge_instrumentation(
            n=n,
            ref_idx=ref_idx,
            ref_timing=ref_timing,
            count_results=count_results,
            pair_results=pair_results,
            prefiltered_pairs=prefiltered_pairs,
            counts_meta=counts_meta,
            pairs_meta=pairs_meta,
            rss_start_kib=_rss_start_kib,
            rss_end_kib=_rss_kib(),
            rss_counts_end_kib=_rss_counts_end,
            rss_ref_end_kib=_rss_ref_end,
            rss_pairs_end_kib=_rss_pairs_end,
            wall_s=_t_pairs_end - _t_gauge0,
            prefilter=_prefilter,
            dedup_reference=_dedup_ref,
            bounded=_bounded,
            tiled_reference=_tiled_ref,
            chunk_px=int(canvas.height) * int(canvas.width),
        )

    return gauge, tuple(frame_ids)


def _gauge_instrumentation(
    *,
    n: int,
    ref_idx: int,
    ref_timing: dict,
    count_results: list,
    pair_results: list,
    prefiltered_pairs: list,
    counts_meta: dict,
    pairs_meta: dict,
    rss_start_kib: int,
    rss_end_kib: int,
    rss_counts_end_kib: int = 0,
    rss_ref_end_kib: int = 0,
    rss_pairs_end_kib: int = 0,
    wall_s: float,
    prefilter: bool,
    dedup_reference: bool,
    bounded: bool,
    tiled_reference: bool = False,
    chunk_px: int,
) -> dict:
    """Bounded per-subsystem gauge instrumentation (ZM-ZEGRID-R32a).

    Pure aggregation of the worker-reported scalars and the phase metadata. It
    never touches pixels and never alters science; it only describes what was
    done (timings, areas, bytes, RSS, and the internal lever switches).
    """

    def _col(results, key):
        return [r.get(key) for r in results if r.get(key) is not None]

    counts_reproject = _col(count_results, "reproject_s")
    counts_area = _col(count_results, "bbox_area")
    counts_bytes = sum(int(r.get("bytes_reprojected", 0)) for r in count_results)

    pairs_reproject = _col(pair_results, "reproject_s")
    pairs_normalize = _col(pair_results, "normalize_s")
    pairs_weight = _col(pair_results, "weight_s")
    pairs_prepare = _col(pair_results, "prepare_s")
    pairs_area = _col(pair_results, "bbox_area")
    pairs_bytes = sum(int(r.get("bytes_reprojected", 0)) for r in pair_results)

    worker_rss = [
        int(r.get("rss_kib", 0))
        for r in list(count_results) + list(pair_results)
        if int(r.get("rss_kib", 0)) > 0
    ]

    # Cost per reprojected pixel (the O(area) claim): pairs_total_reproject_s /
    # pairs_total_px. Reported so the R32b scaling gate can compare bbox areas.
    pairs_px_total = int(sum(pairs_area)) if pairs_area else 0
    pairs_reproject_sum = float(sum(pairs_reproject)) if pairs_reproject else 0.0
    per_px = (
        pairs_reproject_sum / pairs_px_total if pairs_px_total > 0 else None
    )

    return {
        "schema": "zegrid-gauge-instrumentation-r32a",
        "levers": {
            "prefilter_pairs": bool(prefilter),
            "dedup_reference": bool(dedup_reference),
            "bounded_pairs": bool(bounded),
            "tiled_reference": bool(tiled_reference),
        },
        "counts": {
            "frames": int(n),
            "executor": counts_meta.get("executor"),
            "phase_seconds": counts_meta.get("seconds"),
            "seconds_per_unit": counts_meta.get("seconds_per_unit"),
            "reproject_s": _dist_stats(counts_reproject),
            "bbox_area_px": _dist_stats(counts_area),
            "bytes_reprojected": int(counts_bytes),
            "rss_kib_max": max(worker_rss) if worker_rss else None,
        },
        "pairs": {
            "pairs": int(n - 1),
            "pairs_prefiltered": int(len(prefiltered_pairs)),
            "executor": pairs_meta.get("executor"),
            "phase_seconds": pairs_meta.get("seconds"),
            "seconds_per_unit": pairs_meta.get("seconds_per_unit"),
            "reproject_s": _dist_stats(pairs_reproject),
            "prepare_s": _dist_stats(pairs_prepare),
            "normalize_s": _dist_stats(pairs_normalize),
            "weight_s": _dist_stats(pairs_weight),
            "crop_area_px": _dist_stats(pairs_area),
            "bytes_reprojected": int(pairs_bytes),
            "reproject_s_per_crop_px": (
                round(float(per_px), 12) if per_px is not None else None
            ),
            "rss_kib_max": max(worker_rss) if worker_rss else None,
        },
        "reference": {
            "index": int(ref_idx),
            "reproject_s": round(float(ref_timing["reproject_s"]), 6),
            "normalize_s": round(float(ref_timing["normalize_s"]), 6),
            "weight_s": round(float(ref_timing["weight_s"]), 6),
            "total_s": round(float(ref_timing["total_s"]), 6),
            "area_px": int(ref_timing["area_px"]),
            "bytes_reprojected": int(ref_timing["bytes_reprojected"]),
            "tiled": bool(ref_timing.get("tiled", False)),
            "tile_px": ref_timing.get("tile_px"),
        },
        "totals": {
            "wall_seconds": round(float(wall_s), 6),
            "counts_seconds": counts_meta.get("seconds"),
            "pairs_seconds": pairs_meta.get("seconds"),
            "reproject_seconds_sum": round(
                float(sum(counts_reproject) + sum(pairs_reproject)), 6
            ),
            "normalize_seconds_sum": round(float(sum(pairs_normalize)), 6),
            "weight_seconds_sum": round(float(sum(pairs_weight)), 6),
            "reference_seconds": round(float(ref_timing["total_s"]), 6),
            "bytes_lifted": int(counts_bytes + pairs_bytes + ref_timing["bytes_reprojected"]),
        },
        "memory": {
            "peak_rss_kib_start": int(rss_start_kib),
            "peak_rss_kib_end": int(rss_end_kib),
            "worker_rss_kib_max": max(worker_rss) if worker_rss else None,
            "stage_peak_rss_kib": {
                "after_counts": int(rss_counts_end_kib),
                "after_reference": int(rss_ref_end_kib),
                "after_pairs": int(rss_pairs_end_kib),
            },
            "canvas_px": int(chunk_px),
            "reference_plane_canvas_px": int(chunk_px),
            "note": (
                "R32a keeps the reference frame as ONE full-canvas float32 plane "
                "(+ bool support) in RAM (see R32b for the tiled/bounded "
                "reference). peak_rss_kib_* are SINGLE-process ru_maxrss, not the "
                "sum of worker peaks."
            ),
        },
    }


# ---------------------------------------------------------------------------
# ZM-CELL-PHOTOMETRY-R34B — per-Cell self-anchored gauge
# ---------------------------------------------------------------------------

def _provider_valid_counts(provider) -> np.ndarray:
    """Per-frame valid support counts over the provider's OWN domain.

    Reuses the FROZEN ``prepare_canonical_inputs`` (one 1-frame batch per frame, the
    exact count basis ``canonical_streaming._phase1`` uses), so the per-Cell
    reference selection is computed on the same footing as the global gauge.
    """
    n = provider.n_frames
    counts = np.zeros(n, dtype=np.int64)
    for i in range(n):
        raw, sup = provider.get_raw_frame(i)
        counts[i] = int(prepare_canonical_inputs([raw], [sup]).frame_valid_counts[0])
    return counts


def compute_cell_gauge(provider, request) -> "CellGaugeResult":
    """Compute ONE Cell's SELF-ANCHORED photometric gauge (R34B, opt-in per_cell).

    Unlike ``subset_fixed_normalization`` (which subsets the R11 global gauge and
    deliberately does NOT re-anchor — so a Cell with no globally-active frame
    raises ``cell gauge has no active frame``), this computes the Cell's OWN gauge
    from the Cell's OWN aligned frames:

    * the reference is the Cell's greatest-valid-support frame (ties resolved to
      the lowest sorted FrameId index — mirroring ``select_canonical_reference``,
      which the frozen estimator invokes internally);
    * every other contributor gets its normalization offset against that LOCAL
      reference.

    The offsets are produced by the FROZEN estimator verbatim
    (``canonical_streaming.compute_fixed_normalization`` -> ``_phase1`` ->
    ``prepare_canonical_inputs`` / ``normalize_canonical_images`` /
    ``compute_canonical_quality_weights``); this function does NOT re-implement
    any science.

    ``provider`` is a :class:`~zemosaic.core.canonical_streaming.CanonicalFrameProvider`
    over the Cell's aligned frames (e.g. the Cell's
    :class:`~zemosaic.core.zegrid.file_provider.MemmapCanonicalProvider`), and
    ``request`` is the ``CanonicalStackRequest`` carrying the science tokens.
    ``request.reference_index`` should be ``None`` (auto) so the Cell selects its
    own reference; an explicit index is honoured (frozen semantics).

    A Cell with ZERO finite support is returned as an EXPLICIT, deterministic
    ``status="empty"`` result (``gauge=None``) — never a fabricated gauge and
    never an exception; a Cell with positive support always yields a valid
    ``FixedNormalization`` with at least one active frame (else it raises loudly,
    it does not fabricate).
    """
    counts = _provider_valid_counts(provider)
    total = int(counts.sum())
    frame_ids = tuple(getattr(provider, "frame_ids", ())) or None
    if total <= 0:
        # Zero finite support across the Cell's frames: explicit + deterministic.
        return CellGaugeResult(
            status=CELL_GAUGE_STATUS_EMPTY,
            gauge=None,
            mode=CELL_GAUGE_MODE_PER_CELL,
            frame_ids=frame_ids,
            reference_index=None,
            total_valid_support=0,
            active_frames=0,
            detail="zero_valid_support",
            reference_frame_id=None,
            reference_support=None,
        )

    # FROZEN estimator, reused verbatim: reference = greatest valid support
    # (tie -> lowest sorted index), per-frame offsets against that local reference.
    gauge = compute_fixed_normalization(provider, request)
    active = int(np.count_nonzero(gauge.weight_active))
    if active <= 0:
        # Positive support but nothing survives weighting: fail loudly, never
        # fabricate a gauge. This is not the zero-support case.
        raise CanonicalStackFailure(
            "per-cell gauge has no active frame despite positive valid support"
        )
    ref = int(gauge.reference_index)
    ref_frame_id = (
        frame_ids[ref] if (frame_ids is not None and 0 <= ref < len(frame_ids)) else None
    )
    return CellGaugeResult(
        status=CELL_GAUGE_STATUS_OK,
        gauge=gauge,
        mode=CELL_GAUGE_MODE_PER_CELL,
        frame_ids=frame_ids,
        reference_index=ref,
        total_valid_support=total,
        active_frames=active,
        detail=None,
        reference_frame_id=ref_frame_id,
        reference_support=int(counts[ref]),
    )
