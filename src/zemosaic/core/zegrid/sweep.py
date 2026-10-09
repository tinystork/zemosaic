"""ZM-ZEGRID-R2 — multi-frame candidate sweep + strict memory guard.

R2 must demonstrate **genuine** multi-frame local canonical stacking on a real
Cell (effective contributor count >= 3) and check adjacent-Cell continuity, under
a strict memory budget. This module provides:

* a deterministic candidate ordering (ascending patch-contributor count ``N``,
  tie-break by cell id, excluding the frozen corner ``r0000c0000`` and the
  symmetric weak corner ``r0003c0004``),
* a strict memory gate (read ``/proc/meminfo`` ``MemAvailable``; require
  ``>= 1.2 GiB`` per run; ``>= 1.6 GiB`` for ``N > 28``),
* a bounded per-cell run that reuses the R1 geometry/execution/science_adapter/
  assembly pipeline unchanged and records peak RSS (``RUSAGE_SELF``).

Memory policy (do not ignore): the host has ~7.5 GiB RAM with ~1.0–1.5 GiB
available and a 17 GiB swap; ``/tmp`` is tmpfs (= RAM) and is NOT used for
fixtures. Cells are run ONE AT A TIME; N is capped at 28 by default and 40 only
when ``>= 1.6 GiB`` is available. If the gate cannot be satisfied the runner
aborts gracefully with an explicit ``MemoryInsufficient`` (BLOCKED), never OOM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

try:
    import resource
except ImportError:  # pragma: no cover - Windows (no POSIX resource module)
    resource = None

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is a product dependency
    psutil = None

import numpy as np

from .geometry import (
    FrameDescriptor,
    GlobalCanvas,
    ZeGridCell,
    build_layout,
    build_patch,
    cell_id,
    compute_membership,
    plan_source_roi,
)

# --- memory policy constants -------------------------------------------------
MIN_AVAILABLE_BYTES = int(1.2 * 1024**3)      # required before every cell run
EXTENDED_AVAILABLE_BYTES = int(1.6 * 1024**3)  # required when N > 28
DEFAULT_MAX_CONTRIBUTORS = 28
EXTENDED_MAX_CONTRIBUTORS = 40
DEFAULT_MAX_CELLS = 6

# Frozen corner (R1 single-frame witness) + symmetric weak corner (r0003c0004):
# both are corners with thin overlap and are excluded from the multi-frame sweep.
EXCLUDED_CELLS = frozenset({"r0000c0000", "r0003c0004"})

HALO_PX = 8
NX = 5
NY = 4


class MemoryInsufficient(RuntimeError):
    """Raised when the strict memory gate cannot be satisfied (graceful abort)."""


def read_available_memory() -> int:
    """Return available memory in bytes (PORTABLE, psutil-first).

    ``psutil.virtual_memory().available`` is the product's portable probe (psutil
    is already a dependency) and works on Windows/macOS/Linux. A ``/proc/meminfo``
    fallback is kept only for environments where psutil is unavailable; the
    production wiring MUST use this function (never ``resource`` / ``/proc``
    directly) so ZeGrid stays Windows-portable.
    """
    if psutil is not None:
        try:
            return int(psutil.virtual_memory().available)
        except Exception:  # pragma: no cover - defensive
            pass
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception as exc:  # pragma: no cover - fallback failure
        raise OSError("unable to determine available memory (no psutil, no /proc)") from exc
    raise OSError("unable to determine available memory (no psutil, no /proc)")


def required_available_bytes(n_contributors: int) -> int:
    """Required available memory (bytes) for a run with ``n_contributors``."""
    return EXTENDED_AVAILABLE_BYTES if n_contributors > DEFAULT_MAX_CONTRIBUTORS else MIN_AVAILABLE_BYTES


def peak_rss_kib() -> int:
    """Peak RSS (KiB) of the current process so far (portable).

    Uses ``resource.getrusage(RUSAGE_SELF).ru_maxrss`` where available (Linux) and
    falls back to ``psutil.Process().memory_info().rss`` (current RSS, Windows /
    non-POSIX) so reporting never crashes on Windows.
    """
    if resource is not None:
        try:
            return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        except Exception:  # pragma: no cover - defensive
            pass
    if psutil is not None:
        try:
            return int(psutil.Process().memory_info().rss // 1024)
        except Exception:  # pragma: no cover - defensive
            pass
    return 0


def aggregate_peak_rss_kib(parent_peak_kib: int, worker_peak_kib: list[int]) -> int:
    """Upper-bound aggregate peak RSS across the parent + all cell-worker processes.

    ZM-ZEGRID-R22 honesty fix: ``peak_rss_kib`` is a SINGLE process's peak
    (``RUSAGE_SELF`` on Linux / current RSS on Windows); it does NOT capture the
    sum of a parallel batch. This helper reports the batch aggregate as
    ``parent_peak + sum(worker_peaks)``. Each worker's peak includes its own
    ~340 MiB import baseline, so this is an UPPER BOUND (shared/copy-on-write
    pages are not deduplicated) — reported alongside ``peak_rss_kib``, never as a
    replacement for it.
    """
    try:
        parent = int(parent_peak_kib)
    except (TypeError, ValueError):
        parent = 0
    total = parent
    for w in worker_peak_kib:
        try:
            total += int(w)
        except (TypeError, ValueError):
            pass
    return total


@dataclass(frozen=True)
class MemoryGate:
    """Result of one memory-gate check."""

    available: int
    required: int
    ok: bool


def check_memory_gate(n_contributors: int) -> MemoryGate:
    """Check the memory gate for a run with ``n_contributors`` contributors."""
    available = read_available_memory()
    required = required_available_bytes(n_contributors)
    return MemoryGate(available=available, required=required, ok=available >= required)


@dataclass(frozen=True)
class Candidate:
    """One sweep candidate cell with its (patch/core) contributor counts."""

    cell_id: str
    row: int
    col: int
    n_contributors: int        # patch contributors actually stacked
    n_core_contributors: int   # core-only (geometric) contributors


def candidate_order(
    frames: Sequence[FrameDescriptor],
    canvas: GlobalCanvas,
    nx: int = NX,
    ny: int = NY,
    halo_px: int = HALO_PX,
    excluded: frozenset[str] = EXCLUDED_CELLS,
) -> list[Candidate]:
    """Deterministic candidate ordering: ascending N, tie-break by cell id.

    ``N`` is the number of **patch** contributors (the frames actually read and
    stacked), matching the mission's R0 counts (r0000c0001=23, ... r0002c0004=28)
    and the memory-budget semantics (contributors per run). Cells in ``excluded``
    (frozen corner ``r0000c0000`` + symmetric weak corner ``r0003c0004``) are
    skipped. Result is sorted by ``(n_contributors, cell_id)``.
    """
    layout = build_layout(canvas, nx, ny)
    out: list[Candidate] = []
    for row, col, bounds in layout.iter_cells(canvas):
        cid = cell_id(row, col)
        if cid in excluded:
            continue
        cell = ZeGridCell(cid, canvas.canvas_id, layout.layout_id, row, col, bounds)
        patch = build_patch(canvas, cell, halo_px)
        mem = compute_membership(frames, canvas, cell, patch)
        out.append(
            Candidate(
                cell_id=cid,
                row=row,
                col=col,
                n_contributors=len(mem.patch_ids),
                n_core_contributors=len(mem.core_ids),
            )
        )
    out.sort(key=lambda c: (c.n_contributors, c.cell_id))
    return out


def build_cell_context(
    frames: Sequence[FrameDescriptor],
    canvas: GlobalCanvas,
    row: int,
    col: int,
    nx: int = NX,
    ny: int = NY,
    halo_px: int = HALO_PX,
):
    """Build ``(cell, patch, membership)`` for one cell (R1 geometry unchanged)."""
    layout = build_layout(canvas, nx, ny)
    cid = cell_id(row, col)
    cell = ZeGridCell(cid, canvas.canvas_id, layout.layout_id, row, col,
                      layout.cell_bounds(row, col, canvas))
    patch = build_patch(canvas, cell, halo_px)
    mem = compute_membership(frames, canvas, cell, patch)
    return cell, patch, mem


@dataclass
class CellRunResult:
    """Everything produced by running one cell (stack + adequacy + RSS)."""

    cell_id: str
    row: int
    col: int
    patch: object
    membership: object
    minitile: object
    science_result: object
    adequacy: object
    n_contributors: int
    n_core_contributors: int
    reference_frame_id: str | None
    excluded: tuple[tuple[str, str, str], ...]
    section_reads: list = field(default_factory=list)
    peak_rss_kib: int = 0
    peak_rss_delta_kib: int = 0
    mem_available_before: int = 0
    mem_available_after: int = 0


def run_cell_stack(
    frames: Sequence[FrameDescriptor],
    canvas: GlobalCanvas,
    prepared_paths: dict[str, str],
    row: int,
    col: int,
    config,
    nx: int = NX,
    ny: int = NY,
    halo_px: int = HALO_PX,
    tracker=None,
) -> CellRunResult:
    """Run the full R1 local pipeline for one cell (unchanged science path).

    Reuses ``execution`` (section reads + local reprojection), ``science_adapter``
    (one canonical request), and ``assembly`` (core crop). Records peak RSS and
    memory before/after, and derives witness adequacy. Does NOT modify the R1
    science path — only wraps it with R2 measurement/adequacy.
    """
    from . import assembly as za
    from . import execution as zx
    from . import science_adapter as zs
    from .adequacy import compute_adequacy

    cell, patch, mem = build_cell_context(frames, canvas, row, col, nx, ny, halo_px)
    by_id = {f.frame_id.logical_path: f for f in frames}
    patch_frames = [by_id[k] for k in mem.patch_ids]
    crop_plans = {
        f.frame_id.logical_path: plan_source_roi(f, canvas, patch) for f in patch_frames
    }
    if tracker is None:
        tracker = zx.SectionReadTracker()

    rss_before = peak_rss_kib()
    mem_before = read_available_memory()

    contribs = zx.build_patch_contributors(
        patch_frames, prepared_paths, canvas, patch, crop_plans, tracker=tracker
    )
    sres = zs.run_minitile_stack(
        [c.rgb for c in contribs],
        [c.geometric_support for c in contribs],
        [c.frame_id for c in contribs],
        config,
    )
    mt = za.extract_minitile(patch, sres)

    rss_after = peak_rss_kib()
    mem_after = read_available_memory()
    adequacy = compute_adequacy(
        len(contribs),
        sres.excluded,
        sres.result.surviving_sample_count,
        sres.result.valid_mask,
    )
    return CellRunResult(
        cell_id=cell.cell_id,
        row=row,
        col=col,
        patch=patch,
        membership=mem,
        minitile=mt,
        science_result=sres,
        adequacy=adequacy,
        n_contributors=len(contribs),
        n_core_contributors=len(mem.core_ids),
        reference_frame_id=sres.reference_frame_id,
        excluded=sres.excluded,
        section_reads=list(tracker.records),
        peak_rss_kib=rss_after,
        peak_rss_delta_kib=max(0, rss_after - rss_before),
        mem_available_before=mem_before,
        mem_available_after=mem_after,
    )
