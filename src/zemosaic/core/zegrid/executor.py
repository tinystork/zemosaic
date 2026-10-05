"""ZM-ZEGRID-R3 — full-layout deterministic executor (Scope A).

Turns the R1/R2 single-cell machinery into a full ``Nx x Ny`` executor over a
frozen layout, in a deterministic ROW-MAJOR Cell-ID order, using the R3
``sky_mean`` LABELLED variant (``r1_frozen_default="linear_fit"``) exactly as in
R2. This module is isolated from production dispatch and does NOT blend / DBE /
equalize / denoise.

Responsibilities:
* Deterministic row-major cell iteration (stable, independent of input order).
* One canonical stack per Cell (reusing the R1/R2 pipeline unchanged).
* Explicit EMPTY-cell and empty-source-ROI recording (no invented data).
* Per-Cell record: ``N`` (patch contributors), effective contributors,
  ``max_surviving``, ``valid_fraction``, exclusions, section-read counts, peak
  RSS, and a non-silent ``status`` (``complete`` / ``empty`` / ``incomplete`` /
  ``blocked_memory`` / ``killed``).

Memory discipline (do not ignore): cells are run ONE AT A TIME in fresh
subprocesses (see ``tools/zegrid_r3/run_executor.py``); the strict memory gate
from ``sweep`` (``>= 1.2 GiB`` available, ``>= 1.6 GiB`` for N > 28) is checked
before every run. Swap usage is accepted; a killed/OOM cell is reported
explicitly and the executor CONTINUES with the remaining cells.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from . import geometry as zg
from . import science_adapter as zs
from . import sweep as zsw

# Frozen R3 executor policy (identical to the R2 sky_mean variant).
EXECUTOR_NORMALIZATION = "sky_mean"
R1_FROZEN_DEFAULT = "linear_fit"
SKY_MEAN_VARIANT_REASON = (
    "linear_fit rejects every non-reference frame on this corpus (single-frame "
    "witness) for TWO reasons: (a) most contributors overlap only in thin/"
    "low-variance common strips whose raw OLS slope on the common mask is already "
    "~0 (< the 0.25 gate), an ill-conditioned fit independent of the MAD step; "
    "(b) for near-full-overlap contributors the MAD robust refinement rejects the "
    "brightest (high-leverage) core pixels first, collapsing the slope to noise "
    "and tripping the gate. Mechanism (b) is demonstrated on near-full "
    "contributors, not the thin ones. Approved sky_mean variant."
)

# Cell execution statuses (non-silent, persisted).
STATUS_COMPLETE = "complete"
STATUS_EMPTY = "empty"
STATUS_INCOMPLETE = "incomplete"
STATUS_BLOCKED_MEMORY = "blocked_memory"
STATUS_KILLED = "killed"


@dataclass(frozen=True)
class ExecutorConfig:
    """Frozen R3 executor science configuration (labelled sky_mean variant)."""

    normalization: str = EXECUTOR_NORMALIZATION
    weighting: str = zs.DEFAULT_WEIGHTING
    rejection: str = zs.DEFAULT_REJECTION
    combine: str = zs.DEFAULT_COMBINE
    backend: str = zs.DEFAULT_BACKEND
    taper: str = zs.DEFAULT_TAPER
    taper_px: float = zs.DEFAULT_TAPER_PX
    taper_floor: float = zs.DEFAULT_TAPER_FLOOR
    r1_frozen_default: str = R1_FROZEN_DEFAULT
    variant_reason: str = SKY_MEAN_VARIANT_REASON

    def science_config(self) -> zs.MiniTileScienceConfig:
        return zs.MiniTileScienceConfig(
            normalization=self.normalization,
            weighting=self.weighting,
            rejection=self.rejection,
            combine=self.combine,
            backend=self.backend,
            taper=self.taper,
            taper_px=self.taper_px,
            taper_floor=self.taper_floor,
        )

    def to_dict(self) -> dict:
        return {
            "normalization": self.normalization,
            "r1_frozen_default": self.r1_frozen_default,
            "variant_reason": self.variant_reason,
            "weighting": self.weighting,
            "rejection": self.rejection,
            "combine": self.combine,
            "backend": self.backend,
            "taper": self.taper,
            "taper_px": self.taper_px,
            "taper_floor": self.taper_floor,
        }


def cell_order(
    canvas: zg.GlobalCanvas, nx: int, ny: int
) -> list[tuple[int, int, str]]:
    """Deterministic ROW-MAJOR Cell-ID order ``(row, col, cell_id)``.

    Row-major: iterate rows 0..ny-1, within each row columns 0..nx-1. Cell IDs
    (``r%04dc%04d``) sort identically, so this order is stable and independent
    of manifest order.
    """
    layout = zg.build_layout(canvas, nx, ny)
    out: list[tuple[int, int, str]] = []
    for row, _col, bounds in layout.iter_cells(canvas):
        out.append((row, _col, zg.cell_id(row, _col)))
    return out


@dataclass
class CellExecutionRecord:
    """Everything observed for one Cell run (status + adequacy + memory)."""

    cell_id: str
    row: int
    col: int
    status: str
    n_patch_contributors: int
    n_core_contributors: int
    core: list  # [x0, y0, x1, y1]
    patch: list  # [x0, y0, x1, y1]
    core_ids: list
    patch_ids: list
    empty_roi_frames: list
    adequacy: dict | None = None
    reference_frame_id: str | None = None
    excluded: list = field(default_factory=list)
    section_read_count: int = 0
    sum_local_px: int = 0
    peak_rss_kib: int = 0
    mem_available_before: int = 0
    mem_available_after: int = 0
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "cell_id": self.cell_id,
            "row": self.row,
            "col": self.col,
            "status": self.status,
            "n_patch_contributors": self.n_patch_contributors,
            "n_core_contributors": self.n_core_contributors,
            "core": self.core,
            "patch": self.patch,
            "core_ids": self.core_ids,
            "patch_ids": self.patch_ids,
            "empty_roi_frames": self.empty_roi_frames,
            "adequacy": self.adequacy,
            "reference_frame_id": self.reference_frame_id,
            "excluded": self.excluded,
            "section_read_count": self.section_read_count,
            "sum_local_px": self.sum_local_px,
            "peak_rss_kib": self.peak_rss_kib,
            "mem_available_before": self.mem_available_before,
            "mem_available_after": self.mem_available_after,
            "message": self.message,
        }


def run_cell_executor(
    frames: Sequence[zg.FrameDescriptor],
    canvas: zg.GlobalCanvas,
    prepared_paths: dict[str, str],
    row: int,
    col: int,
    config: ExecutorConfig,
    nx: int = zsw.NX,
    ny: int = zsw.NY,
    halo_px: int = zsw.HALO_PX,
    tracker=None,
) -> CellExecutionRecord:
    """Run ONE cell's full R1/R2 pipeline and return a full execution record.

    Reuses ``sweep.run_cell_stack`` unchanged (section reads -> local
    reprojection -> ONE canonical request -> core extraction). Handles the EMPTY
    cell case explicitly (0 patch contributors -> ``status=empty``, no invented
    data) and records empty source-ROI frames explicitly.
    """
    cell, patch, mem = zsw.build_cell_context(frames, canvas, row, col, nx, ny, halo_px)
    core = [cell.core.x0, cell.core.y0, cell.core.x1, cell.core.y1]
    patch_bounds = [patch.patch.x0, patch.patch.y0, patch.patch.x1, patch.patch.y1]

    if len(mem.patch_ids) == 0:
        return CellExecutionRecord(
            cell_id=cell.cell_id,
            row=row,
            col=col,
            status=STATUS_EMPTY,
            n_patch_contributors=0,
            n_core_contributors=0,
            core=core,
            patch=patch_bounds,
            core_ids=list(mem.core_ids),
            patch_ids=list(mem.patch_ids),
            empty_roi_frames=[],
            message="empty cell: no patch contributors (no invented data)",
        )

    # Explicit empty source-ROI accounting (patch contributor with zero-area
    # overlap -> plan_source_roi returns None -> skipped by build_patch_contributors).
    by_id = {f.frame_id.logical_path: f for f in frames}
    empty_roi = [
        k
        for k in mem.patch_ids
        if zg.plan_source_roi(by_id[k], canvas, patch) is None
    ]

    res = zsw.run_cell_stack(
        frames, canvas, prepared_paths, row, col, config.science_config(),
        nx=nx, ny=ny, halo_px=halo_px, tracker=tracker,
    )
    adequacy = res.adequacy.to_dict()
    return CellExecutionRecord(
        cell_id=cell.cell_id,
        row=row,
        col=col,
        status=STATUS_COMPLETE,
        n_patch_contributors=res.n_contributors,
        n_core_contributors=res.n_core_contributors,
        core=core,
        patch=patch_bounds,
        core_ids=list(mem.core_ids),
        patch_ids=list(mem.patch_ids),
        empty_roi_frames=empty_roi,
        adequacy=adequacy,
        reference_frame_id=res.reference_frame_id,
        excluded=[list(e) for e in res.excluded],
        section_read_count=len(res.section_reads),
        sum_local_px=sum(r.n_pixels_read for r in res.section_reads),
        peak_rss_kib=res.peak_rss_kib,
        mem_available_before=res.mem_available_before,
        mem_available_after=res.mem_available_after,
    )


def check_executor_gate(n_contributors: int = 0) -> zsw.MemoryGate:
    """Flat R3 executor memory gate: require >= 1.2 GiB before EVERY cell.

    Unlike the R2 sweep gate (which raises to 1.6 GiB for N > 28), the R3
    full-layout executor uses a FLAT 1.2 GiB gate per the mission contract
    (deep-but-small M16 cells and heavy M106 cells are both run under the same
    gate; swap usage is accepted and OOM is reported, never silent).
    """
    available = zsw.read_available_memory()
    required = int(1.2 * 1024**3)
    return zsw.MemoryGate(available=available, required=required, ok=available >= required)


def run_cell_executor_gated(
    frames: Sequence[zg.FrameDescriptor],
    canvas: zg.GlobalCanvas,
    prepared_paths: dict[str, str],
    row: int,
    col: int,
    config: ExecutorConfig,
    nx: int = zsw.NX,
    ny: int = zsw.NY,
    halo_px: int = zsw.HALO_PX,
) -> CellExecutionRecord:
    """Run one cell WITH the strict memory gate (graceful BLOCKED, never OOM)."""
    cell, patch, mem = zsw.build_cell_context(frames, canvas, row, col, nx, ny, halo_px)
    gate = zsw.check_memory_gate(len(mem.patch_ids))
    if not gate.ok:
        return CellExecutionRecord(
            cell_id=cell.cell_id,
            row=row,
            col=col,
            status=STATUS_BLOCKED_MEMORY,
            n_patch_contributors=len(mem.patch_ids),
            n_core_contributors=len(mem.core_ids),
            core=[cell.core.x0, cell.core.y0, cell.core.x1, cell.core.y1],
            patch=[patch.patch.x0, patch.patch.y0, patch.patch.x1, patch.patch.y1],
            core_ids=list(mem.core_ids),
            patch_ids=list(mem.patch_ids),
            empty_roi_frames=[],
            message=(
                f"memory gate unsatisfied: available={gate.available/2**30:.2f}GiB "
                f"< required={gate.required/2**30:.2f}GiB"
            ),
        )
    return run_cell_executor(
        frames, canvas, prepared_paths, row, col, config, nx=nx, ny=ny, halo_px=halo_px
    )
