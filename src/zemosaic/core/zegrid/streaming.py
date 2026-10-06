"""ZM-ZEGRID-R6 — opt-in streaming cell executor (file-backed provider + parity).

Runs ONE real Cell through the R5 streaming canonical executor
(:func:`zemosaic.core.canonical_streaming.run_canonical_stack_streaming`) via the
file-backed :class:`zemosaic.core.zegrid.file_provider.MemmapCanonicalProvider`,
so the aligned input is served from disk instead of being fully RAM-resident.

This is an OPT-IN path (new module + tool). It does NOT change the default
in-memory R1/R2/R3 path (``sweep.run_cell_stack`` / ``executor.run_cell_executor``),
does NOT wire production dispatch, and reuses the frozen ``ExecutorConfig``
(labelled ``sky_mean`` variant) exactly as R3/R4.

Deliverables proven here (see the tool + tests):
1. The resulting MiniTile (science, estimator_weight_sum, support_w1/w2,
   n_eff_support, valid_mask, surviving_sample_count, reference id/exclusions)
   is BIT-EQUAL to the in-memory path for the SAME cell/inputs.
2. Peak RSS is bounded by tile area (measured by the tool).
3. A memory gate is checked before the run and the cache/result are resumable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from zemosaic.core.canonical_engine import (
    CanonicalStackRequest,
    CanonicalStackResult,
)
from zemosaic.core.canonical_streaming import run_canonical_stack_streaming

from . import assembly as za
from . import execution as zxe
from . import geometry as zg
from . import science_adapter as zs
from . import sweep as zsw
from .executor import ExecutorConfig
from .file_provider import (
    MemmapCanonicalProvider,
    build_aligned_cache_from_sources,
)

__all__ = [
    "StreamingCellResult",
    "run_cell_streaming",
    "check_streaming_gate",
    "estimate_streaming_peak_kib",
    "build_streaming_request",
    "compare_minitile_planes",
]


# ---------------------------------------------------------------------------
# Streaming request (placeholder images: only len() is read by the executor)
# ---------------------------------------------------------------------------

def build_streaming_request(
    config: "zs.MiniTileScienceConfig", n_frames: int
) -> CanonicalStackRequest:
    """Build a canonical request for the streaming executor.

    The R5 streaming executor reads ``request.images`` only for ``len()`` (the
    provider serves the actual pixels), so placeholder entries of the right
    length are used — the aligned data is NEVER materialised here.
    """
    return CanonicalStackRequest(
        images=[None] * n_frames,
        geometric_support=[None] * n_frames,
        normalization=config.normalization,
        weighting=config.weighting,
        rejection=config.rejection,
        combine=config.combine,
        reference_index=config.reference_index,
        taper=config.taper,
        taper_px=config.taper_px,
        taper_floor=config.taper_floor,
        backend=config.backend,
        equalize_rgb=config.equalize_rgb,
    )


# ---------------------------------------------------------------------------
# Memory gate + peak estimate (streaming)
# ---------------------------------------------------------------------------

def estimate_streaming_peak_kib(
    n_contributors: int, patch_hw: tuple[int, int], tile_size, channels: int = 3
) -> int:
    """Conservative streaming peak-RSS estimate (KiB), tile-area bound.

    Accounts for (a) the R4-measured process baseline, (b) the per-frame tile
    workspace (float32 + float64 + bool intermediates over ``N x tile_area``),
    (c) the ``O(patch_area)`` output planes, and (d) the residual
    ``rejection_mask`` bool ``(N, H, W, C)`` output plane.
    """
    h, w = patch_hw
    th = min(tile_size, h) if tile_size else h
    tw = min(tile_size, w) if tile_size else w
    # per-frame tile workspace bytes per cell: images_t f32 (4) + images64 (8) +
    # wmap (8) + a_map (8) + survivor bool (1), times channels for the NHWC axes.
    ws_per_cell = channels * (4 + 8 + 8 + 8 + 1)
    workspace_bytes = n_contributors * th * tw * ws_per_cell
    # O(patch_area) output planes.
    out_bytes = h * w * (channels * (4 + 8 + 1 + 8) + 2 * 8 + 8)
    # residual rejection_mask bool (N, H, W, C).
    rej_bytes = n_contributors * h * w * channels * 1
    baseline_kib = 355080  # R4 two-term model intercept (~347 MiB process baseline)
    return baseline_kib + int((workspace_bytes + out_bytes + rej_bytes) / 1024)


def check_streaming_gate(
    n_contributors: int, patch_hw: tuple[int, int], tile_size, channels: int = 3
) -> "zsw.MemoryGate":
    """Memory gate for a streaming cell run (estimated bound + 15% margin)."""
    available = zsw.read_available_memory()
    est = estimate_streaming_peak_kib(n_contributors, patch_hw, tile_size, channels)
    required = int(est * 1.15 * 1024)  # KiB -> bytes (estimate is in KiB)
    return zsw.MemoryGate(available=available, required=required, ok=available >= required)


# ---------------------------------------------------------------------------
# Streaming cell run
# ---------------------------------------------------------------------------

@dataclass
class StreamingCellResult:
    """Everything produced by one streaming cell run (MiniTile + provenance)."""

    cell_id: str
    row: int
    col: int
    minitile: "za.MiniTile"
    canonical_result: CanonicalStackResult
    manifest: dict
    tile_size: object
    n_contributors: int
    reference_frame_id: str | None
    excluded: tuple
    section_read_count: int = 0
    sum_local_px: int = 0
    gate: dict = field(default_factory=dict)
    peak_rss_kib: int = 0
    peak_rss_delta_kib: int = 0

    def to_dict(self) -> dict:
        return {
            "cell_id": self.cell_id,
            "row": self.row,
            "col": self.col,
            "tile_size": self.tile_size,
            "n_contributors": self.n_contributors,
            "reference_frame_id": self.reference_frame_id,
            "excluded": [list(e) for e in self.excluded],
            "section_read_count": self.section_read_count,
            "sum_local_px": self.sum_local_px,
            "gate": self.gate,
            "peak_rss_kib": self.peak_rss_kib,
            "peak_rss_delta_kib": self.peak_rss_delta_kib,
            "manifest": {
                "n_frames": self.manifest.get("n_frames"),
                "height": self.manifest.get("height"),
                "width": self.manifest.get("width"),
                "channels": self.manifest.get("channels"),
                "original_mono": self.manifest.get("original_mono"),
                "total_bytes": self.manifest.get("total_bytes"),
                "schema": self.manifest.get("schema"),
            },
        }


def run_cell_streaming(
    frames,
    canvas,
    prepared_paths: dict[str, str],
    row: int,
    col: int,
    config: "zs.MiniTileScienceConfig",
    cache_dir,
    *,
    tile_size=None,
    nx: int = zsw.NX,
    ny: int = zsw.NY,
    halo_px: int = zsw.HALO_PX,
    reuse_cache: bool = True,
    enforce_gate: bool = True,
) -> StreamingCellResult:
    """Run ONE cell through the streaming executor via the file-backed provider.

    Builds the aligned disk cache (streaming reprojection, resumable), then runs
    the R5 streaming canonical executor with the frozen ``sky_mean`` config and
    extracts the same MiniTile the in-memory path produces. Does NOT run the
    in-memory path (that is the parity reference; see the tool/tests).
    """
    cell, patch, mem = zsw.build_cell_context(frames, canvas, row, col, nx, ny, halo_px)
    if len(mem.patch_ids) == 0:
        raise ValueError(f"cell {cell.cell_id} has no patch contributors")

    if enforce_gate:
        gate = check_streaming_gate(len(mem.patch_ids), patch.patch_shape_hw, tile_size)
        if not gate.ok:
            raise zsw.MemoryInsufficient(
                f"streaming memory gate unsatisfied: available={gate.available/2**30:.2f}GiB "
                f"< required={gate.required/2**30:.2f}GiB"
            )
    else:
        gate = check_streaming_gate(len(mem.patch_ids), patch.patch_shape_hw, tile_size)

    by_id = {f.frame_id.logical_path: f for f in frames}
    patch_frames = [by_id[k] for k in mem.patch_ids]
    crop_plans = {
        f.frame_id.logical_path: zg.plan_source_roi(f, canvas, patch) for f in patch_frames
    }

    rss_before = zsw.peak_rss_kib()
    tracker = zxe.SectionReadTracker()
    manifest = build_aligned_cache_from_sources(
        patch_frames, prepared_paths, canvas, patch, crop_plans, cache_dir,
        tracker=tracker, reuse_cache=reuse_cache,
    )
    provider = MemmapCanonicalProvider(cache_dir)
    request = build_streaming_request(config, provider.n_frames)
    result = run_canonical_stack_streaming(provider, request, tile_size=tile_size)
    rss_after = zsw.peak_rss_kib()

    order = list(provider.frame_ids)
    ref_idx = int(result.provenance["reference"]["index"])
    reference_frame_id = order[ref_idx] if 0 <= ref_idx < len(order) else None
    excluded = tuple(
        (order[idx] if 0 <= idx < len(order) else f"<index {idx}>", stage, reason)
        for idx, stage, reason in result.provenance["excluded_frames"]
    )
    sres = zs.MiniTileScienceResult(
        result=result,
        reference_frame_id=reference_frame_id,
        excluded=excluded,
        frame_order=tuple(order),
    )
    mt = za.extract_minitile(patch, sres)

    return StreamingCellResult(
        cell_id=cell.cell_id,
        row=row,
        col=col,
        minitile=mt,
        canonical_result=result,
        manifest=manifest,
        tile_size=tile_size,
        n_contributors=len(order),
        reference_frame_id=reference_frame_id,
        excluded=excluded,
        section_read_count=len(tracker.records),
        sum_local_px=sum(r.n_pixels_read for r in tracker.records),
        gate={"available": gate.available, "required": gate.required, "ok": gate.ok},
        peak_rss_kib=rss_after,
        peak_rss_delta_kib=max(0, rss_after - rss_before),
    )


# ---------------------------------------------------------------------------
# Parity comparison (in-memory vs streaming MiniTile)
# ---------------------------------------------------------------------------

def compare_minitile_planes(inmem: "za.MiniTile", stream: "za.MiniTile") -> dict:
    """Bit-exact comparison of every MiniTile plane + reference/exclusions.

    Returns a dict ``{plane: {"equal": bool, "max_abs_diff": float}}`` and a
    top-level ``all_equal`` flag. ``max_abs_diff`` is 0.0 when bit-equal (the
    required contract).
    """
    planes = [
        "science",
        "estimator_weight_sum",
        "support_w1",
        "support_w2",
        "n_eff_support",
        "valid_mask",
        "surviving_sample_count",
    ]
    out: dict = {}
    all_equal = True
    for name in planes:
        a = np.asarray(getattr(inmem, name))
        b = np.asarray(getattr(stream, name))
        eq = bool(np.array_equal(a, b, equal_nan=True))
        if eq:
            max_abs = 0.0
        else:
            # NaN-safe max abs diff for the report (bit-exact contract means 0).
            both = ~(np.isnan(a) & np.isnan(b))
            max_abs = float(np.nanmax(np.abs(a - b))) if both.any() else float("nan")
        out[name] = {"equal": eq, "max_abs_diff": max_abs}
        all_equal = all_equal and eq

    out["reference_frame_id"] = {
        "equal": inmem.reference_frame_id == stream.reference_frame_id,
        "inmem": inmem.reference_frame_id,
        "stream": stream.reference_frame_id,
    }
    all_equal = all_equal and out["reference_frame_id"]["equal"]
    out["excluded"] = {
        "equal": inmem.excluded == stream.excluded,
        "inmem": list(inmem.excluded),
        "stream": list(stream.excluded),
    }
    all_equal = all_equal and out["excluded"]["equal"]
    out["all_equal"] = all_equal
    return out
