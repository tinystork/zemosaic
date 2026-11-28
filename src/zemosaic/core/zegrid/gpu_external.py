"""ZM-M33 lot 3 — typed GPU admission for the EXTERNAL (disk-spool) class.

This module answers ONE question, purely and explicitly:

    Can a fresh GPU request execute the SAME external two-producer CPU
    preparation + disk spool with the EXISTING ``reduce_spooled_tile``
    GPU reduction, under a bounded device block, or must the product
    truthfully decline and keep the CPU external path?

It deliberately does NOT reuse the whole-tile resident GPU streaming planner
(``gpu.choose_gpu_tile_size`` / ``gpu.gpu_tile_workspace_bytes``). That model
prices ``N x extended_tile_area`` resident frames for the RAM streaming path.
The external reducer never holds ``N x tile`` on the device: it uploads ONE
column block of ``(N, positions)`` at a time and the block always contains ALL
N contributors of every pixel in that block (slicing the N axis would break
quantiles/medians). Applying the whole-tile formula here would either refuse a
feasible external plan or, worse, under-price a large block.

Device allocation audit (the coefficient is empirical, NOT a proved peak)
--------------------------------------------------------------------------
``reduce_spooled_tile`` (backend="gpu") executes, per block:

* rejection stage (only when rejection != none): ``orig2`` f64 (8 B/cell),
  ``survivor2`` bool (1), ``count`` (positions,) int64 (8 B/position), and the
  step result ``new2`` bool (1) while ``survivor2`` is still live; the
  ``degenerate_seen`` accumulator is a ``(positions,)`` bool VECTOR, not a
  per-cell scalar. Winsorized clip additionally allocates its interval/keep
  temporaries (bool masks + f64 bounds on ``(N, positions)``);
* combine stage: ``orig2`` is NOT explicitly released before the combine (the
  combine rebinds ``images64_xp = xp.asarray(orig_b)``, so the original f64
  block can stay live). The combine adds ``images64_xp`` f64 (8),
  ``wmap_xp`` f64 (8), ``eligible_xp`` bool (1), ``masked`` f64 (8),
  ``w_contrib`` f64 (8), ``product`` f64 (8); the median combine sorts along N,
  i.e. a full ``(N, positions)`` sort copy (8) plus its elementwise temporaries;
* every ``(positions,)`` quantile/count/index temporary of the host model is
  also materialised on the device in the quantile helper.

Worst simultaneous live set audited from the code is ~60 B per
(position x frame) with a median sort able to duplicate the largest f64 plane.
We price ``GPU_EXTERNAL_BLOCK_BYTES_PER_POSITION_FRAME = 96`` B, the SAME
coefficient the accepted whole-tile planner uses. It is a conservative POLICY
COEFFICIENT, empirically unqualified for this reducer; it is NOT demonstrated
to be an upper bound (sort/quantile natives, CuPy pool retention and allocator
rounding are not proven by this number).

On top of the array model we reserve an explicit fixed residual
(``GPU_EXTERNAL_FIXED_RESIDUAL_BYTES``) for the CUDA context/first-call
libraries, allocator block rounding and the CuPy pool's retention of blocks
freed between tiles. The accepted policy also keeps 10 % of CURRENTLY FREE VRAM
untouched (``gpu.GPU_VRAM_SAFETY_PERCENT``), so the budget is
``90 % free - fixed residual``. Neither number is a proved device peak or an
8 GiB fit guarantee. The runtime executor owns a classified OOM/CUDA sticky CPU
replay AND a fresh point-of-use revalidation before the first actual GPU
reduction of each Cell; residual shortfalls therefore degrade truthfully to
CPU instead of failing the run.

Single-owner contract
---------------------
The external class is already single-owner (``cells_in_flight == 1``,
prefetch-free). This module REFUSES to admit GPU for any other concurrency: no
two product processes/waves may touch the device without an explicit owner
contract, which this lot does not introduce. The selection is therefore
documented as ``external_cell_exclusive`` K=1.

No science, no I/O, no CuPy import: pure integer arithmetic over a probe dict
supplied by the caller.
"""

from __future__ import annotations

import numbers
from typing import Any

from zemosaic.core.zegrid import gpu as zgpu

__all__ = [
    "GPU_EXTERNAL_SCHEMA",
    "GPU_EXTERNAL_OWNER_MODE",
    "GPU_EXTERNAL_DEVICE_BYTES_PER_POSITION_FRAME",
    "GPU_EXTERNAL_FIXED_POSITION_BYTES",
    "GPU_EXTERNAL_FIXED_RESIDUAL_BYTES",
    "GPU_EXTERNAL_SAFETY_PERCENT",
    "external_gpu_block_bytes",
    "device_budget_bytes",
    "block_geometry_for_positions",
    "largest_device_block_positions",
    "plan_external_gpu",
    "validate_external_gpu_point_of_use",
]

GPU_EXTERNAL_SCHEMA = "zemosaic-zegrid-external-gpu-admission-v1"

#: The whole external class is one Cell at a time; this is the ONLY admitted
#: device owner contract in this lot.
GPU_EXTERNAL_OWNER_MODE = "external_cell_exclusive"

#: Conservative device bytes per (position x frame) for ONE reduction block.
#: Audited against ``reduce_spooled_tile`` (see module docstring): ~60 B live
#: f64/bool planes, doubled for a median sort copy, rounded up to 96 B.
GPU_EXTERNAL_DEVICE_BYTES_PER_POSITION_FRAME = 96

#: Conservative fixed device bytes per position (quantile/count/index and
#: elementwise temporaries). Same figure as the host model.
GPU_EXTERNAL_FIXED_POSITION_BYTES = 512

#: Explicit fixed residual beyond the array model: CUDA context/libraries,
#: allocator block rounding and pool retention between tiles. NOT a proved
#: peak; it is a declared reserve.
GPU_EXTERNAL_FIXED_RESIDUAL_BYTES = 16 * 1024 * 1024

#: Reuse the SINGLE product VRAM policy (90 % of currently free VRAM).
GPU_EXTERNAL_SAFETY_PERCENT = int(zgpu.GPU_VRAM_SAFETY_PERCENT)


def _plain_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a plain int, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


def _probe_integral(value: Any) -> tuple[int | None, bool]:
    """Normalize a probe integer; ``bool`` is never a byte count.

    Returns ``(value, ok)``. Missing (``None``) is ``(None, True)``; a malformed
    value is ``(None, False)``. ``numbers.Integral`` accepts ``numpy.int64``.
    """
    if value is None:
        return None, True
    if isinstance(value, bool):
        return None, False
    if isinstance(value, numbers.Integral):
        return int(value), True
    return None, False


def _normalize_probe(probe: Any) -> tuple[dict | None, str | None]:
    """Validate/normalize ONE probe dict; NEVER raises.

    A probe is caller/library input, not a programmer-controlled parameter:
    every malformed shape becomes ``probe_invalid`` instead of an exception.
    Unknown FREE VRAM is deliberately distinct from a known total: a total is
    NOT a substitute for a free figure.
    """
    if not isinstance(probe, dict):
        return None, "probe_invalid"
    try:
        available = probe.get("available")
        if not isinstance(available, bool):
            return None, "probe_invalid"
        free, ok_free = _probe_integral(probe.get("vram_free_bytes"))
        total, ok_total = _probe_integral(probe.get("vram_total_bytes"))
        if not ok_free or not ok_total:
            return None, "probe_invalid"
        if free is not None and free < 0:
            return None, "probe_invalid"
        if total is not None and total < 0:
            return None, "probe_invalid"
        if free is not None and total is not None and free > total:
            return None, "probe_invalid"
        device = probe.get("device")
        if device is not None and not isinstance(device, str):
            return None, "probe_invalid"
        cupy_version = probe.get("cupy_version")
        if cupy_version is not None and not isinstance(cupy_version, str):
            return None, "probe_invalid"
        reason = probe.get("reason")
        if reason is not None and not isinstance(reason, str):
            return None, "probe_invalid"
        record = {
            "available": bool(available),
            "device": device,
            "cupy_version": cupy_version,
            "vram_total_bytes": total,
            "vram_free_bytes": free,
            "reason": reason,
            "label": "instantaneous_snapshot_not_high_water",
        }
        return record, None
    except Exception:
        # Defensive: a hostile mapping must still decline, not raise.
        return None, "probe_invalid"


def external_gpu_block_bytes(
    *,
    n_frames: int,
    positions: int,
    sample_bytes: int = GPU_EXTERNAL_DEVICE_BYTES_PER_POSITION_FRAME,
    fixed_position_bytes: int = GPU_EXTERNAL_FIXED_POSITION_BYTES,
) -> int:
    """Complete device live-set bound for ONE block (all N contributors)."""
    n = _plain_int(n_frames, "n_frames", minimum=1)
    pos = _plain_int(positions, "positions", minimum=1)
    sample = _plain_int(sample_bytes, "sample_bytes", minimum=1)
    fixed = _plain_int(fixed_position_bytes, "fixed_position_bytes", minimum=0)
    return int(pos * (n * sample + fixed))


def device_budget_bytes(probe: dict) -> tuple[int | None, str | None]:
    """``(budget, reason)`` from a fresh ``probe_gpu_backend`` dict.

    ``budget`` is ``90 % of CURRENT free VRAM`` (integer), and ``None`` with a
    closed reason when the device is unavailable, the probe is malformed, the
    free figure is unknown, or the resulting budget is below the accepted floor.
    A known TOTAL is NEVER used as a substitute for an unknown FREE figure: an
    unknown free observation declines. The budget is NEVER inflated to the
    floor. This function never raises on probe input.
    """
    normalized, err = _normalize_probe(probe)
    if err is not None or normalized is None:
        return None, str(err or "probe_invalid")
    if not normalized["available"]:
        return None, "gpu_unavailable"
    free = normalized.get("vram_free_bytes")
    if free is None:
        # A total is not a free observation: another process may own the card.
        return None, "vram_unknown"
    budget = (int(free) * GPU_EXTERNAL_SAFETY_PERCENT) // 100
    if budget < int(zgpu.GPU_MIN_VRAM_BUDGET_BYTES):
        return None, "vram_below_floor"
    return int(budget), None


def block_geometry_for_positions(
    positions: int, *, interior_hw: tuple[int, int], channels: int
) -> tuple[int, int, int]:
    """Map a POSITION budget to ``(rows, cols, positions)`` inside the tile.

    Identical shape policy to :func:`external_stream.batch_positions_for`:
    whole interior rows when the budget covers at least one row, otherwise a
    single row split into whole pixels (>= 1 pixel). Never crosses the interior.
    """
    pos = _plain_int(positions, "positions", minimum=1)
    ih = _plain_int(interior_hw[0], "interior_height", minimum=1)
    iw = _plain_int(interior_hw[1], "interior_width", minimum=1)
    c = _plain_int(channels, "channels", minimum=1)
    per_row = iw * c
    if pos >= per_row:
        rows = min(ih, pos // per_row)
        return int(rows), int(iw), int(rows * iw * c)
    cols = min(iw, max(1, pos // c))
    return 1, int(cols), int(cols * c)


def largest_device_block_positions(
    *,
    block_budget_bytes: int,
    n_frames: int,
    interior_hw: tuple[int, int],
    channels: int,
    sample_bytes: int = GPU_EXTERNAL_DEVICE_BYTES_PER_POSITION_FRAME,
    fixed_position_bytes: int = GPU_EXTERNAL_FIXED_POSITION_BYTES,
) -> int:
    """Largest pixel-granular block that fits ``block_budget_bytes``.

    The returned value is in POSITIONS (``rows * cols * channels``) and is
    always a whole number of pixels. Zero means not even ONE pixel (a block
    that still contains ALL N contributors) fits. Integer arithmetic only.
    """
    budget = _plain_int(block_budget_bytes, "block_budget_bytes", minimum=0)
    n = _plain_int(n_frames, "n_frames", minimum=1)
    ih = _plain_int(interior_hw[0], "interior_height", minimum=1)
    iw = _plain_int(interior_hw[1], "interior_width", minimum=1)
    c = _plain_int(channels, "channels", minimum=1)
    per_position = n * _plain_int(sample_bytes, "sample_bytes", minimum=1) + (
        _plain_int(fixed_position_bytes, "fixed_position_bytes", minimum=0)
    )
    if budget < per_position * c:
        return 0
    positions = (budget // per_position // c) * c
    positions = min(int(positions), int(ih * iw * c))
    rows, cols, clamped_positions = block_geometry_for_positions(
        positions, interior_hw=(ih, iw), channels=c
    )
    return int(clamped_positions)


def plan_external_gpu(
    *,
    requested: bool,
    probe: dict | None,
    host_phase_budget_bytes: int,
    host_batch: int,
    n_frames: int,
    interior_hw: tuple[int, int],
    channels: int,
    tile_size: int,
    cells_in_flight: int = 1,
    active_scheduler: bool = True,
    frozen_backend: str | None = None,
) -> dict:
    """Decide ONE external GPU admission; NEVER raises on a refusal.

    The result is a closed, JSON-safe record. ``admitted`` is True ONLY when a
    fresh probe was available AND the largest device block that also respects
    the already-admitted host batch fits the bounded device budget. Every
    refusal names a stable ``reason`` so the product can persist a truthful
    requested/selected/effective provenance and keep the CPU external path.
    """
    n = _plain_int(n_frames, "n_frames", minimum=1)
    c = _plain_int(channels, "channels", minimum=1)
    ih = _plain_int(interior_hw[0], "interior_height", minimum=1)
    iw = _plain_int(interior_hw[1], "interior_width", minimum=1)
    host_positions = _plain_int(host_batch, "host_batch", minimum=1)
    phase_budget = _plain_int(
        host_phase_budget_bytes, "host_phase_budget_bytes", minimum=1
    )
    k = _plain_int(cells_in_flight, "cells_in_flight", minimum=1)
    normalized_probe, _probe_err = _normalize_probe(probe)
    record: dict[str, Any] = {
        "schema": GPU_EXTERNAL_SCHEMA,
        "requested_backend": "gpu" if requested else "cpu",
        "selected_backend": "cpu",
        "effective_backend": "cpu",
        "admitted": False,
        "reason": None,
        "detail": None,
        "owner": {
            "mode": GPU_EXTERNAL_OWNER_MODE,
            "cells_in_flight": int(k),
            "concurrent_cells": int(k),
        },
        "probe": (
            dict(normalized_probe)
            if normalized_probe is not None
            else {
                "available": False,
                "device": None,
                "cupy_version": None,
                "vram_total_bytes": None,
                "vram_free_bytes": None,
                "reason": "probe_invalid",
                "label": "instantaneous_snapshot_not_high_water",
            }
        ),
        "policy": {
            "safety_percent": int(GPU_EXTERNAL_SAFETY_PERCENT),
            "device_bytes_per_position_frame": int(
                GPU_EXTERNAL_DEVICE_BYTES_PER_POSITION_FRAME
            ),
            "fixed_position_bytes": int(GPU_EXTERNAL_FIXED_POSITION_BYTES),
            "fixed_residual_bytes": int(GPU_EXTERNAL_FIXED_RESIDUAL_BYTES),
            "host_phase_budget_bytes": int(phase_budget),
            "note": (
                "conservative POLICY coefficient, empirically unqualified for "
                "this reducer; NOT a proved upper bound. The 16 MiB residual "
                "is a declared reserve, not a guarantee. Point-of-use "
                "revalidation is fail-closed but is not a peak proof."
            ),
        },
        "batch": {
            "requested": int(host_positions),
            "selected": int(host_positions),
            "rows": None,
            "cols": None,
            "positions": int(host_positions),
            "clamped": False,
        },
        "device": None,
        "tile_size": int(tile_size),
        "host_batch": int(host_positions),
        "interior_hw": [int(ih), int(iw)],
        "n_frames": int(n),
        "channels": int(c),
    }

    def _decline(reason: str, detail: str, **extra) -> dict:
        record["reason"] = str(reason)
        record["detail"] = str(detail)
        record.update(extra)
        return record

    if _probe_err is not None or normalized_probe is None:
        return _decline(
            "probe_invalid",
            "the GPU probe is malformed (non-integral/negative/contradictory "
            "bytes); declining instead of raising in planning",
        )
    if not requested:
        return _decline("gpu_not_requested", "the product did not request a GPU")
    if not bool(active_scheduler):
        return _decline(
            "scheduler_not_active",
            "external GPU admission is limited to the ACTIVE scheduler; the "
            "legacy/observe scheduler keeps the CPU external class",
        )
    if k != 1:
        return _decline(
            "gpu_owner_concurrency_refused",
            "the external class is single-owner (K=1); no concurrent per-Cell "
            f"GPU is admitted without a device-owner contract (got K={k})",
        )
    if frozen_backend is not None and str(frozen_backend) == "cpu":
        return _decline(
            "frozen_backend_cpu",
            "the run is frozen on the CPU backend; a GPU backend is never "
            "injected into a CPU-frozen run",
        )
    budget, why = device_budget_bytes(normalized_probe)
    if budget is None:
        return _decline(
            str(why or "gpu_unavailable"),
            "no usable fresh VRAM observation; declining the GPU before "
            "frozen admission instead of promising an unvalidated fit",
        )
    block_budget = int(budget) - int(GPU_EXTERNAL_FIXED_RESIDUAL_BYTES)
    record["device"] = {
        "budget_bytes": int(budget),
        "block_budget_bytes": int(block_budget),
        "free_vram_bytes": (
            None
            if probe.get("vram_free_bytes") is None
            else int(probe["vram_free_bytes"])
        ),
        "residual_bytes": int(GPU_EXTERNAL_FIXED_RESIDUAL_BYTES),
    }
    if block_budget <= 0:
        return _decline(
            "vram_residual_floor",
            "the 90 % VRAM budget does not cover the declared fixed residual "
            f"({GPU_EXTERNAL_FIXED_RESIDUAL_BYTES} B)",
        )
    device_positions = largest_device_block_positions(
        block_budget_bytes=int(block_budget),
        n_frames=int(n),
        interior_hw=(int(ih), int(iw)),
        channels=int(c),
    )
    if device_positions < int(c):
        return _decline(
            "vram_block_floor",
            "not even ONE pixel with ALL N contributors fits the device block "
            f"budget {block_budget} B",
            device_block_minimum_bytes=int(
                external_gpu_block_bytes(n_frames=int(n), positions=int(c))
            ),
        )
    # Never GROW the batch the CPU/host admission already accepted: the batch
    # is a staging parameter and the device may only shrink it.
    selected_positions = min(int(host_positions), int(device_positions))
    rows, cols, selected_positions = block_geometry_for_positions(
        selected_positions, interior_hw=(int(ih), int(iw)), channels=int(c)
    )
    block_bytes = external_gpu_block_bytes(
        n_frames=int(n), positions=int(selected_positions)
    )
    if block_bytes > int(block_budget):
        # Defensive: the exact geometry must fit the same bound.
        return _decline(
            "vram_block_floor",
            "the selected pixel block does not fit the device budget after "
            f"geometry resolution ({block_bytes} B > {block_budget} B)",
        )
    record["batch"] = {
        "requested": int(host_positions),
        "selected": int(selected_positions),
        "rows": int(rows),
        "cols": int(cols),
        "positions": int(selected_positions),
        "clamped": bool(int(selected_positions) != int(host_positions)),
    }
    record["device"]["block_bytes"] = int(block_bytes)
    record["device"]["block_positions"] = int(selected_positions)
    record["host"] = {
        "phase_budget_bytes": int(phase_budget),
        "batch": int(selected_positions),
        "note": (
            "the host peak is priced by the caller at the SAME selected batch; "
            "a batch shrink can only lower it"
        ),
    }
    record["selected_backend"] = "gpu"
    record["effective_backend"] = "gpu"
    record["admitted"] = True
    record["reason"] = "vram_and_host_fit"
    record["detail"] = (
        f"device block {selected_positions} positions x {n} contributors = "
        f"{block_bytes} B <= {block_budget} B (90 % free - declared residual); "
        "K=1 external owner"
    )
    return record


def validate_external_gpu_point_of_use(
    *,
    probe,
    n_frames: int,
    positions: int,
    interior_hw: tuple[int, int],
    channels: int,
) -> dict:
    """Fail-closed point-of-use revalidation of the FROZEN external block.

    Called immediately before the FIRST actual GPU reduction of each Cell (never
    for a Cell that only reuses checkpoints). It re-observes the CURRENT free
    VRAM and checks the SAME frozen batch workspace + declared residual. It
    never adapts or re-identifies the batch: an unknown/insufficient current
    observation must decline to the sticky CPU execution path.

    ``probe`` is caller/library input: every malformed shape declines with
    ``probe_invalid`` and never raises. This is a point check, NOT a peak proof.
    """
    n = _plain_int(n_frames, "n_frames", minimum=1)
    pos = _plain_int(positions, "positions", minimum=1)
    ih = _plain_int(interior_hw[0], "interior_height", minimum=1)
    iw = _plain_int(interior_hw[1], "interior_width", minimum=1)
    c = _plain_int(channels, "channels", minimum=1)
    if pos < c or pos > ih * iw * c:
        raise ValueError(
            f"positions {pos} is outside the frozen interior block "
            f"[{c}, {ih * iw * c}]"
        )
    normalized, err = _normalize_probe(probe)
    record: dict[str, Any] = {
        "schema": "zemosaic-zegrid-external-gpu-point-of-use-v1",
        "ok": False,
        "reason": None,
        "detail": None,
        "positions": int(pos),
        "n_frames": int(n),
        "probe": (
            dict(normalized)
            if normalized is not None
            else {
                "available": False,
                "device": None,
                "cupy_version": None,
                "vram_total_bytes": None,
                "vram_free_bytes": None,
                "reason": "probe_invalid",
                "label": "instantaneous_snapshot_not_high_water",
            }
        ),
        "policy": {
            "safety_percent": int(GPU_EXTERNAL_SAFETY_PERCENT),
            "device_bytes_per_position_frame": int(
                GPU_EXTERNAL_DEVICE_BYTES_PER_POSITION_FRAME
            ),
            "fixed_position_bytes": int(GPU_EXTERNAL_FIXED_POSITION_BYTES),
            "fixed_residual_bytes": int(GPU_EXTERNAL_FIXED_RESIDUAL_BYTES),
            "note": (
                "point-of-use fail-closed policy check; the coefficient and "
                "residual are NOT a proved peak guarantee"
            ),
        },
        "required_block_bytes": int(
            external_gpu_block_bytes(n_frames=n, positions=pos)
        ),
        "residual_bytes": int(GPU_EXTERNAL_FIXED_RESIDUAL_BYTES),
        "budget_bytes": None,
    }

    def _decline(reason: str, detail: str) -> dict:
        record["reason"] = str(reason)
        record["detail"] = str(detail)
        return record

    if err is not None or normalized is None:
        return _decline(
            "probe_invalid",
            "the current GPU probe is malformed; declining to CPU before any "
            "GPU allocation attempt",
        )
    if not normalized["available"]:
        return _decline(
            "gpu_unavailable",
            "the current GPU probe reports the device unavailable",
        )
    free = normalized.get("vram_free_bytes")
    if free is None:
        return _decline(
            "vram_unknown",
            "current free VRAM is unknown; a known total is not a free figure",
        )
    budget = (int(free) * GPU_EXTERNAL_SAFETY_PERCENT) // 100
    record["budget_bytes"] = int(budget)
    if budget < int(zgpu.GPU_MIN_VRAM_BUDGET_BYTES):
        return _decline(
            "vram_below_floor",
            f"current 90 % budget {budget} B is below the accepted floor",
        )
    required = int(record["required_block_bytes"]) + int(
        GPU_EXTERNAL_FIXED_RESIDUAL_BYTES
    )
    if budget < required:
        return _decline(
            "vram_block_floor",
            f"current 90 % budget {budget} B cannot hold the FROZEN block "
            f"{record['required_block_bytes']} B + declared residual "
            f"{GPU_EXTERNAL_FIXED_RESIDUAL_BYTES} B",
        )
    record["ok"] = True
    record["reason"] = "vram_block_fit"
    record["detail"] = (
        f"current 90 % budget {budget} B holds frozen block "
        f"{record['required_block_bytes']} B + residual "
        f"{GPU_EXTERNAL_FIXED_RESIDUAL_BYTES} B"
    )
    return record
