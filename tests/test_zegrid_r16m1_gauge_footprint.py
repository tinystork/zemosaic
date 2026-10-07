"""ZM-ZEGRID-R16-M1 — canvas-aware per-worker footprint for the gauge.

Nono's R16 M1: the static `_GAUGE_PER_WORKER_FOOTPRINT_BYTES` (~386 MiB, a ~1.9 Mpx
bbox) under-counted the real gauge working set — the pairs pass reprojects the
UNION of the reference's and the frame's bboxes, which can approach the CANVAS
area (~18 Mpx on the real Caldwell canvas). The `RAM // footprint` clamp could
therefore admit more workers than the machine can hold (OOM risk).

Fix (production side; the R16 rule itself is unchanged):
* the footprint is derived from the ACTUAL pixel areas
  (`gauge_footprint_bytes` / `cache_footprint_bytes`), and
* the caller budgets on `RAM_SAFETY_FRACTION` (0.8) of the available RAM,
so `workers * footprint <= 0.8 * available` holds at the decision point.
"""

from __future__ import annotations

from zemosaic.core.zegrid import parallel as zp


CALDWELL_CANVAS_PX = 4117 * 4377          # ~18.02 Mpx (real mosaic canvas)
CALDWELL_AVAIL = int(13.17 * 2**30)       # the user's available RAM at the decision point


def _production_workers(cpu: int, available: int, footprint: int) -> int:
    """Exactly what the production call site does (headroom + adaptive rule)."""
    budget = int(available * zp.RAM_SAFETY_FRACTION)
    return zp.choose_workers(None, budget, footprint)


def test_gauge_footprint_scales_with_the_canvas_area():
    small = zp.gauge_footprint_bytes(1206 * 2040)     # M16 canvas
    big = zp.gauge_footprint_bytes(CALDWELL_CANVAS_PX)
    assert big > small > zp._PER_WORKER_BASELINE_BYTES
    expected_delta = CALDWELL_CANVAS_PX * (3 * 8 + 3 * 4)
    assert big == zp._PER_WORKER_BASELINE_BYTES + expected_delta


def test_caldwell_gauge_worker_count_is_memory_safe():
    """The real case must bound the peak under the safety budget (was 14)."""
    footprint = zp.gauge_footprint_bytes(CALDWELL_CANVAS_PX)
    budget = int(CALDWELL_AVAIL * zp.RAM_SAFETY_FRACTION)
    # cpu=16 explicitly (the user's box); on the real call site the CPU term is
    # os.cpu_count() and binds only when it is smaller than the RAM term.
    workers = zp.adaptive_worker_count(16, budget, footprint)
    assert 2 <= workers <= 14
    # Safety invariant: the workers' peak fits in the budget fraction of RAM.
    assert workers * footprint <= zp.RAM_SAFETY_FRACTION * CALDWELL_AVAIL
    # Strictly fewer than the old static-estimate over-spawn (14).
    legacy = zp.adaptive_worker_count(16, budget, zp._GAUGE_PER_WORKER_FOOTPRINT_BYTES)
    assert workers < legacy


def test_small_box_and_tiny_ram_stay_bounded():
    footprint = zp.gauge_footprint_bytes(CALDWELL_CANVAS_PX)
    # 1 CPU -> 1 worker whatever the RAM.
    assert zp.adaptive_worker_count(1, 32 * 2**30, footprint) == 1
    # RAM that holds exactly two workers -> 2 (memory-bound, below cpu-2).
    assert zp.adaptive_worker_count(16, 2 * footprint, footprint) == 2
    # RAM that cannot hold two workers -> the rule floors to 1.
    assert zp.adaptive_worker_count(16, footprint, footprint) == 1
    # A small box stays bounded by its RAM/CPU.
    assert _production_workers(4, 4 * 2**30, footprint) <= 4


def test_legacy_constants_keep_the_documented_bench_values():
    """The static fallbacks still yield the R16 documented values (rule unchanged)."""
    assert zp.adaptive_worker_count(16, CALDWELL_AVAIL, zp._GAUGE_PER_WORKER_FOOTPRINT_BYTES) == 14
    assert zp.adaptive_worker_count(4, 32 * 2**30, zp._GAUGE_PER_WORKER_FOOTPRINT_BYTES) == 2


def test_cache_footprint_is_smaller_than_the_gauge_canvas_bound():
    patch = zp.cache_footprint_bytes(512 * 512)
    gauge = zp.gauge_footprint_bytes(CALDWELL_CANVAS_PX)
    assert patch < gauge
    assert _production_workers(16, CALDWELL_AVAIL, patch) >= _production_workers(16, CALDWELL_AVAIL, gauge)
