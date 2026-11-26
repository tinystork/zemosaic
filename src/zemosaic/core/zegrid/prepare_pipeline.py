"""ZM-M33-PREP-PIPELINE — bounded two-producer tile preparation.

ONE tile is prepared frame by frame by at most TWO independent producers; the
CALLING thread is the ONE writer that stores ``images[i]`` / ``weights[i]`` in
the original frame index order. No contribution can be permuted: every result
is written at its own original index, and the reduction still sees the exact
same spool payload as the legacy sequential writer.

Design constraints (see ZM-M33-PREP-PIPELINE-20261125 mandate):

* admission happens BEFORE any provider/thread/allocation: a byte grant that
  cannot be covered is refused (never silently reduced to ``max(1, ...)``);
* credits are counted in BYTES and cover producer working sets, ready payloads
  and the references held by the single writer. A slow writer therefore
  back-pressures the producers instead of growing an unbounded queue;
* every failure stops admission, cancels pending work, drains already started
  producers and keeps the ORIGINAL exception visible. Threads are not
  preemptible: if a producer does not finish within the drain deadline the
  provider set is QUARANTINED (no reuse, no cleanup of files it may still use)
  and the Cell fails;
* providers are private to one producer, opened from an explicit factory and
  closed before the caller starts the tile reduction. The only cross-producer
  state is a cell-agnostic, immutable CFA min/max publication registry.

Legacy compatibility: this module is only used when a factory is explicitly
injected. ``ExternalSpool.write_tile`` keeps the historical sequential path and
shares the exact per-frame body via :func:`prepare_external_frame`.
"""

from __future__ import annotations

import queue as _queue
import threading
import time
from dataclasses import dataclass

import numpy as np

from .external_stream import (
    ExternalBudgetError,
    ExternalSpoolError,
    prepare_external_frame,
)

__all__ = [
    "PREP_PIPELINE_ENV",
    "PREP_PIPELINE_MODES",
    "HOST_RESERVE_BYTES",
    "CFA_PRESCAN_BYTES",
    "PreparationAborted",
    "PreparationBudgetError",
    "PreparationQuarantineError",
    "CfaMinMaxRegistry",
    "ByteCreditLedger",
    "PreparationPipeline",
    "preparation_grant_components",
    "decide_preparation_pipeline",
    "write_tile_two_producers",
]

#: Internal comparison selector. Product default is ``auto``: two producers
#: ONLY when the constructive budget is admissible and qualified; every fallback
#: is explicit in the published reason. ``legacy`` is the internal A/B control,
#: never the dormant delivery.
PREP_PIPELINE_ENV = "ZEMOSAIC_M33_PREP_PIPELINE"
PREP_PIPELINE_MODES = ("auto", "legacy")

#: Host RAM that MUST remain free after the two preparation grants (the user
#: run shares this host). Not a soft target: admission is refused below it.
HOST_RESERVE_BYTES = 512 * 2 ** 20

#: Per-producer explicit margin over the named components (reprojection native
#: scratch, allocator slack, dict/metadata cost of the stats/p1 plumbing).
PREP_MARGIN_NUM = 5
PREP_MARGIN_DEN = 4
#: Fixed residual per producer: Python objects, closures, small LUTs, threads.
PREP_RESIDUAL_BYTES = 32 * 2 ** 20
#: Metadata is O(N): a small fixed base plus a per-frame conservative allowance.
_PREP_METADATA_BASE_BYTES = 2 * 2 ** 20
_PREP_METADATA_PER_FRAME_BYTES = 1024

#: One CFA global min/max pre-scan chunk is float64 (~8 B/elem) AND its float32
#: copy (~4 B/elem); the exact element count is the product decoder constant.
_CFA_PRESCAN_CHUNK_ELEMS = 4 << 20
CFA_PRESCAN_BYTES = _CFA_PRESCAN_CHUNK_ELEMS * (8 + 4)

_CFA_MARGIN_PX = 4


class PreparationAborted(RuntimeError):
    """Admission was closed (writer/producer failure or explicit stop)."""


class PreparationBudgetError(ExternalBudgetError):
    """A credit request or an admission cannot be covered by the budget."""


class PreparationQuarantineError(RuntimeError):
    """A producer survived the drain deadline: the provider set is unusable.

    The ORIGINAL failure is preserved in ``original`` and chained as ``__cause__``.
    ``retained_tile_dir`` names a payload deliberately NOT deleted because a live
    thread may still reference it; the Cell must fail, and no reuse is allowed.
    """

    def __init__(self, message, *, original, alive_producers, retained_tile_dir=None):
        super().__init__(message)
        self.original = original
        self.alive_producers = int(alive_producers)
        self.retained_tile_dir = (
            None if retained_tile_dir is None else str(retained_tile_dir)
        )


# ---------------------------------------------------------------------------
# Cell-scope CFA min/max registry (single-flight, immutable publication)
# ---------------------------------------------------------------------------

class CfaMinMaxRegistry:
    """Tiny (min, max) publication for CFA frames, safe across producers.

    Keys must bind the SOURCE identity and the decode policy (never a window):
    the value is two floats, never an image or a WCS. Concurrent calls for the
    same key are single-flight: exactly ONE full-frame pre-scan is paid, the
    others wait for its immutable ``(min, max)`` tuple. This is the only state
    shared by the per-producer providers; it preserves the statistics cost
    across provider destruction without sharing mutable pixels.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: dict = {}
        self._inflight: dict = {}
        self._stats = {
            "lookups": 0,
            "hits": 0,
            "misses": 0,
            "computed": 0,
            "waited": 0,
            "failed": 0,
        }

    def get(self, key):
        with self._lock:
            self._stats["lookups"] += 1
            value = self._values.get(key)
            if value is not None:
                self._stats["hits"] += 1
            return value

    def put(self, key, value) -> None:
        if value is None:
            raise ValueError("CFA min/max publication must not be None")
        frozen = (float(value[0]), float(value[1]))
        with self._lock:
            self._values[key] = frozen

    def get_or_compute(self, key, compute):
        """Return ``(value, computed_here)`` with single-flight semantics."""
        with self._lock:
            self._stats["lookups"] += 1
            value = self._values.get(key)
            if value is not None:
                self._stats["hits"] += 1
                return value, False
            event = self._inflight.get(key)
            if event is None:
                event = threading.Event()
                self._inflight[key] = event
                owner = True
            else:
                owner = False
                self._stats["waited"] += 1
        if owner:
            try:
                value = compute()
                if value is None:
                    raise RuntimeError("CFA pre-scan failed")
                frozen = (float(value[0]), float(value[1]))
            except BaseException:
                with self._lock:
                    self._stats["failed"] += 1
                    self._inflight.pop(key, None)
                event.set()
                raise
            with self._lock:
                self._values[key] = frozen
                self._stats["computed"] += 1
                self._stats["misses"] += 1
                self._inflight.pop(key, None)
            event.set()
            return frozen, True
        event.wait()
        with self._lock:
            value = self._values.get(key)
            if value is None:
                # The owner failed: recompute serially rather than fabricate.
                raise RuntimeError("CFA pre-scan owner failed without a value")
            self._stats["hits"] += 1
            return value, False

    def stats(self) -> dict:
        with self._lock:
            out = dict(self._stats)
            out["entries"] = len(self._values)
            return out

    def __len__(self) -> int:
        with self._lock:
            return len(self._values)


# ---------------------------------------------------------------------------
# Constructive preparation budget
# ---------------------------------------------------------------------------

def _require_positive(value, name):
    value = int(value)
    if value < 1:
        raise PreparationBudgetError(f"{name} must be >= 1, got {value}")
    return value


def preparation_grant_components(
    *,
    patch_hw,
    tile_hw,
    source_hw,
    channels: int,
    n_frames: int = 0,
    source_roi_hw=None,
    cfa_prescan: bool = True,
    full_frame_fallback: bool = True,
    provider_window: bool = True,
) -> dict:
    """Explicit HOST-RAM grant for ONE concurrent preparation.

    Every term is named and integer. ``peak_bytes`` is an ADMISSION construct
    (owned anonymous arrays), not a measured maximum.

    Domains are kept SEPARATE so admission cannot pretend an unknown is zero:

    * ``components`` — calculable owned terms (geometry-derived).
    * ``empirical_terms`` — bounded only by the declared majoration/residual
      (reprojection native scratch, allocator-retained heap). They are a
      documented empirical allowance, NOT a proved upper bound.
    * ``non_owned_terms`` — OS page cache / GC timing: reclaimable or outside
      the owned domain; never added to ``peak_bytes``.
    * ``unqualified_terms`` — inputs this function could not price
      (``source_roi_unknown``). Non-empty means ADMISSION MUST FALL BACK.

    The source section is priced from the REAL maximum ROI when supplied: a
    legitimate source rectangle can cover the whole frame, so assuming
    ``tile+margin`` would under-price dense/small-source cells.
    """
    ph, pw = int(patch_hw[0]), int(patch_hw[1])
    th, tw = int(tile_hw[0]), int(tile_hw[1])
    sh, sw = int(source_hw[0]), int(source_hw[1])
    c = _require_positive(channels, "channels")
    n = max(0, int(n_frames))
    if min(ph, pw, th, tw, sh, sw) < 1:
        raise PreparationBudgetError("preparation geometry must be positive")

    patch_px = ph * pw
    tile_px = th * tw
    source_px = sh * sw
    unqualified = []
    if source_roi_hw is None:
        roi_h, roi_w = sh, sw
        unqualified.append("source_roi_unknown")
    else:
        roi_h = min(sh, max(1, int(source_roi_hw[0])))
        roi_w = min(sw, max(1, int(source_roi_hw[1])))

    components = {}
    components["provider_window_bytes"] = (
        patch_px * (4 * c + 1) if provider_window else 0
    )
    # Whole-frame fallback: raw section + float64 intermediate + float32 HWC.
    components["full_frame_decode_bytes"] = (
        source_px * (3 * 4 + 8 + 4) if full_frame_fallback else 0
    )
    components["cfa_prescan_bytes"] = CFA_PRESCAN_BYTES if cfa_prescan else 0
    # Phase-aligned CFA section, its float64 transform and the debayer output.
    section_h = min(sh, roi_h + 2 * _CFA_MARGIN_PX + 2)
    section_w = min(sw, roi_w + 2 * _CFA_MARGIN_PX + 2)
    components["cfa_section_bytes"] = (
        section_h * section_w * (3 * 8 + 3 * 4)
    )
    # Source crop (C,H,W) float64 + coordinate maps + one channel work plane.
    components["reproject_source_bytes"] = section_h * section_w * c * 8
    components["reproject_coords_bytes"] = tile_px * (2 * 8 + 8)
    components["reproject_footprint_bytes"] = tile_px * 8
    components["reproject_output_bytes"] = tile_px * c * 4
    # Declared factor for the native interpolation workspace (NOT measured).
    components["reproject_native_scratch_bytes"] = (
        tile_px * (c * 8 + 2 * 8)
    )
    # Exact per-frame transform chain held by the producer.
    components["transformed_f64_bytes"] = tile_px * c * 8
    components["transformed_f32_bytes"] = tile_px * c * 4
    components["result_frame_bytes"] = tile_px * c * 4
    components["result_weights_bytes"] = tile_px * 4
    components["taper_scratch_bytes"] = tile_px * 8
    # Metadata is O(N) (frame ids/flags/closures), never a constant 4 MiB.
    components["metadata_cache_bytes"] = (
        _PREP_METADATA_BASE_BYTES + n * _PREP_METADATA_PER_FRAME_BYTES
    )

    subtotal = int(sum(components.values()))
    margin = (subtotal * PREP_MARGIN_NUM + PREP_MARGIN_DEN - 1) // PREP_MARGIN_DEN
    peak = subtotal + int(margin) + PREP_RESIDUAL_BYTES
    return {
        "schema": "zemosaic-m33-prep-grant-v2",
        "patch_hw": [ph, pw],
        "tile_hw": [th, tw],
        "source_hw": [sh, sw],
        "source_roi_hw": [int(roi_h), int(roi_w)],
        "n_frames": n,
        "channels": c,
        "components": {k: int(v) for k, v in components.items()},
        "components_subtotal_bytes": int(subtotal),
        # HONEST naming: the code applies subtotal + 1.25*subtotal + residual.
        "majoration_125pct_bytes": int(margin),
        "margin_bytes": int(margin),
        "peak_without_residual_over_subtotal": float(
            (subtotal + margin) / subtotal
        ),
        "peak_over_subtotal": float(peak / subtotal),
        "residual_bytes": int(PREP_RESIDUAL_BYTES),
        "peak_bytes": int(peak),
        "priced_domain": "owned_anonymous_constructive_admission",
        "empirical_terms": [
            "reproject_native_scratch_bytes",
            "allocator_retained_heap_after_join",
            # Delayed GC release retains OWNED arrays: it belongs to the
            # empirical allowance, not to the OS/non-owned domain.
            "python_gc_delayed_release",
        ],
        "non_owned_terms": [
            "filesystem_page_cache_residency",
        ],
        "unqualified_terms": list(unqualified),
        "empirical_allowance": (
            "majoration 125% + residual 32 MiB; declared empirical coverage, "
            "not a proved upper bound of native scratch/allocator retention"
        ),
        "notes": (
            "provider decoded-frame RAM cache is DISABLED for pipelined "
            "producers; source sections are re-read (CFA min/max is shared)"
        ),
    }


def decide_preparation_pipeline(
    *,
    requested_mode: str,
    available_bytes,
    producer_grant_bytes,
    existing_commitment_bytes=None,
    producers: int = 2,
    host_reserve_bytes: int = HOST_RESERVE_BYTES,
    unqualified_terms=(),
    empirical_terms=(),
) -> dict:
    """Resolve AUTO/legacy ONCE, from the constructive budget and real RAM.

    RSS alone never authorizes two producers. ``available_bytes`` must be a
    live net-available observation (MemAvailable/psutil). ``committed`` is
    mandatory: an unknown commitment is FAIL-CLOSED (legacy), never 0. Any
    ``unqualified_terms`` entry also forces legacy; empirical terms are
    published as a declared allowance, not as a proved bound.
    """
    mode = str(requested_mode or "auto").strip().lower()
    if mode not in PREP_PIPELINE_MODES:
        mode = "auto"
    producers = int(producers)
    grant = int(producer_grant_bytes)
    unqualified = [str(t) for t in (unqualified_terms or ()) if str(t)]
    empirical = [str(t) for t in (empirical_terms or ()) if str(t)]
    committed_known = existing_commitment_bytes is not None
    committed = int(existing_commitment_bytes or 0)
    single_required = grant + committed + int(host_reserve_bytes)
    two_required = producers * grant + committed + int(host_reserve_bytes)
    out = {
        "schema": "zemosaic-m33-prep-decision-v2",
        "requested_mode": mode,
        "producers_requested": producers,
        "producer_grant_bytes": grant,
        "existing_commitment_bytes": (
            committed if committed_known else None
        ),
        "host_reserve_bytes": int(host_reserve_bytes),
        "available_bytes": None if available_bytes is None else int(available_bytes),
        "single_required_bytes": int(single_required),
        "two_required_bytes": int(two_required),
        "single_admissible": None,
        "two_admissible": None,
        "effective_mode": "legacy",
        "effective_producers": 1,
        "reason": None,
        "unqualified_terms": unqualified,
        "empirical_terms": empirical,
        "qualified": False,
    }
    if mode == "legacy":
        out["reason"] = "requested_legacy_control"
        return out
    if not committed_known:
        out["reason"] = "committed_budget_unknown"
        return out
    if unqualified:
        out["reason"] = f"unqualified_budget_term:{unqualified[0]}"
        return out
    if available_bytes is None:
        out["reason"] = "host_available_unknown"
        return out
    out["qualified"] = True
    available = int(available_bytes)
    out["single_admissible"] = bool(available >= single_required)
    out["two_admissible"] = bool(available >= two_required)
    if not out["single_admissible"]:
        # The cell itself was admitted by the run-level model; this grant is
        # only the EXTRA concurrent working set. Never hide the shortfall.
        out["reason"] = "insufficient_host_budget_even_single"
        return out
    if not out["two_admissible"]:
        out["reason"] = "insufficient_host_budget_for_two_producers"
        return out
    out["effective_mode"] = "auto"
    out["effective_producers"] = producers
    out["reason"] = "constructive_budget_admissible"
    return out


# ---------------------------------------------------------------------------
# Byte-credit ledger
# ---------------------------------------------------------------------------

class ByteCreditLedger:
    """Bounded byte credits shared by producers and the single writer.

    A producer acquires ONE grant before it allocates anything. The credit is
    released only AFTER the writer stored the payload, so ready payloads and the
    writer's references are both covered. Capacity is ``producers * grant`` by
    construction; a grant larger than the whole capacity is refused up front.
    """

    def __init__(self, capacity_bytes: int) -> None:
        capacity = int(capacity_bytes)
        if capacity < 0:
            raise PreparationBudgetError("credit capacity must be >= 0")
        self.capacity_bytes = capacity
        self._cond = threading.Condition()
        self._used = 0
        self._peak = 0
        self._reservations = 0
        self._waits = 0
        self._wait_s = 0.0
        self._closed_reason = None

    @property
    def used_bytes(self) -> int:
        with self._cond:
            return int(self._used)

    @property
    def peak_bytes(self) -> int:
        with self._cond:
            return int(self._peak)

    @property
    def closed_reason(self):
        with self._cond:
            return self._closed_reason

    def acquire(self, n_bytes: int, *, stop_event=None) -> None:
        n = int(n_bytes)
        if n <= 0:
            raise PreparationBudgetError(f"credit must be > 0, got {n}")
        if n > self.capacity_bytes:
            raise PreparationBudgetError(
                f"credit request {n} B exceeds the whole capacity "
                f"{self.capacity_bytes} B (never max(1)-clipped)"
            )
        with self._cond:
            while self._used + n > self.capacity_bytes:
                if self._closed_reason is not None:
                    raise PreparationAborted(self._closed_reason)
                if stop_event is not None and stop_event.is_set():
                    raise PreparationAborted("stop requested")
                t0 = time.perf_counter()
                self._waits += 1
                self._cond.wait(timeout=0.1)
                self._wait_s += time.perf_counter() - t0
            if self._closed_reason is not None:
                raise PreparationAborted(self._closed_reason)
            if stop_event is not None and stop_event.is_set():
                raise PreparationAborted("stop requested")
            self._used += n
            self._peak = max(self._peak, self._used)
            self._reservations += 1

    def release(self, n_bytes: int) -> None:
        n = int(n_bytes)
        with self._cond:
            if n < 0 or n > self._used:
                raise PreparationBudgetError(
                    f"release {n} B outside [0, {self._used}]"
                )
            self._used -= n
            self._cond.notify_all()

    def close(self, reason: str) -> None:
        with self._cond:
            if self._closed_reason is None:
                self._closed_reason = str(reason)
            self._cond.notify_all()

    def stats(self) -> dict:
        with self._cond:
            return {
                "capacity_bytes": int(self.capacity_bytes),
                "peak_used_bytes": int(self._peak),
                "used_bytes": int(self._used),
                "reservations": int(self._reservations),
                "waits": int(self._waits),
                "wait_s": round(float(self._wait_s), 6),
                "closed_reason": self._closed_reason,
            }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@dataclass
class _Payload:
    index: int
    frame: object
    weights: object


_ITEM = "item"
_ERROR = "error"
_SENTINEL = "sentinel"


class PreparationPipeline:
    """Run the per-frame preparation of ONE tile with <= producers threads.

    The writer callback is invoked ONLY on the calling thread, in the original
    frame index order of completion, and receives the exact `(frame, weights)`
    payload built by :func:`prepare_external_frame`.
    """

    def __init__(
        self,
        *,
        provider_factory,
        p1,
        geometry,
        taper_kind,
        taper_px,
        taper_floor,
        grant_bytes: int,
        producers: int = 2,
        drain_timeout_s: float = 5.0,
        name: str = "prepare",
        progress_callback=None,
        progress_every: int = 0,
        cancel_event=None,
    ) -> None:
        if not callable(provider_factory):
            raise TypeError("provider_factory must be callable")
        self.factory = provider_factory
        self.p1 = p1
        self.ey0, self.ey1, self.ex0, self.ex1 = (int(v) for v in geometry)
        self.taper_kind = str(taper_kind)
        self.taper_px = float(taper_px)
        self.taper_floor = float(taper_floor)
        self.grant_bytes = int(grant_bytes)
        self.producers = int(producers)
        if self.producers < 1:
            raise PreparationBudgetError("producers must be >= 1")
        self.drain_timeout_s = float(drain_timeout_s)
        self.name = str(name)
        self.progress_callback = progress_callback
        self.progress_every = int(progress_every or 0)
        self.cancel_event = cancel_event
        self.ledger = ByteCreditLedger(self.grant_bytes * self.producers)
        self.stop_event = threading.Event()
        self._queue = _queue.Queue()
        self._failure = None
        self._failure_source = None
        self._quarantined = False
        self._quarantine_original = None
        self._retained_tile_dir = None
        self._alive_producers = 0
        self._consumed = False
        self._stop_reason = None
        self._seen = None
        self._stats = {
            "mode": "pipeline",
            "producers_requested": self.producers,
            "producers_effective": self.producers,
            "grant_bytes": self.grant_bytes,
            "prepared": 0,
            "written": 0,
            "dropped": 0,
            "max_inflight": 0,
            "provider_opens": 0,
            "provider_closes": 0,
            "provider_close_failures": 0,
            "drain_incomplete": False,
            "quarantined": False,
            "failure_source": None,
            "stop_reason": None,
        }

    # -- failure handling ---------------------------------------------------
    def _record_failure(self, exc: BaseException, source: str) -> None:
        if self._failure is None:
            self._failure = exc
            self._failure_source = str(source)
        self.stop_event.set()
        self.ledger.close(f"{source}_error")

    def request_stop(self, reason: str = "stop") -> None:
        self._stop_reason = str(reason)
        self.stop_event.set()
        self.ledger.close(str(reason))

    def _progress(self, force: bool = False) -> None:
        cb = self.progress_callback
        if cb is None:
            return
        written = int(self._stats["written"])
        every = int(self.progress_every or 0)
        if not force and every > 0 and written % every != 0:
            return
        try:
            cb({
                "phase": "write",
                "frame_current": written,
                "frame_total": int(self.p1.n_frames),
                "frame_prepared": int(self._stats["prepared"]),
                "bytes_reserved": int(self.ledger.used_bytes),
                "bytes_peak": int(self.ledger.peak_bytes),
                "producers_effective": int(self.producers),
            })
        except Exception:  # noqa: BLE001 - telemetry must never fail a tile
            pass

    def _producer(self, provider, take_next, pre_granted: bool = False) -> None:
        held = bool(pre_granted)
        try:
            while True:
                if self.stop_event.is_set():
                    break
                index = take_next()
                if index is None:
                    break
                granted = False
                try:
                    if held:
                        held = False
                        granted = True
                    else:
                        self.ledger.acquire(
                            self.grant_bytes, stop_event=self.stop_event
                        )
                        granted = True
                    frame, wrow = prepare_external_frame(
                        provider, self.p1, index,
                        self.ey0, self.ey1, self.ex0, self.ex1,
                        taper_kind=self.taper_kind,
                        taper_px=self.taper_px,
                        taper_floor=self.taper_floor,
                    )
                    if self.stop_event.is_set():
                        frame = wrow = None
                        del frame, wrow
                        break
                    self._queue.put(
                        (_ITEM, _Payload(int(index), frame, wrow),
                         self.grant_bytes)
                    )
                    granted = False  # ownership moved to the queue item
                    # B4: drop the producer's own references BEFORE the next
                    # acquire so the ledger really bounds live payloads.
                    frame = wrow = None
                    del frame, wrow
                except PreparationAborted:
                    break
                except BaseException as exc:  # noqa: BLE001 - preserved
                    self._record_failure(exc, "producer")
                    break
                finally:
                    if granted:
                        self.ledger.release(self.grant_bytes)
        finally:
            if held:
                self.ledger.release(self.grant_bytes)
            try:
                if hasattr(provider, "close"):
                    provider.close()
                    self._stats["provider_closes"] += 1
            except BaseException as exc:  # noqa: BLE001 - must fail closed
                self._stats["provider_close_failures"] += 1
                self._record_failure(exc, "provider_close")
            self._queue.put((_SENTINEL, None, 0))

    def _drain_until(self, deadline: float, done: int) -> int:
        """Consume queue items until every producer sent its sentinel."""
        while done < self.producers:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                kind, payload, credit = self._queue.get(timeout=remaining)
            except _queue.Empty:
                break
            if kind == _SENTINEL:
                done += 1
            elif kind == _ITEM:
                # Failure/stop after submission: never write. Clear the
                # payload BEFORE releasing its credit (B4).
                if payload is not None:
                    payload.frame = payload.weights = None
                del payload
                self.ledger.release(credit)
                self._stats["dropped"] += 1
        return done

    # -- run ----------------------------------------------------------------
    def run(self, write_frame) -> dict:
        n = int(self.p1.n_frames)
        # B5: an object is single-use. A quarantined or already-consumed
        # pipeline is refused BEFORE any factory/open/allocation.
        if self._consumed:
            raise PreparationAborted("preparation pipeline already consumed")
        self._consumed = True
        if self._quarantined:
            raise PreparationQuarantineError(
                "preparation pipeline is quarantined: no reuse",
                original=(
                    self._quarantine_original
                    or PreparationAborted("pipeline quarantined")
                ),
                alive_producers=self._alive_producers,
                retained_tile_dir=self._retained_tile_dir,
            )
        if self.stop_event.is_set():
            raise PreparationAborted(self._stop_reason or "stop requested")
        if self.cancel_event is not None and self.cancel_event.is_set():
            self.request_stop("cancelled")
            raise PreparationAborted("cancelled")
        self._seen = bytearray(n)
        providers = []
        threads = []
        try:
            for _ in range(self.producers):
                # B4: the provider-open cost is covered by a grant BEFORE the
                # factory allocates anything. The producer consumes this first
                # grant for its first frame, then acquires per refill.
                self.ledger.acquire(
                    self.grant_bytes, stop_event=self.stop_event
                )
                try:
                    provider = self.factory()
                except BaseException:
                    self.ledger.release(self.grant_bytes)
                    raise
                providers.append(provider)
                self._stats["provider_opens"] += 1
        except BaseException:
            for provider in providers:
                try:
                    self.ledger.release(self.grant_bytes)
                except PreparationBudgetError:
                    pass
                self._close_provider(provider)
            raise

        next_lock = threading.Lock()
        next_index = [0]

        def take_next():
            with next_lock:
                if next_index[0] >= n:
                    return None
                index = next_index[0]
                next_index[0] += 1
                return index

        started = []
        try:
            for provider in providers:
                thread = threading.Thread(
                    target=self._producer,
                    args=(provider, take_next, True),
                    name=f"{self.name}-producer",
                    daemon=True,
                )
                thread.start()
                started.append((provider, thread))
                threads.append(thread)
        except BaseException as exc:
            self._record_failure(exc, "producer_start")
            started_ids = {id(p) for p, _t in started}
            # Providers whose worker never started have no owner: close them
            # here. Started ones drain in their own finally.
            for provider in providers:
                if id(provider) not in started_ids:
                    try:
                        self.ledger.release(self.grant_bytes)
                    except PreparationBudgetError:
                        pass
                    self._close_provider(provider)
            deadline = time.perf_counter() + max(0.0, self.drain_timeout_s)
            for _provider, thread in started:
                thread.join(timeout=max(0.0, deadline - time.perf_counter()))
            alive = [t for t in threads if t.is_alive()]
            if alive:
                self._quarantined = True
                self._quarantine_original = self._failure
                self._alive_producers = len(alive)
                self._stats["drain_incomplete"] = True
                self._stats["quarantined"] = True
                # Do NOT touch credits/resources of a still-running producer:
                # they are retained with the quarantined provider set.
                raise PreparationQuarantineError(
                    "producer start failed and drain is incomplete: "
                    "provider set quarantined",
                    original=self._failure,
                    alive_producers=self._alive_producers,
                    retained_tile_dir=self._retained_tile_dir,
                ) from self._failure
            # Drain confirmed: consume every queued payload, clear its refs
            # BEFORE returning the credit, then publish the failure stats.
            drained_items = 0
            while True:
                try:
                    kind, payload, credit = self._queue.get_nowait()
                except _queue.Empty:
                    break
                drained_items += 1
                if payload is not None:
                    payload.frame = payload.weights = None
                    del payload
                if credit:
                    try:
                        self.ledger.release(credit)
                    except PreparationBudgetError:
                        pass
            self._stats["queue_items"] = 0
            self._stats["drained_after_start_failure"] = int(drained_items)
            self._stats["ledger"] = self.ledger.stats()
            raise

        # Normal operation waits for real work; stop/failure bounds the drain.
        drain_deadline = None
        done = 0
        try:
            while done < self.producers:
                if (
                    self.cancel_event is not None
                    and self.cancel_event.is_set()
                    and not self.stop_event.is_set()
                ):
                    self.request_stop("cancelled")
                if drain_deadline is None and (
                    self._failure is not None or self.stop_event.is_set()
                ):
                    drain_deadline = (
                        time.perf_counter() + max(0.0, self.drain_timeout_s)
                    )
                if drain_deadline is None:
                    try:
                        item = self._queue.get(timeout=0.1)
                    except _queue.Empty:
                        continue
                else:
                    remaining = drain_deadline - time.perf_counter()
                    if remaining <= 0:
                        break
                    try:
                        item = self._queue.get(timeout=remaining)
                    except _queue.Empty:
                        break
                kind, payload, credit = item
                if kind == _SENTINEL:
                    done += 1
                    continue
                if kind == _ERROR:
                    continue
                try:
                    if self._failure is not None or self.stop_event.is_set():
                        self._stats["dropped"] += 1
                        continue
                    index = int(payload.index)
                    if index < 0 or index >= n:
                        self._record_failure(
                            PreparationAborted(
                                f"writer received out-of-range index {index}"
                            ), "writer_index",
                        )
                        self._stats["dropped"] += 1
                        continue
                    if self._seen[index]:
                        self._record_failure(
                            PreparationAborted(
                                f"writer received duplicate index {index}"
                            ), "writer_duplicate",
                        )
                        self._stats["dropped"] += 1
                        continue
                    self._stats["prepared"] += 1
                    write_frame(index, payload.frame, payload.weights)
                    self._seen[index] = 1
                    self._stats["written"] += 1
                    self._stats["max_inflight"] = max(
                        self._stats["max_inflight"],
                        self.ledger.used_bytes,
                    )
                    self._progress()
                except BaseException as exc:  # noqa: BLE001 - preserved
                    self._record_failure(exc, "writer")
                    if drain_deadline is None:
                        drain_deadline = (
                            time.perf_counter()
                            + max(0.0, self.drain_timeout_s)
                        )
                finally:
                    # B4: clear the payload BEFORE releasing its byte credit.
                    payload.frame = payload.weights = None
                    del payload
                    self.ledger.release(credit)
            if drain_deadline is None and (
                self._failure is not None or self.stop_event.is_set()
            ):
                drain_deadline = (
                    time.perf_counter() + max(0.0, self.drain_timeout_s)
                )
            if done < self.producers and (
                self._failure is not None or self.stop_event.is_set()
            ):
                done = self._drain_until(drain_deadline, done)
        finally:
            self.ledger.close(
                self._failure_source
                if self._failure_source is not None
                else "run_finished"
            )

        if drain_deadline is None:
            for thread in threads:
                thread.join()
        else:
            for thread in threads:
                remaining = max(0.0, drain_deadline - time.perf_counter())
                thread.join(timeout=remaining)
        alive = [t for t in threads if t.is_alive()]
        if alive:
            self._quarantined = True
            self._quarantine_original = (
                self._failure
                or PreparationAborted(self._stop_reason or "stop requested")
            )
            self._alive_producers = len(alive)
            self._stats["drain_incomplete"] = True
            self._stats["quarantined"] = True
        self._stats["failure_source"] = self._failure_source
        self._stats["stop_reason"] = self._stop_reason
        self._stats["ledger"] = self.ledger.stats()
        self._stats["thread_sentinels"] = int(done)
        self._stats["written_indices"] = int(sum(self._seen))
        self._progress(force=True)
        if self._failure is not None:
            if self._quarantined:
                raise PreparationQuarantineError(
                    "preparation drain incomplete: provider set quarantined",
                    original=self._failure,
                    alive_producers=self._alive_producers,
                    retained_tile_dir=self._retained_tile_dir,
                ) from self._failure
            raise self._failure
        if self._quarantined:
            raise PreparationQuarantineError(
                "preparation drain incomplete: provider set quarantined",
                original=(
                    self._quarantine_original
                    or PreparationAborted("drain incomplete")
                ),
                alive_producers=self._alive_producers,
                retained_tile_dir=self._retained_tile_dir,
            )
        if self.stop_event.is_set():
            raise PreparationAborted(self._stop_reason or "stop requested")
        if done < self.producers:
            raise PreparationAborted("preparation producers did not report")
        # B1: COMPLETE control before any publication. Every frame index must
        # have been written exactly once, no drop, no shortfall.
        written = int(self._stats["written"])
        prepared = int(self._stats["prepared"])
        dropped = int(self._stats["dropped"])
        missing = n - int(sum(self._seen))
        if written != n or prepared != n or dropped != 0 or missing != 0:
            raise PreparationAborted(
                "incomplete preparation: "
                f"prepared={prepared}/{n} written={written}/{n} "
                f"dropped={dropped} missing={missing}"
            )
        return dict(self._stats)

    def _close_provider(self, provider) -> None:
        try:
            if hasattr(provider, "close"):
                provider.close()
                self._stats["provider_closes"] += 1
        except BaseException as exc:  # noqa: BLE001 - fail closed
            self._stats["provider_close_failures"] += 1
            self._record_failure(exc, "provider_close")

    @property
    def quarantined(self) -> bool:
        return bool(self._quarantined)


def write_tile_two_producers(
    spool,
    provider_factory,
    p1,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    *,
    grant_bytes: int,
    producers: int = 2,
    taper_kind: str,
    taper_px: float,
    taper_floor: float,
    halo_px: int,
    drain_timeout_s: float = 5.0,
    progress_callback=None,
    cancel_event=None,
):
    """Write ONE tile with ``producers`` private providers and ONE writer.

    Returns ``(TileSpool, stats)``. The tile directory is removed on a normal
    failure; it is deliberately RETAINED when a producer survived the drain
    deadline (a live thread may still reference its provider/mappings).
    """
    session = spool.open_tile_writer(
        y0, y1, x0, x1, halo_px=halo_px,
        n_frames=int(p1.n_frames), channels=int(p1.channels),
    )
    _n = int(p1.n_frames)
    pipeline = PreparationPipeline(
        provider_factory=provider_factory,
        p1=p1,
        geometry=(session.ey0, session.ey1, session.ex0, session.ex1),
        taper_kind=taper_kind,
        taper_px=taper_px,
        taper_floor=taper_floor,
        grant_bytes=grant_bytes,
        producers=producers,
        drain_timeout_s=drain_timeout_s,
        name=f"tile_y{session.y0}_x{session.x0}",
        progress_callback=progress_callback,
        progress_every=max(1, min(256, max(1, _n // 16))),
        cancel_event=cancel_event,
    )

    def _write(index: int, frame, wrow) -> None:
        # Use the SESSION's mappings so no caller/closure reference survives an
        # abort() (deterministic close before rmtree, Windows-safe).
        if frame is None:
            session.images[index] = np.nan
            session.weights[index] = 0.0
        else:
            session.images[index] = frame
            session.weights[index] = wrow

    try:
        stats = pipeline.run(_write)
        handle = session.publish()
        return handle, stats
    except PreparationQuarantineError as exc:
        # Do NOT delete a payload a non-drained producer might still touch.
        exc.retained_tile_dir = str(session.tile_dir)
        session.close_without_delete()
        raise
    except BaseException:
        session.abort()
        raise
