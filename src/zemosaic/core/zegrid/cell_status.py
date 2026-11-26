"""ZM-CELL-LIVE-OBS (lot 2) — bounded, truthful live per-cell status.

The problem this module solves is a VISIBILITY one, not a science one:

* a long CPU/external wave produced no per-cell evidence until the whole wave
  returned, so a cell could sit at 5 % or 95 % with no way to tell;
* the historical ``START`` line was emitted for the WHOLE batch BEFORE
  admission, so "started" never meant "a worker really began this cell";
* the per-cell ``phases.log`` breadcrumb is written by the worker but says
  nothing about the CURRENT tile / block, and a run log cannot be written from
  a child process safely.

This module is the small, deliberately boring seam that fixes that. It has two
halves:

* the WORKER half (:class:`LiveTelemetryWriter`) appends RATE-LIMITED, tiny
  JSON lines to a ROTATING set of per-cell segments inside the run's scratch
  cache root. There is no image payload, no array, no dtype and no science value
  in a line: only identifiers and small integers. There is no I/O per frame — a
  progress sample is coalesced and written at most once per ``min_interval_s``
  per cell (state transitions are always written; they are rare and bounded);
* the PARENT half (:class:`LiveTelemetryReader`, :class:`LiveCellStates` and
  :class:`LiveReporter`) reads the NEW bytes of the in-flight cells only, on the
  parent heartbeat (a caller thread/tick, never a new thread before a fork),
  validates the run/generation/cell identity and publishes a PERIODIC aggregate
  heartbeat plus change-only per-cell detail to the run log and the existing
  progress callback.

Design decisions (documented because the plan asked for an explicit choice):

* **Transport = append-only per-cell segments in the run scratch dir.** A
  ``multiprocessing.Queue``/``Manager`` would either need a helper process (a
  thread/process before the fork, or a daemon it cannot start) or would have to
  be pickled into ``ProcessPoolExecutor`` args, which is not spawn-safe. A file
  is: spawn-safe (a plain path travels in the initializer), fork-safe,
  Qt-safe (nothing but str/int crosses the boundary), and it SURVIVES a worker
  hang — exactly the diagnostic the product needed.
* **Unlimited lifetime with a bounded DISK footprint (never a finite quota).**
  A single capped file that stops writing (or wraps) would silently lose every
  later progress sample of a multi-hour cell. Instead the writer ROTATES through
  a bounded ring of segments: when the current segment reaches
  ``max_segment_bytes`` it opens the next one, retires EXACTLY the one segment
  that leaves the ring (O(1) work per rotation, O(rotations) in total — never a
  rescan of every historic path) and RE-EMITS the last state + last progress
  sample into the new segment. The stream therefore stays live indefinitely and
  a reader that starts late (or lags across a rotation) still recovers
  ``RUNNING``/phase/tile/block. A transient unlink failure is retried a BOUNDED
  number of times from a BOUNDED pending set; under a PERMANENT filesystem
  failure the disk bound is explicit BEST EFFORT (never promised) and the
  telemetry keeps flowing.
* **Lifetime/cleanup.** The parent TRUNCATES (removes) a cell's segments when it
  dispatches that cell in a wave, so a previous wave/run can never leak into the
  current one (a line-level identity check is kept as defence in depth). The
  parent removes the cell's segments once its commit is verified
  (commit-then-delete, the same rule as the spool) and removes the whole live
  directory best-effort at the end of the run. Files live under the run scratch
  cache root, never under a scientific output.
* **Rate limit / boundedness.** Per-cell progress writes are coalesced and
  limited to one per ``MIN_WRITE_INTERVAL_S`` (default 2 s). The parent reads at
  most ``MAX_READ_BYTES_PER_POLL`` per segment per heartbeat and never keeps more
  than the last event of each kind per cell. There is no unbounded event list
  anywhere. The periodic heartbeat is rate-limited separately
  (``LIVE_HEARTBEAT_INTERVAL_S``) and is DISTINCT from change-only detail.
* **Best effort, never science.** Every public method swallows its own errors:
  a telemetry failure can never fail a cell, a commit or the run.
* **Truthful states.** ``COMPUTED`` is emitted at result reception and
  ``COMMITTED`` ONLY after a verified durable commit; a running worker is never
  relabelled ``CANCELLED`` (the parent only reports ``CANCELLED`` for a future
  it really cancelled, i.e. one that never started). A heartbeat that sees no
  new worker observation reports its AGE; it never invents progress.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path

LIVE_STATUS_SCHEMA = "zemosaic-zegrid-live-status-v1"
LIVE_DIR_NAME = "zegrid-live-status"
LIVE_FILE_SUFFIX = ".jsonl"

#: Default per-cell progress write interval (coalescing window).
MIN_WRITE_INTERVAL_S = 2.0
#: One telemetry segment never grows past this before it rotates (hard bound).
MAX_SEGMENT_BYTES = 32 * 1024
#: Bounded ring of segments per cell (disk bound = segments * segment bytes).
MAX_SEGMENTS = 3
#: The parent never reads more than this from one segment on one heartbeat.
MAX_READ_BYTES_PER_POLL = 64 * 1024
#: Best-effort cap on a partial line kept across polls (defensive).
MAX_PARTIAL_BYTES = 8 * 1024
#: Default period of the OBLIGATORY parent heartbeat line (independent of change).
LIVE_HEARTBEAT_INTERVAL_S = 15.0
#: A transient unlink failure is retried at most this many times per segment...
RETIRE_MAX_ATTEMPTS = 3
#: ...and at most this much retirement work is done on one rotation.
RETIRE_RETRIES_PER_ROTATION = 2
#: Hard bound on the pending-retirement bookkeeping (never grows with rotations).
MAX_PENDING_RETIREMENTS = 8

QUEUED = "QUEUED"
SUBMITTED = "SUBMITTED"
RUNNING = "RUNNING"
COMPUTED = "COMPUTED"
COMMITTING = "COMMITTING"
COMMITTED = "COMMITTED"
FAILED = "FAILED"
CANCELLED = "CANCELLED"

#: The ordered states of a healthy cell. ``FAILED``/``CANCELLED`` are terminal
#: off this ladder (a cell can fail from any non-committed state).
STATE_ORDER = {
    QUEUED: 0,
    SUBMITTED: 1,
    RUNNING: 2,
    COMPUTED: 3,
    COMMITTING: 4,
    COMMITTED: 5,
}
TERMINAL_STATES = frozenset({COMMITTED, FAILED, CANCELLED})
ALL_STATES = (
    QUEUED, SUBMITTED, RUNNING, COMPUTED, COMMITTING, COMMITTED,
    FAILED, CANCELLED,
)

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
_SEGMENT_SUFFIX = re.compile(r"^\.(\d+)$")


def safe_cell_token(cell_id) -> str:
    """A filesystem-safe token for ONE cell id (never an empty string)."""
    token = _SAFE_NAME.sub("_", str(cell_id or "")).strip("._")
    return token or "cell"


def live_cell_path(live_dir, cell_id) -> Path:
    """The base (segment 0) telemetry path of ONE cell."""
    return Path(live_dir) / (safe_cell_token(cell_id) + LIVE_FILE_SUFFIX)


def live_segment_path(live_dir, cell_id, segment: int) -> Path:
    """Segment ``segment`` of ONE cell (0 is the base path, kept compatible)."""
    base = live_cell_path(live_dir, cell_id)
    return base if int(segment) <= 0 else Path(f"{base}.{int(segment)}")


# ---------------------------------------------------------------------------
# Worker half — writer (rate-limited + rotating, never a finite quota)
# ---------------------------------------------------------------------------


class LiveTelemetryWriter:
    """Rate-limited, coalescing, ROTATING, best-effort writer for ONE cell.

    The writer is created lazily in the WORKER process (or in the parent for a
    serial/thread path) and only ever appends small JSON lines. It holds no file
    handle between writes, so a parent-side removal is harmless.

    ``max_segment_bytes`` bounds ONE segment and ``max_segments`` bounds how many
    segments are kept: the stream rotates instead of stopping, so a multi-hour
    cell keeps being visible. The last state and the last progress sample are
    re-emitted into the fresh segment so a late reader still recovers
    RUNNING/phase/tile/block across a rotation.

    REWORK-2 (R4): each rotation retires exactly the ONE segment that leaves the
    ring, so retirement work and memory are bounded per rotation and linear in
    total. There is no historic-path rescan and no growing set of closed segment
    ids; a transient unlink failure is retried a bounded number of times from a
    bounded pending map, and a permanently failing filesystem is abandoned after
    a bounded number of attempts (``retire_given_up``) instead of retrying
    forever. Under such a permanent failure the DISK bound is best effort only.
    """

    def __init__(self, path, *, run_id, generation, cell_id,
                 min_interval_s: float = MIN_WRITE_INTERVAL_S,
                 max_segment_bytes: int = MAX_SEGMENT_BYTES,
                 max_segments: int = MAX_SEGMENTS,
                 clock=time.monotonic):
        self.path = Path(path)
        self.run_id = str(run_id)
        self.generation = int(generation)
        self.cell_id = str(cell_id)
        self.pid = int(os.getpid())
        self.min_interval_s = float(min_interval_s)
        self.max_segment_bytes = max(256, int(max_segment_bytes))
        self.max_segments = max(2, int(max_segments))
        self._clock = clock
        # ``enabled`` is kept for callers/tests that inspect it; a transient I/O
        # error no longer disables the stream permanently (that was the R1
        # defect), it is counted and the next write retries.
        self.enabled = True
        self.lines_written = 0
        self.progress_written = 0
        self.progress_suppressed = 0
        self.rotations = 0
        self.io_errors = 0
        # ZM-CELL-LIVE-OBS rework-2 (R4): retirement work is O(1) per rotation.
        # Exactly ONE segment falls out of the ring on each rotation; a transient
        # unlink failure is retried a BOUNDED number of times from a BOUNDED
        # pending set. Under a PERMANENT filesystem failure the disk bound is
        # best effort (explicitly not promised), never an unbounded retry storm.
        self.unlink_attempts = 0
        self.segments_retired = 0
        self.retire_failures = 0
        self.retire_given_up = 0
        self._retire_pending: dict[int, int] = {}
        self._segment = 0
        self._seg_bytes = 0
        self._last_progress_ts = 0.0
        self._last_progress_key = None
        self._last_state_event = None
        self._last_progress_event = None

    # -- internals ---------------------------------------------------------
    def _segment_path(self, segment: int) -> Path:
        return (
            self.path if int(segment) <= 0
            else Path(f"{self.path}.{int(segment)}")
        )

    def _base(self, kind: str) -> dict:
        return {
            "schema": LIVE_STATUS_SCHEMA,
            "kind": str(kind),
            "run_id": self.run_id,
            "gen": self.generation,
            "cell_id": self.cell_id,
            "pid": self.pid,
            "t": time.time(),
        }

    def _append_raw(self, payload: dict) -> bool:
        """Append ONE already-built payload to the CURRENT segment (no limits)."""
        try:
            line = json.dumps(payload, separators=(",", ":"), sort_keys=True)
            data = len(line.encode("utf-8")) + 1
        except Exception:  # noqa: BLE001 - never fatal
            return False
        try:
            with self._segment_path(self._segment).open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(line + "\n")
                handle.flush()
        except Exception:  # noqa: BLE001 - telemetry must never fail a cell
            self.io_errors += 1
            return False
        self._seg_bytes += data
        self.lines_written += 1
        return True

    def _retire_segment(self, segment: int) -> bool:
        """Try to remove ONE retired segment (bounded bookkeeping, no storm).

        Returns True when the file is gone (removed now or already absent). A
        transient failure is recorded in a BOUNDED pending map and retried by a
        bounded retry pass (possibly in the same rotation); after ``RETIRE_MAX_ATTEMPTS`` the segment is abandoned
        (``retire_given_up``), so a permanently failing filesystem can never
        turn into unbounded work or unbounded memory.
        """
        if int(segment) < 0:
            return True
        segment = int(segment)
        self.unlink_attempts += 1
        try:
            self._segment_path(segment).unlink()
        except FileNotFoundError:
            # Already gone (a previous attempt succeeded, or a reader/cleanup
            # removed it): a success, never a retry.
            self._retire_pending.pop(segment, None)
            self.segments_retired += 1
            return True
        except Exception:  # noqa: BLE001 - telemetry must never fail a cell
            attempts = int(self._retire_pending.get(segment, 0)) + 1
            self.retire_failures += 1
            if attempts >= RETIRE_MAX_ATTEMPTS:
                self._retire_pending.pop(segment, None)
                self.retire_given_up += 1
            else:
                self._retire_pending[segment] = attempts
                self._bound_pending()
            return False
        self._retire_pending.pop(segment, None)
        self.segments_retired += 1
        return True

    def _bound_pending(self) -> None:
        """Never let the retirement bookkeeping grow with the rotation count."""
        while len(self._retire_pending) > MAX_PENDING_RETIREMENTS:
            oldest = next(iter(self._retire_pending))
            self._retire_pending.pop(oldest, None)
            self.retire_given_up += 1

    def _rotate(self) -> None:
        """Open the next segment, retire EXACTLY ONE and re-emit the latest state."""
        self._segment += 1
        self._seg_bytes = 0
        self.rotations += 1
        # The ring keeps segments [segment - max_segments + 1 .. segment], so
        # exactly one segment leaves it on each rotation: retirement is O(1) per
        # rotation and O(rotations) in total (it used to rescan every historic
        # path on every rotation: O(rotations^2)).
        self._retire_segment(self._segment - self.max_segments)
        # Bounded retry of previously failed retirements (never a full rescan).
        for pending in list(self._retire_pending)[:RETIRE_RETRIES_PER_ROTATION]:
            self._retire_segment(pending)
        # A reader that starts (or resumes) at the new segment must still know
        # this cell is RUNNING and where it is: re-emit the last state + the last
        # progress sample. These are two tiny lines, so the ring stays bounded.
        for payload in (self._last_state_event, self._last_progress_event):
            if payload is not None:
                self._append_raw(payload)

    def _write(self, payload: dict) -> bool:
        if not self.enabled:
            return False
        try:
            size = len(
                json.dumps(payload, separators=(",", ":"), sort_keys=True)
            ) + 1
        except Exception:  # noqa: BLE001 - never fatal
            return False
        if self._seg_bytes > 0 and (
            self._seg_bytes + size > self.max_segment_bytes
        ):
            self._rotate()
        return self._append_raw(payload)

    # -- public API --------------------------------------------------------
    def state(self, state: str, *, note: str | None = None) -> bool:
        """Write a state TRANSITION (always, they are rare and bounded)."""
        token = str(state).upper()
        if token not in ALL_STATES:
            return False
        payload = self._base("STATE")
        payload["state"] = token
        if note:
            payload["note"] = str(note)[:160]
        if not self._write(payload):
            return False
        self._last_state_event = payload
        return True

    def progress(self, phase: str, *, tile_current=None, tile_total=None,
                 block_current=None, block_total=None, elapsed_s=None,
                 frame_current=None, frame_total=None, frame_prepared=None,
                 bytes_reserved=None, bytes_peak=None,
                 producers_effective=None, force: bool = False) -> bool:
        """Write ONE coalesced progress sample (no image, no dtype, no array)."""
        now = self._clock()
        key = (
            str(phase),
            None if tile_current is None else int(tile_current),
            None if tile_total is None else int(tile_total),
            None if block_current is None else int(block_current),
            None if block_total is None else int(block_total),
            None if frame_current is None else int(frame_current),
            None if frame_prepared is None else int(frame_prepared),
            None if bytes_reserved is None else int(bytes_reserved),
            None if bytes_peak is None else int(bytes_peak),
            None if producers_effective is None else int(producers_effective),
        )
        if not force:
            if key == self._last_progress_key:
                # Identical sample: never repeat it.
                self.progress_suppressed += 1
                return False
            if (now - self._last_progress_ts) < self.min_interval_s:
                # Inside the coalescing window: the next sample after the window
                # is written. Phase/tile/block transitions use force=True at
                # their explicit boundaries, so nothing important is lost.
                self.progress_suppressed += 1
                return False
        payload = self._base("PROGRESS")
        payload["phase"] = str(phase)
        if tile_current is not None:
            payload["tile"] = [int(tile_current), int(tile_total or 0)]
        if block_current is not None:
            payload["block"] = [int(block_current), int(block_total or 0)]
        if frame_current is not None:
            payload["frame"] = [int(frame_current), int(frame_total or 0)]
        if frame_current is not None:
            payload["frame_current"] = int(frame_current)
        if frame_total is not None:
            payload["frame_total"] = int(frame_total)
        if frame_prepared is not None:
            payload["frame_prepared"] = int(frame_prepared)
        if bytes_reserved is not None:
            payload["bytes_reserved"] = int(bytes_reserved)
        if bytes_peak is not None:
            payload["bytes_peak"] = int(bytes_peak)
        if producers_effective is not None:
            payload["producers_effective"] = int(producers_effective)
        if elapsed_s is not None:
            payload["elapsed_s"] = round(float(elapsed_s), 3)
        if not self._write(payload):
            return False
        self._last_progress_ts = now
        self._last_progress_key = key
        self._last_progress_event = payload
        self.progress_written += 1
        return True


# ---------------------------------------------------------------------------
# Parent half — state registry
# ---------------------------------------------------------------------------


class LiveCellStates:
    """Run-wide, UNIQUE per-cell state bookkeeping (the parent's only truth).

    Every cell of the frozen plan appears EXACTLY once, so the published counts
    always add up to the plan total. Resumed (already durable at run start) and
    newly committed cells are tracked separately and never double counted.

    Thread-safe: the serial path samples the heartbeat from a monitor thread
    while the parent loop keeps consuming results. The lock is uncontended in
    the pool paths.
    """

    def __init__(self, clock=time.monotonic) -> None:
        self._lock = threading.Lock()
        self._clock = clock
        self._states: dict[str, str] = {}
        self._detail: dict[str, dict] = {}
        self._committed_resumed: set[str] = set()

    def reset(self, cell_ids, *, committed=()) -> None:
        """Initialise the plan: everything QUEUED except already-durable cells."""
        cells = [str(cid) for cid in cell_ids]
        resumed = {str(cid) for cid in committed}
        with self._lock:
            self._states = {}
            self._detail = {}
            self._committed_resumed = set()
            for cid in cells:
                if cid in resumed:
                    self._states[cid] = COMMITTED
                    self._committed_resumed.add(cid)
                else:
                    self._states[cid] = QUEUED

    def is_known(self, cell_id) -> bool:
        with self._lock:
            return str(cell_id) in self._states

    def update(self, cell_id, state, *, detail: dict | None = None,
               observed_at=None) -> bool:
        """Advance ONE cell's state. Returns True when it really changed.

        Monotonic: a strictly OLDER state is ignored (a late/stale event can
        never rewind a committed cell). ``FAILED``/``CANCELLED`` are accepted
        from any non-COMMITTED state and are then terminal.
        """
        cid = str(cell_id)
        token = str(state).upper()
        if token not in ALL_STATES:
            return False
        with self._lock:
            current = self._states.get(cid)
            if current is None:
                # A cell outside this run's frozen plan is never invented.
                return False
            if current in TERMINAL_STATES:
                return False
            if token in (FAILED, CANCELLED):
                self._states[cid] = token
                if detail:
                    self._detail[cid] = dict(detail)
                return True
            cur_order = STATE_ORDER.get(current, -1)
            new_order = STATE_ORDER.get(token, -1)
            if new_order < 0 or new_order <= cur_order:
                return False
            self._states[cid] = token
            if detail:
                self._detail[cid] = dict(detail)
            if observed_at is not None:
                self._detail.setdefault(cid, {})["observed_at"] = float(
                    observed_at
                )
            return True

    def note_progress(self, cell_id, *, phase=None, tile_current=None,
                      tile_total=None, block_current=None, block_total=None,
                      frame_current=None, frame_total=None,
                      frame_prepared=None, bytes_reserved=None,
                      bytes_peak=None, producers_effective=None,
                      elapsed_s=None, observed_at=None) -> bool:
        """Attach the latest PROGRESS facts without changing the state.

        R3 (rework-1): a progress sample may be PARTIAL (a block sample carries
        no tile). Such a sample must never ERASE the current tile, and a block
        from the PREVIOUS tile/phase must never survive a transition:
        ``tile``/``block`` are deep-merged, and the block is cleared whenever the
        phase or the tile really changes.
        """
        cid = str(cell_id)
        # Counters that describe the CURRENT write phase/sample. They must not
        # survive a phase or tile transition (a reduce phase has no producer,
        # and a sticky-fallback tile runs legacy with ONE producer).
        progress_keys = (
            "frame", "frame_current", "frame_total", "frame_prepared",
            "bytes_reserved", "bytes_peak", "producers_effective",
        )
        with self._lock:
            current = self._states.get(cid)
            if current in (None, QUEUED, COMMITTED, FAILED, CANCELLED):
                return False
            entry = self._detail.setdefault(cid, {})
            phase_changed = False
            if phase is not None:
                new_phase = str(phase)
                phase_changed = entry.get("phase") != new_phase
                if phase_changed:
                    # A new phase invalidates the previous block counter and
                    # every per-write progress counter.
                    entry["block"] = None
                    for key in progress_keys:
                        entry.pop(key, None)
                entry["phase"] = new_phase
            tile_changed = False
            if tile_current is not None:
                new_tile = [int(tile_current), int(tile_total or 0)]
                tile_changed = entry.get("tile") != new_tile
                if tile_changed:
                    # A new tile invalidates the previous tile's block counter
                    # and every per-write progress counter.
                    entry["block"] = None
                    for key in progress_keys:
                        entry.pop(key, None)
                entry["tile"] = new_tile
            if block_current is not None:
                entry["block"] = [int(block_current), int(block_total or 0)]
            if frame_current is not None:
                entry["frame"] = [int(frame_current), int(frame_total or 0)]
                entry["frame_current"] = int(frame_current)
                if frame_total is not None:
                    entry["frame_total"] = int(frame_total)
            if frame_prepared is not None:
                entry["frame_prepared"] = int(frame_prepared)
            if bytes_reserved is not None:
                entry["bytes_reserved"] = int(bytes_reserved)
            if bytes_peak is not None:
                entry["bytes_peak"] = int(bytes_peak)
            if producers_effective is not None:
                entry["producers_effective"] = int(producers_effective)
            if elapsed_s is not None:
                entry["elapsed_s"] = float(elapsed_s)
            if observed_at is not None:
                entry["observed_at"] = float(observed_at)
            return True

    def snapshot(self, now=None) -> dict:
        """Aggregate counts + a compact per-cell row list (bounded by the plan)."""
        stamp = float(self._clock() if now is None else now)
        with self._lock:
            counts = {state: 0 for state in ALL_STATES}
            for state in self._states.values():
                counts[state] = counts.get(state, 0) + 1
            rows = []
            newest_observation = None
            for cid, state in self._states.items():
                if state in (QUEUED, COMMITTED):
                    continue
                entry = dict(self._detail.get(cid) or {})
                observed_at = entry.pop("observed_at", None)
                entry["cell_id"] = cid
                entry["state"] = state
                if observed_at is not None:
                    age = max(0.0, stamp - float(observed_at))
                    entry["observed_age_s"] = round(age, 1)
                    if state not in TERMINAL_STATES and (
                        newest_observation is None or age < newest_observation
                    ):
                        newest_observation = age
                rows.append(entry)
            resumed = len(self._committed_resumed)
            total = len(self._states)
            new_commits = counts.get(COMMITTED, 0) - resumed
            return {
                "schema": LIVE_STATUS_SCHEMA,
                "total": int(total),
                "committed": int(counts.get(COMMITTED, 0)),
                "resumed_committed": int(resumed),
                "new_committed": int(max(0, new_commits)),
                "queued": int(counts.get(QUEUED, 0)),
                "submitted": int(counts.get(SUBMITTED, 0)),
                "running": int(counts.get(RUNNING, 0)),
                "computed": int(counts.get(COMPUTED, 0)),
                "committing": int(counts.get(COMMITTING, 0)),
                "failed": int(counts.get(FAILED, 0)),
                "cancelled": int(counts.get(CANCELLED, 0)),
                "active": int(
                    counts.get(RUNNING, 0) + counts.get(SUBMITTED, 0)
                ),
                "pending": int(counts.get(QUEUED, 0)),
                # Age of the MOST RECENT worker observation among the active
                # cells (never an invented progress). None when nothing is
                # active.
                "last_observation_age_s": (
                    None if newest_observation is None
                    else round(float(newest_observation), 1)
                ),
                "rows": sorted(rows, key=lambda row: str(row["cell_id"])),
            }


def live_status_line(snapshot: dict, *, generation=None,
                     phase=None) -> str:
    """ONE aggregate live line for the run log and the GUI progress log."""
    parts = [
        f"ZeGrid: cells live — total={int(snapshot['total'])}",
        f"committed={int(snapshot['committed'])}",
        f"(resumed={int(snapshot['resumed_committed'])} "
        f"new={int(snapshot['new_committed'])})",
        f"running={int(snapshot['running'])}",
        f"submitted={int(snapshot['submitted'])}",
        f"queued={int(snapshot['queued'])}",
        f"computed={int(snapshot['computed'])}",
        f"committing={int(snapshot['committing'])}",
        f"failed={int(snapshot['failed'])}",
        f"cancelled={int(snapshot['cancelled'])}",
    ]
    age = snapshot.get("last_observation_age_s")
    if age is not None:
        parts.append(f"last_worker_observation_age_s={float(age):.1f}")
    if generation is not None:
        parts.append(f"gen={int(generation)}")
    if phase is not None:
        parts.append(f"phase={phase}")
    return " ".join(parts)


def live_cell_line(row: dict) -> str:
    """ONE per-cell live line (phase / tile / block), scalars only."""
    parts = [
        f"ZeGrid: cell {row.get('cell_id')} {row.get('state')}",
    ]
    phase = row.get("phase")
    if phase:
        parts.append(f"phase={phase}")
    tile = row.get("tile")
    if tile:
        parts.append(f"tile={int(tile[0])}/{int(tile[1])}")
    block = row.get("block")
    if block:
        parts.append(f"block={int(block[0])}/{int(block[1])}")
    frame = row.get("frame")
    if frame:
        parts.append(f"frame={int(frame[0])}/{int(frame[1])}")
    prepared = row.get("frame_prepared")
    if prepared is not None:
        parts.append(f"prepared={int(prepared)}")
    reserved = row.get("bytes_reserved")
    if reserved is not None:
        parts.append(f"bytes_reserved={int(reserved)}")
    producers = row.get("producers_effective")
    if producers is not None:
        parts.append(f"producers={int(producers)}")
    elapsed = row.get("elapsed_s")
    if elapsed is not None:
        parts.append(f"elapsed_s={float(elapsed):.1f}")
    age = row.get("observed_age_s")
    if age is not None:
        parts.append(f"observed_age_s={float(age):.1f}")
    return " ".join(parts)


def transitions_of(previous: dict | None, snapshot: dict) -> list[dict]:
    """Per-cell transitions since the previous snapshot (visible, no spam)."""
    if previous is None:
        return []
    old = {
        str(row["cell_id"]): str(row["state"])
        for row in (previous.get("rows") or [])
    }
    out = []
    for row in snapshot.get("rows") or []:
        cid = str(row["cell_id"])
        if old.get(cid) != str(row["state"]):
            out.append(dict(row))
    return out


# ---------------------------------------------------------------------------
# Parent half — reader (rotation-aware, incremental, identity-checked)
# ---------------------------------------------------------------------------


class LiveTelemetryReader:
    """Incremental, identity-checked, ROTATION-AWARE reader of the live files.

    Only the files of the cells the caller names are touched (a wave is bounded
    by the dispatch cap), and only the NEW bytes since the last poll are read. A
    line whose ``run_id``/``gen``/``cell_id`` does not match the CURRENT
    dispatch is rejected as stale (a leftover file, or a pre-recovery attempt).
    Segments that vanished during a rotation are simply dropped; the writer
    re-emits the latest state/progress into the new segment, so the reader still
    recovers RUNNING/phase/tile/block.
    """

    def __init__(self, live_dir, *, run_id, generation,
                 max_read_bytes: int = MAX_READ_BYTES_PER_POLL) -> None:
        self.live_dir = Path(live_dir)
        self.run_id = str(run_id)
        self.generation = int(generation)
        self.max_read_bytes = int(max_read_bytes)
        self._offsets: dict[tuple[str, int], int] = {}
        self._partial: dict[tuple[str, int], bytes] = {}
        self._lock = threading.Lock()
        self.rejected_stale = 0
        self.events_read = 0

    # -- internals ---------------------------------------------------------
    def _scan(self) -> dict[str, list[tuple[int, Path]]]:
        """ONE directory listing -> ``{cell_token: [(segment, path), ...]}``.

        Scanning once per poll (instead of once per cell) keeps the parent cost
        proportional to the number of in-flight files, never to cells squared.
        """
        found: dict[str, list[tuple[int, Path]]] = {}
        try:
            entries = list(self.live_dir.iterdir())
        except Exception:  # noqa: BLE001 - telemetry only
            return found
        for entry in entries:
            name = entry.name
            if not name.endswith(LIVE_FILE_SUFFIX) and LIVE_FILE_SUFFIX not in name:
                continue
            base_name, sep, suffix = name.rpartition(LIVE_FILE_SUFFIX)
            if not sep:
                continue
            segment = 0
            if suffix:
                match = _SEGMENT_SUFFIX.match(suffix)
                if not match:
                    continue
                segment = int(match.group(1))
            found.setdefault(base_name, []).append((segment, entry))
        for rows in found.values():
            rows.sort(key=lambda item: item[0])
        return found

    def _remove_cell_files(self, cell_id) -> None:
        base = live_cell_path(self.live_dir, cell_id)
        try:
            base.unlink()
        except Exception:  # noqa: BLE001 - telemetry only
            pass
        prefix = base.name + "."
        try:
            for entry in self.live_dir.iterdir():
                if (
                    entry.name.startswith(prefix)
                    and _SEGMENT_SUFFIX.match(entry.name[len(base.name):])
                ):
                    try:
                        entry.unlink()
                    except Exception:  # noqa: BLE001 - telemetry only
                        pass
        except Exception:  # noqa: BLE001 - telemetry only
            pass

    def _valid(self, payload: dict, cell_id: str) -> bool:
        if not isinstance(payload, dict):
            return False
        if payload.get("schema") != LIVE_STATUS_SCHEMA:
            return False
        if str(payload.get("run_id")) != self.run_id:
            return False
        if str(payload.get("cell_id")) != cell_id:
            return False
        try:
            if int(payload.get("gen", -1)) != int(self.generation):
                return False
        except (TypeError, ValueError):
            return False
        return True

    # -- public API --------------------------------------------------------
    def retarget(self, *, generation) -> None:
        """Move to a new dispatch generation (offsets are kept per segment)."""
        with self._lock:
            self.generation = int(generation)

    def truncate(self, cell_ids) -> None:
        """Remove the segments of the cells about to be dispatched."""
        with self._lock:
            for cid in cell_ids:
                key = str(cid)
                try:
                    self.live_dir.mkdir(parents=True, exist_ok=True)
                except Exception:  # noqa: BLE001 - telemetry only
                    pass
                self._remove_cell_files(key)
                for (cell, segment) in list(self._offsets):
                    if cell == key:
                        self._offsets.pop((cell, segment), None)
                for (cell, segment) in list(self._partial):
                    if cell == key:
                        self._partial.pop((cell, segment), None)

    def discard(self, cell_ids) -> None:
        """Remove the files of committed cells (commit-then-delete)."""
        with self._lock:
            for cid in cell_ids:
                key = str(cid)
                self._remove_cell_files(key)
                for (cell, segment) in list(self._offsets):
                    if cell == key:
                        self._offsets.pop((cell, segment), None)
                for (cell, segment) in list(self._partial):
                    if cell == key:
                        self._partial.pop((cell, segment), None)

    def poll(self, cell_ids) -> dict:
        """Return ``{cell_id: {"state": ev|None, "progress": ev|None}}``.

        ONE poll may contain several lines of the same cell (a state transition
        followed by progress samples, possibly across a rotation). Keeping only
        the LAST line would LOSE the RUNNING transition, so the last event of
        EACH kind is returned: the parent applies the transition and the newest
        progress together.
        """
        wanted = [str(cid) for cid in cell_ids]
        latest: dict[str, dict] = {}
        with self._lock:
            scanned = self._scan()
            for cid in wanted:
                segments = scanned.get(safe_cell_token(cid), [])
                if not segments:
                    continue
                present = {segment for segment, _path in segments}
                for (cell, segment) in list(self._offsets):
                    if cell == cid and segment not in present:
                        self._offsets.pop((cell, segment), None)
                        self._partial.pop((cell, segment), None)
                state_event = None
                progress_event = None
                for segment, path in segments:
                    key = (cid, segment)
                    try:
                        size = path.stat().st_size
                    except Exception:  # noqa: BLE001 - rotated away
                        continue
                    offset = int(self._offsets.get(key, 0))
                    if size < offset:
                        offset = 0
                        self._partial.pop(key, None)
                    if size == offset:
                        continue
                    try:
                        with path.open("rb") as handle:
                            handle.seek(offset)
                            blob = handle.read(
                                min(self.max_read_bytes, size - offset)
                            )
                    except Exception:  # noqa: BLE001 - telemetry only
                        continue
                    self._offsets[key] = offset + len(blob)
                    buffered = self._partial.get(key, b"") + blob
                    lines = buffered.split(b"\n")
                    tail = lines.pop() if lines else b""
                    if len(tail) <= MAX_PARTIAL_BYTES:
                        self._partial[key] = tail
                    else:  # pragma: no cover - defensive
                        self._partial.pop(key, None)
                    for raw in lines:
                        raw = raw.strip()
                        if not raw:
                            continue
                        try:
                            payload = json.loads(raw.decode("utf-8"))
                        except Exception:  # noqa: BLE001 - torn/foreign line
                            # An interrupted write can glue a torn prefix to the
                            # NEXT valid line. Recover the last complete object
                            # (bounded: one rfind + one parse) instead of losing
                            # the live sample that follows.
                            cut = raw.rfind(b"{")
                            if cut <= 0:
                                continue
                            try:
                                payload = json.loads(raw[cut:].decode("utf-8"))
                            except Exception:  # noqa: BLE001 - really torn
                                continue
                        if not self._valid(payload, cid):
                            self.rejected_stale += 1
                            continue
                        self.events_read += 1
                        if str(payload.get("kind", "")).upper() == "STATE":
                            state_event = payload
                        else:
                            progress_event = payload
                if state_event is not None or progress_event is not None:
                    latest[cid] = {
                        "state": state_event,
                        "progress": progress_event,
                    }
        return latest


def apply_events(states: LiveCellStates, events: dict, *, now=None) -> list[dict]:
    """Fold reader events into the state registry; return changed rows."""
    changed: list[dict] = []
    stamp = None if now is None else float(now)
    for cid, bundle in events.items():
        if not isinstance(bundle, dict):
            continue
        event = bundle.get("state")
        if isinstance(event, dict):
            state = str(event.get("state", "")).upper()
            if state == RUNNING:
                # A real worker start is the ONLY source of RUNNING.
                if states.update(cid, state, observed_at=stamp):
                    changed.append({"cell_id": cid, "state": state})
            elif state in (FAILED, CANCELLED):
                if states.update(cid, state):
                    changed.append({"cell_id": cid, "state": state})
        event = bundle.get("progress")
        if isinstance(event, dict):
            tile = event.get("tile") or None
            block = event.get("block") or None
            frame = event.get("frame") or None
            states.note_progress(
                cid,
                phase=event.get("phase"),
                # R3: a block sample carries its OWN tile, so the tile is never
                # erased and the block is scoped to the right tile.
                tile_current=(tile[0] if tile else None),
                tile_total=(tile[1] if tile else None),
                block_current=(block[0] if block else None),
                block_total=(block[1] if block else None),
                frame_current=(
                    event.get("frame_current")
                    if event.get("frame_current") is not None
                    else (frame[0] if frame else None)
                ),
                frame_total=(
                    event.get("frame_total")
                    if event.get("frame_total") is not None
                    else (frame[1] if frame else None)
                ),
                frame_prepared=event.get("frame_prepared"),
                bytes_reserved=event.get("bytes_reserved"),
                bytes_peak=event.get("bytes_peak"),
                producers_effective=event.get("producers_effective"),
                elapsed_s=event.get("elapsed_s"),
                observed_at=stamp,
            )
    return changed


# ---------------------------------------------------------------------------
# Parent half — reporter (periodic heartbeat + change-only detail)
# ---------------------------------------------------------------------------


class LiveReporter:
    """Sample worker telemetry and publish it with TWO distinct cadences.

    * the AGGREGATE heartbeat is written PERIODICALLY (``interval_s``) even when
      every counter is identical, so a blocked worker or a long micro-block can
      never look like silence. It carries the age of the last worker
      observation; it never invents progress;
    * per-cell DETAIL is written only when a visible fact changes (state, phase,
      tile, block).

    GUI delivery is OPTIONAL per call (``deliver_gui``) and is serialized with
    the journal writes: the serial-path monitor thread can publish the JOURNAL
    heartbeat safely while the GUI callback stays on the owning thread.
    """

    def __init__(self, states: LiveCellStates, reader: LiveTelemetryReader | None,
                 *, emit_line, gui_emit=None,
                 interval_s: float = LIVE_HEARTBEAT_INTERVAL_S,
                 generation: int = 0, clock=time.monotonic):
        self.states = states
        self.reader = reader
        self.emit_line = emit_line
        self.gui_emit = gui_emit
        self.interval_s = float(interval_s)
        self.generation = int(generation)
        self._clock = clock
        self._lock = threading.RLock()
        self._last_line = None
        self._last_signature = None
        self._last_emit_ts = None
        self._last_rows: dict[str, tuple] = {}
        self.heartbeats = 0
        self.detail_lines = 0
        self.samples = 0

    def retarget(self, *, generation) -> None:
        with self._lock:
            self.generation = int(generation)

    def sample(self) -> None:
        """Poll the worker files and fold them into the state registry."""
        if self.reader is None:
            return
        with self._lock:
            try:
                snapshot = self.states.snapshot()
                watch = [
                    str(row["cell_id"])
                    for row in snapshot.get("rows") or ()
                    if str(row.get("state")) not in (
                        QUEUED, COMMITTED, FAILED, CANCELLED
                    )
                ]
                if not watch:
                    return
                events = self.reader.poll(watch)
                now = self._clock()
                apply_events(self.states, events, now=now)
                self.samples += 1
            except Exception:  # noqa: BLE001 - telemetry is never fatal
                return

    def publish(self, *, force: bool = False, deliver_gui: bool = True) -> bool:
        """Write ONE aggregate heartbeat (periodic/forced) + changed detail."""
        with self._lock:
            try:
                now = self._clock()
                snapshot = self.states.snapshot(now=now)
                line = live_status_line(
                    snapshot, generation=self.generation
                )
                # The AGE is deliberately NOT part of the change signature: it
                # grows on every tick, and a heartbeat that "changes" because
                # time passes would defeat the periodic rate limit and spam the
                # log. Counters + per-cell facts decide a change; the periodic
                # deadline decides the rest.
                signature = (
                    snapshot.get("total"),
                    snapshot.get("committed"),
                    snapshot.get("resumed_committed"),
                    snapshot.get("new_committed"),
                    snapshot.get("queued"),
                    snapshot.get("submitted"),
                    snapshot.get("running"),
                    snapshot.get("computed"),
                    snapshot.get("committing"),
                    snapshot.get("failed"),
                    snapshot.get("cancelled"),
                )
                due = (
                    self._last_emit_ts is None
                    or (now - float(self._last_emit_ts)) >= self.interval_s
                )
                emitted = False
                if force or due or self._last_signature != signature:
                    self._last_signature = signature
                    self._last_line = line
                    self._last_emit_ts = now
                    self._emit(line, deliver_gui=deliver_gui)
                    self.heartbeats += 1
                    emitted = True
                for row in snapshot.get("rows") or ():
                    cid = str(row.get("cell_id"))
                    key = (
                        str(row.get("state")),
                        row.get("phase"),
                        tuple(row.get("tile") or ()),
                        tuple(row.get("block") or ()),
                        tuple(row.get("frame") or ()),
                        row.get("frame_prepared"),
                        row.get("bytes_reserved"),
                        row.get("bytes_peak"),
                        row.get("producers_effective"),
                    )
                    if self._last_rows.get(cid) == key:
                        continue
                    self._last_rows[cid] = key
                    self._emit(live_cell_line(row), deliver_gui=deliver_gui)
                    self.detail_lines += 1
                return emitted
            except Exception:  # noqa: BLE001 - telemetry is never fatal
                return False

    def _emit(self, line: str, *, deliver_gui: bool) -> None:
        if self.emit_line is not None:
            try:
                self.emit_line(line)
            except Exception:  # noqa: BLE001 - telemetry is never fatal
                pass
        if deliver_gui and self.gui_emit is not None:
            try:
                self.gui_emit(line)
            except Exception:  # noqa: BLE001 - telemetry is never fatal
                pass


def heartbeat_interval(value=None, default: float = LIVE_HEARTBEAT_INTERVAL_S):
    """Clamp a diary interval to a sane, BOUNDED range (never zero/negative)."""
    try:
        interval = float(default if value is None else value)
    except (TypeError, ValueError):
        return float(default)
    if not (0.05 <= interval <= 300.0):
        return float(default)
    return interval
