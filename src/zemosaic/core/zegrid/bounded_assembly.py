"""ZM-ZEGRID-RESILIENT-RUN-RESUME M5-A — bounded, read-only assembly access.

Why this module exists
----------------------
The M2 durable store publishes one immutable commit per validated cell. Until
M5-A the assembly path reloaded EVERY committed payload at once
(``RunCellStore.ordered_payloads``) and only then started assembling, so a
187-cell mosaic materialised all PATCH planes simultaneously on top of the
canvas itself. The plan (§6.4) requires the assembly to re-read results
"without stacking everything in RAM"; §6.7 requires the pre-offset states to be
immutable on disk and opened read-only.

What this module provides
-------------------------
* :class:`CellReadSession` — a bounded, read-only window on ONE
  :class:`~zemosaic.core.zegrid.run_store.RunCellStore`. Each *immutable
  identity* ``(cell_id, attempt_id, payload_sha256)`` is verified once per
  session (byte-level store check + streaming canonical fingerprint), then its
  planes are memory-mapped read-only. Live mappings are counted with weak
  references (not a mere LRU length), so externally retained views, slices and
  reopen of the same cell are all visible and enforced BEFORE a new mapping is
  opened.
* :class:`BoundedPayloadView` / :class:`BoundedDescriptors` /
  :class:`CoresView` / :class:`PatchesView` — :class:`~typing.Mapping` views
  with the EXACT semantics the existing consumers already use
  (``cores.get(cid)``, ``patches.get(cid)``, ``patches[cid]``,
  ``cid in patches``). They never load eagerly: constructing a view performs
  zero I/O and zero mapping.
* :func:`assembly_plan` / :func:`canvas_plane_bytes` /
  :func:`preflight_assembly` — an explicit, conservative working-set model and
  a fail-closed PREFLIGHT that must run before any payload load, graph solve or
  canvas allocation in the production path. The refusal is armed by a real
  budget, never left "posthoc only".
* :func:`assert_bounded_usage` — a defect guard reading the REAL live-mapping
  counters (not a counter the caller can mutate).
* :class:`ScratchCanvasAllocator` — an optional scratch-backed canvas (not
  activated by default; M5-B decision).

Honest limits (published, not hidden)
------------------------------------
* ``np.memmap(mode="r")`` bounds the *mapped* cell payloads and their bytes.
  Mapped pages are file-backed page cache: touched pages still count in RSS
  until the OS reclaims them. ``live_mapped_bytes`` is therefore a BOUND on
  simultaneously mapped planes, not a claim of "freed RAM".
* The model below is a conservative *bound*, not a measured RSS: it prices the
  canvas accumulators exactly, the simultaneously live cell mappings from their
  declared plane bytes, the canonical consumer's transient ``float64`` working
  copies (with an overlap allowance for the previous cell) and the streaming
  hash chunk buffer. It does not claim to predict allocator/GC behaviour.
* Integrity is verified once per immutable identity per session. A file
  corrupted *after* that verification but before the mapping is read is not
  re-hashed (same contract as the commit-time digest); a new session re-verifies
  from disk. The fingerprint is streamed in fixed chunks, never by copying a
  whole plane.
* Lifetime: arrays handed out keep their mapping alive through the numpy base
  chain. The session NEVER force-closes a mapping a consumer may still hold;
  instead it refuses to open a new mapping when the live envelope would be
  exceeded (external retention included).
* The canonical science path is UNCHANGED: no formula, offset, feather weight,
  ordering or tolerance is modified here.

No new activation default: this module is only used when a durable store is
explicitly configured (``zegrid_durable_store_dir``) or a run is resumed.
"""

from __future__ import annotations

import hashlib
import os
import weakref
from collections import OrderedDict
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from . import run_registry as zrr
from .geometry import cell_id as _geometry_cell_id
from .run_store import (
    STATUS_COMPLETE,
    STATUS_EMPTY_VALIDATED,
    RunCellStore,
    StoreFormatError,
    StoreIntegrityError,
    confined_path,
)

__all__ = [
    "SCHEMA",
    "DEFAULT_MAX_RESIDENT_CELLS",
    "DEFAULT_HASH_CHUNK_BYTES",
    "BoundedAccessError",
    "AssemblyCapacityError",
    "BoundedLifetimeError",
    "CellReadSession",
    "BoundedPayloadView",
    "BoundedDescriptors",
    "CoresView",
    "PatchesView",
    "StoreBackedResults",
    "StoreAssemblyAccess",
    "ScratchCanvasAllocator",
    "array_sha256_streaming",
    "payload_sha256_streaming",
    "canvas_plane_bytes",
    "assembly_plan",
    "preflight_assembly",
    "assert_bounded_usage",
]

SCHEMA = "zemosaic.zegrid.bounded_assembly.v1"

#: Minimum honest bound for the per_cell offset graph: an edge needs the two
#: adjacent patches at the same time. One extra slot would only raise the peak.
DEFAULT_MAX_RESIDENT_CELLS = 2

#: Fixed streaming-hash chunk. This is the ONLY integrity buffer allocated per
#: verification (never a full-plane copy).
DEFAULT_HASH_CHUNK_BYTES = 1024 * 1024


class BoundedAccessError(RuntimeError):
    """Base class for bounded-access/lifetime failures."""


class AssemblyCapacityError(BoundedAccessError):
    """Explicit, fail-closed refusal: the modelled assembly does not fit.

    Raised ONLY when the caller supplies a real budget. It never downgrades to
    a silent over-commit or a partial assembly.
    """

    def __init__(
        self,
        message: str,
        *,
        plan: Mapping[str, Any] | None = None,
        required_bytes: int | None = None,
        available_bytes: int | None = None,
    ) -> None:
        super().__init__(message)
        self.plan = dict(plan or {})
        self.required_bytes = required_bytes
        self.available_bytes = available_bytes


class BoundedLifetimeError(BoundedAccessError):
    """The session is closed, or a new mapping would exceed the live envelope."""


def _prod(values: Iterable[int]) -> int:
    out = 1
    for value in values:
        out *= int(value)
    return out


def _plane_shape(plane: Mapping[str, Any]) -> tuple[int, ...]:
    return tuple(int(v) for v in plane["shape"])


def _plane_nbytes(plane: Mapping[str, Any]) -> int:
    dtype = np.dtype(str(plane["dtype"]))
    return int(dtype.itemsize) * _prod(_plane_shape(plane))


# ---------------------------------------------------------------------------
# Streaming canonical hashes (no full-plane copy)
# ---------------------------------------------------------------------------


def array_sha256_streaming(
    array, *, chunk_bytes: int = DEFAULT_HASH_CHUNK_BYTES
) -> str:
    """Canonical array digest, byte-identical to ``_array_sha256``, streamed.

    ``run_registry._array_sha256`` hashes ``dtype.str``, the canonical shape and
    the raw C-order bytes, but obtains the bytes through
    ``np.ascontiguousarray(...).tobytes()`` (a full-plane copy). This function
    streams the SAME byte sequence through a fixed ``chunk_bytes`` buffer, so a
    700 MiB plane never materialises a second copy.
    """
    a = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(a.dtype.str.encode("ascii"))
    digest.update(zrr._canonical_json([int(v) for v in a.shape]))
    if int(a.size) == 0:
        return digest.hexdigest()
    step = max(1, int(chunk_bytes))
    if a.flags.c_contiguous:
        raw = a.reshape(-1).view(np.uint8).reshape(-1)
        view = memoryview(raw)
        for start in range(0, view.nbytes, step):
            digest.update(view[start:start + step])
        return digest.hexdigest()
    # G2: NON-contiguous plane (strided, transposed, negative stride, core slice
    # of a larger patch). Stream the SAME C-order byte sequence through blocks
    # of at most `step` BYTES: the last axis is walked in element blocks and
    # every leading index is visited in C order. The only temporary is one
    # block (>= one element), never an arbitrary outer slice or the whole plane.
    if a.ndim == 0:
        block = np.ascontiguousarray(a)
        digest.update(memoryview(block.reshape(-1).view(np.uint8)))
        return digest.hexdigest()
    step_elems = max(1, step // max(1, int(a.dtype.itemsize)))
    last = int(a.shape[-1])
    if a.ndim == 1:
        for start in range(0, last, step_elems):
            block = np.ascontiguousarray(a[start:start + step_elems])
            digest.update(memoryview(block.reshape(-1).view(np.uint8)))
        return digest.hexdigest()
    for lead in np.ndindex(*tuple(int(v) for v in a.shape[:-1])):
        row = a[lead]
        for start in range(0, last, step_elems):
            block = np.ascontiguousarray(row[start:start + step_elems])
            digest.update(memoryview(block.reshape(-1).view(np.uint8)))
    return digest.hexdigest()


def payload_sha256_streaming(
    payload: Mapping[str, Any],
    *,
    chunk_bytes: int = DEFAULT_HASH_CHUNK_BYTES,
) -> str:
    """Canonical ``payload_sha256`` of ONE payload, streamed.

    Byte-identical to ``run_registry.fingerprint_result(payload)["payload_sha256"]``
    (status, per-plane canonical digests, reference pin, geometry, backend,
    n_frames, total_bytes) while never copying a whole plane.
    """
    core = payload.get("core")
    core_map = core if isinstance(core, Mapping) else {}
    core_hashes: dict[str, str] = {}
    for key in sorted(core_map):
        core_hashes[str(key)] = array_sha256_streaming(
            core_map[key], chunk_bytes=chunk_bytes
        )
    patch = payload.get("patch")
    patch_hashes: dict[str, Any] | None = None
    patch_map: Mapping[str, Any] = patch if isinstance(patch, Mapping) else {}
    if patch is not None:
        patch_hashes = {}
        for key in sorted(patch_map):
            value = patch_map[key]
            if key in ("bounds", "core_slice"):
                patch_hashes[key] = [int(v) for v in value]
            else:
                patch_hashes[str(key)] = array_sha256_streaming(
                    value, chunk_bytes=chunk_bytes
                )
    record = payload.get("record")
    record_map = record if isinstance(record, Mapping) else {}
    reference_pin = {
        "reference_frame_id": record_map.get("reference_frame_id"),
        "reference_frame_role": record_map.get("reference_frame_role"),
        "bookkeeping_reference_frame_id": record_map.get(
            "bookkeeping_reference_frame_id"
        ),
        "cell_gauge_mode": record_map.get("cell_gauge_mode"),
        "excluded": record_map.get("excluded"),
    }
    geometry = {
        "patch_bounds": record_map.get("patch_bounds"),
        "patch_shape_hw": record_map.get("patch_shape_hw"),
        "bounds": (
            [int(v) for v in patch_map["bounds"]]
            if patch is not None and "bounds" in patch_map
            else None
        ),
        "core_slice": (
            [int(v) for v in patch_map["core_slice"]]
            if patch is not None and "core_slice" in patch_map
            else None
        ),
    }
    status = str(payload.get("status", "")).strip().lower()
    backend = str(
        payload.get("backend_used")
        or record_map.get("backend_used")
        or "unknown"
    )
    return hashlib.sha256(
        zrr._canonical_json(
            {
                "status": status,
                "core": core_hashes,
                "patch": patch_hashes,
                "reference_pin": reference_pin,
                "geometry": geometry,
                "backend": backend,
                "n_frames": int(payload.get("n_frames", 0) or 0),
                "total_bytes": int(payload.get("total_bytes", 0) or 0),
            }
        )
    ).hexdigest()


# ---------------------------------------------------------------------------
# Bounded read session
# ---------------------------------------------------------------------------


class CellReadSession:
    """Bounded, read-only access to the committed payloads of ONE store.

    Parameters
    ----------
    store:
        An OPEN :class:`RunCellStore` (the session never opens/closes it).
    max_resident_cells:
        Hard cap on simultaneously LIVE cell payload identities (``>= 2`` for
        the per_cell offset graph). Values below 2 are refused: a smaller cap
        cannot serve an overlap pair.
    max_live_bytes:
        Optional hard cap on simultaneously mapped plane bytes. When set, a new
        mapping is refused if it would cross it, even if the cell count allows.
    """

    def __init__(
        self,
        store: RunCellStore,
        *,
        max_resident_cells: int = DEFAULT_MAX_RESIDENT_CELLS,
        max_live_bytes: int | None = None,
        hash_chunk_bytes: int = DEFAULT_HASH_CHUNK_BYTES,
        verify_payloads: bool = True,
    ) -> None:
        if isinstance(max_resident_cells, bool) or not isinstance(
            max_resident_cells, int
        ):
            raise BoundedAccessError("max_resident_cells must be a strict int")
        cap = int(max_resident_cells)
        if cap < 2:
            raise BoundedAccessError(
                "max_resident_cells must be >= 2: the per_cell offset graph "
                "needs both patches of an overlap edge at once"
            )
        if max_live_bytes is not None:
            if isinstance(max_live_bytes, bool) or not isinstance(
                max_live_bytes, int
            ):
                raise BoundedAccessError("max_live_bytes must be a strict int")
            if int(max_live_bytes) <= 0:
                raise BoundedAccessError("max_live_bytes must be > 0")
        chunk = int(hash_chunk_bytes)
        if chunk <= 0:
            raise BoundedAccessError("hash_chunk_bytes must be > 0")
        identity = store.identity
        if callable(identity):  # accept method or property implementations
            identity = identity()
        self._store = store
        self._cap = cap
        self._max_live_bytes = (
            None if max_live_bytes is None else int(max_live_bytes)
        )
        self._hash_chunk = chunk
        self._verify_payloads = bool(verify_payloads)
        self._identity_sha = str(store.report().get("identity_sha256"))
        self._gauge_mode = str(identity["science_contract"]["gauge"]["mode"])
        self._resident: "OrderedDict[str, dict]" = OrderedDict()
        self._resident_bytes: dict[str, int] = {}
        self._verified: set[tuple[str, str, str]] = set()
        self._fingerprints: dict[tuple[str, str, str], str] = {}
        self._descriptors: dict[str, dict] = {}
        self._closed = False
        self._loads = 0
        self._hits = 0
        self._evictions = 0
        self._verifications = 0
        self._fingerprint_computations = 0
        self._reopen_generations = 0
        # -- REAL live-mapping accounting (weak references) ---------------
        self._mapping_refs: list[Any] = []
        self._live_token_counts: dict[tuple[str, str, str], int] = {}
        self._live_mappings = 0
        self._live_bytes = 0
        self._peak_live_mappings = 0
        self._peak_live_bytes = 0
        self._peak_live_cells = 0

    # -- introspection ---------------------------------------------------
    @property
    def store(self) -> RunCellStore:
        return self._store

    @property
    def max_resident_cells(self) -> int:
        return self._cap

    @property
    def max_live_bytes(self) -> int | None:
        return self._max_live_bytes

    @property
    def closed(self) -> bool:
        return self._closed

    def enforce_live_budget(self, max_live_bytes: int) -> None:
        """Arm the byte envelope BEFORE any payload is mapped (production)."""
        if isinstance(max_live_bytes, bool) or not isinstance(max_live_bytes, int):
            raise BoundedAccessError("max_live_bytes must be a strict int")
        if int(max_live_bytes) <= 0:
            raise BoundedAccessError("max_live_bytes must be > 0")
        if self._live_mappings and int(max_live_bytes) < self._live_bytes:
            raise BoundedLifetimeError(
                "cannot lower the live-mapping envelope below the payloads "
                f"already mapped ({self._live_bytes} B live > "
                f"{int(max_live_bytes)} B requested)"
            )
        self._max_live_bytes = int(max_live_bytes)

    def cell_ids(self) -> tuple[str, ...]:
        """Every committed cell, deterministic (sorted) order."""
        self._ensure_open()
        return tuple(sorted(str(k) for k in self._store.index().get("cells", {})))

    def record(self, cell_id: str):
        self._ensure_open()
        return self._store.commit_record(str(cell_id))

    def cell_bytes(self, cell_id: str) -> int:
        """Declared bytes of ONE cell's durable planes (no array is touched)."""
        record = self.record(cell_id)
        return int(
            sum(_plane_nbytes(plane) for plane in record.manifest["planes"])
        )

    def descriptor(self, cell_id: str) -> dict:
        """Metadata-only payload descriptor (never reads a plane object)."""
        self._ensure_open()
        cid = str(cell_id)
        cached = self._descriptors.get(cid)
        if cached is not None:
            return cached
        record = self._store.commit_record(cid)
        meta: dict[str, Any] = dict(record.manifest["payload_meta"])
        meta["cell_id"] = record.cell_id
        patch_meta = meta.get("patch")
        if isinstance(patch_meta, Mapping):
            meta["patch"] = {
                key: tuple(patch_meta[key])
                for key in ("bounds", "core_slice")
                if key in patch_meta
            }
        meta["status"] = (
            "complete" if record.status == STATUS_COMPLETE else "empty"
        )
        meta["_store"] = {
            "schema": SCHEMA,
            "run_id": record.manifest["run_id"],
            "attempt_id": record.attempt_id,
            "payload_sha256": record.payload_sha256,
            "commit_file": record.commit_file,
            "source": "run_store",
            "access": "descriptor_only",
            "arrays_loaded": False,
        }
        self._descriptors[cid] = meta
        return meta

    # -- live mapping accounting -----------------------------------------
    def _release_hook(self, token, nbytes: int):
        session = self

        def _cb(_ref) -> None:  # pragma: no cover - GC callback
            try:
                session._live_mappings = max(0, session._live_mappings - 1)
                session._live_bytes = max(0, session._live_bytes - int(nbytes))
                count = session._live_token_counts.get(token, 0) - 1
                if count <= 0:
                    session._live_token_counts.pop(token, None)
                else:
                    session._live_token_counts[token] = count
            except Exception:
                pass

        return _cb

    def _register_mapping(self, token, nbytes: int, array) -> None:
        self._live_mappings += 1
        self._live_bytes += int(nbytes)
        self._live_token_counts[token] = self._live_token_counts.get(token, 0) + 1
        self._peak_live_mappings = max(self._peak_live_mappings, self._live_mappings)
        self._peak_live_bytes = max(self._peak_live_bytes, self._live_bytes)
        self._peak_live_cells = max(
            self._peak_live_cells, len(self._live_token_counts)
        )
        self._mapping_refs.append(
            weakref.ref(array, self._release_hook(token, int(nbytes)))
        )
        if len(self._mapping_refs) > 4096:
            self._mapping_refs = [r for r in self._mapping_refs if r() is not None]

    def live_cells(self) -> int:
        return len(self._live_token_counts)

    def live_mapped_bytes(self) -> int:
        return int(self._live_bytes)

    def _make_room(self, need_bytes: int, token) -> None:
        """Evict OUR references, then refuse if the live envelope is exceeded.

        Eviction happens BEFORE opening the new mapping, so a consumer that
        holds an evicted payload externally keeps it alive and is accounted for
        instead of being silently over-committed.
        """
        need = int(need_bytes)
        while True:
            prospective_cells = len(self._live_token_counts) + (
                0 if token in self._live_token_counts else 1
            )
            over_cells = prospective_cells > self._cap
            over_bytes = (
                self._max_live_bytes is not None
                and (self._live_bytes + need) > self._max_live_bytes
            )
            if not over_cells and not over_bytes:
                return
            if not self._resident:
                break
            evicted, payload = self._resident.popitem(last=False)
            self._resident_bytes.pop(evicted, None)
            self._evictions += 1
            del payload
        raise BoundedLifetimeError(
            "bounded assembly refused to open a new cell payload: the live "
            f"envelope would be exceeded (cells {len(self._live_token_counts)}"
            f"+1 > {self._cap}, bytes {self._live_bytes}+{need} > "
            f"{self._max_live_bytes}); payloads retained outside the session "
            "are still alive and are never force-closed"
        )

    # -- loading ---------------------------------------------------------
    def load(self, cell_id: str) -> dict:
        """Return ONE verified payload with read-only memory-mapped planes."""
        self._ensure_open()
        cid = str(cell_id)
        payload = self._resident.get(cid)
        if payload is not None:
            self._resident.move_to_end(cid)
            self._hits += 1
            return payload
        record = self._store.commit_record(cid)
        token = (
            cid,
            str(record.attempt_id),
            str(record.payload_sha256),
        )
        need = int(
            sum(_plane_nbytes(plane) for plane in record.manifest["planes"])
        )
        if token in self._live_token_counts:
            # Same immutable identity still mapped OUTSIDE the LRU: a second
            # mapping generation is possible only within the byte envelope and
            # is reported honestly.
            self._reopen_generations += 1
        self._make_room(need, token)
        payload = self._open_cell(record, token)
        self._resident[cid] = payload
        self._resident_bytes[cid] = need
        self._resident.move_to_end(cid)
        self._loads += 1
        while len(self._resident) > self._cap:
            evicted, held = self._resident.popitem(last=False)
            self._resident_bytes.pop(evicted, None)
            self._evictions += 1
            del held
        if len(self._live_token_counts) > self._cap:  # pragma: no cover - safety
            raise BoundedLifetimeError(
                "bounded assembly exceeded its live-cell envelope "
                f"({len(self._live_token_counts)} > {self._cap})"
            )
        return payload

    def release(self, cell_id: str) -> bool:
        """Drop the session's reference to one payload (never a mapped view)."""
        self._ensure_open()
        cid = str(cell_id)
        if cid in self._resident:
            payload = self._resident.pop(cid)
            self._resident_bytes.pop(cid, None)
            del payload
            return True
        return False

    def resident_bytes(self) -> int:
        return int(sum(self._resident_bytes.values()))

    # -- integrity -------------------------------------------------------
    def _verify_once(self, record, token) -> None:
        if token in self._verified:
            return
        report = self._store.verify_cell(record.cell_id, record.attempt_id)
        self._verifications += 1
        if not report.get("valid"):
            raise StoreIntegrityError(
                f"cell {record.cell_id} attempt {record.attempt_id!r} is not "
                f"valid: {report.get('reason')}"
            )
        self._verified.add(token)

    def _map_plane(self, plane: Mapping[str, Any], token) -> np.ndarray:
        path = confined_path(self._store.root, plane["path"], "plane path")
        dtype = np.dtype(str(plane["dtype"]))
        shape = _plane_shape(plane)
        expected = int(dtype.itemsize) * _prod(shape)
        if expected != int(plane["nbytes"]):
            raise StoreIntegrityError(
                f"object {plane['path']}: declared shape/dtype need {expected} "
                f"bytes but the manifest declares {plane['nbytes']}"
            )
        if not path.exists():
            raise StoreIntegrityError(f"object {plane['path']} is missing")
        size = int(path.stat().st_size)
        if size != expected:
            raise StoreIntegrityError(
                f"object {plane['path']} holds {size} bytes, expected {expected}"
            )
        if expected == 0:
            array = np.empty(shape, dtype=dtype)
            array.flags.writeable = False
            return array
        array = np.memmap(
            str(path), dtype=dtype, mode="r", shape=shape, order="C"
        )
        self._register_mapping(token, expected, array)
        return array

    def _open_cell(self, record, token) -> dict:
        self._verify_once(record, token)
        payload: dict[str, Any] = dict(record.manifest["payload_meta"])
        if record.status == STATUS_COMPLETE:
            core: dict[str, Any] = {}
            patch_meta = payload.get("patch")
            patch: dict[str, Any] | None = (
                None
                if patch_meta is None
                else {
                    key: tuple(patch_meta[key])
                    for key in ("bounds", "core_slice")
                }
            )
            for plane in record.manifest["planes"]:
                array = self._map_plane(plane, token)
                name = str(plane["name"])
                if name.startswith("core."):
                    core[name[len("core."):]] = array
                elif name.startswith("patch."):
                    if patch is None:
                        patch = {}
                    patch[name[len("patch."):]] = array
            payload["core"] = core
            if patch is not None:
                payload["patch"] = patch
        payload["cell_id"] = record.cell_id
        try:
            if record.status == STATUS_COMPLETE:
                zrr.validate_complete_payload(payload, gauge_mode=self._gauge_mode)
            else:
                zrr.validate_empty_payload(payload, gauge_mode=self._gauge_mode)
        except zrr.AcceptanceError as exc:
            raise StoreIntegrityError(
                f"cell {record.cell_id}: reloaded payload is invalid: {exc}"
            ) from exc
        if self._verify_payloads:
            actual = self._fingerprints.get(token)
            if actual is None:
                actual = payload_sha256_streaming(
                    payload, chunk_bytes=self._hash_chunk
                )
                self._fingerprints[token] = actual
                self._fingerprint_computations += 1
            if actual != record.payload_sha256:
                raise StoreIntegrityError(
                    f"cell {record.cell_id}: reloaded payload fingerprint "
                    f"mismatch ({actual} != {record.payload_sha256})"
                )
        payload["_store"] = {
            "schema": SCHEMA,
            "run_id": record.manifest["run_id"],
            "attempt_id": record.attempt_id,
            "payload_sha256": record.payload_sha256,
            "commit_file": record.commit_file,
            "source": "run_store",
            "access": "read_only_mmap",
            "arrays_loaded": True,
        }
        return payload

    # -- lifecycle -------------------------------------------------------
    def _ensure_open(self) -> None:
        if self._closed:
            raise BoundedLifetimeError(
                "bounded assembly session is closed; no further cell may be read"
            )

    def close(self) -> dict:
        """Drop retained payloads and refuse later loads (idempotent).

        Mappings already handed out are NOT closed here: a consumer may still
        hold a read-only view, and numpy keeps that mapping alive until the
        view dies (documented lifetime contract).
        """
        self._resident.clear()
        self._resident_bytes.clear()
        self._closed = True
        return self.stats()

    def stats(self) -> dict:
        return {
            "schema": SCHEMA,
            "access": "read_only_mmap",
            "mode": "mmap_read_only",
            "run_id": self._store.run_id,
            "identity_sha256": self._identity_sha,
            "closed": bool(self._closed),
            "max_resident_cells": int(self._cap),
            "max_live_bytes": (
                None if self._max_live_bytes is None else int(self._max_live_bytes)
            ),
            "resident_cells": len(self._resident),
            "resident_mapped_bytes": self.resident_bytes(),
            "live_cells": len(self._live_token_counts),
            "live_mappings": int(self._live_mappings),
            "live_mapped_bytes": int(self._live_bytes),
            "peak_live_cells": int(self._peak_live_cells),
            "peak_live_mappings": int(self._peak_live_mappings),
            "peak_live_mapped_bytes": int(self._peak_live_bytes),
            "loads": int(self._loads),
            "hits": int(self._hits),
            "evictions": int(self._evictions),
            "verifications": int(self._verifications),
            "verified_identities": len(self._verified),
            "fingerprints_computed": int(self._fingerprint_computations),
            "cached_identities": len(self._fingerprints),
            "reopen_generations": int(self._reopen_generations),
            "hash_chunk_bytes": int(self._hash_chunk),
            "anonymous_copy_bytes": 0,
            "notes": [
                "live_* counters follow REAL open mappings through weak refs "
                "(external views and slices included), not the LRU length",
                "mapped planes are read-only file-backed pages, not anonymous RAM",
                "live_mapped_bytes is a bound on simultaneously mapped payloads; "
                "it is not a claim of freed RAM",
                "integrity and the canonical fingerprint are checked once per "
                "immutable identity per session; the hash is streamed in fixed "
                "chunks (no full-plane copy)",
            ],
        }


# ---------------------------------------------------------------------------
# Mapping views (same access semantics as the eager dicts they replace)
# ---------------------------------------------------------------------------


class _CellKeyView(Mapping):
    """Shared deterministic key handling for the bounded views."""

    def __init__(self, session: CellReadSession, cell_ids: Sequence[str]) -> None:
        ids = tuple(dict.fromkeys(str(cid) for cid in cell_ids))
        self._session = session
        self._ids = ids
        self._set = frozenset(ids)

    @property
    def session(self) -> CellReadSession:
        return self._session

    @property
    def cell_ids(self) -> tuple[str, ...]:
        return self._ids

    def __len__(self) -> int:
        return len(self._ids)

    def __iter__(self) -> Iterator[str]:
        return iter(self._ids)

    def __contains__(self, key: object) -> bool:
        return str(key) in self._set

    def __getitem__(self, key: str):
        raise NotImplementedError

    def stats(self) -> dict:
        return self._session.stats()


class BoundedPayloadView(_CellKeyView):
    """``cell_id -> payload`` with real live-mapping enforcement."""

    def __getitem__(self, key: str) -> dict:
        cid = str(key)
        if cid not in self._set:
            raise KeyError(key)
        return self._session.load(cid)


class BoundedDescriptors(_CellKeyView):
    """``cell_id -> metadata descriptor`` (never reads a plane object)."""

    def __getitem__(self, key: str) -> dict:
        cid = str(key)
        if cid not in self._set:
            raise KeyError(key)
        return self._session.descriptor(cid)


class CoresView(_CellKeyView):
    """``cell_id -> core planes`` as expected by ``mosaic.assemble_canvas``."""

    def _is_complete(self, cid: str) -> bool:
        return str(self._session.descriptor(cid).get("status")) == "complete"

    def __contains__(self, key: object) -> bool:
        cid = str(key)
        return cid in self._set and self._is_complete(cid)

    def __getitem__(self, key: str) -> dict:
        cid = str(key)
        if cid not in self._set:
            raise KeyError(key)
        payload = self._session.load(cid)
        core = payload.get("core")
        if not core:
            raise KeyError(key)
        return core


class PatchesView(_CellKeyView):
    """``cell_id -> CellPatch`` as expected by the feather/offset consumers."""

    def _is_complete(self, cid: str) -> bool:
        meta = self._session.descriptor(cid)
        return str(meta.get("status")) == "complete" and bool(meta.get("patch"))

    def __contains__(self, key: object) -> bool:
        cid = str(key)
        return cid in self._set and self._is_complete(cid)

    def __getitem__(self, key):
        from . import cell_feather as zcf

        cid = str(key)
        if cid not in self._set:
            raise KeyError(key)
        payload = self._session.load(cid)
        if str(payload.get("status")) != "complete":
            raise KeyError(key)
        patch = payload.get("patch")
        if not patch:
            raise KeyError(key)
        return zcf.CellPatch(
            cell_id=cid,
            bounds=tuple(int(v) for v in patch["bounds"]),
            core_slice=tuple(int(v) for v in patch["core_slice"]),
            science=patch["science"],
            valid_mask=patch["valid_mask"],
            surviving=patch.get("surviving"),
        )


class StoreBackedResults:
    """Drop-in ``Sequence`` of metadata descriptors for the run/assembly seam."""

    def __init__(
        self,
        assembly: "StoreAssemblyAccess",
        cell_ids: Sequence[str] | None = None,
    ) -> None:
        self._assembly = assembly
        ids = (
            tuple(str(cid) for cid in cell_ids)
            if cell_ids is not None
            else assembly.session.cell_ids()
        )
        self._ids = tuple(dict.fromkeys(ids))

    @property
    def assembly(self) -> "StoreAssemblyAccess":
        return self._assembly

    @property
    def descriptors(self) -> BoundedDescriptors:
        return self._assembly.descriptors(self._ids)

    @property
    def cores(self) -> CoresView:
        return self._assembly.cores(self._ids)

    @property
    def patches(self) -> PatchesView:
        return self._assembly.patches(self._ids)

    @property
    def session(self) -> CellReadSession:
        return self._assembly.session

    def __len__(self) -> int:
        return len(self._ids)

    def __iter__(self) -> Iterator[dict]:
        for cid in self._ids:
            yield self._assembly.session.descriptor(cid)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [
                self._assembly.session.descriptor(cid)
                for cid in self._ids[index]
            ]
        return self._assembly.session.descriptor(self._ids[index])

    def __contains__(self, value: object) -> bool:
        return str(value) in set(self._ids)

    def close(self) -> dict:
        return self._assembly.close()

    def stats(self) -> dict:
        return self._assembly.stats()


class StoreAssemblyAccess:
    """One bounded assembly access bundle over an open store."""

    def __init__(
        self,
        store: RunCellStore,
        *,
        max_resident_cells: int = DEFAULT_MAX_RESIDENT_CELLS,
        max_live_bytes: int | None = None,
        hash_chunk_bytes: int = DEFAULT_HASH_CHUNK_BYTES,
        verify_payloads: bool = True,
    ) -> None:
        self._session = CellReadSession(
            store,
            max_resident_cells=max_resident_cells,
            max_live_bytes=max_live_bytes,
            hash_chunk_bytes=hash_chunk_bytes,
            verify_payloads=verify_payloads,
        )

    @property
    def session(self) -> CellReadSession:
        return self._session

    def descriptors(self, cell_ids: Sequence[str]) -> BoundedDescriptors:
        return BoundedDescriptors(self._session, cell_ids)

    def payloads(self, cell_ids: Sequence[str]) -> BoundedPayloadView:
        return BoundedPayloadView(self._session, cell_ids)

    def cores(self, cell_ids: Sequence[str]) -> CoresView:
        return CoresView(self._session, cell_ids)

    def patches(self, cell_ids: Sequence[str]) -> PatchesView:
        return PatchesView(self._session, cell_ids)

    def results(self, cell_ids: Sequence[str]) -> StoreBackedResults:
        return StoreBackedResults(self, cell_ids)

    def close(self) -> dict:
        return self._session.close()

    def stats(self) -> dict:
        return self._session.stats()


# ---------------------------------------------------------------------------
# Honest working-set model
# ---------------------------------------------------------------------------

#: Exact plane dtypes of ``cell_feather.assemble_canvas_feathered``. The current
#: canonical assembly costs 121 B/pixel for 3 channels (24+24+24+12+4+4+8+8+8+4+1),
#: i.e. ~909 MiB for a 2403 x 3278 canvas — NOT the "700 MiB" once quoted.
_CANVAS_PLANES: tuple[tuple[str, str, float], ...] = (
    ("num", "<f8", 3.0),           # (H, W, 3) accumulator
    ("wsum", "<f8", 3.0),          # (H, W, 3) weight accumulator
    ("science_work", "<f8", 3.0),  # (H, W, 3) float64 division buffer
    ("science_out", "<f4", 3.0),   # (H, W, 3) published float32 science
    ("stack_depth", "<i4", 1.0),
    ("owner", "<i4", 1.0),
    ("support_w1", "<f8", 1.0),
    ("support_w2", "<f8", 1.0),
    ("n_eff", "<f8", 1.0),
    ("valid_fraction", "<f4", 1.0),
    ("hole_mask", "|b1", 1.0),
)


def canvas_plane_bytes(
    canvas_hw: tuple[int, int], *, channels: int = 3
) -> dict:
    """Exact bytes of the feathered-assembly accumulators for one canvas.

    These are ANONYMOUS allocations of the current canonical assembly (unless a
    scratch allocator is wired in M5-B). At the canonical 3 channels this is
    121 B/pixel: a 2403 x 3278 canvas costs 953,121,114 B (~909 MiB), NOT the
    ~700 MiB once claimed by a partial inventory.
    """
    h, w = int(canvas_hw[0]), int(canvas_hw[1])
    if h <= 0 or w <= 0:
        raise BoundedAccessError(f"invalid canvas shape {(h, w)}")
    if int(channels) <= 0:
        raise BoundedAccessError("channels must be > 0")
    px = h * w
    planes: dict[str, dict] = {}
    total = 0
    for name, token, mult in _CANVAS_PLANES:
        dtype = np.dtype(token)
        items = 3 if float(mult) == 3.0 else 1
        if items == 3 and int(channels) != 3:
            items = int(channels)
        nbytes = int(dtype.itemsize) * px * items
        planes[name] = {
            "dtype": token,
            "shape": [h, w, items] if items > 1 else [h, w],
            "nbytes": nbytes,
        }
        total += nbytes
    return {
        "canvas_hw": [h, w],
        "channels": int(channels),
        "pixels": int(px),
        "bytes_per_pixel": (int(total) / float(px)) if px else 0.0,
        "planes": planes,
        "total_bytes": int(total),
        "kind": "anonymous_accumulators",
    }


def _per_cell_transient_bytes(
    largest_patch_hw: tuple[int, int] | None,
    *,
    channels: int,
    hash_chunk_bytes: int,
    previous_cell_overlap: bool = True,
) -> dict:
    """Conservative transient bound for the canonical per_cell consumer.

    ``assemble_canvas_feathered`` builds, for ONE patch: a float64 science
    copy, a float64 weight field, a boolean valid/finite mask, a float64 feather
    ramp and a float64 product (``sci * wgt``), while the PREVIOUS cell's locals
    can still be alive during the next ``patches.get``. The estimate is
    therefore doubled when ``previous_cell_overlap`` is set.
    """
    one = 0
    hp = wp = 0
    if largest_patch_hw is not None:
        hp, wp = int(largest_patch_hw[0]), int(largest_patch_hw[1])
    if hp > 0 and wp > 0:
        c = int(channels)
        one = int(
            hp * wp * c * 8      # float64 science copy
            + hp * wp * c * 8    # float64 weight field
            + hp * wp * c        # valid/finite bool work
            + hp * wp * 8        # float64 feather ramp
            + hp * wp * c * 8    # float64 sci*wgt product
        )
    factor = 2 if previous_cell_overlap else 1
    return {
        "one_cell_bytes": int(one),
        "previous_cell_overlap": bool(previous_cell_overlap),
        "factor": factor,
        "hash_chunk_bytes": int(hash_chunk_bytes),
        "total_bytes": int(one * factor + int(hash_chunk_bytes)),
    }



def grid_edge_upper_bound(
    nx: int, ny: int, present_cell_ids: Iterable[str]
) -> int:
    """Exact 4-neighbour edge count of a grid restricted to PRESENT cells.

    ``cell_offset.grid_adjacency`` builds right+down edges only, so this count
    is the real solver's edge upper bound (an edge whose overlap is fully
    invalid is later skipped by the graph builder, never added).
    """
    nx, ny = int(nx), int(ny)
    present = {str(cid) for cid in present_cell_ids}
    if nx < 1 or ny < 1:
        # No layout: the documented sound grid bound (<= 4 neighbours / 2).
        return int(2 * len(present))
    edges = 0
    for row in range(ny):
        for col in range(nx):
            cid = _geometry_cell_id(row, col)
            if cid not in present:
                continue
            if col + 1 < nx and _geometry_cell_id(row, col + 1) in present:
                edges += 1
            if row + 1 < ny and _geometry_cell_id(row + 1, col) in present:
                edges += 1
    return int(edges)


def dense_solve_bytes(
    n_cells: int,
    n_edges: int,
    channels: int = 3,
    *,
    dtype_bytes: int = 8,
    blas_margin_num: int = 1,
    blas_margin_den: int = 4,
) -> dict:
    """Conservative METADATA-ONLY price of the canonical dense offset solve.

    ``cell_offset.solve_additive_offsets`` builds, per channel, a dense
    ``A(n_edges, n_free)`` (``n_free = n_cells - 1`` for one connected
    component; unreachable cells are excluded), the weighted temporary
    ``w[:, None] * a``, the normal matrix ``A^T W A`` and LAPACK's solve copies.
    The previous channel's ``A`` can still be referenced when the next one is
    allocated, so the bound is SUPER-ADDITIVE (not an inventory):

    * ``A`` + weighted ``A`` + previous-channel ``A`` = 3 dense planes;
    * ``A^T A`` + 2 solve copies (matrix + RHS);
    * edge vectors (delta/weights/row indices) and the per-node output copies;
    * a documented BLAS/LAPACK margin of 25 % of the largest matrix.

    Measured reference (N=400, E=760): ``A`` alone is
    8 * 760 * 399 = 2,425,920 B — the previous ``N*C*8*4`` formula priced the
    whole graph at 38,400 B, which was not conservative.
    """
    n = max(0, int(n_cells))
    e = max(0, int(n_edges))
    c = max(1, int(channels))
    item = max(1, int(dtype_bytes))
    nf = max(0, n - 1) if n else 0
    a_bytes = item * e * nf
    weighted_bytes = item * e * nf
    previous_channel_bytes = item * e * nf
    ata_bytes = item * nf * nf
    solve_workspace_bytes = 2 * ata_bytes
    vectors_bytes = int(16 * e) + int(8 * e * c)   # rows(E,2) int64 + delta/w
    offsets_bytes = item * n * c * 2               # solution + per-node copies
    largest = max(a_bytes, ata_bytes)
    blas_bytes = int(
        largest * int(blas_margin_num) // max(1, int(blas_margin_den))
    )
    total = int(
        a_bytes
        + weighted_bytes
        + previous_channel_bytes
        + ata_bytes
        + solve_workspace_bytes
        + vectors_bytes
        + offsets_bytes
        + blas_bytes
    )
    return {
        "mode": "per_cell_dense_least_squares",
        "cells": n,
        "edges": e,
        "free": nf,
        "channels": c,
        "dtype_bytes": item,
        "matrices": {
            "A_bytes": a_bytes,
            "weighted_A_bytes": weighted_bytes,
            "previous_channel_A_bytes": previous_channel_bytes,
            "ATA_bytes": ata_bytes,
            "solve_workspace_bytes": solve_workspace_bytes,
            "blas_margin_bytes": blas_bytes,
        },
        "vectors_bytes": vectors_bytes,
        "offsets_bytes": offsets_bytes,
        "total_bytes": total,
        "assumptions": [
            "n_free = n_cells - 1 (one connected component; unreachable cells "
            "are excluded from the solve)",
            "edges are the actual 4-neighbour pairs among present cells",
            "peak is per channel (channels are solved sequentially)",
            "LAPACK/BLAS workspace is priced as 25 % of the largest matrix",
            "conservative bound, not a measured allocator peak",
        ],
    }


def assembly_plan(
    canvas_hw: tuple[int, int],
    *,
    cell_plane_bytes: Sequence[int] | Mapping[str, int],
    max_resident_cells: int = DEFAULT_MAX_RESIDENT_CELLS,
    largest_patch_hw: tuple[int, int] | None = None,
    channels: int = 3,
    hash_chunk_bytes: int = DEFAULT_HASH_CHUNK_BYTES,
    scratch_bytes: int = 0,
    available_bytes: int | None = None,
    gauge_mode: str = "per_cell",
    n_cells_complete: int | None = None,
    n_edges: int | None = None,
) -> dict:
    """Price one bounded assembly; refuse explicitly when a budget is given.

    ``cell_plane_bytes`` are the DECLARED plane bytes per cell (from the commit
    manifests — no array is read). The result distinguishes, honestly:

    * ``store_total_bytes`` — the whole durable cell set on disk (NOT resident);
    * ``resident_mapped_bytes`` — the worst case simultaneously mapped (cap ×
      the largest cells);
    * ``canvas_bytes`` — exact anonymous accumulators (121 B/px at 3 channels);
    * ``transient_bytes`` — conservative consumer working copies (one cell, or
      two while the previous cell's locals overlap) + streaming hash chunk;
    * ``graph`` — conservative price of the canonical DENSE offset solve
      (``A``, weighted ``A``, previous-channel ``A``, ``A^T A``, LAPACK copies,
      vectors, BLAS margin). In ``global`` mode this is exactly 0: no offset
      solve runs;
    * ``scratch_bytes`` — caller-declared scratch (M5-B generation).
    """
    if isinstance(max_resident_cells, bool) or not isinstance(
        max_resident_cells, int
    ):
        raise BoundedAccessError("max_resident_cells must be a strict int")
    cap = int(max_resident_cells)
    if cap < 1:
        raise BoundedAccessError("max_resident_cells must be >= 1")
    sizes = sorted(int(v) for v in cell_plane_bytes)
    store_total = int(sum(sizes))
    resident = int(sum(sizes[-cap:])) if sizes else 0
    canvas = canvas_plane_bytes(canvas_hw, channels=channels)
    transient = _per_cell_transient_bytes(
        largest_patch_hw,
        channels=channels,
        hash_chunk_bytes=hash_chunk_bytes,
    )
    # G1: price the ACTUAL canonical dense offset solve from metadata. Global
    # assembly runs no offset solve at all (mosaic.assemble_canvas), so it must
    # not be charged a quadratic graph cost it never allocates.
    n_cells = (
        int(n_cells_complete) if n_cells_complete is not None else len(sizes)
    )
    mode = str(gauge_mode)
    if mode == "global":
        graph = {
            "mode": "global_no_solve",
            "cells": n_cells,
            "edges": 0,
            "free": 0,
            "channels": int(channels),
            "total_bytes": 0,
            "assumptions": [
                "global assembly places core slices; it builds no offset graph"
            ],
        }
    else:
        if mode not in ("per_cell",):
            raise BoundedAccessError(
                f"unsupported gauge_mode for the assembly model: {mode!r}"
            )
        edges = int(n_edges) if n_edges is not None else int(2 * n_cells)
        graph = dense_solve_bytes(n_cells, edges, channels)
    graph_bytes = int(graph["total_bytes"])
    scratch = int(scratch_bytes)
    if scratch < 0:
        raise BoundedAccessError("scratch_bytes must be >= 0")
    required = int(
        resident
        + canvas["total_bytes"]
        + transient["total_bytes"]
        + graph_bytes
        + scratch
    )
    plan = {
        "schema": SCHEMA,
        "access": "read_only_mmap",
        "canvas": canvas,
        "cells": {
            "count": len(sizes),
            "store_total_bytes": store_total,
            "max_cell_bytes": int(sizes[-1]) if sizes else 0,
            "median_cell_bytes": int(sizes[len(sizes) // 2]) if sizes else 0,
            "max_resident_cells": cap,
            "resident_mapped_bytes": resident,
        },
        "transient": transient,
        "per_cell_transient_bytes": int(transient["total_bytes"]),
        "graph": graph,
        "graph_bytes": graph_bytes,
        "scratch_bytes": scratch,
        "required_bytes": required,
        "available_bytes": (
            None if available_bytes is None else int(available_bytes)
        ),
        "fits": (
            None
            if available_bytes is None
            else bool(required <= int(available_bytes))
        ),
        "notes": [
            "store_total_bytes is on disk and is never loaded at once",
            "resident_mapped_bytes is a mapped-page bound, not anonymous RAM",
            "canvas bytes are the canonical feathered-assembly accumulators",
            "the model is a conservative bound, not a measured RSS",
        ],
    }
    if available_bytes is not None and required > int(available_bytes):
        raise AssemblyCapacityError(
            "bounded assembly does not fit the supplied budget: required "
            f"{required} B ({resident} B resident mapped + "
            f"{canvas['total_bytes']} B canvas + "
            f"{transient['total_bytes']} B transient + {graph_bytes} B graph + "
            f"{scratch} B scratch) > available {int(available_bytes)} B",
            plan=plan,
            required_bytes=required,
            available_bytes=int(available_bytes),
        )
    return plan


def preflight_assembly(
    session: CellReadSession,
    cell_ids: Sequence[str],
    *,
    canvas_hw: tuple[int, int],
    available_bytes: int | None,
    channels: int = 3,
    require_budget: bool = True,
    purpose: str = "assembly",
    scratch_bytes: int = 0,
    gauge_mode: str = "per_cell",
    grid: tuple[int, int] | None = None,
) -> dict:
    """Fail-closed assembly capacity gate, from METADATA ONLY.

    Reads the commit manifests (plane bytes + patch dimensions) and prices the
    canvas/transients. It opens no payload, solves no graph and allocates no
    canvas. In production ``available_bytes`` is mandatory (``require_budget``);
    a diagnostic-only caller must say so explicitly.

    The returned plan is also used to arm the session's live byte envelope.
    """
    if available_bytes is None and require_budget:
        raise BoundedAccessError(
            "assembly preflight requires a fresh available-memory budget in "
            "production; pass require_budget=False only for diagnostics"
        )
    ids = [str(cid) for cid in cell_ids]
    before = session.stats()
    cell_bytes = [session.cell_bytes(cid) for cid in ids]
    largest_patch = None
    complete: list[str] = []
    for cid in ids:
        meta = session.descriptor(cid)
        if str(meta.get("status")) == "complete" and meta.get("patch"):
            complete.append(cid)
        record = meta.get("record") or {}
        shape = record.get("patch_shape_hw")
        if isinstance(shape, (list, tuple)) and len(shape) == 2:
            hw = (int(shape[0]), int(shape[1]))
            if largest_patch is None or hw[0] * hw[1] > largest_patch[0] * largest_patch[1]:
                largest_patch = hw
    # G1: edges from the REAL layout restricted to complete cells (metadata).
    if grid is not None:
        n_edges = grid_edge_upper_bound(int(grid[0]), int(grid[1]), complete)
    else:
        n_edges = int(2 * len(complete))
    def _record(plan_obj, *, refused):
        graph = plan_obj.get("graph") or {}
        return {
            "purpose": str(purpose),
            "metadata_only": True,
            "gauge_mode": str(gauge_mode),
            "complete_cells": len(complete),
            "grid": (None if grid is None else [int(grid[0]), int(grid[1])]),
            "solver_inputs": {
                "cells": int(graph.get("cells", 0)),
                "edges": int(graph.get("edges", 0)),
                "free": int(graph.get("free", 0)),
            },
            "payloads_loaded_before": int(before["loads"]),
            "payloads_loaded_after": int(session.stats()["loads"]),
            "verifications_before": int(before["verifications"]),
            "verifications_after": int(session.stats()["verifications"]),
            "mappings_opened": int(
                session.stats()["live_mappings"] - before["live_mappings"]
            ),
            "budget_required": bool(require_budget),
            "refused": bool(refused),
        }

    try:
        plan = assembly_plan(
            canvas_hw,
            cell_plane_bytes=cell_bytes,
            max_resident_cells=session.max_resident_cells,
            largest_patch_hw=largest_patch,
            channels=channels,
            hash_chunk_bytes=session._hash_chunk,
            scratch_bytes=scratch_bytes,
            available_bytes=available_bytes,
            gauge_mode=gauge_mode,
            n_cells_complete=len(complete),
            n_edges=n_edges,
        )
    except AssemblyCapacityError as exc:
        # The refusal must be as auditable as the accepted plan: the caller
        # gets the SAME metadata-only provenance even when pricing fails.
        exc.plan["preflight"] = _record(exc.plan, refused=True)
        raise
    after = session.stats()
    plan["preflight"] = _record(plan, refused=False)
    after = session.stats()
    if after["loads"] != before["loads"] or after["verifications"] != before["verifications"]:
        raise BoundedAccessError(
            "assembly preflight must not load or verify any payload"
        )
    return plan


class ScratchCanvasAllocator:
    """Zero-filled canvas planes backed by a confined scratch directory.

    Not activated by default anywhere: M5-A ships the primitive and its exact
    parity test; M5-B wires the publication generation. The contract is the one
    ``assemble_canvas_feathered`` documents: ``allocator(shape, dtype)`` returns
    a ZERO-FILLED writable array with the requested shape/dtype.
    """

    def __init__(self, scratch_dir: str, *, prefix: str = "canvas") -> None:
        self._dir = os.path.abspath(str(scratch_dir))
        os.makedirs(self._dir, exist_ok=True)
        self._prefix = str(prefix)
        self._files: list[str] = []
        self._arrays: list[np.ndarray] = []

    @property
    def directory(self) -> str:
        return self._dir

    def __call__(self, shape, dtype):
        return self.zeros(shape, dtype)

    def zeros(self, shape, dtype) -> np.ndarray:
        index = len(self._files)
        path = os.path.join(self._dir, f"{self._prefix}-{index:03d}.bin")
        dtype = np.dtype(dtype)
        shape = tuple(int(v) for v in shape)
        array = np.memmap(path, dtype=dtype, mode="w+", shape=shape, order="C")
        self._files.append(path)
        self._arrays.append(array)
        return array

    def bytes_allocated(self) -> int:
        return int(sum(arr.nbytes for arr in self._arrays))

    def flush(self) -> None:
        for array in self._arrays:
            try:
                array.flush()
            except Exception:  # pragma: no cover - platform flush edge
                continue

    def discard(self) -> int:
        """Unlink the scratch files (only after the canvas is no longer used)."""
        removed = 0
        for path in self._files:
            try:
                if os.path.exists(path):
                    os.unlink(path)
                    removed += 1
            except OSError:  # pragma: no cover - Windows lock edge
                continue
        self._files.clear()
        self._arrays.clear()
        return removed


def assert_bounded_usage(
    session: CellReadSession,
    *,
    examined_cells: int,
    allow_full_residency: bool = False,
    totals: Mapping[str, int] | None = None,
) -> dict:
    """Defect guard on the REAL live counters (never a caller-mutable number)."""
    stats = session.stats()
    cap = int(stats["max_resident_cells"])
    if int(stats["peak_live_cells"]) > cap:
        raise BoundedAccessError(
            f"bounded assembly exceeded its live-cell envelope: peak "
            f"{stats['peak_live_cells']} live cells > {cap}"
        )
    max_bytes = stats.get("max_live_bytes")
    if max_bytes is not None and int(stats["peak_live_mapped_bytes"]) > int(max_bytes):
        raise BoundedAccessError(
            "bounded assembly exceeded its live-byte envelope: peak "
            f"{stats['peak_live_mapped_bytes']} B > {max_bytes} B"
        )
    if totals is not None and not allow_full_residency:
        total = int(totals.get("store_total_bytes", 0))
        if total and int(stats["peak_live_mapped_bytes"]) >= total:
            raise BoundedAccessError(
                "bounded assembly mapped every payload at once "
                f"({stats['peak_live_mapped_bytes']} B >= {total} B)"
            )
    if not allow_full_residency and int(examined_cells) > cap:
        if int(stats["peak_live_mapped_bytes"]) <= 0 and int(examined_cells) > 0:
            raise BoundedAccessError(
                "bounded assembly examined cells without reading any payload"
            )
    return stats
