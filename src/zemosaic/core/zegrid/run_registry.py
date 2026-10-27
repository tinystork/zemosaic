"""ZM-ZEGRID-RESILIENT-RUN-RESUME M1 — run-owned in-memory result registry.

This module is the M1 runtime-conservation seam. It owns the accepted
per-cell results of ONE run for the whole lifetime of that run, independently
of the GPU producer/consumer pipeline, the worker pool or the result queue.

M1 REWORK-1 invariants (Nono F1–F5, Junior disposition):

* identity is the sparse ``cell_id`` (``r0000c0000``), never a prefix index;
* acceptance validates the REAL producer schema before any accounting:
  COMPLETE requires every core plane, consistent geometry/dtypes and the
  photometric reference pin; per_cell additionally requires the pre-offset /
  pre-feather patch with halo; global legitimately has no patch.
  EMPTY_VALIDATED keeps the real empty-cell producer schema (record with
  zero contributors and patch geometry, no core planes);
* ``status`` is part of the fingerprint and duplicate identity. A contradictory
  duplicate (same cell, different status/payload/attempt/backend/generation) is
  rejected and never overwrites; an identical one is idempotent and counts once;
* only the CURRENT plan generation is accepted. Future and stale generations are
  rejected without state mutation. Production callers capture the generation at
  dispatch; ``accept`` never adopts whatever generation happens to be current
  when a callback arrives;
* staging/validation/accounting failures leave counts, bytes and pending
  unchanged. Fingerprinting happens before the small insertion transaction;
* retained bytes are HONEST: logical plane bytes and unique backing allocations
  are reported separately. A slice/view counts its base allocation once (also
  across core and halo aliases and across cells), never the temporary
  contiguous copy used for hashing, and never a NumPy scalar's detach size.
  ``np.memmap`` is counted conservatively and labelled; an unsupported backing
  owner fails closed;
* the event journal is bounded and holds no payload references;
* ``dispose`` is the single explicit ownership release and is idempotent;
* the registry is COORDINATOR/PROCESS-LOCAL by design. It is not picklable and
  typed exceptions carry a local handle only; the small cross-process stop
  channel is ``SpawnStopChannel`` (multiprocessing events + shared metadata +
  a multiprocessing lock), never the registry itself.

M1 conserves accepted work during a run. It does NOT persist it: after
``dispose`` (or process exit) the payloads are gone. Durable save/restart is
M2+ and must not be claimed from this module.
"""

from __future__ import annotations

import hashlib
import json
import mmap
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

SCHEMA = "zemosaic-zegrid-run-registry-v1"
RECEIPT_SCHEMA = "zemosaic-zegrid-run-registry-receipt-v1"

STATUS_COMPLETE = "COMPLETE"
STATUS_EMPTY_VALIDATED = "EMPTY_VALIDATED"

_CELL_STATUS_MAP = {
    "complete": STATUS_COMPLETE,
    "empty": STATUS_EMPTY_VALIDATED,
}

#: Core planes that MUST be preserved for the global assembly consumers.
CORE_PLANES = (
    "science_core",
    "estimator_weight_sum_core",
    "support_w1_core",
    "support_w2_core",
    "n_eff_support_core",
    "valid_mask_core",
    "surviving_sample_count_core",
)

#: Core planes that must be 2-D ``(H, W)``.
CORE_2D_PLANES = (
    "estimator_weight_sum_core",
    "support_w1_core",
    "support_w2_core",
    "n_eff_support_core",
    "valid_mask_core",
    "surviving_sample_count_core",
)

#: Patch planes that MUST be preserved for the per-cell feather/offset path.
#: ``science``/``valid_mask``/``surviving`` include the pre-offset halo.
PATCH_PLANES = ("science", "valid_mask", "surviving")

GAUGE_MODES = ("global", "per_cell")
DEFAULT_MAX_EVENTS = 512


class RunRegistryError(RuntimeError):
    """Base class for a conservation/acceptance contract failure."""


class RegistryClosedError(RunRegistryError):
    """The registry was explicitly disposed; it accepts nothing further."""


class UnexpectedCellError(RunRegistryError):
    """A result arrived for a cell that is not part of this run's identity."""


class StaleGenerationError(RunRegistryError):
    """A result belongs to an older plan generation than the current one."""


class FutureGenerationError(RunRegistryError):
    """A result belongs to a generation the run has not authorised yet."""


class DuplicateCellError(RunRegistryError):
    """A contradictory payload/attempt tried to overwrite an accepted cell."""


class AcceptanceError(RunRegistryError):
    """The payload is invalid or cannot be fingerprinted/accounted."""


class PayloadReleasedError(RunRegistryError):
    """A released cell's payload is no longer in RAM (M2 durable hand-off).

    The receipt/metadata survive; only the large arrays were released after a
    durable commit. Callers must re-read the cell from the run store instead of
    pretending the in-memory payload still exists.
    """


class UnsupportedBufferOwnerError(AcceptanceError):
    """A payload array has a backing owner this registry cannot account."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_array(value: Any, label: str):
    import numpy as np

    if not isinstance(value, np.ndarray):
        raise AcceptanceError(
            f"{label} must be a numpy array, got {type(value).__name__}"
        )
    return value


def _require_int_sequence(value: Any, label: str, length: int) -> tuple[int, ...]:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != int(length)
        or any(not _is_int(v) for v in value)
    ):
        raise AcceptanceError(
            f"{label} must be {length} strict ints, got {value!r}"
        )
    out = tuple(int(v) for v in value)
    if any(v < 0 for v in out):
        raise AcceptanceError(f"{label} must be non-negative, got {out!r}")
    return out


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AcceptanceError(
            f"{label} must be a mapping, got {type(value).__name__}"
        )
    return value


def _array_sha256(value: Any) -> str:
    """SHA-256 of one array's dtype/shape/raw bytes (copy never retained)."""
    import numpy as np

    arr = _require_array(value, "payload plane")
    contiguous = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(contiguous.dtype.str.encode("ascii"))
    h.update(_canonical_json([int(v) for v in contiguous.shape]))
    h.update(contiguous.tobytes())
    return h.hexdigest()


def _root_buffer(value: Any):
    """Return ``(root_owner, kind)`` for one payload array.

    A view/slice keeps its base allocation alive, so the honest retained size is
    the ROOT allocation, not the view's ``nbytes``. This follows ndarray bases,
    memmap slices to their underlying mapping, and memoryview ``obj`` chains.
    Supported roots are ndarrays, memmaps, mmaps and byte buffers; anything else
    fails closed.
    """
    import numpy as np

    current = _require_array(value, "payload plane")
    seen: set[int] = set()
    while True:
        if id(current) in seen:  # pragma: no cover - defensive cycle guard
            raise UnsupportedBufferOwnerError(
                "payload array has a cyclic base chain"
            )
        seen.add(id(current))
        if isinstance(current, memoryview):
            current = current.obj
            continue
        if isinstance(current, np.memmap):
            base = getattr(current, "base", None)
            if isinstance(base, np.memmap):
                current = base
                continue
            mapping = getattr(current, "_mmap", None)
            if mapping is not None:
                return mapping, "memmap_mapping"
            return current, "memmap"
        if isinstance(current, np.ndarray):
            base = getattr(current, "base", None)
            if base is None:
                return current, "ndarray"
            if isinstance(
                base,
                (np.ndarray, memoryview, mmap.mmap, bytes, bytearray),
            ):
                current = base
                continue
            raise UnsupportedBufferOwnerError(
                "payload array base owner "
                f"{type(base).__name__} is not supported"
            )
        if isinstance(current, (bytes, bytearray)):
            return current, "byte_buffer"
        if isinstance(current, mmap.mmap):
            return current, "memmap_mapping"
        raise UnsupportedBufferOwnerError(
            f"payload array base owner {type(current).__name__} is not supported"
        )


def _root_size(root: Any, kind: str) -> int:
    if kind == "byte_buffer":
        return int(len(root))
    if kind == "memmap_mapping":
        try:
            return int(root.size())
        except Exception:
            return int(len(root))
    return int(getattr(root, "nbytes", 0))


def _plane_values(result: Mapping[str, Any]):
    """Yield ``(label, array)`` for every preserved payload plane."""
    core = _mapping(result.get("core"))
    for key in CORE_PLANES:
        if key in core:
            yield f"core.{key}", core[key]
    patch = result.get("patch")
    if patch is not None:
        patch_map = _require_mapping(patch, "result.patch")
        for key in PATCH_PLANES:
            if key in patch_map:
                yield f"patch.{key}", patch_map[key]


def _buffer_accounting(result: Mapping[str, Any]) -> dict:
    """Return logical bytes and unique backing allocations for one payload."""
    logical_bytes = 0
    buffers: dict[int, dict] = {}
    for label, value in _plane_values(result):
        arr = _require_array(value, label)
        logical_bytes += int(arr.nbytes)
        root, kind = _root_buffer(arr)
        key = id(root)
        entry = buffers.get(key)
        if entry is None:
            buffers[key] = {
                "bytes": _root_size(root, kind),
                "kind": kind,
                "planes": [label],
            }
        else:
            entry["planes"].append(label)
    return {
        "logical_bytes": int(logical_bytes),
        "buffers": buffers,
        "retained_bytes": int(sum(b["bytes"] for b in buffers.values())),
        "memmap_bytes": int(
            sum(
                b["bytes"]
                for b in buffers.values()
                if b["kind"] in ("memmap", "memmap_mapping", "memmap_view")
            )
        ),
    }


def _require_pin(record: Mapping[str, Any]) -> dict:
    ref = record.get("reference_frame_id")
    if not isinstance(ref, str) or not ref:
        raise AcceptanceError(
            "COMPLETE record requires a non-empty reference_frame_id"
        )
    gauge_mode = record.get("cell_gauge_mode")
    if gauge_mode not in GAUGE_MODES:
        raise AcceptanceError(
            f"COMPLETE record has invalid cell_gauge_mode={gauge_mode!r}"
        )
    backend = record.get("backend_used")
    if backend is not None and not isinstance(backend, str):
        raise AcceptanceError("record.backend_used must be a string when present")
    return {
        "reference_frame_id": ref,
        "reference_frame_role": record.get("reference_frame_role"),
        "bookkeeping_reference_frame_id": record.get(
            "bookkeeping_reference_frame_id"
        ),
        "cell_gauge_mode": gauge_mode,
        "excluded": record.get("excluded"),
    }


def validate_complete_payload(
    result: Mapping[str, Any], *, gauge_mode: str | None = None
) -> dict:
    """Validate a COMPLETE result against the REAL producer schema.

    ``gauge_mode`` is the run's resolved mode. When supplied, the record's mode
    must agree and per_cell requires the halo patch. When omitted, the record's
    own mode decides the patch requirement (tests/standalone callers).
    """
    core = _require_mapping(result.get("core"), "COMPLETE result.core")
    missing = [key for key in CORE_PLANES if key not in core]
    if missing:
        raise AcceptanceError(
            "COMPLETE payload is missing core planes: " + ", ".join(missing)
        )
    record = _require_mapping(result.get("record"), "COMPLETE result.record")
    pin = _require_pin(record)
    record_mode = pin["cell_gauge_mode"]
    if gauge_mode is not None:
        if gauge_mode not in GAUGE_MODES:
            raise AcceptanceError(f"invalid registry gauge_mode={gauge_mode!r}")
        if record_mode != gauge_mode:
            raise AcceptanceError(
                "COMPLETE record gauge mode "
                f"{record_mode!r} disagrees with run mode {gauge_mode!r}"
            )
    effective_mode = record_mode

    bounds = _require_int_sequence(
        record.get("patch_bounds"), "record.patch_bounds", 4
    )
    patch_shape = _require_int_sequence(
        record.get("patch_shape_hw"), "record.patch_shape_hw", 2
    )
    # REAL producer convention: record.patch_bounds is (x0, y0, x1, y1) while
    # patch.bounds is (y0, y1, x0, x1) and patch_shape_hw is (height, width).
    _x0, _y0, _x1, _y1 = bounds
    if _x1 <= _x0 or _y1 <= _y0:
        raise AcceptanceError(f"record.patch_bounds is empty: {bounds!r}")
    if (int(_y1 - _y0), int(_x1 - _x0)) != tuple(patch_shape):
        raise AcceptanceError(
            "record.patch_bounds does not match patch_shape_hw (x0,y0,x1,y1): "
            f"{bounds!r} vs {patch_shape!r}"
        )

    import numpy as np

    shape_hw: tuple[int, int] | None = None
    for key in CORE_PLANES:
        arr = _require_array(core[key], f"core.{key}")
        # Real color producers carry a trailing channel axis on every plane
        # (e.g. estimator/support/mask/surviving are (H, W, 3) when the patch
        # is RGB); mono producers are 2-D. Only the spatial axes are fixed.
        if arr.ndim not in (2, 3):
            raise AcceptanceError(
                f"core.{key} must be 2-D or 3-D, got {arr.shape}"
            )
        if arr.shape[0] <= 0 or arr.shape[1] <= 0:
            raise AcceptanceError(f"core.{key} has an empty spatial axis")
        if shape_hw is None:
            shape_hw = (int(arr.shape[0]), int(arr.shape[1]))
        elif (int(arr.shape[0]), int(arr.shape[1])) != shape_hw:
            raise AcceptanceError(
                f"core.{key} shape {arr.shape} disagrees with {shape_hw}"
            )
    science_core = _require_array(core["science_core"], "core.science_core")
    if (int(science_core.shape[0]), int(science_core.shape[1])) != shape_hw:
        raise AcceptanceError(
            f"core.science_core shape {science_core.shape} disagrees with "
            f"{shape_hw}"
        )
    if not np.issubdtype(core["valid_mask_core"].dtype, np.bool_):
        raise AcceptanceError("core.valid_mask_core must be boolean")
    if not np.issubdtype(
        core["surviving_sample_count_core"].dtype, np.integer
    ):
        raise AcceptanceError(
            "core.surviving_sample_count_core must be integer"
        )
    for key in (
        "science_core",
        "estimator_weight_sum_core",
        "support_w1_core",
        "support_w2_core",
        "n_eff_support_core",
    ):
        if not np.issubdtype(core[key].dtype, np.number):
            raise AcceptanceError(f"core.{key} must be numeric")

    patch = result.get("patch")
    patch_geometry = None
    if patch is not None:
        patch_map = _require_mapping(patch, "COMPLETE result.patch")
        p_bounds = _require_int_sequence(
            patch_map.get("bounds"), "patch.bounds", 4
        )
        p_core_slice = _require_int_sequence(
            patch_map.get("core_slice"), "patch.core_slice", 4
        )
        if p_bounds != (_y0, _y1, _x0, _x1):
            raise AcceptanceError(
                f"patch.bounds {p_bounds!r} disagrees with record "
                f"(y0,y1,x0,x1) {(_y0, _y1, _x0, _x1)!r}"
            )
        y0, y1, x0, x1 = p_core_slice
        if y1 <= y0 or x1 <= x0:
            raise AcceptanceError(f"patch.core_slice is empty: {p_core_slice!r}")
        if (
            y0 < 0
            or x0 < 0
            or y1 > patch_shape[0]
            or x1 > patch_shape[1]
        ):
            raise AcceptanceError(
                f"patch.core_slice {p_core_slice!r} is outside patch_shape "
                f"{patch_shape!r}"
            )
        if (y1 - y0, x1 - x0) != shape_hw:
            raise AcceptanceError(
                "patch.core_slice dimensions "
                f"{(y1 - y0, x1 - x0)!r} disagree with core {shape_hw!r}"
            )
        for key in PATCH_PLANES:
            if key not in patch_map:
                raise AcceptanceError(f"per-cell patch is missing {key}")
            arr = _require_array(patch_map[key], f"patch.{key}")
            if arr.ndim not in (2, 3):
                raise AcceptanceError(
                    f"patch.{key} must be 2-D or 3-D, got {arr.shape}"
                )
            if (int(arr.shape[0]), int(arr.shape[1])) != patch_shape:
                raise AcceptanceError(
                    f"patch.{key} shape {arr.shape} disagrees with "
                    f"patch_shape_hw {patch_shape}"
                )
        if not np.issubdtype(patch_map["valid_mask"].dtype, np.bool_):
            raise AcceptanceError("patch.valid_mask must be boolean")
        if not np.issubdtype(patch_map["surviving"].dtype, np.integer):
            raise AcceptanceError("patch.surviving must be integer")
        if not np.issubdtype(patch_map["science"].dtype, np.number):
            raise AcceptanceError("patch.science must be numeric")
        patch_geometry = {
            "bounds": list(p_bounds),
            "core_slice": list(p_core_slice),
        }
    elif effective_mode == "per_cell":
        raise AcceptanceError(
            "per_cell COMPLETE payload requires the pre-offset/pre-feather patch"
        )
    return {
        "status": STATUS_COMPLETE,
        "pin": pin,
        "geometry": {
            "patch_bounds": list(bounds),
            "patch_shape_hw": list(patch_shape),
            "patch": patch_geometry,
        },
        "core_shape_hw": [int(shape_hw[0]), int(shape_hw[1])],
        "gauge_mode": effective_mode,
    }


def validate_empty_payload(
    result: Mapping[str, Any], *, gauge_mode: str | None = None
) -> dict:
    """Validate the REAL zero-support/geometrically-empty producer schema."""
    core = result.get("core")
    if core is not None:
        raise AcceptanceError(
            "EMPTY_VALIDATED must not carry core planes"
        )
    patch = result.get("patch")
    if patch is not None:
        raise AcceptanceError(
            "EMPTY_VALIDATED must not carry a per-cell patch"
        )
    record = _require_mapping(result.get("record"), "EMPTY result.record")
    if str(record.get("cell_id")) != str(result.get("cell_id")):
        raise AcceptanceError("EMPTY record.cell_id disagrees with result")
    contributors = record.get("n_contributors", 0)
    gauge_status = record.get("gauge_status")
    if gauge_status is None:
        gauge_status = result.get("gauge_status")
    if contributors not in (0, None):
        # Real producer case 2: a POPULATED cell whose contributors carry no
        # finite support. `_prepare_stack_cell` returns cell_fixed=None with
        # gauge_status="empty" while preserving n_contributors. Do not accept
        # an arbitrary non-zero count without that evidence, and never rewrite
        # the real contributor count to satisfy the schema.
        if gauge_status != "empty":
            raise AcceptanceError(
                "EMPTY_VALIDATED with n_contributors > 0 requires "
                f"gauge_status='empty', got {gauge_status!r}"
            )
        if not isinstance(contributors, int) or contributors <= 0:
            raise AcceptanceError(
                f"invalid zero-support n_contributors={contributors!r}"
            )
    bounds = _require_int_sequence(
        record.get("patch_bounds"), "EMPTY record.patch_bounds", 4
    )
    patch_shape = _require_int_sequence(
        record.get("patch_shape_hw"), "EMPTY record.patch_shape_hw", 2
    )
    _x0, _y0, _x1, _y1 = bounds
    if _x1 <= _x0 or _y1 <= _y0:
        raise AcceptanceError(f"EMPTY record.patch_bounds is empty: {bounds!r}")
    if (int(_y1 - _y0), int(_x1 - _x0)) != tuple(patch_shape):
        raise AcceptanceError(
            "EMPTY record.patch_bounds does not match patch_shape_hw"
        )
    mode = record.get("cell_gauge_mode")
    if gauge_mode is not None and mode is not None and mode != gauge_mode:
        raise AcceptanceError(
            f"EMPTY record gauge mode {mode!r} disagrees with run {gauge_mode!r}"
        )
    return {
        "status": STATUS_EMPTY_VALIDATED,
        "pin": {
            "reference_frame_id": record.get("reference_frame_id"),
            "reference_frame_role": record.get("reference_frame_role"),
            "bookkeeping_reference_frame_id": record.get(
                "bookkeeping_reference_frame_id"
            ),
            "cell_gauge_mode": mode,
            "excluded": record.get("excluded"),
        },
        "geometry": {
            "patch_bounds": list(bounds),
            "patch_shape_hw": list(patch_shape),
            "patch": None,
            "n_contributors": (
                None if contributors is None else int(contributors)
            ),
            "gauge_status": gauge_status,
        },
    }


def fingerprint_result(result: Mapping[str, Any]) -> dict:
    """Return the complete conservation fingerprint of ONE cell result.

    Validation is separate (``validate_complete_payload`` /
    ``validate_empty_payload``); this function hashes only the preserved planes
    and identity fields, including the status token.
    """
    core = _mapping(result.get("core"))
    core_hashes: dict[str, str] = {}
    for key in sorted(core):
        core_hashes[str(key)] = _array_sha256(core[key])
    patch = result.get("patch")
    patch_hashes: dict[str, Any] | None = None
    if patch is not None:
        patch_map = _require_mapping(patch, "result.patch")
        patch_hashes = {}
        for key in sorted(patch_map):
            value = patch_map[key]
            if key in ("bounds", "core_slice"):
                patch_hashes[key] = [int(v) for v in value]
            else:
                patch_hashes[str(key)] = _array_sha256(value)
    record = _require_mapping(result.get("record"), "result.record")
    reference_pin = {
        "reference_frame_id": record.get("reference_frame_id"),
        "reference_frame_role": record.get("reference_frame_role"),
        "bookkeeping_reference_frame_id": record.get(
            "bookkeeping_reference_frame_id"
        ),
        "cell_gauge_mode": record.get("cell_gauge_mode"),
        "excluded": record.get("excluded"),
    }
    geometry = {
        "patch_bounds": record.get("patch_bounds"),
        "patch_shape_hw": record.get("patch_shape_hw"),
        "bounds": (
            [int(v) for v in patch_map["bounds"]]
            if patch is not None and "bounds" in _mapping(patch)
            else None
        ),
        "core_slice": (
            [int(v) for v in patch_map["core_slice"]]
            if patch is not None and "core_slice" in _mapping(patch)
            else None
        ),
    }
    status = str(result.get("status", "")).strip().lower()
    backend = str(
        result.get("backend_used")
        or record.get("backend_used")
        or "unknown"
    )
    payload_sha256 = hashlib.sha256(
        _canonical_json(
            {
                "status": status,
                "core": core_hashes,
                "patch": patch_hashes,
                "reference_pin": reference_pin,
                "geometry": geometry,
                "backend": backend,
                "n_frames": int(result.get("n_frames", 0) or 0),
                "total_bytes": int(result.get("total_bytes", 0) or 0),
            }
        )
    ).hexdigest()
    return {
        "status": status,
        "core_hashes": core_hashes,
        "patch_hashes": patch_hashes,
        "reference_pin": reference_pin,
        "geometry": geometry,
        "backend": backend,
        "payload_sha256": payload_sha256,
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


@dataclass(frozen=True)
class AcceptanceReceipt:
    """Lightweight hand-off: no payload array is referenced here."""

    cell_id: str
    status: str
    backend: str
    plan_generation: int
    attempt_id: str
    payload_sha256: str
    payload_bytes: int
    accepted: bool
    idempotent: bool = False
    reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "schema": RECEIPT_SCHEMA,
            "cell_id": self.cell_id,
            "status": self.status,
            "backend": self.backend,
            "plan_generation": int(self.plan_generation),
            "attempt_id": self.attempt_id,
            "payload_sha256": self.payload_sha256,
            "payload_bytes": int(self.payload_bytes),
            "accepted": bool(self.accepted),
            "idempotent": bool(self.idempotent),
            "reason": self.reason,
        }


@dataclass
class RetainedCell:
    """One accepted payload owned by the registry until explicit disposal."""

    cell_id: str
    status: str
    backend: str
    plan_generation: int
    attempt_id: str
    payload_sha256: str
    core_hashes: dict[str, str]
    patch_hashes: dict[str, Any] | None
    reference_pin: dict[str, Any]
    geometry: dict[str, Any]
    logical_bytes: int
    retained_bytes: int
    buffer_count: int
    sequence: int
    payload: Mapping[str, Any] | None = field(default=None, repr=False)
    released: bool = False
    released_bytes: int = 0

    def metadata(self) -> dict:
        return {
            "schema": SCHEMA,
            "cell_id": self.cell_id,
            "status": self.status,
            "backend": self.backend,
            "plan_generation": int(self.plan_generation),
            "attempt_id": self.attempt_id,
            "payload_sha256": self.payload_sha256,
            "core_sha256": dict(self.core_hashes),
            "patch_sha256": (
                dict(self.patch_hashes)
                if self.patch_hashes is not None
                else None
            ),
            "reference_pin": dict(self.reference_pin),
            "geometry": dict(self.geometry),
            "logical_bytes": int(self.logical_bytes),
            "retained_bytes": int(self.retained_bytes),
            "buffer_count": int(self.buffer_count),
            "sequence": int(self.sequence),
            "released": bool(self.released),
            "released_bytes": int(self.released_bytes),
        }


class CooperativeStopChannel:
    """Same-process request/ack stop signalling (thread events + lock).

    This object is intentionally NOT picklable: it contains a thread lock and
    plain process-local metadata. Use :class:`SpawnStopChannel` for a real
    process boundary. A pickling attempt fails with a clear error instead of a
    confusing ``_thread.lock`` message.
    """

    def __init__(self, *, clock=time.monotonic) -> None:
        self._clock = clock
        self._requested = threading.Event()
        self._acked = threading.Event()
        self._lock = threading.Lock()
        self._reason: str | None = None
        self._requested_at: float | None = None
        self._ack_state: str | None = None
        self._ack_detail: str | None = None
        self._acked_at: float | None = None

    def request(self, reason: str = "stop requested") -> bool:
        with self._lock:
            first = not self._requested.is_set()
            if first:
                self._reason = str(reason)
                self._requested_at = float(self._clock())
                self._requested.set()
            return bool(first)

    @property
    def requested(self) -> bool:
        return bool(self._requested.is_set())

    @property
    def acknowledged(self) -> bool:
        return bool(self._acked.is_set())

    def acknowledge(self, state: str, detail: str | None = None) -> bool:
        with self._lock:
            if self._acked.is_set():
                return False
            self._ack_state = str(state)
            self._ack_detail = None if detail is None else str(detail)
            self._acked_at = float(self._clock())
            self._acked.set()
            return True

    def wait_ack(self, timeout: float | None = None) -> bool:
        return bool(self._acked.wait(timeout))

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "transport": "thread",
                "requested": bool(self._requested.is_set()),
                "requested_reason": self._reason,
                "requested_monotonic_s": self._requested_at,
                "acknowledged": bool(self._acked.is_set()),
                "ack_state": self._ack_state,
                "ack_detail": self._ack_detail,
                "ack_monotonic_s": self._acked_at,
            }

    def __reduce__(self):
        raise TypeError(
            "CooperativeStopChannel is process-local (thread lock); use "
            "SpawnStopChannel.create(ctx) across processes"
        )


class SpawnStopChannel:
    """Spawn-safe stop request/ack endpoint.

    Holds only multiprocessing primitives (events, shared fixed-size byte
    metadata, and a multiprocessing lock), so it survives ``spawn`` Process
    arguments. The registry and payload arrays never cross this boundary.
    """

    REASON_BYTES = 256
    STATE_BYTES = 64
    DETAIL_BYTES = 512

    def __init__(self, requested, acked, lock, reason, ack_state, ack_detail):
        self._requested = requested
        self._acked = acked
        self._lock = lock
        self._reason = reason
        self._ack_state = ack_state
        self._ack_detail = ack_detail

    @classmethod
    def create(cls, ctx) -> "SpawnStopChannel":
        return cls(
            requested=ctx.Event(),
            acked=ctx.Event(),
            lock=ctx.Lock(),
            reason=ctx.Array("c", cls.REASON_BYTES),
            ack_state=ctx.Array("c", cls.STATE_BYTES),
            ack_detail=ctx.Array("c", cls.DETAIL_BYTES),
        )

    @staticmethod
    def _write(buf, text: str) -> None:
        data = str(text).encode("utf-8", "replace")[: max(0, len(buf) - 1)]
        buf[: len(data)] = data
        buf[len(data)] = b"\x00"

    @staticmethod
    def _read(buf) -> str:
        raw = bytes(buf[:]).split(b"\x00", 1)[0]
        return raw.decode("utf-8", "replace")

    def request(self, reason: str = "stop requested") -> bool:
        with self._lock:
            first = not self._requested.is_set()
            if first:
                self._write(self._reason, reason)
                self._requested.set()
            return bool(first)

    @property
    def requested(self) -> bool:
        return bool(self._requested.is_set())

    @property
    def acknowledged(self) -> bool:
        return bool(self._acked.is_set())

    def acknowledge(self, state: str, detail: str | None = None) -> bool:
        with self._lock:
            if self._acked.is_set():
                return False
            self._write(self._ack_state, state)
            self._write(self._ack_detail, "" if detail is None else detail)
            self._acked.set()
            return True

    def wait_request(self, timeout: float | None = None) -> bool:
        return bool(self._requested.wait(timeout))

    def wait_ack(self, timeout: float | None = None) -> bool:
        return bool(self._acked.wait(timeout))

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "transport": "spawn",
                "requested": bool(self._requested.is_set()),
                "requested_reason": self._read(self._reason),
                "acknowledged": bool(self._acked.is_set()),
                "ack_state": self._read(self._ack_state),
                "ack_detail": self._read(self._ack_detail),
            }


class RunResultRegistry:
    """Run-owned, thread-safe registry of accepted cell payloads."""

    def __init__(
        self,
        *,
        run_id: str,
        expected_cell_ids: Iterable[str],
        gauge_mode: str | None = None,
        max_events: int = DEFAULT_MAX_EVENTS,
        stop_channel=None,
        clock=time.monotonic,
    ) -> None:
        expected = [str(cid) for cid in expected_cell_ids]
        if len(set(expected)) != len(expected):
            raise ValueError("expected_cell_ids contains duplicates")
        if gauge_mode is not None and gauge_mode not in GAUGE_MODES:
            raise ValueError(f"gauge_mode must be one of {GAUGE_MODES}")
        if isinstance(max_events, bool) or not isinstance(max_events, int):
            raise TypeError("max_events must be a strict int")
        if int(max_events) < 1:
            raise ValueError("max_events must be >= 1")
        self.run_id = str(run_id)
        self._expected: tuple[str, ...] = tuple(expected)
        self._expected_set = frozenset(expected)
        self._gauge_mode = gauge_mode
        self._lock = threading.RLock()
        self._cells: dict[str, RetainedCell] = {}
        self._order: list[str] = []
        self._generation = 0
        self._sequence = 0
        self._logical_bytes = 0
        self._retained_bytes = 0
        self._memmap_bytes = 0
        self._buffer_sizes: dict[int, int] = {}
        self._buffer_refs: dict[int, int] = {}
        self._empty_count = 0
        self._released_count = 0
        self._released_bytes = 0
        self._disposed = False
        self._dispose_summary: dict | None = None
        self._events: deque[dict] = deque(maxlen=int(max_events))
        self._events_dropped = 0
        self._max_events = int(max_events)
        self._clock = clock
        self.stop = (
            CooperativeStopChannel() if stop_channel is None else stop_channel
        )
        self._journal(
            "registry_created",
            detail={
                "run_id": self.run_id,
                "expected_cells": len(self._expected),
                "gauge_mode": self._gauge_mode,
            },
        )

    # -- journal -----------------------------------------------------------

    def _journal(self, event: str, **fields: Any) -> None:
        record = {
            "sequence": int(self._events[-1]["sequence"]) + 1
            if self._events
            else 1,
            "event": str(event),
            "monotonic_ns": int(self._clock() * 1_000_000_000),
        }
        record.update(fields)
        if len(self._events) == self._events.maxlen:
            self._events_dropped += 1
        self._events.append(record)

    def journal_events(self, limit: int | None = None) -> list[dict]:
        with self._lock:
            events = list(self._events)
        if limit is not None:
            events = events[-int(limit):]
        return events

    # -- acceptance --------------------------------------------------------

    def accept(
        self,
        result: Mapping[str, Any],
        *,
        backend: str | None,
        plan_generation: int,
        attempt_id: str | None = None,
    ) -> AcceptanceReceipt:
        """Validate, stage and atomically accept ONE cell result.

        ``plan_generation`` is REQUIRED and must be the generation captured when
        the task was dispatched; the registry never adopts the current
        generation implicitly. Validation, fingerprinting and buffer accounting
        happen before the lock, so any failure leaves the registry unchanged.
        """
        if not isinstance(result, Mapping):
            raise AcceptanceError(
                f"cell result must be a mapping, got {type(result).__name__}"
            )
        cell_id = result.get("cell_id")
        if not isinstance(cell_id, str) or not cell_id:
            raise AcceptanceError(
                f"cell result has no stable cell_id: {cell_id!r}"
            )
        raw_status = str(result.get("status", "")).strip().lower()
        status = _CELL_STATUS_MAP.get(raw_status)
        if status is None:
            raise AcceptanceError(
                f"cell {cell_id} has unsupported status "
                f"{result.get('status')!r}"
            )
        if not _is_int(plan_generation):
            raise AcceptanceError(
                "plan_generation must be a strict int captured at dispatch"
            )
        gen = int(plan_generation)
        if gen < 0:
            raise AcceptanceError("plan_generation must be >= 0")

        # --- staging (no registry mutation) --------------------------------
        if status == STATUS_COMPLETE:
            validated = validate_complete_payload(
                result, gauge_mode=self._gauge_mode
            )
        else:
            validated = validate_empty_payload(
                result, gauge_mode=self._gauge_mode
            )
        fingerprint = fingerprint_result(result)
        if status == STATUS_EMPTY_VALIDATED:
            fingerprint["geometry"]["n_contributors"] = (
                validated["geometry"].get("n_contributors")
            )
            fingerprint["geometry"]["gauge_status"] = (
                validated["geometry"].get("gauge_status")
            )
        accounting = _buffer_accounting(result)
        attempt = (
            str(attempt_id)
            if attempt_id is not None
            else f"attempt-{fingerprint['payload_sha256'][:16]}"
        )
        record = _mapping(result.get("record"))
        resolved_backend = str(
            backend
            or result.get("backend_used")
            or record.get("backend_used")
            or fingerprint["backend"]
            or "unknown"
        )
        cell = RetainedCell(
            cell_id=cell_id,
            status=status,
            backend=resolved_backend,
            plan_generation=gen,
            attempt_id=attempt,
            payload_sha256=str(fingerprint["payload_sha256"]),
            core_hashes=dict(fingerprint["core_hashes"]),
            patch_hashes=(
                dict(fingerprint["patch_hashes"])
                if fingerprint["patch_hashes"] is not None
                else None
            ),
            reference_pin=dict(fingerprint["reference_pin"]),
            geometry=dict(fingerprint["geometry"]),
            logical_bytes=int(accounting["logical_bytes"]),
            retained_bytes=int(accounting["retained_bytes"]),
            buffer_count=int(len(accounting["buffers"])),
            payload=result,
            sequence=0,
        )

        # --- small insertion transaction -----------------------------------
        with self._lock:
            if self._disposed:
                raise RegistryClosedError("registry has been disposed")
            if cell_id not in self._expected_set:
                self._journal(
                    "accept_rejected",
                    cell_id=cell_id,
                    reason="unexpected_cell",
                    status=status,
                )
                raise UnexpectedCellError(
                    f"cell {cell_id} is not part of run {self.run_id}"
                )
            if gen < self._generation:
                self._journal(
                    "accept_rejected",
                    cell_id=cell_id,
                    reason="stale_generation",
                    generation=gen,
                    current_generation=int(self._generation),
                    attempt_id=attempt,
                )
                raise StaleGenerationError(
                    f"cell {cell_id} generation {gen} < current "
                    f"{self._generation}"
                )
            if gen > self._generation:
                self._journal(
                    "accept_rejected",
                    cell_id=cell_id,
                    reason="future_generation",
                    generation=gen,
                    current_generation=int(self._generation),
                    attempt_id=attempt,
                )
                raise FutureGenerationError(
                    f"cell {cell_id} generation {gen} > authorised "
                    f"{self._generation}"
                )
            existing = self._cells.get(cell_id)
            if existing is not None:
                same = (
                    existing.payload_sha256 == cell.payload_sha256
                    and existing.backend == cell.backend
                    and existing.plan_generation == cell.plan_generation
                    and existing.attempt_id == cell.attempt_id
                    and existing.status == cell.status
                )
                if same:
                    self._journal(
                        "accept_idempotent",
                        cell_id=cell_id,
                        attempt_id=attempt,
                        status=status,
                        payload_sha256=cell.payload_sha256,
                    )
                    return AcceptanceReceipt(
                        cell_id=cell_id,
                        status=existing.status,
                        backend=existing.backend,
                        plan_generation=existing.plan_generation,
                        attempt_id=existing.attempt_id,
                        payload_sha256=existing.payload_sha256,
                        payload_bytes=existing.retained_bytes,
                        accepted=True,
                        idempotent=True,
                        reason="already_accepted_identical",
                    )
                self._journal(
                    "accept_rejected",
                    cell_id=cell_id,
                    reason="duplicate_conflict",
                    attempt_id=attempt,
                    existing_attempt=existing.attempt_id,
                    status=status,
                    existing_status=existing.status,
                    payload_sha256=cell.payload_sha256,
                    existing_payload_sha256=existing.payload_sha256,
                )
                raise DuplicateCellError(
                    f"cell {cell_id} already accepted with a different "
                    f"status/payload/attempt ({existing.status} vs {status}); "
                    "refused to overwrite"
                )
            self._sequence += 1
            cell.sequence = int(self._sequence)
            self._cells[cell_id] = cell
            self._order.append(cell_id)
            for key, entry in accounting["buffers"].items():
                if key in self._buffer_sizes:
                    self._buffer_refs[key] = int(self._buffer_refs.get(key, 1)) + 1
                    continue
                self._buffer_sizes[key] = int(entry["bytes"])
                self._buffer_refs[key] = 1
                self._retained_bytes += int(entry["bytes"])
                if entry["kind"] in (
                    "memmap", "memmap_mapping", "memmap_view"
                ):
                    self._memmap_bytes += int(entry["bytes"])
            self._logical_bytes += int(cell.logical_bytes)
            if cell.status == STATUS_EMPTY_VALIDATED:
                self._empty_count += 1
            self._journal(
                "accept",
                cell_id=cell_id,
                status=cell.status,
                backend=cell.backend,
                generation=cell.plan_generation,
                attempt_id=cell.attempt_id,
                payload_sha256=cell.payload_sha256,
                logical_bytes=int(cell.logical_bytes),
                retained_bytes=int(cell.retained_bytes),
            )
            return AcceptanceReceipt(
                cell_id=cell_id,
                status=cell.status,
                backend=cell.backend,
                plan_generation=cell.plan_generation,
                attempt_id=cell.attempt_id,
                payload_sha256=cell.payload_sha256,
                payload_bytes=cell.retained_bytes,
                accepted=True,
            )

    def advance_generation(
        self, generation: int | None = None, *, reason: str = "plan_transition"
    ) -> int:
        """Advance the fencing generation; accepted payloads are preserved."""
        with self._lock:
            if self._disposed:
                raise RegistryClosedError("registry has been disposed")
            target = (
                int(self._generation) + 1
                if generation is None
                else int(generation)
            )
            if target <= self._generation:
                raise ValueError(
                    f"new generation {target} must exceed {self._generation}"
                )
            previous = int(self._generation)
            self._generation = int(target)
            self._journal(
                "generation_advanced",
                previous_generation=previous,
                generation=int(self._generation),
                reason=str(reason),
            )
            return int(self._generation)

    def dispatch_generation(self) -> int:
        """Capture this generation when a task is dispatched."""
        with self._lock:
            return int(self._generation)

    # -- cooperative stop --------------------------------------------------

    def request_stop(self, reason: str = "stop requested") -> bool:
        first = self.stop.request(reason)
        with self._lock:
            self._journal(
                "stop_requested",
                reason=str(reason),
                first=bool(first),
            )
        return bool(first)

    @property
    def stop_requested(self) -> bool:
        return bool(self.stop.requested)

    def acknowledge_stop(
        self, state: str, detail: str | None = None
    ) -> bool:
        changed = self.stop.acknowledge(state, detail)
        with self._lock:
            self._journal(
                "stop_acknowledged",
                state=str(state),
                detail=None if detail is None else str(detail),
                changed=bool(changed),
            )
        return bool(changed)

    # -- read-side ---------------------------------------------------------

    def get(self, cell_id: str) -> Mapping[str, Any] | None:
        with self._lock:
            cell = self._cells.get(str(cell_id))
            return None if cell is None else cell.payload

    def release_payload(
        self,
        cell_id: str,
        *,
        expected_payload_sha256: str | None = None,
    ) -> dict:
        """Release ONE cell's large arrays after a durable commit (idempotent).

        The accepted record, its digest and its counts stay; only the payload
        reference and its uniquely-owned buffers are dropped. A shared backing
        allocation is decremented by reference and counted once, so a release
        can never make ``retained_bytes`` lie.
        """
        cid = str(cell_id)
        with self._lock:
            if self._disposed:
                raise RegistryClosedError("registry has been disposed")
            cell = self._cells.get(cid)
            if cell is None:
                raise UnexpectedCellError(
                    f"cell {cid} was never accepted by run {self.run_id}"
                )
            if expected_payload_sha256 is not None and (
                str(expected_payload_sha256) != cell.payload_sha256
            ):
                raise AcceptanceError(
                    f"cell {cid} release refused: durable payload digest "
                    f"{expected_payload_sha256} != accepted "
                    f"{cell.payload_sha256}"
                )
            if cell.released:
                return {
                    "schema": SCHEMA,
                    "cell_id": cid,
                    "released": False,
                    "already_released": True,
                    "payload_sha256": cell.payload_sha256,
                    "released_bytes": 0,
                    "cell_released_bytes": int(cell.released_bytes),
                    "retained_bytes": int(self._retained_bytes),
                }
            accounting = _buffer_accounting(cell.payload)
            freed = 0
            for key, entry in accounting["buffers"].items():
                refs = int(self._buffer_refs.get(key, 0)) - 1
                if refs > 0:
                    self._buffer_refs[key] = refs
                    continue
                self._buffer_refs.pop(key, None)
                size = int(self._buffer_sizes.pop(key, 0))
                freed += size
                if entry["kind"] in (
                    "memmap", "memmap_mapping", "memmap_view"
                ):
                    self._memmap_bytes = max(0, self._memmap_bytes - size)
            self._retained_bytes = max(0, self._retained_bytes - freed)
            cell.released = True
            cell.released_bytes = int(freed)
            cell.payload = None
            self._released_count += 1
            self._released_bytes += int(freed)
            self._journal(
                "payload_released",
                cell_id=cid,
                attempt_id=cell.attempt_id,
                payload_sha256=cell.payload_sha256,
                released_bytes=int(freed),
                retained_bytes=int(self._retained_bytes),
            )
            return {
                "schema": SCHEMA,
                "cell_id": cid,
                "released": True,
                "already_released": False,
                "payload_sha256": cell.payload_sha256,
                "released_bytes": int(freed),
                "cell_released_bytes": int(freed),
                "retained_bytes": int(self._retained_bytes),
            }

    def receipt(self, cell_id: str) -> AcceptanceReceipt | None:
        with self._lock:
            cell = self._cells.get(str(cell_id))
            if cell is None:
                return None
            return AcceptanceReceipt(
                cell_id=cell.cell_id,
                status=cell.status,
                backend=cell.backend,
                plan_generation=cell.plan_generation,
                attempt_id=cell.attempt_id,
                payload_sha256=cell.payload_sha256,
                payload_bytes=cell.retained_bytes,
                accepted=True,
            )

    def metadata(self, cell_id: str) -> dict | None:
        with self._lock:
            cell = self._cells.get(str(cell_id))
            return None if cell is None else cell.metadata()

    def accepted_cell_ids(self) -> list[str]:
        with self._lock:
            return list(self._order)

    def pending_cell_ids(self) -> list[str]:
        with self._lock:
            accepted = set(self._cells)
            return [cid for cid in self._expected if cid not in accepted]

    def ordered_payloads(
        self, cell_ids: Sequence[str] | None = None
    ) -> list[Mapping[str, Any]]:
        with self._lock:
            if self._disposed:
                raise RegistryClosedError("registry has been disposed")
            ordered = self._expected if cell_ids is None else tuple(
                str(cid) for cid in cell_ids
            )
            missing = [cid for cid in ordered if cid not in self._cells]
            if missing:
                raise RunRegistryError(
                    "requested payloads include unaccepted cells: "
                    + ", ".join(missing)
                )
            released = [
                cid
                for cid in ordered
                if self._cells[cid].payload is None
            ]
            if released:
                raise PayloadReleasedError(
                    "payload(s) already released after a durable commit: "
                    + ", ".join(released)
                    + "; re-read them from the run store"
                )
            return [self._cells[cid].payload for cid in ordered]

    def receipts(self) -> list[AcceptanceReceipt]:
        with self._lock:
            out: list[AcceptanceReceipt] = []
            for cid in self._order:
                cell = self._cells[cid]
                out.append(
                    AcceptanceReceipt(
                        cell_id=cell.cell_id,
                        status=cell.status,
                        backend=cell.backend,
                        plan_generation=cell.plan_generation,
                        attempt_id=cell.attempt_id,
                        payload_sha256=cell.payload_sha256,
                        payload_bytes=cell.retained_bytes,
                        accepted=True,
                    )
                )
            return out

    @property
    def retained_bytes(self) -> int:
        with self._lock:
            return int(self._retained_bytes)

    @property
    def logical_bytes(self) -> int:
        with self._lock:
            return int(self._logical_bytes)

    @property
    def generation(self) -> int:
        with self._lock:
            return int(self._generation)

    @property
    def disposed(self) -> bool:
        with self._lock:
            return bool(self._disposed)

    def counts(self) -> dict:
        with self._lock:
            complete = sum(
                1
                for cell in self._cells.values()
                if cell.status == STATUS_COMPLETE
            )
            return {
                "expected": len(self._expected),
                "accepted": len(self._cells),
                "complete": int(complete),
                "empty_validated": int(self._empty_count),
                "pending": len(self._expected) - len(self._cells),
                "logical_bytes": int(self._logical_bytes),
                "retained_bytes": int(self._retained_bytes),
                "memmap_bytes": int(self._memmap_bytes),
                "retained_buffers": len(self._buffer_sizes),
                "released_cells": int(self._released_count),
                "released_bytes": int(self._released_bytes),
                "generation": int(self._generation),
                "disposed": bool(self._disposed),
            }

    def snapshot(self) -> dict:
        """Lightweight audit state: no payload arrays, no payload dicts."""
        with self._lock:
            cells = [
                self._cells[cid].metadata() for cid in self._order
            ]
            return {
                "schema": SCHEMA,
                "run_id": self.run_id,
                "gauge_mode": self._gauge_mode,
                "counts": self.counts(),
                "accepted_cell_ids": list(self._order),
                "pending_cell_ids": [
                    cid for cid in self._expected if cid not in self._cells
                ],
                "cells": cells,
                "stop": self.stop.snapshot(),
                "journal": {
                    "events": len(self._events),
                    "dropped": int(self._events_dropped),
                    "max_events": int(self._max_events),
                },
                "disposed": bool(self._disposed),
                "accounting_note": (
                    "logical_bytes sums plane nbytes; retained_bytes counts "
                    "unique backing allocations once and is NOT added to "
                    "already-measured parent RSS"
                ),
            }

    # -- disposal ----------------------------------------------------------

    def dispose(self) -> dict:
        """Explicitly release every accepted payload (idempotent)."""
        with self._lock:
            if self._disposed:
                return {
                    "schema": SCHEMA,
                    "run_id": self.run_id,
                    "released_cells": 0,
                    "released_logical_bytes": 0,
                    "released_retained_bytes": 0,
                    "already_disposed": True,
                }
            released_cells = len(self._cells)
            released_logical = int(self._logical_bytes)
            released_retained = int(self._retained_bytes)
            self._cells.clear()
            self._order.clear()
            self._buffer_sizes.clear()
            self._buffer_refs.clear()
            self._released_count = 0
            self._released_bytes = 0
            self._empty_count = 0
            self._logical_bytes = 0
            self._retained_bytes = 0
            self._memmap_bytes = 0
            self._disposed = True
            summary = {
                "schema": SCHEMA,
                "run_id": self.run_id,
                "released_cells": int(released_cells),
                "released_logical_bytes": int(released_logical),
                "released_retained_bytes": int(released_retained),
                "already_disposed": False,
            }
            self._dispose_summary = dict(summary)
            self._journal(
                "registry_disposed",
                released_cells=int(released_cells),
                released_logical_bytes=int(released_logical),
                released_retained_bytes=int(released_retained),
            )
            return dict(summary)

    def __len__(self) -> int:
        with self._lock:
            return len(self._cells)

    def __reduce__(self):
        raise TypeError(
            "RunResultRegistry is coordinator/process-local by design and "
            "cannot be pickled; share only SpawnStopChannel metadata across "
            "a process boundary"
        )
