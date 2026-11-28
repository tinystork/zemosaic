"""ZM-M33 lot 2 — durable completed-tile checkpoints for the external CPU path.

A checkpoint is the **reduced** scientific payload of ONE already reduced
external tile (interior only): science / weight_sum / valid_mask / surviving /
support_w1 / support_w2, the tile counters, the executed block geometry and the
exact rejection-sink deltas. It is NOT a second copy of the N-axis tile spool
and never stores the full ``(N, H, W, C)`` rejection mask.

Storage contract
----------------
* one explicit Cell directory under a parent-owned root;
* a typed ``checkpoint.json`` manifest carrying the full compatible identity
  (source order, request/science tokens, gauge digest, backend, tile geometry)
  with per-section digests so a refusal names what changed;
* one directory per canonical tile index:
  ``tiles/<index>/payload.bin`` + ``tiles/<index>/commit.json``;
* payload is written to ``tmp/``, flushed, fsynced, closed, then atomically
  renamed; the JSON marker is published atomically **after** it. A payload
  without a valid marker is never a PASS (torn/partial = recompute);
* every array entry carries dtype/shape/offset/nbytes/sha256 and the marker
  carries the payload sha256 plus its own sha256. No pickle is ever read;
* a single-writer OS lock (``run_store.StoreLock``) with a Windows-safe owner
  record: a live owner is refused, a provably dead owner is recovered;
* a corrupt/truncated tile (payload, marker or hash) is a TILE-level failure:
  the caller logs and recomputes only that tile, preserving every other
  checkpoint. A corrupt Cell manifest is a loud refusal because identity cannot
  be established safely.

Cleanup is deliberately NOT performed here on success: the final durable Cell
commit (parent-owned) is the only authority. The caller removes the Cell
checkpoint directory with :func:`cleanup_cell` after ``require_commit``.
"""

from __future__ import annotations

import hashlib
import json
import re
import os
import shutil
import time
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

from zemosaic.core.zegrid import run_store as zrs

__all__ = [
    "CHECKPOINT_SCHEMA",
    "CHECKPOINT_VERSION",
    "CELL_MANIFEST_NAME",
    "TILE_CHECKPOINT_DIRNAME",
    "SOURCE_BINDING_SCHEMA",
    "TileCheckpointError",
    "TileCheckpointPathError",
    "TileCheckpointIncompatible",
    "TileCheckpointCorrupt",
    "TileCheckpointLocked",
    "TileCheckpointStore",
    "inspect_owned_checkpoint_root",
    "build_tile_identity",
    "source_binding_digest",
    "cleanup_cell",
    "interior_arrays",
    "sink_delta_between",
    "apply_sink_delta",
]

CHECKPOINT_SCHEMA = "zemosaic-zegrid-tile-checkpoint-v1"
CELL_MANIFEST_SCHEMA = "zemosaic-zegrid-tile-checkpoint-cell-v1"
SOURCE_BINDING_SCHEMA = "zemosaic-zegrid-tile-checkpoint-source-v1"
CHECKPOINT_VERSION = 1
CELL_MANIFEST_NAME = "checkpoint.json"
#: Product-owned durable tile-checkpoint root under the output directory.
TILE_CHECKPOINT_DIRNAME = ".zegrid-tile-checkpoints"
TILE_MARKER_NAME = "commit.json"
TILE_PAYLOAD_NAME = "payload.bin"


class TileCheckpointError(RuntimeError):
    """Base class for the checkpoint contract."""


class TileCheckpointPathError(TileCheckpointError, ValueError):
    """A path is unsafe, a symlink, or escapes the explicit root."""


class TileCheckpointIncompatible(TileCheckpointError):
    """The stored Cell identity is incompatible with this attempt."""

    def __init__(self, message: str, *, reasons=None) -> None:
        super().__init__(message)
        self.reasons = list(reasons or ())


class TileCheckpointCorrupt(TileCheckpointError):
    """One tile checkpoint (or its payload) is not trustworthy."""


class TileCheckpointLocked(TileCheckpointError):
    """A live owner holds the Cell checkpoint lock."""


#: Fixed array order inside one payload. ``int64``/``bool`` are platform-stable
#: in NumPy; every array is stored with its explicit dtype in the marker.
_ARRAYS = (
    ("science", "float32", 3),
    ("weight_sum", "float64", 3),
    ("valid_mask", "bool", 3),
    ("surviving", "int64", 3),
    ("support_w1", "float64", 2),
    ("support_w2", "float64", 2),
)
_ARRAY_NAMES = tuple(name for name, _dtype, _ndim in _ARRAYS)


# ---------------------------------------------------------------------------
# Canonical JSON / hashing helpers
# ---------------------------------------------------------------------------

def _canonical_json(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_digest(value) -> str:
    arr = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(str(tuple(int(v) for v in arr.shape)).encode("utf-8"))
    digest.update(arr.tobytes(order="C"))
    return digest.hexdigest()


def _write_array_stream(fd: int, arr: np.ndarray, *digests) -> None:
    """Write ``arr`` in C order, updating every supplied digest.

    A sliced interior plane is usually NOT C-contiguous. Streaming one leading
    row at a time keeps the checkpoint copy bounded to a single tile row and
    avoids materialising a second full tile result set in RAM.
    """
    def _emit(data) -> None:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0:  # pragma: no cover - defensive
                raise OSError("short write")
            view = view[written:]
        for digest in digests:
            digest.update(data)

    if arr.flags.c_contiguous:
        _emit(memoryview(arr).cast("B"))
        return
    if arr.ndim == 0:
        _emit(np.ascontiguousarray(arr).tobytes(order="C"))
        return
    for index in range(int(arr.shape[0])):
        row = np.ascontiguousarray(arr[index])
        _emit(memoryview(row).cast("B"))


def _json_safe(value):
    """Bounded JSON-safe projection (never an array payload)."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in sorted(value.items())}
    if is_dataclass(value):
        return _json_safe(asdict(value))
    return str(value)


def _safe_component(value, label: str) -> str:
    try:
        return zrs.safe_name(value, label)
    except zrs.StorePathError as exc:
        raise TileCheckpointPathError(str(exc)) from exc


def _contained_child(root: Path, *parts: str) -> Path:
    relative = "/".join(str(part) for part in parts)
    try:
        return zrs.confined_path(Path(root), relative, "checkpoint path")
    except zrs.StorePathError as exc:
        raise TileCheckpointPathError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def _request_section(request) -> dict:
    taper = request.taper
    if isinstance(taper, (str, bytes)):
        taper_block = {"kind": str(taper)}
    elif taper is None:
        taper_block = {"kind": None}
    else:
        try:
            arr = np.asarray(taper)
        except Exception:  # noqa: BLE001 - explicit identity failure
            taper_block = {"kind": type(taper).__name__}
        else:
            taper_block = {
                "kind": "array",
                "dtype": str(arr.dtype),
                "shape": [int(v) for v in arr.shape],
                "sha256": _array_digest(arr),
            }
    return {
        "normalization": str(request.normalization),
        "weighting": str(request.weighting),
        "rejection": str(request.rejection),
        "combine": str(request.combine),
        "reference_index": (
            None if request.reference_index is None
            else int(request.reference_index)
        ),
        "taper": taper_block,
        "taper_px": float(request.taper_px),
        "taper_floor": float(request.taper_floor),
        "sigma_low": float(request.sigma_low),
        "sigma_high": float(request.sigma_high),
        "max_iters": int(request.max_iters),
        "winsor_limit_low": float(request.winsor_limit_low),
        "winsor_limit_high": float(request.winsor_limit_high),
        "equalize_rgb": bool(request.equalize_rgb),
    }


def _gauge_section(p1) -> dict:
    exclusions = []
    for item in tuple(getattr(p1, "exclusions", ()) or ()):
        exclusions.append(
            [
                int(getattr(item, "index", -1)),
                str(getattr(item, "stage", "")),
                str(getattr(item, "reason", "")),
                str(getattr(item, "detail", "")),
            ]
        )
    return {
        "reference_index": int(p1.reference_index),
        "coefficients": _array_digest(p1.coefficients),
        "norm_active": _array_digest(p1.norm_active),
        "weights": _array_digest(p1.weights),
        "weight_active": _array_digest(p1.weight_active),
        "exclusions": exclusions,
        "n_frames": int(p1.n_frames),
        "height": int(p1.height),
        "width": int(p1.width),
        "channels": int(p1.channels),
    }


def build_tile_identity(
    *,
    provider,
    request,
    p1,
    cell_id: str,
    tile_size,
    halo_px: int,
    tiles,
    backend: str,
    source_binding_digest: str,
    column_batch: int,
    resolved_batch: dict,
) -> dict:
    """Build the frozen, cross-attempt stable Cell identity.

    ``tiles`` is the canonical ``[(y0, y1, x0, x1), ...]`` order produced by the
    exact ``_iter_tiles`` iterator the executor uses. ``source_binding_digest``
    is a STRONG caller-supplied content binding (pixels + WCS/transforms +
    canvas/patch + mask/calibration provenance): labels alone are never enough.
    """
    if not isinstance(source_binding_digest, str) or not source_binding_digest:
        raise TileCheckpointError(
            "a durable tile checkpoint requires a non-empty frozen source "
            "binding digest (pixels/WCS/canvas/patch/gauge); refusing a "
            "label-only identity"
        )
    frame_ids = getattr(provider, "frame_ids", None)
    source = {
        "n_frames": int(provider.n_frames),
        "height": int(provider.height),
        "width": int(provider.width),
        "channels": int(provider.channels),
        "original_mono": bool(getattr(provider, "original_mono", False)),
        "original_ndim": int(getattr(provider, "original_ndim", 2)),
        "original_shape": [
            int(v) for v in tuple(getattr(provider, "original_shape", ()))
        ],
        "frame_ids": (
            None if frame_ids is None else [str(v) for v in frame_ids]
        ),
    }
    sections = {
        "cell": {"cell_id": str(cell_id)},
        "source": source,
        "source_binding": {
            "schema": SOURCE_BINDING_SCHEMA,
            "digest": str(source_binding_digest),
        },
        "request": _request_section(request),
        "gauge": _gauge_section(p1),
        "backend": {"requested": str(backend)},
        "execution": {
            "column_batch": int(column_batch),
            "rows_per_batch": int(resolved_batch["rows_per_batch"]),
            "cols_per_batch": int(resolved_batch["cols_per_batch"]),
            "batch_positions": int(resolved_batch["batch_positions"]),
        },
        "geometry": {
            "height": int(provider.height),
            "width": int(provider.width),
            "tile_size": _json_safe(tile_size),
            "halo_px": int(halo_px),
            "tiles": [
                [int(y0), int(y1), int(x0), int(x1)]
                for (y0, y1, x0, x1) in tiles
            ],
        },
    }
    section_digests = {
        name: _sha256_bytes(_canonical_json(section))
        for name, section in sections.items()
    }
    digest = _sha256_bytes(_canonical_json(section_digests))
    return {
        "schema": CHECKPOINT_SCHEMA,
        "version": int(CHECKPOINT_VERSION),
        "cell_id": str(cell_id),
        "identity_digest": digest,
        "section_digests": section_digests,
        "sections": sections,
    }


def source_binding_digest(binding: dict) -> str:
    """Canonical digest of a frozen source binding (pixels/WCS/canvas/patch)."""
    if not isinstance(binding, dict):
        raise TileCheckpointError("source binding must be a mapping")
    payload = dict(binding)
    payload.setdefault("schema", SOURCE_BINDING_SCHEMA)
    return _sha256_bytes(_canonical_json(payload))


# ---------------------------------------------------------------------------
# Array / payload helpers
# ---------------------------------------------------------------------------

def interior_arrays(tile: dict, *, iy0: int, ix0: int, h_int: int,
                    w_int: int) -> dict:
    """Slice the reduced tile planes to the exact interior window.

    Views only: :meth:`TileCheckpointStore.save_tile` streams each array in
    bounded row chunks, so checkpointing never materialises a second full tile
    result set in RAM.
    """
    return {
        "science": tile["science"][iy0:iy0 + h_int, ix0:ix0 + w_int, :],
        "weight_sum": tile["weight_sum"][iy0:iy0 + h_int, ix0:ix0 + w_int, :],
        "valid_mask": tile["valid_mask"][iy0:iy0 + h_int, ix0:ix0 + w_int, :],
        "surviving": tile["surviving"][iy0:iy0 + h_int, ix0:ix0 + w_int, :],
        "support_w1": tile["support_w1"],
        "support_w2": tile["support_w2"],
    }


def sink_delta_between(before: dict | None, after: dict | None) -> dict | None:
    """Per-tile additive sink delta + absolute (cumulative) max bytes.

    ``before``/``after`` are ``BoundedCountingRejectionSink.checkpoint()``
    snapshots. The max byte counters are cumulative maxima; storing the
    *after* value and merging with ``max`` on restore reproduces the exact final
    maximum without ever replaying a mask.
    """
    if before is None or after is None:
        return None
    delta = {
        "true_count": int(after["true_count"]) - int(before["true_count"]),
        "written_cells": int(after["written_cells"]) - int(before["written_cells"]),
        "writes": int(after["writes"]) - int(before["writes"]),
        "blocks": int(after["blocks"]) - int(before["blocks"]),
        "max_write_bytes_after": int(after["max_write_bytes"]),
        "max_block_bytes_after": int(after["max_block_bytes"]),
    }
    if any(delta[key] < 0 for key in ("true_count", "written_cells", "writes", "blocks")):
        raise TileCheckpointError(
            "rejection sink counters went backwards for one tile; refusing to "
            "record a checkpoint that cannot be restored exactly"
        )
    return delta


def apply_sink_delta(sink, delta: dict) -> None:
    """Restore ONE tile's exact sink contribution (once, additive)."""
    if sink is None or delta is None:
        return
    method = getattr(sink, "restore_tile_delta", None)
    if not callable(method):
        raise TileCheckpointError(
            "checkpoint reuse requires a rejection sink that supports the "
            "explicit additive 'restore_tile_delta' contract (the product "
            "BoundedCountingRejectionSink does); a dense rejection mask is "
            "refused because it cannot be reconstructed without archiving the "
            "N-axis mask"
        )
    method(dict(delta))


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def cell_directory(base_root, cell_id: str) -> Path:
    """Resolve the explicit per-Cell checkpoint directory under ``base_root``."""
    try:
        root = Path(base_root).expanduser()
    except Exception as exc:  # noqa: BLE001
        raise TileCheckpointPathError(
            f"checkpoint root is not a usable path: {base_root!r} ({exc})"
        ) from exc
    if root.exists() and root.is_symlink():
        raise TileCheckpointPathError(
            f"checkpoint root must not be a symlink: {root}"
        )
    name = _safe_component(cell_id, "cell_id")
    return root / name


class TileCheckpointStore:
    """One Cell's durable tile checkpoints (single writer)."""

    def __init__(
        self,
        root: Path,
        *,
        identity: dict,
        run_id: str | None = None,
        policy: str = "recompute_corrupt",
        session_id: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.identity = dict(identity)
        self.run_id = None if run_id is None else str(run_id)
        self.policy = str(policy)
        if self.policy not in ("recompute_corrupt", "refuse_corrupt"):
            raise TileCheckpointError(
                "checkpoint_policy must be 'recompute_corrupt' or "
                f"'refuse_corrupt', got {policy!r}"
            )
        self.session_id = session_id
        self._lock = None
        self._manifest: dict | None = None
        self._closed = False
        self.reused_tiles = 0
        self.recomputed_tiles = 0
        self.corrupt_recomputed = 0
        self.saved_tiles = 0
        self.write_failures = 0
        self.runtime_complete = False

    # -- open / close ---------------------------------------------------
    def open(self) -> "TileCheckpointStore":
        if self.root.exists() and self.root.is_symlink():
            raise TileCheckpointPathError(
                f"checkpoint Cell root must not be a symlink: {self.root}"
            )
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir():
            raise TileCheckpointPathError(
                f"checkpoint Cell root is not a directory: {self.root}"
            )
        allowed = {"LOCK", "OWNER.json", "tiles", "tmp", CELL_MANIFEST_NAME}
        try:
            foreign = [
                entry.name for entry in self.root.iterdir()
                if entry.name not in allowed
            ]
        except OSError as exc:
            raise TileCheckpointPathError(
                f"cannot inspect checkpoint Cell root {self.root}: {exc}"
            ) from exc
        if foreign:
            raise TileCheckpointPathError(
                "refusing to write tile checkpoints into a directory that "
                f"already holds unrelated content ({', '.join(foreign[:5])}): "
                f"{self.root}"
            )
        (self.root / "tiles").mkdir(exist_ok=True)
        (self.root / "tmp").mkdir(exist_ok=True)
        try:
            self._lock = zrs.StoreLock(
                self.root / "LOCK", session_id=self.session_id
            )
            self._lock.acquire()
        except zrs.StoreLockError as exc:
            raise TileCheckpointLocked(
                f"checkpoint Cell {self.root} is held by a live owner: {exc}"
            ) from exc
        manifest_path = self.root / CELL_MANIFEST_NAME
        try:
            if manifest_path.exists():
                self._manifest = self._read_manifest(manifest_path)
                self._require_compatible(self._manifest)
            else:
                self._manifest = self._new_manifest()
                zrs.atomic_write_bytes(
                    manifest_path,
                    _canonical_json(self._manifest),
                    tmp_dir=self.root / "tmp",
                )
        except BaseException:
            # A refused/invalid manifest must not leak the single-writer lock.
            self.close()
            raise
        return self

    def _new_manifest(self) -> dict:
        record = {
            "schema": CELL_MANIFEST_SCHEMA,
            "version": int(CHECKPOINT_VERSION),
            "cell_id": str(self.identity["cell_id"]),
            "run_id": self.run_id,
            "identity_digest": str(self.identity["identity_digest"]),
            "section_digests": dict(self.identity["section_digests"]),
            "sections": dict(self.identity["sections"]),
            "policy": self.policy,
            "created_unix_ns": int(time.time_ns()),
        }
        record["marker_sha256"] = _sha256_bytes(_canonical_json(record))
        return record

    def _read_manifest(self, path: Path) -> dict:
        try:
            if path.is_symlink():
                raise TileCheckpointPathError(
                    f"checkpoint manifest must not be a symlink: {path}"
                )
            raw = path.read_bytes()
            record = json.loads(raw.decode("utf-8"))
        except TileCheckpointPathError:
            raise
        except Exception as exc:  # noqa: BLE001 - typed corruption
            raise TileCheckpointCorrupt(
                f"checkpoint Cell manifest is not readable JSON: {path} ({exc})"
            ) from exc
        if not isinstance(record, dict):
            raise TileCheckpointCorrupt("checkpoint Cell manifest must be an object")
        if record.get("schema") != CELL_MANIFEST_SCHEMA:
            raise TileCheckpointCorrupt(
                f"checkpoint Cell manifest schema {record.get('schema')!r} is "
                f"not {CELL_MANIFEST_SCHEMA!r}"
            )
        expected = record.get("marker_sha256")
        actual = _sha256_bytes(
            _canonical_json({k: v for k, v in record.items() if k != "marker_sha256"})
        )
        if not isinstance(expected, str) or expected != actual:
            raise TileCheckpointCorrupt(
                "checkpoint Cell manifest SHA-256 mismatch (truncated/corrupted)"
            )
        return record

    def _require_compatible(self, record: dict) -> None:
        stored = str(record.get("identity_digest", ""))
        current = str(self.identity["identity_digest"])
        stored_run = record.get("run_id")
        reasons = []
        if stored != current:
            stored_sections = record.get("section_digests") or {}
            current_sections = self.identity["section_digests"]
            for name in sorted(set(stored_sections) | set(current_sections)):
                if stored_sections.get(name) != current_sections.get(name):
                    reasons.append(
                        {
                            "section": name,
                            "stored": stored_sections.get(name),
                            "current": current_sections.get(name),
                        }
                    )
            raise TileCheckpointIncompatible(
                "the stored tile-checkpoint identity is incompatible with this "
                "attempt: "
                + ", ".join(item["section"] for item in reasons or [{"section": "identity"}])
                + "; refusing reuse instead of silently recomputing or erasing "
                "the stored checkpoints",
                reasons=reasons,
            )
        if str(stored_run) != str(self.run_id):
            raise TileCheckpointIncompatible(
                "the stored tile-checkpoint run binding "
                f"{stored_run!r} differs from this attempt's frozen run "
                f"binding {self.run_id!r}; a durable run id and a store-less "
                "None are NOT interchangeable; refusing reuse",
                reasons=[{"section": "run_id", "stored": stored_run, "current": self.run_id}],
            )

    def close(self) -> bool:
        if self._closed:
            return False
        self._closed = True
        if self._lock is not None:
            self._lock.release()
            self._lock = None
            return True
        return False

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    # -- tile paths -----------------------------------------------------
    def _tile_dir(self, tile_index: int) -> Path:
        index = int(tile_index)
        if index < 1:
            raise TileCheckpointPathError(
                f"tile_index must be >= 1, got {tile_index!r}"
            )
        return _contained_child(self.root, "tiles", f"t{index:06d}")

    # -- load -----------------------------------------------------------
    def load_tile(self, tile_index: int, bounds) -> dict | None:
        """Return a validated reduced tile, ``None`` when absent.

        Raises :class:`TileCheckpointCorrupt` for an existing but untrustworthy
        tile checkpoint (the caller decides recompute vs refuse by policy).
        """
        tile_dir = self._tile_dir(tile_index)
        if not tile_dir.exists():
            return None
        if tile_dir.is_symlink():
            raise TileCheckpointCorrupt(f"tile checkpoint is a symlink: {tile_dir}")
        marker_path = tile_dir / TILE_MARKER_NAME
        if not marker_path.exists():
            # Payload without a marker is an incomplete write, never a PASS.
            return None
        record = self._read_tile_marker(marker_path, tile_index, bounds)
        arrays = self._read_payload(tile_dir, record)
        h_int = int(bounds[1]) - int(bounds[0])
        w_int = int(bounds[3]) - int(bounds[2])
        channels = int(
            (self.identity.get("sections") or {})
            .get("source", {})
            .get("channels", 0)
        )
        expected = {
            "science": (h_int, w_int, channels),
            "weight_sum": (h_int, w_int, channels),
            "valid_mask": (h_int, w_int, channels),
            "surviving": (h_int, w_int, channels),
            "support_w1": (h_int, w_int),
            "support_w2": (h_int, w_int),
        }
        for name, shape in expected.items():
            if tuple(arrays[name].shape) != shape:
                raise TileCheckpointCorrupt(
                    f"tile checkpoint array {name!r} shape "
                    f"{tuple(arrays[name].shape)} != expected {shape}"
                )
        return {
            "arrays": arrays,
            "counters": dict(record["counters"]),
            "batch_geometry": dict(record["batch_geometry"]),
            "sink_delta": (
                None if record.get("sink_delta") is None
                else dict(record["sink_delta"])
            ),
            "backend": (
                None if record.get("backend") is None
                else dict(record["backend"])
            ),
            "bounds": [int(v) for v in record["bounds"]],
        }

    def _read_tile_marker(self, path: Path, tile_index: int, bounds) -> dict:
        try:
            if path.is_symlink():
                raise TileCheckpointCorrupt(
                    f"tile checkpoint marker is a symlink: {path}"
                )
            record = json.loads(path.read_bytes().decode("utf-8"))
        except TileCheckpointCorrupt:
            raise
        except Exception as exc:  # noqa: BLE001
            raise TileCheckpointCorrupt(
                f"tile checkpoint marker is not readable JSON: {path} ({exc})"
            ) from exc
        if not isinstance(record, dict):
            raise TileCheckpointCorrupt("tile checkpoint marker must be an object")
        if record.get("schema") != CHECKPOINT_SCHEMA:
            raise TileCheckpointCorrupt(
                f"tile checkpoint schema {record.get('schema')!r} is not "
                f"{CHECKPOINT_SCHEMA!r}"
            )
        expected = record.get("marker_sha256")
        actual = _sha256_bytes(
            _canonical_json({k: v for k, v in record.items() if k != "marker_sha256"})
        )
        if not isinstance(expected, str) or expected != actual:
            raise TileCheckpointCorrupt(
                "tile checkpoint marker SHA-256 mismatch (truncated/corrupted)"
            )
        if str(record.get("identity_digest")) != str(self.identity["identity_digest"]):
            raise TileCheckpointCorrupt(
                "tile checkpoint identity digest does not match the Cell manifest"
            )
        if int(record.get("tile_index", -1)) != int(tile_index):
            raise TileCheckpointCorrupt(
                f"tile checkpoint index {record.get('tile_index')!r} != "
                f"expected {int(tile_index)}"
            )
        if [int(v) for v in record.get("bounds", [])] != [int(v) for v in bounds]:
            raise TileCheckpointCorrupt(
                f"tile checkpoint bounds {record.get('bounds')!r} != "
                f"expected {list(bounds)!r}"
            )
        for key in ("counters", "batch_geometry", "arrays", "payload"):
            if not isinstance(record.get(key), (dict, type(None))):
                raise TileCheckpointCorrupt(
                    f"tile checkpoint marker field {key!r} is malformed"
                )
        if record.get("arrays") is None or record.get("payload") is None:
            raise TileCheckpointCorrupt("tile checkpoint marker is incomplete")
        if not isinstance(record.get("arrays"), dict):
            raise TileCheckpointCorrupt("tile checkpoint arrays must be an object")
        if not isinstance(record.get("payload"), dict):
            raise TileCheckpointCorrupt("tile checkpoint payload must be an object")
        if not isinstance(record.get("counters"), dict):
            raise TileCheckpointCorrupt("tile checkpoint counters must be an object")
        if not isinstance(record.get("batch_geometry"), dict):
            raise TileCheckpointCorrupt(
                "tile checkpoint batch_geometry must be an object"
            )
        # ZM-M33 lot 3: additive per-tile backend provenance. Absent on legacy
        # CPU markers (the identity already proves the requested backend).
        if record.get("backend") is not None:
            backend = record["backend"]
            if not isinstance(backend, dict):
                raise TileCheckpointCorrupt(
                    "tile checkpoint backend provenance must be an object"
                )
            requested = str(backend.get("requested") or "")
            effective = str(backend.get("effective") or "")
            if requested not in ("cpu", "gpu") or effective not in (
                "cpu", "gpu"
            ):
                raise TileCheckpointCorrupt(
                    "tile checkpoint backend provenance must name cpu/gpu, "
                    f"got requested={requested!r} effective={effective!r}"
                )
        for key in (
            "tile_iters", "initial_count", "rejected_count", "low_n_count",
            "degenerate_count",
        ):
            value = record["counters"].get(key)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TileCheckpointCorrupt(
                    f"tile checkpoint counter {key!r} is missing or not an int"
                )
            if int(value) < 0:
                raise TileCheckpointCorrupt(
                    f"tile checkpoint counter {key!r} is negative"
                )
        for key in (
            "column_batch", "interior_hw", "rows_per_batch",
            "cols_per_batch", "n_blocks", "max_block_positions",
        ):
            if key not in record["batch_geometry"]:
                raise TileCheckpointCorrupt(
                    f"tile checkpoint batch geometry is missing {key!r}"
                )
        return record

    def _read_payload(self, tile_dir: Path, record: dict) -> dict:
        payload = record["payload"]
        file_name = str(payload.get("name", TILE_PAYLOAD_NAME))
        if file_name != TILE_PAYLOAD_NAME:
            raise TileCheckpointCorrupt(
                f"unexpected tile payload name {file_name!r}"
            )
        path = _contained_child(tile_dir, file_name)
        if not path.exists() or path.is_symlink():
            raise TileCheckpointCorrupt(
                f"tile checkpoint payload is missing or a symlink: {path}"
            )
        try:
            size = int(path.stat().st_size)
        except OSError as exc:
            raise TileCheckpointCorrupt(f"cannot stat payload: {path} ({exc})") from exc
        if size != int(payload.get("bytes", -1)):
            raise TileCheckpointCorrupt(
                f"tile checkpoint payload size {size} != recorded "
                f"{payload.get('bytes')!r}"
            )
        if _sha256_file(path) != str(payload.get("sha256")):
            raise TileCheckpointCorrupt(
                "tile checkpoint payload SHA-256 mismatch (corrupted payload)"
            )
        raw = path.read_bytes()
        arrays = {}
        for name, dtype_name, ndim in _ARRAYS:
            entry = record["arrays"].get(name)
            if not isinstance(entry, dict):
                raise TileCheckpointCorrupt(
                    f"tile checkpoint is missing array {name!r}"
                )
            offset = int(entry.get("offset", -1))
            nbytes = int(entry.get("nbytes", -1))
            shape = tuple(int(v) for v in entry.get("shape", ()))
            if offset < 0 or nbytes < 0 or offset + nbytes > len(raw):
                raise TileCheckpointCorrupt(
                    f"tile checkpoint array {name!r} lies outside the payload"
                )
            if len(shape) != ndim or any(v < 0 for v in shape):
                raise TileCheckpointCorrupt(
                    f"tile checkpoint array {name!r} has an invalid shape {shape!r}"
                )
            expected_dtype = np.dtype(dtype_name)
            if str(entry.get("dtype")) != str(expected_dtype):
                raise TileCheckpointCorrupt(
                    f"tile checkpoint array {name!r} dtype "
                    f"{entry.get('dtype')!r} != {dtype_name!r}"
                )
            chunk = raw[offset:offset + nbytes]
            if _sha256_bytes(chunk) != str(entry.get("sha256")):
                raise TileCheckpointCorrupt(
                    f"tile checkpoint array {name!r} SHA-256 mismatch"
                )
            if nbytes != int(np.prod(shape, dtype=np.int64)) * expected_dtype.itemsize:
                raise TileCheckpointCorrupt(
                    f"tile checkpoint array {name!r} byte length does not match "
                    "its shape/dtype"
                )
            arrays[name] = np.frombuffer(chunk, dtype=expected_dtype).reshape(shape).copy()
        return arrays

    # -- save -----------------------------------------------------------
    def save_tile(
        self,
        tile_index: int,
        bounds,
        *,
        arrays: dict,
        interior_origin,
        interior_shape,
        extended_bounds,
        counters: dict,
        batch_geometry: dict,
        sink_delta: dict | None,
        backend: dict | None = None,
    ) -> None:
        """Durably publish ONE reduced tile (payload first, marker last)."""
        tile_dir = self._tile_dir(tile_index)
        if tile_dir.exists() and tile_dir.is_symlink():
            raise TileCheckpointPathError(
                f"refusing to write a tile checkpoint through a symlink: {tile_dir}"
            )
        tile_dir.mkdir(parents=True, exist_ok=True)
        payload_path = tile_dir / TILE_PAYLOAD_NAME
        if payload_path.is_symlink():
            raise TileCheckpointPathError(
                f"refusing to replace a symlinked checkpoint payload: {payload_path}"
            )
        tmp_path = self.root / "tmp" / f"tile-{int(tile_index):06d}-{os.getpid()}-{uuid.uuid4().hex}.bin"
        entries = {}
        payload_digest = hashlib.sha256()
        offset = 0
        try:
            fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except OSError as exc:
            raise TileCheckpointError(
                f"cannot create checkpoint temp payload {tmp_path}: {exc}"
            ) from exc
        try:
            try:
                for name, dtype_name, ndim in _ARRAYS:
                    value = arrays[name]
                    if not isinstance(value, np.ndarray):
                        raise TileCheckpointError(
                            f"checkpoint array {name!r} must be a numpy array"
                        )
                    expected_dtype = np.dtype(dtype_name)
                    if value.dtype != expected_dtype:
                        raise TileCheckpointError(
                            f"checkpoint array {name!r} dtype {value.dtype} != "
                            f"{expected_dtype}"
                        )
                    if value.ndim != ndim:
                        raise TileCheckpointError(
                            f"checkpoint array {name!r} ndim {value.ndim} != {ndim}"
                        )
                    if value.shape != tuple(
                        int(v) for v in _shape_for(name, interior_shape, ndim)
                    ):
                        raise TileCheckpointError(
                            f"checkpoint array {name!r} shape {value.shape} does "
                            "not match the tile interior"
                        )
                    nbytes = int(value.size) * expected_dtype.itemsize
                    array_digest = hashlib.sha256()
                    _write_array_stream(fd, value, payload_digest, array_digest)
                    entries[name] = {
                        "dtype": dtype_name,
                        "shape": [int(v) for v in value.shape],
                        "offset": int(offset),
                        "nbytes": int(nbytes),
                        "sha256": array_digest.hexdigest(),
                    }
                    offset += nbytes
                zrs.fsync_file(fd)
            finally:
                os.close(fd)
        except BaseException:
            try:
                tmp_path.unlink()
            except OSError:
                pass
            raise
        try:
            _replace_with_retry(tmp_path, payload_path)
            zrs.fsync_dir(tile_dir)
        except BaseException:
            try:
                tmp_path.unlink()
            except OSError:
                pass
            raise
        record = {
            "schema": CHECKPOINT_SCHEMA,
            "version": int(CHECKPOINT_VERSION),
            "cell_id": str(self.identity["cell_id"]),
            "identity_digest": str(self.identity["identity_digest"]),
            "tile_index": int(tile_index),
            "bounds": [int(v) for v in bounds],
            "interior_origin": [int(v) for v in interior_origin],
            "interior_shape": [int(v) for v in interior_shape],
            "extended_bounds": [int(v) for v in extended_bounds],
            "arrays": entries,
            "payload": {
                "name": TILE_PAYLOAD_NAME,
                "bytes": int(offset),
                "sha256": payload_digest.hexdigest(),
            },
            "counters": {str(k): int(v) for k, v in counters.items()},
            "batch_geometry": _json_safe(dict(batch_geometry)),
            "sink_delta": None if sink_delta is None else _json_safe(dict(sink_delta)),
            **(
                {"backend": _json_safe(dict(backend))}
                if backend is not None else {}
            ),
            "created_unix_ns": int(time.time_ns()),
        }
        if backend is not None:
            _req = str(record["backend"].get("requested") or "")
            _eff = str(record["backend"].get("effective") or "")
            if _req not in ("cpu", "gpu") or _eff not in ("cpu", "gpu"):
                raise TileCheckpointError(
                    "checkpoint backend provenance must name cpu/gpu, got "
                    f"requested={_req!r} effective={_eff!r}"
                )
        record["marker_sha256"] = _sha256_bytes(_canonical_json(record))
        zrs.atomic_write_bytes(
            tile_dir / TILE_MARKER_NAME,
            _canonical_json(record),
            tmp_dir=self.root / "tmp",
        )
        self.saved_tiles += 1

    def discard_tile(self, tile_index: int) -> None:
        """Remove ONE tile checkpoint (corrupt/failed overwrite path only)."""
        tile_dir = self._tile_dir(tile_index)
        if tile_dir.exists():
            if tile_dir.is_symlink():
                raise TileCheckpointPathError(
                    f"refusing to delete a symlinked tile checkpoint: {tile_dir}"
                )
            shutil.rmtree(tile_dir)

    def mark_cell_complete(self) -> None:
        """Mark the in-memory runtime complete (NOT a durable commit/marker)."""
        self.runtime_complete = True


def _shape_for(name: str, interior_shape, ndim: int):
    h_int, w_int, channels = (int(v) for v in interior_shape)
    if ndim == 3:
        return (h_int, w_int, channels)
    if name in ("support_w1", "support_w2"):
        return (h_int, w_int)
    raise TileCheckpointError(f"unknown checkpoint array {name!r}")


def _replace_with_retry(src: Path, dst: Path) -> None:
    zrs._replace_with_retry(str(src), str(dst))  # noqa: SLF001 - same package


# ---------------------------------------------------------------------------
# Post-commit cleanup
# ---------------------------------------------------------------------------

def _manifest_record_is_valid(record) -> bool:
    if not isinstance(record, dict):
        return False
    if record.get("schema") != CELL_MANIFEST_SCHEMA:
        return False
    if record.get("version") != int(CHECKPOINT_VERSION):
        return False
    identity = record.get("identity_digest")
    if not isinstance(identity, str) or len(identity) != 64:
        return False
    section_digests = record.get("section_digests")
    sections = record.get("sections")
    if not isinstance(section_digests, dict) or not section_digests:
        return False
    if not isinstance(sections, dict) or not sections:
        return False
    cell = sections.get("cell")
    if not isinstance(cell, dict) or not str(cell.get("cell_id") or ""):
        return False
    if str(record.get("policy")) not in (
        "recompute_corrupt", "refuse_corrupt"
    ):
        return False
    run_id = record.get("run_id")
    if run_id is not None and not isinstance(run_id, str):
        return False
    expected = record.get("marker_sha256")
    actual = _sha256_bytes(
        _canonical_json(
            {k: v for k, v in record.items() if k != "marker_sha256"}
        )
    )
    return isinstance(expected, str) and expected == actual


_TILE_DIR_RE = re.compile(r"^t\d{6}$")
_TMP_FILE_RE = re.compile(r"^tile-\d+-\d+-[0-9a-f]{32}\.bin$")


def _cell_contents_are_owned(entry: Path) -> bool:
    """Strict contents of ONE owned Cell directory (no foreign entries).

    Missing control files are legitimate crash residue; a PRESENT but malformed
    owner/lock, any symlink (including dangling), non-object marker or unknown
    entry makes ownership unprovable and returns False (foreign).
    """
    for name in ("LOCK", "OWNER.json"):
        control = entry / name
        if control.is_symlink():
            return False
        if control.exists():
            if not control.is_file():
                return False
            if name == "OWNER.json":
                try:
                    owner = json.loads(control.read_bytes().decode("utf-8"))
                except Exception:
                    return False
                if not isinstance(owner, dict):
                    return False
    tiles = entry / "tiles"
    if tiles.is_symlink():
        return False
    if tiles.exists():
        if not tiles.is_dir():
            return False
        try:
            tile_entries = list(tiles.iterdir())
        except OSError:
            return False
        for tile in tile_entries:
            if (
                tile.is_symlink()
                or not tile.is_dir()
                or not _TILE_DIR_RE.match(tile.name)
            ):
                return False
            try:
                names = {item.name for item in tile.iterdir()}
            except OSError:
                return False
            if names - {TILE_PAYLOAD_NAME, TILE_MARKER_NAME}:
                return False
            for item in tile.iterdir():
                if item.is_symlink() or not item.is_file():
                    return False
            marker = tile / TILE_MARKER_NAME
            if marker.is_symlink():
                return False
            if marker.exists():
                try:
                    record = json.loads(marker.read_bytes().decode("utf-8"))
                except Exception:
                    return False
                if not isinstance(record, dict):
                    return False
                if record.get("schema") != CHECKPOINT_SCHEMA:
                    return False
                expected = record.get("marker_sha256")
                actual = _sha256_bytes(
                    _canonical_json(
                        {
                            k: v for k, v in record.items()
                            if k != "marker_sha256"
                        }
                    )
                )
                if not isinstance(expected, str) or expected != actual:
                    return False
    tmp = entry / "tmp"
    if tmp.is_symlink():
        return False
    if tmp.exists():
        if not tmp.is_dir():
            return False
        try:
            tmp_entries = list(tmp.iterdir())
        except OSError:
            return False
        for item in tmp_entries:
            if item.is_symlink() or not item.is_file():
                return False
            if not _TMP_FILE_RE.match(item.name):
                return False
    return True


def inspect_owned_checkpoint_root(base_root) -> dict:
    """Strict ownership proof for ONE product checkpoint root.

    Publication/overwrite must never treat the directory NAME alone as ours.
    ``owned`` is True ONLY for a plain directory (never a symlink) whose direct
    children are plain Cell directories, each carrying a VALID atomic
    ``checkpoint.json`` manifest (schema + marker sha256) and no live owner. An
    EMPTY plain root is owned residue too. Any file, unknown entry, symlink,
    malformed/forged manifest or live owner makes the root NOT owned, so the
    existing foreign-content refusal keeps protecting it.
    """
    path = Path(base_root)
    if path.is_symlink():
        return {"owned": False, "reason": "symlink"}
    if not path.is_dir():
        return {"owned": False, "reason": "not_plain_directory"}
    try:
        entries = sorted(path.iterdir())
    except OSError as exc:
        return {"owned": False, "reason": f"unreadable:{type(exc).__name__}"}
    if not entries:
        return {"owned": True, "reason": "empty", "cells": 0}
    allowed = {"LOCK", "OWNER.json", "tiles", "tmp", CELL_MANIFEST_NAME}
    cells = 0
    for entry in entries:
        if entry.is_symlink() or not entry.is_dir():
            return {
                "owned": False,
                "reason": "foreign_entry",
                "entry": entry.name,
            }
        try:
            names = {item.name for item in entry.iterdir()}
        except OSError as exc:
            return {
                "owned": False,
                "reason": f"cell_unreadable:{type(exc).__name__}",
                "entry": entry.name,
            }
        if names - allowed:
            return {
                "owned": False,
                "reason": "foreign_cell_entry",
                "entry": entry.name,
            }
        manifest = entry / CELL_MANIFEST_NAME
        if manifest.is_symlink() or not manifest.is_file():
            return {
                "owned": False,
                "reason": "missing_or_linked_manifest",
                "entry": entry.name,
            }
        try:
            record = json.loads(manifest.read_bytes().decode("utf-8"))
        except Exception:
            return {
                "owned": False,
                "reason": "malformed_manifest",
                "entry": entry.name,
            }
        if not _manifest_record_is_valid(record):
            return {
                "owned": False,
                "reason": "invalid_manifest_marker",
                "entry": entry.name,
            }
        if not _cell_contents_are_owned(entry):
            return {
                "owned": False,
                "reason": "foreign_cell_content",
                "entry": entry.name,
            }
        owner = None
        try:
            owner = zrs.StoreLock.peek_owner(entry)
        except Exception:
            return {
                "owned": False,
                "reason": "malformed_owner",
                "entry": entry.name,
            }
        if (
            owner is not None
            and str(owner.get("state", "held")) != "released"
            and zrs.owner_is_live(owner)
        ):
            return {
                "owned": False,
                "reason": "live_owner",
                "entry": entry.name,
            }
        cells += 1
    return {"owned": True, "reason": "valid_manifests", "cells": int(cells)}


def cleanup_cell(base_root, cell_id: str, *, owner_liveness=None) -> dict:
    """Remove ONE Cell's checkpoints AFTER its final durable commit.

    Best effort by design: a failure is reported, never fatal. A directory whose
    owner record names a LIVE process is never removed.
    """
    try:
        cell_dir = cell_directory(base_root, cell_id)
    except TileCheckpointPathError as exc:
        return {"removed": False, "reason": f"path:{exc}"}
    if not cell_dir.exists():
        return {"removed": False, "reason": "absent"}
    if cell_dir.is_symlink() or not cell_dir.is_dir():
        return {"removed": False, "reason": "not_plain_directory"}
    owner = zrs.StoreLock.peek_owner(cell_dir)
    if (
        owner is not None
        and str(owner.get("state", "held")) != "released"
        and zrs.owner_is_live(owner)
    ):
        return {"removed": False, "reason": "live_owner"}
    try:
        shutil.rmtree(cell_dir)
    except Exception as exc:  # noqa: BLE001 - documented retention
        return {
            "removed": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "path": str(cell_dir),
        }
    # ZM-M33 rework-3: a COMPLETE run must not leave the owned root behind.
    # Only an EMPTY plain root is removed; a non-empty/incomplete root stays for
    # resume and is protected by the strict ownership proof above.
    root_removed = False
    try:
        base = Path(base_root)
        if not base.is_symlink() and base.is_dir():
            base.rmdir()
            root_removed = True
    except OSError:
        root_removed = False
    return {
        "removed": True,
        "path": str(cell_dir),
        "root_removed": bool(root_removed),
    }
