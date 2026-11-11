"""ZM-ZEGRID-R6 — file-backed (memmap) ``CanonicalFrameProvider`` + aligned disk cache.

This is the **real consumer** of the R5 streaming executor
(:func:`zemosaic.core.canonical_streaming.run_canonical_stack_streaming`): a
provider that satisfies the R5 ``CanonicalFrameProvider`` Protocol while serving
the ALIGNED patch data from a **disk-backed cache**, so the aligned input is
NOT fully RAM-resident.

Why the RAM peak is bounded
---------------------------
The R1/R2/R3 in-memory path materialises every patch contributor's aligned RGB
(``N x H x W x C`` float32) AND the canonical pipeline's ``O(N x patch_area)``
intermediates (float64 normalization/taper/rejection/combine) at once. For a real
cell with N=66 at ~410k px this peaks at ~3.4-4.1 GiB (R3 measured).

This module removes the **input-resident float64 term** and leaves the R5
executor's ``O(N x tile_area)`` tile workspace as the dominant *tile-scaled*
live term. The HONEST peak decomposition is:

* ``O(N x tile_area)`` float64 intermediates (the tile workspace) — the only term
  reduced vs the in-memory path;
* ``O(N x patch_area) x 1 byte`` residuals: the ``rejection_mask`` bool
  ``(N, H, W, C)`` output plane, and the aligned-input residency (this cache's
  float32/bool pages touched by phase 1/2);
* ``O(patch_area)`` output planes;
* a fixed subprocess baseline.

(That is: NOT ``O(N x patch_area)`` float64, but the ``O(N x patch_area)`` term
shrinks to ~1 byte/px bool residency, which is what makes the streaming run
~3x cheaper than the in-memory path in measured RSS.)

* **Build (streaming reprojection, one frame at a time).** Each contributor's
  aligned patch (RGB float32 + bool geometric support) is materialised, written
  to a ``.npy`` file on disk, and dropped before the next frame. Build peak is
  ``O(patch_area)`` (one frame + reproject internals), independent of N.
* **Phase 1 (``get_raw_frame``)** reads a whole frame's slice once; the R5
  executor keeps <= 2 frames resident.
* **Phase 2 (``get_tile``)** slices the memmap and only materialises the
  ``tile_area`` requested by the executor (the R5 executor's working set is
  ``O(N x tile_area)``).

The disk cache is an **explicit, reusable, resumable artifact** (never ``/tmp``):
a directory with a ``manifest.json`` (per-frame sha256 + sizes) and one
``frame_XXXX_rgb.npy`` / ``frame_XXXX_support.npy`` per contributor. Because the
arrays are written/read with ``numpy`` ``.npy`` round-trips, the served values
are **bit-identical** to the in-memory arrays (verified: float32 NaN payloads
and bool masks round-trip exactly).

Scope / non-goals
-----------------
* New module only; NO existing canonical file is modified, NO production
  dispatch wiring, NO new dependency.
* The cache build reuses the existing R1 reprojection primitives
  (``execution._read_section_chw`` / ``slice_wcs`` / ``reproject_cropped``)
  verbatim — bit-exactness is by construction, not re-implementation.
"""

from __future__ import annotations

import hashlib
import numbers
import json
import logging
import shutil
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

from zemosaic.core.canonical_stacking import (
    CanonicalStackValidationError,
    prepare_canonical_inputs,
)

from . import execution as zxe
from .fd_limits import DEFAULT_MAX_OPEN_CACHE_FILES, FILES_PER_FRAME
from .geometry import (
    FrameDescriptor,
    GlobalCanvas,
    ProcessingPatch,
    SourceCropPlan,
)

CACHE_SCHEMA = "zemosaic-zegrid-r6-aligned-cache-v1"

#: ZM-ZEGRID-FD-EXHAUSTION: hard cap on simultaneously mapped ``.npy`` files in
#: ONE :class:`MemmapCanonicalProvider`. Counted in FILES (2 per frame), so the
#: default keeps at most 32 frames mapped regardless of the cell's frame count.
DEFAULT_MAX_OPEN_MAPPINGS = DEFAULT_MAX_OPEN_CACHE_FILES

__all__ = [
    "CACHE_SCHEMA",
    "AlignedCacheBuilder",
    "BoundedMemmapCanonicalProvider",
    "DEFAULT_MAX_OPEN_MAPPINGS",
    "MemmapCanonicalProvider",
    "WindowedCanonicalProvider",
    "build_aligned_cache_from_sources",
    "write_aligned_cache_from_arrays",
    "cache_manifest_path",
    "cache_is_complete",
    "patch_geometry_identity",
    "load_cache_manifest",
]


# ---------------------------------------------------------------------------
# Paths + hashing helpers
# ---------------------------------------------------------------------------

def cache_manifest_path(cache_dir) -> Path:
    return Path(cache_dir) / "manifest.json"

def _frame_rgb_path(cache_dir, idx: int) -> Path:
    return Path(cache_dir) / f"frame_{idx:04d}_rgb.npy"


def _frame_sup_path(cache_dir, idx: int) -> Path:
    return Path(cache_dir) / f"frame_{idx:04d}_support.npy"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ZM-ZEGRID-R15: Windows releases file handles asynchronously; a memmap/``.npy``
# may still be open when we try to delete the cache, raising ``WinError 32``.
# Deletion is an OPTIMISATION, never fatal: retry a few times with a short
# backoff, and on final failure log a WARN and continue.
_RMTREE_ATTEMPTS = 3
_RMTREE_BACKOFF_S = 0.2


def safe_rmtree(path) -> bool:
    """Best-effort recursive delete (never raises); returns True on success."""
    path = Path(path)
    if not path.exists():
        return True
    last_exc = None
    for attempt in range(_RMTREE_ATTEMPTS):
        try:
            shutil.rmtree(str(path))
            return True
        except OSError as exc:  # noqa: BLE001 - deletion is best-effort by design
            last_exc = exc
            if attempt + 1 < _RMTREE_ATTEMPTS:
                time.sleep(_RMTREE_BACKOFF_S * (attempt + 1))
    logging.getLogger(__name__).warning(
        "cache cleanup FAILED (non-fatal): %s — leftover cache at %s will be reused/ignored",
        last_exc,
        path,
    )
    return False


def _close_memmap(arr) -> None:
    """Release a numpy memmap's underlying ``mmap`` handle (best-effort, never raises).

    ZM-ZEGRID-R15: ``np.load(..., mmap_mode="r")`` returns a ``numpy.memmap`` whose
    ``.base`` is the ``mmap.mmap`` object. Closing that ``mmap`` releases the OS file
    handle, which is REQUIRED on Windows before the ``.npy`` file can be deleted
    (otherwise ``WinError 32``). Falls back to ``_mmap`` if ``.base`` is unavailable.
    """
    for attr in ("base", "_mmap"):
        try:
            obj = getattr(arr, attr, None)
            if obj is not None and hasattr(obj, "close"):
                obj.close()
        except Exception:
            pass


def _load_memmap(path: Path):
    """Open a read-only ``.npy`` mapping (single seam for tests/instrumentation)."""
    return np.load(path, mmap_mode="r")


def _array_meta(rgb: np.ndarray):
    rgb = np.asarray(rgb)
    if rgb.ndim == 2:
        h, w = rgb.shape
        return h, w, 1, True, 2, (h, w)
    h, w, c = rgb.shape
    return h, w, c, False, 3, (h, w, c)


def _meta_from_hwc(h: int, w: int, c: int = 3):
    """Aligned-cache metadata for a known HWC patch shape (parallel path)."""
    if c == 1:
        return h, w, 1, True, 2, (h, w)
    return h, w, c, False, 3, (h, w, c)


def patch_geometry_identity(patch) -> str:
    """Deterministic identity of a :class:`ProcessingPatch` spatial geometry.

    ZM-ZEGRID-R53 F-1: a cache directory is keyed by cell id only. Two runs with
    the same cell id and frame-id list can still build patches that have the
    SAME shape but different global bounds / WCS (a different layout, or a
    shifted canvas). Shape equality alone is therefore not sufficient for safe
    reuse. This identity binds the patch bounds and its patch-WCS header.
    """
    b = patch.patch
    header = str(getattr(patch, "patch_wcs_header", "") or "")
    wcs_digest = hashlib.sha256(header.encode("utf-8")).hexdigest()
    return f"{int(b.x0)},{int(b.y0)},{int(b.x1)},{int(b.y1)}|{wcs_digest}"


def _write_manifest(cache_dir, meta, frames, frame_ids, total_bytes, *,
                    patch_identity=None) -> dict:
    """Write the aligned-cache manifest from pre-collected frame metadata.

    Shared by the serial :class:`AlignedCacheBuilder` and the parallel build path
    so both produce the IDENTICAL manifest schema/content.
    """
    h, w, c, mono, ndim, shape = meta
    manifest = {
        "schema": CACHE_SCHEMA,
        "n_frames": len(frames),
        "height": h,
        "width": w,
        "channels": c,
        "original_mono": mono,
        "original_ndim": ndim,
        "original_shape": list(shape),
        "frame_ids": list(frame_ids),
        "frames": list(frames),
        "total_bytes": int(total_bytes),
    }
    if patch_identity is not None:
        manifest["patch_identity"] = str(patch_identity)
    cache_manifest_path(cache_dir).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_cache_manifest(cache_dir) -> dict:
    mp = cache_manifest_path(cache_dir)
    if not mp.exists():
        raise FileNotFoundError(f"aligned cache manifest missing: {mp}")
    manifest = json.loads(mp.read_text())
    if manifest.get("schema") != CACHE_SCHEMA:
        raise CanonicalStackValidationError(
            f"aligned cache schema {manifest.get('schema')!r} != {CACHE_SCHEMA!r}"
        )
    return manifest


def _normalize_expected_hw(expected_hw):
    """Strictly normalize an optional expected ``(height, width)``.

    Returns ``None`` when no expectation was supplied, ``False`` when the
    supplied expectation is malformed, or a positive ``(int, int)`` tuple.
    """
    if expected_hw is None:
        return None
    try:
        if isinstance(expected_hw, (str, bytes)) or len(expected_hw) != 2:
            return False
        h, w = expected_hw
        if isinstance(h, bool) or isinstance(w, bool):
            return False
        if not isinstance(h, numbers.Integral) or not isinstance(w, numbers.Integral):
            return False
        h, w = int(h), int(w)
        if h <= 0 or w <= 0:
            return False
        return (h, w)
    except Exception:
        return False


def cache_is_complete(cache_dir, frame_ids: list[str], *, expected_hw=None,
                      expected_patch_identity=None) -> bool:
    """True if the cache dir holds a matching, complete manifest + all files.

    Resumability check: rebuild is skipped only when the frame-id list matches,
    every declared file exists (and matches its recorded sha256), and — when
    supplied — the expected patch identity matches exactly.

    ZM-ZEGRID-R53 F-1: ``expected_hw`` is the patch ``(height, width)`` the
    caller is about to use. ``expected_patch_identity`` additionally binds the
    global patch bounds and patch WCS (see :func:`patch_geometry_identity`),
    because same-shaped patches can differ spatially across layouts. Missing,
    malformed or mismatched expected values return ``False`` — this predicate
    is a resumability test and MUST NOT raise.
    """
    mp = cache_manifest_path(cache_dir)
    if not mp.exists():
        return False
    try:
        manifest = load_cache_manifest(cache_dir)
    except Exception:
        return False
    if list(manifest.get("frame_ids", [])) != list(frame_ids):
        return False
    if expected_hw is not None:
        want_hw = _normalize_expected_hw(expected_hw)
        if want_hw is False:
            return False
        try:
            got_hw = (int(manifest["height"]), int(manifest["width"]))
        except Exception:
            return False
        if got_hw != want_hw:
            return False
    if expected_patch_identity is not None:
        if not isinstance(expected_patch_identity, str) or not expected_patch_identity:
            return False
        got_identity = manifest.get("patch_identity")
        if not isinstance(got_identity, str) or got_identity != expected_patch_identity:
            return False
    for i, fr in enumerate(manifest["frames"]):
        rp = Path(cache_dir) / fr["rgb"]
        sp = Path(cache_dir) / fr["support"]
        if not rp.exists() or not sp.exists():
            return False
        if fr.get("rgb_sha256") and _sha256(rp) != fr["rgb_sha256"]:
            return False
        if fr.get("support_sha256") and _sha256(sp) != fr["support_sha256"]:
            return False
    return True


# ---------------------------------------------------------------------------
# Cache builder (streaming write, one frame at a time)
# ---------------------------------------------------------------------------

class AlignedCacheBuilder:
    """Streaming writer of an aligned disk cache (one frame at a time).

    ``add()`` writes a frame's aligned RGB + support to ``.npy`` immediately and
    drops the caller's references; only the current frame is ever live. Metadata
    (shape/channels/mono) is taken from the first frame, so a builder is a
    single-use object.
    """

    def __init__(self, cache_dir, *, patch_identity=None) -> None:
        self.cache_dir = Path(cache_dir)
        safe_rmtree(self.cache_dir)  # ZM-ZEGRID-R15: best-effort wipe (never fatal)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._meta = None
        self._frames: list[dict] = []
        self._frame_ids: list[str] = []
        self._total_bytes = 0
        self._n = 0
        self._patch_identity = (
            None if patch_identity is None else str(patch_identity)
        )

    def add(self, frame_id: str, rgb: np.ndarray, support: np.ndarray) -> None:
        rgb = np.ascontiguousarray(np.asarray(rgb, dtype=np.float32))
        support = np.ascontiguousarray(np.asarray(support, dtype=bool))
        if self._meta is None:
            self._meta = _array_meta(rgb)
        else:
            got = _array_meta(rgb)
            if got != self._meta:
                raise CanonicalStackValidationError(
                    f"frame {frame_id!r}: inconsistent aligned shape {got} != {self._meta}"
                )
        if support.shape != rgb.shape[:2]:
            raise CanonicalStackValidationError(
                f"frame {frame_id!r}: support shape {support.shape} != rgb HW {rgb.shape[:2]}"
            )

        rp = _frame_rgb_path(self.cache_dir, self._n)
        sp = _frame_sup_path(self.cache_dir, self._n)
        np.save(rp, rgb)
        np.save(sp, support)
        rp_size = rp.stat().st_size
        sp_size = sp.stat().st_size
        self._total_bytes += rp_size + sp_size
        self._frames.append(
            {
                "index": self._n,
                "frame_id": frame_id,
                "rgb": rp.name,
                "rgb_sha256": _sha256(rp),
                "rgb_bytes": rp_size,
                "support": sp.name,
                "support_sha256": _sha256(sp),
                "support_bytes": sp_size,
            }
        )
        self._frame_ids.append(frame_id)
        self._n += 1

    def finish(self) -> dict:
        if self._meta is None:
            raise CanonicalStackValidationError("empty aligned cache (no frames written)")
        return _write_manifest(
            self.cache_dir, self._meta, self._frames, self._frame_ids,
            self._total_bytes, patch_identity=self._patch_identity,
        )


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def write_aligned_cache_from_arrays(
    cache_dir,
    frame_ids: list[str],
    rgb_arrays,
    support_arrays,
) -> dict:
    """Write already-aligned arrays to a disk cache (test/demo primitive).

    ``rgb_arrays`` / ``support_arrays`` are sequences of aligned ``(H,W,C)``
    float32 and ``(H,W)`` bool arrays. Used by the equivalence tests; the real
    path is :func:`build_aligned_cache_from_sources`.
    """
    n = len(frame_ids)
    if len(rgb_arrays) != n or len(support_arrays) != n:
        raise CanonicalStackValidationError("frame_ids / arrays length mismatch")
    builder = AlignedCacheBuilder(cache_dir)
    for i in range(n):
        builder.add(frame_ids[i], np.asarray(rgb_arrays[i]), np.asarray(support_arrays[i]))
    return builder.finish()


def build_aligned_cache_from_sources(
    frames: list[FrameDescriptor],
    prepared_paths: dict[str, str],
    canvas: GlobalCanvas,
    patch: ProcessingPatch,
    crop_plans: dict[str, SourceCropPlan],
    cache_dir,
    *,
    tracker: "zxe.SectionReadTracker | None" = None,
    reuse_cache: bool = True,
) -> dict:
    """Stream the R1 reprojection per frame to a disk cache (never all in RAM).

    Mirrors ``execution.build_patch_contributors`` exactly (same sorted FrameId
    order, same section read / WCS slice / reprojection, same skip-on-``None``
    plan) but writes each aligned frame to ``.npy`` and drops it before the next.
    Peak RAM is ``O(patch_area)`` (one frame + reproject internals).

    ``reuse_cache`` (default True) skips the rebuild when an existing cache is
    complete for the same ordered frame-id list (resumable artifact).
    """
    patch_wcs = patch.patch_wcs()
    ordered: list[tuple[FrameDescriptor, SourceCropPlan, str]] = []
    for f in sorted(frames, key=lambda f: f.frame_id):
        key = f.frame_id.logical_path
        plan = crop_plans.get(key)
        if plan is None:
            continue
        prep = prepared_paths.get(key)
        if prep is None:
            raise ValueError(f"no prepared RGB for {key}")
        ordered.append((f, plan, prep))

    frame_ids = [f.frame_id.logical_path for f, _p, _q in ordered]
    patch_identity = patch_geometry_identity(patch)
    if reuse_cache and cache_is_complete(
        cache_dir, frame_ids,
        expected_hw=patch.patch_shape_hw,
        expected_patch_identity=patch_identity,
    ):
        return load_cache_manifest(cache_dir)

    builder = AlignedCacheBuilder(cache_dir, patch_identity=patch_identity)
    for f, plan, prep in ordered:
        cropped = zxe._read_section_chw(prep, plan.source_bounds, tracker=tracker)
        cropped_wcs = zxe.slice_wcs(f.wcs(), plan.source_bounds)
        rgb, geom = zxe.reproject_cropped(
            cropped, cropped_wcs, patch_wcs, patch.patch_shape_hw
        )
        builder.add(f.frame_id.logical_path, rgb, geom)
        del cropped, rgb, geom
    return builder.finish()


# ---------------------------------------------------------------------------
# Provider (satisfies CanonicalFrameProvider Protocol)
# ---------------------------------------------------------------------------

class MemmapCanonicalProvider:
    """File-backed provider serving aligned frames from a disk cache.

    Satisfies the R5 ``CanonicalFrameProvider`` Protocol:

    * ``get_raw_frame(i)`` -> ``(raw_image, raw_support)`` (phase 1; whole frame).
    * ``get_tile(i, y0, y1, x0, x1)`` -> ``(prepared_float32_hwc_tile, valid_bool)``
      (phase 2; only the requested tile is materialised, via a memmap slice).

    The served arrays are bit-identical to the aligned arrays an in-memory
    provider would hold, because they are ``.npy`` round-trips of the same
    ``execution.reproject_cropped`` output (float32 + bool, bit-exact).

    ZM-ZEGRID-R15: the provider is a CONTEXT MANAGER and exposes :meth:`close`,
    which releases every memmap's underlying ``mmap`` handle. On Windows a file
    still mapped/opened cannot be deleted (``WinError 32``), so callers MUST
    ``close()`` (or ``with``) the provider BEFORE deleting the cache directory.

    ZM-ZEGRID-FD-EXHAUSTION: the number of simultaneously mapped cache files is
    BOUNDED (``max_open_mappings``, default 64 files = 32 frames). A small cache
    that fits the cap is opened eagerly exactly as before; a dense cell (e.g.
    527 frames -> 1054 files) now keeps at most the cap mapped and re-opens on
    demand, so one cell can no longer exhaust the process descriptor limit.
    Served values are unchanged (bit-identical ``.npy`` round-trips).
    """

    def __init__(self, cache_dir, *, max_open_mappings=None) -> None:
        self.cache_dir = Path(cache_dir)
        manifest = load_cache_manifest(self.cache_dir)
        self._manifest = manifest
        self.n_frames = int(manifest["n_frames"])
        self.height = int(manifest["height"])
        self.width = int(manifest["width"])
        self.channels = int(manifest["channels"])
        self.original_mono = bool(manifest["original_mono"])
        self.original_ndim = int(manifest["original_ndim"])
        self.original_shape = tuple(manifest["original_shape"])
        self.frame_ids = tuple(manifest["frame_ids"])
        cap = (
            DEFAULT_MAX_OPEN_MAPPINGS
            if max_open_mappings is None
            else int(max_open_mappings)
        )
        # At least one whole frame (2 files) must be mappable at a time.
        self.max_open_mappings = max(2 * FILES_PER_FRAME, cap)
        self._total_files = FILES_PER_FRAME * self.n_frames
        # Eager == the historical shape: every file of a cache that FITS the cap.
        self._eager = self._total_files <= self.max_open_mappings
        self._rgb: list = []
        self._sup: list = []
        self._open_order: "OrderedDict[int, None]" = OrderedDict()
        self.mapping_open_count = 0
        self.mapping_close_count = 0
        self.mapping_open_failures = 0
        self.peak_open_mappings = 0
        # The canonical executor is a single-reader provider (documented), but a
        # bounded mapping MAY be evicted by a later access, so the lazy bookkeeping
        # and the copy-out happen under one lock: a future concurrent caller gets
        # owned data, never a dangling view.
        self._lock = threading.RLock()
        if self._eager:
            for i in range(self.n_frames):
                rgb, sup = self._open_pair_maps(i)
                self._rgb.append(rgb)
                self._sup.append(sup)
                self._open_order[i] = None
            self.peak_open_mappings = FILES_PER_FRAME * len(self._open_order)
        else:
            self._rgb = [None] * self.n_frames
            self._sup = [None] * self.n_frames

    # -- bounded mapping lifecycle (ZM-ZEGRID-FD-EXHAUSTION) -------------
    @property
    def open_mappings(self) -> int:
        """Number of cache files currently mapped (2 per open frame)."""
        return FILES_PER_FRAME * len(self._open_order)

    def _open_pair_maps(self, i: int):
        """Open ``(rgb, support)`` with an all-or-nothing error path."""
        rgb = _load_memmap(_frame_rgb_path(self.cache_dir, i))
        self.mapping_open_count += 1
        try:
            sup = _load_memmap(_frame_sup_path(self.cache_dir, i))
            self.mapping_open_count += 1
        except BaseException:
            self.mapping_open_failures += 1
            _close_memmap(rgb)
            self.mapping_close_count += 1
            raise
        return rgb, sup

    def _drop_index(self, i: int) -> None:
        rgb = self._rgb[i] if i < len(self._rgb) else None
        sup = self._sup[i] if i < len(self._sup) else None
        for arr in (rgb, sup):
            if arr is not None:
                _close_memmap(arr)
                self.mapping_close_count += 1
        if i < len(self._rgb):
            self._rgb[i] = None
            self._sup[i] = None

    def _evict_until_within_cap(self, keep: int) -> None:
        cap_frames = max(1, self.max_open_mappings // FILES_PER_FRAME)
        while len(self._open_order) > cap_frames:
            victim = next(iter(self._open_order))
            if victim == keep:
                # Never evict the frame the caller is about to read.
                if len(self._open_order) <= 1:
                    break
                victim = next(iter(list(self._open_order)[1:]))
            self._open_order.pop(victim, None)
            self._drop_index(victim)

    def _maps(self, i: int):
        """Return the mappings for frame ``i``, opening on demand (LRU)."""
        i = int(i)
        rgb = self._rgb[i] if i < len(self._rgb) else None
        sup = self._sup[i] if i < len(self._sup) else None
        if rgb is not None and sup is not None:
            self._open_order.pop(i, None)
            self._open_order[i] = None
            return rgb, sup
        # Re-open a partially dropped pair from scratch.
        if rgb is not None or sup is not None:
            self._drop_index(i)
            self._open_order.pop(i, None)
        rgb, sup = self._open_pair_maps(i)
        self._rgb[i] = rgb
        self._sup[i] = sup
        self._open_order[i] = None
        self._evict_until_within_cap(i)
        self.peak_open_mappings = max(self.peak_open_mappings, self.open_mappings)
        return rgb, sup

    def fd_telemetry(self) -> dict:
        """Truthful descriptor telemetry for the run manifest / tests."""
        return {
            "provider": type(self).__name__,
            "storage_domain": getattr(
                self, "storage_domain", "aligned-npy-memmap-dense"
            ),
            "eager": bool(self._eager),
            "frames": int(self.n_frames),
            "cache_files": int(self._total_files),
            "max_open_mappings": int(self.max_open_mappings),
            "open_mappings": int(self.open_mappings),
            "peak_open_mappings": int(self.peak_open_mappings),
            "mapping_open_count": int(self.mapping_open_count),
            "mapping_close_count": int(self.mapping_close_count),
            "mapping_open_failures": int(self.mapping_open_failures),
        }

    def close(self) -> None:
        """Release every memmap handle (idempotent; never raises).

        ZM-ZEGRID-R15: must be called (or ``with`` used) before the cache
        directory is deleted, otherwise Windows raises ``WinError 32`` on the
        ``.npy`` files still mapped into the process.
        """
        with self._lock:
            for arr in list(self._rgb) + list(self._sup):
                if arr is not None:
                    _close_memmap(arr)
                    self.mapping_close_count += 1
            self._rgb = []
            self._sup = []
            self._open_order.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # -- metadata -----------------------------------------------------------
    @property
    def shape(self):
        return (self.height, self.width)

    @property
    def manifest(self) -> dict:
        return self._manifest

    # -- phase 1 raw access -------------------------------------------------
    def get_raw_frame(self, i: int):
        if self._eager:
            # Historical dense semantics: the mappings stay open for the life of
            # the provider, so a view is a valid read-only handle.
            rgb, sup = self._maps(int(i))
            return np.asarray(rgb), np.asarray(sup)
        # Bounded semantics: the mapping can be evicted by a LATER access (the
        # canonical phase 1 holds the reference frame across every other frame),
        # so hand out owned copies — identical values, never a dangling view.
        with self._lock:
            rgb, sup = self._maps(int(i))
            return np.array(rgb, copy=True), np.array(sup, copy=True)

    # -- phase 2 prepared tile access (mirrors InMemoryCanonicalProvider) ----
    def get_tile(self, i: int, y0: int, y1: int, x0: int, x1: int):
        if self._eager:
            raw, sup = self._maps(int(i))
            if self.original_mono:
                tile = raw[y0:y1, x0:x1]
            else:
                tile = raw[y0:y1, x0:x1, :]
            sup_tile = sup[y0:y1, x0:x1]
            batch = prepare_canonical_inputs([tile], [sup_tile])
            return batch.images[0], batch.valid_mask[0]
        # Bounded: copy the requested window out of the mapping immediately.
        with self._lock:
            raw, sup = self._maps(int(i))
            if self.original_mono:
                tile = np.array(raw[y0:y1, x0:x1], copy=True)
            else:
                tile = np.array(raw[y0:y1, x0:x1, :], copy=True)
            sup_tile = np.array(sup[y0:y1, x0:x1], copy=True)
        batch = prepare_canonical_inputs([tile], [sup_tile])
        return batch.images[0], batch.valid_mask[0]


# ---------------------------------------------------------------------------
# Bounded-lifetime provider (ZM-ZEGRID-R52-OOC, Lot A)
# ---------------------------------------------------------------------------

#: Storage domain advertised by :class:`BoundedMemmapCanonicalProvider`.
#: Honest label: the aligned cache stays on disk; only bounded owned copies are
#: ever resident, and no mapping outlives a single access call.
BOUNDED_STORAGE_DOMAIN = "aligned-npy-memmap-bounded"


class BoundedMemmapCanonicalProvider:
    """Bounded-lifetime file-backed provider over the same aligned NPY cache.

    Same Protocol surface and the same served values as
    :class:`MemmapCanonicalProvider`, but with a deliberately different
    *lifetime* contract:

    * every access opens the ``.npy`` mapping(s), copies exactly the requested
      data into an **owned** ndarray, and closes the mapping(s) BEFORE returning;
    * no mapping is kept between calls, so ``N`` frames cost no open handles and
      the cache directory can be deleted/moved once the access returns
      (Windows ``WinError 32`` avoidance that does not depend on a later
      ``close()``);
    * if opening the *support* file fails after the *RGB* file opened, the RGB
      mapping is closed too (no leaked handle; Windows-safe error path);
    * ``close()`` is idempotent and never raises; it records that the provider
      was closed. Returned arrays are owned copies, so they stay valid after
      ``close()`` (never a dangling memmap view);
    * no ``madvise``/``MADV_DONTNEED`` is used (not a portable guarantee), and a
      live memmap view is never exposed.

    Diagnostics are truthful and monotonic for tests/provenance:
    ``open_count`` (successful ``np.load`` mappings), ``open_failures`` (failed
    open attempts) and ``bytes_read`` (bytes copied out of the cache into owned
    arrays). ``storage_domain`` identifies the storage model.

    This class is NOT wired into any product path yet (Lot A): the default
    provider/dispatch is unchanged.
    """

    storage_domain = BOUNDED_STORAGE_DOMAIN

    def __init__(self, cache_dir) -> None:
        self.cache_dir = Path(cache_dir)
        manifest = load_cache_manifest(self.cache_dir)
        self._manifest = manifest
        self.n_frames = int(manifest["n_frames"])
        self.height = int(manifest["height"])
        self.width = int(manifest["width"])
        self.channels = int(manifest["channels"])
        self.original_mono = bool(manifest["original_mono"])
        self.original_ndim = int(manifest["original_ndim"])
        self.original_shape = tuple(manifest["original_shape"])
        self.frame_ids = tuple(manifest["frame_ids"])
        self.bytes_read = 0
        self.open_count = 0
        self.open_failures = 0
        self._closed = False

    # -- lifecycle ----------------------------------------------------------
    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Mark the provider closed (idempotent; never raises).

        By construction no mapping is held between calls, so there is nothing to
        release here; the flag only makes access-after-close fail loudly instead
        of silently serving data from a provider the caller considers finished.
        """
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError(
                "BoundedMemmapCanonicalProvider is closed; no further access is allowed"
            )

    # -- metadata -----------------------------------------------------------
    @property
    def shape(self):
        return (self.height, self.width)

    @property
    def manifest(self) -> dict:
        return self._manifest

    # -- bounded open/copy helpers -----------------------------------------
    def _open_pair(self, i: int):
        """Open ``(rgb, support)`` mappings with an all-or-nothing error path.

        ZM-ZEGRID-R52-OOC: if the support open raises after the RGB open
        succeeded, the RGB mapping is closed before re-raising, so a failed
        access cannot leak a file handle (a Windows delete would otherwise fail).
        """
        rgb = _load_memmap(_frame_rgb_path(self.cache_dir, i))
        self.open_count += 1
        try:
            sup = _load_memmap(_frame_sup_path(self.cache_dir, i))
            self.open_count += 1
        except BaseException:
            self.open_failures += 1
            _close_memmap(rgb)
            raise
        return rgb, sup

    @staticmethod
    def _owned_copy(arr) -> np.ndarray:
        """Copy ``arr`` into a plain owned ndarray (never a memmap/view)."""
        return np.array(arr, copy=True)

    # -- phase 1 raw access -------------------------------------------------
    def get_raw_frame(self, i: int):
        self._check_open()
        rgb, sup = self._open_pair(int(i))
        try:
            raw = self._owned_copy(rgb)
            raw_support = self._owned_copy(sup)
            self.bytes_read += int(raw.nbytes) + int(raw_support.nbytes)
            return raw, raw_support
        finally:
            _close_memmap(rgb)
            _close_memmap(sup)

    # -- phase 2 prepared tile access (mirrors MemmapCanonicalProvider) -----
    def get_tile(self, i: int, y0: int, y1: int, x0: int, x1: int):
        self._check_open()
        rgb, sup = self._open_pair(int(i))
        try:
            if self.original_mono:
                tile = self._owned_copy(rgb[y0:y1, x0:x1])
            else:
                tile = self._owned_copy(rgb[y0:y1, x0:x1, :])
            sup_tile = self._owned_copy(sup[y0:y1, x0:x1])
            self.bytes_read += int(tile.nbytes) + int(sup_tile.nbytes)
        finally:
            _close_memmap(rgb)
            _close_memmap(sup)
        batch = prepare_canonical_inputs([tile], [sup_tile])
        return batch.images[0], batch.valid_mask[0]


class WindowedCanonicalProvider:
    """A2a — lazy, window-of-one provider over the SAME preparation as the cache.

    ``prepare(i) -> (rgb_f32, support_bool)`` returns the aligned frame in the
    frozen order. The provider keeps at most ONE prepared frame live, so a cell
    is never materialised as an ``N x patch`` cache. ``from_sources`` rebuilds
    the exact preparation of :func:`build_aligned_cache_from_sources` (same
    sorted order, same section read / WCS slice / reprojection / support).

    The bounded product path opens this provider from a serializable source recipe.
    """

    storage_domain = "windowed_sources"
    live_window = 1

    def __init__(
        self, prepare, n_frames, *, frame_ids=None, expected_hw=None,
        prepare_window=None, expected_channels=None, window_stats=None,
    ):
        self._prepare = prepare
        # ZM-M33 lot 1: optional WINDOWED preparation. ``prepare_window(i, y0,
        # y1, x0, x1)`` must return the exact same pixels as preparing the whole
        # cell and slicing that window. When absent (or when the caller asks for
        # the full cell), the historical full-cell path is used verbatim.
        self._prepare_window = prepare_window
        # Caller-owned mutable counters (decoded/reprojected source pixels).
        self._window_stats = window_stats
        self.n_frames = int(n_frames)
        if self.n_frames < 0:
            raise CanonicalStackValidationError("n_frames must be >= 0")
        ids = (
            tuple(str(i) for i in range(self.n_frames))
            if frame_ids is None
            else tuple(str(x) for x in frame_ids)
        )
        if len(ids) != self.n_frames:
            raise CanonicalStackValidationError(
                "frame_ids length must equal n_frames"
            )
        self.frame_ids = ids
        self.prepared_count = 0
        self.window_prepared_count = 0
        self.full_prepared_count = 0
        self.bytes_read = 0
        self._index = None
        self._rgb = None
        self._sup = None
        self._win_key = None
        self._win_rgb = None
        self._win_sup = None
        self._closed = False
        self.height = self.width = self.channels = None
        self.original_mono = None
        self.original_ndim = None
        self.original_shape = None
        if expected_hw is not None:
            self.height, self.width = (int(expected_hw[0]), int(expected_hw[1]))
        if expected_channels is not None:
            # Declared layout: no whole-cell probe frame is prepared just to
            # learn the shape (that probe would be a full-cell reprojection).
            c = int(expected_channels)
            if c < 1:
                raise CanonicalStackValidationError("expected_channels must be >= 1")
            self.channels = c
            self.original_mono = bool(c == 1)
            self.original_ndim = 2 if self.original_mono else 3
            if self.height is not None and self.width is not None:
                self.original_shape = (
                    (self.height, self.width)
                    if self.original_mono
                    else (self.height, self.width, c)
                )
        elif self.n_frames:
            rgb, _sup = self._load(0)
            h, w, c = rgb.shape
            if expected_hw is not None and (h, w) != (int(expected_hw[0]), int(expected_hw[1])):
                raise CanonicalStackValidationError(
                    f"prepared frame shape {(h, w)} != expected {tuple(expected_hw)}"
                )
            self.height, self.width, self.channels = int(h), int(w), int(c)
            self.original_mono = bool(c == 1)
            self.original_ndim = 2 if self.original_mono else 3
            self.original_shape = (h, w) if self.original_mono else (h, w, c)

    @classmethod
    def from_sources(cls, frames, prepared_paths, canvas, patch, crop_plans, *, tracker=None):
        """Mirror ``build_aligned_cache_from_sources`` without any cache write."""
        patch_wcs = patch.patch_wcs()
        ordered = []
        for f in sorted(frames, key=lambda f: f.frame_id):
            key = f.frame_id.logical_path
            plan = crop_plans.get(key)
            if plan is None:
                continue
            prep = prepared_paths.get(key)
            if prep is None:
                raise ValueError(f"no prepared RGB for {key}")
            ordered.append((f, plan, prep))

        def prepare(i):
            f, plan, prep = ordered[int(i)]
            cropped = zxe._read_section_chw(prep, plan.source_bounds, tracker=tracker)
            cropped_wcs = zxe.slice_wcs(f.wcs(), plan.source_bounds)
            rgb, geom = zxe.reproject_cropped(
                cropped, cropped_wcs, patch_wcs, patch.patch_shape_hw
            )
            return rgb, geom

        return cls(
            prepare,
            len(ordered),
            frame_ids=[f.frame_id.logical_path for f, _p, _q in ordered],
            expected_hw=patch.patch_shape_hw,
        )

    def _load(self, i):
        if self._closed:
            raise RuntimeError("WindowedCanonicalProvider is closed")
        i = int(i)
        if i < 0 or i >= self.n_frames:
            raise IndexError(f"frame index {i} outside [0, {self.n_frames})")
        if self._index == i:
            return self._rgb, self._sup
        # Release the previous owned window BEFORE decode/reprojection starts.
        # If preparation raises, no stale index may resurrect the old frame.
        self._rgb = self._sup = None
        self._index = None
        # ``live_window == 1``: a full-cell frame and a window never coexist.
        self._win_key = None
        self._win_rgb = self._win_sup = None
        rgb, sup = self._prepare(i)
        rgb = np.ascontiguousarray(np.asarray(rgb, dtype=np.float32))
        sup = np.ascontiguousarray(np.asarray(sup, dtype=bool))
        if rgb.ndim != 3:
            raise CanonicalStackValidationError("prepared frame must be (H,W,C)")
        if sup.shape != rgb.shape[:2]:
            raise CanonicalStackValidationError("support shape must be (H,W)")
        if self.height is not None and self.width is not None:
            if rgb.shape[:2] != (self.height, self.width):
                raise CanonicalStackValidationError("prepared frame changed spatial shape")
        if self.channels is not None and rgb.shape[2] != self.channels:
            raise CanonicalStackValidationError("prepared frame changed channel count")
        self._index = i
        self._rgb, self._sup = rgb, sup
        self.prepared_count += 1
        self.full_prepared_count += 1
        self.bytes_read += int(rgb.nbytes) + int(sup.nbytes)
        return self._rgb, self._sup

    def _load_window(self, i, window):
        """Prepare ONLY the requested patch window (bit-exact, one live window)."""
        if self._closed:
            raise RuntimeError("WindowedCanonicalProvider is closed")
        i = int(i)
        if i < 0 or i >= self.n_frames:
            raise IndexError(f"frame index {i} outside [0, {self.n_frames})")
        key = (i, int(window[0]), int(window[1]), int(window[2]), int(window[3]))
        if self._win_key == key:
            return self._win_rgb, self._win_sup
        # Release everything BEFORE decode/reprojection starts.
        self._win_rgb = self._win_sup = None
        self._win_key = None
        self._rgb = self._sup = None
        self._index = None
        rgb, sup = self._prepare_window(*key)
        rgb = np.ascontiguousarray(np.asarray(rgb, dtype=np.float32))
        sup = np.ascontiguousarray(np.asarray(sup, dtype=bool))
        if rgb.ndim != 3:
            raise CanonicalStackValidationError("prepared window must be (H,W,C)")
        expected_hw = (key[2] - key[1], key[4] - key[3])
        if (int(rgb.shape[0]), int(rgb.shape[1])) != expected_hw:
            raise CanonicalStackValidationError(
                f"prepared window shape {rgb.shape[:2]} != requested {expected_hw}"
            )
        if sup.shape != rgb.shape[:2]:
            raise CanonicalStackValidationError("support shape must be (H,W)")
        if self.channels is not None and rgb.shape[2] != self.channels:
            raise CanonicalStackValidationError("prepared window changed channel count")
        self._win_key = key
        self._win_rgb, self._win_sup = rgb, sup
        self.prepared_count += 1
        self.window_prepared_count += 1
        self.bytes_read += int(rgb.nbytes) + int(sup.nbytes)
        return self._win_rgb, self._win_sup

    @property
    def shape(self):
        return (self.height, self.width)

    def window_telemetry(self):
        """Honest counters proving the window bound (source pixels actually read).

        ``prepared_count`` counts EVERY preparation (window or full cell);
        ``source_pixels_decoded``/``reprojected_pixels`` come from the real
        section reads and real reprojection windows, never from the request.
        """
        out = dict(self._window_stats or {})
        out.update({
            "prepared_count": int(self.prepared_count),
            "window_prepared_count": int(self.window_prepared_count),
            "full_prepared_count": int(self.full_prepared_count),
            "prepared_aligned_bytes": int(self.bytes_read),
        })
        return out

    def get_raw_frame(self, i):
        rgb, sup = self._load(i)
        return np.array(rgb, copy=True), np.array(sup, copy=True)

    def get_tile(self, i, y0, y1, x0, x1):
        if self._prepare_window is not None:
            # Bounded path: decode + reproject ONLY this window. The pixels are
            # bit-identical to the full-cell preparation sliced to the window.
            tile, sup_tile = self._load_window(i, (y0, y1, x0, x1))
            batch = prepare_canonical_inputs([tile], [sup_tile])
            return batch.images[0], batch.valid_mask[0]
        rgb, sup = self._load(i)
        if self.original_mono:
            tile = rgb[y0:y1, x0:x1]
        else:
            tile = rgb[y0:y1, x0:x1, :]
        sup_tile = sup[y0:y1, x0:x1]
        # EXACT cache-provider semantics: the returned pair is the canonical
        # preparation of the tile, not the raw slice.
        batch = prepare_canonical_inputs([tile], [sup_tile])
        return batch.images[0], batch.valid_mask[0]

    def close(self):
        self._rgb = self._sup = None
        self._index = None
        self._win_rgb = self._win_sup = None
        self._win_key = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
