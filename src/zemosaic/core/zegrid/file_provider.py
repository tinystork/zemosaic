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
import json
from pathlib import Path

import numpy as np

from zemosaic.core.canonical_stacking import (
    CanonicalStackValidationError,
    prepare_canonical_inputs,
)

from . import execution as zxe
from .geometry import (
    FrameDescriptor,
    GlobalCanvas,
    ProcessingPatch,
    SourceCropPlan,
)

CACHE_SCHEMA = "zemosaic-zegrid-r6-aligned-cache-v1"

__all__ = [
    "CACHE_SCHEMA",
    "AlignedCacheBuilder",
    "MemmapCanonicalProvider",
    "build_aligned_cache_from_sources",
    "write_aligned_cache_from_arrays",
    "cache_manifest_path",
    "cache_is_complete",
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


def _array_meta(rgb: np.ndarray):
    rgb = np.asarray(rgb)
    if rgb.ndim == 2:
        h, w = rgb.shape
        return h, w, 1, True, 2, (h, w)
    h, w, c = rgb.shape
    return h, w, c, False, 3, (h, w, c)


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


def cache_is_complete(cache_dir, frame_ids: list[str]) -> bool:
    """True if the cache dir holds a matching, complete manifest + all files.

    Resumability check: rebuild is skipped only when the frame-id list matches
    AND every declared file exists (and matches its recorded sha256).
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

    def __init__(self, cache_dir) -> None:
        self.cache_dir = Path(cache_dir)
        if self.cache_dir.exists():
            import shutil

            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._meta = None
        self._frames: list[dict] = []
        self._frame_ids: list[str] = []
        self._total_bytes = 0
        self._n = 0

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
        h, w, c, mono, ndim, shape = self._meta
        manifest = {
            "schema": CACHE_SCHEMA,
            "n_frames": self._n,
            "height": h,
            "width": w,
            "channels": c,
            "original_mono": mono,
            "original_ndim": ndim,
            "original_shape": list(shape),
            "frame_ids": list(self._frame_ids),
            "frames": self._frames,
            "total_bytes": self._total_bytes,
        }
        cache_manifest_path(self.cache_dir).write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest


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
    if reuse_cache and cache_is_complete(cache_dir, frame_ids):
        return load_cache_manifest(cache_dir)

    builder = AlignedCacheBuilder(cache_dir)
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
    """

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
        self._rgb = [
            np.load(_frame_rgb_path(self.cache_dir, i), mmap_mode="r")
            for i in range(self.n_frames)
        ]
        self._sup = [
            np.load(_frame_sup_path(self.cache_dir, i), mmap_mode="r")
            for i in range(self.n_frames)
        ]

    # -- metadata -----------------------------------------------------------
    @property
    def shape(self):
        return (self.height, self.width)

    @property
    def manifest(self) -> dict:
        return self._manifest

    # -- phase 1 raw access -------------------------------------------------
    def get_raw_frame(self, i: int):
        return np.asarray(self._rgb[i]), np.asarray(self._sup[i])

    # -- phase 2 prepared tile access (mirrors InMemoryCanonicalProvider) ----
    def get_tile(self, i: int, y0: int, y1: int, x0: int, x1: int):
        raw = self._rgb[i]
        sup = self._sup[i]
        if self.original_mono:
            tile = raw[y0:y1, x0:x1]
        else:
            tile = raw[y0:y1, x0:x1, :]
        sup_tile = sup[y0:y1, x0:x1]
        batch = prepare_canonical_inputs([tile], [sup_tile])
        return batch.images[0], batch.valid_mask[0]
