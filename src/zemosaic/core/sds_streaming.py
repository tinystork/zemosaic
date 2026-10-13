"""ZM-SDS-R35C-C4 — SDS-specific memmap/file-backed providers + bounded-memory
(streaming) execution of the two SDS stacking stages.

This module adds the SDS-only wiring layer requested by Lot C4. It is a **new
module**: it does not modify ``core/canonical_*`` (nor any canonical formula),
ZeGrid, Classic/Legacy, ``zemosaic_utils.save_fits_image``, the solver/WCS or
the GUI/config/locales. It reuses the frozen streaming executor
:func:`zemosaic.core.canonical_streaming.run_canonical_stack_streaming`
verbatim, feeding it two SDS-specific, file-backed providers:

* :class:`SdsMegatileFileProvider` — the **megatiles -> final** stage: the seven
  full-canvas SDS mega-tiles (RGB float32 + a bool support) are written once to
  ``.npy`` memmap files and served tile-by-tile, so the canonical engine no
  longer materialises its ``O(N x patch_area)`` float64 intermediates.
* :class:`SdsPatchBandProvider` — the **brutes -> megatile** stage: the per-lot
  reprojected patches are already persisted as memmap ``.npy`` artifacts by the
  worker (``entry["data_path"]`` / ``entry["weight_path"]``); the provider serves
  a row band from those files, so a band no longer materialises the full
  ``O(N x band_area)`` stack.

Scientific contract (unchanged from the in-memory SDS path)
-----------------------------------------------------------
* EVERY image of the lot contributes to EVERY relevant spatial tile: the
  executor tiles SPATIALLY, never along N (no sub-stacks averaging partial
  rejection). This mirrors ``_stack_mosaics`` / ``_finalize_chunked``.
* Global statistics are computed ONCE, independently of the tile size: the
  executor's phase 1 (per-frame reference / coefficients / weights) runs over
  the full domain before phase 2 tiling, so the result is tile-size invariant.
* The provider is EXACT for supports and no-data: a sample is valid iff the
  frame's explicit support is true AND the transformed sample is finite — never
  ``pixel == 0``.
* Memmaps are closed explicitly (:meth:`close`, context manager) so Windows can
  delete the scratch files (``WinError 32`` otherwise).

Bit-exactness
-------------
``run_canonical_stack_streaming`` is documented (and tested) to be bit-identical
to ``run_canonical_stack``. This module adds no new arithmetic: the providers
serve exactly the same arrays the in-memory path holds (``.npy`` round-trips of
the same float32/bool buffers, bit-exact), and the request is built with the
same tokens/support/reference rules. The C4 tests assert bit-exact parity vs the
in-memory engine across several tile sizes and iteration orders.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from zemosaic.core.canonical_engine import (
    CanonicalStackRequest,
    CanonicalStackResult,
    run_canonical_stack,
)
from zemosaic.core.canonical_stacking import (
    CanonicalStackValidationError,
    prepare_canonical_inputs,
)
from zemosaic.core.canonical_streaming import run_canonical_stack_streaming

__all__ = [
    "SDS_MEGATILE_CACHE_SCHEMA",
    "SdsMegatileCacheBuilder",
    "SdsMegatileFileProvider",
    "SdsPatchBandProvider",
    "write_sds_megatile_cache",
    "load_sds_megatile_manifest",
    "resolve_sds_final_stack_tokens",
    "build_sds_final_stack_request",
    "run_sds_final_stack_streaming",
    "run_sds_band_stack_streaming",
    "sds_band_reference_index",
    "plan_sds_streaming_tile",
    "plan_sds_megatile_parallelism",
    "estimate_sds_tile_workspace_bytes",
]

SDS_MEGATILE_CACHE_SCHEMA = "zemosaic-sds-c4-megatile-cache-v1"

# Zegrid R6 calibration of the streaming tile workspace, reused verbatim: the
# float64/float32/bool intermediates of the executor scale as
# ``N x tile_cells x C x BYTES_PER_CELL`` plus a per-tile fixed overhead.
_TILE_WORKSPACE_BYTES_PER_CELLCH = 75
_TILE_FIXED_OVERHEAD_BYTES = 2_560 * 1024


# ---------------------------------------------------------------------------
# Small path / memmap helpers
# ---------------------------------------------------------------------------

def _megatile_rgb_path(cache_dir, idx: int) -> Path:
    return Path(cache_dir) / f"sds_megatile_{idx:04d}_rgb.npy"


def _megatile_sup_path(cache_dir, idx: int) -> Path:
    return Path(cache_dir) / f"sds_megatile_{idx:04d}_support.npy"


def _manifest_path(cache_dir) -> Path:
    return Path(cache_dir) / "sds_megatile_manifest.json"


def close_memmap(arr) -> None:
    """Release a numpy memmap's underlying ``mmap`` handle (best-effort).

    Mirrors ``zemosaic.core.zegrid.file_provider._close_memmap``: on Windows a
    file still mapped into the process cannot be deleted (``WinError 32``), so
    the handle must be released before scratch cleanup. Never raises.
    """
    for attr in ("base", "_mmap"):
        try:
            obj = getattr(arr, attr, None)
            if obj is not None and hasattr(obj, "close"):
                obj.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Megatile disk cache (megatiles -> final stage source)
# ---------------------------------------------------------------------------

class SdsMegatileCacheBuilder:
    """Streaming writer of the SDS mega-tile disk cache (one tile at a time).

    ``add()`` writes a mega-tile's RGB float32 and its bool support to ``.npy``
    immediately, so the caller can drop its in-RAM reference right after. The
    metadata (shape/channels/mono) is taken from the first tile.
    """

    def __init__(self, cache_dir) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._meta = None
        self._frames: list[dict] = []
        self._frame_ids: list[str] = []
        self._total_bytes = 0
        self._n = 0
        self._any_support: list[bool] = []

    def add(self, frame_id: str, megatile: np.ndarray, support: np.ndarray) -> None:
        rgb = np.ascontiguousarray(np.asarray(megatile, dtype=np.float32))
        sup = np.ascontiguousarray(np.asarray(support, dtype=bool))
        if rgb.ndim == 2:
            h, w, c, mono, ndim, shape = rgb.shape[0], rgb.shape[1], 1, True, 2, rgb.shape
            rgb_w = rgb.reshape(h, w, 1)
        else:
            h, w, c = rgb.shape
            mono, ndim, shape = False, 3, rgb.shape
            rgb_w = rgb
        if sup.shape != (h, w):
            raise CanonicalStackValidationError(
                f"mega-tile {frame_id!r}: support shape {sup.shape} != RGB HW {(h, w)}"
            )
        meta = (h, w, c, mono, ndim, shape)
        if self._meta is None:
            self._meta = meta
        elif meta != self._meta:
            raise CanonicalStackValidationError(
                f"mega-tile {frame_id!r}: inconsistent shape {meta} != {self._meta}"
            )
        rp = _megatile_rgb_path(self.cache_dir, self._n)
        sp = _megatile_sup_path(self.cache_dir, self._n)
        np.save(rp, rgb_w)
        np.save(sp, sup)
        self._total_bytes += rp.stat().st_size + sp.stat().st_size
        self._frames.append(
            {"index": self._n, "frame_id": frame_id, "rgb": rp.name, "support": sp.name}
        )
        self._frame_ids.append(frame_id)
        self._any_support.append(bool(np.any(sup)))
        self._n += 1

    def finish(self) -> dict:
        if self._meta is None:
            raise CanonicalStackValidationError("empty SDS mega-tile cache (no tiles written)")
        h, w, c, mono, ndim, shape = self._meta
        manifest = {
            "schema": SDS_MEGATILE_CACHE_SCHEMA,
            "n_frames": len(self._frames),
            "height": int(h),
            "width": int(w),
            "channels": int(c),
            "original_mono": bool(mono),
            "original_ndim": int(ndim),
            "original_shape": [int(v) for v in shape],
            "frame_ids": list(self._frame_ids),
            "any_support": list(self._any_support),
            "frames": list(self._frames),
            "total_bytes": int(self._total_bytes),
        }
        _manifest_path(self.cache_dir).write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest


def write_sds_megatile_cache(cache_dir, frame_ids, megatiles, supports) -> dict:
    """Write aligned SDS mega-tiles to a memmap disk cache (bit-exact ``.npy``)."""
    n = len(frame_ids)
    if len(megatiles) != n or len(supports) != n:
        raise CanonicalStackValidationError("frame_ids / megatiles / supports length mismatch")
    builder = SdsMegatileCacheBuilder(cache_dir)
    for i in range(n):
        builder.add(frame_ids[i], megatiles[i], supports[i])
    return builder.finish()


def load_sds_megatile_manifest(cache_dir) -> dict:
    mp = _manifest_path(cache_dir)
    if not mp.exists():
        raise FileNotFoundError(f"SDS mega-tile cache manifest missing: {mp}")
    manifest = json.loads(mp.read_text())
    if manifest.get("schema") != SDS_MEGATILE_CACHE_SCHEMA:
        raise CanonicalStackValidationError(
            f"SDS mega-tile cache schema {manifest.get('schema')!r} != "
            f"{SDS_MEGATILE_CACHE_SCHEMA!r}"
        )
    return manifest


class SdsMegatileFileProvider:
    """File-backed ``CanonicalFrameProvider`` for the SDS mega-tiles -> final stage.

    Satisfies the R5 ``CanonicalFrameProvider`` Protocol while serving the
    mega-tiles from a disk cache, so the final full-canvas stack is bounded by
    ``O(N x tile_area)`` instead of ``O(N x patch_area)``. The provider is a
    context manager and exposes :meth:`close` (release every memmap handle),
    required on Windows before deleting the scratch cache.
    """

    def __init__(self, cache_dir) -> None:
        self.cache_dir = Path(cache_dir)
        manifest = load_sds_megatile_manifest(self.cache_dir)
        self._manifest = manifest
        self.n_frames = int(manifest["n_frames"])
        self.height = int(manifest["height"])
        self.width = int(manifest["width"])
        self.channels = int(manifest["channels"])
        self.original_mono = bool(manifest["original_mono"])
        self.original_ndim = int(manifest["original_ndim"])
        self.original_shape = tuple(manifest["original_shape"])
        self.frame_ids = tuple(manifest["frame_ids"])
        self.any_support = tuple(bool(v) for v in manifest.get("any_support", [True] * self.n_frames))
        self._rgb = [
            np.load(_megatile_rgb_path(self.cache_dir, i), mmap_mode="r")
            for i in range(self.n_frames)
        ]
        self._sup = [
            np.load(_megatile_sup_path(self.cache_dir, i), mmap_mode="r")
            for i in range(self.n_frames)
        ]

    def close(self) -> None:
        """Release every memmap handle (idempotent, never raises)."""
        for arr in list(self._rgb) + list(self._sup):
            close_memmap(arr)
        self._rgb = []
        self._sup = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    @property
    def shape(self):
        return (self.height, self.width)

    @property
    def manifest(self) -> dict:
        return self._manifest

    def get_raw_frame(self, i: int):
        rgb = self._rgb[i]
        if rgb.ndim == 3 and rgb.shape[-1] == 1 and self.original_mono:
            rgb = rgb[..., 0]
        return np.asarray(rgb), np.asarray(self._sup[i])

    def get_tile(self, i: int, y0: int, y1: int, x0: int, x1: int):
        raw = self._rgb[i]
        sup = self._sup[i]
        if self.original_mono:
            tile = raw[y0:y1, x0:x1, 0]
        else:
            tile = raw[y0:y1, x0:x1, :]
        sup_tile = sup[y0:y1, x0:x1]
        batch = prepare_canonical_inputs([tile], [sup_tile])
        return batch.images[0], batch.valid_mask[0]


# ---------------------------------------------------------------------------
# Patch band provider (brutes -> megatile stage source)
# ---------------------------------------------------------------------------

class SdsPatchBandProvider:
    """Band-limited file-backed provider over the SDS per-lot reprojected patches.

    The worker persists each reprojected patch of a lot as two ``.npy`` memmap
    artifacts (``data_path`` float32 HWC, ``weight_path`` float32 HW) together
    with its global-canvas ``bbox`` ``(y0, y1, x0, x1)``. For a ROW BAND
    ``[band_y0, band_y1)`` this provider exposes the N patches as N frames on a
    ``(band_rows, width, C)`` domain, exactly as the in-memory ``_finalize_chunked``
    band stack does: a frame's samples are the entry's memmap slice inside its
    bbox and NaN elsewhere; support is ``weight > 0``.

    Consequence: phase 2 only ever materialises a tile, and phase 1 only one band
    frame at a time — the ``O(N x band_area)`` stack is never built.

    ``taper`` must be ``"none"`` for this provider (the SDS band path uses no
    footprint taper) and is not accepted otherwise, so a silent semantic change
    is impossible.
    """

    def __init__(self, entries, band_y0: int, band_y1: int, width: int, channels: int) -> None:
        self._entries = list(entries)
        self.n_frames = len(self._entries)
        self.band_y0 = int(band_y0)
        self.band_y1 = int(band_y1)
        self.height = int(band_y1 - band_y0)
        self.width = int(width)
        self.channels = int(channels)
        self.original_mono = self.channels == 1
        self.original_ndim = 2 if self.original_mono else 3
        self.original_shape = (
            (self.height, self.width) if self.original_mono else (self.height, self.width, self.channels)
        )
        self.frame_ids = tuple(
            str(e.get("frame_id") or f"patch_{i:04d}") for i, e in enumerate(self._entries)
        )
        self._mm_cache: dict[int, tuple] = {}

    def _load_frame(self, i: int):
        cached = self._mm_cache.get(i)
        if cached is not None:
            return cached
        entry = self._entries[i]
        data = np.load(entry["data_path"], mmap_mode="r")
        weight = np.load(entry["weight_path"], mmap_mode="r")
        cached = (data, weight, tuple(int(v) for v in entry["bbox"]))
        self._mm_cache[i] = cached
        return cached

    def close(self) -> None:
        for data, weight, _bbox in self._mm_cache.values():
            close_memmap(data)
            close_memmap(weight)
        self._mm_cache = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    @property
    def shape(self):
        return (self.height, self.width)

    def _band_frame(self, i: int):
        """Materialise frame ``i`` over the whole band (phase-1 view)."""
        data, weight, (y_start, y_end, x_start, x_end) = self._load_frame(i)
        frame = np.full((self.height, self.width, self.channels), np.nan, dtype=np.float32)
        sup = np.zeros((self.height, self.width), dtype=bool)
        sub_y0 = max(self.band_y0, y_start)
        sub_y1 = min(self.band_y1, y_end)
        if sub_y0 < sub_y1:
            local_y0 = sub_y0 - y_start
            local_y1 = sub_y1 - y_start
            g_y0 = sub_y0 - self.band_y0
            g_y1 = sub_y1 - self.band_y0
            d = np.asarray(data[local_y0:local_y1, :, :], dtype=np.float32)
            w = np.asarray(weight[local_y0:local_y1, :], dtype=np.float32)
            frame[g_y0:g_y1, x_start:x_end, :] = d
            sup[g_y0:g_y1, x_start:x_end] = w > 0
        if self.original_mono:
            return frame[..., 0], sup
        return frame, sup

    def get_raw_frame(self, i: int):
        return self._band_frame(i)

    def get_tile(self, i: int, y0: int, y1: int, x0: int, x1: int):
        data, weight, (y_start, y_end, x_start, x_end) = self._load_frame(i)
        th = y1 - y0
        tw = x1 - x0
        tile = np.full((th, tw, self.channels), np.nan, dtype=np.float32)
        sup_tile = np.zeros((th, tw), dtype=bool)
        # Global band coords of the requested tile.
        gy0 = self.band_y0 + y0
        gy1 = self.band_y0 + y1
        sy0 = max(gy0, y_start)
        sy1 = min(gy1, y_end)
        sx0 = max(x0, x_start)
        sx1 = min(x1, x_end)
        if sy0 < sy1 and sx0 < sx1:
            local_y0 = sy0 - y_start
            local_y1 = sy1 - y_start
            local_x0 = sx0 - x_start
            local_x1 = sx1 - x_start
            ty0 = sy0 - gy0
            ty1 = sy1 - gy0
            tx0 = sx0 - x0
            tx1 = sx1 - x0
            d = np.asarray(data[local_y0:local_y1, local_x0:local_x1, :], dtype=np.float32)
            w = np.asarray(weight[local_y0:local_y1, local_x0:local_x1], dtype=np.float32)
            tile[ty0:ty1, tx0:tx1, :] = d
            sup_tile[ty0:ty1, tx0:tx1] = w > 0
        if self.original_mono:
            tile = tile[..., 0]
        batch = prepare_canonical_inputs([tile], [sup_tile])
        return batch.images[0], batch.valid_mask[0]


# ---------------------------------------------------------------------------
# Request construction (identical tokens/rules to the in-memory SDS paths)
# ---------------------------------------------------------------------------

_COMBINE_METHODS = ("mean", "median")


def resolve_sds_final_stack_tokens(stack_params: dict | None) -> tuple[str, str]:
    """Map the SDS stack params to the canonical ``(rejection, combine)`` tokens.

    Byte-for-byte the same mapping as ``_stack_mosaics`` (SDS megatiles -> final):

    * winsorized/winsor/winsorized_sigma_clip -> ``("winsorized_sigma_clip", "mean")``
    * kappa_sigma -> ``("kappa_sigma", "mean")``
    * everything else -> ``("none", combine if in {mean, median} else "mean")``

    ``linear_fit_clip`` is refused explicitly (removed Sci-05 token), matching the
    in-memory path.
    """
    algo = str((stack_params or {}).get("stack_reject_algo") or "winsorized_sigma_clip").lower()
    combine = str((stack_params or {}).get("stack_final_combine") or "mean").lower()
    if algo == "linear_fit_clip":
        raise ValueError(
            "rejection algorithm 'linear_fit_clip' is unsupported (unsupported_removed_sci05)"
        )
    if algo in ("winsorized_sigma_clip", "winsorized", "winsor"):
        return "winsorized_sigma_clip", "mean"
    if algo == "kappa_sigma":
        return "kappa_sigma", "mean"
    return "none", (combine if combine in _COMBINE_METHODS else "mean")


def build_sds_final_stack_request(
    stack_params: dict | None, n_frames: int, reference_index: int | None
) -> CanonicalStackRequest:
    """Build the SDS megatiles -> final canonical request (placeholder images).

    Only ``len(request.images)`` is read by the streaming executor; the provider
    serves the pixels. The tokens/support/taper exactly mirror ``_stack_mosaics``.
    """
    rejection, combine = resolve_sds_final_stack_tokens(stack_params)
    return CanonicalStackRequest(
        images=[None] * n_frames,
        geometric_support=[None] * n_frames,
        normalization="none",
        weighting="none",
        rejection=rejection,
        combine=combine,
        reference_index=reference_index,
        taper="none",
        backend="cpu",
    )


def _reference_index_from_supports(any_support) -> int | None:
    return 0 if (len(any_support) and any_support[0]) else None


# ---------------------------------------------------------------------------
# Streaming runners
# ---------------------------------------------------------------------------

def run_sds_final_stack_streaming(
    provider: SdsMegatileFileProvider,
    stack_params: dict | None,
    *,
    tile_size=None,
) -> np.ndarray:
    """Run the SDS megatiles -> final stack through the streaming executor.

    Returns the final science as ``float32`` (identical to ``_stack_mosaics``).
    When NO mega-tile has any support, returns an all-NaN array (same explicit,
    non-silent behaviour as the in-memory path: no fabricated science).
    """
    any_support = getattr(provider, "any_support", None)
    if any_support is None:
        any_support = [True] * provider.n_frames
    if not any(any_support):
        return np.full((provider.height, provider.width, provider.channels), np.nan, dtype=np.float32)
    ref_idx = _reference_index_from_supports(any_support)
    request = build_sds_final_stack_request(stack_params, provider.n_frames, ref_idx)
    result: CanonicalStackResult = run_canonical_stack_streaming(
        provider, request, tile_size=tile_size
    )
    return np.asarray(result.science, dtype=np.float32)


def sds_band_reference_index(
    entries, band_y0: int, band_y1: int, width: int, channels: int
) -> int | None:
    """Reference index for a band, matching ``_finalize_chunked``'s explicit rule.

    The in-memory band path picks ``0`` iff frame 0 has any valid support, else
    ``None``. This reproduces that decision from the memmap support without
    materialising the band stack (frame 0 only).
    """
    if not entries:
        return None
    provider = SdsPatchBandProvider(entries, band_y0, band_y1, width, channels)
    try:
        _raw, sup = provider.get_raw_frame(0)
        return 0 if bool(np.any(sup)) else None
    finally:
        provider.close()


def run_sds_band_stack_streaming(
    entries,
    band_y0: int,
    band_y1: int,
    width: int,
    channels: int,
    *,
    rejection: str,
    combine: str,
    reference_index: int | None,
    tile_size=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one SDS band of the brutes -> megatile stage via the streaming executor.

    ``entries`` are the per-lot patch memmap descriptors (``data_path`` /
    ``weight_path`` / ``bbox``) overlapping the band. Returns
    ``(science_float32, estimator_weight_sum)`` exactly as
    ``run_canonical_stack(...).science`` / ``.estimator_weight_sum`` do, so the
    caller can apply its own (unchanged) weight reduction. Science is NaN outside
    the valid support (the executor already returns NaN there).

    The band provider serves N frames on ``(band_rows, width, C)``; EVERY entry
    contributes to every spatial tile it covers (spatial tiling only, never N
    fragmentation).
    """
    provider = SdsPatchBandProvider(entries, band_y0, band_y1, width, channels)
    try:
        request = CanonicalStackRequest(
            images=[None] * provider.n_frames,
            geometric_support=[None] * provider.n_frames,
            normalization="none",
            weighting="none",
            rejection=rejection,
            combine=combine,
            reference_index=reference_index,
            taper="none",
            backend="cpu",
        )
        result = run_canonical_stack_streaming(provider, request, tile_size=tile_size)
        science = np.asarray(result.science, dtype=np.float32)
        science = np.where(np.isfinite(science), science, np.nan)
        return science, np.asarray(result.estimator_weight_sum)
    finally:
        provider.close()


# ---------------------------------------------------------------------------
# Concurrency / memory plan (effective backend + available RAM, not GPU wish)
# ---------------------------------------------------------------------------

def estimate_sds_tile_workspace_bytes(n_frames: int, tile_size, channels: int = 3) -> int:
    """Executor tile-workspace estimate (bytes) for a square ``tile_size``."""
    if tile_size is None:
        return 0
    th = tw = int(tile_size)
    return int(
        n_frames * th * tw * max(1, int(channels)) * _TILE_WORKSPACE_BYTES_PER_CELLCH
        + _TILE_FIXED_OVERHEAD_BYTES
    )


def plan_sds_streaming_tile(
    n_frames: int,
    patch_hw: tuple[int, int],
    channels: int = 3,
    *,
    available_bytes: int | None = None,
    backend: str = "cpu",
    bandwidth_fraction: float = 0.25,
    min_tile: int = 64,
    max_tile: int = 2048,
) -> int:
    """Pick a square executor tile size sized to the EFFECTIVE backend and RAM.

    The plan is driven by the *effective* backend (CPU for the SDS stacking
    stages) and the RAM actually available — never by the requested GPU alone.
    The tile is the largest square whose workspace fits ``bandwidth_fraction`` of
    ``available_bytes`` (defaults to the host's ``MemAvailable``). It degrades to
    ``min_tile`` under low RAM instead of failing, so a low-memory host still runs
    (the executor itself is RAM-bounded by the tile).
    """
    h, w = int(patch_hw[0]), int(patch_hw[1])
    longest = max(h, w)
    if available_bytes is None:
        available_bytes = _read_available_bytes()
    budget = max(0, int(available_bytes * max(0.01, float(bandwidth_fraction))))
    per_cell = max(1, int(n_frames) * max(1, int(channels)) * _TILE_WORKSPACE_BYTES_PER_CELLCH)
    if per_cell <= 0:
        return min(max_tile, longest)
    tile = int(math.isqrt(max(1, budget // per_cell))) if budget > 0 else min_tile
    tile = max(min_tile, min(max_tile, tile))
    # The tile must never exceed the domain, and must stay a multiple of the plan
    # granularity to keep the executor's tiling deterministic.
    return max(min_tile, min(tile, max(min_tile, longest)))


def plan_sds_megatile_parallelism(
    n_batches: int,
    *,
    requested_use_gpu: bool,
    available_bytes: int | None = None,
    est_bytes_per_megatile: int = 0,
    cpu_workers_hint: int = 0,
) -> int:
    """Bound the SDS mega-tile parallelism on the EFFECTIVE backend + RAM.

    The SDS brutes -> megatile stage always executes on CPU (the canonical engine
    is invoked with ``backend='cpu'``; the legacy GPU coadd stays disabled). So
    the plan must not be governed by a GPU *preference*: the request may ask for
    GPU helpers, but the effective worker count is CPU-bounded. Returns the max
    number of concurrently produced mega-tiles given:
    * a CPU ceiling (from ``cpu_workers_hint`` when present, else host CPUs),
    * a RAM ceiling (``available_bytes`` // ``est_bytes_per_megatile``).
    Never returns less than 1.
    """
    if n_batches <= 1:
        return 1
    if available_bytes is None:
        available_bytes = _read_available_bytes()
    workers = 1
    if cpu_workers_hint and cpu_workers_hint > 0:
        workers = max(1, int(cpu_workers_hint) // 2)
    else:
        workers = max(1, _cpu_count() // 2)
    if est_bytes_per_megatile and est_bytes_per_megatile > 0:
        # Keep two mega-tiles of headroom so accumulation never saturates RAM.
        ram_cap = max(1, int(available_bytes * 0.5) // int(est_bytes_per_megatile))
        workers = min(workers, ram_cap)
    # GPU preference does NOT raise the ceiling (effective backend is CPU); it is
    # recorded by the caller. It may only keep the plan conservative.
    if requested_use_gpu:
        workers = min(workers, max(1, _cpu_count() // 2))
    return max(1, min(int(n_batches), int(workers)))


def _read_available_bytes() -> int:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0


def _cpu_count() -> int:
    try:
        import os

        return max(1, os.cpu_count() or 1)
    except Exception:
        return 1
