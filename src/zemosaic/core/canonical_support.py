"""SCI-05 Gate E1 — canonical positive support + footprint taper primitives.

Pure CPU, deterministic primitives implementing the donor-exact coverage-support
domain for the frozen SCI-05 contract
(``docs/science/SCI05_CANONICAL_STACKING_CONTRACT.md``), **source-ported** from
the ZSSS donor (``seestar/enhancement/weight_utils.py`` +
``seestar/core/coverage_support.py`` @ ``9b891de``) with **no runtime import or
dependency on ZSSS**:

* ``make_footprint_taper`` — the footprint-following feather taper (``1`` deep
  interior, monotone ramp to ``floor`` near the true boundary, ``0`` outside),
  computed from a scipy EDT primary path with a donor chamfer fallback. Never a
  radial/centre distance.
* ``PositiveSupportAccumulator`` / ``accumulate_support_pair`` — the atomic
  positive-support pair ``SUP_W1 += s_i`` / ``SUP_W2 += s_i**2`` with the derived
  ``N_eff = SUP_W1**2 / SUP_W2`` view (exact-first + overflow-resistant fallback).
* ``build_canonical_estimator_weights`` — integration glue that builds the explicit
  ``(N, H, W)`` float64 estimator-weight map ``w_i = q_i * m_i * a_i`` consumed by
  the C2 ``combine_canonical_samples``.

Gate E1 only: no Coverage render, no final request/result engine, no
GUI/config/migration/locales, no production caller wiring, no CuPy.

Design notes
------------
* Channel-invariant exactly 2-D support maps; no per-channel copies.
* Accumulators are float64 by default (float32 remains an explicit opt-in); a
  per-add fail-before-mutation preflight guarantees the ``SUP_W1``/``SUP_W2`` pair
  updates atomically (both or neither).
* Validation errors: ``make_footprint_taper`` and the accumulator raise plain
  ``ValueError``/``TypeError`` (donor-exact); ``build_canonical_estimator_weights``
  raises :class:`zemosaic.core.canonical_stacking.CanonicalStackValidationError`
  (a ``ValueError`` subclass) so its output is directly consumable by the C2
  combine, which validates the same ``(N, H, W)`` float64 weight-map contract.
"""

from __future__ import annotations

import numpy as np

from zemosaic.core.canonical_stacking import CanonicalStackValidationError

__all__ = [
    "make_footprint_taper",
    "PositiveSupportAccumulator",
    "accumulate_support_pair",
    "build_canonical_estimator_weights",
]

# Supported accumulator dtypes (explicit, small allowlist).
SUPPORT_DTYPES = (np.dtype(np.float32), np.dtype(np.float64))

# Documented neutral value for N_eff_support where SUP_W2 is not positive.
N_EFF_UNDEFINED_VALUE = 0.0

# scipy EDT primary path (module-level reference so the chamfer fallback can be
# forced by monkeypatching; the donor imports this inside the function).
try:  # pragma: no cover - scipy is a declared dependency
    from scipy.ndimage import distance_transform_edt as _distance_transform_edt
except Exception:  # pragma: no cover - defensive
    _distance_transform_edt = None


# ---------------------------------------------------------------------------
# A. Footprint taper
# ---------------------------------------------------------------------------

def _footprint_distance_fallback(mask):
    """Approximate Euclidean distance to nearest boundary without scipy (donor).

    Chamfer two-pass with weights 2 (axis) / 3 (diagonal), divided by ``2.0`` so
    the taper follows the real footprint boundary (never a radial distance from
    the image centre). The mask is padded with invalid (False) support so the
    array boundary feathers symmetrically on all sides; the padding is stripped
    before returning.
    """
    m = np.pad(mask, 1, constant_values=False)
    h, w = m.shape
    INF = 1 << 20
    d = np.where(m, INF, 0).astype(np.int32)
    for i in range(1, h):
        up = d[i - 1]
        d[i] = np.minimum(d[i], up + 2)
        d[i, 1:] = np.minimum(d[i, 1:], up[:-1] + 3)
        d[i, :-1] = np.minimum(d[i, :-1], up[1:] + 3)
    for i in range(h - 2, -1, -1):
        dn = d[i + 1]
        d[i] = np.minimum(d[i], dn + 2)
        d[i, 1:] = np.minimum(d[i, 1:], dn[:-1] + 3)
        d[i, :-1] = np.minimum(d[i, :-1], dn[1:] + 3)
    for j in range(1, w):
        d[:, j] = np.minimum(d[:, j], d[:, j - 1] + 2)
    for j in range(w - 2, -1, -1):
        d[:, j] = np.minimum(d[:, j], d[:, j + 1] + 2)
    d = np.where(m, d, 0).astype(np.float32) / 2.0
    return d[1:-1, 1:-1]


def make_footprint_taper(mask, feather_px=8.0, floor=0.0):
    """Return a float32 ``(H, W)`` taper that follows the real footprint boundary.

    Coverage-aware taper (replaces the historical radial falloff): ``1.0`` in the
    interior of the valid footprint, ramping smoothly toward ``floor`` over
    ``feather_px`` pixels near the actual transformed support boundary, and
    ``0.0`` outside. Translation/rotation invariant (up to rasterization); never
    a radial distance from the image centre.

    The mask is padded with invalid (False) support before the distance transform
    so a footprint that fills the array feathers symmetrically from all four
    boundaries (the array exterior is a support boundary).
    """
    m = np.asarray(mask)
    if m.dtype != np.bool_:
        raise ValueError("make_footprint_taper: mask must be boolean")
    if m.ndim != 2:
        raise ValueError("make_footprint_taper: mask must be 2-D")
    h, w = m.shape
    floor = float(floor)
    if not np.isfinite(floor) or not (0.0 <= floor < 1.0):
        raise ValueError("make_footprint_taper: floor must be finite in [0, 1)")
    feather_px = float(feather_px)
    if not np.isfinite(feather_px) or feather_px <= 0.0:
        raise ValueError("make_footprint_taper: feather_px must be finite > 0")

    if not np.any(m):
        return np.zeros((h, w), dtype=np.float32)

    try:
        if _distance_transform_edt is None:
            raise ImportError("scipy.ndimage.distance_transform_edt unavailable")
        padded = np.pad(m, 1, constant_values=False)
        dist = _distance_transform_edt(padded)[1:-1, 1:-1]
    except Exception:
        dist = _footprint_distance_fallback(m)

    frac = np.clip(dist.astype(np.float32) / float(feather_px), 0.0, 1.0)
    taper = np.where(m, floor + (1.0 - floor) * frac, np.float32(0.0))
    return taper.astype(np.float32)


# ---------------------------------------------------------------------------
# B. Positive-support accumulator
# ---------------------------------------------------------------------------

def accumulate_support_pair(w1, w2, support, *, dtype=np.float64):
    """Atomically accumulate one positive support map into two external arrays.

    Mirrors :meth:`PositiveSupportAccumulator.add` exactly, but mutates
    caller-provided (in-memory or memmap) float32/float64 ``(H, W)`` arrays in
    place. This is the single source of truth for the atomic per-exposure support
    accumulation shared by the in-memory accumulator and the classic memmaps.

    Fail-before-mutation: a shape/dtype/finiteness/negativity/square-overflow or
    cumulative-overflow violation raises and leaves both arrays byte-identical.
    """
    dtype = np.dtype(dtype)
    if dtype not in SUPPORT_DTYPES:
        raise TypeError(f"support dtype must be float32/float64, got {dtype.name!r}")
    w1 = np.asarray(w1)
    w2 = np.asarray(w2)
    if w1.ndim != 2 or w2.ndim != 2:
        raise ValueError("support accumulators must be 2-D (H, W)")
    if w1.shape != w2.shape:
        raise ValueError("support accumulators must share a shape")
    if w1.dtype != dtype or w2.dtype != dtype:
        raise TypeError(
            f"support accumulators must have dtype {dtype}, got {w1.dtype}/{w2.dtype}"
        )
    if not (np.all(np.isfinite(w1)) and np.all(np.isfinite(w2))):
        raise ValueError("support accumulators contain non-finite samples")
    if np.any(w1 < 0.0) or np.any(w2 < 0.0):
        raise ValueError("support accumulators contain negative samples")
    shape = w1.shape
    s = np.asarray(support)
    if s.shape != shape:
        raise ValueError(
            f"support shape {s.shape} does not match accumulator shape {shape}"
        )
    if not np.issubdtype(s.dtype, np.floating):
        s = s.astype(dtype)
    elif s.dtype != dtype:
        s = s.astype(dtype, copy=False)
    else:
        s = s.astype(dtype, copy=False)
    if not np.all(np.isfinite(s)):
        raise ValueError("support must be finite (NaN/Inf rejected)")
    if np.any(s < 0.0):
        raise ValueError("support must be non-negative")
    with np.errstate(over="ignore"):
        s2 = s * s
    if not np.all(np.isfinite(s2)):
        raise ValueError("support**2 overflowed to non-finite")
    with np.errstate(over="ignore", invalid="ignore"):
        new_w1 = w1 + s
        new_w2 = w2 + s2
    if not (np.all(np.isfinite(new_w1)) and np.all(np.isfinite(new_w2))):
        raise ValueError(
            "cumulative support overflow: SUP_W1/SUP_W2 would become non-finite"
        )
    w1[:] = new_w1
    w2[:] = new_w2


class PositiveSupportAccumulator:
    """Accumulate positive per-original-exposure support (SUP_W1 / SUP_W2).

    Channel-invariant exactly 2-D (H, W) support maps; one per-pixel support map
    per original exposure, independent of colour-channel count. float64 by
    default (float32 is an explicit, memory-constrained opt-in).
    """

    def __init__(self, shape, *, dtype=np.float64):
        shape = self._normalize_shape(shape)
        dtype = np.dtype(dtype)
        if dtype not in SUPPORT_DTYPES:
            raise TypeError(
                f"support dtype must be one of {tuple(d.name for d in SUPPORT_DTYPES)}, "
                f"got {dtype.name!r}"
            )
        self._shape = shape
        self._dtype = dtype
        self._w1 = np.zeros(shape, dtype=dtype)
        self._w2 = np.zeros(shape, dtype=dtype)

    @staticmethod
    def _normalize_shape(shape):
        if isinstance(shape, int):
            shape = (shape,)
        try:
            shape = tuple(int(v) for v in shape)
        except (TypeError, ValueError):
            raise ValueError(f"support shape must be a sequence of ints, got {shape!r}")
        if len(shape) != 2:
            raise ValueError(f"support shape must be exactly 2-D (H, W), got {shape!r}")
        if any(v <= 0 for v in shape):
            raise ValueError(f"support shape dims must be positive, got {shape!r}")
        return shape

    @property
    def shape(self):
        """Spatial (H, W) shape of the support maps."""
        return self._shape

    @property
    def dtype(self):
        """Accumulator dtype."""
        return self._dtype

    @property
    def support_w1(self):
        """SUP_W1 (sum of per-exposure support ``s_i``), as an owned copy."""
        return self._w1.copy()

    @property
    def support_w2(self):
        """SUP_W2 (sum of per-exposure support squared ``s_i**2``), as a copy."""
        return self._w2.copy()

    def add(self, support):
        """Accumulate one original exposure's positive per-pixel support.

        Atomic: both SUP_W1 and SUP_W2 update together only after the support map
        has fully validated AND the cumulative sums have been preflighted against
        overflow; a failed call leaves both unchanged.
        """
        accumulate_support_pair(self._w1, self._w2, support, dtype=self._dtype)

    @property
    def n_eff_support(self):
        """Effective support ``N_eff = SUP_W1**2 / SUP_W2`` (owned copy).

        Computed only where ``SUP_W2 > 0``; ``0.0`` (documented neutral value)
        where undefined. Non-negative and finite everywhere. Never mutates state.

        Exact-first: naive ``W1**2 / W2`` where the square stays finite (the common
        case), with an overflow-resistant fallback ``(W1 / sqrt(W2))**2`` for
        pixels whose ``W1**2`` would overflow; algebraically equal so a finite W1
        and W2 never yield a spurious ``0.0`` from intermediate square overflow.
        """
        w1 = self._w1
        w2 = self._w2
        valid = w2 > 0.0
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            w1_sq = w1 * w1
            out = w1_sq / w2
            ratio = w1 / np.sqrt(w2)
            safe = ratio * ratio
        overflowed = ~np.isfinite(w1_sq)
        out = np.where(overflowed & valid, safe, out)
        out = np.where(valid & np.isfinite(out) & (out >= 0.0), out, N_EFF_UNDEFINED_VALUE)
        return out


# ---------------------------------------------------------------------------
# C. Estimator-weight builder (integration glue for C2)
# ---------------------------------------------------------------------------

def _resolve_taper(taper, n, h, w, mask, taper_px, taper_floor):
    """Resolve the ``taper`` argument into an owned ``(N, H, W)`` float64 in [0,1].

    ``None`` -> ``a_i = 1`` (no taper); ``"footprint"`` -> generate per-frame from
    ``mask`` via :func:`make_footprint_taper` (never a radial map); an explicit
    ``(N, H, W)`` float array; or a length-N sequence of per-frame ``(H, W)``
    arrays.
    """
    if taper is None:
        return np.ones((n, h, w), dtype=np.float64)

    if isinstance(taper, str):
        if taper.strip().lower() != "footprint":
            raise CanonicalStackValidationError(
                f"unknown taper {taper!r}; expected None, 'footprint', an (N,H,W) "
                "array, or a length-N sequence of (H,W) arrays"
            )
        a = np.empty((n, h, w), dtype=np.float64)
        for i in range(n):
            a[i] = make_footprint_taper(mask[i], feather_px=taper_px, floor=taper_floor)
        return a

    try:
        ta = np.asarray(taper)
    except Exception as exc:  # pragma: no cover - defensive
        raise CanonicalStackValidationError(f"taper is not array-convertible: {exc}")

    if ta.dtype == np.bool_ or ta.dtype.kind not in "iuf":
        raise CanonicalStackValidationError(
            f"taper must be real numeric, got dtype {ta.dtype}"
        )
    if ta.ndim == 3:
        if ta.shape != (n, h, w):
            raise CanonicalStackValidationError(
                f"taper shape {ta.shape} != (N,H,W) {(n, h, w)}"
            )
        a = np.array(ta, dtype=np.float64, copy=True)
    else:
        # length-N sequence of per-frame (H, W) tapers
        try:
            frames = list(taper)
        except TypeError:
            raise CanonicalStackValidationError(
                "taper must be 'footprint', an (N,H,W) array, or a length-N "
                "sequence of (H,W) arrays"
            )
        if len(frames) != n:
            raise CanonicalStackValidationError(
                f"taper sequence length {len(frames)} != N {n}"
            )
        a = np.empty((n, h, w), dtype=np.float64)
        for i in range(n):
            fr = np.asarray(frames[i])
            if fr.shape != (h, w):
                raise CanonicalStackValidationError(
                    f"taper frame {i} shape {fr.shape} != (H,W) {(h, w)}"
                )
            if fr.dtype == np.bool_ or fr.dtype.kind not in "iuf":
                raise CanonicalStackValidationError(
                    f"taper frame {i} dtype {fr.dtype} not real numeric"
                )
            a[i] = fr

    if not np.all(np.isfinite(a)) or np.any(a < 0.0) or np.any(a > 1.0):
        raise CanonicalStackValidationError("taper must be finite in [0, 1]")
    return a


def build_canonical_estimator_weights(
    q_weights,
    valid_mask,
    *,
    taper=None,
    taper_px=8.0,
    taper_floor=0.0,
    active_frames=None,
):
    """Build the explicit ``(N, H, W)`` float64 estimator-weight map ``w=q*m*a``.

    Produces the pre-rejection canonical estimator-weight map consumed by the C2
    ``combine_canonical_samples``: ``w_i = q_i * m_i * a_i`` where ``q_i`` is the
    B2 scalar quality weight, ``m_i`` the channel-invariant 2-D valid mask, and
    ``a_i`` the footprint taper (or ``1`` when no taper is supplied).

    Parameters
    ----------
    q_weights:
        B2 ``weighting.weights`` ``(N,)`` finite in ``[0, 1]`` (inactive ``0``,
        active ``> 0``).
    valid_mask:
        Channel-invariant ``(N, H, W)`` bool (``normalization.valid_mask``).
    taper:
        ``None`` -> ``a_i = 1`` inside the mask; ``"footprint"`` -> generate a
        per-frame taper from ``valid_mask`` via :func:`make_footprint_taper`
        (never a radial map); an explicit ``(N, H, W)`` float array in ``[0, 1]``;
        or a length-N sequence of per-frame ``(H, W)`` tapers.
    taper_px / taper_floor:
        Forwarded to :func:`make_footprint_taper` in the ``"footprint"`` mode.
    active_frames:
        Optional ``(N,)`` bool. When given, validates consistency with
        ``q_weights`` (active ``> 0``, inactive ``== 0``).

    Returns
    -------
    numpy.ndarray
        Owned contiguous float64 ``(N, H, W)``; exactly ``0`` outside
        ``valid_mask``, exactly ``0`` for inactive frames, bounded ``[0, 1]`` with
        ``w_i <= q_i``.
    """
    q = np.asarray(q_weights)
    if q.dtype == np.bool_ or q.dtype.kind not in "iuf":
        raise CanonicalStackValidationError(
            f"q_weights must be real numeric, got dtype {q.dtype}"
        )
    if q.ndim != 1:
        raise CanonicalStackValidationError(
            f"q_weights must be 1-D (N,), got shape {q.shape}"
        )
    n = int(q.shape[0])
    qf = np.array(q, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(qf)) or np.any(qf < 0.0) or np.any(qf > 1.0):
        raise CanonicalStackValidationError("q_weights must be finite in [0, 1]")

    m = np.asarray(valid_mask)
    if m.dtype != np.bool_:
        raise CanonicalStackValidationError(f"valid_mask must be bool, got {m.dtype}")
    if m.ndim != 3:
        raise CanonicalStackValidationError(
            f"valid_mask must be 3-D (N, H, W), got shape {m.shape}"
        )
    if m.shape[0] != n:
        raise CanonicalStackValidationError(
            f"valid_mask N {m.shape[0]} != q_weights N {n}"
        )
    h, w = int(m.shape[1]), int(m.shape[2])

    if active_frames is not None:
        act = np.asarray(active_frames)
        if act.dtype != np.bool_ or act.shape != (n,):
            raise CanonicalStackValidationError(
                f"active_frames must be (N,) bool, got shape {act.shape} dtype {act.dtype}"
            )
        if np.any(qf[act] <= 0.0):
            raise CanonicalStackValidationError("active frames must have q_weights > 0")
        if np.any(qf[~act] != 0.0):
            raise CanonicalStackValidationError("inactive frames must have q_weights == 0")

    a = _resolve_taper(taper, n, h, w, m, taper_px, taper_floor)

    w = qf[:, None, None] * m.astype(np.float64) * a
    w = np.where(m, w, 0.0)  # exact zero outside valid_mask (defensive)
    return np.ascontiguousarray(w)
