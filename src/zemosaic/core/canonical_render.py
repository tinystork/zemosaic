"""SCI-05 Gate E3 — donor-exact coverage-aware render (preview-only).

Source-port of the donor's ``seestar/enhancement/coverage_render.py``
(``coverage_aware_render``) into a **pure-CPU, preview-only** canonical module,
plus a small preview helper and a bounded render-event dict. This is a cosmetic
**display/preview** transform — **never** science: it must never mutate the
scientific result/support, and is never applied inside ``run_canonical_stack``
(the engine is untouched).

Formula (donor-exact)
---------------------
``SCI = B + D`` (low-frequency background + high-frequency detail); with
``alpha = clip(1 - N_eff_support / n_ref, 0, 1)``:

    RENDER = B + (1 - alpha) * D + alpha * D_denoised

High-support regions (``N_eff_support >= n_ref`` → ``alpha = 0``) are untouched;
low-support regions get progressively stronger noise regularization of the
**detail residual only**.

Honest constraints (non-negotiable, preserved):
* no brightness gain for low coverage (flat fields stay exactly flat);
* no generative inpainting / invented stars or nebulosity;
* no low-frequency signal modification driven by support;
* only the high-frequency detail residual is attenuated in low-support areas.
"""

from __future__ import annotations

import warnings

import numpy as np

__all__ = [
    "coverage_aware_render",
    "render_preview",
    "coverage_render_event",
]

# scipy gaussian_filter primary path (module-level reference so the
# scipy-unavailable fallback can be forced by monkeypatching, mirroring the
# donor's lazy import).
try:  # pragma: no cover - scipy is a declared dependency
    from scipy.ndimage import gaussian_filter as _gaussian_filter
except Exception:  # pragma: no cover - defensive
    _gaussian_filter = None


def coverage_aware_render(
    sci,
    neff_support,
    *,
    n_ref=32.0,
    sigma_denoise=2.0,
    sigma_low=32.0,
):
    """Blend a denoised detail residual in low-support regions (preview-only).

    Returns a NEW float32 render array; never mutates ``sci`` or
    ``neff_support``. ``sci`` is ``(H, W)`` mono or ``(H, W, C)`` (HWC/RGB);
    ``neff_support`` must be 2-D ``(H, W)`` (the E1 ``n_eff_support`` map).

    ``n_ref`` must be finite ``> 0``; no positive support information (empty /
    all-zero / all-NaN / non-finite ``nanmax``) returns ``sci`` unchanged; scipy
    ``gaussian_filter`` unavailable returns ``sci`` unchanged (donor behavior).
    """
    sci = np.asarray(sci, dtype=np.float32)
    sup = np.asarray(neff_support, dtype=np.float32)
    if sup.ndim != 2:
        raise ValueError("neff_support must be 2-D")
    if sci.shape[:2] != sup.shape[:2]:
        raise ValueError("sci and neff_support spatial shapes must match")

    n_ref = float(n_ref)
    if not np.isfinite(n_ref) or n_ref <= 0.0:
        raise ValueError("n_ref must be finite > 0")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sup_max = float(np.nanmax(sup)) if sup.size else 0.0
    if not np.isfinite(sup_max) or sup_max <= 0.0:
        # no support information -> do not regularize
        return sci.astype(np.float32)

    alpha = np.clip(1.0 - sup / n_ref, 0.0, 1.0).astype(np.float32)

    if _gaussian_filter is None:
        return sci.astype(np.float32)
    gaussian_filter = _gaussian_filter

    out = np.empty_like(sci)
    if sci.ndim == 2:
        B = gaussian_filter(sci, sigma=sigma_low)
        D = sci - B
        Dd = gaussian_filter(D, sigma=sigma_denoise)
        out = B + (1.0 - alpha) * D + alpha * Dd
    else:
        C = sci.shape[2]
        for c in range(C):
            ch = sci[..., c]
            B = gaussian_filter(ch, sigma=sigma_low)
            D = ch - B
            Dd = gaussian_filter(D, sigma=sigma_denoise)
            out[..., c] = B + (1.0 - alpha) * D + alpha * Dd
    return out.astype(np.float32)


def render_preview(result, *, n_ref=32.0, sigma_denoise=2.0, sigma_low=32.0):
    """Return a NEW preview-render array from a result (or ``(sci, n_eff_support)``).

    Accepts a ``CanonicalStackResult`` (uses ``.science`` / ``.n_eff_support``)
    or a 2-tuple ``(sci, n_eff_support)``. Pure: never mutates the result —
    ``result.science`` / ``support_w1`` / ``support_w2`` / ``n_eff_support`` stay
    bit-identical.
    """
    if hasattr(result, "science") and hasattr(result, "n_eff_support"):
        sci = result.science
        neff = result.n_eff_support
    else:
        sci, neff = result
    return coverage_aware_render(
        sci, neff, n_ref=n_ref, sigma_denoise=sigma_denoise, sigma_low=sigma_low
    )


def coverage_render_event(*, n_ref=32.0, sigma_denoise=2.0, sigma_low=32.0):
    """Bounded render-event dict (scalars/strings only, JSON-serializable)."""
    return {
        "render": "coverage_aware_render",
        "n_ref": float(n_ref),
        "sigma_denoise": float(sigma_denoise),
        "sigma_low": float(sigma_low),
    }
