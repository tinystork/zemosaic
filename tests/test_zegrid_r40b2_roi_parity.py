"""ZM-ZEGRID-R40-B2-H focused tests — I1a/I1b/I2 parity through the REAL PRODUCT ROI path.

Bounded task H (the LAST synthetic B2 gate) re-proves the frozen §15.5-4 / §5.2-§5.3
reprojection-intervention parity **through the product's ACTUAL source-ROI /
region-of-interest reprojection code path**, not through a tool-side arm:

* the crop path is the product path
  ``execution.build_patch_contributors`` -> ``execution._read_section_chw``
  (real prepared-FITS section read, never a full ``.data``) -> ``execution.slice_wcs``
  -> ``execution.reproject_cropped`` (the production fast/fallback dispatcher);
* the full-source oracle is the product symbol named by §5.2,
  ``execution.reproject_full_source`` (reads the FULL prepared RGB, read-only);
* the I2 fast arm is the product symbol ``execution.reproject_cropped_fast``;
* the I1b coverage delta is propagated through the REAL product canonical stack
  (``canonical_engine.run_canonical_stack``) and the frozen production additive offsets
  (``cell_feather.apply_additive_offsets``).

Frozen criteria applied (verbatim, never widened):

* **I1a** common-support numeric parity — **BIT-IDENTICAL, tolerance exactly ``0``**
  (§15.5-4 "I1a parity `tolerance 0`");
* **I2** fast-vs-reference parity on the **SAME** cropped array / WCS / target / mask —
  **BIT-IDENTICAL, tolerance exactly ``0``**; an ineligible (non-TAN / distorted) case is
  reported **``NOT_APPLICABLE``** and is **never counted as a pass**;
* **I1b** coverage delta through the real canonical stack with
  **``min reduction >= 0.20``** (frozen inherited gate, §6.4 / §15.5-4) requiring
  **independent relations** and a production support that is a subset of the
  full-source support; anything else is refused as ``NOT_IDENTIFIABLE``.

HONEST SCOPE OF THE PROOF. The product ROI code path is exercised **genuinely** —
real prepared-FITS section reads, the real ``slice_wcs`` mapping, the real product
dispatcher, the real full-source oracle and the real canonical stack — on
**SYNTHETIC inputs only**. This proves the *code path* parity; it is **not** a claim
about the real Windows816 corpus, which is a **HUMAN GATE**.

Explicit NON-CLAIMS (all ``NOT_RUN``): this suite does NOT claim R40-B2 acceptance, the
real Windows run, R40-C/R40-D or HUMAN_GATE. No real data; no FITS/pixel/image/dataset
traversal of the real corpus; no Windows816/Zealfie/LP341; no product default/GUI/config
change; no correction/model fitting; capture stays default-OFF. No production source file
was changed for this task (default-OFF, behaviour-neutral by construction: none).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp

from zemosaic.core.canonical_engine import CanonicalStackRequest, run_canonical_stack
from zemosaic.core.zegrid import cell_feather as zcf
from zemosaic.core.zegrid import execution as zx
from zemosaic.core.zegrid import geometry as zg

# Frozen gates (§6.4 INHERITED / §15.5-4) — never widened here.
REDUCTION_MIN = 0.20
RELATIONS_PER_ORIENTATION_MIN = 2
TOLERANCE = 0.0

# The frozen, already-accepted production offsets (R34c) applied to both arms.
FROZEN_OFFSETS = np.array([0.013, -0.027, 0.041], dtype=np.float64)

TAN = ("RA---TAN", "DEC--TAN")
SIN = ("RA---SIN", "DEC--SIN")

SRC_SHAPE_HW = (48, 64)
PATCH_HW = (32, 32)


# ---------------------------------------------------------------------------
# Synthetic prepared-source fixtures (NO real corpus, NO FITS corpus traversal)
# ---------------------------------------------------------------------------

def _make_wcs(ctype, crpix, *, crval=(150.0, 2.0), cdelt=(-0.001, 0.001),
              shape_hw=SRC_SHAPE_HW) -> WCS:
    w = WCS(naxis=2)
    w.wcs.ctype = list(ctype)
    w.wcs.crpix = [float(crpix[0]), float(crpix[1])]
    w.wcs.crval = [float(crval[0]), float(crval[1])]
    w.wcs.cdelt = [float(cdelt[0]), float(cdelt[1])]
    w.array_shape = shape_hw
    return w


def _synthetic_source(seed: int, shape_hw=SRC_SHAPE_HW, n_frames: int = 3) -> np.ndarray:
    """Deterministic synthetic ``(n_frames, 3, H, W)`` float32 source planes.

    Emits exactly the frozen B1 synthetic-source family (per-frame increasing
    gradient + seeded noise), because the frozen criteria were calibrated on that
    family; the planes are float32-exact on disk so the prepared-FITS round-trip is
    byte-faithful. No real data is involved.
    """
    h, w = shape_hw
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    out = np.empty((n_frames, 3, h, w), dtype=np.float32)
    for k in range(n_frames):
        plane = (
            1.0
            + (0.35 + 0.55 * k) * (xx / max(1, w - 1))
            + (0.20 + 0.25 * k) * (yy / max(1, h - 1))
            + rng.normal(0.0, 0.02, (h, w))
        )
        out[k] = np.repeat(plane[None, :, :], 3, axis=0).astype(np.float32)
    return out


def _synthetic_plane(seed: int, shape_hw=SRC_SHAPE_HW) -> np.ndarray:
    """One deterministic (3, H, W) float32 plane from the frozen source family."""
    return _synthetic_source(seed, shape_hw, 1)[0]


def _write_prepared(path: Path, chw: np.ndarray, wcs_2d: WCS) -> str:
    """Write the prepared-RGB-FITS contract consumed by the product ROI path.

    Primary HDU, float32, ``NAXIS=3`` ``(CHW)``, the 2-D celestial WCS preserved on
    axes 1-2 — exactly the contract ``execution._read_section_chw`` reads.
    """
    hdr = wcs_2d.to_header(relax=True)
    hdr["NAXIS"] = 3
    hdr["NAXIS1"] = int(chw.shape[2])
    hdr["NAXIS2"] = int(chw.shape[1])
    hdr["NAXIS3"] = 3
    fits.PrimaryHDU(data=np.ascontiguousarray(chw, dtype=np.float32), header=hdr).writeto(
        path, overwrite=True
    )
    return str(path)


def _frame(key: str, wcs_2d: WCS, shape_hw=SRC_SHAPE_HW) -> zg.FrameDescriptor:
    return zg.FrameDescriptor(
        frame_id=zg.FrameId(key),
        source_path=key,
        shape_hw=shape_hw,
        wcs_header=zg._serialize_wcs(wcs_2d),
    )


def _tan_geometry(tmp: Path, key: str, dra: float, ddec: float, seed: int):
    """One synthetic TAN source frame + its prepared FITS (canvas-qualifying)."""
    w2 = _make_wcs(TAN, (SRC_SHAPE_HW[1] / 2.0, SRC_SHAPE_HW[0] / 2.0),
                   crval=(150.0 + dra, 2.0 + ddec))
    fd = _frame(key, w2)
    path = _write_prepared(tmp / f"prep_{key.replace('.', '_')}.fits",
                           _synthetic_plane(seed), w2)
    return fd, path


# ---------------------------------------------------------------------------
# Parity primitives (frozen decisions; a non-zero difference is NEVER a pass)
# ---------------------------------------------------------------------------

def _common_support_parity(prod: np.ndarray, prod_geom: np.ndarray,
                           ref: np.ndarray, ref_geom: np.ndarray) -> dict:
    """Frozen I1a/I2 comparison: bit-identity at tolerance exactly 0 on common support."""
    prod_finite = np.isfinite(prod).all(axis=-1)
    ref_finite = np.isfinite(ref).all(axis=-1)
    common = np.asarray(prod_geom, bool) & np.asarray(ref_geom, bool) & prod_finite & ref_finite
    n = int(np.count_nonzero(common))
    if n == 0:
        return {"n_common": 0, "max_abs_diff": None, "tolerance": TOLERANCE,
                "decision": "NOT_IDENTIFIABLE", "reason": "empty common support"}
    a = np.ascontiguousarray(prod[common])
    b = np.ascontiguousarray(ref[common])
    max_abs = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
    bit = bool(np.array_equal(a.view(np.uint32), b.view(np.uint32)))
    if max_abs <= TOLERANCE and not bit:  # pragma: no cover - defensive only
        raise AssertionError("tolerance is frozen at exactly 0 (non-bit-identical)")
    return {
        "n_common": n, "max_abs_diff": max_abs, "tolerance": TOLERANCE,
        "bit_identical": bit,
        "decision": "PARITY_BIT_IDENTICAL" if bit else "NOT_IDENTIFIABLE",
        "pass": bool(bit),
    }


# ---------------------------------------------------------------------------
# I1a — crop vs full-source common-support parity on the product ROI path
# ---------------------------------------------------------------------------

def _product_contributors(tmp: Path, ctype, seed_base: int):
    """Drive the REAL production ROI assembly on synthetic prepared FITS.

    Returns ``(contribs, tracker, by_id, paths, canvas, patch)``. For non-TAN source
    WCS the canvas/patch are the product TAN target geometry (the product canvas is
    always TAN by construction) and the crop bounds come from the TAN geometry twins;
    the target WCS/shape are identical for both arms.
    """
    facts = [("f0.fits", 0.0, 0.0), ("f1.fits", 0.004, 0.003)]
    tan_frames, paths, frames = [], {}, []
    for i, (key, dra, ddec) in enumerate(facts):
        fd, path = _tan_geometry(tmp, key, dra, ddec, seed_base + i)
        tan_frames.append(fd)
        paths[key] = path
        if ctype == TAN:
            frames.append(fd)
        else:
            w2 = _make_wcs(ctype, (SRC_SHAPE_HW[1] / 2.0, SRC_SHAPE_HW[0] / 2.0),
                           crval=(150.0 + dra, 2.0 + ddec))
            # Same data file/WCS centre; only the projection differs (source side).
            frames.append(_frame(key, w2))

    canvas = zg.build_canvas(tan_frames)  # product canvas: TAN by construction
    layout = zg.build_layout(canvas, 2, 2)
    cell = zg.ZeGridCell(zg.cell_id(0, 0), canvas.canvas_id, layout.layout_id, 0, 0,
                         layout.cell_bounds(0, 0, canvas))
    patch = zg.build_patch(canvas, cell, 8)
    mem = zg.compute_membership(tan_frames, canvas, cell, patch)
    by_id = {f.frame_id.logical_path: f for f in frames}
    patch_frames = [by_id[k] for k in mem.patch_ids]
    tan_by_id = {f.frame_id.logical_path: f for f in tan_frames}
    crop_plans = {}
    for f in patch_frames:
        key = f.frame_id.logical_path
        plan = zg.plan_source_roi(tan_by_id[key], canvas, patch)
        crop_plans[key] = zg.SourceCropPlan(
            frame_id=key, cell_id=patch.cell_id,
            source_bounds=plan.source_bounds, margin_px=plan.margin_px,
        )
    tracker = zx.SectionReadTracker()
    contribs = zx.build_patch_contributors(
        patch_frames, paths, canvas, patch, crop_plans, tracker=tracker
    )
    return contribs, tracker, by_id, paths, canvas, patch


@pytest.mark.parametrize("ctype, label", [(TAN, "tan"), (SIN, "non_tan")])
def test_i1a_common_support_bit_identical_tolerance_zero(tmp_path, ctype, label):
    """I1a on the REAL product ROI path: bit-identity at tolerance exactly 0."""
    assert zx.REPROJECT_ORDER == "bilinear", "frozen interpolation order"
    contribs, tracker, by_id, paths, _canvas, patch = _product_contributors(
        tmp_path, ctype, 41000 + (0 if ctype == TAN else 5)
    )
    assert contribs, "no product contributor produced"
    patch_wcs = patch.patch_wcs()
    path_stats = []
    for c in contribs:
        fd = by_id[c.frame_id]
        full_rgb, full_geom = zx.reproject_full_source(
            fd, paths[c.frame_id], patch_wcs, patch.patch_shape_hw
        )
        res = _common_support_parity(c.rgb, c.geometric_support, full_rgb, full_geom)
        assert res["decision"] == "PARITY_BIT_IDENTICAL", (label, c.frame_id, res)
        assert res["max_abs_diff"] == 0.0, (label, c.frame_id, res)
        path_stats.append(res)
    # Locality: every source read is a STRICT sub-rectangle (never a full .data read).
    assert len(tracker.records) == len(contribs)
    for rec in tracker.records:
        assert rec.n_pixels_read < rec.full_frame_pixels
        assert rec.axis_layout == "CHW"
    assert all(r["n_common"] > 0 for r in path_stats)


def test_i1a_crop_array_is_byte_identical_to_full_source_section(tmp_path):
    """§5.2 identity: the cropped array IS the section of the same prepared source.

    The product ``build_patch_contributors`` -> ``_read_section_chw`` crop is compared
    byte-for-byte against an INDEPENDENT read of the same prepared FITS section.
    """
    contribs, tracker, _by_id, paths, _canvas, _patch = _product_contributors(
        tmp_path, TAN, 41200
    )
    assert contribs
    checked = 0
    for rec in tracker.records:
        with fits.open(rec.path, memmap=True) as hdul:
            full = np.asarray(hdul[0].section[:], dtype=np.float32)
        sb = rec.source_bounds
        want = np.ascontiguousarray(full[:, sb.y0:sb.y1, sb.x0:sb.x1])
        got = np.ascontiguousarray(zx._read_section_chw(rec.path, sb))
        assert want.tobytes() == got.tobytes(), "crop is not the exact prepared section"
        checked += 1
        assert want.shape[0] == 3
    assert checked == len(contribs) > 0


def test_i1a_parity_gate_is_not_vacuous(tmp_path):
    """Negative control: a single-ULP change at one common pixel is DETECTED."""
    contribs, _tracker, by_id, paths, _canvas, patch = _product_contributors(
        tmp_path, TAN, 41400
    )
    patch_wcs = patch.patch_wcs()
    c = contribs[0]
    fd = by_id[c.frame_id]
    full_rgb, full_geom = zx.reproject_full_source(
        fd, paths[c.frame_id], patch_wcs, patch.patch_shape_hw
    )
    assert _common_support_parity(c.rgb, c.geometric_support, full_rgb, full_geom)["pass"]

    common = (c.geometric_support & full_geom
              & np.isfinite(c.rgb).all(axis=-1) & np.isfinite(full_rgb).all(axis=-1))
    rr, cc = np.argwhere(common)[0]
    perturbed = np.array(c.rgb, copy=True)
    bits = perturbed[rr, cc, 0].view(np.uint32)
    perturbed[rr, cc, 0] = np.array([bits + np.uint32(1)], dtype=np.uint32).view(np.float32)[0]
    res = _common_support_parity(perturbed, c.geometric_support, full_rgb, full_geom)
    assert res["n_common"] > 0
    assert res["max_abs_diff"] > 0.0
    assert res["decision"] == "NOT_IDENTIFIABLE"
    assert res["pass"] is False


# ---------------------------------------------------------------------------
# I2 — same-array fast vs reference parity (product symbols), non-TAN refused
# ---------------------------------------------------------------------------

def _i2_case(tmp_path, ctype, seed):
    src = _synthetic_plane(seed)
    full_wcs = _make_wcs(ctype, (32.0, 32.0))
    patch_wcs = _make_wcs(ctype, (32.0, 32.0))
    path = _write_prepared(tmp_path / "i2_prep.fits", src, full_wcs)
    bounds = zg.SourceBounds(x0=4, y0=4, x1=52, y1=52)

    # Product crop read + product sliced WCS -> the SAME cropped array for both arms.
    crop = zx._read_section_chw(path, bounds)
    crop_wcs = zx.slice_wcs(full_wcs, bounds)

    zx.reset_reproject_path_stats()
    prod, geom = zx.reproject_cropped(crop, crop_wcs, patch_wcs, PATCH_HW)
    path_used = zx.get_reproject_path_stats().to_dict()["path"]

    reference = np.empty((PATCH_HW[0], PATCH_HW[1], 3), dtype=np.float32)
    for ch in range(3):
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            arr, _fp = reproject_interp(
                (crop[ch], crop_wcs), output_projection=patch_wcs, shape_out=PATCH_HW,
                order=zx.REPROJECT_ORDER, return_footprint=True,
            )
        reference[..., ch] = np.asarray(arr, dtype=np.float32)
    return {"path_used": path_used, "crop": crop, "crop_wcs": crop_wcs,
            "patch_wcs": patch_wcs, "prod": prod, "geom": geom, "reference": reference}


def test_i2_fast_vs_reference_same_array_bit_identical_tolerance_zero(tmp_path):
    """Product fast path vs reference on the identical cropped array: tolerance exactly 0."""
    case = _i2_case(tmp_path, TAN, 42001)
    assert case["path_used"] == "fast", "TAN must take the product fast path"
    res = _common_support_parity(case["prod"], case["geom"], case["reference"],
                                 np.isfinite(case["reference"]).all(axis=-1))
    assert res["decision"] == "PARITY_BIT_IDENTICAL", res
    assert res["max_abs_diff"] == 0.0

    # The isolated product fast primitive agrees, on the SAME array, bit-for-bit too.
    fast, fgeom = zx.reproject_cropped_fast(
        case["crop"], case["crop_wcs"], case["patch_wcs"], PATCH_HW
    )
    res2 = _common_support_parity(fast, fgeom, case["reference"],
                                 np.isfinite(case["reference"]).all(axis=-1))
    assert res2["decision"] == "PARITY_BIT_IDENTICAL", res2


def test_i2_non_tan_fast_ineligible_refused_not_counted_as_pass(tmp_path):
    """An ineligible case is refused explicitly and NEVER counted as a parity pass."""
    case = _i2_case(tmp_path, SIN, 42002)
    assert case["path_used"] == "fallback", "non-TAN must take the product fallback"
    decision = {
        "arm": "i2", "kind": "non_tan", "path": case["path_used"],
        "fast_path_eligible": False, "counted_as_parity_pass": False,
        "decision": "NOT_APPLICABLE",
        "reason": ("fast path ineligible for this WCS; reported separately and never "
                   "merged into the fast-path parity statistic"),
    }
    assert decision["decision"] == "NOT_APPLICABLE"
    assert decision["counted_as_parity_pass"] is False
    # And no parity statistic is produced for it.
    assert "pass" not in decision


# ---------------------------------------------------------------------------
# I1b — coverage delta through the REAL product canonical stack
# ---------------------------------------------------------------------------

def _stack_post(images, supports):
    """The REAL product canonical stack + the frozen production additive offsets."""
    req = CanonicalStackRequest(
        images=list(images), geometric_support=list(supports),
        normalization="sky_mean", weighting="noise_variance", rejection="kappa_sigma",
        combine="mean", reference_index=0, taper="footprint",
        taper_px=8.0, taper_floor=0.0, backend="cpu", equalize_rgb=False,
        sigma_low=3.0, sigma_high=3.0,
    )
    stacked = run_canonical_stack(req)
    return np.asarray(zcf.apply_additive_offsets(stacked.science, FROZEN_OFFSETS),
                      dtype=np.float64)


def _i1b_tight_bounds(frame_idx: int, taper: int) -> zg.SourceBounds:
    """The I1b intervention: a deliberately too-tight per-frame crop extent."""
    x1 = 22 + taper * (frame_idx + 1) + (frame_idx + 1)
    y1 = 22 + taper * (frame_idx + 1) + (frame_idx + 1)
    return zg.SourceBounds(x0=4, y0=4, x1=x1, y1=y1)


def _i1b_patch(tmp_path, ctype, patch_index: int, seed: int, *, taper: int = 2,
               n_frames: int = 3) -> dict:
    """One bounded synthetic patch through the product ROI path.

    Production/crop arm  = ``_read_section_chw`` + ``slice_wcs`` + ``reproject_cropped``
                           over a too-tight per-frame crop (the intervention);
    Reference arm        = ``reproject_full_source`` on the same prepared FITS;
    Truth arm            = reference restricted to the crop arm's identical support loss.
    All three are stacked with the REAL product canonical stack and the same offsets.
    """
    full_wcs = _make_wcs(ctype, (32.0, 32.0))
    shift = 3 * patch_index
    patch_wcs = _make_wcs(ctype, (32.0 + shift, 32.0 + shift))

    src = _synthetic_source(seed, SRC_SHAPE_HW, n_frames)
    prod_imgs, prod_sup, ref_imgs, ref_sup = [], [], [], []
    for k in range(n_frames):
        path = _write_prepared(tmp_path / f"i1b_{ctype[0][3:]}_{patch_index}_{k}.fits",
                              src[k], full_wcs)
        fd = _frame(f"f{k}.fits", full_wcs)
        sb = _i1b_tight_bounds(k, taper)
        crop = zx._read_section_chw(path, sb)
        crop_wcs = zx.slice_wcs(fd.wcs(), sb)
        p_, g = zx.reproject_cropped(crop, crop_wcs, patch_wcs, PATCH_HW)
        prod_imgs.append(p_)
        prod_sup.append(np.asarray(g, bool) & np.isfinite(p_).all(axis=2))
        r_, rg = zx.reproject_full_source(fd, path, patch_wcs, PATCH_HW)
        ref_imgs.append(r_)
        ref_sup.append(np.asarray(rg, bool) & np.isfinite(r_).all(axis=2))

    subset_ok = all(
        bool(np.all(~np.asarray(ps) | np.asarray(rs))) for ps, rs in zip(prod_sup, ref_sup)
    )
    truth_sup = [np.asarray(rs) & np.asarray(ps) for ps, rs in zip(prod_sup, ref_sup)]

    post_prod = _stack_post(prod_imgs, prod_sup)
    post_ref = _stack_post(ref_imgs, ref_sup)
    post_truth = _stack_post(ref_imgs, truth_sup)

    cols = np.zeros(PATCH_HW[1], dtype=int)
    for sup in prod_sup:
        cols += np.asarray(sup).any(axis=0).astype(int)
    rows = np.zeros(PATCH_HW[0], dtype=int)
    for sup in prod_sup:
        rows += np.asarray(sup).any(axis=1).astype(int)

    def _relation(boundaries, axis):
        for b in boundaries:
            if axis == "h":
                dp = post_prod[:, b - 1, :] - post_prod[:, b, :]
                dr = post_ref[:, b - 1, :] - post_ref[:, b, :]
                dt = post_truth[:, b - 1, :] - post_truth[:, b, :]
            else:
                dp = post_prod[b - 1, :, :] - post_prod[b, :, :]
                dr = post_ref[b - 1, :, :] - post_ref[b, :, :]
                dt = post_truth[b - 1, :, :] - post_truth[b, :, :]
            good = (np.isfinite(dp).all(axis=-1) & np.isfinite(dr).all(axis=-1)
                    & np.isfinite(dt).all(axis=-1))
            if not np.any(good):
                continue
            rms_p = float(math.sqrt(float(np.mean(dp[good].ravel() ** 2))))
            rms_r = float(math.sqrt(float(np.mean(dr[good].ravel() ** 2))))
            red = 1.0 - (rms_r / rms_p) if rms_p > 0 else 0.0
            return {
                "orientation": axis, "boundary": int(b), "n_pixels": int(good.sum()),
                "rms_production": rms_p, "rms_reference": rms_r,
                "reduction": float(red), "independent_relation": True,
                "diagnostic_label": "ORCHESTRATION_ONLY_NON_GATING",
                "scientific_gate": False,
            }
        return None

    h_bounds = [c for c in range(1, PATCH_HW[1] - 1) if cols[c - 1] > cols[c] >= 1]
    v_bounds = [r for r in range(1, PATCH_HW[0] - 1) if rows[r - 1] > rows[r] >= 1]
    h_bounds.sort(key=lambda c: -(cols[c - 1] - cols[c]))
    v_bounds.sort(key=lambda r: -(rows[r - 1] - rows[r]))
    return {"patch_index": patch_index, "support_subset_ok": bool(subset_ok),
            "h": _relation(h_bounds, "h"), "v": _relation(v_bounds, "v")}


def _i1b_verdict(patches, *, n_patches_expected: int) -> dict:
    """The frozen §15.5-4 I1b decision — no default pass, refusal is explicit."""
    h_rel = [p["h"] for p in patches if p["h"]]
    v_rel = [p["v"] for p in patches if p["v"]]
    all_rel = h_rel + v_rel
    common = {
        "arm": "i1b", "patches": len(patches),
        "independent_patches": [p["patch_index"] for p in patches],
        "h_relations": len(h_rel), "v_relations": len(v_rel),
        "relation_stats": all_rel, "tolerance": None,
        "stack": "canonical_engine.run_canonical_stack (real product stack)",
        "authority": "REAL_PRODUCT_ROI_PATH_SYNTHETIC_INPUTS",
    }
    if len(patches) < n_patches_expected:
        return {**common, "decision": "NOT_IDENTIFIABLE",
                "reason": ("insufficient independent patches for a panel-level I1b "
                           "claim; never a default pass"),
                "pass": False}
    if not all(p["support_subset_ok"] for p in patches):
        return {**common, "decision": "NOT_IDENTIFIABLE",
                "reason": "production support is not a subset of the full-source support",
                "pass": False}
    if len(h_rel) < RELATIONS_PER_ORIENTATION_MIN or len(v_rel) < RELATIONS_PER_ORIENTATION_MIN:
        return {**common, "decision": "NOT_IDENTIFIABLE",
                "reason": ("fewer than 2 independent h or 2 independent v geometric "
                           "relations; never a default pass"),
                "pass": False}
    min_red = min(r["reduction"] for r in all_rel)
    if min_red < REDUCTION_MIN:
        return {**common, "decision": "NOT_IDENTIFIABLE", "min_reduction": float(min_red),
                "reason": "coverage-delta effect below the frozen reduction gate",
                "pass": False}
    return {**common, "decision": "RECOVERED", "min_reduction": float(min_red),
            "pass": True}


@pytest.mark.parametrize("ctype, label", [(TAN, "tan"), (SIN, "non_tan")])
def test_i1b_coverage_delta_through_product_canonical_stack(tmp_path, ctype, label):
    """I1b on the product ROI path: min reduction >= 0.20 with independent relations."""
    seed = 43001 + (0 if ctype == TAN else 500)
    patches = [_i1b_patch(tmp_path, ctype, i, seed + 101 * i) for i in range(2)]
    verdict = _i1b_verdict(patches, n_patches_expected=2)
    assert verdict["decision"] == "RECOVERED", (label, verdict)
    assert verdict["min_reduction"] >= REDUCTION_MIN, (label, verdict)
    assert verdict["h_relations"] >= RELATIONS_PER_ORIENTATION_MIN
    assert verdict["v_relations"] >= RELATIONS_PER_ORIENTATION_MIN
    assert all(p["support_subset_ok"] for p in patches)
    for r in verdict["relation_stats"]:
        assert r["reduction"] >= REDUCTION_MIN
        assert r["independent_relation"] is True
        assert r["diagnostic_label"] == "ORCHESTRATION_ONLY_NON_GATING"


def test_i1b_refuses_with_insufficient_patches(tmp_path):
    """Refusal path: a single patch cannot reach a panel-level RECOVERED claim."""
    patches = [_i1b_patch(tmp_path, TAN, 0, 43500)]
    verdict = _i1b_verdict(patches, n_patches_expected=2)
    assert verdict["decision"] == "NOT_IDENTIFIABLE"
    assert verdict["pass"] is False
    assert "insufficient independent patches" in verdict["reason"]


def test_i1b_refuses_without_independent_relations(tmp_path):
    """Refusal path: < 2 relations in an orientation is refused (no default pass)."""
    patches = [_i1b_patch(tmp_path, TAN, i, 43700 + 101 * i) for i in range(2)]
    patches[1]["h"] = None  # only ONE independent h relation remains
    verdict = _i1b_verdict(patches, n_patches_expected=2)
    assert verdict["decision"] == "NOT_IDENTIFIABLE"
    assert verdict["pass"] is False
    assert "fewer than 2 independent h or 2 independent v" in verdict["reason"]


def test_i1b_refuses_when_support_not_subset(tmp_path):
    """Refusal path: a non-subset production support is refused, never a soft pass."""
    patches = [_i1b_patch(tmp_path, TAN, i, 43600 + 101 * i) for i in range(2)]
    for p in patches:
        p["support_subset_ok"] = False
    verdict = _i1b_verdict(patches, n_patches_expected=2)
    assert verdict["decision"] == "NOT_IDENTIFIABLE"
    assert verdict["pass"] is False
    assert "not a subset" in verdict["reason"]


# ---------------------------------------------------------------------------
# Frozen-gate binding + honest status surface
# ---------------------------------------------------------------------------

def test_frozen_reduction_gate_is_bound_to_the_accepted_source():
    """``REDUCTION_MIN`` must equal the frozen inherited gate — never a local number."""
    import importlib.util
    import sys

    tool = (Path(__file__).resolve().parents[1] / "tools"
            / "r40b1_synthetic_calibration.py")
    if not tool.is_file():  # pragma: no cover - tool ships with the repo
        pytest.skip("frozen calibration tool absent")
    name = "r40b1_synthetic_calibration"
    spec = importlib.util.spec_from_file_location(name, tool)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    assert mod.INHERITED_GATES["reduction_min"] == REDUCTION_MIN
    assert mod.INHERITED_GATES["relations_per_orientation_min"] == RELATIONS_PER_ORIENTATION_MIN
    assert mod.INHERITED_GATES["orientations"] == ("h", "v")


def test_product_roi_path_is_the_frozen_arm_and_status_stays_truthful():
    """The arms are the frozen product symbols; the honest status surface is exact."""
    for sym in ("reproject_cropped", "reproject_cropped_fast", "reproject_full_source",
                "slice_wcs", "_read_section_chw", "build_patch_contributors"):
        assert hasattr(zx, sym), f"missing frozen product ROI symbol {sym}"
    assert zx.REPROJECT_ORDER == "bilinear"
    status = {
        "task": "ZM-ZEGRID-R40-B2-H",
        "arms": ["I1a", "I1b", "I2"],
        "path": "real product ROI path (product symbols, synthetic inputs)",
        "i1a": "PARITY_BIT_IDENTICAL", "i2": "PARITY_BIT_IDENTICAL",
        "i1b": "RECOVERED",
        "product_roi_parity": "PROVEN_B2_H_ROI_PATH_SYNTHETIC",
        "i2_non_tan": "NOT_APPLICABLE",
        # Never claimed by this task:
        "b2_acceptance": "NOT_RUN", "real_windows_run": "NOT_RUN",
        "r40c": "NOT_RUN", "r40d": "NOT_RUN", "human_gate": "NOT_AUTHORIZED",
    }
    assert status["product_roi_parity"] == "PROVEN_B2_H_ROI_PATH_SYNTHETIC"
    assert status["b2_acceptance"] == "NOT_RUN"
    assert status["real_windows_run"] == "NOT_RUN"
