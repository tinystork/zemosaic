"""ZM-ZEGRID-ACTIVE-OOC-FALLBACK — synthetic multi-cell mosaic parity (H64/H128).

For each halo, two COMPLETE public runs on the SAME pinned geometry:

* forced ``dense`` (no fallback);
* ``auto`` with a deterministic typed capacity refusal at the pre-gauge ACTIVE
  admission, so the run really switches to ``bounded`` at the same pin.

The two runs must produce bit-identical science and coverage FITS, identical
multi-cell contributor identities and identical additive offsets/feather.

This is a synthetic corpus witness, never a real M106/Caldwell mosaic.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.zegrid import ram_contract as rcontract

pytestmark = pytest.mark.filterwarnings(
    "ignore::astropy.io.fits.verify.VerifyWarning"
)

MIB = 2 ** 20
RAW = 4 * 2 ** 30
HALO_LAYOUT = {64: "2x2", 128: "2x1"}
FRAMES = 12
FRAME_HW = (900, 900)
SCALE = 0.001
STEP_PX = 540.0


def _write_corpus(root: Path) -> Path:
    rng = np.random.default_rng(20261026)
    inp = root / "in"
    inp.mkdir()
    names = []
    h, w = FRAME_HW
    for i in range(FRAMES):
        col, row = i % 4, i // 4
        ra = 10.0 + col * STEP_PX * SCALE
        dec = 30.0 - row * STEP_PX * SCALE
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.crval = [ra, dec]
        wcs.wcs.crpix = [w / 2.0, h / 2.0]
        wcs.wcs.cd = np.array([[-1.0, 0.0], [0.0, 1.0]]) * SCALE
        wcs.array_shape = (h, w)
        yy, xx = np.mgrid[0:h, 0:w]
        data = (
            100.0
            + 25.0 * np.exp(
                -((yy - h / 2) ** 2 + (xx - w / 2) ** 2) / (2 * 180.0 ** 2)
            )
            + 2.0 * i
            + rng.normal(0.0, 0.5, (h, w))
        ).astype(np.float32)
        hdu = fits.PrimaryHDU(data)
        hdu.header.update(wcs.to_header(relax=True))
        hdu.header["EXPTIME"] = 10.0
        name = f"frame_{i:04d}.fits"
        hdu.writeto(inp / name, overwrite=True)
        names.append(name)
    (inp / "stack_plan.csv").write_text(
        "file_path,exposure\n" + "".join(f"{n},10.0\n" for n in names),
        encoding="utf-8",
    )
    return inp


def _zconfig(halo: int):
    return SimpleNamespace(
        zegrid_scheduler_mode="active",
        zegrid_layout=HALO_LAYOUT[int(halo)],
        zegrid_halo_px=int(halo),
        cell_gauge_mode="per_cell",
    )


def _run(inp, out, halo, *, auto_fallback, monkeypatch):
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: RAW)
    if auto_fallback:
        real_admit = zz._r55_active_admit
        state = {"injected": False}

        def _admit(*args, **kwargs):
            if not state["injected"] and zz._STORAGE_EFFECTIVE == "dense":
                state["injected"] = True
                raise zz.zsched.ActivePlanRejected(
                    zz.renv.REASON_REFUSE_HOST_RAM,
                    "deterministic capacity miss for the AUTO witness",
                )
            return real_admit(*args, **kwargs)

        monkeypatch.setattr(zz, "_r55_active_admit", _admit)
        zz.run_zegrid_mode(
            str(inp), str(out), zconfig=_zconfig(halo), storage_profile=None,
            cell_gauge_mode="per_cell",
        )
        assert state["injected"] is True
        return _manifest(out)
    monkeypatch.setattr(zz, "_r55_active_admit", zz._r55_active_admit)
    zz.run_zegrid_mode(
        str(inp), str(out), zconfig=_zconfig(halo), storage_profile="dense",
        cell_gauge_mode="per_cell",
    )
    return _manifest(out)


def _manifest(out: Path):
    return json.loads((out / "zegrid_manifest.json").read_text())


def _array(path: Path):
    with fits.open(path) as hdul:
        return np.asarray(hdul[0].data)


@pytest.mark.slow
@pytest.mark.parametrize("halo", [64, 128])
def test_synthetic_multicell_dense_vs_auto_bounded_parity(
    tmp_path, monkeypatch, halo,
):
    inp = _write_corpus(tmp_path)
    dense_out = tmp_path / f"dense_h{halo}"
    auto_out = tmp_path / f"auto_h{halo}"

    dense_manifest = _run(inp, dense_out, halo, auto_fallback=False,
                          monkeypatch=monkeypatch)
    auto_manifest = _run(inp, auto_out, halo, auto_fallback=True,
                         monkeypatch=monkeypatch)

    # Fallback really happened at the pre-gauge admission on the SAME pin.
    assert dense_manifest["storage"]["requested"] == "dense"
    assert dense_manifest["storage"]["effective"] == "dense"
    assert dense_manifest["storage"]["fallback_count"] == 0
    assert auto_manifest["storage"]["requested"] == "auto"
    assert auto_manifest["storage"]["effective"] == "bounded"
    assert auto_manifest["storage"]["fallback_count"] == 1
    assert auto_manifest["storage"]["transitions"][0]["phase"] == (
        "admission_preflight"
    )

    # Same geometry / same multi-cell identity.
    _nx, _ny = (int(v) for v in HALO_LAYOUT[halo].split("x"))
    assert dense_manifest["layout"]["nx"] == _nx
    assert dense_manifest["layout"]["ny"] == _ny
    assert auto_manifest["layout"]["nx"] == _nx
    assert auto_manifest["layout"]["ny"] == _ny
    d_cells = [(c["row"], c["col"], c["n_contributors"])
               for c in dense_manifest.get("cells", [])]
    a_cells = [(c["row"], c["col"], c["n_contributors"])
               for c in auto_manifest.get("cells", [])]
    assert d_cells == a_cells
    assert len(d_cells) == _nx * _ny
    assert all(n > 0 for _r, _c, n in d_cells)

    # Science + coverage are bit-identical.
    for name in ("mosaic_grid.fits", "mosaic_grid_coverage.fits"):
        np.testing.assert_array_equal(
            _array(dense_out / name), _array(auto_out / name), err_msg=name
        )

    # Additive offsets and feathering are identical.
    d_feather = dense_manifest["cell_feather"]
    a_feather = auto_manifest["cell_feather"]
    assert d_feather["enabled"] is True
    assert a_feather["enabled"] is True
    assert d_feather["halo_px"] == a_feather["halo_px"] == halo
    assert d_feather["offsets"] == a_feather["offsets"]
    assert d_feather["edges_used"] == a_feather["edges_used"]
    assert d_feather["edges_built"] == a_feather["edges_built"]
    assert d_feather["weighted_residual_rms"] == pytest.approx(
        a_feather["weighted_residual_rms"], rel=0, abs=0
    )

    # The bounded run really ran the bounded provider + sink on every cell.
    bounded_cells = [
        c for c in auto_manifest.get("cells", [])
        if c.get("storage_profile") is not None
    ]
    assert bounded_cells, "no per-cell bounded provenance in the manifest"
    for cell in bounded_cells:
        storage = cell["storage"]
        assert storage["provider"] == "BoundedMemmapCanonicalProvider"
        assert storage["sink"]["type"] == "BoundedCountingRejectionSink"


# ---------------------------------------------------------------------------
# Per-cell SEVEN-plane parity + exact contributor IDs at the SAME pinned cells
# ---------------------------------------------------------------------------

CORE_PLANES = (
    "science_core",
    "estimator_weight_sum_core",
    "support_w1_core",
    "support_w2_core",
    "n_eff_support_core",
    "valid_mask_core",
    "surviving_sample_count_core",
)


@pytest.mark.slow
@pytest.mark.parametrize("halo", [64, 128])
def test_pinned_mosaic_cell_seven_planes_and_ids(tmp_path, monkeypatch, halo):
    """Every pinned Cell of the synthetic mosaic: dense vs bounded, 7 planes.

    Uses an aligned cache built with the EXACT contributor ID list and patch
    shape of each pinned Cell, so the comparison is at the real mosaic cell
    geometry/membership without re-running the full reprojection twice.
    """
    import numpy as np
    from zemosaic.core.zegrid import assembly as za
    from zemosaic.core.zegrid import file_provider as zfp
    from zemosaic.core.zegrid import geometry as zg
    from zemosaic.core.zegrid import science_adapter as zsa

    inp = _write_corpus(tmp_path)
    frames, rejected = zg.read_manifest(inp)
    assert rejected == []
    canvas = zg.build_canvas(frames)
    nx, ny = (int(v) for v in HALO_LAYOUT[halo].split("x"))
    raw = RAW
    budget = rcontract.plan_budget_bytes(raw)

    zz._storage_policy_reset("dense")
    footprints = []
    layout = zz._choose_layout_mode_aware(
        canvas, frames, budget, halo_px=halo,
        pinned_layout=(nx, ny), footprints_out=footprints,
    )
    assert (layout["nx"], layout["ny"]) == (nx, ny)
    ctxs = zz._build_chosen_cell_contexts(
        footprints[0], canvas, nx, ny, None, halo_px=halo
    )
    cfg = zsa.MiniTileScienceConfig(cell_gauge_mode="per_cell")
    monkeypatch.setattr(zz, "_STACK_CONFIG", cfg)
    monkeypatch.setattr(zz, "_STACK_GAUGE", None)
    monkeypatch.setattr(zz, "_STACK_GLOBAL_FRAME_IDS", None)
    monkeypatch.setattr(zz, "_STACK_GLOBAL_REF", None)
    monkeypatch.setattr(zz, "_STACK_TILE_SIZE", 128)
    monkeypatch.setattr(zz, "_STACK_CACHE_WORKERS", 1)

    rng = np.random.default_rng(31 + halo)
    checked = 0
    for _row, _col, cell, patch, mem in ctxs:
        if not mem.patch_ids:
            continue
        ids = list(mem.patch_ids)
        ph, pw = (int(v) for v in patch.patch_shape_hw)
        n = len(ids)
        frames_arr = [
            (rng.normal(0, 2, (ph, pw, 3)).astype(np.float32)
             + np.float32(10 + i))
            for i in range(n)
        ]
        masks = [np.ones((ph, pw), dtype=bool) for _ in range(n)]
        cache = tmp_path / f"cell_{_row}_{_col}"
        zfp.write_aligned_cache_from_arrays(cache, ids, frames_arr, masks)

        dense_info, bounded_info = {}, {}
        dense_mt, dense_sres = zz._run_cell_stream(
            cache, patch, cfg, None, tile_size=128,
            storage_profile="dense", storage_info=dense_info,
        )
        bounded_mt, bounded_sres = zz._run_cell_stream(
            cache, patch, cfg, None, tile_size=128,
            storage_profile="bounded", storage_info=bounded_info,
        )
        dense_core = za.crop_all_planes_to_core(dense_mt)
        bounded_core = za.crop_all_planes_to_core(bounded_mt)
        for name in CORE_PLANES:
            np.testing.assert_array_equal(
                dense_core[name], bounded_core[name],
                err_msg=f"{cell.cell_id}:{name}",
            )
        # Exact contributor identity (same ordered list, no add/drop).
        assert tuple(dense_sres.frame_order) == tuple(ids)
        assert tuple(bounded_sres.frame_order) == tuple(ids)
        assert bounded_info["provider"] == "BoundedMemmapCanonicalProvider"
        assert bounded_info["sink"]["type"] == "BoundedCountingRejectionSink"
        assert bounded_info["sink"]["written_cells"] == n * ph * pw * 3
        assert dense_info["sink"] is None
        checked += 1
    assert checked == nx * ny
