"""ZM-ZEGRID-ADAPTIVE-RESOURCES A4 — Q41: finishing/preview + nearly-full disk.

Gate: when finishing (DBE / RGB equalize / hole fill / uint16) and/or the
aesthetic preview are requested but the disk is nearly full at publication, the
system must NEVER silently disable finishing to make the run fit. It must either
refuse explicitly BEFORE allocating a generation (atomic, previous publication
stays authoritative), or leave an honest resumable state.

The finishing working copies are priced in the RAM model; the delivered members
(including the aesthetic/uint16 members) are priced in the disk model. This test
proves the admission prices finishing truthfully, refuses on a nearly-full disk
WITHOUT dropping finishing, and creates no generation directory on refusal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic.core.zegrid import run_publication as zpub  # noqa: E402

H, W, CH = 64, 64, 3
RUN_ID = "run-q41"


def _finishing_on(export=True, uint16=True):
    return {
        "dbe_enabled": True,
        "rgb_equalize": True,
        "hole_fill_enabled": True,
        "save_uint16": bool(uint16),
        "export_aesthetic_fits": bool(export),
        "scientific_fits_suffix": "_science",
        "aesthetic_fits_suffix": "_aesthetic",
    }


def test_q41_finishing_priced_truthfully_not_silently_disabled():
    # Finishing ON must inflate the RAM peak versus finishing OFF, and the plan
    # must report finishing as enabled (never silently dropped).
    off = zpub.finishing_peak_bytes((H, W), finishing_config={
        "dbe_enabled": False, "rgb_equalize": False,
        "hole_fill_enabled": False, "save_uint16": False,
    })
    on = zpub.finishing_peak_bytes((H, W), finishing_config=_finishing_on())
    assert off["enabled"] is False
    assert on["enabled"] is True
    assert on["peak_bytes"] > off["peak_bytes"]
    # The aesthetic member is charged to disk when requested.
    disk_on = zpub.publication_disk_bytes(
        (H, W), channels=CH, export_aesthetic=True, uint16=True, n_cells=4
    )
    assert disk_on["member_bounds"][zpub.ROLE_AESTHETIC] > 0
    assert disk_on["member_bounds"][zpub.ROLE_UINT16] > 0


def test_q41_nearly_full_disk_refuses_without_dropping_finishing(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    # Give enough free space for the science+coverage+manifest members but NOT
    # the finishing-inflated RAM serialization + aesthetic/uint16 + margin.
    plan = zpub.size_finishing_and_publication(
        output_dir=out,
        shape_hw=(H, W),
        channels=CH,
        finishing_config=_finishing_on(),
        n_cells=4,
        disk_free_bytes=1,
    )
    assert plan["finishing"]["enabled"] is True
    with pytest.raises(zpub.PublicationCapacityError) as excinfo:
        zpub.admit_finishing_and_publication(
            output_dir=out,
            shape_hw=(H, W),
            channels=CH,
            finishing_config=_finishing_on(),
            available_ram_bytes=8 * 2 ** 30,
            n_cells=4,
            disk_free_bytes=1,
        )
    refused = excinfo.value.plan
    assert refused["disk"]["fits"] is False
    # The refusal must still price finishing as enabled (no silent disable).
    assert refused["finishing"]["enabled"] is True
    # Atomic: no generation directory was created by the admission refusal.
    assert not zpub.publication_root(out).exists()


def test_q41_finishing_ram_refusal_keeps_finishing_enabled(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(zpub.PublicationCapacityError) as excinfo:
        zpub.PublicationSession.begin(
            out,
            RUN_ID,
            members={
                zpub.ROLE_SCIENCE: "mosaic_grid.fits",
                zpub.ROLE_COVERAGE: "mosaic_grid_coverage.fits",
                zpub.ROLE_MANIFEST: "zegrid_manifest.json",
                zpub.ROLE_AESTHETIC: "mosaic_grid_aesthetic.fits",
                zpub.ROLE_UINT16: "mosaic_grid_uint16.fits",
            },
            shape_hw=(H, W),
            channels=CH,
            finishing_config=_finishing_on(),
            available_ram_bytes=1024,
            n_cells=4,
        )
    plan = excinfo.value.plan
    assert plan["ram"]["fits"] is False
    assert plan["finishing"]["enabled"] is True
    assert not zpub.publication_root(out).exists()
