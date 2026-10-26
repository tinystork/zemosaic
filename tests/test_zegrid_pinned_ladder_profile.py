"""ZM-PINNED-LADDER-PROFILE — the pinned ladder must use the RUN's storage profile.

The pinned branch of :func:`_choose_layout_mode_aware` used to price its CPU
streaming-tile ladder with ``_estimate_streaming_bytes(n, hw, tile)`` — i.e. the
DENSE default — whatever the run's real ``storage_profile`` was. It also let the
in-memory bound confer admissibility under a BOUNDED profile, where the
in-memory path is not executable at all.

These tests pin both behaviours with synthetic, deterministic corpora (no real
images, no pixels):

1. bounded profile -> EVERY ladder estimate carries ``storage_profile='bounded'``;
2. dense profile   -> NO estimate carries a ``storage_profile`` kwarg and the
   numeric bounds equal the published dense model (monkeypatch compatibility);
3. bounded profile -> a fitting in-memory bound no longer admits the pinned
   layout when no bounded tile fits (refusal instead of a broken plan);
4. bounded profile -> the historical refusal message is preserved verbatim when
   even the bounded model exceeds the budget.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.wcs import WCS

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.zegrid import auto_layout as zal
from zemosaic.core.zegrid import geometry as zg
from zemosaic.core.zegrid import streaming as zstream

MIB = 2 ** 20

#: The fixed ladder, in the order the planner walks it.
LADDER = tuple(int(t) for t in zz.CPU_STREAMING_TILE_LADDER)


# ---------------------------------------------------------------------------
# Synthetic corpus (mirrors the R36 helpers: no user data, no pixels)
# ---------------------------------------------------------------------------

def _tiny_floors():
    return zal.ScientificFloors(
        min_patch_area_px=1, min_contributors=1, max_halo_overhead=1e9
    )


def _make_tan_descriptor(i, h, w, scale, ra_step):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [10.0 + i * ra_step, 30.0]
    wcs.wcs.crpix = [w / 2.0, h / 2.0]
    wcs.wcs.cd = np.array([[-1.0, 0.0], [0.0, 1.0]]) * scale
    wcs.array_shape = (h, w)
    fid = f"d{i}.fits"
    return zg.FrameDescriptor(
        frame_id=zg.FrameId(fid),
        source_path=f"/x/{fid}",
        shape_hw=(h, w),
        wcs_header=wcs.to_header(relax=True).tostring(),
        header_sha256="",
        instrument="",
    )


def _build_tan_corpus(n=4, h=40, w=40, scale=0.001, ra_step=0.004):
    return [_make_tan_descriptor(i, h, w, scale, ra_step) for i in range(n)]


class _FakeModel:
    """Deterministic stand-in for ``zal.FITTED_MEMORY_MODEL`` (constant bound)."""

    def __init__(self, bound_bytes):
        self._b = float(bound_bytes)

    def predict_bound_bytes(self, n, area):
        return self._b

    def predict_peak_bytes(self, n, area):
        return self._b


def _set_inmem_bound(monkeypatch, bound_bytes):
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _FakeModel(bound_bytes))


def _spy_estimator(monkeypatch):
    """Record every ``_estimate_streaming_bytes`` call made by the planner."""
    original = zz._estimate_streaming_bytes
    calls = []

    def spy(n_contributors, patch_hw, tile_size=zz.STREAM_TILE_SIZE,
            channels=3, **kwargs):
        calls.append({
            "n": int(n_contributors),
            "hw": tuple(patch_hw),
            "tile": int(tile_size),
            "channels": int(channels),
            "kwargs": dict(kwargs),
        })
        return original(n_contributors, patch_hw, tile_size, channels, **kwargs)

    monkeypatch.setattr(zz, "_estimate_streaming_bytes", spy)
    return calls


def _set_profile(monkeypatch, profile):
    monkeypatch.setattr(zz, "_STACK_STORAGE_PROFILE", profile)


def _ladder_calls(calls):
    return [c for c in calls if c["tile"] in LADDER]


# ---------------------------------------------------------------------------
# 1. Bounded profile -> the whole ladder is priced with the bounded model
# ---------------------------------------------------------------------------

def test_bounded_profile_prices_every_ladder_call_with_bounded(monkeypatch):
    descs = _build_tan_corpus()
    canvas = zg.build_canvas(descs)
    _set_profile(monkeypatch, "bounded")
    _set_inmem_bound(monkeypatch, 10 ** 12)
    calls = _spy_estimator(monkeypatch)

    layout = zz._choose_layout_mode_aware(
        canvas, descs, ram_budget=int(1e15),
        floors=_tiny_floors(), pinned_layout=(3, 2), footprints_out=[],
    )

    assert layout["layout_source"] == "pinned"
    ladder = _ladder_calls(calls)
    assert ladder, "the pinned ladder must price at least one tile"
    # EVERY ladder estimate must carry the run's real profile.
    assert [c["kwargs"].get("storage_profile") for c in ladder] == (
        ["bounded"] * len(ladder)
    )
    # ... and every ladder tile must actually be tried.
    assert {c["tile"] for c in ladder} == set(LADDER)
    # The estimate is positional exactly like the dense call (no signature drift).
    assert all(c["kwargs"].keys() == {"storage_profile"} for c in ladder)


# ---------------------------------------------------------------------------
# 2. Dense profile -> historical positional call + identical numbers
# ---------------------------------------------------------------------------

def test_dense_profile_keeps_positional_call_and_baseline_bounds(monkeypatch):
    descs = _build_tan_corpus()
    canvas = zg.build_canvas(descs)
    _set_profile(monkeypatch, "dense")
    _set_inmem_bound(monkeypatch, 10 ** 12)
    calls = _spy_estimator(monkeypatch)

    layout = zz._choose_layout_mode_aware(
        canvas, descs, ram_budget=int(1e15),
        floors=_tiny_floors(), pinned_layout=(3, 2), footprints_out=[],
    )

    ladder = _ladder_calls(calls)
    assert ladder
    # Monkeypatch compatibility: no ``storage_profile`` kwarg anywhere.
    assert all("storage_profile" not in c["kwargs"] for c in calls)
    assert all(c["kwargs"] == {} for c in calls)

    # NUMERIC identity with the published dense model, recomputed here from
    # ``zstream`` with its own default profile (independent of ``zz``).
    expected = {t: 0 for t in LADDER}
    for c in ladder:
        b = int(zstream.estimate_streaming_peak_kib(c["n"], c["hw"], c["tile"]) * 1024)
        expected[c["tile"]] = max(expected[c["tile"]], b)
    trials = {int(t["tile"]): int(t["worst_bound_bytes"])
              for t in layout["cpu_streaming"]["trials"]}
    assert trials == expected


# ---------------------------------------------------------------------------
# 3. Bounded profile -> in-memory bound no longer confers admissibility
# ---------------------------------------------------------------------------

def test_bounded_profile_inmem_does_not_admit_pinned_layout(monkeypatch):
    descs = _build_tan_corpus()
    canvas = zg.build_canvas(descs)
    budget = 10 * MIB

    # In-memory would comfortably fit the budget...
    _set_inmem_bound(monkeypatch, 1 * MIB)

    # ... and the DENSE model would fit too, so the only reason for a refusal
    # under `bounded` is the profile itself.
    def fake_peak(n_contributors, patch_hw, tile_size, channels=3, *,
                  storage_profile="dense"):
        assert storage_profile == "bounded"
        return 10 ** 9

    _set_profile(monkeypatch, "bounded")
    monkeypatch.setattr(zstream, "estimate_streaming_peak_kib", fake_peak)

    with pytest.raises(zal.LayoutInfeasible) as exc:
        zz._choose_layout_mode_aware(
            canvas, descs, ram_budget=budget,
            floors=_tiny_floors(), pinned_layout=(3, 2), footprints_out=[],
        )
    msg = str(exc.value)
    assert "cannot fit any mode within budget" in msg
    assert "in-memory worst-cell bound" in msg

    # Control: the SAME geometry/bounds are admissible under the dense profile
    # (the in-memory bound is then executable), so the refusal above is caused
    # by the bounded profile, not by the fixture.
    _set_profile(monkeypatch, "dense")
    monkeypatch.setattr(
        zstream, "estimate_streaming_peak_kib",
        lambda n_contributors, patch_hw, tile_size, channels=3, **kw: 10 ** 9,
    )
    accepted = zz._choose_layout_mode_aware(
        canvas, descs, ram_budget=budget,
        floors=_tiny_floors(), pinned_layout=(3, 2), footprints_out=[],
    )
    assert accepted["layout_source"] == "pinned"
    assert accepted["cpu_streaming"]["selected_tile_size"] is None
    assert accepted["predicted_bound_bytes"] <= budget


# ---------------------------------------------------------------------------
# 4. Bounded profile -> historical refusal message preserved
# ---------------------------------------------------------------------------

def test_bounded_profile_refusal_message_unchanged(monkeypatch):
    descs = _build_tan_corpus()
    canvas = zg.build_canvas(descs)
    _set_profile(monkeypatch, "bounded")
    _set_inmem_bound(monkeypatch, 10 ** 9)
    monkeypatch.setattr(
        zstream, "estimate_streaming_peak_kib",
        lambda n_contributors, patch_hw, tile_size, channels=3, **kw: 10 ** 9,
    )

    with pytest.raises(zal.LayoutInfeasible) as exc:
        zz._choose_layout_mode_aware(
            canvas, descs, ram_budget=10 * MIB,
            floors=_tiny_floors(), pinned_layout=(3, 2), footprints_out=[],
        )
    msg = str(exc.value)
    assert msg.startswith("pinned layout 3x2 cannot fit any mode within budget 10.0 MiB")
    assert "in-memory worst-cell bound 953.7 MiB" in msg
    assert f"(minimum tile {zz.CPU_STREAMING_TILE_MINIMUM})" in msg
    assert "both exceed it" in msg
    assert "deeper disk-backed streaming executor" in msg
    assert "'zegrid_layout' config key" in msg
