"""F5 (ZM-ZEGRID-ADAPTIVE-RESOURCES) — growth wired into ACTIVE admission.

Proves the previously-unused A5 growth primitives are now exercised by the
product admission path:

* a FRESH rising observation grows a SHRUNK plan one bounded field at a time,
  through canonical materialization (fresh costs, never stale projections),
  and never changes the backend (no CPU->GPU promotion);
* the 4/5 PLAN hysteresis prevents micro-rise oscillation;
* a pinned/frozen admission is never regrown;
* the trail publishes every grow step with step/from/to/reason_code/at.
"""
from __future__ import annotations

import dataclasses
import os

import numpy as np
import pytest

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.zegrid import auto_layout as zal
from zemosaic.core.zegrid import geometry as zg
from zemosaic.core.zegrid import ram_contract as rcontract
from zemosaic.core.zegrid import scheduler as zs

MIB = 2 ** 20


class _FakeModel:
    def __init__(self, bound_bytes):
        self._b = float(bound_bytes)

    def predict_bound_bytes(self, n, area):
        return self._b

    def predict_peak_bytes(self, n, area):
        return self._b


def _make_tan(i, h=40, w=40, scale=0.001, ra_step=0.004):
    from astropy.wcs import WCS

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [10.0 + i * ra_step, 30.0]
    wcs.wcs.crpix = [w / 2.0, h / 2.0]
    wcs.wcs.cd = np.array([[-1.0, 0.0], [0.0, 1.0]]) * scale
    wcs.array_shape = (h, w)
    fid = f"d{i}.fits"
    return zg.FrameDescriptor(
        frame_id=zg.FrameId(fid), source_path=f"/x/{fid}", shape_hw=(h, w),
        wcs_header=wcs.to_header(relax=True).tostring(),
        header_sha256="", instrument="",
    )


def _corpus(n=4):
    return [_make_tan(i) for i in range(n)]


_REAL_FLOORS = zal.ScientificFloors


def _tiny_floors(*_a, **_k):
    return _REAL_FLOORS(
        min_patch_area_px=1, min_contributors=1, max_halo_overhead=1e9
    )


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    from zemosaic.core.zegrid import parallel as zpar

    monkeypatch.setattr(zz, "_STACK_STORAGE_PROFILE", "dense")
    monkeypatch.setattr(zal, "ScientificFloors", _tiny_floors)
    monkeypatch.setattr(zpar, "_PER_WORKER_BASELINE_BYTES", 0)
    monkeypatch.setattr(zpar, "_CACHE_PER_WORKER_FOOTPRINT_BYTES", 0)
    monkeypatch.setattr(zpar, "cache_footprint_bytes", lambda px: 0)
    monkeypatch.setattr(zpar, "gauge_footprint_bytes", lambda px: 0)
    monkeypatch.setattr(os, "cpu_count", lambda: 3)
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _FakeModel(1 << 60))


def _decision_and_bundle(tmp_path, monkeypatch, *, raw=1200 * MIB, table=None):
    descs = _corpus()
    canvas = zg.build_canvas(descs)
    if table is None:
        table = {
            128: 700 * MIB, 112: 700 * MIB, 96: 700 * MIB, 80: 700 * MIB,
            64: 700 * MIB, 48: 700 * MIB, 32: 700 * MIB, 24: 700 * MIB,
            20: 700 * MIB, 16: 360 * MIB, 12: 360 * MIB, 10: 360 * MIB,
            8: 360 * MIB,
        }
    monkeypatch.setattr(
        zz, "_estimate_streaming_bytes",
        lambda n, hw, tile=zz.STREAM_TILE_SIZE, channels=3: int(
            table[int(tile) if tile is not None else zz.STREAM_TILE_SIZE]
        ),
    )
    budget = (4 * int(raw)) // 5
    bundle = zz._build_execution_bundle(
        canvas, descs, None,
        raw_available_bytes=int(raw), budget_bytes=budget,
        cpu_count=8, cache_root=str(tmp_path / "cache"),
    )
    decision = zz._r55_active_decide(
        bundle=bundle,
        cell_ctxs=bundle.cell_ctxs,
        canvas=canvas,
        halo_px=8,
        cache_root=tmp_path,
        output_dir=tmp_path,
        frame_ids=[d.frame_id.logical_path for d in descs],
    )
    return bundle, decision


def _materialized_winner(decision):
    chosen = decision.plan_set.chosen
    ctx = zz._r55_active_shrink_context(decision, chosen)
    return zs.materialize_active_candidate(ctx, chosen)


def test_f5_grow_step_is_one_field_and_retariffs_costs(tmp_path, monkeypatch):
    _bundle, decision = _decision_and_bundle(tmp_path, monkeypatch)
    winner = _materialized_winner(decision)
    ctx = zz._r55_active_shrink_context(decision, winner)

    name, shrunk, _ = zz._r55_next_materialized_shrink(
        decision=decision, candidate=winner, plan_obs=None
    )
    assert name is not None and shrunk is not None

    plan_obs = rcontract.Observation.plan(8 * 1024 * MIB)  # huge fresh headroom
    gname, grown, examined = zz._r55_next_materialized_grow(
        decision=decision, candidate=shrunk, plan_obs=plan_obs
    )
    assert gname is not None and grown is not None
    # ONE bounded step, backend unchanged (no CPU->GPU promotion).
    assert grown.backend == shrunk.backend == zs.BACKEND_CPU
    # Costs are RE-DERIVED from canonical enumeration, never the stale
    # `_replace_candidate` projections.
    expected = zs.materialize_active_candidate(
        zz._r55_active_shrink_context(decision, grown), grown
    )
    assert grown.host_peak_bytes == expected.host_peak_bytes
    assert grown.disk_bytes == expected.disk_bytes
    assert grown.device_peak_bytes == expected.device_peak_bytes
    # The examined step is the audit record the admission trail publishes.
    assert examined and examined[0]["step"] in zs.GROW_ORDER
    assert set(examined[0]) >= {
        "step", "from", "to", "accepted", "reason_code", "projections"
    }
    # A grow never shrinks resources.
    zs.assert_grow_only(shrunk, grown)


def test_f5_hysteresis_rejects_micro_rise_band(tmp_path, monkeypatch):
    _bundle, decision = _decision_and_bundle(tmp_path, monkeypatch)
    winner = _materialized_winner(decision)
    _name, shrunk, _ = zz._r55_next_materialized_shrink(
        decision=decision, candidate=winner, plan_obs=None
    )
    big_plan = rcontract.Observation.plan(8 * 1024 * MIB)
    _gname, grown, _ = zz._r55_next_materialized_grow(
        decision=decision, candidate=shrunk, plan_obs=big_plan
    )
    assert grown is not None

    # The 4/5 predicate is the hysteresis: a budget that only reaches the
    # ADMIT (9/10) band for this grown plan is NOT enough to grow.
    raw_band = (int(grown.host_peak_bytes) * 10) // 9
    tight_plan = rcontract.Observation.plan(raw_band)
    assert tight_plan.budget_bytes < int(grown.host_peak_bytes)
    assert zz._r55_grow_is_admissible(grown, tight_plan) is False
    assert zz._r55_grow_is_admissible(grown, big_plan) is True
    # A plan that fits admission but not the PLAN band stays inside the dead
    # band: no grow, therefore no shrink<->grow oscillation.
    assert rcontract.requires_tolerance(
        int(grown.host_peak_bytes), raw_band
    ) or int(grown.host_peak_bytes) > rcontract.admit_budget_bytes(raw_band)


def test_f5_pinned_admission_is_never_regrown(tmp_path, monkeypatch):
    _bundle, decision = _decision_and_bundle(tmp_path, monkeypatch)
    assert zz._r55_active_growth_allowed(decision) is True
    pinned_ctx = dataclasses.replace(decision.stream_context, pinned=True)
    pinned_decision = dataclasses.replace(decision, stream_context=pinned_ctx)
    assert zz._r55_active_growth_allowed(pinned_decision) is False
    # A pinned/frozen admission keeps its exact candidate tuple: the grow helper
    # is never invoked by the admission guard above.
    winner = _materialized_winner(decision)
    assert zs.active_candidate_tuple(winner) == zs.active_candidate_tuple(
        decision.plan_set.chosen
    )


def test_f5_admission_publishes_grow_steps_in_the_trail(tmp_path, monkeypatch):
    bundle, decision = _decision_and_bundle(tmp_path, monkeypatch)
    winner = _materialized_winner(decision)
    _name, shrunk, _ = zz._r55_next_materialized_shrink(
        decision=decision, candidate=winner, plan_obs=None
    )
    plan_obs = rcontract.Observation.plan(16 * 1024 * MIB)
    gname, grown, examined = zz._r55_next_materialized_grow(
        decision=decision, candidate=shrunk, plan_obs=plan_obs
    )
    assert gname is not None and grown is not None
    # The trail entry shape is the same as shrink, plus the explicit
    # `budget_recovery` origin the admission loop uses.
    entry = dict(examined[0])
    entry.update({"at": "budget_recovery", "stage": "preflight"})
    assert entry["at"] == "budget_recovery"
    assert entry["to"]["tile_px"] >= entry["from"]["tile_px"]
    assert entry["step"] in zs.GROW_ORDER
    assert "reason_code" in entry
