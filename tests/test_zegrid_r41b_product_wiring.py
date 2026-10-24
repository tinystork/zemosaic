"""ZM-ZEGRID-R41-B — product wiring of the host-RAM contract (80 % plan / 90 % admit).

Covers the FOCUSED acceptance surface on the REAL product seam
(``_build_execution_bundle`` / ``_run_single`` / ``run_zegrid_mode``), NOT on a
re-implementation:

* A  constant-RAM reproduction of the old 100 %/80 % trap, now resolved by a
   JOINT-FEASIBLE FINER Auto layout (planner evidence);
* B  an impossible initial plan refuses BEFORE the gauge (spy proves the gauge
   was never called);
* C  the 20 GiB plan admitted unchanged at 18 GiB (tolerance) and replanned under
   13.6 GiB (never 15.3 GiB) at 17 GiB;
* D  Auto never regrows; a pinned geometry is invariant (only tile/concurrency
   may shrink);
* E  ``auto`` resolves on the FINAL admitted membership, before the R37 gate;
* F  the admitted bundle really reaches the worker executor + manifest;
* G  the shared/cache aggregate can reduce K / refuse when ONE cell alone fits;
* I  the R37 preflight and the runtime planner agree under the SAME raw/layout;
* J  a nonpositive RAW observation fails closed and the geometry never changes
   after work has started.

The heavy science is mocked ONLY for ordering/propagation claims; the planner /
contract math is always the real one.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

# The product's own FITS write (``_write_raw_science_fits``) emits the benign,
# PRE-EXISTING ``VerifyWarning: Card is too long, comment will be truncated``
# (the same warning the R20/R36 suites avoid only by not running under
# ``-W error``). Ignore ONLY that specific category so this module stays clean
# under the documented ``-W error`` gate without hiding any other warning.
pytestmark = pytest.mark.filterwarnings(
    "ignore::astropy.io.fits.verify.VerifyWarning"
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic import zemosaic_zegrid_mode as zz  # noqa: E402
from zemosaic.core.zegrid import auto_layout as zal  # noqa: E402
from zemosaic.core.zegrid import geometry as zg  # noqa: E402
from zemosaic.core.zegrid import parallel as zpar  # noqa: E402
from zemosaic.core.zegrid import ram_contract as rc  # noqa: E402

MIB = 2 ** 20
GB = 2 ** 30
TOOL = REPO_ROOT / "tools" / "zegrid_routine" / "make_routine_corpus.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REAL_FLOORS = zal.ScientificFloors


def _tiny_floors(**_kw):
    # NB: capture the REAL class first — this function REPLACES ``zal.ScientificFloors``
    # in the tests below, so calling ``zal.ScientificFloors`` here would recurse.
    return _REAL_FLOORS(
        min_patch_area_px=1, min_contributors=1, max_halo_overhead=1e9
    )


class _FakeModel:
    """Constant per-cell bound (mirrors the R36 test-local stand-in)."""

    def __init__(self, bound_bytes):
        self._b = float(bound_bytes)

    def predict_bound_bytes(self, n, area):
        return self._b

    def predict_peak_bytes(self, n, area):
        return self._b


class _AreaModel:
    """Test-local memory model: bound grows with the (halo'd) patch area."""

    def __init__(self, bytes_per_px):
        self._bpp = float(bytes_per_px)

    def predict_bound_bytes(self, n, area):
        return int(max(1, area) * self._bpp)

    def predict_peak_bytes(self, n, area):
        return self.predict_bound_bytes(n, area)


def _make_tan(i, h, w, scale, ra_step):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [10.0 + i * ra_step, 30.0]
    wcs.wcs.crpix = [w / 2.0, h / 2.0]
    wcs.wcs.cd = np.array([[-1.0, 0.0], [0.0, 1.0]]) * scale
    wcs.array_shape = (h, w)
    fid = f"d{i}.fits"
    return zg.FrameDescriptor(
        frame_id=zg.FrameId(fid), source_path=f"/x/{fid}", shape_hw=(h, w),
        wcs_header=wcs.to_header(relax=True).tostring(), header_sha256="", instrument="",
    )


def _corpus(n=4, h=40, w=40, scale=0.001, ra_step=0.004):
    return [_make_tan(i, h, w, scale, ra_step) for i in range(n)]


def _area_bounds(monkeypatch, *, inmem_bpp=4 * MIB, stream_bpp=4 * MIB):
    """Bounds scale with patch area -> finer layouts are genuinely cheaper.

    The STREAM estimate is deliberately TILE-INDEPENDENT here: it makes the fixture
    deterministic (no adaptive-tile recovery), so a coarse layout that fails B_plan
    fails for EVERY permitted ladder tile and only a FINER layout can pass. The
    in-memory bound is EQUAL to the stream bound so the K==1 aggregate is
    mode-independent (the trap cannot be dodged by switching mode).
    """
    monkeypatch.setattr(zal, "ScientificFloors", _tiny_floors)
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _AreaModel(inmem_bpp))
    monkeypatch.setattr(zz, "_estimate_streaming_bytes",
                        lambda n, hw, tile=zz.STREAM_TILE_SIZE, channels=3:
                        int(max(1, int(hw[0]) * int(hw[1])) * stream_bpp))


def _neutralize_workers_only(monkeypatch):
    """Neutralize only the CACHE/worker footprints, KEEPING the real shared
    parent reserve (needed to construct 'the layout is admissible but the JOINT
    CPU phase aggregate busts the budget')."""
    monkeypatch.setattr(zpar, "_CACHE_PER_WORKER_FOOTPRINT_BYTES", 0)
    monkeypatch.setattr(zpar, "cache_footprint_bytes", lambda px: 0)
    monkeypatch.setattr(zpar, "gauge_footprint_bytes", lambda px: 0)
    monkeypatch.setattr(os, "cpu_count", lambda: 3)


def _neutralize_shared_reserve(monkeypatch):
    """R41-B: the byte sentinels in this file are per-cell bounds only."""
    monkeypatch.setattr(zpar, "_PER_WORKER_BASELINE_BYTES", 0)
    monkeypatch.setattr(zpar, "_CACHE_PER_WORKER_FOOTPRINT_BYTES", 0)
    monkeypatch.setattr(zpar, "cache_footprint_bytes", lambda px: 0)
    monkeypatch.setattr(zpar, "gauge_footprint_bytes", lambda px: 0)
    monkeypatch.setattr(os, "cpu_count", lambda: 3)


def _bundle(canvas, descs, *, raw, budget, gpu_used=False, vram_budget_bytes=None, **kw):
    return zz._build_execution_bundle(
        canvas, descs, None, raw_available_bytes=int(raw), budget_bytes=int(budget),
        gpu_used=gpu_used, vram_budget_bytes=vram_budget_bytes,
        cpu_count=8, cache_root=None, **kw,
    )


# ---------------------------------------------------------------------------
# A — constant RAM: the old 100/80 trap, now solved by a JOINT-FEASIBLE layout
# ---------------------------------------------------------------------------

def test_a_constant_ram_trap_resolved_by_joint_feasible_finer_auto(monkeypatch):
    descs = _corpus()
    canvas = zg.build_canvas(descs)
    _area_bounds(monkeypatch)

    # The coarsest floor-feasible candidate (what the pre-fix 100 % scan admitted).
    coarsest = _bundle(canvas, descs, raw=64 * GB, budget=rc.plan_budget_bytes(64 * GB),
                       max_cells_in_flight=1)
    rank0 = int(coarsest.envelope.refinement_rank)
    agg0 = int(coarsest.envelope.aggregate_bytes)
    assert agg0 > int(coarsest.envelope.shared_bytes)

    # Constant RAM = exactly the coarsest K==1 aggregate: the OLD product admitted
    # that layout at 100 % and then REFUSED at execution (0.8 * raw < aggregate).
    raw = agg0
    b_plan = rc.plan_budget_bytes(raw)
    assert b_plan < agg0, "the 100 %/80 % trap did not reproduce"

    # (i) the old contract: the SAME geometry cannot be recovered even by the
    # smallest permitted tile at B_plan (tile-independent stream fixture).
    with pytest.raises((zal.LayoutInfeasible, zpar.ModeInfeasible)):
        _bundle(canvas, descs, raw=raw, budget=b_plan, max_cells_in_flight=1,
                min_refinement_rank=rank0, max_refinement_rank=rank0)

    # (ii) the fix: a STRICTLY FINER Auto layout whose joint aggregate fits B_plan.
    fixed = _bundle(canvas, descs, raw=raw, budget=b_plan, max_cells_in_flight=1)
    assert int(fixed.envelope.refinement_rank) > rank0
    assert int(fixed.envelope.aggregate_bytes) <= b_plan
    assert fixed.envelope.budget_bytes == b_plan
    assert fixed.envelope.raw_available_bytes == raw
    # …and the finer layout is a REAL layout (more cells), never coarser.
    assert fixed.envelope.layout[0] * fixed.envelope.layout[1] > rank0


def test_a2_recorded_caldwell_tables_are_planner_evidence():
    """Recorded Caldwell bounds: 5x3/4x3 refuse, 6x4 passes at BOTH budgets
    (scope: PLANNER evidence only, with shared=0 and the cache increment disabled)."""
    def mib(x):
        return int(round(x * MIB))

    for cells, inmem, stream, budget in (
        (15, 136220.8, 17229.4, 15574.8),
        (12, 164395.4, 18655.4, 17836.1),
    ):
        with pytest.raises(zpar.ModeInfeasible):
            zpar.plan_cell_concurrency(
                [(mib(inmem), mib(stream))] * cells, 16, mib(budget),
                shared_bytes=0, cache_worker_footprint_bytes=0,
            )
    for budget in (15574.8, 17836.1):
        plan = zpar.plan_cell_concurrency(
            [(mib(78995.6), mib(11224.7))] * 24, 16, mib(budget),
            shared_bytes=0, cache_worker_footprint_bytes=0,
        )
        assert plan["mode"] == "stream"
        assert plan["aggregate_bytes"] <= mib(budget)
        assert plan["cells_in_flight"] == 1


# ---------------------------------------------------------------------------
# B — an impossible plan refuses BEFORE the gauge (spy)
# ---------------------------------------------------------------------------

def _write_corpus(tmp_path):
    corpus = tmp_path / "corpus"
    subprocess.run([sys.executable, str(TOOL), str(corpus)], check=True,
                   capture_output=True, timeout=180)
    inp = tmp_path / "in"
    inp.mkdir()
    for p in sorted(corpus.glob("*.fits")):
        (inp / p.name).write_bytes(p.read_bytes())
    (inp / "stack_plan.csv").write_text((corpus / "stack_plan.csv").read_text(),
                                        encoding="utf-8")
    return inp, tmp_path / "out"


def test_b_impossible_plan_refuses_before_the_gauge(tmp_path, monkeypatch):
    inp, out = _write_corpus(tmp_path)
    calls = {"gauge": 0, "stack": 0}

    def _gauge_spy(*a, **k):
        calls["gauge"] += 1
        raise AssertionError("the global gauge must NOT run when no bundle exists")

    monkeypatch.setattr(zz.zphot, "compute_global_gauge", _gauge_spy)
    # Real, positive RAM, but no bound can ever fit: every mode is infeasible.
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: 4 * GB)
    monkeypatch.setattr(zal, "ScientificFloors", _tiny_floors)
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _AreaModel(8 * GB))
    monkeypatch.setattr(zz, "_estimate_streaming_bytes", lambda *a, **k: 64 * GB)

    with pytest.raises((zal.LayoutInfeasible, zpar.ModeInfeasible)):
        zz.run_zegrid_mode(str(inp), str(out))
    assert calls["gauge"] == 0
    assert not (out / "zegrid_manifest.json").exists()


# ---------------------------------------------------------------------------
# C — the 20 -> 18 / 17 GiB admission contract, through the controller
# ---------------------------------------------------------------------------

def _env(aggregate, raw, *, layout=(6, 4), rank=24, cpu_tile=128, k=1):
    return rc.PlanEnvelope(
        phase=rc.PHASE_PLAN, raw_available_bytes=int(raw),
        factor_num=rc.PLAN_FACTOR_NUM, factor_den=rc.PLAN_FACTOR_DEN,
        budget_bytes=rc.plan_budget_bytes(int(raw)), layout=tuple(layout),
        refinement_rank=int(rank), cpu_tile=int(cpu_tile), gpu_tile=None,
        backend=rc.BACKEND_CPU, cells_in_flight=int(k), cache_workers_per_cell=1,
        shared_bytes=0, stack_total_bytes=int(aggregate),
        cache_total_bytes=int(aggregate), aggregate_bytes=int(aggregate),
        pinned=False, reason="test",
    )


def test_c_20gib_plan_admitted_at_18_and_replanned_under_13_6():
    plan = _env(16 * GB, 20 * GB)

    out = rc.AdmissionController().run(plan, lambda: 18 * GB, lambda *a: None)
    assert out.replans == 0 and out.tolerance is True
    assert out.plan.aggregate_bytes == 16 * GB

    seen = []

    def rebuild(b_plan, previous):
        seen.append(b_plan)
        return _env(b_plan, 17 * GB)

    out2 = rc.AdmissionController().run(plan, lambda: 17 * GB, rebuild)
    assert seen == [(4 * 17 * GB) // 5]        # 13.6 GiB
    assert (9 * 17 * GB) // 10 not in seen     # NEVER 15.3 GiB (0.9 * B_plan)
    assert out2.replans == 1


# ---------------------------------------------------------------------------
# D — Auto never regrows; pinned geometry is invariant
# ---------------------------------------------------------------------------

def test_d_auto_replan_never_regrows(monkeypatch):
    descs = _corpus()
    canvas = zg.build_canvas(descs)
    _area_bounds(monkeypatch)
    b0 = _bundle(canvas, descs, raw=32 * GB, budget=rc.plan_budget_bytes(32 * GB))
    b1 = _bundle(canvas, descs, raw=8 * GB, budget=rc.plan_budget_bytes(8 * GB),
                 min_refinement_rank=b0.envelope.refinement_rank,
                 max_cpu_tile=b0.envelope.cpu_tile,
                 max_cells_in_flight=b0.envelope.cells_in_flight)
    rc.assert_no_growth(b0.envelope, b1.envelope)
    assert b1.envelope.refinement_rank >= b0.envelope.refinement_rank
    assert b1.envelope.cpu_tile <= b0.envelope.cpu_tile
    assert b1.envelope.cells_in_flight <= b0.envelope.cells_in_flight


def test_d2_pinned_geometry_is_invariant_or_refuses(monkeypatch):
    """BOTH branches of the pinned contract: the pinned nx/ny adapts tile/K when a
    permitted tile still fits B_plan, and REFUSES quantitatively (nx/ny unchanged)
    when NO permitted tile does."""
    descs = _corpus()
    canvas = zg.build_canvas(descs)
    _neutralize_shared_reserve(monkeypatch)
    monkeypatch.setattr(zal, "ScientificFloors", _tiny_floors)
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _FakeModel(10 ** 12))  # inmem fails
    table = {128: 8000, 112: 7000, 96: 6000, 80: 5000, 64: 4000, 48: 3000, 32: 2000}
    monkeypatch.setattr(zz, "_estimate_streaming_bytes",
                        lambda n, hw, tile=zz.STREAM_TILE_SIZE, channels=3:
                        int(table.get(int(tile) if tile is not None else 128, 8000)))

    # (i) tile 128 does not fit B_plan=8000, tile 64 does -> SAME 2x2, smaller tile.
    big = _bundle(canvas, descs, raw=10000, budget=8000, pinned_layout=(2, 2),
                  max_cells_in_flight=4)
    assert big.envelope.cpu_tile == 128
    small = _bundle(canvas, descs, raw=5000, budget=4000, pinned_layout=(2, 2),
                    max_cpu_tile=big.envelope.cpu_tile,
                    max_cells_in_flight=big.envelope.cells_in_flight)
    assert small.envelope.layout == (2, 2) == big.envelope.layout
    assert small.envelope.pinned is True
    assert small.envelope.cpu_tile == 64          # adapted DOWN the ladder
    rc.assert_no_growth(big.envelope, small.envelope)

    # (ii) no permitted tile fits -> quantitative refusal, geometry NEVER changes.
    with pytest.raises((zal.LayoutInfeasible, zpar.ModeInfeasible)) as exc:
        _bundle(canvas, descs, raw=1000, budget=1000, pinned_layout=(2, 2))
    assert "2x2" in str(exc.value)


# ---------------------------------------------------------------------------
# G — shared/cache aggregate: K reduction and ONE-CELL refusal
# ---------------------------------------------------------------------------

def test_g_aggregate_can_reduce_k_when_one_cell_alone_fits():
    cells = [(GB, GB)] * 8
    loose = zpar.plan_cell_concurrency(
        cells, 16, 8 * GB, shared_bytes=0, cache_worker_px=0,
    )
    tight = zpar.plan_cell_concurrency(
        cells, 16, 3 * GB, shared_bytes=0, cache_worker_px=0,
    )
    assert tight["cells_in_flight"] == 3
    assert loose["cells_in_flight"] >= tight["cells_in_flight"]
    # ONE cell alone fits a 2 GiB budget ONLY from K == 1; the aggregate reduces K.
    one = zpar.plan_cell_concurrency(
        [(1800 * MIB, 1800 * MIB)] * 4, 16, 2 * GB,
        shared_bytes=0, cache_worker_px=0,
    )
    assert one["cells_in_flight"] == 1
    # …and an aggregate that no K can hold REFUSES (never a clamp-to-1 rescue).
    with pytest.raises(zpar.ModeInfeasible):
        zpar.plan_cell_concurrency(
            [(3 * GB, 3 * GB)] * 2, 16, 2 * GB,
            shared_bytes=0, cache_worker_px=0,
        )


# ---------------------------------------------------------------------------
# I — R37 preflight and the runtime planner agree on the SAME raw/layout
# ---------------------------------------------------------------------------

def test_i_preflight_and_runtime_agree_on_the_same_raw(tmp_path, monkeypatch):
    import importlib.util
    tool = REPO_ROOT / "tools" / "r37_capture_runner.py"
    spec = importlib.util.spec_from_file_location("r37_capture_runner_r41b", tool)
    runner = importlib.util.module_from_spec(spec)
    sys.modules["r37_capture_runner_r41b"] = runner
    spec.loader.exec_module(runner)

    inp, _out = _write_corpus(tmp_path)
    pin = 4 * GB
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: pin)
    monkeypatch.setattr(zal, "ScientificFloors", _tiny_floors)
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _AreaModel(1024))
    monkeypatch.setattr(zz, "_estimate_streaming_bytes",
                        lambda n, hw, tile=zz.STREAM_TILE_SIZE, channels=3:
                        int(max(1, int(hw[0]) * int(hw[1])) * 512))
    monkeypatch.setattr(zz.zgpu, "probe_gpu_backend",
                        lambda: {"available": False, "reason": "test"})

    paths = [p.name for p in sorted(inp.glob("*.fits"))]
    sel = runner.resolve_selection(inp, paths)
    baseline = SimpleNamespace(nx=2, ny=2, canvas_width=384, canvas_height=384)
    cfg = SimpleNamespace(zegrid_sip_mode="keep")
    pf = runner.preflight(inp, sel, baseline, cfg)

    assert pf.report["available_bytes"] == pin
    assert pf.report["ram_budget_bytes"] == rc.plan_budget_bytes(pin)
    assert pf.report["plan_factor"] == {"num": 4, "den": 5}
    assert pf.report["admit_factor"] == {"num": 9, "den": 10}
    contract = pf.report["ram_contract"]
    assert contract["joint_plan"] is not None
    env = contract["plan_envelope"]
    assert env["raw_available_bytes"] == pin
    assert env["budget_bytes"] == rc.plan_budget_bytes(pin)
    assert env["aggregate_bytes"] <= env["budget_bytes"]

    # The RUNTIME bundle, built on the SAME raw/pinned layout, agrees exactly.
    from zemosaic.zemosaic_stack_plan import FrameInfo

    descs, _rej = zz._build_frame_descriptors(
        [FrameInfo(path=inp / p, order=i) for i, p in enumerate(paths)],
        str(inp), None, sip_mode="keep",
    )
    canvas = zg.build_canvas(descs)
    runtime = _bundle(canvas, descs, raw=pin, budget=rc.plan_budget_bytes(pin),
                      pinned_layout=(baseline.nx, baseline.ny))
    assert list(runtime.envelope.layout) == list(env["layout"])
    assert runtime.envelope.cpu_tile == env["cpu_tile"]
    assert runtime.envelope.cells_in_flight == env["cells_in_flight"]
    assert runtime.envelope.aggregate_bytes == env["aggregate_bytes"]


# ---------------------------------------------------------------------------
# J — nonpositive RAW fails closed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [0, -1, -10 ** 6])
def test_j_nonpositive_raw_ram_fails_closed(tmp_path, bad, monkeypatch):
    inp, out = _write_corpus(tmp_path)
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: bad)
    with pytest.raises(RuntimeError, match="nonpositive"):
        zz.run_zegrid_mode(str(inp), str(out))
    assert not (out / "zegrid_manifest.json").exists()


# ---------------------------------------------------------------------------
# E/F — ordering: the bundle precedes the gauge, admission precedes the auto
# resolution, and the admitted values reach the manifest
# ---------------------------------------------------------------------------

def test_e_f_source_ordering_and_manifest_truth():
    import ast

    src = Path(zz.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "_run_single")
    calls = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            f = node.func
            name = getattr(f, "attr", None) or getattr(f, "id", None)
            if name:
                calls.append((node.lineno, name))
    line = {}
    for ln, name in calls:
        line.setdefault(name, ln)
    assert line["_build_execution_bundle"] < line["compute_global_gauge"], line
    assert line["compute_global_gauge"] < line["_admit"], line
    # R37 capture gate raises AFTER the auto resolution (it consumes the effective
    # mode) and after admission.
    assert line["_admit"] < line["_run_cell_batch"], line


def test_f_admitted_bundle_values_reach_the_executor(tmp_path, monkeypatch):
    inp, out = _write_corpus(tmp_path)
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: 8 * GB)
    monkeypatch.setattr(zal, "ScientificFloors", _tiny_floors)
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _AreaModel(4 * MIB))
    table = {t: 32 * GB for t in zz.CPU_STREAMING_TILE_LADDER}
    table.update({64: 1024, 48: 512, 32: 256})
    monkeypatch.setattr(zz, "_estimate_streaming_bytes",
                        lambda n, hw, tile=zz.STREAM_TILE_SIZE, channels=3:
                        int(table.get(int(tile) if tile is not None else 128, 32 * GB)))
    monkeypatch.setattr(os, "cpu_count", lambda: 3)

    seen = []
    real = zz.run_canonical_stack_streaming

    def spy(provider, request, tile_size=None, fixed=None):
        seen.append(int(tile_size) if tile_size is not None else None)
        return real(provider, request, tile_size=tile_size, fixed=fixed)

    monkeypatch.setattr(zz, "run_canonical_stack_streaming", spy)
    zz.run_zegrid_mode(str(inp), str(out))

    m = json.loads((out / "zegrid_manifest.json").read_text(encoding="utf-8"))
    pcd = m["per_cell_diagnostics"]
    contract = pcd["ram_contract"]
    assert contract["schema"] == "zemosaic-zegrid-r41-ram-contract-v1"
    assert contract["plan_factor"] == {"num": 4, "den": 5}
    assert contract["admit_factor"] == {"num": 9, "den": 10}
    assert contract["plan_observation"]["budget_bytes"] == rc.plan_budget_bytes(8 * GB)
    assert contract["replans"] >= 0
    assert isinstance(contract["tolerance"], bool)
    assert contract["launch_observation"]["factor_num"] == 9
    assert contract["final_envelope"]["aggregate_bytes"] == contract["aggregate_bytes"]
    # The admitted tile is the tile the workers really used.
    tile = int(pcd["cpu_streaming_tile"])
    assert pcd["mode"] == "stream"
    assert seen and all(t == tile for t in seen), (seen, tile)
    # No stale second plan: the manifest values are the admitted envelope's.
    assert pcd["cells_in_flight"] == contract["final_envelope"]["cells_in_flight"]
    assert pcd["cache_workers_per_cell"] == contract["final_envelope"]["cache_workers_per_cell"]
    assert pcd["requested_cell_gauge_mode"] is not None
    # Shared reserve counted ONCE and explicit.
    assert contract["shared_bytes"] == int(zpar._PER_WORKER_BASELINE_BYTES)
    assert contract["stack_total_bytes"] >= contract["shared_bytes"]
    assert contract["cache_total_bytes"] >= contract["shared_bytes"]
    assert contract["aggregate_bytes"] == max(
        contract["stack_total_bytes"], contract["cache_total_bytes"]
    )


# ---------------------------------------------------------------------------
# 4/6 — ladder reselection, non-monotone tiles, pinned recovery by smaller tile
# ---------------------------------------------------------------------------

def test_ladder_reselects_non_monotone_tile_before_refusing(monkeypatch):
    """The ladder is NOT monotone: a smaller tile must never be assumed better, and
    the builder must reselect over the REAL permitted tiles (64 beats 32 here)."""
    descs = _corpus()
    canvas = zg.build_canvas(descs)
    _neutralize_shared_reserve(monkeypatch)
    monkeypatch.setattr(zal, "ScientificFloors", _tiny_floors)
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _FakeModel(10 ** 12))
    table = {128: 9000, 112: 9000, 96: 9000, 80: 9000,
             64: 4000, 48: 9000, 32: 9000}   # NON-monotone: 48/32 are worse than 64
    monkeypatch.setattr(zz, "_estimate_streaming_bytes",
                        lambda n, hw, tile=zz.STREAM_TILE_SIZE, channels=3:
                        int(table.get(int(tile) if tile is not None else 128, 9000)))
    b = _bundle(canvas, descs, raw=5000, budget=4000)
    assert b.envelope.cpu_tile == 64
    assert b.cpu_streaming["selected_tile_size"] == 64
    trials = {int(t["tile"]): t["fits"] for t in b.cpu_streaming["trials"]}
    assert trials[128] is False and trials[64] is True
    # Provenance: the reported bound must be the REAL estimator at the SELECTED
    # tile (integer keys!), non-zero, and never a stale/zero default.
    _cells = [
        (len(m.patch_ids), tuple(p.patch_shape_hw))
        for (_r, _c, _cell, p, m) in b.cell_ctxs if m.patch_ids
    ]
    _real_worst_64 = max(
        zz._estimate_streaming_bytes(n, hw, 64) for n, hw in _cells
    )
    assert _real_worst_64 == 4000
    assert b.cpu_streaming["predicted_peak_bytes"] == _real_worst_64 != 0
    assert b.cpu_streaming["stream_bound_bytes"] == _real_worst_64
    # The standard trial schema is preserved (tile / bound_bytes / fits).
    for tr in b.cpu_streaming["trials"]:
        assert set(("tile", "bound_bytes", "fits")) <= set(tr)
        assert int(tr["bound_bytes"]) == int(table[int(tr["tile"])])


def test_ladder_pinned_cannot_recover_refuses_quantitatively(monkeypatch):
    descs = _corpus()
    canvas = zg.build_canvas(descs)
    _neutralize_shared_reserve(monkeypatch)
    monkeypatch.setattr(zal, "ScientificFloors", _tiny_floors)
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _FakeModel(10 ** 12))
    monkeypatch.setattr(zz, "_estimate_streaming_bytes",
                        lambda n, hw, tile=zz.STREAM_TILE_SIZE, channels=3: 9000)
    with pytest.raises((zal.LayoutInfeasible, zpar.ModeInfeasible)) as exc:
        _bundle(canvas, descs, raw=5000, budget=4000, pinned_layout=(2, 2))
    assert "MiB" in str(exc.value)


# ---------------------------------------------------------------------------
# H — GPU host pre-execution fallback + runtime GPU->CPU fresh 80/90
# ---------------------------------------------------------------------------

def _gpu_on(monkeypatch, *, gpu_tile, vram=4 * GB):
    monkeypatch.setattr(zz.zgpu, "probe_gpu_backend", lambda: {
        "available": True, "device": "FakeGPU", "cupy_version": "fake",
        "vram_total_bytes": 8 * GB, "vram_free_bytes": int(7.5 * GB), "reason": None,
    })
    monkeypatch.setattr(zz.zgpu, "vram_budget_bytes", lambda probe: vram)
    monkeypatch.setattr(zz.zgpu, "choose_gpu_tile_size", lambda *a, **k: gpu_tile)


def test_h_gpu_host_bound_pre_execution_fallback(monkeypatch):
    """A GPU whose OWN tile host bound busts B_plan degrades to exact CPU BEFORE
    execution, inside the SAME bundle, and the CPU fallback is fully planned."""
    descs = _corpus()
    canvas = zg.build_canvas(descs)
    _neutralize_shared_reserve(monkeypatch)
    monkeypatch.setattr(zal, "ScientificFloors", _tiny_floors)
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _FakeModel(10 ** 12))
    table = {128: 4000, 112: 4000, 96: 4000, 80: 4000, 64: 4000, 48: 4000, 32: 4000}
    table[48] = 90000            # the GPU's own tile busts...
    table[32] = 90000            # ...and so does the only smaller GPU ladder tile
    monkeypatch.setattr(zz, "_estimate_streaming_bytes",
                        lambda n, hw, tile=zz.STREAM_TILE_SIZE, channels=3:
                        int(table.get(int(tile) if tile is not None else 128, 90000)))
    _gpu_on(monkeypatch, gpu_tile=48)
    b = _bundle(canvas, descs, raw=5000, budget=4000, gpu_used=True,
                vram_budget_bytes=4 * GB)
    assert b.gpu_used is False
    assert b.envelope.backend == rc.BACKEND_CPU
    assert b.envelope.gpu_tile is None
    assert b.gpu_pre_execution_fallback is not None
    assert b.gpu_pre_execution_fallback["reason"] == "gpu_stream_host_ram_infeasible"
    # The CPU fallback bundle is fully planned (not a stale/second guess).
    assert b.plan["mode"] in ("stream", "inmem")
    assert b.envelope.aggregate_bytes <= 4000


def test_h2_gpu_allowed_when_no_cpu_fallback_fits(monkeypatch):
    """R36 F-M2 (D) restored: a feasible GPU plan is NOT aborted by an infeasible
    CPU fallback — it is recorded, never raised.

    Construction: the REAL shared parent reserve is kept; the layout is admitted
    via a CPU ladder tile that fits B_plan PER-CELL, but the JOINT CPU stack
    aggregate (shared + bound) busts B_plan, while the GPU's own tile fits.
    """
    descs = _corpus()
    canvas = zg.build_canvas(descs)
    _neutralize_workers_only(monkeypatch)
    monkeypatch.setattr(zal, "ScientificFloors", _tiny_floors)
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _FakeModel(10 ** 12))  # inmem fails
    mib = 1024 * 1024
    table = {t: 500 * mib for t in zz.CPU_STREAMING_TILE_LADDER}
    table[40] = 10 * mib           # the GPU's own tile is small
    monkeypatch.setattr(zz, "_estimate_streaming_bytes",
                        lambda n, hw, tile=zz.STREAM_TILE_SIZE, channels=3:
                        int(table.get(int(tile) if tile is not None else 128, 500 * mib)))
    _gpu_on(monkeypatch, gpu_tile=40)
    raw = 1000 * mib
    b = _bundle(canvas, descs, raw=raw, budget=rc.plan_budget_bytes(raw),
                gpu_used=True, vram_budget_bytes=4 * GB)
    assert b.gpu_used is True
    assert b.envelope.backend == rc.BACKEND_GPU
    assert b.cpu_fallback_info["applicable"] is True
    assert b.cpu_fallback_info["feasible"] is False
    assert b.cpu_fallback_plan is None
    assert b.cpu_fallback_tasks is None


# ---------------------------------------------------------------------------
# 6 — behavioural ordering: auto resolves on the FINAL replan memberships
# ---------------------------------------------------------------------------

def test_e_auto_resolves_on_final_memberships_and_gate_follows(tmp_path, monkeypatch):
    inp, out = _write_corpus(tmp_path)
    order = []

    real_bundle = zz._build_execution_bundle
    real_select = zz.zphot.select_auto_gauge_mode

    def bundle_spy(*a, **k):
        b = real_bundle(*a, **k)
        order.append(("bundle", int(b.envelope.refinement_rank)))
        return b

    def select_spy(global_gauge, frame_ids, memberships):
        order.append(("auto", len(list(memberships))))
        return real_select(global_gauge, frame_ids, memberships)

    monkeypatch.setattr(zz, "_build_execution_bundle", bundle_spy)
    monkeypatch.setattr(zz.zphot, "select_auto_gauge_mode", select_spy)

    # RAM DROPS after the plan probe so admission must replan to a FINER layout.
    probes = iter([64 * GB, 64 * GB, 4 * GB, 4 * GB, 4 * GB])
    monkeypatch.setattr(zz, "available_memory_bytes",
                        lambda: next(probes, 4 * GB))
    monkeypatch.setattr(zal, "ScientificFloors", _tiny_floors)
    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _AreaModel(1024))
    monkeypatch.setattr(zz, "_estimate_streaming_bytes",
                        lambda n, hw, tile=zz.STREAM_TILE_SIZE, channels=3:
                        int(max(1, int(hw[0]) * int(hw[1])) * 512))
    monkeypatch.setattr(os, "cpu_count", lambda: 3)

    zz.run_zegrid_mode(str(inp), str(out), cell_gauge_mode="auto")
    m = json.loads((out / "zegrid_manifest.json").read_text(encoding="utf-8"))
    kinds = [k for k, _v in order]
    assert "bundle" in kinds and "auto" in kinds
    final_rank = [v for k, v in order if k == "bundle"][-1]
    auto_cells = [v for k, v in order if k == "auto"][-1]
    # auto saw the FINAL admitted memberships (same cell count), i.e. it ran AFTER
    # the last (re)plan — never against the pre-replan layout.
    assert auto_cells == final_rank
    assert kinds.index("auto") > max(i for i, k in enumerate(kinds) if k == "bundle")
    assert m["per_cell_diagnostics"]["requested_cell_gauge_mode"] == "auto"
    # top-level manifest mirror of the contract
    assert m["ram_contract"] == m["per_cell_diagnostics"]["ram_contract"]
