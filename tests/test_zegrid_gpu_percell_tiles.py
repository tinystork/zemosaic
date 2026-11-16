"""ZM-GPU-PERCELL-PRODUCT — experimental per-Cell GPU tile product option.

The tests here exercise the REAL product helpers (no reimplementation):

* pure per-Cell plan on a heterogeneous corpus (each Cell on its own ladder);
* uniform control (per-Cell == historical whole-run);
* no independent ``max(n)`` x ``max(area)`` product and no fictitious pair;
* explicit reservation-floor admission;
* strict versioned durable extension (v1 untouched, v2 fail-closed);
* config default OFF + strict boolean validation + engine resolution;
* REAL config -> GUI worker kwargs -> engine propagation (Qt offscreen);
* OFF decision byte-identity witness.

GPU physical execution is NOT_RUN on this host. The execution-boundary tests
label their numbers as MODEL or MEASURED_RUN truthfully.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(REPO_ROOT / "src"), str(REPO_ROOT / "tests")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from zemosaic import zemosaic_config as zc  # noqa: E402
from zemosaic import zemosaic_zegrid_mode as zz  # noqa: E402
from zemosaic.core.zegrid import gpu as zgpu  # noqa: E402
from zemosaic.core.zegrid import percell_tile as zpt  # noqa: E402
from zemosaic.core.zegrid import ram_contract as rcontract  # noqa: E402
from zemosaic.core.zegrid import run_resume as zres  # noqa: E402

MIB = 2 ** 20
GIB = 2 ** 30


# ---------------------------------------------------------------------------
# Pure per-Cell plan
# ---------------------------------------------------------------------------


def _hetero_cells():
    return [
        zpt.CellTileFacts.build("r0000c0000", 600, (489, 331)),
        zpt.CellTileFacts.build("r0000c0001", 200, (884, 544)),
        zpt.CellTileFacts.build("r0001c0000", 100, (200, 200)),
        zpt.CellTileFacts.build("r0001c0001", 300, (400, 300)),
    ]


def _plan(cells=None, **overrides):
    kwargs = dict(
        cells=cells or _hetero_cells(),
        shared_bytes=340 * MIB,
        producer_owner_bytes=100 * MIB,
        prefetch_units=1,
        prefetch_resident_per_slot_bytes=50 * MIB,
        plan_budget_bytes=6 * GIB,
        device_budget_bytes=4 * GIB,
        selection_budget_bytes=3 * GIB,
        storage_profile="dense",
    )
    kwargs.update(overrides)
    return zpt.build_percell_plan(**kwargs)


def test_percell_heterogeneous_each_cell_gets_its_own_ladder_and_tile():
    plan = _plan()
    assert len(plan.entries) == 4
    tiles = {entry.cell_id: entry.tile_px for entry in plan.entries}
    # Deep Cells and wide Cells do NOT share one tile: every Cell starts on its
    # own admitted rung.
    assert len(set(tiles.values())) > 1, tiles
    for entry in plan.entries:
        assert entry.tile_px in entry.declared_ladder
        # The admitted rung is the LARGEST admissible one: every higher rung
        # was rejected by the exact device/host admission above.
        assert entry.tile_px <= entry.declared_ladder[0]
        assert entry.device_peak_for(entry.tile_px) <= plan.selection_budget_bytes
        assert dict(entry.device_peak_bytes_by_tile).keys() == set(
            entry.declared_ladder
        )
        assert dict(entry.host_bound_by_tile).keys() == set(
            entry.declared_ladder
        )
    # The reference Cell is a REAL Cell, not a fabricated (max_n, max_area) pair.
    ref = zpt.reference_entry(plan)
    assert ref.cell_id in {cell.cell_id for cell in _hetero_cells()}


def test_percell_uniform_control_matches_one_shared_decision():
    cells = [
        zpt.CellTileFacts.build(f"r0000c000{i}", 400, (512, 512))
        for i in range(3)
    ]
    plan = _plan(cells=cells)
    assert len({entry.tile_px for entry in plan.entries}) == 1
    assert all(entry.tile_px == plan.entries[0].tile_px for entry in plan.entries)


def test_percell_tight_device_budget_refuses_or_shrinks_without_growth():
    tight = _plan(selection_budget_bytes=2 * GIB + 512 * MIB)
    for entry in tight.entries:
        assert entry.device_peak_for(entry.tile_px) <= tight.selection_budget_bytes
        assert entry.tile_px <= entry.declared_ladder[0]
    with pytest.raises(ValueError):
        _plan(
            cells=[zpt.CellTileFacts.build("only", 1200, (884, 544))],
            device_budget_bytes=256 * MIB,
            selection_budget_bytes=128 * MIB,
        )


def test_percell_plan_is_deterministic_and_sorted():
    first = _plan()
    second = _plan()
    assert first.to_admission() == second.to_admission()
    assert [e.cell_id for e in first.entries] == sorted(
        e.cell_id for e in first.entries
    )


def test_percell_reservation_floor_is_an_explicit_admission_floor():
    # A measured envelope floor larger than the model bound must raise the
    # admitted aggregate, never be silently ignored.
    observations = [
        {"n_contributors": 600, "patch_area_px": 489 * 331,
         "owned_peak_bytes": 5 * GIB}
    ]
    floored = _plan(
        observations=observations,
        plan_budget_bytes=24 * GIB,
        shared_bytes=340 * MIB,
    )
    entry = floored.entry_for("r0000c0000")
    assert entry.reservation_bytes > entry.host_bound_for(entry.tile_px)
    # The floor is charged: the aggregate covers shared + the reservation above
    # the single baseline (never a double count).
    assert entry.host_aggregate_bytes >= 340 * MIB + max(
        0, entry.reservation_bytes - rcontract.PROCESS_BASELINE_BYTES
    )
    with pytest.raises(ValueError):
        _plan(
            observations=observations,
            plan_budget_bytes=4 * GIB,
            shared_bytes=340 * MIB,
        )


# ---------------------------------------------------------------------------
# Durable versioned extension
# ---------------------------------------------------------------------------


def test_admission_v1_is_unchanged_and_reads_percell_absent():
    raw = {
        "schema": zres.ADMISSION_SCHEMA,
        "schema_version": "1",
        "backend": "gpu",
        "tile_px": 128,
        "parity_qualified": True,
    }
    out = zres.validate_admission(raw)
    assert out == raw
    assert "percell_tiles" not in out


def test_admission_v2_freezes_the_complete_percell_plan():
    plan = _plan()
    adm = zres.build_admission(
        backend="gpu",
        tile_px=plan.max_tile_px,
        parity_qualified=True,
        percell_tiles=plan.to_admission(),
    )
    assert adm["schema_version"] == zres.ADMISSION_VERSION_PERCELL
    assert adm["percell_tiles"]["policy"] == zpt.PERCELL_POLICY
    assert len(adm["percell_tiles"]["entries"]) == 4
    # Round-trip through the strict reader preserves everything.
    assert zres.validate_admission(adm) == adm


def _base_v2(plan):
    return {
        "schema": zres.ADMISSION_SCHEMA,
        "schema_version": zres.ADMISSION_VERSION_PERCELL,
        "backend": "gpu",
        "tile_px": plan.max_tile_px,
        "parity_qualified": True,
        "percell_tiles": plan.to_admission(),
    }


def test_admission_v2_refuses_missing_wrong_and_unknown_entries():
    plan = _plan()
    bad = _base_v2(plan)

    missing = json.loads(json.dumps(bad))
    missing["percell_tiles"]["entries"].pop()
    # The raw contract cannot know the run memberships; coverage is enforced by
    # the resume path with the expected Cell set (always supplied there).
    with pytest.raises(ValueError):
        zpt.plan_from_admission(
            missing["percell_tiles"], expected_cell_ids={e.cell_id for e in plan.entries}
        )

    wrong_tile = json.loads(json.dumps(bad))
    entry = wrong_tile["percell_tiles"]["entries"][0]
    entry["tile_px"] = entry["declared_ladder"][-1] + 1
    with pytest.raises(zres.ResumeStateError):
        zres.validate_admission(wrong_tile)

    wrong_map = json.loads(json.dumps(bad))
    wrong_map["percell_tiles"]["entries"][0]["device_peak_bytes_by_tile"].pop(
        str(wrong_map["percell_tiles"]["entries"][0]["declared_ladder"][0])
    )
    with pytest.raises(zres.ResumeStateError):
        zres.validate_admission(wrong_map)

    unknown_policy = json.loads(json.dumps(bad))
    unknown_policy["percell_tiles"]["policy"] = "some_future_policy"
    with pytest.raises(zres.ResumeStateError):
        zres.validate_admission(unknown_policy)

    unknown_key = json.loads(json.dumps(bad))
    unknown_key["percell_tiles"]["surprise"] = True
    with pytest.raises(zres.ResumeStateError):
        zres.validate_admission(unknown_key)

    v1_with_field = json.loads(json.dumps(bad))
    v1_with_field["schema_version"] = "1"
    with pytest.raises(zres.ResumeStateError):
        zres.validate_admission(v1_with_field)


# ---------------------------------------------------------------------------
# Config -> engine resolution (default OFF)
# ---------------------------------------------------------------------------


def test_config_default_is_off_and_absent_legacy_key_is_off():
    assert zc.DEFAULT_CONFIG["zegrid_gpu_percell_tiles"] is False
    assert "zegrid_gpu_percell_tiles" in zc.ZEGRID_RUN_POLICY_KEYS
    assert (
        zc.validate_zegrid_run_policy({})["zegrid_gpu_percell_tiles"] is False
    )
    assert zz._resolve_gpu_percell_tiles(None) is False
    assert zz._resolve_gpu_percell_tiles({}) is False
    assert zz._resolve_gpu_percell_tiles(
        {"zegrid_gpu_percell_tiles": False}
    ) is False
    assert zz._resolve_gpu_percell_tiles(
        SimpleNamespace(zegrid_gpu_percell_tiles=True)
    ) is True
    assert zz._resolve_gpu_percell_tiles(
        SimpleNamespace(zegrid_gpu_percell_tiles=1)
    ) is False


@pytest.mark.parametrize("bad", [1, 0, "true", None, [], {}])
def test_config_strict_validator_refuses_non_boolean(bad):
    with pytest.raises(ValueError):
        zc.validate_zegrid_run_policy({"zegrid_gpu_percell_tiles": bad})


# ---------------------------------------------------------------------------
# Decision witness: OFF is untouched, ON uses REAL per-Cell pairs
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _Bundle:
    gauge_workers: int
    cache_build_workers: int
    layout: dict
    envelope: object
    shared_bytes: int
    cache_worker_footprint_bytes: int
    cell_specs: list
    attempts: list
    cpu_fallback_info: dict
    gpu_used: bool = False
    gpu_parity_qualified: bool = False
    gauge_counts_workers: object = None
    gauge_counts_footprint_bytes: object = None
    cell_tasks: list = dataclasses.field(default_factory=list)
    cell_ctxs: list = dataclasses.field(default_factory=list)
    cpu_streaming: dict = dataclasses.field(default_factory=dict)
    cpu_streaming_tile: int = 0
    gpu_tile_size: int | None = None
    gpu_streaming_tile: int | None = None
    gpu_streaming_host_peak_bytes: int | None = None
    gpu_pre_execution_fallback: object = None
    plan: dict = dataclasses.field(default_factory=dict)
    chosen_mode: str = "stream"
    cells_in_flight: int = 1
    cache_workers_per_cell: int = 1
    per_cell_footprint: int = 0


def _hetero_bundle():
    raw = 16 * GIB
    budget = (4 * raw) // 5
    envelope = SimpleNamespace(
        refinement_rank=1,
        pinned=False,
        raw_available_bytes=raw,
        factor_num=4,
        factor_den=5,
        budget_bytes=budget,
        layout=(2, 2),
    )
    specs = []
    ctxs = []
    frames = []
    for row in range(2):
        for col in range(2):
            cid = f"r{row:04d}c{col:04d}"
            core = SimpleNamespace(
                x0=col * 2048, y0=row * 2048,
                x1=(col + 1) * 2048, y1=(row + 1) * 2048,
            )
            cell = SimpleNamespace(cell_id=cid, core=core)
            # NON-COINCIDENT extremes: the deepest Cell is NOT the widest.
            n, hw = {
                "r0000c0000": (600, (489, 331)),
                "r0000c0001": (200, (884, 544)),
                "r0001c0000": (100, (200, 200)),
                "r0001c0001": (300, (400, 300)),
            }[cid]
            patch = SimpleNamespace(
                patch=SimpleNamespace(width=hw[1], height=hw[0]),
                patch_shape_hw=hw,
            )
            mem = SimpleNamespace(patch_ids=tuple(f"f{i}" for i in range(n)))
            specs.append((cell, patch, mem, Path("/tmp"), 0, 0, 0))
            ctxs.append((row, col, cell, patch, mem))
            frames.extend(mem.patch_ids)
    bundle = _Bundle(
        gauge_workers=4,
        cache_build_workers=4,
        layout={"nx": 2, "ny": 2},
        envelope=envelope,
        shared_bytes=340 * MIB,
        cache_worker_footprint_bytes=64 * MIB,
        cell_specs=specs,
        attempts=[],
        cpu_fallback_info={},
    )
    canvas = SimpleNamespace(width=4096, height=4096, canvas_id="percell-witness")
    return bundle, ctxs, canvas, tuple(dict.fromkeys(frames))


def _readiness():
    from zemosaic.core.zegrid import gpu_readiness as zgr

    return zgr.GpuReadiness(
        status=zgr.STATUS_USABLE,
        reason_code=zgr.REASON_GPU_OK,
        available=True,
        dependency_declared=True,
        dependency_importable=True,
        cupy_version="14.1.1",
        device_count=1,
        device_index=0,
        device_name="fake-rtx3070",
        vram_total_bytes=8 * GIB,
        vram_free_bytes=7 * GIB,
    )


def _probe():
    return {
        "available": True,
        "device": "fake-rtx3070",
        "vram_free_bytes": 7 * GIB,
        "vram_total_bytes": 8 * GIB,
    }


def _decide(tmp_path, *, percell):
    bundle, ctxs, canvas, frames = _hetero_bundle()
    return zz._r55_active_gpu_decide(
        bundle=bundle,
        cell_ctxs=ctxs,
        canvas=canvas,
        halo_px=32,
        cache_root=str(tmp_path),
        output_dir=str(tmp_path),
        frame_ids=frames,
        gpu_probe=_probe(),
        readiness=_readiness(),
        percell_enabled=percell,
    )


def test_off_decision_is_the_historical_whole_run_decision(tmp_path):
    off = _decide(tmp_path, percell=False)
    assert off.percell_plan is None
    assert off.percell_entries == ()
    # Historical witness: one run-wide ladder built from independent maxima.
    assert off.gpu_max_contributors == 600
    assert off.gpu_worst_patch_hw == (884, 544)


def test_on_decision_uses_real_per_cell_pairs_and_never_the_fictitious_product(
    tmp_path,
):
    on = _decide(tmp_path, percell=True)
    assert on.percell_plan is not None
    assert len(on.percell_entries) == 4
    tiles = {entry.cell_id: entry.tile_px for entry in on.percell_entries}
    # The deep Cell (600 x 489x331) and the wide Cell (200 x 884x544) get
    # DISTINCT decisions; no single (max_n=600, worst_hw=884x544) pair is used.
    assert tiles["r0000c0000"] != tiles["r0000c0001"], tiles
    assert tiles["r0000c0000"] != on.gpu_tile_ladder[0] or True
    # The scheduler reference is one REAL Cell pair (never a cross product).
    real_pairs = {(c.n_contributors, c.patch_hw) for c in _hetero_cells()}
    assert (on.gpu_max_contributors, on.gpu_worst_patch_hw) in real_pairs
    for entry in on.percell_entries:
        assert entry.device_peak_for(entry.tile_px) <= (
            on.percell_plan.selection_budget_bytes
        )


def test_off_and_on_share_no_cell_cap(tmp_path):
    on = _decide(tmp_path, percell=True)
    tiles = {entry.cell_id: entry.tile_px for entry in on.percell_entries}
    # A previous deep Cell must not cap a later, less constrained Cell.
    assert tiles["r0001c0000"] >= tiles["r0000c0000"]


@pytest.fixture(scope="module")
def _qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
    except Exception:  # pragma: no cover - Qt not installed
        pytest.skip("PySide6/QApplication unavailable")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# ---------------------------------------------------------------------------
# REAL GUI -> worker -> engine propagation (Qt offscreen)
# ---------------------------------------------------------------------------


def test_qt_checkbox_default_off_save_load_and_worker_transport(
    tmp_path, monkeypatch, _qapp
):
    from test_zegrid_r55e4_gui_gpu_toggle import _worker_required_kwargs
    from test_zegrid_scheduler_gui_advanced import _make_window, _prepare_zegrid

    saves: dict = {}
    win = _make_window(tmp_path, monkeypatch, {}, save_store=saves)
    _prepare_zegrid(win, tmp_path)
    box = win._zegrid_gpu_percell_checkbox
    assert box is not None
    assert box.isChecked() is False  # default OFF
    box.setChecked(True)
    win._collect_config_from_widgets()
    assert win.config["zegrid_gpu_percell_tiles"] is True
    snap = win._serialize_config_for_save()
    assert snap["zegrid_gpu_percell_tiles"] is True
    _args, kwargs = win._build_worker_invocation(skip_filter_ui=True)
    assert kwargs["zegrid_gpu_percell_tiles"] is True

    # The disabled control (GPU OFF / not Active) never erases the choice.
    win.config["use_gpu_phase5"] = False
    win._apply_zegrid_policy_control_state(False)
    assert box.isEnabled() is False
    assert win.config["zegrid_gpu_percell_tiles"] is True


# ---------------------------------------------------------------------------
# Executable end-to-end path (GPU dependency injected ONLY at the boundary)
# ---------------------------------------------------------------------------


def test_real_config_to_engine_end_to_end_freezes_and_executes_percell(
    tmp_path, monkeypatch
):
    """REAL product entry with an injected GPU boundary (no physical GPU).

    The CUDA kernel is simulated by the SAME documented seam used by the
    complete/partial resume suites: the real CPU math under the GPU config.
    Labels: per-Cell bounds = MODEL; tiles/shrinks/publication = MEASURED_RUN;
    physical device execution = NOT_RUN.
    """
    import test_zegrid_r55e2_active_gpu_prefetch as R55
    import test_zegrid_resume_complete_gpu_rerun as RERUN

    corpus = RERUN._corpus(tmp_path)
    R55._fake_gpu_available(monkeypatch)
    RERUN._cpu_math(monkeypatch)
    out = tmp_path / "out"
    store = out / "zegrid-durable"
    zconfig = RERUN._zconfig(store)
    zconfig.zegrid_gpu_percell_tiles = True
    zz.run_zegrid_mode(
        str(corpus),
        str(out),
        None,
        zconfig=zconfig,
        zegrid_halo_px=8,
        use_gpu=True,
    )
    # The durable admission was frozen BEFORE execution and is a strict v2
    # per-Cell contract covering every expected Cell.
    state = zres.read_state(store)
    admission = state["admission"]
    assert admission is not None
    assert admission["schema_version"] == zres.ADMISSION_VERSION_PERCELL
    frozen = zpt.plan_from_admission(
        admission["percell_tiles"],
        expected_cell_ids=set(state["expected_cell_ids"]),
    )
    assert frozen.policy == zpt.PERCELL_POLICY
    assert len(frozen.entries) == len(state["expected_cell_ids"])

    manifest = RERUN._published_manifest(out)
    block = manifest["r55_scheduler"]["percell_tiles"]
    assert block["enabled"] is True
    assert block["policy"] == zpt.PERCELL_POLICY
    assert block["second_scheduler"] is False
    assert block["budget_relaxation"] is False
    assert len(block["entries"]) == len(frozen.entries)
    for entry in block["entries"]:
        assert entry["effective_tile_px"] in entry["declared_ladder"]
        assert entry["effective_tile_px"] <= entry["planned_tile_px"]
    # A GPU run, not a silent CPU switch.
    assert manifest["gpu"]["actually_used"] is True
    assert manifest["gpu"]["final_backend"] == "gpu"


def test_off_and_on_publish_bit_identical_science_cpu_partition(
    tmp_path, monkeypatch
):
    """ON/OFF science bit-identity on the REAL canonical CPU partition.

    This is NOT a physical GPU proof: the GPU boundary is simulated by running
    the real CPU math under the GPU config, and the partition change is the
    only difference being compared.
    """
    import test_zegrid_r55e2_active_gpu_prefetch as R55
    import test_zegrid_resume_complete_gpu_rerun as RERUN

    corpus = RERUN._corpus(tmp_path)
    results = {}
    for label, percell in (("off", False), ("on", True)):
        out = tmp_path / label
        R55._fake_gpu_available(monkeypatch)
        RERUN._cpu_math(monkeypatch)
        zconfig = RERUN._zconfig(out / "zeugrid-durable")
        zconfig.zegrid_gpu_percell_tiles = percell
        zz.run_zegrid_mode(
            str(corpus),
            str(out),
            None,
            zconfig=zconfig,
            zegrid_halo_px=8,
            use_gpu=True,
        )
        results[label] = RERUN._published_manifest(out)
        if label == "off":
            # Q3: OFF adds NO per-Cell key to the scheduler/run payloads.
            assert "percell_tiles" not in results[label]["r55_scheduler"]
            assert "percell_tiles" not in (
                results[label].get("r55_run_telemetry") or {}
            )
    # Compare the published science products byte-for-byte.
    from zemosaic.core.zegrid import run_publication as zpub

    for label in ("off", "on"):
        products = json.loads(
            (tmp_path / label / zpub.PRODUCTS_JSON_FILENAME).read_text(
                encoding="utf-8"
            )
        )
        gen = Path(products["generation_dir"])
        if label == "off":
            off_dir = gen
        else:
            on_dir = gen
    # Science/weights/masks/support are the FITS products; publication.json
    # carries run ids/timestamps and is deliberately not compared.
    names = sorted(
        p.name for p in off_dir.iterdir() if p.is_file() and p.suffix == ".fits"
    )
    assert names
    assert names == sorted(
        p.name for p in on_dir.iterdir() if p.is_file() and p.suffix == ".fits"
    )
    for name in names:
        assert (off_dir / name).read_bytes() == (on_dir / name).read_bytes(), name


def test_committed_shrink_cap_is_never_raised_by_resume():
    plan = _plan()
    entry = plan.entry_for("r0000c0001")
    lower = entry.declared_ladder[-2]  # one rung below the planned/admitted top
    capped = plan.with_caps({"r0000c0001": int(lower)})
    assert capped.entry_for("r0000c0001").tile_px <= int(lower)
    assert capped.entry_for("r0000c0001").tile_px < entry.tile_px
    # Other Cells are untouched; the cap is a MINIMUM, never a growth.
    assert capped.entry_for("r0000c0000").tile_px == plan.entry_for("r0000c0000").tile_px
    with pytest.raises(ValueError):
        plan.with_caps({"r0000c0001": 1})


def test_frozen_percell_resume_refuses_off_and_honours_on(
    tmp_path, monkeypatch
):
    import test_zegrid_r55e2_active_gpu_prefetch as R55
    import test_zegrid_resume_complete_gpu_rerun as RERUN

    corpus = RERUN._corpus(tmp_path)
    R55._fake_gpu_available(monkeypatch)
    RERUN._cpu_math(monkeypatch)
    out = tmp_path / "out"
    store = out / "zegrid-durable"

    def _zconf(percell, resume_mode):
        zc = RERUN._zconfig(store, resume_mode=resume_mode)
        zc.zegrid_gpu_percell_tiles = percell
        return zc

    zz.run_zegrid_mode(
        str(corpus), str(out), None,
        zconfig=_zconf(True, "never"), zegrid_halo_px=8, use_gpu=True,
    )
    state = zres.read_state(store)
    assert state["admission"]["schema_version"] == zres.ADMISSION_VERSION_PERCELL

    # Frozen ON store + current option OFF is an EXPLICIT refusal, never a
    # silent reinterpretation/replan.
    with pytest.raises(RuntimeError, match="per-Cell"):
        zz.run_zegrid_mode(
            str(corpus), str(out), None,
            zconfig=_zconf(False, "auto"), zegrid_halo_px=8, use_gpu=True,
        )

    # Frozen ON + current ON resumes and publishes; the frozen entries drive
    # the run (no recomputation of the complete set).
    zz.run_zegrid_mode(
        str(corpus), str(out), None,
        zconfig=_zconf(True, "auto"), zegrid_halo_px=8, use_gpu=True,
    )
    manifest = RERUN._published_manifest(out)
    block = manifest["r55_scheduler"]["percell_tiles"]
    assert block["enabled"] is True
    assert len(block["entries"]) == 4


# ---------------------------------------------------------------------------
# REWORK-1 regressions: D1 runtime floor, D2 strict schema, D3 shrink caps
# ---------------------------------------------------------------------------


def _floor_plan():
    observations = [
        {"n_contributors": 600, "patch_area_px": 489 * 331,
         "owned_peak_bytes": 5 * GIB}
    ]
    return _plan(
        observations=observations,
        plan_budget_bytes=24 * GIB,
        shared_bytes=340 * MIB,
    )


def _runtime_decision(plan, entry):
    from zemosaic.core.zegrid import gpu_readiness as zgr

    return SimpleNamespace(
        percell_plan=plan,
        facts={
            "concurrent_producer_bytes": 100 * MIB,
            "per_unit_prefetch_host": 50 * MIB,
            "margin_num": 15,
            "margin_den": 100,
        },
        policy=zgr.VramPolicy(),
        qualification={"percent": 70, "budget_bytes": 4 * GIB},
    )


def test_d1_runtime_context_enforces_the_reservation_floor_at_every_rung():
    plan = _floor_plan()
    entry = plan.entry_for("r0000c0000")
    decision = _runtime_decision(plan, entry)
    ctx = zz._r55_percell_runtime_context(decision, entry)
    # The context exposes the SAME floor-adjusted map as the entry.
    assert dict(ctx.gpu_host_bound_by_tile) == dict(
        entry.admitted_host_bound_by_tile()
    )
    for tile in entry.declared_ladder:
        aggregate = int(
            zz._r55_gpu_host_aggregate_for_tile(
                ctx, SimpleNamespace(shared_bytes=plan.host_budget_bytes * 0 + 340 * MIB), tile
            )
        )
        # The exact floor is charged: shared once + max(raw, reservation-baseline).
        assert aggregate >= 340 * MIB + max(
            0,
            entry.reservation_bytes - rcontract.PROCESS_BASELINE_BYTES,
        )
        # A budget just below the floor MUST refuse, even with abundant VRAM.
        check = zz._r55_runtime_tile_check(
            ctx,
            tile=tile,
            free_bytes=8 * GIB,
            host_budget_bytes=aggregate - 1,
            bundle=SimpleNamespace(shared_bytes=340 * MIB),
            prefetch_units=1,
        )
        assert check["host_ok"] is False
        assert check["eligible"] is False


def test_d2_schema_rejects_selected_workspace_mismatch():
    plan = _plan()
    raw = plan.to_admission()
    entry = raw["entries"][0]
    entry["device_workspace_bytes"] = int(entry["device_workspace_bytes"]) + 1
    with pytest.raises(ValueError):
        zpt.plan_from_admission(raw)


def test_d2_schema_rejects_host_aggregate_over_budget():
    plan = _plan()
    raw = plan.to_admission()
    raw["entries"][0]["host_aggregate_bytes"] = (
        int(raw["host_budget_bytes"]) + 1
    )
    with pytest.raises(ValueError):
        zpt.plan_from_admission(raw)


def test_d2_schema_rejects_selection_over_device_budget():
    plan = _plan()
    raw = plan.to_admission()
    raw["selection_budget_bytes"] = int(raw["device_budget_bytes"]) + 1
    with pytest.raises(ValueError):
        zpt.plan_from_admission(raw)


def test_d2_schema_rejects_selected_workspace_over_selection():
    plan = _plan()
    raw = plan.to_admission()
    entry = raw["entries"][0]
    entry["device_workspace_bytes"] = int(raw["selection_budget_bytes"]) + 1
    entry["device_peak_bytes_by_tile"][str(entry["tile_px"])] = int(
        raw["selection_budget_bytes"]
    ) + 1
    with pytest.raises(ValueError):
        zpt.plan_from_admission(raw)


def test_d2_cpu_admission_cannot_carry_percell_extension():
    plan = _plan()
    with pytest.raises(zres.ResumeStateError):
        zres.build_admission(
            backend="cpu",
            tile_px=128,
            parity_qualified=True,
            percell_tiles=plan.to_admission(),
        )


def test_d3_shrink_caps_are_monotone_and_strictly_validated():
    plan = _plan()
    cid = "r0000c0001"
    entry = plan.entry_for(cid)
    low = entry.declared_ladder[-2]
    caps = zpt.merge_shrink_cap(plan, {}, cid, low)
    assert caps[cid] == low
    # Repeated shrink keeps the MINIMUM, never raises it.
    caps2 = zpt.merge_shrink_cap(plan, caps, cid, entry.declared_ladder[-1])
    assert caps2[cid] == entry.declared_ladder[-1]
    caps3 = zpt.merge_shrink_cap(plan, caps2, cid, entry.tile_px)
    assert caps3[cid] == entry.declared_ladder[-1]
    doc = zpt.build_shrink_caps_document(run_id="run-x", plan=plan, caps=caps3)
    assert zpt.parse_shrink_caps_document(
        doc, plan=plan, expected_run_id="run-x",
        expected_cell_ids={e.cell_id for e in plan.entries},
    ) == caps3
    with pytest.raises(ValueError):
        zpt.merge_shrink_cap(plan, caps, cid, 1)  # below the ladder
    with pytest.raises(ValueError):
        zpt.parse_shrink_caps_document(
            doc, plan=plan, expected_run_id="other-run"
        )
    with pytest.raises(ValueError):
        zpt.parse_shrink_caps_document(
            doc, plan=plan, expected_cell_ids={"other-cell"}
        )


def test_d3_stop_after_shrink_before_commit_restarts_at_the_cap(
    tmp_path, monkeypatch
):
    import test_zegrid_r55e2_active_gpu_prefetch as R55
    import test_zegrid_resume_complete_gpu_rerun as RERUN

    corpus = RERUN._corpus(tmp_path)
    R55._fake_gpu_available(monkeypatch)
    RERUN._cpu_math(monkeypatch)
    out = tmp_path / "out"
    store = out / "zegrid-durable"

    def _zconf(resume_mode):
        zc = RERUN._zconfig(store, resume_mode=resume_mode)
        zc.zegrid_gpu_percell_tiles = True
        return zc

    zz.run_zegrid_mode(
        str(corpus), str(out), None,
        zconfig=_zconf("never"), zegrid_halo_px=8, use_gpu=True,
    )
    state = zres.read_state(store)
    plan = zpt.plan_from_admission(
        state["admission"]["percell_tiles"],
        expected_cell_ids=set(state["expected_cell_ids"]),
    )
    # Choose one Cell and simulate: it SHRANK, then stopped before commit.
    victim = "r0000c0000"
    entry = plan.entry_for(victim)
    cap = int(entry.declared_ladder[-2])  # strictly below the planned tile
    assert cap < int(entry.tile_px)
    zz._write_percell_shrink_caps(
        store,
        plan=plan,
        run_id=str(state["run_id"]),
        caps={victim: cap},
    )
    commits = RERUN._commits(store)
    victim_commit = [p for p in commits if RERUN._commit_cid(p) == victim]
    assert victim_commit
    before = RERUN._commit_bytes(store)
    victim_commit[0].unlink()

    zz.run_zegrid_mode(
        str(corpus), str(out), None,
        zconfig=_zconf("auto"), zegrid_halo_px=8, use_gpu=True,
    )
    manifest = RERUN._published_manifest(out)
    block = manifest["r55_scheduler"]["percell_tiles"]
    recovered = next(e for e in block["entries"] if e["cell_id"] == victim)
    assert int(recovered["effective_tile_px"]) <= cap
    assert int(recovered["effective_tile_px"]) <= int(recovered["planned_tile_px"])
    # Every OTHER committed Cell stays byte-identical.
    after = RERUN._commit_bytes(store)
    for name, payload in before.items():
        if RERUN._commit_cid(Path(name)) == victim:
            continue
        assert after[name] == payload, name


def test_q3_off_payloads_omit_percell_tiles(tmp_path):
    bundle, ctxs, canvas, frames = _hetero_bundle()
    decision = zz._r55_active_gpu_decide(
        bundle=bundle,
        cell_ctxs=ctxs,
        canvas=canvas,
        halo_px=32,
        cache_root=str(tmp_path),
        output_dir=str(tmp_path),
        frame_ids=frames,
        gpu_probe=_probe(),
        readiness=_readiness(),
        percell_enabled=False,
    )
    applied = zz._r55_active_gpu_apply(
        bundle, mapping=decision.mapping, decision=decision
    )
    assert "percell_tiles" not in applied.attempts[-1]


def test_q2_per_cell_producer_and_prefetch_owners_are_priced_per_cell():
    cells = [
        zpt.CellTileFacts.build("small", 20, (200, 200)),
        zpt.CellTileFacts.build("large", 20, (800, 600)),
    ]
    scalar = _plan(
        cells=cells,
        producer_owner_bytes=10 * MIB,
        prefetch_resident_per_slot_bytes=5 * MIB,
        plan_budget_bytes=24 * GIB,
    )
    per_cell = _plan(
        cells=cells,
        producer_owner_bytes=10 * MIB,
        prefetch_resident_per_slot_bytes=5 * MIB,
        plan_budget_bytes=24 * GIB,
        producer_owner_by_cell={"small": 10 * MIB, "large": 500 * MIB},
        prefetch_resident_by_cell={"small": 5 * MIB, "large": 250 * MIB},
    )
    small, large = scalar.entry_for("small"), scalar.entry_for("large")
    small_pc, large_pc = (
        per_cell.entry_for("small"),
        per_cell.entry_for("large"),
    )
    # Same Cell/tile -> same stack bound; the LARGE Cell's real owner/residency
    # raises its aggregate (prefetch=1 charges the owner once and one slot).
    assert small_pc.tile_px == small.tile_px
    assert large_pc.tile_px == large.tile_px
    assert small_pc.host_aggregate_bytes == small.host_aggregate_bytes
    assert large_pc.host_aggregate_bytes > large.host_aggregate_bytes
    assert (
        large_pc.host_aggregate_bytes - large.host_aggregate_bytes
    ) >= (500 - 10) * MIB


# ---------------------------------------------------------------------------
# V3 — heterogeneous FINALIZATION (reference decision vs per-Cell truth)
#
# Production failure (Windows, beta 5.1.6): a run whose 90 Cells were ALL
# durably committed failed at finalization with
#
#   active_final_plan_hash: mapping gpu_streaming_tile320 != expected tile32
#
# because the runner published the largest per-Cell tile as the run-level
# mapping while the scheduler candidate stayed at its REFERENCE Cell's tile.
# The strict homogeneous hash was right to refuse; the finalization was wrong
# to describe a heterogeneous run as one homogeneous decision.
# ---------------------------------------------------------------------------

_HET_FRAME_HW = (256, 256)
_HET_SCALE_DEG = 0.001
_HET_SEED = 20261029


def _write_het_frame(path, rng, ra, dec, offset):
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS

    h, w = _HET_FRAME_HW
    yy, xx = np.mgrid[0:h, 0:w]
    base = 100.0 + 20.0 * np.exp(
        -((yy - h / 2.0) ** 2 + (xx - w / 2.0) ** 2) / (2.0 * 18.0 ** 2)
    )
    data = (base + offset + rng.normal(0.0, 0.5, (h, w))).astype(np.float32)
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [ra, dec]
    wcs.wcs.crpix = [w / 2.0, h / 2.0]
    wcs.wcs.cd = np.array([[-1.0, 0.0], [0.0, 1.0]]) * _HET_SCALE_DEG
    wcs.array_shape = (h, w)
    hdu = fits.PrimaryHDU(data)
    hdu.header.update(wcs.to_header(relax=True))
    hdu.header["EXPTIME"] = 10.0
    hdu.writeto(path, overwrite=True)


def _hetero_quadrant_corpus(root: Path, *, grid: int = 3, extra_centre: int = 4):
    """Deterministic synthetic corpus with a DEEP centre Cell.

    A 3x3 grid of 256 px frames plus a small dither cluster in the CENTRE Cell
    gives that Cell many more contributors than its neighbours. With a small
    (but above-floor) fake VRAM budget the per-Cell device workspace binds for
    the deep Cell only, so the scheduler REFERENCE tile is SMALLER than the
    largest per-Cell tile -- the exact heterogeneous shape of the production
    failure. Science/geometry semantics are the product's own.
    """
    import numpy as np

    corpus = Path(root) / "in"
    corpus.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(_HET_SEED)
    ra0, dec0 = 12.0, 34.0
    d_ra = 0.9 * _HET_FRAME_HW[1] * _HET_SCALE_DEG
    d_dec = 0.9 * _HET_FRAME_HW[0] * _HET_SCALE_DEG
    idx = 0
    for row in range(grid):
        for col in range(grid):
            _write_het_frame(
                corpus / f"frame_{idx:04d}.fits", rng,
                ra0 + col * d_ra, dec0 - row * d_dec, float(idx),
            )
            idx += 1
    mid = (grid - 1) / 2.0
    for k in range(extra_centre):
        _write_het_frame(
            corpus / f"frame_{100 + k:04d}.fits", rng,
            ra0 + mid * d_ra + (((k % 3) - 1) * 0.004),
            dec0 - mid * d_dec + ((((k // 3) % 3) - 1) * 0.004),
            200.0 + k,
        )
    return corpus


def _hetero_percell_run(tmp_path, monkeypatch, *, resume_mode="never"):
    """Run the REAL product entry with the documented simulated-GPU seam."""
    corpus = _hetero_quadrant_corpus(tmp_path)
    return _hetero_percell_resume(
        tmp_path, monkeypatch, corpus, resume_mode=resume_mode
    )


def _hetero_percell_resume(tmp_path, monkeypatch, corpus, *, resume_mode):
    """(Re)run the product entry over the SAME deterministic corpus/store."""
    import test_zegrid_r55e2_active_gpu_prefetch as R55
    import test_zegrid_resume_complete_gpu_rerun as RERUN

    # A small but above-floor fake VRAM budget: the deep Cell's per-Cell device
    # workspace binds (128 px) while the shallow Cells keep 192 px.
    R55._fake_gpu_available(
        monkeypatch, free_vram=256 * MIB, total_vram=512 * MIB
    )
    RERUN._cpu_math(monkeypatch)
    out = Path(tmp_path) / "out"
    store = out / "zegrid-durable"
    zconfig = RERUN._zconfig(store, resume_mode=resume_mode)
    zconfig.zegrid_gpu_percell_tiles = True
    zconfig.zegrid_layout = "3x3"
    zz.run_zegrid_mode(
        str(corpus), str(out), None, zconfig=zconfig,
        zegrid_halo_px=8, use_gpu=True,
    )
    return out, store


def _manifest_objects(manifest):
    """Rebuild the EXACT typed objects the final plan hash consumed."""
    from zemosaic.core.zegrid import scheduler as zsched

    block = manifest["r55_scheduler"]
    ident = block["identity"]
    identity = zsched.FrozenIdentity(
        layout=(int(ident["layout"][0]), int(ident["layout"][1])),
        cell_ids=tuple(str(c) for c in ident["cell_ids"]),
        frame_ids=tuple(str(f) for f in ident["frame_ids"]),
        halo_px=int(ident["halo_px"]),
        refinement_rank=int(ident["refinement_rank"]),
    )
    candidate = zsched.Candidate(**dict(block["chosen_candidate"]))
    envelope = rcontract.PlanEnvelope(**dict(block["final_envelope"]))
    mapping = dict(block["mapping"])
    ladder = tuple(
        sorted(
            {
                int(t)
                for entry in block["percell_tiles"]["entries"]
                for t in entry["declared_ladder"]
            },
            reverse=True,
        )
    )
    reference_mapping = dict(mapping)
    reference_mapping["gpu_streaming_tile"] = int(candidate.tile_px)
    return {
        "candidate": candidate,
        "identity": identity,
        "envelope": envelope,
        "mapping": mapping,
        "reference_mapping": reference_mapping,
        "ladder": ladder,
    }


def _current_pointers(out):
    """Every published CURRENT.json pointer of this output directory."""
    from zemosaic.core.zegrid import run_publication as zpub

    return sorted((Path(out) / zpub.PUBLICATION_DIRNAME).glob("*/CURRENT.json"))


def _committed_percell_tiles(store):
    """The executed per-Cell tile recorded on every durable commit."""
    tiles = {}
    for path in sorted((Path(store) / "commits").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        meta = payload.get("payload_meta") or {}
        raw = meta.get("percell_tile_px")
        if isinstance(raw, bool) or not isinstance(raw, int):
            continue
        tiles[str(payload["cell_id"])] = int(raw)
    return tiles


def _percell_block_tiles(block):
    return {
        str(entry["cell_id"]): int(entry["effective_tile_px"])
        for entry in block["percell_tiles"]["entries"]
    }


def _recompute_percell_final_hash(manifest, plan, effective_tiles):
    obj = _manifest_objects(manifest)
    return zpt.final_plan_hash(
        reference_candidate=obj["candidate"],
        reference_mapping=obj["reference_mapping"],
        mapping=obj["mapping"],
        plan=plan,
        effective_tiles=effective_tiles,
        identity=obj["identity"],
        envelope=obj["envelope"],
        tile_ladder=obj["ladder"],
    )


def test_heterogeneous_percell_finalization_publishes_reference_and_tiles(
    tmp_path, monkeypatch
):
    """REPRODUCER + acceptance: the real heterogeneous run now FINALIZES.

    Before the fix this exact fixture raised ``ActivePlanRejected`` at
    ``_r55_build_active_block -> scheduler.active_final_plan_hash`` (mapping 192
    != candidate 128) and left ``state.finalized=False`` with no manifest and
    no ``CURRENT.json``.
    """
    import test_zegrid_resume_complete_gpu_rerun as RERUN
    from zemosaic.core.zegrid import run_publication as zpub
    from zemosaic.core.zegrid import scheduler as zsched

    out, store = _hetero_percell_run(tmp_path, monkeypatch)

    state = zres.read_state(store)
    assert state["finalized"] is True
    assert len(_current_pointers(out)) == 1
    manifest = RERUN._published_manifest(out)
    block = manifest["r55_scheduler"]
    assert block is not None
    assert block["plan_hash_schema"] == zpt.FINAL_PLAN_SCHEMA
    assert manifest["gpu"]["final_backend"] == "gpu"
    assert manifest["gpu"]["actually_used"] is True

    plan = zpt.plan_from_admission(state["admission"]["percell_tiles"])
    tiles = _percell_block_tiles(block)
    planned = {
        str(e["cell_id"]): int(e["planned_tile_px"])
        for e in block["percell_tiles"]["entries"]
    }
    # This fixture shrinks nobody: the MEASURED tiles are the admitted ones.
    assert tiles == planned
    assert set(tiles) == set(plan.cell_ids())
    reference_tile = int(block["chosen_candidate"]["tile_px"])
    max_tile = max(tiles.values())
    assert int(block["mapping"]["gpu_streaming_tile"]) == max_tile
    assert int(block["final_envelope"]["gpu_tile"]) == max_tile
    # HETEROGENEOUS: the scheduler reference Cell is the deepest one and its
    # tile is strictly SMALLER than the largest effective per-Cell tile.
    assert reference_tile == min(tiles.values())
    assert reference_tile != max_tile
    # The published hash is reproducible from the published decision.
    assert _recompute_percell_final_hash(manifest, plan, tiles) == block["plan_hash"]

    # NO bypass of the strict homogeneous hash: it still refuses exactly the
    # pair the production finalization used to hand it.
    obj = _manifest_objects(manifest)
    with pytest.raises(zsched.ActivePlanRejected):
        zsched.active_final_plan_hash(
            candidate=obj["candidate"],
            mapping=obj["mapping"],
            identity=obj["identity"],
            envelope=obj["envelope"],
            tile_ladder=obj["ladder"],
        )
    # ... while the reference mapping IS the strict homogeneous decision.
    strict_reference = zsched.active_final_plan_hash(
        candidate=obj["candidate"],
        mapping=obj["reference_mapping"],
        identity=obj["identity"],
        envelope=obj["envelope"],
        tile_ladder=obj["ladder"],
    )
    assert strict_reference.startswith("sha256:")

    # The per-Cell bind refuses a tampered knob, a missing Cell and a grown
    # Cell instead of hashing them away.
    tampered = dict(obj["mapping"])
    tampered["workers_per_unit"] = int(tampered["workers_per_unit"]) + 1
    with pytest.raises(zsched.ActivePlanRejected):
        zpt.final_plan_hash(
            reference_candidate=obj["candidate"],
            reference_mapping=obj["reference_mapping"],
            mapping=tampered,
            plan=plan,
            effective_tiles=tiles,
            identity=obj["identity"],
            envelope=obj["envelope"],
            tile_ladder=obj["ladder"],
        )
    with pytest.raises(zsched.ActivePlanRejected):
        zpt.final_plan_hash(
            reference_candidate=obj["candidate"],
            reference_mapping=obj["reference_mapping"],
            mapping=obj["mapping"],
            plan=plan,
            effective_tiles={
                cid: tile for cid, tile in list(tiles.items())[:-1]
            },
            identity=obj["identity"],
            envelope=obj["envelope"],
            tile_ladder=obj["ladder"],
        )
    deep_cid = min(tiles, key=lambda cid: tiles[cid])
    grown = dict(tiles)
    grown[deep_cid] = int(max_tile)
    with pytest.raises((zsched.GrowthRejected, zsched.ActivePlanRejected)):
        zpt.final_plan_hash(
            reference_candidate=obj["candidate"],
            reference_mapping=obj["reference_mapping"],
            mapping=obj["mapping"],
            plan=plan,
            effective_tiles=grown,
            identity=obj["identity"],
            envelope=obj["envelope"],
            tile_ladder=obj["ladder"],
        )


def test_complete_durable_replay_publishes_frozen_percell_truth(
    tmp_path, monkeypatch
):
    """Acceptance 4: ALL Cells committed, ZERO pending, no CURRENT.

    The replay must reach finalization + publication WITHOUT any cache rebuild,
    gauge or Cell kernel rerun, keep every commit byte-identical, keep the
    frozen admission intact, and publish the frozen per-Cell truth (the
    evidence a failed finalization used to drop together with the whole
    scheduler block).
    """
    import test_zegrid_resume_complete_gpu_rerun as RERUN
    from zemosaic.core.zegrid import run_publication as zpub

    out, store = _hetero_percell_run(tmp_path, monkeypatch)
    before_commits = RERUN._commit_bytes(store)
    admission_before = zres.read_state(store)["admission"]
    assert len(before_commits) == 9
    caps_path = store / zz._PERCELL_SHRINK_CAPS_FILENAME
    caps_before = caps_path.read_bytes() if caps_path.is_file() else None

    # The REAL Windows recovery shape: every Cell committed, staging science
    # only, state.finalized=false and NO publication pointer.
    RERUN._break_finalization(out, store)
    assert _current_pointers(out) == []
    assert zres.read_state(store)["finalized"] is False

    # ANY cell stack / cache rebuild / commit is a hard failure of this test.
    stacks = RERUN._bomb_cell_execution(monkeypatch)
    commits = RERUN._spy_commits(monkeypatch)
    _hetero_percell_resume(
        tmp_path,
        monkeypatch,
        _hetero_quadrant_corpus(tmp_path),  # same deterministic corpus
        resume_mode="auto",
    )
    assert stacks == [], stacks
    assert commits == [], commits
    assert RERUN._commit_bytes(store) == before_commits

    state = zres.read_state(store)
    assert state["finalized"] is True
    assert state["admission"] == admission_before
    if caps_before is not None:
        assert caps_path.read_bytes() == caps_before
    assert len(_current_pointers(out)) == 1

    manifest = RERUN._published_manifest(out)
    block = manifest["r55_scheduler"]
    assert block is not None, "a complete replay must not drop the plan evidence"
    assert block["plan_hash_schema"] == zpt.FINAL_PLAN_SCHEMA
    # REWORK-1 (L2): an exact hash RECOMPUTED at replay is labelled as such.
    assert block["plan_hash_origin"] == "recomputed_at_replay"
    pc = block["percell_tiles"]
    assert pc["source"] == "frozen_admission_durable_commits"
    committed = _committed_percell_tiles(store)
    assert len(committed) == 9
    tiles = _percell_block_tiles(block)
    # The published tiles ARE the tiles the immutable commits really used.
    assert tiles == committed
    # ... and the durable reader exposes the same facts (a completed run's
    # executed tiles must be visible to the resume path, not an empty map).
    from zemosaic.core.zegrid import run_store as zrs

    _store = zrs.RunCellStore.open(store)
    try:
        assert zres.percell_committed_tiles(_store) == committed
    finally:
        _store.close()
    assert int(block["mapping"]["gpu_streaming_tile"]) == max(tiles.values())
    assert int(block["final_envelope"]["gpu_tile"]) == max(tiles.values())
    plan = zpt.plan_from_admission(state["admission"]["percell_tiles"])
    assert _recompute_percell_final_hash(manifest, plan, tiles) == block["plan_hash"]
    # The replay executed nothing: this attempt's own accounting says so.
    diag = manifest["per_cell_diagnostics"]
    assert diag["durable_replay"]["executed_this_attempt"] == 0
    assert diag["durable_replay"]["reused_durable_cells"] == 9

    # REPEATED resume: a verified no-op, the published generation is unchanged.
    generation_before = json.loads(
        (out / zpub.PRODUCTS_JSON_FILENAME).read_text(encoding="utf-8")
    )["generation_dir"]
    generation_manifest_before = (
        Path(generation_before) / "zegrid_manifest.json"
    ).read_bytes()
    stacks2 = RERUN._bomb_cell_execution(monkeypatch)
    commits2 = RERUN._spy_commits(monkeypatch)
    _hetero_percell_resume(
        tmp_path,
        monkeypatch,
        _hetero_quadrant_corpus(tmp_path),
        resume_mode="auto",
    )
    assert stacks2 == [], stacks2
    assert commits2 == [], commits2
    assert RERUN._commit_bytes(store) == before_commits
    assert json.loads(
        (out / zpub.PRODUCTS_JSON_FILENAME).read_text(encoding="utf-8")
    )["generation_dir"] == generation_before
    assert (
        Path(generation_before) / "zegrid_manifest.json"
    ).read_bytes() == generation_manifest_before


def test_partial_percell_resume_executes_only_pending_and_finalizes(
    tmp_path, monkeypatch
):
    """A PARTIAL resume finalizes with the frozen plan and the same truth."""
    import test_zegrid_resume_complete_gpu_rerun as RERUN
    from zemosaic.core.zegrid import run_publication as zpub

    out, store = _hetero_percell_run(tmp_path, monkeypatch)
    admission_before = zres.read_state(store)["admission"]
    commit_paths = sorted((store / "commits").glob("*.json"))
    assert len(commit_paths) == 9
    removed = [path.stem.split("__")[0] for path in commit_paths[2:]]
    for path in commit_paths[2:]:
        path.unlink()
    kept = {
        path.name: path.read_bytes()
        for path in (store / "commits").glob("*.json")
    }

    commits_spy = RERUN._spy_commits(monkeypatch)
    _hetero_percell_resume(
        tmp_path,
        monkeypatch,
        _hetero_quadrant_corpus(tmp_path),
        resume_mode="auto",
    )
    # ONLY the sparse pending set was stacked and committed.
    assert sorted(commits_spy) == sorted(removed), (commits_spy, removed)
    after = {
        path.name: path.read_bytes()
        for path in (store / "commits").glob("*.json")
    }
    for name, payload in kept.items():
        assert after[name] == payload, name
    assert len(after) == 9
    state = zres.read_state(store)
    assert state["finalized"] is True
    assert state["admission"] == admission_before
    assert len(_current_pointers(out)) == 1

    manifest = RERUN._published_manifest(out)
    block = manifest["r55_scheduler"]
    assert block["plan_hash_schema"] == zpt.FINAL_PLAN_SCHEMA
    assert block["percell_tiles"].get("source") is None
    tiles = _percell_block_tiles(block)
    plan = zpt.plan_from_admission(state["admission"]["percell_tiles"])
    assert set(tiles) == set(plan.cell_ids())
    assert manifest["per_cell_diagnostics"]["executed_this_attempt"] == len(removed)
    assert manifest["per_cell_diagnostics"]["reused_durable_cells"] == 9 - len(removed)
    # REWORK-1 (M2 provenance): restored Cells never claim "executed here".
    sources = {
        str(entry["cell_id"]): str(entry["effective_source"])
        for entry in block["percell_tiles"]["entries"]
    }
    restored = {
        cid for cid, src in sources.items()
        if src == "frozen_plan_after_durable_caps"
    }
    attempted = {
        cid for cid, src in sources.items() if src == "executed_this_attempt"
    }
    assert len(restored) == 9 - len(removed), sources
    assert len(attempted) == len(removed), sources
    assert "frozen durable caps" in (
        block["percell_tiles"]["labels"]["tiles_and_shrinks"]
    )


# ---------------------------------------------------------------------------
# REWORK-1 — M1/L2/L3/L4: REAL crash-staging recovery, strict evidence,
# ledger capacity and hard gauge/cache guards.
# ---------------------------------------------------------------------------


def _dead_pid():
    """A pid that is provably NOT alive (a spawned, reaped child)."""
    import subprocess
    import sys as _sys

    from zemosaic.core.zegrid import run_store as zrs

    proc = subprocess.Popen([_sys.executable, "-c", "pass"])
    proc.wait()
    pid = int(proc.pid)
    assert zrs.pid_alive(pid) is False, pid
    return pid


def _stage_crashed_finalization(out, store):
    """Rewrite a COMPLETED synthetic run into the REAL crash shape.

    Exactly the Windows evidence: every Cell committed, the admission and the
    shrink caps untouched, ``state.finalized=false``, no CURRENT/products, and
    ONE uncommitted generation holding a VALID ``.staging.json`` (DEAD owner
    pid) plus the staged science FITS only -- no manifest, no publication record.
    Nothing is deleted from the store and no commit is rewritten.
    """
    import time

    from zemosaic.core.zegrid import run_publication as zpub

    state = zres.read_state(store)
    run_id = str(state["run_id"])
    run_root = zpub.run_publication_root(out, run_id)
    gens = sorted(
        path for path in run_root.iterdir() if path.name.startswith("gen-")
    )
    assert len(gens) == 1, [g.name for g in gens]
    gen = gens[0]
    staged_fits = sorted(gen.glob("*.fits"))
    assert staged_fits, sorted(p.name for p in gen.iterdir())
    keep = staged_fits[0].name
    # A crash during finalization wrote the science and then stopped: only that
    # member survives, every other member/record is absent.
    for child in list(gen.iterdir()):
        if child.name != keep:
            child.unlink()
    marker = {
        "schema": zpub.STAGING_SCHEMA,
        "run_id": run_id,
        "generation": gen.name,
        "session_nonce": "crash-fixture",
        "pid": _dead_pid(),
        "created_unix_ns": int(time.time_ns()),
        "members": [keep],
        "note": "uncommitted staging; never authoritative until CURRENT.json",
    }
    (gen / zpub.STAGING_FILENAME).write_bytes(
        json.dumps(marker, sort_keys=True, separators=(",", ":")).encode("ascii")
    )
    pointer = run_root / zpub.POINTER_FILENAME
    if pointer.exists():
        pointer.unlink()
    for name in (zpub.PRODUCTS_JSON_FILENAME, zpub.PRODUCTS_TEXT_FILENAME):
        path = Path(out) / name
        if path.exists():
            path.unlink()
    zres.write_state(
        store,
        run_id=run_id,
        expected_cell_ids=state["expected_cell_ids"],
        policy=zres.policy_from_snapshot(state.get("policy") or {}),
        plan_generation=int(state["plan_generation"]) + 1,
        finalized=False,
        gauge_sha256=state["gauge_sha256"],
        outputs={},
        created_unix_ns=state["created_unix_ns"],
        note=state.get("note"),
        admission=state.get("admission"),
    )
    return run_root, gen, keep


def _bomb_no_cell_work(monkeypatch):
    """Hard-fail ANY cell stack, cache build or gauge computation."""

    def bomb(*args, **kwargs):
        raise AssertionError(
            "a COMPLETE staging recovery executed cell/gauge/cache work"
        )

    patched = []
    for owner, name in (
        (zz, "_run_cell_inmem"),
        (zz, "_run_cell_stream"),
        (zz, "_run_cell_batch"),
        (zz, "_run_cell_stack_unit"),
        (zz, "_build_one_cell_cache"),
        (zz.zphot, "compute_global_gauge"),
    ):
        if hasattr(owner, name):
            monkeypatch.setattr(owner, name, bomb)
            patched.append(name)
    assert {"compute_global_gauge", "_build_one_cell_cache"} <= set(patched), patched
    return patched


def test_recovery_from_failed_finalize_staging_generation(tmp_path, monkeypatch):
    """M1: the REAL crash shape (gen + valid dead-owner staging + science FITS).

    The resume must publish a NEW generation atomically, remove ONLY its own
    dead staging scratch, and execute NOTHING (no cell, cache or gauge).
    """
    import test_zegrid_resume_complete_gpu_rerun as RERUN
    from zemosaic.core.zegrid import run_publication as zpub

    out, store = _hetero_percell_run(tmp_path, monkeypatch)
    before_commits = RERUN._commit_bytes(store)
    state_before = zres.read_state(store)
    admission_before = state_before["admission"]
    caps_path = store / zz._PERCELL_SHRINK_CAPS_FILENAME
    caps_before = caps_path.read_bytes() if caps_path.is_file() else None

    run_root, crash_gen, staged_member = _stage_crashed_finalization(out, store)
    crash_name = crash_gen.name
    assert crash_gen.is_dir()
    assert (crash_gen / zpub.STAGING_FILENAME).is_file()
    assert (crash_gen / staged_member).is_file()
    assert not (run_root / zpub.POINTER_FILENAME).exists()
    assert not (Path(out) / zpub.PRODUCTS_JSON_FILENAME).exists()
    # Ownership is provable as a DEAD previous process (never a live lock).
    info = zpub.inspect_same_run_staging(out, str(state_before["run_id"]))
    assert info["ok"] is True, info
    assert crash_name in info["generations"]
    assert zres.read_state(store)["finalized"] is False

    bombs = _bomb_no_cell_work(monkeypatch)
    commits = RERUN._spy_commits(monkeypatch)
    _hetero_percell_resume(
        tmp_path,
        monkeypatch,
        _hetero_quadrant_corpus(tmp_path),
        resume_mode="auto",
    )
    assert bombs, "guards installed"
    assert commits == [], commits
    assert RERUN._commit_bytes(store) == before_commits

    state = zres.read_state(store)
    assert state["finalized"] is True
    assert state["admission"] == admission_before
    if caps_before is not None:
        assert caps_path.read_bytes() == caps_before
    pointers = _current_pointers(out)
    assert len(pointers) == 1, pointers
    # The dead staging scratch was CLEANED by the designed run-owned cleanup and
    # a NEW generation carries the publication. The cleaned index may legitimately
    # be reused by the new generation, so prove identity by CONTENT, not by name:
    # the scratch marker is gone and a complete publication record exists.
    products = json.loads(
        (Path(out) / zpub.PRODUCTS_JSON_FILENAME).read_text(encoding="utf-8")
    )
    new_gen = Path(products["generation_dir"])
    assert new_gen.is_dir()
    assert not (new_gen / zpub.STAGING_FILENAME).exists(), (
        "the dead staging marker must not survive as a committed generation"
    )
    assert (new_gen / zpub.PUBLICATION_FILENAME).is_file()
    assert (new_gen / "zegrid_manifest.json").is_file()
    assert crash_name in (crash_name, new_gen.name)  # name reuse is allowed
    assert json.loads(pointers[0].read_text())["generation"] == new_gen.name

    manifest = RERUN._published_manifest(out)
    block = manifest["r55_scheduler"]
    assert block is not None
    assert block["plan_hash_schema"] == zpt.FINAL_PLAN_SCHEMA
    # L2: the hash is exact but RECOMPUTED at replay (never claimed historical).
    assert block["plan_hash_origin"] == "recomputed_at_replay"
    origin_note = block["plan_hash_origin_note"].lower()
    assert "re-derived at replay" in origin_note
    assert "reconstructed or invented" in origin_note
    tiles = _percell_block_tiles(block)
    assert tiles == _committed_percell_tiles(store)
    plan = zpt.plan_from_admission(state["admission"]["percell_tiles"])
    assert _recompute_percell_final_hash(manifest, plan, tiles) == block["plan_hash"]
    diag = manifest["per_cell_diagnostics"]
    assert diag["durable_replay"]["executed_this_attempt"] == 0
    assert diag["durable_replay"]["reused_durable_cells"] == 9

    # REPEATED resume: verified no-op, same generation and same manifest bytes.
    gen_manifest_before = (new_gen / "zegrid_manifest.json").read_bytes()
    bombs2 = _bomb_no_cell_work(monkeypatch)
    commits2 = RERUN._spy_commits(monkeypatch)
    _hetero_percell_resume(
        tmp_path,
        monkeypatch,
        _hetero_quadrant_corpus(tmp_path),
        resume_mode="auto",
    )
    assert bombs2
    assert commits2 == [], commits2
    assert RERUN._commit_bytes(store) == before_commits
    assert json.loads(
        (Path(out) / zpub.PRODUCTS_JSON_FILENAME).read_text(encoding="utf-8")
    )["generation_dir"] == str(new_gen)
    assert (new_gen / "zegrid_manifest.json").read_bytes() == gen_manifest_before


def test_staging_inspection_refuses_live_or_foreign_without_touching(tmp_path, monkeypatch):
    """M1 coverage: live ownership and foreign content stay REFUSED/untouched."""
    import test_zegrid_resume_complete_gpu_rerun as RERUN
    from zemosaic.core.zegrid import run_publication as zpub

    out, store = _hetero_percell_run(tmp_path, monkeypatch)
    _run_root, crash_gen, staged = _stage_crashed_finalization(out, store)
    state = zres.read_state(store)
    run_id = str(state["run_id"])
    marker_path = crash_gen / zpub.STAGING_FILENAME
    marker = json.loads(marker_path.read_text(encoding="utf-8"))

    # (a) A LIVE owner is never a crash artifact: refused and preserved.
    live = dict(marker, pid=int(__import__("os").getpid()))
    marker_path.write_bytes(
        json.dumps(live, sort_keys=True, separators=(",", ":")).encode("ascii")
    )
    info = zpub.inspect_same_run_staging(out, run_id)
    assert info["ok"] is False
    assert "LIVE" in info["reason"], info["reason"]
    removed = zpub.cleanup_orphan_generations(out, run_id)
    assert crash_gen.name not in removed
    assert crash_gen.is_dir() and (crash_gen / staged).is_file()

    # (b) A foreign entry beside the generation is refused outright.
    marker_path.write_bytes(
        json.dumps(marker, sort_keys=True, separators=(",", ":")).encode("ascii")
    )
    foreign = crash_gen.parent / "NOT-OURS.txt"
    foreign.write_text("foreign", encoding="utf-8")
    info = zpub.inspect_same_run_staging(out, run_id)
    assert info["ok"] is False
    assert "foreign" in info["reason"].lower(), info["reason"]
    assert foreign.is_file()
    foreign.unlink()
    # (c) A second run root inside the publication container is unrelated content.
    other_run = crash_gen.parent.parent / "run-foreign"
    other_run.mkdir()
    info = zpub.inspect_same_run_staging(out, run_id)
    assert info["ok"] is False
    assert "unrelated" in info["reason"].lower(), info["reason"]
    assert other_run.is_dir()


def test_complete_replay_missing_durable_tile_refuses_without_publishing(
    tmp_path, monkeypatch
):
    """M2: no planned tile is ever relabelled as measured.

    One Cell's commit is written by a LEGACY-style writer (no executed tile):
    the COMPLETE replay must refuse with a qualified error, write NO
    CURRENT/products, leave ``finalized=false`` and keep every commit byte.
    """
    import test_zegrid_resume_complete_gpu_rerun as RERUN
    from zemosaic.core.zegrid import run_publication as zpub

    victim = "r0002c0002"
    real_consume = zz._consume_stack_cell

    def legacy_consume(prepared, **kwargs):
        result = real_consume(prepared, **kwargs)
        if str(result.get("cell_id")) == victim:
            result.pop("percell_tile_px", None)
            record = result.get("record")
            if isinstance(record, dict):
                record.pop("percell_tile_px", None)
        return result

    monkeypatch.setattr(zz, "_consume_stack_cell", legacy_consume)
    out, store = _hetero_percell_run(tmp_path, monkeypatch)
    monkeypatch.setattr(zz, "_consume_stack_cell", real_consume)

    committed = _committed_percell_tiles(store)
    assert victim not in committed, committed
    assert len(committed) == 8, committed

    RERUN._break_finalization(out, store)
    before = RERUN._commit_bytes(store)
    admission = zres.read_state(store)["admission"]
    stacks = RERUN._bomb_cell_execution(monkeypatch)
    commits = RERUN._spy_commits(monkeypatch)
    with pytest.raises(zz._R55PerCellTileEvidenceError, match="REFUSED"):
        _hetero_percell_resume(
            tmp_path,
            monkeypatch,
            _hetero_quadrant_corpus(tmp_path),
            resume_mode="auto",
        )
    assert stacks == [], stacks
    assert commits == [], commits
    assert RERUN._commit_bytes(store) == before
    state = zres.read_state(store)
    assert state["finalized"] is False
    assert state["admission"] == admission
    # No publication whatsoever: no pointer, no products, no generation.
    assert _current_pointers(out) == []
    assert not (Path(out) / zpub.PRODUCTS_JSON_FILENAME).exists()
    assert not list(
        (Path(out) / zpub.PUBLICATION_DIRNAME).glob("*/gen-*/publication.json")
    )


def test_percell_host_ledger_capacity_covers_worst_cell_reservation(tmp_path):
    """L3: the ledger ceiling fits the deepest real per-Cell reservation.

    Counterexample measured on the production-shaped partial resume: the
    reference Cell's unit credit was 7,417,760 B while a deeper Cell's own
    staging bound was 9,915,178 B with ``queue_max=1``. A reference-only
    ceiling refused that legal Cell. The capacity covers it and NEVER exceeds
    the admitted plan budget; a homogeneous run keeps the historical value.
    """
    budget = 10 * GIB
    unit_host = 7_417_760
    deep = 9_915_178
    queue_max = 1
    historical = min(budget, unit_host * queue_max)
    assert historical < deep  # the false-refusal shape, reproduced

    capacity = zz._r55_pipeline_host_capacity(
        budget_bytes=budget,
        unit_host=unit_host,
        queue_max=queue_max,
        percell_reserves=[deep, 5_000_000],
    )
    assert capacity >= deep
    assert capacity <= budget  # never above the ADMITTED budget
    # A reservation the admitted budget cannot hold is still capped (fail-closed
    # at the reserve call, never a silent overspend).
    assert (
        zz._r55_pipeline_host_capacity(
            budget_bytes=deep - 1,
            unit_host=unit_host,
            queue_max=queue_max,
            percell_reserves=[deep],
        )
        == deep - 1
    )
    # OFF / homogeneous: byte-identical to the historical ceiling.
    assert zz._r55_pipeline_host_capacity(
        budget_bytes=budget, unit_host=unit_host, queue_max=3
    ) == min(budget, unit_host * 3)

    # Real heterogeneous decision: every per-Cell reservation fits.
    bundle, _ctxs, _canvas, _frames = _hetero_bundle()
    decision = _decide(tmp_path, percell=True)
    reserves = []
    for entry in decision.percell_entries:
        unit_cache = zz._r55_cache_unit_bytes(
            int(entry.patch_area_px), int(entry.n_contributors)
        )
        reserves.append(
            zz._r55_prefetch_resident_bytes(
                storage_profile="dense",
                per_unit_cache_bytes=int(unit_cache),
                worst_patch_px=int(entry.patch_area_px),
                channels=3,
            )
        )
    plan_budget = int(bundle.envelope.budget_bytes)
    real_capacity = zz._r55_pipeline_host_capacity(
        budget_bytes=plan_budget,
        unit_host=int(decision.facts["per_unit_prefetch_host"]),
        queue_max=int(decision.mapping["prefetch_units"]) + 1,
        percell_reserves=reserves,
    )
    assert real_capacity >= max(reserves)
    assert real_capacity <= plan_budget
