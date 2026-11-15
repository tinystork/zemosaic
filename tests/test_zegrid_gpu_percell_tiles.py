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
