"""ZM-ZEGRID-R55-E3 — beta readiness: Qt selector, propagation, Windows contract.

Acceptance coverage:

* the Qt selector persists the exact ``legacy``/``observe``/``active`` tokens,
  defaults to legacy, refuses an invalid value, and warns that ``active`` is
  experimental (it never enables the GPU by itself);
* GUI -> config/handoff -> worker -> ``run_zegrid_mode`` propagates the token
  without remapping, and every kwargs crossing the worker boundary is either
  accepted or explicitly filtered;
* CPU legacy/observe/active paths trigger no NEW R55 readiness/VRAM probe and
  no GPU execution call (the historical engine startup probe is unchanged);
  active+GPU reaches the E2 pipeline; missing CuPy/device refuses or degrades
  honestly (no silent crash);
* the spawn/thread lifecycle is hermetic and never requires ``fork``;
* packaging keeps CuPy optional, ships every R55 module, and starts CPU-only
  with CuPy genuinely unavailable (historical CuPy import when installed is
  pre-existing and is not claimed otherwise);
* the E3 evidence is byte-deterministic with a standard ``sha256sum -c``
  sidecar, and the Windows checker is fail-closed and read-only.

No test asserts a machine-specific path, GPU model, driver or performance.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
TESTS_DIR = REPO_ROOT / "tests"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from _r55e3_spawn_probe import probe as _spawn_engine_probe  # noqa: E402

from zemosaic import zemosaic_config  # noqa: E402
from zemosaic import zemosaic_worker as zw  # noqa: E402
from zemosaic import zemosaic_zegrid_mode as zz  # noqa: E402
from zemosaic.core.zegrid import gpu_readiness as zgr  # noqa: E402
from zemosaic.core.zegrid import parallel as zpar  # noqa: E402
from zemosaic.core.zegrid import scheduler as zsched  # noqa: E402

LOCALES = ("en", "fr", "de", "es", "is", "nl", "pl")
SCHEDULER_KEYS = (
    "qt_field_zegrid_scheduler_label",
    "qt_field_zegrid_scheduler_tip",
    "qt_zegrid_scheduler_opt_legacy",
    "qt_zegrid_scheduler_opt_observe",
    "qt_zegrid_scheduler_opt_active",
    "qt_zegrid_scheduler_active_warning",
    "qt_zegrid_scheduler_active_run_warning",
    "qt_zegrid_scheduler_invalid_value",
)
ROUTINE_TOOL = REPO_ROOT / "tools" / "zegrid_routine" / "make_routine_corpus.py"
GIB = 2 ** 30
FITS_NAMES = ("mosaic_grid.fits", "mosaic_grid_coverage.fits")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def _qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
    except Exception:  # pragma: no cover - Qt not installed
        pytest.skip("PySide6 unavailable")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _make_window(tmp_path, monkeypatch, config: dict, save_store: dict | None = None):
    import zemosaic.zemosaic_gui_qt as gui

    cfg_path = tmp_path / "zemosaic_config.json"
    merged = dict(config)

    def _save(data):
        if save_store is not None:
            save_store.clear()
            save_store.update(data)
        return None

    fake_cfg = SimpleNamespace(
        load_config=lambda: dict(merged),
        save_config=_save,
        get_config_path=lambda: cfg_path,
        DEFAULT_CONFIG=dict(zemosaic_config.DEFAULT_CONFIG),
    )
    monkeypatch.setattr(gui, "zemosaic_config", fake_cfg)
    monkeypatch.setattr(gui, "_detect_analysis_backend", lambda: ("none", None, None))
    return gui.ZeMosaicQtMainWindow()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _readiness(available=True, status="usable"):
    return zgr.GpuReadiness(
        status=status,
        reason_code=zgr.REASON_GPU_OK if available else zgr.REASON_GPU_NOT_PROBED,
        available=bool(available),
        dependency_declared=True,
        dependency_importable=bool(available),
        device_index=0,
        device_name="Synthetic GPU" if available else None,
        cupy_version="0.test" if available else None,
        vram_total_bytes=4 * GIB if available else None,
        vram_free_bytes=2 * GIB if available else None,
    )


def _fake_gpu_available(monkeypatch, *, free_vram=2 * GIB, total_vram=4 * GIB):
    probe = {
        "available": True,
        "device": "Synthetic GPU",
        "cupy_version": "0.test",
        "vram_total_bytes": int(total_vram),
        "vram_free_bytes": int(free_vram),
        "reason": None,
    }
    monkeypatch.setattr(zz.zgpu, "probe_gpu_backend", lambda: dict(probe))
    monkeypatch.setattr(
        zz.zgr, "probe_gpu_readiness_safe",
        lambda **kw: _readiness(True, "usable"),
    )
    monkeypatch.setattr(
        zz.zgr, "read_vram_free_bytes",
        lambda device_index=0: int(free_vram),
    )
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: 12 * GIB)
    return probe


def _fake_gpu_missing(monkeypatch):
    monkeypatch.setattr(
        zz.zgpu, "probe_gpu_backend",
        lambda: {
            "available": False,
            "device": None,
            "cupy_version": None,
            "vram_total_bytes": None,
            "vram_free_bytes": None,
            "reason": "cupy_missing",
        },
    )
    monkeypatch.setattr(
        zz.zgr, "probe_gpu_readiness_safe",
        lambda **kw: _readiness(False, "dependency_missing"),
    )
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: 12 * GIB)


@pytest.fixture(scope="module")
def routine_input(tmp_path_factory):
    root = tmp_path_factory.mktemp("r55e3_routine")
    corpus = root / "corpus"
    proc = subprocess.run(
        [sys.executable, str(ROUTINE_TOOL), str(corpus)],
        capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    inp = root / "input"
    inp.mkdir()
    for path in sorted(corpus.glob("*.fits")):
        shutil.copy2(path, inp / path.name)
    return inp


def _run_engine(inp, out, mode, *, use_gpu=None):
    zz.run_zegrid_mode(
        str(inp), str(out), zegrid_halo_px=32,
        zegrid_scheduler_mode=mode, use_gpu=use_gpu,
    )


@pytest.fixture(scope="module")
def legacy_run(routine_input, tmp_path_factory):
    out = tmp_path_factory.mktemp("r55e3_legacy")
    before = zz.available_memory_bytes
    zz.available_memory_bytes = lambda: 12 * GIB
    try:
        _run_engine(routine_input, out, "legacy")
    finally:
        zz.available_memory_bytes = before
    return out


def _manifest(out: Path) -> dict:
    return json.loads((out / "zegrid_manifest.json").read_text())


# ---------------------------------------------------------------------------
# A. Qt selector: visible, exact, persistent, fail-visible
# ---------------------------------------------------------------------------

def test_a_scheduler_combo_uses_exact_engine_tokens(tmp_path, monkeypatch):
    win = _make_window(tmp_path, monkeypatch, {})
    combo = win._zegrid_scheduler_combo

    assert combo.count() == len(zsched.MODES) == 3
    assert tuple(combo.itemData(i) for i in range(combo.count())) == zsched.MODES
    assert all(isinstance(combo.itemData(i), str) for i in range(combo.count()))
    assert combo.isEditable() is False
    from PySide6.QtWidgets import QComboBox

    assert combo.insertPolicy() == QComboBox.NoInsert
    assert combo.currentData() == "legacy"
    assert win.config["zegrid_scheduler_mode"] == "legacy"
    assert win._config_fields["zegrid_scheduler_mode"]["kind"] == "combobox"
    assert win._config_fields["zegrid_scheduler_mode"]["type"] is str

    label = win._zegrid_scheduler_label
    assert label.buddy() is combo
    assert "legacy" in combo.toolTip().lower()
    assert "experimental" in combo.toolTip().lower()
    assert "does not enable the gpu" in combo.toolTip().lower()
    assert win._zegrid_scheduler_warning.isHidden() is True


def test_a_round_trip_all_tokens_and_active_warning(tmp_path, monkeypatch):
    save_store: dict = {}
    win = _make_window(tmp_path, monkeypatch, {}, save_store=save_store)

    for token in zsched.MODES:
        win._zegrid_scheduler_combo.setCurrentIndex(
            win._zegrid_scheduler_combo.findData(token)
        )
        assert win.config["zegrid_scheduler_mode"] == token
        win._collect_config_from_widgets()
        assert win.config["zegrid_scheduler_mode"] == token
        snapshot = win._serialize_config_for_save()
        assert snapshot["zegrid_scheduler_mode"] == token
        win._save_config()
        assert save_store["zegrid_scheduler_mode"] == token

    win._zegrid_scheduler_combo.setCurrentIndex(
        win._zegrid_scheduler_combo.findData("active")
    )
    assert win._zegrid_scheduler_warning.isHidden() is False
    assert "experimental" in win._zegrid_scheduler_warning.text().lower()
    assert "gpu" in win._zegrid_scheduler_warning.text().lower()

    # Reload from the persisted token: exact round-trip, no remap.
    win.config["zegrid_scheduler_mode"] = "observe"
    assert win._sync_zegrid_scheduler_combo_from_config() is True
    assert win._zegrid_scheduler_combo.currentData() == "observe"


def test_a_invalid_config_is_preserved_and_refused_for_zegrid(tmp_path, monkeypatch):
    import zemosaic.zemosaic_gui_qt as gui

    win = _make_window(tmp_path, monkeypatch, {"zegrid_scheduler_mode": "BOGUS"})
    combo = win._zegrid_scheduler_combo
    assert combo.currentIndex() == -1
    assert combo.property("zegrid_scheduler_invalid") is True
    assert win.config["zegrid_scheduler_mode"] == "BOGUS"
    assert win._collect_config_from_widgets() is None
    assert win.config["zegrid_scheduler_mode"] == "BOGUS"

    # An unrelated classic run is not blocked.
    assert win._selected_processing_mode() == "classic"
    assert win._validate_zegrid_scheduler_for_start() is True

    # An effective ZeGrid run is blocked with a dialog.
    win._mode_checkboxes["zegrid"].setChecked(True)
    warnings = []
    monkeypatch.setattr(
        gui.QMessageBox, "warning",
        lambda parent, title, message: warnings.append((title, message)),
    )
    assert win._validate_zegrid_scheduler_for_start() is False
    assert warnings and "BOGUS" in warnings[-1][1]
    assert win.config["zegrid_scheduler_mode"] == "BOGUS"


def test_a_locales_define_every_scheduler_string():
    for lang in LOCALES:
        payload = json.loads(
            (REPO_ROOT / "src/zemosaic/locales" / f"{lang}.json").read_text(encoding="utf-8")
        )
        for key in SCHEDULER_KEYS:
            assert key in payload, (lang, key)
            assert isinstance(payload[key], str) and payload[key].strip()
        assert payload["qt_zegrid_scheduler_opt_legacy"] != "qt_zegrid_scheduler_opt_legacy"
        assert "{value}" in payload["qt_zegrid_scheduler_invalid_value"]


# ---------------------------------------------------------------------------
# B. Generic propagation GUI -> config/handoff -> worker -> engine
# ---------------------------------------------------------------------------

def _worker_required_kwargs(input_folder: str, output_folder: str) -> dict:
    return dict(
        input_folder=input_folder,
        output_folder=output_folder,
        astap_exe_path="",
        astap_data_dir_param="",
        astap_search_radius_config=3.0,
        astap_downsample_config=2,
        astap_sensitivity_config=100,
        cluster_threshold_config=0.03,
        cluster_target_groups_config=0,
        cluster_orientation_split_deg_config=0.0,
        progress_callback=lambda *a, **k: None,
        stack_ram_budget_gb_config=0.0,
        stack_norm_method="sky_mean",
        stack_weight_method="noise_variance",
        stack_reject_algo="winsorized_sigma_clip",
        stack_kappa_low=3.0,
        stack_kappa_high=3.0,
        parsed_winsor_limits=(0.05, 0.05),
        stack_final_combine="mean",
        poststack_equalize_rgb_config=False,
        apply_radial_weight_config=False,
        radial_feather_fraction_config=0.8,
        radial_shape_power_config=2.0,
        min_radial_weight_floor_config=0.0,
        final_assembly_method_config="mean",
        inter_master_merge_enable_config=False,
        inter_master_overlap_threshold_config=0.6,
        inter_master_min_group_size_config=2,
        inter_master_stack_method_config="winsor",
        inter_master_memmap_policy_config="auto",
        inter_master_local_scale_config="final",
        inter_master_max_group_config=64,
        num_base_workers_config=0,
        apply_master_tile_crop_config=False,
        master_tile_crop_percent_config=0.0,
        quality_crop_enabled_config=False,
        quality_crop_band_px_config=32,
        quality_crop_k_sigma_config=2.0,
        quality_crop_margin_px_config=8,
        quality_crop_min_run_config=2,
        altaz_cleanup_enabled_config=False,
        altaz_margin_percent_config=5.0,
        altaz_decay_config=0.15,
        altaz_nanize_config=True,
        quality_gate_enabled_config=False,
        quality_gate_threshold_config=0.48,
        quality_gate_edge_band_px_config=64,
        quality_gate_k_sigma_config=2.5,
        quality_gate_erode_px_config=3,
        quality_gate_move_rejects_config=True,
        save_final_as_uint16_config=False,
        legacy_rgb_cube_config=False,
        coadd_use_memmap_config=True,
        coadd_memmap_dir_config="",
        coadd_cleanup_memmap_config=True,
        assembly_process_workers_config=1,
        auto_limit_frames_per_master_tile_config=False,
        winsor_max_frames_per_pass_config=0,
        winsor_worker_limit_config=0,
        max_raw_per_master_tile_config=0,
        filter_overrides={"processing_mode": "zegrid"},
    )


def test_b_gui_handoff_carries_the_exact_token(tmp_path, monkeypatch):
    win = _make_window(tmp_path, monkeypatch, {})
    win._mode_checkboxes["zegrid"].setChecked(True)
    win._zegrid_scheduler_combo.setCurrentIndex(
        win._zegrid_scheduler_combo.findData("active")
    )
    win._collect_config_from_widgets()
    _args, kwargs = win._build_worker_invocation(skip_filter_ui=True)
    assert kwargs["zegrid_scheduler_mode"] == "active"


def test_b_worker_to_engine_receives_the_token_without_remap(tmp_path, monkeypatch):
    captured: dict = {}

    def fake_run(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(zz, "run_zegrid_mode", fake_run)
    kwargs = _worker_required_kwargs(
        str(tmp_path / "in"), str(tmp_path / "out")
    )
    kwargs["zegrid_scheduler_mode"] = "active"
    zw.run_hierarchical_mosaic(**kwargs)
    assert captured["zegrid_scheduler_mode"] == "active"

    captured.clear()
    kwargs["zegrid_scheduler_mode"] = "BOGUS"
    zw.run_hierarchical_mosaic(**kwargs)
    assert captured["zegrid_scheduler_mode"] == "BOGUS"


def test_b_every_worker_boundary_kwarg_is_accepted_or_filtered(tmp_path, monkeypatch):
    real_sig = inspect.signature(zw.run_hierarchical_mosaic)
    captured: dict = {}

    def stub(**final_kwargs):
        captured.update(final_kwargs)

    stub.__signature__ = real_sig  # type: ignore[attr-defined]
    monkeypatch.setattr(zw, "run_hierarchical_mosaic", stub)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))

    class _RecordingQueue:
        def put(self, *_a, **_k):
            return None

    zw.run_hierarchical_mosaic_process(
        _RecordingQueue(),
        solver_settings_dict=None,
        zegrid_scheduler_mode="observe",
        crash_breadcrumbs_mode="off",
        input_dir=str(tmp_path / "in"),
        output_dir=str(tmp_path / "out"),
    )
    assert captured.get("zegrid_scheduler_mode") == "observe"
    unexpected = set(captured) - set(real_sig.parameters)
    assert not unexpected, f"kwargs not accepted by the worker signature: {sorted(unexpected)}"
    assert "zegrid_scheduler_mode" in inspect.signature(zz.run_zegrid_mode).parameters


def test_b_absent_token_keeps_legacy_default():
    assert zemosaic_config.DEFAULT_CONFIG["zegrid_scheduler_mode"] == "legacy"
    assert zsched.normalize_scheduler_mode(None) == "legacy"
    assert zsched.DEFAULT_MODE == "legacy"
    for bad in ("", "auto", "ACTIVE", "OBSERVE", " legacy", True, 0, 1.0):
        with pytest.raises(zsched.InvalidSchedulerMode):
            zsched.normalize_scheduler_mode(bad)


# ---------------------------------------------------------------------------
# C. CPU/GPU matrix and honest CuPy/device absence
# ---------------------------------------------------------------------------

def test_c_cpu_matrix_is_byte_identical_and_triggers_no_r55_gpu_work(
    routine_input, legacy_run, tmp_path, monkeypatch
):
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: 12 * GIB)

    # The historical engine startup probe is pre-existing; this test proves the
    # R55/E3 CPU paths add no NEW readiness/VRAM probe and no GPU execution.
    probe_calls = {"readiness": 0, "vram": 0}
    real_readiness = zz.zgr.probe_gpu_readiness_safe

    def readiness_probe(**kwargs):
        probe_calls["readiness"] += 1
        return real_readiness(**kwargs)

    def vram_probe(device_index=0):
        probe_calls["vram"] += 1
        raise AssertionError("R55 VRAM probe must not run on a CPU path")

    monkeypatch.setattr(zz.zgr, "probe_gpu_readiness_safe", readiness_probe)
    monkeypatch.setattr(zz.zgr, "read_vram_free_bytes", vram_probe)

    real_consume = zz._consume_stack_cell

    def assert_cpu_consume(prepared):
        if isinstance(prepared, dict) and "empty_result" in prepared:
            return real_consume(prepared)
        assert getattr(zz._STACK_CONFIG, "backend", "cpu") == "cpu"
        return real_consume(prepared)

    monkeypatch.setattr(zz, "_consume_stack_cell", assert_cpu_consume)

    outputs = {}
    for mode in ("observe", "active"):
        before = dict(probe_calls)
        out = tmp_path / mode
        _run_engine(routine_input, out, mode)
        outputs[mode] = out
        delta_readiness = probe_calls["readiness"] - before["readiness"]
        delta_vram = probe_calls["vram"] - before["vram"]
        # Observe records one readiness snapshot by design (R55-C); active CPU
        # must not probe readiness at all. Neither may probe VRAM.
        assert delta_readiness == (1 if mode == "observe" else 0)
        assert delta_vram == 0

        block = _manifest(out)["r55_scheduler"]
        if mode == "observe":
            assert block["mode"] == "observe_only"
            assert block["observed_metrics"] is None
        else:
            assert block["mode"] == "active"
            assert block["mapping"]["backend"] == "cpu"
            assert block["mapping"]["prefetch_units"] == 0
        for name in FITS_NAMES:
            assert _digest(out / name) == _digest(legacy_run / name), name

    assert _manifest(outputs["active"])["gpu"]["actually_used"] is False


def test_c_cpu_path_is_independent_of_a_preexisting_cupy_module(
    routine_input, legacy_run, tmp_path, monkeypatch
):
    """Regression for the old order-dependent ``cupy not in sys.modules`` claim.

    A product environment with CuPy installed may already have it imported
    (historical ``parallel_utils`` behaviour). The CPU path must still be exact
    and must not execute GPU work.
    """
    import types

    monkeypatch.setitem(sys.modules, "cupy", types.ModuleType("cupy"))
    _fake_gpu_missing(monkeypatch)
    out = tmp_path / "active_with_cupy_preloaded"
    _run_engine(routine_input, out, "active", use_gpu=True)
    manifest = _manifest(out)
    assert manifest["gpu"]["requested"] is True
    assert manifest["gpu"]["available"] is False
    assert manifest["gpu"]["actually_used"] is False
    assert manifest["gpu"]["final_backend"] == "cpu"
    for name in FITS_NAMES:
        assert _digest(out / name) == _digest(legacy_run / name), name


def test_c_active_gpu_reaches_the_e2_pipeline(
    routine_input, legacy_run, tmp_path, monkeypatch
):
    """R55-E9 routing proof: the SELECTED (>128) tile reaches the runner.

    The GPU math is a CPU-backed deterministic success double (no CuPy on this
    host): this proves ROUTING only and makes no CUDA/performance claim.
    """
    _fake_gpu_available(monkeypatch)
    seen_tiles = []
    real_stream = zz.run_canonical_stack_streaming

    def spy_stream(provider, request, *, tile_size=None, fixed=None):
        seen_tiles.append(None if tile_size is None else int(tile_size))
        return real_stream(provider, request, tile_size=tile_size, fixed=fixed)

    monkeypatch.setattr(zz, "run_canonical_stack_streaming", spy_stream)
    real_consume = zz._consume_stack_cell

    def fake_gpu_consume(prepared):
        if isinstance(prepared, dict) and "empty_result" in prepared:
            return real_consume(prepared)
        # Simulated GPU: the real per-cell math runs on the CPU (no CuPy on this
        # host), and the result is labelled as GPU exactly like the E2 witness.
        cfg = zz._STACK_CONFIG
        zz._STACK_CONFIG = dataclasses.replace(cfg, backend="cpu")
        try:
            result = real_consume(prepared)
        finally:
            zz._STACK_CONFIG = cfg
        result["backend_used"] = "gpu"
        result["record"]["backend_used"] = "gpu"
        return result

    monkeypatch.setattr(zz, "_consume_stack_cell", fake_gpu_consume)
    out = tmp_path / "active_gpu"
    _run_engine(routine_input, out, "active", use_gpu=True)
    manifest = _manifest(out)
    block = manifest["r55_scheduler"]
    assert block["mapping"]["backend"] == "gpu"
    assert block["mapping"]["units_in_flight"] == 1
    assert block["gpu_execution"]["qualified"] is True

    # R55-E9: the dynamic ladder selection is what actually reaches execution.
    tile = int(block["mapping"]["gpu_streaming_tile"])
    assert tile > 128
    assert block["gpu_tile_decision"]["selected_tile_px"] == tile
    assert block["gpu_tile_decision"]["max_tile_px"] >= tile
    assert tile in block["gpu_tile_decision"]["declared_ladder"]
    assert block["execution"]["gpu_streaming_tile"] == tile
    assert manifest["gpu"]["gpu_streaming_tile"] == tile
    # The product seam handed EXACTLY that tile to the existing streaming runner.
    assert seen_tiles
    assert set(seen_tiles) == {tile}
    wave_mapping = manifest["r55_run_telemetry"]["waves"][0]["mapping"]
    assert wave_mapping["gpu_streaming_tile"] == tile
    telemetry = manifest["r55_run_telemetry"]["gpu_pipeline"]
    assert telemetry["queue_max"] == block["mapping"]["prefetch_units"] + 1
    assert telemetry["ledger"]["balanced"] is True
    assert telemetry["threads_joined"] is True
    for name in FITS_NAMES:
        assert _digest(out / name) == _digest(legacy_run / name), name


def test_c_missing_cupy_or_device_degrades_honestly(
    routine_input, legacy_run, tmp_path, monkeypatch
):
    _fake_gpu_missing(monkeypatch)
    out = tmp_path / "active_gpu_missing"
    _run_engine(routine_input, out, "active", use_gpu=True)
    manifest = _manifest(out)
    block = manifest["r55_scheduler"]
    gpu = manifest["gpu"]
    # No crash, no fake GPU: the manifest states the requested-but-unavailable
    # GPU and publishes the CPU backend.
    assert gpu["requested"] is True
    assert gpu["available"] is False
    assert gpu["actually_used"] is False
    assert gpu["final_backend"] == "cpu"
    assert block["mapping"]["backend"] == "cpu"
    for name in FITS_NAMES:
        assert _digest(out / name) == _digest(legacy_run / name), name


# ---------------------------------------------------------------------------
# D. Windows / spawn / thread lifecycle (hermetic)
# ---------------------------------------------------------------------------

def test_d_spawn_path_imports_engine_without_cupy_or_fork():
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(target=_spawn_engine_probe, args=(queue,))
    process.start()
    process.join(120)
    assert process.exitcode == 0
    result = queue.get(timeout=10)
    assert result["start_method"] == "spawn"
    assert result["tokens"] == list(zsched.MODES)
    assert result["invalid_refused"] is True
    assert result["cupy_after_r55_import"] is False
    assert result["engine_import_ok"] is True


def test_d_pipeline_threads_join_on_every_run():
    def run_once():
        pipe = zpar.BoundedPipeline(
            tasks=list(range(8)),
            produce=lambda task: {"value": task},
            consume=lambda payload: payload["value"],
            release=lambda payload: None,
            prefetch_units=2,
            host_budget_bytes=64,
            device_budget_bytes=64,
            host_bytes=lambda payload: 8,
            device_bytes=lambda payload: 8,
        )
        assert pipe.run() == list(range(8))
        telemetry = pipe.telemetry()
        assert telemetry["threads_joined"] is True
        assert telemetry["ledger"]["balanced"] is True

    for _ in range(10):
        run_once()
    import time

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        leftovers = [
            thread.name for thread in threading.enumerate()
            if thread.name.startswith("r55e2-pipeline-")
        ]
        if not leftovers:
            break
        time.sleep(0.05)
    assert not leftovers


def test_d_worker_spawn_never_requests_fork(tmp_path, monkeypatch):
    import zemosaic.zemosaic_gui_qt as gui
    from zemosaic.zemosaic_gui_qt import ZeMosaicQtWorker

    calls: list[str] = []

    class _FakeProcess:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.daemon = kwargs.get("daemon")

        def start(self):
            return None

        def is_alive(self):
            return False

    class _FakeQueue:
        def close(self):
            return None

    class _FakeContext:
        def Queue(self):
            return _FakeQueue()

        def Process(self, **kwargs):
            return _FakeProcess(**kwargs)

    def _get_context(method):
        calls.append(str(method))
        assert method != "fork"
        return _FakeContext()

    fake_mp = SimpleNamespace(
        get_context=_get_context,
        Queue=lambda: _FakeQueue(),
        Process=lambda **kwargs: _FakeProcess(**kwargs),
    )
    monkeypatch.setattr(gui, "multiprocessing", fake_mp)
    monkeypatch.setattr(gui, "run_hierarchical_mosaic_process", object())
    monkeypatch.setattr(gui.platform, "system", lambda: "Linux")
    worker = ZeMosaicQtWorker()
    result = worker.spawn_worker_process((), {"output_dir": str(tmp_path)})
    assert result is not None
    assert calls == ["spawn"]

    # Windows path: the default context is used, never get_context("fork").
    calls.clear()
    monkeypatch.setattr(gui.platform, "system", lambda: "Windows")
    worker = ZeMosaicQtWorker()
    result = worker.spawn_worker_process((), {"output_dir": str(tmp_path)})
    assert result is not None
    assert calls == []
    assert all(call != "fork" for call in calls)


# ---------------------------------------------------------------------------
# E. Packaging / CPU-only import smoke
# ---------------------------------------------------------------------------

def _resolve_module(module: str) -> Path | None:
    parts = module.split(".")
    base = SRC.joinpath(*parts)
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    return None


def _module_name(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][:-3]
    return ".".join(parts)


def _imported_modules(path: Path, root: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    package = _module_name(path, root).split(".")[:-1]
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                keep = len(package) - (node.level - 1)
                if keep < 0:
                    continue
                base = package[:keep]
                module = ".".join(base + ([module] if module else []))
            if module:
                found.add(module)
                # ``from .pkg import submodule`` also imports ``pkg.submodule``.
                for alias in node.names:
                    candidate = f"{module}.{alias.name}"
                    if _resolve_module(candidate) is not None:
                        found.add(candidate)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
    return found


def test_e_r55_modules_are_discoverable_and_shipped_by_packaging_rules():
    """Read the packaging RULES without importing setuptools at test runtime.

    ``setuptools`` is a build-system dependency, not a runtime dependency; the
    product venv intentionally does not provide it.
    """
    import tomllib

    payload = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    find_cfg = payload["tool"]["setuptools"]["packages"]["find"]
    assert find_cfg["where"] == ["src"]
    assert "zemosaic*" in find_cfg["include"]
    assert "locales/*.json" in payload["tool"]["setuptools"]["package-data"]["zemosaic"]

    # Every R55 package has a package marker and every module sits under it, so
    # the declared ``zemosaic*`` discovery rule ships it in a wheel.
    assert (SRC / "zemosaic/core/zegrid/__init__.py").is_file()
    for path in sorted((SRC / "zemosaic/core/zegrid").glob("*.py")):
        assert path.is_file()
    assert (SRC / "zemosaic/zemosaic_zegrid_mode.py").is_file()


def test_e_static_import_closure_covers_lazy_r55_modules_and_keeps_cupy_lazy():
    spec_text = (REPO_ROOT / "ZeMosaic.spec").read_text(encoding="utf-8")
    roots = {"zemosaic._app"}
    for token in ("zemosaic.zemosaic_worker", "zemosaic.zemosaic_gui_qt",
                  "zemosaic.zemosaic_zegrid_mode", "zemosaic.core.zegrid"):
        if token in spec_text:
            roots.add(token)
    roots.add("run_zemosaic")

    reachable: set[str] = set()
    pending = list(roots)
    while pending:
        module = pending.pop()
        if module in reachable:
            continue
        reachable.add(module)
        path = (
            REPO_ROOT / "run_zemosaic.py"
            if module == "run_zemosaic"
            else _resolve_module(module)
        )
        if path is None:
            continue
        for imported in _imported_modules(path, SRC if path != REPO_ROOT / "run_zemosaic.py" else REPO_ROOT):
            if imported.startswith("zemosaic"):
                pending.append(imported)

    # R55 modules actually used by the engine must be statically reachable so
    # PyInstaller's existing bytecode analysis (no spec change) includes them.
    r55_modules = (
        "zemosaic.core.zegrid.scheduler",
        "zemosaic.core.zegrid.gpu_readiness",
        "zemosaic.core.zegrid.parallel",
        "zemosaic.core.zegrid.gpu",
        "zemosaic.core.zegrid.runtime_control",
        "zemosaic.core.zegrid.resource_envelope",
        "zemosaic.core.zegrid.ram_contract",
    )
    assert "zemosaic.zemosaic_zegrid_mode" in reachable
    for module in r55_modules:
        assert module in reachable, module

    # CuPy must never be imported at module import time on CPU paths.
    class _TopLevelImports(ast.NodeVisitor):
        def __init__(self):
            self.names: list[str] = []

        def visit_FunctionDef(self, node):  # noqa: N802 - ast API
            return  # never descend into a function body

        def visit_AsyncFunctionDef(self, node):  # noqa: N802 - ast API
            return

        def visit_Lambda(self, node):  # noqa: N802 - ast API
            return

        def visit_Import(self, node):  # noqa: N802 - ast API
            self.names.extend(alias.name for alias in node.names)

        def visit_ImportFrom(self, node):  # noqa: N802 - ast API
            self.names.append(node.module or "")

    for path in [SRC / "zemosaic/zemosaic_zegrid_mode.py"] + sorted(
        (SRC / "zemosaic/core/zegrid").glob("*.py")
    ):
        visitor = _TopLevelImports()
        visitor.visit(ast.parse(path.read_text(encoding="utf-8")))
        assert not any("cupy" in name for name in visitor.names), (
            f"top-level cupy import in {path}"
        )


def test_e_cpu_only_import_smoke_with_cupy_genuinely_unavailable():
    """Hermetic CPU-only start: CuPy is blocked in the child, not merely unused.

    The historical engine import may import CuPy when it IS installed; this test
    proves the package still starts when it is genuinely unavailable.
    """
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    base, optional = pyproject.split("[project.optional-dependencies]", 1)
    assert "cupy" not in base.lower().split("dependencies = [", 1)[1]
    assert '"cupy-cuda12x"' in optional

    code = (
        "import sys; "
        "sys.modules['cupy'] = None; sys.modules['cupyx'] = None; "
        "import zemosaic._app; "
        "import zemosaic.zemosaic_worker; "
        "import zemosaic.zemosaic_zegrid_mode; "
        "from zemosaic.core.zegrid import scheduler, gpu_readiness, parallel; "
        "assert scheduler.normalize_scheduler_mode(None) == 'legacy'; "
        "assert sys.modules.get('cupy') is None; "
        "print('OK')"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=120, env=env,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "OK" in proc.stdout


# ---------------------------------------------------------------------------
# F. Evidence + Windows checker
# ---------------------------------------------------------------------------

def test_f_e3_evidence_is_deterministic_and_sha256sum_compatible(tmp_path):
    tool = REPO_ROOT / "tools/zegrid_r55/e3_readiness_evidence.py"
    out1 = tmp_path / "e1"
    out2 = tmp_path / "e2"
    for out in (out1, out2):
        proc = subprocess.run(
            [sys.executable, str(tool), "--out", str(out)],
            capture_output=True, text=True, timeout=120,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
    name = "r55e3_readiness_evidence.json"
    assert (out1 / name).read_bytes() == (out2 / name).read_bytes()
    sidecar = (out1 / (name + ".sha256")).read_text(encoding="ascii")
    assert sidecar == (out2 / (name + ".sha256")).read_text(encoding="ascii")
    check = subprocess.run(
        ["sha256sum", "-c", name + ".sha256"],
        cwd=str(out1), capture_output=True, text=True,
    )
    assert check.returncode == 0, check.stderr

    payload = json.loads((out1 / name).read_text(encoding="utf-8"))
    assert payload["status"] == "WINDOWS_NOT_RUN"
    assert payload["windows_qualification"]["qualified"] is False
    assert payload["git_head"]
    assert payload["packaging"]["cupy_in_base_dependencies"] is False
    assert payload["source_hashes"]["src/zemosaic/zemosaic_gui_qt.py"]
    verify = subprocess.run(
        [sys.executable, str(tool), "--out", str(tmp_path / "e3"), "--verify", str(out1)],
        capture_output=True, text=True, timeout=120,
    )
    assert verify.returncode == 0, verify.stdout + verify.stderr


def _write_synthetic_windows_result(root: Path, *, fallback: bool = False) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "mosaic_grid.fits").write_bytes(b"synthetic-science")
    (root / "mosaic_grid_coverage.fits").write_bytes(b"synthetic-coverage")
    (root / "run.log").write_text("synthetic windows run log\n", encoding="utf-8")
    manifest = {
        "r55_scheduler": {
            "mode": "active",
            "requested_mode": "active",
            "status": "executed",
            "identity": {
                "layout": [1, 1], "cell_ids": ["r0000c0000"],
                "frame_ids": ["frame_0000.fits"], "halo_px": 64,
                "refinement_rank": 1,
            },
            "mapping": {
                "backend": "gpu", "units_in_flight": 1, "prefetch_units": 2,
            },
            "gpu_execution": {
                "qualified": True, "percent": 70, "device_peak_bytes": 1000,
            },
        },
        "r55_run_telemetry": {
            "gpu_pipeline": {
                "queue_max": 3, "threads_joined": True,
                "ledger": {"balanced": True}, "cancelled": bool(fallback),
            }
        },
        "gpu": {
            "final_backend": "cpu" if fallback else "gpu",
            "actually_used": not fallback,
        },
    }
    if fallback:
        manifest["r55_scheduler"]["gpu_fallback"] = {
            "reason_code": "R55_GPU_RUNTIME",
        }
    (root / "zegrid_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    parity = {
        name: _digest(root / name) for name in FITS_NAMES
    }
    result = {
        "schema": "zemosaic-r55e3-windows-result-v1",
        "status": "DONE",
        "os": {"name": "Windows", "version": "11", "arch": "AMD64"},
        "gpu": {
            "name": "Synthetic RTX", "driver": "000.00", "runtime": "12.x",
            "vram_total_bytes": 4 * GIB, "vram_free_bytes_at_start": 2 * GIB,
        },
        "config": {
            "zegrid_mode": True, "use_gpu": True,
            "zegrid_scheduler_mode": "active", "zegrid_halo_px": 64,
        },
        "run": {"corpus_frames": 1, "peak_rss_bytes": 1024},
        "fallback": {
            "used": bool(fallback),
            "reason_code": "R55_GPU_RUNTIME" if fallback else None,
        },
        "parity_reference": parity,
        "artifacts": {
            "zegrid_manifest.json": _digest(root / "zegrid_manifest.json"),
            "run.log": _digest(root / "run.log"),
            "mosaic_grid.fits": _digest(root / "mosaic_grid.fits"),
            "mosaic_grid_coverage.fits": _digest(root / "mosaic_grid_coverage.fits"),
        },
    }
    (root / "windows_result.json").write_text(
        json.dumps(result, sort_keys=True), encoding="utf-8"
    )


def test_f_windows_checker_is_read_only_fail_closed_and_unqualified(tmp_path):
    checker = REPO_ROOT / "tools/zegrid_r55/windows_result_check.py"

    missing = subprocess.run(
        [sys.executable, str(checker), "--result", str(tmp_path / "absent")],
        capture_output=True, text=True,
    )
    assert missing.returncode == 2
    assert "WINDOWS_NOT_RUN" in missing.stdout

    valid_dir = tmp_path / "valid"
    _write_synthetic_windows_result(valid_dir)
    before = {p.name: p.read_bytes() for p in valid_dir.iterdir()}
    valid = subprocess.run(
        [sys.executable, str(checker), "--result", str(valid_dir)],
        capture_output=True, text=True,
    )
    assert valid.returncode == 0, valid.stdout + valid.stderr
    verdict = json.loads(valid.stdout)
    assert verdict["status"] == "WINDOWS_RESULT_VALID_UNQUALIFIED"
    assert verdict["qualified"] is False
    after = {p.name: p.read_bytes() for p in valid_dir.iterdir()}
    assert before == after

    fallback_dir = tmp_path / "fallback"
    _write_synthetic_windows_result(fallback_dir, fallback=True)
    fallback = subprocess.run(
        [sys.executable, str(checker), "--result", str(fallback_dir)],
        capture_output=True, text=True,
    )
    assert fallback.returncode == 0, fallback.stdout + fallback.stderr

    bad_dir = tmp_path / "bad"
    _write_synthetic_windows_result(bad_dir)
    (bad_dir / "mosaic_grid.fits").write_bytes(b"tampered")
    bad = subprocess.run(
        [sys.executable, str(checker), "--result", str(bad_dir)],
        capture_output=True, text=True,
    )
    assert bad.returncode == 3
    assert "WINDOWS_RESULT_INVALID" in bad.stdout
