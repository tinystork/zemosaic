"""ZM-ZEGRID-R55-E4 — GUI GPU toggle is authoritative, end to end.

The Qt checkbox ``Use GPU acceleration when available`` (``use_gpu_phase5``) is
the single canonical user intent. This module locks the contract in BOTH senses:

    real QCheckBox
      -> real ``_collect_config_from_widgets`` / ``_serialize_config_for_save``
      -> persisted config file
      -> real ``_build_worker_invocation`` payload (process handoff)
      -> real ``run_hierarchical_mosaic_process``
      -> real ``run_hierarchical_mosaic`` ZeGrid branch
      -> ``run_zegrid_mode(use_gpu=...)`` (captured at the engine boundary)

Why this regression exists: on the published beta, unchecking the box produced
``use_gpu_phase5=False`` but left the legacy ``use_gpu_grid=True`` in the saved
snapshot. The worker therefore resolved ``use_gpu=True`` and kept using the GPU
while the user had explicitly disabled it. The existing R22 test only covered
the positive direction.

Documented priority: ``use_gpu_phase5`` wins whenever it is present; the legacy
``use_gpu_grid`` / ``stack_use_gpu`` / ``use_gpu_stack`` aliases are fallbacks
only when the canonical key is genuinely absent, and all keys are rewritten to
one coherent value so an obsolete grid value can never contradict the checkbox.

No GPU, no science path, no real corpus: ``run_zegrid_mode`` is the capture
boundary and returns immediately.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic import zemosaic_config as zconfig_mod  # noqa: E402
from zemosaic import zemosaic_worker as zw  # noqa: E402
from zemosaic import zemosaic_zegrid_mode as zz  # noqa: E402

GPU_KEYS = ("use_gpu_phase5", "use_gpu_grid", "stack_use_gpu", "use_gpu_stack")


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _real_config_path(tmp_path, monkeypatch) -> Path:
    """Point the REAL config module at a hermetic temp config file."""
    cfg_path = tmp_path / "zemosaic_config.json"
    monkeypatch.setattr(zconfig_mod, "get_config_path", lambda: cfg_path)
    return cfg_path


def _new_window(tmp_path, monkeypatch):
    import zemosaic.zemosaic_gui_qt as gui

    cfg_path = _real_config_path(tmp_path, monkeypatch)
    monkeypatch.setattr(gui, "_detect_analysis_backend", lambda: ("none", None, None))
    return gui.ZeMosaicQtMainWindow(), cfg_path


def _gpu_checkbox(win):
    binding = win._config_fields["use_gpu_phase5"]
    assert binding["kind"] == "checkbox"
    return binding["widget"]


def _worker_required_kwargs(input_folder: str, output_folder: str) -> dict:
    """Minimal kwargs that reach the ZeGrid branch of the real worker."""
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
        stack_reject_algo="kappa_sigma",
        stack_kappa_low=3.0,
        stack_kappa_high=3.0,
        parsed_winsor_limits=(0.05, 0.05),
        stack_final_combine="mean",
        poststack_equalize_rgb_config=False,
        apply_radial_weight_config=False,
        radial_feather_fraction_config=0.8,
        radial_shape_power_config=2.0,
        min_radial_weight_floor_config=0.0,
        final_assembly_method_config="reproject_coadd",
        inter_master_merge_enable_config=False,
        inter_master_overlap_threshold_config=0.3,
        inter_master_min_group_size_config=1,
        inter_master_stack_method_config="mean",
        inter_master_memmap_policy_config="auto",
        inter_master_local_scale_config="none",
        inter_master_max_group_config=1,
        num_base_workers_config=2,
        apply_master_tile_crop_config=False,
        master_tile_crop_percent_config=0.0,
        quality_crop_enabled_config=False,
        quality_crop_band_px_config=0,
        quality_crop_k_sigma_config=3.0,
        quality_crop_margin_px_config=0,
        quality_crop_min_run_config=0,
        altaz_cleanup_enabled_config=False,
        altaz_margin_percent_config=0.0,
        altaz_decay_config=0.0,
        altaz_nanize_config=False,
        quality_gate_enabled_config=False,
        quality_gate_threshold_config=0.5,
        quality_gate_edge_band_px_config=0,
        quality_gate_k_sigma_config=3.0,
        quality_gate_erode_px_config=0,
        quality_gate_move_rejects_config=False,
        save_final_as_uint16_config=False,
        legacy_rgb_cube_config=False,
        coadd_use_memmap_config=False,
        coadd_memmap_dir_config="",
        coadd_cleanup_memmap_config=False,
        assembly_process_workers_config=1,
        auto_limit_frames_per_master_tile_config=False,
        winsor_max_frames_per_pass_config=0,
        winsor_worker_limit_config=1,
        max_raw_per_master_tile_config=100,
        filter_overrides={"processing_mode": "zegrid"},
    )


class _RecordingQueue:
    def put(self, *_args, **_kwargs):
        return None


def _prepare_zegrid_window(win, tmp_path) -> None:
    win.config["input_dir"] = str(tmp_path / "in")
    win.config["output_dir"] = str(tmp_path / "out")
    win._mode_checkboxes["zegrid"].setChecked(True)
    assert win._selected_processing_mode() == "zegrid"


def _drive_gui_to_engine(win, cfg_path, tmp_path, monkeypatch, *, gpu: bool) -> dict:
    """Run the REAL GUI -> save -> handoff -> worker -> engine chain once."""
    _prepare_zegrid_window(win, tmp_path)

    _gpu_checkbox(win).setChecked(gpu)
    win._collect_config_from_widgets()

    snapshot = win._serialize_config_for_save()
    for key in GPU_KEYS:
        assert snapshot[key] is gpu, (key, snapshot.get(key))

    # Real persistence: the worker reloads THIS file.
    win._save_config()
    saved = json.loads(cfg_path.read_text(encoding="utf-8"))
    for key in GPU_KEYS:
        assert saved[key] is gpu, (key, saved.get(key))

    # Real process handoff payload.
    _args, worker_kwargs = win._build_worker_invocation(skip_filter_ui=True)
    for key in GPU_KEYS:
        assert worker_kwargs[key] is gpu, (key, worker_kwargs.get(key))

    captured: dict = {}

    def _capture(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(zz, "run_zegrid_mode", _capture)

    # Complete the minimal kwargs the worker needs; everything GPU-related comes
    # from the real GUI handoff payload, not from this fixture.
    required = _worker_required_kwargs(
        str(tmp_path / "in"), str(tmp_path / "out")
    )
    required.update(worker_kwargs)
    required["filter_overrides"] = {"processing_mode": "zegrid"}
    required["crash_breadcrumbs_mode"] = "off"
    # The GUI snapshot already carries this key; the wrapper owns it explicitly.
    required.pop("solver_settings_dict", None)

    # REAL process wrapper + REAL run_hierarchical_mosaic + REAL ZeGrid branch.
    zw.run_hierarchical_mosaic_process(
        _RecordingQueue(), solver_settings_dict=None, **required
    )

    assert captured, "ZeGrid engine was not reached"
    device_flag = captured["use_gpu"]
    assert device_flag is gpu, (gpu, device_flag)
    assert bool(captured["use_gpu"]) is gpu
    return captured


# ---------------------------------------------------------------------------
# 1. End-to-end contract, both senses
# ---------------------------------------------------------------------------

def test_gui_checkbox_off_propagates_all_the_way_to_engine(tmp_path, monkeypatch):
    win, cfg_path = _new_window(tmp_path, monkeypatch)
    _drive_gui_to_engine(win, cfg_path, tmp_path, monkeypatch, gpu=False)


def test_gui_checkbox_on_propagates_all_the_way_to_engine(tmp_path, monkeypatch):
    win, cfg_path = _new_window(tmp_path, monkeypatch)
    _drive_gui_to_engine(win, cfg_path, tmp_path, monkeypatch, gpu=True)


def test_gui_toggle_round_trips_both_senses_in_one_window(tmp_path, monkeypatch):
    """OFF -> ON -> OFF on the same real widget must never leak a stale alias."""
    win, cfg_path = _new_window(tmp_path, monkeypatch)
    for gpu in (False, True, False):
        _drive_gui_to_engine(win, cfg_path, tmp_path, monkeypatch, gpu=gpu)


def test_scheduler_active_never_enables_gpu_when_checkbox_is_off(tmp_path, monkeypatch):
    """R55 scheduler selection stays independent from the GPU toggle."""
    win, cfg_path = _new_window(tmp_path, monkeypatch)
    _prepare_zegrid_window(win, tmp_path)
    combo = win._zegrid_scheduler_combo
    combo.setCurrentIndex(combo.findData("active"))
    win._collect_config_from_widgets()
    assert win.config["zegrid_scheduler_mode"] == "active"

    captured = _drive_gui_to_engine(
        win, cfg_path, tmp_path, monkeypatch, gpu=False
    )
    assert captured["zegrid_scheduler_mode"] == "active"
    assert captured["use_gpu"] is False


# ---------------------------------------------------------------------------
# 2. Persisted/legacy config normalization (defense in depth)
# ---------------------------------------------------------------------------

def test_legacy_config_normalization_cannot_contradict_the_checkbox():
    cases = (
        # stale grid/stack aliases say True, canonical checkbox says False
        (
            {
                "use_gpu_phase5": False,
                "use_gpu_grid": True,
                "stack_use_gpu": True,
                "use_gpu_stack": True,
            },
            False,
        ),
        # string forms, same rule
        (
            {
                "use_gpu_phase5": "0",
                "use_gpu_grid": "1",
                "stack_use_gpu": "yes",
                "use_gpu_stack": "on",
            },
            False,
        ),
        # canonical ON wins over stale OFF aliases
        (
            {
                "use_gpu_phase5": "true",
                "use_gpu_grid": "false",
                "stack_use_gpu": False,
                "use_gpu_stack": "0",
            },
            True,
        ),
    )
    for raw, expected in cases:
        cfg = dict(raw)
        zconfig_mod._normalize_gpu_flags(cfg)
        assert all(cfg[key] is expected for key in GPU_KEYS), (raw, cfg)


def test_legacy_grid_only_config_keeps_its_intent_as_last_resort():
    """With no canonical key at all, the historical alias still works."""
    for legacy_value, expected in ((True, True), ("1", True), (False, False), ("off", False)):
        cfg = {"use_gpu_grid": legacy_value}
        zconfig_mod._normalize_gpu_flags(cfg)
        assert all(cfg[key] is expected for key in GPU_KEYS), (legacy_value, cfg)


def test_real_load_config_normalizes_a_stale_grid_key(tmp_path, monkeypatch):
    cfg_path = _real_config_path(tmp_path, monkeypatch)
    cfg_path.write_text(
        json.dumps(
            {
                "use_gpu_phase5": False,
                "use_gpu_grid": True,
                "stack_use_gpu": "yes",
                "use_gpu_stack": 1,
            }
        ),
        encoding="utf-8",
    )
    loaded = zconfig_mod.load_config()
    assert all(loaded[key] is False for key in GPU_KEYS), loaded

    cfg_path.write_text(
        json.dumps({"use_gpu_phase5": "true", "use_gpu_grid": "0"}),
        encoding="utf-8",
    )
    loaded = zconfig_mod.load_config()
    assert all(loaded[key] is True for key in GPU_KEYS), loaded


def test_worker_explicit_handoff_beats_a_stale_persisted_grid_key(monkeypatch):
    """The process handoff is authoritative even if the persisted config is stale.

    This isolates the transport itself: the persisted config only carries the
    obsolete ``use_gpu_grid=True`` (no canonical key), which historically made
    the engine resolve ``use_gpu=True``. The explicit GUI handoff (False) must
    still win.
    """
    from types import SimpleNamespace

    captured: dict = {}

    def _capture(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(zz, "run_zegrid_mode", _capture)
    monkeypatch.setattr(zw, "ZEMOSAIC_CONFIG_AVAILABLE", True)
    monkeypatch.setattr(
        zw,
        "zemosaic_config",
        SimpleNamespace(
            load_config=lambda: {"use_gpu_grid": True, "zegrid_mode_enabled": True}
        ),
    )

    kwargs = _worker_required_kwargs("/tmp/r55e4-in", "/tmp/r55e4-out")
    kwargs["filter_overrides"] = {"processing_mode": "zegrid"}
    kwargs["use_gpu_phase5"] = False
    zw.run_hierarchical_mosaic(**kwargs)

    assert captured, "ZeGrid engine was not reached"
    assert captured["use_gpu"] is False


# ---------------------------------------------------------------------------
# 3. Engine-level precedence (no contradictory combination can resurrect GPU)
# ---------------------------------------------------------------------------

def test_canonical_phase5_precedes_the_obsolete_grid_alias():
    from types import SimpleNamespace

    stale = SimpleNamespace(
        use_gpu_phase5=False,
        use_gpu_grid=True,
        stack_use_gpu=True,
        use_gpu_stack=True,
    )
    assert zz.resolve_gpu_preference(None, stale) == (False, "use_gpu_phase5")

    enabled = SimpleNamespace(
        use_gpu_phase5=True,
        use_gpu_grid=False,
        stack_use_gpu=False,
        use_gpu_stack=False,
    )
    assert zz.resolve_gpu_preference(None, enabled) == (True, "use_gpu_phase5")

    # Canonical absent: legacy relative order is preserved (legacy behavior).
    legacy = SimpleNamespace(use_gpu_grid=False, stack_use_gpu=True)
    assert zz.resolve_gpu_preference(None, legacy) == (False, "use_gpu_grid")
