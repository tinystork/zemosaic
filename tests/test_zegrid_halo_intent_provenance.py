"""ZM-HALO-EXECUTION-DECOUPLING REWORK-1 — truthful halo user-intent provenance.

Nono review-0 M1: the first implementation mapped ``explicit_argument``/``config``
to advisory, but the Qt path always forwards ``zegrid_halo_px`` (including the
untouched product default 64), and ``load_config`` merges ``DEFAULT_CONFIG``.
Every normal GUI run therefore became advisory, silently broadening behaviour.

These tests pin the END-TO-END distinction:

* untouched product default 64        -> ``product_default`` -> HARD FLOOR
* user-selected width (incl. 64)      -> advisory
* legacy persisted non-default width  -> migrated advisory
* legacy/default 64 without metadata  -> untouched default / hard floor
* direct explicit API (incl. 64)      -> advisory
* internal/absent                     -> product_default / hard floor
* worker transport + engine + rebuild -> the SAME frozen intent/policy

No pixels are stacked beyond a tiny 2-frame smoke corpus; no output folder other
than pytest's own temp dir is touched.
"""

from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.wcs import WCS

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic import zegrid_halo as zhalo  # noqa: E402
from zemosaic import zemosaic_config as zconf  # noqa: E402
from zemosaic.core.zegrid import auto_layout as zal  # noqa: E402
from zemosaic.core.zegrid import geometry as zg  # noqa: E402

ADVISORY = zhalo.HALO_OVERHEAD_POLICY_ADVISORY
HARD = zhalo.HALO_OVERHEAD_POLICY_HARD_FLOOR
INTENT_KEY = "zegrid_halo_px_user_selected"


def _policy(prov: dict) -> str:
    return zhalo.halo_overhead_policy_for_source(prov["source"])


# ---------------------------------------------------------------------------
# A. Provenance + policy contract
# ---------------------------------------------------------------------------

def test_untouched_default_64_is_product_default_and_hard_floor():
    prov = zhalo.resolve_run_halo_provenance(
        64, 64, explicit_intent=False, config_intent=False
    )
    assert prov == {
        "requested_px": None,
        "effective_px": 64,
        "source": "product_default",
        "units": "output_canvas_px",
    }
    assert _policy(prov) == HARD


def test_explicitly_selected_64_is_advisory():
    # GUI user selected 64 (intent True) and the worker forwards it.
    prov = zhalo.resolve_run_halo_provenance(
        64, 64, explicit_intent=True, config_intent=True
    )
    assert prov["source"] == "explicit_argument"
    assert prov["requested_px"] == 64 and prov["effective_px"] == 64
    assert _policy(prov) == ADVISORY


def test_gui_selected_128_is_advisory():
    prov = zhalo.resolve_run_halo_provenance(
        128, 128, explicit_intent=True, config_intent=True
    )
    assert prov["source"] == "explicit_argument"
    assert prov["requested_px"] == 128
    assert _policy(prov) == ADVISORY


def test_legacy_config_non_default_migrates_to_advisory():
    # Persisted 128 without the new metadata (older build / CLI config).
    assert zhalo.resolve_halo_user_selected(None, 128, intent_persisted=False) is True
    prov = zhalo.resolve_run_halo_provenance(
        None, 128, explicit_intent=None, config_intent=None
    )
    assert prov["source"] == "config"
    assert prov["requested_px"] == 128
    assert _policy(prov) == ADVISORY


def test_legacy_default_64_without_metadata_is_hard_floor():
    assert zhalo.resolve_halo_user_selected(None, 64, intent_persisted=False) is False
    prov = zhalo.resolve_run_halo_provenance(
        None, 64, explicit_intent=None, config_intent=None
    )
    assert prov["source"] == "product_default"
    assert prov["requested_px"] is None
    assert _policy(prov) == HARD


def test_direct_explicit_api_keeps_advisory_semantics():
    # Direct API caller passing the keyword explicitly (no transport metadata).
    for value in (8, 64, 128, 256):
        prov = zhalo.resolve_run_halo_provenance(value, None)
        assert prov["source"] == "explicit_argument"
        assert prov["requested_px"] == value
        assert _policy(prov) == ADVISORY


def test_internal_absent_is_hard_floor():
    prov = zhalo.resolve_run_halo_provenance(None, None)
    assert prov["source"] == "product_default"
    assert _policy(prov) == HARD


def test_contradictory_transport_intent_fails_closed():
    with pytest.raises(zhalo.ZeGridHaloError):
        zhalo.resolve_run_halo_provenance(
            128, None, explicit_intent=False
        )
    # The persisted contradiction is normalized UP (non-default = user intent),
    # never silently downgraded to "untouched default".
    assert zhalo.resolve_halo_user_selected(False, 128, intent_persisted=True) is True
    assert (
        zhalo.resolve_run_halo_provenance(
            None, 128, config_intent=False
        )["source"]
        == "config"
    )


@pytest.mark.parametrize(
    "raw,value,expected",
    [
        (True, 64, True),
        (False, 64, False),
        (None, 64, False),
        (None, 256, True),
        (False, 256, True),      # contradiction normalized up
        ("true", 64, True),
        ("false", 128, True),    # invalid-for-value -> value proves intent
        (0, 64, False),
        (1, 64, True),
        ("maybe", 64, False),    # unusable metadata -> migrate by value
        ("maybe", 128, True),
    ],
)
def test_user_selected_migration_matrix(raw, value, expected):
    assert (
        zhalo.resolve_halo_user_selected(raw, value, intent_persisted=True)
        is expected
    )


# ---------------------------------------------------------------------------
# B. Persistence boundary (real config loader)
# ---------------------------------------------------------------------------

def _load_persisted(tmp_path, payload):
    path = tmp_path / "zemosaic_config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return zconf.load_config(path)


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({}, False),                                   # no file intent, 64 default
        ({"zegrid_halo_px": 64}, False),               # legacy default
        ({"zegrid_halo_px": 128}, True),               # legacy non-default -> intent
        ({"zegrid_halo_px": 64, INTENT_KEY: True}, True),
        ({"zegrid_halo_px": 128, INTENT_KEY: False}, True),  # normalized up
        ({"zegrid_halo_px": 64, INTENT_KEY: "maybe"}, False),
        ({"zegrid_halo_px": 128, INTENT_KEY: "maybe"}, True),
    ],
)
def test_config_loader_migrates_halo_intent(tmp_path, payload, expected):
    loaded = _load_persisted(tmp_path, payload)
    assert loaded[INTENT_KEY] is expected


# ---------------------------------------------------------------------------
# C. GUI persistence, snapshot and worker transport
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
        DEFAULT_CONFIG={},
    )
    monkeypatch.setattr(gui, "zemosaic_config", fake_cfg)
    monkeypatch.setattr(gui, "_detect_analysis_backend", lambda: ("none", None, None))
    return gui.ZeMosaicQtMainWindow()


def _select_halo(win, value):
    combo = win._zegrid_halo_combo
    index = combo.findData(int(value))
    assert index >= 0
    combo.setCurrentIndex(index)


def test_untouched_gui_default_is_hard_floor_through_the_worker(tmp_path, monkeypatch):
    win = _make_window(tmp_path, monkeypatch, {})
    assert win.config["zegrid_halo_px"] == 64
    assert win.config[INTENT_KEY] is False
    _args, worker_kwargs = win._build_worker_invocation(skip_filter_ui=True)
    assert worker_kwargs[INTENT_KEY] is False
    prov = zhalo.resolve_run_halo_provenance(
        worker_kwargs["zegrid_halo_px"],
        worker_kwargs["zegrid_halo_px"],
        explicit_intent=worker_kwargs[INTENT_KEY],
        config_intent=worker_kwargs[INTENT_KEY],
    )
    assert prov["source"] == "product_default"
    assert _policy(prov) == HARD


def test_gui_selected_128_is_advisory_through_the_worker(tmp_path, monkeypatch):
    win = _make_window(tmp_path, monkeypatch, {"zegrid_halo_px": 128})
    assert win.config[INTENT_KEY] is True
    _args, worker_kwargs = win._build_worker_invocation(skip_filter_ui=True)
    assert worker_kwargs["zegrid_halo_px"] == 128
    assert worker_kwargs[INTENT_KEY] is True
    prov = zhalo.resolve_run_halo_provenance(
        worker_kwargs["zegrid_halo_px"], None,
        explicit_intent=worker_kwargs[INTENT_KEY],
    )
    assert prov["source"] == "explicit_argument"
    assert _policy(prov) == ADVISORY


def test_gui_explicit_64_selection_persists_and_reloads_advisory(
    tmp_path, monkeypatch
):
    save_store: dict = {}
    win = _make_window(
        tmp_path, monkeypatch, {"zegrid_halo_px": 128}, save_store=save_store
    )
    _select_halo(win, 64)
    assert win.config["zegrid_halo_px"] == 64
    assert win.config[INTENT_KEY] is True
    win._save_config()
    assert save_store["zegrid_halo_px"] == 64
    assert save_store[INTENT_KEY] is True

    reloaded = _make_window(tmp_path, monkeypatch, dict(save_store))
    assert reloaded.config["zegrid_halo_px"] == 64
    assert reloaded.config[INTENT_KEY] is True
    _args, worker_kwargs = reloaded._build_worker_invocation(skip_filter_ui=True)
    prov = zhalo.resolve_run_halo_provenance(
        worker_kwargs["zegrid_halo_px"], None,
        explicit_intent=worker_kwargs[INTENT_KEY],
    )
    assert prov["source"] == "explicit_argument"
    assert _policy(prov) == ADVISORY


def test_gui_selected_128_persists_and_reloads_advisory(tmp_path, monkeypatch):
    save_store: dict = {}
    win = _make_window(tmp_path, monkeypatch, {}, save_store=save_store)
    _select_halo(win, 128)
    win._save_config()
    assert save_store[INTENT_KEY] is True
    reloaded = _make_window(tmp_path, monkeypatch, dict(save_store))
    assert reloaded.config[INTENT_KEY] is True
    assert reloaded.config["zegrid_halo_px"] == 128


def test_worker_signature_accepts_and_forwards_the_intent(monkeypatch):
    import zemosaic.zemosaic_worker as zw

    params = inspect.signature(zw.run_hierarchical_mosaic).parameters
    assert INTENT_KEY in params

    real_run = zw.run_hierarchical_mosaic
    for flag in (False, True):
        captured: dict = {}

        def _capture(**kwargs):
            captured.update(kwargs)

        _capture.__signature__ = inspect.signature(real_run)  # type: ignore[attr-defined]
        monkeypatch.setattr(zw, "run_hierarchical_mosaic", _capture)
        zw.run_hierarchical_mosaic_process(
            SimpleNamespace(put=lambda *_: None),
            solver_settings_dict=None,
            crash_breadcrumbs_mode="off",
            input_dir="/in",
            output_dir="/out",
            zegrid_mode_enabled=True,
            zegrid_halo_px=64,
            zegrid_halo_px_user_selected=flag,
        )
        assert captured["zegrid_halo_px"] == 64
        assert captured[INTENT_KEY] is flag
        monkeypatch.setattr(zw, "run_hierarchical_mosaic", real_run)


# ---------------------------------------------------------------------------
# D. Real engine: the manifest tells the truth for both intents
# ---------------------------------------------------------------------------

def _write_smoke_corpus(root: Path) -> Path:
    from astropy.io import fits

    root.mkdir(parents=True, exist_ok=True)
    for i, (ra, dec) in enumerate(((10.0, 30.0), (10.005, 30.0))):
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.crval = [ra, dec]
        wcs.wcs.crpix = [256.0, 256.0]
        wcs.wcs.cd = np.array([[-1.0, 0.0], [0.0, 1.0]]) * 0.001
        wcs.array_shape = (512, 512)
        rng = np.random.default_rng(i)
        data = (100.0 + rng.normal(0.0, 0.5, (512, 512))).astype(np.float32)
        hdu = fits.PrimaryHDU(data)
        hdu.header.update(wcs.to_header(relax=True))
        hdu.writeto(root / f"f{i:02d}.fits", overwrite=True)
    return root


def _run_engine(tmp_path, monkeypatch, *, halo, selected):
    import zemosaic.zemosaic_worker as zw
    import zemosaic.zemosaic_zegrid_mode as zz

    inp = _write_smoke_corpus(tmp_path / f"in_{halo}_{selected}")
    out = tmp_path / f"out_{halo}_{selected}"
    base_config = {
        "input_dir": str(inp),
        "output_dir": str(out),
        "zegrid_mode_enabled": True,
        "zegrid_halo_px": halo,
        INTENT_KEY: selected,
        "zegrid_layout": "1x1",
        "cell_gauge_mode": "global",
        "export_aesthetic_fits": False,
        "final_mosaic_dbe_enabled": False,
        "save_final_as_uint16": False,
        "grid_rgb_equalize": False,
        "stack_ram_budget_gb": 0,
    }
    win = _make_window(tmp_path, monkeypatch, base_config)
    win._collect_config_from_widgets()
    _args, worker_kwargs = win._build_worker_invocation(skip_filter_ui=True)
    assert worker_kwargs[INTENT_KEY] is selected

    monkeypatch.setattr(
        zw, "zemosaic_config",
        SimpleNamespace(load_config=lambda: dict(base_config)),
    )
    monkeypatch.setattr(zw, "ZEMOSAIC_CONFIG_AVAILABLE", True)
    monkeypatch.setattr(zz, "available_memory_bytes", lambda: 8 * 2 ** 30)

    real_run = zw.run_hierarchical_mosaic
    sig = inspect.signature(real_run)
    full_kwargs = {}
    for name, param in sig.parameters.items():
        if name in worker_kwargs:
            full_kwargs[name] = worker_kwargs[name]
        elif param.default is inspect._empty:
            ann = str(param.annotation)
            if "bool" in ann:
                full_kwargs[name] = False
            elif "int" in ann:
                full_kwargs[name] = 0
            elif "float" in ann:
                full_kwargs[name] = 0.0
            elif "callable" in ann:
                full_kwargs[name] = lambda *a, **k: None
            elif "tuple" in ann:
                full_kwargs[name] = (0.05, 0.05)
            else:
                full_kwargs[name] = ""
    full_kwargs.update(
        input_folder=str(inp),
        output_folder=str(out),
        progress_callback=lambda *a, **k: None,
        stack_norm_method="linear_fit",
        stack_weight_method="noise_variance",
        stack_reject_algo="winsorized_sigma_clip",
        stack_final_combine="mean",
        parsed_winsor_limits=(0.05, 0.05),
        stack_kappa_low=3.0,
        stack_kappa_high=3.0,
    )
    assert INTENT_KEY in full_kwargs
    real_run(**full_kwargs)
    return json.loads((out / "zegrid_manifest.json").read_text(encoding="utf-8"))


def test_real_engine_untouched_64_reports_product_default_hard_floor(
    tmp_path, monkeypatch
):
    manifest = _run_engine(tmp_path, monkeypatch, halo=64, selected=False)
    halo = manifest["zegrid_halo"]
    assert halo["source"] == "product_default"
    assert halo["requested_px"] is None
    assert halo["effective_px"] == 64
    # Hard floor keeps the historical manifest layout shape (no policy keys).
    assert "halo_overhead_policy" not in manifest["layout"]
    assert "halo_overhead_advisory" not in manifest["layout"]


def test_real_engine_selected_128_reports_explicit_advisory(
    tmp_path, monkeypatch
):
    manifest = _run_engine(tmp_path, monkeypatch, halo=128, selected=True)
    halo = manifest["zegrid_halo"]
    assert halo["source"] == "explicit_argument"
    assert halo["requested_px"] == 128
    assert halo["effective_px"] == 128
    assert manifest["layout"]["halo_overhead_policy"] == ADVISORY
    adv = manifest["layout"]["halo_overhead_advisory"]
    assert adv["policy"] == ADVISORY
    assert adv["halo_px"] == 128
    assert adv["geometry_changed"] is False


# ---------------------------------------------------------------------------
# E. Rebuild seam: dense -> bounded retry keeps the SAME frozen policy
# ---------------------------------------------------------------------------

class _HugeModel:
    def predict_bound_bytes(self, n, area):
        return 10.0 ** 16

    def predict_peak_bytes(self, n, area):
        return 10.0 ** 16

    def to_dict(self):
        return {"stub": "huge"}


def _corpus():
    descs = []
    for i in range(4):
        for j in range(3):
            wcs = WCS(naxis=2)
            wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
            wcs.wcs.crval = [10.0 + i * 0.45, 30.0 + j * 0.4]
            wcs.wcs.crpix = [300.0, 500.0]
            wcs.wcs.cd = np.array([[-1.0, 0.0], [0.0, 1.0]]) * 0.001
            wcs.array_shape = (1000, 600)
            fid = f"it_{i}_{j}.fits"
            descs.append(
                zg.FrameDescriptor(
                    zg.FrameId(fid), f"/x/{fid}", (1000, 600),
                    wcs.to_header(relax=True).tostring(), "", "",
                )
            )
    return descs


def test_dense_to_bounded_rebuild_keeps_the_same_halo_policy(monkeypatch):
    from zemosaic import zemosaic_zegrid_mode as zz
    from zemosaic.core.zegrid import ram_contract as rcontract

    descs = _corpus()
    canvas = zg.build_canvas(descs)
    raw = 4 * 2 ** 20
    budget = int(rcontract.plan_budget_bytes(raw))

    monkeypatch.setattr(zal, "FITTED_MEMORY_MODEL", _HugeModel())

    def _huge_stream(n, hw, tile=zz.STREAM_TILE_SIZE, channels=3,
                     storage_profile=None):
        return 10 ** 15

    monkeypatch.setattr(zz, "_estimate_streaming_bytes", _huge_stream)

    calls: list[dict] = []
    real_build = zz._build_execution_bundle

    def _spy(*args, **kwargs):
        calls.append(dict(kwargs))
        return real_build(*args, **kwargs)

    monkeypatch.setattr(zz, "_build_execution_bundle", _spy)
    zz._storage_policy_reset(None)
    with pytest.raises(Exception):
        zz._plan_bundle_auto(
            canvas=canvas, descs=descs, footprints=None,
            raw=raw, budget=budget, active=True, emit=None,
            bundle_common={
                "cpu_count": 2, "gpu_used": False, "cache_root": None,
                "halo_px": 128, "halo_policy": ADVISORY,
            },
            layout_substep=None,
        )
    # At least the dense build and its bounded rebuild were attempted, and BOTH
    # carried the frozen advisory policy (never re-derived per attempt).
    assert len(calls) >= 2
    assert all(c["halo_policy"] == ADVISORY for c in calls)
    assert all(int(c["halo_px"]) == 128 for c in calls)
