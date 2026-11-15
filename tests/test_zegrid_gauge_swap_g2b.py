"""ZM-GAUGE-SWAP-G2b — configuration, GUI -> worker transport, observability, Stop.

Scope (mission ``ZM-GAUGE-SWAP-G2B-20261115``, contract
``docs/refactor/ZEGRID_GAUGE_SWAP_G0_CONTRACT_20261115.md`` §10.4, §10.6, §11
G2b, §13.4):

* ``zegrid_gauge_memory_policy`` is declared with the ``ram_only`` default and
  validated by a STRICT closed enum in the ENGINE config path (not only in the
  Qt GUI); an unknown value is refused, never coerced;
* the value crosses the GUI -> worker boundary EXPLICITLY (the
  ``SEAM_ONLY_KWARGS`` lesson) and the run snapshot publishes the EFFECTIVE
  policy;
* the GUI exposes the exact label, is disabled off-Linux WITH a visible reason,
  and unchecking it is strictly RAM-only without touching the durable stores;
* the manifest carries a structured ``gauge_swap`` block owning the
  ``gauge_credit_open`` / ``gauge_credit_closed`` events and the O1 admission
  fields (policy / A / B_ram / S_credit / sources);
* a Stop during the gauge leaves no "complete" artifact claim, no persisted
  credit and an acknowledged safe boundary; a policy change between attempts
  cannot invalidate a durable resume.

No engine run, no pressure, no benchmark, no Qt window shown (offscreen only).
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic import zemosaic_config as zc  # noqa: E402
from zemosaic.core.zegrid import gauge_credit as zgcredit  # noqa: E402
from zemosaic.core.zegrid import gauge_swap_capacity as cap  # noqa: E402

MIB = 2 ** 20
GIB = 2 ** 30

ENGINE_PATH = SRC / "zemosaic" / "zemosaic_zegrid_mode.py"
CONFIG_PATH = SRC / "zemosaic" / "zemosaic_config.py"
WORKER_PATH = SRC / "zemosaic" / "zemosaic_worker.py"
GUI_PATH = SRC / "zemosaic" / "zemosaic_gui_qt.py"
CREDIT_PATH = SRC / "zemosaic" / "core" / "zegrid" / "gauge_credit.py"
RESUME_PATH = SRC / "zemosaic" / "core" / "zegrid" / "run_resume.py"
STORE_PATH = SRC / "zemosaic" / "core" / "zegrid" / "run_store.py"

POLICY_LABEL = (
    "Autoriser un secours swap limité pour la jauge ; peut ralentir ; "
    "aucun crédit pour le stack"
)


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _function_source(path: Path, name: str) -> str:
    """Exact source of one top-level function (never a truncated grep)."""
    tree = ast.parse(_source(path))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(_source(path), node) or ""
    raise AssertionError(f"{name} not found in {path}")


# ---------------------------------------------------------------------------
# 1. Configuration: declared default + STRICT engine-side closed enum
# ---------------------------------------------------------------------------

def test_policy_is_declared_with_the_ram_only_default():
    assert "zegrid_gauge_memory_policy" in zc.DEFAULT_CONFIG
    assert zc.DEFAULT_CONFIG["zegrid_gauge_memory_policy"] == "ram_only"
    # It is part of the strict run-policy key set (single validation path).
    assert "zegrid_gauge_memory_policy" in zc.ZEGRID_RUN_POLICY_KEYS


def test_engine_validator_accepts_defaults_absent_and_both_tokens():
    resolved = zc.validate_zegrid_run_policy(zc.DEFAULT_CONFIG)
    assert resolved["zegrid_gauge_memory_policy"] == "ram_only"
    assert zc.validate_zegrid_run_policy(None)["zegrid_gauge_memory_policy"] == "ram_only"
    assert (
        zc.validate_zegrid_run_policy(SimpleNamespace())[
            "zegrid_gauge_memory_policy"
        ]
        == "ram_only"
    )
    assert (
        zc.validate_zegrid_run_policy({"zegrid_gauge_memory_policy": "bounded_swap"})[
            "zegrid_gauge_memory_policy"
        ]
        == "bounded_swap"
    )
    # Whitespace/case are normalised, never guessed into a third token.
    assert (
        zc.validate_zegrid_run_policy(
            {"zegrid_gauge_memory_policy": "  BOUNDED_SWAP  "}
        )["zegrid_gauge_memory_policy"]
        == "bounded_swap"
    )
    assert (
        zc.validate_zegrid_run_policy({"zegrid_gauge_memory_policy": "   "})[
            "zegrid_gauge_memory_policy"
        ]
        == "ram_only"
    )
    assert (
        zc.validate_zegrid_run_policy({"zegrid_gauge_memory_policy": None})[
            "zegrid_gauge_memory_policy"
        ]
        == "ram_only"
    )


@pytest.mark.parametrize(
    "value",
    [
        "swapped",
        "bounded",
        "ram-only",
        True,
        False,
        1,
        1.5,
        ["ram_only"],
    ],
)
def test_engine_validator_rejects_an_unknown_policy(value):
    """The ENGINE refuses an unknown enum (never only the GUI)."""
    with pytest.raises(ValueError) as excinfo:
        zc.validate_zegrid_run_policy({"zegrid_gauge_memory_policy": value})
    assert "zegrid_gauge_memory_policy" in str(excinfo.value)


def test_engine_entry_uses_the_validator_as_the_single_authority():
    """``run_zegrid_mode`` validates BEFORE any work (no GUI dependency)."""
    source = _function_source(ENGINE_PATH, "run_zegrid_mode")
    assert "zconf.validate_zegrid_run_policy(zconfig)" in source
    # And the credit resolver refuses an unknown value at the gauge seam.
    with pytest.raises(zgcredit.GaugePolicyError):
        zgcredit.resolve_gauge_memory_policy(
            SimpleNamespace(zegrid_gauge_memory_policy="swapped")
        )
    assert (
        zgcredit.resolve_gauge_memory_policy(SimpleNamespace())
        == cap.POLICY_RAM_ONLY
    )


def test_persisted_unknown_value_is_refused_by_the_loader(tmp_path, monkeypatch):
    cfg_path = tmp_path / "zemosaic_config.json"
    payload = dict(zc.DEFAULT_CONFIG)
    payload["zegrid_gauge_memory_policy"] = "swapped"
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(zc, "get_config_path", lambda: str(cfg_path))
    with pytest.raises(ValueError) as excinfo:
        zc.load_config()
    assert "zegrid_gauge_memory_policy" in str(excinfo.value)


def test_explicit_policy_round_trips_through_save_and_load(tmp_path, monkeypatch):
    cfg_path = tmp_path / "zemosaic_config.json"
    monkeypatch.setattr(zc, "get_config_path", lambda: str(cfg_path))
    cfg = dict(zc.DEFAULT_CONFIG)
    cfg["zegrid_gauge_memory_policy"] = "bounded_swap"
    assert zc.save_config(cfg) is True
    assert zc.load_config()["zegrid_gauge_memory_policy"] == "bounded_swap"


# ---------------------------------------------------------------------------
# 2. GUI -> worker transport (SEAM_ONLY_KWARGS: no forgotten kwargs seam)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def _qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
    except Exception:  # pragma: no cover - Qt unavailable
        pytest.skip("PySide6 unavailable")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _make_window(tmp_path, monkeypatch, config: dict):
    import zemosaic.zemosaic_gui_qt as gui

    cfg_path = tmp_path / "zemosaic_config.json"
    fake_cfg = SimpleNamespace(
        load_config=lambda: dict(config),
        save_config=lambda data: None,
        get_config_path=lambda: cfg_path,
        DEFAULT_CONFIG={},
    )
    monkeypatch.setattr(gui, "zemosaic_config", fake_cfg)
    monkeypatch.setattr(gui, "_detect_analysis_backend", lambda: ("none", None, None))
    return gui, gui.ZeMosaicQtMainWindow()


def test_gui_label_is_exact_and_control_enabled_on_linux(tmp_path, monkeypatch):
    gui, window = _make_window(tmp_path, monkeypatch, {})
    monkeypatch.setattr(gui, "_zegrid_gauge_swap_platform_supported", lambda: True)
    window._apply_zegrid_gauge_swap_control_state()
    checkbox = window._zegrid_gauge_swap_checkbox
    assert checkbox.text() == POLICY_LABEL
    assert checkbox.isEnabled() is True
    # Q2: the same row explains why the policy has no effect elsewhere.
    note = window._zegrid_gauge_swap_note.text()
    assert "ACTIVE" in note
    assert "restored or imported" in note
    assert "never credited" in note


def test_gui_control_is_disabled_with_a_visible_reason_off_linux(
    tmp_path, monkeypatch
):
    gui, window = _make_window(tmp_path, monkeypatch, {})
    monkeypatch.setattr(gui, "_zegrid_gauge_swap_platform_supported", lambda: False)
    window._apply_zegrid_gauge_swap_control_state()
    checkbox = window._zegrid_gauge_swap_checkbox
    assert checkbox.isEnabled() is False
    note = window._zegrid_gauge_swap_note.text()
    assert "Linux-only" in note
    assert "RAM-only" in note
    # The effective policy of this launch can never claim bounded_swap here.
    window.config["zegrid_gauge_memory_policy"] = "bounded_swap"
    assert window._effective_gauge_memory_policy() == "ram_only"
    # ... while the persisted user choice is preserved verbatim.
    assert window.config["zegrid_gauge_memory_policy"] == "bounded_swap"


def test_gui_checkbox_maps_to_the_enum_and_snapshot_matches_transport(
    tmp_path, monkeypatch
):
    gui, window = _make_window(
        tmp_path,
        monkeypatch,
        {
            "zegrid_gauge_memory_policy": "ram_only",
            "zegrid_durable_store_dir": str(tmp_path / "store"),
            "zegrid_resume_mode": "auto",
            "zegrid_resume_run_id": "run-1",
        },
    )
    monkeypatch.setattr(gui, "_zegrid_gauge_swap_platform_supported", lambda: True)
    window._apply_zegrid_gauge_swap_control_state()
    checkbox = window._zegrid_gauge_swap_checkbox
    checkbox.setChecked(True)
    assert window.config["zegrid_gauge_memory_policy"] == "bounded_swap"

    snapshot = window._snapshot_for_run()
    _, worker_kwargs = window._build_worker_invocation(skip_filter_ui=False)
    assert snapshot["zegrid_gauge_memory_policy"] == "bounded_swap"
    assert worker_kwargs["zegrid_gauge_memory_policy"] == "bounded_swap"
    # snapshot == transport (the GUI cannot announce a different run).
    assert worker_kwargs["zegrid_gauge_memory_policy"] == snapshot[
        "zegrid_gauge_memory_policy"
    ]

    # Unchecking is strictly RAM-only and NEVER touches the durable stores.
    store_before = (
        snapshot["zegrid_durable_store_dir"],
        snapshot["zegrid_resume_mode"],
        snapshot["zegrid_resume_run_id"],
    )
    checkbox.setChecked(False)
    assert window.config["zegrid_gauge_memory_policy"] == "ram_only"
    snapshot_off = window._snapshot_for_run()
    _, worker_kwargs_off = window._build_worker_invocation(skip_filter_ui=False)
    assert snapshot_off["zegrid_gauge_memory_policy"] == "ram_only"
    assert worker_kwargs_off["zegrid_gauge_memory_policy"] == "ram_only"
    assert (
        snapshot_off["zegrid_durable_store_dir"],
        snapshot_off["zegrid_resume_mode"],
        snapshot_off["zegrid_resume_run_id"],
    ) == store_before


def test_worker_signature_and_transport_accept_the_policy():
    """The transported kwarg exists (no ``TypeError``) and reaches ``zconfig``."""
    from zemosaic import zemosaic_worker as zw

    parameters = inspect.signature(zw.run_hierarchical_mosaic).parameters
    assert "zegrid_gauge_memory_policy" in parameters
    sentinel = zw._ZEGRID_POLICY_UNSET
    explicit = zw._zegrid_explicit_policy_transport(
        zegrid_gauge_memory_policy="bounded_swap",
        zegrid_layout=sentinel,
    )
    # Only the explicitly transported key is applied; UNSET keeps the persisted
    # config value (never an accidental overwrite with a sentinel object).
    assert explicit == {"zegrid_gauge_memory_policy": "bounded_swap"}
    zconfig = SimpleNamespace()
    cache: dict = {}
    for key, value in explicit.items():
        cache[key] = value
        setattr(zconfig, key, value)
    assert cache["zegrid_gauge_memory_policy"] == "bounded_swap"
    assert (
        zgcredit.resolve_gauge_memory_policy(zconfig) == cap.POLICY_BOUNDED_SWAP
    )
    # The process entry forwards arbitrary worker kwargs (GUI spawn path).
    process_parameters = inspect.signature(
        zw.run_hierarchical_mosaic_process
    ).parameters
    assert any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in process_parameters.values()
    )


def test_worker_policy_transport_key_is_declared():
    from zemosaic import zemosaic_worker as zw

    assert "zegrid_gauge_memory_policy" in zw._ZEGRID_POLICY_TRANSPORT_KEYS


# ---------------------------------------------------------------------------
# 3. Observability O1/O2: structured manifest block + admission fields
# ---------------------------------------------------------------------------

def _capacity(swap_free=6 * GIB, overcommit_mode=0):
    device = cap.SwapDevice(
        name="/swapfile",
        kind="file",
        size_bytes=int(swap_free) + MIB,
        used_bytes=MIB,
        zram=False,
        eligible=True,
        reason="backed_file",
    )
    level = cap.CgroupLevel(
        path="/sys/fs/cgroup",
        is_root=True,
        memory_controller=True,
        ram_status=cap.UNLIMITED,
        ram_headroom=None,
        swap_status=cap.UNLIMITED,
        swap_headroom=None,
        detail="root_cgroup_is_unbounded",
    )
    return zgcredit.LiveCapacity(
        devices=(device,),
        cgroup_levels=(level,),
        overcommit=cap.read_overcommit(overcommit_mode, None, None),
        facts={"schema": "test-injected", "swaps_available": True},
    )


def _session(**overrides):
    data = dict(
        policy=cap.POLICY_BOUNDED_SWAP,
        capacity=_capacity(),
        platform="linux",
        restored=False,
        imported=False,
        workers=1,
        cell_gauge_mode="global",
    )
    data.update(overrides)
    return zgcredit.GaugeCreditSession(**data)


def _bundle(raw=10 * GIB, budget=(4 * 10 * GIB) // 5):
    return SimpleNamespace(
        envelope=SimpleNamespace(
            raw_available_bytes=int(raw), budget_bytes=int(budget)
        )
    )


def test_o2_block_is_absent_for_ram_only_and_complete_for_bounded_swap():
    from zemosaic import zemosaic_zegrid_mode as zz

    assert zz._gauge_swap_diagnostics_block(policy="ram_only") is None
    assert zz._gauge_swap_diagnostics_block(policy=None) is None

    session = _session()
    # 10 GiB of raw gauge footprint against B_ram = 8 GiB needs ~2 GiB of
    # credit, which the injected 6 GiB swap device covers.
    decision = session.decision_for_bundle(
        _bundle(), p_gauge_raw=10 * GIB
    )[0]
    opened = session.open_for_compute(_bundle(), p_gauge_raw=10 * GIB)
    closed = session.close()
    block = zz._gauge_swap_diagnostics_block(
        policy="bounded_swap",
        session=session,
        admission_decision=decision,
        open_event=opened,
        closed_event=closed,
        credit_applied=bool(opened is not None),
        gauge_workers_effective=1,
    )
    assert block is not None
    # JSON-safe, so it can be written into the manifest as-is.
    json.dumps(block)
    assert block["schema"] == "zemosaic-zegrid-gauge-swap-v1"
    assert block["policy"] == "bounded_swap"
    assert block["applicable"] is True
    events = [event["event"] for event in block["events"]]
    assert events == [zgcredit.EVENT_OPEN, zgcredit.EVENT_CLOSED]
    assert block["events"][-1]["closed"] is True
    # A PERMIT, never an observed usage, and never durable state.
    assert block["swap_used_bytes"] is None
    assert "never written to a store" in block["durable_persistence"]
    # Q3: ``open_for_compute`` re-reads the live facts, so the block publishes
    # the sources of the observation that actually armed the credit.
    assert block["sources"] == dict(session.capacity.facts)
    assert "platform" in block["sources"]
    assert block["admission"]["policy"] == cap.POLICY_BOUNDED_SWAP
    assert block["admission"]["s_credit_bytes"] >= 0


def test_o2_block_explains_a_not_applicable_policy(monkeypatch):
    from zemosaic import zemosaic_zegrid_mode as zz

    block = zz._gauge_swap_diagnostics_block(
        policy="bounded_swap",
        session=None,
        not_applicable_reason="the gauge is restored/imported",
    )
    assert block["applicable"] is False
    assert block["reason_code"] == "GAUGE_CREDIT_NOT_APPLICABLE"
    assert "restored" in block["reason"]
    assert block["events"] == []


def test_o1_credit_fields_are_additive_only_when_a_credit_was_applied():
    from zemosaic import zemosaic_zegrid_mode as zz

    session = _session()
    decision = session.decision_for_bundle(_bundle(), p_gauge_raw=10 * GIB)[0]
    off = zz._r55_gauge_credit_diagnostics_fields(credit_applied=False)
    assert off == {}
    fields = zz._r55_gauge_credit_diagnostics_fields(
        credit_applied=True,
        policy="bounded_swap",
        a_raw_bytes=decision.a_raw,
        b_ram_bytes=decision.b_ram,
        s_credit_bytes=decision.s_credit,
        credit_needed_bytes=decision.credit_needed_bytes,
        sources={"schema": "test-injected"},
    )
    assert fields["gauge_workers_effective"] == 1
    assert fields["gauge_credit_applied"] is True
    assert fields["gauge_memory_policy"] == "bounded_swap"
    assert fields["a_raw_bytes"] == decision.a_raw
    assert fields["b_ram_bytes"] == decision.b_ram
    assert fields["s_credit_bytes"] == decision.s_credit
    assert fields["credit_needed_bytes"] == decision.credit_needed_bytes
    assert fields["capacity_sources"] == {"schema": "test-injected"}


def test_o1_admission_record_absent_for_a_ram_only_admission():
    from zemosaic import zemosaic_zegrid_mode as zz

    assert zz._r55_gauge_credit_admission_record(
        policy="ram_only", decision=None
    ) is None

    session = _session()
    decision = session.decision_for_bundle(_bundle(), p_gauge_raw=10 * GIB)[0]
    record = zz._r55_gauge_credit_admission_record(
        policy="bounded_swap", decision=decision
    )
    json.dumps(record)
    assert record["policy"] == "bounded_swap"
    assert record["a_raw_bytes"] == decision.a_raw
    assert record["b_ram_bytes"] == decision.b_ram
    assert record["s_credit_bytes"] == decision.s_credit
    assert record["status"] == decision.status

    class _Obs:
        def to_dict(self):
            return {"raw": 1}

    class _Outcome:
        observations = ()
        history = ()
        replans = 0
        tolerance = False
        final_observation = _Obs()

    plain = zz._r55_admission_dict(_Outcome())
    assert "gauge_credit" not in plain
    credited = zz._r55_admission_dict(_Outcome(), gauge_credit=record)
    assert credited["gauge_credit"] == record


def test_manifest_publishes_the_block_only_for_bounded_swap():
    """Additive manifest key: absent for OFF, present as a first-class block."""
    source = _function_source(ENGINE_PATH, "_write_outputs")
    assert '"gauge_swap"' in source
    assert 'per_cell_diagnostics.get("gauge_swap") is not None' in source
    run_source = _function_source(ENGINE_PATH, "_run_single")
    assert "_gauge_swap_diagnostics_block(" in run_source
    assert '{"gauge_swap": _gauge_swap_block}' in run_source
    # O1: the same observation is handed to the active admission block.
    assert "gauge_credit=_gauge_swap_admission_record" in run_source


# ---------------------------------------------------------------------------
# 4. Stop / resume (§13.4)
# ---------------------------------------------------------------------------

def _registry(run_id="run-g2b"):
    from zemosaic.core.zegrid.run_registry import RunResultRegistry

    return RunResultRegistry(
        run_id=run_id,
        expected_cell_ids=["r0000c0000", "r0000c0001"],
        gauge_mode="global",
    )


def test_stop_during_the_gauge_acknowledges_a_boundary_and_keeps_no_credit():
    """Q6 (bounded witness): ack <= 30 s, exit path <= 120 s, no lost commit."""
    session = _session()
    opened = session.open_for_compute(_bundle(), p_gauge_raw=10 * GIB)
    assert opened is not None
    registry = _registry()

    requested_at = time.monotonic()
    registry.request_stop("stop requested during the gauge")
    # The gauge has NO durable commit, so the first safe boundary is the wave 0
    # checkpoint: the engine polls ``stop_requested`` there.
    acknowledged = False
    while time.monotonic() - requested_at <= 30.0:
        if registry.stop_requested:
            registry.acknowledge_stop(
                "STOPPED_RESUMABLE", "stop honoured before CPU wave 0"
            )
            acknowledged = True
            break
        time.sleep(0.001)
    ack_latency = time.monotonic() - requested_at
    assert acknowledged is True
    assert ack_latency <= 30.0
    assert registry.stop.acknowledged is True
    assert registry.stop.snapshot()["ack_state"] == "STOPPED_RESUMABLE"

    # The credit is closed in the ``finally`` path and is unreachable afterwards.
    closed = session.close()
    exit_latency = time.monotonic() - requested_at
    assert exit_latency <= 120.0
    assert closed["credit_granted_during_gauge"] is True
    assert closed["swap_used_bytes"] is None
    assert session.credit_for_compute() is None
    session.require_closed_for_stack()
    assert session.closed is True


def test_credit_path_has_no_write_api_at_all():
    """The credit can never be persisted as an acquired fact (§13.4)."""
    source = _source(CREDIT_PATH)
    for forbidden in (
        "write_text",
        "write_bytes",
        "atomic_write",
        "json.dump",
        "os.replace",
        "shutil.",
        "mkdir",
        "unlink",
    ):
        assert forbidden not in source, f"{forbidden} must not exist in {CREDIT_PATH.name}"
    # Only read-only access to the live facts.
    assert "read_text" in source


def test_a_policy_change_cannot_invalidate_a_durable_resume():
    """The policy is execution-only: no resume/store identity reads it."""
    for path in (RESUME_PATH, STORE_PATH):
        source = _source(path)
        assert "zegrid_gauge_memory_policy" not in source
        assert "gauge_credit" not in source
    # The scientific identity builder does not read the policy either.
    identity_source = _function_source(ENGINE_PATH, "_r55_build_run_store_identity")
    assert "zegrid_gauge_memory_policy" not in identity_source
    assert "gauge_credit" not in identity_source


def test_stop_does_not_enter_the_manifest_as_a_completed_publication():
    """A stop record never claims a publication (contract §13.4 wording)."""
    engine = _source(ENGINE_PATH)
    # The typed cooperative-stop carrier exists and its refusal record states
    # that no FITs/manifest/assembly was published.
    assert "_R55RunStopRequested" in engine
    assert '"publication": {' in engine
    assert '"fits": False' in engine
