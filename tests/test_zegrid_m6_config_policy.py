"""ZM-ZEGRID-M6 Part A — declared run-policy keys and strict validation.

Covers:
* the recovery/store/gauge/layout/SIP keys are declared in ``DEFAULT_CONFIG``
  with exactly the historical defaults (recovery OFF, resume never);
* save/load round-trip preserves explicit values;
* invalid values are rejected with an explicit error, never coerced;
* ``load_config`` (the single config-load path) enforces the same validation.
"""

from __future__ import annotations

import json

import pytest

from zemosaic import zemosaic_config as zc

EXPECTED_DEFAULTS = {
    "zegrid_resource_recovery_enabled": False,
    "zegrid_resource_recovery_deadline_s": None,
    "zegrid_durable_store_budget_bytes": None,
    "zegrid_cell_gauge_mode": None,
    "zegrid_layout": None,
    "zegrid_sip_mode": "keep",
}


def test_policy_defaults_are_declared_and_preserve_today_behaviour():
    for key, expected in EXPECTED_DEFAULTS.items():
        assert key in zc.DEFAULT_CONFIG, f"{key} must be declared in DEFAULT_CONFIG"
        assert zc.DEFAULT_CONFIG[key] == expected
    # The pre-existing defaults this mission must not change.
    assert zc.DEFAULT_CONFIG["zegrid_resume_mode"] == "never"
    assert zc.DEFAULT_CONFIG["zegrid_durable_store_dir"] is None


def test_validate_accepts_defaults_and_none():
    resolved = zc.validate_zegrid_run_policy(zc.DEFAULT_CONFIG)
    assert resolved == EXPECTED_DEFAULTS
    # An attribute-style config (SimpleNamespace) and None resolve identically.
    from types import SimpleNamespace

    assert zc.validate_zegrid_run_policy(SimpleNamespace()) == EXPECTED_DEFAULTS
    assert zc.validate_zegrid_run_policy(None) == EXPECTED_DEFAULTS


@pytest.mark.parametrize(
    "key,value",
    [
        ("zegrid_resource_recovery_enabled", "false"),
        ("zegrid_resource_recovery_enabled", 0),
        ("zegrid_resource_recovery_enabled", 1),
        ("zegrid_resource_recovery_deadline_s", 0),
        ("zegrid_resource_recovery_deadline_s", -5.0),
        ("zegrid_resource_recovery_deadline_s", True),
        ("zegrid_resource_recovery_deadline_s", "60"),
        ("zegrid_resource_recovery_deadline_s", float("nan")),
        ("zegrid_resource_recovery_deadline_s", float("inf")),
        ("zegrid_durable_store_budget_bytes", 0),
        ("zegrid_durable_store_budget_bytes", -1),
        ("zegrid_durable_store_budget_bytes", True),
        ("zegrid_durable_store_budget_bytes", 1.5),
        ("zegrid_durable_store_budget_bytes", "1024"),
        ("zegrid_cell_gauge_mode", "per-cell"),
        ("zegrid_cell_gauge_mode", True),
        ("zegrid_layout", 2),
        ("zegrid_layout", (2, 2)),
        ("zegrid_sip_mode", "stripped"),
        ("zegrid_sip_mode", True),
    ],
)
def test_invalid_values_are_rejected_never_coerced(key, value):
    cfg = dict(zc.DEFAULT_CONFIG)
    cfg[key] = value
    with pytest.raises(ValueError) as excinfo:
        zc.validate_zegrid_run_policy(cfg)
    assert key in str(excinfo.value)


def test_explicit_valid_values_round_trip(tmp_path, monkeypatch):
    cfg_path = tmp_path / "zemosaic_config.json"
    monkeypatch.setattr(zc, "get_config_path", lambda: str(cfg_path))
    cfg = dict(zc.DEFAULT_CONFIG)
    cfg.update(
        {
            "zegrid_resource_recovery_enabled": True,
            "zegrid_resource_recovery_deadline_s": 12.5,
            "zegrid_durable_store_budget_bytes": 2 * 2**30,
            "zegrid_cell_gauge_mode": "per_cell",
            "zegrid_layout": "2x2",
            "zegrid_sip_mode": "strip",
            "zegrid_resume_mode": "auto",
            "zegrid_resume_run_id": "run-abc",
            "zegrid_durable_store_dir": str(tmp_path / "store"),
        }
    )
    assert zc.save_config(cfg) is True

    loaded = zc.load_config()
    assert loaded["zegrid_resource_recovery_enabled"] is True
    assert loaded["zegrid_resource_recovery_deadline_s"] == 12.5
    assert loaded["zegrid_durable_store_budget_bytes"] == 2 * 2**30
    assert loaded["zegrid_cell_gauge_mode"] == "per_cell"
    assert loaded["zegrid_layout"] == "2x2"
    assert loaded["zegrid_sip_mode"] == "strip"
    assert loaded["zegrid_resume_mode"] == "auto"
    assert loaded["zegrid_resume_run_id"] == "run-abc"
    assert loaded["zegrid_durable_store_dir"] == str(tmp_path / "store")


def test_load_config_rejects_a_persisted_invalid_value(tmp_path, monkeypatch):
    cfg_path = tmp_path / "zemosaic_config.json"
    payload = dict(zc.DEFAULT_CONFIG)
    payload["zegrid_resource_recovery_enabled"] = "yes"
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(zc, "get_config_path", lambda: str(cfg_path))

    with pytest.raises(ValueError) as excinfo:
        zc.load_config()
    assert "zegrid_resource_recovery_enabled" in str(excinfo.value)
