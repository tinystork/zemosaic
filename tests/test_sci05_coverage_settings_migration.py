"""SCI-05 Gate E5a — coverage settings + legacy radial migration (deterministic).

Deterministic, hermetic tests for the canonical Coverage settings defaults, the
no-value-mapping legacy radial migration, and the seven locale files. No
randomness, no network, no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

from zemosaic import zemosaic_config as zc

_LOCALES = ("de", "en", "es", "fr", "is", "nl", "pl")
_NEW_LOCALE_KEYS = (
    "stacking_coverage_support_taper_label",
    "stacking_coverage_support_taper_note",
    "stacking_coverage_reconstruction_label",
    "stacking_coverage_reconstruction_note",
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_coverage_defaults_present_and_correct(self):
        assert zc.DEFAULT_CONFIG["coverage_support_taper"] is True
        assert zc.DEFAULT_CONFIG["coverage_aware_reconstruction"] is False

    def test_legacy_radial_keys_still_present(self):
        for key in ("apply_radial_weight", "radial_feather_fraction", "radial_shape_power"):
            assert key in zc.DEFAULT_CONFIG

    def test_no_taper_knob_exposed(self):
        # internal taper params (8.0 px / 0.0 floor) are NOT config keys
        for key in ("coverage_taper_px", "coverage_taper_floor", "taper_px",
                    "taper_floor", "feather_px", "footprint_floor"):
            assert key not in zc.DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

class TestMigration:
    def test_apply_radial_true(self):
        cfg, notes = zc.migrate_coverage_settings({"apply_radial_weight": True})
        assert cfg["coverage_support_taper"] is True
        assert cfg["coverage_aware_reconstruction"] is False
        assert cfg["apply_radial_weight"] is False  # legacy radial inert
        assert "legacy_radial:inert" in notes
        assert "coverage_support_taper:default_on" in notes

    def test_apply_radial_false_same_result(self):
        cfg, notes = zc.migrate_coverage_settings({"apply_radial_weight": False})
        assert cfg["coverage_support_taper"] is True
        assert cfg["coverage_aware_reconstruction"] is False
        assert cfg["apply_radial_weight"] is False

    def test_no_value_mapping(self):
        cfg, notes = zc.migrate_coverage_settings({
            "apply_radial_weight": True,
            "radial_feather_fraction": 0.99,
            "min_radial_weight_floor": 0.5,
            "radial_shape_power": 9.9,
        })
        # never map legacy values to a taper px/floor knob
        for key in ("coverage_taper_px", "coverage_taper_floor", "taper_px",
                    "taper_floor", "feather_px", "footprint_floor"):
            assert key not in cfg
        # legacy value keys remain readable (copied, not consumed)
        assert cfg.get("radial_feather_fraction") == 0.99
        assert cfg.get("min_radial_weight_floor") == 0.5
        assert cfg.get("radial_shape_power") == 9.9

    def test_idempotent(self):
        src = {"apply_radial_weight": True, "language": "en"}
        cfg1, notes1 = zc.migrate_coverage_settings(src)
        cfg2, notes2 = zc.migrate_coverage_settings(cfg1)
        assert cfg2 == cfg1
        assert notes2 == []

    def test_does_not_alter_unrelated_keys(self):
        src = {"apply_radial_weight": True, "language": "fr", "num_processing_workers": 3}
        cfg, _ = zc.migrate_coverage_settings(src)
        assert cfg["language"] == "fr"
        assert cfg["num_processing_workers"] == 3

    def test_non_dict_unchanged(self):
        cfg, notes = zc.migrate_coverage_settings("not a dict")
        assert cfg == "not a dict"
        assert notes == []

    def test_bounded_notes(self):
        _, notes = zc.migrate_coverage_settings({"apply_radial_weight": True})
        assert isinstance(notes, list)
        assert all(isinstance(n, str) for n in notes)

    def test_no_input_mutation(self):
        src = {"apply_radial_weight": True, "language": "en"}
        before = dict(src)
        zc.migrate_coverage_settings(src)
        assert src == before  # caller's dict untouched


# ---------------------------------------------------------------------------
# Config round-trip (load_config)
# ---------------------------------------------------------------------------

class TestConfigRoundTrip:
    def test_load_config_yields_new_keys_and_migrates(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "zemosaic_config.json"
        cfg_file.write_text(
            json.dumps({"apply_radial_weight": True, "language": "en"}), encoding="utf-8"
        )
        monkeypatch.setattr(zc, "get_config_path", lambda: str(cfg_file))
        cfg = zc.load_config()
        assert cfg["coverage_support_taper"] is True
        assert cfg["coverage_aware_reconstruction"] is False
        assert cfg["apply_radial_weight"] is False  # migrated inert
        # legacy keys remain readable
        assert "radial_feather_fraction" in cfg
        assert "radial_shape_power" in cfg

    def test_load_config_no_file_uses_defaults(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "nonexistent.json"
        monkeypatch.setattr(zc, "get_config_path", lambda: str(cfg_file))
        cfg = zc.load_config()  # must not raise
        assert cfg["coverage_support_taper"] is True
        assert cfg["coverage_aware_reconstruction"] is False


# ---------------------------------------------------------------------------
# Locales
# ---------------------------------------------------------------------------

class TestLocales:
    def test_all_locales_valid_json_with_new_keys(self):
        locales_dir = Path(zc.__file__).parent / "locales"
        for lang in _LOCALES:
            p = locales_dir / f"{lang}.json"
            data = json.loads(p.read_text(encoding="utf-8"))  # valid JSON
            for key in _NEW_LOCALE_KEYS:
                assert key in data, f"{lang}: missing {key}"
                assert isinstance(data[key], str) and data[key].strip(), (
                    f"{lang}: empty {key}"
                )
