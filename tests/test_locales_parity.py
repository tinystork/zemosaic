"""Locale catalogue parity against the English reference (ZM-I18N-R55D2-DEBT).

Every shipped locale must expose exactly the same key set as ``en.json``:
no missing key (which silently falls back to English at runtime) and no
orphan key (dead translation debt).

This test intentionally fails on the pre-fix baseline, where ``fr`` missed 15
keys and ``de``/``es``/``is``/``nl``/``pl`` missed 72 keys each.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

LOCALES_DIR = Path(__file__).resolve().parents[1] / "src" / "zemosaic" / "locales"
REFERENCE = "en"
LOCALES = ("fr", "de", "es", "is", "nl", "pl")


def _load(locale: str) -> dict:
    path = LOCALES_DIR / f"{locale}.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def reference() -> dict:
    return _load(REFERENCE)


@pytest.mark.parametrize("locale", LOCALES)
def test_locale_is_valid_json(locale: str) -> None:
    """Each catalogue parses; a broken file must fail loudly, not silently."""
    assert isinstance(_load(locale), dict)


@pytest.mark.parametrize("locale", LOCALES)
def test_locale_has_exactly_the_reference_keys(locale: str, reference: dict) -> None:
    """No missing key (English fallback) and no orphan key (dead debt)."""
    catalogue = _load(locale)

    missing = sorted(set(reference) - set(catalogue))
    extra = sorted(set(catalogue) - set(reference))

    assert not missing, f"{locale}.json is missing {len(missing)} key(s): {missing}"
    assert not extra, f"{locale}.json has {len(extra)} orphan key(s): {extra}"


@pytest.mark.parametrize("locale", LOCALES)
def test_locale_values_are_non_empty_strings(locale: str) -> None:
    """A present-but-empty value would be worse than a missing key."""
    catalogue = _load(locale)

    empty = sorted(key for key, value in catalogue.items() if value is None)
    blank = sorted(key for key, value in catalogue.items() if isinstance(value, str) and not value.strip())
    wrong_type = sorted(key for key, value in catalogue.items() if not isinstance(value, str))

    assert not empty, f"{locale}.json has null value(s): {empty}"
    assert not blank, f"{locale}.json has empty value(s): {blank}"
    assert not wrong_type, f"{locale}.json has non-string value(s): {wrong_type}"
