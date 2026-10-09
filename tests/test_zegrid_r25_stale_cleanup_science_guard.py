"""ZM-ZEGRID-R25 (L1) stale-cleanup science guard tests.

Non-blocking LOW hardening closing the Nono advisory L1 from ZM-ZEGRID-R24
(review-0): ``_cleanup_stale_aesthetic`` must NEVER delete a raw science output.

The guard is a pure-policy function; these tests exercise it directly with
hand-edited/corrupt manifests (the exact attack surface Junior reproduced) and
assert the REAL returned record (``reason`` / ``removed`` / ``kept`` / warnings),
so they are genuinely discriminating — not vacuous smoke.

Covered:

* ``mosaic_grid_science.fits`` (prior-declared, not current)      -> REFUSED
* the prior manifest's OWN scientific-suffix name                 -> REFUSED
* a legitimate changed / OFF aesthetic                            -> REMOVED
* every R24 adversarial name still refused                        -> REFUSED
* contract ``role != "AESTH"`` never yields a deletable name      -> REFUSED
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zemosaic import zemosaic_zegrid_mode as zz


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _manifest(tmp_path, obj):
    (tmp_path / "zegrid_manifest.json").write_text(json.dumps(obj))
    return tmp_path / "zegrid_manifest.json"


def _touch(tmp_path, name, data=b"x"):
    p = tmp_path / name
    p.write_bytes(data)
    return p


# ---------------------------------------------------------------------------
# Rule 1 — scientific-pattern names are never removed
# ---------------------------------------------------------------------------

def test_science_suffix_default_name_refused(tmp_path):
    """A prior manifest declaring ``mosaic_grid_science.fits`` as its aesthetic
    must NOT delete it (default scientific suffix ``_science`` always honoured).
    """
    _touch(tmp_path, "mosaic_grid_science.fits")
    _manifest(tmp_path, {"outputs": {"aesthetic": "mosaic_grid_science.fits"}})

    rec = zz._cleanup_stale_aesthetic(tmp_path, None, protected_names=())

    assert rec["removed"] == []
    assert rec["reason"] == "invalid_prior_declaration"
    assert "mosaic_grid_science.fits" in rec["kept"]
    assert any("scientific output name" in w for w in rec["warnings"])
    assert (tmp_path / "mosaic_grid_science.fits").exists()


def test_science_suffix_prior_declared_suffix_refused(tmp_path):
    """The prior manifest's OWN scientific suffix (recorded in its contract) is
    honoured, even when it is not the default ``_science``.
    """
    _touch(tmp_path, "mosaic_grid_newsci.fits")
    _manifest(tmp_path, {
        "outputs": {"aesthetic": "mosaic_grid_newsci.fits",
                    "science": "mosaic_grid_newsci.fits"},
        "science_reference": "mosaic_grid_newsci.fits",
        "science_output_contract": {
            "raw": {"file": "mosaic_grid_newsci.fits", "role": "SCI",
                    "dtype": "float32", "sha256": "0" * 64},
        },
    })

    rec = zz._cleanup_stale_aesthetic(tmp_path, None, protected_names=())

    assert rec["removed"] == []
    assert rec["reason"] == "invalid_prior_declaration"
    assert "mosaic_grid_newsci.fits" in rec["kept"]
    assert (tmp_path / "mosaic_grid_newsci.fits").exists()


def test_current_scientific_suffix_explicitly_honoured(tmp_path):
    """The current run's resolved scientific suffix is honoured explicitly, even
    when the (hand-edited) prior manifest records NO science of its own.
    """
    _touch(tmp_path, "mosaic_grid_mysci.fits")
    _manifest(tmp_path, {"outputs": {"aesthetic": "mosaic_grid_mysci.fits"}})

    rec = zz._cleanup_stale_aesthetic(
        tmp_path, None, protected_names=(), scientific_suffix="_mysci"
    )

    assert rec["removed"] == []
    assert rec["reason"] == "invalid_prior_declaration"
    assert (tmp_path / "mosaic_grid_mysci.fits").exists()


def test_science_pattern_refused_even_when_not_current_output(tmp_path):
    """The scientific-pattern refusal is independent of the protected-names
    (current-output) collision check: a science-suffixed name that is NOT one of
    the current outputs is still refused (the exact L1 gap Junior reproduced).
    """
    _touch(tmp_path, "mosaic_grid_science.fits")
    _manifest(tmp_path, {"outputs": {"aesthetic": "mosaic_grid_science.fits"}})

    # protected_names is EMPTY — exactly the reproduced call — yet refused.
    rec = zz._cleanup_stale_aesthetic(tmp_path, None, protected_names=())

    assert rec["removed"] == []
    assert rec["reason"] == "invalid_prior_declaration"
    assert (tmp_path / "mosaic_grid_science.fits").exists()


# ---------------------------------------------------------------------------
# Rule 2 — the contract's aesthetic record must really be an aesthetic
# ---------------------------------------------------------------------------

def test_contract_aesthetic_wrong_role_not_deleted(tmp_path):
    """``science_output_contract.aesthetic`` with a non-AESTH role must not be
    used as a deletable prior-aesthetic name (and its science-suffixed file is
    protected by rule 1 anyway).
    """
    _touch(tmp_path, "mosaic_grid_foo.fits")
    _manifest(tmp_path, {
        "science_output_contract": {
            "aesthetic": {"file": "mosaic_grid_foo.fits", "role": "SCI"},
        },
    })

    rec = zz._cleanup_stale_aesthetic(tmp_path, None, protected_names=())

    assert rec["removed"] == []
    # role != AESTH -> no prior aesthetic is trusted.
    assert rec["reason"] == "no_prior_aesthetic_declared"
    assert (tmp_path / "mosaic_grid_foo.fits").exists()


def test_contract_aesthetic_role_aesth_still_used_for_legit_removal(tmp_path):
    """The contract's AESTH record remains a legitimate prior-aesthetic source:
    a genuine stale aesthetic declared there (role AESTH) is still removed.
    """
    _touch(tmp_path, "mosaic_grid_aesthetic.fits")
    _manifest(tmp_path, {
        "science_output_contract": {
            "raw": {"file": "mosaic_grid_science.fits", "role": "SCI"},
            "aesthetic": {"file": "mosaic_grid_aesthetic.fits", "role": "AESTH"},
        },
    })

    rec = zz._cleanup_stale_aesthetic(tmp_path, None, protected_names=())

    assert rec["removed"] == ["mosaic_grid_aesthetic.fits"]
    assert not (tmp_path / "mosaic_grid_aesthetic.fits").exists()


def test_outputs_aesthetic_science_name_refused_even_if_contract_contradicts(tmp_path):
    """A hand-edited ``outputs.aesthetic`` naming a science file is refused even
    when the contract's aesthetic record names something else (contradiction).
    """
    _touch(tmp_path, "mosaic_grid_science.fits")
    _manifest(tmp_path, {
        "outputs": {"aesthetic": "mosaic_grid_science.fits"},
        "science_output_contract": {
            "raw": {"file": "mosaic_grid_science.fits", "role": "SCI"},
            "aesthetic": {"file": "mosaic_grid_aesthetic.fits", "role": "AESTH"},
        },
    })

    rec = zz._cleanup_stale_aesthetic(tmp_path, None, protected_names=())

    assert rec["removed"] == []
    assert rec["reason"] == "invalid_prior_declaration"
    assert (tmp_path / "mosaic_grid_science.fits").exists()


# ---------------------------------------------------------------------------
# Rule 3 — the legitimate case still works
# ---------------------------------------------------------------------------

def test_legit_changed_aesthetic_removed(tmp_path):
    """Export checkbox OFF (current_aesthetic_path=None) -> prior aesthetic
    removed exactly as today.
    """
    _touch(tmp_path, "mosaic_grid_ae.fits")
    _manifest(tmp_path, {"outputs": {"aesthetic": "mosaic_grid_ae.fits"}})

    rec = zz._cleanup_stale_aesthetic(
        tmp_path, None,
        protected_names={"mosaic_grid.fits", "mosaic_grid_coverage.fits",
                         "zegrid_run.log"},
    )

    assert rec["removed"] == ["mosaic_grid_ae.fits"]
    assert not (tmp_path / "mosaic_grid_ae.fits").exists()


def test_legit_suffix_changed_aesthetic_removed(tmp_path):
    """Aesthetic suffix changed -> prior suffix removed, current kept.
    """
    _touch(tmp_path, "mosaic_grid_ae.fits")
    _touch(tmp_path, "mosaic_grid_ae2.fits")
    _manifest(tmp_path, {"outputs": {"aesthetic": "mosaic_grid_ae.fits"}})

    rec = zz._cleanup_stale_aesthetic(
        tmp_path, tmp_path / "mosaic_grid_ae2.fits",
        protected_names={"mosaic_grid.fits", "mosaic_grid_coverage.fits",
                         "mosaic_grid_ae2.fits", "zegrid_run.log"},
    )

    assert rec["removed"] == ["mosaic_grid_ae.fits"]
    assert not (tmp_path / "mosaic_grid_ae.fits").exists()
    assert (tmp_path / "mosaic_grid_ae2.fits").exists()


# ---------------------------------------------------------------------------
# Rule 4 — all R24 adversarial guards remain
# ---------------------------------------------------------------------------

def test_r24_adversarial_outside_path_refused(tmp_path):
    _manifest(tmp_path, {"outputs": {"aesthetic": "../evil.fits"}})
    rec = zz._cleanup_stale_aesthetic(tmp_path, None, protected_names=())
    assert rec["removed"] == []
    assert rec["reason"] == "invalid_prior_declaration"
    assert "../evil.fits" in rec["kept"]


def test_r24_adversarial_absolute_path_refused(tmp_path):
    _manifest(tmp_path, {"outputs": {"aesthetic": "/etc/passwd"}})
    rec = zz._cleanup_stale_aesthetic(tmp_path, None, protected_names=())
    assert rec["removed"] == []
    assert rec["reason"] == "invalid_prior_declaration"


def test_r24_adversarial_subdir_path_refused(tmp_path):
    _manifest(tmp_path, {"outputs": {"aesthetic": "sub/evil.fits"}})
    rec = zz._cleanup_stale_aesthetic(tmp_path, None, protected_names=())
    assert rec["removed"] == []
    assert rec["reason"] == "invalid_prior_declaration"


def test_r24_reserved_bare_science_name_refused(tmp_path):
    _touch(tmp_path, "mosaic_grid.fits")
    _manifest(tmp_path, {"outputs": {"aesthetic": "mosaic_grid.fits"}})
    rec = zz._cleanup_stale_aesthetic(
        tmp_path, None,
        protected_names={"mosaic_grid_coverage.fits", "zegrid_run.log"},
    )
    assert rec["removed"] == []
    assert rec["reason"] == "invalid_prior_declaration"
    assert (tmp_path / "mosaic_grid.fits").exists()


def test_r24_reserved_coverage_name_refused(tmp_path):
    _touch(tmp_path, "mosaic_grid_coverage.fits")
    _manifest(tmp_path, {"outputs": {"aesthetic": "mosaic_grid_coverage.fits"}})
    rec = zz._cleanup_stale_aesthetic(tmp_path, None, protected_names=())
    assert rec["removed"] == []
    assert rec["reason"] == "invalid_prior_declaration"
    assert (tmp_path / "mosaic_grid_coverage.fits").exists()


def test_r24_reserved_uint16_name_refused(tmp_path):
    _touch(tmp_path, "mosaic_grid_uint16.fits")
    _manifest(tmp_path, {"outputs": {"aesthetic": "mosaic_grid_uint16.fits"}})
    rec = zz._cleanup_stale_aesthetic(tmp_path, None, protected_names=())
    assert rec["removed"] == []
    assert rec["reason"] == "invalid_prior_declaration"
    assert (tmp_path / "mosaic_grid_uint16.fits").exists()


def test_r24_reserved_run_log_name_refused(tmp_path):
    _touch(tmp_path, "zegrid_run.log")
    _manifest(tmp_path, {"outputs": {"aesthetic": "zegrid_run.log"}})
    rec = zz._cleanup_stale_aesthetic(tmp_path, None, protected_names=())
    assert rec["removed"] == []
    assert rec["reason"] == "invalid_prior_declaration"
    assert (tmp_path / "zegrid_run.log").exists()


def test_r24_missing_manifest_noop(tmp_path):
    rec = zz._cleanup_stale_aesthetic(tmp_path, None, protected_names=())
    assert rec["removed"] == []
    assert rec["reason"] == "no_prior_manifest"


def test_r24_corrupt_manifest_noop(tmp_path):
    (tmp_path / "zegrid_manifest.json").write_text("{not valid json")
    rec = zz._cleanup_stale_aesthetic(tmp_path, None, protected_names=())
    assert rec["removed"] == []
    assert rec["reason"] == "prior_manifest_unreadable"


def test_r24_non_object_manifest_noop(tmp_path):
    _manifest(tmp_path, ["a", "list", "not", "an", "object"])
    rec = zz._cleanup_stale_aesthetic(tmp_path, None, protected_names=())
    assert rec["removed"] == []
    assert rec["reason"] == "prior_manifest_not_object"


def test_r24_no_prior_aesthetic_declared_noop(tmp_path):
    _manifest(tmp_path, {"outputs": {"aesthetic": None}})
    rec = zz._cleanup_stale_aesthetic(tmp_path, None, protected_names=())
    assert rec["removed"] == []
    assert rec["reason"] == "no_prior_aesthetic_declared"


def test_r24_current_output_collision_refused(tmp_path):
    """A declared prior name equal to a current output name is never removed.
    """
    _touch(tmp_path, "mosaic_grid_ae.fits")
    _manifest(tmp_path, {"outputs": {"aesthetic": "mosaic_grid_ae.fits"}})
    rec = zz._cleanup_stale_aesthetic(
        tmp_path, tmp_path / "mosaic_grid_ae.fits",
        protected_names={"mosaic_grid_ae.fits"},
    )
    # Equal to current aesthetic -> prior_aesthetic_is_current (kept, not removed).
    assert rec["removed"] == []
    assert rec["reason"] == "prior_aesthetic_is_current"
    assert (tmp_path / "mosaic_grid_ae.fits").exists()


def test_r25_never_raises(tmp_path):
    """The cleaner records failures and never raises on a removal error path.
    """
    _touch(tmp_path, "mosaic_grid_ae.fits")
    _manifest(tmp_path, {"outputs": {"aesthetic": "mosaic_grid_ae.fits"}})
    # Patch unlink to raise, mirroring the auditable-failure contract.
    import os
    orig_unlink = Path.unlink

    def _boom(self, *a, **k):
        raise OSError("boom")

    Path.unlink = _boom
    try:
        rec = zz._cleanup_stale_aesthetic(tmp_path, None, protected_names=())
    finally:
        Path.unlink = orig_unlink

    assert rec["removed"] == []
    assert rec["reason"] == "removal_failed"
    assert any("could not remove" in w for w in rec["warnings"])
    assert "mosaic_grid_ae.fits" in rec["kept"]
