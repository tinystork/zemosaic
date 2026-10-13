"""ZM-SDS-R35C-C3: robust SDS inter-megatile photometry + honest options.

Synthetic gates for the Lot C3 correction:

* ``none`` is a genuinely neutral no-op.
* ``sky_mean`` removes an added background offset WITHOUT distorting scale.
* null / negative medians never produce an ``abs()``-driven multiplier or a sign flip.
* bright-source (star) and weak-signal levels are preserved.
* the alignment uses only the valid COMMON overlap (coverage masks respected).
* unsupported methods are explicit; requested/effective is recorded durably.

These tests do NOT rerun M106; they exercise the pure helpers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import zemosaic.zemosaic_worker as zw  # noqa: E402


def _recording_pcb():
    events = []

    def _cb(message, prog=None, lvl=None, **kwargs):
        events.append({"message": message, "lvl": lvl, **kwargs})

    return _cb, events


def _field(seed=7, shape=(32, 32, 3), sigma=5.0):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, sigma, shape).astype(np.float32)


def test_none_is_truly_neutral():
    ref = _field(1)
    src = (ref + 123.0).astype(np.float32)
    out = zw._normalize_sds_megatiles_photometry(
        [ref, src], [None, None], ref_index=0, method="none"
    )
    assert out[0].dtype == np.float32 and out[1].dtype == np.float32
    assert np.array_equal(out[0], ref)
    assert np.array_equal(out[1], src)
    # returned arrays are copies, not the inputs
    assert out[0] is not ref and out[1] is not src


def test_sky_mean_removes_added_background_offset():
    base = _field(2)
    ref = base.copy()
    offset = 47.5
    src = (base + offset).astype(np.float32)
    out = zw._normalize_sds_megatiles_photometry(
        [ref, src], [None, None], ref_index=0, method="sky_mean"
    )
    residual = out[1] - base
    assert abs(float(np.mean(residual))) < 1e-3
    assert float(np.std(residual)) < 1e-3
    # scale is preserved, not rescaled by a median ratio
    assert float(np.std(out[1])) == pytest.approx(float(np.std(base)), rel=0.02)


def test_scale_is_not_derived_from_signal_level():
    """A tile whose signal is 3x brighter must NOT be multiplicatively flattened.

    The legacy contract scaled by |ref_median|/|tile_median|; a level/offset contract
    must preserve relative scale and only remove a common level difference.
    """
    ref = _field(3)
    brighter = (ref * 3.0).astype(np.float32)
    out = zw._normalize_sds_megatiles_photometry(
        [ref, brighter], [None, None], ref_index=0, method="sky_mean"
    )
    assert float(np.std(out[1])) == pytest.approx(float(np.std(brighter)), rel=0.05)
    # a pure additive alignment cannot turn the 3x scale into a 1x scale
    assert float(np.std(out[1])) > 2.0 * float(np.std(out[0]))


def test_null_and_negative_medians_not_flipped_or_rescaled():
    rng = np.random.default_rng(4)
    base = rng.normal(0.0, 5.0, (32, 32, 1)).astype(np.float32)
    # tile with a NEGATIVE median: must not be sign-flipped nor multiplied by abs-gain
    negative = (base - 0.5).astype(np.float32)
    out = zw._normalize_sds_megatiles_photometry(
        [base, negative], [None, None], ref_index=0, method="sky_mean"
    )
    assert float(np.std(out[1])) == pytest.approx(float(np.std(negative)), rel=0.05)
    # near-zero median tile must not blow up (no multiplicative gain at all)
    near_zero = (base * 1e-6).astype(np.float32)
    out2 = zw._normalize_sds_megatiles_photometry(
        [base, near_zero], [None, None], ref_index=0, method="sky_mean"
    )
    assert np.all(np.isfinite(out2[1]))
    assert float(np.max(np.abs(out2[1]))) < 1.0


def test_star_and_weak_signal_levels_preserved():
    rng = np.random.default_rng(5)
    sky = rng.normal(0.0, 2.0, (40, 40, 2)).astype(np.float32)
    # a bright "star" and a faint source on the reference tile
    star = sky.copy()
    star[10:14, 10:14, 0] += 500.0
    star[30:32, 30:32, 1] += 8.0
    faint = sky.copy()
    faint[30:32, 30:32, 1] += 8.0
    shifted = (star + 25.0).astype(np.float32)  # same star, +25 ADU background
    out = zw._normalize_sds_megatiles_photometry(
        [star, shifted], [None, None], ref_index=0, method="sky_mean"
    )
    aligned = out[1]
    # star peak preserved (no multiplicative distortion)
    assert float(np.max(aligned[..., 0])) == pytest.approx(float(np.max(star[..., 0])), rel=0.02)
    # weak signal preserved too
    assert float(aligned[30, 30, 1]) == pytest.approx(float(star[30, 30, 1]), abs=1.0)
    assert float(np.std(aligned)) == pytest.approx(float(np.std(star)), rel=0.02)
    # faint reference tile also participates without an abs() gain
    assert np.all(np.isfinite(aligned))


def test_alignment_uses_only_valid_common_overlap():
    ref = _field(6, shape=(32, 32, 1))
    cov_ref = np.ones((32, 32), dtype=np.float32)
    cov_ref[:, 16:] = 0.0  # reference valid only on the LEFT half
    src = (ref + 30.0).astype(np.float32)
    cov_src = np.ones((32, 32), dtype=np.float32)
    out = zw._normalize_sds_megatiles_photometry(
        [ref, src], [cov_ref, cov_src], ref_index=0, method="sky_mean"
    )
    # offset estimated on the common (left) half, then applied to the whole source tile
    residual = out[1] - ref
    assert abs(float(np.mean(residual[:, :16, 0]))) < 1e-3
    assert abs(float(np.mean(residual[:, 16:, 0]))) < 1e-3


def test_unsupported_method_is_explicit():
    ref = _field(8)
    with pytest.raises(ValueError):
        zw._normalize_sds_megatiles_photometry([ref], [None], method="linear_fit")
    with pytest.raises(ValueError):
        zw._normalize_sds_megatiles_photometry([ref], [None], method="noise_variance")


def test_requested_effective_option_mapping():
    assert zw._resolve_sds_megatile_photometry_method({"stack_norm_method": "none"}) == "none"
    assert zw._resolve_sds_megatile_photometry_method({"stack_norm_method": "sky_mean"}) == "sky_mean"
    assert zw._resolve_sds_megatile_photometry_method({"stack_norm_method": ""}) == "sky_mean"
    # unsupported requested method -> neutral, and SAID explicitly (not silent)
    pcb, events = _recording_pcb()
    eff = zw._resolve_sds_megatile_photometry_method(
        {"stack_norm_method": "noise_variance"}, pcb
    )
    assert eff == "none"
    assert events and events[0]["message"] == "sds_megatile_photometry_method"
    assert events[0]["effective"] == "none"
    assert "unsupported" in str(events[0]["note"])


def test_option_contract_is_recorded():
    pcb, events = _recording_pcb()
    zw._emit_sds_option_contract(
        pcb,
        stage="megatiles_to_final",
        requested={"normalization": "sky_mean", "weighting": "noise_variance", "rgb_equalize": False},
        effective={"normalization": "sky_mean", "weighting": "none", "rgb_equalize": False},
        deviations={"weighting": "sds_final_canonical_imposes_none"},
    )
    assert len(events) == 1
    ev = events[0]
    assert ev["message"] == "sds_option_contract"
    assert ev["stage"] == "megatiles_to_final"
    assert ev["requested"]["normalization"] == "sky_mean"
    assert ev["effective"]["weighting"] == "none"
    assert ev["deviations"]["weighting"] == "sds_final_canonical_imposes_none"


def test_option_contract_logging_is_symmetric():
    """rework-0: the SDS requested/effective contract must be emitted in EVERY SDS path."""
    import ast

    src = Path(zw.__file__).read_text(encoding="utf-8-sig")
    lines = src.splitlines()
    wanted = {"run_hierarchical_mosaic", "run_hierarchical_mosaic_classic_legacy"}
    found: dict[str, bool] = {}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            seg = "\n".join(lines[node.lineno - 1 : node.end_lineno])
            found[node.name] = "_emit_sds_option_contract(" in seg
    assert found.get("run_hierarchical_mosaic"), "modern SDS path must emit the option contract"
    assert found.get(
        "run_hierarchical_mosaic_classic_legacy"
    ), "legacy SDS path must emit the option contract (symmetric logging)"


def test_level_align_event_records_requested_method():
    pcb, events = _recording_pcb()
    base = _field(9)
    src = (base + 12.0).astype(np.float32)
    zw._normalize_sds_megatiles_photometry(
        [base, src], [None, None], ref_index=0, method="sky_mean", pcb=pcb
    )
    align = [e for e in events if e["message"] == "[SDS] sds_megatile_level_align"]
    assert len(align) == 2
    assert all(e["method"] == "sky_mean" for e in align)
    # the non-reference tile records a non-empty offset vector and a level change
    moved = align[1]
    assert moved["tile_index"] == 1
    assert len(moved["offsets"]) == base.shape[-1]
