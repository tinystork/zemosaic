"""Cheap protocol-consistency checks for the M3 Linux pressure protocol.

These tests do not import the product, allocate pressure or run a mosaic. They
guard the F4 ordering contract that the preflight itself cannot exercise.
"""

from __future__ import annotations

from pathlib import Path

PROTOCOL = (
    Path(__file__).resolve().parents[1]
    / "docs/refactor/ZEGRID_M3_LINUX_PRESSURE_PROTOCOL.md"
)


def _normalized() -> str:
    return " ".join(PROTOCOL.read_text(encoding="utf-8").split())


def test_handshake_precedes_the_original_safe_wave_sampler():
    text = _normalized()
    start = text.index("Correct order for the designated safe wave")
    section = text[start : start + 1800]
    hook = section.index("safe_wave_boundary_hook")
    sampler = section.index(
        "original, unmodified `facts = _r55_run_checkpoint_sample(disk_root)`"
    )
    checkpoint = section.index("original, unmodified `_r55_run_checkpoint")
    assert hook < sampler < checkpoint
    assert ":11681" in text
    assert "Wrapping only `_r55_run_checkpoint` is too late" in text
    assert "there is no later re-selection and no re-arm" in text


def test_other_sampler_sites_are_explicitly_excluded():
    text = _normalized()
    assert "Do not arm at:" in text
    for site in (":11576", ":11405", ":11919", ":12705", ":13036"):
        assert site in text
    assert "heartbeat" in text
    assert "M3 coordinator probe" in text
    assert "M3 post-wait re-evaluation sampler" in text
    assert "GPU inter-cell/unit samplers" in text
    assert "If attribution is ambiguous or the designated call is missed" in text
    assert "the wave is `NOT_RUN`" in text
    assert "Arm exactly once; never re-arm during wait/recovery/heartbeat" in text


def test_one_shot_bounded_ipc_and_transparency_are_mandatory():
    text = _normalized()
    assert "One-shot bounded IPC" in text
    assert "parent observer owns pressure" in text
    assert "<=2 s" in text
    assert "On timeout or refusal: release pressure" in text
    assert "classify `NOT_RUN`" in text
    assert "Never fabricate, rewrite, clamp, normalize or post-process" in text
    assert "original sampled facts" in text
    assert "checkpoint result" in text


def test_pass_requires_barrier_trace_and_post_arm_governing_fact():
    text = _normalized()
    assert "hook < pressure_applied <= sampler" in text
    assert "Barrier before the governing sampler" in text
    assert "Governing fact post-arm" in text
    assert "barrier-trace.jsonl" in text
    assert "latched_host.monotonic_ns >= pressure_applied" in text
    assert "host_source" in text
    assert "otherwise `NOT_RUN`" in text


def test_residual_race_is_stated_as_scenario_arithmetic():
    text = _normalized()
    assert "Residual race, stated honestly" in text
    assert "A-P" in text or "`A-P`" in text
    assert "scenario arithmetic" in text
    assert "not proof that pressure RSS equals" in text
    assert "physical `MemAvailable` drop" in text
