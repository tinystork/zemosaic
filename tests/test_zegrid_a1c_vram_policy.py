"""A1c — one VRAM policy for the legacy tiled planner and ACTIVE readiness."""

from __future__ import annotations

import pytest

from zemosaic.core.zegrid import gpu
from zemosaic.core.zegrid import gpu_readiness as gr

MIB = 2**20


def test_one_percent_is_shared_by_both_paths():
    assert gpu.GPU_VRAM_SAFETY_PERCENT == 70
    assert gpu.GPU_VRAM_SAFETY_FRACTION == 0.70
    assert gr.INITIAL_TARGET_PERCENT == gpu.GPU_VRAM_SAFETY_PERCENT


@pytest.mark.parametrize("free", (1 * 1024 * MIB, 2048 * MIB, 7118 * MIB, 8192 * MIB))
def test_legacy_and_policy_budgets_are_identical(free):
    probe = {"available": True, "vram_free_bytes": free}
    assert gpu.vram_budget_bytes(probe) == gr.policy_budget_bytes(
        free, gpu.GPU_VRAM_SAFETY_PERCENT
    )


def test_below_floor_degrades_loudly_not_up():
    free = 64 * MIB  # 70 % of 64 MiB < 64 MiB floor
    assert gpu.vram_budget_bytes({"available": True, "vram_free_bytes": free}) is None
