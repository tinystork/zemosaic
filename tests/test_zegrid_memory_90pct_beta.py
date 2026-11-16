"""ZM-MEMORY-90PCT-BETA — effective 90 % RAM/VRAM ceilings on beta.

Small, fast policy gate. It pins only the constant-level contract:

* the PLAN factor is 9/10, applied ONCE to raw AVAILABLE memory with integer
  arithmetic (admission was already 9/10 and is unchanged);
* the VRAM policy is 90 % of fresh free VRAM;
* the R55-E10 selection headroom is 100 %, so no second 0.85 x 0.90 haircut;
* the default :class:`VramPolicy` still constructs and shrinks through the
  historical 75 -> 70 -> 65 -> 50 rungs.

No GPU, no integration run, no product mosaic.
"""

from __future__ import annotations

from zemosaic.core.zegrid import gpu
from zemosaic.core.zegrid import gpu_readiness as gr
from zemosaic.core.zegrid import ram_contract as rc

MIB = 2 ** 20
GIB = 2 ** 30

# Recorded M106 free VRAM observation (historical witness input).
M106_FREE = 7_463_763_968


def test_plan_and_admit_factors_are_90_percent_with_one_integer_application():
    assert (rc.PLAN_FACTOR_NUM, rc.PLAN_FACTOR_DEN) == (9, 10)
    assert (rc.ADMIT_FACTOR_NUM, rc.ADMIT_FACTOR_DEN) == (9, 10)

    for raw in (1, 10, 16 * GIB + 7, 10 ** 9 + 3):
        expected = (9 * raw) // 10
        assert rc.plan_budget_bytes(raw) == expected
        assert rc.admit_budget_bytes(raw) == expected
        assert rc.Observation.plan(raw).budget_bytes == expected
        assert rc.Observation.admit(raw).budget_bytes == expected

    raw = 10 ** 9 + 7
    once = (9 * raw) // 10
    assert rc.plan_budget_bytes(raw) == once
    # A second application would be visibly smaller; the factor is applied once.
    assert rc.plan_budget_bytes(raw) != (9 * once) // 10


def test_tolerance_band_is_empty_under_equal_plan_and_admit_factors():
    raw = 1000
    assert rc.plan_budget_bytes(raw) == 900
    assert rc.admit_budget_bytes(raw) == 900
    assert rc.requires_tolerance(900, raw) is False
    assert rc.requires_tolerance(901, raw) is False


def test_vram_ceiling_is_90_percent_of_fresh_free_vram():
    assert gpu.GPU_VRAM_SAFETY_PERCENT == 90
    assert gpu.GPU_VRAM_SAFETY_FRACTION == 0.9
    assert gr.INITIAL_TARGET_PERCENT == 90

    expected = (M106_FREE * 90) // 100
    assert gpu.vram_budget_bytes(
        {"available": True, "vram_free_bytes": M106_FREE}
    ) == expected
    policy = gr.VramPolicy()
    assert policy.initial_percent == 90
    candidate = policy.initial(M106_FREE)
    assert candidate.percent == 90
    assert candidate.budget_bytes == expected
    assert candidate.eligible is True


def test_selection_uses_the_full_90_percent_budget_not_a_second_haircut():
    from zemosaic import zemosaic_zegrid_mode as zz

    assert zz._R55E10_VRAM_SELECTION_HEADROOM_PERCENT == 100
    policy = gr.VramPolicy()
    budget = policy.initial(M106_FREE).budget_bytes
    selection_budget = (budget * zz._R55E10_VRAM_SELECTION_HEADROOM_PERCENT) // 100
    assert selection_budget == budget
    # The historical 0.85 x 0.90 (~76.5 % of free VRAM) haircut is gone.
    assert selection_budget != (budget * 85) // 100
    assert selection_budget == (M106_FREE * 90) // 100


def test_default_vram_policy_is_valid_and_preserves_shrink_rungs():
    policy = gr.VramPolicy()
    assert policy.ladder == (50, 65, 70, 75, 90)
    assert tuple(sorted(set(policy.ladder))) == policy.ladder
    assert policy.initial_percent in policy.ladder

    free = 4 * GIB
    assert policy.initial(free).percent == 90
    assert policy.next_shrink(90, free).percent == 75
    assert policy.next_shrink(75, free).percent == 70
    assert policy.next_shrink(70, free).percent == 65
    assert policy.next_shrink(65, free).percent == 50
    assert policy.next_shrink(50, free) is None
