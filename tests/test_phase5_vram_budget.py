import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import zemosaic_worker as zw

MB = 1024**2
GB = 1024**3


def _make_ctx(*, power_plugged: bool, on_battery: bool, is_hybrid: bool = True):
    return SimpleNamespace(
        power_plugged=power_plugged,
        on_battery=on_battery,
        is_hybrid_graphics=is_hybrid,
        vram_free_bytes=6 * GB,
    )


def test_phase5_budget_ignores_plan_cap_on_ac():
    budget_bytes, meta = zw._compute_phase5_vram_budget_bytes(
        _make_ctx(power_plugged=True, on_battery=False),
        {},
        False,
        phase5_chunk_cap_bytes=128 * MB,
    )

    assert budget_bytes > 128 * MB
    assert meta["fraction"] == 0.60
    assert "plan_cap" not in meta["reasons"]
    assert "plan_cap_ignored_on_ac" in meta["reasons"]


def test_phase5_budget_keeps_plan_cap_on_battery():
    budget_bytes, meta = zw._compute_phase5_vram_budget_bytes(
        _make_ctx(power_plugged=False, on_battery=True),
        {},
        False,
        phase5_chunk_cap_bytes=128 * MB,
    )

    assert budget_bytes == 128 * MB
    assert meta["fraction"] == 0.25
    assert "plan_cap" in meta["reasons"]
