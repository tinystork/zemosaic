"""ZM-ZEGRID-RESILIENT-RUN-RESUME M3 — product seam (CPU wave boundary).

The M3 recovery engine is wired into the real CPU wave loop behind the explicit
``zegrid_resource_recovery_enabled`` opt-in. These tests exercise the exact
product helpers without launching a mosaic:

* the required raw memory is derived from the 90 % RUN budget, not guessed;
* one validated RUN fact set projects into the M3 observation contract;
* the boundary release report is honest (no speculative unit, balanced credits)
  and never releases accepted payloads;
* a cooperative stop request from the M1 registry cancels the product wait
  promptly, returning ``STOPPED_RESUMABLE`` without erasing committed work.

No GPU, Windows or full product run is claimed here.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zemosaic import zemosaic_zegrid_mode as zz  # noqa: E402
from zemosaic.core.zegrid import resource_recovery as rec  # noqa: E402
from zemosaic.core.zegrid import run_registry as zrr  # noqa: E402
from zemosaic.core.zegrid import runtime_control as zrt  # noqa: E402


def _facts(raw=100, disk_free=10 ** 9) -> dict:
    budget = zrt.run_budget_bytes(int(raw))
    return {
        "schema": zrt.RUN_CONTROL_SCHEMA,
        "host": {
            "raw_available_bytes": int(raw),
            "budget_bytes": int(budget),
            "factor_num": 9,
            "factor_den": 10,
            "source": "test",
            "total_bytes": 8 * 2 ** 30,
        },
        "disk": {"free_bytes": int(disk_free), "unavailable_reason": None},
        "sampled_monotonic_ns": time.monotonic_ns(),
    }


def test_required_raw_is_the_exact_inverse_of_the_run_budget():
    for aggregate in (0, 1, 900, 1024, 10 ** 9 + 1):
        required = zz._r55_m3_required_raw(aggregate)
        if aggregate <= 0:
            assert required == 1
            continue
        assert zrt.run_budget_bytes(required) >= int(aggregate)
        # No gratuitous inflation: one byte less must NOT fit.
        assert zrt.run_budget_bytes(required - 1) < int(aggregate)


def test_checkpoint_facts_project_into_the_observation_contract():
    facts = _facts(raw=12345)
    observation = zz._r55_m3_observation_from_facts(facts)
    assert observation["raw_available_bytes"] == 12345
    parsed = rec.parse_observation(observation, max_age_s=None)
    assert parsed.source == "test"


def test_boundary_release_is_honest_and_never_releases_accepted_cells():
    registry = zrr.RunResultRegistry(
        run_id="run-seam",
        expected_cell_ids=["r0000c0000"],
        gauge_mode="global",
    )
    coordinator = zz._r55_m3_make_coordinator(
        result_registry=registry,
        probe=lambda: {"raw_available_bytes": 10, "observed_monotonic": time.monotonic()},
        profile_requested="auto",
        profile_effective="bounded",
    )
    report = coordinator.release_speculative(
        phase="run_wave_3", reason_code="REFUSE_HOST_RAM"
    )
    assert report.stopped_admission is True
    assert report.units_released == 0
    assert report.credits_balanced is True
    # A boundary release is not a registry disposal: no accepted payload exists
    # here, and the report makes no claim that one was freed.
    assert report.notes and "accepted payloads are immutable" in report.notes[0]


def test_registry_stop_cancels_the_product_wait_promptly():
    registry = zrr.RunResultRegistry(
        run_id="run-stop-seam",
        expected_cell_ids=["r0000c0000"],
        gauge_mode="global",
    )

    def probe():
        return {
            "raw_available_bytes": 10,
            "observed_monotonic": time.monotonic(),
            "source": "test",
        }

    coordinator = zz._r55_m3_make_coordinator(
        result_registry=registry,
        probe=probe,
        profile_requested="auto",
        profile_effective="bounded",
    )
    errors: list[BaseException] = []

    def run():
        try:
            coordinator.recover(
                facts={
                    "raw_available_bytes": 10,
                    "observed_monotonic": time.monotonic(),
                    "source": "test",
                },
                phase="run_wave_0",
                reason_code="REFUSE_HOST_RAM",
                detail="synthetic",
                minimum_bytes=10 ** 9,
            )
        except BaseException as exc:  # noqa: BLE001 - asserted below
            errors.append(exc)

    thread = threading.Thread(target=run, name="m3-product-wait")
    started = time.monotonic()
    thread.start()
    time.sleep(0.15)
    registry.request_stop("user stop")
    thread.join(timeout=2.0)
    elapsed = time.monotonic() - started
    assert not thread.is_alive()
    assert len(errors) == 1 and isinstance(
        errors[0], rec.CapacityWaitCancelled
    )
    assert errors[0].state == rec.STATE_STOPPED_RESUMABLE
    assert elapsed < 2.0
    assert coordinator.state == rec.STATE_STOPPED_RESUMABLE
    snapshot = coordinator.snapshot()
    assert snapshot["release_done"] is True
    assert snapshot["release"]["credits_balanced"] is True
