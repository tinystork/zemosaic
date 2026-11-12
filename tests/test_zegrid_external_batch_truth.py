"""ZM-ZEGRID-BATCH-TRUTH-AND-USAGE — product-level batch truth + telemetry.

These regressions pin the ONE thing the M33 post-mortem could not establish:
which batch actually reaches the external reducer, and which process the
per-cell memory telemetry describes.

They are deliberately small: the product run is ONE synthetic Cell (4 frames,
256x256, tile 128, bounded + tight RAM plan) driven through the real product
entry, and the heavy matrix stays opt-in behind ``external_batch_truth.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "tests", str(REPO / "tests" / "support")):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import external_batch_truth as TRUTH  # noqa: E402
from zemosaic.core.zegrid import external_stream as xstream  # noqa: E402
from zemosaic.core.zegrid import sweep as zsw  # noqa: E402

TILE = 128
CHANNELS = 3
POSITIONS_PER_ROW = TILE * CHANNELS          # 384
INTERIOR_POSITIONS = TILE * TILE * CHANNELS  # 49_152


@pytest.fixture(scope="module")
def truth(tmp_path_factory) -> dict:
    """ONE product-level external Cell, measured once."""
    workdir = tmp_path_factory.mktemp("batch_truth")
    art = TRUTH.measure_one_cell(workdir=str(workdir))
    assert art["cell"]["working_set"] == "external_spool", art["cell"]
    return art


def test_product_cell_emits_the_full_truth_tuple(truth: dict) -> None:
    t = truth["tuple"]
    for key in (
        "recorded_admission_column_batch",
        "requested_column_batch",
        "effective_column_batch",
        "column_batch_clamped",
        "rows_per_batch",
        "cols_per_batch",
        "batch_positions",
        "model_priced_peak_bytes",
        "peak_phase",
        "reduction_batch_bytes",
        "executed_column_batch",
        "executed_max_block_positions",
    ):
        assert key in t, key
    assert int(t["model_priced_peak_bytes"]) > 0
    assert int(t["reduction_batch_bytes"]) > 0
    assert t["peak_phase"] == "reduce"


def test_executed_batch_is_the_resolved_priced_batch(truth: dict) -> None:
    """The executor runs the batch the admission PRICED, not a phantom request."""
    t = truth["tuple"]
    assert int(t["effective_column_batch"]) == INTERIOR_POSITIONS
    assert t["rows_per_batch"] == TILE
    assert t["cols_per_batch"] == TILE
    assert t["batch_positions"] == INTERIOR_POSITIONS
    # One source of truth: the value handed to the executor == priced value.
    assert int(t["executed_column_batch"]) == INTERIOR_POSITIONS
    assert truth["verdict"]["executed_equals_effective_priced"] is True
    # ...and the reducer's own measured block agrees (it formed one tile-sized
    # block, never a wider one).
    assert int(t["executed_max_block_positions"]) == INTERIOR_POSITIONS


def test_recorded_admission_value_was_the_unexecutable_request(truth: dict) -> None:
    """Pinned evidence of the divergence the report documents.

    The durable working-set value here is an undivided request several tile
    interiors wide. It is NOT what reduced; the priced/effective value is. A
    reader must be able to see both, so the executed value is published too.
    """
    t = truth["tuple"]
    recorded = int(t["recorded_admission_column_batch"])
    effective = int(t["effective_column_batch"])
    assert recorded > effective
    assert recorded % POSITIONS_PER_ROW == 0
    assert effective == INTERIOR_POSITIONS
    assert truth["verdict"]["executed_equals_recorded"] is False
    assert truth["verdict"]["recorded_exceeds_one_tile_interior"] is True


def test_per_cell_memory_telemetry_names_the_reducing_process(truth: dict) -> None:
    r = truth["reducer"]
    assert r["pid"] is not None and r["role"] in ("coordinator", "cell_worker")
    assert r["source"] in (
        "resource_ru_maxrss", "psutil_peak_wset", "proc_status_VmHWM",
        "psutil_rss_not_peak",
    )
    assert r["is_true_peak"] is True
    assert int(r["peak_rss_bytes"]) > 0
    # The external class is single-owner: on this path the coordinator IS the
    # reducer, and the witness states that explicitly instead of assuming it.
    assert r["role"] == "coordinator"
    assert truth["verdict"]["reducer_is_coordinator"] is True


class _FakeMemInfo:
    def __init__(self, rss: int, peak_wset: int | None = None) -> None:
        self.rss = int(rss)
        if peak_wset is not None:
            self.peak_wset = int(peak_wset)


class _FakeProcess:
    def __init__(self, info: _FakeMemInfo) -> None:
        self._info = info

    def memory_info(self):
        return self._info


class _FakePsutil:
    def __init__(self, info: _FakeMemInfo) -> None:
        self._info = info

    def Process(self, *a, **k):  # noqa: N802 - psutil API name
        return _FakeProcess(self._info)


def test_peak_rss_source_reports_a_real_peak_not_a_post_hoc_sample(monkeypatch):
    """The historical Windows path published current RSS *after* the reduction.

    That is exactly how a ~9 GiB reducer could report 1.06 GiB: the arrays were
    already freed when the single sample was taken. The fix must (a) prefer the
    platform's true peak and (b) label a non-peak sample honestly.
    """
    monkeypatch.setattr(zsw, "resource", None)
    monkeypatch.setattr(zsw, "psutil", _FakePsutil(_FakeMemInfo(1_100 * 2 ** 20)))
    # No true-peak source is reachable either: this is the exact Windows shape
    # (no ``resource``, no ``peak_wset``) that produced the 1.06 GiB figure.
    monkeypatch.setattr(zsw, "_proc_status_vmhwm_kib", lambda: None)
    value, source = zsw.peak_rss_kib_with_source()
    assert value == 1_100 * 1024
    assert source == "psutil_rss_not_peak"
    assert zsw.peak_rss_kib() == value

    monkeypatch.setattr(
        zsw, "psutil",
        _FakePsutil(_FakeMemInfo(1_100 * 2 ** 20, peak_wset=9_000 * 2 ** 20)),
    )
    value, source = zsw.peak_rss_kib_with_source()
    assert source == "psutil_peak_wset"
    assert value == 9_000 * 1024  # the TRUE peak, not the current sample


def test_aligned_batch_at_the_admission_ceiling_fits_and_above_does_not():
    """Nothing may exceed the admitted budget (invariant, no constant moved)."""
    n, h, w, c, tile = 24, 256, 256, 3, 64
    halo = 9
    ppr = tile * c                      # 192 positions per interior row
    per_position = n * 96 + 512
    rows_budget = 32                    # < the 64-row interior: no tile clamp
    budget = rows_budget * ppr * per_position

    batch = xstream.largest_admissible_column_batch(
        budget_bytes=budget,
        n_frames=n,
        interior_width=tile,
        channels=c,
        fixed_position_bytes=512,
    )
    priced = xstream.estimate_external_working_set(
        n_frames=n, height=h, width=w, channels=c, tile_size=tile,
        halo_px=halo, column_batch=batch,
    )
    # The batch term this decision admits is inside the batch budget...
    assert int(priced["host"]["reduction_batch_bytes"]) <= budget
    # ...and it is exactly a whole-row batch of the priced geometry.
    assert int(priced["ram"]["effective_column_batch"]) == int(
        priced["ram"]["batch_positions"]
    )
    assert int(priced["ram"]["rows_per_batch"]) == rows_budget

    # One more whole row than the budget can hold must NOT fit, and the model
    # still prices it (no clamp hides it at this interior).
    wider = int(priced["ram"]["batch_positions"]) + ppr
    wider_priced = xstream.estimate_external_working_set(
        n_frames=n, height=h, width=w, channels=c, tile_size=tile,
        halo_px=halo, column_batch=wider,
    )
    assert int(wider_priced["ram"]["batch_positions"]) == wider
    assert int(wider_priced["host"]["reduction_batch_bytes"]) > budget
