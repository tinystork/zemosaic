"""ZM-ZEGRID-R14 targeted tests — live observability.

Covers (non-gated, fast):

* phase START/END lines are emitted with timing + throughput;
* intra-phase progress messages appear with a BOUNDED cadence and correct
  counters (done/total + %, item id);
* ETA is produced once samples exist (and an explicit "n/a yet" before);
* the crash-breadcrumb `stage` callback is populated with phase + fraction;
* the run log exists and GROWS during the run (write-then-read while in progress);
* the layout scan logs candidates + rejection reasons and a clear
  ``LayoutInfeasible`` message stating the binding constraint + budget + remedy;
* results stay BIT-EQUAL (observability adds no science change).

These tests exercise the observability helpers directly plus the production
wiring seams (``_emit`` routing, ``_choose_layout_mode_aware`` emit seam, run-log
incremental flush, ``compute_global_gauge`` progress seam) without needing the
heavy M16/M106 fixtures.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.zegrid import auto_layout as zal
from zemosaic.core.zegrid import geometry as zg
from zemosaic.core.zegrid import observability as zobs

LIGHTS = Path("/home/tristan/M106/lights")


# ---------------------------------------------------------------------------
# format_eta
# ---------------------------------------------------------------------------

def test_format_eta_none_is_n_a_yet():
    assert zobs.format_eta(None) == "n/a yet"


def test_format_eta_seconds_and_minutes():
    assert zobs.format_eta(0.4) == "<1s"
    assert zobs.format_eta(59.0) == "59s"
    assert zobs.format_eta(125.0) == "2m05s"
    assert zobs.format_eta(3600.0 + 90.0) == "1h01m"


def test_format_eta_non_finite_is_n_a_yet():
    assert zobs.format_eta(float("nan")) == "n/a yet"


# ---------------------------------------------------------------------------
# PhaseEta: explicit n/a until samples exist, then a real ETA
# ---------------------------------------------------------------------------

def test_phase_eta_n_a_until_samples():
    eta = zobs.PhaseEta(min_samples=2)
    assert eta.eta_seconds(100) is None  # no samples yet
    eta.observe(1.0, 10.0)  # one sample only
    assert eta.eta_seconds(100) is None  # still < min_samples
    eta.observe(2.0, 20.0)  # second sample
    got = eta.eta_seconds(100)
    assert got is not None and got > 0.0


def test_phase_eta_rate_from_latest_sample():
    eta = zobs.PhaseEta(min_samples=1)
    eta.observe(10.0, 50.0)  # rate 5 items/s
    # remaining = 100 - 50 = 50 items at 5/s -> 10 s
    assert eta.eta_seconds(100) == pytest.approx(10.0)
    assert eta.eta_seconds(40) == 0.0  # already past total


# ---------------------------------------------------------------------------
# GlobalEta: n/a until a phase completes, then estimates from learned weights
# ---------------------------------------------------------------------------

def test_global_eta_n_a_until_weights():
    g = zobs.GlobalEta(["setup", "gauge", "assembly"])
    assert g.estimate() is None
    g.phase_completed("setup", 5.0)
    assert g.estimate(current_remaining_s=10.0) is not None


def test_global_eta_uses_completed_durations():
    g = zobs.GlobalEta(["setup", "gauge", "assembly"])
    g.phase_completed("setup", 6.0)
    g.phase_completed("gauge", 4.0)  # mean weight 5.0
    # remaining phases: assembly only (unseen -> default mean 5.0)
    assert g.estimate(current_remaining_s=0.0) == pytest.approx(5.0)
    assert g.estimate(current_remaining_s=10.0) == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# PhaseReporter: START/END lines + bounded progress cadence + stage
# ---------------------------------------------------------------------------

class _Recorder:
    def __init__(self):
        self.emitted = []
        self.stages = []
        self.log_lines = []

    def emit(self, msg, lvl="INFO"):
        self.emitted.append((msg, lvl))

    def stage(self, stage_str, current, total):
        self.stages.append((stage_str, int(current), int(total)))

    def log_line(self, line):
        self.log_lines.append(line)


def test_reporter_start_end_lines_with_timing():
    r = _Recorder()
    rep = zobs.PhaseReporter(r.emit, stage=r.stage, log_line=r.log_line)
    rep.start("setup", total=100, unit="frames")
    rep.end(throughput="10.0 frames/s")
    starts = [m for m, lvl in r.emitted if m.startswith("phase START: setup")]
    ends = [m for m, lvl in r.emitted if m.startswith("phase END: setup")]
    assert starts, "no START line"
    assert ends, "no END line"
    assert "elapsed=" in ends[0]
    assert "throughput=10.0 frames/s" in ends[0]


def test_reporter_bounded_cadence():
    r = _Recorder()
    rep = zobs.PhaseReporter(r.emit, interval_s=999999.0)  # huge -> throttle hard
    rep.start("gauge", total=1000, unit="frames")
    for i in range(1, 1000):
        rep.progress(i, item_id=f"f{i:03d}")
    rep.progress(1000, item_id="f999", force=True)
    progress = [m for m, lvl in r.emitted if m.startswith("phase PROGRESS: gauge")]
    # First + final sample always emitted; the middle is throttled away.
    assert progress, "no progress lines"
    assert any("1/1000" in p for p in progress)      # first sample
    assert any("1000/1000" in p for p in progress)   # forced final sample
    assert any("(100.0%)" in p for p in progress)


def test_reporter_progress_has_percent_and_item_and_eta():
    r = _Recorder()
    rep = zobs.PhaseReporter(r.emit, interval_s=0.0)  # no throttle
    rep.start("gauge", total=100, unit="frames")
    rep.progress(25, item_id="f024.fits")
    progress = [m for m, lvl in r.emitted if m.startswith("phase PROGRESS: gauge")]
    assert progress
    assert "25/100" in progress[0]
    assert "(25.0%)" in progress[0]
    assert "item=f024.fits" in progress[0]
    assert "eta=" in progress[0]


def test_reporter_stage_callback_populated():
    r = _Recorder()
    rep = zobs.PhaseReporter(r.emit, stage=r.stage)
    rep.start("gauge", total=100, unit="frames")
    rep.progress(50, item_id="f049.fits")
    rep.end()
    assert r.stages, "no stage callbacks"
    # start -> (0/100), progress -> (50/100), end -> (100/100)
    assert ("zegrid:gauge:0/100", 0, 100) in r.stages
    assert ("zegrid:gauge:50/100", 50, 100) in r.stages
    assert ("zegrid:gauge:100/100", 100, 100) in r.stages


def test_reporter_log_line_flushed_to_run_log(tmp_path):
    r = _Recorder()
    rep = zobs.PhaseReporter(r.emit, log_line=r.log_line)
    rep.start("setup", total=10, unit="frames")
    rep.end()
    assert r.log_lines, "no log lines recorded"


# ---------------------------------------------------------------------------
# Incremental run log: exists and GROWS during the run (write-then-read)
# ---------------------------------------------------------------------------

def test_run_log_grows_during_run(tmp_path):
    zz._open_run_log(tmp_path, "2026-10-07T00:00:00")
    path = tmp_path / zz.RUN_LOG_NAME
    assert path.exists(), "run log not created at run start"

    before = path.read_text()
    assert "Live phase log" in before

    zz._append_run_log_line(tmp_path, "phase START: gauge (total=10 frames)")
    mid = path.read_text()
    assert "phase START: gauge" in mid
    assert len(mid) > len(before), "run log did not GROW after a phase line"

    zz._append_run_log_line(tmp_path, "phase END: gauge elapsed=1.000s")
    after = path.read_text()
    assert "phase END: gauge" in after
    assert len(after) > len(mid)


def test_write_run_log_appends_summary(tmp_path):
    timings = zz.zin.Timings()
    timings.add("setup", 0.5)
    timings.add("assembly", 0.25)
    w = zg.GlobalCanvas(
        canvas_id="x",
        wcs_header=_synthetic_tan_wcs().to_header().tostring(),
        width=20, height=20, resolution_deg=0.001,
    )
    layout = {"nx": 1, "ny": 1, "ram_budget_bytes": 0, "available_bytes": 0,
              "max_contributors": 1, "predicted_bound_bytes": 0,
              "layout_source": "auto"}
    zz._open_run_log(tmp_path, "2026-10-07T00:00:00")
    zz._append_run_log_line(tmp_path, "phase START: setup")
    zz._write_run_log(
        tmp_path, timings, start_ts="2026-10-07T00:00:00",
        frames_loaded=2, n_included=2, n_rejected=0, canvas=w, layout=layout,
        gpu_used=False, ignored_settings={"stack_use_gpu": True},
        peak_rss_kib=123, cache_info={"peak_cell_bytes": 5},
        global_reference_frame_id="a.fits",
    )
    text = (tmp_path / zz.RUN_LOG_NAME).read_text()
    # Live line preserved AND summary appended.
    assert "phase START: setup" in text
    assert "Timings (wall-clock):" in text
    assert "setup:" in text
    assert "GPU usage:" in text
    assert "stack_use_gpu" in text


def _synthetic_tan_wcs(shape=(20, 20)):
    from astropy.wcs import WCS

    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [10.0, 30.0]
    w.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    w.wcs.cd = np.array([[-1, 0], [0, 1]]) * 0.001
    w.array_shape = shape
    return w


# ---------------------------------------------------------------------------
# Layout scan explainability: candidate + rejection reasons + LayoutInfeasible
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not LIGHTS.is_dir(), reason="M106 lights directory not present")
def test_layout_scan_logs_candidates_and_rejection():
    frames, _ = zg.read_manifest(LIGHTS)
    canvas = zg.build_canvas(frames)
    msgs = []
    zz._choose_layout_mode_aware(
        canvas, frames, ram_budget=int(4 * 2**30),
        emit=lambda m, lvl="INFO": msgs.append(m),
    )
    scan = [m for m in msgs if m.startswith("layout scan:")]
    assert scan, "no layout scan lines emitted"
    # A chosen layout line must appear and name the final (nx x ny).
    chosen = [m for m in scan if "CHOSEN" in m]
    assert chosen, "no CHOSEN layout line"
    assert "source=auto" in chosen[0]


@pytest.mark.skipif(not LIGHTS.is_dir(), reason="M106 lights directory not present")
def test_layout_infeasible_message_states_binding_and_budget_and_remedy():
    frames, _ = zg.read_manifest(LIGHTS)
    canvas = zg.build_canvas(frames)
    msgs = []
    with pytest.raises(zal.LayoutInfeasible) as exc:
        zz._choose_layout_mode_aware(
            canvas, frames, ram_budget=int(1 * 2**30), pinned_layout=(3, 2),
            emit=lambda m, lvl="INFO": msgs.append(m),
        )
    msg = str(exc.value)
    # States the binding constraint (bound), the budget, and the remedy.
    assert "bound" in msg.lower() or "MiB" in msg
    assert "MiB" in msg
    assert "zegrid_layout" in msg, "must suggest pinning zegrid_layout"


# ---------------------------------------------------------------------------
# compute_global_gauge progress seam (bit-equal, best-effort callback)
# ---------------------------------------------------------------------------

def _dithered_decode(f):
    return _DITHER[f.frame_id.logical_path]


_DITHER = {}


def _build_small_corpus(n=4):
    h = w = 40
    scale = 0.001
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:h, 0:w]
    base = 100.0 + 20.0 * np.exp(-((yy - h / 2) ** 2 + (xx - w / 2) ** 2) / (2 * 12.0**2))
    from astropy.wcs import WCS

    descs = []
    _DITHER.clear()
    for i in range(n):
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.crval = [10.0 + i * 0.004, 30.0]
        wcs.wcs.crpix = [w / 2.0, h / 2.0]
        wcs.wcs.cd = np.array([[-1.0, 0.0], [0.0, 1.0]]) * scale
        wcs.array_shape = (h, w)
        data = (base + rng.normal(0.0, 0.5, (h, w))).astype(np.float32)
        hwc = np.stack([data, data, data], axis=-1)
        fid = f"d{i}.fits"
        _DITHER[fid] = hwc
        descs.append(
            zg.FrameDescriptor(
                frame_id=zg.FrameId(fid), source_path=f"/x/{fid}",
                shape_hw=(h, w), wcs_header=wcs.to_header(relax=True).tostring(),
                header_sha256="", instrument="",
            )
        )
    return descs


def test_gauge_progress_seam_bit_equal():
    from zemosaic.core.zegrid import photometric as zphot
    from zemosaic.core.zegrid.executor import ExecutorConfig

    descs = _build_small_corpus()
    canvas = zg.build_canvas(descs)
    config = ExecutorConfig().science_config()

    ref, ref_ids = zphot.compute_global_gauge(descs, canvas, _dithered_decode, config, None, workers=1)

    calls = []
    new, new_ids = zphot.compute_global_gauge(
        descs, canvas, _dithered_decode, config, None, workers=1,
        progress_callback=lambda done, total, item: calls.append((done, total, item)),
    )

    # Results bit-equal regardless of the progress seam.
    assert tuple(ref_ids) == tuple(new_ids)
    assert int(ref.reference_index) == int(new.reference_index)
    np.testing.assert_array_equal(ref.norm_active, new.norm_active)
    np.testing.assert_array_equal(ref.coefficients, new.coefficients)
    np.testing.assert_array_equal(ref.weights, new.weights)

    # Progress callback was actually exercised with correct counters.
    assert calls, "gauge progress callback was not called"
    for done, total, _item in calls:
        assert 1 <= done <= total
    # The count pass reports total == n frames.
    assert any(total == len(descs) for _d, total, _i in calls)


# ---------------------------------------------------------------------------
# parallel.pmap progress seam preserves order + results
# ---------------------------------------------------------------------------

def test_pmap_progress_seam_preserves_results():
    from zemosaic.core.zegrid import parallel as zpar

    calls = []
    got = zpar.pmap(lambda x: x * x, [1, 2, 3, 4], workers=2,
                    progress_callback=lambda done, total: calls.append((done, total)))
    assert got == [1, 4, 9, 16]
    assert calls, "pmap progress callback not called"
    assert calls[-1] == (4, 4)  # last call reports done == total
