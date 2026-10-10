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


def test_phase_eta_rate_from_recent_window():
    # rework-1 (I1): rate is computed over the RECENT WINDOW (last k samples),
    # not the whole-phase cumulative average.
    eta = zobs.PhaseEta(min_samples=2, window=3)
    eta.observe(1.0, 5.0)
    eta.observe(2.0, 10.0)
    eta.observe(3.0, 15.0)  # steady 5 items/s
    # remaining = 100 - 15 = 85 items at 5/s -> 17 s
    assert eta.eta_seconds(100) == pytest.approx(17.0)
    assert eta.eta_seconds(40) is not None


def test_phase_eta_recent_window_reacts_to_slowdown():
    # rework-1 (I1): a mid-phase slowdown must be reflected quickly, not masked
    # by the earlier fast portion.
    eta = zobs.PhaseEta(min_samples=2, window=3)
    # Fast start: 100 items/s for the first samples.
    for i in range(1, 6):
        eta.observe(float(i), float(i * 100))
    # Slowdown: 1 item/s afterwards.
    eta.observe(6.0, 501.0)
    eta.observe(7.0, 502.0)
    eta.observe(8.0, 503.0)
    # The recent window now sees ~1 item/s; remaining = 1000-503 = 497 -> ~497 s.
    # (A whole-phase cumulative average would still report ~62 items/s -> ~8 s.)
    got = eta.eta_seconds(1000)
    assert got is not None and got > 300.0


def test_phase_eta_window_stable_not_jittery():
    # rework-1 (I1): the window smooths a single anomalous sample — it must not
    # flip the ETA sign or produce a wildly different number.
    eta = zobs.PhaseEta(min_samples=2, window=4)
    for i in range(1, 6):
        eta.observe(float(i), float(i * 10))
    before = eta.eta_seconds(100)
    eta.observe(6.0, 61.0)  # one slightly-off sample
    after = eta.eta_seconds(100)
    assert before is not None and after is not None
    # The estimate stays on the same order (no sign flip / no explosion).
    assert after > 0.0 and after < before * 5.0


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
    # ZM-PROGRESS-CONTRACT-R29: the stage id is STABLE (``zegrid:gauge``) with the
    # counters in the separate current/total callback fields (no counter-embedded
    # id like the legacy ``zegrid:gauge:50/100``).
    assert ("zegrid:gauge", 0, 100) in r.stages
    assert ("zegrid:gauge", 50, 100) in r.stages
    assert ("zegrid:gauge", 100, 100) in r.stages
    # Counters never appear inside the emitted id.
    for stage_str, _cur, _tot in r.stages:
        assert "/" not in stage_str


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


def test_run_log_replaces_previous_is_explicit(tmp_path):
    # rework-1 (I2): a re-run into the SAME output folder overwrites the run log,
    # but that replacement is now recorded explicitly (not silent).
    zz._open_run_log(tmp_path, "2026-10-07T00:00:00")
    first = (tmp_path / zz.RUN_LOG_NAME).read_text()
    assert "replaces_previous_log" not in first

    zz._open_run_log(tmp_path, "2026-10-07T01:00:00")
    second = (tmp_path / zz.RUN_LOG_NAME).read_text()
    assert "replaces_previous_log: true" in second
    assert "started: 2026-10-07T01:00:00" in second


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

@pytest.mark.slow
@pytest.mark.skipif(not LIGHTS.is_dir(), reason="M106 lights directory not present")
def test_layout_scan_logs_candidates_and_rejection(m106_corpus):
    frames, _rejected, canvas = m106_corpus
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


@pytest.mark.slow
@pytest.mark.skipif(not LIGHTS.is_dir(), reason="M106 lights directory not present")
def test_layout_infeasible_message_states_binding_and_budget_and_remedy(m106_corpus):
    frames, _rejected, canvas = m106_corpus
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

    # Progress callback was actually exercised with correct cumulative counters.
    assert calls, "gauge progress callback was not called"
    # rework-1 (M1): total is CUMULATIVE = 2N-1 across both sub-passes, and the
    # done counter is monotone 1..2N-1 (no reset at the counts->pairs boundary).
    n = len(descs)
    assert all(total == 2 * n - 1 for _d, total, _i in calls)
    done_seq = [d for d, _t, _i in calls]
    assert done_seq == sorted(done_seq), "done is not monotone across the pass boundary"
    assert done_seq[-1] == 2 * n - 1
    # The sub-pass is made explicit via the counts:/pairs: item prefix.
    assert any(i.startswith("counts:") for _d, _t, i in calls)
    assert any(i.startswith("pairs:") for _d, _t, i in calls)


def test_gauge_eta_does_not_explode_at_pass_boundary(monkeypatch):
    """rework-1 (M1): the gauge ETA must stay honest across the counts->pairs
    boundary. The two sub-passes report CUMULATIVE done over total=2N-1, so the
    rate never collapses and the ETA never explodes when the second pass starts.
    """
    r = _Recorder()
    rep = zobs.PhaseReporter(r.emit, interval_s=0.0)  # no throttle
    n = 10
    total = 2 * n - 1
    rep.start("gauge", total=total, unit="frame-ops")

    # Simulate elapsed time via a controllable clock.
    clock = {"t": 0.0}
    real_perf = zobs.time.perf_counter
    monkeypatch.setattr(zobs.time, "perf_counter", lambda: clock["t"])

    # Counts pass: 10 items over 10 s (1 item/s), done 1..10.
    for d in range(1, n + 1):
        clock["t"] = float(d)
        rep.progress(d, item_id=f"counts:d{d}")
    # Pairs pass continues at done=11..19 over the next 9 s (still 1 item/s).
    for k in range(1, n):
        d = n + k
        clock["t"] = float(n + k)
        rep.progress(d, item_id=f"pairs:d{k}")

    progress = [m for m, lvl in r.emitted if m.startswith("phase PROGRESS: gauge")]
    assert progress
    # The % reflects the WHOLE phase: the final line is 19/19 = 100.0%.
    assert any("19/19" in p and "(100.0%)" in p for p in progress)
    # The ETA never explodes: at every emitted progress line the eta is a small
    # bounded value (rate ~1 item/s, remaining <= 19 s), never None and never
    # a huge number.
    for p in progress:
        assert "eta=" in p
        eta_tok = p.split("eta=", 1)[1].split(" ", 1)[0]
        assert eta_tok != "n/a yet"
        # Format is e.g. "19s" / "18s" — a bounded seconds value, not minutes/hours.
        assert "m" not in eta_tok and "h" not in eta_tok

    monkeypatch.setattr(zobs.time, "perf_counter", real_perf)


def test_gauge_reporter_total_is_cumulative():
    """rework-1 (M1): the gauge phase total must be 2N-1 (counts N + pairs N-1),
    matching the cumulative counters emitted by compute_global_gauge."""
    r = _Recorder()
    rep = zobs.PhaseReporter(r.emit, interval_s=0.0)
    rep.start("gauge", total=2 * 10 - 1, unit="frame-ops")
    rep.progress(19, item_id="pairs:d9")
    starts = [m for m, lvl in r.emitted if "phase START: gauge" in m]
    assert starts and "(total=19)" in starts[0]


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
