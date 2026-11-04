"""ZM-PHASE3-TILE-PREFETCH — bounded intra-cell single-tile prefetch.

Acceptance coverage (mission ZM-PHASE3-TILE-PREFETCH-20261104):

* admission: exact one-slot byte model, admitted/refused, CPU/single-tile/
  no-evidence refusals, historical fallback with identical science;
* mechanics: bit-exact off/on parity (science, weights, masks, support,
  surviving, rejection mask/counts, coefficients/diagnostics), mono + RGB,
  NaN/inf, halo/borders, low-N, footprint taper;
* provider contract: same calls and same order, every (tile, frame) read once,
  consumer never reads while prefetch is active;
* bounded pipeline: queue max == 1, real overlap proven with Events (no timing
  thresholds), clean stop/drain/join on provider error (first/middle/last),
  consumer error, cancellation and a GPU-OOM-like classification;
* telemetry: profiling OFF adds no perf_counter/CUDA event/sync; ON publishes
  coherent counters and the declared overlap topology.

The GPU product path is SIMULATED (backend token "gpu", NumPy execution): the
mission explicitly excludes a real GPU/Windows run here. A real GPU bit-exact
test remains the existing opt-in `test_sci05_canonical_streaming_gpu.py`.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

import zemosaic.core.canonical_streaming as cstream
from zemosaic.core.canonical_engine import CanonicalStackRequest
from zemosaic.core.canonical_streaming import (
    InMemoryCanonicalProvider,
    admit_tile_prefetch,
    estimate_tile_prefetch_slot_bytes,
    run_canonical_stack_streaming,
)
from zemosaic.core.zegrid import stack_phase_profile as zspp
from zemosaic.core.zegrid import streaming as zstream
from zemosaic.core.zegrid.file_provider import (
    MemmapCanonicalProvider,
    write_aligned_cache_from_arrays,
)

MIB = 1 << 20
_PREFETCH_THREAD = "cogrid-tile-prefetch"


# ---------------------------------------------------------------------------
# Deterministic corpora
# ---------------------------------------------------------------------------

def _mono_corpus(n=6, h=41, w=37, seed=11):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = 100.0 + 0.4 * yy + 0.2 * xx
    for _ in range(4):
        cy, cx = rng.uniform(0.15, 0.85) * h, rng.uniform(0.15, 0.85) * w
        base = base + 35.0 * np.exp(
            -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.0 ** 2)
        )
    frames = []
    for i in range(n):
        f = (base * (1.0 + 0.01 * i) + 0.5 * i + rng.normal(0, 1.5, (h, w))).astype(
            np.float32
        )
        frames.append(f)
    masks = [np.ones((h, w), dtype=bool) for _ in range(n)]
    def _frame(k):
        return frames[k] if k < n else None
    def _mask(k):
        return masks[k] if k < n else None
    if n > 1:
        frames[1][3:9, 5:12] = np.nan      # NaN hole
    if n > 2:
        frames[2][0, 0] = np.inf           # non-finite pixel
        frames[2][-1, -1] = -np.inf
        frames[2][min(7, h - 1), min(7, w - 1)] = 1.0e6  # outlier
    if n > 3:
        masks[3][:, w // 2:] = False       # right border unsupported
    if n > 4:
        masks[4][0:2, :] = False           # top border band
    if n > 5:
        frames[5][15:20, 15:20] = np.nan   # interior hole
    return frames, masks


def _rgb_corpus(n=5, h=33, w=35, seed=17):
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n):
        f = rng.normal(0, 2.0, (h, w, 3)).astype(np.float32)
        f += np.array([40.0, 45.0, 50.0], dtype=np.float32) + 1.5 * i
        f[rng.random((h, w, 3)) < 0.05] = np.nan
        frames.append(f)
    frames[0][2, 2, :] = np.inf
    frames[1][5:9, 5:9, 1] = np.nan
    masks = [np.ones((h, w), dtype=bool) for _ in range(n)]
    masks[2][:, -3:] = False
    masks[3][:2, :] = False
    return frames, masks


# ---------------------------------------------------------------------------
# Simulated GPU path (backend token only; execution stays host NumPy)
# ---------------------------------------------------------------------------

@pytest.fixture
def sim_gpu(monkeypatch):
    monkeypatch.setattr(cstream, "canonical_gpu_available", lambda: True)
    monkeypatch.setattr(cstream, "_get_xp", lambda backend: np)
    return True


# ---------------------------------------------------------------------------
# Recording provider
# ---------------------------------------------------------------------------

class _RecordingProvider(InMemoryCanonicalProvider):
    def __init__(self, images, supports):
        super().__init__(images, supports)
        self.calls = []
        self.closed = False
        self.reads_after_close = 0

    def get_tile(self, i, y0, y1, x0, x1):
        if self.closed:
            self.reads_after_close += 1
        self.calls.append((int(i), int(y0), int(y1), int(x0), int(x1)))
        return super().get_tile(i, y0, y1, x0, x1)

    def close(self):
        self.closed = True


class _Boom(RuntimeError):
    pass


class _FakeCupyOOM(RuntimeError):
    pass


class _HookProfiler(zspp.StackPhaseProfiler):
    """Profiler that can observe phases and raise on a chosen event."""

    def __init__(self, backend="gpu", *, taper_event=None, writes_event=None,
                 entered_event=None, checked_event=None, raise_phase=None,
                 raise_exc=None, raise_after=1):
        super().__init__(backend=backend)
        self.taper_event = taper_event
        self.writes_event = writes_event
        self.entered_event = entered_event
        self.checked_event = checked_event
        self.raise_phase = raise_phase
        self.raise_exc = raise_exc
        self.raise_after = int(raise_after)
        self._seen = {}

    def add(self, phase, seconds):
        super().add(phase, seconds)
        self._seen[phase] = self._seen.get(phase, 0) + 1
        if self.taper_event is not None and phase == "taper_weights_s":
            self.taper_event.set()
            if self.entered_event is not None and self._seen[phase] == 1:
                # Proof that the producer is CONCURRENTLY inside the next read,
                # not merely that it was scheduled before this tile finished.
                assert self.entered_event.wait(10.0), "producer never entered next read"
                if self.checked_event is not None:
                    # Wait until the producer has observed the overlap; the
                    # first tile's writes_s therefore cannot have happened yet.
                    assert self.checked_event.wait(10.0), "producer never checked overlap"
        if self.writes_event is not None and phase == "writes_s":
            self.writes_event.set()
        if self.raise_phase == phase and self._seen[phase] == self.raise_after:
            raise self.raise_exc

    def note_tile_prefetch(self, record):
        # Keep the base telemetry even when a test raises from add().
        super().note_tile_prefetch(record)


# ---------------------------------------------------------------------------
# Request / comparison helpers
# ---------------------------------------------------------------------------

def _request(images, supports, backend, **kw):
    return CanonicalStackRequest(
        images=images, geometric_support=supports, backend=backend, **kw
    )


_FIELDS = (
    "science",
    "estimator_weight_sum",
    "valid_mask",
    "surviving_sample_count",
    "support_w1",
    "support_w2",
    "n_eff_support",
    "rejection_mask",
)


def _assert_results_bit_equal(off, on, *, ignore_prefetch_provenance=True):
    for name in _FIELDS:
        a = getattr(off, name)
        b = getattr(on, name)
        assert np.array_equal(a, b, equal_nan=True), name
    for name in (
        "iterations_used",
        "initial_sample_count",
        "rejected_sample_count",
        "rejected_fraction",
        "low_n_cell_count",
        "degenerate_cell_count",
        "rejection_method",
        "n_frames",
        "height",
        "width",
        "channels",
        "original_mono",
        "original_ndim",
        "original_shape",
    ):
        assert getattr(off, name) == getattr(on, name), name
    pa = dict(off.provenance)
    pb = dict(on.provenance)
    if ignore_prefetch_provenance:
        pa.pop("tile_prefetch", None)
        pb.pop("tile_prefetch", None)
    assert pa == pb


def _run(provider, request, tile_size, **kw):
    return run_canonical_stack_streaming(provider, request, tile_size=tile_size, **kw)


# ---------------------------------------------------------------------------
# 1. Admission model
# ---------------------------------------------------------------------------

def test_admission_slot_byte_model_is_exact():
    # tile 8, halo 9, patch 40x50 -> ext 26x26; N=5, C=1
    slot = estimate_tile_prefetch_slot_bytes(5, 40, 50, 1, tile_size=8, halo_px=9)
    assert slot == 5 * 26 * 26 * (4 * 1 + 1) + MIB
    # single tile charges the full patch (only relevant if the token is used)
    full = estimate_tile_prefetch_slot_bytes(5, 40, 50, 1, tile_size=None)
    assert full == 5 * 40 * 50 * 5 + MIB
    # clipping: a tile larger than the patch never over-charges
    clipped = estimate_tile_prefetch_slot_bytes(5, 10, 12, 3, tile_size=64, halo_px=9)
    assert clipped == 5 * 10 * 12 * (4 * 3 + 1) + MIB


def test_admission_accepts_only_with_explicit_headroom():
    base = dict(
        requested="auto", backend="gpu", n_frames=6, height=40, width=40,
        channels=1, tile_size=8, halo_px=9,
    )
    slot = estimate_tile_prefetch_slot_bytes(6, 40, 40, 1, 8, 9)
    required = (slot * 115 + 99) // 100
    ok = admit_tile_prefetch(**base, budget_bytes=required)
    assert ok["admitted"] and ok["effective"] == 1 and ok["reason"] == "admitted"
    refused = admit_tile_prefetch(**base, budget_bytes=required - 1)
    assert not refused["admitted"] and refused["effective"] == 0
    assert refused["reason"] == "insufficient_budget"
    # Fresh available RAM is an equivalent admission source.
    assert admit_tile_prefetch(**base, available_bytes=required)["admitted"]
    assert (
        admit_tile_prefetch(**base, available_bytes=required - 1)["reason"]
        == "insufficient_available_ram"
    )
    # Fail closed without ANY evidence.
    assert admit_tile_prefetch(**base)["reason"] == "admission_unavailable"
    # CPU stays historical; a single tile has no next tile; off is off.
    assert admit_tile_prefetch(**{**base, "backend": "cpu"}, budget_bytes=10**12)["reason"] == "cpu_backend"
    assert admit_tile_prefetch(**{**base, "tile_size": None}, budget_bytes=10**12)["reason"] == "single_tile_no_next"
    assert admit_tile_prefetch(**{**base, "requested": None}, budget_bytes=10**12)["reason"] == "requested_off"


# ---------------------------------------------------------------------------
# 2. Bit-exact off/on parity (mechanism, mono + RGB + halo/borders/low-N)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tile", [7, 9, 16])
@pytest.mark.parametrize(
    "kw",
    [
        dict(normalization="sky_mean", weighting="noise_variance",
             rejection="winsorized_sigma_clip", combine="mean",
             taper="footprint", taper_px=4.0, taper_floor=0.0),
        dict(normalization="none", weighting="none",
             rejection="kappa_sigma", combine="median",
             taper="none"),
        dict(normalization="linear_fit", weighting="noise_fwhm",
             rejection="kappa_sigma", combine="mean",
             taper="footprint", taper_px=3.0, taper_floor=0.0),
    ],
)
def test_mono_off_on_parity(sim_gpu, tile, kw):
    images, masks = _mono_corpus()
    req = _request(images, masks, "gpu", **kw)
    off = _run(InMemoryCanonicalProvider(images, masks), req, tile)
    on = _run(
        InMemoryCanonicalProvider(images, masks), req, tile,
        tile_prefetch="1", tile_prefetch_available_bytes=10**12,
    )
    assert on.provenance["tile_prefetch"]["effective"] == 1
    assert off.provenance.get("tile_prefetch") is None
    _assert_results_bit_equal(off, on)


def test_rgb_off_on_parity_borders_and_low_n(sim_gpu):
    images, masks = _rgb_corpus()
    req = _request(
        images, masks, "gpu",
        normalization="sky_mean", weighting="noise_variance",
        rejection="winsorized_sigma_clip", combine="median",
        taper="footprint", taper_px=5.0, taper_floor=0.0,
    )
    off = _run(InMemoryCanonicalProvider(images, masks), req, 8)
    on = _run(
        InMemoryCanonicalProvider(images, masks), req, 8,
        tile_prefetch="auto", tile_prefetch_available_bytes=10**12,
    )
    assert on.provenance["tile_prefetch"]["effective"] == 1
    _assert_results_bit_equal(off, on)


def test_single_tile_no_prefetch_but_same_science(sim_gpu):
    images, masks = _mono_corpus(n=4, h=20, w=18)
    req = _request(images, masks, "gpu", normalization="sky_mean",
                   weighting="none", rejection="none", combine="mean",
                   taper="none")
    off = _run(InMemoryCanonicalProvider(images, masks), req, None)
    on = _run(
        InMemoryCanonicalProvider(images, masks), req, None,
        tile_prefetch="auto", tile_prefetch_available_bytes=10**12,
    )
    assert on.provenance["tile_prefetch"]["effective"] == 0
    assert on.provenance["tile_prefetch"]["reason"] == "single_tile_no_next"
    _assert_results_bit_equal(off, on)


# ---------------------------------------------------------------------------
# 3. Provider call contract: same calls, same order, exactly once
# ---------------------------------------------------------------------------

def test_provider_calls_identical_and_each_pair_once(sim_gpu):
    images, masks = _mono_corpus()
    kw = dict(normalization="sky_mean", weighting="none",
              rejection="winsorized_sigma_clip", combine="mean",
              taper="footprint", taper_px=4.0, taper_floor=0.0)
    req = _request(images, masks, "gpu", **kw)
    p_off = _RecordingProvider(images, masks)
    p_on = _RecordingProvider(images, masks)
    off = _run(p_off, req, 9)
    on = _run(p_on, req, 9, tile_prefetch="1",
              tile_prefetch_available_bytes=10**12)
    _assert_results_bit_equal(off, on)
    assert p_on.calls == p_off.calls
    # tile-major, frame-minor, every (tile, frame) exactly once.
    assert len(p_on.calls) == len({(c[0], c[1], c[2], c[3], c[4]) for c in p_on.calls})
    ntiles = len({(c[1], c[2], c[3], c[4]) for c in p_on.calls})
    assert len(p_on.calls) == ntiles * len(images)
    for t in range(ntiles):
        block = p_on.calls[t * len(images):(t + 1) * len(images)]
        assert [c[0] for c in block] == list(range(len(images)))
    # Provider closed by the CALLER after the executor returned: the executor
    # (and its stopped producer) must have performed no access after close.
    p_on.close()
    assert p_on.reads_after_close == 0


# ---------------------------------------------------------------------------
# 4. Real overlap, proven with events (never with a timing threshold)
# ---------------------------------------------------------------------------

class _CoordinatedProvider(_RecordingProvider):
    def __init__(self, images, supports, taper_event, writes_event, entered_event,
                 checked_event):
        super().__init__(images, supports)
        self.taper_event = taper_event
        self.writes_event = writes_event
        self.entered_event = entered_event
        self.checked_event = checked_event
        self.overlap_observed = False
        self._tile_ids = []
        self._n = len(images)

    def get_tile(self, i, y0, y1, x0, x1):
        key = (y0, y1, x0, x1)
        if key not in self._tile_ids:
            self._tile_ids.append(key)
        if len(self._tile_ids) == 2 and i == 0:
            # Announce that we are INSIDE the second tile's read, then wait for
            # the consumer to have started the first tile's compute. The first
            # tile must finish AFTER this point for the assertion to be honest.
            self.entered_event.set()
            assert self.taper_event.wait(10.0), "consumer never entered compute"
            self.overlap_observed = not self.writes_event.is_set()
            self.checked_event.set()
        return super().get_tile(i, y0, y1, x0, x1)


def test_overlap_real_and_queue_max_one(sim_gpu):
    images, masks = _mono_corpus(n=4, h=30, w=30)
    taper_event = threading.Event()
    writes_event = threading.Event()
    entered_event = threading.Event()
    checked_event = threading.Event()
    provider = _CoordinatedProvider(
        images, masks, taper_event, writes_event, entered_event, checked_event
    )
    profiler = _HookProfiler(
        backend="gpu", taper_event=taper_event, writes_event=writes_event,
        entered_event=entered_event, checked_event=checked_event,
    )
    req = _request(images, masks, "gpu", normalization="none", weighting="none",
                   rejection="none", combine="mean", taper="none")
    run = _run(provider, req, 8, tile_prefetch="1",
               tile_prefetch_available_bytes=10**12, phase_profiler=profiler)
    assert provider.overlap_observed is True
    snap = profiler.snapshot()
    pref = snap["tile_prefetch"]
    assert pref["queue_max"] <= 1
    assert pref["queue_capacity"] == 1
    assert pref["tiles_prefetched"] == pref["tiles_total"]
    assert pref["tiles_consumed"] == pref["tiles_total"]
    assert pref["provider_reads"] == pref["tiles_total"] * len(images)
    assert pref["producer_alive_after_join"] is False
    assert pref["overlap_s"] is not None and pref["overlap_s"] > 0.0
    assert run.provenance["tile_prefetch"]["effective"] == 1


# ---------------------------------------------------------------------------
# 5. Terminal paths: provider error, consumer error, cancel, OOM-like
# ---------------------------------------------------------------------------

def _prefetch_threads():
    return [t for t in threading.enumerate() if t.name == _PREFETCH_THREAD]


class _FailingProvider(_RecordingProvider):
    def __init__(self, images, supports, fail_at):
        super().__init__(images, supports)
        self.fail_at = int(fail_at)
        self._count = 0

    def get_tile(self, i, y0, y1, x0, x1):
        if self._count == self.fail_at:
            self._count += 1
            raise _Boom(f"provider failure at call {self.fail_at}")
        self._count += 1
        return super().get_tile(i, y0, y1, x0, x1)


@pytest.mark.parametrize("which", ["first", "middle", "last"])
def test_provider_exception_preserved_and_thread_joined(sim_gpu, which):
    images, masks = _mono_corpus(n=4, h=30, w=30)
    req = _request(images, masks, "gpu", normalization="none", weighting="none",
                   rejection="none", combine="mean", taper="none")
    n_tiles = 4 * 4  # 30/8 -> 4x4
    total = n_tiles * len(images)
    fail_at = {"first": 0, "middle": total // 2, "last": total - 1}[which]
    provider = _FailingProvider(images, masks, fail_at)
    with pytest.raises(_Boom):
        _run(provider, req, 8, tile_prefetch="1",
             tile_prefetch_available_bytes=10**12)
    assert _prefetch_threads() == []
    provider.close()
    assert provider.reads_after_close == 0


def test_consumer_exception_preserved_thread_joined_and_no_late_reads(sim_gpu):
    images, masks = _mono_corpus(n=4, h=30, w=30)
    provider = _RecordingProvider(images, masks)
    profiler = _HookProfiler(
        backend="gpu", raise_phase="taper_weights_s",
        raise_exc=_Boom("consumer"), raise_after=2,
    )
    req = _request(images, masks, "gpu", normalization="none", weighting="none",
                   rejection="none", combine="mean", taper="none")
    with pytest.raises(_Boom):
        _run(provider, req, 8, tile_prefetch="1",
             tile_prefetch_available_bytes=10**12, phase_profiler=profiler)
    assert _prefetch_threads() == []
    calls_at_return = len(provider.calls)
    provider.close()
    assert provider.reads_after_close == 0
    assert len(provider.calls) == calls_at_return  # no late producer read


def test_cancellation_stops_and_joins(sim_gpu):
    images, masks = _mono_corpus(n=4, h=30, w=30)
    provider = _RecordingProvider(images, masks)
    profiler = _HookProfiler(
        backend="gpu", raise_phase="writes_s",
        raise_exc=KeyboardInterrupt(), raise_after=1,
    )
    req = _request(images, masks, "gpu", normalization="none", weighting="none",
                   rejection="none", combine="mean", taper="none")
    with pytest.raises(KeyboardInterrupt):
        _run(provider, req, 8, tile_prefetch="1",
             tile_prefetch_available_bytes=10**12, phase_profiler=profiler)
    assert _prefetch_threads() == []
    provider.close()
    assert provider.reads_after_close == 0


def test_gpu_oom_like_consumer_error_keeps_classification(sim_gpu):
    images, masks = _mono_corpus(n=4, h=30, w=30)
    provider = _RecordingProvider(images, masks)
    profiler = _HookProfiler(
        backend="gpu", raise_phase="reject_combine_host_s",
        raise_exc=_FakeCupyOOM("out of memory"), raise_after=1,
    )
    req = _request(images, masks, "gpu", normalization="none", weighting="none",
                   rejection="none", combine="mean", taper="none")
    with pytest.raises(_FakeCupyOOM):
        _run(provider, req, 8, tile_prefetch="1",
             tile_prefetch_available_bytes=10**12, phase_profiler=profiler)
    assert _prefetch_threads() == []
    # The existing historical path (fallback target) still runs normally.
    hist = _run(InMemoryCanonicalProvider(images, masks), req, None)
    assert hist.science.shape == (30, 30)


# ---------------------------------------------------------------------------
# 6. Admission refusal = historical path, identical science
# ---------------------------------------------------------------------------

def test_refused_admission_runs_historical_path(sim_gpu):
    images, masks = _mono_corpus(n=4, h=26, w=26)
    req = _request(images, masks, "gpu", normalization="sky_mean",
                   weighting="none", rejection="kappa_sigma",
                   combine="mean", taper="footprint",
                   taper_px=4.0, taper_floor=0.0)
    p_off = _RecordingProvider(images, masks)
    p_ref = _RecordingProvider(images, masks)
    off = _run(p_off, req, 7)
    refused = _run(p_ref, req, 7, tile_prefetch="auto",
                   tile_prefetch_available_bytes=1)
    assert refused.provenance["tile_prefetch"]["effective"] == 0
    assert refused.provenance["tile_prefetch"]["reason"] == "insufficient_available_ram"
    _assert_results_bit_equal(off, refused)
    assert p_ref.calls == p_off.calls


# ---------------------------------------------------------------------------
# 7. Profiling OFF adds no timer/event/sync; ON publishes coherent topology
# ---------------------------------------------------------------------------

def test_profiling_off_uses_no_timer(sim_gpu, monkeypatch):
    class _NoClock:
        def perf_counter(self):  # pragma: no cover - must never run
            raise AssertionError("perf_counter called with profiling OFF")

    # ``time`` is only ever used for optional profiling in canonical_streaming.
    monkeypatch.setattr(cstream, "time", _NoClock())
    images, masks = _rgb_corpus()
    req = _request(images, masks, "gpu", normalization="sky_mean",
                   weighting="noise_variance", rejection="kappa_sigma",
                   combine="mean", taper="footprint",
                   taper_px=3.0, taper_floor=0.0)
    on = _run(
        InMemoryCanonicalProvider(images, masks), req, 8,
        tile_prefetch="1", tile_prefetch_available_bytes=10**12,
    )
    assert on.provenance["tile_prefetch"]["effective"] == 1
    off = _run(InMemoryCanonicalProvider(images, masks), req, 8)
    _assert_results_bit_equal(off, on)


def test_profiling_on_topology_and_counters(sim_gpu):
    images, masks = _mono_corpus(n=4, h=30, w=30)
    profiler = _HookProfiler(backend="gpu")
    req = _request(images, masks, "gpu", normalization="none", weighting="none",
                   rejection="none", combine="mean", taper="none")
    _run(InMemoryCanonicalProvider(images, masks), req, 8,
         tile_prefetch="1", tile_prefetch_available_bytes=10**12,
         phase_profiler=profiler)
    snap = profiler.snapshot()
    assert "tile_prefetch" in snap
    assert snap["tile_prefetch"]["queue_max"] <= 1
    assert snap["tile_prefetch"]["overlap_s"] is not None
    topo = snap["phase_topology"]
    assert "overlap_with_tile_prefetch" in topo
    _tile_read_doc = topo["overlap_with_tile_prefetch"]["tile_read_s"]
    assert "DISJOINT" in _tile_read_doc
    assert "never compare tile_read_s" in _tile_read_doc
    # Counters are internally coherent.
    assert snap["tile_prefetch"]["provider_reads"] == snap["frames_read"]
    assert snap["tile_prefetch"]["tiles_consumed"] == snap["tiles"]


# ---------------------------------------------------------------------------
# 8. Ledger term: separate from the inter-cell prefetch credit
# ---------------------------------------------------------------------------

def test_ledger_slot_term_is_separate_and_additive():
    classic = zstream.estimate_streaming_peak_kib(6, (40, 40), 8)
    with_slot = zstream.estimate_streaming_peak_kib(
        6, (40, 40), 8, tile_prefetch_slots=1
    )
    slot_kib = zstream.estimate_tile_prefetch_slot_kib(6, (40, 40), 8)
    assert with_slot - classic == slot_kib
    # The slot data is ONE extended tile for every frame, never the full patch
    # (no double charge with the inter-cell prefetch credit's full-patch copy).
    slot_data = estimate_tile_prefetch_slot_bytes(
        6, 40, 40, 3, tile_size=8, halo_px=zstream._TAPER_HALO_PX
    ) - MIB
    assert slot_data == 6 * 26 * 26 * (4 * 3 + 1)
    assert slot_data < 6 * 40 * 40 * (4 * 3 + 1)
    # Same value at zero slots: every historical plan/hash is byte-identical.
    assert zstream.estimate_streaming_peak_kib(
        6, (40, 40), 8, tile_prefetch_slots=0
    ) == classic


# ---------------------------------------------------------------------------
# 9. Real file-backed provider (product-shaped) parity
# ---------------------------------------------------------------------------

def test_file_provider_prefetch_parity(sim_gpu, tmp_path):
    images, masks = _mono_corpus(n=5, h=28, w=28)
    frame_ids = [f"f{i}" for i in range(len(images))]
    cache_dir = tmp_path / "cache"
    write_aligned_cache_from_arrays(
        cache_dir, frame_ids,
        [np.ascontiguousarray(np.repeat(f[..., None], 1, axis=-1)) for f in images],
        masks,
    )
    req = _request(images, masks, "gpu", normalization="sky_mean",
                   weighting="none", rejection="winsorized_sigma_clip",
                   combine="mean", taper="footprint",
                   taper_px=4.0, taper_floor=0.0)
    off_provider = MemmapCanonicalProvider(cache_dir)
    try:
        off = _run(off_provider, req, 9)
    finally:
        off_provider.close()
    on_provider = MemmapCanonicalProvider(cache_dir)
    try:
        on = _run(on_provider, req, 9, tile_prefetch="1",
                  tile_prefetch_available_bytes=10**12)
    finally:
        on_provider.close()
    assert on.provenance["tile_prefetch"]["effective"] == 1
    _assert_results_bit_equal(off, on)


# ===========================================================================
# REWORK R1 (F1–F4)
# ===========================================================================

import weakref

from zemosaic import zemosaic_zegrid_mode as zz
from zemosaic.core.zegrid import science_adapter as zsci
from zemosaic.core.zegrid.geometry import GlobalBounds, PatchBounds, ProcessingPatch


class _ResidencyProvider(_RecordingProvider):
    """Tracks live provider image arrays with weakref finalizers."""

    def __init__(self, images, supports):
        super().__init__(images, supports)
        self.alive_images = 0
        self.max_alive_images = 0
        self.alive_at_tile_start = []
        self._tile_ids = []

    def _dec(self):
        self.alive_images -= 1

    def get_tile(self, i, y0, y1, x0, x1):
        key = (y0, y1, x0, x1)
        if key not in self._tile_ids:
            self._tile_ids.append(key)
            self.alive_at_tile_start.append(int(self.alive_images))
        img, valid = super().get_tile(i, y0, y1, x0, x1)
        self.alive_images += 1
        self.max_alive_images = max(self.max_alive_images, self.alive_images)
        weakref.finalize(img, self._dec)
        return img, valid


def test_f1_slot_released_only_after_raw_set_dropped(sim_gpu):
    images, masks = _mono_corpus(n=4, h=16, w=24)
    provider = _ResidencyProvider(images, masks)
    req = _request(images, masks, "gpu", normalization="none", weighting="none",
                   rejection="none", combine="mean", taper="none")
    _run(provider, req, 8, tile_prefetch="1",
         tile_prefetch_available_bytes=10**12)
    # At the moment tile N+1 starts being built, the raw set of tile N is gone.
    assert provider.alive_at_tile_start[1] == 0
    # Never two complete prepared image sets at once.
    assert provider.max_alive_images <= len(images)


def test_f3_consumer_wait_does_not_inflate_overlap(sim_gpu):
    # Two tiles. The producer is gated so tile 1 is read only AFTER tile 0's
    # consumer work has ended: the whole tile-1 read happens during the
    # consumer's wait. Overlap must therefore be exactly 0.
    images, masks = _mono_corpus(n=3, h=8, w=16)
    tile0_done = threading.Event()
    provider = _RecordingProvider(images, masks)
    orig = provider.get_tile
    tile_ids = []

    def gated_get_tile(i, y0, y1, x0, x1):
        key = (y0, y1, x0, x1)
        if key not in tile_ids:
            tile_ids.append(key)
        if len(tile_ids) == 2 and i == 0:
            assert tile0_done.wait(10.0), "consumer never finished tile 0"
        return orig(i, y0, y1, x0, x1)

    provider.get_tile = gated_get_tile
    profiler = _HookProfiler(backend="gpu", writes_event=tile0_done)
    req = _request(images, masks, "gpu", normalization="none", weighting="none",
                   rejection="none", combine="mean", taper="none")
    _run(provider, req, 8, tile_prefetch="1",
         tile_prefetch_available_bytes=10**12, phase_profiler=profiler)
    pref = profiler.snapshot()["tile_prefetch"]
    assert pref["consumer_wait_count"] >= 1
    assert pref["consumer_wait_s"] is not None and pref["consumer_wait_s"] > 0.0
    assert pref["overlap_s"] == 0.0


def test_f3_runtime_counters_published_without_profiling(sim_gpu):
    images, masks = _mono_corpus(n=3, h=16, w=16)
    req = _request(images, masks, "gpu", normalization="none", weighting="none",
                   rejection="none", combine="mean", taper="none")
    res = _run(InMemoryCanonicalProvider(images, masks), req, 8,
               tile_prefetch="1", tile_prefetch_available_bytes=10**12)
    pref = res.provenance["tile_prefetch"]
    assert pref["effective"] == 1
    assert pref["queue_max"] <= 1 and pref["queue_capacity"] == 1
    assert pref["provider_reads"] == pref["tiles_total"] * len(images)
    assert pref["tiles_prefetched"] == pref["tiles_total"]
    assert pref["tiles_consumed"] == pref["tiles_total"]
    assert pref["producer_alive_after_join"] is False
    # Structural counters are present; durations are absent when OFF.
    assert isinstance(pref["consumer_wait_count"], int)
    assert pref["consumer_wait_s"] is None
    assert pref["producer_work_s"] is None
    assert pref["overlap_s"] is None


def _required_for(n, h, w, c, tile, halo=0):
    slot = estimate_tile_prefetch_slot_bytes(n, h, w, c, tile, halo)
    return slot, (slot * 115 + 99) // 100


def test_f2_run_budget_headroom_admission_and_projection(sim_gpu):
    images, masks = _mono_corpus(n=3, h=16, w=16)
    req = _request(images, masks, "gpu", normalization="none", weighting="none",
                   rejection="none", combine="mean", taper="none")
    slot, required = _required_for(len(images), 16, 16, 1, 8)
    base = 10_000_000
    # Raw MemAvailable is enormous, but the 90 % RUN headroom is too small.
    refused = _run(InMemoryCanonicalProvider(images, masks), req, 8,
                   tile_prefetch="auto", tile_prefetch_available_bytes=10**12,
                   tile_prefetch_budget_bytes=required - 1,
                   tile_prefetch_base_aggregate_bytes=base,
                   tile_prefetch_host_budget_bytes=base + required - 1)
    pref = refused.provenance["tile_prefetch"]
    assert pref["effective"] == 0
    assert pref["reason"] == "insufficient_run_budget"
    assert pref["projected_aggregate_bytes"] == base
    assert pref["base_aggregate_bytes"] == base
    # Exactly enough headroom (and raw RAM) -> admitted with an exact projection.
    admitted = _run(InMemoryCanonicalProvider(images, masks), req, 8,
                    tile_prefetch="auto", tile_prefetch_available_bytes=10**12,
                    tile_prefetch_budget_bytes=required,
                    tile_prefetch_base_aggregate_bytes=base,
                    tile_prefetch_host_budget_bytes=base + required)
    pref = admitted.provenance["tile_prefetch"]
    assert pref["effective"] == 1
    assert pref["required_bytes"] == required
    assert pref["budget_headroom_bytes"] == required
    assert pref["projected_aggregate_bytes"] == base + required


def _product_patch(h, w):
    return ProcessingPatch(
        cell_id="cell",
        patch=GlobalBounds(x0=0, y0=0, x1=w, y1=h),
        core_slice=PatchBounds(x0=0, y0=0, x1=w, y1=h),
        halo_px=0,
        patch_wcs_header="",
    )


def _product_config(backend="gpu"):
    return zsci.MiniTileScienceConfig(
        normalization="none", weighting="none", rejection="none",
        combine="mean", taper="none", backend=backend,
    )


def test_f2_product_run_budget_refusal_and_ledger(sim_gpu, tmp_path):
    images, masks = _mono_corpus(n=3, h=16, w=16)
    cache = tmp_path / "cache"
    write_aligned_cache_from_arrays(
        cache, [f"f{i}" for i in range(len(images))],
        [np.ascontiguousarray(f[..., None]) for f in images], masks,
    )
    patch = _product_patch(16, 16)
    slot, required = _required_for(len(images), 16, 16, 1, 8)
    base = 5_000_000
    info = {}
    zz._run_cell_stream(
        cache, patch, _product_config(), None, tile_size=8,
        storage_profile="dense", storage_info=info,
        tile_prefetch="auto", tile_prefetch_available_bytes=10**12,
        tile_prefetch_budget_bytes=1,
        tile_prefetch_base_aggregate_bytes=base,
        tile_prefetch_host_budget_bytes=base + 1,
    )
    pref = info["tile_prefetch"]
    assert pref["effective"] == 0
    assert pref["reason"] == "insufficient_run_budget"
    assert pref["base_aggregate_bytes"] == base
    assert pref["host_budget_bytes"] == base + 1
    assert pref["budget_headroom_bytes"] == 1
    assert pref["projected_aggregate_bytes"] == base

    info_ok = {}
    zz._run_cell_stream(
        cache, patch, _product_config(), None, tile_size=8,
        storage_profile="dense", storage_info=info_ok,
        tile_prefetch="auto", tile_prefetch_available_bytes=10**12,
        tile_prefetch_budget_bytes=required,
        tile_prefetch_base_aggregate_bytes=base,
        tile_prefetch_host_budget_bytes=base + required,
    )
    pref = info_ok["tile_prefetch"]
    assert pref["effective"] == 1
    assert pref["projected_aggregate_bytes"] == base + required
    # Runtime counters (not just the admission plan) are in the product ledger.
    assert pref["provider_reads"] == pref["tiles_total"] * len(images)
    assert pref["queue_max"] <= 1


def test_f4_uninterruptible_read_is_an_explicit_failure_not_success(
    sim_gpu, monkeypatch
):
    monkeypatch.setattr(cstream, "_TILE_PREFETCH_JOIN_TIMEOUT_S", 0.05)
    images, masks = _mono_corpus(n=3, h=16, w=16)
    release = threading.Event()
    entered = threading.Event()
    provider = _RecordingProvider(images, masks)
    orig = provider.get_tile
    tile_ids = []

    def blocking_get_tile(i, y0, y1, x0, x1):
        key = (y0, y1, x0, x1)
        if key not in tile_ids:
            tile_ids.append(key)
        if len(tile_ids) == 2 and i == 0:
            entered.set()
            release.wait(10.0)  # simulates an uninterruptible provider call
        return orig(i, y0, y1, x0, x1)

    provider.get_tile = blocking_get_tile
    # Wait until the producer is REALLY inside the blocked next read before the
    # consumer raises, so the test is deterministic.
    class _WaitEntered(_HookProfiler):
        def add(self, phase, seconds):
            super().add(phase, seconds)
            if phase == "taper_weights_s" and self._seen[phase] == 1:
                assert entered.wait(10.0), "producer never entered the blocked read"

    profiler = _WaitEntered(
        backend="gpu", raise_phase="writes_s", raise_exc=_Boom("consumer"),
        raise_after=1,
    )
    req = _request(images, masks, "gpu", normalization="none", weighting="none",
                   rejection="none", combine="mean", taper="none")
    with pytest.raises(_Boom) as ei:
        _run(provider, req, 8, tile_prefetch="1",
             tile_prefetch_available_bytes=10**12, phase_profiler=profiler)
    # Root exception preserved, annotated, with the truthful report attached.
    report = ei.value.prefetch_report
    assert report["producer_alive_after_join"] is True
    assert any("not interruptible" in n for n in ei.value.__notes__)
    release.set()
    for t in _prefetch_threads():
        t.join(5.0)
    assert _prefetch_threads() == []
    provider.close()
    assert provider.reads_after_close == 0


def test_f4_cooperative_blocked_read_joins_cleanly(sim_gpu, monkeypatch):
    monkeypatch.setattr(cstream, "_TILE_PREFETCH_JOIN_TIMEOUT_S", 0.5)
    images, masks = _mono_corpus(n=3, h=16, w=16)
    release = threading.Event()
    provider = _RecordingProvider(images, masks)
    orig = provider.get_tile
    tile_ids = []

    def slow_get_tile(i, y0, y1, x0, x1):
        key = (y0, y1, x0, x1)
        if key not in tile_ids:
            tile_ids.append(key)
        if len(tile_ids) == 2 and i == 0:
            release.wait(10.0)
        return orig(i, y0, y1, x0, x1)

    provider.get_tile = slow_get_tile
    timer = threading.Timer(0.05, release.set)
    timer.start()
    try:
        req = _request(images, masks, "gpu", normalization="none",
                       weighting="none", rejection="none", combine="mean",
                       taper="none")
        _run(provider, req, 8, tile_prefetch="1",
             tile_prefetch_available_bytes=10**12)
    finally:
        timer.cancel()
        release.set()
    assert _prefetch_threads() == []
    # Normal success implies a clean join.
    assert provider.reads_after_close == 0


# ===========================================================================
# REWORK R2 (F4 ownership/lifetime, F3 bucket count)
# ===========================================================================

def test_f3_tile_read_count_exact_off_and_on(sim_gpu):
    images, masks = _mono_corpus(n=4, h=30, w=30)
    req = _request(images, masks, "gpu", normalization="none", weighting="none",
                   rejection="none", combine="mean", taper="none")
    pf_off = _HookProfiler(backend="gpu")
    _run(InMemoryCanonicalProvider(images, masks), req, 8,
         phase_profiler=pf_off)
    pf_on = _HookProfiler(backend="gpu")
    _run(InMemoryCanonicalProvider(images, masks), req, 8,
         tile_prefetch="1", tile_prefetch_available_bytes=10**12,
         phase_profiler=pf_on)
    off = pf_off.snapshot()
    on = pf_on.snapshot()
    assert off["tiles"] == on["tiles"] == 16
    # Exactly ONE tile_read_s add per tile in BOTH modes.
    assert off["host_phases"]["tile_read_s"]["count"] == 16
    assert on["host_phases"]["tile_read_s"]["count"] == 16


class _BlockingProductProvider:
    """Product-level provider wrapper: blocks one read, tracks close/read state."""

    def __init__(self, inner, release, entered):
        self._inner = inner
        self._release = release
        self._entered = entered
        self.closed = False
        self.close_event = threading.Event()
        self.reads_after_close = 0
        self._tile_ids = []
        self.n_frames = inner.n_frames
        self.height = inner.height
        self.width = inner.width
        self.channels = inner.channels
        self.original_mono = inner.original_mono
        self.original_ndim = inner.original_ndim
        self.original_shape = inner.original_shape
        self.frame_ids = inner.frame_ids

    @property
    def manifest(self):
        return self._inner.manifest

    def get_raw_frame(self, i):
        if self.closed:
            self.reads_after_close += 1
        return self._inner.get_raw_frame(i)

    def get_tile(self, i, y0, y1, x0, x1):
        key = (y0, y1, x0, x1)
        if key not in self._tile_ids:
            self._tile_ids.append(key)
        if len(self._tile_ids) == 2 and i == 0:
            self._entered.set()
            self._release.wait(10.0)
        if self.closed:
            self.reads_after_close += 1
        return self._inner.get_tile(i, y0, y1, x0, x1)

    def close(self):
        if not self.closed:
            self.closed = True
            self._inner.close()
        self.close_event.set()


class _WaitEnteredProfiler(_HookProfiler):
    def __init__(self, *a, entered=None, **kw):
        super().__init__(*a, **kw)
        self.entered = entered

    def add(self, phase, seconds):
        super().add(phase, seconds)
        if phase == "taper_weights_s" and self._seen[phase] == 1:
            assert self.entered.wait(10.0), "producer never entered the read"


def test_f4_product_defers_provider_close_and_retains_cache(sim_gpu, monkeypatch,
                                                           tmp_path):
    monkeypatch.setattr(cstream, "_TILE_PREFETCH_JOIN_TIMEOUT_S", 0.05)
    images, masks = _mono_corpus(n=3, h=16, w=16)
    cache = tmp_path / "cache"
    write_aligned_cache_from_arrays(
        cache, [f"f{i}" for i in range(len(images))],
        [np.ascontiguousarray(f[..., None]) for f in images], masks,
    )
    release = threading.Event()
    entered = threading.Event()
    inner = MemmapCanonicalProvider(cache)
    provider = _BlockingProductProvider(inner, release, entered)
    monkeypatch.setattr(
        zz, "_make_cell_provider", lambda cache_dir, profile: provider
    )
    profiler = _WaitEnteredProfiler(
        backend="gpu", entered=entered, raise_phase="writes_s",
        raise_exc=_Boom("consumer"), raise_after=1,
    )
    info = {}
    with pytest.raises(_Boom) as ei:
        zz._run_cell_stream(
            cache, _product_patch(16, 16), _product_config(), None,
            tile_size=8, storage_profile="dense", storage_info=info,
            phase_profiler=profiler,
            tile_prefetch="1", tile_prefetch_available_bytes=10**12,
            tile_prefetch_budget_bytes=10**12,
            tile_prefetch_base_aggregate_bytes=0,
            tile_prefetch_host_budget_bytes=10**12,
        )
    assert ei.value.prefetch_report["producer_alive_after_join"] is True
    # Provider is NOT closed, cache still exists, no read-after-close.
    assert provider.closed is False
    assert cache.exists()
    assert provider.reads_after_close == 0
    assert info["tile_prefetch_deferred_cleanup"] is True
    assert info["tile_prefetch_cache_retained_due_live_prefetch"] is True
    # Release the real read: the producer's own final block closes exactly once.
    release.set()
    assert provider.close_event.wait(10.0)
    for t in _prefetch_threads():
        t.join(5.0)
    assert _prefetch_threads() == []
    assert provider.closed is True
    assert provider.reads_after_close == 0
    # The cache is deliberately retained on this terminal failure (diagnostic).
    assert cache.exists()
    provider.close()  # idempotent
    assert provider.reads_after_close == 0


def test_f4_release_callback_retains_live_cache_and_deletes_clean(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "frame.npy").write_text("x")
    payload = {
        "_unit_index": 3,
        "cache_dir": str(cache),
        "_tile_prefetch_deferred_cleanup": True,
    }
    assert zz._release_unit_cache(payload) is False
    assert cache.exists()
    assert payload["_tile_prefetch_cache_retained_due_live_prefetch"] is True
    clean = {"_unit_index": 4, "cache_dir": str(cache)}
    assert zz._release_unit_cache(clean) is True
    assert not cache.exists()


def test_f4_payload_marker_propagates_from_consume(monkeypatch):
    monkeypatch.setattr(zz, "_STACK_CONFIG", zsci.MiniTileScienceConfig(
        normalization="none", weighting="none", rejection="none",
        combine="mean", taper="none", backend="gpu",
    ))
    def _boom(*a, **kw):
        kw["storage_info"]["tile_prefetch_deferred_cleanup"] = True
        raise _Boom("stream failed")
    monkeypatch.setattr(zz, "_run_cell_stream", _boom)
    import types
    prepared = {
        "cell": types.SimpleNamespace(cell_id="c", row=0, col=0),
        "patch": object(), "mem": types.SimpleNamespace(patch_ids=[1]),
        "cache_dir": "/tmp/none", "mode": "stream", "bound": 1,
        "patch_bounds": (0, 1, 0, 1), "patch_shape_hw": (1, 1),
        "manifest": {"n_frames": 1, "total_bytes": 0}, "cache_build_s": 0.0,
        "reproject": None, "cell_gauge_mode": "global",
        "cell_frame_ids": ("f0",), "cell_fixed": None,
    }
    with pytest.raises(_Boom):
        zz._consume_stack_cell(prepared)
    assert prepared["_tile_prefetch_deferred_cleanup"] is True
    assert prepared["_tile_prefetch_cache_retained_due_live_prefetch"] is True


def test_f4_deferred_close_exact_once_and_finish_race(sim_gpu, monkeypatch):
    monkeypatch.setattr(cstream, "_TILE_PREFETCH_JOIN_TIMEOUT_S", 0.05)
    calls = []
    release = threading.Event()
    entered = threading.Event()
    images, masks = _mono_corpus(n=2, h=8, w=16)
    provider = _RecordingProvider(images, masks)
    orig = provider.get_tile

    def blocking(i, y0, y1, x0, x1):
        if i == 0:
            entered.set()
            release.wait(10.0)
        return orig(i, y0, y1, x0, x1)

    provider.get_tile = blocking
    pf = cstream._TilePrefetcher(
        provider, list(cstream._iter_tiles(8, 16, 8)), 2, 8, 16, 0, timed=False
    )
    pf.start()
    assert entered.wait(10.0)
    # The producer is alive and inside the read: arm a deferred close.
    assert pf.arm_deferred_close(lambda: calls.append(1)) is True
    release.set()
    # No consumer in this unit test: stop/drain/join the producer directly.
    pf.shutdown()
    for t in _prefetch_threads():
        t.join(5.0)
    assert calls == [1]
    # Thread finished: a second arm must refuse and never double-call.
    assert pf.arm_deferred_close(lambda: calls.append(2)) is False
    assert calls == [1]
