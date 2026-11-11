"""ZM-ZEGRID-FD-EXHAUSTION — bounded cache descriptors + descriptor-limit hardening.

Production incident this locks down: a 527-frame ZeGrid GPU run died at cell
``r0000c0001`` with ``OSError: [Errno 24] Too many open files``. The aligned
cache is 2 ``.npy`` files per frame (~1054 for one cell) and the dense provider
opened one ``np.load(mmap_mode="r")`` mapping per file at construction, in a
process whose soft ``RLIMIT_NOFILE`` was 1024.

Coverage (mission required tests 1-4):

1. bounded handles: a wide synthetic cell never holds more than the mapping cap
   open, measured both from the provider's own truthful telemetry and from
   ``/proc/self/fd`` while the REAL streaming executor runs to completion;
2. low limit: the fail-closed refusal happens BEFORE any cache wipe/write (a
   pre-existing cache is preserved byte-for-byte and no Cell can be committed),
   and a limit that fits completes normally;
3. limit hardening: raise attempted + recorded, denial is a recorded no-op, and
   a non-POSIX platform is an explicit no-op — never a crash, never a lowering;
4. science invariance: cache bytes are digest-identical across builds and the
   bounded provider serves bit-identical values to the eager one.
"""

from __future__ import annotations

import hashlib
import os
from types import SimpleNamespace

import numpy as np
import pytest

import zemosaic.core.canonical_streaming as cs
from zemosaic.core.canonical_engine import CanonicalStackRequest
from zemosaic.core.canonical_streaming import (
    InMemoryCanonicalProvider,
    run_canonical_stack_streaming,
)
from zemosaic.core.zegrid import fd_limits as zfdl
from zemosaic.core.zegrid import file_provider as zfp
from zemosaic import zemosaic_zegrid_mode as zz


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mono_corpus(n=40, h=48, w=48, seed=11):
    rng = np.random.default_rng(seed)
    base = np.full((h, w), 100.0, dtype=np.float64)
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(5):
        cy = rng.uniform(0.2, 0.8) * h
        cx = rng.uniform(0.2, 0.8) * w
        base += 30.0 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.5 ** 2))
    frames = [
        (base * np.float32(1.0 + 0.002 * i) + rng.normal(0.0, 2.0, (h, w))).astype(
            np.float32
        )
        for i in range(n)
    ]
    masks = [np.ones((h, w), dtype=bool) for _ in range(n)]
    masks[1][8:12, 8:12] = False
    return frames, masks


def _request(frames, masks):
    return CanonicalStackRequest(
        images=frames,
        geometric_support=masks,
        normalization="none",
        weighting="none",
        rejection="none",
        combine="mean",
        taper="none",
    )


def _write_cache(cache, frames, masks):
    ids = [f"frame_{i}" for i in range(len(frames))]
    zfp.write_aligned_cache_from_arrays(cache, ids, frames, masks)
    return cache


def _tree_digest(root) -> dict:
    out = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            out[str(path.relative_to(root))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return out


def _fd_count() -> int | None:
    return zfdl.count_open_fds()


class _FakeResource:
    """Minimal ``resource`` stand-in (POSIX semantics, no real limits touched)."""

    RLIMIT_NOFILE = 7
    RLIM_INFINITY = -1

    def __init__(self, soft, hard, *, deny=False, ignore_set=False):
        self.soft = soft
        self.hard = hard
        self.deny = deny
        self.ignore_set = ignore_set
        self.calls = []

    def getrlimit(self, which):
        assert which == self.RLIMIT_NOFILE
        return (self.soft, self.hard)

    def setrlimit(self, which, values):
        assert which == self.RLIMIT_NOFILE
        self.calls.append(tuple(values))
        if self.deny:
            raise PermissionError(1, "Operation not permitted")
        if not self.ignore_set:
            self.soft, self.hard = values[0], values[1]


# ---------------------------------------------------------------------------
# 1 — bounded handles
# ---------------------------------------------------------------------------

def test_cache_build_holds_no_per_frame_handles(tmp_path):
    """The WRITER never accumulates handles: np.save closes each file."""
    frames, masks = _mono_corpus(n=120, h=24, w=24)
    cache = tmp_path / "cache"
    baseline = _fd_count()
    samples = []
    original_add = zfp.AlignedCacheBuilder.add

    def sampling_add(self, frame_id, rgb, support):
        result = original_add(self, frame_id, rgb, support)
        if baseline is not None:
            samples.append(_fd_count() - baseline)
        return result

    zfp.AlignedCacheBuilder.add = sampling_add
    try:
        _write_cache(cache, frames, masks)
    finally:
        zfp.AlignedCacheBuilder.add = original_add

    assert (cache / "manifest.json").is_file()
    assert len(list(cache.glob("frame_*_rgb.npy"))) == 120
    assert samples, "no write was sampled"
    assert max(samples) <= 4, f"writer leaked handles across frames: {max(samples)}"


def test_provider_mapping_cap_is_enforced_and_measured(tmp_path):
    frames, masks = _mono_corpus(n=60, h=24, w=24)
    cache = _write_cache(_cache_dir(tmp_path), frames, masks)
    cap = 8  # files => 4 frames
    baseline = _fd_count()
    provider = zfp.MemmapCanonicalProvider(cache, max_open_mappings=cap)
    deltas = []
    try:
        assert provider.n_frames == 60
        # 60 frames x 2 files = 120 cache files, but at most `cap` are mapped.
        assert provider.max_open_mappings == cap
        for i in range(provider.n_frames):
            provider.get_raw_frame(i)
            provider.get_tile(i, 2, 10, 3, 11)
            assert provider.open_mappings <= cap
            if baseline is not None:
                deltas.append(_fd_count() - baseline)
        tele = provider.fd_telemetry()
        assert tele["peak_open_mappings"] <= cap
        assert tele["cache_files"] == 120
        assert tele["eager"] is False
        if deltas:
            assert max(deltas) <= cap + 4, deltas
    finally:
        provider.close()
    assert provider.open_mappings == 0
    assert provider._rgb == [] and provider._sup == []


def test_streaming_executor_completes_with_bounded_mappings(tmp_path):
    """The REAL R5 executor runs a wide cell with a tiny mapping cap."""
    frames, masks = _mono_corpus(n=40, h=48, w=48)
    cache = _write_cache(_cache_dir(tmp_path), frames, masks)
    req = _request(frames, masks)

    dense = zfp.MemmapCanonicalProvider(cache, max_open_mappings=10 ** 6)
    assert dense._eager is True
    ref = run_canonical_stack_streaming(dense, req, tile_size=8)
    dense.close()

    base_fds = _fd_count()
    bounded = zfp.MemmapCanonicalProvider(cache, max_open_mappings=6)
    try:
        got = run_canonical_stack_streaming(bounded, req, tile_size=8)
        tele = bounded.fd_telemetry()
    finally:
        bounded.close()
    assert tele["peak_open_mappings"] <= 6
    if base_fds is not None:
        assert (_fd_count() - base_fds) <= 6
    for name in (
        "science",
        "estimator_weight_sum",
        "support_w1",
        "support_w2",
        "n_eff_support",
        "valid_mask",
        "surviving_sample_count",
    ):
        np.testing.assert_array_equal(
            getattr(ref, name), getattr(got, name), err_msg=name
        )
    assert ref.provenance == got.provenance


# ---------------------------------------------------------------------------
# 2 — low limit behaviour
# ---------------------------------------------------------------------------

def test_budget_fits_the_real_527_frame_incident():
    """The production shape (527 frames, soft 1024) now fits with huge margin."""
    est = zfdl.assert_cell_fd_budget(
        "cell r0000c0001",
        n_frames=527,
        base_fds=100,
        workers=4,
        effective_soft=1024,
    )
    assert est["cache_files"] == 1054
    assert est["resident_cache_files"] == zfdl.DEFAULT_MAX_OPEN_CACHE_FILES
    assert est["bounded"] is True
    assert est["required_fds"] <= 1024
    assert est["fits"] is True


def test_budget_refusal_is_actionable_and_names_the_numbers():
    with pytest.raises(zfdl.FdLimitExceeded) as excinfo:
        zfdl.assert_cell_fd_budget(
            "cell r0000c0001",
            n_frames=527,
            base_fds=1000,
            workers=4,
            max_open_files=64,
            effective_soft=1024,
        )
    msg = str(excinfo.value)
    for token in ("527", "1054", "1024", "ulimit -n", "Nothing was written"):
        assert token in msg, msg


def test_low_limit_fails_closed_before_any_cache_work(tmp_path, monkeypatch):
    """A refusal must precede the wipe/build: a pre-existing cache survives."""
    cache = tmp_path / "r0000c0000"
    cache.mkdir()
    sentinel = cache / "frame_0000_rgb.npy"
    sentinel.write_bytes(b"pre-existing-cache-bytes")

    fake_frames = [
        (
            SimpleNamespace(frame_id=SimpleNamespace(logical_path=f"f{i}.fits")),
            object(),
        )
        for i in range(50)
    ]
    monkeypatch.setattr(zz, "_plan_cell_source_frames", lambda *a, **k: fake_frames)
    monkeypatch.setattr(
        zfdl,
        "read_nofile_limits",
        lambda: {
            "platform": "linux",
            "supported": True,
            "soft": 8,
            "hard": 8,
            "infinite_hard": False,
            "error": None,
            "reason": "simulated",
        },
    )
    calls = []
    monkeypatch.setattr(zz.zpar, "pmap", lambda *a, **k: calls.append(a) or [])

    with pytest.raises(zfdl.FdLimitExceeded) as excinfo:
        zz._build_one_cell_cache(
            [],
            None,
            SimpleNamespace(cell_id="r0000c0000"),
            None,
            None,
            cache,
            workers=2,
        )
    assert "Nothing was written" in str(excinfo.value)
    # Nothing ran: no cache wipe, no worker spawn, cache bytes preserved.
    assert not calls
    assert sentinel.read_bytes() == b"pre-existing-cache-bytes"


def test_low_limit_is_fine_when_the_budget_fits(monkeypatch):
    monkeypatch.setattr(
        zfdl,
        "read_nofile_limits",
        lambda: {
            "platform": "linux",
            "supported": True,
            "soft": 256,
            "hard": 524288,
            "infinite_hard": False,
            "error": None,
            "reason": "simulated",
        },
    )
    est = zfdl.assert_cell_fd_budget(
        "cell r0000c0000", n_frames=60, base_fds=40, workers=2
    )
    assert est["required_fds"] <= 256
    assert est["fits"] is True


# ---------------------------------------------------------------------------
# 3 — limit hardening
# ---------------------------------------------------------------------------

def test_raise_below_hard_is_attempted_and_recorded(monkeypatch):
    fake = _FakeResource(512, 4096)
    monkeypatch.setattr(zfdl, "_resource_module", lambda: fake)
    rec = zfdl.raise_nofile_soft_limit()
    assert rec["attempted"] is True
    assert rec["raised"] is True
    assert rec["soft_before"] == 512
    assert rec["target"] == 4096  # min(hard, 131072)
    assert rec["soft_effective"] == 4096
    assert rec["hard"] == 4096
    assert fake.calls and fake.calls[0][0] == 4096
    # Never above the hard limit.
    assert fake.calls[0][0] <= rec["hard"]


def test_raise_denied_is_a_recorded_no_op(monkeypatch):
    fake = _FakeResource(512, 524288, deny=True)
    monkeypatch.setattr(zfdl, "_resource_module", lambda: fake)
    rec = zfdl.raise_nofile_soft_limit()
    assert rec["attempted"] is True
    assert rec["raised"] is False
    assert rec["soft_effective"] == 512
    assert "PermissionError" in (rec["error"] or "")


def test_raise_never_lowers_an_existing_limit(monkeypatch):
    fake = _FakeResource(131072, 524288)
    monkeypatch.setattr(zfdl, "_resource_module", lambda: fake)
    rec = zfdl.raise_nofile_soft_limit()
    assert rec["attempted"] is False
    assert rec["soft_effective"] == 131072
    assert fake.calls == []


def test_non_posix_is_an_explicit_no_op(monkeypatch):
    monkeypatch.setattr(zfdl, "_resource_module", lambda: None)
    rec = zfdl.raise_nofile_soft_limit()
    assert rec["supported"] is False
    assert rec["attempted"] is False
    assert rec["raised"] is False
    assert "non-POSIX" in rec["reason"]
    # The budget guard still works with no observed limit (treated as unlimited).
    est = zfdl.assert_cell_fd_budget("cell x", n_frames=1000)
    assert est["fits"] is True


def test_real_limit_is_never_lowered(monkeypatch):
    """Against the REAL POSIX limit: the soft limit only ever grows."""
    if os.name != "posix":
        pytest.skip("POSIX-only")
    zfdl.reset_cache()
    before = zfdl.read_nofile_limits()
    rec = zfdl.raise_nofile_soft_limit()
    after = zfdl.read_nofile_limits()
    assert rec["supported"] is True
    if before["soft"] is not None and after["soft"] is not None:
        assert after["soft"] >= before["soft"]
    if before["hard"] is not None and after["soft"] is not None:
        assert after["soft"] <= before["hard"]


# ---------------------------------------------------------------------------
# 4 — science invariance
# ---------------------------------------------------------------------------

def test_cache_bytes_are_digest_identical_across_builds(tmp_path):
    frames, masks = _mono_corpus(n=25, h=20, w=20)
    first_path = tmp_path / "a"
    second_path = tmp_path / "b"
    first = _write_cache(first_path, frames, masks)
    second = _write_cache(second_path, frames, masks)
    assert _tree_digest(first) == _tree_digest(second)
    first_manifest = zfp.load_cache_manifest(first)
    second_manifest = zfp.load_cache_manifest(second)
    assert first_manifest["total_bytes"] == second_manifest["total_bytes"]
    assert [f["rgb_sha256"] for f in first_manifest["frames"]] == [
        f["rgb_sha256"] for f in second_manifest["frames"]
    ]


def test_bounded_reads_are_bit_identical_and_do_not_touch_the_cache(tmp_path):
    frames, masks = _mono_corpus(n=30, h=24, w=24)
    cache = _write_cache(_cache_dir(tmp_path), frames, masks)
    before = _tree_digest(cache)
    provider = zfp.MemmapCanonicalProvider(cache, max_open_mappings=4)
    try:
        for i in range(provider.n_frames):
            raw, sup = provider.get_raw_frame(i)
            np.testing.assert_array_equal(raw, frames[i])
            np.testing.assert_array_equal(sup, masks[i])
    finally:
        provider.close()
    assert _tree_digest(cache) == before, "bounded reads must not rewrite the cache"


def _cache_dir(tmp_path):
    path = tmp_path / "cache"
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path
