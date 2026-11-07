"""A2a — WindowedCanonicalProvider is byte-identical to the aligned cache."""

from __future__ import annotations

import numpy as np

from zemosaic.core.zegrid import file_provider as fp


def test_windowed_provider_matches_aligned_cache_byte_for_byte(tmp_path):
    rng = np.random.default_rng(7)
    n, h, w = 3, 40, 40
    rgb = [np.ascontiguousarray(rng.normal(size=(h, w, 3)).astype(np.float32)) for _ in range(n)]
    sup = [rng.random((h, w)) > 0.1 for _ in range(n)]
    # deterministic adversarial content: NaN + outlier in one frame
    rgb[1][2:4, 2:4, :] = np.nan
    rgb[2][10:12, 10:12, :] = 1.0e6
    ids = [f"frame_{i}.fits" for i in range(n)]

    fp.write_aligned_cache_from_arrays(tmp_path / "cache", ids, rgb, sup)
    cache = fp.MemmapCanonicalProvider(tmp_path / "cache")

    def prepare(i):
        return rgb[i], sup[i]

    win = fp.WindowedCanonicalProvider(prepare, n, frame_ids=ids, expected_hw=(h, w))
    assert win.frame_ids == cache.frame_ids
    assert (win.height, win.width, win.channels) == (h, w, 3)
    assert win.live_window == 1

    for i in range(n):
        a, sa = win.get_raw_frame(i)
        b, sb = cache.get_raw_frame(i)
        assert a.dtype == np.float32
        assert a.tobytes() == np.ascontiguousarray(b, dtype=np.float32).tobytes()
        assert sa.tobytes() == np.ascontiguousarray(sb, dtype=bool).tobytes()

        ta, tsa = win.get_tile(i, 0, 10, 0, 12)
        tb, tsb = cache.get_tile(i, 0, 10, 0, 12)
        assert ta.tobytes() == np.ascontiguousarray(tb, dtype=np.float32).tobytes()
        assert tsa.tobytes() == np.ascontiguousarray(tsb, dtype=bool).tobytes()

    # Window discipline: only ONE preparation beyond the eager metadata frame.
    assert win.prepared_count <= n
    win.close()
    assert win._rgb is None
