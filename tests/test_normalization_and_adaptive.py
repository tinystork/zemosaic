import numpy as np

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import zemosaic_align_stack as zas
from zemosaic_worker import _prepare_adaptive_master_tile_inputs


class _DummyWCS:
    """Minimal stub that mimics the subset of astropy.wcs.WCS used in tests."""

    is_celestial = True

    def __init__(self, offset_x: float = 0.0, offset_y: float = 0.0, shape: tuple[int, int] = (4, 4)) -> None:
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.pixel_shape = shape
        self.array_shape = shape
        self.wcs = self
        self.crpix = [shape[1] / 2.0, shape[0] / 2.0]

    def deepcopy(self) -> "_DummyWCS":
        clone = _DummyWCS(self.offset_x, self.offset_y, self.pixel_shape)
        clone.crpix = list(self.crpix)
        return clone

    def wcs_pix2world(self, pixels, origin):
        pixels = np.asarray(pixels, dtype=float)
        return np.stack(
            (pixels[..., 0] + self.offset_x, pixels[..., 1] + self.offset_y),
            axis=-1,
        )

    def wcs_world2pix(self, world, origin):
        world = np.asarray(world, dtype=float)
        return np.stack(
            (world[..., 0] - self.offset_x, world[..., 1] - self.offset_y),
            axis=-1,
        )


class _DummyConfig:
    adaptive_master_tile_enable = True
    max_master_tile_megapixels = 1.0
    max_master_tile_scale = 2.0


def test_sky_mean_normalization_subtracts_reference_background():
    ref = np.full((2, 2, 1), 100.0, dtype=np.float32)
    other = np.array(
        [[[150.0], [150.0]], [[200.0], [120.0]]],
        dtype=np.float32,
    )
    images = [ref, other]

    normalized = zas._normalize_images_sky_mean(
        images,
        reference_index=0,
        sky_percentile=50.0,
        progress_callback=None,
        use_gpu=False,
    )

    assert np.allclose(np.nanmedian(normalized[0]), 0.0, atol=1e-5)
    assert np.allclose(np.nanmedian(normalized[1]), 0.0, atol=1e-5)
    # Les valeurs relatives (différences entre pixels) doivent être conservées
    diff_original = other - np.nanmedian(other)
    diff_normalized = normalized[1]
    assert np.allclose(diff_original, diff_normalized, atol=1e-5)


def test_adaptive_canvas_uses_aligned_data_when_possible(monkeypatch):
    base_wcs = _DummyWCS(0.0, 0.0)
    shifted_wcs = _DummyWCS(-1.0, 0.0)

    original_images = [
        np.full((4, 4, 1), 10.0, dtype=np.float32),
        np.full((4, 4, 1), 20.0, dtype=np.float32),
    ]
    aligned_images = [img.copy() for img in original_images]
    raw_infos = [
        {"wcs": base_wcs, "preprocessed_shape": (4, 4, 1)},
        {"wcs": shifted_wcs, "preprocessed_shape": (4, 4, 1)},
    ]

    calls = []

    def _fake_reproject(data_tuple, wcs_out, shape_out=None, return_footprint=False):
        calls.append(shape_out)
        array = np.full(shape_out, 5.0, dtype=np.float32)
        if return_footprint:
            return array, np.ones(shape_out, dtype=np.float32)
        return array

    monkeypatch.setattr("zemosaic_worker.reproject_interp", _fake_reproject)

    result = _prepare_adaptive_master_tile_inputs(
        original_images,
        aligned_images,
        raw_infos,
        base_wcs,
        (4, 4),
        _DummyConfig(),
    )

    assert result is not None
    adaptive_images, adaptive_wcs, kept = result
    assert len(adaptive_images) == 2
    assert kept == [0, 1]

    first = adaptive_images[0]
    assert first.shape == (4, 5, 1)
    # La première image doit être copiée depuis les données alignées et insérée avec le décalage attendu
    np.testing.assert_allclose(first[:, 1:, 0], aligned_images[0][:, :, 0])
    assert np.isnan(first[:, 0, 0]).all()

    second = adaptive_images[1]
    assert second.shape == (4, 5, 1)
    # La deuxième image provient de la reprojection fictive (remplie avec 5.0)
    np.testing.assert_allclose(second, 5.0)
    assert calls, "reproject_interp should have been invoked for shifted frame"
