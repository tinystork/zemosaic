import numpy as np

import grid_mode


def test_grid_final_dbe_applies_on_rgb_with_valid_mask():
    h, w = 64, 64
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    bg = (0.3 * x + 0.2 * y).astype(np.float32)
    mosaic = np.stack([bg + 1.0, bg + 1.1, bg + 0.9], axis=-1).astype(np.float32)
    valid = np.ones((h, w), dtype=bool)

    out, info = grid_mode._apply_grid_final_dbe(mosaic.copy(), valid_mask_hw=valid, strength="normal")

    assert out.shape == mosaic.shape
    assert bool(info.get("applied", False)) is True
    assert int(info.get("applied_channels", 0)) >= 1


def test_grid_final_dbe_skips_non_rgb():
    gray = np.ones((32, 32), dtype=np.float32)
    out, info = grid_mode._apply_grid_final_dbe(gray)
    assert out.shape == gray.shape
    assert bool(info.get("applied", False)) is False
    assert info.get("reason") == "non_rgb"


def test_grid_final_dbe_strength_aliases_from_gui_labels():
    h, w = 48, 48
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    bg = (0.25 * xx + 0.15 * yy).astype(np.float32)
    mosaic = np.stack([bg + 1.0, bg + 1.0, bg + 1.0], axis=-1)
    valid = np.ones((h, w), dtype=bool)

    _, info_weak = grid_mode._apply_grid_final_dbe(mosaic.copy(), valid_mask_hw=valid, strength="weak")
    _, info_strong = grid_mode._apply_grid_final_dbe(mosaic.copy(), valid_mask_hw=valid, strength="strong")

    assert float(info_weak.get("sigma", 0.0)) == 24.0
    assert float(info_strong.get("sigma", 0.0)) == 52.0


def test_grid_final_dbe_protects_bright_star_core_from_dark_haloing():
    h, w = 96, 96
    y, x = np.mgrid[0:h, 0:w]
    cx, cy = w // 2, h // 2
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    bg = (0.002 * x + 0.0015 * y).astype(np.float32)
    star = (3.5 * np.exp(-r2 / (2.0 * 2.2**2))).astype(np.float32)
    base = (bg + star + 0.8).astype(np.float32)
    mosaic = np.stack([base, base * 1.02, base * 0.98], axis=-1)
    valid = np.ones((h, w), dtype=bool)

    out, info = grid_mode._apply_grid_final_dbe(mosaic.copy(), valid_mask_hw=valid, strength="normal")

    assert bool(info.get("applied", False)) is True
    # Star core should stay essentially unchanged (protected object mask)
    for c in range(3):
        before = float(mosaic[cy, cx, c])
        after = float(out[cy, cx, c])
        assert abs(after - before) < 5e-3
