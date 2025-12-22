def _auto_orient_alpha_mask(
    alpha_mask_arr: np.ndarray,
    data_arr: np.ndarray,
    *,
    logger_obj=None,
    context_label: str = "existing_master_tiles",
    flip_margin: float = 0.05,
) -> tuple[np.ndarray, bool]:
    """Detect inverted alpha masks and auto-flip when clearly beneficial."""

    flipped = False
    try:
        alpha_arr = np.asarray(alpha_mask_arr, dtype=np.float32, order="C")
        data = np.asarray(data_arr)
        if data.ndim == 3:
            nz2d = np.any((np.abs(data) > 1e-6) & np.isfinite(data), axis=-1)
        else:
            nz2d = (np.abs(data) > 1e-6) & np.isfinite(data)
        nz_frac = float(np.mean(nz2d)) if nz2d.size else 0.0
        valid_frac = float(np.mean(alpha_arr > ALPHA_OPACITY_THRESHOLD)) if alpha_arr.size else 0.0
        inv_valid_frac = (
            float(np.mean((1.0 - alpha_arr) > ALPHA_OPACITY_THRESHOLD)) if alpha_arr.size else 0.0
        )
        score = abs(valid_frac - nz_frac)
        inv_score = abs(inv_valid_frac - nz_frac)
        if inv_score + float(flip_margin) < score:
            alpha_arr = 1.0 - alpha_arr
            flipped = True
            logger_target = logger_obj or logger
            try:
                logger_target.info(
                    "[Alpha] %s: auto-inverted alpha mask valid_frac=%.3f inv_valid_frac=%.3f nz_frac=%.3f",
                    context_label,
                    valid_frac,
                    inv_valid_frac,
                    nz_frac,
                )
            except Exception:
                pass
        return alpha_arr, flipped
    except Exception:
        return alpha_mask_arr, flipped


    tile_pairs: list[tuple[np.ndarray, Any] | tuple[np.ndarray, Any, np.ndarray]] = []
            mask2d_float: np.ndarray | None = None
            alpha_mask_arr: np.ndarray | None = None
                    if "ALPHA" in hdul and hdul["ALPHA"].data is not None:
                        try:
                            alpha_mask_arr = np.asarray(hdul["ALPHA"].data)
                        except Exception:
                            alpha_mask_arr = None
            if alpha_mask_arr is not None:
                alpha_mask_arr = np.squeeze(alpha_mask_arr)
                if alpha_mask_arr.ndim == 3 and alpha_mask_arr.shape[0] == 1:
                    alpha_mask_arr = alpha_mask_arr[0]
                alpha_mask_arr = np.nan_to_num(alpha_mask_arr, nan=0.0, posinf=0.0, neginf=0.0)
                max_alpha_val = float(np.nanmax(alpha_mask_arr)) if alpha_mask_arr.size else 0.0
                if alpha_mask_arr.dtype.kind in {"i", "u"} and max_alpha_val > 1.0:
                    alpha_mask_arr = alpha_mask_arr.astype(np.float32, copy=False) / 255.0
                elif alpha_mask_arr.dtype.kind not in {"f"}:
                    alpha_mask_arr = alpha_mask_arr.astype(np.float32, copy=False)
                alpha_mask_arr = np.clip(alpha_mask_arr, 0.0, 1.0)
                if alpha_mask_arr.shape != tile_arr.shape[:2]:
                    alpha_mask_arr = None
                else:
                    alpha_mask_arr, _ = _auto_orient_alpha_mask(
                        alpha_mask_arr,
                        tile_arr,
                        logger_obj=logger_obj or logger,
                        context_label="existing_master_tiles_intertile",
                    )
                    valid2d = alpha_mask_arr > ALPHA_OPACITY_THRESHOLD
                    mask2d_float = np.asarray(alpha_mask_arr, dtype=np.float32, order="C")
                    if tile_arr.ndim == 3:
                        tile_arr[~valid2d, :] = np.nan
                    else:
                        tile_arr[~valid2d] = np.nan
            tile_pairs.append((tile_arr, src.wcs) if mask2d_float is None else (tile_arr, src.wcs, mask2d_float))
            if existing_master_tiles_mode:
                alpha_mask_arr, _ = _auto_orient_alpha_mask(
                    alpha_mask_arr,
                    data,
                    logger_obj=logger,
                    context_label="existing_master_tiles",
                )
