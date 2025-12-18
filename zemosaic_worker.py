def _nanize_by_coverage(
    final_hwc: np.ndarray | None,
    coverage_hw: np.ndarray | None,
    *,
    alpha_u8: np.ndarray | None = None,
) -> np.ndarray | None:
    """Replace pixels with ``NaN`` wherever coverage (or alpha) indicates no data."""

    if final_hwc is None or coverage_hw is None:
        return final_hwc
    try:
        mosaic = np.asarray(final_hwc, dtype=np.float32, copy=False)
        coverage = np.asarray(coverage_hw, dtype=np.float32, copy=False)
    except Exception:
        return final_hwc
    if mosaic.ndim < 2 or coverage.shape[:2] != mosaic.shape[:2]:
        return final_hwc
    invalid = coverage <= 0
    if alpha_u8 is not None:
        try:
            alpha_arr = np.asarray(alpha_u8, copy=False)
            if alpha_arr.ndim == 3 and alpha_arr.shape[-1] == 1:
                alpha_arr = alpha_arr[..., 0]
            alpha_arr = np.squeeze(alpha_arr)
            if alpha_arr.ndim >= 2 and alpha_arr.shape[:2] == mosaic.shape[:2]:
                invalid = np.logical_or(invalid, alpha_arr == 0)
        except Exception:
            pass
    try:
        invalid_mask = invalid if mosaic.ndim == 2 else invalid[..., None]
        mosaic = np.where(invalid_mask, np.nan, mosaic)
    except Exception:
        return mosaic
    return np.asarray(mosaic, dtype=np.float32, copy=False)


    if coverage_arr is not None:
        coverage_arr = np.where(np.isfinite(coverage_arr), coverage_arr, 0.0).astype(np.float32, copy=False)
    if mosaic_arr is not None and coverage_arr is not None:
        mosaic_arr = np.asarray(mosaic_arr, dtype=np.float32, copy=False)
        mosaic_arr[~np.isfinite(mosaic_arr)] = np.nan
        nanized_mask = coverage_arr <= 0
        if final_alpha_u8 is not None and final_alpha_u8.shape[:2] == coverage_arr.shape:
            nanized_mask = np.logical_or(nanized_mask, final_alpha_u8 == 0)
        nanized_pixels = int(np.count_nonzero(nanized_mask))
        mosaic_arr = _nanize_by_coverage(mosaic_arr, coverage_arr, alpha_u8=final_alpha_u8)
        if nanized_pixels > 0:
            logger.info("sds_global: nanized %d pixels via coverage/alpha", nanized_pixels)
    if isinstance(coverage, np.ndarray):
        coverage = np.where(np.isfinite(coverage), coverage, 0.0).astype(np.float32, copy=False)
    if mosaic_data is not None and coverage is not None:
        mosaic_data = np.asarray(mosaic_data, dtype=np.float32, copy=False)
        mosaic_data[~np.isfinite(mosaic_data)] = np.nan
        nanized_mask = coverage <= 0
        if alpha_final is not None and alpha_final.shape[:2] == coverage.shape:
            nanized_mask = np.logical_or(nanized_mask, alpha_final == 0)
        nanized_pixels = int(np.count_nonzero(nanized_mask))
        mosaic_data = _nanize_by_coverage(mosaic_data, coverage, alpha_u8=alpha_final)
        if nanized_pixels > 0:
            logger.info(
                "assemble_reproject_coadd: nanized %d pixels where coverage/alpha == 0",
                nanized_pixels,
            )

                                preview_view = np.where(mask_zero[..., None], np.nan, preview_view)
                                preview_view = np.where(mask_zero[..., None], np.nan, preview_view)
        final_image = np.asarray(final_image, dtype=np.float32, copy=False)
        final_image[~np.isfinite(final_image)] = np.nan
        coverage_map = np.asarray(coverage_map, dtype=np.float32, copy=False)
        coverage_map = np.where(np.isfinite(coverage_map), coverage_map, 0.0)
        nanized_mask = coverage_map <= 0
        if alpha_map is not None and alpha_map.shape[:2] == coverage_map.shape:
            nanized_mask = np.logical_or(nanized_mask, alpha_map == 0)
        nanized_pixels = int(np.count_nonzero(nanized_mask))
        final_image = _nanize_by_coverage(final_image, coverage_map, alpha_u8=alpha_map)
        if nanized_pixels > 0:
            logger.info("global_coadd: nanized %d pixels via coverage/alpha (gpu helper)", nanized_pixels)
            result = np.asarray(result, dtype=np.float32, copy=False)
            result[~np.isfinite(result)] = np.nan
            result = np.where(weight_grid[..., None] > 0, result, np.nan)
            coverage = np.where(np.isfinite(weight_grid), weight_grid, 0.0).astype(np.float32, copy=False)
            mean_map = np.asarray(mean_map, dtype=np.float32, copy=False)
            second_moment = np.asarray(second_moment, dtype=np.float32, copy=False)
            mean_map[~np.isfinite(mean_map)] = np.nan
            second_moment[~np.isfinite(second_moment)] = np.nan
            std_map = np.where(np.isfinite(std_map), std_map, 0.0)
            clipped = np.asarray(clipped, dtype=np.float32, copy=False)
            clipped[~np.isfinite(clipped)] = np.nan
            coverage_base = np.where(clip_weight > 0, clip_weight, weight_grid)
            coverage = np.where(np.isfinite(coverage_base), coverage_base, 0.0).astype(np.float32, copy=False)
            return clipped, coverage
                chunk_result = np.asarray(chunk_result, dtype=np.float32, copy=False)
                chunk_result[~np.isfinite(chunk_result)] = np.nan
                chunk_weight = np.where(np.isfinite(chunk_weight), chunk_weight, 0.0).astype(np.float32, copy=False)
                invalid = chunk_weight <= 0
                if np.any(invalid):
                    chunk_result = np.where(invalid[..., None], np.nan, chunk_result)
        final_image = np.asarray(final_image, dtype=np.float32, copy=False)
        final_image[~np.isfinite(final_image)] = np.nan
        coverage_map = np.asarray(coverage_map, dtype=np.float32, copy=False)
        coverage_map = np.where(np.isfinite(coverage_map), coverage_map, 0.0)
        nanized_mask = coverage_map <= 0
        if alpha_map is not None and alpha_map.shape[:2] == coverage_map.shape:
            nanized_mask = np.logical_or(nanized_mask, alpha_map == 0)
        nanized_pixels = int(np.count_nonzero(nanized_mask))
        final_image = _nanize_by_coverage(final_image, coverage_map, alpha_u8=alpha_map)
        if nanized_pixels > 0:
            logger.info("global_coadd: nanized %d pixels via coverage/alpha (cpu)", nanized_pixels)
