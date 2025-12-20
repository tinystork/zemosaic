    try:
        az_nanize_threshold = float(cfg.get("altaz_nanize_threshold", hard_threshold))
    except Exception:
        az_nanize_threshold = hard_threshold
    if not math.isfinite(az_nanize_threshold):
        az_nanize_threshold = 1e-3
    else:
        az_nanize_threshold = float(np.clip(az_nanize_threshold, 0.0, 1.0))
                try:
                    logger.info("lecropper: altaz_nanize_threshold=%.3f", az_nanize_threshold)
                except Exception:
                    pass
                mask_zero = alpha_mask_norm <= az_nanize_threshold
                    az_nanize_threshold,
