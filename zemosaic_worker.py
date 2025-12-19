    try:
        az_alpha_soft = float(cfg.get("altaz_alpha_soft_threshold", 1e-3))
    except Exception:
        az_alpha_soft = 1e-3
    if not math.isfinite(az_alpha_soft):
        az_alpha_soft = 1e-3
    else:
        az_alpha_soft = float(np.clip(az_alpha_soft, 0.0, 1.0))
    hard_threshold = az_alpha_soft
                    "MT_PIPELINE: altaz_cleanup applied: masked_used=%s mask2d_used=%s threshold=%g",
                    hard_threshold,
