    force_safe_mode: bool | None = None,
    requested_workers = cpu_workers if cpu_workers is not None else "auto"

    if force_safe_mode:
        force_single_reason = "config_safe_mode"

    probe_reason = None
    if not force_single_worker and cpu_workers and cpu_workers > 1:
        try:
            ctx = multiprocessing.get_context("spawn" if sys.platform == "win32" else None)
            with ctx.Pool(processes=min(cpu_workers, max(1, cpu_workers))) as pool:
                pool.map(int, range(min(cpu_workers, 2)))
        except Exception as exc_probe:
            force_single_worker = True
            probe_reason = f"probe_failed_{exc_probe.__class__.__name__}"
            force_single_reason = probe_reason
            if logger_obj:
                try:
                    logger_obj.warning(
                        "[Intertile] Multiprocessing probe failed (requested=%s): %s — falling back to single worker",
                        requested_workers,
                        exc_probe,
                    )
                except Exception:
                    pass

                    "[Intertile] Safe-mode: forcing single worker (requested=%s, reason=%s)",
                    force_single_reason or probe_reason or "unspecified",
        if cpu_workers and cpu_workers > 1 and not force_safe_mode:
            if logger_obj:
                try:
                    logger_obj.warning(
                        "[Intertile] Calibration failed with multiple workers (%s): %s — retrying with single worker",
                        cpu_workers,
                        exc,
                    )
                    logger_obj.debug("Traceback (intertile failure multi-worker):", exc_info=True)
                except Exception:
                    pass
            cpu_workers = 1
            try:
                corrections = zemosaic_utils.compute_intertile_affine_calibration(

                    tile_pairs,
                    final_output_wcs,
                    final_output_shape_hw,
                    preview_size=preview_size,
                    min_overlap_fraction=min_overlap_fraction,
                    sky_percentile=sky_percentile,
                    robust_clip_sigma=robust_clip_sigma,
                    use_auto_intertile=use_auto_intertile,
                    logger=logger_obj,
                    progress_callback=_intertile_progress_bridge,
                    tile_weights=tile_weights,
                    cpu_workers=cpu_workers,
                )
            except Exception as exc_single:
                if logger_obj:
                    logger_obj.warning(
                        "Intertile photometric calibration failed after fallback to single worker: %s",
                        exc_single,
                    )
                    logger_obj.debug("Traceback (intertile failure single-worker):", exc_info=True)
                return None, False, "compute_failed", str(exc_single)
        else:
            if logger_obj:
                logger_obj.warning(
                    "Intertile photometric calibration failed: %s",
                    exc,
                )
                logger_obj.debug("Traceback (intertile failure):", exc_info=True)
            return None, False, "compute_failed", str(exc)
    intertile_force_safe_mode_config = bool(phase5_options.get("intertile_force_safe_mode"))
                intertile_force_safe_mode=intertile_force_safe_mode_config,
                intertile_force_safe_mode=intertile_force_safe_mode_config,
    intertile_force_safe_mode: bool | None = None,
                force_safe_mode=intertile_force_safe_mode,
    intertile_force_safe_mode: bool | None = None,
                force_safe_mode=intertile_force_safe_mode,
    intertile_force_safe_mode_config: bool = False,
            "intertile_force_safe_mode": intertile_force_safe_mode_config,
    intertile_force_safe_mode_config: bool = False,
            "intertile_force_safe_mode": intertile_force_safe_mode_config,
