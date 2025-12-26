    caps_for_phase5: ParallelCapabilities | None = None
    def _refresh_phase5_vram_bytes() -> int | None:
        refreshed_vram: int | None = None
        try:
            refreshed_ctx = probe_gpu_runtime_context(caps=caps_for_phase5)
            refreshed_vram = getattr(refreshed_ctx, "vram_free_bytes", None)
            if refreshed_vram is not None and gpu_safety_ctx_phase5 is not None:
                try:
                    gpu_safety_ctx_phase5.vram_free_bytes = refreshed_vram
                except Exception:
                    pass
        except Exception:
            try:
                refreshed_vram = getattr(gpu_safety_ctx_phase5, "vram_free_bytes", None)
            except Exception:
                refreshed_vram = None
        return refreshed_vram

            min_chunk_bytes = 32 * 1024 * 1024
            current_chunk_bytes = getattr(parallel_plan_phase5, "gpu_max_chunk_bytes", None)
            attempt_index = 0
            while True:
                attempt_index += 1
                refreshed_vram_bytes = _refresh_phase5_vram_bytes()
                vram_free_mb_str = (
                    f"{refreshed_vram_bytes / (1024.0 ** 2):.1f}" if refreshed_vram_bytes is not None else "n/a"
                )
                chunk_mb = float(current_chunk_bytes or 0) / (1024.0 ** 2)
                if attempt_index > 1:
                    logger.info(
                        "Phase5 assembly retry #%d with chunk=%.1fMB (vram_free_mb=%s)",
                        attempt_index - 1,
                        chunk_mb,
                        vram_free_mb_str,
                    )
                try:
                    if effective_use_gpu:
                        import cupy

                        cupy.cuda.Device(0).use()
                    final_mosaic_data_HWC, final_mosaic_coverage_HW, final_alpha_map = assemble_final_mosaic_reproject_coadd(
                        master_tile_fits_with_wcs_list=valid_master_tiles_for_assembly,
                        final_output_wcs=final_output_wcs,
                        final_output_shape_hw=final_output_shape_hw,
                        progress_callback=progress_callback,
                        n_channels=3,
                        match_bg=True,
                        apply_crop=apply_crop_for_assembly,
                        crop_percent=master_tile_crop_percent_config,
                        use_gpu=effective_use_gpu,
                        use_memmap=bool(coadd_use_memmap_config),
                        memmap_dir=(coadd_memmap_dir_config or output_folder),
                        cleanup_memmap=False,
                        base_progress_phase5=base_progress_phase5,
                        progress_weight_phase5=progress_weight_phase5,
                        start_time_total_run=start_time_total,
                        intertile_photometric_match=intertile_match_flag,
                        intertile_preview_size=int(intertile_preview_size_config),
                        intertile_overlap_min=float(intertile_overlap_min_config),
                        intertile_sky_percentile=intertile_sky_percentile_tuple,
                        intertile_robust_clip_sigma=float(intertile_robust_clip_sigma_config),
                        intertile_global_recenter=bool(intertile_global_recenter_config),
                        intertile_recenter_clip=intertile_recenter_clip_tuple,
                        use_auto_intertile=bool(use_auto_intertile_config),
                        collect_tile_data=collected_tiles_for_second_pass,
                        global_anchor_shift=global_anchor_shift,
                        phase45_enabled=phase45_active_flag,
                        parallel_plan=parallel_plan_phase5,
                        enable_tile_weighting=tile_weighting_enabled_flag,
                        tile_weight_mode=tile_weight_mode,
                        stats_callback=_emit_phase5_stats,
                        existing_master_tiles_mode=existing_master_tiles_mode,
                    )
                    break
                except (MemoryError, BrokenProcessPool) as exc_retry:
                    logger.warning(
                        "Phase5 assembly %s at chunk=%.1fMB (vram_free_mb=%s)",
                        exc_retry.__class__.__name__,
                        chunk_mb,
                        vram_free_mb_str,
                    )
                    if not current_chunk_bytes or current_chunk_bytes <= min_chunk_bytes:
                        logger.error(
                            "Phase5 assembly failed at minimal chunk=%.1fMB (vram_free_mb=%s)",
                            chunk_mb,
                            vram_free_mb_str,
                        )
                        _emit_phase5_stats(0, tiles_total_phase5, force=True, stage="chunk_retry_failed")
                        raise RuntimeError(
                            f"Phase5 assembly failed even at minimal chunk {chunk_mb:.1f}MB (vram_free_mb={vram_free_mb_str})"
                        ) from exc_retry

                    current_chunk_bytes = max(min_chunk_bytes, int(current_chunk_bytes // 2))
                    try:
                        setattr(parallel_plan_phase5, "gpu_max_chunk_bytes", int(current_chunk_bytes))
                    except Exception:
                        pass
                    _emit_phase5_stats(0, tiles_total_phase5, force=True, stage="chunk_retry")
                    continue

            _emit_phase5_stats(0, tiles_total_phase5, force=True, stage="chunk_final")
