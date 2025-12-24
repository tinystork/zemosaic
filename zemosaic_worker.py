            if final_mosaic_data_HWC is None or final_mosaic_coverage_HW is None:
                pcb(
                    log_key_phase5_failed or "run_error_phase5_assembly_failed_unknown",
                    prog=base_progress_phase5 + progress_weight_phase5,
                    lvl="ERROR",
                )
                _emit_phase5_stats(0, tiles_total_phase5, force=True, stage="failed")
                raise RuntimeError("Phase 5 assembly returned no mosaic output")
            pcb(
                log_key_phase5_failed or "run_error_phase5_assembly_failed_unknown",
                prog=base_progress_phase5 + progress_weight_phase5,
                lvl="ERROR",
            )
            _emit_phase5_stats(0, tiles_total_phase5, force=True, stage="failed")
            raise
    if mosaic_data is None or coverage is None:
        raise RuntimeError(
            "assemble_final_mosaic_reproject_coadd produced no mosaic or coverage output",
        )
    if not isinstance(mosaic_data, np.ndarray) or not isinstance(coverage, np.ndarray):
        raise RuntimeError("assemble_final_mosaic_reproject_coadd returned invalid array types")
    if mosaic_data.shape[:2] != coverage.shape[:2]:
        raise RuntimeError(
            f"assemble_final_mosaic_reproject_coadd output shape mismatch: mosaic={mosaic_data.shape}, coverage={coverage.shape}",
        )

