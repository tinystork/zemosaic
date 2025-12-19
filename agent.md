Mission (surgical / no refactor): use tiles_coverage as weights in two-pass reprojection

Context:
Project: ZeMosaic
File: zemosaic_worker.py
Function to modify:

def run_second_pass_coverage_renorm(
    tiles: list[np.ndarray],
    tiles_wcs: list[Any],
    final_wcs_p1: Any,
    coverage_p1: np.ndarray,
    shape_out: tuple[int, int],
    *,
    sigma_px: int,
    gain_clip: tuple[float, float],
    logger=None,
    use_gpu_two_pass: bool | None = None,
    tiles_coverage: list[np.ndarray | None] | None = None,
    parallel_plan: ParallelPlan | None = None,
    telemetry_ctrl: ResourceTelemetryController | None = None,
    debug_diag: _TwoPassDiagnostics | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:


We already use tiles_coverage when computing gains via compute_per_tile_gains_from_coverage, but the second-pass reprojection itself does not use these coverage maps as weights. This makes the second-pass mosaic ignore per-tile masks/coverage (AltAz masks, edge trim, etc.), which leads to dark rectangular frames when two-pass is enabled, even though the first-pass mosaic is correct.

Goal:
Patch run_second_pass_coverage_renorm so that, inside its per-channel reprojection, it passes a proper input_weights list to zemosaic_utils.reproject_and_coadd_wrapper, built from tiles_coverage. Also, add a small DEBUG log comparing coverage_p1 and second-pass chan_cov to help diagnose coverage mismatches.

Scope:

Only modify run_second_pass_coverage_renorm in zemosaic_worker.py.

Do not change its signature.

Do not touch any other function or file.

Do not refactor; only add the minimal logic needed.

Keep existing logging style and diagnostics (_TwoPassDiagnostics).

1) Build per-tile 2D weight maps from tiles_coverage

Inside run_second_pass_coverage_renorm, after we have:

corrected_tiles: list[np.ndarray] = [...]
...
shape_out_hw = tuple(map(int, shape_out))


and before _process_channel is defined, add code to prepare a list of 2D weights aligned with corrected_tiles:

If tiles_coverage is None or empty, leave weights as None and do not change behavior.

Otherwise:

weights_2d_list: list[np.ndarray | None] | None = None
if tiles_coverage is not None:
    try:
        weights_tmp: list[np.ndarray | None] = []
        for cov_arr, tile_arr in zip(tiles_coverage, corrected_tiles):
            if cov_arr is None:
                # If no coverage map, fallback to ones over the tile slice
                weights_tmp.append(None)
                continue
            cov_np = np.asarray(cov_arr, dtype=np.float32)
            # Reduce any channel dimension: coverage is per-pixel, same for all channels
            if cov_np.ndim == 3:
                cov2d = cov_np[..., 0]
            else:
                cov2d = cov_np
            # Sanity: match tile spatial shape, ignore if incompatible
            tile_h, tile_w = tile_arr.shape[0], tile_arr.shape[1]
            if cov2d.shape != (tile_h, tile_w):
                if logger:
                    logger.debug(
                        "[TwoPass] tiles_coverage shape mismatch for tile: cov=%s tile=%s → ignoring coverage for this tile",
                        cov2d.shape,
                        (tile_h, tile_w),
                    )
                weights_tmp.append(None)
                continue
            # Convert to binary [0,1] weights: >0 means "valid"
            weight2d = (cov2d > 0.0).astype(np.float32)
            weights_tmp.append(weight2d)
        # Only use weights if at least one tile has a non-None weight map
        if any(w is not None for w in weights_tmp):
            weights_2d_list = weights_tmp
    except Exception:
        weights_2d_list = None
        if logger and logger.isEnabledFor(logging.DEBUG):
            logger.debug("[TwoPass] failed to prepare tiles_coverage weights → skip", exc_info=True)


The idea: for each tile, derive a 2D weight map from tiles_coverage (coverage > 0 → weight 1, else 0), with the same spatial shape as corrected_tiles[i]. If a given tile has no coverage or inconsistent shape, we skip it (weight None → behaves as “no extra weighting” for that tile).

2) Use weights_2d_list as input_weights in _process_channel

Currently _process_channel looks like this (simplified):

def _process_channel(ch_idx: int, use_gpu_flag: bool) -> tuple[int, np.ndarray, np.ndarray]:
    if logger:
        logger.debug(
            "[TwoPass] Reproject channel %d/%d with %d tiles (shape_out=%s, gpu=%s)",
            ch_idx + 1,
            n_channels,
            len(corrected_tiles),
            shape_out_hw,
            use_gpu_flag,
        )
    data_list = [tile[..., ch_idx] if tile.ndim == 3 else tile[..., 0] for tile in corrected_tiles]

    def _invoke_reproj(use_gpu_local: bool, local_kwargs: dict[str, Any]):
        return zemosaic_utils.reproject_and_coadd_wrapper(
            data_list=data_list,
            wcs_list=tiles_wcs,
            shape_out=shape_out_hw,
            use_gpu=use_gpu_local,
            cpu_func=reproject_and_coadd,
            **local_kwargs,
        )

    local_kwargs = dict(reproj_kwargs)
    try:
        chan_mosaic, chan_cov = _invoke_reproj(use_gpu_flag, local_kwargs)
    ...


Modify _process_channel as follows:

After data_list = [...], build a per-channel input_weights_list if weights_2d_list is available:

    input_weights_list = None
    if weights_2d_list is not None:
        try:
            iw: list[np.ndarray | None] = []
            for base_w, tile_arr in zip(weights_2d_list, corrected_tiles):
                if base_w is None:
                    iw.append(None)
                    continue
                # base_w is already 2D and aligned with tile spatial shape
                # Just sanity-check against the slice for this channel
                tile_slice = tile_arr[..., ch_idx] if tile_arr.ndim == 3 else tile_arr[..., 0]
                if base_w.shape != tile_slice.shape:
                    if logger and logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "[TwoPass] input_weights shape mismatch for channel %d: w=%s slice=%s → ignoring weight",
                            ch_idx + 1,
                            base_w.shape,
                            tile_slice.shape,
                        )
                    iw.append(None)
                else:
                    iw.append(base_w)
            if any(w is not None for w in iw):
                input_weights_list = iw
        except Exception:
            input_weights_list = None
            if logger and logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[TwoPass] failed to prepare input_weights for channel %d → skip weights",
                    ch_idx + 1,
                    exc_info=True,
                )


Before calling _invoke_reproj, inject input_weights_list into local_kwargs if it’s not None:

    local_kwargs = dict(reproj_kwargs)
    if input_weights_list is not None:
        # Use the same key as in Phase 5 (assemble_final_mosaic_reproject_coadd)
        local_kwargs["input_weights"] = input_weights_list


Leave the rest of _process_channel unchanged (exception handling, NoConvergence fallback, etc.).

This ensures that both CPU and GPU paths in reproject_and_coadd_wrapper see a proper input_weights list built from tiles_coverage, just like Phase 5 uses input_weights_list built from alpha/coverage.

3) Add a DEBUG log comparing first-pass coverage and second-pass coverage

Still in _process_channel, after we get chan_mosaic, chan_cov, add a small debug-only coverage comparison:

    try:
        chan_mosaic, chan_cov = _invoke_reproj(use_gpu_flag, local_kwargs)
        if logger and logger.isEnabledFor(logging.DEBUG):
            cov1 = np.asarray(coverage_p1, dtype=np.float32)
            cov2 = np.asarray(chan_cov, dtype=np.float32)
            if cov1.shape == cov2.shape:
                bad_mask = (cov1 > 0.0) & (cov2 <= 0.0)
                n_bad = int(np.count_nonzero(bad_mask))
                logger.debug(
                    "[TwoPass] channel %d coverage mismatch: cov1>0 & cov2==0 → %d pixels",
                    ch_idx + 1,
                    n_bad,
                )
            else:
                logger.debug(
                    "[TwoPass] channel %d coverage shape mismatch for debug: cov1=%s cov2=%s",
                    ch_idx + 1,
                    cov1.shape,
                    cov2.shape,
                )
    except wcs_module.NoConvergence as conv_exc:
        ...


Do not change how exceptions are handled or how chan_mosaic / chan_cov are stored in mosaic_channels / coverage_channels.

Success criteria

No signature changes; no refactor outside run_second_pass_coverage_renorm.

With two_pass_coverage_renorm enabled, reprojected second-pass mosaics must now use tiles_coverage as input_weights, so they respect the same per-tile masks/coverage as the first pass.

Dark rectangular frames caused by two-pass should be significantly reduced or disappear on datasets where first-pass mosaic is already clean.

In DEBUG logs, for a healthy run, coverage_p1>0 & chan_cov==0 counts should be low or zero, and coverage shapes should match.