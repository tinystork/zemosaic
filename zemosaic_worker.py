    weights_2d_list: list[np.ndarray | None] | None = None
    if tiles_coverage is not None:
        try:
            weights_tmp: list[np.ndarray | None] = []
            for cov_arr, tile_arr in zip(tiles_coverage, corrected_tiles):
                if cov_arr is None:
                    weights_tmp.append(None)
                    continue
                cov_np = np.asarray(cov_arr, dtype=np.float32)
                if cov_np.ndim == 3:
                    cov2d = cov_np[..., 0]
                else:
                    cov2d = cov_np
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
                weight2d = (cov2d > 0.0).astype(np.float32)
                weights_tmp.append(weight2d)
            if any(w is not None for w in weights_tmp):
                weights_2d_list = weights_tmp
        except Exception:
            weights_2d_list = None
            if logger and logger.isEnabledFor(logging.DEBUG):
                logger.debug("[TwoPass] failed to prepare tiles_coverage weights → skip", exc_info=True)
        input_weights_list = None
        if weights_2d_list is not None:
            try:
                iw: list[np.ndarray | None] = []
                for base_w, tile_arr in zip(weights_2d_list, corrected_tiles):
                    if base_w is None:
                        iw.append(None)
                        continue
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
        if input_weights_list is not None:
            local_kwargs["input_weights"] = input_weights_list
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
