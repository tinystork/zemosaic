                existing_master_tiles_mode=bool(use_existing_master_tiles_mode),
    existing_master_tiles_mode: bool = False,
    if isinstance(coverage, np.ndarray) and existing_master_tiles_mode:
        try:
            cov_mask_bool = np.asarray(coverage, dtype=np.float32, order="C")
            cov_mask_bool = np.nan_to_num(cov_mask_bool, nan=0.0, posinf=0.0, neginf=0.0)
            cov_mask_bool = cov_mask_bool > 0.0
            alpha_target_shape = cov_mask_bool.shape
            override_message = None
            if alpha_final is None:
                alpha_final = np.ascontiguousarray(cov_mask_bool.astype(np.uint8) * 255)
                override_message = "alpha_final was None -> rebuilt"
            else:
                alpha_arr = np.asarray(alpha_final)
                if alpha_arr.ndim == 3 and alpha_arr.shape[-1] == 1:
                    alpha_arr = alpha_arr[..., 0]
                alpha_arr = np.squeeze(alpha_arr)
                if alpha_arr.shape != alpha_target_shape:
                    override_message = (
                        f"overriding alpha_final due to shape mismatch (alpha={alpha_arr.shape}, coverage={alpha_target_shape})"
                    )
                    alpha_final = np.ascontiguousarray(cov_mask_bool.astype(np.uint8) * 255)
                else:
                    alpha_zero = alpha_arr == 0
                    mismatch = int(np.count_nonzero(cov_mask_bool & alpha_zero))
                    if mismatch > 0:
                        total_cov = int(np.count_nonzero(cov_mask_bool))
                        pct = (100.0 * mismatch) / max(1, total_cov)
                        override_message = f"overriding alpha_final (mismatch={mismatch} px, {pct:.2f}% of covered)"
                        alpha_final = np.ascontiguousarray(cov_mask_bool.astype(np.uint8) * 255)
            if alpha_final is not None and alpha_final.shape == alpha_target_shape:
                mismatch_after = int(np.count_nonzero(cov_mask_bool & (np.asarray(alpha_final) == 0)))
                if mismatch_after != 0 and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "alpha_from_coverage: mismatch_after=%d (existing_master_tiles_mode)",
                        mismatch_after,
                    )
            if override_message:
                logger.info(
                    "alpha_from_coverage (existing_master_tiles_mode): %s",
                    override_message,
                )
        except Exception as exc_alpha_cov:
            logger.debug("alpha_from_coverage: override skipped due to error: %s", exc_alpha_cov)
