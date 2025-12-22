    def _best_effort_anchor_photometry() -> None:
        if not existing_master_tiles_mode:
            return
        if len(effective_tiles) < 2:
            return
        if not (REPROJECT_AVAILABLE and reproject_interp and ASTROPY_AVAILABLE and fits):
            logger.info("existing_master_tiles_mode: best-effort anchor skipped (reproject/Astropy unavailable)")
            return
        try:
            preview_h, preview_w = map(int, final_output_shape_hw)
        except Exception:
            preview_h = preview_w = 0
        if preview_h <= 0 or preview_w <= 0:
            logger.info("existing_master_tiles_mode: best-effort anchor skipped (invalid final shape)")
            return

        max_preview = 1024
        scale = min(1.0, max_preview / max(preview_h, preview_w)) if max(preview_h, preview_w) > 0 else 1.0
        preview_shape = (max(32, int(preview_h * scale)), max(32, int(preview_w * scale)))
        min_overlap_pixels = 256
        gain_clip = (0.5, 2.0)

        def _resolve_mask(entry: dict[str, Any]) -> np.ndarray | None:
            mask = None
            if isinstance(entry, dict):
                mask = entry.get("alpha_weight2d")
                if mask is None:
                    mask = entry.get("coverage_mask")
            if mask is None:
                return None
            try:
                mask_arr = np.asarray(mask, dtype=np.float32, order="C", copy=False)
            except Exception:
                return None
            if mask_arr.ndim > 2:
                mask_arr = np.squeeze(mask_arr)
            return mask_arr

        def _project_tile(entry: dict[str, Any]) -> tuple[np.ndarray, np.ndarray | None] | None:
            tile_wcs = entry.get("wcs")
            data = entry.get("data")
            if tile_wcs is None or data is None:
                return None
            data_arr = np.asarray(data, dtype=np.float32, order="C", copy=False)
            if data_arr.ndim != 3:
                return None
            mask_arr = _resolve_mask(entry)
            projected_channels = []
            mask_proj = None
            for c in range(data_arr.shape[-1]):
                try:
                    reproj, footprint = reproject_interp(
                        (data_arr[..., c], tile_wcs),
                        final_output_wcs,
                        shape_out=preview_shape,
                        return_footprint=True,
                    )
                except Exception:
                    return None
                projected_channels.append(np.asarray(reproj, dtype=np.float32))
                if mask_proj is None:
                    if mask_arr is not None:
                        try:
                            mask_proj_raw, mask_fp = reproject_interp(
                                (mask_arr.astype(np.float32, copy=False), tile_wcs),
                                final_output_wcs,
                                shape_out=preview_shape,
                                return_footprint=True,
                            )
                            mask_proj = np.asarray(mask_proj_raw, dtype=np.float32)
                            if mask_fp is not None:
                                mask_proj = mask_proj * (np.asarray(mask_fp, dtype=np.float32) > 0)
                        except Exception:
                            mask_proj = None
                    elif footprint is not None:
                        try:
                            mask_proj = (np.asarray(footprint, dtype=np.float32) > 0).astype(np.float32)
                        except Exception:
                            mask_proj = None
            try:
                projected = np.stack(projected_channels, axis=-1)
            except Exception:
                return None
            return projected, mask_proj

        def _anchor_candidate_area(entry: dict[str, Any]) -> int:
            mask_arr = _resolve_mask(entry)
            if mask_arr is None:
                data_arr = np.asarray(entry.get("data"), dtype=np.float32, order="C") if isinstance(entry, dict) else None
                if data_arr is None:
                    return 0
                if data_arr.ndim == 3:
                    valid_mask = np.all(np.isfinite(data_arr), axis=-1)
                else:
                    valid_mask = np.isfinite(data_arr)
            else:
                valid_mask = mask_arr > 0
            try:
                return int(np.count_nonzero(valid_mask))
            except Exception:
                return 0

        anchor_idx = None
        largest_area = -1
        for idx_entry, entry in enumerate(effective_tiles):
            area = _anchor_candidate_area(entry)
            if area > largest_area:
                largest_area = area
                anchor_idx = idx_entry
        if anchor_idx is None:
            logger.info("existing_master_tiles_mode: best-effort anchor skipped (no valid tiles)")
            return

        anchor_entry = effective_tiles[anchor_idx]
        anchor_proj = _project_tile(anchor_entry)
        if anchor_proj is None:
            logger.info("existing_master_tiles_mode: best-effort anchor skipped (anchor reprojection failed)")
            return
        anchor_data, anchor_mask = anchor_proj
        for idx_entry, entry in enumerate(effective_tiles):
            if idx_entry == anchor_idx:
                continue
            tile_proj = _project_tile(entry)
            if tile_proj is None:
                logger.info(
                    "existing_master_tiles_mode: skip photometric anchor (tile reprojection failed)",
                )
                continue
            tile_data, tile_mask = tile_proj
            base_mask = np.isfinite(anchor_data).all(axis=-1) & np.isfinite(tile_data).all(axis=-1)
            if anchor_mask is not None:
                base_mask &= anchor_mask > 0
            if tile_mask is not None:
                base_mask &= tile_mask > 0
            try:
                anchor_nonzero = np.any(anchor_data != 0.0, axis=-1)
                tile_nonzero = np.any(tile_data != 0.0, axis=-1)
                base_mask &= anchor_nonzero & tile_nonzero
            except Exception:
                pass

            overlap_count = int(np.count_nonzero(base_mask))
            if overlap_count < min_overlap_pixels:
                logger.info(
                    "existing_master_tiles_mode: skip photometric anchor (overlap too small: %d)",
                    overlap_count,
                )
                continue
            anchor_vals = anchor_data[base_mask]
            tile_vals = tile_data[base_mask]
            try:
                anchor_med = np.nanmedian(anchor_vals, axis=0)
                tile_med = np.nanmedian(tile_vals, axis=0)
            except Exception:
                continue
            if anchor_med.shape[0] == 0 or tile_med.shape[0] == 0:
                continue
            if not (np.all(np.isfinite(anchor_med)) and np.all(np.isfinite(tile_med))):
                logger.info("existing_master_tiles_mode: skip photometric anchor (non-finite medians)")
                continue
            if np.any(np.abs(tile_med) < 1e-6):
                logger.info("existing_master_tiles_mode: skip photometric anchor (tile median ~0)")
                continue
            gains = anchor_med / tile_med
            gains = np.clip(gains, gain_clip[0], gain_clip[1])
            try:
                data_arr = np.asarray(entry["data"], dtype=np.float32, order="C", copy=False)
            except Exception:
                continue
            applied_channels = 0
            for c in range(min(data_arr.shape[-1], gains.shape[0])):
                gain_val = float(gains[c])
                if not math.isfinite(gain_val):
                    continue
                np.multiply(data_arr[..., c], gain_val, out=data_arr[..., c], casting="unsafe")
                applied_channels += 1
            if applied_channels > 0:
                try:
                    pcb_msg = "[ExistingMT] applied anchor gain"
                    logger.info(
                        "existing_master_tiles_mode: anchor gain applied to tile %d (overlap=%d, gains=%s)",
                        idx_entry,
                        overlap_count,
                        ", ".join(f"{g:.3f}" for g in gains.tolist()),
                    )
                    _pcb(
                        pcb_msg,
                        prog=None,
                        lvl="INFO_DETAIL",
                    )
                except Exception:
                    pass

    _best_effort_anchor_photometry()

