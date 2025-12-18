    alpha_info_logged = False
        alpha_weight2d = None
                valid_mask = alpha_mask_arr > ALPHA_OPACITY_THRESHOLD
                alpha_weight2d = valid_mask.astype(np.float32, copy=False)
                data = np.asarray(data, dtype=np.float32, order="C", copy=True)
                if data.ndim == 3:
                    data[~valid_mask, :] = 0.0
                else:
                    data[~valid_mask] = 0.0
                if not alpha_info_logged:
                    alpha_min = float(np.nanmin(alpha_mask_arr)) if alpha_mask_arr.size else 0.0
                    alpha_max = float(np.nanmax(alpha_mask_arr)) if alpha_mask_arr.size else 0.0
                    valid_fraction = float(np.mean(valid_mask)) if valid_mask.size else 0.0
                    _pcb(
                        "[Alpha] mask stats (first tile)",
                        prog=None,
                        lvl="INFO_DETAIL",
                        alpha_min=f"{alpha_min:.4f}",
                        alpha_max=f"{alpha_max:.4f}",
                        valid_fraction=f"{valid_fraction:.4f}",
                        weight_shape=str(alpha_weight2d.shape),
                        data_shape=str(data.shape),
                    )
                    alpha_info_logged = True
        if alpha_weight2d is not None:
            coverage_mask_entry = np.asarray(alpha_weight2d, dtype=np.float32, order="C")
            "alpha_weight2d": alpha_weight2d,
    alpha_present = any(
        isinstance(entry, dict) and entry.get("alpha_weight2d") is not None for entry in effective_tiles
    )
    if use_gpu and alpha_present:
        use_gpu = False
        try:
            _pcb("[Alpha] Per-pixel alpha weights require CPU coadd; forcing CPU for Phase 5", lvl="INFO_DETAIL")
        except Exception:
            pass
        logger.info("[Alpha] Per-pixel alpha weights detected; forcing CPU coadd for Phase 5")

            weights_list = []
            for entry, data_plane in zip(valid_entries, data_list):
                weight_map = None
                if isinstance(entry, dict):
                    weight_map = entry.get("alpha_weight2d")
                if weight_map is not None:
                    weights_list.append(np.asarray(weight_map, dtype=np.float32, order="C"))
                else:
                    weights_list.append(np.ones_like(data_plane, dtype=np.float32))
                invoke_kwargs["input_weights"] = weights_list
