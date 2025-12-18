    alpha_zero_mask = None
    if isinstance(coverage, np.ndarray) and alpha_final is not None:
        try:
            alpha_arr = np.asarray(alpha_final)
            if alpha_arr.ndim == 3 and alpha_arr.shape[-1] == 1:
                alpha_arr = alpha_arr[..., 0]
            alpha_arr = np.squeeze(alpha_arr)
            if alpha_arr.ndim >= 2 and alpha_arr.shape[:2] == coverage.shape:
                alpha_zero_mask = alpha_arr <= 0
                if np.any(alpha_zero_mask):
                    coverage = np.where(alpha_zero_mask, 0.0, coverage).astype(np.float32, copy=False)
        except Exception:
            alpha_zero_mask = None
        if alpha_zero_mask is not None:
            nanized_mask = np.logical_or(nanized_mask, alpha_zero_mask)
        elif alpha_final is not None and alpha_final.shape[:2] == coverage.shape:
