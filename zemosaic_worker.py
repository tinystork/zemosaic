    if weights is None:
        if data.ndim == 2:
            finite_mask = np.isfinite(data)
            invalid_mask = ~finite_mask
        elif data.ndim == 3:
            finite_mask = np.all(np.isfinite(data), axis=-1)
            invalid_mask = ~finite_mask
        else:
            finite_mask = None
            invalid_mask = None
        if finite_mask is not None:
            weights = finite_mask.astype(np.float32, copy=False)
            if invalid_mask is not None and np.any(invalid_mask):
                if data.ndim == 2:
                    data = np.where(invalid_mask, np.nan, data)
                else:
                    data = np.where(invalid_mask[..., None], np.nan, data)
    data = np.asarray(data, dtype=np.float32, order="C", copy=False)
