## Unreleased
- Added configurable `winsor_worker_limit` (CLI `--winsor-workers` / `-W` and GUI field)
- Fixed intra-tile alignment failure (`aligngroup_warn_value_error`) by using
  grayscale detection and applying the transform to each color channel via
  `skimage.transform.warp` (requires `scikit-image`).
