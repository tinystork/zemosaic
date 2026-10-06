# ZeGrid R0 diagnostic — not a production engine

Run from the repository root with its scientific environment:

```sh
.venv/bin/python tools/zegrid_r0/geometry.py \
  --input /home/tristan/M106/lights \
  --output /tmp/M106_geometry.json --halos 0,8,32
.venv/bin/python -m pytest -q tests/test_zegrid_r0_geometry.py \
  tests/test_sci05_gate_f4_grid_route.py tests/test_grid_mode_stack_plan_paths.py
```

Input directory is scanned recursively. Use a **sources-only** directory, not a
mixture with mosaics, coverage maps or output tiles. No downloads, writes to sources,
stacking, source `.data` access or production Grid dispatch. All unsupported files
are reported; a partially supported corpus is explicitly PARTIAL.

Supported witness domain: primary-HDU 2D FITS, valid RA/DEC undistorted TAN WCS,
no SIP/PV/lookup distortion. Four **outer pixel-edge** corners define a convex
projected polygon in a common visible TAN hemisphere. Other geometry/3D axis
formats require qualification; they are not silently reduced to four corners.

Dependencies are existing project dependencies: NumPy/Astropy/Reproject/Shapely.
WCS and filenames determine geometry; pixel values/alpha/FWHM/reference are unknown.
Header hashes freeze geometry provenance only, not full source contents.

## Metrics and assumptions

- Layouts are aspect-aware comparison points near 4/16/36/64 cells, not Auto defaults.
- Cell integer slices own pixels exactly once. Patch = Cell + explicitly requested
  target-pixel halo, clipped to canvas.
- Narrow phase: projected polygon ∩ pixel-edge rectangle, positive area >1e-8 px²
  (a witness floating sliver threshold, not a frozen production rule).
- Source crop = inverse intersection vertices, conservative integer outward bounds,
  **2 source-pixel interpolation margin**. This is a geometric estimate, not a
  qualified compressed FITS/CFA decoding policy.
- Estimated source pixels = sum of source ROI rectangle areas; RGB planes and
  physical compression overhead excluded. Redundancy denominator = per-source union
  of those rectangles, not canvas area or projected polygon area.
- Full-source/crop ratio compares full-source reads for the **same patch candidates**;
  neither old Grid layout performance nor target pixel reprojection work is measured.
- Coverage union area is geometric; depth histogram is an 80×80 grid sample. Neither
  is canonical positive support or finite scientific coverage. Min/median/mean/max
  frame counts include empty cells. A core can be empty while its halo is not.

## Reproduce the independent order / old-geometry witness

```sh
PYTHONPATH=src:tools/zegrid_r0 .venv/bin/python - <<'PY'
import copy, json
from pathlib import Path
import numpy as np
import geometry as g
from zemosaic import grid_mode as old
frames, rejected = g.read_frames('/home/tristan/M106/lights')
assert not rejected
old._emit = lambda *a, **k: None  # this diagnostic process only
legacy = old.build_global_grid([
    old.FrameInfo(Path(f.key), wcs=copy.deepcopy(f.wcs), shape_hw=f.shape)
    for f in frames], 1., .1)
print('legacy H,W / offset / tile count:', legacy.global_shape_hw,
      legacy.offset_xy, len(legacy.tiles))
base = g.run(frames, layouts=[(5,4)], halos=(8,))
for label, order in [('reverse', frames[::-1]),
        ('random42', [frames[i] for i in np.random.default_rng(42).permutation(len(frames))])]:
    print(label, g.run(order, layouts=[(5,4)], halos=(8,)) == base)
PY
```

The legacy observation supplies WCS/shape directly from headers so its lazy `.data`
loader is not used. No FITS tile is stacked/written. Old scale warnings are expected:
R0 characterizes the CD/CDELT defect, not repairs it.

See `docs/refactor/ZM_ZEGRID_R0_ARCHITECTURE_FREEZE.md` for the proposed architecture,
real measurements, source code evidence, CFA constraint, R1 scope and human gate.
