# Follow-up checklist — TwoPass diagnostics

[x] 1) Locate code
- Open zemosaic_worker.py and find run_second_pass_coverage_renorm.
- Identify where gains are computed (it currently logs: "[TwoPass] Computed gains count=... min=... max=...").

[x] 2) Add helper
- Add _two_pass_tile_rgb_stats(arr) near other debug helpers (e.g., near _dbg_rgb_stats).
- Requirements:
  - supports HWC RGB and HW mono
  - uses finite mask only
  - returns: valid_fraction, min/mean/median per channel

[x] 3) Add per-tile logging
- In run_second_pass_coverage_renorm:
  - right after gains computed and before reprojection loop:
    - for each tile:
      - compute stats_pre
      - apply existing gain code (do not modify)
      - compute stats_post
      - logger.debug lines:
        - "[TwoPassTile] idx=%d gain=%.6f pre  valid=... median=[...,...,...] mean=[...] min=[...]"
        - "[TwoPassTile] idx=%d gain=%.6f post valid=... median=[...,...,...] mean=[...] min=[...]"

[x] 4) Optional delta map (only if implemented)
- Ensure it is off by default.
- Only runs under DEBUG (or explicit flag default False).
- Downsample factor fixed (e.g., 8) for speed.
- Save .npy and log a single line:
  - "[TwoPassDelta] wrote delta map: <path> shape=<...> ds=<...>"

[x] 5) Run test
- Launch a run that triggers TwoPass Phase 5 and confirm:
  - TwoPass start/prepared/reproject logs still present.
  - New per-tile logs appear.
  - Gains min/max log unchanged.
- If delta map enabled, confirm file exists.

6) Report back
- Paste 10-20 lines showing:
  - 2-3 tiles pre/post stats (including a “bad-looking seam” tile if identifiable)
  - gains min/max
  - delta map path line (if enabled)
