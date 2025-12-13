## Unreleased
## Fixed
- **Legacy stacking color cast (green tint)**  
  Fixed a long-standing green color bias in legacy mode caused by **mixed CPU/GPU execution paths within the same RGB stack**.  
  The issue occurred when different color channels were processed through different backends (CPU vs GPU), leading to subtle but cumulative normalization divergences.
  ➤ The stacking pipeline is now **backend-consistent per stack**: a given stack is processed entirely on CPU or entirely on GPU.  
  ➤ Partial CPU/GPU channel splitting is explicitly forbidden to guarantee photometric and color integrity.
  This change restores color parity with historical legacy results and stabilizes RGB behavior across runs.

### Fixed
* Robust FITS writing for ASTAP CLI (colour order normalised)
* Filter window now opens once via the streaming scanner path in the GUI
* Restored lightweight WCS footprint previews on large datasets with updated guidance copy
* Winsorized Phase 3 stacking now auto-recovers from NumPy `ArrayMemoryError` by reducing batch sizes or streaming from disk
* Suppressed ASTAP "Access violation" pop-ups by coordinating solver launches through a cooperative inter-process lock (opt-out via `ZEMOSAIC_ASTAP_DISABLE_IPC_LOCK`)
* Hardened ASTAP concurrency handling: Windows `SetErrorMode`, per-image locks, isolated working dirs, rate limiting, and automatic retries (tunable via `ZEMOSAIC_ASTAP_RATE_*` / `ZEMOSAIC_ASTAP_RETRIES` / `ZEMOSAIC_ASTAP_BACKOFF_SEC`) keep the GUI responsive even when multiple tools solve simultaneously
### Added
* Solver option "Convert to Luminance" to force mono before plate-solve
* Added configurable `winsor_worker_limit` (CLI `--winsor-workers` / `-W` and GUI field)
* Added `winsor_max_frames_per_pass` streaming limit for Winsorized rejection (GUI/config)
* Manual frame cap via `max_raw_per_master_tile` (CLI/GUI/config)
* Automatic memory fallback controls for Winsorized stacking (`winsor_auto_fallback_on_memory_error`, `winsor_min_frames_per_pass`, `winsor_memmap_fallback`, `winsor_split_strategy`)
* Fixed incremental assembly with reproject>=0.11
* 16-bit FITS export now writes a 2D luminance primary with R/G/B extensions for broader viewer compatibility, with a legacy RGB cube available from Advanced options
* Added cross-platform GUI icon fallback using existing PNG assets
* Automatic ASTAP executable/data path detection across Windows/macOS/Linux, with environment variable overrides exposed via `zemosaic_config.detect_astap_installation`
* POSIX PyInstaller helper script (`compile/build_zemosaic_posix.sh`) plus macOS/Linux setup documentation

### Changed
* Improved cross-platform startup: GUI icon fallback, optional dependency handling, and GPU detection now degrade gracefully on macOS/Linux (CUDA acceleration remains Windows-only)
* `requirements.txt` no longer lists the unsupported `tk` wheel and only installs `wmi` on Windows, eliminating pip failures on macOS/Linux
