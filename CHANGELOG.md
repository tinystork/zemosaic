## Unreleased
### Fixed
* Robust FITS writing for ASTAP CLI (colour order normalised)
* Filter window now opens once via the streaming scanner path in the GUI
* Restored lightweight WCS footprint previews on large datasets with updated guidance copy
* Winsorized Phase 3 stacking now auto-recovers from NumPy `ArrayMemoryError` by reducing batch sizes or streaming from disk
### Added
* Solver option “Convert to Luminance” to force mono before plate-solve
* Added configurable `winsor_worker_limit` (CLI `--winsor-workers` / `-W` and GUI field)
* Added `winsor_max_frames_per_pass` streaming limit for Winsorized rejection (GUI/config)
* Manual frame cap via `max_raw_per_master_tile` (CLI/GUI/config)
* Automatic memory fallback controls for Winsorized stacking (`winsor_auto_fallback_on_memory_error`, `winsor_min_frames_per_pass`, `winsor_memmap_fallback`, `winsor_split_strategy`)
* Fixed incremental assembly with reproject>=0.11
* 16-bit FITS export now writes a 2D luminance primary with R/G/B extensions for broader viewer compatibility, with a legacy RGB cube available from Advanced options
* Added cross-platform GUI icon fallback using existing PNG assets
* Added MAC OS support

### Changed
* Improved cross-platform startup: GUI icon fallback, optional dependency handling, and GPU detection now degrade gracefully on macOS/Linux (CUDA acceleration remains Windows-only)
