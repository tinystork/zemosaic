# ZeMosaic — Architecture Archaeology R0 (rework-1)

Mission: `ZM-ARCH-CLEANUP-R0-20261003`
Phase: rework-1 (R0 documentation/archaeology corrections only)
Date: 2026-10-03
Author: Coco (implementation worker), for Junior (architect/reviewer) and Nono (independent reviewer)

> This document is **archaeology**, not cleanup. It records evidence, not decisions to
> delete. No product or test code was modified. No commit was created. This revision
> incorporates the material findings of the Nono review (HIGH-1…HIGH-4, MEDIUM-5/6,
> LOW-7/8) and Junior's independent confirmation.

## 0. Canonical anchors

| Anchor | Value |
| --- | --- |
| Canonical base SHA | `c03d0bb965d073b12ad9978094327829f0d0c366` |
| `origin/main` | `c03d0bb965d073b12ad9978094327829f0d0c366` (verified) |
| `origin/beta` | `c03d0bb965d073b12ad9978094327829f0d0c366` (verified) |
| Docs checkpoint HEAD (this branch) | `035119eb72266eeb8279a63fd244beff803ec213` (`docs: plan ZeMosaic architectural cleanup`) |
| Branch | `refactor/zm-architecture-cleanup-r0-r3` |
| Remote | `https://github.com/tinystork/zemosaic.git` |
| Version | `4.7.0` (`src/zemosaic/__init__.py:__version__`; `version.txt` agrees) |
| Worktree | **clean at R0 start** (after `035119e`); **intentionally dirty at R0 end** (`M todo.md` + untracked `docs/refactor/`) |

### 0.1 Environment (observed)

- OS: Debian GNU/Linux 13 (trixie), kernel `6.12.111+deb13-amd64`, x86_64.
- Python: `3.13.5` (venv `.venv/bin/python`, GCC 14.2.0).
- Key deps (venv `pip list`): numpy `2.5.1`, scipy `1.18.0`, astropy `8.0.1`,
  reproject `0.21.0`, astroalign `2.6.2`, opencv-python-headless `5.0.0.93`,
  photutils `3.0.0`, shapely `2.1.2`, PySide6 `6.11.1`, matplotlib `3.11.1`,
  bottleneck `1.6.0`, zarr `3.3.0`, fsspec `2026.7.0`, psutil `7.2.2`,
  threadpoolctl `3.6.0`, packaging `26.2`, pytest `9.1.1`, cupy-cuda12x `14.1.1`.
- GPU runtime witness (non-invasive): `nvidia-smi` reports **NVIDIA GeForce MX150**,
  driver `550.163.01`, CUDA `12.4`, 2048 MiB VRAM (compute capability 6.1).
  `import cupy` → `14.1.1`; `cupy.cuda.runtime.getDeviceCount()` == 1.
- **JIT / NVRTC fact (corrected, HIGH-2)**: `nvcc` is **not installed**, but this is
  **irrelevant to CuPy JIT**. CuPy ships its own NVRTC; `cupy_backends.cuda.libs.nvrtc`
  imports and reports version `(12, 9)`. A `cupy.RawKernel` witness succeeded on the MX150
  with `nvcc` absent — `RawKernel` produced `[3. 4. 5. 6.]` from `[1,2,3,4]` (+2.0).
  **RawKernel/RawModule/JIT works without nvcc.** The only qualification limits are
  ~2 GB VRAM and that GPU stacking / CPU↔GPU parity were **NOT_RUN** in R0 (no GPU stacking
  execution performed), *not* JIT availability.
- Backend distinction recorded throughout: **requested** (config/flag), **effective**
  (post-guard/fallback decision), **executed** (what actually ran).

## 1. Supported invocation inventory

| Invocation | Evidence | Status |
| --- | --- | --- |
| `zemosaic` (gui-script) | `pyproject.toml` `[project.gui-scripts] zemosaic = "zemosaic._app:main"` — **gui-script, not console_scripts** | ACTIVE official |
| `python -m zemosaic` | `src/zemosaic/__main__.py` calls `_app.main` | ACTIVE official |
| `python run_zemosaic.py` (checkout) | root `run_zemosaic.py` inserts `src/` into `sys.path` then delegates to `zemosaic._app:main` | COMPATIBILITY wrapper |
| PyInstaller frozen | `ZeMosaic.spec:160` `Analysis(['run_zemosaic.py'], hiddenimports=...)`; `_version.py` added as datas (`:38-40`) | ACTIVE build input |
| Direct API `run_hierarchical_mosaic_classic_legacy(...)` | imported directly by `tests/test_solver_port_integration.py:218` | reachable, NOT a documented public API |

`_app._determine_backend` (`_app.py:41`) maps `--qt-gui` (ignored) and `--tk-gui`
(prints "no longer supported", stripped) → backend is always `qt`. This is the sole
legacy-flag normalization; the runtime is Qt-only.

## 2. Official execution maps

### 2.1 Entry → Qt bootstrap

```
zemosaic console script / python -m zemosaic / run_zemosaic.py
   └─> zemosaic._app.main            (_app.py)   [multiprocessing.freeze_support()]
         └─ _determine_backend(argv)             (qt-only)
         └─ _play_opening_gif_animation_once()   (optional PySide6 QMovie splash)
         └─ zemosaic_gui_qt.run_qt_main()        (zemosaic_gui_qt.py:7090)
              (finally: cupy memory pool free_all_blocks() on shutdown)
```

- `_app.main` owns GPU cleanup: `_app.py:307` frees CuPy default pool when
  `CUPY_AVAILABLE` and `cupy` already imported.
- `CUPY_AVAILABLE` is imported from `.cuda_utils` (`_app.py:23`).

### 2.2 GUI → worker process → dispatcher

```
run_qt_main (zemosaic_gui_qt.py:7090)
  └─ WorkerController.spawn_worker_process        (zemosaic_gui_qt.py:562)
       └─ multiprocessing.get_context("spawn")    (POSIX; :586)
       └─ Process(target=run_hierarchical_mosaic_process, args=(queue,) + args, kwargs=...)
  └─ _build_worker_invocation (zemosaic_gui_qt.py:6133)
       └─ _disable_phase45_config(cfg)            (:4167)  forces inter_master_merge_enable=False
       └─ _serialize_config_for_save()            → worker kwargs snapshot
       └─ _build_solver_settings_dict()           → solver_settings_dict payload
       └─ skip_filter_ui / filter_overrides / filtered_header_items / early_filter_enabled
```

Worker side (`zemosaic_worker.py`):

```
run_hierarchical_mosaic_process (zemosaic_worker.py:36149)   [the Process target]
  └─ queue_callback(...) proxies legacy + STAGE_PROGRESS onto progress_queue
  └─ rename_map (GUI name -> worker arg name)   (:36251+)   e.g. stacking_normalize_method -> stack_norm_method
  └─ winsor limits string parse -> parsed_winsor_limits tuple
  └─ "_config" suffix promotion via inspect.signature(run_hierarchical_mosaic)
  └─ drop kwargs not in run_hierarchical_mosaic signature  (silent filter)
  └─ defaults for stack_ram_budget_gb_config=0.0, num_base_workers_config=0
  └─ SIGTERM/SIGINT handler -> cancel_active_zesolver_solves() + KeyboardInterrupt
  └─ heartbeat writer thread (crash breadcrumbs)
  └─ run_hierarchical_mosaic(**final_kwargs)      (:36373 / retry :36558)
```

`run_hierarchical_mosaic` (dispatcher, `zemosaic_worker.py:30675`) precedence:

1. **Grid first**: `detect_grid_mode(input_folder)` (presence of `stack_plan.csv`,
   `grid_mode.py:667`). If detected and `grid_mode.run_grid_mode` importable →
   `grid_mode.run_grid_mode(...)`, `return`. If module missing → `raise RuntimeError`
   **with NO classic fallback** (`:30920-30930`). If `run_grid_mode` raises → logged and
   **re-raised (abort, no fallback)**.
2. **SDS**: `sds_mode_flag` resolved by precedence
   `filter_overrides["sds_mode"]` → `global_wcs_plan_override["sds_mode"]` →
   `sds_mode_default` config → `False` (`:30949-30971`). When true, SDS runs inline.
3. **Classic**: `if (not grid) and (not sds) → run_hierarchical_mosaic_classic_legacy(...)`
   (`:30973`, `:30983`).

### 2.3 Classic / SDS / Grid

- **Classic** = `run_hierarchical_mosaic_classic_legacy` (`:23352`), called only when
  neither grid nor SDS. It is an *official active path*. Note: this function **also
  contains its own SDS resolution block** (`:23690-23715`) and shares SDS finalizers
  (see §4.3) — reachable only if called directly (e.g. tests) or if its internal
  `sds_mode_flag` resolves true. Within the normal dispatcher it is always entered with
  `sds_mode_flag=False`.
- **SDS** = inline branch of `run_hierarchical_mosaic` (`:31448+`); shared finalizers
  `_finalize_sds_global_mosaic` (`:5251`), `_mask_sds_low_coverage_pixels` (`:4968`) are
  also called from classic_legacy's SDS path (`:27207`).
- **Grid** = `grid_mode.run_grid_mode` (`grid_mode.py:4152`).

### 2.4 Phase 4.5 / Phase 5 (shared pipeline)

- Shared helper `_run_shared_phase45_phase5_pipeline` (`zemosaic_worker.py:11009`).
- `phase45_active_flag = bool(phase45_options.get("enable"))` gates only the Phase 4.5
  inter-master merge block. **Phase 5 always executes afterward** regardless of the 4.5
  flag (`:11042` branch, then `:11099+` Phase 5 setup unconditionally).
- `phase45_options["enable"]` comes from `_build_phase45_options_dict` →
  `bool(inter_master_merge_enable_config)` (`:26950`, `:33157`).
- Config default `inter_master_merge_enable=False` (`zemosaic_config.py:200`).
- Qt GUI **forces** it False via `_disable_phase45_config` (`zemosaic_gui_qt.py:4167`),
  called from `_build_worker_invocation` (`:6136`). Legacy Tk GUI also forces False
  (`zemosaic_gui.py:370`).
- **Phase 4.5 reachability conclusion**: DISABLED on both official GUI paths (hard-forced
  off). The worker pipeline still supports it programmatically through
  `inter_master_merge_enable_config=True` (config cache / direct call), so it is
  **DORMANT BUT REACHABLE** (programmatic/compatibility), not dead. The shared helper must
  not be treated as 4.5-only: it runs Phase 5 even when 4.5 is off.

## 3. Worker/process contract map

- **Spawn**: `multiprocessing.get_context("spawn")` preferred on POSIX
  (`zemosaic_gui_qt.py:581-592`) so the worker does not inherit a live Qt app. Worker is
  pickled as `target=run_hierarchical_mosaic_process` with `(queue,) + args` and kwargs.
  Legacy GUI (`zemosaic_gui.py:4827-4836`) also targets `run_hierarchical_mosaic_process`.
- **Queues/protocol**: single `multiprocessing.Queue`. Callback tuples:
  `(message_key_or_raw, progress_value, level, cb_kwargs)` legacy form, plus
  `("STAGE_PROGRESS", stage, current, {"total": total})` stage form
  (`zemosaic_worker.py:36183-36220`). Control messages `WARN`/`ERROR` also emit crash
  breadcrumbs.
- **Cancellation**: GUI `WorkerController.stop(graceful=...)`; worker installs
  SIGTERM/SIGINT handlers that call `cancel_active_zesolver_solves()` then raise
  `KeyboardInterrupt` (`:36330-36340`). Filter-close during solve is guarded by
  `test_zesolver_filter_cancel_hg2.py`.
- **WCS filter handoff**: `_build_worker_invocation` builds `filtered_header_items` /
  `filter_overrides` when `skip_filter_ui` is set and prior filter results exist
  (`:6154-6163`); WCS count logged via `count_filter_handoff_wcs` (`solver_port.py:218`).
- **kwargs filtering/precedence**: the process wrapper renames, parses, suffixes `_config`,
  then **silently drops** any key not in `run_hierarchical_mosaic`'s signature
  (`zemosaic_worker.py:36330`). GUI parameter *values* flow through the snapshot; the
  wrapper only reshapes *names*.
- **Checkpoints/resume/caches**: `use_existing_master_tiles`, master-tile reuse,
  `cache_retention` (`run_end`/`per_tile`/`keep`), Phase 1/5 checkpoint and crash
  breadcrumbs (`_configure_crash_breadcrumbs`, heartbeat). No cache/restart format
  changed in R0.
- **Resources/GPU**: Phase 5 GPU gate = `use_gpu_phase5 and gpu_id_phase5 is not None and
  CUPY_AVAILABLE and gpu_is_available()`, then `apply_gpu_safety_to_phase5_flag`
  (`zemosaic_gpu_safety`), `CUDA_VISIBLE_DEVICES` masking, `ensure_cupy_pool_initialized`
  (`:31486-31521`). Phase 3 GPU auto uses `_p3_gpu_stack_from_paths` (imported `:13389`,
  guarded), with OOM shrink + one retry then hard-disable for the run
  (`_stack_master_tile_auto`, `:15763`).

## 4. Stacking maps (caller → normalize → weight → reject → combine)

### 4.1 Classic (CPU/GPU) — `zemosaic_align_stack`

Entry: `stack_aligned_images` (`:4767`) → chooses
`stack_winsorized_sigma_clip` (`:1756`) / `stack_kappa_sigma_clip` (`:2399`) /
`stack_linear_fit_clip` (`:2526`) by `stack_reject_algo` (worker dispatch `:15644-15668`).
Each wrapper resolves `use_gpu` via `stack_use_gpu` → `use_gpu_stack` → `use_gpu` and
calls `gpu_stack_winsorized` (`:1539`) / `gpu_stack_kappa` (`:1628`) /
`gpu_stack_linear` (`:1695`) when the GPU plan allows, else the CPU implementations.

- Normalization: `normalize_method` ∈ `linear_fit` (`_normalize_images_linear_fit`,
  `:3296`) | `sky_mean` (`_normalize_images_sky_mean`, `:3389`) | (else none)
  (`:4991-4996`).
- Weighting: `_compute_quality_weights` (`:4241`) → `noise_variance`
  (`_calculate_image_weights_noise_variance`, `:3519`) | `noise_fwhm`
  (`_calculate_image_weights_noise_fwhm`, `:3772`, falls back to `noise_variance` if
  photutils absent or no usable weights) | `none`/unknown → no weights.
- Rejection: kappa-sigma (`_reject_outliers_kappa_sigma`, `:4371`), winsorized sigma clip
  (`_reject_outliers_winsorized_sigma_clip`, `:4486`), linear-fit clip
  (`_reject_outliers_linear_fit_clip`, `:4726`).
- Combine: mean (weighted) or median (median ignores weights).
- Backend: CPU (numpy) or GPU (cupy) via `_plan_gpu_stack_execution` (`:1139`) and
  `_should_use_wsc_streaming` (`:1294`) for WSC streaming. WSC impl resolved by
  `resolve_wsc_impl` → `wsc_pixinsight_core` / `wsc_pixinsight_core_streaming_numpy`
  (`:1394-1520`, `:1798`), with `wsc_parity_check` (`:1050`).
- **linear_fit naming** (two distinct concepts):
  - *linear-fit NORMALIZATION* = `_normalize_images_linear_fit` (`:3296`).
  - *linear-fit REJECTION* = `stack_linear_fit_clip` → `_reject_outliers_linear_fit_clip`
    (`:4726`) / `gpu_stack_linear` (`:1695`).
  These are separate code paths; do not conflate.

### 4.2 Grid — `grid_mode`

- CPU path `_stack_weighted_patches` (`:1913`): normalize via `_normalize_patches`
  (`:1798`), rejection via the **established** `_reject_outliers_kappa_sigma` /
  `_reject_outliers_winsorized_sigma_clip` (imported from `zemosaic_align_stack`,
  `grid_mode.py:133-139`), then weighted mean/median combine.
- GPU path `_stack_weighted_patches_gpu` (`:1984`): normalizes on GPU, then if `stack_core`
  available calls `stack_core(..., backend='gpu')` with `normalize_method='none'` and the
  user's `rejection_algorithm`/`final_combine` (`:2010-2028`). See §11 for the exact
  `stack_core` rejection semantics (kappa-sigma uses the established helper via
  `cp.asnumpy`; winsorized-sigma is simplified). If `stack_core` is unavailable it falls
  back to the legacy GPU logic (`:2034+`) using the established numpy rejection helpers
  via `cp.asnumpy` round-trips.
- `run_grid_mode` defaults (`:4157-4163`): `stack_norm_method="linear_fit"`,
  `stack_weight_method="noise_variance"`, `stack_reject_algo="kappa_sigma"`,
  `stack_kappa_low/high=3.0`, `stack_final_combine="mean"`.
- Grid GPU flag: `use_gpu` param else config `use_gpu_grid` (default `True`,
  `zemosaic_config.py:165`).

### 4.3 SDS — inline in `run_hierarchical_mosaic` / `classic_legacy`

- Final SDS stack uses `stack_winsorized_sigma_clip` / `stack_kappa_sigma_clip` via
  `zemosaic_align_stack` (`:38702-38730`) with `weight_method`, `kappa`, `winsor_limits`
  propagated; plain `nanmedian`/`nanmean` combine otherwise. Shared finalizers
  `_finalize_sds_global_mosaic` / `_mask_sds_low_coverage_pixels`.

### 4.4 Phase 4.5 — inter-master merge (corrected rework-2)

Entry `_run_phase4_5_inter_master_merge` (`zemosaic_worker.py:7335`), called from
`_run_shared_phase45_phase5_pipeline` only when `phase45_options["enable"]` is true.
Backend throughout is **CPU (numpy)**. The Phase 4.5 normalization/photometry data-flow is
**five labeled steps grouped into three temporal stages (pre-stack / stack / post-stack),
A…E** (below).
- **Backend is CPU (numpy).** The configured rejection wrappers are called with
  `zconfig=None` (`:8343`, `:8361`, `:8372`), and each wrapper resolves `use_gpu` **from
  `zconfig` only** (`stack_winsorized_sigma_clip` `zemosaic_align_stack.py:1792-1795`;
  `stack_kappa_sigma_clip` `:2413-2416`; `stack_linear_fit_clip` `:2540-2543`) →
  `zconfig is None` → `use_gpu=False`. `parallel_plan` does not turn these wrappers GPU,
  and `stack_cfg_phase45` contains no GPU key. **Phase 4.5 is CPU-only in code.**
- **Alpha-weighted early branch (separate scientific path).** `weights_ready =
  any(isinstance(w, np.ndarray) for w in frame_weights)` (`:8315`). When true, the code
  computes a direct numpy weighted mean: `num = np.nansum(frames_np * weight_expanded,
  axis=0)`, `den = np.nansum(weight_expanded, axis=0)`, `super_arr = np.where(den > 0,
  num / den, np.nan)` and `alpha_out = (np.nanmax(weight_stack, axis=0) * 255.0)`
  (`:8317-8337`). Weight maps are `np.clip(np.nan_to_num(w, nan=0.0), 0.0, 1.0)`,
  non-array/non-matching-shape entries become `ones`. **This branch bypasses the
  configured reject/combine entirely.** On exception, `super_arr=None` and it falls
  through to the configured rejection path.
- **Configured rejection path** (only when `super_arr is None`): selection comes from
  `reject_algo = stack_cfg.get("reject_algo", stack_cfg.get("stack_reject_algo",
  "winsorized_sigma_clip"))` (`:7682`), NOT from `inter_cfg["stack_method"]` (which is
  passed into `inter_cfg` but **never read** inside the merge). Branches: `winsor`/
  `winsorized_sigma_clip` → `stack_winsorized_sigma_clip`; `kappa_sigma` →
  `stack_kappa_sigma` (if present) else `stack_kappa_sigma_clip`; `linear_fit_clip` →
  `stack_linear_fit_clip`. All with `zconfig=None`, `weight_method`, and
  `stack_kwargs` (kappa/winsor_limits/winsor_max_*). If still `None`, a final
  `nanmedian`/`nanmean` combine (`final_combine`) is the last resort.
- **`inter_master_stack_method` is passed but not consumed** by the merge implementation;
  the real rejection comes from `stack_cfg["reject_algo"]`. This is existing
  scientific/architectural debt — recorded, not fixed.

**A. Helper-gated pre-stack affine / micro-align branches — INERT at BASE.** The helper
availability gates `micro_align_available` / `photometry_estimator_available` /
`photometry_apply_available` are `hasattr(zemosaic_align_stack, ...)` checks
(`:7412-7414`). At BASE, `zemosaic_align_stack` defines **none** of `micro_align_stack`,
`estimate_affine_photometry`, `apply_affine_photometry` (repo-wide `rg` finds only call
sites; runtime `hasattr` is `False False False`). Therefore micro-align (`:8034`, gated
`micro_align_available and not alpha_weights_present`), intra-group affine photometry
`do_chunk_photometry` (`:7701-7705`, requires `photometry_estimator_available and
photometry_apply_available`) with its `estimate_affine_photometry`/`apply_affine_photometry`
calls (`:7766`, `:8084`, `:8095`), legacy affine normalization (`:8151-8154`, gated the same
way), and the later global-affine inter-super branch (`:8865`, gated the same way) are all
**DORMANT / UNREACHABLE at BASE**. They are NOT PROVEN DEAD (compatibility/future helper
injection is not excluded). The config flags `photometry_intragroup`/`photometry_intersuper`/
`photometry_clip_sigma` are read at `:7364-7366`, but the affine branches additionally
require the absent helpers.

**B. ACTIVE pre-stack explicit linear_fit / sky_mean normalization** (`:8202-8290`). The
only genuinely active normalization at BASE; runs **before** the stack call, mutating
`frames` **in place**. Gate: `if norm_method in ("linear_fit", "sky_mean") and len(frames)
>= 2`. `norm_method` read at `:7676-7700` from `stack_cfg["stacking_normalize_method"]` →
`normalize_method` → `stack_norm_method` → `none`. Details:
- `linear_fit`: per channel, common-finite mask between ref (`frames[0]`) and source; min
  overlap `max(5000, ceil(1% pixels))`; sigma clipping via `clip_sigma_norm` (derived from
  `photometry_clip_sigma`, clipped `[0.1, 10.0]`, `:8168-8174`); x/y mean-centered OLS
  `slope = dot(xv,yv)/dot(xv,xv)`; **slope clipped to `[0.25, 4.0]`**;
  `intercept = ym - slope*xm`; slope/intercept applied in place to valid source pixels.
- `sky_mean`: per channel, percentile band `[sky_low, sky_high]` (from
  `intertile_sky_percentile`, default `30.0, 70.0`, `:8175-8200`); per-channel medians;
  additive `delta = bg_ref - bg_src` added in place to valid source pixels.
- Exceptions are caught and logged (`"Chunk photometric normalization skipped (error)"`)
  and processing continues; however, `frames`/channels are mutated **in place** inside the
  nested per-frame/per-channel loops with **no copy or rollback**, so if an exception occurs
  after some assignments, **partial in-place normalization may remain** for the already-
  processed frames/channels.

**C. Stacking — alpha-weighted early branch OR configured rejection branch.** (unchanged
from rework-1; see the two bullets immediately above the `inter_master_stack_method` note).

**D. ACTIVE post-stack inter-super gain-only normalization** (`:8622-8832`). Independent of
the absent affine helpers. Gate: `photometry_intersuper and len(candidate_super_tiles) >= 2`
where candidates are saved replacement super tiles (`:8624-8629`). Reads each saved
super-tile FITS (`fits.open(memmap=True, do_not_scale_image_data=True)`), computes
per-channel medians with valid-pixel counts as weights (IQR-based sigma clip when
`photometry_clip_sigma > 0`), builds a weighted-median reference (`ref_mode="median"`, falls
back to dominant tile), computes per-channel gains `ref/med` **clipped by
`two_pass_cov_gain_clip` default `[0.85, 1.18]`** (`:8700-8712`), then **reopens each saved
FITS `mode="update"`**, multiplies channels in place, writes `ZM45NORM=True` + HISTORY
(`:8758-8795`). This path is **genuinely reachable at BASE**.

**E. Global-affine inter-super branch** (`:8837+`, `estimate_affine_photometry` at `:8865`)
is helper-gated (`photometry_estimator_available and photometry_apply_available`) and
therefore **INERT at BASE**.

**Nothing applies affine photometry to `super_arr` in memory.** Active B acts on `frames`
pre-stack; active D rewrites saved super-tile FITS post-stack. No affine correction is
applied to the in-memory stacked result.

## 5. Filter Qt vs legacy/Tk + shared helpers + reverse coupling

- **Qt filter** `zemosaic_filter_gui_qt.py` is the official path. It imports three legacy
  helpers from `zemosaic_filter_gui` with inline fallback copies
  (`:448-470`): `_merge_small_groups`, `_split_group_by_orientation`,
  `_circular_dispersion_deg`. It also imports worker helpers
  (`cluster_seestar_stacks_connected`, `_auto_split_groups`,
  `_compute_max_angular_separation_deg`) from `zemosaic_worker` (`:434-441`) — **reverse
  coupling** filter→worker.
- **Legacy filter** `zemosaic_filter_gui.py` contains `launch_filter_interface` (`:681`)
  and Tk imports that are **inside functions**, not at module top (`:1062`, `:4979`,
  `:8840`). The worker imports `launch_filter_interface` **dynamically** at two sites
  (`zemosaic_worker.py:25615`, `:31844`), only when `early_filter_enabled`, wrapped in
  `try/except ImportError`. Legacy GUI (`zemosaic_gui.py`) also imports it (`:2708`,
  `:4271`).
- `core/tk_safe.py` (`patch_tk_variables`) is imported only by the legacy GUI
  (`zemosaic_gui.py:48`) and legacy filter (`zemosaic_filter_gui.py:1065`).

## 6. Solver / ZeAnalyser

- **SolverPort** (`solver_port.py`): `SolverAdapter` Protocol (`:148`),
  `LegacySolverAdapter` (`:243`) reproducing ASTROMETRY→astrometry+ASTAP fallback,
  ANSVR→ansvr+ASTAP fallback, else direct ASTAP (`:243-...`). `count_filter_handoff_wcs`
  (`:218`) counts WCS-bearing filtered header items.
- **ZeSolver** (`zesolver_adapter.py`): `ZeSolverAdapter` (`:217`) talks to the public
  `zesolver.api.v1`, lazy `SolverRuntime`/`SolverSession`, cancellation tokens,
  `discover_zesolver()` (`:86`). Standalone: ZeAlfie not required at runtime.
- **ZeAnalyser launch**: interop via `zesoftware_interop.json` + `test_zeanalyser_launch.py`
  (24 passed) exercising the public installed contract.

## 7. Data-plane execution maps (HIGH-1 — completed)

Each entry: **callers → main functions/modules → outputs/effects**, mode(s), evidence
kind (static vs runtime), error/fallback behavior, extraction risk.

### 7.1 FITS read / validation / write (axis & header semantics)

- **Read + validate**: `zemosaic_utils.load_and_validate_fits` (`zemosaic_utils.py:3903`).
  Opens with `fits.open(..., memmap=False, do_not_scale_image_data=True)`; selects a
  priority image HDU (`idx==0` or name `SCI`/`IMAGE`/`PRIMARY`, else first image HDU);
  extracts an `ALPHA` HDU into `info["alpha_mask"]` (uint8); copies header. Handles
  BZERO/BSCALE and optional float32 normalization / non-finite fix. Returns
  `(data, header, info)`; empty/corrupt → `(None, fallback_header, info)`.
- **Grid read**: `grid_mode._open_fits_safely` (`:379`) = `fits.open(memmap=False,
  do_not_scale_image_data=True)`; `lecropper.load_fits_rgb` (`:1140`) and
  `save_cropped_fits` (`:1175`).
- **Header WCS write-in-place**: filter `_write_header_to_fits_local`
  (`zemosaic_filter_gui_qt.py:1512`, legacy `zemosaic_filter_gui.py:1228`) using
  `fits.open(mode="update", memmap=False)`.
- **Final write**: `zemosaic_utils.write_final_fits_uint16_color_aware`
  (`zemosaic_utils.py:1495`) — rescales to uint16 preserving RGB planes
  (`force_rgb_planes`, `legacy_rgb_cube`), NaN→0, `_ensure_float32_no_nan`, with
  auto-range detection (`[0,1]` vs `[0,65535]` vs arbitrary).
- Axis semantics: HWC (channels-last) is canonical downstream; `_ensure_hwc_tile`
  (`stack_core.py:71`) / `_ensure_hwc_array` (`grid_mode.py:387`) normalize 2D/CHW inputs.
- Mode: CPU. Evidence: static (signatures/bodies). Fallback: missing astropy → explicit
  `RuntimeError` in the uint16 writer. Risk: header/HDU-selection semantics must be
  preserved byte-for-byte on any extraction.

### 7.2 WCS acquisition / validation / global-plan / write / handoff

- **Handoff counter**: `solver_port.header_carries_wcs_material` (`:169`) and
  `count_filter_handoff_wcs` (`:218`) count WCS-bearing filtered header items.
- **Filter-side WCS**: `_build_wcs_from_header` (`zemosaic_filter_gui_qt.py:1480`),
  `_header_has_wcs` (`:1495`), `_persist_wcs_header_if_requested` (`:1535`); legacy
  counterparts (`zemosaic_filter_gui.py:1214-1265`, `:2359`, `:2569`).
- **Global plan (worker)**: `_prepare_global_wcs_plan` (`zemosaic_worker.py:5891`) builds
  a Mosaic-First plan (`enabled/fits_path/json_path/descriptor/meta/wcs/width/height/mode/
  coadd_method/coadd_k/winsor_limits`) from `filter_overrides["global_wcs_path"]`,
  `global_wcs_json`, `global_wcs_output_path`, then loads via
  `_load_global_wcs_descriptor_safe`; `_runtime_build_global_wcs_plan` (`:6071`) is the
  runtime variant. Overrides precedence is handled here (`mode` override etc.).
- **Grid WCS**: `_load_frame_wcs` (`grid_mode.py:875`), `_extract_pixel_scale_deg` (`:934`),
  `_is_degenerate_global_wcs` (`:961`), `_strip_wcs_distortion` (`:991`),
  `_build_fallback_global_wcs` (`:1031`), `_clone_tile_wcs` (`:1131`).
- **Solver WCS write**: `zemosaic_astrometry._update_fits_header_with_wcs_za` (`:1224`),
  `_parse_wcs_file_content_za`/`_v2` (`:964`, `:1022`).
- Handoff chain: filter solves → `filtered_header_items` (in-memory) →
  `_build_worker_invocation` → process kwargs → `run_hierarchical_mosaic` →
  `filtered_header_items_arg` → Phase 1 reuse (`:25576-25654`). WCS is passed in memory;
  no second solve (4.7.0 invariant).
- Mode: CPU. Evidence: static + runtime (handoff tests passed). Risk: high — WCS handoff
  and write-WCS choice are 4.7.0 invariants.

### 7.3 Reprojection (CPU/GPU selection and fallback)

- Imports: `find_optimal_celestial_wcs` / `reproject_and_coadd` from
  `reproject.mosaicking`, `reproject_interp` from `reproject` (`zemosaic_worker.py:13253-13255`),
  guarded; `reproject_and_coadd_wrapper` from `zemosaic_utils` (`:13285`, impl
  `zemosaic_utils.py:6579`).
- **CPU tile reprojection**: `reproject_tile_to_mosaic` (`:14343`) uses `reproject_interp`
  on `(plane, wcs)` → `(data, footprint)`; Grid `_reproject_frame_to_tile`
  (`grid_mode.py:1590`); Phase 4.5 `_reproject_chunk_tile` (`:7894`) via a
  `ThreadPoolExecutor` (`:7978`) — CPU.
- **Final coadd**: `reproject_and_coadd_wrapper` (`:19511`, `:21623`) — CPU (reproject's
  numpy path).
- **GPU assembly placeholders**: `zemosaic_utils.gpu_assemble_final_mosaic_reproject_coadd`
  and `gpu_assemble_final_mosaic_incremental` (`zemosaic_utils.py:5268`, `:5279`) are
  **deprecated `NotImplementedError` placeholders** — GPU final assembly is routed
  through `assemble_final_mosaic_reproject_coadd(use_gpu=True)` instead (per the docstring),
  and incremental GPU is explicitly "not implemented; use CPU". In R0 no GPU assembly was
  executed.
- Fallback: `reproject_interp is None` / `REPROJECT_AVAILABLE` false → dependency gate
  short-circuits (`:7352`, `:13137`, `:14371`). Error during a single tile reprojection →
  logged and tile skipped/None (`:19281`, `:19308`).
- Mode: CPU (executed); GPU only via the `use_gpu=True` worker path (NOT_RUN). Risk:
  medium-high — reprojection dominates Phase 5 runtime.

### 7.4 Photometry / background matching / equalization

- **Grid photometry**: `estimate_tile_background` (`grid_mode.py:559`) sigma-clipped median;
  `compute_tile_photometric_scaling` / `apply_tile_photometric_scaling`
  (`stack_core.py:125`, `:194`) per-channel gain/offset.
- **RGB equalization**: `equalize_rgb_medians_inplace` (`zemosaic_align_stack.py:303`),
  `grid_post_equalize_rgb` (`grid_mode.py:2472`), `_normalize_background`
  (`grid_mode.py:2452`), worker `equalize_black_point_rgb` (`:3042`),
  `_equalize_rgb_black_level_hwc` (`:3201`), `_apply_final_mosaic_rgb_equalization` (`:2902`).
- **Background matching (GPU)**: `estimate_background_map_gpu` (`zemosaic_utils.py:6693`),
  `_finalize_match_background` (`:5735`).
- **SDS photometry**: `_normalize_sds_megatiles_photometry` (`zemosaic_worker.py:5178`),
  `_sds_compute_tile_payload` (`:5095`), `_sds_choose_reference_index` (`:5151`).
- **Inter-master photometry (Phase 4.5)**: see §4.4 A–E for the full temporal map.
  Summary: (A) helper-gated affine/micro-align branches are **INERT at BASE** —
  `zemosaic_align_stack` defines no `estimate_affine_photometry` / `apply_affine_photometry` /
  `micro_align_stack` (runtime `hasattr` `False False False`; availability gates
  `zemosaic_worker.py:7412-7414`; config flags read at `:7364-7366`); (B) ACTIVE pre-stack
  `linear_fit`/`sky_mean` normalization in place on `frames` (`:8202-8290`); (D) ACTIVE
  post-stack inter-super gain-only normalization rewriting saved super-tile FITS
  (`:8622-8832`); (E) global-affine inter-super branch helper-gated and INERT (`:8837+`).
  Summary helper `_log_affine_photometric_summary` (`:6927`). Nothing applies affine
  photometry to the in-memory `super_arr`.
- Mode: CPU mostly; GPU for `estimate_background_map_gpu`. Risk: medium — these are
  science stages; any extraction must preserve exact scaling/clip semantics.

### 7.5 Assembly paths + final write

- Assembly dispatch: `USE_INCREMENTAL_ASSEMBLY = final_assembly_method == "incremental"`
  (`:11099`); `assemble_final_mosaic_incremental` (`:18868`) vs
  `assemble_final_mosaic_reproject_coadd` (`:19526`, aliased
  `assemble_final_mosaic_with_reproject_coadd` `:23318`). SDS uses
  `assemble_global_mosaic_sds` (`:37965`); Classic non-SDS uses
  `assemble_global_mosaic_first` (`:38833`, `_assemble_global_mosaic_first_impl` `:36561`).
- Final write: `write_final_fits_uint16_color_aware` (see §7.1) after quality pipeline
  (`_apply_final_mosaic_quality_pipeline` `:1310`) and DBE (`_apply_final_mosaic_dbe_per_channel`
  `:3648`).
- Mode: CPU (reproject+coadd), GPU final assembly NOT_RUN (placeholder docstring only).
  Risk: high — this is the output boundary.

### 7.6 Checkpoints / cache / resume / retention / invalidation

- Crash breadcrumbs: `_configure_crash_breadcrumbs` (`:164`), `_emit_crash_breadcrumb`
  (`:224`); modes `always`/`errors_only`/`off`; heartbeat writer thread (`:36359`).
- Phase 5 checkpoint: `_write_phase5_checkpoint` (`:24539`) / `_try_load_phase5_checkpoint`
  (`:24612`), keyed by `final_assembly_method` (`:24634-24636`).
- Phase 1 resume: `_try_resume_phase1` (`:24711`) / `_write_phase1_resume_cache` (`:24829`);
  `_set_eta_prior_total_from_history` (`:23825`, `:31243`).
- Tile cache: `_safe_load_cache` (`:15531`), `_register_tile_cache_paths` (`:28253`),
  `_release_tile_cache_paths` (`:28280`), `_cache_allowed` (`:12273`).
- Retention: `cache_retention` ∈ `run_end`/`per_tile`/`keep` (`:23770-23778`, `:8607`).
- Invalidation: `_invalidate_zesolver_discovery_cache` (`:14171`).
- Mode: CPU (filesystem). Risk: high — resume/restart formats must not be migrated or
  invalidated incidentally by any refactor.

### 7.7 Preview / progress / queue / crash-breadcrumb lifecycle

- Progress callback protocol: legacy tuples + `STAGE_PROGRESS` (see §3); `_eta_seconds_from_progress`
  (`:2403`); per-phase progress weights (`PROGRESS_WEIGHT_*`, `:31485+`).
- Grid progress: `_GridProgressReporter` (`grid_mode.py:164`) throttles stage/tile/ETA
  emissions (`emit_eta` `:208`, `_emit_stage_progress` `:267`).
- Preview: `_save_preview_png` (`:29954`), `_apply_preview_quality_crop` (`:6566`),
  preview reprojection `reproject_interp` (`:10261`), `intertile_preview_size` config.
- Inter-tile bridge: `_intertile_progress_bridge` (`:9529`); tile callbacks
  `_tile_progress_callback` (`:16209`), `_touch_progress` (`:16178`).
- Mode: CPU. Risk: low-medium — protocol/order must be preserved across any worker split.

### 7.8 Resource / GPU planning and cleanup

- VRAM budget: `_compute_phase5_vram_budget_bytes` (`:4179`).
- Memory orchestration: `_memory_orchestrator_profile` (`:16037`),
  `_phase3_memory_orchestrator_profile` (`:15938`), `_write_memory_orchestrator_report`
  (`:16076`), `_apply_ram_budget_to_groups` (`:12635`), `_estimate_group_memory_bytes`
  (`:10754`), `_estimate_per_frame_cost_mb` (`:10787`).
- Parallel-plan RAM probe: `parallel_utils._probe_ram` (`:127`), `_estimate_bytes` (`:238`),
  `_compute_rows_per_chunk` (`:248`).
- GPU safety: `apply_gpu_safety_to_parallel_plan` (`zemosaic_gpu_safety.py:378`),
  `apply_gpu_safety_to_phase5_flag` (`:498`); `_shrink_parallel_plan_for_gpu`
  (`zemosaic_worker.py:15746`); `_log_memory_usage` (`:13618`),
  `_emit_gpu_info_summary` (`:10978`).
- Cleanup: `_app.main` frees CuPy default pool at shutdown (`_app.py:307`).
- Mode: CPU planning; GPU guarded. Risk: medium — budgets/retries/fallbacks and their
  observability are preserved invariants.

## 8. Module / path classification table

Classification vocabulary: ACTIVE / COMPATIBILITY / TEST-DIAGNOSTIC /
DORMANT BUT REACHABLE / SUSPECTED DEAD / PROVEN DEAD / UNKNOWN.

| Module (src/zemosaic) | Classification | Evidence / callers | Risk / review need |
| --- | --- | --- | --- |
| `_app.py` | ACTIVE | gui-script entry; `__main__` + `run_zemosaic.py` delegate here | official bootstrap; Qt-only |
| `__main__.py` | ACTIVE | `python -m zemosaic` alias | trivial |
| `zemosaic_gui_qt.py` | ACTIVE | `_app.main` → `run_qt_main`; owns worker spawn + phase45-off + solver dict | core UI; high risk |
| `zemosaic_filter_gui_qt.py` | ACTIVE | official filter; imports worker + legacy filter helpers | reverse coupling to worker; high risk |
| `zemosaic_worker.py` | ACTIVE | dispatcher, classic_legacy, SDS, Phase 4.5/5, process wrapper (38 882 lines) | monolith; all extraction risk |
| `zemosaic_align_stack.py` | ACTIVE | classic CPU/GPU stacking; imported by worker, grid, core, tests | science-critical |
| `zemosaic_align_stack_gpu.py` | ACTIVE | Phase-3 GPU stack (`_p3_gpu_stack_from_paths`); imported by worker | science-critical |
| `zemosaic_stack_core.py` | ACTIVE | `stack_core` reused by Grid GPU; imported by grid_mode | see §11 |
| `grid_mode.py` | ACTIVE | Grid/Survey dispatcher; imported by worker | science-critical |
| `zemosaic_config.py` | ACTIVE | config defaults + normalization; imported by worker | config precedence |
| `zemosaic_utils.py` | ACTIVE | 12 importers | high coupling |
| `zemosaic_astrometry.py` | ACTIVE | 4 importers | solve/astrometry |
| `zemosaic_gpu_safety.py` | ACTIVE | 2 importers (worker + Phase 5 flag guard) | GPU safety |
| `zemosaic_resource_telemetry.py` | ACTIVE | 2 importers | telemetry |
| `parallel_utils.py` | ACTIVE | 3 importers (incl. worker, align_stack) | parallel plans |
| `solver_port.py` | ACTIVE | 4 importers | solver boundary |
| `zesolver_adapter.py` | ACTIVE | 2 importers (filter_qt, worker) | optional ZeSolver |
| `solver_settings.py` | ACTIVE | 2 importers | settings contract |
| `_resources.py` | ACTIVE | 5 importers | importlib.resources |
| `lecropper.py` | ACTIVE | 1 importer (worker) | crop/altaz |
| `zequalityMT.py` | ACTIVE | 1 importer (worker) | quality |
| `zemosaic_time_utils.py` | ACTIVE | 2 importers | time helpers |
| `core/path_helpers.py` | ACTIVE | 6 importers (worker, astrometry, filter_qt, gui_qt, filter, zequalityMT) | shared helpers |
| `core/robust_rejection.py` | ACTIVE | WSC impl; imported by align_stack + align_stack_gpu | science-critical (WSC) |
| `core/tk_safe.py` | DORMANT BUT REACHABLE | imported only by legacy GUI + legacy filter | Tk-only helper |
| `core/cuda_utils.py` | SUSPECTED DEAD | defines `enforce_nvidia_gpu`; **no importer found** in src or spec hiddenimports | verify before any deletion |
| `cuda_utils.py` | ACTIVE (module) / DORMANT functions | `CUPY_AVAILABLE` imported by `_app.py`; but `gpu_supported()` / `enforce_nvidia_gpu()` have **no callers** anywhere (src+tests) | `CUPY_AVAILABLE` live; the two fns are SUSPECTED DEAD |
| `zemosaic_gui.py` (legacy Tk) | DORMANT BUT REACHABLE | **no importer in src**; NOT in spec hiddenimports; `--tk-gui` rejected at `_app._determine_backend`; still a full Tk GUI + worker target | legacy UI; verify before delete |
| `zemosaic_filter_gui.py` (legacy Tk) | ACTIVE | `launch_filter_interface` imported dynamically by worker (`:25615`,`:31844`) + legacy GUI; three helpers imported by Qt filter | helper extraction candidates |

`build/` directory (containing `build/lib/zemosaic/`) is **untracked and gitignored**:
`git ls-files build/` returns 0 entries and `.gitignore:188` ignores `build/`. It is a
local, regenerable build artifact (24 MB), **not** part of source, git history, or
packaging input. Do not delete it without evidence, and do not infer packaging behavior
from its presence.

## 9. Phase 4.5 support/reachability (explicit)

**Conclusion: DORMANT BUT REACHABLE (programmatic), DISABLED on official Qt path.**

- Qt GUI forces `inter_master_merge_enable=False` (`zemosaic_gui_qt.py:4167`) — a Task-O
  regression guard; legacy GUI also forces it (`zemosaic_gui.py:370`).
- Config default is False (`zemosaic_config.py:200`).
- Worker still executes the full Phase 4.5 block when `phase45_options["enable"]` is true,
  reachable only via config cache override or direct programmatic call.
- Backend is CPU-only (see §4.4); alpha-weighted early branch is a distinct path.
- `_run_shared_phase45_phase5_pipeline` runs Phase 5 **even when 4.5 is disabled** —
  the helper must never be deleted under the assumption 4.5 is obsolete.

## 10. Qt/Tk dependency conclusion

- Official runtime is Qt-only (`_app._determine_backend` strips `--tk-gui`; `pyproject`
  only has a `tools` extra for PyQt5 legacy viewer).
- Tk is still **transitively present**: legacy `zemosaic_gui.py`/`zemosaic_filter_gui.py`
  import `tkinter` inside functions; `core/tk_safe.py` imports `tkinter` at module top; the
  worker dynamically imports `launch_filter_interface` (which can reach Tk).
- The CI guard `.github/workflows/no-tk-on-official-path.yml` blocks only **direct**
  `import tkinter` in three official modules; it is **not** proof of no transitive/fallback
  Tk dependency. Do not equate the CI guard with "Tk is gone".

## 11. Scientific anomaly ledger (preserve, do not fix)

- **SCI-01 (renamed) — Grid CPU vs Grid GPU-core winsorized-sigma divergence.**
  Grid CPU uses the established `_reject_outliers_winsorized_sigma_clip`
  (`grid_mode.py:1950-1957`) — full winsorize-then-clip. Grid GPU via `stack_core` uses a
  **simplified median/σ clip (not real winsorization)** for `winsorized_sigma_clip`
  (`zemosaic_stack_core.py:320-327`), for both backends. This is the genuine structural
  divergence; numeric impact **not measured** in R0.
- **SCI-02 — kappa-sigma is NOT divergent on GPU.** In `stack_core` (`zemosaic_stack_core.py:291-317`),
  when the imported helper is available, `kappa_sigma` on GPU does `cp.asnumpy` then calls
  the **same established `_reject_outliers_kappa_sigma`**; CPU calls it directly. The
  simplified median/σ clip runs only when the helper import failed, for **both** backends.
  (Corrected from the R0 revision, which wrongly attributed a GPU-only simplified path.)
- **SCI-03 — linear_fit placeholder.** `stack_core` has a `linear_fit` normalization
  placeholder (`:294`) that just does median subtraction. Grid GPU normalizes *upstream*
  (`_normalize_patches_gpu`, `:1853`) then passes `normalize_method='none'` to `stack_core`
  (`grid_mode.py:2012`). Do not attribute the core placeholder to Grid GPU without proof.
- **SCI-04 — masks/weights Grid CPU vs GPU.** Grid CPU masks non-positive weights before
  rejection; GPU core receives `weights` directly. All-invalid / `winsor_limits` aliases
  not fully characterized. UNKNOWN.
- **SCI-05 — Phase 4.5 alpha-weighted branch vs configured rejection.** When weight maps
  exist, the merge takes a direct weighted-mean path (bypassing reject/combine); otherwise
  it uses `stack_cfg["reject_algo"]`. `inter_master_stack_method` is passed but not
  consumed. Recorded as architectural debt (§4.4), not fixed.
- **SCI-07 — Phase 4.5 affine/micro-align helpers ABSENT at BASE.**
  `zemosaic_align_stack` defines no `estimate_affine_photometry`, `apply_affine_photometry`,
  or `micro_align_stack` (repo-wide search finds only call sites; runtime `hasattr` is
  `False False False`). The availability gates at `zemosaic_worker.py:7412-7414` therefore
  disable the micro-align branch (`:8034`), intra-group affine photometry (`:7701-7705`,
  `:7766`, `:8084`, `:8095`), legacy affine normalization (`:8151-8154`), and the global-affine
  inter-super branch (`:8865`) — all DORMANT/UNREACHABLE at BASE. NOT PROVEN DEAD
  (future helper injection is not excluded). The only active Phase 4.5 normalization is the
  pre-stack `linear_fit`/`sky_mean` block (`:8202-8290`) and the post-stack inter-super
  gain-only block (`:8622-8832`).
- **SCI-06 — Classic/SDS/Phase 4.5 low-N / chunking variants** not inventoried beyond the
  stacking maps. UNKNOWN.

## 12. Test witness inventory

All run with isolated `HOME=/tmp/zm-r0-home`, `XDG_CONFIG_HOME`/`XDG_CACHE_HOME=/tmp/zm-r0-xdg`,
`QT_QPA_PLATFORM=offscreen`, `MPLBACKEND=Agg`, using `.venv/bin/python -m pytest -q -ra`.
Exact per-file results (reproduced):

| File | Result | Duration | Notes |
| --- | --- | --- | --- |
| `test_packaging.py` | 17 passed | 19.66s | gui-script entry, namespace, CWD, resources, config migration, CPU-only |
| `test_phase3_adaptive_invariants.py` | 22 passed, **11 skipped** | 4.37s | skips = flat `importorskip("zemosaic_worker")` (see §13) |
| `test_grid_mode_dbe.py` | 4 passed | 3.08s | DBE + star protection only |
| `test_grid_mode_stack_plan_paths.py` | 1 passed | 3.76s | CSV path resolution only |
| `test_solver_port.py` | 12 passed | 0.03s | solver boundary contracts |
| `test_solver_port_integration.py` | 10 passed | 3.71s | imports `run_hierarchical_mosaic_classic_legacy` |
| `test_zesolver_adapter.py` | 13 passed | 0.06s | ZeSolver adapter contract |
| `test_zesolver_hardening.py` | 24 passed | 3.99s | solver hardening |
| `test_zesolver_filter_handoff.py` | 19 passed | 9.57s | filter→GUI→process→Phase 1, WCS in memory |
| `test_zesolver_filter_handoff_hg2.py` | 18 passed | 11.45s | WCS handoff preserved |
| `test_zesolver_filter_qt.py` | 24 passed | 6.16s | Qt filter cycle |
| `test_zesolver_filter_cancel_hg2.py` | 4 passed, 4 warnings | 7.08s | cancel/close during solve |
| `test_zesoftware_interop.py` | 10 passed | 4.63s | standalone interop |
| `test_zeanalyser_launch.py` | 24 passed | 5.22s | ZeAnalyser launch contract |
| `test_cupy_platform_guard.py` | 2 passed | 0.04s | optional-GPU guard |
| `test_version_gpu.py` | **0 collected** (diagnostic script) | 0.05s | **no `test_*` fns**; run as script prints CUDA versions |
| `test_phase5_vram_budget.py` | 2 passed | 3.07s | Phase 5 VRAM budget |
| `test_resource_telemetry.py` | 3 passed | 2.95s | telemetry |

Totals: **209 passed, 11 skipped, 0 failed, 0 xfailed, 4 warnings** across the targeted
set. `test_version_gpu.py` is a diagnostic script (collects 0 items) — a gap worth noting:
it is named like a test but asserts nothing (recorded as TEST-02 in todo.md).

## 13. Phase-3 `importorskip` audit (TEST-01)

`tests/test_phase3_adaptive_invariants.py` contains **11** flat
`pytest.importorskip("zemosaic_worker")` uses (lines 311, 445, 566, 579, 592, 608, 620,
649, 659, 668, 683) plus 2 flat `parallel_utils` (312, 446) and module-level
`zas = pytest.importorskip("zemosaic.zemosaic_align_stack")` (16, correctly namespaced).

Flat `zemosaic_worker` / `parallel_utils` are **not importable** as top-level modules
(verified: `ModuleNotFoundError: No module named 'zemosaic_worker'`), and
`tests/test_packaging.py` (NamespaceTests) forbids flat aliases. Consequence, observed at
runtime: **22 passed, 11 skipped** — the 11 skips are exactly the 11 flat `zemosaic_worker`
importorskips. Those tests (worker-source text invariants) **silently do not run**. This
was **not repaired** in R0 (per mission: establish skip, do not fix namespace).

## 14. Candidate PROVEN DEAD list

**None.** No module or path meets the PROVEN DEAD bar (no supported caller + no import
dependency + no Qt/CLI/package/API path + no dynamic import + no packaging/resource
dependency + no relevant test contract + no compatibility need). Closest candidates are
`SUSPECTED DEAD` (not proven):

- `core/cuda_utils.py` `enforce_nvidia_gpu` — no importer found.
- `cuda_utils.py` `gpu_supported()` / `enforce_nvidia_gpu()` — no callers (but `CUPY_AVAILABLE`
  from the same module is live).

## 15. Recommended next order

1. **Characterization-test missions first** (missing witnesses, before any deletion):
   - Dispatch propagation witness: GUI param values → `run_hierarchical_mosaic` effective
     args (exercise the rename/suffix/drop path).
   - Cache/resume + low-N/all-invalid stacking witnesses.
   - A real (tiny) worker-process spawn witness to pin `spawn`/picklability/import order.
   - Fix TEST-01 witness (migrate the 11 flat importorskips to namespaced imports) as its
     own bounded test-only mission — do **not** add a product flat alias.
2. **Then bounded R1 deletions** only for any unit that graduates to PROVEN DEAD after the
   missing witnesses; each with acceptance criteria + Nono review. If none graduate, R1
   ends with no deletion.
3. **Then bounded R2 extractions** (candidate order): pure shared filter helpers
   (`_merge_small_groups`, `_split_group_by_orientation`, `_circular_dispersion_deg`) then
   small transversal responsibilities — not wholesale Classic/SDS moves.
4. **R3 stacking-contracts map** (preliminary version already started, see
   `STACKING_CONTRACTS_R3.md`) must be completed *before* R2.
