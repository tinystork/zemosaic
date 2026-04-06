"""
╔═══════════════════════════════════════════════════════════════════════════════════╗
║ ZeMosaic / ZeSeestarStacker Project                                               ║
║                                                                                   ║
║ Auteur  : Tinystork, seigneur des couteaux à beurre (aka Tristan Nauleau)         ║
║ Partenaire : J.A.R.V.I.S. (/ˈdʒɑːrvɪs/) — Just a Rather Very Intelligent System   ║
║              (aka ChatGPT, Grand Maître du ciselage de code)                      ║
║                                                                                   ║
║ Licence : GNU General Public License v3.0 (GPL-3.0)                               ║
║                                                                                   ║
║ Description :                                                                     ║
║   Ce programme a été forgé à la lueur des pixels et de la caféine,                ║
║   dans le but noble de transformer des nuages de photons en art                   ║
║   astronomique. Si vous l’utilisez, pensez à dire “merci”,                        ║
║   à lever les yeux vers le ciel, ou à citer Tinystork et J.A.R.V.I.S.             ║
║   (le karma des développeurs en dépend).                                          ║
║                                                                                   ║
║ Avertissement :                                                                   ║
║   Aucune IA ni aucun couteau à beurre n’a été blessé durant le                    ║
║   développement de ce code.                                                       ║
╚═══════════════════════════════════════════════════════════════════════════════════╝


╔═══════════════════════════════════════════════════════════════════════════════════╗
║ ZeMosaic / ZeSeestarStacker Project                                               ║
║                                                                                   ║
║ Author  : Tinystork, Lord of the Butter Knives (aka Tristan Nauleau)              ║
║ Partner : J.A.R.V.I.S. (/ˈdʒɑːrvɪs/) — Just a Rather Very Intelligent System      ║
║           (aka ChatGPT, Grand Master of Code Chiseling)                           ║
║                                                                                   ║
║ License : GNU General Public License v3.0 (GPL-3.0)                               ║
║                                                                                   ║
║ Description:                                                                      ║
║   This program was forged under the sacred light of pixels and                    ║
║   caffeine, with the noble intent of turning clouds of photons into               ║
║   astronomical art. If you use it, please consider saying “thanks,”               ║
║   gazing at the stars, or crediting Tinystork and J.A.R.V.I.S. —                  ║
║   developer karma depends on it.                                                  ║
║                                                                                   ║
║ Disclaimer:                                                                       ║
║   No AIs or butter knives were harmed in the making of this code.                 ║
╚═══════════════════════════════════════════════════════════════════════════════════╝
"""

# zemosaic_config.py
import json
import os
import platform
import shutil
import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from zemosaic_utils import ensure_user_config_dir, get_user_config_dir
except Exception:  # pragma: no cover - fallback when utils unavailable
    def get_user_config_dir() -> Path:
        return Path.home() / "ZeMosaic"

    def ensure_user_config_dir() -> Path:
        path = get_user_config_dir()
        path.mkdir(parents=True, exist_ok=True)
        return path

fd = None
mb = None


def _ensure_tk_dialogs_loaded() -> tuple[object | None, object | None]:
    """Load tkinter dialog helpers lazily for legacy interactive prompts only."""
    global fd, mb
    if fd is not None or mb is not None:
        return fd, mb
    try:  # pragma: no cover - optional dependency
        import importlib
        _fd = importlib.import_module("tkinter.filedialog")
        _mb = importlib.import_module("tkinter.messagebox")
    except Exception:
        fd = None
        mb = None
        return fd, mb
    fd = _fd
    mb = _mb
    return fd, mb

CONFIG_FILE_NAME = "zemosaic_config.json"
SYSTEM_NAME = platform.system().lower()
IS_WINDOWS = SYSTEM_NAME == "windows"
IS_MAC = SYSTEM_NAME == "darwin"

DEFAULT_CONFIG = {
    "astap_executable_path": "",
    "astap_data_directory_path": "",
    "astap_default_search_radius": 3.0,
    "astap_default_downsample": 2,
    "astap_default_sensitivity": 100,
    "astap_max_instances": 1,
    "astap_drizzled_fallback_enabled": False,
    "language": "en",
    "preferred_gui_backend": "qt",  # official runtime is Qt-only
    "preferred_gui_backend_explicit": False,
    "qt_theme_mode": "system",
    "qt_main_window_geometry": None,
    "qt_filter_window_geometry": None,
    "qt_main_columns_state": None,
    "qt_main_left_state": None,
    "input_dir": "",
    "output_dir": "",
    "use_existing_master_tiles": False,
    "num_processing_workers": -1,  # -1 pour auto
    "grid_workers": 0,  # 0 = auto
    "stacking_normalize_method": "linear_fit",
    "stacking_weighting_method": "noise_variance",
    "stacking_rejection_algorithm": "winsorized_sigma_clip",
    "stacking_kappa_low": 3.0,
    "stacking_kappa_high": 3.0,
    "stacking_winsor_limits": "0.05,0.05",  # String, sera parsé
    "wsc_impl": "pixinsight",
    "stacking_final_combine_method": "mean",
    "poststack_equalize_rgb": False,
    "poststack_rgb_equalize_gain_clip": [0.95, 1.05],
    "poststack_rgb_equalize_bg_percentile": [5.0, 85.0],
    "poststack_rgb_equalize_min_samples": 5000,
    "poststack_rgb_equalize_min_coverage": 0.01,
    "final_mosaic_rgb_equalize_enabled": False,
    "final_mosaic_rgb_equalize_clip_enabled": True,
    "final_mosaic_rgb_equalize_gain_clip": [0.80, 1.25],
    # Power-user only (JSON): preview PNG stretch controls (no dedicated GUI fields)
    "preview_png_p_low": 1.0,
    "preview_png_p_high": 99.9,
    "preview_png_asinh_a": 12.0,
    "preview_png_max_dim": 3200,
    "preview_png_apply_wb": False,
    "visual_seam_heal_enabled": False,
    "visual_seam_heal_strength": 0.45,
    "visual_seam_heal_sigma_small": 24.0,
    "visual_seam_heal_sigma_large": 96.0,
    "visual_seam_heal_seam_sigma": 2.5,
    "visual_seam_heal_max_rel_delta": 0.08,
    # Optional preview-only multiscale seam heal (Phase 6 visual output)
    "visual_seam_heal_multiscale_enabled": False,
    "visual_seam_heal_multiscale_mid_gain": 0.35,
    "visual_seam_heal_multiscale_mid_sigma_scale": 0.60,
    "visual_seam_heal_multiscale_mid_rel_scale": 0.70,
    "existing_master_tiles_rgb_balance_prephase5": True,
    "existing_master_tiles_rgb_balance_gain_clip": [0.90, 1.10],
    "existing_master_tiles_rgb_balance_min_pixels": 5000,
    # Existing-master anchor photometry (post-refactor safeguard)
    "existing_master_tiles_anchor_photometry_enabled": True,
    "existing_master_tiles_anchor_gain_clip": [0.90, 1.10],
    "existing_master_tiles_final_rgb_equalize_gain_clip": [0.90, 1.10],
    "sds_enable_final_rgb_equalize": False,
    "sds_final_rgb_equalize_gain_clip": [0.95, 1.05],
    "sds_enable_final_black_point_equalize": False,
    "apply_radial_weight": False,
    "radial_feather_fraction": 0.8,
    "radial_shape_power": 2.0,
    "batch_overlap_pct": 40,
    "allow_batch_duplication": True,
    "min_safe_stack": 3,
    "target_stack": 5,
    "use_gpu_phase5": True,
    "use_gpu_grid": True,
    "gpu_id_phase5": 0,
    "gpu_selector": "",
    "gpu_hybrid_vram_guard": True,
    "phase5_chunk_auto": True,
    "phase5_chunk_mb": 128,
    "enable_tile_weighting": True,
    "tile_weight_mode": "n_frames",
    "tile_weight_v4_enabled": False,
    "tile_weight_v4_curve": "sqrt",
    "tile_weight_v4_strength": 1.0,
    "tile_weight_v4_min": 0.75,
    "tile_weight_v4_max": 1.35,
    "tile_weight_v4_residual_penalty_enabled": False,
    "tile_weight_v4_residual_penalty_strength": 0.35,
    "tile_weight_v4_temporal_penalty_enabled": False,
    "tile_weight_v4_temporal_penalty_strength": 0.20,
    "tile_weight_v4_temporal_penalty_hours": 6.0,
    "final_assembly_method": "reproject_coadd",  # Options: "reproject_coadd", "incremental",
    "auto_detect_seestar": True,
    "force_seestar_mode": False,
    "global_wcs_output_path": "global_mosaic_wcs.fits",
    "global_wcs_pixelscale_mode": "median",
    "global_wcs_padding_percent": 2.0,
    "global_wcs_res_override": None,
    "global_wcs_orientation": "north_up",
    "global_wcs_autocrop_enabled": True,
    "global_wcs_autocrop_margin_px": 128,
    "sds_mode_default": False,
    "sds_coverage_threshold": 0.92,
    "sds_min_batch_size": 5,
    "sds_target_batch_size": 10,
    "sds_min_coverage_keep": 0.4,
    "global_coadd_method": "kappa_sigma",
    "global_coadd_k": 2.0,
    "inter_master_merge_enable": False,
    "inter_master_overlap_threshold": 0.60,
    "inter_master_min_group_size": 2,
    "inter_master_stack_method": "winsor",
    "inter_master_memmap_policy": "auto",
    "inter_master_local_scale": "final",
    "inter_master_max_group": 64,
    "inter_master_photometry_intragroup": True,
    "inter_master_photometry_intersuper": True,
    "inter_master_photometry_clip_sigma": 3.0,
    "two_pass_coverage_renorm": False,
    "two_pass_cov_sigma_px": 50,
    "two_pass_cov_gain_clip": [0.85, 1.18],
    "solver_method": "ansvr",
    "astrometry_local_path": "",
    "astrometry_api_key": "",
    "save_final_as_uint16": False,
    "legacy_rgb_cube": False,
    "save_display_fits": False,
    "coadd_use_memmap": True,
    "coadd_memmap_dir": "",
    "coadd_cleanup_memmap": True,
    "cleanup_temp_artifacts": True,
    "enable_resource_telemetry": False,
    "resource_telemetry_interval_sec": 1.5,
    "resource_telemetry_log_to_csv": True,
    # Parallel tuning defaults calibrated from resource_telemetry.csv on a 32 GB RAM / 8 GB VRAM host.
    # These stay moderately conservative and lean on auto-tune to enforce safety checks at runtime.
    # Applied to both the global plan and the Phase 5 mosaic plan ("global_reproject")
    # derived from the final mosaic size and number of master tiles.
    "parallel_autotune_enabled": True,
    "parallel_target_cpu_load": 0.95,
    "parallel_target_ram_fraction": 0.9,
    "parallel_gpu_vram_fraction": 0.8,
    "parallel_max_cpu_workers": 0,  # 0 → no explicit cap beyond detected logical cores
    # Resume policy for classic legacy runs: "off", "auto", "force".
    "resume": "auto",
    # Cache retention policy for Phase 1 preprocessed .npy files.
    # Allowed values: "run_end", "per_tile", "keep".
    "cache_retention": "run_end",
    "assembly_process_workers": 0,  # Worker count for final assembly (both methods)
    "auto_limit_frames_per_master_tile": True,
    "winsor_worker_limit": 0,
    "winsor_max_frames_per_pass": 0,
    "winsor_auto_fallback_on_memory_error": True,
    "winsor_min_frames_per_pass": 4,
    "winsor_memmap_fallback": "auto",
    "winsor_split_strategy": "sequential",
    "max_raw_per_master_tile": 0,
    "center_out_normalization_p3": True,
    "center_out_anchor_mode": "auto_central_quality",
    "anchor_quality_probe_limit": 12,
    "anchor_quality_span_range": [0.02, 6.0],
    "anchor_quality_median_clip_sigma": 2.5,
    # Phase 3.9 post-stack anchor review
    "enable_poststack_anchor_review": True,
    "poststack_anchor_probe_limit": 8,
    "poststack_anchor_span_range": [0.004, 10.0],
    "poststack_anchor_median_clip_sigma": 3.5,
    "poststack_anchor_min_improvement": 0.12,
    "poststack_anchor_use_overlap_affine": True,
    "p3_center_sky_percentile": [25.0, 60.0],
    "p3_center_robust_clip_sigma": 2.5,
    "p3_center_preview_size": 256,
    "p3_center_min_overlap_fraction": 0.03,
    # --- Intertile photometric calibration options ---
    "intertile_photometric_match": True,
    "intertile_preview_size": 512,
    "intertile_overlap_min": 0.05,
    "intertile_sky_percentile": [30.0, 70.0],
    "intertile_robust_clip_sigma": 2.5,
    "intertile_prune_k": 8,
    "intertile_prune_weight_mode": "area",
    "intertile_offset_only_v1": False,
    "intertile_gain_offset_v2": False,
    "intertile_gain_prior_lambda": 0.02,
    "intertile_gain_clip": [0.90, 1.10],
    "intertile_offset_clip": [-2000.0, 2000.0],
    "intertile_pair_gain_clip": [0.5, 2.0],
    "intertile_pair_offset_abs_max": 5000.0,
    "intertile_max_irls_iters": 3,
    "intertile_enforce_requested_solver": False,
    "intertile_global_recenter": True,
    "force_resolve_existing_wcs": False,
    "intertile_recenter_clip": [0.85, 1.18],
    # Power-user only (JSON): blend affine corrections toward neutral to reduce seam visibility
    "intertile_affine_blend": 1.0,
    "use_auto_intertile": False,
    "match_background_for_final": True,
    "incremental_feather_parity": False,
    "final_mosaic_dbe_enabled": True,
    "final_mosaic_dbe_strength": "normal",
    # Final black-point equalization (power-user; can alter low/mid ADU chroma balance)
    "final_mosaic_black_point_equalize_enabled": False,
    "final_mosaic_black_point_percentile": 0.1,
    "final_mosaic_dbe_obj_k": 3.0,
    "final_mosaic_dbe_obj_dilate_px": 3,
    "final_mosaic_dbe_sample_step": 24,
    "final_mosaic_dbe_smoothing": 0.6,
    # Early GUI filter option (Phase 0 header-only scan)
    "enable_early_filter": True,
    # --- CLES POUR LE ROGNAGE DES MASTER TUILES ---
    "apply_master_tile_crop": True,  # Désactivé par défaut
    "master_tile_crop_percent": 3.0,  # Pourcentage par côté si activé (ex: 10%)
    "quality_crop_enabled": False,
    "quality_crop_band_px": 32,
    "quality_crop_k_sigma": 2.0,
    "quality_crop_margin_px": 8,
    # --- Master Tile Quality Gate (ZeQualityMT) ---
    "quality_gate_enabled": False,
    "quality_gate_threshold": 0.48,
    "quality_gate_edge_band_px": 64,
    "quality_gate_k_sigma": 2.5,
    "quality_gate_erode_px": 3,
    "quality_gate_move_rejects": True,
    # Existing-master-tiles pre-check gate (protect Phase 5 from catastrophic outliers)
    "existing_master_tiles_quality_gate_enabled": True,
    "existing_master_tiles_quality_gate_mode": "warn",  # warn|fail
    "existing_master_tiles_quality_gate_sigma_threshold": 8.0,
    "existing_master_tiles_quality_gate_ratio_threshold": 5000.0,
    "existing_master_tiles_quality_gate_min_valid_frac": 0.05,
    # --- Alt-Az cleanup (lecropper altZ) ---
    "altaz_cleanup_enabled": False,
    "altaz_margin_percent": 5.0,  # UI: "AltAz margin %"
    "altaz_decay": 0.15,  # UI: "AltAz decay"
    "altaz_nanize": True,  # UI: "Alt-Az → NaN"
    "altaz_alpha_soft_threshold": 0.001,
    "altaz_nanize_threshold": 0.001,
    # Policy for groups with unknown EQMODE markers:
    # - auto: treat UNKNOWN as ALT_AZ only for Seestar-like datasets
    # - on:   always treat UNKNOWN as ALT_AZ
    # - off:  never treat UNKNOWN as ALT_AZ
    "altaz_unknown_policy": "auto",
    # --- Alt-Az coverage-based mask (Master Tiles) ---
    "altaz_min_coverage_abs": 3.0,
    "altaz_min_coverage_frac": 0.5,
    "altaz_morph_open_px": 3,
    # --- Alt-Az strict overlap crop (Master Tiles) ---
    # Crops Alt-Az master tiles to the high-confidence core bbox from coverage-driven alpha.
    "altaz_strict_overlap_crop_enabled": True,
    "altaz_strict_overlap_crop_alpha_threshold": 0.02,
    "altaz_strict_overlap_crop_min_nonzero_frac": 0.08,
    "altaz_strict_overlap_crop_bbox_pad_px": 6,
    "altaz_strict_overlap_crop_min_size_px": 256,
    # --- Alt-Az Alpha export ---
    "altaz_export_alpha_fits": True,
    "altaz_export_alpha_sidecar": False,
    "altaz_alpha_sidecar_format": "png",
    # --- Qualité avancée ---
    "quality_crop_min_run": 2,  # UI: "min run"
    "crop_follow_signal": True,
    # --- FIN CLES POUR LE ROGNAGE ---
}


def _home_path(*parts: str) -> str:
    """Safely join parts under the current user's home directory."""
    try:
        return str(Path.home().joinpath(*parts))
    except Exception:
        return ""


def _expand_user_path(candidate: Optional[str]) -> Optional[Path]:
    """Return a ``Path`` expanded relative to the user's home directory."""
    if not candidate:
        return None
    try:
        return Path(candidate).expanduser()
    except Exception:
        return None


def _resolve_astap_executable(candidate: Optional[str]) -> Optional[str]:
    """Expand and validate a potential ASTAP executable path."""
    expanded = _expand_user_path(candidate.strip() if candidate else None)
    if expanded is None:
        return None
    if expanded.is_file():
        return str(expanded)
    if expanded.suffix.lower() == ".app" and expanded.is_dir():
        app_exec = expanded / "Contents" / "MacOS" / "astap"
        if app_exec.is_file():
            return str(app_exec)
    if expanded.is_dir():
        exe_name = "astap.exe" if IS_WINDOWS else "astap"
        nested = expanded / exe_name
        if nested.is_file():
            return str(nested)
    resolved = shutil.which(str(expanded)) or shutil.which(expanded.name)
    if resolved and Path(resolved).is_file():
        return str(Path(resolved))
    return None


def _resolve_astap_data_dir(candidate: Optional[str]) -> Optional[str]:
    """Expand and validate a potential ASTAP data directory path."""
    expanded = _expand_user_path(candidate.strip() if candidate else None)
    if expanded and expanded.is_dir():
        return str(expanded)
    return None


def _candidate_astap_binary_paths() -> List[str]:
    exe_name = "astap.exe" if IS_WINDOWS else "astap"
    candidates: list[str] = []
    # Environment overrides first
    for env_key in ("ASTAP_EXE", "ASTAP_BIN", "ASTAP_PATH"):
        env_val = os.environ.get(env_key)
        if env_val:
            candidates.append(env_val)

    if IS_WINDOWS:
        program_files = os.environ.get("ProgramFiles")
        program_files_x86 = os.environ.get("ProgramFiles(x86)")
        for base in filter(None, (program_files, program_files_x86)):
            candidates.append(str(Path(base).expanduser() / "astap" / exe_name))
        candidates.extend(
            [
                str(Path("C:/ASTAP") / exe_name),
                str(Path("C:/astap") / exe_name),
            ]
        )
    elif IS_MAC:
        candidates.extend(
            [
                str(Path("/Applications/ASTAP.app")),
                str(Path("/Applications/ASTAP/astap")),
                str(Path("/Applications/ASTAP.app/Contents/MacOS/astap")),
                _home_path("Applications", "ASTAP.app"),
                _home_path("Applications", "ASTAP.app", "Contents", "MacOS", "astap"),
            ]
        )
    else:
        candidates.extend(
            [
                str(Path("/usr/bin") / exe_name),
                str(Path("/usr/local/bin") / exe_name),
                str(Path("/opt/astap") / exe_name),
                str(Path("/opt/astap/bin") / exe_name),
                _home_path(".local", "bin", exe_name),
                _home_path("astap", exe_name),
            ]
        )

    # Allow PATH lookups as a last resort
    candidates.append(exe_name)
    return [c for c in candidates if c]


def _candidate_astap_data_dirs(astap_exe_path: Optional[str]) -> List[str]:
    candidates: list[str] = []
    for env_key in ("ASTAP_DATA_DIR", "ASTAP_STAR_DB", "ASTAP_DATABASE"):
        env_val = os.environ.get(env_key)
        if env_val:
            candidates.append(env_val)

    if astap_exe_path:
        exe_path_obj = Path(astap_exe_path)
        exe_dir = exe_path_obj.parent
        if exe_dir:
            candidates.append(str(exe_dir))
            candidates.append(str(exe_dir / "data"))
            if IS_MAC:
                # Bundle Resources folder
                try:
                    bundle_root = exe_path_obj.resolve().parents[1]
                    candidates.append(str(bundle_root / "Resources"))
                except IndexError:
                    pass

    if IS_WINDOWS:
        program_files = os.environ.get("ProgramFiles")
        program_files_x86 = os.environ.get("ProgramFiles(x86)")
        for base in filter(None, (program_files, program_files_x86)):
            candidates.append(str(Path(base).expanduser() / "astap"))
    elif IS_MAC:
        candidates.extend(
            [
                str(Path("/Applications/ASTAP.app/Contents/Resources")),
                _home_path("Applications", "ASTAP.app", "Contents", "Resources"),
                _home_path("Library", "Application Support", "ASTAP"),
                str(Path("/Library/Application Support/ASTAP")),
            ]
        )
    else:
        candidates.extend(
            [
                "/usr/share/astap",
                "/usr/local/share/astap",
                "/opt/astap",
                "/opt/astap/share",
                _home_path(".local", "share", "astap"),
            ]
        )
    return [c for c in candidates if c]


def _auto_find_astap_executable() -> Optional[str]:
    seen: set[str] = set()
    for candidate in _candidate_astap_binary_paths():
        resolved = _resolve_astap_executable(candidate)
        if resolved and resolved not in seen:
            return resolved
        if resolved:
            seen.add(resolved)
    return None


def _auto_find_astap_data_dir(astap_exe_path: Optional[str]) -> Optional[str]:
    seen: set[str] = set()
    for candidate in _candidate_astap_data_dirs(astap_exe_path):
        resolved = _resolve_astap_data_dir(candidate)
        if resolved and resolved not in seen:
            return resolved
        if resolved:
            seen.add(resolved)
    return None


def detect_astap_installation() -> Tuple[Optional[str], Optional[str]]:
    """
    Public helper for callers that need to inspect the auto-detected ASTAP paths.
    Returns (exe_path, data_dir) when detection succeeds, or (None, None).
    """
    exe_path = _auto_find_astap_executable()
    data_dir = _auto_find_astap_data_dir(exe_path)
    return exe_path, data_dir


def _coerce_boolish(value: Any, default: Optional[bool] = False) -> Optional[bool]:
    """Convert diverse truthy/falsy representations to bool."""

    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "none", "auto"}:
            return default
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    return default


def _normalize_gpu_flags(config: dict) -> None:
    """Ensure GPU toggles are synchronised across legacy keys."""

    canonical = _coerce_boolish(config.get("use_gpu_phase5"), None)
    if canonical is None:
        for key in ("stack_use_gpu", "use_gpu_stack"):
            canonical = _coerce_boolish(config.get(key), None)
            if canonical is not None:
                break
    if canonical is None:
        canonical = False
    config["use_gpu_phase5"] = canonical
    for key in ("stack_use_gpu", "use_gpu_stack"):
        config[key] = _coerce_boolish(config.get(key), canonical)
    config["use_gpu_grid"] = _coerce_boolish(config.get("use_gpu_grid"), canonical)


def _apply_astap_platform_defaults(config: dict) -> bool:
    """
    Ensure ASTAP paths are usable on the current platform by auto-detecting
    well-known installation locations when the config value is empty or invalid.
    """
    updated = False

    current_exe = config.get("astap_executable_path", "")
    resolved_exe = _resolve_astap_executable(current_exe)
    if current_exe and not resolved_exe:
        config["astap_executable_path"] = ""
    if not resolved_exe:
        resolved_exe = _auto_find_astap_executable()
        if resolved_exe:
            config["astap_executable_path"] = resolved_exe
            updated = True
    else:
        config["astap_executable_path"] = resolved_exe

    current_data = config.get("astap_data_directory_path", "")
    resolved_data = _resolve_astap_data_dir(current_data)
    if current_data and not resolved_data:
        config["astap_data_directory_path"] = ""
    if not resolved_data:
        resolved_data = _auto_find_astap_data_dir(resolved_exe)
        if resolved_data:
            config["astap_data_directory_path"] = resolved_data
            updated = True
    else:
        config["astap_data_directory_path"] = resolved_data

    return updated


_SCRIPT_DIR = Path(__file__).resolve().parent


def get_config_path():
    """
    Retourne le chemin du fichier de configuration.
    Le fichier est stocké dans le répertoire du repo ZeMosaic.
    """
    return str(_SCRIPT_DIR / CONFIG_FILE_NAME)

def _sync_path_aliases(config_obj: dict) -> dict:
    """Keep legacy/new path keys synchronized.

    Source of truth: prefer *_dir keys (used by Qt GUI), fallback to *_folder.
    Always mirror both directions so external edits are visible in UI and vice versa.
    """
    if not isinstance(config_obj, dict):
        return config_obj

    def _norm(v):
        if v is None:
            return ""
        txt = str(v).strip()
        return txt

    input_dir = _norm(config_obj.get("input_dir", ""))
    input_folder = _norm(config_obj.get("input_folder", ""))
    output_dir = _norm(config_obj.get("output_dir", ""))
    output_folder = _norm(config_obj.get("output_folder", ""))

    canonical_input = input_dir or input_folder
    canonical_output = output_dir or output_folder

    config_obj["input_dir"] = canonical_input
    config_obj["input_folder"] = canonical_input
    config_obj["output_dir"] = canonical_output
    config_obj["output_folder"] = canonical_output

    return config_obj

def load_config():
    config_path = Path(get_config_path())
    try:
        logger.info("Loading ZeMosaic config from %s", config_path)
    except Exception:
        pass
    current_config = DEFAULT_CONFIG.copy()
    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as f:  # Spécifier encoding
                loaded_config = json.load(f)
                for key, default_value in DEFAULT_CONFIG.items():
                    current_config[key] = loaded_config.get(key, default_value)
                # Preserve power-user / hidden keys that are present in JSON but not
                # declared in DEFAULT_CONFIG so GUI round-trips never erase them.
                if isinstance(loaded_config, dict):
                    for key, value in loaded_config.items():
                        if key not in current_config:
                            current_config[key] = value
        except json.JSONDecodeError:
            msg_title = "Config Error"
            msg_text = f"Error reading {config_path}. Using default configuration."
            print(f"WARNING: {msg_title} - {msg_text}")
        except Exception as e:
            msg_title = "Config Error"
            msg_text = f"Unexpected error reading {config_path}: {e}. Using defaults."
            print(f"ERROR: {msg_title} - {msg_text}")
    # else:
        # print(f"Config file not found at {config_path}. Using default configuration.")
    current_config.setdefault("crop_follow_signal", False)
    current_config.setdefault("cache_retention", "run_end")
    for key in (
        "astap_executable_path",
        "astap_data_directory_path",
        "coadd_memmap_dir",
        "gpu_selector",
        "input_dir",
        "output_dir",
    ):
        value = current_config.get(key)
        if isinstance(value, str) and value:
            expanded = _expand_user_path(value)
            if expanded:
                current_config[key] = str(expanded)
    _apply_astap_platform_defaults(current_config)
    _normalize_gpu_flags(current_config)

    # Qt-only migration: backend selection is obsolete in official runtime.
    # Keep backward readability for legacy config files, but neutralize runtime state.
    legacy_backend = str(current_config.get("preferred_gui_backend", "") or "").strip().lower()
    if legacy_backend == "tk":
        current_config["preferred_gui_backend"] = "qt"
    elif legacy_backend != "qt":
        current_config["preferred_gui_backend"] = "qt"
    current_config["preferred_gui_backend_explicit"] = False

    def _env_flag(env_key: str, default: bool) -> bool:
        val = os.environ.get(env_key)
        if val is None:
            return default
        if val.strip().lower() in {"0", "false", "off", "no"}:
            return False
        return True

    current_config["altaz_export_alpha_fits"] = _env_flag(
        "ZEM_ALTALPHA_FITS",
        bool(current_config.get("altaz_export_alpha_fits", True)),
    )
    current_config["altaz_export_alpha_sidecar"] = _env_flag(
        "ZEM_ALTALPHA_SIDECAR",
        bool(current_config.get("altaz_export_alpha_sidecar", False)),
    )

    fmt_env = os.environ.get("ZEM_ALTALPHA_SIDECAR_FORMAT")
    fmt_cfg = current_config.get("altaz_alpha_sidecar_format", "png")
    fmt_val = fmt_env if fmt_env is not None else fmt_cfg
    if not isinstance(fmt_val, str):
        fmt_val = str(fmt_val or "png")
    fmt_val = fmt_val.strip().lower() or "png"
    if fmt_val in {"tif"}:
        fmt_val = "tiff"
    elif fmt_val not in {"png", "tiff"}:
        fmt_val = "png"
    current_config["altaz_alpha_sidecar_format"] = fmt_val

    _sync_path_aliases(current_config)
    return current_config

def save_config(config_data):
    _normalize_gpu_flags(config_data)
    # Qt-only official runtime: always persist neutral backend selection state.
    if isinstance(config_data, dict):
        config_data["preferred_gui_backend"] = "qt"
        config_data["preferred_gui_backend_explicit"] = False
    config_path = Path(get_config_path())
    config_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing_resume = None
        if config_path.exists():
            try:
                with config_path.open("r", encoding="utf-8") as existing_handle:
                    loaded_existing = json.load(existing_handle)
                if isinstance(loaded_existing, dict):
                    existing_resume = loaded_existing.get("resume")
            except Exception:
                existing_resume = None
        # Preserve unknown/power-user keys already present in JSON, then overlay
        # current config values. This guarantees GUI saves are non-destructive.
        config_to_save = {}
        existing_payload = None
        if config_path.exists():
            try:
                with config_path.open("r", encoding="utf-8") as existing_handle:
                    existing_payload = json.load(existing_handle)
            except Exception:
                existing_payload = None
        if isinstance(existing_payload, dict):
            config_to_save.update(existing_payload)

        if isinstance(config_data, dict):
            config_to_save.update(config_data)

        # Ensure known defaults always exist for forward compatibility.
        for key, default_value in DEFAULT_CONFIG.items():
            if key not in config_to_save:
                config_to_save[key] = default_value

        default_resume = DEFAULT_CONFIG.get("resume")
        if existing_resume is not None and isinstance(config_data, dict):
            resume_in_config = config_data.get("resume")
            if (
                "resume" not in config_data
                or (resume_in_config == default_resume and existing_resume != default_resume)
            ):
                config_to_save["resume"] = existing_resume


        # Si config_to_save est vide (si config_data n'avait aucune clé de DEFAULT_CONFIG),
        # on pourrait choisir de sauvegarder DEFAULT_CONFIG à la place, ou rien.
        # Pour l'instant, on sauvegarde ce qui a été filtré.
        # S'il est vide, cela pourrait indiquer un problème en amont.
        if not config_to_save and config_data: # Si config_data n'était pas vide mais qu'aucune clé n'a matché
            print(f"AVERT (save_config): Aucune clé de DEFAULT_CONFIG trouvée dans config_data. Sauvegarde de config_data tel quel.")
            config_to_save = config_data # Sauvegarder ce qu'on a reçu pour ne pas perdre d'info, mais c'est suspect
        elif not config_to_save and not config_data: # Si config_data était vide
             print(f"AVERT (save_config): config_data est vide, rien à sauvegarder pour {config_path}.")
             return False # Ne pas créer un fichier vide


        _sync_path_aliases(config_to_save)
        with config_path.open("w", encoding="utf-8") as f: # Spécifier encoding
            json.dump(config_to_save, f, indent=4, ensure_ascii=False) # ensure_ascii=False pour les caractères non-ASCII
        print(f"Configuration sauvegardée vers {config_path}")
        return True
    except IOError as e:
        msg_title = "Config Error"
        msg_text = f"Unable to save configuration to {config_path}:\n{e}"
        print(f"ERROR: {msg_title} - {msg_text}")
        return False

def ask_and_set_astap_path(current_config):
    """Prompt the user for the ASTAP executable in a cross-platform friendly way."""
    dialog_fd, dialog_mb = _ensure_tk_dialogs_loaded()
    if dialog_fd is None:
        print("ASTAP path prompt unavailable (tkinter non installé).")
        return current_config.get("astap_executable_path", "")

    if IS_WINDOWS:
        filetypes = (("Fichiers exécutables", "*.exe"), ("Tous les fichiers", "*.*"))
    elif IS_MAC:
        filetypes = (
            ("Applications ASTAP", "*.app"),
            ("Binaires", "astap"),
            ("Tous les fichiers", "*"),
        )
    else:
        filetypes = (("Binaires", "astap"), ("Tous les fichiers", "*"))

    astap_path = dialog_fd.askopenfilename(
        title="Sélectionner l'exécutable ASTAP",
        filetypes=filetypes,
    )
    if IS_MAC and not astap_path:
        # Allow selecting the .app bundle if the binary is hidden.
        astap_path = dialog_fd.askdirectory(title="Sélectionner l'application ASTAP (.app)")

    resolved_path = _resolve_astap_executable(astap_path)
    if not resolved_path:
        if astap_path:
            message = f"Le chemin sélectionné ne semble pas contenir l'exécutable ASTAP: {astap_path}"
            if dialog_mb:
                dialog_mb.showwarning("Chemin ASTAP invalide", message, parent=None)
            else:
                print(f"WARNING: {message}")
        return current_config.get("astap_executable_path", "")

    current_config["astap_executable_path"] = resolved_path
    if save_config(current_config):
        msg = f"Chemin ASTAP défini à : {resolved_path}"
        if dialog_mb:
            dialog_mb.showinfo("Chemin ASTAP Défini", msg, parent=None)
        else:
            print(msg)
    return resolved_path


def ask_and_set_astap_data_dir_path(current_config):
    """Prompt for the ASTAP star database directory."""
    dialog_fd, dialog_mb = _ensure_tk_dialogs_loaded()
    if dialog_fd is None:
        print("ASTAP data directory prompt indisponible (tkinter non installé).")
        return current_config.get("astap_data_directory_path", "")

    astap_data_dir = dialog_fd.askdirectory(
        title="Sélectionner le dossier de données ASTAP (contenant G17, H17, etc.)"
    )
    resolved_dir = _resolve_astap_data_dir(astap_data_dir)
    if not resolved_dir:
        if astap_data_dir:
            message = f"Le dossier sélectionné n'existe pas ou ne contient pas les catalogues ASTAP: {astap_data_dir}"
            if dialog_mb:
                dialog_mb.showwarning("Dossier ASTAP invalide", message, parent=None)
            else:
                print(f"WARNING: {message}")
        return current_config.get("astap_data_directory_path", "")

    current_config["astap_data_directory_path"] = resolved_dir
    if save_config(current_config):
        msg = f"Dossier de données ASTAP défini à : {resolved_dir}"
        if dialog_mb:
            dialog_mb.showinfo("Dossier Données ASTAP Défini", msg, parent=None)
        else:
            print(msg)
    return resolved_dir


def get_astap_executable_path():
    config = load_config()
    return config.get("astap_executable_path", "")

def get_astap_data_directory_path():
    config = load_config()
    return config.get("astap_data_directory_path", "") # Retourne une chaîne vide si non défini

def get_astap_default_search_radius():
    config = load_config()
    return config.get("astap_default_search_radius", DEFAULT_CONFIG["astap_default_search_radius"])

def get_astap_default_downsample():
    config = load_config()
    return config.get("astap_default_downsample", DEFAULT_CONFIG["astap_default_downsample"])

def get_astap_default_sensitivity():
    config = load_config()
    return config.get("astap_default_sensitivity", DEFAULT_CONFIG["astap_default_sensitivity"])

def get_astap_max_instances():
    """Return the configured ASTAP concurrency limit (>=1)."""
    config = load_config()
    value = config.get("astap_max_instances", DEFAULT_CONFIG.get("astap_max_instances", 1))
    try:
        return max(1, int(value))
    except Exception:
        return max(1, DEFAULT_CONFIG.get("astap_max_instances", 1))
