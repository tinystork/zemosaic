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
from pathlib import Path
from typing import List, Optional, Tuple

try:  # Tkinter is optional for headless/CLI usage
    import tkinter.filedialog as fd
    import tkinter.messagebox as mb
except Exception:  # pragma: no cover - depends on OS packages
    fd = None
    mb = None

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
    "language": "en",
    "qt_theme_mode": "system",
    "input_dir": "",
    "output_dir": "",
    "num_processing_workers": -1, # -1 pour auto
    "stacking_normalize_method": "linear_fit",
    "stacking_weighting_method": "noise_variance",
    "stacking_rejection_algorithm": "winsorized_sigma_clip", 
    "stacking_kappa_low": 3.0,
    "stacking_kappa_high": 3.0,
    "stacking_winsor_limits": "0.05,0.05", # String, sera parsé
    "stacking_final_combine_method": "mean",
    "poststack_equalize_rgb": True,
    "apply_radial_weight": False,
    "radial_feather_fraction": 0.8,
    "radial_shape_power": 2.0,
    "use_gpu_phase5": True,
    "gpu_id_phase5": 0,
    "gpu_selector": "",
    "final_assembly_method": "reproject_coadd", # Options: "reproject_coadd", "incremental",
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
    "coadd_use_memmap": True,
    "coadd_memmap_dir": "",
    "coadd_cleanup_memmap": True,
    # Cache retention policy for Phase 1 preprocessed .npy files.
    # Allowed values: "run_end", "per_tile", "keep".
    "cache_retention": "run_end",
    "assembly_process_workers": 0,  # Worker count for final assembly (both methods)
    "auto_limit_frames_per_master_tile": True,
    "winsor_worker_limit": 10,
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
    "intertile_global_recenter": True,
    "force_resolve_existing_wcs": False,
    "intertile_recenter_clip": [0.85, 1.18],
    "use_auto_intertile": False,
    "match_background_for_final": True,
    "incremental_feather_parity": False,
    # Early GUI filter option (Phase 0 header-only scan)
    "enable_early_filter": True,
    # --- CLES POUR LE ROGNAGE DES MASTER TUILES ---
    "apply_master_tile_crop": True,       # Désactivé par défaut
    "master_tile_crop_percent": 3.0,     # Pourcentage par côté si activé (ex: 10%)
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
    # --- Alt-Az cleanup (lecropper altZ) ---
    "altaz_cleanup_enabled": False,
    "altaz_margin_percent": 5.0,   # UI: "AltAz margin %"
    "altaz_decay": 0.15,           # UI: "AltAz decay"
    "altaz_nanize": True,          # UI: "Alt-Az → NaN"
    # --- Alt-Az Alpha export ---
    "altaz_export_alpha_fits": True,
    "altaz_export_alpha_sidecar": False,
    "altaz_alpha_sidecar_format": "png",
    # --- Qualité avancée ---
    "quality_crop_min_run": 2,     # UI: "min run"
    "crop_follow_signal": True,
    # --- FIN CLES POUR LE ROGNAGE ---
}


def _home_path(*parts: str) -> str:
    """Safely join parts under the current user's home directory."""
    try:
        return str(Path.home().joinpath(*parts))
    except Exception:
        return ""


def _resolve_astap_executable(candidate: Optional[str]) -> Optional[str]:
    """Expand and validate a potential ASTAP executable path."""
    if not candidate:
        return None
    expanded = os.path.normpath(os.path.expanduser(candidate.strip()))
    if os.path.isfile(expanded):
        return expanded
    if expanded.lower().endswith(".app") and os.path.isdir(expanded):
        app_exec = os.path.join(expanded, "Contents", "MacOS", "astap")
        if os.path.isfile(app_exec):
            return os.path.normpath(app_exec)
    if os.path.isdir(expanded):
        exe_name = "astap.exe" if IS_WINDOWS else "astap"
        nested = os.path.join(expanded, exe_name)
        if os.path.isfile(nested):
            return os.path.normpath(nested)
    resolved = shutil.which(expanded) or shutil.which(os.path.basename(expanded))
    if resolved and os.path.isfile(resolved):
        return os.path.normpath(resolved)
    return None


def _resolve_astap_data_dir(candidate: Optional[str]) -> Optional[str]:
    """Expand and validate a potential ASTAP data directory path."""
    if not candidate:
        return None
    expanded = os.path.normpath(os.path.expanduser(candidate.strip()))
    if os.path.isdir(expanded):
        return expanded
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
        program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        for base in (program_files, program_files_x86):
            if base:
                candidates.append(os.path.join(base, "astap", exe_name))
        candidates.extend(
            [
                rf"C:\ASTAP\{exe_name}",
                rf"C:\astap\{exe_name}",
            ]
        )
    elif IS_MAC:
        candidates.extend(
            [
                "/Applications/ASTAP.app",
                "/Applications/ASTAP/astap",
                "/Applications/ASTAP.app/Contents/MacOS/astap",
                _home_path("Applications", "ASTAP.app"),
                _home_path("Applications", "ASTAP.app", "Contents", "MacOS", "astap"),
            ]
        )
    else:
        candidates.extend(
            [
                f"/usr/bin/{exe_name}",
                f"/usr/local/bin/{exe_name}",
                f"/opt/astap/{exe_name}",
                f"/opt/astap/bin/{exe_name}",
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
        exe_dir = os.path.dirname(astap_exe_path)
        if exe_dir:
            candidates.append(exe_dir)
            candidates.append(os.path.join(exe_dir, "data"))
            if IS_MAC:
                # Bundle Resources folder
                try:
                    bundle_root = Path(astap_exe_path).resolve().parents[1]
                    candidates.append(str(bundle_root / "Resources"))
                except IndexError:
                    pass

    if IS_WINDOWS:
        program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        for base in (program_files, program_files_x86):
            if base:
                candidates.append(os.path.join(base, "astap"))
    elif IS_MAC:
        candidates.extend(
            [
                "/Applications/ASTAP.app/Contents/Resources",
                _home_path("Applications", "ASTAP.app", "Contents", "Resources"),
                _home_path("Library", "Application Support", "ASTAP"),
                "/Library/Application Support/ASTAP",
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


def get_config_path():
    """
    Retourne le chemin du fichier de configuration.
    Le fichier sera situé dans le même dossier que ce script (zemosaic_config.py).
    """
    # __file__ est le chemin du script actuel (zemosaic_config.py)
    # os.path.dirname(__file__) donne le dossier contenant ce script
    # os.path.abspath() assure que le chemin est absolu
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, CONFIG_FILE_NAME)

def load_config():
    config_path = get_config_path()
    current_config = DEFAULT_CONFIG.copy()
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f: # Spécifier encoding
                loaded_config = json.load(f)
                for key, default_value in DEFAULT_CONFIG.items():
                    current_config[key] = loaded_config.get(key, default_value)
                # Gérer les clés obsolètes ou nouvelles non présentes dans DEFAULT_CONFIG
                # Par exemple, on pourrait choisir de ne garder que les clés de DEFAULT_CONFIG
                # ou d'ajouter les nouvelles clés de loaded_config qui ne sont pas dans DEFAULT_CONFIG.
                # Pour l'instant, la boucle ci-dessus s'assure que toutes les clés de DEFAULT_CONFIG
                # sont présentes dans current_config, en prenant la valeur chargée si elle existe.
        except json.JSONDecodeError:
            # Utiliser mb (messagebox) si disponible, sinon print
            msg_title = "Config Error"
            msg_text = f"Error reading {config_path}. Using default configuration."
            try:
                if mb: mb.showwarning(msg_title, msg_text)
                else: print(f"WARNING: {msg_title} - {msg_text}")
            except Exception: print(f"WARNING: {msg_title} - {msg_text} (messagebox error)")
        except Exception as e:
            msg_title = "Config Error"
            msg_text = f"Unexpected error reading {config_path}: {e}. Using defaults."
            try:
                if mb: mb.showerror(msg_title, msg_text)
                else: print(f"ERROR: {msg_title} - {msg_text}")
            except Exception: print(f"ERROR: {msg_title} - {msg_text} (messagebox error)")
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
            current_config[key] = os.path.expanduser(value)
    _apply_astap_platform_defaults(current_config)

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

    return current_config

def save_config(config_data):
    config_path = get_config_path()
    try:
        # Avant de sauvegarder, s'assurer que config_data ne contient que les clés attendues
        # pour éviter d'écrire des clés temporaires ou obsolètes.
        # On ne garde que les clés qui sont dans DEFAULT_CONFIG.
        config_to_save = {}
        for key in DEFAULT_CONFIG.keys():
            if key in config_data:
                config_to_save[key] = config_data[key]
            # else: # Optionnel: si une clé par défaut manque dans config_data, la remettre
            #     config_to_save[key] = DEFAULT_CONFIG[key]


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


        with open(config_path, 'w', encoding='utf-8') as f: # Spécifier encoding
            json.dump(config_to_save, f, indent=4, ensure_ascii=False) # ensure_ascii=False pour les caractères non-ASCII
        print(f"Configuration sauvegardée vers {config_path}")
        return True
    except IOError as e:
        msg_title = "Config Error"
        msg_text = f"Unable to save configuration to {config_path}:\n{e}"
        try:
            if mb: mb.showerror(msg_title, msg_text)
            else: print(f"ERROR: {msg_title} - {msg_text}")
        except Exception: print(f"ERROR: {msg_title} - {msg_text} (messagebox error)")
        return False

# Les fonctions ask_and_set_... et get_... restent les mêmes,
# elles utiliseront le nouveau chemin via get_config_path().
# Assurez-vous que tkinter.filedialog (fd) est importé si vous l'utilisez dans ces fonctions.
# Par exemple :
# import tkinter.filedialog as fd # Au début du fichier si ce n'est pas déjà fait globalement
# ... (vos fonctions ask_and_set_astap_path, etc.)

def ask_and_set_astap_path(current_config):
    """Prompt the user for the ASTAP executable in a cross-platform friendly way."""
    if fd is None:
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

    astap_path = fd.askopenfilename(
        title="Sélectionner l'exécutable ASTAP",
        filetypes=filetypes,
    )
    if IS_MAC and not astap_path:
        # Allow selecting the .app bundle if the binary is hidden.
        astap_path = fd.askdirectory(title="Sélectionner l'application ASTAP (.app)")

    resolved_path = _resolve_astap_executable(astap_path)
    if not resolved_path:
        if astap_path:
            message = f"Le chemin sélectionné ne semble pas contenir l'exécutable ASTAP: {astap_path}"
            if mb:
                mb.showwarning("Chemin ASTAP invalide", message, parent=None)
            else:
                print(f"WARNING: {message}")
        return current_config.get("astap_executable_path", "")

    current_config["astap_executable_path"] = resolved_path
    if save_config(current_config):
        msg = f"Chemin ASTAP défini à : {resolved_path}"
        if mb:
            mb.showinfo("Chemin ASTAP Défini", msg, parent=None)
        else:
            print(msg)
    return resolved_path


def ask_and_set_astap_data_dir_path(current_config):
    """Prompt for the ASTAP star database directory."""
    if fd is None:
        print("ASTAP data directory prompt indisponible (tkinter non installé).")
        return current_config.get("astap_data_directory_path", "")

    astap_data_dir = fd.askdirectory(
        title="Sélectionner le dossier de données ASTAP (contenant G17, H17, etc.)"
    )
    resolved_dir = _resolve_astap_data_dir(astap_data_dir)
    if not resolved_dir:
        if astap_data_dir:
            message = f"Le dossier sélectionné n'existe pas ou ne contient pas les catalogues ASTAP: {astap_data_dir}"
            if mb:
                mb.showwarning("Dossier ASTAP invalide", message, parent=None)
            else:
                print(f"WARNING: {message}")
        return current_config.get("astap_data_directory_path", "")

    current_config["astap_data_directory_path"] = resolved_dir
    if save_config(current_config):
        msg = f"Dossier de données ASTAP défini à : {resolved_dir}"
        if mb:
            mb.showinfo("Dossier Données ASTAP Défini", msg, parent=None)
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
