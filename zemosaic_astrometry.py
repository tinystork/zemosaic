# zemosaic_astrometry.py

import os
import numpy as np
import warnings
import time
import re
# import tempfile # Plus utilisé directement si on nettoie manuellement
import traceback
import subprocess
import shutil
import gc
import logging
import psutil
from concurrent.futures import ProcessPoolExecutor

import multiprocessing


logger = logging.getLogger("ZeMosaicAstrometry")
# ... (pas besoin de reconfigurer le logger ici s'il hérite du worker)

try:
    from astropy.io import fits
    from astropy.wcs import WCS as AstropyWCS, FITSFixedWarning 
    from astropy.utils.exceptions import AstropyWarning
    from astropy import units as u # Nécessaire pour _update_fits_header_with_wcs_za
    ASTROPY_AVAILABLE_ASTROMETRY = True
    warnings.filterwarnings('ignore', category=FITSFixedWarning)
    warnings.filterwarnings('ignore', category=AstropyWarning)
except ImportError:
    logger.error("Astropy non installée. Certaines fonctionnalités de zemosaic_astrometry seront limitées.")
    ASTROPY_AVAILABLE_ASTROMETRY = False
    class AstropyWCS: pass 
    class FITSFixedWarning(Warning): pass
    u = None


def _log_memory_usage(progress_callback: callable, context_message: str = ""):
    """Logue l'utilisation mémoire du processus courant."""
    if not progress_callback or not callable(progress_callback):
        return
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        rss_mb = mem_info.rss / (1024 * 1024)
        vms_mb = mem_info.vms / (1024 * 1024)

        virtual_mem = psutil.virtual_memory()
        available_ram_mb = virtual_mem.available / (1024 * 1024)
        total_ram_mb = virtual_mem.total / (1024 * 1024)
        percent_ram_used = virtual_mem.percent

        swap_mem = psutil.swap_memory()
        used_swap_mb = swap_mem.used / (1024 * 1024)
        total_swap_mb = swap_mem.total / (1024 * 1024)
        percent_swap_used = swap_mem.percent

        log_msg = (
            f"Memory Usage ({context_message}): "
            f"Proc RSS: {rss_mb:.1f}MB, VMS: {vms_mb:.1f}MB. "
            f"Sys RAM: Avail {available_ram_mb:.0f}MB / Total {total_ram_mb:.0f}MB ({percent_ram_used}% used). "
            f"Sys Swap: Used {used_swap_mb:.0f}MB / Total {total_swap_mb:.0f}MB ({percent_swap_used}% used)."
        )
        progress_callback(log_msg, None, "DEBUG")
    except Exception as e_mem_log:
        progress_callback(f"Erreur lors du logging mémoire ({context_message}): {e_mem_log}", None, "WARN")


def _run_astap_subprocess(cmd_list: list, cwd: str, timeout_sec: int):
    """Fonction exécutée dans un ProcessPoolExecutor pour lancer ASTAP."""
    return subprocess.run(
        cmd_list,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=timeout_sec,
        check=False,
        cwd=cwd,
    )


def _calculate_pixel_scale_from_header(header: fits.Header, progress_callback: callable = None) -> float | None:
    # ... (corps de la fonction inchangé, il semble correct)
    if not header:
        return None
    focal_len_mm = None
    pixel_size_um = None
    focal_keys = ['FOCALLEN', 'FOCAL', 'FLENGTH']
    for key in focal_keys:
        if key in header and isinstance(header[key], (int, float)) and header[key] > 0:
            focal_len_mm = float(header[key])
            if progress_callback: progress_callback(f"  ASTAP ScaleCalc: Trouvé {key}={focal_len_mm} mm", None, "DEBUG_DETAIL")
            break
    if focal_len_mm is None:
        if progress_callback: progress_callback("  ASTAP ScaleCalc: FOCALLEN non trouvée ou invalide dans le header.", None, "DEBUG_DETAIL")
        return None
    pix_size_keys = ['XPIXSZ', 'PIXSIZE', 'PIXELSIZE', 'PIXSCAL1', 'SCALE']
    for key in pix_size_keys:
        if key in header and isinstance(header[key], (int, float)) and header[key] > 0:
            if key.upper() == 'PIXSCAL1':
                unit_key = f"CUNIT{key[-1]}" if key[-1].isdigit() else None
                if unit_key and unit_key in header and str(header[unit_key]).lower() in ['arcsec', 'asec', '"']:
                    if progress_callback: progress_callback(f"  ASTAP ScaleCalc: Trouvé {key}={header[key]} arcsec/pix directement.", None, "DEBUG_DETAIL")
                    return float(header[key])
            pixel_size_um = float(header[key])
            if progress_callback: progress_callback(f"  ASTAP ScaleCalc: Trouvé {key}={pixel_size_um} µm", None, "DEBUG_DETAIL")
            break
    if pixel_size_um is None:
        if progress_callback: progress_callback("  ASTAP ScaleCalc: XPIXSZ (ou équivalent) non trouvé ou invalide.", None, "DEBUG_DETAIL")
        return None
    try:
        calculated_scale_arcsec_pix = (pixel_size_um / focal_len_mm) * 206.264806
        if progress_callback: progress_callback(f"  ASTAP ScaleCalc: Échelle calculée: {calculated_scale_arcsec_pix:.3f} arcsec/pix", None, "INFO_DETAIL")
        return calculated_scale_arcsec_pix
    except ZeroDivisionError:
        if progress_callback: progress_callback("  ASTAP ScaleCalc ERREUR: Division par zéro (FOCALLEN nulle ?).", None, "WARN")
        return None

def _parse_wcs_file_content_za(wcs_file_path, image_shape_hw, progress_callback=None):
    # ... (corps de la fonction inchangé, il semble correct)
    filename_log = os.path.basename(wcs_file_path)
    if progress_callback: progress_callback(f"  ASTAP WCS Parse: Tentative parsing '{filename_log}' pour shape {image_shape_hw}", None, "DEBUG_DETAIL")
    if not (os.path.exists(wcs_file_path) and os.path.getsize(wcs_file_path) > 0):
        if progress_callback: progress_callback(f"    ASTAP WCS Parse ERREUR: Fichier WCS '{filename_log}' non trouvé ou vide.", None, "WARN")
        return None
    if not ASTROPY_AVAILABLE_ASTROMETRY:
        if progress_callback: progress_callback("    ASTAP WCS Parse ERREUR: Astropy non disponible pour parser WCS.", None, "ERROR")
        return None
    try:
        with open(wcs_file_path, 'r', errors='replace') as f: wcs_text = f.read()
        wcs_hdr_from_text = fits.Header.fromstring(wcs_text.replace('\r\n', '\n').replace('\r', '\n'), sep='\n')
        if 'NAXIS1' not in wcs_hdr_from_text and image_shape_hw:
            wcs_hdr_from_text['NAXIS1'] = image_shape_hw[1]
        if 'NAXIS2' not in wcs_hdr_from_text and image_shape_hw:
            wcs_hdr_from_text['NAXIS2'] = image_shape_hw[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            wcs_obj = AstropyWCS(wcs_hdr_from_text, naxis=2, relax=True)
        if wcs_obj and wcs_obj.is_celestial:
            if image_shape_hw and image_shape_hw[0] > 0 and image_shape_hw[1] > 0:
                try:
                    wcs_obj.pixel_shape = (image_shape_hw[1], image_shape_hw[0])
                except Exception as e_ps_parse:
                    if progress_callback: progress_callback(f"    ASTAP WCS Parse AVERT: Échec set pixel_shape sur WCS parsé: {e_ps_parse}", None, "WARN")
            if progress_callback: progress_callback(f"    ASTAP WCS Parse: Objet WCS parsé avec succès depuis '{filename_log}'.", None, "DEBUG_DETAIL")
            return wcs_obj
        else:
            if progress_callback: progress_callback(f"    ASTAP WCS Parse ERREUR: Échec création WCS valide/céleste depuis '{filename_log}'.", None, "WARN")
            return None
    except Exception as e:
        if progress_callback: progress_callback(f"    ASTAP WCS Parse ERREUR: Exception lors du parsing WCS '{filename_log}': {e}", None, "ERROR")
        logger.error(f"Erreur parsing WCS '{wcs_file_path}': {e}", exc_info=True)
        return None


def _update_fits_header_with_wcs_za(fits_header_to_update: fits.Header, 
                                   wcs_object_solution: AstropyWCS, 
                                   solver_name="ASTAP_ZeMosaic", 
                                   progress_callback=None):
    if not (fits_header_to_update is not None and wcs_object_solution and wcs_object_solution.is_celestial):
        if progress_callback: progress_callback("  ASTAP HeaderUpdate: MàJ header annulée: header/WCS invalide.", None, "WARN")
        return False 
    if progress_callback: progress_callback(f"  ASTAP HeaderUpdate: MàJ header FITS avec solution WCS de {solver_name}...", None, "DEBUG_DETAIL")
    if not ASTROPY_AVAILABLE_ASTROMETRY:
        if progress_callback: progress_callback("  ASTAP HeaderUpdate ERREUR: Astropy non disponible pour MàJ header.", None, "ERROR")
        return False
    try:
        wcs_keys_to_remove = [
            'WCSAXES', 'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2', 
            'CTYPE1', 'CTYPE2', 'CUNIT1', 'CUNIT2',
            'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2',
            'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2',
            'CDELT1', 'CDELT2', 'CROTA1', 'CROTA2', 
            'LONPOLE', 'LATPOLE', 'EQUINOX', 'RADESYS',
            'PV1_0', 'PV1_1', 'PV1_2', 'PV2_0', 'PV2_1', 'PV2_2' 
        ]
        for key_del in wcs_keys_to_remove:
            if key_del in fits_header_to_update:
                try:
                    del fits_header_to_update[key_del]
                except KeyError:
                    pass
        
        # Correction de la coquille ici :
        new_wcs_header_cards = wcs_object_solution.to_header(relax=True) # Utiliser relax=True est plus simple et robuste
        
        fits_header_to_update.update(new_wcs_header_cards)
        fits_header_to_update[f'{solver_name.upper()}_SOLVED'] = (True, f'{solver_name} solution')
        
        if u is not None: # S'assurer que astropy.units est importé
            try:
                if hasattr(wcs_object_solution, 'proj_plane_pixel_scales') and callable(wcs_object_solution.proj_plane_pixel_scales):
                    scales_deg = wcs_object_solution.proj_plane_pixel_scales()
                    pixscale_arcsec = np.mean(np.abs(scales_deg.to_value(u.arcsec)))
                    fits_header_to_update[f'{solver_name.upper()}_PSCALE'] = (float(f"{pixscale_arcsec:.4f}"), f'[asec/pix] Scale from {solver_name} WCS')
            except Exception:
                pass
            
        if progress_callback: progress_callback("  ASTAP HeaderUpdate: Header FITS MàJ avec WCS.", None, "DEBUG_DETAIL")
        return True
    except Exception as e_upd:
        if progress_callback: progress_callback(f"  ASTAP HeaderUpdate ERREUR: {e_upd}", None, "ERROR")
        logger.error(f"Erreur MàJ header FITS avec WCS: {e_upd}", exc_info=True) # Log le traceback complet
        return False



# DANS zemosaic_astrometry.py

def solve_with_astap(image_fits_path: str,
                     original_fits_header: fits.Header,
                     astap_exe_path: str,
                     astap_data_dir: str,
                     search_radius_deg: float | None = None,    # Kept for compatibility
                     downsample_factor: int | None = None,      # Kept for compatibility
                     sensitivity: int | None = None,            # Kept for compatibility
                     timeout_sec: int = 120,
                     update_original_header_in_place: bool = False,
                     progress_callback: callable = None):

    if not ASTROPY_AVAILABLE_ASTROMETRY:
        if progress_callback: progress_callback("ASTAP Solve ERREUR: Astropy non disponible, ASTAP solve annulé.", None, "ERROR")
        return None

    img_basename_log = os.path.basename(image_fits_path)
    if progress_callback: progress_callback(f"ASTAP Solve: Début pour '{img_basename_log}'", None, "INFO_DETAIL")
    logger.debug(f"ASTAP Solve params (entrée fonction): image='{img_basename_log}', radius={search_radius_deg}, "
                 f"downsample={downsample_factor}, sensitivity={sensitivity}")

    if not (astap_exe_path and os.path.isfile(astap_exe_path)):
        if progress_callback: progress_callback(f"ASTAP Solve ERREUR: Chemin ASTAP exe invalide: '{astap_exe_path}'.", None, "ERROR")
        return None
    if not (astap_data_dir and os.path.isdir(astap_data_dir)):
        if progress_callback: progress_callback(f"ASTAP Solve AVERT: Chemin ASTAP data non spécifié ou invalide: '{astap_data_dir}'. ASTAP pourrait ne pas trouver ses bases.", None, "WARN")
    if not (image_fits_path and os.path.isfile(image_fits_path)):
        if progress_callback: progress_callback(f"ASTAP Solve ERREUR: Chemin image FITS invalide: '{image_fits_path}'.", None, "ERROR")
        return None
    if original_fits_header is None: # Should not happen if called from worker
        if progress_callback: progress_callback("ASTAP Solve ERREUR: Header FITS original non fourni.", None, "ERROR")
        return None

    current_image_dir = os.path.dirname(image_fits_path)

    raw_dir = os.path.join(current_image_dir, "raw_source")
    os.makedirs(raw_dir, exist_ok=True)
    moved_path = os.path.join(raw_dir, os.path.basename(image_fits_path))
    try:
        shutil.move(image_fits_path, moved_path)
        image_fits_path = moved_path
    except Exception as e_move:
        if progress_callback:
            progress_callback(f"ASTAP Solve ERREUR: déplacement vers raw_source échoué: {e_move}", None, "ERROR")
        logger.error(f"Erreur déplacement fichier pour ASTAP: {e_move}", exc_info=True)
        return None

    solved_path = os.path.join(current_image_dir, "image_solved.fits")
    if os.path.exists(solved_path):
        try:
            os.remove(solved_path)
        except Exception:
            pass

    wcs_solved_obj = None

    try:
        subprocess.run(
            ["astap", "-s", "-w", image_fits_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_sec,
            cwd=current_image_dir
        )

        if os.path.exists(solved_path):
            with fits.open(solved_path) as hdul:
                solved_header = hdul[0].header.copy()
            if ASTROPY_AVAILABLE_ASTROMETRY:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FITSFixedWarning)
                    wcs_solved_obj = AstropyWCS(solved_header, naxis=2, relax=True)

            if wcs_solved_obj and update_original_header_in_place:
                _update_fits_header_with_wcs_za(original_fits_header, wcs_solved_obj, progress_callback=progress_callback)
                try:
                    with fits.open(image_fits_path, mode="update") as hdul_upd:
                        _update_fits_header_with_wcs_za(hdul_upd[0].header, wcs_solved_obj)
                        hdul_upd.flush()
                except Exception as e_upd:
                    if progress_callback:
                        progress_callback(f"ASTAP Solve AVERT: MàJ fichier FITS échouée: {e_upd}", None, "WARN")
                    logger.error(f"Erreur mise à jour FITS déplacé: {e_upd}", exc_info=True)
        else:
            if progress_callback:
                progress_callback("ASTAP Solve ERREUR: fichier image_solved.fits manquant après ASTAP", None, "ERROR")
    except subprocess.TimeoutExpired:
        if progress_callback: progress_callback(f"ASTAP Solve ERREUR: Timeout ({timeout_sec}s) pour '{img_basename_log}'.", None, "ERROR")
        logger.error(f"ASTAP command timed out for {img_basename_log}", exc_info=False)
    except FileNotFoundError:
        if progress_callback: progress_callback("ASTAP Solve ERREUR: exécutable 'astap' non trouvé.", None, "ERROR")
        logger.error("ASTAP executable not found in PATH", exc_info=False)
    except Exception as e_astap_glob:
        if progress_callback: progress_callback(f"ASTAP Solve ERREUR Inattendue: {e_astap_glob}", None, "ERROR")
        logger.error(f"Unexpected error during ASTAP execution for {img_basename_log}: {e_astap_glob}", exc_info=True)
    finally:
        astap_log = os.path.join(current_image_dir, "astap.log")
        if os.path.exists(astap_log):
            try:
                os.remove(astap_log)
            except Exception:
                pass
        gc.collect()

    if wcs_solved_obj:
        if progress_callback: progress_callback(f"ASTAP Solve: WCS trouvé pour {img_basename_log}.", None, "INFO_DETAIL")
    else:
        if progress_callback: progress_callback(f"ASTAP Solve: Pas de WCS final pour {img_basename_log}.", None, "WARN")
    return wcs_solved_obj



