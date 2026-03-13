# 🌌 ZeMosaic

**ZeMosaic** is an open-source tool for assembling **large astronomical mosaics** from FITS images, with particular support for all-in-one sensors like the **Seestar S50**.

It was born out of a need from an astrophotography Discord community called the seestar collective stacking tens of **thousands of FITS images** into clean wide-field mosaics — a task where most existing tools struggled with scale, automation, or quality.

---

## 🚀 Key Features

> Note: `lecropper` remains an annex/standalone legacy tool and is not part of the official runtime path.

- Astrometric alignment using **ASTAP**
- Smart tile grouping and automatic clustering
- Configurable stacking with:
  - **Noise-based weighting** (1/σ²)
  - **Kappa-Sigma** and **Winsorized** rejection
  - Radial feathering to blend tile borders
- Two mosaic assembly modes:
  - `Reproject & Coadd` (high quality, RAM-intensive)
  - `Incremental` (low memory, scalable)
- Stretch preview generation (ASIFits-style)
- Official GUI built with **PySide6 (Qt)**, fully translatable (EN/FR)
- Flexible FITS export with configurable `axis_order` (default `HWC`) and
  proper `BSCALE`/`BZERO` for float images
- Option to save the final mosaic as 16-bit integer FITS
- Phase-specific auto-tuning of worker threads (alignment capped at 50% of CPU threads)
- Process-based parallelization for final mosaic assembly (both `reproject_coadd` and `incremental`)

- Configurable `assembly_process_workers` to tune process count for assembly (used by both methods)

- Optional CUDA acceleration for the Mosaic-First reprojection+coadd path (Phase 4). When
  `use_gpu_phase5` is enabled and a compatible CUDA device is detected, ZeMosaic now leverages the GPU
  for mean, median, winsorized sigma-clip, and kappa-sigma stacking modes.


---

## How It Works: ZeAnalyser & Grid Mode

### ZeAnalyser: Frame Quality Analysis and Selection

ZeAnalyser is the analysis engine used by ZeMosaic to evaluate the quality of individual frames before stacking.
Its goal is simple: keep signal, reject noise and pathological frames, without requiring calibration files or manual tuning.

For each input frame, ZeAnalyser computes a set of objective image quality metrics, such as:

*   Star detection and count (robust to noise and gradients)
*   Star shape statistics (eccentricity / elongation indicators)
*   Global sharpness / structure metrics
*   Noise and background behavior
*   Optional SNR-related estimations

These metrics are combined to:

*   Reject unusable frames (e.g., tracking errors, clouds, severe blur)
*   Weight or filter frames consistently across large datasets
*   Ensure homogeneous data quality before stacking

ZeAnalyser operates fully automatically and is designed to scale efficiently to tens of thousands of frames, making it suitable for long multi-night Seestar and traditional imaging sessions.

### Grid Mode: Mosaic-First Processing Strategy

Grid Mode introduces a mosaic-first approach, specifically designed for wide fields, large sky coverage, and datasets with variable overlap.

Instead of stacking everything into a single reference frame, the field of view is divided into a regular grid of tiles. Each tile is processed independently before being reassembled into the final mosaic.

Key steps in Grid Mode:

1.  **Spatial Partitioning**
    The sky coverage is divided into overlapping grid tiles. Each input frame contributes only to the tiles it actually covers.

2.  **Local Analysis with ZeAnalyser**
    ZeAnalyser is applied per tile, not globally. This allows for local quality decisions: a frame may be rejected in one tile but accepted in another. Local seeing, tracking, or distortion issues are handled naturally.

3.  **Independent Tile Stacking**
    Each tile is stacked using only frames validated for that tile. This improves local sharpness and signal consistency, while reducing edge artifacts and uneven coverage.

4.  **Tile Cropping and Normalization**
    Invalid or low-coverage borders are automatically trimmed. Tile intensity and background are normalized prior to reprojection.

5.  **Final Mosaic Assembly**
    All tiles are reprojected using WCS information. The final mosaic is assembled with consistent geometry and color behavior.

### Why Grid Mode Matters

Grid Mode solves several classic problems of wide-field and mosaic stacking:

*   Uneven frame overlap
*   Local tracking distortions
*   Field rotation and edge degradation
*   Quality variations across large datasets

By combining ZeAnalyser’s per-frame analysis with Grid Mode’s spatially aware stacking, ZeMosaic achieves:

*   Better local sharpness
*   Reduced ghosting and duplication artifacts
*   More stable mosaics on large or imperfect datasets
*   A processing strategy that remains robust as dataset size grows

### Design Philosophy

ZeAnalyser and Grid Mode are intentionally designed to be:

*   **Automatic** – minimal user tuning required
*   **Deterministic** – same input, same output
*   **Scalable** – from a few hundred to tens of thousands of frames
*   **Instrument-agnostic** – optimized for Seestar but not limited to it

Together, they form the backbone of ZeMosaic’s modern stacking pipeline.

---

## Fonctionnement : ZeAnalyser et Mode Grille (Français)

### ZeAnalyser : Analyse et Sélection de la Qualité des Images

ZeAnalyser est le moteur d'analyse utilisé par ZeMosaic pour évaluer la qualité de chaque image individuelle avant l'empilement. Son objectif est simple : conserver le signal, rejeter le bruit et les images inutilisables, sans nécessiter de fichiers de calibration ou de réglages manuels.

Pour chaque image, ZeAnalyser calcule un ensemble de métriques objectives de qualité, telles que :

*   Détection et comptage d'étoiles (robuste au bruit et aux gradients)
*   Statistiques sur la forme des étoiles (indicateurs d'excentricité / d'élongation)
*   Métrique globale de netteté / structure
*   Analyse du bruit et du comportement du fond de ciel
*   Estimations optionnelles liées au rapport Signal/Bruit (SNR)

Ces métriques sont combinées pour :

*   Rejeter les images inexploitables (ex: erreurs de suivi, nuages, flou important)
*   Pondérer ou filtrer les images de manière cohérente sur de grands ensembles de données
*   Assurer une qualité de données homogène avant l'empilement

ZeAnalyser fonctionne de manière entièrement automatique et est conçu pour traiter efficacement des dizaines de milliers d'images, le rendant adapté aux longues sessions d'imagerie multi-nuits avec un Seestar ou un équipement traditionnel.

### Mode Grille : Stratégie de Traitement « Mosaïque d'Abord »

Le Mode Grille (Grid Mode) introduit une approche centrée sur la mosaïque, spécialement conçue pour les grands champs, les vastes couvertures célestes et les ensembles de données avec un chevauchement variable.

Au lieu d'empiler toutes les images en une seule trame de référence, le champ de vision est divisé en une grille régulière de tuiles. Chaque tuile est traitée indépendamment avant d'être réassemblée dans la mosaïque finale.

Étapes clés du Mode Grille :

1.  **Partitionnement Spatial**
    La couverture céleste est divisée en tuiles de grille qui se chevauchent. Chaque image ne contribue qu'aux tuiles qu'elle recouvre réellement.

2.  **Analyse Locale avec ZeAnalyser**
    ZeAnalyser est appliqué à chaque tuile individuellement, et non globalement. Cela permet des décisions de qualité locales : une image peut être rejetée pour une tuile mais acceptée pour une autre. Les problèmes locaux de seeing, de suivi ou de distorsion sont ainsi gérés naturellement.

3.  **Empilement Indépendant des Tuiles**
    Chaque tuile est empilée en utilisant uniquement les images validées pour cette tuile spécifique. Cela améliore la netteté locale et la cohérence du signal, tout en réduisant les artefacts de bord et les couvertures inégales.

4.  **Rognage et Normalisation des Tuiles**
    Les bordures invalides ou à faible couverture sont automatiquement rognées. L'intensité et le fond de ciel de chaque tuile sont normalisés avant la reprojection.

5.  **Assemblage Final de la Mosaïque**
    Toutes les tuiles sont reprojetées en utilisant leurs informations WCS. La mosaïque finale est assemblée avec une géométrie et une colorimétrie cohérentes.

### Pourquoi le Mode Grille est Important

Le Mode Grille résout plusieurs problèmes classiques de l'empilement de grands champs et de mosaïques :

*   Chevauchement inégal des images
*   Distorsions de suivi locales
*   Rotation de champ et dégradation des bords
*   Variations de qualité sur de grands ensembles de données

En combinant l'analyse par image de ZeAnalyser avec l'empilement spatialisé du Mode Grille, ZeMosaic obtient :

*   Une meilleure netteté locale
*   Une réduction des artefacts de « ghosting » (images fantômes) et de duplication
*   Des mosaïques plus stables sur des ensembles de données volumineux ou imparfaits
*   Une stratégie de traitement qui reste robuste à mesure que la taille de l'ensemble de données augmente

### Philosophie de Conception

ZeAnalyser et le Mode Grille sont intentionnellement conçus pour être :

*   **Automatiques** – Réglages manuels minimaux requis
*   **Déterministes** – Mêmes entrées, mêmes résultats
*   **Évolutifs** (*Scalable*) – De quelques centaines à des dizaines de milliers d'images
*   **Indépendants de l'instrument** – Optimisés pour le Seestar mais non limités à celui-ci

Ensemble, ils forment la colonne vertébrale du pipeline d'empilement moderne de ZeMosaic.

---

Quality Crop (edge artifact removal)

ZeMosaic includes an optional Quality Crop step designed to automatically remove low-quality borders that can appear after alignment/reprojection (dark rims, stretched edges, noisy bands, stacking seams, etc.). The idea is to analyze the image edges and crop away regions that statistically look “worse” than the interior.

Parameters

Enable quality crop (default: OFF)
Turns the whole feature on/off.
When OFF, ZeMosaic keeps the full tile image and does not run any edge quality analysis.

Band width (px) (default: 32)
Defines the thickness (in pixels) of the edge bands inspected for quality.
ZeMosaic analyzes borders within this width (top/bottom/left/right) to decide where quality drops.

K-sigma (default: 2.0)
Controls the sigma threshold used to decide whether a pixel/run is considered “bad” compared to expected background statistics.
Lower values = more aggressive cropping (more pixels flagged as outliers).
Higher values = more conservative cropping.

Minimum run (default: 2)
Sets the minimum length (in pixels) of a continuous bad segment before it is considered meaningful.
This helps ignore isolated bad pixels and prevents overreacting to tiny defects.

Margin (px) (default: 8)
Adds a safety margin (in pixels) when cropping.
Once a low-quality edge region is detected, ZeMosaic crops slightly deeper by this amount to avoid leaving a thin residual artifact line.

Practical guidance

If you still see obvious borders/seams, try increasing Band width slightly (e.g. 48–64) and/or lowering K-sigma (e.g. 1.5–1.8).

If you feel ZeMosaic crops too much, increase K-sigma or increase Minimum run.

---

## 📷 Requirements

### Mandatory:

- Python ≥ 3.9  
- [ASTAP](https://www.hnsky.org/astap.htm) installed with G17/H17 star catalogs

### Recommended Python packages:

```bash
pip install numpy astropy reproject opencv-python photutils scipy psutil
```
The worker originally required `DirectoryStore`, removed in `zarr>=3`.
ZeMosaic now falls back to `LocalStore`, and skips the old
`LRUStoreCache` wrapper when running against Zarr 3.
Both Zarr 2.x and 3.x are supported (tested on Python 3.11+).

🧠 Inspired by PixInsight
ZeMosaic draws strong inspiration from the image integration strategies of PixInsight, developed by Juan Conejero at Pleiades Astrophoto.

Specifically, the implementations of:

Noise Variance Weighting (1/σ²)

Kappa-Sigma and Winsorized Rejection

Radial feather blending

...are adapted from methods described in:

📖 PixInsight 1.6.1 – New ImageIntegration Features
Juan Conejero, 2010
Forum thread

🙏 We gratefully acknowledge Juan Conejero's contributions to astronomical image processing.

🛠 Dependencies
ZeMosaic uses several powerful open-source Python libraries:

numpy and scipy for numerical processing

astropy for FITS I/O and WCS handling

reproject for celestial reprojection

opencv-python for debayering

photutils for source detection and background estimation

psutil for memory monitoring

PySide6 (Qt) for the official graphical user interface

> **Note (Linux/macOS):** The official ZeMosaic frontend is Qt-only and requires `PySide6`.

📦 Installation & Usage
1. 🔧 Install Python dependencies
If you have a local clone of the repository, make sure you're in the project folder, then run:

pip install -r requirements.txt
💡 Requirements are mostly flexible. ZeMosaic now supports both zarr 2.x and
3.x, automatically falling back to `LocalStore` when `DirectoryStore` is
unavailable. The project is tested with Python 3.11+.

If you prefer to install manually:

pip install numpy astropy reproject opencv-python photutils scipy psutil

2. 🚀 Launch ZeMosaic
Once the dependencies are installed:
python run_zemosaic.py

The GUI will open. From there:

Select your input folder (with raw FITS images)

Choose your output folder

Configure ASTAP paths and options

Adjust stacking & mosaic settings

Click "Start Hierarchical Mosaic"

📁 Requirements Summary
✅ Python 3.9 or newer

✅ ASTAP installed + star catalogs (D50 or H18)

✅ FITS images (ideally calibrated, debayered or raw from Seestar)
✅ Python multiprocessing enabled (ProcessPoolExecutor is used for assembly)

✅ `assembly_process_workers` can be set in `zemosaic_config.json` to control
   how many processes handle final mosaic assembly (0 = auto, applies to both methods)


🖥️ How to Run
After installing Python and dependencies:

python run_zemosaic.py
Use the GUI to:

Choose your input/output folders

Configure ASTAP paths

Select stacking and assembly options

Click Start Hierarchical Mosaic

### Official Qt interface

ZeMosaic now uses a PySide6/Qt interface as the only official frontend.

To run the official frontend:

1. Install the optional dependency:

   ```bash
   pip install PySide6
   ```

2. Launch ZeMosaic with either of the following options:

   ```bash
   python run_zemosaic.py
   ```

If PySide6 is unavailable, ZeMosaic reports a clear startup error. No Tk fallback is used on the official path.

#### Automatic ZeAnalyser / Beforehand Tool Discovery (Qt GUI)
To enable the `Analyse` button, install a compatible analysis tool in the parent directory of `zemosaic/`. ZeMosaic auto-detects them at startup.

**Discovery Rules & UI Behavior:**
1.  It first checks for a `zeanalyser/` directory. If found, the **Analyse** button is enabled, using **ZeAnalyser** as the backend.
2.  If not found, it looks for `seestar/beforehand/`. If this directory exists, the button is enabled, using the legacy **Beforehand** backend.
3.  If neither is found, the `Analyse` button is not displayed, keeping the UI clean.

The button's tooltip will always indicate which backend is active. If both are installed, ZeAnalyser takes priority.

#### Découverte automatique des outils ZeAnalyser / Beforehand (IUG Qt)
Pour activer le bouton `Analyser`, installez un outil d'analyse compatible dans le répertoire parent de `zemosaic/`. ZeMosaic les détecte automatiquement au démarrage.

**Règles de découverte et comportement de l'interface :**
1.  Le logiciel vérifie d'abord la présence d'un répertoire `zeanalyser/`. S'il est trouvé, le bouton **Analyser** est activé et utilise le moteur **ZeAnalyser**.
2.  Sinon, il recherche `seestar/beforehand/`. Si ce répertoire existe, le bouton est activé et utilise le moteur historique **Beforehand**.
3.  Si aucun des deux n'est trouvé, le bouton `Analyser` n'est pas affiché, gardant l'interface épurée.

L'info-bulle du bouton indiquera toujours quel moteur est actif. Si les deux sont installés, ZeAnalyser est prioritaire.

### Force Seestar workflow checkbox

The Main tab of both GUIs exposes two related toggles for Seestar datasets:

- **Auto-detect Seestar frames** stays on by default and inspects the FITS `INSTRUME`
  header (or any instrument hint provided by the filter UI). When the label
  contains “Seestar/S50/S30”, ZeMosaic enters the Seestar/Mosaic-First workflow
  automatically.
- **Force Seestar workflow** is a manual override. When it is checked, the filter
  dialog and the worker assume the Mosaic-First path regardless of what the FITS
  headers say. The filter always prepares/reuses the global WCS descriptor,
  exports the `global_wcs_meta`/FITS/JSON paths, and sets the workflow mode to
  `seestar`, so the worker skips the classic per-master-tile stack even if the
  dataset mixes instruments or carries incomplete metadata.

Enable this override whenever the automatic detection fails (e.g. FITS files
stripped of `INSTRUME`), or when you deliberately want to run non-Seestar data
through the Seestar-optimized Mosaic-First pipeline. Disabling it reverts to the
classic workflow unless the headers clearly advertise Seestar frames.

### Global WCS auto-cropping (English)

ZeMosaic can optionally trim the Mosaic-first canvas so the exported FITS only contains sky regions with real coverage.  
Enable this by setting `global_wcs_autocrop_enabled` to `true` inside `zemosaic_config.py` (see the `DEFAULT_CONFIG` block near the other `global_wcs_*` keys) or in your personal `zemosaic_config.json`.  
Once enabled, the worker inspects the Phase 5 coverage map, removes empty borders, and shifts the global WCS `CRPIX`/`NAXIS` values automatically so downstream tools see the reduced frame.  
Use `global_wcs_autocrop_margin_px` (same section) to keep a safety border in pixels—default is 64 px.

### Recadrage automatique du WCS global (Français)

ZeMosaic peut rogner automatiquement la mosaïque finale (mode Mosaic-first) pour ne conserver que la zone réellement couverte par les images.  
Activez `global_wcs_autocrop_enabled` en le passant à `true` dans `zemosaic_config.py` (section `DEFAULT_CONFIG`, proche des autres clés `global_wcs_*`) ou dans votre fichier `zemosaic_config.json`.  
Une fois l’option activée, l’ouvrier analyse la carte de couverture de la Phase 5, supprime les bordures vides et ajuste `CRPIX` / `NAXIS` du WCS global afin que les outils en aval utilisent la toile réduite.  
La marge de sécurité se règle avec `global_wcs_autocrop_margin_px` (en pixels, 64 px par défaut).

### GPU helper for Phase 4

Setting `use_gpu_phase5` to `true` (via the worker configuration or overrides) now enables the CUDA helper
for the entire Mosaic-First reprojection+coadd stage. If a supported GPU is available, ZeMosaic will run
mean, median, winsorized, and kappa-sigma global stacking directly on the GPU and automatically fall back
to the CPU only when the helper is unavailable. For integration testing you can optionally set
`gpu_helper_verify_tolerance` in the global plan to log the max per-pixel delta between the GPU and CPU
reference implementation.

### macOS quickstart

1. Install Python 3.11+ from [python.org](https://www.python.org/downloads/).
2. Download the macOS ASTAP `.dmg`, drag `ASTAP.app` into `/Applications`, then install the star catalogs
   (D50/H17) under `/Library/Application Support/ASTAP` or `~/Library/Application Support/ASTAP`.
3. Inside the project folder run:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python3 -m pip install --upgrade pip
   python3 -m pip install -r requirements.txt
   python3 run_zemosaic.py
   ```

4. ZeMosaic now auto-detects `/Applications/ASTAP.app/Contents/MacOS/astap` and the associated catalog
   directories, but you can override them at any time:

   ```bash
   export ASTAP_EXE=/Applications/ASTAP.app/Contents/MacOS/astap
   export ASTAP_DATA_DIR="/Library/Application Support/ASTAP"
   ```

### Linux quickstart

1. Install Python, pip, and build tooling via your package manager. Example for Debian/Ubuntu:

   ```bash
   sudo apt update
   sudo apt install python3 python3-venv python3-pip python3-dev build-essential
   ```

   Fedora/RHEL users can run `sudo dnf install python3 python3-venv python3-pip`.

2. Install the ASTAP Linux package (from https://www.hnsky.org/astap.htm) and the desired star catalogs.
   The default installer places binaries under `/usr/bin/astap` and data under `/opt/astap`.

3. Create a virtual environment and launch ZeMosaic:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python3 -m pip install --upgrade pip
   python3 -m pip install -r requirements.txt
   python3 run_zemosaic.py
   ```

4. ASTAP locations are discovered automatically on `/usr/bin`, `/usr/local/bin`, and `/opt/astap`, but
   you can pin custom installs:

   ```bash
   export ASTAP_EXE=/usr/local/bin/astap
   export ASTAP_DATA_DIR=/opt/astap
   ```

### ASTAP path detection & overrides

`zemosaic_config.py` now validates the stored executable/data paths on startup and, if needed, scans the
common installation directories for Windows (`Program Files`), macOS (`/Applications/ASTAP*.app`), and
Linux (`/usr/bin`, `/opt/astap`, etc.). The following environment variables are respected ahead of the
auto-detected paths:

- `ASTAP_EXE`, `ASTAP_BIN`, or `ASTAP_PATH` for the binary
- `ASTAP_DATA_DIR`, `ASTAP_STAR_DB`, or `ASTAP_DATABASE` for the star catalogs

You can inspect what was found by running:

```bash
python - <<'PY'
from zemosaic_config import detect_astap_installation
print(detect_astap_installation())
PY
```

### ASTAP concurrency guard

Running more than one ASTAP instance in parallel can trigger the `Access violation` pop-up shown by the
native solver. To keep ZeMosaic/ZeSeestarStacker runs unattended, every ASTAP call now cooperates through
an inter-process lock:

- The configured `astap_max_instances` value (GUI + `zemosaic_config.json`) becomes a global cap shared by
  every ZeMosaic utility launched on the same machine. Extra workers simply wait for a free slot.
- Progress logs mention when we are waiting for another process and resume automatically once the lock is
  released, eliminating the Windows dialog.
- Advanced users who prefer the legacy behaviour can set `ZEMOSAIC_ASTAP_DISABLE_IPC_LOCK=1` before
  launching the tools. Override the lock directory with `ZEMOSAIC_ASTAP_LOCK_DIR=/path/to/tmp` if the
  default `%TEMP%/zemosaic_astap_slots` is not suitable (portable drives, RAM disks, etc.).
- SetErrorMode is enabled on Windows so ASTAP cannot raise modal crash boxes, and additional guards keep the
  solver healthy under bursty workloads: per-image file locks, a configurable rate limiter
  (`ZEMOSAIC_ASTAP_RATE_SEC` / `ZEMOSAIC_ASTAP_RATE_BURST`), and automatic retries with back-off
  (`ZEMOSAIC_ASTAP_RETRIES`, `ZEMOSAIC_ASTAP_BACKOFF_SEC`).

🔧 Build & Compilation (Windows)

🇬🇧 Instructions (English)
1. Install Python 3.13 (x64) from python.org.
2. From the project root:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   python -m pip install --upgrade pyinstaller
   pyinstaller --noconfirm --clean ZeMosaic.spec
   ```

   The default output is `dist/ZeMosaic/ZeMosaic.exe` (onedir).
   This path keeps the current GPU-enabled setup because `requirements.txt` currently includes `cupy-cuda12x`.

   If you want a CPU-only package with a much smaller payload:

   ```powershell
   set ZEMOSAIC_REQUIREMENTS_FILE=requirements_no_gpu.txt
   compile\compile_zemosaic._win.bat
   ```

   Or manually:

   ```powershell
   python -m pip install -r requirements_no_gpu.txt
   pyinstaller --noconfirm --clean ZeMosaic.spec
   ```

3. Optional onefile build:

   ```powershell
   set ZEMOSAIC_BUILD_MODE=onefile
   set ZEMOSAIC_RUNTIME_TMPDIR=C:\Temp
   pyinstaller --noconfirm --clean ZeMosaic.spec
   ```

   Output becomes `dist/ZeMosaic.exe` (onefile).

4. Optional debug build (console ON) for diagnostics:

   ```powershell
   set ZEMOSAIC_DEBUG_BUILD=1
   pyinstaller --noconfirm --clean ZeMosaic.spec
   ```

   Clear the env var for release builds (console OFF):

   ```powershell
   set ZEMOSAIC_DEBUG_BUILD=
   ```

5. Helper script (uses the same spec):

   ```powershell
   compile\compile_zemosaic._win.bat
   ```

Notes:
- Resources `locales/`, `icon/`, and `gif/` are bundled via `ZeMosaic.spec`.
- `PySide6` is required for the official packaged frontend (Qt-only).
- `matplotlib` is optional: if missing, the Qt filter preview is disabled.
- `cupy-cuda12x` is optional: if missing (or if NVIDIA drivers are missing/incompatible, or if required CUDA DLLs are not present on the target machine), ZeMosaic falls back to CPU. On Windows this often means having CUDA Toolkit (or at least its runtime DLLs) available via `%CUDA_PATH%\\bin`/`PATH`.
- `requirements.txt` keeps the current working GPU setup (`cupy-cuda12x`). `requirements_no_gpu.txt` is available for CPU-only builds when you want a smaller artifact.
- Prefer onedir for reliability; onefile can hit Windows path length issues (example: Shapely `WinError 206`), mitigated by using a short `ZEMOSAIC_RUNTIME_TMPDIR` like `C:\Temp` and/or enabling long paths in Windows.
- `zemosaic_installer.iss` does not choose the CUDA package. It simply packages `dist\\ZeMosaic\\*`. If you built with `requirements.txt`, the installer packages the GPU-enabled build. If you built with `requirements_no_gpu.txt`, it packages the CPU-only build.
- The relevant `.iss` lines are the `[Files]` entry pointing to `dist\\ZeMosaic\\*`, plus `[Icons]` / `[Run]` pointing to `{app}\\ZeMosaic.exe`. You normally do not need to change them for a different CUDA version.

Mini smoke-test (packaged build):
- Launch `dist/ZeMosaic/ZeMosaic.exe` (or `dist/ZeMosaic.exe` in onefile mode).
- Confirm the UI starts without crash and icons load (window icon / toolbar icons).
- Switch language (if available) to confirm `locales/` loads.
- Pick an input folder and start a small run to confirm the worker starts and writes output.

🇫🇷 Instructions (Francais)
1. Installez Python 3.13 (x64) depuis python.org.
2. Depuis la racine du projet :

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   python -m pip install --upgrade pyinstaller
   pyinstaller --noconfirm --clean ZeMosaic.spec
   ```

   La sortie par défaut est `dist/ZeMosaic/ZeMosaic.exe` (onedir).
   Ce chemin conserve le build GPU actuel car `requirements.txt` contient actuellement `cupy-cuda12x`.

   Si vous voulez un paquet CPU-only beaucoup plus léger :

   ```powershell
   set ZEMOSAIC_REQUIREMENTS_FILE=requirements_no_gpu.txt
   compile\compile_zemosaic._win.bat
   ```

   Ou en manuel :

   ```powershell
   python -m pip install -r requirements_no_gpu.txt
   pyinstaller --noconfirm --clean ZeMosaic.spec
   ```

3. Build onefile optionnel :

   ```powershell
   set ZEMOSAIC_BUILD_MODE=onefile
   set ZEMOSAIC_RUNTIME_TMPDIR=C:\Temp
   pyinstaller --noconfirm --clean ZeMosaic.spec
   ```

   La sortie devient `dist/ZeMosaic.exe` (onefile).

4. Build debug optionnel (console ON) pour diagnostiquer :

   ```powershell
   set ZEMOSAIC_DEBUG_BUILD=1
   pyinstaller --noconfirm --clean ZeMosaic.spec
   ```

   Remettez l'env var a vide pour la release (console OFF) :

   ```powershell
   set ZEMOSAIC_DEBUG_BUILD=
   ```

5. Script helper (meme spec) :

   ```powershell
   compile\compile_zemosaic._win.bat
   ```

Notes :
- Les ressources `locales/`, `icon/`, et `gif/` sont embarquees via `ZeMosaic.spec`.
- `PySide6` est requis pour l'interface officielle packagée (Qt-only).
- `matplotlib` est optionnel : s'il manque, l'aperçu (Qt filter preview) est désactivé.
- `cupy-cuda12x` est optionnel : s'il manque (ou si les drivers NVIDIA sont absents/incompatibles, ou si les DLL CUDA nécessaires ne sont pas présentes sur la machine cible), ZeMosaic retombe en CPU. Sous Windows cela implique souvent CUDA Toolkit (ou au minimum ses DLL runtime) accessibles via `%CUDA_PATH%\\bin`/`PATH`.
- `requirements.txt` conserve le setup GPU actuel qui fonctionne (`cupy-cuda12x`). `requirements_no_gpu.txt` est disponible pour produire des builds CPU-only plus petits.
- Préférez onedir pour la fiabilité ; onefile peut déclencher des soucis de longueur de chemin Windows (ex: Shapely `WinError 206`), atténués via un `ZEMOSAIC_RUNTIME_TMPDIR` court comme `C:\Temp` et/ou l'activation des long paths dans Windows.
- `zemosaic_installer.iss` ne choisit pas le paquet CUDA. Il empaquette simplement `dist\\ZeMosaic\\*`. Si vous avez buildé avec `requirements.txt`, l'installateur emballera la version GPU. Si vous avez buildé avec `requirements_no_gpu.txt`, il emballera la version CPU-only.
- Les lignes `.iss` pertinentes sont l'entrée `[Files]` qui pointe sur `dist\\ZeMosaic\\*`, ainsi que `[Icons]` / `[Run]` qui pointent sur `{app}\\ZeMosaic.exe`. En pratique vous n'avez pas à les changer pour une autre version de CUDA.

Mini smoke-test (build packagé) :
- Lancez `dist/ZeMosaic/ZeMosaic.exe` (ou `dist/ZeMosaic.exe` en onefile).
- Vérifiez que l'UI démarre sans crash et que les icônes se chargent (icône fenêtre / toolbar).
- Changez la langue (si dispo) pour confirmer le chargement de `locales/`.
- Choisissez un dossier d'entrée et lancez un petit run pour valider le démarrage du worker et l'écriture de sortie.

🛠️ Build & Compilation (macOS/Linux)
🇬🇧 Instructions (English)

1. Ensure you have a virtual environment (`python3 -m venv .venv`), then activate it: `source .venv/bin/activate`.
2. Install the runtime requirements once: `python3 -m pip install -r requirements.txt`.
3. Make the helper executable and launch it:

   ```bash
   chmod +x compile/build_zemosaic_posix.sh
   ./compile/build_zemosaic_posix.sh
   ```

   The script installs/updates PyInstaller inside `.venv` and produces `dist/zemosaic`.
   By default it uses `requirements.txt`, which preserves the current GPU-enabled setup when `cupy-cuda12x` is available for the platform.

   For a CPU-only build:

   ```bash
   ZEMOSAIC_REQUIREMENTS_FILE=requirements_no_gpu.txt ./compile/build_zemosaic_posix.sh
   ```

🇫🇷 Instructions (Français)

1. Créez/activez votre environnement virtuel (`python3 -m venv .venv` puis `source .venv/bin/activate`).
2. Installez les dépendances (`python3 -m pip install -r requirements.txt`).
3. Rendez le script exécutable puis lancez-le :

   ```bash
   chmod +x compile/build_zemosaic_posix.sh
   ./compile/build_zemosaic_posix.sh
   ```

   L'exécutable macOS/Linux sera généré dans `dist/zemosaic`.
   Par défaut le script utilise `requirements.txt`, ce qui conserve le setup GPU actuel quand `cupy-cuda12x` existe pour la plateforme.

   Pour un build CPU-only :

   ```bash
   ZEMOSAIC_REQUIREMENTS_FILE=requirements_no_gpu.txt ./compile/build_zemosaic_posix.sh
   ```

### Windows GitHub release

1. Build the Windows onedir package:

   ```powershell
   compile\compile_zemosaic._win.bat
   ```

2. Optionally build a CPU-only variant if the GPU bundle is too large:

   ```powershell
   set ZEMOSAIC_REQUIREMENTS_FILE=requirements_no_gpu.txt
   compile\compile_zemosaic._win.bat
   ```

3. Create a zip from `dist\ZeMosaic`.
4. Publish that zip as a GitHub Release asset rather than committing `dist/` to the repository.

Notes:
- GitHub blocks regular Git files above 100 MiB, so `dist/` should not be committed.
- GitHub Release assets are the right place for Windows binaries.
- The installer script `zemosaic_installer.iss` packages `dist\ZeMosaic\*` exactly as built.

### Release Windows sur GitHub

1. Générez le build Windows onedir :

   ```powershell
   compile\compile_zemosaic._win.bat
   ```

2. En option, générez une variante CPU-only si le bundle GPU est trop gros :

   ```powershell
   set ZEMOSAIC_REQUIREMENTS_FILE=requirements_no_gpu.txt
   compile\compile_zemosaic._win.bat
   ```

3. Créez une archive zip à partir de `dist\ZeMosaic`.
4. Publiez ce zip dans une GitHub Release au lieu de versionner `dist/` dans le dépôt.

Notes :
- GitHub bloque les fichiers Git classiques au-delà de 100 MiB, donc `dist/` ne doit pas être commit.
- Les GitHub Releases sont le bon endroit pour publier les binaires Windows.
- `zemosaic_installer.iss` empaquette exactement `dist\ZeMosaic\*` tel qu'il a été construit.

### Memory-mapped coadd (enabled by default)

```jsonc
{
  "final_assembly_method": "reproject_coadd",
  "coadd_use_memmap": true,
  "coadd_memmap_dir": "D:/ZeMosaic_memmap",
  "coadd_cleanup_memmap": true
  "assembly_process_workers": 0
}
```
`assembly_process_workers` also defines how many workers the incremental method uses.
A final mosaic of 20 000 × 20 000 px in RGB needs ≈ 4.8 GB
(4 × H × W × float32). Make sure the target disk/SSD has enough space.
Hot pixel masks detected during preprocessing are also written to the temporary
cache directory to further reduce memory usage.

### Memory-saving parameters

The configuration file exposes a few options to control memory consumption:

- `auto_limit_frames_per_master_tile` – automatically split raw stacks based on available RAM.
- `max_raw_per_master_tile` – manual cap on raw frames stacked per master tile (0 disables).
- `winsor_worker_limit` – maximum parallel workers during the Winsorized rejection step.
- `winsor_max_frames_per_pass` – maximum frames processed at once during Winsorized rejection (0 keeps previous behaviour).
- `winsor_auto_fallback_on_memory_error` – proactively halve the batch size, then fall back to disk streaming when NumPy cannot allocate RAM.
- `winsor_min_frames_per_pass` – lower bound for the streaming fallback (default 4).
- `winsor_memmap_fallback` – `auto` (default) activates disk-backed memmap only when needed, `always` forces it, `never` keeps pure RAM processing.
- `winsor_split_strategy` – choose `sequential` (default) or `roundrobin` chunk scheduling to balance memory pressure across large stacks.

Machines with < 16 GB RAM benefit from setting `winsor_max_frames_per_pass` to 32–48 and keeping the automatic fallback enabled. En dessous de 16 Gio de RAM, conservez l’option de secours automatique activée et limitez les passes à 32–48 images ; le mode `memmap` sera déclenché si nécessaire pour éviter les erreurs `ArrayMemoryError`.

6 ▸ Quick CLI example
```bash
run_zemosaic.py \
  --final_assembly_method reproject_coadd \
  --coadd_memmap_dir D:/ZeMosaic_memmap \
  --coadd_cleanup_memmap \
  --assembly_process_workers 4
```




🧪 Troubleshooting
If astrometric solving fails:

Check ASTAP path and data catalogs

Ensure your images contain enough stars

Use a Search Radius of ~3.0°

Watch zemosaic_worker.log for full tracebacks

📎 License
ZeMosaic is licensed under GPLv3 — feel free to use, adapt, and contribute.

🤝 Contributions
Feature requests, bug reports, and pull requests are welcome!
Please include log files and test data if possible when reporting issues.

🌠 Happy mosaicking!
