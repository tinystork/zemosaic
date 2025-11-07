# 🌌 ZeMosaic

**ZeMosaic** is an open-source tool for assembling **large astronomical mosaics** from FITS images, with particular support for all-in-one sensors like the **Seestar S50**.

It was born out of a need from an astrophotography Discord community called the seestar collective stacking tens of **thousands of FITS images** into clean wide-field mosaics — a task where most existing tools struggled with scale, automation, or quality.

---

## 🚀 Key Features

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
- GUI built with **Tkinter**, fully translatable (EN/FR)
- Flexible FITS export with configurable `axis_order` (default `HWC`) and
  proper `BSCALE`/`BZERO` for float images
- Option to save the final mosaic as 16-bit integer FITS
- Phase-specific auto-tuning of worker threads (alignment capped at 50% of CPU threads)
- Process-based parallelization for final mosaic assembly (both `reproject_coadd` and `incremental`)

- Configurable `assembly_process_workers` to tune process count for assembly (used by both methods)


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

tkinter for the graphical user interface

> **Note (Linux/macOS):** Tkinter is bundled with the official Python installers.
> On minimal Linux distributions you must install it via your package manager
> (e.g. `sudo apt install python3-tk` or `sudo dnf install python3-tkinter`).
> It is not published as a pip package named `tk`.

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

### macOS quickstart

1. Install Python 3.11+ from [python.org](https://www.python.org/downloads/) (includes Tk).  
   Homebrew users should also `brew install tcl-tk` and follow the caveats so Tk can be located.
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

1. Install Python, pip, Tk, and build tooling via your package manager. Example for Debian/Ubuntu:

   ```bash
   sudo apt update
   sudo apt install python3 python3-venv python3-pip python3-tk python3-dev build-essential
   ```

   Fedora/RHEL users can run `sudo dnf install python3 python3-venv python3-pip python3-tkinter`.

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

🔧 Build & Compilation (Windows) / Compilation (Windows)
🇬🇧 Instructions (English)
To build the standalone executable version of ZeMosaic, follow these steps:

Install Python 3.13 from python.org

Create and activate a virtual environment (if not already done):

powershell
Copier
Modifier
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Build the .exe by running:

powershell
Copier
Modifier
compile\build_zemosaic.bat
The final executable will be created in dist/zemosaic.exe.

✅ Translations (locales/*.json) and application icons (icon/zemosaic.ico) are automatically included.

🇫🇷 Instructions (Français)
Pour créer l'exécutable autonome de ZeMosaic, suivez ces étapes :

Installez Python 3.13 depuis python.org

Créez et activez un environnement virtuel (si ce n’est pas déjà fait) :

powershell
Copier
Modifier
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Lancez la compilation de l’exécutable avec :

powershell
Copier
Modifier
compile\build_zemosaic.bat
L’exécutable final se trouvera dans dist/zemosaic.exe.

✅ Les fichiers de traduction (locales/*.json) et les icônes (icon/zemosaic.ico) sont inclus automatiquement.

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

🇫🇷 Instructions (Français)

1. Créez/activez votre environnement virtuel (`python3 -m venv .venv` puis `source .venv/bin/activate`).
2. Installez les dépendances (`python3 -m pip install -r requirements.txt`).
3. Rendez le script exécutable puis lancez-le :

   ```bash
   chmod +x compile/build_zemosaic_posix.sh
   ./compile/build_zemosaic_posix.sh
   ```

   L'exécutable macOS/Linux sera généré dans `dist/zemosaic`.

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
