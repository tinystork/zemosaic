# 🌌 ZeMosaic

**ZeMosaic** is an open-source tool for building large astronomical mosaics from FITS images, with strong support for Seestar-style high-volume datasets.

Built by **Tinystork (Tristan Nauleau)** with **J.A.R.V.I.S.** (ChatGPT), it targets a simple promise: **robust, scalable mosaic production without sacrificing visual quality**.

---

## 🚀 Key features

- Astrometric alignment with **ASTAP**
- Smart grouping/clustering for large image sets
- Configurable stacking:
  - noise-variance weighting
  - kappa-sigma / winsorized rejection
- Two final assembly methods:
  - `reproject_coadd` (quality-first)
  - `incremental` (memory-friendly)
- Adaptive performance controls (workers, chunking, memory guardrails)
- Intertile photometric harmonization + sparse-graph safety guardrails
- Patchwork/seam suppression in Phase 5
- **Dual output workflow**:
  - scientific FITS (`*_science.fits`)
  - aesthetic FITS (`*_aesthetic.fits`)
- **Progressive resume** support (`resume=auto`) for long-running jobs
- Qt GUI (official runtime)

---

## 📷 Requirements

### Mandatory

- Python **3.9+**
- [ASTAP](https://www.hnsky.org/astap.htm) installed
- ASTAP star databases (G17/H17 or equivalent)

### Python packages

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install numpy astropy reproject opencv-python photutils scipy psutil
```

---

## ▶️ Run

```bash
python run_zemosaic.py
```

Then in the GUI:
1. Select input FITS folder
2. Select output folder
3. Check ASTAP executable + data paths
4. Tune stacking/assembly options
5. Start mosaic processing

---

## ⚙️ Configuration model (`zemosaic_config.py` vs `zemosaic_config.json`)

ZeMosaic uses a layered configuration model:

1. **Python defaults** from `zemosaic_config.py` (`DEFAULT_CONFIG`)
2. **Persistent overrides** from `zemosaic_config.json`
3. **Environment overrides** for selected keys (notably `ZEM_ALTALPHA_*`)
4. Final normalization/synchronization (GPU aliases, path aliases, Qt backend enforcement, etc.)

**Effective priority:** `Environment > JSON > Python defaults`.

Notes:
- `input_dir`/`input_folder` and `output_dir`/`output_folder` are synchronized aliases.
- GPU flags (`use_gpu_phase5`, `stack_use_gpu`, `use_gpu_stack`) are normalized for consistency.
- Intertile safety key: `intertile_force_safe_mode` in `zemosaic_config.json` (`"auto"` default, `true` to force single-worker intertile, `false` to disable forced safe-mode). In `"auto"`, Windows resolves to safe-mode ON by default; other OS keep automatic parallel mode.

---

## 🔁 Resume mechanism (long-run safety)

Resume is designed to avoid full restart after interruption/crash.

Set:

```json
{
  "resume": "auto"
}
```

State is stored under:

- `.zemosaic_state/phase1_manifest.json`
- `.zemosaic_state/stack_plan_manifest.json`
- `.zemosaic_state/master_tiles_manifest.json`
- `.zemosaic_state/phase5_manifest.json`
- `.zemosaic_state/phase5_mosaic.npy`
- `.zemosaic_state/phase5_coverage.npy`
- `.zemosaic_state/phase5_alpha.npy`

Principles:
- Reuse valid artifacts
- Rebuild only missing/incompatible/corrupted parts
- Local invalidation only (no global purge for one missing file)

Practical recommendation:
- Keep `resume=auto`
- Use distinct output folders per run generation to avoid ambiguity

---

## 🧠 Recommended production baseline (example)

```json
{
  "resume": "auto",
  "cache_retention": "run_end",
  "chunk_profile_mode": "safe_dynamic_plus",
  "parallel_autotune_enabled": true,
  "use_gpu_phase5": true,
  "stacking_normalize_method": "linear_fit",
  "stacking_rejection_algorithm": "winsorized_sigma_clip",
  "final_assembly_method": "incremental",
  "phase5_alpha_soft_weights": true,
  "phase5_alpha_weight_floor": 0.02,
  "patchwork_suppressor_enabled": true,
  "patchwork_suppressor_strength": "strong",
  "export_aesthetic_fits": true,
  "aesthetic_hole_fill_enabled": true,
  "aesthetic_profile_preset": "strong"
}
```

---

## 🧪 Troubleshooting

If solving or processing fails:

- Verify ASTAP executable/data paths
- Check that images have enough stars and valid FITS headers
- Inspect logs (`zemosaic_worker.log` and runtime logs)
- If memory pressure appears, reduce workers/chunk aggressiveness

---

## 🛠 Build (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
compile\build_zemosaic.bat
```

Output executable:

- `dist/zemosaic.exe`

---

## 🙏 Credits

- **Core project:** Tinystork (Tristan Nauleau)
- **Engineering copilot:** J.A.R.V.I.S. (ChatGPT)
- Strong methodological inspiration from **PixInsight** image integration work by **Juan Conejero** (noise weighting, robust rejection strategies, integration concepts).

---

## 📎 License

ZeMosaic is released under **GPL-3.0**.

---

## 🤝 Contributing

Issues and PRs are welcome. When possible, include:
- logs,
- a minimal reproduction,
- config excerpt,
- dataset characteristics (frame count, resolution, camera/profile).

---

🌠 Happy mosaicking.

---

## 🇫🇷 Version française

## 🌌 ZeMosaic

**ZeMosaic** est un outil open-source pour construire de grandes mosaïques astronomiques à partir d’images FITS, avec un support fort des jeux de données volumineux type Seestar.

Créé par **Tinystork (Tristan Nauleau)** avec **J.A.R.V.I.S.** (ChatGPT), il vise une promesse simple: **une production de mosaïques robuste et scalable, sans sacrifier la qualité visuelle**.

---

## 🚀 Fonctionnalités clés

- Alignement astrométrique avec **ASTAP**
- Regroupement/clustering intelligent pour gros volumes d’images
- Stacking configurable:
  - pondération par variance du bruit
  - rejet kappa-sigma / winsorisé
- Deux méthodes d’assemblage final:
  - `reproject_coadd` (qualité prioritaire)
  - `incremental` (économe en mémoire)
- Contrôles de performance adaptatifs (workers, chunking, garde-fous mémoire)
- Harmonisation photométrique intertile + garde-fous graphes clairsemés
- Suppression patchwork/coutures en phase 5
- **Workflow double sortie**:
  - FITS scientifique (`*_science.fits`)
  - FITS esthétique (`*_aesthetic.fits`)
- **Resume progressif** (`resume=auto`) pour les runs longs
- GUI Qt (runtime officiel)

---

## 📷 Prérequis

### Obligatoire

- Python **3.9+**
- [ASTAP](https://www.hnsky.org/astap.htm) installé
- Bases d’étoiles ASTAP (G17/H17 ou équivalent)

### Paquets Python

```bash
pip install -r requirements.txt
```

Ou manuellement:

```bash
pip install numpy astropy reproject opencv-python photutils scipy psutil
```

---

## ▶️ Lancer

```bash
python run_zemosaic.py
```

Puis dans la GUI:
1. Sélectionner le dossier FITS d’entrée
2. Sélectionner le dossier de sortie
3. Vérifier les chemins ASTAP (exécutable + données)
4. Ajuster les options stacking/assemblage
5. Lancer le traitement mosaïque

---

## ⚙️ Modèle de configuration (`zemosaic_config.py` vs `zemosaic_config.json`)

ZeMosaic utilise un modèle de configuration en couches:

1. **Défauts Python** depuis `zemosaic_config.py` (`DEFAULT_CONFIG`)
2. **Overrides persistants** depuis `zemosaic_config.json`
3. **Overrides environnement** pour certaines clés (notamment `ZEM_ALTALPHA_*`)
4. Normalisation/synchronisation finale (alias GPU, alias chemins, verrouillage backend Qt, etc.)

**Priorité effective:** `Environnement > JSON > Défauts Python`.

Notes:
- `input_dir`/`input_folder` et `output_dir`/`output_folder` sont des alias synchronisés.
- Les flags GPU (`use_gpu_phase5`, `stack_use_gpu`, `use_gpu_stack`) sont normalisés pour rester cohérents.
- Clé de sécurité intertile: `intertile_force_safe_mode` dans `zemosaic_config.json` (défaut `"auto"`, `true` pour forcer le mode intertile mono-worker, `false` pour désactiver ce forçage). En mode `"auto"`, Windows active le safe-mode par défaut; les autres OS restent en mode parallèle automatique.

---

## 🔁 Mécanisme de resume (sécurité runs longs)

Le resume sert à éviter un redémarrage complet après interruption/crash.

Configurer:

```json
{
  "resume": "auto"
}
```

L’état est stocké sous:

- `.zemosaic_state/phase1_manifest.json`
- `.zemosaic_state/stack_plan_manifest.json`
- `.zemosaic_state/master_tiles_manifest.json`
- `.zemosaic_state/phase5_manifest.json`
- `.zemosaic_state/phase5_mosaic.npy`
- `.zemosaic_state/phase5_coverage.npy`
- `.zemosaic_state/phase5_alpha.npy`

Principes:
- Réutiliser les artefacts valides
- Reconstruire uniquement les parties manquantes/incompatibles/corrompues
- Invalidation locale uniquement (pas de purge globale pour un seul fichier manquant)

Recommandation pratique:
- Garder `resume=auto`
- Utiliser un dossier de sortie distinct par génération de run pour éviter les ambiguïtés

---

## 🧠 Baseline de prod recommandée (exemple)

```json
{
  "resume": "auto",
  "cache_retention": "run_end",
  "chunk_profile_mode": "safe_dynamic_plus",
  "parallel_autotune_enabled": true,
  "use_gpu_phase5": true,
  "stacking_normalize_method": "linear_fit",
  "stacking_rejection_algorithm": "winsorized_sigma_clip",
  "final_assembly_method": "incremental",
  "phase5_alpha_soft_weights": true,
  "phase5_alpha_weight_floor": 0.02,
  "patchwork_suppressor_enabled": true,
  "patchwork_suppressor_strength": "strong",
  "export_aesthetic_fits": true,
  "aesthetic_hole_fill_enabled": true,
  "aesthetic_profile_preset": "strong"
}
```

---

## 🧪 Dépannage

Si la résolution astrométrique ou le traitement échoue:

- Vérifier les chemins ASTAP (binaire + données)
- Vérifier que les images contiennent suffisamment d’étoiles et des headers FITS valides
- Inspecter les logs (`zemosaic_worker.log` et logs runtime)
- En cas de pression mémoire, réduire l’agressivité workers/chunk

---

## 🛠 Build (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
compile\build_zemosaic.bat
```

Exécutable de sortie:

- `dist/zemosaic.exe`

---

## 🙏 Crédits

- **Projet principal:** Tinystork (Tristan Nauleau)
- **Copilote engineering:** J.A.R.V.I.S. (ChatGPT)
- Inspiration méthodologique forte: travail d’intégration d’images **PixInsight** par **Juan Conejero** (pondération bruit, rejets robustes, concepts d’intégration).

---

## 📎 Licence

ZeMosaic est publié sous **GPL-3.0**.

---

## 🤝 Contributions

Issues et PR bienvenues. Si possible, inclure:
- logs,
- reproduction minimale,
- extrait de config,
- caractéristiques dataset (nombre d’images, résolution, caméra/profil).

---

🌠 Bonne mosaïque.
