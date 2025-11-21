
### 0. Contexte général

Projet : **ZeMosaic V4.1.x – Superacervandi**
Objectif de cette mission : ajouter une **option globale** dans la configuration + GUI Qt pour **supprimer les fichiers temporaires de traitement une fois le run terminé**, et ne les **conserver qu’en cas de debug**.

Actuellement :

* Le groupe *System resources & cache* du **Qt GUI** contient :

  * `coadd_use_memmap`
  * `coadd_cleanup_memmap`
  * `coadd_memmap_dir`
  * `cache_retention` (run_end / per_tile / keep)
* Le worker a déjà une fonction `_cleanup_memmap_artifacts()` qui :

  * nettoie certains fichiers `.dat` (`mosaic_*`, `coverage_*`, `zemosaic_*`) **dans le dossier memmap**,
  * efface certains répertoires `mosaic_first_*` **dans ce même dossier**,
  * supprime les fichiers WCS globaux `global_mosaic_wcs*.fits` / `.json` associés au plan global.
* En **Phase 7 (Nettoyage)**, le worker efface :

  * le cache des brutes pré-traitées (`temp_image_cache_dir`) selon `cache_retention`,
  * le dossier des **master tiles FITS** (`temp_master_tile_storage_dir`) seulement si `two_pass_enabled` est `False`,
  * le dossier SDS runtime (`sds_runtime_tile_dir`) est supprimé sans option centrale.

Limites actuelles :

* Certains artefacts demeurent sur disque **même après succès** :

  * répertoires `mosaic_first_<uuid>` (notamment dans le `runtime_temp_dir` ou ailleurs que le memmap dir),
  * dossiers de master tiles / super / mega tiles,
  * WCS globaux `global_mosaic_wcs.fits` / `.json` quand le nettoyage memmap n’est pas activé,
  * fichiers `.dat` comme `coverage_3261x2388.dat`, `mosaic_3261x2388x3.dat` hors du scope strict actuel.
* Le comportement dépend de plusieurs flags spécifiques (`coadd_use_memmap`, `coadd_cleanup_memmap`, `cache_retention`, etc.) mais **l’utilisateur n’a pas un bouton unique** “je veux que tout ce qui est temporaire disparaisse après le run”.

### 1. Objectif fonctionnel

Ajouter une **nouvelle option globale** :

> **Checkbox Qt (System)** :
> `qt_field_cleanup_temp_artifacts`
> Libellé EN : **“Delete temporary processing files after run”**
> Libellé FR : **“Supprimer les fichiers temporaires de traitement après l’exécution”**

* Cette option doit :

  * être **activée par défaut** (`True`),
  * contrôler un nouveau flag de config (ex. `cleanup_temp_artifacts`),
  * agir comme **master switch** pour la suppression des artefacts de traitement **non utiles après la fin du run**.
* Quand **cochée** : on supprime automatiquement les fichiers/dossiers temporaires listés plus bas.
* Quand **décochée** : on **conserve** ces artefacts pour debug/inspection (sans toucher aux sorties finales classiques : FITS/PNG de la mosaïque finale).

### 2. Artefacts à traiter

La nouvelle option doit piloter la suppression **après la fin du traitement** (Phase 7) des éléments suivants :

1. **Répertoires `mosaic_first_*`**

   * Dans le **memmap directory** (si configuré) – déjà partiellement géré par `_cleanup_memmap_artifacts`.
   * Dans le **runtime temp dir global** (`get_runtime_temp_dir()`) ou tout autre emplacement où ces dossiers sont créés.
2. **Fichiers memmap `.dat`** **inutiles après run** :

   * fichiers `mosaic_*.dat` (ex: `mosaic_3261x2388x3.dat`),
   * fichiers `coverage_*.dat` (ex: `coverage_3261x2388.dat`),
   * fichiers `zemosaic_*.dat`.
3. **Fichiers WCS globaux temporaires** :

   * `global_mosaic_wcs.fits`,
   * `global_mosaic_wcs.json`,
   * et tout fichier dont le nom contient `global_mosaic_wcs` généré comme artefact interne, dans le dossier de sortie ou dans le chemin du plan WCS global.
   * ⚠ Important : **ne pas supprimer** les fichiers WCS dont le nom **ne contient pas** `global_mosaic_wcs` (cases avancées où l’utilisateur fournit un nom spécifique pour archiver le WCS).
4. **Dossier des Master Tiles / Super/Mega Tiles** :

   * `temp_master_tile_storage_dir` doit être supprimé **même si** `two_pass_enabled == True`, quand `cleanup_temp_artifacts` est activé, car les passes sont terminées et le dossier n’est plus nécessaire.
5. **Dossier SDS runtime (méga/super tiles)** :

   * `sds_runtime_tile_dir` doit être supprimé **uniquement** si `cleanup_temp_artifacts` est activé. Aujourd’hui il est toujours supprimé.

Règle d’or : **Ne jamais supprimer** :

* La mosaïque finale FITS/PNG/JPEG (quel que soit le format),
* Les sorties explicitement choisies par l’utilisateur (output folder),
* Les logs.

### 3. Plomberie de configuration

#### 3.1. Nouveau champ de config

Dans `zemosaic_config.py` :

* Ajouter un booléen `cleanup_temp_artifacts: bool = True` dans la structure principale de configuration (ou dict, selon l’implémentation). 
* S’assurer que :

  * au **chargement**, si la clé est absente, on retombe sur `True` (compatibilité ascendante),
  * à la **sauvegarde**, la clé est persistée comme les autres booléens (ex : `coadd_use_memmap`, `coadd_cleanup_memmap`, `cache_retention`, etc.).

#### 3.2. Passage au worker

* Le worker central dans `zemosaic_worker.py` lit déjà une copie de la config (`worker_config_cache`). 

* Introduire un bool interne, par ex. :

  ```python
  cleanup_temp_artifacts_config = bool(
      (worker_config_cache or {}).get("cleanup_temp_artifacts", True)
  )
  ```

* Ce flag doit être accessible :

  * au moment d’appeler `_cleanup_memmap_artifacts()`,
  * durant la **Phase 7** quand on gère caches, master tiles et SDS runtime.

### 4. GUI Qt – System / Cache Tab

Fichier : `zemosaic_gui_qt.py`

Dans la méthode `_create_system_resources_group` :

* Ajouter une **nouvelle checkbox** après les options memmap, avant ou après `cache_retention` (à ton choix, mais dans le même group box) :

  * Clé config : `"cleanup_temp_artifacts"`
  * Texte : `self._tr("qt_field_cleanup_temp_artifacts", "Delete temporary processing files after run")`

* Utiliser `_register_checkbox` comme pour :

  * `coadd_use_memmap`
  * `coadd_cleanup_memmap`

Exemple conceptuel (ne pas copier tel quel, adapter au style existant) :

```python
self._register_checkbox(
    "cleanup_temp_artifacts",
    layout,
    self._tr(
        "qt_field_cleanup_temp_artifacts",
        "Delete temporary processing files after run",
    ),
    default_checked=True,  # si supporté par le helper ; sinon gérer via config default
)
```

Le champ doit être correctement relié au dict `self.config` pour être pris en compte dans le worker comme les autres options.

### 5. Worker – Logiciel de nettoyage

Fichier : `zemosaic_worker.py` 

#### 5.1. Refactor de `_cleanup_memmap_artifacts`

* Modifier `_cleanup_memmap_artifacts()` pour qu’elle :

  * **Reçoive** explicitement les paramètres nécessaires, plutôt que d’utiliser uniquement des variables externes :

    * `coadd_use_memmap_config`
    * `coadd_cleanup_memmap_config`
    * `coadd_memmap_dir_config`
    * `worker_config_cache`
    * `output_folder`
    * `global_wcs_plan`
    * `cleanup_temp_artifacts_config`
    * éventuellement `runtime_temp_dir` via `get_runtime_temp_dir()`
  * OU, si tu préfères, lire `cleanup_temp_artifacts_config` comme variable fermée mais **ne pas casser** la signature actuelle pour le reste.

* Comportement souhaité :

  1. Si `cleanup_temp_artifacts_config` est **False** → **sortir immédiatement** de `_cleanup_memmap_artifacts()` (ne rien supprimer, y compris WCS globaux).
  2. Si `cleanup_temp_artifacts_config` est **True** :

     * Si `coadd_use_memmap_config` & `coadd_cleanup_memmap_config` & `coadd_memmap_dir_config` sont valides :

       * garder la logique actuelle de nettoyage des `.dat` & `mosaic_first_*` dans le **memmap dir**.
     * En plus, scanner :

       * le `runtime_temp_dir` global (`get_runtime_temp_dir()`),
       * et supprimer tout répertoire `mosaic_first_*` trouvé à ce niveau (sans effacer d’autres sous-dossiers).
     * Nettoyage des WCS globaux `global_mosaic_wcs*.fits` / `.json` tel que déjà implémenté aujourd’hui, mais **sans exiger** que le bloc memmap ait tourné (donc même si memmap est désactivé).

#### 5.2. Phase 7 – Nettoyage général

Toujours dans `zemosaic_worker.py` (bloc Phase 7 déjà présent)  :

* Laisser **inchangé** le comportement de `cache_retention` pour le cache de brutes (temp_image_cache_dir) :

  * `cache_retention == "keep"` → on garde le cache,
  * sinon → on supprime le cache comme aujourd’hui.
  * Ce comportement est **indépendant** de `cleanup_temp_artifacts` (donc même si `cleanup_temp_artifacts` est False, si l’utilisateur a choisi de ne pas garder le cache, on le nettoie).
* Ajouter l’usage de `cleanup_temp_artifacts_config` pour :

  1. **Master tiles / super tiles** (`temp_master_tile_storage_dir`) :

     * Aujourd’hui : on ne les supprime que si `two_pass_enabled` est False.
     * Nouveau : si `cleanup_temp_artifacts_config` est **True**, supprimer **toujours** ce dossier s’il existe, indépendamment de `two_pass_enabled`.

       * Si besoin, garder un log de type `run_info_temp_master_tiles_fits_cleaned` déjà existant.
     * Si `cleanup_temp_artifacts_config` est **False**, ne pas supprimer ce répertoire (même si `two_pass_enabled` est False).
  2. **SDS runtime dir** (`sds_runtime_tile_dir`) :

     * Aujourd’hui : toujours supprimé via `shutil.rmtree(..., ignore_errors=True)`. 
     * Nouveau : entourer cet appel d’un test `if cleanup_temp_artifacts_config:` :

       * `True` → comportement actuel (supprimer),
       * `False` → **ne rien faire** (donc le dossier SDS reste pour debug).
  3. S’il existe d’autres répertoires temporaires clairs (**runtime tiles, caches mosaïque-first, etc.**) qui ne contiennent **jamais** les sorties finales de l’utilisateur, ils peuvent être inclus dans ce même master switch.

### 6. Localisation / Traductions

Fichiers : `locales/en.json`, `locales/fr.json`, et toutes les autres langues disponibles (es, pl, de, nl, is, …).

Tu DOIS :

1. Ajouter la **clé d’interface** suivante dans **tous les fichiers de langue** :

   * `"qt_field_cleanup_temp_artifacts": "Delete temporary processing files after run"` (EN)
   * `"qt_field_cleanup_temp_artifacts": "Supprimer les fichiers temporaires de traitement après l’exécution"` (FR)
   * Pour les autres langues (ES/PL/DE/NL/IS…), soit :

     * fournir une vraie traduction si tu es sûr,
     * soit **dupliquer le texte anglais** (mais ne jamais laisser la clé manquante).

2. Vérifier qu’aucun nouvel identifiant (clé Qt ou log_key) n’est utilisé sans être présent dans **tous** les JSON de localisation.
   Si tu ajoutes d’autres log messages, ils doivent suivre la même règle.

### 7. Invariants à respecter

* Ne rien casser dans le comportement de **stacking**, SDS, Phase 4.5, two-pass renorm, GPU, etc.
* Ne pas modifier le comportement de :

  * `coadd_use_memmap` (active juste l’usage memmap),
  * `coadd_cleanup_memmap` (contrôle la suppression des `.dat` *quand* `cleanup_temp_artifacts` est `True`),
  * `cache_retention`.
* Ne jamais supprimer les fichiers finaux du dossier de sortie :

  * mosaïque finale FITS / PNG / JPG,
  * preview finale,
  * logs.
* Ne pas introduire de dépendance supplémentaire lourde (pas de nouvelle lib externe).

---

