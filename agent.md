# agent.md — ZeMosaic V1 Resume (après Phase 1)

## Contexte
ZeMosaic (mode classic legacy) supprime actuellement systématiquement `.zemosaic_img_cache` au début de `run_hierarchical_mosaic_classic_legacy()`, ce qui empêche toute reprise.
Objectif V1 : permettre de **reprendre un run après la Phase 1** si un cache valide existe, en gardant un comportement **strictement identique** quand la reprise est désactivée.

## Objectif (V1)
Ajouter une reprise **après Phase 1** via `.zemosaic_img_cache` :
- [x] Si `.zemosaic_img_cache` + un **manifest** + un **marker Phase 1** existent et sont valides → **skip Phase 1**, reprendre directement à la Phase 2.
- [x] Sinon → comportement actuel inchangé (run complet avec suppression/recréation du cache au début).

## Périmètre (anti-régression)
✅ CIBLE : `zemosaic_worker.py` → fonction `run_hierarchical_mosaic_classic_legacy()`

🚫 HORS PÉRIMÈTRE V1 :
- Ne pas modifier SDS / grid mode / autres pipelines.
- Ne pas implémenter la reprise Phase 2 ou Phase 3 (ce sera V2/V3).
- Ne pas changer le comportement existant quand `resume=off`.

## Contraintes clés
1) **Par défaut : aucune régression**
- Nouveau paramètre config `resume` (string) ∈ `{ "off", "auto", "force" }`
- Valeur par défaut : `"off"` si absent/invalid.
- Si `resume == "off"` → laisser le code se comporter EXACTEMENT comme aujourd’hui (notamment suppression/recréation de `.zemosaic_img_cache` au début).

2) **Garde-fous de mode**
La reprise V1 doit être désactivée (comme si `resume=off`) si l’un de ces cas est vrai :
- `sds_mode_flag` est actif
- `use_existing_master_tiles_config` est actif (ou `use_existing_master_tiles_mode` est détecté)
- tout autre mode non-classic legacy (si détecté)

3) **Pas de pickle**
Le cache de reprise doit être écrit en JSON (manifest + data), pas de pickle.

## Nouveaux artefacts (dans `.zemosaic_img_cache/`)
Créer uniquement si `resume != off` ET si la Phase 1 s’exécute (donc run “producteur de cache”).

- [x] `cache_manifest.json`
- [x] `phase1_processed_info.json`
- [x] `phase1.done`

### `cache_manifest.json` (schema minimal V1)
Contenu minimal recommandé :
```json
{
  "schema_version": 1,
  "pipeline": "classic_legacy",
  "created_utc": "...",
  "run_signature": "<sha256 hex>",
  "input_folder_norm": "...",
  "output_folder_norm": "...",
  "phase1": {
    "done": true,
    "done_marker": "phase1.done",
    "processed_info_file": "phase1_processed_info.json",
    "num_entries": 1234
  }
}
````

### `phase1_processed_info.json`

Liste JSON de dicts, un par image valide, contenant uniquement des champs sérialisables + de quoi reconstruire les objets nécessaires aux phases suivantes :
Champs obligatoires par entrée :

* `path_raw` (str, chemin absolu original)
* `path_preprocessed_cache` (str, chemin absolu vers le `.npy` cache)
* `path_hotpix_mask` (str ou null)
* `preprocessed_shape` (liste d’int)
* `header_str` (str) : header FITS complet **mis à jour** (celui qui permet de reconstruire le WCS)
  Champs optionnels à conserver si présents dans `entry` actuel :
* `phase0_index`, `phase0_center`, `phase0_shape`, `phase0_wcs` (si déjà injectés)

IMPORTANT :

* `header_str` doit permettre une reconstruction fiable via `astropy.io.fits.Header.fromstring(...)`
* On ne stocke PAS les objets `wcs` ni `header` directement (non sérialisables).

## Run signature (V1)

Implémenter une fonction de hash déterministe (sha256) sur un JSON canonique (keys triées).
Inclure au minimum :

* [x] pipeline: `"classic_legacy"`
* [x] input fingerprint: liste triée des fichiers FITS du `input_folder` (chemins relatifs) + (size, mtime)
* [x] paramètres ASTAP (radius/downsample/sensitivity) + solver timeout si utilisé en Phase 1
* [x] tout paramètre structurant de Phase 1 si facilement accessible
* (optionnel) une version pipeline si dispo

BUT : si l’utilisateur ajoute/retire des fichiers bruts ou change des options → signature ≠ → reprise refusée (sauf force).

## Nouvelle logique de reprise (V1)

### Ajouter un helper `try_resume_phase1(...)`

Rôle :

* [x] détecter `.zemosaic_img_cache`
* [x] lire/valider `cache_manifest.json` + `phase1.done`
* [x] recalculer `run_signature_current` (via scan input_folder)
* [x] si `resume=="auto"` : exiger signature match
* [x] si `resume=="force"` : ignorer mismatch signature MAIS exiger présence des fichiers essentiels
* [x] vérifier que toutes les entrées dans `phase1_processed_info.json` pointent vers des fichiers existants (`path_preprocessed_cache` au minimum)
* [x] si OK : charger la liste et reconstruire en mémoire les champs requis par les phases suivantes :

  * [x] `header = fits.Header.fromstring(header_str, sep="\n")`
  * [x] `wcs = astropy.wcs.WCS(header)`
  * [x] injecter `entry["header"]=header`, `entry["wcs"]=wcs`
  * [x] supprimer `header_str` du dict en mémoire (optionnel)

Retour :

* [x] `resume_ok: bool`
* [x] `loaded_all_raw_files_processed_info: list[dict] | None`
* [x] `reason: str` (pour log)

### Placement dans `run_hierarchical_mosaic_classic_legacy()`

À l’endroit où le code gère actuellement :

```py
cache_dir_name = ".zemosaic_img_cache"
temp_image_cache_dir = ...
if _path_exists(temp_image_cache_dir): shutil.rmtree(temp_image_cache_dir)
os.makedirs(temp_image_cache_dir, exist_ok=True)
```

Modifier ainsi :

* [x] Calculer `resume_mode` (`off/auto/force`) depuis `worker_config_cache.get("resume")` (et éventuellement `filter_overrides["resume"]` si fourni).
* [x] Si `resume_mode == "off"` → garder EXACTEMENT le bloc actuel (rmtree + mkdir).
* [x] Sinon :

  1. [x] Tenter `try_resume_phase1(...)`
  2. [x] Si reprise acceptée :

     * [x] NE PAS supprimer `.zemosaic_img_cache`
     * [x] définir un flag local `resume_after_phase1 = True`
     * [x] définir `all_raw_files_processed_info = loaded_list`
     * [x] ajuster la progression pour être cohérente :

       * [x] logger un message INFO “Phase 1 skipped (resume)”
       * [x] avancer `current_global_progress` comme si Phase 1 était finie :
         `current_global_progress = base_progress_phase1 + PROGRESS_WEIGHT_PHASE1_RAW_SCAN`
  3. [x] Si reprise refusée :

     * [x] renommer le cache en `.zemosaic_img_cache_<timestamp>.old` (préféré) OU supprimer, puis recréer
     * [x] continuer run normal

Ensuite :

* [x] Le bloc “Phase 1” (`# --- Phase 1 ...`) doit être conditionné :

  * [x] Phase 1 s’exécute uniquement si `not use_existing_master_tiles_mode` ET `not resume_after_phase1`.

### Écriture du cache de reprise (fin Phase 1)

Juste après le log `run_info_phase1_finished_cache` :

* [x] si `resume_mode != "off"` :

  * [x] écrire `phase1_processed_info.json` (liste sérialisable avec `header_str`)
  * [x] écrire `cache_manifest.json`
  * [x] créer `phase1.done`

- [x] Ne pas faire échouer le run si l’écriture du manifest échoue : log WARN, puis continuer.

## Logs

* [x] Utiliser `pcb("...")` avec un message direct string (pas besoin d’ajouter des clés i18n).
* [x] Logs requis :

  * [x] resume demandé + mode (`auto/force`)
  * [x] resume accepté + nb d’entrées
  * [x] resume refusé + raison
  * [x] si force : avertissement clair quand signature mismatch ignorée

## Tests / Validation minimale (sans framework)

Ajouter au moins une petite fonction de validation interne (ou bloc test manuel) n’est pas requis, MAIS le code doit être structuré pour être testable.
Pas de modifications des tests existants demandées en V1.

## Fichiers à modifier

* [x] `zemosaic_worker.py` uniquement (V1)

  * [x] ajout helpers (signature, manifest read/write, try_resume_phase1)
  * [x] patch dans `run_hierarchical_mosaic_classic_legacy()`

## Critères d’acceptation

1. Avec `resume` absent → comportement identique à avant (cache supprimé au début).
2. Avec `resume="auto"` + cache valide :

   * Phase 1 est sautée
   * Phase 2 démarre avec `all_raw_files_processed_info` reconstruit (WCS OK)
3. Avec `resume="auto"` + cache invalide/mismatch :

   * reprise refusée
   * pipeline normal continue (cache clean)
4. Avec `resume="force"` + signature mismatch MAIS fichiers présents :

   * reprise acceptée avec WARN
5. Aucun changement SDS/grid/existing-master-tiles : reprise désactivée dans ces cas.

