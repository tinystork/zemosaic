## `agent.md`

### Mission

Restaurer le comportement **classique** (hors Grid/SDS) de ZeMosaic pour les master tiles et la mosaïque finale :

* les master tiles doivent repasser par **lecropper.py** (quality crop + alt-az cleanup + alpha) **exactement comme dans le worker “classique” qui donne une couleur correcte** ;
* la mosaïque finale **ne doit plus subir de ré-égalisation RGB agressive** qui casse la chromie (gains R≈4.6, etc.) ;
* Grid mode / SDS ne doivent pas être cassés.

Objectif visible côté utilisateur :

* en mode classique, la mosaïque finale doit avoir un histogramme RGB similaire aux master tiles et à Grid mode (courbes serrées, pas de dominante verte/rouge, pas de tuiles “fantômes” non croppées).

---

### Contexte technique (résumé)

* Fichier central : `zemosaic_worker.py`.

* Le flux **Phase 3** (master tiles) effectue déjà :

  * stacking + `poststack_equalize_rgb` (OK) ;
  * `apply_center_out_normalization_p3` (OK) ;
  * un bloc **quality crop + WCS shift** basé sur `lecropper.detect_autocrop_rgb` (présent, mais il faut s’assurer que la voie classique l’utilise toujours correctement) ;
  * un appel à `_apply_lecropper_pipeline(...)` qui applique `quality_crop`, `altaz_cleanup`, et fabrique un **alpha mask normalisé**.

* Un pipeline équivalent existe pour la mosaïque finale : `_apply_final_mosaic_quality_pipeline(...)` + `_apply_master_tile_crop_mask_to_mosaic(...)`. 

* Un second étage, plus récent, applique une **égalisation RGB finale sur la mosaïque** via `_apply_final_mosaic_rgb_equalization(...)`, avec des gains extrêmes (ex. `gains=(4.6025, 0.9736, 1.0000)`), ce qui débalance complètement les canaux. C’est cette étape qu’on veut **neutraliser proprement** pour le moment.

* Un worker “classique” sans Grid (fourni hors repo) montre que :

  * en laissant `poststack_equalize_rgb` + `center_out` + lecropper sur les master tiles,
  * et **sans égalisation RGB finale** sur la mosaïque,
  * la voie classique donne un rendu couleur propre (pas de dominante verte, pas de rouge saturé), avec des tuiles bien croppées.

---

### Scope

**Fichiers à modifier :**

* `zemosaic_worker.py` (obligatoire)
* éventuellement `zemosaic_config.py` si un flag de config propre est nécessaire pour activer/désactiver l’égalisation RGB finale.

**Fichiers à NE PAS modifier :**

* `grid_mode.py` et tout ce qui concerne spécifiquement Grid mode.
* Le code de la GUI (Tk / Qt).
* `lecropper.py` (sauf bug bloquant évident, mais en principe inutile).
* La logique de stacking GPU/CPU (sauf si un bug est directement lié à cette mission).

---

### Exigences fonctionnelles

#### 1. Master tiles (Phase 3, voie classique)

1.1. **Garantie que le pipeline lecropper est appliqué à chaque master tile** dans la voie classique (hors SDS/Grid) :

* À la fin de `create_master_tile(...)`, juste avant la sauvegarde FITS, on doit avoir *toujours* :

  * le bloc `quality_crop` basé sur `detect_autocrop_rgb` (si `quality_crop_enabled` est vrai), avec mise à jour de `wcs_for_master_tile` ;
  * puis `pipeline_cfg = {...}` ;
  * puis `master_tile_stacked_HWC, pipeline_alpha_mask = _apply_lecropper_pipeline(...)` ;
  * puis `_normalize_alpha_mask(...)` et passage de `alpha_mask_out` à `zemosaic_utils.save_fits_image(...)`.

* Si la voie Grid/SDS a des chemins conditionnels spécifiques, s’assurer que **la voie classique** continue à exécuter ce pipeline sans être court-circuitée :

  * pas de `if grid_mode: return ...` avant `_apply_lecropper_pipeline(...)` ;
  * pas de condition qui met `quality_crop_enabled=False` en voie classique par erreur.

1.2. **Compatibilité logs / GUI :**

* Conserver les logs existants :

  * `MT_CROP: quality-based rect=...` ;
  * avertissements `MT_CROP: quality crop skipped ...` si le crop est jugé inutile ;
* Ne pas modifier les clés `[CLÉ_POUR_GUI: ...]` ni la structure des callbacks.

#### 2. Mosaïque finale (Phase 5, voie classique)

2.1. **Désactiver l’égalisation RGB finale agressive** :

* Actuellement, le code applique `_apply_final_mosaic_rgb_equalization(...)` conditionnellement, typiquement :

  ```python
  if final_mosaic_data_HWC is not None and not sds_mode_phase5:
      final_mosaic_data_HWC, final_rgb_info = _apply_final_mosaic_rgb_equalization(...)
  ```

* Objectif : **ne plus appliquer cette étape par défaut** en voie classique, pour éviter les gains extrêmes qui démolissent la chromie.

* Implémenter un flag explicite, par exemple :

  * dans `zemosaic_config.py` : `final_mosaic_rgb_equalize_enabled: bool = False` (ou récupéré depuis la config utilisateur si elle existe déjà) ;
  * dans `zemosaic_worker.py` :

    * lire ce flag (ou valeur par défaut False) ;
    * entourer l’appel à `_apply_final_mosaic_rgb_equalization(...)` avec :

      ```python
      if (
          final_mosaic_rgb_equalize_enabled
          and final_mosaic_data_HWC is not None
          and not sds_mode_phase5
      ):
          ...
      ```

* **Par défaut** dans le repo : mettre ce flag à `False` (comportement sûr).

2.2. **Conserver le pipeline qualité lecropper pour la mosaïque** :

* Ne pas toucher à `_apply_final_mosaic_quality_pipeline(...)` ni `_apply_master_tile_crop_mask_to_mosaic(...)`, sauf pour corriger un bug avéré. 
* Vérifier que ces fonctions sont toujours appelées à la fin de la Phase 5 pour la voie classique, afin que :

  * les artefacts Alt-Az / bords soient bien nettoyés ;
  * `final_alpha_map` soit cohérent avec `final_mosaic_coverage`.

#### 3. Compatibilité Grid / SDS

* Ne pas modifier la logique propre à Grid mode ou SDS (flags `grid_mode`, `sds_mode_phase5`, options SDS, etc.).
* L’appel à `_apply_final_mosaic_rgb_equalization(...)` doit **rester désactivé en SDS** (comme actuellement) sauf si explicitement demandé par config (a priori non).
* La restauration du pipeline lecropper sur master tiles ne doit pas casser :

  * les chemins de stacking par super-tiles ;
  * la gestion des caches intermédiaires.

---

### Plan d’action proposé

- [x] **Analyser `create_master_tile(...)`** dans le `zemosaic_worker.py` actuel :

  * repérer le bloc `poststack_equalize_rgb` + `apply_center_out_normalization_p3` ;
  * confirmer la présence du bloc `quality_crop` + `_apply_lecropper_pipeline(...)` + alpha ;
  * s’assurer que ce bloc est exécuté en voie classique (hors Grid/SDS) et non conditionné par des flags Grid/SDS inappropriés.

- [ ] **Comparer avec le worker “classique” fourni (sans Grid)** :

  * si des différences existent sur la partie lecropper (quality_crop/altaz/alpha), les harmoniser en faveur de la version qui fonctionne (celle du worker classique).

- [x] **Isoler l’appel à `_apply_final_mosaic_rgb_equalization(...)`** :

  * créer un flag de config `final_mosaic_rgb_equalize_enabled` (ou équivalent) ;
  * désactiver l’appel par défaut (flag False) ;
  * laisser le code de la fonction tel quel pour pouvoir la réactiver plus tard si besoin, mais **ne pas l’appeler en pratique**.

- [x] **S’assurer que `_apply_final_mosaic_quality_pipeline(...)` et `_apply_master_tile_crop_mask_to_mosaic(...)` restent en place** :

  * vérifier l’ordre d’appel en fin de Phase 5 ;
  * garantir qu’ils ne sont pas conditionnés par le flag de RGB equalization (ce sont des pipelines orthogonaux).

- [ ] **Mettre à jour les commentaires/docstrings** au besoin pour documenter :

  * que la couleur finale repose sur `poststack_equalize_rgb` + `center_out` au niveau master tiles ;
  * que l’égalisation RGB mosaïque est optionnelle et désactivée par défaut.

---

### Tests / Validation attendus

1. **Test M106 voie classique (dataset déjà utilisé)** :

   * lancer un run complet **hors Grid/SDS** ;
   * vérifier dans le log :

     * présence de `[RGB-EQ] poststack_equalize_rgb ...` pour chaque master tile ;
     * présence de lignes `MT_CROP: quality-based rect=...` (sauf si crop inutile) ;
     * **absence** de ligne `[RGB-EQ] final mosaic: applied=True, gains=...` (ou équivalent) ;
   * ouvrir la mosaïque finale dans un viewer :

     * histogramme RGB proche de celui des master tiles ;
     * pas de dominante verte ou rouge violente ;
     * pas de bandes/carreaux non croppés autour des tuiles.

2. **Test Grid mode (M106 ou dataset simple)** :

   * lancer un run Grid mode ;
   * confirmer que :

     * le comportement actuel de Grid n’est pas dégradé ;
     * pas de crash ni de changement de couleur inattendu.

3. **Test SDS (si facilement accessible)** :

   * lancer un run SDS simple ;
   * vérifier que les master tiles sont toujours croppées/masquées correctement ;
   * pas de modification inattendue des logs ou du flux.

4. **Régression rapide sur petit dataset mono** (si possible) :

   * pour s’assurer que le pipeline lecropper gère toujours les cas mono-canal (pas uniquement RGB).

---

### Contraintes / style

* Ne pas introduire de nouvelles dépendances.
* Conserver la compatibilité Python actuelle.
* Garder les messages de log existants ; ajouter de nouveaux logs uniquement si utiles pour le debug.
* Ne jamais lever une exception fatale si `lecropper` n’est pas dispo : dans ce cas, le pipeline doit se désactiver proprement (comportement déjà implémenté, à respecter).

