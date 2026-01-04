# Follow-up — Validation & garde-fous (référence centrale master tiles)

## Checklist “ne pas casser”
- [x] Aucun changement sur le tri/ordering global des images (juste choix de référence).
- [x] Ne pas changer la logique `allow_batch_duplication` / `target_stack_size` / `min_safe_stack_size`.
- [x] `reference_image_index` envoyé à `align_images_in_group` est **toujours** un index dans `tile_images_data_HWC_adu`.
- [x] `ZMT_REF` (header) pointe vers le même raw que la référence (pas un autre index).
- [x] Après `allow_batch_duplication`, vérifier que la référence est retrouvée via un identifiant stable (ex: path_raw) et non un index qui pourrait devenir invalide.
- [x] Ne pas modifier la gestion du sémaphore `_PH3_CONCURRENCY_SEMAPHORE`. Tout `acquire` doit être suivi d'un `release` (généralement dans un `finally`), même en cas d'erreur ou de `return` prématuré.

## Tests manuels rapides
### 1) Cas nominal (cache OK)
- Lancer une mosaïque standard (Seestar EQ ou alt-az).
- Vérifier dans les logs `mastertile_info_reference_set`:
  - `ref_index_loaded` varie (pas toujours 0)
  - `ref_mode` = central (si loggé)
- Vérifier visuellement: moins de zones “vides” ou crops agressifs intra tuile (tendance).

### 2) Cas “cache manquant” (robustesse index)
- Simuler 1-2 caches absents (renommer temporairement quelques `.npy` dans le cache)
- Relancer sur une tuile concernée
Attendu:
- pas de crash `invalid_ref_index`
- si la réf préférée est absente: fallback sur une référence centrale parmi les images chargées. Si ce calcul de fallback échoue aussi, le choix final doit être l'index `0` de la liste chargée.

### 3) Cas “cache invalide”
- Injecter un `.npy` invalide (shape/dtype incorrect) pour une frame non-référence
Attendu:
- frame skip loggée
- pas de décalage d’index qui casse l’aligneur

### 4) Cas “aucune image chargée”
Ne pas modifier la gestion actuelle du cas où tile_images_data_HWC_adu est vide (aucun cache chargé) : conserver exactement le même comportement (abort / log) qu’avant, sans essayer d’inventer un ref_loaded_idx

### 5) Log et outils externes
- **Un seul log `*_info_reference_set` par tuile**: Le log `mastertile_info_reference_set` ne doit être émis qu'une seule fois par appel à `create_master_tile`, et il doit refléter les informations de la **référence finale** réellement utilisée (`ref_loaded_idx`, `ref_group_idx`). Éviter les logs multiples ou ambigus qui pourraient être générés avant que la référence finale ne soit choisie.
- Ne pas renommer ni supprimer les champs déjà présents dans mastertile_info_reference_set (ajouts seulement), pour ne pas casser d’éventuels parsers de logs externes.

### 6) Tests de modes spécifiques (anti-régression)
- **SDS success path** :
  - Lancer un projet avec `SeeStar_Stack_Method = sds`.
  - Attendu : vérifier dans les logs que la Phase 3 (`create_master_tile`) n'est **pas** appelée. La sortie doit être identique à la version précédente.
- **Existing master tiles mode** :
  - Lancer un projet qui réutilise des master tiles existants.
  - Attendu : vérifier que le log `run_info_existing_master_tiles_mode` s'affiche et que la Phase 3 est bien sautée (skip).
- **Grid mode run** :
  - Lancer un petit projet en mode grille (`grid_mode`).
  - Attendu : le traitement doit se terminer sans exception et les outputs doivent être cohérents (mosaïque assemblée).

## Tests unitaires (si la suite existe dans le repo)
Ajouter/adapter un test léger (sans IO) : [ ]
- Construire un faux groupe de 5 infos dict:
  - 1) ra/dec = (10, 0)
  - 2) ra/dec = (11, 0)
  - 3) ra/dec = (12, 0)  <-- central attendu
  - 4) ra/dec = (13, 0)
  - 5) ra/dec = (14, 0)
  + wcs fake: `is_celestial=True`
  + header fake: contient `CRVAL1/CRVAL2`
- Vérifier que `_pick_central_reference_index(infos, require_cache_exists=False)` renvoie index 2.

Test mapping:
- Simuler `loaded_group_indices = [0,1,3,4]` (index 2 manquant)
- Vérifier fallback central sur loaded ⇒ proche de 3 (selon distribution) mais surtout:
  - `ref_loaded_idx` ∈ [0..len(loaded)-1]

Test mapping "failed indices" :
- Rappel: `failed_alignment_indices` sont des indices dans la liste chargée (`tile_images_data_HWC_adu` / `loaded_infos`), pas dans `seestar_stack_group_info`.
- loaded_group_indices=[0,2,5]
- failed_alignment_indices=[1]
- attendu : retry_group contient seestar_stack_group_info[2] (pas [1])

## Perf
- **Recommandation unifiée**: Toujours utiliser `require_cache_exists=False` pour ne pas causer d'I/O disque (`os.path.exists`). N'activer cette option que si l'information sur l'existence du cache est déjà disponible en mémoire “gratuitement”.
- Vérifier que le choix central n’ajoute pas de lecture FITS.
- Complexité O(N) (N = taille du groupe), coût négligeable vs align+stack.
