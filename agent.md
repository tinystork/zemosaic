# Mission — Master Tiles: Référence centrale (safe, déterministe, sans surcoût)

## Objectif
Lors de `create_master_tile` (Phase 3), remplacer le choix implicite `reference_image_index_in_group = 0` par un choix **référence centrale** (meilleur compromis recouvrement / crop / interpolation),
tout en restant:
- **déterministe** (même dataset ⇒ même référence)
- **O(N)** (pas de O(N²), pas d’IO additionnel lourd)
- **sans régression** sur les cas où certaines entrées ne sont pas chargées (cache manquant / cache invalide).

## Contrainte critique (anti-bug)
⚠️ `align_images_in_group` attend un `reference_image_index` qui est un index dans la **liste chargée** (`tile_images_data_HWC_adu`), pas dans `seestar_stack_group_info`.
Or `tile_images_data_HWC_adu` peut être plus courte (cache manquant / data invalide / skip).
Donc la mission DOIT:
- calculer un **index de référence “group”** (dans `seestar_stack_group_info`)
- puis le **mapper** vers un **index de référence “loaded”** (dans `tile_images_data_HWC_adu`)
- et si la référence “group” n’a pas été chargée, choisir une référence centrale **parmi les chargées** (fallback).

- Mapping explicite obligatoire pour `failed_alignment_indices` (voir Implémentation, section 3).
- Ordre strict : si `tile_images_data_HWC_adu` est vide ⇒ abort comme aujourd’hui, avant tout calcul de `ref_loaded_idx` / `ref_group_idx` / `ref_info_for_tile`.
- WCS/Header : validation uniquement à partir de ref_info_for_tile = loaded_infos[ref_loaded_idx].

## Portée (scope)
- Fichier principal: `zemosaic_worker.py`
- Fonction: `create_master_tile`
- Aucun changement GUI.
- Aucun ajout de clés i18n/locales (pas de nouvelles strings obligatoires).
- Ne pas modifier le comportement “batch size = 0” et “batch size > 1” (intouchable).

## Stratégie de sélection “centrale”
On choisit l’image dont le centre (RA/DEC) est le plus proche du centroïde du groupe.
Implémentation recommandée (robuste RA wrap):
- convertir RA/DEC → vecteurs unitaires (x,y,z)
- moyenne des vecteurs → vecteur centroïde normalisé
- score = 1 - dot(v, centroïde)
- prendre le minimum (tie-break: plus petit index original)

### Sources de RA/DEC
Réutiliser les helpers existants:
- `_extract_ra_dec_deg(info)` (déjà robuste: WCS pixel_to_world puis CRVAL fallback)
- `_unit_vector_from_ra_dec(ra_deg, dec_deg)`

### Candidats valides (pour éviter des crashs)
Ne considérer “candidat” que si:
- `info` est un `dict`
- `info.get("wcs")` existe et `wcs.is_celestial == True`
- `info.get("header")` non vide
- et `_extract_ra_dec_deg(info)` retourne bien (ra, dec)
Optionnel mais recommandé côté “group” (pré-sélection): exiger aussi que `path_preprocessed_cache` existe sur disque (évite de choisir une réf qui ne chargera jamais).

## Implémentation (pas à pas)

### 1) [x] Ajouter un helper local (dans zemosaic_worker.py)
Ajouter une fonction utilitaire (près de `create_master_tile` ou en helpers) :

`_pick_central_reference_index(infos: list[dict], require_cache_exists: bool) -> int | None`

- **Garde-fou “jamais de crash”**: La fonction DOIT être encapsulée dans un `try/except` global. Si un calcul échoue (ex: `NaN`, données non-finies), elle doit immédiatement retourner un fallback sûr (`None` ou `0` selon le contexte d'appel) pour éviter tout crash.
- Renvoie un index dans la liste `infos`
- Retourne `None` si aucun candidat valide
- Tie-break déterministe: (score, index)

Pseudo:
- construire `candidates = [(idx, ra, dec)]`
- centroïde via somme des unit vectors
- sélectionner meilleur idx
- si aucun candidat:
  - fallback: premier idx qui a wcs+header (et cache si require_cache_exists)
  - sinon `None`

### 2) [x] Dans create_master_tile: calculer un “preferred_group_idx”
Pour éviter de biaiser le choix en cas de duplication (`allow_batch_duplication`), le `preferred_group_idx` est calculé sur la liste originale (**avant** la duplication), puis l'index est mappé sur la première occurrence correspondante dans la liste dupliquée.

L'opération se déroule donc ainsi :
- Calculer `preferred_group_idx` sur `seestar_stack_group_info` avant sa modification.
- Si `None`, fallback `preferred_group_idx = 0`.
- Après `allow_batch_duplication`, retrouver la nouvelle position de cet index.

⚠️ MUST NOT fix `wcs_for_master_tile` / `header_for_master_tile_base` at this stage (car la réf peut ne pas être chargée finalement).

### 3) [x] Pendant le chargement cache: conserver un mapping group→loaded
Dans la boucle `for i, raw_file_info in enumerate(seestar_stack_group_info):`
quand un cache est effectivement chargé et validé:
- `tile_images_data_HWC_adu.append(img_data_adu)`
- `tile_original_raw_headers.append(raw_file_info.get("header"))`
- AJOUTER:
  - `loaded_infos.append(raw_file_info)`
  - `loaded_group_indices.append(i)`

Puis construire après la boucle:
- `group_to_loaded = {group_i: loaded_i for loaded_i, group_i in enumerate(loaded_group_indices)}`

#### Mapping des `failed_alignment_indices` (obligatoire)
Lors de la construction de `retry_group`, considérer chaque `idx_fail` comme un index dans la liste chargée (`tile_images_data_HWC_adu` / `loaded_infos`), pas comme un index dans `seestar_stack_group_info`.

Implémentation attendue :
- Vérifier `0 <= idx_fail < len(loaded_group_indices)`.
- Calculer `group_idx = loaded_group_indices[idx_fail]`.
- Utiliser ensuite `group_idx` pour récupérer `seestar_stack_group_info[group_idx]` lors de la construction de `retry_group`.

Exemple :
- `loaded_group_indices = [0, 2, 5]`
- `failed_alignment_indices = [1]`
- ⇒ `group_idx = loaded_group_indices[1] = 2` ⇒ on pousse bien `seestar_stack_group_info[2]` dans `retry_group`.

#### Ordre exact — cas “aucune image chargée”
Ne pas déplacer le bloc existant qui abort quand aucune image n’a été chargée :
- Conserver le `return (None, None), failed_groups_to_retry` sous le test `if not tile_images_data_HWC_adu:`.
- Le calcul de `ref_loaded_idx` / `ref_group_idx` / `ref_info_for_tile` doit se faire après ce `if`, mais avant l’appel à `align_images_in_group`.

### 4) [x] Déterminer la référence finale “loaded”
- Si `preferred_group_idx` est dans `group_to_loaded`:
  - `ref_loaded_idx = group_to_loaded[preferred_group_idx]`
- Sinon:
  - choisir une référence centrale parmi `loaded_infos`:
    - `ref_loaded_idx = _pick_central_reference_index(loaded_infos, require_cache_exists=False)`
  - si `None`, fallback `ref_loaded_idx = 0`

Ensuite:
- `ref_group_idx = loaded_group_indices[ref_loaded_idx]`
- `ref_info_for_tile = loaded_infos[ref_loaded_idx]`

### 5) [x] Définir WCS/Header à partir de la référence finale
Remplacer la logique actuelle qui fait:
- `ref_info_for_tile = seestar_stack_group_info[reference_image_index_in_group]`
par celle basée sur `ref_info_for_tile` (chargée).
Si reference_image_index_in_group existe encore dans la fonction, il ne doit plus être utilisé que pour du log ou de la compatibilité interne ; l’unique index passé à align_images_in_group est ref_loaded_idx.

Définir:
- `wcs_for_master_tile = ref_info_for_tile.get("wcs")`
- `header_dict_for_master_tile_base = ref_info_for_tile.get("header")`
et garder la validation existante:
- si pas (wcs celestial + header), erreur propre + abort tuile

### 6) [x] Appel aligner avec l’index “loaded”
Appeler `align_images_in_group` en passant `reference_image_index=ref_loaded_idx`.
⚠️ plus jamais passer un index “group” à l’aligneur.

### 7) [x] Sauvegarde FITS: ZMT_REF doit correspondre à la même référence
Dans la partie sauvegarde header, remplacer toute dérivation de `ZMT_REF` (et champs associés) à partir de `seestar_stack_group_info[reference_image_index_in_group]` par une dérivation à partir de `ref_info_for_tile` (chargée), par ex. via `ref_info_for_tile.get("path_raw")`.

But: le FITS final doit annoncer comme référence exactement celle utilisée pour WCS/base.

### 8) [x] Logs (sans nouvelle clé locale obligatoire)
Ne pas ajouter de nouvelles clés i18n.
Tu peux enrichir le log existant `mastertile_info_reference_set` en passant:
- `ref_index_group=ref_group_idx`
- `ref_index_loaded=ref_loaded_idx`
- `ref_mode="central"`
mais garder la clé existante pour éviter de toucher aux locales.

## Critères d’acceptation
- Sur un dataset normal: la référence n’est plus systématiquement 0 (souvent proche du centre).
- Aucun crash quand certains caches sont manquants/invalides (ref index toujours valide dans liste chargée).
- Le FITS master tile contient `ZMT_REF` cohérent avec la référence réellement utilisée.
- Pas de changement de comportement ailleurs (Phase 4/5 inchangées, GPU/CPU inchangés, batch behavior inchangé).

## Fichiers à modifier
- `zemosaic_worker.py` uniquement (mission minimale)
