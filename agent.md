# agent.md

## Mission
Corriger deux problèmes en Phase 3 (Master Tiles) de ZeMosaic, introduits/renforcés par les mécanismes de protection mémoire :

1) **Master tiles entièrement vides** (noires / NaNisées) “quelque soit la normalisation” alors qu’avant ce n’était pas le cas.
   - Suspect principal : **ALT-AZ cleanup / mask propagation** appliqué à des groupes **EQ** (ou appliqué globalement au lieu d’être appliqué *par groupe*).
   - Contexte : l’UI Qt (zemosaic_filter_gui_qt.py) construit déjà des clusters séparés **EQ vs ALT_AZ** (pas de mélange), donc le worker doit respecter cette séparation.

2) **Crash mémoire CPU kappa-sigma** :
   - Trace typique :
     `_cpu_stack_kappa_fallback -> np.nanmedian(arr, axis=0)` puis
     `Unable to allocate 32.4 GiB for an array with shape (N,H,W,3) float32`.
   - Ici, le fallback CPU fait un `np.stack` de *toutes* les frames en RAM (N,H,W,C), ce qui explose sur gros groupes.

Objectif : corriger ces deux points **sans changer le comportement scientifique** (résultat du stack) pour les cas qui ne crashaient pas, et **sans régression** pour **SDS mode** et **Grid/Survey mode**.

## Contraintes critiques
- **Zéro régression SDS** : ne pas modifier le pipeline SDS (ni sa logique d’assemblage). Toute modification doit être neutre pour SDS.
- **Zéro régression Grid mode** : grid mode est géré par `grid_mode.run_grid_mode(...)`. Ne pas modifier grid_mode.
- Ne pas changer l’API publique ni les noms d’options de GUI.
- Patch minimal, ciblé.
- Préserver la parité CPU/GPU (là où applicable) : la correction doit être “même résultat, moins de RAM”.
- Le dataset peut contenir un mix EQ + ALT_AZ : il faut appliquer ALT-AZ cleanup **uniquement** aux groupes ALT_AZ (et jamais aux groupes EQ).

## Périmètre fichiers (autorisé)
- `zemosaic_worker.py`
- `zemosaic_align_stack.py`

(Pas d’autres fichiers, sauf si absolument indispensable pour un bug bloquant — et dans ce cas, justifier dans le PR.)

## Diagnostic attendu (root causes)
### A) Master tiles vides
Dans `zemosaic_worker.py`, Phase 3 appelle `create_master_tile(...)` en passant `altaz_cleanup_effective_flag` **global** à *tous* les groupes.
Or :
- `create_master_tile()` active `propagate_mask_for_coverage` dès que `altaz_cleanup_enabled_effective` est vrai,
- et la pipeline lecropper peut “nanize/zeroize” selon un masque,
- donc si tu appliques ça à un stack EQ (ou à un groupe dont la couverture/masque est “mauvais”), tu peux **détruire** la master tile.

On doit donc :
- Calculer un **flag ALT-AZ cleanup par groupe** (par tile) en inférant le mount mode depuis les métadonnées déjà présentes dans les entries (ex: `_eqmode_mode`, `MOUNT_MODE`, `EQMODE`, `header["EQMODE"]`).
- Appliquer le cleanup **uniquement** aux groupes ALT_AZ.

### B) Crash mémoire `_cpu_stack_kappa_fallback`
Dans `zemosaic_align_stack.py`, `_cpu_stack_kappa_fallback()` fait :
- `frames_list = [...]`
- `arr = np.stack(frames_list, axis=0)`  => RAM énorme
- `np.nanmedian(arr, axis=0)` => allocations internes supplémentaires

Il faut un fallback CPU “low-mem” :
- qui traite **par bandes de lignes (row-chunks)** (et idéalement compatible RGB),
- sans empiler tout (N,H,W,C) d’un coup,
- tout en conservant le même algorithme (kappa-sigma basé sur median/std) et le même résultat (à tolérance float32).

## Implémentation — étapes détaillées

### Étape 1 — Helper “infer mount mode” (worker)
Dans `zemosaic_worker.py`, ajouter une petite fonction pure, locale au worker (ou module-level privé) :

`_infer_group_eqmode(group: list[dict]) -> str`

Règles :
- Lire en priorité sur la **première entry** (puis fallback sur scan de quelques entries si nécessaire) :
  - `entry.get("MOUNT_MODE")` si str,
  - sinon `entry.get("_eqmode_mode")` si str,
  - sinon `entry.get("EQMODE")` (int/str) : **même mapping que Qt** :
    - 1 => "EQ"
    - 0 => "ALT_AZ"
  - sinon `entry.get("header")` ou `entry.get("header_subset")` et tenter `header.get("EQMODE")`.
- Si impossible : retourner `"UNKNOWN"`.

Important : on **ne doit pas ouvrir** de FITS sur disque ici (pas de lecture fichier). On s’appuie uniquement sur ce qui est déjà en mémoire.

### Étape 2 — Calculer un flag global “contains_altaz”
`WORKER_EQMODE_SUMMARY` existe déjà (cf. log). On a donc `contains_altaz`.

Modifier la logique globale de `altaz_cleanup_effective_flag` :
- Si `contains_altaz == False` : **forcer** `altaz_cleanup_effective_flag = False`.
  - Si l’utilisateur a explicitement demandé le cleanup : log WARN clair du style :
    `ALTaz cleanup requested but no ALT_AZ frames detected -> disabled (EQ-only run).`
- Si `contains_altaz == True` :
  - conserver la logique existante : requested => true, sinon auto-enable => true (ou garder la logique actuelle).

⚠️ But : empêcher *définitivement* un run EQ-only de se faire “nanizer” par un cleanup alt-az.

### Étape 3 — Passer un flag ALT-AZ cleanup **par groupe** en Phase 3
Dans la boucle Phase 3, au moment de `executor_ph3.submit(create_master_tile, ...)`, aujourd’hui on passe `altaz_cleanup_effective_flag`.

Remplacer par :
- `group_eqmode = _infer_group_eqmode(group_info_list)`
- `altaz_cleanup_for_tile = bool(altaz_cleanup_effective_flag) and (group_eqmode == "ALT_AZ")`

Puis passer `altaz_cleanup_for_tile` à `create_master_tile(...)`.

Important :
- Ne pas casser le retry path : quand un sous-groupe est re-soumis (`retry_groups`), il doit garder ses keys (`_eqmode_mode`, etc.) → l’inférence doit marcher pareil.

Ajouter un log DEBUG_DETAIL par tile :
`P3_ALTaz_GATING: tile_id=X eqmode=ALT_AZ global_enabled=True => tile_enabled=True`
et pour EQ :
`P3_ALTaz_GATING: tile_id=Y eqmode=EQ global_enabled=True => tile_enabled=False`

### Étape 4 — Fallback CPU kappa-sigma “low-mem” (align_stack)
Dans `zemosaic_align_stack.py`, remplacer `_cpu_stack_kappa_fallback()` par une implémentation chunkée :

- Conserver le comportement pour petits N/H/W (chemin “fast” = ancien algo) :
  - Estimer bytes du stack complet : `N*H*W*C*4` et comparer à un seuil (ex: 512MB) ou un % RAM (si psutil dispo, optionnel).
- Si au-delà : exécuter un mode `chunk_rows` :
  - allouer `out = np.empty((H,W,C), float32)`
  - pour y0:y1 :
    - construire `arr_chunk = np.stack([f[y0:y1,...] for f in frames_list], axis=0)`
    - calculer `med/std/low/high` sur chunk
    - calculer mask
    - combiner (mean ou weighted mean) **sans créer un arr_clip énorme si possible** :
      - unweighted : in-place set invalid to 0 + sum/count
      - weighted (poids 1D typique) : accumuler `sum += w_i * frame_i` et `den += w_i * mask_i`
    - écrire dans `out[y0:y1,...]`
  - calculer `rejected_pct` global via compteur cumulatif.
  - retourner `(out, rejected_pct)`.

Contraintes :
- dtype final : float32 (comme actuellement).
- Résultat doit matcher l’ancien fallback (sur petits tableaux) à une tolérance float32.
- Ne pas dépendre “fortement” de psutil : si dispo, ok; sinon fallback sur seuil fixe.

### Étape 5 — Micro-tests intégrés (sans framework)
Il n’y a pas de dossier tests dans ce snapshot, donc ajouter des **self-tests optionnels** (sans exécution par défaut) :

- Dans `zemosaic_align_stack.py` :
  - une fonction `_selftest_cpu_kappa_chunk_equivalence()` qui :
    - génère un petit set random (N=8,H=64,W=64,C=3) + quelques NaN,
    - calcule “ancienne version” (garder un helper privé `_cpu_stack_kappa_fallback_fullframe` ou reconstruire localement),
    - calcule “nouvelle version chunk” (forcer chunk_rows petit),
    - compare max abs diff < ~1e-4 (float32).
- Dans `zemosaic_worker.py` :
  - une fonction `_selftest_infer_group_eqmode()` avec 3 cas : EQ, ALT_AZ, UNKNOWN.

Ces fonctions doivent **ne rien impacter** en production et n’être appelées nulle part sauf debug manuel.

## Critères d’acceptation
1) Run **EQ-only** :
   - `contains_altaz=False`
   - ALT-AZ cleanup **désactivé** même si coché (avec WARN clair).
   - Plus de master tiles vides causées par ALT-AZ cleanup.

2) Run **mixte EQ + ALT_AZ** :
   - En Phase 3 :
     - tiles EQ : `tile_enabled=False` (pas de lecropper altaz nanize)
     - tiles ALT_AZ : `tile_enabled=True`
   - Les masters ne sont plus détruites pour les groupes EQ.

3) Plus de crash RAM du type “Unable to allocate XX GiB … (N,H,W,3)” en fallback CPU kappa-sigma :
   - le fallback doit passer sur gros groupes via chunking.

4) SDS et Grid :
   - Aucun changement de code sur `grid_mode.py`
   - SDS pipeline inchangé
   - Pas de modifications de paramètres / options SDS/Grid

## Notes
- Ne pas toucher à `zemosaic_filter_gui_qt.py` : il fait déjà la séparation.
- La correction doit être robuste même si certaines entrées n’ont pas `_eqmode_mode` (fallbacks).
