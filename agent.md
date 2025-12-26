# agent.md — Mission: Remonter eqmode_summary au Worker + Alpha→Transparence (ALT/AZ)

## Objectif
1) **Faire remonter `eqmode_summary`** (résumé EQ vs ALT_AZ vs UNKNOWN) depuis le **Filter GUI Qt** vers le **worker**.
2) **Auto-activer `altaz_cleanup`** (lecropper altZ) lorsque le dataset contient de l’ALT/AZ (détecté via `eqmode_summary`).
3) **Garantir que l’alpha mask est propagé** jusqu’aux sorties (au minimum la preview PNG) afin d’avoir de la **transparence** et **pas de “gros trous noirs”**.
4) **Ne pas inventer de fond** : pas de remplissage “sky”, pas de reconstruction artificielle — on préfère alpha=0 / NaN.

> **Important (datasets mixtes EQ+ALT/AZ)**  
> Si `altaz_count > 0`, alors `altaz_cleanup` est **effectif pour tout le run** (toute la mosaïque).  
> Codex ne doit **PAS** essayer d’introduire des heuristiques plus “malines” (par groupe, par master tile, etc.).  
> Toute granularité plus fine (par cluster/tuile) sera l’objet d’une mission future, pas de celle-ci.

## Contraintes (anti-régression)
- **Ne pas modifier** la logique scientifique de coadd/reproject/weighting hors activation ALT/AZ (pas de refactor massif).
- **Ne pas changer** le comportement EQ-only : si pas d’ALT/AZ → rien ne s’active automatiquement, résultats inchangés.
- **Ne pas toucher** au worker “borrowing”, à la logique de clustering existante, ni aux mécanismes de safe-mode/mémoire.
- **Ne pas toucher** à la GUI Tk (lecropper UI) : **Qt only** + Worker.
- Pas de “fill background” (fond inventé). La sortie doit refléter la couverture réelle.

## Suivi d’avancement
- [x] Étape A — Qt Filter: produire un `eqmode_summary` structuré
- [x] Étape B — Qt Filter: injecter `eqmode_summary` dans les overrides envoyés au worker
- [x] Étape C — Worker: lire `eqmode_summary`, normaliser et stocker la version canonique
- [x] Étape D — Worker: auto-activer `altaz_cleanup` (ordre atomique, flag effectif au niveau run)
- [x] Étape E — Alpha propagation: transparence jusqu’aux previews (pas de noircissement du RGB science)
- [x] Hotfix — Intertile: forcer 1 worker sur Windows/hybride (crash natif)

### Amendement critique (anti “zones noires”)
- **Interdit** d’appliquer l’alpha au RGB “scientifique” (pas de `rgb *= alpha`, pas de noircissement des pixels couverts).
- L’alpha sert **uniquement** à la **transparence/masquage** (preview/export) et/ou à des couches séparées (coverage/weight),
  mais ne doit pas altérer `mosaic_data` / `final_mosaic` (science).

## Fichiers autorisés à modifier (scope strict)
- `zemosaic_filter_gui_qt.py`
- `zemosaic_worker.py`

(Optionnel uniquement si nécessaire pour sauvegarde RGBA sans OpenCV)
- `zemosaic_utils.py` (uniquement si on ajoute une fonction utilitaire de save PNG RGBA fallback)

## Plan d’implémentation

### Étape A — Qt Filter: produire un `eqmode_summary` structuré
Dans `zemosaic_filter_gui_qt.py` :

1) **Modifier** `_prefetch_eqmode_for_candidates(...)` :
   - Aujourd’hui : calcule eq/altaz/unknown et log une string.
   - À faire :
     - Construire un dict `eqmode_summary` **stable** (pas de grosses datas) :

       ```python
       eqmode_summary = {
         "eq_count": int(eq_count),
         "altaz_count": int(altaz_count),
         "unknown_count": int(unknown_count),
         "total": int(total_count),
         "cache_hits": int(cache_hits),
         "cache_miss": int(cache_miss),
         "reads_header": int(reads_header),
         "source": "qt_prefetch_eqmode",
       }
       ```

     - Stocker aussi sur l’instance : `self._last_eqmode_summary = eqmode_summary`
     - Retourner `eqmode_summary` (return) **sans casser les appels existants**.
       - Si tu veux rester ultra-safe : garder le même comportement côté callers (ils peuvent ignorer le retour).

2) Dans `_compute_auto_groups(...)` :
   - Capturer le retour :

     ```python
     eqmode_summary = self._prefetch_eqmode_for_candidates(candidate_infos, messages)
     ```

   - Stocker dans `result_payload` (le dict renvoyé) :
     - `result_payload["eqmode_summary"] = eqmode_summary` si dict.

> BUT: On veut que `overrides()` puisse récupérer `eqmode_summary` sans parsing de logs.

### Étape B — Qt Filter: injecter `eqmode_summary` dans les overrides envoyés au worker
Toujours dans `zemosaic_filter_gui_qt.py`, dans `overrides()` (là où `global_wcs_meta` est rempli après `_ensure_global_wcs_for_selection`) :

1) Récupérer le résumé :
   - Priorité : `result_payload.get("eqmode_summary")` si dispo.
   - Sinon fallback : `getattr(self, "_last_eqmode_summary", None)`.

2) **Injecter dans `global_wcs_meta`** (meilleur chemin car le worker met ça dans `global_wcs_plan["meta"]`) :

   ```python
   if eqmode_summary:
       meta_payload["eqmode_summary"] = eqmode_summary
   ```

3) **Injecter aussi dans `global_wcs_plan_override`** (facultatif mais robuste) :

   ```python
   if eqmode_summary:
       plan_override = metadata_update.get("global_wcs_plan_override")
       if not isinstance(plan_override, dict):
           plan_override = {}
       plan_override["eqmode_summary"] = eqmode_summary
       metadata_update["global_wcs_plan_override"] = plan_override
   ```

4) Log Qt clair (INFO) :

   * `FILTER_EQMODE_SUMMARY: eq=.. altaz=.. unknown=.. total=..`

### Étape C — Worker: lire `eqmode_summary` et déterminer `contains_altaz` (avec parsing ultra défensif)

Dans `zemosaic_worker.py`, dans `run_hierarchical_mosaic_classic_legacy(...)` juste après :

```python
global_wcs_plan = _prepare_global_wcs_plan(...)
plan_override_raw = filter_overrides.get("global_wcs_plan_override") if isinstance(filter_overrides, dict) else None
plan_override = plan_override_raw if isinstance(plan_override_raw, dict) else {}
```

1. Résoudre `eqmode_summary` (ordre de priorité) :

   * `plan_override.get("eqmode_summary")` (plan_override dict-safe, sinon `{}`)
   * `global_wcs_plan.get("meta", {}).get("eqmode_summary")` si dict
   * `filter_overrides.get("eqmode_summary")` si dict (fallback)
   * sinon `None` / `{}`

2. **Normalisation défensive** (amendement):

   * Ne jamais supposer que c’est un int (peut être str "66", "66.0", float, etc.)

   * Ajouter un helper local minimal (dans la fonction, ou en haut de module si déjà une convention) :

     ```python
     def _safe_int(v, default=0):
         try:
             if v is None:
                 return default
             if isinstance(v, bool):
                 return int(v)
             if isinstance(v, int):
                 return v
             if isinstance(v, float):
                 return int(v)
             if isinstance(v, str):
                 s = v.strip()
                 if not s:
                     return default
                 # autorise "66.0"
                 return int(float(s))
             return int(v)
         except Exception:
             return default
     ```

   * Puis normaliser :

     ```python
     summary = eqmode_summary or {}
     eq_count = _safe_int(summary.get("eq_count"), 0)
     altaz_count = _safe_int(summary.get("altaz_count"), 0)
     unknown_count = _safe_int(summary.get("unknown_count"), 0)
     total = _safe_int(summary.get("total"), eq_count + altaz_count + unknown_count)

     contains_altaz = altaz_count > 0
     ```

   * Log INFO unique :

     * `WORKER_EQMODE_SUMMARY: eq=. altaz=. unknown=. total=. contains_altaz=.`

3. Attacher à `global_wcs_plan` pour usage plus tard :

> **Note (EQMODE top-level vs meta)**
> Pour cette mission, la version **canonique** est celle stockée dans `global_wcs_plan["meta"]["eqmode_summary"]`.
> Le champ top-level `global_wcs_plan["eqmode_summary"]` (si présent) est **redondant / debug** :
> le worker ne doit **pas** le relire plus tard, et Codex ne doit pas introduire de lecteur ailleurs dans le pipeline.

```python
# Canonical location for this mission:
meta = global_wcs_plan.get("meta")
if not isinstance(meta, dict):
    meta = {}
    global_wcs_plan["meta"] = meta
meta["eqmode_summary"] = {
  "eq_count": eq_count,
  "altaz_count": altaz_count,
  "unknown_count": unknown_count,
  "total": total,
}

# Optional (debug / backward compatibility):
global_wcs_plan["eqmode_summary"] = dict(meta["eqmode_summary"])
```

### Étape D — Worker: auto-activer `altaz_cleanup` (sans casser EQ-only) + ordre d’exécution “atomique”

Toujours dans `run_hierarchical_mosaic_classic_legacy(...)` :

#### D0 — Amendement critique (anti états incohérents)

* Calculer `contains_altaz` + `altaz_cleanup_effective_flag` **AVANT** :

  * `lecropper_pipeline_flag`
  * `pipeline_flags_msg`
  * `final_quality_pipeline_cfg`
  * et tout log “MT_PIPELINE_FLAGS” qui reflète les flags.

#### D1 — Définir requested vs effective

1. Partir du flag de config :

```python
altaz_cleanup_requested_flag = bool(altaz_cleanup_enabled_config)
altaz_cleanup_effective_flag = altaz_cleanup_requested_flag
```

2. Si `contains_altaz` est True → forcer ON pour **tout le run** :

```python
if contains_altaz:
    if not altaz_cleanup_effective_flag:
        logger.info("AUTO_ALTaz_cleanup: enabled (contains_altaz=True, config_requested=False)")
    altaz_cleanup_effective_flag = True
```

* **Important** : comportement **global au run**.
* Interdiction d’introduire une logique par groupe/tile/cluster basée sur `eqmode_summary`.

#### D2 — Consommer *uniquement* effective_flag

* `lecropper_pipeline_flag` dépend de `altaz_cleanup_effective_flag`
* `pipeline_flags_msg` affiche **l’état effectif**
* `final_quality_pipeline_cfg["altaz_cleanup_enabled"] = altaz_cleanup_effective_flag`

#### D3 — Invariants

* Si `contains_altaz=False` → `altaz_cleanup_effective_flag == requested_flag` (comportement identique à avant)
* Dataset EQ-only : pas de changement.

### Étape E — Alpha propagation: garantir transparence et éviter “trous noirs”

Objectif: si alpha existe, **preview PNG = RGBA** (alpha=0 → transparent), pas RGB noir.

1. Ajouter des logs “sanity” low-cost au moment où `pipeline_alpha_mask` (ou équivalent) est disponible :

   * Calculer au moins :

     ```python
     alpha_nonzero_frac = float(np.count_nonzero(alpha) / alpha.size)
     alpha_min = float(alpha.min())
     alpha_max = float(alpha.max())
     ```

   * Log INFO :

     * `ALPHA_STATS: nonzero_frac=.. min=.. max=.. shape=H×W`

   * WARN si :

     * `altaz_cleanup_effective_flag=True` **et** `alpha_nonzero_frac < 0.01` → suspect (alpha quasi vide)

2. Côté Phase 6 / preview :

   * Si alpha dispo :

     * construire `alpha_preview` (downscale cohérent avec preview RGB)
     * construire RGBA (ou BGRA pour cv2) :

       ```python
       rgba_view = np.dstack([rgb_view[..., 0], rgb_view[..., 1], rgb_view[..., 2], alpha_preview])
       ```

   * Log :

     * `PHASE6_PREVIEW_ALPHA: saved_rgba=True (backend=cv2)`

   * Sinon :

     * `PHASE6_PREVIEW_ALPHA: skipped (no_alpha_map)`

3. Fallback si OpenCV indisponible (anti “trou noir”) :

   * Essayer PIL en RGBA si dispo.
   * Sinon WARN clair :

     * `PHASE6_PREVIEW_ALPHA: saved_rgba=False (no_opencv_no_pil_alpha_lost)`
   * **Interdit** de “remplir le fond” :

     * pas de fill des pixels alpha=0 (on garde NaN/0 et alpha=0).

## Critères d’acceptation

* Dataset **EQ only** :

  * logs montrent `contains_altaz=False`, `altaz_cleanup_effective_flag=False` si config off.
  * résultat scientifique inchangé (mêmes outputs principaux).

* Dataset **ALT/AZ** ou **mixte** :

  * logs montrent `altaz_count > 0`, `contains_altaz=True`.
  * `AUTO_ALTaz_cleanup` loggué si la config ne le demandait pas explicitement.
  * `altaz_cleanup_effective_flag=True` pour tout le run (pas de logique par groupe).

* Quand un alpha map existe :

  * preview PNG écrite en **RGBA** (ou BGRA côté cv2) :

    * zones alpha=0 transparentes dans un viewer compatible,
    * pas de grandes zones “noir plein” inventées.

* Pas de régression :

  * aucun changement de format/nommage du FITS principal,
  * aucun impact sur borrowing / clustering / GPU safety.

## Tests manuels (checklist)

1. Run court EQ-only (10–20 frames) :

   * vérifier logs `FILTER_EQMODE_SUMMARY` + `WORKER_EQMODE_SUMMARY` (`contains_altaz=False`)
   * vérifier `MT_PIPELINE_FLAGS` (altaz_cleanup off si non demandé)
   * vérifier preview (doit être comme avant)

2. Run ALT/AZ-only ou mixte (Seestar) :

   * vérifier logs :

     * `FILTER_EQMODE_SUMMARY: eq=.. altaz>0 ...`
     * `WORKER_EQMODE_SUMMARY: ... contains_altaz=True`
     * `AUTO_ALTaz_cleanup: enabled ...` si config off
   * vérifier preview RGBA (viewer qui affiche l’alpha : PS/GIMP/Affinity, etc.) :

     * zones sans coverage → transparentes, pas noir

3. Vérifier logs alpha :

   * `ALPHA_STATS: ...`
   * `PHASE6_PREVIEW_ALPHA: saved_rgba=True (...)` ou WARN explicite si fallback

4. Vérifier qu’aucun “fill sky” n’a été ajouté (inspection visuelle + aucun code de fill dans le diff)

## Livraison attendue

* PR/commit avec modifications **minimales** aux fichiers listés.
* Logs d’exemple (3–6 lignes clés) collés dans `followup.md` :

  * un run EQ-only,
  * un run ALT/AZ ou mixte avec ALT/AZ présent.
