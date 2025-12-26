## followup.md (version amendée)

```md
# followup.md — Résultats attendus + Notes de vérification

## Résumé des changements (à remplir par Codex)

- [x] `zemosaic_filter_gui_qt.py`
  - [x] `_prefetch_eqmode_for_candidates` retourne un dict `eqmode_summary` + stocke `self._last_eqmode_summary`
  - [x] `_prefetch_eqmode_for_candidates` **ne casse pas** les appels existants (signatures compatibles / callers peuvent ignorer le retour)
  - [x] `_compute_auto_groups` ajoute `eqmode_summary` à `result_payload`
  - [x] `overrides()` injecte `eqmode_summary` dans :
    - [x] `global_wcs_meta["eqmode_summary"]`
    - [x] `global_wcs_plan_override["eqmode_summary"]`
  - [x] logs Qt: `FILTER_EQMODE_SUMMARY: ...`

- [x] `zemosaic_worker.py`
  - [x] Lecture `eqmode_summary` (plan_override dict-safe → plan.meta (canonique) → fallback → `{}` si absent)
  - [x] **Plan override dict-safe (anti crash)** :
    - [x] `plan_override_raw = filter_overrides.get("global_wcs_plan_override")` uniquement si `filter_overrides` est un dict
    - [x] `plan_override = plan_override_raw if isinstance(plan_override_raw, dict) else {}` avant tout `.get(...)`
  - [x] **EQMODE meta canonique (anti “lecteur inventé”)** :
    - [x] le worker consomme `global_wcs_plan["meta"]["eqmode_summary"]` comme source **canonique**
    - [x] `global_wcs_plan["eqmode_summary"]` top-level (si écrit) reste **debug / compat**, et n’est **pas relu** ailleurs dans cette mission
  - [x] **Normalisation ultra défensive** des compteurs (amendement):
    - [x] helper `_safe_int()` (gère `None`, int, float, str `"66"`, `"66.0"`, etc. + fallback default)
    - [x] `eq_count = _safe_int(summary.get("eq_count"), 0)`
    - [x] `altaz_count = _safe_int(summary.get("altaz_count"), 0)`
    - [x] `unknown_count = _safe_int(summary.get("unknown_count"), 0)`
    - [x] `total = _safe_int(summary.get("total"), eq_count + altaz_count + unknown_count)`
  - [x] `contains_altaz` calculé (`altaz_count > 0`) + log `WORKER_EQMODE_SUMMARY: ... contains_altaz=...`
  - [x] **Ordre “atomique”** (amendement anti incohérence):
    - [x] calcul `contains_altaz` + `altaz_cleanup_effective_flag` AVANT `lecropper_pipeline_flag`, `pipeline_flags_msg`, `final_quality_pipeline_cfg`, `MT_PIPELINE_FLAGS`
  - [x] `altaz_cleanup_effective_flag` auto activé si `contains_altaz=True` (niveau **run entier**, pas par groupe)
  - [x] pipeline lecropper/quality cfg utilisent **effective_flag**
  - [x] **Alpha = masque uniquement** (amendement anti zones noires):
    - [x] aucune multiplication/altération de `mosaic_data`/RGB “science” par l’alpha
  - [x] logs alpha sanity (`ALPHA_STATS: ...`) au niveau master tiles / pipeline finale
  - [x] Phase 6 preview PNG : si alpha existe → **RGBA**, log `PHASE6_PREVIEW_ALPHA: ...`
  - [x] (si fait) fallback PIL RGBA si OpenCV absent, sinon WARN clair

- [x] Hotfix intertile (Windows/hybrid)
  - [x] Calibration intertile bridée à un seul worker sur Windows/hybride pour éviter les crashes natifs

## Extraits de logs attendus (exemples)

### Cas EQ only
```text
FILTER_EQMODE_SUMMARY: eq=20 altaz=0 unknown=0 total=20
WORKER_EQMODE_SUMMARY: eq=20 altaz=0 unknown=0 total=20 contains_altaz=False
MT_PIPELINE_FLAGS: ... altaz_cleanup_enabled=False ...
PHASE6_PREVIEW_ALPHA: saved_rgba=True (backend=cv2)
```

### Cas ALT/AZ présent (ou dataset mixte)

```text
FILTER_EQMODE_SUMMARY: eq=12 altaz=66 unknown=0 total=78
WORKER_EQMODE_SUMMARY: eq=12 altaz=66 unknown=0 total=78 contains_altaz=True
AUTO_ALTaz_cleanup: enabled (contains_altaz=True, config_requested=False)
ALPHA_STATS: nonzero_frac=0.42 min=0.00 max=1.00 shape=...
PHASE6_PREVIEW_ALPHA: saved_rgba=True (backend=cv2)
```

> Remarque : pour un dataset **mixte** (EQ+ALT/AZ), `contains_altaz=True` et `altaz_cleanup_effective_flag=True` pour **tout le run**.
> Aucune logique plus fine (par tile/cluster) ne doit être introduite dans cette mission.

## Vérification visuelle (anti “trous noirs”)

* Ouvrir la preview PNG dans un viewer qui respecte l’alpha :
  * Les zones masquées (alpha=0) doivent être **transparentes**, pas “noires pleines”.
* Aucun fond ajouté artificiellement.
* Les zones avec coverage réel doivent garder leur contenu (pas de noircissement via alpha).

## Non-régression

* [ ] Aucun changement du pipeline EQ-only (mêmes outputs, mêmes chemins principaux)
* [ ] Pas de refactor large (diff ciblé aux sections décrites)
* [ ] Pas de modification Tk / lecropper UI
* [ ] Aucun changement dans la logique borrowing / clustering / GPU safety

## Notes / Risques

* Si un environnement n’a pas OpenCV:
  * [x] le fallback conserve l’alpha (PIL RGBA) ou log WARN clair si impossible.
* Si `eqmode_summary` absent (anciens projets / autres workflows que Qt Filter):
  * [x] comportement identique à avant (pas d’auto activation, altaz_cleanup uniquement si demandé dans la config).
* Dataset mixte EQ+ALT/AZ:
  * [x] `contains_altaz=True`
  * [x] `altaz_cleanup_effective_flag=True` appliqué à l’ensemble du run (pas d’heuristique par groupe)
```

---
