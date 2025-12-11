## 🧾 agent.md

### Contexte

Projet : **ZeMosaic / ZeSeestarStacker**
Objectif de cette mission :

1. [x] **Mission 1 – zemosaic_filter.log**

   * Le fichier `zemosaic_filter.log` grossit indéfiniment.
   * On veut **le supprimer au lancement** du filtre (GUI Qt) pour repartir d’un log propre à chaque ouverture.

2. [x] **Mission 2 – Dominante verte dans le flux classique**

   * Les **master tiles** du flux classique sont correctement équilibrées en couleurs (poststack_equalize_rgb OK).
   * La **mosaïque finale classique** présente encore une **dominante verte**, qui apparaît après la Phase 5 (reprojection / coadd / renorm).
   * On veut ajouter une **étape d’égalisation RGB globale sur la mosaïque finale** (en utilisant la même logique que `poststack_equalize_rgb`), **sans toucher au flux Grid mode**.

> ⚠️ **Crucial : ne pas modifier le flux Grid mode.**
>
> * Ne pas éditer `grid_mode.py`.
> * Ne pas changer les chemins d’exécution spécifiques Grid dans `zemosaic_stack_core.py` ou `zemosaic_worker.py`.
> * Les changements de Mission 2 doivent s’appliquer **uniquement au flux mosaïque classique**, pas au script `grid_mode.py`.

---

### Fichiers concernés

* `zemosaic_filter_gui_qt.py`  ✅ (Mission 1)
* `zemosaic_worker.py`         ✅ (Mission 2, ajout de l’étape d’equalize sur la mosaïque finale)
* `zemosaic_align_stack.py`    🔍 (Mission 2 : réutilisation de `_poststack_rgb_equalization` / `equalize_rgb_medians_inplace`, **sans** changer leur comportement)

**À ne pas modifier :**

* `grid_mode.py`
* Tout autre fichier lié uniquement au flux Grid (sauf import passif déjà existant).

---

### Bug constaté / correctif à appliquer en priorité

* `zemosaic_worker.log` contient `"[RGB-EQ] Unexpected error during final mosaic RGB equalization: name 'zconfig' is not defined"` → l’égalisation RGB finale n’est pas exécutée et la dominante verte persiste.
* Source : `_run_shared_phase45_phase5_pipeline(...)` utilise `zconfig` alors que ce nom n’existe pas dans son scope.
* Correctif attendu :
  * Ajouter `zconfig` (kw-only, optionnel) dans la signature de `_run_shared_phase45_phase5_pipeline(...)` et passer la véritable instance depuis `run_hierarchical_mosaic(...)` (flux classique) ainsi que depuis le chemin SDS qui appelle ce helper.
  * Utiliser ce `zconfig` local (fallback `SimpleNamespace()` si besoin) pour l’appel à `_apply_final_mosaic_rgb_equalization(...)` et pour les `setattr(..., "parallel_plan_phase5", ...)` déjà présents.
* Validation : plus aucun warning `name 'zconfig' is not defined` et présence de `[RGB-EQ] final mosaic: ...` dans `zemosaic_worker_cl.log`.

---

## Mission 1 – Réinitialiser `zemosaic_filter.log` au lancement

### But

Au lancement du filtre via l’interface Qt, **supprimer le fichier `zemosaic_filter.log` s’il existe**, avant que le logger ne commence à écrire dedans, afin d’éviter qu’il ne grossisse indéfiniment.

### Implémentation attendue

1. Dans `zemosaic_filter_gui_qt.py` (c’est le point de référence principal pour cette mission) :

   * Le module importe déjà `Path` depuis `pathlib`.
   * Ajouter une fonction utilitaire **tout en haut du fichier, après les imports**, par exemple :

   ```python
   from pathlib import Path
   # ... autres imports déjà présents ...

   def _reset_filter_log() -> None:
       """
       Supprime le log zemosaic_filter.log au lancement de l'outil,
       pour éviter qu'il ne grossisse indéfiniment.
       """
       try:
           # Même dossier que le script ; adapter si le log est ailleurs
           log_path = Path(__file__).with_name("zemosaic_filter.log")
           if log_path.exists():
               log_path.unlink()
       except Exception:
           # On ne bloque jamais le démarrage pour un problème de log
           pass

   # Appelé au chargement du module
   _reset_filter_log()
   ```

2. Contraintes :

   * **Ne pas modifier la configuration logging existante** : on ne touche pas aux handlers, formatters, etc.
   * On se contente de **supprimer le fichier** avant que les handlers ne l’ouvrent.
   * Le code doit être **robuste** :

     * En cas d’exception (droits, verrouillage, etc.), on ignore l’erreur et on laisse le programme continuer.
   * Ne pas introduire de dépendance circulaire.
   * Ne pas dupliquer cette logique dans 15 endroits : un seul helper `_reset_filter_log()` suffit.

3. Optionnel mais autorisé :

   * Si, dans le code, le vrai “main” du filtre est dans `zemosaic_filter_gui.py`, le même helper peut être placé là **à la place** de `zemosaic_filter_gui_qt.py`, mais il doit être **appelé une seule fois au démarrage**.
   * Dans tous les cas, documenter clairement dans un commentaire où et pourquoi on réinitialise le log.

---

## Mission 2 – Equalize RGB sur la mosaïque finale (flux classique uniquement)

### But

* Les **master tiles** sont déjà équilibrées par `_poststack_rgb_equalization` (via `equalize_rgb_medians_inplace`).
* Après la Phase 5 (reprojection / coadd / renormalisation inter-tuiles / two-pass coverage), la mosaïque finale du **flux classique** présente une pente verte.
* On veut ajouter une **étape d’égalisation RGB globale sur la mosaïque finale**, juste avant l’écriture des fichiers (FITS/PNG/TIFF), avec logs propres, en **réutilisant la même logique que `_poststack_rgb_equalization`**.

> ❗ Important :
>
> * Cette étape doit dépendre du **même flag de config** que pour les master tiles (`poststack_equalize_rgb`).
> * Elle ne doit **pas modifier le comportement du script `grid_mode.py`**.

### Points d’ancrage dans le code

* `zemosaic_align_stack.py`

  * Contient déjà :

    * `equalize_rgb_medians_inplace(img: np.ndarray)`
    * `_poststack_rgb_equalization(stacked, zconfig, stack_metadata=None)`
      → c’est cette logique qu’on veut **réutiliser** pour la mosaïque finale.

* `zemosaic_worker.py`

  * Contient les fonctions de Phase 5 :

    * `assemble_final_mosaic_incremental(...)`
    * `assemble_final_mosaic_reproject_coadd(...)`
    * `_apply_phase5_post_stack_pipeline(...)`
    * `_apply_final_mosaic_quality_pipeline(...)`
    * `_auto_crop_global_mosaic_if_requested(...)`
    * `run_hierarchical_mosaic(...)` (orchestration principale du flux classique + SDS)
  * C’est dans ce fichier qu’on doit **brancher l’égalisation RGB finale**.

### Stratégie d’implémentation

#### 2.1. Importer proprement `_poststack_rgb_equalization`

En haut de `zemosaic_worker.py`, avec les autres imports conditionnels :

```python
try:
    from zemosaic_align_stack import _poststack_rgb_equalization
except Exception:  # pragma: no cover - fallback si import cassé
    _poststack_rgb_equalization = None
```

> Ne pas changer `_poststack_rgb_equalization` lui-même, ni `equalize_rgb_medians_inplace`.

#### 2.2. Nouveau helper : égalisation RGB sur mosaïque finale

Toujours dans `zemosaic_worker.py`, ajouter un helper interne, par exemple juste avant `_apply_phase5_post_stack_pipeline` ou dans la même zone :

```python
def _apply_final_mosaic_rgb_equalization(
    final_mosaic_data: np.ndarray | None,
    zconfig: Any,
    logger: logging.Logger | None = None,
) -> tuple[np.ndarray | None, dict]:
    """
    Applique la même logique que `_poststack_rgb_equalization` sur la mosaïque finale.

    - Respecte le flag de config `poststack_equalize_rgb`.
    - Ne fait rien si la fonction d'origine est indisponible ou si l'image n'est pas RGB.
    - Retourne (final_mosaic_data éventuellement modifiée, info_dict).
    """
    info: dict = {
        "enabled": False,
        "applied": False,
        "gain_r": 1.0,
        "gain_g": 1.0,
        "gain_b": 1.0,
        "target_median": float("nan"),
    }

    if final_mosaic_data is None or _poststack_rgb_equalization is None:
        return final_mosaic_data, info

    # On réutilise exactement la même fonction que pour les master tiles
    metadata: dict = {}
    try:
        info = _poststack_rgb_equalization(final_mosaic_data, zconfig=zconfig, stack_metadata=metadata)
    except Exception as exc:  # robustesse : ne jamais casser la Phase 5
        if logger is not None:
            logger.warning("[RGB-EQ] Final mosaic RGB equalization failed: %s", exc)
        return final_mosaic_data, info

    if logger is not None and info.get("applied"):
        logger.info(
            "[RGB-EQ] final mosaic: applied=True, gains=(%.6f, %.6f, %.6f), target_median=%.2f",
            info.get("gain_r", 1.0),
            info.get("gain_g", 1.0),
            info.get("gain_b", 1.0),
            info.get("target_median", float("nan")),
        )

    return final_mosaic_data, info
```

Contraintes :

* Le helper doit être **no-op** si :

  * `final_mosaic_data` est `None`,
  * `_poststack_rgb_equalization` est indisponible,
  * ou si `poststack_equalize_rgb` est désactivé (la fonction d’origine gère déjà ce cas).
* Ne pas lever d’exception vers l’appelant en cas d’erreur (log + retour no-op).

#### 2.3. Appeler le helper uniquement pour le flux classique

Dans `run_hierarchical_mosaic(...)`, après que :

* La Phase 5 a produit `final_mosaic_data_HWC`, `final_mosaic_coverage_HW`, `final_alpha_map`,
* Les post-traitements communs type `_apply_phase5_post_stack_pipeline(...)` sont passés,
* **Mais avant** :

  * `_finalize_sds_global_mosaic` (pour SDS) ou toute écriture disque.

Ajouter un appel au helper **uniquement pour la mosaïque finale du flux classique**.

Idée de câblage (pseudo-code, à adapter au code réel) :

```python
# Après les appels à assemble_final_mosaic_* et à _apply_phase5_post_stack_pipeline
# et avant la finalisation / écriture des fichiers.

# On s'assure qu'on n'est pas dans une branche SDS/grid spécifique
if final_mosaic_data_HWC is not None and not sds_mode_phase5:
    try:
        final_mosaic_data_HWC, final_rgb_info = _apply_final_mosaic_rgb_equalization(
            final_mosaic_data_HWC,
            zconfig=zconfig,
            logger=logger,
        )
        # Optionnel : exposer les infos dans les callbacks ou la télémétrie
        # (pas obligatoire, mais possible)
    except Exception as exc:
        logger.warning(
            "[RGB-EQ] Unexpected error during final mosaic RGB equalization: %s",
            exc,
        )
```

Points importants :

* **Conditionner** l’appel sur `not sds_mode_phase5` (ou flag équivalent dans le code courant) pour cibler le **flux mosaïque classique**.
* **Ne pas appeler ce helper dans le script `grid_mode.py`**.
* Ne pas modifier la signature publique des fonctions déjà appelées par `grid_mode.py`.

  * Si une signature doit évoluer, vérifier que les appels Grid n’en dépendent pas.

#### 2.4. Logging

* Le helper logge déjà une ligne du type :

  ```text
  [RGB-EQ] final mosaic: applied=True, gains=(..., ..., ...), target_median=...
  ```

* Ne pas multiplier les logs localisés via `pcb(...)` pour cette étape : un log direct sur `logger` est suffisant.

* Vérifier que le logger utilisé est bien `logger = logging.getLogger("ZeMosaicWorker")` ou un de ses children.

---

### Contraintes générales

* **Ne pas modifier `grid_mode.py`.**
* Ne pas changer le comportement de `poststack_equalize_rgb` sur les master tiles.
* Ne pas toucher aux signatures publiques utilisées par d’autres modules, sauf si absolument nécessaire, et dans ce cas :

  * Mettre des valeurs par défaut compatibles pour ne rien casser.
* Toute nouvelle logique doit être **robuste aux erreurs** :

  * Try/except préventifs.
  * Pas d’exception non gérée qui ferait tomber tout le run.

---

## Tests attendus

Après implémentation :

1. **Mission 1 – zemosaic_filter.log**

   * Lancer `zemosaic_filter_gui_qt.py`.
   * Vérifier que :

     * Si `zemosaic_filter.log` existait, il a été **supprimé puis recréé**.
     * En relançant plusieurs fois, la taille du log repart bien de zéro à chaque démarrage.

2. **Mission 2 – mosaïque finale classique**

   * Utiliser un dataset de test classique (non Grid).
   * Activer `poststack_equalize_rgb=True` dans la config.
   * Lancer un run complet :

     * Vérifier dans `zemosaic_worker_cl.log` :

       * présence d’une ligne `[RGB-EQ] final mosaic: applied=True, gains=(...)`.
     * Comparer la mosaïque finale :

       * La **dominante verte doit être fortement réduite voire disparue**.
   * Vérifier que :

     * Les **master tiles** ont toujours l’air correctes.
     * Le flux Grid mode (script `grid_mode.py`) fonctionne exactement comme avant (double-check au moins un dataset Grid).

---

