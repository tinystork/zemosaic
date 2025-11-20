# Mission — Synchroniser le mode SDS entre la GUI principale Qt et la GUI de filtrage Qt

## Contexte

Dans ZeMosaic, il existe aujourd’hui deux cases à cocher liées au mode SDS (ZeSupaDupStack) :

1. **Dans la GUI principale Qt** (`zemosaic_gui_qt.py`, onglet *Main*) :
   - Case : **"Enable SDS mode by default"**
   - Cette case est liée à la clé de config `sds_mode_default` (booléen).

2. **Dans la GUI de filtrage Qt** (`zemosaic_filter_gui_qt.py`) :
   - Case : **"Enable ZeSupaDupStack (SDS)"**
   - Cette case est liée à `sds_mode` (overrides / metadata) et à la config `sds_mode_default` pour certains chemins.

Actuellement :
- La case du *filter* est **activée par défaut** côté `DEFAULT_FILTER_CONFIG` (`sds_mode_default=True`).
- La case de la GUI principale est **désactivée par défaut** (`sds_mode_default=False`).
- La synchronisation n’est que partielle : l’état n’est pas systématiquement partagé entre les deux interfaces.

**Objectif ergonomique** :

> Avoir une **préférence SDS unique** (`sds_mode_default`) :
> - La case *Main* « Enable SDS mode by default » reflète / contrôle `sds_mode_default`.
> - La case *Filter* « Enable ZeSupaDupStack (SDS) » :
>   - s’initialise selon `sds_mode_default` (ou un override explicite `sds_mode`) ;
>   - renvoie son état vers la GUI principale, qui met à jour `sds_mode_default` + sa propre case.

En résumé, **les deux cases doivent se suivre** :
- Cocher / décocher dans le *Main* → le *Filter* ouvre dans le même état.
- Cocher / décocher dans le *Filter* → le *Main* met à jour sa case et la config.

---

## Fichiers à modifier

Les fichiers sont disponibles dans le repo, et une copie de travail existe aussi dans l’environnement suivant (à titre de référence) :

- `/mnt/data/zemosaic_gui_qt.py`
- `/mnt/data/zemosaic_filter_gui_qt.py`

**Ne modifier que ces deux fichiers** pour cette mission.

---

## Détail des modifications à effectuer

### 1. zemosaic_filter_gui_qt.py

#### 1.1. Aligner le défaut `sds_mode_default` sur le main (False)

Repérer le bloc de définition de `DEFAULT_FILTER_CONFIG` :

```python
DEFAULT_FILTER_CONFIG: dict[str, Any] = dict(_DEFAULT_GUI_CONFIG_MAP)
DEFAULT_FILTER_CONFIG.setdefault("auto_detect_seestar", True)
DEFAULT_FILTER_CONFIG.setdefault("force_seestar_mode", False)
DEFAULT_FILTER_CONFIG.setdefault("sds_mode_default", True)
DEFAULT_FILTER_CONFIG.setdefault("sds_min_batch_size", 5)
DEFAULT_FILTER_CONFIG.setdefault("sds_target_batch_size", 10)
DEFAULT_FILTER_CONFIG.setdefault("global_coadd_method", "kappa_sigma")
DEFAULT_FILTER_CONFIG.setdefault("global_coadd_k", 2.0)
````

**Modification demandée :**

* Passer le défaut de `True` à `False` pour être cohérent avec le `fallback_defaults` du main :

DEFAULT_FILTER_CONFIG.setdefault("sds_mode_default", False)
```

> Ne pas toucher aux autres lignes (min / target batch size, coadd, etc.).

---

#### 1.2. Initialiser `_sds_mode_initial` à partir de `sds_mode` OU `sds_mode_default`

Repérer dans `FilterQtDialog.__init__` le calcul actuel :

```python
self._sds_mode_initial = self._coerce_bool(
    (initial_overrides or {}).get("sds_mode")
    if isinstance(initial_overrides, dict)
    else None,
    self._coerce_bool(
        (config_overrides or {}).get("sds_mode")
        if isinstance(config_overrides, dict)
        else None,
        True,
    ),
)
```

**Objectif :**

1. Priorité à `initial_overrides["sds_mode"]` (si présent).
2. Sinon, utiliser `config_overrides["sds_mode_default"]` si disponible.
3. Sinon, fallback sur `DEFAULT_FILTER_CONFIG["sds_mode_default"]` (désormais False).

**Remplacer le bloc par :**

```python
self._sds_mode_initial = self._coerce_bool(
    (initial_overrides or {}).get("sds_mode")
    if isinstance(initial_overrides, dict)
    else None,
    self._coerce_bool(
        (config_overrides or {}).get("sds_mode_default")
        if isinstance(config_overrides, dict)
        else None,
        bool(DEFAULT_FILTER_CONFIG.get("sds_mode_default", False)),
    ),
)
```

Ne pas modifier la suite, qui associe cette valeur à la case :

```python
self._sds_checkbox = QCheckBox(
    self._localizer.get("filter_chk_sds_mode", "Enable ZeSupaDupStack (SDS)"),
    box,
)
self._sds_checkbox.setChecked(bool(self._sds_mode_initial))
...
```

---

### 2. zemosaic_gui_qt.py

Nous avons deux modifications à faire :

1. **Passer l’état SDS du main vers le filter.**
2. **Récupérer l’état SDS du filter vers le main.**

#### 2.1. Propager `sds_mode_default` → `initial_overrides["sds_mode"]`

Dans `_launch_filter_dialog`, repérer l’endroit où `initial_overrides` est construit :

```python
initial_overrides: Dict[str, Any] | None = None
try:
    initial_overrides = {
        "cluster_panel_threshold": float(self.config.get("cluster_panel_threshold", 0.05)),
        "cluster_target_groups": int(self.config.get("cluster_target_groups", 0)),
        "cluster_orientation_split_deg": float(self.config.get("cluster_orientation_split_deg", 0.0)),
    }
except Exception:
    initial_overrides = None
```

**Modification demandée :**

* Ajouter une entrée `sds_mode` qui reflète l’état courant de `sds_mode_default` dans la config principale.

Par exemple :

```python
initial_overrides = {
    "cluster_panel_threshold": float(self.config.get("cluster_panel_threshold", 0.05)),
    "cluster_target_groups": int(self.config.get("cluster_target_groups", 0)),
    "cluster_orientation_split_deg": float(self.config.get("cluster_orientation_split_deg", 0.0)),
    # Synchroniser l’état de la case "Enable SDS mode by default" vers le Filter Qt
    "sds_mode": bool(self.config.get("sds_mode_default", False)),
}
```

> L’idée est que, lorsque l’utilisateur ouvre le filter, la case « Enable ZeSupaDupStack (SDS) » s’aligne sur ce que l’utilisateur a choisi dans le *Main*.

---

#### 2.2. Appliquer `overrides["sds_mode"]` au `sds_mode_default` du main

Repérer la méthode :

```python
def _apply_filter_overrides_to_config(self, overrides: Dict[str, Any] | None) -> None:
    if not overrides:
        return
    for key in (
        "cluster_panel_threshold",
        "cluster_target_groups",
        "cluster_orientation_split_deg",
    ):
        if key in overrides:
            self._update_widget_from_config(key, overrides[key])
    if "astap_max_instances" in overrides:
        self._update_widget_from_config("astap_max_instances", overrides["astap_max_instances"])
        self._apply_astap_concurrency_setting()
```

**Modification demandée :**

* Entre la boucle `for key in (...)` et le bloc `if "astap_max_instances" ...`, ajouter un bloc pour synchroniser la préférence SDS :

```python
def _apply_filter_overrides_to_config(self, overrides: Dict[str, Any] | None) -> None:
    if not overrides:
        return
    for key in (
        "cluster_panel_threshold",
        "cluster_target_groups",
        "cluster_orientation_split_deg",
    ):
        if key in overrides:
            self._update_widget_from_config(key, overrides[key])

    # Synchroniser le choix SDS du Filter Qt avec la case "Enable SDS mode by default"
    if "sds_mode" in overrides:
        self._update_widget_from_config("sds_mode_default", overrides["sds_mode"])

    if "astap_max_instances" in overrides:
        self._update_widget_from_config("astap_max_instances", overrides["astap_max_instances"])
        self._apply_astap_concurrency_setting()
```

`_update_widget_from_config("sds_mode_default", ...)` doit déjà :

* mettre à jour `self.config["sds_mode_default"]`, et
* mettre à jour l’état de la case « Enable SDS mode by default » dans la GUI principale (via la logique existante déjà utilisée pour les autres champs).

---

## Contraintes

* Ne PAS modifier la logique des autres options (coverage, overcap, coadd, etc.).
* Ne PAS refactoriser le reste du code : la mission est purement cosmétique / synchronisation d’état.
* Laisser intacte toute la logique SDS existante (plan SDS, `global_wcs_plan_override`, etc.), sauf les points explicitement indiqués ci-dessus.
* Conserver les noms de fonctions et clés (`sds_mode_default`, `sds_mode`) tels quels.

---

## Résultat attendu (comportement utilisateur)

1. **Au démarrage, si aucune config existante** :

   * `sds_mode_default` = False par défaut (Main + Filter).
   * La case « Enable SDS mode by default » est décochée.
   * La case « Enable ZeSupaDupStack (SDS) » est décochée à l’ouverture du filter.

2. **Si l’utilisateur coche la case SDS dans le Main puis ouvre le Filter** :

   * `sds_mode_default` passe à True.
   * À l’ouverture du Filter, la case « Enable ZeSupaDupStack (SDS) » est **cochée** automatiquement.

3. **Si l’utilisateur change la case SDS dans le Filter et valide** :

   * Les `overrides` contiennent `sds_mode`.
   * `_apply_filter_overrides_to_config` met à jour `sds_mode_default` avec cette valeur.
   * La case « Enable SDS mode by default » dans le Main se met à jour automatiquement.
   * Le JSON de config persistant (via le mécanisme existant) garde cette préférence pour la prochaine session.

Si ces conditions sont remplies, la mission est considérée comme réussie.

````
