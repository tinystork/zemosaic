---

# ✅ **agent.md**

```markdown
# 🟦 Mission Codex High — Pondération par profondeur pour Reproject (mode non-SDS)

> 🎯 Objectif : ajouter une pondération physique des master tiles (**N_raw_frames**) dans la Phase 5 (Reproject & Coadd) **UNIQUEMENT** pour le mode non-SDS afin d’empêcher les tuiles peu profondes (ex : 10 brutes) de dégrader les zones bien couvertes (ex : 600 brutes).

> ⚠ Zone protégée : **Tout le code SDS est sanctuarisé.**  
> NE PAS modifier :
> - assemble_global_mosaic_sds
> - assemble_global_mosaic_first
> - toute la chaîne SDS (méga-tiles, super-stack, coverage_sds)
> - aucun fichier/branche/baseline lié au SDS.

---

# 🔵 Contrainte majeure

Le pipeline existant doit rester *strictement inchangé* en dehors du bloc de reproject/coadd non-SDS.

- Phase 1–2 : inchangé  
- Phase 3 (master tiles) : inchangé (sauf ajout header `N_FRAMES`)  
- Phase 4 : inchangé  
- Phase 5 SDS : inchangé  
- Phase 5 non-SDS : **modifié pour intégrer les poids**  
- Phase 6 : inchangé

---

# 🧩 Fichiers concernés

Tu dois modifier **uniquement** les fichiers suivants :

- `zemosaic_worker.py`
- `zemosaic_utils.py`
- `zemosaic_align_stack.py` (si nécessaire pour accéder à N_raw)
- Les headers FITS des master tiles (ajout d’un champ `MT_NFRAMES`)

Ne pas toucher aux fichiers SDS (alignement, astrometry, mega tiles).

---

# 📐 Principe à implémenter

## 1. Récupérer le nombre de brutes par master tile
Chaque master tile possède déjà une structure `tile_info`.  
Tu dois y ajouter lors de la Phase 3 :

```

header["MT_NFRAMES"] = <nombre de brutes ayant servi à créer cette tuile>

````

Si le nombre de brutes n'est pas disponible directement, utiliser :
- la longueur de la liste des frames qui ont servi à créer la tile.

---

## 2. Préparer un vecteur `tile_weights[]` pour la Phase 5

Dans `zemosaic_worker.py`, juste avant l’appel à `reproject_and_coadd_wrapper`, construire :

```python
tile_weights = [ header["MT_NFRAMES"] for each master tile ]
````

Avec fallback :

```python
if missing: tile_weights[i] = 1
```

---

## 3. Injection dans la voie CPU

Dans `zemosaic_utils.reproject_and_coadd_wrapper`, lorsque la voie CPU Astropy est utilisée :

* Fournir `input_weights` comme une liste **d’images 2D constantes** :

Pour chaque tuile i :

```python
weight_map = np.full_like(tile_data[i], tile_weights[i], dtype=np.float32)
```

Puis :

```python
result = reproject_and_coadd(
    input_data,
    wcs_output,
    input_weights=weight_maps,
    combine="mean",
    ...
)
```

Le comportement attendu :

[
I(p) = \frac{\sum_i I_i(p) \cdot w_i}{\sum_i w_i}
]

---

## 4. Injection dans la voie GPU (implémentation interne)

Dans `gpu_reproject_and_coadd_impl()`, remplacer :

```python
sum_gpu += sampled
weight_gpu += sampled_mask
```

par **la version pondérée** :

```python
sum_gpu += sampled * weight_i
weight_gpu += sampled_mask * weight_i
```

avec :

```python
weight_i = tile_weights[i]
```

Cela doit **imiter exactement la logique Astropy** :

* `sampled` est l’image reprojetée
* `sampled_mask` vaut 0/1
* on multiplie par le poids de la tuile
* le résultat final est `sum_gpu / weight_gpu`

---

## 5. API / Config / GUI

Ajouter dans `zemosaic_config.py` :

```python
"enable_tile_weighting": true,
"tile_weight_mode": "n_frames"    # réservé à l'avenir
```

GUI (Qt/Tk) :

* une case “Tile weighting (recommended)” cochée par défaut
* pas d’impact sur SDS (désactive option si SDS activé)

## Traductions à ajouter en EN/FR.

## 6. Tests obligatoires

Codex doit valider :

1. Mode non-SDS (`enable_tile_weighting=true`)

   * deux tuiles 600/10 → la 10 contribue ~1,6 % en overlap
   * pas de régression dans forme/couverture/dimensions

2. Mode non-SDS (`enable_tile_weighting=false`)
   → comportement identique à avant (flat weighting)

3. Mode SDS → aucun changement, aucune régression

4. GPU vs CPU → même résultat (à tolérance float près)

---

# 🟩 Succès =

* La Shark Nebula n’est plus détruite par les tuiles faibles
* Le bruit ne “flood” plus les zones profondes
* Le pipeline reste 100 % rétrocompatible
* SDS intact
* GPU/CPU cohérents
* Aucun impact sur les autres phases
* Performance inchangée

````

