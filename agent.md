
### 🎯 Mission

1. **Corriger la géométrie des tiles dans Grid Mode**

   * assurer que *toutes* les reprojections utilisent **exactement la même WCS** et **exactement le même canevas (shape_out)**
   * supprimer les décalages accumulés actuellement visibles dans la mosaïque

2. **Activer et appliquer la normalisation photométrique inter-tile**

   * compute_tile_photometric_scaling + apply_tile_photometric_scaling
   * utiliser un masque de recouvrement coverage/WCS
   * appliquer *avant* la reprojection

3. **Réintroduire correctement l’égalisation RGB par tuile**

4. **Garantir un pipeline cohérent, sans fallback silencieux**

---

# 1 — Correctifs PROBLÉMATIQUES obligatoires

(⚠️ À implémenter absolument)

## 1.1 — Verrouiller la WCS globale ET la taille du canevas

Dans `grid_mode.py`, après :

```python
global_wcs, shape_hw = find_optimal_celestial_wcs(...)
```

➡️ Codex doit **imposer ce shape** à *toutes* les reprojections :

```python
array, footprint = reproject_interp(
    tile_data,
    tile_wcs,
    global_wcs,
    shape_out=shape_hw,
    return_footprint=True
)
```

⚠️ Sans ce `shape_out`, chaque tuile obtient un canevas différent → **décalages + mosaïque en escalier**.

## 1.2 — Propager shape_hw partout

Dans le plan global utilisé par le worker :

```python
plan["width"] = shape_hw[1]
plan["height"] = shape_hw[0]
```

Et **jamais** remplacer ces valeurs plus bas dans le pipeline.

---

# 2 — Normalisation photométrique inter-tile

(Mêmes instructions que ta version précédente mais **avec requirement strict d’application AVANT reproject**)

## 2.1 Avant reproject pour CHAQUE tuile :

```
→ stack tile
→ equalize_rgb_medians_inplace (si RGB)
→ compute_tile_photometric_scaling (masque basé coverage/WCS)
→ apply_tile_photometric_scaling
→ reproject_interp(..., shape_out=shape_hw)
```

---

# 3 — Correction de crop / CRPIX / bounding box

Lors du crop automatique de la mosaïque globale :

```
CRPIX1 -= x0
CRPIX2 -= y0
NAXIS1 = width
NAXIS2 = height
```

➡️ Codex doit **déplacer ce correctif AVANT** toute validation/finalisation du plan dans worker.
➡️ Sinon : double crop → **tuile décalée**, exactement ce que tu observes.

---

# 4 — Égalisation RGB par tuile

Identique à ta version précédente, mais ajouté explicitement dans l’ordre d’exécution.

---

# 5 — Perf minimal

(identique à ton fichier, rien à modifier)

---

# 6 — Critères d’acceptation (ajout)

### Le Grid Mode est validé quand :

* les tiles n'ont **plus aucun décalage** géométrique
* plus de damier
* plus de bandes photométriques
* plus de fallback vers le flux classique
* la coverage globale correspond **exactement** aux tiles

