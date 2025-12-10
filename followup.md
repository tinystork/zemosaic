
# 📄 **followup.md — Version corrigée et complétée**


*(Mais **mis à jour ici** — copie complète ci-dessous)*

---

## followup.md — Validation Grid Mode après correctifs WCS + Photométrie

### 1 — Vérification géométrique (NOUVEAU BLOC CRITIQUE)

1. Dans les logs, vérifier que *chaque* reproject appelle :

```
shape_out = shape_hw
```

2. Vérifier que **shape_hw reste identique** du début à la fin du run.

3. Vérifier que le plan final n'est **plus recadré deux fois** :

* un seul shift CRPIX
* un seul crop

4. Charger la mosaïque finale → **SUPERPOSER les footprints WCS** dans DS9 :

* toutes les tuiles doivent se chevaucher
* aucun décalage de 1 à 10 pixels comme avant

---

### 2 — Vérification photométrique

* afficher pour chaque tile :

  * median avant scaling
  * median après scaling
  * gain/offset
* vérifier que :

  * gain ≈ 1 ± 0.2
  * offset raisonnable
  * pas de NaN

---

### 3 — Vérification fallback (doit être désactivé)

Dans le log de worker :
**AUCUNE occurrence de :**

```
[GRID] fallback to classic
```

Si un fallback apparaît → la géométrie n’est toujours pas correcte.

---

### 4 — Checklist finale

* [x] shape_hw transmis à toutes les reprojections
* [x] scaling appliqué avant reprojection
* [x] equalize_rgb_medians_inplace appliqué avant scaling
* [ ] CRPIX mis à jour une seule fois
* [ ] plus de damier
* [ ] plus de bandes verticales
* [ ] coverage correcte
* [ ] aucun fallback

