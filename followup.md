
---

# ✅ **followup.md**

```markdown
# Relecture mission + actions demandées

Salut Codex High,

Voici les points à exécuter juste après ton patch :

---

## [x] 1️⃣ Générer un diff clair, propre et minimal  
- Fichiers modifiés listés exactement dans agent.md  
- Aucun changement hors du bloc Reproject non-SDS  
- SDS = zone intouchable (vérifier dans le diff)

---

## [x] 2️⃣ Ajouter logs utiles

Dans la Phase 5 non-SDS, ajouter :

````

[INFO] Tile-weighting enabled — mode=N_FRAMES
[INFO] Weights summary: min=**, max=**, mean=__

```

et côté GPU :

```

[DEBUG] gpu_coadd: tile i uses weight W

```

---

## [x] 3️⃣ Ajouter le header FITS MT_NFRAMES dans Phase 3

Vérifier :

```

MT_NFRAMES = <int>

```

et que cette valeur remonte bien jusque Phase 5.

---

## [x] 4️⃣ CPU = utiliser input_weights  
Confirmer que `input_weights` est bien passé comme une liste d’images constantes.

---

## [x] 5️⃣ GPU = miroir exact  
Confirmer que la pondération respecte la formule :

```

sum_gpu += sampled * weight
weight_gpu += sampled_mask * weight

```

---

## [x] 6️⃣ Ajouter la config + traduction GUI  
Clé config :
```

enable_tile_weighting

```

Clé traduction (FR/EN) :
- “Enable tile weighting”
- “Pondération des tuiles (recommandé)”

---

## [ ] 7️⃣ Tests automatiques à fournir  
Captures ou logs indiquant :

- SDS on → aucune modification du flux  
- Non-SDS + 600/10 frames → pondération correcte  
- GPU/CPU → cohérence > 99.9% des pixels  
- Option OFF → pipeline identique à avant

---

Merci !
