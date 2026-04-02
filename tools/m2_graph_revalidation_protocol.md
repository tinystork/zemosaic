# M2 + revalidation graphe/pruning — protocole opérationnel

Objectif: obtenir un **step change mesurable** sur les seams (pas un micro-retune invisible).

## 1) Matrice minimale de runs (ordre recommandé)

Sur **le même dataset**, mêmes entrées/master tiles, mêmes options non concernées.

- **Run R0 (référence)**
  - config actuelle stable (celle qui sert de baseline visuelle)
- **Run R1 (M2 ON)**
  - `intertile_gain_offset_v2=true`
  - `intertile_offset_only_v1=false`
  - autres paramètres inchangés vs R0
- **Run R2 (M2 ON + graphe/pruning renforcé)**
  - même que R1
  - + variation contrôlée de pruning/scoring (ex: `intertile_prune_k`, `intertile_prune_weight_mode`)

Optionnel:
- **Run R3 (M2 ON + weighting OFF)** pour isoler l'impact pur du solve photométrique.

---

## 2) Garde-fous de comparabilité

Ne pas changer entre runs:
- dataset / input_dir
- mode d’assemblage final
- options preview/DBE (sauf campagne dédiée)
- machine si possible (sinon noter clairement la machine)

Toujours conserver:
- `run_config_snapshot.json`
- `zemosaic_worker.log`
- `tile_weights_final.csv`
- `zemosaic_MT*_R0.fits`
- `zemosaic_MT*_R0_preview.png`

---

## 3) Critères GO / NO-GO (M2)

GO si, sur R1/R2 vs R0:
1. baisse visible des seams (inspection humaine)
2. baisse sur indicateurs log (ex: `TwoPassWorst abs_delta_med`)
3. pas de nouvelle dérive couleur/halo/banding
4. pas de régression de coverage/geometry

NO-GO si:
- aucune amélioration visible + métriques stables
- ou amélioration locale contrebalancée par artefacts nouveaux

---

## 4) Outils de comparaison

Utiliser:
- `tools/m2_seam_report.py` pour comparer 2 runs

Exemple:

```bash
python3 tools/m2_seam_report.py \
  "/media/tristan/X10 Pro/mosaic/test/test run J" \
  "/media/tristan/X10 Pro/mosaic/test/test run K"
```

---

## 5) Interprétation rapide

- Si diff de mosaïque quasi nulle + `TwoPassWorst` inchangé:
  - le levier testé n’a pas produit de step change.
- Si `TwoPassWorst` baisse mais seams visuelles persistent:
  - compléter avec piste résiduelle low-frequency (finitions visuelles optionnelles).
- Si `TwoPassWorst` baisse + visuel meilleur:
  - consolider paramètres et valider non-régression multi-mode.
