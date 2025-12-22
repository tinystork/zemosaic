## Checklist d’implémentation (Codex)

### 1) Preview noir : CPU
- [ ] Dans `zemosaic_utils.py / stretch_auto_asifits_like()`, remplacer `np.percentile` par une logique NaN-aware (nanpercentile + garde-fous).
- [ ] Vérifier qu’aucun gros temporaire float64 n’est introduit (rester en float32).
- [ ] Vérifier le comportement si le canal est 100% NaN → sortie canal = 0 (pas d’exception).

### 2) Preview noir : GPU
- [ ] Dans `zemosaic_utils.py / stretch_auto_asifits_like_gpu()`, remplacer `cp.percentile` par `cp.nanpercentile` (si dispo) ou fallback sur valeurs finies.
- [ ] Ajouter garde-fous (canal vide / vmin-vmax trop faible) → sortie canal = 0.
- [ ] Ne pas modifier la logique de fallback CPU existante.

### 3) Pondération perdue : keywords
- [ ] Dans `zemosaic_worker.py`, dans `assemble_final_mosaic_reproject_coadd()` → `_extract_tile_weight()` :
  - [ ] ajouter `NRAWPROC`, `NRAWINIT` à la liste des keywords.
- [ ] Ne PAS modifier la construction de `input_weights_list` (éviter double pondération).
- [ ] Ne PAS toucher au mode “I'm using master tiles (skip clustering_master tile creation)”.

---

## Procédure de validation manuelle (sans changer d’autres fichiers)

### A) Valider preview
1) Lancer un run qui produit des master tiles / mosaïque avec alpha/NaN masking.
2) Ouvrir le `*_preview.png` :
   - Vérifier qu’il n’est plus noir (RGB non nuls) et que la transparence est conservée.

### B) Valider pondération
1) Prendre un FITS d’entrée (existing master tile) contenant `NRAWPROC` / `NRAWINIT`.
2) Lancer l’assemblage final en mode “existing master tiles”.
3) Vérifier dans les logs :
   - ligne “Tile-weighting enabled — mode=N_FRAMES”
   - summary min/max/mean ≠ 1.0 si les headers ont des valeurs > 1
4) Vérifier que le rendu final n’est plus “écrasé” par des tuiles moins profondes (qualitativement, le signal doit mieux tenir).

---

## Anti-régressions (à surveiller)
- Preview : ne pas changer l’alpha, seulement rendre le stretch robuste aux NaN.
- Poids : ne pas doubler les poids (ne pas multiplier `input_weights` *et* passer `tile_weights`).
- Aucun impact sur le clustering, la création de master tiles, ni le pipeline non-existing-master-tiles.
