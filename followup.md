# Validation checklist (manual)

## 1) Vérifier perf log
- [ ] Le log contient "sky_preview_perf: ..." à chaque refresh.
- [ ] Sur gros dataset, on voit clairement collect/build/artists/draw.

## 2) Footprints en LineCollection (option C)
- [ ] Quand footprint_radec est disponible, les footprints sont visibles.
- [ ] Le code n’utilise pas Polygon + add_patch dans une boucle.
- [ ] Une seule LineCollection est ajoutée (ou une par couche si séparation explicite).

## 3) Centres vectorisés
- [ ] Il n’y a plus un scatter par groupe (ou drastiquement moins).
- [ ] Les points restent colorés comme avant (si colorize by group est actif).

## 4) Légende
- [ ] Si groups > LEGEND_MAX, pas de legend (pas de freeze).
- [ ] Le hint UI indique que la legend est masquée.

## 5) Non-régression
- [ ] Petits datasets: rendu OK.
- [ ] Aucun warning Matplotlib nouveau.
- [ ] Pas de crash si footprint_radec absent / incomplet.
