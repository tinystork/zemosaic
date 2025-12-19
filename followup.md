# Validation checklist (manual)

## 1) Vérifier perf log
- [x] Le log contient "sky_preview_perf: ..." à chaque refresh.
- [x] Sur gros dataset, on voit clairement collect/build/artists/draw.

## 2) Footprints en LineCollection (option C)
- [x] Quand footprint_radec est disponible, les footprints sont visibles.
- [x] Le code n’utilise pas Polygon + add_patch dans une boucle.
- [x] Une seule LineCollection est ajoutée (ou une par couche si séparation explicite).

## 3) Centres vectorisés
- [x] Il n’y a plus un scatter par groupe (ou drastiquement moins).
- [x] Les points restent colorés comme avant (si colorize by group est actif).

## 4) Légende
- [x] Si groups > LEGEND_MAX, pas de legend (pas de freeze).
- [x] Le hint UI indique que la legend est masquée.

## 5) Non-régression
- [x] Petits datasets: rendu OK.
- [x] Aucun warning Matplotlib nouveau.
- [x] Pas de crash si footprint_radec absent / incomplet.
