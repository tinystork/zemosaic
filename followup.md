# Validation checklist (manual)

## Perf instrumentation
- [x] Le log contient "sky_preview_perf: ..." avec N, build_dt, artists_dt, draw_dt, redraws.
- [ ] Sur dataset 5000 images, on observe clairement où le temps part.

## Optimisation Matplotlib
- [x] Le preview est rendu via LineCollection/PatchCollection (pas 5000 add_patch).
- [x] redraws << avant (idéalement 1 draw final, ou quelques draws throttlés).
- [ ] L’UI ne “freeze” plus pendant l’auto-organisation.

## Gros datasets
- [x] Si N très grand, le preview passe en mode simplifié (centroids/lines) automatiquement.
- [x] La légende est désactivée ou limitée (Top N + others) pour num_groups élevé.

## Non-régression
- [ ] Petits datasets: rendu identique ou visuellement acceptable.
- [ ] Aucune exception Matplotlib.
