# Mission (surgical / no refactor)
Accélérer drastiquement le Sky preview dans zemosaic_filter_gui_qt.py pour gros datasets (5000+ FITS).

Symptôme:
- Clustering OK, mais l’UI semble “bloquée” pendant/juste après l’auto-organisation.
- Le Sky preview Matplotlib est soupçonné (trop d’objets, trop de redraws, légende lourde).

Objectifs:
1) Instrumenter précisément les timings Sky preview:
   - build_geometry_dt (construction segments/rects)
   - create_artists_dt (création objets Matplotlib)
   - draw_dt (canvas draw/draw_idle)
   - redraw_count
   Logguer un résumé: "sky_preview_perf: N=... build=... artists=... draw=... redraws=..."

2) Réduire le coût Matplotlib:
   - Remplacer l’approche "un patch par footprint" par une approche "Collection":
     - utiliser matplotlib.collections.LineCollection (préféré) ou PatchCollection
     - ajouter 1 collection à l’axes (au lieu de milliers d’add_patch)
   - Éviter les redraws répétés:
     - interdire la reconstruction complète de figure/axes pendant la phase d’auto-group
     - faire 1 seul draw_idle() à la fin (ou au max 1 par X secondes via throttle)

3) Gestion gros N:
   - Si N_footprints > PREVIEW_HARD_LIMIT (ex 1500 ou param existant):
     - mode "centroid only" (scatter) OU "thin lines only" (LineCollection)
   - Désactiver la légende si num_groups > LEGEND_MAX (ex 30),
     ou remplacer par "Top N + others".

Contraintes:
- 1 seul fichier: zemosaic_filter_gui_qt.py
- Pas de changement dans le clustering lui-même, uniquement l’affichage Sky preview.
- Ne pas toucher au workflow utilisateur (mêmes boutons/onglets).
- Pas de dépendance externe nouvelle.

Détails d’implémentation attendus:
- Introduire une petite fonction utilitaire interne:
  - _render_sky_preview_fast(axes, footprints, colors_by_group, mode, ...)
  - qui construit une LineCollection en une passe.
- Ajouter un throttle simple pour draw_idle:
  - self._last_preview_draw_ts, et ne draw que si > 0.3s (par ex) pendant opérations longues,
    puis un draw final forcé en fin de traitement.

Livrables:
- Diff git propre
- Log "sky_preview_perf" visible dans zemosaic_filter.log pendant auto-group.
