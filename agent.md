# Mission (surgical / no refactor)
Accélérer le Sky preview dans zemosaic_filter_gui_qt.py en remplaçant le rendu "patch par patch / scatter par groupe"
par du rendu batché (option C) via LineCollection + scatter vectorisé.

Contexte actuel (à vérifier dans le fichier):
- Le sky preview est rendu dans _update_preview_plot().
- Il fait actuellement un loop sur grouped_points et appelle axes.scatter(...) une fois par groupe (potentiellement 100+).
- Les bounding-box de groupes utilisent déjà LineCollection (bien).
- Le rectangle de sélection est un unique Rectangle patch (OK, garder tel quel).

Objectifs
1) Option C pour footprints:
   - Si des entries possèdent footprint_radec (liste de coins RA/DEC), dessiner ces footprints en une seule fois:
     - construire une liste "segments" (chaque footprint = liste de (ra,dec) fermée),
     - utiliser matplotlib.collections.LineCollection,
     - ajouter 1 seule collection via axes.add_collection().
   - Ne PAS créer de Polygon ni ax.add_patch en boucle.
   - Appliquer couleur par groupe si disponible (sinon couleur défaut).

2) Scatter des centres: vectoriser
   - Au lieu de scatter par groupe, faire un seul axes.scatter avec:
     - x array (RA), y array (DEC),
     - un array de couleurs (par point) basé sur group_idx,
     - alpha, size identiques au rendu actuel.
   - (Conserver la possibilité de désactiver le "colorize by group" si l’option existe déjà, sinon garder comportement actuel.)

3) Légende: garde-fou
   - Ne pas afficher la légende si num_groups > LEGEND_MAX (ex 30).
   - Si > LEGEND_MAX, afficher un hint dans _preview_hint_label: "Legend hidden (too many groups)".

4) Instrumentation perf
   - Ajouter un timer monotonic au début/fin de _update_preview_plot() et aux étapes:
     - collect_points_dt (appel _collect_preview_points),
     - build_arrays_dt (construction arrays/segments),
     - add_artists_dt (add_collection/scatter),
     - draw_dt (canvas draw),
     - redraw_count (si tu peux compter les appels _safe_draw_preview_canvas).
   - Logguer en logger.info et dans activity log:
     "sky_preview_perf: N=%d groups=%d collect=%.3fs build=%.3fs artists=%.3fs draw=%.3fs"
   - Ne pas spammer: un log par refresh.

Contraintes
- Modifier uniquement zemosaic_filter_gui_qt.py
- Ne pas changer la logique de clustering, seulement l’affichage.
- Ne pas introduire de dépendances externes.
- Compat Windows/Qt: ne pas bloquer l’UI plus que nécessaire.

Détails techniques (rendu LineCollection footprints)
- Importer LineCollection si pas déjà importé.
- Pour chaque entry/point, si entry.footprint_radec existe:
  - récupérer la séquence de (ra, dec),
  - fermer le contour (répéter le premier point à la fin si nécessaire),
  - ajouter à segments.
- Couleurs:
  - si group_idx est int: couleur via _resolve_group_color(group_idx),
  - sinon couleur par défaut.
- Créer:
  coll = LineCollection(segments, colors=colors, linewidths=1.0, alpha=0.8, zorder=3)
  axes.add_collection(coll)

Scatter vectorisé
- Construire ra_list, dec_list, color_list (par point).
- Faire un seul axes.scatter(ra_list, dec_list, c=color_list, s=24, alpha=0.85, edgecolors="none")

Livrable
- Diff git propre.
- Le rendu doit rester correct (au minimum équivalent visuellement) et beaucoup plus rapide sur 5000 entrées.
