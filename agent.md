✅ Mission

Corriger les bandes de jointures photométriques réapparues dans les mosaïques ZeMosaic, en Phase 5 / Reproject & Coadd, sans toucher :

au routage CPU/GPU,

au pipeline SDS vs non-SDS,

ni à la correction de dominante verte / égalisation RGB.

Les flux CPU et GPU doivent produire exactement la même image (à bruit numérique près), sans bandes au niveau des bords de tuiles.

🧩 Symptôme

En sortie de Phase 5, la mosaïque finale présente des rectangles visibles qui suivent les contours des master tiles (voir capture fournie par l’utilisateur).

Le problème est identique en mode CPU et GPU, donc la cause est dans la logique commune à Phase 5, pas dans les kernels GPU eux-mêmes.

Avant certaines modifications récentes (RGB equalization / Phase 5 refactor), ces bandes n’étaient pas visibles.

🎯 Objectifs détaillés

Merci de :

 Identifier la cause des bandes de jointure : perte de pondération / masques, ordre de pipeline, renorm désactivée, etc.

 Garantir que chaque tuile passe à nouveau correctement par :

le pipeline lecropper (quality crop + Alt-Az cleanup quand activé),

la map de poids radiale (feathering en bord de champ),

la normalisation photométrique/coverage attendue en Phase 5.

 Corriger la photometric blending de Phase 5 pour supprimer les bandes, tout en conservant :

 le comportement actuel CPU/GPU (structure du code, toggles, parallel plan),

 la correction de dominante verte (RGB median equalization),

 la compatibilité SDS (super-tiles / mega-tiles) existante.

 Ajouter des tests unitaires / d’intégration ciblant les jointures, pour éviter toute régression future.

📂 Fichiers & Fonctions concernées

Cœur Phase 5 & pipeline qualité (OK à modifier, mais de façon minimale et localisée) :

zemosaic_worker.py

_apply_lecropper_pipeline(...)

_apply_final_mosaic_quality_pipeline(...)

_apply_master_tile_crop_mask_to_mosaic(...)

_apply_phase5_post_stack_pipeline(...)

_apply_two_pass_coverage_renorm_if_requested(...)

_apply_final_mosaic_quality_pipeline(...)

_mask_sds_low_coverage_pixels(...)

_sanitize_sds_megatile_payload(...)

_auto_crop_global_mosaic_if_requested(...)

_emit_coverage_summary_log(...)

Toute fonction strictement nécessaire pour la photometric blending et la propagation des masques/coverage.

Radial weights & égalisation RGB (ne modifier que si absolument nécessaire, et sans casser la correction de dominante verte) :

zemosaic_align_stack.py

make_radial_weight_map (import via zemosaic_utils)

equalize_rgb_medians_inplace(...)

_poststack_rgb_equalization(...)

Utilitaires (à toucher seulement si vraiment indispensable) :

zemosaic_utils.py

Fonctions liées à reproject_coadd, aux coverage maps et à la gestion WCS.

🚧 Zones à NE PAS modifier

Merci de ne pas toucher aux éléments suivants (sauf micro-bug évident et documenté) :

 Routage CPU/GPU, détection GPU, ParallelPlan, auto_tune_parallel_plan, detect_parallel_capabilities (parallel_utils.py, zemosaic_worker.py).

 Logique d’orchestration des phases (1 à 5) et SDS vs non-SDS (mécanique globale).

 Toggles GUI / propagation des flags GPU (Tk/Qt) :

zemosaic_gui.py

zemosaic_gui_qt.py

 Clustering / grouping / autosplit (Phase 2 / 3), hard-merge, etc.

 Correction de dominante verte / poststack_equalize_rgb : ne pas revenir en arrière.

🧠 Hypothèses techniques (pour t’orienter)

Tu peux les vérifier / infirmer, mais ne pas les appliquer aveuglément :

Radial weight map non appliquée / écrasée

make_radial_weight_map peut ne plus être appelé ou utilisé correctement.

Si la map radiale n’est pas appliquée (ou est revenue à un comportement neutre), les tuiles gardent un bord “dur” → bandes visibles.

Coverage map perdue ou uniformisée

Dans _sanitize_sds_megatile_payload, _mask_sds_low_coverage_pixels, _apply_final_mosaic_quality_pipeline, certains fallbacks mettent coverage = ones(...).

Si un chemin de code tombe trop souvent sur ce fallback, la coaddition devient quasi uniforme → contours de tuiles visibles.

Pipeline lecropper non (ou plus) appliqué au global

_apply_lecropper_pipeline & _apply_final_mosaic_quality_pipeline doivent être appelés après la première coadd, avant toute renorm ou cropping final.

Si lecropper ne masque plus les bords/artefacts, les transitions deviennent visibles.

Two-pass renorm désactivée / cassée silencieusement

_apply_two_pass_coverage_renorm_if_requested peut retourner None en cas de problème et laisser le résultat brut.

Si la seconde passe échoue systématiquement (par exemple à cause d’un param GPU/CPU ou parallel_plan), la mosaïque garde un gradient de jointure.

🔬 Plan de travail suggéré
1. Analyse / diff

 Identifier le dernier commit où les bandes étaient absentes (commit 38c876a ).

 Comparer Phase 5 (fonctions listées plus haut) entre ce commit et HEAD pour isoler :

 changements sur coverage / weights,

 modifications sur lecropper / radial weights,

 ajustements sur RGB equalization.

Documenter dans un commentaire de MR ou dans followup.md :

quelles fonctions ont changé,

quelle partie du pipeline est susceptible d’avoir réintroduit les bandes.

2. Reconstruction du pipeline attendu (concept)

L’objectif est :

Coadd global → première passe (reproject + somme pondérée par coverage/weights).

Pipeline qualité :

 application éventuelle de lecropper sur la mosaïque globale,

 application éventuelle du crop % master tile (quand activé),

 masques/alpha associés correctement injectés dans coverage.

Two-pass renorm (si activée) :

 utiliser la coverage map et/ou les tiles sources pour lisser les variations de fond,

 respecter les limites de gain DEFAULT_INTERTILE_GAIN_LIMITS et offset.

Autocrop global (si activé), en respectant les offsets dans le WCS global.

Merci de :

 Vérifier que cet ordre est bien respecté,

 Corriger les cas où coverage / alpha / masques ne sont pas propagés d’une étape à l’autre.

3. Implémentation

 Corriger les fonctions identifiées pour :

 s’assurer que chaque tile/mosaïque passe par le pipeline lecropper quand activé dans la config,

 garantir que la coverage map :

 est nettoyée des NaN,

 correspond bien à la géométrie de l’image,

 est utilisée comme poids dans la coadd (et non remplacée par ones(...) sauf cas exceptionnel très clairement loggé).

 s’assurer que les masques/bords cropés → coverage = 0, image = NaN (ou 0), de façon cohérente sur CPU et GPU.

 Si nécessaire, renforcer ou réintroduire le radial feathering sur les tiles :

attention : ne pas casser le comportement antérieur en cas de désactivation du feathering dans la config.

4. Tests automatiques

Merci d’ajouter des tests ciblés (dans tests/ — tu peux créer un nouveau fichier si besoin, ex. test_phase5_blending.py) :

 Test de jointure simple CPU

Créer deux “tiles” 2D (ou 3D RGB) de taille identique, avec une zone d’overlap :

tile A = niveau 1.0, tile B = niveau 2.0.

Simuler une coadd + pipeline Phase 5 (en appelant les fonctions internes de façon contrôlée).

Vérifier que dans la zone d’overlap, la transition est lisse (pas de step net).

 Test de jointure avec coverage

Couvrir une zone où seule la tile B est présente → vérifier absence de “bande” en bordure.

 Test lecropper activé

Forcer l’application du pipeline lecropper sur une mosaïque avec un bord clairement marqué et vérifier que ces bords sont masqués (coverage à 0, image à NaN/0).

 Optionnel : test GPU

Si possible, simuler un plan avec use_gpu=True mais en gardant le même code Python, pour vérifier que CPU/GPU retournent des résultats identiques (mêmes masques/coverage).

Les tests ne doivent pas dépendre d’Astropy/Photutils s’il n’y en a pas besoin : privilégier du NumPy pur.

5. Logging / observabilité

 Ajouter des logs INFO_DETAIL / DEBUG centrés sur la coverage/weights en Phase 5 :

fraction de pixels couverts,

bbox coverage,

nombre de pixels masqués par lecropper,

application ou non de la seconde passe.

Veiller à ne pas inonder le log par défaut (niveau INFO). Utiliser de préférence :

lvl="INFO_DETAIL" ou lvl="DEBUG" pour les infos verbeuses.

✅ Critères d’acceptation

La mission est réussie si :

 Les mosaïques CPU et GPU issues du même dataset ne présentent plus de bandes visibles aux jointures de tiles (inspection visuelle + éventuellement différence numérique).

 En désactivant lecropper / radial weights dans la config, le comportement reste cohérent (pas de crash, bandes potentiellement visibles, mais contrôlé).

 La correction de dominante verte est toujours effective (aucun retour à une image verdie).

 Les tests ajoutés passent localement (pytest) et sont stables.

 Aucun changement n’a été apporté au routage CPU/GPU, au clustering ou aux GUI au-delà de ce qui est explicitement demandé.