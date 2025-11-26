Salut Codex,

Merci pour ton aide sur la suppression des bandes de jointure en Phase 5.
Voici ce dont j’ai besoin en retour de ta part après implémentation :

1️⃣ Résumé de ta solution

Merci de m’expliquer, en quelques paragraphes :

 Quelle était la cause exacte de la réapparition des bandes (perte de coverage, lecropper bypassé, radial weights, two-pass, etc.).

 Quelles fonctions et fichiers tu as modifiés (liste précise).

 Comment le pipeline Phase 5 s’enchaîne désormais :

première coadd,

lecropper,

crop master tile % (si activé),

two-pass renorm (si activée),

autocrop global (si activé).

2️⃣ Diffs et contraintes respectées

 Confirme que tu n’as pas modifié :

le routage CPU/GPU global,

ParallelPlan / auto_tune_parallel_plan,

les GUI Tk/Qt (hors éventuelle mise à jour de labels, si vraiment nécessaire),

la correction de dominante verte (egalisation RGB médiane),

la logique SDS vs non-SDS (super-tiles / mega-tiles).

 Indique les principaux diffs pertinents (extraits de code ou explications) pour chaque fonction modifiée.

3️⃣ Tests ajoutés / mis à jour

Merci de détailler :

 Les nouveaux tests (nom des fonctions / fichiers) que tu as ajoutés pour valider :

la disparition des bandes de jointure,

la bonne propagation des coverage/masques,

la robustesse du pipeline avec/ sans lecropper / two-pass.

 Comment exécuter ces tests (pytest ...) et ce qu’ils vérifient exactement.

4️⃣ Validation visuelle & recommandations

Après tes changements, je vais :

Relancer un dataset de test en mode CPU et GPU.

Vérifier visuellement l’absence de bandes aux jointures.

Merci donc de :

 Me dire quels flags de config tu recommandes pour valider le plus facilement ton travail (ex. activer/désactiver lecropper, two-pass, radial weights).

 Me préciser dans les logs les messages clefs à chercher pour confirmer que :

lecropper a bien été appliqué sur la mosaïque finale,

le two-pass renorm a bien tourné (ou a été désactivé proprement).

Si quelque chose t’oblige à toucher à une zone “hors scope” (CPU/GPU, SDS, clustering, GUI…), merci de l’indiquer clairement dans ta réponse, avec la justification et l’impact.

Merci 🙏

---

État d’avancement (itération en cours) :

- [x] 1️⃣ Résumé de la solution : réactivation du feather radial/alpha par tuile dans reproject+coadd (CPU/GPU) et pipeline qualité final (lecropper + crop % maître) hors SDS ; couverture SDS/two-pass alignée sur le même weighting.
- [x] 2️⃣ Diffs et contraintes : zemosaic_worker.py (pondération radiale Phase 5, pipeline qualité activé, loader two-pass SDS), zemosaic_utils.py (tile_weight_maps CPU/GPU), tests/test_phase5_blending.py, tests/test_sds_postprocessing.py ; pas de changement de routage GPU/GUI.
- [x] 3️⃣ Tests ajoutés/mis à jour : `tests/test_phase5_blending.py` (poids par tuile + couverture radiale + passage des flags radiaux en Phase 5), `tests/test_sds_postprocessing.py` (pipeline qualité appliqué hors SDS). Commande : `pytest tests/test_phase5_blending.py tests/test_sds_postprocessing.py -q --capture=no` (capture désactivée pour contourner un bug pytest local).
- [x] 4️⃣ Validation visuelle & recommandations : à confirmer sur dataset utilisateur ; flags conseillés = radial_weight + quality_crop/altaz activés, two_pass_coverage_renorm selon besoin (défaut conseillé ON), CPU/GPU sur même dataset. Logs à surveiller : `[Phase5] Reproject & coadd running on ...`, `run_info_phase5_finished_reproject_coadd` et, si two-pass actif, `[TwoPass] Second pass requested` puis `[TwoPass] coverage-renorm OK`; aucune alerte `lecropper pipeline skipped`.
