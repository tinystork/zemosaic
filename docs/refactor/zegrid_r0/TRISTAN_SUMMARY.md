# ZM-ZEGRID-R0 — synthèse pour Tristan

**R0 STATUS: ACCEPT CANDIDATE.** Junior : acceptation technique ; Nono review-0 : **ACCEPT**, aucun finding HIGH/MEDIUM sur la validité de R0. Gate humain en attente.

## Trois découvertes majeures

1. **La géométrie Grid historique n'est pas réutilisable telle quelle** : échelle CD mal interprétée et offset non intégré au WCS final. Sur M106, le helper renvoie 1°/pixel au lieu d'environ 0,000659°/pixel.
2. **L'ordre physique et les chunks changent la science**, malgré `order` CSV inerte : référence index 0, ties canoniques par index, populations de rejet/médiane séparées. Un appel canonique par chunk ne constitue pas un stack canonique global.
3. **La localité doit préserver la préparation des données** : alpha source non reprojeté actuellement, support canonique perdu dans le wrapper ; Bayer dépend du min/max pleine frame. Un halo ne rend pas les statistiques locales équivalentes aux statistiques globales.

## Architecture recommandée

**GlobalCanvas → ZeGridLayout → Cell → ProcessingPatch = Cell + contexte → SCI-05 → MiniTile du patch entier → crop du cœur → placement déterministe.**

Structures séparées : FrameDescriptor, GlobalCanvas, ZeGridLayout, ZeGridCell, ProcessingPatch, CellMembership, SourceCropPlan, MiniTileGeometry/Result. Cell ne chevauche pas ses voisines ; halo jamais compté deux fois dans la science/support final.

- **Réutiliser :** moteur/API SCI-05, primitives WCS, sérialisation/telemetry, infrastructure de preview Qt et principe de placement sans reprojection finale.
- **Abandonner dans le futur chemin :** gros GridTile confondant propriété/calcul, stack par chunks d'expositions, validité déduite de luminosité, assemblage photométrique/blending historique hérité sans qualification. Rien supprimé en R0.
- **Déterminisme :** IDs stables de manifest, tri des membres avant géométrie et SCI-05, auto-référence canonique max support avec tie stable par identité, tous les contributeurs dans une seule requête R1, placement par Cell ID.
- **Auto à benchmarker :** dimensions du canvas et médianes séparées largeur/hauteur des footprints ; facteur variable sous contraintes mesurées mémoire/support/qualité. Aucune constante ni gagnant produit décidé.
- **Halo :** contexte cible lié au rayon du taper ; marge d'interpolation source distincte. Pas de halo photométrique magique ni blend imposé.

## Preuves

**Real Seestar geometry witness: PASS**, borné à la géométrie : 66/66 brutes M106 S50, canvas 2403×3278 ; 2×2, 5×4, 7×5, 9×7 avec halos 0/8/32. Géométrie/membership/mesures identiques en ordre normal/inversé/aléatoire. Nono a reproduit toutes les mesures ; Junior a comparé les JSON : égalité complète.

**29 tests ciblés PASS**, relancés indépendamment par Nono. Aucun test de science réelle multi-cellules ni gain de temps prétendu. **Phase 4.5 impact: NONE OBSERVED** sur cette route ; aucune modification Classic/SDS/SCI-05/Phase4.5.

## R1 recommandé — périmètre exact soumis à accord

Une MiniTile réelle, Cell M106 **r0000c0000**, layout témoin 5×4 : cœur 480×819, patch 488×827 avec contexte h=8 pour le taper actuel, **7 candidats patch**. Entrées RGB préparées une seule fois via un décodeur existant figé ; leur préparation pleine frame est déclarée séparément, pas vendue comme du CFA local.

Lecture par sections → reprojection locale → une requête canonique CPU → résultat complet + crop Cell. Oracle : reprojection pleine source **vers le même patch**, puis même moteur canonique. Qualification CFA locale et raccords entre cellules restent des missions ultérieures. Aucun moteur Nx×Ny complet, GUI, GPU ou progressive preview.

## Décision demandée

Valider ou amender : architecture ; Cell/Patch/Halo ; politique déterministe ; première famille Auto à mesurer ; stratégie de reprojection locale ; R1 sur RGB préparé avec une seule Cell.

**Recommandation Junior : approuver ce R1 borné.** Alternative : exiger d'abord la qualification CFA locale, ce qui ajoute un chantier de décodeur distinct. Cet accord est demandé conformément au gate humain final explicite de la mission (§34). **R1 n'est pas lancé.** Aucun commit/push/merge/tag.
