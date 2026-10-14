# R38 — Calibration indépendante du découpage : proposition et plan

Date : 2026-10-14. Statut : PROPOSITION, aucune implémentation autorisée par ce document.
Baseline inspectée : beta / 10dcba3 (arbre propre avant cette note).
Objet : coutures ZeGrid Caldwell Windows 816, pas le chantier SDS.

## 1. Décision proposée

Déplacer la recherche vers la cohérence des données AVANT leur empilement par cellule.
Une cellule doit être une partition de calcul, pas une nouvelle référence photométrique.
Une brute doit recevoir la même correction et la même définition des poids au même
pixel céleste, quelle que soit la cellule qui le calcule.

Il s'agit d'une architecture candidate, pas d'une correction déjà démontrée.
Première étape obligatoire : falsifier ou confirmer la responsabilité des jauges
locales par des expériences contrôlées. Si elles n'expliquent pas le défaut, ne pas
implémenter aveuglément un nouveau solveur photométrique.

## 2. Faits établis et limites

- Capture Windows EXACT : 816 brutes, canvas 4117x4377, 6x4, 24 cellules,
  114 relations canal, SHA256 a260b3309c39fbfc5264cacc60ba3048c678f83cf8bf6bf9f39c94a134a84032.
- R37 M1 : RMS heldout 46,625 -> 18,967 ADU, mais 7,918 sigma,
  conditionnement 14952 et deux relations dégradées : modèle refusé.
- La dépendance au niveau ne prouve PAS à elle seule un défaut de gain : elle peut
  aussi provenir de structure spatiale, PSF, sélection, masques ou pondération.
- STATIC : compute_cell_gauge, dans core/zegrid/photometric.py, recalcule
  compute_fixed_normalization sur le domaine propre à chaque cellule. Référence,
  coefficients, poids et exclusions peuvent donc dépendre du découpage.
- STATIC : canonical_streaming fournit déjà FixedNormalization et l'application
  de coefficients fixes. Une intégration future pourrait réutiliser ce contrat,
  sans modifier les estimateurs canonical_*.
- R33 a déjà invalidé transfert additif simple, affine global et plusieurs variantes
  gain+plan. Refaire le même modèle sous un nouveau nom n'est pas une solution.
- Le cache aligné du run Windows a été supprimé par le pipeline. L'artefact R37
  compact ne permet pas de reconstruire les contributions individuelles.
- Caldwell 816 est désormais DEV/diagnostic. Un autre split aléatoire de ces pixels
  ne devient pas une qualification indépendante. Les anciens témoins 653 ne sont
  pas indépendants s'ils partagent les mêmes acquisitions.

## 3. R38-A — diagnostic causal borné, premier lot exécutable

Livrables : runner offline en lecture seule des sources, capture locale versionnée,
rapport causal, manifeste des sélections et coûts. Aucun câblage production.

1. Sélectionner et figer AVANT comparaison six relations géométriques DEV : les
   cas problématiques, des cas médians/faibles, les deux orientations, plus un coin.
   Déclarer le biais de cette sélection : elle sert au diagnostic, pas à qualifier.
2. Retrouver toutes les brutes des cellules concernées via manifeste/frame_ids.
   Reproduire leurs domaines, références, coefficients, poids et exclusions exacts.
   Ne PAS recalculer la normalisation seulement sur une petite bande : cela change
   précisément l'effet étudié. Lire les domaines complets en mémoire bornée pour
   les statistiques, puis ne restacker que les bandes/ROI nécessaires.
3. Comparer aux observations R37 compatibles et vérifier WCS/TAN-SIP, support,
   reprojection des mêmes pixels, ordre, mode de rejet et précision CPU/GPU.
4. Expérience factorielle sur pixels communs, avant offset/feather/DBE :
   - contributions locales habituelles vs même intersection de frame_ids ;
   - coefficients locaux vs coefficients partagés figés pour les brutes communes ;
   - poids locaux vs poids partagés, en traçant séparément exclusions et masques ;
   - rejet canonique habituel, puis témoin sans rejet UNIQUEMENT diagnostique.
   Pour le témoin partagé, tester les deux choix de jauge et une jauge indépendante
   du découpage ; aucune de ces jauges témoins n'est présumée scientifiquement juste.
5. Pour chaque expérience : RMS/p90 de différence, tendance spatiale et au niveau,
   incertitude par blocs, contributeurs avant/après rejet, valeurs/poids par brute.
   Une même entrée corrigée/pondérée au même pixel doit être identique entre cellules
   sur un backend fixé ; si elle ne l'est pas, isoler d'abord reprojection/support.
6. Une réduction causale reproductible doit apparaître dans les deux orientations,
   pas seulement sur la pire frontière. Rapporter les interactions, sans attribuer
   artificiellement une somme de pourcentages à des effets non additifs.

Gate A : cause dominante démontrée ou rapport INCONCLUSIVE. Pas de run complet
Windows de trois heures pour chaque hypothèse. Préflight RAM/disque obligatoire ;
le coût réel des lectures/ROI est mesuré, aucun délai garanti.

## 4. R38-B — prototype de calibration partagé, conditionnel au diagnostic

### Architecture cible

- Catalogue stable des brutes et identités d'acquisition, distinct du layout RAM.
- Graphe clairsemé des recouvrements réellement mesurés ; références multiples et
  cycles de contrôle, sans imposer un recouvrement direct avec UNE brute centrale.
- Une correction par brute/canal et des poids définis sur un domaine stable,
  réutilisés par toutes les cellules. La jauge globale fixe une origine/échelle
  relative ; elle ne prétend pas retrouver le fond céleste absolu.
- Commencer par la correction minimale justifiée par A. Si un gain est nécessaire,
  le mesurer sur des flux stellaires après soustraction locale du fond, étoiles
  non saturées/non variables, avec contrôle PSF et couleur ; ne pas estimer le gain
  sur la luminosité diffuse servant aussi à mesurer le fond.
- Après fixation des gains admissibles, mesurer les différences additives sur les
  mêmes coordonnées célestes. Séparer signal commun et nuisance instrumentale.
  Un fond spatial n'est admissible qu'avec preuve de leverage/identifiabilité ;
  ni spline libre, ni polynôme d'ordre croissant pour forcer l'accord Caldwell.
- Si la diversité des acquisitions permet un modèle instrumental lié au détecteur,
  l'étudier explicitement ; sinon signaler la dégénérescence, ne pas inventer cette
  information. Des flats/darks ou acquisitions complémentaires peuvent être requis.
- Étoiles insuffisantes, mismatch spectral, cycles incohérents ou composantes non
  ancrées : refus explicite, jamais gain extrapolé ni connexion fabriquée.

Le solveur est pur/offline et clairsemé, sans imports depuis le chemin produit.
Les différences poids/contributeurs seront corrigées séparément si A les incrimine.
Une liste de candidats par cellule ne doit pas supprimer une brute valide pour
un pixel parce que celui-ci se trouve de l'autre côté d'une frontière de calcul.

Gate B : contrat scientifique et famille de modèles figés AVANT validation réservée.
Les nouveaux observables stellaires/identités de brutes doivent apporter une
information distincte des ajustements R33 ; sinon STOP, pas de répétition aveugle.

## 5. R38-C — qualification indépendante

- Séparer DEV, sélection de modèles et TEST final réellement réservé.
- Corpus : scènes synthétiques à vérité connue (étoiles, IFN, gradients partagés,
  faibles gains, gradients instrumentaux, PSF variables, couleurs incompatibles,
  identités/null, supports fragmentés) et acquisitions réelles non consommées.
- Figer les critères avant ouverture du TEST. Si les données réelles disponibles
  ne sont pas indépendantes, statut NEEDS_INDEPENDENT_DATA, pas de faux PASS.
- Reprendre les exigences de fidélité R37 : gradient/IFN conservé à 1 %, biais de
  flux stellaire médian <=0,1 % et p95 <=0,5 %, masques/support sans invention.
  Les étoiles de contrôle doivent être distinctes de celles calibrant les gains.
- Cible résiduelle : RMS des différences <=1,25 sigma (sigma tenant compte des
  contributions partagées et de leur covariance), amélioration >=20 % du baseline,
  aucune relation p90 dégradée de plus de 5 %, les deux orientations améliorées.
  Garder la métrique R37 pour comparabilité, publier séparément toute nouvelle
  estimation du bruit/covariance ; aucune redéfinition après observation du TEST.
- Matrice layout : 6x4, autre layout faisable et origine de grille décalée sur DEV,
  puis validation réservée. Les données par pixel, corrections et poids doivent
  être indépendants du layout ; chiffrer les différences numériques du stack
  séparément des différences scientifiques (pas de promesse bit-exact CPU/GPU).
- Exiger rang/condition/incertitudes et cohérence des cycles adaptés au modèle,
  fixés avant TEST. Ne pas réutiliser aveuglément le seuil de condition d'une
  matrice dont les unités/paramètres ont changé.
- Vérifier étoiles, fond diffus et frontières à plusieurs distances AVANT finition.
  Les coupes intérieures contrôles aident au diagnostic, mais ne sont pas une
  vérité scientifique sur un ciel structuré : aucun simple ratio visuel ne suffit.
- Un TEST consulté devient consommé ; révision du modèle implique un nouveau TEST.

Gate C : tous les critères passent ; sinon refus documenté et pas de promotion.

## 6. R38-D — intégration ZeGrid uniquement, après qualification

- Adaptateur ZeGrid vers les coefficients fixes et la pondération acceptée ; ne
  modifier ni canonical_*, ni Legacy, ni SDS, ni politique WCS/reprojection.
- Si une correction spatiale est indispensable, l'appliquer dans un provider
  ZeGrid expérimental séparé, pas en détournant FixedNormalization scalaire.
- Option expérimentale OFF par défaut, ancien chemin inchangé et testé bit-à-bit
  option OFF. Manifeste : modèle/version, coefficients, sources, incertitudes,
  composantes, backend réellement exécuté, refus/fallback et métriques.
- Aucun filtrage/suppression de brutes silencieux. Repli historique explicitement
  annoncé comme potentiellement porteur de coutures, pas comme correction réussie.
- Garder feather comme assemblage ; ne pas l'élargir pour maquiller le score.
  DBE/égalisation esthétique restent hors qualification science.
- Mémoire bornée, graphes clairsemés, pas de cube N_brutes x canvas. Mesurer temps
  jauge/reprojection/stack, RAM/VRAM/cache, pas de promesse d'accélération.
- Régression Classic/SDS, ZeGrid global/per_cell, TAN/TAN-SIP, FITS R26,
  propagation GUI-worker, CPU/GPU/fallback et Windows/spawn ; revue indépendante.

## 7. R38-E — témoin produit et sortie de mission

Après les gates courts : un run Windows complet Caldwell 816 en sortie neuve
et un corpus indépendant réservé, comparaison scientifique avant/après, manifeste
et captures conservés. Caldwell est une preuve de réparation de ce cas, pas une
preuve indépendante de généralisation.

Publication beta séparée, après décision explicite de Tristan. Aucun tag/release
implicite. Si la calibration n'est pas identifiable : recommander acquisitions de
calibration, regroupement explicite par conditions ou rendu esthétique distinct ;
ne jamais appeler cette dernière option une réparation du FITS science.

## 8. État et prochaine action

Réalisé : inspection du code et des preuves R33/R37 ; ce plan uniquement.
R38-A implémenté (harness + tests + note `R38_A_CAUSAL_PROBE.md`), puis R38-A
**exécuté sur le panneau DEV figé complet** contre l'artefact Windows 816 exact
(HEAD `beta` `ff3803b3`). Détail et résultats : §7 de `R38_A_CAUSAL_PROBE.md` et
rapport `.a2a-reports/ZM-ZEGRID-R38-A-CAUSAL.extended-full-result.md`. Aucun
changement production, aucune correction, aucun câblage.

Résultat final R38-A (vérifié sur l'artefact strict-JSON) :

- artefact `r38a_factors_extended_full.json` : 1 168 958 octets, sha256
  `218187eb1ebb3bf806735cf160fca1580472b164acf0e411af137934bb2685e0` ;
- `status = PASS` (technique : toutes les relations prévues exécutées),
  `qualification = DEV_NOT_QUALIFICATION`, `scientific_acceptance = false`,
  `r38b_authorized = false`, `conclusion = PARTIAL_CAUSE` ;
- 6/6 relations du panneau exécutées, deux orientations ; 5/6 qualifient un
  mécanisme de **niveau additif quasi constant par cellule** (ratio jauge/observé
  entre 0,994589 et 1,002271) ; seul le contrôle vertical à très faible pas observé
  (~0,44/0,66/2,11 ADU, cohérence MAD/pas 1,63–3,23) ne qualifie pas ;
- aucun `NOT_RUN`, fuites de chemins privés = 0 ; déterminisme = aller-retour JSON
  dans le même processus, **pas** une répétition indépendante sur données réelles ;
- le `PASS` technique n'est **pas** une acceptation scientifique.

Note : la règle gelée du harness (§4.7 de la note R38-A) distingue quatre états ;
le résultat est `PARTIAL_CAUSE` — mécanisme qualifié sur 5/6 relations dans les deux
orientations, mais aucune preuve de dominance (indisponible dans un harness borné).

Décision : **R38 est CLOS — arrêté à Gate A.**

- R38-B/C/D/E = **NOT_RUN / NON AUTORISÉS** : non exécutés, et ce résultat ne les
  ouvre pas. Aucune reprise automatique vers R38-B.
- Le mécanisme mesuré est un niveau additif par cellule **avant** offset, déjà
  estimé/retiré par le graphe accepté R34c ; il n'explique pas le **résidu spatial
  POST-OFFSET / POST-FEATHER** et n'autorise donc pas un solveur de calibration
  partagé (R38-B).

Prochain lot recommandé — un **nouveau diagnostic pré-enregistré** (données de
validation indépendantes) ciblant directement le résidu spatial post-offset :
constant vs pente/courbure, biais d'estimation d'offset vs structure de ciel,
support/géométrie/taper, à plusieurs distances et dans les deux orientations.

Aucun changement de comportement produit/runtime/défaut/GUI/FITS dans R38-A.
Aucun changement aux FITS utilisateur, aucun run lourd. Ce lot est
**documentation seulement** : aucun commit/push, diff de travail laissé pour revue.
