# ZM-SDS-R35C — M106 : diagnostic et plan isolé SDS

Date : 2026-10-13. Statut : **DIAGNOSTIC CONFIRMÉ / PLAN PROPOSÉ** ; **Lots C1, C2 et C3 IMPLÉMENTÉS** (interop FITS des intermédiaires SDS ; finition SDS : origine de niveau, fusion insensible au signe, unité de couverture, ALPHA réintersecté ; C3 : calibration inter-mégatuile robuste sur origine de niveau commune + options honnêtes requested/effective), lots C4–C6 non implémentés.
Demande : comparer SDS et ZeGrid, expliquer les niveaux/histogrammes et les intermédiaires noirs sous Linux, proposer une voie streaming/GPU sans modifier les autres modes.

## Périmètre et baseline

- Dépôt : `/home/tristan/.openclaw/workspace/projects/zemosaic`.
- Branche `beta`, HEAD `ad970f60386d76a93c12420391e9b7aa18eb1b1c`, arbre initial propre.
- Le worker du runtime ZeAlfie `rt-76adb5483c83` est identique octet par octet au worker du dépôt ; l'ancien slot `rt-810c47b7891a` ne l'est pas. Les helpers de finition/normalisation analysés sont identiques dans les deux slots.
- Entrées/sorties M106 uniquement lues. Aucun FITS, cache, configuration ou journal utilisateur modifié/supprimé. Aucun commit/push.
- Référence ZeGrid : `/home/tristan/M106/OUT/R35_A/mosaic_grid_science.fits`, coverage et manifeste correspondants. Ne pas utiliser le snapshot du dossier parent pour identifier ZeGrid : il a été réécrit par un run Classic ultérieur.
- SDS : `/home/tristan/M106/OUT/R35_A/sds_fixed/zemosaic_MT0_R66.fits`, coverage, sept `sds_tiles_1gsio_x3/sds_tile_*.fits`, configuration et journal.
- Le dossier parent contient AUSSI les sorties Classic `zemosaic_MT12_R66_*` ; elles ne sont pas ZeGrid.
- Deux runs de 66 brutes, mais traitements/configurations effectifs différents : pas de promesse d'égalité scientifique SDS/ZeGrid.

## Résumé des constats

### 1. Intermédiaires noirs : correctif NaN R26 absent de cet appel

Les sept primaires SDS sont float32 (`BITPIX=-32`), sans BSCALE/BZERO, et contiennent respectivement **19,317 %, 15,200 %, 17,947 %, 16,006 %, 18,503 %, 16,293 %, 16,421 % de NaN**. Les données finies ne sont pas noires.

`_persist_sds_tile` (`zemosaic_worker.py:38097`, appel du writer vers 38141) utilise `save_fits_image` sans `sanitize_nonfinite=True` ; sa valeur par défaut est False. Le fallback Astropy écrit aussi directement les NaN. Le correctif R26 était explicitement limité à l'export final : l'oubli de ce site explique la réapparition de l'incompatibilité connue ASIFitsView/Gwenview Linux.

Le final SDS est, lui, entièrement fini et porte `ZNFILL=0`, `ZNFREPL=5698887` : **corriger seulement les NaN intermédiaires ne corrigerait pas l'histogramme final**.

Témoin A/B créé hors des données utilisateur : `sds_tile_0001.nanfill_only.fits`. Seule transformation : 4 214 184 NaN remplacés par zéro, pixels finis vérifiés exactement identiques, WCS et ALPHA conservés. L'ouverture effective de cette copie dans ASIFitsView Linux n'a pas été réalisée dans ce diagnostic.

### 2. Cause principale du double histogramme : mélange de deux origines photométriques

Chaîne réelle :

1. `_reproject_frame` (worker vers 37024) soustrait la médiane de chaque canal lorsque `match_background=True` — cas de ce run. Les mégatuiles travaillent donc sur un signal signé proche de zéro.
2. Le coadd des mégatuiles est canonique CPU, avec `normalization='none'`, `weighting='none'`, rejet winsor et moyenne.
3. `two_pass_coverage_renorm=True` relance une coaddition des mégatuiles. La voie GPU historique (`zemosaic_utils.py:6090`, `_finalize_match_background`) soustrait le percentile 0,1 % s'il est négatif : elle ajoute ainsi un piédestal, **séparément par canal**.
4. `_apply_two_pass_coverage_renorm_if_requested` (worker vers 2815) calcule `lum2/(lum1+eps)` puis choisit pixel par pixel la seconde passe si ce ratio est >=0,4. Sur des premières passes signées proches de zéro, cette décision sélectionne les luminances positives et garde les négatives à l'ancienne origine.

**Preuve sur données réelles, sans rerun lourd** : reconstruction des sept mégatuiles signées en retirant leur offset HISTORY ; réempilement canonique sur une grille prenant un pixel sur huit, sans interpolation (WCS des mégatuiles et du final reliés par translations entières vérifiées). Comparaison au final dont le seul shift d'export +477,327423 a été retiré.

- 40 611 pixels communs valides.
- 20 037 pixels restent au résultat de la première passe ; écarts p1–p99 d'environ ±0,00008 ADU (arrondi de la reconstitution float32).
- 20 574 pixels ont changé ; décalage médian R/G/B **+320,924 / +468,506 / +466,833**.
- Classification changé/inchangé = signe positif/négatif de la luminance avant finition, **100 % d'accord sur l'échantillon**.

C'est la signature exacte du raccordement de deux passes à niveaux incompatibles, pas un simple autostretch. Le replay est échantillonné ; il ne constitue pas une revalidation complète de toute la Phase 5.

### 3. Défaut connexe : carte coverage à unités mélangées

La couverture de première passe additionne les contributions des brutes dans les lots. La seconde passe compte les mégatuiles via supports binaires. Le même `np.where` remplace aussi la couverture.

Sur le même échantillon :

- pixels restés en première passe : couverture **24 à 58** ;
- pixels remplacés : couverture **6 ou 7**.

Un même FITS `BUNIT=count` mélange donc deux unités. Cette incohérence concerne les données, pas seulement leur présentation.

### 4. ALPHA final incohérent et champ réduit

- Final SDS : ALPHA est **255 partout**, y compris sur **1 899 629 pixels** dont coverage=0 et RGB exporté=0, soit **42,226 %** du rectangle final.
- `_derive_final_alpha_mask` (worker:21980) privilégie un alpha fourni sans le réintersecter systématiquement avec la validité finale. La finition SDS part aussi avec `final_alpha_map=None` ; la validité calculée en amont n'est donc pas transportée explicitement jusqu'au bout.
- La politique réelle `sds_min_coverage_keep=0.4` masque tout pixel sous 40 % du maximum (58), soit sous 23,2 avant la seconde passe. Ce choix de configuration puis l'autocrop expliquent une part importante de la réduction de champ ; ce n'est pas à confondre avec le bug NaN.
- SDS final : 1723×2611, **2 599 124 pixels valides** ; ZeGrid science : 2403×3278, **6 668 452 pixels valides**. Les surfaces sont différentes ; comparaisons histogramme faites après alignement WCS et intersection de supports.

Ne pas supprimer silencieusement la politique 0,4 : expliciter ce qu'elle exclut et séparer cette décision produit de la correction des masques/unités.

### 5. Autre instabilité scientifique : gain calculé sur médianes quasi nulles

`_normalize_sds_megatiles_photometry` (worker:5026) applique `abs(médiane_ref) / abs(médiane_tuile)` après la soustraction du fond. Le dénominateur est donc un résidu de bruit/fond, pas une mesure robuste de gain photométrique.

Après retrait des décalages d'export, les sept médianes réelles valent **+1,229980, +1,229980, +1,229980, −1,229980, +1,229980, +1,229980, +1,229980**. La prise de valeur absolue conserve même une tuile au signe contraire. Les dispersions et amplitudes stellaires varient fortement entre mégatuiles.

Les grands niveaux des FITS intermédiaires sont également influencés par le writer : chaque tuile reçoit un shift de minimum indépendant, enregistré dans HISTORY (**+1336 à +5858 ADU**). Ce n'est ni une échelle BZERO ni le niveau interne de calcul. Le chemin nominal utilise les tableaux en mémoire, non les fichiers réouverts ; ne pas attribuer le double histogramme au seul writer. Le fallback qui relit ces fichiers mérite un contrat de roundtrip explicite.

Les gains initiaux exacts ne sont pas récupérables dans le journal conservé : INFO_DETAIL n'a gardé que les clés sans leurs arguments. Les effets de cette normalisation doivent être isolés dans une ablation dédiée ; elle ne doit pas être présentée comme l'unique cause du double histogramme déjà prouvé.

### 6. Paramètres affichés et paramètres appliqués ne coïncident pas tous

Le snapshot annonce `sky_mean`, `noise_variance` et égalisation RGB désactivée. Les deux appels canoniques SDS imposent `none/none` ; la préparation active pourtant `use_align_helpers=True` puis appelle `equalize_rgb_medians_inplace`. Cela interdit une comparaison naïve des options GUI avec celles réellement exécutées. La correction devra tracer requested/effective et faire respecter explicitement le contrat SDS, sans toucher au moteur canonique.

## Comparaison quantitative/visuelle

Sur 162 486 échantillons WCS communs (un pixel sur quatre du final SDS ; interpolation bilinéaire de ZeGrid science, support nearest) :

| Indicateur | ZeGrid science | SDS final |
|---|---:|---:|
| médianes R/G/B exportées | 1223,64 / 1551,69 / 1181,64 | 722,39 / 869,59 / 875,89 |
| largeur p90−p10 R | 31,79 | 422,40 |
| largeur p90−p10 G | 35,52 | 601,01 |
| largeur p90−p10 B | 33,02 | 586,67 |

Une différence de fond absolu seule n'est pas un défaut ; ici la bimodalité et les discontinuités spatiales sont confirmées, ainsi que leur apparition à la finition. `comparison.png` montre le même ciel avec un stretch commun et une soustraction de médiane par canal uniquement pour la visualisation. Ce n'est pas une correction des FITS.

## Performance : ne pas optimiser uniquement la reprojection

Durées des runs conservés, **pas un benchmark A/B contrôlé** :

- SDS total journalisé : **2307,03 s = 38 min 27 s**.
- ZeGrid : environ **1176 s = 19 min 36 s** en additionnant les étapes principales et finalize (ne pas additionner cache_build et per_cell_stack imprimés avec la même durée).
- Classic du même dossier : **825,61 s = 13 min 46 s**.
- SDS observé ≈1,96× ZeGrid et ≈2,79× Classic. Configurations, caches, pression mémoire et finitions diffèrent : ces ratios ne garantissent pas un gain après correction.

Ventilation SDS par événements du journal :

| Poste | Durée | Lecture |
|---|---:|---|
| prétraitement + préparation | ~34 s | faible |
| préparation/reprojection/accumulation/écriture des 66 patches | **395,82 s** | borne large, pas reprojection pure |
| finalisation canonique des 7 lots | **1148,42 s** | dominant |
| normalisation/export intermédiaires + stack global + opérations associées | ~532 s | intervalle non finement instrumenté |
| finition SDS/second passage | ~181 s | contient le défaut photométrique |
| sortie/preview/nettoyage | ~12 s | faible |

Le GPU est demandé mais la construction des mégatuiles et leur empilement imposent `backend='cpu'`. L'ancien GPU global-coadd est volontairement désactivé pour divergences scientifiques (`legacy_combine_disabled`). Ne pas le réactiver. Le drapeau GPU demandé force en outre `max_parallel_megatiles=1`, même lorsque le backend réel de ces étapes est CPU.

Le stack final matérialise les sept grandes images et les cubes/temporaires du moteur canonique. Les tableaux images seuls représentent ~965 Mio (7×3331×3615×3×4), sans leurs copies, couvertures et buffers float64. Le journal voit le swap système passer d'environ 4,2 à 9,2 Gio (ce n'est pas une attribution de tout le swap au process SDS).

**Micro-mesure existante** : sur un patch réel M106 TAN 512×512 RGB, trois répétitions :

- reprojection SDS actuelle (3 appels `reproject_interp`) : médiane **0,7933 s** ;
- helper R21 existant (`reproject_cropped_fast`) : **0,3158 s**, soit **×2,51** ;
- support identique, pixels communs exactement égaux, max_abs=0.

Cette preuve est locale et CPU ; ce n'est ni une qualification TAN/SIP complète ni un gain bout-en-bout. Même supprimer tout le poste 395,82 s ne réduirait ce run que d'environ 17 %. D'où priorité au streaming du stack, pas à une nouvelle reprojection GPU seule.

## Plan séquentiel de correction — SDS seulement

### Lot C1 — sorties intermédiaires compatibles Linux (petit, indépendant)

> **Statut C1 (2026-10-13) : IMPLÉMENTÉ (mission ZM-SDS-R35C-C1).** L’export
> des intermédiaires SDS est délégué à un helper dédié `_write_sds_tile_fits`
> (worker) qui réutilise `save_fits_image(..., sanitize_nonfinite=True)` ; le
> fallback Astropy assainit désormais NaN/+Inf/−Inf → 0 sur une **copie**
> float32 en conservant WCS et ALPHA, avec `ZNFILL`/`ZNFREPL`/HISTORY. Les
> tableaux scientifiques en mémoire restent inchangés ; le défaut global de
> `save_fits_image` reste à `False` ; ZeGrid/Legacy/canonical non touchés.
> Tests : `tests/test_sds_intermediate_fits_interop_c1.py` (11 cas) + R26/F5/F6.
> Détail et preuves : rapport `ZM-SDS-R35C-C1.coco.r0.md`. Témoin visuel
> ASIFitsView Linux/Windows à faire côté ZeAlfie (non exécuté ici).

- Activer le helper R26 à la frontière d’export `_persist_sds_tile`, y compris fallback Astropy ; ne pas changer le défaut global du writer.
- Préserver les pixels finis, WCS, ordre CHW, ALPHA ; enregistrer ZNFILL/ZNFREPL/HISTORY. Garder les NaN dans le calcul.
- Pour toute relecture interne : no-data déterminé par masque explicite, jamais par `pixel==0`. Ne pas interpréter le remplissage comme une mesure.
- Définir séparément le roundtrip des shifts de baseline : cache scientifique signé inchangé / export de consultation, ou offset réversible explicitement relu. Ne pas introduire une modification photométrique globale du writer.
- Gates : roundtrip RGB/mono, NaN/Inf/zéros/négatifs valides, input non muté, fallback testé, témoin ASIFitsView Linux et Windows. Résultat nominal calculé avant/après export identique.

### Lot C2 — corriger la finition SDS, les deux origines et les unités (prioritaire)

> **Statut C2 (2026-10-13) : IMPLÉMENTÉ (mission ZM-SDS-R35C-C2).** Le correctif
> est **opt-in** (`sds_level_origin_fix`, défaut `False`) et n'est activé que par
> `_finalize_sds_global_mosaic` (voie SDS). Il (a) aligne l'origine de niveau de
> la 2e passe sur celle de la 1re avant toute décision, (b) remplace la décision
> `lum2/(lum1+eps)>=0.4` (piloteé par le signe) par une décision de support
> insensible au signe, (c) conserve une carte de couverture à unité unique
> (couverture de 1re passe), et (d) réintersecte l'ALPHA final avec le support
> final (`coverage<=0 => ALPHA==0`). Classic/ZeGrid/canonical non touchés : le
> chemin par défaut des fonctions partagées est inchangé. Détail et preuves :
> rapport `ZM-SDS-R35C-C2.coco.r0.md`. Constats C3–C6 ci-dessous non modifiés.

- Sauvegarder un témoin SDS scientifique **avant finition**, plus les étapes after-polish/export, pour une ablation reproductible.
- Ne jamais mélanger une passe signée et une passe relevée au percentile. Garder les décalages de visionneuse uniquement à l'export/preview.
- Adapter uniquement le chemin SDS : contrat de sortie signée du second passage et référence photométrique cohérente ; remplacer la décision `lum2/lum1` lorsqu'elle dépend du signe d'un signal centré. Aucun patch du comportement par défaut Classic.
- Recommandation : science SDS canonique conservée comme produit immuable ; les transformations esthétiques non équivalentes (second combine mean vs winsor notamment) restent dans un produit esthétique séparé. Ne pas faire passer une désactivation silencieuse d'une option pour une correction.
- Garder une carte de couverture à unité unique. Comptage brutes et support mégatuiles sont des produits distincts, pas deux valeurs interchangeables.
- Propager et réintersecter l'ALPHA SDS avec le support final après toute opération/crop ; `coverage==0` doit impliquer `ALPHA==0`. Ne pas utiliser la positivité du flux comme validité.
- Gates : cas signé traversant zéro, piédestal RGB par canal, second pass ON/OFF, CPU/GPU, absence de ruptures dépendantes du signe, coverage d'unité stable et alpha cohérent. Rejouer les vrais témoins ci-dessus puis M106 complet en nouvelle sortie scratch.

### Lot C3 — stabiliser la normalisation SDS et rendre les options honnêtes

> **Statut C3 (2026-10-13) : IMPLÉMENTÉ (mission ZM-SDS-R35C-C3).** Mesure
> préalable sur les vraies mégatuiles M106 (lecture seule) : après retrait des
> décalages d'export, les médianes signées valent ≈ ±1,3 ADU alors que le fond
> par brute avant soustraction vaut ≈ 1040–1609 ADU. Le recouvrement commun
> valide des mégatuiles (toutes sur la même grille) montre des **échelles
> relatives réalisées de ≈ 1,0× à ≈ 5,0×** (ratios de flux de sources brillantes),
> et ces mêmes facteurs expliquent les rapports de bruit par tuile : signature
> d'un facteur multiplicatif par tuile, c.-à-d. le gain `|médiane_ref|/|médiane_|
> tuile|` calculé sur des médianes quasi nulles. Le défaut est donc démontré et
> matériel. Correctif **dans le périmètre SDS uniquement** :
> `_normalize_sds_megatiles_photometry` ne dérive plus aucun multiplicateur de
> fonds déjà soustraits ; `method="none"` est un vrai no-op neutre, et
> `method="sky_mean"` applique un **décalage additif de niveau** par canal aligné
> sur une référence partagée explicite, sur le **recouvrement commun valide**,
> en réutilisant la statistique canonique `_sky_mean_channel` (a == 1). Un mode
> non supporté lève explicitement (jamais de no-op silencieux). La méthode
> inter-mégatuile effective est dérivée de la normalisation **demandée**
> (`stack_norm_method`) et journalisée requested/effective pour les deux étages
> SDS (brutes→mégatuile, mégatuiles→final), y compris l'écart d'égalisation RGB
> (`use_align_helpers` applique `equalize_rgb_medians_inplace` alors que
> `poststack_equalize_rgb` est demandé désactivé) — l'écart est journalisé
> explicitement, sans changer le sens de l'option. Tests :
> `tests/test_sds_robust_photometry_c3.py` (10 cas) + mise à jour des
> caractérisations F6/Phase4.5. Aucun rerun M106 lourd ; Classic/ZeGrid/canonical
> non touchés. Détail et preuves : rapport `ZM-SDS-R35C-C3.coco.r0.md`.

- Remplacer le ratio de médianes quasi nulles par un contrat SDS de calibration **avant centrage**, avec référence explicite et statistiques sur recouvrements communs valides. Réutiliser les fonctions canoniques existantes, ne pas inventer un gain depuis des fonds déjà soustraits.
- Premier candidat : `sky_mean` avec référence partagée, comme l'intention GUI ; `none` doit être réellement neutre. Ne pas introduire un nouveau modèle affine spatial ou étendre le chantier scientifique Caldwell.
- Conserver l'identité de chaque brut et chaque lot SDS. Pas de changement de grouping pour maquiller le résultat, pas d'égalisation RGB cachée.
- Clarifier la pondération et le rejet à chacun des deux étages SDS (brutes→mégatuile, mégatuiles→final). Ne pas substituer un poids scalaire de lot à une profondeur locale sans validation. Un empilement hiérarchique avec rejet n'est pas associatif : SDS n'a pas à être bit-identique à ZeGrid.
- Journal durable requested/effective, références, coefficients par canal/lot, médianes avant/après, supports et exclusions.
- Gates : invariance à un offset de fond ajouté, médianes nulles/négatives, niveaux stellaires/injection faible signal préservés, options RGB/normalisation effectivement propagées. Mesurer fond résiduel, sauts aux limites, photométrie d'étoiles, bruit sur support commun ; pas seulement l'apparence de l'histogramme.
- Ne pas changer silencieusement `sds_min_coverage_keep=0.4`. Exposer combien de pixels cette règle écarte. Une baisse du seuil est une décision de qualité/couverture distincte.

### Lot C4 — streaming CPU exact et mémoire bornée

> **Statut C4 (2026-10-13) : IMPLÉMENTÉ (mission ZM-SDS-R35C-C4).** Nouveau module
> `core/sds_streaming.py` (aucun `core/canonical_*` modifié) : provider SDS
> fichier/memmap des deux étages — `SdsMegatileFileProvider` (mégatuiles -> final)
> et `SdsPatchBandProvider` (brutes -> mégatuile, bandes horizontales sur les
> patches memmappés existants) — réutilisant tel quel
> `run_canonical_stack_streaming`. Les DEUX étages SDS passent par l'exécuteur
> streaming (stack final plein-canvas en priorité), avec tuilage SPATIAL
> uniquement (aucune fragmentation de N), statistiques globales calculées une
> fois (invariance à la taille de bloc), supports/no-data exacts, fermeture
> explicite des memmaps (Windows) et repli in-memory explicite (WARN) en cas
> d'échec. Parité CPU **bit à bit** vs le moteur canonique in-memory prouvée pour
> plusieurs tailles de blocs et ordres de scheduling. Mémoire bornée **mesurée**
> (pic RSS ajouté 1,72 Gio in-memory -> 420/362 Mio streaming à 1280², ×4,2–4,9).
> Plan de concurrence borné par backend **effectif** (CPU) + RAM, pas par la
> préférence GPU. Témoin réel I1 : sur `sds_fixed` (pré-C3), l'alignement additif
> C3 laisse un résidu **multiplicatif** inter-mégatuiles (facteurs par tuile
> 1,00–4,95×, span 4,95×) → documenté candidat **C5**, non corrigé ici. Détail et
> preuves : rapport `ZM-SDS-R35C-C4.coco.r0.md`. Constats C5–C6 non modifiés.

- Ajouter un provider **spécifique SDS** (patches/mégatuiles memmappés), réutilisant `run_canonical_stack_streaming` sans modifier les formules `canonical_*` ni les dispatchs ZeGrid/Classic.
- Streamer les deux étages, surtout le stack final actuellement full-canvas. Toutes les images du lot contribuent à chaque tuile spatiale pertinente : ne pas fractionner scientifiquement N, ni moyenner des sous-stacks de rejet.
- Calculer les statistiques globales une fois, indépendamment de la taille des blocs ; provider exact pour les supports et le no-data. Fermer les memmaps explicitement pour Windows.
- Éviter les copies conservées en listes de tous les tableaux, les accumulateurs morts pour le chemin canonique et les reprojections répétées de tuiles déjà sur la même grille quand une identité WCS est réellement prouvée.
- Plan de concurrence selon backend **effectif** et RAM disponible, non selon seule préférence GPU ; ne pas augmenter la concurrence avant d'avoir borné les allocations.
- Gates : parité CPU bit à bit avec la baseline scientifique corrigée pour plusieurs tailles de blocs et ordres de scheduling, même support/rejet/couverture ; mémoire bornée mesurée, pas seulement estimée ; fonctionnement RAM basse et fermeture Windows.

### Lot C5 — accélération reprojection CPU puis GPU canonique optionnel

- CPU : réutiliser R21, grille de coordonnées partagée RGB, travail borné aux empreintes et aux blocs. Fast path uniquement WCS qualifiés ; conserver reproject_interp pour SIP/distorsions et cas non qualifiés. Ne pas retirer les distorsions ni remplacer pixel_to_pixel par une approximation.
- Empilement GPU : réutiliser le backend **streaming canonique** déjà présent depuis R22 (l'ancien docstring CPU-only de canonical_streaming est périmé ; le code accepte bien cpu/gpu). Réutiliser le planificateur VRAM et la classification d'erreurs existants via un adaptateur SDS.
- Un propriétaire GPU, VRAM réelle bornée, réexécution entière de l'unité concernée sur CPU en cas d'erreur CUDA qualifiée, aucun assemblage de morceaux issus de deux backends. Tracer le backend réellement exécuté par étape.
- Reprojection GPU pure : phase suivante seulement si profil résiduel le justifie. Séparer resampling et combine ; ne pas remettre en service l'ancien GPU coadd avec correction de fond/quantiles non canoniques. Qualifier bords/NaN/support/precision CPU vs GPU et coût PCIe.
- Gates : TAN/CD, TAN/PC+CDELT, angles, bords, canaux partiellement invalides, SIP fallback ; science/support/rejet identiques pour CPU exact et contrat GPU existant. A/B réel sur MX150 2 Gio et, si ciblée, machine RTX, sans promettre que le GPU est plus rapide partout.

### Lot C6 — acceptation isolée / non-régression

- Même 66 brutes, mêmes options effectives, même grouping, mêmes finitions pour l'A/B perf, scratch distincts, état caches et swap consignés.
- Baseline science = SDS **corrigé**, puis comparer streaming OFF/ON et GPU OFF/ON. Ne pas exiger d'un optimiseur qu'il reproduise les bugs de finition actuels.
- Objectif initial mesurable proposé : ≥30 % de baisse du temps SDS total sur le témoin contrôlé, aucune hausse du pic RAM, science/couverture invariantes pour les optimisations seules. À réévaluer après C2/C3 ; aucun gain n'est acquis à ce stade.
- Tests ciblés : SDS F5/F6 et formes de poids, R26 export, canonical streaming, R21 reprojection, CPU/GPU. Témoins ZeGrid et Classic conservés à sorties/science/configuration identiques ; pas de modifications dans leurs dispatchs ni formules communes.
- Revue indépendante du diff (en particulier helpers partagés), puis validation Windows/Linux via ZeAlfie. Publication seulement dans le cadre de l'autorisation applicable à la future implémentation ; ce document ne publie rien.

## Invariants / hors périmètre

- Aucun changement des formules `canonical_*`, algorithmes ZeGrid, science Legacy, solver/WCS sources, locales/GUI sans nécessité liée à l'honnêteté des options SDS.
- Aucune nouvelle dépendance majeure, aucun nouveau modèle scientifique global, aucun relâchement de tests, aucune suppression de données.
- Aucune reprise automatique ni overwrite des runs M106 actuels.
- Mécanique et correction scientifique séparées en lots : interop, finition, photométrie, mémoire, accélération.
- Prochaine action recommandée : **C1 puis C2**, avant C3 et avant toute accélération. Aucun worker d'implémentation lancé pour cette demande de diagnostic/plan.

## Artefacts reproductibles

Répertoire : `/home/tristan/.openclaw/workspace/.a2a-reports/ZM-SDS-R35C-diagnostic/`

- `inspect_fits.py`, `fits_stats.json` : statistiques complètes des primaires/HDU.
- `compare.py`, `comparison.json`, `comparison.png` : supports communs WCS et visualisation.
- `replay_final_sample.py`, `replay_final_sample.json`, `replay_final_sample.npz`, `stage_comparison.png` : preuve première passe vs final.
- `reproject_probe.py`, `reproject_probe.json` : micro-mesure existante R21, vraie source M106.
- `timings.json` : ventilation par lot tirée des timestamps du run.
- `sds_tile_0001.nanfill_only.fits` : témoin A/B de compatibilité, pixels finis inchangés.

Limites : pas de rerun complet M106, pas de test interactif ASIFitsView, pas de mesure reprojection GPU nouvelle, pas de correction production. Le RuntimeWarning `invalid value encountered in subtract` dans canonical_stacking:1405 est réapparu sur le replay à supports troués ; le replay a terminé et les pixels valides ont été vérifiés. Ce warning n'explique pas la bimodalité de la finition et n'a pas été masqué/modifié.
