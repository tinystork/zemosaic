# R41 — Cohérence RAM du layout Auto et du plan d'exécution

Date : 2026-10-16. Statut : diagnostic vérifié / plan proposé, PAS implémenté.
Baseline locale : beta fa72db956469d03a9cbc4104da7a8029308b1461, propre avant cette note.
Objet : refus ModeInfeasible rapportés après les commandes directes global/per_cell.
Ce plan de réparation mémoire ne qualifie aucun correctif photométrique.

## 1. Diagnostic

Dans src/zemosaic/zemosaic_zegrid_mode.py :
- ligne 3198 : layout Auto testé avec ram_budget=available, soit 100 % ;
- ligne 3446 : plan d'exécution testé avec 0.8 * available_now ;
- ligne 3540 : admission stricte par plan_cell_concurrency ; si CPU et aucune
  option admissible, exception sans raffinement du layout ni nouvelle recherche
  de la tuile CPU au budget courant.
- Entre les deux mesures se trouve le calcul de la jauge globale : l'erreur n'est
  donc PAS nécessairement antérieure à tout travail coûteux.

Le premier scan s'arrête au premier layout admissible au budget trop généreux.
Plus de RAM peut produire moins de cellules, donc des cellules plus coûteuses,
ensuite refusées au budget d'exécution. Fermer les applications ne répare pas
cette incohérence. Il faut conserver le refus strict, corriger l'admission amont.

Deux échelles distinctes :
1. cellules de mosaïque (nx x ny, p.ex. 4x3 -> 5x3 -> 6x4) ;
2. blocs internes streaming (128,112,96,80,64,48,32).
L'estimation n'est PAS monotone en taille du bloc. 32 n'est pas toujours le
meilleur choix ; toute l'échelle doit être évaluée.

## 2. Preuve dynamique bornée

Sonde : /home/tristan/.openclaw/workspace/.a2a-reports/ZM-ZEGRID-R41-RAM-PLAN/probe.py
Résultats : même répertoire, probe.json.
816 descripteurs qualifiés, 0 rejet, canvas 4117x4377, sources lecture seule.
Aucun empilement/reprojection, aucune sortie utilisateur modifiée. Le chargeur
existant load_frame_wcs accède toutefois à .data pour obtenir la forme : la sonde
n'était donc pas strictement header-only en I/O, contrairement à l'annonce initiale.

Les bornes des deux erreurs sont reproduites sur les géométries correspondantes :

| Géométrie | Bloc | In-memory MiB | Streaming MiB | Budget utilisateur MiB |
|---|---:|---:|---:|---:|
| 5x3 (15) | 128 | 136220,8 | 17229,4 | 15574,8 : refus |
| 4x3 (12) | 80 | 164395,4 | 18655,4 | 17836,1 : refus |
| 6x4 (24) | 128 | 78995,6 | 11224,7 | passe aux deux budgets |

Les blocs 128/80 sont inférés par correspondance des bornes, pas lus dans les
logs Windows manquants. Le meilleur bloc de l'échelle est 64 dans ces cas :
- 5x3 : 15648,8 MiB, encore au-dessus de 15574,8 ;
- 4x3 : 18587,1 MiB, encore au-dessus de 17836,1.
Donc réduire le bloc seul ne suffit pas aux deux configurations rapportées.

Simulation de RAM constante (pas reconstitution des variations de RAM Windows) :
- layout 100 % puis admission 80 % : refus aux deux budgets ;
- budget identique dès le layout et à l'exécution : 6x4/128 passe à 15574,8 MiB ;
  5x3/128 passe à 17836,1 MiB ;
- 6x4 épinglé : 11224,7 MiB et PASS aux deux budgets.
Cela prouve le défaut de contrat sans dépendre d'une baisse de RAM pendant la jauge.
Ce PASS est un PASS du planificateur, pas la preuve d'un futur run complet.

## 3. Historique et erreurs de conseil à corriger

- Le passage du budget brut au layout date de 83ac030 (2026-10-07) ; le facteur
  0.8 côté exécution de 52eb5a7 (2026-10-08). R36/492c91d (2026-10-13) rend le refus
  strict et vérifie notamment les bornes host au vrai bloc GPU.
- auto_layout.py et parallel.py n'ont aucun diff entre 10dcba3 et fa72db9.
  Le défaut existe sur beta actuelle mais n'est pas introduit par le dernier
  lot R40. L'appel direct non épinglé a exposé un défaut auparavant masqué ici.
- R37E et R40C réussis sont épinglés 6x4. Leurs blocs CPU sont 128 ; R37 a utilisé
  le GPU, R40C le CPU. Ce ne sont pas les mêmes paramètres que les one-liners.
- Les one-liners omettaient zconfig : Auto layout, CPU par défaut, kappa_sigma par
  défaut ; ils ne garantissaient ni le même layout entre eux ni les réglages des
  captures. Le prétendu A/B « seul gauge change » n'était pas garanti.
- Le défaut de jauge actuel est auto (resolve_cell_gauge_mode), pas global ;
  des commentaires anciens dans le caller sont trompeurs. Global forcé ne se
  réduit pas à « offsets OFF » et peut retrouver le défaut de cellules non ancrées
  déjà vu sur Caldwell. Ne pas relancer la nuit un A/B mal défini pour tester la RAM.
- Les 653 anciennes entrées venaient du plan partiel ; ne pas attribuer ce nombre
  à une datation « avant SIP » sans preuve.

## 4. Plan d'implémentation produit borné

### Lot 1 — Contrat commun, deux seuils explicites : planifier à 80 %, admettre à 90 %

Amendement demandé et accepté avec Tristan le 2026-10-16. Il remplace la proposition
initiale 80 % / 80 %, sans changer les preuves historiques de la section 2.
Ce n'est PAS un simple remplacement global de RAM_SAFETY_FRACTION par 0.9.

Définir deux budgets à partir de la RAM DISPONIBLE (pas la RAM physique totale) :
- `B_plan = floor(0.80 * available_plan)` : budget de construction du plan ;
- `B_admit = floor(0.90 * available_now)` : plafond d'admission du plan existant,
  avec une nouvelle mesure juste avant l'exécution.

Le layout, les blocs CPU/GPU et la concurrence sont choisis sous B_plan. Leur
empreinte simultanée totale estimée doit tenir dans ce budget : pas uniquement
la borne d'une cellule si plusieurs tournent ensemble. Expliciter les buffers
communs, la construction du cache et les allocations déjà résidentes ; employer
la même comptabilité à chaque phase, sans omission ni double comptage.

À la seconde phase, accepter le plan inchangé si sa borne totale tient dans
B_admit, même si elle dépasse désormais 80 % de la RAM disponible. Le seuil de
90 % est uniquement une tolérance d'admission : il ne doit ni grossir les cellules,
ni augmenter les blocs ou le nombre de workers/cellules simultanés. La réserve
estimée à l'admission reste de 10 % de la disponibilité remesurée. Ce n'est pas
une garantie contre une allocation externe survenant après la vérification.

Exemple avec la même comptabilité des allocations : 20 GiB disponibles au planning
-> budget 16 GiB. Un plan borné à 16 GiB reste admissible si la disponibilité
passe à 18 GiB (plafond 16.2 GiB) ; il doit être replanifié si elle passe à
17 GiB (plafond 15.3 GiB). Pour un plan remplissant exactement son budget initial,
la baisse tolérée est au maximum 1 - 0.8/0.9, soit environ 11.1 %, pas 20 %.

Créer un contrat partagé portant les deux observations, facteurs et budgets,
le layout, les blocs, les bornes par cellule ET agrégées, le backend et la
concurrence. Appliquer chaque facteur UNE fois à son observation brute respective :
jamais 0.9 * B_plan. Conserver les modèles d'estimation prudents, aucun bypass ni
admission forcée à concurrence=1 si une seule cellule dépasse encore le plafond.
Le 80/90 concerne la RAM hôte ; la politique de réserve VRAM reste inchangée.

Réutiliser ce contrat dans les préflights R37/R40, sans nouveau runner CLI : le
préflight construit à 80 %, il ne promet pas la disponibilité future. Faire cette
admission initiale AVANT jauge/cache coûteux, puis revalider à 90 % de la RAM
remesurée après la jauge et avant lancement des workers. Ne pas modifier les
règles des workers de jauge, Legacy ou SDS par effet de bord d'une constante globale.

### Lot 2 — Replanification bornée avant le premier stack

Si la borne du plan existant dépasse B_admit :
1. construire un nouveau plan sous 80 % de la disponibilité courante, pas 90 % ;
2. réduire d'abord la concurrence si cela suffit, sans changer la géométrie ;
3. évaluer toute l'échelle de blocs au layout actuel (pas supposer 32 optimal),
   le vrai coût host GPU et les replis CPU autorisés ;
4. en Auto seulement, poursuivre vers les layouts plus fins, dans l'ordre existant
   et avec les floors scientifiques conservés ; accepter seulement un plan joint
   dont l'empreinte simultanée respecte le nouveau B_plan ;
5. reconstruire memberships, spécifications, estimations/concurrence et toute
   décision dépendant du layout, notamment résolution auto de la jauge ;
6. au maximum deux replanifications dues à des changements de RAM, puis refus
   explicite et chiffré. Jamais de boucle de retries infinie.

Avant lancement, le nouveau plan repasse l'admission à 90 % de la disponibilité
remesurée. La recherche ne doit pas regrossir le layout ni augmenter la concurrence
lorsqu'une mesure ultérieure retrouve plus de RAM. Aucun redécoupage après qu'une
cellule scientifique a été empilée. Réutiliser seulement les données indépendantes
du layout dont la validité est établie ; jamais un cache de cellule sous une
nouvelle géométrie. Ne pas refaire une coûteuse jauge globale sans nécessité.

Un layout épinglé doit RESTER épinglé : adapter concurrence/streaming si possible,
sinon refus précoce et suggestion de layout plus fin, jamais changement silencieux.
Les captures fondées sur une géométrie fixe restent reproductibles.

Un plan de fallback CPU est construit à 80 % et admis à 90 % au moment du repli,
après libération des ressources de la tentative GPU. Sans plan CPU valide, refus
explicite ; ne pas transformer le repli en dépassement mémoire ni redécouper une
mosaïque déjà partiellement calculée.

### Lot 3 — Logs et tests de propagation

Journal/manifeste : disponibilités aux deux phases, facteurs 0.80/0.90, B_plan,
B_admit, ancienne/nouvelle grille, bloc CPU/GPU, bornes par cellule et agrégées,
backend, concurrence, cause/compteur de replan, admission via la tolérance 80–90 %.
Présenter des messages utilisateur compréhensibles, pas seulement ModeInfeasible.

Tests nécessaires :
- défaut 100 %/80 % reproduit à RAM constante, puis résolu ;
- cas 15 cellules et 12 cellules avec les budgets rapportés (tables de bornes
  déterministes, complétées par la sonde réelle sans full stack) ;
- RAM stable, en baisse entre layout/jauge/exécution, en hausse ;
- 20 -> 18 GiB avec plan 16 GiB : admis sans modifier le plan ;
- 20 -> 17 GiB avec plan 16 GiB : replan sous 13.6 GiB, pas sous 15.3 GiB ;
- frontière exacte B_admit acceptée, un octet au-dessus refusé/replanifié ;
- tolérance 90 % n'augmente jamais concurrence/bloc/taille des cellules ;
- empreinte agrégée refusée si elle dépasse le plafond même lorsque chaque cellule
  isolément tient ; pas de double application des facteurs ;
- cas où un bloc plus petit suffit ET où le layout doit raffiner ;
- mémoire non monotone : 64 peut être meilleur que 32 ;
- épinglé : géométrie inchangée ou refus ; aucun candidat ne passe -> refus borné ;
- propagation réelle plan -> executor -> bloc/concurrence réellement utilisés ;
- GPU réel bloc/host, indisponibilité et fallback CPU sous budget ;
- petite intégration synthétique par le vrai point d'entrée avec RAM contrôlée,
  préflight et runtime cohérents (pas seulement tests unitaires de deux planners) ;
- régressions R36/R37/R40, Legacy/SDS, Windows/spawn.

À layout, science et backend fixés : sorties inchangées. En Auto, un layout corrigé
peut modifier la science per_cell existante ; ne pas promettre une identité entre
layouts. C'est une réparation de faisabilité, pas une correction des coutures.

### Lot 4 — Revue et validation Windows courte

Revue indépendante, commit local sous autorisation de mission, publication beta
selon l'autorisation utilisateur applicable. Sur Windows : préflight produit
cohérent avec le vrai plan d'exécution AVANT de demander un nouveau run nocturne.
Une fois accepté, un run de confirmation RAM suffit ; aucune nouvelle paire A/B
scientifique exigée pour prouver cette réparation de planning.

## 5. Périmètre / état

Zones attendues : orchestration zemosaic_zegrid_mode.py, éventuel helper de planning
ZeGrid, préflight partagé accepté et tests. Ne pas modifier les formules mémoire
pour les rendre artificiellement optimistes, canonical_*, calibration, reprojection,
rejet, poids, Legacy ou SDS. Aucun nouveau bouton requis.

Réalisé : inspection code/historique, simulation sur géométrie réelle, plan écrit,
amendement 80 % planning / 90 % admission approuvé par Tristan. Les nouveaux cas
80/90 sont des critères de tests à implémenter, pas des mesures déjà exécutées.
Non réalisé : patch produit, tests de correctif, commit/push, nouveau run Windows.
Prochaine action : implémenter Lots 1-3 comme un correctif de planning borné,
puis revue et validation courte. Aucun nouveau run utilisateur demandé à ce stade.
