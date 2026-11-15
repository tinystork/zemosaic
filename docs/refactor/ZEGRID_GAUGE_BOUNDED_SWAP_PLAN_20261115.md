# ZeGrid — budget jauge avec secours swap borné

Date : 2026-11-15 — **PLAN v3 / EXÉCUTION G0→G4 AUTORISÉE — G0 EN COURS**.
Orientation explicite Tristan : budget dynamique à 90 % du swap disponible
utilisable, sans plafond fixe en Gio ; cache récupérable déjà compté dans RAM.
Mandat actualisé : Tristan autorise la mission complète G0→G4 sur la nuit,
implémentation et qualification comprises, avec point demain sur ce chantier et M33.
Aucun push, publication, installation runtime ou changement du run M33 autorisé.

## 1. Baseline et constat vérifiés

- Source pertinente : `/home/tristan/.openclaw/workspace/worktrees/zemosaic-adaptive-resources`,
  branche `feat/zegrid-adaptive-resources`, HEAD `e25fb5133768adf4eb26950c63f86b970909b370`.
- Le checkout documentaire historique `projects/zemosaic` est plus ancien (`ec708a1`).
- Égalité SHA-256 des quatre fichiers source/runtime vérifiée :
  `zemosaic_zegrid_mode.py`, `core/zegrid/scheduler.py`,
  `core/zegrid/resource_envelope.py`, `core/zegrid/runtime_control.py`.
  Runtime témoin : slot ZeAlfie `rt-13ba3ae6df9b`, Python 3.13.
- Dernière exception NGC7023 : 2026-11-14T22:29:00Z, dans
  `/home/tristan/NGC7023/out/instrum/worker_crash_breadcrumbs.jsonl`.
- Jauge compute, 1 worker, `pairs_normalize`, bbox 8 914 455 px,
  empreinte brute 2 203 281 820 B ; pic +15 % = 2 533 774 093 B.
- B_plan = 1 664 142 540 B ; dépassement = **869 631 553 B = 829,345 Mio**.
- Les admissions RAM ET external échouent sur ce même terme jauge.
  Le spool a été évalué, pas admis avec succès. Il est incorrect de conclure
  « external sélectionné » depuis les seules lignes de layout CHOSEN.
- L'hôte possède du swapfile ET de la zram : SwapFree global n'est pas
  assimilable à autant de mémoire disque indépendante disponible.
- Aucun benchmark nouveau ni test moteur exécuté pour cette analyse documentaire.

## 2. Décision proposée

Conserver la politique RAM-only pour tous les chemins actuels ; ajouter un secours
**opt-in, borné, propre à la jauge globale calculée avant le stack**.
Aucun changement de calcul scientifique, référence, dtype, ordre des réductions,
normalisation, poids, rejet, géométrie ou backend stack.

Deux restrictions structurantes :

1. Essayer la jauge RAM-only avec les ladders existantes jusqu'à 1 worker.
   Le swap ne sert jamais à augmenter la concurrence. Si un worker ne tient pas,
   et si la jauge est le seul obstacle RAM restant, essayer le secours à 1 worker.
2. Le crédit expire à la sortie du calcul de jauge. Cache, stack, GPU staging,
   assembly et export restent sous leurs budgets RAM/VRAM actuels.

Périmètre v1 : jauge globale pré-stack, CPU stack et GPU stack, classe RAM ou
external. Ne pas étendre implicitement aux jauges per_cell exécutées pendant les
cellules : elles nécessitent une preuve séparée de non-chevauchement avec le stack.
Une jauge restaurée/importée conserve sa tarification résidente sans crédit compute.

## 3. Budget proposé (paramètres à qualifier, pas valeurs déjà validées)

Notations : A = RAM disponible effective, cache récupérable inclus ; S = swap
libre réellement utilisable après restrictions OS/cgroup, sans double comptage
zram ; R_min = réserve système minimale éventuelle, à évaluer pendant la revue.

```text
B_ram_plan  = floor(4*A/5)       # inchangé
B_ram_admit = floor(9*A/10)      # inchangé
S_credit    = floor(9*S/10)     # direction demandée, sans plafond fixe en Gio
B_gauge     = B_ram + S_credit
P_gauge     = ceil(P_gauge_raw * 115/100)
credit_needed = max(0, P_gauge - B_ram)
admis ssi P_gauge <= B_gauge ET toutes les phases sans crédit sont admissibles
```

Option à évaluer par Nono, pas décision déjà validée :
`S_credit = min(floor(9*S/10), max(0, S - R_min))` pour que 10 % d'un petit
swap ne laisse pas une réserve trop faible. Ni R_min ni une nouvelle valeur par
défaut ne sont arbitrairement fixés avant la revue.

Politique proposée : `gauge_memory_policy = ram_only | bounded_swap`, défaut
ram_only à évaluer ; en bounded_swap, fraction entière 9/10, pas de paramètre
limitant à 1 Gio. Les crédits peuvent dépasser largement 1 Gio si S le permet.
Les faits RAM/swap sont rééchantillonnés avant lancement. Les 10 % ne sont pas
une réservation OS et n'éliminent pas l'OOM ni le risque de thrashing.
Ne pas remplacer B_ram par A sans facteur ; ne pas ajouter le cache à A :
MemAvailable intègre déjà sa part récupérable. Ne pas ajouter SwapCached à SwapFree.

Sur le dernier refus, le déficit estimé vaut 869 631 553 B. Il faudrait au moins
966 257 282 B de S utilisable pour le couvrir avec la seule règle 90 %, à faits
RAM inchangés, hors réserve minimale supplémentaire. Ceci lèverait ce refus de
jauge, pas nécessairement les autres refus du pipeline.

## 4. Sens exact de « borné »

Le crédit dynamique borne le dépassement d'enveloppe AUTORISÉ, pas les pages que
l'OS choisit de mettre en swap. L'estimation mémoire n'est pas une limite physique.
L'OS peut évincer d'autres applications, conserver des pages en RAM ou refuser une
allocation ; aucun mécanisme portable ne réserve ici le crédit swap au processus.
Pas de changement de swappiness, pas de mlock, pas de création/modification du swap.
Un quota physique strict exigerait un autre chantier d'isolation de processus.

Capacités :

- Linux v1 : lire les faits RAM, swapfile/partition et les limites effectives du
  cgroup courant, y compris ses ancêtres contraignants. Réduire le crédit si le
  swap est interdit/plafonné, même lorsque l'hôte en possède beaucoup.
- Zram : exclue du crédit additionnel v1 ; la compression consomme de la RAM et
  son taux n'est pas garanti. Ne pas additionner aveuglément tous les SwapFree.
- Vérifier la méthode de disponibilité swap par périphérique : les compteurs
  globaux ne suffisent pas toujours ; documenter toute estimation conservatrice.
- Tenir compte de la marge de commit lorsque l'OS l'impose ; ne pas confondre
  MemAvailable, commit disponible et pagefile libre.
- Windows : adaptateur spécifique pagefile/commit avant activation sur Windows ;
  `psutil.swap_memory().free` seul n'est pas une preuve suffisante de capacité.
- Faits de crédit inconnus/non fiables : crédit 0 avec raison explicite ; le
  chemin RAM-only continue s'il tient. Pas de crédit illimité par défaut.

## 5. Implications code

### Contrat et sérialisation

`core/zegrid/resource_envelope.py` impose actuellement
`host.budget_bytes == floor(factor*raw_available/den)`.
Préserver ce champ comme budget RAM réel. Ajouter une politique de crédit de
phase explicite, validée/versionnée, au lieu de gonfler raw_available ou factor.

`scheduler.py:build_plan_set` et `build_joint_plan_set` comparent aujourd'hui le
maximum global à un plafond unique. Créer un évaluateur commun par groupes de
chevauchement : marge 15 % UNE fois par groupe ; crédit seulement pour le groupe
singleton gauge compute. Un groupe gauge chevauchant une phase non éligible est
refusé en secours v1. Garder le vrai pic agrégé dans les diagnostics : ne jamais
soustraire le crédit du pic pour fabriquer une empreinte RAM plus faible.

Conserver les erreurs typées et distinguer refus RAM, crédit insuffisant,
capacité inconnue et disque plein. Compatibilité stricte des anciens payloads et
hashes lorsque la politique est absente/ram_only ; nouvelle politique versionnée
pour les nouveaux plans. Ne pas ajouter les observations volatiles à l'identité
scientifique durable ; auditer séparément le hash du plan d'exécution.

### Câblage produit

Dans `zemosaic_zegrid_mode.py` :

- faits/bundle de jauge et worker sizing ; `_r55_active_facts`,
  `_r55_active_phase_requests`, décisions CPU/GPU/external et sondes layout ;
- même évaluateur à tous ces points : aucun refus précoce avec ancien plafond,
  aucun contournement tardif du validateur final ;
- vérifier d'abord les autres phases RAM-only. Une jauge swap-feasible ne doit
  ni masquer un stack impossible ni provoquer un fallback GPU/CPU injustifié ;
- sonde fraîche avant `zphot.compute_global_gauge`, workers effectifs = 1 pour
  le secours ; résultats persistés atomiquement selon le contrat existant ;
- sortie de jauge : terminer les workers/libérer les temporaires normalement,
  fermer le crédit, puis réadmission RAM-only fraîche avant cache/stack ;
- ne pas transmettre le crédit aux dispatchers, au prefetch, aux réservations de
  cellule, au staging CUDA ou au contrôleur de pression RUN.

`photometric.py` : noyaux mathématiques inchangés. Ajouter au plus le raccord
observation/annulation manquant ; ne pas introduire de subdivision ou de passe.
`parallel.py` : workers plafonnés par l'admission, jamais recalculés avec RAM+swap.
`runtime_control.py` : fournisseur de faits de capacité séparé ; préserver les
faits RAM-only et le contrat RUN 90 % existants.

### Configuration et UX

Exposer la politique dans la configuration et le snapshot GUI→worker,
avec validation moteur autoritaire. Tester le chemin public complet, pas seulement
la sauvegarde du réglage. Le contrôle GUI doit expliquer : « Autoriser un secours
swap limité pour la jauge ; peut ralentir ; aucun crédit pour le stack ».
L'option est masquée/désactivée avec raison sur plateforme non qualifiée.
Désactivation = comportement RAM-only, sans modifier les stores existants.

## 6. Observabilité honnête

Événements start/admit/progress/end : policy, source des mesures, A, budget RAM,
S effectif, fraction 9/10, réserve éventuelle, crédit disponible/nécessaire, pic + marge,
workers demandés/effectifs, classe stack/backend, motifs de refus, durée.

Ne pas appeler le crédit nécessaire « swap réellement utilisé ». Rapporter
séparément les observations disponibles : swap processus/arbre, RSS, défauts de
pages majeurs, compteurs swap-in/out système, pression mémoire/IO, avec portée et
raison d'indisponibilité. Les deltas système ne sont pas attribuables au seul run ;
la somme RSS peut compter deux fois des pages partagées ; le RSS ignore le swap.

Heartbeat léger pendant la jauge, annulation coopérative aux frontières sûres.
Pas d'arrêt sur le seul fait d'avoir swappé ni de timeout fixe arbitraire qui
casserait une longue jauge 20k. En cas de capacité dégradée : ne plus admettre de
nouvelle tâche si les conditions ne tiennent plus, puis arrêter proprement à une
frontière sûre. Pas de garantie de rattrapage d'un OOM-kill au milieu d'une paire.
Après jauge : journal explicite `gauge_credit_closed`, puis budget stack sans crédit.

## 7. Plan d'exécution séquentiel

| Lot | Livraison | Gate mesurable |
| --- | --- | --- |
| G0 — contrat | Figer paramètres, sources Linux/cgroup/zram, périmètre, seuils perf et oracles ; replay read-only NGC7023 | Refus exact reproduit ; disponibilité crédit définie sans double comptage ; baseline enregistrée |
| G1 — politique pure | Fournisseur de capacité + évaluateur par groupes + schéma/hash versionnés | Matrice seuil exact/+1 B ; RAM-only inchangé ; aucun groupe stack éligible |
| G2 — câblage | CPU/GPU/external, layout/preflight, jauge 1 worker, configuration/snapshot, fermeture crédit et réadmission | Chemin public synthétique fonctionne ; couper la transmission de policy fait échouer le test ; aucun crédit après jauge |
| G3 — qualification | Parité, pression réelle contrôlée, télémétrie et Stop/Resume | Résultats exacts, aucune perte de commits, mesure des coûts et limites physiques |
| G4 — revue/clôture | Nono indépendante, vérification Junior et documentation utilisateur | Findings matériels résolus ; résultats bornés aux plateformes réellement testées |

Chaque lot d'implémentation est une mission Coco bornée, asynchrone, rapport
durable et callback exact ; revue puis acceptation Junior avant le lot suivant.
Revue Nono seule autorisée à ce stade ; aucune implémentation Coco lancée.
Aucun commit/push/installation implicite.

## 8. Matrice de validation

- Policy OFF : sorties et décisions actuelles conservées, pas seulement pixels.
- RAM suffisante + policy ON : aucun secours, même concurrence que le chemin normal.
- RAM insuffisante, crédit suffisant : jauge compute 1 worker admise ; plafond
  exact accepté, dépassement de 1 octet refusé, marge appliquée une seule fois.
- Swap absent/interdit/inconnu, zram seule, faible swap disque, cgroup contraint,
  changement de disponibilité entre plan et launch : refus ou revalidation explicite.
- RAM-only impossible ailleurs (stack/cache/assembly/export) : reste refusé,
  quel que soit le plafond de jauge ; CPU/GPU/external testés séparément.
- Jauge restaurée/importée : compute non exécuté, pas de crédit inutile,
  identité et commits inchangés ; anciens stores toujours lisibles.
- Per-cell : aucune extension du crédit à une jauge imbriquée dans le stack.
- Stop pendant jauge : pas d'artefact partiel annoncé complet ; recalcul possible
  si le contrat courant ne sait pas reprendre une jauge partielle.
- Comparaison science : à workers identiques, coefficients, poids, exclusions,
  référence et sorties science/coverage identiques bit-à-bit entre RAM et secours.
  Comparer aussi le chemin multiworker habituel ; tout écart impose analyse,
  aucune tolérance nouvelle introduite pour faire passer la qualification.
- Tests ciblés existants : `test_zegrid_r41_ram_contract.py`,
  `test_zegrid_r55e8_coherent_resource_adaptation.py`,
  `test_zegrid_restored_gauge_admission.py`, `test_zegrid_active_external_wiring.py`,
  et suites scheduler/envelopes/runtime/identité/reprise affectées.

Qualification physique en sorties neuves, corpus utilisateur en lecture seule :

1. Petit synthétique pour le bout-en-bout et les injections de capacité.
2. NGC7023, copies de sorties séparées : témoin RAM confortable ; puis secours
   sous pression contrôlée et autorisée. Comparer jauge et produits finaux.
3. Mesurer séparément effet du passage à 1 worker et effet de pagination :
   temps jauge, temps total, pic arbre, swap observé, IO, latence Stop.
   Un faux MemAvailable prouve la logique, PAS la réalité du swap.
4. Pour 20k : probe bornée représentative avant tout run complet ; ne pas
   supposer la jauge courte ni extrapoler un gain depuis NGC7023.

Les seuils de ralentissement et d'annulation seront gelés en G0 avant les mesures.
Sur machine confortable, pas de changement de chemin ; sous pression, le produit
promet une possibilité supplémentaire, pas la vitesse du mode tout-RAM.

## 9. Conclusion

Modification faisable sans retoucher la science ni découper davantage la jauge.
Elle n'est pas un changement de formule isolé : admission multi-phase, limites OS,
compatibilité reprise, configuration et diagnostics doivent converger.
Recommandation à examiner : secours Linux opt-in, 90 % du swap utilisable sans
plafond fixe en Gio, 1 worker,
RAM-only inchangé pour le stack, pas d'extension external GPU dans cette mission.


## 10. Mandat de revue v2

Évaluer le travail réel et les angles morts, indépendamment du plan proposé :
- correction de la formule 90 %, réserve minimale, capacité non paginable,
  commit limits, courses entre processus, zram/zswap, cgroups et plateformes ;
- faisabilité d'un crédit par phase sans fausser les agrégats et hashes ;
- revue critique du choix un worker, opt-in et périmètre global/per_cell :
  propositions architecte, pas exigences explicites de Tristan ;
- monitoring/annulation réalistes en cas de thrashing, reprise sans perte,
  coût de collecte et garanties honnêtes sur OOM et débit ;
- éventuels contrôles RAM résiduels en amont/aval et tous les chemins CPU/GPU/external ;
- charge relative par lot, dépendances, critères bloquants avant implémentation,
  tests manquants et proposition de découpage minimal.
Rapport avec constats sourcés, sévérité, décisions nécessaires et plan recommandé.
Pas de modification du moteur ni du plan par le reviewer.


## 11. Revue Nono reçue et arbitrage Junior (2026-11-15)

Mission `ZM-GAUGE-DYNAMIC-SWAP-PLAN-20261115`, review-0, verdict Nono FINDINGS.
Rapport : `/home/tristan/.openclaw/workspace/.a2a-reports/ZM-GAUGE-DYNAMIC-SWAP-PLAN-20261115.nono.r0.md`.
Revue de plan TERMINÉE ; implémentation NON LANCÉE. G0 doit fermer les définitions
ci-dessous avant qu'un développeur dispose d'un contrat directement implémentable.

Vérifications indépendantes Junior : branche/HEAD conformes, état des fichiers
préexistants inchangé ; scheduler.py:2283 et :2672 comparent toujours le maximum
avec un unique plafond ; mode.py:10975–11022 dimensionne les workers et distingue
la jauge restaurée du compute. Aucun test ni run de pression exécuté.
Les compteurs système actuels ne sont pas ceux de l'incident : ne pas mélanger
MemAvailable relevée aujourd'hui avec le déficit du run historique.

### Constats retenus

- M1 : rendre la capacité de commit opérationnelle dans le contrat, en fonction
  du mode d'overcommit et de la plateforme ; simple mention qualitative insuffisante.
- M2 : calculer le swap disque par périphérique ; exclure zram du crédit ; tracer
  les sources et ne pas utiliser SwapFree global comme preuve de disque libre.
- M3 : résoudre le cgroup effectif et ses ancêtres ; prévoir séparément une
  contrainte RAM résidente/non paginable et une contrainte swap. Un fichier absent
  ne prouve ni limite infinie ni swap interdit : établir absent/non applicable/inconnu.
- M4 : qualification physique et coûts séparés (1 worker vs pagination), sans
  promettre un débit ni l'impossibilité d'un OOM. Aucun auto-timeout ajouté ici.
- L1 : évaluer une réserve minimale complémentaire aux 10 %, sans plafond en Gio
  sur le crédit. La valeur chiffrée suggérée par Nono n'est PAS encore validée.
- L2–L5 : schéma/hash d'exécution explicite, compatibilité des anciens plans,
  limite per_cell documentée, adaptateur Windows séparé, charge réelle de Stop.

### Corrections aux recommandations de Nono

1. M1 : `CommitLimit - Committed_AS - B_ram` ne peut pas être adopté tel quel
   comme formule portable. L'hôte contrôlé par Junior est en overcommit mode 0,
   pas 2. Il faut comparer le commit disponible à une estimation des NOUVELLES
   allocations engageant du commit, sans recompter les allocations déjà présentes.
   G0 doit définir ces bases et la réserve ; pas soustraire automatiquement B_ram
   d'un headroom dont les sémantiques diffèrent. Sous Windows, même distinction.
2. M3 : `S_disque - non_pageable_reserve` n'est pas une preuve de faisabilité
   RAM résidente. Les pages non paginables ne peuvent pas être financées par du
   swap. Il faut un garde RAM distinct, avec comptage du socle existant et des
   nouvelles allocations ; éviter de redéduire d'A les ressources déjà consommées.
3. M5 : alternative intéressante, NON démontrée et non autorisée comme nouvelle
   implémentation par la demande de revue. `photometric.py:1115` appelle le noyau
   canonique ; `canonical_stacking.py:517,709,915` utilise des statistiques sigma-
   clippées globales centrées sur médiane, et le fit robuste utilise aussi des
   médianes (:683). Calculer ces statistiques séparément par sous-blocs n'est
   pas équivalent. Aucune preuve de bit-parité, de gain ou d'absence de passes
   supplémentaires ; ne pas transformer cette suggestion en prérequis d'un
   nouveau chantier. G0 peut comparer la faisabilité sur papier, tout prototype
   de noyau borné demande un périmètre propre et une qualification scientifique.
4. La phrase « le plan ignore zram/cgroup/commit » est trop forte : v2 les cite
   explicitement. La lacune réelle est une définition opérationnelle incomplète,
   pas une absence totale de prise en compte. La qualification reste à produire.

### Contrat de capacité à finaliser en G0

Conserver la direction utilisateur : crédit au plus 90 % du swap utilisable,
sans plafond fixe en Gio. Définir séparément :

- S_effectif : swap disque libre, limité par headrooms swap des cgroups ;
- S_credit : au plus floor(9*S_effectif/10), diminué seulement si une réserve
  système minimale définie exige plus de 10 % ;
- garde RAM résidente/non paginable : impossible à compenser par S_credit ;
- garde commit selon l'OS et son mode : nouvelles allocations vs headroom ;
- garde par groupes : jauge seule éligible, aucune autre phase subventionnée.

Le crédit admissible n'est donc pas une nouvelle valeur globale de RAM disponible.
Une incapacité de prouver une condition requise interdit le secours, pas un run
RAM-only déjà admissible. Les marges et bases doivent être définies sans double
comptage. Aucun plafond arbitraire de 1 Gio n'est réintroduit.

## 12. Charge et prochaine étape après revue

Découpage recommandé, charge RELATIVE (pas estimation en jours) :

| Lot | Charge | Point dominant |
| --- | --- | --- |
| G0 contrat | S à M | Définir bases commit/RAM/cgroup, réserve, oracles et seuils |
| G1a capacité Linux | M | Collecte fiable, capabilities et cas limites système |
| G1b admission | M | Groupes, marge unique, hash/compatibilité |
| G2a câblage | L | Tous les early gates CPU/GPU/external et fermeture du crédit |
| G2b UX et Stop | M | Snapshot public et frontières coopératives vérifiées |
| G3 qualification | L | Pression réelle, parité, temps/IO et reprise |
| G4 clôture | S | Revue, vérification et limites publiées |

Travail le plus délicat : capacité réellement utilisable et câblage cohérent de
TOUTES les admissions. Aucun chiffre de durée fiable avant G0. Windows est un
lot additionnel, pas une capacité annoncée par la qualification Linux.

Prochaine action proposée : une mission G0 documentaire/pure bornée, sans
implémentation produit ni test de pression sur l'hôte courant. La présente demande
autorisait plan + revue seulement ; aucun dispatch Coco n'a été effectué.


## 13. Mandat nocturne et pilotage actif

Tristan autorise explicitement G0→G4 le 2026-11-15. Les mentions antérieures
« revue seule autorisée / implémentation non lancée » décrivent l'état historique
avant ce mandat et ne limitent plus la mission.

État durable de pilotage : `docs/refactor/ZEGRID_GAUGE_SWAP_EXECUTION_STATE_20261115.md`.
G0 lancé d'abord ; Junior vérifie puis enchaîne les lots bornés par callbacks,
avec revue indépendante et corrections si nécessaires. Aucun reset en cours de lot.
Pas de commit, push, merge, déploiement ni modification des données originales.
Qualification physique uniquement isolée et bornée, sans stress global dangereux.
M33 reste intact ; point de situation demandé demain, pas de modification implicite.
