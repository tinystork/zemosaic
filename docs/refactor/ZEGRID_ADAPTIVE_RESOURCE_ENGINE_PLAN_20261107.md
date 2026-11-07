# ZeGrid — moteur adaptatif RAM / VRAM / disque, 20 000 images et plus

Date : 2026-11-07  
Mission proposée : **ZM-ZEGRID-ADAPTIVE-RESOURCES-20261107**  
Statut : **READY_NOT_STARTED — plan et contrat de conduite renforcés ; lancement par Tristan encore attendu.**

Version documentaire : **v2, 2026-11-07**. Lire les §10–14 avant toute exécution.
Point d'entrée de reprise : [fiche d'état](ZEGRID_ADAPTIVE_RESOURCE_ENGINE_STATE_20261107.md).

## 1. Décision produit proposée

La mémoire disponible doit déterminer le volume de travail simultané, pas un plafond artificiel du nombre d'images.

Cible : traiter le corpus M33 complet, puis qualifier plus de 20 000 images, avec un budget RAM variable de **1 à 32 Gio pour le moteur**, sans changement de science. « 1 Gio » n'est pas une machine Windows de 1 Gio : distinguer budget moteur, socle processus/GUI/CUDA et réserve système. Le lot A0 doit établir le plancher réel et préciser ce que le plafond mesuré inclut ; ne pas ajouter arbitrairement un socle non mesuré pour déclarer le gate réussi.

Utiliser le GPU dès qu'il est exploitable et utile ; optimiser le débit bout-en-bout, pas le pourcentage de VRAM occupée. GPU indisponible ou VRAM momentanément insuffisante : continuer en CPU exact ou attendre de façon explicite et annulable lorsque même l'unité CPU minimale ne tient plus. Conserver une réserve de sécurité mesurée pour le système et les autres applications.

Cette mission est une évolution de l'exécution et du stockage ZeGrid, pas une refonte de sa science, ni un remplacement par legacy.

## 2. Baseline réellement inspectée

- Référence locale `origin/beta = 515004078cfb64b327fb02aeaa323e48cca62bd1`.
- Sources lues dans `/home/tristan/.openclaw/workspace/worktrees/zemosaic-auto-external-streaming`, branche `feat/zegrid-auto-external-streaming`, HEAD identique ; checkout propre lors de la lecture.
- Référence locale `origin/main = f8acdbb0ec34bef7dd21e69deadf9352103fdc3c`.
- Le checkout documentaire `projects/zemosaic` est plus ancien, HEAD `ec708a1`, beta en retard de 37 commits ; ses modifications documentaires préexistantes sont préservées. Il n'a PAS servi de baseline code.
- Référence externe déjà clonée : `/tmp/astro-stack-ref`, HEAD `c06ba830b4d969c8c9da2beb729298c04995d8c1`, licence MIT.
- Aucun fetch, benchmark, test GPU ou run scientifique frais dans cette réévaluation.

### État de la preuve M33

L'audit précédent rapporte 21 818 frames retenues, layout 12×20 / tuile CPU 20 accepté, puis refus joint `R55_REFUSE_HOST_RAM` avant jauge/stack (run 12:38–12:53). Ce sont des constats antérieurs, pas une nouvelle lecture du run.

La nouvelle tentative de lecture Windows a échoué : `Invalid system.run request: raw-command-mismatch`. Tristan a ensuite indiqué que Windows était planté ; la cause du plantage n'est pas établie par cet échange. Aucun état plus récent du dossier Windows n'est donc certifié ici.

A0 doit capturer les logs/configs/manifestes du dernier run et son vrai build. Ne pas confondre les 22 414 entrées d'une ancienne tentative, les 21 818 frames retenues d'une autre et les N propres à chaque cellule.

## 3. Corrections de la réévaluation précédente

### 3.1 RAM libre ne signifie pas exécution prouvée possible

15 Go libres au moment d'un refus ne démontrent pas que le pic simultané du pipeline tient. Un refus statique peut être trop conservateur, incohérent entre phases, ou justifié par un vrai pic. Il faut réconcilier modèles et allocations, pas supprimer l'admission ni diviser ses coefficients arbitrairement.

### 3.2 La référence ne garantit pas « au moins une ligne = ça passe »

Dans `gpustacker/backend.py:rows_per_band`, le coût courant est N × largeur × canaux × 4 × 6. C'est un **heuristique de 24 B/échantillon**, pas un pic mesuré universel.

Dans `gpustacker/stacking.py:band_overhead`, GESD utilise 14 × 4 = 56 B et le rejet large-scale ajoute encore 2 × 4 B. `_stack_band_rows` peut explicitement refuser si le halo large-scale ne tient pas. Un clamp à une ligne peut dépasser le budget quand une seule ligne est déjà trop grosse.

Le chemin CUDA utilise aussi un ou deux buffers host pinned de taille N×bande. Les copies, le dtype float32 du moteur de référence, les algorithmes et les marges ne sont pas équivalents aux calculs float64 ZeGrid. On ne peut pas conclure que remplacer 96 par 24 suffit à rendre M33 GPU admissible.

### 3.3 Beta possède déjà une partie du mécanisme

Les lots M0–M6 ne sont pas à recommencer :

- petites tuiles CPU, sélection RAM/externe, spool transactionnel ;
- réduction externe utilisant les primitives canoniques ;
- reprise par cellules durables et déclin GPU impossible.

Mais `external_stream.run_cell_streaming_external` refuse actuellement tout backend autre que CPU. `largest_admissible_column_batch` impose au minimum **une ligne intérieure de tuile**, pas une position pixel-canal arbitraire. Le workspace reste O(N × largeur de ligne × canaux), avec d'autres coûts de préparation et de sorties.

Le modèle du réducteur externe compte déjà 31 B/position/frame pour une partie des buffers. Cela ne décrit ni le pipeline complet ni la VRAM.

### 3.4 Le problème n'est pas seulement le stack

- Cache aligné par cellule et spool peuvent dominer le disque.
- Jauge, décodage/reprojection, halos et workers ont des coûts fixes ou liés à la surface.
- Le réducteur externe alloue encore des plans de sortie à la taille du patch.
- L'assemblage canonique décrit 121 B/pixel RGB en plans plein canevas, avant certains transitoires.
- `ScratchCanvasAllocator` existe mais n'est pas raccordé au chemin de production inspecté.
- Finition, preview, hachage et écriture FITS doivent aussi être bornés.
- La reprise possède des contraintes d'admission figée : autoriser de nouveaux lots d'exécution sans invalider la science demande un contrat/versionnement explicite.

Ainsi, les anciens chiffres de téraoctets de cache sont des estimations conditionnelles d'un chemin et d'une géométrie, pas une borne fondamentale du problème M33. Ils doivent être recalculés sur le dernier run.

### 3.5 Science exacte et ordre flottant

Un mean sans rejet admet une accumulation somme/poids, mais un changement d'ordre flottant n'est pas automatiquement bit-identique. Les rejets actuels ne peuvent pas être remplacés par une moyenne de sous-stacks.

Conserver l'axe N complet pour un **petit bloc spatial** est la première solution à qualifier. Des algorithmes multi-passes/external-memory peuvent traiter certaines réductions sans conserver tous les échantillons simultanément en RAM ; leur exactitude et leurs coûts sont à démontrer. Les quantiles/médianes exacts nécessitent notamment sélection/tri externe si même N valeurs d'une position ne tiennent plus.

Aucune promesse « N illimité, RAM quelconque, GPU toujours occupé » : objectif mesurable 20k/50k puis 100k sur témoins spatiaux bornés, et plancher physique explicite.

## 4. Architecture cible

Séparer strictement :

1. **Plan scientifique immuable** : inventaire ordonné, WCS, référence/jauge, grille, halos, taper, poids, rejet, combine, identité de cellule.
2. **Plan d'exécution adaptatif** : bloc spatial, lot de lecture, backend, concurrence, prefetch, stockage et fenêtre de scratch.

Chemin logique :

Sources → préparation locale bornée → cache/spool de fenêtre → réduction canonique CPU/GPU par micro-blocs → résultats durables → assemblage disque par bandes → finition/publication bornée.

### Gouverneur de ressources commun

Comptabiliser sous une même réservation :

- mémoire de base mesurée, enfants inclus ;
- tableaux vivants, transitoires, allocateurs et bibliothèques ;
- staging host et pinned, profondeur de prefetch ;
- GPU : mémoire libre, buffers vivants, pool réservé, contexte et copies en vol ;
- fenêtres mmap/pages résidentes, handles ouverts ;
- disque : cache, spool, résultats, transaction/publication et marge ;
- phases qui se chevauchent réellement, sans additionner arbitrairement des phases exclusives.

Choisir le plus gros bloc utile qui tient. Réduire d'abord les crédits de concurrence/prefetch, puis les dimensions du bloc ; adapter hauteur **et largeur**, éventuellement canaux après preuve de parité. Le halo scientifique ne doit pas imposer la taille du bloc de réduction : préparer son contexte séparément puis consommer seulement les positions nécessaires, sans recréer des N×grands halos en VRAM.

En pression : arrêter les nouvelles admissions, finir ou annuler proprement l'unité en vol, libérer ses ressources, réduire et rejouer cette unité sans double comptage. Croissance prudente à une frontière sûre après stabilité ; hystérésis et nombre de retries borné. Jamais une boucle OOM infinie.

### Stockage de fenêtre

Ne pas exiger le cache aligné complet N×patch de chaque cellule avant tout progrès. Produire/consommer une fenêtre ou un spool borné, puis recycler son espace.

Privilégier les données préparées locales et la réutilisation ; accepter une nouvelle lecture/reprojection si le scratch manque. Calculer les dépendances spatiales de normalisation/taper/reprojection exactement. Une bbox sans padding est une optimisation possible après preuve, pas une solution suffisante à elle seule quand toutes les frames couvrent la cellule.

Séparer **pic disque vivant** et **volume total d'I/O**. Relire plusieurs fois peut être très lent mais doit rester possible et annoncé. Limiter aussi le nombre de fichiers/handles ouverts, important sous Windows à N=20k.

### GPU

Réutiliser CuPy et les primitives ZeGrid qualifiées ; ne pas introduire PyTorch seulement pour imiter la référence.

Exécuter le même micro-bloc de réduction sur GPU lorsque son workspace tient, indépendamment de l'ancien plancher de tuile 32. Distinguer réduction et reprojection : qualifier chaque chemin réellement accéléré.

Double-buffer et transferts asynchrones uniquement sous crédit RAM+pinned+VRAM. Si le disque est le goulot, 100 % d'occupation GPU n'est pas un critère de succès. Aucun remplissage artificiel de VRAM.

### Plancher et erreurs légitimes

Le mécanisme doit dégrader le débit avant de refuser. Un budget insuffisant pour le socle + une unité atomique minimale reste une limite physique documentée. Source invalide, disque indispensable indisponible, droits, corruption et erreur CUDA non récupérable ne deviennent pas des faux succès.

Si le budget tombe sous ce plancher en cours de run : état suspendu/reprenable avec raison, annulation possible et cellules validées conservées ; pas d'attente silencieuse infinie.

## 5. Lots séquentiels et gates

Chaque lot sera découpé en missions Coco bornées, suivies de vérification Junior et revue Nono des changements matériels. Ce document seul n'autorise aucune délégation ni publication ; le contrat de futur lancement est défini au §10.1.

### A0 — Preuve du problème et contrats (premier lot recommandé)

Livrables :

- capture reproductible du dernier M33 : SHA/runtime, inventaire, N_cell, géométrie, paramètres et causes des refus ;
- cartographie complète setup → jauge → cache → stack → assemblage → FITS/finition ;
- pics prédits vs observés par phase CPU/GPU, durées, I/O, fichiers/handles ;
- inventaire des coefficients/marges, duplications et contraintes de reprise ;
- comparaison ciblée avec legacy et la référence, notamment normalisation, rejet, dtype, ordre ;
- témoins de science figés avant modification et définition vérifiable du budget 1 Gio ;
- protocole court des premiers blocs M33 à 20k+ avant toute campagne complète.

Gate : aucune attribution « modèle faux » sans chaîne d'allocations identifiée ; un défaut d'admission reproduit ou une limite réelle quantifiée. Rechiffrer ensuite le reste, sans lancer la suite complète de tests ni des benchmarks longs pour chaque itération.

### A1 — Découplage scientifique/exécution et budget partagé

Livrables : contrat de réservation unique, candidats complets CPU/GPU/externe, tests de cohérence PLAN/ADMIT/RUN, paramètres d'exécution exclus de l'identité scientifique quand la parité le permet.

Gate : un plan déclaré faisable est le même plan réellement admis ; aucun passage dans une seconde admission utilisant de vieilles dimensions. Les décisions de repli sont explicites et les profils utilisateur stricts restent distingués d'AUTO. Pas de relâchement des garde-fous existants sans remplacement prouvé.

### A2 — Préparation et scratch réellement bornés

Livrables : provider par fenêtres, cache borné sans obligation N×patch, sink de rejet borné, sorties de cellule sur disque si nécessaire, jauge/décodage compatibles avec le petit budget.

Gate : une cellule à N≥20k progresse sans matérialiser le cache complet ; pic scratch sous quota, données identiques, gestion des verrous/handles Windows et arrêt propre. Mesurer l'amplification I/O ; ne pas prétendre que le seul stack est désormais borné.

### A3 — Réduction canonique CPU/GPU par micro-blocs

Livrables : découpage 2D jusqu'à des blocs inférieurs à une ligne, conservation des N contributeurs, réutilisation prudente des buffers, extension GPU du chemin externe, fallback transactionnel.

Gate : mean/none, kappa_sigma, winsorized_sigma_clip et toute autre combinaison actuellement exposée conservent leur contrat ; tests NaN, poids nuls, faible N, outliers, mono/RGB, bords et halos. OOM injecté → un seul résultat validé, aucun pixel doublé/perdu. Mesure sur GPU physique requise pour accepter le chemin GPU.

Ne pas fixer 24/32/40 B comme objectif scientifique : les allocations sont optimisées, puis le modèle est recalibré. Pas de float16, ni d'abaissement float64→float32 par défaut.

**Jalon intermédiaire A1–A3 : une vraie cellule profonde calculée avec faible RAM et GPU exploitable. Ce n'est PAS encore une qualification produit complète.**

### A4 — Assemblage et livrables bornés

Livrables : brancher l'allocation scratch, finalisation par bandes, écriture FITS transactionnelle et hachage streaming, support/coverage et preview bornés ; traiter explicitement les finitions activées.

Gate : canevas dont les plans à 121 B/pixel dépassent la RAM disponible, pourtant FITS final valide et conforme au témoin. Aucun readback/cast/copie plein canevas à la dernière étape. Une première preuve science-only est un sous-gate, jamais la clôture si la configuration demandée comporte une finition.

### A5 — Adaptation dynamique, reprise et débit

Livrables : réduction/croissance prudente des crédits, fenêtres, pinned et prefetch ; orchestration pression/OOM ; reprise avec moins de RAM/VRAM ou sans GPU ; traces lisibles et ETA honnête.

Gate : budget qui baisse puis remonte, échec/restart au milieu d'une unité, GPU indisponible, scratch presque plein, stop/reprise : tous les résultats validés sont conservés sans omission/duplication. Toute nouvelle tentative est bornée, motifs et backend réellement exécuté sont visibles.

### A6 — Qualification produit et clôture

Livrables :

- M33 réel complet avec ses paramètres scientifiques inchangés, sortie scientifique inspectée ;
- Caldwell grand canevas et halos demandés ; M106/M16 témoins de non-régression ;
- tests sous budgets 1/2/4/8/16/32 Gio (tous les points sur synthétiques, sous-ensemble représentatif sur vrais corpus) ;
- N=2k/20k/50k et témoin 100k spatialement borné ; séparer profondeur et surface ;
- CPU-only et CUDA réel ; Windows/spawn/verrous et Linux ;
- rapport temps, pics RAM/VRAM, I/O, scratch, débit, parité, raisons de fallback.

Gate : M33 termine de bout en bout ; la preuve petit budget n'est pas un simple dry-plan. Qualification de 1 Gio limitée aux workloads réellement exécutés, sans revendiquer tous les corpus sur toute machine.

## 6. Critères transversaux

- Tous les contributeurs valides restent présents ; aucune hiérarchie de moyennes de sous-stacks.
- Aucun changement automatique halo, grille scientifique gelée, WCS, référence, poids, normalisation ou rejet.
- Le chemin CPU déterministe garde l'égalité exacte lorsque seul le stockage/découpage spatial change ; comparer les pixels/diagnostics, pas les fichiers incluant des timestamps.
- CPU↔GPU : utiliser le contrat numérique existant et le quantifier avant modification. Aucune nouvelle tolérance inventée après un échec ; si nécessaire, décision scientifique explicite.
- Aucun changement d'ordre de réduction N/dtype non qualifié.
- Pic RAM de l'arbre de processus et pic VRAM observés, pas uniquement sommes de nbytes ; relever page cache, pool et pinned séparément. Limites sous processus/OS contrôlés, pas seulement mocks de mémoire libre.
- Enveloppe modèle ≥ pic pertinent observé avec marge explicitée sur les témoins ; modèle non présenté comme garantie universelle.
- Le scheduler réduit les lots si le budget baisse et exploite une hausse stable lorsque cela améliore le débit.
- Tests ciblés courts à chaque lot ; campagne intégrée aux jalons A3/A4/A6, exhaustive au gate final/nightly pertinent, pas à chaque correction.
- Aucun succès sur la seule acceptation de layout/admission : preuve jusqu'à publication des sorties.
- Aucune implémentation concurrente sur le vieux checkout documentaire ; worktree isolé basé sur beta vérifiée au lancement.

## 7. Réutilisation et périmètre exclu

### À conserver

M0–M6, spool et commits de cellules, primitives scientifiques, observations GPU existantes, contrôles d'intégrité et reprise, instrumentation disponible.

Legacy offre déjà des bandes adaptatives, memmaps et chemins CPU streaming ; s'en inspirer d'abord dans le même environnement. Certains chemins legacy comptent du swap et possèdent des politiques de chunks/incrémental : ne pas les importer sans audit scientifique ni assimiler leur succès utilisateur à une preuve sur le même corpus et les mêmes paramètres.

### Référence Astro-Stack-and-Stitch-Studio

À reprendre : dimensionnement des bandes, staging pinned sous budget, prefetch, lecture memmap.

À ne pas copier aveuglément : 24 B universels, clamp à une ligne, canevas pleine taille par frame, dtype float32, algorithmes de rejet différents, allocations de sortie pleine image.

Les sources sont déjà disponibles : aucun nouvel envoi nécessaire pour cette étude. Une reprise de code MIT est possible sous conservation de l'avis de copyright/licence ; vérifier la compatibilité du fichier ciblé, les dépendances et la redistribution avant intégration. Pas de nouvelle dépendance majeure proposée ici.

### Hors mission

Refonte UI du filtre, recherche Caldwell photométrique, changement de rejet, FP16, downsampling, exclusion de frames, modification de halo pour faire tenir, sous-stacks scientifiques hiérarchiques, multi-GPU/distribué, promesse N infini, publication/version/release.

## 8. Ordre, ampleur et décision suivante

Ordre proposé : **A0 → A1 → A2 → A3 → A4 → A5 → A6**. Les enseignements de A0 peuvent déplacer une extraction technique, mais pas supprimer la preuve bout-en-bout.

La recommandation précédente « mesures + baisse du workspace suffiront probablement » est retirée : elle ne couvre ni préparation, ni disque, ni canevas final.

C'est une **mission moteur structurante**, pas un réglage. Ordre de grandeur de planification : plusieurs semaines ; conserver une enveloppe indicative **6–10 semaines d'ingénierie et qualification**, avec forte incertitude avant A0. Ce n'est ni un engagement calendaire ni une durée de calcul agent. Pas de promesse M33 débloqué en 1–2 semaines.

Premier engagement utile : **A0 court**, puis décision sur les contrats et le chiffrage révisé. Son résultat doit déjà permettre d'expliquer où passent les octets du dernier M33 et quel chemin minimal complet peut respecter le budget.

Recommandation de priorité : traiter ce moteur avant l'interface ZeGrid planifiée le 5 novembre. L'interface doit représenter une capacité réelle, pas masquer des refus successifs. Cette priorité reste proposée tant que Tristan n'a pas lancé l'implémentation.

## 9. Index de preuves sources inspectées

Baseline ZeGrid 5150040 :

- `src/zemosaic/core/zegrid/gpu.py` : workspace 96 B, halo de taper et marges GPU.
- `src/zemosaic/core/zegrid/external_stream.py` : 31 B partiels, minimum une ligne, CPU-only, sorties patch.
- `src/zemosaic/core/zegrid/bounded_assembly.py` : plans 121 B/pixel, limites mmap, ScratchCanvasAllocator.
- `src/zemosaic/zemosaic_zegrid_mode.py` : plan initial, admissions ACTIVE, repli M6, contraintes de reprise.
- `src/zemosaic/core/zegrid/dry_plan.py` : portée du dry-plan historique à réconcilier avec le vrai dispatcher externe ; ses commentaires anciens ne prouvent pas l'absence du wiring produit.
- `src/zemosaic/zemosaic_align_stack_gpu.py` : rows_per_chunk, budget VRAM et CPU WSC streaming.
- `src/zemosaic/zemosaic_align_stack.py` : memmap, chunks, choix mémoire legacy.
- `docs/refactor/ZEGRID_AUTO_EXTERNAL_STREAMING_PLAN_20261107.md` : acquis/invariants de la mission précédente, sans confondre plan et qualification réelle.

Référence externe c06ba83 : `gpustacker/backend.py`, `gpustacker/stacking.py`, `LICENSE`.

Documentation uniquement : aucun code, config, runtime Windows ou données scientifiques modifiés ; aucun commit/push ni mission worker lancés.


## 10. Contrat de conduite A0 → A6 — version renforcée du 2026-11-07

Cette section précise les gates des §5–6 ; elle ne relâche aucun invariant. Les règles
canoniques AGENTS.md restent la source de vérité du transport A2A et des actions Git.

### 10.1 Autorisation présente et futur lancement

**Instruction actuelle de Tristan : renforcer le document, puis attendre son lancement.**
L'état reste `READY_NOT_STARTED`. Ni A0, ni Coco, ni Nono ne sont lancés par cette rédaction.

Une instruction ultérieure lançant **cette mission complète** autorisera, dans le
périmètre de ce plan, la progression séquentielle A0 → A6, les implémentations locales,
les tests et les runs de qualification nécessaires dans des espaces isolés.
Consigner alors la portée et la référence exacte de cette instruction dans la fiche
d'état ; une autorisation limitée à A0 ne vaut pas autorisation A1–A6.

Après lancement complet, ne pas solliciter un nouvel accord entre chaque lot :
Junior décide la progression sur preuves. A0 est un gate technique interne,
pas un arrêt humain automatique. Un chiffrage révisé seul ne déclenche pas de gate ;
une modification significative d'objectif ou d'architecture hors plan, oui.

Ne sont PAS implicitement autorisés : publication Git, merge beta/main, tag/release,
installation/remplacement du runtime utilisateur, écrasement d'anciens résultats,
migration destructive, changement scientifique ou nouvelle dépendance majeure.
Les commits locaux nécessitent une autorisation explicite de mission ou de Tristan ;
ne pas les déduire du simple renforcement documentaire. À défaut, conserver le travail
dans le worktree isolé avec diff et preuves durables. Une sortie FITS de test dans un
dossier neuf n'est pas une publication Git ni un déploiement.

### 10.2 Décisions déléguées et limites du périmètre

**Junior peut décider**, après lancement et preuves appropriées :

- extractions locales, noms des modules internes, format versionné du scratch ;
- dimensions des blocs, quotas, réservations, ordre des optimisations et instrumentation ;
- conservation ou réutilisation d'un mécanisme existant plutôt que réécriture ;
- découpage d'un lot en tâches bornées, sans élargir ses critères ;
- contrôle automatique A0 puis progression, tant que les gates sont satisfaits.

**Zones de modification prévues** :

| Domaine | Zones autorisées dans le cadre de la mission |
| --- | --- |
| Ressources/exécution | `core/zegrid/{resource_envelope,ram_contract,scheduler,parallel,execution,runtime_control,resource_recovery,gpu_memory,gpu,external_stream,storage_profile,dry_plan,auto_layout}.py`, modules voisins strictement nécessaires |
| Préparation/données | `core/zegrid/{file_provider,calibration,photometric,rejection_sink}.py`, seams de préparation et de jauge identifiés en A0 |
| Assemblage/sortie | `core/zegrid/{bounded_assembly,assembly,mosaic,cell_feather,final_mosaic_finishing,run_publication,run_artifacts}.py` |
| Durabilité/intégration | `core/zegrid/{run_store,run_resume,run_registry}.py`, câblage ciblé dans `zemosaic_zegrid_mode.py` et CLI/config existantes |
| Science canonique | extraction/réutilisation de primitives dans `core/canonical_*.py` seulement sous témoins ; aucune modification des formules ou du contrat numérique |
| Preuves | tests ciblés, outils de qualification, documentation et état de mission |

Cette table est une carte fonctionnelle, pas une permission de modifier tous les fichiers.
Chaque mission worker nomme ses fichiers attendus, non-objectifs et tests.
Une dépendance technique locale supplémentaire peut être ajoutée avec justification
dans le brief ; une extension produit ou scientifique exige décision humaine.

**Lecture seule par défaut** pour legacy et le dépôt de référence : extraire une idée
ou réutiliser une primitive qualifiée n'autorise pas une refonte de legacy.
Pas de réorganisation générale, nettoyage opportuniste, refonte GUI ni import
d'un second framework GPU. Tout changement de schéma durable doit prévoir lecture
compatible ou refus explicite non destructif des anciennes reprises.

### 10.3 Cycle d'exécution et identité des missions

- Programme stable : `ZM-ZEGRID-ADAPTIVE-RESOURCES-20261107`.
- Une tâche indépendante : suffixe `-A0-T01`, `-A1-T01`, etc. Le même identifiant
  de tâche reste inchangé pendant implémentation, revue et toutes ses corrections.
- Une seule implémentation Coco active à la fois. Junior établit le brief, vérifie
  la baseline, délègue selon le protocole A2A canonique, inspecte le vrai diff,
  exécute les vérifications importantes, puis fait revoir les changements matériels.
- Rapports workers sous `/home/tristan/.openclaw/workspace/.a2a-reports/`,
  fichiers distincts par rôle et itération ; callback vers la source exacte.
- Pas de polling, pas de reset worker en cours de mission, pas de changement
  silencieux de transport en cas de panne. Les règles AGENTS.md s'appliquent.
- Maximum **3 itérations correctives Coco par tâche** ; ne pas renommer une tâche
  inchangée pour remettre le compteur à zéro.
- Nono rend un avis ; Junior seul prononce l'acceptation technique.
- Mise à jour de l'état à chaque délégation, callback accepté, verdict et blocage.

Machine d'état :

`READY_NOT_STARTED → A0_RUNNING → A0_ACCEPTED → A1_RUNNING → … → A6_ACCEPTED`

À chaque lot : `RUNNING → REVIEW → ACCEPTED` ou `REWORK / BLOCKED / HUMAN_GATE`.
Une preuve manquante est `NOT_RUN` ou `BLOCKED`, jamais `PASS`.
Aucun lot dépendant ne commence avant l'acceptation du précédent. Préparer
documentation ou tests indépendants ne permet pas de contourner ce verrou.

## 11. Gate A0 obligatoire : paquet de référence figé avant A1

Créer dans le worktree de mission :
`docs/qualification/zegrid-adaptive-resources/` avec un index des preuves.
Les gros artefacts peuvent être hors Git ; l'index donne chemins absolus,
tailles, empreintes et environnement. Pas de données privées publiées.

Les six livrables suivants sont obligatoires ; aucun champ critique ne reste
« à définir » à l'acceptation A0 :

1. **BASELINE** : SHA, branche, statut Git, versions Python/NumPy/CuPy/CUDA,
   plateforme, CPU/GPU, build Windows réellement exécuté, détection des travaux
   concurrents. Au lancement, vérifier les refs ; si beta a avancé, analyser le
   diff et enregistrer la baseline choisie sans déplacer les branches existantes.
2. **CORPORA** : inventaires ordonnés et empreintes, config figée, filtres, WCS,
   canvas/grille/halos, N total et N par cellule, sources et dossiers de sortie
   neufs. Réconcilier 22 414/21 818 ; aucune exclusion pour faciliter un test.
   Un mélange de filtres non voulu ou une source invalide ne se corrige pas
   silencieusement.
3. **SCIENCE_ORACLE** : sorties témoins, méthodes/combinations exposées, plans
   canoniques, compteurs et règles exactes de comparaison. CPU déterministe :
   égalité bitwise des valeurs finies et masques de validité équivalents ;
   si la baseline n'est pas déterministe, caractériser AVANT changement.
   CPU↔GPU : citer les tolérances existantes, leurs valeurs numériques et les
   tests sources. Contrat absent ou incompatible : gate explicite, pas de seuil
   inventé après observation des résultats.
4. **RESOURCE_CONTRACT** : métriques, budget, unités, réserves et capteur (§12).
   Fournir une trace courte de chaque phase atteignable. Si M33 s'arrête avant
   stack, utiliser un témoin faisable pour les phases suivantes et distinguer
   « mesuré sur témoin », « modèle », « extrapolé à M33 ».
5. **QUALIFICATION_MATRIX** : commandes exécutables, identités machines/corpus,
   quotas RAM/VRAM/scratch, timeouts/watchdogs et seuils figés (§13). Les commandes
   ne peuvent pas utiliser des chemins fictifs. Déclarer les moyens indisponibles
   comme blocages, sans prétendre les avoir testés.
6. **A0_VERDICT** : défaut/limite reproduit, inventaire des allocations dominantes,
   delta modèle/observations, proposition de chemin minimal complet, découpage
   A1 et risques restants. Revue indépendante des conclusions matérielles.

A0 n'a pas à lancer un M33 complet défaillant. Probes courts et ciblés, au plus
300 s par probe ordinaire ; une validation connue comme longue peut aller à
1200 s selon AGENTS.md. Un témoin produit plus long sera supervisé comme run de
qualification avec progression/checkpoints, pas maintenu dans un exec bloquant
indéfini. Avant toute nouvelle campagne : motif, coût prévu, quota et condition
d'arrêt ; pas de relance complète automatique après timeout.

**A0_ACCEPTED exige les six livrables vérifiés.** Si les contrats ne peuvent pas
respecter l'objectif, conserver l'évidence et escalader ; ne pas retirer
« 1 Gio », GPU, M33 complet ou la parité pour rendre le plan acceptable.

## 12. Contrat de mesure des ressources

### RAM : sens précis de 1–32 Gio

- 1 Gio = 2^30 octets. Points à qualifier : 1, 2, 4, 8, 16, 32 Gio.
- Budget moteur = arbre des processus de calcul, imports et bibliothèques,
  workers, buffers, pinned, pages mappées résidentes et allocations d'assemblage.
  Mesurer dès le démarrage du moteur, pas seulement après un plateau.
- L'OS et une GUI réellement séparée sont hors budget moteur, mais leur mémoire
  est relevée et une réserve système est maintenue. Si GUI et moteur partagent
  un processus, ne pas soustraire artificiellement la GUI : compter ce processus
  ou qualifier le budget sur un runner moteur séparé puis mesurer l'E2E GUI.
- Aucun « socle gratuit » ajouté au plafond après un dépassement ; le socle est
  mesuré et inclus. Swap/pagefile non comptés comme RAM disponible.
- Publier au minimum : pic working set/RSS de l'arbre, mémoire privée/commit
  selon plateforme, mémoire système disponible et pression. Sous Linux, inclure
  la mesure du groupe/cgroup et la mémoire file-backed lorsqu'applicable ;
  éviter le double comptage des pages partagées dans la métrique d'acceptation.
- A0 fixe par plateforme le capteur principal, sa fréquence, le traitement du
  partage/page cache et la limite OS utilisée. Cible d'échantillonnage ≤100 ms ;
  compléter par high-water OS/allocateur lorsque possible pour ne pas manquer
  les transitoires.
- Critère : pic de la métrique principale ≤ budget moteur déclaré, pas de kill
  mémoire, pas de dépendance au swap pour réussir. Pas de tolérance de dépassement
  ajoutée a posteriori ; réserve interne au budget.
- Un budget simulé vérifie le scheduler, PAS la preuve mémoire. Au moins un E2E
  synthétique à 1 Gio réellement contraint doit finir ; sinon A6 reste bloqué.
  Un point 32 Gio non matériellement disponible reste non qualifié, non inventé.

### VRAM, disque et temps

- RAM et VRAM ne s'additionnent pas comme une mémoire interchangeable.
- Distinguer VRAM libre système, allocation vivante, pool réservé et contexte.
  Plafonds de qualification : 1/2/4 Gio et budget nominal sûr de la carte réelle,
  si techniquement disponibles ; enregistrer le plafond effectivement appliqué.
  Simulations réservées aux transitions/tests de panne, pas à la preuve CUDA.
- Pic scratch inclut cache, spool, résultats provisoires et marge de transaction.
  Distinguer résultats finaux obligatoires et scratch recyclable dans le rapport.
  Le disque minimal dépend du corpus ; A0 chiffre et fige les quotas avant runs.
- Saturation GPU n'est pas un gate ; prouver l'exécution CUDA réelle, son résultat,
  les transferts et le temps. Comparer CPU/GPU à science et entrée identiques.
  A0 fixe un seuil de régression sur témoins courts avant optimisation ; aucune
  promesse d'accélération universelle et aucun raccourci scientifique pour gagner.
- Les watchdogs détectent l'absence de progrès, pas la lenteur légitime. A0 fixe
  la durée sans progrès et le nombre maximal de tentatives à partir de traces ;
  chaque reprise réduit strictement les ressources ou change de backend une
  fois qualifié. Au minimum : interdiction de réessayer indéfiniment le même
  tuple backend/bloc/crédits sous le même budget.

## 13. Matrice minimale d'acceptation, non réductible après échec

A0 choisit et enregistre les dimensions exactes, seeds, chemins et commandes.
Il ne peut retirer une ligne ci-dessous pour fermer un lot.

| ID | Lot | Cas obligatoire | Condition de succès |
| --- | --- | --- | --- |
| Q00 | A0 | Dernier refus M33 + témoin faisable par phase | cause liée au code/build et tableau modèle/pics ; faits récents séparés des historiques |
| Q10 | A1 | Même demande via PLAN/ADMIT/RUN ; budgets 1/2/4/8/16/32 ; ressource abaissée entre plan/admission | aucun candidat périmé, comptabilité commune, refus physique ou adaptation explicitement justifiés |
| Q20 | A2 | Préparation d'une fenêtre avec N≥20k ; quota scratch inférieur au cache complet estimé | progression, égalité préparation avec oracle, quota respecté, handles bornés |
| Q21 | A2 | Arrêt et erreur I/O pendant préparation, Windows/spawn | aucune source modifiée, aucun résultat valide écrasé, nettoyage limité aux artefacts possédés |
| Q30 | A3 | N=2k/20k/50k, plus 100k sur surface bornée ; rejets/combine exposés, NaN/outliers/bords | tout N consommé, oracle respecté, blocs plus étroits qu'une ligne exercés |
| Q31 | A3 | CUDA physique + VRAM contrainte, puis OOM injecté | kernel réellement exécuté, parité, repli/rejeu unique sans perte ni doublon |
| Q32 | A3 | Une cellule réelle profonde M33, paramètres gelés | toutes ses contributions traitées ; ne pas confondre ce témoin avec le run complet |
| Q40 | A4 | Canevas dont 121×H×W > budget disponible | science/support/coverage et FITS complets dans le budget, aucune copie finale plein canevas |
| Q41 | A4 | Finitions/preview demandées, disque presque plein à la publication | aucune désactivation silencieuse, sortie atomique ou état reprenable honnête |
| Q50 | A5 | Budget 4→1→4 Gio pendant run instrumenté, puis restart avec moins de ressources/sans GPU | réduction puis croissance exploitable, mêmes résultats, identités scientifiques préservées |
| Q51 | A5 | Interruption milieu d'unité, cellules déjà validées, reprise ; corruption témoin isolée | zéro double comptage, cellules validées réutilisées, corruption détectée et non acceptée |
| Q60 | A6 | E2E synthétiques à chacun des 6 budgets RAM | résultat final vérifié et mesures réelles, pas uniquement dry-plan ou mocks |
| Q61 | A6 | M33 réel complet Windows avec GPU disponible | inventaire/config figés, sorties finales vérifiées, GPU réellement utilisé quand admissible |
| Q62 | A6 | Caldwell grand canevas, halo demandé (témoin prévu 128), machine Linux contrainte | run complet sans diminuer le halo pour réussir ; mesures et artefacts |
| Q63 | A6 | M106/M16, CPU/GPU et ancien/nouveau chemin | non-régression scientifique, temps/pics publiés, critères figés en A0 |
| Q64 | A6 | Chemin GUI→backend, paramètres mémoire/reprise | propagation effective et run synthétique via ce chemin, pas seulement persistance de settings |

Les synthétiques profonds doivent comporter des échantillons variés et des cas
adversariaux déterministes, pas uniquement 20 000 références au même petit tableau.
Un générateur borné peut éviter de stocker tout le corpus ; il doit réellement
alimenter les N contributions et exercer I/O/spool dans les témoins dédiés.

Pour un grand N impossible à traiter par l'ancien moteur complet, comparer des
positions/régions réduites avec l'oracle canonique sur les N valeurs et vérifier
la totalité du produit par intégrité, compteurs et invariants. Ne pas présenter
cela comme une comparaison bitwise intégrale avec un oracle non exécuté.

A6 exige tous les gates requis au statut PASS, ou une modification explicite
du périmètre par Tristan documentée comme telle. Un matériel absent, corpus
indisponible ou test non exécuté interdit la qualification correspondante et la
clôture complète. Le travail déjà vérifié reste accepté sans prétendre A6 terminé.

## 14. Journal de preuves, recette de reprise et sortie de mission

### Livrables de chaque tâche

Le rapport et l'index de preuves contiennent :

- identifiant programme/tâche, lot, phase/itération ;
- objectif, fichiers réellement changés, diff expliqué, changements inattendus ;
- branche, HEAD et état Git ; SHA si commit autorisé ;
- commandes/tests exacts, environnement, codes de sortie, passed/failed/skipped ;
- artefacts avec chemins/empreintes, budgets et observations ;
- verdict Coco puis revue Nono, vérifications propres à Junior ;
- défauts ouverts, compte de corrections et prochaine action unique.

Aucun `ACCEPT` basé seulement sur « tests verts » dans un rapport worker.
Conserver les échecs initiaux ; ne pas réécrire l'historique pour masquer un
changement de critère. Les écarts documentaires mineurs sont notés, les erreurs
matérielles sont corrigées avant progression.

### Reprise après changement de modèle

Point d'entrée : [fiche d'état](ZEGRID_ADAPTIVE_RESOURCE_ENGINE_STATE_20261107.md).

1. Lire les instructions workspace, ce plan et la fiche d'état.
2. Lire l'instruction de lancement consignée ; si absente, rester NOT_STARTED.
3. Résoudre le worktree actif indiqué dans la fiche, vérifier branche/HEAD/status
   et l'index des preuves ; ne pas se baser sur le checkout documentaire ancien.
4. Vérifier le dernier verdict Junior et les tâches en vol. Ne jamais reset ou
   redéléguer sur simple absence de contexte. Suivre la récupération A2A autorisée
   seulement si nécessaire ; le changement de modèle n'autorise pas le polling.
5. Continuer la prochaine action unique depuis le dernier gate réellement accepté,
   sans relancer les lots clos ni inventer des tests manquants.
6. Si preuve manquante, état contradictoire ou worker possiblement actif : préserver,
   clarifier le blocage et ne pas fabriquer une continuité.

Au lancement, copier le plan renforcé et la fiche dans le nouveau worktree, puis
désigner ce dernier comme source active unique dans la fiche d'entrée documentaire.
Les copies historiques restent identifiées comme telles ; pas de double tableau
de bord actif divergent.

### Arrêts et résultat final

Arrêt motivé si invariant scientifique violé, architecture hors plan nécessaire,
trois corrections infructueuses, état worker ambigu, outil requis indisponible,
ou preuve matérielle indispensable absente. Fournir l'état préservé, la preuve,
les tentatives et une seule décision/action suivante. Ne pas contourner l'arrêt
en abaissant un gate ou en substituant un transport.

Clôture **A6_ACCEPTED** seulement après revue finale et vérification Junior :
matrice complète, M33 final, cas grand canevas/petit budget, CUDA physique,
reprise et propagation GUI, diff entièrement compris, limites documentées.
La validation visuelle humaine, lorsqu'elle reste nécessaire, est un gate
distinct annoncé et non un succès scientifique inventé.

La clôture technique ne publie rien et ne remplace pas le runtime Windows.
Remettre à Tristan le bilan concis, les chemins des artefacts, les pics/temps,
les limites qualifiées et l'éventuelle décision de publication séparée.
