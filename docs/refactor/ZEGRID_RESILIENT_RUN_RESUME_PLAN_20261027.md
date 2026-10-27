# ZeGrid — exécution résiliente et reprise durable
Date : 2026-10-27
Mission proposée : ZM-ZEGRID-RESILIENT-RUN-RESUME-20261027
Statut : PLAN À VALIDER — révision critique 2 ; aucune implémentation, aucun commit ni push.
Propriétaire produit : Tristan. Architecture/acceptation : Junior. Implémentation : Coco. Revue : Nono.

### Révision du 2026-10-27 (relecture critique)

Cette relecture a comblé des angles morts : **durabilité de la jauge** (longue
phase pré-empilement non reprise), **libération réelle des ressources pendant
l'attente**, **contrat d'annulation/sortie propre**, **entrée de reprise et clé
de config dédiée**, **coût disque des checkpoints de patchs**, **tolérance à la
mise en veille Windows**, **provenance de reprise dans le manifeste**, et
**clarification du sens unique dense→bounded**. Détails en sections 3, 5.3, 5.4,
6.1, 6.5, 8, 9 et 11.

### Deuxième revue — priorités de validation

Cette révision corrige aussi des exigences irréalistes de la première relecture
(arrêt toujours capable de committer, flush avant une veille non anticipée,
VRAM supposée sans coût hôte), et transforme les points critiques en tests bloquants.
Les nouveaux contrats 5.6–5.8, 6.6–6.7 et les gates 8.1–8.2 font partie de la
mission, pas d'une dette facultative. Tableau des constats en section 12.

## 1. Objectif produit

> Utiliser la RAM pour accélérer quand elle est disponible, sans perdre les cellules
> terminées lorsqu'une autre activité du PC réduit la mémoire disponible.

Le moteur doit :
1. Continuer entre cellules avec une stratégie moins gourmande quand elle est admissible.
2. Attendre sans accumulation mémoire quand aucune stratégie sûre ne tient temporairement.
3. Conserver durablement chaque cellule validée.
4. Reprendre après annulation, fermeture ou arrêt brutal au dernier checkpoint valide.
5. Produire un résultat scientifiquement équivalent au run de référence.

Une pression RAM ne doit plus provoquer automatiquement :
« jeter toutes les cellules GPU → reprendre tout sur CPU → échouer à l'admission ».

La garantie est la conservation des checkpoints valides et l'absence d'allocation
délibérément hors budget, PAS l'immunité à toute panne matérielle, disque plein,
perte d'alimentation ou suppression des données. Sans mémoire minimale, attendre
ou s'arrêter proprement reste nécessaire.

## 2. Baseline et preuves

Base auditée : beta publiée 231b597867b4d251afac6b767000c8a37e2cf3b0.
Source inspectée : worktrees/zemosaic-i18n-debt, arbre propre lors de l'audit.
Le fichier moteur du runtime ZeAlfie Windows a le même SHA-256 :
8ccc629350b921764f1d291b76a0ce426fabc861509052857cc0ff731387c768.

Attention : le checkout principal local est encore à 48a0b65 et possède des
documents non suivis sans rapport. Ce plan y est ajouté comme documentation
uniquement. L'implémentation devra partir d'une beta fraîchement vérifiée,
dans un worktree isolé ; ne pas confondre cette copie locale avec la baseline.

### Incident témoin M31

- 4 171 entrées, grille 17×11, 187 unités, photométrie per_cell après décision auto.
- 81 unités consommées, dont des unités vides : ce n'est PAS une certification
  de 81 cellules scientifiques complètes.
- GPU tile32, prefetch0 ; arrêt après environ 10 h.
- Refus hôte GPU : 9 803 345 920 octets > budget 9 152 678 707.
- VRAM admissible ; puis refus CPU : 10 147 193 856 > 9 114 904 166.
- Pic RAM système observé 28,43/31,71 Gio ; pic processus Windows 9,42 Gio.
- Tristan confirme une utilisation simultanée du PC dans une autre session.
  L'ampleur exacte imputable à cette activité n'est pas mesurée.
- Le test N5000 réussi forçait bounded dès le départ : il ne couvre pas la
  transition runtime dense→bounded d'une mosaïque multicellule.

Preuves : rapport
`/home/tristan/.openclaw/workspace/.a2a-reports/ZM-M31-RUNTIME-ADMISSION-20261027.junior-diagnostic.md`.
Windows : `C:\Users\JarvisJunior\Desktop\zemosaic out\M31`.

### Existant réutilisable, limites à respecter

- CPU : `_run_r55_active_cpu_waves._bounded_run_recovery` possède DÉJÀ un
  secours dense→bounded entre vagues, conserve les vagues terminées et fige
  géométrie/science. Corrige la formulation trop générale du diagnostic initial :
  le manque concerne le secours GPU et sa réadmission CPU, pas tout le runtime.
- GPU : `_run_r55_active_gpu_units`, `_r55_runtime_tile_check/walk` puis
  `except _R55GpuFallback` dans `_run_single` abandonnent actuellement les
  résultats du lot avant le repli CPU.
- `_consume_stack_cell` retourne core, record et, en per_cell, les plans patch
  science/valid_mask/surviving avant corrections intercellules.
- `_assemble_per_cell_feathered` construit les CellPatch et résout les offsets
  puis le feathering. Le point de sauvegarde doit être AVANT ces transformations.
- Les checkpoints runtime actuels sont des contrôles/observations : ils ne
  constituent pas un journal de reprise durable des résultats scientifiques.
- Le resume Classic existe, mais ses signatures et son mode force ne doivent
  pas être copiés aveuglément dans ZeGrid.
- Fichiers principaux : zemosaic_zegrid_mode.py, core/zegrid/{parallel,
  runtime_control,ram_contract,scheduler,storage_profile,assembly,cell_feather,
  cell_offset}.py, zemosaic_worker.py, zemosaic_gui_qt.py.

## 3. Périmètre

### Inclus

- ZeGrid ACTIVE, CPU et GPU, dense/bounded et politique auto.
- Pression RAM aux frontières sûres, réessai borné de la seule unité incomplète.
- Conservation des résultats terminés, checkpoints disque, reprise inter-processus.
- Jauges global/per_cell, cellules vides, assemblage et finitions.
- Retour utilisateur : attente ressources, sauvegarde, reprise, erreurs explicites.
- Qualification Windows prioritaire, tests portables Linux.

### Exclus de la première version

- Reprise au milieu d'un bloc GPU ou d'une cellule partiellement calculée.
- Changement automatique de grille/halo ou suppression de contributeurs après
  démarrage pour faire tenir la mémoire.
- Fusion de plusieurs runs, reprise forcée de science incompatible.
- Optimisation générale du throughput, changement d'estimateur ou nouvelle
  calibration mémoire sans preuve indépendante.
- Refonte du resume Classic, exécution distribuée, gestion de la mise en veille
  par l'application (suspendre le travail avant veille).
- Exclu de gestion automatique, mais PAS de tolérance : les checkpoints doivent
  survivre à une veille/hibernation Windows, et au réveil un contexte GPU
  invalidé doit être recréé, jamais réutilisé (voir 5.4).
- Récupération garantie du M31 déjà échoué : ce run n'avait pas les nouveaux
  checkpoints et ses caches ont été nettoyés. Ne rien promettre rétroactivement.

### Hors périmètre explicite (complété à la relecture)

- Reprise au milieu de la jauge : la jauge est traitée comme UN artefact
  atomique et resumable (voir 6.1), pas comme un flux reprenable image par image.
  Si le run meurt pendant la jauge, celle-ci est perdue et relancée — un coût
  réel (≈3 h sur M31) à assumer et à afficher clairement.

## 4. Invariants non négociables

1. Même inventaire ordonné, WCS, canvas, grille, halo, appartenance aux cellules,
   critères d'exclusion, normalisation, pondération et rejet.
2. Chaque cellule conserve tous ses contributeurs légitimes ; pas de découpage
   des brutes en sous-stacks indépendants pour simuler un gros stack.
3. Les références, coefficients de jauge et identités photométriques sont
   figés/reconstitués exactement, jamais recalculés à partir des seules cellules restantes.
4. Une cellule COMPLETE n'est ni effacée ni recalculée à cause d'une pression
   sur une autre cellule, sauf checkpoint corrompu explicitement identifié.
5. Une cellule vide géométriquement est distincte d'une cellule perdue,
   interrompue ou invalide.
6. Budgets RAM/VRAM, garde disque et planchers scientifiques inchangés.
7. Les sorties finales ne sont déclarées complètes qu'après toutes les étapes ;
   aucune mosaïque partielle présentée comme un succès.
8. L'ordre canonique d'assemblage et de résolution des offsets reste déterministe.
9. Dense→bounded ne modifie pas la science. Un changement CPU/GPU nécessite une
   qualification explicite de compatibilité numérique, pas une supposition.
10. Erreur de science, observation invalide, corruption et manque de ressources
    sont des catégories distinctes ; pas de retry aveugle « sur toute Exception ».

## 5. Architecture cible

### 5.1 Coordinateur de run

Extraire progressivement de petites interfaces testables, sans refonte globale
du grand moteur. Modules candidats : run_state.py, checkpoint_store.py,
resource_recovery.py ; noms définitifs après inventaire M0.

Le coordinateur possède :
- l'identité scientifique immuable ;
- le journal des cellules ;
- la politique de ressources et ses générations de plans ;
- le store durable ;
- le contexte de jauge et l'état d'assemblage.

La cellule, et non le lot complet, devient la transaction de résultat. Les
caches de projection restent jetables ; les checkpoints de sortie ne le sont pas.

### 5.2 États explicites

Run :
NEW → PREPARING → RUNNING → ASSEMBLING → FINALIZING → COMPLETE.
États latéraux : WAITING_RESOURCES, STOPPED_RESUMABLE, FAILED_DIAGNOSTIC.
Un run interrompu brutalement est récupéré à partir du journal, pas d'un booléen
« succès » laissé dans un fichier.

Cellule :
PENDING → PREPARING → STACKING → COMMITTING → COMPLETE.
Autres états : EMPTY_VALIDATED, RETRY_PENDING, INVALID_CHECKPOINT.
Au redémarrage, PREPARING/STACKING/COMMITTING non validé redevient PENDING ;
COMPLETE n'est retenu qu'après vérification du commit et des artefacts.

Ne pas incrémenter la progression durable avant le commit. Afficher séparément
calculée, sauvegardée, réutilisée et vide, sans doubler le compteur après reprise.

### 5.3 Politique de pression mémoire

Au prochain point sûr :
1. Arrêter l'admission de nouvelles unités et le prefetch ; drainer/annuler
   proprement le travail spéculatif. Conserver les unités déjà validées.
2. Fermer les mappings/copies temporaires et libérer les crédits possédés.
3. Refaire une observation valide et appliquer les réductions admissibles :
   concurrence/prefetch/workers/blocs suivant les limites existantes.
4. Sous AUTO, si la pression RAM persiste, reconstruire les seules tâches
   restantes en bounded sur la même identité. Garder le GPU s'il reste admissible.
5. Si le minimum ne tient pas, entrer en WAITING_RESOURCES, sans garder une file
   de gros buffers ni réessayer en boucle serrée.
6. Réadmettre sur faits frais après retour durable de capacité.

Libérer réellement avant d'attendre (angle mort corrigé). Entrer en
WAITING_RESOURCES ne suffit pas : il faut drainer la file producteur, annuler le
travail spéculatif NON committé, rendre les crédits/staging détenus et fermer les
grandes allocations temporaires. Le propriétaire GPU dont l'unité est DÉJÀ en
cours a le choix : la laisser finir puis committer si admissible, ou l'annuler
et la réessayer plus tard ; il ne doit pas rester bloqué en détenant du staging
hôte pendant l'attente, sinon la mémoire ne se libère jamais. Ne pas confondre
VRAM dédiée, mémoire épinglée hôte, allocations du pilote et mémoire GPU partagée :
garder un contexte GPU peut aussi retenir de la RAM. Mesurer leur résidence,
libérer les pools/mappings sûrs et, si nécessaire, recycler un worker plutôt que
supposer qu'un appel au ramasse-miettes rendra les pages à Windows.

Une observation gouvernante reste cohérente pendant UN épisode de recovery ;
ne pas mélanger plusieurs échantillons pour fabriquer un budget favorable.
Un nouvel épisode après attente peut utiliser de nouvelles mesures, avec un
journal explicite. Un plan ne grossit pas automatiquement dans cette V1.

Le passage dense→bounded est À SENS UNIQUE (une seule bascule par run, déjà
garanti par `_activate_bounded_storage`). Le risque réel n'est pas un
« ping-pong dense/bounded » (impossible par construction) mais : (a) bounded
qui ne tient pas non plus → attente ou arrêt reprenable ; (b) un profil
explicitement dense qui ne doit jamais basculer en silence.

Politique proposée : sondes espacées (5 s, backoff jusqu'à 30 s), reprise après
deux observations admissibles espacées ; chiffres à figer par tests M3.
Attente sans échéance par défaut, visible et annulable ; une échéance optionnelle
doit aboutir à STOPPED_RESUMABLE, jamais à l'effacement du travail.
Ne pas conserver indéfiniment un ancien minimum de RAM après retour de capacité.

Profil explicitement dense : pas de substitution silencieuse ; attendre ou
s'arrêter de manière reprenable, proposer AUTO dans l'interface.
Profil explicitement bounded : rester bounded.
Les mêmes règles s'appliquent aux workers CPU, y compris les vagues partiellement terminées.

### 5.4 OOM et backend

- Pression détectée avant lancement : aucune allocation lourde de l'unité ;
  le coordinateur et le diagnostic gardent une petite enveloppe de contrôle budgétée.
- OOM récupérable pendant une cellule : ne pas committer son résultat ; libérer
  ses ressources et permettre un nombre borné de tentatives réduites.
- Contexte CUDA perdu ou worker tué : ne pas continuer dans un contexte compromis.
  Recréer un worker/processus si le protocole le permet, sinon arrêt reprenable.
- CPU/GPU mixte : autoriser uniquement après qualification du contrat numérique
  et publication du backend par cellule. Sinon conserver les checkpoints GPU
  et attendre un GPU admissible ; ne pas les jeter automatiquement.
- Les erreurs non récupérables restent visibles, avec checkpoints préservés.

Annulation et sortie propre (angle mort corrigé). Il faut distinguer :
- arrêt propre demandé (GUI/CLI) : cesser les nouveaux lancements, accuser
  réception immédiatement, finir une transaction COMMITTING seulement si elle
  reste sûre et accessible, puis écrire STOPPED_RESUMABLE au mieux ;
- arrêt forcé : interruption explicite distincte, possibilité de perdre la seule
  tentative non committée. Ne pas faire attendre la fin d'un stack de 40 minutes
  pour accepter un arrêt ; un appel natif non interruptible doit être signalé ;
- si disque/OOM empêche même le dernier marqueur, les commits précédents restent
  l'autorité : ne jamais conditionner leur reprise à STOPPED_RESUMABLE ;
- panne/erreur : préserver les commits déjà validés, marquer l'unité en cours
  comme incomplète, et produire un diagnostic sans faux COMPLETE ;
- arrêt brutal (coupure, kill -9) : l'état est reconstruit depuis le journal et
  les commits (6.4), pas depuis un drapeau « succès ».

Veille/hibernation Windows : chaque commit possède déjà sa propre garantie
flush/sync (6.2), sans dépendre d'un préavis de veille. On ne peut pas promettre
de finir une écriture en cours avant une suspension imprévisible. Au réveil,
revalider ressources et contexte CUDA avant nouvel accès ; le recréer seulement
s'il est invalide ou indéterminé. Les délais d'attente et verrous ne doivent pas
prendre une suspension ou un changement d'heure pour un décès du propriétaire.
La détection relève de M6, la récupération du store/backend de M2/M4.

### 5.5 Modèle mémoire

Séparer RAM résidente, nouvelles allocations, sorties conservées, staging,
VRAM et disque. Ne jamais ajouter tout le RSS à available pour « faire passer ».
Auditer la réservation pire-cellule figée et les allocations déjà résidentes
au point d'observation ; toute modification nécessite un témoin reproductible.

Première livraison : conserver un modèle conservateur cohérent, sans attendre
une optimisation par cellule. La reprise doit fonctionner même si la borne
reste pessimiste. Le store ne doit pas devenir un nouveau cumul RAM de résultats.

### 5.6 Contrôle de bout en bout et propriété des résultats

Constats source (baseline 231b597) :
- GUI WorkerController.stop(graceful=True) envoie SIGTERM puis terminate après
  quatre secondes ; « Stop and Save » utilise ce chemin.
- Le worker transforme SIGTERM en KeyboardInterrupt ; le pipeline lève/annule.
- BoundedPipeline._consume_loop conserve les gros résultats dans _results ;
  les vagues CPU les accumulent dans results_local.

Conséquences obligatoires :
1. Introduire un canal coopératif dédié (événement/commande + accusé d'état),
   indépendant des logs et compatible spawn Windows. La GUI reste à l'écoute
   jusqu'à l'accusé STOPPED/COMPLETE ; pas de terminate automatique à quatre
   secondes pour un arrêt propre. Préserver le comportement Classic séparément.
2. Le callback de cellule réussie remet le résultat au store avant sa libération.
   Le pipeline/coordinateur ne garde ensuite qu'un reçu/descripteur léger.
   Éviter les doublons cachés dans futures, queues IPC, listes, tracebacks,
   closures et views NumPy : une vue core peut retenir le patch entier.
3. Un seul propriétaire des buffers et du commit, un release idempotent ;
   crédits rendus après libération effective, pas après simple changement d'état.
4. Une vague CPU peut finir hors ordre : committer chaque succès indépendant,
   sans attendre le retour réussi de toute la vague.
5. Le lancement, le retour après pause et le redémarrage passent tous par le
   même contrat d'état ; aucune restauration implicite des globals du processus.

### 5.7 Générations, drainage et attente bornée en ressources

Chaque tâche/payload porte run_id, cell_id, attempt_id et plan_generation.
Lors d'une transition, empêcher les nouveaux envois et attendre/neutraliser
les anciens producteurs avant activation du nouveau profil. Une closure
bounded_storage figée ou un _STACK_* obsolète ne doit pas servir le nouveau plan.
Un résultat tardif de génération annulée n'est pas publié par accident ;
les résultats déjà committés restent valables et leurs identités immuables.

Ne jamais attendre un crédit en conservant celui dont l'autre thread a besoin
pour vider la file. Tester arrêt/recovery lorsque producteur bloqué sur crédit,
file pleine, consumer en COMMITTING et worker en retour IPC.

Attente ≠ worker mort ni run terminé. Le heartbeat de contrôle reste léger et
indépendant du débit de calcul ; surveiller séparément liveness et progression.
Pas de watchdog qui tue un stack long mais vivant. Les répétitions d'une même
panne au même plan ne relancent pas indéfiniment la même cellule : retries bornés,
puis attente d'un changement de capacité réel ou diagnostic reprenable.
Si le processus conserve un working set excessif après cleanup, recyclage sûr
depuis checkpoints plutôt qu'attente en gardant la pression.

### 5.8 Toutes les phases doivent rester admissibles

Le contrat couvre découverte/hachage, géométrie, jauge, cache_build, stack,
écriture des checkpoints, validation de reprise, offsets, assemblage,
finitions et publication. M0 établit pour chacune : propriétaire, borne RAM,
point d'arrêt, stratégie de réduction et preuve de libération.

La jauge actuelle lance des pmap counts puis pairs : un contrôle seulement
entre cellules ne la protège pas. Réduire/mettre en pause aux frontières de
tâches sans changer l'ordre canonique des résultats ni la référence choisie.
La reprise durable au milieu de la jauge reste hors V1 ; ne pas confondre cette
limite avec l'abandon volontaire du run au premier signal de pression.

Allouer une réserve mesurée pour contrôle/commit/cleanup. Sérialiser et hasher
par blocs, sans tableau contigu géant ni compression pleine taille. Le commit
peut lui aussi OOM ou manquer de disque : préserver les précédents commits,
abandonner uniquement la tentative non publiée si nécessaire.

L'assemblage actuel alloue notamment num/wsum float64 et plusieurs plans canvas.
M0 doit chiffrer ce minimum et celui des finitions, pas seulement la lecture
des patches. M5 doit fournir un parcours borné scientifiquement validé OU une
attente/reprise explicite de cette phase. Si le minimum structurel dépasse la
machine même au repos, annoncer le blocage de capacité : ne pas faire attendre
indéfiniment ni découvrir après dix heures une impossibilité prévisible.

La RAM disponible peut chuter entre la sonde et l'allocation (TOCTOU). Les
marges ne sont pas une réservation OS : prévoir MemoryError/échec natif/worker
tué et reprise à partir des commits. Sur Windows, distinguer RAM physique,
working set, mémoire épinglée et limite d'engagement (commit/pagefile) lorsque
les sondes sont disponibles ; aucune assimilation automatique swap=RAM.

## 6. Checkpoints et reprise

### 6.1 Contenu sauvegardé

Par run :
- version du schéma et contrat scientifique ;
- inventaire, identité de contenu des entrées et ordre ;
- canvas/WCS, grille, halo, memberships et paramètres effectifs ;
- décision global/per_cell et **résultat de jauge complet et atomique** :
  référence(s), poids, exclusions et coefficients. La jauge est un artefact
  resumable de premier ordre, pas un détail ; en mode global, l'identité de la
  trame de référence fait partie de l'identité scientifique figée ;
- dépendances/version de calcul nécessaires à la compatibilité ;
- journal d'événements, generations de plans et statut de finalisation.

Par cellule :
- identité, statut, bornes, core_slice, contributeurs et références ;
- tous les plans réellement requis par l'assemblage global, dont science,
  estimateur, support_w1/w2, n_eff, validité et surviving ;
- en per_cell, PATCH avec halo pré-offset/pré-feather, pas seulement le core ;
- exclusions et provenance, backend, paramètres effectifs ;
- en per_cell, **référence photométrique propre à la cellule** (jauge auto →
  per_cell) : elle est sauvegardée avec la cellule, jamais re-dérivée à la reprise
  à partir des seules cellules restantes ;
- tailles, dtypes, shapes et empreintes des artefacts.

Les champs exacts proviennent de l'audit des consommateurs M0. Préférer stocker
une représentation canonique sans doubler inutilement core et patch.

Les offsets calculés sur un jeu partiel ne sont jamais une autorité de reprise.
En V1, ils sont recalculés à l'assemblage depuis le jeu COMPLET validé de patchs.
Une solution finale peut être publiée comme provenance, avec l'empreinte de ce
jeu complet ; cela n'autorise ni une réutilisation partielle ni une mutation des
patchs pré-offset. L'ancre
centrale (offset 0) est déterministe ; si sa cellule est recalculée après
corruption, tout le graphe d'offsets est recalculé, jamais « réparé » localement.

### 6.2 Transaction disque

1. Réserver l'espace pour la cellule et ses métadonnées avant publication.
2. Écrire dans un espace temporaire propre à cette tentative/run.
3. Flush et synchronisation des fichiers selon les garanties de l'OS.
4. Vérifier tailles/shapes/empreintes.
5. Publier atomiquement le commit de cellule, en dernier, sur le même volume.
6. Mettre à jour l'index global atomiquement ; l'index doit être reconstruisible
   depuis les commits de cellule si une interruption survient entre 5 et 6.

Tester les vraies primitives Windows et Linux ; rename seul ne garantit pas
une résistance absolue à toute coupure électrique.
Éviter pickle/exécution de contenu ; schéma strict et chemins confinés au run.
Verrou mono-écrivain pour empêcher deux reprises simultanées ; PID seul
insuffisant (réutilisation), identifier session/processus et traiter le verrou
orphelin sans supprimer celui d'un processus vivant.

### 6.3 Identité et compatibilité

Une taille/date de fichier seule ne certifie pas une entrée identique.
Prévoir une empreinte de contenu vérifiée ; coût I/O mesuré et annoncé.
Distinguer empreintes d'octets fichier et de tableaux (le harnais N5000 avait
confondu ces deux contrats).

- Changement de RAM libre, workers ou prefetch : ne doit pas invalider la science.
- Changement de tile/backend : seulement selon contrat de parité qualifié.
- Changement d'entrées, WCS, halo, memberships, jauge ou science : pas de reprise
  silencieuse ; nouveau run ou refus avec raison.
- Nouvelle version logicielle : compatibilité explicite par schéma/contrat ;
  V1 conservatrice, pas de « force » contournant les signatures.
- Déplacement du dataset : différencier localisation et identité de contenu.
- Paramètres de finition : fingerprint séparé ; V1 peut rejouer uniquement la
  finition si sa compatibilité est prouvée, sinon refuser explicitement.

### 6.4 Reconstruction

Valider le store, restaurer identité/jauge, charger les descripteurs de résultats
terminés, puis ne planifier que les cellules manquantes.
Checkpoint isolé corrompu : invalider cette cellule et ses sorties dérivées,
recalculer si entrées/jauge valides ; sinon diagnostic sans mélange.
Jauge ou identité globale corrompue : ne pas récupérer des cellules « au jugé ».

L'assemblage doit pouvoir relire les résultats sans tout empiler en RAM.
Auditer les charges simultanées des patches, graphe d'offsets, feather et canvas.
Si une adaptation par lecture bornée est nécessaire, préserver ordre, arithmétique
et géométrie ; en faire un sous-lot dédié avec tests de parité.

Interruption pendant assemblage/finitions : recommencer cette étape à partir des
cellules immuables, sans réempiler et sans appliquer les offsets deux fois.
Les sorties finales restent temporaires jusqu'à publication cohérente.

### 6.5 Disque et conservation

Budgéter séparément caches jetables, cellules durables, assemblage et marge
transactionnelle. Le stockage des résultats croît avec la mosaïque ; le dire.
Ordre de grandeur à annoncer : sauvegarder les plans PATCH (science + masques +
support + surviving, avec halo) de 187 cellules peut représenter plusieurs fois
la taille d'une mosaïque finale en RAM ; c'est précisément pour cela que la
relecture bornée de l'assemblage (M5) est obligatoire, et que le store de
checkpoints est lui-même une ligne du budget disque, pas un résidu gratuit.
Disque plein : préserver les anciens commits ; pas de faux COMPLETE. Après
libération d'espace, recalculer seulement les tentatives non committées ;
plusieurs peuvent être en vol dans une vague CPU.
Ne jamais supprimer automatiquement des checkpoints après échec/annulation.
Après succès, rétention/purge explicite et documentée ; ne pas supprimer les
anciens résultats utilisateur pour faire de la place.

Les caches de projection/cache_build sont produits de façon incrémentale par
cellule dans le pipeline borné et restent jetables ; la jauge, elle, est un
artefact durable à ne pas confondre avec un cache.

### 6.6 Amorçage de reprise et compatibilité exécutable

Ordre impératif : choisir/ouvrir le run → verrou → lire identité/état minimal →
valider les entrées/artefacts → restaurer géométrie et jauge → admettre seulement
le travail restant. Ne pas passer d'abord par un nouveau layout adaptatif,
une nouvelle jauge ou l'admission du lot original. Un run dont toutes les cellules
sont committées doit pouvoir reprendre l'assemblage sans admission GPU de stack.

La reprise porte sur l'ENSEMBLE des cell_id valides, pas un préfixe 0..K.
Une corruption au milieu, des retours CPU hors ordre ou une cellule vide créent
des ensembles clairsemés ; ne pas utiliser offset=len(commits) pour les ignorer.
L'identité des cellules reste globale, distincte de l'index de la file restante.
Vérifier la condition de complétude exacte attendus = COMPLETE + EMPTY_VALIDATED.

Politique V1 proposée : zegrid_resume_mode=auto|never, auto pour une nouvelle
configuration ZeGrid ACTIVE ; never crée un run neuf sans écraser les anciens.
Never désactive la réutilisation au lancement, PAS la sauvegarde de sécurité du
nouveau run : reprise automatique et production des checkpoints sont distinctes.
Pas de force. Si plusieurs runs correspondent, sélection explicite, jamais
« dernier dossier trouvé ». Le dossier contient des générations run_id séparées.
Un auto incompatible s'arrête avec explication, ne repart pas dix heures de
calcul silencieusement. Un run déjà complet est un no-op vérifié.

Restaurer la politique effective et les transitions dense→bounded ; le restart
ne doit pas réinitialiser automatiquement le run en dense ni son compteur de
fallback. Revalider ressources et environnement du nouveau processus ; nouvelle
génération de plan sans changer identité scientifique. Un changement explicite
de politique incompatible donne un diagnostic, pas une reprise ambiguë.

Les paramètres réellement utilisés (y compris valeurs implicites, type/dtype,
rejet, unités, flags d'optimisation de jauge et version de contrat) entrent dans
le fingerprint, pas seulement le JSON GUI. Figer l'inventaire de sélection ;
ne pas redécouvrir de nouvelles images ajoutées au dossier pendant le run.
Les contrôles de fichiers doivent détecter une modification pendant hachage/
lecture autant que possible ; une entrée changeante invalide la tentative.
L'identité au seul nom/mtime ou un hash jamais revalidé ne suffit pas.

### 6.7 Publication, orphelins et formats

- Un reçu de commit perdu après écriture ne justifie pas un recalcul :
  rechercher la transaction par cell_id/attempt_id avant retry. Un retry de
  publication doit être idempotent ; rejeter deux contenus différents pour
  une même identité validée.
- Les états pré-offset sont immuables sur disque et ouverts en lecture seule.
  Interdire qu'un mmap writable applique en place un offset ou une finition
  aux checkpoints, sinon chaque reprise modifierait les données de référence.
- Matrice de compatibilité obligatoire : schéma/contrat/backend/tile/dtype/
  dépendances. Égalité scientifique qualifiée avant autorisation d'un mélange ;
  à défaut, rester dans le même chemin qualifié, attendre ou arrêter de façon
  reprenable. Un « petit écart visuel » n'est pas un critère d'acceptation.
- Atomicité du fichier final ne rend pas atomique l'ensemble FITS/coverage/
  manifest. Publier une génération complète puis son marqueur final ; ne pas
  montrer une nouvelle science avec une ancienne coverage ou un manifest ancien.
- Distinguer déchets temporaires de ce run, commits valides orphelins d'index,
  caches réutilisables et anciens runs. Cleanup explicite confiné à run_id ;
  tester symlinks/jonctions, chemins absolus non attendus et fichiers étrangers.
- Première qualification : système de fichiers local Windows NTFS et Linux ;
  réseau/cloud synchronisé/non-local déclaré non qualifié, sans promesse de
  verrou/rename durable identique.
- Verrou fondé sur propriété OS/instance, pas expiration de heartbeat seule :
  une veille ne permet pas de voler le verrou d'un processus vivant.
- Les journaux d'événements et l'index ne doivent pas devenir O(nombre de pixels)
  en RAM ni être réécrits en entier à chaque heartbeat. Borner les traces
  répétitives et conserver les événements de transition nécessaires à l'audit.

## 7. Découpage et gates

Chaque jalon = une mission Coco bornée, vérification Junior, revue Nono.
Pas de lancement du jalon suivant avant acceptation ; maximum trois corrections
par tâche. Les tâches restent petites et séquentielles.

| Jalon | Livrable | Critère de sortie |
|---|---|---|
| M0 — contrats et reproduction | Inventaire payloads/owners, machine d'états, témoins du vrai moteur, budget de toutes les phases | Défaut baseline reproduit ; contrats des sections 5–6 figés ; squelette de tranche verticale testé sur Windows |
| M1 — conservation runtime | Registre de résultats par cellule, journal en mémoire, transaction d'acceptation, séparation erreurs/backend | Une panne après K cellules n'efface pas K résultats ; pas de double comptage ni dépendance à la file GPU |
| M2 — store durable | Identité/jauge, commits atomiques, verrou, intégrité, budgets disque ; libération des gros tableaux terminés | Kill pendant chaque étape d'écriture : seuls les commits complets sont récupérés |
| M3 — recovery ressources | CPU/GPU, AUTO→bounded, attente annulable, générations, préparation/jauge/commit inclus | Pression après K cellules puis retour RAM : terminaison sans refaire K ; attente réellement légère ; aucun résultat tardif accepté |
| M4 — reprise bout en bout | Reconstruction du run et des seules tâches restantes, CPU/GPU selon compatibilité | Nouveau processus après kill : résultat de référence, aucune cellule valide recalculée |
| M5 — assemblage reprenable | Relecture bornée, offsets/feather/finition transactionnels | Kill pendant assemblage : reprise sans double offset, trous nouveaux ou pics non budgétés |
| M6 — GUI/CLI et documentation | Finalisation UX/traductions ; contrôle coopératif et reprise amorcés dès M1–M4 | Chemin GUI Windows réel, Stop and Save sans kill automatique à 4 s ; fermeture contrôlée ; Classic inchangé |
| M7 — qualification Windows | Petit corpus instrumenté puis M31 réel et preuve de reprise | Dégradation et reprise réelles validées ; artefacts scientifiques/mémoire documentés |

M1 apporte la conservation runtime ; M2 fonde la durabilité ; aucun de ces lots
isolés ne sera présenté comme « resume complet ». Le contrôle coopératif GUI/worker
et un outil CLI de test arrivent dès M1–M4 : M6 n'est PAS la première vérification
Windows. Avant de généraliser, valider une tranche verticale minimale :
vrai moteur → commit de cellule → baisse RAM → attente → kill → nouveau worker
→ chargement du commit → assemblage identique. Si elle échoue, ne pas empiler
les jalons suivants sur des composants uniquement testés séparément.

### Responsabilités par zone

- Coordinateur/recovery : moteur ZeGrid, runtime_control, scheduler, ram_contract.
- Ownership/threads : parallel ; arrêt producteur, leases et queue drainées.
- Checkpoints : nouveau store isolé + couche de résultats existante.
- Science : consommateurs assembly/cell_feather/cell_offset, modifications
  minimales d'accès aux données, pas de nouvelle formule.
- Intégration : worker/config/GUI/CLI/locales.
- Tests : compléter les familles R55-E1/E2/E11, OOC recovery, photométrie/assembly,
  nouveaux tests transactionnels et d'intégration produit.

## 8. Matrice de validation obligatoire

### Pression / exécution

- Pression avant cellule, entre cellules, durant préparation, pendant stack.
- Dense→bounded avec GPU conservé ; CPU déjà bounded ; profils forcés respectés.
- Prefetch 0 et >0 : file drainée, aucune lease perdue, aucun deadlock.
- Workers/concurrence réduits ; minimum atteint → attente puis reprise.
- RAM oscillante : hystérésis, pas de boucle chaude. Le dense→bounded étant à
  sens unique, le cas critique est « bounded insuffisant → attente », pas un
  ping-pong de modes.
- Retour mémoire n'autorisant pas une croissance implicite du plan.
- Annuler pendant attente : accusé <2 s dans le témoin, hors opération native
  non interruptible clairement documentée.
- Arrêt propre demandé pendant COMMITTING : transaction terminée si possible ;
  sinon aucune cellule partielle validée. Reprise des anciens commits même si
  STOPPED_RESUMABLE n'a pas pu être écrit.
- Arrêt forcé après un arrêt propre trop lent : état récupérable, un seul écrivain,
  aucun descendant survivant qui publie tardivement dans le run.
- OOM contrôlé et contexte GPU perdu : unité incomplète non committée.
- Observation RAM absente/invalide, erreur I/O/science : aucune fausse récupération.
- Pression pendant la JAUGE : réduire/attendre aux frontières de tâches sûres
  sans abandon automatique du run ; si une allocation native échoue, nettoyer
  la tentative et réessayer selon la politique bornée. Sans checkpoint interne,
  un processus interrompu recommence la jauge non committée, coût affiché.
- Préparation, hash, cache, commit et assemblage/finitions : la gestion mémoire
  n'est pas limitée à per_cell_stack ; témoin de pression pour chaque phase.

### Durabilité / compatibilité

- Kill réel de sous-processus après K commits, pendant data/commit/index.
- Redémarrage Windows, fichiers ouverts/verrous et remplacement atomique.
- Veille/hibernation puis réveil : checkpoints intacts, contexte GPU recréé,
  reprise correcte.
- Disque plein, permissions, checksum faux, payload tronqué et métadonnées corrompues.
- Deux lanceurs sur le même run : un seul propriétaire.
- Entrée modifiée même taille, paramètre scientifique changé, version incompatible.
- Déplacement compatible du corpus ; ordre des entrées inchangé.
- Commit valide oublié par l'index : récupération sans recalcul.
- Stop/reprise au moins deux fois ; cache jetable absent.
- Entrée de reprise dédiée : `zegrid_resume_mode` (ou équivalent) distinct de la
  clé `resume` Classic, auto-détection du run par dossier, un seul propriétaire.

### Science / produit

- Global et per_cell, cellules vides, composantes déconnectées, exclusions réelles.
- Comparaison des plans science/coverage/support/validité/surviving, références,
  exclusions et offsets — pas seulement l'image affichée.
- Même backend/environnement : identité exacte là où le contrat est déterministe.
- CPU/GPU ou autre chemin numérique : tolérances fixées AVANT mesure et justifiées
  par les tests scientifiques existants, jamais élargies pour obtenir du vert.
  Sans preuve acceptable, mélange backend interdit.
- Reprise assemblage/finition : résultat identique, pas de correction doublée.
- Le coût de la preuve « reprise == référence » : la prouver sur petits corpus
  complets avec un témoin sans interruption. Pour M31, ne pas inventer une
  référence : le run échoué n'en fournit pas. Comparer à une référence existante
  seulement si entrées/version/science/géométrie sont effectivement compatibles.
  Sinon qualifier la complétude et la reprise du grand run, publier les contrôles
  locaux et conserver explicitement « parité M31 complète non mesurée ».
- Test de propagation config→GUI/CLI→worker→signature moteur, pas seulement settings.
- Régressions Classic/legacy/observe et mode sans reprise.
- Tests adversariaux branchés sur le chemin produit, pas uniquement des mocks
  du helper qui évitent la frontière ayant échoué sur M31.

### 8.1 Stratégie courte avant tout grand run

Le but est d'éviter une correction qui passe les helpers mais échoue dix heures
plus tard. Construire d'abord une matrice de chemins, pas une copie miniature
uniquement en jauge globale à WCS identiques.

**T0 — modèle de transitions**, horloge et observations injectables :
budget haut→bas→haut, invalidité, refus réel, même cellule répétée, génération
périmée, sparse commits, STOP/WAIT, commit sans ack. Aucun sleep réel long.

**T1 — petit end-to-end produit**, quelques cellules, halo valide, vrais FITS,
vraie jauge et vrais chemins d'assemblage. Cas global puis auto→per_cell,
cellule vide et chevauchements, prefetch0 puis >0. Ne pas baisser les planchers
scientifiques pour rendre le corpus plus petit. Injecter la pression après
au moins deux commits, pas seulement à l'admission initiale.
Objectif de coût : quelques minutes par scénario représentatif ; mesurer et
garder un budget de suite. Si lent, isoler les fixtures, pas supprimer les gates.

**T2 — vrai arrêt de processus et GUI Windows**, depuis le runtime ciblé :
Stop and Save, fermeture, kill pendant commit, relance, reprise. Prouver que la
commande coopérative n'est pas interceptée par l'ancien terminate à quatre secondes.
Exercer CPU spawn et chemin thread sous parent daemon, pas seulement Linux fork.

**T3 — stress des dimensions**, sans stack M31 complet :
- beaucoup de cellules/manifests et résultats tardifs ;
- gros patch/canvas pour détecter les allocations de M5 ;
- forte profondeur de contributeurs pour le working set ;
- accumulation de commits pour mesurer la pente RSS ;
- chemins longs/espaces et contention fichier Windows.
Un petit N ne prouve pas un gros N ; de nombreux petits fichiers ne prouvent
pas l'assemblage d'un grand canvas. Couvrir séparément ces dimensions.
Un replay des faits M31 valide la décision, pas les pics mémoire réels.

**T4 — canari Windows puis M31**, uniquement après T0–T3 verts.
Première tranche verticale au plus tôt M2–M4, et tests cumulés à chaque jalon.
Le test de reprise doit démontrer via instrumentation du vrai calcul que les
cellules réutilisées et la jauge committée ne sont pas recalculées : lire un
compteur du manifest écrit par l'implémentation ne constitue pas une preuve.

### 8.2 Barrières contre les faux verts

- Le témoin initial reproduit l'erreur baseline sans la marquer comme succès.
  Pour le candidat, aucune régression ciblée xfail/skippée ; un test GPU sans
  GPU est NOT_RUN, jamais une qualification GPU.
- Matrice critères→tests→preuves jointe à chaque rapport, avec SHA produit,
  chemin du runtime, commande exacte, durée et statut.
- Réutiliser les corpus déterministes entre variantes ; empreintes et profils
  explicites. Baseline et candidat ont les mêmes paramètres scientifiques.
- Mesurer un témoin sans pression pour estimer le surcoût checkpoint (temps,
  écriture, RSS). Ne pas promettre « rapide » sans mesure ; éviter de sérialiser
  tout le dataset ou tous les résultats à chaque cellule.
- Mesurer le plateau de WAITING_RESOURCES et la pente mémoire vs cellules
  committées, pas seulement le pic d'un test de trois cellules.
- Vérifier zéro calcul perdu pour les commits valides, au plus les tentatives
  en vol perdues (plusieurs possibles en CPU), pas « au plus une » globalement.
- Revue indépendante des propriétaires et des transitions avant grand run.
- Aucun lancement M31 tant qu'une preuve obligatoire manque ; le rapport peut
  rester BLOCKED plutôt que déléguer la découverte d'un défaut au test nocturne.

## 9. Qualification et observabilité

Journal structuré : raison de pause, budget/observation, mode avant/après,
génération du plan, cellule et tentative, checkpoints committés/réutilisés,
chemins des artefacts, backend réel. Distinguer pause RAM/disque/utilisateur.

Le manifeste final doit publier la **provenance de reprise** : `resumed_from`,
nombre de cellules réutilisées vs recalculées, backend par cellule, générations
de plan traversées et raison de chaque pause. Sans cela, impossible d'auditer
qu'« aucune cellule valide n'a été recalculée ».

Mémoire : RAM système disponible, RSS processus/arbre (avec limites de mesure),
VRAM et réservations séparées. Ne pas appeler une réservation « pic mesuré ».
ETA suspendue pendant attente, calculée sur travail restant et vitesse observée,
sans compter à nouveau les cellules reprises.

Avant M31 : petit corpus réel/synthétique multicellule couvrant per_cell,
avec pression injectée déterministe et interruption réelle.
Sur Windows : toutes les sorties d'essai dans un nouveau dossier sous
C:\Users\JarvisJunior\Desktop\zgprobe ; ne pas écraser le M31 échoué.
Éviter d'épuiser physiquement la RAM du PC pour tester ; utiliser d'abord des
observations contrôlées, puis une concurrence utilisateur bornée/mesurée.
M31 longue durée seulement après gates courts verts ; mesurer temps de stack,
temps d'attente, coût des checkpoints, disque et mémoire.
Ne pas utiliser le PASS N5000 comme substitut à cette qualification.

Mesurer explicitement le coût de la jauge re-lancée et le coût disque réel des
checkpoints de patchs (en % de l'espace final), pour que la note de version
annonce des chiffres et non des ordres de grandeur vagues.

## 10. Définition de terminé et publication

Mission ACCEPT seulement si :
- pression temporaire puis rétablissement → run complet sans perte des commits ;
- kill/restart → seules cellules incomplètes recalculées ;
- incompatibilité/corruption → diagnostic correct et aucun faux succès ;
- assemblage scientifique validé, identités photométriques préservées ;
- disque/RAM bornés par le modèle déclaré, aucune protection contournée ;
- état GUI/CLI fidèle ; rapport de qualification Windows et artefacts disponibles ;
- toutes les constatations importantes Nono résolues et vérifiées par Junior.

Aucun push, merge beta/main, tag ou déploiement autorisé par cette demande de plan.
Préparer une note de version expliquant l'espace disque, le mode de reprise,
l'absence de récupération rétroactive et les limites d'une pause de ressources.

## 11. Taille et risques

Chantier significatif : plusieurs lots, pas un patch unique. Le code CPU existant
réduit le coût du recovery ; la durabilité Windows, l'identité photométrique,
l'ownership GPU et la mémoire d'assemblage portent l'essentiel du risque.

Risques renforcés à la relecture : (1) la durabilité de la JAUGE — une phase de
plusieurs heures, seule en amont de tout, dont la perte n'est pas couverte par le
resume par cellule ; (2) la taille disque des checkpoints de patchs ; (3) la
libération réelle des ressources pendant l'attente ; (4) la tolérance à la
veille Windows. Leurs contrats doivent être figés dans M0 et leurs preuves
réparties sur M2–M6 ; aucun jalon antérieur ne suffit à déclarer le resume complet.

Ne pas annoncer un délai ferme avant M0. Pas besoin présumé d'une grosse nouvelle
dépendance : formats tableaux + JSON strict et primitives OS existantes à évaluer.
Tout changement scientifique, dépendance majeure, nouvelle migration ou changement
de périmètre sortant du plan revient à Tristan avant exécution.

### Prochain pas unique après validation du plan

Lancer M0 : reproduire de façon courte la perte du travail GPU après K cellules,
cartographier les payloads/science/ownership et figer le contrat de checkpoint.
Ne pas commencer par réécrire le moteur ni par relancer M31 dix heures.


## 12. Registre de deuxième revue — angles morts rendus bloquants

| Constat | Conséquence possible | Contrat/gate ajouté |
|---|---|---|
| Stop and Save termine le worker après 4 s | Checkpoint interrompu malgré promesse GUI | Contrôle coopératif M1–M4, test GUI Windows T2 |
| Résultats conservés dans pipeline/listes/views | RSS croît même avec un store disque | Reçus légers et mesure pente RSS T3 |
| Reprise via bootstrap normal | Nouvelle grille/jauge ou refus avant lecture du store | Amorçage checkpoint-first 6.6 |
| Résultats partiels d'une vague et trous de commits | Cellules ignorées ou recalculées | Ensemble cell_id, pas compteur/prefixe |
| Ancien producteur encore vivant | Payload dense servi au plan bounded, commit tardif | Generations/fencing et drainage 5.7 |
| Attente tenant des crédits ou pools hôte | Deadlock ou pression auto-entretenue | Plateau d'attente mesuré, recyclage sûr |
| Pression limitée au stack | Panne jauge/hash/commit/finition | Contrat par phase 5.8 |
| Gros accumulateurs canvas | Échec après tous les stacks | Préflight M0 et M5 obligatoires |
| Allocation après sonde favorable | OOM malgré admission verte | Gestion de la course, commits préservés |
| Même erreur relancée sans changement | Livelock au lieu de crash visible | Retry borné puis attente/diagnostic |
| Identité basée sur settings/chemins | Mauvais run repris ou science modifiée | Configuration effective et contenu, inventaire figé |
| Ack perdu / index incomplet | Double calcul ou double publication | Commit idempotent, scan des commits |
| Offset appliqué sur mmap du checkpoint | Dérive à chaque reprise | Source immuable pré-offset |
| Publication de plusieurs fichiers non atomique | Mélange science/coverage de runs différents | Génération finale et marqueur dernier |
| « Référence M31 existante » supposée | Parité proclamée sans témoin | Référence compatible vérifiée ou NOT_MEASURED |
| Test court seulement N petit/global | Faux vert sur multicellule per_cell/gros N | T0–T4 et stress dimensionnel |
| Flush avant veille et commit toujours garanti | Promesse impossible lors d'arrêt imprévu | Garantie limitée aux commits déjà validés |
| Plan immense exécuté d'un seul bloc | Découverte tardive des interfaces incompatibles | Tranche verticale tôt, gates cumulatifs |

Pas de garantie « premier coup » pour un changement de cette ampleur.
L'objectif vérifiable est de faire échouer les cas difficiles en quelques minutes
sur le banc avant d'engager le run nocturne, et de ne plus perdre ce qui a été
validé si un incident inédit apparaît ensuite.

Revue réalisée par Junior sur le code et les preuves ; pas une revue Nono,
pas une qualification d'implémentation. Aucun code produit modifié.
