# ZeGrid — plan détaillé : petites tuiles RAM + streaming externe automatique

Date : 2026-11-07  
Mission proposée : `ZM-ZEGRID-AUTO-EXTERNAL-STREAMING-20261107`  
Baseline : `origin/beta=f8acdbb0ec34bef7dd21e69deadf9352103fdc3c` — ZeMosaic 5.1.1  
Worktree : `/home/tristan/.openclaw/workspace/worktrees/zemosaic-auto-external-streaming`  
Branche : `feat/zegrid-auto-external-streaming`  
Statut : **PLAN VALIDÉ PAR TRISTAN — progression séquentielle autorisée jusqu'à M5 si les gates restent verts ; aucun push/merge/tag/release**.

## 1. Objectif produit

Rendre ZeGrid capable de traiter automatiquement un stack arbitrairement profond sans changer sa sémantique scientifique :

1. utiliser le chemin in-memory quand il tient confortablement ;
2. utiliser le streaming RAM borné quand une tuile canonique tient ;
3. réduire la tuile CPU sous 32 px lorsque cela suffit ;
4. basculer automatiquement vers un workspace externe sur disque quand aucun workspace RAM ne tient ;
5. refuser avant calcul seulement si ni RAM ni disque ne permettent un plan complet et honnête.

Les deux fronts sont complémentaires :

- **Front A — sous-tuilage CPU RAM** : étendre la ladder sous le plancher historique de performance 32 ;
- **Front B — streaming externe** : supprimer la dépendance résiduelle `O(N × tile_area)` de la RAM.

Le choix doit être automatique sous la politique existante `storage_profile=auto`, sans réglage manuel nécessaire dans le GUI.

## 2. Témoins immédiats

### 2.1 M33/S50 Windows

Artefacts audités : copie de `D:\M33\S50\out\ircut` sous `C:\Users\JarvisJunior\Desktop\M33\ircut`.

Faits :

- 22 414 entrées ;
- machine : 31,7 Gio de RAM, RTX 3070 Laptop 8 Gio ;
- trois tentatives : halo 64, 32 puis 8 ;
- dernier budget PLAN : 10 022 Mio ;
- halo 8 : dernière grille respectant les planchers = 30×51, soit 1 530 cellules ;
- pire cellule bounded, CPU tile32 : 12 469,2 Mio ;
- dense : 195 485,8 Mio ;
- refus correct avant jauge/stack, pic processus observé ≈1,03 Gio ;
- le modèle implique que presque toutes les 22 414 entrées contribuent à la pire cellule.

Le halo n'est plus le facteur dominant. La limite est le workspace canonique par tuile, encore proportionnel à N.

Alerte données : au moins un nom `_LP_` apparaît dans l'inventaire parcouru alors que l'entrée est IRCUT. Le lancement scientifique doit produire un inventaire des filtres/noms et s'arrêter pour décision humaine si le mélange est confirmé ; aucune suppression automatique.

### 2.2 Caldwell TinyDebian

Témoin demandé par Tristan pour le lendemain : stack Caldwell avec halo 128, actuellement refusé sur TinyDebian.

Le lancement devra utiliser :

- le corpus Caldwell réel déjà présent ;
- halo explicitement fixé à 128 ;
- un dossier de sortie neuf ;
- scheduler ACTIVE, stockage AUTO, checkpoints/reprise actifs ;
- aucune baisse automatique du halo, aucun retrait de brute, aucun abaissement de plancher scientifique.

Le chemin exact du corpus et le disque de scratch seront vérifiés au dry-plan avant tout lancement lourd.

## 3. Diagnostic architectural

Le profil bounded actuel supprime déjà :

- la résidence simultanée des caches alignés N×patch ;
- le masque de rejet dense N×patch×C ;
- la conservation de plusieurs caches de cellules.

Il conserve cependant le workspace de phase 2 :

`N × extended_tile_height × extended_tile_width × channels`

Le code rassemble aujourd'hui, pour une tuile, `images_t`, `valid_t`, `images64`, `a_map`, `wmap`, `survivor`, `masked`, `winsor` et les temporaires du rejet. À N≈22 414, tile32 avec halo de taper interne produit encore une borne d'environ 12,5 Gio.

Un simple accumulateur `numérateur + poids` ne peut pas remplacer ce workspace lorsque `winsorized_sigma_clip` ou `kappa_sigma` est demandé : les quantiles, écarts-types itératifs et masques survivants nécessitent l'axe N. La solution exacte est un **spool externe des échantillons**, puis une réduction par micro-lots de colonnes.

## 4. Invariants non négociables

1. Même inventaire ordonné, WCS, canvas, grille retenue, halo, memberships et contributeurs.
2. Une cellule logique reçoit toujours toutes ses brutes légitimes ; aucun sous-stack scientifique indépendant puis fusion approximative.
3. Même normalisation, référence, coefficients de jauge, poids, rejet, combine, taper et ordre des frames.
4. Aucun changement automatique du halo ou des planchers `min_patch_area` / `max_halo_overhead`.
5. Le choix RAM/disque est un choix de résidence et d'exécution, pas de science.
6. Aucun fallback silencieux de `winsorized_sigma_clip` vers `none`, mean hiérarchique ou autre approximation.
7. Sorties et diagnostics canoniques identiques au chemin de référence là où le contrat est déterministe.
8. Les cellules déjà committées restent valides et ne sont pas recalculées après pression ou redémarrage.
9. Pas de swap compté comme RAM disponible.
10. Disque plein, permission, corruption ou observation RAM invalide restent des erreurs distinctes.

## 5. Architecture cible

### 5.1 Trois classes d'exécution sous AUTO

Le planificateur énumère, dans cet ordre de préférence, les candidats :

1. `inmem` ;
2. `stream_ram(tile)` ;
3. `stream_external(tile, column_batch)`.

Le profil utilisateur reste `auto|dense|bounded`. `stream_external` est un **mode de working set interne au profil bounded**, afin de ne pas casser le schéma durable de politique ni créer un bouton utilisateur prématuré.

Le manifeste ajoute :

- `stack_working_set = inmem|ram_tile|external_spool` ;
- tuile spatiale ;
- taille du micro-lot de colonnes ;
- schéma du spool ;
- RAM, disque et I/O prédits ;
- raison du choix et candidats refusés.

### 5.2 Front A — ladder CPU sous 32

Créer une ladder CPU distincte de la ladder GPU. Première proposition à qualifier :

`128, 112, 96, 80, 64, 48, 32, 24, 20, 16, 12, 10, 8`

Règles :

- choisir la plus grande tuile admissible, jamais « la plus petite par défaut » ;
- conserver le minimum GPU 32 tant que les petites tuiles GPU ne sont pas qualifiées ;
- ne pas modifier le halo de taper interne ;
- tenir compte du coût fixe par tuile : la mémoire n'est pas monotone sous une certaine taille ;
- arrêter l'échelle RAM lorsqu'elle recommence à coûter plus cher, puis évaluer `stream_external`.

Le plancher 32 est documenté dans le code comme plancher de performance, jamais scientifique. Néanmoins aucune petite tuile ne sera promue sans preuve exacte des sept plans canoniques, diagnostics et provenance.

### 5.3 Front B — spool externe exact

Pour chaque cellule, puis chaque tuile spatiale étendue :

1. lire chaque frame dans l'ordre canonique depuis le provider bounded ;
2. appliquer les coefficients fixes de normalisation exactement comme aujourd'hui ;
3. calculer validité et taper exactement comme aujourd'hui ;
4. écrire dans un spool versionné :
   - échantillons normalisés float32 ;
   - validité nécessaire ;
   - facteur de poids/taper dans le dtype requis pour reproduire les opérations float64 ;
   - ordre et identité des frames ;
5. fermer/flush le spool ;
6. réduire les colonnes pixel-canal par micro-lots `B` :
   - charger seulement N×B ;
   - convertir dans les mêmes dtypes ;
   - appeler les primitives canoniques existantes de rejet ;
   - conserver le même ordre de réduction N ;
   - écrire immédiatement science, poids, validité, surviving et diagnostics ;
7. supprimer le spool de la tuile après validation de son résultat ;
8. committer la cellule par le store durable existant.

La matrice N×B doit être la seule structure proportionnelle à N en RAM. `B` est choisi par l'admission à partir du budget frais, avec une borne minimale explicite.

### 5.4 Format et propriété du scratch

Racine proposée : sous le cache visible du run, sur le volume de sortie, jamais sous `/tmp` :

`zegrid-cache/external-spool/<run_id>/<cell_id>/<attempt_id>/`

Contrats :

- marqueur de propriété et chemins confinés ;
- fichiers par génération/tentative ;
- métadonnées JSON strictes, aucun pickle ;
- flush/close avant réduction ;
- stale scratch d'une tentative morte supprimable uniquement après vérification du verrou/propriétaire ;
- scratch non considéré comme checkpoint scientifique ;
- résultat de cellule durable publié en dernier par le mécanisme existant ;
- après crash au milieu d'une cellule : reprise de cette cellule, réutilisation des cellules COMPLETE.

### 5.5 Planification jointe RAM + disque + travail

L'externalisation ne doit pas faire choisir naïvement la première grille grossière : une grosse cellule peut tenir en RAM externe tout en produisant un cache, un scratch et un temps absurdes.

Pour chaque candidat de grille admissible scientifiquement, calculer :

- max et somme des contributeurs par cellule ;
- surface de patch max et totale ;
- RAM par mode/tuile/B ;
- cache aligné maximal d'une cellule ;
- spool maximal d'une tuile ;
- résultats durables ;
- marge transactionnelle ;
- volume d'écriture estimé sur tout le run ;
- unités de travail `Σ(N_cell × patch_area_cell)` et `Σ(N_cell × tile_count_cell)`.

Sélection : meilleur coût prédit parmi les plans RAM+disque faisables, sous planchers scientifiques inchangés. Le dry-plan publie les trois meilleurs candidats et la raison du choix.

### 5.6 Sélection et fallback automatiques

Au PLAN, avant jauge :

1. observer RAM et disque ;
2. construire les candidats complets ;
3. sélectionner le plus rapide admissible ;
4. figer géométrie et identité scientifique.

À ADMIT/LAUNCH :

1. revalider RAM et disque ;
2. réduire concurrence/prefetch ;
3. réduire la tuile RAM ;
4. passer de `stream_ram` à `stream_external` si nécessaire ;
5. ne jamais regrossir automatiquement le plan dans le même épisode.

En RUN, basculer uniquement à une frontière transactionnelle sans unité en vol. Aucun résultat partiel de tuile/cellule ne doit être accepté après une transition.

TinyDebian choisira directement `stream_external` si toutes les tuiles RAM refusent. Windows M33 pourra choisir tile24/20 si la RAM suffit, sinon external automatiquement.

### 5.7 Reprise et identité

Ajouter à l'identité d'exécution durable :

- schéma de l'algorithme external ;
- mode de working set effectif ;
- qualification de parité.

La taille `B` et la petite tuile peuvent changer après reprise seulement si leur invariance exacte a été qualifiée et enregistrée. Sinon elles restent gelées et un manque de capacité produit un arrêt reprenable explicite.

Les anciens stores 5.1.1 restent lisibles ; aucun ancien run n'est implicitement converti en external.

## 6. Modèle de ressources

### 6.1 RAM

`stream_ram` conserve le modèle actuel étendu aux petites tuiles.

`stream_external` facture au minimum :

- baseline processus ;
- provider bounded : référence + frame courante ;
- préparation d'une tuile d'une frame ;
- matrice N×B valeurs/survivants ;
- temporaires exacts du rejet pour B colonnes ;
- sorties O(patch_area) ;
- cache-build worker réellement simultané ;
- réserve de contrôle/commit.

Le modèle doit être calibré par mesures, mais jamais remplacé par un facteur optimiste universel.

### 6.2 Disque

Budget peak distinct du volume d'écriture cumulé :

- cache aligné d'une cellule ;
- spool de la tuile active ;
- outputs temporaires de la cellule ;
- transaction de commit ;
- résultats durables de toutes les cellules ;
- publication finale et marge.

Le plan refuse avant jauge si le peak + marge ne tient pas. Une baisse d'espace pendant le run suspend/refuse avant nouvelle cellule ; elle ne détruit jamais les commits antérieurs.

### 6.3 Temps et usure disque

M33 possède 1 530 cellules candidates et 22 414 frames. Même après correction RAM, le run peut être très long. Le dry-plan doit annoncer :

- frame-cell memberships totales ;
- octets lus/écrits prédits ;
- temps estimé par phase avec incertitude ;
- avertissement explicite si le run se compte en jours ou écrit plusieurs To cumulés.

Aucune promesse « le disque résout de facto le temps ».

## 7. Découpage de la mission nocturne

Chaque lot est vérifié par Junior ; une revue indépendante intervient avant les runs réels. Pas de suite exhaustive de 4 500 tests dans la boucle.

Autorisation de validation nocturne : M106 peut servir de témoin réel court et
borné quand une preuve synthétique ciblée ne suffit pas. Ne pas lancer de mosaïque
complète ; préférer un dry-plan, une cellule ou le plus petit sous-témoin capable
de fermer le contrat. Caldwell et M33 restent les validations utilisateur du
lendemain, pas des corpus de développement nocturne.

### M0 — contrats et dry-plan

Livrables :

- reproduction déterministe des refus M33 depuis les faits capturés ;
- modèle pur des candidats petites tuiles/external ;
- rapport RAM/disque/travail avant calcul ;
- contrat de manifeste/refus.

Gate : M33 H8 et Caldwell H128 produisent un dry-plan lisible sans jauge, stack ni écriture scientifique.

### M1 — Front A petites tuiles CPU

Livrables :

- ladder CPU séparée ;
- plan/admission/runtime alignés ;
- GPU inchangé ;
- propagation réelle jusqu'à l'executor.

Gate : parité exacte tile32 vs 24/20/16/12/10 sur témoins déterministes ; M33 facts-replay choisit la plus grande tuile faisable, probablement 24 selon le modèle actuel.

### M2 — primitive external par micro-lots

Livrables :

- spool confiné/versionné ;
- reducer CPU exact sur N×B ;
- diagnostics/rejection sink exacts ;
- nettoyage et erreurs typées.

Gate : résultats identiques au streaming RAM pour plusieurs N, B, tailles de tuile et algorithmes de rejet.

### M3 — wiring AUTO et admission disque

Livrables :

- candidat `stream_external` dans layout/scheduler/admission ;
- transition automatique RAM→external ;
- disk preflight ;
- manifeste et logs.

Gate : un scénario où dense et toutes tuiles RAM refusent termine via external sans modification de géométrie/science.

### M4 — durabilité et reprise

Livrables :

- tentative/tile scratch transactionnels ;
- kill/disque plein/interruption ;
- reprise des seules cellules manquantes ;
- aucun stale scratch accepté.

Gate : kill à chaque étape sensible, résultat final identique, commits antérieurs inchangés.

### M5 — intégration produit courte et revue

Livrables :

- chemin GUI→worker→moteur ;
- vérification des kwargs ;
- petit E2E global et per_cell ;
- revue Nono, défauts matériels corrigés.

Gate : tests ciblés verts, diff compris, dry-plans réels Caldwell/M33 admis, aucun blocker connu avant lancement utilisateur.

## 8. Matrice de tests ciblés

### Science

- none / kappa / winsorized ;
- mean / median ;
- sky_mean et jauge fixe ;
- RGB et mono ;
- NaN, faible N, colonnes dégénérées, égalités de quantile ;
- global et per_cell ;
- taper avec bords/coins et petites tuiles ;
- sept plans canoniques, exclusions, compteurs, provenance.

Critère : byte-identique lorsque le chemin de référence est déterministe ; aucune tolérance élargie pour obtenir un PASS.

### Ressources et erreurs

- RAM juste au-dessus/au-dessous de chaque rung ;
- minimum RAM en hausse puis baisse ;
- disque suffisant, limite exacte, un octet insuffisant ;
- disque plein pendant spool/commit ;
- permission refusée, spool tronqué/corrompu ;
- arrêt au milieu de l'écriture et de la réduction ;
- aucun fallback sur exception scientifique ou programmation.

### Produit

- propagation config → worker → moteur ;
- AUTO sans nouveau bouton ;
- profils forcés respectés ;
- store fresh et reprise ;
- Windows spawn/NTFS ciblé ;
- Linux ext4 ciblé ;
- Classic/SDS inchangés par tests témoins courts.

Suite complète : nightly/release seulement, idéalement shardée. Pas de relance exhaustive à chaque correction.

## 9. Protocole de validation réelle du lendemain

Les gros runs ne démarrent que si M0–M5 sont verts et la revue indépendante ne contient aucun défaut bloquant.

### 9.1 Caldwell H128 — TinyDebian

1. vérifier corpus et espace disque ;
2. créer un dossier de sortie neuf ;
3. dry-plan avec halo 128 et paramètres scientifiques habituels ;
4. exiger un plan AUTO explicitant le chemin choisi ;
5. lancer avec une cellule en vol, checkpoints/reprise actifs ;
6. suivre RSS arbre, RAM disponible, scratch, débit et commits ;
7. conserver le rapport et les artefacts même en cas d'arrêt.

Acceptation immédiate :

- aucune baisse de halo ;
- plan admis là où 5.1.1 refusait ;
- si external : RAM reste sous l'enveloppe annoncée et première cellule committe ;
- reprise possible après arrêt contrôlé ;
- objectif final : run complet et sorties validées.

### 9.2 M33/S50 — Windows

1. auditer le mélange éventuel LP/IRCUT ; décision humaine si confirmé ;
2. utiliser l'inventaire exact et la configuration scientifique du run capturé, halo 8 sauf décision explicite contraire ;
3. nouveau dossier de sortie, jamais écrasement de `D:\M33\S50\out\ircut` ;
4. dry-plan réel avec RAM et espace D: frais ;
5. laisser AUTO choisir petite tuile RAM ou external ;
6. vérifier que la grille/halo/memberships ne changent pas pour contourner la capacité ;
7. instrumenter RAM, VRAM, cache/spool, I/O et commits.

Acceptation immédiate :

- plus de `NO executable plan` à la même capacité si RAM ou disque est réellement suffisant ;
- chemin choisi et budgets visibles ;
- pas d'OOM ni de fallback scientifique ;
- première cellule durable ;
- reprise vérifiée avant de laisser tourner plusieurs jours ;
- objectif final : run complet, mais durée non promise avant dry-plan.

## 10. Gates humains et sécurité

Demander Tristan avant :

- changement de science ou approximation hiérarchique ;
- suppression automatique d'un fichier LP/IRCUT ;
- dépendance majeure ;
- lancement des deux gros runs si le dry-plan annonce plusieurs To d'écriture ou plusieurs jours ;
- push, merge beta/main, tag, release ou déploiement.

Les tests et runs utilisent uniquement de nouveaux dossiers. Aucun dossier utilisateur existant n'est supprimé ou écrasé.

## 11. Définition d'acceptation de la mission

La mission technique est ACCEPT seulement si :

- les petites tuiles sont scientifiquement identiques et réellement propagées ;
- AUTO sélectionne external avant refus lorsque RAM insuffisante et disque suffisant ;
- le reducer external est exact pour winsor/kappa/mean/median ;
- RAM mesurée respecte l'enveloppe ;
- disque préflight/runtime est honnête ;
- interruption et reprise ne recalculent pas les cellules valides ;
- aucun halo, contributeur ou plancher scientifique n'est modifié ;
- Caldwell H128 et M33 disposent chacun d'un dry-plan réel admissible ;
- les deux lancements réels produisent les preuves immédiates prévues en section 9 ;
- les différences de dépôt sont comprises et les findings matériels résolus.

## 12. Risques ouverts

1. M33 peut être faisable en mémoire mais prohibitif en temps/I/O.
2. Le spool exact du winsor nécessite un layout disque efficace ; un memmap naïf peut provoquer du thrashing.
3. Les très petites tuiles augmentent fortement le coût fixe et peuvent devenir plus gourmandes.
4. Les sorties O(patch_area) restent en RAM dans la première version ; le dry-plan doit les refuser honnêtement si elles seules dépassent le budget.
5. L'assemblage final/finitions doit rester dans son propre budget ; ne pas découvrir une impossibilité après toutes les cellules.
6. Le mélange apparent LP dans IRCUT peut invalider le témoin scientifique indépendamment du moteur mémoire.
7. Une nuit d'implémentation ne garantit pas deux runs complets le lendemain : le gate autorise le lancement seulement si les preuves courtes sont réellement vertes.

## 13. Prochain pas après validation du plan

Lancer une seule mission d'implémentation bornée à **M0 + M1** : dry-plan joint et ladder CPU sous 32, sans encore câbler le reducer external. Junior vérifie ensuite les faits M33 et la parité scientifique, puis délègue M2. Ne pas confier M0–M5 comme un bloc vague ni lancer un grand run avant la tranche verticale external complète.