# ZeGrid — G0 : contrat opérationnel du crédit swap borné

- **Mission** : `ZM-GAUGE-SWAP-G0-20261115`
- **Phase** : G0 (contrat + baseline + replay ; aucun code produit modifié)
- **Date** : 2026-11-15 (Europe/Paris)
- **Base** : `/home/tristan/.openclaw/workspace/worktrees/zemosaic-adaptive-resources`,
  branche `feat/zegrid-adaptive-resources`, HEAD `e25fb5133768adf4eb26950c63f86b970909b370`
- **Plan de référence** : `docs/refactor/ZEGRID_GAUGE_BOUNDED_SWAP_PLAN_20261115.md` (v3, §11–13)
- **État de pilotage** : `docs/refactor/ZEGRID_GAUGE_SWAP_EXECUTION_STATE_20261115.md`
- **Revue amont** : `ZM-GAUGE-DYNAMIC-SWAP-PLAN-20261115.nono.r0.md`
- **Statut G0** : contrat **gelé pour implémentation** ; les points non prouvés physiquement
  sont explicitement listés (§14), pas contournés.

---

## 0. Résumé exécutif — ce que G0 ferme

1. **Formule** : crédit uniquement pour le groupe d'overlap singleton `gauge` **compute**,
   `S_credit = floor(9 * S_effectif / 10)`, sans plafond fixe en Gio. `S_effectif` est le
   minimum de bornes **indépendantes**, jamais une soustraction de `B_ram`.
2. **Swap disque par périphérique** : `S_disque = Σ free(devices non-zram, backing prouvé)`.
   `SwapFree` global n'est **jamais** utilisé comme capacité ; zram est exclu ; SwapCached
   n'est ajouté nulle part.
3. **Cgroups** : chaîne complète jusqu'à la racine, avec sémantique trinaire
   `non applicable / illimité / inconnu` ; un fait inconnu ⇒ crédit 0 avec raison.
   La limite `memory.swap.max` réduit `S_effectif` ; `memory.max` ne finance pas le crédit
   mais borne la partie non paginable.
4. **Commit** : opérationnel selon `vm.overcommit_memory`. Mode 0/1 = pas de gate dur ;
   mode 2 = `P_gauge_raw ≤ CommitLimit − Committed_AS`, sur les **nouvelles** allocations.
   Aucune soustraction de `B_ram` d'un headroom de sémantique différente.
5. **Non paginable** : garde RAM séparée, `G_np_new ≤ B_ram_eff` ; le swap ne compense
   jamais une page non paginable. Le socle résident existant est déjà exclu de
   `MemAvailable` et n'est pas re-soustrait.
6. **Refus NGC7023 reproduit exactement** : raw `2 203 281 820` ; +15 % `2 533 774 093` ;
   `B_plan 1 664 142 540` ; déficit `869 631 553` B ; `S_effectif` minimal
   `966 257 282` B. Le replay est outillé et vérifie l'arithmétique à chaque exécution.
7. **Gates cartographiés** : 32 points de câblage CPU/GPU/external, layout/preflight,
   jauge compute/restaurée/per-cell, configuration/GUI→worker, fermeture du crédit et
   réadmission RAM-only, hash/identité/reprise (§10).
8. **Isolation physique faisable** : un scope utilisateur systemd transient avec
   `MemoryMax` **et** `MemorySwapMax` fonctionne sur cet hôte, sans toucher au swap, au
   swappiness ni aux limites système. L'OOM éventuel est confiné au scope.
9. **Limite physique majeure** : zram a priorité 100 contre 10 pour `/swapfile`. Tant que
   zram a du libre, les pages neuves vont d'abord dans de la RAM compressée. Cet hôte ne
   peut donc **pas** prouver physiquement un crédit réellement adossé au disque sans
   modification autorisée de la configuration swap (hors périmètre). Le crédit reste une
   **autorisation**, pas une réservation ni une garantie anti-OOM.
10. **Aucun code produit modifié** ; aucun commit/push ; fichiers préexistants préservés.

---

## 1. Baseline vérifiée (lecture seule)

### 1.1 Git

| Élément | Valeur observée |
| --- | --- |
| Branche | `feat/zegrid-adaptive-resources` |
| HEAD | `e25fb5133768adf4eb26950c63f86b970909b370` |
| `git status --porcelain` | 1 modifié préexistant + 5 untracked préexistants (voir ci-dessous) |
| `git diff --stat` | `docs/refactor/ZEGRID_ADAPTIVE_RESOURCE_ENGINE_STATE_20261107.md \| 103 +++++` |

Fichiers préexistants à **préserver strictement** (ni nettoyés, ni commités) :

```text
 M docs/refactor/ZEGRID_ADAPTIVE_RESOURCE_ENGINE_STATE_20261107.md
?? docs/qualification/ZM-RAM-TILE-TO-EXTERNAL-QUAL-20261114.md
?? docs/qualification/zm-ram-tile-external-qual/
?? docs/refactor/ZEGRID_GAUGE_BOUNDED_SWAP_PLAN_20261115.md
?? docs/refactor/ZEGRID_GAUGE_SWAP_EXECUTION_STATE_20261115.md
?? docs/refactor/ZEGRID_THROUGHPUT_M33_FLIGHT_PLAN_20261108.md
```

### 1.2 Environnement d'exécution

| Élément | Valeur |
| --- | --- |
| OS | Linux 6.12.117+deb13-amd64 (Debian 13), x86_64 |
| Python | 3.13.5 |
| pytest | 8.3.5 |
| psutil | 7.0.0 |
| CuPy | **absent** (`ModuleNotFoundError`) |
| GPU physique | NVIDIA GeForce MX150, visible via `nvidia-smi -L` |
| Backend GPU moteur | **non qualifiable ici** (CuPy absent) |
| `systemd-run` | présent ; manager utilisateur `degraded` mais fonctionnel |

### 1.3 Faits d'hôte au 2026-11-15T00:29:57+01:00 (instantané ponctuel)

Ces valeurs sont **volatiles** et ne sont **pas** celles de l'incident du 2026-11-14T22:29Z.

| Grandeur | Valeur | Source |
| --- | --- | --- |
| MemTotal | 7 882 012 kB = 8 071 180 288 B | `/proc/meminfo` |
| MemAvailable | 877 656 kB = 898 719 744 B | `/proc/meminfo` |
| SwapTotal | 33 047 544 kB = 33 840 685 056 B | `/proc/meminfo` |
| SwapFree global | 26 231 920 kB = 26 861 486 080 B | `/proc/meminfo` |
| SwapCached | 992 344 kB = 1 016 160 256 B | `/proc/meminfo` |
| `/swapfile` | size 25 165 820 kB, used 1 370 044 kB, free 24 366 874 624 B | `/proc/swaps` |
| `/dev/zram0` | size 7 881 724 kB, used 5 445 580 kB, free 2 494 611 456 B | `/proc/swaps` |
| Priorités swap | zram **100**, swapfile **10** | `/proc/swaps` |
| zram algorithme | `zstd` | `/sys/block/zram0/comp_algorithm` |
| zswap | **désactivé** (`N`) | `/sys/module/zswap/parameters/enabled` |
| CommitLimit | 36 988 548 kB = 37 876 273 152 B | `/proc/meminfo` |
| Committed_AS | 19 448 896 kB = 19 915 669 504 B | `/proc/meminfo` |
| Headroom commit brut | 17 960 603 648 B | calcul |
| `vm.overcommit_memory` | **0** (heuristique) | `/proc/sys/vm/overcommit_memory` |
| `vm.overcommit_ratio` / `_kbytes` | 50 / 0 | `/proc/sys/vm/` |
| `vm.swappiness` | 60 (inchangé) | `/proc/sys/vm/swappiness` |
| Unevictable | 415 812 kB = 425 791 488 B | `/proc/meminfo` |
| Mlocked | 216 kB = 221 184 B ; `ulimit -l` = 8 192 kB | `/proc/meminfo`, shell |
| cgroup du process | `/user.slice/user-1000.slice/user@1000.service/app.slice/openclaw-gateway.service` | `/proc/self/cgroup` |
| `memory.max` / `memory.swap.max` | `max` / `max` à tous les niveaux non racine | `/sys/fs/cgroup/...` |
| Racine cgroup | fichiers `memory.*` **absents** ; `memory` dans `cgroup.controllers` | `/sys/fs/cgroup` |

**Vérification arithmétique du piège zram** :

```text
zram free  2 494 611 456 B
+ swapfile 24 366 874 624 B
= 26 861 486 080 B = SwapFree global observé
```

Donc 90 % de `SwapFree` global donnerait `24 175 337 472` B, alors que 90 % du disque
donne `21 930 187 161` B : **2 245 150 311 B (≈ 2,09 Gio) de crédit fantôme** si zram
était compté. C'est exactement le double comptage RAM dénoncé par Nono (M2).

### 1.4 Suites courtes exécutées (non destructives)

| Commande | Résultat observé |
| --- | --- |
| `python3 -m pytest tests/test_zegrid_r41_ram_contract.py -q` | **51 passed** en 2,91 s |
| `python3 -m pytest tests/test_zegrid_restored_gauge_admission.py -q` | **8 passed, 3 failed** en 23,10 s |

Les 3 échecs ont la **même signature observée** : le test force un backend GPU, CuPy est
absent, le moteur retombe sur CPU, puis refuse le commit durable car l'identité gelée dit
`backend='gpu'` (`StoreFormatError: cell declares backend_used='cpu' but the frozen run
identity contract is backend='gpu'`). C'est une **limite d'environnement préexistante**
(pas de CuPy), sans rapport avec cette mission ; elle est documentée, pas masquée.

Aucun benchmark long, aucun run de pression, aucun run moteur n'a été lancé en G0.

---

## 2. Portée, non-objectifs, invariants

### 2.1 Dans le périmètre G0

- Définir un contrat **opérationnel implémentable** pour un crédit swap Linux, opt-in.
- Fermer les bases : RAM, non paginable, commit, swap disque par device, cgroups/ancêtres,
  zram/zswap, sémantique des faits inconnus.
- Reproduire le refus historique en lecture seule.
- Cartographier **tous** les gates à câbler et découper G1/G2.
- Figer les oracles bit-exacts et les seuils de coût/latence **avant** mesures.
- Analyser la faisabilité d'une qualification physique isolée et bornée.

### 2.2 Hors périmètre (non-goals explicites)

- Aucun code produit modifié, aucun noyau mathématique, aucun chunking.
- Aucune extension du crédit au GPU external ni aux jauges `per_cell`.
- Aucun changement science/backend/dtype/ordre des réductions.
- Aucun changement M33, Windows, runtime, ZeAlfie, données originales.
- Aucun commit, push, merge, tag, déploiement, changement de configuration système.
- Aucune promesse de débit ni d'absence d'OOM.
- Aucun stress global non isolé sur l'hôte.

### 2.3 Invariants

1. `gauge_memory_policy ∈ {ram_only, bounded_swap}`, défaut **ram_only**.
2. Le crédit ne sert **jamais** à augmenter la concurrence : jauge secours à **1 worker**.
3. Le stack, le cache, l'assemblage, l'export, le staging GPU et le contrôleur RUN restent
   **RAM-only** avec leurs budgets actuels.
4. `host.budget_bytes` reste `floor(factor*raw/den)` = budget **RAM** ; il n'est jamais gonflé.
5. Aucune identité scientifique ni aucun hash de plan legacy ne change quand la politique
   est absente ou `ram_only`.
6. Aucun fait inconnu n'est converti en « illimité » ; le crédit est 0 avec raison.
7. Une incapacité à prouver une condition requise interdit le **secours**, pas un run
   RAM-only déjà admissible.

---

## 3. Notations et frontière « socle existant » / « nouvelles allocations »

| Symbole | Définition | Base / source | Ne pas faire |
| --- | --- | --- | --- |
| `A` | RAM disponible **nette** | `MemAvailable` (via `available_memory_bytes()`, psutil → `/proc/meminfo` sur Linux) | Ne pas ajouter le cache : `MemAvailable` l'inclut déjà. Ne pas ajouter `SwapCached`. |
| `B_ram` | `floor(4*A/5)` | `ram_contract.plan_budget_bytes` | Ne pas remplacer `A` par `MemTotal`. |
| `B_admit` | `floor(9*A/10)` | `ram_contract.admit_budget_bytes` | Ne pas réutiliser pour le crédit. |
| `P_gauge_raw` | pic brut du groupe jauge, `footprint × gauge_workers` (ou résidence restaurée) | `_r55_active_facts` / modèle de jauge | Ne pas retirer le crédit du pic pour fabriquer une empreinte plus faible. |
| `P_gauge_margined` | `ceil(P_gauge_raw*115/100)` | marge unique du groupe | Ne pas appliquer la marge une seconde fois. |
| `S_disque` | somme des libres des devices de swap **disque** | `/proc/swaps` + backing | Ne pas utiliser `SwapFree` global. |
| `S_cgroup_swap` | min des headrooms `memory.swap.max − memory.swap.current` sur la chaîne | `/sys/fs/cgroup/...` | Ne pas traiter un fichier absent comme infini. |
| `S_effectif` | `min(S_disque, S_cgroup_swap)` si la chaîne est prouvée | dérivé | Ne pas soustraire le non-paginable du swap. |
| `S_credit` | `floor(9*S_effectif/10)` sous conditions | formule gelée | Aucun plafond fixe en Gio. |
| `G_np_new` | bytes **non paginables nouveaux** de la phase jauge | borne conservative (`PROCESS_BASELINE_BYTES = 340 MiB`) + pinned/mlock explicites (0 en v1) | Ne pas confondre avec `Unevictable` système, déjà reflété dans `A`. |
| `C_new` | bytes de **commit virtuel nouveau** de la phase jauge | `P_gauge_raw` (allocations privées anonymes) | Ne pas faire `CommitLimit − Committed_AS − B_ram`. |

**Socle existant** : le processus parent, les bibliothèques, le cache récupérable et les
pages non paginables du système sont **déjà exclus** de `A` (observation nette). Ils sont
donc facturés **zéro fois de plus** dans `B_ram`, `S_credit` et `C_new`. Le seul socle
nouveau facturé est celui des **worker(s) de jauge** (baseline 340 MiB, comptée une fois
dans le pic de phase par le modèle existant).

**Nouvelles allocations** : `P_gauge_raw` est la seule grandeur financée par
`B_ram + S_credit`. `B_ram` finance indifféremment du résident ou du paginable ;
`S_credit` ne finance que du **paginable anonyme**. `G_np_new` doit tenir dans `B_ram`
(et dans le headroom RAM du cgroup s'il est limité), sans quoi le crédit est refusé.

---

## 4. Formule gelée (v1, Linux)

### 4.1 Budgets RAM (inchangés)

```text
A          = MemAvailable
B_ram      = floor(4*A/5)          # plan
B_admit    = floor(9*A/10)         # admission / launch / RUN
```

Aucune de ces trois valeurs n'est modifiée par la politique. Aucun cache n'est ajouté.

### 4.2 Swap disque par périphérique

```text
pour chaque device d dans /proc/swaps :
    free(d) = max(0, Size(d) - Used(d))            # en octets
    si type(d) == file :
        exiger stat(d).st_blocks * st_blksize >= Size(d)
        sinon : device exclu (backing non prouvé)
    si nom(d) correspond à zram[0-9]+ (ou sysfs block zram) :
        device exclu                              # RAM compressée, pas du disque
    si type(d) ∉ {file, partition} :
        device exclu                              # type non qualifié
S_disque = Σ free(d) pour les devices éligibles
```

Règles :

- La somme est légitime : les devices de swap s'additionnent (le noyau choisit par
  priorité, mais la capacité libre totale reste la somme des slots libres).
- Un swapfile est déjà alloué au `swapon` ; la vérification `st_blocks` empêche de
  créditer un fichier creux non adossé.
- Les priorités hétérogènes sont une question d'**ordre d'usage**, pas de capacité ;
  elles sont journalisées et traitées en §7.
- `SwapFree`, `SwapTotal`, `psutil.swap_memory().free` : **interdits comme preuve**.

### 4.3 Swap cgroup (chaîne complète)

```text
chaîne = [root, ..., cgroup du process]          # /proc/self/cgroup, entrée 0::
pour chaque niveau :
    si niveau == root et memory.max absent       -> NON APPLICABLE (racine non limitable)
    si contrôleur memory absent du niveau        -> NON APPLICABLE (pas de comptabilité ici)
    si fichier présent et valeur == "max"        -> ILLIMITÉ
    si fichier présent et entier                -> LIMITÉ, headroom = max(0, max - current)
    si fichier attendu mais absent               -> INCONNU
S_cgroup_swap = min(headrooms LIMITÉS)           # IGNORE NA et ILLIMITÉ
               = INCONNU si un niveau est INCONNU
               = ILLIMITÉ si aucun niveau LIMITÉ
```

- `memory.swap.max`/`memory.swap.current` sont lus **ensemble** : un `max` sans `current`
  exploitable est **inconnu**.
- Les niveaux `NA` (racine, contrôleur désactivé) ne contraignent pas et ne rendent pas
  la chaîne inconnue.
- cgroup v1 : si détecté, v1 n'est **pas** qualifié en G0. Un calcul `memsw.limit −
  memory.limit` n'est accepté que si `memsw` et `memory` sont tous deux présents et
  cohérents ; sinon `INCONNU` ⇒ crédit 0.
- `memory.max` ne réduit pas `S_effectif` (ce n'est pas du swap) ; il intervient dans la
  garde non paginable (§4.5).

### 4.4 Crédit

```text
si S_disque == 0                         : crédit 0, raison no_eligible_disk_swap
si S_cgroup_swap == INCONNU              : crédit 0, raison cgroup_swap_unknown
S_effectif = S_disque                    si S_cgroup_swap ∈ {NA, ILLIMITÉ}
           = min(S_disque, headroom)     si S_cgroup_swap == LIMITÉ
si S_effectif < S_MIN_USABLE             : crédit 0, raison effective_swap_below_minimum
S_credit = floor(9 * S_effectif / 10)
```

**Paramètres gelés** :

| Paramètre | Valeur v1 | Justification |
| --- | --- | --- |
| `CREDIT_NUM/DEN` | `9/10` | direction utilisateur, arithmétique entière exacte |
| plafond fixe | **aucun** | exigence explicite ; le crédit peut dépasser 1 Gio |
| `R_MIN` | **0** | la valeur chiffrée L1 de Nono n'est pas validée ; les 10 % retenus ne sont **pas** une réservation OS |
| `S_MIN_USABLE` | **1 GiB** | porte d'éligibilité, pas un plafond : un swap de 512 Mio ne peut jamais activer le secours (réponse directe à L1) |
| worker de secours | **1** | le swap n'augmente pas la concurrence ; plusieurs workers multiplieraient l'ensemble paginable |

Le crédit n'est **pas** une nouvelle valeur de RAM disponible. C'est une autorisation
d'excéder `B_ram` pour **un seul groupe**, pendant **une seule phase**.

### 4.5 Garde RAM résidente / non paginable

```text
G_np_new = PROCESS_BASELINE_BYTES (340 MiB)
         + pinned_staging_bytes (0 en v1 CPU)
         + mlock_bytes (0 : aucun mlock ajouté)
B_ram_eff = min(B_ram, headroom_RAM_cgroup)   si memory.max est LIMITÉ
          = B_ram                             sinon
condition : G_np_new <= B_ram_eff
```

- `PROCESS_BASELINE_BYTES` est une **borne supérieure conservative** (interpréteur, code,
  piles, métadonnées d'allocateur), pas une mesure du non-paginable réel. Elle est déjà
  comptée une fois dans `P_gauge_raw` ; on ne la re-soustrait pas de `S_credit`.
- `min(B_ram, headroom)` combine deux **plafonds indépendants** ; ce n'est jamais une
  soustraction de l'un par l'autre.
- `Unevictable`/`Mlocked` système ne sont **pas** re-soustraits : ils sont déjà absents de
  `A` (observation nette).
- Si `G_np_new > B_ram_eff` : crédit refusé, raison
  `b_ram_below_non_pageable_new`. Aucun secours swap.

### 4.6 Garde commit (selon OS et mode)

| `vm.overcommit_memory` | Sémantique | Gate G0 |
| --- | --- | --- |
| 0 | heuristique | **non applicable** ; le noyau peut encore refuser une allocation absurde, mais il n'y a pas de plafond de commit par allocation |
| 1 | always overcommit | **non applicable** |
| 2 | strict | `C_new ≤ max(0, CommitLimit − Committed_AS)` |
| illisible/inconnu | — | **inconnu** ⇒ crédit 0, raison `overcommit_mode_unknown` |

- `C_new = P_gauge_raw` : les allocations neuves de la jauge sont des mappings privés
  anonymes. On ne déduit **pas** `B_ram` du headroom : `B_ram` est un budget de RAM
  physique, `Committed_AS` compte du virtuel déjà engagé ; les soustraire mélangerait
  deux sémantiques (correction Junior à M1).
- `Committed_AS` inclut déjà le processus courant : pas de double comptage.
- Le gate commit est une condition **nécessaire distincte** du gate swap. Satisfaire les
  deux n'est pas redondant : l'un borne le virtuel, l'autre la capacité physique.
- `CommitLimit` est lu, jamais recalculé à la main quand il est présent. Si le mode est 2
  et qu'un champ manque ⇒ inconnu (pas d'inférence depuis `overcommit_ratio`).

### 4.7 Groupes d'overlap et marge

Le modèle existant agrège par **groupes d'overlap** : somme intra-groupe, maximum
inter-groupes, **une seule marge** au boundary (`resource_envelope._aggregate` /
`aggregate_host_peak_bytes`). G0 conserve cette topologie et introduit un **plafond par
groupe** :

```text
pour chaque groupe g :
    raw_g = Σ peak(p) pour p ∈ g
    ceiling_g = B_ram                si g n'est pas le groupe jauge compute éligible
              = B_ram + S_credit     si g == {gauge} et toutes les conditions du crédit
    admis(g) ssi ceil(raw_g * 115/100) <= ceiling_g
admission globale ssi admis(g) pour TOUS les groupes
```

Éligibilité du groupe jauge (toutes obligatoires) :

1. `policy == bounded_swap` ;
2. groupe exactement `("gauge",)` — aucun overlap avec stack/cache/assembly/export ;
3. jauge **compute** fraîche (`gauge_restored == False`, pas d'artefact importé) ;
4. `gauge_workers == 1` ;
5. plateforme Linux qualifiée ;
6. gardes §4.5 et §4.6 satisfaites ;
7. capacité prouvée (`S_effectif` non inconnu, ≥ `S_MIN_USABLE`).

Tout groupe non éligible reste évalué contre `B_ram`. Le crédit ne peut **jamais** sauver
un stack, un cache, un assemblage ou un export. Un groupe jauge qui chevauche une phase
non éligible est refusé (`gauge_group_not_singleton`), pas subventionné.

**Diagnostics** : le pic agrégé vrai (somme/max des pics **bruts**) reste journalisé.
On ne soustrait jamais `S_credit` du pic pour publier une empreinte RAM plus faible.

### 4.8 Décision complète (pseudocode de référence)

```text
function decide(policy, facts, groups):
    if policy != bounded_swap: return ram_only_decision(groups, B_ram)   # byte-identique

    cap = collect_capacity()                       # §6
    credit = evaluate_credit(cap, P_gauge_raw)     # §4.2–4.6
    if credit.status != OK:
        # le RAM-only reste possible si et seulement si tous les groupes tiennent B_ram
        return ram_only_decision_or_typed_refusal(groups, B_ram, credit.reason)

    for g in groups:
        ceiling = B_ram + credit.bytes if g == {gauge} else B_ram
        if ceil_margin(raw(g)) > ceiling:
            return typed_refusal(g, ceiling)       # aucun fallback implicite
    return bounded_swap_decision(groups, credit)
```

`ram_only_decision` doit être **exactement** la décision actuelle (mêmes comparaisons,
mêmes hashes, mêmes reason codes).

---

## 5. Sémantique `non applicable` / `illimité` / `inconnu`

| Fait | `non applicable` | `illimité` | `inconnu` | Effet sur le crédit |
| --- | --- | --- | --- | --- |
| `memory.max` racine absent | oui (racine non limitable) | — | — | aucun |
| contrôleur `memory` absent d'un niveau | oui | — | — | aucun (le parent s'applique) |
| `memory.max == max` | — | oui | — | aucun (pas de plafond RAM cgroup) |
| `memory.max` entier | — | — | — | headroom RAM pour `G_np_new` |
| `memory.swap.max` absent sur un niveau avec contrôleur | — | — | **oui** | **crédit 0** |
| `memory.swap.max == max` | — | oui | — | aucun plafond swap cgroup |
| `memory.swap.max` entier | — | — | — | réduit `S_effectif` |
| `overcommit_memory` 0/1 | oui | — | — | aucun gate commit |
| `overcommit_memory` 2 + champs présents | — | — | — | gate commit |
| `overcommit_memory` 2 + champ manquant | — | — | **oui** | **crédit 0** |
| device zram | oui (exclu) | — | — | aucun |
| swapfile non alloué | oui (exclu) | — | — | aucun |
| type de swap inconnu | — | — | **oui** (device exclu) | aucun pour ce device |
| `/proc/swaps` illisible | — | — | **oui** | **crédit 0** |

Principe : **un seul `inconnu` requis ⇒ crédit 0**, jamais « illimité par défaut ».
Un `non applicable` n'est pas un `inconnu` : il signifie qu'aucune contrainte de ce type
n'existe à ce niveau.

---

## 6. Sources exactes et protocole de collecte

### 6.1 Sources autoritaires

| Fait | Fichier/API | Champs | Remarque |
| --- | --- | --- | --- |
| `A` | `available_memory_bytes()` (psutil `virtual_memory().available`) | — | sur Linux = `MemAvailable` ; **source unique** pour `A` |
| RAM détaillée | `/proc/meminfo` | `MemTotal`, `MemAvailable`, `SwapTotal`, `SwapFree`, `SwapCached`, `CommitLimit`, `Committed_AS`, `Unevictable`, `Mlocked` | unités kB → ×1024 |
| Swap devices | `/proc/swaps` | Filename, Type, Size, Used, Priority | Type zram = `partition` : **ne pas** se fier au type |
| Backing swapfile | `stat()` | `st_blocks`, `st_blksize` | exclure si `st_blocks*st_blksize < Size` |
| zram | `/sys/block/zram*/`, nom du device | — | exclu du crédit |
| zswap | `/sys/module/zswap/parameters/enabled` | `Y/N` | jamais additionné ; voir §7 |
| Overcommit | `/proc/sys/vm/overcommit_memory`, `overcommit_ratio`, `overcommit_kbytes` | — | mode 0/1/2 |
| cgroup | `/proc/self/cgroup` (ligne `0::`), `/sys/fs/cgroup/...` | `cgroup.controllers`, `memory.max`, `memory.current`, `memory.swap.max`, `memory.swap.current` | chaîne complète, racine incluse |
| RUN diagnostics | `runtime_control.collect_run_facts` | cgroup courant, RSS, IO | **ne décide pas** du crédit |

### 6.2 Fraîcheur et atomicité

- Un **snapshot unique** est collecté par décision, avec un timestamp monotone.
- Les faits sont relus avant : plan (4/5), admission (9/10), launch (9/10), **juste avant
  la jauge**, et à chaque frontière de vague RUN (90 % RAM-only).
- Le crédit est recalculé à chaque snapshot ; le crédit effectif d'une exécution est le
  **minimum** des crédits observés entre l'ouverture et la fermeture de la phase.
- Une erreur de lecture d'un fait **requis** ⇒ `inconnu` ⇒ crédit 0 ; jamais une valeur
  par défaut.
- Les observations volatiles ne rejoignent **pas** l'identité scientifique durable.

### 6.3 Pièges explicitement fermés

- `MemAvailable` inclut déjà la part récupérable du cache : ne rien ajouter.
- `SwapCached` est de la RAM tenant une copie de pages swappées : ne l'ajouter ni à `A`
  ni à `SwapFree`.
- `SwapFree` global = disque + zram ; ne jamais l'utiliser.
- `psutil.swap_memory().free` seul n'est pas une preuve (zram, devices, cgroup).
- Un faux `MemAvailable` injecté prouve la **logique**, jamais le swap physique.

---

## 7. zram, zswap et priorité d'usage — analyse

### 7.1 zram

- zram est un device de swap **adossé à la RAM** : ses slots libres ne sont pas de la
  mémoire supplémentaire. Il est exclu de `S_disque`, donc de `S_credit`.
- **Risque d'ordre d'usage** : sur cet hôte, zram a priorité 100 et `/swapfile` priorité
  10. Tant que zram a du libre (2,49 GB à la baseline), les pages swappées par le run
  iront **d'abord** dans zram, c'est-à-dire en RAM compressée. Le crédit « disque » ne
  sera consommé qu'après saturation de zram.
- Conséquence honnête : le crédit reste une **borne de capacité de swap** (les slots
  disque existent), mais il ne garantit pas que les pages excédentaires quittent la RAM.
  Le run peut donc subir une pression RAM réelle avant d'utiliser le disque.
- La qualification G3 doit mesurer séparément les deltas `used` de `/swapfile` et de
  `/dev/zram0` (et `memory.swap.current` du cgroup). Si le crédit est absorbé par zram,
  la preuve physique porte sur zram, **pas** sur le disque.

### 7.2 zswap

- zswap n'est **pas** un device de swap : c'est un cache compressé en RAM devant le
  swap. Il n'apparaît pas dans `/proc/swaps` et n'ajoute aucune capacité.
- Il ne doit jamais être additionné à `S_disque` ni à `A`.
- S'il est activé, son pool consomme de la RAM et peut réduire `A` plus vite qu'un swap
  disque simple. G3 doit le journaliser (`/sys/module/zswap/parameters/enabled`,
  `memory.zswap.current` si présent) ; aucune formule ne le crédite.
- Sur cet hôte, zswap est **désactivé** à la baseline.

### 7.3 SwapCached

- `SwapCached = 992 344 kB` à la baseline : de la RAM qui détient une copie de pages
  déjà swappées. Ce n'est ni du swap libre, ni de la RAM disponible « en plus ».
- Interdiction : `A + SwapCached`, `SwapFree + SwapCached`, `S_disque + SwapCached`.

---

## 8. Reproduction exacte du refus historique NGC7023

### 8.1 Source et statut

- Source : `/home/tristan/NGC7023/out/instrum/worker_crash_breadcrumbs.jsonl`,
  événements `GUI_LOG`/`WORKER_EXCEPTION` du **2026-11-14T22:29:00Z**, dernière entrée
  `R55_REFUSE_HOST_RAM` (lignes 161–163 du fichier au moment du replay).
- Lecture **strictement seule** : aucun fichier du run n'a été modifié.
- Les mesures d'hôte du §1.3 sont **d'aujourd'hui** et ne sont **pas** celles de
  l'incident. Le replay ne mélange jamais les deux : il ne recalcule que l'arithmétique
  du message historique.

### 8.2 Arithmétique vérifiée

```text
raw                 = 2 203 281 820 B
marge 15/100        = ceil(2 203 281 820 * 115 / 100)
                    = 2 533 774 093 B          (15 % = 330 492 273 B)
B_plan              = 1 664 142 540 B
déficit             = 2 533 774 093 - 1 664 142 540
                    = 869 631 553 B
S_effectif minimal  = ceil(869 631 553 * 10 / 9)
                    = 966 257 282 B
```

Le message historique est :

```text
aggregate host peak 2533774093 B (raw 2203281820 B + margin 15/100)
> budget 1664142540 B
```

Les deux admissions, RAM **et** external, rapportent le même terme de jauge ; le spool a
été évalué puis refusé, pas « sélectionné ».

### 8.3 Origine de `B_plan` (inférence bornée, pas mesure actuelle)

`B_plan = floor(4*A/5)` implique `A ∈ {2 080 178 175, 2 080 178 176}` B :

```text
4 * 2 080 178 175 = 8 320 712 700 ; /5 = 1 664 142 540
4 * 2 080 178 176 = 8 320 712 704 ; /5 = 1 664 142 540.8 -> floor 1 664 142 540
```

Cette valeur est **déduite du budget historique**, pas lue comme un `MemAvailable`
d'aujourd'hui. Elle est étiquetée comme telle dans l'outil de replay.

### 8.4 Replay outillé

```bash
python3 tools/zegrid_gauge_swap/g0_capacity.py --replay \
  /home/tristan/NGC7023/out/instrum/worker_crash_breadcrumbs.jsonl
```

Sortie observée (extrait) :

```json
{
  "raw_bytes": 2203281820,
  "peak_bytes": 2533774093,
  "budget_bytes": 1664142540,
  "deficit_bytes": 869631553,
  "s_effectif_needed_bytes": 966257282,
  "checks": {
    "peak_matches_margin": true,
    "deficit_matches": true,
    "s_needed_matches": true
  },
  "status": "REPRODUCED"
}
```

Ce que cela prouve : l'arithmétique du refus et le besoin de crédit.
Ce que cela **ne prouve pas** : qu'un run réel avec ce crédit aurait abouti, ni que le
swap physique aurait été utilisé, ni l'absence d'OOM ou de thrashing.

### 8.5 Couverture live (informative, non contractuelle)

À la baseline, `S_disque = 24 366 874 624 B` et `S_credit = 21 930 187 161 B` : le
déficit de 869 631 553 B serait couvert **sur le papier**. Cette phrase ne doit jamais
être citée comme preuve de faisabilité physique (zram, concurrence, commit, thrashing).

---

## 9. Portée exacte du crédit : ce qui est financé, ce qui ne l'est pas

| Élément | Financé par | Commentaire |
| --- | --- | --- |
| Socle résident existant | déjà exclu de `A` | aucune re-soustraction |
| Workers de jauge (`P_gauge_raw`) | `B_ram + S_credit` (groupe jauge uniquement) | 1 worker en secours |
| Partie non paginable de la jauge (`G_np_new`) | `B_ram_eff` uniquement | jamais `S_credit` |
| Cache build, stack CPU/GPU, assembly, export | `B_ram` | aucun crédit |
| RUN checkpoints (90 %) | `B_admit` RAM-only | crédit fermé |
| Staging CUDA / pinned | `B_ram`/VRAM | 0 crédit ; GPU exclu v1 |
| Jauge `per_cell` | budget existant | aucun crédit (limite documentée) |
| Jauge restaurée/importée | résidence tarifée | aucun crédit compute |
| Commit virtuel | `C_new` vs headroom | gate séparé, mode 2 seulement |

---

## 10. Inventaire sourcé de TOUS les gates à câbler (G1/G2)

Légende « Lot » : **G1a** capacité pure, **G1b** admission par groupes, **G2a** câblage,
**G2b** config/UX/Stop, **G3** qualification.

### 10.1 Préflight, layout, bundle

| ID | Point de code | Comportement actuel observé | Changement requis | Lot | Test |
| --- | --- | --- | --- | --- | --- |
| P1 | `_build_execution_bundle` `mode.py:8722` ; appel `_plan_bundle_auto` `mode.py:4773` | construit l'enveloppe et `B_plan=4/5`, choisit workers jauge/cache sous marge | **inchangé** : aucun crédit ici ; le stack reste RAM-only | G2a | policy ON ne modifie pas le bundle hors jauge |
| P2 | `_layout_joint_plan_probe` `mode.py:10624` | balaie les layouts et compare le pic margé à `B_plan` | évaluateur par groupes ; crédit **seulement** pour le groupe jauge ; les autres groupes doivent tenir `B_ram` avant d'accepter un layout | G2a | layout identique policy OFF ; layout plus fin refusé si stack dépasse |
| P3 | `_choose_layout_mode_aware` `mode.py:2443` ; `_scan_layout_mode_aware` `1961` ; `_select_cpu_streaming_tile` `2024` | floors RAM et tuile sous `B_plan` | **inchangé** (RAM-only) ; ne jamais exposer `S_credit` | G2a | OFF/ON sans pression : mêmes tuiles |
| P4 | `_select_external_fallback` `mode.py:8028` ; `_external_phase_budget_block` `7988` | pricing du fallback external sous `B_plan`/90 % | **inchangé** ; external n'est pas un secours de jauge | G2a | external refusé avec crédit énorme |

### 10.2 Jauge compute, CPU/GPU/external

| ID | Point de code | Comportement actuel observé | Changement requis | Lot | Test |
| --- | --- | --- | --- | --- | --- |
| C1 | `_r55_active_facts` `mode.py:10882` ; sizing `10974–11022` | `gauge_workers`/`cache_workers` cohérents sous `B_plan`/marge ; jauge restaurée = résidence | crédit ⇒ **jauge forcée à 1** ; cache workers restent calculés sous `B_ram` ; restauré ⇒ aucun crédit | G2a | `gauge_workers==1` ; cache inchangé ; restauré sans crédit |
| C2 | `_r55_active_decide` `mode.py:11664` | `plan_joint` CPU inmem+stream, marge 15 % vs `B_plan` | évaluateur par groupes ; crédit éligible uniquement `{gauge}` ; tous les autres groupes `≤ B_ram` | G1b/G2a | matrice seuil exact/+1 B ; stack jamais crédité |
| C3 | `_r55_active_external_decide` `mode.py:11514` | external single-owner, même `B_plan`/marge | même évaluateur ; le stack external reste `≤ B_ram` | G2a | external + crédit → refus honnête |
| C4 | `_r55_active_gpu_decide` `mode.py:12002` | décision GPU host+VRAM | **aucun crédit** v1 ; pas de fuite de policy dans les contextes GPU | G2a | GPU non qualifié → refus explicite |
| C5 | `_r55_active_admit` `mode.py:13578` | plan-fit 4/5, admission 9/10, shrink/replan | ajouter le crédit **au plafond du groupe jauge** dans les mêmes comparaisons ; ne pas créditer les shrink steps de stack | G2a | admission jauge OK ; stack échoue encore |
| C6 | `_r55_active_admit_external` `mode.py:13440` | admission external, shrink same-class | **aucun crédit** ; conserver le refus honnête | G2a | crédit ne change pas l'external |
| C7 | `_r55_active_gpu_admit` `mode.py:12792` | admission GPU | **aucun crédit** | G2a | — |
| C8 | `_r55_active_revalidate_launch` `mode.py:13845` | revalidation 9/10 après jauge, avant stack | crédit **fermé** ; RAM-only strict | G2a/G2b | RAM insuffisante après jauge ⇒ refus, malgré crédit |
| C9 | `_r55_run_checkpoint` `mode.py:15193` ; `runtime_control.run_budget_bytes` `runtime_control.py:107` | checkpoint de vague à 90 % RAM | **aucun crédit** ; journaliser `credit=closed` | G2a | vague refuse avec crédit disponible |
| C10 | `_bounded_assembly_preflight` `mode.py:16212` ; `_bounded_assembly_available_bytes` `16201` | préflight assemblage 90 % RAM | **aucun crédit** | G2a | assemblage refuse avec crédit |
| C11 | jauge `per_cell` : `_prepare_stack_cell` `mode.py:6287–6334` ; `cell_gauge_mode` `5070`, `6146`, `17091` | jauge par cellule dans le stack | **aucun crédit** ; diagnostic/refus si policy bounded_swap + per_cell compute | G2a | crédit jamais transmis aux contextes per-cell |
| C12 | `compute_global_gauge` `mode.py:18279` ; `workers = _final.gauge_workers` `18383` | exécute la jauge réelle | ouvrir le crédit juste avant, forcer 1 worker, fermer en `finally`, émettre `gauge_credit_closed` | G2a/G2b | événement présent ; aucun crédit après retour |
| C13 | `build_plan_set` `scheduler.py:2284` ; `build_joint_plan_set` `scheduler.py:2673` | comparaison finale `peak > context.budget_bytes` (marge déjà appliquée) | **validateur final** : recevoir des plafonds par groupe (ou une décision déjà évaluée) ; ne jamais laisser un `B_plan` unique écraser le crédit, ni un crédit contourner la validation finale | G1b/G2a | seuil exact/+1 B sur le chemin final ; aucun plan admis sans passage par l'évaluateur |

### 10.3 Jauge restaurée / importée / reprise

| ID | Point de code | Comportement actuel observé | Changement requis | Lot | Test |
| --- | --- | --- | --- | --- | --- |
| R1 | `_r55_active_facts` branche restaurée `mode.py:10975–11022` ; `_restored_gauge_resident_bytes` `10276` | résidence tarifée, compute non facturé | **aucun crédit** ; conserver la tarification résidente | G2a | `test_zegrid_restored_gauge_admission` (hors 3 échecs GPU) |
| R2 | `_r55_execution_contract_from_bundle` `mode.py:15774` ; `_r55_build_run_store_identity` `15889` ; `compute_plan_hash` `resource_envelope.py:882` | hash des champs immuables ; identité run/store | ajouter la policy au **digest d'exécution versionné**, pas à l'identité scientifique ; `host.budget_bytes` inchangé ; policy absente ⇒ hash legacy identique | G1b/G2a | golden hashes OFF ; science identique ON |
| R3 | appels de décision (`17585`, `17961`, `23650`, etc.) | restauré/importé ⇒ pas de compute | aucun crédit ; un changement de policy entre attempts ne doit **pas** invalider les cellules durables | G2a | reprise avec policy différente : cellules réutilisées |

### 10.4 Configuration, GUI → worker, snapshot

| ID | Point de code | Comportement actuel observé | Changement requis | Lot | Test |
| --- | --- | --- | --- | --- | --- |
| U1 | `zemosaic_config.py` defaults (~`164`) | clés additives avec défauts | `zegrid_gauge_memory_policy: "ram_only"` + validation enum | G2b | round-trip ; valeur inconnue rejetée par le moteur |
| U2 | `zemosaic_gui_qt.py` pattern `_config_fields` `2359` ; handlers `3438–3475` | checkbox + explication pour gauge import | checkbox « Autoriser un secours swap limité pour la jauge ; peut ralentir ; aucun crédit pour le stack » ; masquée/désactivée hors Linux avec raison | G2b | `test_zegrid_gauge_import_gui` voisin ; test UI policy |
| U3 | `_snapshot_for_run` `zemosaic_gui_qt.py:10227` ; `_build_worker_invocation` `10109` ; `run_config_snapshot.json` | snapshot effectif du run | transporter la policy effective ; moteur autoritaire ; clé absente = `ram_only` | G2b | snapshot contient la policy ; worker la reçoit ; OFF si absente |

### 10.5 Fermeture du crédit et réadmission RAM-only

| ID | Point de code | Comportement actuel observé | Changement requis | Lot | Test |
| --- | --- | --- | --- | --- | --- |
| X1 | sortie de `compute_global_gauge` `18279` ; admission suivante `_admit` `18313` / `_r55_active_revalidate_launch` `18570` | admission fraîche 9/10 après jauge | journal `gauge_credit_closed` avec faits observés ; **zéro** crédit transmis ; refus typé si stack ne tient pas | G2a/G2b | si stack échoue, raison stack, jamais « jauge » |
| X2 | workers/libération temporaires de jauge | cache jauge supprimé après usage | terminer les workers, libérer les temporaires, fermer le crédit **avant** le cache/stack | G2b | aucun crédit actif pendant le stack |
| X3 | contrôleur RUN / prefetch / réservations cellule / staging CUDA | budgets RAM/VRAM existants | ne jamais propager la policy ni le crédit | G2a | instrumentation : credit absent de ces contextes |

### 10.6 Observabilité, erreurs, Stop/Resume

| ID | Point de code | Comportement actuel observé | Changement requis | Lot | Test |
| --- | --- | --- | --- | --- | --- |
| O1 | `_r55_admission_dict` `13955` ; `_ram_contract_diagnostics` `10095` ; `_build_execution_bundle_infeasible_message` `10087` | diagnostics d'admission et messages | champs policy, sources, `A`, `B_ram`, `S_effectif`, `S_credit`, `credit_needed`, statuts, raisons, workers ; jamais « swap utilisé » | G2a | schéma JSON stable + test |
| O2 | `_emit` / `zegrid_run.log` / manifest | logs existants | start/admit/end + `gauge_credit_closed` ; compteurs observés séparés (swap-in/out, défauts majeurs, PSI, RSS) avec portée | G2b | événements présents, valeurs non inventées |
| O3 | `runtime_control.read_cgroup_memory` `runtime_control.py:318` | diagnostics cgroup RAM | fournisseur **séparé** pour swap/cgroup ; ne pas changer les décisions RUN | G1a | faits absents ⇒ `None` + raison |
| S1 | `_gauge_rep` `18238` ; boucle de jauge | progression, Stop coopératif | Stop pendant jauge : aucun artefact partiel annoncé complet ; crédit non persisté ; recalcul possible | G2b | test Stop mi-jauge |
| S2 | reprise durable `run_store` / commits | cellules durables immuables | la policy n'est pas une identité de cellule ; reprise sans perte | G2b | Stop/Resume : zéro perte, zéro doublon |
| S3 | frontières de vague `_run_r55_active_cpu_waves` `19720` | checkpoints 90 % | crédit fermé ; arrêt propre à la frontière si capacité dégradée | G2b | pas d'OOM ; refus honnête |

### 10.7 Codes de refus proposés (ensemble fermé, versionné)

```text
R55_REFUSE_HOST_RAM              # existant, RAM-only insuffisant
GAUGE_CREDIT_DISABLED
GAUGE_CREDIT_NOT_APPLICABLE      # restauré/importé/per-cell/GPU/non-Linux
GAUGE_CREDIT_REQUIRES_ONE_WORKER
GAUGE_CREDIT_GROUP_NOT_SINGLETON
GAUGE_CREDIT_UNKNOWN_CAPACITY    # un fait requis est inconnu
GAUGE_CREDIT_SWAP_BELOW_MINIMUM
GAUGE_CREDIT_INSUFFICIENT        # crédit < credit_needed
GAUGE_CREDIT_COMMIT_INSUFFICIENT
GAUGE_CREDIT_NONPAGEABLE_RAM_INSUFFICIENT
R55_REFUSE_DISK / R55_REFUSE_VRAM / R55_UNCALIBRATED  # existants, inchangés
```

Chaque refus porte : policy, version de schéma, sources, valeurs brutes, raison, et
l'acteur (jauge/stack/cache/assembly/export).

---

## 11. Découpage G1/G2 et critères d'acceptation exacts

### G1a — Fournisseur de capacité Linux (pur, hors moteur)

Livrable : module pur `core/zegrid/gauge_swap_capacity.py` (nom indicatif), schéma
`gauge-swap-capacity-v1`, aucune écriture, aucun appel produit.

Fonctions :

```text
read_meminfo_text(text) -> dict
read_swap_devices(text, stat_fn) -> [SwapDevice]
read_overcommit(mode, ratio, kbytes) -> CommitFacts
read_cgroup_levels(own_cgroup, root) -> [CgroupLevel]
combine_cgroup(levels, kind) -> trinary
evaluate_credit(capacity, p_gauge_raw, policy) -> CreditDecision
```

Critères d'acceptation (tous obligatoires) :

1. Matrice de vecteurs exacts (celle de `g0_capacity.py --self-test`, étendue) :
   zram seul, zram+file, fichier creux, `SwapCached` ignoré, cgroup inconnu,
   cgroup capé, racine absente, contrôleur désactivé, mode 0/1/2, champs commit
   manquants, non-paginable > `B_ram`, `S = 1 GiB−1 / 1 GiB / 1 GiB+1`, arrondis impairs.
2. Aucun float dans les décisions ; sérialisation JSON canonique stable.
3. Faits injectés par fichiers temporaires, aucun accès `/proc` obligatoire dans les tests.
4. Tout `inconnu` requis ⇒ crédit 0 + raison ; jamais d'infini.
5. Aucun import de `zemosaic_zegrid_mode` / `scheduler` (test d'import).

### G1b — Admission par groupes et hash versionné

Livrable : évaluateur pur `core/zegrid/gauge_admission.py` :

```text
evaluate_groups(groups, b_ram, credit) -> GroupDecision
```

Critères :

1. **Policy OFF** : pour N fixtures, décision et hash **byte-identiques** au chemin
   actuel (`aggregate_host_peak_bytes`), y compris le groupe gagnant.
2. Marge 15 % appliquée **une fois par groupe** (somme intra-groupe avant marge).
3. Seuil exact : accepté à `ceiling`, refusé à `ceiling + 1 B`, pour la jauge **et**
   pour chaque autre groupe.
4. Crédit refusé si le groupe jauge n'est pas singleton, si restauré, si workers ≠ 1,
   si plateforme non Linux.
5. Crédit ne sauve jamais stack/cache/assembly/export.
6. Hash : policy absente/`ram_only` ⇒ hash legacy identique ; policy `bounded_swap` ⇒
   digest d'exécution versionné **seulement si un crédit est réellement appliqué**
   (`credit_bytes > 0`), sinon `decision_hash` reste le hash legacy ; identité
   scientifique identique dans tous les cas. (Décision architecte du rework r1 :
   lève la tension G1b-6/G2a-3.)
7. Table de refus typés exhaustive (aucun `except` générique).

### G2a — Câblage produit

Critères :

1. Tous les gates §10.1–10.3, §10.5 utilisent **le même** évaluateur (test AST/grep :
   aucune comparaison directe `aggregate_host_peak_bytes(...) > budget` sur le chemin
   jauge éligible).
2. Test de mutation : retirer la propagation de policy d'un seul point ⇒ un test échoue.
3. Run synthétique public :
   - policy OFF : décisions/hashes identiques à la baseline ;
   - policy ON + RAM confortable : aucune différence de décision, de workers ou de hash ;
   - policy ON + capacité contrainte synthétique : jauge admise à 1 worker, stack refusé
     s'il ne tient pas `B_ram`.
4. Fermeture : après jauge, une RAM insuffisante refuse même avec un crédit énorme.
5. CPU, external, GPU testés séparément ; aucun ne reçoit de crédit indu.
6. Jauge restaurée/importée : zéro crédit, identité et commits inchangés.
7. Per-cell : aucun crédit transmis ; diagnostic explicite.

### G2b — Configuration, UX, Stop

Critères :

1. `zegrid_gauge_memory_policy` round-trip, enum strict, défaut `ram_only` ; valeur
   inconnue rejetée par le moteur (pas seulement par la GUI).
2. Snapshot `run_config_snapshot.json` contient la policy effective ; le worker la reçoit.
3. GUI : libellé exact, case désactivée/masquée sur plateforme non qualifiée **avec
   raison visible** ; désactivation = comportement RAM-only sans toucher aux stores.
4. Stop pendant la jauge : aucun artefact partiel annoncé complet, aucun crédit persisté,
   recalcul possible ; latence mesurée.
5. Reprise : cellules durables réutilisées, aucun doublon, aucun échec dû à un changement
   de policy.

### G3 — Qualification

Voir §13 (protocole et seuils gelés). Aucun run complet 20k sans probe bornée
représentative préalable.

### G4 — Revue et clôture

Revue Nono indépendante, vérification Junior, limites publiées bornées aux plateformes
réellement testées. Windows = lot séparé, non annoncé qualifié.

---

## 12. Oracles bit-exacts

| Oracle | Objet | Définition exacte | Verdict |
| --- | --- | --- | --- |
| `G0-ORACLE-1` | replay NGC7023 | JSON `raw=2203281820`, `peak=2533774093`, `budget=1664142540`, `deficit=869631553`, `s_needed=966257282`, 3 checks `true`, `status=REPRODUCED` | byte-exact |
| `G0-ORACLE-2` | capacité | vecteurs d'entrée/sortie entiers du §11 G1a, golden JSON | byte-exact |
| `G0-ORACLE-3` | décision OFF | `plan_hash`, tuples de décision, `aggregate_bytes`, reason codes d'une fixture synthétique déterministe, capturés **avant** G1b puis figés | byte-exact |
| `G0-ORACLE-4` | science | à workers identiques : SHA-256 canonique des plans science, coverage, poids, coefficients, exclusions, référence, par cellule et global | bit-à-bit |
| `G0-ORACLE-5` | fermeture | après `gauge_credit_closed`, tout refus est un refus de stack/cache/assembly/export ; aucun reason code de crédit | exact |

Règles :

- Aucune tolérance nouvelle n'est introduite pour faire passer un oracle.
- Un `faux MemAvailable` / des faits injectés valident **uniquement** la logique.
- Toute différence science impose une analyse ; elle n'est jamais absorbée par un seuil.
- Le hash legacy est calculé sans policy ; le hash d'exécution versionné est séparé.

---

## 13. Qualification physique : isolation, protocole, seuils gelés

### 13.1 Isolation disponible (vérifiée)

Sonde réellement exécutée (aucune pression, aucun changement système) :

```bash
systemd-run --user --scope --quiet \
  --property=MemoryMax=268435456 \
  --property=MemorySwapMax=0 \
  /bin/sh -c 'cat /proc/self/cgroup; ...'
```

Résultat observé :

```text
cgroup = /user.slice/.../app.slice/run-p201559-i201859.scope
memory.max       = 268435456
memory.swap.max  = 0
memory.current   = 720896
exit             = 0
```

Conséquences :

- un cgroup v2 **transient** avec plafond RAM **et** plafond swap est disponible sans
  privilège root, sans modifier `/proc/sys/vm/swappiness`, sans toucher aux devices de
  swap ni aux limites système ;
- un OOM éventuel est **confiné au scope** ; l'hôte n'est pas visé ;
- le scope disparaît à la sortie ; rien n'est persistant.

Limites :

- on ne peut pas choisir le **device** de swap utilisé (priorité système zram > swapfile) ;
- on ne peut pas tester physiquement `overcommit_memory=2` sans changer un sysctl
  système (interdit en G0) : le gate commit mode 2 reste **unit-testé par injection** ;
- le manager utilisateur est `degraded` mais la création de scope fonctionne.

### 13.2 Protocole de qualification (G3)

Préconditions :

1. Sorties neuves dans `/var/tmp/openclaw-tristan/...` ; corpus utilisateur en **lecture
   seule** ; aucune écriture dans `/home/tristan/NGC7023` ni dans les stores existants.
2. M33 intact ; aucun runtime partagé ; aucun run concurrent sur l'hôte.
3. Scope systemd avec `MemoryMax` et `MemorySwapMax` explicites ; taille du scope
   dimensionnée pour contenir `P_gauge + marge` sans jamais approcher la RAM totale.
4. Télémétrie : `/proc/swaps` par device, `memory.swap.current`, `memory.max`,
   `/proc/vmstat` (`pswpin/pswpout`, `pgmajfault`), PSI mémoire/IO, RSS arbre, timestamps
   Stop, `dmesg` (lecture seule) pour vérifier l'absence d'OOM hôte.
5. Aucun timeout moteur ajouté ; le watchdog est **externe** au moteur.

Séquence :

1. **Petit synthétique** : bout-en-bout, injections de capacité, policy OFF/ON.
2. **NGC7023 témoin** : RAM confortable, 1 worker ; capturer les oracles science.
3. **NGC7023 secours** : scope borné, 1 worker, policy ON ; mêmes entrées, sorties
   séparées ; comparer jauge et produits finaux.
4. **Séparation des effets** : refaire le témoin à 1 worker **sans** crédit (mesure de
   l'effet « 1 worker ») puis avec crédit (effet « pagination »).
5. **Stop/Resume** : arrêt avant jauge, pendant jauge (mi-décodage et mi-écriture),
   après jauge avant admission, pendant le stack ; reprise ; vérifier la parité science sur l'ensemble des cellules.
6. **20k** : probe bornée représentative d'abord ; aucun run complet extrapolé depuis
   NGC7023.

### 13.3 Seuils de coût/latence gelés **avant** mesures

Ce sont des **critères de qualification**, pas des timeouts moteur. Les dépasser signifie
« non qualifié sur cette plateforme », pas « bug » ni « annulation automatique ».

| ID | Critère | Seuil |
| --- | --- | --- |
| `Q1` | temps de jauge, même 1 worker | `≤ 4,0 ×` le témoin RAM confortable |
| `Q2` | temps total du run | `≤ 3,0 ×` le témoin |
| `Q3` | trafic swap pendant la jauge | `≤ 4 ×` la partie paginable modélisée (`P_gauge_raw − G_np_new`) |
| `Q4` | signature de thrashing | aucun intervalle `> 10 min` avec progression de frames nulle **et** swap-out `> 50 MiB/s` |
| `Q5` | pic mémoire | `RSS arbre + Δ memory.swap.current ≤ P_gauge_raw + 15 % + 256 MiB` de slack de mesure |
| `Q6` | Stop | acquittement coopératif `≤ 30 s` ; sortie process `≤ 120 s` ; aucun commit perdu |
| `Q7` | reprise | zéro perte, zéro doublon, seules les cellules pendantes exécutées |
| `Q8` | science | `G0-ORACLE-4` bit-à-bit à workers identiques ; tout écart = échec |
| `Q9` | sécurité hôte | zéro OOM hôte ; un OOM **dans le scope** = qualification invalide (pas un succès) |
| `Q10` | télémétrie | tous les champs requis présents ; sinon **inconclusif**, jamais « pass » |

### 13.4 Stop/Resume — contrat de preuve

- La jauge n'a **pas** de commit durable : un Stop mi-jauge ne doit jamais produire un
  artefact annoncé complet. Le redémarrage recalcule la jauge.
- Le crédit n'est jamais persisté : aucun store, aucun snapshot durable ne doit le
  contenir comme un fait acquis.
- Le Stop doit être coopératif aux frontières sûres ; un `SIGKILL` n'est pas une preuve
  de Stop propre.
- Après reprise, vérifier : identité, cellules réutilisées, absence de doublons, science
  finale bit-exacte, et que le run ne revendique pas un crédit qui n'existe plus.

### 13.5 Ce qu'une qualification physique peut/ne peut pas prouver ici

| Affirmation | Statut sur cet hôte |
| --- | --- |
| La logique de crédit fonctionne | **prouvable** (unités + synthétique + injections) |
| Un run réel admet la jauge sous pression | **prouvable** dans un scope borné |
| Les pages excédentaires vont sur le **disque** | **non prouvable** sans modifier zram/priorités (interdit) |
| Pas d'OOM | **non promis** ; un OOM de scope invalide la qualification |
| Débit acceptables | **mesurable**, seuils Q1–Q4 |
| Parité science | **prouvable** bit-à-bit à workers identiques |
| Windows / GPU qualifiés | **non** : lots séparés, CuPy absent ici |

---

## 14. Risques, incertitudes et blockers

### 14.1 Blockers physiques (à traiter en G3/G4, pas masqués)

1. **zram prioritaire** : tant que zram a du libre, le crédit « disque » n'est pas
   réellement consommé. Preuve disque impossible sans changement autorisé de swap.
2. **Mode commit 2** : l'hôte est en mode 0 ; le gate strict n'est validable que par
   injection. Une validation physique exigerait un sysctl interdit ici.
3. **GPU** : CuPy absent ; le backend GPU n'est pas qualifiable sur cette machine.
   Aucun crédit GPU n'est proposé en v1, donc ce n'est pas bloquant pour le crédit, mais
   interdit toute revendication GPU.
4. **Windows** : adaptateur pagefile/commit non écrit, non qualifié, hors v1 Linux.

### 14.2 Incertitudes de modèle

- `G_np_new = 340 MiB` est une **borne conservative**, pas une mesure du non-paginable.
- `P_gauge_raw` est un modèle ; le pic réel peut différer. La marge 15 % n'est pas une
  garantie.
- Le commit est compté en virtuel ; le noyau peut refuser pour d'autres raisons.
- Les faits d'hôte sont volatils (MemAvailable a varié de 877 MB à 1 198 MB pendant G0).
- La concurrence d'autres processus du cgroup n'est pas modélisée : `memory.swap.max`
  peut être consommé par un voisin entre le snapshot et l'allocation.

### 14.3 Risques opérationnels

- Thrashing : le crédit peut transformer un refus rapide en run très lent ; les seuils
  Q1–Q4 bornent la qualification, pas le runtime.
- OOM : jamais exclu ; un OOM de scope = échec de qualification.
- Le crédit peut cannibaliser de la RAM via zram (compression) avant d'utiliser le disque.
- Un `memory.max` cgroup plus serré que `B_ram` n'est **pas** ré-admis en RAM-only dans
  G0 (pour préserver l'identité OFF) ; il est utilisé pour la garde non paginable du
  crédit. C'est une limite documentée, pas une garantie supplémentaire.

### 14.4 Ce que G0 n'a pas fait

- Aucun code produit modifié.
- Aucun test de pression, aucun OOM, aucun benchmark long.
- Aucun commit, push, merge, tag, déploiement.
- Aucune modification de swap, swappiness, zram, cgroup système, M33 ou données.

---

## 15. Checklist d'acceptation G0

| Critère de la mission | Preuve |
| --- | --- |
| Formules et sources sans double comptage | §4, §5, §6, §7 ; `SwapFree` global interdit, zram exclu, SwapCached ignoré |
| Inconnus non masqués | §5 ; tout `inconnu` requis ⇒ crédit 0 + raison |
| Refus exact reproduit | §8 ; replay `REPRODUCED`, 3 checks `true` |
| Gates cartographiés | §10 ; 32 gates avec fichier:ligne, lot et test |
| Tâches suivantes concrètes | §11 ; G1a/G1b/G2a/G2b avec critères exacts |
| Limites de qualification explicites | §13.5, §14 ; zram/commit/GPU/Windows |
| Pas de code produit modifié | §1.1, §16 ; seuls doc + outil G0 ajoutés |
| Opt-in Linux + 1 worker | §4.4, §2.3 ; aucune objection concrète |
| Pas de timeout moteur arbitraire | §13.3 ; seuils = qualification externe |
| Scope allocations vs socle défini | §3, §9 ; pas de soustraction `B_ram`/non-paginable |
| Oracle bit-exacts | §12 ; `G0-ORACLE-1..5` |
| Isolation physique analysée | §13.1 ; scope systemd vérifié, sans changement système |

---

## 16. Artefacts livrés et état Git

### 16.1 Ajouts G0 (untracked, non commités)

| Fichier | Rôle |
| --- | --- |
| `docs/refactor/ZEGRID_GAUGE_SWAP_G0_CONTRACT_20261115.md` | ce document |
| `tools/zegrid_gauge_swap/g0_capacity.py` | outil pur/read-only : self-test, faits live, replay NGC7023 |

### 16.2 État Git final attendu

- Branche/HEAD inchangés : `feat/zegrid-adaptive-resources` @ `e25fb51…`.
- Fichiers préexistants inchangés (mêmes entrées `M`/`??`).
- Deux nouveaux untracked G0.
- **Aucun commit créé ; aucun push.**

### 16.3 Reproductibilité

```bash
python3 tools/zegrid_gauge_swap/g0_capacity.py --self-test
python3 tools/zegrid_gauge_swap/g0_capacity.py --facts
python3 tools/zegrid_gauge_swap/g0_capacity.py --replay \
  /home/tristan/NGC7023/out/instrum/worker_crash_breadcrumbs.jsonl
```

L'outil ne lit que `/proc`, `/sys` et le fichier de breadcrumbs fourni ; il n'écrit rien,
ne swap rien, ne modifie aucune limite.

---

## Annexe A — Décisions à confirmer par Junior (pas des blockers G0)

1. **`R_MIN = 0` + `S_MIN_USABLE = 1 GiB`** : G0 remplace la réserve chiffrée L1 (non
   validée) par une porte d'éligibilité. Si Junior veut une réserve relative
   (`max(256 MiB, 2 % RAM)`), la formule est prête (`R_MIN`) mais la valeur doit être
   validée avant G1b.
2. **Cgroup RAM en RAM-only** : G0 ne l'ajoute pas aux décisions RAM-only (pour garder
   OFF byte-identique). Il ne sert qu'à la garde non paginable du crédit. Si Junior veut
   un gate RAM-only cgroup, c'est un changement de comportement séparé, à qualifier.
3. **Codes de refus** : la liste §10.7 est proposée ; les noms peuvent être alignés sur
   la convention du scheduler avant G1b.
4. **Périmètre 1 worker** : conservé ; aucune objection concrète identifiée en G0.

## Annexe B — Preuves et chemins

- Refus : `/home/tristan/NGC7023/out/instrum/worker_crash_breadcrumbs.jsonl` (lecture seule),
  événements du 2026-11-14T22:29:00Z.
- Plan : `docs/refactor/ZEGRID_GAUGE_BOUNDED_SWAP_PLAN_20261115.md`.
- Revue : `/home/tristan/.openclaw/workspace/.a2a-reports/ZM-GAUGE-DYNAMIC-SWAP-PLAN-20261115.nono.r0.md`.
- Code : `src/zemosaic/zemosaic_zegrid_mode.py`, `src/zemosaic/core/zegrid/scheduler.py`,
  `src/zemosaic/core/zegrid/resource_envelope.py`, `src/zemosaic/core/zegrid/ram_contract.py`,
  `src/zemosaic/core/zegrid/runtime_control.py`, `src/zemosaic/zemosaic_gui_qt.py`,
  `src/zemosaic/zemosaic_config.py`.
- Rapport de mission : `/home/tristan/.openclaw/workspace/.a2a-reports/ZM-GAUGE-SWAP-G0-20261115.coco.r0.md`.
