# 06 — A0_VERDICT

Statut : **défaut d'admission reproduit et quantifié** ; limite GPU quantifiée ;
aucune correction produit en A0.

## 1. Défaut principal : PLAN accepte ce que ADMIT interdit

Le dernier run M33 (2026-11-07 12:38–12:53, build `rt-4085889f2625`, M6) :

1. setup : 21 818 frames, 0 rejet WCS ;
2. layout dense refusé, retry borné unique ;
3. layout **12×20 / 240 cellules / tuile 20** accepté avec
   `stream@20 = 9 289,1 MiB` pour `B_plan = 9 931,5 MiB` ;
4. GPU décliné explicitement (`R55_REFUSE_VRAM`) ;
5. refus **avant jauge/stack** :
   `R55_REFUSE_HOST_RAM: no coherent gauge/cache worker configuration fits
   B_plan with the 15/100 margin after the full bounded ladders
   (gauge [2,1] × cache [14..1])`.

### 1.1 Arithmétique (MODÈLE, fonctions produit)

`B_plan = (4 × 12 414,3) / 5 = 9 931,5 MiB`.

La marge d'agrégation 15/100 est appliquée **après** la somme des phases, en un
seul site : `resource_envelope.py:781–791` (`aggregate_host_peak_bytes`, qui
appelle `ceil_margin_div(raw, margin_num, margin_den)`). Les lignes 111–112 ne
sont que les constantes `DEFAULT_MARGIN_NUM/DEN`. Le plafond « brut » compatible
avec le budget
est donc :

```
B_plan / 1,15 = 8 636,1 MiB
```

Or le stack déjà accepté par le PLAN vaut **9 289,1 MiB**, soit
**653,0 MiB au-dessus du plafond brut**, avant même d'ajouter la jauge, le cache
ou le socle partagé.

Calculs (voir `probes/results/admission.json`) :

| Grandeur | MiB |
| --- | ---: |
| Budget `B_plan` | 9 931,5 |
| Plafond brut compatible (÷1,15) | **8 636,1** |
| Stack accepté `stream@20` | **9 289,1** |
| Dépassement du seul stack | **+653,0** |
| Jauge 1 worker (340 MiB + canevas×36) | **2 939,0** |
| Cache 1 worker (340 MiB + patch×36) | 350,8 |
| Agrégat minimal (jauge 1 + cache 1 + stack, +15 %) | **14 856,7** |
| Dépassement de l'agrégat minimal | **+4 925,2** |

### 1.2 Le refus est forcé, quelle que soit la lecture

- **Lecture A** — `stream@20` est un brut : il dépasse à lui seul le plafond
  brut de 653 MiB → aucun `gauge × cache` ne peut passer.
- **Lecture B** — `stream@20` inclut déjà la marge (brut ≈ 8 077,5 MiB) : il
  reste 558,6 MiB pour le socle + jauge + cache, alors que la jauge 1 coûte
  **2 939 MiB** → aucun passage possible.
- Les deux lectures concluent au refus. La conclusion ne dépend donc pas d'une
  interprétation du champ.

### 1.3 Pourquoi les échelles de repli ne peuvent pas sauver le run

- La jauge est proportionnelle au **canevas** (75,7 Mpx), pas à la tuile :
  même 1 worker coûte 2,9 GiB.
- Le stack est proportionnel à `N × tuile²` ; à tuile 20 il vaut 9 289 MiB.
  Pour passer sous 8 636 MiB brut, il faudrait une tuile plus petite, mais :
  - l'overhead fixe par tuile (2 560 KiB × nombre de tuiles) **augmente** quand
    la tuile diminue ; à tuile 12 on multiplie les tuiles par ~2,8 ;
  - le layout est déjà à 240 cellules (12×20) et le scan s'arrête au plancher
    `min_patch_area = 50 000 px` et `max_halo_overhead = 0,5`.
- Le refus n'est donc pas un défaut de « une ligne de plus dans la ladder » :
  c'est une **incohérence entre le critère d'acceptation du layout et le critère
  d'admission joint**.

### 1.4 Ce n'est pas un OOM

Au moment du refus (`worker_last_state.json`) :

- RSS processus = **814,3 MB**, pic = **1 002,8 MB** ;
- RAM système = 17,4 / 32,5 GB utilisés, ≈15 GB libres ;
- GPU utilisé = 1 352,7 MB, libres = 6 839,3 MB.

Le processus n'a jamais approché les 9,9 GiB. Le modèle statique refuse une
configuration que la machine aurait peut-être pu exécuter ; inversement, il
n'a jamais été prouvé que le pic réel aurait tenu. **A0 ne conclut ni « modèle
trop conservateur » ni « machine insuffisante » : il conclut que PLAN et ADMIT
ne mesurent pas la même chose.**

## 2. Défaut GPU : plancher qualifié vs VRAM réelle

- Budget VRAM appliqué : `5 224 634 777 B` = **4,8658 Gio = 4 982,6 Mio**.
  Ancrage : ligne `R55_REFUSE_VRAM` de `worker_crash_breadcrumbs.jsonl`
  (SHA-256 `0AACECEB…E9615F`). La valeur est exactement
  `70 % × 7 118,0 MiB` lus à `11:42:25Z`/`11:53:50Z` dans
  `resource_telemetry.csv` (`gpu_free_mb=7118.0`), et non 70 % des 6 839 MB
  post-contexte de `worker_last_state.json`.
- Workspace GPU à 96 B/échantillon, tuile étendue `t+18` :

| Tuile | Étendue | Workspace | Verdict |
| ---: | ---: | ---: | --- |
| 32 | 50 | 14 985,2 MiB | refus |
| 24 | 42 | 10 574,8 MiB | refus |
| 20 | 38 | 8 657,2 MiB | refus |
| 16 | 34 | 6 931,3 MiB | refus |
| 12 | 30 | 5 397,2 MiB | refus |
| 10 | 28 | 4 702,1 MiB | **tient** |
| 8 | 26 | 4 054,9 MiB | tient |

- La ladder GPU qualifiée s'arrête à **32** (`GPU_PRODUCT_TILE_LADDER`) : les
  tuiles 8–10 qui tiendraient ne sont **pas qualifiées**.
- Conclusion : le GPU est refusé par **absence de rung qualifiée**, pas par
  impossibilité physique. Le déclin M6 est correct (pas de résultat partiel),
  mais la capacité GPU reste inexploitée à N=21 818.
- Le rework-1 a exécuté la parité GPU sur MX150 (30 tests PASS). Cela prouve le
  **contrat numérique** et le runtime Linux, **pas** la capacité GPU à N=21 818 :
  le verdict M33 ci-dessus reste inchangé.

## 3. Inventaire des allocations dominantes

| Rang | Poste | Taille à N=21 818 | Où |
| ---: | --- | ---: | --- |
| 1 | Workspace phase 2 CPU (75 B/sample) | 7 089 MB | `streaming.py:160,267` |
| 2 | Jauge (canevas × 36 B + 340 MiB) **par worker** | 3 081,7 MB | `parallel.py:141–150` |
| 3 | Overhead fixe tuiles (2 560 KiB × 841) | ≈2,1 GiB | `streaming.py:161,268` |
| 4 | Assemblage plein canevas (121 B/px) | 8 530,5 MiB | `bounded_assembly.py:975–999` |
| 5 | Cache aligné d'une cellule (N × patch × 13 B) | **93,8 GB** | `_r55_cache_unit_bytes` |
| 6 | Workspace GPU (96 B/sample) à tuile 32 | 14 985 MiB | `gpu.py:77,162` |
| 7 | Réduction externe (31 B/position/frame) | 657,9 MiB | `external_stream.py:127` |
| 8 | Spool externe, pire tuile | 480,7 MiB | `external_stream.py:1027` |
| 9 | Socle processus | 340 MiB | `ram_contract.py:109` |
| 10 | Sorties/plans par cellule | O(patch) ≈ 27 MB | `streaming.py:270` |

Deux postes sont **structurellement hors budget** et ne se règlent pas par la
ladder : la jauge par canevas (2,9 GiB/worker) et l'assemblage plein canevas
(8,5 GiB). Le cache aligné (94 GB) est un problème disque, pas RAM.

## 4. Delta modèle / observations

| Élément | Modèle | Observé | Écart |
| --- | ---: | ---: | --- |
| Pic processus au refus | 9 289 MiB (stack) | 1 002,8 MB | ×9,3 |
| RAM système | budget 9 931,5 MiB | 15 073 MB libres | le budget n'est pas la RAM libre |
| GPU workspace tuile 20 | 8 657 MiB | 1 352,7 MB utilisés (contexte seul) | aucun kernel |
| Durée setup | — | 219,744 s (99,4 f/s) | mesuré |
| Durée layout | — | 685,150 s | mesuré |
| Cache disque cellule | 93,8 GB | non écrit (refus avant cache) | non observé |

Le « pic » de 9 289 MiB est une **borne d'admission**, pas une consommation.
A0 ne présente pas ce chiffre comme une mesure RSS. Inversement, le pic observé
de 1,0 GB ne prouve pas que le run aurait tenu : il n'a jamais alloué le
workspace de phase 2.

## 5. Chemin minimal complet proposé

Objectif : permettre à une cellule M33 réelle de progresser sous budget, sans
changer la science.

1. **Une seule enveloppe PLAN = ADMIT = RUN** (A1)
   - une marge unique, appliquée une fois ;
   - le layout n'accepte que si l'agrégat joint complet (jauge + cache + stack +
     assemblage) tient, pas seulement le stack ;
   - aucun re-calcul avec des dimensions périmées.
2. **Fenêtre de préparation + spool borné** (A2)
   - supprimer l'obligation du cache aligné N × patch (94 GB) ;
   - produire/consommer une fenêtre, recycler l'espace ;
   - conserver tous les N contributeurs.
3. **Réduction canonique par micro-blocs < une ligne** (A3)
   - le workspace devient `O(batch × N) + O(tile)` : 657,9 MiB à batch 1024,
     au lieu de 7 089 MB ;
   - GPU seulement si une rung qualifiée tient (< 32 px à N=21 818) ou après
     qualification explicite ; sinon déclin honnête.
4. **Assemblage par bandes + FITS streaming** (A4)
   - ne jamais matérialiser 121 B/px sur 75,7 Mpx (8,5 GiB) ;
   - hash streaming, écriture transactionnelle.
5. **Adaptation et reprise** (A5)
   - réduction/croissance des crédits, prefetch, pinned ;
   - reprise avec moins de RAM/VRAM ou sans GPU.
6. **Qualification** (A6)
   - M33 complet, Caldwell halo 128, budgets réels, CUDA physique, GUI.

Le chemin minimal **n'exige pas** de changer le halo, le rejet, la normalisation
ou les frames. Il exige de changer la **résidence** et le **découpage**, pas la
science.

## 6. Découpage A1 proposé

| Sous-tâche | Contenu | Preuve de sortie |
| --- | --- | --- |
| A1a | Contrat unique d'enveloppe : PLAN/ADMIT/RUN, marge une fois, agrégat par phases | Q10, tests de cohérence, rejeu M33 sans refus forcé |
| A1b | Admission du layout sur l'agrégat complet, pas sur le stack seul | test « layout accepté ⇒ agrégat tient » |
| A1c | Politique VRAM unique (50 % legacy vs 70 % ACTIVE) et déclin GPU explicite | test de décision GPU + trace |
| A1d | Identité scientifique vs paramètres d'exécution (reprise compatible) | test de reprise, hash de plan stable |

A1 ne doit pas commencer par « réduire les coefficients » : il doit d'abord
rendre les trois décisions cohérentes. La réduction des coefficients est un
levier de A2/A3, pas un prérequis de A1.

## 7. Risques restants

| Risque | Impact | Mitigation |
| --- | --- | --- |
| Ordre flottant modifié par le découpage | parité CPU perdue | oracle bitwise W1/W2 avant tout changement |
| Cache aligné 94 GB/cellule | disque saturé | fenêtre + spool, quota figé |
| Jauge 2,9 GiB/worker | refus persistant | admission par phases, workers dérivés du budget complet |
| Assemblage 8,5 GiB | échec en fin de run | bandes + FITS streaming (A4) |
| Filtre `_LP_` possible dans l'inventaire | science mélangée | audit D: avant A1, arrêt humain si confirmé |
| 22 414 non expliqué | baseline corpus ambiguë | comptage D: et reconciliation fichier par fichier |
| GPU non qualifié <32 px | capacité perdue | qualification explicite ou acceptation CPU |
| D: inaccessible au node | preuves corpus bloquées | session interactive ou partage lecture seule |
| 1,5 s de télémétrie | pics courts manqués | capteur ≤100 ms en A1 |
| Deux fractions VRAM | décisions incohérentes | politique unique en A1 |

## 7 bis. État d'implémentation A1 (rework-1, 2026-11-07)

- **PLAN = ADMIT** : le layout passe désormais par `_layout_joint_plan_probe`,
  qui appelle le vrai `build_plan_set` (marge 15/100 une seule fois au boundary).
  Le rejeu M33 refuse au layout avec un motif numérique, plus après acceptation.
- **RUN non qualifié sur l'assemblage** : le probe tarife l'assemblage à la
  convention produit **16 B/px** (cores retenus + stack_depth). L'assembleur
  feathered réel alloue **121 B/px** (`bounded_assembly.canvas_plane_bytes`,
  poste #4 A0) ; ce working set plein canevas est **HORS périmètre A1** et
  **déféré à A4**. Le probe expose `unpriced_assembly_canvas_bytes` et
  `assembly_canvas_deferred_to_a4=True` pour que le manifeste/log ne laisse pas
  croire à un PLAN=ADMIT=RUN complet.
- Aucune tarification 121 B/px n'a été ajoutée à A1 ; aucune marge nouvelle.

## 8. Verdict A0

- **Défaut reproduit/quantifié** : oui — l'acceptation du layout 12×20 est
  incompatible avec l'admission jointe ; le stack accepté dépasse à lui seul le
  plafond brut de 653 MiB, et l'agrégat minimal atteint 14 856,7 MiB pour
  9 931,5 MiB de budget.
- **Limite GPU quantifiée** : oui — à N=21 818 et 4 982,6 Mio (4,8658 Gio) de
  budget VRAM, seule une tuile ≤10 px tient ; la ladder qualifiée s'arrête à 32.
- **Chaîne d'allocations identifiée** : oui — workspace N×tuile², jauge par
  canevas, overhead par tuile, assemblage 121 B/px, cache 13 B/px/frame.
- **Chemin minimal complet** : proposé, sans changement scientifique.
- **Preuves manquantes** : empreintes M33 (ACL D:), run CUDA, run produit,
  GUI — toutes NOT_RUN/BLOCKED, jamais PASS.

A0 est **prêt pour revue Junior/Nono** avec ces limites explicites. Aucun
A1 ne doit démarrer avant `A0_ACCEPTED`.
