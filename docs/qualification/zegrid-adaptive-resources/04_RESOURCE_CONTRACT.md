# 04 — RESOURCE_CONTRACT

Statut : **COMPLET** (contrat figé). Les phases non atteintes sont NOT_REACHED,
jamais PASS. Les nombres sont étiquetés MESURÉ / OBSERVÉ / MODÈLE / EXTRAPOLÉ.

## 1. Définitions et unités

- **1 Gio = 2^30 octets** (1 073 741 824). Les points de qualification sont
  **1, 2, 4, 8, 16, 32 Gio**.
- **Budget moteur** = arbre des processus de calcul : imports et bibliothèques,
  workers, buffers vivants, transitoires, pinned, pages mappées résidentes,
  allocations d'assemblage. Mesuré **dès le démarrage du moteur**.
- **Hors budget moteur mais relevés** : OS, GUI réellement séparée, page cache,
  swap/pagefile. Aucun « socle gratuit » n'est ajouté au plafond.
- **Swap non compté comme RAM disponible.** Un run qui ne réussit qu'avec swap
  est un échec mémoire.
- **RAM et VRAM ne s'additionnent pas.** Ce sont deux domaines distincts.
- **Disque** : distinguer le **pic vivant** (cache + spool + résultats
  provisoires + marge) du **volume cumulé d'I/O**.

## 2. Budgets applicatifs existants (cités)

| Constante | Valeur | Source |
| --- | --- | --- |
| `B_plan` | `(4 × available_plan) / 5` | `ram_contract.py:89–90`, `plan_budget_bytes` |
| `B_admit` | `(9 × available_now) / 10` | `ram_contract.py:91–92`, `admit_budget_bytes` |
| Marge d'agrégation | **15/100**, appliquée une fois au pic agrégé | `resource_envelope.py:111–112` |
| Socle processus | 340 MiB (356 515 840 B) | `ram_contract.py:109` |
| Workspace CPU streaming | **75 B** / (pixel × canal × frame) | `streaming.py:160` |
| Workspace GPU | **96 B** / (pixel × canal × frame) | `gpu.py:77` |
| Halo de taper GPU/CPU | 9 px | `gpu.py:66`, `streaming.py:164` |
| Overhead fixe par tuile | 2 560 KiB | `streaming.py:161` |
| VRAM legacy | **50 %** de la VRAM libre | `gpu.py:55` |
| VRAM ACTIVE | **70 %** de la VRAM libre (provisoire, observe-only) | `gpu_readiness.py:182–188` |
| Assemblage canonique | **121 B/pixel** RGB | `bounded_assembly.py:975, 999` |
| Spool externe | `4×canaux + 4` B / (position × frame) | `external_stream.py:1025` |
| Réduction externe | **31 B** / (position × frame) | `external_stream.py:127` |
| Nettoyage spool | 5 %, min 1 MiB | `external_stream.py:131–134` |

Deux fractions VRAM coexistent (50 % legacy, 70 % ACTIVE). A0 les **enregistre**
comme un point de contrat à unifier en A1 ; aucune n'est déclarée fausse ici.
La valeur réellement appliquée au dernier run est **5 224 634 777 B =
4,8658 Gio = 4 982,6 Mio**. Elle est ancrée par la ligne exacte du message
`R55_REFUSE_VRAM` dans `worker_crash_breadcrumbs.jsonl`
(SHA-256 `0AACECEB73389F62CBAAB38ED36C25E24698005BC3201ABD08CF09A0E0E9615F`) :

```
R55_REFUSE_VRAM: no GPU streaming tile fits the fresh VRAM budget
(5224634777 B); refusing rather than extrapolating
```

La proportion se referme exactement : `0,70 × 7 118,0 MiB × 2^20 = 5 224 634 777 B`,
avec `gpu_free_mb = 7118.0` lu dans `resource_telemetry.csv` à
`2026-11-07T11:42:25.647843Z` (setup) et à `11:53:50.805816Z` (layout). Le
`gpu_free_mb = 6839.3` de `worker_last_state.json` (11:53:52Z) est une lecture
postérieure, après allocation du contexte CuPy (`gpu_used_mb = 1352.7`) ; il ne
sert pas de base au budget. La proportion « 70 % » est donc **mesurée**, pas
approximative.

## 3. Capteur et métriques

### 3.1 Exigence §12

- échantillonnage cible **≤ 100 ms** ;
- compléter par high-water OS/allocateur ;
- publier au minimum : pic working set/RSS de l'arbre, mémoire privée/commit,
  mémoire système disponible et pression ; sous Linux, cgroup et mémoire
  file-backed ; GPU : libre système, allocation vivante, pool, contexte.

### 3.2 Capteur réellement disponible

| Plateforme | Capteur | Fréquence | High-water |
| --- | --- | --- | --- |
| Windows | `psutil` RSS + `peak_wset` (`process_peak_rss_source=psutil_peak_wset`) ; NVML/`cupy_meminfo` | **1,5 s** (`resource_telemetry_interval_sec`) | `peak_wset` |
| Linux (A0) | `VmHWM`/`VmRSS` de `/proc/self/status` | fin de phase | `VmHWM` |

**Écart contractuel** : 1,5 s > 100 ms. Le capteur ne peut pas manquer un pic
court de moins de 1,5 s. Action A1 : ajouter un échantillonneur ≤100 ms et
conserver `peak_wset` comme borne haute. A0 ne prétend pas que 1,5 s suffit.

### 3.3 Trace mesurée — dernier run M33 (OBSERVÉ)

| Phase | Durée | RSS processus | Pic processus | RAM système | GPU |
| --- | --- | ---: | ---: | --- | --- |
| Setup | 219,744 s | — | — | 20 055 / 32 469 MB utilisés ; 12 414 MB dispo | 1 073,5 / 8 191,5 MB |
| Layout | 685,150 s | 814 MB | **1 002,8 MB** | 17 397 / 32 469 MB ; 15 073 MB dispo | 1 352,7 / 8 192 MB ; 6 839 MB libres |
| Gauge | **NOT_REACHED** | — | — | — | — |
| Cache | **NOT_REACHED** | — | — | — | — |
| Stack | **NOT_REACHED** | — | — | — | — |
| Assemblage | **NOT_REACHED** | — | — | — | — |
| FITS | **NOT_REACHED** | — | — | — | — |

Sources : `resource_telemetry.csv`, `worker_last_state.json`,
`zemosaic_worker.log`.

Lecture : au moment du refus, **le processus moteur ne pèse que 1,0 GB** et il
reste 15 GB système. Le refus n'est pas un OOM observé ; c'est une décision
d'admission statique. La RAM « disponible » de 12 414 MB au setup donne
`B_plan = 9 931,5 MiB`.

### 3.4 Trace mesurée — témoin canonique (MESURÉ, Linux)

| Phase | Mémoire | Externe |
| --- | ---: | ---: |
| 48 frames 64×64 RGB, WSC+mean+taper | 0,472 s | 0,902 s |
| `VmHWM` | 172 253 184 B (164,3 MiB) | 145 276 928 B (138,6 MiB) |
| Delta au-dessus des tableaux d'entrée | 30,9 MB | 3,9 MB |
| Fichiers résiduels du spool | — | **0** |
| Dossiers résiduels du spool | — | **1** dossier cellule vide (`a0cell`) sur racine persistante |

Le chemin externe réduit le pic de pic de 27 MB sur ce témoin minuscule ; il ne
prouve pas le comportement à N=21 818, mais il valide le capteur et le contrat
« O(batch × N) + O(tile) ».

**Résidu spool, distinction exacte (rework-1)** : après succès, la primitive
`run_cell_streaming_external` ne laisse **aucun fichier** ; elle peut laisser le
dossier de cellule **vide** (`a0cell`). Le test M2 n'exige que la disparition du
dossier d'essai (`attempt0`), pas celle du dossier cellule. Le chemin produit
`cleanup_after_commit` (`external_stream.py:518–536`) supprime explicitement les
dossiers vides (cellule puis racine). Il n'y a donc **pas de résidu de données** ;
le dossier vide est un artefact de nommage de la primitive, pas une fuite.

## 4. Modèle par phase pour M33 (MODÈLE)

Paramètres : N = 21 818, tuile 20, halo de taper 9 (étendue 38×38), 3 canaux,
canevas 75 698 820 px.

**Deux aires de patch, deux usages (L2)** :

- la phase `model` utilise `patch = (200, 200)` comme **prix de forme** : les
  termes dominants (N × tuile², baseline, jauge) ne dépendent pas du patch, et
  cette valeur évite de simuler une géométrie non prouvée ;
- la phase `admission` utilise l'**aire de cellule réelle** ≈ 315 411 px
  (canevas / 240 cellules), et l'ajustement du modèle au 9 289,1 MiB observé
  donne 659 × 502 = 330 818 px (les patchs se chevauchent, donc l'aire réelle
  est supérieure au quotient canevas/cellules). Les deux valeurs ne sont pas
  interchangeables ; elles sont nommées ici pour éviter toute confusion
  d'audit.

| Poste | Valeur | Nature |
| --- | ---: | --- |
| Workspace phase 2 CPU | 7 089 MB (6,76 GiB) | MODÈLE (75 B/échantillon) |
| Overhead fixe tuiles | ≈ 2,1 GiB (841 tuiles × 2 560 KiB) | MODÈLE |
| Baseline | 340 MiB | MODÈLE |
| Borne streaming@20 | **9 289,1 MiB** | OBSERVÉ (log), reproduit à 8 MiB près |
| Jauge, 1 worker | **3 081,7 MB** (2 939 MiB) | MODÈLE (`340 MiB + canvas×36`) |
| Cache, 1 worker | **367,9 MB** | MODÈLE (`340 MiB + patch×36`) |
| Stack + marge 15 % | 10 682,5 MiB | EXTRAPOLÉ |
| Agrégat minimal (jauge 1 + cache 1 + stack, +15 %) | **14 856,7 MiB** | EXTRAPOLÉ |
| Budget | **9 931,5 MiB** | OBSERVÉ |
| GPU, tuile 32 | 14 985,2 MiB | MODÈLE (96 B) |
| GPU, tuile 20 | 8 657,2 MiB | MODÈLE |
| GPU, tuile 10 | 4 702,1 MiB | MODÈLE |
| Budget VRAM | **4 982,6 Mio (= 4,8658 Gio)** | OBSERVÉ (`5224634777 B`) |
| Assemblage plein canevas | **8 530,5 MiB** (121 B/px) | MODÈLE |
| Cache aligné d'une cellule | **93,8 GB** (N × patch × 13 B) | MODÈLE |
| Spool externe, pire tuile | 480,7 MiB | MODÈLE |
| Réduction externe, batch 1024 | 657,9 MiB | MODÈLE |

Points structurants :

- **La jauge est proportionnelle au canevas**, pas à la tuile : 36 B/px ×
  75,7 Mpx = 2,7 GiB **par worker**, avant le socle.
- **Le workspace phase 2 est proportionnel à N × tuile²**, pas au patch :
  descendre la tuile réduit ce terme de façon quadratique.
- **L'overhead fixe par tuile** (2 560 KiB × nombre de tuiles) devient dominant
  quand la tuile est petite ; à tuile 20 il ajoute ≈2,1 GiB.
- **L'assemblage plein canevas (121 B/px) dépasse à lui seul le budget** :
  A4 doit être traité comme un problème distinct, pas comme une conséquence du
  stack.
- **Le cache aligné N × patch × 13 B ≈ 94 GB par cellule** : à halo 8 et
  layout 12×20, ce n'est pas 2,2 TB (halo 64) mais reste proche du disque C:
  annoncé (≈115 GB). La fenêtre/spool doit remplacer cette obligation.

## 5. Réserves et plancher

- Réserve système : maintenue hors budget moteur, mesurée par la RAM système
  disponible. Aucune valeur fixe n'est ajoutée au plafond.
- En pression : arrêt des admissions, fin/annulation de l'unité en vol,
  réduction puis rejeu, sans double comptage.
- Plancher physique : si le socle + une unité minimale ne tient pas, l'état est
  **suspendu/reprenable avec raison**, pas une attente infinie.
- Le plancher GPU **qualifié** est 32 px (`GPU_PRODUCT_TILE_LADDER`). Le calcul
  montre qu'à N=21 818 seule une tuile ≤10 px tient dans 4,87 GiB ; elle n'est
  pas qualifiée. Le refus GPU est donc **une limite de qualification**, pas une
  impossibilité physique.

## 6. Quotas figés pour A1+

| Ressource | Linux TINYDEBIAN | Windows MSGE66 |
| --- | --- | --- |
| RAM totale | 7,52 GiB | 31,7 GiB |
| Points RAM qualifiables | 1, 2, 4 Gio (8 Gio = total, non qualifiable) | 1, 2, 4, 8, 16 Gio (32 Gio > total, non qualifiable) |
| VRAM | 2 GiB (MX150) → plafonds 1 Gio et nominal sûr | 8 GiB → plafonds 1, 2, 4 Gio + nominal |
| Scratch minimal | 20 GB libres sur `/` | 20 GB libres sur C: (à re-mesurer avant run) |
| Swap | interdit comme RAM | pagefile interdit comme RAM |

Un point non matériellement disponible est **non qualifié**, jamais inventé.

## 7. Verdict ressource

Le contrat est **figé** : unités, budgets, marges, formules et capteurs sont
cités avec leur source. La trace M33 est mesurée jusqu'au layout puis
NOT_REACHED. Le modèle explique le refus par une **agrégation de phases
incompatible avec le budget**, pas par un OOM. Les deux écarts à corriger en A1
sont : (1) capteur 1,5 s → ≤100 ms ; (2) deux fractions VRAM 50 %/70 % → une
seule politique.
