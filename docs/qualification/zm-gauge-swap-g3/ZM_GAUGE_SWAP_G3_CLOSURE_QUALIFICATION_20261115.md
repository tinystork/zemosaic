# ZeGrid — secours swap borné de la jauge : fiche de clôture de qualification

- **Mission** : `ZM-GAUGE-SWAP-G4-20261115` (clôture G0→G4), lot de clôture du
  chantier `ZM-GAUGE-SWAP`.
- **Base** : `feat/zegrid-adaptive-resources`,
  HEAD `e25fb5133768adf4eb26950c63f86b970909b370` (**aucun commit**).
- **Contrat** : `docs/refactor/ZEGRID_GAUGE_SWAP_G0_CONTRACT_20261115.md` (§10.6 O1, §13).
- **Preuves** : `docs/qualification/zm-gauge-swap-g3/physical_qualification.json`
  (schéma `g3-physical-qualification-v2`) et `logical_qualification.json`
  (schéma `g3-logical-qualification-v1`) ; évidences brutes sous
  `/var/tmp/openclaw-tristan/zm-gauge-g3/`.
- **Revue indépendante** : Nono review-1 `ZM-GAUGE-SWAP-G3-20261115.nono.r1.md`
  (**ACCEPT**), puis acceptation Junior. G4 corrige le seul résidu bloquant pour
  l'observabilité (R1) et écrit cette fiche (R2).
- **Publication** : **aucune**. Aucun push, tag, bump de version, build ou
  déploiement. La publication d'une beta est une décision de Tristan.

## 1. Verdict composite

> **QUALIFIED WITH LIMITS** — qualifié sur le chemin ACTIVE-CPU stream/canvas
> (crédit de jauge réellement consommé, pagination observée, science bit-à-bit,
> Stop/Resume propre) et sur l'**admission** du crédit dans le cas
> `external_spool` (crédit ouvert/fermé, plafond stack à `B_ram`, digest publié,
> science bit-à-bit). **Pas** « tous seuils verts » : quatre seuils sont
> partiels, non exercés ou inconclusifs, et plusieurs limites physiques de cet
> hôte restent non prouvables (voir §3).

Ce verdict n'autorise aucune revendication sur : le swap **disque** réel, le
mode commit 2, le GPU, Windows, un corpus réel, ni un gain de performance.

## 2. Seuils Q1–Q10 — libellés exacts

Verdicts **tels que publiés** dans `physical_qualification.json` ; aucun seuil
n'a été desserré et aucun verdict non démontrable n'a été promu.

| ID | Critère (contrat §13.3) | Seuil gelé | Observé | Verdict exact |
| --- | --- | --- | --- | --- |
| `Q1` | temps de jauge, même 1 worker | `≤ 4,0 ×` témoin | `57,745 s` vs `79,445 s` = **0,7269 ×** | `PASS_INDICATIVE` |
| `Q2` | temps total du run | `≤ 3,0 ×` témoin | `231,542 s` vs `185,252 s` = **1,2499 ×** | `PASS_INDICATIVE` |
| `Q3` | trafic swap pendant la jauge | `≤ 4 ×` paginable modélisé (`3 581 818 560 B`) | pic scope `623 493 120 B` ; `pswpout` run `620 077 056 B` | `PASS` |
| `Q4` | signature de thrashing | aucun intervalle `> 10 min` à progression nulle **et** swap-out `> 50 MiB/s` | run borné `< 10 min` (231,5 s) | `NOT_EXERCISED` |
| `Q5` | pic mémoire (fenêtre jauge) | `≤ P_gauge_raw + 15 % + 256 MiB` = `1 708 201 508 B` | `1 311 150 080 B` (`memory.current` 1 180 430 336 + swap 130 719 744) | `PASS` |
| `Q6` | Stop (acquittement, sortie, commits) | ack `≤ 30 s` ; sortie `≤ 120 s` ; aucun commit perdu | ack **2,105 s** ; sortie **3,612 s** ; `commits_at_stop = 0` | `PASS` |
| `Q7` | reprise | zéro perte, zéro doublon, seules les cellules pendantes exécutées | 4/4 cellules, 0 doublon, 0 perte ; **réutilisation d'une cellule déjà commitée non exercée** | `PASS_PARTIAL` |
| `Q8` | science | `G0-ORACLE-4` bit-à-bit à workers identiques | canvas `a125029f…` = témoin ; external `52e396e3…` = témoin ; `digest_present = true` | `PASS` |
| `Q9` | sécurité hôte | zéro OOM hôte ; OOM **dans le scope** = qualification invalide | scope `exit_code 0`, `signal null`, aucun marqueur OOM ; dérivé (`scope_oom = false`) | `PASS_WITH_LIMIT` |
| `Q10` | télémétrie | tous les champs requis ; sinon **inconclusif**, jamais « pass » | manquants : `/proc/pressure/memory`, `/proc/pressure/io`, `dmesg` noyau | `INCONCLUSIVE` |

Détails de dérivation :

- `Q6` sous **crédit ouvert** (preuve rework-2) : `gauge_credit_open` à
  `0,124 s`, requête Stop à `1,501 s`, `gauge_credit_closed` à `3,536 s` —
  ordre `open < request < close` vérifié dans les événements bruts ;
  `swap_used_bytes = null` dans l'événement de fermeture ; **zéro** chaîne de
  crédit dans le store au Stop ; reprise `4/4`, `finalized = true`.
- `Q9` : le verdict est **dérivé** de la méta de scope
  (`exit_code in (137) | signal == 9 | marqueurs OOM`). La méta du cas canvas
  est `NOT_CAPTURED` (harnais r0 antérieur) : l'absence d'OOM y est **indirecte**.
- `Q3` : `pswpout` couvre tout le run (borne **supérieure** de la phase jauge).
- `Q8` : comparaison à workers identiques (1 worker), canvas **et** external.

## 3. Limites complètes (reprises de l'artefact, non édulcorées)

1. L'observation `A` (`MemAvailable`) est **injectée** par le harnais
   (monkeypatch), jamais par un flag produit : cela prouve la **logique**, pas
   une pénurie RAM réelle.
2. L'application du swap est **réelle** dans le cas canvas : scope systemd
   transitoire, pic `memory.swap.current` 623 Mo, `pswpout` 620 Mo.
3. Le swap **disque** n'est **pas** prouvé : zram (priorité 100) absorbe
   d'abord ; le prouver exigerait de changer les priorités de swap (interdit).
4. Le mode commit 2 (`overcommit_memory=2`) n'est validé que par **injection
   unitaire** ; le changer exigerait un sysctl système (interdit).
5. Le **GPU** n'est pas qualifié : CuPy absent de l'interpréteur officiel.
6. Le corpus est **synthétique** (8 frames déterministes, deux géométries), pas
   le corpus réel de l'utilisateur ; aucune donnée utilisateur n'a été lue.
7. `Q10` est `INCONCLUSIVE` : pas de PSI, pas de `dmesg` noyau lisible.
8. `Q4` est `NOT_EXERCISED` : un run borné (< 10 min) ne peut pas déclencher une
   fenêtre de thrashing de 10 min.
9. `Q1`/`Q2` sont `INDICATIVE` : `n = 1`, le témoin a tourné **en premier**,
   le page cache n'est pas contrôlé — **aucun gain n'est revendiqué** (le
   `0,73 ×` n'est pas un gain).
10. Les preuves **external** et **Stop sous crédit** sont postérieures au
    rework-1 ; les runs canvas/Stop de r0 les précédaient (aucun crédit external,
    aucun crédit ouvert au Stop) — c'est le rework-2 qui comble ces deux trous.
11. `Q7` : la **réutilisation** d'une cellule déjà commitée n'est pas exercée
    (0 commit au Stop) ; la preuve porte sur zéro perte / zéro doublon.
12. La reprise **external** exige de rejouer le **même `A`** injecté : avec le
    `A` réel, la tuile gelée diffère (128 vs 112) et le moteur refuse
    correctement la reprise (conservatisme d'identité, pas un défaut).
13. Le cas **external** n'a **pas** consommé de swap (`peak_swap_current = 0`,
    pic mémoire 599 Mo < plafond 1 GiB) : il prouve l'admission du crédit, la
    fermeture avant stack, le plafond stack à `B_ram`, le digest et la parité
    science — **pas** une pagination réelle. La pagination réelle est prouvée
    par le cas **canvas**.

## 4. Preuves de qualification (chiffres de référence)

### 4.1 Cas canvas (pagination réelle)

| Fait | Témoin (`ram_only`) | Secours (`bounded_swap`) |
| --- | --- | --- |
| `A` injecté | 1 800 000 000 B | 1 600 000 000 B |
| durée | 185,252 s | 231,542 s |
| phase jauge | 79,445 s | 57,745 s |
| pic `memory.current` | 2 349 559 808 B | 1 181 339 648 B |
| pic `memory.swap.current` | 0 B | **623 493 120 B** |
| `pswpout` | — | **620 077 056 B** |
| science `sha256` | `a125029fa7ec…3073f3` | **identique** |

Crédit observé (canvas) : `S_credit = 1 449 551 462 B`,
`S_effectif = 1 610 612 736 B`, `B_ram = 1 280 000 000 B` (`b_ram_eff`
928 739 328 B), `P_gauge_raw = 1 251 970 480 B`, `credit_needed = 159 766 052 B`,
`gauge_workers_effective = 1`, scope `memory.max = 1 181 114 368 B`,
`memory.swap.max = 1 610 612 736 B`.

### 4.2 Cas external (admission du crédit)

`execution.cell_task_mode = "external"`, classe `external_spool`, backend CPU,
1 worker ; `gauge_swap.credit_applied = true`,
`credit_bytes = 1 449 551 462 B`, `credit_needed = 51 935 960 B`,
`B_ram = 440 000 000 B`, `S_effectif = 1 610 612 736 B` ;
`stack_ceiling_bytes = 440 000 000 B` (**stack à `B_ram`**,
`stack_credit_applied = false`, `stack_envelope_bytes = 77 084 362 B`) ;
`credit_closed_before_stack = true` ; `execution_digest =
sha256:6798f2690e4f72fff88842311626e272ae59012ea29fb6934472bbd75c3ad47a`
(identique dans le manifeste et l'artefact) ; scope `memory.max = 1 GiB`,
`swap.max = 1 610 612 736 B`, `exit_code = 0`, pic mémoire 598 523 904 B,
pic swap 0 ; science `52e396e3a2af…c1064` **identique** au témoin external,
couverture 608 820 px et trous 2 343 px identiques.

### 4.3 Qualification logique

`logical_qualification.json` : **50 checks, 0 échec**, familles L1–L8 (L8 =
nouvelle famille external), durée 0,762 s, faits **injectés uniquement**
(aucun `/proc`, aucun moteur, aucune pression). Bornes vérifiées, dont :
plafond jauge `14 388 140 441 B` (admis à la borne, refusé à `+1`),
dernière admission stack `7 469 508 340 B` (refus à `+1`).

## 5. Décisions d'architecte entérinées

1. **External crédité** : la classe de stack `external_spool` reçoit le crédit
   de jauge **singleton** (rework-1 G3). Le crédit porte **toujours** sur la
   jauge uniquement ; le stack external reste strictement RAM-only et plafonné
   à `B_ram`. Vérifié par le run `fallback-ext3-out`.
2. **Digest seulement si crédit** : `execution_digest` n'est publié que
   lorsqu'un crédit a **réellement** été appliqué (absent sinon, jamais `null`
   mensonger). Verrouillé par
   `test_gauge_swap_block_publishes_execution_digest_only_when_credited`.
3. **Pas de flag CLI** : la politique n'existe que comme clé de configuration
   (`zegrid_gauge_memory_policy`, enum fermé `ram_only` / `bounded_swap`) et
   case GUI. L'injection de l'observation `A` n'existe **que** dans le harnais
   de qualification ; aucun chemin produit ne permet de forger la capacité.
4. **Double-baseline = dette assumée** : `Q1`/`Q2` comparent un témoin
   `ram_only` (exécuté en premier, `A` injecté 1,8 GB) à un run secouru
   (`A` injecté 1,6 GB), `n = 1`. Ce n'est **pas** une baseline unique et
   contrôlée : les ratios sont donc `PASS_INDICATIVE` et **aucun gain** n'est
   revendiqué. Rejouer une baseline à `A` identique, n ≥ 3, cache contrôlé est
   une dette de mesure, pas un bloquant de correction.
5. **R1 (G4)** : les champs O1 sont rétablis dans `ram_contract` via un helper
   unique (`_gauge_credit_ram_contract_fields`) appliqué aux **six** sites
   `_ram_contract_diagnostics` de `_run_single` (bloc initial, deux
   écrasements per-cell, trois blocs GPU/fallback). `gauge_swap` n'est donc
   **pas** la seule surface O1.

## 6. Résidus et dettes (non bloquants)

| ID | Description | Statut G4 |
| --- | --- | --- |
| `R1` | champs O1 perdus dans `ram_contract` sur le chemin per-cell | **corrigé + testé** (§5.5) |
| `R2` | `Q10 INCONCLUSIVE` et verdicts `PASS_*` partiels à inscrire dans la fiche | **inscrit** (§2) |
| `R3` | external sans consommation de swap (admission seulement) | **déclaré** (§3.13) |
| `D1` | baseline unique contrôlée pour `Q1`/`Q2` (n ≥ 3, cache contrôlé) | dette de mesure |
| `D2` | `Q7` réutilisation de cellule déjà commitée | non exercé |
| `D3` | `Q4` fenêtre de thrashing 10 min | non exerçable ici |
| `D4` | deux suites Stop/Resume Qt (`m6_qt_stop_resume`, `m6_unified_stop_capability`) non exécutées en G3 | dette de non-régression hors périmètre G4 |
| `D5` | `tests/test_zegrid_r41b_product_wiring.py` (test `B`) bloque dans GEOS/shapely sur cet hôte, hors du jeu ciblé G1/G2/G3 | non attribué à G4 (voir rapport G4 §validation) |

## 7. Ce que cette qualification n'autorise pas

- Aucune revendication « swap disque garanti », « pas d'OOM garanti »,
  « GPU/Windows qualifiés », « gain de performance mesuré ».
- Aucune promesse de tenue sur un corpus 20k réel : la qualification est
  bornée au corpus synthétique et à un run `< 10 min`.
- Le crédit reste une **PERMISSION**, jamais une réservation :
  `swap_used_bytes` est toujours `null` (fait observé séparément, hors bloc).

## 8. Reproductibilité

- Interpréteur officiel :
  `/home/tristan/.openclaw/workspace/projects/zemosaic/.venv/bin/python`,
  `PYTHONPATH` retiré, tests Qt en `offscreen`.
- Artefacts : `docs/qualification/zm-gauge-swap-g3/{logical,physical}_qualification.json`
  (+ `logical_qualification.py`, `physical_summary.py`), évidences brutes et
  logs de scope sous `/var/tmp/openclaw-tristan/zm-gauge-g3/`.
- Les runs physiques ne sont **pas** rejoués en G4 (aucun nouveau run, aucune
  pression, aucun benchmark).
