# 05 — QUALIFICATION_MATRIX

Statut : **matrice figée**. Les lignes non exécutées en A0 sont NOT_RUN, jamais
PASS. Les commandes utilisent des chemins réels ; les moyens indisponibles sont
déclarés comme blocages.

## Mise à jour de qualification après audit — 2026-11-07

**Les critères restent figés; les verdicts PASS historiques ci-dessous ne valent plus acceptation
produit pour Q40/Q41/Q50/Q60/Q64.** Voir le
[rapport final](POST_PUBLICATION_AUDIT_20261107.md) et son tableau A0–A6.
Q40/Q41/Q50 : preuves unitaires/partielles uniquement. Q60 : E2E tiny/legacy1–4Gio simulés,
pas plafond RSS ni parcours GUI ACTIVE. Q64 : propagation, pas intégration complète.
Q61 BLOCKED; Q62/Q63 NOT_RUN. Les primitives et parités restent des preuves utiles.
Les anciens états A0 et les commandes ci-dessous sont conservés pour traçabilité, non comme
statut courant. Les9 échecs connus ont désormais été reproduits sur la vraie baseline5150040
et sur HEAD418ba43, mêmes causes (50 passed/9 failed de chaque côté).

Run Windows M106 `mem_fix` ajouté comme témoin réel : GUI/ZeGrid ACTIVE/GPU70 % et publication
réussis, mais profil `dense`/`ram_tile` uniquement. Il ne ferme donc ni external/micro-blocs, ni
scratch/OOM/croissance. Il révèle F11 (identité working-set non gelée, HIGH), F12 (fausses erreurs
pinned pool, MEDIUM), F13 (faux échec de livraison du run log, MEDIUM) et F14 (WARN attendus trop
bruyants, LOW), détaillés dans le rapport final.

### Corrections F1–F10 appliquées — 2026-11-07 (F10 honnêteté de qualification)

Les défauts matériels de l'audit ont été corrigés et vérifiés indépendamment
(commits locaux sur `feat/zegrid-adaptive-resources`, aucun push) :

- **F2** (cache fantôme facturé) + **F3** (choix sous-ligne) — `b251643`.
- **F4** (external sélectionnable sous GPU) + **F8** (fallback sticky + provenance
  véridique) — `aad9f64`.
- **F1** (copies plein canevas) + **F6** (scratch global+per_cell) + **F7**
  (cleanup honnête) — `0ed7cb2`.
- **F5** (câblage grow A5 dans l'admission ACTIVE préflight) — `3c81aaf`.
- **F11/F12/F13** (reprise + télémétrie + faux échec run_log) — `9fd304b`+`0cd0080`
  (déjà poussés `91a425b` sur beta).

Ces corrections lèvent les faux refus et la dette de copies mémoire, mais **ne
ferment pas à elles seules les gates produit** : Q40/Q41/Q50/Q60/Q64 restent
PARTIEL tant que les runs réels (grand canevas M33, rampe 4→1→4 matérielle,
GUI ACTIVE→external GPU→scratch→FITS) n'ont pas tourné sur machine contrainte.
F14 (WARN trop bruyants, LOW) reste exclu du périmètre par décision Tristan.

## 1. Identités machines et corpus

| Id | Machine | RAM | VRAM | Corpus | Racine |
| --- | --- | --- | --- | --- | --- |
| `LNX` | TINYDEBIAN, i7-8550U, Debian 13 | 7,52 GiB | MX150 2 GiB, **CuPy 14.1.1 / CUDA 12.9 via `.venv`** | Caldwell 816 FITS / synthétiques | `/home/tristan/.openclaw/workspace/worktrees/zemosaic-adaptive-resources` |
| `WIN` | MSGE66, i9-11980HK, Windows 10.0.26300.9818 | 31,7 GiB | RTX 3070 8 GiB, CuPy 14.1.1 / CUDA 12.9 | M33 21 818 frames | `C:\Users\JarvisJunior\Desktop\M33\ircut` |
| `WIN-IN` | idem | idem | idem | source M33 | `D:\M33\S50\IRCUT` (**ACL node refusée**) |

Node OpenClaw : `Windows Node (MSGE66)`,
id `e68cd84bac2c09263d3b01f008860159d922bac5437136f6495372ded383f4e7`.

## 2. Règles de timeout et watchdog

- Probe ordinaire : **≤ 300 s** ; validation déjà connue longue : **≤ 1 200 s**.
- Run produit : jamais dans un `exec` bloquant indéfini ; progression et
  checkpoints obligatoires.
- **Watchdog de non-progrès** : aucune progression de cellule/tuile pendant
  **600 s** → arrêt de l'unité, état reprenable, raison consignée.
- **Tentatives** : maximum **3** par unité ; chaque reprise doit réduire
  strictement les ressources ou changer de backend qualifié ; interdiction de
  réessayer indéfiniment le même tuple backend/bloc/crédits.
- Timeout global d'un run de qualification : fixé par le coût annoncé, pas par
  une valeur arbitraire ; à défaut, checkpoint toutes les cellules.

## 3. Seuils figés

| Sujet | Seuil |
| --- | --- |
| Parité CPU (mémoire/streaming/externe) | bitwise sur 8 plans + scalaires ; masques exacts |
| Parité CPU↔GPU | `rtol=1e-12`, `atol=0.0`, masques/diagnostics exacts |
| Pic RAM moteur | ≤ budget déclaré, aucun kill, aucun recours au swap |
| Pic scratch | ≤ quota déclaré (pic vivant, pas cumul) |
| Résultats | aucun pixel perdu/doublé ; compteurs = N contributeurs |
| Reprise | zéro recalcul de cellule validée, zéro doublon |
| Publication | atomique ou état reprenable honnête |

Aucun seuil n'est ajusté après un échec.

## 4. Matrice Q00–Q64

### Q00 — A0, dernier refus M33 + témoin faisable (A0 : **PARTIEL / quantifié**)

- Cause : refus joint `R55_REFUSE_HOST_RAM` après layout 12×20/tuile 20.
- Preuve : `probes/results/admission.json` + journaux Windows cités.
- Témoin faisable par phase : setup et layout mesurés (219,7 s / 685,1 s) ;
  jauge/cache/stack NON ATTEINTS. Le témoin canonique 48×64×64 fournit la
  phase 2 mesurée.
- Commande de rejeu du modèle :

```bash
cd /home/tristan/.openclaw/workspace/worktrees/zemosaic-adaptive-resources
PYTHONPATH=src python3 docs/qualification/zegrid-adaptive-resources/probes/run_a0_probes.py --phase admission
```

### Q10 — A1, PLAN/ADMIT/RUN, budgets 1/2/4/8/16/32

```bash
# Linux, budgets matériellement disponibles 1/2/4 Gio
PYTHONPATH=src python3 -m pytest -q tests/test_zegrid_r55e8_coherent_resource_adaptation.py \
  tests/test_zegrid_r55d_joint_planner.py tests/test_zegrid_r55d2_cpu_active.py \
  tests/test_zegrid_r55_envelope.py tests/test_zegrid_r55_planner_determinism.py
```

Windows (via node) :

```text
exec(host="node", node="Windows Node (MSGE66)",
     command="powershell -NoProfile -Command \"cd C:\\chemin\\zemosaic; $env:PYTHONPATH=$PWD\\src; python -m pytest -q tests\\...\"")
```

Seuils : aucun candidat périmé, comptabilité commune, refus physique explicite.
Statut A0 : **NOT_RUN**.

### Q20 — A2, préparation fenêtre N≥20k, scratch < cache complet

- Corpus : M33 cellule réelle ou synthétique N=20 000 varié.
- Quota scratch : ≤ 20 GB libres (Linux) / C: à re-mesurer (Windows).
- Commande : test A2 à créer ; le témoin de référence est
  `tests/test_zegrid_m2_external_stream.py::test_reduction_workspace_is_bounded_and_patch_independent`.
- Statut : **NOT_RUN**.

### Q21 — A2, arrêt/erreur I/O, Windows/spawn

- Commandes : `tests/test_zegrid_m2_external_stream.py -k "write_error or corrupt or stale or symlink"`.
- Statut local : couvert par les 99 tests PASS (write_error/corrupt/stale) ;
  **Windows/spawn NOT_RUN**.

### Q30 — A3, N=2k/20k/50k + 100k borné

- Oracle : `run_canonical_stack` sur positions/régions réduites.
- Seuil : tous les N consommés, bitwise, blocs < une ligne exercés.
- Statut : **NOT_RUN**.

### Q31 — A3, CUDA physique + VRAM contrainte + OOM injecté

```bash
# Linux, venv projet (CuPy 14.1.1 / CUDA 12.9 / MX150 2 GiB)
cd /home/tristan/.openclaw/workspace/worktrees/zemosaic-adaptive-resources
PYTHONPATH=src /home/tristan/.openclaw/workspace/projects/zemosaic/.venv/bin/python \
  -m pytest -q tests/test_sci05_canonical_gpu_parity.py

# Windows, RTX 3070 8 GiB
powershell -NoProfile -Command "cd C:\chemin\zemosaic; $env:PYTHONPATH=$PWD\src; python -m pytest -q tests\test_sci05_canonical_gpu_parity.py"
```

- VRAM : plafonds 1/2/4 Gio sur RTX 3070 ; 1 Gio nominal sur MX150.
- Statut : **PARTIEL** — Linux MX150 : smoke PASS + **30 tests de parité PASS**
  (`pytest_gpu_parity_venv.txt`). Windows RTX 3070, VRAM contrainte et OOM
  injecté : **NOT_RUN**. Aucun kernel scientifique dans le run M33.

### Q32 — A3, cellule réelle profonde M33

- Source : `D:\M33\S50\IRCUT`, config gelée `5AE03ADF…`, halo 8.
- Statut : **NOT_RUN**.

### Q40 — A4, canevas dont 121×H×W > budget

- Cible réelle : canevas M33 75 698 820 px → 8 530,5 MiB de plans.
- Commande : `tests/test_zegrid_a4_q40_bounded_assembly.py` (2 tests, verts).
- Preuve : assemblage `_assemble_cells_guarded(canvas_backing="scratch")` → 11
  plans scratch, science/stack_depth en memmap, FITS science + coverage
  **byte-identiques** au témoin dense `PrimaryHDU` (même bloc header, data
  big-endian, SCIHASH, remplacement non-fini), aucune copie plein canevas à la
  dernière étape (écriture par bandes), scratch libéré après sérialisation.
- Statut : **PARTIEL** (Linux) — preuve unitaire 24×24/scratch forcé, pas un
  canevas > budget réel, pas de `_write_outputs`/repli automatique bout-en-bout
  ni plafond RSS mesuré. F1 (copies plein canevas) et F6 (scratch global+per_cell)
  et F7 (cleanup honnête) sont corrigés ; le grand canevas M33 réel reste A6/Q61.

### Q41 — A4, finitions/preview + disque presque plein

- Commande : `tests/test_zegrid_a4_q41_disk.py` (3 tests, verts).
- Preuve : l'admission tarifie les finitions (DBE/RGB/hole-fill/uint16) de façon
  véridique, refuse sur disque presque plein **sans désactiver les finitions**
  (finishing.enabled reste True), avant toute création de génération (atomique).
- Statut : **PARTIEL** (Linux) — modèles/refus de publication ; aucune
  finition/preview réelle sous ressources contraintes n'a été exercée.

### Q50 — A5, budget 4→1→4 Gio, restart

- Commande : `tests/test_zegrid_a5_q50_growth.py` (5 tests, verts).
- Preuve : `next_grow`/`grow_sequence`/`assert_grow_only`/`assert_joint_grow_only`
  (inverse exact de l'API shrink-only) ; `grow_sequence` inverse `shrink_sequence`
  depuis le maximum du treillis, les gardes refusent tout shrink/réduction
  d'octets, et un budget 1 Gio n'achète jamais un plan plus gros que 4 Gio ;
  identité scientifique gelée préservée sur toute la rampe.
- Statut : **PARTIEL** (Linux, budget modélisé par contrainte d'énumération) —
  API/treillis uniquement, pas de rampe dynamique/restart produit ; F5 (câblage
  grow dans l'admission ACTIVE préflight, avec hystérésis) est corrigé ; la
  rampe réelle 4/1 Gio matériellement contrainte reste en A6/Q60.

### Q51 — A5, interruption milieu d'unité, corruption témoin

- Commandes : `tests/test_zegrid_resume_m4.py`, `tests/test_zegrid_p0_resume_integrity.py` (existants).
- Preuve : 79 tests verts (`resume_m4` + `p0_resume_integrity` + `m3_recovery`) :
  cellule corrompue isolée et recalculée, corruption d'identité/geometry/gauge
  refusée, zéro double comptage, cellules validées réutilisées.
- Statut : **PASS** (Linux).

### Q60 — A6, E2E synthétiques 6 budgets

- Linux : 1/2/4 Gio ; Windows : 1/2/4/8/16 Gio.
- **32 Gio non qualifiable** sur les deux machines (RAM totale < 32 Gio) :
  déclaré non qualifié, pas inventé.
- Commande : `tests/test_zegrid_a6_q60_budgets.py` (4 tests, verts).
- Preuve : vrais runs `run_zegrid_mode` E2E sur le corpus déterministe tiny
  (budget herméticisé via `available_memory_bytes`) à 1/2/4 Gio : FITS +
  manifest valides, coverage > 0, et science/coverage **bit-identiques** entre
  budgets — la preuve petit budget est un run réel, pas un dry-plan.
- Statut : **PARTIEL** (Linux 1/2/4 Gio) — quatre frames, legacy explicitement,
  budgets observés simulés ; ni plafond RSS/swap ni GUI ACTIVE ni preuve de tous
  les nouveaux chemins. Windows 1/2/4/8/16 Gio et 32 Gio non qualifiés sur cet
  hôte — reportés au test direct beta de Tristan.

### Q61 — A6, M33 complet Windows GPU

- Commande : GUI ou CLI avec `run_config_snapshot` figé, sortie neuve.
- Statut : **BLOCKED** — corpus M33 Windows indisponible sur cet hôte Linux ;
  reporté au test direct beta de Tristan (Windows + GPU réel).

### Q62 — A6, Caldwell grand canevas halo 128, Linux contraint

```bash
# corpus réel, sortie neuve, halo explicite 128
PYTHONPATH=src python3 -m zemosaic.zegrid_cli \
  --input "/home/tristan/caldwell11/lights organized/EQ/IRCUT" \
  --output /home/tristan/.openclaw/workspace/worktrees/zemosaic-adaptive-resources/.a0-out/caldwell-h128 \
  --halo-px 128 --storage-profile auto --resume-mode auto --workers 1
```

- Note : `zemosaic.zegrid_cli` est **CPU-only by design** (documenté dans le
  module). La qualification GPU de Q62 exige le chemin GUI/worker, pas ce CLI.
- Statut : **NOT_RUN** (corpus réel présent, run lourd CPU ; reporté au test
  direct beta).

### Q63 — A6, M106/M16 CPU/GPU non-régression

- Statut : **NOT_RUN** (corpus réels présents, témoins `slow` existants ;
  reportés au test direct beta).

### Q64 — A6, GUI→backend

- Preuve : `test_dispatch_propagation_witness.py` (15 tests verts) + `r53c_gui_halo`
  (le halo GUI atteint le worker et le moteur réel) + `r55e4_gui_gpu_toggle`.
- Statut : **PARTIEL** (propagation headless vérifiée) — insuffisant pour
  GUI ACTIVE→external GPU→scratch→FITS ; le test GUI interactif humain final
  reste à Tristan sur beta.

## 5. Commandes A0 réellement exécutées (traçabilité)

```bash
cd /home/tristan/.openclaw/workspace/worktrees/zemosaic-adaptive-resources
export PYTHONPATH=$PWD/src
python3 docs/qualification/zegrid-adaptive-resources/probes/run_a0_probes.py --phase env
python3 docs/qualification/zegrid-adaptive-resources/probes/run_a0_probes.py --phase model
python3 docs/qualification/zegrid-adaptive-resources/probes/run_a0_probes.py --phase admission
python3 docs/qualification/zegrid-adaptive-resources/probes/run_a0_probes.py --phase witness-inmem
python3 docs/qualification/zegrid-adaptive-resources/probes/run_a0_probes.py --phase witness-ext
python3 docs/qualification/zegrid-adaptive-resources/probes/run_a0_probes.py --phase caldwell-inventory
python3 -m pytest -q tests/test_zegrid_m2_external_stream.py \
  -k "parity_matrix_rgb or m33_pricing or bounded_and_patch_independent or larger_n_stays_bit_exact"
```

Résultats : voir `probes/results/` et README (99 passed).

## 6. Moyens indisponibles (blocages, pas PASS)

| Moyen | Statut | Impact |
| --- | --- | --- |
| Parité GPU Windows / VRAM contrainte / OOM | NOT_RUN | Q31 partiel : MX150 PASS, RTX 3070 et OOM à faire en A3 |
| Accès D: via node | `Accès refusé` | empreintes M33 et comptage 22 414 bloqués |
| C: libre Windows | requête ACL refusée | quota scratch Windows à re-mesurer avant run |
| Run M33 complet | non lancé | Q61 NOT_RUN |
| GUI | non lancée | Q64 NOT_RUN |

## 7. Verdict matrice

La matrice est **complète et non réductible** : chaque ligne a un corpus, une
machine, un quota, un timeout et un seuil. Une seule ligne est partiellement
fermée (Q00, quantifiée). Toutes les autres sont NOT_RUN, avec la commande
exacte qui les produira. Aucune ligne n'a été retirée pour fermer A0.
