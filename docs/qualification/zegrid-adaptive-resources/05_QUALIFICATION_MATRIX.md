# 05 — QUALIFICATION_MATRIX

Statut : **matrice figée**. Les lignes non exécutées en A0 sont NOT_RUN, jamais
PASS. Les commandes utilisent des chemins réels ; les moyens indisponibles sont
déclarés comme blocages.

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
- Statut : **PASS** (Linux ; le grand canevas M33 réel reste en A6/Q61).

### Q41 — A4, finitions/preview + disque presque plein

- Commande : `tests/test_zegrid_a4_q41_disk.py` (3 tests, verts).
- Preuve : l'admission tarifie les finitions (DBE/RGB/hole-fill/uint16) de façon
  véridique, refuse sur disque presque plein **sans désactiver les finitions**
  (finishing.enabled reste True), avant toute création de génération (atomique).
- Statut : **PASS** (Linux).

### Q50 — A5, budget 4→1→4 Gio, restart

- Linux : budgets 4/1 Gio réels.
- Statut : **NOT_RUN**.

### Q51 — A5, interruption milieu d'unité, corruption témoin

- Commandes : `tests/test_zegrid_resume_m4.py`, `tests/test_zegrid_p0_resume_integrity.py` (existants).
- Statut local : non exécutés en A0 ; **NOT_RUN**.

### Q60 — A6, E2E synthétiques 6 budgets

- Linux : 1/2/4 Gio ; Windows : 1/2/4/8/16 Gio.
- **32 Gio non qualifiable** sur les deux machines (RAM totale < 32 Gio) :
  déclaré non qualifié, pas inventé.
- Statut : **NOT_RUN**.

### Q61 — A6, M33 complet Windows GPU

- Commande : GUI ou CLI avec `run_config_snapshot` figé, sortie neuve.
- Statut : **NOT_RUN**.

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
- Statut : **NOT_RUN**.

### Q63 — A6, M106/M16 CPU/GPU non-régression

- Statut : **NOT_RUN**.

### Q64 — A6, GUI→backend

- Test GUI offscreen + run synthétique via le chemin GUI.
- Statut : **NOT_RUN**.

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
