# 01 — BASELINE

Statut : **COMPLET** (les seuls champs manquants sont des requêtes refusées par
l'ACL Windows, explicitement NOT_AVAILABLE).
Date : 2026-11-07. Aucune branche déplacée, aucun commit créé.

## 1. Références Git

| Réf | SHA | Date | Sujet |
| --- | --- | --- | --- |
| `HEAD` / `feat/zegrid-adaptive-resources` | `515004078cfb64b327fb02aeaa323e48cca62bd1` | 2026-11-07 11:28:17 +0100 | `fix(zegrid): decline an impossible ACTIVE GPU instead of killing the run (M6)` |
| `origin/beta` | `515004078cfb64b327fb02aeaa323e48cca62bd1` | idem | idem |
| `origin/main` | `f8acdbb0ec34bef7dd21e69deadf9352103fdc3c` | 2026-11-05 10:42:42 +0100 | `chore(release): bump ZeMosaic to 5.1.1` |
| checkout documentaire ancien | `ec708a1` | 2026-11-02 | **non utilisé** |
| référence externe `/tmp/astro-stack-ref` | `c06ba830b4d969c8c9da2beb729298c04995d8c1` | — | MIT, lecture seule |

**Vérification au lancement** : `git fetch --all --prune` exécuté le 2026-11-07,
sans nouveau réf. `git rev-list --count origin/beta ^HEAD` = **0** et
`git rev-list --count HEAD ^origin/beta` = **0** : beta n'a pas bougé, baseline
retenue = HEAD `5150040`. Aucun déplacement de branche.

## 2. État Git du worktree

```
## feat/zegrid-adaptive-resources
?? docs/qualification/
?? docs/refactor/ZEGRID_ADAPTIVE_RESOURCE_ENGINE_PLAN_20261107.md
?? docs/refactor/ZEGRID_ADAPTIVE_RESOURCE_ENGINE_STATE_20261107.md
```

- Aucun fichier suivi modifié ; aucun code produit/science touché.
- `git diff --check` → code de sortie **0**, aucune sortie.
- Travaux concurrents : aucun autre worker sur ce worktree. Le worktree
  `zemosaic-auto-external-streaming` partage le même SHA mais n'a pas été
  modifié ni utilisé (lecture documentaire seulement).

## 3. Plateforme Linux (hôte de cette session)

| Champ | Valeur | Source |
| --- | --- | --- |
| Hôte | `TINYDEBIAN` | `uname -a` |
| OS | Debian GNU/Linux 13 (trixie), noyau `6.12.113+deb13-amd64` | MESURÉ |
| CPU | Intel Core i7-8550U @ 1.80 GHz, 4 cœurs / 8 threads | `lscpu` |
| RAM | 8 071 196 672 B (7,52 GiB) ; swap 33 840 685 056 B | `free -b` |
| GPU | NVIDIA GeForce MX150, 2 048 MiB, pilote 550.163.01, compute 6.1 | `nvidia-smi` |
| Disque `/` | 253 G, 168 G utilisés, **73 G libres** | `df -h` |
| Disque `/tmp` | tmpfs 3,8 G ; `tempfile` écrit sous `/var/tmp/openclaw-tristan` | `df -h` |

### Deux interpréteurs Linux (correction rework-1)

| Interpréteur | Python | NumPy | CuPy | CUDA runtime | MX150 |
| --- | --- | --- | --- | --- | --- |
| `/usr/bin/python3` | 3.13.5 | 2.2.4 | **absent** (`ModuleNotFoundError`) | — | non interrogé |
| `/home/tristan/.openclaw/workspace/projects/zemosaic/.venv/bin/python` | 3.13.5 | **2.5.1** | **14.1.1** | **12 090 (CUDA 12.9)** | **1 device, 2 091 253 760 B** |

Le venv du projet est le runtime de développement canonique. Il répond :

```
cupy 14.1.1 / cuda_runtime 12090 / cuda_device_count 1
device NVIDIA GeForce MX150 (compute 6.1)
memGetInfo: free 2 050 621 440 B, total 2 091 253 760 B
```

Un smoke CuPy a été exécuté via ce venv (allocation 64 MiB, `sum` exacte,
libération) : **PASS_SMOKE**, artefact `probes/results/gpu_smoke.json`.
`nvcc` reste absent du système, mais le runtime CUDA 12.9 embarqué par CuPy
fonctionne.

Conséquence : la branche GPU n'est **pas absente** sur Linux ; elle n'a
simplement **pas été exécutée dans le périmètre A0** (aucun test de parité, aucun
kernel scientifique). La parité GPU Linux est **disponible pour A3** via ce venv.
Les tests exécutés avec `/usr/bin/python3` se sont auto-skippés
(`pytest.importorskip("cupy")`) et ne constituent pas une preuve GPU.

## 4. Plateforme Windows (machine du dernier run M33)

Accès via le node OpenClaw `Windows Node (MSGE66)`
(`e68cd84bac2c09263d3b01f008860159d922bac5437136f6495372ded383f4e7`).

| Champ | Valeur | Source |
| --- | --- | --- |
| Compte | `msge66\jarvisjunior` | `whoami` (node) |
| OS | Microsoft Windows `10.0.26300.9818` | `ver` (node) |
| CPU | 11th Gen Intel Core i9-11980HK @ 2.60 GHz | registre (node) |
| RAM | 32 469,5 MB (31,7 GiB) | télémétrie du run (`ram_total_mb`) |
| GPU | NVIDIA GeForce RTX 3070 Laptop GPU, 8 192 MiB | télémétrie + `nvidia-smi` |
| Pilote GPU | 576.80, compute capability 8.6 | `nvidia-smi` (node) |
| Disque C: libre | **NOT_AVAILABLE** — requête ACL refusée (`FreeSpace=0`) ; l'audit antérieur rapportait ≈115 GB libres, non re-vérifié | node |
| Disque D: | **NOT_AVAILABLE** — `D:\M33\S50\IRCUT` renvoie « Accès refusé » au compte du node | node |

### Build réellement exécuté

Le run de 12:38–12:53 a tourné dans le slot
`C:\Users\JarvisJunior\AppData\Local\zealfie\runtime\slots\rt-4085889f2625`
(cité dans les tracebacks des breadcrumbs). Versions interrogées dans ce slot :

| Composant | Version |
| --- | --- |
| ZeMosaic | 5.1.1 |
| Python | 3.13.15 |
| NumPy | 2.5.4 |
| CuPy | 14.1.1 |
| CUDA runtime | 12 090 (CUDA 12.9) |

Le run de 10:13 cité par l'audit antérieur utilisait un autre slot
(`rt-58d40221cd27`), antérieur au commit M6 `5150040` : son échec `RuntimeError`
n'est **pas** représentatif du code de tête. Les deux runs sont donc séparés :

- 10:13 → ancien runtime, échec par exception (pré-M6) ;
- 12:38 → `rt-4085889f2625`, comportement M6 (déclin GPU explicite), puis refus
  CPU `R55_REFUSE_HOST_RAM` : **c'est la baseline de comportement retenue**.

## 5. Dépendances et limites de mesure

- Aucune dépendance ajoutée (pas de PyTorch, pas de fp16, pas de nouveau paquet CuPy : le venv existant est utilisé en lecture seule).
- Les probes A0 utilisent NumPy/stdlib et les fonctions produit importées en
  lecture seule.
- Le capteur de pointe Linux est `VmHWM` (`/proc/self/status`) ; Windows utilise
  `psutil_peak_wset` (champ `process_peak_rss_source`).
- Aucune écriture dans le corpus, aucun run de mutation, aucun commit, aucun push.

## 6. Verdict baseline

La baseline de code est **unique et non ambiguë** : `5150040` sur
`feat/zegrid-adaptive-resources`, identique à `origin/beta`, arbre suivi propre.
Les environnements d'exécution sont identifiés séparément : Linux
`/usr/bin/python3` (sans CuPy) et venv projet (CuPy 14.1.1/CUDA 12.9/MX150),
Windows `rt-4085889f2625` (CuPy 14.1.1/CUDA 12.9/RTX 3070). Le build Windows réellement exécuté est
connu (`rt-4085889f2625`, ZeMosaic 5.1.1) ; son disque source D: n'est pas
lisible par le node, ce qui bloque uniquement les empreintes du corpus M33
(voir `02_CORPORA.md`), pas l'identification du binaire.
