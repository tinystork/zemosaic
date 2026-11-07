# A0 — Index des preuves (ZeGrid adaptive resources)

Programme : `ZM-ZEGRID-ADAPTIVE-RESOURCES-20261107`
Tâche : `ZM-ZEGRID-ADAPTIVE-RESOURCES-20261107-A0-T01`
Lot : **A0 — diagnostic et contrats** (aucun code produit modifié)
Date d'exécution : **2026-11-07**, Europe/Paris
Worktree : `/home/tristan/.openclaw/workspace/worktrees/zemosaic-adaptive-resources`
Branche / HEAD : `feat/zegrid-adaptive-resources` / `515004078cfb64b327fb02aeaa323e48cca62bd1`

Statut du paquet : **COMPLET avec preuves manquantes explicitement BLOCKED / NOT_RUN**.
Rework-1 appliqué : deux interpréteurs Linux documentés, smoke CuPy + **30 tests
de parité GPU PASS sur MX150 via le venv**, comptage du résidu spool corrigé
(0 fichier, 1 dossier cellule vide bénin).
Aucun champ critique n'est laissé « à définir » : chaque élément non obtenu est
nommé, avec la commande exacte qui le produira.

## Livrables

| # | Fichier | Contenu |
| --- | --- | --- |
| 1 | [01_BASELINE.md](01_BASELINE.md) | refs/SHA, Git, plateformes Linux/Windows, versions runtime, build réellement exécuté, concurrence |
| 2 | [02_CORPORA.md](02_CORPORA.md) | M33 (21 818 / 22 414), Caldwell 816, config figée, canvas/grille/halos, filtres, sorties |
| 3 | [03_SCIENCE_ORACLE.md](03_SCIENCE_ORACLE.md) | méthodes/combinaisons, plans canoniques, compteurs, règles bitwise et tolérances CPU↔GPU |
| 4 | [04_RESOURCE_CONTRACT.md](04_RESOURCE_CONTRACT.md) | métriques, budgets, unités, réserves, capteur, traces mesurées/modèle/extrapolé |
| 5 | [05_QUALIFICATION_MATRIX.md](05_QUALIFICATION_MATRIX.md) | Q00–Q64, commandes exécutables, quotas, watchdog, seuils |
| 6 | [06_A0_VERDICT.md](06_A0_VERDICT.md) | défaut quantifié, allocations dominantes, chemin minimal, découpage A1, risques |

## Artefacts d'exécution A0

Tous sous
`/home/tristan/.openclaw/workspace/worktrees/zemosaic-adaptive-resources/docs/qualification/zegrid-adaptive-resources/probes/`.

| Artefact | SHA-256 | Nature |
| --- | --- | --- |
| `run_a0_probes.py` | `24a1b9efe8138e50477a3f5aeab4a910c490547e5c1314a4a7df3fafcf452c94` | script de mesure (hors code produit) |
| `results/gpu_smoke.json` | `430bfe3a1e61f8c00480f5fc51a3993de0ab14b54570a8434eea57b2c7961d68` | smoke CuPy venv (MX150), PASS_SMOKE |
| `results/pytest_gpu_parity_venv.txt` | `0b48f57ce47c7ca5d5575077cbdc855e648f2b190ed26613b816cbbe34324a4b` | 30 tests de parité GPU MX150, PASS |
| `results/env.json` | `72100d9808243bb7ad5db7edc874afe40f15d53a4871da623c4495eda6424511` | environnement Linux mesuré |
| `results/model.json` | `72e7ddf31696283963643024ddee3d3c5ee5c3411666f3611d91b023dd6261e8` | pricing M33 par fonctions réelles |
| `results/admission.json` | `215c825f1c40b88f74610f77b9dee73ebe83a7253321241bb4d6267ac18801ee` | reproduction de l'admission jointe du dernier run |
| `results/witness-ext.json` | `5ef7a251afd8153a01b7b2a8677fb0453fb1de84d934717e8c26dcc691d00c9c` | témoin canonique, chemin externe, MESURÉ (rework-1 : comptage résidu exact) |
| `results/witness-inmem.json` | `096c9dd92bd8d7dcfeeb8ac64375fa3bc063ed73bdc218f43714e8797285e242` | témoin canonique, chemin mémoire, MESURÉ |
| `results/pytest_external_stream.txt` | `1883dc9e0a7d41ccf2e86217a56a5d6544a9bf2a35e5d8a2ef273813f9357a6a` | 99 tests de parité/prix externes, PASS |
| `results/caldwell_inventory_summary.json` | `5eb8b676ec711b12770c87ca7b6c5556a8c8c9eac6609655a21d351a724bb0cb` | inventaire Caldwell, 816 FITS |
| `results/caldwell_ircut_inventory.jsonl` | `5d53194fcc4abaa882c04d1318e54d59202674e30f955a47cf2b5e64540b76e4` | liste ordonnée + SHA-256 par fichier |

Les `.err` sont vides (SHA-256 de la chaîne vide
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`), preuve
qu'aucune phase n'a écrit sur stderr.

## Preuves Windows (dernier run M33, 2026-11-07 12:38–12:53)

Chemin hôte : `C:\Users\JarvisJunior\Desktop\M33\ircut\`.
Capturées par le node `Windows Node (MSGE66)` le 2026-11-07 (lecture seule).
Empreintes calculées sur le node avec `Get-FileHash -Algorithm SHA256`.

| Fichier | Octets | SHA-256 |
| --- | ---: | --- |
| `run_config_snapshot.json` | 13 158 | `5AE03ADFF7FAF61C481BA5DA8B6739F6EB9CEE560FB987D93327A44670EB0F04` |
| `zegrid_run.log` | 13 866 | `7470881BB2FE35041C96C4AF8B01E6860EA2C02E3995E0762F648ED8BDF4E6BD` |
| `worker_crash_breadcrumbs.jsonl` | 4 821 030 | `0AACECEB73389F62CBAAB38ED36C25E24698005BC3201ABD08CF09A0E0E9615F` |
| `worker_last_state.json` | 1 677 | `FD9ACE69499D7B3DECF19FF61F0AD5FD4C6D9E6C39E936EF5A5EEE5B6596E98C` |
| `resource_telemetry.csv` | 1 010 | `6A22C661F6BA527B22A99291E2613222A1DB70CDA00A7B0D012DFD594348C98C` |

Le journal complet `zemosaic_worker.log` du run est
`C:\Users\JarvisJunior\AppData\Roaming\ZeMosaic\logs\zemosaic_worker.log`
(45 017 octets, 2026-11-07 12:53:51), non copié intégralement : seules les
lignes citées dans les livrables ont été extraites. Les citations sont
recopiées textuellement dans `02_CORPORA.md` et `06_A0_VERDICT.md`.

## Ce qui n'est PAS prouvé ici

- Un run M33 complet : **NOT_RUN** (le dernier run s'arrête à l'admission, avant jauge/stack).
- Une exécution CUDA **Windows/RTX 3070** : **NOT_RUN** — le run M33 a décliné
  le GPU avant tout kernel. Côté Linux, le smoke et les 30 tests de parité
  passent sur MX150 via le venv ; la VRAM contrainte et l'OOM restent A3.
- Les empreintes des 21 818 FITS M33 : **BLOCKED** (`D:\M33\S50\IRCUT` refuse
  l'accès au compte du node ; le worker applicatif, lui, l'a lu).
- Le comptage 22 414 vs 21 818 : **PARTIEL** — le dernier run prouve
  21 818 candidats / 21 818 inclus / 0 rejet WCS ; les 22 414 proviennent d'un
  inventaire antérieur distinct et ne peuvent pas être rejoués sans D:.

## Note L3 — valeurs `admission` codées en dur

La phase `admission` de `run_a0_probes.py` recopie des valeurs du run journalisé
(canevas, cellules, budget, `stream@20`) au lieu de parser le snapshot. C'est
accepté pour A0 parce que chaque valeur est ancrée par les SHA-256 Windows
ci-dessus. **A1 doit lire directement `run_config_snapshot.json` et
`resource_telemetry.csv`** (ou un fichier de rejeu versionné) ; aucune conclusion
ne doit être rejouée sur des littéraux périmés.

## Règle de lecture

Toute affirmation de ce paquet est étiquetée :

- **MESURÉ** : produit par une commande de ce paquet, sortie conservée ;
- **OBSERVÉ (log)** : texte présent dans un artefact Windows cité ;
- **MODÈLE** : fonction produit exécutée sans pixels ;
- **EXTRAPOLÉ** : calcul explicite à partir des deux précédents ;
- **BLOCKED / NOT_RUN** : preuve absente, jamais promue en PASS.
