# 03 — SCIENCE_ORACLE

Statut : **COMPLET et renforcé par le rework-1** pour le contrat CPU déterministe
et les tolérances CPU↔GPU existantes. La parité GPU a été **exécutée sur MX150
via le venv Linux : 30 tests PASS** ; aucun test VRAM-contrainte/OOM n'a été
lancé. Le run Windows a décliné le GPU avant tout kernel.
Aucun seuil n'a été inventé : toutes les valeurs ci-dessous sont lues dans le
code ou les tests du worktree `5150040`.

## 1. Surface scientifique exposée

### 1.1 Rejet et combinaison

`src/zemosaic/core/canonical_stacking.py` :

| Paramètre | Valeurs acceptées | Défaut moteur |
| --- | --- | --- |
| `rejection` | `none`, `kappa_sigma`, `winsorized_sigma_clip` | `kappa_sigma` (`science_adapter.py:51`) |
| `combine` | `mean`, `median` | `mean` (`science_adapter.py:52`) |

Les alias et inconnus sont refusés explicitement
(`canonical_stacking.py:1546–1577`). `linear_fit_clip` est retiré et refusé.
`kappa_sigma` et `winsorized_sigma_clip` honorent les paramètres sigma/winsor
de l'utilisateur (`science_adapter.py:80`).

### 1.2 Normalisation, pondération, taper

- Normalisation : `sky_mean` (défaut) ; contrat canonique B1.
- Pondération : `noise_variance` ; support géométrique et poids par frame.
- Taper : `footprint` (défaut configuré) ou `none` ; halo de taper
  `_TAPER_HALO_PX = 9` (`canonical_streaming.py`, `streaming.py:164`).
- Aucun de ces paramètres n'est modifié par A0.

### 1.3 Plans canoniques

Trois plans équivalents doivent rester comparables :

1. **mémoire** : `zemosaic.core.canonical_engine.run_canonical_stack` ;
2. **streaming spatial** : `canonical_streaming.run_canonical_stack_streaming` ;
3. **externe (spool + colonnes)** : `zegrid.external_stream.run_cell_streaming_external`
   (CPU uniquement aujourd'hui, `external_stream.py:1359–1366`).

Le contrat A0 exige que (2) et (3) restent **bit-identiques** à (1) sur les
témoins, et que (3) conserve tous les contributeurs N.

### 1.4 Sorties et compteurs

`CanonicalStackResult` (`canonical_engine.py:149–179`) expose :

- plans : `science`, `estimator_weight_sum`, `valid_mask`,
  `surviving_sample_count`, `support_w1`, `support_w2`, `n_eff_support`,
  `rejection_mask` ;
- scalaires : `iterations_used`, `initial_sample_count`,
  `rejected_sample_count`, `rejected_fraction`, `low_n_cell_count`,
  `degenerate_cell_count`, `rejection_method` ;
- provenance bornée : référence, taper, equalize, frames exclues, événement.

Ces champs constituent l'oracle comparable ; les fichiers FITS ne le sont pas
(timestamps, en-têtes variables).

## 2. Règle CPU déterministe

**Règle** : à entrée, config, N et ordre identiques, tous les plans et scalaires
comparables sont **bit-identiques** (`np.testing.assert_array_equal`, pas de
tolérance), y compris les masques booléens et les compteurs entiers.

Sources :

- `tests/test_zegrid_m2_external_stream.py` : `_assert_bit_exact` compare les
  8 plans avec `np.array_equal(..., equal_nan=True)` et les 7 scalaires par
  égalité stricte, plus `provenance` (lignes 110–126) ;
- `tests/test_sci05_canonical_streaming.py` : `_assert_bit_exact` utilise
  `np.testing.assert_array_equal` sur `science`, `estimator_weight_sum`,
  `valid_mask`, `surviving_sample_count`, `support_w1`, `support_w2`,
  `n_eff_support`, `rejection_mask` (lignes 68–77).

Aucune tolérance CPU n'est autorisée. Si le chemin mémoire diverge, c'est un
défaut, pas un seuil à ajuster.

### 2.1 Preuve MESURÉE dans A0

Témoin : 48 frames RGB 64×64, gains variables, NaN injectés, outlier 1e6,
masques partiels, colonnes à faible N ; `winsorized_sigma_clip` + `mean` +
`taper=footprint` ; deux **processus séparés** (mémoire vs externe tile 16,
`column_batch=1024`).

| Métrique | Mémoire | Externe | Égal ? |
| --- | --- | --- | --- |
| `science` SHA-256 | `266a0d2b…1edae7d` | `266a0d2b…1edae7d` | **OUI** |
| `surviving_sample_count` SHA-256 | identique | identique | **OUI** |
| `iterations_used` | 5 | 5 | OUI |
| `initial_sample_count` | 557 301 | 557 301 | OUI |
| `rejected_sample_count` | 5 205 | 5 205 | OUI |
| `rejected_fraction` | 0,00933966 | 0,00933966 | OUI |
| `low_n_cell_count` / `degenerate_cell_count` | 0 / 0 | 0 / 0 | OUI |
| temps | 0,472 s | 0,902 s (1er run), 1,497 s (rework, cache froid) | — |
| pic `VmHWM` | 172 253 184 B | 145 276 928 B | — |
| delta pic au-dessus des tableaux | 30 937 088 B | 3 862 528 B | — |

Artefacts : `probes/results/witness-inmem.json`, `witness-ext.json`.
Cette double exécution prouve aussi la **déterminisme inter-processus** de la
baseline sur ce témoin.

### 2.2 Suite de tests exécutée

```
PYTHONPATH=src python3 -m pytest -q tests/test_zegrid_m2_external_stream.py \
  -k "parity_matrix_rgb or m33_pricing or bounded_and_patch_independent or larger_n_stays_bit_exact"
→ 99 passed, 28 deselected in 32.04s (exit 0)
```

Couverture : tiles {32,24,16,10} × rejet {none,kappa,wsc} × combine {mean,median}
× taper {footprint,none} × `column_batch` {8,1024}, plus N plus grand à batch
fixe, indépendance du workspace à la surface du patch, et pricing M33.
Artefact : `probes/results/pytest_external_stream.txt`.

## 3. Règle CPU ↔ GPU

**Tolérance existante, citée sans invention** :

```
tests/test_sci05_canonical_gpu_parity.py:14  "* science / estimator_weight_sum parity within rtol=1e-12"
tests/test_sci05_canonical_gpu_parity.py:35  SCIENCE_RTOL = 1e-12
tests/test_sci05_canonical_gpu_parity.py:116 np.testing.assert_allclose(
                                                 cpu.science, gpu.science,
                                                 rtol=SCIENCE_RTOL, atol=0.0,
                                                 equal_nan=True)
```

- `rtol = 1e-12`, `atol = 0.0` pour `science` et `estimator_weight_sum` ;
- **masques de rejet et de validité : égalité booléenne exacte** (bit-identique) ;
- **diagnostics entiers : égalité exacte** ;
- `tests/test_sci05_canonical_engine.py:428` utilise aussi `rtol=1e-12` pour la
  comparaison CPU/GPU du moteur complet.

Justification documentée dans le test : une réduction arborescente GPU peut
différer de la sommation pairwise NumPy d'environ 1 ULP ; 1e-12 est une marge
×1000, pas un seuil arbitraire.

### 3.1 Statut GPU A0

- **Linux, venv projet** (`/home/tristan/.openclaw/workspace/projects/zemosaic/.venv/bin/python`) :
  CuPy 14.1.1, CUDA runtime 12 090, 1 device MX150 2 GiB. Smoke :
  `cuda_device_count=1`, `memGetInfo` free 2 050 621 440 B / total 2 091 253 760 B,
  allocation 64 MiB + `sum` exacte → **PASS_SMOKE** (`gpu_smoke.json`).
- **Parité GPU Linux, MESURÉE** :
  `pytest -q tests/test_sci05_canonical_gpu_parity.py` avec le venv →
  **30 passed, 2 warnings in 4.87 s, exit 0** (`pytest_gpu_parity_venv.txt`).
  Cela couvre les masques exacts, les diagnostics et `rtol=1e-12` sur MX150.
- **Linux, `/usr/bin/python3`** : sans CuPy ; les tests lancés avec cet
  interpréteur se sont auto-skippés, ce qui ne prouve rien sur le GPU.
- **Windows** : CuPy 14.1.1 / CUDA 12.9 présents, mais le run M33 a **décliné**
  le GPU (`R55_REFUSE_VRAM`) avant jauge/stack ; **aucun kernel n'a été exécuté**.
- **Non couvert** : VRAM volontairement contrainte, OOM injecté, RTX 3070,
  exécution scientifique M33 sur GPU. Ces points restent A3/A6.
- Aucune nouvelle tolérance n'a été proposée après observation d'un échec.

## 4. Comparaisons autorisées pour A1+

| Comparaison | Règle | Interdit |
| --- | --- | --- |
| CPU mémoire vs CPU streaming | bitwise sur les 8 plans + scalaires | toute tolérance |
| CPU mémoire vs CPU externe | bitwise, même N, même ordre | comparer des FITS |
| CPU vs GPU | `rtol=1e-12`, `atol=0`, masques exacts | nouveau seuil après échec |
| Avant/après refactor CPU | bitwise | moyenne de sous-stacks |
| N partiel (100k borné) | positions/régions vs oracle + intégrité globale | prétendre bitwise intégral |

## 5. Témoins à figer avant A1

1. **W1** — le témoin 48×64×64 ci-dessus (SHA science
   `266a0d2b43ef061048cfc66b35a6caf45a92976981c68fca47e2ec50c1edae7d`) :
   non-régression immédiate, < 2 s.
2. **W2** — M33 cellule réelle profonde (Q32) : à extraire en A1 avec N réel et
   config gelée ; oracle CPU mémoire sur la cellule, pas sur 21 818 frames.
3. **W3** — Caldwell halo 128 (Q62) : oracle CPU sur une cellule, puis run
   complet en A6.
4. **W4** — M106/M16 (Q63) : non-régression produit.

Les SHA de W1 sont dans `probes/results/`. W2–W4 sont **NOT_RUN** en A0 et
doivent être produits avant leur lot, jamais simulés.

## 6. Verdict science

Le contrat CPU est **exact et prouvé** sur un témoin adversarial et 99 tests
ciblés. Le contrat CPU↔GPU est **cité avec ses valeurs numériques et ses tests
sources** (`rtol=1e-12`, masques/diagnostics exacts) mais **non exécuté** dans
A0. Aucun seuil nouveau n'a été créé. La condition « oracle figé avant
modification » est remplie pour W1 ; W2–W4 restent à produire aux lots
concernés, avec statut NOT_RUN explicite.
