# 02 — CORPORA

Statut : **M33 partiellement BLOCKED (empreintes), Caldwell COMPLET.**
Aucune exclusion de frame n'a été faite ni proposée. Aucune écriture dans un corpus.

## 1. M33 — dernier run (2026-11-07 12:38:44 → 12:53:52)

### 1.1 Sources et sorties (OBSERVÉ, `run_config_snapshot.json`)

| Champ | Valeur |
| --- | --- |
| Entrée | `D:\M33\S50\IRCUT` (`input_dir = "D:/M33/S50/IRCUT"`) |
| Sortie | `C:\Users\JarvisJunior\Desktop\M33\ircut` |
| Store durable | `C:\Users\JarvisJunior\Desktop\M33\ircut\zegrid-durable` |
| Cache | `C:\Users\JarvisJunior\Desktop\M33\ircut\zegrid-cache` |
| Profil | `processing_mode = zegrid` |
| Scheduler | `zegrid_scheduler_mode = active` |
| Reprise | `zegrid_resume_mode = auto`, `zegrid_stop_capability = zegrid_durable` |
| Halo scientifique | `zegrid_halo_px = 8` |
| Layout | `zegrid_layout = null` (auto) |
| GPU | `use_gpu_stack = true`, `gpu_selector = NVIDIA GeForce RTX 3070 Laptop GPU` |
| Rejet | `stacking_rejection_algorithm = kappa_sigma` |
| Combine | `stacking_final_combine_method = mean` |
| Normalisation | `stacking_normalize_method = sky_mean` (`stacking_normalization_method` vaut `null` et laisse le défaut moteur) |
| Pondération | `stacking_weighting_method = noise_variance` |
| Télémétrie | `enable_resource_telemetry = true`, intervalle 1,5 s, CSV activé |

Empreinte du snapshot : `5AE03ADF…0EB0F04` (voir README).

### 1.2 Inventaire et WCS (OBSERVÉ, `zemosaic_worker.log`)

```
12:38:45  [ZEGRID] ZeGrid: setup - 21818 frame(s) -> C:\Users\JarvisJunior\Desktop\M33\ircut
12:42:25  [ZEGRID] ZeGrid inventory (pass=single): candidates=21818 included=21818 rejected_wcs=0
12:42:25  [ZEGRID] phase END: setup elapsed=219.744s throughput=99.4 frames/s
```

- **N total = 21 818**, inclus = 21 818, **rejet WCS = 0**.
- Aucun filtre n'est appliqué dans ce run : `pass=single`, un seul inventaire.
- Tous les noms échantillonnés par la progression (`layout:footprints`) sont des
  `Light_*_IRCUT_*.fit` (mono-image `M 33` et `mosaic M 33`, 10/20/30 s). Aucun
  `_LP_` n'apparaît dans les 27 échantillons du journal.
- **Ordre** : l'inventaire est ordonné par le moteur, mais la liste ordonnée
  complète n'est pas publiée dans le journal (seulement 27 échantillons).
  **NOT_CAPTURED** — voir §1.5.
- **Empreintes des 21 818 FITS : BLOCKED.** `D:\M33\S50\IRCUT` renvoie
  « Accès refusé » au compte du node (le worker applicatif, lui, a lu les
  fichiers). Commande de résolution en §1.5.

### 1.3 Géométrie (OBSERVÉ, ligne ETA du worker log)

```
12:42:25  [ZEGRID] ZeGrid: ETA context - backend=gpu frames=21818 input_px=45490636800
          stack_work_px=None canvas_px=75698820
```

| Grandeur | Valeur |
| --- | ---: |
| Frames | 21 818 |
| Pixels d'entrée cumulés | 45 490 636 800 |
| Canevas | **75 698 820 px** |
| Layout choisi | **12 × 20 = 240 cellules** |
| Tuile CPU retenue | **20 px** (intérieur) |
| Halo de taper (interne) | 9 px → tuile étendue 38 × 38 |
| Halo scientifique demandé | 8 px |
| Borne streaming@20 | **9 289,1 MiB** |
| Budget PLAN | **9 931,5 MiB** (`B_plan = (4 × 12 414,3) / 5`) |

**N par cellule** : le journal publie `chosen_cells 240/240` mais **pas** les
memberships. Le nombre de contributeurs par cellule n'est donc pas publié par ce
run ; il est **BLOCKED** jusqu'au manifeste de layout/plan. Le modèle du run
implique que presque toutes les 21 818 frames contribuent à la pire cellule,
mais A0 ne le présente pas comme un N_cell mesuré.

Le layout 12×20 a été accepté après un premier scan dense refusé (tous les
candidats jusqu'à 30×51) puis un retry borné unique (journal, `storage profile
AUTO fell back to 'bounded'`). Le run n'a **jamais** atteint jauge, cache,
stack, assemblage ni FITS.

### 1.4 Filtres, WCS et risque de mélange

- Le run ne rejette **aucune** frame (`rejected_wcs=0`) et n'applique pas de
  porte de filtre explicite.
- L'audit antérieur (plan v2, §2.1) signalait « au moins un nom `_LP_` » dans
  un inventaire alors que l'entrée était IRCUT. Si ce fichier appartient bien à
  `D:\M33\S50\IRCUT`, il serait **inclus** par le run actuel.
- A0 ne peut pas trancher : D: est illisible par le node. **Décision requise
  avant A1** :auditer les tokens de filtre de l'inventaire (commande §1.5) ;
  si un mélange est confirmé, **arrêt pour décision humaine**, pas d'exclusion
  silencieuse.

### 1.5 Preuves manquantes et commandes exactes

À exécuter quand D: est accessible (node ou session interactive) :

```powershell
# 1) comptage par extension et par token de filtre
$p='D:\M33\S50\IRCUT'
$f=Get-ChildItem -LiteralPath $p -File
$f.Count
$f | Group-Object Extension | Sort-Object Count -Descending
$f | Where-Object Name -match '_LP_' | Select-Object -First 20 Name,Length
# 2) inventaire ordonné + SHA-256 (écriture HORS corpus)
$f | Sort-Object Name | ForEach-Object {
  [pscustomobject]@{name=$_.Name;bytes=$_.Length;sha256=(Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash}
} | ConvertTo-Json | Set-Content -Encoding UTF8 'C:\Users\JarvisJunior\Desktop\M33\a0_inventory.json'
```

Blocage constaté le 2026-11-07 : `Test-Path`/`dir` sur D: → « Accès refusé »
pour le compte du service node ; ne pas confondre avec une absence de données.

## 2. Réconciliation 22 414 vs 21 818

| Nombre | Origine | Nature |
| ---: | --- | --- |
| 22 414 | audit antérieur, « copie de `D:\M33\S50\out\ircut` sous le Bureau » (plan v2 §2.1) | **entrées d'un arbre différent**, pas le compte du run actuel |
| 21 818 | `zemosaic_worker.log`, 2026-11-07 12:42:25 | **candidats = inclus** du run actuel, `rejected_wcs=0` |

- Delta arithmétique : **22 414 − 21 818 = 596** (2,66 %).
- **Ce delta n'est pas une exclusion du dernier run** : le dernier run déclare
  21 818 candidats et 21 818 inclus. Il ne peut donc pas avoir retiré 596
  frames.
- L'hypothèse « 596 fichiers non-FITS/sidecars dans l'ancien arbre » ou
  « 596 frames d'un autre filtre/une autre nuit » n'est **pas prouvée**.
  Aucune des deux n'est retenue comme fait.
- **Verdict** : la réconciliation est **PARTIELLE et honnête** — les deux
  nombres ne mesurent pas le même arbre. La clôture exige le comptage D: §1.5
  et, si 22 414 est bien un compte de frames, l'explication fichier par fichier.
  Aucune exclusion n'a été faite pour faire passer un test.

## 3. Caldwell (témoin réel disponible localement)

Corpus : `/home/tristan/caldwell11/lights organized/EQ/IRCUT`.

| Champ | Valeur | Source |
| --- | --- | --- |
| Fichiers FITS | **816** | MESURÉ (`caldwell_inventory_summary.json`) |
| Octets | **3 393 936 000** (3,16 GiB) | MESURÉ |
| Premier / dernier (ordre lexical) | `Light_mosaic_C 11_30.0s_IRCUT_20250710-005358.fit` / `…20250802-060327.fit` | MESURÉ |
| Tokens de nom | `Light` = 816, `IRCUT` = 816 | MESURÉ |
| Empreinte de l'inventaire ordonné | `fb36d68355f1c547c653aed50c19c93e5d4cdeeeb9ba7177efc311c04c2a29e8` | MESURÉ |
| Liste détaillée (nom, octets, SHA-256) | `probes/results/caldwell_ircut_inventory.jsonl` | MESURÉ |
| Durée du scan | 13,7 s | MESURÉ |

Aucun fichier hors FITS n'a été pris pour une frame ; les 5 entrées annexes du
dossier (`analyse_resultats.log`, `rejected_low_snr`, `stack_plan.csv`,
`telescopes_pollution.csv`, `ZeAnalyser.marker.json`) sont explicitement
exclues de l'inventaire par extension. Cette exclusion est **de format**, pas
scientifique.

Ce corpus est retenu pour Q62 (grand canevas / halo 128) et comme témoin réel
court. Il n'a pas été exécuté en A0 (pas de run produit).

## 4. Dossiers de sortie neufs

A0 n'a créé aucun dossier de sortie produit. Les seuls chemins écrits sont :

- `docs/qualification/zegrid-adaptive-resources/` (documents et probes) ;
- `probes/results/a0-spool-*` (racine temporaire, supprimée à la sortie du
  contexte ; **0 fichier résiduel**). Le probe de rework, qui utilise une racine
  persistante, observe en plus un dossier cellule **vide** (`a0cell`) laissé par
  la primitive ; il est bénin et le chemin produit `cleanup_after_commit` le
  purge. Voir `04_RESOURCE_CONTRACT.md` §3.4.

Pour A1+, les sorties neuves seront
`/home/tristan/.openclaw/workspace/worktrees/zemosaic-adaptive-resources/.a0-out/<run-id>/`
(Linux) et `C:\Users\JarvisJunior\Desktop\M33\a1-out\<run-id>` (Windows), sans
réutilisation d'un dossier existant.

## 5. Config figée pour les comparaisons

Toute comparaison A1 doit utiliser exactement :

- M33 : les clés du snapshot `5AE03ADF…` (halo 8, rejet `kappa_sigma`,
  combine `mean`, normalisation `sky_mean`, pondération `noise_variance`,
  scheduler `active`) ;
- Caldwell : halo 128 explicite, mêmes rejet/combine/normalisation/pondération,
  sortie neuve, aucune baisse de halo ;
- aucun changement de dtype, halo, rejet, poids, normalisation ou inventaire.

## 6. Verdict corpora

Le corpus M33 est **identifié** (source, N, canvas, grille, halo, config) et son
inventaire est **prouvé non filtré** dans le dernier run. Ses **empreintes**
restent BLOCKED par l'ACL D:. Le delta 596 est **documenté, non expliqué** et
n'autorise aucune conclusion scientifique. Caldwell est **entièrement inventorié
et empreinté**. Aucune frame n'a été exclue.
