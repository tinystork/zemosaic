# M33 preparation pipeline — état au 2026-11-25 (rework-1 inclus)

Mission `ZM-M33-PREP-PIPELINE-20261125`. Branche `perf/m33-preparation-pipeline`,
baseline/HEAD `0d4e37abcb200583705951d988152a3a764ee7e3` (= `origin/beta`).
Implémentation Coco + corrections REWORK1 (B1–B7), revue Junior/Nono à suivre.
**Aucun commit/push/install, pas de publication : ce document décrit un état de
travail, pas une clôture.**

## 1. Ce qui est livré dans le diff

| Fichier | Rôle |
|---|---|
| `src/zemosaic/core/zegrid/prepare_pipeline.py` (nouveau) | registre CFA cell-scope single-flight, budget constructif, ledger de crédits **octets**, pipeline borné 2 producteurs + writer unique, drain/quarantaine. |
| `src/zemosaic/core/zegrid/external_stream.py` | helper **unique** `prepare_external_frame` (casts/ordre/maths/taper/NaN inchangés), `TileWriteSession` (géométrie + memmaps + publish/abort autoritatifs), `write_tile` legacy réécrit dessus, injection facultative `provider_factory` + `preparation_budget`, `tile_gate` coopératif, publication des stats. |
| `src/zemosaic/zemosaic_zegrid_mode.py` | `_CellSourceContext.open_provider(cfa_registry=…)` + factory explicite, sélecteur interne `ZEMOSAIC_M33_PREP_PIPELINE` (`auto` défaut / `legacy` contrôle), admission au boundary cellule, télémétrie fusionnée. |
| `tests/test_zegrid_prepare_pipeline.py` (nouveau) | 23 tests scientifiques/adversariaux. |
| `.tmp/openclaw-spikes/m33-prep-pipeline/witness_integrated.py` | témoin intégré jetable (non commité) appelant le **vrai** `run_cell_streaming_external`. |

### Contrat scientifique préservé

`prepare_external_frame` est le seul corps par-frame. Il conserve exactement :

* `a = p1.coefficients[i]` puis `a[c,0]*src64 + a[c,1]` en float64,
* cast float32 sous `errstate(over/invalid=ignore)`,
* `post_valid = pre_valid & all(isfinite(f32))`,
* `frame = NaN hors valid`, taper footprint / ones, `weights = valid ? taper : 0`,
* frame inactive ⇒ images NaN / weights 0 (écriture scalaire historique).

Le writer écrit `images[i]`/`weights[i]` à l'index **original**; une fin hors ordre
ne peut donc pas permuter une contribution. La réduction, le spool et le format
sont inchangés.

## 2. Budget et admission (constructif, pas RSS seul)

`preparation_grant_components(...)` nomme chaque terme :

* fenêtre provider (`patch_px*(4c+1)`), décodage pleine image de fallback
  (brut + f64 + f32 HWC), prescan CFA (chunk float64+f32 ≈ 48 MiB), section CFA
  réelle (`ROI source + marge 4 px`, 3×8+3×4 B/px), source reprojetée float64,
  coordonnées, footprint, sortie f32, scratch natif déclaré, chaîne transform
  f64/f32, frame/poids/taper de tuile, métadonnées **O(N)** (2 MiB + 1 KiB/frame).

Le ROI source est celui **réellement planifié** (max des `source_bounds`), pas
`tile+10` : une section légitime peut couvrir toute la frame. Si le ROI est
inconnu, `unqualified_terms=["source_roi_unknown"]` force le séquentiel.

`peak_bytes` = sous-total + **majoration 125 %** (facteur 2,25 documenté, ce
n'est pas 25 %) + résidu fixe 32 MiB. C'est une **borne constructive owned
d'ADMISSION**, pas un maximum mesuré. Domaines publiés séparément :

* `components` — termes owned calculables ;
* `empirical_terms` — scratch natif, heap retenu et **libération GC différée**
  (tableaux owned), couverts par la majoration déclarée (empirique, pas une
  preuve de majorant) ;
* `non_owned_terms` — page-cache OS, hors du domaine owned ;
* `unqualified_terms` — entrées non chiffrables ; toute entrée force legacy.

Le cache RAM de frames décodées est **désactivé** dans les providers du pipeline
(et libéré sur le provider de base) ; les sections source sont relues et le
prescan CFA est partagé.

`decide_preparation_pipeline` résout une fois :

* `legacy` demandé ⇒ séquentiel (contrôle A/B) ;
* `existing_commitment_bytes` inconnu ⇒ `committed_budget_unknown` ⇒ séquentiel
  (**fail-closed**, jamais supposé 0) ;
* `unqualified_terms` non vide ⇒ `unqualified_budget_term:<terme>` ⇒ séquentiel ;
* probe indisponible ⇒ `host_available_unknown` ⇒ séquentiel ;
* `2*grant + committed + 512 MiB > available` ⇒ séquentiel, raison exacte
  (`insufficient_host_budget_for_two_producers` ou `..._even_single`) ;
* sinon `auto` 2 producteurs.

**Révalidation aux frontières de tuiles** : à chaque tuile >1, APRÈS la
réduction précédente et AVANT tout nouveau provider/producteur, le gate relit
le probe live et rejoue la décision (domaines connus + `committed`). Un échec
(probe inconnue, budget 2 insuffisant, terme non qualifié, exception) rend le
fallback **sticky** : les tuiles restantes utilisent le writer séquentiel
historique, mêmes plans/mêmes spools. Statistiques publiées : `tiles_pipeline`,
`tiles_legacy`, `gate_probes`, `fallback_at_tile`, `sticky_fallback_reason`,
`effective_mode_last`. Ce n'est pas une borne RSS globale ; c'est une
revalidation ponctuelle par frontière, avec réserve hôte 512 MiB.

`committed` = plans résultats persistants déjà nécessaires à la cellule. Aucune
réserve de réduction n'est diminuée, `column_batch`/K/concurrency ne changent pas.
Un grant supérieur à la capacité du ledger est refusé avant toute allocation
(pas de `max(1)` masquant).

## 3. Cycle de vie, erreurs, quarantaine

* **Grants avant ouverture** : chaque provider est couvert par un crédit acquis
  AVANT l'appel à la factory ; le producteur consomme ce crédit pour sa première
  frame, puis acquiert par refill. Capacity = `producers*grant`, refus si un
grant dépasse la capacité.
* Providers privés, fermés **dans le producteur avant le join**, donc avant la
  réduction. Un échec de `close()` fait échouer le pipeline (jamais de succès
  silencieux avant reduce).
* Crédit libéré **après** effacement du payload (writer et drain), et le
  producteur supprime ses réf. locales après le `put` : le ledger borne
  réellement les payloads vivants.
* **Contrôle complet avant publication** : `written == N`, `prepared == N`,
  0 drop, chaque index écrit exactement une fois. Sinon `PreparationAborted` et
  `session.abort()` ; aucune tuile partielle n'est publiée.
* Erreur producteur/writer/close : admission fermée, items en attente
  annulés/libérés, producteurs drainés, **exception originale** relancée.
* Drain incomplet : `PreparationQuarantineError` (cause chaînée), providers non
  fermés de force, tuile retenue, cellule échouée, objet pipeline non
  réutilisable (garde `_consumed`/`_quarantined` avant factory). Politique
  inter-cellules : providers par tuile, donc pas de réemploi ; la cellule échoue.
* Mappings fermés explicitement (`flush` + `mmap.close`) avant `rmtree`, sur
  succès comme sur abort/échec de publish/allocation. Aucun `del` seul.
* Le spool n'est pas nettoyé quand la quarantaine est active (fichiers
  potentiellement encore référencés).
* Heartbeat borné pendant le write : le writer appelle le `progress_probe`
  existant toutes les ~N/16 frames (max 256) avec prepared/written/total, bytes
  réservés/peak, producteurs effectifs ; exception callback avalée, science
  intacte. Ces champs traversent `_live_progress` -> `CellStatusWriter` ->
  journal -> `LiveTelemetryReader` -> `LiveCellStates`/`live_cell_line` ; la clé
  de déduplication inclut `frame_current/frame_prepared/bytes_*` et la ligne
  détail affiche `frame`, `prepared`, `bytes_reserved`, `producers`. Le
  rate-limit temporel reste actif (pas de `force=True` par frame). Aucune
  refonte ETA.
* **Invalidation aux transitions** : un changement effectif de phase ou de tuile
  efface `frame`, `frame_current`, `frame_total`, `frame_prepared`,
  `bytes_reserved`, `bytes_peak`, `producers_effective` (comme `block`), puis
  applique les champs du nouvel échantillon. Un échantillon partiel de même
  phase/tuile conserve le contexte. Le chemin write legacy publie
  `producers_effective=1` pour la tuile courante (jamais un `2` hérité), et
  aucune valeur de préparation ne survit pendant `reduce`.

## 4. CFA cell-scope

`CfaMinMaxRegistry` publie un tuple immuable `(min,max)` pour une clé
`(chemin source, header_sha256, shape, politique decode)`. `get_or_compute`
est single-flight : un seul prescan plein-frame payé, les autres providers
attendent. Aucune image/WCS mutable partagée ; compteurs `prescans` (payés) et
`prescans_shared` (réutilisés) publiés.

## 5. Sélecteur et visibilité

`ZEMOSAIC_M33_PREP_PIPELINE=auto|legacy` (défaut `auto`, jamais un réglage GUI).
La décision, la raison, le grant, les composants et le probe source sont publiés
dans `storage.external.preparation_pipeline`. Les compteurs source (`source_window`)
somment le provider de base et les providers de tuile : un mode AUTO ne publie
plus un `window_prepared_count` trompeusement nul. Les logs bornés indiquent
producteurs effectifs, grant, available/committed, prepared/written, bytes
réservés/prêts, prescans, attente ledger, drain/quarantaine.

## 6. Preuves exécutées (Linux, hôte partagé)

Commandes exactes et résultats observés le 2026-11-25 :

```bash
python -m pytest tests/test_zegrid_m2_external_stream.py \
  tests/test_zegrid_external_lifetimes.py tests/test_zegrid_external_gather.py \
  tests/test_zegrid_m33_windowed_source.py tests/test_zegrid_m33_cfa_bounded.py \
  tests/test_zegrid_m4_external_durability.py tests/test_zegrid_prepare_pipeline.py \
  tests/test_zegrid_active_external_wiring.py tests/test_sci05_canonical_streaming.py -q
# 361 passed, 8 skipped (148.20 s) — R2

# REWORK3 (O1) : rejeu ciblé imposé
python -m pytest tests/test_zegrid_live_obs_l2.py \
  tests/test_zegrid_prepare_pipeline.py -q
# 79 passed (53.80 s)

python -m pytest tests/test_zegrid_a2b_source_plumbing.py -q \
  -k "not external_caller_publishes_gpu_fallback"
# 10 passed, 1 skipped, 1 deselected (8.06 s)

git diff --check
# exit 0
```

Le test `test_external_caller_publishes_gpu_fallback` échoue **aussi sur un
worktree baseline intact** (`backend 'gpu' … CuPy/GPU is unavailable`) : limite
de l'hôte, pas une régression. De même, `tests/test_zegrid_m3_working_set.py`
a 3 échecs identiques sur baseline (facteur d'enveloppe 9/10 vs valeur attendue
du test), non causés par ce diff.

Couverture du nouveau fichier (43 tests) : parité legacy/canonique RGB+mono sur
≥12 tuiles, CFA GRBG/RGGB/GBRG/BGGR, mono, fallback CFA inconnu pleine image,
coefficients linear_fit non triviaux, frame inactive, NaN loader, ordre inversé
par events, barrière réelle 2 producteurs, writer = thread appelant, weakrefs
payloads relâchés à chaque refill, ledger insuffisant, décision de budget
(unqualified/committed inconnu), marge 2,25 documentée, metadata O(N), erreur
producteur/writer/close, stop avant et pendant sans publication, échec de
publish/flush et seconde allocation avec mappings fermés, réemploi refusé,
heartbeat borné, quarantaine, single-flight CFA, sélecteur produit et **chemin
caller réel** `_run_cell_external` AUTO vs legacy avec parité.

### Témoin intégré NGC7023

**Portée** : ce témoin r0 appelle le vrai `run_cell_streaming_external` mais
calcule lui-même le budget et injecte `provider_factory`/`preparation_budget`.
Il ne traverse donc **pas** le nouveau sélecteur/admission (`_resolve…`,
`committed`, probe). Les chiffres ci-dessous restent la preuve du **pipeline
configuré**, pas du nouveau budget/admission ; le chemin AUTO produit est
couvert par le test caller réel sur petit corpus (AUTO vs legacy, parité et
`committed` inconnu ⇒ legacy). Aucune nouvelle campagne n'est lancée pour R1.

Cellule acceptée `r0000c0007` (447 contributions, patch 554×358, 6 tuiles,
tile 192/halo 32/batch 4096), référence `/home/tristan/NGC7023/out` en lecture
seule, scratch ext4 isolé dans le checkout (`/tmp` interdit). Deux paires
alternées OFF/AUTO puis AUTO/OFF, **667.9 s** au total (budget 1080 s).

| Paire | Mode | Écriture (s) | Réduction (s) | Cellule (s) |
|---|---|---:|---:|---:|
| 1 | legacy | 156.444 | 44.787 | 201.264 |
| 1 | auto | 95.225 | 44.532 | 139.796 |
| 2 | auto | 98.212 | 45.006 | 143.255 |
| 2 | legacy | 128.359 | 43.838 | 172.242 |

Ratios appariés convention `legacy/auto` :

* paire 1 : préparation **1.643×**, réduction 1.006×, cellule **1.440×** ;
* paire 2 : préparation **1.307×**, réduction 0.974×, cellule **1.202×**.

Médianes : préparation ≈ **1.475×**, cellule ≈ **1.321×**. La cible locale
(préparation ≥1.25× **et** cellule ≥1.25×) est atteinte en médiane et sur la
paire 1, **pas** sur la cellule de la paire 2 (1.202×). Les deux paires sont des
mesures locales quasi-chaudes, non causales, avec cache/pression hôte
confondants ; aucun gain M33/Windows n'est revendiqué.

Parité : les **7 plans core historiques** (science, weight_sum, support_w1,
support_w2, n_eff reconstruit, valid_mask, surviving) sont **bit-exacts** sur
chaque run (hashes des objets référence vérifiés avant lecture) et entre tous
les runs. 6/6 tuiles, 447/447 contributions, 2682 préps/écritures par run AUTO,
0 drop, 12 ouvertures/fermetures de providers par cellule, aucun drain incomplet.

Mémoire : RSS HWM **695 160 KiB = 678.87 MiB (~679 MiB)** (erratum r1 :
« 695 MiB » était une confusion KiB/MiB), max échantillonné 695 828 KiB,
min MemAvailable **1 783 916 KiB (~1.70 GiB)** ; plafond témoin 1 GiB et réserve
hôte 512 MiB respectés. Le pic de crédits ledger observé est 833 547 118 B
(2 × grant), borne d'admission et non RSS. Aucun OOM, aucun stop guard.
Référence et inventaire source inchangés avant/après.

Limites : Linux seulement (Windows non exercé, aucun runner réel dans cette
mission) ; une cellule de bord, un hôte, pas de preuve de borne mémoire dure ni
d'attribution causale ; `RSS` échantillonné n'est pas un majorant garanti.

## 7. Garanti vs estimé

**Garanti par le code/tests** : parité bit-exacte du spool et du résultat pour
les cas exercés ; writer unique sur le thread appelant ; index jamais permuté ;
publication seulement après N/N frames écrites une fois ; stop/erreur ⇒ aucune
tuile publiée ; mappings fermés explicitement avant suppression (succès, abort,
échec d'allocation/publish) ; providers fermés avant réduction en succès ;
un échec de close échoue le pipeline ; exception originale conservée ; crédits
acquis avant factory et payloads effacés avant release ; objet pipeline
mono-usage/quarantaine refusé avant allocation ; refus explicite des crédits/
budgets insuffisants ; `committed` inconnu ⇒ legacy ; `unqualified_terms` ⇒
legacy ; prescan CFA payé une fois par source et par cellule ; fallback
séquentiel motivé et visible ; heartbeat borné non fatal.

**Estimé / non qualifié** : borne constructive du grant (majoration 125 %
déclarée, scratch natif/heap retenu empiriques non mesurés), pics mémoire hors
cas témoin, gain de performance hors paire mesurée, comportement Windows
(handles/mappings/threads), comportement sous pression disque/OS réelle,
témoin r0 non représentatif du nouveau sélecteur.

## 8. Acceptation Junior — 26 novembre 2026

Lot accepté pour publication beta. Revue indépendante Nono R2 : tous les
points science/lifecycle/admission clos, seul O1 d'affichage périmé restant.
O1 corrigé en R3 et vérifié indépendamment par Junior : write→reduce nettoie
les compteurs de préparation ; write legacy suivant indique producers=1 ;
les échantillons partiels conservent leur contexte.

Validation finale Junior : 79 tests pipeline/live_obs passés en 54,86 s,
171 avertissements numériques de soustraction sur les cas non finis.
Validation élargie R2 : 361 passés, 8 skips (Coco et Nono) ; contrôle Junior
R2 : 132 passés. Diff-check propre. Aucun benchmark supplémentaire.
Erratum mineur du rapport R3 : 41→43 cas pytest, soit deux tests ajoutés
couvrant plusieurs assertions/transitions, pas trois nouveaux cas.

Livraison : AUTO deux préparations si admission mémoire, sinon séquentiel ;
writer unique, revalidation entre tuiles et repli sticky. Pas de réglage GUI,
pas de changement de jauge/identité/science/reprise ni de concurrence des
réductions. Le contrôle interne legacy reste disponible pour diagnostic.

La revue Nono R2 n'est pas un ACCEPT R3 : la clôture O1 et l'acceptation finale
sont celles de Junior. Aucun PASS Windows : mappings testés sous Linux ; le
budget conserve une marge empirique et n'est pas un majorant RSS universel.
Les temps NGC r0 ne sont pas une promesse de gain M33 ni une mesure du nouveau
sélecteur. La publication n'installe rien sur le Windows actuellement actif.

Prochain pas : attendre la fin de cellule souhaitée par Tristan, puis installer
la beta identifiée et reprendre M33 sous contrôle, en vérifiant jauge réutilisée,
cellules déjà committées conservées, mode de préparation effectif et mémoire.
La décision d'arrêter/reprendre le run Windows appartient à Tristan.
