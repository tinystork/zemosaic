# Phase 2 produit — ZM-LAYOUT-PRODUCT-20261115

- [x] Autorisation Tristan : intégration, publication beta après gates, test ZeAlfie → ZeMosaic Windows.
- [x] Baseline a59e72c / beta 5.1.4 ; préexistants documentaires préservés.
- [x] Corpus indiqué : D:\SH2-101_mosaic_sub\lights ; listing lecture seule 3188 fichiers / 13248817920 octets (FITS à confirmer).
- [x] Coco précédente mission close, reset nouvelle mission réussi compteurs zéro.
- [x] Implémentation bornée du chooser réel, parité et performance locale (r0).
- [x] Rework-1 après revue Nono (findings M1/M2/M3, L1–L4) :
      - **M1** protocole Windows §6 supprimé/requalifié : `zegrid_cli` lance le moteur COMPLET, aucun point d'entrée décision-seule, aucun nouveau CLI.
      - **M2** protocole §2 corrigé : log worker par utilisateur (`%APPDATA%\ZeMosaic\logs\zemosaic_worker.log`, `mode='w'`) → troncature à tout nouveau lancement ; séparation des sorties insuffisante ; sérialisation stricte + archivage du log avant tout autre lancement ; implémentation logging non modifiée.
      - **M3** protocole §3 réécrit : sortie/provenance/chemins/timings/diagnostics additifs volontairement différents ; budget RAM effectif explicite identique obligatoire, avec règle de récupération pour une baseline `auto` (pin sinon comparaison observationnelle/confondue, jamais un faux échec de parité) ; Tristan lance le candidat, Junior ne lance jamais.
      - **L1** diagnostic `layout index:` dans le vrai journal d'exécution (build + chaque sortie normale/refus, WARN si dégradé), hors lignes de décision et hors dict ; 4 tests dédiés (fallback, query_failures, refus, invariance de décision).
      - **L2** `rss_peak_bytes` → `rss_after_bytes_max` (mêmes nombres, probe + evidence) ; revendication corrigée en « RSS post-appel », pas un pic transitoire ; pas de benchmark relancé.
      - **L3** revendication d'équivalence d'erreur restreinte ; test bowtie (bbox chevauchante → même exception ; bbox disjointe → comptage exact sans levée) ; contrat non-fini/exact préservé.
      - **L4** rétention O(cellules) documentée et mesurée : 67x39 = 2613 cellules ≈ 1,22 MiB d'objets Python retenus (micro-sonde bornée, evidence/retained_cells.json).
      - README qualification mis à jour : 10 chemins de parité (et non 7), résolutions de revue, limites restantes.
- [x] Vérification Junior du rework-1 : diff réel et protocole inspectés, `git diff --check` OK ; suite produit indépendante **62 passed in 87.68s** ; 5 signatures baseline/candidate identiques et libellés RSS vérifiés. Préexistants préservés, aucune publication ni action Windows.
- [x] Revue Nono review-1 **ACCEPT**, rapport `.a2a-reports/ZM-LAYOUT-PRODUCT-20261115.nono.r1.md` : M1–M3 et L1–L4 résolus. Vérification Junior des sources et preuves effectuée ; acceptation locale, qualification Windows encore NON EXÉCUTÉE.
- [x] Lot beta **5.1.5** accepté pour publication autorisée : 62 tests produit Junior, 5 signatures identiques, Nono ACCEPT ; gate post-bump packaging/titre Qt **20 passed in 8.72s**. SHA et résultat du push consignés dans le rapport final Junior de mission.
- [ ] Qualification Windows par chemin réel ZeAlfie ; sorties isolées, mêmes réglages effectifs, baseline et candidat.

## Artefacts Windows fournis par Tristan (2026-11-15)

- Référence (beta 5.1.4) : `D:\SH2-101_mosaic_sub\out\newgauge` — **lecture seule**, jamais modifiée, jamais relancée.
- Candidat : `D:\SH2-101_mosaic_sub\out\newgaugev2` — sortie **volontairement distincte**, run lancé **par Tristan** via ZeAlfie/ZeMosaic après publication beta.
- Coco ne lance aucun run Windows, ne met pas à jour le runtime, ne copie pas le corpus ; Junior compare ensuite les deux arbres en lecture seule.
- Protocole de comparaison (corrigé M1/M2/M3) : `docs/refactor/ZEGRID_LAYOUT_PRODUCT_WINDOWS_AB_PROTOCOL_20261115.md`.

M33 actif D:\M33\S50\out\fresh à protéger : aucun second run concurrent, et le log worker global (`mode='w'`) est tronqué par tout nouveau lancement → archivage obligatoire avant tout autre run.
Rapport : /home/tristan/.openclaw/workspace/.a2a-reports/ZM-LAYOUT-PRODUCT-20261115.coco.r1.md (r0 conservé pour historique).
Implémentation ACCEPT Junior après revue finale Nono ; lot beta 5.1.5 validé pour le push autorisé. Qualification Windows réservée à Tristan, puis comparaison en lecture seule par Junior.
