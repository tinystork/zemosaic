# A2 — Audit cache aligné → fenêtre/spool borné

Mission : `ZM-ZEGRID-ADAPTIVE-RESOURCES-20261107-A2-T01` (audit lecture seule)
Baseline : `bf0f832`, branche `feat/zegrid-adaptive-resources`, arbre propre.
Statut : **AUDIT TERMINÉ — aucun code produit modifié**.

## 1. Cartographie du cache aligné

| Élément | Fichier / lignes | Rôle |
| --- | --- | --- |
| `_build_aligned_cache_frame_major` | `zemosaic_zegrid_mode.py:1005–1104` | construit `<cache_root>/<cell_id>/frame_XXXX_rgb.npy` + `_support.npy` pour **chaque frame de la cellule**, écrit un manifest, réutilise si complet |
| `AlignedCacheBuilder` / `build_aligned_cache_from_sources` | `file_provider.py:324–475` | écriture et manifeste du cache par frame |
| `_r55_cache_unit_bytes` | `zemosaic_zegrid_mode.py:1185–…` | borne exacte `N × patch_px × (4×C + 1) + en-têtes` ; C=3 → **13 B/px/frame** |
| `_make_cell_provider` | `zemosaic_zegrid_mode.py:2351–2361` | choisit `MemmapCanonicalProvider` (dense) ou `BoundedMemmapCanonicalProvider` |
| `_run_cell_inmem` / `_run_cell_stream` / `_run_cell_external` | `:2364` / `:2412` / `:2523` | consomment ce provider |
| `estimate_external_working_set` | `external_stream.py:1004` | prix du spool par tuile et de la réduction |
| `run_cell_streaming_external` | `external_stream.py:1329` | spool par tuile + réduction par colonnes, **CPU-only** |

**Flux actuel** : setup → `_build_aligned_cache_frame_major` (cache complet
N×patch par cellule) → `_make_cell_provider` → stack (inmem/stream/externe).

**Poste disque** : à N=21 818, patch ≈ 330 818 px (A0), le cache aligné vaut
`21 818 × 330 818 × 13 ≈ 93,8 GB` **par cellule**, avant manifestes et marge.
C'est le poste #5 de l'inventaire A0 et le blocage disque principal.

## 2. Ce qui est réutilisable

- `external_stream.py` : le **spool par tuile** (`tile_*/images.npy`,
  `weights.npy`), la réduction par colonnes (`reduce_spooled_tile`), les
  manifests transactionnels et le cleanup. Le prix est déjà borné :
  pire tuile 38×38 → ~481 MiB de spool à N=21 818, réduction ~658 MiB à
  `column_batch=1024`.
- `file_provider.BoundedMemmapCanonicalProvider` : lit une frame/tuile à la
  fois au lieu de garder N×patch résident.
- `storage_profile` : sélection `dense`/`bounded`/`auto` déjà en place.
- `rejection_sink` : sink de comptage borné, déjà utilisé par le profil borné.

**À changer** : le cache aligné reste la **source** du provider. Il faut un
provider qui lit les sources directement (ou un cache de fenêtre) et alimente
le spool par tuile sans écrire les N×patch fichiers.

## 3. Design proposé — fenêtre + spool par tuile (hybride)

1. **Fenêtre de préparation** : pour une cellule et une tuile (ou un groupe de
   tuiles), lire les frames dans l'ordre figé, appliquer la normalisation/jauge
   et la reprojection **déjà calculées**, et écrire uniquement le spool de la
   tuile (`images` f32 + `weights` f32). La fenêtre ne garde que quelques
   frames en RAM (2–3 frames + tuile étendue).
2. **Consommation** : `run_cell_streaming_external` réduit le spool par colonnes
   (CPU exact) et produit les plans de sortie de la tuile.
3. **Recyclage** : après commit de la tuile/cellule, supprimer le spool
   (cleanup existant) ; l'espace est réutilisé pour la tuile suivante. Le cache
   N×patch n'est jamais matérialisé.
4. **Jauge/photométrie** : les statistiques par frame (coefficients, σ, poids)
   doivent être calculées en passe 1 streaming (ou réutilisées du cache de
   calibration existant) ; la passe 2 ne fait que reproduire les échantillons
   normalisés. C'est le seam le plus délicat pour la parité bit-identique.

## 4. Enveloppe disque

| Poste | Cache complet actuel | Fenêtre + spool |
| --- | ---: | ---: |
| Cache aligné N×patch | **~93,8 GB / cellule** | 0 |
| Spool d'une tuile (38×38, N=21 818) | — | **~481 MiB** |
| Réduction (batch 1024) | — | ~658 MiB (RAM) |
| Fenêtre de frames | — | quelques frames (~12–25 MB) |
| Marge transactionnelle | — | incluse dans le spool |

Q20 devient démontrable : quota scratch ≪ 94 GB (par ex. 2 GB), tout en
conservant les N contributeurs.

## 5. Plan d'implémentation borné

| Sous-tâche | Fichiers | Contenu | Test |
| --- | --- | --- | --- |
| A2a | `file_provider.py` | `WindowedCanonicalProvider` : itère les frames par fenêtre, sans cache N×patch | test provider : mêmes tableaux/ordre que le cache pour une petite cellule |
| A2b | `zemosaic_zegrid_mode.py` | brancher le provider fenêtré quand `storage_profile` bounded ; garder le cache dense | test de sélection de provider |
| A2c | `external_stream.py` | accepter le provider fenêtré, borner le spool, recyclage | Q20 : N≥20k, quota < 94 GB, parité oracle |
| A2d | `external_stream.py` / `rejection_sink.py` | erreurs I/O/arrêt, nettoyage limité aux artefacts possédés | Q21 |
| A2e | tests | parité préparation bit-identique (W1), handles bornés | suite ciblée |

## 6. Risques

- **Parité de préparation** : la jauge/normalisation doit rester identique ;
  toute statistique recalculée en streaming doit être bit-identique à l'oracle.
- **Ordre des frames** : la fenêtre doit suivre l'inventaire figé.
- **Handles Windows** : fermer chaque memmap/fichier avant recyclage.
- **Reprise** : les cellules validées ne doivent pas être recalculées ; le
  spool est transactionnel (M4) mais le provider fenêtré doit respecter les
  commits.
- **Coût I/O** : relire les frames par tuile peut amplifier les lectures ;
  mesurer reads/writes et ne pas promettre un gain de temps.

## 7. Verdict

- **Faisable** : oui, en réutilisant le spool par tuile et le provider borné.
- **Décision proposée** : hybride **fenêtre de frames + spool par tuile**, sans
  cache N×patch ; garder le cache dense pour les runs qui le demandent.
- **Plus petite sous-tâche suivante** : A2a — `WindowedCanonicalProvider` avec
  test de parité sur une petite cellule, sans toucher au produit.
