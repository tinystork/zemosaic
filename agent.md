# Mission : Audit géométrique complet entre V4.2.0 et V4WIP
# Alignement – Master Tiles – WCS – Reprojection – Coverage

## Contexte
Sur un dataset identique, on observe :

### ✔ V4.2.0
- Aucune erreur WCS.
- preview.png généré.
- coverage_xxx.dat généré.
- mosaïque homogène, pas de patchwork.
- footprint maîtrisé.
- couleurs cohérentes.

### ❌ V4WIP
- Erreur critique Astropy :
WCS.all_world2pix failed to converge … solution is diverging

markdown
Copier le code
- **Pas de preview.png**.
- **Pas de fichier de coverage**.
- **Patchwork visible**, zones sombres/vertes, bruit amplifié.
- Alignement inter-tuiles incohérent.

### IMPORTANT
Les logs montrent que :
- `[RGB-EQ] poststack_equalize_rgb` fonctionne correctement et identiquement en V4.2.0 (gains identiques).  
➜ **À exclure du périmètre** (ne rien modifier dans la normalisation intra-stack).
- Les problèmes restants sont cohérents avec :
- une modification dans le **stacking CPU/GPU**,
- un changement dans la **géométrie / offsets**,
- une altération du **WCS interne** des master tiles,
- un padding différent,
- un pivot ou une orientation modifiée,
- une transformation incohérente passée à `reproject_interp`.

## OBJECTIF : restaurer la qualité V4.2.0

L’objectif de cette mission est de :

1. **Comparer précisément V4.2.0 et V4WIP** dans les fichiers :
 - `zemosaic_align_stack.py`
 - `zemosaic_align_stack_gpu.py`

2. **Identifier toute différence** concernant :
 - la construction de la master tile,
 - les offsets appliqués,
 - les translations (dx, dy),
 - les rotations éventuelles,
 - l’origine du tableau (origin=upper/lower),
 - le padding,
 - la propagation ou non des masques alpha,
 - le dtype et les arrondis,
 - la génération du WCS interne (CRPIX, CRVAL, CDELT, CD matrix),
 - la logique d'estimation du champ,
 - la forme (shape) finale passée à la Phase 5.

3. **Vérifier les arguments passés à :**
 - `reproject_interp`
 - `reproject_exact`
 - `reproject.adaptive`
 - ainsi que les valeurs de « output_projection », « order », « boundary ».

4. **Corriger la branche V4WIP** pour faire respecter EXACTEMENT les règles suivantes :

 ### 🔥 Règles obligatoires (identiques à V4.2.0)
 - Même géométrie (shape HxW) des master tiles.
 - Même orientation (`origin='lower'` ou `'upper'`, selon V4.2.0).
 - Même padding autour des stars.
 - Même logique d’offset.
 - Même normalisation *avant* assemblage.
 - WCS strictement compatible (CRPIX, CRVAL, CD matrix identiques).
 - Les master tiles doivent être géométriquement superposables.
 - La Phase 5 (coadd final) doit recevoir des tuiles compatibles.

5. **Assurer que :**
 - `preview.png` est de nouveau généré.
 - `coverage_*.dat` est de nouveau généré.
 - plus aucune erreur WCS n’apparaît.
 - la mosaïque finale V4WIP est visuellement identique à V4.2.0.

## Fichiers à auditer et corriger
- `zemosaic_align_stack.py`
- `zemosaic_align_stack_gpu.py`
- éventuellement :
- `zemosaic_utils.build_mastertile_wcs`
- `zemosaic_utils.estimate_pixel_scale`
- `zemosaic_utils.make_mastertile_projection`

## Contraintes strictes
- Ne pas modifier la logique SDS.
- Ne pas toucher au post-anchor (déjà corrigé).
- Ne pas toucher à poststack_equalize_rgb.
- Ne pas modifier l’API GUI.
- Ne pas changer la gestion CPU/GPU hormis ce qui touche à la géométrie.

## Résultat attendu
1. Un diff clair montrant :
 - ce qui a changé entre V4.2.0 et V4WIP,
 - pourquoi ces changements provoquaient des divergences WCS,
 - ce qui a été corrigé.

2. Un commit qui restaure :
 - la géométrie correcte des master tiles,
 - une WCS fiable,
 - un reprojetage stable,
 - preview.png,
 - coverage.dat,
 - une mosaïque propre sans patchwork.

3. Une mise à jour du fichier `followup.md`.

Merci d’être exhaustif et rigoureux.