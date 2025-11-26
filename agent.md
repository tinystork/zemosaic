Mission : Corriger les régressions d’alignement et de couleur introduites après le commit 38c876a

Le commit 38c876a est confirmé comme dernier état fonctionnel.

Deux régressions distinctes sont apparues dans V4WIP :

Déformation géométrique + mauvais alignement

Dérive colorimétrique vers le vert

L’objectif est de corriger les deux, sans toucher à la voie GPU.

🎯 Objectifs

 Restaurer un WCS valide (plus d’erreur all_world2pix failed to converge)

 Restaurer un alignement géométrique correct équivalent à 38c876a

 Corriger la dérive verte (balance RVB)

 Garantir la stabilité multi-thread (Phase 3)

 Ne modifier aucun fichier GPU (zemosaic_align_stack_gpu.py)

📌 Contexte & Hypothèses
Hypothèse 1 — Instabilité numérique dans la transformation WCS

Les logs montrent une `UserWarning` d'Astropy :

`WCS.all_world2pix failed to converge`

**Analyse de l'erreur :**
Cette erreur indique que l'algorithme itératif utilisé par Astropy pour convertir des coordonnées mondiales (célestes) en coordonnées de pixels n'a pas trouvé de solution stable. Ce n'est pas nécessairement un signe de WCS corrompu, mais plutôt d'une **instabilité numérique**.

**Causes possibles :**
1.  Les coordonnées mondiales demandées sont très éloignées de la zone de l'image.
2.  Le modèle de distorsion (WCS) est très prononcé.
3.  L'appel à `all_world2pix` est connu pour être moins robuste que d'autres fonctions dans certains cas.

**Action recommandée par Astropy :**
La documentation et la communauté Astropy suggèrent de remplacer `wcs.all_world2pix` par `wcs.wcs_world2pix` pour les conversions de paires de coordonnées, car cette dernière implémente un algorithme plus robuste. Le problème n'est donc probablement pas dans la création du WCS (`_parse_wcs_file_content_za_v2`), mais dans son **utilisation**.

Hypothèse 2 — Alignement local (FFT + astroalign) potentiellement touché

⚠️ Très important :
Si les master tiles elles-mêmes sont déjà déformées, le bug n’est pas uniquement WCS.

Dans ce cas, la fonction à inspecter est :

`align_images_in_group` dans `zemosaic_align_stack.py`

Cette fonction utilise :

pré-alignement FFT

alignement fin avec `astroalign.register`

Une régression locale (pré-alignement, warp, dtype, normalisation) peut provoquer une déformation même hors WCS.

Hypothèse 3 — Dérive colorimétrique vers le vert

Probable cause :

mauvaise balance des gains dans la fonction
`equalize_rgb_medians_inplace`
(appelée via `_poststack_rgb_equalization`).

Cette fonction calcule des gains canal par canal.
Une médiane faussée, un clipping ou un double-apply provoque rapidement un `green cast`.

📁 Fichiers à analyser et éventuellement corriger
Priorité 1

 **Localiser les appels à `.all_world2pix`** dans le code (probablement dans `zemosaic_worker.py` ou `zemosaic_utils.py`) et les remplacer.

Priorité 2

 `zemosaic_align_stack.py`

`align_images_in_group`

`equalize_rgb_medians_inplace`

`_poststack_rgb_equalization`

 `zemosaic_astrometry.py` (si le remplacement de `all_world2pix` ne suffit pas)

`_parse_wcs_file_content_za_v2`

🚫 À ne jamais modifier

 `zemosaic_align_stack_gpu.py`

 Tout kernel ou pipeline GPU
(Le GPU empile seulement → les déformations viennent avant.)

💡 Tâches détaillées
✔️ Étape 1 — Correction de l'instabilité WCS

 **Action immédiate :** Remplacer les appels à `wcs.all_world2pix` par `wcs.wcs_world2pix`. Cette fonction est plus stable pour les transformations de coordonnées. Il faudra adapter le code si les entrées/sorties diffèrent légèrement (ex: gestion de tableaux).

 **Vérifier** la disparition de la `UserWarning`:
`WCS.all_world2pix failed to converge`

✔️ Étape 2 — Validation géométrique
🎯 Test crucial : mono-thread

Pour exclure une race condition :

 Forcer Phase 3 en mono-thread :

`winsor_worker_limit = 1`

`actual_num_workers_ph3 = 1`

 Retraiter un dataset problématique

 Comparer master tiles mono-thread vs multi-thread

Si déformation présente dans les deux → ce n'est pas un bug multi-thread.

🎯 Inspecter l’alignement intra-tile (`align_images_in_group`)

Si les master tiles montrent déjà la déformation, alors :

 Instrumenter `align_images_in_group`

 Tracer :

 `dx, dy` FFT

 image source et référence envoyées à `astroalign`

 matrice affine retournée par `astroalign.register`

 `shape`, `dtype` et normalisation des entrées

 Comparer le comportement à celui du commit 38c876a

✔️ Étape 3 — Correction du green cast

 Inspecter `equalize_rgb_medians_inplace`

 Logguer :

 médiane R, G, B

 gains appliqués

 avant/après

 Vérifier pas de double-application

 S'assurer que la normalisation RGB est identique à celle de 38c876a

 Corriger l’algorithme pour obtenir une balance neutre

✔️ Validation finale

Le correctif est validé lorsque, pour un même dataset :

 Plus d’erreur WCS `...failed to converge`.

 Les master tiles sont géométriquement correctes.

 La mosaïque finale ne présente aucune déformation.

 La balance des couleurs est neutre.

 Aucun changement n’a touché la voie GPU.

 Le comportement est identique ou meilleur que 38c876a.
