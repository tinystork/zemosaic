Merci ! Tu peux maintenant exécuter le plan complet suivant :

🎯 **TO-DO**

1.  **[x] Priorité absolue : Stabiliser la transformation WCS**
    *   **Action :** Rechercher dans tout le projet les appels à la fonction `.all_world2pix()` sur des objets WCS d'Astropy.
    *   **Correction :** Remplacer chaque appel `wcs.all_world2pix(coordinates, 1)` par `wcs.wcs_world2pix(coordinates, 1)`.
    *   **Justification :** `wcs_world2pix` utilise un algorithme itératif plus robuste, recommandé par Astropy pour éviter les erreurs de non-convergence (`failed to converge`) qui sont observées. C'est la cause la plus probable et la plus simple à corriger.
    *   **Validation :** L'avertissement `UserWarning: 'WCS.all_world2pix' failed to converge` doit disparaître des logs.

2.  **[ ] Vérification de l’alignement local (si l'étape 1 ne suffit pas)**
    *   Effectuer le test Phase 3 mono-thread.
    *   Retraiter un dataset problématique.
    *   Si la déformation est encore présente :
        *   instrumenter `align_images_in_group`
        *   comparer son comportement à `38c876a`
        *   corriger les écarts (FFT ou astroalign)

3.  **[ ] Correction du green cast**
    *   Inspecter `equalize_rgb_medians_inplace`.
    *   Tracer les gains et médianes.
    *   Corriger l’équilibrage colorimétrique.

4.  **[ ] Contraintes strictes**
    *   Aucun changement GPU.
    *   Aucun changement SDS.
    *   Patchs propres, documentés et ciblés.
    *   Explication claire de la cause racine.

🎯 **Résultat attendu**

Un comportement stable, propre, et identique ou supérieur au commit `38c876a`, sans déformation géométrique et sans dérive verte.
