# Followup.md: Procédure pour corriger les bandes noires

Ce document explique comment rétablir le fonctionnement normal de l'application et supprimer les bandes noires des images produites.

## Objectif

L'objectif est d'annuler la modification qui transforme les zones sans données (`NaN`) en pixels noirs (`0.0`). Pour ce faire, nous allons restaurer le comportement précédent du code dans le fichier `zemosaic_utils.py`.

## Procédure de correction

1.  [x] **Ouvrir le fichier** : `zemosaic_utils.py`.

2.  [x] **Localiser le bloc de code suivant** (aux environs de la ligne 3443) :

    ```python
            data_to_write_temp = _ensure_float32_no_nan(image_data)
            if not image_data_isfinite:
                _log_util_save(
                    "SAVE_DEBUG: Valeurs non finies détectées. Remplacement par 0.0 avant export float.",
                    "WARN",
                )
    ```

3.  [x] **Remplacer ce bloc** par le code original ci-dessous :

    ```python
            # Avoid an extra full-size copy if already float32 (important for huge mosaics)
            if isinstance(image_data, np.ndarray) and image_data.dtype == np.float32:
                data_to_write_temp = image_data
            else:
                data_to_write_temp = image_data.astype(np.float32, copy=False)
    ```

## Résultat attendu

Après cette modification, le programme ne convertira plus les valeurs `NaN` en `0.0` lors de la sauvegarde. Les zones sans données ne seront plus représentées par des bandes noires, restaurant ainsi la qualité visuelle attendue des mosaïques.

**Note** : Le commit original visait probablement à résoudre un problème lié à la présence de valeurs non-finies dans les fichiers FITS. Bien que cette modification corrige le problème des bandes noires, il est possible que le problème sous-jacent que le commit tentait de résoudre refasse surface dans certains cas. Une solution plus robuste pourrait consister à appliquer la conversion `NaN` -> `0` de manière plus ciblée et optionnelle, plutôt que systématique.
