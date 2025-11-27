# Agent.md: Analyse de l'anomalie des bandes noires

Ce document détaille la démarche suivie par l'agent pour identifier la cause des bandes noires apparues sur les images après le commit `3ff316977365884ee0c76018b7a89569b278e196`.

## 1. Rapport initial

L'utilisateur a signalé l'apparition de "bandes noires" aux jointures des images produites par le logiciel. Le problème a été lié au commit `3ff316977365884ee0c76018b7a89569b278e196`.

## 2. Analyse du commit

L'analyse du commit a révélé une modification dans le fichier `zemosaic_utils.py` au sein de la fonction `save_fits_image`.

Le changement principal était le remplacement du bloc de conversion de type de données :

**Ancien code :**
```python
# Avoid an extra full-size copy if already float32 (important for huge mosaics)
if isinstance(image_data, np.ndarray) and image_data.dtype == np.float32:
    data_to_write_temp = image_data
else:
    data_to_write_temp = image_data.astype(np.float32, copy=False)
```

**Nouveau code (introduit par le commit) :**
```python
data_to_write_temp = _ensure_float32_no_nan(image_data)
if not image_data_isfinite:
    _log_util_save(
        "SAVE_DEBUG: Valeurs non finies détectées. Remplacement par 0.0 avant export float.",
        "WARN",
    )
```

## 3. Identification de la cause racine

Le nouveau code fait appel à une fonction `_ensure_float32_no_nan`. Un message de log associé indiquait clairement : "Valeurs non finies détectées. Remplacement par 0.0 avant export float."

L'inspection du code de la fonction a confirmé son comportement :

```python
def _ensure_float32_no_nan(arr: np.ndarray) -> np.ndarray:
    """Return a float32 view/copy of ``arr`` with NaN/Inf replaced by zero."""
    arr_float = arr.astype(np.float32, copy=False)
    if not np.all(np.isfinite(arr_float)):
        arr_float = arr_float.copy()
        np.nan_to_num(arr_float, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return arr_float
```

Cette fonction remplace explicitement les valeurs `NaN` (Not a Number) par `0.0`.

## 4. Conclusion

Dans le contexte de la création de mosaïques d'images, les zones sans données (par exemple, les espaces entre les images avant l'alignement complet) sont souvent représentées par des `NaN`. En convertissant ces `NaN` en `0.0`, le commit a involontairement transformé ces zones "vides" en zones noires, créant ainsi les bandes observées par l'utilisateur.
