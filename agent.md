# Mission — SDS GPU fallback fix (CuPy nanpercentile)

## 🎯 Objectif UNIQUE (strict)
Corriger **exclusivement** le mode **SuperDupStack (SDS)** afin d’éliminer le fallback CPU causé par :

    AttributeError: module 'cupy' has no attribute 'nanpercentile'

⚠️ Toute modification hors SDS sera considérée comme incorrecte.

---

## 🧠 Contexte
Lors d’un run SDS avec GPU activé, la route GPU échoue dans le helper :

    helper = gpu_reproject

Le worker bascule ensuite sur la voie CPU avec le message :

    gpu_fallback_runtime_error: cupy has no attribute nanpercentile

Ce fallback **ne doit plus se produire**.

---

## 🚫 Interdictions absolues
- ❌ Ne pas modifier le mode classique
- ❌ Ne pas modifier le mode grid
- ❌ Ne pas refactorer des utilitaires globaux “pour faire mieux”
- ❌ Ne pas toucher à la photométrie, normalisation, assemblage
- ❌ Ne pas modifier le comportement batch size = 0 / > 1
- ❌ Ne pas faire de “grep & replace” global sur cp.nanpercentile

👉 **Tout changement hors du chemin SDS est interdit.**

---

## 🧭 Périmètre autorisé
Uniquement :
- le chemin d’exécution **SDS**
- les fonctions réellement appelées lorsque :
  - mode = SuperDupStack
  - helper = gpu_reproject
  - GPU actif

---

## 🛠️ Travail attendu
- [x] Identifier **précisément** le chemin d’appel SDS menant à `cp.nanpercentile`
   - ne pas supposer
   - suivre le flux réel (SDS → gpu_reproject → stats/percentiles)

- [x] Pour **chaque appel SDS** à `cp.nanpercentile` :
   - remplacer par un wrapper **local SDS**
   - compatible CuPy sans `nanpercentile`

### Wrapper attendu (exemple de comportement)
```python
def _sds_cp_nanpercentile(arr_gpu, percentiles, *, axis=None):
    import cupy as cp
    if hasattr(cp, "nanpercentile"):
        return cp.nanpercentile(arr_gpu, percentiles, axis=axis)
    if hasattr(cp, "nanquantile"):
        if np.isscalar(percentiles):
            q = float(percentiles) / 100.0
        else:
            q = cp.asarray(percentiles, dtype=cp.float32) / 100.0
        return cp.nanquantile(arr_gpu, q, axis=axis)
    raise RuntimeError("CuPy missing nanpercentile/nanquantile")
📌 Le wrapper :

doit être local à SDS

ou utilisé uniquement dans la voie SDS

ne doit pas modifier les autres modes

✅ Résultat attendu
Plus aucun fallback CPU lié à nanpercentile

Le helper gpu_reproject reste sur la voie GPU en SDS

Les autres modes produisent exactement les mêmes logs et résultats qu’avant

📦 Livrable
Un seul commit

Message :

Fix SDS GPU nanpercentile compatibility (no CPU fallback)
Diff minimal, SDS only

