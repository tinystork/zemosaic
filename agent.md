# Mission : Optimiser et paralléliser la Second Pass (Phase 5) du pipeline ZeMosaic

## 🎯 Objectif global

La Phase 5 comporte deux sous-étapes :

1. **Reproject & Coadd classique**  
2. **Second Pass Coverage Renormalization (Two-Pass)**

Le premier bloc a déjà été stabilisé.  
La seconde passe, elle, reste **largement séquentielle**, très lente, et n’émet presque aucune télémétrie.

👉 **Ta mission est d’optimiser fortement la Second Pass**, en :

- parallélisant les opérations **au niveau ZeMosaic** (pas reproject lui-même),
- utilisant **cpu_workers** et **chunking ParallelPlan**,
- utilisant le **GPU** quand `use_gpu_phase5 = True`,
- garantissant que **SDS reste strictement intouché**.

---

# 🧱 Contexte technique

La seconde passe est pilotée depuis :

### Fichier :
- `zemosaic_worker.py`

### Fonctions clés :
- `run_second_pass_coverage_renorm(...)`
- `compute_per_tile_gains_from_coverage(...)`
- projection du coverage vers chaque tuile
- boucle `for ch in range(n_channels)` pour reprojection par canal
- (toute logique entre `[TwoPass] Second pass requested...` et `[TwoPass] coverage-renorm OK`)

### Problème actuel :

1. **Boucle par tuile** → Séquentielle  
2. **Boucle par canal** → Séquentielle  
3. **Reprojection** exécutée dans 1 appel global, sans chunking ZeMosaic  
4. **cpu_workers affichés mais non utilisés**  
5. **gpu=True loggé mais la logique reste CPU-bound majoritairement**  
6. **Télémétrie Phase 5 minimaliste**  
7. **rows(cpu/gpu)=0/0** à cause d’absence de découpage pour la TwoPass.

---

# ✔️ Ce que tu dois faire

## 1. Paralléliser compute_per_tile_gains_from_coverage

Dans `compute_per_tile_gains_from_coverage(...)` :

- Chaque tuile est aujourd’hui traitée dans une boucle Python séquentielle.
- Tu dois utiliser **ThreadPoolExecutor** ou **ProcessPoolExecutor** suivant ParallelPlan :

  - **Si GPU actif (`use_gpu=True`)** → utiliser un **ThreadPoolExecutor**  
    (les opérations CuPy libèrent le GIL → bénéfice immédiat).
  
  - **Si GPU inactif** → utiliser un **ProcessPoolExecutor**  
    (les opérations NumPy/Scipy/CV2 sont CPU-bound).

### Détails :

Pour chaque tuile :

- projection coverage → WCS tuile
- calcul médian → gain
- clamp dans [gain_clip_min, gain_clip_max]

Le parallélisme doit :

- respecter `plan.cpu_workers`
- respecter les limites mémoire (`max_chunk_bytes`) en batchant intelligemment la liste des tuiles
- renvoyer les gains dans l’ordre d’origine

⚠️ Interdiction de changer la logique mathématique.  
Simplement paralléliser.

---

## 2. Paralléliser la reprojection per-channel

Aujourd’hui :

```python
for ch in range(n_channels):
    ...
    chan_mosaic, chan_cov = _invoke_reproj(...)
➡️ Cette boucle doit être parallélisée :

Stratégie :
lancer 1 worker par canal quand n_channels >= 2

sinon 1 seul worker évidemment

respecter plan.cpu_workers (ne pas dépasser)

si GPU actif :

autoriser un seul canal à utiliser le GPU à la fois
(use_gpu = True uniquement pour 1 task)

les autres canaux → CPU
(sinon VRAM saturée)

si GPU inactif :

paralléliser tous les canaux en CPU

Contraintes :
Les résultats doivent être recombinés dans l’ordre original [H, W, C].

_invoke_reproj ne doit pas être modifié.

Si l’utilisateur a un GPU 8/12/16 Go → parallèle CPU+GPU hybride automatique.

3. Ajouter un vrai chunking pour la TwoPass (rows_per_chunk)
Actuellement, pour TwoPass :

bash
Copier le code
rows(cpu/gpu) = 0/0
chunk_mb(cpu/gpu) = 1144MB
→ aucune découpe.

Tu dois :

réutiliser le ParallelPlan appliqué en Phase 5
(celui obtenu juste avant pour Reproject & Coadd),

découper la coverage + la grille finale en blocs de lignes (row-chunks),

exécuter les opérations lourdes (gaussian blur, reprojection coverage→tile, gains apply) par chunk.

Les chunk doivent être définis par :

plan.rows_per_chunk (si disponible)

ou plan.max_chunk_bytes / plan.gpu_max_chunk_bytes (fallback)

ou au pire un découpage fixe 512–1024 lignes par chunk si aucun plan n’est disponible

⚠️ Encore une fois : pas de changement mathématique.

4. Ajouter télémétrie Phase 5 complète
Aujourd’hui, aucun STATS_UPDATE n’est émis pendant la seconde passe.

Tu dois :

envoyer un STATS_UPDATE au début,

un STATS_UPDATE toutes les X tuiles OU tous les X chunks,

un STATS_UPDATE à la fin.

Le stats_dict doit contenir (mêmes clés que Phase 3) :

makefile
Copier le code
phase_index=5
phase_name="Phase 5: Two-Pass Coverage Renorm"
cpu_percent
ram_used_mb
gpu_used_mb
cpu_workers=plan.cpu_workers
use_gpu=plan.use_gpu
use_gpu_phase5=true/false
tiles_done=X
tiles_total=Y
chunk_index
chunk_total
Tu peux réutiliser _log_and_callback("STATS_UPDATE", ...).

🔒 Ce que tu NE DOIS PAS toucher
AUCUN fichier/fonction SDS

AUCUNE logique mathématique (gaussian blur, gains, clamp)

AUCUN comportement Phase 1/3

AUCUN paramètre de configuration existant

AUCUN test

AUCUNE signature publique du pipeline

📂 Fichiers à modifier
Exclusivement :

zemosaic_worker.py

zemosaic_utils.py (si nécessaire pour ajouter un petit helper de parallélisation non-intrusif)

éventuellement parallel_utils.py pour exposer un petit helper parallel_map() réutilisable (non obligatoire)

✔️ Résultat attendu
Après implémentation :

La seconde passe doit diviser son temps de traitement par 2× à 8× selon CPU/GPU.

Le moniteur de ressources doit montrer :

CPU multi-workers actifs

GPU actif si use_gpu_phase5=True

Le log ne doit plus montrer rows(cpu/gpu)=0/0

La télémétrie Phase 5 doit apparaître clairement dans resource_telemetry.csv

Le pipeline SDS reste strictement identique.