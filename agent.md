### 🔍 Ce qu’on a vu dans le code (sans blabla)

* `process_tile`, `_reproject_frame_to_tile`, géométrie du canevas, bboxes, assignation des frames → **identiques** entre `grid_mode_last_know_geomtric_tiles_ok.py` et le `grid_mode.py` actuel.
* Le gros changement **fonctionnel** est dans **`_stack_weighted_patches`** :

  * **Ancienne version (OK)** : implémentation maison du stacking/sigma-clipping.
  * **Nouvelle version (FAULTY)** : `_stack_weighted_patches` essaie de **passer par le core stacker** (`stack_core`) et son écosystème (winsorized sigma-clip, weights, etc.), avec une grosse couche de logique autour.

👉 C’est exactement là que tu peux te retrouver avec :

* des zones où les stats voient “rien” → `Mean of empty slice`, `All-NaN slice`, etc.
* des poids qui tombent à zéro sur la majorité des pixels,
* donc des tuiles avec juste une bande de signal et 70–80 % de zéros.

**Conclusion bourrine mais rationnelle :**
On arrête de faire du zèle : on remet **exactement** l’implémentation de `_stack_weighted_patches` de la version OK dans le `grid_mode.py` actuel, et on ne touche à rien d’autre (ni GPU, ni multithread, ni géométrie).

---

## ✅ Ce qu’on va demander à Codex

Je te fais un **agent.md / followup.md** spécial “REVERT `_stack_weighted_patches`” que tu peux donner tel quel.

### `agent.md`

````markdown
# Mission

Restore the **last-known-good tile stacking behaviour** in Grid mode by reverting
`_stack_weighted_patches` in the current `grid_mode.py` to the implementation
from `grid_mode_last_know_geomtric_tiles_ok.py`.

Goal: make grid tiles (`tile_000x.fits`) produced by the current Grid mode
**numerically consistent** with the “good geometry” version, without touching:

- tile geometry,
- global WCS,
- multithreading,
- GPU-specific code.

---

## Context

The user has two Grid mode implementations:

- `grid_mode_last_know_geomtric_tiles_ok.py` → old Grid mode:
  - tiles are *geometrically* correct,
  - tile content is complete (no large black zones),
  - no GPU/multithread integration.
- `grid_mode.py` (current) → new Grid mode:
  - same grid geometry (same global canvas, tile bboxes, frame counts per tile),
  - GPU + multithread integration,
  - **but** tiles 1–5, 7, 8 are much more “empty”:
    - only a narrow band of signal,
    - 60–80 % of pixels are exactly zero (`zero_frac` high),
  - lots of `RuntimeWarning: Mean of empty slice / All-NaN slice / DoF <= 0`
    from Astropy/NumPy stats.

Code analysis shows:

- `process_tile(...)` is almost identical between both versions, except for
  extra logging and GPU/CPU messages.
- `_reproject_frame_to_tile(...)` is identical (geometry and reproject logic).
- Grid geometry (global canvas, bboxes) and frame→tile assignment are identical.

The **only significant functional difference** is the implementation of:

```python
def _stack_weighted_patches(...):
    ...
````

* In `grid_mode_last_know_geomtric_tiles_ok.py`:

  * pure “legacy” implementation using NumPy, doing local normalization,
    sigma-clipping and combining patches.
* In `grid_mode.py`:

  * `_stack_weighted_patches` tries to delegate to a **shared core stacker**
    (via `stack_core`/`stack_core_gpu` logic) with a larger config surface.
  * This new path is much more complex and is the most likely cause of:

    * tiles being under-filled,
    * weights suppressed over large areas,
    * many NaN/empty-slice warnings from stats.

The user wants a **“bourrin but safe” fix**:

> “Just make the current grid mode stack tiles like the old one again, without
> breaking the code tree or over-refactoring.”

---

## Files to modify

* `grid_mode.py` (current Grid mode, faulty tile content)
* `grid_mode_last_know_geomtric_tiles_ok.py` (reference for the correct stacking)

Do **NOT** modify:

* `zemosaic_worker.py`
* the classic stacking pipeline
* GPU-specific helpers (`_stack_weighted_patches_gpu`, etc.)
* WCS/global grid construction logic
* tile geometry or frame assignment

---

## Requirements

### 1. Restore legacy `_stack_weighted_patches` in `grid_mode.py`

1. Open `grid_mode_last_know_geomtric_tiles_ok.py` and locate:

   ```python
   def _stack_weighted_patches(
       patches: list[np.ndarray],
       weights: list[np.ndarray],
       config: GridModeConfig,
       *,
       reference_median: float | None = None,
       return_weight_sum: bool = False,
       return_ref_median: bool = False,
   ) -> np.ndarray | tuple | None:
       ...
   ```

   This is the **last-known-good** implementation.

2. Open `grid_mode.py` and locate the *same* function definition.

3. **Replace the entire body** of `_stack_weighted_patches` in `grid_mode.py`
   with the body from `grid_mode_last_know_geomtric_tiles_ok.py`, preserving:

   * the same signature (arguments, type hints, return type),
   * any docstring that’s useful (you can keep the newer docstring if you like,
     as long as the behaviour matches the old implementation).

   In other words:

   * `_stack_weighted_patches` in `grid_mode.py` must behave **exactly like**
     the version in `grid_mode_last_know_geomtric_tiles_ok.py`.
   * **Remove or ignore** the delegation to `stack_core` / shared stacker inside
     `_stack_weighted_patches`. Grid mode should use its own legacy stacker
     again for CPU.

4. Make sure the restored implementation:

   * still returns:

     * either a single `np.ndarray`,
     * or a tuple containing `(stacked, weight_sum, ref_median)` when
       `return_weight_sum` / `return_ref_median` are `True`,
   * still operates on H×W×C (or H×W) layout, as in the old file,
   * preserves float32 outputs (as in the old file).

### 2. Keep GPU / multithread support intact

* Do **NOT** modify `_stack_weighted_patches_gpu` (if present) or any CuPy-based
  helper.
* Do **NOT** modify:

  * how `process_tile(...)` decides between GPU and CPU paths (it should still
    choose GPU if `config.use_gpu` is True and the GPU helper is available),
  * chunking logic or ThreadPool usage.

The only behavioural change must be:

* When Grid mode stacks tile patches on **CPU**, it uses the **legacy**
  `_stack_weighted_patches` behaviour from the good version.

### 3. Clean up any dead code introduced by the revert

After restoring the old `_stack_weighted_patches` body:

* If there are imports in `grid_mode.py` that are now **only used by the
  “new” stack_core path** and no longer referenced, you may:

  * either keep them (harmless but a bit noisy),
  * or **safely remove** them if you can confirm they are not used anywhere
    else in `grid_mode.py`.

Do **not** remove any GPU-related imports unless you are sure they are unused.

### 4. Keep logs and diagnostics

* Keep any existing logging in `process_tile` / `assemble_tiles`:

  * `[GRID][TILE_GEOM] ... nan_frac=... zero_frac=...`,
  * other `[GRID]` and `DEBUG_SHAPE` logs.
* You may add a **single** INFO log when `_stack_weighted_patches` is called,
  indicating that the **legacy stacker** is used for CPU in Grid mode, to make
  future debugging easier.

---

## Tests

Use the same dataset that produced:

* `zemosaic_worker_ok.log` (tiles correct, last-known-good grid)
* `zemosaic_worker_faulty.log` (tiles incomplete, current grid)

### Test 1 – Basic regression of Grid mode with fixed `_stack_weighted_patches`

1. Run `zemosaic_worker.py` using the **current** `grid_mode.py` with the
   restored `_stack_weighted_patches`.
2. Ensure Grid mode runs to completion (no new exceptions).

### Test 2 – Compare tiles vs last-known-good

1. For at least tiles 1, 2, 4, 5, 7, 8:

   * Load `tile_000X.fits` produced by:

     * **last-known-good** `grid_mode_last_know_geomtric_tiles_ok.py`,
     * **current** `grid_mode.py` (after the revert).

2. Check:

   * Shapes and dtypes are identical.
   * Visual content is now similar:

     * tiles are no longer “mostly black” in the current version,
     * the narrow bands of signal seen previously are replaced by fuller,
       more homogeneous coverage similar to the old version.

3. Optionally, compute quick stats (per-tile):

   * min / max / median,
   * fraction of non-zero pixels,
   * and compare between old vs new.

   They should be close (allow for very small floating-point differences).

### Test 3 – GPU sanity (if possible)

If you have a config where `config.use_gpu` is True for Grid mode:

1. Run Grid mode with GPU enabled.
2. Confirm:

   * the pipeline still runs without error,
   * tiles look visually consistent (no regression introduced by the revert).

---

## Constraints

* Only touch `_stack_weighted_patches` in `grid_mode.py` and any negligible
  dead imports created by this change.
* Do NOT change geometry, WCS, tile indexing, or multithreading.
* Do NOT modify the classic pipeline stacker; this mission is **Grid mode only**.
* The goal is a **minimal, robust revert** to the old, working per-tile stacker.

