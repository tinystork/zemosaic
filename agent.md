Parfait Tinystork — voici **le couple complet et final `agent.md` + `followup.md`**, propre, concis, et *strictement opérationnel* pour que **Codex implémente enfin la vraie duplication des images dans les batches**, active dans la pipeline, et garantisse **zéro trou dans la mosaïque**, même en cas de rejets massifs ou de groupes fragiles.

Ce couple est calibré pour être compris immédiatement par Codex, sans interprétation ni reformulation parasite.

---

# ✅ **agent.md — Mission : Implémentation complète de l’Overlapping + Duplication des Frames**

````markdown
# Agent Mission — Implement Full Overlapping + Frame Duplication in Batch Construction

## 🎯 Objective
Ensure ZeMosaic never produces "holes" in the master-tile grid, even when many frames are rejected or some groups are sparse.

You MUST implement actual operational duplication of frames **in the Phase 3 batch builder**, not only in the GUI.

This requires modifications in:
- `zemosaic_filter_gui_qt.py` (UI parameter already present but must be wired)
- `zemosaic_worker.py` → batch construction function (`_auto_split_groups` or equivalent)
- `build_stack_batches()` if separate
- Logging of duplication behavior

You MUST NOT modify:
- WCS logic  
- Reproject logic  
- ZeQualityMT algorithm itself  
- Phase 5 coadd logic  
- GPU/CPU fallback logic  

---

# ✔ Required New Behaviour (non-negotiable)

## 1) **Implement overlapping sliding-window batches**
A batch must be allowed to reuse frames from neighboring spatially adjacent frames.

Use:

```python
def make_overlapping_batches(indices, cap, overlap_frac):
    n = len(indices)
    if n <= cap:
        return [indices]
    step = max(1, int(cap * (1.0 - overlap_frac)))
    batches = []
    start = 0
    while start < n:
        end = min(n, start + cap)
        batch = indices[start:end]
        if len(batch) > 1:
            batches.append(batch)
        if end == n:
            break
        start += step
    return batches
````

Integrate this into the real batch planner used in Phase 3.

---

# 2) **Implement explicit frame duplication inside each batch**

If a batch is too small (common for Seestar data), duplicate frames *inside the batch* until it reaches a target size.

Add two new configuration parameters (backed by GUI):

* `MIN_SAFE_STACK = 3`
* `TARGET_STACK = 5`

Inside the batch builder:

```python
if allow_duplication:
    if len(frames) < TARGET_STACK:
        repeat = math.ceil(TARGET_STACK / len(frames))
        frames = (frames * repeat)[:TARGET_STACK]
```

This MUST be applied **before** stacking begins.

This behavior MUST be reflected in logs:

```
[Batch] Duplicating frames: original=2 → final=5 for tile 12
```

---

# 3) **Implement salvage mode when n_used < MIN_SAFE_STACK**

Inside the master-tile creation flow (Phase 3):

```python
if n_used < MIN_SAFE_STACK:
    logger.warning(f"Tile {tile_id}: salvage mode (n={n_used}). Relaxing QC and crop.")
    disable_zequalityMT_for_this_tile = True
    disable_quality_crop = True
```

**Do NOT** disable these features globally—only for the affected tile.

This prevents zero-coverage areas in the final mosaic.

---

# 4) **Wire the GUI overlap parameter to the batch planner**

The existing GUI field “Overlap %” must be saved into config and passed into the worker.

The worker must receive:

```python
overlap_frac = config.overlap_pct / 100.0
```

---

# 5) Maintain complete compatibility with the full pipeline

All outputs must remain valid:

* The number of master tiles must not change unpredictably.
* All existing `stack_plan.csv` expectations remain valid.
* No file paths or folder layouts must change.

Make no changes to WCS, reprojection, or Phase 5.

---

# ✔ Acceptance Conditions

To consider the task complete, Codex must verify all of the following:

* Sparse datasets do NOT produce vertical or diagonal “holes”.
* Every batch smaller than TARGET_STACK is automatically expanded by duplication.
* Salvage mode works only on affected tiles, never globally.
* Overlapping batches produce seamless coverage.
* Reprojection output matches CPU/GPU parity except expected minor floating-point differences.
* GPU/CPU fallback logic is unchanged.

---

# 📦 Deliverables required from Codex

* All patched code for:

  * `zemosaic_filter_gui_qt.py`
  * `zemosaic_worker.py`
  * any helper function touched (`_auto_split_groups`, `_prepare_batches`, etc.)
* Unified diff or full file rewrites.
* Logging improvements.
* Integration tests or manual verification snippets.



