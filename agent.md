
# 🧾 `agent.md` — Mission : *Restore Correct Pipeline Flows (Option A)*

```markdown
# AGENT MISSION FILE — Restore Classic Master Tile Pipeline When SDS is OFF
### ZeMosaic / ZeSeestarStacker — SDS Flow Fix (Option A)

You are an autonomous coding agent working on **ZeMosaic**.

Your mission is to **restore the correct pipeline architecture**, exactly as originally designed:

---

# 🔥 PRIMARY SPECIFICATION (NON-NEGOTIABLE)

## 1. Pipeline when SDS is OFF
When SDS mode is **NOT activated**, the worker must follow the **classic pipeline**:

```

Phase 1 → Phase 2 → Phase 3 (Master Tiles) → Phase 4 → Phase 5–6

````

This means:

### ❗ You must enforce:
- **NEVER call `assemble_global_mosaic_first` when SDS is OFF.**
- **NEVER call `assemble_global_mosaic_sds` when SDS is OFF.**

### ❗ Instead:
- Force `final_mosaic_data_HWC = None`
- Force the worker into:
  ```python
  if final_mosaic_data_HWC is None:
      # → run classic master-tile pipeline (Phase 3)
````

This restores the original behavior:

> **Master tiles are always built when SDS is OFF.**

---

## 2. Pipeline when SDS is ON

When SDS mode **IS activated**:

### The SDS pipeline becomes the primary flow:

```
Phase 1 → Phase 2 → SDS megapasses (lights → mega-tiles) → super-stack → Phase 5–6
```

### Implementation rules:

1. Call **ONLY** the following in this order:

   * `assemble_global_mosaic_sds(...)`
2. If — and ONLY if — SDS fails (e.g. returns `(None, None, None)`):

   * fallback to `assemble_global_mosaic_first(...)`
3. If Mosaic-First also returns `None`, allow classic Phase 3 fallback.

### ❗ SDS ON must NEVER go into classic master tiles unless SDS + Mosaic-First both fail.

---

# 🔒 SAFETY RULES — DO NOT BREAK ANYTHING ELSE

* **Do NOT modify normal (non-Seestar) workflow.**
* **Do NOT modify Tk GUIs.**
* **Do NOT modify SDS batch policy (already handled separately).**
* **Do NOT modify normalization, Phase 4.5, two-pass renorm or lecropper.**
* **Do NOT change major function signatures.**

You only modify the **flow logic** inside the worker.

---

# 🎯 EXACT FILE TO MODIFY

* `zemosaic_worker.py`

The key function is:

* `run_hierarchical_mosaic(...)`
  (the area where global_wcs_plan is evaluated and mosaic_result is computed)

---

# 🎯 EXPECTED LOGIC AFTER YOUR PATCH

Pseudo-code target:

```python
# SDS OFF → skip Mosaic-First, skip SDS
if not sds_mode_flag:
    final_mosaic_data_HWC = None
else:
    # SDS ON
    mosaic_result = assemble_global_mosaic_sds(...)
    if mosaic_result[0] is None:
        mosaic_result = assemble_global_mosaic_first(...)
    final_mosaic_data_HWC = mosaic_result[0]

# Only if SDS is OFF OR SDS failed AND Mosaic-First failed:
if final_mosaic_data_HWC is None:
    # → Phase 3 master-tile pipeline (classic)
else:
    # → P5/P6 post-processing
```

This restores the **original spirit**:

* SDS OFF = master-tile mode
* SDS ON = mega-tile mode

---

# ✔️ DELIVERABLE

Implement the modification cleanly and minimally inside `zemosaic_worker.py`, following the above specification 1:1.

````
