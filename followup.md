
---

# 🔁 `followup.md` — Step-by-Step Instructions for Codex

```markdown
# FOLLOW-UP — Step-by-Step Instructions
### SDS Flow Fix (Option A)

This follow-up gives you exact steps to implement the mission described in `agent.md`.

---

# 1. Open file:
`zemosaic_worker.py`

Locate the function:

````

run_hierarchical_mosaic(...)

````

Inside, find the block:

```python
if global_wcs_plan and global_wcs_plan["enabled"]:
    mosaic_result = (None, None, None)

    if sds_mode_flag:
        mosaic_result = assemble_global_mosaic_sds(...)
    if mosaic_result[0] is None:
        mosaic_result = assemble_global_mosaic_first(...)

    final_mosaic_data_HWC, final_mosaic_coverage_HW, final_alpha_map = mosaic_result
    ...
````

---

# 2. MODIFY THE FLOW ACCORDING TO OPTION A

Replace the logic above with:

## Case A — SDS is OFF

Insert:

```python
if not sds_mode_flag:
    # SDS is OFF → force master tiles mode
    final_mosaic_data_HWC = None
    final_mosaic_coverage_HW = None
    final_alpha_map = None
```

And **skip**:

* `assemble_global_mosaic_sds`
* `assemble_global_mosaic_first`

## Case B — SDS is ON

Implement:

```python
else:
    # SDS is ON
    mosaic_result = assemble_global_mosaic_sds(...)

    if mosaic_result[0] is None:
        mosaic_result = assemble_global_mosaic_first(...)

    final_mosaic_data_HWC, final_mosaic_coverage_HW, final_alpha_map = mosaic_result
```

---

# 3. LET PHASE 3 HANDLE THE REST

After the SDS block you will find:

```python
if final_mosaic_data_HWC is None:
    # → classic master tile block (Phase 3)
```

Do **not modify this block**.

This ensures:

* SDS OFF → Phase 3 runs ALWAYS
* SDS ON → Phase 3 only runs if SDS AND Mosaic-First failed

---

# 4. Ensure NO OTHER BEHAVIOR CHANGES

Do **not** touch:

* master tile construction code
* SDS batch creation
* normalization pipelines
* GPU/CPU branches
* lecropper / alt-az cleanup
* Phase 4.5 / two-pass
* Tk GUI
* Qt GUI (outside SDS mode flag logic)

---

# 5. Add minimal logging (optional but recommended)

When SDS is OFF, add:

```python
self._log_and_callback(pcb_fn, "info", "sds_off_classic_mastertile")
```

When SDS is ON:

```python
self._log_and_callback(pcb_fn, "info", "sds_on_megatile_mode")
```

If SDS fails:

```python
self._log_and_callback(pcb_fn, "warning", "sds_failed_fallback_mosaic_first")
```

Do NOT create new localization keys unless necessary.

---

# 6. Test Scenarios

### Test 1 — SDS OFF, Seestar data

* Must ALWAYS go to Phase 3 master tiles.
* Must NOT call SDS nor Mosaic-First lights.

### Test 2 — SDS ON, healthy WCS

* SDS → mega-tiles → super-stack
* Phase 3 bypassed

### Test 3 — SDS ON, SDS fail

* SDS fails → Mosaic-First
* If Mosaic-First fails too → Phase 3

### Test 4 — Non-Seestar data

* Must remain unchanged

---

This completes the modification.

```

---

# 🎉 Résultat

Avec ces deux fichiers :

- SDS OFF = **tu retrouves EXACTEMENT ton pipeline original**  
  → Master tiles → mosaïque → renorm → save

- SDS ON =  
  → reproject lights par batches SDS → stack de mega tiles → P5/P6

