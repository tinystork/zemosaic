
## `agent.md`

````markdown
# AGENT MISSION FILE — SDS vs Classic Pipeline Flow (Option A)

You are an autonomous coding agent working on the **ZeMosaic / ZeSeestarStacker** project.

The repository already contains (non-exhaustive):

- `run_zemosaic.py`
- `zemosaic_gui.py`, `zemosaic_gui_qt.py`
- `zemosaic_filter_gui.py`, `zemosaic_filter_gui_qt.py`
- `zemosaic_worker.py`
- `zemosaic_align_stack.py`
- `zemosaic_utils.py`, `zemosaic_config.py`
- `zequalityMT.py`, `lecropper.py`
- `locales/en.json`, `locales/fr.json`
- helper modules (`core/`, `locales/`, etc.)

Your mission is to **restore and enforce the correct high-level pipeline architecture** depending on whether **SDS mode** (ZeSupaDupStack) is enabled or not, while preserving all existing optimizations and scalability (10 000+ frames) and **not breaking** the SDS batch policy already implemented.

---

## 1. High-level behaviour (non-negotiable)

### 1.1. When SDS is **OFF** (classic mode)

The worker **must** follow the classic master-tile pipeline:

```text
P1 — Astrometry / WCS
P2 — Clustering / grouping
P3 — Master Tiles stacking
P4 — Grid / inter-master / final reprojection
P5 — Final assembly, cleanup, save
````

This implies:

* **Master tiles are ALWAYS built when SDS is OFF.**
* The worker must **reach Phase 3** and run the master-tile stacking code as today.
* No SDS-specific global mosaic assembly (`assemble_global_mosaic_sds`) or “Mosaic-First lights” (`assemble_global_mosaic_first`) must be used in this SDS-OFF case.

Concretely:

> If SDS is OFF, the **global WCS “seestar mosaic-first” “lights → global_coadd” path must NOT be used.**
> The final mosaic must come from the **master-tile pipeline**, not from direct reprojection of raw lights.

### 1.2. When SDS is **ON** (ZeSupaDupStack)

The worker **must** use the SDS mega-tiles pipeline:

```text
P1 — Astrometry / WCS
P2 — Clustering / grouping
P3 — (classic master tiles is skipped for SDS)
P3bis — SDS mega-tiles: RAW lights → global WCS mega-tiles (per SDS batch)
P4 — Stack mega-tiles (super-stack)
P5 — Final post-processing (crop, Alt-Az cleanup, 2-pass coverage renorm, save)
```

Operationally:

1. Use the Seestar global WCS plan as today.
2. Build SDS batches (using the **existing SDS batch policy**:

   * coverage threshold
   * min batch size
   * target batch size
     already implemented in previous changes).
3. For each SDS batch:

   * reproject the **raw lights** of that batch into the global WCS
   * run the “Mosaic-First-like” batch-local pipeline to produce **one mega-tile per batch**
     (image H×W×C + coverage + alpha).
4. Stack all mega-tiles together using `zemosaic_align_stack`, with coverage-based weighting, into:

   * `final_mosaic_data_HWC`,
   * `final_mosaic_coverage_HW`,
   * `final_alpha_map`.
5. Pass these to the existing Phase 5/6 code path:

   * autocrop, lecropper, Alt-Az cleanup, two-pass coverage renorm, save, etc.

Fallback behaviour when SDS is ON:

* If `assemble_global_mosaic_sds(...)` fails (returns `(None, None, None)` or no batches):

  * You may **fallback to `assemble_global_mosaic_first(...)`** as a secondary strategy.
* If Mosaic-First also fails:

  * Then, as a last resort, the worker may run the classic Phase 3 master-tile pipeline.

But in the **normal SDS-ON case**, the result **must** come from SDS mega-tiles, not from classic master tiles.

---

## 2. Scalability requirements (10 000+ frames)

The project is explicitly designed to handle very large image sets (e.g. 10 000 frames or more). You **must not** introduce any change which:

* Materializes a huge H×W×N cube of all frames in memory.
* Turns any O(N) part of the pipeline into O(N²).
* Breaks the existing chunking / streaming logic.

You must:

* Reuse the existing global mosaic / SDS implementation’s chunking and memory management.
* Ensure all new logic operates on **indices / descriptors**, not raw data blobs.
* Keep SDS batch building and mega-tile assembly linear in N and tile-size-agnostic.

Example:

* If the user has 60 frames, they might end up with ~12 mega-tiles of 5 frames.
* If the user has 10 000 frames, the same logic must still work, creating a reasonable number of batches, without exhausting memory, using the existing streaming approach.

---

## 3. Flow control rules in the worker

You will adjust the **flow control** in `zemosaic_worker.py` (most likely in `run_hierarchical_mosaic(...)` or equivalent orchestration function):

### 3.1. Define / detect SDS mode flag

There is already a `sds_mode_flag` or equivalent boolean derived from:

* user config,
* Filter GUI overrides,
* Seestar detection.

You must reuse this flag; do not invent new ones.

### 3.2. Required decisions

**Case A — SDS OFF**

* **Do NOT call `assemble_global_mosaic_sds`.**
* **Do NOT call `assemble_global_mosaic_first`.**
* Force the code so that:

  * `final_mosaic_data_HWC` remains `None` after any global-WCS branch,
  * causing the worker to fall through into the **classic Phase 3 master-tile pipeline**.

**Case B — SDS ON**

* Call `assemble_global_mosaic_sds(...)` as the **primary** strategy.
* Only if SDS fails (no batches / returns None):

  * fallback to `assemble_global_mosaic_first(...)`.
* Only if both SDS and Mosaic-First fail:

  * allow the classic Phase 3 master-tile pipeline as emergency fallback.

### 3.3. Important: do NOT change SDS batch policy

Previous work has implemented:

* `sds_min_batch_size`
* `sds_target_batch_size`
* coverage-based SDS grouping both in Filter GUI Qt and worker.

You must **not** modify this logic in this mission. Only adjust:

* when SDS is used vs classic,
* how we decide to enter or skip SDS / Mosaic-First / Phase 3.

---

## 4. Logging & diagnostics

You should preserve and/or slightly improve logging:

* When SDS is OFF and we commit to the classic master-tile path:

  * Log something like `"sds_off_classic_mastertile_pipeline"` (through the existing logging mechanism).
* When SDS is ON and SDS mega-tiles is attempted:

  * `"sds_on_mega_tile_pipeline"`.
* When SDS fails and we fallback to Mosaic-First:

  * `"sds_failed_fallback_mosaic_first"`.
* When both SDS and Mosaic-First fail and we fallback to Phase 3:

  * `"sds_and_mosaic_first_failed_fallback_mastertiles"`.

Any new log keys should follow the existing style but **do not require UI-visible localization** (they are primarily for logs).

---

## 5. Non-goals (what you MUST NOT do)

* Do NOT change Tk GUIs.
* Do NOT change the Qt Filter GUI behaviour, except where it injects SDS flags into worker overrides (only if necessary to align SDS ON/OFF).
* Do NOT modify:

  * ZeQualityMT internals,
  * lecropper logic,
  * Alt-Az cleanup,
  * Two-pass coverage renorm,
  * Phase 4.5 / inter-master merge internals.
* Do NOT change GPU / CPU logic or chunking.
* Do NOT remove / rename existing configuration keys.

Your work is strictly focused on **high-level flow control** in the worker to enforce:

> **SDS OFF → classic master tiles**
> **SDS ON → mega-tiles from RAW lights + super-stack**

while preserving scalability and existing SDS batch policy.

````

---

