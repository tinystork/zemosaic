# Follow-up — Filter GUI mount/angle hardening

## 1. Quick checklist after implementation

- [x] `zemosaic_filter_gui.py` compiles (no syntax errors).
- [x] `zemosaic_filter_gui_qt.py` compiles (no syntax errors).
- [x] New helper `_classify_mount_mode_from_header` exists and behaves safely on:
      - missing header,
      - missing EQMODE,
      - non-numeric EQMODE,
      - EQMODE = 0 / 1.
- [x] New helper `_split_group_by_mount_mode` exists and:
      - returns `[group]` when there is no real ambiguity,
      - splits EQ + ALT_AZ groups into separate subgroups.
      - treats any `EQMODE_<N>` with `N != 1` as UNKNOWN for splitting (follows the majority bucket).
- [x] New helper `_apply_borrowing_per_mount_mode(final_groups, logger)` exists in the Filter GUI layer and:
      - calls `apply_borrowing_v1(...)` **at most once per mount-mode bucket**,
      - returns a **single flat** `borrow_stats` dict with the same keys as `apply_borrowing_v1` normally returns.
- [x] Both Tk and Qt Filter GUIs use `_apply_borrowing_per_mount_mode(...)` instead of calling `apply_borrowing_v1(...)` directly.
- [x] `candidate_infos` entries now have a `"MOUNT_MODE"` key when possible.
- [x] Mount-mode split is applied **before** orientation auto-split, but only in the clustering path (not SDS coverage batches).
- [x] Existing orientation auto-split logic (PA_DEG, `_split_group_by_orientation`) is untouched.
- [x] Post-check après borrowing: aucun groupe ne contient simultanément "EQ" et "ALT_AZ" (UNKNOWN ignoré).
- [x] Le wrapper _apply_borrowing_per_mount_mode partitionne uniquement quand EQ ET ALT_AZ sont tous deux présents,
      sinon il appelle apply_borrowing_v1 une seule fois sur final_groups (comportement identique à avant).
      - Même avec 0 ou 1 groupe, on appelle apply_borrowing_v1 une fois pour récupérer des stats cohérentes.

---

## 2. Behaviour verification scenarios

### 2.1. Synthetic checks (REPL / small script)

- Create a mock `group` list of dicts with only `MOUNT_MODE` keys and confirm:

  1. `["EQ", "EQ", "EQ"]` → 1 subgroup.
  2. `["ALT_AZ", "ALT_AZ"]` → 1 subgroup.
  3. `["UNKNOWN", "UNKNOWN"]` → 1 subgroup.
  4. `["EQ", "UNKNOWN", "EQ"]` → 1 subgroup (UNKNOWN attached to EQ).
  5. `["ALT_AZ", "UNKNOWN", "ALT_AZ"]` → 1 subgroup.
  6. `["EQ", "ALT_AZ"]` → 2 subgroups.
  7. `["EQ", "ALT_AZ", "UNKNOWN"]` → 2 subgroups, UNKNOWN in the majority bucket.

- Make sure order inside each subgroup reflects the original order for predictable behaviour.

---

### 2.2. Real data — Seestar EQ only

- Use a Seestar S50 project in **pure equatorial mode** (all headers contain `EQMODE = 1`).
- Steps:
  - Launch Filter GUI.
  - Load the dataset and run clustering with your usual settings.
  - Inspect the log:
    - “Mount-mode guard” should either:
      - not appear, or
      - appear with `N=0`.
    - Orientation auto-split logs should be present if enabled (as before).
  - Compare number of groups and group sizes with a pre-patch run:
    - They should match (modulo non-deterministic ordering), because all frames share the same mode.

---

### 2.3. Real data — Seestar mixed EQ + ALT/AZ

- Build or use a dataset mixing Seestar EQ and ALT/AZ captures (e.g. some headers with `EQMODE = 1`, others with `EQMODE = 0`).
- Steps:
  - Launch Filter GUI and load all frames.
  - Run clustering as usual.
  - Expected:
    - A log line `Mount-mode guard: split {N} group(s) by EQMODE / MOUNT_MODE.` with `N > 0`.
    - The number of groups after mount-mode split is higher than before (when you run the old version).
    - Orientation auto-split log still appears for groups with large PA dispersion.
  - Manually inspect:
    - A group that previously contained both EQ and ALT/AZ frames should now be split into two groups.
    - No crash, and GUI remains responsive.

---

### 2.4. Real data — Non-Seestar cameras

- Use data from another instrument (no EQMODE header).
- Steps:
  - Run Filter GUI clustering before and after the patch with the same settings.
  - Expected:
    - No “Mount-mode guard” log (or only with `N=0`).
    - Same number of groups and similar group composition (allowing for normal clustering noise).
    - Orientation auto-split behaviour unchanged.

If there is any visible regression (missing groups, fewer or more groups, weird logs), **do not** touch worker/GPU code. Instead, revisit:

- `_classify_mount_mode_from_header`
- `_split_group_by_mount_mode`
- The place where `MOUNT_MODE` is attached to `candidate_infos`.
- The wrapper `_apply_borrowing_per_mount_mode`.

---

### 2.5. Qt Filter GUI parity check

- Repeat at least one full workflow (EQ-only and mixed EQ+ALT/AZ) using the **Qt Filter GUI** (`zemosaic_filter_gui_qt.py`):
  - Load exactly the same datasets and reuse the same clustering / coverage settings.
  - Compare:
    - Number of groups and their sizes.
    - Presence and content of mount-mode guard logs.
    - Behaviour of borrowing (no cross-mode mixing, same final group counts).
- Small differences in internal ordering are acceptable, but **semantics must match** the Tk Filter GUI.

---

## 3. Guardrails / things NOT to change

- Do not modify:
  - Any worker module (`zemosaic_worker`, `zemosaic_align_stack_*`, etc.)
  - GPU safety / Phase 5 logic
  - SDS / ZeSupaDupStack behaviour
  - The signature or stats structure of `zemosaic_utils.apply_borrowing_v1`
- Do not change the semantics of:
  - `auto_angle_split_var`
  - `cluster_orientation_split_deg`
  - `cluster_panel_threshold` / `panel_clustering_threshold_deg`
- Do not add new GUI widgets for mount mode.
- Do not make `EQMODE` mandatory: the code must work on headers without this keyword.

---

## 4. Success criteria

- Mixed Seestar datasets (EQ + ALT/AZ) no longer produce clusters that mix modes in the same master tile.
- Orientation auto-split logs still show up as expected.
- Non-Seestar datasets behave exactly like before.
- No new crash or noticeable slowdown in Filter GUI clustering.
````

