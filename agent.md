# Mission Codex — Filter GUI mount/angle hardening (EQMODE + orientation)

## 1. Scope & goals

**Goal:**  
Harden clustering in `zemosaic_filter_gui.py` by:

1. Using the FITS header keyword `EQMODE` (e.g. Seestar S50: `EQ...uatorial mode`) to classify each source as **EQ / ALT_AZ / UNKNOWN**.
2. Ensuring **clusters/master tiles do not mix EQ and ALT_AZ frames**:
   - Split groups by mount mode before any orientation-based split.
3. Keep the existing **orientation auto-split** (PA_DEG) logic, but now it runs on mode-homogeneous groups.

This mission is **Filter GUIs only** (Tk + Qt):

- ✅ `zemosaic_filter_gui.py`      (Tk Filter GUI)
- ✅ `zemosaic_filter_gui_qt.py`   (Qt / PySide6 Filter GUI)
- ❌ No changes to worker modules (`zemosaic_worker.py`, etc.)
- ❌ No changes to Phase 5 / GPU / SDS logic

The aim is to make clustering more robust, especially for Seestar datasets mixing EQ + ALT/AZ, without changing the underlying worker logic, SDS mode, or GPU safety.

---

## 2. Background

### 2.1. Problem

Currently, clustering and borrowing do not explicitly segregate:

- Equatorial mode (`EQMODE=1` on Seestar),
- Alt-az mode (`EQMODE=0`),
- “Unknown” / other instruments (no EQMODE present).

This can lead to:

- Clusters (master tiles) mixing EQ and ALT/AZ frames,
- Borrowing (border frame duplication) re-mixing modes even if clustering was split.

Given:

- ALT/AZ stacks have rotational artefacts and coverage “triangles/bands”,
- EQ stacks are more stable and rectangular,

it is scientifically safer to avoid mixing them within the same master tile.

### 2.2. Existing tools

- FITS headers for Seestar S50 include `EQMODE` with documented meaning:
  - `1` → equatorial mode,
  - `0` → alt-az mode.
- Filter GUIs already:
  - Read headers,
  - Build `candidate_infos` lists,
  - Have an orientation-based split (`_split_group_by_orientation`) based on `PA_DEG`,
  - Optionally run borrowing via `apply_borrowing_v1(...)`.

We want to **use EQMODE** to derive a `MOUNT_MODE` tag and enforce:

- “No EQ+ALT_AZ mixture per group”,
- Borrowing that respects this segregation.

---

## 3. Tasks

### 3.1. New helper: classify mount mode from header [x]

Add a small helper function to classify mount mode from a FITS header:

```python
def _classify_mount_mode_from_header(header: Any) -> str:
    """Return 'EQ', 'ALT_AZ', 'EQMODE_<N>', or 'UNKNOWN' based on header['EQMODE'].

    Rules:
    - If header is falsy or has no EQMODE -> 'UNKNOWN'.
    - Coerce EQMODE to int safely.
        - 1          -> 'EQ'
        - 0          -> 'ALT_AZ'
        - other int  -> f"EQMODE_{N}"
    - On any error -> 'UNKNOWN'.
    """
````

Implementation requirements:

* If `header` is falsy → return `"UNKNOWN"`.
* Read `EQMODE = header.get("EQMODE")`.
* Attempt to coerce to `int`:

  * `1` → `"EQ"`
  * `0` → `"ALT_AZ"`
  * any other int `N` → `f"EQMODE_{N}"`.
* If conversion fails → return `"UNKNOWN"`.
* Be tolerant to types (`str`, `float`, etc.), and never raise.

This function **must not** depend on any GUI or Tk/Qt object.
It can be defined in each GUI file or in a small shared helper, as long as you avoid circular imports.

---

### 3.2. Attach `MOUNT_MODE` to candidate_infos [x]

In the Filter GUI clustering path (Tk + Qt), each selected item is converted to an `entry` dict (`candidate_infos` building).

Current code looks like:

```python
entry = dict(item.src)
if "path" not in entry:
    entry["path"] = item.path
...
if "header" not in entry:
    entry["header"] = item.header
...
candidate_infos.append(entry)
```

Extend this block so that each `entry` has a `MOUNT_MODE` field:

* Retrieve a header object:

  ```python
  hdr = entry.get("header") or item.header
  ```

* Set `MOUNT_MODE` if not already present:

  ```python
  if "MOUNT_MODE" not in entry:
      entry["MOUNT_MODE"] = _classify_mount_mode_from_header(hdr)
  ```

Constraints:

* Do **not** overwrite any existing `entry["MOUNT_MODE"]` if present.
* If `hdr` is `None` or the header has no usable `EQMODE`, `_classify_mount_mode_from_header` will return `"UNKNOWN"`.

This ensures Seestar frames get `"EQ"` / `"ALT_AZ"` tags, while all other cameras remain `"UNKNOWN"` and behave as before.

For the **Qt Filter GUI** (`zemosaic_filter_gui_qt.py`):

* Locate the equivalent clustering path where `candidate_infos` (or an equivalent list of dicts) is built.
* Apply the **same logic** (attach `header`, compute `MOUNT_MODE` via `_classify_mount_mode_from_header`) before appending the entry.
* You may either:

  * re-use the same helper implementation (by copy) in the Qt file, or
  * factor these tiny helpers into a small shared section/module, **as long as** you do not introduce circular imports between Tk and Qt GUIs.

The semantics of `MOUNT_MODE` MUST be identical in Tk and Qt paths.

---

### 3.3. New helper: split a single group by mount mode [x]

Add a helper at module level (near `_split_group_by_orientation`), to split **one**
group by mount-mode homogeneity:

```python
def _split_group_by_mount_mode(group: list[dict]) -> list[list[dict]]:
    """Split group into subgroups based on entry['MOUNT_MODE'].

    Rules:
    - Modes: 'EQ', 'ALT_AZ', 'UNKNOWN', and possibly 'EQMODE_<N>'.
    - Unknowns join the majority mode when both EQ and ALT_AZ are present.
    - If effectively single-mode (or only UNKNOWN), return [group].
    """
```

Detailed behaviour:

1. Extract modes from the group:

   ```python
   modes = []
   for entry in group:
       mode = entry.get("MOUNT_MODE", "UNKNOWN")
       if not isinstance(mode, str):
           mode = str(mode)
       modes.append(mode)
   ```

2. Partition entries into three logical buckets:

   * `eq_entries`      → mode `"EQ"` or `"EQMODE_1"` (to be conservative)
   * `altaz_entries`   → `"ALT_AZ"`
   * `unknown_entries` → `"UNKNOWN"` or anything else.

   > Note: any `EQMODE_<N>` with `N != 1` is treated as UNKNOWN for the split (so they follow the majority).

3. Decide:

   * If only one of `eq_entries`, `altaz_entries`, `unknown_entries` is non-empty
     **or**
     if only UNKNOWN + a single other mode → return `[group]` (no split).
   * If you have both EQ and ALT_AZ present:

     * Determine majority among EQ vs ALT_AZ.
     * Attach `unknown_entries` to the majority group.
     * Return `[eq_like_group, altaz_like_group]`, keeping a stable order inside each.

Implementation notes:

* Do **not** crash if `group` is empty.
* Prefer to preserve the original ordering of entries within each subgroup.
* This helper is pure and does not access GUI components.

---

### 3.4. Apply mount-mode split before orientation auto-split [x]

In the clustering path where `cluster_func(...)` is called (both Tk and Qt Filter GUIs), we currently have:

```python
groups_initial = cluster_func(...)
...
groups_used = groups_initial
# (optional relax on epsilon)
...
angle_split_effective = ...
...
if auto_angle_enabled ...:
    # compute dispersion, use _split_group_by_orientation(...)
...
groups_after_autosplit = autosplit_func(...)
```

Insert a **mount-mode split stage** just after selecting `groups_used`
(and after any epsilon relax) but before orientation auto-split:

1. After `groups_used` is finally chosen:

   ```python
   groups_mode_guarded = []
   mode_splits = 0
   for grp in groups_used:
       subgroups = _split_group_by_mount_mode(grp)
       if len(subgroups) > 1:
           mode_splits += 1
       groups_mode_guarded.extend(subgroups)
   groups_used = groups_mode_guarded
   ```

2. Log once (INFO level) when `mode_splits > 0`:

   ```python
   logger.info("Mount-mode guard: split %d group(s) by EQMODE / MOUNT_MODE.", mode_splits)
   ```

3. Orientation auto-split (`_split_group_by_orientation`) then runs on **the already mode-homogeneous `groups_used`** exactly as before.

Notes:

* Do **not** change the semantics of `auto_angle_split_var`, `angle_split_deg_var`, `cluster_orientation_split_deg` settings.
* Do **not** change `auto_angle_detect_threshold` logic or `ANGLE_SPLIT_DEFAULT_DEG` in this mission.
* The only “hardening” is that groups that mix EQ + ALT_AZ are split before we look at PA_DEG dispersion.

---

### 3.5. Borrowing guard: borrowing must NOT re-mix EQ and ALT_AZ [x]

⚠️ Important: `apply_borrowing_v1(...)` can duplicate “border” frames across neighboring groups.
If called on a mixed-mode population, it can **re-introduce ALT_AZ into EQ groups (and vice versa)**,
which defeats the mount-mode guard.

Therefore, when borrowing is enabled (coverage mode), and **in both Filter GUIs (Tk + Qt)**,
apply borrowing **separately per mount-mode bucket**, but keep the external `borrow_stats`
API identical to today.

Objectif: empêcher STRICTEMENT tout mélange EQ + ALT_AZ après borrowing,
sans changer le comportement historique quand ALT_AZ n’est pas présent.

Règle de déclenchement (IMPORTANT):
- Appliquer borrowing par bucket (partition) UNIQUEMENT si les modes effectifs contiennent à la fois "EQ" et "ALT_AZ".
- Si on a seulement ("EQ" + "UNKNOWN") ou seulement ("ALT_AZ" + "UNKNOWN") ou seulement "UNKNOWN":
  -> comportement strictement identique à aujourd’hui: appel unique à apply_borrowing_v1(final_groups, ...)

Cas final_groups vide ou < 2:
- Retourner exactement le résultat de apply_borrowing_v1(final_groups, ...) (groups ET stats),
  pour rester bit-à-bit compatible avec l’existant.
#### 3.5.1. Helper wrapper `_apply_borrowing_per_mount_mode(...)`

Add a small helper in the Filter GUI layer (Tk side), e.g.:

```python
def _apply_borrowing_per_mount_mode(
    final_groups: list[list[dict]],
    logger: logging.Logger,
) -> tuple[list[list[dict]], dict[str, Any]]:
    ...
```

You will use the **same semantics** for the Qt Filter GUI (either by reusing the helper
or re-implementing it in the Qt file with identical behaviour).

Rules for this helper:

1. **Never modify** `zemosaic_utils.apply_borrowing_v1` itself.

   * Do not change its function signature.
   * Do not change the internal layout of the `stats` dict it returns.

2. If `final_groups` is empty or has fewer than 2 groups, simply return
   `(final_groups, stats)` where `stats` is a *single* dict with the same keys as
   `apply_borrowing_v1` would return (call `apply_borrowing_v1` once to get these stats).

3. Determine a **group-level mode** for each group (it should already be homogeneous
   after 3.4):

   * `group_mode = "EQ" | "ALT_AZ" | "UNKNOWN"`
   * Use a safe majority vote on the entries’ `MOUNT_MODE` (but do not crash if empty).

4. Compute the set of effective modes present among `group_mode`s.

   * If the set has size **0 or 1** (all UNKNOWN, or all EQ, etc.), keep behaviour
     **strictly identical** to the current code:

     * Call `apply_borrowing_v1(final_groups, None, logger=logger, ...)` once.
     * Return its `(groups, stats)` unchanged (no partitioning).

5. If there are **2 or more modes**:

   * Partition `final_groups` into buckets:

     * EQ bucket:      `group_mode == "EQ"`
     * ALT_AZ bucket:  `group_mode == "ALT_AZ"`
     * UNKNOWN bucket: `group_mode == "UNKNOWN"`

   * For each non-empty bucket:

     * If the bucket has fewer than 2 groups, keep it as-is and simulate a stats dict
       with `executed=False` and zeros ailleurs.
     * Else, call `apply_borrowing_v1(bucket_groups, None, logger=logger, ...)`
       and get `(bucket_groups_new, bucket_stats)`.

   * Recombine les groupes dans un ordre déterministe, par exemple :
     `final_groups_new = final_eq + final_alt + final_unk`

6. Build a **single aggregated stats dict** qui garde la même structure *plate*
   que `apply_borrowing_v1` aujourd’hui (pas de `{ "EQ": ..., "ALT_AZ": ... }` imbriqué) :

   * Initialise `global_stats` comme un dict vide.
   * Pour chaque `bucket_stats` :

     * Si `global_stats` est vide, deep-copy `bucket_stats`.
     * Sinon, pour toutes les clés présentes dans les deux :

       * Si les deux valeurs sont `int` / `float`, les sommer.
       * Si les deux valeurs sont `list`, étendre la liste.
       * Pour les autres types (ex. `str`), garder la valeur originale ou la dernière ; c’est purement debug.
   * S’assurer que `global_stats["executed"]` est le OR logique des `"executed"` de chaque bucket.

   Résultat : `global_stats` ressemble à la sortie d’un unique `apply_borrowing_v1`,
   et tout code existant qui attend un `borrow_stats` **plat** continue à fonctionner.

7. Retourner `(final_groups_new, global_stats)` depuis `_apply_borrowing_per_mount_mode`.

8. Ajouter un post-check (debug/info) après borrowing :

   * Compter les groupes où `{entry["MOUNT_MODE"]}` a une taille > 1 ; ça doit être 0.
   * Sinon, logger un warning mais **ne pas** crasher.

#### 3.5.2. Using the helper in Tk and Qt GUIs

Là où le Filter GUI appelle aujourd’hui :

```python
if coverage_enabled and final_groups:
    final_groups, _borrow_stats = apply_borrowing_v1(final_groups, None, logger=logger)
```

remplacer par :

```python
if coverage_enabled and final_groups:
    final_groups, _borrow_stats = _apply_borrowing_per_mount_mode(final_groups, logger=logger)
```

Appliquer le **même pattern** dans `zemosaic_filter_gui_qt.py` partout où borrowing est appelé.

Résumé des garanties :

* S’il n’y a qu’un **seul mode effectif** (tout UNKNOWN, ou tout EQ, etc.),
  le comportement de borrowing est bit-à-bit identique à la version actuelle.
* Borrowing n’est jamais appliqué à des groupes de modes différents.
* `borrow_stats` reste un dict plat (même structure qu’aujourd’hui), donc
  **pas de changement d’API** pour les consommateurs existants.

---
Post-check (debug/info) après borrowing:
- Pour chaque groupe, calculer has_eq / has_altaz en normalisant:
    EQ si entry["MOUNT_MODE"] == "EQ"
    ALT_AZ si entry["MOUNT_MODE"] == "ALT_AZ"
  (tout le reste est traité comme UNKNOWN)
- Invariant: (has_eq and has_altaz) doit être False pour tous les groupes.
- Si violation: logger WARNING (ne pas crasher).

## 4. Constraints / non-goals

* **No UI change**:

  * La checkbox “Auto split by orientation” et le spinbox d’angle restent tels quels.
  * Pas de nouveau bouton / label / option pour le mount mode.

* **No worker changes**:

  * Ne pas toucher `zemosaic_worker.py`, `zemosaic_align_stack_*`, la GPU safety ou la Phase 5.

* **No SDS behavior change**:

  * Le path ZeSupaDupStack (`sds_mode`) qui construit les coverage batches via `_build_sds_batches_for_indices` doit rester intact.

* **No change to borrowing core API**:

  * Ne pas modifier `zemosaic_utils.apply_borrowing_v1`, `BORROW_ENABLE`, ni la structure du dict `stats` qu’il renvoie.
  * Tous les garde-fous de borrowing par mount mode doivent vivre côté Filter GUI (Tk + Qt) via le wrapper `_apply_borrowing_per_mount_mode`.

* **Backwards compatibility**:

  * S’il n’y a pas de header ou pas de `EQMODE`, le comportement doit rester identique (tout `"UNKNOWN"` → pas de split par mode).
  * Pas de nouvelle dépendance obligatoire.

---

## 5. Logging expectations

Nouveau log (niveau INFO) quand le split par mode se déclenche :

* Clé de traduction : `filter_log_mount_mode_split`
* Message par défaut :
  `"Mount-mode guard: split {N} group(s) by EQMODE / MOUNT_MODE."`

Le wrapper de borrowing peut aussi logger en DEBUG un résumé par mode si besoin, mais ce n’est pas obligatoire. L’important est :

* Un log clair quand `mode_splits > 0`.
* Aucune erreur fatale si `MOUNT_MODE` manque ou est incohérent.

---

## 6. Tests to run (dev side)

1. **Test “unitaire” dans un REPL** :

   * Créer des groupes synthétiques avec différentes distributions de `MOUNT_MODE`.
   * Vérifier `_split_group_by_mount_mode` :

     * all-EQ → 1 groupe
     * EQ + ALT_AZ → 2 groupes
     * EQ + UNKNOWN → 1 groupe
     * ALT_AZ + UNKNOWN → 1 groupe
     * EQ + ALT_AZ + UNKNOWN → 2 groupes, UNKNOWN attaché au mode majoritaire.

2. **Test manuel GUI – Seestar EQ only** :

   * Dataset Seestar S50 en pur mode EQ (`EQMODE=1`).
   * Lancer Filter GUI (Tk puis Qt) et comparer avec la version précédente :

     * Nombre de groupes et taille des groupes identiques (à bruit près).
     * Orientation auto-split inchangé.
     * Pas de split par mode (ou `N=0` dans les logs).

3. **Test manuel GUI – Seestar mix EQ + ALT/AZ** :

   * Dataset mêlant `EQMODE=1` et `EQMODE=0`.
   * Lancer clustering (Tk / Qt) :

     * Log “Mount-mode guard…” avec `N > 0`.
     * Nombre de groupes plus élevé qu’avant.
     * Orientation auto-split uniquement au sein d’un même mode.
   * Activer coverage/borrowing :

     * Vérifier qu’aucun groupe final ne contient de mix EQ + ALT_AZ.
     * Vérifier que `borrow_stats` est bien un dict plat (pas de dict imbriqué par mode).

4. **Test manuel GUI – Autres caméras (pas de EQMODE)** :

   * Dataset sans `EQMODE`.
   * Clustering avant / après :

     * Pas de log guard significatif,
     * Nombre de groupes comparable,
     * Auto-split par orientation identique.

5. **Parité Tk / Qt** :

   * Pour au moins un dataset EQ only et un dataset mix EQ+ALT/AZ :

     * Faire le même run dans les deux GUIs.
     * Vérifier que les résultats (groupes, logs, borrowing) sont cohérents ; seules de petites différences d’ordre interne sont acceptables.

Si tu vois une régression (explosion de groupes, logs bizarres, crash), il faut d’abord revoir :

* `_classify_mount_mode_from_header`
* `_split_group_by_mount_mode`
* L’endroit où `MOUNT_MODE` est attaché à `candidate_infos`
* Le wrapper `_apply_borrowing_per_mount_mode`

avant d’aller toucher les workers ou la Phase 5.

Additionally, for **Qt / PySide6**:

* Run at least one full manual test path using `zemosaic_filter_gui_qt.py` on the mêmes datasets.
* Vérifier que :

  * Les comptes et compositions de groupes matchent le Tk Filter GUI pour des réglages identiques.
  * Les logs de mount-mode guard et le comportement de borrowing sont cohérents entre Tk et Qt.

````
