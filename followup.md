---

# ✅ **followup.md (step-by-step tasks for Codex)**

```markdown
# FOLLOW-UP TASKS — CENTRALIZED ETA USING ETACalculator

Follow these tasks exactly. Do not modify anything not explicitly listed.

---

## 1. Import the centralized ETA system

In both:
- `zemosaic_gui.py`
- `zemosaic_gui_qt.py`

Add:

```python
from zemosaic_time_utils import ETACalculator, format_eta_hms
````

---

## 2. Initialize the ETA calculator

In the method that starts the stacking process (Tk: `start_processing`, Qt: corresponding slot):

Add:

```python
self._eta_calc = ETACalculator(total_items=100)
self._eta_seconds_smoothed = None
```

Place this **before launching the worker thread**.

---

## 3. Replace local ETA math with ETACalculator

Locate the code that handles global progress updates.
Replace the ETA computation block with:

```python
self._eta_calc.update(int(global_progress))
eta = self._eta_calc.get_eta_seconds()

if eta is not None:
    if self._eta_seconds_smoothed is None:
        smoothed = eta
    else:
        smoothed = 0.3 * eta + 0.7 * self._eta_seconds_smoothed

    self._eta_seconds_smoothed = smoothed

    eta_str = format_eta_hms(smoothed)
    self._set_eta_label(eta_str)
```

---

## 4. Preserve GPU/CPU override behavior

Locate the worker ETA override handlers.
Do **NOT** modify them, only ensure that:

* When override is active
  → **Skip ETACalculator update**

* When override ends
  → Resume ETACalculator uninterrupted

The existing override logic must stay exactly as it is.

---

## 5. Deduplicate formatting

Remove any local helpers like:

* `_format_eta_string`
* `_human_readable_eta`
* Any custom hh:mm:ss logic

Use instead:

```python
eta_str = format_eta_hms(seconds)
```

---

## 6. Do not modify anything else

* No changes to worker code
* No changes to queue messages
* No changes to progress calculation
* No changes to phase weighting
* No changes to Tk or Qt layouts
* No changes to batch size behavior
* No changes to icons / translations

---

## 7. Testing checklist

Codex must ensure the following still works:

* ETA updates continuously during stacking
* Switching between GPU ETA override and global ETA works
* CPU helper ETA override still forces the displayed ETA
* Final ETA resets to "Idle" at end
* Tk and Qt now produce identical ETA behavior
* No regression in processing flow or performance

---

## 8. SDS / Mosaic-First progress integration

1. **Phase label parity** — Update `zemosaic_gui_qt.ZeMosaicQtWorker._handle_payload` to mirror the Tk handling of `PHASE_UPDATE:*` so the Qt phase label reflects the worker signals even when no `STAGE_PROGRESS` messages arrive (as seen in SDS runs).
2. **SDS progress wiring** — Intercept the worker payloads keyed as `p4_global_coadd_progress` (and the matching `..._finished`) in both GUIs. Use their `done`/`total` counts to synthesize a Phase 4 stage update so `_update_stage_progress` and ETACalculator advance even when the worker skips the legacy stage callback.
3. **More descriptive phase text** — While the SDS handler is active, show an explicit string such as `P4 - Mosaic-First global coadd` in the phase label using localized text so operators can see the actual operation being performed.
4. **Testing** — Run an SDS sample (e.g., the `example/lights` bundle) and confirm ETA decreases smoothly, the phase label updates past Phase 1, and the tiles/files counters keep matching the log.

---

## End of follow-up

