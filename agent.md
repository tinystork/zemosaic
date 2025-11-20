
# AGENT MISSION — CENTRALIZED ETA SYSTEM FOR ZEMOSAIC

You are an autonomous coding agent working on the ZeMosaic project.

Your task is to **centralize ETA calculation** using the new module:

`zemosaic_time_utils.py`

This module contains the class **ETACalculator**, which will become the *single source of truth* for ETA estimation in both GUIs (Tkinter and PySide6/Qt).

The goal is:
- Keep the existing behavior EXACTLY identical for users.
- Preserve all overrides:
  - GPU ETA overrides
  - CPU helper ETA overrides (`ETA_UPDATE:hh:mm:ss`)
- Keep the current global weighted progress system (per-phase weighting).
- Improve ETA stability and consistency by using ETACalculator.

Do **not** modify:
- Business logic in worker (`zemosaic_worker.py`)
- Phase weight mapping
- Progress queue message formats
- Worker ETA messages
- Batch modes (0 and >1), which MUST remain untouched

Your mission is limited to:
- GUI ETA logic in:
  - `zemosaic_gui.py` (Tk)
  - `zemosaic_gui_qt.py` (Qt)

Only refactor ETA computation, not the layout or workflow.


## OBJECTIVES

### 1. Add centralized ETA calculation

Import ETACalculator in both GUIs:

```python
from zemosaic_time_utils import ETACalculator, format_eta_hms
````

### 2. Instantiate ETACalculator on "Start"

Before the run begins:

```python
self._eta_calc = ETACalculator(total_items=100)
self._eta_seconds_smoothed = None
```

### 3. Replace local ETA math with ETACalculator

Every time `global_progress` (0–100) is updated:

```python
self._eta_calc.update(int(global_progress))
eta = self._eta_calc.get_eta_seconds()

if eta is not None:
    # keep exponential smoothing
    if self._eta_seconds_smoothed is None:
        smoothed = eta
    else:
        smoothed = 0.3 * eta + 0.7 * self._eta_seconds_smoothed
    self._eta_seconds_smoothed = smoothed

    eta_str = format_eta_hms(smoothed)
    self._set_eta_label(eta_str)
```

### 4. Preserve ETA overrides

#### GPU override

Keep EXACTLY the existing code path:

* When GPU helper sends ETA → **ignore ETACalculator** temporarily
* Show the GPU ETA
* Resume ETACalculator when override ends

#### CPU helper override

Same as GPU override.

### 5. Replace local ETA formatting helpers with centralized formatter

Remove duplicated formatting functions in GUIs:

* `_format_eta_string`
* `_human_readable_eta`

Replace with:

```python
from zemosaic_time_utils import format_eta_hms
```

### 6. Do not change:

* Worker logic
* GUI layout
* Threading
* Queue handling
* Worker → GUI messages
* Phase system
* Progress percentages
* Any behavior of batch processing
* Icons / translations

## FILES TO MODIFY

* `zemosaic_gui.py`
* `zemosaic_gui_qt.py`
* `zemosaic_time_utils.py` (only if needed: add format_eta_hms)

## EXPECTED RESULT

* ETA becomes more stable, smoother, and consistent.
* Tk and Qt display the same ETA.
* No regression in functionality.
* GPU & CPU ETA overrides still work.
* Rest of the project remains unchanged.

````

