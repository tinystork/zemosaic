# AGENT MISSION FILE — ZEMOSAIC QT PORT

You are an autonomous coding agent working on the **ZeMosaic** project.

The repository already contains:
- `run_zemosaic.py`
- `zemosaic_gui.py`
- `zemosaic_filter_gui.py`
- `zemosaic_worker.py`
- `zemosaic_localization.py`
- `zemosaic_config.py`
- `zemosaic_utils.py`
- `lecropper.py`
- `zequalityMT.py`
- `tk_safe.py`
- `locales/en.json`, `locales/fr.json`
- various helper modules (astrometry, cleaner, etc.)

Your job is to gradually add and refine a **PySide6 (Qt)** GUI backend, **without breaking** the existing Tkinter-based GUI or the astro/stacking business logic.


## GLOBAL MISSION

**Goal:**  
Introduce a complete PySide6 GUI for ZeMosaic (main window + filter GUI), side-by-side with the existing Tkinter GUI:

- Tkinter GUI must continue to work exactly as before.
- PySide6 GUI must reach feature parity over time (but can start minimal).
- No core business logic must be rewritten (WCS solving, stacking, coverage, alpha, lecropper pipelines, ZeQualityMT, etc. stay in the existing Python files).

**Long-term objective:**

- `zemosaic_gui_qt.py` — main ZeMosaic Qt GUI
- `zemosaic_filter_gui_qt.py` — filter Qt GUI
- A small launcher (e.g. in `run_zemosaic.py`) that selects the backend:
  - Default: Tk
  - Optional: Qt, via environment variable `ZEMOSAIC_GUI_BACKEND=qt` or a CLI flag like `--qt-gui`.

You will NOT accomplish this in a single run.  
You must work **incrementally**, guided by `followup.md`.


## HARD CONSTRAINTS

These rules are **strict** and must always be respected:

1. **Do not delete or break the Tkinter GUI**:
   - Do NOT remove or heavily refactor `zemosaic_gui.py`, `zemosaic_filter_gui.py`, or `tk_safe.py`.
   - Tkinter remains the default backend unless explicitly changed by the user.

2. **Do not rewrite business logic**:
   - Do NOT rewrite:
     - The WCS / stacking / mosaicing algorithms in `zemosaic_worker.py`.
     - The cropping / Alt-Az / alpha / coverage logic in `lecropper.py`.
     - The ZeQualityMT filter logic in `zequalityMT.py`.
     - Configuration loading/saving logic in `zemosaic_config.py`.
     - Localization infra in `zemosaic_localization.py`.
   - You may call these functions, pass parameters, and handle callbacks, but you must not re-implement their internal algorithms inside the GUI.

3. **Keep public APIs backward compatible**:
   - The following must keep working as-is:
     - Tk main GUI entry point (`zemosaic_gui.py` main function or equivalent).
     - `zemosaic_filter_gui.launch_filter_interface(...)` (Tk variant).
     - Worker functions like `run_hierarchical_mosaic_process(...)`.
   - If you add new Qt variants, use *new* functions or modules (`*_qt`) so that existing code paths are untouched.

4. **PySide6 is optional**:
   - Users may not have PySide6 installed.
   - The project must still import and run the Tk GUI even if PySide6 is missing.
   - Qt modules (`zemosaic_gui_qt.py`, `zemosaic_filter_gui_qt.py`) must use guarded imports.

5. **Small, testable steps**:
   - You must never attempt a huge refactor in a single step.
   - Work in **phases** and mark progress in `followup.md`.
   - Do not skip ahead to later tasks until previous checkboxes are satisfied.


## WORKFLOW AND BEHAVIOR

Whenever you are invoked to work on this project, you MUST:

1. **Read `agent.md` (this file)** entirely, to understand:
   - The mission.
   - The constraints.
   - The phase structure.

2. **Read `followup.md`**:
   - Identify the next unchecked, top-most item in the checklist.
   - Work ONLY on that next item (or that very small group of items).

3. **Plan before coding**:
   - Briefly outline what changes you will make (file by file).
   - Ensure these changes do not violate the constraints.

4. **Apply changes**:
   - Modify only the necessary files.
   - Avoid noisy style-only changes; focus on the current task.
   - Keep code readable and consistent with the surrounding style.

5. **Update `followup.md`**:
   - Check the boxes for the tasks you actually completed.
   - Optionally add notes or small corrections in the “Notes / Known Issues” section.
   - Task C guard: if `followup.md` is not staged when you review `git status`, stop and update it before proceeding.

6. **Summarize**:
   - After changes, summarize what you did and where.
   - Mention if any TODO or uncertainties were found and where they are left in comments.

### Follow-up Discipline Guard (Task C)

- **Every commit touching Qt migration work MUST edit `followup.md` in the same change set.**
- Before running tests or drafting the final summary, execute `git status followup.md` to verify it is staged. If it is missing, open the file and:
  - Mark any completed checklist items with `[x]`.
  - Add partial progress notes or caveats in the Notes section.
- Do not defer the update to later commits; Task C mandates that the log mirrors the current session immediately.

You MUST NOT:
- Reorder tasks.
- Implement tasks that are not the next unchecked item in `followup.md`.
- Propose future patches beyond the current task; just execute the next scheduled one.


## PHASES OVERVIEW (HISTORICAL)

Initial Qt port was structured in six phases (see `followup.md` for detailed checklists):

1. Qt skeleton and guarded imports  
2. Basic Qt main window structure  
3. Layout and widgets mirroring Tk GUI  
4. Worker integration and threading  
5. Qt filter GUI  
6. Polishing and parity with Tk  

Most of these are now completed and tracked in `followup.md`.  
Further work now focuses on **post-portage audit tasks** described below.


## POST-PORTAGE AUDIT TASKS (MANDATORY)

A technical audit of the PySide6 port has identified **two critical parity gaps** compared with the Tkinter GUI, and **one structural requirement** related to process discipline.

These MUST be implemented sequentially and only via entries added in `followup.md`.  
You, the agent, MUST follow these new tasks after the original Phases 1–6 are completed.


### Task A — GPU Configuration Parity

The audit detected that the Qt UI currently updates:

- `use_gpu_phase5`
- `gpu_selector`

But Tk also updates:

- `stack_use_gpu`
- `use_gpu_stack`

Some stacking utilities still rely on these legacy keys.  
Therefore, cross-backend consistency requires:

**TASK A — REQUIREMENT**

Whenever the Qt GUI writes GPU-related settings, the agent MUST also:

- Write `stack_use_gpu` based on the Qt GPU checkbox.
- Write `use_gpu_stack` consistently.
- Ensure that config snapshots are aligned with what Tk produces.
- Guarantee that switching back and forth between Tk and Qt preserves GPU behavior identically.

This change MUST NOT modify business logic modules — only the GUI-to-config layer.


### Task B — GPU Helper & ETA Feedback Parity

Tkinter GUI handles special GPU helper messages generated by workers:

- `global_coadd_info_helper_path`
- GPU helper warnings
- ETA override events
- Aggregated GPU progress

The Qt implementation currently treats these worker messages as plain log strings.

Result: GPU-accelerated runs under PySide6 **lose ETA overlays and helper feedback**.

**TASK B — REQUIREMENT**

The agent MUST extend the Qt queue/payload handling to:

- Detect GPU-helper payloads exactly as Tk does.
- Trigger proper GUI updates:
  - ETA labels.
  - Overlays.
  - Warnings.
- Mirror Tk’s `_handle_gpu_helper_*` logic, using Qt-safe patterns (signals/slots).
- Do this without duplicating business logic — reuse worker payloads and semantics.


### Task C — Ensure `followup.md` Is Always Updated

Codex MUST follow this rule:

Whenever the agent completes any of the above tasks (or any future task from `followup.md`):

- It MUST update `followup.md`.
- It MUST mark the corresponding checkbox(es) as `[x]`.
- It MUST add notes if the implementation required deviation or added subtasks.
- It MUST NOT proceed to the next item until explicitly instructed by the user (or until the current item is clearly complete).


## HOW TO USE FOLLOWUP.MD

`followup.md` is your **checklist and logbook**.

- Before each coding session:
  - Read `followup.md`.
  - Identify the next unchecked item.
- During the session:
  - Implement only that item (or that small coherent group of items).
- After the session:
  - Update `followup.md`:
    - Mark completed tasks with `[x]`.
    - Add short notes if something remains partially done or has caveats.
- Never mark as `[x]` a task you haven’t actually implemented and verified.


## SUMMARY

Your role:

- Be the disciplined Qt migration agent for ZeMosaic.
- Always respect the constraints and phases in this document.
- Always keep `followup.md` up to date.
- Make incremental, safe progress toward a fully functional PySide6 GUI backend,
  **without ever breaking** the existing Tkinter implementation or the core astrophotography logic.
- Complete the Audit Parity Tasks (A, B, C) one by one, in the order defined in `followup.md`.
- Ensure that configuration written by Qt and Tk is identical for any feature both support (including GPU toggles and helper feedback).
