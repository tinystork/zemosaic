AGENT MISSION FILE — ZEMOSAIC QT 2.0
Tabbed interface • Skin selector • Language manager • Tk/Qt backend switch

You are an autonomous coding agent improving the ZeMosaic application.
ZeMosaic provides two GUI front-ends:

Tkinter GUI (zemosaic_gui.py) — reference implementation

PySide6 / Qt GUI (zemosaic_gui_qt.py) — new interface

The launcher run_zemosaic.py chooses which GUI to run based on:

CLI flags (--qt-gui, --tk-gui)

Environment variable ZEMOSAIC_GUI_BACKEND

User preference in zemosaic_config.json → "preferred_gui_backend"

Default → "tk"

The Qt GUI is now being redesigned to match the Tk feature set, but with a more modern layout.

Your tasks focus on the Qt GUI only, without modifying any stacking / mosaic / solver logic.

🎯 GLOBAL OBJECTIVE

Upgrade the PySide6 GUI (zemosaic_gui_qt.py) to a stable, fully tabbed, themable, multilingual interface while maintaining 100% behavioural parity with the Tkinter GUI.

✔️ REQUIRED RESULTS
1. Qt interface layout

Implement a QTabWidget-based main window with six tabs in this exact order:

Main

Solver

System

Advanced

Skin

Language

All GUI controls already implemented must be moved to these tabs while preserving their behaviour.

The bottom command bar stays always visible outside the tabs:

Button Filter…

Button Start

Button Stop

Progress bar

ETA / phase information

No behavioural change is allowed — only relocation of widgets.

2. Tab contents
Main tab

Contains options required for a normal run:

Folders (input/output/global WCS)

Instrument / Seestar options

Basic mosaic options

Final output options

Solver tab

All solver-related configuration:

Solver selection

ASTAP executable / database / parameters

Astrometry.net / ANSVR if applicable

System tab

Low-level and diagnostic settings:

Memmap

Cache retention

GPU acceleration

Logging controls & live log panel

Advanced tab

Expert-only workflow settings:

Quality crop

Master tile crop

Alt-Az cleanup

ZeQualityMT options

Super-tiles / Phase 4.5

Radial weighting

Post-stack anchor review

All experimental toggles

Skin tab

Appearance + backend preference:

Theme mode:

System

Dark

Light

Persist theme to config

Apply theme live

Backend preference:

Tk GUI

Qt GUI

Persist backend to preferred_gui_backend key

Language tab

Multilingual support (shared with Tk):

A language QComboBox offering:

English (EN)

Français (FR)

Español (ES)

Polski (PL)

Must update UI live using zemosaic_localization.py

Saves language to config

On Qt restart, language loads before widgets

🌍 INTERNATIONALISATION REQUIREMENTS

New keys for:

Tab labels

Skin/backend settings

Language names

Notices / hints

Add es.json and pl.json, initially clones of English.

Changing language updates ALL visible Qt widgets instantly.

Tk GUI must pick up the new language on next launch.

🖥️ CROSS-PLATFORM REQUIREMENTS

All modifications must remain compatible with:

Windows

macOS

Linux

Rules:

Use only standard Qt (PySide6) features

File dialogs → system-native

Icons → existing ones only

No Windows-specific APIs

No OS-specific palette hacks

Qt theme implementation must rely only on:

QApplication.setPalette

QPalette

optional lightweight stylesheet

🧩 NON-GOALS (DO NOT TOUCH)

Stacking logic

Mosaic logic

Cropping pipeline

Worker threads

Solver backends

FITS/PNG creation

Tkinter GUI behaviour

All astro business-logic files

Only Qt GUI layout, theming, and language/backend settings may be changed.

🧱 IMPLEMENTATION GUIDELINES

Keep all existing slots/callbacks untouched.

Move widgets into tabs; do not modify what they do.

Add helper methods for:

Creating tab pages

Applying themes

Refreshing text on language change

Save theme + backend + language directly using zemosaic_config.

✅ DONE WHEN

Qt GUI has 6 functional tabs.

Themes work and persist.

Backend preference in Skin tab works on next launch.

Language tab manages EN/FR/ES/PL.

UI text updates instantly when switching languages.

Tk and Qt share the same config keys.

No functionality is lost compared to Tk GUI.

All workflows run identically.