# Follow-up checklist – duplicated label in WCS / Master Tile controls

Please confirm each item before closing the task.

## Implementation review

- [ ] Show the **final snippet** of the layout construction code for the **WCS / Master tile controls** section in `zemosaic_filter_gui_qt.py`.
  - Include the `addRow` / `addWidget` calls for:
    - Max ASTAP instances
    - Coverage-first clustering toggle
    - Over-cap allowance (%)
    - Overlap between batches (%)
    - Any remaining orientation/split control
- [ ] Explicitly point out which **duplicate label / widget(s)** were removed and in which file(s).
- [ ] If any **orientation/auto-split** field remains:
  - Show how it is labeled and which config field it controls.

## Localization

- [ ] List the **localization keys** now used for each label in that group.
- [ ] If new keys were added:
  - [ ] Show the additions to `locales/en.json`.
  - [ ] Show the corresponding entries in `locales/fr.json`.

## Visual check (screenshots optional but helpful)

- [ ] Confirm that in the Qt filter dialog:
  - “Overlap between batches (%)” appears once with a correct label.
  - There is no garbled or overlapping text below it.
  - The layout looks aligned and consistent.
- [ ] Confirm the same after switching language EN ↔ FR.

## Behavior / regression

- [ ] Confirm that the **Overlap between batches (%)** value still flows correctly into the configuration / worker (i.e. no broken connections).
- [ ] Confirm that the change did **not** modify behavior of SDS toggle, coverage-first clustering, or ASTAP instance count.

If any doubts remain, mention them explicitly so they can be tested by the maintainer on real datasets.
