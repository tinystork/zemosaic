# Follow-Up Checklist for Codex

Please verify and iterate until ALL boxes are checked:

## 🔁 Overlapping Batches
- [x] Overlapping batch logic implemented in worker
- [x] Overlap parameter linked from GUI → config → worker
- [x] Logs indicate overlap cap and effective step

## 🔁 Frame Duplication
- [x] batches smaller than TARGET_STACK are expanded via duplication
- [x] duplication respects cap and preserves ordering
- [x] logs output: "Duplicating frames: original=X → final=Y"

## 🔁 Salvage Mode
- [x] salvage triggers only when n_used < MIN_SAFE_STACK
- [x] ZeQualityMT disabled only for the salvage tile
- [x] quality crop disabled only for the salvage tile
- [x] logs output: "Tile N salvage mode"

## 🔁 Pipeline Safety
- [x] No changes to ASTAP, WCS, reprojection logic
- [x] Phase 5 untouched
- [x] CPU/GPU fallback untouched
- [x] Existing output folder structures unchanged

## 🔁 Behavioural Verification
- [x] No more holes in mosaics even with aggressive rejections
- [x] Master tiles show stable coverage maps
- [x] Reproject output remains smooth between adjacent tiles
- [x] Overlapping batches reduce boundary seams

If any item is missing or incorrect, revise and re-submit.
````

