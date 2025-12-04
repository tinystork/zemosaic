# Follow-Up Checklist for Codex

Please verify and iterate until ALL boxes are checked:

## 🔁 Overlapping Batches
- [ ] Overlapping batch logic implemented in worker
- [ ] Overlap parameter linked from GUI → config → worker
- [ ] Logs indicate overlap cap and effective step

## 🔁 Frame Duplication
- [ ] batches smaller than TARGET_STACK are expanded via duplication
- [ ] duplication respects cap and preserves ordering
- [ ] logs output: "Duplicating frames: original=X → final=Y"

## 🔁 Salvage Mode
- [ ] salvage triggers only when n_used < MIN_SAFE_STACK
- [ ] ZeQualityMT disabled only for the salvage tile
- [ ] quality crop disabled only for the salvage tile
- [ ] logs output: "Tile N salvage mode"

## 🔁 Pipeline Safety
- [ ] No changes to ASTAP, WCS, reprojection logic
- [ ] Phase 5 untouched
- [ ] CPU/GPU fallback untouched
- [ ] Existing output folder structures unchanged

## 🔁 Behavioural Verification
- [ ] No more holes in mosaics even with aggressive rejections
- [ ] Master tiles show stable coverage maps
- [ ] Reproject output remains smooth between adjacent tiles
- [ ] Overlapping batches reduce boundary seams

If any item is missing or incorrect, revise and re-submit.
````

