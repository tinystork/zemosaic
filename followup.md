# Follow-up Instructions for Codex

Please verify and iterate according to this checklist:

## ✔ Functional Checklist
- [x] Overlap % parameter appears in the GUI
- [x] Default overlap=40% is applied
- [x] Autosplit is replaced by overlapping sliding-window batching
- [x] Same image may appear in multiple batches
- [x] Output structure of batches remains unchanged
- [x] No change in Phase 3, 5, or stacking logic
- [x] No dependency on ZeQualityMT
- [x] Logs clearly show batch sizes and overlap
- [x] Setting overlap to 0 reproduces the previous strict partitioning

## ✔ Behavioural Checklist
- [x] No more holes in the mosaic even when strong rejections occur
- [x] Reproject normalizes overlaps correctly
- [x] Stacking remains deterministic
- [x] GPU/CPU fallback behaviour unaffected
- [x] Performance remains acceptable

If any part does not validate, refine the implementation and re-submit the patch.
