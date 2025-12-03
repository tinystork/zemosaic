# Follow-up Instructions for Codex

Please verify and iterate according to this checklist:

## ✔ Functional Checklist
- [ ] Overlap % parameter appears in the GUI
- [ ] Default overlap=40% is applied
- [ ] Autosplit is replaced by overlapping sliding-window batching
- [ ] Same image may appear in multiple batches
- [ ] Output structure of batches remains unchanged
- [ ] No change in Phase 3, 5, or stacking logic
- [ ] No dependency on ZeQualityMT
- [ ] Logs clearly show batch sizes and overlap
- [ ] Setting overlap to 0 reproduces the previous strict partitioning

## ✔ Behavioural Checklist
- [ ] No more holes in the mosaic even when strong rejections occur
- [ ] Reproject normalizes overlaps correctly
- [ ] Stacking remains deterministic
- [ ] GPU/CPU fallback behaviour unaffected
- [ ] Performance remains acceptable

If any part does not validate, refine the implementation and re-submit the patch.
