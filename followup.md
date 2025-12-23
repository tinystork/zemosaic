# Follow-up: implement + verify GPU safety scaling cap

## Files to edit
- `zemosaic_gpu_safety.py`

## Patch details
Edit `_clamp_gpu_chunks()`.

### Before (current behavior)
- cap starts at 256 MB
- then `cap = min(cap, 0.6 * total_vram)`
- then `cap = min(cap, 0.8 * free_vram)`
=> cannot ever exceed 256 MB.

### After (new behavior)
- cap starts at 256 MB
- if total_vram known: cap = max(256 MB, 10% total_vram), then cap = min(cap, 2 GB)
- if free_vram known: cap = min(cap, 80% free_vram)
- cap floor = 32 MB
- keep plan.gpu_max_chunk_bytes clamp semantics identical.

## Suggested diff (apply carefully)

```diff
diff --git a/zemosaic_gpu_safety.py b/zemosaic_gpu_safety.py
index 0000000..0000000 100644
--- a/zemosaic_gpu_safety.py
+++ b/zemosaic_gpu_safety.py
@@ -163,19 +163,27 @@ def _clamp_gpu_chunks(plan: ParallelPlan, ctx: GpuRuntimeContext) -> bool:
     if not use_gpu or not ctx.safe_mode:
         return False
 
-    cap_bytes = 256 * 1024 * 1024  # 256 MB default ceiling in safe mode
-    if ctx.vram_total_bytes:
-        try:
-            cap_bytes = min(cap_bytes, int(ctx.vram_total_bytes * 0.6))
-        except Exception:
-            pass
-    if ctx.vram_free_bytes:
-        try:
-            cap_bytes = min(cap_bytes, int(ctx.vram_free_bytes * 0.8))
-        except Exception:
-            pass
-    cap_bytes = max(cap_bytes, 32 * 1024 * 1024)
+    # Safe mode GPU chunk budget:
+    # - default conservative base: 256 MB
+    # - scale up on "real" GPUs: 10% of total VRAM (min 256 MB), hard-capped to 2 GB
+    # - never exceed 80% of free VRAM
+    # - absolute floor: 32 MB
+    cap_bytes = 256 * 1024 * 1024
+    if ctx.vram_total_bytes:
+        try:
+            scaled = int(ctx.vram_total_bytes * 0.10)
+            cap_bytes = max(cap_bytes, scaled)
+        except Exception:
+            pass
+        cap_bytes = min(cap_bytes, 2 * 1024 * 1024 * 1024)
+    if ctx.vram_free_bytes:
+        try:
+            cap_bytes = min(cap_bytes, int(ctx.vram_free_bytes * 0.8))
+        except Exception:
+            pass
+    cap_bytes = max(cap_bytes, 32 * 1024 * 1024)
Verification checklist
Grep logs for [GPU_SAFETY] ... gpu_chunk_mb=:

On large VRAM GPU in safe_mode, it should now be >256 MB when free VRAM allows.

On low VRAM / low free VRAM, it should reduce accordingly.

Ensure no exceptions on systems without psutil/wmi/cupy (this file guards optional deps already).

Notes
This patch intentionally does not change the “0 => auto rows_per_chunk” behavior (separate patch).