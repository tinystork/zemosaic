# followup.md — execution tracker (aesthetic-first mission)

## Mission status snapshot

### Implemented
- [x] Patchwork suppressor (A) wired + logged
- [x] Underconstrained intertile guardrail (B) wired + logged

### Not implemented yet
- [ ] Dual FITS export (`_science` + `_aesthetic`) (C)
- [ ] Aesthetic hole-fill / seam completion (D)

---

## Field observation (latest)

From TESTC/TESTD/TESTE style runs:
- clear visual progress vs older NGC6888 runs,
- but residual holes remain in specific regions,
- logs indicate most gains are photometric harmonization,
- geometric coverage mask changes remain small.

Interpretation:
- current A/B stack is working,
- final remaining issue is primarily **coverage hole completion in visual branch**.

---

## Active objective shift

Primary product target is now:
- aesthetic mosaic output that is smooth/homogeneous and editor-friendly,
- while preserving scientific output and high throughput.

---

## TODO by objective

### C — Dual FITS export
- [x] Add config keys (`export_aesthetic_fits`, suffixes)
- [x] Add UI switch + help text
- [x] Export both files in one run when enabled
- [x] Add explicit logs + header provenance

### D — Aesthetic hole-fill
- [x] Add hole-mask detection from coverage/alpha
- [x] Add local inpainting/fill (aesthetic branch only)
- [x] Add blend/feather controls
- [x] Add star/detail protection guard
- [x] Add diagnostics (`[AestheticFill]` suggested)

### Performance guardrails
- [ ] Measure runtime overhead on sparse + dense runs
- [ ] Measure peak RAM delta
- [x] Keep default mode fast (bounded pass count)

Validation snapshot (2026-04-10, log corpus `/media/tristan/X10 Pro/mosaic/test`):
- Reviewed 62 run logs + 4 crash-breadcrumb files.
- Bounded-pass guard confirmed in logs via `Winsor streaming limit set to ... frame(s) per pass`:
  - 24 frames/pass on current M106 Linux runs (`M106_8`, `M106_9`, `M106_10`),
  - 256 frames/pass on heavy historical Markarian/M31 runs.
- Runtime overhead (sparse+dense) is not yet signed off: latest M106 enhanced runs (dual export + hole-fill) are not all completed/paired against strict same-input baseline.
- Peak RAM delta is partially observable but not yet finalized as a guardrail metric:
  - breadcrumbs maxima: `M106_8` 95.7% (~7366 MB), `M106_9` 96.6% (~7438 MB), `M106_10` 97.0% (~7466 MB), `marikian/C` 97.9% (~31775 MB).
  - need final paired completed runs to publish authoritative delta numbers.
- Additional heavy-run context from `/media/tristan/X10 Pro/mosaic/andromeda/out` (older code branch, still useful for guardrail scale):
  - root Andromeda run: `38136s` (~10h35), peak proc RSS `~9705 MB`, max system RAM usage `~63.7%`.
  - `V4 RUN A/B/D`: `34986s / 17519s / 18480s`, peak proc RSS `~3645 / 3192 / 4092 MB`, max system RAM usage around `~94–95%`.
  - all these runs show `winsor_max_frames_per_pass=0` in snapshots; bounded-pass protection is still enforced by runtime caps where applicable (seen in newer logs via explicit `Winsor streaming limit set ...`).

---

## Proposed config/profile defaults (draft)

Baseline fast aesthetic profile:
- `patchwork_suppressor_enabled=true`
- `patchwork_suppressor_strength=normal`
- `intertile_underconstrained_guard_enabled=true`
- `intertile_underconstrained_force_mode=offset_only`
- `export_aesthetic_fits=false` (default conservative)

Strong aesthetic profile (opt-in):
- `patchwork_suppressor_strength=strong`
- `aesthetic_hole_fill_enabled=true`

---

## Validation plan (next runs)

1. Sparse existing-master baseline
2. Sparse A+B
3. Sparse A+B+C
4. Sparse A+B+C+D
5. Dense sanity + perf check

Record for each:
- visual verdict,
- key log lines,
- runtime and memory overhead.

---

## Exit criteria

Mission considered complete when:
1. aesthetic output is consistently editable and visually smooth,
2. science output remains trustworthy and clearly separated,
3. throughput remains compatible with high-volume workflows.

## Markarian crash investigation update (2026-04-10)

### Current status
- Root cause is still not identified (intermittent behavior).
- A full Markarian run (`.../marikian/C/`) completed successfully with outputs + no `TASK_RESULT_EXCEPTION` in breadcrumbs.
- Fast stress run (`FAST_CRASH_HUNT_STRESS_20260410_090040`) also completed in ~12m33s, but with only 5 groups and `WORKERS_PHASE3: Utilisation de 1 worker(s)`, so limited concurrency stress.
- A larger stress run (450 FIT + target groups ~195) is much slower and currently shows long Phase 3 and Phase 5 durations.
- Crash breadcrumbs remain enabled and available for future incident forensics (`worker_crash_breadcrumbs.jsonl` + `worker_last_state.json`).

### Resource utilization findings from latest large stress log
- VRAM mostly low-use: about 1.1 to 2.0 GB used on ~8.2 GB total.
- CPU median low (around 16%).
- RAM system load is non-trivial (roughly 18 to 26.6 GB used on 32.5 GB), but process-level usage is moderate and variable.
- GPU chunk cap remains conservative (`gpu_max_chunk_bytes=134217728`), likely limiting throughput.

### Guardrail to keep
- Any tuning must preserve the ability to process very large datasets (up to 10k images) without OOM.
- Prefer adaptive memory scaling and bounded chunk growth over unsafe static maxing.


### New evidence + test plan (2026-04-10 evening)
- New captures (`capture 1` and `capture 2`) show abrupt stop in Phase 3 without Python traceback, with pending in-flight tiles.
- Breadcrumb sequence now localizes the stop after `P3_STACK_CORE_DONE` / `P3_POSTSTACK_EQ_STATE` and before save for some tiles.
- Windows WER around crash time reports `LiveKernelEvent 141/117` patterns (GPU watchdog/driver reset), consistent with native GPU instability rather than Python exception flow.

Immediate validation plan:
- [ ] Re-run same dataset with GPU fully disabled (`use_gpu_phase5=false`, `stack_use_gpu=false`, `use_gpu_stack=false`) to confirm CPU stability baseline.
- [ ] Run normalization A/B focused on NaN-heavy data: `linear_fit` vs `sky_mean` (CPU first, then GPU).
- [ ] If CPU is stable and GPU fails, keep crash-track open as GPU/driver-sensitive path; do not treat as resolved.

Implementation note:
- [x] Qt propagation fix: toggling GPU in GUI now synchronizes all three keys (`use_gpu_phase5`, `stack_use_gpu`, `use_gpu_stack`).

### Remaining TODO (crash track)
Status: ⏸️ paused by decision (2026-04-10) because recent runs are stable and no native crash reproduced.
- [x] Keep crash breadcrumbs enabled and compare event timelines when failure reappears.
- [ ] (Paused) Produce a safe A/B memory profile: `SAFE+` then `AGGRESSIVE`, with explicit rollback.
- [ ] (Paused) Measure phase timing deltas (P3/P5), peak RSS, and swap impact on each profile.
- [ ] (Paused) Correlate any future native crash with WER/Event Viewer module data.
- [ ] (Paused) Decide production defaults only after proving no OOM regressions on large stacks.

## Mission update (2026-04-10) — ETA GUI first

### Decision
Before any new performance work (auto-chunking dynamic, RAM/VRAM tuning), fix ETA behavior in GUI.

### Why this is first
- Current long runs (Markarian stress) complete, but ETA is not reliable enough for operational confidence.
- We need trustworthy timing to quantify gains/losses from future tuning.

### Active TODO (ETA)
- [x] Audit current ETA pipeline (emitters + GUI display path).
- [x] Separate phase ETA and global ETA clearly in GUI/state.
- [x] Add smoothing + hysteresis to reduce oscillations.
- [x] Reinitialize ETA model at phase boundaries (notably P3 and P5).
- [x] Add ETA confidence/status (warmup/learning/stable) to prevent misleading early values.
- [x] Add diagnostic logs for ETA source and windowing.
- [x] Validate on M106 runs (normal + resume).

### ETA mission status
- ✅ Considered accomplished (2026-04-10, validated in real runs).
- GUI now maintains a coherent countdown between updates.
- Resume startup ETA no longer relies on obviously unrealistic short values.

### Constraints for next step (after ETA)
- Keep 10k-images-no-OOM objective as hard requirement.
- Dynamic chunking must be SAFE by default (reserved RAM/VRAM + rollback on pressure).
- Aggressive profile remains opt-in.

### Next block after ETA is validated
- [x] Implement SAFE_DYNAMIC chunking profile.
- [x] Run A/B benchmark vs baseline.
- [x] Report total time gain + P3/P5 gain + memory/swap deltas.

A/B results (2026-04-10, same M106 dataset):
- SAFE_DYNAMIC run: `/media/tristan/X10 Pro/mosaic/test/M106_10/`
- BASELINE run: `/media/tristan/X10 Pro/mosaic/test/M106_AB_BASELINE/`

Measured deltas (baseline relative to SAFE_DYNAMIC):
- Total runtime: `1814.57s` vs `1695.35s` → baseline `+119.22s` (`+7.03%`) => SAFE_DYNAMIC faster.
- Phase 3 runtime: `607.74s` vs `565.90s` → baseline `+41.84s` (`+7.39%`).
- Phase 5 runtime: `1182.62s` vs `1103.43s` → baseline `+79.19s` (`+7.18%`).
- Peak process RSS: `2038.5 MB` vs `2542.1 MB` → baseline lower (`-503.6 MB`, `-19.81%`).
- Max system RAM%: `85.3%` vs `89.5%` → baseline lower (`-4.2 pts`).
- Max swap used: `9183 MB` vs `8920 MB` → baseline higher (`+263 MB`, `+2.95%`), so SAFE_DYNAMIC slightly better on swap pressure.

Interpretation:
- SAFE_DYNAMIC gives a clear throughput gain on this dataset (about 7% overall, consistent on P3/P5).
- Memory profile tradeoff observed: higher process RSS / system RAM peaks, but slightly reduced swap usage.

---

## New dedicated mission (2026-04-10 evening) — Resume progressif et tolérant

Context trigger:
- Current resume is still too strict (all-or-nothing in Phase 1 in practical scenarios).
- Large runs can lose 1–3 days after interruption/crash even when most artifacts are still reusable.

### Design & implementation backlog (open)
- [ ] Write short design note (data model, manifests, local invalidation, atomic writes, partial resume cases, controlled-abort cases).
- [ ] Introduce `.zemosaic_state/` manifests (`run_state`, `phase1`, `stack_plan`, `master_tiles`, `final_assembly`).
- [x] Implement partial Phase 1 resume (recompute only missing/invalid entries, never global purge for one miss).
- [x] Implement stack-plan reuse with compatibility check.
- [x] Implement master-tile reuse with per-tile validation (readable FITS + WCS + shape + alpha/coverage coherence).
- [ ] Implement resume selection by highest valid level (final > masters > plan > phase1 partial > full restart).
- [x] Add `Quit and Save Progress` (priority): clean stop, flush checkpoints/state, safe restart.
- [ ] Add `Pause` semantics (secondary milestone).
- [ ] Add explicit resume logs (`phase1 partial reuse X/Y`, `stack plan reused`, `master reused A/B`, `rebuild tiles [...]`).
- [ ] Add interruption/resume test scenarios matching acceptance criteria.

### Acceptance scenarios (open)
- [ ] Phase 1 cache partial hole: only missing cache rebuilt.
- [ ] Interruption after grouping: plan reused if compatible.
- [ ] Interruption mid-Phase 3: valid masters reused, only missing/corrupted rebuilt.
- [ ] Quit-and-save then restart: no full rescans/recompute when not required.
- [ ] Local corruption remains local (no global invalidation).

