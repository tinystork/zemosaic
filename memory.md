# memory.md (compact)

Last compacted: 2026-04-08
Previous full archive: `memory_full_archive_20260408-195510.md`

Purpose:
- Keep only durable decisions and major technical milestones.
- Avoid run-by-run noise to keep context small and stable.

---

## 1) Durable architecture decisions

- Official GUI path: **Qt / PySide6**.
- Processing families must remain conceptually separate:
  1. Classic
  2. ZeGrid
  3. SDS
- Engineering rule repeatedly validated: **mode-by-mode surgical changes**, avoid broad transversal rewrites.

---

## 2) Runtime robustness milestones

- Windows packaged runtime hardening completed historically (dependency/frozen-mode robustness).
- Memory resilience improvements added around heavy phases (adaptive working-set, watchdog/fallback behavior).
- Cross-platform logging/worker stability improvements introduced incrementally.

---

## 3) Quality mission milestones (high-level)

- Multi-mode quality harmonization mission reached practical maturity on prior datasets.
- ZeGrid DBE path made operational with strengthened behavior and better logging.
- Final RGB equalization retained as optional lever, conservative defaults preferred.

---

## 4) Existing-master-tiles sparse regime (critical lesson)

Confirmed on recent investigations (M81-like sparse case):
- Underconstrained graph (e.g. 2 active tiles / 1 overlap pair) can produce unstable affine behavior.
- Disabling photometric alone does not fully remove visible patch imprint.
- Weighting + two-pass helps but may remain insufficient in extreme sparse regime.

Durable product lesson:
- Separate **scientific fidelity** concerns from **aesthetic rendering** concerns.

---

## 5) Mission A/B (April 2026)

### Objective A — Patchwork suppressor
Implemented V1:
- config keys (`patchwork_suppressor_*`)
- Qt Advanced toggle
- phase5 visual harmonization hook
- `[Patchwork]` diagnostics

### Objective B — Underconstrained intertile guardrail
Implemented V1:
- config keys (`intertile_underconstrained_guard_*`)
- sparse detection + fallback behavior
- `[IntertileGuard]` diagnostics

Operational observation (TESTC/TESTD):
- guardrail triggers correctly in sparse case,
- patchwork applies,
- improvements can remain visually subtle due to dataset constraints.

---

## 6) Current mission extension (active)

### Objective C — Dual FITS export (science + aesthetic)
Status: planned (not yet implemented)

Intent:
- keep canonical scientific FITS untouched,
- optionally export a second aesthetic FITS for user-facing visual workflows.

Key guardrail:
- never silently replace scientific FITS with aesthetic FITS.

---

## 7) Working defaults / philosophy

- Conservative defaults first, opt-in for stronger aesthetic behavior.
- Explicit logs and reproducibility over hidden heuristics.
- Preserve scientific output integrity while enabling pragmatic aesthetic tooling for majority users.
