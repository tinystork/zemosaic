# SCI-01 — Grid WSC numerical characterization (NO FIX)

- **Mission:** `ZM-SCI-01-GRID-WSC-CHAR-20261004`
- **Phase:** implementation (characterization-only witness)
- **Branch:** `science/sci-01-grid-wsc-characterization`
- **Base / HEAD:** `2a56d2be845373c03fbe7a99d54c72e87f5601c9` (`origin/beta`)
- **Status:** DIVERGENCE PROVEN / CHARACTERIZED — no correction, no tuning, no winner chosen.
- **Witness:** `tests/test_grid_wsc_characterization.py` (26 tests, all passing)

This document records **current behavior and its magnitude** so a future product
decision has evidence. It deliberately does **not** correct, harmonize, tune, or
choose a winner.

---

## 1. Objective

SCI-01 is a **characterization-only** scientific witness. It proves and quantifies
the existing numerical divergence between:

1. **Grid CPU** and **Grid GPU-legacy fallback**, which call
   `_reject_outliers_winsorized_sigma_clip` **without** an explicit `wsc_impl`, so
   `resolve_wsc_impl()` resolves `ZEMOSAIC_WSC_IMPL` env > (supplied zconfig) > default;
   **these Grid calls supply no zconfig**, so their effective order is **env > default**,
   landing on **PixInsight WSC by default**.
2. **Grid GPU core**, which passes `winsorized_sigma_clip` to
   `zemosaic_stack_core.stack_core`, whose implementation is an explicitly
   **simplified median/std sigma clip** — neither PixInsight WSC nor legacy quantile
   winsorization.

No algorithm is declared scientifically superior. Impact is data-dependent and
quantified below. Any science behavior change remains a **human gate**.

---

## 2. Symbol / caller map (anchored to `2a56d2be`)

| Symbol | File / function | Role in divergence |
| --- | --- | --- |
| `resolve_wsc_impl(zconfig=None)` | `src/zemosaic/core/robust_rejection.py` | Resolver: `ZEMOSAIC_WSC_IMPL` env (`pixinsight` \| `legacy_quantile`) → **supplied** `zconfig` keys (`wsc_impl`/`winsor_impl`/`stack_winsor_impl`) → default `pixinsight`. Invalid env falls through honestly; no new mode is created. The Grid helper calls pass **no `zconfig`**, so for those routes the effective order is **env > default** (the `zconfig` tier is inert). |
| `_reject_outliers_winsorized_sigma_clip(..., wsc_impl=None)` | `src/zemosaic/zemosaic_align_stack.py` | `effective_impl = wsc_impl or resolve_wsc_impl()`. PixInsight branch → `wsc_pixinsight_core(np, …)` + `broadcast_to`; legacy branch → SciPy `winsorize` + Astropy `sigma_clipped_stats` (rewinsorizes rejected pixels). |
| `wsc_pixinsight_core(xp, X_block, …, sigma_low, sigma_high, max_iters, weights_block, huber, …)` | `src/zemosaic/core/robust_rejection.py` | PixInsight-like winsorized sigma clip (median/MAD init, Huber IRLS scale update), float64 internal, float32 output. |
| `_stack_weighted_patches(patches, weights, config, …)` | `src/zemosaic/grid_mode.py` | **Grid CPU.** Normalize → call helper **without `wsc_impl`** (`max_workers=1`) → weighted mean/median combine. |
| `_stack_weighted_patches_gpu(patches, weights, config, …)` | `src/zemosaic/grid_mode.py` | **Grid GPU.** If `stack_core` available → `stack_core(images=normalized, weights=cp_weights, stack_config={'normalize_method':'none', …}, backend='gpu')`. If `stack_core` is `None` → legacy cp fallback: `cp.asnumpy` → same helper **without `wsc_impl`**. |
| `stack_core(images, weights, stack_config, backend, …)` | `src/zemosaic/zemosaic_stack_core.py` | Shared core. `winsorized_sigma_clip` branch = **simplified** `nanmedian`/`nanstd` clip (placeholder); **not** PixInsight WSC, **not** quantile winsorization. |

Line numbers are intentionally avoided except where anchored to SHA `2a56d2be` in the
test file docstrings; symbol names are the stable identity.

---

## 3. Numeric matrix (CPU runtime, deterministic float32 `1×1×1` stacks)

All values are reproduced against the real code paths in the witness; they are
re-derived, not copied. `sigma_low = sigma_high = 2.5`, `winsor_limits = (0.05, 0.05)`,
`stack_norm_method = 'none'`, `stack_final_combine = 'mean'`, equal positive weights (=1).

| Corpus (per-frame values) | Grid CPU default = PixInsight WSC (env absent) | Grid CPU explicit `legacy_quantile` | `stack_core` simplified (CPU backend) | abs delta (core − default grid) | rejected / support |
| --- | --- | --- | --- | --- | --- |
| impulse `[0,0,0,0,100]` | `5.0e-11` | `16.0` | `20.0` | `20.0` | core: 0% rejected, support 5/5 · grid: support 5/5 |
| nondegenerate `[1,1,1,2,100]` | `1.0` | `17.08` | `1.25` | `0.25` | core: 20% rejected, support 4/5 · grid: support 5/5 |
| gradient `[1,2,3,4,5,100]` | `6.5740323` | `15.208333` | `3.0` | `3.5740323` | core: 16.67% rejected, support 5/6 · grid: support 6/6 |

Observations:

- **Grid CPU default** (PixInsight WSC) winsorizes the `100` outlier to ~`2.5e-10`
  (impulse collapses to ~`0`, not the `20.0` arithmetic mean), and to the WSC median
  for the other corpora.
- **Grid CPU legacy** (SciPy winsorize + Astropy sigma-clipped stats) rewinsorizes the
  `100` outlier to ~`80` (`80`, `80.4`, `76.25`) and averages — `16.0`, `17.08`,
  `15.208333`.
- **`stack_core` simplified** treats `100` as a plain outlier: rejected in the
  nondegenerate and gradient corpora (median/`nanstd` bounds), kept in the impulse
  corpus (where `0` is the median and `100` is within `±2.5·σ` of a large σ). Its
  `weight_sum` drops to the surviving support count.
- **Materiality:** the deltas are large for impulse (`20.0`) and gradient
  (`3.57`), smaller but non-trivial for nondegenerate (`0.25`). Impact is
  **data-dependent** and dominated by high-contrast outliers.

**`weight_sum = N` for Grid CPU / legacy does *not* mean no sample was clip-processed
or rejected internally.** The PixInsight helper returns a finite rewinsorized/broadcast
aggregate (and the legacy path rewinsorizes rejected samples in place), so the Grid
finite-mask combine sees `N` finite inputs and produces `weight_sum = N`. `weight_sum`
here is the finite-mask support of the *combine*, not an internal rejection count; only
`stack_core` exposes an explicit rejected-% and a reduced `weight_sum`.

All three implementations agree on output **shape/dtype** (`H×W×C`, `float32`) on these
**all-finite** corpora; they disagree on the *value* and on *rejection/support*. General
NaN / masked / finite-mask semantics are **NOT_RUN** here — the selected corpora are
all-finite `1×1×1` stacks and do not exercise NaN handling or finite-mask behavior.

---

## 4. Evidence labels

- **STATIC** — source inspection of the resolver, helper, `_stack_weighted_patches`,
  `_stack_weighted_patches_gpu`, and `stack_core` (Section 2).
- **CPU RUNTIME** — real execution of `_stack_weighted_patches` (Grid CPU, env absent /
  `pixinsight` / `legacy_quantile`) and `stack_core(backend='cpu')` on the three
  corpora (Section 3; witness tests B, C, legacy).
- **ROUTING SEAM** — hermetic NumPy-backed fake CuPy + recording `stack_core`
  stand-in to capture the Grid GPU core call contract (`backend='gpu'`,
  `normalize_method='none'`, `rejection_algorithm='winsorized_sigma_clip'`,
  `sigma_clip_low/high=2.5`, no `winsor_limits`), and the Grid GPU-legacy fallback
  route through the real helper **without `wsc_impl`** (witness tests D, E).
- **NOT_RUN (physical GPU)** — no physical CuPy/GPU numerical execution was performed;
  no CPU↔GPU parity claim is made.
- **INFERENCE** — that Grid GPU core and Grid GPU-legacy would follow the same
  *dispatcher* semantics as their CPU counterparts is inferred from the identical
  call contracts captured at the seams, not from GPU arithmetic.

---

## 5. What the witness proves (dynamically)

1. **Resolver contract (hermetic env).** `resolve_wsc_impl()` returns `pixinsight`
   with env absent; `legacy_quantile` (case-insensitive, trimmed) overrides;
   invalid/empty env falls through to `pixinsight` and never creates a new mode.
2. **Grid CPU omits `wsc_impl`** (a `*args/**kwargs` spy over the real helper asserts
   `'wsc_impl'` is absent from the forwarded kwargs — an explicit `wsc_impl=None` would
   bind the key, so this genuinely proves omission), and its env-absent result
   **equals** its explicit-`pixinsight` result and matches the **direct
   `wsc_pixinsight_core`** output within float32 accumulation tolerance.
3. **`stack_core` materially diverges** from default Grid CPU on all three corpora,
   with abs deltas quantified in Section 3.
4. **Grid GPU core routing seam** captures the real `stack_core` call contract
   (backend `gpu`, normalize `none`, requested winsor rejection, sigma `2.5/2.5`,
   weights/images passed, `winsor_limits` dropped). The seam cannot pass while
   bypassing `stack_core` (recording stand-in must fire exactly once).
5. **Grid GPU-legacy routing seam** (with `stack_core` forced `None`) routes through
   the real helper **without `wsc_impl`** (dynamic `*args/**kwargs` spy proves the
   keyword is absent); env absent resolves to PixInsight (impulse → `5e-11`),
   `legacy_quantile` switches to the legacy implementation (impulse → `16.0`), with no
   silent path switch.

---

## 6. Conclusion

- **SCI-01 divergence is PROVEN / CHARACTERIZED.** The default/dispatch behavior and
  its hidden coupling (env `ZEMOSAIC_WSC_IMPL` → default PixInsight when callers omit
  `wsc_impl`; the general resolver order is env > **supplied** zconfig > default, but
  these Grid calls supply no zconfig, so their effective order is env > default) are
  explicit and dynamically verified.
- **Impact is data-dependent**, ranging from `0.25` to `20.0` absolute on the selected
  corpora, dominated by high-contrast outliers.
- **No algorithm is declared scientifically superior; no correction is chosen.**
- Production behavior is **unchanged** (no `src/`, config, version, dependency,
  packaging, or workflow edits).

---

## 7. Recommended next decision options (decision only — not implementation)

1. **Keep the divergence documented** (accept as-is, treat Grid CPU/legacy = PixInsight
   and Grid GPU core = simplified as known, monitored behavior).
2. **Unify `stack_core` on PixInsight WSC** — only after physical-GPU qualification of
   the PixInsight WSC core on CuPy (currently NOT_RUN).
3. **Expose/persist an explicit Grid implementation policy** (a real config/UI setting
   that makes the WSC implementation choice explicit at the Grid level instead of
   relying on an env override + default).

Any science behavior change is a **human gate** and is explicitly out of scope here.
