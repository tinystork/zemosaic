# ZM-ZEGRID-ACTIVE-OOC-FALLBACK — bounded storage as an automatic ACTIVE fallback

Status: implemented on `fix/zegrid-active-ooc-fallback` (local commits only; Junior owns
publication). Applies to the **ACTIVE** scheduler only; legacy and observe are unchanged.

## What the product now does

`run_zegrid_mode(..., storage_profile=None)` (every product call from
`zemosaic_worker.py`) means **requested = `auto`**:

1. try the historical **dense** model first (unchanged plans, tasks, science);
2. if, and only if, a **qualified host-RAM refusal** occurs, switch the run ONCE to the
   **bounded** profile (`BoundedMemmapCanonicalProvider` + a fresh
   `BoundedCountingRejectionSink` per Cell) and retry at the **same** geometry, halo,
   contributors and science identity;
3. if bounded also refuses, fail loudly with BOTH causes (never a false success).

There is **no GUI/config selector** and no new dependency. `bounded` is not advertised as a
user option; it is reached automatically, or explicitly through the internal
`storage_profile="dense" | "bounded"` seam (tests / A-B witnesses). An explicit value is
**forced**: AUTO witnesses never fall back silently.

## Fallback phases (safe boundaries)

| Phase | Trigger | Geometry |
|---|---|---|
| `plan_layout` | first layout filter / complete bundle build refuses for RAM | same layout request, halo and floors |
| `admission_preflight` | pre-gauge ACTIVE decide/admit refuses for RAM | pinned to the exact admitted `(nx, ny)` and refinement rank |
| `run_wave_N` | mandatory RUN checkpoint refuses for host RAM at a wave boundary | exact frozen layout/rank; completed waves kept, only not-yet-started Cells are rebuilt bounded |

A wave boundary has no in-flight unit, so no partial result can survive. There is exactly
ONE profile transition per run (`fallback_count <= 1`): no dense/bounded ping-pong and no
recursive restart of the whole run.

At `run_wave_N`, a bounded rebuild that cannot reproduce the exact layout (or the completed
cell prefix) is refused and the original honest refusal propagates. GPU→CPU runtime
fallbacks keep the **effective** profile; the profile travels in the 7th task element and in
the prepared/record payloads, so a child never reads the parent's global to know which
provider/sink to use.

## Which errors qualify

Only memory-qualified refusals:

* `ModeInfeasible` (no stacking mode/geometry fits the RAM budget);
* `LayoutInfeasible` carrying an explicit budget miss (`... exceeds/exceed budget ...`);
* the post-refinement rank-window refusal (the bounded window admitted no candidate after
  memory-driven refinement);
* the RUN checkpoint `R55_REFUSE_HOST_RAM` reason.

NOT eligible and always propagated unchanged: scientific floors without a budget miss,
WCS/FITS errors, permissions, cache corruption, disk full, programming errors,
cancellation. A halo/floor-only refusal never triggers a retry, so genuinely invalid floors
are not hidden; an AUTO retry that also fails under bounded is reported as a double refusal.

## Provenance

* GUI/log: `ZeGrid: storage policy — requested=... effective=...` at entry, plus one WARN
  line per transition with phase, reason code and detail.
* Manifest (ACTIVE only): `storage` block with `requested`, `effective`, `model`, `auto`,
  `forced`, `fallback_count`, `dense_refusal` (`phase`, `reason_code`, `detail`) and the
  ordered `transitions`. Legacy/observe keep their exact previous manifest shape.
* Refusal records distinguish "no plan fits in either profile" from a single-profile refusal.

## Invariants preserved

RAM 80/90, VRAM 70 % / headroom 85 % / legacy 50 %, `max_halo_overhead 50 %`, KGPU=1,
`cleanup sync→purge`, the single whole-batch GPU→CPU fallback, halo/K/offset/feather
semantics, normalisation/rejection/weights/combine. No automatic H128 reduction, no raw
loss, no swap counted as RAM. `linear_fit` is out of scope.

## Not proven here

* Real CUDA execution and Windows spawn/close/remove smoke: **NOT RUN** in this mission
  environment (unchanged from Lot C).
* A full Caldwell/M106 mosaic dense-vs-bounded parity remains a host-RAM gate; this mission
  proves the 816-header AUTO planner path and small complete synthetic mosaics instead.
