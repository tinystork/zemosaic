# ZeSolver integration (optional)

This document describes how ZeMosaic integrates the optional
[ZeSolver](https://github.com/zemosaic/zesolver) plate solver through an
internal, transport-neutral **SolverPort** boundary.

The guiding contract is simple and non-negotiable:

> **ZeSolver is optional.** Its absence, incompatibility or an unhealthy
> installation must never break importing ZeMosaic, and must never change the
> behaviour of the existing legacy solvers (ASTAP / Astrometry.net / ANSVR /
> NONE).

---

## 1. SolverPort boundary

The internal boundary lives in `zemosaic/solver_port.py`. It imports **only the
standard library** and never imports `zesolver` (or any submodule) at import
time. It defines:

- `SolveStatus` (`SOLVED`, `FAILED`, `SKIPPED`, `UNAVAILABLE`)
- `SolverOutcome` — the transport-neutral result returned by every adapter.
  It carries `wcs`, `header`, `should_write_header_back`, `failure_code`,
  `message` and `backend_used`.
- `DiscoveryState` (`NOT_INSTALLED`, `INCOMPATIBLE`, `UNHEALTHY`, `AVAILABLE`)
- `SolverDiscovery` — the result of a lazy ZeSolver discovery/health check.
- `SolverAdapter` — the minimal `Protocol` implemented by both adapters.
- `LegacySolverAdapter` — a faithful wrapper preserving the exact pre-port
  dispatch semantics (see §3).

`zemosaic.zemosaic_worker` routes the Phase 1 WCS solve through
`_solve_through_solver_port(...)`, which dispatches to either the legacy adapter
or the ZeSolver adapter. The SolverPort boundary is the **only** choke point
between the worker and any concrete solver.

---

## 2. Discovery states

`zemosaic.zesolver_adapter.discover_zesolver()` lazily imports the public API
module (`zesolver.api.v1`) and health-checks it. It never performs a catalog
scan, never imports GPU/CuPy and never touches the network.

| State            | Meaning                                                                                     |
|------------------|---------------------------------------------------------------------------------------------|
| `NOT_INSTALLED`  | `zesolver.api.v1` cannot be imported (ZeSolver absent).                                     |
| `INCOMPATIBLE`   | The public `API_MAJOR` (parsed from `API_VERSION` when absent) is not `1`.                  |
| `UNHEALTHY`      | The API is present but `probe(check_catalogs=False, check_gpu=False)` raised.               |
| `AVAILABLE`      | Public API v1 present, major version compatible and probe succeeded.                        |

Compatibility is decided **exclusively** on the public `API_VERSION` /
`API_MAJOR` (major == 1). It never inspects the Git branch, commit or product
version of the installed ZeSolver.

In the worker, a `ZESOLVER` solve is only attempted after discovery returns
`AVAILABLE`; any other state yields an `UNAVAILABLE` outcome (with
`failure_code` set to the discovery state) instead of raising.

---

## 3. Legacy preservation

The legacy solver choices and their semantics are **untouched**:

| Choice        | Pre-port behaviour (preserved byte-for-byte)                                              |
|---------------|--------------------------------------------------------------------------------------------|
| `ASTAP`       | Direct ASTAP solve.                                                                        |
| `ASTROMETRY`  | Astrometry.net solve, with ASTAP fallback when ASTAP paths are valid.                      |
| `ANSVR`       | ANSVR (local server) solve, with ASTAP fallback when ASTAP paths are valid.                |
| `NONE`        | Falls through to direct ASTAP (historical fall-through, intentionally kept).                |
| *(unknown)*   | Falls through to direct ASTAP (historical fall-through, intentionally kept).                |

`LegacySolverAdapter` delegates to the existing `solve_with_astrometry`,
`solve_with_ansvr` and `zemosaic_astrometry.solve_with_astap` callables and
reproduces the historical dispatch, progress-message keys and header-write
flags. `ZESOLVER` is a *new* sixth choice; it is not part of the legacy set and
can only be selected explicitly.

---

## 4. ZeSolverAdapter lifecycle

`zemosaic.zesolver_adapter.ZeSolverAdapter` talks to ZeSolver strictly through
the public `zesolver.api.v1` surface (see §6).

- **One runtime per settings fingerprint per process.** The worker caches the
  adapter in `_ZESOLVER_ADAPTER_INSTANCE`, keyed by a fingerprint of the
  runtime-affecting settings (`zesolver_resources_path`, `zesolver_gpu_policy`,
  `zesolver_backend_policy`). A `SolverRuntime` is created lazily on first use
  and shared for the lifetime of that fingerprint.
- **Fingerprint change rebuilds.** When any runtime-affecting setting changes
  between runs in the same process, the previous adapter is `close()`d and a
  fresh one is created — a stale runtime is never silently reused.
- **One session per thread.** `SolverSession` objects are kept in
  `threading.local` storage and created on demand; every tracked session is
  registered and closed together with the runtime.
- **Close at run end.** Both run entrypoints
  (`run_hierarchical_mosaic`, `run_hierarchical_mosaic_classic_legacy`) are
  wrapped with `@_close_zesolver_on_run_exit`, which calls
  `_close_zesolver_adapter()` on success, failure and cancellation. A
  long-lived process (e.g. a GUI hosting several batches) therefore never leaks
  a `SolverRuntime` between runs.
- **Idempotent close.** `close()` is safe to call repeatedly and from any
  thread; closing twice, or closing an adapter whose runtime was never created,
  is a no-op. After close the adapter remains usable — the next `solve()` lazily
  recreates a fresh runtime and session.

---

## 5. Settings keys

The following keys are read from the solver settings (`solver_settings` dict)
and surfaced in the Qt GUI configuration:

| Key                        | Meaning                                              | Default |
|----------------------------|------------------------------------------------------|---------|
| `zesolver_resources_path`  | Catalog resources directory passed to the runtime.   | *(empty)* |
| `zesolver_gpu_policy`      | `auto` / `disabled` / `required`.                    | `auto`  |
| `zesolver_backend_policy`  | `auto` / `near_only` / `blind_only` (per-solve).     | `auto`  |
| `zesolver_timeout_s`       | Solve timeout in seconds (per-solve).                | `300`   |

Optional solve hints (also read from settings, mapped onto public `SolveHints`
fields): `zesolver_ra_deg`, `zesolver_dec_deg`, `zesolver_radius_deg`,
`zesolver_pixel_scale_arcsec`, `zesolver_fov_deg`, `zesolver_focal_length_mm`,
`zesolver_pixel_size_um`.

---

## 6. Network policy and anti-import rules

- **Network is disabled by default.** The adapter always builds
  `SolveOptions` with `network_policy = NetworkPolicy.DISABLED`, regardless of
  settings, so a ZeSolver solve can never perform a network query unless the
  upstream public API itself is explicitly changed.
- **Public API only.** The adapter references exactly one external module,
  `zesolver.api.v1`. It never imports private/internal ZeSolver modules
  (`zesolver.zeblindsolver`, `zesolver.zewcs290`, `zesolver.gpu_support`,
  `zesolver.profiles`, …), never searches a sibling ZeSolver checkout, and never
  mutates `sys.path` (no `sys.path.insert`/`sys.path.append`, no `../ZeSolver`).
- **Optionality at the packaging level.** `pyproject.toml` declares **no**
  ZeSolver dependency (base or optional). ZeSolver is discovered at runtime, not
  at install time.

---

## 7. Runtime guards

- Importing `zemosaic.solver_port` or `zemosaic.zesolver_adapter` never imports
  `zesolver`.
- Importing `zemosaic.zemosaic_worker` never puts `zesolver` in `sys.modules` —
  the adapter is imported lazily and the public API is only resolved inside
  `discover_zesolver()` / `_import_api()`.
- Any ZeSolver failure (import error, incompatible API, unhealthy probe, solve
  exception, invalid WCS) is converted into a `SolverOutcome`
  (`UNAVAILABLE`/`FAILED`/`SKIPPED`) — it never propagates and never breaks the
  legacy solving path.
