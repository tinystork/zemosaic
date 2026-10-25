# ZM-ZEGRID-BOUNDED-PREFETCH-20261025 — measured artifacts

Owner: Coco (implementation). Acceptance: Junior after verification, then Nono review.

## What is in here

`linux-cuda-smoke-halo64-pinned.json` — output of the common instrumented runner
`tools/zegrid_r56_prefetch_campaign.py` on the ONLY GPU available to this
worktree (NVIDIA GeForce MX150, 2 GiB, Linux). It is a SHORT bounded CUDA smoke,
NOT the LP341 corpus and NOT a Windows run.

Reproduce:

```
PYTHONPATH=src python tools/zegrid_r56_prefetch_campaign.py \
  --input <12x768px synthetic corpus> --out <scratch> \
  --halo 64 --gpu --storage-profile bounded \
  --available-bytes 4294967296 --sample-period 0.1
```

`--available-bytes` PINS the host-RAM observation for BOTH arms. Without it the
planner re-observes RAM per run; on this host (1.2–1.8 GiB available, the
OpenClaw gateway itself holds ~1.6 GiB RSS) the admitted plan drifted between
arms, which makes an A/B meaningless. The pin is recorded in the artifact as
`memory_observation_pinned_bytes`; RSS/VRAM remain real observations.

## Observed result (this artifact)

| quantity | control (cap 0) | candidate (planner) |
|---|---|---|
| PLAN prefetch | 2 | 2 |
| RUN prefetch (effective) | 0 | 2 |
| queue_max | 1 | 3 |
| host credit peak (reservation) | 7 817 082 B | 23 451 246 B |
| disk peak (reservation) | 93 808 056 B | 179 224 404 B |
| device peak (reservation) | 974 984 704 B | 974 984 704 B |
| sampled process-tree RSS peak | 5 514 735 616 B | 5 312 466 944 B |
| sampled device free VRAM min | 1 522 139 136 B | 1 522 139 136 B |
| producer work | 9.912 s | 10.027 s |
| consumer work | 19.103 s | 18.404 s |
| observed overlap intersection | 8.857 s | 8.869 s |

Gates: `scientific_outputs_identical=true`, `frames_identical=true`,
`complete_cells_identical=true`, `real_overlap_proved=true`,
`observed_memory_available=true`, `vram_observed=true`.

## How to read it (and how NOT to)

* The host/disk/device numbers are RESERVATIONS from the product's credit model
  and pipeline ledger. They are not RSS, not VRAM and not a high-water mark.
* The RSS number is a SUM over the sampled process tree at a 0.1 s period. Shared
  pages are counted once per process, so it is an aggregate UPPER bound that can
  legitimately exceed physical RAM. It is not a process RSS and not an OS peak.
* VRAM free/total is DEVICE-WIDE; `vram_pool_used_peak_bytes` is this process'
  CuPy default pool only. Boundary snapshots in the product remain instantaneous.
* The candidate's ~200 MiB lower sampled RSS is within run-to-run noise on this
  host and is NOT reported as a memory win. No performance claim is made.
* Halo here is 64 px (chosen so the existing halo-overhead floor admits the
  geometry on a 1.2 GiB box). The LP341 halo128 invariant is unchanged in code;
  its runtime proof is NOT_RUN on this hardware.

## Outstanding gates (not satisfied here)

* LP341 halo128 sky_mean before/after on a capable Windows node: **NOT_RUN**.
  No such node/tool is available in this worktree; this host cannot even build
  the halo128 LP geometry (PLAN budget ~959 MiB).
* LP341 planning at prefetch>=1 is proved at PLAN level by
  `tests/test_zegrid_r56_bounded_prefetch_credit.py` using the exact published
  Windows numbers (341 frames, patch 1625x1415, cache 10 193 216 671 B,
  B_plan 14 687 984 025 B).
