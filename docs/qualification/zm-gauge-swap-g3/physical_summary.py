#!/usr/bin/env python3
"""G3 — assemble the PHYSICAL qualification evidence into one JSON record (v2).

Rework-2 additions:
* a real EXTERNAL + bounded_swap run (credit applied, versioned digest, stack
  external at B_ram, science vs the external witness);
* a real Stop with the credit OPEN (open before the request, closed after, no
  persisted credit) and its resume;
* ``Q9.scope_oom`` is DERIVED from the scope meta (exit code / signal / stderr
  OOM markers), never a literal;
* ``Q4`` is NOT_EXERCISED (the bounded run is shorter than the 10-min window);
* ``Q1/Q2`` are explicitly INDICATIVE (n=1, non-randomised order, no cache
  control) and never presented as a gain.

Run (official interpreter, PYTHONPATH removed):

    env -u PYTHONPATH QT_QPA_PLATFORM=offscreen \
      /home/tristan/.openclaw/workspace/projects/zemosaic/.venv/bin/python \
      docs/qualification/zm-gauge-swap-g3/physical_summary.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path("/var/tmp/openclaw-tristan/zm-gauge-g3")
EVID = ROOT / "evidence"
RUNS = ROOT / "runs"
MIB = 2 ** 20
GIB = 2 ** 30


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def window_peak(record: dict, start: float, end: float) -> dict:
    rows = [r for r in record["sampler"]["series"] if start <= r["t"] <= end]
    if not rows:
        return {"samples": 0}
    return {
        "samples": len(rows),
        "peak_memory_current": max(r["cgroup_memory_current"] for r in rows),
        "peak_swap_current": max(r["cgroup_swap_current"] for r in rows),
        "peak_rss": max(r["rss_bytes"] for r in rows),
        "swap_first": rows[0]["cgroup_swap_current"],
        "swap_last": rows[-1]["cgroup_swap_current"],
    }


def gauge_artifact(path: Path) -> dict:
    doc = load(path)
    return {
        "reference_index": doc["reference_index"],
        "reference_frame_id": doc["reference_frame_id"],
        "gauge_sha256": doc["gauge_sha256"],
        "frame_count": doc["frame_count"],
        "exclusions": len(doc["exclusions"]),
        "weights": [float(v) for v in doc["weights"]],
        "norm_active": [bool(v) for v in doc["norm_active"]],
        "coefficients_sha256": hashlib.sha256(
            json.dumps(doc["coefficients"], sort_keys=True).encode("ascii")
        ).hexdigest(),
    }


def manifest(path: Path) -> dict:
    return load(path)


def science(doc: dict) -> dict:
    return ((doc.get("science_output_contract") or {}).get("raw") or {})


def credit_events(record: dict) -> dict:
    opened = [e for e in record.get("progress_events", []) if "gauge credit OPEN" in e["msg"]]
    closed = [e for e in record.get("progress_events", []) if "gauge credit CLOSED" in e["msg"]]
    return {
        "open_t_s": None if not opened else opened[0]["t"],
        "closed_t_s": None if not closed else closed[0]["t"],
    }


def main() -> int:
    # --- canvas stream/RAM case (r0, unchanged evidence) -------------------
    w = load(EVID / "witness_canvas2.json")
    f = load(EVID / "fallback_canvas2.json")
    wm = manifest(RUNS / "witness-canvas2-out/zegrid_manifest.json")
    fm = manifest(RUNS / "fallback-canvas2-out/zegrid_manifest.json")
    w_t, f_t = wm.get("timings") or {}, fm.get("timings") or {}
    w_gauge, f_gauge = float(w_t.get("gauge") or 0), float(f_t.get("gauge") or 0)
    w_total, f_total = float(w["exit"]["duration_s"]), float(f["exit"]["duration_s"])

    series = f["sampler"]["series"]
    f_start = series[-1]["t"] - f_total
    gauge_start = f_start + float(f_t.get("setup", 0)) + float(f_t.get("layout", 0))
    gauge_end = gauge_start + f_gauge
    f_gauge_window = window_peak(f, gauge_start, gauge_end)

    canvas_block = fm.get("gauge_swap") or {}
    admission = canvas_block.get("admission") or {}
    p_gauge_raw = int(admission.get("p_gauge_raw_bytes") or 0)
    p_gauge_margined = int(admission.get("p_gauge_margined_bytes") or 0)
    g_np = 356_515_840
    paginable = max(0, p_gauge_raw - g_np)
    peak_swap = int(f["sampler"]["peak_cgroup_swap_current"] or 0)
    pswpout_bytes = int(f["vmstat_delta"].get("pswpout", 0)) * 4096

    # --- external_spool case (rework-2, M1) -------------------------------
    ew = load(EVID / "witness_ext2.json")
    ef = load(EVID / "fallback_ext3.json")
    ewm = manifest(RUNS / "witness-ext2-out/zegrid_manifest.json")
    efm = manifest(RUNS / "fallback-ext3-out/zegrid_manifest.json")
    ef_meta = load(EVID / "fallback-ext3.meta.json")
    eblock = efm.get("gauge_swap") or {}
    eadmission = eblock.get("admission") or {}
    external = {
        "witness_record": str(EVID / "witness_ext2.json"),
        "fallback_record": str(EVID / "fallback_ext3.json"),
        "scope": {
            "memory_max": ef["sampler"]["cgroup_memory_max"],
            "memory_swap_max": ef["sampler"]["cgroup_swap_max"],
            "cgroup": ef["sampler"]["cgroup_path"],
        },
        "witness": {
            "duration_s": float(ew["exit"]["duration_s"]),
            "class": (ewm["r55_scheduler"]["mapping"] or {}).get("working_set_class"),
            "cell_task_mode": (ewm["r55_scheduler"]["mapping"] or {}).get("cell_task_mode"),
            "science_sha256": science(ewm).get("sha256"),
            "coverage_pixels": ewm.get("coverage_pixels"),
            "hole_pixels": ewm.get("hole_pixels"),
            "credit": None,
        },
        "fallback": {
            "duration_s": float(ef["exit"]["duration_s"]),
            "class": (efm["r55_scheduler"]["mapping"] or {}).get("working_set_class"),
            "cell_task_mode": (efm["r55_scheduler"]["mapping"] or {}).get("cell_task_mode"),
            "credit_applied": eblock.get("credit_applied"),
            "gauge_workers_effective": eblock.get("gauge_workers_effective"),
            "execution_digest": eblock.get("execution_digest"),
            "admission": eadmission,
            "events": eblock.get("events"),
            "stack_envelope_bytes": efm["r55_scheduler"]["final_envelope"]["aggregate_bytes"],
            "stack_budget_bytes": efm["r55_scheduler"]["final_envelope"]["budget_bytes"],
            "stack_ceiling_bytes": eadmission.get("b_ram_bytes"),
            "stack_credit_applied": False,
            "stack_credit_note": (
                "group-scoped evaluator: only the ('gauge',) group can be "
                "widened (L8 checks); the external stack ceiling stays B_ram"
            ),
            "credit_closed_before_stack": any(
                e.get("event") == "gauge_credit_closed"
                for e in (eblock.get("events") or [])
            ),
            "stack_within_b_ram": (
                int(efm["r55_scheduler"]["final_envelope"]["aggregate_bytes"])
                <= int(eadmission.get("b_ram_bytes") or 0)
            ),
            "science_sha256": science(efm).get("sha256"),
            "coverage_pixels": efm.get("coverage_pixels"),
            "hole_pixels": efm.get("hole_pixels"),
            "peak_memory_current": ef["sampler"]["peak_cgroup_memory_current"],
            "peak_swap_current": ef["sampler"]["peak_cgroup_swap_current"],
        },
        "science_bit_identical": science(ewm).get("sha256") == science(efm).get("sha256"),
        "digest_present": bool(eblock.get("execution_digest")),
        "scope_meta": ef_meta,
    }

    # --- Stop with the credit OPEN + resume (rework-2, M2) ----------------
    q6s = load(EVID / "q6c_stop.json")
    q6r = load(EVID / "q6c_resume.json")
    q6_state = load(EVID / "q6c_stop_state_observed.json")
    q6rm = manifest(RUNS / "q6c-resume-out/zegrid_manifest.json")
    commits = sorted((RUNS / "q6c-store/commits").glob("*.json"))
    q6_events = credit_events(q6s)
    req = q6s["stop"].get("request_monotonic_s")
    open_before = (
        q6_events["open_t_s"] is not None and req is not None
        and q6_events["open_t_s"] < req
    )
    closed_after = (
        q6_events["closed_t_s"] is not None and req is not None
        and q6_events["closed_t_s"] > req
    )
    # The closed event is not in the resume manifest (restored gauge): read it
    # from the RUN LOG line, which is the durable record of the stopped run.
    stop_log = (RUNS / "q6c-stop-out/zegrid_run.log").read_text(
        encoding="utf-8", errors="replace"
    )
    closed_event = next(
        (
            json.loads(line.split("ZeGrid gauge credit: ", 1)[1])
            for line in stop_log.splitlines()
            if '"event": "gauge_credit_closed"' in line
        ),
        None,
    )
    closed_swap_null = bool(
        closed_event is not None
        and closed_event.get("swap_used_bytes", "missing") is None
        and closed_event.get("was_open") is True
    )
    q6 = {
        "stop_record": str(EVID / "q6c_stop.json"),
        "resume_record": str(EVID / "q6c_resume.json"),
        "scope": {
            "memory_max": q6s["sampler"]["cgroup_memory_max"],
            "memory_swap_max": q6s["sampler"]["cgroup_swap_max"],
        },
        "credit_open_t_s": q6_events["open_t_s"],
        "stop_request_t_s": req,
        "credit_closed_t_s": q6_events["closed_t_s"],
        "open_before_request": open_before,
        "closed_after_request": closed_after,
        "ack_latency_s": q6s["stop"].get("ack_latency_s"),
        "process_exit_s": q6s["exit"]["duration_s"],
        "ack_state": q6s["stop_snapshot"].get("ack_state"),
        "swap_used_bytes_null_in_closed_event": closed_swap_null,
        "commits_at_stop": q6_state.get("commits"),
        "state_finalized_at_stop": q6_state.get("finalized"),
        "credit_strings_in_store_at_stop": q6_state.get("credit_strings_in_store"),
        "commits_after_resume": len(commits),
        "resume_complete_cells": q6rm.get("complete_cells"),
        "resume_incomplete_cells": q6rm.get("incomplete_cells"),
        "resume_gauge_applicable": (q6rm.get("gauge_swap") or {}).get("applicable"),
        "resume_credit_applied": (q6rm.get("gauge_swap") or {}).get("credit_applied"),
        "resume_science_sha256": science(q6rm).get("sha256"),
        "witness_science_sha256": science(ewm).get("sha256"),
        "finalized_after_resume": load(RUNS / "q6c-store/state.json").get("finalized"),
        "note_first_resume_attempt": (
            "a FIRST resume with the REAL A was refused (ResumeIncompatible: "
            "stored tile_px=128 vs recomputed 112) because the injected A "
            "changes the admission tile. The accepted resume replays the SAME "
            "A=550000000; this is the identity conservatism, not a defect."
        ),
    }

    external_meta = ef_meta
    scope_oom = bool(external_meta.get("scope_oom"))
    q9_verdict = "FAIL" if scope_oom else "PASS_WITH_LIMIT"

    thresholds = {
        "Q1_gauge_time_ratio": {
            "threshold": "<= 4.0 x witness",
            "observed": round(f_gauge / w_gauge, 4) if w_gauge else None,
            "witness_gauge_s": w_gauge,
            "fallback_gauge_s": f_gauge,
            "quality": "INDICATIVE (n=1, order not randomised, page cache not controlled)",
            "verdict": "PASS_INDICATIVE" if w_gauge and f_gauge <= 4.0 * w_gauge else "FAIL",
            "note": "the 0.73x ratio is NOT a gain claim: the fallback ran second on a warm cache",
        },
        "Q2_total_time_ratio": {
            "threshold": "<= 3.0 x witness",
            "observed": round(f_total / w_total, 4) if w_total else None,
            "witness_total_s": w_total,
            "fallback_total_s": f_total,
            "quality": "INDICATIVE (n=1, order not randomised)",
            "verdict": "PASS_INDICATIVE" if w_total and f_total <= 3.0 * w_total else "FAIL",
        },
        "Q3_swap_traffic": {
            "threshold": "<= 4 x modeled paginable (P_gauge_raw - G_np_new)",
            "modeled_paginable_bytes": paginable,
            "modeled_4x_bytes": 4 * paginable,
            "observed_pswpout_bytes_whole_run": pswpout_bytes,
            "observed_peak_scope_swap_bytes": peak_swap,
            "note": "whole-run pswpout is an UPPER bound for the gauge phase",
            "verdict": "PASS" if pswpout_bytes <= 4 * paginable else "FAIL",
        },
        "Q4_thrash_signature": {
            "threshold": "no interval > 10 min with zero frame progress AND swap-out > 50 MiB/s",
            "run_duration_s": f_total,
            "swap_out_average_Bps": round(pswpout_bytes / f_total, 1) if f_total else None,
            "verdict": "NOT_EXERCISED",
            "note": (
                "the bounded run lasts < 10 min, so the criterion cannot trigger; "
                "a real thrash signature belongs to a long 20k run"
            ),
        },
        "Q5_peak_memory_gauge": {
            "threshold": "RSS tree + delta swap.current <= P_gauge_raw + 15% + 256 MiB",
            "p_gauge_raw_bytes": p_gauge_raw,
            "ceiling_bytes": int(p_gauge_margined + 256 * MIB),
            "observed_gauge_window": f_gauge_window,
            "observed_bytes": (
                int(f_gauge_window["peak_memory_current"])
                + max(0, int(f_gauge_window["peak_swap_current"])
                      - int(f_gauge_window["swap_first"]))
                if f_gauge_window.get("samples") else None
            ),
            "note": "gauge window delimited from the manifest phase durations",
            "verdict": (
                "PASS"
                if f_gauge_window.get("samples")
                and int(f_gauge_window["peak_memory_current"])
                + max(0, int(f_gauge_window["peak_swap_current"])
                      - int(f_gauge_window["swap_first"]))
                <= int(p_gauge_margined + 256 * MIB)
                else "FAIL" if f_gauge_window.get("samples") else "INCONCLUSIVE"
            ),
        },
        "Q6_stop": {
            "threshold": "ack <= 30 s; process exit <= 120 s; no lost commit",
            "ack_latency_s": q6["ack_latency_s"],
            "process_exit_s": q6["process_exit_s"],
            "ack_state": q6["ack_state"],
            "credit_open_before_request": open_before,
            "credit_closed_after_request": closed_after,
            "swap_used_bytes_null": closed_swap_null,
            "commits_at_stop": q6["commits_at_stop"],
            "state_finalized_at_stop": q6["state_finalized_at_stop"],
            "verdict": (
                "PASS"
                if (q6["ack_latency_s"] or 1e9) <= 30.0
                and (q6["process_exit_s"] or 1e9) <= 120.0
                and open_before and closed_after and closed_swap_null
                and q6["state_finalized_at_stop"] is False
                and q6["commits_at_stop"] == 0
                else "FAIL"
            ),
        },
        "Q7_resume": {
            "threshold": "zero loss, zero duplicate, only pending cells executed",
            "expected_cells": 4,
            "commits_after_resume": q6["commits_after_resume"],
            "complete_cells": q6["resume_complete_cells"],
            "incomplete_cells": q6["resume_incomplete_cells"],
            "finalized": q6["finalized_after_resume"],
            "committed_cell_reuse": "NOT_EXERCISED (all cells were pending at the stop)",
            "verdict": (
                "PASS_PARTIAL"
                if q6["finalized_after_resume"] is True
                and q6["commits_after_resume"] == 4
                and not q6["resume_incomplete_cells"]
                else "FAIL"
            ),
            "note": "zero-loss/zero-duplicate proven; reusing an already-committed cell is not",
        },
        "Q8_science": {
            "threshold": "G0-ORACLE-4 bit-identical at identical workers",
            "canvas": {
                "witness_science_sha256": science(wm).get("sha256"),
                "fallback_science_sha256": science(fm).get("sha256"),
                "coverage_pixels": [wm.get("coverage_pixels"), fm.get("coverage_pixels")],
                "hole_pixels": [wm.get("hole_pixels"), fm.get("hole_pixels")],
            },
            "external": {
                "witness_science_sha256": external["witness"]["science_sha256"],
                "fallback_science_sha256": external["fallback"]["science_sha256"],
                "coverage_pixels": [
                    external["witness"]["coverage_pixels"],
                    external["fallback"]["coverage_pixels"],
                ],
                "hole_pixels": [
                    external["witness"]["hole_pixels"],
                    external["fallback"]["hole_pixels"],
                ],
                "digest_present": external["digest_present"],
            },
            "workers": 1,
            "verdict": (
                "PASS"
                if science(wm).get("sha256") == science(fm).get("sha256")
                and wm.get("coverage_pixels") == fm.get("coverage_pixels")
                and wm.get("hole_pixels") == fm.get("hole_pixels")
                and external["science_bit_identical"]
                and external["digest_present"]
                else "FAIL"
            ),
        },
        "Q9_host_safety": {
            "threshold": "zero host OOM; OOM inside the scope invalidates the run",
            "scope_meta": external_meta,
            "scope_oom_derived": scope_oom,
            "canvas_case_meta": "NOT_CAPTURED (the r0 harness predates meta capture); the canvas run exited with exception_type=null and no OOM marker in its scope log",
            "derivation": "scope_oom = exit_code in (137) or signal == 9 or any OOM marker in the scope stderr; FAIL if true, else PASS_WITH_LIMIT",
            "scope_exit_code": external_meta.get("exit_code"),
            "scope_signal": external_meta.get("signal"),
            "oom_markers": external_meta.get("oom_markers"),
            "host_kernel_journal_readable": False,
            "host_dmesg_readable": False,
            "host_oom_evidence": (
                "derived from the scope meta: exit code/signal and stderr OOM "
                "markers; kernel dmesg/journal are unreadable for this user, so "
                "HOST OOM absence is still not independently proven"
            ),
            "verdict": q9_verdict,
        },
        "Q10_telemetry": {
            "threshold": "all required fields present; otherwise INCONCLUSIVE",
            "present": [
                "memory.max", "memory.swap.max", "memory.current",
                "memory.swap.current", "/proc/swaps per device (manifest)",
                "/proc/vmstat pswpin/pswpout", "RSS", "phase timings",
                "gauge_credit_open/closed phase timestamps",
                "scope exit code / signal / stderr",
            ],
            "missing": ["/proc/pressure/memory", "/proc/pressure/io", "kernel dmesg"],
            "verdict": "INCONCLUSIVE (PSI memory/IO not sampled; dmesg restricted)",
        },
    }

    record = {
        "schema": "g3-physical-qualification-v2",
        "mission": "ZM-GAUGE-SWAP-G3-20261115",
        "base_head": "e25fb5133768adf4eb26950c63f86b970909b370",
        "corpus": {
            "kind": "SYNTHETIC deterministic 8 frames x 2048 px / 8 x 512 px",
            "note": "NOT the user's real NGC7023; no user data was read",
            "frames": [
                {"name": p.name, "sha256": sha256(p)}
                for p in sorted((ROOT / "corpus/canvas8_2048/in").glob("*.fits"))
            ],
        },
        "canvas_case": {
            "witness_record": str(EVID / "witness_canvas2.json"),
            "fallback_record": str(EVID / "fallback_canvas2.json"),
            "witness": {
                "policy": w["policy"],
                "inject_a_raw": w["inject_a_raw"],
                "duration_s": w_total,
                "timings": w_t,
                "scope": {
                    "memory_max": w["sampler"]["cgroup_memory_max"],
                    "memory_swap_max": w["sampler"]["cgroup_swap_max"],
                },
                "peak_memory_current": w["sampler"]["peak_cgroup_memory_current"],
                "peak_swap_current": w["sampler"]["peak_cgroup_swap_current"],
                "science": science(wm),
                "coverage_pixels": wm.get("coverage_pixels"),
                "hole_pixels": wm.get("hole_pixels"),
                "gauge_artifact": gauge_artifact(RUNS / "witness-canvas2-store/gauge.json"),
            },
            "fallback": {
                "policy": f["policy"],
                "inject_a_raw": f["inject_a_raw"],
                "duration_s": f_total,
                "timings": f_t,
                "scope": {
                    "memory_max": f["sampler"]["cgroup_memory_max"],
                    "memory_swap_max": f["sampler"]["cgroup_swap_max"],
                },
                "peak_memory_current": f["sampler"]["peak_cgroup_memory_current"],
                "peak_swap_current": peak_swap,
                "pswpout_bytes": pswpout_bytes,
                "pswpin_pages": f["vmstat_delta"].get("pswpin"),
                "credit_admission": admission,
                "credit_applied": canvas_block.get("credit_applied"),
                "gauge_workers_effective": canvas_block.get("gauge_workers_effective"),
                "credit_events": canvas_block.get("events"),
                "science": science(fm),
                "coverage_pixels": fm.get("coverage_pixels"),
                "hole_pixels": fm.get("hole_pixels"),
                "gauge_artifact": gauge_artifact(RUNS / "fallback-canvas2-store/gauge.json"),
            },
        },
        "external_case": external,
        "q6_credit": q6,
        "thresholds": thresholds,
        "limits": [
            "The A observation (MemAvailable) is INJECTED by the harness (monkeypatch), never a product flag; that proves the LOGIC, not a real host RAM shortage.",
            "Swap enforcement is REAL in the canvas case: transient systemd scope, peak memory.swap.current 623 Mo, pswpout 620 Mo. The external case did not need swap (peak 599 Mo < 1 GiB cap) but is isolated and OOM-free.",
            "Disk-backed swap is NOT proven: zram (priority 100) absorbs first; proving disk swap would require changing swap priorities (forbidden).",
            "overcommit mode 2 is unit-tested by injection only (sysctl change forbidden).",
            "GPU is not qualified: CuPy is absent from the official interpreter.",
            "The corpus is synthetic, not the user's NGC7023 dataset.",
            "Q10 is INCONCLUSIVE (no PSI, no kernel dmesg).",
            "Q4 is NOT_EXERCISED: the bounded run (< 10 min) cannot trigger a 10-min thrash window.",
            "Q1/Q2 are INDICATIVE (n=1, witness ran first, page cache not controlled); no gain is claimed.",
            "The EXTERNAL physical evidence (M1/M2) was produced AFTER rework-1; the earlier canvas/Stop runs (r0) predate it and had no external credit / no credit-open Stop. Rework-2 closes both gaps.",
            "The Stop under credit was initially NOT covered (r0 Q6 had no open credit); it is covered by the rework-2 run.",
            "Q7 committed-cell reuse is NOT_EXERCISED (all cells were pending at the Stop).",
            "External resume requires replaying the SAME injected A: with the real A the frozen tile differs (128 vs 112) and the engine correctly refuses the resume.",
        ],
        "artifacts_root": str(ROOT),
    }

    target = HERE / "physical_qualification.json"
    target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {target}")
    for name, item in thresholds.items():
        print(f"{name}: {item['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
