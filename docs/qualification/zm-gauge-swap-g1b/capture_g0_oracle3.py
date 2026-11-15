"""Capture the G0-ORACLE-3 golden A POSTERIORI (rework L6).

G0-ORACLE-3 (``plan_hash`` + decision tuples + ``aggregate_bytes`` + reason
codes of a deterministic fixture) should have been captured BEFORE G1b. It was
not. This script captures it now from the fixtures that G1b already exercises,
and the G1b test suite asserts that the captured decisions still recompute
exactly. It is a capture tool, not product code.

Run (official interpreter, PYTHONPATH removed, from the repo root):

    env -u PYTHONPATH QT_QPA_PLATFORM=offscreen \
      /home/tristan/.openclaw/workspace/projects/zemosaic/.venv/bin/python \
      docs/qualification/zm-gauge-swap-g1b/capture_g0_oracle3.py

The output file is written next to this script. Regenerate it ONLY when the
fixtures change on purpose; the test fails otherwise, which is the point.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SRC = REPO / "src"
TESTS = REPO / "tests"
for path in (str(SRC), str(TESTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

from test_zegrid_gauge_admission import (  # noqa: E402
    OFF_FIXTURES,
    _absent_credit,
    _groups,
    _plan,
)
from zemosaic.core.zegrid import gauge_admission as adm  # noqa: E402


CAPTURE = {
    "schema": "g0-oracle3-capture-a-posteriori-v1",
    "oracle": "G0-ORACLE-3 (decision OFF)",
    "kind": "a_posteriori",
    "note": (
        "G0-ORACLE-3 was NOT captured before G1b as §12 required. This is a "
        "post-hoc capture from the deterministic G1b fixtures; it is a "
        "reproducible golden, not a pre-registered oracle."
    ),
    "contract": "docs/refactor/ZEGRID_GAUGE_SWAP_G0_CONTRACT_20261115.md",
    "baseline_head": "e25fb5133768adf4eb26950c63f86b970909b370",
    "policy": "absent (== ram_only)",
    "fields": [
        "aggregate_raw_bytes",
        "aggregate_margined_bytes",
        "admitted",
        "reason_code",
        "peak_group",
        "legacy_plan_hash",
        "decision_hash",
        "execution_digest",
        "groups",
    ],
}


def capture() -> dict:
    cases = []
    for peaks, overlaps, b_ram in OFF_FIXTURES:
        envs = _plan(peaks, overlaps=overlaps)
        decision = adm.evaluate_groups(_groups(envs), b_ram, _absent_credit())
        cases.append(
            {
                "peaks": {key: int(value) for key, value in peaks.items()},
                "overlaps": [list(edge) for edge in overlaps],
                "b_ram": int(b_ram),
                "decision": {
                    "aggregate_raw_bytes": decision.aggregate_raw_bytes,
                    "aggregate_margined_bytes": decision.aggregate_margined_bytes,
                    "admitted": decision.admitted,
                    "reason_code": decision.reason_code,
                    "peak_group": list(decision.peak_group),
                    "legacy_plan_hash": decision.legacy_plan_hash,
                    "decision_hash": decision.decision_hash,
                    "execution_digest": decision.execution_digest,
                    "groups": [group.to_dict() for group in decision.groups],
                },
            }
        )
    return {**CAPTURE, "cases": cases}


def main() -> int:
    payload = capture()
    target = HERE / "g0_oracle3_capture_a_posteriori.json"
    target.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    total = len(payload["cases"])
    admitted = sum(1 for case in payload["cases"] if case["decision"]["admitted"])
    print(f"wrote {target} cases={total} admitted={admitted} refused={total - admitted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
